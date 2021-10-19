#include "lazy_tensor_core/csrc/tensor.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <set>
#include <stdexcept>
#include <unordered_set>

#include "lazy_tensor_core/csrc/debug_util.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ir_dump_util.h"
#include "lazy_tensor_core/csrc/layout_manager.h"
#include "lazy_tensor_core/csrc/lazy_graph_executor.h"
#include "lazy_tensor_core/csrc/ops/arithmetic_ir_ops.h"
#include "lazy_tensor_core/csrc/ops/cast.h"
#include "lazy_tensor_core/csrc/ops/device_data.h"
#include "lazy_tensor_core/csrc/ops/expand.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensor_core/csrc/ops/ops.h"
#include "lazy_tensor_core/csrc/ops/view.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/torch_util.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_computation_client.h"
#include "lazy_tensors/computation_client/cache.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/metrics.h"
#include "lazy_tensors/computation_client/sys_util.h"
#include "lazy_tensors/computation_client/thread_pool.h"
#include "lazy_tensors/literal_util.h"
#include "lazy_tensors/shape_util.h"
#include "lazy_tensors/str_join.h"
#include "torch/csrc/autograd/variable.h"

namespace torch_lazy_tensors {
namespace {

// Routing values to device data maximizes the changes for compilation cache
// hits, but it can prevent the compiler to perform optimizations. So tensor
// values which are within a given set, are routed to constant scalars if this
// API returns true.
bool IsSpecialScalar(const at::Scalar& value) {
  static bool no_scalars =
      lazy_tensors::sys_util::GetEnvBool("NO_SPECIAL_SCALARS", false);
  if (!no_scalars && (value.isIntegral() || value.isFloatingPoint())) {
    double scalar_value = value.toDouble();
    return scalar_value == 0.0 || std::fabs(scalar_value) == 1.0;
  }
  return false;
}

}  // namespace

LazyTensor::Data::~Data() { LazyGraphExecutor::Get()->UnregisterTensor(this); }

LazyTensor LazyTensor::Create(const at::Tensor& tensor, const Device& device) {
  LTC_CHECK_NE(tensor.device().type(), at::kLazy);
  LazyTensor xtensor(tensor, device);
  LazyGraphExecutor::Get()->RegisterTensor(xtensor.data_ptr());
  return xtensor;
}

LazyTensor LazyTensor::Create(
    lazy_tensors::ComputationClient::DataPtr handle,
    c10::optional<at::ScalarType> logical_element_type) {
  LazyTensor xtensor(std::move(handle), logical_element_type);
  LazyGraphExecutor::Get()->RegisterTensor(xtensor.data_ptr());
  return xtensor;
}

LazyTensor LazyTensor::Create(
    torch::lazy::Value ir_value, const Device& device,
    c10::optional<at::ScalarType> logical_element_type) {
  LazyTensor xtensor(std::move(ir_value), device, logical_element_type);
  LazyGraphExecutor::Get()->RegisterTensor(xtensor.data_ptr());
  return xtensor;
}

LazyTensor LazyTensor::Create(
    std::shared_ptr<View> view, const Device& device,
    c10::optional<at::ScalarType> logical_element_type) {
  LazyTensor xtensor(std::move(view), device, logical_element_type);
  LazyGraphExecutor::Get()->RegisterTensor(xtensor.data_ptr());
  return xtensor;
}

LazyTensor LazyTensor::Create(std::shared_ptr<Data> data) {
  return LazyTensor(std::move(data));
}

LazyTensor::LazyTensor(const at::Tensor& tensor, const Device& device)
    : data_(std::make_shared<Data>(tensor, device)) {}

LazyTensor::LazyTensor(lazy_tensors::ComputationClient::DataPtr handle,
                       c10::optional<at::ScalarType> logical_element_type)
    : data_(std::make_shared<Data>(handle, Device(handle->device()),
                                   logical_element_type)) {}

LazyTensor::LazyTensor(torch::lazy::Value ir_value, const Device& device,
                       c10::optional<at::ScalarType> logical_element_type)
    : data_(std::make_shared<Data>(std::move(ir_value), device,
                                   logical_element_type)) {
  TryLimitGraphSize();
}

LazyTensor::LazyTensor(std::shared_ptr<View> view, const Device& device,
                       c10::optional<at::ScalarType> logical_element_type)
    : data_(std::make_shared<Data>(std::move(view), device,
                                   logical_element_type)) {}

LazyTensor::LazyTensor(std::shared_ptr<Data> data) : data_(std::move(data)) {}

LazyTensor::Data* LazyTensor::data() const {
  LTC_CHECK(data_ != nullptr) << "Trying to access a null cursor";
  return data_.get();
}

lazy_tensors::int64 LazyTensor::size(lazy_tensors::int64 dim) const {
  auto tensor_shape = shape();
  int rank = tensor_shape.get().rank();
  int dim_index = Helpers::GetCanonicalDimensionIndex(dim, rank);
  return tensor_shape.get().dimensions(dim_index);
}

at::ScalarType LazyTensor::dtype() const {
  return data()->logical_element_type
             ? *data()->logical_element_type
             : TensorTypeFromLtcType(shape().get().element_type());
}

c10::optional<at::ScalarType> LazyTensor::dtype_optional() const {
  return data()->logical_element_type;
}

lazy_tensors::util::MaybeRef<lazy_tensors::Shape> LazyTensor::shape() const {
  if (data()->view != nullptr) {
    return data()->view->shape();
  }
  if (data()->handle != nullptr) {
    return lazy_tensors::Shape(data()->handle->shape());
  }
  if (data()->ir_value) {
    // TODO(whc) remove shape from LazyTensor API too!
    return ir::GetShapeFromTsValue(data()->ir_value);
  }
  LTC_CHECK(data()->tensor_data);
  const Device& device = GetDevice();
  return lazy_tensors::ShapeUtil::MakeShape(
      MakeLtcPrimitiveType(data()->tensor_data->type().scalarType(), &device),
      Helpers::I64List(data()->tensor_data->sizes()));
}

lazy_tensors::Shape LazyTensor::shape_with_layout() const {
  auto tensor_shape = shape();
  return MakeArrayShapeFromDimensions(
      tensor_shape.get().dimensions(), tensor_shape.get().dynamic_dimensions(),
      tensor_shape.get().element_type(), GetDevice().hw_type);
}

const Device& LazyTensor::GetDevice() const { return data()->device; }

lazy_tensors::int64 LazyTensor::GetUniqueId() const {
  return data()->unique_id;
}

std::ptrdiff_t LazyTensor::GetViewAliasId() const {
  return data()->view != nullptr
             ? reinterpret_cast<std::ptrdiff_t>(data()->view->alias().get())
             : 0;
}

lazy_tensors::ComputationClient::DataPtr LazyTensor::GetDataHandle() {
  // Data can coexist with a view, but we need to check that the view did
  // not receive any updates before calling the current IR valid.
  bool up_to_date = true;
  torch::lazy::Value ir_value;
  if (data()->view != nullptr) {
    View::IrNode ir_value_updated = GetViewUpdate(data()->view);
    up_to_date = !ir_value_updated.updated;
    ir_value = std::move(ir_value_updated.ir_value);
  }
  if (up_to_date) {
    lazy_tensors::ComputationClient::DataPtr handle = CurrentDataHandle();
    if (handle != nullptr) {
      LTC_CHECK(handle->HasValue())
          << "Trying to access data while an async operation is in flight: "
          << lazy_tensors::Shape(handle->shape());
      return handle;
    }
  }
  if (ir_value) {
    // The view gave us an updated IR value. We usually do not have a valid IR
    // value field together with a view, but to allow code reuse in
    // ApplyPendingGraph() we temporarily set it here. The following call to
    // ApplyPendingGraph() will clear it.
    AssignIrValue(std::move(ir_value));
  }
  if (data()->ir_value) {
    ApplyPendingGraph();
  } else {
    LTC_CHECK(data()->tensor_data);
    data()->handle = TensorToDataHandle(*data()->tensor_data, GetDevice());
  }
  return data()->handle;
}

lazy_tensors::ComputationClient::DataPtr LazyTensor::CurrentDataHandle() const {
  return data()->handle;
}

std::string LazyTensor::DumpBackendComputation(
    const std::vector<LazyTensor>& tensors) {
  std::vector<torch::lazy::Value> ir_values;
  for (auto& tensor : tensors) {
    torch::lazy::Value ir_value = tensor.CurrentIrValue();
    if (ir_value) {
      ir_values.push_back(std::move(ir_value));
    }
  }
  return !ir_values.empty()
             ? ir::DumpUtil::ToBackend(ir_values, GetCurrentDevice())
             : std::string();
}

void LazyTensor::SetDataHandle(
    lazy_tensors::ComputationClient::DataPtr handle) {
  SetDataHandle(std::move(handle), /*sync=*/true);
}

void LazyTensor::SetDataHandle(lazy_tensors::ComputationClient::DataPtr handle,
                               bool sync) {
  data()->handle = std::move(handle);
  // Assigning a device data should always clear the IR node, to allow graph
  // trimming. A view cannot be reset though, unless we are at a step-end sync.
  AssignIrValue(torch::lazy::Value());
  if (sync) {
    data()->view = nullptr;
    data()->tensor_data = c10::nullopt;
  }
}

void LazyTensor::SetIrValue(torch::lazy::Value ir_value) {
  data()->handle = nullptr;
  data()->tensor_data = c10::nullopt;
  if (data()->view != nullptr) {
    // If we have an active view, and a SetIrValue() happens, it means we are
    // within an in-place execution context, and we need to update the view's
    // alias as well.
    data()->view = UpdateView(data()->view, std::move(ir_value));
    data()->generation += 1;
  } else {
    AssignIrValue(std::move(ir_value));
    TryLimitGraphSize();
  }
}

void LazyTensor::SetInPlaceIrValue(torch::lazy::Value ir_value) {
  auto tensor_shape = shape();
  if (tensor_shape.get().element_type() != ir::GetShapeFromTsValue(ir_value).element_type()) {
    ir_value = torch::lazy::MakeNode<ir::ops::Cast>(ir_value,
                                           tensor_shape.get().element_type());
  }
  SetIrValue(std::move(ir_value));
}

void LazyTensor::AssignIrValue(torch::lazy::Value ir_value) const {
  data()->ir_value = std::move(ir_value);
  data()->generation += 1;
}

void LazyTensor::TryLimitGraphSize() {
  static const size_t kCheckFrequency =
      lazy_tensors::sys_util::GetEnvInt("TRIM_GRAPH_CHECK_FREQUENCY", 5000);
  static const size_t kMaxPendingGraphSize =
      lazy_tensors::sys_util::GetEnvInt("TRIM_GRAPH_SIZE", 100000);
  if (data()->ir_value &&
      LazyGraphExecutor::Get()->IncTrimCounter() % kCheckFrequency == 0) {
    size_t graph_size = ir::Util::GetGraphSize({data()->ir_value.node.get()});
    if (graph_size > kMaxPendingGraphSize) {
      LTC_COUNTER("TrimIrGraph", 1);
      ApplyPendingGraph();
    }
  }
}

torch::lazy::Value LazyTensor::GetIrValue() const {
  torch::lazy::Value ir_value = CurrentIrValue();
  if (ir_value) {
    return ir_value;
  }
  lazy_tensors::ComputationClient::DataPtr handle = CurrentDataHandle();
  if (handle != nullptr) {
    // In case of tensor node, we do not clear the data when we set the IR
    // node. This because we want further calls to GetIrValue() to fetch the
    // same IR node, and not create new ones (even though the lowering context
    // will still collapse them all into a single parameter op). So the call
    // which wants the data will still find it, w/out having to fetch it via
    // a computation client from-server call.
    AssignIrValue(CreateTensorNode(handle, /*read_only=*/false));
    return data()->ir_value;
  }
  c10::optional<at::Tensor> tensor_data = CurrentTensorData();
  LTC_CHECK(tensor_data);
  AssignIrValue(GetIrValueForTensor(*tensor_data, GetDevice()));
  return data()->ir_value;
}

torch::lazy::Value LazyTensor::CurrentIrValue() const {
  if (data()->view != nullptr) {
    return GetViewUpdate(data()->view).ir_value;
  }
  return data()->ir_value;
}

void LazyTensor::SetTensorData(at::Tensor tensor_data) {
  data()->tensor_data = std::move(tensor_data);
}

c10::optional<at::Tensor> LazyTensor::CurrentTensorData() const {
  if (data()->view != nullptr && !data()->view->IsUpToDate()) {
    return c10::nullopt;
  }
  return data()->tensor_data;
}

torch::lazy::Value LazyTensor::GetIrValueForTensor(const at::Tensor& tensor,
                                          const Device& device) const {
  lazy_tensors::ComputationClient::DataPtr data;
  bool read_only = false;
  if (tensor.dim() == 0 && tensor.numel() == 1) {
    at::Scalar value = tensor.item();
    if (IsSpecialScalar(value)) {
      return ir::ops::ScalarOp(
          std::move(value),
          MakeLtcPrimitiveType(tensor.scalar_type(), &device));
    }
    data = LazyGraphExecutor::Get()->GetDeviceData(tensor.cpu(), device);
    read_only = true;
  } else {
    LTC_TIMED("IrValueTensorToDataHandle");
    data = TensorToDataHandle(tensor, device);
  }
  return CreateTensorNode(std::move(data), read_only);
}

torch::lazy::Value LazyTensor::GetDeviceDataIrValue(const at::Scalar& value,
                                           lazy_tensors::PrimitiveType type,
                                           const Device& device) {
  lazy_tensors::ComputationClient::DataPtr data =
      LazyGraphExecutor::Get()->GetDeviceData(
          value, TensorTypeFromLtcType(type), device);
  data->SetInfo(std::make_shared<LazyGraphExecutor::DeviceDataInfo>(
      /*tensor_id=*/-1, /*read_only=*/true));
  return torch::lazy::MakeNode<ir::ops::DeviceData>(std::move(data));
}

torch::lazy::Value LazyTensor::GetIrValueForScalar(const at::Scalar& value,
                                          lazy_tensors::PrimitiveType type,
                                          const Device& device) {
  if (IsSpecialScalar(value)) {
    return ir::ops::ScalarOp(std::move(value), type);
  }
  return GetDeviceDataIrValue(value, type, device);
}

torch::lazy::Value LazyTensor::GetIrValueForScalar(const at::Scalar& value,
                                          const Device& device) {
  return GetIrValueForScalar(
      value, MakeLtcPrimitiveType(GetScalarType(value), &device), device);
}

torch::lazy::Value LazyTensor::GetIrValueForScalar(
    const at::Scalar& value, lazy_tensors::PrimitiveType type,
    lazy_tensors::Span<const lazy_tensors::int64> dimensions,
    const Device& device) {
  torch::lazy::Value ir_value = GetIrValueForScalar(value, type, device);
  if (!dimensions.empty()) {
    ir_value = torch::lazy::MakeNode<ir::ops::Expand>(
        ir_value, lazy_tensors::util::ToVector<lazy_tensors::int64>(dimensions),
        /*is_scalar_expand=*/true);
  }
  return ir_value;
}

torch::lazy::Value LazyTensor::GetIrValueForScalar(const at::Scalar& value,
                                          const lazy_tensors::Shape& shape,
                                          const Device& device) {
  return GetIrValueForScalar(value, shape.element_type(), shape.dimensions(),
                             device);
}

torch::lazy::Value LazyTensor::GetIrValueForScalar(
    const at::Scalar& value, const lazy_tensors::Shape& shape,
    c10::optional<at::ScalarType> logical_element_type, const Device& device) {
  lazy_tensors::PrimitiveType type =
      logical_element_type
          ? MakeLtcPrimitiveType(*logical_element_type, &device)
          : shape.element_type();
  return GetIrValueForScalar(value, type, shape.dimensions(), device);
}

View::IrNode LazyTensor::GetViewUpdate(
    const std::shared_ptr<View>& view) const {
  View::IrNode ir_value_updated = view->GetViewIrNode();
  if (ir_value_updated.updated) {
    data()->handle = nullptr;
    data()->tensor_data = c10::nullopt;
  }
  return ir_value_updated;
}

std::shared_ptr<View> LazyTensor::UpdateView(std::shared_ptr<View> view,
                                             torch::lazy::Value ir_value) const {
  if (ir::GetShapeFromTsValue(ir_value).dimensions() != view->shape().dimensions()) {
    LTC_CHECK_EQ(lazy_tensors::util::Multiply<lazy_tensors::int64>(
                     ir::GetShapeFromTsValue(ir_value).dimensions()),
                 lazy_tensors::util::Multiply<lazy_tensors::int64>(
                     view->shape().dimensions()));

    ViewInfo view_info(ViewInfo::Type::kReshape, ir::GetShapeFromTsValue(ir_value),
                       view->shape());
    view = view->CreateSubView(view_info.shape, view_info);
  }
  view->Update(std::move(ir_value));
  return view;
}

void LazyTensor::SetSubView(ViewInfo view_info) const {
  data()->view = data()->view->CreateSubView(view_info.shape, view_info);
  data()->generation += 1;
}

void LazyTensor::ModifyCurrentView(ViewInfo view_info) const {
  if (data()->view != nullptr) {
    data()->view = data()->view->CreateSubView(view_info.shape, view_info);
    return;
  }
  // This node is not a view. Since this function is meant to modify a view
  // in place, we need to turn this existing tensor into a view.
  torch::lazy::Value ir_value = GetIrValue();
  std::shared_ptr<Alias> alias = std::make_shared<Alias>(ir_value);
  data()->view =
      std::make_shared<View>(ir::GetShapeFromTsValue(ir_value), alias, std::move(view_info));
  AssignIrValue(torch::lazy::Value());
}

std::shared_ptr<View> LazyTensor::CreateView(ViewInfo view_info) const {
  if (data()->view != nullptr) {
    return data()->view->CreateSubView(view_info.shape, view_info);
  }
  // This node is not a view, and creating a view forks the current node into
  // becoming one itself. This means creating an alias with the current IR
  // Node, and using the same alias for the created IR Node.
  torch::lazy::Value ir_value = GetIrValue();
  std::shared_ptr<Alias> alias = std::make_shared<Alias>(ir_value);
  ViewInfo this_view_info(ViewInfo::Type::kNoOp, ir::GetShapeFromTsValue(ir_value),
                          ir::GetShapeFromTsValue(ir_value));
  data()->view = std::make_shared<View>(ir::GetShapeFromTsValue(ir_value), alias,
                                        std::move(this_view_info));
  AssignIrValue(torch::lazy::Value());
  return std::make_shared<View>(view_info.shape, alias, view_info);
}

LazyTensor LazyTensor::CreateViewTensor(ViewInfo view_info) const {
  return Create(CreateView(std::move(view_info)), GetDevice(),
                dtype_optional());
}

at::Tensor LazyTensor::ToTensor(bool detached) {
  at::Tensor tensor;
  c10::optional<at::Tensor> tensor_data = CurrentTensorData();
  if (!tensor_data) {
    LazyGraphExecutor::Get()->DeviceBarrier(GetDevice());
    // The GetDataHandle() call will trigger an ApplyPendingGraph() if an IR
    // Node is available on the tensor.
    std::vector<at::Tensor> tensors =
        DataHandlesToTensors({GetDataHandle()}, dtype());
    tensor = std::move(tensors.front());
    if (!detached) {
      SetTensorData(tensor);
    }
  } else {
    tensor = *tensor_data;
    if (detached) {
      if (data()->ir_value || data()->handle != nullptr ||
          data()->view != nullptr) {
        // If we have other authoritive sources, just drop our reference and
        // transfer it to the caller.
        data()->tensor_data = c10::nullopt;
      } else {
        // Otherwise we need to make a copy to prevent the caller changing our
        // version.
        tensor = CopyTensor(tensor);
      }
    }
  }
  return tensor;
}

void LazyTensor::ShallowCopyTo(LazyTensor* dest) const {
  dest->SetIrValue(GetIrValue());
}

void LazyTensor::SetScalarType(
    c10::optional<at::ScalarType> logical_element_type) {
  data()->logical_element_type = logical_element_type;
}

void LazyTensor::SetTensor(at::Tensor tensor) {
  SetTensorData(tensor);
  data()->view = nullptr;
  data()->handle = nullptr;
  AssignIrValue(torch::lazy::Value());
}

void LazyTensor::UpdateFromTensor(at::Tensor tensor, bool sync) {
  if (sync) {
    at::Tensor typed_tensor = CopyTensor(tensor, dtype(), /*copy=*/false);
    SetIrValue(GetIrValueForTensor(typed_tensor, GetDevice()));
  } else {
    SetTensorData(tensor);
    data()->handle = nullptr;
    AssignIrValue(torch::lazy::Value());
    if (data()->view != nullptr) {
      torch::lazy::Value ir_value = GetIrValueForTensor(tensor, GetDevice());
      data()->view = UpdateView(data()->view, std::move(ir_value));
    }
  }
}

void LazyTensor::UpdateFromTensorOut(at::Tensor tensor) {
  if (data()->view != nullptr &&
      lazy_tensors::ShapeUtil::ElementsIn(shape()) != tensor.numel()) {
    data()->view = nullptr;
  }
  UpdateFromTensor(std::move(tensor), /*sync=*/false);
}

void LazyTensor::UpdateFromTensorOut(const LazyTensor& tensor) {
  if (data()->view != nullptr &&
      lazy_tensors::ShapeUtil::ElementsIn(shape()) !=
          lazy_tensors::ShapeUtil::ElementsIn(tensor.shape())) {
    data()->view = nullptr;
  }
  SetIrValue(tensor.GetIrValue());
}


std::vector<LazyTensor> LazyTensor::CreateTensors(
    const std::vector<at::Tensor>& tensors,
    const std::vector<std::string>& devices) {
  std::vector<lazy_tensors::ComputationClient::DataPtr> handles =
      CreateTensorsData(tensors, devices);
  std::vector<LazyTensor> ltc_tensors;
  for (size_t i = 0; i < handles.size(); ++i) {
    ltc_tensors.push_back(
        Create(std::move(handles[i]), tensors[i].scalar_type()));
  }
  return ltc_tensors;
}

torch::lazy::Value LazyTensor::CreateTensorNode(
    lazy_tensors::ComputationClient::DataPtr data, bool read_only) const {
  data->SetInfo(std::make_shared<LazyGraphExecutor::DeviceDataInfo>(
      GetUniqueId(), read_only));
  return torch::lazy::MakeNode<ir::ops::DeviceData>(std::move(data));
}

std::vector<LazyTensor> LazyTensor::MakeOutputTensors(NodePtr node) const {
  std::vector<LazyTensor> tensors;
  tensors.reserve(node->num_outputs());
  for (size_t i = 0; i < node->num_outputs(); ++i) {
    tensors.push_back(CreateFrom(torch::lazy::Value(node, i)));
  }
  return tensors;
}

LazyTensor LazyTensor::CopyTensorToDevice(const Device& device) {
  // TODO: This can be optimized.
  return Create(ToTensor(/*detached=*/true), device);
}

torch::lazy::Value LazyTensor::MaybeCastIrValue(
    torch::lazy::Value ir_value, const Device& device,
    c10::optional<at::ScalarType> logical_element_type) const {
  if (!logical_element_type) {
    logical_element_type = dtype_optional();
  }
  if (logical_element_type &&
      RequiresRawTypeCasting(*logical_element_type, &device)) {
    ir_value = torch::lazy::MakeNode<ir::ops::Cast>(ir_value, *logical_element_type);
  }
  return ir_value;
}

LazyTensor LazyTensor::CreateFrom(torch::lazy::Value ir_value) const {
  ir_value = MaybeCastIrValue(std::move(ir_value), GetDevice(),
                              /*logical_element_type=*/c10::nullopt);
  return Create(std::move(ir_value), GetDevice(), dtype_optional());
}

LazyTensor LazyTensor::CreateFrom(torch::lazy::Value ir_value,
                                  const Device& device) const {
  ir_value = MaybeCastIrValue(std::move(ir_value), device,
                              /*logical_element_type=*/c10::nullopt);
  return Create(std::move(ir_value), device, dtype_optional());
}

LazyTensor LazyTensor::CreateFrom(torch::lazy::Value ir_value,
                                  at::ScalarType logical_element_type) const {
  ir_value =
      MaybeCastIrValue(std::move(ir_value), GetDevice(), logical_element_type);
  return Create(std::move(ir_value), GetDevice(), logical_element_type);
}

LazyTensor LazyTensor::CreateFrom(
    torch::lazy::Value ir_value,
    c10::optional<at::ScalarType> logical_element_type_opt) const {
  ir_value = MaybeCastIrValue(std::move(ir_value), GetDevice(),
                              logical_element_type_opt);
  return Create(std::move(ir_value), GetDevice(), logical_element_type_opt);
}

LazyTensor LazyTensor::CreateFrom(torch::lazy::Value ir_value, const Device& device,
                                  at::ScalarType logical_element_type) const {
  ir_value =
      MaybeCastIrValue(std::move(ir_value), device, logical_element_type);
  return Create(std::move(ir_value), device, logical_element_type);
}

void LazyTensor::ApplyPendingGraph() {
  LazyGraphExecutor::Get()->DeviceBarrier(GetDevice());
  // This method is called to ensure that the tensor data is available on
  // device, so that a call to CurrentDataHandle() returns a valid pointer.
  if (CurrentDataHandle() == nullptr) {
    std::vector<LazyTensor> tensors({*this});
    LazyGraphExecutor::Get()->SyncTensorsGraph(&tensors, {}, /*wait=*/true,
                                               /*sync_ltc_data=*/false);
  }
}

lazy_tensors::int64 LazyTensor::GetNextTensorId() {
  static std::atomic<lazy_tensors::int64>* id_generator =
      new std::atomic<lazy_tensors::int64>(1);
  return id_generator->fetch_add(1);
}

}  // namespace torch_lazy_tensors
