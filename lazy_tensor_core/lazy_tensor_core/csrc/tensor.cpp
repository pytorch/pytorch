#include "lazy_tensor_core/csrc/tensor.h"

#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/internal_ops/cast.h>
#include <torch/csrc/lazy/core/internal_ops/device_data.h>
#include <torch/csrc/lazy/core/internal_ops/scalar.h>
#include <torch/csrc/lazy/core/ir_dump_util.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/tensor_util.h>

#include "lazy_tensor_core/csrc/lazy_graph_executor.h"
#include "lazy_tensor_core/csrc/tensor_impl.h"
#include "lazy_tensors/computation_client/sys_util.h"

namespace torch_lazy_tensors {
namespace {
LazyTensor GetOrCreateLtcTensor(const at::Tensor& tensor,
                                const torch::lazy::BackendDevice& device) {
  if (!tensor.defined()) {
    return LazyTensor();
  }
  auto xtensor = TryGetLtcTensor(tensor);
  return xtensor ? xtensor : LazyTensor::Create(tensor, device);
}
}  // namespace

LazyTensor::Data::~Data() { LazyGraphExecutor::Get()->UnregisterTensor(this); }

LazyTensor LazyTensor::Create(const at::Tensor& tensor, const torch::lazy::BackendDevice& device) {
  CHECK_NE(tensor.device().type(), at::kLazy);
  LazyTensor xtensor(tensor, device);
  LazyGraphExecutor::Get()->RegisterTensor(xtensor.data_ptr());
  return xtensor;
}

LazyTensor LazyTensor::Create(torch::lazy::Value ir_value,
                              const torch::lazy::BackendDevice& device) {
  LazyTensor xtensor(std::move(ir_value), device);
  LazyGraphExecutor::Get()->RegisterTensor(xtensor.data_ptr());
  return xtensor;
}

LazyTensor LazyTensor::Create(std::shared_ptr<torch::lazy::LazyView> view,
                              const torch::lazy::BackendDevice& device) {
  LazyTensor xtensor(std::move(view), device);
  LazyGraphExecutor::Get()->RegisterTensor(xtensor.data_ptr());
  return xtensor;
}

LazyTensor LazyTensor::Create(torch::lazy::BackendDataPtr handle) {
  LazyTensor xtensor(std::move(handle));
  LazyGraphExecutor::Get()->RegisterTensor(xtensor.data_ptr());
  return xtensor;
}

LazyTensor LazyTensor::Create(std::shared_ptr<Data> data) {
  return LazyTensor(std::move(data));
}

LazyTensor::LazyTensor(const at::Tensor& tensor, const torch::lazy::BackendDevice& device)
    : data_(std::make_shared<Data>(tensor, device)) {}

LazyTensor::LazyTensor(torch::lazy::BackendDataPtr handle)
    : data_(std::make_shared<Data>(handle, handle->device())) {}

LazyTensor::LazyTensor(torch::lazy::Value ir_value,
                       const torch::lazy::BackendDevice& device)
    : data_(std::make_shared<Data>(std::move(ir_value), device)) {
  TryLimitGraphSize();
}

LazyTensor::LazyTensor(std::shared_ptr<torch::lazy::LazyView> view,
                       const torch::lazy::BackendDevice& device)
    : data_(std::make_shared<Data>(std::move(view), device)) {}

LazyTensor::LazyTensor(std::shared_ptr<Data> data) : data_(std::move(data)) {}

LazyTensor::Data* LazyTensor::data() const {
  CHECK(data_ != nullptr) << "Trying to access a null cursor";
  return data_.get();
}

int64_t LazyTensor::size(int64_t dim) const {
  auto tensor_shape = shape();
  int rank = tensor_shape.Get().dim();
  int dim_index = torch::lazy::GetCanonicalDimensionIndex(dim, rank);
  return tensor_shape.Get().size(dim_index);
}

at::ScalarType LazyTensor::dtype() const { return shape().Get().scalar_type(); }

torch::lazy::MaybeRef<torch::lazy::Shape> LazyTensor::shape() const {
  if (data()->view != nullptr) {
    return data()->view->shape();
  }
  if (data()->handle != nullptr) {
    return torch::lazy::Shape(data()->handle->shape());
  }
  if (data()->ir_value) {
    // TODO(whc) remove shape from LazyTensor API too!
    return torch::lazy::GetShapeFromTsValue(data()->ir_value);
  }
  CHECK(data()->tensor_data);
  return torch::lazy::Shape(
      data()->tensor_data->scalar_type(),
      torch::lazy::ToI64Vector(data()->tensor_data->sizes()));
}

const torch::lazy::BackendDevice& LazyTensor::GetDevice() const { return data()->device; }

int64_t LazyTensor::GetUniqueId() const { return data()->unique_id; }

std::ptrdiff_t LazyTensor::GetViewAliasId() const {
  return data()->view != nullptr
             ? reinterpret_cast<std::ptrdiff_t>(data()->view->alias().get())
             : 0;
}

torch::lazy::BackendDataPtr LazyTensor::GetDataHandle() {
  // Data can coexist with a view, but we need to check that the view did
  // not receive any updates before calling the current IR valid.
  bool up_to_date = true;
  torch::lazy::Value ir_value;
  if (data()->view != nullptr) {
    bool updated = false;
    std::tie(ir_value, updated) = GetViewUpdate(data()->view);
    up_to_date = !updated;
  }
  if (up_to_date) {
    torch::lazy::BackendDataPtr handle = CurrentDataHandle();
    if (handle != nullptr) {
      CHECK(handle->HasValue())
          << "Trying to access data while an async operation is in flight: "
          << torch::lazy::Shape(handle->shape());
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
    CHECK(data()->tensor_data);
    data()->handle = TensorToDataHandle(*data()->tensor_data, GetDevice());
  }
  return data()->handle;
}

torch::lazy::BackendDataPtr LazyTensor::CurrentDataHandle() const {
  return data()->handle;
}

void LazyTensor::SetDataHandle(torch::lazy::BackendDataPtr handle) {
  SetDataHandle(std::move(handle), /*sync=*/true);
}

void LazyTensor::SetDataHandle(torch::lazy::BackendDataPtr handle, bool sync) {
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
  if (tensor_shape.Get().scalar_type() !=
      torch::lazy::GetShapeFromTsValue(ir_value).scalar_type()) {
    ir_value = torch::lazy::MakeNode<torch::lazy::Cast>(
        ir_value, tensor_shape.Get().scalar_type());
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
    size_t graph_size =
        torch::lazy::Util::GetGraphSize({data()->ir_value.node.get()});
    if (graph_size > kMaxPendingGraphSize) {
      TORCH_LAZY_COUNTER("TrimIrGraph", 1);
      ApplyPendingGraph();
    }
  }
}

torch::lazy::Value LazyTensor::GetIrValue() const {
  torch::lazy::Value ir_value = CurrentIrValue();
  if (ir_value) {
    return ir_value;
  }
  torch::lazy::BackendDataPtr handle = CurrentDataHandle();
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
  CHECK(tensor_data);
  AssignIrValue(GetIrValueForTensor(*tensor_data, GetDevice()));
  return data()->ir_value;
}

torch::lazy::Value LazyTensor::CurrentIrValue() const {
  if (data()->view != nullptr) {
    return std::get<0>(GetViewUpdate(data()->view));
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
                                                   const torch::lazy::BackendDevice& device) const {
  torch::lazy::BackendDataPtr data;
  bool read_only = false;
  if (tensor.dim() == 0 && tensor.numel() == 1) {
    at::Scalar value = tensor.item();
    if (torch::lazy::IsSpecialScalar(value)) {
      return torch::lazy::MakeNode<torch::lazy::Scalar>(std::move(value),
                                                    tensor.scalar_type());
    }
    data = LazyGraphExecutor::Get()->GetDeviceData(tensor.cpu(), device);
    read_only = true;
  } else {
    TORCH_LAZY_TIMED("IrValueTensorToDataHandle");
    data = TensorToDataHandle(tensor, device);
  }
  return CreateTensorNode(std::move(data), read_only);
}

std::tuple<torch::lazy::Value, bool> LazyTensor::GetViewUpdate(
    const std::shared_ptr<torch::lazy::LazyView>& view) const {
  auto value_with_update = view->GetViewIrNode();
  if (std::get<1>(value_with_update)) {
    data()->handle = nullptr;
    data()->tensor_data = c10::nullopt;
  }
  return value_with_update;
}

std::shared_ptr<torch::lazy::LazyView> LazyTensor::UpdateView(
    std::shared_ptr<torch::lazy::LazyView> view,
    torch::lazy::Value ir_value) const {
  if (torch::lazy::GetShapeFromTsValue(ir_value).sizes() !=
      view->shape().sizes()) {
    CHECK_EQ(torch::lazy::GetShapeFromTsValue(ir_value).numel(),
             view->shape().numel());

    torch::lazy::ViewInfo view_info(torch::lazy::ViewInfo::Type::kReshape,
                                    torch::lazy::GetShapeFromTsValue(ir_value),
                                    view->shape());
    view = view->CreateSubView(view_info.shape, view_info);
  }
  view->Update(std::move(ir_value));
  return view;
}

void LazyTensor::SetSubView(torch::lazy::ViewInfo view_info) const {
  data()->view = data()->view->CreateSubView(view_info.shape, view_info);
  data()->generation += 1;
}

void LazyTensor::ModifyCurrentView(torch::lazy::ViewInfo view_info) const {
  if (data()->view != nullptr) {
    SetSubView(view_info);
    return;
  }
  // This node is not a view. Since this function is meant to modify a view
  // in place, we need to turn this existing tensor into a view.
  torch::lazy::Value ir_value = GetIrValue();
  std::shared_ptr<torch::lazy::Alias> alias =
      std::make_shared<torch::lazy::Alias>(ir_value);
  data()->view = std::make_shared<torch::lazy::LazyView>(view_info.shape, alias,
                                                         std::move(view_info));
  AssignIrValue(torch::lazy::Value());
}

std::shared_ptr<torch::lazy::LazyView> LazyTensor::CreateView(
    torch::lazy::ViewInfo view_info) const {
  if (data()->view != nullptr) {
    return data()->view->CreateSubView(view_info.shape, view_info);
  }
  // This node is not a view, and creating a view forks the current node into
  // becoming one itself. This means creating an alias with the current IR
  // Node, and using the same alias for the created IR Node.
  torch::lazy::Value ir_value = GetIrValue();
  std::shared_ptr<torch::lazy::Alias> alias =
      std::make_shared<torch::lazy::Alias>(ir_value);
  torch::lazy::ViewInfo this_view_info(
      torch::lazy::ViewInfo::Type::kNoOp,
      torch::lazy::GetShapeFromTsValue(ir_value),
      torch::lazy::GetShapeFromTsValue(ir_value));
  data()->view = std::make_shared<torch::lazy::LazyView>(
      torch::lazy::GetShapeFromTsValue(ir_value), alias,
      std::move(this_view_info));
  AssignIrValue(torch::lazy::Value());
  return std::make_shared<torch::lazy::LazyView>(view_info.shape, alias,
                                                 view_info);
}

LazyTensor LazyTensor::CreateViewTensor(torch::lazy::ViewInfo view_info) const {
  return Create(CreateView(std::move(view_info)), GetDevice());
}

at::Tensor LazyTensor::ToTensor(bool detached) {
  at::Tensor tensor;
  c10::optional<at::Tensor> tensor_data = CurrentTensorData();
  if (!tensor_data) {
    LazyGraphExecutor::Get()->DeviceBarrier(GetDevice());
    // The GetDataHandle() call will trigger an ApplyPendingGraph() if an IR
    // Node is available on the tensor.
    std::vector<at::Tensor> tensors =
        torch::lazy::DataHandlesToTensors({GetDataHandle()}, dtype());
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
        tensor = torch::lazy::CopyTensor(tensor);
      }
    }
  }
  return tensor;
}

void LazyTensor::ShallowCopyTo(LazyTensor* dest) const {
  dest->SetIrValue(GetIrValue());
}

void LazyTensor::SetTensor(at::Tensor tensor) {
  SetTensorData(tensor);
  data()->view = nullptr;
  data()->handle = nullptr;
  AssignIrValue(torch::lazy::Value());
}

void LazyTensor::UpdateFromTensor(at::Tensor tensor, bool sync) {
  if (sync) {
    at::Tensor typed_tensor =
        torch::lazy::CopyTensor(tensor, dtype(), /*copy=*/false);
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
  if (data()->view != nullptr && shape().Get().numel() != tensor.numel()) {
    data()->view = nullptr;
  }
  UpdateFromTensor(std::move(tensor), /*sync=*/false);
}

void LazyTensor::UpdateFromTensorOut(const LazyTensor& tensor) {
  if (data()->view != nullptr && shape().Get().numel() != tensor.shape().Get().numel()) {
    data()->view = nullptr;
  }
  SetIrValue(tensor.GetIrValue());
}

torch::lazy::Value LazyTensor::CreateTensorNode(
    torch::lazy::BackendDataPtr data, bool read_only) const {
  data->SetInfo(std::make_shared<LazyGraphExecutor::DeviceDataInfo>(
      GetUniqueId(), read_only));
  return torch::lazy::MakeNode<torch::lazy::DeviceData>(std::move(data));
}

std::vector<LazyTensor> LazyTensor::MakeOutputTensors(
    torch::lazy::NodePtr node) const {
  std::vector<LazyTensor> tensors;
  tensors.reserve(node->num_outputs());
  for (size_t i = 0; i < node->num_outputs(); ++i) {
    tensors.push_back(Create(torch::lazy::Value(node, i), GetDevice()));
  }
  return tensors;
}

LazyTensor LazyTensor::CopyTensorToDevice(const torch::lazy::BackendDevice& device) {
  // TODO: This can be optimized.
  return Create(ToTensor(/*detached=*/true), device);
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

int64_t LazyTensor::GetNextTensorId() {
  static std::atomic<int64_t>* id_generator = new std::atomic<int64_t>(1);
  return id_generator->fetch_add(1);
}

LazyTensor TryGetLtcTensor(const at::Tensor& tensor) {
  auto* impl = dynamic_cast<LTCTensorImpl*>(tensor.unsafeGetTensorImpl());
  if (impl == nullptr) {
    return LazyTensor();
  }
  return impl->tensor();
}

LazyTensor GetLtcTensor(const at::Tensor& tensor) {
  auto lazy_tensor = TryGetLtcTensor(tensor);
  CHECK(lazy_tensor) << "Input tensor is not a lazy tensor: " << tensor.toString();
  return lazy_tensor;
}

std::vector<LazyTensor> GetLtcTensors(c10::ArrayRef<at::Tensor> tensors) {
  std::vector<LazyTensor> ltc_tensors;
  ltc_tensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    ltc_tensors.push_back(TryGetLtcTensor(tensor));
  }
  return ltc_tensors;
}

LazyTensor GetOrCreateLtcTensor(const c10::optional<at::Tensor>& tensor,
                                const torch::lazy::BackendDevice& device) {
  return GetOrCreateLtcTensor(tensor.value_or(at::Tensor()), device);
}

LazyTensor GetLtcTensorOrCreateForWrappedNumber(const at::Tensor& tensor, const torch::lazy::BackendDevice& device) {
  return tensor.unsafeGetTensorImpl()->is_wrapped_number() ?
      GetOrCreateLtcTensor(tensor, device) : GetLtcTensor(tensor);
}

at::Tensor CreateAtenFromLtcTensor(const LazyTensor& ltc_tensor) {
  return ltc_tensor.is_null() ? at::Tensor()
                              : at::Tensor(c10::make_intrusive<LTCTensorImpl>(ltc_tensor));
}

at::Tensor CreateAtenFromLtcTensor(LazyTensor&& ltc_tensor) {
  return ltc_tensor.is_null() ? at::Tensor()
                              : at::Tensor(c10::make_intrusive<LTCTensorImpl>(std::move(ltc_tensor)));
}

}  // namespace torch_lazy_tensors
