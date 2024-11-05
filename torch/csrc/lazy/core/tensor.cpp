#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/tensor.h>

#include <c10/util/irange.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/ir_builder.h>
#include <torch/csrc/lazy/core/ir_dump_util.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/tensor_impl.h>
#include <torch/csrc/lazy/core/tensor_util.h>

#include <ATen/FunctionalTensorWrapper.h>

#include <utility>

namespace torch::lazy {
namespace {
LazyTensorPtr GetOrCreateLtcTensor(
    const at::Tensor& tensor,
    const BackendDevice& device) {
  if (!tensor.defined()) {
    return torch::lazy::LazyTensorPtr();
  }
  auto lazy_tensor = TryGetLtcTensor(tensor);
  return lazy_tensor ? lazy_tensor : LazyTensor::Create(tensor, device);
}
} // namespace

LazyTensor::Data::~Data() {
  LazyGraphExecutor::Get()->UnregisterTensor(this);
}

LazyTensorPtr LazyTensor::Create(
    const at::Tensor& tensor,
    const BackendDevice& device) {
  TORCH_CHECK(tensor.device().type() != at::kLazy);
  LazyTensorPtr lazy_tensor =
      c10::make_intrusive<LazyTensor>(LazyTensor(tensor, device));
  LazyGraphExecutor::Get()->RegisterTensor(lazy_tensor->data());
  return lazy_tensor;
}

LazyTensorPtr LazyTensor::Create(Value ir_value, const BackendDevice& device) {
  LazyTensorPtr lazy_tensor =
      c10::make_intrusive<LazyTensor>(LazyTensor(std::move(ir_value), device));
  LazyGraphExecutor::Get()->RegisterTensor(lazy_tensor->data());
  return lazy_tensor;
}

LazyTensorPtr LazyTensor::Create(const BackendDataPtr& handle) {
  LazyTensorPtr lazy_tensor =
      c10::make_intrusive<LazyTensor>(LazyTensor(handle));
  LazyGraphExecutor::Get()->RegisterTensor(lazy_tensor->data());
  return lazy_tensor;
}

LazyTensorPtr LazyTensor::Create(std::shared_ptr<Data> data) {
  return c10::make_intrusive<LazyTensor>(LazyTensor(std::move(data)));
}

LazyTensor::LazyTensor(const at::Tensor& tensor, const BackendDevice& device)
    : LazyTensor(std::make_shared<Data>(tensor, device)) {}

LazyTensor::LazyTensor(const BackendDataPtr& handle)
    : LazyTensor(std::make_shared<Data>(handle, handle->device())) {}

LazyTensor::LazyTensor(Value ir_value, const BackendDevice& device)
    : LazyTensor(std::make_shared<Data>(std::move(ir_value), device)) {
  TryLimitGraphSize();
}

LazyTensor::LazyTensor(std::shared_ptr<Data> data) : data_(std::move(data)) {}

auto LazyTensor::data() const -> const std::shared_ptr<Data>& {
  TORCH_CHECK(data_ != nullptr, "Trying to access a null cursor");
  return data_;
}

int64_t LazyTensor::size(int64_t dim) const {
  auto tensor_shape = shape();
  auto rank = tensor_shape.Get().dim();
  auto dim_index = GetCanonicalDimensionIndex(dim, rank);
  return tensor_shape.Get().size(dim_index);
}

at::ScalarType LazyTensor::dtype() const {
  return shape().Get().scalar_type();
}

MaybeRef<Shape> LazyTensor::shape() const {
  if (data()->handle != nullptr) {
    return Shape(data()->handle->shape());
  }
  if (data()->ir_value) {
    // TODO(whc) remove shape from LazyTensor API too!
    return data()->ir_value.shape();
  }
  TORCH_CHECK(data()->tensor_data);
  return Shape(
      data()->tensor_data->scalar_type(),
      ToI64Vector(data()->tensor_data->sizes()));
}

const BackendDevice& LazyTensor::GetDevice() const {
  return data()->device;
}

int64_t LazyTensor::GetUniqueId() const {
  return data()->unique_id;
}

BackendDataPtr LazyTensor::GetDataHandle() {
  BackendDataPtr handle = CurrentDataHandle();
  if (handle != nullptr) {
    TORCH_CHECK(
        handle->HasValue(),
        "Trying to access data while an async operation is in flight: ",
        handle->shape().to_string());
    return handle;
  }

  if (data()->ir_value) {
    ApplyPendingGraph();
  } else {
    TORCH_CHECK(data()->tensor_data);
    data()->handle = TensorToDataHandle(*data()->tensor_data, GetDevice());
  }

  return data()->handle;
}

BackendDataPtr LazyTensor::CurrentDataHandle() const {
  return data()->handle;
}

void LazyTensor::SetDataHandle(BackendDataPtr handle) {
  SetDataHandle(std::move(handle), /*sync=*/true);
}

void LazyTensor::SetDataHandle(BackendDataPtr handle, bool sync) {
  data()->handle = std::move(handle);
  // Assigning a device data should always clear the IR node, to allow graph
  // trimming.
  AssignIrValue(Value());
  if (sync) {
    data()->tensor_data = std::nullopt;
  }
}

void LazyTensor::SetIrValue(Value ir_value) {
  data()->handle = nullptr;
  data()->tensor_data = std::nullopt;
  AssignIrValue(std::move(ir_value));
  TryLimitGraphSize();
}

void LazyTensor::SetInPlaceIrValue(Value ir_value) {
  auto tensor_shape = shape();
  if (tensor_shape.Get().scalar_type() != ir_value.shape().scalar_type()) {
    ir_value =
        MakeCast(ir_value, tensor_shape.Get().scalar_type(), std::nullopt);
  }
  SetIrValue(std::move(ir_value));
}

void LazyTensor::AssignIrValue(Value ir_value) const {
  data()->ir_value = std::move(ir_value);
  data()->generation += 1;
}

void LazyTensor::TryLimitGraphSize() {
  if (data()->ir_value &&
      LazyGraphExecutor::Get()->IncTrimCounter() %
              FLAGS_torch_lazy_trim_graph_check_frequency ==
          0) {
    size_t graph_size = Util::GetGraphSize({data()->ir_value.node.get()});
    if (static_cast<int64_t>(graph_size) > FLAGS_torch_lazy_trim_graph_size) {
      TORCH_LAZY_COUNTER("TrimIrGraph", 1);
      ApplyPendingGraph();
    }
  }
}

Value LazyTensor::GetIrValue() const {
  Value ir_value = CurrentIrValue();
  if (ir_value) {
    return ir_value;
  }
  BackendDataPtr handle = CurrentDataHandle();
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
  std::optional<at::Tensor> tensor_data = CurrentTensorData();
  TORCH_CHECK(tensor_data);
  AssignIrValue(GetIrValueForTensor(*tensor_data, GetDevice()));
  return data()->ir_value;
}

Value LazyTensor::CurrentIrValue() const {
  return data()->ir_value;
}

void LazyTensor::SetTensorData(at::Tensor tensor_data) {
  data()->tensor_data = std::move(tensor_data);
}

std::optional<at::Tensor> LazyTensor::CurrentTensorData() const {
  return data()->tensor_data;
}

Value LazyTensor::GetIrValueForTensor(
    const at::Tensor& tensor,
    const BackendDevice& device) const {
  BackendDataPtr data;
  bool read_only = false;
  if (tensor.dim() == 0 && tensor.numel() == 1) {
    at::Scalar value = tensor.item();
    if (IsSpecialScalar(value)) {
      return MakeScalar(value, tensor.scalar_type());
    }
    data = LazyGraphExecutor::Get()->GetDeviceData(tensor.cpu(), device);
    read_only = true;
  } else {
    TORCH_LAZY_TIMED("IrValueTensorToDataHandle");
    data = TensorToDataHandle(tensor, device);
  }
  return CreateTensorNode(data, read_only);
}

at::Tensor LazyTensor::ToTensor(bool detached) {
  at::Tensor tensor;
  std::optional<at::Tensor> tensor_data = CurrentTensorData();
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
      if (data()->ir_value || data()->handle != nullptr) {
        // If we have other authoritive sources, just drop our reference and
        // transfer it to the caller.
        data()->tensor_data = std::nullopt;
      } else {
        // Otherwise we need to make a copy to prevent the caller changing our
        // version.
        tensor = CopyTensor(tensor);
      }
    }
  }
  return tensor;
}

void LazyTensor::ShallowCopyTo(const LazyTensorPtr& dest) const {
  dest->SetIrValue(GetIrValue());
}

void LazyTensor::SetTensor(at::Tensor tensor) {
  SetTensorData(std::move(tensor));
  data()->handle = nullptr;
  AssignIrValue(Value());
}

void LazyTensor::UpdateFromTensor(const at::Tensor& tensor, bool sync) {
  if (sync) {
    at::Tensor typed_tensor = CopyTensor(tensor, dtype(), /*copy=*/false);
    SetIrValue(GetIrValueForTensor(typed_tensor, GetDevice()));
  } else {
    SetTensorData(tensor);
    data()->handle = nullptr;
    AssignIrValue(Value());
  }
}

void LazyTensor::UpdateFromTensorOut(const at::Tensor& tensor) {
  UpdateFromTensor(tensor, /*sync=*/false);
}

void LazyTensor::UpdateFromTensorOut(const LazyTensorPtr& tensor) {
  SetIrValue(tensor->GetIrValue());
}

Value LazyTensor::CreateTensorNode(const BackendDataPtr& data, bool read_only)
    const {
  data->SetInfo(std::make_shared<LazyGraphExecutor::DeviceDataInfo>(
      GetUniqueId(), read_only));
  return MakeDeviceData(data);
}

std::vector<LazyTensorPtr> LazyTensor::MakeOutputTensors(
    const NodePtr& node) const {
  std::vector<LazyTensorPtr> tensors;
  tensors.reserve(node->num_outputs());
  for (const auto i : c10::irange(node->num_outputs())) {
    tensors.push_back(Create(Value(node, i), GetDevice()));
  }
  return tensors;
}

LazyTensorPtr LazyTensor::CopyTensorToDevice(const BackendDevice& device) {
  // TODO: This can be optimized.
  return Create(ToTensor(/*detached=*/true), device);
}

void LazyTensor::ApplyPendingGraph() {
  LazyGraphExecutor::Get()->DeviceBarrier(GetDevice());
  // This method is called to ensure that the tensor data is available on
  // device, so that a call to CurrentDataHandle() returns a valid pointer.
  if (CurrentDataHandle() == nullptr) {
    std::vector<LazyTensorPtr> tensors(
        {c10::make_intrusive<LazyTensor>(LazyTensor(*this))});
    LazyGraphExecutor::Get()->SyncTensorsGraph(
        &tensors,
        {},
        /*wait=*/true,
        /*sync_ltc_data=*/false);
  }
}

int64_t LazyTensor::GetNextTensorId() {
  static std::atomic<int64_t>* id_generator = new std::atomic<int64_t>(1);
  return id_generator->fetch_add(1);
}

torch::lazy::Value GetTensorList(at::ITensorListRef tensors) {
  std::vector<Value> values;
  for (const auto& t : tensors) {
    auto* impl = dynamic_cast<LTCTensorImpl*>(t.unsafeGetTensorImpl());
    TORCH_INTERNAL_ASSERT(
        impl,
        "GetTensorList only supports lists of valid tensors, but optional support could be added");
    values.push_back(impl->tensor()->GetIrValue());
  }

  return torch::lazy::Value(torch::lazy::MakeTensorList(values));
}

LazyTensorPtr TryGetLtcTensor(const at::Tensor& tensor) {
  auto* impl = dynamic_cast<LTCTensorImpl*>(
      maybe_unwrap_functional(tensor).unsafeGetTensorImpl());
  if (impl == nullptr) {
    // return c10::make_intrusive<LazyTensor>();
    return LazyTensorPtr();
  }
  return impl->tensor();
}

LazyTensorPtr GetLtcTensor(const at::Tensor& tensor) {
  auto lazy_tensor = TryGetLtcTensor(tensor);
  TORCH_CHECK(
      lazy_tensor, "Input tensor is not a lazy tensor: ", tensor.toString());
  return lazy_tensor;
}

std::vector<LazyTensorPtr> GetLtcTensors(c10::ArrayRef<at::Tensor> tensors) {
  std::vector<LazyTensorPtr> ltc_tensors;
  ltc_tensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    ltc_tensors.emplace_back(TryGetLtcTensor(tensor));
  }
  return ltc_tensors;
}

LazyTensorPtr GetOrCreateLtcTensor(
    const std::optional<at::Tensor>& tensor,
    const BackendDevice& device) {
  return GetOrCreateLtcTensor(tensor.value_or(at::Tensor()), device);
}

LazyTensorPtr GetLtcTensorOrCreateForWrappedNumber(
    const at::Tensor& tensor,
    const BackendDevice& device) {
  // TODO: There are places in core where a scalar is wrapped but not marked as
  // wrapped.
  return (tensor.unsafeGetTensorImpl()->is_wrapped_number() ||
          (tensor.dim() == 0 && tensor.numel() == 1))
      ? GetOrCreateLtcTensor(tensor, device)
      : GetLtcTensor(tensor);
}

at::Tensor CreateAtenFromLtcTensor(const LazyTensorPtr& ltc_tensor) {
  return ltc_tensor ? at::Tensor(c10::make_intrusive<LTCTensorImpl>(ltc_tensor))
                    : at::Tensor();
}

at::Tensor CreateAtenFromLtcTensor(LazyTensor&& ltc_tensor) {
  return at::Tensor(c10::make_intrusive<LTCTensorImpl>(std::move(ltc_tensor)));
}

at::Tensor to_lazy_tensor(
    const at::Tensor& self,
    const c10::TensorOptions& options,
    at::Device device,
    bool non_blocking,
    bool functionalize_output) {
  TORCH_INTERNAL_ASSERT(self.device().type() != c10::kLazy);
  TORCH_INTERNAL_ASSERT(device.type() == c10::kLazy);

  auto eager_tensor =
      self.to(options, /*non_blocking=*/non_blocking, /*copy=*/true);
  auto lazy_self = torch::lazy::GetOrCreateLtcTensor(
      eager_tensor, torch::lazy::atenDeviceToBackendDevice(device));
  auto out = torch::lazy::CreateAtenFromLtcTensor(lazy_self);
  if (functionalize_output) {
    // See Note [Lazy Tensor Functionalization]
    return at::functionalization::impl::to_functional_tensor(out);
  } else {
    return out;
  }
}

} // namespace torch::lazy
