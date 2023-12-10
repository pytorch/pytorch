#pragma once

#include <c10/core/SymNodeImpl.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/util.h>

namespace torch {
namespace lazy {

class TORCH_API SymNodeImpl : public c10::SymNodeImpl {
 public:
  SymNodeImpl(NodePtr ptr) : node_(std::move(ptr)){};
  NodePtr node_;
};

class LazyTensor;
using LazyTensorPtr = c10::intrusive_ptr<LazyTensor>;

class TORCH_API LazyTensor : public c10::intrusive_ptr_target {
 public:
  // This is the core lazy tensor data structure where all the tensor data is
  // held. The lazy tensor is nothing more than a shared pointer to a Data
  // object.
  struct Data {
    Data(BackendDataPtr handle, BackendDevice device)
        : handle(std::move(handle)),
          device(std::move(device)),
          unique_id(GetNextTensorId()) {}
    Data(Value ir_value, BackendDevice device)
        : ir_value(std::move(ir_value)),
          device(std::move(device)),
          unique_id(GetNextTensorId()) {}
    Data(at::Tensor tensor_data, BackendDevice device)
        : tensor_data(std::move(tensor_data)),
          device(std::move(device)),
          unique_id(GetNextTensorId()) {}
    // TODO(alanwaketan): Remove this ctor. This is a
    // temporary ctor to ease XLA LTC migration. It depends on
    // XLA's Functionalization integration.
    Data(BackendDevice device)
        : device(std::move(device)), unique_id(GetNextTensorId()) {}

    virtual ~Data();

    BackendDataPtr handle;
    Value ir_value;
    c10::optional<at::Tensor> tensor_data;
    const BackendDevice device;
    const int64_t unique_id = 0;
    size_t generation = 1;
  };

  static LazyTensorPtr Create(
      const at::Tensor& tensor,
      const BackendDevice& device);
  static LazyTensorPtr Create(Value ir_value, const BackendDevice& device);
  static LazyTensorPtr Create(BackendDataPtr handle);
  static LazyTensorPtr Create(std::shared_ptr<Data> data);

  // The default ctor previously created a null LazyTensor (one with no 'data'
  // obj). Creating a null LazyTensor is no longer possible, since the same can
  // be achieved by creating a null LazyTensorPtr and it is way too confusing to
  // have to check both lazy_tensor_ptr && *lazy_tensor_ptr, so everywhere that
  // used to rely on a LazyTensor obj with a null Data can now rely on a null
  // LazyTensorPtr instead.
  LazyTensor() = delete;
  LazyTensor(const LazyTensor&) = default;
  LazyTensor(LazyTensor&&) noexcept = default;

  ~LazyTensor() override = default;

  size_t generation() const {
    return data()->generation;
  }

  // Override it to use your own Shape.
  virtual int64_t size(int64_t dim) const;

  // Override it to use your own graph executor.
  virtual at::Tensor ToTensor(bool detached);

  void ShallowCopyTo(LazyTensorPtr dest) const;

  // Assigns the tensor value to the lazy tensor.
  void SetTensor(at::Tensor tensor);

  void UpdateFromTensor(at::Tensor tensor, bool sync);
  void UpdateFromTensorOut(at::Tensor tensor);
  void UpdateFromTensorOut(const LazyTensorPtr& tensor);

  const std::shared_ptr<Data>& data() const;

  // Override it to use your own type conversion.
  virtual at::ScalarType dtype() const;

  MaybeRef<Shape> shape() const;

  const BackendDevice& GetDevice() const;
  int64_t GetUniqueId() const;

  // Fetches the data behind the tensor. If the tensor has a graph defining
  // its current value, executes the graph and fetches the data result.
  BackendDataPtr GetDataHandle();

  // Fetches the current value of the data, which can be missing (nullptr)
  // in case the tensor has a graph defining its current value,
  BackendDataPtr CurrentDataHandle() const;

  void SetDataHandle(BackendDataPtr handle);
  void SetDataHandle(BackendDataPtr handle, bool sync);

  // Retrieves the current IR Node, or nullptr in case no active IR Node is
  // available.
  Value CurrentIrValue() const;

  // Retrieves the IR Node representing this LazyTensor. One will be created if
  // missing. Note that although this is a const API, it actually changes the
  // internal state ofthe object.
  Value GetIrValue() const;

  void SetIrValue(Value ir_value);
  void SetInPlaceIrValue(Value ir_value);

  c10::optional<at::Tensor> CurrentTensorData() const;

  std::vector<LazyTensorPtr> MakeOutputTensors(NodePtr node) const;

  LazyTensorPtr CopyTensorToDevice(const BackendDevice& device);

  // Applies the queue of operations in preparation for using the data.
  // Override it to use your own graph executor.
  virtual void ApplyPendingGraph();

  // Override it to set extra information.
  virtual void AssignIrValue(Value ir_value) const;

 protected:
  explicit LazyTensor(std::shared_ptr<Data> data);

  void SetTensorData(at::Tensor tensor_data);

  // We build a graph accumulating operations, but at a given point we
  // need to force a rendering, otherwise the graph can grow without control.
  // Think:
  //   for i in range(0, 100000):
  //     a = a + b
  void TryLimitGraphSize();

  // Override it to instantiate your own data.
  virtual Value GetIrValueForTensor(
      const at::Tensor& tensor,
      const BackendDevice& device) const;

  Value CreateTensorNode(BackendDataPtr data, bool read_only) const;

 private:
  LazyTensor(const at::Tensor& tensor, const BackendDevice& device);
  LazyTensor(Value ir_value, const BackendDevice& device);
  explicit LazyTensor(BackendDataPtr handle);

  static int64_t GetNextTensorId();

  std::shared_ptr<Data> data_;
};

// Utils to convert at::Tensor to LazyTensor, and vice versa.

// Section 0: c10::Tensorlist ==> lazy::TensorList
// note: GetTensorList is not totally parallel to GetLtcTensor; A TensorList
// skips
//       the LazyTensor wrappers, assuming that the list of underlying IR nodes
//       is actually more useful for downstream computations.  TBD.
TORCH_API torch::lazy::Value GetTensorList(at::ITensorListRef tensors);

// Section 1: at::Tensor => LazyTensor.
// Extracts the LazyTensor out of an at::Tensor. Returns a null LazyTensor
// if the tensor is not a lazy tensor.
TORCH_API LazyTensorPtr TryGetLtcTensor(const at::Tensor& tensor);

// Extracts the LazyTensor out of an at::Tensor. Throws an exception
// if the tensor is not a lazy tensor.
TORCH_API LazyTensorPtr GetLtcTensor(const at::Tensor& tensor);

// Same as above, applied to a list of tensors.
TORCH_API std::vector<LazyTensorPtr> GetLtcTensors(
    c10::ArrayRef<at::Tensor> tensors);

// If tensor is a lazy tensor type, returns the LazyTensor embedded within it,
// otherwise creates a new lazy tensor type with tensor as data.
TORCH_API LazyTensorPtr GetOrCreateLtcTensor(
    const c10::optional<at::Tensor>& tensor,
    const BackendDevice& device);

TORCH_API LazyTensorPtr GetLtcTensorOrCreateForWrappedNumber(
    const at::Tensor& tensor,
    const BackendDevice& device);

// Section 2: LazyTensor => at::Tensor.
// Creates an ATen tensor from an LazyTensor.
TORCH_API at::Tensor CreateAtenFromLtcTensor(const LazyTensorPtr& ltc_tensor);
TORCH_API at::Tensor CreateAtenFromLtcTensor(LazyTensor&& ltc_tensor);

// Note [Lazy Tensor Functionalization]
// The functionalization pass is implemented by wrapping all TensorImpl
// objects in C++ with an extra FunctionalTensorWrapper object,
// that knows how to perform functionalization
//
// Certain functions in the aten API serve as entry/exit points for
// functionalization, where we need to perform the wrapping/unwrapping:
// - aten::to.device
// - aten::empty

// Given a non-lazy tensor, this function creates a lazy tensor on the specified
// (lazy) device. The functionalize_output determines whether or not we should
// wrap the output in a "functional wrapper".
//
// How do you know whether to pass true/false for functionalize_output?
//
// Case 1: nonlazy -> lazy
//   If you're implementing a function that takes in nonlazy tensors and returns
//   lazy tensors, then you should think of that function as an "entrypoint" to
//   functionalization, and use functionalize_output=true Examples include:
//   - factory functions (the LTC kernel for at::empty)
//   - CPU -> Lazy device converions (the LTC kernel for at::to_device)
//
// Case 2: lazy -> lazy
//   If you're implementing a function that takes in lazy tensors and returns
//   lazy tensors,
//   **but** requires creating lazy tensors internally,
//   then you can assume that the current function is running inside of some
//   outer context where functionalization is already running, that will take
//   care of doing the wrapping for you, and use functionalize_output=true
//   Examples include:
//   - CPU fallback (takes in lazy tensors, converts to cpu, calls kernel,
//   converts returns back to lazy tensors).
TORCH_API at::Tensor to_lazy_tensor(
    const at::Tensor& self,
    const c10::TensorOptions& options,
    at::Device device,
    bool non_blocking,
    bool functionalize_output);

template <size_t... Indices>
auto TupleAtenFromLtcTensorsImpl(
    const std::vector<LazyTensorPtr>& tensors,
    std::index_sequence<Indices...>) {
  return std::make_tuple(CreateAtenFromLtcTensor(tensors[Indices])...);
}

template <size_t N>
auto TupleAtenFromLtcTensors(const std::vector<LazyTensorPtr>& tensors) {
  return TupleAtenFromLtcTensorsImpl(tensors, std::make_index_sequence<N>{});
}

} // namespace lazy
} // namespace torch
