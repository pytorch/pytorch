#pragma once

#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/lazy_view.h>
#include <torch/csrc/lazy/core/util.h>

namespace torch_lazy_tensors {

std::unordered_set<int64_t>& GetDestroyedBackendDatas();

class LazyTensor {
 public:
  // This is the core lazy tensor data structure where all the tensor data is
  // held. The lazy tensor is nothing more than a shared pointer to a Data
  // object.
  struct Data {
    Data(torch::lazy::BackendDataPtr handle,
         const torch::lazy::BackendDevice& device)
        : handle(std::move(handle)),
          device(device),
          unique_id(GetNextTensorId()) {}
    Data(torch::lazy::Value ir_value, const torch::lazy::BackendDevice& device)
        : ir_value(std::move(ir_value)),
          device(device),
          unique_id(GetNextTensorId()) {}
    Data(std::shared_ptr<torch::lazy::LazyView> view,
         const torch::lazy::BackendDevice& device)
        : view(std::move(view)), device(device), unique_id(GetNextTensorId()) {}
    Data(at::Tensor tensor_data, const torch::lazy::BackendDevice& device)
        : tensor_data(std::move(tensor_data)),
          device(device),
          unique_id(GetNextTensorId()) {}

    ~Data();

    torch::lazy::BackendDataPtr handle;
    torch::lazy::Value ir_value;
    std::shared_ptr<torch::lazy::LazyView> view;
    c10::optional<at::Tensor> tensor_data;
    const torch::lazy::BackendDevice device;
    const int64_t unique_id = 0;
    size_t generation = 1;
  };

  static LazyTensor Create(const at::Tensor& tensor, const torch::lazy::BackendDevice& device);
  static LazyTensor Create(torch::lazy::Value ir_value,
                           const torch::lazy::BackendDevice& device);
  static LazyTensor Create(torch::lazy::BackendDataPtr handle);
  static LazyTensor Create(std::shared_ptr<Data> data);

  // Creates an empty/null tensor.
  LazyTensor() = default;

  bool is_null() const { return data_ptr() == nullptr; }
  operator bool() const { return !is_null(); }

  size_t generation() const { return data()->generation; }

  LazyTensor alias() const { return LazyTensor(data_ptr()); }

  int64_t size(int64_t dim) const;

  at::Tensor ToTensor(bool detached);

  void ShallowCopyTo(LazyTensor* dest) const;

  // Assigns the tensor value to the lazy tensor.
  void SetTensor(at::Tensor tensor);

  void UpdateFromTensor(at::Tensor tensor, bool sync);
  void UpdateFromTensorOut(at::Tensor tensor);
  void UpdateFromTensorOut(const LazyTensor& tensor);

  Data* data() const;

  at::ScalarType dtype() const;

  torch::lazy::MaybeRef<torch::lazy::Shape> shape() const;

  const torch::lazy::BackendDevice& GetDevice() const;
  int64_t GetUniqueId() const;

  // Retrieves an opaque ID of the alias object upon which the tensor's view is
  // rooted, or 0 if this tensor is not a view.
  std::ptrdiff_t GetViewAliasId() const;

  // Fetches the data behind the tensor. If the tensor has a graph defining
  // its current value, executes the graph and fetches the data result.
  torch::lazy::BackendDataPtr GetDataHandle();

  // Fetches the current value of the data, which can be missing (nullptr)
  // in case the tensor has a graph defining its current value,
  torch::lazy::BackendDataPtr CurrentDataHandle() const;

  void SetDataHandle(torch::lazy::BackendDataPtr handle);
  void SetDataHandle(torch::lazy::BackendDataPtr handle, bool sync);

  // Retrieves the current IR Node, or nullptr in case no active IR Node is
  // available.
  torch::lazy::Value CurrentIrValue() const;

  // Retrieves the IR Node representing this LazyTensor. One will be created if
  // missing. Note that although this is a const API, it actually changes the
  // internal state ofthe object.
  torch::lazy::Value GetIrValue() const;

  void SetIrValue(torch::lazy::Value ir_value);
  void SetInPlaceIrValue(torch::lazy::Value ir_value);

  void SetSubView(torch::lazy::ViewInfo view_info) const;

  c10::optional<at::Tensor> CurrentTensorData() const;

  std::vector<LazyTensor> MakeOutputTensors(torch::lazy::NodePtr node) const;

  LazyTensor CreateViewTensor(torch::lazy::ViewInfo view_info) const;
  LazyTensor CopyTensorToDevice(const torch::lazy::BackendDevice& device);

  void ModifyCurrentView(torch::lazy::ViewInfo view_info) const;

  // Applies the queue of operations in preparation for using the data.
  void ApplyPendingGraph();
  void AssignIrValue(torch::lazy::Value ir_value) const;

 private:
  LazyTensor(const at::Tensor& tensor, const torch::lazy::BackendDevice& device);
  LazyTensor(torch::lazy::Value ir_value,
             const torch::lazy::BackendDevice& device);
  LazyTensor(std::shared_ptr<torch::lazy::LazyView> view,
             const torch::lazy::BackendDevice& device);
  LazyTensor(torch::lazy::BackendDataPtr handle);
  LazyTensor(std::shared_ptr<Data> data);

  static LazyTensor Create(std::shared_ptr<torch::lazy::LazyView> view,
                           const torch::lazy::BackendDevice& device);

  std::shared_ptr<Data> data_ptr() const { return data_; }

  
  void SetTensorData(at::Tensor tensor_data);

  torch::lazy::Value CreateTensorNode(torch::lazy::BackendDataPtr data,
                                      bool read_only) const;

  std::tuple<torch::lazy::Value, bool> GetViewUpdate(
      const std::shared_ptr<torch::lazy::LazyView>& view) const;

  std::shared_ptr<torch::lazy::LazyView> UpdateView(
      std::shared_ptr<torch::lazy::LazyView> view,
      torch::lazy::Value ir_value) const;

  std::shared_ptr<torch::lazy::LazyView> CreateView(
      torch::lazy::ViewInfo view_info) const;

  // We build a graph accumulating operations, but at a given point we
  // need to force a rendering, otherwise the graph can grow without control.
  // Think:
  //   for i in range(0, 100000):
  //     a = a + b
  void TryLimitGraphSize();

  torch::lazy::Value GetIrValueForTensor(const at::Tensor& tensor,
                                         const torch::lazy::BackendDevice& device) const;

  static int64_t GetNextTensorId();

  std::shared_ptr<Data> data_;
};

// Utils to convert at::Tensor to LazyTensor, and vice versa.
// Section 1: at::Tensor => LazyTensor.
// Extracts the LazyTensor out of an at::Tensor. Returns a null LazyTensor
// if the tensor is not a lazy tensor.
LazyTensor TryGetLtcTensor(const at::Tensor& tensor);

// Extracts the LazyTensor out of an at::Tensor. Throws an exception
// if the tensor is not a lazy tensor.
LazyTensor GetLtcTensor(const at::Tensor& tensor);

// Same as above, applied to a list of tensors.
std::vector<LazyTensor> GetLtcTensors(c10::ArrayRef<at::Tensor> tensors);

// If tensor is a lazy tensor type, returns the LazyTensor embedded within it,
// otherwise creates a new lazy tensor type with tensor as data.
LazyTensor GetOrCreateLtcTensor(const c10::optional<at::Tensor>& tensor,
                                const torch::lazy::BackendDevice& device);

LazyTensor GetLtcTensorOrCreateForWrappedNumber(const at::Tensor& tensor, const torch::lazy::BackendDevice& device);

// Section 2: LazyTensor => at::Tensor.
// Creates an ATen tensor from an LazyTensor.
at::Tensor CreateAtenFromLtcTensor(const LazyTensor& ltc_tensor);
at::Tensor CreateAtenFromLtcTensor(LazyTensor&& ltc_tensor);

template <size_t... Indices>
auto TupleAtenFromLtcTensorsImpl(const std::vector<LazyTensor>& tensors, std::index_sequence<Indices...>) {
    return std::make_tuple(CreateAtenFromLtcTensor(tensors[Indices])...);
}

template <size_t N>
auto TupleAtenFromLtcTensors(const std::vector<LazyTensor>& tensors) {
    return TupleAtenFromLtcTensorsImpl(tensors, std::make_index_sequence<N>{});
}

}  // namespace torch_lazy_tensors
