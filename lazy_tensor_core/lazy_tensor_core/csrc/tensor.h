#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "lazy_tensor_core/csrc/cross_replica_reduces.h"
#include "lazy_tensor_core/csrc/device.h"
#include "lazy_tensor_core/csrc/ir_util.h"
#include "lazy_tensor_core/csrc/lowering_context.h"
#include "lazy_tensor_core/csrc/view.h"
#include "lazy_tensors/computation_client/async_task.h"
#include "lazy_tensors/computation_client/cache.h"
#include "lazy_tensors/computation_client/computation_client.h"
#include "lazy_tensors/computation_client/multi_wait.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/status.h"
#include "lazy_tensors/types.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/lazy/core/ir.h"

namespace torch_lazy_tensors {

class LazyTensor {
 public:
  // This is the core lazy tensor data structure where all the tensor data is
  // held. The lazy tensor is nothing more than a shared pointer to a Data
  // object.
  struct Data {
    Data(lazy_tensors::ComputationClient::DataPtr handle, const Device& device,
         c10::optional<at::ScalarType> logical_element_type)
        : handle(std::move(handle)),
          logical_element_type(logical_element_type),
          device(device),
          unique_id(GetNextTensorId()) {}
    Data(ir::Value ir_value, const Device& device,
         c10::optional<at::ScalarType> logical_element_type)
        : ir_value(std::move(ir_value)),
          logical_element_type(logical_element_type),
          device(device),
          unique_id(GetNextTensorId()) {}
    Data(std::shared_ptr<View> view, const Device& device,
         c10::optional<at::ScalarType> logical_element_type)
        : view(std::move(view)),
          logical_element_type(logical_element_type),
          device(device),
          unique_id(GetNextTensorId()) {}
    Data(at::Tensor tensor_data, const Device& device)
        : logical_element_type(tensor_data.scalar_type()),
          tensor_data(std::move(tensor_data)),
          device(device),
          unique_id(GetNextTensorId()) {}

    ~Data();

    lazy_tensors::ComputationClient::DataPtr handle;
    ir::Value ir_value;
    std::shared_ptr<View> view;
    c10::optional<at::ScalarType> logical_element_type;
    c10::optional<at::Tensor> tensor_data;
    const Device device;
    const lazy_tensors::int64 unique_id = 0;
    size_t generation = 1;
  };

  static LazyTensor Create(const at::Tensor& tensor, const Device& device);
  static LazyTensor Create(
      lazy_tensors::ComputationClient::DataPtr handle,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static LazyTensor Create(
      ir::Value ir_value, const Device& device,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static LazyTensor Create(std::shared_ptr<Data> data);

  // TODO(whc) just a hack for now to get codegen to compile... need to refactor
  // Create a new lazy tensor with the same metadata of the input tensor (with
  // possible overrides), and the new IR value.
  LazyTensor CreateFrom(ir::Value ir_value) const;
  LazyTensor CreateFrom(ir::Value ir_value, const Device& device) const;
  LazyTensor CreateFrom(ir::Value ir_value,
                        at::ScalarType logical_element_type) const;
  LazyTensor CreateFrom(
      ir::Value ir_value,
      c10::optional<at::ScalarType> logical_element_type_opt) const;
  LazyTensor CreateFrom(ir::Value ir_value, const Device& device,
                        at::ScalarType logical_element_type) const;

  // Creates an empty/null tensor.
  LazyTensor() = default;

  bool is_null() const { return data_ptr() == nullptr; }

  size_t generation() const { return data()->generation; }

  LazyTensor alias() const { return LazyTensor(data_ptr()); }

  lazy_tensors::int64 size(lazy_tensors::int64 dim) const;

  at::Tensor ToTensor(bool detached);

  void ShallowCopyTo(LazyTensor* dest) const;

  // Assigns the tensor value to the lazy tensor.
  void SetTensor(at::Tensor tensor);

  void UpdateFromTensor(at::Tensor tensor, bool sync);
  void UpdateFromTensorOut(at::Tensor tensor);
  void UpdateFromTensorOut(const LazyTensor& tensor);

  Data* data() const;

  at::ScalarType dtype() const;
  c10::optional<at::ScalarType> dtype_optional() const;

  // Set logical_element_type which is visible to upstream PyTorch.
  void SetScalarType(c10::optional<at::ScalarType> logical_element_type);

  lazy_tensors::util::MaybeRef<lazy_tensors::Shape> shape() const;
  lazy_tensors::Shape shape_with_layout() const;

  const Device& GetDevice() const;
  lazy_tensors::int64 GetUniqueId() const;

  // Retrieves an opaque ID of the alias object upon which the tensor's view is
  // rooted, or 0 if this tensor is not a view.
  std::ptrdiff_t GetViewAliasId() const;

  // Fetches the data behind the tensor. If the tensor has a graph defining
  // its current value, executes the graph and fetches the data result.
  lazy_tensors::ComputationClient::DataPtr GetDataHandle();

  // Fetches the current value of the data, which can be missing (nullptr)
  // in case the tensor has a graph defining its current value,
  lazy_tensors::ComputationClient::DataPtr CurrentDataHandle() const;

  void SetDataHandle(lazy_tensors::ComputationClient::DataPtr handle);
  void SetDataHandle(lazy_tensors::ComputationClient::DataPtr handle,
                     bool sync);

  // Retrieves the current IR Node, or nullptr in case no active IR Node is
  // available.
  ir::Value CurrentIrValue() const;

  // Retrieves the IR Node representing this LazyTensor. One will be created if
  // missing. Note that although this is a const API, it actually changes the
  // internal state ofthe object.
  ir::Value GetIrValue() const;

  void SetIrValue(ir::Value ir_value);
  void SetInPlaceIrValue(ir::Value ir_value);

  void SetSubView(ViewInfo view_info) const;

  c10::optional<at::Tensor> CurrentTensorData() const;

  std::vector<LazyTensor> MakeOutputTensors(ir::NodePtr node) const;

  LazyTensor CreateViewTensor(ViewInfo view_info) const;
  LazyTensor CopyTensorToDevice(const Device& device);

  void ModifyCurrentView(ViewInfo view_info) const;

  // Applies the queue of operations in preparation for using the data.
  void ApplyPendingGraph();

  static ir::Value GetDeviceDataIrValue(const at::Scalar& value,
                                        lazy_tensors::PrimitiveType type,
                                        const Device& device);
  static ir::Value GetIrValueForScalar(const at::Scalar& value,
                                       lazy_tensors::PrimitiveType type,
                                       const Device& device);
  static ir::Value GetIrValueForScalar(const at::Scalar& value,
                                       const Device& device);
  static ir::Value GetIrValueForScalar(
      const at::Scalar& value, lazy_tensors::PrimitiveType type,
      lazy_tensors::Span<const lazy_tensors::int64> dimensions,
      const Device& device);
  static ir::Value GetIrValueForScalar(const at::Scalar& value,
                                       const lazy_tensors::Shape& shape,
                                       const Device& device);
  static ir::Value GetIrValueForScalar(
      const at::Scalar& value, const lazy_tensors::Shape& shape,
      c10::optional<at::ScalarType> logical_element_type, const Device& device);

  // Dumps the backend specific text of the computation accumulated in the graph
  // which is attached the tensors.
  static std::string DumpBackendComputation(
      const std::vector<LazyTensor>& tensors);

  // Operation which creates lazy tensors out of PyTorch CPU tensors by batching
  // the requests to the computation servers.
  static std::vector<LazyTensor> CreateTensors(
      const std::vector<at::Tensor>& tensors,
      const std::vector<std::string>& devices);

 private:
  LazyTensor(const at::Tensor& tensor, const Device& device);
  LazyTensor(lazy_tensors::ComputationClient::DataPtr handle,
             c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  LazyTensor(ir::Value ir_value, const Device& device,
             c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  LazyTensor(std::shared_ptr<View> view, const Device& device,
             c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  LazyTensor(std::shared_ptr<Data> data);

  static LazyTensor Create(
      std::shared_ptr<View> view, const Device& device,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  std::shared_ptr<Data> data_ptr() const { return data_; }

  void AssignIrValue(ir::Value ir_value) const;

  void SetTensorData(at::Tensor tensor_data);

  ir::Value CreateTensorNode(lazy_tensors::ComputationClient::DataPtr data,
                             bool read_only) const;

  View::IrNode GetViewUpdate(const std::shared_ptr<View>& view) const;

  std::shared_ptr<View> UpdateView(std::shared_ptr<View> view,
                                   ir::Value ir_value) const;

  std::shared_ptr<View> CreateView(ViewInfo view_info) const;

  ir::Value MaybeCastIrValue(
      ir::Value ir_value, const Device& device,
      c10::optional<at::ScalarType> logical_element_type) const;

  // We build a graph accumulating operations, but at a given point we
  // need to force a rendering, otherwise the graph can grow without control.
  // Think:
  //   for i in range(0, 100000):
  //     a = a + b
  void TryLimitGraphSize();

  ir::Value GetIrValueForTensor(const at::Tensor& tensor,
                                const Device& device) const;


  static lazy_tensors::int64 GetNextTensorId();

  std::shared_ptr<Data> data_;
};

}  // namespace torch_lazy_tensors
