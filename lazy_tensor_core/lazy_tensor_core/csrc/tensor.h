#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "lazy_tensor_core/csrc/cross_replica_reduces.h"
#include "lazy_tensor_core/csrc/device.h"
#include "lazy_tensor_core/csrc/ir.h"
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

 public:
  static LazyTensor Create(const at::Tensor& tensor, const Device& device);
  static LazyTensor Create(
      lazy_tensors::ComputationClient::DataPtr handle,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static LazyTensor Create(
      ir::Value ir_value, const Device& device,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static LazyTensor Create(std::shared_ptr<Data> data);

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

  // Retrieves the set of lazy tensors which are currently live in the system,
  // for the given device. If device is nullptr, the live tensors for all
  // devices will be returned. Returned tensors are sorted by device as primary
  // key, and by unique ID as secondary key.
  static std::vector<LazyTensor> GetLiveTensors(const Device* device);

  // Applies all the pending IR operations queued over the input tensors. All
  // the tensors must be on the same device. If wait is true, the sync operation
  // will be run synchronously. The devices argument, if not empty, tells the
  // devices which should be partecipating into the replicated computation.
  static void SyncTensorsGraph(std::vector<LazyTensor>* tensors,
                               lazy_tensors::Span<const std::string> devices,
                               bool wait, bool sync_ltc_data);

  // Makes sure that any outstanding IR operation accumulated over live tensors,
  // gets turned into device data. If wait is true, the sync operation will be
  // run synchronously. The devices argument, if not empty, tells the devices
  // which should be partecipating into the replicated computation.
  static void SyncLiveTensorsGraph(
      const Device* device, lazy_tensors::Span<const std::string> devices,
      bool wait);

  // Marks an execution step, which allows the tensor framework to understand
  // the computation boundaries.
  static void MarkStep(const Device& device);

  // Waits for all the outstanding operations on all the supplied devices.
  // If devices is empty, the wait will happen for all local devices.
  static void WaitDeviceOps(lazy_tensors::Span<const std::string> devices);

  // Retrieves the PyTorch CPU tensors behind the lazy tensors IR operations.
  // All the tensors must be on the same device.
  static std::vector<at::Tensor> GetTensors(std::vector<LazyTensor>* tensors);

  // Operation which creates lazy tensors out of PyTorch CPU tensors by batching
  // the requests to the computation servers.
  static std::vector<LazyTensor> CreateTensors(
      const std::vector<at::Tensor>& tensors,
      const std::vector<std::string>& devices);

 private:
  struct SyncTensorsConfig {
    // Whether we want to force data on the target tensors (hence trimming
    // the IR graph above them).
    bool force_ltc_data = true;
    // Whether when setting the data, the other properties of the tensor
    // state should be reset.
    bool sync_ltc_data = true;
  };

  struct SyncTensorCollection {
    SyncTensorCollection() : hash(0) {}

    SyncTensorsConfig config;
    std::vector<size_t> indices;
    torch::lazy::hash_t hash;
    std::vector<lazy_tensors::util::ExceptionCleanup> unlocker;
    Device device;
  };

  struct PostOrderData {
    std::vector<const ir::Node*> post_order;
    ir::Util::EmissionMap emission_map;
    std::vector<lazy_tensors::ComputationClient::DataPtr> parameters_data;
    std::vector<size_t> parameter_sequence;
  };

  struct CompilationResult {
    Device device;
    size_t emitted_nodes = 0;
    std::shared_ptr<lazy_tensors::ComputationClient::Computation> computation;
    std::vector<lazy_tensors::ComputationClient::DataPtr> parameters_data;
  };

  struct CachedComputation {
    CachedComputation(
        std::shared_ptr<lazy_tensors::ComputationClient::Computation>
            computation)
        : computation(std::move(computation)) {}

    std::shared_ptr<lazy_tensors::ComputationClient::Computation> computation;
  };

  using ComputationCache =
      lazy_tensors::util::Cache<torch::lazy::hash_t, CachedComputation,
                                torch::lazy::HashReducer>;

  struct Async {
    Async(SyncTensorCollection* coll,
          std::vector<lazy_tensors::ComputationClient::DataPtr> parameters_data,
          std::vector<lazy_tensors::ComputationClient::DataPtr> tensors_data,
          ComputationCache::TypePtr cached_computation);

    void Wait();

    lazy_tensors::util::MultiWait mwait;
    std::vector<size_t> indices;
    std::vector<lazy_tensors::util::ExceptionCleanup> unlocker;
    std::vector<lazy_tensors::ComputationClient::DataPtr> parameters_data;
    std::string device;
    ComputationCache::TypePtr cached_computation;
    std::vector<lazy_tensors::ComputationClient::DataPtr> tensors_data;
  };

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

  void SetDataHandle(lazy_tensors::ComputationClient::DataPtr handle,
                     bool sync);

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

public:
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

private:
  // We build a graph accumulating operations, but at a given point we
  // need to force a rendering, otherwise the graph can grow without control.
  // Think:
  //   for i in range(0, 100000):
  //     a = a + b
  void TryLimitGraphSize();

  ir::Value GetIrValueForTensor(const at::Tensor& tensor,
                                const Device& device) const;

  static ComputationCache* GetComputationCache();

  static SyncTensorCollection CollectSyncTensors(
      const std::vector<LazyTensor>& tensors, const SyncTensorsConfig& config);

  // Implementation of the GetTensors() API using the op-by-op executor.
  static std::vector<at::Tensor> GetTensorsOpByOp(
      std::vector<LazyTensor>* tensors);

  static std::vector<at::Tensor> GetTensorsFused(
      std::vector<LazyTensor>* tensors);

  // Runs an asynchronous syn operation using the op-by-op executor.
  using OpByOpAsync = lazy_tensors::util::AsyncTask<int>;
  static OpByOpAsync SyncTensorsGraphOpByOp(
      std::vector<LazyTensor>* tensors,
      lazy_tensors::Span<const std::string> devices,
      const SyncTensorsConfig& config);

  // Gathers the device data for all the input tensors, after an
  // asynchronous operation.
  static std::vector<lazy_tensors::ComputationClient::DataPtr>
  GatherTensorsData(
      const std::vector<LazyTensor>& tensors,
      lazy_tensors::Span<const size_t> indices,
      lazy_tensors::Span<const lazy_tensors::ComputationClient::DataPtr>
          tensors_data);

  static std::vector<ir::Value> CollectRoots(
      const std::vector<LazyTensor>& tensors,
      lazy_tensors::Span<const size_t> indices);

  static std::vector<lazy_tensors::ComputationClient::DataPtr> FetchTensorData(
      std::vector<LazyTensor>* tensors, const SyncTensorsConfig& config,
      lazy_tensors::Span<const size_t> indices);

  static std::vector<at::Tensor> FetchTensors(
      std::vector<LazyTensor>* tensors,
      lazy_tensors::Span<const lazy_tensors::ComputationClient::DataPtr>
          tensors_data,
      const std::vector<size_t>* indices);

  // Schedules the execution of a sync tensors operation in background. The
  // asynchronous operation will hold the device locks by capturing the ones
  // present within the coll structure.
  static std::shared_ptr<LazyTensor::Async> ScheduleSyncTensorsGraph(
      SyncTensorCollection* coll,
      std::vector<lazy_tensors::ComputationClient::DataPtr> parameters_data,
      std::vector<lazy_tensors::ComputationClient::DataPtr> tensors_data,
      ComputationCache::TypePtr cached_computation);

  static std::shared_ptr<Async> ScheduleSyncTensorsGraph(
      std::vector<LazyTensor>* tensors, SyncTensorCollection* coll,
      std::vector<lazy_tensors::ComputationClient::DataPtr> parameters_data,
      std::string device, ComputationCache::TypePtr cached_computation);

  static PostOrderData RunPostOrder(const std::vector<LazyTensor>& tensors,
                                    lazy_tensors::Span<const size_t> indices);

  static ComputationCache::TypePtr LookupCachedCompile(
      const std::vector<LazyTensor>& tensors, const torch::lazy::hash_t& hash);

  static std::shared_ptr<Async> TryRunCachedSync(
      std::vector<LazyTensor>* tensors, SyncTensorCollection* coll,
      PostOrderData* po_data);

  static void BuildInputOutputAliases(const std::vector<LazyTensor>& tensors,
                                      lazy_tensors::Span<const size_t> indices,
                                      ir::LoweringContext* lowering_ctx);

  static CompilationResult Compile(
      const std::vector<LazyTensor>& tensors,
      lazy_tensors::Span<const std::string> devices,
      const SyncTensorCollection& coll, PostOrderData* po_data);

  static std::shared_ptr<Async> SyncTensorsGraphInternal(
      std::vector<LazyTensor>* tensors,
      lazy_tensors::Span<const std::string> devices,
      const SyncTensorsConfig& config);

  static lazy_tensors::int64 GetNextTensorId();

  std::shared_ptr<Data> data_;
};

}  // namespace torch_lazy_tensors
