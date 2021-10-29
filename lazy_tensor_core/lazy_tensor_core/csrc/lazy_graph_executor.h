#pragma once

#include "lazy_tensor_core/csrc/ir_util.h"
#include "lazy_tensor_core/csrc/lowering_context.h"
#include "lazy_tensor_core/csrc/tensor.h"
#include "lazy_tensors/computation_client/async_task.h"
#include "lazy_tensors/computation_client/cache.h"
#include "lazy_tensors/computation_client/multi_wait.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {

class LazyGraphExecutor {
 public:
  struct DeviceDataInfo : public compiler::Data::Info {
    DeviceDataInfo(int64_t tensor_id, bool read_only)
        : tensor_id(tensor_id), read_only(read_only) {}

    int64_t tensor_id = 0;
    bool read_only = false;
  };

  static LazyGraphExecutor* Get();

  void RegisterTensor(std::shared_ptr<LazyTensor::Data> data);
  void UnregisterTensor(LazyTensor::Data* data);

  // Seed for random generator
  torch::lazy::Value GetRngSeed(const Device& device);
  uint64_t GetRunningSeed(const Device& device);
  void SetRngSeed(const Device& device, uint64_t seed);

  void DeviceBarrier(const Device& device);

  compiler::DataPtr GetDeviceData(
      const at::Tensor& tensor, const Device& device);

  compiler::DataPtr GetDeviceData(
      const at::Scalar& value, at::ScalarType scalar_type,
      const Device& device);

  // Retrieves the set of lazy tensors which are currently live in the system,
  // for the given device. If device is nullptr, the live tensors for all
  // devices will be returned. Returned tensors are sorted by device as primary
  // key, and by unique ID as secondary key.
  std::vector<LazyTensor> GetLiveTensors(const Device* device);

  // Makes sure that any outstanding IR operation accumulated over live tensors,
  // gets turned into device data. If wait is true, the sync operation will be
  // run synchronously. The devices argument, if not empty, tells the devices
  // which should be partecipating into the replicated computation.
  void SyncLiveTensorsGraph(const Device* device,
                            c10::ArrayRef<std::string> devices, bool wait);

  // Applies all the pending IR operations queued over the input tensors. All
  // the tensors must be on the same device. If wait is true, the sync operation
  // will be run synchronously. The devices argument, if not empty, tells the
  // devices which should be partecipating into the replicated computation.
  void SyncTensorsGraph(std::vector<LazyTensor>* tensors,
                        c10::ArrayRef<std::string> devices, bool wait,
                        bool sync_ltc_data);

  // Marks an execution step, which allows the tensor framework to understand
  // the computation boundaries.
  void MarkStep(const Device& device);

  // Waits for all the outstanding operations on all the supplied devices.
  // If devices is empty, the wait will happen for all local devices.
  void WaitDeviceOps(c10::ArrayRef<std::string> devices);

  // Retrieves the PyTorch CPU tensors behind the lazy tensors IR operations.
  // All the tensors must be on the same device.
  std::vector<at::Tensor> GetTensors(std::vector<LazyTensor>* tensors);

  size_t IncTrimCounter();

  // Dumps the backend specific text of the computation accumulated in the graph
  // which is attached the tensors.
  std::string DumpBackendComputation(const std::vector<LazyTensor>& tensors);

  torch::lazy::Value GetDeviceDataIrValue(const at::Scalar& value,
                                          c10::ScalarType type,
                                          const Device& device);
  torch::lazy::Value GetIrValueForScalar(const at::Scalar& value,
                                         c10::ScalarType type,
                                         const Device& device);
  torch::lazy::Value GetIrValueForScalar(const at::Scalar& value,
                                         const Device& device);
  torch::lazy::Value GetIrValueForScalar(const at::Scalar& value,
                                         c10::ScalarType type,
                                         c10::ArrayRef<int64_t> dimensions,
                                         const Device& device);
  torch::lazy::Value GetIrValueForScalar(const at::Scalar& value,
                                         const lazy_tensors::Shape& shape,
                                         const Device& device);
  torch::lazy::Value GetIrValueForScalar(
      const at::Scalar& value, const lazy_tensors::Shape& shape,
      c10::optional<at::ScalarType> logical_element_type, const Device& device);

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
    std::vector<torch::lazy::Node*> post_order;
    ir::Util::EmissionMap emission_map;
    std::vector<compiler::DataPtr> parameters_data;
    std::vector<size_t> parameter_sequence;
  };

  struct CompilationResult {
    Device device;
    size_t emitted_nodes = 0;
    std::shared_ptr<compiler::Computation> computation;
    std::vector<compiler::DataPtr> parameters_data;
  };

  struct CachedComputation {
    CachedComputation(
        std::shared_ptr<compiler::Computation>
            computation)
        : computation(std::move(computation)) {}

    std::shared_ptr<compiler::Computation> computation;
  };

  using ComputationCache =
      lazy_tensors::util::Cache<torch::lazy::hash_t, CachedComputation,
                                torch::lazy::HashReducer>;

  struct Async {
    Async(SyncTensorCollection* coll,
          std::vector<compiler::DataPtr> parameters_data,
          std::vector<compiler::DataPtr> tensors_data,
          ComputationCache::TypePtr cached_computation);

    void Wait();

    lazy_tensors::util::MultiWait mwait;
    std::vector<size_t> indices;
    std::vector<lazy_tensors::util::ExceptionCleanup> unlocker;
    std::vector<compiler::DataPtr> parameters_data;
    std::string device;
    ComputationCache::TypePtr cached_computation;
    std::vector<compiler::DataPtr> tensors_data;
  };

  SyncTensorCollection CollectSyncTensors(
      const std::vector<LazyTensor>& tensors, const SyncTensorsConfig& config);

  std::vector<torch::lazy::Value> CollectRoots(
      const std::vector<LazyTensor>& tensors, c10::ArrayRef<size_t> indices);

  std::vector<compiler::DataPtr> FetchTensorData(
      std::vector<LazyTensor>* tensors, const SyncTensorsConfig& config,
      c10::ArrayRef<size_t> indices);

  PostOrderData RunPostOrder(const std::vector<LazyTensor>& tensors,
                             c10::ArrayRef<size_t> indices);
  std::shared_ptr<Async> TryRunCachedSync(std::vector<LazyTensor>* tensors,
                                          SyncTensorCollection* coll,
                                          PostOrderData* po_data);

  CompilationResult Compile(const std::vector<LazyTensor>& tensors,
                            c10::ArrayRef<std::string> devices,
                            const SyncTensorCollection& coll,
                            PostOrderData* po_data);

  ComputationCache* GetComputationCache();

  ComputationCache::TypePtr LookupCachedCompile(
      const std::vector<LazyTensor>& tensors, const torch::lazy::hash_t& hash);

  void BuildInputOutputAliases(const std::vector<LazyTensor>& tensors,
                               c10::ArrayRef<size_t> indices,
                               ir::LoweringContext* lowering_ctx);

  // Runs an asynchronous syn operation using the op-by-op executor.
  using OpByOpAsync = lazy_tensors::util::AsyncTask<int>;
  OpByOpAsync SyncTensorsGraphOpByOp(std::vector<LazyTensor>* tensors,
                                     c10::ArrayRef<std::string> devices,
                                     const SyncTensorsConfig& config);

  std::shared_ptr<Async> SyncTensorsGraphInternal(
      std::vector<LazyTensor>* tensors, c10::ArrayRef<std::string> devices,
      const SyncTensorsConfig& config);

  // Schedules the execution of a sync tensors operation in background. The
  // asynchronous operation will hold the device locks by capturing the ones
  // present within the coll structure.
  std::shared_ptr<Async> ScheduleSyncTensorsGraph(
      SyncTensorCollection* coll,
      std::vector<compiler::DataPtr> parameters_data,
      std::vector<compiler::DataPtr> tensors_data,
      ComputationCache::TypePtr cached_computation);

  std::shared_ptr<Async> ScheduleSyncTensorsGraph(
      std::vector<LazyTensor>* tensors, SyncTensorCollection* coll,
      std::vector<compiler::DataPtr> parameters_data,
      std::string device, ComputationCache::TypePtr cached_computation);

  // Implementation of the GetTensors() API using the op-by-op executor.
  std::vector<at::Tensor> GetTensorsOpByOp(std::vector<LazyTensor>* tensors);
  std::vector<at::Tensor> GetTensorsFused(std::vector<LazyTensor>* tensors);

  std::vector<at::Tensor> FetchTensors(
      std::vector<LazyTensor>* tensors,
      c10::ArrayRef<compiler::DataPtr> tensors_data,
      const std::vector<size_t>* indices);

  // Gathers the device data for all the input tensors, after an
  // asynchronous operation.
  std::vector<compiler::DataPtr> GatherTensorsData(
      const std::vector<LazyTensor>& tensors, c10::ArrayRef<size_t> indices,
      c10::ArrayRef<compiler::DataPtr> tensors_data);
};

}  // namespace torch_lazy_tensors
