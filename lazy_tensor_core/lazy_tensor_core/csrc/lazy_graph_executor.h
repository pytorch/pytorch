#pragma once

#include <c10/util/ArrayRef.h>
#include <torch/csrc/lazy/backend/lowering_context.h>
#include <torch/csrc/lazy/core/cache.h>
#include <torch/csrc/lazy/core/ir_util.h>
#include <torch/csrc/lazy/core/multi_wait.h>
#include <torch/csrc/lazy/core/util.h>

#include "lazy_tensor_core/csrc/tensor.h"

namespace torch_lazy_tensors {

class LazyGraphExecutor {
 public:
  struct DeviceDataInfo : public torch::lazy::BackendData::Info {
    DeviceDataInfo(int64_t tensor_id, bool read_only)
        : tensor_id(tensor_id), read_only(read_only) {}

    int64_t tensor_id = 0;
    bool read_only = false;
  };

  static LazyGraphExecutor* Get();

  void RegisterTensor(std::shared_ptr<LazyTensor::Data> data);
  void UnregisterTensor(LazyTensor::Data* data);

  // Seed for random generator
  torch::lazy::Value GetRngSeed(const torch::lazy::BackendDevice& device);
  uint64_t GetRunningSeed(const torch::lazy::BackendDevice& device);
  void SetRngSeed(const torch::lazy::BackendDevice& device, uint64_t seed);

  void DeviceBarrier(const torch::lazy::BackendDevice& device);

  torch::lazy::BackendDataPtr GetDeviceData(
      const at::Tensor& tensor, const torch::lazy::BackendDevice& device);

  torch::lazy::BackendDataPtr GetDeviceData(
      const at::Scalar& value, at::ScalarType scalar_type,
      const torch::lazy::BackendDevice& device);

  // Retrieves the set of lazy tensors which are currently live in the system,
  // for the given device. If device is nullptr, the live tensors for all
  // devices will be returned. Returned tensors are sorted by device as primary
  // key, and by unique ID as secondary key.
  std::vector<LazyTensor> GetLiveTensors(const torch::lazy::BackendDevice* device);

  // Makes sure that any outstanding IR operation accumulated over live tensors,
  // gets turned into device data. If wait is true, the sync operation will be
  // run synchronously. The devices argument, if not empty, tells the devices
  // which should be partecipating into the replicated computation.
  void SyncLiveTensorsGraph(const torch::lazy::BackendDevice* device,
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
  void MarkStep(const torch::lazy::BackendDevice& device);

  // Waits for all the outstanding operations on all the supplied devices.
  // If devices is empty, the wait will happen for all local devices.
  void WaitDeviceOps(c10::ArrayRef<torch::lazy::BackendDevice> devices);

  // Retrieves the PyTorch CPU tensors behind the lazy tensors IR operations.
  // All the tensors must be on the same device.
  std::vector<at::Tensor> GetTensors(std::vector<LazyTensor>* tensors);

  size_t IncTrimCounter();

  // Dumps the backend specific text of the computation accumulated in the graph
  // which is attached the tensors.
  std::string DumpBackendComputation(const std::vector<LazyTensor>& tensors);

  torch::lazy::Value GetDeviceDataIrValue(const at::Scalar& value,
                                          c10::ScalarType type,
                                          const torch::lazy::BackendDevice& device);
  torch::lazy::Value GetIrValueForScalar(const at::Scalar& value,
                                         c10::ScalarType type,
                                         const torch::lazy::BackendDevice& device);
  torch::lazy::Value GetIrValueForScalar(const at::Scalar& value,
                                         const torch::lazy::BackendDevice& device);

  // TODO: even though this API is currently used **only** in codegen to
  // generate real scalar IR values vs scalar tensors, we would like to
  // use it in other cases where `GetIrValueForXXXScalar` is used, as well
  // In order to do that, we need to untangle the cases where we don't need
  // `expand` and where we don't expect a scalar tensor
  torch::lazy::Value GetIrValueForScalarFromCodegen(const at::Scalar& value);
  torch::lazy::Value GetIrValueForExpandedScalar(
      const at::Scalar& value, const torch::lazy::Shape& shape,
      const torch::lazy::BackendDevice& device);

  // Configure the executor treat compile/execute API calls as no-ops
  // for use when profiling lazy trace overheads
  void SetNoOpExecutionMode(bool enable_noop) { noop_execution_mode_ = enable_noop; }

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
    std::vector<torch::lazy::ExceptionCleanup> unlocker;
    torch::lazy::BackendDevice device;
  };

  struct PostOrderData {
    std::vector<torch::lazy::Node*> post_order;
    torch::lazy::Util::EmissionMap emission_map;
    std::vector<torch::lazy::BackendDataPtr> parameters_data;
    std::vector<size_t> parameter_sequence;
  };

  struct CompilationResult {
    torch::lazy::BackendDevice device;
    size_t emitted_nodes = 0;
    torch::lazy::ComputationPtr computation;
    std::vector<torch::lazy::BackendDataPtr> parameters_data;
  };

  struct CachedComputation {
    CachedComputation(torch::lazy::ComputationPtr computation)
        : computation(std::move(computation)) {}

    torch::lazy::ComputationPtr computation;
  };

  using ComputationCache =
      torch::lazy::Cache<torch::lazy::hash_t, CachedComputation,
                         torch::lazy::HashReducer>;

  struct Async {
    Async(SyncTensorCollection* coll,
          std::vector<torch::lazy::BackendDataPtr> parameters_data,
          std::vector<torch::lazy::BackendDataPtr> tensors_data,
          ComputationCache::TypePtr cached_computation);

    void Wait();

    torch::lazy::MultiWait mwait;
    std::vector<size_t> indices;
    std::vector<torch::lazy::ExceptionCleanup> unlocker;
    std::vector<torch::lazy::BackendDataPtr> parameters_data;
    torch::lazy::BackendDevice device;
    ComputationCache::TypePtr cached_computation;
    std::vector<torch::lazy::BackendDataPtr> tensors_data;
  };

  SyncTensorCollection CollectSyncTensors(
      const std::vector<LazyTensor>& tensors, const SyncTensorsConfig& config);

  std::vector<torch::lazy::Value> CollectRoots(
      const std::vector<LazyTensor>& tensors, c10::ArrayRef<size_t> indices);

  std::vector<torch::lazy::BackendDataPtr> FetchTensorData(
      std::vector<LazyTensor>* tensors, const SyncTensorsConfig& config,
      c10::ArrayRef<size_t> indices);


  std::vector<LazyTensor> FetchLazyTensors(
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
                               torch::lazy::LoweringContext* lowering_ctx);

  std::shared_ptr<Async> SyncTensorsGraphInternal(
      std::vector<LazyTensor>* tensors, c10::ArrayRef<std::string> devices,
      const SyncTensorsConfig& config);

  // Schedules the execution of a sync tensors operation in background. The
  // asynchronous operation will hold the device locks by capturing the ones
  // present within the coll structure.
  std::shared_ptr<Async> ScheduleSyncTensorsGraph(
      SyncTensorCollection* coll,
      const std::vector<LazyTensor>& lazy_tensors,
      std::vector<torch::lazy::BackendDataPtr> parameters_data,
      std::vector<torch::lazy::BackendDataPtr> tensors_data,
      ComputationCache::TypePtr cached_computation);

  std::shared_ptr<Async> ScheduleSyncTensorsGraph2(
      std::vector<LazyTensor>* tensors, SyncTensorCollection* coll,
      std::vector<torch::lazy::BackendDataPtr> parameters_data,
      ComputationCache::TypePtr cached_computation);

  std::vector<at::Tensor> GetTensorsFused(std::vector<LazyTensor>* tensors);

  std::vector<at::Tensor> FetchTensors(
      std::vector<LazyTensor>* tensors,
      c10::ArrayRef<torch::lazy::BackendDataPtr> tensors_data,
      const std::vector<size_t>* indices);

  // Gathers the device data for all the input tensors, after an
  // asynchronous operation.
  std::vector<torch::lazy::BackendDataPtr> GatherTensorsData(
      const std::vector<LazyTensor>& tensors, c10::ArrayRef<size_t> indices,
      c10::ArrayRef<torch::lazy::BackendDataPtr> tensors_data);

  bool noop_execution_mode_ = false;
};

}  // namespace torch_lazy_tensors
