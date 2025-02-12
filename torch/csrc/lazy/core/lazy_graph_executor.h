#pragma once

#include <c10/util/ArrayRef.h>
#include <torch/csrc/lazy/backend/lowering_context.h>
#include <torch/csrc/lazy/core/cache.h>
#include <torch/csrc/lazy/core/ir_util.h>
#include <torch/csrc/lazy/core/multi_wait.h>
#include <torch/csrc/lazy/core/tensor.h>
#include <torch/csrc/lazy/core/util.h>

namespace torch::lazy {

class TORCH_API LazyGraphExecutor {
 public:
  struct DeviceDataInfo : public BackendData::Info {
    DeviceDataInfo(int64_t tensor_id, bool read_only)
        : tensor_id(tensor_id), read_only(read_only) {}

    int64_t tensor_id = 0;
    bool read_only = false;
  };

  // Register a lazy graph executor instance that can be retrieved using Get()
  static void Register(LazyGraphExecutor*);
  static LazyGraphExecutor* Get();

  virtual ~LazyGraphExecutor() = default;

  // Override these methods to perform custom tensor registration and
  // unregistration Note: It is vital that the parent implementations are also
  // called in order for the tensors to show up in the live tensor list
  virtual void RegisterTensor(std::shared_ptr<LazyTensor::Data> data);
  virtual void UnregisterTensor(LazyTensor::Data* data);

  // Seed for random generator.
  // Override to supply your own DeviceContextArena.
  virtual Value GetRngSeed(const BackendDevice& device);
  virtual uint64_t GetRunningSeed(const BackendDevice& device);
  virtual void SetRngSeed(const BackendDevice& device, uint64_t seed);

  void DeviceBarrier(const BackendDevice& device);

  BackendDataPtr GetDeviceData(
      const at::Tensor& tensor,
      const BackendDevice& device);

  BackendDataPtr GetDeviceData(
      const at::Scalar& value,
      at::ScalarType scalar_type,
      const BackendDevice& device);

  // Retrieves the set of lazy tensors which are currently live in the system,
  // for the given device. If device is nullptr, the live tensors for all
  // devices will be returned. Returned tensors are sorted by device as primary
  // key, and by unique ID as secondary key.
  std::vector<LazyTensorPtr> GetLiveTensors(const BackendDevice* device);

  // Makes sure that any outstanding IR operation accumulated over live tensors,
  // gets turned into device data. If wait is true, the sync operation will be
  // run synchronously. The devices argument, if not empty, tells the devices
  // which should be partecipating into the replicated computation.
  virtual void SyncLiveTensorsGraph(
      const BackendDevice* device,
      c10::ArrayRef<std::string> devices,
      bool wait);

  // Applies all the pending IR operations queued over the input tensors. All
  // the tensors must be on the same device. If wait is true, the sync operation
  // will be run synchronously. The devices argument, if not empty, tells the
  // devices which should be partecipating into the replicated computation.
  void SyncTensorsGraph(
      std::vector<LazyTensorPtr>* tensors,
      c10::ArrayRef<std::string> devices,
      bool wait,
      bool sync_ltc_data);

  // Marks an execution step, which allows the tensor framework to understand
  // the computation boundaries.
  // Override to supply your own DeviceContextArena.
  virtual void MarkStep(const BackendDevice& device);

  // Waits for all the outstanding operations on all the supplied devices.
  // If devices is empty, the wait will happen for all local devices.
  void WaitDeviceOps(c10::ArrayRef<BackendDevice> devices);

  // Retrieves the PyTorch CPU tensors behind the lazy tensors IR operations.
  // All the tensors must be on the same device.
  std::vector<at::Tensor> GetTensors(std::vector<LazyTensorPtr>* tensors);

  size_t IncTrimCounter() const;

  // Dumps the backend specific text of the computation accumulated in the graph
  // which is attached the tensors.
  std::string DumpBackendComputation(const std::vector<LazyTensorPtr>& tensors);

  Value GetDeviceDataIrValue(
      const at::Scalar& value,
      c10::ScalarType type,
      const BackendDevice& device);
  Value GetIrValueForScalar(
      const at::Scalar& value,
      c10::ScalarType type,
      const BackendDevice& device);
  Value GetIrValueForScalar(
      const at::Scalar& value,
      const BackendDevice& device);

  // TODO: even though this API is currently used **only** in codegen to
  // generate real scalar IR values vs scalar tensors, we would like to
  // use it in other cases where `GetIrValueForXXXScalar` is used, as well
  // In order to do that, we need to untangle the cases where we don't need
  // `expand` and where we don't expect a scalar tensor
  Value GetIrValueForScalarFromCodegen(
      const at::Scalar& value,
      const BackendDevice& device);
  Value GetIrValueForExpandedScalar(
      const at::Scalar& value,
      const Shape& shape,
      const BackendDevice& device);

  struct CachedComputation {
    explicit CachedComputation(ComputationPtr computation)
        : computation(std::move(computation)) {}

    ComputationPtr computation;
  };

  using ComputationCache = Cache<hash_t, CachedComputation, HashReducer>;

  ComputationCache* GetComputationCache();

  hash_t GetGraphHash(const std::vector<LazyTensorPtr>& tensors);

  // Clear the computation cache.
  void ClearComputationCache();
  // Remove a specific computation cache entry from its hash.
  void RemoveFromComputationCache(const hash_t& hash);

 protected:
  // TODO(alanwaketan): Revisit if all of them need to be accessible to
  // derived classes.

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
    hash_t hash;
    std::vector<ExceptionCleanup> unlocker;
    BackendDevice device;
  };

  struct PostOrderData {
    std::vector<const Node*> post_order;
    Util::EmissionMap emission_map;
    std::vector<BackendDataPtr> parameters_data;
    std::vector<size_t> parameter_sequence;
  };

  // Locking:
  // We perform two kinds of operations of tensors, synchronous and
  // asynchronous. The ApplyPendingGraph() are synchronous, as we need the
  // device data result immediately. Before the synchronous operations can
  // start, they need to wait that the pending asynchronous operations have
  // completed. Synchronous operations do not hold device locks, since they are
  // strictly sequential, dictated by the PyTorch execution order. The
  // SyncTensorsGraph() is asynchronous, and returns immediately after having
  // scheduled the asynchronous operation. While executing, the asynchronous
  // operations will hold locks on all the participating devices (in most common
  // cases there will be only one device).
  // Since asynchronous operations capture device locks, only one asynchronous
  // operation can execute at the same time, on a given device. Tensor
  // operations which send data to device do not need to hold any device locks
  // while doing so. Only operations which _use_ device data (computations, and
  // transfer from server) need to wait for asynchronous operations to complete
  // (barrier).

  class DeviceLocker {
   public:
    explicit DeviceLocker(BackendDevice device) : device_(std::move(device)) {}

    const BackendDevice& device() const {
      return device_;
    }

    void Lock();
    void Unlock(std::exception_ptr exptr);
    void Barrier();

   private:
    void CheckResetException();

    BackendDevice device_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool locked_ = false;
    std::exception_ptr exptr_;
  };

  class DeviceLockerArena {
   public:
    static DeviceLockerArena* Get();

    std::shared_ptr<DeviceLocker> GetLocker(const BackendDevice& device);

    void DeviceBarrier(const BackendDevice& device);

    // Use a set to impose an order on the device locking sequence (ABBA
    // prevention).
    std::vector<ExceptionCleanup> LockDevices(
        const std::set<BackendDevice>& devices);

   private:
    ExceptionCleanup LockDevice(const BackendDevice& device);

    std::mutex mutex_;
    std::map<BackendDevice, std::shared_ptr<DeviceLocker>> lockers_;
  };

  class DataCacheArena {
   public:
    static DataCacheArena* Get();

    BackendDataPtr GetDeviceData(
        const at::Tensor& tensor,
        const BackendDevice& device);

    BackendDataPtr GetDeviceData(
        const at::Scalar& value,
        at::ScalarType scalar_type,
        const BackendDevice& device);

   private:
    struct TensorHasher {
      size_t operator()(const at::Tensor& tensor) const;
    };
    struct TensorComparer {
      bool operator()(const at::Tensor& tensor1, const at::Tensor& tensor2)
          const;
    };

    explicit DataCacheArena(size_t max_cache_size);

    using DataCache =
        Cache<at::Tensor, BackendData, TensorHasher, TensorComparer>;

    DataCache* GetDataCache(const BackendDevice& device);

    size_t max_cache_size_ = 0;
    std::mutex mutex_;
    std::map<BackendDevice, std::unique_ptr<DataCache>> device_caches_;
  };

  // The DeviceContextArena holds per device live information and statistics,
  // among which the lazy tensors which are currently alive in the system. This
  // is used to create computation "barriers" in order to flush pending
  // operations and ensure the same computations are created during the training
  // loops.
  // TODO(alanwaketan): Add a registry such that we don't need to make all
  // related methods virtual.
  class DeviceContextArena {
   protected:
    struct DeviceContext {
      std::mutex lock;
      std::map<int64_t, std::weak_ptr<LazyTensor::Data>> tensors_data;
      uint64_t seed = 101;
      uint64_t running_seed = 101;
      Value seed_ir_value;
    };

   public:
    static DeviceContextArena* Get();
    virtual ~DeviceContextArena() = default;

    void RegisterTensor(std::shared_ptr<LazyTensor::Data> data);
    void UnregisterTensor(LazyTensor::Data* data);

    std::vector<LazyTensorPtr> GetLiveTensors(const BackendDevice* device);

    // Overriding it allow derived class to use their own IRs for Value.
    virtual Value GetRngSeed(const BackendDevice& device);
    uint64_t GetRunningSeed(const BackendDevice& device);
    void SetRngSeed(const BackendDevice& device, uint64_t seed);

    void MarkStep(const BackendDevice& device);

    std::vector<BackendDevice> GetActiveDevices();

   protected:
    DeviceContext* GetDeviceContext(const BackendDevice& device);

    void ForAllDeviceContexts(
        const std::function<void(DeviceContext*)>& fn,
        const BackendDevice* device);

    // Overriding it allow derived class to use their own conversions.
    virtual Value IrValueFromScalar(
        const at::Scalar& value,
        at::ScalarType scalar_type,
        const BackendDevice& device);

   private:
    std::vector<DeviceContext*> GetAllDeviceContexts();

    std::mutex lock_;
    std::map<BackendDevice, DeviceContext*> device_contexts_;
  };

  struct Async {
    Async(
        SyncTensorCollection* coll,
        std::vector<BackendDataPtr> parameters_data,
        std::vector<BackendDataPtr> tensors_data,
        ComputationCache::TypePtr cached_computation);
    virtual ~Async() = default;

    void Wait();

    MultiWait mwait;
    std::vector<size_t> indices;
    std::vector<ExceptionCleanup> unlocker;
    std::vector<BackendDataPtr> parameters_data;
    BackendDevice device;
    ComputationCache::TypePtr cached_computation;
    std::vector<BackendDataPtr> tensors_data;
  };

  void ResetTrimCounter() const;

  // Waits for this SyncTensorCollection's device barrier and acquire the lock.
  virtual void TensorCollectionBarrier(SyncTensorCollection* coll);

  // One can override to insert your own profiler.
  virtual PostOrderData RunPostOrder(
      const std::vector<Value>& ir_values,
      SyncTensorCollection* coll);

 private:
  struct CompilationResult {
    BackendDevice device;
    size_t emitted_nodes = 0;
    ComputationPtr computation;
    std::vector<BackendDataPtr> parameters_data;
  };

  virtual bool ShouldSyncTensor(const LazyTensorPtr& tensor) const;

  SyncTensorCollection CollectSyncTensors(
      const std::vector<LazyTensorPtr>& tensors,
      const SyncTensorsConfig& config);

  std::vector<Value> CollectRoots(
      const std::vector<LazyTensorPtr>& tensors,
      c10::ArrayRef<size_t> indices);

  std::vector<BackendDataPtr> SetTensorData(
      std::vector<LazyTensorPtr>* tensors,
      const SyncTensorsConfig& config,
      c10::ArrayRef<size_t> indices,
      const std::vector<torch::lazy::BackendDataPtr>& tensor_data_vec);

  void ExtractIRAndPrepareTensorData(
      std::vector<LazyTensorPtr>* tensors,
      const SyncTensorsConfig& config,
      c10::ArrayRef<size_t> indices,
      std::vector<Value>& ir_values,
      std::vector<BackendDataPtr>& tensor_data_vec);

  std::shared_ptr<Async> TryRunCachedSync(
      std::vector<LazyTensorPtr>* tensors,
      SyncTensorCollection* coll,
      PostOrderData* po_data,
      const std::vector<BackendDataPtr>& tensor_data_vec);

  CompilationResult Compile(
      const std::vector<LazyTensorPtr>& tensors,
      c10::ArrayRef<std::string> devices,
      const SyncTensorCollection& coll,
      PostOrderData* po_data,
      const std::vector<Value>& ir_values);

  ComputationCache::TypePtr LookupCachedCompile(const hash_t& hash);

  std::shared_ptr<Async> SyncTensorsGraphInternal(
      std::vector<LazyTensorPtr>* tensors,
      c10::ArrayRef<std::string> devices,
      const SyncTensorsConfig& config);

  // Schedules the execution of a sync tensors operation in background. The
  // asynchronous operation will hold the device locks by capturing the ones
  // present within the coll structure.
  std::shared_ptr<Async> ScheduleSyncTensorsGraph(
      SyncTensorCollection* coll,
      std::vector<BackendDataPtr> parameters_data,
      std::vector<BackendDataPtr> tensors_data,
      ComputationCache::TypePtr cached_computation);

  std::shared_ptr<Async> ScheduleSyncTensorsGraph(
      std::vector<LazyTensorPtr>* tensors,
      SyncTensorCollection* coll,
      std::vector<BackendDataPtr> parameters_data,
      ComputationCache::TypePtr cached_computation,
      const std::vector<BackendDataPtr>& tensor_data_vec);

  std::vector<at::Tensor> GetTensorsFused(std::vector<LazyTensorPtr>* tensors);

  std::vector<at::Tensor> FetchTensors(
      std::vector<LazyTensorPtr>* tensors,
      c10::ArrayRef<BackendDataPtr> tensors_data,
      const std::vector<size_t>* indices);

  // Gathers the device data for all the input tensors, after an
  // asynchronous operation.
  std::vector<BackendDataPtr> GatherTensorsData(
      const std::vector<LazyTensorPtr>& tensors,
      c10::ArrayRef<size_t> indices,
      c10::ArrayRef<BackendDataPtr> tensors_data);
};

} // namespace torch::lazy
