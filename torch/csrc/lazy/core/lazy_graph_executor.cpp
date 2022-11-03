#include <torch/csrc/lazy/core/lazy_graph_executor.h>

#include <ATen/ScalarOps.h>
#include <c10/util/Logging.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/core/ir_dump_util.h>
#include <torch/csrc/lazy/core/ir_util.h>
#include <torch/csrc/lazy/core/tensor_util.h>
#include <torch/csrc/lazy/core/unique.h>

#include <torch/csrc/lazy/core/debug_util.h>
#include <torch/csrc/lazy/core/ir_builder.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/ops/arithmetic_ir_ops.h>
#include <torch/csrc/lazy/core/thread_pool.h>

#include <ATen/ScalarOps.h>

namespace torch {
namespace lazy {
namespace {

struct TlsData {
  void Reset() {
    trim_counter = 0;
  }

  size_t trim_counter = 0;
};

thread_local TlsData g_tls_data;

bool TensorCompare(const at::Tensor& t1, const at::Tensor& t2) {
  if (t1.scalar_type() != t2.scalar_type() || t1.sizes() != t2.sizes()) {
    return false;
  }
  // PyTorch currently has an issue comparing tensors which have NaN values in
  // it. The compare is not deterministic. So we do memory compare here until
  // the PyTorch equal() API is fixed.
  at::Tensor contiguous_t1 = t1.contiguous();
  at::Tensor contiguous_t2 = t2.contiguous();
  return std::memcmp(
             contiguous_t1.data_ptr(),
             contiguous_t2.data_ptr(),
             contiguous_t1.numel() * contiguous_t1.itemsize()) == 0;
}

// Locking:
// We perform two kinds of operations of tensors, synchronous and asynchronous.
// The ApplyPendingGraph() are synchronous, as we need the device data result
// immediately. Before the synchronous operations can start, they need to wait
// that the pending asynchronous operations have completed.
// Synchronous operations do not hold device locks, since they are strictly
// sequential, dictated by the PyTorch execution order.
// The SyncTensorsGraph() is asynchronous, and returns immediately after having
// scheduled the asynchronous operation. While executing, the asynchronous
// operations will hold locks on all the participating devices (in most common
// cases there will be only one device).
// Since asynchronous operations capture device locks, only one asynchronous
// operation can execute at the same time, on a given device. Tensor operations
// which send data to device do not need to hold any device locks while doing
// so. Only operations which _use_ device data (computations, and transfer from
// server) need to wait for asynchronous operations to complete (barrier).

class DeviceLocker {
 public:
  explicit DeviceLocker(BackendDevice device) : device_(std::move(device)) {}

  const BackendDevice& device() const {
    return device_;
  }

  void Lock() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !locked_; });
    CheckResetException();
    locked_ = true;
  }

  void Unlock(std::exception_ptr exptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    locked_ = false;
    exptr_ = std::move(exptr);
    cv_.notify_all();
  }

  void Barrier() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !locked_; });
    cv_.notify_all();
    CheckResetException();
  }

 private:
  void CheckResetException() {
    std::exception_ptr exptr = std::move(exptr_);
    exptr_ = nullptr;
    if (exptr != nullptr) {
      std::rethrow_exception(exptr);
    }
  }

  BackendDevice device_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool locked_ = false;
  std::exception_ptr exptr_;
};

class DeviceLockerArena {
 public:
  static DeviceLockerArena* Get() {
    static DeviceLockerArena* arena = new DeviceLockerArena();
    return arena;
  }

  std::shared_ptr<DeviceLocker> GetLocker(const BackendDevice& device) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = lockers_.find(device);
    if (it == lockers_.end()) {
      it = lockers_.emplace(device, std::make_shared<DeviceLocker>(device))
               .first;
    }
    return it->second;
  }

  void DeviceBarrier(const BackendDevice& device) {
    auto locker = DeviceLockerArena::Get()->GetLocker(device);
    locker->Barrier();
  }

  // Use a set to impose an order on the device locking sequence (ABBA
  // prevention).
  std::vector<ExceptionCleanup> LockDevices(
      const std::set<BackendDevice>& devices) {
    std::vector<ExceptionCleanup> unlocker;
    unlocker.reserve(devices.size());
    for (auto& device : devices) {
      unlocker.emplace_back(LockDevice(device));
    }
    return unlocker;
  }

 private:
  ExceptionCleanup LockDevice(const BackendDevice& device) {
    auto locker = DeviceLockerArena::Get()->GetLocker(device);
    locker->Lock();
    return ExceptionCleanup(
        [locker = std::move(locker)](ExceptionCleanup::StatusType status) {
          locker->Unlock(std::move(status));
        });
  }

  std::mutex mutex_;
  std::map<BackendDevice, std::shared_ptr<DeviceLocker>> lockers_;
};

class DataCacheArena {
 public:
  static DataCacheArena* Get() {
    static DataCacheArena* arena =
        new DataCacheArena(FLAGS_torch_lazy_device_data_cache_size);
    return arena;
  }

  explicit DataCacheArena(size_t max_cache_size)
      : max_cache_size_(max_cache_size) {}

  BackendDataPtr GetDeviceData(
      const at::Tensor& tensor,
      const BackendDevice& device) {
    DataCacheArena::DataCache* cache = Get()->GetDataCache(device);
    ;
    BackendDataPtr device_data = cache->Get(tensor);
    if (device_data == nullptr) {
      at::Tensor tensor_copy = CopyTensor(tensor);
      device_data = TensorToDataHandle(tensor_copy, device);
      cache->Add(std::move(tensor_copy), device_data);
      TORCH_LAZY_COUNTER("DeviceDataCacheMiss", 1);
    }
    return device_data;
  }

  BackendDataPtr GetDeviceData(
      const at::Scalar& value,
      at::ScalarType scalar_type,
      const BackendDevice& device) {
    // Workaround since at::scalar_tensor doesn't support bfloat16 yet.
    at::Tensor t = at::scalar_tensor(
        value,
        at::TensorOptions(
            scalar_type == at::ScalarType::BFloat16 ? at::ScalarType::Float
                                                    : scalar_type));
    if (scalar_type == at::ScalarType::BFloat16) {
      t = t.to(scalar_type);
    }
    return GetDeviceData(t, device);
  }

 private:
  struct TensorHasher {
    size_t operator()(const at::Tensor& tensor) const {
      return HashReduce(
          HashCombine(GetEnumValue(tensor.scalar_type()), TensorHash(tensor)));
    }
  };
  struct TensorComparer {
    bool operator()(const at::Tensor& tensor1, const at::Tensor& tensor2)
        const {
      return TensorCompare(tensor1, tensor2);
    }
  };

  using DataCache =
      Cache<at::Tensor, BackendData, TensorHasher, TensorComparer>;

  DataCache* GetDataCache(const BackendDevice& device) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = device_caches_.find(device);
    if (it == device_caches_.end()) {
      std::unique_ptr<DataCache> cache(new DataCache(max_cache_size_));
      it = device_caches_.emplace(device, std::move(cache)).first;
    }
    return it->second.get();
  }

  size_t max_cache_size_ = 0;
  std::mutex mutex_;
  std::map<BackendDevice, std::unique_ptr<DataCache>> device_caches_;
};

// The DeviceContextArena holds per device live information and statistics,
// among which the lazy tensors which are currently alive in the system. This is
// used to create computation "barriers" in order to flush pending operations
// and ensure the same computations are created during the training loops.
class DeviceContextArena {
  struct DeviceContext {
    std::mutex lock;
    std::map<int64_t, std::weak_ptr<LazyTensor::Data>> tensors_data;
    uint64_t seed = 101;
    uint64_t running_seed = 101;
    Value seed_ir_value;
  };

 public:
  static DeviceContextArena* Get() {
    static DeviceContextArena* arena = new DeviceContextArena();
    return arena;
  }

  void RegisterTensor(std::shared_ptr<LazyTensor::Data> data) {
    DeviceContext* devctx = GetDeviceContext(data->device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    devctx->tensors_data.emplace(data->unique_id, data);
    TORCH_LAZY_COUNTER("CreateLtcTensor", 1);
  }

  void UnregisterTensor(LazyTensor::Data* data) {
    DeviceContext* devctx = GetDeviceContext(data->device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    devctx->tensors_data.erase(data->unique_id);
    TORCH_LAZY_COUNTER("DestroyLtcTensor", 1);
  }

  std::vector<LazyTensorPtr> GetLiveTensors(const BackendDevice* device) {
    std::vector<LazyTensorPtr> tensors;
    auto fn = [&](DeviceContext* devctx) {
      std::lock_guard<std::mutex> lock(devctx->lock);
      for (auto& uid_wptr : devctx->tensors_data) {
        std::shared_ptr<LazyTensor::Data> data = uid_wptr.second.lock();
        if (data != nullptr) {
          tensors.push_back(LazyTensor::Create(std::move(data)));
        }
      }
    };
    ForAllDeviceContexts(fn, device);
    return tensors;
  }

  Value GetRngSeed(const BackendDevice& device) {
    static const at::ScalarType kSeedType = at::ScalarType::Long;
    static const uint64_t kSeedMul = 214013;
    static const uint64_t kSeedAdd = 2531011;
    DeviceContext* devctx = GetDeviceContext(device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    if (!devctx->seed_ir_value) {
      devctx->seed_ir_value =
          IrValueFromScalar(MakeIntScalar(devctx->seed), kSeedType, device);
    }
    // Keep the running seed as scalar as well, so we can return it directly
    // without executing graphs.
    devctx->running_seed = kSeedAdd + kSeedMul * devctx->running_seed;
    // Compose new seeds from the root seed, to avoid creating too many
    // computation parameters which might overflow the device capacity.
    Value k = MakeScalar(MakeIntScalar(kSeedMul), kSeedType);
    Value b = MakeScalar(MakeIntScalar(kSeedAdd), kSeedType);
    devctx->seed_ir_value = b + k * devctx->seed_ir_value;
    return devctx->seed_ir_value;
  }

  uint64_t GetRunningSeed(const BackendDevice& device) {
    DeviceContext* devctx = GetDeviceContext(device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    return devctx->running_seed;
  }

  void SetRngSeed(const BackendDevice& device, uint64_t seed) {
    DeviceContext* devctx = GetDeviceContext(device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    devctx->seed = seed;
    devctx->running_seed = devctx->seed;
    devctx->seed_ir_value = Value();
  }

  void MarkStep(const BackendDevice& device) {
    DeviceContext* devctx = GetDeviceContext(device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    devctx->seed = 1012031 + devctx->seed * 7012063;
    devctx->running_seed = devctx->seed;
    devctx->seed_ir_value = Value();
  }

  std::vector<BackendDevice> GetActiveDevices() {
    std::vector<BackendDevice> active_devices;
    std::lock_guard<std::mutex> lock(lock_);
    active_devices.reserve(device_contexts_.size());
    for (auto& device_contexts : device_contexts_) {
      active_devices.push_back(device_contexts.first);
    }
    return active_devices;
  }

 private:
  std::vector<DeviceContext*> GetAllDeviceContexts() {
    std::vector<DeviceContext*> all_device_contexts;
    std::lock_guard<std::mutex> lock(lock_);
    all_device_contexts.reserve(device_contexts_.size());
    for (auto& device_contexts : device_contexts_) {
      all_device_contexts.push_back(device_contexts.second);
    }
    return all_device_contexts;
  }

  void ForAllDeviceContexts(
      const std::function<void(DeviceContext*)>& fn,
      const BackendDevice* device) {
    if (device == nullptr) {
      for (auto devctx : GetAllDeviceContexts()) {
        fn(devctx);
      }
    } else {
      fn(GetDeviceContext(*device));
    }
  }

  DeviceContext* GetDeviceContext(const BackendDevice& device) {
    std::lock_guard<std::mutex> lock(lock_);
    auto it = device_contexts_.find(device);
    if (it == device_contexts_.end()) {
      it = device_contexts_.emplace(device, new DeviceContext()).first;
    }
    return it->second;
  }

  Value IrValueFromScalar(
      const at::Scalar& value,
      at::ScalarType scalar_type,
      const BackendDevice& device) {
    at::Tensor tensor =
        at::scalar_tensor(value, at::TensorOptions(scalar_type));
    BackendDataPtr device_data = TensorToDataHandle(tensor, device);
    return MakeDeviceData(std::move(device_data));
  }

  std::mutex lock_;
  std::map<BackendDevice, DeviceContext*> device_contexts_;
};

// Return true if no tensor in the list has an underlying IR (leaf or
// operation).
bool TensorsHaveIR(const std::vector<LazyTensorPtr>& tensors) {
  for (const auto& tensor : tensors) {
    if (tensor->CurrentDataHandle() || tensor->CurrentIrValue()) {
      return true;
    }
  }
  return false;
}

std::atomic<LazyGraphExecutor*> lazy_graph_executor_registry;
} // namespace

void LazyGraphExecutor::Register(LazyGraphExecutor* executor) {
  lazy_graph_executor_registry.store(executor);
}
LazyGraphExecutor* LazyGraphExecutor::Get() {
  auto* executor = lazy_graph_executor_registry.load();
  TORCH_CHECK(executor, "Lazy graph executor not registered.");
  return executor;
}

void LazyGraphExecutor::RegisterTensor(std::shared_ptr<LazyTensor::Data> data) {
  DeviceContextArena::Get()->RegisterTensor(data);
}

void LazyGraphExecutor::UnregisterTensor(LazyTensor::Data* data) {
  DeviceContextArena::Get()->UnregisterTensor(data);
}

Value LazyGraphExecutor::GetRngSeed(const BackendDevice& device) {
  return DeviceContextArena::Get()->GetRngSeed(device);
}

uint64_t LazyGraphExecutor::GetRunningSeed(const BackendDevice& device) {
  return DeviceContextArena::Get()->GetRunningSeed(device);
}

void LazyGraphExecutor::SetRngSeed(const BackendDevice& device, uint64_t seed) {
  DeviceContextArena::Get()->SetRngSeed(device, seed);
}

void LazyGraphExecutor::DeviceBarrier(const BackendDevice& device) {
  DeviceLockerArena::Get()->DeviceBarrier(device);
}

BackendDataPtr LazyGraphExecutor::GetDeviceData(
    const at::Tensor& tensor,
    const BackendDevice& device) {
  return DataCacheArena::Get()->GetDeviceData(tensor, device);
}

BackendDataPtr LazyGraphExecutor::GetDeviceData(
    const at::Scalar& value,
    at::ScalarType scalar_type,
    const BackendDevice& device) {
  return DataCacheArena::Get()->GetDeviceData(value, scalar_type, device);
}

std::vector<LazyTensorPtr> LazyGraphExecutor::GetLiveTensors(
    const BackendDevice* device) {
  return DeviceContextArena::Get()->GetLiveTensors(device);
}

void LazyGraphExecutor::SyncLiveTensorsGraph(
    const BackendDevice* device,
    c10::ArrayRef<std::string> devices,
    bool wait) {
  auto tensors = GetLiveTensors(device);
  VLOG(4) << tensors.size() << " live tensors: devices=("
          << c10::Join(", ", devices) << ")";
  SyncTensorsGraph(&tensors, devices, wait, /*sync_ltc_data=*/true);
}

void LazyGraphExecutor::SyncTensorsGraph(
    std::vector<LazyTensorPtr>* tensors,
    c10::ArrayRef<std::string> devices,
    bool wait,
    bool sync_ltc_data) {
  VLOG(4) << "Trying to sync the value of " << tensors->size() << " tensor(s)";
  SyncTensorsConfig config;
  config.sync_ltc_data = sync_ltc_data;

  auto async = SyncTensorsGraphInternal(tensors, devices, config);
  if (FLAGS_torch_lazy_use_thread_pool && wait && async != nullptr) {
    async->mwait.Wait();
  }
}

void LazyGraphExecutor::MarkStep(const BackendDevice& device) {
  TORCH_LAZY_COUNTER("MarkStep", 1);
  DeviceContextArena::Get()->MarkStep(device);
  ScopePusher::ResetScopes();
  g_tls_data.Reset();
  // Move TrieCache's current pointer back to its root
  TrieCache::Get()->ResetCurrent();
}

void LazyGraphExecutor::WaitDeviceOps(c10::ArrayRef<BackendDevice> devices) {
  std::set<BackendDevice> wait_devices;
  if (!devices.empty()) {
    for (auto& device : devices) {
      wait_devices.insert(device);
    }
  } else {
    for (auto& device_str : DeviceContextArena::Get()->GetActiveDevices()) {
      // TODO: Remove the last use of Device(const std::string& device_spec).
      wait_devices.insert(BackendDevice(device_str));
    }
  }
  // The LockDevices() API returns a vector of
  // ExceptionCleanup object, which is going to be freed
  // immediately, turning this operation into a lock barrier.
  // NOLINTNEXTLINE
  DeviceLockerArena::Get()->LockDevices(wait_devices);
}

std::vector<at::Tensor> LazyGraphExecutor::GetTensors(
    std::vector<LazyTensorPtr>* tensors) {
  VLOG(4) << "Trying to get the value of " << tensors->size() << " tensor(s)";
  return GetTensorsFused(tensors);
}

size_t LazyGraphExecutor::IncTrimCounter() {
  return ++g_tls_data.trim_counter;
}

std::string LazyGraphExecutor::DumpBackendComputation(
    const std::vector<LazyTensorPtr>& tensors) {
  std::vector<Value> ir_values;
  for (auto& tensor : tensors) {
    Value ir_value = tensor->CurrentIrValue();
    if (ir_value) {
      ir_values.push_back(std::move(ir_value));
    }
  }
  return !ir_values.empty() ? DumpUtil::ToBackend(ir_values, BackendDevice())
                            : std::string();
}

Value LazyGraphExecutor::GetDeviceDataIrValue(
    const at::Scalar& value,
    c10::ScalarType type,
    const BackendDevice& device) {
  BackendDataPtr data = GetDeviceData(value, type, device);
  data->SetInfo(std::make_shared<DeviceDataInfo>(
      /*tensor_id=*/-1, /*read_only=*/true));
  return MakeDeviceData(std::move(data));
}

Value LazyGraphExecutor::GetIrValueForScalarFromCodegen(
    const at::Scalar& value,
    const BackendDevice& device) {
  if (IsSpecialScalar(value)) {
    return MakeScalar(value, value.type());
  }
  BackendDataPtr data =
      getBackend()->MakeComputationDataFromScalar(value, device);
  data->SetInfo(
      std::make_shared<DeviceDataInfo>(/*tensor_id=*/-1, /*read_only=*/true));
  return MakeDeviceData(std::move(data));
}

Value LazyGraphExecutor::GetIrValueForScalar(
    const at::Scalar& value,
    c10::ScalarType type,
    const BackendDevice& device) {
  if (IsSpecialScalar(value)) {
    return MakeScalar(value, type);
  }
  return GetDeviceDataIrValue(value, type, device);
}

Value LazyGraphExecutor::GetIrValueForScalar(
    const at::Scalar& value,
    const BackendDevice& device) {
  return GetIrValueForScalar(value, value.type(), device);
}

Value LazyGraphExecutor::GetIrValueForExpandedScalar(
    const at::Scalar& value,
    const Shape& shape,
    const BackendDevice& device) {
  c10::ArrayRef<int64_t> dimensions = shape.sizes();
  auto type = shape.scalar_type();
  Value ir_value = GetIrValueForScalar(value, type, device);
  if (!dimensions.empty()) {
    ir_value = MakeExpand(
        ir_value,
        dimensions.vec(),
        /*is_scalar_expand=*/true);
  }
  return ir_value;
}

LazyGraphExecutor::Async::Async(
    SyncTensorCollection* coll,
    std::vector<BackendDataPtr> parameters_data,
    std::vector<BackendDataPtr> tensors_data,
    ComputationCache::TypePtr cached_computation)
    : mwait(1),
      indices(std::move(coll->indices)),
      unlocker(std::move(coll->unlocker)),
      parameters_data(std::move(parameters_data)),
      device(coll->device),
      cached_computation(std::move(cached_computation)),
      tensors_data(std::move(tensors_data)) {}

void LazyGraphExecutor::Async::Wait() {
  mwait.Wait();
  // Accessing other Async members is safe only after MultiWait::Wait()
  // completes.
  ExceptionCleanup::StatusType status;
  for (auto& cleanup : unlocker) {
    const ExceptionCleanup::StatusType& cleanup_status = cleanup.GetStatus();
    if (cleanup_status != nullptr) {
      if (status == nullptr) {
        status = cleanup_status;
      }
      // If we observe the status here, no need to let it propagate to the next
      // device lock operation.
      cleanup.SetStatus(nullptr);
    }
  }
  if (status != nullptr) {
    std::rethrow_exception(status);
  }
}

bool LazyGraphExecutor::ShouldSyncTensor(const LazyTensorPtr tensor) const {
  return tensor->GetIrValue()->op() != ltc_not_supported;
}

LazyGraphExecutor::SyncTensorCollection LazyGraphExecutor::CollectSyncTensors(
    const std::vector<LazyTensorPtr>& tensors,
    const SyncTensorsConfig& config) {
  Unique<BackendDevice> unique_device;
  for (const auto& tensor : tensors) {
    unique_device.set(tensor->GetDevice());
  }
  SyncTensorCollection coll;
  if (!unique_device) {
    return coll;
  }
  if (!config.force_ltc_data && !TensorsHaveIR(tensors)) {
    return coll;
  }

  std::vector<at::Tensor> at_tensors;
  std::vector<BackendDevice> devices;
  std::vector<size_t> at_tensor_index;
  std::unordered_set<int64_t> tensor_ids;
  // The force_ltc_data controls aliasing compilation, so effectively the same
  // graph with on/off force_ltc_data should not match, hash wise.
  coll.hash = MHash(config.force_ltc_data);
  coll.config = config;
  coll.device = *unique_device;
  coll.indices.reserve(tensors.size());

  for (const auto i : c10::irange(tensors.size())) {
    if (tensor_ids.insert(tensors[i]->GetUniqueId()).second &&
        tensors[i]->CurrentDataHandle() == nullptr) {
      Value ir_value = tensors[i]->CurrentIrValue();
      if (ir_value) {
        if (ShouldSyncTensor(tensors[i])) {
          // Add only tensors which need to be synced.
          coll.hash = HashCombine(coll.hash, ir_value.hash());
          coll.indices.push_back(i);
        }
      } else if (config.force_ltc_data) {
        // The tensor only has at::Tensor data. We need to queue it for a
        // device upload.
        c10::optional<at::Tensor> tensor_data = tensors[i]->CurrentTensorData();
        TORCH_CHECK(tensor_data);
        at_tensors.push_back(*tensor_data);
        devices.push_back(tensors[i]->GetDevice());
        at_tensor_index.push_back(i);
      }
    }
  }
  if (!at_tensors.empty()) {
    TORCH_LAZY_COUNTER("SyncTensorsToData", at_tensors.size());
    std::vector<BackendDataPtr> handles =
        CreateTensorsData(at_tensors, devices);
    for (const auto i : c10::irange(handles.size())) {
      // If we are here, it means that the IR Value for the tensor is not
      // present. Also, we uploaded the at::Tensor data to the device, but such
      // data is still valid so we leave it live on the lazy tensor (so that a
      // following ToTensor() does not need to fetch it from device).
      tensors[at_tensor_index[i]]->data()->handle = std::move(handles[i]);
    }
  }
  VLOG(4) << "Tensors graph hash " << HashToString(coll.hash) << " on device "
          << coll.device;
  return coll;
}

std::vector<Value> LazyGraphExecutor::CollectRoots(
    const std::vector<LazyTensorPtr>& tensors,
    c10::ArrayRef<size_t> indices) {
  std::vector<Value> roots;
  roots.reserve(indices.size());
  for (auto index : indices) {
    roots.push_back(tensors.at(index)->CurrentIrValue());
  }
  return roots;
}

std::vector<BackendDataPtr> LazyGraphExecutor::FetchTensorData(
    std::vector<LazyTensorPtr>* tensors,
    const SyncTensorsConfig& config,
    c10::ArrayRef<size_t> indices) {
  std::vector<BackendDataPtr> tensors_data;
  tensors_data.reserve(indices.size());
  for (auto index : indices) {
    LazyTensorPtr& tensor = (*tensors)[index];
    // If the config.force_ltc_data flag is true, the purpose of this tensor
    // sync operation is to truncate the IR graph and materialize device data in
    // place of IR graph, on selected tensors. But since operation will complete
    // asynchronously, if a tensor does not already have device data, we need to
    // install a placeholder. Since at this point we hold a lock on the device
    // where the tensors reside (locks held within the coll structure, and moved
    // into the async variable), any other operation trying to access the
    // tensor's device data will have to wait until the asynchronous operation
    // completes.
    BackendDataPtr handle = tensor->CurrentDataHandle();
    if (handle == nullptr && config.force_ltc_data) {
      const BackendDevice& tensor_device = tensor->GetDevice();
      handle = getBackend()->CreateDataPlaceholder(
          tensor_device, std::move(tensor->shape()));

      tensor->SetDataHandle(handle, config.sync_ltc_data);
    }
    tensors_data.emplace_back(std::move(handle));
  }
  return tensors_data;
}

LazyGraphExecutor::PostOrderData LazyGraphExecutor::RunPostOrder(
    const std::vector<LazyTensorPtr>& tensors,
    SyncTensorCollection* coll) {
  std::vector<Node*> roots;
  roots.reserve(coll->indices.size());
  for (auto index : coll->indices) {
    Value ir_value = tensors.at(index)->CurrentIrValue();
    roots.push_back(ir_value.node.get());
  }
  PostOrderData po_data;
  po_data.post_order = Util::ComputePostOrder(roots, &po_data.emission_map);
  std::unordered_map<BackendData::Handle, size_t> data_handles;
  for (auto node : po_data.post_order) {
    const auto backend_data = getBackend()->GetComputationDataFromNode(node);
    if (backend_data) {
      /* Acceptable race condition: HasValue may return false. This is OK
       * since the conditional barrier is a performance optimization. */
      if (!backend_data->HasValue()) {
        TensorCollectionBarrier(coll);
      }
      BackendData::Handle handle = backend_data->GetHandle();
      auto it = data_handles.find(handle);
      if (it != data_handles.end()) {
        po_data.parameter_sequence.push_back(it->second);
      } else {
        po_data.parameter_sequence.push_back(po_data.parameters_data.size());
        data_handles[handle] = po_data.parameters_data.size();
        po_data.parameters_data.push_back(backend_data);
      }
    }
  }
  return po_data;
}

std::shared_ptr<LazyGraphExecutor::Async> LazyGraphExecutor::TryRunCachedSync(
    std::vector<LazyTensorPtr>* tensors,
    SyncTensorCollection* coll,
    PostOrderData* po_data) {
  ComputationCache::TypePtr cached_computation =
      LookupCachedCompile(coll->hash);
  if (cached_computation == nullptr) {
    return nullptr;
  }
  if (GRAPH_DUMP_ENABLED) {
    auto* comp = cached_computation->computation.get();
    LOG(ERROR) << "Run a cached graph: " << comp->to_string() << std::endl;
  }
  TORCH_LAZY_VALUE_METRIC("TensorsGraphSize", po_data->post_order.size());
  VLOG(5) << "TensorsGraphSize=" << po_data->post_order.size();

  return ScheduleSyncTensorsGraph(
      tensors,
      coll,
      std::move(po_data->parameters_data),
      std::move(cached_computation));
}

LazyGraphExecutor::CompilationResult LazyGraphExecutor::Compile(
    const std::vector<LazyTensorPtr>& tensors,
    c10::ArrayRef<std::string> devices,
    const SyncTensorCollection& coll,
    PostOrderData* po_data) {
  auto lowering_ctx = LoweringContext::Create(
      "SyncTensorsGraph",
      coll.device,
      po_data->post_order,
      std::move(po_data->emission_map));
  for (auto index : coll.indices) {
    Value ir_value = tensors[index]->CurrentIrValue();
    lowering_ctx->AddResult(ir_value);
  }

  ComputationPtr computation = lowering_ctx->Build();
  // If force_ltc_data is true it means that we did a proper sync and are
  // inside a mark step. If GetTensors was called, force_ltc_data will
  // be false meaning we are prematurely evaluating some value.
  computation->in_mark_step = coll.config.force_ltc_data;

  VLOG(3) << "Compiling IR graph hash " << HashToString(coll.hash)
          << " on device " << coll.device << " ...";
  std::vector<ComputationPtr> computations =
      getBackend()->Compile({computation});
  VLOG(3) << "Compiling IR graph hash " << HashToString(coll.hash)
          << " on device " << coll.device << " done!";
  if (computation) {
    // TODO(whc) should computation be allowed null here? (because it is in one
    // case)
    TORCH_CHECK(
        computation->parameters_size() == po_data->parameters_data.size());
  }

  return {
      /*device=*/coll.device,
      /*emitted_nodes=*/lowering_ctx->GetEmittedNodeCount(),
      /*computation=*/std::move(computations.front()),
      /*parameters_data=*/std::move(po_data->parameters_data)};
}

LazyGraphExecutor::ComputationCache* LazyGraphExecutor::GetComputationCache() {
  static ComputationCache* cache =
      new ComputationCache(FLAGS_torch_lazy_compilation_cache_size);
  return cache;
}

LazyGraphExecutor::ComputationCache::TypePtr LazyGraphExecutor::
    LookupCachedCompile(const hash_t& hash) {
  ComputationCache::TypePtr cached_computation =
      GetComputationCache()->Get(hash);
  if (cached_computation == nullptr) {
    TORCH_LAZY_COUNTER("UncachedCompile", 1);
    return nullptr;
  }
  TORCH_LAZY_COUNTER("CachedCompile", 1);
  return cached_computation;
}

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

std::shared_ptr<LazyGraphExecutor::Async> LazyGraphExecutor::
    SyncTensorsGraphInternal(
        std::vector<LazyTensorPtr>* tensors,
        c10::ArrayRef<std::string> devices,
        const SyncTensorsConfig& config) {
  SyncTensorCollection coll = CollectSyncTensors(*tensors, config);
  if (coll.indices.empty()) {
    /* Enure previous execution is complete before exiting this
     * function */
    TensorCollectionBarrier(&coll);
    return nullptr;
  }
  PostOrderData po_data = RunPostOrder(*tensors, &coll);
  DebugUtil::SaveTensorsGraphInfo(
      "ScheduleSyncTensorsGraph", *tensors, &coll.indices);
  coll.hash = HashCombine(coll.hash, Hash(po_data.parameter_sequence));
  VLOG(4) << "Parameter sequence graph hash " << HashToString(coll.hash);
  std::shared_ptr<Async> async = TryRunCachedSync(tensors, &coll, &po_data);
  if (async != nullptr) {
    return async;
  }

  CompilationResult compile_result = Compile(*tensors, devices, coll, &po_data);
  if (GRAPH_DUMP_ENABLED) {
    auto* comp = compile_result.computation.get();
    LOG(ERROR) << "Add a cached computation with hash " << coll.hash
               << std::endl;
    LOG(ERROR) << "Add a graph to cache: " << comp->to_string() << std::endl;
  }

  TORCH_LAZY_VALUE_METRIC("TensorsGraphSize", compile_result.emitted_nodes);
  VLOG(5) << "TensorsGraphSize=" << compile_result.emitted_nodes;

  auto cached_computation = std::make_shared<CachedComputation>(
      std::move(compile_result.computation));
  GetComputationCache()->Add(coll.hash, cached_computation);

  return ScheduleSyncTensorsGraph(
      tensors,
      &coll,
      std::move(compile_result.parameters_data),
      std::move(cached_computation));
}

std::shared_ptr<LazyGraphExecutor::Async> LazyGraphExecutor::
    ScheduleSyncTensorsGraph(
        SyncTensorCollection* coll,
        std::vector<BackendDataPtr> parameters_data,
        std::vector<BackendDataPtr> tensors_data,
        ComputationCache::TypePtr cached_computation) {
  TensorCollectionBarrier(coll);
  std::shared_ptr<Async> async = std::make_shared<Async>(
      coll,
      std::move(parameters_data),
      std::move(tensors_data),
      std::move(cached_computation));

  auto syncfn = [this, async, hash = coll->hash]() {
    // For profiling lazy trace overhead
    if (noop_execution_mode_) {
      return;
    }

    try {
      VLOG(3) << "Executing IR graph hash " << HashToString(hash)
              << " on device " << async->device << " ...";
      auto results = getBackend()->ExecuteComputation(
          async->cached_computation->computation,
          async->parameters_data,
          async->device);
      VLOG(3) << "Executing IR graph hash " << HashToString(hash)
              << " on device " << async->device << " done!";

      TORCH_CHECK(
          async->tensors_data.size() == results.size(),
          "Expected number of outputs does not match TorchScript Stack size: ",
          async->tensors_data.size(),
          " != ",
          results.size());

      for (const auto i : c10::irange(results.size())) {
        if (async->tensors_data[i] != nullptr) {
          async->tensors_data[i]->Assign(*results[i]);
        } else {
          async->tensors_data[i] = std::move(results[i]);
        }
      }
    } catch (...) {
      // There are two paths of discovery of an exception happening on an
      // asynchronous task. One happens if the creator of the asynchronous task
      // explicitly waits for completion, in which case the exception will be
      // thrown from the Wait() API. Re-throwing the exception below makes sure
      // this will be captured by the completer function created below, and
      // surfaced by the Wait() API. But we also need to surface the exception
      // even in case the caller does not wait, and that is accomplished by
      // setting the unlockers status. In that case the exception will be
      // surfaced when the user tries to acquire the device locks the next time.
      // std::exception_ptr exptr = std::current_exception();
      for (auto& unlocker : async->unlocker) {
        std::exception_ptr exptr = std::current_exception();
        unlocker.SetStatus(std::move(exptr));
      }
      throw;
    }
  };

  if (FLAGS_torch_lazy_use_thread_pool) {
    ScheduleIoClosure(async->mwait.Completer(std::move(syncfn)));
  } else {
    syncfn();
  }
  return async;
}

std::shared_ptr<LazyGraphExecutor::Async> LazyGraphExecutor::
    ScheduleSyncTensorsGraph(
        std::vector<LazyTensorPtr>* tensors,
        SyncTensorCollection* coll,
        std::vector<BackendDataPtr> parameters_data,
        ComputationCache::TypePtr cached_computation) {
  auto tensors_data = FetchTensorData(tensors, coll->config, coll->indices);
  return ScheduleSyncTensorsGraph(
      coll,
      std::move(parameters_data),
      std::move(tensors_data),
      std::move(cached_computation));
}

std::vector<at::Tensor> LazyGraphExecutor::GetTensorsFused(
    std::vector<LazyTensorPtr>* tensors) {
  SyncTensorsConfig config;
  config.force_ltc_data = false;
  auto async = SyncTensorsGraphInternal(tensors, {}, config);
  if (FLAGS_torch_lazy_use_thread_pool && async != nullptr) {
    async->mwait.Wait();
  }
  std::vector<BackendDataPtr> tensors_data = GatherTensorsData(
      *tensors,
      async != nullptr ? async->indices : c10::ArrayRef<size_t>(),
      async != nullptr ? async->tensors_data : c10::ArrayRef<BackendDataPtr>());
  return FetchTensors(
      tensors, tensors_data, async != nullptr ? &async->indices : nullptr);
}

// This gets tensors from the backend
// for TS backend, we'd ideally just cut through these layers and
// not need to copy the tensor, just move it

// for XLA backend, a copy is going to have to happen,

// could we replace the 'Data' object with an at::Tensor, which is 'undefined'
// unless a backend attaches a buffer to it?  That way we can have a
// 'PopulateTensor' method on backend, which can either attach an existing
// tensor buffer to the wrapper, or copy data?
std::vector<at::Tensor> LazyGraphExecutor::FetchTensors(
    std::vector<LazyTensorPtr>* tensors,
    c10::ArrayRef<BackendDataPtr> tensors_data,
    const std::vector<size_t>* indices) {
  std::vector<at::Tensor> results;
  size_t literals_index = 0;
  size_t sync_index = 0;
  results.reserve(tensors->size());
  for (const auto i : c10::irange(tensors->size())) {
    if (indices != nullptr && sync_index < indices->size() &&
        i == (*indices)[sync_index]) {
      results.push_back(getBackend()->MakeTensorFromComputationData(
          tensors_data[literals_index], (*tensors)[i]->dtype()));
      ++literals_index;
      ++sync_index;
    } else {
      c10::optional<at::Tensor> tensor_data =
          (*tensors)[i]->CurrentTensorData();
      if (tensor_data) {
        results.push_back(*tensor_data);
      } else {
        TORCH_CHECK(literals_index < tensors_data.size());
        results.push_back(getBackend()->MakeTensorFromComputationData(
            tensors_data[literals_index], (*tensors)[i]->dtype()));
        ++literals_index;
      }
    }
  }
  return results;
}

std::vector<BackendDataPtr> LazyGraphExecutor::GatherTensorsData(
    const std::vector<LazyTensorPtr>& tensors,
    c10::ArrayRef<size_t> indices,
    c10::ArrayRef<BackendDataPtr> tensors_data) {
  std::vector<BackendDataPtr> result_tensors_data;
  std::unordered_map<int64_t, size_t> uid_index_map;
  size_t indices_index = 0;
  for (const auto i : c10::irange(tensors.size())) {
    int64_t tensor_id = tensors[i]->GetUniqueId();
    auto it = uid_index_map.find(tensor_id);
    if (it != uid_index_map.end()) {
      // Current tensor is a duplicate of a previously processed tensor that had
      // an IR Node to sync. Get the data from the tensor_data_map.
      result_tensors_data.push_back(result_tensors_data[it->second]);
    } else if (indices_index < indices.size() && i == indices[indices_index]) {
      // If we are at the current index (it means that the tensor at index
      // 'i' had an IR node to sync), use the data held within the Async
      // object.
      uid_index_map.emplace(tensor_id, result_tensors_data.size());
      result_tensors_data.push_back(tensors_data[indices_index]);
      ++indices_index;
    } else if (!tensors[i]->CurrentTensorData()) {
      BackendDataPtr handle = tensors[i]->CurrentDataHandle();
      TORCH_CHECK(handle != nullptr);
      result_tensors_data.push_back(std::move(handle));
    }
  }
  return result_tensors_data;
}

void LazyGraphExecutor::TensorCollectionBarrier(SyncTensorCollection* coll) {
  static const std::string invalid_device(
      "Unknown0"); /* Temp solution to idetify unassigned devices */
  if (coll->device.toString().compare(invalid_device) == 0 ||
      coll->unlocker.size() > 0) {
    return;
  }
  if (coll) {
    VLOG(4) << "Waiting on device barrier for device " << coll->device
            << " ...";
    {
      TORCH_LAZY_TIMED("DeviceLockWait");
      coll->unlocker = DeviceLockerArena::Get()->LockDevices({coll->device});
    }
    VLOG(4) << "Waiting on device barrier for device " << coll->device
            << " done!";
  }
}

hash_t LazyGraphExecutor::GetGraphHash(
    const std::vector<LazyTensorPtr>& tensors) {
  SyncTensorsConfig config;
  config.sync_ltc_data = false;

  auto coll = CollectSyncTensors(tensors, config);
  auto po_data = RunPostOrder(tensors, &coll);
  coll.hash = HashCombine(coll.hash, Hash(po_data.parameter_sequence));
  return coll.hash;
}

} // namespace lazy
} // namespace torch
