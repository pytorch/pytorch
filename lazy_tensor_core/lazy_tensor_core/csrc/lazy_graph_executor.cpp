#include "lazy_tensor_core/csrc/lazy_graph_executor.h"

#include <c10/util/Logging.h>
#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/core/ir_dump_util.h>
#include <torch/csrc/lazy/core/ir_util.h>
#include <torch/csrc/lazy/core/tensor_util.h>
#include <torch/csrc/lazy/core/unique.h>

// TODO: DebugUtil will be upstreamed after LazyTensor is in.
//#include "lazy_tensor_core/csrc/debug_util.h"
#include <torch/csrc/lazy/core/internal_ops/arithmetic_ir_ops.h>
#include <torch/csrc/lazy/core/internal_ops/device_data.h>
#include <torch/csrc/lazy/core/internal_ops/expand.h>
#include <torch/csrc/lazy/core/internal_ops/scalar.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/thread_pool.h>

#include "lazy_tensors/computation_client/sys_util.h"

namespace torch_lazy_tensors {
namespace {

using namespace torch::lazy;

struct TlsData {
  void Reset() { trim_counter = 0; }

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
  return std::memcmp(contiguous_t1.data_ptr(), contiguous_t2.data_ptr(),
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
  explicit DeviceLocker(torch::lazy::BackendDevice device) : device_(std::move(device)) {}

  const torch::lazy::BackendDevice& device() const { return device_; }

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

  torch::lazy::BackendDevice device_;
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

  std::shared_ptr<DeviceLocker> GetLocker(const torch::lazy::BackendDevice& device) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = lockers_.find(device);
    if (it == lockers_.end()) {
      it = lockers_.emplace(device, std::make_shared<DeviceLocker>(device))
               .first;
    }
    return it->second;
  }

  void DeviceBarrier(const torch::lazy::BackendDevice& device) {
    auto locker = DeviceLockerArena::Get()->GetLocker(device);
    locker->Barrier();
  }

  // Use a set to impose an order on the device locking sequence (ABBA
  // prevention).
  std::vector<torch::lazy::ExceptionCleanup> LockDevices(
      const std::set<torch::lazy::BackendDevice>& devices) {
    std::vector<torch::lazy::ExceptionCleanup> unlocker;
    unlocker.reserve(devices.size());
    for (auto& device : devices) {
      unlocker.emplace_back(LockDevice(device));
    }
    return unlocker;
  }

 private:
  torch::lazy::ExceptionCleanup LockDevice(const torch::lazy::BackendDevice& device) {
    auto locker = DeviceLockerArena::Get()->GetLocker(device);
    locker->Lock();
    return torch::lazy::ExceptionCleanup(
        [locker = std::move(locker)](
            torch::lazy::ExceptionCleanup::StatusType status) {
          locker->Unlock(std::move(status));
        });
  }

  std::mutex mutex_;
  std::map<torch::lazy::BackendDevice, std::shared_ptr<DeviceLocker>> lockers_;
};

class DataCacheArena {
 public:
  static DataCacheArena* Get() {
    static const size_t kMaxCacheSize =
        lazy_tensors::sys_util::GetEnvInt("DEVDATA_CACHE_SIZE", 128);
    static DataCacheArena* arena = new DataCacheArena(kMaxCacheSize);
    return arena;
  }

  explicit DataCacheArena(size_t max_cache_size)
      : max_cache_size_(max_cache_size) {}

  torch::lazy::BackendDataPtr GetDeviceData(
      const at::Tensor& tensor, const torch::lazy::BackendDevice& device) {
    DataCacheArena::DataCache* cache = Get()->GetDataCache(device);
    ;
    torch::lazy::BackendDataPtr device_data = cache->Get(tensor);
    if (device_data == nullptr) {
      at::Tensor tensor_copy = torch::lazy::CopyTensor(tensor);
      device_data = TensorToDataHandle(tensor_copy, device);
      cache->Add(std::move(tensor_copy), device_data);
      TORCH_LAZY_COUNTER("DeviceDataCacheMiss", 1);
    }
    return device_data;
  }

  torch::lazy::BackendDataPtr GetDeviceData(
      const at::Scalar& value, at::ScalarType scalar_type,
      const torch::lazy::BackendDevice& device) {
    // Workaround since at::scalar_tensor doesn't support bfloat16 yet.
    at::Tensor t = at::scalar_tensor(
        value, at::TensorOptions(scalar_type == at::ScalarType::BFloat16
                                     ? at::ScalarType::Float
                                     : scalar_type));
    if (scalar_type == at::ScalarType::BFloat16) t = t.to(scalar_type);
    return GetDeviceData(t, device);
  }

 private:
  struct TensorHasher {
    size_t operator()(const at::Tensor& tensor) const {
      return torch::lazy::HashReduce(torch::lazy::HashCombine(
          torch::lazy::GetEnumValue(tensor.scalar_type()),
          torch::lazy::TensorHash(tensor)));
    };
  };
  struct TensorComparer {
    bool operator()(const at::Tensor& tensor1,
                    const at::Tensor& tensor2) const {
      return TensorCompare(tensor1, tensor2);
    }
  };

  using DataCache = torch::lazy::Cache<at::Tensor, torch::lazy::BackendData,
                                       TensorHasher, TensorComparer>;

  DataCache* GetDataCache(const torch::lazy::BackendDevice& device) {
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
  std::map<torch::lazy::BackendDevice, std::unique_ptr<DataCache>> device_caches_;
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
    torch::lazy::Value seed_ir_value;
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

  std::vector<LazyTensor> GetLiveTensors(const torch::lazy::BackendDevice* device) {
    std::vector<LazyTensor> tensors;
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

  torch::lazy::Value GetRngSeed(const torch::lazy::BackendDevice& device) {
    static const at::ScalarType kSeedType = at::ScalarType::Long;
    static const uint64_t kSeedMul = 214013;
    static const uint64_t kSeedAdd = 2531011;
    DeviceContext* devctx = GetDeviceContext(device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    if (!devctx->seed_ir_value) {
      devctx->seed_ir_value = IrValueFromScalar(
          torch::lazy::MakeIntScalar(devctx->seed), kSeedType, device);
    }
    // Keep the running seed as scalar as well, so we can return it directly
    // without executing graphs.
    devctx->running_seed = kSeedAdd + kSeedMul * devctx->running_seed;
    // Compose new seeds from the root seed, to avoid creating too many
    // computation parameters which might overflow the device capacity.
    torch::lazy::Value k = torch::lazy::MakeNode<torch::lazy::Scalar>(
        torch::lazy::MakeIntScalar(kSeedMul), kSeedType);
    torch::lazy::Value b = torch::lazy::MakeNode<torch::lazy::Scalar>(
        torch::lazy::MakeIntScalar(kSeedAdd), kSeedType);
    devctx->seed_ir_value = b + k * devctx->seed_ir_value;
    return devctx->seed_ir_value;
  }

  uint64_t GetRunningSeed(const torch::lazy::BackendDevice& device) {
    DeviceContext* devctx = GetDeviceContext(device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    return devctx->running_seed;
  }

  void SetRngSeed(const torch::lazy::BackendDevice& device, uint64_t seed) {
    DeviceContext* devctx = GetDeviceContext(device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    devctx->seed = seed;
    devctx->running_seed = devctx->seed;
    devctx->seed_ir_value = torch::lazy::Value();
  }

  void MarkStep(const torch::lazy::BackendDevice& device) {
    DeviceContext* devctx = GetDeviceContext(device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    devctx->seed = 1012031 + devctx->seed * 7012063;
    devctx->running_seed = devctx->seed;
    devctx->seed_ir_value = torch::lazy::Value();
  }

  std::vector<torch::lazy::BackendDevice> GetActiveDevices() {
    std::vector<torch::lazy::BackendDevice> active_devices;
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

  void ForAllDeviceContexts(const std::function<void(DeviceContext*)>& fn,
                            const torch::lazy::BackendDevice* device) {
    if (device == nullptr) {
      for (auto devctx : GetAllDeviceContexts()) {
        fn(devctx);
      }
    } else {
      fn(GetDeviceContext(*device));
    }
  }

  DeviceContext* GetDeviceContext(const torch::lazy::BackendDevice& device) {
    std::lock_guard<std::mutex> lock(lock_);
    auto it = device_contexts_.find(device);
    if (it == device_contexts_.end()) {
      it = device_contexts_.emplace(device, new DeviceContext()).first;
    }
    return it->second;
  }

  torch::lazy::Value IrValueFromScalar(const at::Scalar& value,
                                       at::ScalarType scalar_type,
                                       const torch::lazy::BackendDevice& device) {
    at::Tensor tensor =
        at::scalar_tensor(value, at::TensorOptions(scalar_type));
    torch::lazy::BackendDataPtr device_data =
        TensorToDataHandle(tensor, device);
    return torch::lazy::MakeNode<torch::lazy::DeviceData>(std::move(device_data));
  }

  std::mutex lock_;
  std::map<torch::lazy::BackendDevice, DeviceContext*> device_contexts_;
};

bool ShouldSyncIrValue(const torch::lazy::Value& ir_value) {
  return ir_value->op() != torch::lazy::ltc_not_supported;
}

// Return true if no tensor in the list has an underlying IR (leaf or
// operation).
bool TensorsHaveIR(const std::vector<LazyTensor>& tensors) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i].CurrentDataHandle() || tensors[i].CurrentIrValue()) {
      return true;
    }
  }
  return false;
}

}  // namespace

LazyGraphExecutor* LazyGraphExecutor::Get() {
  static LazyGraphExecutor* executor = new LazyGraphExecutor();
  return executor;
}

void LazyGraphExecutor::RegisterTensor(std::shared_ptr<LazyTensor::Data> data) {
  DeviceContextArena::Get()->RegisterTensor(data);
}

void LazyGraphExecutor::UnregisterTensor(LazyTensor::Data* data) {
  DeviceContextArena::Get()->UnregisterTensor(data);
}

torch::lazy::Value LazyGraphExecutor::GetRngSeed(const torch::lazy::BackendDevice& device) {
  return DeviceContextArena::Get()->GetRngSeed(device);
}

uint64_t LazyGraphExecutor::GetRunningSeed(const torch::lazy::BackendDevice& device) {
  return DeviceContextArena::Get()->GetRunningSeed(device);
}

void LazyGraphExecutor::SetRngSeed(const torch::lazy::BackendDevice& device, uint64_t seed) {
  DeviceContextArena::Get()->SetRngSeed(device, seed);
}

void LazyGraphExecutor::DeviceBarrier(const torch::lazy::BackendDevice& device) {
  DeviceLockerArena::Get()->DeviceBarrier(device);
}

torch::lazy::BackendDataPtr LazyGraphExecutor::GetDeviceData(
    const at::Tensor& tensor, const torch::lazy::BackendDevice& device) {
  return DataCacheArena::Get()->GetDeviceData(tensor, device);
}

torch::lazy::BackendDataPtr LazyGraphExecutor::GetDeviceData(
    const at::Scalar& value, at::ScalarType scalar_type,
    const torch::lazy::BackendDevice& device) {
  return DataCacheArena::Get()->GetDeviceData(value, scalar_type, device);
}

std::vector<LazyTensor> LazyGraphExecutor::GetLiveTensors(
    const torch::lazy::BackendDevice* device) {
  return DeviceContextArena::Get()->GetLiveTensors(device);
}

void LazyGraphExecutor::SyncLiveTensorsGraph(const torch::lazy::BackendDevice* device,
                                             c10::ArrayRef<std::string> devices,
                                             bool wait) {
  auto tensors = GetLiveTensors(device);
  VLOG(4) << tensors.size() << " live tensors: devices=("
          << c10::Join(", ", devices) << ")";
  SyncTensorsGraph(&tensors, devices, wait, /*sync_ltc_data=*/true);
}

void LazyGraphExecutor::SyncTensorsGraph(std::vector<LazyTensor>* tensors,
                                         c10::ArrayRef<std::string> devices,
                                         bool wait, bool sync_ltc_data) {
  VLOG(4) << "Trying to sync the value of " << tensors->size() << " tensor(s)";
  SyncTensorsConfig config;
  config.sync_ltc_data = sync_ltc_data;

  auto async = SyncTensorsGraphInternal(tensors, devices, config);
  if (wait && async != nullptr) {
    async->mwait.Wait();
  }
}

void LazyGraphExecutor::MarkStep(const torch::lazy::BackendDevice& device) {
  TORCH_LAZY_COUNTER("MarkStep", 1);
  DeviceContextArena::Get()->MarkStep(device);
  torch::lazy::ScopePusher::ResetScopes();
  g_tls_data.Reset();
}

void LazyGraphExecutor::WaitDeviceOps(c10::ArrayRef<torch::lazy::BackendDevice> devices) {
  std::set<torch::lazy::BackendDevice> wait_devices;
  if (!devices.empty()) {
    for (auto& device : devices) {
      wait_devices.insert(device);
    }
  } else {
    for (auto& device_str : DeviceContextArena::Get()->GetActiveDevices()) {
    // TODO: Remove the last use of Device(const std::string& device_spec).
    wait_devices.insert(torch::lazy::BackendDevice(device_str));
  }
  }
  // The LockDevices() API returns a vector of
  // torch::lazy::ExceptionCleanup object, which is going to be freed
  // immediately, turning this operation into a lock barrier.
  DeviceLockerArena::Get()->LockDevices(wait_devices);
}

std::vector<at::Tensor> LazyGraphExecutor::GetTensors(
    std::vector<LazyTensor>* tensors) {
  VLOG(4) << "Trying to get the value of " << tensors->size() << " tensor(s)";
  return GetTensorsFused(tensors);
}

size_t LazyGraphExecutor::IncTrimCounter() { return ++g_tls_data.trim_counter; }

std::string LazyGraphExecutor::DumpBackendComputation(
    const std::vector<LazyTensor>& tensors) {
  std::vector<torch::lazy::Value> ir_values;
  for (auto& tensor : tensors) {
    torch::lazy::Value ir_value = tensor.CurrentIrValue();
    if (ir_value) {
      ir_values.push_back(std::move(ir_value));
    }
  }
  return !ir_values.empty() ? torch::lazy::DumpUtil::ToBackend(
                                  ir_values, torch::lazy::BackendDevice())
                            : std::string();
}

torch::lazy::Value LazyGraphExecutor::GetDeviceDataIrValue(
    const at::Scalar& value, c10::ScalarType type, const torch::lazy::BackendDevice& device) {
  torch::lazy::BackendDataPtr data = GetDeviceData(value, type, device);
  data->SetInfo(std::make_shared<DeviceDataInfo>(
      /*tensor_id=*/-1, /*read_only=*/true));
  return torch::lazy::MakeNode<torch::lazy::DeviceData>(std::move(data));
}

torch::lazy::Value LazyGraphExecutor::GetIrValueForScalar(
    const at::Scalar& value, c10::ScalarType type, const torch::lazy::BackendDevice& device) {
  if (torch::lazy::IsSpecialScalar(value)) {
    return torch::lazy::MakeNode<torch::lazy::Scalar>(value, type);
  }
  return GetDeviceDataIrValue(value, type, device);
}

torch::lazy::Value LazyGraphExecutor::GetIrValueForScalar(
    const at::Scalar& value, const torch::lazy::BackendDevice& device) {
  return GetIrValueForScalar(value, value.type(), device);
}

torch::lazy::Value LazyGraphExecutor::GetIrValueForScalar(
    const at::Scalar& value, c10::ScalarType type,
    c10::ArrayRef<int64_t> dimensions, const torch::lazy::BackendDevice& device) {
  torch::lazy::Value ir_value = GetIrValueForScalar(value, type, device);
  if (!dimensions.empty()) {
    ir_value = torch::lazy::MakeNode<torch::lazy::Expand>(
        ir_value, dimensions.vec(),
        /*is_scalar_expand=*/true);
  }
  return ir_value;
}

torch::lazy::Value LazyGraphExecutor::GetIrValueForScalar(
    const at::Scalar& value, const torch::lazy::Shape& shape,
    const torch::lazy::BackendDevice& device) {
  return GetIrValueForScalar(value, shape.scalar_type(), shape.sizes(),
                             device);
}

LazyGraphExecutor::Async::Async(
    SyncTensorCollection* coll,
    std::vector<torch::lazy::BackendDataPtr> parameters_data,
    std::vector<torch::lazy::BackendDataPtr> tensors_data,
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
  torch::lazy::ExceptionCleanup::StatusType status;
  for (auto& cleanup : unlocker) {
    const torch::lazy::ExceptionCleanup::StatusType& cleanup_status =
        cleanup.GetStatus();
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

LazyGraphExecutor::SyncTensorCollection LazyGraphExecutor::CollectSyncTensors(
    const std::vector<LazyTensor>& tensors, const SyncTensorsConfig& config) {
  torch::lazy::Unique<torch::lazy::BackendDevice> unique_device;
  for (size_t i = 0; i < tensors.size(); ++i) {
    unique_device.set(tensors[i].GetDevice());
  }
  SyncTensorCollection coll;
  if (!unique_device) {
    return coll;
  }
  if (!config.force_ltc_data && !TensorsHaveIR(tensors)) {
    return coll;
  }

  std::vector<at::Tensor> at_tensors;
  std::vector<torch::lazy::BackendDevice> devices;
  std::vector<size_t> at_tensor_index;
  std::unordered_set<int64_t> tensor_ids;
  // The force_ltc_data controls aliasing compilation, so effectively the same
  // graph with on/off force_ltc_data should not match, hash wise.
  coll.hash = torch::lazy::MHash(config.force_ltc_data);
  coll.config = config;
  coll.device = *unique_device;
  coll.indices.reserve(tensors.size());
  VLOG(4) << "Waiting on device barrier for device " << coll.device << " ...";
  {
    TORCH_LAZY_TIMED("DeviceLockWait");
    coll.unlocker =
        DeviceLockerArena::Get()->LockDevices(unique_device.AsSet());
  }
  VLOG(4) << "Waiting on device barrier for device " << coll.device << " done!";
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensor_ids.insert(tensors[i].GetUniqueId()).second &&
        tensors[i].CurrentDataHandle() == nullptr) {
      torch::lazy::Value ir_value = tensors[i].CurrentIrValue();
      if (ir_value) {
        if (ShouldSyncIrValue(ir_value)) {
          // Add only tensors which need to be synced.
          coll.hash = torch::lazy::HashCombine(coll.hash, ir_value.hash());
          coll.indices.push_back(i);
        }
      } else if (config.force_ltc_data) {
        // The tensor only has at::Tensor data. We need to queue it for a
        // device upload.
        c10::optional<at::Tensor> tensor_data = tensors[i].CurrentTensorData();
        CHECK(tensor_data);
        at_tensors.push_back(*tensor_data);
        devices.push_back(tensors[i].GetDevice());
        at_tensor_index.push_back(i);
      }
    }
  }
  if (!at_tensors.empty()) {
    TORCH_LAZY_COUNTER("SyncTensorsToData", at_tensors.size());
    std::vector<torch::lazy::BackendDataPtr> handles =
        torch::lazy::CreateTensorsData(at_tensors, devices);
    for (size_t i = 0; i < handles.size(); ++i) {
      // If we are here, it means that the IR Value for the tensor is not
      // present. Also, we uploaded the at::Tensor data to the device, but such
      // data is still valid so we leave it live on the lazy tensor (so that a
      // following ToTensor() does not need to fetch it from device).
      tensors[at_tensor_index[i]].data()->handle = std::move(handles[i]);
    }
  }
  VLOG(4) << "Tensors graph hash " << torch::lazy::HashToString(coll.hash)
          << " on device " << coll.device;
  return coll;
}

std::vector<torch::lazy::Value> LazyGraphExecutor::CollectRoots(
    const std::vector<LazyTensor>& tensors, c10::ArrayRef<size_t> indices) {
  std::vector<torch::lazy::Value> roots;
  roots.reserve(indices.size());
  for (auto index : indices) {
    roots.push_back(tensors.at(index).CurrentIrValue());
  }
  return roots;
}

std::vector<torch::lazy::BackendDataPtr> LazyGraphExecutor::FetchTensorData(
    std::vector<LazyTensor>* tensors, const SyncTensorsConfig& config,
    c10::ArrayRef<size_t> indices) {
  std::vector<torch::lazy::BackendDataPtr> tensors_data;
  tensors_data.reserve(indices.size());
  for (auto index : indices) {
    LazyTensor& tensor = (*tensors)[index];
    // If the config.force_ltc_data flag is true, the purpose of this tensor
    // sync operation is to truncate the IR graph and materialize device data in
    // place of IR graph, on selected tensors. But since operation will complete
    // asynchronously, if a tensor does not already have device data, we need to
    // install a placeholder. Since at this point we hold a lock on the device
    // where the tensors reside (locks held within the coll structure, and moved
    // into the async variable), any other operation trying to access the
    // tensor's device data will have to wait until the asynchronous operation
    // completes.
    torch::lazy::BackendDataPtr handle = tensor.CurrentDataHandle();
    if (handle == nullptr && config.force_ltc_data) {
      const torch::lazy::BackendDevice& tensor_device = tensor.GetDevice();
      handle = torch::lazy::getBackend()->CreateDataPlaceholder(
          tensor_device, std::move(tensor.shape()));
      tensor.SetDataHandle(handle, config.sync_ltc_data);
    }
    tensors_data.emplace_back(std::move(handle));
  }
  return tensors_data;
}

LazyGraphExecutor::PostOrderData LazyGraphExecutor::RunPostOrder(
    const std::vector<LazyTensor>& tensors, c10::ArrayRef<size_t> indices) {
  std::vector<torch::lazy::Node*> roots;
  roots.reserve(indices.size());
  for (auto index : indices) {
    torch::lazy::Value ir_value = tensors.at(index).CurrentIrValue();
    roots.push_back(ir_value.node.get());
  }
  PostOrderData po_data;
  po_data.post_order =
      torch::lazy::Util::ComputePostOrder(roots, &po_data.emission_map);
  std::unordered_map<torch::lazy::BackendData::Handle, size_t> data_handles;
  for (auto node : po_data.post_order) {
    const torch::lazy::DeviceData* device_data = torch::lazy::DeviceData::Cast(node);
    if (device_data != nullptr) {
      torch::lazy::BackendData::Handle handle =
          device_data->data()->GetHandle();
      auto it = data_handles.find(handle);
      if (it != data_handles.end()) {
        po_data.parameter_sequence.push_back(it->second);
      } else {
        po_data.parameter_sequence.push_back(po_data.parameters_data.size());
        data_handles[handle] = po_data.parameters_data.size();
        po_data.parameters_data.push_back(device_data->data());
      }
    }
  }
  return po_data;
}

std::shared_ptr<LazyGraphExecutor::Async> LazyGraphExecutor::TryRunCachedSync(
    std::vector<LazyTensor>* tensors, SyncTensorCollection* coll,
    PostOrderData* po_data) {
  ComputationCache::TypePtr cached_computation =
      LookupCachedCompile(*tensors, coll->hash);
  if (cached_computation == nullptr) {
    return nullptr;
  }
  TORCH_LAZY_VALUE_METRIC("TensorsGraphSize", po_data->post_order.size());
  VLOG(5) << "TensorsGraphSize=" << po_data->post_order.size();

  return ScheduleSyncTensorsGraph(
      tensors, coll, std::move(po_data->parameters_data), std::move(cached_computation));
}

LazyGraphExecutor::CompilationResult LazyGraphExecutor::Compile(
    const std::vector<LazyTensor>& tensors, c10::ArrayRef<std::string> devices,
    const SyncTensorCollection& coll, PostOrderData* po_data) {
  static const bool enable_aliasing =
      lazy_tensors::sys_util::GetEnvBool("ENABLE_PARAM_ALIASING", false);
  auto lowering_ctx = torch::lazy::LoweringContext::Create(
      "SyncTensorsGraph", coll.device, po_data->post_order,
      std::move(po_data->emission_map));
  for (auto index : coll.indices) {
    torch::lazy::Value ir_value = tensors[index].CurrentIrValue();
    lowering_ctx->AddResult(ir_value);
  }
  if (enable_aliasing && coll.config.sync_ltc_data) {
    // We can only alias at the step barrier, when force_ltc_data is true.
    // Consider the case:
    //   1. Tensor A(DEVICE_DATA)
    //   2. Tensor B = A + 0.9
    //   3. A += 0.4
    // If we activate aliasing for A's graph, and we do:
    //   print(A)
    //   print(A)
    // The first print will update DEVICE_DATA' with DEVICE_DATA+0.4, and the
    // second print will again update DEVICE_DATA" with DEVICE_DATA'+0.4, which
    // will lead to incorrect results.
    // We cannot normally turn A's state into DEVICE_DATA, as if any of the
    // sources is a view, this will not lead to correct results (as A's value
    // taken at different times need to reflect view source changes):
    //   1. Tensor A = some_graph_with_view_source(V)
    //   2. print(A)
    //   3. V += 1
    //   4. print(A)
    // The second print should reflect the new value due to V's changes.
    // Also in the first example, unless we are doing a step barrier and hence
    // include all live tensors, if the B value is not part of the graph, it
    // will later fetch the new value of A, which is incorrect.
    // But, when we issue a step barrier (force_ltc_data == true) we have to
    // turn everything into DEVICE_DATA, so we can activate aliasing.
    BuildInputOutputAliases(tensors, coll.indices, lowering_ctx.get());
  }

  torch::lazy::ComputationPtr computation = lowering_ctx->Build();

  VLOG(3) << "Compiling IR graph hash " << torch::lazy::HashToString(coll.hash)
          << " on device " << coll.device << " ...";
  std::vector<torch::lazy::ComputationPtr> computations =
      torch::lazy::getBackend()->Compile({computation});
  VLOG(3) << "Compiling IR graph hash " << torch::lazy::HashToString(coll.hash)
          << " on device " << coll.device << " done!";
  if (computation) {
    // TODO(whc) should computation be allowed null here? (becuase it is in one
    // case)
    CHECK_EQ(computation->parameters_size(), po_data->parameters_data.size());
  }

  return {/*device=*/coll.device,
          /*emitted_nodes=*/lowering_ctx->GetEmittedNodeCount(),
          /*computation=*/std::move(computations.front()),
          /*parameters_data=*/std::move(po_data->parameters_data)};
}

LazyGraphExecutor::ComputationCache* LazyGraphExecutor::GetComputationCache() {
  static const size_t kMaxCacheSize =
      lazy_tensors::sys_util::GetEnvInt("COMPILATION_CACHE_SIZE", 1024);
  static ComputationCache* cache = new ComputationCache(kMaxCacheSize);
  return cache;
}

LazyGraphExecutor::ComputationCache::TypePtr
LazyGraphExecutor::LookupCachedCompile(const std::vector<LazyTensor>& tensors,
                                       const torch::lazy::hash_t& hash) {
  ComputationCache::TypePtr cached_computation =
      GetComputationCache()->Get(hash);
  if (cached_computation == nullptr) {
    TORCH_LAZY_COUNTER("UncachedCompile", 1);
    return nullptr;
  }
  TORCH_LAZY_COUNTER("CachedCompile", 1);
  return cached_computation;
}

void LazyGraphExecutor::BuildInputOutputAliases(
    const std::vector<LazyTensor>& tensors, c10::ArrayRef<size_t> indices,
    torch::lazy::LoweringContext* lowering_ctx) {
  std::unordered_map<int64_t, size_t> output_tensor_id_map;
  for (size_t i = 0; i < indices.size(); ++i) {
    size_t tensor_index = indices[i];
    int64_t tensor_id = tensors[tensor_index].GetUniqueId();
    output_tensor_id_map[tensor_id] = i;
  }
  const std::vector<torch::lazy::BackendDataPtr>& parameters_data =
      lowering_ctx->GetParametersData();
  std::vector<ssize_t> alias_map(indices.size(), -1);
  for (size_t i = 0; i < parameters_data.size(); ++i) {
    DeviceDataInfo* data_info =
        dynamic_cast<DeviceDataInfo*>(parameters_data[i]->info());
    if (data_info != nullptr && !data_info->read_only) {
      auto it = output_tensor_id_map.find(data_info->tensor_id);
      if (it != output_tensor_id_map.end()) {
        size_t output_index = it->second;
        const torch::lazy::Shape& root_shape =
            lowering_ctx->GetResultShape(output_index);
        if (torch::lazy::Shape(parameters_data[i]->shape()) == root_shape &&
            alias_map[output_index] < 0) {

          // TODO(whc) deleted this interface until we see a need (no TS impl)
          // lowering_ctx->SetUpAlias({static_cast<int64_t>(output_index)}, i, {});
          alias_map[output_index] = i;

          VLOG(6) << "Aliased paramter " << i << " with output " << output_index
                  << ": " << torch::lazy::Shape(parameters_data[i]->shape());
        }
      }
    }
  }
  TORCH_LAZY_VALUE_METRIC("InputOutputAliasCount", alias_map.size());
}

std::shared_ptr<LazyGraphExecutor::Async>
LazyGraphExecutor::SyncTensorsGraphInternal(std::vector<LazyTensor>* tensors,
                                            c10::ArrayRef<std::string> devices,
                                            const SyncTensorsConfig& config) {
  SyncTensorCollection coll = CollectSyncTensors(*tensors, config);
  if (coll.indices.empty()) {
    return nullptr;
  }
  // DebugUtil::SaveTensorsGraphInfo("ScheduleSyncTensorsGraph", *tensors,
  //                                &coll.indices);

  PostOrderData po_data = RunPostOrder(*tensors, coll.indices);
  coll.hash = torch::lazy::HashCombine(
      coll.hash, torch::lazy::Hash(po_data.parameter_sequence));
  VLOG(4) << "Parameter sequence graph hash "
          << torch::lazy::HashToString(coll.hash);
  std::shared_ptr<Async> async = TryRunCachedSync(tensors, &coll, &po_data);
  if (async != nullptr) {
    return async;
  }

  CompilationResult compile_result = Compile(*tensors, devices, coll, &po_data);

  TORCH_LAZY_VALUE_METRIC("TensorsGraphSize", compile_result.emitted_nodes);
  VLOG(5) << "TensorsGraphSize=" << compile_result.emitted_nodes;

  auto cached_computation = std::make_shared<CachedComputation>(
      std::move(compile_result.computation));
  GetComputationCache()->Add(coll.hash, cached_computation);

  return ScheduleSyncTensorsGraph(
      tensors, &coll, std::move(compile_result.parameters_data), std::move(cached_computation));
}

std::shared_ptr<LazyGraphExecutor::Async>
LazyGraphExecutor::ScheduleSyncTensorsGraph(
    SyncTensorCollection* coll,
    std::vector<torch::lazy::BackendDataPtr> parameters_data,
    std::vector<torch::lazy::BackendDataPtr> tensors_data,
    ComputationCache::TypePtr cached_computation) {
  std::shared_ptr<Async> async = std::make_shared<Async>(
      coll, std::move(parameters_data), std::move(tensors_data),
      std::move(cached_computation));

  auto syncfn = [this, async, hash = coll->hash]() {
    // For profiling lazy trace overhead
    if (noop_execution_mode_)
      return;

    try {
      VLOG(3) << "Executing IR graph hash " << torch::lazy::HashToString(hash)
              << " on device " << async->device << " ...";
      auto results = torch::lazy::getBackend()->ExecuteComputation(
          *async->cached_computation->computation, async->parameters_data,
          async->device);
      VLOG(3) << "Executing IR graph hash " << torch::lazy::HashToString(hash)
              << " on device " << async->device << " done!";

      for (size_t i = 0; i < results.size(); ++i) {
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
      std::exception_ptr exptr = std::current_exception();
      for (auto& unlocker : async->unlocker) {
        unlocker.SetStatus(std::move(exptr));
      }
      throw;
    }
  };

  torch::lazy::ScheduleIoClosure(async->mwait.Completer(std::move(syncfn)));
  return async;
}

std::shared_ptr<LazyGraphExecutor::Async>
LazyGraphExecutor::ScheduleSyncTensorsGraph(
    std::vector<LazyTensor>* tensors, SyncTensorCollection* coll,
    std::vector<torch::lazy::BackendDataPtr> parameters_data,
    ComputationCache::TypePtr cached_computation) {
  auto tensors_data = FetchTensorData(tensors, coll->config, coll->indices);
  return ScheduleSyncTensorsGraph(coll, std::move(parameters_data),
                                  std::move(tensors_data),
                                  std::move(cached_computation));
}

std::vector<at::Tensor> LazyGraphExecutor::GetTensorsFused(
    std::vector<LazyTensor>* tensors) {
  SyncTensorsConfig config;
  config.force_ltc_data = false;
  auto async = SyncTensorsGraphInternal(tensors, {}, config);
  if (async != nullptr) {
    async->mwait.Wait();
  }
  std::vector<torch::lazy::BackendDataPtr> tensors_data = GatherTensorsData(
      *tensors, async != nullptr ? async->indices : c10::ArrayRef<size_t>(),
      async != nullptr ? async->tensors_data
                       : c10::ArrayRef<torch::lazy::BackendDataPtr>());
  return FetchTensors(tensors, tensors_data,
                      async != nullptr ? &async->indices : nullptr);
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
    std::vector<LazyTensor>* tensors,
    c10::ArrayRef<torch::lazy::BackendDataPtr> tensors_data,
    const std::vector<size_t>* indices) {
  std::vector<at::Tensor> results;
  size_t literals_index = 0;
  size_t sync_index = 0;
  results.reserve(tensors->size());
  for (size_t i = 0; i < tensors->size(); ++i) {
    if (indices != nullptr && sync_index < indices->size() &&
        i == (*indices)[sync_index]) {
      results.push_back(
          torch::lazy::getBackend()->MakeTensorFromComputationData(
              tensors_data[literals_index], (*tensors)[i].dtype()));
      ++literals_index;
      ++sync_index;
    } else {
      c10::optional<at::Tensor> tensor_data = (*tensors)[i].CurrentTensorData();
      if (tensor_data) {
        results.push_back(*tensor_data);
      } else {
        CHECK_LT(literals_index, tensors_data.size());
        results.push_back(
            torch::lazy::getBackend()->MakeTensorFromComputationData(
                tensors_data[literals_index], (*tensors)[i].dtype()));
        ++literals_index;
      }
    }
  }
  return results;
}

std::vector<torch::lazy::BackendDataPtr> LazyGraphExecutor::GatherTensorsData(
    const std::vector<LazyTensor>& tensors, c10::ArrayRef<size_t> indices,
    c10::ArrayRef<torch::lazy::BackendDataPtr> tensors_data) {
  std::vector<torch::lazy::BackendDataPtr> result_tensors_data;
  std::unordered_map<int64_t, size_t> uid_index_map;
  size_t indices_index = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    int64_t tensor_id = tensors[i].GetUniqueId();
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
    } else if (!tensors[i].CurrentTensorData()) {
      torch::lazy::BackendDataPtr handle = tensors[i].CurrentDataHandle();
      CHECK(handle != nullptr);
      result_tensors_data.push_back(std::move(handle));
    }
  }
  return result_tensors_data;
}

// LazyGraphExecutor
}  // namespace torch_lazy_tensors
