#include "lazy_tensor_core/csrc/lazy_graph_executor.h"

#include "lazy_tensor_core/csrc/ops/arithmetic_ir_ops.h"
#include "lazy_tensor_core/csrc/ops/device_data.h"
#include "lazy_tensor_core/csrc/ops/ops.h"
#include "lazy_tensor_core/csrc/ops/scalar.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/torch_util.h"

namespace torch_lazy_tensors {
namespace {

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
  explicit DeviceLocker(Device device) : device_(std::move(device)) {}

  const Device& device() const { return device_; }

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

  Device device_;
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

  std::shared_ptr<DeviceLocker> GetLocker(const Device& device) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = lockers_.find(device);
    if (it == lockers_.end()) {
      it = lockers_.emplace(device, std::make_shared<DeviceLocker>(device))
               .first;
    }
    return it->second;
  }

  void DeviceBarrier(const Device& device) {
    auto locker = DeviceLockerArena::Get()->GetLocker(device);
    locker->Barrier();
  }

  // Use a set to impose an order on the device locking sequence (ABBA
  // prevention).
  std::vector<lazy_tensors::util::ExceptionCleanup> LockDevices(
      const std::set<Device>& devices) {
    std::vector<lazy_tensors::util::ExceptionCleanup> unlocker;
    unlocker.reserve(devices.size());
    for (auto& device : devices) {
      unlocker.emplace_back(LockDevice(device));
    }
    return unlocker;
  }

 private:
  lazy_tensors::util::ExceptionCleanup LockDevice(const Device& device) {
    auto locker = DeviceLockerArena::Get()->GetLocker(device);
    locker->Lock();
    return lazy_tensors::util::ExceptionCleanup(
        [locker = std::move(locker)](
            lazy_tensors::util::ExceptionCleanup::StatusType status) {
          locker->Unlock(std::move(status));
        });
  }

  std::mutex mutex_;
  std::map<Device, std::shared_ptr<DeviceLocker>> lockers_;
};

// The DeviceContextArena holds per device live information and statistics,
// among which the lazy tensors which are currently alive in the system. This is
// used to create computation "barriers" in order to flush pending operations
// and ensure the same computations are created during the training loops.
class DeviceContextArena {
  struct DeviceContext {
    std::mutex lock;
    std::map<lazy_tensors::int64, std::weak_ptr<LazyTensor::Data>> tensors_data;
    lazy_tensors::uint64 seed = 101;
    lazy_tensors::uint64 running_seed = 101;
    ir::Value seed_ir_value;
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
    LTC_COUNTER("CreateLtcTensor", 1);
  }

  void UnregisterTensor(LazyTensor::Data* data) {
    DeviceContext* devctx = GetDeviceContext(data->device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    devctx->tensors_data.erase(data->unique_id);
    LTC_COUNTER("DestroyLtcTensor", 1);
  }

  std::vector<LazyTensor> GetLiveTensors(const Device* device) {
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

  ir::Value GetRngSeed(const Device& device) {
    static const at::ScalarType kSeedType = at::ScalarType::Long;
    static const lazy_tensors::uint64 kSeedMul = 214013;
    static const lazy_tensors::uint64 kSeedAdd = 2531011;
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
    ir::Value k = ir::ops::ScalarOp(MakeIntScalar(kSeedMul),
                                    MakeLtcPrimitiveType(kSeedType, &device));
    ir::Value b = ir::ops::ScalarOp(MakeIntScalar(kSeedAdd),
                                    MakeLtcPrimitiveType(kSeedType, &device));
    devctx->seed_ir_value = b + k * devctx->seed_ir_value;
    return devctx->seed_ir_value;
  }

  lazy_tensors::uint64 GetRunningSeed(const Device& device) {
    DeviceContext* devctx = GetDeviceContext(device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    return devctx->running_seed;
  }

  void SetRngSeed(const Device& device, lazy_tensors::uint64 seed) {
    DeviceContext* devctx = GetDeviceContext(device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    devctx->seed = seed;
    devctx->running_seed = devctx->seed;
    devctx->seed_ir_value = ir::Value();
  }

  void MarkStep(const Device& device) {
    DeviceContext* devctx = GetDeviceContext(device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    devctx->seed = 1012031 + devctx->seed * 7012063;
    devctx->running_seed = devctx->seed;
    devctx->seed_ir_value = ir::Value();
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
                            const Device* device) {
    if (device == nullptr) {
      for (auto devctx : GetAllDeviceContexts()) {
        fn(devctx);
      }
    } else {
      fn(GetDeviceContext(*device));
    }
  }

  DeviceContext* GetDeviceContext(const Device& device) {
    std::lock_guard<std::mutex> lock(lock_);
    auto it = device_contexts_.find(device);
    if (it == device_contexts_.end()) {
      it = device_contexts_.emplace(device, new DeviceContext()).first;
    }
    return it->second;
  }

  ir::Value IrValueFromScalar(const at::Scalar& value,
                              at::ScalarType scalar_type,
                              const Device& device) {
    at::Tensor tensor =
        at::scalar_tensor(value, at::TensorOptions(scalar_type));
    lazy_tensors::ComputationClient::DataPtr device_data =
        TensorToDataHandle(tensor, device);
    return ir::MakeNode<ir::ops::DeviceData>(std::move(device_data));
  }

  std::mutex lock_;
  std::map<Device, DeviceContext*> device_contexts_;
};
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

ir::Value LazyGraphExecutor::GetRngSeed(const Device& device) {
  return DeviceContextArena::Get()->GetRngSeed(device);
}

lazy_tensors::uint64 LazyGraphExecutor::GetRunningSeed(const Device& device) {
  return DeviceContextArena::Get()->GetRunningSeed(device);
}

void LazyGraphExecutor::SetRngSeed(const Device& device,
                                   lazy_tensors::uint64 seed) {
  DeviceContextArena::Get()->SetRngSeed(device, seed);
}

std::vector<LazyTensor> LazyGraphExecutor::GetLiveTensors(
    const Device* device) {
  return DeviceContextArena::Get()->GetLiveTensors(device);
}

void LazyGraphExecutor::MarkStep(const Device& device) {
  DeviceContextArena::Get()->MarkStep(device);
}

void LazyGraphExecutor::DeviceBarrier(const Device& device) {
  DeviceLockerArena::Get()->DeviceBarrier(device);
}

std::vector<lazy_tensors::util::ExceptionCleanup>
LazyGraphExecutor::LockDevices(const std::set<Device>& devices) {
  return DeviceLockerArena::Get()->LockDevices(devices);
}

}  // namespace torch_lazy_tensors
