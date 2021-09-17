#include "lazy_tensor_core/csrc/tensor.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <set>
#include <stdexcept>
#include <unordered_set>

#include "lazy_tensor_core/csrc/debug_util.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ir_dump_util.h"
#include "lazy_tensor_core/csrc/layout_manager.h"
#include "lazy_tensor_core/csrc/op_by_op_executor.h"
#include "lazy_tensor_core/csrc/ops/arithmetic_ir_ops.h"
#include "lazy_tensor_core/csrc/ops/cast.h"
#include "lazy_tensor_core/csrc/ops/device_data.h"
#include "lazy_tensor_core/csrc/ops/expand.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensor_core/csrc/ops/ops.h"
#include "lazy_tensor_core/csrc/ops/view.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/torch_util.h"
// #include "lazy_tensor_core/csrc/ts_backend/ts_computation_client.h"
#include "lazy_tensors/computation_client/cache.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/ltc_util.h"
#include "lazy_tensors/computation_client/metrics.h"
#include "lazy_tensors/computation_client/sys_util.h"
#include "lazy_tensors/computation_client/thread_pool.h"
#include "lazy_tensors/computation_client/unique.h"
#include "lazy_tensors/literal_util.h"
#include "lazy_tensors/shape_util.h"
#include "lazy_tensors/str_join.h"
#include "torch/csrc/autograd/variable.h"

namespace torch_lazy_tensors {
namespace {

struct TlsData {
  void Reset() { trim_counter = 0; }

  size_t trim_counter = 0;
};

thread_local TlsData g_tls_data;

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

 private:
  std::mutex mutex_;
  std::map<Device, std::shared_ptr<DeviceLocker>> lockers_;
};

lazy_tensors::util::ExceptionCleanup LockDevice(const Device& device) {
  auto locker = DeviceLockerArena::Get()->GetLocker(device);
  locker->Lock();
  return lazy_tensors::util::ExceptionCleanup(
      [locker = std::move(locker)](
          lazy_tensors::util::ExceptionCleanup::StatusType status) {
        locker->Unlock(std::move(status));
      });
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

class DataCacheArena {
 public:
  struct TensorHasher {
    size_t operator()(const at::Tensor& tensor) const {
      return lazy_tensors::util::HashReduce(lazy_tensors::util::HashCombine(
          lazy_tensors::util::GetEnumValue(tensor.scalar_type()),
          TensorHash(tensor)));
    };
  };
  struct TensorComparer {
    bool operator()(const at::Tensor& tensor1,
                    const at::Tensor& tensor2) const {
      return TensorCompare(tensor1, tensor2);
    }
  };

  using DataCache =
      lazy_tensors::util::Cache<at::Tensor, lazy_tensors::client::Data,
                                TensorHasher, TensorComparer>;

  explicit DataCacheArena(size_t max_cache_size)
      : max_cache_size_(max_cache_size) {}

  DataCache* Get(const Device& device) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = device_caches_.find(device);
    if (it == device_caches_.end()) {
      std::unique_ptr<DataCache> cache(new DataCache(max_cache_size_));
      it = device_caches_.emplace(device, std::move(cache)).first;
    }
    return it->second.get();
  }

 private:
  size_t max_cache_size_ = 0;
  std::mutex mutex_;
  std::map<Device, std::unique_ptr<DataCache>> device_caches_;
};

DataCacheArena::DataCache* GetDataCache(const Device& device) {
  static const size_t kMaxCacheSize =
      lazy_tensors::sys_util::GetEnvInt("DEVDATA_CACHE_SIZE", 128);
  static DataCacheArena* arena = new DataCacheArena(kMaxCacheSize);
  return arena->Get(device);
}

ir::Value IrValueFromScalar(const at::Scalar& value, at::ScalarType scalar_type,
                            const Device& device) {
  at::Tensor tensor = at::scalar_tensor(value, at::TensorOptions(scalar_type));
  lazy_tensors::ComputationClient::DataPtr device_data =
      TensorToDataHandle(tensor, device);
  return ir::MakeNode<ir::ops::DeviceData>(std::move(device_data));
}

lazy_tensors::ComputationClient::DataPtr GetDeviceData(const at::Tensor& tensor,
                                                       const Device& device) {
  DataCacheArena::DataCache* cache = GetDataCache(device);
  lazy_tensors::ComputationClient::DataPtr device_data = cache->Get(tensor);
  if (device_data == nullptr) {
    at::Tensor tensor_copy = CopyTensor(tensor);
    device_data = TensorToDataHandle(tensor_copy, device);
    cache->Add(std::move(tensor_copy), device_data);
    LTC_COUNTER("DeviceDataCacheMiss", 1);
  }
  return device_data;
}

lazy_tensors::ComputationClient::DataPtr GetDeviceData(
    const at::Scalar& value, at::ScalarType scalar_type, const Device& device) {
  // Workaround since at::scalar_tensor doesn't support bfloat16 yet.
  at::Tensor t = at::scalar_tensor(
      value, at::TensorOptions(scalar_type == at::ScalarType::BFloat16
                                   ? at::ScalarType::Float
                                   : scalar_type));
  if (scalar_type == at::ScalarType::BFloat16) t = t.to(scalar_type);
  return GetDeviceData(t, device);
}

// Routing values to device data maximizes the changes for compilation cache
// hits, but it can prevent the compiler to perform optimizations. So tensor
// values which are within a given set, are routed to constant scalars if this
// API returns true.
bool IsSpecialScalar(const at::Scalar& value) {
  static bool no_scalars =
      lazy_tensors::sys_util::GetEnvBool("NO_SPECIAL_SCALARS", false);
  if (!no_scalars && (value.isIntegral() || value.isFloatingPoint())) {
    double scalar_value = value.toDouble();
    return scalar_value == 0.0 || std::fabs(scalar_value) == 1.0;
  }
  return false;
}

bool ShouldSyncIrValue(const ir::Value& ir_value) {
  return ir_value->op() != ir::ops::ltc_not_supported;
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

// The DeviceContextArena holds per device live information and statistics,
// among which the lazy tensors which are currently alive in the system. This is
// used to create computation "barriers" in order to flush pending operations
// and ensure the same computations are created during the training loops.
class LazyTensor::DeviceContextArena {
  struct DeviceContext {
    std::mutex lock;
    std::map<lazy_tensors::int64, std::weak_ptr<Data>> tensors_data;
    lazy_tensors::uint64 seed = 101;
    lazy_tensors::uint64 running_seed = 101;
    ir::Value seed_ir_value;
  };

 public:
  static DeviceContextArena* Get() {
    static DeviceContextArena* arena = new DeviceContextArena();
    return arena;
  }

  void RegisterTensor(std::shared_ptr<Data> data) {
    DeviceContext* devctx = GetDeviceContext(data->device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    devctx->tensors_data.emplace(data->unique_id, data);
    LTC_COUNTER("CreateLtcTensor", 1);
  }

  void UnregisterTensor(Data* data) {
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
        std::shared_ptr<Data> data = uid_wptr.second.lock();
        if (data != nullptr) {
          tensors.push_back(LazyTensor(std::move(data)));
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

  std::mutex lock_;
  std::map<Device, DeviceContext*> device_contexts_;
};

struct DeviceDataInfo : public lazy_tensors::client::Data::Info {
  DeviceDataInfo(lazy_tensors::int64 tensor_id, bool read_only)
      : tensor_id(tensor_id), read_only(read_only) {}

  lazy_tensors::int64 tensor_id = 0;
  bool read_only = false;
};

LazyTensor::Data::~Data() { DeviceContextArena::Get()->UnregisterTensor(this); }

LazyTensor::Async::Async(
    SyncTensorCollection* coll,
    std::vector<lazy_tensors::ComputationClient::DataPtr> parameters_data,
    std::vector<lazy_tensors::ComputationClient::DataPtr> tensors_data,
    ComputationCache::TypePtr cached_computation)
    : mwait(1),
      indices(std::move(coll->indices)),
      unlocker(std::move(coll->unlocker)),
      parameters_data(std::move(parameters_data)),
      device(coll->device.ToString()),
      cached_computation(std::move(cached_computation)),
      tensors_data(std::move(tensors_data)) {}

void LazyTensor::Async::Wait() {
  mwait.Wait();
  // Accessing other Async members is safe only after MultiWait::Wait()
  // completes.
  lazy_tensors::util::ExceptionCleanup::StatusType status;
  for (auto& cleanup : unlocker) {
    const lazy_tensors::util::ExceptionCleanup::StatusType& cleanup_status =
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

LazyTensor LazyTensor::Create(const at::Tensor& tensor, const Device& device) {
  LTC_CHECK_NE(tensor.device().type(), at::kLazy);
  LazyTensor xtensor(tensor, device);
  DeviceContextArena::Get()->RegisterTensor(xtensor.data_ptr());
  return xtensor;
}

LazyTensor LazyTensor::Create(
    lazy_tensors::ComputationClient::DataPtr handle,
    c10::optional<at::ScalarType> logical_element_type) {
  LazyTensor xtensor(std::move(handle), logical_element_type);
  DeviceContextArena::Get()->RegisterTensor(xtensor.data_ptr());
  return xtensor;
}

LazyTensor LazyTensor::Create(
    ir::Value ir_value, const Device& device,
    c10::optional<at::ScalarType> logical_element_type) {
  LazyTensor xtensor(std::move(ir_value), device, logical_element_type);
  DeviceContextArena::Get()->RegisterTensor(xtensor.data_ptr());
  return xtensor;
}

LazyTensor LazyTensor::Create(
    std::shared_ptr<View> view, const Device& device,
    c10::optional<at::ScalarType> logical_element_type) {
  LazyTensor xtensor(std::move(view), device, logical_element_type);
  DeviceContextArena::Get()->RegisterTensor(xtensor.data_ptr());
  return xtensor;
}

LazyTensor::LazyTensor(const at::Tensor& tensor, const Device& device)
    : data_(std::make_shared<Data>(tensor, device)) {}

LazyTensor::LazyTensor(lazy_tensors::ComputationClient::DataPtr handle,
                       c10::optional<at::ScalarType> logical_element_type)
    : data_(std::make_shared<Data>(handle, Device(handle->device()),
                                   logical_element_type)) {}

LazyTensor::LazyTensor(ir::Value ir_value, const Device& device,
                       c10::optional<at::ScalarType> logical_element_type)
    : data_(std::make_shared<Data>(std::move(ir_value), device,
                                   logical_element_type)) {
  TryLimitGraphSize();
}

LazyTensor::LazyTensor(std::shared_ptr<View> view, const Device& device,
                       c10::optional<at::ScalarType> logical_element_type)
    : data_(std::make_shared<Data>(std::move(view), device,
                                   logical_element_type)) {}

LazyTensor::LazyTensor(std::shared_ptr<Data> data) : data_(std::move(data)) {}

LazyTensor::Data* LazyTensor::data() const {
  LTC_CHECK(data_ != nullptr) << "Trying to access a null cursor";
  return data_.get();
}

lazy_tensors::int64 LazyTensor::size(lazy_tensors::int64 dim) const {
  auto tensor_shape = shape();
  int rank = tensor_shape.get().rank();
  int dim_index = Helpers::GetCanonicalDimensionIndex(dim, rank);
  return tensor_shape.get().dimensions(dim_index);
}

at::ScalarType LazyTensor::dtype() const {
  return data()->logical_element_type
             ? *data()->logical_element_type
             : TensorTypeFromLtcType(shape().get().element_type());
}

c10::optional<at::ScalarType> LazyTensor::dtype_optional() const {
  return data()->logical_element_type;
}

lazy_tensors::util::MaybeRef<lazy_tensors::Shape> LazyTensor::shape() const {
  if (data()->view != nullptr) {
    return data()->view->shape();
  }
  if (data()->handle != nullptr) {
    return lazy_tensors::Shape(data()->handle->shape());
  }
  if (data()->ir_value) {
    return data()->ir_value.shape();
  }
  LTC_CHECK(data()->tensor_data);
  const Device& device = GetDevice();
  return lazy_tensors::ShapeUtil::MakeShape(
      MakeLtcPrimitiveType(data()->tensor_data->type().scalarType(), &device),
      Helpers::I64List(data()->tensor_data->sizes()));
}

lazy_tensors::Shape LazyTensor::shape_with_layout() const {
  auto tensor_shape = shape();
  return MakeArrayShapeFromDimensions(
      tensor_shape.get().dimensions(), tensor_shape.get().dynamic_dimensions(),
      tensor_shape.get().element_type(), GetDevice().hw_type);
}

const Device& LazyTensor::GetDevice() const { return data()->device; }

lazy_tensors::int64 LazyTensor::GetUniqueId() const {
  return data()->unique_id;
}

std::ptrdiff_t LazyTensor::GetViewAliasId() const {
  return data()->view != nullptr
             ? reinterpret_cast<std::ptrdiff_t>(data()->view->alias().get())
             : 0;
}

lazy_tensors::ComputationClient::DataPtr LazyTensor::GetDataHandle() {
  // Data can coexist with a view, but we need to check that the view did
  // not receive any updates before calling the current IR valid.
  bool up_to_date = true;
  ir::Value ir_value;
  if (data()->view != nullptr) {
    View::IrNode ir_value_updated = GetViewUpdate(data()->view);
    up_to_date = !ir_value_updated.updated;
    ir_value = std::move(ir_value_updated.ir_value);
  }
  if (up_to_date) {
    lazy_tensors::ComputationClient::DataPtr handle = CurrentDataHandle();
    if (handle != nullptr) {
      LTC_CHECK(handle->HasValue())
          << "Trying to access data while an async operation is in flight: "
          << lazy_tensors::Shape(handle->shape());
      return handle;
    }
  }
  if (ir_value) {
    // The view gave us an updated IR value. We usually do not have a valid IR
    // value field together with a view, but to allow code reuse in
    // ApplyPendingGraph() we temporarily set it here. The following call to
    // ApplyPendingGraph() will clear it.
    AssignIrValue(std::move(ir_value));
  }
  if (data()->ir_value) {
    ApplyPendingGraph();
  } else {
    LTC_CHECK(data()->tensor_data);
    data()->handle = TensorToDataHandle(*data()->tensor_data, GetDevice());
  }
  return data()->handle;
}

lazy_tensors::ComputationClient::DataPtr LazyTensor::CurrentDataHandle() const {
  return data()->handle;
}

std::string LazyTensor::DumpBackendComputation(
    const std::vector<LazyTensor>& tensors) {
  std::vector<ir::Value> ir_values;
  for (auto& tensor : tensors) {
    ir::Value ir_value = tensor.CurrentIrValue();
    if (ir_value) {
      ir_values.push_back(std::move(ir_value));
    }
  }
  return !ir_values.empty()
             ? ir::DumpUtil::ToBackend(ir_values, GetCurrentDevice())
             : std::string();
}

void LazyTensor::SetDataHandle(
    lazy_tensors::ComputationClient::DataPtr handle) {
  SetDataHandle(std::move(handle), /*sync=*/true);
}

void LazyTensor::SetDataHandle(lazy_tensors::ComputationClient::DataPtr handle,
                               bool sync) {
  data()->handle = std::move(handle);
  // Assigning a device data should always clear the IR node, to allow graph
  // trimming. A view cannot be reset though, unless we are at a step-end sync.
  AssignIrValue(ir::Value());
  if (sync) {
    data()->view = nullptr;
    data()->tensor_data = c10::nullopt;
  }
}

void LazyTensor::SetIrValue(ir::Value ir_value) {
  data()->handle = nullptr;
  data()->tensor_data = c10::nullopt;
  if (data()->view != nullptr) {
    // If we have an active view, and a SetIrValue() happens, it means we are
    // within an in-place execution context, and we need to update the view's
    // alias as well.
    data()->view = UpdateView(data()->view, std::move(ir_value));
    data()->generation += 1;
  } else {
    AssignIrValue(std::move(ir_value));
    TryLimitGraphSize();
  }
}

void LazyTensor::SetInPlaceIrValue(ir::Value ir_value) {
  auto tensor_shape = shape();
  if (tensor_shape.get().element_type() != ir_value.shape().element_type()) {
    ir_value = ir::MakeNode<ir::ops::Cast>(ir_value,
                                           tensor_shape.get().element_type());
  }
  SetIrValue(std::move(ir_value));
}

void LazyTensor::AssignIrValue(ir::Value ir_value) const {
  data()->ir_value = std::move(ir_value);
  data()->generation += 1;
}

void LazyTensor::TryLimitGraphSize() {
  static const size_t kCheckFrequency =
      lazy_tensors::sys_util::GetEnvInt("TRIM_GRAPH_CHECK_FREQUENCY", 5000);
  static const size_t kMaxPendingGraphSize =
      lazy_tensors::sys_util::GetEnvInt("TRIM_GRAPH_SIZE", 100000);
  if (data()->ir_value && ++g_tls_data.trim_counter % kCheckFrequency == 0) {
    size_t graph_size = ir::Util::GetGraphSize({data()->ir_value.node.get()});
    if (graph_size > kMaxPendingGraphSize) {
      LTC_COUNTER("TrimIrGraph", 1);
      ApplyPendingGraph();
    }
  }
}

ir::Value LazyTensor::GetIrValue() const {
  ir::Value ir_value = CurrentIrValue();
  if (ir_value) {
    return ir_value;
  }
  lazy_tensors::ComputationClient::DataPtr handle = CurrentDataHandle();
  if (handle != nullptr) {
    // In case of tensor node, we do not clear the data when we set the IR
    // node. This because we want further calls to GetIrValue() to fetch the
    // same IR node, and not create new ones (even though the lowering context
    // will still collapse them all into a single parameter op). So the call
    // which wants the data will still find it, w/out having to fetch it via
    // a computation client from-server call.
    AssignIrValue(CreateTensorNode(handle, /*read_only=*/false));
    return data()->ir_value;
  }
  c10::optional<at::Tensor> tensor_data = CurrentTensorData();
  LTC_CHECK(tensor_data);
  AssignIrValue(GetIrValueForTensor(*tensor_data, GetDevice()));
  return data()->ir_value;
}

ir::Value LazyTensor::CurrentIrValue() const {
  if (data()->view != nullptr) {
    return GetViewUpdate(data()->view).ir_value;
  }
  return data()->ir_value;
}

void LazyTensor::SetTensorData(at::Tensor tensor_data) {
  data()->tensor_data = std::move(tensor_data);
}

c10::optional<at::Tensor> LazyTensor::CurrentTensorData() const {
  if (data()->view != nullptr && !data()->view->IsUpToDate()) {
    return c10::nullopt;
  }
  return data()->tensor_data;
}

ir::Value LazyTensor::GetIrValueForTensor(const at::Tensor& tensor,
                                          const Device& device) const {
  lazy_tensors::ComputationClient::DataPtr data;
  bool read_only = false;
  if (tensor.dim() == 0 && tensor.numel() == 1) {
    at::Scalar value = tensor.item();
    if (IsSpecialScalar(value)) {
      return ir::ops::ScalarOp(
          std::move(value),
          MakeLtcPrimitiveType(tensor.scalar_type(), &device));
    }
    data = GetDeviceData(tensor.cpu(), device);
    read_only = true;
  } else {
    LTC_TIMED("IrValueTensorToDataHandle");
    data = TensorToDataHandle(tensor, device);
  }
  return CreateTensorNode(std::move(data), read_only);
}

ir::Value LazyTensor::GetDeviceDataIrValue(const at::Scalar& value,
                                           lazy_tensors::PrimitiveType type,
                                           const Device& device) {
  lazy_tensors::ComputationClient::DataPtr data =
      GetDeviceData(value, TensorTypeFromLtcType(type), device);
  data->SetInfo(
      std::make_shared<DeviceDataInfo>(/*tensor_id=*/-1, /*read_only=*/true));
  return ir::MakeNode<ir::ops::DeviceData>(std::move(data));
}

ir::Value LazyTensor::GetIrValueForScalar(const at::Scalar& value,
                                          lazy_tensors::PrimitiveType type,
                                          const Device& device) {
  if (IsSpecialScalar(value)) {
    return ir::ops::ScalarOp(std::move(value), type);
  }
  return GetDeviceDataIrValue(value, type, device);
}

ir::Value LazyTensor::GetIrValueForScalar(const at::Scalar& value,
                                          const Device& device) {
  return GetIrValueForScalar(
      value, MakeLtcPrimitiveType(GetScalarType(value), &device), device);
}

ir::Value LazyTensor::GetIrValueForScalar(
    const at::Scalar& value, lazy_tensors::PrimitiveType type,
    lazy_tensors::Span<const lazy_tensors::int64> dimensions,
    const Device& device) {
  ir::Value ir_value = GetIrValueForScalar(value, type, device);
  if (!dimensions.empty()) {
    ir_value = ir::MakeNode<ir::ops::Expand>(
        ir_value, lazy_tensors::util::ToVector<lazy_tensors::int64>(dimensions),
        /*is_scalar_expand=*/true);
  }
  return ir_value;
}

ir::Value LazyTensor::GetIrValueForScalar(const at::Scalar& value,
                                          const lazy_tensors::Shape& shape,
                                          const Device& device) {
  return GetIrValueForScalar(value, shape.element_type(), shape.dimensions(),
                             device);
}

ir::Value LazyTensor::GetIrValueForScalar(
    const at::Scalar& value, const lazy_tensors::Shape& shape,
    c10::optional<at::ScalarType> logical_element_type, const Device& device) {
  lazy_tensors::PrimitiveType type =
      logical_element_type
          ? MakeLtcPrimitiveType(*logical_element_type, &device)
          : shape.element_type();
  return GetIrValueForScalar(value, type, shape.dimensions(), device);
}

View::IrNode LazyTensor::GetViewUpdate(
    const std::shared_ptr<View>& view) const {
  View::IrNode ir_value_updated = view->GetViewIrNode();
  if (ir_value_updated.updated) {
    data()->handle = nullptr;
    data()->tensor_data = c10::nullopt;
  }
  return ir_value_updated;
}

std::shared_ptr<View> LazyTensor::UpdateView(std::shared_ptr<View> view,
                                             ir::Value ir_value) const {
  if (ir_value.shape().dimensions() != view->shape().dimensions()) {
    LTC_CHECK_EQ(lazy_tensors::util::Multiply<lazy_tensors::int64>(
                     ir_value.shape().dimensions()),
                 lazy_tensors::util::Multiply<lazy_tensors::int64>(
                     view->shape().dimensions()));

    ViewInfo view_info(ViewInfo::Type::kReshape, ir_value.shape(),
                       view->shape());
    view = view->CreateSubView(view_info.shape, view_info);
  }
  view->Update(std::move(ir_value));
  return view;
}

void LazyTensor::SetSubView(ViewInfo view_info) const {
  data()->view = data()->view->CreateSubView(view_info.shape, view_info);
  data()->generation += 1;
}

void LazyTensor::ModifyCurrentView(ViewInfo view_info) const {
  if (data()->view != nullptr) {
    data()->view = data()->view->CreateSubView(view_info.shape, view_info);
    return;
  }
  // This node is not a view. Since this function is meant to modify a view
  // in place, we need to turn this existing tensor into a view.
  ir::Value ir_value = GetIrValue();
  std::shared_ptr<Alias> alias = std::make_shared<Alias>(ir_value);
  data()->view =
      std::make_shared<View>(ir_value.shape(), alias, std::move(view_info));
  AssignIrValue(ir::Value());
}

std::shared_ptr<View> LazyTensor::CreateView(ViewInfo view_info) const {
  if (data()->view != nullptr) {
    return data()->view->CreateSubView(view_info.shape, view_info);
  }
  // This node is not a view, and creating a view forks the current node into
  // becoming one itself. This means creating an alias with the current IR
  // Node, and using the same alias for the created IR Node.
  ir::Value ir_value = GetIrValue();
  std::shared_ptr<Alias> alias = std::make_shared<Alias>(ir_value);
  ViewInfo this_view_info(ViewInfo::Type::kNoOp, ir_value.shape(),
                          ir_value.shape());
  data()->view = std::make_shared<View>(ir_value.shape(), alias,
                                        std::move(this_view_info));
  AssignIrValue(ir::Value());
  return std::make_shared<View>(view_info.shape, alias, view_info);
}

LazyTensor LazyTensor::CreateViewTensor(ViewInfo view_info) const {
  return Create(CreateView(std::move(view_info)), GetDevice(),
                dtype_optional());
}

at::Tensor LazyTensor::ToTensor(bool detached) {
  at::Tensor tensor;
  c10::optional<at::Tensor> tensor_data = CurrentTensorData();
  if (!tensor_data) {
    DeviceBarrier(GetDevice());
    // The GetDataHandle() call will trigger an ApplyPendingGraph() if an IR
    // Node is available on the tensor.
    std::vector<at::Tensor> tensors =
        DataHandlesToTensors({GetDataHandle()}, dtype());
    tensor = std::move(tensors.front());
    if (!detached) {
      SetTensorData(tensor);
    }
  } else {
    tensor = *tensor_data;
    if (detached) {
      if (data()->ir_value || data()->handle != nullptr ||
          data()->view != nullptr) {
        // If we have other authoritive sources, just drop our reference and
        // transfer it to the caller.
        data()->tensor_data = c10::nullopt;
      } else {
        // Otherwise we need to make a copy to prevent the caller changing our
        // version.
        tensor = CopyTensor(tensor);
      }
    }
  }
  return tensor;
}

void LazyTensor::ShallowCopyTo(LazyTensor* dest) const {
  dest->SetIrValue(GetIrValue());
}

void LazyTensor::SetScalarType(
    c10::optional<at::ScalarType> logical_element_type) {
  data()->logical_element_type = logical_element_type;
}

void LazyTensor::SetTensor(at::Tensor tensor) {
  SetTensorData(tensor);
  data()->view = nullptr;
  data()->handle = nullptr;
  AssignIrValue(ir::Value());
}

void LazyTensor::UpdateFromTensor(at::Tensor tensor, bool sync) {
  if (sync) {
    at::Tensor typed_tensor = CopyTensor(tensor, dtype(), /*copy=*/false);
    SetIrValue(GetIrValueForTensor(typed_tensor, GetDevice()));
  } else {
    SetTensorData(tensor);
    data()->handle = nullptr;
    AssignIrValue(ir::Value());
    if (data()->view != nullptr) {
      ir::Value ir_value = GetIrValueForTensor(tensor, GetDevice());
      data()->view = UpdateView(data()->view, std::move(ir_value));
    }
  }
}

void LazyTensor::UpdateFromTensorOut(at::Tensor tensor) {
  if (data()->view != nullptr &&
      lazy_tensors::ShapeUtil::ElementsIn(shape()) != tensor.numel()) {
    data()->view = nullptr;
  }
  UpdateFromTensor(std::move(tensor), /*sync=*/false);
}

void LazyTensor::UpdateFromTensorOut(const LazyTensor& tensor) {
  if (data()->view != nullptr &&
      lazy_tensors::ShapeUtil::ElementsIn(shape()) !=
          lazy_tensors::ShapeUtil::ElementsIn(tensor.shape())) {
    data()->view = nullptr;
  }
  SetIrValue(tensor.GetIrValue());
}

std::vector<LazyTensor> LazyTensor::GetLiveTensors(const Device* device) {
  return DeviceContextArena::Get()->GetLiveTensors(device);
}

std::vector<lazy_tensors::ComputationClient::DataPtr>
LazyTensor::GatherTensorsData(
    const std::vector<LazyTensor>& tensors,
    lazy_tensors::Span<const size_t> indices,
    lazy_tensors::Span<const lazy_tensors::ComputationClient::DataPtr>
        tensors_data) {
  std::vector<lazy_tensors::ComputationClient::DataPtr> result_tensors_data;
  std::unordered_map<lazy_tensors::int64, size_t> uid_index_map;
  size_t indices_index = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    lazy_tensors::int64 tensor_id = tensors[i].GetUniqueId();
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
      lazy_tensors::ComputationClient::DataPtr handle =
          tensors[i].CurrentDataHandle();
      LTC_CHECK(handle != nullptr);
      result_tensors_data.push_back(std::move(handle));
    }
  }
  return result_tensors_data;
}

std::vector<at::Tensor> LazyTensor::GetTensorsOpByOp(
    std::vector<LazyTensor>* tensors) {
  SyncTensorsConfig config;
  config.force_ltc_data = false;
  SyncTensorCollection coll = CollectSyncTensors(*tensors, config);
  std::vector<lazy_tensors::ComputationClient::DataPtr> async_tensors_data;
  if (!coll.indices.empty()) {
    DebugUtil::SaveTensorsGraphInfo("GetTensorsOpByOp", *tensors,
                                    &coll.indices);

    std::vector<ir::Value> roots = CollectRoots(*tensors, coll.indices);
    async_tensors_data =
        OpByOpExecutor::Get()->Execute(roots, coll.device.ToString(), {});
  }

  std::vector<lazy_tensors::ComputationClient::DataPtr> tensors_data =
      GatherTensorsData(*tensors, coll.indices, async_tensors_data);
  std::vector<lazy_tensors::Literal> literals =
      lazy_tensors::ComputationClient::Get()->TransferFromServer(tensors_data);

  return FetchTensors(tensors, tensors_data, &coll.indices);
}

std::vector<at::Tensor> LazyTensor::GetTensors(
    std::vector<LazyTensor>* tensors) {
  LTC_VLOG(4) << "Trying to get the value of " << tensors->size()
              << " tensor(s)";
  static const bool op_by_op =
      lazy_tensors::sys_util::GetEnvBool("GET_TENSORS_OPBYOP", false);
  return op_by_op ? GetTensorsOpByOp(tensors) : GetTensorsFused(tensors);
}

std::vector<at::Tensor> LazyTensor::GetTensorsFused(
    std::vector<LazyTensor>* tensors) {
  SyncTensorsConfig config;
  config.force_ltc_data = false;
  auto async = SyncTensorsGraphInternal(tensors, {}, config);
  if (async != nullptr) {
    async->mwait.Wait();
  }
  std::vector<lazy_tensors::ComputationClient::DataPtr> tensors_data =
      GatherTensorsData(
          *tensors,
          async != nullptr ? async->indices
                           : lazy_tensors::Span<const size_t>(),
          async != nullptr
              ? async->tensors_data
              : lazy_tensors::Span<
                    const lazy_tensors::ComputationClient::DataPtr>());
  return FetchTensors(tensors, tensors_data,
                      async != nullptr ? &async->indices : nullptr);
}

std::vector<at::Tensor> LazyTensor::FetchTensors(
    std::vector<LazyTensor>* tensors,
    lazy_tensors::Span<const lazy_tensors::ComputationClient::DataPtr>
        tensors_data,
    const std::vector<size_t>* indices) {
  std::vector<at::Tensor> results;
  size_t literals_index = 0;
  size_t sync_index = 0;
  results.reserve(tensors->size());
  for (size_t i = 0; i < tensors->size(); ++i) {
    if (indices != nullptr && sync_index < indices->size() &&
        i == (*indices)[sync_index]) {
      results.push_back(lazy_tensors::MakeTensorFromComputationData(
          tensors_data[literals_index], (*tensors)[i].dtype()));
      ++literals_index;
      ++sync_index;
    } else {
      c10::optional<at::Tensor> tensor_data = (*tensors)[i].CurrentTensorData();
      if (tensor_data) {
        results.push_back(*tensor_data);
      } else {
        LTC_CHECK_LT(literals_index, tensors_data.size());
        results.push_back(lazy_tensors::MakeTensorFromComputationData(
            tensors_data[literals_index], (*tensors)[i].dtype()));
        ++literals_index;
      }
    }
  }
  return results;
}

std::vector<LazyTensor> LazyTensor::CreateTensors(
    const std::vector<at::Tensor>& tensors,
    const std::vector<std::string>& devices) {
  std::vector<lazy_tensors::ComputationClient::DataPtr> handles =
      CreateTensorsData(tensors, devices);
  std::vector<LazyTensor> ltc_tensors;
  for (size_t i = 0; i < handles.size(); ++i) {
    ltc_tensors.push_back(
        Create(std::move(handles[i]), tensors[i].scalar_type()));
  }
  return ltc_tensors;
}

ir::Value LazyTensor::CreateTensorNode(
    lazy_tensors::ComputationClient::DataPtr data, bool read_only) const {
  data->SetInfo(std::make_shared<DeviceDataInfo>(GetUniqueId(), read_only));
  return ir::MakeNode<ir::ops::DeviceData>(std::move(data));
}

std::vector<LazyTensor> LazyTensor::MakeOutputTensors(ir::NodePtr node) const {
  std::vector<LazyTensor> tensors;
  tensors.reserve(node->num_outputs());
  for (size_t i = 0; i < node->num_outputs(); ++i) {
    tensors.push_back(CreateFrom(ir::Value(node, i)));
  }
  return tensors;
}

LazyTensor LazyTensor::CopyTensorToDevice(const Device& device) {
  // TODO: This can be optimized.
  return Create(ToTensor(/*detached=*/true), device);
}

ir::Value LazyTensor::MaybeCastIrValue(
    ir::Value ir_value, const Device& device,
    c10::optional<at::ScalarType> logical_element_type) const {
  if (!logical_element_type) {
    logical_element_type = dtype_optional();
  }
  if (logical_element_type &&
      RequiresRawTypeCasting(*logical_element_type, &device)) {
    ir_value = ir::MakeNode<ir::ops::Cast>(ir_value, *logical_element_type);
  }
  return ir_value;
}

LazyTensor LazyTensor::CreateFrom(ir::Value ir_value) const {
  ir_value = MaybeCastIrValue(std::move(ir_value), GetDevice(),
                              /*logical_element_type=*/c10::nullopt);
  return Create(std::move(ir_value), GetDevice(), dtype_optional());
}

LazyTensor LazyTensor::CreateFrom(ir::Value ir_value,
                                  const Device& device) const {
  ir_value = MaybeCastIrValue(std::move(ir_value), device,
                              /*logical_element_type=*/c10::nullopt);
  return Create(std::move(ir_value), device, dtype_optional());
}

LazyTensor LazyTensor::CreateFrom(ir::Value ir_value,
                                  at::ScalarType logical_element_type) const {
  ir_value =
      MaybeCastIrValue(std::move(ir_value), GetDevice(), logical_element_type);
  return Create(std::move(ir_value), GetDevice(), logical_element_type);
}

LazyTensor LazyTensor::CreateFrom(
    ir::Value ir_value,
    c10::optional<at::ScalarType> logical_element_type_opt) const {
  ir_value = MaybeCastIrValue(std::move(ir_value), GetDevice(),
                              logical_element_type_opt);
  return Create(std::move(ir_value), GetDevice(), logical_element_type_opt);
}

LazyTensor LazyTensor::CreateFrom(ir::Value ir_value, const Device& device,
                                  at::ScalarType logical_element_type) const {
  ir_value =
      MaybeCastIrValue(std::move(ir_value), device, logical_element_type);
  return Create(std::move(ir_value), device, logical_element_type);
}

void LazyTensor::ApplyPendingGraph() {
  DeviceBarrier(GetDevice());
  // This method is called to ensure that the tensor data is available on
  // device, so that a call to CurrentDataHandle() returns a valid pointer.
  if (CurrentDataHandle() == nullptr) {
    std::vector<LazyTensor> tensors({*this});
    SyncTensorsGraph(&tensors, {}, /*wait=*/true, /*sync_ltc_data=*/false);
  }
}

LazyTensor::SyncTensorCollection LazyTensor::CollectSyncTensors(
    const std::vector<LazyTensor>& tensors, const SyncTensorsConfig& config) {
  lazy_tensors::util::Unique<Device> unique_device;
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
  std::vector<std::string> devices;
  std::vector<size_t> at_tensor_index;
  std::unordered_set<lazy_tensors::int64> tensor_ids;
  // The force_ltc_data controls aliasing compilation, so effectively the same
  // graph with on/off force_ltc_data should not match, hash wise.
  coll.hash = lazy_tensors::util::MHash(config.force_ltc_data);
  coll.config = config;
  coll.device = *unique_device;
  coll.indices.reserve(tensors.size());
  LTC_VLOG(4) << "Waiting on device barrier for device " << coll.device
              << " ...";
  {
    LTC_TIMED("DeviceLockWait");
    coll.unlocker = LockDevices(unique_device.AsSet());
  }
  LTC_VLOG(4) << "Waiting on device barrier for device " << coll.device
              << " done!";
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensor_ids.insert(tensors[i].GetUniqueId()).second &&
        tensors[i].CurrentDataHandle() == nullptr) {
      ir::Value ir_value = tensors[i].CurrentIrValue();
      if (ir_value) {
        if (ShouldSyncIrValue(ir_value)) {
          // Add only tensors which need to be synced.
          coll.hash =
              lazy_tensors::util::HashCombine(coll.hash, ir_value.hash());
          coll.indices.push_back(i);
        }
      } else if (config.force_ltc_data) {
        // The tensor only has at::Tensor data. We need to queue it for a
        // device upload.
        c10::optional<at::Tensor> tensor_data = tensors[i].CurrentTensorData();
        LTC_CHECK(tensor_data);
        at_tensors.push_back(*tensor_data);
        devices.push_back(tensors[i].GetDevice().ToString());
        at_tensor_index.push_back(i);
      }
    }
  }
  // Mix the hash with the resource domain hashes as compile handles are only
  // valid within a domain (usually a single host).
  coll.hash = lazy_tensors::util::MHash(
      coll.hash, lazy_tensors::ComputationClient::Get()->GetResourceDomain(
                     coll.device.ToString()));
  if (!at_tensors.empty()) {
    LTC_COUNTER("SyncTensorsToData", at_tensors.size());
    std::vector<lazy_tensors::ComputationClient::DataPtr> handles =
        CreateTensorsData(at_tensors, devices);
    for (size_t i = 0; i < handles.size(); ++i) {
      // If we are here, it means that the IR Value for the tensor is not
      // present. Also, we uploaded the at::Tensor data to the device, but such
      // data is still valid so we leave it live on the lazy tensor (so that a
      // following ToTensor() does not need to fetch it from device).
      tensors[at_tensor_index[i]].data()->handle = std::move(handles[i]);
    }
  }
  LTC_VLOG(4) << "Tensors graph hash " << lazy_tensors::util::HexHash(coll.hash)
              << " on device " << coll.device;
  return coll;
}

LazyTensor::ComputationCache::TypePtr LazyTensor::LookupCachedCompile(
    const std::vector<LazyTensor>& tensors, const lazy_tensors::hash_t& hash) {
  ComputationCache::TypePtr cached_computation =
      GetComputationCache()->Get(hash);
  if (cached_computation == nullptr) {
    LTC_COUNTER("UncachedCompile", 1);
    return nullptr;
  }
  LTC_COUNTER("CachedCompile", 1);
  return cached_computation;
}

std::shared_ptr<LazyTensor::Async> LazyTensor::TryRunCachedSync(
    std::vector<LazyTensor>* tensors, SyncTensorCollection* coll,
    PostOrderData* po_data) {
  ComputationCache::TypePtr cached_computation =
      LookupCachedCompile(*tensors, coll->hash);
  if (cached_computation == nullptr) {
    return nullptr;
  }
  LTC_VALUE_METRIC("TensorsGraphSize", po_data->post_order.size());
  LTC_VLOG(5) << "TensorsGraphSize=" << po_data->post_order.size();

  return ScheduleSyncTensorsGraph(
      tensors, coll, std::move(po_data->parameters_data),
      coll->device.ToString(), std::move(cached_computation));
}

LazyTensor::ComputationCache* LazyTensor::GetComputationCache() {
  static const size_t kMaxCacheSize =
      lazy_tensors::sys_util::GetEnvInt("COMPILATION_CACHE_SIZE", 1024);
  static ComputationCache* cache = new ComputationCache(kMaxCacheSize);
  return cache;
}

LazyTensor::PostOrderData LazyTensor::RunPostOrder(
    const std::vector<LazyTensor>& tensors,
    lazy_tensors::Span<const size_t> indices) {
  std::vector<const ir::Node*> roots;
  roots.reserve(indices.size());
  for (auto index : indices) {
    ir::Value ir_value = tensors.at(index).CurrentIrValue();
    roots.push_back(ir_value.node.get());
  }
  PostOrderData po_data;
  po_data.post_order = ir::Util::ComputePostOrder(roots, &po_data.emission_map);
  std::unordered_map<lazy_tensors::client::Data::OpaqueHandle, size_t>
      data_handles;
  for (auto node : po_data.post_order) {
    const ir::ops::DeviceData* device_data = ir::ops::DeviceData::Cast(node);
    if (device_data != nullptr) {
      lazy_tensors::client::Data::OpaqueHandle handle =
          device_data->data()->GetOpaqueHandle();
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

std::vector<ir::Value> LazyTensor::CollectRoots(
    const std::vector<LazyTensor>& tensors,
    lazy_tensors::Span<const size_t> indices) {
  std::vector<ir::Value> roots;
  roots.reserve(indices.size());
  for (auto index : indices) {
    roots.push_back(tensors.at(index).CurrentIrValue());
  }
  return roots;
}

std::vector<lazy_tensors::ComputationClient::DataPtr>
LazyTensor::FetchTensorData(std::vector<LazyTensor>* tensors,
                            const SyncTensorsConfig& config,
                            lazy_tensors::Span<const size_t> indices) {
  std::vector<lazy_tensors::ComputationClient::DataPtr> tensors_data;
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
    lazy_tensors::ComputationClient::DataPtr handle =
        tensor.CurrentDataHandle();
    if (handle == nullptr && config.force_ltc_data) {
      const Device& tensor_device = tensor.GetDevice();
      lazy_tensors::Shape shape =
          MakeShapeWithDeviceLayout(tensor.shape(), tensor_device.hw_type);
      handle = lazy_tensors::ComputationClient::Get()->CreateDataPlaceholder(
          tensor_device.ToString(), std::move(shape));
      tensor.SetDataHandle(handle, config.sync_ltc_data);
    }
    tensors_data.emplace_back(std::move(handle));
  }
  return tensors_data;
}

std::shared_ptr<LazyTensor::Async> LazyTensor::ScheduleSyncTensorsGraph(
    SyncTensorCollection* coll,
    std::vector<lazy_tensors::ComputationClient::DataPtr> parameters_data,
    std::vector<lazy_tensors::ComputationClient::DataPtr> tensors_data,
    ComputationCache::TypePtr cached_computation) {
  std::shared_ptr<Async> async = std::make_shared<Async>(
      coll, std::move(parameters_data), std::move(tensors_data),
      std::move(cached_computation));

  auto syncfn = [async, hash = coll->hash]() {
    lazy_tensors::ComputationClient::ExecuteComputationOptions options;
    try {
      LTC_VLOG(3) << "Executing IR graph hash "
                  << lazy_tensors::util::HexHash(hash) << " on device "
                  << async->device << " ...";
      auto results = lazy_tensors::ComputationClient::Get()->ExecuteComputation(
          *async->cached_computation->computation, async->parameters_data,
          async->device, options);
      LTC_VLOG(3) << "Executing IR graph hash "
                  << lazy_tensors::util::HexHash(hash) << " on device "
                  << async->device << " done!";

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
        unlocker.SetStatus(exptr);
      }
      throw;
    }
  };

  lazy_tensors::env::ScheduleIoClosure(
      async->mwait.Completer(std::move(syncfn)));
  return async;
}

std::shared_ptr<LazyTensor::Async> LazyTensor::ScheduleSyncTensorsGraph(
    std::vector<LazyTensor>* tensors, SyncTensorCollection* coll,
    std::vector<lazy_tensors::ComputationClient::DataPtr> parameters_data,
    std::string device, ComputationCache::TypePtr cached_computation) {
  auto tensors_data = FetchTensorData(tensors, coll->config, coll->indices);
  return ScheduleSyncTensorsGraph(coll, std::move(parameters_data),
                                  std::move(tensors_data),
                                  std::move(cached_computation));
}

void LazyTensor::SyncTensorsGraph(std::vector<LazyTensor>* tensors,
                                  lazy_tensors::Span<const std::string> devices,
                                  bool wait, bool sync_ltc_data) {
  LTC_VLOG(4) << "Trying to sync the value of " << tensors->size()
              << " tensor(s)";
  static const bool op_by_op =
      lazy_tensors::sys_util::GetEnvBool("SYNC_TENSORS_OPBYOP", false);
  SyncTensorsConfig config;
  config.sync_ltc_data = sync_ltc_data;
  if (op_by_op) {
    OpByOpAsync async = SyncTensorsGraphOpByOp(tensors, devices, config);
    if (wait) {
      async.Wait();
    }
  } else {
    auto async = SyncTensorsGraphInternal(tensors, devices, config);
    if (wait && async != nullptr) {
      async->mwait.Wait();
    }
  }
}

void LazyTensor::SyncLiveTensorsGraph(
    const Device* device, lazy_tensors::Span<const std::string> devices,
    bool wait) {
  auto tensors = GetLiveTensors(device);
  LTC_VLOG(4) << tensors.size() << " live tensors: devices=("
              << lazy_tensors::StrJoin(devices, ",") << ")";
  SyncTensorsGraph(&tensors, devices, wait, /*sync_ltc_data=*/true);
}

void LazyTensor::MarkStep(const Device& device) {
  LTC_COUNTER("MarkStep", 1);
  DeviceContextArena::Get()->MarkStep(device);
  ir::ScopePusher::ResetScopes();
  g_tls_data.Reset();
}

void LazyTensor::WaitDeviceOps(lazy_tensors::Span<const std::string> devices) {
  std::set<Device> wait_devices;
  if (!devices.empty()) {
    for (auto& device_str : devices) {
      wait_devices.insert(Device(device_str));
    }
  } else {
    for (auto& device_str :
         lazy_tensors::ComputationClient::Get()->GetLocalDevices()) {
      wait_devices.insert(Device(device_str));
    }
  }
  // The LockDevices() API returns a vector of
  // lazy_tensors::util::ExceptionCleanup object, which is going to be freed
  // immediately, turning this operation into a lock barrier.
  LockDevices(wait_devices);
}

LazyTensor::OpByOpAsync LazyTensor::SyncTensorsGraphOpByOp(
    std::vector<LazyTensor>* tensors,
    lazy_tensors::Span<const std::string> devices,
    const SyncTensorsConfig& config) {
  struct Async {
    explicit Async(
        SyncTensorCollection coll,
        std::vector<lazy_tensors::ComputationClient::DataPtr> tensors_data,
        std::vector<ir::Value> roots,
        lazy_tensors::Span<const std::string> devices)
        : coll(std::move(coll)),
          tensors_data(std::move(tensors_data)),
          roots(std::move(roots)),
          devices(devices.begin(), devices.end()) {}

    SyncTensorCollection coll;
    std::vector<lazy_tensors::ComputationClient::DataPtr> tensors_data;
    std::vector<ir::Value> roots;
    std::vector<std::string> devices;
  };

  SyncTensorCollection coll = CollectSyncTensors(*tensors, config);
  DebugUtil::SaveTensorsGraphInfo("SyncTensorsGraphOpByOp", *tensors,
                                  &coll.indices);

  std::vector<ir::Value> roots = CollectRoots(*tensors, coll.indices);
  auto tensors_data = FetchTensorData(tensors, coll.config, coll.indices);
  auto async = std::make_shared<Async>(std::move(coll), std::move(tensors_data),
                                       std::move(roots), devices);

  auto syncfn = [async]() -> int {
    try {
      LTC_VLOG(3) << "Executing (OpByOp) IR graph hash "
                  << lazy_tensors::util::HexHash(async->coll.hash)
                  << " on device " << async->coll.device << " ...";
      std::vector<lazy_tensors::ComputationClient::DataPtr> results =
          OpByOpExecutor::Get()->Execute(
              async->roots, async->coll.device.ToString(), async->devices);
      LTC_VLOG(3) << "Executing (OpByOp) IR graph hash "
                  << lazy_tensors::util::HexHash(async->coll.hash)
                  << " on device " << async->coll.device << " done!";

      for (size_t i = 0; i < results.size(); ++i) {
        if (async->tensors_data[i] != nullptr) {
          async->tensors_data[i]->Assign(*results[i]);
        }
      }
    } catch (...) {
      std::exception_ptr exptr = std::current_exception();
      for (auto& unlocker : async->coll.unlocker) {
        unlocker.SetStatus(exptr);
      }
      throw;
    }
    return 0;
  };
  OpByOpAsync async_op(std::move(syncfn));
  return async_op.Schedule();
}

void LazyTensor::BuildInputOutputAliases(
    const std::vector<LazyTensor>& tensors,
    lazy_tensors::Span<const size_t> indices,
    ir::LoweringContext* lowering_ctx) {
  std::unordered_map<lazy_tensors::int64, size_t> output_tensor_id_map;
  for (size_t i = 0; i < indices.size(); ++i) {
    size_t tensor_index = indices[i];
    lazy_tensors::int64 tensor_id = tensors[tensor_index].GetUniqueId();
    output_tensor_id_map[tensor_id] = i;
  }
  const std::vector<lazy_tensors::ComputationClient::DataPtr>& parameters_data =
      lowering_ctx->GetParametersData();
  std::vector<ssize_t> alias_map(indices.size(), -1);
  for (size_t i = 0; i < parameters_data.size(); ++i) {
    DeviceDataInfo* data_info =
        dynamic_cast<DeviceDataInfo*>(parameters_data[i]->info());
    if (data_info != nullptr && !data_info->read_only) {
      auto it = output_tensor_id_map.find(data_info->tensor_id);
      if (it != output_tensor_id_map.end()) {
        size_t output_index = it->second;
        const lazy_tensors::Shape& root_shape =
            lowering_ctx->GetResultShape(output_index);
        if (lazy_tensors::Shape(parameters_data[i]->shape()) == root_shape &&
            alias_map[output_index] < 0) {
          lowering_ctx->SetUpAlias(
              {static_cast<lazy_tensors::int64>(output_index)}, i, {});
          alias_map[output_index] = i;

          LTC_VLOG(6) << "Aliased paramter " << i << " with output "
                      << output_index << ": "
                      << lazy_tensors::Shape(parameters_data[i]->shape());
        }
      }
    }
  }
  LTC_VALUE_METRIC("InputOutputAliasCount", alias_map.size());
}

LazyTensor::CompilationResult LazyTensor::Compile(
    const std::vector<LazyTensor>& tensors,
    lazy_tensors::Span<const std::string> devices,
    const SyncTensorCollection& coll, PostOrderData* po_data) {
  static const bool enable_aliasing =
      lazy_tensors::sys_util::GetEnvBool("ENABLE_PARAM_ALIASING", false);
  auto lowering_ctx = ir::LoweringContext::Create(
      "SyncTensorsGraph", coll.device, po_data->post_order,
      std::move(po_data->emission_map));
  for (auto index : coll.indices) {
    ir::Value ir_value = tensors[index].CurrentIrValue();
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

  auto computation = ConsumeValue(lowering_ctx->Build());
  lazy_tensors::ProgramShape program_shape =
      ConsumeValue(computation->GetProgramShape());
  lazy_tensors::Shape shape =
      MakeShapeWithDeviceLayout(program_shape.result(), coll.device.hw_type);

  std::vector<lazy_tensors::ComputationClient::CompileInstance> instances;
  instances.push_back(
      {std::move(computation), coll.device.ToString(),
       lazy_tensors::ComputationClient::Get()->GetCompilationDevices(
           coll.device.ToString(), devices),
       &shape});

  LTC_VLOG(3) << "Compiling IR graph hash "
              << lazy_tensors::util::HexHash(coll.hash) << " on device "
              << coll.device << " ...";
  std::vector<std::shared_ptr<lazy_tensors::ComputationClient::Computation>>
      computations =
          lazy_tensors::ComputationClient::Get()->Compile(std::move(instances));
  LTC_VLOG(3) << "Compiling IR graph hash "
              << lazy_tensors::util::HexHash(coll.hash) << " on device "
              << coll.device << " done!";
  LTC_CHECK_EQ(program_shape.parameters_size(),
               po_data->parameters_data.size());

  return {/*device=*/coll.device,
          /*emitted_nodes=*/lowering_ctx->GetEmittedNodeCount(),
          /*computation=*/std::move(computations.front()),
          /*parameters_data=*/std::move(po_data->parameters_data)};
}

std::shared_ptr<LazyTensor::Async> LazyTensor::SyncTensorsGraphInternal(
    std::vector<LazyTensor>* tensors,
    lazy_tensors::Span<const std::string> devices,
    const SyncTensorsConfig& config) {
  SyncTensorCollection coll = CollectSyncTensors(*tensors, config);
  if (coll.indices.empty()) {
    return nullptr;
  }
  DebugUtil::SaveTensorsGraphInfo("ScheduleSyncTensorsGraph", *tensors,
                                  &coll.indices);

  PostOrderData po_data = RunPostOrder(*tensors, coll.indices);
  coll.hash = lazy_tensors::util::HashCombine(
      coll.hash, lazy_tensors::util::Hash(po_data.parameter_sequence));
  LTC_VLOG(4) << "Parameter sequence graph hash "
              << lazy_tensors::util::HexHash(coll.hash);
  std::shared_ptr<Async> async = TryRunCachedSync(tensors, &coll, &po_data);
  if (async != nullptr) {
    return async;
  }

  CompilationResult compile_result = Compile(*tensors, devices, coll, &po_data);

  LTC_VALUE_METRIC("TensorsGraphSize", compile_result.emitted_nodes);
  LTC_VLOG(5) << "TensorsGraphSize=" << compile_result.emitted_nodes;

  auto cached_computation = std::make_shared<CachedComputation>(
      std::move(compile_result.computation));
  GetComputationCache()->Add(coll.hash, cached_computation);

  return ScheduleSyncTensorsGraph(
      tensors, &coll, std::move(compile_result.parameters_data),
      compile_result.device.ToString(), std::move(cached_computation));
}

lazy_tensors::int64 LazyTensor::GetNextTensorId() {
  static std::atomic<lazy_tensors::int64>* id_generator =
      new std::atomic<lazy_tensors::int64>(1);
  return id_generator->fetch_add(1);
}

ir::Value LazyTensor::GetRngSeed(const Device& device) {
  return DeviceContextArena::Get()->GetRngSeed(device);
}

void LazyTensor::SetRngSeed(const Device& device, lazy_tensors::uint64 seed) {
  DeviceContextArena::Get()->SetRngSeed(device, seed);
}

lazy_tensors::uint64 LazyTensor::GetRunningSeed(const Device& device) {
  return DeviceContextArena::Get()->GetRunningSeed(device);
}

}  // namespace torch_lazy_tensors
