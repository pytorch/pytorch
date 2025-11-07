#ifdef USE_C10D_NCCL

#include <nlohmann/json.hpp>
#include <exception>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <c10/util/WaitCounter.h>
#include <c10/util/hash.h>
#include <c10/util/irange.h>
#include <c10/util/thread_name.h>
#include <torch/csrc/cuda/CUDAPluggableAllocator.h>
#include <torch/csrc/cuda/nccl.h>
#include <torch/csrc/distributed/c10d/FlightRecorder.hpp>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/NanCheck.hpp>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/TraceUtils.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/cuda/utils.hpp>
#include <torch/torch.h>
#include <optional>

namespace c10d {

constexpr const char* const kNCCLAbortedCommStoreKey = "NCCLABORTEDCOMM";
using FlightRecorderCUDA = FlightRecorder<at::cuda::CUDAEvent>;

namespace {

#if defined(NCCL_MAJOR) && \
    ((NCCL_MAJOR > 2) || (NCCL_MAJOR == 2) && (NCCL_MINOR >= 10))
#define NCCL_HAS_AVG 1
#endif // NCCL version >= 2.10

// NCCL op mapping
const std::map<ReduceOp::RedOpType, ncclRedOp_t> ncclOp = {
    {ReduceOp::MIN, ncclMin},
    {ReduceOp::MAX, ncclMax},
    {ReduceOp::SUM, ncclSum},
    {ReduceOp::PRODUCT, ncclProd},
#ifdef NCCL_HAS_AVG
    {ReduceOp::AVG, ncclAvg},
#endif // NCCL_HAS_AVG
};

inline bool isUnsupportedFloat8(at::ScalarType t) {
  return (
      t == at::ScalarType::Float8_e5m2fnuz ||
      t == at::ScalarType::Float8_e4m3fnuz ||
      t == at::ScalarType::Float8_e8m0fnu
#ifndef NCCL_SUPPORTS_FP8
      || t == at::ScalarType::Float8_e5m2 || t == at::ScalarType::Float8_e4m3fn
#endif
  );
}

#ifdef ENABLE_NCCL_PREMUL_SUM_SUPPORT
template <typename T, ncclDataType_t dataType>
ncclRedOpRAII unpackPreMulSum(
    const ReduceOp& reduceOp,
    const ncclComm_t& comm) {
  const auto* preMulSupplement =
      reinterpret_cast<NCCLPreMulSumSupplement*>(reduceOp.supplement_.get());
  ncclRedOp_t preMulSum{};
  bool has_tensor = preMulSupplement->tensor_factor.defined();
  auto residence = has_tensor ? ncclScalarDevice : ncclScalarHostImmediate;
  const T* ptr_factor = has_tensor
      ? preMulSupplement->tensor_factor.const_data_ptr<T>()
      : nullptr;
  T scalar_factor = T(preMulSupplement->double_factor);
  ncclRedOpCreatePreMulSum(
      &preMulSum,
      // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/ops.html#ncclredopcreatepremulsum
      // tells us that the scalar input is strictly a multiplier.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      /*scalar=*/has_tensor ? const_cast<T*>(ptr_factor) : &scalar_factor,
      dataType,
      residence,
      comm);
  return ncclRedOpRAII(preMulSum, comm);
}
#endif // ENABLE_NCCL_PREMUL_SUM_SUPPORT

ncclRedOpRAII getNcclReduceOp(
    const ReduceOp& reduceOp,
    at::Tensor& input,
    const ncclDataType_t& dataType,
    const ncclComm_t& comm) {
  try {
    if (input.scalar_type() == at::kBool) {
      if (reduceOp == ReduceOp::SUM) {
        // For bool tensors, map sum to max, which both represent a bitwise or.
        // This is to prevent overflow issues with sum, since we use uint8 to
        // represent a bool (see ncclDataType mapping).
        return ncclMax;
      }
#ifdef NCCL_HAS_AVG
      if (reduceOp == ReduceOp::AVG) {
        C10_THROW_ERROR(
            TypeError, "Cannot use ReduceOp.AVG with boolean inputs");
      }
#endif // NCCL_HAS_AVG
    }
    if (reduceOp == ReduceOp::PREMUL_SUM) {
#ifdef ENABLE_NCCL_PREMUL_SUM_SUPPORT
      switch (dataType) {
        case ncclHalf:
          return unpackPreMulSum<at::Half, ncclHalf>(reduceOp, comm);
        case ncclFloat:
          return unpackPreMulSum<float, ncclFloat>(reduceOp, comm);
        case ncclBfloat16:
          return unpackPreMulSum<float, ncclBfloat16>(reduceOp, comm);
        case ncclDouble:
          return unpackPreMulSum<double, ncclDouble>(reduceOp, comm);
        default:
          C10_THROW_ERROR(
              TypeError,
              "PreMulSum Data type must be half, float, bfloat16 or double");
          return ncclRedOp_t{};
      }
#else
      C10_THROW_ERROR(ValueError, "PreMulSum requires NCCL>=2.11.1");
#endif // ENABLE_NCCL_PREMUL_SUM_SUPPORT
    }
    return ncclOp.at(reduceOp);
  } catch (const std::out_of_range&) {
    switch (reduceOp) {
      case ReduceOp::AVG:
        C10_THROW_ERROR(
            ValueError,
            c10::str(
                "AVG requires NCCL 2.10+. The current version is ",
                NCCL_MAJOR,
                ".",
                NCCL_MINOR));
        break;
      case ReduceOp::BAND:
        C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BAND with NCCL");
        break;
      case ReduceOp::BOR:
        C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BOR with NCCL");
        break;
      case ReduceOp::BXOR:
        C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BXOR with NCCL");
        break;
      default:
        C10_THROW_ERROR(ValueError, "Unhandled ReduceOp");
        break;
    }
  }
}

// Get a key string from device
inline std::string getKeyFromDevice(const at::Device& device) {
  return std::to_string(device.index());
}

std::string getKeySendRecv(int myRank, int peer) {
  int lowRank = myRank < peer ? myRank : peer;
  int highRank = myRank < peer ? peer : myRank;
  std::string sendRecvPair =
      std::to_string(lowRank) + ":" + std::to_string(highRank);
  return sendRecvPair;
}

// Get device from tensor
inline at::Device getDevice(at::Tensor& tensor) {
  return tensor.device();
}

// [Sync Streams] Helper that lets the input ncclStreams to wait for the current
// stream. NCCL communications run on ncclStreams, but input tensors are
// allocated on different streams (i.e., current streams). Communications on
// ncclStreams cannot start before pending input tensor ops on current streams
// finish. Otherwise, ops on two streams might read/write same tensors
// concurrently.
//
// The synchronization above alone is not enough. We also need to make sure
// input tensors are not freed before their usages on ncclStreams finish. This
// can be achieved by calling c10::cuda::CUDACachingAllocator::recordStream,
// which remembers the usage stream (ncclStream), creates an event on the usage
// stream when GC attempts to free the input tensor, and delays GC until that
// event is done.
void syncStream(
    at::Device& device,
    at::cuda::CUDAEvent& ncclEvent,
    at::cuda::CUDAStream& ncclStream) {
  ncclEvent.record(at::cuda::getCurrentCUDAStream(device.index()));
  ncclEvent.block(ncclStream);
}

std::string getNcclAbortedCommStoreKey(const std::string& ncclIdStr) {
  return std::string(kNCCLAbortedCommStoreKey) + ":" + ncclIdStr;
}

// Returns exception's what() given an exception_ptr instance.
std::string getExceptionMsgFromExceptionPtr(
    const std::exception_ptr& exceptionPtr) {
  TORCH_CHECK(exceptionPtr != nullptr);
  try {
    std::rethrow_exception(exceptionPtr);
  } catch (const std::exception& e) {
    return e.what();
  } catch (...) {
    return "Unknown exception type";
  }
}

inline void errorIfCapturingNonCapturableNCCL(c10::cuda::CaptureStatus status) {
  // parentheses avoid some compiler warnings
  static const uint64_t min_version =
      (((uint64_t)2) << 32) + (((uint64_t)9) << 16) + ((uint64_t)6);
  static const uint64_t cur_version = torch::cuda::nccl::version();
  if (cur_version < min_version) {
    TORCH_CHECK_WITH(
        NotImplementedError,
        status == c10::cuda::CaptureStatus::None,
        "Capturing NCCL collectives is only allowed with NCCL >= 2.9.6");
  }
}

// When TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK is set, all tensors (no
// matter how they have been allocated) are registered with all NCCL comms.
bool shouldAllCommunicatorsRegisterAllTensors() {
#ifdef NCCL_HAS_COMM_REGISTER
  static const bool flag = [] {
    const bool flag =
        getCvarBool(TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK, false);
    if (flag &&
        c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::
            expandable_segments()) {
      LOG(INFO)
          << "disables TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK because it is not compatible with CUDA allocator expandable segments mode.";
      return false;
    }
    return flag;
  }();
  return flag;
#else
  return false;
#endif // NCCL_HAS_COMM_REGISTER
}

} // namespace

// Map each communicator to the memory pools registered with it.
// This map is used when the caching allocator allocates or frees segments, in
// order to register or deregister them with the relevant NCCL communicators.
// There are two modes to do so:
// - If TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=1 then *ALL* segments
//   will be registered with *ALL* NCCL communicators (for the same device),
//   even if they were allocated with cudaMalloc (which NCCL doesn't like).
// - If a MemPool is explicitly registered with a ProcessGroup, then all its
//   segments (current and future) will be registered with the NCCL communicator
//   corresponding to the pool's device. This works best if the MemPool is set
//   up to use ncclMemAlloc (which is exposed by the ProcessGroup).
// Implementation notes:
// - We cannot reuse devNCCLCommMap_ in each ProcessGroup because the key may be
//   ranks rather than device in point-to-point case.
// - This map has also to be maintained as global variable since the register
//   hooks are called outside the scope of any PG, thus we need traverse
//   communicators in all PGs.

// MemPoolSet has ids of mempools used with this communicator, and whether they
// were registered with window APIs or not
using MemPoolSet = std::unordered_set<
    std::tuple<c10::cuda::MempoolId_t, bool>,
    c10::hash<std::tuple<c10::cuda::MempoolId_t, bool>>>;
static std::unordered_map<std::shared_ptr<NCCLComm>, MemPoolSet>
    ncclCommMemPoolMap;
static std::mutex ncclCommMemPoolMapMutex;

std::atomic<bool> ProcessGroupNCCL::shouldDump_(false);

static void cacheAllocatorRegisterHook(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  // Register after SEGMENT_ALLOC
  if (te.action_ !=
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_ALLOC) {
    return;
  }

  std::lock_guard<std::mutex> lock(ncclCommMemPoolMapMutex);
  for (auto& [ncclComm, memPools] : ncclCommMemPoolMap) {
    if (te.device_ == ncclComm->getDeviceIndex()) {
      bool symm = false;
      bool should_register = shouldAllCommunicatorsRegisterAllTensors();
      auto it =
          std::find_if(memPools.begin(), memPools.end(), [&](const auto& tup) {
            return std::get<0>(tup) == te.mempool_;
          });
      if (it != memPools.end()) {
        should_register = true;
        symm = std::get<1>(*it);
      }
      if (should_register) {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        ncclComm->registerSegment(
            reinterpret_cast<void*>(te.addr_),
            te.size_,
            /*errorOnRereg*/ false,
            /*window*/ symm);
      }
    }
  }
}

static void cacheAllocatorDeregisterHook(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  // deregister before SEGMENT_FREE
  if (te.action_ !=
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_FREE) {
    return;
  }

  std::lock_guard<std::mutex> lock(ncclCommMemPoolMapMutex);
  for (auto& [ncclComm, memPools] : ncclCommMemPoolMap) {
    if (te.device_ == ncclComm->getDeviceIndex()) {
      bool symm = false;
      bool should_register = shouldAllCommunicatorsRegisterAllTensors();
      auto it =
          std::find_if(memPools.begin(), memPools.end(), [&](const auto& tup) {
            return std::get<0>(tup) == te.mempool_;
          });
      if (it != memPools.end()) {
        should_register = true;
        symm = std::get<1>(*it);
      }
      if (should_register) {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        ncclComm->deregisterSegment(reinterpret_cast<void*>(te.addr_), symm);
      }
    }
  }
}

static void attachAllocatorHooks() {
  static auto flag [[maybe_unused]] = [] {
    // Attaching hooks fails if CUDACachingAllocator is not initialized, so
    // Init for CUDA is called (and is a no-op if CUDA is already
    // initialized).
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
    c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker(
        &cacheAllocatorRegisterHook);
    c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker(
        &cacheAllocatorDeregisterHook);
    return true;
  }();
}

static std::
    unordered_map<std::string, std::unordered_map<std::string, std::string>>
    getNCCLCommDumpMap() {
#if (defined(IS_NCCLX) || defined(USE_ROCM)) && defined(NCCL_COMM_DUMP)
  std::unordered_map<
      std::string /* ncclUniqueID */,
      std::unordered_map<std::string, std::string> /* dump from this comm */>
      ncclDumpMap;
  // dump_nccl_trace is only called from the default PG (local_id_=0), but we
  // want to dump from all comms so we need to iterate over ncclCommMemPoolMap,
  // which is static
  std::vector<std::shared_ptr<NCCLComm>> allNCCLComms;
  // within the critical section, we don't want to dump while holding the lock
  // as dump might hang
  {
    std::lock_guard<std::mutex> lock(ncclCommMemPoolMapMutex);
    for (auto& [ncclComm, _] : ncclCommMemPoolMap) {
      allNCCLComms.push_back(ncclComm);
    }
  }
  for (auto& ncclComm : allNCCLComms) {
    ncclDumpMap[ncclComm->getUniqueHash()] = ncclComm->ncclCommDump();
  }
  return ncclDumpMap;
#else
  return std::unordered_map<
      std::string,
      std::unordered_map<std::string, std::string>>();
#endif // (defined(IS_NCCLX) || defined(USE_ROCM)) && defined(NCCL_COMM_DUMP)
}

void reset_nccl_trace() {
  FlightRecorderCUDA::get()->reset_all();
}

std::string dump_nccl_trace(
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive) {
  auto ncclDumpMap = getNCCLCommDumpMap();
#if defined(USE_ROCM) && defined(NCCL_COMM_DUMP)
  for (const auto& [ncclUniqueIDStr, dump] : ncclDumpMap) {
    printNcclCommProxyTrace("Received dump signal " + ncclUniqueIDStr, dump);
  }
#endif // defined(USE_ROCM) && defined(NCCL_COMM_DUMP)
  return FlightRecorderCUDA::get()->dump(
      ncclDumpMap, includeCollectives, includeStackTraces, onlyActive);
}

std::string dump_nccl_trace_json(bool includeCollectives, bool onlyActive) {
  auto ncclDumpMap = getNCCLCommDumpMap();
  return FlightRecorderCUDA::get()->dump_json(
      ncclDumpMap, includeCollectives, onlyActive);
}

std::optional<std::function<void(std::function<void(const std::string&)>)>>&
get_cpp_trace_dumper() {
  static std::optional<
      std::function<void(std::function<void(const std::string&)>)>>
      dumper(std::nullopt);
  return dumper;
}

gil_checker_t& get_gil_checker() {
  static gil_checker_t gil_checker = nullptr;
  return gil_checker;
}

static std::future<bool> launchAsyncGilCheck() {
  std::promise<bool> resultPromise;
  std::future<bool> resultFuture = resultPromise.get_future();
  TORCH_CHECK(get_gil_checker(), "Can't check GIL with null GIL checker");
  std::thread workerThread([promise = std::move(resultPromise)]() mutable {
    c10::setThreadName("pt_nccl_gil_chk");

    try {
      auto& gil_checker = get_gil_checker();
      promise.set_value((*gil_checker)());
    } catch (...) {
      promise.set_exception(std::current_exception());
    }
  });

  // Detach the thread to allow it to run independently
  workerThread.detach();

  return resultFuture;
}

const int64_t ProcessGroupNCCL::kWatchdogThreadSleepMillis = 100;
constexpr int64_t kSynchronizeBusyWaitMillis = 1;
thread_local uint64_t ProcessGroupNCCL::ncclActiveGroupCounter_ = 0;

std::ostream& operator<<(
    std::ostream& output,
    const ProcessGroupNCCL::WorkNCCL& workNCCL) {
  std::string workInfo;
  workInfo = c10::str(
      "WorkNCCL(",
      "SeqNum=",
      workNCCL.seq_,
      ", OpType=",
      opTypeToString(workNCCL.opType_),
      ", NumelIn=",
      workNCCL.numelIn_,
      ", NumelOut=",
      workNCCL.numelOut_,
      ", Timeout(ms)=",
      workNCCL.opTimeout_.count(),
      ")");
  return output << workInfo;
}

/* Implementation of TensorShelf class */

void TensorShelf::stash(std::vector<at::Tensor>& tensors) {
  std::lock_guard<std::mutex> lock(mutex_);
  tVector_.insert(tVector_.end(), tensors.begin(), tensors.end());
}

void TensorShelf::stash(TensorShelf& other) {
  std::vector<at::Tensor>& otherVec = other.get();
  this->stash(otherVec);
}

void TensorShelf::unstash() {
  this->clear();
}

bool TensorShelf::empty() {
  std::lock_guard<std::mutex> lock(mutex_);
  return tVector_.empty();
}

void TensorShelf::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  tVector_.clear();
}

std::vector<at::Tensor>& TensorShelf::get() {
  return tVector_;
}

ProcessGroupNCCL::WorkNCCL::WorkNCCL(
    std::string pgUID,
    std::string pgDesc,
    at::Device& device,
    int rank,
    OpType opType,
    uint64_t seq,
    bool isP2P,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputs,
    bool enableTiming,
    bool cudaEventCacheEnabled,
    DebugLevel distDebugLevel)
    : Work(rank, opType, profilingTitle, inputs),
      pgUID_(std::move(pgUID)),
      pgDesc_(std::move(pgDesc)),
      device_(device),
      workStartTime_(std::chrono::steady_clock::now()),
      seq_(seq),
      isP2P_(isP2P),
      timingEnabled_(enableTiming),
      distDebugLevel_(distDebugLevel) {
  // Creates the CUDA event wrappers
  // Note: The actual events are lazily created when first recorded to with
  // DEFAULT_FLAGS = cudaEventDisableTiming.
  if (cudaEventCacheEnabled) {
    ncclStartEvent_ = enableTiming
        ? CUDAEventCache::get(device.index())->create(enableTiming)
        : nullptr;
    ncclEndEvent_ = CUDAEventCache::get(device.index())->create(enableTiming);
  } else {
    ncclStartEvent_ = enableTiming
        ? std::make_shared<at::cuda::CUDAEvent>(cudaEventDefault)
        : nullptr;
    ncclEndEvent_ = std::make_shared<at::cuda::CUDAEvent>(
        enableTiming ? cudaEventDefault : cudaEventDisableTiming);
  }
  futureWorkResult_ =
      c10::make_intrusive<at::ivalue::Future>(c10::AnyEnumType::get());
  // other functions expect an initialized ptr
  stashed_for_allocator_safety_ = std::make_shared<TensorShelf>();
}

ProcessGroupNCCL::WorkNCCL::WorkNCCL(const WorkNCCL& w)
    : Work(w.rank_, w.opType_),
      std::enable_shared_from_this<WorkNCCL>(w),
      pgUID_(w.pgUID_),
      pgDesc_(w.pgDesc_),
      device_(w.device_),
      ncclStartEvent_(w.ncclStartEvent_),
      ncclEndEvent_(w.ncclEndEvent_),
      ncclComm_(w.ncclComm_),
      blockingWait_(w.blockingWait_),
      opTimeout_(w.opTimeout_),
      ownedEphermeralTimeout_(w.ownedEphermeralTimeout_),
      workStartTime_(w.workStartTime_),
      seq_(w.seq_),
      isP2P_(w.isP2P_),
      startTraceUpdated_(w.startTraceUpdated_),
      numelIn_(w.numelIn_),
      numelOut_(w.numelOut_),
      store_(w.store_),
      // Note: the `work` returned to user and the `work` enqueued to watchdog
      // share the pointer to the tensor stash.  At least one of them should
      // clean the tensor stash, the earlier the better, i.e. user calling
      // `work.wait` than watchdog detecting work completion.
      stashed_for_allocator_safety_(w.stashed_for_allocator_safety_),
      futureWorkResult_(w.futureWorkResult_),
      timingEnabled_(w.timingEnabled_),
      trace_id_(w.trace_id_),
      trace_reset_epoch_(w.trace_reset_epoch_),
      distDebugLevel_(w.distDebugLevel_) {
  exception_ = w.exception_;
}

bool ProcessGroupNCCL::WorkNCCL::isCompleted() {
  if (!ncclComm_->isAborted()) {
    checkAndSetException();
  }
  return exception() || finishedGPUExecutionInternal();
}

bool ProcessGroupNCCL::WorkNCCL::isStarted() {
  if (!ncclComm_->isAborted()) {
    checkAndSetException();
  }
  return exception() || startedGPUExecutionInternal();
}

bool ProcessGroupNCCL::WorkNCCL::isSuccess() const {
  C10_THROW_ERROR(NotImplementedError, "WorkNCCL::isSuccess() is deprecated");
}

void ProcessGroupNCCL::WorkNCCL::checkAndSetException() {
  if (exception()) {
    // We already have an exception.
    return;
  }

  auto exception_ptr = checkForNCCLErrors();
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = exception_ptr;
  if (exception_) {
    LOG(ERROR) << logPrefix() << "Collective " << *this
               << " raised the following async exception: "
               << getExceptionMsgFromExceptionPtr(exception_);

    // Mark future result as ERROR
    if (futureWorkResult_ && !futureWorkResult_->completed()) {
      futureWorkResult_->markCompleted(
          at::IValue(static_cast<uint8_t>(WorkResult::COMM_ERROR)));
    }
  }
}

const std::string& ProcessGroupNCCL::WorkNCCL::logPrefix() const {
  static std::string prefix = c10::str("[Rank ", rank_, "] ");
  return prefix;
}

void ProcessGroupNCCL::WorkNCCL::setException(
    std::exception_ptr exception_ptr) {
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = std::move(exception_ptr);
}

// Helper that checks if the NCCL kernels are completed on the GPUs
bool ProcessGroupNCCL::WorkNCCL::finishedGPUExecution() {
  checkAndSetException();
  return finishedGPUExecutionInternal();
}

bool ProcessGroupNCCL::WorkNCCL::startedGPUExecutionInternal() const {
  // if timing is disabled we won't have allocated start events
  if (!timingEnabled_) {
    return false;
  }
  // Checking the work's corresponding CUDA event's status
  if (!ncclStartEvent_->query()) {
    return false;
  }
  return true;
}

bool ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const {
  // Checking the work's corresponding CUDA event's status
  // It calls `cudaEventQuery` eventually. Although this seems to be a
  // non-blocking call, but we did notice hangs in the past. It can
  // hang if another thread is holding the CUDA global context lock. For
  // example, when doing a `cudaDeviceSynchronize` or even
  // `cudaStreamSynchronize`.
  if (!ncclEndEvent_->query()) {
    return false;
  }
  return true;
}

bool ProcessGroupNCCL::WorkNCCL::checkTimeout(
    std::optional<std::chrono::milliseconds> timeout) {
  STATIC_SCOPED_WAIT_COUNTER(
      pytorch.wait_counter.ProcessGroupNCCL__checkTimeout);
  auto currentTimepoint = std::chrono::steady_clock::now();
  auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      currentTimepoint - workStartTime_);
  auto workTimeout = timeout ? *timeout : opTimeout_;

  if (timeElapsed < workTimeout) {
    return false;
  }

  // Timed out

  std::string exceptionMsg = c10::str(
      logPrefix(),
      "Watchdog caught collective operation timeout: ",
      *this,
      " ran for ",
      timeElapsed.count(),
      " milliseconds before timing out.");

  LOG(ERROR) << exceptionMsg;

  std::exception_ptr exception_ptr =
      std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exceptionMsg));
  if (!exception()) {
    // if there is already an error, we don't override it
    setException(exception_ptr);
  }

  // Mark future result as TIMEOUT
  if (futureWorkResult_ && !futureWorkResult_->completed()) {
    futureWorkResult_->markCompleted(
        at::IValue(static_cast<uint8_t>(WorkResult::TIMEOUT)));
  }
  return true;
}

// Print the traceback of the collective at call time
std::string ProcessGroupNCCL::WorkNCCL::getTraceback() const {
  // First step we get the corresponding record entry from FR, based on work's
  // trace_id_ and trace_reset_epoch_
  std::optional<FlightRecorderCUDA::Entry> entry =
      FlightRecorderCUDA::get()->getEntry(trace_id_, trace_reset_epoch_);
  if (entry.has_value()) {
    auto entryVal = entry.value();
    // Get stack trace from FR entry, in string format
    // Note: `getTraceback` call below invokes `torch::symbolize`, which may
    // need to acquire the GIL. In order for watchdog to be block-free, we make
    // the call with std::async.
    auto future = std::async(
        std::launch::async, [&entryVal]() { return entryVal.getTraceback(); });
    // Wait for the future to complete or timeout
    auto status = future.wait_for(std::chrono::seconds(8));
    if (status == std::future_status::ready) {
      return future.get();
    }
  }
  return "";
}

// Print the traceback of the collective at call time
void ProcessGroupNCCL::WorkNCCL::printTraceback() const {
  std::string tracebackStr = getTraceback();
  if (!tracebackStr.empty()) {
    LOG(ERROR) << "Stack trace of the failed collective: \n" << tracebackStr;
  } // else, symbolizer probably timed out, we skip logging the stack trace.
  else {
    LOG(ERROR)
        << "Stack trace of the failed collective not found, "
        << "potentially because FlightRecorder is disabled. "
        << "You can enable it by setting TORCH_NCCL_TRACE_BUFFER_SIZE to a non-zero value.";
  }
}

void ProcessGroupNCCL::WorkNCCL::handleException(
    ErrorHandlingMode errorHandling) {
  if (exception_) {
    auto exceptionMsg = c10::str(
        "Some NCCL operations have failed or timed out. Due to the ",
        "asynchronous nature of CUDA kernels, subsequent GPU operations ",
        "might run on corrupted/incomplete data.");
    LOG(ERROR) << logPrefix() << exceptionMsg;
    C10_LOG_API_USAGE_ONCE("ProcessGroupNCCL.WorkNCCL.handleException");

    auto logger = c10d::C10dLogger::getLogger();
    if (logger) {
      ::c10d::C10dLoggingData data;
      data.strings["work_nccl_exception"] =
          getExceptionMsgFromExceptionPtr(exception_);
      logger->log(data);
    }

    if (SHOULD_TEAR_DOWN(errorHandling)) {
      auto tearDownMsg = c10::str(
          "To avoid data inconsistency, we are taking the entire process down.");
      LOG(ERROR) << logPrefix() << tearDownMsg;
      std::rethrow_exception(exception_);
    }
  }
}

void ProcessGroupNCCL::WorkNCCL::synchronize() {
  synchronizeStream();
  if (c10d::allow_inflight_collective_as_graph_input()) {
    c10d::unregister_work(
        c10::intrusive_ptr<
            ProcessGroupNCCL::WorkNCCL>::unsafe_reclaim_from_nonowning(this));
  }
}

void ProcessGroupNCCL::WorkNCCL::synchronizeStream() {
  auto currentStream = at::cuda::getCurrentCUDAStream(device_.index());
  // Block the current stream on the NCCL stream
  ncclEndEvent_->block(currentStream);
  // Unstage the stashed tensors so that CachingAllocator can recycle them
  // THIS MUST HAPPEN AFTER THE BLOCKING CALL ABOVE
  stashed_for_allocator_safety_->unstash();
}

// Same as calling synchronize() when blockingWait_ is false
bool ProcessGroupNCCL::WorkNCCL::wait(std::chrono::milliseconds timeout) {
  RECORD_PARAM_COMMS(
      std::make_tuple(static_cast<int64_t>(this->seq_), this->isP2P_), // seq
      std::make_tuple(pgUID_, pgDesc_), // PG name tuple
      rank_, // rank
      "wait", // collective name
      0, // inNelems
      0, // outNelems
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      -1,
      -1,
      static_cast<int>(1)); // number of device?

  // synchronize() will block the current stream on the NCCL stream
  synchronize();

  // In case of blockingWait or a timeout value is specified by the user, we
  // block the CPU thread until the work is completed or timed out.
  if (blockingWait_ || timeout != kNoTimeout) {
    while (!isCompleted()) {
      bool timedOut = checkTimeout(
          timeout == kNoTimeout ? std::nullopt : std::make_optional(timeout));
      // Explicitly abort ncclComms here before throwing this timed out
      // exception to users.
      // If throwing timed out excepiton without aborting nccl communicators
      // here, it was observed that CUDA GPU will have 100% utilization and
      // can not run new events successfully.
      if (timedOut) {
        std::string exceptionMsg = c10::str(
            logPrefix(), "Work ", (*this), " timed out in blocking wait.");
        LOG(ERROR) << exceptionMsg;
        break;
      }
      // Yield
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
    }
  } else if (isBarrierOp_ && !isCompleted()) {
    // For barrier wait when timeout is unspecified, we block the CPU thread on
    // current stream. This is to minimize the CPU barrier wait time in healthy
    // path
    auto currentStream = at::cuda::getCurrentCUDAStream(device_.index());
    // CUDAStream wrapper will correctly use a DeviceGuard here
    currentStream.synchronize();
  }

  // If exception is detected, throw it from the main CPU thread
  if (exception()) {
    // Abort NCCL communicators
    abort();
    // Throw exception (from main thread here)
    handleException(TearDown);
  }

  // TODO(kwen2501): this should be moved to c10d tests, to qualify a NCCL
  // upgrade. Once a NCCL version is qualified, this code should not be needed
  // at runtime.
#ifdef PGNCCL_ENABLE_HASH
  if (enableCollectiveHashDebug_.load()) {
    auto numel = getTensorsNumel(*outputs_);
    auto hashValue = hashTensors(*outputs_);
    PRINT_COLLECTIVE_HASH_SIGNATURE(
        "output", opTypeToString(opType_), numel, hashValue);
  }
#endif // PGNCCL_ENABLE_HASH
  // Always return true, because abort API is not implemented.
  return true;
}

void ProcessGroupNCCL::WorkNCCL::abort() {
  // dump before aborting for rcclexp
#if defined(USE_ROCM) && defined(NCCL_COMM_DUMP)
  auto dumpMap = ncclComm_->ncclCommDump();
  printNcclCommProxyTrace("WorkNCCL::abort", dumpMap);
#endif // USE_ROCM && NCCL_COMM_DUMP

  // Abort all communicators of this work
  ncclComm_->abort();

  {
    std::lock_guard<std::mutex> lock(ncclCommMemPoolMapMutex);
    ncclCommMemPoolMap.erase(ncclComm_);
  }
}

static std::atomic<size_t> process_group_id = 0;

constexpr const char* MULTI_DEVICE_ERROR_MSG =
    "Expecting one tensor only but got multiple. You are probably using multiple "
    "devices under one thread. The support for such usage has been deprecated. "
    "For details, please refer to "
    "https://pytorch.org/docs/stable/distributed.html#multi-gpu-collective-functions. "
    "ProcessGroupNCCL continues supporting multi-process and multi-thread modes.";

ProcessGroupNCCL::ProcessGroupNCCL(
    c10::intrusive_ptr<Store> store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : Backend(rank, size),
      store_(std::move(store)),
      options_(std::move(options)),
      terminateProcessGroup_(false),
      local_id_(process_group_id++),
      intraNodeComm_(initIntraNodeComm()) {
  TORCH_CHECK_WITH(
      ValueError,
      at::cuda::getNumGPUs() != 0,
      "ProcessGroupNCCL is only supported with GPUs, no GPUs found!");

  // getNcclVersion needs to get called before launching threads which can
  // potentially call getenv. getNcclVersion internally calls setenv to set some
  // environment variables from config file, which can race with getenv from
  // other threads and cause segfaults.
  const auto ncclVersion = getNcclVersion();
  this->setGroupUid(options_->group_name);
  this->localDeviceCount_ = static_cast<int>(at::cuda::getNumGPUs());
  logPrefix_ = createLogPrefix();
  blockingWait_ = getCvarBool(TORCH_NCCL_BLOCKING_WAIT, false);
  asyncErrorHandling_ = static_cast<ErrorHandlingMode>(
      getCvarInt(TORCH_NCCL_ASYNC_ERROR_HANDLING, 3 /*SkipCleanUp*/));
  enableNanCheck_ = getCvarBool(TORCH_NCCL_NAN_CHECK, false);
  cudaEventCacheEnabled_.store(getCvarBool(TORCH_NCCL_CUDA_EVENT_CACHE, true));
  traceBufferSize_ = getCvarInt(TORCH_NCCL_TRACE_BUFFER_SIZE, 2000);
  enableCollectiveHashDebug_ = (dist_debug_level_ >= DebugLevel::Detail);
  // store_ usually is wrapped with PrefixStore and the prefix is different
  // across different ProcessGroupNCCL(PG) instances. We need to get the
  // underlying non-PrefixStore for sharing global information shared across
  // different PGs.
  PrefixStore* prefixStore = dynamic_cast<PrefixStore*>(store_.get());
  globalStore_ =
      prefixStore ? prefixStore->getUnderlyingNonPrefixStore() : store_;
  auto desyncDebug = getCvarBool(TORCH_NCCL_DESYNC_DEBUG, false) ||
      (dist_debug_level_ >= DebugLevel::Detail);
#ifdef ENABLE_NCCL_ERROR_CHECKING
  enableTiming_.store(
      getCvarBool(TORCH_NCCL_ENABLE_TIMING, false) || desyncDebug);
#endif // ENABLE_NCCL_ERROR_CHECKING
  if (getCvarBool(TORCH_NCCL_AVOID_RECORD_STREAMS, false)) {
    TORCH_WARN_ONCE(
        "TORCH_NCCL_AVOID_RECORD_STREAMS is the default now, this environment variable is thus deprecated.");
  }
  showSerializationWarning_ =
      getCvarBool(TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING, true);

  if (blockingWait_) {
    LOG(INFO)
        << logPrefix()
        << "TORCH_NCCL_BLOCKING_WAIT is enabled, NO watchdog thread is created.";
  } else {
    if (desyncDebug && asyncErrorHandling_ == NoHandling) {
      LOG(INFO)
          << logPrefix()
          << "TORCH_NCCL_DESYNC_DEBUG and TORCH_NCCL_ASYNC_ERROR_HANDLING "
          << "must both be enabled. "
          << "Enabling TORCH_NCCL_ASYNC_ERROR_HANDLING.";
      asyncErrorHandling_ = SkipCleanUp;
    }
  }

  // If deterministic mode is enabled, we need to disable the NVLS algorithm in
  // NCCL.
  // TODO: remove this once NVLS supports deterministic mode.
  if (at::globalContext().deterministicAlgorithms()) {
    // Check if user have already set NCCL_ALGO. If already set, leave it.
    auto nccl_algo = c10::utils::get_env("NCCL_ALGO");
    if (!nccl_algo.has_value()) {
      LOG(INFO)
          << "torch deterministic mode is enabled, "
          << "disabling NVLS algorithm in NCCL which can lead to non-deterministic reduction.";
      // Sorry we have to disable NVLS for all collectives, be it all-reduce
      // or all-gather, because NCCL does not support per-collective
      // algorithm selection today.
      c10::utils::set_env("NCCL_ALGO", "^NVLS");
    }
  }

  // Initialize the heartbeat monitor/watchdog instance. This has to be done
  // before the corresponding thread is launched to avoid the error.
  heartbeatMonitor_ = std::make_unique<HeartbeatMonitor>(this);
  watchdog_ = std::make_unique<Watchdog>(this);

#ifdef ENABLE_NCCL_ERROR_CHECKING
  // in blockingWait mode, we don't need to enable the watchdog thread to check
  // the timeout or nccl error because the main thread would throw an exception
  // and it is the user's responsibility to handle the exception.
  if (!blockingWait_) {
    watchdog_->start();
  }
#endif // ENABLE_NCCL_ERROR_CHECKING

  init();
  const std::string OFF = "OFF";
  std::string torch_distributed_debug =
      getCvarString({"TORCH_DISTRIBUTED_DEBUG"}, OFF.c_str());
  LOG(INFO) << logPrefix()
            << "ProcessGroupNCCL initialization options: " << "size: " << size
            << ", global rank: " << globalRank()
            << ", TIMEOUT(ms): " << options_->timeout.count()
            << ", USE_HIGH_PRIORITY_STREAM: "
            << options_->is_high_priority_stream
            << ", SPLIT_FROM: " << options_->split_from
            << ", SPLIT_COLOR: " << options_->split_color
            << ", PG Name: " << options_->group_name;

  LOG(INFO) << logPrefix() << "ProcessGroupNCCL environments: "
            << "NCCL version: " << ncclVersion
            << ", TORCH_NCCL_ASYNC_ERROR_HANDLING: " << asyncErrorHandling_
            << ", TORCH_NCCL_ENABLE_TIMING: " << enableTiming_.load()
            << ", TORCH_NCCL_BLOCKING_WAIT: " << blockingWait_
            << ", TORCH_DISTRIBUTED_DEBUG: " << torch_distributed_debug
#ifdef NCCL_HAS_COMM_REGISTER
            << ", TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: "
            << shouldAllCommunicatorsRegisterAllTensors()
#endif // NCCL_HAS_COMM_REGISTER
            << ", TORCH_NCCL_TRACE_BUFFER_SIZE: " << traceBufferSize_
            << ", TORCH_NCCL_NAN_CHECK: " << enableNanCheck_
            << ", TORCH_NCCL_CUDA_EVENT_CACHE: " << cudaEventCacheEnabled_;

  getGlobalRankStartAndStride(
      options_->global_ranks_in_group,
      this->globalRankStart_,
      this->globalRankStride_);

  // Attach hooks to cache allocator to trigger the hooks whenever a traced
  // action is called. In the following hooks, we register a newly allocated
  // segment when SEGMENT_ALLOC action occurs, and deregister a segment when
  // SEGMENT_FREE action occurs.
  if (shouldAllCommunicatorsRegisterAllTensors()) {
    // This call is idempotent.
    attachAllocatorHooks();
  }
}

void ProcessGroupNCCL::eagerConnectSingleDevice(at::Device device) {
  const auto key = getKeyFromDevice(device);
  LOG(INFO) << logPrefix() << "Eagerly connecting nccl backend with device "
            << device;
  initNCCLComm(key, device, OpType::ALLREDUCE);
  eagerInit_ = true;
}

bool ProcessGroupNCCL::useNonblocking() {
#ifndef NCCL_HAS_COMM_NONBLOCKING
  return false;
#endif // NCCL_HAS_COMM_NONBLOCKING
  // Already parsed, return the cached value
  if (useNonblocking_.has_value()) {
    return useNonblocking_.value();
  }
  // Get environment variable.
  auto nbEnv = c10::utils::check_env("TORCH_NCCL_USE_COMM_NONBLOCKING");

  // 1st priority: Respect the user's setting
  if (options_->config.blocking != NCCL_CONFIG_UNDEF_INT) {
    useNonblocking_ = options_->config.blocking == 0;
  }
  // 2nd priority: Respect the environment variable
  else if (nbEnv.has_value()) {
    useNonblocking_ = nbEnv;
  }
  // 3rd priority: automatically use nonblocking if we are in eager init mode
  // Note: this automatic selection is disabled in torch 2.7.1 to work around a
  // hang in NCCL 2.26 in non-blocking mode. We can revisit if NCCL fixes the
  // bug. See https://github.com/pytorch/pytorch/issues/153960
  // else if (getBoundDeviceId()) {
  //   useNonblocking_ = true;
  // }
  // 4th priority: otherwise, nonblocking = false to preserve old behavior
  else {
    useNonblocking_ = false;
  }

  LOG(INFO) << logPrefix()
            << "Using non-blocking mode: " << useNonblocking_.value();
  return useNonblocking_.value();
}

void ProcessGroupNCCL::performNocolorSplit(at::Device device) {
  // If our backend doesn't support splitting, this is a no-op for
  // ranks not in the new subgroup (and ranks that would be in it will
  // just use a new communicator rather than split).
#ifdef NCCL_HAS_COMM_SPLIT
  const auto key = getKeyFromDevice(device);
  LOG(INFO) << logPrefix() << "Performing nocolor split on backend device "
            << device << ", key " << key << ", i am " << this;
  bool useNb = useNonblocking();
  options_->config.blocking = useNb ? 0 : 1;
  auto comm = getNCCLComm(key);
  if (comm == nullptr) {
    LOG(ERROR) << logPrefix()
               << "No parent communicator exists for nocolor split";
  }
  NCCLComm::split(comm.get(), NCCL_SPLIT_NOCOLOR, rank_, options_->config);
#endif // NCCL_HAS_COMM_SPLIT
}

bool ProcessGroupNCCL::isInitialized() {
  if (devNCCLCommMap_.empty()) {
    return false;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  bool initialized = true;
  for (const auto& [_, comm] : devNCCLCommMap_) {
    if (!comm->isInitialized()) {
      initialized = false;
      break;
    }
  }
  return initialized;
}

ErrorType ProcessGroupNCCL::getError() {
  std::lock_guard<std::mutex> lock(errorMutex_);
  return error_;
}

void ProcessGroupNCCL::registerMemPool(c10::cuda::MemPool* pool, bool symm) {
  const auto key = std::to_string(pool->device());
  LOG(INFO) << logPrefix()
            << "Performing NCCL user buffer registration for all buffers in "
            << "MemPool: " << pool->id() << ", device index: " << key
            << ", i am " << this;
  auto ncclComm = getNCCLComm(key);
  if (ncclComm == nullptr) {
    C10_THROW_ERROR(
        DistBackendError,
        "NCCL communicator has not been initialized before mem pool creation. You can pass `device_id` to init_process_group -- one way of eager initialization -- to work around this issue");
  }
  {
    std::lock_guard<std::mutex> lock(ncclCommMemPoolMapMutex);
    auto iter = ncclCommMemPoolMap.find(ncclComm);
    iter->second.insert(std::make_tuple(pool->id(), symm));
  }
  // We must ensure we're listening for allocator trace events in order to
  // register future segments allocated in this pool (this call is idempotent).
  attachAllocatorHooks();
  auto snapshot = c10::cuda::CUDACachingAllocator::snapshot(pool->id());
  for (const auto& segmentInfo : snapshot.segments) {
    TORCH_INTERNAL_ASSERT(
        segmentInfo.device == pool->device(),
        "Mismatch between CUDA memory segment device and pool's device");
    ncclComm->registerSegment(
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        reinterpret_cast<void*>(segmentInfo.address),
        segmentInfo.total_size,
        /*errorOnRereg=*/false, // ignores reregistration error
        /*window*/ symm); // whether to use NCCL symmetric memory
  }
}

void ProcessGroupNCCL::deregisterMemPool(c10::cuda::MemPool* pool) {
  const auto key = std::to_string(pool->device());
  LOG(INFO) << logPrefix()
            << "Performing NCCL user buffer deregistration for all buffers in "
            << "MemPool: " << pool->id() << ", device index: " << key
            << ", i am " << this;
  auto ncclComm = getNCCLComm(key);
  if (ncclComm == nullptr) {
    C10_THROW_ERROR(
        DistBackendError,
        "NCCL communicator has not been initialized before mem pool creation. You can pass `device_id` to init_process_group -- one way of eager initialization -- to work around this issue");
  }
  bool symm;
  {
    std::lock_guard<std::mutex> lock(ncclCommMemPoolMapMutex);
    auto iter = ncclCommMemPoolMap.find(ncclComm);
    auto mempool_it = std::find_if(
        iter->second.begin(), iter->second.end(), [&](const auto& tup) {
          return std::get<0>(tup) == pool->id();
        });
    TORCH_CHECK(
        mempool_it != iter->second.end(),
        "Trying to unregister not previously registered pool");
    symm = std::get<1>(*mempool_it);
    iter->second.erase(mempool_it);
  }
  auto snapshot = c10::cuda::CUDACachingAllocator::snapshot(pool->id());
  for (const auto& segmentInfo : snapshot.segments) {
    TORCH_INTERNAL_ASSERT(
        segmentInfo.device == pool->device(),
        "Mismatch between CUDA memory segment device and pool's device");
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    ncclComm->deregisterSegment(
        reinterpret_cast<void*>(segmentInfo.address), symm);
  }
}

c10::intrusive_ptr<intra_node_comm::IntraNodeComm> ProcessGroupNCCL::
    initIntraNodeComm() {
  using IntraNodeComm = intra_node_comm::IntraNodeComm;
  if (!IntraNodeComm::isEnabled()) {
    return nullptr;
  }
  auto prefixStore = c10::make_intrusive<PrefixStore>("IntraNodeComm", store_);
  auto comm = c10::make_intrusive<IntraNodeComm>(prefixStore, rank_, size_);
  if (comm->rendezvous()) {
    return comm;
  } else {
    return nullptr;
  }
}

void ProcessGroupNCCL::setSequenceNumberForGroup() {
} // NCCL just starts sequence numbers at 0.

uint64_t ProcessGroupNCCL::getSequenceNumberForGroup() {
  return seqCollective_;
}

void ProcessGroupNCCL::registerOnCompletionHook(
    std::function<void(std::shared_ptr<WorkInfo>)>&& hook) {
  TORCH_WARN_ONCE(
      "ProcessGroupNCCL OnCompletion hook will be deprecated in favor of Flight Recorder. "
      "Please check out FlightRecorder.hpp for information that is recorded at work completion. "
      "You can file an issue if you want additional information to be recorded. "
      "You can also file an RFC if you want Flight Recorder to accept plugins that customize the recording.")

  TORCH_CHECK_WITH(
      DistBackendError,
      onCompletionHook_ == nullptr,
      "ProcessGroupNCCL OnCompletion hook already registered");

  TORCH_CHECK_WITH(
      ValueError,
      enableTiming_.load(),
      "ProcessGroupNCCL OnCompletion hook requires recording start and end "
      "events which require setting TORCH_NCCL_ENABLE_TIMING environment variable. "
      "This is only available for NCCL version >= 2.4.");
  onCompletionHook_ = std::move(hook);
  onCompletionHookThread_ = std::thread(&ProcessGroupNCCL::runHookLoop, this);
}

// must release GIL when calling this method
void ProcessGroupNCCL::waitForPendingWorks() {
  // Reasoning about hook completion:
  // 1. waitForPendingWorks should be called after user code has finished
  // calling
  //    all collectives. This means, when we got here, all of the collectives
  //    are either in workMetaList_ or has been erased from workMetaList_.
  // 2. The watchdog thread grabs both locks to move Work object from the
  //    workMetaList_ to the completedWorkList_, and the hook thread only erases
  //    a Work object after the hook is returned. Therefore, after user code
  //    calls a collective, its Work object is either in workMetaList_ or in
  //    completedWorkList_ before it finishes.
  // 3. We have three threads and two locks.
  //      a. main thread (this function) grabs two locks atomically
  //      b. watchdog thread (runLoop function) always grabs
  //      workMetaListMutex_
  //         first and then grabs completedWorkListMutex_.
  //      c. hook thread (runHookLoop function) only grabs
  //      completedWorkListMutex_. Therefore, locks are always acquired in the
  //      same order and hence no deadlocks.
  while (true) {
    {
      std::lock(workMetaListMutex_, completedWorkListMutex_);
      std::lock_guard<std::mutex> lockWork(workMetaListMutex_, std::adopt_lock);
      std::lock_guard<std::mutex> lockHook(
          completedWorkListMutex_, std::adopt_lock);

      if (workMetaList_.empty() && completedWorkList_.empty()) {
        return;
      }
    }

    std::this_thread::sleep_for(
        std::chrono::milliseconds(kWatchdogThreadSleepMillis));
  }
}

void ProcessGroupNCCL::enableCollectivesTiming() {
  enableTiming_.store(true);
}

c10::intrusive_ptr<Backend> ProcessGroupNCCL::split(
    const c10::intrusive_ptr<Store>& store,
    const std::vector<int>& ranks,
    const c10::intrusive_ptr<Backend::Options>& opts) {
  auto deviceIdx = guessDeviceId();
  TORCH_CHECK(
      deviceIdx >= 0,
      "ProcessGroupNCCL::split: rank ",
      rank_,
      " has no device is bound to this rank.");
  auto device = at::Device(at::DeviceType::CUDA, deviceIdx);
  auto it = std::find(ranks.begin(), ranks.end(), rank_);
  int groupRank;
  if (it == ranks.end()) {
    // This rank is not in the new group, so no_color split should be called
    performNocolorSplit(device);
    return nullptr;
  } else {
    groupRank = std::distance(ranks.begin(), it);
  }

  auto ncclOpts = c10::dynamic_intrusive_pointer_cast<Options>(opts);
  TORCH_CHECK(ncclOpts != nullptr, "opts not a ProcessGroupNCCL::Options.");

  // TODO: we need to get rid of globalRanksInGroup eventually.
  std::vector<uint64_t> globalRanksInGroup;
  for (auto rank : ranks) {
    globalRanksInGroup.emplace_back(groupRanks()[rank]);
  }
  ncclOpts->split_from =
      c10::intrusive_ptr<ProcessGroupNCCL>::unsafe_reclaim_from_nonowning(this);
  ncclOpts->global_ranks_in_group = std::move(globalRanksInGroup);
  auto color = genNcclSplitColor(ranks);
  ncclOpts->split_color = color;
  auto pg = c10::make_intrusive<ProcessGroupNCCL>(
      store->clone(), groupRank, ranks.size(), ncclOpts);
  pg->eagerConnectSingleDevice(device);
  return c10::static_intrusive_pointer_cast<Backend>(pg);
}

c10::intrusive_ptr<Backend> ProcessGroupNCCL::merge(
    const c10::intrusive_ptr<Store>& store,
    const c10::intrusive_ptr<Backend::Options>& opts,
    const int& rank,
    const int& size) {
  auto ncclOpts = c10::dynamic_intrusive_pointer_cast<Options>(opts);
  TORCH_CHECK(ncclOpts != nullptr, "opts not a ProcessGroupNCCL::Options.");
  auto pg = c10::make_intrusive<ProcessGroupNCCL>(
      store->clone(), rank, size, ncclOpts);
  return c10::static_intrusive_pointer_cast<Backend>(pg);
}

bool ProcessGroupNCCL::waitForFutureOrTimeout(
    std::future<bool>& fut,
    const std::chrono::milliseconds& timeOutMilSec,
    const std::string& futDescription,
    ::c10d::C10dLoggingData& debugLog,
    bool throwException) {
  std::string errorMsg;
  bool complete = false;

  TORCH_CHECK(fut.valid(), "Expected a valid future");
  std::future_status status = fut.wait_for(timeOutMilSec);
  if (status == std::future_status::ready) {
    // Calling .get() will re-raise any exception from the future, and we don't
    // care about the retval
    try {
      bool result = fut.get();
      if (result) {
        VLOG(2) << logPrefix()
                << "future successfully executed for: " << futDescription;
        debugLog.strings["status"] = "SUCCESS";
        complete = true;
      }
    } catch (const std::exception& e) {
      errorMsg = c10::str(
          logPrefix(),
          "Exception thrown when waiting for future ",
          futDescription,
          ": ",
          e.what());

      debugLog.strings["status"] = "EXCEPTION";
      debugLog.strings["exception_msg"] = e.what();
      LOG(ERROR) << errorMsg;
    } catch (...) {
      errorMsg = c10::str(
          logPrefix(),
          "Unknown exception thrown when waiting for future ",
          futDescription);
      debugLog.strings["status"] = "EXCEPTION";
      debugLog.strings["exception_msg"] = "Unknown exception";
      LOG(ERROR) << errorMsg;
    }
  } else {
    errorMsg = c10::str(
        logPrefix(),
        "Future for ",
        futDescription,
        " timed out after ",
        timeOutMilSec.count(),
        " ms");
    debugLog.strings["status"] = "TIMEOUT";
    LOG(ERROR) << errorMsg;
  }
  if (throwException && !errorMsg.empty()) {
    C10_THROW_ERROR(DistBackendError, errorMsg);
  }
  return complete;
}

void ProcessGroupNCCL::abortCommsFromMap(
    std::unordered_map<std::string, std::shared_ptr<NCCLComm>>& ncclCommsMap,
    const std::optional<std::string>& abortReason) {
  // The process may control multiple devices, loop through the communicators on
  // each device
  // NCCL expects Group abort when there are multiple communicators created in a
  // device. Group abort requires 2.22.0 release and up.
  if (getNcclVersionNumber() >= NCCL_VERSION(2, 22, 0)) {
    groupStart();
  }
  for (auto& it : ncclCommsMap) {
    auto& devName = it.first;
    auto& ncclComm = it.second;
    VLOG(2) << logPrefix() << "ProcessGroupNCCL destroying ncclComm_ "
            << ncclComm->repr() << " on CUDA device: " << devName;
    // abort() call now has GPU guard inside
    ncclComm->abort(abortReason);
    // Note that we don't remove the aborted communicators from the
    // cache. The reason is that if we do remove the communicator
    // from the cache, it is possible that a new collective operation
    // calls `ncclCommInitRank` to create a new communicator whereas
    // other ranks might have failed/timed out and didn't enter
    // `ncclCommInitRank`. As a result, when there is a failure on
    // a communicator the application receives an exception and its
    // their responsibility to destroy the process group and recreate
    // it to recover from errors.

    VLOG(2) << logPrefix() << "ProcessGroupNCCL destroyed "
            << " communicator on CUDA device: " << devName;
  }
  if (getNcclVersionNumber() >= NCCL_VERSION(2, 22, 0)) {
    groupEnd();
  }
}

// Abort all communicators on this rank
// Note: original name of this method is `abort`. It was renamed to
// `abortComms` to distinguish from the `abort` method below. The `abort`
// method calls `abortComms` but does more destruction than the latter.
bool ProcessGroupNCCL::abortComms(
    const std::optional<std::string>& abortReason) {
  // Remove record from global ncclCommMemPoolMapMutex before aboarting,
  // so that a new cache segment would not register to already aborted
  // communicators. Note that ncclCommMemPoolMap is a global container which may
  // contain other PG's communicators, thus we need to only erase communicators
  // for the current PG.
  {
    std::lock_guard<std::mutex> lock(ncclCommMemPoolMapMutex);
    for (auto& [_, ncclComm] : devNCCLCommMap_) {
      ncclCommMemPoolMap.erase(ncclComm);
    }
  }

  std::lock_guard<std::mutex> lock(mutex_);
  abortCommsFromMap(devNCCLCommMap_, abortReason);
  abortCommsFromMap(inInitializationCommMap_, abortReason);
  return true;
}

void ProcessGroupNCCL::dumpExtraDebuggingInfo() {
  // This extra dump is intended to capture the current snapshot of collectives
  // When this process group is terminated for some exception out of NCCL
  bool dumpExtraOnExec_ = getCvarBool(TORCH_NCCL_EXTRA_DUMP_ON_EXEC, false);
  if (dumpExtraOnExec_) {
    bool should_dump_local = false;
    bool succeeded = shouldDump_.compare_exchange_strong(
        should_dump_local,
        true,
        std::memory_order_release,
        std::memory_order_acquire);
    if (succeeded) {
      LOG(INFO) << logPrefix() << "Sending extra dumping signal";
      broadcastDumpSignal();
      // When this routine is called, exception is captured so
      // dumping by default_pg is not guaranteed due to early termination of
      // process So we call dumping manually here
      bool onlyActive = getCvarBool(TORCH_INCLUDE_ONLY_ACTIVE, false);
      // Stacktrace is not included at the moment to prevent deadlock due to GIL
      dumpDebuggingInfo(false, onlyActive);
    }
  }
}

// Abort this backend.
void ProcessGroupNCCL::abort() {
  // This will log counter for how long the abort actually takes.
  STATIC_SCOPED_WAIT_COUNTER(pytorch.ProcessGroupNCCL__abort);

  dumpExtraDebuggingInfo();
  // Don't join threads here since the purpose of this method is to abort all
  // communicators and signal the threads to exit. Joining on the threads could
  // potentially block and hence avoid it in this method.
  terminateProcessGroup_.store(true);
  watchdog_->notify();
  // launch abort asynchronously and wait for it to complete or timeout
  LOG(INFO) << logPrefix()
            << "Launching ProcessGroupNCCL abort asynchronously.";
  std::future<bool> fut =
      std::async(std::launch::async, [this]() { return this->abortComms(); });

  ::c10d::C10dLoggingData debugLog;
  waitForFutureOrTimeout(
      fut, options_->timeout, "ProcessGroup abort", debugLog, true);
  LOG(INFO) << logPrefix() << "ProcessGroupNCCL aborts successfully.";

  // We need to wait for abort to finish before we can safely shut down
  // heartbeat monitoring thread.
  heartbeatMonitor_->stop();
}

// Difference between `abort()` and `shutdown()`:
// 1. `abort()` will signal communicators to terminate all NCCL kernels
// immediately.
// 2. `shutdown()` will wait for all NCCL kernels to finish before destroying
// communicators.

// Destroy (shutdown) this backend -- normal exit.
void ProcessGroupNCCL::shutdown() {
  LOG(INFO) << logPrefix()
            << "Starting to destroy process group, flushing operations.";
  // Flush all collectives
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& it : devNCCLCommMap_) {
      auto& ncclComm = it.second;
      ncclComm->finalize();
    }
  }
  // Wait for all operations to complete.  If NCCL comm is non-blocking and
  // timeout is reach, this will throw an exception.
  for (auto& it : devNCCLCommMap_) {
    auto& ncclComm = it.second;
    // Use long interval to avoid acquiring CPU too frequently
    ncclComm->waitReady(true);
  }
  // Deregister memory pool after finalizing all collectives
  if (memPool_) {
    try {
      deregisterMemPool(memPool_.get());
    } catch (...) {
      LOG(ERROR) << logPrefix() << "Failed to deregister memory pool, ignoring";
    }
  }
  // Tell watchdog to (1) flush its queue and (2) do not use comm objects
  // anymore because I am going to destroy them now
  LOG(INFO) << logPrefix() << "Operations flushed, joining watchdog thread.";
  terminateProcessGroup_.store(true);
  watchdog_->notify();
  watchdog_->join();
  if (onCompletionHookThread_.joinable()) {
    onCompletionHookThread_.join();
  }
  // Watchdog thread exiting, retire heartbeat monitoring thread now to avoid
  // false alarm
  heartbeatMonitor_->stop();
  // Destroy the communicator, reclaim resources
  LOG(INFO) << logPrefix() << "Watchdog joined, destroying NCCL communicators.";
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& it : devNCCLCommMap_) {
      auto& ncclComm = it.second;
      ncclComm->destroy();
    }
  }
  LOG(INFO) << logPrefix() << "Destroy complete.";
}

// NOLINTNEXTLINE(bugprone-exception-escape)
ProcessGroupNCCL::~ProcessGroupNCCL() {
  LOG(INFO) << logPrefix() << "ProcessGroupNCCL destructor entered.";

  // `shutdown()` or `abort` already called. Skip the favor of disposing
  // communicators.
  if (!terminateProcessGroup_.load()) {
    // If user haven't explicitly destroy/shutdown process group, destructor
    // needs to do so
    // First print warning on first rank of each node
    if (rank_ % localDeviceCount_ == 0) {
      TORCH_WARN_ONCE(
          "WARNING: destroy_process_group() was not called before program exit, "
          "which can leak resources. For more info, please see "
          "https://pytorch.org/docs/stable/distributed.html#shutdown");
    }

    // Note 1: in distributed_c10d.py, a reference to PG is held by the global
    // context. Therefore, we are here only when the global context is tearing
    // down, which means the entire program is exiting.  At this point, user
    // will no longer care about the result of any collective, thus we can use
    // abort instead of destroy to make the destruction non-blocking.

    // TODO: Note 1 is not true in case of a C++ program using libtorch, which
    // does not have the global context mentioned. In that case, calling
    // `abort()` here could lead to corrupted result. We should consider not
    // doing anything and just let things leak. Adversarial example:
    /*
      Work routine(Tensor& t) {
        pg = ProcessGroupNCCL(…);
        w = pg.allReduce(t);
        return w;
      }
    */
    abort();
  }

  // Make sure we've told threads to stop; doesn't hurt if we'd done so before.
  // Tell watchdog and onCompletionHook:
  terminateProcessGroup_.store(true);
  watchdog_->notify();
  // Tell heartbeat thread:
  heartbeatMonitor_->stop();

  // Wait for all threads to finish before returning
  watchdog_->join();
  heartbeatMonitor_->join();
  if (onCompletionHookThread_.joinable()) {
    onCompletionHookThread_.join();
    LOG(INFO) << logPrefix()
              << "ProcessGroupNCCL onCompletionHookThread thread joined.";
  }
}

bool ProcessGroupNCCL::dumpDebuggingInfo(
    bool includeStackTrace /*=true*/,
    bool onlyActive /*=false*/) {
  // This will log counter for how long dumpDebuggingInfo actually takes.
  STATIC_SCOPED_WAIT_COUNTER(pytorch.ProcessGroupNCCL__dumpDebuggingInfo);

  // Serialize all calls to this function to avoid corrupting data, but allow
  // multiple calls in one runtime. User is responsible for preserving the
  // output file from an earlier call before a later call overwrites it.
  static std::mutex writeDebugInfoMutex;
  LOG(ERROR)
      << logPrefix()
      << "ProcessGroupNCCL preparing to dump debug info. Include stack trace: "
      << includeStackTrace << ", only active collectives: " << onlyActive;
  if (traceBufferSize_ > 0) {
    // We dump nccl trace into local disk by default and users can register
    // their customized writer by inheriting `DebugInfoWriter` via
    // `registerDebugInfoWriter`.
    auto ncclTrace = dump_nccl_trace(true, includeStackTrace, onlyActive);
    // dump_nccl_trace will hang so we don't grab the global lock until we get
    // the trace.
    std::lock_guard<std::mutex> lock(writeDebugInfoMutex);
    DebugInfoWriter& writer = DebugInfoWriter::getWriter(globalRank());
    LOG(INFO) << logPrefix() << "ProcessGroupNCCL dumping nccl trace to "
              << writer.getWriterTarget();
    writer.write(ncclTrace);
    LOG(INFO) << logPrefix() << "Flight Recorder trace successfully dumped.";
    return true;
  }
  return false;
}

void ProcessGroupNCCL::terminateProcess(const std::string& errMsg) {
  // Logging with `FATAL`, after errMsg printed, it calls `std::abort()`
  // to terminate the program execution.
  LOG(FATAL) << logPrefix() << errMsg;
}

static long computeDeltaMS(
    std::chrono::time_point<std::chrono::steady_clock> start,
    std::chrono::time_point<std::chrono::steady_clock> end) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

void ProcessGroupNCCL::setEnableNanCheck(bool enableNanCheck) {
  enableNanCheck_ = enableNanCheck;
}

std::string ProcessGroupNCCL::HeartbeatMonitor::getNCCLWatchdogTimeoutErrorMsg(
    const std::string& extraMsg) {
  return c10::str(
      pg_->logPrefix(),
      "Received a dump signal due to a collective timeout from ",
      extraMsg,
      " and we will try our best to dump the debug info. ",
      "Last enqueued NCCL work: ",
      pg_->pgStatus_->lastEnqueuedSeq,
      ", last completed NCCL work: ",
      pg_->pgStatus_->lastCompletedSeq,
      ".",
      "This is most likely caused by incorrect usages of collectives, e.g., wrong ",
      "sizes used across ranks, the order of collectives is not same for all ranks ",
      "or the scheduled collective, for some reason, didn't run. Additionally, ",
      "this can be caused by GIL deadlock or other reasons such as network errors or ",
      "bugs in the communications library (e.g. NCCL), etc. ");
}

std::string ProcessGroupNCCL::HeartbeatMonitor::getNCCLWatchdogTimeoutExitMsg(
    const std::string& exitReason) {
  return c10::str(
      pg_->logPrefix(),
      "Terminating the process after attempting to dump debug info, due to ",
      exitReason,
      ".");
}

void ProcessGroupNCCL::HeartbeatMonitor::setLastWorkListUpdateTime(
    std::chrono::time_point<std::chrono::steady_clock> time) {
  // We intentionally let the race condition to happen but this is ok
  // as long as we update the time, we know we are making progress.
  lastWorkListUpdateTime_ = time;
}

int ProcessGroupNCCL::HeartbeatMonitor::getDumpTimeout() const {
  return waitTimeoutDumpInMilSec_;
}

ProcessGroupNCCL::HeartbeatMonitor::HeartbeatMonitor(ProcessGroupNCCL* pg) {
  pg_ = pg;
  heartbeatTimeoutInSec_ =
      getCvarInt(TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC, 60 * 8 /*8 Mins*/);
  waitTimeoutDumpInMilSec_ =
      getCvarInt(TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC, 15 * 1000 /*15 Sec*/);
  coordCheckIntervalMilSec_ = getCvarInt(TORCH_NCCL_COORD_CHECK_MILSEC, 1000);
  // TODO, we should either deprecate TORCH_NCCL_DUMP_ON_TIMEOUT
  // or change its name to reflect that dump happens on exception including
  // both timeout and other errors.
  dumpOnTimeoutOrEx_ = getCvarBool(TORCH_NCCL_DUMP_ON_TIMEOUT, true);
  // logging C++ stack isn't safe. Gate it with an ENV.
  logCppStackOnUncleanShutdown_ =
      getCvarBool(TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN, true);
  watchdogHeartbeatMonitorEnabled_ =
      getCvarBool(TORCH_NCCL_ENABLE_MONITORING, true);

  // print out ENV settings for the heartbeat monitor thread.
  LOG(INFO)
      << pg_->logPrefix() << "HeartbeatMonitor environments: "
      << "TORCH_NCCL_ENABLE_MONITORING (Whether to kill program when no watchdog heartbeat detected): "
      << watchdogHeartbeatMonitorEnabled_
      << ", TORCH_NCCL_DUMP_ON_TIMEOUT: " << dumpOnTimeoutOrEx_
      << ", TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: " << waitTimeoutDumpInMilSec_
      << ", TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: " << heartbeatTimeoutInSec_
      << ", TORCH_NCCL_COORD_CHECK_MILSEC: " << coordCheckIntervalMilSec_
      << ", TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: "
      << logCppStackOnUncleanShutdown_;
}

void ProcessGroupNCCL::HeartbeatMonitor::stop() {
  terminateHeartbeatMonitorThread_.store(true);
  monitorWakeUpCV_.notify_one();
}

void ProcessGroupNCCL::HeartbeatMonitor::start() {
  TORCH_CHECK(
      !ncclHeartbeatMonitorThread_.joinable(),
      "HeartbeatMonitor thread already started");
  ncclHeartbeatMonitorThread_ =
      std::thread(&ProcessGroupNCCL::HeartbeatMonitor::runLoop, this);
}

void ProcessGroupNCCL::HeartbeatMonitor::join() {
  if (ncclHeartbeatMonitorThread_.joinable()) {
    ncclHeartbeatMonitorThread_.join();
    LOG(INFO) << pg_->logPrefix()
              << "ProcessGroupNCCL heart beat monitor thread joined.";
  }
}

void ProcessGroupNCCL::HeartbeatMonitor::runLoop() {
  c10::setThreadName("pt_nccl_heartbt");
  STATIC_SCOPED_WAIT_COUNTER(
      pytorch.ProcessGroupNCCL__HeartbeatMonitor__runLoop);

  uint64_t heartBeatCounter = 0ULL;
  std::string errorMsg;
  std::string exitReason;
  bool checkDumpSignal = (dumpOnTimeoutOrEx_ && pg_->getUid() == 0);
  int monitorPollInterval = checkDumpSignal ? coordCheckIntervalMilSec_
                                            : heartbeatTimeoutInSec_ * 1000;
  auto lastTimePollStore = std::chrono::steady_clock::now();
  auto lastTimeHeartBeatCheck = std::chrono::steady_clock::now();
  std::optional<DumpPipe> dumpPipe = std::nullopt;
  // Use a pool to temporarily store the futures to avoid blocking when the code
  // exits the scope of when future is generated by std::async.
  std::vector<std::future<bool>> futures;

  if (pg_->getUid() == 0) {
    // DumpPipe is one per-trainer process, and its convenient to name them
    // after 'global' ranks in the system, So we assume processgroup (uid)==0 is
    // the global PG and has globally unique rank ids across trainers.
    dumpPipe.emplace(pg_->globalRank());
  }
  while (true) {
    // This won't have any lock since this lock is only used here.
    // Please be aware that mutex `monitorMutex_` should not be used
    // somewhere else to avoid the deadlock.
    std::unique_lock<std::mutex> lock(monitorMutex_);
    if (monitorWakeUpCV_.wait_for(
            lock, std::chrono::milliseconds(monitorPollInterval), [&] {
              return terminateHeartbeatMonitorThread_.load();
            })) {
      // For the normal complete or user interception, monitorWakeUpCV_
      // will get notified, we early return and exit heartbeatMonitor.
      return;
    }
    auto currentTime = std::chrono::steady_clock::now();

    // We put extra functionality in the thread for the default PG (aka,
    // local_id_=0) because the signal is same across different PGs. We only
    // need to run once per process to avoid duplicate things performed in too
    // many separate threads. For example, we check a global flag on the
    // TCPStore periodically to see if any PG on any rank observed a timeout and
    // signaled peers to dump debugging info, and we avoid hammering the
    // TCPStore from all PGs on the same rank.
    if (checkDumpSignal) {
      // There are two scenarios where monitor thread will dump on timeout:
      // 1. The current rank is the first to observe a timeout in watchdog.
      // (shouldDump_ was set to true by the watchdog thread).
      // 2. Other ranks detected the timeout and signal the current rank to
      // dump. In addition, monitor threads will dump if watchdog threads has no
      // heartbeat or dumpPipe is not empty.
      if (shouldDump_.load()) {
        errorMsg = getNCCLWatchdogTimeoutErrorMsg("this local rank");
        exitReason = "collective timeout or exception";
        break;
      }
      // We poll store to see if some ranks have flagged a timeout when
      // we haven't polled for `heartbeat_timeout` seconds and there haven't
      // any work added or removed for `watchdog_timeout` seconds.
      if (computeDeltaMS(lastWorkListUpdateTime_, currentTime) >=
              kWatchdogThreadSleepMillis &&
          computeDeltaMS(lastTimePollStore, currentTime) >=
              coordCheckIntervalMilSec_) {
        lastTimePollStore = currentTime;
        auto handleError = [&](const std::string& errorMessage) {
          LOG(WARNING)
              << pg_->logPrefix()
              << "Failed to check the \"should dump\" flag on TCPStore, "
              << "(maybe TCPStore server has shut down too early), with error: "
              << errorMessage;
          // We give up for now assuming TCPStore has been torn down.
          return;
        };
        // Wrap globalStore_->check() in a try-catch block to avoid crashing if
        // the store is not available.
        bool checkExceptionDump = false;
        try {
          checkExceptionDump =
              pg_->globalStore()->check({std::string(kStoreDumpKey)});
        } catch (const c10::DistNetworkError& e) {
          handleError(e.msg());
        } catch (const std::exception& e) {
          handleError(e.what());
        }

        if (checkExceptionDump) {
          int timeOutRank = -1;
          if (!shouldDump_.load()) {
            LOG(ERROR)
                << pg_->logPrefix()
                << "Observed flight recorder dump signal from another rank via TCPStore.";
          }
          shouldDump_.store(true);
          try {
            auto vec = pg_->globalStore()->get(std::string(kStoreDumpKey));
            TORCH_CHECK_WITH(
                DistBackendError,
                vec.size() == sizeof(int),
                "Invalid size for the timeout rank ID");
            std::memcpy(&timeOutRank, vec.data(), vec.size());
          } catch (const std::exception& e) {
            LOG(ERROR) << pg_->logPrefix()
                       << "Failed to get timeout rank ID from TCPStore."
                       << e.what();
          }
          errorMsg =
              getNCCLWatchdogTimeoutErrorMsg(c10::str(" rank ", timeOutRank));
          exitReason = "collective timeout or exception";
          break;
        }
      }
    }

    if (computeDeltaMS(lastTimeHeartBeatCheck, currentTime) >=
        heartbeatTimeoutInSec_ * 1000l) {
      // Check the heart beat of watchdog thread.
      lastTimeHeartBeatCheck = currentTime;
      auto heartbeat = pg_->getWatchdogHeartbt();
      if (heartbeat != heartBeatCounter) {
        heartBeatCounter = heartbeat;
      } else {
        shouldDump_.store(true);
        // Watchdog heartbeat timeout.
        errorMsg = c10::str(
            pg_->logPrefix(),
            "ProcessGroupNCCL's watchdog got stuck for ",
            heartbeatTimeoutInSec_,
            " seconds without making progress in monitoring enqueued collectives. ",
            "This typically indicates a NCCL/CUDA API (e.g., CudaEventDestroy) hang blocking the watchdog, ",
            "and could be triggered by another thread holding the GIL inside a ",
            "CUDA api (for example, CudaEventDestroy), or other deadlock-prone behaviors.",
            "If you suspect the watchdog is not actually stuck and a longer timeout would help, ",
            "you can either increase the timeout (TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC) to a larger value "
            "or disable the heartbeat monitor (TORCH_NCCL_ENABLE_MONITORING=0)."
            "If either of aforementioned helps, feel free to file an issue to PyTorch about the short timeout "
            "or false positive abort; otherwise, please attempt to debug the hang. ");
        exitReason = "ProcessGroupNCCL watchdog hang";
        break;
      }
    }
    // process a request to dump the trace. only PG uid 0 will respond to dump
    // requests, but this is fine since all PG's feed into the same flight
    // recorder and dump. After dump, the training should continue.
    if (dumpPipe.has_value() && dumpPipe->shouldDump()) {
      // best effort dump, not waiting for the dump here
      bool onlyActive = getCvarBool(TORCH_INCLUDE_ONLY_ACTIVE, false);
      LOG(INFO) << pg_->logPrefix()
                << "Dump signal received through pipe, triggering FR dump.";
      futures.emplace_back(std::async(std::launch::async, [this, onlyActive]() {
        return this->pg_->dumpDebuggingInfo(true, onlyActive);
      }));
    }
  }
  LOG(ERROR) << errorMsg;

  // We perform some checks to help users debug the timeout/hang issue:
  // 1. Dump the nccl trace (flight recorder) to help debug the issue
  //    (timeout after waitTimeoutDumpInMilSec_, which is one minute).
  // 2. Check if there is a GIL deadlock (timeout after 300ms).
  // 3. Try to dump the c++ stacktraces (blocking and would hang,
  //    users can turn this off by set
  //    TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN=0).

  // Dump the nccl trace (flight recorder).
  if (checkDumpSignal && shouldDump_.load()) {
    // Store debug info to storage if no other thread does it. (By default to
    // local disk)
    bool dumpStackTrace = getCvarBool(TORCH_INCLUDE_STACK_TRACE, true);
    bool onlyActive = getCvarBool(TORCH_INCLUDE_ONLY_ACTIVE, false);
    ::c10d::C10dLoggingData debugLog;
    debugLog.integers["pg_id"] = static_cast<int64_t>(pg_->getUid());
    debugLog.integers["rank"] = pg_->getRank();
    debugLog.integers["global_rank"] = pg_->globalRank();
    debugLog.integers["world_size"] = pg_->getSize();
    debugLog.strings["flight_recorder_version"] = c10d::version_val_str;
    for (int i = 0; i < 2; i++) {
      std::future<bool> asyncDebugDump =
          std::async(std::launch::async, [this, dumpStackTrace, onlyActive]() {
            return this->pg_->dumpDebuggingInfo(dumpStackTrace, onlyActive);
          });

      // wait for the dump until timeout - log data
      auto complete = pg_->waitForFutureOrTimeout(
          asyncDebugDump,
          std::chrono::milliseconds(waitTimeoutDumpInMilSec_),
          "Flight recorder dump in heartbeatMonitor",
          debugLog,
          false);

      if (complete) {
        LOG(INFO)
            << pg_->logPrefix()
            << "Finished flight recorder successfully. Output can be analyzed using the fr_trace script.";
        if (i > 0) {
          debugLog.strings["exception_msg"] = "Dump with stack trace failed.";
        }
        break;
      }
      // If we failed to dump, try dumping without stack trace in the 2nd
      // iteration.
      dumpStackTrace = false;
      futures.emplace_back(std::move(asyncDebugDump));
    }
    debugLog.integers["trace_enabled"] = int64_t(dumpStackTrace);
    auto logger = c10d::C10dLogger::getLogger();
    if (logger) {
      logger->log(debugLog);
    }
  }

  // GIL deadlock check.
  if (get_gil_checker() != nullptr) {
    auto fut = launchAsyncGilCheck();
    auto kGilCheckTimeout = std::chrono::milliseconds(300);
    auto futStatus = fut.wait_for(kGilCheckTimeout);
    if (futStatus != std::future_status::ready) {
      TORCH_CHECK(
          futStatus != std::future_status::deferred,
          "Expected the future to have been launched eagerly.");
      LOG(ERROR)
          << pg_->logPrefix()
          << "Could not acquire GIL within 300 ms on exit, possible GIL induced hang";
    }
  } else {
    VLOG(2)
        << pg_->logPrefix()
        << "GIL checker was not registered, perhaps this is a no-python build?";
  }

  // Dump the c++ stacktraces.
  auto& cpp_dumper = get_cpp_trace_dumper();
  if (logCppStackOnUncleanShutdown_ && cpp_dumper.has_value()) {
    LOG(INFO) << pg_->logPrefix() << "Dumping c++ stacktraces:";
    cpp_dumper.value()([&](const std::string& line) {
      LOG(INFO) << pg_->logPrefix() << line;
    });
    LOG(INFO) << pg_->logPrefix() << "Finished c++ stacktraces dump.";
  }

  // There are two possible cases for the watchdog thread exit:
  // Case one: desync report runs quickly, and it follows the step:
  // collective timeout -> desync -> exception handling -> throwing exception.
  // The program will exit because of exception thrown and the code below will
  // not be run.
  //
  // Case two: desync might be slow or get stuck and we need to wait
  // extra time to avoid we kill the program too early.
  //
  // Or we get stuck in destructors, we will sleep for some time before calling
  // std::abort() to kill the whole process.
  if ((pg_->terminateProcessGroup_.load() || shouldDump_.load()) &&
      !terminateHeartbeatMonitorThread_.load()) {
    std::this_thread::sleep_for(std::chrono::seconds(heartbeatTimeoutInSec_));
    LOG(INFO)
        << pg_->logPrefix() << "slept for " << heartbeatTimeoutInSec_
        << " because we want to wait longer to verify there is indeed a watchdog hang.";
  }

  // At this point, we either already sleep for another `heartbeatTimeoutInSec_`
  // or the thread has finished. Because we don't want to block the monitor
  // thread, so We mark the thread detach and the dump of debug info becomes
  // "best effort". If the process exit normally, marking it detach also makes
  // sense because we don't really care about dumping the debug info.

  // We already log completion inside the thread, so it may not be necessary to
  // check the return value here.  We mainly use a future so we can exit early
  // if done.
  if (!terminateHeartbeatMonitorThread_.load()) {
    // Create a error message reported from MonitorThread, so
    // we throw exception and make the whole process to be killed.
    // TODO(fduwjj): After having a hang debug wiki, we need to update the wiki
    // url here.
    if (watchdogHeartbeatMonitorEnabled_) {
      pg_->terminateProcess(getNCCLWatchdogTimeoutExitMsg(exitReason));
    } else {
      // Ideally we want to merge this one with the above one, but we are going
      // to remove the kill switch for monitor thread soon, so we keep this one
      // for now.
      LOG(ERROR)
          << pg_->logPrefix()
          << "ProcessGroupNCCL monitor thread is disabled, but would have terminated the process"
          << "after attempting to dump debug info, due to " << exitReason
          << ".";
    }
  }
}

ProcessGroupNCCL::Watchdog::Watchdog(ProcessGroupNCCL* pg) {
  pg_ = pg;
  heartbeat_ = 1ULL;
  rethrowCUDAErrors_ = getCvarBool(TORCH_NCCL_RETHROW_CUDA_ERRORS, true);
  propagatePgError_ = getCvarBool(TORCH_NCCL_PROPAGATE_ERROR, false);
  desyncDebug_ = getCvarBool(TORCH_NCCL_DESYNC_DEBUG, false) ||
      (pg_->dist_debug_level_ >= DebugLevel::Detail);

  // print out ENV settings for the watchdog thread.
  LOG(INFO) << pg_->logPrefix() << "PGNCCL Watchdog environments: "
            << "TORCH_NCCL_RETHROW_CUDA_ERRORS: " << rethrowCUDAErrors_
            << ", TORCH_NCCL_PROPAGATE_ERROR: " << propagatePgError_
            << ", TORCH_NCCL_DESYNC_DEBUG: " << desyncDebug_;

  // Enable Desync Debugger per user setting
  if (desyncDebug_) {
    desyncDebugger_.init(
        pg_->getRank(),
        pg_->getSize(),
        pg_->globalRank(),
        pg_->getUid(),
        pg_->store_);
  }
}

void ProcessGroupNCCL::Watchdog::notify() {
  workMetaListCV_.notify_one();
}

void ProcessGroupNCCL::Watchdog::start() {
  TORCH_CHECK(
      !ncclCommWatchdogThread_.joinable(), "Watchdog thread already started");
  ncclCommWatchdogThread_ = std::thread(&ProcessGroupNCCL::Watchdog::run, this);
}

void ProcessGroupNCCL::Watchdog::join() {
  if (ncclCommWatchdogThread_.joinable()) {
    ncclCommWatchdogThread_.join();
    LOG(INFO) << pg_->logPrefix() << "ProcessGroupNCCL watchdog thread joined.";
  }
}

void ProcessGroupNCCL::Watchdog::run() {
  c10::setThreadName("pt_nccl_watchdg");
  STATIC_SCOPED_WAIT_COUNTER(pytorch.ProcessGroupNCCL__Watchdog__run);

  try {
    VLOG(2) << pg_->logPrefix() << "Process group watchdog thread started!";
    pg_->heartbeatMonitor_->start();
    runLoop();
    VLOG(2) << pg_->logPrefix()
            << "Process group watchdog thread terminated normally";
  } catch (std::exception& e) {
    // This condition is triggered when any routine in watchdog gets an
    // exception
    pg_->dumpExtraDebuggingInfo();
    if (std::string(e.what()).find("driver shutting down") !=
        std::string::npos) {
      VLOG(2)
          << pg_->logPrefix()
          << "main process destroyed cuda before watchdog loop exited, terminating watchdog."
          << " (Watchdog caught exception: " << e.what();

    } else {
      // Append error message reported from runLoop
      const auto exitMsg = c10::str(
          pg_->logPrefix(),
          "Process group watchdog thread terminated with exception: ",
          e.what());
      LOG(ERROR) << exitMsg;
      if (C10_LIKELY(rethrowCUDAErrors_) ||
          !(std::string(e.what()).find("CUDA Error"))) {
        // TODO(whc) clean up the rethrow - why is it stored in a class var and
        // rethrown?
        watchDogException_ =
            std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exitMsg));
        std::rethrow_exception(watchDogException_);
      }
    }
  } catch (...) {
    const auto exitMsg = c10::str(
        pg_->logPrefix(),
        "Process group watchdog thread terminated with exception: unknown");
    LOG(ERROR) << exitMsg;
    watchDogException_ =
        std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exitMsg));
    std::rethrow_exception(watchDogException_);
  }
}

int ProcessGroupNCCL::Watchdog::getSignalSrcRank(
    c10::intrusive_ptr<Store>& store,
    const std::string& signal) {
  // This function is 'non blocking'. We first 'check' if the key exists in the
  // store, then read/get the value only if the key exists.
  int srcRank = -1;
  bool signalExists = false;
  try {
    signalExists = store->check({signal});
  } catch (const std::exception& e) {
    LOG(WARNING) << pg_->logPrefix() << "Failed to check the signal " << signal
                 << " on TCPStore, " << e.what();
  }
  if (!signalExists) {
    return srcRank;
  }

  // key exists, now read and parse the value (source rank)
  std::vector<uint8_t> vec;
  try {
    vec = store->get(std::string(signal));
  } catch (const std::exception& e) {
    LOG(ERROR) << pg_->logPrefix() << "Failed to get source rank of the signal "
               << signal << " from TCPStore." << e.what();
  }
  TORCH_CHECK_WITH(
      DistBackendError,
      vec.size() == sizeof(int),
      "Invalid size for the timeout rank ID");
  std::memcpy(&srcRank, vec.data(), vec.size());
  return srcRank;
}

void ProcessGroupNCCL::Watchdog::checkAndSetRemoteError() {
  // if the error is already set, no need to check again
  if (pg_->getError() != ErrorType::SUCCESS) {
    return;
  }
  // key/signal to read from the tcpstore is a string and pg specific:
  // format is: remote_error:pg_uid
  int remoteErrorRank = getSignalSrcRank(
      pg_->store_, std::string(kStoreErrorSignalKey) + ':' + pg_->pg_uid_);
  if (remoteErrorRank != -1) {
    std::lock_guard<std::mutex> lock(pg_->errorMutex_);
    pg_->error_ = ErrorType::REMOTE_ERROR;
    LOG(ERROR) << c10::str(
        pg_->logPrefix(),
        " remote error detected from rank: ",
        remoteErrorRank);
  }
}

void ProcessGroupNCCL::Watchdog::runLoop() {
  bool done = false;
  pg_->heartbeatMonitor_->setLastWorkListUpdateTime(
      std::chrono::steady_clock::now());
  auto lastStatusUpdateTime = std::chrono::steady_clock::now();
  std::list<ProcessGroupNCCL::WorkNCCL> completedWorkList;

  while (!done || !pg_->terminateProcessGroup_.load()) {
    std::unique_lock<std::mutex> lock(pg_->workMetaListMutex_);
    // We busy-poll the work vector every kWatchdogThreadSleepMillis
    // milliseconds as long as the atomic is True.
    workMetaListCV_.wait_for(
        lock,
        std::chrono::milliseconds(kWatchdogThreadSleepMillis),
        [&]() -> bool { return pg_->terminateProcessGroup_.load(); });
    // Bump up heart beat by one.
    heartbeat_++;

// Some versions of GLOG support less-spammy version of LOG_EVERY_MS
// in which case we don't want to spam the logs.
#ifdef LOG_EVERY_MS
    // Log the progress of this PG periodically
    C10_LOG_EVERY_MS(INFO, kWorkStatusUpdatePeriodMs) << c10::str(
        logPrefix(),
        "NCCL Work update periodically: ",
        "last enqueued NCCL work: ",
        pg_->pgStatus_->lastEnqueuedSeq,
        ", last completed NCCL work: ",
        pg_->pgStatus_->lastCompletedSeq,
        ".");
#endif // LOG_EVERY_MS
    auto logger = ::c10d::C10dLogger::getLogger();
    if (logger &&
        computeDeltaMS(
            lastStatusUpdateTime, std::chrono::steady_clock::now()) >=
            kWorkStatusUpdatePeriodMs) {
      ::c10d::C10dLoggingData data;
      // logging integers
      data.integers["pg_id"] = static_cast<int64_t>(pg_->local_id_);
      data.integers["rank"] = pg_->rank_;
      data.integers["global_rank"] = pg_->globalRank();
      data.integers["last_enqueued_work"] = pg_->pgStatus_->lastEnqueuedSeq;
      data.integers["last_started_work"] = pg_->pgStatus_->lastStartedSeq;
      data.integers["last_completed_work"] = pg_->pgStatus_->lastCompletedSeq;
      data.integers["last_enqueued_numel_in"] =
          static_cast<int64_t>(pg_->pgStatus_->lastEnqueuedNumelIn);
      data.integers["last_enqueued_numel_out"] =
          static_cast<int64_t>(pg_->pgStatus_->lastEnqueuedNumelOut);
      data.integers["last_completed_numel_in"] =
          static_cast<int64_t>(pg_->pgStatus_->lastCompletedNumelIn);
      data.integers["last_completed_numel_out"] =
          static_cast<int64_t>(pg_->pgStatus_->lastCompletedNumelOut);
      data.integers["last_started_numel_in"] =
          static_cast<int64_t>(pg_->pgStatus_->lastStartedNumelIn);
      data.integers["last_started_numel_out"] =
          static_cast<int64_t>(pg_->pgStatus_->lastStartedNumelOut);
      // logging strings
      data.strings["last_enqueued_work_name"] =
          pg_->pgStatus_->lastEnqueuedWorkName;
      data.strings["last_started_work_name"] =
          pg_->pgStatus_->lastStartedWorkName;
      data.strings["last_completed_work_name"] =
          pg_->pgStatus_->lastCompletedWorkName;
      data.strings["pg_name"] = pg_->pg_uid_;
      data.strings["pg_desc"] = pg_->pg_desc_;
      logger->log(data);
      lastStatusUpdateTime = std::chrono::steady_clock::now();
    }

    if (propagatePgError_) {
      // Check and set remote error if it has not been set before
      checkAndSetRemoteError();
    }

    for (auto it = pg_->workMetaList_.begin(); it != pg_->workMetaList_.end();
         /* no increment */) {
      auto& work = *it;
      // When terminateProcessGroup_ is true, communicators have already been
      // aborted, So cannot check exception based on them. But watchdog needs to
      // finish the check for the works that have already been enqueued to
      // workMetaList_

      // check NCCL errors first
      if (!pg_->terminateProcessGroup_.load()) {
        work.checkAndSetException();
      }

      if (work.exception()) {
        // set the error to the first error found
        std::lock_guard<std::mutex> lock(pg_->errorMutex_);
        if (pg_->error_ == ErrorType::SUCCESS) {
          pg_->error_ = ErrorType::COMM_ERROR;
        }
      }

      // Then check if work has timed out
      // Skip if work has encountered an error
      bool timedout = !work.exception() && work.checkTimeout();

      // Report desync state in case of timeout (if TORCH_NCCL_DESYNC_DEBUG is
      // turned on; otherwise, run() is no-op)
      if (timedout) {
        std::lock_guard<std::mutex> lock(pg_->errorMutex_);
        if (pg_->error_ == ErrorType::SUCCESS) {
          pg_->error_ = ErrorType::TIMEOUT;
        }
        desyncDebugger_.run();
      }

      // If work hits an exception (either an error or timeout)
      if (work.exception()) {
        LOG(ERROR) << c10::str(
            pg_->logPrefix(),
            " failure detected by watchdog at work sequence id: ",
            work.seq_,
            " PG status: last enqueued work: ",
            pg_->pgStatus_->lastEnqueuedSeq,
            ", last completed work: ",
            pg_->pgStatus_->lastCompletedSeq);

        // Print the traceback of the collective at call time
        work.printTraceback();

        // broadcast remote error signal to all other ranks in this specific PG.
        // key/signal to write in the tcpstore is a string and pg specific:
        // format is: remote_error:pg_uid
        if (propagatePgError_) {
          pg_->broadcastSignal(
              pg_->store_,
              std::string(kStoreErrorSignalKey) + ':' + pg_->pg_uid_,
              pg_->rank_);
        }

        // try to notify other ranks via global TCPStore to dump the flight
        // recorder when a collective timeout or exception happens. Flight
        // recorder behavior is independent of desync Debug.
        pg_->broadcastDumpSignal();
        // Give time for dumping before throwing exception for all ranks.
        // It is hard to presume or control what the pattern of watchdog might
        // look like, so it is better to let all ranks universally sleep for a
        // short period of time, in this case, 60 seconds, which is also the
        // maximum time we leave for FR dump.
        std::this_thread::sleep_for(std::chrono::milliseconds(
            pg_->heartbeatMonitor_->getDumpTimeout() * 4));

        if (SHOULD_CLEAN_UP(pg_->asyncErrorHandling_)) {
          // Abort work and corresponding communicators
          work.abort();
          // PG level abort, which would abort all other communicators on this
          // rank
          pg_->abortComms();
        }
        // Throw exception
        work.handleException(pg_->asyncErrorHandling_);
      }

      // Work status logging for desync debug
      desyncDebugger_.logWorkStart(work);

      // allow watchdog to do an event query on a side thread
      at::cuda::CUDAGuard device_guard(work.ncclEndEvent_->device_index());
      at::cuda::CUDAStreamCaptureModeGuard g{cudaStreamCaptureModeThreadLocal};

      // a work could be started but not completed, so we should not update
      // lastStartedSeq and lastStartedOpName if the work state is checked
      // multiple times after the start
      if (pg_->pgStatus_->lastStartedSeq < static_cast<int64_t>(work.seq_) &&
          work.isStarted()) {
        pg_->pgStatus_->lastStartedSeq = static_cast<int64_t>(work.seq_);
        pg_->pgStatus_->lastStartedWorkName = opTypeToString(work.opType_);
        pg_->pgStatus_->lastStartedNumelIn = work.numelIn_;
        pg_->pgStatus_->lastStartedNumelOut = work.numelOut_;
      }

      // Clean up completed work
      if (work.isCompleted()) {
        // In case user didn't call `work.wait()` with async collectives,
        // watchdog would unstage the stashed tensors when detecting completion
        // of the collective, to prevent ProcessGroupNCCL from holding reference
        // to those tensors forever.
        // work.stashed_for_allocator_safety_->unstash();
        // Update: it seems directly unstashing from watchdog thread would cause
        // some rare problems. We thus move the unstashing to main thread,
        // triggered by a next user call, see `workEnqueue`. But `work` is going
        // to be destructed, so we transfer the work's shelf to a shelves
        // structure owned by the PG.
        if (!work.stashed_for_allocator_safety_->empty()) {
          std::lock_guard<std::mutex> lock(pg_->shelvesMutex_);
          // We are just pushing back a shared_ptr here, so the cost should be
          // minimal
          pg_->shelvesToUnstash_.push_back(work.stashed_for_allocator_safety_);
        }

        if (pg_->enableTiming_ && logger) {
          ::c10d::C10dLoggingData data;
          // logging integers
          data.strings["collective_duration"] =
              std::to_string(work.getDuration());
          data.integers["global_rank"] = pg_->globalRank();
          data.integers["pg_id"] = static_cast<int64_t>(pg_->local_id_);
          data.strings["pg_name"] = pg_->pg_uid_;
          data.strings["pg_desc"] = pg_->pg_desc_;
          data.integers["pg_rank"] = pg_->rank_;
          data.integers["world_size"] = pg_->size_;
          data.strings["comm_backend"] = "nccl";
          data.strings["comm_backend_version"] = getNcclVersion();
          // TODO: We see errors for this line, revert it for now.
          data.strings["collective_stack"] = "";
          data.strings["collective_name"] = opTypeToString(work.opType_);
          logger->log(data);
        }

        // Work status logging for desync debug
        desyncDebugger_.logWorkEnd(work);

        if (work.futureWorkResult_ && work.finishedGPUExecutionInternal() &&
            !work.futureWorkResult_->completed()) {
          work.futureWorkResult_->markCompleted(
              at::IValue(static_cast<uint8_t>(WorkResult::SUCCESS)));
        }
        {
          // Reset the timeout and first work if the work is completed.
          std::lock_guard<std::mutex> timeoutLock(pg_->mtxTimeoutExtension_);
          if (work.ownedEphermeralTimeout_.count() > 0) {
            pg_->ephemeralTimeoutActive_ -= work.ownedEphermeralTimeout_;
            pg_->ephemeralTimeoutInflight_ -= work.ownedEphermeralTimeout_;
          }
        }
        pg_->pgStatus_->lastCompletedSeq = static_cast<int64_t>(work.seq_);
        pg_->pgStatus_->lastCompletedWorkName = opTypeToString(work.opType_);
        pg_->pgStatus_->lastCompletedNumelIn = work.numelIn_;
        pg_->pgStatus_->lastCompletedNumelOut = work.numelOut_;
        FlightRecorderCUDA::get()->retire_id(
            work.trace_id_, work.trace_reset_epoch_, true);
        if (pg_->onCompletionHook_) {
          // Move Work object to completedWorkList_ to be consumed by the hook
          // thread
          {
            const std::lock_guard<std::mutex> lock(
                pg_->completedWorkListMutex_);
            pg_->completedWorkList_.splice(
                pg_->completedWorkList_.end(), pg_->workMetaList_, it++);
          }
          pg_->completedWorkListCV_.notify_one();
        } else {
          it = pg_->workMetaList_.erase(it);
          pg_->heartbeatMonitor_->setLastWorkListUpdateTime(
              std::chrono::steady_clock::now());
        }
      } else {
        // Increment the iterator if the current WorkNCCL object is not
        // completed.
        ++it;
      }
      // Increment heartbeat after each work processed,
      // in case processing is slowed down (but not hung) by cuda api contention
      heartbeat_++;
    }
    done = pg_->workMetaList_.empty();
  }
}

uint64_t ProcessGroupNCCL::Watchdog::getHeartbt() const {
  return heartbeat_.load();
}

void ProcessGroupNCCL::Watchdog::setDesyncDebug(bool desyncDebug) {
  desyncDebug_ = desyncDebug;
}

// Initialize and enable DesyncDebugger
void ProcessGroupNCCL::DesyncDebugger::init(
    int rank,
    int size,
    int globalRank,
    int pgId,
    c10::intrusive_ptr<Store> store) {
  rank_ = rank;
  size_ = size;
  globalRank_ = globalRank;
  pgId_ = pgId;
  store_ = std::move(store);
  enabled_ = true;
  traceKeyStart_ = getTraceStartKey("NCCL", rank);
  traceKeyEnd_ = getTraceEndKey("NCCL", rank);
}

// Run desync debug. This function is called by watchdog at time of timeout.
void ProcessGroupNCCL::DesyncDebugger::run() {
  if (!enabled_)
    return;
  auto logPrefix = c10::str("Rank ", rank_);
  ::c10d::C10dLoggingData log;
  log.integers["pg_id"] = pgId_;
  log.integers["rank"] = rank_;
  log.integers["global_rank"] = globalRank_;
  log.integers["world_size"] = size_;
  // Use this to differentiate between flight recorder and desync debug report.
  log.strings["flight_recorder_version"] = "-1";

  try {
    std::string desyncMsg = retrieveDesyncReport(store_, "NCCL", rank_, size_);
    log.strings["status"] = "SUCCESS";
    LOG(ERROR) << logPrefix << desyncMsg;
  } catch (const std::exception& e) {
    log.strings["status"] = "EXCEPTION";
    log.strings["exception_msg"] = e.what();
    enabled_ = false;
    LOG(ERROR) << logPrefix
               << " Failed to retrieve TORCH_NCCL_DESYNC_DEBUG report. "
               << " Please file an issue. Error: " << e.what();
  } catch (...) {
    enabled_ = false;
    log.strings["status"] = "EXCEPTION";
    log.strings["exception_msg"] = "Unknown exception";
    LOG(ERROR)
        << logPrefix
        << " Failed to rerieve TORCH_NCCL_DESYNC_DEBUG report with unknown error."
        << " Please file an issue.";
  }
  auto logger = c10d::C10dLogger::getLogger();
  if (logger) {
    logger->log(log);
  }
}

// Log work start to store.
void ProcessGroupNCCL::DesyncDebugger::logWorkStart(WorkNCCL& work) {
  if (!enabled_)
    return;
  if (work.startTraceUpdated_)
    return;

  work.startTraceUpdated_ = true;
  // If not successful, disable the debugger
  enabled_ = c10d::traceUpdate(
      store_, traceKeyStart_, work.seq_, opTypeToString(work.opType_));
}

// Log work end to store.
void ProcessGroupNCCL::DesyncDebugger::logWorkEnd(WorkNCCL& work) {
  if (!enabled_)
    return;

  // In case the start of the work hasn't been logged
  if (!work.startTraceUpdated_) {
    logWorkStart(work);
  }

  // If not successful, disable the debugger
  enabled_ = c10d::traceUpdate(
      store_, traceKeyEnd_, work.seq_, opTypeToString(work.opType_));
}

// We want to have both PG ID and global unique ID (guid) for the logging
// prefix. PG ID records how many ProcessGroupNCCL objects were created on a
// specific rank and is a stable index across ranks, which lets users reason
// about, for example, the second PG we initialized on this rank is for FSDP,
// and corresponds with PG ID = 1 on other ranks as well. Unlike PG ID, guid (or
// group name) is a global unique ID across ranks. The guid is either a hash of
// all the ranks in the group or a counter of how many times
// `_process_group_name` is called, essentially it means how many times we
// have PGs users have created. Before using split_group, even if
// we are creating a new sub-PG, all ranks have to call the API at the same
// time, and this makes `group_name` a unique identifier for a group (PG).
std::string ProcessGroupNCCL::createLogPrefix() const {
  if (!pg_desc_.empty() && pg_desc_ != "undefined") {
    return c10::str(
        "[PG ID ",
        local_id_,
        " PG GUID ",
        pg_uid_,
        "(",
        pg_desc_,
        ") Rank ",
        rank_,
        "] ");
  }
  return c10::str(
      "[PG ID ", local_id_, " PG GUID ", pg_uid_, " Rank ", rank_, "] ");
}

const std::string& ProcessGroupNCCL::logPrefix() const {
  return logPrefix_;
}

const int& ProcessGroupNCCL::globalRank() const {
  static int globalRank = rank_;
  return globalRank;
}

const c10::intrusive_ptr<Store>& ProcessGroupNCCL::globalStore() const {
  return globalStore_;
}

const std::vector<uint64_t>& ProcessGroupNCCL::groupRanks() const {
  if (options_->global_ranks_in_group.empty() && local_id_ == 0) {
    static std::vector<uint64_t> globalRanks(size_);
    std::iota(globalRanks.begin(), globalRanks.end(), 0);
    return globalRanks;
  }
  return options_->global_ranks_in_group;
}

void ProcessGroupNCCL::addEphemeralTimeout(
    const std::chrono::milliseconds& timeout) {
  std::lock_guard<std::mutex> timeoutLock(mtxTimeoutExtension_);
  ephemeralTimeoutActive_ += timeout;
}

bool ProcessGroupNCCL::verifyWorkTimeoutForTest(
    const c10::intrusive_ptr<Work>& work,
    const std::chrono::milliseconds& timeout) {
  // Since collective returns a c10d::Work, we need to cast it to WorkNCCL.
  if (auto workNCCL = c10::dynamic_intrusive_pointer_cast<WorkNCCL>(work)) {
    // workNCCL is now a c10::intrusive_ptr<WorkNCCL>
    return workNCCL->opTimeout_ == timeout;
  }
  C10_THROW_ERROR(
      DistBackendError, "Non c10d::WorkNCCL object returned from collective");
}

void ProcessGroupNCCL::broadcastSignal(
    c10::intrusive_ptr<Store>& store,
    const std::string& signal,
    int srcRank) {
  try {
    auto vec = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(&srcRank),
        reinterpret_cast<uint8_t*>(&srcRank) + sizeof(srcRank));
    store->set(signal, vec);
    LOG(INFO) << logPrefix() << "Broadcasting signal " << signal
              << " to other ranks via TCPStore.";
  } catch (const std::exception& e) {
    LOG(ERROR) << logPrefix() << "Failed to broadcast signal " << signal
               << " through TCPStore. Error: " << e.what();
  }
}

void ProcessGroupNCCL::broadcastDumpSignal() {
  // broadcast dump signal to all other global ranks.
  broadcastSignal(globalStore_, std::string(kStoreDumpKey), globalRank());
  // signal the local rank to start dumping
  if (!shouldDump_.load()) {
    LOG(ERROR) << logPrefix() << "First PG on this rank to signal dumping.";
    // signal the monitor thread on PG0 to start dumping
    shouldDump_.store(true);
  }
}

// NCCL recommends to evenly distribute ncclUniqueIds across the ranks
// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#init-rank-config
// Let’s consider an example where:
// nRanks = 10 (total ranks),
// nIds = 3 (roots),
// rmr = 10 % 3 = 1 (1 larger group),
// rpr = 10 / 3 = 3 (base number of ranks per group).
// rlim = 4
// Output root:
// For ranks [0, 1, 2, 3], root rank is 0 and index is 0.
// For ranks [4, 5, 6], root rank is 4 and index is 1.
// For ranks [7, 8, 9], root rank is 7 and index is 2.
static int getRootIndex(const int rank, const int nRanks, const int nIds) {
  const int rmr = nRanks % nIds;
  const int rpr = nRanks / nIds;
  // For the first rmr roots, we assign one more rank to the root.
  const int rlim = rmr * (rpr + 1);
  if (rank < rlim) {
    // Root with `rpr + 1` ranks, (0, 1, 2, ..., rmr - 1).
    return rank % (rpr + 1) ? -1 : rank / (rpr + 1);
  } else {
    // Root with `rpr` ranks, (rmr, rmr + 1, ..., nIds - 1).
    return (rank - rlim) % rpr ? -1 : ((rank - rlim) / rpr) + rmr;
  }
}

void ProcessGroupNCCL::runHookLoop() {
  c10::setThreadName("pt_nccl_runhook");

  bool done = false;
  while (!done || !terminateProcessGroup_.load()) {
    std::unique_lock<std::mutex> lock(completedWorkListMutex_);
    // We busy-poll the work vector every kWatchdogThreadSleepMillis
    // milliseconds as long as the atomic is True.
    completedWorkListCV_.wait_for(
        lock,
        std::chrono::milliseconds(kWatchdogThreadSleepMillis),
        [&]() -> bool {
          return !completedWorkList_.empty() || terminateProcessGroup_.load();
        });

    try {
      for (auto it = completedWorkList_.begin(); it != completedWorkList_.end();
           /* no increment */) {
        const WorkNCCL& work = *it;
        // Hook might grab GIL, unlock first to prevent deadlock
        lock.unlock();

        auto timeFinished = std::chrono::steady_clock::now();
        auto timeStarted =
            timeFinished +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                work.workStartTime_ - std::chrono::steady_clock::now());
        onCompletionHook_(std::make_shared<WorkInfo>(
            work.retrieveOpType(), // OpType
            work.getSequencenumber(), // seq
            timeStarted, // timeStarted
            timeFinished, // timeFinished
            std::chrono::duration<float, std::milli>(
                work.getDuration()) // activeDuration
            ));

        lock.lock();
        it = completedWorkList_.erase(it);
      }
    } catch (std::exception& e) {
      if (std::string(e.what()).find("driver shutting down") !=
          std::string::npos) {
        LOG(INFO)
            << logPrefix()
            << "main process destroyed cuda before runHookLoop exited, terminating runHookLoop."
            << " (runHookLoop caught exception: " << e.what();

      } else {
        // PythonOnCompletionHook has already extracted Python exception message
        // and wrapped it with a cpp one. So we no longer need to acquire GIL
        // here.
        const auto errorStr = c10::str(
            "Caught exception on rank ",
            rank_,
            " while running onCompletion hook for ProcessGroupNCCL: ",
            e.what(),
            ". Aborting all communicators.");

        // No need to call abort() on WorkNCCL here as that collective has
        // already finished successfully at this point. We just need to abort
        // the process Abort all NCCL Communicators on this ProcessGroupNCCL
        // instance.
        abortComms(errorStr);
      }
    }

    // Lock is still acquired at this point
    done = completedWorkList_.empty();
  }
}

std::exception_ptr ProcessGroupNCCL::WorkNCCL::checkForNCCLErrors() {
  return checkForNCCLErrorsInternal(ncclComm_);
}

std::exception_ptr ProcessGroupNCCL::checkForNCCLErrors(
    std::shared_ptr<NCCLComm>& ncclComm) {
  return checkForNCCLErrorsInternal(ncclComm);
}

std::exception_ptr ProcessGroupNCCL::checkForNCCLErrorsInternal(
    std::shared_ptr<NCCLComm>& ncclComm) {
  // Prioritize commFailureReason over checkForNcclError() result if
  // commFailureReason is set.
  auto commFailureReason = ncclComm->getNcclCommFailureReason();
  if (commFailureReason != std::nullopt) {
    return std::make_exception_ptr(C10_BUILD_ERROR(
        DistBackendError,
        c10::str(
            "NCCL communicator encountered error set by ProcessGroupNCCL: ",
            *commFailureReason)));
  }
  ncclResult_t ncclAsyncErr = ncclComm->checkForNcclError();
  // When nonblocking mode is enabled by TORCH_NCCL_USE_COMM_NONBLOCKING,
  // ncclInProgress could be returned when there are pending NCCL calls.
  // In this case, no exception should be thrown
#ifdef NCCL_HAS_COMM_NONBLOCKING
  // ncclInProgress is defined only if NCCL_HAS_COMM_NONBLOCKING is defined
  if (ncclAsyncErr != ncclSuccess && ncclAsyncErr != ncclInProgress) {
#else
  if (ncclAsyncErr != ncclSuccess) {
#endif // NCCL_HAS_COMM_NONBLOCKING
    return std::make_exception_ptr(C10_BUILD_ERROR(
        DistBackendError,
        "NCCL error: " + ncclGetErrorWithVersion(ncclAsyncErr) + "\n" +
            getNcclErrorDetailStr(ncclAsyncErr)));
  }

  return nullptr;
}

void ProcessGroupNCCL::broadcastUniqueNCCLID(
    ncclUniqueId* ncclID,
    bool isSingleP2POp,
    const std::string& p2pKey,
    int p2pRank) {
  // For collective operations:
  // For every NCCL communicator that we create we need to broadcast
  // a unique ID from rank 0 to all other ranks. This broadcast is
  // done by rank 0 setting a key in the store and all other ranks
  // retrieving the contents of that key. A single process group
  // may create multiple NCCL communicators, so we use a sequence
  // number to differentiate between them.
  // For single point-to-point operations:
  // The sequence number will only be increased on 2 out of all the
  // processes in a Process Group. So all following collective
  // operations will see different sequence numbers which will cause
  // runtime errors. To avoid that, use the src:target pair instead
  // of sequence number for p2p communications.

  std::string storeKey;
  RECORD_PARAM_COMMS(
      std::make_tuple(0, false), // seq
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      rank_, // TODO: this might not work for P2P
      "broadcastUniqueNCCLID", // collective name
      0, // inNelems
      0, // outNelems
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart_, // globalRankStart_
      globalRankStride_, // globalRankStride_
      size_); // worldSize

  if (!isSingleP2POp) {
    storeKey = std::to_string(ncclCommCounter_++);
  } else {
    storeKey = p2pKey;
  }
  if (rank_ == 0 || (isSingleP2POp && p2pRank == 0)) {
    auto vec = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(ncclID),
        reinterpret_cast<uint8_t*>(ncclID) + NCCL_UNIQUE_ID_BYTES);
    store_->set(storeKey, vec);
  } else {
    try {
      auto vec = store_->get(storeKey);
      TORCH_CHECK_WITH(
          DistBackendError,
          vec.size() == NCCL_UNIQUE_ID_BYTES,
          "Invalid size for ncclUniqueId");
      std::memcpy(ncclID, vec.data(), vec.size());
    } catch (const std::exception& e) {
      std::string exceptionMsg = c10::str(
          "[",
          rank_,
          "] is setting up NCCL communicator and "
          "retrieving ncclUniqueId from [0] via c10d key-value store by key '",
          storeKey,
          "', but store->get('",
          storeKey,
          "') got error: ");
      C10_THROW_ERROR(
          DistBackendError,
          exceptionMsg + e.what() +
              ". This may indicate a possible application crash on rank 0 or a network set up issue.");
    } catch (...) {
      C10_THROW_ERROR(
          DistBackendError,
          c10::str(
              "Unknown exception while [",
              rank_,
              "] is setting up NCCL communicator and "
              "retrieving ncclUniqueId from [0] via c10d key-value store by key '",
              storeKey,
              "'",
              ". This may indicate a possible application crash on rank 0 or a network set up issue."));
    }
  }
}

// We want to all-gather unique NCCL IDs from all roots using TCPStore.
// This is first done by setting the ID by each root and then `multiGet` by all
// ranks.
void ProcessGroupNCCL::allgatherUniqueNCCLIDs(
    int rootIdx,
    ncclUniqueId* ncclID,
    std::vector<ncclUniqueId>& ncclIDs) {
  std::vector<std::string> storeKeys;
  std::vector<std::vector<uint8_t>> results;
  RECORD_PARAM_COMMS(
      std::make_tuple(0, false), // seq
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      rank_, // rank
      "allgatherUniqueNCCLIDs", // collective name
      0, // inNelems
      0, // outNelems
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart_, // globalRankStart_
      globalRankStride_, // globalRankStride_
      size_); // worldSize

  for (size_t r = 0; r < ncclIDs.size(); r++) {
    storeKeys.emplace_back("UniqueNCCLID:" + std::to_string(r));
  }
  // For non-root rank, rootIdx is set to -1.
  if (rootIdx >= 0) {
    auto vec = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(ncclID),
        reinterpret_cast<uint8_t*>(ncclID) + NCCL_UNIQUE_ID_BYTES);
    store_->set(storeKeys[rootIdx], vec);
  }
  try {
    results = store_->multiGet(storeKeys);
  } catch (const std::exception& e) {
    nlohmann::json json_vec = storeKeys;
    std::string exceptionMsg = c10::str(
        "[",
        rank_,
        "] is setting up NCCL communicators and "
        "retrieving ncclUniqueId from roots via TCPStore by key '",
        json_vec.dump(),
        "', but got error: ");
    C10_THROW_ERROR(
        DistBackendError,
        exceptionMsg + e.what() +
            ". This may indicate a possible application crash on rank 0 or a network set up issue.");
  } catch (...) {
    nlohmann::json json_vec = storeKeys;
    C10_THROW_ERROR(
        DistBackendError,
        c10::str(
            "Unknown exception while [",
            rank_,
            "] is setting up NCCL communicators and "
            "retrieving ncclUniqueIds from roots via TCPStore by key '",
            json_vec.dump(),
            "'",
            ". This may indicate a possible application crash on rank 0 or a network set up issue."));
  }

  for (size_t r = 0; r < ncclIDs.size(); r++) {
    TORCH_CHECK_WITH(
        DistBackendError,
        results[r].size() == NCCL_UNIQUE_ID_BYTES,
        "Invalid size for ncclUniqueId");
    std::memcpy(&ncclIDs[r], results[r].data(), results[r].size());
  }
}

void ProcessGroupNCCL::destroyNCCLComms(const std::string& devNCCLCommMapKey) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (devNCCLCommMap_.find(devNCCLCommMapKey) == devNCCLCommMap_.end()) {
    TORCH_INTERNAL_ASSERT(
        false,
        "Expected to find key ",
        devNCCLCommMapKey,
        " in NCCL communicator map.");
  }
  std::shared_ptr<NCCLComm>& ncclComm = devNCCLCommMap_[devNCCLCommMapKey];
  // ncclCommDestroy(comm->getNcclComm()) results in segfault when PG is being
  // destroyed, so using ncclCommAbort here.
  ncclComm->abort();
  // Remove communicators from the cache.
  devNCCLCommMap_.erase(devNCCLCommMapKey);
  // Clear used device indices.
  usedDeviceIdxs_.clear();

  {
    std::lock_guard<std::mutex> lock(ncclCommMemPoolMapMutex);
    ncclCommMemPoolMap.erase(ncclComm);
  }
}

std::shared_ptr<NCCLComm> ProcessGroupNCCL::initNCCLComm(
    const std::string& deviceKey,
    at::Device& device,
    OpType opType,
    int p2pRank,
    bool isSendRecvSelf) {
  // Sanity check
  if (deviceKey.empty()) {
    C10_THROW_ERROR(
        DistBackendError,
        "Not able to create/get the NCCL Communicator since "
        "the GPU devices are not known");
  }
  if (bound_device_id_) {
    if (*bound_device_id_ != device) {
      LOG(ERROR) << logPrefix() << "Tensor found on device " << device
                 << " but backend constrained to " << *bound_device_id_;
      C10_THROW_ERROR(
          DistBackendError,
          "Attempt to perform collective on tensor not on device passed to init_process_group");
    }
  }

  usedDeviceIdxs_.insert(device.index());

  // NCCL communicator not cached, create a new entry
  std::shared_ptr<NCCLComm> ncclComm;

  // Create the unique NCCL ID and broadcast it
  ncclUniqueId ncclID;

  // reset log prefix to include group_desc
  logPrefix_ = createLogPrefix();

#ifdef NCCL_COMM_DESCRIPTION
  // Pass process group name and description to NCCL communicator
  std::string commDesc = pg_desc_ + ':' + pg_uid_;
  options_->config.commDesc = strdup(commDesc.c_str());
#endif // NCCL_COMM_DESCRIPTION

  // For batch_isend_irecv, ncclGroupStart() would be called upfront
  bool batchP2P = ncclActiveGroupCounter_ > 0;
  bool singleP2POp = isP2POp(opType, batchP2P);

  // Get the device index
  auto deviceIndex = device.index();
  at::cuda::OptionalCUDAGuard gpuGuard(device);

  // [Group Start/End Note] This is used to ensure that nccl communicator will
  // be created before communication primitives are called. Let's look at this
  // example: Using the batch_isend_irecv to send a tensor to a target process.
  // On the sender side, the corresponding underlying NCCL calls will look like
  //   ncclGroupStart() // This is in batch_isend_irecv
  //   ncclCommInitRank() // Inside NCCLComm::create
  //   ncclSend()
  //   ncclGroupEnd() // This is in batch_isend_irecv
  // With this pattern, the nccl communicator will be created in the last
  // ncclGroupEnd which means when ncclSend is processed, the passed
  // communicator argument is NULL which will lead to runtime error. So we need
  // to "close" all active nccl groups to ensure nccl communicator is actually
  // created before encountering any communication calls. This is why we need
  // the following for loop.
  for (const auto i : c10::irange(ncclActiveGroupCounter_)) {
    (void)i;
    // comms have not been initiated yet, so can only check in blocking-way
    C10D_NCCL_CHECK(ncclGroupEnd(), std::nullopt);
  }

  // GPU world size and GPU rank
  int numRanks = -1, rank = -1;

  if (!singleP2POp) {
    // Collective, all-to-all, or batch P2P
    numRanks = getSize();
    rank = getRank();
  } else if (isSendRecvSelf) {
    // Same process send and recv.
    numRanks = 1;
    rank = 0;
  } else {
    // For single point-to-point operation, there are only 2 processes
    // involved so the GPU rank is either 0 or 1.
    numRanks = 2;
    rank = p2pRank;
  }

  RECORD_PARAM_COMMS(
      std::make_tuple(0, false), // seq
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      rank, // rank
      "init", // collective name
      0, // inNelems
      0, // outNelems
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart_, // globalRankStart_
      globalRankStride_, // globalRankStride_
      size_); // worldSize

#ifdef NCCL_HAS_COMM_NONBLOCKING
  bool useNb = useNonblocking();
  options_->config.blocking = useNb ? 0 : 1;
#endif // NCCL_HAS_COMM_NONBLOCKING

#ifdef NCCL_HAS_COMM_SPLIT
  // Use split to create a new communicator only if:
  // 1. The parent comm is known; AND
  // 2. The new comm is not for a point-to-point operation.
  // ncclCommSplit() is a collective call, so it does not work for P2P
  // operations.
  if (options_->split_from && !singleP2POp) {
    // Find a valid, healthy communicator to split from if possible.
    std::lock_guard<std::mutex> lock(options_->split_from->mutex_);
    auto& other_comms = options_->split_from->devNCCLCommMap_;
    auto dit = other_comms.find(getKeyFromDevice(device));
    if (dit != other_comms.end()) {
      auto& parentComm = dit->second;
      if (parentComm != nullptr && !parentComm->isAborted()) {
        LOG(INFO) << logPrefix() << "Splitting NCCL communicator from "
                  << parentComm->repr();
        ncclComm = NCCLComm::split(
            parentComm.get(), options_->split_color, rank, options_->config);
      }
    }
  }
#endif // NCCL_HAS_COMM_SPLIT

  bool useScalableInit = false;
  // (nranks / nroots) == 128 was the default NCCL recommended
  // according to
  // https://github.com/pytorch/pytorch/pull/136789#discussion_r1779171615.
  auto ranksPerRoot = getCvarInt(TORCH_NCCL_RANKS_PER_ROOT, 128);
#if defined(NCCL_HAS_INIT_RANK_SCALABLE) && defined(NCCL_HAS_CONFIG)
  useScalableInit = !singleP2POp && (getSize() > ranksPerRoot);
#endif // NCCL_HAS_INIT_RANK_SCALABLE && NCCL_HAS_CONFIG

  if (useScalableInit) {
    auto numRoots = (getSize() + ranksPerRoot - 1) / ranksPerRoot;
    std::vector<ncclUniqueId> ncclIDs(numRoots);

    if (!ncclComm) {
      auto rootIdx = getRootIndex(rank_, getSize(), numRoots);
      // We only need to get unique IDs for roots. For non-root rank, index is
      // set to -1.
      if (rootIdx >= 0) {
        C10D_NCCL_CHECK(ncclGetUniqueId(&ncclID), std::nullopt);
      }
      // We only need to all-gather the ncclID if the rank is root.
      auto timeStarted = std::chrono::steady_clock::now();
      allgatherUniqueNCCLIDs(rootIdx, &ncclID, ncclIDs);
      auto timerDeltaMs =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              std::chrono::steady_clock::now() - timeStarted)
              .count() *
          1000;
      LOG(INFO) << logPrefix()
                << "ProcessGroupNCCL all-gather unique IDs through store took "
                << timerDeltaMs << " ms";
#if defined(NCCL_HAS_INIT_RANK_SCALABLE) && defined(NCCL_HAS_CONFIG)
      ncclComm = NCCLComm::create_scalable(
          numRanks, rank, ncclIDs, deviceIndex, options_->config);
#else
      C10_THROW_ERROR(
          DistBackendError,
          c10::str(
              logPrefix(),
              "create_scalable is called when useScalableInit is enabled but ",
              "neither NCCL_HAS_INIT_RANK_SCALABLE nor NCCL_HAS_CONFIG is not defined, this should not happen "));
#endif // NCCL_HAS_INIT_RANK_SCALABLE
    }
  } else {
    // To simplify conditional nesting, just create the ncclComms[i]
    // entry if it hasn't been yet rather than untangling the
    // conditions that might have resulted in a split above.
    if (!ncclComm) {
      if (getCvarBool(TORCH_NCCL_BCAST_UNIQUEID, true) && !isSendRecvSelf) {
        // For point-to-point communication, lower rank of the two will get
        // unique id.
        if (rank_ == 0 || (singleP2POp && p2pRank == 0)) {
          C10D_NCCL_CHECK(ncclGetUniqueId(&ncclID), std::nullopt);
        }

        // Broadcast so that each process can have a unique NCCL ID
        auto timeStarted = std::chrono::steady_clock::now();
        broadcastUniqueNCCLID(&ncclID, singleP2POp, deviceKey, p2pRank);
        auto timerDeltaMs =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::steady_clock::now() - timeStarted)
                .count() *
            1000;
        LOG(INFO) << logPrefix()
                  << "ProcessGroupNCCL broadcast unique ID through store took "
                  << timerDeltaMs << " ms";
      }

#ifdef NCCL_HAS_CONFIG
      ncclComm = NCCLComm::create(
          numRanks, rank, ncclID, deviceIndex, options_->config);
#else
      ncclComm = NCCLComm::create(numRanks, rank, ncclID, deviceIndex);
#endif // NCCL_HAS_CONFIG
    }
  }

  // Creates the NCCL streams
  bool force_high = getCvarBool(TORCH_NCCL_HIGH_PRIORITY, false);
  auto streamVal = at::cuda::getStreamFromPool(
      options_->is_high_priority_stream || force_high);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    inInitializationCommMap_.emplace(deviceKey, ncclComm);
  }

  FlightRecorderCUDA::get()->record_pg_ranks(
      std::make_tuple(pg_uid_, pg_desc_), groupRanks());
  FlightRecorderCUDA::get()->record_accelerator_version(getNcclVersion());

  VLOG(2) << logPrefix() << "ProcessGroupNCCL created ncclComm_ "
          << ncclComm->repr()
          << " on CUDA device: " << static_cast<int>(deviceIndex);

  // At this point NCCL should have been initialized, hence we can accurately
  // get the env value even if NCCL sets it by reading from nccl.conf file
  LOG(INFO) << logPrefix()
            << "NCCL_DEBUG: " << getCvarString({"NCCL_DEBUG"}, "N/A");

  // See [Group Start/End Note]
  for (const auto i : c10::irange(ncclActiveGroupCounter_)) {
    (void)i;
    C10D_NCCL_CHECK(ncclGroupStart(), std::nullopt);
  }

  ncclStreams_.emplace(deviceKey, streamVal);

  // Note: these events are created with the (default) cudaEventDisableTiming
  // flag This flag provides the best performance when used with
  // cudaStreamWaitEvent() and cudaEventQuery(). Since we here don't measure the
  // performance using cudaEvent, this should be set.
  // TODO(kwen2501): is ncclEvents_ used anywhere else?
  ncclEvents_.emplace(deviceKey, at::cuda::CUDAEvent(cudaEventDisableTiming));

  // Move the NCCL resource to cache
  auto it = inInitializationCommMap_.find(deviceKey);
  // A previous thread could've already removed devicesKey from
  // inInitializationCommMap_ and added it to devNCCLCommMap_
  if (it != inInitializationCommMap_.end()) {
    devNCCLCommMap_.emplace(deviceKey, std::move(it->second));
    inInitializationCommMap_.erase(deviceKey);

    // Now ncclComms are fully initialized.
    // Register all active CUDA memory segments in cache allocator to
    // the new NCCL communicators
    if (shouldAllCommunicatorsRegisterAllTensors()) {
      auto snapshot = c10::cuda::CUDACachingAllocator::snapshot();
      // Register the segment to a new NCCL communicator if on the same device
      for (const auto& segmentInfo : snapshot.segments) {
        TORCH_INTERNAL_ASSERT(
            segmentInfo.device == device.index(),
            "Mismatch between CUDA memory segment device and current device");
        ncclComm->registerSegment(
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            reinterpret_cast<void*>(segmentInfo.address),
            segmentInfo.total_size);
      }
    }
    // Record the mapping between ncclComm and device index so that later
    // register hook can register a newly allocated segment to communicators
    // on the same device.
    // NOTE: we need remove the communicator from this map when it is
    // destroyed, otherwise may register onto an invalid communicator.
    {
      std::lock_guard<std::mutex> lock(ncclCommMemPoolMapMutex);
      ncclCommMemPoolMap.emplace(ncclComm, MemPoolSet{});
    }
  }

  it = devNCCLCommMap_.find(deviceKey);
  TORCH_INTERNAL_ASSERT(
      it != devNCCLCommMap_.end(), "Communicators not populated in cache!");
  return it->second;
}

int64_t ProcessGroupNCCL::getCommPtr() {
  // Get the collective communicator on the current CUDA device.
  auto device = at::Device(at::kCUDA, at::cuda::current_device());
  std::string deviceKey = getKeyFromDevice(device);
  auto ncclComm = getNCCLComm(deviceKey);

  // ncclComm is a nullptr if the communicator does not exist.
  ncclComm_t comm = nullptr;
  if (ncclComm != nullptr) {
    comm = ncclComm->getNcclComm();
  }
  const int64_t commPtr = reinterpret_cast<int64_t>(comm);
  return commPtr;
}

std::shared_ptr<NCCLComm> ProcessGroupNCCL::getNCCLComm(
    const std::string& deviceKey) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (devNCCLCommMap_.find(deviceKey) != devNCCLCommMap_.end()) {
    // Reuse the cached communicator if there is one.
    return devNCCLCommMap_[deviceKey];
  }
  return nullptr;
}

uint64_t ProcessGroupNCCL::getCommSplitCounter() const {
  uint64_t ret = 0;
  for (const auto& i : devNCCLCommMap_) {
    auto& ncclComm = i.second;
    ret += ncclComm->getCommSplitCounter();
  }
  return ret;
}

namespace {

// Check validity of tensor
void check_gpu_single_tensor(
    const at::Tensor& tensor,
    const bool p2p = false // whether operation is a P2P operation
) {
  if (!tensor.is_cuda() || tensor.is_sparse()) {
    C10_THROW_ERROR(ValueError, "Tensors must be CUDA and dense");
  }
  // Check memory format
  if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
    // P2P is a bit relaxed, supporting transfer of a transposed tensor
    if (p2p) {
      // But must be dense still
      if (!tensor.is_non_overlapping_and_dense()) {
        C10_THROW_ERROR(
            ValueError, "Tensors for P2P must be non-overlapping and dense");
      }
      TORCH_WARN_ONCE(
          "Detected non-contiguous tensor in P2P operations. It is user "
          "responsibility to guarantee that source and destination tensors have "
          "the same contiguity format.");
    } else {
      C10_THROW_ERROR(ValueError, "Tensors must be contiguous");
    }
  }
}

// Checks that all `tensors' have the same type and shape and reside on the same
// GPU.
// TODO: test_c10d_nccl.py should consider adding tests for the error conditions
// here, ie, that deliberately pass invalid tensors and check the right
// exception is thrown. The "Expected list of tensors on the same device"
// condition may be a challenge because the test would need to pass tensors on
// different devices in the same process.
int64_t check_gpu_tensors_same_device(const std::vector<at::Tensor>& tensors) {
  if (tensors.empty()) {
    C10_THROW_ERROR(ValueError, "Tensor list must be nonempty");
  }

  const auto& first = tensors.front();

  int64_t total_numel = 0;
  for (const auto& t : tensors) {
    if (!t.is_cuda() || t.is_sparse()) {
      C10_THROW_ERROR(ValueError, "Tensors must be CUDA and dense");
    }
    if (t.scalar_type() != first.scalar_type()) {
      C10_THROW_ERROR(TypeError, "Tensors must have identical type");
    }
    if (!t.is_non_overlapping_and_dense()) {
      C10_THROW_ERROR(ValueError, "Tensors must be non-overlapping and dense");
    }
    // If we're in this function, the user called a _coalesced collective
    // on a set of tensors with potentially different sizes and strides.
    // Therefore, we don't check for matching sizes and strides,
    // but we do double-check tensors are on the same device.
    TORCH_CHECK_WITH(
        ValueError,
        t.get_device() == tensors[0].get_device(),
        "Expected list of tensors on the same device");
    total_numel += t.numel();
  }

  return total_numel;
}

bool check_same_size(const std::vector<at::Tensor>& input_tensors) {
  for (const auto& input_tensor : input_tensors) {
    if (!input_tensors[0].is_same_size(input_tensor)) {
      return false;
    }
  }
  return true;
}

} // namespace

c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL> ProcessGroupNCCL::initWork(
    at::Device& device,
    int rank,
    OpType opType,
    bool isP2P,
    const char* profilingTitle,
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& outputs, // TODO(kwen2501): necessary?
    bool record) {
  auto r = c10::make_intrusive<ProcessGroupNCCL::WorkNCCL>(
      pg_uid_,
      pg_desc_,
      device,
      rank,
      opType,
      isP2P ? seqP2P_ : seqCollective_,
      isP2P,
      profilingTitle,
      profilingTitle != nullptr ? std::optional<std::vector<at::Tensor>>(inputs)
                                : std::nullopt,
      enableTiming_.load(),
      cudaEventCacheEnabled_.load(),
      dist_debug_level_);

  if (record) {
    bool isP2P = isP2POp(opType);
    // Ideally record every work that we enqueue, rather than every work we
    // create.
    // - at the time of this PR we do not currently enqueue every created work
    // - but it is unsafe to steal refs to start/end cuda events from Works that
    //   may go out of scope before flight recorder has retired them,
    //   so we must ensure that any work that is initialized via initWork will
    //   be enqueued
    // - initially, moved record() into workEnqueue(), but found that makes it
    //   hard to get access to profilingTitle,
    //   inputs, and outputs for metadata recording, and we don't want to attach
    //   these objects to the Work because it has implications for keeping those
    //   tensors alive longer and adds overhead when copying Work objects
    //   between threads
    auto traceId = FlightRecorderCUDA::get()->recordWithResetEnabled(
        local_id_,
        std::make_tuple(pg_uid_, pg_desc_),
        seqCollective_,
        seqP2P_,
        op_id_,
        profilingTitle ? profilingTitle : "",
        inputs,
        outputs,
        r->ncclStartEvent_.get(),
        r->ncclEndEvent_.get(),
        options_->timeout,
        pgStatus_,
        isP2P);
    r->trace_id_ = traceId.id;
    r->trace_reset_epoch_ = traceId.reset_epoch;
  }
  return r;
}

// TODO(kwen2501): deprecate
std::vector<at::Tensor> ProcessGroupNCCL::WorkNCCL::result() {
  return *outputs_;
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupNCCL::WorkNCCL::
    getFuture() {
  return future_;
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupNCCL::WorkNCCL::
    getFutureResult() {
  return futureWorkResult_;
}

float ProcessGroupNCCL::WorkNCCL::getDuration() const {
  TORCH_CHECK(timingEnabled_, "getDuration only works if timing was enabled");
  TORCH_CHECK(
      ncclStartEvent_,
      "getDuration only works if ncclStartEvents_ is populated, true if timing enabled");
  TORCH_CHECK(
      ncclEndEvent_,
      "getDuration only works if ncclEndEvents_ is populated, which should always be true");
  return ncclStartEvent_->elapsed_time(*ncclEndEvent_);
}

uint64_t ProcessGroupNCCL::WorkNCCL::getSequencenumber() const {
  return seq_;
}

void ProcessGroupNCCL::assignTimeoutToWork(
    const c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work,
    const c10::intrusive_ptr<ProcessGroupNCCL::Options>& option) {
  std::chrono::milliseconds timeout = option->timeout;
  std::lock_guard<std::mutex> timeoutLock(mtxTimeoutExtension_);
  if (ephemeralTimeoutActive_.count() > 0) {
    timeout += ephemeralTimeoutActive_;
  }
  work->opTimeout_ = timeout;
  work->ownedEphermeralTimeout_ =
      ephemeralTimeoutActive_ - ephemeralTimeoutInflight_;
  ephemeralTimeoutInflight_ = ephemeralTimeoutActive_;
}

void ProcessGroupNCCL::workEnqueue(
    const c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
  // We clean up the TensorShelf's in case user hasn't called `work.wait()`.
  // This has nothing to do with new work enqueue. We are just using a place
  // that would be triggered by a next user call.
  {
    std::lock_guard<std::mutex> lock(shelvesMutex_);
    for (auto& shelf : shelvesToUnstash_) {
      shelf->unstash();
    }
    shelvesToUnstash_.clear();
  }

  // in blockingWait_ mode, we don't need watchdog thread, so no need to enqueue
  // the work
  if (!terminateProcessGroup_.load() && !blockingWait_) {
    std::lock_guard<std::mutex> lock(workMetaListMutex_);
    // Avoid view tensors to be processed in cleanup thread.
    // View tensors' destruction invokes autograd_meta, which
    // needs to be destructed in user thread. Otherwise will
    // get deadlock. Here we enqueue work without outputs_.
    workMetaList_.emplace_back(*work);
    // update the PG status related to the last enqueued work
    pgStatus_->lastEnqueuedSeq = static_cast<int64_t>(work->seq_);
    pgStatus_->lastEnqueuedWorkName = opTypeToString(work->opType_);
    pgStatus_->lastEnqueuedNumelIn = work->numelIn_;
    pgStatus_->lastEnqueuedNumelOut = work->numelOut_;
    heartbeatMonitor_->setLastWorkListUpdateTime(
        std::chrono::steady_clock::now());
  }
}

ProcessGroupNCCL::Options::Options(bool is_high_priority_stream)
    : Backend::Options(NCCL_BACKEND_NAME, kProcessGroupNCCLDefaultTimeout),
      is_high_priority_stream(is_high_priority_stream) {}

static constexpr int CoalActive = 0x01, CoalColl = 0x02, CoalP2P = 0x04;

uint64_t ProcessGroupNCCL::getWatchdogHeartbt() const {
  return watchdog_->getHeartbt();
}

void ProcessGroupNCCL::startCoalescing() {
  // Other collective ops bump seq_ before creating a work. Thus, if coalesced
  // ops bump seq_ only after initing a work they will collide with (reuse) the
  // seq_ of the last non-coalesced collective.  Previously, seq_ was bumped
  // inside endCoalescing, but before initWork. Since we now record individual
  // ops from a coalesce group into the flight recorder, we want to have the
  // same seq_ for those ops and its 'endCoalescing' op. Hence we bump during
  // start, which has one minor downside- we burn a seq_ if someone ever does a
  // 'start' and 'end' coalescing region without doing an operation in between.

  coalescedDevice_.set_index(-1);
  coalescedComm_ = nullptr;
  coalescedTensors_.clear();
  coalescing_state_ |= CoalActive;
  groupStart();
}

// `optype` is for specifying a composite optype, such as ALLGATHER and
// REDUCE_SCATTER
c10::intrusive_ptr<Work> ProcessGroupNCCL::endCoalescing(OpType optype) {
  if (coalescedComm_ == nullptr) {
    // There is no actual work being coalesced, return here
    groupEnd();
    coalescing_state_ = 0;
    return nullptr;
  }
  TORCH_CHECK(
      coalescedDevice_.index() >= 0,
      "Something went wrong. Did you call end_coalescing before start_coalescing?");

  // `coalescedComm_` should have same set of comms across collectives
  auto comm = coalescedComm_;
  // `coalescedDevice_` should have same set of devices across collectives
  auto device = coalescedDevice_;

  // `getKeyFromDevice` is how we get keys for both collectives and batch P2P
  const auto key = getKeyFromDevice(device);
  auto ncclStream = ncclStreams_.at(key);
  auto opProfilerTitle = optype != OpType::COALESCED
      ? "nccl:" + opTypeToString(optype) + "_coalesced"
      : "nccl:coalesced";

  // Create Work object
  c10::cuda::CaptureStatus capture_status =
      c10::cuda::currentStreamCaptureStatusMayInitCtx();
  bool enqueue =
      (coalescing_state_) && capture_status == c10::cuda::CaptureStatus::None;
  auto work = initWork(
      device,
      rank_,
      optype,
      coalescing_state_ & CoalP2P,
      opProfilerTitle.c_str(),
      {},
      {},
      enqueue);
  work->ncclComm_ = comm;
  work->blockingWait_ = blockingWait_;
  work->store_ = store_;
  assignTimeoutToWork(work, options_);

  // Hand over references to tensors during coalescing to work's stash
  work->stashed_for_allocator_safety_->stash(coalescedTensors_);

  // Record start before ncclGroupEnd
  if (work->timingEnabled_) {
    work->ncclStartEvent_->record(ncclStream);
  }

  if (useNonblocking()) {
    groupEndNonblocking(comm);
  } else {
    groupEnd();
  }

  // Record end after ncclGroupEnd
  // TODO(eqy): is this still necessary if avoidRecordStreams_ is set?
  work->ncclEndEvent_->record(ncclStream);

  if (enqueue) {
    workEnqueue(work);
  }

  {
    c10::cuda::CUDAMultiStreamGuard streamGuard(ncclStream);
    std::vector<at::Device> devices{device};
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);

    // Add a callback that runs profiling end callbacks. wrapCallback() in CUDA
    // future blocks the stream this callback runs on the corresponding
    // ncclEndEvents_ ensuring appropriate synchronization.
    if (work->recordFunctionEndCallback_) {
      work->future_->addCallback(
          [work](at::ivalue::Future& /* unused */) {
            work->recordFunctionEndCallback_();
          },
          // uses_future = false allows us to skip synchronization in
          // ivalue::Future, but is only valid as long as the lambda doesn't use
          // the "Future" argument.
          /*uses_future=*/false);
    }
    // Mark the future as completed since coalesced operations complete
    // immediately
    work->future_->markCompleted(at::IValue(std::vector<at::Tensor>{}));
  }

  // Reset coalescing state
  coalescing_state_ = 0;
  coalescedComm_ = nullptr;
  coalescedTensors_.clear();
  // If in async mode, return work; otherwise, kernel is enqueued on current
  // stream, no need to return work
  return coalescedAsync_ ? work : nullptr;
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::endCoalescing() {
  // Default OpType to COALESCED if not specified
  return endCoalescing(OpType::COALESCED);
}

void ProcessGroupNCCL::startTimeEstimate() {
  groupStart();
}

float ProcessGroupNCCL::endTimeEstimate() {
#ifdef NCCL_SIM_INFO_INITIALIZER
  ncclSimInfo_t simInfo = NCCL_SIM_INFO_INITIALIZER;
  C10D_NCCL_CHECK(ncclGroupSimulateEnd(&simInfo), std::nullopt);
  --ncclActiveGroupCounter_;
  return simInfo.estimatedTime;
#else
  TORCH_CHECK(
      false,
      c10::str(
          "The current nccl version does not support nccl comm time estimation. "));
#endif
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupNCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    OpType opType,
    bool asyncOp,
    const char* profilingTitle,
    bool nanCheck) {
  // Environment setting by the user may add onto collective call's option
  nanCheck &= enableNanCheck_;

  auto device = getDevice(inputs[0]);
  // Guard must be created before `currentStreamCaptureStatusMayInitCtx`;
  // otherwise, extra CUDA context could be created on device 0.
  at::cuda::OptionalCUDAGuard gpuGuard(device);

  c10::cuda::CaptureStatus capture_status =
      c10::cuda::currentStreamCaptureStatusMayInitCtx();
  errorIfCapturingNonCapturableNCCL(capture_status);

  // Bump collective counter
  if (!coalescing_state_) {
    seqCollective_++;
  }
  op_id_++;

  const auto key = getKeyFromDevice(device);
  std::shared_ptr<NCCLComm> ncclComm = getNCCLComm(key);
  if (ncclComm == nullptr) {
    ncclComm = initNCCLComm(key, device, opType);
  }

  if (coalescing_state_ & CoalActive) {
    if ((coalescing_state_ & CoalColl) == 0) {
      // First op in coalesced operations
      seqCollective_++;
    }
    coalescing_state_ |= CoalColl;
    if (coalescedDevice_.index() < 0) {
      coalescedDevice_ = device;
    } else {
      TORCH_CHECK(
          coalescedDevice_.index() == device.index(), MULTI_DEVICE_ERROR_MSG);
    }
    if (coalescedComm_ == nullptr) {
      coalescedComm_ = ncclComm;
    } else {
      TORCH_CHECK(coalescedComm_ == ncclComm, MULTI_DEVICE_ERROR_MSG);
    }
    coalescedAsync_ = asyncOp;
  }

  // in asyncOp=false [default] mode, we use currentStream as ncclStream
  // otherwise, we use separate ncclStream and let it sync on currentStream
  auto ncclStream = asyncOp ? ncclStreams_.at(key)
                            : at::cuda::getCurrentCUDAStream(device.index());
  if (asyncOp) {
    // First let NCCL streams wait for input tensors allocation streams
    syncStream(device, ncclEvents_[key], ncclStream);
  }

  bool enqueue =
      !coalescing_state_ && capture_status == c10::cuda::CaptureStatus::None;
  auto work = initWork(
      device, rank_, opType, false, profilingTitle, inputs, outputs, enqueue);
  if (coalescing_state_) {
    // When coalescing, we record events per op that lack timing/state
    // information because there is no 'work' associated with them, and then
    // later in endCoalescing we record a 'coalesced' Work which has
    // timing/state updates via watchdog thread, but lacks op metadata such as
    // input/output sizes and profilingTitle per-op in the group.
    FlightRecorderCUDA::get()->recordWithResetEnabled(
        local_id_,
        std::make_tuple(pg_uid_, pg_desc_),
        seqCollective_,
        seqP2P_,
        op_id_,
        profilingTitle,
        inputs,
        outputs,
        nullptr,
        nullptr,
        options_->timeout,
        pgStatus_,
        /*isP2P=*/false);
  }

  // Store references to outputs to be used by WorkNCCL::result and operator<<.
  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);

  // If we are performing sync operations, i.e. equeuing kernel onto "current"
  // stream, we don't need to do anything for tensor lifetime management.
  // Otherwise, we need to stage the tensors will `work.wait()`.
  if (asyncOp) {
    // First select which shelf to stash onto: to `work` if single collective;
    // to an inflight shelf if coalescing.
    if (coalescing_state_) {
      coalescedTensors_.stash(inputs);
      coalescedTensors_.stash(outputs);
    } else {
      work->stashed_for_allocator_safety_->stash(inputs);
      work->stashed_for_allocator_safety_->stash(outputs);
    }
  }

  if (nanCheck) {
    for (const auto& input : inputs) {
      checkForNan(input, ncclStream);
    }
  }

  // Start event should only be recorded before the ncclGroupStart()
  if (work->timingEnabled_ && !coalescing_state_) {
    work->ncclStartEvent_->record(ncclStream);
  }

  pre(ncclStream, work);

  ncclComm_t comm = ncclComm->getNcclComm();

  // Both `inputs' and `outputs' are created on a worker stream and used in
  // different ncclStreams.  Hence, both must record the ncclStream to
  // prevent being freed before the collective finishes.
  //
  // We only record `inputs' here, and leave recording `outputs' to `fn' for
  // operations where `inputs' and `outputs' are not the same.
  //
  // See [Sync Streams].

// Not all collectives have the same signature, e.g, all-reduce take in a Tensor
// as the input and output while all-to-all take in a vector of Tensors as input
// and output. Because we define the signature of the fn to take only single
// tensor as input and output, we need to do a hack to get the first element in
// the vector and pass it to fn.
// TODO: we should clean up this in future (by either entirely removing lambda's
// or removing input and output from lambda's signature).
#ifndef NCCL_HAS_COMM_NONBLOCKING
  C10D_NCCL_CHECK(
      fn(inputs[0], outputs[0], comm, ncclStream),
      ncclComm->getNcclCommFailureReason());
#else
  C10D_NCCL_CHECK_TIMEOUT(
      fn(inputs[0], outputs[0], comm, ncclStream),
      comm,
      ncclComm->getNcclCommFailureReason());
#endif // NCCL_HAS_COMM_NONBLOCKING

  post(ncclStream, work);

  // End event should only be recorded after the ncclGroupEnd()
  if (!coalescing_state_) {
    work->ncclEndEvent_->record(ncclStream);
  }
  work->ncclComm_ = ncclComm;

  {
    c10::cuda::CUDAMultiStreamGuard streamGuard(ncclStream);
    std::vector<at::Device> devices{device};
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);

    // Add a callback that runs profiling end callbacks. wrapCallback() in CUDA
    // future blocks the stream this callback runs on the corresponding
    // ncclEndEvents_ ensuring appropriate synchronization.
    if (work->recordFunctionEndCallback_) {
      work->future_->addCallback(
          [work](at::ivalue::Future& /* unused */) {
            work->recordFunctionEndCallback_();
          },
          // uses_future = false allows us to skip synchronization in
          // ivalue::Future, but is only valid as long as the lambda doesn't use
          // the "Future" argument.
          /*uses_future=*/false);
    }
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }

  // Set appropriate work parameters.
  work->blockingWait_ = blockingWait_;
  work->store_ = store_;
  assignTimeoutToWork(work, options_);
  // Record size info for debug. We only record the size on the first device as
  // multi-device per process is deprecated
  work->numelIn_ = 0;
  work->numelOut_ = 0;
  for (const auto& input : inputs) {
    work->numelIn_ += input.numel();
  }
  for (const auto& output : outputs) {
    work->numelOut_ += output.numel();
  }

  if (enqueue) {
    workEnqueue(work);
  }

  return asyncOp ? work : nullptr;
}

template <typename Fn>
c10::intrusive_ptr<Work> ProcessGroupNCCL::collectiveCoalesced(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    OpType opType,
    bool asyncOp,
    const char* profilingTitle) {
  // Currently, the API permits one scenario where inputs.size() and
  // outputs.size() are > 0.
  // 1. If the call was a _coalesced call, all inputs must be on the same
  // device.
  //    The group of nccl calls applies the collective separately to each input,
  //    but the group as a whole should be efficient, and might even execute as
  //    a single fused kernel.
  auto device = getDevice(inputs[0]);
  // Guard must be created before `currentStreamCaptureStatusMayInitCtx`;
  // otherwise, extra CUDA context could be created on device 0.
  at::cuda::OptionalCUDAGuard gpuGuard(device);

  c10::cuda::CaptureStatus capture_status =
      c10::cuda::currentStreamCaptureStatusMayInitCtx();
  errorIfCapturingNonCapturableNCCL(capture_status);

  // Bump collective counter
  seqCollective_++;

  // For coalescingManager collectives, there is no individual c++ call per
  // collective so there is no flight record and we increment seqCollective_ and
  // op_id_ together. Compare this to startCoalescing/endCoalescing flow where
  // we increment either seqP2P_ or seqCollective_ once per group and increment
  // op_id_ once per individual operation within the group
  op_id_++;

  const auto key = getKeyFromDevice(device);
  std::shared_ptr<NCCLComm> ncclComm = getNCCLComm(key);
  if (ncclComm == nullptr) {
    ncclComm = initNCCLComm(key, device, opType);
  }

  if (coalescing_state_ & CoalActive) {
    coalescing_state_ |= CoalColl;
    if (coalescedDevice_.index() < 0) {
      coalescedDevice_ = device;
    } else {
      TORCH_CHECK(
          coalescedDevice_.index() == device.index(), MULTI_DEVICE_ERROR_MSG);
    }
    if (coalescedComm_ == nullptr) {
      coalescedComm_ = ncclComm;
    } else {
      TORCH_CHECK(coalescedComm_ == ncclComm, MULTI_DEVICE_ERROR_MSG);
    }
    coalescedAsync_ = asyncOp;
  }

  // in asyncOp=false [default] mode, we use currentStream as ncclStream
  // otherwise, we use separate ncclStream and let it sync on currentStream
  auto ncclStream = asyncOp ? ncclStreams_.at(key)
                            : at::cuda::getCurrentCUDAStream(device.index());
  if (asyncOp) {
    // First let NCCL streams wait for input tensors allocation streams
    syncStream(device, ncclEvents_[key], ncclStream);
  }

  auto work = initWork(
      device,
      rank_,
      opType,
      false,
      profilingTitle,
      inputs,
      outputs,
      /*record=*/true);

  // Store references to outputs to be used by WorkNCCL::result and operator<<.
  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);

  // If we are performing sync operations, i.e. equeuing kernel onto "current"
  // stream, we don't need to do anything for tensor lifetime management.
  // Otherwise, we need to stage the tensors will `work.wait()`.
  if (asyncOp) {
    work->stashed_for_allocator_safety_->stash(inputs);
    work->stashed_for_allocator_safety_->stash(outputs);
  }

  // Start event should only be recorded before the ncclGroupStart() (which
  // happens inside AutoNcclGroup guard below)
  if (work->timingEnabled_) {
    work->ncclStartEvent_->record(ncclStream);
  }

  ncclComm_t comm = ncclComm->getNcclComm();

// TODO(kwen2501): this should be moved to c10d tests, to qualify a NCCL
// upgrade. Once a NCCL version is qualified, this code should not be needed at
// runtime.
#ifdef PGNCCL_ENABLE_HASH
  if (enableCollectiveHashDebug_.load()) {
    auto numel = getTensorsNumel(inputs);
    auto hashValue = hashTensors(inputs);
    PRINT_COLLECTIVE_HASH_SIGNATURE(
        "input", opTypeToString(opType), numel, hashValue);
  }
#endif // PGNCCL_ENABLE_HASH

  {
    torch::cuda::nccl::AutoNcclGroup nccl_group_guard(comm, useNonblocking());
    for (const auto i : c10::irange(inputs.size())) {
#ifndef NCCL_HAS_COMM_NONBLOCKING
      C10D_NCCL_CHECK(
          fn(inputs[i], outputs[i], comm, ncclStream),
          ncclComm->getNcclCommFailureReason());
#else
      C10D_NCCL_CHECK_TIMEOUT(
          fn(inputs[i], outputs[i], comm, ncclStream),
          comm,
          ncclComm->getNcclCommFailureReason());
#endif // NCCL_HAS_COMM_NONBLOCKING
    }
  }

  work->ncclEndEvent_->record(ncclStream);
  work->ncclComm_ = ncclComm;

  {
    c10::cuda::CUDAMultiStreamGuard streamGuard(ncclStream);
    std::vector<at::Device> devices{device};
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);

    // Add a callback that runs profiling end callbacks. wrapCallback() in CUDA
    // future blocks the stream this callback runs on the corresponding
    // ncclEndEvents_ ensuring appropriate synchronization.
    if (work->recordFunctionEndCallback_) {
      work->future_->addCallback(
          [work](at::ivalue::Future& /* unused */) {
            work->recordFunctionEndCallback_();
          },
          // uses_future = false allows us to skip synchronization in
          // ivalue::Future, but is only valid as long as the lambda doesn't use
          // the "Future" argument.
          /*uses_future=*/false);
    }
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }

  // Set appropriate work parameters.
  work->blockingWait_ = blockingWait_;
  work->store_ = store_;
  assignTimeoutToWork(work, options_);
  // Record size info for debug. We only record the size on the first device as
  // multi-device per process is deprecated
  work->numelIn_ = inputs[0].numel();
  work->numelOut_ = outputs[0].numel();

  /* Note [cuda graph capture and workEnqueue]

  Normal behavior of the C10D watchdog is to query cuda events on work objects.
  We disable this event query behavior during graph capture as it is disallowed
  during capture under the strictest capture mode setting.
  Note that previously recorded events (e.g., before the capture) can be queried
  as the watchdog capture mode has been changed to thread-local, but user-side
  event queries (from the main thread) via .is_completed() are still disallowed.
  TODO(eqy): Is there a path to allowing workEnqueue during graph capture for
  watchdog-thread usage only?

  TODO:
   - Is our design for flight recorder safe in this context?  are we recording
  any FR events during cudagraph capture? if so, they won't be safe to poll for
  completion status.
  */
  if (capture_status == c10::cuda::CaptureStatus::None) {
    workEnqueue(work);
  }
  // TODO(whc) if the work isn't enqueued, I don't feel great about returning
  // it, since interactions with it by usercode won't behave normally - they
  // won't observe work completion, for instance.  Will this lead to silent
  // problems during capture?
  return asyncOp ? work : nullptr;
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupNCCL::pointToPoint(
    at::Tensor& tensor,
    Fn fn,
    int peer,
    OpType opType,
    PreProcess pre,
    PostProcess post,
    const char* profilingTitle) {
  // avoidRecordStreams_ note:
  // send, recv, and irecv should be ok with avoidRecordStreams,
  // However, for isend, I don't think the API requires the user
  // to wait() on the returned handle, so ProcessGroupNCCL can't know
  // when it's safe to release the input back to the allocator,
  // and the present call has no way to know it's not an isend.
  // Therefore, we warn and fall back to the typical recordStream logic.
  // TODO( kwen2501 ): revisit this when we have a better solution.
  auto device = getDevice(tensor);
  at::cuda::OptionalCUDAGuard gpuGuard(device);

  std::string key;
  int p2pRank = -1, p2pTargetRank = -1;
  bool isSendRecvSelf = rank_ == peer;
  // For batch_isend_irecv, ncclGroupStart() would be called upfront
  bool batchP2P = ncclActiveGroupCounter_ > 0;

  std::shared_ptr<NCCLComm> ncclComm = nullptr;
  if (this->eagerInit_) {
    /* In eagerInit mode, reuse the parent comm.  Do not lazily create
     * p2p communicators. */
    if (!batchP2P && showSerializationWarning_) {
      TORCH_WARN_ONCE(c10::str(
          logPrefix(),
          "An unbatched P2P op (send/recv) was called on this ProcessGroup with size ",
          groupRanks().size(),
          ".  In eager initialization mode, unbatched P2P ops are treated as ",
          "independent collective ops, and are thus serialized with ",
          "all other ops on this ProcessGroup, including other P2P ",
          "ops. To avoid serialization, either create additional ",
          "independent ProcessGroups for the P2P ops or use batched ",
          "P2P ops. You can squash this warning by setting the environment variable ",
          "TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING to false."));
    }

    key = getKeyFromDevice(device);
    p2pRank = rank_;
    p2pTargetRank = peer;
    ncclComm = getNCCLComm(key);

    TORCH_INTERNAL_ASSERT(
        ncclComm != nullptr,
        "Parent communicator missing in eager initialization mode.");

    if (!coalescing_state_) {
      // Bump P2P sequence number. Don't do so if it's a batch P2P, it will be
      // bumped in `startCoalescing`.
      seqP2P_++;
    }
  } else if (batchP2P) {
    // TODO(whc) - unclear why we special-case batchP2P to avoid this path, but
    // I preserved this existing special case.
    key = getKeyFromDevice(device);
    p2pRank = rank_;
    p2pTargetRank = peer;
    ncclComm = getNCCLComm(key);
  } else {
    // We create special 2-rank communicators for each pair of
    // send/recv ranks.  This limitation exists for two reasons: (1)
    // we use a single stream per communicator, so if multiple
    // unbatched p2p operations are issued on the same communicator,
    // they would map to the same stream and thus would be serialized;
    // and (2) Nvidia NCCL does not allow multiple p2p operations to
    // be issued on the same communicator over different streams.

    TORCH_WARN_ONCE(
        "An unbatched P2P op (send/recv) was called on this ",
        "ProcessGroup with size ",
        groupRanks().size(),
        ".  In lazy initialization mode, this will result in a new 2-rank",
        " NCCL communicator to be created.");

    key = getKeySendRecv(rank_, peer);
    /* if we are creating a new comm, reset the p2pRank and
     * p2pTargetRank to correspond to this new 2-process communicator */
    p2pRank = rank_ <= peer ? 0 : 1;
    p2pTargetRank = isSendRecvSelf ? 0 : 1 - p2pRank;
    ncclComm = getNCCLComm(key);

    if (!coalescing_state_) {
      // Bump P2P sequence number.
      seqP2P_++;
    }
  }

  // Bump the logical operation counter regardless of whether this op is
  // coalesced or individual
  op_id_++;

  if (ncclComm == nullptr) {
    // ncclComm should never be a nullptr in eager init mode.
    // For lazy init mode, isSendRecvSelf is only valid for non-batch
    // point-to-point operations.  For batch operations, force the
    // argument to be false.
    ncclComm =
        initNCCLComm(key, device, opType, p2pRank, isSendRecvSelf && !batchP2P);
  }

  if (coalescing_state_ & CoalActive) {
    // Bump  seqP2P_ once per coalesced group, not once per individual op.
    if ((coalescing_state_ & CoalP2P) == 0) {
      seqP2P_++;
    }
    coalescing_state_ |= CoalP2P;
    if (coalescedDevice_.index() < 0) {
      coalescedDevice_ = device;
    } else {
      TORCH_CHECK(
          coalescedDevice_.index() == device.index(), MULTI_DEVICE_ERROR_MSG);
    }
    if (coalescedComm_ == nullptr) {
      coalescedComm_ = ncclComm;
    } else {
      TORCH_CHECK(coalescedComm_ == ncclComm, MULTI_DEVICE_ERROR_MSG);
    }
    // For now, P2P ops are always put on internal stream
    coalescedAsync_ = true;
  }

  // Used many times below, so we stash the unordered_map lookup
  auto ncclStream = ncclStreams_.at(key);
  // First let NCCL streams wait for input tensors allocation streams
  syncStream(device, ncclEvents_[key], ncclStream);

  // Work itself will create the CUDA events on all GPUs of tensors
  c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL> work;
  if (coalescing_state_) {
    // When coalescing, we record events per op that lack timing/state
    // information because there is no 'work' associated with them, and then
    // later in endCoalescing we record a 'coalesced' Work which has
    // timing/state updates via watchdog thread, but lacks op metadata such as
    // input/output sizes and profilingTitle per-op in the group.
    FlightRecorderCUDA::get()->record(
        local_id_,
        std::make_tuple(pg_uid_, pg_desc_),
        seqCollective_,
        seqP2P_,
        op_id_,
        profilingTitle,
        {tensor},
        {tensor},
        nullptr,
        nullptr,
        options_->timeout,
        pgStatus_,
        /*isP2P=*/true);
    // TODO(whc) if we want to make the per-p2p-op flightrecorder entries get
    // their timings/states updated by proxy when the Work obj representing the
    // coalesce group gets its update, we could accumulate these trace_ids
    // together and ask FlightRecorder to take the update from one Work and
    // apply it to multiple entries
  } else {
    // Store references to outputs to be used by WorkNCCL::result and
    // operator<<. Note that these outputs are only valid for recv(), as send()
    // does not modify the inputs but we still create these outputs for use
    // cases such as profiling.

    work = initWork(
        device,
        rank_,
        opType,
        true,
        profilingTitle,
        {tensor},
        {},
        /*record=*/false);
    // This bypasses something in Work() that crashes if {tensor} is given as
    // output, not sure what
    work->outputs_ = std::make_shared<std::vector<at::Tensor>>();
    work->outputs_->push_back(tensor);
    // TODO(whc) because we don't pass output {tensor} to initWork, we tell
    // initWork to not record, and then we manually call record passing all the
    // information it wants.
    auto traceId = FlightRecorderCUDA::get()->recordWithResetEnabled(
        local_id_,
        std::make_tuple(pg_uid_, pg_desc_),
        seqCollective_,
        seqP2P_,
        op_id_,
        profilingTitle,
        {tensor},
        {tensor},
        work->ncclStartEvent_.get(),
        work->ncclEndEvent_.get(),
        options_->timeout,
        pgStatus_,
        /*isP2P=*/true);
    work->trace_id_ = traceId.id;
    work->trace_reset_epoch_ = traceId.reset_epoch;
  }

  // Only check for NaN for send ops, for recv ops `tensor` can be a random
  // placeholder
  if (enableNanCheck_ && opType == OpType::SEND) {
    checkForNan(tensor, ncclStream);
  }

  if (!coalescing_state_) {
    // Start event should only be recorded before the ncclGroupStart()
    if (work->timingEnabled_) {
      work->ncclStartEvent_->record(ncclStream);
    }

    pre(ncclStream, work);
  }

  // Both send tensor and recv tensor are created on a worker stream and used
  // in different ncclStreams.  Hence, both must record the ncclStream to
  // prevent being freed before the collective finishes.
  //
  // See [Sync Streams].
  c10::cuda::CUDACachingAllocator::recordStream(
      tensor.storage().data_ptr(), ncclStream);

  // This part seems common to both p2p and coalesced-p2p usage?
  ncclComm_t comm_ = ncclComm->getNcclComm();

#ifndef NCCL_HAS_COMM_NONBLOCKING
  C10D_NCCL_CHECK(
      fn(tensor, comm_, ncclStream, p2pTargetRank),
      ncclComm->getNcclCommFailureReason());
#else
  // In non-blocking mode, we need to use ncclGroup semantics to ensure that the
  // kernel is enqueued for single-P2P ops.  Otherwise, the event record below
  // may not capture the kernel, leading to data corruption.
  ncclGroupStart();
  C10D_NCCL_CHECK_NONBLOCKING(
      fn(tensor, comm_, ncclStream, p2pTargetRank), std::nullopt);
  C10D_NCCL_CHECK_TIMEOUT_GROUPEND(
      ncclGroupEnd(), ncclComm, ncclComm->getNcclCommFailureReason());
#endif // NCCL_HAS_COMM_NONBLOCKING

  if (!coalescing_state_) {
    post(ncclStream);

    // End event should only be recorded after the ncclGroupEnd()
    work->ncclEndEvent_->record(ncclStream);
    work->ncclComm_ = ncclComm;
    work->blockingWait_ = blockingWait_;
    work->store_ = store_;
    assignTimeoutToWork(work, options_);
    // Record size info for debug. We only record the size on the first device
    // as multi-device per process is deprecated
    work->numelIn_ = work->numelOut_ = tensor.numel();

    // Future only needs to be created and marked completed with outputs for
    // recv(), but still create future for use cases such as profiling even for
    // send().
    {
      c10::cuda::CUDAMultiStreamGuard streamGuard(ncclStream);
      std::vector<at::Device> devices{device};
      work->future_ = c10::make_intrusive<at::ivalue::Future>(
          c10::ListType::create(c10::TensorType::get()), devices);
      work->future_->markCompleted(at::IValue(*work->outputs_));
    }

    // Add a callback that runs profiling end callbacks. wrapCallback() in CUDA
    // future blocks the stream this callback runs on the corresponding
    // ncclEndEvents_ ensuring appropriate synchronization.
    if (work->recordFunctionEndCallback_) {
      work->future_->addCallback(
          [work](at::ivalue::Future& /* unused */) {
            work->recordFunctionEndCallback_();
          },
          // uses_future = false allows us to skip synchronization in
          // ivalue::Future, but is only valid as long as the lambda doesn't use
          // the "Future" argument.
          /*uses_future=*/false);
    }
  }

  // Enqueue P2P op so that it can be cancelled by NCCL watchdog
  c10::cuda::CaptureStatus capture_status =
      c10::cuda::currentStreamCaptureStatusMayInitCtx();

  if (!coalescing_state_ && capture_status == c10::cuda::CaptureStatus::None) {
    workEnqueue(work);
  }
  return work;
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupNCCL::collective(
    at::Tensor& input,
    at::Tensor& output,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    OpType opType,
    bool asyncOp,
    const char* profilingTitle,
    bool nanCheck) {
  auto inputs = std::vector<at::Tensor>{input};
  auto outputs = std::vector<at::Tensor>{output};
  return collective(
      inputs,
      outputs,
      fn,
      pre,
      post,
      opType,
      asyncOp,
      profilingTitle,
      nanCheck);
}

template <typename Fn>
c10::intrusive_ptr<Work> ProcessGroupNCCL::collective(
    at::Tensor& input,
    at::Tensor& output,
    Fn fn,
    OpType opType,
    bool asyncOp,
    const char* profilingTitle,
    bool nanCheck) {
  auto inputs = std::vector<at::Tensor>{input};
  auto outputs = std::vector<at::Tensor>{output};
  return collective(
      inputs,
      outputs,
      fn,
      [](at::cuda::CUDAStream&,
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
      [](at::cuda::CUDAStream&,
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
      opType,
      asyncOp,
      profilingTitle,
      nanCheck);
}

template <typename Fn>
c10::intrusive_ptr<Work> ProcessGroupNCCL::pointToPoint(
    at::Tensor& tensor,
    Fn fn,
    int peer,
    OpType opType,
    const char* profilingTitle) {
  return pointToPoint(
      tensor,
      fn,
      peer,
      opType,
      [](at::cuda::CUDAStream&,
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
      [](at::cuda::CUDAStream&) {},
      profilingTitle);
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allreduce_sparse(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  TORCH_CHECK(
      !isUnsupportedFloat8(tensor.scalar_type()),
      "Unsupported Float8 type for NCCL reduction");
#ifdef IS_NCCLX
  tensor = tensor.coalesce();
  at::Tensor outputTensor =
      torch::zeros(tensor.sizes(), tensor.options().layout(torch::kStrided));
  auto work = collective(
      tensor,
      outputTensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        auto ncclDataType = getNcclDataType(input.scalar_type());
        auto ncclReduceOp =
            getNcclReduceOp(opts.reduceOp, input, ncclDataType, comm);
        auto indices = input.indices();
        auto sizes = input.sizes();
        int colSize = sizes[1];
        auto rows = indices[0];
        size_t blockCount = rows.sizes()[0];
        auto recvIndices = indices[0] * colSize;

        // prevent output and recvIndices from being freed
        // TODO: not changing the lifetime management of outputs this time,
        // revisit later
        c10::cuda::CUDACachingAllocator::recordStream(
            output.storage().data_ptr(), stream);
        c10::cuda::CUDACachingAllocator::recordStream(
            recvIndices.storage().data_ptr(), stream);
        auto result = ncclAllReduceSparseBlock(
            input._values().data_ptr(), // sendbuff
            recvIndices.data_ptr<int64_t>(), // recv_indices
            blockCount, // block_count
            colSize, // block_length
            output.data_ptr(), // recvbuff
            output.numel(), // recv_count
            ncclDataType,
            ncclReduceOp,
            comm,
            stream.stream());
        return result;
      },
      [](at::cuda::CUDAStream& ncclStream,
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
      [&](at::cuda::CUDAStream& ncclStream,
          c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
        // Convert output tensors to sparse and back into tensors.
        at::cuda::CUDAStreamGuard guard(ncclStream);
        if (opts.sparseIndices.has_value()) {
          tensor = at::sparse_coo_tensor(
              opts.sparseIndices.value(), outputTensor, tensor.sizes());
        } else {
          tensor = outputTensor.to_sparse();
        }
      },
      OpType::_ALLREDUCE_SPARSE,
      opts.asyncOp,
      "nccl:all_reduce_sparse");
  return work;
#else
  // If the nccl branch is not "exp" then we just error
  C10_THROW_ERROR(
      Error,
      "NCCL does not support all_reduce with sparse tensors. Please use dense tensors instead.");
#endif // IS_NCCLX
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allreduce_impl(
    at::Tensor& tensor,
    const char* profilingTitle,
    const AllreduceOptions& opts) {
  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        auto ncclDataType = getNcclDataType(input.scalar_type());
        auto ncclReduceOp =
            getNcclReduceOp(opts.reduceOp, input, ncclDataType, comm);
        return ncclAllReduce(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            ncclDataType,
            ncclReduceOp,
            comm,
            stream.stream());
      },
      OpType::ALLREDUCE,
      opts.asyncOp,
      profilingTitle);
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  if (tensor.is_complex()) {
    TORCH_CHECK(
        c10d::isComplexViewAsRealAllowed(opts.reduceOp),
        "all_reduce does not support",
        opts.reduceOp,
        "on complex tensors");
    tensor = at::view_as_real(tensor);
  }
  check_gpu_single_tensor(tensor);

  if (intraNodeComm_ != nullptr && opts.reduceOp == ReduceOp::SUM) {
    using namespace intra_node_comm;
    auto algo = intraNodeComm_->selectAllReduceAlgo(tensor);
    if (algo != intra_node_comm::AllReduceAlgo::NONE) {
      intraNodeComm_->allReduce(tensor, algo);
      return c10::make_intrusive<IntraNodeCommWork>();
    }
  }
  TORCH_CHECK(
      !isUnsupportedFloat8(tensor.scalar_type()),
      "Unsupported Float8 type for NCCL reduction");
  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      rank_, // rank
      "allreduce", // collective name
      tensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart_, // globalRankStart_
      globalRankStride_, // globalRankStride_
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash tensors.
  return allreduce_impl(tensor, "nccl:all_reduce", opts);
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  auto total_numel = check_gpu_tensors_same_device(tensors);
  TORCH_CHECK(
      !isUnsupportedFloat8(tensors.back().scalar_type()),
      "Unsupported Float8 type for NCCL reduction");

  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective and assume only one collective
                  // in coalesced range
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      rank_, // rank
      "allreduce_coalesced", // collective name
      total_numel, // inNelems
      total_numel, // outNelems
      tensors[0].scalar_type(), // dType
      // I'm not sure what in,outSplitSizes mean here.
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart_, // globalRankStart_
      globalRankStride_, // globalRankStride_
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash tensors.
  return collectiveCoalesced(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        auto ncclDataType = getNcclDataType(input.scalar_type());
        auto ncclReduceOp =
            getNcclReduceOp(opts.reduceOp, input, ncclDataType, comm);
        return ncclAllReduce(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            ncclDataType,
            ncclReduceOp,
            comm,
            stream.stream());
      },
      OpType::COALESCED,
      opts.asyncOp,
      "nccl:allreduce_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  if (tensor.is_complex()) {
    tensor = at::view_as_real(tensor);
  }
  check_gpu_single_tensor(tensor);

  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      opts.rootRank, // root rank
      "broadcast", // collective name
      tensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart_, // globalRankStart_
      globalRankStride_, // globalRankStride_
      this->getSize()); // worldSize

  const auto root = opts.rootRank + opts.rootTensor;
  bool nanCheck = (root == rank_);

  // avoidRecordStreams_ note: collective() will stash tensors.
  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        return ncclBcast(
            input.data_ptr(),
            input.numel(),
            getNcclDataType(input.scalar_type()),
            static_cast<int>(root),
            comm,
            stream.stream());
      },
      OpType::BROADCAST,
      opts.asyncOp,
      "nccl:broadcast",
      nanCheck);
}

// _broadcast_oop adds an out-of-place broadcast in PGNCCL
// Custom collectives may be implemented by coalescing broadcast operations
// One use-case is implementing a vector all_gather (all_gather_v)
// where unevenly sized inputs are gathered among participating ranks
// Since all_gather provides an out-of-place API, an all_gather_v
// semantic implemented inside pg_nccl.all_gather also needs to support
// out-of-place, for which an out-of-place broadcast is required to be added
c10::intrusive_ptr<Work> ProcessGroupNCCL::_broadcast_oop(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const BroadcastOptions& opts) {
  if (outputTensor.numel() != inputTensor.numel()) {
    C10_THROW_ERROR(
        ValueError,
        "Tensor input and output of _broadcast_oop must have the same number of elements ");
  }
  const auto root = opts.rootRank + opts.rootTensor;
  bool nanCheck = (root == rank_);
  return collective(
      inputTensor,
      outputTensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        return ncclBroadcast(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            getNcclDataType(input.scalar_type()),
            static_cast<int>(root),
            comm,
            stream.stream());
      },
      OpType::BROADCAST,
      opts.asyncOp,
      "nccl:_broadcast_oop",
      nanCheck);
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  if (tensor.is_complex()) {
    TORCH_CHECK(
        c10d::isComplexViewAsRealAllowed(opts.reduceOp),
        "reduce does not support",
        opts.reduceOp,
        "on complex tensors");
    tensor = at::view_as_real(tensor);
  }
  check_gpu_single_tensor(tensor);
  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      opts.rootRank, // root rank
      "reduce", // collective name
      tensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart_, // globalRankStart_
      globalRankStride_, // globalRankStride_
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash tensors.
  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        const auto root = opts.rootRank + opts.rootTensor;
        auto ncclDataType = getNcclDataType(input.scalar_type());
        auto ncclReduceOp =
            getNcclReduceOp(opts.reduceOp, input, ncclDataType, comm);
        return ncclReduce(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            ncclDataType,
            ncclReduceOp,
            static_cast<int>(root),
            comm,
            stream.stream());
      },
      OpType::REDUCE,
      opts.asyncOp,
      "nccl:reduce");
}

// _reduce_oop exposes an out-of-place reduce from PGNCCL
// Custom collectives may be implemented by coalescing reduce operations
// One use-case is implementing a vector reduce_scatter (reduce_scatter_v)
// where inputs are reduced and scattered unevenly among participating ranks
// Since reduce_scatter provides an out-of-place API, a reduce_scatter_v
// semantic implemented inside pg_nccl.reduce_scatter also needs to support
// out-of-place, for which an out-of-place reduce is required to be added
c10::intrusive_ptr<Work> ProcessGroupNCCL::_reduce_oop(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const ReduceOptions& opts) {
  if (outputTensor.numel() != inputTensor.numel()) {
    C10_THROW_ERROR(
        ValueError,
        "Tensor input and output of _reduce_oop must have the same number of elements ");
  }
  return collective(
      inputTensor,
      outputTensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        const auto root = opts.rootRank + opts.rootTensor;
        const auto ncclDataType = getNcclDataType(input.scalar_type());
        const auto ncclReduceOp =
            getNcclReduceOp(opts.reduceOp, input, ncclDataType, comm);
        return ncclReduce(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            ncclDataType,
            ncclReduceOp,
            (int)root,
            comm,
            stream.stream());
      },
      OpType::REDUCE,
      opts.asyncOp,
      "nccl:_reduce_oop");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  TORCH_CHECK(inputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto inputTensor = inputTensors.back();
  check_gpu_single_tensor(inputTensor);
  auto outputTensors_ = outputTensors.back();

  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "all_gather", // collective name
      inputTensor.numel(), // inNelems
      inputTensor.numel() * // outNelems
          this->getSize(),
      inputTensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSize
      globalRankStart_, // globalRankStart_
      globalRankStride_, // globalRankStride_
      this->getSize()); // worldSize

  bool same_size = check_same_size(outputTensors_);
  if (same_size) {
    // Flatten a vector of tensors into a single, stacked tensor.
    // we can handle only contiguous inputs, because we are
    // just sending ptr and numel to nccl
    inputTensor = inputTensor.contiguous();
    at::Tensor outputFlattened = newLikeFlat(outputTensors_);

    return collective(
        inputTensor,
        outputFlattened,
        [&](at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream) {
          // See [We actually don't need to stash anything here].
          return ncclAllGather(
              input.data_ptr(),
              output.data_ptr(),
              input.numel(),
              getNcclDataType(input.scalar_type()),
              comm,
              stream.stream());
        },
        [](at::cuda::CUDAStream& ncclStream,
           c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
          // avoidRecordStreams_ note: We actually don't need to stash anything
          // here.
          //  - inputTensors is stashed onto work->stashed_for_allocator_safety_
          //    in collective().
          //  - outputFlattened is stashed onto work->outputs_ in collective().
        },
        [&](at::cuda::CUDAStream& ncclStream,
            c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
          // User-facing outputTensors should be held by the user until after
          // waiting on work_, or the call makes no sense. We do a stashing here
          // in case user doesn't hold the outputTensors in downstream code,
          // which can cause an early recycle by the CachingAllocator, which can
          // lead to segfault or data corruption.
          if (opts.asyncOp) {
            work->stashed_for_allocator_safety_->stash(outputTensors_);
          }
          // Copy the flattened output tensors to the outputs.
          at::cuda::CUDAStreamGuard guard(ncclStream);
          for (const auto j : c10::irange(outputTensors_.size())) {
            // See [We actually don't need to stash anything here].
            outputTensors_[j].copy_(
                outputFlattened[static_cast<int64_t>(j)], true);
          }
        },
        OpType::ALLGATHER,
        opts.asyncOp,
        "nccl:all_gather");
  } else {
    const auto num_reduces = outputTensors_.size();
    startCoalescing();
    for (const int64_t i : c10::irange(static_cast<int64_t>(num_reduces))) {
      auto& output = outputTensors_[i];
      auto& input = (i == rank_) ? inputTensor : output;
      auto broadcastOpts =
          BroadcastOptions{i, int64_t(0), opts.timeout, opts.asyncOp};
      _broadcast_oop(output, input, broadcastOpts);
    }
    auto work = endCoalescing(OpType::ALLGATHER);
    return work;
  }
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllgatherOptions& /* unused */) {
  C10_THROW_ERROR(
      NotImplementedError,
      "ProcessGroupNCCL does not support allgather_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& opts) {
  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective and assume only one collective
                  // in coalesced range
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputs, // inputTensors
      outputs, // outputTensors
      rank_, // rank
      "allgather_into_tensor_coalesced", // collective name
      getTensorsNumel(inputs), // inNelems
      getTensorsNumel(outputs), // outNelems
      inputs[0].scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart_, // globalRankStart_
      globalRankStride_, // globalRankStride_
      this->getSize()); // worldSize

  return collectiveCoalesced(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        return ncclAllGather(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            getNcclDataType(input.scalar_type()),
            comm,
            stream.stream());
      },
      OpType::COALESCED,
      opts.asyncOp,
      "nccl:all_gather_into_tensor_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  TORCH_CHECK(outputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto outputTensor = outputTensors.back();
  check_gpu_single_tensor(outputTensor);
  auto inputTensors_ = inputTensors.back();
  TORCH_CHECK(
      !isUnsupportedFloat8(outputTensor.scalar_type()),
      "Unsupported Float8 type for NCCL reduction");

  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "reduce_scatter", // collective name
      outputTensor.numel() * this->getSize(), // inNelems
      outputTensor.numel(), // outNelems
      outputTensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart_, // globalRankStart_
      globalRankStride_, // globalRankStride_
      this->getSize()); // worldSize

  bool same_size = check_same_size(inputTensors_);
  if (same_size) {
    // Flatten a vector of tensors into a single, stacked tensor.
    outputTensor = outputTensor.contiguous();
    at::Tensor inputFlattened = newLikeFlat(inputTensors_);

    return collective(
        inputFlattened,
        outputTensor,
        [&](at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream) {
          const auto ncclDataType = getNcclDataType(input.scalar_type());
          const auto ncclReduceOp =
              getNcclReduceOp(opts.reduceOp, input, ncclDataType, comm);
          return ncclReduceScatter(
              input.data_ptr(),
              output.data_ptr(),
              output.numel(),
              ncclDataType,
              ncclReduceOp,
              comm,
              stream.stream());
        },
        [&](at::cuda::CUDAStream& ncclStream,
            c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
          // We only need to stash inputTensors.
          //  - inputFlattened is stashed onto
          //  work->stashed_for_allocator_safety_ in collective().
          //  - User-facing outputTensors is stashed onto work->outputs_ in
          //  collective(), and should also be held by the user until after
          //  waiting on work_.
          if (opts.asyncOp) {
            work->stashed_for_allocator_safety_->stash(inputTensors_);
          }
          // Copy the input tensors to the flattened inputs.
          at::cuda::CUDAStreamGuard guard(ncclStream);
          for (const auto j : c10::irange(inputTensors_.size())) {
            inputFlattened[static_cast<int64_t>(j)].copy_(
                inputTensors_[j], true);
          }
        },
        [&](at::cuda::CUDAStream&,
            c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
        OpType::REDUCE_SCATTER,
        opts.asyncOp,
        "nccl:reduce_scatter");
  } else {
    const auto num_reduces = inputTensors_.size();
    startCoalescing();
    for (const int i : c10::irange(static_cast<int>(num_reduces))) {
      auto& input = inputTensors_[i];
      auto& output = (i == rank_) ? outputTensor : input;
      auto reduceOpts = ReduceOptions{
          opts.reduceOp,
          static_cast<int64_t>(i),
          static_cast<int64_t>(0),
          opts.timeout,
          opts.asyncOp};
      _reduce_oop(output, input, reduceOpts);
    }
    auto work = endCoalescing(OpType::REDUCE_SCATTER);
    return work;
  }
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::_reduce_scatter_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const ReduceScatterOptions& opts) {
  if (inputTensor.dtype() != outputTensor.dtype()) {
    C10_THROW_ERROR(
        TypeError, "input tensor must be the same type as the output tensor.");
  }

  if (inputTensor.numel() != outputTensor.numel() * size_) {
    C10_THROW_ERROR(
        ValueError,
        "input tensor must be the same size as output size times world size");
  }

  const auto& tensor = outputTensor;
  TORCH_CHECK(
      !isUnsupportedFloat8(tensor.scalar_type()),
      "Unsupported Float8 type for NCCL reduction");
  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputTensor, // inputTensor
      outputTensor, // outputTensor
      rank_, // rank
      "_reduce_scatter_base", // collective name
      inputTensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dtype
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart_, // globalRankStart_
      globalRankStride_, // globalRankStride_
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash inputs and outputs.
  // Note 2: for asyncOp = false, we don't want to record streams because we
  // know that the NCCL stream will join back to the "current" stream right
  // after this op. So we might just as well keep the stream ownership of the
  // input/output tensors unchanged. The benefit would be that the
  // allocation/free of the tensors would look deterministic to the "current"
  // stream so that the caching allocator can reuse memory pool for this stream
  // in a clever way. This setting is added for libraries like FSDP which uses
  // `reduce_scatter_tensor`.

  return collective(
      inputTensor,
      outputTensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        auto ncclDataType = getNcclDataType(input.scalar_type());
        auto ncclReduceOp =
            getNcclReduceOp(opts.reduceOp, input, ncclDataType, comm);
        return ncclReduceScatter(
            input.data_ptr(),
            output.data_ptr(),
            output.numel(),
            ncclDataType,
            ncclReduceOp,
            comm,
            stream.stream());
      },
      OpType::_REDUCE_SCATTER_BASE,
      opts.asyncOp,
      "nccl:_reduce_scatter_base");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const ReduceScatterOptions& opts) {
  TORCH_CHECK(
      !isUnsupportedFloat8(inputs.back().scalar_type()),
      "Unsupported Float8 type for NCCL reduction");

  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective and assume only one collective
                  // in coalesced range
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputs, // inputTensors
      outputs, // outputTensors
      rank_, // rank
      "reduce_scatter_tensor_coalesced", // collective name
      getTensorsNumel(inputs), // inNelems
      getTensorsNumel(outputs), // outNelems
      inputs[0].scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart_, // globalRankStart_
      globalRankStride_, // globalRankStride_
      this->getSize()); // worldSize

  return collectiveCoalesced(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        auto ncclDataType = getNcclDataType(input.scalar_type());
        auto ncclReduceOp =
            getNcclReduceOp(opts.reduceOp, input, ncclDataType, comm);
        return ncclReduceScatter(
            input.data_ptr(),
            output.data_ptr(),
            output.numel(),
            ncclDataType,
            ncclReduceOp,
            comm,
            stream.stream());
      },
      OpType::COALESCED,
      opts.asyncOp,
      "nccl:reduce_scatter_tensor_coalesced");
}

c10::DeviceIndex ProcessGroupNCCL::guessDeviceId() const {
  // 1st choice: don't use this function if your API can take a device_id
  // argument.
  if (getBoundDeviceId().has_value()) {
    // 2nd choice: Use the bound GPU device id if available.
    // Bounded device id can be passed to `init_process_group`.
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return getBoundDeviceId().value().index();
  } else if (!usedDeviceIdxs_.empty()) {
    // 3rd choice: infer the device id from the used device ids.
    return *usedDeviceIdxs_.begin();
  }
  // This means there is not yet a NCCL collective being called
  // Here we have to use the best guesses and will use a single GPU to call
  // allreduce to achieve barrier.
  // In case the multiple processes fall into the same node, we use rank to
  // ensure that each process is on a different GPU
  // Note: it is better to use global rank because the group-local rank can be
  // offset wrt the device id if intra-node GPUs are sharded into multiple
  // dimensions.
  int devIdx = globalRank() % localDeviceCount_;
  if (devIdx == 0) { // only log on first rank of each node
    LOG(WARNING) << c10::str(
        "Guessing device ID based on global rank. ",
        "This can cause a hang if rank to GPU mapping is heterogeneous. ",
        "You can specify device_id in init_process_group()");
  }
  return static_cast<c10::DeviceIndex>(devIdx);
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::barrier(const BarrierOptions& opts) {
  RECORD_PARAM_COMMS(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      rank_, // rank
      "barrier", // collective name
      0, // inNelems
      0, // outNelems
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart_, // globalRankStart_
      globalRankStride_, // globalRankStride_
      this->getSize()); // worldSize

  // Device to use for barrier
  c10::DeviceIndex barDevIdx = -1;

  // Select device to use for barrier
  // 1st choice: Use user defined GPU device ids if provided
  if (!opts.device_ids.empty()) {
    // Use the first device id because PG NCCL is single-device now
    barDevIdx = static_cast<c10::DeviceIndex>(opts.device_ids[0]);
  } else {
    // 2nd choice: Use the bound or used GPU device id if available.
    barDevIdx = guessDeviceId();
  }

  TORCH_CHECK_WITH(
      ValueError,
      barDevIdx >= 0,
      "Failed to infer a GPU device id to perform barrier. ");
  auto barDevice = at::Device(at::DeviceType::CUDA, barDevIdx);

  // Create a dummy tensor on the device
  // Note: we use zeros() instead of empty() to prevent barrier from triggering
  // alarm when NaN checker is enabled.
  at::Tensor barrierTensor =
      at::zeros({1}, at::TensorOptions().device(barDevice).dtype(at::kFloat));

  // All reduce to achieve the barrier
  AllreduceOptions arOpts = AllreduceOptions();
  arOpts.asyncOp = opts.asyncOp;
  auto work = allreduce_impl(barrierTensor, "nccl:all_reduce_barrier", arOpts);

  if (opts.asyncOp) {
    // Work will take over barrierTensors
    auto ncclWork = dynamic_cast<ProcessGroupNCCL::WorkNCCL*>(work.get());
    // If user specified async, the work should not be nullptr
    TORCH_CHECK(ncclWork);
    // Put a marker here so that `work.wait()` issue by users does
    // barrier-specific thing: CPU sync
    ncclWork->isBarrierOp_ = true;
    return work;
  }

  // Otherwise, we are in sync mode, we directly wait here.
  // (It is a CPU wait for barrier)
  auto currentStream = at::cuda::getCurrentCUDAStream(barDevIdx);
  // CUDAStream wrapper will correctly use a DeviceGuard here
  currentStream.synchronize();
  // No work to return
  return nullptr;
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& opts) {
  check_gpu_single_tensor(outputTensor);
  check_gpu_single_tensor(inputTensor);
  if (outputSplitSizes.empty() && inputSplitSizes.empty()) {
    RECORD_PARAM_COMMS_DATA(
        std::make_tuple(
            static_cast<int64_t>(seqCollective_) + 1,
            false), // seq + 1 to match collective
        std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
        inputTensor, // inputTensor
        outputTensor, // outputTensor
        rank_, // rank
        "all_to_allv", // collective name
        inputTensor.numel(), // inNelems
        outputTensor.numel(), // outNelems
        inputTensor.scalar_type(), // dType
        std::vector<int64_t>(), // inSplitSizes
        std::vector<int64_t>(), // outSplitSizes
        globalRankStart_, // globalRankStart_
        globalRankStride_, // globalRankStride_
        this->getSize()); // worldSize

    // avoidRecordStreams_ note: collective() will stash inputTensors and
    // outputTensors.
    return collective(
        inputTensor,
        outputTensor,
        [&](at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream) {
          torch::cuda::nccl::all2all_single_equal_split(
              input, output, this->getSize(), comm, stream);
          return ncclSuccess;
        },
        OpType::ALLTOALL_BASE,
        opts.asyncOp,
        "nccl:all_to_all");
  } else {
    c10d::checkSplitSizes(inputSplitSizes, inputTensor, size_);
    c10d::checkSplitSizes(outputSplitSizes, outputTensor, size_);

    RECORD_PARAM_COMMS_DATA(
        std::make_tuple(
            static_cast<int64_t>(seqCollective_) + 1,
            false), // seq + 1 to match collective
        std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
        inputTensor, // inputTensor
        outputTensor, // outputTensor
        rank_, // rank
        "all_to_allv", // collective name
        inputTensor.numel(), // inNelems
        outputTensor.numel(), // outNelems
        inputTensor.scalar_type(), // dType
        inputSplitSizes, // inSplitSizes
        outputSplitSizes, // outSplitSizes
        globalRankStart_, // globalRankStart_
        globalRankStride_, // globalRankStride_
        this->getSize()); // worldSize

    // avoidRecordStreams_ note: collective() will stash inputTensors and
    // outputTensors.
    return collective(
        inputTensor,
        outputTensor,
        [&](at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream) {
          std::vector<size_t> send_lengths(size_);
          std::vector<size_t> recv_lengths(size_);
          std::vector<size_t> send_offsets(size_);
          std::vector<size_t> recv_offsets(size_);
          c10d::computeLengthsAndOffsets(
              inputSplitSizes, input, &send_lengths, &send_offsets);
          c10d::computeLengthsAndOffsets(
              outputSplitSizes, output, &recv_lengths, &recv_offsets);
          // See [Sync Streams].
          torch::cuda::nccl::all2all_single_unequal_split(
              input.data_ptr(),
              send_lengths.data(),
              send_offsets.data(),
              output.data_ptr(),
              recv_lengths.data(),
              recv_offsets.data(),
              input.element_size(),
              input.scalar_type(),
              comm,
              stream);
          return ncclSuccess;
        },
        OpType::ALLTOALL_BASE,
        opts.asyncOp,
        "nccl:all_to_all");
  }
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& opts) {
  int64_t input_total_numel = 0;
  int64_t output_total_numel = 0;
  // considering uneven all2all bw calculation
  // use split sizes field to record tensor list sizes
  std::vector<int64_t> inSplitSizes;
  std::vector<int64_t> outSplitSizes;

  auto device = outputTensors[0].device();
  for (const auto r : c10::irange(outputTensors.size())) {
    check_gpu_single_tensor(outputTensors[r]);
    check_gpu_single_tensor(inputTensors[r]);
    TORCH_CHECK(
        device == outputTensors[r].device() &&
            device == inputTensors[r].device(),
        "Tensors must be on the same device")
    input_total_numel += inputTensors[r].numel();
    output_total_numel += outputTensors[r].numel();
    inSplitSizes.push_back(inputTensors[r].numel());
    outSplitSizes.push_back(outputTensors[r].numel());
  }

  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "all_to_all", // collective name
      input_total_numel, // inNelems
      output_total_numel, // outNelems
      inputTensors.front().scalar_type(), // dType
      inSplitSizes, // inSplitSizes
      outSplitSizes, // outSplitSizes
      globalRankStart_, // globalRankStart_
      globalRankStride_, // globalRankStride_
      this->getSize()); // worldSize

  return collective(
      inputTensors,
      outputTensors,
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        torch::cuda::nccl::all2all(outputTensors, inputTensors, comm, stream);
        return ncclSuccess;
      },
      [&](at::cuda::CUDAStream&,
          c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
      [](at::cuda::CUDAStream&,
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
      OpType::ALLTOALL,
      opts.asyncOp,
      "nccl:all_to_all");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int /* unused */) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  check_gpu_single_tensor(tensor, true);

  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqP2P_) + (coalescing_state_ & CoalP2P ? 0 : 1),
          true), // the 1st p2p in coalesced range sets coalescing_state_ and
                 // bumps seqP2P_
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      dstRank, // dst rank
      "send", // collective name
      tensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart_, // globalRankStart_
      globalRankStride_, // globalRankStride_
      this->getSize()); // worldSize

  auto ret = pointToPoint(
      tensor,
      [&](at::Tensor& input,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream,
          int dst) {
        auto ncclDataType = getNcclDataType(input.scalar_type());
        return ncclSend(
            input.data_ptr(),
            input.numel(),
            ncclDataType,
            dst,
            comm,
            stream.stream());
      },
      dstRank,
      OpType::SEND,
      c10::str("nccl:send ", rank_, "->", dstRank).c_str());
  return ret;
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int /* unused */) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  check_gpu_single_tensor(tensor, true);

  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqP2P_) + (coalescing_state_ & CoalP2P ? 0 : 1),
          true), // the 1st p2p in coalesced range sets coalescing_state_ and
                 // bumps seqP2P_
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      srcRank, // src rank
      "recv", // collective name
      tensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart_, // globalRankStart_
      globalRankStride_, // globalRankStride_
      this->getSize()); // worldSize

  auto ret = pointToPoint(
      tensor,
      [&](at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream,
          int src) {
        auto ncclDataType = getNcclDataType(output.scalar_type());
        return ncclRecv(
            output.data_ptr(),
            output.numel(),
            ncclDataType,
            src,
            comm,
            stream.stream());
      },
      srcRank,
      OpType::RECV,
      c10::str("nccl:recv ", rank_, "<-", srcRank).c_str());
  return ret;
}

void ProcessGroupNCCL::groupStart() {
  C10D_NCCL_CHECK(ncclGroupStart(), std::nullopt);
  ++ncclActiveGroupCounter_;
}

void ProcessGroupNCCL::groupEnd() {
  C10D_NCCL_CHECK(ncclGroupEnd(), std::nullopt);
  --ncclActiveGroupCounter_;
}

void ProcessGroupNCCL::groupEndNonblocking(
    const std::shared_ptr<NCCLComm>& comm) {
#ifndef NCCL_HAS_COMM_NONBLOCKING
  C10D_NCCL_CHECK(ncclGroupEnd(), std::nullopt);
#else
  if (!useNonblocking()) {
    C10D_NCCL_CHECK(ncclGroupEnd(), std::nullopt);
  } else {
    C10D_NCCL_CHECK_TIMEOUT_GROUPEND(ncclGroupEnd(), comm, std::nullopt);
  }
#endif // NCCL_HAS_COMM_NONBLOCKING
  --ncclActiveGroupCounter_;
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    C10_THROW_ERROR(ValueError, "ProcessGroupNCCL::gather: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);

  TORCH_CHECK(inputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto inputTensor = inputTensors.back();
  check_gpu_single_tensor(inputTensor);

  std::vector<at::Tensor> outputs;

  if (getRank() == opts.rootRank) {
    if (outputTensors.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element output list containing a list with "
         << getSize() << " tensors.";
      invalidArgument(ss.str());
    } else if (outputTensors[0].size() != static_cast<size_t>(getSize())) {
      std::stringstream ss;
      ss << "Incorrect output list size " << outputTensors[0].size()
         << ". Output list size should be " << getSize()
         << ", same as size of the process group.";
      invalidArgument(ss.str());
    }

    const auto& options = inputTensor.options();
    const auto& sizes = inputTensor.sizes();
    assertTypeAndSizesMatch(invalidArgument, outputTensors[0], options, sizes);
    outputs = outputTensors[0];
  } else {
    // if not in the root rank, initialize outputs as empty list
    if (!outputTensors.empty()) {
      invalidArgument("requires empty output on non-root");
    }
    outputs = {};
    // append a empty tensor to the list, we don't use it but the
    // `collective` template function requires it to invoke its function
    outputs.emplace_back();
  }

  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      opts.rootRank, // root rank
      "gather", // collective name
      inputTensor.numel(), // inNelems
      inputTensor.numel() * this->getSize(), // outNelems
      inputTensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSize
      globalRankStart_, // globalRankStart_
      globalRankStride_, // globalRankStride_
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash inputTensors and
  // outputs, which == outputTensors[0] on the root rank where it matters.

  auto inputs = std::vector<at::Tensor>{inputTensor};
  return collective(
      inputs,
      outputs, // just to fit the collective interface
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        const auto root = opts.rootRank;
        torch::cuda::nccl::gather(
            inputTensor, outputs, comm, stream, static_cast<int32_t>(root));
        return ncclSuccess;
      },
      [](at::cuda::CUDAStream&,
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
      [](at::cuda::CUDAStream&,
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
      OpType::GATHER,
      opts.asyncOp,
      "nccl:gather");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    C10_THROW_ERROR(ValueError, "ProcessGroupNCCL::scatter: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);

  TORCH_CHECK(outputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto outputTensor = outputTensors.back();

  std::vector<at::Tensor> inputs;

  if (getRank() == opts.rootRank) {
    if (inputTensors.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element input list containing a list with "
         << getSize() << " tensors.";
      invalidArgument(ss.str());
    } else if (inputTensors[0].size() != static_cast<size_t>(getSize())) {
      std::stringstream ss;
      ss << "Incorrect input list size " << inputTensors[0].size()
         << ". Input list size should be " << getSize()
         << ", same as size of the process group.";
      invalidArgument(ss.str());
    }

    const auto& options = outputTensor.options();
    const auto& sizes = outputTensor.sizes();
    assertTypeAndSizesMatch(invalidArgument, inputTensors[0], options, sizes);
    inputs = inputTensors[0];
  } else {
    // if not in the root rank, initialize inputTensors as empty place holder
    // with an empty list
    if (!inputTensors.empty()) {
      invalidArgument("requires empty input on non-root");
    }
    inputs = {};
    // append a empty tensor to the list, we don't use it but the
    // `collective` template function requires it to invoke its function
    inputs.emplace_back();
  }

  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      opts.rootRank, // root rank
      "scatter", // collective name
      outputTensor.numel() * this->getSize(), // inNelems
      outputTensor.numel(), // outNelems
      outputTensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSize
      globalRankStart_, // globalRankStart_
      globalRankStride_, // globalRankStride_
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash outputTensors and
  // inputs, which == inputTensors[0] on the root rank where it matters.
  const auto root = opts.rootRank;
  bool nanCheck = (rank_ == root);

  auto outputs = std::vector<at::Tensor>{outputTensor};
  return collective(
      outputs,
      inputs, // just to fit the collective interface
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        torch::cuda::nccl::scatter(
            inputs, outputTensor, comm, stream, static_cast<int32_t>(root));
        return ncclSuccess;
      },
      [](at::cuda::CUDAStream&,
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
      [](at::cuda::CUDAStream&,
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
      OpType::SCATTER,
      opts.asyncOp,
      "nccl:scatter",
      nanCheck);
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::recvAnysource(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */) {
  C10_THROW_ERROR(
      NotImplementedError, "ProcessGroupNCCL does not support recvAnysource");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::_allgather_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const AllgatherOptions& opts) {
  check_gpu_single_tensor(input_tensor);
  check_gpu_single_tensor(output_tensor);

  if (input_tensor.dtype() != output_tensor.dtype()) {
    C10_THROW_ERROR(
        TypeError, "output tensor must have the same type as input tensor");
  }

  if (input_tensor.numel() * size_ != output_tensor.numel()) {
    C10_THROW_ERROR(
        ValueError,
        "output tensor size must be equal to world_size times input tensor size");
  }

  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      input_tensor, // inputTensors
      output_tensor, // outputTensors
      rank_, // rank
      "_allgather_base", // collective name
      input_tensor.numel(), // inNelems
      output_tensor.numel(), // outNelems
      output_tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSize
      globalRankStart_, // globalRankStart_
      globalRankStride_, // globalRankStride_
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash inputs and outputs.
  // Note 2: for asyncOp = false, we don't want to record streams because we
  // know that the NCCL stream will join back to the "current" stream right
  // after this op. So we might just as well keep the stream ownership of the
  // input/output tensors unchanged. The benefit would be that the
  // allocation/free of the tensors would look deterministic to the "current"
  // stream so that the caching allocator can reuse memory pool for this stream
  // in a clever way. This setting is added for libraries like FSDP which uses
  // `all_gather_into_tensor`.

  return collective(
      input_tensor,
      output_tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        return ncclAllGather(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            getNcclDataType(input.scalar_type()),
            comm,
            stream.stream());
      },
      OpType::_ALLGATHER_BASE,
      opts.asyncOp,
      "nccl:_all_gather_base");
}

// Create a memory allocator for NCCL. This allocator is used to allocate memory
// that supports NVLink Sharp functionality. This allocator is later pybinded to
// python, so that users can use it to create MemPool. For example:
// >>> pool = torch.cuda.MemPool(backend.mem_allocator)

// Allocate function
static void* _ncclMemAlloc(size_t size, int device, void* stream) {
#ifndef NCCL_HAS_MEM_ALLOC
  TORCH_CHECK(
      false, "NCCL mem allocator is not supported in this NCCL version");
#else
  LOG(INFO) << "NCCL mem allocator: allocating " << size << " bytes";
  at::cuda::OptionalCUDAGuard gpuGuard(device);
  void* ptr = nullptr;
  TORCH_CHECK(ncclMemAlloc(&ptr, size) == ncclSuccess, "ncclMemAlloc failed");
  return ptr;
#endif // NCCL_HAS_MEM_ALLOC
}

// Free function
static void _ncclMemFree(void* ptr, size_t size, int device, void* stream) {
#ifndef NCCL_HAS_MEM_ALLOC
  TORCH_CHECK(
      false, "NCCL mem allocator is not supported in this NCCL version");
#else
  LOG(INFO) << "NCCL mem allocator: freeing " << size << " bytes";
  at::cuda::OptionalCUDAGuard gpuGuard(device);
  TORCH_CHECK(ncclMemFree(ptr) == ncclSuccess, "ncclMemFree failed");
#endif // NCCL_HAS_MEM_ALLOC
}

// Create a `CUDAPluggableAllocator` that uses the above functions.
std::shared_ptr<c10::Allocator> ProcessGroupNCCL::getMemAllocator() {
  C10_LOG_API_USAGE_ONCE("ProcessGroupNCCL.getMemAllocator");
  c10::DeviceIndex deviceIdx = guessDeviceId();
  if (!supportsTensorAlloc(deviceIdx)) {
    TORCH_CHECK(
        false, "NCCL mem allocator is not supported in this NCCL version");
  }
  static std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
      ncclMemAllocator =
          torch::cuda::CUDAPluggableAllocator::createCustomAllocator(
              _ncclMemAlloc, _ncclMemFree);
  return ncclMemAllocator;
}

bool ProcessGroupNCCL::supportsTensorAlloc(c10::DeviceIndex deviceIdx) {
  // Check if NCCL has `ncclMemAlloc` and `ncclMemFree` functions
  int version = 0;
  // Rely on link-time versioning
  ncclGetVersion(&version);
  if (version < NCCL_VERSION(2, 19, 0)) {
    return false;
  }

  // We do an extra check to see if CUDA driver supports multicast.  If not, we
  // will return false. Although `ncclMemAlloc` will fall back to regular
  // `cudaMalloc` and hence not error out, we may still want to avoid creating a
  // separate memory pool for NCCL.
  return c10d::cuda::deviceSupportsMulticast(deviceIdx);
}

at::Tensor ProcessGroupNCCL::allocateTensor(
    long size,
    at::TensorOptions options) {
  // Some checks
  TORCH_CHECK_VALUE(options.has_device(), "Tensor options must include device");
  auto device = options.device();
  TORCH_CHECK_VALUE(
      device.is_cuda(),
      "NCCL tensor allocator expects cuda type but got " + c10::str(device))

  at::cuda::OptionalCUDAGuard gpuGuard(device);

  // Create memory pool
  if (!memPool_) {
    // Needs a CUDAAllocator
    auto allocator =
        reinterpret_cast<c10::cuda::CUDACachingAllocator::CUDAAllocator*>(
            getMemAllocator().get());
    // Pool is created
    memPool_ = std::make_unique<c10::cuda::MemPool>(allocator);
    // Register so that we call ncclCommRegister on all new allocations
    registerMemPool(memPool_.get(), /*symmetric*/ false);
    LOG(INFO) << logPrefix() << "Created memory pool";
  }

  // Allocate tensor under this MemPool's context
  auto tid = std::this_thread::get_id();
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      memPool_->device(), memPool_->id(), [=](cudaStream_t) {
        auto current_tid = std::this_thread::get_id();
        return current_tid == tid;
      });
  at::Tensor tensor = at::empty({size}, options);
  c10::cuda::CUDACachingAllocator::endAllocateToPool(
      memPool_->device(), memPool_->id());
  c10::cuda::CUDACachingAllocator::releasePool(
      memPool_->device(), memPool_->id());
  LOG(INFO) << logPrefix() << "Allocated tensor of size " << size
            << " from memory pool";
  return tensor;
}

#ifdef NCCL_HAS_COMM_SHRINK
c10::intrusive_ptr<Backend> ProcessGroupNCCL::shrink(
    const std::vector<int64_t>& ranks_to_exclude,
    int shrink_flags,
    const c10::intrusive_ptr<Backend::Options>& opts_override) {
  // Runtime version check with better error message
  auto runtime_version = torch::cuda::nccl::version();
  TORCH_CHECK(
      runtime_version >= NCCL_VERSION(2, 27, 0),
      "ProcessGroupNCCL::shrink requires NCCL version 2.27.0 or later. "
      "Found version: ",
      runtime_version);

  // Early validation with detailed error messages
  TORCH_CHECK_VALUE(
      !ranks_to_exclude.empty(), "ranks_to_exclude cannot be empty");
  TORCH_CHECK_VALUE(
      static_cast<int>(ranks_to_exclude.size()) < size_,
      "Cannot exclude all ranks (",
      ranks_to_exclude.size(),
      " >= ",
      size_,
      ")");

  // Validate ranks and convert to int efficiently
  std::vector<int> int_ranks_to_exclude;
  int_ranks_to_exclude.reserve(ranks_to_exclude.size());
  for (int64_t rank : ranks_to_exclude) {
    TORCH_CHECK_VALUE(
        rank >= 0 && rank < size_,
        "Invalid rank ",
        rank,
        " for group size ",
        size_);
    int_ranks_to_exclude.push_back(static_cast<int>(rank));
  }

  // Get primary communicator with better error context
  auto primary_device_index = guessDeviceId();
  auto primary_device = at::Device(at::kCUDA, primary_device_index);
  const auto primary_key = getKeyFromDevice(primary_device);

  std::shared_ptr<NCCLComm> primary_comm = getNCCLComm(primary_key);
  TORCH_CHECK(
      primary_comm,
      "Primary NCCL communicator for device ",
      primary_device,
      " (key: ",
      primary_key,
      ") is not initialized");

  // Cache device index before shrink operation
  at::DeviceIndex parent_device_index = primary_comm->getDeviceIndex();

  ncclConfig_t* config = nullptr;
  // Default to inheriting from parent options
  bool high_priority_stream = options_->is_high_priority_stream;
  if (opts_override) {
    auto nccl_opts =
        c10::static_intrusive_pointer_cast<ProcessGroupNCCL::Options>(
            opts_override);
    config = &nccl_opts->config;
    // If user provided override options, honor is_high_priority_stream as well
    high_priority_stream = nccl_opts->is_high_priority_stream;
  }

  std::shared_ptr<NCCLComm> shrunk_comm = NCCLComm::shrink(
      primary_comm.get(),
      int_ranks_to_exclude,
      (config != nullptr ? config : &options_->config),
      shrink_flags);

  // Calculate new size and get NCCL-assigned rank
  int new_size = size_ - static_cast<int>(ranks_to_exclude.size());
  int new_rank = shrunk_comm->rank_;

  // Create new ProcessGroupNCCL with optimized options cloning
  auto new_store = store_->clone();
  auto new_opts = ProcessGroupNCCL::Options::create(high_priority_stream);
  new_opts->timeout = options_->timeout;
  if (config != nullptr) {
    new_opts->config = *config;
  } else {
    new_opts->config = options_->config;
  }

  auto new_pg = c10::make_intrusive<ProcessGroupNCCL>(
      new_store, new_rank, new_size, new_opts);

  // Set up the new process group with optimized device setup
  new_pg->initializeDeviceStateForComm(
      at::Device(at::kCUDA, parent_device_index), shrunk_comm);

  return c10::static_intrusive_pointer_cast<Backend>(new_pg);
}

#else // !NCCL_HAS_COMM_SHRINK
// Backend interface override: raise consistent error when shrink is
// unsupported.
c10::intrusive_ptr<Backend> ProcessGroupNCCL::shrink(
    const std::vector<int64_t>& /*ranks_to_exclude*/,
    int /*shrink_flags*/,
    const c10::intrusive_ptr<Backend::Options>& /*opts_override*/) {
  TORCH_CHECK(
      false,
      "ProcessGroupNCCL::shrink requires NCCL version 2.27.0 or later, "
      "but PyTorch was built with an older version or without NCCL shrink support.");
}

#endif // NCCL_HAS_COMM_SHRINK

void ProcessGroupNCCL::initializeDeviceStateForComm(
    const at::Device& device,
    std::shared_ptr<NCCLComm> comm) {
  const auto key = getKeyFromDevice(device);
  std::unique_lock<std::mutex> lock(mutex_);
  at::cuda::OptionalCUDAGuard gpuGuard(device);

  bool force_high = getCvarBool(TORCH_NCCL_HIGH_PRIORITY, false);
  auto stream = at::cuda::getStreamFromPool(
      options_->is_high_priority_stream || force_high);

  devNCCLCommMap_[key] = comm;
  ncclStreams_.emplace(key, stream);
  ncclEvents_.emplace(key, at::cuda::CUDAEvent(cudaEventDisableTiming));
  usedDeviceIdxs_.insert(device.index());

  if (shouldAllCommunicatorsRegisterAllTensors()) {
    std::lock_guard<std::mutex> map_lock(ncclCommMemPoolMapMutex);
    ncclCommMemPoolMap.emplace(std::move(comm), MemPoolSet{});
  }
}

} // namespace c10d

#endif // USE_C10D_NCCL
