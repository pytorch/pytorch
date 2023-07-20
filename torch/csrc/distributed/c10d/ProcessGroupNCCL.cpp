#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/UCCForNCCL.hpp>
#include <sstream>

#ifdef USE_C10D_NCCL

#include <exception>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#include <torch/csrc/distributed/c10d/TraceUtils.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <torch/csrc/cuda/nccl.h>
#include <torch/torch.h>

namespace c10d {

constexpr const char* const kNCCLAbortedCommStoreKey = "NCCLABORTEDCOMM";

namespace {

#if defined(NCCL_MAJOR) && \
    ((NCCL_MAJOR > 2) || (NCCL_MAJOR == 2) && (NCCL_MINOR >= 10))
#define NCCL_HAS_AVG 1
#endif

// NCCL op mapping
const std::map<ReduceOp::RedOpType, ncclRedOp_t> ncclOp = {
    {ReduceOp::MIN, ncclMin},
    {ReduceOp::MAX, ncclMax},
    {ReduceOp::SUM, ncclSum},
    {ReduceOp::PRODUCT, ncclProd},
#ifdef NCCL_HAS_AVG
    {ReduceOp::AVG, ncclAvg},
#endif
};

// NCCL type typing
std::map<at::ScalarType, ncclDataType_t> ncclDataType = {
    {at::kChar, ncclInt8},
    {at::kByte, ncclUint8},
    {at::kFloat, ncclFloat},
    {at::kDouble, ncclDouble},
    {at::kInt, ncclInt32},
    {at::kLong, ncclInt64},
    {at::kHalf, ncclHalf},
    {at::kBool, ncclUint8},
#if HAS_NCCL_BF16_DATATYPE
    {at::kBFloat16, ncclBfloat16},
#endif
};

// Helper function that gets the data type and issues error if not supported
ncclDataType_t getNcclDataType(at::ScalarType type) {
  auto it = ncclDataType.find(type);
  TORCH_CHECK(
      it != ncclDataType.end(),
      "Input tensor data type is not supported for NCCL process group: ",
      type);
  return it->second;
}

#ifdef ENABLE_NCCL_PREMUL_SUM_SUPPORT
template <typename T, ncclDataType_t dataType>
ncclRedOpRAII unpackPreMulSum(
    const ReduceOp& reduceOp,
    const ncclComm_t& comm,
    int dev_in_group) {
  const auto* preMulSupplement =
      reinterpret_cast<NCCLPreMulSumSupplement*>(reduceOp.supplement_.get());
  ncclRedOp_t preMulSum;
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
      /*scalar=*/has_tensor ? const_cast<T*>(ptr_factor) : &scalar_factor,
      dataType,
      residence,
      comm);
  return ncclRedOpRAII(preMulSum, comm);
}
#endif

ncclRedOpRAII getNcclReduceOp(
    const ReduceOp& reduceOp,
    at::Tensor& input,
    const ncclDataType_t& dataType,
    const ncclComm_t& comm,
    int dev_in_group) {
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
        TORCH_CHECK(false, "Cannot use ReduceOp.AVG with boolean inputs");
      }
#endif
    }
    if (reduceOp == ReduceOp::PREMUL_SUM) {
#ifdef ENABLE_NCCL_PREMUL_SUM_SUPPORT
      switch (dataType) {
        case ncclHalf:
          return unpackPreMulSum<at::Half, ncclHalf>(
              reduceOp, comm, dev_in_group);
        case ncclFloat:
          return unpackPreMulSum<float, ncclFloat>(
              reduceOp, comm, dev_in_group);
        case ncclDouble:
          return unpackPreMulSum<double, ncclDouble>(
              reduceOp, comm, dev_in_group);
        default:
          TORCH_CHECK(
              false, "PreMulSum Data type must be half, float, or double");
          ncclRedOp_t unused;
          return unused;
      }
#else
      TORCH_CHECK(false, "PreMulSum requires NCCL>=2.11.1");
#endif
    }
    return ncclOp.at(reduceOp);
  } catch (const std::out_of_range& e) {
    switch (reduceOp) {
      case ReduceOp::AVG:
        TORCH_CHECK(
            false,
            "AVG requires NCCL 2.10+. The current version is ",
            NCCL_MAJOR,
            ".",
            NCCL_MINOR);
        break;
      case ReduceOp::BAND:
        TORCH_CHECK(false, "Cannot use ReduceOp.BAND with NCCL");
        break;
      case ReduceOp::BOR:
        TORCH_CHECK(false, "Cannot use ReduceOp.BOR with NCCL");
        break;
      case ReduceOp::BXOR:
        TORCH_CHECK(false, "Cannot use ReduceOp.BXOR with NCCL");
        break;
      default:
        TORCH_CHECK(false, "Unhandled ReduceOp");
        break;
    }
  }
}

// Get the deviceList String from the list of devices
std::string getKeyFromDevices(const std::vector<at::Device>& devices) {
  std::string deviceList;
  for (auto& device : devices) {
    if (deviceList.empty()) {
      deviceList = std::to_string(device.index());
    } else {
      deviceList += "," + std::to_string(device.index());
    }
  }
  return deviceList;
}

std::string getKeySendRecv(int myRank, int peer) {
  int lowRank = myRank < peer ? myRank : peer;
  int highRank = myRank < peer ? peer : myRank;
  std::string sendRecvPair =
      std::to_string(lowRank) + ":" + std::to_string(highRank);
  return sendRecvPair;
}

// Get the list of devices from list of tensors
std::vector<at::Device> getDeviceList(const std::vector<at::Tensor>& tensors) {
  std::vector<at::Device> res;
  res.reserve(tensors.size());
  for (auto& tensor : tensors) {
    // tensors must all be on the same device, or all on distinct devices.
    // The line below assumes that constraint has already been enforced
    // (by check_gpu_tensors_same_device or
    // check_gpu_tensors_different_devices).
    if (res.size() == 0 || tensor.device() != res[0]) {
      res.push_back(tensor.device());
    }
  }
  return res;
}

// Return CUDA device with ordinal given by input rank.
at::Device getDeviceForRank(int rank) {
  TORCH_CHECK(rank >= 0, "Invalid rank ", rank);
  auto numGPUs = at::cuda::getNumGPUs();
  int16_t deviceIdx = static_cast<int16_t>(rank % numGPUs);
  return at::Device(at::DeviceType::CUDA, deviceIdx);
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
void syncStreams(
    const std::vector<at::Device>& devices,
    std::vector<at::cuda::CUDAEvent>& ncclEvents,
    std::vector<at::cuda::CUDAStream>& ncclStreams) {
  for (const auto i : c10::irange(devices.size())) {
    at::cuda::CUDAStream& ncclStream = ncclStreams[i];
    at::cuda::CUDAEvent& ncclEvent = ncclEvents[i];
    ncclEvent.record(at::cuda::getCurrentCUDAStream(devices[i].index()));
    ncclEvent.block(ncclStream);
  }
}

// Given a ncclUniqueId, convert it to a string representation that can be put
// in the store.
std::string buildNcclUniqueIdStr(const ncclUniqueId& ncclID) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&ncclID);
  std::ostringstream oss;
  for (const auto i : c10::irange(NCCL_UNIQUE_ID_BYTES)) {
    oss << std::hex << static_cast<int>(bytes[i]);
  }
  return oss.str();
}

std::string getNcclAbortedCommStoreKey(const std::string ncclIdStr) {
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
    TORCH_CHECK(
        status == c10::cuda::CaptureStatus::None,
        "Capturing NCCL collectives is only allowed with NCCL >= 2.9.6");
  }
}

} // namespace

const int64_t ProcessGroupNCCL::kWatchdogThreadSleepMillis = 1000;
constexpr int64_t kSynchronizeBusyWaitMillis = 10;
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

ProcessGroupNCCL::WorkNCCL::WorkNCCL(
    const std::vector<at::Device>& devices,
    int rank,
    OpType opType,
    uint64_t seq,
    const char* profilingTitle,
    const c10::optional<std::vector<at::Tensor>>& inputs,
    bool desyncDebug)
    : Work(rank, opType, profilingTitle, inputs),
      devices_(devices),
      workStartTime_(std::chrono::steady_clock::now()),
      seq_(seq) {
  // Creates the CUDA event wrappers
  // Note: The actual events are lazily created when first recorded to with
  // DEFAULT_FLAGS = cudaEventDisableTiming.
  if (desyncDebug) {
    ncclStartEvents_ =
        std::make_shared<std::vector<at::cuda::CUDAEvent>>(devices.size());
  }
  ncclEndEvents_ =
      std::make_shared<std::vector<at::cuda::CUDAEvent>>(devices.size());
  ncclComms_.resize(devices.size());
}

ProcessGroupNCCL::WorkNCCL::WorkNCCL(const WorkNCCL& w)
    : Work(w.rank_, w.opType_),
      std::enable_shared_from_this<WorkNCCL>(w),
      devices_(w.devices_),
      ncclStartEvents_(w.ncclStartEvents_),
      ncclEndEvents_(w.ncclEndEvents_),
      ncclComms_(w.ncclComms_),
      blockingWait_(w.blockingWait_),
      opTimeout_(w.opTimeout_),
      workStartTime_(w.workStartTime_),
      seq_(w.seq_),
      startTraceUpdated_(w.startTraceUpdated_),
      numelIn_(w.numelIn_),
      numelOut_(w.numelOut_),
      store_(w.store_) {
  exception_ = w.exception_;
}

ProcessGroupNCCL::WorkNCCL::~WorkNCCL() = default;

bool ProcessGroupNCCL::WorkNCCL::isCompleted() {
  checkAndSetException();
  return exception() || finishedGPUExecutionInternal();
}

bool ProcessGroupNCCL::WorkNCCL::isStarted() {
  checkAndSetException();
  return exception() || startedGPUExecutionInternal();
}

bool ProcessGroupNCCL::WorkNCCL::isSuccess() const {
  if (exception()) {
    // Already detected an exception.
    return false;
  }

  return !checkForNCCLErrors(ncclComms_) && finishedGPUExecutionInternal();
}

void ProcessGroupNCCL::WorkNCCL::checkAndSetException() {
  if (exception()) {
    // We already have an exception.
    return;
  }

  auto exception_ptr = checkForNCCLErrors(ncclComms_);
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = exception_ptr;
  if (exception_) {
    LOG(INFO) << "[Rank " << rank_ << "]"
              << " found async exception when checking for NCCL errors: "
              << getExceptionMsgFromExceptionPtr(exception_);
  }
}

void ProcessGroupNCCL::WorkNCCL::setException(
    std::exception_ptr exception_ptr) {
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = exception_ptr;
}

// Helper that checks if the NCCL kernels are completed on the GPUs
bool ProcessGroupNCCL::WorkNCCL::finishedGPUExecution() {
  checkAndSetException();
  return finishedGPUExecutionInternal();
}

bool ProcessGroupNCCL::WorkNCCL::startedGPUExecutionInternal() const {
  for (const auto i : c10::irange(devices_.size())) {
    // Checking the work's corresponding CUDA events' status
    if (!(*ncclStartEvents_)[i].query()) {
      return false;
    }
  }
  return true;
}

bool ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const {
  try {
    for (const auto i : c10::irange(devices_.size())) {
      // Checking the work's corresponding CUDA events' status
      if (!(*ncclEndEvents_)[i].query()) {
        return false;
      }
    }
  } catch (const std::exception& e) {
    if (std::string(e.what()).find("driver shutting down") ==
        std::string::npos) {
      throw;
    }
    LOG(INFO) << "[Rank " << rank_
              << "] Event query failed with exception: " << e.what();
  }
  return true;
}

bool ProcessGroupNCCL::WorkNCCL::checkTimeout(
    c10::optional<std::chrono::milliseconds> timeout) {
  auto currentTimepoint = std::chrono::steady_clock::now();
  auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      currentTimepoint - workStartTime_);
  auto workTimeout = timeout ? *timeout : opTimeout_;

  if (timeElapsed < workTimeout)
    return false;

  // Timed out

  // There is already an error, we don't override it
  if (exception())
    return true;

  std::string exceptionMsg = c10::str(
      "[Rank ",
      rank_,
      "] ",
      "Watchdog caught collective operation timeout: ",
      *this,
      " ran for ",
      timeElapsed.count(),
      " milliseconds before timing out.");

  LOG(ERROR) << exceptionMsg;
  std::exception_ptr exception_ptr =
      std::make_exception_ptr(std::runtime_error(exceptionMsg));
  setException(exception_ptr);
  return true;
}

void ProcessGroupNCCL::WorkNCCL::handleException(
    ErrorHandlingMode errorHandling) {
  if (exception_) {
    auto exceptionMsg = c10::str(
        "Some NCCL operations have failed or timed out. Due to the ",
        "asynchronous nature of CUDA kernels, subsequent GPU operations ",
        "might run on corrupted/incomplete data.");
    LOG(ERROR) << exceptionMsg;
    C10_LOG_API_USAGE_ONCE("ProcessGroupNCCL.WorkNCCL.handleException");

    if (SHOULD_TEAR_DOWN(errorHandling)) {
      auto tearDownMsg = c10::str(
          "To avoid data inconsistency, we are taking the entire process down.");
      LOG(ERROR) << tearDownMsg;
      std::rethrow_exception(exception_);
    }
  }
}

void ProcessGroupNCCL::WorkNCCL::synchronize() {
  // Call Synchronize without a timeout. We use this method to avoid adding a
  // timeout argument to the public synchronize API.
  synchronizeInternal(kNoTimeout);
}

void ProcessGroupNCCL::WorkNCCL::synchronizeStreams() {
  for (const auto i : c10::irange(devices_.size())) {
    auto currentStream = at::cuda::getCurrentCUDAStream(devices_[i].index());
    // Block the current stream on the NCCL stream
    (*ncclEndEvents_)[i].block(currentStream);
  }

  if (avoidRecordStreams_) {
    stashed_for_allocator_safety_->clear();
  }
}

// Waiting on the work's corresponding CUDA events
void ProcessGroupNCCL::WorkNCCL::synchronizeInternal(
    std::chrono::milliseconds timeout) {
  synchronizeStreams();

  // In case of blocking, wait for the operation to complete.
  if (blockingWait_) {
    while (!isCompleted()) {
      bool timedOut = checkTimeout(
          timeout == kNoTimeout ? c10::nullopt : c10::make_optional(timeout));
      // Explicitly abort ncclComms here before throwing this timed out
      // exception to users.
      // If throwing timed out excepiton without aborting nccl communicators
      // here, it was observed that CUDA GPU will have 100% utilization and
      // can not run new events successfully.
      if (timedOut) {
        std::string exceptionMsg = c10::str(
            "[Rank ",
            rank_,
            "] Work ",
            (*this),
            " timed out in blocking wait (NCCL_BLOCKING_WAIT=1).");
        LOG(ERROR) << exceptionMsg;
        break;
      }
      // Yield
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
    }
    // exception() includes timeout and error during blocking wait
    if (exception()) {
      // Abort NCCL communicators
      abort();
      // Throw exception (from main thread here)
      handleException(TearDown);
    }
  }

  // Device synchronize only after we've completed timeout checks.
  if (!barrierTensors_.empty()) {
    // If we use the work to do barrier, we should block here
    at::cuda::OptionalCUDAGuard gpuGuard;
    for (auto& device : devices_) {
      gpuGuard.set_index(device.index());
      AT_CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
}

// Same as calling synchronize().
bool ProcessGroupNCCL::WorkNCCL::wait(std::chrono::milliseconds timeout) {
  RECORD_PARAM_COMMS(
      static_cast<int>(this->seq_), // seq
      0, // process group ptr
      rank_, // rank
      "wait", // colName
      0, // inSize
      0, // outSize
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>()); // outSplitSizes
  synchronizeInternal(timeout);
  // Always return true, because abort API is not implemented.
  return true;
}

void ProcessGroupNCCL::WorkNCCL::abort() {
  // Abort all communicators of this work
  for (const auto& ncclComm : ncclComms_) {
    ncclComm->ncclCommAbort();
  }
}

ProcessGroupNCCL::CoalescedWorkNCCL::CoalescedWorkNCCL(
    std::vector<ProcessGroupNCCL::WorkNCCL> works,
    int rank,
    OpType opType)
    : Work(rank, opType, nullptr), works_(std::move(works)) {}

ProcessGroupNCCL::CoalescedWorkNCCL::~CoalescedWorkNCCL() = default;

c10::intrusive_ptr<ProcessGroupNCCL::CoalescedWorkNCCL> ProcessGroupNCCL::
    initCoalescedWork(
        const std::vector<c10::intrusive_ptr<Work>>& works,
        int rank,
        OpType opType) {
  std::vector<ProcessGroupNCCL::WorkNCCL> ncclWorks;
  ncclWorks.reserve(works.size());
  for (auto& work : works) {
    ncclWorks.push_back(*static_cast<ProcessGroupNCCL::WorkNCCL*>(work.get()));
  }
  return c10::make_intrusive<ProcessGroupNCCL::CoalescedWorkNCCL>(
      ncclWorks, rank, opType);
}

// Same as calling synchronize().
bool ProcessGroupNCCL::CoalescedWorkNCCL::wait(
    std::chrono::milliseconds timeout) {
  for (auto& w : works_) {
    w.wait(timeout);
  }
  // Always return true, because abort API is not implemented.
  return true;
}

ProcessGroupNCCL::ProcessGroupNCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : Backend(rank, size),
      store_(store),
      options_(options),
      ncclCommCounter_(0),
      traceKeyStart_(getTraceStartKey("NCCL", rank)),
      traceKeyEnd_(getTraceEndKey("NCCL", rank)),
      terminateProcessGroup_(false) {
  TORCH_CHECK(
      at::cuda::getNumGPUs() != 0,
      "ProcessGroupNCCL is only supported with GPUs, no GPUs found!");
  blockingWait_ = parseEnvVarFlag(NCCL_BLOCKING_WAIT);
  asyncErrorHandling_ = static_cast<ErrorHandlingMode>(
      parseEnvVarIntDefault(NCCL_ASYNC_ERROR_HANDLING, 1 /*TearDown*/));
  desyncDebug_ = parseEnvVarFlag(NCCL_DESYNC_DEBUG) ||
      (dist_debug_level_ >= DebugLevel::Detail);
  avoidRecordStreams_ = parseEnvVarFlag(TORCH_NCCL_AVOID_RECORD_STREAMS);

  if (blockingWait_) {
    if (asyncErrorHandling_ != NoHandling || desyncDebug_) {
      LOG(INFO) << "[Rank " << rank_ << "] NCCL_BLOCKING_WAIT and "
                << "NCCL_ASYNC_ERROR_HANDLING|NCCL_DESYNC_DEBUG"
                << "should not both be enabled. "
                << "Only NCCL_BLOCKING_WAIT is being used in this process.";
      asyncErrorHandling_ = NoHandling;
      desyncDebug_ = false;
    }
  } else {
    if (desyncDebug_ && asyncErrorHandling_ == NoHandling) {
      LOG(INFO) << "[Rank " << rank_
                << "] NCCL_DESYNC_DEBUG and NCCL_ASYNC_ERROR_HANDLING "
                << "must both be enabled. "
                << "Enabling NCCL_ASYNC_ERROR_HANDLING.";
      asyncErrorHandling_ = TearDown;
    }
  }

  if (parseEnvVarFlag(ENABLE_NCCL_HEALTH_CHECK)) {
    // Perform health check by initializing dummy communicators and destroying
    // them. This will help indicate any NCCL-related issues prior to the first
    // collective.
    // Run it in a separate thread and wait on CV to handle timeouts, since
    // majority of getNCCLComm failures are hangs.
    runHealthCheck();
  }

#ifdef ENABLE_NCCL_ERROR_CHECKING
  ncclCommWatchdogThread_ =
      std::thread(&ProcessGroupNCCL::ncclCommWatchdog, this);
#endif

  init();
  const std::string OFF = "OFF";
  const char* torch_distributed_debug =
      parseEnvVarString("TORCH_DISTRIBUTED_DEBUG", OFF.c_str());
  const char* nccl_debug = parseEnvVarString("NCCL_DEBUG", OFF.c_str());
  LOG(INFO) << "[Rank " << rank_
            << "] ProcessGroupNCCL initialized with following options:"
            << "\nNCCL_ASYNC_ERROR_HANDLING: " << asyncErrorHandling_
            << "\nNCCL_DESYNC_DEBUG: " << desyncDebug_
            << "\nNCCL_BLOCKING_WAIT: " << blockingWait_
            << "\nTIMEOUT(ms): " << options_->timeout.count()
            << "\nUSE_HIGH_PRIORITY_STREAM: "
            << options_->is_high_priority_stream
            << "\n TORCH_DISTRIBUTED_DEBUG: "
            << std::string(torch_distributed_debug)
            << "\n NCCL_DEBUG: " << std::string(nccl_debug);

  RECORD_PARAM_COMMS(
      0, // seq
      reinterpret_cast<std::intptr_t>(this), // process group ptr
      rank, // rank
      "init", // colName
      0, // inSize
      0, // outSize
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>()); // outSplitSizes

#ifdef USE_NCCL_WITH_UCC
  static c10::once_flag initialize_ucc_lib_flag;
  c10::call_once(initialize_ucc_lib_flag, [&] {
    uccLib_ = loadTorchUCC();
    if (uccLib_ != nullptr) {
      LOG(INFO) << "[Rank " << rank_ << "] torch_ucc.so loaded";
    }
  });

  if (uccLib_ != nullptr) {
    LOG(INFO) << "[Rank " << rank_ << "] torch_ucc.so loaded";
    typedef c10::intrusive_ptr<Backend> fn(
        const c10::intrusive_ptr<Store>& store, int rank, int size);
    auto createProcessGroupUCC =
        reinterpret_cast<fn*>(uccLib_->sym("createProcessGroupUCC"));
    if (createProcessGroupUCC != nullptr) {
      uccPG_ = createProcessGroupUCC(store, rank_, size_);
      LOG(INFO) << "[Rank " << rank_ << "] ProcessGroupUCC created.";
    }
  }
#endif
}

void ProcessGroupNCCL::runHealthCheck() {
  // Run health check in a separate thread and wait on CV to handle timeouts,
  // since majority of getNCCLComm failures are hangs.

  struct HealthCheckData {
    std::mutex healthCheckMutex;
    std::condition_variable healthCheckCv;
    bool healthCheckSuccess = false;
    std::exception_ptr healthCheckException;
  };

  HealthCheckData healthCheckData;
  auto t = std::thread([&healthCheckData, this]() {
    try {
      std::vector<at::Device> rankDevice = {getDeviceForRank(rank_)};
      const auto key = getKeyFromDevices(rankDevice);
      // OpType does not matter, only need to set to not go through send/recv
      // path.
      getNCCLComm(key, rankDevice, OpType::ALLREDUCE);
      // Now destroy the communicators and remove them from cache so we don't
      // use destroyed communicators.
      destroyNCCLComms(key);
      // Notify main thread the health check is complete.
      {
        std::lock_guard<std::mutex> lk(healthCheckData.healthCheckMutex);
        healthCheckData.healthCheckSuccess = true;
      }
      healthCheckData.healthCheckCv.notify_one();
    } catch (const std::exception& e) {
      // Populate exception ptr.
      healthCheckData.healthCheckException = std::current_exception();
      // Unblock waiting main thread which will report exception.
      healthCheckData.healthCheckCv.notify_one();
    } // Unknown exceptions will just cause the program to terminate.
  });
  // We don't need to join the thread, just need to verify health check via the
  // CV. Hence we detach the thread here.
  t.detach(); // NOLINT
  LOG(INFO) << "[Rank " << rank_ << "]"
            << " will wait up to " << options_->timeout.count()
            << " msec for NCCL health check to complete.";
  std::unique_lock<std::mutex> lock(healthCheckData.healthCheckMutex);
  healthCheckData.healthCheckCv.wait_for(
      lock, options_->timeout, [&healthCheckData]() {
        return healthCheckData.healthCheckSuccess;
      });

  if (healthCheckData.healthCheckException) {
    std::rethrow_exception(healthCheckData.healthCheckException);
  }
  // If there is no exception, the likely culprit is a timeout/hang which is how
  // most communicator init issues manifest themselves.
  TORCH_CHECK(
      healthCheckData.healthCheckSuccess,
      "ProcessGroupNCCL: Health check failure: Failed to initialize NCCL communicator on rank ",
      rank_);
}

void ProcessGroupNCCL::setSequenceNumberForGroup() {
} // NCCL just starts sequence numbers at 0.

uint64_t ProcessGroupNCCL::getSequenceNumberForGroup() {
  return seq_;
}

void abortCommsFromMap(
    std::unordered_map<std::string, std::vector<std::shared_ptr<NCCLComm>>>&
        ncclCommsMap,
    const int rank,
    c10::optional<std::string> abortReason) {
  // The process may control multiple devices, loop through the communicators on
  // each device
  for (auto& it : ncclCommsMap) {
    auto& devName = it.first;
    auto& ncclComms = it.second;

    for (const auto& ncclComm : ncclComms) {
      ncclComm->ncclCommAbort(abortReason);
    }
    // Note that we don't remove the aborted communicators from the
    // cache. The reason is that if we do remove the communicator
    // from the cache, it is possible that a new collective operation
    // calls `ncclCommInitRank` to create a new communicator whereas
    // other ranks might have failed/timed out and didn't enter
    // `ncclCommInitRank`. As a result, when there is a failure on
    // a communicator the application receives an exception and its
    // their responsibility to destroy the process group and recreate
    // it to recover from errors.

    LOG(INFO) << "[Rank " << rank << "] Destroyed " << ncclComms.size()
              << "communicators on CUDA device " << devName;
  }
}

// Abort all communicators on this rank
void ProcessGroupNCCL::abort(c10::optional<std::string> abortReason) {
  std::lock_guard<std::mutex> lock(mutex_);
  abortCommsFromMap(devNCCLCommMap_, rank_, abortReason);
  abortCommsFromMap(inInitializationCommMap_, rank_, abortReason);
}

ProcessGroupNCCL::~ProcessGroupNCCL() {
  terminateProcessGroup_.store(true);

  workMetaListCV_.notify_one();
#ifdef ENABLE_NCCL_ERROR_CHECKING
  ncclCommWatchdogThread_.join();
#endif

  // Abort all NCCL Communicators on Process Group Destruction
  std::string abortReason = c10::str("Process Group destroyed on rank ", rank_);
  abort(abortReason);
}

void ProcessGroupNCCL::ncclCommWatchdog() {
  try {
    LOG(INFO) << "[Rank " << rank_ << "] NCCL watchdog thread started!";
    workCleanupLoop();
    LOG(INFO) << "[Rank " << rank_
              << "] NCCL watchdog thread terminated normally";
  } catch (std::exception& e) {
    // Append error message reported from workCleanupLoop
    const auto exitMsg = c10::str(
        "[Rank ",
        rank_,
        "] NCCL watchdog thread terminated with exception: ",
        e.what());
    LOG(ERROR) << exitMsg;
    watchDogException_ = std::make_exception_ptr(std::runtime_error(exitMsg));
    std::rethrow_exception(watchDogException_);
  } catch (...) {
    const auto exitMsg = c10::str(
        "[Rank ",
        rank_,
        "] NCCL watchdog thread terminated with exception: unknown");
    LOG(ERROR) << exitMsg;
    watchDogException_ = std::make_exception_ptr(std::runtime_error(exitMsg));
    std::rethrow_exception(watchDogException_);
  }
}

void ProcessGroupNCCL::logWorkStart(WorkNCCL& work) {
  if (work.startTraceUpdated_)
    return;

  if (terminateProcessGroup_.load() || storeError_)
    return;

  work.startTraceUpdated_ = true;
  storeError_ = !c10d::traceUpdate(
      store_, traceKeyStart_, work.seq_, opTypeToString(work.opType_));
}

void ProcessGroupNCCL::logWorkEnd(WorkNCCL& work) {
  if (terminateProcessGroup_.load() || storeError_)
    return;

  // In case the start of the work hasn't been logged
  if (!work.startTraceUpdated_) {
    logWorkStart(work);
  }

  storeError_ = !c10d::traceUpdate(
      store_, traceKeyEnd_, work.seq_, opTypeToString(work.opType_));
}

void ProcessGroupNCCL::workCleanupLoop() {
  bool done = false;
  while (!done || !terminateProcessGroup_.load()) {
    std::unique_lock<std::mutex> lock(workMetaListMutex_);
    // We busy-poll the work vector every kWatchdogThreadSleepMillis
    // milliseconds as long as the atomic is True.
    workMetaListCV_.wait_for(
        lock,
        std::chrono::milliseconds(kWatchdogThreadSleepMillis),
        [&]() -> bool { return terminateProcessGroup_.load(); });

    for (auto it = workMetaList_.begin(); it != workMetaList_.end();
         /* no increment*/) {
      auto& work = *it;
      work.checkAndSetException();
      bool timedOut = work.checkTimeout();

      // If work hits an exception (either an error or timeout)
      if (work.exception()) {
        if (SHOULD_CLEAN_UP(asyncErrorHandling_)) {
          // Abort work and corresponding communicators
          work.abort();
          // PG level abort, which would abort all other communicators on this
          // rank
          abort();
        }
        // Report desync state in case of timeout
        if (desyncDebug_ && timedOut) {
          try {
            auto desyncMsg = retrieveDesyncReport(store_, "NCCL", rank_, size_);
            LOG(ERROR) << desyncMsg;
          } catch (const std::exception& e) {
            LOG(ERROR) << "Failed to retrieve NCCL_DESYNC_DEBUG report. "
                       << " Please file an issue. Error: " << e.what();
          } catch (...) {
            LOG(ERROR)
                << "Failed to rerieve NCCL_DESYNC_DEBUG report with unknown error."
                << " Please file an issue.";
          }
        }
        // Throw exception
        work.handleException(asyncErrorHandling_);
      }

      // Work status logging for desync debug
      if (desyncDebug_) {
        if (work.isStarted()) {
          logWorkStart(work);
        }
        if (work.isCompleted()) {
          logWorkEnd(work);
        }
      }

      // Clean up completed work
      if (work.isCompleted()) {
        it = workMetaList_.erase(it);
      } else {
        // Increment the iterator if the current WorkNCCL object is not
        // completed.
        ++it;
      }
    }
    done = workMetaList_.empty();
  }
}

std::exception_ptr ProcessGroupNCCL::WorkNCCL::checkForNCCLErrors(
    const std::vector<std::shared_ptr<NCCLComm>>& ncclComms) const {
  return checkForNCCLErrorsInternal(ncclComms);
}

std::exception_ptr ProcessGroupNCCL::checkForNCCLErrors(
    const std::vector<std::shared_ptr<NCCLComm>>& ncclComms) {
  return checkForNCCLErrorsInternal(ncclComms);
}

std::exception_ptr ProcessGroupNCCL::checkForNCCLErrorsInternal(
    const std::vector<std::shared_ptr<NCCLComm>>& ncclComms) {
  for (const auto& ncclComm : ncclComms) {
    // Prioritize commFailureReason over checkForNcclError() result if
    // commFailureReason is set.
    auto commFailureReason = ncclComm->getNcclCommFailureReason();
    if (commFailureReason != c10::nullopt) {
      return std::make_exception_ptr(std::runtime_error(c10::str(
          "NCCL communicator encountered error set by ProcessGroupNCCL: ",
          *commFailureReason)));
    }
    ncclResult_t ncclAsyncErr = ncclComm->checkForNcclError();
    if (ncclAsyncErr != ncclSuccess) {
      return std::make_exception_ptr(std::runtime_error(
          "NCCL error: " + ncclGetErrorWithVersion(ncclAsyncErr) + "\n" +
          getNcclErrorDetailStr(ncclAsyncErr)));
    }
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
      TORCH_CHECK(vec.size() == NCCL_UNIQUE_ID_BYTES);
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
      TORCH_CHECK(
          false,
          exceptionMsg + e.what() +
              ". This may indicate a possible application crash on rank 0 or a network set up issue.");
    } catch (...) {
      TORCH_CHECK(
          false,
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

void ProcessGroupNCCL::destroyNCCLComms(const std::string& devNCCLCommMapKey) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (devNCCLCommMap_.find(devNCCLCommMapKey) == devNCCLCommMap_.end()) {
    TORCH_INTERNAL_ASSERT(
        false,
        "Expected to find key ",
        devNCCLCommMapKey,
        " in NCCL communicator map.");
  }
  std::vector<std::shared_ptr<NCCLComm>>& ncclComms =
      devNCCLCommMap_[devNCCLCommMapKey];
  // Loop through communicators and call ncclCommAbort.
  for (const auto& comm : ncclComms) {
    // ncclCommDestroy(comm->getNcclComm()) results in segfault when PG is being
    // destroyed, so using ncclCommAbort here.
    comm->ncclCommAbort();
  }
  // Remove communicators from the cache.
  devNCCLCommMap_.erase(devNCCLCommMapKey);
  // Clear used device indices.
  usedDeviceIdxs_.clear();
}

std::vector<std::shared_ptr<NCCLComm>>& ProcessGroupNCCL::getNCCLComm(
    const std::string& devicesKey,
    const std::vector<at::Device>& devices,
    OpType opType,
    int p2pRank,
    bool isSendRecvSelf) {
  // Sanity check
  if (devicesKey.empty()) {
    TORCH_CHECK(
        false,
        "Not able to create/get the NCCL Communicator since "
        "the GPU devices are not known");
  }

  for (auto& device : devices) {
    usedDeviceIdxs_.insert(device.index());
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (devNCCLCommMap_.find(devicesKey) != devNCCLCommMap_.end()) {
      // Reuse the cached communicator if there is one.
      return devNCCLCommMap_[devicesKey];
    }
  }

  // NCCL communicator not cached, create a new entry
  std::vector<std::shared_ptr<NCCLComm>> ncclComms;
  ncclComms.resize(devices.size());

  // Create the unique NCCL ID and broadcast it
  ncclUniqueId ncclID;

  // For batch_isend_irecv, ncclGroupStart() would be called upfront
  bool batchP2P = ncclActiveGroupCounter_ > 0;
  bool singleP2POp = isP2POp(opType, batchP2P);
  // For point-to-point communication, lower rank of the two will get unique id.
  if (rank_ == 0 || (singleP2POp && p2pRank == 0)) {
    C10D_NCCL_CHECK(ncclGetUniqueId(&ncclID), c10::nullopt);
  }

  // For point-to-point communication on the same process, don't need broadcast.
  if (!isSendRecvSelf) {
    // Broadcast so that each process can have a unique NCCL ID
    broadcastUniqueNCCLID(&ncclID, singleP2POp, devicesKey, p2pRank);
  }

  at::cuda::OptionalCUDAGuard gpuGuard;

  std::vector<at::cuda::CUDAStream> streamVal;
  streamVal.reserve(devices.size());

  // [Group Start/End Note] This is used to ensure that nccl communicator will
  // be created before communication primitives are called. Let's look at this
  // example: Using the batch_isend_irecv to send a tensor to a target process.
  // On the sender side, the corresponding underlying NCCL calls will look like
  //   ncclGroupStart() // This is in batch_isend_irecv
  //   ncclGroupStart() // This is [Note 1]
  //   ncclCommInitRank() // Inside NCCLComm::create
  //   ncclSend()
  //   ncclGroupEnd() // This is [Note 2]
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
    C10D_NCCL_CHECK(ncclGroupEnd(), c10::nullopt);
  }

  // [Note 1] Create the NCCL communicators for each GPU
  C10D_NCCL_CHECK(ncclGroupStart(), c10::nullopt);

  for (const auto i : c10::irange(devices.size())) {
    // GPU world size and GPU rank
    int numRanks, rank;

    if (!singleP2POp) {
      // Collective, all-to-all, or batch P2P
      numRanks = getSize() * devices.size();
      rank = getRank() * devices.size() + i;
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
    // Get the device index
    int deviceIndex = devices[i].index();

    gpuGuard.set_index(deviceIndex);
#ifdef NCCL_HAS_COMM_NONBLOCKING
    ncclComms[i] = NCCLComm::create(numRanks, rank, ncclID, options_->config);
#else
    ncclComms[i] = NCCLComm::create(numRanks, rank, ncclID);
#endif

    // Creates the NCCL streams
    streamVal.push_back(
        at::cuda::getStreamFromPool(options_->is_high_priority_stream));
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    inInitializationCommMap_.emplace(devicesKey, ncclComms);
  }

  // [Note 2 ]
#ifndef NCCL_HAS_COMM_NONBLOCKING
  C10D_NCCL_CHECK(ncclGroupEnd(), c10::nullopt);
#else
  if (!nccl_use_nonblocking()) {
    C10D_NCCL_CHECK(ncclGroupEnd(), c10::nullopt);
  } else {
    C10D_NCCL_CHECK_TIMEOUT_GROUPEND(ncclGroupEnd(), ncclComms, c10::nullopt);
  }
#endif

  // At this point NCCL should have been initialized, hence we can accurately
  // get the env value even if NCCL sets it by reading from nccl.conf file
  if (getRank() == 0) {
    LOG(INFO) << "NCCL_DEBUG: " << parse_env("NCCL_DEBUG");
  }

  // See [Group Start/End Note]
  for (const auto i : c10::irange(ncclActiveGroupCounter_)) {
    (void)i;
    C10D_NCCL_CHECK(ncclGroupStart(), c10::nullopt);
  }

  ncclStreams_.emplace(devicesKey, std::move(streamVal));

  // Note: these events are created with the (default) cudaEventDisableTiming
  // flag This flag provides the best performance when used with
  // cudaStreamWaitEvent() and cudaEventQuery(). Since we here don't measure the
  // performance using cudaEvent, this should be set.
  ncclEvents_.emplace(
      std::piecewise_construct,
      std::make_tuple(devicesKey),
      std::make_tuple(devices.size()));

  // Hold the lock before modifying the cache.
  std::lock_guard<std::mutex> lock(mutex_);

  // Record the communicators based on ncclUniqueId.
  ncclIdToCommMap_.emplace(buildNcclUniqueIdStr(ncclID), ncclComms);

  // Move the NCCL resource to cache
  auto it = inInitializationCommMap_.find(devicesKey);
  // A previous thread could've already removed devicesKey from
  // inInitializationCommMap_ and added it to devNCCLCommMap_
  if (it != inInitializationCommMap_.end()) {
    devNCCLCommMap_.emplace(devicesKey, std::move(it->second));
    inInitializationCommMap_.erase(devicesKey);
  }

  it = devNCCLCommMap_.find(devicesKey);
  TORCH_INTERNAL_ASSERT(
      it != devNCCLCommMap_.end(), "Communicators not populated in cache!");
  return it->second;
}

namespace {

// Check validity of tensor
void check_gpu_single_tensor(const at::Tensor& tensor) {
  if (!tensor.is_cuda() || tensor.is_sparse()) {
    TORCH_CHECK(false, "Tensors must be CUDA and dense");
  }
  if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
    TORCH_CHECK(false, "Tensors must be contiguous");
  }
}

// Checks that all `tensors' have the same type and shape and reside on distinct
// GPUs.
// TODO: test_c10d_nccl.py should consider adding tests for the error conditions
// here, ie, that deliberately pass invalid tensors and check the right
// exception is thrown.
void check_gpu_tensors_different_devices(
    const std::vector<at::Tensor>& tensors) {
  if (tensors.size() == 0) {
    TORCH_CHECK(false, "Tensor list must be nonempty");
  }
  if (tensors.size() > static_cast<size_t>(at::cuda::getNumGPUs())) {
    TORCH_CHECK(
        false,
        "Tensor list mustn't be larger than the number of available GPUs");
  }

  const auto& first = tensors.front();

  // Set for ensuring that tensors are on separate devices.
  std::unordered_set<decltype(first.get_device())> usedDevices;
  usedDevices.reserve(tensors.size());

  for (const auto& t : tensors) {
    if (!t.is_cuda() || t.is_sparse()) {
      TORCH_CHECK(false, "Tensors must be CUDA and dense");
    }
    if (t.scalar_type() != first.scalar_type()) {
      TORCH_CHECK(false, "Tensors must have identical type");
    }
    if (t.sizes() != first.sizes()) {
      TORCH_CHECK(false, "Tensors must have identical size");
    }
    if (t.strides() != first.strides()) {
      TORCH_CHECK(false, "Tensors must have identical strides");
    }
    if (!t.is_contiguous(t.suggest_memory_format())) {
      TORCH_CHECK(false, "Tensors must be contiguous");
    }
    const auto inserted = usedDevices.insert(t.get_device()).second;
    if (!inserted) {
      TORCH_CHECK(false, "Tensors must be on distinct GPU devices");
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
  if (tensors.size() == 0) {
    TORCH_CHECK(false, "Tensor list must be nonempty");
  }

  const auto& first = tensors.front();

  int64_t total_numel = 0;
  for (const auto& t : tensors) {
    if (!t.is_cuda() || t.is_sparse()) {
      TORCH_CHECK(false, "Tensors must be CUDA and dense");
    }
    if (t.scalar_type() != first.scalar_type()) {
      TORCH_CHECK(false, "Tensors must have identical type");
    }
    if (!t.is_non_overlapping_and_dense()) {
      TORCH_CHECK(false, "Tensors must be non-overlapping and dense");
    }
    // If we're in this function, the user called a _coalesced collective
    // on a set of tensors with potentially different sizes and strides.
    // Therefore, we don't check for matching sizes and strides,
    // but we do double-check tensors are on the same device.
    TORCH_CHECK(
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

// Flatten each list in `tensor_lists' for a gather or scatter operation, and
// ensure compatibility with the corresponding tensor in `other'.
std::vector<at::Tensor> flatten_for_scatter_gather(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    std::vector<at::Tensor>& other,
    size_t world_size) {
  if (tensor_lists.size() != other.size()) {
    TORCH_CHECK(
        false,
        "Tensor list operands to scatter/gather must have the same length");
  }
  const auto num_devices = tensor_lists.size();

  std::vector<at::Tensor> flattened;
  flattened.resize(num_devices);

  for (const auto i : c10::irange(size_t{}, num_devices)) {
    if (tensor_lists[i].size() != world_size * num_devices) {
      TORCH_CHECK(
          false,
          "Tensor list input to scatter/gather must match number of collective"
          " participants");
    }

    // Only check device match for the first tensor in the list; the call to
    // newLikeFlat() below will check the rest.
    if (tensor_lists[i].front().get_device() != other[i].get_device()) {
      TORCH_CHECK(
          false,
          "Corresponding input/output tensors to scatter/gather must all reside"
          " on the same device");
    }

    for (const auto& t : tensor_lists[i]) {
      if (t.numel() != other[i].numel()) {
        TORCH_CHECK(
            false,
            "All tensor operands to scatter/gather must have the same number of elements");
      }
    }
    // Flatten the tensors (from all ranks) into a single big tensor.
    flattened[i] = newLikeFlat(tensor_lists, i);
  }
  return flattened;
}

} // namespace

c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL> ProcessGroupNCCL::initWork(
    std::vector<at::Device> devices,
    int rank,
    OpType opType,
    const char* profilingTitle,
    const c10::optional<std::vector<at::Tensor>>& inputs) {
  return c10::make_intrusive<ProcessGroupNCCL::WorkNCCL>(
      devices, rank, opType, seq_, profilingTitle, inputs, desyncDebug_);
}

std::vector<at::Tensor> ProcessGroupNCCL::WorkNCCL::result() {
  return *outputs_;
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupNCCL::WorkNCCL::
    getFuture() {
  return future_;
}

void ProcessGroupNCCL::workEnqueue(
    c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL> work) {
  if (!terminateProcessGroup_.load()) {
    std::lock_guard<std::mutex> lock(workMetaListMutex_);
    // Avoid view tensors to be processed in cleanup thread.
    // View tensors' destruction invokes autograd_meta, which
    // needs to be destructed in user thread. Otherwise will
    // get deadlock. Here we enqueue work without outputs_.
    workMetaList_.emplace_back(*work);
  }
}

ProcessGroupNCCL::Options::Options(bool is_high_priority_stream)
    : Backend::Options(NCCL_BACKEND_NAME),
      is_high_priority_stream(is_high_priority_stream) {}

static constexpr int CoalActive = 0x01, CoalColl = 0x02, CoalP2P = 0x04;

void ProcessGroupNCCL::startCoalescing() {
  coalescedDevices_.clear();
  coalescedComms_.clear();
  coalescing_state_ |= CoalActive;
  groupStart();
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::endCoalescing() {
  if (!nccl_use_nonblocking() ||
      coalescedComms_.size() == 0) { // There is no actual work being coalesced
    groupEnd();
  } else {
    // `coalescedComms_` should have same set of comms across collectives
    auto comms = coalescedComms_[0];
    groupEndNonblocking(comms);
  }

  coalescing_state_ = 0;

  if (coalescedDevices_.size() == 0) {
    // There is no actual work being coalesced
    return nullptr;
  }

  // `coalescedDevices_` should have same set of devices across collectives
  auto devices = coalescedDevices_[0];

  // Create Work object
  auto work = initWork(
      devices, rank_, OpType::COALESCED, "nccl:coalesced", c10::nullopt);

  // Record stream event
  // `getKeyFromDevices` is how we get keys for both collectives and batch P2P
  const auto key = getKeyFromDevices(devices);
  auto& ncclStreams = ncclStreams_[key];
  for (const auto i : c10::irange(devices.size())) {
    auto& devEvent = (*work->ncclEndEvents_)[i];
    devEvent.record(ncclStreams[i]);
  }

  // Set appropriate work parameters.
  work->blockingWait_ = blockingWait_;
  work->avoidRecordStreams_ = avoidRecordStreams_;
  work->opTimeout_ = options_->timeout;
  work->store_ = store_;

  c10::cuda::CaptureStatus capture_status =
      c10::cuda::currentStreamCaptureStatusMayInitCtx();

  if ((coalescing_state_ & CoalColl) &&
      capture_status == c10::cuda::CaptureStatus::None) {
    workEnqueue(work);
    // TODO: it seems we never enqueue work for single send/recv or batch P2P,
    // see the `pointToPoint` function. This should be fixed. Otherwise, we risk
    // not being able to abort hanged P2P ops.
  }

  return work;
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupNCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    OpType opType,
    const char* profilingTitle) {
  c10::cuda::CaptureStatus capture_status =
      c10::cuda::currentStreamCaptureStatusMayInitCtx();
  errorIfCapturingNonCapturableNCCL(capture_status);

  // Bump collective counter
  seq_++;

  // Currently, the API permits two scenarios where inputs.size() and
  // outputs.size() are > 0.
  // 1. If the call was a _coalesced call, all inputs must be on the same
  // device.
  //    The group of nccl calls applies the collective separately to each input,
  //    but the group as a whole should be efficient, and might even execute as
  //    a single fused kernel.
  // 2. If the call was a _multigpu call, all inputs must be on different
  // devices.
  //    The nccl group applies the collective across them (eg, if the collective
  //    is an allreduce, the output on each device contains contributions summed
  //    across `inputs' tensors).
  const auto devices = getDeviceList(inputs);
  const bool inputs_same_dev = (devices.size() == 1);
  const auto key = getKeyFromDevices(devices);
  auto& ncclComms = getNCCLComm(key, devices, opType);

  if (coalescing_state_ & CoalActive) {
    coalescing_state_ |= CoalColl;
    coalescedDevices_.push_back(devices);
    coalescedComms_.push_back(ncclComms);
  }

  // Used many times below, so we stash the unordered_map lookup
  auto& ncclStreams = ncclStreams_[key];

  // First let NCCL streams wait for input tensors allocation streams
  syncStreams(devices, ncclEvents_[key], ncclStreams);

  // Work itself will create the CUDA events on all GPUs of tensors
  bool can_profile = outputs.size() == 1;
  auto work = initWork(
      devices,
      rank_,
      opType,
      can_profile ? profilingTitle : nullptr,
      can_profile ? c10::optional<std::vector<at::Tensor>>(inputs)
                  : c10::nullopt);

  // Store references to outputs to be used by WorkNCCL::result and operator<<.
  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);

  if (avoidRecordStreams_) {
    work->stashed_for_allocator_safety_ =
        std::make_shared<std::vector<at::Tensor>>(inputs);
  }

  at::cuda::OptionalCUDAGuard gpuGuard;

  // Start event should only be recorded before the ncclGroupStart()
  if (desyncDebug_) {
    for (const auto i : c10::irange(devices.size())) {
      at::cuda::CUDAStream& ncclStream = ncclStreams[i];
      (*work->ncclStartEvents_)[i].record(ncclStream);
    }
  }

  pre(ncclStreams, work);

  std::vector<void*> comms_;
  if (nccl_use_nonblocking()) {
    for (const auto i : c10::irange(inputs.size())) {
      decltype(i) stream_comm_i = (inputs_same_dev ? 0 : i);
      comms_.push_back((void*)ncclComms[stream_comm_i]->getNcclComm());
    }
  }

  {
    torch::cuda::nccl::AutoNcclGroup nccl_group_guard(
        comms_, nccl_use_nonblocking());
    for (const auto i : c10::irange(inputs.size())) {
      if (!inputs_same_dev || (inputs_same_dev && i == 0)) {
        gpuGuard.set_index(devices[i].index());
      }
      decltype(i) stream_comm_i = (inputs_same_dev ? 0 : i);
      auto& ncclStream = ncclStreams[stream_comm_i];
      auto& ncclComm = ncclComms[stream_comm_i];
      // Both `inputs' and `outputs' are created on a worker stream and used in
      // different ncclStreams.  Hence, both must record the ncclStream to
      // prevent being freed before the collective finishes.
      //
      // We only record `inputs' here, and leave recording `outputs' to `fn' for
      // operations where `inputs' and `outputs' are not the same.
      //
      // See [Sync Streams].
      if (!avoidRecordStreams_) {
        if (!inputs[i].is_sparse()) {
          c10::cuda::CUDACachingAllocator::recordStream(
              inputs[i].storage().data_ptr(), ncclStream);
        } else {
          // for sparse input case record streams on both index and value
          // tensors
          c10::cuda::CUDACachingAllocator::recordStream(
              inputs[i].values().storage().data_ptr(), ncclStream);
          c10::cuda::CUDACachingAllocator::recordStream(
              inputs[i].indices().storage().data_ptr(), ncclStream);
        }
      }
#ifndef NCCL_HAS_COMM_NONBLOCKING
      C10D_NCCL_CHECK(
          fn(inputs[i], outputs[i], ncclComm->getNcclComm(), ncclStream),
          ncclComm->getNcclCommFailureReason());
#else
      C10D_NCCL_CHECK_TIMEOUT(
          fn(inputs[i], outputs[i], ncclComm->getNcclComm(), ncclStream),
          ncclComm->getNcclComm(),
          ncclComm->getNcclCommFailureReason());
#endif
    }
  }
  post(ncclStreams, work);

  // End event should only be recorded after the ncclGroupEnd()
  for (const auto i : c10::irange(devices.size())) {
    at::cuda::CUDAStream& ncclStream = ncclStreams[i];
    if (!coalescing_state_) {
      (*work->ncclEndEvents_)[i].record(ncclStream);
    }
    work->ncclComms_[i] = ncclComms[i];
  }

  {
    c10::cuda::CUDAMultiStreamGuard streamGuard(ncclStreams);
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);

    // Add a callback that runs profiling end callbacks. wrapCallback() in CUDA
    // future blocks the stream this callback runs on the corresponding
    // ncclEndEvents_ ensuring appropriate synchronization.
    if (work->recordFunctionEndCallback_) {
      work->future_->addCallback([work](at::ivalue::Future& /* unused */) {
        work->recordFunctionEndCallback_();
      });
    }
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }

  // Set appropriate work parameters.
  work->blockingWait_ = blockingWait_;
  work->avoidRecordStreams_ = avoidRecordStreams_;
  work->opTimeout_ = options_->timeout;
  work->store_ = store_;
  // Record size info for debug. We only record the size on the first device as
  // multi-device per process is deprecated
  work->numelIn_ = inputs[0].numel();
  work->numelOut_ = outputs[0].numel();

  if (!coalescing_state_ && capture_status == c10::cuda::CaptureStatus::None) {
    workEnqueue(work);
  }

  return work;
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupNCCL::pointToPoint(
    std::vector<at::Tensor>& tensors,
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
  // Therefore, we warn and fall back to the typical recordStream logic:
  TORCH_WARN_ONCE(
      avoidRecordStreams_,
      "NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point "
      "collectives.");

  const auto devices = getDeviceList(tensors);
  std::string key;
  int p2pRank = 0, p2pTargetRank = 0;
  bool isSendRecvSelf = false;
  // For batch_isend_irecv, ncclGroupStart() would be called upfront
  bool batchP2P = ncclActiveGroupCounter_ > 0;
  if (batchP2P) {
    // For batch P2P, we need to treat it like a collective when selecting
    // communicator, because other ranks can call into this batch other than my
    // rank and my peer
    key = getKeyFromDevices(devices);
    p2pRank = rank_;
    p2pTargetRank = peer;
  } else {
    // For single P2P, preserve the old two-rank behavior (to avoid perf diff)
    key = getKeySendRecv(rank_, peer);
    p2pRank = rank_ <= peer ? 0 : 1;
    isSendRecvSelf = rank_ == peer;
    p2pTargetRank = isSendRecvSelf ? 0 : 1 - p2pRank;
  }
  auto& ncclComms = getNCCLComm(key, devices, opType, p2pRank, isSendRecvSelf);

  if (coalescing_state_ & CoalActive) {
    coalescing_state_ |= CoalP2P;
    coalescedDevices_.push_back(devices);
    coalescedComms_.push_back(ncclComms);
  }

  // First let NCCL streams wait for input tensors allocation streams
  syncStreams(devices, ncclEvents_[key], ncclStreams_[key]);

  // Work itself will create the CUDA events on all GPUs of tensors
  bool can_profile = tensors.size() == 1;
  auto work = initWork(
      devices,
      rank_,
      opType,
      can_profile ? profilingTitle : nullptr,
      can_profile ? c10::optional<std::vector<at::Tensor>>(tensors)
                  : c10::nullopt);

  // Store references to outputs to be used by WorkNCCL::result and operator<<.
  // Note that these outputs are only valid for recv(), as send() does not
  // modify the inputs but we still create these outputs for use cases such as
  // profiling.
  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(tensors);

  at::cuda::OptionalCUDAGuard gpuGuard;

  // Start event should only be recorded before the ncclGroupStart()
  if (desyncDebug_) {
    for (const auto i : c10::irange(tensors.size())) {
      at::cuda::CUDAStream& ncclStream = ncclStreams_[key][i];
      (*work->ncclStartEvents_)[i].record(ncclStream);
    }
  }

  pre(ncclStreams_[key], work);

  for (const auto i : c10::irange(tensors.size())) {
    gpuGuard.set_index(devices[i].index());
    at::cuda::CUDAStream& ncclStream = ncclStreams_[key][i];

    // Both send tensor and recv tensor are created on a worker stream and used
    // in different ncclStreams.  Hence, both must record the ncclStream to
    // prevent being freed before the collective finishes.
    //
    // See [Sync Streams].
    c10::cuda::CUDACachingAllocator::recordStream(
        tensors[i].storage().data_ptr(), ncclStream);
  }

  std::vector<void*> comms_;
  if (nccl_use_nonblocking()) {
    for (const auto i : c10::irange(tensors.size())) {
      comms_.push_back((void*)ncclComms[i]->getNcclComm());
    }
  }
  {
    torch::cuda::nccl::AutoNcclGroup nccl_group_guard(
        comms_, nccl_use_nonblocking());
    for (const auto i : c10::irange(tensors.size())) {
      gpuGuard.set_index(devices[i].index());
      at::cuda::CUDAStream& ncclStream = ncclStreams_[key][i];
#ifndef NCCL_HAS_COMM_NONBLOCKING
      C10D_NCCL_CHECK(
          fn(tensors[i],
             ncclComms[i]->getNcclComm(),
             ncclStream,
             p2pTargetRank),
          ncclComms[i]->getNcclCommFailureReason());
#else
      C10D_NCCL_CHECK_TIMEOUT(
          fn(tensors[i],
             ncclComms[i]->getNcclComm(),
             ncclStream,
             p2pTargetRank),
          ncclComms[i]->getNcclComm(),
          ncclComms[i]->getNcclCommFailureReason());
#endif
    }
  }

  post(ncclStreams_[key]);

  // End event should only be recorded after the ncclGroupEnd()
  for (const auto i : c10::irange(tensors.size())) {
    at::cuda::CUDAStream& ncclStream = ncclStreams_[key][i];
    if (!coalescing_state_) {
      (*work->ncclEndEvents_)[i].record(ncclStream);
    }
    work->ncclComms_[i] = ncclComms[i];
    work->blockingWait_ = blockingWait_;
    work->opTimeout_ = options_->timeout;
    work->store_ = store_;
  }

  // Record size info for debug. We only record the size on the first device as
  // multi-device per process is deprecated
  work->numelIn_ = work->numelOut_ = tensors[0].numel();

  // Future only needs to be created and marked completed with outputs for
  // recv(), but still create future for use cases such as profiling even for
  // send().
  {
    c10::cuda::CUDAMultiStreamGuard streamGuard(ncclStreams_[key]);
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }

  // Add a callback that runs profiling end callbacks. wrapCallback() in CUDA
  // future blocks the stream this callback runs on the corresponding
  // ncclEndEvents_ ensuring appropriate synchronization.
  if (work->recordFunctionEndCallback_) {
    work->future_->addCallback([work](at::ivalue::Future& /* unused */) {
      work->recordFunctionEndCallback_();
    });
  }

  return work;
}

template <typename Fn>
c10::intrusive_ptr<Work> ProcessGroupNCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    OpType opType,
    const char* profilingTitle) {
  return collective(
      inputs,
      outputs,
      fn,
      [](std::vector<at::cuda::CUDAStream>&,
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
      [](std::vector<at::cuda::CUDAStream>&,
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
      opType,
      profilingTitle);
}

template <typename Fn>
c10::intrusive_ptr<Work> ProcessGroupNCCL::pointToPoint(
    std::vector<at::Tensor>& tensor,
    Fn fn,
    int peer,
    OpType opType,
    const char* profilingTitle) {
  return pointToPoint(
      tensor,
      fn,
      peer,
      opType,
      [](std::vector<at::cuda::CUDAStream>&,
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
      [](std::vector<at::cuda::CUDAStream>&) {},
      profilingTitle);
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allreduce_sparse(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
#ifdef IS_NCCL_EXP
  std::vector<at::Tensor> outputTensors(tensors.size());
  for (std::vector<at::Tensor>::size_type i = 0; i < tensors.size(); i++) {
    tensors[i] = tensors[i].coalesce();
    outputTensors[i] = torch::zeros(
        tensors[i].sizes(), tensors[i].options().layout(torch::kStrided));
  }
  int dev_in_group = 0;
  auto work = collective(
      tensors,
      outputTensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        auto ncclDataType = getNcclDataType(input.scalar_type());
        auto ncclReduceOp = getNcclReduceOp(
            opts.reduceOp, input, ncclDataType, comm, dev_in_group++);

        size_t num_elements = output.numel();
        auto indices = input.indices();
        auto sizes = input.sizes();
        int colSize = sizes[1];
        auto rows = indices[0];
        size_t blockCount = rows.sizes()[0];
        auto recvIndices = indices[0] * colSize;

        // prevent output and recvIndices from being freed
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
      [](std::vector<at::cuda::CUDAStream>& ncclStreams,
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
      [&](std::vector<at::cuda::CUDAStream>& ncclStreams,
          c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
        // Convert output tensors to sparse and back into tensors.
        for (const auto i : c10::irange(outputTensors.size())) {
          at::cuda::CUDAStreamGuard guard(ncclStreams[i]);
          if (opts.sparseIndices.has_value()) {
            tensors[i] = at::sparse_coo_tensor(
                opts.sparseIndices.value(),
                outputTensors[i],
                tensors[i].sizes());
          } else {
            tensors[i] = outputTensors[i].to_sparse();
          }
        }
      },
      OpType::_ALLREDUCE_SPARSE,
      "nccl:all_reduce_sparse");
  return work;
#else
  // If the nccl branch is not "exp" then we just error
  TORCH_CHECK(
      false,
      "allreduce_sparse is only available in the NCCL experimental branch.");
#endif
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allreduce_impl(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  int dev_in_group = 0;
  return collective(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        auto ncclDataType = getNcclDataType(input.scalar_type());
        auto ncclReduceOp = getNcclReduceOp(
            opts.reduceOp, input, ncclDataType, comm, dev_in_group++);
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
      "nccl:all_reduce");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  check_gpu_tensors_different_devices(tensors);

  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      reinterpret_cast<std::intptr_t>(this), // process group ptr
      tensors, // inputTensors
      tensors, // outputTensors
      rank_, // rank
      "allreduce", // colName
      tensor.numel(), // inSize
      tensor.numel(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>()); // outSplitSizes

  // avoidRecordStreams_ note: collective() will stash tensors.
  return allreduce_impl(tensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  auto total_numel = check_gpu_tensors_same_device(tensors);

  // @lint-ignore CLANGTIDY
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      reinterpret_cast<std::intptr_t>(this), // process group ptr
      tensors, // inputTensors
      tensors, // outputTensors
      rank_, // rank
      "allreduce_coalesced", // colName
      total_numel, // inSize
      total_numel, // outSize
      tensors[0].scalar_type(), // dType
      // I'm not sure what in,outSplitSizes mean here.
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>()); // outSplitSizes

  // avoidRecordStreams_ note: collective() will stash tensors.
  return allreduce_impl(tensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  check_gpu_tensors_different_devices(tensors);

  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();

  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      reinterpret_cast<std::intptr_t>(this), // process group ptr
      tensors, // inputTensors
      tensors, // outputTensors
      rank_, // rank
      "broadcast", // colName
      tensor.numel(), // inSize
      tensor.numel(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>()); // outSplitSizes

  // avoidRecordStreams_ note: collective() will stash tensors.
  return collective(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        const auto root = opts.rootRank * tensors.size() + opts.rootTensor;
        return ncclBcast(
            input.data_ptr(),
            input.numel(),
            getNcclDataType(input.scalar_type()),
            root,
            comm,
            stream.stream());
      },
      OpType::BROADCAST,
      "nccl:broadcast");
}

// _broadcast_oop adds an out-of-place broadcast in PGNCCL
// Custom collectives may be implemented by coalescing broadcast operations
// One use-case is implementing a vector all_gather (all_gather_v)
// where unevenly sized inputs are gathered among participating ranks
// Since all_gather provides an out-of-place API, an all_gather_v
// semantic implemented inside pg_nccl.all_gather also needs to support
// out-of-place, for which an out-of-place broadcast is required to be added
c10::intrusive_ptr<Work> ProcessGroupNCCL::_broadcast_oop(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const BroadcastOptions& opts) {
  check_gpu_tensors_different_devices(outputTensors);
  check_gpu_tensors_different_devices(inputTensors);

  // @lint-ignore CLANGTIDY
  auto tensor = outputTensors.back();
  // @lint-ignore CLANGTIDY
  auto in_tensor = inputTensors.back();
  if (tensor.numel() != in_tensor.numel()) {
    TORCH_CHECK(
        false,
        "Tensor input and output of _broadcast_oop must have the same number of elements ");
  }
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() +
          1), // seq + 1 to match collective increment.
      reinterpret_cast<std::intptr_t>(this), // process group ptr
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "_broadcast_oop", // colName
      tensor.numel(), // inSize
      tensor.numel(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>()); // outSplitSizes

  return collective(
      inputTensors,
      outputTensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        const auto root = opts.rootRank * inputTensors.size() + opts.rootTensor;
        return ncclBroadcast(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            getNcclDataType(input.scalar_type()),
            root,
            comm,
            stream.stream());
      },
      OpType::BROADCAST,
      "nccl:_broadcast_oop");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  check_gpu_tensors_different_devices(tensors);
  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      reinterpret_cast<std::intptr_t>(this),
      tensors, // inputTensors
      tensors, // outputTensors
      rank_, // rank
      "reduce", // colName
      tensor.numel(), // inSize
      tensor.numel(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>()); // outSplitSizes

  int dev_in_group = 0;
  // avoidRecordStreams_ note: collective() will stash tensors.
  return collective(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        const auto root = opts.rootRank * tensors.size() + opts.rootTensor;
        auto ncclDataType = getNcclDataType(input.scalar_type());
        auto ncclReduceOp = getNcclReduceOp(
            opts.reduceOp, input, ncclDataType, comm, dev_in_group++);
        return ncclReduce(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            ncclDataType,
            ncclReduceOp,
            root,
            comm,
            stream.stream());
      },
      OpType::REDUCE,
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
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const ReduceOptions& opts) {
  check_gpu_tensors_different_devices(outputTensors);
  check_gpu_tensors_different_devices(inputTensors);
  // @lint-ignore CLANGTIDY
  auto tensor = outputTensors.back();
  // @lint-ignore CLANGTIDY
  auto in_tensor = inputTensors.back();
  if (tensor.numel() != in_tensor.numel()) {
    TORCH_CHECK(
        false,
        "Tensor input and output of _reduce_oop must have the same number of elements ");
  }
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      reinterpret_cast<std::intptr_t>(this), // process group ptr
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "_reduce_oop", // colName
      tensor.numel(), // inSize
      tensor.numel(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>()); // outSplitSizes

  int dev_in_group{0};
  return collective(
      inputTensors,
      outputTensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        const auto root = opts.rootRank * inputTensors.size() + opts.rootTensor;
        const auto ncclDataType = getNcclDataType(input.scalar_type());
        const auto ncclReduceOp = getNcclReduceOp(
            opts.reduceOp, input, ncclDataType, comm, dev_in_group++);
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
      "nccl:_reduce_oop");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  check_gpu_tensors_different_devices(inputTensors);
  // @lint-ignore CLANGTIDY
  bool same_size = check_same_size(outputTensors.back());

  if (same_size) {
    auto outputFlattened =
        flatten_for_scatter_gather(outputTensors, inputTensors, size_);
    check_gpu_tensors_different_devices(outputFlattened);

    // @lint-ignore CLANGTIDY
    auto tensor = inputTensors.back();
    RECORD_PARAM_COMMS_DATA(
        static_cast<int>(
            this->getSequenceNumberForGroup() +
            1), // seq + 1 to match collective
        reinterpret_cast<std::intptr_t>(this), // process group ptr
        inputTensors, // inputTensors
        outputTensors, // outputTensors
        rank_, // rank
        "all_gather", // colName
        tensor.numel(), // inSize
        tensor.numel() * // outSize
            this->getSize(), // dType
        tensor.scalar_type(),
        std::vector<int64_t>(), // inSplitSizes
        std::vector<int64_t>()); // outSplitSize

    return collective(
        inputTensors,
        outputFlattened,
        [&](at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream) {
          if (!avoidRecordStreams_) {
            c10::cuda::CUDACachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
          return ncclAllGather(
              input.data_ptr(),
              output.data_ptr(),
              input.numel(),
              getNcclDataType(input.scalar_type()),
              comm,
              stream.stream());
        },
        [](std::vector<at::cuda::CUDAStream>& ncclStreams,
           c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
          // avoidRecordStreams_ note: We actually don't need to stash anything
          // here.
          //  - inputTensors is stashed onto work->stashed_for_allocator_safety_
          //    in collective().
          //  - outputFlattened is stashed onto work->outputs_ in collective().
          //  - User-facing outputTensors should be held by the user until after
          //    waiting on work_, or the call makes no sense.
          // So all participating tensors are accounted for, and won't be
          // released back to their allocation streams until after work_ is
          // waited on.
        },
        [&](std::vector<at::cuda::CUDAStream>& ncclStreams,
            c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
          // Copy the flattened output tensors to the outputs.
          for (const auto i : c10::irange(outputTensors.size())) {
            at::cuda::CUDAStreamGuard guard(ncclStreams[i]);
            for (const auto j : c10::irange(outputTensors[0].size())) {
              // See [Sync Streams].
              if (!avoidRecordStreams_) {
                c10::cuda::CUDACachingAllocator::recordStream(
                    outputTensors[i][j].storage().data_ptr(), ncclStreams[i]);
              }
              outputTensors[i][j].copy_(outputFlattened[i][j], true);
            }
          }
        },
        OpType::ALLGATHER,
        "nccl:all_gather");
  } else {
    const auto num_devices = outputTensors.size();
    const auto num_reduces = outputTensors[0].size();
    std::vector<c10::intrusive_ptr<Work>> works;
    startCoalescing();
    for (const auto i : c10::irange(num_reduces)) {
      std::vector<at::Tensor> inputs_multi_dev(num_devices);
      std::vector<at::Tensor> outputs_multi_dev(num_devices);
      for (const auto j : c10::irange(num_devices)) {
        // @lint-ignore CLANGTIDY
        outputs_multi_dev[j] = outputTensors[j][i];
        inputs_multi_dev[j] =
            // @lint-ignore CLANGTIDY
            i == (rank_ * num_devices + j) ? inputTensors[j]
                                           : outputs_multi_dev[j];
      }
      auto broadcastOpts = BroadcastOptions{
          static_cast<int64_t>(i / num_devices),
          static_cast<int64_t>(i % num_devices),
          opts.timeout};
      auto work =
          _broadcast_oop(outputs_multi_dev, inputs_multi_dev, broadcastOpts);
      works.push_back(work);
    }
    auto work = endCoalescing();
    return work;
  }
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllgatherOptions& /* unused */) {
  TORCH_CHECK(false, "ProcessGroupNCCL does not support allgather_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& opts) {
  return collective(
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
      "nccl:all_gather_into_tensor_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  check_gpu_tensors_different_devices(outputTensors);
  // @lint-ignore CLANGTIDY
  bool same_size = check_same_size(inputTensors.back());

  if (same_size) {
    // @lint-ignore CLANGTIDY
    auto tensor = outputTensors.back();

    int dev_in_group{0};
    auto inputFlattened =
        flatten_for_scatter_gather(inputTensors, outputTensors, size_);
    check_gpu_tensors_different_devices(inputFlattened);

    RECORD_PARAM_COMMS_DATA(
        static_cast<int>(
            this->getSequenceNumberForGroup() +
            1), // seq + 1 to match collective
        reinterpret_cast<std::intptr_t>(this), // process group ptr
        inputTensors, // inputTensors
        outputTensors, // outputTensors
        rank_, // rank
        "reduce_scatter", // colName
        tensor.numel() * this->getSize(), // inSize
        tensor.numel(), // outSize
        tensor.scalar_type(), // dType
        std::vector<int64_t>(), // inSplitSizes
        std::vector<int64_t>()); // outSplitSizes

    return collective(
        inputFlattened,
        outputTensors,
        [&](at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream) {
          if (!avoidRecordStreams_) {
            c10::cuda::CUDACachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
          const auto ncclDataType = getNcclDataType(input.scalar_type());
          const auto ncclReduceOp = getNcclReduceOp(
              opts.reduceOp, input, ncclDataType, comm, dev_in_group++);
          return ncclReduceScatter(
              input.data_ptr(),
              output.data_ptr(),
              output.numel(),
              ncclDataType,
              ncclReduceOp,
              comm,
              stream.stream());
        },
        [&](std::vector<at::cuda::CUDAStream>& ncclStreams,
            c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
          if (avoidRecordStreams_) {
            // We only need to stash inputTensors.
            //  - inputFlattened is stashed onto
            //  work->stashed_for_allocator_safety_
            //    in collective().
            //  - User-facing outputTensors is stashed onto work->outputs_ in
            //  collective(),
            //    and should also be held by the user until after waiting on
            //    work_.
            auto& v = work->stashed_for_allocator_safety_;
            for (const auto i : c10::irange(inputTensors.size())) {
              v->insert(
                  v->end(), inputTensors[i].begin(), inputTensors[i].end());
            }
          }

          // Copy the input tensors to the flattened inputs.
          for (const auto i : c10::irange(inputTensors.size())) {
            at::cuda::CUDAStreamGuard guard(ncclStreams[i]);
            for (const auto j : c10::irange(inputTensors[0].size())) {
              // See [Sync Streams].
              if (!avoidRecordStreams_) {
                c10::cuda::CUDACachingAllocator::recordStream(
                    inputTensors[i][j].storage().data_ptr(), ncclStreams[i]);
              }
              inputFlattened[i][j].copy_(inputTensors[i][j], true);
            }
          }
        },
        [&](std::vector<at::cuda::CUDAStream>&,
            c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
        OpType::REDUCE_SCATTER,
        "nccl:reduce_scatter");
  } else {
    const auto num_devices = inputTensors.size();
    const auto num_reduces = inputTensors[0].size();
    std::vector<c10::intrusive_ptr<Work>> works;
    startCoalescing();
    for (const auto i : c10::irange(num_reduces)) {
      std::vector<at::Tensor> inputs_multi_dev(num_devices);
      std::vector<at::Tensor> outputs_multi_dev(num_devices);
      for (const auto j : c10::irange(num_devices)) {
        // @lint-ignore CLANGTIDY
        inputs_multi_dev[j] = inputTensors[j][i];
        outputs_multi_dev[j] =
            // @lint-ignore CLANGTIDY
            i == (rank_ * num_devices + j) ? outputTensors[j]
                                           : inputs_multi_dev[j];
      }
      auto reduceOpts = ReduceOptions{
          opts.reduceOp,
          static_cast<int64_t>(i / num_devices),
          static_cast<int64_t>(i % num_devices),
          opts.timeout};
      auto work = _reduce_oop(outputs_multi_dev, inputs_multi_dev, reduceOpts);
      works.push_back(work);
    }
    auto work = endCoalescing();
    return work;
  }
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::_reduce_scatter_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const ReduceScatterOptions& opts) {
  if (inputTensor.dtype() != outputTensor.dtype()) {
    TORCH_CHECK(
        false, "input tensor must be the same type as the output tensor.");
  }

  if (inputTensor.numel() != outputTensor.numel() * size_) {
    TORCH_CHECK(
        false,
        "input tensor must be the same size as output size times world size");
  }

  // @lint-ignore CLANGTIDY
  const auto& tensor = outputTensor;

  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      reinterpret_cast<std::intptr_t>(this), // process group ptr
      inputTensor, // inputTensor
      outputTensor, // outputTensor
      rank_, // rank
      "_reduce_scatter_base", // colName
      tensor.numel() * // inSize
          this->getSize(),
      tensor.numel(), // outSize
      tensor.scalar_type(), // dtype
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>()); // outSplitSizes

  auto inputs = std::vector<at::Tensor>{inputTensor};
  auto outputs = std::vector<at::Tensor>{outputTensor};

  int dev_in_group = 0;
  // avoidRecordStreams_ note: collective() will stash inputs and outputs.
  return collective(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        if (!avoidRecordStreams_) {
          c10::cuda::CUDACachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
        auto ncclDataType = getNcclDataType(input.scalar_type());
        auto ncclReduceOp = getNcclReduceOp(
            opts.reduceOp, input, ncclDataType, comm, dev_in_group++);
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
      "nccl:_reduce_scatter_base");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const ReduceScatterOptions& opts) {
  return collective(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        if (!avoidRecordStreams_) {
          c10::cuda::CUDACachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
        auto ncclDataType = getNcclDataType(input.scalar_type());
        auto ncclReduceOp = getNcclReduceOp(
            opts.reduceOp, input, ncclDataType, comm, /*dev_in_group=*/0);
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
      "nccl:reduce_scatter_tensor_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::barrier(const BarrierOptions& opts) {
  RECORD_PARAM_COMMS(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      reinterpret_cast<std::intptr_t>(this), // process group ptr
      rank_, // rank
      "barrier", // colName
      0, // inSize
      0, // outSize
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>()); // outSplitSizes

  std::vector<at::Device> devices;

  // Use user defined GPU device ids if provided
  if (!opts.device_ids.empty()) {
    for (auto device : opts.device_ids) {
      devices.emplace_back(at::DeviceType::CUDA, device);
    }
  } else if (usedDeviceIdxs_.empty()) {
    // This means there is not yet a NCCL collective being called
    // Here we have to use the best guesses and will use a single GPU to call
    // allreduce to achieve barrier.
    // In case the multiple processes fall into the same node, we use rank to
    // ensure that each process is on a different GPU
    auto numGPUs = at::cuda::getNumGPUs();
    int16_t deviceIdx = static_cast<int16_t>(rank_ % numGPUs);
    LOG(INFO) << c10::str(
        "Rank ",
        this->getRank(),
        " using GPU ",
        deviceIdx,
        " to perform barrier as devices used by this process are currently unknown. ",
        "This can potentially cause a hang if this rank to GPU mapping is incorrect.",
        "Specify device_ids in barrier() to force use of a particular device.");
    devices.emplace_back(getDeviceForRank(rank_));
  } else {
    for (auto usedDeviceIdx : usedDeviceIdxs_) {
      devices.emplace_back(at::DeviceType::CUDA, usedDeviceIdx);
    }
  }

  std::vector<at::Tensor> barrierTensors;
  barrierTensors.reserve(devices.size());

  at::cuda::OptionalCUDAGuard gpuGuard;
  for (auto& device : devices) {
    gpuGuard.set_index(device.index());
    barrierTensors.push_back(at::empty(
        {1},
        at::TensorOptions().device(at::DeviceType::CUDA).dtype(at::kByte)));
  }

  // All reduce to achieve the barrier
  auto work = allreduce(barrierTensors);

  // Work will take over barrierTensors
  auto ncclWork = dynamic_cast<ProcessGroupNCCL::WorkNCCL*>(work.get());
  TORCH_CHECK(ncclWork);
  ncclWork->barrierTensors_ = std::move(barrierTensors);

  return work;
}

#ifdef ENABLE_NCCL_P2P_SUPPORT
c10::intrusive_ptr<Work> ProcessGroupNCCL::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {
  check_gpu_single_tensor(outputTensor);
  check_gpu_single_tensor(inputTensor);
  if (outputSplitSizes.size() == 0 && inputSplitSizes.size() == 0) {
    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};

    RECORD_PARAM_COMMS_DATA(
        static_cast<int>(
            this->getSequenceNumberForGroup() +
            1), // seq + 1 to match collective
        reinterpret_cast<std::intptr_t>(this), // process group ptr
        inputTensor, // inputTensor
        outputTensor, // outputTensor
        rank_, // rank
        "all_to_all", // colName
        inputTensor.numel(), // inSize
        outputTensor.numel(), // outSize
        inputTensor.scalar_type(), // dType
        std::vector<int64_t>(), // inSplitSizes
        std::vector<int64_t>()); // outSplitSizes

    // avoidRecordStreams_ note: collective() will stash inputTensors and
    // outputTensors.
    return collective(
        inputTensors,
        outputTensors,
        [&](at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream) {
          // See [Sync Streams].
          if (!avoidRecordStreams_) {
            c10::cuda::CUDACachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
          torch::cuda::nccl::all2all_single_equal_split(
              input, output, this->getSize(), comm, stream);
          return ncclSuccess;
        },
        OpType::ALLTOALL_BASE,
        "nccl:all_to_all");
  } else {
    c10d::checkSplitSizes(inputSplitSizes, inputTensor, size_);
    c10d::checkSplitSizes(outputSplitSizes, outputTensor, size_);
    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};

    RECORD_PARAM_COMMS_DATA(
        static_cast<int>(
            this->getSequenceNumberForGroup() +
            1), // seq + 1 to match collective
        reinterpret_cast<std::intptr_t>(this), // process group ptr
        inputTensor, // inputTensor
        outputTensor, // outputTensor
        rank_, // rank
        "all_to_allv", // colName
        inputTensor.numel(), // inSize
        outputTensor.numel(), // outSize
        inputTensor.scalar_type(), // dType
        inputSplitSizes, // inSplitSizes
        outputSplitSizes); // outSplitSizes

    // avoidRecordStreams_ note: collective() will stash inputTensors and
    // outputTensors.
    return collective(
        inputTensors,
        outputTensors,
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
          if (!avoidRecordStreams_) {
            c10::cuda::CUDACachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
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
        "nccl:all_to_all");
  }
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& /* unused */) {
  auto device = outputTensors[0].device();
  for (const auto r : c10::irange(outputTensors.size())) {
    check_gpu_single_tensor(outputTensors[r]);
    check_gpu_single_tensor(inputTensors[r]);
    TORCH_CHECK(
        device == outputTensors[r].device() &&
            device == inputTensors[r].device(),
        "Tensors must be on the same device")
  }
  std::vector<at::Tensor> inputTensor0 = {inputTensors[0]};
  std::vector<at::Tensor> outputTensor0 = {outputTensors[0]};
  return collective(
      inputTensor0,
      outputTensor0,
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        torch::cuda::nccl::all2all(outputTensors, inputTensors, comm, stream);
        return ncclSuccess;
      },
      [&](std::vector<at::cuda::CUDAStream>&,
          c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
        if (avoidRecordStreams_) {
          // inputTensor0 and outputTensor0 are stashed redundantly by
          // collective(), but that's ok.
          auto& v = work->stashed_for_allocator_safety_;
          v->insert(v->end(), inputTensors.begin(), inputTensors.end());
          v->insert(v->end(), outputTensors.begin(), outputTensors.end());
        }
      },
      [](std::vector<at::cuda::CUDAStream>&,
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
      OpType::ALLTOALL);
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int /* unused */) {
  check_gpu_tensors_different_devices(tensors);
  auto ret = pointToPoint(
      tensors,
      [&](at::Tensor& input,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream,
          int dst) {
        torch::cuda::nccl::send(input, comm, stream, dst);
        return ncclSuccess;
      },
      dstRank,
      OpType::SEND,
      "nccl:send");
  return ret;
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int /* unused */) {
  check_gpu_tensors_different_devices(tensors);
  auto ret = pointToPoint(
      tensors,
      [&](at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream,
          int src) {
        torch::cuda::nccl::recv(output, comm, stream, src);
        return ncclSuccess;
      },
      srcRank,
      OpType::RECV,
      "nccl:recv");
  return ret;
}
#else
c10::intrusive_ptr<Work> ProcessGroupNCCL::alltoall_base(
    at::Tensor& /* unused */,
    at::Tensor& /* unused */,
    std::vector<int64_t>& /* unused */,
    std::vector<int64_t>& /* unused */,
    const AllToAllOptions& /* unused */) {
  TORCH_CHECK(
      false,
      "ProcessGroupNCCL only supports alltoall* for NCCL lib version >= 2.7.0");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::alltoall(
    std::vector<at::Tensor>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllToAllOptions& /* unused */) {
  TORCH_CHECK(
      false,
      "ProcessGroupNCCL only supports alltoall* for NCCL lib version >= 2.7.0");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::send(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */,
    int /* unused */) {
  TORCH_CHECK(
      false,
      "ProcessGroupNCCL only supports send for NCCL lib version >= 2.7.0");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::recv(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */,
    int /* unused */) {
  TORCH_CHECK(
      false,
      "ProcessGroupNCCL only supports recv for NCCL lib version >= 2.7.0");
}
#endif

void ProcessGroupNCCL::groupStart() {
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
  C10D_NCCL_CHECK(ncclGroupStart(), c10::nullopt);
#endif
  ++ncclActiveGroupCounter_;
}

void ProcessGroupNCCL::groupEnd() {
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
#ifndef NCCL_HAS_COMM_NONBLOCKING
  C10D_NCCL_CHECK(ncclGroupEnd(), c10::nullopt);
#else
  if (!nccl_use_nonblocking()) {
    C10D_NCCL_CHECK(ncclGroupEnd(), c10::nullopt);
  } else {
    TORCH_WARN(
        "ProcessGroupNCCL::groupEnd() called in nonblocking communicator mode without involved communicators specified; gathering all mapped communicators...");
    std::unique_lock<std::mutex> lock(mutex_);
    std::vector<std::shared_ptr<NCCLComm>> ncclComms_;
    for (auto& it : devNCCLCommMap_) {
      ncclComms_.insert(ncclComms_.end(), it.second.begin(), it.second.end());
    }
    C10D_NCCL_CHECK_TIMEOUT_GROUPEND(ncclGroupEnd(), ncclComms_, c10::nullopt);
  }
#endif
#endif
  --ncclActiveGroupCounter_;
}

void ProcessGroupNCCL::groupEndNonblocking(
    std::vector<std::shared_ptr<NCCLComm>> comms) {
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
#ifndef NCCL_HAS_COMM_NONBLOCKING
  C10D_NCCL_CHECK(ncclGroupEnd(), c10::nullopt);
#else
  if (!nccl_use_nonblocking()) {
    C10D_NCCL_CHECK(ncclGroupEnd(), c10::nullopt);
  } else {
    C10D_NCCL_CHECK_TIMEOUT_GROUPEND(ncclGroupEnd(), comms, c10::nullopt);
  }
#endif
#endif
  --ncclActiveGroupCounter_;
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupNCCL::gather: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);
  check_gpu_tensors_different_devices(inputTensors);
  assertSingleElementInput(invalidArgument, inputTensors);

  // @lint-ignore CLANGTIDY
  auto tensor = inputTensors.back();

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

    const auto& options = inputTensors[0].options();
    const auto& sizes = inputTensors[0].sizes();
    assertTypeAndSizesMatch(invalidArgument, outputTensors[0], options, sizes);
    outputs = outputTensors[0];
  } else {
    // if not in the root rank, initialize outputs as empty list
    if (outputTensors.size() != 0) {
      invalidArgument("requires empty output on non-root");
    }
    outputs = {};
    // append a empty tensor to the list, we don't use it but the
    // `collective` template function requires it to invoke its function
    outputs.emplace_back();
  }

  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      reinterpret_cast<std::intptr_t>(this), // process group ptr
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "gather", // colName
      tensor.numel(), // inSize
      tensor.numel() * this->getSize(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>()); // outSplitSize

  // avoidRecordStreams_ note: collective() will stash inputTensors and
  // outputs, which == outputTensors[0] on the root rank where it matters.
  return collective(
      inputTensors,
      outputs,
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        const auto root = opts.rootRank;
        if (getRank() == root) {
          if (!avoidRecordStreams_) {
            for (auto output : outputs) {
              c10::cuda::CUDACachingAllocator::recordStream(
                  output.storage().data_ptr(), stream);
            }
          }
        }
        torch::cuda::nccl::gather(inputTensors[0], outputs, comm, stream, root);
        return ncclSuccess;
      },
      OpType::GATHER,
      "nccl:gather");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupNCCL::scatter: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);
  check_gpu_tensors_different_devices(outputTensors);
  assertSingleElementInput(invalidArgument, outputTensors);

  // @lint-ignore CLANGTIDY
  auto tensor = outputTensors.back();

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

    const auto& options = outputTensors[0].options();
    const auto& sizes = outputTensors[0].sizes();
    assertTypeAndSizesMatch(invalidArgument, inputTensors[0], options, sizes);
    inputs = inputTensors[0];
  } else {
    // if not in the root rank, initialize inputTensors as empty place holder
    // with an empty list
    if (inputTensors.size() != 0) {
      invalidArgument("requires empty input on non-root");
    }
    inputs = {};
    // append a empty tensor to the list, we don't use it but the
    // `collective` template function requires it to invoke its function
    inputs.emplace_back();
  }

  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      reinterpret_cast<std::intptr_t>(this), // process group ptr
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "scatter", // colName
      tensor.numel(), // inSize
      tensor.numel() * this->getSize(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>()); // outSplitSize

  // avoidRecordStreams_ note: collective() will stash outputTensors and
  // inputs, which == inputTensors[0] on the root rank where it matters.
  return collective(
      outputTensors,
      inputs,
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        const auto root = opts.rootRank;
        if (getRank() == root) {
          if (!avoidRecordStreams_) {
            for (auto input : inputs) {
              c10::cuda::CUDACachingAllocator::recordStream(
                  input.storage().data_ptr(), stream);
            }
          }
        }
        torch::cuda::nccl::scatter(
            inputs, outputTensors[0], comm, stream, root);
        return ncclSuccess;
      },
      OpType::SCATTER,
      "nccl:scatter");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::recvAnysource(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */) {
  TORCH_CHECK(false, "ProcessGroupNCCL does not support recvAnysource");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::_allgather_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const AllgatherOptions& /*unused */) {
  check_gpu_single_tensor(input_tensor);
  check_gpu_single_tensor(output_tensor);

  if (input_tensor.dtype() != output_tensor.dtype()) {
    TORCH_CHECK(false, "output tensor must have the same type as input tensor");
  }

  if (input_tensor.numel() * size_ != output_tensor.numel()) {
    TORCH_CHECK(
        false,
        "output tensor size must be equal to world_size times input tensor size");
  }

  // just a wrapper to fit the collective interface
  auto inputs = std::vector<at::Tensor>{input_tensor};
  auto outputs = std::vector<at::Tensor>{output_tensor};

  // avoidRecordStreams_ note: collective() will stash inputs and outputs.
  return collective(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        if (!avoidRecordStreams_) {
          c10::cuda::CUDACachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
        return ncclAllGather(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            getNcclDataType(input.scalar_type()),
            comm,
            stream.stream());
      },
      OpType::_ALLGATHER_BASE,
      "nccl:_all_gather_base");
}

#ifdef USE_NCCL_WITH_UCC
std::shared_ptr<at::DynamicLibrary> ProcessGroupNCCL::uccLib_ = nullptr;
#endif

bool ProcessGroupNCCL::isUCCAvailable() const {
#ifdef USE_NCCL_WITH_UCC
  return (uccPG_ != nullptr);
#else
  return false;
#endif
}

} // namespace c10d

#endif // USE_C10D_NCCL
