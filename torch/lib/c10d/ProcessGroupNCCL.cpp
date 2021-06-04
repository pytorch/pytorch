#include <c10/util/Optional.h>
#include <c10d/ProcessGroupNCCL.hpp>

#include <exception>
#include <map>
#include <tuple>
#include <unordered_set>

#include <THC/THC.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Logging.h>
#include <c10d/ParamCommsUtils.hpp>
#include <torch/csrc/cuda/nccl.h>

#include <c10d/Utils.hpp>
namespace c10d {

constexpr const char* const kNCCLAbortedCommStoreKey = "NCCLABORTEDCOMM";

namespace {

constexpr int kBytes = 8;

// RAII helper class to manage NCCL group API and CUDA free mutex.
// The destructor is allowed to throw since this helper class only
// manages group and lock lifetimes.
struct AutoNcclGroup {
  AutoNcclGroup() {
    (c10::cuda::CUDACachingAllocator::getFreeMutex())->lock();
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
    C10D_NCCL_CHECK(ncclGroupStart());
#endif
  }
  ~AutoNcclGroup() noexcept(false) {
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
    C10D_NCCL_CHECK(ncclGroupEnd());
#endif
    (c10::cuda::CUDACachingAllocator::getFreeMutex())->unlock();
  }
};

// NCCL op mapping
const std::map<ReduceOp, ncclRedOp_t> ncclOp = {
    {ReduceOp::MIN, ncclMin},
    {ReduceOp::MAX, ncclMax},
    {ReduceOp::SUM, ncclSum},
    {ReduceOp::PRODUCT, ncclProd},
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
#if defined(__HIP_PLATFORM_HCC__) && HIP_VERSION >= 301
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

ncclRedOp_t getNcclReduceOp(const ReduceOp reduceOp, at::Tensor& input) {
  try {
    if (reduceOp == ReduceOp::SUM && input.scalar_type() == at::kBool) {
      // For bool tensors, map sum to max, which both represent a bitwise or.
      // This is to prevent overflow issues with sum, since we use uint8 to
      // represent a bool (see ncclDataType mapping).
      return ncclMax;
    }
    return ncclOp.at(reduceOp);
  } catch (const std::out_of_range& e) {
    switch (reduceOp) {
      case ReduceOp::BAND:
        throw std::runtime_error("Cannot use ReduceOp.BAND with NCCL");
        break;
      case ReduceOp::BOR:
        throw std::runtime_error("Cannot use ReduceOp.BOR with NCCL");
        break;
      case ReduceOp::BXOR:
        throw std::runtime_error("Cannot use ReduceOp.BXOR with NCCL");
        break;
      default:
        throw std::runtime_error("Unhandled ReduceOp");
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
    res.push_back(tensor.device());
  }
  return res;
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
  for (size_t i = 0; i < devices.size(); ++i) {
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
  for (size_t i = 0; i < NCCL_UNIQUE_ID_BYTES; i++) {
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

} // namespace

const int64_t ProcessGroupNCCL::kWatchdogThreadSleepMillis = 10000;
const int64_t ProcessGroupNCCL::kWorkCleanupThreadSleepMillis = 1000;
constexpr int64_t kWaitForAbortCommStoreKey = 1000;
constexpr int64_t kSynchronizeBusyWaitMillis = 10;
thread_local uint64_t ProcessGroupNCCL::ncclActiveGroupCounter_ = 0;

std::ostream& operator<<(
    std::ostream& output,
    const ProcessGroupNCCL::WorkNCCL& workNCCL) {
  std::string workInfo;
  if (workNCCL.outputs_) {
    workInfo = c10::str(
        "WorkNCCL(",
        "OpType=",
        opTypeToString(workNCCL.opType_),
        ", TensorShape=",
        (*workNCCL.outputs_)[0].sizes(),
        ", Timeout(ms)=",
        workNCCL.opTimeout_.count(),
        ")");
  } else {
    workInfo = c10::str(
        "WorkNCCL(",
        "OpType=",
        opTypeToString(workNCCL.opType_),
        ", Timeout(ms)=",
        workNCCL.opTimeout_.count(),
        ")");
  }
  return output << workInfo;
}

ProcessGroupNCCL::WorkNCCL::WorkNCCL(
    const std::vector<at::Device>& devices,
    int rank,
    OpType opType,
    const char* profilingTitle,
    const c10::optional<std::vector<at::Tensor>>& inputs)
    : Work(rank, opType, profilingTitle, inputs),
      devices_(devices),
      workStartTime_(std::chrono::steady_clock::now()) {
  // Creates the CUDA event wrappers
  // Note: The actual events are lazily created when first recorded to with
  // DEFAULT_FLAGS = cudaEventDisableTiming.
  cudaEvents_ =
      std::make_shared<std::vector<at::cuda::CUDAEvent>>(devices.size());
  ncclComms_.resize(devices.size());
}

ProcessGroupNCCL::WorkNCCL::WorkNCCL(const WorkNCCL& w)
    : Work(w.rank_, w.opType_),
      std::enable_shared_from_this<WorkNCCL>(w),
      devices_(w.devices_),
      cudaEvents_(w.cudaEvents_),
      ncclComms_(w.ncclComms_),
      blockingWait_(w.blockingWait_),
      opTimeout_(w.opTimeout_),
      workStartTime_(w.workStartTime_) {
  completed_ = w.completed_;
  exception_ = w.exception_;
}

ProcessGroupNCCL::WorkNCCL::~WorkNCCL() {}

bool ProcessGroupNCCL::WorkNCCL::isCompleted() {
  checkAndSetException();
  return exception() || finishedGPUExecutionInternal();
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

bool ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const {
  for (size_t i = 0; i < devices_.size(); ++i) {
    // Checking the work's corresponding CUDA events' status
    if (!(*cudaEvents_)[i].query()) {
      return false;
    }
  }
  return true;
}

void ProcessGroupNCCL::WorkNCCL::checkAndThrowException() {
  // Set the appropriate exception if found.
  checkAndSetException();

  // Throw an exception, only if we have a valid exception.
  if (exception()) {
    std::rethrow_exception(exception());
  }
}

void ProcessGroupNCCL::WorkNCCL::handleNCCLGuard() {
  std::lock_guard<std::mutex> lock(mutex_);
  completed_ = true;
  if (exception_) {
    auto exceptionMsg = c10::str(
        "Some NCCL operations have failed or timed out. Due to the ",
        "asynchronous nature of CUDA kernels, subsequent GPU operations ",
        "might run on corrupted/incomplete data. To avoid this inconsistency, ",
        "we are taking the entire process down.");
    LOG(ERROR) << exceptionMsg;
    C10_LOG_API_USAGE_ONCE("ProcessGroupNCCL.WorkNCCL.handleNCCLGuard");
    std::rethrow_exception(exception_);
  }
}

void ProcessGroupNCCL::WorkNCCL::synchronize() {
  // Call Synchronize without a timeout. We use this method to avoid adding a
  // timeout argument to the public synchronize API.
  synchronizeInternal(kNoTimeout);
}

void ProcessGroupNCCL::WorkNCCL::synchronizeStreams() {
  for (size_t i = 0; i < devices_.size(); ++i) {
    auto currentStream = at::cuda::getCurrentCUDAStream(devices_[i].index());
    // Block the current stream on the NCCL stream
    (*cudaEvents_)[i].block(currentStream);
  }
}

// Waiting on the work's corresponding CUDA events
void ProcessGroupNCCL::WorkNCCL::synchronizeInternal(
    std::chrono::milliseconds timeout) {
  synchronizeStreams();

  // In case of blocking, wait for the operation to complete.
  if (blockingWait_) {
    // Use the passed in timeout if provided, otherwise use the default
    // opTimeout for each WorkNCCL object.
    std::chrono::milliseconds workTimeout =
        timeout == kNoTimeout ? opTimeout_ : timeout;
    // Wait for the operation to complete.
    while (!isCompleted()) {
      if (timedOut()) {
        // When operation times out due to some errors that are not
        // detected by nccl communicators, ncclCommWatchdog can not check this
        // time out error and thus can not abort ncclComms accordingly.
        // So explicitly abort ncclComms here before throwing this timed out
        // exception to users, after this, ncclCommWatchdog can detect nccl
        // communicators are aborted and clean up devNCCLCommMap_ accordingly.
        // if throwing timed out excepiton without aborting nccl communicators
        // here, it was observed that CUDA GPU will have 100% utilization and
        // can not run new events successfully.
        for (const auto& ncclComm : ncclComms_) {
          ncclComm->ncclCommAbort();
          const auto& storeKey = getNcclAbortedCommStoreKey(
              buildNcclUniqueIdStr(ncclComm->getNcclId()));
          auto rankStr = std::to_string(rank_);
          store_->set(
              storeKey,
              std::vector<uint8_t>(
                  reinterpret_cast<const uint8_t*>(rankStr.data()),
                  reinterpret_cast<const uint8_t*>(rankStr.data()) +
                      rankStr.size()));
          LOG(INFO) << "[Rank " << rank_
                    << "] Wrote aborted communicator id to store: " << storeKey;
        }
        auto currentTimepoint = std::chrono::steady_clock::now();
        auto timeElapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                currentTimepoint - workStartTime_);
        std::string exceptionMsg = c10::str(
            "[Rank ",
            rank_,
            "] ",
            "Caught collective operation timeout: ",
            (*this),
            " ran for ",
            timeElapsed.count(),
            " milliseconds before timing out.");
        throw std::runtime_error(exceptionMsg);
      }
      // Check for errors and throw appropriate exception.
      checkAndThrowException();
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
    }
    checkAndThrowException();
  }

  // Device synchronize only after we've completed timeout checks.
  if (!barrierTensors_.empty()) {
    // If we use the work to do barrier, we should block here
    for (auto& device : devices_) {
      at::cuda::CUDAGuard gpuGuard(device);
      AT_CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
}

// Same as calling synchronize().
bool ProcessGroupNCCL::WorkNCCL::wait(std::chrono::milliseconds timeout) {
  RECORD_PARAM_COMMS(
      rank_,                    // rank
      "wait",                   // colName
      0,                        // inSize
      0,                        // outSize
      at::kByte,                // dType
      std::vector<int64_t>(),   // inSplitSizes
      std::vector<int64_t>());  // outSplitSizes
  synchronizeInternal(timeout);
  // Always return true, because abort API is not implemented.
  return true;
}

void ProcessGroupNCCL::WorkNCCL::abort() {
  TORCH_CHECK(false, "ProcessGroupNCCL::WorkNCCL::abort not implemented.");
}

bool ProcessGroupNCCL::WorkNCCL::timedOut() {
  auto currentTimepoint = std::chrono::steady_clock::now();
  return (
      std::chrono::duration_cast<std::chrono::milliseconds>(
          currentTimepoint - workStartTime_) >= opTimeout_);
}

ProcessGroupNCCL::ProcessGroupNCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : ProcessGroup(rank, size),
      store_(store),
      options_(options),
      ncclCommCounter_(0),
      terminateProcessGroup_(false) {
  TORCH_CHECK(
      at::cuda::getNumGPUs() != 0,
      "ProcessGroupNCCL is only supported with GPUs, no GPUs found!");
  blockingWait_ = parseEnvVarFlag(NCCL_BLOCKING_WAIT);
  asyncErrorHandling_ = parseEnvVarFlag(NCCL_ASYNC_ERROR_HANDLING);

  if (blockingWait_ && asyncErrorHandling_) {
    LOG(INFO) << "[Rank " << rank_
              << "] NCCL_BLOCKING_WAIT and NCCL_ASYNC_ERROR_HANDLING "
              << "should not both be enabled. "
              << "Only NCCL_BLOCKING_WAIT is being used in this process.";
    asyncErrorHandling_ = false;
  }

#ifdef ENABLE_NCCL_ERROR_CHECKING
  ncclCommWatchdogThread_ =
      std::thread(&ProcessGroupNCCL::ncclCommWatchdog, this);
#endif

  if (asyncErrorHandling_) {
    workCleanupThread_ = std::thread(&ProcessGroupNCCL::workCleanupLoop, this);
  }

  const char* ncclDebugLevel = std::getenv("NCCL_DEBUG");

  if (!ncclDebugLevel) {
    ncclDebugLevel = "UNSET";
  }

  LOG(INFO) << "[Rank " << rank_
            << "] ProcessGroupNCCL initialized with following options:"
            << "\nNCCL_ASYNC_ERROR_HANDLING: " << asyncErrorHandling_
            << "\nNCCL_BLOCKING_WAIT: " << blockingWait_
            << "\nTIMEOUT(ms): " << options_->timeout.count()
            << "\nUSE_HIGH_PRIORITY_STREAM: "
            << options_->is_high_priority_stream
            << "\nNCCL_DEBUG: " << ncclDebugLevel;
}

void ProcessGroupNCCL::setSequenceNumberForGroup() {
  if (rank_ == 0) {
    // Create and broadcast sequence number
    auto seq = 1 + rand();
    sequenceNum_ = c10d::SequenceNum(seq);
    std::vector<uint8_t> values = c10d::toVec<uint8_t>(seq, kBytes);
    store_->set(kSeqNumStoreKey, values);
  } else {
    // Read rank 0's sequence number from store.
    sequenceNum_ = c10d::SequenceNum();
    store_->wait({kSeqNumStoreKey}, options_->timeout);
    std::vector<uint8_t> values = store_->get(kSeqNumStoreKey);
    uint64_t num = c10d::fromVec<uint8_t>(values);
    sequenceNum_->set(num);
  }
}

uint64_t ProcessGroupNCCL::getSequenceNumberForGroup() {
  if (sequenceNum_ == c10::nullopt) {
    return 0;
  }
  return sequenceNum_->get();
}

ProcessGroupNCCL::~ProcessGroupNCCL() {
  terminateProcessGroup_.store(true);

  watchdogCV_.notify_one();
#ifdef ENABLE_NCCL_ERROR_CHECKING
  ncclCommWatchdogThread_.join();
#endif

  if (asyncErrorHandling_) {
    workMetaListCV_.notify_one();
    workCleanupThread_.join();
  }

  {
    // Abort all NCCL Communicators on Process Group Destruction
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& it : devNCCLCommMap_) {
      auto& ncclComms = it.second;

      for (const auto& ncclComm : ncclComms) {
        ncclComm->ncclCommAbort();
      }
    }
  }
}

void ProcessGroupNCCL::abortTimedOutCollectives(
    std::unordered_set<std::string>& abortedCommIds) {
  std::unique_lock<std::mutex> lock(workMetaListMutex_);
  for (auto& work : workMetaList_) {
    work.checkAndSetException();
    // Aborting NCCL Communicators due to errors is already handled above.
    if (work.exception()) {
      continue;
    }

    // Check for Timeouts in the WorkNCCL Operations, and abort all
    // communicators accordingly.
    if (work.timedOut()) {
      auto currentTimepoint = std::chrono::steady_clock::now();
      auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          currentTimepoint - work.workStartTime_);
      std::string exceptionMsg = c10::str(
          "[Rank ",
          rank_,
          "] ",
          "Watchdog caught collective operation timeout: ",
          work,
          " ran for ",
          timeElapsed.count(),
          " milliseconds before timing out.");
      LOG(ERROR) << exceptionMsg;
      std::exception_ptr exception_ptr =
          std::make_exception_ptr(std::runtime_error(exceptionMsg));
      work.setException(exception_ptr);
      for (const auto& ncclComm : work.ncclComms_) {
        ncclComm->ncclCommAbort();
        abortedCommIds.emplace(buildNcclUniqueIdStr(ncclComm->getNcclId()));
      }
    }
  }
}

void ProcessGroupNCCL::ncclCommWatchdog() {
  try {
    LOG(INFO) << "[Rank " << rank_ << "] NCCL watchdog thread started!";
    ncclCommWatchdogInternal();
    LOG(INFO) << "[Rank " << rank_
              << "] NCCL watchdog thread terminated normally";
  } catch (std::exception& e) {
    LOG(INFO) << "[Rank " << rank_
              << "] NCCL watchdog thread terminated with exception: "
              << e.what();
  } catch (...) {
    LOG(INFO) << "[Rank " << rank_
              << "] NCCL watchdog thread terminated with unknown exception";
  }
}

void ProcessGroupNCCL::ncclCommWatchdogInternal() {
  while (!terminateProcessGroup_.load()) {
    std::unordered_set<std::string> abortedCommIds;
    std::unordered_set<std::string> allCommIds;

    {
      // Loop through the cache of communicators for NCCL errors.
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto& it : devNCCLCommMap_) {
        auto& ncclComms = it.second;

        for (const auto& ncclComm : ncclComms) {
          allCommIds.emplace(buildNcclUniqueIdStr(ncclComm->getNcclId()));
        }
        std::exception_ptr ncclErrorException = checkForNCCLErrors(ncclComms);
        if (ncclErrorException) {
          LOG(INFO)
              << "[Rank " << rank_
              << "] Received NCCL errors for communicators in the cache: \n"
              << "NCCL error: \n"
              << getExceptionMsgFromExceptionPtr(ncclErrorException);

          if (blockingWait_ || asyncErrorHandling_) {
            LOG(INFO) << "[Rank " << rank_
                      << "] Aborting communicators that received errors";
            // We abort NCCL communicators that have received errors from this
            // thread, and exceptions are set on the corresponding work objects.
            // The workCleanupThread will then loop through the unfinished
            // collectives and throw exceptions if an exception has been set on
            // any of the work objects from this thread.
            for (const auto& ncclComm : ncclComms) {
              ncclComm->ncclCommAbort();
              // Note that we don't remove the aborted communicators from the
              // cache. The reason is that if we do remove the communicator
              // from the cache, it is possible that a new collective operation
              // calls `ncclCommInitRank` to create a new communicator whereas
              // other ranks might have failed/timed out and didn't enter
              // `ncclCommInitRank`. As a result, when there is a failure on
              // a communicator the application receives an exception and its
              // their responsibility to destroy the process group and recreate
              // it to recover from errors.
              abortedCommIds.emplace(
                  buildNcclUniqueIdStr(ncclComm->getNcclId()));
            }
          }
        }
      }
    }

    if (asyncErrorHandling_) {
      abortTimedOutCollectives(abortedCommIds);
    }

    if (blockingWait_) {
      // When we abort a communicator on one rank, it is likely that might cause
      // other ranks to hang indefinitely. As a result, whenever we abort a
      // communicator, we write its ID to the store. The watchdog on other ranks
      // then monitor the store, find an aborted communicator ID and abort their
      // respective communicator as well.

      // Record the aborted communicators locally and in the store.
      for (const auto& abortedCommId : abortedCommIds) {
        abortedComms_.emplace(abortedCommId);
        const auto& storeKey = getNcclAbortedCommStoreKey(abortedCommId);
        auto rankStr = std::to_string(rank_);
        store_->set(
            storeKey,
            std::vector<uint8_t>(
                reinterpret_cast<const uint8_t*>(rankStr.data()),
                reinterpret_cast<const uint8_t*>(rankStr.data()) +
                    rankStr.size()));
        LOG(INFO) << "[Rank " << rank_
                  << "] Watchdog wrote aborted communicator id to store: "
                  << storeKey;
      }

      // Check for any communicators in the store and abort them if needed.
      for (const auto& commId : allCommIds) {
        if (abortedComms_.find(commId) == abortedComms_.end()) {
          // Check if we need to abort them if not already aborted (shouldn't
          // wait more than the watchdog sleep time.).
          const auto& storeKey = getNcclAbortedCommStoreKey(commId);
          try {
            store_->wait(
                {storeKey},
                std::chrono::milliseconds(kWaitForAbortCommStoreKey));
            auto val = store_->get(storeKey);
            std::string rank(reinterpret_cast<char*>(val.data()), val.size());
            LOG(INFO) << "[Rank " << rank_
                      << "] Found key in store: " << storeKey
                      << ", from rank: " << rank
                      << ", aborting appropriate communicators";

            // Now abort the appropriate communicators.
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = ncclIdToCommMap_.find(commId);
            TORCH_INTERNAL_ASSERT(it != ncclIdToCommMap_.end());
            for (const auto& ncclComm : it->second) {
              ncclComm->ncclCommAbort();
            }
            abortedComms_.emplace(commId);
            LOG(INFO) << "[Rank " << rank_
                      << "] Aborted communicators for key in store: "
                      << storeKey;
          } catch (std::exception& e) {
            VLOG(1) << "Did not find key in store: " << storeKey
                    << ", error: " << e.what();
          }
        }
      }
    }

    std::unique_lock<std::mutex> lock(watchdogCVMutex_);
    watchdogCV_.wait_for(
        lock,
        std::chrono::milliseconds(kWatchdogThreadSleepMillis),
        [&]() -> bool { return terminateProcessGroup_.load(); });
  }
}

void ProcessGroupNCCL::workCleanupLoop() {
  bool done = false;
  while (!terminateProcessGroup_.load() || !done) {
    std::list<WorkNCCL> doneWorks;
    {
      std::unique_lock<std::mutex> lock(workMetaListMutex_);
      // We busy-poll the work vector every kWatchdogThreadSleepMillis
      // milliseconds as long as the atomic is True.
      workMetaListCV_.wait_for(
          lock,
          std::chrono::milliseconds(kWorkCleanupThreadSleepMillis),
          [&]() -> bool { return terminateProcessGroup_.load(); });

      for (auto it = workMetaList_.begin(); it != workMetaList_.end();
           /* no increment*/) {
        auto& work = *it;
        if (work.isCompleted()) {
          // Handle Exceptions on failed GPU operations and remove completed
          // workNCCL objects from work vector.
          if (!terminateProcessGroup_.load()) {
            work.handleNCCLGuard();
          }
          doneWorks.push_back(std::move(*it));
          it = workMetaList_.erase(it);
        } else {
          // Increment the iterator if the current WorkNCCL object is not
          // completed.
          ++it;
        }
      }
      done = workMetaList_.empty();
    }
    doneWorks.clear();
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
    OpType opType,
    const std::string& p2pKey,
    int p2pRank) {
  // For collective operations:
  // For every NCCL communicator that we create we need to broadcast
  // a unique ID from rank 0 to all other ranks. This broadcast is
  // done by rank 0 setting a key in the store and all other ranks
  // retrieving the contents of that key. A single process group
  // may create multiple NCCL communicators, so we use a sequence
  // number to differentiate between them.
  // For point-to-point operations:
  // The sequence number will only be increased on 2 out of all the
  // processes in a Process Group. So all following collective
  // operations will see different sequence numbers which will cause
  // runtime errors. To avoid that, use the src:target pair instead
  // of sequence number for p2p communications.

  std::string storeKey;
  if (!isP2POp(opType)) {
    storeKey = std::to_string(ncclCommCounter_++);
  } else {
    storeKey = p2pKey;
  }
  if (rank_ == 0 || (isP2POp(opType) && p2pRank == 0)) {
    auto vec = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(ncclID),
        reinterpret_cast<uint8_t*>(ncclID) + NCCL_UNIQUE_ID_BYTES);
    store_->set(storeKey, vec);
  } else {
    auto vec = store_->get(storeKey);
    TORCH_CHECK(vec.size() == NCCL_UNIQUE_ID_BYTES);
    std::memcpy(ncclID, vec.data(), vec.size());
  }
}

std::vector<std::shared_ptr<NCCLComm>>& ProcessGroupNCCL::getNCCLComm(
    const std::string& devicesKey,
    const std::vector<at::Device>& devices,
    OpType opType,
    int p2pRank,
    bool isSendRecvSelf) {
  // Sanity check
  if (devicesKey.empty()) {
    throw std::runtime_error(
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

  // For point-to-point communication, lower rank of the two will get unique id.
  if (rank_ == 0 || (isP2POp(opType) && p2pRank == 0)) {
    C10D_NCCL_CHECK(ncclGetUniqueId(&ncclID));
  }

  // For point-to-point communication on the same process, don't need broadcast.
  if (!isSendRecvSelf) {
    // Broadcast so that each process can have a unique NCCL ID
    broadcastUniqueNCCLID(&ncclID, opType, devicesKey, p2pRank);
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
  for (size_t i = 0; i < ncclActiveGroupCounter_; ++i) {
    C10D_NCCL_CHECK(ncclGroupEnd());
  }

  // [Note 1] Create the NCCL communicators for each GPU
  C10D_NCCL_CHECK(ncclGroupStart());

  for (size_t i = 0; i < devices.size(); ++i) {
    // GPU world size and GPU rank
    int numRanks, rank;

    if (!isP2POp(opType)) {
      numRanks = getSize() * devices.size();
      rank = getRank() * devices.size() + i;
    } else if (isSendRecvSelf) {
      // Same process send and recv.
      numRanks = 1;
      rank = 0;
    } else {
      // For point-to-point operation, there are only 2 processes involved so
      // the GPU rank is either 0 or 1.
      numRanks = 2;
      rank = p2pRank;
    }
    // Get the device index
    int deviceIndex = devices[i].index();

    gpuGuard.set_index(deviceIndex);
    ncclComms[i] = NCCLComm::create(numRanks, rank, ncclID);

    // Creates the NCCL streams
    streamVal.push_back(
        at::cuda::getStreamFromPool(options_->is_high_priority_stream));
  }

  // [Note 2 ]
  C10D_NCCL_CHECK(ncclGroupEnd());

  // See [Group Start/End Note]
  for (size_t i = 0; i < ncclActiveGroupCounter_; ++i) {
    C10D_NCCL_CHECK(ncclGroupStart());
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
  devNCCLCommMap_.emplace(devicesKey, std::move(ncclComms));
  return devNCCLCommMap_[devicesKey];
}

namespace {

// Check validity of tensor
void check_gpu_single_tensor(const at::Tensor& tensor) {
  if (!tensor.is_cuda() || tensor.is_sparse()) {
    throw std::runtime_error("Tensors must be CUDA and dense");
  }
  if (!tensor.is_contiguous()) {
    throw std::runtime_error("Tensors must be contiguous");
  }
}

// Check that all `tensors' have the same type and shape and are distributed
// across distinct GPUs.
void check_gpu_tensors(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() == 0) {
    throw std::runtime_error("Tensor list must be nonempty");
  }
  if (tensors.size() > static_cast<size_t>(at::cuda::getNumGPUs())) {
    throw std::runtime_error(
        "Tensor list mustn't be larger than the number of available GPUs");
  }

  const auto& first = tensors.front();

  // Set for ensuring that tensors are on separate devices.
  std::unordered_set<decltype(first.get_device())> usedDevices;
  usedDevices.reserve(tensors.size());

  for (const auto& t : tensors) {
    if (!t.is_cuda() || t.is_sparse()) {
      throw std::runtime_error("Tensors must be CUDA and dense");
    }
    if (t.scalar_type() != first.scalar_type()) {
      throw std::runtime_error("Tensors must have identical type");
    }
    if (t.sizes() != first.sizes()) {
      throw std::runtime_error("Tensors must have identical size");
    }
    if (t.strides() != first.strides()) {
      throw std::runtime_error("Tensors must have identical strides");
    }
    if (!t.is_non_overlapping_and_dense()) {
      throw std::runtime_error("Tensors must be non-overlapping and dense");
    }
    const auto inserted = usedDevices.insert(t.get_device()).second;
    if (!inserted) {
      throw std::runtime_error("Tensors must be on distinct GPU devices");
    }
  }
}

// Flatten each list in `tensor_lists' for a gather or scatter operation, and
// ensure compatibility with the corresponding tensor in `other'.
std::vector<at::Tensor> flatten_for_scatter_gather(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    std::vector<at::Tensor>& other,
    size_t world_size) {
  if (tensor_lists.size() != other.size()) {
    throw std::runtime_error(
        "Tensor list operands to scatter/gather must have the same length");
  }
  const auto num_devices = tensor_lists.size();

  std::vector<at::Tensor> flattened;
  flattened.resize(num_devices);

  for (auto i = size_t{}; i < num_devices; ++i) {
    if (tensor_lists[i].size() != world_size * num_devices) {
      throw std::runtime_error(
          "Tensor list input to scatter/gather must match number of collective"
          " participants");
    }

    // Only check device match for the first tensor in the list; the call to
    // newLikeFlat() below will check the rest.
    if (tensor_lists[i].front().get_device() != other[i].get_device()) {
      throw std::runtime_error(
          "Corresponding input/output tensors to scatter/gather must all reside"
          " on the same device");
    }

    for (const auto& t : tensor_lists[i]) {
      if (t.numel() != other[i].numel()) {
        throw std::runtime_error(
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
      devices, rank, opType, profilingTitle, inputs);
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
    workMetaList_.emplace_back(WorkNCCL(*work));
  }
}

ProcessGroupNCCL::Options::Options(bool is_high_priority_stream)
    : ProcessGroup::Options(NCCL_BACKEND_NAME),
      is_high_priority_stream(is_high_priority_stream) {}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    OpType opType,
    const char* profilingTitle) {

  // Bump collective counter
  if (sequenceNum_) {
    sequenceNum_->increment();
  }
  const auto devices = getDeviceList(inputs);
  const auto key = getKeyFromDevices(devices);
  auto& ncclComms = getNCCLComm(key, devices, opType);

  // First let NCCL streams wait for input tensors allocation streams
  syncStreams(devices, ncclEvents_[key], ncclStreams_[key]);

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

  at::cuda::OptionalCUDAGuard gpuGuard;

  pre(ncclStreams_[key]);

  for (size_t i = 0; i < inputs.size(); ++i) {
    gpuGuard.set_index(devices[i].index());
    at::cuda::CUDAStream& ncclStream = ncclStreams_[key][i];

    // Both `inputs' and `outputs' are created on a worker stream and used in
    // different ncclStreams.  Hence, both must record the ncclStream to
    // prevent being freed before the collective finishes.
    //
    // We only record `inputs' here, and leave recording `outputs' to `fn' for
    // operations where `inputs' and `outputs' are not the same.
    //
    // See [Sync Streams].
    c10::cuda::CUDACachingAllocator::recordStream(
        inputs[i].storage().data_ptr(), ncclStream);
  }

  {
    AutoNcclGroup nccl_group_guard;
    for (size_t i = 0; i < inputs.size(); ++i) {
      gpuGuard.set_index(devices[i].index());
      at::cuda::CUDAStream& ncclStream = ncclStreams_[key][i];
      C10D_NCCL_CHECK(
          fn(inputs[i], outputs[i], ncclComms[i]->getNcclComm(), ncclStream));
    }
  }

  post(ncclStreams_[key]);

  // Event should only be recorded after the ncclGroupEnd()
  for (size_t i = 0; i < inputs.size(); ++i) {
    at::cuda::CUDAStream& ncclStream = ncclStreams_[key][i];
    (*work->cudaEvents_)[i].record(ncclStream);
    work->ncclComms_[i] = ncclComms[i];
  }

  {
    c10::cuda::CUDAMultiStreamGuard streamGuard(ncclStreams_[key]);
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()),
        devices);

    // Add a callback that runs profiling end callbacks. wrapCallback() in CUDA
    // future blocks the stream this callback runs on the corresponding
    // cudaEvents_ ensuring appropriate synchronization.
    if (work->recordFunctionEndCallback_) {
      work->future_->addCallback(
          [work](at::ivalue::Future& /* unused */) { work->recordFunctionEndCallback_(); });
    }
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }

  // Set appropriate work parameters.
  work->blockingWait_ = blockingWait_;
  work->opTimeout_ = options_->timeout;
  work->store_ = store_;

  if (asyncErrorHandling_) {
    workEnqueue(work);
  }

  return work;
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::pointToPoint(
    std::vector<at::Tensor>& tensors,
    Fn fn,
    int peer,
    OpType opType,
    PreProcess pre,
    PostProcess post,
    const char* profilingTitle) {
  const auto devices = getDeviceList(tensors);
  const auto key = getKeySendRecv(rank_, peer);
  int p2pRank = rank_ <= peer ? 0 : 1;
  auto isSendRecvSelf = rank_ == peer;
  auto& ncclComms = getNCCLComm(key, devices, opType, p2pRank, isSendRecvSelf);

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

  pre(ncclStreams_[key]);

  for (size_t i = 0; i < tensors.size(); ++i) {
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

  {
    AutoNcclGroup nccl_group_guard;
    for (size_t i = 0; i < tensors.size(); ++i) {
      gpuGuard.set_index(devices[i].index());
      at::cuda::CUDAStream& ncclStream = ncclStreams_[key][i];
      // For point-to-point communication, NCCL ranks can only
      // be 0 or 1.
      int p2pTargetRank = isSendRecvSelf ? 0 : 1 - p2pRank;
      C10D_NCCL_CHECK(fn(
          tensors[i], ncclComms[i]->getNcclComm(), ncclStream, p2pTargetRank));
    }
  }

  post(ncclStreams_[key]);

  // Event should only be recorded after the ncclGroupEnd()
  for (size_t i = 0; i < tensors.size(); ++i) {
    at::cuda::CUDAStream& ncclStream = ncclStreams_[key][i];
    (*work->cudaEvents_)[i].record(ncclStream);
    work->ncclComms_[i] = ncclComms[i];
    work->blockingWait_ = blockingWait_;
    work->opTimeout_ = options_->timeout;
    work->store_ = store_;
  }

  // Future only needs to be created and marked completed with outputs for
  // recv(), but still create future for use cases such as profiling even for
  // send().
  {
    c10::cuda::CUDAMultiStreamGuard streamGuard(ncclStreams_[key]);
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()),
        devices);
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }

  // Add a callback that runs profiling end callbacks. wrapCallback() in CUDA
  // future blocks the stream this callback runs on the corresponding
  // cudaEvents_ ensuring appropriate synchronization.
  if (work->recordFunctionEndCallback_) {
    work->future_->addCallback(
        [work](at::ivalue::Future& /* unused */) { work->recordFunctionEndCallback_(); });
  }

  return work;
}

template <typename Fn>
c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    OpType opType,
    const char* profilingTitle) {
  return collective(
      inputs,
      outputs,
      fn,
      [](std::vector<at::cuda::CUDAStream>&) {},
      [](std::vector<at::cuda::CUDAStream>&) {},
      opType,
      profilingTitle);
}

template <typename Fn>
c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::pointToPoint(
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
      [](std::vector<at::cuda::CUDAStream>&) {},
      [](std::vector<at::cuda::CUDAStream>&) {},
      profilingTitle);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  check_gpu_tensors(tensors);

  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();
  RECORD_PARAM_COMMS(
      rank_, // rank
      "allreduce", // colName
      tensor.numel(), // inSize
      tensor.numel(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>()); // outSplitSizes

  return collective(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        return ncclAllReduce(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            getNcclDataType(input.scalar_type()),
            getNcclReduceOp(opts.reduceOp, input),
            comm,
            stream.stream());
      },
      OpType::ALLREDUCE,
      "nccl:all_reduce");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  throw std::runtime_error(
      "allreduce_coalesced is currently not supported with NCCL");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  check_gpu_tensors(tensors);

  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();
  RECORD_PARAM_COMMS(
      rank_,                    // rank
      "broadcast",              // colName
      tensor.numel(),           // inSize
      tensor.numel(),           // outSize
      tensor.scalar_type(),     // dType
      std::vector<int64_t>(),   // inSplitSizes
      std::vector<int64_t>());  // outSplitSizes

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

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  check_gpu_tensors(tensors);
  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();
  RECORD_PARAM_COMMS(
      rank_,                    // rank
      "reduce",                 // colName
      tensor.numel(),           // inSize
      tensor.numel(),           // outSize
      tensor.scalar_type(),     // dType
      std::vector<int64_t>(),   // inSplitSizes
      std::vector<int64_t>());  // outSplitSizes

  return collective(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        const auto root = opts.rootRank * tensors.size() + opts.rootTensor;
        return ncclReduce(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            getNcclDataType(input.scalar_type()),
            getNcclReduceOp(opts.reduceOp, input),
            root,
            comm,
            stream.stream());
      },
      OpType::REDUCE,
      "nccl:reduce");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  check_gpu_tensors(inputTensors);

  auto outputFlattened =
      flatten_for_scatter_gather(outputTensors, inputTensors, size_);
  check_gpu_tensors(outputFlattened);

  // @lint-ignore CLANGTIDY
  auto tensor = inputTensors.back();
  RECORD_PARAM_COMMS(
      rank_,                    // rank
      "all_gather",             // colName
      tensor.numel(),           // inSize
      tensor.numel() *          // outSize
        this->getSize(),        // dType
      tensor.scalar_type(),
      std::vector<int64_t>(),   // inSplitSizes
      std::vector<int64_t>());  // outSplitSize

  return collective(
      inputTensors,
      outputFlattened,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        c10::cuda::CUDACachingAllocator::recordStream(
            output.storage().data_ptr(), stream);
        return ncclAllGather(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            getNcclDataType(input.scalar_type()),
            comm,
            stream.stream());
      },
      [&](std::vector<at::cuda::CUDAStream>& ncclStreams) {},
      [&](std::vector<at::cuda::CUDAStream>& ncclStreams) {
        // Copy the flattened output tensors to the outputs.
        for (size_t i = 0; i < outputTensors.size(); ++i) {
          at::cuda::CUDAStreamGuard guard(ncclStreams[i]);
          for (size_t j = 0; j < outputTensors[0].size(); ++j) {
            // See [Sync Streams].
            c10::cuda::CUDACachingAllocator::recordStream(
                outputTensors[i][j].storage().data_ptr(), ncclStreams[i]);

            outputTensors[i][j].copy_(outputFlattened[i][j], true);
          }
        }
      },
      OpType::ALLGATHER,
      "nccl:all_gather");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllgatherOptions& /* unused */) {
  throw std::runtime_error(
      "ProcessGroupNCCL does not support allgather_coalesced");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  check_gpu_tensors(outputTensors);

  // @lint-ignore CLANGTIDY
  auto tensor = outputTensors.back();
  RECORD_PARAM_COMMS(
      rank_,                    // rank
      "reduce_scatter",         // colName
      tensor.numel() *          // inSize
        this->getSize(),        // outSize
      tensor.numel(),           // dType
      tensor.scalar_type(),
      std::vector<int64_t>(),   // inSplitSizes
      std::vector<int64_t>());  // outSplitSizes

  auto inputFlattened =
      flatten_for_scatter_gather(inputTensors, outputTensors, size_);
  check_gpu_tensors(inputFlattened);

  return collective(
      inputFlattened,
      outputTensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        c10::cuda::CUDACachingAllocator::recordStream(
            output.storage().data_ptr(), stream);
        return ncclReduceScatter(
            input.data_ptr(),
            output.data_ptr(),
            output.numel(),
            getNcclDataType(input.scalar_type()),
            getNcclReduceOp(opts.reduceOp, input),
            comm,
            stream.stream());
      },
      [&](std::vector<at::cuda::CUDAStream>& ncclStreams) {
        // Copy the input tensors to the flattened inputs.
        for (size_t i = 0; i < inputTensors.size(); ++i) {
          at::cuda::CUDAStreamGuard guard(ncclStreams[i]);
          for (size_t j = 0; j < inputTensors[0].size(); ++j) {
            // See [Sync Streams].
            c10::cuda::CUDACachingAllocator::recordStream(
                inputTensors[i][j].storage().data_ptr(), ncclStreams[i]);

            inputFlattened[i][j].copy_(inputTensors[i][j], true);
          }
        }
      },
      [&](std::vector<at::cuda::CUDAStream>& ncclStreams) {},
      OpType::REDUCE_SCATTER,
      "nccl:reduce_scatter");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::barrier(
    const BarrierOptions& opts) {
  RECORD_PARAM_COMMS(
      rank_,                    // rank
      "barrier",                // colName
      0,                        // inSize
      0,                        // outSize
      at::kByte,                // dType
      std::vector<int64_t>(),   // inSplitSizes
      std::vector<int64_t>());  // outSplitSizes

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
    LOG(WARNING) << c10::str(
        "Rank ",
        this->getRank(),
        " using best-guess GPU ",
        deviceIdx,
        " to perform barrier as devices used by this process are currently unknown. ",
        "This can potentially cause a hang if this rank to GPU mapping is incorrect.",
        "Specify device_ids in barrier() to force use of a particular device.");
    devices.emplace_back(at::DeviceType::CUDA, deviceIdx);
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
c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::alltoall_base(
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

    RECORD_PARAM_COMMS(
        rank_,                    // rank
        "all_to_all",             // colName
        inputTensor.numel(),      // inSize
        outputTensor.numel(),     // outSize
        inputTensor.scalar_type(),// dType
        std::vector<int64_t>(),   // inSplitSizes
        std::vector<int64_t>());  // outSplitSizes

    return collective(
        inputTensors,
        outputTensors,
        [&](at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream) {
          // See [Sync Streams].
          c10::cuda::CUDACachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
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

    RECORD_PARAM_COMMS(
        rank_,                    // rank
        "all_to_allv",            // colName
        inputTensor.numel(),      // inSize
        outputTensor.numel(),     // outSize
        inputTensor.scalar_type(),// dType
        inputSplitSizes,          // inSplitSizes
        outputSplitSizes);        // outSplitSizes

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
          c10::cuda::CUDACachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
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

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& /* unused */) {
  auto device = outputTensors[0].device();
  for (size_t r = 0; r < outputTensors.size(); r++) {
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
      OpType::ALLTOALL);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int /* unused */) {
  check_gpu_tensors(tensors);
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

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int /* unused */) {
  check_gpu_tensors(tensors);
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
c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::alltoall_base(
    at::Tensor& /* unused */,
    at::Tensor& /* unused */,
    std::vector<int64_t>& /* unused */,
    std::vector<int64_t>& /* unused */,
    const AllToAllOptions& /* unused */) {
  throw std::runtime_error(
      "ProcessGroupNCCL only supports alltoall* for NCCL lib version >= 2.7.0");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::alltoall(
    std::vector<at::Tensor>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllToAllOptions& /* unused */) {
  throw std::runtime_error(
      "ProcessGroupNCCL only supports alltoall* for NCCL lib version >= 2.7.0");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::send(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */,
    int /* unused */) {
  throw std::runtime_error(
      "ProcessGroupNCCL only supports send for NCCL lib version >= 2.7.0");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::recv(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */,
    int /* unused */) {
  throw std::runtime_error(
      "ProcessGroupNCCL only supports recv for NCCL lib version >= 2.7.0");
}
#endif

void ProcessGroupNCCL::groupStart() {
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
  C10D_NCCL_CHECK(ncclGroupStart());
#endif
  ++ncclActiveGroupCounter_;
}

void ProcessGroupNCCL::groupEnd() {
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
  C10D_NCCL_CHECK(ncclGroupEnd());
#endif
  --ncclActiveGroupCounter_;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const GatherOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupNCCL does not support gather");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ScatterOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupNCCL does not support scatter");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::recvAnysource(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */) {
  throw std::runtime_error("ProcessGroupNCCL does not support recvAnysource");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::_allgather_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const AllgatherOptions& /*unused */) {
  check_gpu_single_tensor(input_tensor);
  check_gpu_single_tensor(output_tensor);

  if (input_tensor.dtype() != output_tensor.dtype()) {
    throw std::runtime_error("output tensor must have the same type as input tensor");
  }

  if (input_tensor.numel() * size_ != output_tensor.numel()) {
    throw std::runtime_error("output tensor size must be equal to world_size times input tensor size");
  }

  // just a wrapper to fit the collective interface
  auto inputs = std::vector<at::Tensor> {input_tensor};
  auto outputs = std::vector<at::Tensor> {output_tensor};

  return collective(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        c10::cuda::CUDACachingAllocator::recordStream(
            output.storage().data_ptr(), stream);
        return ncclAllGather(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            getNcclDataType(input.scalar_type()),
            comm,
            stream.stream());
      },
      [&](std::vector<at::cuda::CUDAStream>&) {},
      [&](std::vector<at::cuda::CUDAStream>&) {},
      OpType::_ALLGATHER_BASE,
      "nccl:_all_gather_base");
}

} // namespace c10d
