// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/hash.h>
#include <nccl.h>
#include <torch/csrc/distributed/c10d/nccl2/Logging.hpp>
#include <torch/csrc/distributed/c10d/nccl2/NCCLCachingAllocatorHook.hpp>
#include <torch/csrc/distributed/c10d/nccl2/NCCLMemAllocator.hpp>
#include <algorithm>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace c10d::nccl2 {

namespace {

// Scaling factor for a PREMUL_SUM reduction: either a per-element device tensor
// or a host scalar.
using PreMulSumFactorT = std::variant<at::Tensor, double>;

bool isUnsupportedFloat8(at::ScalarType type) {
  return type == at::ScalarType::Float8_e5m2fnuz ||
      type == at::ScalarType::Float8_e4m3fnuz ||
      type == at::ScalarType::Float8_e8m0fnu
#ifndef NCCL_SUPPORTS_FP8
      || type == at::ScalarType::Float8_e5m2 ||
      type == at::ScalarType::Float8_e4m3fn
#endif
      ;
}

// Extract the scaling factor from a c10d PREMUL_SUM ReduceOp supplement.
PreMulSumFactorT getPreMulSumFactor(const ::c10d::ReduceOp& op) {
  TORCH_CHECK(
      op.supplement_ != nullptr,
      "PREMUL_SUM operation requires a supplement, but none was provided");
  const auto* preMulSupplement =
      dynamic_cast<const ::c10d::NCCLPreMulSumSupplement*>(
          op.supplement_.get());
  TORCH_CHECK(
      preMulSupplement != nullptr,
      "PREMUL_SUM operation supplement must be of type NCCLPreMulSumSupplement");
  if (preMulSupplement->tensor_factor.defined()) {
    return preMulSupplement->tensor_factor;
  }
  return preMulSupplement->double_factor;
}

ncclDataType_t getNcclDataTypeInternal(const at::Tensor& tensor) {
  switch (tensor.scalar_type()) {
    case at::ScalarType::Float:
      return ncclFloat32;
    case at::ScalarType::Double:
      return ncclFloat64;
    case at::ScalarType::Half:
      return ncclFloat16;
    case at::ScalarType::BFloat16:
      return ncclBfloat16;
    case at::ScalarType::Int:
      return ncclInt32;
    case at::ScalarType::Long:
      return ncclInt64;
    case at::ScalarType::Char:
      return ncclInt8;
#ifdef NCCL_SUPPORTS_FP8
    case at::ScalarType::Float8_e5m2:
      return ncclFloat8e5m2;
    case at::ScalarType::Float8_e4m3fn:
      return ncclFloat8e4m3;
#else
    case at::ScalarType::Float8_e5m2:
    case at::ScalarType::Float8_e4m3fn:
#endif
    case at::ScalarType::Byte:
    case at::ScalarType::Bool:
    case at::ScalarType::Float8_e4m3fnuz:
    case at::ScalarType::Float8_e5m2fnuz:
    case at::ScalarType::Float4_e2m1fn_x2:
      return ncclUint8;
    default:
      TORCH_CHECK(false, "Unsupported tensor data type for NCCL");
  }
}

template <typename T, ncclDataType_t dataType>
void createPreMulSum(
    ncclRedOp_t* op,
    const PreMulSumFactorT& factor,
    const ncclComm_t& comm,
    NcclApi* nccl_api) {
  const bool is_tensor = std::holds_alternative<at::Tensor>(factor);
  const auto residence = is_tensor ? ncclScalarDevice : ncclScalarHostImmediate;

  at::Tensor tensor = is_tensor ? std::get<at::Tensor>(factor) : at::Tensor();
  T scalar_factor = is_tensor ? T{} : static_cast<T>(std::get<double>(factor));
  void* scalar = is_tensor ? tensor.data_ptr() : &scalar_factor;

  TORCH_INTERNAL_ASSERT(
      !is_tensor || dataType == getNcclDataTypeInternal(tensor),
      "PreMulSum factor type must match input data type");
  NCCL_CHECK(
      nccl_api,
      comm,
      nccl_api->redOpCreatePreMulSum(op, scalar, dataType, residence, comm),
      "NCCL redOpCreatePreMulSum failed");
}

// This utility function gets a stream that is guaranteed not to be
// part of another stream capture. This is intended to prevent
// problems cited in #185836, due to pytorch's current fixed size pool
// of 32 streams. This will return the currently capturing internal
// stream for current_stream's capture if one already exists.
//
// We maintain a process-lifetime stream pool
// CUDACachingAllocator::recordStream() calls can make a cudaStream_t
// outlive the ProcessGroupNCCL instance.
cudaStream_t getStreamForThisCapture(
    const at::cuda::CUDAStream& current_stream,
    int priority,
    at::cuda::CUDAEvent& dependency_event) {
  const auto capture_info =
      c10::cuda::captureInfoMayInitCtx(current_stream.stream());
  TORCH_INTERNAL_ASSERT(
      capture_info.status == c10::cuda::CaptureStatus::Active,
      "getStreamForThisCapture requires a capturing stream");
  const auto capture_id = capture_info.id;
  const auto device = current_stream.device_index();
  auto sync_stream = [&](const at::cuda::CUDAStream& stream) {
    dependency_event.record(current_stream);
    dependency_event.block(stream);
    return stream.stream();
  };

  static std::mutex pool_mutex;
  using StreamPool = std::vector<at::cuda::CUDAStream>;
  using StreamPoolKey = std::pair<c10::DeviceIndex, int>;
  static ska::flat_hash_map<StreamPoolKey, StreamPool, c10::hash<StreamPoolKey>>
      pools;
  std::lock_guard<std::mutex> lock(pool_mutex);
  auto& streams = pools[{device, priority}];
  std::optional<at::cuda::CUDAStream> idle_stream;
  // Scanning the pool makes N nested captures take O(N^2) queries.
  // We cannot use a hash table keyed on capture id because we have no
  // way to know when a stream exits stream capture. If we could
  // register a callback here that runs when cudaStreamEndCapture()
  // runs, we could fix the quadratic growth.
  for (const auto& stream : streams) {
    auto info = c10::cuda::captureInfoMayInitCtx(stream.stream());
    if (info.status == c10::cuda::CaptureStatus::Active &&
        info.id == capture_id) {
      return sync_stream(stream);
    }
    if (info.status == c10::cuda::CaptureStatus::None && !idle_stream) {
      idle_stream = stream;
    }
  }
  if (!idle_stream) {
    cudaStream_t raw_stream = nullptr;
    C10_CUDA_CHECK(cudaStreamCreateWithPriority(
        &raw_stream, cudaStreamNonBlocking, priority));
    idle_stream = at::cuda::getStreamFromExternal(raw_stream, device);
    streams.push_back(*idle_stream);
  }
  // Join before unlocking so another capture cannot take the same idle stream.
  return sync_stream(*idle_stream);
}

} // namespace

ProcessGroupNCCL::RedOpRAII::RedOpRAII(ncclRedOp_t op) : ncclRedOp_(op) {}

ProcessGroupNCCL::RedOpRAII::RedOpRAII(
    const ::c10d::ReduceOp& op,
    ncclComm_t comm,
    const ncclDataType_t dataType,
    std::shared_ptr<NcclApi> nccl_api)
    : comm_(comm), nccl_api_(std::move(nccl_api)) {
  TORCH_INTERNAL_ASSERT(
      op == ::c10d::ReduceOp::PREMUL_SUM,
      "Constructing premul_sum RedOpRAII with non-premul_sum RedOpType");

  const auto factor = getPreMulSumFactor(op);
  switch (dataType) {
    case ncclFloat16:
      createPreMulSum<at::Half, ncclFloat16>(
          &ncclRedOp_, factor, comm, nccl_api_.get());
      break;
    case ncclFloat32:
      createPreMulSum<float, ncclFloat32>(
          &ncclRedOp_, factor, comm, nccl_api_.get());
      break;
    case ncclBfloat16:
      createPreMulSum<at::BFloat16, ncclBfloat16>(
          &ncclRedOp_, factor, comm, nccl_api_.get());
      break;
    case ncclFloat64:
      createPreMulSum<double, ncclFloat64>(
          &ncclRedOp_, factor, comm, nccl_api_.get());
      break;
    default:
      TORCH_CHECK(
          false, "PreMulSum Data type must be half, float, bfloat16 or double");
  }
}

ProcessGroupNCCL::RedOpRAII::~RedOpRAII() {
  if (comm_) {
    NCCL_CHECK_IGNORE(
        nccl_api_,
        nccl_api_->redOpDestroy(ncclRedOp_, comm_),
        "NCCL redOpDestroy failed");
  }
}

size_t ProcessGroupNCCL::wordSize(ncclDataType_t type) const {
  switch (type) {
    case ncclChar:
#if NCCL_MAJOR >= 2
    // case ncclInt8:
    case ncclUint8:
#endif
#ifdef NCCL_SUPPORTS_FP8
    case ncclFloat8e4m3:
    case ncclFloat8e5m2:
#endif
      return 1;
    case ncclHalf:
    case ncclBfloat16:
      // case ncclFloat16:
      return 2;
    case ncclInt:
    case ncclFloat:
#if NCCL_MAJOR >= 2
    // case ncclInt32:
    case ncclUint32:
      // case ncclFloat32:
#endif
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclDouble:
      // case ncclFloat64:
      return 8;
    default:
      TORCH_CHECK(false, "Unsupported ncclDataType_t in wordSize: ", type);
  }
}

ncclDataType_t ProcessGroupNCCL::getNcclDataType(const at::Tensor& tensor) {
  return getNcclDataTypeInternal(tensor);
}

ProcessGroupNCCL::RedOpRAII ProcessGroupNCCL::getNcclReduceOp(
    const ::c10d::ReduceOp& op,
    ncclComm_t comm,
    const at::Tensor& tensor) {
  TORCH_CHECK(
      !isUnsupportedFloat8(tensor.scalar_type()),
      "Unsupported Float8 type for NCCL reduction");
  TORCH_CHECK(
      tensor.scalar_type() != at::ScalarType::Float4_e2m1fn_x2,
      "Unsupported Float4 type for NCCL reduction");
  if (tensor.scalar_type() == at::kBool) {
    if (op == ::c10d::ReduceOp::SUM) {
      return ncclMax;
    }
    TORCH_CHECK_TYPE(
        op != ::c10d::ReduceOp::AVG,
        "Cannot use ReduceOp.AVG with boolean inputs");
  }

  switch (op) {
    case ::c10d::ReduceOp::SUM:
      return ncclSum;
    case ::c10d::ReduceOp::PRODUCT:
      return ncclProd;
    case ::c10d::ReduceOp::MIN:
      return ncclMin;
    case ::c10d::ReduceOp::MAX:
      return ncclMax;
    case ::c10d::ReduceOp::BAND:
      C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BAND with NCCL");
    case ::c10d::ReduceOp::BOR:
      C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BOR with NCCL");
    case ::c10d::ReduceOp::BXOR:
      C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BXOR with NCCL");
    case ::c10d::ReduceOp::PREMUL_SUM:
      return RedOpRAII(op, comm, getNcclDataType(tensor), nccl_api_);
    case ::c10d::ReduceOp::AVG:
      return ncclAvg;
    default:
      TORCH_CHECK(false, "Unsupported reduce operation");
  }
}

void ProcessGroupNCCL::checkWorkQueue() {
  drainRetiredGraphWork();
  WorkNCCL::WorkStatus status = workq_.garbageCollect();

  // Abort hooks run where a failure is DETECTED, not only where the process is
  // torn down, because the teardown paths run no hook at all in the
  // configurations where the process survives the failure and a post-mortem is
  // worth the most: abortProcess() returns early when
  // abort_process_on_timeout_or_error_ is off or reconfigure is on. So despite
  // the name, a hook may run here with no abort following it. That matches what
  // the hooks are for (capture debug info about the failure, e.g.
  // c10d::FlightRecorderHook writes its trace) and callers must tolerate being
  // called more than once per failure, since the teardown paths still fire.
  switch (status) {
    case WorkNCCL::WorkStatus::TIMEDOUT:
      comm_state_ = CommState::TIMEOUT;
      runAbortHooks();
      break;
    case WorkNCCL::WorkStatus::ERROR:
      comm_state_ = CommState::ERROR;
      runAbortHooks();
      break;
    default:
      // For COMPLETED, NOT_STARTED, and INPROGRESS, no state change needed
      break;
  }
}

// Retire completed work and graph states and check for timeouts.
void ProcessGroupNCCL::timeoutWatchdog() noexcept {
  TC_LOG(INFO, this) << "Timeout thread starting for rank: " << rank_;

  // Honor the noexcept contract: the loop issues NCCL probes (NCCL_CHECK) and
  // abort paths that can throw; swallow here so nothing escapes this thread.
  try {
    // A new thread defaults to device 0; even setting capture mode can create
    // a CUDA context, so bind the communicator device first.
    c10::cuda::CUDAGuard device_guard(device_);
    c10::cuda::CUDAStreamCaptureModeGuard capture_mode_guard(
        cudaStreamCaptureModeThreadLocal);
    while (!shutdown_) {
      {
        std::unique_lock<std::mutex> lock(timeout_mutex_);
        // Wait for a shorter interval to check work objects periodically
        // Wake up either after 1 second or immediately if shutdown is requested
        timeout_cv_.wait_for(lock, std::chrono::seconds(1), [this]() {
          return shutdown_.load();
        });

        // If we're shutting down, exit the loop
        if (shutdown_) {
          break;
        }
      }

      // Check work objects for completion or timeout
      // Thread-safety: checkWorkQueue() calls garbageCollect() which acquires
      // work_queues_mutex_ before accessing the work queue, ensuring safe
      // concurrent access with the main thread's enqueueWork() calls.
      //
      // NOTE: garbageCollect may pop a completed work item whose destruction
      // releases the last shared_ptr to this comm, triggering our destructor.
      // In that case, the destructor sets shutdown_=true and detaches this
      // thread. We must check shutdown_ immediately after to avoid accessing
      // potentially destroyed member state.
      checkWorkQueue();
      if (shutdown_) {
        break;
      }
      if (comm_state_ != CommState::NORMAL) {
        handleWatchdogFailure(
            comm_state_ == CommState::TIMEOUT
                ? "timeout - timeout watchdog detected operation timeout"
                : "error - timeout watchdog detected operation error");
      }

      // Detect a communicator-level async error while the comm is still
      // healthy.
      if (comm_state_ == CommState::NORMAL) {
        ncclResult_t asyncErr{};
        NCCL_CHECK(
            nccl_api_,
            nccl_comm_,
            nccl_api_->commGetAsyncError(nccl_comm_, &asyncErr),
            "failed to get async error");
        if (asyncErr != ncclSuccess && asyncErr != ncclInProgress) {
          comm_state_ = CommState::ERROR;
          // Detected here rather than through the work queue, so this needs its
          // own notification; see checkWorkQueue() for why detection and not
          // just teardown.
          runAbortHooks();
          if (!options_c10d_->enable_reconfigure) {
            TC_LOG(ERROR, this) << "nccl hit async error on rank " << rank_
                                << ": " << ncclGetErrorString(asyncErr);
          } else {
            // Revoked below by the reconfigurable-mode handler.
            TC_LOG(ERROR, this)
                << "Async error on rank " << rank_ << ": "
                << ncclGetErrorString(asyncErr) << " (reconfigurable mode)";
          }
          handleWatchdogFailure(
              std::string("error - nccl hit async error: ") +
              ncclGetErrorString(asyncErr));
        }
      }
    }
  } catch (const std::exception& e) {
    TC_LOG(ERROR, this) << "Timeout watchdog caught exception: " << e.what();
  } catch (...) {
    TC_LOG(ERROR, this) << "Timeout watchdog caught unknown exception.";
  }

  TC_LOG(INFO, this) << "Timeout thread exiting for rank: " << rank_;
}

void ProcessGroupNCCL::checkInitialized() {
  if (options_c10d_->lazy_init &&
      init_state_ == InitializationState::UNINITIALIZED &&
      getBoundDeviceId().has_value()) {
    ensureInitialized(*getBoundDeviceId());
  }
  TORCH_CHECK(
      init_state_ == InitializationState::INITIALIZED,
      options_c10d_->enable_reconfigure
          ? "ProcessGroupNCCL has not been initialized. Call reconfigure() "
            "before issuing operations when enable_reconfigure=True."
          : "ProcessGroupNCCL not initialized");
}

void ProcessGroupNCCL::checkAndAbortIfTimedOutOrError() {
  // Nothing to check in graph capture mode
  if (getGraphCaptureMode()) {
    return;
  }

  // First, check work queue status
  checkWorkQueue();

  if (comm_state_ == CommState::TIMEOUT) {
    if (options_c10d_->enable_reconfigure) {
      revokeNcclComm();
      TORCH_CHECK(false, "NCCL operation timed out");
    } else {
      handleWatchdogFailure("timeout - collective operation timed out");
      TORCH_CHECK(false, "NCCL operation timed out");
    }
  } else if (comm_state_ == CommState::ERROR) {
    // CleanUpOnly may have already removed the communicator on the watchdog
    // thread, so a later collective cannot query the original NCCL error.
    TORCH_CHECK(
        nccl_comm_, "NCCL communicator was aborted after a previous error");
    ncclResult_t asyncErr{};
    NCCL_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->commGetAsyncError(nccl_comm_, &asyncErr),
        "failed to get async error");
    NCCLException ncclException(
        *nccl_api_, "NCCL Async Error", asyncErr, nccl_comm_);
    if (options_c10d_->enable_reconfigure) {
      // In reconfigurable mode we never abort the process: revoke the comm so
      // it can be reconfigured and surface the error to the caller.
      revokeNcclComm();
      // The constructor reads the communicator's last error, which the
      // commRevoke() inside revokeNcclComm() overwrites, so the exception has
      // to be built first. A check macro would raise before the revoke ran.
      // @allow-raw-throw: the revoke above clobbers its last error
      throw std::move(ncclException);
    }
    handleWatchdogFailure(std::string("error - ") + ncclException.what());
    // @allow-raw-throw: its what() is an argument to the call above
    throw std::move(ncclException);
  }
}

bool ProcessGroupNCCL::getGraphCaptureMode() {
  auto current_stream = at::cuda::getCurrentCUDAStream(device_.index());
  return c10::cuda::isStreamCapturingMayInitCtx(current_stream);
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::createWork(
    cudaStream_t stream,
    std::chrono::milliseconds timeout,
    const std::vector<at::Tensor>& inputTensors) {
  // Only create the work object without enqueuing it
  auto [workTimeout, ownedTimeout] = applyEphemeralTimeout(timeout);
  auto work =
      c10::make_intrusive<WorkNCCL>(this, stream, workTimeout, inputTensors);
  work->setOwnedEphemeralTimeout(ownedTimeout);
  work->setSequenceNumber(sequence_number_);
  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::createWork(
    cudaStream_t stream,
    std::chrono::milliseconds timeout,
    const at::Tensor& inputTensor) {
  // Single-tensor overload to avoid vector allocation
  auto [workTimeout, ownedTimeout] = applyEphemeralTimeout(timeout);
  auto work =
      c10::make_intrusive<WorkNCCL>(this, stream, workTimeout, inputTensor);
  work->setOwnedEphemeralTimeout(ownedTimeout);
  work->setSequenceNumber(sequence_number_);
  return work;
}

std::pair<std::chrono::milliseconds, std::chrono::milliseconds>
ProcessGroupNCCL::applyEphemeralTimeout(std::chrono::milliseconds timeout) {
  std::lock_guard<std::mutex> lock(ephemeral_timeout_mutex_);
  timeout += ephemeral_timeout_active_;
  auto ownedTimeout = ephemeral_timeout_active_ - ephemeral_timeout_inflight_;
  ephemeral_timeout_inflight_ = ephemeral_timeout_active_;
  return {timeout, ownedTimeout};
}

void ProcessGroupNCCL::releaseEphemeralTimeout(
    std::chrono::milliseconds timeout) {
  std::lock_guard<std::mutex> lock(ephemeral_timeout_mutex_);
  ephemeral_timeout_active_ -= timeout;
  ephemeral_timeout_inflight_ -= timeout;
}

void ProcessGroupNCCL::addEphemeralTimeout(
    const std::chrono::milliseconds& timeout) {
  std::lock_guard<std::mutex> lock(ephemeral_timeout_mutex_);
  ephemeral_timeout_active_ += timeout;
}

void ProcessGroupNCCL::enqueueWork(
    const c10::intrusive_ptr<WorkNCCL>& work,
    cudaStream_t stream) {
  if (coalescing_batch_) {
    coalescing_works_.push_back(work);
  }

  // In graph capture mode, keep the completion state and events alive until
  // the graph gets destroyed, organized per graph.
  if (getGraphCaptureMode()) {
    auto capture_info = c10::cuda::captureInfoMayInitCtx(stream);
    if (capture_info.status == c10::cuda::CaptureStatus::Active) {
      bool is_first_work = false;
      {
        std::lock_guard<std::mutex> lock(graph_capture_work_mutex_);
        auto& graph_work_refs = graph_capture_work_refs_[capture_info.id];
        is_first_work = graph_work_refs.empty();
        graph_work_refs.push_back(work->state_);
      }

      // If this is the first work object for this graph, set up automatic
      // cleanup. Do not hold graph_capture_work_mutex_ here: HIP runs
      // graphCleanupCallback while holding a runtime lock that the user object
      // APIs called by retainGraphUserObject also take.
      if (is_first_work) {
        c10::cuda::retainGraphUserObject(
            capture_info.graph,
            std::make_unique<GraphCleanupData>(this, capture_info.id),
            graphCleanupCallback);
      }
    }
  } else {
    // Add work to stream's queue after events have been recorded
    workq_.enqueueWork(work, stream);
  }
}

// Static callback function for CUDA user object cleanup
void ProcessGroupNCCL::graphCleanupCallback(void* userData) {
  auto* cleanup_data = static_cast<GraphCleanupData*>(userData);
  TORCH_CHECK(
      cleanup_data != nullptr && cleanup_data->comm != nullptr,
      "Invalid cleanup data");

  // CUDA user-object callbacks cannot call CUDA APIs, including event destroy.
  std::lock_guard<std::mutex> lock(
      cleanup_data->comm->graph_capture_work_mutex_);
  auto& comm = *cleanup_data->comm;
  auto it = comm.graph_capture_work_refs_.find(cleanup_data->graph_id);
  if (it != comm.graph_capture_work_refs_.end()) {
    comm.retired_graph_work_refs_.emplace_back(std::move(it->second));
    comm.graph_capture_work_refs_.erase(it);
  }

  // Clean up the cleanup data itself
  delete cleanup_data;
}

void ProcessGroupNCCL::drainRetiredGraphWork() {
  std::list<std::vector<std::shared_ptr<WorkNCCL::State>>> retired;
  {
    std::lock_guard<std::mutex> lock(graph_capture_work_mutex_);
    retired.swap(retired_graph_work_refs_);
  }
}

cudaStream_t ProcessGroupNCCL::getOperationStream(bool async_op) {
  c10::cuda::CUDAGuard gpuGuard(device_);
  auto device = device_.index();
  auto current_stream = at::cuda::getCurrentCUDAStream(device);
  if (!async_op) {
    return current_stream.stream();
  }
  TORCH_CHECK(
      dependency_event_.has_value() && internal_stream_.has_value(),
      "NCCL stream resources are not initialized");
  auto& dependency_event = dependency_event_.value();
  const auto& internal_stream = internal_stream_.value();
  auto sync_stream = [&](const at::cuda::CUDAStream& stream) {
    dependency_event.record(current_stream);
    dependency_event.block(stream);
    return stream.stream();
  };

  if (C10_LIKELY(!c10::cuda::CUDACachingAllocator::isCaptureContext(device))) {
    return sync_stream(internal_stream);
  }

  // Otherwise, current stream is capturing. Put the internal stream
  // into capture mode if it isn't already in capture mode for this
  // capture.
  auto cur_info = c10::cuda::captureInfoMayInitCtx(current_stream.stream());
  auto nccl_info = c10::cuda::captureInfoMayInitCtx(internal_stream.stream());
  if (nccl_info.status != c10::cuda::CaptureStatus::Active ||
      nccl_info.id == cur_info.id) {
    return sync_stream(internal_stream);
  }

  // We reach this point only if more than one stream capture is
  // happening at once. See getStreamForThisCapture() comment for
  // details.
  return getStreamForThisCapture(
      current_stream, internal_stream.priority(), dependency_event);
}

void ProcessGroupNCCL::ensureTensorContiguous(const at::Tensor& tensor) {
  if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
    C10_THROW_ERROR(ValueError, "Tensors must be contiguous");
  }
}

void ProcessGroupNCCL::checkTensorDevice(const at::Tensor& tensor) const {
  TORCH_CHECK(
      tensor.device() == device_,
      "Expected tensor on ",
      device_,
      " but found tensor on ",
      tensor.device());
}

void ProcessGroupNCCL::checkTensorsDevice(
    const std::vector<at::Tensor>& tensors) const {
  for (const auto& t : tensors) {
    checkTensorDevice(t);
  }
}

void ProcessGroupNCCL::enableCollectivesTiming() {
  event_pool_->enableTiming();
}

void ProcessGroupNCCL::attachMemoryHook() {
  NCCLCachingAllocatorHook::getInstance().registerComm(this);
  NCCLMemAllocator::global()->attach(*this);
  std::lock_guard<std::mutex> lock(symm_allocator_mutex_);
  if (symm_allocator_) {
    symm_allocator_->attach(*this);
  }
}

void ProcessGroupNCCL::detachMemoryHook() {
  // Windows go first: deregisterComm only drops the plain registrations.
  {
    std::lock_guard<std::mutex> lock(symm_allocator_mutex_);
    if (symm_allocator_) {
      symm_allocator_->detach(*this);
    }
  }
  NCCLMemAllocator::global()->detach(*this);
  NCCLCachingAllocatorHook::getInstance().deregisterComm(this);
}

void ProcessGroupNCCL::register_address(void* addr, size_t len) {
  if (nccl_comm_ == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(memory_registration_mutex_);
  auto [it, inserted] = memoryRegistrationHandles_.try_emplace(
      addr, RegistrationHandle{nullptr, nullptr, len});
  TORCH_CHECK(
      it->second.regHandle == nullptr, "Memory already registered with NCCL");
  void* handle = nullptr;
  auto rc = nccl_api_->commRegister(nccl_comm_, addr, len, &handle);
  if (rc != ncclSuccess && inserted) {
    memoryRegistrationHandles_.erase(it);
  }
  NCCL_CHECK(nccl_api_, nccl_comm_, rc, "Failed to register memory with NCCL");
  it->second.regHandle = handle;
}

void ProcessGroupNCCL::deregister_address(void* addr) {
  std::lock_guard<std::mutex> lock(memory_registration_mutex_);
  auto it = memoryRegistrationHandles_.find(addr);
  if (it == memoryRegistrationHandles_.end() ||
      it->second.regHandle == nullptr) {
    return;
  }
  void* handle = std::exchange(it->second.regHandle, nullptr);
  if (it->second.winHandle == nullptr) {
    memoryRegistrationHandles_.erase(it);
  }
  // Without a communicator the handle died with it.
  if (nccl_comm_ != nullptr) {
    NCCL_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->commDeregister(nccl_comm_, handle),
        "Failed to deregister memory with NCCL");
  }
}

void ProcessGroupNCCL::registerWindow(
    void* addr,
    size_t len,
    c10::DeviceIndex device) {
  if (nccl_comm_ == nullptr) {
    return;
  }
  TORCH_CHECK(
      device == device_.index(),
      "Cannot register a MemPool segment on cuda:",
      static_cast<int>(device),
      " with a process group bound to ",
      device_);
  std::lock_guard<std::mutex> lock(memory_registration_mutex_);
  auto [it, inserted] = memoryRegistrationHandles_.try_emplace(
      addr, RegistrationHandle{nullptr, nullptr, len});
  auto rc = ensureSymmetricWindow(it);
  if (rc != ncclSuccess && inserted) {
    memoryRegistrationHandles_.erase(it);
  }
  TORCH_CHECK(
      rc != ncclInvalidUsage,
      "Cannot register segment ",
      addr,
      " of a symm=True MemPool as an NCCL symmetric window: this communicator "
      "does not support symmetric memory (no symmetric-capable transport, or "
      "NCCL_WIN_ENABLE=0)");
  TORCH_CHECK(
      rc == ncclSuccess,
      "Failed to register segment ",
      addr,
      " as an NCCL symmetric window: ",
      nccl_api_->getErrorString(rc));
}

void ProcessGroupNCCL::deregisterWindow(void* addr) {
  std::lock_guard<std::mutex> lock(memory_registration_mutex_);
  auto it = memoryRegistrationHandles_.find(addr);
  if (it == memoryRegistrationHandles_.end() ||
      it->second.winHandle == nullptr) {
    return;
  }
  auto win = std::exchange(it->second.winHandle, nullptr);
  if (it->second.regHandle == nullptr) {
    memoryRegistrationHandles_.erase(it);
  }
  if (nccl_comm_ != nullptr) {
    NCCL_CHECK_IGNORE(
        nccl_api_,
        nccl_api_->commWindowDeregister(nccl_comm_, win),
        "ncclCommWindowDeregister failed for segment");
  }
}

ProcessGroupNCCL::RegistrationMap::iterator ProcessGroupNCCL::
    findContainingRegistrationLocked(const void* ptr) {
  const auto target = reinterpret_cast<uintptr_t>(ptr);
  // memoryRegistrationHandles_ is sorted by base address. upper_bound + step
  // back finds the segment whose base is less than or equal to target.
  auto it = memoryRegistrationHandles_.upper_bound(ptr);
  if (it == memoryRegistrationHandles_.begin()) {
    return memoryRegistrationHandles_.end();
  }
  --it;
  const auto base = reinterpret_cast<uintptr_t>(it->first);
  if (target < base || target - base >= it->second.len) {
    return memoryRegistrationHandles_.end();
  }
  return it;
}

std::pair<ncclWindow_t, size_t> ProcessGroupNCCL::lookupSegmentWindow(
    const void* ptr) {
  std::lock_guard<std::mutex> lock(memory_registration_mutex_);
  auto it = findContainingRegistrationLocked(ptr);
  if (it == memoryRegistrationHandles_.end() ||
      it->second.winHandle == nullptr) {
    return {nullptr, 0};
  }
  const auto target = reinterpret_cast<uintptr_t>(ptr);
  const auto base = reinterpret_cast<uintptr_t>(it->first);
  return {it->second.winHandle, target - base};
}

ncclResult_t ProcessGroupNCCL::ensureSegmentWindow(const void* ptr) {
  if (nccl_comm_ == nullptr) {
    return ncclInvalidUsage;
  }
  std::lock_guard<std::mutex> lock(memory_registration_mutex_);
  auto it = findContainingRegistrationLocked(ptr);
  if (it == memoryRegistrationHandles_.end()) {
    return ncclInvalidArgument;
  }
  if (it->second.winHandle == nullptr &&
      !NCCLMemAllocator::global()->ownsSegment(it->first)) {
    return ncclInvalidArgument;
  }
  return ensureSymmetricWindow(it);
}

ncclResult_t ProcessGroupNCCL::ensureSymmetricWindow(
    RegistrationMap::iterator it) {
  if (it->second.winHandle != nullptr) {
    return ncclSuccess;
  }
  ncclWindow_t win = nullptr;
  auto rc = nccl_api_->commWindowRegister(
      nccl_comm_, it->first, it->second.len, &win, NCCL_WIN_COLL_SYMMETRIC);
  if (rc == ncclInProgress) {
    // NCCL retains &win in its async registration task, so keep this stack
    // frame alive until the task has populated the handle.
    waitForNcclCompletion(
        *nccl_api_,
        nccl_comm_,
        rc,
        options_c10d_->timeout,
        "NCCL symmetric window registration failed");
    rc = ncclSuccess;
  }
  if (rc != ncclSuccess) {
    return rc;
  }
  if (win == nullptr) {
    // NCCL returned success but left the window handle unset. Observed on
    // configurations without a transport capable of symmetric memory (no
    // NVLink and no InfiniBand). Treat as unsupported so callers can surface
    // a meaningful error or skip.
    return ncclInvalidUsage;
  }
  it->second.winHandle = win;
  return ncclSuccess;
}

void ProcessGroupNCCL::registerMemPool(at::cuda::MemPool* pool, bool symm) {
  auto allocator =
      std::dynamic_pointer_cast<NCCLMemAllocator>(pool->allocator());
  const bool poolSymm = allocator && allocator->symm();
  const char* fix = symm ? "torch.cuda.MemPool(backend.allocator(symm=True))"
                         : "torch.cuda.MemPool(backend.allocator(symm=False))";
  TORCH_CHECK(
      poolSymm == symm,
      "MemPool was created with symm=",
      poolSymm ? "True" : "False",
      " but registered with symm=",
      symm ? "True" : "False",
      "; create it with ",
      fix);
  if (symm) {
    std::lock_guard<std::mutex> lock(symm_allocator_mutex_);
    TORCH_CHECK(
        allocator == symm_allocator_,
        "MemPool was created by another process group's allocator; create it "
        "with ",
        fix);
  }
}

void ProcessGroupNCCL::deregisterMemPool(at::cuda::MemPool* /*pool*/) {}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
