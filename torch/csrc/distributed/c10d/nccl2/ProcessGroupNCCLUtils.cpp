// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>

#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <nccl.h>
#include <torch/csrc/distributed/c10d/nccl2/Logging.hpp>
#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCLCCA.hpp>
#include <stdexcept>
#include <string>
#include <variant>

namespace c10d::nccl2 {

namespace {

// Scaling factor for a PREMUL_SUM reduction: either a per-element device tensor
// or a host scalar.
using PreMulSumFactorT = std::variant<at::Tensor, double>;

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
#if HAVE_FP8
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
      return ncclUint8;
    default:
      throw std::runtime_error("Unsupported tensor data type for NCCL");
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
      is_tensor ? dataType == getNcclDataTypeInternal(tensor)
                : dataType != ncclBfloat16,
      "PreMulSum factor type must match input data type");
  NCCL_CHECK(
      nccl_api,
      comm,
      nccl_api->redOpCreatePreMulSum(op, scalar, dataType, residence, comm),
      "NCCL redOpCreatePreMulSum failed");
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
      createPreMulSum<float, ncclBfloat16>(
          &ncclRedOp_, factor, comm, nccl_api_.get());
      break;
    case ncclFloat64:
      createPreMulSum<double, ncclFloat64>(
          &ncclRedOp_, factor, comm, nccl_api_.get());
      break;
    default:
      throw std::runtime_error(
          "PreMulSum Data type must be half, float, bfloat16 or double");
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
#if HAVE_FP8
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
      throw std::runtime_error(
          "Unsupported ncclDataType_t in wordSize: " + std::to_string(type));
  }
}

ncclDataType_t ProcessGroupNCCL::getNcclDataType(const at::Tensor& tensor) {
  return getNcclDataTypeInternal(tensor);
}

ProcessGroupNCCL::RedOpRAII ProcessGroupNCCL::getNcclReduceOp(
    const ::c10d::ReduceOp& op,
    ncclComm_t comm,
    const ncclDataType_t dataType) {
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
      TORCH_CHECK(false, "Cannot use ReduceOp.BAND with NCCL");
    case ::c10d::ReduceOp::BOR:
      TORCH_CHECK(false, "Cannot use ReduceOp.BOR with NCCL");
    case ::c10d::ReduceOp::BXOR:
      TORCH_CHECK(false, "Cannot use ReduceOp.BXOR with NCCL");
    case ::c10d::ReduceOp::PREMUL_SUM:
      return RedOpRAII(op, comm, dataType, nccl_api_);
    case ::c10d::ReduceOp::AVG:
      return ncclAvg;
    default:
      TORCH_CHECK(false, "Unsupported reduce operation");
  }
}

void ProcessGroupNCCL::checkWorkQueue() {
  WorkNCCL::WorkStatus status = workq_.garbageCollect();

  switch (status) {
    case WorkNCCL::WorkStatus::TIMEDOUT:
      comm_state_ = CommState::TIMEOUT;
      break;
    case WorkNCCL::WorkStatus::ERROR:
      comm_state_ = CommState::ERROR;
      break;
    default:
      // For COMPLETED, NOT_STARTED, and INPROGRESS, no state change needed
      break;
  }
}

// The timeout thread cannot make NCCL calls.  The only CUDA call it can make
// it cudaEventQuery.
void ProcessGroupNCCL::timeoutWatchdog() noexcept {
  TC_LOG(INFO, this) << "Timeout thread starting for rank: " << rank_;

  // Honor the noexcept contract: the loop issues NCCL probes (NCCL_CHECK) and
  // abort paths that can throw; swallow here so nothing escapes this thread.
  try {
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
      if (comm_state_ != CommState::NORMAL &&
          options_c10d_->abort_process_on_timeout_or_error &&
          !options_c10d_->enable_reconfigure) {
        if (comm_state_ == CommState::TIMEOUT) {
          TC_LOG(ERROR, this)
              << "Aborting process due to timeout on rank " << rank_
              << " - timeout watchdog detected operation timeout";
        } else if (comm_state_ == CommState::ERROR) {
          TC_LOG(ERROR, this)
              << "Aborting process due to error on rank " << rank_
              << " - timeout watchdog detected operation error. ";
        }

        runAbortHooks();

        ::abort();
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
        if (asyncErr != ncclSuccess) {
          comm_state_ = CommState::ERROR;
          if (!options_c10d_->enable_reconfigure) {
            TC_LOG(ERROR, this)
                << "Aborting process due to error on rank " << rank_
                << " - nccl hit async error: " << ncclGetErrorString(asyncErr);

            runAbortHooks();

            abort();
          } else {
            // Revoked below by the reconfigurable-mode handler.
            TC_LOG(ERROR, this)
                << "Async error on rank " << rank_ << ": "
                << ncclGetErrorString(asyncErr) << " (reconfigurable mode)";
          }
        }
      }

      // In reconfigurable mode, gracefully revoke the communicator on any
      // failure
      // -- timeout or error, whether surfaced by the work queue or an async
      // comm error -- so in-flight operations are stopped and the comm can
      // later be reconfigured. This is the only revoke path under CUDA graph
      // replay, where no synchronous collective reaches
      // checkAndAbortIfTimedOutOrError(); isAborted() then reports the revoked
      // state to the caller. revokeNcclComm() is idempotent and the revoked_
      // check keeps the watchdog from logging every iteration.
      if (comm_state_ != CommState::NORMAL &&
          options_c10d_->enable_reconfigure && !revoked_.load()) {
        TC_LOG(ERROR, this)
            << "Revoking communicator on rank " << rank_
            << " - watchdog detected "
            << (comm_state_ == CommState::TIMEOUT ? "timeout" : "error")
            << " (reconfigurable mode)";
        revokeNcclComm();
      }
    }
  } catch (const std::exception& e) {
    TC_LOG(ERROR, this) << "Timeout watchdog caught exception: " << e.what();
  } catch (...) {
    TC_LOG(ERROR, this) << "Timeout watchdog caught unknown exception.";
  }

  TC_LOG(INFO, this) << "Timeout thread exiting for rank: " << rank_;
}

void ProcessGroupNCCL::checkInitialized() const {
  if (init_state_ != InitializationState::INITIALIZED) {
    throw std::runtime_error("ProcessGroupNCCL not initialized");
  }
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
      throw std::runtime_error("NCCL operation timed out");
    } else {
      abortNcclComm();
      if (options_c10d_->abort_process_on_timeout_or_error) {
        TC_LOG(ERROR, this) << "Aborting process due to timeout";
        runAbortHooks();
        ::abort();
      } else {
        throw std::runtime_error("NCCL operation timed out");
      }
    }
  } else if (comm_state_ == CommState::ERROR) {
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
      throw std::move(ncclException);
    }
    abortNcclComm();
    if (options_c10d_->abort_process_on_timeout_or_error) {
      TC_LOG(ERROR, this) << "Aborting process due to error: "
                          << ncclException.what();
      runAbortHooks();
      ::abort();
    } else {
      throw std::move(ncclException);
    }
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
  auto work =
      c10::make_intrusive<WorkNCCL>(this, stream, timeout, inputTensors);
  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::createWork(
    cudaStream_t stream,
    std::chrono::milliseconds timeout,
    const at::Tensor& inputTensor) {
  // Single-tensor overload to avoid vector allocation
  auto work = c10::make_intrusive<WorkNCCL>(this, stream, timeout, inputTensor);
  return work;
}

void ProcessGroupNCCL::enqueueWork(
    c10::intrusive_ptr<WorkNCCL> work,
    cudaStream_t stream) {
  // In graph capture mode, keep a reference to the work object to prevent
  // premature destruction until the graph gets destroyed, organized per graph
  if (getGraphCaptureMode()) {
    auto capture_info = c10::cuda::captureInfoMayInitCtx(stream);
    if (capture_info.status == c10::cuda::CaptureStatus::Active) {
      std::lock_guard<std::mutex> lock(graph_capture_work_mutex_);

      // Check if this is the first work object for this graph
      bool is_first_work = graph_capture_work_refs_[capture_info.id].empty();

      // Add work reference to the per-graph container
      graph_capture_work_refs_[capture_info.id].push_back(work);

      // If this is the first work object for this graph, set up automatic
      // cleanup
      if (is_first_work) {
        c10::cuda::retainGraphUserObject(
            capture_info.graph,
            std::make_unique<GraphCleanupData>(this, capture_info.id),
            graphCleanupCallback);
      }
    }
  } else {
    // Add work to stream's queue after events have been recorded
    workq_.enqueueWork(std::move(work), stream);
  }
}

// Static callback function for CUDA user object cleanup
void ProcessGroupNCCL::graphCleanupCallback(void* userData) {
  auto* cleanup_data = static_cast<GraphCleanupData*>(userData);
  if (cleanup_data == nullptr || cleanup_data->comm == nullptr) {
    throw std::runtime_error("Invalid cleanup data");
  }

  // Clear the work references for this graph
  std::lock_guard<std::mutex> lock(
      cleanup_data->comm->graph_capture_work_mutex_);
  cleanup_data->comm->graph_capture_work_refs_.erase(cleanup_data->graph_id);

  // Clean up the cleanup data itself
  delete cleanup_data;
}

cudaStream_t ProcessGroupNCCL::getOperationStream(bool async_op) {
  // c10d does not guarantee the ambient CUDA device matches this comm's device
  // (unlike upstream torchcomms, which ran with the device already set). Pin it
  // here -- the first call in every collective -- so subsequent event/record
  // ops in this op target device_ (events are pooled per device_).
  c10::cuda::set_device(device_.index());
  if (async_op) {
    auto current_stream = at::cuda::getCurrentCUDAStream(device_.index());
    if (!dependency_event_.has_value() || !internal_stream_.has_value()) {
      throw std::runtime_error("NCCL stream resources are not initialized");
    }
    auto& dependency_event = dependency_event_.value();
    auto& internal_stream = internal_stream_.value();

    dependency_event.record(current_stream);
    dependency_event.block(internal_stream);

    return internal_stream.stream();
  } else {
    return at::cuda::getCurrentCUDAStream(device_.index()).stream();
  }
}

void ProcessGroupNCCL::ensureTensorContiguous(const at::Tensor& tensor) {
  if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
    C10_THROW_ERROR(ValueError, "Tensors must be contiguous");
  }
}

void ProcessGroupNCCL::checkTensorDevice(const at::Tensor& tensor) const {
  TORCH_CHECK(
      tensor.device().type() == device_.type(),
      "Expected tensor on ",
      device_.type(),
      " but found tensor on ",
      tensor.device());
}

void ProcessGroupNCCL::checkTensorsDevice(
    const std::vector<at::Tensor>& tensors) const {
  for (const auto& t : tensors) {
    checkTensorDevice(t);
  }
}

// Protected methods (not in the private section of the header)
std::unique_ptr<at::cuda::CUDAEvent> ProcessGroupNCCL::getEvent() {
  std::lock_guard<std::mutex> lock(event_pool_mutex_);

  if (!event_pool_.empty()) {
    auto event = std::move(event_pool_.front());
    event_pool_.pop();
    return event;
  }

  return std::make_unique<at::cuda::CUDAEvent>(cudaEventDisableTiming);
}

void ProcessGroupNCCL::returnEvent(std::unique_ptr<at::cuda::CUDAEvent> event) {
  std::lock_guard<std::mutex> lock(event_pool_mutex_);

  if (event_pool_.size() < max_event_pool_size_) {
    event_pool_.push(std::move(event));
  }
}

void ProcessGroupNCCL::attachMemoryHook() {
  NcclCachingAllocatorHook::getInstance().registerComm(this);
}

void ProcessGroupNCCL::detachMemoryHook() {
  NcclCachingAllocatorHook::getInstance().deregisterComm(this);
}

void ProcessGroupNCCL::register_address(void* addr, size_t len) {
  if (nccl_comm_ == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(memory_registration_mutex_);
  TORCH_CHECK(
      !memoryRegistrationHandles_.count(addr),
      "Memory already registered with NCCL");
  void* handle = nullptr;
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commRegister(nccl_comm_, addr, len, &handle),
      "Failed to register memory with NCCL");
  // Symmetric-window (NCCL_WIN_COLL_SYMMETRIC) registration is collective and
  // cannot run from the allocator hook, which fires on arbitrary threads. It
  // happens lazily in ensureSegmentWindow(), keyed by the base recorded here.
  memoryRegistrationHandles_.emplace(
      addr, RegistrationHandle{handle, nullptr, len});
}

void ProcessGroupNCCL::deregister_address(void* addr) {
  if (nccl_comm_ == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(memory_registration_mutex_);
  auto it = memoryRegistrationHandles_.find(addr);
  if (it == memoryRegistrationHandles_.end()) {
    return;
  }
  if (it->second.winHandle != nullptr) {
    NCCL_CHECK_IGNORE(
        nccl_api_,
        nccl_api_->commWindowDeregister(nccl_comm_, it->second.winHandle),
        "ncclCommWindowDeregister failed for segment");
  }
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commDeregister(nccl_comm_, it->second.regHandle),
      "Failed to deregister memory with NCCL");
  memoryRegistrationHandles_.erase(it);
}

std::pair<ncclWindow_t, size_t> ProcessGroupNCCL::lookupSegmentWindow(
    const void* ptr) {
  std::lock_guard<std::mutex> lock(memory_registration_mutex_);
  const auto target = reinterpret_cast<uintptr_t>(ptr);
  // memoryRegistrationHandles_ is sorted by base address; upper_bound + step
  // back finds the segment whose base <= target.
  auto it = memoryRegistrationHandles_.upper_bound(ptr);
  if (it == memoryRegistrationHandles_.begin()) {
    return {nullptr, 0};
  }
  --it;
  const auto base = reinterpret_cast<uintptr_t>(it->first);
  if (target >= base + it->second.len || it->second.winHandle == nullptr) {
    return {nullptr, 0};
  }
  return {it->second.winHandle, target - base};
}

ncclResult_t ProcessGroupNCCL::ensureSegmentWindow(const void* ptr) {
  if (nccl_comm_ == nullptr) {
    return ncclInvalidUsage;
  }
  std::lock_guard<std::mutex> lock(memory_registration_mutex_);
  const auto target = reinterpret_cast<uintptr_t>(ptr);
  auto it = memoryRegistrationHandles_.upper_bound(ptr);
  if (it == memoryRegistrationHandles_.begin()) {
    return ncclInvalidArgument;
  }
  --it;
  const auto base = reinterpret_cast<uintptr_t>(it->first);
  if (target >= base + it->second.len) {
    return ncclInvalidArgument;
  }
  if (it->second.winHandle != nullptr) {
    return ncclSuccess;
  }
  ncclWindow_t win = nullptr;
  auto rc = nccl_api_->commWindowRegister(
      nccl_comm_, it->first, it->second.len, &win, NCCL_WIN_COLL_SYMMETRIC);
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

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
