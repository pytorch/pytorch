// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>

#include <nccl.h>
#include <torch/csrc/distributed/c10d/nccl2/Logging.hpp>
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
    case at::ScalarType::Byte:
      return ncclUint8;
    case at::ScalarType::Bool:
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

ProcessGroupNCCL::RedOpRAII::RedOpRAII(ncclRedOp_t op)
    : ncclRedOp_(op), comm_(nullptr) {}

ProcessGroupNCCL::RedOpRAII::RedOpRAII(
    const ::c10d::ReduceOp& op,
    const ncclComm_t comm,
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
    const ncclComm_t comm,
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
      throw std::runtime_error("Cannot use ReduceOp.BAND with NCCL");
    case ::c10d::ReduceOp::BOR:
      throw std::runtime_error("Cannot use ReduceOp.BOR with NCCL");
    case ::c10d::ReduceOp::BXOR:
      throw std::runtime_error("Cannot use ReduceOp.BXOR with NCCL");
    case ::c10d::ReduceOp::PREMUL_SUM:
      return RedOpRAII(op, comm, dataType, nccl_api_);
    case ::c10d::ReduceOp::AVG:
      return ncclAvg;
    default:
      throw std::runtime_error("Unsupported reduce operation");
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

  cudaStreamCaptureMode mode = cudaStreamCaptureModeThreadLocal;
  CUDA_CHECK_IGNORE(
      cuda_api_,
      cuda_api_->threadExchangeStreamCaptureMode(&mode),
      "Failed to swap capture mode for timeout thread");

  // Honor the noexcept contract: the loop issues NCCL probes (NCCL_CHECK) and
  // abort paths that can throw; swallow here so nothing escapes this thread.
  try {
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
        ncclResult_t asyncErr;
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
    ncclResult_t asyncErr;
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
      throw ncclException;
    }
    abortNcclComm();
    if (options_c10d_->abort_process_on_timeout_or_error) {
      TC_LOG(ERROR, this) << "Aborting process due to error: "
                          << ncclException.what();
      runAbortHooks();
      ::abort();
    } else {
      throw ncclException;
    }
  }
}

bool ProcessGroupNCCL::getGraphCaptureMode() {
  cudaStream_t current_stream =
      cuda_api_->getCurrentCUDAStream(device_.index());
  cudaStreamCaptureStatus capture_status;

  cudaError_t err =
      cuda_api_->streamIsCapturing(current_stream, &capture_status);
  if (err == cudaSuccess) {
    return capture_status == cudaStreamCaptureStatusActive;
  }

  throw std::runtime_error(
      "Failed to check CUDA stream capture status: " +
      std::string(cuda_api_->getErrorString(err)));
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
    cudaStreamCaptureStatus capture_status;
    unsigned long long graph_id;
    cudaGraph_t graph;

    cudaError_t err = cuda_api_->streamGetCaptureInfo_v2(
        stream, &capture_status, &graph_id, &graph, nullptr, nullptr);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          "Failed to get CUDA stream capture info: " +
          std::string(cuda_api_->getErrorString(err)));
    } else if (capture_status == cudaStreamCaptureStatusActive) {
      std::lock_guard<std::mutex> lock(graph_capture_work_mutex_);

      // Check if this is the first work object for this graph
      bool is_first_work = graph_capture_work_refs_[graph_id].empty();

      // Add work reference to the per-graph container
      graph_capture_work_refs_[graph_id].push_back(work);

      // If this is the first work object for this graph, set up automatic
      // cleanup
      if (is_first_work) {
        // Create cleanup data that will be passed to the callback
        auto* cleanup_data = new GraphCleanupData(this, graph_id);

        // Create a CUDA user object with our cleanup callback
        cudaUserObject_t user_object;
        err = cuda_api_->userObjectCreate(
            &user_object,
            cleanup_data,
            graphCleanupCallback,
            1, // initial reference count
            cudaUserObjectNoDestructorSync);
        if (err != cudaSuccess) {
          // If we failed to create the user object, clean up manually
          delete cleanup_data;
          throw std::runtime_error(
              "Failed to create user object: " +
              std::string(cuda_api_->getErrorString(err)));
        } else {
          // Retain the user object in the graph so it gets cleaned up when the
          // graph is destroyed
          err = cuda_api_->graphRetainUserObject(
              graph,
              user_object,
              1, // reference count
              cudaGraphUserObjectMove);
          if (err != cudaSuccess) {
            // If we failed to retain the user object, clean up manually
            delete cleanup_data;
            throw std::runtime_error(
                "Failed to retain user object: " +
                std::string(cuda_api_->getErrorString(err)));
          }
        }
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
  CUDA_CHECK(
      cuda_api_,
      cuda_api_->setDevice(device_.index()),
      "Failed to set CUDA device for operation");
  if (async_op) {
    // Get current PyTorch CUDA stream for this device
    cudaStream_t current_stream =
        cuda_api_->getCurrentCUDAStream(device_.index());

    // Record event on current stream and wait for it on internal stream
    CUDA_CHECK(
        cuda_api_,
        cuda_api_->eventRecord(dependency_event_, current_stream),
        "Failed to record dependency event");

    CUDA_CHECK(
        cuda_api_,
        cuda_api_->streamWaitEvent(internal_stream_, dependency_event_, 0),
        "Failed to make internal stream wait for dependency event");

    return internal_stream_;
  } else {
    // Use the current PyTorch CUDA stream for synchronous operations
    return cuda_api_->getCurrentCUDAStream(device_.index());
  }
}

void ProcessGroupNCCL::ensureTensorContiguous(const at::Tensor& tensor) {
  if (!tensor.is_contiguous()) {
    throw std::runtime_error("Tensor must be contiguous for NCCL operations");
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
cudaEvent_t ProcessGroupNCCL::getEvent() {
  std::lock_guard<std::mutex> lock(event_pool_mutex_);

  if (!event_pool_.empty()) {
    cudaEvent_t event = event_pool_.front();
    event_pool_.pop();
    return event;
  }

  // Create new event if pool is empty
  cudaEvent_t event;
  CUDA_CHECK(
      cuda_api_,
      cuda_api_->eventCreateWithFlags(&event, cudaEventDisableTiming),
      "Failed to create event");
  return event;
}

void ProcessGroupNCCL::returnEvent(cudaEvent_t event) {
  std::lock_guard<std::mutex> lock(event_pool_mutex_);

  if (event_pool_.size() < max_event_pool_size_) {
    event_pool_.push(event);
  } else {
    // Pool is full, destroy the event
    CUDA_CHECK(
        cuda_api_, cuda_api_->eventDestroy(event), "Failed to destroy event");
  }
}

// CCA (CUDA caching allocator) memory-hook registration is deferred: it
// auto-registers allocator segments with NCCL for symmetric-memory / window
// support, which is not part of this initial port. Collectives work without it.
void ProcessGroupNCCL::attachMemoryHook() {}

void ProcessGroupNCCL::detachMemoryHook() {}

} // namespace c10d::nccl2
