// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/comms/nccl/TorchCommNCCL.hpp>
#include <torch/csrc/comms/nccl/TorchCommNCCLCCA.hpp>

#include <nccl.h> // @manual
#include <torch/csrc/comms/utils/Logging.hpp>
#include <stdexcept>
#include <string>

namespace torch::comms {

namespace {

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

TorchCommNCCL::RedOpRAII::RedOpRAII(ncclRedOp_t op)
    : ncclRedOp_(op), comm_(nullptr) {}

TorchCommNCCL::RedOpRAII::RedOpRAII(
    const ReduceOp& op,
    const ncclComm_t comm,
    const ncclDataType_t dataType,
    std::shared_ptr<NcclApi> nccl_api)
    : comm_(comm), nccl_api_(std::move(nccl_api)) {
  TORCH_INTERNAL_ASSERT(
      op == ReduceOp::RedOpType::PREMUL_SUM,
      "Constructing premul_sum RedOpRAII with non-premul_sum RedOpType");

  if (!op.factor().has_value()) {
    ncclRedOp_ = ncclSum;
    comm_ = nullptr;
    return;
  }

  const auto& factor = op.factor().value();
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

TorchCommNCCL::RedOpRAII::~RedOpRAII() {
  if (comm_) {
    NCCL_CHECK_IGNORE(
        nccl_api_,
        nccl_api_->redOpDestroy(ncclRedOp_, comm_),
        "NCCL redOpDestroy failed");
  }
}

size_t TorchCommNCCL::wordSize(ncclDataType_t type) const {
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

ncclDataType_t TorchCommNCCL::getNcclDataType(const at::Tensor& tensor) {
  return getNcclDataTypeInternal(tensor);
}

TorchCommNCCL::RedOpRAII TorchCommNCCL::getNcclReduceOp(
    const ReduceOp& op,
    const ncclComm_t comm,
    const ncclDataType_t dataType) {
  switch (op) {
    case ReduceOp::RedOpType::SUM:
      return ncclSum;
    case ReduceOp::RedOpType::PRODUCT:
      return ncclProd;
    case ReduceOp::RedOpType::MIN:
      return ncclMin;
    case ReduceOp::RedOpType::MAX:
      return ncclMax;
    case ReduceOp::RedOpType::BAND:
      throw std::runtime_error("Cannot use ReduceOp.BAND with NCCL");
    case ReduceOp::RedOpType::BOR:
      throw std::runtime_error("Cannot use ReduceOp.BOR with NCCL");
    case ReduceOp::RedOpType::BXOR:
      throw std::runtime_error("Cannot use ReduceOp.BXOR with NCCL");
    case ReduceOp::RedOpType::PREMUL_SUM:
      return RedOpRAII(op, comm, dataType, nccl_api_);
    case ReduceOp::RedOpType::AVG:
      return ncclAvg;
    default:
      throw std::runtime_error("Unsupported reduce operation");
  }
}

void TorchCommNCCL::checkWorkQueue() {
  TorchWorkNCCL::WorkStatus status = workq_.garbageCollect();

  switch (status) {
    case TorchWorkNCCL::WorkStatus::TIMEDOUT:
      comm_state_ = CommState::TIMEOUT;
      break;
    case TorchWorkNCCL::WorkStatus::ERROR:
      comm_state_ = CommState::ERROR;
      break;
    default:
      // For COMPLETED, NOT_STARTED, and INPROGRESS, no state change needed
      break;
  }
}

// The timeout thread cannot make NCCL calls.  The only CUDA call it can make
// it cudaEventQuery.
void TorchCommNCCL::timeoutWatchdog() noexcept {
  TC_LOG(INFO, this) << "Timeout thread starting for rank: " << rank_;

  cudaStreamCaptureMode mode = cudaStreamCaptureModeThreadLocal;
  CUDA_CHECK_IGNORE(
      cuda_api_,
      cuda_api_->threadExchangeStreamCaptureMode(&mode),
      "Failed to swap capture mode for timeout thread");

  while (!shutdown_) {
    {
      std::unique_lock<std::mutex> lock(timeout_mutex_);
      // Wait for a shorter interval to check work objects periodically
      // Wake up either after 1 second or immediately if shutdown is requested
      timeout_cv_.wait_for(
          lock, std::chrono::seconds(1), [this]() { return shutdown_.load(); });

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
        options_.abort_process_on_timeout_or_error &&
        !options_.enable_reconfigure) {
      if (comm_state_ == CommState::TIMEOUT) {
        TC_LOG(ERROR, this)
            << "Aborting process due to timeout on rank " << rank_
            << " - timeout watchdog detected operation timeout";
      } else if (comm_state_ == CommState::ERROR) {
        TC_LOG(ERROR, this) << "Aborting process due to error on rank " << rank_
                            << " - timeout watchdog detected operation error. ";
      }
      ::abort();
    }

    // Check communicator for async error. This runs on a noexcept watchdog
    // thread, so log (don't throw) if the query itself fails.
    if (comm_state_ == CommState::NORMAL) {
      ncclResult_t asyncErr = ncclSuccess;
      NCCL_CHECK_IGNORE(
          nccl_api_,
          nccl_api_->commGetAsyncError(nccl_comm_, &asyncErr),
          "failed to get async error");
      if (asyncErr != ncclSuccess) {
        comm_state_ = CommState::ERROR;
        TC_LOG(ERROR, this)
            << "Aborting process due to error on rank " << rank_
            << " - nccl hit async error: " << ncclGetErrorString(asyncErr);
        abort();
      }
    }
  }

  TC_LOG(INFO, this) << "Timeout thread exiting for rank: " << rank_;
}

void TorchCommNCCL::checkInitialized() const {
  if (init_state_ != InitializationState::INITIALIZED) {
    throw std::runtime_error("TorchCommNCCL not initialized");
  }
}

void TorchCommNCCL::checkAndAbortIfTimedOutOrError() {
  // Nothing to check in graph capture mode
  if (getGraphCaptureMode()) {
    return;
  }

  // First, check work queue status
  checkWorkQueue();

  if (comm_state_ == CommState::TIMEOUT) {
    if (options_.enable_reconfigure) {
      revokeNcclComm();
      throw std::runtime_error("NCCL operation timed out");
    } else {
      abortNcclComm();
      if (options_.abort_process_on_timeout_or_error) {
        TC_LOG(ERROR, this) << "Aborting process due to timeout";
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
    abortNcclComm();
    if (options_.abort_process_on_timeout_or_error) {
      TC_LOG(ERROR, this) << "Aborting process due to error: "
                          << ncclException.what();
      ::abort();
    } else {
      throw ncclException;
    }
  }
}

bool TorchCommNCCL::getGraphCaptureMode() {
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

c10::intrusive_ptr<TorchWorkNCCL> TorchCommNCCL::createWork(
    cudaStream_t stream,
    std::chrono::milliseconds timeout,
    const std::vector<at::Tensor>& inputTensors) {
  // Only create the work object without enqueuing it
  auto work = c10::make_intrusive<TorchWorkNCCL>(
      shared_from_this(), stream, timeout, inputTensors);
  return work;
}

c10::intrusive_ptr<TorchWorkNCCL> TorchCommNCCL::createWork(
    cudaStream_t stream,
    std::chrono::milliseconds timeout,
    const at::Tensor& inputTensor) {
  // Single-tensor overload to avoid vector allocation
  auto work = c10::make_intrusive<TorchWorkNCCL>(
      shared_from_this(), stream, timeout, inputTensor);
  return work;
}

void TorchCommNCCL::enqueueWork(
    c10::intrusive_ptr<TorchWorkNCCL> work,
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
void CUDART_CB TorchCommNCCL::graphCleanupCallback(void* userData) {
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

cudaStream_t TorchCommNCCL::getOperationStream(bool async_op) {
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

void TorchCommNCCL::ensureTensorContiguous(const at::Tensor& tensor) {
  if (!tensor.is_contiguous()) {
    throw std::runtime_error("Tensor must be contiguous for NCCL operations");
  }
}

void TorchCommNCCL::checkTensorDevice(const at::Tensor& tensor) const {
  TORCH_CHECK(
      tensor.device().type() == device_.type(),
      "Expected tensor on ",
      device_.type(),
      " but found tensor on ",
      tensor.device());
}

void TorchCommNCCL::checkTensorsDevice(
    const std::vector<at::Tensor>& tensors) const {
  for (const auto& t : tensors) {
    checkTensorDevice(t);
  }
}

// Protected methods (not in the private section of the header)
cudaEvent_t TorchCommNCCL::getEvent() {
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

void TorchCommNCCL::returnEvent(cudaEvent_t event) {
  std::lock_guard<std::mutex> lock(event_pool_mutex_);

  if (event_pool_.size() < max_event_pool_size_) {
    event_pool_.push(event);
  } else {
    // Pool is full, destroy the event
    CUDA_CHECK(
        cuda_api_, cuda_api_->eventDestroy(event), "Failed to destroy event");
  }
}

void TorchCommNCCL::attachMemoryHook() {
  NcclCachingAllocatorHook::getInstance().registerComm(this);
}

void TorchCommNCCL::detachMemoryHook() {
  NcclCachingAllocatorHook::getInstance().deregisterComm(this);
}

} // namespace torch::comms
