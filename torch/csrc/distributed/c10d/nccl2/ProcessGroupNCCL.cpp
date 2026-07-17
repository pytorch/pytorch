// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>

#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <fmt/core.h>
#include <nccl.h>
#include <torch/csrc/cuda/CUDAPluggableAllocator.h>
#include <torch/csrc/distributed/c10d/nccl2/Logging.hpp>
#include <torch/csrc/distributed/c10d/nccl2/NCCLBootstrap.hpp>
#include <torch/csrc/distributed/c10d/nccl2/TracingGuard.hpp>
#include <torch/csrc/distributed/c10d/nccl2/Utils.hpp>

namespace c10d::nccl2 {

namespace {

void checkSameDtype(const at::Tensor& reference, const at::Tensor& tensor) {
  if (reference.scalar_type() != tensor.scalar_type()) {
    C10_THROW_ERROR(TypeError, "Tensors must have identical type");
  }
}

void checkSameDtype(
    const at::Tensor& reference,
    const std::vector<at::Tensor>& tensors) {
  for (const auto& tensor : tensors) {
    checkSameDtype(reference, tensor);
  }
}

} // namespace

ncclResult_t NCCLException::getResult() const noexcept {
  return result_;
}

ProcessGroupNCCL::~ProcessGroupNCCL() {
  if (init_state_ == InitializationState::INITIALIZED) {
    TC_LOG(WARNING, this)
        << "ProcessGroupNCCL " << name_
        << " was not finalized before destruction. "
        << "This may indicate a resource leak. Please call finalize() explicitly.";

    // Signal shutdown to timeout watchdog thread to prevent it from accessing
    // this object after destruction
    shutdown_ = true;

    // Wake up the timeout watchdog thread
    {
      std::lock_guard<std::mutex> lock(timeout_mutex_);
      timeout_cv_.notify_all();
    }

    // Wait for timeout thread to finish. If we're being called from within
    // the timeout thread itself (e.g., garbageCollect popped a work item whose
    // destruction released the last shared_ptr to this comm), we must detach
    // instead of join to avoid a deadlock.
    if (timeout_thread_.joinable()) {
      if (std::this_thread::get_id() != timeout_thread_.get_id()) {
        timeout_thread_.join();
      } else {
        timeout_thread_.detach(); // NOLINT(facebook-hte-BadCall-detach)
      }
    }

    // Abort the NCCL communicator since we can't do a clean finalization
    // Note: We don't call the full abortNcclComm() to avoid potential abort()
    // calls from options_.abort_process_on_timeout_or_error
    if (nccl_comm_) {
      // Best effort to abort the communicator - ignore errors since we're
      // in the destructor
      if (nccl_api_) {
        (void)nccl_api_->commAbort(nccl_comm_);
      }
      nccl_comm_ = nullptr;
    }
  }

  // We need to detach the memory hook in case finalize is not called,
  // so that we don't encounter a memory corruption.
  detachMemoryHook();
}

void ProcessGroupNCCL::init(at::Device device) {
  TC_LOG(INFO, this) << "Initializing ProcessGroupNCCL for device: " << device;
  device_ = device;

  if (init_state_ == InitializationState::INITIALIZED) {
    throw std::runtime_error("ProcessGroupNCCL already initialized");
  } else if (init_state_ == InitializationState::FINALIZED) {
    throw std::runtime_error("ProcessGroupNCCL already finalized");
  }

  if (!nccl_api_) {
    nccl_api_ = std::make_unique<DefaultNcclApi>();
  }

  if (device_.index() == -1 || nccl_comm_ == nullptr) {
    auto bootstrap = std::make_unique<NCCLBootstrap>(
        store_,
        device_,
        getRank(),
        getSize(),
        bootstrap_generation_++,
        nccl_api_,
        options_c10d_->timeout);
    device_ = bootstrap->getDevice();

    if (nccl_comm_ == nullptr) {
      nccl_comm_ = bootstrap->createNcclComm(name_, options_c10d_->hints);
    }
  }

  initNcclResources();

  init_state_ = InitializationState::INITIALIZED;
  TracingGuard tracingGuard(name_, comm_size_, "init", rank_);

  TC_LOG(INFO, this) << "ProcessGroupNCCL initialized for rank: " << rank_;
}

void ProcessGroupNCCL::initNcclResources() {
  c10::cuda::CUDAGuard gpuGuard(device_);

  is_high_priority_stream_ = options_c10d_->is_high_priority_stream;

  if (!internal_stream_) {
    internal_stream_.emplace(
        at::cuda::getStreamFromPool(is_high_priority_stream_, device_.index()));
  }

  if (!dependency_event_) {
    dependency_event_.emplace(cudaEventDisableTiming);
  }

  if (!barrier_buffer_) {
    barrier_buffer_ =
        c10::cuda::CUDACachingAllocator::get()->allocate(sizeof(float));
  }

  max_event_pool_size_ = kDefaultMaxEventPoolSize;
  if (auto it = options_c10d_->hints.find(std::string(kHintMaxEventPoolSize));
      it != options_c10d_->hints.end()) {
    max_event_pool_size_ = static_cast<size_t>(std::stoull(it->second));
  }

  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commUserRank(nccl_comm_, &rank_),
      "NCCL User Rank failed");

  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commCount(nccl_comm_, &comm_size_),
      "NCCL Count failed");

  if (!shutdown_) {
    timeout_thread_ = std::thread(&ProcessGroupNCCL::timeoutWatchdog, this);
  }

  attachMemoryHook();
}

void ProcessGroupNCCL::abort() {
  if (options_c10d_->enable_reconfigure) {
    revokeNcclComm();
  } else {
    abortNcclComm();
  }
  comm_state_ = CommState::ERROR;
}

void ProcessGroupNCCL::suspend() {
  checkInitialized();
  c10::cuda::CUDAGuard gpuGuard(device_);
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commSuspend(nccl_comm_, NCCL_SUSPEND_MEM),
      "NCCL Suspend failed (requires NCCL 2.29.7+)");
}

void ProcessGroupNCCL::resume() {
  checkInitialized();
  c10::cuda::CUDAGuard gpuGuard(device_);
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commResume(nccl_comm_),
      "NCCL Resume failed (requires NCCL 2.29.7+)");
}

std::unordered_map<std::string, uint64_t> ProcessGroupNCCL::getMemoryStats() {
  checkInitialized();
  c10::cuda::CUDAGuard gpuGuard(device_);
  // Stat indices follow ncclCommMemStat_t: suspend=0, suspended=1, persist=2,
  // total=3. Keys match ProcessGroupNCCL (the original backend).
  static constexpr std::array<std::pair<const char*, int>, 4> kStats = {
      {{"suspend", 0}, {"suspended", 1}, {"persist", 2}, {"total", 3}}};
  std::unordered_map<std::string, uint64_t> stats;
  for (const auto& [name, stat] : kStats) {
    uint64_t value = 0;
    NCCL_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->commMemStats(nccl_comm_, stat, &value),
        "NCCL MemStats failed (requires NCCL 2.29.7+)");
    stats.emplace(name, value);
  }
  return stats;
}

::c10d::ErrorType ProcessGroupNCCL::getError() {
  switch (comm_state_.load()) {
    case CommState::TIMEOUT:
      return ::c10d::ErrorType::TIMEOUT;
    case CommState::ERROR:
      return ::c10d::ErrorType::COMM_ERROR;
    default:
      return ::c10d::ErrorType::SUCCESS;
  }
}

void ProcessGroupNCCL::finalize() {
  if (init_state_ == InitializationState::UNINITIALIZED) {
    throw std::runtime_error("ProcessGroupNCCL not initialized");
  } else if (init_state_ == InitializationState::FINALIZED) {
    throw std::runtime_error("ProcessGroupNCCL already finalized");
  }
  init_state_ = InitializationState::FINALIZED;

  // Signal shutdown to timeout watchdog
  shutdown_ = true;

  // Wake up the timeout watchdog thread
  {
    std::lock_guard<std::mutex> lock(timeout_mutex_);
    timeout_cv_.notify_all();
  }

  // Wait for timeout thread to finish
  if (timeout_thread_.joinable()) {
    timeout_thread_.join();
  }

  // Wait for all pending work objects to complete and get final status
  auto work_status = workq_.finalize();

  if (work_status == WorkNCCL::WorkStatus::NOT_STARTED ||
      work_status == WorkNCCL::WorkStatus::INPROGRESS) {
    throw std::runtime_error(
        "WorkQ finalize returned in progress or not started state");
  }

  // Update comm_state_ based on the work status
  if (work_status == WorkNCCL::WorkStatus::TIMEDOUT) {
    comm_state_ = CommState::TIMEOUT;
    abortNcclComm();
    throw std::runtime_error("Work timed out during finalize");
  } else if (work_status == WorkNCCL::WorkStatus::ERROR) {
    comm_state_ = CommState::ERROR;
    ncclResult_t asyncErr{};
    NCCL_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->commGetAsyncError(nccl_comm_, &asyncErr),
        "failed to get async error");
    NCCLException ncclException(
        *nccl_api_, "NCCL Async Error", asyncErr, nccl_comm_);
    abortNcclComm();
    throw std::move(ncclException);
  }

  // Clean up event pool
  {
    std::lock_guard<std::mutex> lock(event_pool_mutex_);
    while (!event_pool_.empty()) {
      event_pool_.pop();
    }
  }

  barrier_buffer_.clear();

  dependency_event_.reset();
  internal_stream_.reset();

  // Destroy NCCL communicator
  // Note: If abortNcclComm() was called, nccl_comm_ is already nullptr and this
  // is skipped. We must not call commDestroy after commAbort per NCCL docs.
  if (nccl_comm_) {
    detachMemoryHook();
    // Deregister comm from the CachingAllocator
    NCCL_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->commDestroy(nccl_comm_),
        "NCCL Destroy failed");
    nccl_comm_ = nullptr;
  }
}

void ProcessGroupNCCL::abortNcclComm() {
  detachMemoryHook();
  if (nccl_comm_) {
    NCCL_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->commAbort(nccl_comm_),
        "NCCL Abort failed");
    nccl_comm_ = nullptr;
  }
  // Never abort the process in reconfigurable mode: callers fall back to
  // revoke + throw so the failure can be handled by reconfiguring.
  if (options_c10d_->abort_process_on_timeout_or_error &&
      !options_c10d_->enable_reconfigure) {
    TC_LOG(ERROR, this) << "Aborting process due to timeout";
    runAbortHooks();
    ::abort();
  }
}

void ProcessGroupNCCL::revokeNcclComm() {
  // Idempotent: the timeout watchdog and a synchronous collective may both
  // observe the same timeout and attempt a revoke. Run the teardown (abort
  // hooks, memory hook detach, commRevoke) at most once per communicator
  // generation. revoked_ is reset on reconfigure.
  if (revoked_.exchange(true)) {
    return;
  }
  TC_LOG(INFO, this) << "Calling abort hooks before commRevoke.";
  runAbortHooks();
  detachMemoryHook();
  if (nccl_comm_) {
    // Best-effort: this may run on the timeout watchdog thread, so log instead
    // of throwing on failure (the communicator is already being torn down).
    NCCL_CHECK_IGNORE(
        nccl_api_, nccl_api_->commRevoke(nccl_comm_), "NCCL Revoke failed");
  }
}

int64_t ProcessGroupNCCL::getCommPtr() const {
  return reinterpret_cast<int64_t>(nccl_comm_);
}

// Point-to-Point Operations
c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::sendImpl(
    const at::Tensor& tensor,
    int dst,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);
  checkTensorDevice(tensor);

  TracingGuard tracingGuard(name_, comm_size_, "send", dst, tensor, tensor);

  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op ? createWork(stream, timeout, tensor)
                       : createWork(stream, timeout);

  // Record start event before NCCL operation
  work->recordStart("send");

  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->send(
          tensor.data_ptr(),
          tensor.numel(),
          getNcclDataType(tensor),
          dst,
          nccl_comm_,
          stream),
      "NCCL Send failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::recvImpl(
    at::Tensor& tensor,
    int src,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);
  checkTensorDevice(tensor);

  TracingGuard tracingGuard(name_, comm_size_, "recv", src, tensor, tensor);

  cudaStream_t stream = getOperationStream(async_op);
  auto work = createWork(stream, timeout);

  // Record start event before NCCL operation
  work->recordStart("recv");

  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->recv(
          tensor.data_ptr(),
          tensor.numel(),
          getNcclDataType(tensor),
          src,
          nccl_comm_,
          stream),
      "NCCL Recv failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

// Batch P2P Operations
c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::batch_op_issue(
    const std::vector<BatchSendRecv::P2POp>& ops,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  if (ops.empty()) {
    throw std::runtime_error("Cannot issue empty batch operation");
  }

  // Collect input and output tensors for work tracking
  std::vector<at::Tensor> input_tensors;
  std::vector<at::Tensor> output_tensors;

  for (const auto& op : ops) {
    checkTensorDevice(op.tensor);
    if (op.type == BatchSendRecv::P2POp::OpType::SEND) {
      at::Tensor tensor = op.tensor;
      ensureTensorContiguous(tensor);
      input_tensors.push_back(tensor);
    } else if (op.type == BatchSendRecv::P2POp::OpType::RECV) {
      at::Tensor tensor = op.tensor;
      ensureTensorContiguous(tensor);
      output_tensors.push_back(tensor);
    } else {
      throw std::runtime_error("Unknown op type");
    }
  }

  TracingGuard tracingGuard(
      name_,
      comm_size_,
      "batch_op_issue",
      rank_,
      input_tensors,
      output_tensors);

  cudaStream_t stream = getOperationStream(async_op);
  auto work = createWork(stream, timeout, input_tensors);

  // Record start event before NCCL operations
  work->recordStart("batch_op_issue");

  // Start NCCL group for batched operations
  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");

  // Issue each operation individually
  for (const auto& op : ops) {
    if (op.type == BatchSendRecv::P2POp::OpType::SEND) {
      ncclResult_t result = nccl_api_->send(
          op.tensor.data_ptr(),
          op.tensor.numel(),
          getNcclDataType(op.tensor),
          op.peer,
          nccl_comm_,
          stream);

      if (result != ncclSuccess) {
        throw NCCLException(
            *nccl_api_,
            "NCCL Send failed in batch operation",
            result,
            nccl_comm_);
      }
    } else if (op.type == BatchSendRecv::P2POp::OpType::RECV) {
      ncclResult_t result = nccl_api_->recv(
          op.tensor.data_ptr(),
          op.tensor.numel(),
          getNcclDataType(op.tensor),
          op.peer,
          nccl_comm_,
          stream);

      if (result != ncclSuccess) {
        throw NCCLException(
            *nccl_api_,
            "NCCL Recv failed in batch operation",
            result,
            nccl_comm_);
      }
    }
  }

  // End NCCL group
  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");

  // Record end event after NCCL operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

// Collective Operations
c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::broadcastImpl(
    at::Tensor& tensor,
    int root,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);
  checkTensorDevice(tensor);

  TracingGuard tracingGuard(
      name_, comm_size_, "broadcast", root, tensor, tensor);

  cudaStream_t stream = getOperationStream(async_op);

  auto work = async_op ? createWork(stream, timeout, tensor)
                       : createWork(stream, timeout);

  // Record start event before NCCL operation
  work->recordStart("broadcast");

  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->bcast(
          tensor.data_ptr(),
          tensor.numel(),
          getNcclDataType(tensor),
          root,
          nccl_comm_,
          stream),
      "NCCL Broadcast failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::all_reduce(
    at::Tensor& tensor,
    const ::c10d::ReduceOp& op,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);
  checkTensorDevice(tensor);

  TracingGuard tracingGuard(
      name_, comm_size_, "all_reduce", rank_, tensor, tensor);

  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op ? createWork(stream, timeout, tensor)
                       : createWork(stream, timeout);

  // Record start event before NCCL operation
  work->recordStart("all_reduce");

  const auto dataType = getNcclDataType(tensor);
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->allReduce(
          tensor.data_ptr(),
          tensor.data_ptr(), // In-place operation
          tensor.numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, dataType),
          nccl_comm_,
          stream),
      "NCCL AllReduce failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::reduceImpl(
    const at::Tensor& tensor,
    int root,
    const ::c10d::ReduceOp& op,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);
  checkTensorDevice(tensor);

  TracingGuard tracingGuard(name_, comm_size_, "reduce", root, tensor, tensor);

  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op ? createWork(stream, timeout, tensor)
                       : createWork(stream, timeout);

  // Record start event before NCCL operation
  work->recordStart("reduce");

  const auto dataType = getNcclDataType(tensor);
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->reduce(
          tensor.data_ptr(),
          rank_ == root ? tensor.data_ptr() : nullptr,
          tensor.numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, dataType),
          root,
          nccl_comm_,
          stream),
      "NCCL Reduce failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::all_gather(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  if (tensor_list.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error(
        "tensor_list size must equal comm_size for all_gather");
  }

  // Ensure input tensor is contiguous
  ensureTensorContiguous(tensor);

  // Check that all output tensors are contiguous and have correct size
  for (const auto& t : tensor_list) {
    ensureTensorContiguous(t);
    if (t.numel() != tensor.numel()) {
      throw std::runtime_error(
          "All tensors in tensor_list must have same size as input tensor");
    }
  }

  checkTensorDevice(tensor);
  checkTensorsDevice(tensor_list);
  checkSameDtype(tensor, tensor_list);

  TracingGuard tracingGuard(
      name_, comm_size_, "all_gather", rank_, tensor_list, {tensor});

  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op ? createWork(stream, timeout, tensor)
                       : createWork(stream, timeout);

  work->recordStart("all_gather");

  // Use multiple broadcast operations for all_gather
  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");

  for (int i = 0; i < comm_size_; ++i) {
    ncclResult_t opResult = nccl_api_->broadcast(
        tensor.data_ptr(),
        tensor_list[i].data_ptr(),
        tensor.numel(),
        getNcclDataType(tensor_list[i]),
        i,
        nccl_comm_,
        stream);
    if (opResult != ncclSuccess) {
      throw NCCLException(
          *nccl_api_,
          "NCCL Broadcast failed in all_gather",
          opResult,
          nccl_comm_);
    }
  }

  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::allGatherSingleImpl(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);
  checkTensorDevice(output);
  checkTensorDevice(input);
  checkSameDtype(input, output);

  if (output.numel() != input.numel() * comm_size_) {
    throw std::runtime_error(
        "Output tensor size must be input_size * comm_size for allGatherSingleImpl");
  }

  TracingGuard tracingGuard(
      name_, comm_size_, "allGatherSingleImpl", rank_, input, output);

  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op ? createWork(stream, timeout, input)
                       : createWork(stream, timeout);

  work->recordStart("allGatherSingleImpl");

  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->allGather(
          input.data_ptr(),
          output.data_ptr(),
          input.numel(),
          getNcclDataType(input),
          nccl_comm_,
          stream),
      "NCCL AllGather failed");

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::reduce_scatter(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ::c10d::ReduceOp& op,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);

  if (input_list.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error(
        "input_list size must equal comm_size for reduce_scatter");
  }

  // Check that all input tensors are contiguous and have correct size
  for (const auto& t : input_list) {
    ensureTensorContiguous(t);
    if (t.numel() != output.numel()) {
      throw std::runtime_error(
          "All input tensors must have same size as output tensor");
    }
  }

  checkTensorsDevice(input_list);
  checkTensorDevice(output);
  checkSameDtype(output, input_list);

  TracingGuard tracingGuard(
      name_, comm_size_, "reduce_scatter", rank_, input_list, {output});

  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op ? createWork(stream, timeout, input_list)
                       : createWork(stream, timeout);

  work->recordStart("reduce_scatter");

  // Use multiple reduce operations for reduce_scatter
  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");

  for (int i = 0; i < comm_size_; ++i) {
    const auto dataType = getNcclDataType(input_list[i]);
    ncclResult_t opResult{};
    if (i == rank_) {
      // This rank receives the reduced result
      opResult = nccl_api_->reduce(
          input_list[i].data_ptr(),
          output.data_ptr(),
          output.numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, dataType),
          i,
          nccl_comm_,
          stream);
    } else {
      // Other ranks contribute to the reduction
      opResult = nccl_api_->reduce(
          input_list[i].data_ptr(),
          nullptr, // Non-root ranks don't receive
          input_list[i].numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, dataType),
          i,
          nccl_comm_,
          stream);
    }
    if (opResult != ncclSuccess) {
      throw NCCLException(
          *nccl_api_,
          "NCCL Reduce failed in reduce_scatter",
          opResult,
          nccl_comm_);
    }
  }

  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::reduceScatterSingleImpl(
    at::Tensor& output,
    const at::Tensor& input,
    const ::c10d::ReduceOp& op,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);
  checkTensorDevice(output);
  checkTensorDevice(input);
  checkSameDtype(input, output);

  if (input.numel() != output.numel() * comm_size_) {
    throw std::runtime_error(
        "Input tensor size must be output_size * comm_size for reduceScatterSingleImpl");
  }

  TracingGuard tracingGuard(
      name_, comm_size_, "reduceScatterSingleImpl", rank_, input, output);

  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op ? createWork(stream, timeout, input)
                       : createWork(stream, timeout);

  // Record start event before NCCL operation
  work->recordStart("reduceScatterSingleImpl");

  const auto dataType = getNcclDataType(input);
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->reduceScatter(
          input.data_ptr(),
          output.data_ptr(),
          output.numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, dataType),
          nccl_comm_,
          stream),
      "NCCL ReduceScatter failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::allToAllSingleImpl(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);
  checkTensorDevice(output);
  checkTensorDevice(input);
  checkSameDtype(input, output);

  if (input.numel() != output.numel()) {
    throw std::runtime_error(
        "Input and output tensors must have same size for allToAllSingleImpl");
  }

  if (input.numel() % comm_size_ != 0) {
    throw std::runtime_error(
        "Tensor size must be divisible by comm_size for allToAllSingleImpl");
  }

  TracingGuard tracingGuard(
      name_, comm_size_, "allToAllSingleImpl", rank_, input, output);

  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op ? createWork(stream, timeout, input)
                       : createWork(stream, timeout);

  // Record start event before NCCL operation
  work->recordStart("allToAllSingleImpl");

  size_t chunk_size = input.numel() / comm_size_;
  const auto data_type = getNcclDataType(input);

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->allToAll(
          input.data_ptr(),
          output.data_ptr(),
          chunk_size,
          data_type,
          nccl_comm_,
          stream),
      "NCCL AllToAll failed");
#else
  size_t offset = chunk_size * wordSize(data_type);
  char* sptr = static_cast<char*>(input.data_ptr());
  char* rptr = static_cast<char*>(output.data_ptr());
  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");

  for (int i = 0; i < comm_size_; ++i) {
    // Send to rank i
    ncclResult_t opResult = nccl_api_->send(
        sptr + i * offset, chunk_size, data_type, i, nccl_comm_, stream);
    if (opResult != ncclSuccess) {
      throw NCCLException(
          *nccl_api_,
          "NCCL Send failed in allToAllSingleImpl",
          opResult,
          nccl_comm_);
    }

    // Receive from rank i
    opResult = nccl_api_->recv(
        rptr + i * offset, chunk_size, data_type, i, nccl_comm_, stream);
    if (opResult != ncclSuccess) {
      throw NCCLException(
          *nccl_api_,
          "NCCL Recv failed in allToAllSingleImpl",
          opResult,
          nccl_comm_);
    }
  }

  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");
#endif

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::all_to_all_v_single(
    at::Tensor& output,
    const at::Tensor& input,
    const std::vector<uint64_t>& output_split_sizes,
    const std::vector<uint64_t>& input_split_sizes,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);
  checkTensorDevice(output);
  checkTensorDevice(input);
  checkSameDtype(input, output);

  // Validate split sizes vectors
  if (input_split_sizes.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error(
        "input_split_sizes length must equal comm_size for all_to_all_v_single");
  }

  if (output_split_sizes.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error(
        "output_split_sizes length must equal comm_size for all_to_all_v_single");
  }

  // Validate that split sizes sum does not exceed tensor dimensions
  uint64_t input_total = 0;
  uint64_t output_total = 0;
  for (int i = 0; i < comm_size_; ++i) {
    input_total += input_split_sizes[i];
    output_total += output_split_sizes[i];
  }

  if (input_total > static_cast<uint64_t>(input.size(0))) {
    throw std::runtime_error(
        "Sum of input_split_sizes exceeds input tensor size for all_to_all_v_single");
  }

  if (output_total > static_cast<uint64_t>(output.size(0))) {
    throw std::runtime_error(
        "Sum of output_split_sizes exceeds output tensor size for all_to_all_v_single");
  }

  TracingGuard tracingGuard(
      name_, comm_size_, "all_to_all_v_single", rank_, input, output);

  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op ? createWork(stream, timeout, input)
                       : createWork(stream, timeout);

  // Record start event before NCCL operation
  work->recordStart("all_to_all_v_single");

  // Convert split sizes to arrays and calculate displacements
  std::vector<size_t> sendcounts(comm_size_);
  std::vector<size_t> recvcounts(comm_size_);
  std::vector<size_t> senddispls(comm_size_);
  std::vector<size_t> recvdispls(comm_size_);

  // Calculate the number of elements per slice along the first dimension
  // For a tensor with shape [N, D1, D2, ..., Dk], each slice of size S along
  // dim 0 contains S * D1 * D2 * ... * Dk elements
  // Use input tensor for send counts and output tensor for recv counts
  size_t send_elements_per_slice =
      input.numel() ? input.numel() / input.size(0) : 0;
  size_t recv_elements_per_slice =
      output.numel() ? output.numel() / output.size(0) : 0;
  const auto data_type = getNcclDataType(input);
  const size_t type_size = wordSize(data_type);

  size_t sendoffset = 0;
  size_t recvoffset = 0;
  for (int i = 0; i < comm_size_; ++i) {
    sendcounts[i] = input_split_sizes[i] * send_elements_per_slice;
    recvcounts[i] = output_split_sizes[i] * recv_elements_per_slice;
    senddispls[i] = sendoffset;
    recvdispls[i] = recvoffset;
    sendoffset += sendcounts[i];
    recvoffset += recvcounts[i];
  }

  char* sptr = static_cast<char*>(input.data_ptr());
  char* rptr = static_cast<char*>(output.data_ptr());

  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");

  for (int i = 0; i < comm_size_; ++i) {
    ncclResult_t opResult = nccl_api_->send(
        sptr + senddispls[i] * type_size,
        sendcounts[i],
        data_type,
        i,
        nccl_comm_,
        stream);
    if (opResult != ncclSuccess) {
      throw NCCLException(
          *nccl_api_,
          "NCCL Send failed in all_to_all_v_single",
          opResult,
          nccl_comm_);
    }
    opResult = nccl_api_->recv(
        rptr + recvdispls[i] * type_size,
        recvcounts[i],
        data_type,
        i,
        nccl_comm_,
        stream);
    if (opResult != ncclSuccess) {
      throw NCCLException(
          *nccl_api_,
          "NCCL Recv failed in all_to_all_v_single",
          opResult,
          nccl_comm_);
    }
  }

  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::all_to_all(
    const std::vector<at::Tensor>& output_tensor_list,
    const std::vector<at::Tensor>& input_tensor_list,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  checkTensorsDevice(output_tensor_list);
  checkTensorsDevice(input_tensor_list);
  if (output_tensor_list.size() != static_cast<size_t>(comm_size_) ||
      input_tensor_list.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error(
        "Tensor list sizes must equal comm_size for all_to_all");
  }

  // Validate all tensors
  for (int i = 0; i < comm_size_; ++i) {
    ensureTensorContiguous(input_tensor_list[i]);
    ensureTensorContiguous(output_tensor_list[i]);
    checkSameDtype(input_tensor_list[0], input_tensor_list[i]);
    checkSameDtype(input_tensor_list[0], output_tensor_list[i]);
  }

  TracingGuard tracingGuard(
      name_,
      comm_size_,
      "all_to_all",
      rank_,
      input_tensor_list,
      output_tensor_list);

  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op ? createWork(stream, timeout, input_tensor_list)
                       : createWork(stream, timeout);

  // Record start event before NCCL operations
  work->recordStart("all_to_all");

  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");

  for (int i = 0; i < comm_size_; ++i) {
    // Send to rank i
    ncclResult_t opResult = nccl_api_->send(
        input_tensor_list[i].data_ptr(),
        input_tensor_list[i].numel(),
        getNcclDataType(input_tensor_list[i]),
        i,
        nccl_comm_,
        stream);
    if (opResult != ncclSuccess) {
      throw NCCLException(
          *nccl_api_, "NCCL Send failed in all_to_all", opResult, nccl_comm_);
    }

    // Receive from rank i
    opResult = nccl_api_->recv(
        output_tensor_list[i].data_ptr(),
        output_tensor_list[i].numel(),
        getNcclDataType(output_tensor_list[i]),
        i,
        nccl_comm_,
        stream);
    if (opResult != ncclSuccess) {
      throw NCCLException(
          *nccl_api_, "NCCL Recv failed in all_to_all", opResult, nccl_comm_);
    }
  }

  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");

  // Record end event after NCCL operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::barrierImpl(
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();

  TracingGuard tracingGuard(name_, comm_size_, "barrier", rank_);
  cudaStream_t stream = getOperationStream(async_op);
  auto work = createWork(stream, timeout);

  // Record start event before NCCL operation
  work->recordStart("barrier");

  // Use pre-allocated CUDA buffer for barrier
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->allReduce(
          barrier_buffer_.get(),
          barrier_buffer_.get(),
          1,
          ncclFloat32,
          ncclSum,
          nccl_comm_,
          stream),
      "NCCL Barrier failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::scatterImpl(
    at::Tensor& output_tensor,
    const std::vector<at::Tensor>& input_tensor_list,
    int root,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output_tensor);
  checkTensorDevice(output_tensor);
  checkTensorsDevice(input_tensor_list);

  // Only the root rank needs valid input tensors
  if (rank_ == root) {
    if (input_tensor_list.size() != static_cast<size_t>(comm_size_)) {
      throw std::runtime_error(
          "input_tensor_list size must equal comm_size for scatter");
    }

    for (const auto& t : input_tensor_list) {
      ensureTensorContiguous(t);
      checkSameDtype(output_tensor, t);
      if (t.numel() != output_tensor.numel()) {
        throw std::runtime_error(
            "All input tensors must have same size as output tensor");
      }
    }
  }

  TracingGuard tracingGuard(
      name_, comm_size_, "scatter", root, input_tensor_list, {output_tensor});

  cudaStream_t stream = getOperationStream(async_op);
  std::vector<at::Tensor> input_tensors;
  if (async_op && rank_ == root) {
    input_tensors = input_tensor_list;
  }
  auto work = createWork(stream, timeout, input_tensors);

  // Record start event before NCCL operations
  work->recordStart("scatter");

  // Implement scatter using point-to-point operations
  if (rank_ == root) {
    // Root sends to all ranks (except itself)
    NCCL_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->groupStart(),
        "NCCL GroupStart failed");
    for (int i = 0; i < comm_size_; ++i) {
      if (i != root) {
        ncclResult_t opResult = nccl_api_->send(
            input_tensor_list[i].data_ptr(),
            input_tensor_list[i].numel(),
            getNcclDataType(input_tensor_list[i]),
            i,
            nccl_comm_,
            stream);
        if (opResult != ncclSuccess) {
          throw NCCLException(
              *nccl_api_, "NCCL Send failed in scatter", opResult, nccl_comm_);
        }
      }
    }
    NCCL_CHECK(
        nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");

    at::cuda::CUDAStreamGuard stream_guard(
        at::cuda::getStreamFromExternal(stream, device_.index()));
    output_tensor.copy_(input_tensor_list[root], true);
  } else {
    // Non-root ranks receive from root
    NCCL_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->recv(
            output_tensor.data_ptr(),
            output_tensor.numel(),
            getNcclDataType(output_tensor),
            root,
            nccl_comm_,
            stream),
        "NCCL Recv failed in scatter");
  }

  // Record end event after NCCL operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::gatherImpl(
    const std::vector<at::Tensor>& output_tensor_list,
    const at::Tensor& input_tensor,
    int root,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(input_tensor);
  checkTensorDevice(input_tensor);
  checkTensorsDevice(output_tensor_list);

  // Only the root rank needs valid output tensors
  if (rank_ == root) {
    if (output_tensor_list.size() != static_cast<size_t>(comm_size_)) {
      throw std::runtime_error(
          "output_tensor_list size must equal comm_size for gather");
    }

    for (const auto& t : output_tensor_list) {
      ensureTensorContiguous(t);
      checkSameDtype(input_tensor, t);
      if (t.numel() != input_tensor.numel()) {
        throw std::runtime_error(
            "All output tensors must have same size as input tensor");
      }
    }
  }

  TracingGuard tracingGuard(
      name_, comm_size_, "gather", root, {input_tensor}, output_tensor_list);

  cudaStream_t stream = getOperationStream(async_op);
  std::vector<at::Tensor> output_tensors;
  if (rank_ == root) {
    output_tensors = output_tensor_list;
  }
  auto work = async_op ? createWork(stream, timeout, input_tensor)
                       : createWork(stream, timeout);

  // Record start event before NCCL operations
  work->recordStart("gather");

  if (rank_ == root) {
    // Root receives from all ranks (except itself)
    NCCL_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->groupStart(),
        "NCCL GroupStart failed");
    for (int i = 0; i < comm_size_; ++i) {
      if (i != root) {
        ncclResult_t opResult = nccl_api_->recv(
            output_tensor_list[i].data_ptr(),
            output_tensor_list[i].numel(),
            getNcclDataType(output_tensor_list[i]),
            i,
            nccl_comm_,
            stream);
        if (opResult != ncclSuccess) {
          throw NCCLException(
              *nccl_api_, "NCCL Recv failed in gather", opResult, nccl_comm_);
        }
      }
    }
    NCCL_CHECK(
        nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");

    at::cuda::CUDAStreamGuard stream_guard(
        at::cuda::getStreamFromExternal(stream, device_.index()));
    output_tensor_list[root].copy_(input_tensor, true);
  } else {
    // Non-root ranks send to root
    NCCL_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->send(
            input_tensor.data_ptr(),
            input_tensor.numel(),
            getNcclDataType(input_tensor),
            root,
            nccl_comm_,
            stream),
        "NCCL Send failed in gather");
  }

  // Record end event after NCCL operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

NCCLException::NCCLException(
    NcclApi& nccl_api,
    const std::string& message,
    ncclResult_t result,
    ncclComm_t comm)
    : message_(fmt::format(
          "{}: {} \nNCCL Last Error: {}",
          message,
          nccl_api.getErrorString(result),
          nccl_api.getLastError(comm))),
      result_(result) {}

const char* NCCLException::what() const noexcept {
  return message_.c_str();
}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
