// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/comms/BackendWrapper.hpp>
#include <torch/csrc/comms/TorchComm.hpp>
#include <torch/csrc/comms/utils/Logging.hpp>

#include <c10/core/DeviceGuard.h> // @manual=//caffe2:c10

namespace torch::comms {

namespace {

// Extract the scaling factor from NCCL's PREMUL_SUM operation supplement.
// NCCLPreMulSumSupplement stores either a tensor or double scaling factor
// that is applied before summation.
PreMulSumFactorT getPreMulSumFactor(const c10d::ReduceOp& op) {
  TORCH_CHECK(
      op.supplement_ != nullptr,
      "PREMUL_SUM operation requires a supplement, but none was provided");

  const auto* preMulSupplement =
      dynamic_cast<const c10d::NCCLPreMulSumSupplement*>(op.supplement_.get());
  TORCH_CHECK(
      preMulSupplement != nullptr,
      "PREMUL_SUM operation supplement must be of type NCCLPreMulSumSupplement");

  if (preMulSupplement->tensor_factor.defined()) {
    return preMulSupplement->tensor_factor;
  }
  return preMulSupplement->double_factor;
}

ReduceOp toReduceOp(const c10d::ReduceOp& op) {
  switch (op) {
    case c10d::ReduceOp::SUM:
      return ReduceOp::SUM;
    case c10d::ReduceOp::AVG:
      return ReduceOp::AVG;
    case c10d::ReduceOp::MIN:
      return ReduceOp::MIN;
    case c10d::ReduceOp::MAX:
      return ReduceOp::MAX;
    case c10d::ReduceOp::PRODUCT:
      return ReduceOp::PRODUCT;
    case c10d::ReduceOp::BAND:
      return ReduceOp::BAND;
    case c10d::ReduceOp::BOR:
      return ReduceOp::BOR;
    case c10d::ReduceOp::BXOR:
      return ReduceOp::BXOR;
    case c10d::ReduceOp::PREMUL_SUM:
      return ReduceOp::make_nccl_premul_sum(getPreMulSumFactor(op));
    default:
      throw std::runtime_error("Unsupported reduce op");
  }
}

std::vector<uint64_t> toVecUint64(const std::vector<int64_t>& vec) {
  std::vector<uint64_t> vecUint64;
  vecUint64.reserve(vec.size());
  for (auto i : vec) {
    vecUint64.push_back(i);
  }
  return vecUint64;
}

} // namespace

WorkWrapper::WorkWrapper(
    c10::intrusive_ptr<TorchWork> work,
    std::vector<at::Tensor> outputTensors)
    : work_(std::move(work)), outputTensors_(std::move(outputTensors)) {
  std::vector<c10::Device> devices;
  // CPU needs to wait for the TorchWork to complete before marking Future
  // as completed
  for (const auto& tensor : outputTensors_) {
    if (tensor.device().type() != c10::DeviceType::CPU) {
      devices.push_back(tensor.device());
      break;
    }
  }
  future_ = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);

  // If we are doing a barrier collective, for all device types, devices vector
  // would be empty we would fallback to same CPU synchronization logic
  if (!devices.empty()) {
    work_->markCompleted(
        c10::intrusive_ptr<c10::ivalue::Future>(future_), outputTensors_);
  } else if (work_->isCompleted()) {
    // For other device types (CPU) synchronous op already finished —
    // resolve now.
    future_->markCompleted(c10::IValue(outputTensors_));
  } else {
    // For other device types (CPU) async: register end hook so
    // future completes when setStatus fires.
    work_->registerWorkEndHook([future = future_, tensors = outputTensors_]() {
      if (!future->completed()) {
        future->markCompleted(c10::IValue(tensors));
      }
    });
  }
}

bool WorkWrapper::wait(std::chrono::milliseconds timeout) {
  if (timeout != kNoTimeout) {
    auto ex = std::make_exception_ptr(
        std::runtime_error("wait timeout not supported"));
    finish(ex);
    std::rethrow_exception(ex);
  }
  try {
    work_->wait();
  } catch (...) {
    finish(std::current_exception());
    throw;
  }
  if (!future_->completed()) {
    future_->markCompleted(c10::IValue(outputTensors_));
  }
  finish();
  return true;
}
void WorkWrapper::synchronize() {
  try {
    work_->wait();
  } catch (...) {
    finish(std::current_exception());
    throw;
  }
  if (!future_->completed()) {
    future_->markCompleted(c10::IValue(outputTensors_));
  }
  finish();
}
std::vector<at::Tensor> WorkWrapper::result() {
  return outputTensors_;
}
c10::intrusive_ptr<c10::ivalue::Future> WorkWrapper::getFuture() {
  return future_;
}

BackendWrapper::BackendWrapper(std::shared_ptr<TorchComm> comm)
    : Backend(comm->getRank(), comm->getSize()),
      comm_(comm),
      options_(c10::make_intrusive<Options>()) {}

c10::intrusive_ptr<c10d::Work> BackendWrapper::broadcast(
    std::vector<at::Tensor>& tensors,
    const c10d::BroadcastOptions& opts) {
  TORCH_CHECK(
      tensors.size() == 1,
      "Only single tensor supported, but got ",
      tensors.size(),
      " tensors");
  BroadcastOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  return c10::make_intrusive<WorkWrapper>(
      comm_->broadcast(
          tensors.at(0), static_cast<int>(opts.rootRank), opts.asyncOp, bopts),
      tensors);
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::allreduce(
    std::vector<at::Tensor>& tensors,
    const c10d::AllreduceOptions& opts) {
  TORCH_CHECK(
      tensors.size() == 1,
      "Only single tensor supported, but got ",
      tensors.size(),
      " tensors");
  AllReduceOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  return c10::make_intrusive<WorkWrapper>(
      comm_->all_reduce(
          tensors.at(0), toReduceOp(opts.reduceOp), opts.asyncOp, bopts),
      tensors);
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const c10d::AllreduceCoalescedOptions& opts) {
  TORCH_CHECK(
      tensors.size() == 1,
      "Only single tensor supported, but got ",
      tensors.size(),
      " tensors");
  AllReduceOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  return c10::make_intrusive<WorkWrapper>(
      comm_->all_reduce(
          tensors.at(0), toReduceOp(opts.reduceOp), opts.asyncOp, bopts),
      tensors);
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::reduce(
    std::vector<at::Tensor>& tensors,
    const c10d::ReduceOptions& opts) {
  TORCH_CHECK(
      tensors.size() == 1,
      "Only single tensor supported, but got ",
      tensors.size(),
      " tensors");
  ReduceOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  return c10::make_intrusive<WorkWrapper>(
      comm_->reduce(
          tensors.at(0),
          static_cast<int>(opts.rootRank),
          toReduceOp(opts.reduceOp),
          opts.asyncOp,
          bopts),
      tensors);
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts) {
  TORCH_CHECK(
      outputTensors.size() == 1,
      "Only single output tensor list supported, but got ",
      outputTensors.size());
  TORCH_CHECK(
      inputTensors.size() == 1,
      "Only single input tensor supported, but got ",
      inputTensors.size());
  const auto& input = inputTensors.at(0);
  auto& outputList = outputTensors.at(0);
  TORCH_CHECK(
      static_cast<int>(outputList.size()) == getSize(),
      "Expected ",
      getSize(),
      " output tensors (one per rank), but got ",
      outputList.size());

  AllGatherOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }

  // Fast path: when per-rank output tensors point to distinct memory,
  // delegate straight to the backend's list-based all_gather (no extra
  // alloc/copy). Simply check the first two ranks.
  bool aliased = outputList.size() > 1 &&
      outputList[0].data_ptr() == outputList[1].data_ptr();
  if (!aliased) {
    return c10::make_intrusive<WorkWrapper>(
        comm_->all_gather(outputList, input, opts.asyncOp, bopts), outputList);
  }

  // Slow path (aliased outputs): each outputList[r] points to the same
  // K-element buffer, but the gather needs world_size * K bytes. Allocate
  // a contiguous staging tensor shaped {world_size, K}, gather into it,
  // then copy each rank's row back into the caller's per-rank tensor.
  AllGatherSingleOptions sopts;
  sopts.timeout = bopts.timeout;
  auto staging = at::empty(
      {getSize() * input.numel()},
      input.options().memory_format(at::MemoryFormat::Contiguous));
  auto work = c10::make_intrusive<WorkWrapper>(
      comm_->all_gather_single(staging, input, opts.asyncOp, sopts),
      outputList);
  auto rows = staging.view({getSize(), input.numel()});
  work->wait(kNoTimeout);
  for (int r = 0; r < getSize(); ++r) {
    outputList.at(r).copy_(rows[r].view_as(outputList.at(r)));
  }
  return work;
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& outputTensorLists,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts) {
  TORCH_CHECK(
      outputTensorLists.size() == 1,
      "Only single output tensor list supported, but got ",
      outputTensorLists.size());
  TORCH_CHECK(
      inputTensors.size() == 1,
      "Only single input tensor supported, but got ",
      inputTensors.size());
  AllGatherOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  return c10::make_intrusive<WorkWrapper>(
      comm_->all_gather(
          outputTensorLists.at(0), inputTensors.at(0), opts.asyncOp, bopts),
      outputTensorLists.at(0));
}

// Note: Coalesced operations with multiple input/output tensors are not yet
// supported. Currently only single tensor is supported. When extending this,
// iterate over all tensors and coalesce them into a single backend call.
c10::intrusive_ptr<c10d::Work> BackendWrapper::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& output_tensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts) {
  TORCH_CHECK(
      output_tensors.size() == 1,
      "Only single output tensor supported, but got ",
      output_tensors.size());
  TORCH_CHECK(
      inputTensors.size() == 1,
      "Only single input tensor supported, but got ",
      inputTensors.size());
  AllGatherSingleOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  return c10::make_intrusive<WorkWrapper>(
      comm_->all_gather_single(
          output_tensors.at(0), inputTensors.at(0), opts.asyncOp, bopts),
      output_tensors);
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::_allgather_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const c10d::AllgatherOptions& opts) {
  AllGatherSingleOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  return c10::make_intrusive<WorkWrapper>(
      comm_->all_gather_single(outputTensor, inputTensor, opts.asyncOp, bopts),
      std::vector<at::Tensor>{outputTensor});
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::GatherOptions& opts) {
  if (getRank() == opts.rootRank) {
    TORCH_CHECK(
        outputTensors.size() == 1,
        "Only single output tensor list supported on root rank, but got ",
        outputTensors.size());
  } else if (outputTensors.empty()) {
    // Normalize non-root c10d gather outputs to wrapper's empty list shape
    outputTensors = {};
    outputTensors.emplace_back();
  } else {
    TORCH_CHECK(
        outputTensors.size() == 1,
        "Only single output tensor list supported on non-root ranks, but got ",
        outputTensors.size());
  }
  TORCH_CHECK(
      inputTensors.size() == 1,
      "Only single input tensor supported, but got ",
      inputTensors.size());
  GatherOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  return c10::make_intrusive<WorkWrapper>(
      comm_->gather(
          outputTensors.at(0),
          inputTensors.at(0),
          static_cast<int>(opts.rootRank),
          opts.asyncOp,
          bopts),
      outputTensors.at(0));
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const c10d::ScatterOptions& opts) {
  TORCH_CHECK(
      outputTensors.size() == 1,
      "Only single output tensor supported, but got ",
      outputTensors.size());
  ScatterOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  if (getRank() == opts.rootRank) {
    TORCH_CHECK(
        inputTensors.size() == 1,
        "Only single input tensor list supported on root rank, but got ",
        inputTensors.size());
  } else {
    // if not in the root rank, initialize inputTensors as empty place holder
    // with an empty list
    inputTensors = {};
    inputTensors.emplace_back();
  }
  return c10::make_intrusive<WorkWrapper>(
      comm_->scatter(
          outputTensors.at(0),
          inputTensors.at(0),
          static_cast<int>(opts.rootRank),
          opts.asyncOp,
          bopts),
      outputTensors);
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const c10d::ReduceScatterOptions& opts) {
  TORCH_CHECK(
      outputTensors.size() == 1,
      "Only single output tensor supported, but got ",
      outputTensors.size());
  TORCH_CHECK(
      inputTensors.size() == 1,
      "Only single input tensor list supported, but got ",
      inputTensors.size());
  ReduceScatterOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  return c10::make_intrusive<WorkWrapper>(
      comm_->reduce_scatter(
          outputTensors.at(0),
          inputTensors.at(0),
          toReduceOp(opts.reduceOp),
          opts.asyncOp,
          bopts),
      outputTensors);
}

// Note: Coalesced operations with multiple input/output tensors are not yet
// supported. Currently only single tensor is supported. When extending this,
// iterate over all tensors and coalesce them into a single backend call.
c10::intrusive_ptr<c10d::Work> BackendWrapper::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::ReduceScatterOptions& opts) {
  TORCH_CHECK(
      outputTensors.size() == 1,
      "Only single output tensor supported, but got ",
      outputTensors.size());
  TORCH_CHECK(
      inputTensors.size() == 1,
      "Only single input tensor supported, but got ",
      inputTensors.size());
  ReduceScatterSingleOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  return c10::make_intrusive<WorkWrapper>(
      comm_->reduce_scatter_single(
          outputTensors.at(0),
          inputTensors.at(0),
          toReduceOp(opts.reduceOp),
          opts.asyncOp,
          bopts),
      outputTensors);
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::_reduce_scatter_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const c10d::ReduceScatterOptions& opts) {
  ReduceScatterSingleOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  return c10::make_intrusive<WorkWrapper>(
      comm_->reduce_scatter_single(
          outputTensor,
          inputTensor,
          toReduceOp(opts.reduceOp),
          opts.asyncOp,
          bopts),
      std::vector<at::Tensor>{outputTensor});
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const c10d::AllToAllOptions& opts) {
  if (outputSplitSizes.empty() && inputSplitSizes.empty()) {
    AllToAllSingleOptions bopts;
    if (opts.timeout != kUnsetTimeout) {
      bopts.timeout = opts.timeout;
    } else {
      bopts.timeout = options_->timeout;
    }
    return c10::make_intrusive<WorkWrapper>(
        comm_->all_to_all_single(
            outputTensor, inputTensor, opts.asyncOp, bopts),
        std::vector<at::Tensor>{outputTensor});
  } else {
    AllToAllvSingleOptions bopts;
    if (opts.timeout != kUnsetTimeout) {
      bopts.timeout = opts.timeout;
    } else {
      bopts.timeout = options_->timeout;
    }
    return c10::make_intrusive<WorkWrapper>(
        comm_->all_to_all_v_single(
            outputTensor,
            inputTensor,
            toVecUint64(outputSplitSizes),
            toVecUint64(inputSplitSizes),
            opts.asyncOp,
            bopts),
        std::vector<at::Tensor>{outputTensor});
  }
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllToAllOptions& opts) {
  AllToAllOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  return c10::make_intrusive<WorkWrapper>(
      comm_->all_to_all(outputTensors, inputTensors, opts.asyncOp, bopts),
      outputTensors);
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::barrier(
    const c10d::BarrierOptions& opts) {
  BarrierOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  return c10::make_intrusive<WorkWrapper>(comm_->barrier(opts.asyncOp, bopts));
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int /*tag*/) {
  TORCH_CHECK(
      tensors.size() == 1,
      "Only single tensor supported, but got ",
      tensors.size(),
      " tensors");
  if (coalescing_batch_.has_value()) {
    coalescing_batch_->send(tensors.at(0), dstRank);
    // Per-op Work returned during coalescing is a no-op sentinel; the real
    // Work covering the whole batch is returned by endCoalescing(). c10d's
    // batch_isend_irecv discards these per-op returns.
    return c10::make_intrusive<WorkWrapper>(
        c10::make_intrusive<TorchWorkCompleted>(), tensors);
  }
  return c10::make_intrusive<WorkWrapper>(
      comm_->send(tensors.at(0), dstRank, /*async_op=*/true), tensors);
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int /*tag*/) {
  TORCH_CHECK(
      tensors.size() == 1,
      "Only single tensor supported, but got ",
      tensors.size(),
      " tensors");
  if (coalescing_batch_.has_value()) {
    coalescing_batch_->recv(tensors.at(0), srcRank);
    return c10::make_intrusive<WorkWrapper>(
        c10::make_intrusive<TorchWorkCompleted>(), tensors);
  }
  return c10::make_intrusive<WorkWrapper>(
      comm_->recv(tensors.at(0), srcRank, /*async_op=*/true), tensors);
}

void BackendWrapper::startCoalescing() {
  TORCH_CHECK(
      !coalescing_batch_.has_value(),
      "BackendWrapper::startCoalescing called while a batch is already active");
  coalescing_batch_.emplace(comm_->batch_op_create());
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::endCoalescing() {
  TORCH_CHECK(
      coalescing_batch_.has_value(),
      "BackendWrapper::endCoalescing called without a matching startCoalescing");
  // Move the batch out so we always reset state, even if issue() throws.
  auto batch = std::move(*coalescing_batch_);
  coalescing_batch_.reset();
  if (batch.ops.empty()) {
    // Empty coalescing window — return a completed sentinel so callers can
    // .wait() without blocking.
    return c10::make_intrusive<WorkWrapper>(
        c10::make_intrusive<TorchWorkCompleted>());
  }
  BatchP2POptions bopts;
  bopts.timeout = options_->timeout;
  return c10::make_intrusive<WorkWrapper>(
      batch.issue(/*async_op=*/true, bopts));
}

std::shared_ptr<TorchComm> BackendWrapper::getComm() const {
  return comm_;
}

std::shared_ptr<c10::Allocator> BackendWrapper::getMemAllocator() {
  return comm_->getMemAllocator();
}

const std::string BackendWrapper::getBackendName() const {
  return comm_->getBackend();
}

std::string_view BackendWrapper::getBackendVersion() const {
  return comm_->getBackendVersion();
}

c10::intrusive_ptr<c10d::Backend::Options> BackendWrapper::getBackendOptions() {
  return c10::static_intrusive_pointer_cast<c10d::Backend::Options>(options_);
}

bool BackendWrapper::verifyWorkTimeoutForTest(
    const c10::intrusive_ptr<c10d::Work>& work,
    const std::chrono::milliseconds& timeout) {
  // The work must be a WorkWrapper that wraps a TorchWork
  auto workWrapper = c10::dynamic_intrusive_pointer_cast<WorkWrapper>(work);
  if (!workWrapper) {
    TORCH_CHECK(false, "Work is not a WorkWrapper");
  }

  // Get the timeout from the underlying TorchWork
  return workWrapper->work_->getTimeout() == timeout;
}

void BackendWrapper::setTimeout(std::chrono::milliseconds timeout) {
  options_->timeout = timeout;
}
void BackendWrapper::shutdown() {
  // Idempotent: destroy_process_group iterates all backends and calls
  // shutdown() on each, but multiple BackendWrappers can share the same
  // underlying TorchComm (mixed cpu:gloo,cuda:nccl PGs registered through
  // the backendType-to-wrapper dedup path). Finalize-on-already-finalized
  // throws "TorchCommNCCL already finalized" — log and continue so destroy
  // is always safe to call.
  if (comm_) {
    try {
      comm_->finalize();
    } catch (const std::exception& e) {
      TC_LOG(WARNING)
          << "BackendWrapper::shutdown: TorchComm::finalize() raised, "
          << "treating as no-op (likely already finalized): " << e.what();
    }
  }
}

void BackendWrapper::abort() {
  if (comm_) {
    try {
      comm_->abort();
    } catch (const std::exception& e) {
      TC_LOG(WARNING) << "BackendWrapper::abort: TorchComm::abort() raised, "
                      << "treating as no-op (likely already aborted): "
                      << e.what();
    }
  }
}

c10::intrusive_ptr<c10d::Backend> BackendWrapper::split(
    const c10::intrusive_ptr<c10d::Store>& /* store */,
    const std::vector<int>& ranks,
    const c10::intrusive_ptr<c10d::Backend::Options>& opts) {
  auto comm = getComm();
  CommOptions commOpts;
  auto backendOpts = c10::dynamic_intrusive_pointer_cast<Options>(opts);
  if (backendOpts) {
    commOpts.abort_process_on_timeout_or_error =
        backendOpts->abort_process_on_timeout_or_error;
    commOpts.timeout = backendOpts->timeout;
    commOpts.high_priority_stream = backendOpts->high_priority_stream;
    commOpts.store = backendOpts->store;
    commOpts.hints = backendOpts->hints;
  }
  auto new_comm = comm->split(ranks, opts->group_name, commOpts);
  if (new_comm == nullptr) {
    return nullptr;
  }
  return c10::make_intrusive<BackendWrapper>(new_comm);
}

} // namespace torch::comms
