// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// c10d::Backend surface for ProcessGroupNCCL: the constructor, device binding,
// and the virtual overrides. Each override unwraps the c10d
// tensor-list shape and forwards the c10d option fields (c10d::ReduceOp,
// rootRank, asyncOp, resolved timeout) directly to the internal NCCL engine
// helpers, then tags the returned c10d::Work with its output tensors.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>

#include <c10/cuda/CUDAGuard.h>
#include <c10/util/irange.h>
#include <torch/csrc/cuda/CUDAPluggableAllocator.h>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/cuda/utils.hpp>

#include <torch/csrc/distributed/c10d/nccl2/Logging.hpp>
#include <torch/csrc/distributed/c10d/nccl2/WindowNCCL.hpp>

namespace c10d::nccl2 {

namespace {

std::vector<uint64_t> normalizeSplitSizes(
    const std::vector<int64_t>& split_sizes,
    const at::Tensor& tensor,
    int group_size) {
  c10d::checkSplitSizes(split_sizes, tensor, group_size);

  if (split_sizes.empty()) {
    return std::vector<uint64_t>(
        group_size, static_cast<uint64_t>(tensor.size(0) / group_size));
  }

  std::vector<uint64_t> out;
  out.reserve(split_sizes.size());
  for (auto split_size : split_sizes) {
    out.push_back(static_cast<uint64_t>(split_size));
  }
  return out;
}

// Trivially-completed c10d::Work used for empty coalescing windows and for the
// per-op sentinels returned while a coalescing batch is being accumulated.
class CompletedWork : public ::c10d::Work {
 public:
  explicit CompletedWork(std::vector<at::Tensor> outputs = {})
      : outputs_(std::move(outputs)) {}
  bool isCompleted() override {
    return true;
  }
  bool isSuccess() const override {
    return true;
  }
  bool wait(std::chrono::milliseconds /*timeout*/ = kNoTimeout) override {
    return true;
  }
  void synchronize() override {}
  std::vector<at::Tensor> result() override {
    return outputs_;
  }

 private:
  std::vector<at::Tensor> outputs_;
};

c10::intrusive_ptr<WorkNCCL> coalesceWorks(
    std::vector<c10::intrusive_ptr<WorkNCCL>> works,
    std::vector<at::Tensor> outputs) {
  TORCH_INTERNAL_ASSERT(!works.empty());
  auto work = std::move(works.back());
  works.pop_back();
  work->setChildren(std::move(works));
  work->setOutputs(std::move(outputs));
  return work;
}

} // namespace

ProcessGroupNCCL::ProcessGroupNCCL(
    c10::intrusive_ptr<::c10d::Store> store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : Backend(rank, size),
      device_(at::kCUDA),
      store_(std::move(store)),
      event_pool_(std::make_shared<NCCLEventPool>(
          getCvarBool(::c10d::TORCH_NCCL_CUDA_EVENT_CACHE, true),
          getCvarBool(::c10d::TORCH_NCCL_ENABLE_TIMING, false),
          kDefaultMaxEventPoolSize)),
      async_error_handling_(static_cast<::c10d::ErrorHandlingMode>(getCvarInt(
          ::c10d::TORCH_NCCL_ASYNC_ERROR_HANDLING,
          ::c10d::SkipCleanUp))),
      blocking_wait_(getCvarBool(::c10d::TORCH_NCCL_BLOCKING_WAIT, false)),
      options_c10d_(options ? std::move(options) : Options::create()) {
  name_ = options_c10d_->group_name.empty() ? std::string(kBackendName)
                                            : options_c10d_->group_name;

  setGroupUid(options_c10d_->group_name);

  if (options_c10d_->config.blocking == NCCL_CONFIG_UNDEF_INT) {
    auto nonblocking = c10::utils::check_env("TORCH_NCCL_USE_COMM_NONBLOCKING");
    options_c10d_->config.blocking = nonblocking.value_or(false) ? 0 : 1;
  }
#if NCCL_VERSION_CODE < NCCL_VERSION(2, 28, 0) || defined(USE_ROCM)
  TORCH_CHECK(
      !options_c10d_->enable_reconfigure,
      "nccl2 reconfigure requires NCCL 2.28 or later and is not supported "
      "with RCCL");
#endif
}

std::chrono::milliseconds ProcessGroupNCCL::operationTimeout(
    std::chrono::milliseconds opt_timeout) const {
  // c10d leaves per-op timeouts unset as kUnsetTimeout (-1); fall back to the
  // communicator default in that case.
  return opt_timeout != ::c10d::kUnsetTimeout ? opt_timeout
                                              : options_c10d_->timeout;
}

void ProcessGroupNCCL::ensureInitialized(at::Device device) {
  TORCH_CHECK(
      device.is_cuda(), "ProcessGroupNCCL requires CUDA tensors/devices");
  if (init_state_ == InitializationState::INITIALIZED) {
    TORCH_CHECK(
        device_.index() == device.index(),
        "ProcessGroupNCCL is bound to device ",
        device_,
        " but an operation was issued on ",
        device);
    return;
  }
  TORCH_CHECK(
      init_state_ == InitializationState::UNINITIALIZED,
      "ProcessGroupNCCL has been finalized");
  // In the reconfigure regime the communicator is only ever created by
  // reconfigure(); collectives before the first reconfigure are an error.
  TORCH_CHECK(
      !options_c10d_->enable_reconfigure,
      "ProcessGroupNCCL has not been initialized. Call reconfigure() before "
      "issuing collectives when enable_reconfigure=True.");
  init(device);
}

c10::intrusive_ptr<::c10d::Backend::Options> ProcessGroupNCCL::
    getBackendOptions() {
  return c10::static_intrusive_pointer_cast<::c10d::Backend::Options>(
      options_c10d_);
}

void ProcessGroupNCCL::startTimeEstimate() {
#ifdef NCCL_SIM_INFO_INITIALIZER
  checkInitialized();
  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");
#else
  TORCH_CHECK(false, "NCCL time estimation requires NCCL 2.22 or later");
#endif
}

float ProcessGroupNCCL::endTimeEstimate() {
#ifdef NCCL_SIM_INFO_INITIALIZER
  ncclSimInfo_t simInfo = NCCL_SIM_INFO_INITIALIZER;
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->groupSimulateEnd(&simInfo),
      "NCCL GroupSimulateEnd failed");
  return simInfo.estimatedTime;
#else
  TORCH_CHECK(false, "NCCL time estimation requires NCCL 2.22 or later");
#endif
}

void ProcessGroupNCCL::setTimeout(std::chrono::milliseconds timeout) {
  options_c10d_->timeout = timeout;
}

void ProcessGroupNCCL::setBoundDeviceId(std::optional<at::Device> device) {
  if (options_c10d_->enable_reconfigure) {
    Backend::setBoundDeviceId(device);
    return;
  }
  if (!device.has_value() && init_state_ == InitializationState::INITIALIZED) {
    device = device_;
  }
  if (!device.has_value()) {
    const auto deviceCount = c10::cuda::device_count_ensure_non_zero();
    uint64_t deviceRank = static_cast<uint64_t>(getRank());
    const auto& globalRanks = options_c10d_->global_ranks_in_group;
    if (static_cast<size_t>(getRank()) < globalRanks.size()) {
      deviceRank = globalRanks[getRank()];
    }
    device = at::Device(
        at::kCUDA, static_cast<c10::DeviceIndex>(deviceRank % deviceCount));
  }
  Backend::setBoundDeviceId(device);
  ensureInitialized(*device);
}

void ProcessGroupNCCL::eagerConnectSingleDevice(at::Device device) {
  setBoundDeviceId(device);
}

void ProcessGroupNCCL::runAbortHooks() {
  // Snapshot rather than hold the lock across the hooks: a hook may
  // unregister itself, and the FlightRecorder hook blocks other threads for the
  // length of its dump, which must not also block an unrelated unregister.
  std::vector<::c10d::AbortHook> hooks;
  {
    std::lock_guard<std::mutex> lock(abort_hooks_mutex_);
    hooks.reserve(abortHooks_.size());
    for (const auto& [_, hook] : abortHooks_) {
      hooks.push_back(hook);
    }
  }
  for (const auto& hook : hooks) {
    try {
      hook();
    } catch (const std::exception& e) {
      LOG(ERROR) << "[TC] Abort hook threw exception: " << e.what();
    } catch (...) {
      LOG(ERROR) << "[TC] Abort hook threw unknown exception.";
    }
  }
}

void ProcessGroupNCCL::registerAbortHook(
    int64_t hook_id,
    ::c10d::AbortHook hook) {
  std::lock_guard<std::mutex> lock(abort_hooks_mutex_);
  abortHooks_.emplace(hook_id, std::move(hook));
}

void ProcessGroupNCCL::unregisterAbortHook(int64_t hook_id) {
  std::lock_guard<std::mutex> lock(abort_hooks_mutex_);
  abortHooks_.erase(hook_id);
}

bool ProcessGroupNCCL::hasCompletionHooks() {
  std::lock_guard<std::mutex> lock(completion_hooks_mutex_);
  return !completionHooks_.empty();
}

void ProcessGroupNCCL::runCompletionHooks(
    uint64_t completion_key,
    std::optional<float> duration_ms) {
  // Snapshot rather than hold the lock across the hooks, and for a harder
  // reason than runAbortHooks has: a hook's owner unregisters with its own lock
  // held (c10d::FlightRecorderHook::remove does), so calling into a hook while
  // holding completion_hooks_mutex_ would be the reverse order and deadlock.
  std::vector<::c10d::CompletionHook> hooks;
  {
    std::lock_guard<std::mutex> lock(completion_hooks_mutex_);
    hooks.reserve(completionHooks_.size());
    for (const auto& [_, hook] : completionHooks_) {
      hooks.push_back(hook);
    }
  }
  ::c10d::CompletionHookArgs args;
  args.completionKey = completion_key;
  args.duration_ms = duration_ms;
  for (const auto& hook : hooks) {
    try {
      hook(args);
    } catch (const std::exception& e) {
      LOG(ERROR) << "[TC] Completion hook threw exception: " << e.what();
    } catch (...) {
      LOG(ERROR) << "[TC] Completion hook threw unknown exception.";
    }
  }
}

void ProcessGroupNCCL::registerCompletionHook(
    int64_t hook_id,
    ::c10d::CompletionHook hook) {
  std::lock_guard<std::mutex> lock(completion_hooks_mutex_);
  completionHooks_.emplace(hook_id, std::move(hook));
}

void ProcessGroupNCCL::unregisterCompletionHook(int64_t hook_id) {
  std::lock_guard<std::mutex> lock(completion_hooks_mutex_);
  completionHooks_.erase(hook_id);
}

void ProcessGroupNCCL::shutdown() {
  // Called by destroy_process_group(). Drain in-flight work and close the comm
  // gracefully. Idempotent: finalize-on-already-finalized throws, so swallow.
  if (init_state_ != InitializationState::INITIALIZED) {
    return;
  }
  try {
    finalize();
  } catch (const std::exception& e) {
    TC_LOG(WARNING) << "ProcessGroupNCCL::shutdown: finalize() raised, "
                    << "treating as no-op: " << e.what();
  }
}

std::shared_ptr<c10::Allocator> ProcessGroupNCCL::getMemAllocator() {
  // Symmetric (VMM-backed) CUDA allocator backed by ncclMemAlloc/ncclMemFree.
  // Moved here from torchcomms' TorchCommFactory allocator registration.
  static std::shared_ptr<c10::Allocator> allocator = [] {
    auto nccl_api = std::make_shared<DefaultNcclApi>();
    return torch::cuda::CUDAPluggableAllocator::createCustomAllocator(
        [nccl_api](size_t size, int device, cudaStream_t stream) {
          at::cuda::OptionalCUDAGuard gpuGuard(device);
          void* ptr = nullptr;
          ncclResult_t result = nccl_api->memAlloc(&ptr, size);
          TORCH_CHECK(
              result == ncclSuccess,
              "ncclMemAlloc failed: ",
              nccl_api->getErrorString(result));
          return ptr;
        },
        [nccl_api](void* ptr, size_t size, int device, cudaStream_t stream) {
          at::cuda::OptionalCUDAGuard gpuGuard(device);
          ncclResult_t result = nccl_api->memFree(ptr);
          TORCH_CHECK(
              result == ncclSuccess,
              "ncclMemFree failed: ",
              nccl_api->getErrorString(result));
        });
  }();
  return allocator;
}

bool ProcessGroupNCCL::supportsTensorAlloc(c10::DeviceIndex deviceIdx) {
  return ::c10d::cuda::deviceSupportsMulticast(deviceIdx);
}

at::Tensor ProcessGroupNCCL::allocateTensor(
    long size,
    at::TensorOptions options) {
  TORCH_CHECK_VALUE(options.has_device(), "Tensor options must include device");
  auto device = options.device();
  TORCH_CHECK_VALUE(
      device.is_cuda(),
      "NCCL tensor allocator expects a CUDA device but got ",
      device);
  if (!device.has_index()) {
    device = getBoundDeviceId().value_or(
        at::Device(at::kCUDA, at::cuda::current_device()));
    options = options.device(device);
  }
  TORCH_CHECK(
      supportsTensorAlloc(device.index()),
      "NCCL tensor allocation is not supported on ",
      device);

  checkInitialized();
  TORCH_CHECK(
      device == device_,
      "ProcessGroupNCCL is bound to device ",
      device_,
      " but allocation was requested on ",
      device);
  at::cuda::OptionalCUDAGuard gpuGuard(device);

  if (!memPool_) {
    auto allocator = std::static_pointer_cast<
        c10::cuda::CUDACachingAllocator::CUDAAllocator>(getMemAllocator());
    auto pool = std::make_unique<at::cuda::MemPool>(std::move(allocator));
    registerMemPool(pool.get(), /*symm=*/false);
    memPool_ = std::move(pool);
  }

  auto threadId = std::this_thread::get_id();
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      memPool_->device(), memPool_->id(), [threadId](cudaStream_t) {
        return std::this_thread::get_id() == threadId;
      });
  auto tensor = at::empty({size}, options);
  c10::cuda::CUDACachingAllocator::endAllocateToPool(
      memPool_->device(), memPool_->id());
  c10::cuda::CUDACachingAllocator::releasePool(
      memPool_->device(), memPool_->id());
  return tensor;
}

c10::intrusive_ptr<::c10d::Window> ProcessGroupNCCL::new_window(
    const std::optional<at::Tensor>& tensor) {
  TORCH_CHECK(
      supportsWindow(),
      "ProcessGroupNCCL windows require NCCL 2.29 or later and are not "
      "supported on ROCm");
  checkInitialized();
  auto window = c10::make_intrusive<WindowNCCL>(
      c10::intrusive_ptr<ProcessGroupNCCL>::unsafe_reclaim_from_nonowning(
          this));
  if (tensor.has_value()) {
    window->tensor_register(*tensor);
  }
  return window;
}

bool ProcessGroupNCCL::supportsWindow() const {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0) && !defined(USE_ROCM)
  int runtime_version = 0;
  return ncclGetVersion(&runtime_version) == ncclSuccess &&
      runtime_version >= NCCL_VERSION(2, 29, 0);
#else
  return false;
#endif
}

// ---------------------------------------------------------------------------
// Collective / point-to-point overrides. Each forwards the c10d option fields
// directly to the internal engine helper and tags the work with its outputs.
// ---------------------------------------------------------------------------

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const ::c10d::BroadcastOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, "Only single tensor supported");
  auto tensor = tensors.at(0);
  if (tensor.is_complex()) {
    tensor = at::view_as_real(tensor);
  }
  ++sequence_number_;
  auto work = broadcastImpl(
      tensor,
      static_cast<int>(opts.rootRank),
      opts.asyncOp,
      operationTimeout(opts.timeout));
  work->setOutputs(tensors);
  return work;
}

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const ::c10d::AllreduceOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, "Only single tensor supported");
  auto tensor = tensors.at(0);
  if (tensor.is_complex()) {
    TORCH_CHECK(
        ::c10d::isComplexViewAsRealAllowed(opts.reduceOp),
        "all_reduce does not support",
        opts.reduceOp,
        "on complex tensors");
    tensor = at::view_as_real(tensor);
  }
  ++sequence_number_;
  auto work = all_reduce(
      tensor, opts.reduceOp, opts.asyncOp, operationTimeout(opts.timeout));
  work->setOutputs(tensors);
  return work;
}

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCL::allreduce_sparse(
    std::vector<at::Tensor>& /* tensors */,
    const ::c10d::AllreduceOptions& /* opts */) {
  C10_THROW_ERROR(
      Error,
      "NCCL does not support all_reduce with sparse tensors. Please use dense tensors instead.");
}

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const ::c10d::AllreduceCoalescedOptions& opts) {
  TORCH_CHECK(!tensors.empty(), "Tensor list must be nonempty");
  ++sequence_number_;
  std::vector<c10::intrusive_ptr<WorkNCCL>> works;
  works.reserve(tensors.size());
  for (auto& tensor : tensors) {
    works.push_back(all_reduce(
        tensor, opts.reduceOp, opts.asyncOp, operationTimeout(opts.timeout)));
  }
  auto work = coalesceWorks(std::move(works), tensors);
  if (coalescing_batch_) {
    coalesced_work_ = work;
  }
  return work;
}

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ::c10d::ReduceOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, "Only single tensor supported");
  auto tensor = tensors.at(0);
  if (tensor.is_complex()) {
    TORCH_CHECK(
        ::c10d::isComplexViewAsRealAllowed(opts.reduceOp),
        "reduce does not support",
        opts.reduceOp,
        "on complex tensors");
    tensor = at::view_as_real(tensor);
  }
  ++sequence_number_;
  auto work = reduceImpl(
      tensor,
      static_cast<int>(opts.rootRank),
      opts.reduceOp,
      opts.asyncOp,
      operationTimeout(opts.timeout));
  work->setOutputs(tensors);
  return work;
}

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const ::c10d::AllgatherOptions& opts) {
  TORCH_CHECK(
      outputTensors.size() == 1 && inputTensors.size() == 1,
      "Only single tensor / single list supported");
  ++sequence_number_;
  const auto& input = inputTensors.at(0);
  auto& outputList = outputTensors.at(0);
  TORCH_CHECK(
      static_cast<int>(outputList.size()) == getSize(),
      "Expected ",
      getSize(),
      " output tensors, got ",
      outputList.size());
  auto timeout = operationTimeout(opts.timeout);

  // Fast path: distinct per-rank output buffers -> list-based all_gather.
  bool aliased = outputList.size() > 1 &&
      outputList[0].data_ptr() == outputList[1].data_ptr();
  if (!aliased) {
    auto work = all_gather(outputList, input, opts.asyncOp, timeout);
    work->setOutputs(outputList);
    return work;
  }

  // Slow path (aliased outputs): gather into a contiguous staging tensor and
  // copy each rank's row back (port of BackendWrapper).
  auto staging = at::empty(
      {static_cast<int64_t>(getSize()) * input.numel()},
      input.options().memory_format(at::MemoryFormat::Contiguous));
  auto work = allGatherSingleImpl(staging, input, opts.asyncOp, timeout);
  work->setOutputs(outputList);
  auto rows = staging.view({getSize(), input.numel()});
  work->wait();
  for (int r = 0; r < getSize(); ++r) {
    outputList.at(r).copy_(rows[r].view_as(outputList.at(r)));
  }
  return work;
}

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& outputTensorLists,
    std::vector<at::Tensor>& inputTensors,
    const ::c10d::AllgatherOptions& opts) {
  TORCH_CHECK(
      !inputTensors.empty() && outputTensorLists.size() == inputTensors.size(),
      "Input and output tensor lists must have the same nonzero size");
  ++sequence_number_;
  std::vector<c10::intrusive_ptr<WorkNCCL>> works;
  std::vector<at::Tensor> outputs;
  works.reserve(inputTensors.size());
  for (const auto i : c10::irange(inputTensors.size())) {
    works.push_back(all_gather(
        outputTensorLists.at(i),
        inputTensors.at(i),
        opts.asyncOp,
        operationTimeout(opts.timeout)));
    outputs.insert(
        outputs.end(),
        outputTensorLists.at(i).begin(),
        outputTensorLists.at(i).end());
  }
  auto work = coalesceWorks(std::move(works), std::move(outputs));
  if (coalescing_batch_) {
    coalesced_work_ = work;
  }
  return work;
}

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCL::
    allgather_into_tensor_coalesced(
        std::vector<at::Tensor>& outputs,
        std::vector<at::Tensor>& inputs,
        const ::c10d::AllgatherOptions& opts) {
  TORCH_CHECK(
      !inputs.empty() && outputs.size() == inputs.size(),
      "Input and output tensor lists must have the same nonzero size");
  ++sequence_number_;
  std::vector<c10::intrusive_ptr<WorkNCCL>> works;
  works.reserve(inputs.size());
  for (const auto i : c10::irange(inputs.size())) {
    works.push_back(allGatherSingleImpl(
        outputs.at(i),
        inputs.at(i),
        opts.asyncOp,
        operationTimeout(opts.timeout)));
  }
  auto work = coalesceWorks(std::move(works), outputs);
  if (coalescing_batch_) {
    coalesced_work_ = work;
  }
  return work;
}

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCL::_allgather_base(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    const ::c10d::AllgatherOptions& opts) {
  if (inputBuffer.dtype() != outputBuffer.dtype()) {
    C10_THROW_ERROR(
        TypeError, "output tensor must have the same type as input tensor");
  }
  if (inputBuffer.numel() * getSize() != outputBuffer.numel()) {
    C10_THROW_ERROR(
        ValueError,
        "output tensor size must be equal to world_size times input tensor size");
  }
  ++sequence_number_;
  auto work = allGatherSingleImpl(
      outputBuffer, inputBuffer, opts.asyncOp, operationTimeout(opts.timeout));
  work->setOutputs(std::vector<at::Tensor>{outputBuffer});
  return work;
}

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCL::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const ::c10d::GatherOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    C10_THROW_ERROR(ValueError, "ProcessGroupNCCL::gather: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, getSize());
  TORCH_CHECK(inputTensors.size() == 1, "Only single input tensor supported");
  std::vector<at::Tensor> outputs;
  if (getRank() == opts.rootRank) {
    assertGatherOutputTensorList(invalidArgument, outputTensors, getSize());
    assertTypeAndSizesMatch(
        invalidArgument,
        outputTensors[0],
        inputTensors[0].options(),
        inputTensors[0].sizes());
    outputs = outputTensors[0];
  } else {
    assertEmptyOutputTensorList(invalidArgument, outputTensors);
  }
  ++sequence_number_;
  auto work = gatherImpl(
      outputs,
      inputTensors.at(0),
      static_cast<int>(opts.rootRank),
      opts.asyncOp,
      operationTimeout(opts.timeout));
  work->setOutputs(std::move(outputs));
  return work;
}

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCL::gather_single(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    const ::c10d::GatherOptions& opts) {
  TORCH_CHECK(
      opts.rootRank >= 0 && opts.rootRank < getSize(),
      "invalid root rank: ",
      opts.rootRank);
  // outputBuffer is only meaningful on the root; elsewhere the caller passes an
  // empty placeholder and gatherImpl ignores the (empty) output list.
  std::vector<at::Tensor> outputList;
  if (getRank() == opts.rootRank) {
    ensureTensorContiguous(outputBuffer);
    TORCH_CHECK(
        inputBuffer.dtype() == outputBuffer.dtype(),
        "output tensor must have the same type as input tensor");
    const auto count = inputBuffer.numel();
    TORCH_CHECK(
        outputBuffer.numel() == count * getSize(),
        "output tensor size must be equal to world_size times input tensor size");
    auto flat = outputBuffer.view(-1);
    outputList.reserve(getSize());
    for (const auto r : c10::irange(getSize())) {
      outputList.push_back(flat.narrow(0, r * count, count));
    }
  }
  ++sequence_number_;
  auto work = gatherImpl(
      outputList,
      inputBuffer,
      static_cast<int>(opts.rootRank),
      opts.asyncOp,
      operationTimeout(opts.timeout));
  work->setOutputs(std::vector<at::Tensor>{outputBuffer});
  return work;
}

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCL::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ::c10d::ScatterOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    C10_THROW_ERROR(ValueError, "ProcessGroupNCCL::scatter: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, getSize());
  TORCH_CHECK(outputTensors.size() == 1, "Only single output tensor supported");
  std::vector<at::Tensor> inputs;
  if (getRank() == opts.rootRank) {
    assertScatterInputTensorList(invalidArgument, inputTensors, getSize());
    assertTypeAndSizesMatch(
        invalidArgument,
        inputTensors[0],
        outputTensors[0].options(),
        outputTensors[0].sizes());
    inputs = inputTensors[0];
  } else {
    assertEmptyInputTensorList(invalidArgument, inputTensors);
  }
  ++sequence_number_;
  auto work = scatterImpl(
      outputTensors.at(0),
      inputs,
      static_cast<int>(opts.rootRank),
      opts.asyncOp,
      operationTimeout(opts.timeout));
  work->setOutputs(outputTensors);
  return work;
}

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ::c10d::ReduceScatterOptions& opts) {
  TORCH_CHECK(
      outputTensors.size() == 1 && inputTensors.size() == 1,
      "Only single tensor / single list supported");
  ++sequence_number_;
  auto work = reduce_scatter(
      outputTensors.at(0),
      inputTensors.at(0),
      opts.reduceOp,
      opts.asyncOp,
      operationTimeout(opts.timeout));
  work->setOutputs(outputTensors);
  return work;
}

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCL::
    reduce_scatter_tensor_coalesced(
        std::vector<at::Tensor>& outputs,
        std::vector<at::Tensor>& inputs,
        const ::c10d::ReduceScatterOptions& opts) {
  TORCH_CHECK(
      !outputs.empty() && inputs.size() == outputs.size(),
      "Input and output tensor lists must have the same nonzero size");
  ++sequence_number_;
  std::vector<c10::intrusive_ptr<WorkNCCL>> works;
  works.reserve(outputs.size());
  for (const auto i : c10::irange(outputs.size())) {
    works.push_back(reduceScatterSingleImpl(
        outputs.at(i),
        inputs.at(i),
        opts.reduceOp,
        opts.asyncOp,
        operationTimeout(opts.timeout)));
  }
  auto work = coalesceWorks(std::move(works), outputs);
  if (coalescing_batch_) {
    coalesced_work_ = work;
  }
  return work;
}

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCL::_reduce_scatter_base(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    const ::c10d::ReduceScatterOptions& opts) {
  if (inputBuffer.dtype() != outputBuffer.dtype()) {
    C10_THROW_ERROR(
        TypeError, "input tensor must be the same type as the output tensor.");
  }
  if (inputBuffer.numel() != outputBuffer.numel() * getSize()) {
    C10_THROW_ERROR(
        ValueError,
        "input tensor must be the same size as output size times world size");
  }
  ++sequence_number_;
  auto work = reduceScatterSingleImpl(
      outputBuffer,
      inputBuffer,
      opts.reduceOp,
      opts.asyncOp,
      operationTimeout(opts.timeout));
  work->setOutputs(std::vector<at::Tensor>{outputBuffer});
  return work;
}

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCL::alltoall_base(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const ::c10d::AllToAllOptions& opts) {
  ++sequence_number_;
  auto timeout = operationTimeout(opts.timeout);
  if (outputSplitSizes.empty() && inputSplitSizes.empty()) {
    c10d::checkSplitSizes(inputSplitSizes, inputBuffer, size_);
    c10d::checkSplitSizes(outputSplitSizes, outputBuffer, size_);
    auto work =
        allToAllSingleImpl(outputBuffer, inputBuffer, opts.asyncOp, timeout);
    work->setOutputs(std::vector<at::Tensor>{outputBuffer});
    return work;
  }
  auto work = all_to_all_v_single(
      outputBuffer,
      inputBuffer,
      normalizeSplitSizes(outputSplitSizes, outputBuffer, size_),
      normalizeSplitSizes(inputSplitSizes, inputBuffer, size_),
      opts.asyncOp,
      timeout);
  work->setOutputs(std::vector<at::Tensor>{outputBuffer});
  return work;
}

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCL::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const ::c10d::AllToAllOptions& opts) {
  TORCH_CHECK(!inputTensors.empty(), "alltoall requires input tensors");
  ++sequence_number_;
  auto work = all_to_all(
      outputTensors,
      inputTensors,
      opts.asyncOp,
      operationTimeout(opts.timeout));
  work->setOutputs(outputTensors);
  return work;
}

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCL::barrier(
    const ::c10d::BarrierOptions& opts) {
  ++sequence_number_;
  return barrierImpl(/*async_op=*/false, operationTimeout(opts.timeout));
}

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCL::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    [[maybe_unused]] int tag) {
  TORCH_CHECK(tensors.size() == 1, "Only single tensor supported");
  if (coalescing_batch_.has_value()) {
    coalescing_batch_->send(tensors.at(0), dstRank);
    return c10::make_intrusive<CompletedWork>(tensors);
  }
  auto work = sendImpl(
      tensors.at(0), dstRank, /*async_op=*/true, options_c10d_->timeout);
  work->setOutputs(tensors);
  return work;
}

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCL::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    [[maybe_unused]] int tag) {
  TORCH_CHECK(tensors.size() == 1, "Only single tensor supported");
  if (coalescing_batch_.has_value()) {
    coalescing_batch_->recv(tensors.at(0), srcRank);
    return c10::make_intrusive<CompletedWork>(tensors);
  }
  auto work = recvImpl(
      tensors.at(0), srcRank, /*async_op=*/true, options_c10d_->timeout);
  work->setOutputs(tensors);
  return work;
}

void ProcessGroupNCCL::startCoalescing() {
  TORCH_CHECK(
      !coalescing_batch_.has_value(),
      "startCoalescing called while a batch is already active");
  coalesced_work_.reset();
  coalescing_batch_.emplace();
}

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCL::endCoalescing() {
  TORCH_CHECK(
      coalescing_batch_.has_value(),
      "endCoalescing called without a matching startCoalescing");
  auto batch = std::move(*coalescing_batch_);
  coalescing_batch_.reset();
  if (batch.ops.empty()) {
    if (coalesced_work_) {
      auto work = std::move(coalesced_work_);
      coalesced_work_.reset();
      return work;
    }
    return c10::make_intrusive<CompletedWork>();
  }
  return batch_op_issue(batch.ops, /*async_op=*/true, options_c10d_->timeout);
}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
