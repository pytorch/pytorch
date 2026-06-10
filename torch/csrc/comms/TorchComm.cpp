// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/comms/TorchComm.hpp>
#include <torch/csrc/comms/TorchCommFactory.hpp>

#include <atomic>
#include <limits>

namespace torch::comms {

namespace {

// Singleton to generate globally unique increasing op_ids across all TorchComm
// instances. This ensures that when multiple communicators share the same
// FlightRecorder, their op_id values don't collide.
class GlobalOpIdGenerator {
 public:
  static GlobalOpIdGenerator& instance() {
    static GlobalOpIdGenerator inst;
    return inst;
  }

  size_t nextOpId() {
    return nextOpId_.fetch_add(1, std::memory_order_relaxed);
  }

  // Reset the counter to 0. Used when creating isolated FlightRecorder
  // instances to ensure each test gets a fresh op_id space.
  void reset() {
    nextOpId_.store(0, std::memory_order_relaxed);
  }

  GlobalOpIdGenerator(const GlobalOpIdGenerator&) = delete;
  GlobalOpIdGenerator& operator=(const GlobalOpIdGenerator&) = delete;

 private:
  GlobalOpIdGenerator() = default;
  std::atomic<size_t> nextOpId_{0};
};

} // namespace

void resetGlobalOpIdGenerator() {
  GlobalOpIdGenerator::instance().reset();
}

TorchComm::TorchComm(
    const std::string& backend_name,
    std::shared_ptr<TorchCommBackend> impl)
    : backend_(backend_name), impl_(std::move(impl)) {
  // In dynamic regime (enable_reconfigure), rank/size are not known until
  // reconfigure() is called. Defer ranks_ initialization to initRanks().
  if (!impl_->getOptions().enable_reconfigure) {
    initRanks();
  }
}

void TorchComm::initRanks() {
  if (!impl_->isInitialized()) {
    return;
  }

  int size = impl_->getSize();
  ranks_.clear();
  ranks_.reserve(size);
  for (int i = 0; i < size; ++i) {
    ranks_.push_back(i);
  }
}

TorchComm::TorchComm(
    const std::string& backend_name,
    std::shared_ptr<TorchCommBackend> impl,
    std::vector<int> ranks)
    : backend_(backend_name),
      impl_(std::move(impl)),
      ranks_(std::move(ranks)) {}

std::shared_ptr<TorchComm> new_comm(
    const std::string& backend_name,
    at::Device device,
    const std::string& name,
    const CommOptions& options) {
  auto backend_impl = TorchCommFactory::get().create_backend(
      backend_name, device, name, options);

  return std::shared_ptr<TorchComm>(
      new TorchComm(backend_name, std::move(backend_impl)));
}

void TorchComm::finalize() {
  // finalize is a lifecycle event, not a collective — don't consume a
  // global op_id slot. Built-in hooks (FlightRecorder, Clog) all
  // short-circuit on finalize, and FlightRecorder keys its ring-buffer
  // slot off op_id. Bumping the global counter here would leave a hole
  // in the FR's index space. Use a sentinel op_id instead so
  // pre/post hooks still pair on the same value for finalize-aware
  // Python custom hooks.
  constexpr size_t kFinalizeOpId = std::numeric_limits<size_t>::max();
  preHook(kFinalizeOpId, FinalizePreHookArgs{});
  impl_->finalize();
  postHook(kFinalizeOpId, FinalizePostHookArgs{});
}

int TorchComm::getRank() const {
  return impl_->getRank();
}

int TorchComm::getSize() const {
  return impl_->getSize();
}

std::vector<int> TorchComm::getRanks() const {
  return ranks_;
}

std::string_view TorchComm::getCommName() const {
  return impl_->getCommName();
}

std::string_view TorchComm::getBackendVersion() const {
  return impl_->getBackendVersion();
}

void TorchComm::validateRank(int rank, const char* param_name) const {
  TORCH_CHECK(
      rank >= 0 && rank < getSize(),
      param_name,
      " must be in range [0, ",
      getSize(),
      "), but got ",
      rank);
}

// Point-to-Point Operations
c10::intrusive_ptr<TorchWork> TorchComm::send(
    const at::Tensor& tensor,
    int dst,
    bool async_op,
    const SendOptions& options) {
  validateRank(dst, "dst");
  auto op_id = GlobalOpIdGenerator::instance().nextOpId();
  preHook(op_id, SendPreHookArgs(tensor, dst, async_op));

  auto work = impl_->send(tensor, dst, async_op, options);

  postHook(op_id, SendPostHookArgs(c10::weak_intrusive_ptr<TorchWork>(work)));

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::recv(
    at::Tensor& tensor,
    int src,
    bool async_op,
    const RecvOptions& options) {
  validateRank(src, "src");
  auto op_id = GlobalOpIdGenerator::instance().nextOpId();
  preHook(op_id, RecvPreHookArgs(tensor, src, async_op));

  auto work = impl_->recv(tensor, src, async_op, options);

  postHook(op_id, RecvPostHookArgs(c10::weak_intrusive_ptr<TorchWork>(work)));

  return work;
}

// Collective Operations
c10::intrusive_ptr<TorchWork> TorchComm::broadcast(
    at::Tensor& tensor,
    int root,
    bool async_op,
    const BroadcastOptions& options) {
  validateRank(root, "root");
  auto op_id = GlobalOpIdGenerator::instance().nextOpId();
  preHook(op_id, BroadcastPreHookArgs(tensor, root, async_op));

  auto work = impl_->broadcast(tensor, root, async_op, options);

  postHook(
      op_id, BroadcastPostHookArgs(c10::weak_intrusive_ptr<TorchWork>(work)));

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::all_reduce(
    at::Tensor& tensor,
    const ReduceOp& op,
    bool async_op,
    const AllReduceOptions& options) {
  auto op_id = GlobalOpIdGenerator::instance().nextOpId();
  preHook(op_id, AllReducePreHookArgs(tensor, op, async_op));

  auto work = impl_->all_reduce(tensor, op, async_op, options);

  postHook(
      op_id, AllReducePostHookArgs(c10::weak_intrusive_ptr<TorchWork>(work)));

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::reduce(
    const at::Tensor& tensor,
    int root,
    const ReduceOp& op,
    bool async_op,
    const ReduceOptions& options) {
  validateRank(root, "root");
  auto op_id = GlobalOpIdGenerator::instance().nextOpId();
  preHook(op_id, ReducePreHookArgs(tensor, root, op, async_op));

  auto work = impl_->reduce(tensor, root, op, async_op, options);

  postHook(op_id, ReducePostHookArgs(c10::weak_intrusive_ptr<TorchWork>(work)));

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::all_gather(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& options) {
  auto op_id = GlobalOpIdGenerator::instance().nextOpId();
  preHook(op_id, AllGatherPreHookArgs(tensor, tensor_list, async_op));

  auto work = impl_->all_gather(tensor_list, tensor, async_op, options);

  postHook(
      op_id, AllGatherPostHookArgs(c10::weak_intrusive_ptr<TorchWork>(work)));

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::all_gather_v(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& options) {
  auto op_id = GlobalOpIdGenerator::instance().nextOpId();
  preHook(op_id, AllGatherVPreHookArgs(tensor, tensor_list, async_op));

  auto work = impl_->all_gather_v(tensor_list, tensor, async_op, options);

  postHook(
      op_id, AllGatherVPostHookArgs(c10::weak_intrusive_ptr<TorchWork>(work)));

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::all_gather_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllGatherSingleOptions& options) {
  auto op_id = GlobalOpIdGenerator::instance().nextOpId();
  preHook(op_id, AllGatherSinglePreHookArgs(input, output, async_op));

  auto work = impl_->all_gather_single(output, input, async_op, options);

  postHook(
      op_id,
      AllGatherSinglePostHookArgs(c10::weak_intrusive_ptr<TorchWork>(work)));

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::reduce_scatter(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterOptions& options) {
  auto op_id = GlobalOpIdGenerator::instance().nextOpId();
  preHook(op_id, ReduceScatterPreHookArgs(input_list, output, op, async_op));

  auto work = impl_->reduce_scatter(output, input_list, op, async_op, options);

  postHook(
      op_id,
      ReduceScatterPostHookArgs(c10::weak_intrusive_ptr<TorchWork>(work)));

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::reduce_scatter_v(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterOptions& options) {
  auto op_id = GlobalOpIdGenerator::instance().nextOpId();
  preHook(op_id, ReduceScatterVPreHookArgs(input_list, output, op, async_op));

  auto work =
      impl_->reduce_scatter_v(output, input_list, op, async_op, options);

  postHook(
      op_id,
      ReduceScatterVPostHookArgs(c10::weak_intrusive_ptr<TorchWork>(work)));

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::reduce_scatter_single(
    at::Tensor& output,
    const at::Tensor& input,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterSingleOptions& options) {
  auto op_id = GlobalOpIdGenerator::instance().nextOpId();
  preHook(op_id, ReduceScatterSinglePreHookArgs(input, output, op, async_op));

  auto work =
      impl_->reduce_scatter_single(output, input, op, async_op, options);

  postHook(
      op_id,
      ReduceScatterSinglePostHookArgs(
          c10::weak_intrusive_ptr<TorchWork>(work)));

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::all_to_all_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllToAllSingleOptions& options) {
  auto op_id = GlobalOpIdGenerator::instance().nextOpId();
  preHook(op_id, AllToAllSinglePreHookArgs(input, output, async_op));

  auto work = impl_->all_to_all_single(output, input, async_op, options);

  postHook(
      op_id,
      AllToAllSinglePostHookArgs(c10::weak_intrusive_ptr<TorchWork>(work)));

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::all_to_all_v_single(
    at::Tensor& output,
    const at::Tensor& input,
    const std::vector<uint64_t>& output_split_sizes,
    const std::vector<uint64_t>& input_split_sizes,
    bool async_op,
    const AllToAllvSingleOptions& options) {
  auto op_id = GlobalOpIdGenerator::instance().nextOpId();
  preHook(
      op_id,
      AllToAllVSinglePreHookArgs(
          input, output, input_split_sizes, output_split_sizes, async_op));

  auto work = impl_->all_to_all_v_single(
      output, input, output_split_sizes, input_split_sizes, async_op, options);

  postHook(
      op_id,
      AllToAllVSinglePostHookArgs(c10::weak_intrusive_ptr<TorchWork>(work)));

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::all_to_all(
    const std::vector<at::Tensor>& output_tensor_list,
    const std::vector<at::Tensor>& input_tensor_list,
    bool async_op,
    const AllToAllOptions& options) {
  auto op_id = GlobalOpIdGenerator::instance().nextOpId();
  preHook(
      op_id,
      AllToAllPreHookArgs(input_tensor_list, output_tensor_list, async_op));

  auto work = impl_->all_to_all(
      output_tensor_list, input_tensor_list, async_op, options);

  postHook(
      op_id, AllToAllPostHookArgs(c10::weak_intrusive_ptr<TorchWork>(work)));

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::barrier(
    bool async_op,
    const BarrierOptions& options) {
  auto op_id = GlobalOpIdGenerator::instance().nextOpId();
  preHook(op_id, BarrierPreHookArgs(async_op));

  auto work = impl_->barrier(async_op, options);

  postHook(
      op_id, BarrierPostHookArgs(c10::weak_intrusive_ptr<TorchWork>(work)));

  return work;
}

// Scatter and Gather Operations
c10::intrusive_ptr<TorchWork> TorchComm::scatter(
    at::Tensor& output_tensor,
    const std::vector<at::Tensor>& input_tensor_list,
    int root,
    bool async_op,
    const ScatterOptions& options) {
  validateRank(root, "root");
  auto op_id = GlobalOpIdGenerator::instance().nextOpId();
  preHook(
      op_id,
      ScatterPreHookArgs(output_tensor, input_tensor_list, root, async_op));

  auto work =
      impl_->scatter(output_tensor, input_tensor_list, root, async_op, options);

  postHook(
      op_id, ScatterPostHookArgs(c10::weak_intrusive_ptr<TorchWork>(work)));

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::gather(
    const std::vector<at::Tensor>& output_tensor_list,
    const at::Tensor& input_tensor,
    int root,
    bool async_op,
    const GatherOptions& options) {
  validateRank(root, "root");
  auto op_id = GlobalOpIdGenerator::instance().nextOpId();
  preHook(
      op_id,
      GatherPreHookArgs(input_tensor, output_tensor_list, root, async_op));

  auto work =
      impl_->gather(output_tensor_list, input_tensor, root, async_op, options);

  postHook(op_id, GatherPostHookArgs(c10::weak_intrusive_ptr<TorchWork>(work)));

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::gather_single(
    at::Tensor& output,
    const at::Tensor& input,
    int root,
    bool async_op,
    const GatherSingleOptions& options) {
  validateRank(root, "root");
  auto op_id = GlobalOpIdGenerator::instance().nextOpId();
  preHook(op_id, GatherSinglePreHookArgs(input, output, root, async_op));

  auto work = impl_->gather_single(output, input, root, async_op, options);

  postHook(
      op_id,
      GatherSinglePostHookArgs(c10::weak_intrusive_ptr<TorchWork>(work)));

  return work;
}

std::shared_ptr<TorchCommWindow> TorchComm::new_window(
    const std::optional<at::Tensor>& tensor) {
  auto op_id = GlobalOpIdGenerator::instance().nextOpId();
  preHook(op_id, NewWindowPreHookArgs());
  auto window = impl_->new_window(tensor);
  postHook(
      op_id, NewWindowPostHookArgs(std::weak_ptr<TorchCommWindow>(window)));
  return window;
}

// Persistent AllGather operations
TorchComm::AllGatherPHandle TorchComm::all_gather_p_init(
    at::Tensor& output,
    const AllGatherPInitOptions& options) {
  return impl_->all_gather_p_init(output, options);
}

c10::intrusive_ptr<TorchWork> TorchComm::all_gather_p_exec(
    AllGatherPHandle handle,
    const at::Tensor& input,
    bool async_op,
    const AllGatherPExecOptions& options) {
  return impl_->all_gather_p_exec(handle, input, async_op, options);
}

void TorchComm::all_gather_p_free(AllGatherPHandle handle) {
  impl_->all_gather_p_free(handle);
}

// Fault Tolerance API
InitHandle TorchComm::getInitHandle() const {
  return impl_->getInitHandle();
}

c10::intrusive_ptr<TorchWork> TorchComm::reconfigure(
    const ReconfigureOptions& opts) {
  auto work = impl_->reconfigure(opts);
  work->waitBlocking();

  if (work->isCompleted()) {
    initRanks();
  } else {
    ranks_.clear();
  }

  return work;
}

void TorchComm::abort() {
  impl_->abort();
}

bool TorchComm::isAbortSupported() const {
  return impl_->isAbortSupported();
}

bool TorchComm::isAborted() const {
  return impl_->isAborted();
}

void TorchComm::tensor_register(const at::Tensor& tensor) {
  impl_->tensor_register(tensor);
}

void TorchComm::tensor_deregister(const at::Tensor& tensor) {
  impl_->tensor_deregister(tensor);
}

// Communicator Management
std::shared_ptr<TorchComm> TorchComm::split(
    const std::vector<int>& ranks,
    const std::string& name,
    const CommOptions& options) {
  auto op_id = GlobalOpIdGenerator::instance().nextOpId();
  preHook(op_id, SplitPreHookArgs(ranks, name));
  auto new_impl = impl_->split(ranks, name, options);
  if (new_impl == nullptr) {
    return nullptr;
  }
  // Map the local ranks to global ranks from this communicator
  std::vector<int> global_ranks;
  global_ranks.reserve(ranks.size());
  for (int local_rank : ranks) {
    global_ranks.push_back(ranks_[local_rank]);
  }
  auto comm = std::shared_ptr<TorchComm>(
      new TorchComm(backend_, std::move(new_impl), std::move(global_ranks)));
  postHook(op_id, SplitPostHookArgs(std::weak_ptr<TorchComm>(comm)));
  return comm;
}

const CommOptions& TorchComm::getOptions() const {
  return impl_->getOptions();
}

const at::Device& TorchComm::getDevice() const {
  return impl_->getDevice();
}

// Batch Operations

BatchSendRecv::BatchSendRecv(std::shared_ptr<TorchComm> parent)
    : parent_(std::move(parent)) {}

BatchSendRecv::P2POp::P2POp(OpType type, const at::Tensor& tensor, int peer) {
  this->type = type;
  this->tensor = tensor;
  this->peer = peer;
}

BatchSendRecv TorchComm::batch_op_create() {
  return BatchSendRecv(shared_from_this());
}

void BatchSendRecv::send(const at::Tensor& tensor, int dst) {
  auto op = P2POp(P2POp::OpType::SEND, tensor, dst);
  ops.push_back(op);
}

void BatchSendRecv::recv(at::Tensor& tensor, int src) {
  auto op = P2POp(P2POp::OpType::RECV, tensor, src);
  ops.push_back(op);
}

c10::intrusive_ptr<TorchWork> BatchSendRecv::issue(
    bool async_op,
    const BatchP2POptions& options) {
  auto op_id = GlobalOpIdGenerator::instance().nextOpId();
  parent_->preHook(op_id, BatchOpIssuePreHookArgs(ops.size(), async_op));

  auto work = parent_->getBackendImpl()->batch_op_issue(ops, async_op, options);

  parent_->postHook(
      op_id,
      BatchOpIssuePostHookArgs(c10::weak_intrusive_ptr<TorchWork>(work)));

  return work;
}

// Global memory allocator function implementation
std::shared_ptr<c10::Allocator> get_mem_allocator(const std::string& backend) {
  return TorchCommFactory::get().get_allocator(backend);
}

std::shared_ptr<c10::Allocator> TorchComm::getMemAllocator() const {
  return get_mem_allocator(backend_);
}

std::unique_ptr<RemovableHandle> TorchComm::registerPreHook(
    TorchComm::PreHook preHook) {
  auto hookId = nextHookId_++;
  preHooks_.emplace(hookId, std::move(preHook));
  return RemovableHandle::create([self = weak_from_this(), hookId]() {
    if (auto selfPtr = self.lock()) {
      selfPtr->preHooks_.erase(hookId);
    }
  });
}

std::unique_ptr<RemovableHandle> TorchComm::registerPostHook(
    TorchComm::PostHook postHook) {
  auto hookId = nextHookId_++;
  postHooks_.emplace(hookId, std::move(postHook));
  return RemovableHandle::create([self = weak_from_this(), hookId]() {
    if (auto selfPtr = self.lock()) {
      selfPtr->postHooks_.erase(hookId);
    }
  });
}

std::unique_ptr<RemovableHandle> TorchComm::registerAbortHook(
    TorchComm::AbortHook hook) {
  auto hookId = nextHookId_++;
  impl_->registerAbortHook(hookId, std::move(hook));
  return RemovableHandle::create([self = weak_from_this(), hookId]() {
    if (auto selfPtr = self.lock()) {
      selfPtr->impl_->unregisterAbortHook(hookId);
    }
  });
}

std::unique_ptr<RemovableHandle> TorchComm::registerGraphReplayHook(
    TorchComm::GraphReplayHook hook) {
  auto hookId = nextHookId_++;
  impl_->registerGraphReplayHook(hookId, std::move(hook));
  return RemovableHandle::create([self = weak_from_this(), hookId]() {
    if (auto selfPtr = self.lock()) {
      selfPtr->impl_->unregisterGraphReplayHook(hookId);
    }
  });
}

void TorchComm::preHook(size_t op_id, PreHookArgs&& args) {
  for (auto& hook : preHooks_) {
    hook.second(op_id, args);
  }
}

void TorchComm::postHook(size_t op_id, PostHookArgs&& args) {
  for (auto& hook : postHooks_) {
    hook.second(op_id, args);
  }
}

} // namespace torch::comms
