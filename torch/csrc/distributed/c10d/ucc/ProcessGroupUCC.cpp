#include <c10d/ucc/ProcessGroupUCC.hpp>
#include <c10d/ucc/UCXUtils.hpp>
#include <c10d/ucc/BlockingQueue.hpp>
#include <c10/macros/Export.h>
#include <thread>

// TODO support profiler:
// Reference PR: https://github.com/pytorch/pytorch/pull/52004/files

namespace {

static void check_tensor(const std::vector<at::Tensor>& tensors) {
  TORCH_CHECK(tensors.size() == 1, "ProcessGroupUCC takes 1 tensor");
  TORCH_CHECK(tensors[0].is_non_overlapping_and_dense(), "ProcessGroupUCC input tensor has to be non-overlapping and dense");
  TORCH_CHECK(!tensors[0].is_sparse(), "ProcessGroupUCC input tensor has to be dense");
}

namespace tagging {
// Note [Receive from an endpoint]:
// UCP does not support receiving from a specific endpoint. So we use tag
// matching to simulate this behavior. In PyTorch, the world_size is int,
// the tag is also int, and in UCP, the ucp_tag_t is uint64_t. So we use
// the higher 32 bits of ucp_tag_t for rank, and use lower 32 bits for the
// real tag. When receiving from a specified endpoint, the entire ucp_tag_t
// should match. And when receiving from any source, tag mask is used to
// disable the matching of the higher bits.

// TODO: add test for INT_MAX tag

using world_size_t = int;
using tag_t = int;

union tag_union {
  ucp_tag_t raw;
  struct fields_t {
    tag_t tag;
    world_size_t rank;
  } fields;
};

static_assert(
  sizeof(tag_union) == sizeof(ucp_tag_t) &&
  sizeof(tag_union) == sizeof(tag_union::fields_t),
  "The implementation of UCP tag matching has unsatisfied assumptions.");

constexpr ucp_tag_t wrap_tag(world_size_t rank, tag_t tag) {
  tag_union u = {
    .fields = { .tag = tag, .rank = rank }
  };
  return u.raw;
}

constexpr world_size_t get_rank_from_tag(ucp_tag_t tag) {
  tag_union u = { .raw = tag };
  return u.fields.rank;
}

constexpr ucp_tag_t any_source_mask() {
  return wrap_tag(0, ~tag_t(0));
}

constexpr ucp_tag_t complete_tag_mask() {
  return wrap_tag(~world_size_t(0), ~tag_t(0));
}

} // namespace tagging
} // namespace

namespace c10d {

constexpr const char* UCC_BACKEND_NAME = "_internal_ucc";

// ProcessGroupUCC implements UCC & UCX bindings for c10d. UCC is used for
// collective operations, and UCX is used for P2P operations.
//
// The UCC & UCX binding is not published to the user directly, but it provided
// a process group called `_internal_ucc`. The `_internal_ucc` is only for
// testing purposes, and for power users who really knows what they are doing.
//
// All functions of the class are expected to be called in the same order
// across all processes in the process group.  This is the only way that we
// can guarantee to match up the same calls among all processes.
//
// Links:
// ucx: https://github.com/openucx/ucx
// ucc: https://github.com/openucx/ucc
// Original torch_ucc: https://github.com/facebookresearch/torch_ucc
//
// *****************************************************************************
// This ProcessGroup is still under development, and there are some know issues:
// - Only send and recv are supported.
// - It is fake async: UCP worker are progressed only when checking status.
class ProcessGroupUCC final : public ProcessGroup {
public:
  class WorkUCP : public ProcessGroup::Work {
    std::shared_ptr<UCPRequest> request;
    std::shared_ptr<UCPWorker> worker;
  public:
    WorkUCP(
      const std::shared_ptr<UCPWorker> &worker,
      const std::shared_ptr<UCPRequest> &request,
      int rank, OpType opType,
      const char* profilingTitle = nullptr,
      const c10::optional<std::vector<at::Tensor>>& inputs = c10::nullopt)
      : Work(rank, opType, profilingTitle, inputs), request(request), worker(worker) {}

    bool isCompleted() override {
      worker->progress();
      bool is_finished = (request->status() != UCS_INPROGRESS);
      if (is_finished) {
        finish();
      }
      return is_finished;
    }

    bool isSuccess() const override {
      return request->status() == UCS_OK;
    }

    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override {
      while(!isCompleted());
      return true;
    }

    tagging::world_size_t sourceRank() const override {
      return tagging::get_rank_from_tag(request->info().sender_tag);
    }
  };

  explicit ProcessGroupUCC(
      const c10::intrusive_ptr<Store>& store,
      tagging::world_size_t rank,
      tagging::world_size_t size);

  ~ProcessGroupUCC();

  const std::string getBackendName() const override {
      return std::string(UCC_BACKEND_NAME);
  }

  c10::intrusive_ptr<ProcessGroup::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> gather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> scatter(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      const ScatterOptions& opts = ScatterOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> reduce_scatter(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> send(
      std::vector<at::Tensor>& tensors,
      tagging::world_size_t dstRank,
      tagging::tag_t tag) override;

  c10::intrusive_ptr<ProcessGroup::Work> recv(
      std::vector<at::Tensor>& tensors,
      tagging::world_size_t srcRank,
      tagging::tag_t tag) override;

  c10::intrusive_ptr<ProcessGroup::Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      tagging::tag_t tag) override;

  c10::intrusive_ptr<ProcessGroup::Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

private:
  c10::intrusive_ptr<Store> store;
  std::shared_ptr<UCPWorker> worker;
  std::vector<std::shared_ptr<UCPEndpoint>> ucp_endpoints = {};
  BlockingQueue<c10::intrusive_ptr<Work>> pending_works;

  std::thread book_keeper;  // background thread that progresses pending work
  bool stop_book_keeper = false;
  static void book_keeper_fn(ProcessGroupUCC *);
};

ProcessGroupUCC::ProcessGroupUCC(
    const c10::intrusive_ptr<Store>& store,
    tagging::world_size_t rank,
    tagging::world_size_t size): ProcessGroup(rank, size),
  store(store), worker(std::make_shared<UCPWorker>()),
  book_keeper(book_keeper_fn, this)
{
  store->set("ucp_address:" + std::to_string(rank_), worker->address());
  for (int i = 0; i < size_; i++) {
    UCPWorker::Address peer_addr = store->get("ucp_address:" + std::to_string(i));
    ucp_endpoints.emplace_back(worker->connect(peer_addr));
  }
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  TORCH_CHECK(false, "ProcessGroupUCC does not support broadcast");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  TORCH_CHECK(false, "ProcessGroupUCC does not support allreduce");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  TORCH_CHECK(false, "ProcessGroupUCC does not support allreduce_coalesced");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  TORCH_CHECK(false, "ProcessGroupUCC does not support reduce");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::allgather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& opts) {
  TORCH_CHECK(false, "ProcessGroupUCC does not support allgather");
};

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& outputTensorLists,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  TORCH_CHECK(false, "ProcessGroupUCC does not support allgather_coalesced");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::gather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    const GatherOptions& opts) {
  TORCH_CHECK(false, "ProcessGroupUCC does not support gather");
};

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    const ScatterOptions& opts) {
  TORCH_CHECK(false, "ProcessGroupUCC does not support scatter");
};

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::reduce_scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    const ReduceScatterOptions& opts) {
  TORCH_CHECK(false, "ProcessGroupUCC does not support reduce_scatter");
};

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& opts) {
  TORCH_CHECK(false, "ProcessGroupUCC does not support alltoall_base");
};

// See note: [Receive from an endpoint]
c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::send(
    std::vector<at::Tensor>& tensors,
    tagging::world_size_t dstRank,
    tagging::tag_t tag) {
  check_tensor(tensors);
  TORCH_CHECK(dstRank < ucp_endpoints.size(), "Invalid dest rank");
  auto& tensor = tensors[0];
  auto request = ucp_endpoints[dstRank]->send_with_tag(
    tensor.data_ptr(), tensor.element_size() * tensor.numel(),
    tagging::wrap_tag(rank_, tag), tensor.device().type());
  auto work = c10::make_intrusive<ProcessGroupUCC::WorkUCP>(
    worker, request, getRank(), OpType::SEND, "ucc:send", tensors);
  pending_works.push(work);
  return work;
}

// See note: [Receive from an endpoint]
c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::recv(
    std::vector<at::Tensor>& tensors,
    tagging::world_size_t srcRank,
    tagging::tag_t tag) {
  check_tensor(tensors);
  TORCH_CHECK(srcRank < ucp_endpoints.size(), "Invalid src rank");
  auto& tensor = tensors[0];
  auto request = worker->recv_with_tag_and_mask(
    tensor.data_ptr(), tensor.element_size() * tensor.numel(),
    tagging::wrap_tag(srcRank, tag), tagging::complete_tag_mask(), tensor.device().type());
  auto work = c10::make_intrusive<ProcessGroupUCC::WorkUCP>(
    worker, request, getRank(), OpType::RECV, "ucc:recv", tensors);
  pending_works.push(work);
  return work;
}

// See note: [Receive from an endpoint]
c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::recvAnysource(
    std::vector<at::Tensor>& tensors,
    tagging::tag_t tag) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  auto request = worker->recv_with_tag_and_mask(
    tensor.data_ptr(), tensor.element_size() * tensor.numel(),
    tagging::wrap_tag(0, tag), tagging::any_source_mask(), tensor.device().type());
  auto work = c10::make_intrusive<ProcessGroupUCC::WorkUCP>(
    worker, request, getRank(), OpType::RECVANYSOURCE, "ucc:recvAnySource", tensors);
  pending_works.push(work);
  return work;
};

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::barrier(
    const BarrierOptions& /* unused */) {
  TORCH_CHECK(false, "ProcessGroupUCC does not support barrier");
};

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::_allgather_base(
    at::Tensor& /*unused */,
    at::Tensor& /*unused */,
    const AllgatherOptions& /*unused */) {
  TORCH_CHECK(false, "ProcessGroupUCC does not support _allgather_base");
}

void ProcessGroupUCC::book_keeper_fn(ProcessGroupUCC *pg) {
  while (!pg->stop_book_keeper) {
    auto maybe_work = pg->pending_works.pop();
    if (maybe_work) {
      auto work = maybe_work.value();
      if (!work->isCompleted()) {  // `isCompleted` also progress the worker
        pg->pending_works.push(work);
      }
    }
  }
}

ProcessGroupUCC::~ProcessGroupUCC() {
  stop_book_keeper = true;
  book_keeper.join();
}

} // namespace c10d

C10_EXPORT c10::intrusive_ptr<c10d::ProcessGroup> createProcessGroupUCC(
  const c10::intrusive_ptr<c10d::Store>& store,
  tagging::world_size_t rank,
  tagging::world_size_t size) {
  return c10::make_intrusive<c10d::ProcessGroupUCC>(store, rank, size);
}

static_assert(std::is_same<c10d::CreateProcessGroupUCCType, decltype(&createProcessGroupUCC)>::value,
  "CreateProcessGroupUCCType mismatch with createProcessGroupUCC, "
  "if you changed one of them, you should change the other as well");
