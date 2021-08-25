#include <c10d/ucc/ProcessGroupUCC.hpp>
#include <c10d/ucc/UCXUtils.hpp>
#include <c10/macros/Export.h>

// TODO support profiler:
// Reference PR: https://github.com/pytorch/pytorch/pull/52004/files

namespace {

static void check_tensor(const std::vector<at::Tensor>& tensors) {
  TORCH_CHECK(tensors.size() == 1, "ProcessGroupUCC takes 1 tensor");
  TORCH_CHECK(tensors[0].is_non_overlapping_and_dense(), "ProcessGroupUCC input tensor has to be non-overlapping and dense");
  TORCH_CHECK(!tensors[0].is_sparse(), "ProcessGroupUCC input tensor has to be dense");
}

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
    WorkUCP(const std::shared_ptr<UCPWorker> &worker, const std::shared_ptr<UCPRequest> &request)
      : request(request), worker(worker) {}

    bool isCompleted() override {
      // TODO: progress worker in a side thread for true async
      worker->progress();
      return request->status() != UCS_INPROGRESS;
    }

    bool isSuccess() const override {
      // TODO: progress worker in a side thread for true async
      worker->progress();
      return request->status() == UCS_OK;
    }

    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override {
      while(!isCompleted());
      return true;
    }

    int sourceRank() const {
      return get_rank_from_tag(request->info().sender_tag);
    }
  };

  explicit ProcessGroupUCC(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size);

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
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<ProcessGroup::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  c10::intrusive_ptr<ProcessGroup::Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override;

  c10::intrusive_ptr<ProcessGroup::Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

private:
  c10::intrusive_ptr<Store> store;
  std::shared_ptr<UCPWorker> worker;
  std::vector<std::shared_ptr<UCPEndpoint>> ucp_endpoints = {};
};

ProcessGroupUCC::ProcessGroupUCC(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size) : ProcessGroup(rank, size), store(store), worker(std::make_shared<UCPWorker>()) {
  static_assert(std::is_same<decltype(size), world_size_type>::value,
    "If you updated the type of `size`, please check with note [Receive from an endpoint]."
  );
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
    int dstRank,
    int tag) {
  static_assert(std::is_same<decltype(tag), tag_type>::value,
    "If you updated the type of tag, please check with note [Receive from an endpoint]."
  );
  check_tensor(tensors);
  TORCH_CHECK(dstRank < ucp_endpoints.size(), "Invalid dest rank");
  auto& tensor = tensors[0];
  std::cout << "send tag: " << std::hex << wrap_tag(rank_, tag) << std::endl;
  auto request = ucp_endpoints[dstRank]->send_with_tag(
    tensor.data_ptr(), tensor.element_size() * tensor.numel(),
    wrap_tag(rank_, tag), tensor.device().type());
  return c10::make_intrusive<ProcessGroupUCC::WorkUCP>(worker, request);
}

// See note: [Receive from an endpoint]
c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  static_assert(std::is_same<decltype(tag), tag_type>::value,
    "If you updated the type of tag, please check with note [Receive from an endpoint]."
  );
  check_tensor(tensors);
  TORCH_CHECK(srcRank < ucp_endpoints.size(), "Invalid src rank");
  auto& tensor = tensors[0];
  std::cout << "recv tag: " << std::hex << wrap_tag(srcRank, tag) << std::endl;
  auto request = worker->recv_with_tag_and_mask(
    tensor.data_ptr(), tensor.element_size() * tensor.numel(),
    wrap_tag(srcRank, tag), complete_tag(), tensor.device().type());
  return c10::make_intrusive<ProcessGroupUCC::WorkUCP>(worker, request);
}

// See note: [Receive from an endpoint]
c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  static_assert(std::is_same<decltype(tag), tag_type>::value,
    "If you updated the type of tag, please check with note [Receive from an endpoint]."
  );
  check_tensor(tensors);
  auto& tensor = tensors[0];
  auto request = worker->recv_with_tag_and_mask(
    tensor.data_ptr(), tensor.element_size() * tensor.numel(),
    wrap_tag(0, tag), any_source_mask(), tensor.device().type());
  return c10::make_intrusive<ProcessGroupUCC::WorkUCP>(worker, request);
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

} // namespace c10d

C10_EXPORT c10::intrusive_ptr<c10d::ProcessGroup> createProcessGroupUCC(
  const c10::intrusive_ptr<c10d::Store>& store,
  int rank,
  int size) {
  return c10::make_intrusive<c10d::ProcessGroupUCC>(store, rank, size);
}

static_assert(std::is_same<c10d::CreateProcessGroupUCCType, decltype(&createProcessGroupUCC)>::value,
  "CreateProcessGroupUCCType mismatch with createProcessGroupUCC, "
  "if you changed one of them, you should change the other as well");
