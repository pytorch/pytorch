#include <c10d/ProcessGroupUCC.hpp>

#ifdef USE_C10D_UCC

namespace {

static void check_tensor(const std::vector<at::Tensor>& tensors) {
  TORCH_CHECK(tensors.size() == 1, "ProcessGroupUCC takes 1 tensor");
  TORCH_CHECK(tensors[0].is_contiguous(), "ProcessGroupUCC input tensor has to be contiguous");
  TORCH_CHECK(!tensors[0].is_sparse(), "ProcessGroupUCC input tensor has to be dense");
  // TODO: check cuda
  // TODO: check non-overlapping and dense instead of contiguous
}

} // namespace

namespace c10d {

ProcessGroupUCC::ProcessGroupUCC(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size) : ProcessGroup(rank, size), store(store) {
}

void ProcessGroupUCC::lazyInitUCP() {
  if (ucp_endpoints.size() > 0) {
    return;  // already initialized
  }

  UCPContext *ucp_context = UCPContext::get();
  ucs_status_t st;
  ucp_address_t* local_addr;
  size_t local_addr_len;

  st = ucp_worker_get_address(ucp_context->worker, &local_addr, &local_addr_len);
  TORCH_UCX_CHECK(st, "Failed to get worker address.");
  std::vector<uint8_t> val = std::vector<uint8_t>(
      reinterpret_cast<uint8_t*>(local_addr),
      reinterpret_cast<uint8_t*>(local_addr) + local_addr_len);
  store->set("ucp_address:" + std::to_string(rank_), val);
  ucp_worker_release_address(ucp_context->worker, local_addr);

  for (int i = 0; i < size_; i++) {
    std::vector<uint8_t> peer_addr = store->get("ucp_address:" + std::to_string(i));
    ucp_address_t *address = reinterpret_cast<ucp_address_t*>(peer_addr.data());
    ucp_endpoints.emplace_back(std::make_shared<UCPEndpoint>(address));
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

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  lazyInitUCP();

  ucp_request_param_t params;
  params.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE | UCP_OP_ATTR_FIELD_MEMORY_TYPE;
  params.datatype = ucp_dt_make_contig(tensor.numel() * tensor.element_size());  // TODO: support all contiguity types
  params.memory_type = getUCSMemoryType(tensor.device().type());
  ucs_status_ptr_t request = ucp_tag_send_nbx(
    ucp_endpoints[dstRank]->endpoint, tensor.data_ptr(), 1, tag, &params);
  auto work = c10::make_intrusive<ProcessGroupUCC::WorkUCP>(request);
  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  TORCH_CHECK(false, "ProcessGroupUCC does not support recv");
  check_tensor(tensors);
  auto& tensor = tensors[0];
  lazyInitUCP();

  ucp_request_param_t params;
  params.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE | UCP_OP_ATTR_FIELD_MEMORY_TYPE;
  params.datatype = ucp_dt_make_contig(tensor.numel() * tensor.element_size());  // TODO: support all contiguity types
  params.memory_type = getUCSMemoryType(tensor.device().type());
  ucs_status_ptr_t request = ucp_tag_recv_nbx(
    UCPContext::get()->worker, tensor.data_ptr(), 1, tag, 0, &params);
  auto work = c10::make_intrusive<ProcessGroupUCC::WorkUCP>(request);
  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::recvAnysource(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */) {
  TORCH_CHECK(false, "ProcessGroupUCC does not support recvAnysource");
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

#endif
