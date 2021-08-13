#include <c10d/ProcessGroupNCCLWithUCC.hpp>

#include <cstdlib>
#include <string>

#ifdef USE_C10D_NCCL
#include <c10d/ProcessGroupNCCL.hpp>
#endif

#ifdef USE_C10D_UCC
#include <c10d/ucc/ProcessGroupUCC.hpp>
#endif

#if defined(USE_C10D_NCCL) || defined(USE_C10D_UCC)

constexpr const char *TORCH_NCCL_ENABLED = "TORCH_NCCL_ENABLED";
constexpr const char *TORCH_UCC_ENABLED = "TORCH_UCC_ENABLED";

inline bool string_is_true(char *s) {
  return s == nullptr || (std::string(s) != "0" && std::string(s) != "false");
}

inline bool nccl_enabled() {
  static bool result = string_is_true(std::getenv(TORCH_NCCL_ENABLED));
  return result;
}

inline bool ucc_enabled() {
  static bool result = string_is_true(std::getenv(TORCH_UCC_ENABLED));
  return result;
}

namespace c10d {

ProcessGroupNCCLWithUCC::ProcessGroupNCCLWithUCC(
  const c10::intrusive_ptr<Store>& store,
  int rank,
  int size,
  c10::intrusive_ptr<Options> options):
  ProcessGroup(rank, size), options_(options), libucc("libtorch_ucc.so", nullptr, true)
{
#ifdef USE_C10D_NCCL
  pg_nccl = c10::make_intrusive<ProcessGroupNCCL>(store, rank, size, options);
#endif
#ifdef USE_C10D_UCC
  if (libucc.available()) {
    CreateProcessGroupUCCType createProcessGroupUCC = 
      reinterpret_cast<CreateProcessGroupUCCType>(libucc.sym("_Z21createProcessGroupUCCRKN3c1013intrusive_ptrIN4c10d5StoreENS_6detail34intrusive_target_default_null_typeIS2_EEEEii"));
    pg_ucc = createProcessGroupUCC(store, rank, size);
  }
#endif
}

ProcessGroupNCCLWithUCC::~ProcessGroupNCCLWithUCC() {}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCLWithUCC::broadcast(
  std::vector<at::Tensor>& tensors,
  const BroadcastOptions& opts) {
  if (nccl_enabled() && pg_nccl.get() != nullptr && tensors[0].device().type() == c10::kCUDA) {
    return pg_nccl->broadcast(tensors, opts);
  }
  if (ucc_enabled() && pg_ucc.get() != nullptr) {
    return pg_ucc->broadcast(tensors, opts);
  }
  TORCH_CHECK(false, "Can not find a backend to execute broadcast");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCLWithUCC::allreduce(
  std::vector<at::Tensor>& tensors,
  const AllreduceOptions& opts) {
  if (nccl_enabled() && pg_nccl.get() != nullptr && tensors[0].device().type() == c10::kCUDA) {
    return pg_nccl->allreduce(tensors, opts);
  }
  if (ucc_enabled() && pg_ucc.get() != nullptr) {
    return pg_ucc->allreduce(tensors, opts);
  }
  TORCH_CHECK(false, "Can not find a backend to execute allreduce");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCLWithUCC::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  if (nccl_enabled() && pg_nccl.get() != nullptr && tensors[0].device().type() == c10::kCUDA) {
    return pg_nccl->allreduce_coalesced(tensors, opts);
  }
  if (ucc_enabled() && pg_ucc.get() != nullptr) {
    return pg_ucc->allreduce_coalesced(tensors, opts);
  }
  TORCH_CHECK(false, "Can not find a backend to execute allreduce_coalesced");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCLWithUCC::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  if (nccl_enabled() && pg_nccl.get() != nullptr && tensors[0].device().type() == c10::kCUDA) {
    return pg_nccl->reduce(tensors, opts);
  }
  if (ucc_enabled() && pg_ucc.get() != nullptr) {
    return pg_ucc->reduce(tensors, opts);
  }
  TORCH_CHECK(false, "Can not find a backend to execute reduce");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCLWithUCC::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  if (nccl_enabled() && pg_nccl.get() != nullptr && inputTensors[0].device().type() == c10::kCUDA) {
    return pg_nccl->allgather(outputTensors, inputTensors, opts);
  }
  if (ucc_enabled() && pg_ucc.get() != nullptr) {
    return pg_ucc->allgather(outputTensors, inputTensors, opts);
  }
  TORCH_CHECK(false, "Can not find a backend to execute allgather");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCLWithUCC::_allgather_base(
    at::Tensor& outputbuffer,
    at::Tensor& inputbuffer,
    const AllgatherOptions& opts) {
  if (nccl_enabled() && pg_nccl.get() != nullptr && inputbuffer.device().type() == c10::kCUDA) {
    return pg_nccl->_allgather_base(outputbuffer, inputbuffer, opts);
  }
  if (ucc_enabled() && pg_ucc.get() != nullptr) {
    return pg_ucc->_allgather_base(outputbuffer, inputbuffer, opts);
  }
  TORCH_CHECK(false, "Can not find a backend to execute _allgather_base");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCLWithUCC::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& outputTensorLists,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  if (nccl_enabled() && pg_nccl.get() != nullptr && inputTensors[0].device().type() == c10::kCUDA) {
    return pg_nccl->allgather_coalesced(outputTensorLists, inputTensors, opts);
  }
  if (ucc_enabled() && pg_ucc.get() != nullptr) {
    return pg_ucc->allgather_coalesced(outputTensorLists, inputTensors, opts);
  }
  TORCH_CHECK(false, "Can not find a backend to execute allgather_coalesced");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCLWithUCC::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  if (nccl_enabled() && pg_nccl.get() != nullptr && outputTensors[0].device().type() == c10::kCUDA) {
    return pg_nccl->reduce_scatter(outputTensors, inputTensors, opts);
  }
  if (ucc_enabled() && pg_ucc.get() != nullptr) {
    return pg_ucc->reduce_scatter(outputTensors, inputTensors, opts);
  }
  TORCH_CHECK(false, "Can not find a backend to execute reduce_scatter");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCLWithUCC::_reduce_scatter_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const ReduceScatterOptions& opts) {
  if (nccl_enabled() && pg_nccl.get() != nullptr && inputTensor.device().type() == c10::kCUDA) {
    return pg_nccl->_reduce_scatter_base(outputTensor, inputTensor, opts);
  }
  if (ucc_enabled() && pg_ucc.get() != nullptr) {
    return pg_ucc->_reduce_scatter_base(outputTensor, inputTensor, opts);
  }
  TORCH_CHECK(false, "Can not find a backend to execute _reduce_scatter_base");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCLWithUCC::barrier(
    const BarrierOptions& opts) {
  if (nccl_enabled() && pg_nccl.get() != nullptr) {
    return pg_nccl->barrier(opts);
  }
  if (ucc_enabled() && pg_ucc.get() != nullptr) {
    return pg_ucc->barrier(opts);
  }
  // TODO: what if both nccl and ucc are enabled?
  TORCH_CHECK(false, "Can not find a backend to execute barrier");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCLWithUCC::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& opts) {
  if (nccl_enabled() && pg_nccl.get() != nullptr && inputTensor.device().type() == c10::kCUDA) {
    return pg_nccl->alltoall_base(outputTensor, inputTensor, outputSplitSizes, inputSplitSizes, opts);
  }
  if (ucc_enabled() && pg_ucc.get() != nullptr) {
    return pg_ucc->alltoall_base(outputTensor, inputTensor, outputSplitSizes, inputSplitSizes, opts);
  }
  TORCH_CHECK(false, "Can not find a backend to execute alltoall_base");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCLWithUCC::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& opts) {
  if (nccl_enabled() && pg_nccl.get() != nullptr && inputTensors[0].device().type() == c10::kCUDA) {
    return pg_nccl->alltoall(outputTensors, inputTensors, opts);
  }
  if (ucc_enabled() && pg_ucc.get() != nullptr) {
    return pg_ucc->alltoall(outputTensors, inputTensors, opts);
  }
  TORCH_CHECK(false, "Can not find a backend to execute alltoall");
}


c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCLWithUCC::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  if (nccl_enabled() && pg_nccl.get() != nullptr && inputTensors[0].device().type() == c10::kCUDA) {
    return pg_nccl->gather(outputTensors, inputTensors, opts);
  }
  if (ucc_enabled() && pg_ucc.get() != nullptr) {
    return pg_ucc->gather(outputTensors, inputTensors, opts);
  }
  TORCH_CHECK(false, "Can not find a backend to execute gather");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCLWithUCC::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  if (nccl_enabled() && pg_nccl.get() != nullptr && outputTensors[0].device().type() == c10::kCUDA) {
    return pg_nccl->scatter(outputTensors, inputTensors, opts);
  }
  if (ucc_enabled() && pg_ucc.get() != nullptr) {
    return pg_ucc->scatter(outputTensors, inputTensors, opts);
  }
  TORCH_CHECK(false, "Can not find a backend to execute scatter");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCLWithUCC::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  if (nccl_enabled() && pg_nccl.get() != nullptr && tensors[0].device().type() == c10::kCUDA) {
    return pg_nccl->send(tensors, dstRank, tag);
  }
  if (ucc_enabled() && pg_ucc.get() != nullptr) {
    return pg_ucc->send(tensors, dstRank, tag);
  }
  TORCH_CHECK(false, "Can not find a backend to execute send");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCLWithUCC::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  if (ucc_enabled() && pg_nccl.get() != nullptr && tensors[0].device().type() == c10::kCUDA) {
    return pg_nccl->recv(tensors, srcRank, tag);
  }
  if (ucc_enabled() && pg_ucc.get() != nullptr) {
    return pg_ucc->recv(tensors, srcRank, tag);
  }
  TORCH_CHECK(false, "Can not find a backend to execute recv");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCLWithUCC::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  if (nccl_enabled() && pg_nccl.get() != nullptr && tensors[0].device().type() == c10::kCUDA) {
    return pg_nccl->recvAnysource(tensors, tag);
  }
  if (ucc_enabled() && pg_ucc.get() != nullptr) {
    return pg_ucc->recvAnysource(tensors, tag);
  }
  TORCH_CHECK(false, "Can not find a backend to execute recvAnysource");
}

void ProcessGroupNCCLWithUCC::setSequenceNumberForGroup() {
  if (nccl_enabled() && pg_nccl.get() != nullptr) {
    pg_nccl->setSequenceNumberForGroup();
    return;
  }
  TORCH_CHECK(false, "Can not find a backend to execute setSequenceNumberForGroup");
}

uint64_t ProcessGroupNCCLWithUCC::getSequenceNumberForGroup() {
  if (nccl_enabled() && pg_nccl.get() != nullptr) {
    return pg_nccl->getSequenceNumberForGroup();
  }
  TORCH_CHECK(false, "Can not find a backend to execute getSequenceNumberForGroup");
}

void ProcessGroupNCCLWithUCC::groupStart() {
#ifdef USE_C10D_NCCL
  ProcessGroupNCCL::groupStart();
#endif
}

void ProcessGroupNCCLWithUCC::groupEnd() {
#ifdef USE_C10D_NCCL
  ProcessGroupNCCL::groupEnd();
#endif
}

} // namespace c10d

#endif // USE_C10D_NCCL
