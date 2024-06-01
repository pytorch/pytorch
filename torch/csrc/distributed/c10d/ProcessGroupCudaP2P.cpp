#ifdef USE_C10D_NCCL
#include <torch/csrc/distributed/c10d/ProcessGroupCudaP2P.hpp>

#include <torch/csrc/distributed/c10d/PrefixStore.hpp>

namespace c10d {

using namespace c10d::intra_node_comm;

ProcessGroupCudaP2P::ProcessGroupCudaP2P(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : Backend(rank, size), stream_(c10::cuda::getStreamFromPool()) {
  nccl_backend_ = c10::make_intrusive<ProcessGroupNCCL>(
      c10::make_intrusive<PrefixStore>("nccl", store),
      rank,
      size,
      options->nccl_options);
  nccl_backend_->setSequenceNumberForGroup();

  p2p_backend_ = c10::make_intrusive<IntraNodeComm>(
      c10::make_intrusive<PrefixStore>("p2p", store),
      rank,
      size,
      options->buffer_size);
  if (!p2p_backend_->rendezvous()) {
    p2p_backend_ = nullptr;
  }
}

bool ProcessGroupCudaP2P::is_p2p_available() {
  return p2p_backend_ != nullptr &&
      p2p_backend_->getTopology() == Topology::FULLY_CONNECTED;
}

size_t ProcessGroupCudaP2P::get_buffer_size() {
  if (p2p_backend_ == nullptr) {
    return 0;
  }
  return p2p_backend_->getBufferSize();
}

c10::Stream ProcessGroupCudaP2P::stream() {
  return stream_;
}

c10::intrusive_ptr<Work> ProcessGroupCudaP2P::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  return nccl_backend_->broadcast(tensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupCudaP2P::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  return nccl_backend_->allreduce(tensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupCudaP2P::allreduce_sparse(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  return nccl_backend_->allreduce_sparse(tensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupCudaP2P::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  return nccl_backend_->allreduce_coalesced(tensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupCudaP2P::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  return nccl_backend_->reduce(tensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupCudaP2P::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  return nccl_backend_->allgather(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupCudaP2P::_allgather_base(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    const AllgatherOptions& opts) {
  return nccl_backend_->_allgather_base(outputBuffer, inputBuffer, opts);
}

c10::intrusive_ptr<Work> ProcessGroupCudaP2P::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& outputTensorLists,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  return nccl_backend_->allgather_coalesced(
      outputTensorLists, inputTensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupCudaP2P::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& opts) {
  return nccl_backend_->allgather_into_tensor_coalesced(outputs, inputs, opts);
}

c10::intrusive_ptr<Work> ProcessGroupCudaP2P::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  return nccl_backend_->gather(outputTensors, inputTensors);
}

c10::intrusive_ptr<Work> ProcessGroupCudaP2P::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  return nccl_backend_->scatter(outputTensors, inputTensors);
}

c10::intrusive_ptr<Work> ProcessGroupCudaP2P::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  return nccl_backend_->reduce_scatter(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupCudaP2P::_reduce_scatter_base(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    const ReduceScatterOptions& opts) {
  return nccl_backend_->_reduce_scatter_base(outputBuffer, inputBuffer, opts);
}

c10::intrusive_ptr<Work> ProcessGroupCudaP2P::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const ReduceScatterOptions& opts) {
  return nccl_backend_->reduce_scatter_tensor_coalesced(outputs, inputs, opts);
}

c10::intrusive_ptr<Work> ProcessGroupCudaP2P::alltoall_base(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& opts) {
  return nccl_backend_->alltoall_base(
      outputBuffer, inputBuffer, outputSplitSizes, inputSplitSizes);
}

c10::intrusive_ptr<Work> ProcessGroupCudaP2P::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& opts) {
  return nccl_backend_->alltoall(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupCudaP2P::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  return nccl_backend_->send(tensors, dstRank, tag);
}

c10::intrusive_ptr<Work> ProcessGroupCudaP2P::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  return nccl_backend_->recv(tensors, srcRank, tag);
}

c10::intrusive_ptr<Work> ProcessGroupCudaP2P::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  return nccl_backend_->recvAnysource(tensors, tag);
}

c10::intrusive_ptr<Work> ProcessGroupCudaP2P::barrier(
    const BarrierOptions& opts) {
  return nccl_backend_->barrier(opts);
}

c10::intrusive_ptr<Work> ProcessGroupCudaP2P::intra_node_barrier(
    c10::optional<std::vector<int64_t>> ranks) {
  TORCH_CHECK(p2p_backend_ != nullptr);
  p2p_backend_->barrier(ranks);
  return c10::make_intrusive<IntraNodeCommWork>();
}

at::Tensor ProcessGroupCudaP2P::get_p2p_buffer(
    size_t rank,
    const std::vector<int64_t>& sizes,
    c10::ScalarType dtype,
    int64_t storage_offset) {
  TORCH_CHECK(p2p_backend_ != nullptr);
  return p2p_backend_->getBuffer(rank, sizes, dtype, storage_offset);
}

void ProcessGroupCudaP2P::shutdown(c10::optional<std::string> reason) {
  nccl_backend_->shutdown(reason);
}

} // namespace c10d
#endif // USE_C10D_NCCL
