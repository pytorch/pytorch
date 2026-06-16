#pragma once

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

namespace c10d {

class TORCH_API ProcessGroupNCCL2 : public Backend {
 public:
  static constexpr auto kBackendName = "nccl2";

  ProcessGroupNCCL2(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<ProcessGroupNCCL::Options> options =
          ProcessGroupNCCL::Options::create())
      : Backend(rank, size),
        nccl_(c10::make_intrusive<ProcessGroupNCCL>(
            store,
            rank,
            size,
            std::move(options))) {}

  explicit ProcessGroupNCCL2(c10::intrusive_ptr<ProcessGroupNCCL> nccl)
      : Backend(nccl->getRank(), nccl->getSize()), nccl_(std::move(nccl)) {}

  const std::string getBackendName() const override {
    return kBackendName;
  }

  c10::intrusive_ptr<Options> getBackendOptions() override {
    return nccl_->getBackendOptions();
  }

  c10::intrusive_ptr<ProcessGroupNCCL::Options> getOptions() {
    return nccl_->getOptions();
  }

  bool supportsSplitting() const override {
    return nccl_->supportsSplitting();
  }

  bool supportsCoalescing() const override {
    return nccl_->supportsCoalescing();
  }

  bool supportsTimeEstimation() const override {
    return nccl_->supportsTimeEstimation();
  }

  void setTimeout(std::chrono::milliseconds timeout) override {
    nccl_->setTimeout(timeout);
  }

  void startCoalescing() override {
    nccl_->startCoalescing();
  }

  c10::intrusive_ptr<Work> endCoalescing() override {
    return nccl_->endCoalescing();
  }

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override {
    return nccl_->broadcast(tensors, opts);
  }

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override {
    return nccl_->allreduce(tensors, opts);
  }

  c10::intrusive_ptr<Work> allreduce_sparse(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override {
    return nccl_->allreduce_sparse(tensors, opts);
  }

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override {
    return nccl_->allreduce_coalesced(tensors, opts);
  }

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override {
    return nccl_->reduce(tensors, opts);
  }

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    return nccl_->allgather(outputTensors, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> all_gather_single(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    return nccl_->all_gather_single(outputBuffer, inputBuffer, opts);
  }

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    return nccl_->_allgather_base(outputBuffer, inputBuffer, opts);
  }

  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    return nccl_->allgather_coalesced(outputTensorLists, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> all_gather_single_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    return nccl_->all_gather_single_coalesced(outputs, inputs, opts);
  }

  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    return nccl_->allgather_into_tensor_coalesced(outputs, inputs, opts);
  }

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override {
    return nccl_->gather(outputTensors, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override {
    return nccl_->scatter(outputTensors, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    return nccl_->reduce_scatter(outputTensors, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> reduce_scatter_single(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    return nccl_->reduce_scatter_single(outputBuffer, inputBuffer, opts);
  }

  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    return nccl_->_reduce_scatter_base(outputBuffer, inputBuffer, opts);
  }

  c10::intrusive_ptr<Work> reduce_scatter_single_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    return nccl_->reduce_scatter_single_coalesced(outputs, inputs, opts);
  }

  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    return nccl_->reduce_scatter_tensor_coalesced(outputs, inputs, opts);
  }

  c10::intrusive_ptr<Work> all_to_all_single(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override {
    return nccl_->all_to_all_single(
        outputBuffer, inputBuffer, outputSplitSizes, inputSplitSizes, opts);
  }

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override {
    return nccl_->alltoall_base(
        outputBuffer, inputBuffer, outputSplitSizes, inputSplitSizes, opts);
  }

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override {
    return nccl_->alltoall(outputTensors, inputTensors, opts);
  }

  void setSequenceNumberForGroup() override {
    nccl_->setSequenceNumberForGroup();
  }

  uint64_t getSequenceNumberForGroup() override {
    return nccl_->getSequenceNumberForGroup();
  }

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override {
    return nccl_->send(tensors, dstRank, tag);
  }

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override {
    return nccl_->recv(tensors, srcRank, tag);
  }

  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override {
    return nccl_->recvAnysource(tensors, tag);
  }

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override {
    return nccl_->barrier(opts);
  }

  void registerOnCompletionHook(
      std::function<void(std::shared_ptr<WorkInfo>)>&& hook) override {
    nccl_->registerOnCompletionHook(std::move(hook));
  }

  void waitForPendingWorks() override {
    nccl_->waitForPendingWorks();
  }

  void enableCollectivesTiming() override {
    nccl_->enableCollectivesTiming();
  }

  uint64_t getCommSplitCounter() const {
    return nccl_->getCommSplitCounter();
  }

  c10::intrusive_ptr<Backend> split(
      const c10::intrusive_ptr<Store>& store,
      const std::vector<int>& ranks,
      const c10::intrusive_ptr<Options>& opts) override {
    auto backend = nccl_->split(store, ranks, opts);
    auto nccl = c10::dynamic_intrusive_pointer_cast<ProcessGroupNCCL>(backend);
    TORCH_CHECK(
        nccl != nullptr, "ProcessGroupNCCL2 split expected NCCL backend");
    return c10::make_intrusive<ProcessGroupNCCL2>(std::move(nccl));
  }

  c10::intrusive_ptr<Backend> merge(
      const c10::intrusive_ptr<Store>& store,
      const c10::intrusive_ptr<Options>& opts,
      const int& rank,
      const int& size) override {
    auto backend = nccl_->merge(store, opts, rank, size);
    auto nccl = c10::dynamic_intrusive_pointer_cast<ProcessGroupNCCL>(backend);
    TORCH_CHECK(
        nccl != nullptr, "ProcessGroupNCCL2 merge expected NCCL backend");
    return c10::make_intrusive<ProcessGroupNCCL2>(std::move(nccl));
  }

  void setGroupUid(const std::string& pg_uid) override {
    Backend::setGroupUid(pg_uid);
    nccl_->setGroupUid(pg_uid);
  }

  void eagerConnectSingleDevice(at::Device device) override {
    setBoundDeviceId(device);
    nccl_->setBoundDeviceId(device);
    nccl_->eagerConnectSingleDevice(device);
  }

  ErrorType getError() override {
    return nccl_->getError();
  }

  std::shared_ptr<c10::Allocator> getMemAllocator() override {
    return nccl_->getMemAllocator();
  }

  at::Tensor allocateTensor(long size, at::TensorOptions options = {})
      override {
    return nccl_->allocateTensor(size, options);
  }

  bool supportsTensorAlloc(c10::DeviceIndex deviceIdx) override {
    return nccl_->supportsTensorAlloc(deviceIdx);
  }

  void abort() override {
    nccl_->abort();
  }

  void shutdown() override {
    nccl_->shutdown();
  }

  void suspend() override {
    nccl_->suspend();
  }

  void resume() override {
    nccl_->resume();
  }

  std::unordered_map<std::string, uint64_t> getMemoryStats() override {
    return nccl_->getMemoryStats();
  }

 private:
  c10::intrusive_ptr<ProcessGroupNCCL> nccl_;
};

} // namespace c10d
