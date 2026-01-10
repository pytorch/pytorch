#pragma once

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/utils.h>

namespace c10d {

class FakeWork : public Work {
 public:
  int seq_id = -1;
  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
    return true;
  }

  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
    auto fut = c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());
    fut->markCompleted();
    return fut;
  }
};

class FakeProcessGroup : public Backend {
 public:
  struct Options : Backend::Options {
    explicit Options() : Backend::Options("fake") {}

    int fake_option = 0;
    bool error_on_collective = false;
  };

  // Static factory method for official APIs
  static c10::intrusive_ptr<FakeProcessGroup> _create_internal(
      int rank,
      int size,
      c10::intrusive_ptr<Options> options = c10::make_intrusive<Options>()) {
    return c10::make_intrusive<FakeProcessGroup>(
        rank, size, std::move(options));
  }

  const std::string getBackendName() const override {
    return "fake";
  }

  c10::intrusive_ptr<Backend::Options> getBackendOptions() override {
    return c10::static_intrusive_pointer_cast<Backend::Options>(options_);
  }

  // FakeProcessGroup supports splitting to enable consistent process group
  // naming between fake and real backends (both use hash-based names).
  bool supportsSplitting() const override {
    return true;
  }

  // Split creates a child FakeProcessGroup for the given ranks.
  // This enables DeviceMesh to use split_group() path with fake backend,
  // ensuring consistent hash-based process group naming.
  c10::intrusive_ptr<Backend> split(
      const c10::intrusive_ptr<Store>& /* store */,
      const std::vector<int>& ranks,
      const c10::intrusive_ptr<Backend::Options>& /* opts */) override {
    // Find the new rank within the split group
    int new_rank = -1;
    for (size_t i = 0; i < ranks.size(); ++i) {
      if (ranks[i] == rank_) {
        new_rank = static_cast<int>(i);
        break;
      }
    }

    // If current rank is not in the split, return nullptr
    // (this matches NCCL behavior)
    if (new_rank < 0) {
      return nullptr;
    }

    // Create a new FakeProcessGroup with default options
    // (we don't use the passed opts because it might not be FakeProcessGroup::Options)
    return c10::make_intrusive<FakeProcessGroup>(
        new_rank, static_cast<int>(ranks.size()), c10::make_intrusive<FakeProcessGroup::Options>());
  }

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& /* tensors */,
      const BroadcastOptions& /* opts */ = BroadcastOptions()) override {
    checkCollectiveError();
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& /* tensors */,
      const AllreduceOptions& /* opts */ = AllreduceOptions()) override {
    checkCollectiveError();
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> allreduce_sparse(
      std::vector<at::Tensor>& /* tensors */,
      const AllreduceOptions& /* opts */ = AllreduceOptions()) override {
    checkCollectiveError();
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& /* tensors */,
      const AllreduceCoalescedOptions& /* opts */ =
          AllreduceCoalescedOptions()) override {
    checkCollectiveError();
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& /* tensors */,
      const ReduceOptions& /* opts */ = ReduceOptions()) override {
    checkCollectiveError();
    return c10::make_intrusive<FakeWork>();
  }

  // NOTE [allgather on FakeProcessGroup]
  // Assume each rank have the same input tensor so we just copy to the results
  // since it's not a real allgather, we simply make this copying logic to let
  // some simple validation works (i.e. calling allgather to see if each rank
  // have the same tensor or not).
  //
  // NOTE: in general it's not good form to try to make FakeProcessGroup work
  // with real data, but the reasoning here is that we want FakeProcessGroup to
  // work with DeviceMesh's init code that have the data validation, which
  // makes it worth the tradeoff.
  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) override {
    checkCollectiveError();
    for (auto& tensor : outputTensors[0]) {
      tensor.copy_(inputTensors[0]);
    }
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) override {
    checkCollectiveError();
    auto chunks = outputBuffer.chunk(size_);
    for (auto& tensor : chunks) {
      tensor.copy_(inputBuffer);
    }
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& /* outputTensorLists */,
      std::vector<at::Tensor>& /* inputTensors */,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) override {
    checkCollectiveError();
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) override {
    checkCollectiveError();
    for (size_t i = 0; i < outputs.size(); ++i) {
      auto chunks = outputs[i].chunk(size_);
      for (auto& chunk : chunks) {
        chunk.copy_(inputs[i]);
      }
    }
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& /* outputTensors */,
      std::vector<at::Tensor>& /* inputTensors */,
      const GatherOptions& /* opts */ = GatherOptions()) override {
    checkCollectiveError();
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& /* outputTensors */,
      std::vector<std::vector<at::Tensor>>& /* inputTensors */,
      const ScatterOptions& /* opts */ = ScatterOptions()) override {
    checkCollectiveError();
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& /* outputTensors */,
      std::vector<std::vector<at::Tensor>>& /* inputTensors */,
      const ReduceScatterOptions& /* opts */ =
          ReduceScatterOptions()) override {
    checkCollectiveError();
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& /* outputBuffer */,
      at::Tensor& /* inputBuffer */,
      const ReduceScatterOptions& /* opts */ =
          ReduceScatterOptions()) override {
    checkCollectiveError();
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& /* outputs */,
      std::vector<at::Tensor>& /* inputs */,
      const ReduceScatterOptions& /* opts */ =
          ReduceScatterOptions()) override {
    checkCollectiveError();
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& /* outputBuffer */,
      at::Tensor& /* inputBuffer */,
      std::vector<int64_t>& /* outputSplitSizes */,
      std::vector<int64_t>& /* inputSplitSizes */,
      const AllToAllOptions& /* opts */ = AllToAllOptions()) override {
    checkCollectiveError();
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& /* outputTensors */,
      std::vector<at::Tensor>& /* inputTensors */,
      const AllToAllOptions& opts = AllToAllOptions()) override {
    checkCollectiveError();
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& /* tensors */,
      int /* dstRank */,
      int /* tag */) override {
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& /* tensors */,
      int /* srcRank */,
      int /* tag */) override {
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& /* tensors */,
      int /* tag */) override {
    return c10::make_intrusive<FakeWork>();
  }

  void startCoalescing() override {
    // No-op
  }

  c10::intrusive_ptr<Work> endCoalescing(OpType /* optype */) {
    checkCollectiveError();
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> endCoalescing() override {
    checkCollectiveError();
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& /* opts */ = BarrierOptions()) override {
    checkCollectiveError();
    return c10::make_intrusive<FakeWork>();
  }

  // Private constructor used by official APIs
  FakeProcessGroup(int rank, int size, c10::intrusive_ptr<Options> options)
      : Backend(rank, size), options_(std::move(options)) {}
  c10::intrusive_ptr<Options> options_;

 private:
  void checkCollectiveError() {
    TORCH_CHECK(
        !options_ || !options_->error_on_collective,
        "FakeProcessGroup collective operation error (error_on_collective=true)");
  }
};

} // namespace c10d
