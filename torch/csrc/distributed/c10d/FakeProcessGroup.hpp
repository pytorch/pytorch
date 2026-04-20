#pragma once

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
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

  // NOTE [FakeProcessGroup collective semantics]
  // All collectives copy input to output following single-rank semantics
  // (rank 0 communicating with itself). This avoids returning uninitialized
  // memory and enables single-process validation of distributed code paths.
  //
  // Limitation: scatter on non-root rank leaves output uninitialized because
  // the root's data is unavailable in single-process simulation. This only
  // triggers when explicitly calling scatter with src != rank.
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
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) override {
    checkCollectiveError();
    TORCH_CHECK(
        outputTensorLists.size() == inputTensors.size(),
        "allgather_coalesced: output tensor lists (",
        outputTensorLists.size(),
        ") must have the same length as input tensor list (",
        inputTensors.size(),
        ")");
    for (size_t i = 0; i < inputTensors.size(); ++i) {
      for (auto& tensor : outputTensorLists[i]) {
        tensor.copy_(inputTensors[i]);
      }
    }
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
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& /* opts */ = GatherOptions()) override {
    checkCollectiveError();
    if (!outputTensors.empty()) {
      for (auto& tensor : outputTensors[0]) {
        tensor.copy_(inputTensors[0]);
      }
    }
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& /* opts */ = ScatterOptions()) override {
    checkCollectiveError();
    if (!inputTensors.empty()) {
      TORCH_CHECK(
          static_cast<int>(inputTensors[0].size()) == size_,
          "Incorrect input list size ",
          inputTensors[0].size(),
          ". Input list size should be ",
          size_,
          ", same as size of the process group.");
      outputTensors[0].copy_(inputTensors[0][rank_]);
    }
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& /* opts */ =
          ReduceScatterOptions()) override {
    checkCollectiveError();
    for (size_t i = 0; i < outputTensors.size(); ++i) {
      TORCH_CHECK(
          static_cast<int>(inputTensors[i].size()) == size_,
          "invalid input tensor list size, must be world size");
      outputTensors[i].copy_(inputTensors[i][rank_]);
    }
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const ReduceScatterOptions& /* opts */ =
          ReduceScatterOptions()) override {
    checkCollectiveError();
    TORCH_CHECK(
        inputBuffer.numel() == outputBuffer.numel() * size_,
        "input tensor must be the same size as output size times world size");
    auto chunks = inputBuffer.chunk(size_);
    outputBuffer.copy_(chunks[rank_]);
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ReduceScatterOptions& /* opts */ =
          ReduceScatterOptions()) override {
    checkCollectiveError();
    for (size_t i = 0; i < outputs.size(); ++i) {
      TORCH_CHECK(
          inputs[i].numel() == outputs[i].numel() * size_,
          "input tensor must be the same size as output size times world size");
      auto chunks = inputs[i].chunk(size_);
      outputs[i].copy_(chunks[rank_]);
    }
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& /* opts */ = AllToAllOptions()) override {
    checkCollectiveError();
    c10d::checkSplitSizes(inputSplitSizes, inputBuffer, size_);
    c10d::checkSplitSizes(outputSplitSizes, outputBuffer, size_);
    if (outputSplitSizes.empty() && inputSplitSizes.empty()) {
      outputBuffer.copy_(inputBuffer);
    } else {
      // We receive outputSplitSizes[j] elements from rank j. In reality,
      // rank j would send from an offset determined by rank j's own
      // inputSplitSizes, which we don't have. As an approximation, we
      // copy from input[0:min(outputSplitSizes[j], inputSize)] for each
      // output slot, repeating input as needed when the slot is larger
      // than the input buffer.
      int64_t out_offset = 0;
      auto in_size = inputBuffer.size(0);
      for (int j = 0; j < size_; ++j) {
        int64_t remaining = outputSplitSizes[j];
        int64_t dst = out_offset;
        while (remaining > 0) {
          auto chunk = std::min(remaining, in_size);
          outputBuffer.narrow(0, dst, chunk)
              .copy_(inputBuffer.narrow(0, 0, chunk));
          dst += chunk;
          remaining -= chunk;
        }
        out_offset += outputSplitSizes[j];
      }
    }
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& /* opts */ = AllToAllOptions()) override {
    checkCollectiveError();
    for (size_t i = 0; i < outputTensors.size(); ++i) {
      outputTensors[i].copy_(inputTensors[i]);
    }
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
      : Backend(rank, size), options_(std::move(options)) {
    TORCH_CHECK(
        rank >= 0 && rank < size,
        "Cannot init process group where rank (",
        rank,
        ") >= world_size (",
        size,
        ")");
  }
  c10::intrusive_ptr<Options> options_;

 private:
  void checkCollectiveError() {
    TORCH_CHECK(
        !options_ || !options_->error_on_collective,
        "FakeProcessGroup collective operation error (error_on_collective=true)");
  }
};

} // namespace c10d
