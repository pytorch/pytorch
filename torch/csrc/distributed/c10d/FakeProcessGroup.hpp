#pragma once

#include <ATen/core/LegacyTypeDispatch.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
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

  // Forward collectives to a real backend when set.
  void setDelegate(c10::intrusive_ptr<Backend> delegate) {
    TORCH_CHECK(
        delegate != nullptr,
        "FakeProcessGroup::setDelegate: delegate must not be null");
    TORCH_CHECK(
        delegate->getRank() == rank_ && delegate->getSize() == size_,
        "FakeProcessGroup::setDelegate: rank/size mismatch: delegate (",
        delegate->getRank(),
        "/",
        delegate->getSize(),
        ") vs fake (",
        rank_,
        "/",
        size_,
        ")");
    delegate_ = std::move(delegate);
  }

  c10::intrusive_ptr<Backend> getDelegate() const {
    return delegate_;
  }


  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override {
    if (delegate_) {
      return delegate_->broadcast(tensors, opts);
    }
    checkCollectiveError();
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override {
    if (delegate_) {
      AllreduceOptions d_opts = opts;
      auto factor = extractPremulFactor(d_opts.reduceOp);
      auto work = delegate_->allreduce(tensors, d_opts);
      if (factor.has_value()) {
        postMultiply(tensors, *factor);
      }
      return work;
    }
    checkCollectiveError();
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> allreduce_sparse(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override {
    if (delegate_) {
      AllreduceOptions d_opts = opts;
      auto factor = extractPremulFactor(d_opts.reduceOp);
      auto work = delegate_->allreduce_sparse(tensors, d_opts);
      if (factor.has_value()) {
        postMultiply(tensors, *factor);
      }
      return work;
    }
    checkCollectiveError();
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override {
    if (delegate_) {
      AllreduceCoalescedOptions d_opts = opts;
      auto factor = extractPremulFactor(d_opts.reduceOp);
      auto work = delegate_->allreduce_coalesced(tensors, d_opts);
      if (factor.has_value()) {
        postMultiply(tensors, *factor);
      }
      return work;
    }
    checkCollectiveError();
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override {
    if (delegate_) {
      ReduceOptions d_opts = opts;
      auto factor = extractPremulFactor(d_opts.reduceOp);
      auto work = delegate_->reduce(tensors, d_opts);
      if (factor.has_value()) {
        postMultiply(tensors, *factor);
      }
      return work;
    }
    checkCollectiveError();
    return c10::make_intrusive<FakeWork>();
  }

  // NOTE [FakeProcessGroup collective semantics]
  // Collectives use deterministic single-process approximations. When output
  // can be derived from local inputs, fake collectives copy those values into
  // local outputs so tests do not consume uninitialized memory. For scatter on
  // non-root ranks, the root's input list is unavailable in this single-process
  // simulation, so the output tensor is left unchanged.
  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    if (delegate_) {
      return delegate_->allgather(outputTensors, inputTensors, opts);
    }
    checkCollectiveError();
    // See note in _allgather_base below.
    at::AutoDispatchBelowAutograd guard;
    for (auto& tensor : outputTensors[0]) {
      tensor.copy_(inputTensors[0]);
    }
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    if (delegate_) {
      return delegate_->_allgather_base(outputBuffer, inputBuffer, opts);
    }
    checkCollectiveError();
    // Real collective backends (e.g. NCCL) write into the output from C++
    // kernels that autograd never sees. We emulate that here: chunk() produces
    // multi-output views, and without this guard autograd would reject the
    // subsequent copy_() when the input requires grad.
    at::AutoDispatchBelowAutograd guard;
    auto chunks = outputBuffer.chunk(size_);
    for (auto& tensor : chunks) {
      tensor.copy_(inputBuffer);
    }
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    if (delegate_) {
      return delegate_->allgather_coalesced(
          outputTensorLists, inputTensors, opts);
    }
    checkCollectiveError();
    auto invalidArgument = [](const std::string& msg) {
      TORCH_CHECK(false, "FakeProcessGroup::allgather_coalesced: ", msg);
    };
    assertNonEmptyInputTensorList(invalidArgument, inputTensors.size());
    assertAllgatherCoalescedOutputTensorLists(
        invalidArgument, outputTensorLists, inputTensors.size(), size_);
    // See note in _allgather_base above.
    at::AutoDispatchBelowAutograd guard;
    for (auto& outputTensorList : outputTensorLists) {
      for (size_t i = 0; i < inputTensors.size(); ++i) {
        outputTensorList[i].copy_(inputTensors[i]);
      }
    }
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    if (delegate_) {
      return delegate_->allgather_into_tensor_coalesced(outputs, inputs, opts);
    }
    checkCollectiveError();
    // See note in _allgather_base above.
    at::AutoDispatchBelowAutograd guard;
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
      const GatherOptions& opts = GatherOptions()) override {
    if (delegate_) {
      return delegate_->gather(outputTensors, inputTensors, opts);
    }
    checkCollectiveError();
    auto invalidArgument = [](const std::string& msg) {
      TORCH_CHECK(false, "FakeProcessGroup::gather: ", msg);
    };
    assertRootRank(invalidArgument, opts.rootRank, size_);
    assertSingleElementInput(invalidArgument, inputTensors);

    if (rank_ == opts.rootRank) {
      assertGatherOutputTensorList(invalidArgument, outputTensors, size_);
      // See note in _allgather_base above.
      at::AutoDispatchBelowAutograd guard;
      for (auto& tensor : outputTensors[0]) {
        tensor.copy_(inputTensors[0]);
      }
    } else {
      assertEmptyOutputTensorList(invalidArgument, outputTensors);
    }
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override {
    if (delegate_) {
      return delegate_->scatter(outputTensors, inputTensors, opts);
    }
    checkCollectiveError();
    auto invalidArgument = [](const std::string& msg) {
      TORCH_CHECK(false, "FakeProcessGroup::scatter: ", msg);
    };
    assertRootRank(invalidArgument, opts.rootRank, size_);
    assertSingleElementOutput(invalidArgument, outputTensors);

    if (rank_ == opts.rootRank) {
      assertScatterInputTensorList(invalidArgument, inputTensors, size_);
      // See note in _allgather_base above.
      at::AutoDispatchBelowAutograd guard;
      outputTensors[0].copy_(inputTensors[0][rank_]);
    } else {
      assertEmptyInputTensorList(invalidArgument, inputTensors);
    }
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    if (delegate_) {
      ReduceScatterOptions d_opts = opts;
      auto factor = extractPremulFactor(d_opts.reduceOp);
      auto work =
          delegate_->reduce_scatter(outputTensors, inputTensors, d_opts);
      if (factor.has_value()) {
        postMultiply(outputTensors, *factor);
      }
      return work;
    }
    checkCollectiveError();
    auto invalidArgument = [](const std::string& msg) {
      TORCH_CHECK(false, "FakeProcessGroup::reduce_scatter: ", msg);
    };
    assertInputOutputTensorListsSameSize(
        invalidArgument, outputTensors.size(), inputTensors.size());
    // See note in _allgather_base above.
    at::AutoDispatchBelowAutograd guard;
    for (size_t i = 0; i < outputTensors.size(); ++i) {
      assertInputTensorListSizeEqualsWorldSize(
          invalidArgument, inputTensors[i].size(), size_);
      outputTensors[i].copy_(inputTensors[i][rank_]);
    }
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    if (delegate_) {
      ReduceScatterOptions d_opts = opts;
      auto factor = extractPremulFactor(d_opts.reduceOp);
      auto work =
          delegate_->_reduce_scatter_base(outputBuffer, inputBuffer, d_opts);
      if (factor.has_value()) {
        postMultiply(outputBuffer, *factor);
      }
      return work;
    }
    checkCollectiveError();
    TORCH_CHECK(
        inputBuffer.numel() == outputBuffer.numel() * size_,
        "input tensor must be the same size as output size times world size");
    // See note in _allgather_base above.
    at::AutoDispatchBelowAutograd guard;
    auto chunks = inputBuffer.chunk(size_);
    outputBuffer.copy_(chunks[rank_]);
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    if (delegate_) {
      ReduceScatterOptions d_opts = opts;
      auto factor = extractPremulFactor(d_opts.reduceOp);
      auto work =
          delegate_->reduce_scatter_tensor_coalesced(outputs, inputs, d_opts);
      if (factor.has_value()) {
        postMultiply(outputs, *factor);
      }
      return work;
    }
    checkCollectiveError();
    auto invalidArgument = [](const std::string& msg) {
      TORCH_CHECK(
          false, "FakeProcessGroup::reduce_scatter_tensor_coalesced: ", msg);
    };
    assertInputOutputTensorListsSameSize(
        invalidArgument, outputs.size(), inputs.size());
    // See note in _allgather_base above.
    at::AutoDispatchBelowAutograd guard;
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
      const AllToAllOptions& opts = AllToAllOptions()) override {
    if (delegate_) {
      return delegate_->alltoall_base(
          outputBuffer, inputBuffer, outputSplitSizes, inputSplitSizes, opts);
    }
    checkCollectiveError();
    c10d::checkSplitSizes(inputSplitSizes, inputBuffer, size_);
    c10d::checkSplitSizes(outputSplitSizes, outputBuffer, size_);
    // See note in _allgather_base above.
    at::AutoDispatchBelowAutograd guard;
    if (outputSplitSizes.empty() && inputSplitSizes.empty()) {
      outputBuffer.copy_(inputBuffer);
    } else {
      // Approximation: rank j's inputSplitSizes are unavailable here, so
      // each output slot is filled by repeating inputBuffer[0:slot]. The
      // values are deterministic but arbitrary; do not assert on them.
      int64_t out_offset = 0;
      auto in_size = inputBuffer.size(0);
      for (int j = 0; j < size_; ++j) {
        int64_t remaining = outputSplitSizes[j];
        if (remaining > 0) {
          TORCH_CHECK(
              in_size > 0,
              "alltoall_base: inputBuffer is empty but outputSplitSizes[",
              j,
              "] > 0");
        }
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
      const AllToAllOptions& opts = AllToAllOptions()) override {
    if (delegate_) {
      return delegate_->alltoall(outputTensors, inputTensors, opts);
    }
    checkCollectiveError();
    auto invalidArgument = [](const std::string& msg) {
      TORCH_CHECK(false, "FakeProcessGroup::alltoall: ", msg);
    };
    assertAllToAllTensorListSizes(
        invalidArgument, outputTensors.size(), inputTensors.size(), size_);
    // See note in _allgather_base above.
    at::AutoDispatchBelowAutograd guard;
    for (size_t i = 0; i < outputTensors.size(); ++i) {
      outputTensors[i].copy_(inputTensors[i]);
    }
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override {
    if (delegate_) {
      return delegate_->send(tensors, dstRank, tag);
    }
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override {
    if (delegate_) {
      return delegate_->recv(tensors, srcRank, tag);
    }
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override {
    if (delegate_) {
      return delegate_->recvAnysource(tensors, tag);
    }
    return c10::make_intrusive<FakeWork>();
  }

  void startCoalescing() override {
    if (delegate_) {
      delegate_->startCoalescing();
      return;
    }
    // No-op
  }

  c10::intrusive_ptr<Work> endCoalescing(OpType optype) {
    if (delegate_) {
      return delegate_->endCoalescing();
    }
    checkCollectiveError();
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> endCoalescing() override {
    if (delegate_) {
      return delegate_->endCoalescing();
    }
    checkCollectiveError();
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override {
    if (delegate_) {
      return delegate_->barrier(opts);
    }
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
  c10::intrusive_ptr<Backend> delegate_;

  void checkCollectiveError() {
    TORCH_CHECK(
        !options_ || !options_->error_on_collective,
        "FakeProcessGroup collective operation error (error_on_collective=true)");
  }

  // PREMUL_SUM decomposition: extract the scalar factor, change op to SUM.
  // Returns nullopt if the op is not PREMUL_SUM.
  static std::optional<double> extractPremulFactor(ReduceOp& reduceOp) {
    if (reduceOp.op_ != ReduceOp::PREMUL_SUM) {
      return std::nullopt;
    }
    TORCH_CHECK(reduceOp.supplement_, "PREMUL_SUM ReduceOp has no supplement");
    auto* supplement =
        dynamic_cast<PreMulSumSupplement*>(reduceOp.supplement_.get());
    TORCH_CHECK(supplement, "PREMUL_SUM supplement has unexpected type");
    double factor = supplement->tensor_factor.defined()
        ? supplement->tensor_factor.item<double>()
        : supplement->double_factor;
    reduceOp = ReduceOp(ReduceOp::SUM);
    return factor;
  }

  static void postMultiply(std::vector<at::Tensor>& tensors, double factor) {
    at::AutoDispatchBelowAutograd guard;
    for (auto& t : tensors) {
      t.mul_(factor);
    }
  }

  static void postMultiply(at::Tensor& tensor, double factor) {
    at::AutoDispatchBelowAutograd guard;
    tensor.mul_(factor);
  }
};

} // namespace c10d
