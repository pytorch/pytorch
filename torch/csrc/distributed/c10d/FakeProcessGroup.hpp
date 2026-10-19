#pragma once

#include <algorithm>
#include <iterator>

#include <ATen/core/LegacyTypeDispatch.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/utils.h>

namespace c10d {

class FakeWork : public Work {
 public:
  int seq_id = -1;

  // `result` is the tensors the collective produced. Real backends resolve
  // their future to those tensors; callers using async_op=True read them via
  // get_future().value().
  explicit FakeWork(std::vector<at::Tensor> result = {})
      : result_(std::move(result)) {}

  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
    return true;
  }

  // A fake collective is done the moment it is "issued". Work's default reads
  // completed_, which nothing here ever sets, so without this a fake op would
  // report as permanently in flight to anyone polling it even though wait()
  // returns immediately.
  bool isCompleted() override {
    return true;
  }

  std::vector<at::Tensor> result() override {
    TORCH_CHECK(!result_.empty(), "FakeWork: this work recorded no output.");
    return result_;
  }

  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
    if (result_.empty()) {
      auto fut = c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());
      fut->markCompleted();
      return fut;
    }
    auto fut = c10::make_intrusive<c10::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()));
    fut->markCompleted(result_);
    return fut;
  }

 private:
  std::vector<at::Tensor> result_;
};

class FakeProcessGroup : public Backend {
 public:
  struct Options : Backend::Options {
    explicit Options() : Backend::Options("fake") {}

    c10::intrusive_ptr<Backend::Options> clone() const override {
      return c10::make_intrusive<Options>(*this);
    }

    int fake_option = 0;
    bool error_on_collective = false;
    // See NOTE [FakeProcessGroup uniform-rank simulation]
    bool simulate_uniform_ranks = false;
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

  // Nullable accessor exposed as the Python `.options` property, mirroring the
  // getOptions()/getBackendOptions() split on ProcessGroupNCCL and
  // ProcessGroupGloo. Returns null when the user constructed the group without
  // options, which callers (and test_device_mesh) rely on to tell whether an
  // options override was supplied.
  c10::intrusive_ptr<Options> getOptions() {
    return options_;
  }

  // options_ may be null when the user passed no options. splitGroup and
  // mergeRemoteGroup unconditionally dereference the result, so coalesce to a
  // fresh default Options rather than returning null. The child of a
  // no-options parent thus carries a real Options, matching NCCL/Gloo.
  c10::intrusive_ptr<Backend::Options> getBackendOptions() override {
    auto opts = options_ ? options_ : c10::make_intrusive<Options>();
    return c10::static_intrusive_pointer_cast<Backend::Options>(opts);
  }

  void setTimeout(std::chrono::milliseconds /* timeout */) override {
    // FakeProcessGroup does no real communication, so there is no timeout to
    // configure. Override as a no-op so callers don't hit the warning the
    // Backend base class emits for unsupported backends.
  }

  bool supportsSplitting() const override {
    return true;
  }

  // Create a sub-group from a subset of the parent's ranks. The fake backend
  // performs no real communication, so there is no split collective to join:
  // ranks outside the subgroup simply return nullptr (signalling
  // non-membership), and members return a fresh FakeProcessGroup whose rank is
  // their position within the sorted subgroup.
  c10::intrusive_ptr<Backend> split(
      const c10::intrusive_ptr<Store>& /* store */,
      const std::vector<int>& ranks,
      const c10::intrusive_ptr<Backend::Options>& opts) override {
    auto it = std::find(ranks.begin(), ranks.end(), rank_);
    if (it == ranks.end()) {
      return nullptr;
    }
    auto groupRank = static_cast<int>(std::distance(ranks.begin(), it));
    auto fakeOpts = c10::dynamic_intrusive_pointer_cast<Options>(opts);
    TORCH_CHECK(fakeOpts != nullptr, "opts not a FakeProcessGroup::Options.");
    return c10::make_intrusive<FakeProcessGroup>(
        groupRank, static_cast<int>(ranks.size()), std::move(fakeOpts));
  }

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& /* opts */ = BroadcastOptions()) override {
    checkCollectiveError();
    // Identity under either contract: every rank already holds the value.
    return uniformRanks() ? c10::make_intrusive<FakeWork>(tensors)
                          : c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override {
    checkCollectiveError();
    return uniformReduceAll(tensors, opts.reduceOp);
  }

  c10::intrusive_ptr<Work> allreduce_sparse(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override {
    checkCollectiveError();
    if (uniformRanks()) {
      TORCH_CHECK(
          opts.reduceOp.op_ != ReduceOp::PRODUCT,
          "FakeProcessGroup: allreduce_sparse does not support PRODUCT under "
          "simulate_uniform_ranks.");
    }
    return uniformReduceAll(tensors, opts.reduceOp);
  }

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override {
    checkCollectiveError();
    return uniformReduceAll(tensors, opts.reduceOp);
  }

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override {
    checkCollectiveError();
    // Real backends write the reduced value only into the root's tensor and
    // leave every other rank's unspecified. Mirror that, so a caller that
    // wrongly reads a non-root result sees the same thing it would in
    // production instead of a value only this backend would produce.
    if (rank_ != opts.rootRank) {
      return c10::make_intrusive<FakeWork>();
    }
    return uniformReduceAll(tensors, opts.reduceOp);
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
      const AllgatherOptions& /* opts */ = AllgatherOptions()) override {
    checkCollectiveError();
    // See note in _allgather_base below.
    at::AutoDispatchBelowAutograd guard;
    for (auto& tensor : outputTensors[0]) {
      tensor.copy_(inputTensors[0]);
    }
    return uniformRanks() ? c10::make_intrusive<FakeWork>(outputTensors[0])
                          : c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> all_gather_single(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) override {
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
    return uniformRanks()
        ? c10::make_intrusive<FakeWork>(std::vector<at::Tensor>{outputBuffer})
        : c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) override {
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
    if (uniformRanks()) {
      std::vector<at::Tensor> results;
      for (const auto& outputTensorList : outputTensorLists) {
        results.insert(
            results.end(), outputTensorList.begin(), outputTensorList.end());
      }
      return c10::make_intrusive<FakeWork>(std::move(results));
    }
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> all_gather_single_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) override {
    checkCollectiveError();
    // See note in _allgather_base above.
    at::AutoDispatchBelowAutograd guard;
    for (size_t i = 0; i < outputs.size(); ++i) {
      auto chunks = outputs[i].chunk(size_);
      for (auto& chunk : chunks) {
        chunk.copy_(inputs[i]);
      }
    }
    return uniformRanks() ? c10::make_intrusive<FakeWork>(outputs)
                          : c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override {
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
      if (uniformRanks()) {
        return c10::make_intrusive<FakeWork>(outputTensors[0]);
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
    return uniformRanks() && rank_ == opts.rootRank
        ? c10::make_intrusive<FakeWork>(outputTensors)
        : c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
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
      if (uniformRanks()) {
        applyUniformReduction(outputTensors[i], opts.reduceOp);
      }
    }
    return uniformRanks() ? c10::make_intrusive<FakeWork>(outputTensors)
                          : c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> reduce_scatter_single(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    checkCollectiveError();
    TORCH_CHECK(
        inputBuffer.numel() == outputBuffer.numel() * size_,
        "input tensor must be the same size as output size times world size");
    // See note in _allgather_base above.
    at::AutoDispatchBelowAutograd guard;
    auto chunks = inputBuffer.chunk(size_);
    outputBuffer.copy_(chunks[rank_]);
    if (uniformRanks()) {
      applyUniformReduction(outputBuffer, opts.reduceOp);
      return c10::make_intrusive<FakeWork>(
          std::vector<at::Tensor>{outputBuffer});
    }
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> reduce_scatter_single_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
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
      if (uniformRanks()) {
        applyUniformReduction(outputs[i], opts.reduceOp);
      }
    }
    return uniformRanks() ? c10::make_intrusive<FakeWork>(outputs)
                          : c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> all_to_all_single(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& /* opts */ = AllToAllOptions()) override {
    checkCollectiveError();
    c10d::checkSplitSizes(inputSplitSizes, inputBuffer, size_);
    c10d::checkSplitSizes(outputSplitSizes, outputBuffer, size_);
    TORCH_CHECK(
        inputBuffer.is_contiguous(inputBuffer.suggest_memory_format()),
        "Input tensor must be contiguous");
    TORCH_CHECK(
        outputBuffer.is_contiguous(outputBuffer.suggest_memory_format()),
        "Output tensor must be contiguous");
    // See note in _allgather_base above.
    at::AutoDispatchBelowAutograd guard;
    // Approximation: inputs from other ranks are unavailable here, so copy as
    // much of the local input as fits and zero the remainder.
    auto flat_input = inputBuffer.as_strided(
        {inputBuffer.numel()}, {1}, inputBuffer.storage_offset());
    auto flat_output = outputBuffer.as_strided(
        {outputBuffer.numel()}, {1}, outputBuffer.storage_offset());
    if (uniformRanks()) {
      // Every peer holds what we hold, so the segment peer j sends to us is
      // the segment we would send to a peer in our own position: our input
      // split at index rank_. Fill each output slot from it, which keeps the
      // split structure self-consistent and lets callers reshape the result.
      //
      // A slot need not be the size of that segment: a caller may declare
      // asymmetric splits or a receive-only rank. The output shape the caller
      // asked for is what it gets, so tile or truncate to fill it.
      auto mine = splitSegment(
          flat_input, inputSplitSizes, rank_, size_, rowSizeOf(inputBuffer));
      const auto outputRowSize = rowSizeOf(outputBuffer);
      for (int64_t j = 0; j < size_; ++j) {
        auto slot = splitSegment(
            flat_output, outputSplitSizes, j, size_, outputRowSize);
        fillByTiling(slot, mine);
      }
      return c10::make_intrusive<FakeWork>(
          std::vector<at::Tensor>{outputBuffer});
    }
    flat_output.zero_();
    auto copy_size = std::min(flat_input.numel(), flat_output.numel());
    if (copy_size > 0) {
      flat_output.narrow(0, 0, copy_size)
          .copy_(flat_input.narrow(0, 0, copy_size));
    }
    return c10::make_intrusive<FakeWork>();
  }

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& /* opts */ = AllToAllOptions()) override {
    checkCollectiveError();
    auto invalidArgument = [](const std::string& msg) {
      TORCH_CHECK(false, "FakeProcessGroup::alltoall: ", msg);
    };
    assertAllToAllTensorListSizes(
        invalidArgument, outputTensors.size(), inputTensors.size(), size_);
    // See note in _allgather_base above.
    at::AutoDispatchBelowAutograd guard;
    if (uniformRanks()) {
      // Every peer sends us what it would send to our position, which is our
      // own entry at index rank_, so it is the same source for every slot.
      // flatView walks storage linearly, so a non-contiguous tensor would read
      // or write the wrong elements. all_to_all_single rejects those outright;
      // do the same here rather than corrupt them silently.
      TORCH_CHECK(
          inputTensors[rank_].is_contiguous(),
          "FakeProcessGroup::alltoall: input tensor must be contiguous "
          "under simulate_uniform_ranks");
      auto mine = flatView(inputTensors[rank_]);
      for (auto& output : outputTensors) {
        TORCH_CHECK(
            output.is_contiguous(),
            "FakeProcessGroup::alltoall: output tensor must be contiguous "
            "under simulate_uniform_ranks");
        // Shapes need not match, so tile.
        fillByTiling(flatView(output), mine);
      }
      return c10::make_intrusive<FakeWork>(outputTensors);
    }
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
  // NOTE [FakeProcessGroup uniform-rank simulation]
  // With Options::simulate_uniform_ranks the group models a world in which
  // every rank holds data identical to this one. That single assumption makes
  // every collective well defined from local inputs alone, which the default
  // approximations are not: they ignore the reduce op entirely and fill
  // all-to-all outputs without regard to the split structure, so a caller that
  // reshapes collective output (TorchRec's KeyedJaggedTensor does) gets a
  // wrongly sized tensor.
  //
  // The modeling cost is that ranks cannot diverge, so this cannot surface
  // rank-divergence bugs. It is off by default; existing behavior is untouched.
  //
  // `scatter` on a non-root rank stays outside the contract because it has no
  // access to the root's list. Sparse all-reduce supports the same contract
  // except for PRODUCT, which sparse tensors do not support.
  bool uniformRanks() const {
    return options_ != nullptr && options_->simulate_uniform_ranks;
  }

  // Apply the uniform-rank reduction to every tensor in place, or no-op when
  // the contract is off.
  c10::intrusive_ptr<Work> uniformReduceAll(
      std::vector<at::Tensor>& tensors,
      const ReduceOp& reduceOp) const {
    if (!uniformRanks()) {
      return c10::make_intrusive<FakeWork>();
    }
    at::AutoDispatchBelowAutograd guard;
    for (auto& tensor : tensors) {
      applyUniformReduction(tensor, reduceOp);
    }
    return c10::make_intrusive<FakeWork>(tensors);
  }

  // Combine `size_` identical contributions in place. Every reduce op has a
  // closed form once the contributions are known to be equal, so the contract
  // is total: no op falls back to a silently wrong approximation.
  // `tensor` is an out parameter: every branch mutates it in place.
  void applyUniformReduction(at::Tensor& tensor, const ReduceOp& reduceOp)
      const {
    // Bool is the one dtype where the scaling below has no in-place form:
    // mul_/pow_ promote to an integral result that cannot be stored back into
    // a bool tensor. Under c10d's nonzero-is-true convention both reductions
    // are the identity on equal bools anyway, so take that branch instead of
    // raising where a real backend succeeds.
    const bool isBool = tensor.scalar_type() == at::kBool;
    switch (reduceOp.op_) {
      case ReduceOp::SUM:
        if (!isBool) {
          tensor.mul_(size_);
        }
        return;
      case ReduceOp::AVG:
      case ReduceOp::MIN:
      case ReduceOp::MAX:
      case ReduceOp::BAND:
      case ReduceOp::BOR:
        // Averaging, taking an extremum, and bitwise AND/OR are all idempotent
        // on equal operands, so each leaves the local value unchanged.
        return;
      case ReduceOp::PRODUCT:
        // Overflows silently for integer dtypes once the world is more than a
        // few ranks wide, and saturates to inf for floats. That matches what a
        // real backend would produce for the same inputs, so it is left alone
        // rather than clamped.
        if (!isBool) {
          tensor.pow_(size_);
        }
        return;
      case ReduceOp::BXOR:
        // x ^ x == 0, so an even number of equal contributions cancels out
        // entirely and an odd number leaves one behind.
        if (size_ % 2 == 0) {
          tensor.zero_();
        }
        return;
      case ReduceOp::PREMUL_SUM: {
        // Summing `factor * x` over `size_` equal ranks scales the local
        // value by `size_ * factor`.
        // mul_ by a floating factor cannot cast into an integral or bool
        // tensor. Real NCCL PREMUL_SUM is float-only too, so say that rather
        // than surface an opaque in-place dtype error.
        TORCH_CHECK(
            tensor.is_floating_point() || tensor.is_complex(),
            "FakeProcessGroup: PREMUL_SUM requires a floating point or "
            "complex tensor, got ",
            tensor.scalar_type(),
            ".");
        auto supplement =
            c10::dynamic_intrusive_pointer_cast<PreMulSumSupplement>(
                reduceOp.supplement_);
        TORCH_CHECK(
            supplement != nullptr,
            "FakeProcessGroup: PREMUL_SUM was given without its scaling "
            "factor. Build it with torch.distributed._make_nccl_premul_sum.");
        if (supplement->tensor_factor.defined()) {
          // The factor may have been built on another device; real backends
          // co-locate it internally rather than failing the multiply.
          tensor.mul_(supplement->tensor_factor.to(tensor.device()));
        } else {
          tensor.mul_(supplement->double_factor);
        }
        tensor.mul_(size_);
        return;
      }
      default:
        TORCH_CHECK(
            false,
            "FakeProcessGroup: unrecognized reduce op ",
            static_cast<int>(reduceOp.op_),
            " under simulate_uniform_ranks.");
    }
  }

  // Fill `dst` by repeating `src`, or cut it short when `dst` is the smaller.
  // Both callers let a slot have a shape of its own, so there is no size
  // equality to lean on. Real backends write the whole slot; truncating would
  // leave the tail uninitialized and a caller that reshapes the output would
  // then read garbage.
  static void fillByTiling(const at::Tensor& dst, const at::Tensor& src) {
    const auto dstNumel = dst.numel();
    const auto srcNumel = src.numel();
    if (dstNumel == 0) {
      return;
    }
    if (srcNumel == 0) {
      // This peer contributes nothing, so the slot receives nothing. Zero it
      // rather than leaving whatever the caller's buffer happened to hold.
      dst.zero_();
      return;
    }
    int64_t written = 0;
    while (written < dstNumel) {
      const auto n = std::min(srcNumel, dstNumel - written);
      dst.narrow(0, written, n).copy_(src.narrow(0, 0, n));
      written += n;
    }
  }

  // A flat 1-D view over `t`'s storage, for element-wise segment copies.
  static at::Tensor flatView(const at::Tensor& t) {
    return t.as_strided({t.numel()}, {1}, t.storage_offset());
  }

  // The contiguous segment of `flat` belonging to peer `index`. An empty split
  // list is the c10d convention for an equal division; otherwise each split is
  // a row count that `rowSize` converts into elements, matching
  // computeLengthsAndOffsets. Getting that conversion wrong silently
  // under-writes the output for anything that is not 1-D.
  static at::Tensor splitSegment(
      const at::Tensor& flat,
      const std::vector<int64_t>& splitSizes,
      int64_t index,
      int64_t worldSize,
      int64_t rowSize) {
    if (splitSizes.empty()) {
      // checkSplitSizes has already verified size(0) divides the world, so
      // numel does too and this covers the buffer exactly.
      const auto chunk = flat.numel() / worldSize;
      return flat.narrow(0, chunk * index, chunk);
    }
    TORCH_CHECK(
        static_cast<int64_t>(splitSizes.size()) >= worldSize,
        "FakeProcessGroup: split list has ",
        splitSizes.size(),
        " entries for a world of ",
        worldSize);
    int64_t offset = 0;
    for (int64_t i = 0; i < index; ++i) {
      offset += splitSizes[i] * rowSize;
    }
    const auto length = splitSizes[index] * rowSize;
    TORCH_CHECK(
        offset + length <= flat.numel(),
        "FakeProcessGroup: split segment [",
        offset,
        ", ",
        offset + length,
        ") exceeds the ",
        flat.numel(),
        "-element buffer");
    return flat.narrow(0, offset, length);
  }

  // Elements per row along dim 0. c10d split sizes count rows, not elements,
  // so every split has to be scaled by this to become an offset into the flat
  // buffer. Kept character-for-character in step with the row_size line of
  // computeLengthsAndOffsets (Utils.hpp), including its guard for an empty
  // leading dim, so the fake backend segments a buffer exactly as a real one.
  static int64_t rowSizeOf(const at::Tensor& t) {
    const auto dim0 = t.size(0);
    return dim0 ? t.numel() / dim0 : 1;
  }

  void checkCollectiveError() {
    TORCH_CHECK(
        !options_ || !options_->error_on_collective,
        "FakeProcessGroup collective operation error (error_on_collective=true)");
  }
};

} // namespace c10d
