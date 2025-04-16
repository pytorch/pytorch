#include <torch/csrc/autograd/input_buffer.h>

#include <ATen/CachedTensorUtils.h>
#include <ATen/LegacyBatchedTensorImpl.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/core/grad_mode.h>
#include <ATen/native/SparseTensorUtils.h>

#include <c10/core/DeviceGuard.h>
#include <c10/core/Event.h>
#include <c10/core/StreamGuard.h>
#include <optional>

#include <cstddef>
#include <utility>
#include <vector>

namespace torch::autograd {

namespace {
// look what you made me do >.<
// Divergent paths for per-Impl stream recording that leak implementation
// details of the impls should not be needed here.
// See https://github.com/pytorch/pytorch/issues/60306
// TODO: clean this up when https://github.com/pytorch/pytorch/issues/60306 is
// improved
void mb_record_stream_any_impl(
    Variable& var,
    const c10::Stream& stream,
    const c10::Stream& prev_stream) {
  if (stream == prev_stream) {
    return;
  }
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  const auto guard = c10::impl::VirtualGuardImpl(device_of(var).value().type());

  if (C10_UNLIKELY(at::isBatchedTensor(var))) {
    auto* impl = at::maybeGetBatchedImpl(var);
    if (impl) {
      guard.recordDataPtrOnStream(impl->value().storage().data_ptr(), stream);
    } else {
      TORCH_INTERNAL_ASSERT(false, "Expected batched tensor");
    }
  } else {
    switch (var.layout()) {
      case c10::kSparseCsr:
      case c10::kSparseCsc:
      case c10::kSparseBsr:
      case c10::kSparseBsc: {
        auto* impl = at::sparse_csr::get_sparse_csr_impl(var);
        guard.recordDataPtrOnStream(
            impl->values().storage().data_ptr(), stream);
        guard.recordDataPtrOnStream(
            impl->compressed_indices().storage().data_ptr(), stream);
        guard.recordDataPtrOnStream(
            impl->plain_indices().storage().data_ptr(), stream);
        break;
      }
      case c10::kSparse: {
        auto* impl = at::sparse::get_sparse_impl(var);
        guard.recordDataPtrOnStream(
            impl->values().storage().data_ptr(), stream);
        guard.recordDataPtrOnStream(
            impl->indices().storage().data_ptr(), stream);
        break;
      }
      case c10::kStrided:
        guard.recordDataPtrOnStream(var.storage().data_ptr(), stream);
        break;
      default:
        TORCH_INTERNAL_ASSERT(
            false, "Unknown layout in record_stream_any_impl");
    }
  }
}

bool can_accumulate_inplace(const Variable& v) {
  return (
      // `v` is a "vanilla" Tensor
      !(at::isTensorSubclassLike(v) || v._is_zerotensor() || v.is_nested()) &&

      // with a favorable memory layout
      v.is_non_overlapping_and_dense() &&

      // and we hold the last reference
      at::caching::adjusted_use_count(v) == 1 && v.has_storage() &&
      v.storage().use_count() == 1);
}

void mb_wait_stream(
    const std::optional<c10::Stream>& lhs,
    const std::optional<c10::Stream>& rhs,
    c10::DeviceType device_type) {
  TORCH_INTERNAL_ASSERT(lhs && rhs);
  if (*lhs == *rhs) {
    return;
  }
  auto event = c10::Event{device_type};
  event.record(*rhs);
  lhs->wait(event);
}

} // anonymous namespace

static void accumulate(
    std::vector<Variable>& buffer,
    const size_t pos,
    Variable&& var) {
  TORCH_INTERNAL_ASSERT(pos < buffer.size());
  auto& old_var = buffer[pos];
  // If we hold the last reference to `old_var` AND its storage we will try to
  // repurpose it to store the output. (Or, if `old_var` is sparse then `var`
  // becomes the candidate output Tensor.) We only do this if:
  //  1) GradMode is disabled since Autograd has special handling for inplace
  //     mutation which we don't want to trigger.
  //
  //  2) We hold the last reference.
  //     (Both `.use_count` and `.storage().use_count()` are one)
  //
  //  3) The candidate tensor is a contiguous, non-overlapping, dense, and
  //     otherwise stock standard Tensor.
  //
  //  4) The candidate is mutable. Currently only ZeroTensors are immutable.
  //
  //  5) The other Tensor is not a Tensor subclass (except sparse), since
  //     it's hard to predict the semantics of arbitrary subclass behavior.

  // NOLINTNEXTLINE(bugprone-branch-clone)
  if (at::GradMode::is_enabled()) {
    buffer[pos] = old_var + var;
  } else if (
      // ATen doesn't route sparse additions correctly...
      old_var.is_sparse() || old_var.is_sparse_csr()) {
    if (can_accumulate_inplace(var)) {
      buffer[pos] = var.add_(old_var);
    } else {
      buffer[pos] = var + old_var;
    }
  } else if (
      can_accumulate_inplace(old_var) && !at::isTensorSubclassLike(var)) {
    buffer[pos] = old_var.add_(var);
  } else {
    buffer[pos] = old_var + var;
  }
}

// Note [Stream sync contract when dealing with multi-deviced-ness]
//
// An operator can deal with multiple devices, e.g. if it does a device
// transfer, etc. However, for the purpose of stream synchronization, the engine
// is only aware of single canonical device/stream for each op/node.
//
// For the proper synchronization, the op author should make sure of the
// following:
//
// 1) A node producing a gradient should have it ready on the current
//   stream during node execution.
// 2) A node consuming a gradient should wait on the current stream before using
//    it.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
void InputBuffer::add(
    size_t pos,
    Variable&& var,
    const std::optional<c10::Stream>& opt_producer_stream_,
    const std::optional<c10::Stream>& opt_consumer_stream,
    int num_dependencies) {
  TORCH_INTERNAL_ASSERT(pos < buffer.size());

  if (!var.defined()) {
    return;
  }
  int curr_producer_idx = accum_counts[pos]++;
  auto const device = device_of(var);
  TORCH_INTERNAL_ASSERT(device.has_value());
  const auto device_type = device->type();
  bool is_accelerator =
      device->is_cuda() || device->is_mtia() || device->is_privateuseone();

  //
  // [ Non-accelerator case ]
  //
  if (!is_accelerator) {
    if (curr_producer_idx == 0) {
      TORCH_INTERNAL_ASSERT(!buffer[pos].defined())
      buffer[pos] = std::move(var);
    } else {
      c10::OptionalDeviceGuard device_guard{device};
      accumulate(buffer, pos, std::move(var));
    }
    return;
  }

  // Handle the case where var is on an accelerator but producer node has no
  // canonical stream, e.g. this can happen if forward is DtoH
  const std::optional<c10::Stream>& opt_producer_stream =
      (opt_producer_stream_.has_value()
           ? opt_producer_stream_
           : std::optional<c10::Stream>(
                 at::accelerator::getCurrentStream(device->index())));

  TORCH_INTERNAL_ASSERT(opt_consumer_stream && opt_producer_stream);

  //
  // [ Single producer case ]
  //
  // If during forward, a tensor is used only once, there's only a single
  // producer node corresponding to the input buffer we're adding to.
  // There is no need to accumulate in this case.
  if (num_dependencies == 1) {
    mb_wait_stream(opt_consumer_stream, opt_producer_stream, device_type);
    mb_record_stream_any_impl(
        var, *opt_consumer_stream, /*prev_stream=*/*opt_producer_stream);
    TORCH_INTERNAL_ASSERT(curr_producer_idx == 0);
    TORCH_INTERNAL_ASSERT(!buffer[pos].defined());
    buffer[pos] = std::move(var);
    return;
  }
  //
  // [ Multiple producer case ]
  //
  // First producer
  // ==============
  //
  // - Determine the accumulation stream:
  //    case 1) var's device matches consumer node's canonical device
  //            (The producer node's canonical device may or may not match)
  //            -> accumulator stream = consumer stream
  //    case 2) var's device matches producer node's canonical device
  //            and does not match consumer node's canonical device
  //            -> accumulator stream = producer stream
  //    case 3) var device matches neither
  //            -> accumulator stream = var device's current stream
  //            See Note [Stream sync contract when dealing with
  //            multi-deviced-ness]
  //
  // - Record stream (var will be used on the accum stream)
  // - Because we are the first producer, there's no accumulation necessary.
  //   Just move var into the buffer.
  //
  // Nth producer
  // ============
  //
  // - The accumulation stream (determined at "first producer") waits for
  //   the new producer stream
  // - Record stream (var will be used on the accum stream)
  // - Accumulate var into buffer
  // - If we are the last producer, the consumer waits for the accumulation
  //   stream. (No-op for case 1)
  //
  //
  // Note: [Delay synchronizing the first producer]
  //
  // In case 1, we delay synchronizing the first producer with the
  // consumer until we see the second producer. For more details, see
  // "test_side_stream_backward_overlap".
  //
  // There are two pieces of logic that happen:
  //    Part 1: At the first iteration, record an event on the first
  //            producer stream and stash it.
  //    Part 2: At the second iteration, have the consumer wait for
  //            for the stashed event.
  //
  bool matches_consumer = opt_consumer_stream->device() == *device;

  if (curr_producer_idx == 0) {
    // [ First producer ]
    // Determine the accumulation stream
    if (matches_consumer) {
      // Case 1
      opt_accum_streams[pos] = opt_consumer_stream;
      // Part 1 of Note: [Delay synchronizing the first producer]
      opt_first_producer_evts[pos] = c10::Event{device_type};
      opt_first_producer_evts[pos]->record(*opt_producer_stream);
      opt_first_producer_streams[pos] = opt_producer_stream;
    } else if (opt_producer_stream->device() == *device) {
      // Case 2
      opt_accum_streams[pos] = opt_producer_stream;
    } else {
      // Case 3
      opt_accum_streams[pos] =
          at::accelerator::getCurrentStream(device->index());
    }
    mb_record_stream_any_impl(
        var,
        *opt_accum_streams[pos],
        /*prev_stream=*/*opt_producer_stream);
    TORCH_INTERNAL_ASSERT(!buffer[pos].defined());
    buffer[pos] = std::move(var);
  } else {
    // [ Nth producer ]
    auto accum_stream = opt_accum_streams[pos];
    TORCH_INTERNAL_ASSERT(accum_stream);
    mb_wait_stream(accum_stream, opt_producer_stream, device_type);
    mb_record_stream_any_impl(
        var, *accum_stream, /*prev_stream=*/*opt_producer_stream);
    if (matches_consumer && curr_producer_idx == 1) {
      TORCH_INTERNAL_ASSERT(
          opt_first_producer_streams[pos] && opt_first_producer_evts[pos]);
      // Part 2 of Note: [Delay synchronizing the first producer]
      if (*accum_stream != *opt_first_producer_streams[pos]) {
        accum_stream->wait(*opt_first_producer_evts[pos]);
      }
    }
    {
      c10::OptionalStreamGuard stream_guard{accum_stream};
      accumulate(buffer, pos, std::move(var));
    }
    if (curr_producer_idx == num_dependencies - 1) {
      mb_wait_stream(opt_consumer_stream, accum_stream, device_type);
    }
  }
}
// NOLINTEND(bugprone-unchecked-optional-access)

auto InputBuffer::variables(InputBuffer&& g) -> std::vector<Variable> {
  std::vector<Variable> result = std::move(g.buffer);
  return result;
}

} // namespace torch::autograd
