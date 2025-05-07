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

// look what you made me do >.<
// Divergent paths for per-Impl stream recording that leak implementation
// details of the impls should not be needed here.
// See https://github.com/pytorch/pytorch/issues/60306
// TODO: clean this up when https://github.com/pytorch/pytorch/issues/60306 is
// improved
void _record_stream_any_impl(Variable& var, const c10::Stream& stream) {
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)

  if (stream.device_index() != var.device().index()) {
    return;
  }

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

// Note: [Stream sync contract when dealing with multi-deviced-ness]
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
//
// Note: [Autograd Producer-Consumer Stream Syncs]
//
// The actual wait and record stream happens prior to the consumer's
// execution. The logic here is mainly responsible for handling the
// synchronization needed for accumulation and recording the event that the
// consumer should wait on later.
//
// First producer
// ==============
// 1) Determine the accumulation stream (which may or may not be used):
//    case A) var's device matches consumer node's canonical device
//            (The producer node's canonical device may or may not match)
//            -> accumulator stream = consumer stream
//    case B) var's device matches producer node's canonical device
//            and does not match consumer node's canonical device
//            -> accumulator stream = producer stream
//    case C) var device matches neither
//            -> accumulator stream = var device's current stream
//            See Note [Stream sync contract when dealing with
//            multi-deviced-ness]
// 2) Because we are the first producer, there's no accumulation necessary.
//    Just move var into the buffer.
// 3) Update the ready_events and streams for the current position.
//    ready_events are events you need to wait for to ensure the corresponding
//    buffers are ready. The events are updated as we accumulate into the
//    buffer.
//
// Nth producer
// ============
// 1) Synchronize for accumulation. Accumulation operates on both the new
//   incoming gradient and the existing gradient in the buffer.
//   (i) wait stream and (ii) record stream to make sure both are ready to be
//   used on the accumulation stream.
// 2) Accumulation on the accumulation straem
// 3) Update the ready event and stream for the current position.
//
// NOLINTBEGIN(bugprone-unchecked-optional-access)
void InputBuffer::add(
    size_t pos,
    Variable&& var,
    const std::optional<c10::Stream>& opt_producer_stream_,
    const std::optional<c10::Stream>& opt_consumer_stream) {
  TORCH_INTERNAL_ASSERT(pos < buffer.size());

  if (!var.defined()) {
    return;
  }
  const auto device = var.device();
  const auto device_type = device.type();
  // TODO: Use at::accelerator::isAccelerator(device->type()) instead
  bool is_accelerator =
      device.is_cuda() || device.is_mtia() || device.is_privateuseone();
  //
  // [ Non-accelerator case ]
  //
  if (!is_accelerator) {
    if (!buffer[pos].defined()) {
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
                 at::accelerator::getCurrentStream(device.index())));

  TORCH_INTERNAL_ASSERT(opt_consumer_stream && opt_producer_stream);

  // See Note: [Autograd Producer-Consumer Stream Syncs]
  if (!opt_accum_streams[pos].has_value()) {
    TORCH_INTERNAL_ASSERT(!buffer[pos].defined());
    // [ First producer ]
    // 1)
    if (opt_consumer_stream->device() == device) {
      // Case A
      opt_accum_streams[pos] = opt_consumer_stream;
    } else if (opt_producer_stream->device() == device) {
      // Case B
      opt_accum_streams[pos] = opt_producer_stream;
    } else {
      // Case C
      opt_accum_streams[pos] =
          at::accelerator::getCurrentStream(device.index());
    }
    // 2)
    buffer[pos] = std::move(var);
    // 3)
    ready_events[pos] = c10::Event{device_type};
    ready_events[pos]->record(*opt_producer_stream);
    ready_streams[pos] = opt_producer_stream;
  } else {
    // [ Nth producer ]
    auto accum_stream = opt_accum_streams[pos];
    // 1)
    if (*accum_stream != *opt_producer_stream) {
      auto event = c10::Event{device_type};
      event.record(*opt_producer_stream);
      accum_stream->wait(event);
      _record_stream_any_impl(var, *accum_stream);
    }
    if (*accum_stream != *ready_streams[pos]) {
      accum_stream->wait(*ready_events[pos]);
      _record_stream_any_impl(buffer[pos], *opt_accum_streams[pos]);
    }
    // 2)
    c10::OptionalStreamGuard stream_guard{accum_stream};
    accumulate(buffer, pos, std::move(var));
    // 3)
    ready_events[pos] = c10::Event{device_type};
    ready_events[pos]->record(*accum_stream);
    ready_streams[pos] = accum_stream;
  }
}
// NOLINTEND(bugprone-unchecked-optional-access)

auto InputBuffer::variables(InputBuffer&& g) -> std::vector<Variable> {
  std::vector<Variable> result = std::move(g.buffer);
  return result;
}

} // namespace torch::autograd
