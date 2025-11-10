#include <torch/csrc/autograd/functions/accumulate_grad.h>
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
#include <c10/util/Logging.h>
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
void record_stream_any_impl(Variable& var, const c10::Stream& stream) {
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

// Note: [Stream sync contract when dealing with multi-deviced-ness]
//
// An operator can deal with multiple devices, e.g. if it does a device
// transfer, etc. However, for the purpose of stream synchronization, the engine
// is only aware of single canonical device/stream for each autograd Node.
//
// For the proper synchronization, the Node author should make sure of the
// following:
//
// 1) A node consuming a gradient should wait on the canonical stream before
//    using it.
// 2) A node producing a gradient should have it ready on the canonical
//    stream during node execution.
//

// Note: [Autograd Producer-Consumer Stream Syncs]
//
// The producer-consumer stream syncs are partially handled in this method
// and partially handled in the engine prior to the consumer's execution.
// The logic here is mainly responsible for handling the synchronization needed
// for accumulation and recording the event that the consumer should wait on
// later. The corresponding wait and record_stream happens in the engine.
//
// First producer
// ==============
// There are several things we need to do upon seeing the first producer:
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
// 3) Update the ready_events and streams for the current position.**
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
// 2) Accumulate on the accumulation stream
// 3) Update the ready event and stream for the current position.**
//
// **As an optimization, we avoid creating and recording an event if we
// know that we won't need to wait on it, saving on the order of microseconds.
//
void InputBuffer::add(
    size_t pos,
    Variable&& var,
    const std::optional<c10::Stream>& opt_producer_stream_,
    const std::optional<c10::Stream>& opt_consumer_stream_,
    Node* fn) {
  TORCH_INTERNAL_ASSERT(pos < buffer.size());

  if (!var.defined()) {
    return;
  }
  const auto device = var.device();
  const auto device_type = device.type();
  bool is_accelerator = at::accelerator::isAccelerator(device.type());
  //
  // Non-accelerator case
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

  // opt_consumer_stream is always non-null when is_accelerator is true
  // when InputBuffer is used in the engine. InputBuffer is also called
  // elsewhere however! (e.g. other engine implementations)
  const std::optional<c10::Stream>& opt_consumer_stream =
      (opt_consumer_stream_.has_value()
           ? opt_consumer_stream_
           : std::optional<c10::Stream>(
                 at::accelerator::getCurrentStream(device.index())));

  TORCH_INTERNAL_ASSERT(opt_consumer_stream && opt_producer_stream);

  if (*opt_consumer_stream != *opt_producer_stream &&
      dynamic_cast<AccumulateGrad*>(fn) &&
      at::globalContext().warnOnAccumulateGradStreamMismatch()) {
    TORCH_WARN_ONCE(
        "The AccumulateGrad node's stream does not match the stream of the node that produced "
        "the incoming gradient. This may incur unnecessary synchronization and break CUDA graph "
        "capture if the AccumulateGrad node's stream is the default stream. This mismatch is "
        "caused by an AccumulateGrad node created prior to the current iteration being kept alive. "
        "This can happen if the autograd graph is still being kept alive by tensors such as the "
        "loss, or if you are using DDP, which will stash a reference to the node. To resolve the "
        "mismatch, delete all references to the autograd graph or ensure that DDP initialization is "
        "performed under the same stream as subsequent forwards. If the mismatch is intentional, "
        "you can use torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False) to suppress this "
        "warning.");
  }
  // See Note: [Autograd Producer-Consumer Stream Syncs]
  if (!opt_accum_streams[pos].has_value()) {
    // [ First producer ]
    TORCH_INTERNAL_ASSERT(!buffer[pos].defined());
    // 1)
    if (opt_consumer_stream->device() == device) {
      // Case A
      opt_accum_streams[pos] = opt_consumer_stream;
      if (*opt_consumer_stream != *opt_producer_stream) {
        // We will end up doing record_stream on the accumulation stream
        // (which is the consumer stream) later, but we also need to do
        // it here in case we don't end up accumulating.
        record_stream_any_impl(var, *opt_consumer_stream);
      }
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
    auto& opt_accum_stream = opt_accum_streams[pos];
    TORCH_INTERNAL_ASSERT(opt_accum_stream.has_value());
    if (*opt_consumer_stream != *opt_producer_stream ||
        *opt_accum_stream != *opt_producer_stream) {
      // Either the consumer or accum stream waits for the producer
      // stream depending on whether accumulation is needed.
      auto event = c10::Event{device_type};
      event.record(*opt_producer_stream);
      ready_events[pos] = std::move(event);
    }
    ready_streams[pos] = opt_producer_stream;
  } else {
    // [ Nth producer ]
    auto accum_stream = opt_accum_streams[pos];
    auto& ready_event = ready_events[pos];
    auto& ready_stream = ready_streams[pos];
    TORCH_INTERNAL_ASSERT(accum_stream && ready_stream);
    // 1)
    if (*accum_stream != *opt_producer_stream) {
      auto event = c10::Event{device_type};
      event.record(*opt_producer_stream);
      accum_stream->wait(event);
      record_stream_any_impl(var, *accum_stream);
    }
    if (*accum_stream != *ready_stream) {
      TORCH_INTERNAL_ASSERT(ready_event);
      accum_stream->wait(*ready_event);
      // This is redundant for case A, but needed for case C
      record_stream_any_impl(buffer[pos], *accum_stream);
    }
    // 2)
    c10::OptionalStreamGuard stream_guard{accum_stream};
    accumulate(buffer, pos, std::move(var));
    // 3)
    if (*opt_consumer_stream != *accum_stream) {
      // Only the consumer stream needs to wait for this event
      auto event = c10::Event{device_type};
      event.record(*accum_stream);
      ready_events[pos] = std::move(event);
    }
    ready_streams[pos] = accum_stream;
  }
}

auto InputBuffer::variables(InputBuffer&& g) -> std::vector<Variable> {
  std::vector<Variable> result = std::move(g.buffer);
  return result;
}

} // namespace torch::autograd
