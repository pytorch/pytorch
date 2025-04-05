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
void record_stream_any_impl(Variable& var, c10::Stream& stream) {
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

void sync_streams(
    std::optional<c10::Stream>& stream_be_synced,
    std::optional<c10::Stream>& sync_stream,
    Variable& var,
    const c10::DeviceType device_type) {
  TORCH_INTERNAL_ASSERT(sync_stream.has_value());
  if (!stream_be_synced) {
    return;
  }
  c10::OptionalDeviceGuard device_guard{stream_be_synced->device()};
  auto event = c10::Event{device_type};
  event.record(*stream_be_synced);
  sync_stream->wait(event);
  record_stream_any_impl(var, *sync_stream);
}

} // anonymous namespace

static void accumulate(
    std::vector<Variable>& buffer,
    const size_t pos,
    Variable& var) {
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

static void execute_accumulation(
    std::vector<Variable>& buffer,
    size_t pos,
    Variable& var,
    const std::optional<c10::Stream>& accumulate_stream) {
  TORCH_INTERNAL_ASSERT(accumulate_stream.has_value());
  if (buffer[pos].defined()) {
    c10::OptionalStreamGuard stream_guard{accumulate_stream};
    accumulate(buffer, pos, var);
  } else {
    buffer[pos] = var;
  }
}

void InputBuffer::add(
    size_t pos,
    Variable&& var,
    const std::optional<c10::Stream>& opt_producer_stream,
    const std::optional<c10::Stream>& opt_consumer_stream) {
  TORCH_INTERNAL_ASSERT(pos < buffer.size());
  if (!var.defined()) {
    return;
  }

  // Switches to accumulate device
  // The device (and stream) chosen for accumulation is:
  //  (1) var is not an accelerator variable. Accumulation happens on
  //  var's device. (2) var is an accelerator variable and it, the
  //  consumer, and the producer share the same device:
  //       (2a) Uses the consumer's stream as the accumulation stream
  //       (2b) Syncs the accumulation stream with the producer's stream (if
  //       different) (2c) Accumulates.
  //  (3) var is an accelerator variable and it shares a device with the
  //  consumer but not the producer:
  //       (3a) Uses the consumer's stream as the accumulation stream
  //       (3b) Syncs the accumulation stream with the producer's stream if it
  //       is not null (3c) Accumulates.
  //  (4) var is an accelerator variable and it shares a device with the
  //  producer but not the consumer:
  //       (4a) Uses the producer stream as the accumulation stream
  //       (4b) Accumulates
  //       (4c) Syncs the consumer's stream with the accumulation stream if it
  //       is not null.
  //  (5) var is an accelerator variable and it does not share a device
  //  with the consumer or producer.
  //      Accumulation happens on the var device's stream.
  //      (5a) Syncs the accumulation stream with the producer's stream
  //      (5b) Accumulates
  //      (5c) Syncs the consumer's stream with the accumulation stream

  auto const device = device_of(var);
  TORCH_INTERNAL_ASSERT(device.has_value());
  std::optional<c10::Stream> opt_accumulate_stream = std::nullopt;
  const auto device_type = device->type();
  if (at::isAccelerator(device_type)) {
    const auto on_producer =
        opt_producer_stream && device == opt_producer_stream->device();
    const auto on_consumer =
        opt_consumer_stream && device == opt_consumer_stream->device();

    if (on_producer && on_consumer) {
      // (2a)
      opt_accumulate_stream = opt_consumer_stream;
      if (opt_accumulate_stream != opt_producer_stream) {
        // (2b)
        auto event = c10::Event{device_type};
        event.record(*opt_producer_stream);
        opt_accumulate_stream->wait(event);
        record_stream_any_impl(var, *opt_accumulate_stream);
      }
    } else {
      std::optional<c10::Stream> opt_sync_stream = std::nullopt;
      if (on_consumer && !on_producer) {
        // (3a)
        opt_accumulate_stream = opt_consumer_stream;
        opt_sync_stream = opt_producer_stream;
        // (3b)
        sync_streams(opt_sync_stream, opt_accumulate_stream, var, device_type);
        // (3c)
        execute_accumulation(buffer, pos, var, opt_accumulate_stream);
        return;
      } else if (on_producer && !on_consumer) {
        // (4a)
        opt_accumulate_stream = opt_producer_stream;
        opt_sync_stream = opt_consumer_stream;
        // (4b)
        execute_accumulation(buffer, pos, var, opt_accumulate_stream);
        // (4c)
        sync_streams(opt_sync_stream, opt_accumulate_stream, var, device_type);
        return;
      } else {
        opt_accumulate_stream = getStreamForDeviceIdx(device.value());
        opt_sync_stream = opt_producer_stream;
        // (5a)
        sync_streams(opt_sync_stream, opt_accumulate_stream, var, device_type);
        // (5b)
        execute_accumulation(buffer, pos, var, opt_accumulate_stream);
        // (5c)
        opt_sync_stream = opt_accumulate_stream;
        opt_accumulate_stream = opt_consumer_stream;
        sync_streams(opt_sync_stream, opt_accumulate_stream, var, device_type);
        return;
      }
    }
  }

  auto& old_var = buffer[pos];
  if (!old_var.defined()) {
    buffer[pos] = var;
  } else {
    if (opt_accumulate_stream) {
      c10::OptionalStreamGuard stream_guard{opt_accumulate_stream};
      accumulate(buffer, pos, var);
    } else {
      // (1) non-accelerator variable
      //     Accumulation happens on variable's device
      c10::OptionalDeviceGuard device_guard{device};
      accumulate(buffer, pos, var);
    }
  }
}

auto InputBuffer::variables(InputBuffer&& g) -> std::vector<Variable> {
  std::vector<Variable> result = std::move(g.buffer);
  return result;
}

c10::Stream InputBuffer::getStreamForDeviceIdx(c10::Device device) {
  auto it = device_streams.find(device.index());
  if (it == device_streams.end()) {
    // Lazily initialize the stream for this device index
    const auto guard = c10::impl::VirtualGuardImpl{device.type()};
    c10::Stream stream = guard.getNewStream(device);
    device_streams.emplace(device.index(), stream);
    return stream;
  }
  return it->second;
}

} // namespace torch::autograd
