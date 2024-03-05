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
#include <c10/util/Optional.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace torch {
namespace autograd {

namespace {
// look what you made me do >.<
// Divergent paths for per-Impl stream recording that leak implementation
// details of the impls should not be needed here.
// See https://github.com/pytorch/pytorch/issues/60306
// TODO: clean this up when https://github.com/pytorch/pytorch/issues/60306 is
// improved
void record_stream_any_impl(Variable& var, c10::Stream& stream) {
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

void InputBuffer::add(
    size_t pos,
    Variable&& var,
    const c10::optional<c10::Stream>& opt_producer_stream,
    const c10::optional<c10::Stream>& opt_consumer_stream) {
  TORCH_INTERNAL_ASSERT(pos < buffer.size());
  if (!var.defined()) {
    return;
  }

  // Switches to accumulate device
  // The device (and stream) chosen for accumulation is:
  //  (1) var is not a CUDA/privateuse1 variable. Accumulation happens on var's
  //  device. (2) var is a CUDA/privateuse1 variable and it, the consumer, and
  //  the producer share the same device:
  //       (2a) Uses the consumer's stream as the accumulation stream
  //       (2b) Syncs the accumulation stream with the producer's stream (if
  //       different) (2c) Accumulates.
  //  (3) var is a CUDA/privateuse1 variable and it shares a device with the
  //  consumer but not the producer:
  //       (3a) Uses the consumer's stream as the accumulation stream
  //       (3b) Syncs the accumulation stream with the consumer device's default
  //       stream (3c) Accumulates.
  //  (4) var is a CUDA/privateuse1 variable and it shares a device with the
  //  producer but not the consumer:
  //       (4a) Uses the producer device's default stream as the accumulation
  //       stream (4b) Syncs the accumulation stream with the producer's
  //       stream (4c) Accumulates.
  //  (5) var is a CUDA/privateuse1 variable and it does not share a device with
  //  the consumer or producer.
  //      Accumulation happens on the var device's default stream.

  TORCH_INTERNAL_ASSERT(device_of(var));
  c10::optional<c10::Stream> opt_accumulate_stream = c10::nullopt;
  const auto device_type = device_of(var).value().type();
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  if (device_of(var)->is_cuda() || device_of(var)->is_privateuseone()) {
    const auto on_producer =
        opt_producer_stream && device_of(var) == opt_producer_stream->device();
    const auto on_consumer =
        opt_consumer_stream && device_of(var) == opt_consumer_stream->device();

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
      c10::optional<c10::Stream> opt_sync_stream = c10::nullopt;
      const auto guard = c10::impl::VirtualGuardImpl{device_type};
      if (on_consumer && !on_producer) {
        // (3a)
        opt_accumulate_stream = opt_consumer_stream;
        opt_sync_stream = guard.getDefaultStream(opt_consumer_stream->device());
      } else if (on_producer && !on_consumer) {
        // (4a)
        opt_accumulate_stream =
            guard.getDefaultStream(opt_producer_stream->device());
        opt_sync_stream = opt_producer_stream;
      } else {
        // (5)
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        opt_accumulate_stream = guard.getDefaultStream(*device_of(var));
      }
      if (opt_sync_stream && (opt_accumulate_stream != opt_sync_stream)) {
        // (3b), (4b)
        c10::OptionalDeviceGuard device_guard{opt_sync_stream->device()};
        auto event = c10::Event{device_type};
        event.record(*opt_sync_stream);
        opt_accumulate_stream->wait(event);
        const auto guard = c10::impl::VirtualGuardImpl(device_type);
        record_stream_any_impl(var, *opt_accumulate_stream);
      }
    }
  }

  auto& old_var = buffer[pos];
  if (!old_var.defined()) {
    buffer[pos] = std::move(var);
  } else {
    if (opt_accumulate_stream) {
      c10::OptionalStreamGuard stream_guard{opt_accumulate_stream};
      accumulate(buffer, pos, std::move(var));
    } else {
      // (1) non-CUDA/privateuse1 variable
      //     Accumulation happens on variable's device
      c10::OptionalDeviceGuard device_guard{device_of(var)};
      accumulate(buffer, pos, std::move(var));
    }
  }
}

auto InputBuffer::device() const -> at::Device {
  // Since we pick the first non-CPU tensor, this won't work with
  // mixed device-type operations (e.g., an op that is both CUDA
  // and XLA).  This is *incredibly* unlikely, so we don't worry
  // about it.
  for (auto& var : buffer) {
    if (var.defined()) {
      auto device = var.device();
      if (device.type() != at::kCPU) {
        return device;
      }
    }
  }
  // Only report to the CPU thread if there really were no tensors
  // from other devices.
  return at::kCPU;
}

auto InputBuffer::variables(InputBuffer&& g) -> std::vector<Variable> {
  std::vector<Variable> result = std::move(g.buffer);
  return result;
}

} // namespace autograd
} // namespace torch
