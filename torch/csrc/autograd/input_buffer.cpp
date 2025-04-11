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
void record_stream_any_impl(Variable& var, const c10::Stream& stream) {
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

static inline void _wait_stream(
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

void InputBuffer::add(
    size_t pos,
    Variable&& var,
    const std::optional<c10::Stream>& opt_producer_stream,
    const std::optional<c10::Stream>& opt_consumer_stream,
    int num_dependencies) {
  TORCH_INTERNAL_ASSERT(pos < buffer.size());

  if (!var.defined()) {
    return;
  }
  int current_index = accum_counts[pos]++;
  auto const device = device_of(var);
  TORCH_INTERNAL_ASSERT(device.has_value());
  const auto device_type = device->type();
  bool is_accelerator =
      device->is_cuda() || device->is_mtia() || device->is_privateuseone();

  // [ Single producer ]
  // If there's only a single producer, there is no accumulation involved.
  // All we need to do is (if streams are involved at all):
  // - Have the consumer canonical stream wait for the producer canonical stream
  // - Move var into the buffer.
  if (num_dependencies == 1) {
    if (is_accelerator) {
      _wait_stream(opt_consumer_stream, opt_producer_stream, device_type);
      record_stream_any_impl(var, *opt_consumer_stream);
    }
    TORCH_INTERNAL_ASSERT(current_index == 0);
    TORCH_INTERNAL_ASSERT(!buffer[pos].defined())
    buffer[pos] = std::move(var);
    return;
  }
  // [ First producer ]
  // In the multiple producer case, we need to handle accumulation.
  // When we see the very first producer, we do a couple things:
  // - Determine the accumulation stream:
  //    case 1)   non-accelerator             -> n/a
  //    case 2/3) var device matches consumer -> consumer stream
  //    case 4)   var device matches producer -> producer stream
  //    case 5)   var device matches neither  -> var device's current stream
  // - Move var into the buffer.
  // - In the 2/3 case we also stash the producer stream because the
  //   accumulation stream would need to sync with it later.
  //   See test_side_stream_backward_overlap
  if (current_index == 0) {
    if (is_accelerator) {
      // Try assuming that these devices all have non-opt streams
      TORCH_INTERNAL_ASSERT(opt_consumer_stream && opt_producer_stream)
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      if (opt_consumer_stream->device() == *device) {
        // Case 2/3
        opt_accum_streams[pos] = opt_consumer_stream;
        opt_first_producer_streams[pos] = opt_producer_stream;
      } else {
        // Case 4/5
        if (opt_producer_stream->device() == *device) {
          opt_accum_streams[pos] = opt_producer_stream;
        } else {
          c10::impl::VirtualGuardImpl guard(device_type);
          opt_accum_streams[pos] = guard.getDefaultStream(*device);
        }
      }
    }
    TORCH_INTERNAL_ASSERT(!buffer[pos].defined())
    buffer[pos] = std::move(var);
    return;
  }
  // [ Nth producer ]
  // At this point, the accumulation stream has been determined. For both 2/3
  // and 4/5, everytime subsequent producers call InputBuffer.add:
  // - The accumulation stream (determined at n=1) waits for the new
  //   producer stream
  // - Call var.record_stream(accumution_stream)
  // - Accumulate var into buffer
  //
  // Also, depending on whether you are in case 2/3 or 4/5, you must do
  // something extra:
  // - If you are in the 2/3 regime, you skipped synchronizing the first
  //   producer with the consumer during the first iteration. We make this
  //   up in the second iteration.
  // - If you are in the 4/5 regime, all you've done so far is synchronize
  //   between producers. You still need to have the consumer wait for the
  //   producer.
  auto accum_stream = opt_accum_streams[pos];
  if (is_accelerator) {
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    if (opt_consumer_stream->device() == *device) {
      // Case 2/3
      if (current_index == 1) {
        // 2/3 extra logic
        _wait_stream(
            accum_stream, opt_first_producer_streams[pos], device_type);
      }
      _wait_stream(accum_stream, opt_producer_stream, device_type);
      record_stream_any_impl(var, *accum_stream);
      c10::OptionalStreamGuard stream_guard{accum_stream};
      accumulate(buffer, pos, std::move(var));
    } else {
      // Case 4/5
      _wait_stream(accum_stream, opt_producer_stream, device_type);
      record_stream_any_impl(var, *accum_stream);
      {
        c10::OptionalStreamGuard stream_guard{accum_stream};
        accumulate(buffer, pos, std::move(var));
      }
      if (current_index == num_dependencies - 1) {
        // 4/5 case extra logic
        _wait_stream(opt_consumer_stream, accum_stream, device_type);
      }
    }
  } else {
    // Case 1
    TORCH_INTERNAL_ASSERT(!accum_stream)
    c10::OptionalDeviceGuard device_guard{device};
    accumulate(buffer, pos, std::move(var));
  }
}

auto InputBuffer::variables(InputBuffer&& g) -> std::vector<Variable> {
  std::vector<Variable> result = std::move(g.buffer);
  return result;
}

} // namespace torch::autograd
