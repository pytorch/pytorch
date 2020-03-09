#include <torch/csrc/autograd/input_buffer.h>

#include <c10/core/DeviceGuard.h>
#include <c10/core/StreamGuard.h>
#include <c10/core/Event.h>
#include <c10/util/Optional.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace torch { namespace autograd {

  static void accumulate(std::vector<Variable>& buffer,
                         const size_t pos,
                         Variable&& var) {
    TORCH_INTERNAL_ASSERT(pos < buffer.size());
    auto& old_var = buffer[pos];
    // ATen doesn't route sparse additions correctly...
    // do dense + sparse in-place if possible
    if (old_var.is_sparse()) {
      //storage use_count is a big hammer, but for anything lighter there's an adversarial example with unexpected inplace modification
      if (!var.is_sparse() && var.is_contiguous() && var.storage().use_count() == 1) {
          buffer[pos] = var.add_(old_var);
      } else {
          buffer[pos] = var + old_var;
      }
    } else {
      if (var.is_sparse() && !old_var.is_sparse() && old_var.is_contiguous() && old_var.storage().use_count() == 1) {
          buffer[pos] = old_var.add_(var);
      } else {
          buffer[pos] = old_var + var;
      }
    }
  }

  void InputBuffer::add(size_t pos,
                        Variable&& var,
                        const c10::optional<c10::Stream>& opt_producer_stream,
                        const c10::optional<c10::Stream>& opt_consumer_stream) {
  TORCH_INTERNAL_ASSERT(pos < buffer.size());
  if (!var.defined()) {
    return;
  }

  // Switches to accumulate device
  // The device (and stream) chosen for accumulation is:
  //  (1) var is not a CUDA variable. Accumulation happens on var's device.
  //  (2) var is a CUDA variable and it, the consumer, and the producer share the same device:
  //       (2a) Uses the consumer's stream as the accumulation stream
  //       (2b) Syncs the accumulation stream with the producer's stream (if different)
  //       (2c) Accumulates.
  //  (3) var is a CUDA variable and it shares a device with the consumer but not the producer:
  //       (3a) Uses the consumer's stream as the accumulation stream
  //       (3b) Syncs the accumulation stream with the consumer device's default stream
  //       (3c) Accumulates.
  //  (4) var is a CUDA variable and it shares a device with the producer but not the consumer:
  //       (4a) Uses the producer device's default stream as the accumulation stream
  //       (4b) Syncs the accumulation stream with the the producer's stream
  //       (4c) Accumulates.
  //  (5) var is a CUDA variable and it does not share a device with the consumer or producer.
  //      Accumulation happens on the var device's default stream.

  TORCH_INTERNAL_ASSERT(device_of(var));
  c10::optional<c10::Stream> opt_accumulate_stream = c10::nullopt;
  if (device_of(var)->is_cuda()) {
    const auto on_producer = opt_producer_stream
                        && device_of(var) == opt_producer_stream->device();
    const auto on_consumer = opt_consumer_stream
                        && device_of(var) == opt_consumer_stream->device();
    if (on_producer && on_consumer) {
      // (2a)
      opt_accumulate_stream = opt_consumer_stream;
      if (opt_accumulate_stream != opt_producer_stream) {
        // (2b)
        auto event = c10::Event{c10::DeviceType::CUDA};
        event.record(*opt_producer_stream);
        opt_accumulate_stream->wait(event);
      }
    } else {
      c10::optional<c10::Stream> opt_sync_stream = c10::nullopt;
      const auto guard = c10::impl::VirtualGuardImpl{c10::DeviceType::CUDA};
      if (on_consumer && !on_producer) {
        // (3a)
        opt_accumulate_stream = opt_consumer_stream;
        opt_sync_stream = guard.getDefaultStream(opt_consumer_stream->device());
        TORCH_INTERNAL_ASSERT(opt_sync_stream == guard.getStream(*device_of(var)));
      } else if (on_producer && !on_consumer) {
        // (4a)
        opt_accumulate_stream = guard.getDefaultStream(opt_producer_stream->device());
        opt_sync_stream = opt_producer_stream;
      } else {
        // (5)
        opt_accumulate_stream = guard.getDefaultStream(*device_of(var));
      }
      if (opt_sync_stream && (opt_accumulate_stream != opt_sync_stream)) {
        // (3b), (4b)
        c10::OptionalDeviceGuard device_guard{opt_sync_stream->device()};
        auto event = c10::Event{c10::DeviceType::CUDA};
        event.record(*opt_sync_stream);
        opt_accumulate_stream->wait(event);
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
      // (1) non-CUDA variable
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

}}  // namespace torch::autograd
