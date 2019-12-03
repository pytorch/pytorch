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
  //  (1) If the variable is not a CUDA variable, accumulation happens on the
  //      device of the variable.
  //  (2) If the variable is a CUDA variable, and the producer and consumer
  //      share its device, then:
  //        (2a) if the producer and consumer do not share a stream,
  //             the consumer is synced with the producer.
  //        (2b) accumulation happens on the consumer's stream
  //  (3) If the variable is a CUDA variable but it, the producer, and the
  //      consumer are on multiple devices, then accumulation happens on
  //      the default stream of the variable's device.

  TORCH_INTERNAL_ASSERT(device_of(var));
  c10::optional<c10::Stream> opt_accumulate_stream = c10::nullopt;
  if (device_of(var)->is_cuda()) {
    const auto on_producer = opt_producer_stream
                        && device_of(var) == opt_producer_stream->device();
    const auto on_consumer = opt_consumer_stream
                        && device_of(var) == opt_consumer_stream->device();
    if (on_producer && on_consumer) {
      // (2) CUDA variable with producer and consumer sharing a device
      //     Accumulation happens on consumer's stream
      opt_accumulate_stream = opt_consumer_stream;
      if (opt_producer_stream != opt_consumer_stream) {
        // (2a) Syncs consumer with producer
        auto event = c10::Event{c10::DeviceType::CUDA};
        event.record(*opt_producer_stream);
        opt_consumer_stream->wait(event);
      }
    } else {
      // (3) CUDA variable with multiple devices
      //     Accumulation happens on variable's device's default stream
      const auto guard = c10::impl::VirtualGuardImpl{c10::DeviceType::CUDA};
      const auto default_stream = guard.getDefaultStream(*device_of(var));
      opt_accumulate_stream = default_stream;
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
