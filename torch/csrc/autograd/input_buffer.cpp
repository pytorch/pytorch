#include <torch/csrc/autograd/input_buffer.h>

#include <c10/core/DeviceGuard.h>
#include <c10/core/StreamGuard.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace torch { namespace autograd {


  void InputBuffer::add(size_t pos,
                        Variable var,
                        const c10::optional<c10::Stream>& opt_producer_stream,
                        const c10::optional<c10::Stream>& opt_consumer_stream) {
  TORCH_INTERNAL_ASSERT(pos < buffer.size());
  if (!var.defined()) {
    return;
  }

  // Switches to accumulate stream
  // Note: streams are only used with CUDA variables.
  // The stream chosen for accumulation is decided as follows:
  //
  //  (1) If both the producer and consumer stream are CUDA streams on the
  //      same device, then the consumer is synced with the producer and
  //      accumulation happens on the consumer's stream.
  //  (2) If the producer is not a CUDA function the consumer is synced
  //      with its device's current stream and accumulation happens on the
  //      consumer's stream.
  //  (3) If the consumer is not a CUDA function the default stream
  //      on the producer's device is synced with the producer's stream
  //      and accumulation happens on the default stream.
  //  (4) If neither the producer nor consumer is a CUDA function then
  //      accumulation happens on the CPU. (This case actually
  //      occurs and is not an error.)
  //
  // The use of the default stream in (3) is because the producer has
  // set its own stream on the device. Another option would be to pass in
  // the device's previously current stream to this function.
  //
  TORCH_INTERNAL_ASSERT(device_of(var));
  c10::OptionalStreamGuard stream_guard{opt_consumer_stream};
  if (device_of(var)->is_cuda()) {
    const auto on_producer = opt_producer_stream
                        && device_of(var) == opt_producer_stream->device();
    const auto on_consumer = opt_consumer_stream
                        && device_of(var) == opt_consumer_stream->device();

    if (on_producer && on_consumer && opt_consumer_stream == opt_producer_stream) {
      // (1a) no synchronization necessary, accumulates on consumer
    } else if (on_producer && on_consumer) {
      // (1b) Syncs consumer with producer, accumulates on consumer
      auto event = c10::Event{c10::DeviceType::CUDA};
      event.record(*opt_producer_stream);
      opt_consumer_stream->wait(event);
    } else if (on_consumer) {
      // (2) Syncs consumer with current, accumulates on consumer
      const auto guard = c10::impl::VirtualGuardImpl{c10::DeviceType::CUDA};
      const auto current_stream = guard.getStream(opt_consumer_stream->device());
      auto event = c10::Event{c10::DeviceType::CUDA};
      event.record(current_stream);
      opt_consumer_stream->wait(event);
    } else if (on_producer) {
      // (3) Syncs default with producer, accumulates on default
      const auto guard = c10::impl::VirtualGuardImpl{c10::DeviceType::CUDA};
      const auto default_stream = guard.getDefaultStream(opt_producer_stream->device());
      auto event = c10::Event{c10::DeviceType::CUDA};
      event.record(*opt_producer_stream);
      default_stream.wait(event);
      stream_guard.reset_stream(default_stream);
    } else {
      // (4) accumulates on cpu
    }
  }

  auto& old_var = buffer[pos];
  if (!old_var.defined()) {
    buffer[pos] = std::move(var);
  } else {
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
