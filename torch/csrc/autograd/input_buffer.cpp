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
  //  (1) If var is not a CUDA variable, accumulation happens on var's device.
  //  (2) If var is a CUDA variable, and opt_producer_stream and opt_consumer_stream
  //      share its device:
  //        (2a) if opt_producer_stream != opt_consumer_stream,
  //             sync opt_consumer_stream with opt_producer_stream.
  //        (2b) accumulation happens on opt_consumer_stream.
  //  (3) If var is a CUDA variable, and it's on opt_consumer_stream's device but not
  //      opt_producer_stream's device:
  //        (3a) Assume var was populated on its device's current stream.  In this sense,
  //             var's device's current stream is the "effective producer stream."
  //             The stream guard in Engine::evaluate_function (which calls InputBuffer::add)
  //             has set opt_producer_stream on opt_producer_stream's device, but that didn't
  //             affect var's device.  Therefore, we further assume (and assert) that var's
  //             device's current stream is its default stream.  Putting the pieces together,
  //             this tells us var's device's default stream is the "effective producer stream."
  //             If this "effective producer stream" != opt_consumer_stream,
  //             sync opt_consumer_stream with effective producer stream.
  //        (3b) accumulation happens on opt_consumer_stream.
  //  (4) If var is a CUDA variable, and it's on opt_producer_stream's device but not
  //      opt_consumer_stream's device:
  //        (4a) Assume var was populated on opt_producer_stream.
  //             During the consumer op's evaluation, opt_consumer_stream will be made current
  //             on the consumer device.  But this won't affect var's device, so at that time
  //             the current stream on var's device should be its default stream.
  //             In general, ops that collect grads from other devices (like BroadcastBackward's
  //             gather) should internally sync with the current streams of those other devices
  //             before using the gradients.  They would be crazy not to, right?
  //             In other words, we anticipate the consumer op will sync with the default stream
  //             of var's device before using var.  var's device's default stream is the
  //             "effective consumer stream" in the sense that it's the stream the consumer op
  //             will internally sync on during evaluation.
  //             If opt_producer_stream != var's device's default stream (aka the producer device's
  //             default stream), sync var's device's default stream with opt_producer_stream.
  //        (4b) accumulation happens on var's device's default stream (the "effective consumer stream").
  //  (5) If var is on neither opt_producer_stream nor opt_consumer_stream's device(s),
  //      throw an error (assume this never happens).

  TORCH_INTERNAL_ASSERT(device_of(var));
  c10::optional<c10::Stream> opt_accumulate_stream = c10::nullopt;
  if (device_of(var)->is_cuda()) {
    const auto on_producer = opt_producer_stream
                        && device_of(var) == opt_producer_stream->device();
    const auto on_consumer = opt_consumer_stream
                        && device_of(var) == opt_consumer_stream->device();
    if (on_producer && on_consumer) {
      // (2) CUDA variable with producer and consumer sharing a device
      // Accumulation happens on opt_consumer_stream
      opt_accumulate_stream = opt_consumer_stream;
      if (opt_accumulate_stream != opt_producer_stream) {
        // (2a) Sync accumulate with producer
        auto event = c10::Event{c10::DeviceType::CUDA};
        event.record(*opt_producer_stream);
        opt_accumulate_stream->wait(event);
      }
    } else if (on_consumer && !on_producer) {
      // (3) CUDA variable, on consumer stream's device but not on producer stream's device
      // Accumulation happens on opt_consumer_stream
      opt_accumulate_stream = opt_consumer_stream;
      // var's device's default stream is the "effective producer"
      const auto guard = c10::impl::VirtualGuardImpl{c10::DeviceType::CUDA};
      const auto default_stream = guard.getDefaultStream(*device_of(var));
      // double check our belief that var's device's current stream is its default stream
      TORCH_INTERNAL_ASSERT(guard.getStream(*device_of(var)) == default_stream);
      if (opt_accumulate_stream != default_stream) {
        // (3a) Sync accumulate with default_stream (the "effective producer")
        //      For this purpose, we'd like to record an event in that default stream.
        //      The cuda api says we must create that event on the same device as the stream.
        //      However, calling code has made the producer stream (and device) current,
        //      so we temporarily guard onto default_stream's device.
        c10::OptionalDeviceGuard device_guard{default_stream->device()};
        auto event = c10::Event{c10::DeviceType::CUDA};
        event.record(*default_stream);
        opt_accumulate_stream->wait(event);
      }
    } else if (on_producer && !on_consumer) {
      // (4) CUDA variable, on producer stream's device but not consumer stream's device
      // Accumulation happens on var's device's default stream
      const auto guard = c10::impl::VirtualGuardImpl{c10::DeviceType::CUDA};
      const auto default_stream = guard.getDefaultStream(*device_of(var));
      opt_accumulate_stream = default_stream;
      if (opt_accumulate_stream != opt_producer_stream) {
        // (4a) Sync accumulate with producer
        auto event = c10::Event{c10::DeviceType::CUDA};
        event.record(*opt_producer_stream);
        opt_accumulate_stream->wait(event);
      }
    } else {
      // (5) CUDA variable on neither producer stream nor consumer stream's device(s)
      AT_ERROR("Gradient (var) is on an unexpected device.");
    }
  }
      c10::OptionalStreamGuard stream_guard{opt_accumulate_stream};

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
