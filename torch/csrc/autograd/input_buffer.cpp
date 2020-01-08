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
  //  (2) If var is a CUDA variable, and the producer and consumer
  //      share its device, then:
  //        (2a) if the producer and consumer do not share a stream,
  //             the consumer is synced with the producer.
  //        (2b) accumulation happens on the consumer's stream
  //  (3) If var is a CUDA variable but it, the producer, and the
  //      consumer are on multiple devices, then:
  //        (2a) We assume var was created on its device's current stream.
  //             If that stream is different from the consumer stream,
  //             we sync the consumer with var's device's current stream.
  //        (3a) accumulation still happens on the consumer's stream

  TORCH_INTERNAL_ASSERT(device_of(var));
  c10::optional<c10::Stream> opt_accumulate_stream = c10::nullopt;
  if (device_of(var)->is_cuda()) {
    // Accumulation, if necessary, always happens on the consumer stream.
    opt_accumulate_stream = opt_consumer_stream;
    // Stream that we expect was used to populate var, with which the accumulate
    // stream must be synced.  opt_var_stream is different depending whether
    // we hit case (2) or case (3), as explained below.
    c10::optional<c10::Stream> opt_var_stream = c10::nullopt;

    const auto on_producer = opt_producer_stream
                        && device_of(var) == opt_producer_stream->device();
    const auto on_consumer = opt_consumer_stream
                        && device_of(var) == opt_consumer_stream->device();
    if (on_producer && on_consumer) {
      // (2) CUDA variable with producer and consumer sharing a device
      opt_var_stream = opt_producer_stream;
    } else {
      // (3) CUDA variable with multiple devices
      // The current stream for each device is thread local.  User-side calls that set streams affect the
      // current stream in the main thread.  However, backward is executed on new threads (one per device).
      // Therefore, we expect that the ambient current stream in backward-pass threads is the default stream,
      // unless it's been explicitly changed by something on that same thread (e.g. the stream guard in
      // Engine::evaluate_function).  For cross-device backward ops in particular, however, that stream guard
      // only sets the stream on the producer device (opt_producer_stream).  If the gradient (var) that the op
      // created is on a different device, that device wasn't affected by the stream guard.
      // Therefore, var's device's current stream should be the ambient current stream that the backward
      // thread holds for var's device, which should be that device's default stream.
      //
      // tldr: for case (3) we assume var was populated on its device's default stream.

      //  get_stream_helper is a hack to access streams that will compile even if the build is CPU-only.
      const auto get_stream_helper = c10::impl::VirtualGuardImpl{c10::DeviceType::CUDA};

      // double check our belief that the current stream on var's device is the default stream
      TORCH_INTERNAL_ASSERT(get_stream_helper.getStream(*device_of(var)) ==
                            get_stream_helper.getDefaultStream(*device_of(var)));
      opt_var_stream = get_stream_helper.getDefaultStream(*device_of(var));
    }

    if (opt_var_stream != opt_consumer_stream) {
      // Sync consumer with var stream
      // Events are associated with the current device on construction, and
      // "cudaEventRecord() will fail if the input event and input stream are associated to different devices"
      // so just in case, we guard onto the the var & consumer's device before creating the event.
      // TODO:  Maybe we should guard onto var's device for the entirety of InputBuffer::add?
      c10::DeviceGuard device_guard{opt_var_stream->device()};
      auto event = c10::Event{c10::DeviceType::CUDA};
      event.record(*opt_var_stream);
      opt_consumer_stream->wait(event);
      // TODO:  I think we need a
      // c10::cuda::CUDACachingAllocator::recordStream(var.storage().data_ptr(), opt_consumer_stream);
      // but I'm not sure if a call to c10::cuda::xyz here will allow a CPU-only build to compile.
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
