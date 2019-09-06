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
                        c10::optional<c10::Event>& opt_event,
                        const c10::Stream& consumer_stream) {
  AT_ASSERT(pos < buffer.size());
  if (!var.defined()) {
    return;
  }
    
  // Sets consumer stream if CUDA
  if (consumer_stream.device_type() == c10::DeviceType::CUDA) {
    const auto guard = c10::impl::VirtualGuardImpl{c10::DeviceType::CUDA};
    guard.exchangeStream(consumer_stream);
    
    // Syncs (optional) producer stream with consumer stream if necessary
    if (opt_producer_stream && consumer_stream != *opt_producer_stream) {
      opt_event->recordOnce(*opt_producer_stream);
      consumer_stream.wait(*opt_event);
    }
  }

  auto& old_var = buffer[pos];
  if (!old_var.defined()) {
    buffer[pos] = std::move(var);
  } else {
    at::OptionalDeviceGuard device_guard(device_of(var));

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
