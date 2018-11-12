#include "torch/csrc/autograd/input_buffer.h"

#include "torch/csrc/autograd/functions/basic_ops.h"

#include <ATen/DeviceGuard.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace torch { namespace autograd {


void InputBuffer::add(size_t pos, Variable var) {
  AT_ASSERT(pos < buffer.size());
  if (!var.defined()) {
    return;
  }
  auto& old_var = buffer[pos];
  if (!old_var.defined()) {
    buffer[pos] = std::move(var);
  } else {
    at::OptionalDeviceGuard device_guard(device_of(var));
    // ATen doesn't route sparse additions correctly...
    if (old_var.is_sparse()) {
      buffer[pos] = var + old_var;
    } else {
      buffer[pos] = old_var + var;
    }
  }
}

auto InputBuffer::device() const -> int {
  for (auto& var : buffer) {
    if (var.defined() && var.is_cuda()) {
      return var.get_device();
    }
  }
  return -1;
}

auto InputBuffer::variables(InputBuffer&& g) -> std::vector<Variable> {
  std::vector<Variable> result = std::move(g.buffer);
  return result;
}

}}  // namespace torch::autograd
