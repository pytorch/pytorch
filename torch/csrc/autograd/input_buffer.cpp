#include "torch/csrc/autograd/input_buffer.h"

#include "torch/csrc/assertions.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/utils/auto_gpu.h"

namespace torch { namespace autograd {

InputBuffer::InputBuffer(size_t size)
  : buffer(size)
  {}

void InputBuffer::add(size_t pos, Variable var) {
  TORCH_ASSERT(pos >= 0 && pos < buffer.size());
  if (!var.defined()) {
    return;
  }
  auto& item = buffer[pos];
  if (!item.first.defined()) {
    auto current_version = var.current_version();
    buffer[pos] = std::make_pair<>(std::move(var), current_version);
  } else {
    auto result = apply_fn<Add>()(item.first, std::move(var));
    buffer[pos] = std::make_pair<>(std::move(result), 0);
  }
}

auto InputBuffer::device() const -> int {
  for (auto& pair : buffer) {
    if (pair.first.defined() && pair.first.type().isCuda()) {
      return pair.first.get_device();
    }
  }
  return -1;
}

auto InputBuffer::variables(InputBuffer&& g) -> std::vector<Variable> {
  InputBuffer _buffer = std::move(g);
  auto& buffer = _buffer.buffer;
  int size = buffer.size();
  std::vector<Variable> result;
  result.reserve(size);
  for (int i = 0; i != size; ++i) {
    result.emplace_back(buffer[i].first);
  }
  return result;
}

}}  // namespace torch::autograd
