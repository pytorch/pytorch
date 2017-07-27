#include "torch/csrc/autograd/input_buffer.h"

#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/utils/auto_gpu.h"

namespace torch { namespace autograd {

InputBuffer::InputBuffer(size_t size)
  : buffer(size)
  {}

void InputBuffer::add(size_t pos, std::shared_ptr<Variable>&& var) {
  if (!var) {
    return;
  }
  auto& item = buffer[pos];
  auto& saved_var_ptr = item.first;
  if (!saved_var_ptr) {
    auto version = **var->version_counter;
    buffer[pos] = std::make_pair<>(std::move(var), version);
  } else {
    variable_list result = Add().apply({item.first, var});
    buffer[pos] = std::make_pair<>(std::move(result[0]), 0);
  }
}

auto InputBuffer::device() const -> int {
  for (auto& pair : buffer) {
    if (pair.first && pair.first->data.type().isCuda()) {
      return pair.first->data.get_device();
    }
  }
  return -1;
}

auto InputBuffer::variables(InputBuffer&& g) -> std::vector<std::shared_ptr<Variable>> {
  InputBuffer _buffer = std::move(g);
  auto& buffer = _buffer.buffer;
  int size = buffer.size();
  std::vector<std::shared_ptr<Variable>> result;
  result.reserve(size);
  for (int i = 0; i != size; ++i) {
    auto var_ptr = buffer[i].first;
    result.emplace_back(var_ptr ? std::move(buffer[i].first) : nullptr);
  }
  return result;
}

}}  // namespace torch::autograd
