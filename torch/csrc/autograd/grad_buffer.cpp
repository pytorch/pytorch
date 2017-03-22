#include "torch/csrc/autograd/grad_buffer.h"

#include "torch/csrc/utils/auto_gpu.h"

namespace torch { namespace autograd {

GradBuffer::GradBuffer(size_t size)
  : buffer(size)
  {}

auto GradBuffer::addGrad(size_t pos, std::shared_ptr<Variable>&& var) -> void {
  auto& item = buffer[pos];
  if (!var) {
    return;
  }
  auto& tensor = var->data;
  if (!item.first) {
    buffer[pos] = std::make_pair<>(std::move(tensor), true);
  } else {
    AutoGPU auto_gpu(item.first->getDevice());
    if (item.first->isSparse() && !tensor->isSparse()) {
      auto* sum = tensor->clone();
      sum->cadd(*sum, *item.first);
      item.first.reset(sum);
    } else {
      if (item.second) {
        item.first.reset(item.first->clone());
      }
      item.first->cadd(*item.first, *tensor);
    }
    item.second = false;
  }
}

auto GradBuffer::device() const -> int {
  for (auto& pair : buffer) {
    if (pair.first) {
      return pair.first->getDevice();
    }
  }
  return -1;
}

auto GradBuffer::variables(GradBuffer&& g) -> std::vector<std::shared_ptr<Variable>> {
  auto buffer = std::move(g.buffer);
  int size = buffer.size();
  std::vector<std::shared_ptr<Variable>> result(size);
  for (int i = 0; i != size; ++i) {
    if (buffer[i].first) {
      result[i] = std::make_shared<Variable>(
          std::move(buffer[i].first), false, true);
    }
  }
  return result;
}

}}  // namespace torch::autograd
