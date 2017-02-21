#include "torch/csrc/autograd/grad_buffer.h"

#ifdef WITH_CUDA
#include "torch/csrc/cuda/AutoGPU.h"
#endif

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
#ifdef WITH_CUDA
    THCPAutoGPU auto_gpu(tensor->getDevice());
#endif
    if (item.second) {
      item.first.reset(item.first->clone());
      item.second = false;
    }
    item.first->cadd(*item.first, *tensor);
  }
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
