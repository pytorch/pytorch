#include <torch/torch.h>

using namespace at;

Tensor exp_add(Tensor x, Tensor y) {
  return x.exp() + y.exp();
}
