#include <torch/extension.h>

struct Doubler {
  Doubler(int A, int B) {
    tensor_ =
        torch::ones({A, B}, torch::dtype(torch::kDouble).requires_grad(true));
  }
  at::Tensor forward() {
    return tensor_ * 2;
  }
  at::Tensor get() const {
    return tensor_;
  }

 private:
  at::Tensor tensor_;
};
