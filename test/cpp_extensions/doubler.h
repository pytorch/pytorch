#include <torch/torch.h>

struct Doubler {
  Doubler(int A, int B) {
    tensor_ = at::ones({A, B}, torch::CPU(at::kDouble));
    torch::set_requires_grad(tensor_, true);
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
