#include <torch/torch.h>

struct Doubler {
  Doubler(int A, int B) {
     tensor_ = at::ones({A, B}, at::CPU(at::kDouble));
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
