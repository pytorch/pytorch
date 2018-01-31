#include <torch/torch.h>

struct Doubler {
  Doubler(int A, int B) {
     tensor_ = at::CPU(at::kDouble).ones({A, B});
  }
  at::Tensor forward() {
    return tensor_ * 2;
  }

private:
  at::Tensor tensor_;
};
