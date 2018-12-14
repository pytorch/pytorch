#include <ATen/ATen.h>

namespace at { namespace native {

Tensor to_one_hot(const Tensor &self, int64_t num_classes) {
    AT_CHECK(self.min().item().toLong() >= 0, "Class values must be positive");
    if (num_classes <= 0) {
        num_classes = self.max().item().toLong() + 1;
    } else {
        AT_CHECK(self.max().item().toLong() < num_classes, "Class values must be smaller than num_classes");
    }

    auto shape = self.sizes().vec();
    shape.push_back(num_classes);

    Tensor ret = at::zeros(shape, self.options());
    ret.scatter_(-1, self.unsqueeze(-1), 1);
    return ret;
}

} // namespace native
} // namespace at
