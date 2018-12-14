#include <ATen/ATen.h>

namespace at { namespace native {

Tensor to_one_hot(Tensor self, int64_t num_classes) {
    AT_CHECK(self.min().item().toLong() >= 0, "Class values must be positive");
    if (num_classes <= 0) {
        num_classes = self.max().item().toLong();
    } else {
        AT_CHECK(self.max().item().toLong() <= num_classes, "Class values can not be larger than num_classes");
    }

    auto shape = self.sizes().vec();
    shape.push_back(num_classes);

    auto expanded_indices = self.unsqueeze(-1).expand(shape);
    Tensor ret = at::zeros_like(expanded_indices);
    ret.scatter_(-1, expanded_indices, 1);
    return ret;
}

} // namespace native
} // namespace at