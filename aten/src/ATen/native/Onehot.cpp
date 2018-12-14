#include <ATen/ATen.h>

namespace at { namespace native {

Tensor one_hot(const Tensor &self, int64_t num_classes) {
    auto shape = self.sizes().vec();

    // empty tensor could be converted to one hot representation,
    // but shape inference is not possible.
    if (self.numel() == 0) {
        if (num_classes <= 0) {
            AT_ERROR("Can not infer total number of classes from empty tensor.");
        } else {
            shape.push_back(num_classes);
            return at::empty(shape, self.options());
        }
    }

    // non-empty tensor
    if (num_classes <= 0) {
        num_classes = self.max().item().toLong() + 1;
    } else {
        AT_CHECK(self.max().item().toLong() < num_classes, "Class values must be smaller than num_classes.");
    }
    shape.push_back(num_classes);

    AT_CHECK(self.min().item().toLong() >= 0, "Class values must be non-negative.");

    Tensor ret = at::zeros(shape, self.options());
    ret.scatter_(-1, self.unsqueeze(-1), 1);
    return ret;
}

} // namespace native
} // namespace at
