#include <ATen/ATen.h>

namespace at { namespace native {

Tensor one_hot(const Tensor &self, int64_t num_classes, c10::optional<ScalarType> dtype) {
    TORCH_CHECK(self.dtype() == kLong, "one_hot is only applicable to index tensor.");
    auto shape = self.sizes().vec();
    auto output_dtype = dtype.value_or(kLong);
    auto output_options = self.options().dtype(output_dtype);

    // empty tensor could be converted to one hot representation,
    // but shape inference is not possible.
    if (self.numel() == 0) {
        if (num_classes <= 0) {
            AT_ERROR("Can not infer total number of classes from empty tensor.");
        } else {
            shape.push_back(num_classes);
            return at::empty(shape, output_options);
        }
    }

    // non-empty tensor
    TORCH_CHECK(self.min().item().toLong() >= 0, "Class values must be non-negative.");
    if (num_classes == -1) {
        num_classes = self.max().item().toLong() + 1;
    } else {
        TORCH_CHECK(num_classes > self.max().item().toLong(), "Class values must be smaller than num_classes.");
    }

    shape.push_back(num_classes);
    Tensor ret = at::zeros(shape, output_options);
    ret.scatter_(-1, self.unsqueeze(-1), 1);
    return ret;
}

} // namespace native
} // namespace at
