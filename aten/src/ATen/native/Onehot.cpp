#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/one_hot_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace at { namespace native {

Tensor one_hot(const Tensor &self, int64_t num_classes) {
    TORCH_CHECK(self.dtype() == kLong, "one_hot is only applicable to index tensor.");
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

    // using meta bit test to catch Fake Tensor as well until __torch__function defined
    if (self.key_set().has_all(DispatchKeySet(BackendComponent::MetaBit))) {
        AT_ERROR("Can not infer total number of classes from meta tensor.");
    }

    // non-empty tensor
    if (self.device().type() != at::kCUDA && self.device().type() != at::kMPS) {
      //for cuda, rely on device assert thrown by scatter
      TORCH_CHECK(self.min().item().toLong() >= 0, "Class values must be non-negative.");
    }
    if (num_classes == -1) {
        num_classes = self.max().item().toLong() + 1;
    } else {
        if (self.device().type() != at::kCUDA && self.device().type() != at::kMPS) {
          //rely on device asserts from scatter to avoid sync here
          TORCH_CHECK(num_classes > self.max().item().toLong(), "Class values must be smaller than num_classes.");
        } else {
            //for cuda, assert that num_classes is at least 1
            TORCH_CHECK(num_classes >= 1, "num_classes should be positive");
        }
    }

    shape.push_back(num_classes);
    Tensor ret = at::zeros(shape, self.options());
    ret.scatter_(-1, self.unsqueeze(-1), 1);
    return ret;
}

} // namespace native
} // namespace at
