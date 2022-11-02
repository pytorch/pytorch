#pragma once

// ${generated_comment}

#ifdef TORCH_ASSERT_NO_OPERATORS
#error This change adds a dependency on native_functions.yaml,            \
  meaning the file will need to be re-compiled every time an operator     \
  is changed or added. Consider if your change would be better placed in  \
  another file, or if a more specific header might achieve the same goal. \
  See NOTE: [Tensor vs. TensorBase]
#endif

#if defined(AT_PER_OPERATOR_HEADERS) && defined(TORCH_ASSERT_ONLY_METHOD_OPERATORS)
#error This change adds a dependency on all pytorch operators, meaning the      \
  file will need to be re-compiled every time an operator is changed or added.  \
  Consider including a specific operator from <ATen/ops/{my_operator}_native.h> \
  and see NOTE [TORCH_ASSERT_ONLY_METHOD_OPERATORS].
#endif

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <ATen/Functions.h>
#include <tuple>
#include <vector>

${NativeFunctions_includes}

${NativeFunctions_declarations}

namespace at {
  namespace native {
    struct TORCH_API structured_mul_out : public at::meta::structured_mul_Tensor {
    void impl(const at::Tensor & self, const at::Tensor & other, const at::Tensor & out);
    };

    inline TORCH_API at::Tensor mul(const at::Tensor & self, const at::Tensor & other) {
      return self.mul(other);
    }

    inline TORCH_API at::Tensor & mul_(at::Tensor& self, const at::Tensor & other) {
      self.mul_(other);
      return self;
    }

    inline TORCH_API at::Tensor & mul_out(at::Tensor const& self, at::Tensor const& other, at::Tensor& out) {
      return at::mul_out(out, self, other);
    }

  } // namespace native
} // namespace at