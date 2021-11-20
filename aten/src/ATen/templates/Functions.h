#pragma once

// ${generated_comment}

#ifdef TORCH_ASSERT_NO_OPERATORS
#error This change adds a dependency on native_functions.yaml,            \
  meaning the file will need to be re-compiled every time an operator     \
  is changed or added. Consider if your change would be better placed in  \
  another file, or if a more specific header might achieve the same goal. \
  See NOTE: [Tensor vs. TensorBase]
#endif

#ifdef TORCH_ASSERT_ONLY_METHOD_OPERATORS
#error This change adds a dependency on all pytorch operators, meaning the     \
  file will need to be re-compiled every time an operator is changed or added. \
  Consider including a specific operator from <ATen/ops/{my_operator}.h>
#endif

#include <ATen/ops/from_blob.h>
#include <ATen/ops/tensor.h>

${Functions_includes}

namespace at {

// Special C++ only overloads for std()-like functions (See gh-40287)
// These are needed because int -> bool conversion takes precedence over int -> IntArrayRef
// So, for example std(0) would select the std(unbiased=False) overload
TORCH_API inline Tensor var(const Tensor& self, int dim) {
  return at::var(self, IntArrayRef{dim});
}
TORCH_API inline std::tuple<Tensor, Tensor> var_mean(const Tensor& self, int dim) {
  return at::var_mean(self, IntArrayRef{dim});
}
TORCH_API inline Tensor std(const Tensor& self, int dim) {
  return at::std(self, IntArrayRef{dim});
}
TORCH_API inline std::tuple<Tensor, Tensor> std_mean(const Tensor& self, int dim) {
  return at::std_mean(self, IntArrayRef{dim});
}

inline int64_t numel(const Tensor& tensor) {
  return tensor.numel();
}

inline int64_t size(const Tensor& tensor, int64_t dim) {
  return tensor.size(dim);
}

inline int64_t stride(const Tensor& tensor, int64_t dim) {
  return tensor.stride(dim);
}

inline bool is_complex(const Tensor& tensor) {
  return tensor.is_complex();
}

inline bool is_floating_point(const Tensor& tensor) {
  return tensor.is_floating_point();
}

inline bool is_signed(const Tensor& tensor) {
  return tensor.is_signed();
}

inline bool is_inference(const Tensor& tensor) {
  return tensor.is_inference();
}

inline bool is_conj(const Tensor& tensor) {
  return tensor.is_conj();
}

inline Tensor conj(const Tensor& tensor) {
  return tensor.conj();
}

inline bool is_neg(const Tensor& tensor) {
  return tensor.is_neg();
}

}
