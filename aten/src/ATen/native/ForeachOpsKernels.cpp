#include <ATen/ATen.h>
#include <ATen/native/ForeachUtils.h>

namespace at { namespace native {

#define FOREACH_BINARY_OP_SCALAR(NAME)                                                                    \
void foreach_tensor_##NAME##_scalar_kernel_slow_(TensorList tensors, Scalar scalar) {                     \
  check_foreach_api_restrictions(tensors);                                                                \
                                                                                                          \
  for (auto& t: tensors) {                                                                                \
    t.NAME##_(scalar);                                                                                    \
  }                                                                                                       \
}                                                                                                         \
                                                                                                          \
std::vector<Tensor> foreach_tensor_##NAME##_scalar_kernel_slow(TensorList tensors, Scalar scalar) {       \
  check_foreach_api_restrictions(tensors);                                                                \
                                                                                                          \
  std::vector<Tensor> result;                                                                             \
  result.reserve(tensors.size());                                                                         \
  for (const auto& t: tensors) {                                                                          \
    result.emplace_back(t.NAME(scalar));                                                                  \
  }                                                                                                       \
                                                                                                          \
  return result;                                                                                          \
}

#define FOREACH_BINARY_OP_SCALARLIST(NAME)                                                                    \
void foreach_tensor_##NAME##_scalarlist_kernel_slow_(TensorList tensors, ScalarList scalars) {                \
  check_foreach_api_restrictions(tensors);                                                                    \
                                                                                                              \
  for (int i = 0; i < tensors.size(); i++) {                                                                  \
      tensors[i].NAME##_(scalars[i]);                                                                         \
    }                                                                                                         \
}                                                                                                             \
                                                                                                              \
std::vector<Tensor> foreach_tensor_##NAME##_scalarlist_kernel_slow(TensorList tensors, ScalarList scalars) {  \
  check_foreach_api_restrictions(tensors);                                                                    \
                                                                                                              \
  std::vector<Tensor> result;                                                                                 \
  result.reserve(tensors.size());                                                                             \
  for (int i = 0; i < tensors.size(); i++) {                                                                  \
    result.emplace_back(tensors[i].NAME(scalars[i]));                                                         \
  }                                                                                                           \
                                                                                                              \
  return result;                                                                                              \
}

#define FOREACH_BINARY_OP_LIST(NAME)                                                                      \
std::vector<Tensor> foreach_tensor_##NAME##_list_kernel_slow(TensorList tensors1, TensorList tensors2) {  \
  check_foreach_api_restrictions(tensors1, tensors2);                                                     \
                                                                                                          \
  std::vector<Tensor> result;                                                                             \
  result.reserve(tensors1.size());                                                                        \
  for (int i = 0; i < tensors1.size(); i++) {                                                             \
    result.emplace_back(tensors1[i].NAME(tensors2[i]));                                                   \
  }                                                                                                       \
                                                                                                          \
  return result;                                                                                          \
}                                                                                                         \
                                                                                                          \
void foreach_tensor_##NAME##_list_kernel_slow_(TensorList tensors1, TensorList tensors2) {                \
  check_foreach_api_restrictions(tensors1, tensors2);                                                     \
                                                                                                          \
  for (int i = 0; i < tensors1.size(); i++) {                                                             \
    tensors1[i].NAME##_(tensors2[i]);                                                                     \
  }                                                                                                       \
}

#define FOREACH_BINARY_OP_LIST_ALPHA(NAME)                                                                              \
std::vector<Tensor> foreach_tensor_##NAME##_list_kernel_slow(TensorList tensors1, TensorList tensors2, Scalar alpha) {  \
  check_foreach_api_restrictions(tensors1, tensors2);                                                                   \
                                                                                                                        \
  std::vector<Tensor> result;                                                                                           \
  result.reserve(tensors1.size());                                                                                      \
  for (int i = 0; i < tensors1.size(); i++) {                                                                           \
    result.emplace_back(tensors1[i].NAME(tensors2[i], alpha));                                                          \
  }                                                                                                                     \
                                                                                                                        \
  return result;                                                                                                        \
}                                                                                                                       \
                                                                                                                        \
void foreach_tensor_##NAME##_list_kernel_slow_(TensorList tensors1, TensorList tensors2, Scalar alpha) {                \
  check_foreach_api_restrictions(tensors1, tensors2);                                                                   \
                                                                                                                        \
  for (int i = 0; i < tensors1.size(); i++) {                                                                           \
    tensors1[i].NAME##_(tensors2[i], alpha);                                                                            \
  }                                                                                                                     \
}

#define FOREACH_UNARY_OP(NAME)                                             \
std::vector<Tensor> foreach_tensor_##NAME##_slow(TensorList tensors) {     \
  check_foreach_api_restrictions(tensors);                                 \
                                                                           \
  std::vector<Tensor> result;                                              \
  result.reserve(tensors.size());                                          \
  for (const auto& t : tensors) {                                          \
    result.emplace_back(t.NAME());                                         \
  }                                                                        \
                                                                           \
  return result;                                                           \
}                                                                          \
                                                                           \
void foreach_tensor_##NAME##_slow_(TensorList tensors) {                   \
  check_foreach_api_restrictions(tensors);                                 \
                                                                           \
  for (auto& t : tensors) {                                                \
    t.NAME##_();                                                           \
  }                                                                        \
}

#define FOREACH_POINTWISE_OP(NAME)                                                                                            \
std::vector<Tensor> foreach_tensor_##NAME##_slow(TensorList input, TensorList tensors1, TensorList tensors2, Scalar scalar) { \
  TORCH_CHECK(input.size() > 0, "Tensor list must have at least one tensor.");                                                \
  TORCH_CHECK(input.size() == tensors1.size(), "Tensor lists must be of the same length.");                                   \
  TORCH_CHECK(tensors1.size() == tensors2.size(), "Tensor lists must be of the same length.");                                \
                                                                                                                              \
  std::vector<Tensor> result;                                                                                                 \
  for (int i = 0; i < input.size(); i++) {                                                                                    \
    result.emplace_back(input[i].NAME(tensors1[i], tensors2[i], scalar));                                                     \
  }                                                                                                                           \
                                                                                                                              \
  return result;                                                                                                              \
}                                                                                                                             \
                                                                                                                              \
void foreach_tensor_##NAME##_slow_(TensorList input, TensorList tensors1, TensorList tensors2, Scalar scalar) {               \
  TORCH_CHECK(input.size() > 0, "Tensor list must have at least one tensor.");                                                \
  TORCH_CHECK(input.size() == tensors1.size(), "Tensor lists must be of the same length.");                                   \
  TORCH_CHECK(tensors1.size() == tensors2.size(), "Tensor lists must be of the same length.");                                \
                                                                                                                              \
  for (int i = 0; i < input.size(); i++) {                                                                                    \
    input[i].NAME##_(tensors1[i], tensors2[i], scalar);                                                                       \
  }                                                                                                                           \
}                                                                                                                             \

FOREACH_BINARY_OP_LIST_ALPHA(add);
FOREACH_BINARY_OP_LIST_ALPHA(sub);
FOREACH_BINARY_OP_SCALAR(add);
FOREACH_BINARY_OP_SCALAR(sub);
FOREACH_BINARY_OP_SCALAR(mul);
FOREACH_BINARY_OP_SCALAR(div);
FOREACH_BINARY_OP_SCALARLIST(add);
FOREACH_BINARY_OP_SCALARLIST(sub);
FOREACH_BINARY_OP_SCALARLIST(mul);
FOREACH_BINARY_OP_SCALARLIST(div);
FOREACH_BINARY_OP_LIST(mul);
FOREACH_BINARY_OP_LIST(div);
FOREACH_UNARY_OP(sqrt);
FOREACH_UNARY_OP(exp);
FOREACH_POINTWISE_OP(addcdiv);
FOREACH_POINTWISE_OP(addcmul);

}} // namespace at::native
