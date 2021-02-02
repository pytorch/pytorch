#include <ATen/ATen.h>
#include <ATen/native/ForeachUtils.h>

namespace at { namespace native {

#define FOREACH_BINARY_OP_SCALAR(OP)                                                                      \
void foreach_tensor_##OP##_scalar_kernel_slow_(TensorList tensors, Scalar scalar) {                       \
  check_foreach_api_restrictions(tensors);                                                                \
                                                                                                          \
  for (auto& t: tensors) {                                                                                \
    t.OP##_(scalar);                                                                                      \
  }                                                                                                       \
}                                                                                                         \
                                                                                                          \
std::vector<Tensor> foreach_tensor_##OP##_scalar_kernel_slow(TensorList tensors, Scalar scalar) {         \
  check_foreach_api_restrictions(tensors);                                                                \
                                                                                                          \
  std::vector<Tensor> result;                                                                             \
  result.reserve(tensors.size());                                                                         \
  for (const auto& t: tensors) {                                                                          \
    result.emplace_back(t.OP(scalar));                                                                    \
  }                                                                                                       \
                                                                                                          \
  return result;                                                                                          \
}

#define FOREACH_BINARY_OP_SCALARLIST(OP)                                                                                \
void foreach_tensor_##OP##_scalarlist_kernel_slow_(TensorList tensors, at::ArrayRef<Scalar> scalars) {                  \
  check_foreach_api_restrictions(tensors, scalars);                                                                     \
                                                                                                                        \
  for (size_t i = 0; i < tensors.size(); i++) {                                                                            \
      tensors[i].OP##_(scalars[i]);                                                                                     \
    }                                                                                                                   \
}                                                                                                                       \
                                                                                                                        \
std::vector<Tensor> foreach_tensor_##OP##_scalarlist_kernel_slow(TensorList tensors, at::ArrayRef<Scalar> scalars) {    \
  check_foreach_api_restrictions(tensors, scalars);                                                                     \
  std::vector<Tensor> result;                                                                                           \
  result.reserve(tensors.size());                                                                                       \
  for (size_t i = 0; i < tensors.size(); i++) {                                                                            \
    result.emplace_back(tensors[i].OP(scalars[i]));                                                                     \
  }                                                                                                                     \
                                                                                                                        \
  return result;                                                                                                        \
}

#define FOREACH_BINARY_OP_LIST(OP)                                                                        \
std::vector<Tensor> foreach_tensor_##OP##_list_kernel_slow(TensorList tensors1, TensorList tensors2) {    \
  check_foreach_api_restrictions(tensors1, tensors2);                                                     \
                                                                                                          \
  std::vector<Tensor> result;                                                                             \
  result.reserve(tensors1.size());                                                                        \
  for (size_t i = 0; i < tensors1.size(); i++) {                                                             \
    result.emplace_back(tensors1[i].OP(tensors2[i]));                                                     \
  }                                                                                                       \
                                                                                                          \
  return result;                                                                                          \
}                                                                                                         \
                                                                                                          \
void foreach_tensor_##OP##_list_kernel_slow_(TensorList tensors1, TensorList tensors2) {                  \
  check_foreach_api_restrictions(tensors1, tensors2);                                                     \
                                                                                                          \
  for (size_t i = 0; i < tensors1.size(); i++) {                                                             \
    tensors1[i].OP##_(tensors2[i]);                                                                       \
  }                                                                                                       \
}

#define FOREACH_BINARY_OP_LIST_ALPHA(OP)                                                                                \
std::vector<Tensor> foreach_tensor_##OP##_list_kernel_slow(TensorList tensors1, TensorList tensors2, Scalar alpha) {    \
  check_foreach_api_restrictions(tensors1, tensors2);                                                                   \
                                                                                                                        \
  std::vector<Tensor> result;                                                                                           \
  result.reserve(tensors1.size());                                                                                      \
  for (size_t i = 0; i < tensors1.size(); i++) {                                                                           \
    result.emplace_back(tensors1[i].OP(tensors2[i], alpha));                                                            \
  }                                                                                                                     \
                                                                                                                        \
  return result;                                                                                                        \
}                                                                                                                       \
                                                                                                                        \
void foreach_tensor_##OP##_list_kernel_slow_(TensorList tensors1, TensorList tensors2, Scalar alpha) {                  \
  check_foreach_api_restrictions(tensors1, tensors2);                                                                   \
                                                                                                                        \
  for (size_t i = 0; i < tensors1.size(); i++) {                                                                           \
    tensors1[i].OP##_(tensors2[i], alpha);                                                                              \
  }                                                                                                                     \
}

#define FOREACH_UNARY_OP(OP)                                               \
std::vector<Tensor> foreach_tensor_##OP##_slow(TensorList tensors) {       \
  check_foreach_api_restrictions(tensors);                                 \
                                                                           \
  std::vector<Tensor> result;                                              \
  result.reserve(tensors.size());                                          \
  for (const auto& t : tensors) {                                          \
    result.emplace_back(t.OP());                                           \
  }                                                                        \
                                                                           \
  return result;                                                           \
}                                                                          \
                                                                           \
void foreach_tensor_##OP##_slow_(TensorList tensors) {                     \
  check_foreach_api_restrictions(tensors);                                 \
                                                                           \
  for (auto& t : tensors) {                                                \
    t.OP##_();                                                             \
  }                                                                        \
}

#define FOREACH_POINTWISE_OP_SCALAR(OP)                                                                                              \
std::vector<Tensor> foreach_tensor_##OP##_scalar_slow(TensorList input, TensorList tensors1, TensorList tensors2, Scalar scalar) {   \
  check_foreach_api_restrictions(input, tensors1, tensors2);                                                                         \
                                                                                                                                     \
  std::vector<Tensor> result;                                                                                                        \
  for (size_t i = 0; i < input.size(); i++) {                                                                                           \
    result.emplace_back(input[i].OP(tensors1[i], tensors2[i], scalar));                                                              \
  }                                                                                                                                  \
                                                                                                                                     \
  return result;                                                                                                                     \
}                                                                                                                                    \
                                                                                                                                     \
void foreach_tensor_##OP##_scalar_slow_(TensorList input, TensorList tensors1, TensorList tensors2, Scalar scalar) {                 \
  check_foreach_api_restrictions(input, tensors1, tensors2);                                                                         \
                                                                                                                                     \
  for (size_t i = 0; i < input.size(); i++) {                                                                                           \
    input[i].OP##_(tensors1[i], tensors2[i], scalar);                                                                                \
  }                                                                                                                                  \
}                                                                                                                                    \

#define FOREACH_POINTWISE_OP_SCALARLIST(OP)                                                                                                             \
std::vector<Tensor> foreach_tensor_##OP##_scalarlist_slow(TensorList input, TensorList tensors1, TensorList tensors2, at::ArrayRef<Scalar> scalars) {   \
  check_foreach_api_restrictions(input, tensors1, tensors2, scalars);                                                                                   \
                                                                                                                                                        \
  std::vector<Tensor> result;                                                                                                                           \
  for (size_t i = 0; i < input.size(); i++) {                                                                                                              \
    result.emplace_back(input[i].OP(tensors1[i], tensors2[i], scalars[i]));                                                                             \
  }                                                                                                                                                     \
                                                                                                                                                        \
  return result;                                                                                                                                        \
}                                                                                                                                                       \
                                                                                                                                                        \
void foreach_tensor_##OP##_scalarlist_slow_(TensorList input, TensorList tensors1, TensorList tensors2, at::ArrayRef<Scalar> scalars) {                 \
  check_foreach_api_restrictions(input, tensors1, tensors2, scalars);                                                                                   \
                                                                                                                                                        \
  for (size_t i = 0; i < input.size(); i++) {                                                                                                              \
    input[i].OP##_(tensors1[i], tensors2[i], scalars[i]);                                                                                               \
  }                                                                                                                                                     \
}                                                                                                                                                       \

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
FOREACH_UNARY_OP(abs);
FOREACH_UNARY_OP(acos);
FOREACH_UNARY_OP(asin);
FOREACH_UNARY_OP(atan);
FOREACH_UNARY_OP(ceil);
FOREACH_UNARY_OP(cos);
FOREACH_UNARY_OP(cosh);
FOREACH_UNARY_OP(erf);
FOREACH_UNARY_OP(erfc);
FOREACH_UNARY_OP(expm1);
FOREACH_UNARY_OP(floor);
FOREACH_UNARY_OP(log);
FOREACH_UNARY_OP(log10);
FOREACH_UNARY_OP(log1p);
FOREACH_UNARY_OP(log2);
FOREACH_UNARY_OP(neg);
FOREACH_UNARY_OP(tan);
FOREACH_UNARY_OP(tanh);
FOREACH_UNARY_OP(sin);
FOREACH_UNARY_OP(sinh);
FOREACH_UNARY_OP(round);
FOREACH_UNARY_OP(lgamma);
FOREACH_UNARY_OP(frac);
FOREACH_UNARY_OP(trunc);
FOREACH_UNARY_OP(reciprocal);
FOREACH_UNARY_OP(sigmoid);

FOREACH_POINTWISE_OP_SCALAR(addcdiv);
FOREACH_POINTWISE_OP_SCALAR(addcmul);

FOREACH_POINTWISE_OP_SCALARLIST(addcdiv);
FOREACH_POINTWISE_OP_SCALARLIST(addcmul);

#define FOREACH_MAXIMUM_MINIMUM_OP(NAME)                                                     \
std::vector<Tensor> foreach_tensor_##NAME##_slow(TensorList tensors1, TensorList tensors2) { \
  check_foreach_api_restrictions(tensors1, tensors2);                                        \
                                                                                             \
  std::vector<Tensor> result;                                                                \
  result.reserve(tensors1.size());                                                           \
  for (size_t i = 0; i < tensors1.size(); i++) {                                             \
    result.emplace_back(at::NAME(tensors1[i], tensors2[i]));                                 \
  }                                                                                          \
                                                                                             \
  return result;                                                                             \
}                                                                                            \

FOREACH_MAXIMUM_MINIMUM_OP(maximum)
FOREACH_MAXIMUM_MINIMUM_OP(minimum)

void foreach_tensor_zero_slow_(TensorList tensors) {
  check_foreach_api_restrictions(tensors);

  for (auto& t : tensors) {
    t.zero_();
  }
}

}} // namespace at::native
