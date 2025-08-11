// This included is needed for the core Tensor class
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/Attention.h>

// The includes below are required to call functions we want within the at namespace
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/matmul.h>
#include <ATen/ops/tanh.h>
#endif

namespace at::native {
  std::tuple<at::Tensor, at::Tensor> attention(const at::Tensor & query,
                                      const at::Tensor & key,
                                      const at::Tensor & value
                                      ) {
    TORCH_CHECK(query.dim() == 2 && key.dim() == 2 && value.dim() == 2,
               "Expected input tensors to be 2D, but got query: ", query.dim(),
               ", key: ", key.dim(),
               ", value: ", value.dim());
    TORCH_CHECK(query.sym_size(0) == key.sym_size(0) && query.sym_size(0) == value.sym_size(0),
               "Expected input tensors to have the same first dimension, but got query: ",
               query.sym_size(0), ", key: ", key.sym_size(0), ", value: ",
               value.sym_size(0));
    TORCH_CHECK(query.sym_size(1) == key.sym_size(1),
               "Expected query and key to have the same second dimension, but got query: ",
                query.sym_size(1), ", key: ", key.sym_size(1)); 
    auto a = at::tanh(at::matmul(query, key.transpose(-2, -1)));
    auto o = at::matmul(a, value);
    return std::make_tuple(o, a);
  }
} // namespace at::native

