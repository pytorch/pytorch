#pragma once
#include <ATen/native/zendnn/ZenDNN_utils.hpp>
#include <cstdint>
#include <functional> // For std::reference_wrapper, std::ref, std::cref
#include <iostream>
#include <optional> // For std::optional, std::nullopt
#include <string>
#include <string_view>
#include <unordered_map>
#if AT_ZENDNN_ENABLED()
#include <zendnnl.hpp>
namespace at::native {
using namespace zendnnl::interface;

inline std::vector<int64_t> get_matmul_output_sizes(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2) {
  auto tensor1_size = tensor1.sizes();
  std::vector<int64_t> output_size(
      tensor1_size.begin(), tensor1_size.end() - 1);
  auto tensor2_last_dim_size = tensor2.size(tensor2.dim() - 1);
  output_size.emplace_back(tensor2_last_dim_size);
  return output_size;
}
// this function returns the output stride for matrix multiplication of two
// tensors - tensor1 @ tensor2 and also it returns the output stride for
// linear operation of these two tensors
inline std::vector<int64_t> get_matmul_and_linear_output_strides(
    const std::vector<int64_t>& output_size) {
  int output_size_sz = output_size.size();
  std::vector<int64_t> output_strides;
  int64_t mul = 1;
  for (int cnt = 0; cnt < output_size_sz; cnt++) {
    if (cnt > 0) {
      mul *= output_size[output_size_sz - cnt];
    }
    output_strides.emplace_back(mul);
  }
  std::reverse(output_strides.begin(), output_strides.end());
  return output_strides;
}
inline void check_valid_sizes_for_matmul(
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Tensor& bias,
    const at::Tensor& result,
    const std::vector<at::Tensor>& post_op_buffers) {
  // The flow of this check is as follows:
  // -> Generic dim check for the mat1 and mat2. The functionality of aten::mv
  //    is covered here.
  // -> Next the result shape is checked to be compatible with the matrix
  //    multiplication shape. This is done at the second stage as rest of the
  //    tensors can be optional, and irrespective of the other tensors, the
  //    matrix multiplication of mat1 and mat2 will happen.
  // -> Bias being optional in the addmm variants, needs to be checked if it is
  //    a defined tensor or not, and based on that shape of bias is checked.
  //    Here, only 1-D bias case is checked, as bias if 2-D or 3-D will be
  //    passed as post op and checked in the post op checks.
  // -> Based on the post op buffer vector size, the shapes of all the post ops
  //    are determined. Again here, all the post op buffers must be of the same
  //    shape as the matmul product shape or result tensor shape.
  const int mat1_dim = mat1.dim();
  const int mat2_dim = mat2.dim();
  TORCH_CHECK(
      ((mat1_dim == 3 &&
        mat2_dim == 3) || // dimensionality check for matrix multiplication
       (mat1_dim == 2 &&
        mat2_dim == 2) || // dimensionality check for matrix multiplication
       (mat1_dim == 2 && mat2_dim == 1) || // specific case for aten::mv
       (mat1_dim == 1 && mat2_dim == 1) // specific case for aten::dot
       ),
      "unsupported dims for mat1 and mat2");
  // Array access is faster than .size(n)
  const auto mat1_sizes = mat1.sizes();
  const auto mat2_sizes = mat2.sizes();
  if (mat1_dim == 2 && mat2_dim == 1) {
    TORCH_CHECK(
        post_op_buffers.size() == 0,
        "Post Op support currently unavailable for aten::mv via ZenDNN");
    // TODO
    // Need to understand how to the result is in these cases and need to add a
    // check for the result buffer as well.
    return;
  }
  if (mat1_dim == 1 && mat2_dim == 1) {
    TORCH_CHECK(
        post_op_buffers.size() == 0,
        "Post Op support currently unavailable for aten::dot via ZenDNN");
    // TODO
    // Need to understand how to the result is in these cases and need to add a
    // check for the result buffer as well.
    return;
  }
  if (mat1_dim == 3) {
    TORCH_CHECK(
        mat1_sizes[0] == mat2_sizes[0],
        "Tensor shapes incompatible for batch matrix multiplication");
  }
  TORCH_CHECK(
      mat1_sizes[mat1_dim - 1] == mat2_sizes[mat1_dim - 2],
      "Tensor shapes incompatible for matrix multiplication");
  const bool is_bias_defined = bias.defined();
  if (is_bias_defined) {
    if (bias.dim() == 1) {
      const auto bias_sizes = bias.sizes();
      TORCH_CHECK(
          bias_sizes[0] == mat2_sizes[1],
          "input/bias/self shape is incompatible for addition with "
          "matrix multiplication product (",
          mat1_sizes[0],
          "x",
          mat1_sizes[1],
          " @ ",
          mat2_sizes[0],
          "x",
          mat2_sizes[1],
          " != ",
          mat1_sizes[0],
          "x",
          bias_sizes[0],
          ")");
    } else {
      TORCH_CHECK(false, "unsupported dimensions for input/bias/self");
    }
  }
  if (post_op_buffers.size() != 0) {
    bool are_postops_dim_compatible = true;
    bool are_postops_shape_compatible = true;
    for (const at::Tensor& buffer : post_op_buffers) {
      are_postops_dim_compatible =
          are_postops_dim_compatible && (buffer.dim() == mat1_dim);
      are_postops_shape_compatible = are_postops_shape_compatible &&
          (buffer.sizes() ==
           c10::IntArrayRef(get_matmul_output_sizes(mat1, mat2)));
    }
    TORCH_CHECK(
        are_postops_dim_compatible,
        "unsupported dims for mat1, mat2 and "
        "post op buffers");
    TORCH_CHECK(
        are_postops_shape_compatible,
        "unsupported shapes for mat1, mat2 and "
        "post op buffers");
  } else {
    LOG(INFO) << "Post Op buffers are not present!\n";
  }
}
// this function returns the 2-d size for n-d inp_tensor,
// also if inp_tensor is packed on the last dim it will
// support the unpacking the size of last dim of inp_tensor
inline std::vector<int64_t> get_2d_size_for_tensor(
    const at::Tensor& inp_tensor,
    const int64_t unpacking_ratio = 1) {
  const int64_t dim = inp_tensor.dim();
  std::vector<int64_t> output_size(2);
  output_size[0] = inp_tensor.numel() / inp_tensor.size(dim - 1);
  output_size[1] = inp_tensor.size(dim - 1) * unpacking_ratio;
  return output_size;
}

inline void check_valid_dtypes_for_matmul(
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Tensor& bias,
    const at::Tensor& result,
    const std::vector<at::Tensor>& post_op_buffers) {
  // The flow of this check is as follows:
  // -> The individual datatypes of the tensors are inferred.
  // -> Bias being optional in the addmm variants, needs to be checked if it is
  //    a defined tensor or not.
  // -> The tensors which are inputs to the actual matmul call are confirmed
  //    to be either of datatype float32 or bfloat16, but not a combination.
  // -> The previous check is combined with the check of the
  //    destination (result) buffer.
  // -> If the dataype is bfloat16, the machine capability is checked.
  // -> Based on the post op buffer vector size, the dtypes of all the post ops
  //    are determined. Again here, all the post op buffers must be of the same
  //    dtype as the matmul parameters, either float32 or bfloat16, not a
  //    combination of both.
  const bool is_bias_defined = bias.defined();
  const bool is_mat1_fp32 = (mat1.scalar_type() == c10::ScalarType::Float);
  const bool is_mat1_bf16 = (mat1.scalar_type() == c10::ScalarType::BFloat16);
  const bool is_mat2_fp32 = (mat2.scalar_type() == c10::ScalarType::Float);
  const bool is_mat2_bf16 = (mat2.scalar_type() == c10::ScalarType::BFloat16);
  const bool is_result_fp32 = (result.scalar_type() == c10::ScalarType::Float);
  const bool is_result_bf16 =
      (result.scalar_type() == c10::ScalarType::BFloat16);
  bool is_bias_fp32, is_bias_bf16;
  if (is_bias_defined) {
    is_bias_fp32 = (bias.scalar_type() == c10::ScalarType::Float);
    is_bias_bf16 = (bias.scalar_type() == c10::ScalarType::BFloat16);
  }
  const bool are_params_fp32 = is_bias_defined
      ? (is_mat1_fp32 && is_mat2_fp32 && is_bias_fp32 && is_result_fp32)
      : (is_mat1_fp32 && is_mat2_fp32 && is_result_fp32);
  const bool are_params_bf16 = is_bias_defined
      ? (is_mat1_bf16 && is_mat2_bf16 && is_bias_bf16 && is_result_bf16)
      : (is_mat1_bf16 && is_mat2_bf16 && is_result_bf16);
  TORCH_CHECK(
      are_params_fp32 ^ are_params_bf16,
      "zendnn_linear only supports Float and BFloat16");
  if (are_params_bf16) {
    TORCH_CHECK(
        zendnn_bf16_device_check(),
        "zendnn_linear bf16 path needs the cpu support "
        "avx512bf16");
  }
  if (post_op_buffers.size() != 0) {
    bool are_postops_fp32 = true;
    bool are_postops_bf16 = true;
    for (const at::Tensor& buffer : post_op_buffers) {
      are_postops_fp32 =
          are_postops_fp32 && (buffer.scalar_type() == c10::ScalarType::Float);
      are_postops_bf16 = are_postops_bf16 &&
          (buffer.scalar_type() == c10::ScalarType::BFloat16);
    }
    if (are_params_fp32 && !are_params_bf16) {
      TORCH_CHECK(
          (are_postops_fp32 && !are_postops_bf16),
          "zendnn_linear only supports Float post ops "
          "when input matrix is Float");
    } else if (are_params_bf16 && !are_params_fp32) {
      TORCH_CHECK(
          (are_postops_bf16 && !are_postops_fp32),
          "zendnn_linear only supports BFloat16 post ops "
          "when input matrix is BFloat16");
    } else {
      TORCH_CHECK(
          false,
          "zendnn_linear only supports Float and BFloat16 "
          "parameters and postops");
    }
  } else {
    LOG(INFO) << "Post Op buffers are not present!\n";
  }
}
inline void set_matmul_context_attributes(
    matmul_context_t& matmul_context,
    tensor_t& weights,
    const std::vector<std::string_view>& post_op_ids,
    std::optional<std::reference_wrapper<tensor_t>> bias_opt_ref =
        std::nullopt) {
  matmul_context.set_param("weights", weights);
  if (bias_opt_ref.has_value()) {
    tensor_t& bias = bias_opt_ref->get();
    matmul_context.set_param("bias", bias);
  }
  const std::unordered_map<std::string_view, post_op_type_t> post_op_map = {
      {"relu", post_op_type_t::relu},
      {"gelu_tanh", post_op_type_t::gelu_tanh},
      {"gelu_erf", post_op_type_t::gelu_erf},
      {"silu", post_op_type_t::swish},
      {"sigmoid", post_op_type_t::sigmoid},
      {"tanh", post_op_type_t::tanh},
      {"mul", post_op_type_t::binary_mul},
      {"add", post_op_type_t::binary_add}};
  for (const auto& op_str : post_op_ids) {
    auto it = post_op_map.find(op_str);
    if (it != post_op_map.end()) {
      auto post_op = post_op_t{it->second};
      matmul_context.set_post_op(post_op);
    } else {
      if (op_str != "no_post_op_!")
        TORCH_CHECK(false, "Unsupported post operation: ", op_str);
    }
  }
}
} // namespace at::native
#endif // AT_ZENDNN_ENABLED()
