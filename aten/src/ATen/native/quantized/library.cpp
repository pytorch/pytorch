#include <torch/library.h>

TORCH_LIBRARY(quantized, m) {
  m.def("add(Tensor qa, Tensor qb, float scale, int zero_point) -> Tensor qc");
  m.def("add_relu(Tensor qa, Tensor qb, float scale, int zero_point) -> Tensor qc");
  m.def("add_out(Tensor qa, Tensor qb, Tensor(a!) out) -> Tensor(a!) out");
  m.def("add_relu_out(Tensor qa, Tensor qb, Tensor(a!) out) -> Tensor(a!) out");
  m.def("add_scalar(Tensor qa, Scalar b) -> Tensor qc");
  m.def("add_scalar_relu(Tensor qa, Scalar b) -> Tensor qc");
  m.def("add_scalar_out(Tensor qa, Scalar b, Tensor(a!) out) -> Tensor(a!) out");
  m.def("add_scalar_relu_out(Tensor qa, Scalar b, Tensor(a!) out) -> Tensor(a!) out");
  m.def("batch_norm2d(Tensor qx, Tensor weight, Tensor bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor");
  m.def("batch_norm2d_relu(Tensor qx, Tensor weight, Tensor bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor");
  m.def("batch_norm3d(Tensor qx, Tensor weight, Tensor bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor");
  m.def("batch_norm3d_relu(Tensor qx, Tensor weight, Tensor bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor");
  m.def("clamp(Tensor qx, Scalar? min, Scalar? max) -> Tensor qy");
  m.def("cat(Tensor[] qx, int dim, float? scale, int? zero_point) -> Tensor");
  m.def("cat_relu(Tensor[] qx, int dim, float? scale, int? zero_point) -> Tensor");
  m.def("cat_out(Tensor[] qx, int dim, Tensor(a!) out) -> Tensor(a!)");
  m.def("cat_relu_out(Tensor[] qx, int dim, Tensor(a!) out) -> Tensor(a!)");
  m.def("conv2d(Tensor qx, Tensor weight, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point) -> Tensor");
  m.def("conv2d_relu(Tensor qx, Tensor weight, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point) -> Tensor");
  m.def("conv3d(Tensor qx, Tensor weight, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point) -> Tensor");
  m.def("conv3d_relu(Tensor qx, Tensor weight, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point) -> Tensor");
  // conv_prepack is deprecated, please use conv2d_prepack for 2D conv.
  m.def("conv_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor");
  m.def("conv2d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor");
  m.def("conv3d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor");
  // conv_unpack is deprecated, please use conv2d_unpack for 2D conv.
  m.def("conv_unpack(Tensor packed_weights) -> (Tensor unpacked_weights, Tensor? B_origin)");
  m.def("conv2d_unpack(Tensor packed_weights) -> (Tensor unpacked_weights, Tensor? B_origin)");
  m.def("conv3d_unpack(Tensor packed_weights) -> (Tensor unpacked_weights, Tensor? B_origin)");
  m.def("group_norm(Tensor input, int num_groups, Tensor weight, Tensor bias, float eps, float output_scale, int output_zero_point) -> Tensor");
  m.def("instance_norm(Tensor input, Tensor weight, Tensor bias, float eps, float output_scale, int output_zero_point) -> Tensor");
  m.def("layer_norm(Tensor input, int[] normalized_shape, Tensor weight, Tensor bias, float eps, float output_scale, int output_zero_point) -> Tensor");
  m.def("linear(Tensor X, Tensor W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y");
  m.def("linear_relu(Tensor X, Tensor W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y");
  m.def("linear_dynamic(Tensor X, Tensor W_prepack) -> Tensor Y");
  m.def("linear_relu_dynamic(Tensor X, Tensor W_prepack) -> Tensor Y");
  m.def("linear_dynamic_fp16(Tensor X, Tensor W_prepack) -> Tensor Y");
  m.def("linear_prepack(Tensor W, Tensor? B=None) -> Tensor W_prepack");
  m.def("linear_prepack_fp16(Tensor W, Tensor? B=None) -> Tensor W_prepack");
  m.def("linear_unpack(Tensor W_prepack) -> (Tensor W_origin, Tensor? B_origin)");
  m.def("linear_unpack_fp16(Tensor W_prepack) -> (Tensor W_origin, Tensor? B_origin)");
  m.def("mul(Tensor qa, Tensor qb, float scale, int zero_point)-> Tensor qc");
  m.def("mul_relu(Tensor qa, Tensor qb, float scale, int zero_point)-> Tensor qc");
  m.def("mul_out(Tensor qa, Tensor qb, Tensor(a!) out)-> Tensor(a!) out");
  m.def("mul_relu_out(Tensor qa, Tensor qb, Tensor(a!) out)-> Tensor(a!) out");
  m.def("mul_scalar(Tensor qa, Scalar b)-> Tensor qc");
  m.def("mul_scalar_relu(Tensor qa, Scalar b)-> Tensor qc");
  m.def("mul_scalar_out(Tensor qa, Scalar b, Tensor(a!) out)-> Tensor(a!) out");
  m.def("mul_scalar_relu_out(Tensor qa, Scalar b, Tensor(a!) out)-> Tensor(a!) out");
  // NB: missing a space after comma here...
  m.def("max_pool2d(Tensor qx, int[] kernel_size, int[] stride, int[] padding, int[] dilation,bool ceil_mode) -> Tensor");
  m.def("relu6(Tensor qx, bool inplace=False) -> Tensor");
}

// According to #33294: The "_" prefix registration will be
// removed when the operators are all migrated to mobile.
// https://github.com/pytorch/pytorch/issues/36510
TORCH_LIBRARY(_quantized, m) {
  m.def("add(Tensor qa, Tensor qb, float scale, int zero_point) -> Tensor qc");
  m.def("conv2d(Tensor qx, Tensor weight, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point) -> Tensor");
  m.def("conv2d_relu(Tensor qx, Tensor weight, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point) -> Tensor");
  m.def("conv2d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor");
  m.def("linear(Tensor X, Tensor W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y");
  m.def("linear_dynamic(Tensor X, Tensor W_prepack) -> Tensor Y");
  m.def("linear_prepack(Tensor W, Tensor? B=None) -> Tensor W_prepack");
  m.def("linear_prepack_fp16(Tensor W, Tensor? B=None) -> Tensor W_prepack");
}
