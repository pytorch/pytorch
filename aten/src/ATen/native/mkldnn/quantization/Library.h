#include <torch/library.h>

#include <ATen/native/quantized/cpu/conv_packed_params.h>
#include <ATen/native/quantized/cpu/packed_params.h>
#include <ATen/native/quantized/cpu/embedding_packed_params.h>
#include <torch/custom_class.h>

void MKLDNN_LIBRARY_init_quantized (torch::Library& m) {
  // conv
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv1d_mkldnn(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv1d_relu_mkldnn(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_mkldnn.new(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_relu_mkldnn.new(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d_mkldnn.new(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d_relu_mkldnn.new(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_mkldnn(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase weight, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_relu_mkldnn(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase weight, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d_mkldnn(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase weight, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d_relu_mkldnn(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase weight, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point) -> Tensor"));
  // conv_prepack
  // conv_prepack is deprecated, please use conv2d_prepack for 2D conv.
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_prepack_mkldnn(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv2dPackedParamsBase"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv1d_prepack_mkldnn(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv2dPackedParamsBase"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_prepack_mkldnn(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv2dPackedParamsBase"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d_prepack_mkldnn(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv3dPackedParamsBase"));
  // conv_unpack
  // conv_unpack is deprecated, please use conv2d_unpack for 2D conv.
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_unpack_mkldnn(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> (Tensor unpacked_weights, Tensor? B_origin)"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv1d_unpack_mkldnn(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> (Tensor unpacked_weights, Tensor? B_origin)"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_unpack_mkldnn(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> (Tensor unpacked_weights, Tensor? B_origin)"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d_unpack_mkldnn(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> (Tensor unpacked_weights, Tensor? B_origin)"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_stride_mkldnn(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int[]"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_padding_mkldnn(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int[]"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_output_padding_mkldnn(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int[]"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_dilation_mkldnn(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int[]"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_groups_mkldnn(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_transpose_mkldnn(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d_stride_mkldnn(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int[]"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d_padding_mkldnn(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int[]"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d_output_padding_mkldnn(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int[]"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d_dilation_mkldnn(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int[]"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d_groups_mkldnn(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d_transpose_mkldnn(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int"));
  // conv_tranpsose
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose1d_mkldnn(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose2d_mkldnn(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose3d_mkldnn(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose1d_prepack_mkldnn(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv2dPackedParamsBase"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose1d_unpack_mkldnn(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> (Tensor unpacked_weights, Tensor? B_origin)"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose2d_prepack_mkldnn(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv2dPackedParamsBase"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose2d_unpack_mkldnn(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> (Tensor unpacked_weights, Tensor? B_origin)"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose2d_stride_mkldnn(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int[]"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose2d_padding_mkldnn(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int[]"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose2d_output_padding_mkldnn(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int[]"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose2d_dilation_mkldnn(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int[]"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose2d_groups_mkldnn(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose2d_transpose_mkldnn(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose3d_prepack_mkldnn(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv3dPackedParamsBase"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose3d_unpack_mkldnn(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> (Tensor unpacked_weights, Tensor? B_origin)"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose3d_stride_mkldnn(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int[]"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose3d_padding_mkldnn(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int[]"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose3d_output_padding_mkldnn(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int[]"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose3d_dilation_mkldnn(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int[]"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose3d_groups_mkldnn(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose3d_transpose_mkldnn(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int"));
  // Linear
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_mkldnn(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_relu_mkldnn(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_dynamic_mkldnn(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, bool reduce_range=False) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_relu_dynamic_mkldnn(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, bool reduce_range=False) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_prepack_mkldnn(Tensor W, Tensor? B=None) -> __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_prepack_legacy_mkldnn(Tensor W, Tensor? B=None) -> Tensor W_prepack"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_unpack_mkldnn(__torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack) -> (Tensor W_origin, Tensor? B_origin)"));
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_unpack_mkldnn.legacy(Tensor W_prepack) -> (Tensor W_origin, Tensor? B_origin)"));
}