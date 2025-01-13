#include <torch/library.h>

int register_linear_params();

template <int kSpatialDim = 2>
int register_conv_params();

extern template int register_conv_params<2>();
extern template int register_conv_params<3>();
int register_embedding_params();

TORCH_LIBRARY(quantized, m) {
  m.set_python_module("caffe2.torch.fb.model_transform.splitting.split_dispatcher");
  register_linear_params();
  register_conv_params<2>();
  register_conv_params<3>();
  register_embedding_params();

  m.def(TORCH_SELECTIVE_SCHEMA("quantized::add(Tensor qa, Tensor qb, float scale, int zero_point) -> Tensor qc"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::add.out(Tensor qa, Tensor qb, Tensor(a!) out) -> Tensor(a!) out"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::add.Scalar(Tensor qa, Scalar b) -> Tensor qc"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::add.Scalar2(Scalar b, Tensor qa) -> Tensor qc"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::add.Scalar_out(Tensor qa, Scalar b, Tensor(a!) out) -> Tensor(a!) out"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::add_relu(Tensor qa, Tensor qb, float scale, int zero_point) -> Tensor qc"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::add_relu.Scalar(Tensor qa, Scalar b) -> Tensor qc"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::add_relu.Scalar2(Scalar b, Tensor qa) -> Tensor qc"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::add_relu.out(Tensor qa, Tensor qb, Tensor(a!) out) -> Tensor(a!) out"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::add_relu.Scalar_out(Tensor qa, Scalar b, Tensor(a!) out) -> Tensor(a!) out"), {at::Tag::pt2_compliant_tag});
  // deprecated functions, kept for backward compatibility
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::add_out(Tensor qa, Tensor qb, Tensor(a!) out) -> Tensor(a!) out"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::add_relu_out(Tensor qa, Tensor qb, Tensor(a!) out) -> Tensor(a!) out"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::add_scalar(Tensor qa, Scalar b) -> Tensor qc"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::add_scalar_relu(Tensor qa, Scalar b) -> Tensor qc"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::add_scalar_out(Tensor qa, Scalar b, Tensor(a!) out) -> Tensor(a!) out"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::add_scalar_relu_out(Tensor qa, Scalar b, Tensor(a!) out) -> Tensor(a!) out"), {at::Tag::pt2_compliant_tag});
  // TODO: remove after broadcasting is supported
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::add_scalar_out.Tensor(Tensor qa, Tensor b, Tensor(a!) out) -> Tensor(a!) out"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::add_scalar.Tensor(Tensor qa, Tensor b) -> Tensor qc"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::add_scalar_relu.Tensor(Tensor qa, Tensor b) -> Tensor qc"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::add_scalar_relu_out.Tensor(Tensor qa, Tensor b, Tensor(a!) out) -> Tensor(a!) out"), {at::Tag::pt2_compliant_tag});
  // This is needed for graph mode quantization, when we fuse
  // dequant - aten::batch_norm - quant into quantized::batch_norm
  // and dimension is unknown given only the aten op call
  // quantized::batch_norm supports both 2d and 3d batch norm right now
  // it should also support 1d batch_norm after quantized::batch_norm1d is
  // implemented
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::batch_norm(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::batch_norm_relu(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::batch_norm1d(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::batch_norm1d_relu(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::batch_norm2d(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::batch_norm2d_relu(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::batch_norm3d(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::batch_norm3d_relu(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::clamp(Tensor qx, Scalar? min=None, Scalar? max=None) -> Tensor qy"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::threshold(Tensor qx, Scalar threshold, Scalar value) -> Tensor qy"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::cat(Tensor[] qx, int dim, float? scale, int? zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::cat_relu(Tensor[] qx, int dim, float? scale, int? zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::cat_out(Tensor[] qx, int dim, Tensor(a!) out) -> Tensor(a!)"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::cat_relu_out(Tensor[] qx, int dim, Tensor(a!) out) -> Tensor(a!)"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv1d(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv1d_relu(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d.new(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_relu.new(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_add(Tensor qx, Tensor qaccum, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_add_relu(Tensor qx, Tensor qaccum, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d.new(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d_relu.new(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase weight, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_relu(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase weight, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase weight, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d_relu(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase weight, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv1d_dynamic(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, bool reduce_range=False) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_dynamic(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, bool reduce_range=False) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d_dynamic(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weight, bool reduce_range=False) -> Tensor"), {at::Tag::pt2_compliant_tag});

  // conv_prepack is deprecated, please use conv2d_prepack for 2D conv.
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv2dPackedParamsBase"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv1d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv2dPackedParamsBase"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv2dPackedParamsBase"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv3dPackedParamsBase"), {at::Tag::pt2_compliant_tag});
  // conv_unpack is deprecated, please use conv2d_unpack for 2D conv.
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_unpack(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> (Tensor unpacked_weights, Tensor? B_origin)"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv1d_unpack(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> (Tensor unpacked_weights, Tensor? B_origin)"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_unpack(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> (Tensor unpacked_weights, Tensor? B_origin)"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_unpack_sizes(Any packed_weights) -> (Any)"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d_unpack(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> (Tensor unpacked_weights, Tensor? B_origin)"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_stride(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int[]"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_padding(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int[]"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_output_padding(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int[]"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_dilation(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int[]"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_groups(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv2d_transpose(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d_stride(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int[]"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d_padding(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int[]"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d_output_padding(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int[]"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d_dilation(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int[]"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d_groups(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv3d_transpose(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int"), {at::Tag::pt2_compliant_tag});
  // conv_transpose
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose1d(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose2d(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose3d(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose1d_dynamic(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, bool reduce_range=False) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose2d_dynamic(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, bool reduce_range=False) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose3d_dynamic(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weight, bool reduce_range=False) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose1d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv2dPackedParamsBase"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose1d_unpack(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> (Tensor unpacked_weights, Tensor? B_origin)"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose2d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv2dPackedParamsBase"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose2d_unpack(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> (Tensor unpacked_weights, Tensor? B_origin)"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose2d_stride(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int[]"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose2d_padding(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int[]"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose2d_output_padding(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int[]"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose2d_dilation(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int[]"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose2d_groups(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose2d_transpose(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose3d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv3dPackedParamsBase"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose3d_unpack(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> (Tensor unpacked_weights, Tensor? B_origin)"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose3d_stride(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int[]"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose3d_padding(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int[]"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose3d_output_padding(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int[]"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose3d_dilation(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int[]"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose3d_groups(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::conv_transpose3d_transpose(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int"), {at::Tag::pt2_compliant_tag});

  m.def(TORCH_SELECTIVE_SCHEMA("quantized::elu(Tensor self, float output_scale, int output_zero_point, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::dropout(Tensor self, float output_scale, int output_zero_point, Scalar p=0.5, bool training=False) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::embedding_bag_prepack(Tensor weight) -> __torch__.torch.classes.quantized.EmbeddingPackedParamsBase W_prepack"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::embedding_bag_unpack(__torch__.torch.classes.quantized.EmbeddingPackedParamsBase W_prepack) -> Tensor W_origin"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::embedding_bag_byte_prepack(Tensor weight) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::embedding_bag_byte_unpack(Tensor weight) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::embedding_bag_4bit_prepack(Tensor weight, bool optimized_qparams=False, int nbins=200, float ratio=0.16) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::embedding_bag_4bit_unpack(Tensor weight) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::embedding_bag_2bit_prepack(Tensor weight, bool optimized_qparams=False, int nbins=200, float ratio=0.16) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::embedding_bag_2bit_unpack(Tensor weight) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::embedding_bag_byte_rowwise_offsets(Tensor weight, Tensor indices, Tensor? offsets=None, bool scale_grad_by_freq=False, int mode=0, bool pruned_weights=False, Tensor? per_sample_weights=None, Tensor? compressed_indices_mapping=None, bool include_last_offset=False) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::embedding_bag_4bit_rowwise_offsets(Tensor weight, Tensor indices, Tensor? offsets=None, bool scale_grad_by_freq=False, int mode=0, bool pruned_weights=False, Tensor? per_sample_weights=None, Tensor? compressed_indices_mapping=None, bool include_last_offset=False) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::embedding_bag_2bit_rowwise_offsets(Tensor weight, Tensor indices, Tensor? offsets=None, bool scale_grad_by_freq=False, int mode=0, bool pruned_weights=False, Tensor? per_sample_weights=None, Tensor? compressed_indices_mapping=None, bool include_last_offset=False) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::embedding_bag_byte(__torch__.torch.classes.quantized.EmbeddingPackedParamsBase weight, Tensor indices, Tensor? offsets=None, bool scale_grad_by_freq=False, int mode=0, bool pruned_weights=False, Tensor? per_sample_weights=None, Tensor? compressed_indices_mapping=None, bool include_last_offset=False) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::embedding_bag_4bit(__torch__.torch.classes.quantized.EmbeddingPackedParamsBase weight, Tensor indices, Tensor? offsets=None, bool scale_grad_by_freq=False, int mode=0, bool pruned_weights=False, Tensor? per_sample_weights=None, Tensor? compressed_indices_mapping=None, bool include_last_offset=False) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::embedding_byte(__torch__.torch.classes.quantized.EmbeddingPackedParamsBase weight, Tensor indices, bool pruned_weights=False) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::embedding_4bit(__torch__.torch.classes.quantized.EmbeddingPackedParamsBase weight, Tensor indices, bool pruned_weights=False) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::celu(Tensor self, float output_scale, int output_zero_point, Scalar alpha=1) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::group_norm(Tensor input, int num_groups, Tensor? weight, Tensor? bias, float eps, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::hardswish(Tensor input, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::instance_norm(Tensor input, Tensor? weight, Tensor? bias, float eps, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_relu(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_dynamic(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, bool reduce_range=False) -> Tensor Y"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_relu_dynamic(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, bool reduce_range=False) -> Tensor Y"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_dynamic_fp16(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack) -> Tensor Y"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_dynamic_fp16_unpacked_weight(Tensor X, Tensor weight, Tensor bias) -> Tensor Y"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_relu_dynamic_fp16(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack) -> Tensor Y"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_leaky_relu(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i, float negative_slope) -> Tensor Y"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_tanh(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y"), {at::Tag::pt2_compliant_tag});
  // Corresponding pattern (the ops with `*` are part of the pattern that
  // represents the computation of quantized::linear_with_input_q_dq_qweight_dq_output_fp32):
  // input -> q* -> dq* -> linear* ->
  //         qweight -> dq* /
  //
  // After fusion:
  // input -> quantized::linear_with_input_q_dq_qweight_dq_output_fp32* ->
  //         qweight /
  //
  // Additional Note: the weight is packed as well
  // Params:
  //    X: float32 Tensor, will be quantized to quint8 in the op
  //    W_prepack: packed qint8 quantized weight and bias
  // Returns:
  //    Y: float32 Tensor
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_with_input_q_dq_qweight_dq_output_fp32(Tensor X, float X_scale, int X_zero_point, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack) -> Tensor Y"), {at::Tag::pt2_compliant_tag});
  // Corresponding pattern (the ops with `*` are part of the pattern that
  // represents the computation of quantized::linear_with_input_q_dq_qweight_dq_relu_output_fp32):
  // input -> q* -> dq* -> linear* -> relu* ->
  //         qweight -> dq* /
  //
  // After fusion:
  // input -> quantized::linear_with_input_q_dq_qweight_dq_relu_output_fp32* ->
  //         qweight /
  //
  // Additional Note: the weight is packed as well
  // Params:
  //    X: float32 Tensor, will be quantized to quint8 in the op
  //    W_prepack: packed qint8 quantized weight and bias
  // Returns:
  //    Y: float32 Tensor
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_with_input_q_dq_qweight_dq_relu_output_fp32(Tensor X, float X_scale, int X_zero_point, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack) -> Tensor Y"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_prepack(Tensor W, Tensor? B=None) -> __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_prepack_fp16(Tensor W, Tensor? B=None) -> __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_prepack_legacy(Tensor W, Tensor? B=None) -> Tensor W_prepack"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_prepack_fp16_legacy(Tensor W, Tensor? B=None) -> Tensor W_prepack"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_unpack(__torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack) -> (Tensor W_origin, Tensor? B_origin)"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_unpack_fp16(__torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack) -> (Tensor W_origin, Tensor? B_origin)"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_unpack.legacy(Tensor W_prepack) -> (Tensor W_origin, Tensor? B_origin)"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_unpack_fp16.legacy(Tensor W_prepack) -> (Tensor W_origin, Tensor? B_origin)"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::matmul(Tensor qa, Tensor qb, float scale, int zero_point)-> Tensor qc"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::mul(Tensor qa, Tensor qb, float scale, int zero_point)-> Tensor qc"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::mul.out(Tensor qa, Tensor qb, Tensor(a!) out)-> Tensor(a!) out"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::mul.Scalar(Tensor qa, Scalar b)-> Tensor qc"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::mul.Scalar2(Scalar b, Tensor qa)-> Tensor qc"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::mul.Scalar_out(Tensor qa, Scalar b, Tensor(a!) out)-> Tensor(a!) out"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::mul_relu(Tensor qa, Tensor qb, float scale, int zero_point)-> Tensor qc"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::mul_relu.out(Tensor qa, Tensor qb, Tensor(a!) out)-> Tensor(a!) out"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::mul_relu.Scalar(Tensor qa, Scalar b)-> Tensor qc"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::mul_relu.Scalar2(Scalar b, Tensor qa)-> Tensor qc"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::mul_relu.Scalar_out(Tensor qa, Scalar b, Tensor(a!) out)-> Tensor(a!) out"), {at::Tag::pt2_compliant_tag});
  // deprecated functions, kept for backward compatibility
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::mul_out(Tensor qa, Tensor qb, Tensor(a!) out)-> Tensor(a!) out"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::mul_relu_out(Tensor qa, Tensor qb, Tensor(a!) out)-> Tensor(a!) out"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::mul_scalar(Tensor qa, Scalar b)-> Tensor qc"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::mul_scalar_relu(Tensor qa, Scalar b)-> Tensor qc"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::mul_scalar_out(Tensor qa, Scalar b, Tensor(a!) out)-> Tensor(a!) out"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::mul_scalar_relu_out(Tensor qa, Scalar b, Tensor(a!) out)-> Tensor(a!) out"), {at::Tag::pt2_compliant_tag});
  // TODO: remove after broadcasting is supported
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::mul_scalar.Tensor(Tensor qa, Tensor b)-> Tensor qc"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::mul_scalar_relu.Tensor(Tensor qa, Tensor b)-> Tensor qc"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::mul_scalar_out.Tensor(Tensor qa, Tensor b, Tensor(a!) out)-> Tensor(a!) out"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::mul_scalar_relu_out.Tensor(Tensor qa, Tensor b, Tensor(a!) out)-> Tensor(a!) out"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::max_pool1d(Tensor qx, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::max_pool2d(Tensor qx, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::relu6(Tensor qx, bool inplace=False) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::leaky_relu(Tensor qx, Scalar negative_slope, bool inplace, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::prelu(Tensor qx, Tensor weight, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::sigmoid(Tensor qx, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::softmax(Tensor qx, int dim, float output_scale, int output_zero_point) -> Tensor"), {at::Tag::pt2_compliant_tag});
}

// According to #33294: The "_" prefix registration will be
// removed when the operators are all migrated to mobile.
// https://github.com/pytorch/pytorch/issues/36510
TORCH_LIBRARY(_quantized, m) {
  m.def(TORCH_SELECTIVE_SCHEMA("_quantized::add(Tensor qa, Tensor qb, float scale, int zero_point) -> Tensor qc"));
  m.def(TORCH_SELECTIVE_SCHEMA("_quantized::conv2d(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("_quantized::conv2d_relu(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("_quantized::conv2d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv2dPackedParamsBase"));
  m.def(TORCH_SELECTIVE_SCHEMA("_quantized::conv3d(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("_quantized::conv3d_relu(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("_quantized::conv3d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv3dPackedParamsBase"));
  m.def(TORCH_SELECTIVE_SCHEMA("_quantized::conv_transpose1d(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("_quantized::conv_transpose2d(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("_quantized::conv_transpose1d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv2dPackedParamsBase"));
  m.def(TORCH_SELECTIVE_SCHEMA("_quantized::conv_transpose2d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv2dPackedParamsBase"));
  m.def(TORCH_SELECTIVE_SCHEMA("_quantized::conv_transpose3d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv3dPackedParamsBase"));
  m.def(TORCH_SELECTIVE_SCHEMA("_quantized::linear(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA("_quantized::linear_dynamic(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, bool reduce_range=False) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA("_quantized::linear_prepack(Tensor W, Tensor? B=None) -> __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack"));
  m.def(TORCH_SELECTIVE_SCHEMA("_quantized::linear_prepack_fp16(Tensor W, Tensor? B=None) -> __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack"));
  m.def(TORCH_SELECTIVE_SCHEMA("_quantized::linear_prepack_legacy(Tensor W, Tensor? B=None) -> Tensor W_prepack"));
  m.def(TORCH_SELECTIVE_SCHEMA("_quantized::linear_prepack_fp16_legacy(Tensor W, Tensor? B=None) -> Tensor W_prepack"));
  m.def(TORCH_SELECTIVE_SCHEMA("_quantized::wrapped_fbgemm_pack_gemm_matrix_fp16(Tensor W) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("_quantized::wrapped_fbgemm_linear_fp16_weight(Tensor X, Tensor W, Tensor B, int out_channel) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("_quantized::wrapped_quantized_linear(Tensor X, Tensor X_scale, Tensor X_zero_point, Tensor W, Tensor W_scale, Tensor W_zero_point, Tensor B, Tensor output_scale, Tensor output_zero_point, int out_channel) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA("_quantized::_wrapped_linear_prepack(Tensor W, Tensor W_scale, Tensor W_zero_point, Tensor B) -> Tensor"), {at::Tag::flexible_layout});
  m.def(TORCH_SELECTIVE_SCHEMA("_quantized::_wrapped_quantized_linear_prepacked(Tensor X, Tensor X_scale, Tensor X_zero_point, Tensor W_prepack, Tensor output_scale, Tensor output_zero_point, int out_channel) -> Tensor Y"), {at::Tag::flexible_layout});
}

TORCH_LIBRARY_FRAGMENT(onednn, m) {
  // New OP definition for Quantization in PyTorch 2.0 Export
  // Weight Prepack
  m.def(TORCH_SELECTIVE_SCHEMA("onednn::qconv_prepack(Tensor weight, Tensor w_scales, float x_scale, int x_zp, int[] stride, int[] padding, int[] dilation, int groups, int[]? x_shape=None) -> Tensor"));

  // Conv1D/2D/3D with unary postop
  m.def(TORCH_SELECTIVE_SCHEMA("onednn::qconv1d_pointwise(Tensor qx, float x_scale, int x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point, ScalarType? output_dtype, str attr, Scalar?[] scalars, str? algorithm) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("onednn::qconv2d_pointwise(Tensor qx, float x_scale, int x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point, ScalarType? output_dtype, str attr, Scalar?[] scalars, str? algorithm) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("onednn::qconv2d_pointwise.tensor(Tensor qx, Tensor x_scale, Tensor x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point, ScalarType? output_dtype, str attr, Scalar?[] scalars, str? algorithm) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("onednn::qconv3d_pointwise(Tensor qx, float x_scale, int x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point, ScalarType? output_dtype, str attr, Scalar?[] scalars, str? algorithm) -> Tensor"));

  // Conv2D with binary postop
  m.def(TORCH_SELECTIVE_SCHEMA("onednn::qconv2d_pointwise.binary(Tensor qx, float x_scale, int x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor qaccum, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point, ScalarType? output_dtype, float accum_scale, int accum_zero_point, str binary_attr, Scalar? alpha, str? unary_attr, Scalar?[] unary_scalars, str? unary_algorithm) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("onednn::qconv2d_pointwise.binary_tensor(Tensor qx, Tensor x_scale, Tensor x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor qaccum, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point, ScalarType? output_dtype, float accum_scale, int accum_zero_point, str binary_attr, Scalar? alpha, str? unary_attr, Scalar?[] unary_scalars, str? unary_algorithm) -> Tensor"));

  // Linear prepack
  m.def(TORCH_SELECTIVE_SCHEMA("onednn::qlinear_prepack(Tensor weight, int[]? x_shape) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("onednn::linear_prepack_fp16(Tensor weight, int[]? x_shape) -> Tensor"));

  // Linear
  m.def(TORCH_SELECTIVE_SCHEMA("onednn::linear_dynamic_fp16(Tensor x, Tensor w, Tensor? bias) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("onednn::linear_relu_dynamic_fp16(Tensor x, Tensor w, Tensor? bias) -> Tensor"));
  // Linear with unary postop
  m.def(TORCH_SELECTIVE_SCHEMA("onednn::qlinear_pointwise(Tensor qx, float x_scale, int x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? bias, float output_scale, int output_zero_point, ScalarType? output_dtype, str post_op_name, Scalar?[] post_op_args, str post_op_algorithm) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("onednn::qlinear_pointwise.tensor(Tensor qx, Tensor x_scale, Tensor x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? bias, float output_scale, int output_zero_point, ScalarType? output_dtype, str post_op_name, Scalar?[] post_op_args, str post_op_algorithm) -> Tensor"));
  // Linear with binary postop
  m.def(TORCH_SELECTIVE_SCHEMA("onednn::qlinear_pointwise.binary(Tensor qx, float x_scale, int x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? other, Tensor? bias, float output_scale, int output_zero_point, ScalarType? output_dtype, float other_scale, int other_zp, str binary_post_op, float binary_alpha, str unary_post_op, Scalar?[] unary_post_op_args, str unary_post_op_algorithm) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("onednn::qlinear_pointwise.binary_tensor(Tensor qx, Tensor x_scale, Tensor x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? other, Tensor? bias, float output_scale, int output_zero_point, ScalarType? output_dtype, float other_scale, int other_zp, str binary_post_op, float binary_alpha, str unary_post_op, Scalar?[] unary_post_op_args, str unary_post_op_algorithm) -> Tensor"));
}
