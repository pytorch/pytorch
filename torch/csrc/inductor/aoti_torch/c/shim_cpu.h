#ifndef AOTI_TORCH_SHIM_CPU
#define AOTI_TORCH_SHIM_CPU

#include <ATen/Config.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#ifdef __cplusplus
extern "C" {
#endif

#if AT_MKLDNN_ENABLED()

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_cpu_mkldnn__convolution_pointwise_binary(
    AtenTensorHandle X,
    AtenTensorHandle other,
    AtenTensorHandle W,
    AtenTensorHandle* B,
    const int64_t* padding,
    int64_t padding_len_,
    const int64_t* stride,
    int64_t stride_len_,
    const int64_t* dilation,
    int64_t dilation_len_,
    int64_t groups,
    const char* binary_attr,
    double* alpha,
    const char** unary_attr,
    const double** unary_scalars,
    int64_t unary_scalars_len_,
    const char** unary_algorithm,
    AtenTensorHandle* ret0);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_cpu_mkldnn__convolution_pointwise_binary_(
    AtenTensorHandle other,
    AtenTensorHandle X,
    AtenTensorHandle W,
    AtenTensorHandle* B,
    const int64_t* padding,
    int64_t padding_len_,
    const int64_t* stride,
    int64_t stride_len_,
    const int64_t* dilation,
    int64_t dilation_len_,
    int64_t groups,
    const char* binary_attr,
    double* alpha,
    const char** unary_attr,
    const double** unary_scalars,
    int64_t unary_scalars_len_,
    const char** unary_algorithm,
    AtenTensorHandle* ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_mkldnn__convolution_pointwise(
    AtenTensorHandle X,
    AtenTensorHandle W,
    AtenTensorHandle* B,
    const int64_t* padding,
    int64_t padding_len_,
    const int64_t* stride,
    int64_t stride_len_,
    const int64_t* dilation,
    int64_t dilation_len_,
    int64_t groups,
    const char* attr,
    const double** scalars,
    int64_t scalars_len_,
    const char** algorithm,
    AtenTensorHandle* ret0);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_cpu_mkldnn__convolution_transpose_pointwise(
    AtenTensorHandle X,
    AtenTensorHandle W,
    AtenTensorHandle* B,
    const int64_t* padding,
    int64_t padding_len_,
    const int64_t* output_padding,
    int64_t output_padding_len_,
    const int64_t* stride,
    int64_t stride_len_,
    const int64_t* dilation,
    int64_t dilation_len_,
    int64_t groups,
    const char* attr,
    const double** scalars,
    int64_t scalars_len_,
    const char** algorithm,
    AtenTensorHandle* ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_mkldnn_rnn_layer(
    AtenTensorHandle input,
    AtenTensorHandle weight0,
    AtenTensorHandle weight1,
    AtenTensorHandle weight2,
    AtenTensorHandle weight3,
    AtenTensorHandle hx_,
    AtenTensorHandle cx_,
    int32_t reverse,
    const int64_t* batch_sizes,
    int64_t batch_sizes_len_,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    int32_t has_biases,
    int32_t bidirectional,
    int32_t batch_first,
    int32_t train,
    AtenTensorHandle* ret0,
    AtenTensorHandle* ret1,
    AtenTensorHandle* ret2,
    AtenTensorHandle* ret3);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__linear_pointwise(
    AtenTensorHandle X,
    AtenTensorHandle W,
    AtenTensorHandle* B,
    const char* attr,
    const double** scalars,
    int64_t scalars_len_,
    const char** algorithm,
    AtenTensorHandle* ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__linear_pointwise_binary(
    AtenTensorHandle X,
    AtenTensorHandle other,
    AtenTensorHandle W,
    AtenTensorHandle* B,
    const char* attr,
    AtenTensorHandle* ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__qlinear_pointwise_tensor(
    AtenTensorHandle X,
    AtenTensorHandle act_scale,
    AtenTensorHandle act_zero_point,
    AtenTensorHandle onednn_weight,
    AtenTensorHandle weight_scales,
    AtenTensorHandle weight_zero_points,
    AtenTensorHandle* B,
    double output_scale,
    int64_t output_zero_point,
    const int32_t* output_dtype,
    const char* post_op_name,
    const double** post_op_args,
    int64_t post_op_args_len_,
    const char* post_op_algorithm,
    AtenTensorHandle* ret0);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_cpu__qlinear_pointwise_binary_tensor(
    AtenTensorHandle X,
    AtenTensorHandle act_scale,
    AtenTensorHandle act_zero_point,
    AtenTensorHandle onednn_weight,
    AtenTensorHandle weight_scales,
    AtenTensorHandle weight_zero_points,
    AtenTensorHandle* other,
    AtenTensorHandle* B,
    double output_scale,
    int64_t output_zero_point,
    const int32_t* output_dtype,
    double other_scale,
    int64_t other_zero_point,
    const char* binary_post_op,
    double binary_alpha,
    const char* unary_post_op,
    const double** unary_post_op_args,
    int64_t unary_post_op_args_len_,
    const char* unary_post_op_algorithm,
    AtenTensorHandle* ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__qconv_pointwise_tensor(
    AtenTensorHandle X,
    AtenTensorHandle act_scale,
    AtenTensorHandle act_zero_point,
    AtenTensorHandle onednn_weight,
    AtenTensorHandle weight_scales,
    AtenTensorHandle weight_zero_points,
    AtenTensorHandle* B,
    const int64_t* stride,
    int64_t stride_len_,
    const int64_t* padding,
    int64_t padding_len_,
    const int64_t* dilation,
    int64_t dilation_len_,
    int64_t groups,
    double output_scale,
    int64_t output_zero_point,
    const int32_t* output_dtype,
    const char* attr,
    const double** post_op_args,
    int64_t post_op_args_len_,
    const char** algorithm,
    AtenTensorHandle* ret0);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_cpu__qconv2d_pointwise_binary_tensor(
    AtenTensorHandle X,
    AtenTensorHandle act_scale,
    AtenTensorHandle act_zero_point,
    AtenTensorHandle onednn_weight,
    AtenTensorHandle weight_scales,
    AtenTensorHandle weight_zero_points,
    AtenTensorHandle accum,
    AtenTensorHandle* B,
    const int64_t* stride_args,
    int64_t stride_len_,
    const int64_t* padding_args,
    int64_t padding_len_,
    const int64_t* dilation_args,
    int64_t dilation_len_,
    int64_t groups,
    double output_scale,
    int64_t output_zero_point,
    const int32_t* output_dtype,
    double accum_scale,
    int64_t accum_zero_point,
    const char* binary_attr,
    double* alpha,
    const char** unary_attr,
    const double** unary_scalars,
    int64_t unary_scalars_len_,
    const char** unary_algorithm,
    AtenTensorHandle* ret0);

#if AT_MKL_ENABLED()

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__mkl_linear(
    AtenTensorHandle X,
    AtenTensorHandle W,
    AtenTensorHandle origin_W,
    AtenTensorHandle* B,
    int64_t prepack_batch_size,
    AtenTensorHandle* ret0);

#endif // AT_MKL_ENABLED

#endif // AT_MKLDNN_ENABLED()

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__weight_int4pack_mm_cpu_tensor(
    AtenTensorHandle X,
    AtenTensorHandle w,
    AtenTensorHandle qGroupSize,
    AtenTensorHandle qScaleAndZeros,
    AtenTensorHandle* ret0);

#ifdef __cplusplus
} // extern "C"
#endif
#endif // AOTI_TORCH_SHIM_CPU
