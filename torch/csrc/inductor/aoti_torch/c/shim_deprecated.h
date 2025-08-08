#ifndef AOTI_TORCH_SHIM_DEPRECATED
#define AOTI_TORCH_SHIM_DEPRECATED

#include <torch/csrc/inductor/aoti_torch/c/macros.h>

#ifdef __cplusplus
extern "C" {
#endif

[[deprecated(
    "aoti_torch__embedding_bag is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError aoti_torch__embedding_bag(
    AtenTensorHandle weight,
    AtenTensorHandle indices,
    AtenTensorHandle offsets,
    int32_t scale_grad_by_freq,
    int32_t mode,
    int32_t sparse,
    AtenTensorHandle per_sample_weights, // optional argument
    int32_t include_last_offset,
    int32_t padding_idx,
    AtenTensorHandle* ret0, // returns new reference
    AtenTensorHandle* ret1, // returns new reference
    AtenTensorHandle* ret2, // returns new reference
    AtenTensorHandle* ret3 // returns new reference
);

[[deprecated(
    "aoti_torch__fft_c2c is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError aoti_torch__fft_c2c(
    AtenTensorHandle self,
    const int64_t* dim_ptr,
    int64_t dim_size,
    int64_t normalization,
    int32_t forward,
    AtenTensorHandle* ret // returns new reference
);

[[deprecated(
    "aoti_torch__scaled_mm is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError aoti_torch__scaled_mm(
    AtenTensorHandle self,
    AtenTensorHandle mat2,
    AtenTensorHandle bias,
    int32_t* out_dtype,
    AtenTensorHandle scale_a,
    AtenTensorHandle scale_b,
    AtenTensorHandle scale_result,
    int8_t use_fast_accum,
    AtenTensorHandle* ret0,
    AtenTensorHandle* ret1);

[[deprecated(
    "aoti_torch__scaled_mm_v2 is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError aoti_torch__scaled_mm_v2(
    AtenTensorHandle self,
    AtenTensorHandle mat2,
    AtenTensorHandle scale_a,
    AtenTensorHandle scale_b,
    AtenTensorHandle bias,
    AtenTensorHandle scale_result,
    int32_t* out_dtype,
    int8_t use_fast_accum,
    AtenTensorHandle* ret0);

[[deprecated(
    "aoti_torch_addmm_out is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_addmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat1,
    AtenTensorHandle mat2,
    float beta,
    float alpha);

[[deprecated(
    "aoti_torch_bmm is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_bmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2);

[[deprecated(
    "aoti_torch_convolution is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_convolution(
    AtenTensorHandle input,
    AtenTensorHandle weight,
    AtenTensorHandle bias, // optional argument
    const int64_t* stride_ptr,
    int64_t stride_size,
    const int64_t* padding_ptr,
    int64_t padding_size,
    const int64_t* dilation_ptr,
    int64_t dilation_size,
    int transposed,
    const int64_t* output_padding_ptr,
    int64_t output_padding_size,
    int64_t groups,
    AtenTensorHandle* ret // returns new reference
);

[[deprecated(
    "aoti_torch_mm_out is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2);

[[deprecated(
    "aoti_torch_nonzero is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_nonzero(AtenTensorHandle self, AtenTensorHandle* out);

[[deprecated(
    "aoti_torch_repeat_interleave_Tensor is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_repeat_interleave_Tensor(
    AtenTensorHandle repeats,
    int64_t* output_size,
    AtenTensorHandle* out);

[[deprecated(
    "aoti_torch_view_as_real is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_view_as_real(
    AtenTensorHandle self,
    AtenTensorHandle* ret // returns new reference
);

[[deprecated(
    "aoti_torch_view_dtype is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_view_dtype(
    AtenTensorHandle self,
    int32_t dtype,
    AtenTensorHandle* ret // returns new reference
);

[[deprecated(
    "aoti_torch__scaled_dot_product_flash_attention is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError aoti_torch__scaled_dot_product_flash_attention(
    AtenTensorHandle query,
    AtenTensorHandle key,
    AtenTensorHandle value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    double scale,
    AtenTensorHandle* ret0, // returns new reference
    AtenTensorHandle* ret1, // returns new reference
    AtenTensorHandle* ret2, // returns new reference
    AtenTensorHandle* ret3, // returns new reference
    int64_t* ret4,
    int64_t* ret5,
    AtenTensorHandle* ret6, // returns new reference
    AtenTensorHandle* ret7, // returns new reference
    AtenTensorHandle* ret8 // returns new reference
);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch__scaled_dot_product_flash_attention_v2(
    AtenTensorHandle query,
    AtenTensorHandle key,
    AtenTensorHandle value,
    double dropout_p,
    int is_causal,
    int return_debug_mask,
    double* scale, // optional argument
    AtenTensorHandle* ret0, // returns new reference
    AtenTensorHandle* ret1, // returns new reference
    AtenTensorHandle* ret2, // returns new reference
    AtenTensorHandle* ret3, // returns new reference
    int64_t* ret4,
    int64_t* ret5,
    AtenTensorHandle* ret6, // returns new reference
    AtenTensorHandle* ret7, // returns new reference
    AtenTensorHandle* ret8 // returns new reference
);

[[deprecated(
    "aoti_torch__scaled_dot_product_efficient_attention is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch__scaled_dot_product_efficient_attention(
    AtenTensorHandle query,
    AtenTensorHandle key,
    AtenTensorHandle value,
    AtenTensorHandle attn_bias, // optional argument
    int compute_log_sumexp,
    double dropout_p,
    int is_causal,
    double* scale, // optional argument
    AtenTensorHandle* ret0, // returns new reference
    AtenTensorHandle* ret1, // returns new reference
    AtenTensorHandle* ret2, // returns new reference
    AtenTensorHandle* ret3 // returns new reference
);

#ifdef __cplusplus
} // extern "C"

#endif
#endif // AOTI_TORCH_SHIM_DEPRECATED
