#pragma once
#include <unordered_map>
#include <string>
#include <ATen/ATen.h>
#include <caffe2/core/context.h>
#include <caffe2/core/operator.h>
#include <caffe2/utils/math.h>
#include <iostream>

// a map from descriptor strings (see [DESCRIPTORS])
// to the key in the switch statement that implements them
static std::unordered_map<std::string, int> op_to_key = {
  { "_cast_Byte-non_blocking-1", 0 },
  { "_cast_Byte-1", 1 },
  { "_cast_Char-non_blocking-1", 2 },
  { "_cast_Char-1", 3 },
  { "_cast_Double-non_blocking-1", 4 },
  { "_cast_Double-1", 5 },
  { "_cast_Float-non_blocking-1", 6 },
  { "_cast_Float-1", 7 },
  { "_cast_Int-non_blocking-1", 8 },
  { "_cast_Int-1", 9 },
  { "_cast_Long-non_blocking-1", 10 },
  { "_cast_Long-1", 11 },
  { "_cast_Short-non_blocking-1", 12 },
  { "_cast_Short-1", 13 },
  { "_cast_Half-non_blocking-1", 14 },
  { "_cast_Half-1", 15 },
  { "data-1", 16 },
  { "is_leaf-1", 17 },
  { "output_nr-1", 18 },
  { "_version-1", 19 },
  { "align_as-2", 20 },
  { "align_tensors-*", 21 },
  { "_cudnn_ctc_loss-blank-deterministic-input_lengths-target_lengths-zero_infinity-2", 22 },
  { "_cudnn_rnn_flatten_weight-batch_first-bidirectional-hidden_size-input_size-mode-num_layers-weight_stride0-*", 23 },
  { "_cudnn_rnn-batch_first-batch_sizes-bidirectional-dropout-hidden_size-mode-num_layers-train-weight_stride0-*", 24 },
  { "_debug_has_internal_overlap-1", 25 },
  { "_fused_dropout-p-1", 26 },
  { "_masked_scale-scale-2", 27 },
  { "_reshape_from_tensor-2", 28 },
  { "_shape_as_tensor-1", 29 },
  { "dropout-p-train-1", 30 },
  { "feature_dropout-p-train-1", 31 },
  { "alpha_dropout-p-train-1", 32 },
  { "feature_alpha_dropout-p-train-1", 33 },
  { "abs-1", 34 },
  { "acos-1", 35 },
  { "avg_pool1d-ceil_mode-count_include_pad-kernel_size-padding-stride-1", 36 },
  { "avg_pool1d-ceil_mode-kernel_size-padding-stride-1", 37 },
  { "avg_pool1d-kernel_size-padding-stride-1", 38 },
  { "avg_pool1d-kernel_size-stride-1", 39 },
  { "avg_pool1d-kernel_size-1", 40 },
  { "adaptive_avg_pool1d-output_size-1", 41 },
  { "adaptive_max_pool1d-output_size-1", 42 },
  { "add-alpha-2", 43 },
  { "add-2", 44 },
  { "add-alpha-other-1", 45 },
  { "add-other-1", 46 },
  { "addmv-alpha-beta-3", 47 },
  { "addmv-beta-3", 48 },
  { "addmv-3", 49 },
  { "addr-alpha-beta-3", 50 },
  { "addr-beta-3", 51 },
  { "addr-3", 52 },
  { "affine_grid_generator-align_corners-size-1", 53 },
  { "affine_grid_generator_backward-align_corners-size-1", 54 },
  { "all-dim-keepdim-1", 55 },
  { "all-dim-1", 56 },
  { "allclose-atol-equal_nan-rtol-2", 57 },
  { "allclose-atol-rtol-2", 58 },
  { "allclose-rtol-2", 59 },
  { "allclose-2", 60 },
  { "any-dim-keepdim-1", 61 },
  { "any-dim-1", 62 },
  { "_dim_arange-dim-1", 63 },
  { "argmax-1", 64 },
  { "argmin-1", 65 },
  { "as_strided-size-stride-1", 66 },
  { "asin-1", 67 },
  { "atan-1", 68 },
  { "baddbmm-alpha-beta-3", 69 },
  { "baddbmm-beta-3", 70 },
  { "baddbmm-3", 71 },
  { "batch_norm-cudnn_enabled-eps-momentum-training-5", 72 },
  { "_batch_norm_impl_index-cudnn_enabled-eps-momentum-training-5", 73 },
  { "_batch_norm_impl_index_backward-eps-impl_index-output_mask-train-7", 74 },
  { "bernoulli-1", 75 },
  { "bernoulli-p-1", 76 },
  { "bilinear-4", 77 },
  { "binary_cross_entropy_with_logits-reduction-4", 78 },
  { "binary_cross_entropy_with_logits-4", 79 },
  { "binary_cross_entropy_with_logits-3", 80 },
  { "binary_cross_entropy_with_logits-2", 81 },
  { "binary_cross_entropy_with_logits_backward-reduction-5", 82 },
  { "binary_cross_entropy_with_logits_backward-5", 83 },
  { "binary_cross_entropy_with_logits_backward-4", 84 },
  { "binary_cross_entropy_with_logits_backward-3", 85 },
  { "bincount-minlength-2", 86 },
  { "bincount-2", 87 },
  { "bincount-1", 88 },
  { "bitwise_not-1", 89 },
  { "logical_not-1", 90 },
  { "logical_xor-2", 91 },
  { "bmm-2", 92 },
  { "broadcast_tensors-*", 93 },
  { "cat-dim-*", 94 },
  { "cat-*", 95 },
  { "ceil-1", 96 },
  { "chain_matmul-*", 97 },
  { "chunk-chunks-dim-1", 98 },
  { "chunk-chunks-1", 99 },
  { "clamp-1", 100 },
  { "clamp_max-max-1", 101 },
  { "clamp_min-min-1", 102 },
  { "cudnn_is_acceptable-1", 103 },
  { "constant_pad_nd-pad-value-1", 104 },
  { "constant_pad_nd-pad-1", 105 },
  { "contiguous-1", 106 },
  { "convolution-dilation-groups-output_padding-padding-stride-transposed-3", 107 },
  { "convolution_overrideable-dilation-groups-output_padding-padding-stride-transposed-3", 108 },
  { "convolution_backward_overrideable-dilation-groups-output_mask-output_padding-padding-stride-transposed-3", 109 },
  { "_convolution-benchmark-cudnn_enabled-deterministic-dilation-groups-output_padding-padding-stride-transposed-3", 110 },
  { "_convolution_nogroup-dilation-output_padding-padding-stride-transposed-3", 111 },
  { "_convolution_double_backward-benchmark-cudnn_enabled-deterministic-dilation-groups-output_mask-output_padding-padding-stride-transposed-6", 112 },
  { "conv1d-dilation-groups-padding-stride-3", 113 },
  { "conv1d-dilation-padding-stride-3", 114 },
  { "conv1d-padding-stride-3", 115 },
  { "conv1d-stride-3", 116 },
  { "conv1d-3", 117 },
  { "conv1d-2", 118 },
  { "conv2d-dilation-groups-padding-stride-3", 119 },
  { "conv2d-dilation-padding-stride-3", 120 },
  { "conv2d-padding-stride-3", 121 },
  { "conv2d-stride-3", 122 },
  { "conv2d-3", 123 },
  { "conv2d-2", 124 },
  { "conv3d-dilation-groups-padding-stride-3", 125 },
  { "conv3d-dilation-padding-stride-3", 126 },
  { "conv3d-padding-stride-3", 127 },
  { "conv3d-stride-3", 128 },
  { "conv3d-3", 129 },
  { "conv3d-2", 130 },
  { "conv_tbc-pad-3", 131 },
  { "conv_tbc-3", 132 },
  { "conv_tbc_backward-pad-4", 133 },
  { "conv_transpose1d-dilation-groups-output_padding-padding-stride-3", 134 },
  { "conv_transpose1d-groups-output_padding-padding-stride-3", 135 },
  { "conv_transpose1d-output_padding-padding-stride-3", 136 },
  { "conv_transpose1d-padding-stride-3", 137 },
  { "conv_transpose1d-stride-3", 138 },
  { "conv_transpose1d-3", 139 },
  { "conv_transpose1d-2", 140 },
  { "conv_transpose2d-dilation-groups-output_padding-padding-stride-3", 141 },
  { "conv_transpose2d-groups-output_padding-padding-stride-3", 142 },
  { "conv_transpose2d-output_padding-padding-stride-3", 143 },
  { "conv_transpose2d-padding-stride-3", 144 },
  { "conv_transpose2d-stride-3", 145 },
  { "conv_transpose2d-3", 146 },
  { "conv_transpose2d-2", 147 },
  { "conv_transpose3d-dilation-groups-output_padding-padding-stride-3", 148 },
  { "conv_transpose3d-groups-output_padding-padding-stride-3", 149 },
  { "conv_transpose3d-output_padding-padding-stride-3", 150 },
  { "conv_transpose3d-padding-stride-3", 151 },
  { "conv_transpose3d-stride-3", 152 },
  { "conv_transpose3d-3", 153 },
  { "conv_transpose3d-2", 154 },
  { "_copy_from-non_blocking-2", 155 },
  { "_copy_from-2", 156 },
  { "cos-1", 157 },
  { "cosh-1", 158 },
  { "cosine_embedding_loss-margin-reduction-3", 159 },
  { "cosine_embedding_loss-margin-3", 160 },
  { "cosine_embedding_loss-3", 161 },
  { "cudnn_affine_grid_generator-C-H-N-W-1", 162 },
  { "cudnn_affine_grid_generator_backward-C-H-N-W-1", 163 },
  { "cudnn_batch_norm-epsilon-exponential_average_factor-training-5", 164 },
  { "cudnn_batch_norm_backward-epsilon-7", 165 },
  { "cudnn_convolution-benchmark-deterministic-dilation-groups-padding-stride-3", 166 },
  { "cudnn_convolution_backward_input-benchmark-deterministic-dilation-groups-padding-self_size-stride-2", 167 },
  { "cudnn_convolution_backward-benchmark-deterministic-dilation-groups-output_mask-padding-stride-3", 168 },
  { "cudnn_convolution_backward_bias-1", 169 },
  { "cudnn_convolution_backward_weight-benchmark-deterministic-dilation-groups-padding-stride-weight_size-2", 170 },
  { "cudnn_convolution_transpose-benchmark-deterministic-dilation-groups-output_padding-padding-stride-3", 171 },
  { "cudnn_convolution_transpose_backward-benchmark-deterministic-dilation-groups-output_mask-output_padding-padding-stride-3", 172 },
  { "cudnn_convolution_transpose_backward_bias-1", 173 },
  { "cudnn_convolution_transpose_backward_input-benchmark-deterministic-dilation-groups-padding-stride-2", 174 },
  { "cudnn_convolution_transpose_backward_weight-benchmark-deterministic-dilation-groups-padding-stride-weight_size-2", 175 },
  { "cudnn_grid_sampler-2", 176 },
  { "cudnn_grid_sampler_backward-3", 177 },
  { "cumsum-dim-1", 178 },
  { "cumprod-dim-1", 179 },
  { "ctc_loss-blank-input_lengths-reduction-target_lengths-zero_infinity-2", 180 },
  { "ctc_loss-blank-input_lengths-reduction-target_lengths-2", 181 },
  { "ctc_loss-blank-input_lengths-target_lengths-2", 182 },
  { "ctc_loss-input_lengths-target_lengths-2", 183 },
  { "ctc_loss-blank-reduction-zero_infinity-4", 184 },
  { "ctc_loss-blank-reduction-4", 185 },
  { "ctc_loss-blank-4", 186 },
  { "ctc_loss-4", 187 },
  { "_ctc_loss-blank-input_lengths-target_lengths-zero_infinity-2", 188 },
  { "_ctc_loss-blank-input_lengths-target_lengths-2", 189 },
  { "_ctc_loss-input_lengths-target_lengths-2", 190 },
  { "_ctc_loss_backward-blank-input_lengths-target_lengths-zero_infinity-5", 191 },
  { "_ctc_loss_backward-blank-input_lengths-target_lengths-5", 192 },
  { "det-1", 193 },
  { "diag_embed-dim1-dim2-offset-1", 194 },
  { "diag_embed-dim1-offset-1", 195 },
  { "diag_embed-offset-1", 196 },
  { "diag_embed-1", 197 },
  { "diagflat-offset-1", 198 },
  { "diagflat-1", 199 },
  { "diagonal-dim1-dim2-offset-1", 200 },
  { "diagonal-dim1-offset-1", 201 },
  { "diagonal-offset-1", 202 },
  { "diagonal-1", 203 },
  { "div-2", 204 },
  { "div-other-1", 205 },
  { "dot-2", 206 },
  { "embedding-padding_idx-scale_grad_by_freq-sparse-2", 207 },
  { "embedding-padding_idx-scale_grad_by_freq-2", 208 },
  { "embedding-padding_idx-2", 209 },
  { "embedding-2", 210 },
  { "embedding_backward-num_weights-padding_idx-scale_grad_by_freq-sparse-2", 211 },
  { "embedding_dense_backward-num_weights-padding_idx-scale_grad_by_freq-2", 212 },
  { "embedding_sparse_backward-num_weights-padding_idx-scale_grad_by_freq-2", 213 },
  { "embedding_bag-mode-scale_grad_by_freq-sparse-4", 214 },
  { "embedding_bag-mode-scale_grad_by_freq-sparse-3", 215 },
  { "embedding_bag-mode-scale_grad_by_freq-3", 216 },
  { "embedding_bag-scale_grad_by_freq-3", 217 },
  { "embedding_bag-3", 218 },
  { "_embedding_bag-mode-scale_grad_by_freq-sparse-4", 219 },
  { "_embedding_bag-mode-scale_grad_by_freq-sparse-3", 220 },
  { "_embedding_bag-mode-scale_grad_by_freq-3", 221 },
  { "_embedding_bag-scale_grad_by_freq-3", 222 },
  { "_embedding_bag-3", 223 },
  { "_embedding_bag_backward-mode-num_weights-scale_grad_by_freq-sparse-7", 224 },
  { "_embedding_bag_sparse_backward-mode-num_weights-scale_grad_by_freq-6", 225 },
  { "_embedding_bag_dense_backward-mode-num_weights-scale_grad_by_freq-7", 226 },
  { "_embedding_bag_per_sample_weights_backward-mode-5", 227 },
  { "erf-1", 228 },
  { "erfc-1", 229 },
  { "exp-1", 230 },
  { "expm1-1", 231 },
  { "expand-implicit-size-1", 232 },
  { "expand-size-1", 233 },
  { "expand_as-2", 234 },
  { "flatten-end_dim-start_dim-1", 235 },
  { "flatten-start_dim-1", 236 },
  { "flatten-1", 237 },
  { "floor-1", 238 },
  { "frac-1", 239 },
  { "grid_sampler-align_corners-interpolation_mode-padding_mode-2", 240 },
  { "grid_sampler_2d-align_corners-interpolation_mode-padding_mode-2", 241 },
  { "grid_sampler_2d_backward-align_corners-interpolation_mode-padding_mode-3", 242 },
  { "grid_sampler_3d-align_corners-interpolation_mode-padding_mode-2", 243 },
  { "grid_sampler_3d_backward-align_corners-interpolation_mode-padding_mode-3", 244 },
  { "hinge_embedding_loss-margin-reduction-2", 245 },
  { "hinge_embedding_loss-margin-2", 246 },
  { "hinge_embedding_loss-2", 247 },
  { "ger-2", 248 },
  { "group_norm-cudnn_enabled-eps-num_groups-3", 249 },
  { "group_norm-eps-num_groups-3", 250 },
  { "group_norm-num_groups-3", 251 },
  { "group_norm-num_groups-2", 252 },
  { "group_norm-num_groups-1", 253 },
  { "fft-normalized-signal_ndim-1", 254 },
  { "fft-signal_ndim-1", 255 },
  { "ifft-normalized-signal_ndim-1", 256 },
  { "ifft-signal_ndim-1", 257 },
  { "rfft-normalized-onesided-signal_ndim-1", 258 },
  { "rfft-normalized-signal_ndim-1", 259 },
  { "rfft-signal_ndim-1", 260 },
  { "irfft-normalized-onesided-signal_ndim-signal_sizes-1", 261 },
  { "irfft-normalized-onesided-signal_ndim-1", 262 },
  { "irfft-normalized-signal_ndim-1", 263 },
  { "irfft-signal_ndim-1", 264 },
  { "_fft_with_size-checked_signal_sizes-complex_input-complex_output-inverse-normalized-onesided-output_sizes-signal_ndim-1", 265 },
  { "_cufft_get_plan_cache_size-device_index-0", 266 },
  { "_cufft_get_plan_cache_max_size-device_index-0", 267 },
  { "index-*", 268 },
  { "index_copy-dim-3", 269 },
  { "index_put-accumulate-*", 270 },
  { "index_put-*", 271 },
  { "instance_norm-cudnn_enabled-eps-momentum-use_input_stats-5", 272 },
  { "inverse-1", 273 },
  { "_inverse_helper-1", 274 },
  { "isclose-atol-equal_nan-rtol-2", 275 },
  { "isclose-atol-rtol-2", 276 },
  { "isclose-rtol-2", 277 },
  { "isclose-2", 278 },
  { "isnan-1", 279 },
  { "is_distributed-1", 280 },
  { "is_floating_point-1", 281 },
  { "is_complex-1", 282 },
  { "is_nonzero-1", 283 },
  { "is_same_size-2", 284 },
  { "is_signed-1", 285 },
  { "kl_div-reduction-2", 286 },
  { "kl_div-2", 287 },
  { "kl_div_backward-reduction-3", 288 },
  { "kl_div_backward-3", 289 },
  { "kthvalue-dim-k-keepdim-1", 290 },
  { "kthvalue-dim-k-1", 291 },
  { "kthvalue-k-1", 292 },
  { "layer_norm-cudnn_enable-eps-normalized_shape-3", 293 },
  { "layer_norm-eps-normalized_shape-3", 294 },
  { "layer_norm-normalized_shape-3", 295 },
  { "layer_norm-normalized_shape-2", 296 },
  { "layer_norm-normalized_shape-1", 297 },
  { "native_layer_norm-M-N-eps-3", 298 },
  { "native_layer_norm_backward-M-N-output_mask-5", 299 },
  { "native_layer_norm_double_backward-M-N-output_mask-8", 300 },
  { "linear-3", 301 },
  { "linear-2", 302 },
  { "mkldnn_linear-3", 303 },
  { "mkldnn_linear-2", 304 },
  { "fbgemm_linear_int8_weight_fp32_activation-weight_scale-weight_zero_point-5", 305 },
  { "fbgemm_linear_int8_weight-weight_scale-weight_zero_point-5", 306 },
  { "fbgemm_pack_gemm_matrix_fp16-1", 307 },
  { "fbgemm_linear_fp16_weight_fp32_activation-3", 308 },
  { "fbgemm_linear_fp16_weight-3", 309 },
  { "fbgemm_pack_quantized_matrix-1", 310 },
  { "fbgemm_pack_quantized_matrix-K-N-1", 311 },
  { "fbgemm_is_cpu_supported-0", 312 },
  { "log-1", 313 },
  { "log10-1", 314 },
  { "log1p-1", 315 },
  { "log2-1", 316 },
  { "logdet-1", 317 },
  { "log_softmax-dim-1", 318 },
  { "_log_softmax-dim-half_to_float-1", 319 },
  { "_log_softmax_backward_data-dim-3", 320 },
  { "logsumexp-dim-keepdim-1", 321 },
  { "logsumexp-dim-1", 322 },
  { "margin_ranking_loss-margin-reduction-3", 323 },
  { "margin_ranking_loss-margin-3", 324 },
  { "margin_ranking_loss-3", 325 },
  { "matmul-2", 326 },
  { "matrix_rank-symmetric-tol-1", 327 },
  { "matrix_rank-tol-1", 328 },
  { "matrix_rank-symmetric-1", 329 },
  { "matrix_rank-1", 330 },
  { "matrix_power-n-1", 331 },
  { "max-dim-keepdim-1", 332 },
  { "max-dim-1", 333 },
  { "max_values-dim-keepdim-1", 334 },
  { "max_values-dim-1", 335 },
  { "max_pool1d_with_indices-ceil_mode-dilation-kernel_size-padding-stride-1", 336 },
  { "max_pool1d_with_indices-dilation-kernel_size-padding-stride-1", 337 },
  { "max_pool1d_with_indices-kernel_size-padding-stride-1", 338 },
  { "max_pool1d_with_indices-kernel_size-stride-1", 339 },
  { "max_pool1d_with_indices-kernel_size-1", 340 },
  { "max_pool1d-ceil_mode-dilation-kernel_size-padding-stride-1", 341 },
  { "max_pool1d-dilation-kernel_size-padding-stride-1", 342 },
  { "max_pool1d-kernel_size-padding-stride-1", 343 },
  { "max_pool1d-kernel_size-stride-1", 344 },
  { "max_pool1d-kernel_size-1", 345 },
  { "max_pool2d-ceil_mode-dilation-kernel_size-padding-stride-1", 346 },
  { "max_pool2d-dilation-kernel_size-padding-stride-1", 347 },
  { "max_pool2d-kernel_size-padding-stride-1", 348 },
  { "max_pool2d-kernel_size-stride-1", 349 },
  { "max_pool2d-kernel_size-1", 350 },
  { "mkldnn_max_pool2d-ceil_mode-dilation-kernel_size-padding-stride-1", 351 },
  { "mkldnn_max_pool2d-dilation-kernel_size-padding-stride-1", 352 },
  { "mkldnn_max_pool2d-kernel_size-padding-stride-1", 353 },
  { "mkldnn_max_pool2d-kernel_size-stride-1", 354 },
  { "mkldnn_max_pool2d-kernel_size-1", 355 },
  { "quantized_max_pool2d-dilation-kernel_size-padding-stride-1", 356 },
  { "quantized_max_pool2d-kernel_size-padding-stride-1", 357 },
  { "quantized_max_pool2d-kernel_size-stride-1", 358 },
  { "quantized_max_pool2d-kernel_size-1", 359 },
  { "max_pool3d-ceil_mode-dilation-kernel_size-padding-stride-1", 360 },
  { "max_pool3d-dilation-kernel_size-padding-stride-1", 361 },
  { "max_pool3d-kernel_size-padding-stride-1", 362 },
  { "max_pool3d-kernel_size-stride-1", 363 },
  { "max_pool3d-kernel_size-1", 364 },
  { "mean-1", 365 },
  { "mean-dim-keepdim-1", 366 },
  { "mean-dim-1", 367 },
  { "median-dim-keepdim-1", 368 },
  { "median-dim-1", 369 },
  { "min-dim-keepdim-1", 370 },
  { "min-dim-1", 371 },
  { "min_values-dim-keepdim-1", 372 },
  { "min_values-dim-1", 373 },
  { "mkldnn_convolution-dilation-groups-padding-stride-3", 374 },
  { "mkldnn_convolution_backward_input-bias_defined-dilation-groups-padding-self_size-stride-2", 375 },
  { "mkldnn_convolution_backward_weights-bias_defined-dilation-groups-padding-stride-weight_size-2", 376 },
  { "mkldnn_convolution_backward-dilation-groups-output_mask-padding-stride-3", 377 },
  { "miopen_batch_norm-epsilon-exponential_average_factor-training-5", 378 },
  { "miopen_batch_norm_backward-epsilon-7", 379 },
  { "miopen_convolution-benchmark-deterministic-dilation-groups-padding-stride-3", 380 },
  { "miopen_convolution_backward_input-benchmark-deterministic-dilation-groups-padding-self_size-stride-2", 381 },
  { "miopen_convolution_backward-benchmark-deterministic-dilation-groups-output_mask-padding-stride-3", 382 },
  { "miopen_convolution_backward_bias-1", 383 },
  { "miopen_convolution_backward_weight-benchmark-deterministic-dilation-groups-padding-stride-weight_size-2", 384 },
  { "miopen_convolution_transpose-benchmark-deterministic-dilation-groups-output_padding-padding-stride-3", 385 },
  { "miopen_convolution_transpose_backward-benchmark-deterministic-dilation-groups-output_mask-output_padding-padding-stride-3", 386 },
  { "miopen_convolution_transpose_backward_input-benchmark-deterministic-dilation-groups-padding-stride-2", 387 },
  { "miopen_convolution_transpose_backward_weight-benchmark-deterministic-dilation-groups-padding-stride-weight_size-2", 388 },
  { "miopen_depthwise_convolution-benchmark-deterministic-dilation-groups-padding-stride-3", 389 },
  { "miopen_depthwise_convolution_backward_input-benchmark-deterministic-dilation-groups-padding-self_size-stride-2", 390 },
  { "miopen_depthwise_convolution_backward-benchmark-deterministic-dilation-groups-output_mask-padding-stride-3", 391 },
  { "miopen_depthwise_convolution_backward_weight-benchmark-deterministic-dilation-groups-padding-stride-weight_size-2", 392 },
  { "miopen_rnn-batch_first-batch_sizes-bidirectional-dropout-hidden_size-mode-num_layers-train-weight_stride0-*", 393 },
  { "mm-2", 394 },
  { "_sparse_mm-2", 395 },
  { "mode-dim-keepdim-1", 396 },
  { "mode-dim-1", 397 },
  { "mode-1", 398 },
  { "mul-2", 399 },
  { "mul-other-1", 400 },
  { "mv-2", 401 },
  { "mvlgamma-p-1", 402 },
  { "narrow_copy-dim-length-start-1", 403 },
  { "narrow-dim-length-start-1", 404 },
  { "native_batch_norm-eps-momentum-training-5", 405 },
  { "batch_norm_stats-eps-1", 406 },
  { "batch_norm_elemt-eps-5", 407 },
  { "batch_norm_gather_stats-count-eps-momentum-5", 408 },
  { "batch_norm_gather_stats_with_counts-counts-eps-momentum-5", 409 },
  { "native_batch_norm_backward-eps-output_mask-train-7", 410 },
  { "batch_norm_backward_reduce-bias_g-input_g-weight_g-5", 411 },
  { "batch_norm_backward_elemt-7", 412 },
  { "batch_norm_update_stats-momentum-3", 413 },
  { "_nnpack_available-0", 414 },
  { "_nnpack_spatial_convolution-padding-3", 415 },
  { "_nnpack_spatial_convolution_backward-output_mask-padding-3", 416 },
  { "_nnpack_spatial_convolution_backward_input-padding-3", 417 },
  { "_nnpack_spatial_convolution_backward_weight-padding-weightsize-2", 418 },
  { "pairwise_distance-eps-keepdim-p-2", 419 },
  { "pairwise_distance-eps-p-2", 420 },
  { "pairwise_distance-p-2", 421 },
  { "pairwise_distance-2", 422 },
  { "cdist-p-2", 423 },
  { "cdist-2", 424 },
  { "_cdist_backward-p-4", 425 },
  { "pdist-p-1", 426 },
  { "pdist-1", 427 },
  { "_pdist_forward-p-1", 428 },
  { "_pdist_forward-1", 429 },
  { "_pdist_backward-p-3", 430 },
  { "cosine_similarity-dim-eps-2", 431 },
  { "cosine_similarity-dim-2", 432 },
  { "cosine_similarity-2", 433 },
  { "permute-dims-1", 434 },
  { "numpy_T-1", 435 },
  { "pixel_shuffle-upscale_factor-1", 436 },
  { "is_pinned-1", 437 },
  { "pin_memory-1", 438 },
  { "pinverse-rcond-1", 439 },
  { "pinverse-1", 440 },
  { "poisson_nll_loss-eps-full-log_input-reduction-2", 441 },
  { "reciprocal-1", 442 },
  { "neg-1", 443 },
  { "repeat-repeats-1", 444 },
  { "repeat_interleave-1", 445 },
  { "repeat_interleave-2", 446 },
  { "repeat_interleave-repeats-1", 447 },
  { "reshape-shape-1", 448 },
  { "_mkldnn_reshape-shape-1", 449 },
  { "reshape_as-2", 450 },
  { "round-1", 451 },
  { "rrelu-lower-training-upper-1", 452 },
  { "rrelu-lower-upper-1", 453 },
  { "rrelu-lower-1", 454 },
  { "rrelu-1", 455 },
  { "relu-1", 456 },
  { "prelu-2", 457 },
  { "prelu_backward-3", 458 },
  { "gelu-1", 459 },
  { "gelu_backward-2", 460 },
  { "hardshrink-lambd-1", 461 },
  { "hardshrink-1", 462 },
  { "hardshrink_backward-lambd-2", 463 },
  { "rsqrt-1", 464 },
  { "select-dim-index-1", 465 },
  { "selu-1", 466 },
  { "celu-alpha-1", 467 },
  { "celu-1", 468 },
  { "sigmoid-1", 469 },
  { "sin-1", 470 },
  { "sinh-1", 471 },
  { "detach-1", 472 },
  { "size-dim-1", 473 },
  { "slice-dim-end-start-step-1", 474 },
  { "slice-dim-end-start-1", 475 },
  { "slice-dim-start-1", 476 },
  { "slice-dim-1", 477 },
  { "slice-1", 478 },
  { "slogdet-1", 479 },
  { "smm-2", 480 },
  { "softmax-dim-1", 481 },
  { "_softmax-dim-half_to_float-1", 482 },
  { "_softmax_backward_data-dim-3", 483 },
  { "split-dim-split_size-1", 484 },
  { "split-split_size-1", 485 },
  { "split_with_sizes-dim-split_sizes-1", 486 },
  { "split_with_sizes-split_sizes-1", 487 },
  { "squeeze-1", 488 },
  { "squeeze-dim-1", 489 },
  { "sspaddmm-alpha-beta-3", 490 },
  { "sspaddmm-beta-3", 491 },
  { "sspaddmm-3", 492 },
  { "stack-dim-*", 493 },
  { "stack-*", 494 },
  { "stft-n_fft-1", 495 },
  { "stride-dim-1", 496 },
  { "sum-1", 497 },
  { "sum-dim-keepdim-1", 498 },
  { "sum-dim-1", 499 },
  { "sum_to_size-size-1", 500 },
  { "sqrt-1", 501 },
  { "std-unbiased-1", 502 },
  { "std-1", 503 },
  { "std-dim-keepdim-unbiased-1", 504 },
  { "std-dim-unbiased-1", 505 },
  { "std-dim-1", 506 },
  { "std_mean-unbiased-1", 507 },
  { "std_mean-1", 508 },
  { "std_mean-dim-keepdim-unbiased-1", 509 },
  { "std_mean-dim-unbiased-1", 510 },
  { "std_mean-dim-1", 511 },
  { "prod-1", 512 },
  { "prod-dim-keepdim-1", 513 },
  { "prod-dim-1", 514 },
  { "t-1", 515 },
  { "tan-1", 516 },
  { "tanh-1", 517 },
  { "tensordot-dims_other-dims_self-2", 518 },
  { "threshold-threshold-value-1", 519 },
  { "threshold_backward-threshold-2", 520 },
  { "transpose-dim0-dim1-1", 521 },
  { "_mkldnn_transpose-dim0-dim1-1", 522 },
  { "one_hot-num_classes-1", 523 },
  { "one_hot-1", 524 },
  { "flip-dims-1", 525 },
  { "roll-dims-shifts-1", 526 },
  { "roll-shifts-1", 527 },
  { "rot90-dims-k-1", 528 },
  { "rot90-k-1", 529 },
  { "rot90-1", 530 },
  { "trapz-dim-2", 531 },
  { "trapz-2", 532 },
  { "trapz-dim-dx-1", 533 },
  { "trapz-dx-1", 534 },
  { "trapz-1", 535 },
  { "_trilinear-expand1-expand2-expand3-sumdim-unroll_dim-3", 536 },
  { "_trilinear-expand1-expand2-expand3-sumdim-3", 537 },
  { "triplet_margin_loss-eps-margin-p-reduction-swap-3", 538 },
  { "triplet_margin_loss-eps-margin-p-swap-3", 539 },
  { "triplet_margin_loss-eps-margin-p-3", 540 },
  { "triplet_margin_loss-margin-p-3", 541 },
  { "triplet_margin_loss-margin-3", 542 },
  { "triplet_margin_loss-3", 543 },
  { "trunc-1", 544 },
  { "type_as-2", 545 },
  { "_has_compatible_shallow_copy_type-2", 546 },
  { "_unique-return_inverse-sorted-1", 547 },
  { "_unique-sorted-1", 548 },
  { "_unique-1", 549 },
  { "unique_dim-dim-return_counts-return_inverse-sorted-1", 550 },
  { "unique_dim-dim-return_inverse-sorted-1", 551 },
  { "unique_dim-dim-sorted-1", 552 },
  { "unique_dim-dim-1", 553 },
  { "unique_consecutive-return_counts-return_inverse-1", 554 },
  { "unique_consecutive-return_inverse-1", 555 },
  { "unique_consecutive-1", 556 },
  { "unique_dim_consecutive-dim-return_counts-return_inverse-1", 557 },
  { "unique_dim_consecutive-dim-return_inverse-1", 558 },
  { "unique_dim_consecutive-dim-1", 559 },
  { "_unique2-return_counts-return_inverse-sorted-1", 560 },
  { "_unique2-return_inverse-sorted-1", 561 },
  { "_unique2-sorted-1", 562 },
  { "_unique2-1", 563 },
  { "_unsafe_view-size-1", 564 },
  { "unsqueeze-dim-1", 565 },
  { "var-unbiased-1", 566 },
  { "var-1", 567 },
  { "var-dim-keepdim-unbiased-1", 568 },
  { "var-dim-unbiased-1", 569 },
  { "var-dim-1", 570 },
  { "var_mean-unbiased-1", 571 },
  { "var_mean-1", 572 },
  { "var_mean-dim-keepdim-unbiased-1", 573 },
  { "var_mean-dim-unbiased-1", 574 },
  { "var_mean-dim-1", 575 },
  { "view_as-2", 576 },
  { "where-3", 577 },
  { "where-1", 578 },
  { "_s_where-3", 579 },
  { "norm_except_dim-dim-pow-1", 580 },
  { "norm_except_dim-pow-1", 581 },
  { "norm_except_dim-1", 582 },
  { "_weight_norm-dim-2", 583 },
  { "_weight_norm-2", 584 },
  { "_weight_norm_cuda_interface-dim-2", 585 },
  { "_weight_norm_cuda_interface-2", 586 },
  { "_weight_norm_cuda_interface_backward-dim-4", 587 },
  { "_weight_norm_differentiable_backward-dim-4", 588 },
  { "_standard_gamma_grad-2", 589 },
  { "_standard_gamma-1", 590 },
  { "_dirichlet_grad-3", 591 },
  { "_sample_dirichlet-1", 592 },
  { "poisson-1", 593 },
  { "native_norm-p-1", 594 },
  { "native_norm-1", 595 },
  { "_sparse_sum-1", 596 },
  { "_sparse_sum-dim-1", 597 },
  { "_sparse_sum_backward-dim-2", 598 },
  { "norm-p-1", 599 },
  { "norm-1", 600 },
  { "frobenius_norm-1", 601 },
  { "frobenius_norm-dim-keepdim-1", 602 },
  { "frobenius_norm-dim-1", 603 },
  { "nuclear_norm-keepdim-1", 604 },
  { "nuclear_norm-1", 605 },
  { "nuclear_norm-dim-keepdim-1", 606 },
  { "nuclear_norm-dim-1", 607 },
  { "clone-1", 608 },
  { "pow-exponent-1", 609 },
  { "sub-alpha-2", 610 },
  { "sub-2", 611 },
  { "sub-alpha-other-1", 612 },
  { "sub-other-1", 613 },
  { "rsub-alpha-2", 614 },
  { "rsub-2", 615 },
  { "rsub-alpha-other-1", 616 },
  { "rsub-other-1", 617 },
  { "_sparse_addmm-alpha-beta-3", 618 },
  { "_sparse_addmm-beta-3", 619 },
  { "_sparse_addmm-3", 620 },
  { "addmm-alpha-beta-3", 621 },
  { "addmm-beta-3", 622 },
  { "addmm-3", 623 },
  { "sparse_mask-2", 624 },
  { "to_dense-1", 625 },
  { "to_dense_backward-2", 626 },
  { "sparse_dim-1", 627 },
  { "_dimI-1", 628 },
  { "dense_dim-1", 629 },
  { "_dimV-1", 630 },
  { "_nnz-1", 631 },
  { "coalesce-1", 632 },
  { "is_coalesced-1", 633 },
  { "_indices-1", 634 },
  { "_values-1", 635 },
  { "indices-1", 636 },
  { "values-1", 637 },
  { "hspmm-2", 638 },
  { "numel-1", 639 },
  { "unbind-dim-1", 640 },
  { "unbind-1", 641 },
  { "to_sparse-sparse_dim-1", 642 },
  { "to_sparse-1", 643 },
  { "to_mkldnn-1", 644 },
  { "mkldnn_reorder_conv2d_weight-dilation-groups-padding-stride-1", 645 },
  { "mkldnn_reorder_conv2d_weight-dilation-padding-stride-1", 646 },
  { "mkldnn_reorder_conv2d_weight-padding-stride-1", 647 },
  { "mkldnn_reorder_conv2d_weight-padding-1", 648 },
  { "mkldnn_reorder_conv2d_weight-1", 649 },
  { "to_mkldnn_backward-2", 650 },
  { "dequantize-1", 651 },
  { "q_zero_point-1", 652 },
  { "q_per_channel_scales-1", 653 },
  { "q_per_channel_zero_points-1", 654 },
  { "q_per_channel_axis-1", 655 },
  { "int_repr-1", 656 },
  { "_make_per_tensor_quantized_tensor-scale-zero_point-1", 657 },
  { "_make_per_channel_quantized_tensor-axis-3", 658 },
  { "fake_quantize_per_tensor_affine-quant_max-quant_min-scale-zero_point-1", 659 },
  { "fake_quantize_per_tensor_affine_backward-quant_max-quant_min-scale-zero_point-2", 660 },
  { "meshgrid-*", 661 },
  { "cartesian_prod-*", 662 },
  { "combinations-r-with_replacement-1", 663 },
  { "combinations-r-1", 664 },
  { "combinations-1", 665 },
  { "item-1", 666 },
  { "_local_scalar_dense-1", 667 },
  { "_thnn_fused_lstm_cell-5", 668 },
  { "_thnn_fused_lstm_cell-4", 669 },
  { "_thnn_fused_lstm_cell-3", 670 },
  { "_thnn_fused_lstm_cell_backward-has_bias-5", 671 },
  { "_thnn_fused_gru_cell-5", 672 },
  { "_thnn_fused_gru_cell-4", 673 },
  { "_thnn_fused_gru_cell-3", 674 },
  { "_thnn_fused_gru_cell_backward-has_bias-2", 675 },
  { "lstm-batch_first-bidirectional-dropout-has_biases-num_layers-train-*", 676 },
  { "lstm-bidirectional-dropout-has_biases-num_layers-train-*", 677 },
  { "gru-batch_first-bidirectional-dropout-has_biases-num_layers-train-*", 678 },
  { "gru-bidirectional-dropout-has_biases-num_layers-train-*", 679 },
  { "rnn_tanh-batch_first-bidirectional-dropout-has_biases-num_layers-train-*", 680 },
  { "rnn_tanh-bidirectional-dropout-has_biases-num_layers-train-*", 681 },
  { "rnn_relu-batch_first-bidirectional-dropout-has_biases-num_layers-train-*", 682 },
  { "rnn_relu-bidirectional-dropout-has_biases-num_layers-train-*", 683 },
  { "lstm_cell-*", 684 },
  { "gru_cell-6", 685 },
  { "gru_cell-5", 686 },
  { "gru_cell-4", 687 },
  { "rnn_tanh_cell-6", 688 },
  { "rnn_tanh_cell-5", 689 },
  { "rnn_tanh_cell-4", 690 },
  { "rnn_relu_cell-6", 691 },
  { "rnn_relu_cell-5", 692 },
  { "rnn_relu_cell-4", 693 },
  { "quantized_lstm-batch_first-bidirectional-dropout-has_biases-num_layers-train-*", 694 },
  { "quantized_gru-batch_first-bidirectional-dropout-has_biases-num_layers-train-*", 695 },
  { "quantized_gru-bidirectional-dropout-has_biases-num_layers-train-*", 696 },
  { "quantized_lstm_cell-scale_hh-scale_ih-zero_point_hh-zero_point_ih-*", 697 },
  { "quantized_gru_cell-scale_hh-scale_ih-zero_point_hh-zero_point_ih-10", 698 },
  { "quantized_rnn_relu_cell-scale_hh-scale_ih-zero_point_hh-zero_point_ih-10", 699 },
  { "quantized_rnn_tanh_cell-scale_hh-scale_ih-zero_point_hh-zero_point_ih-10", 700 },
  { "_pack_padded_sequence-batch_first-2", 701 },
  { "_pack_padded_sequence_backward-batch_first-input_size-2", 702 },
  { "_pad_packed_sequence-batch_first-padding_value-total_length-2", 703 },
  { "is_set_to-2", 704 },
  { "masked_fill-value-2", 705 },
  { "masked_fill-3", 706 },
  { "masked_scatter-3", 707 },
  { "view-size-1", 708 },
  { "index_add-dim-3", 709 },
  { "index_fill-dim-value-2", 710 },
  { "index_fill-dim-3", 711 },
  { "scatter-dim-3", 712 },
  { "scatter-dim-value-2", 713 },
  { "scatter_add-dim-3", 714 },
  { "__and__-other-1", 715 },
  { "__and__-2", 716 },
  { "__or__-other-1", 717 },
  { "__or__-2", 718 },
  { "__xor__-other-1", 719 },
  { "__xor__-2", 720 },
  { "__lshift__-other-1", 721 },
  { "__lshift__-2", 722 },
  { "__rshift__-other-1", 723 },
  { "__rshift__-2", 724 },
  { "addbmm-alpha-beta-3", 725 },
  { "addbmm-beta-3", 726 },
  { "addbmm-3", 727 },
  { "diag-diagonal-1", 728 },
  { "diag-1", 729 },
  { "cross-2", 730 },
  { "triu-diagonal-1", 731 },
  { "triu-1", 732 },
  { "tril-diagonal-1", 733 },
  { "tril-1", 734 },
  { "trace-1", 735 },
  { "ne-other-1", 736 },
  { "ne-2", 737 },
  { "eq-other-1", 738 },
  { "eq-2", 739 },
  { "ge-other-1", 740 },
  { "ge-2", 741 },
  { "le-other-1", 742 },
  { "le-2", 743 },
  { "gt-other-1", 744 },
  { "gt-2", 745 },
  { "lt-other-1", 746 },
  { "lt-2", 747 },
  { "take-2", 748 },
  { "index_select-dim-2", 749 },
  { "masked_select-2", 750 },
  { "nonzero-1", 751 },
  { "nonzero_numpy-1", 752 },
  { "gather-dim-sparse_grad-2", 753 },
  { "gather-dim-2", 754 },
  { "_gather_sparse_backward-dim-3", 755 },
  { "addcmul-value-3", 756 },
  { "addcmul-3", 757 },
  { "addcdiv-value-3", 758 },
  { "addcdiv-3", 759 },
  { "lstsq-2", 760 },
  { "triangular_solve-transpose-unitriangular-upper-2", 761 },
  { "triangular_solve-transpose-upper-2", 762 },
  { "triangular_solve-upper-2", 763 },
  { "triangular_solve-2", 764 },
  { "_triangular_solve_helper-transpose-unitriangular-upper-2", 765 },
  { "symeig-eigenvectors-upper-1", 766 },
  { "symeig-eigenvectors-1", 767 },
  { "symeig-1", 768 },
  { "_symeig_helper-eigenvectors-upper-1", 769 },
  { "eig-eigenvectors-1", 770 },
  { "eig-1", 771 },
  { "svd-compute_uv-some-1", 772 },
  { "svd-some-1", 773 },
  { "svd-1", 774 },
  { "_svd_helper-compute_uv-some-1", 775 },
  { "cholesky-upper-1", 776 },
  { "cholesky-1", 777 },
  { "_cholesky_helper-upper-1", 778 },
  { "cholesky_solve-upper-2", 779 },
  { "cholesky_solve-2", 780 },
  { "_cholesky_solve_helper-upper-2", 781 },
  { "solve-2", 782 },
  { "_solve_helper-2", 783 },
  { "cholesky_inverse-upper-1", 784 },
  { "cholesky_inverse-1", 785 },
  { "qr-some-1", 786 },
  { "qr-1", 787 },
  { "_qr_helper-some-1", 788 },
  { "geqrf-1", 789 },
  { "orgqr-2", 790 },
  { "ormqr-left-transpose-3", 791 },
  { "ormqr-left-3", 792 },
  { "ormqr-3", 793 },
  { "_lu_with_info-check_errors-pivot-1", 794 },
  { "_lu_with_info-pivot-1", 795 },
  { "_lu_with_info-1", 796 },
  { "lu_solve-3", 797 },
  { "_lu_solve_helper-3", 798 },
  { "multinomial-num_samples-replacement-1", 799 },
  { "multinomial-num_samples-1", 800 },
  { "_multinomial_alias_setup-1", 801 },
  { "_multinomial_alias_draw-num_samples-2", 802 },
  { "lgamma-1", 803 },
  { "digamma-1", 804 },
  { "polygamma-n-1", 805 },
  { "erfinv-1", 806 },
  { "sign-1", 807 },
  { "dist-p-2", 808 },
  { "dist-2", 809 },
  { "atan2-2", 810 },
  { "lerp-weight-2", 811 },
  { "lerp-3", 812 },
  { "histc-bins-max-min-1", 813 },
  { "histc-bins-min-1", 814 },
  { "histc-bins-1", 815 },
  { "histc-1", 816 },
  { "fmod-other-1", 817 },
  { "fmod-2", 818 },
  { "remainder-other-1", 819 },
  { "remainder-2", 820 },
  { "min-2", 821 },
  { "min-1", 822 },
  { "max-2", 823 },
  { "max-1", 824 },
  { "median-1", 825 },
  { "sort-descending-dim-1", 826 },
  { "sort-dim-1", 827 },
  { "sort-1", 828 },
  { "argsort-descending-dim-1", 829 },
  { "argsort-dim-1", 830 },
  { "argsort-1", 831 },
  { "topk-dim-k-largest-sorted-1", 832 },
  { "topk-dim-k-largest-1", 833 },
  { "topk-dim-k-1", 834 },
  { "topk-k-1", 835 },
  { "all-1", 836 },
  { "any-1", 837 },
  { "renorm-dim-maxnorm-p-1", 838 },
  { "unfold-dimension-size-step-1", 839 },
  { "equal-2", 840 },
  { "pow-2", 841 },
  { "pow-self-1", 842 },
  { "alias-1", 843 },
  { "_addr-alpha-beta-3", 844 },
  { "_addr-beta-3", 845 },
  { "_addr-3", 846 },
  { "_cumsum-dim-1", 847 },
  { "_cumprod-dim-1", 848 },
  { "_var-unbiased-1", 849 },
  { "_var-1", 850 },
  { "_std-unbiased-1", 851 },
  { "_std-1", 852 },
  { "_cat-dim-*", 853 },
  { "_cat-*", 854 },
  { "_mode-dim-keepdim-1", 855 },
  { "_mode-dim-1", 856 },
  { "_mode-1", 857 },
  { "_max-dim-keepdim-1", 858 },
  { "_max-dim-1", 859 },
  { "_min-dim-keepdim-1", 860 },
  { "_min-dim-1", 861 },
  { "binary_cross_entropy-reduction-3", 862 },
  { "binary_cross_entropy-3", 863 },
  { "binary_cross_entropy-2", 864 },
  { "binary_cross_entropy_backward-reduction-4", 865 },
  { "binary_cross_entropy_backward-4", 866 },
  { "binary_cross_entropy_backward-3", 867 },
  { "mse_loss-reduction-2", 868 },
  { "mse_loss-2", 869 },
  { "mse_loss_backward-reduction-3", 870 },
  { "l1_loss-reduction-2", 871 },
  { "l1_loss-2", 872 },
  { "l1_loss_backward-reduction-3", 873 },
  { "multi_margin_loss-margin-p-reduction-3", 874 },
  { "multi_margin_loss-margin-p-3", 875 },
  { "multi_margin_loss-margin-p-2", 876 },
  { "multi_margin_loss-p-2", 877 },
  { "multi_margin_loss-2", 878 },
  { "multi_margin_loss_backward-margin-p-reduction-4", 879 },
  { "multi_margin_loss_backward-margin-p-4", 880 },
  { "multi_margin_loss_backward-margin-p-3", 881 },
  { "multilabel_margin_loss-reduction-2", 882 },
  { "multilabel_margin_loss-2", 883 },
  { "multilabel_margin_loss_forward-reduction-2", 884 },
  { "multilabel_margin_loss_backward-reduction-4", 885 },
  { "nll_loss-ignore_index-reduction-3", 886 },
  { "nll_loss-reduction-3", 887 },
  { "nll_loss-3", 888 },
  { "nll_loss-2", 889 },
  { "nll_loss_forward-ignore_index-reduction-3", 890 },
  { "nll_loss_backward-ignore_index-reduction-5", 891 },
  { "nll_loss2d-ignore_index-reduction-3", 892 },
  { "nll_loss2d-reduction-3", 893 },
  { "nll_loss2d-3", 894 },
  { "nll_loss2d-2", 895 },
  { "nll_loss2d_forward-ignore_index-reduction-3", 896 },
  { "nll_loss2d_backward-ignore_index-reduction-5", 897 },
  { "smooth_l1_loss-reduction-2", 898 },
  { "smooth_l1_loss-2", 899 },
  { "smooth_l1_loss_backward-reduction-3", 900 },
  { "soft_margin_loss-reduction-2", 901 },
  { "soft_margin_loss-2", 902 },
  { "soft_margin_loss_backward-reduction-3", 903 },
  { "elu-alpha-input_scale-scale-1", 904 },
  { "elu-alpha-scale-1", 905 },
  { "elu-alpha-1", 906 },
  { "elu-1", 907 },
  { "elu_backward-alpha-input_scale-scale-2", 908 },
  { "glu-dim-1", 909 },
  { "glu-1", 910 },
  { "glu_backward-dim-2", 911 },
  { "hardtanh-max_val-min_val-1", 912 },
  { "hardtanh-min_val-1", 913 },
  { "hardtanh-1", 914 },
  { "hardtanh_backward-max_val-min_val-2", 915 },
  { "leaky_relu-negative_slope-1", 916 },
  { "leaky_relu-1", 917 },
  { "leaky_relu_backward-negative_slope-2", 918 },
  { "log_sigmoid-1", 919 },
  { "log_sigmoid_forward-1", 920 },
  { "log_sigmoid_backward-3", 921 },
  { "rrelu_with_noise-lower-training-upper-2", 922 },
  { "rrelu_with_noise-lower-upper-2", 923 },
  { "rrelu_with_noise-lower-2", 924 },
  { "rrelu_with_noise-2", 925 },
  { "rrelu_with_noise_backward-lower-training-upper-3", 926 },
  { "softplus-beta-threshold-1", 927 },
  { "softplus-beta-1", 928 },
  { "softplus-1", 929 },
  { "softplus_backward-beta-threshold-3", 930 },
  { "softshrink-lambd-1", 931 },
  { "softshrink-1", 932 },
  { "softshrink_backward-lambd-2", 933 },
  { "adaptive_avg_pool2d-output_size-1", 934 },
  { "mkldnn_adaptive_avg_pool2d-output_size-1", 935 },
  { "_adaptive_avg_pool2d-output_size-1", 936 },
  { "_adaptive_avg_pool2d_backward-2", 937 },
  { "adaptive_avg_pool3d-output_size-1", 938 },
  { "adaptive_avg_pool3d_backward-2", 939 },
  { "adaptive_max_pool2d-output_size-1", 940 },
  { "adaptive_max_pool2d_backward-3", 941 },
  { "adaptive_max_pool3d-output_size-1", 942 },
  { "adaptive_max_pool3d_backward-3", 943 },
  { "avg_pool2d-ceil_mode-count_include_pad-kernel_size-padding-stride-1", 944 },
  { "avg_pool2d-ceil_mode-kernel_size-padding-stride-1", 945 },
  { "avg_pool2d-kernel_size-padding-stride-1", 946 },
  { "avg_pool2d-kernel_size-stride-1", 947 },
  { "avg_pool2d-kernel_size-1", 948 },
  { "avg_pool3d-ceil_mode-count_include_pad-kernel_size-padding-stride-1", 949 },
  { "avg_pool3d-ceil_mode-kernel_size-padding-stride-1", 950 },
  { "avg_pool3d-kernel_size-padding-stride-1", 951 },
  { "avg_pool3d-kernel_size-stride-1", 952 },
  { "avg_pool3d-kernel_size-1", 953 },
  { "fractional_max_pool2d-kernel_size-output_size-2", 954 },
  { "fractional_max_pool2d_backward-kernel_size-output_size-3", 955 },
  { "fractional_max_pool3d-kernel_size-output_size-2", 956 },
  { "fractional_max_pool3d_backward-kernel_size-output_size-3", 957 },
  { "max_pool2d_with_indices-ceil_mode-dilation-kernel_size-padding-stride-1", 958 },
  { "max_pool2d_with_indices-dilation-kernel_size-padding-stride-1", 959 },
  { "max_pool2d_with_indices-kernel_size-padding-stride-1", 960 },
  { "max_pool2d_with_indices-kernel_size-stride-1", 961 },
  { "max_pool2d_with_indices-kernel_size-1", 962 },
  { "max_pool2d_with_indices_backward-ceil_mode-dilation-kernel_size-padding-stride-3", 963 },
  { "max_pool3d_with_indices-ceil_mode-dilation-kernel_size-padding-stride-1", 964 },
  { "max_pool3d_with_indices-dilation-kernel_size-padding-stride-1", 965 },
  { "max_pool3d_with_indices-kernel_size-padding-stride-1", 966 },
  { "max_pool3d_with_indices-kernel_size-stride-1", 967 },
  { "max_pool3d_with_indices-kernel_size-1", 968 },
  { "max_pool3d_with_indices_backward-ceil_mode-dilation-kernel_size-padding-stride-3", 969 },
  { "max_unpool2d-output_size-2", 970 },
  { "max_unpool2d_backward-output_size-3", 971 },
  { "max_unpool3d-output_size-padding-stride-2", 972 },
  { "max_unpool3d_backward-output_size-padding-stride-3", 973 },
  { "reflection_pad1d-padding-1", 974 },
  { "reflection_pad1d_backward-padding-2", 975 },
  { "reflection_pad2d-padding-1", 976 },
  { "reflection_pad2d_backward-padding-2", 977 },
  { "replication_pad1d-padding-1", 978 },
  { "replication_pad1d_backward-padding-2", 979 },
  { "replication_pad2d-padding-1", 980 },
  { "replication_pad2d_backward-padding-2", 981 },
  { "replication_pad3d-padding-1", 982 },
  { "replication_pad3d_backward-padding-2", 983 },
  { "upsample_linear1d-align_corners-output_size-1", 984 },
  { "upsample_linear1d_backward-align_corners-input_size-output_size-1", 985 },
  { "upsample_bilinear2d-align_corners-output_size-1", 986 },
  { "upsample_bilinear2d_backward-align_corners-input_size-output_size-1", 987 },
  { "upsample_bicubic2d-align_corners-output_size-1", 988 },
  { "upsample_bicubic2d_backward-align_corners-input_size-output_size-1", 989 },
  { "upsample_trilinear3d-align_corners-output_size-1", 990 },
  { "upsample_trilinear3d_backward-align_corners-input_size-output_size-1", 991 },
  { "upsample_nearest1d-output_size-1", 992 },
  { "upsample_nearest1d_backward-input_size-output_size-1", 993 },
  { "upsample_nearest2d-output_size-1", 994 },
  { "upsample_nearest2d_backward-input_size-output_size-1", 995 },
  { "upsample_nearest3d-output_size-1", 996 },
  { "upsample_nearest3d_backward-input_size-output_size-1", 997 },
  { "sigmoid_backward-2", 998 },
  { "tanh_backward-2", 999 },
  { "slow_conv_transpose2d-dilation-kernel_size-output_padding-padding-stride-3", 1000 },
  { "slow_conv_transpose2d-kernel_size-output_padding-padding-stride-3", 1001 },
  { "slow_conv_transpose2d-kernel_size-padding-stride-3", 1002 },
  { "slow_conv_transpose2d-kernel_size-stride-3", 1003 },
  { "slow_conv_transpose2d-kernel_size-3", 1004 },
  { "slow_conv_transpose2d-kernel_size-2", 1005 },
  { "slow_conv_transpose2d_backward-dilation-kernel_size-output_mask-output_padding-padding-stride-5", 1006 },
  { "slow_conv_transpose3d-dilation-kernel_size-output_padding-padding-stride-3", 1007 },
  { "slow_conv_transpose3d-kernel_size-output_padding-padding-stride-3", 1008 },
  { "slow_conv_transpose3d-kernel_size-padding-stride-3", 1009 },
  { "slow_conv_transpose3d-kernel_size-stride-3", 1010 },
  { "slow_conv_transpose3d-kernel_size-3", 1011 },
  { "slow_conv_transpose3d-kernel_size-2", 1012 },
  { "slow_conv_transpose3d_backward-dilation-kernel_size-output_mask-output_padding-padding-stride-5", 1013 },
  { "thnn_conv2d-kernel_size-padding-stride-3", 1014 },
  { "thnn_conv2d-kernel_size-stride-3", 1015 },
  { "thnn_conv2d-kernel_size-3", 1016 },
  { "thnn_conv2d-kernel_size-2", 1017 },
  { "thnn_conv2d_forward-kernel_size-padding-stride-3", 1018 },
  { "thnn_conv2d_backward-kernel_size-output_mask-padding-stride-5", 1019 },
  { "thnn_conv_depthwise2d-dilation-kernel_size-padding-stride-3", 1020 },
  { "thnn_conv_depthwise2d-kernel_size-padding-stride-3", 1021 },
  { "thnn_conv_depthwise2d-kernel_size-stride-3", 1022 },
  { "thnn_conv_depthwise2d-kernel_size-3", 1023 },
  { "thnn_conv_depthwise2d-kernel_size-2", 1024 },
  { "thnn_conv_depthwise2d_forward-dilation-kernel_size-padding-stride-3", 1025 },
  { "thnn_conv_depthwise2d_backward-dilation-kernel_size-output_mask-padding-stride-3", 1026 },
  { "thnn_conv3d-kernel_size-padding-stride-3", 1027 },
  { "thnn_conv3d-kernel_size-stride-3", 1028 },
  { "thnn_conv3d-kernel_size-3", 1029 },
  { "thnn_conv3d-kernel_size-2", 1030 },
  { "thnn_conv3d_forward-kernel_size-padding-stride-3", 1031 },
  { "thnn_conv3d_backward-kernel_size-output_mask-padding-stride-5", 1032 },
  { "slow_conv_dilated2d-dilation-kernel_size-padding-stride-3", 1033 },
  { "slow_conv_dilated2d-kernel_size-padding-stride-3", 1034 },
  { "slow_conv_dilated2d-kernel_size-stride-3", 1035 },
  { "slow_conv_dilated2d-kernel_size-3", 1036 },
  { "slow_conv_dilated2d-kernel_size-2", 1037 },
  { "slow_conv_dilated2d_backward-dilation-kernel_size-output_mask-padding-stride-3", 1038 },
  { "slow_conv_dilated3d-dilation-kernel_size-padding-stride-3", 1039 },
  { "slow_conv_dilated3d-kernel_size-padding-stride-3", 1040 },
  { "slow_conv_dilated3d-kernel_size-stride-3", 1041 },
  { "slow_conv_dilated3d-kernel_size-3", 1042 },
  { "slow_conv_dilated3d-kernel_size-2", 1043 },
  { "slow_conv_dilated3d_backward-dilation-kernel_size-output_mask-padding-stride-3", 1044 },
  { "col2im-dilation-kernel_size-output_size-padding-stride-1", 1045 },
  { "col2im_backward-dilation-kernel_size-padding-stride-1", 1046 },
  { "im2col-dilation-kernel_size-padding-stride-1", 1047 },
  { "im2col_backward-dilation-input_size-kernel_size-padding-stride-1", 1048 },
};

namespace caffe2 {

using at::Half; // for AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, ...)

template <class Context>
class ATenOp : public Operator<Context> {
 public:
  ATenOp(const OperatorDef& operator_def, Workspace* ws)
  : Operator<Context>(operator_def, ws) {
    VLOG(2) << "ATen OpDef: " << ProtoDebugString(operator_def) << "\n";
    switch(findImplementation(operator_def)) {
      case 0: { // _cast_Byte
          bool non_blocking = readAttribute<int64_t>("non_blocking");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_cast_Byte(self, non_blocking);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1: { // _cast_Byte
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_cast_Byte(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 2: { // _cast_Char
          bool non_blocking = readAttribute<int64_t>("non_blocking");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_cast_Char(self, non_blocking);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 3: { // _cast_Char
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_cast_Char(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 4: { // _cast_Double
          bool non_blocking = readAttribute<int64_t>("non_blocking");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_cast_Double(self, non_blocking);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 5: { // _cast_Double
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_cast_Double(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 6: { // _cast_Float
          bool non_blocking = readAttribute<int64_t>("non_blocking");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_cast_Float(self, non_blocking);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 7: { // _cast_Float
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_cast_Float(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 8: { // _cast_Int
          bool non_blocking = readAttribute<int64_t>("non_blocking");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_cast_Int(self, non_blocking);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 9: { // _cast_Int
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_cast_Int(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 10: { // _cast_Long
          bool non_blocking = readAttribute<int64_t>("non_blocking");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_cast_Long(self, non_blocking);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 11: { // _cast_Long
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_cast_Long(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 12: { // _cast_Short
          bool non_blocking = readAttribute<int64_t>("non_blocking");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_cast_Short(self, non_blocking);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 13: { // _cast_Short
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_cast_Short(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 14: { // _cast_Half
          bool non_blocking = readAttribute<int64_t>("non_blocking");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_cast_Half(self, non_blocking);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 15: { // _cast_Half
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_cast_Half(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 16: { // data
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.data();
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 17: { // is_leaf
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.is_leaf();
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 18: { // output_nr
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.output_nr();
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 19: { // _version
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self._version();
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 20: { // align_as
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = self.align_as(other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 21: { // align_tensors
      
          run_op = [=] {
              auto tensors = peekSlice(0, InputSize() - 0, InputSize());
              auto the_result = at::align_tensors(tensors);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 22: { // _cudnn_ctc_loss
          auto input_lengths = readIntArrayRef("input_lengths");
          auto target_lengths = readIntArrayRef("target_lengths");
          int64_t blank = readAttribute<int64_t>("blank");
          bool deterministic = readAttribute<int64_t>("deterministic");
          bool zero_infinity = readAttribute<int64_t>("zero_infinity");
          run_op = [=] {
              auto log_probs = peek(0, 2);
              auto targets = peek(1, 2);
              auto the_result = at::_cudnn_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, deterministic, zero_infinity);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 23: { // _cudnn_rnn_flatten_weight
          int64_t weight_stride0 = readAttribute<int64_t>("weight_stride0");
          int64_t input_size = readAttribute<int64_t>("input_size");
          int64_t mode = readAttribute<int64_t>("mode");
          int64_t hidden_size = readAttribute<int64_t>("hidden_size");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          bool batch_first = readAttribute<int64_t>("batch_first");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          run_op = [=] {
              auto weight_arr = peekSlice(0, InputSize() - 0, InputSize());
              auto the_result = at::_cudnn_rnn_flatten_weight(weight_arr, weight_stride0, input_size, mode, hidden_size, num_layers, batch_first, bidirectional);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 24: { // _cudnn_rnn
          int64_t weight_stride0 = readAttribute<int64_t>("weight_stride0");
          int64_t mode = readAttribute<int64_t>("mode");
          int64_t hidden_size = readAttribute<int64_t>("hidden_size");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          bool batch_first = readAttribute<int64_t>("batch_first");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          auto batch_sizes = readIntArrayRef("batch_sizes");
          run_op = [=] {
              auto input = peek(0, InputSize());
              auto weight = peekSlice(1, InputSize() - 5, InputSize());
              auto weight_buf = peek(1, 5);
              auto hx = peek(2, 5);
              auto cx = peek(3, 5);
              auto dropout_state = peek(4, 5);
              auto the_result = at::_cudnn_rnn(input, weight, weight_stride0, weight_buf, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
                if(OutputSize() > 4) {assignTo(Output(4),std::get<4>(the_result));}
              return true;
          };
      } break;
      case 25: { // _debug_has_internal_overlap
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_debug_has_internal_overlap(self);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 26: { // _fused_dropout
          double p = readAttribute<float>("p");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_fused_dropout(self, p);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 27: { // _masked_scale
          double scale = readAttribute<float>("scale");
          run_op = [=] {
              auto self = peek(0, 2);
              auto mask = peek(1, 2);
              auto the_result = at::_masked_scale(self, mask, scale);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 28: { // _reshape_from_tensor
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto shape = peek(1, 2);
              auto the_result = at::_reshape_from_tensor(self, shape);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 29: { // _shape_as_tensor
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_shape_as_tensor(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 30: { // dropout
          double p = readAttribute<float>("p");
          bool train = readAttribute<int64_t>("train");
          run_op = [=] {
              auto input = peek(0, 1);
              auto the_result = at::dropout(input, p, train);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 31: { // feature_dropout
          double p = readAttribute<float>("p");
          bool train = readAttribute<int64_t>("train");
          run_op = [=] {
              auto input = peek(0, 1);
              auto the_result = at::feature_dropout(input, p, train);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 32: { // alpha_dropout
          double p = readAttribute<float>("p");
          bool train = readAttribute<int64_t>("train");
          run_op = [=] {
              auto input = peek(0, 1);
              auto the_result = at::alpha_dropout(input, p, train);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 33: { // feature_alpha_dropout
          double p = readAttribute<float>("p");
          bool train = readAttribute<int64_t>("train");
          run_op = [=] {
              auto input = peek(0, 1);
              auto the_result = at::feature_alpha_dropout(input, p, train);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 34: { // abs
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::abs(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 35: { // acos
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::acos(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 36: { // avg_pool1d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          bool count_include_pad = readAttribute<int64_t>("count_include_pad");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::avg_pool1d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 37: { // avg_pool1d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::avg_pool1d(self, kernel_size, stride, padding, ceil_mode);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 38: { // avg_pool1d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::avg_pool1d(self, kernel_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 39: { // avg_pool1d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::avg_pool1d(self, kernel_size, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 40: { // avg_pool1d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::avg_pool1d(self, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 41: { // adaptive_avg_pool1d
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::adaptive_avg_pool1d(self, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 42: { // adaptive_max_pool1d
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::adaptive_max_pool1d(self, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 43: { // add
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::add(self, other, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 44: { // add
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::add(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 45: { // add
          at::Scalar other = readScalarAttribute("other");
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::add(self, other, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 46: { // add
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::add(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 47: { // addmv
          at::Scalar beta = readScalarAttribute("beta");
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              auto self = peek(0, 3);
              auto mat = peek(1, 3);
              auto vec = peek(2, 3);
              auto the_result = at::addmv(self, mat, vec, beta, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 48: { // addmv
          at::Scalar beta = readScalarAttribute("beta");
          run_op = [=] {
              auto self = peek(0, 3);
              auto mat = peek(1, 3);
              auto vec = peek(2, 3);
              auto the_result = at::addmv(self, mat, vec, beta);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 49: { // addmv
      
          run_op = [=] {
              auto self = peek(0, 3);
              auto mat = peek(1, 3);
              auto vec = peek(2, 3);
              auto the_result = at::addmv(self, mat, vec);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 50: { // addr
          at::Scalar beta = readScalarAttribute("beta");
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              auto self = peek(0, 3);
              auto vec1 = peek(1, 3);
              auto vec2 = peek(2, 3);
              auto the_result = at::addr(self, vec1, vec2, beta, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 51: { // addr
          at::Scalar beta = readScalarAttribute("beta");
          run_op = [=] {
              auto self = peek(0, 3);
              auto vec1 = peek(1, 3);
              auto vec2 = peek(2, 3);
              auto the_result = at::addr(self, vec1, vec2, beta);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 52: { // addr
      
          run_op = [=] {
              auto self = peek(0, 3);
              auto vec1 = peek(1, 3);
              auto vec2 = peek(2, 3);
              auto the_result = at::addr(self, vec1, vec2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 53: { // affine_grid_generator
          auto size = readIntArrayRef("size");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              auto theta = peek(0, 1);
              auto the_result = at::affine_grid_generator(theta, size, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 54: { // affine_grid_generator_backward
          auto size = readIntArrayRef("size");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              auto grad = peek(0, 1);
              auto the_result = at::affine_grid_generator_backward(grad, size, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 55: { // all
          int64_t dim = readAttribute<int64_t>("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::all(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 56: { // all
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::all(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 57: { // allclose
          double rtol = readAttribute<float>("rtol");
          double atol = readAttribute<float>("atol");
          bool equal_nan = readAttribute<int64_t>("equal_nan");
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::allclose(self, other, rtol, atol, equal_nan);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 58: { // allclose
          double rtol = readAttribute<float>("rtol");
          double atol = readAttribute<float>("atol");
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::allclose(self, other, rtol, atol);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 59: { // allclose
          double rtol = readAttribute<float>("rtol");
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::allclose(self, other, rtol);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 60: { // allclose
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::allclose(self, other);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 61: { // any
          int64_t dim = readAttribute<int64_t>("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::any(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 62: { // any
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::any(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 63: { // _dim_arange
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto like = peek(0, 1);
              auto the_result = at::_dim_arange(like, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 64: { // argmax
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::argmax(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 65: { // argmin
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::argmin(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 66: { // as_strided
          auto size = readIntArrayRef("size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::as_strided(self, size, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 67: { // asin
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::asin(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 68: { // atan
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::atan(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 69: { // baddbmm
          at::Scalar beta = readScalarAttribute("beta");
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              auto self = peek(0, 3);
              auto batch1 = peek(1, 3);
              auto batch2 = peek(2, 3);
              auto the_result = at::baddbmm(self, batch1, batch2, beta, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 70: { // baddbmm
          at::Scalar beta = readScalarAttribute("beta");
          run_op = [=] {
              auto self = peek(0, 3);
              auto batch1 = peek(1, 3);
              auto batch2 = peek(2, 3);
              auto the_result = at::baddbmm(self, batch1, batch2, beta);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 71: { // baddbmm
      
          run_op = [=] {
              auto self = peek(0, 3);
              auto batch1 = peek(1, 3);
              auto batch2 = peek(2, 3);
              auto the_result = at::baddbmm(self, batch1, batch2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 72: { // batch_norm
          bool training = readAttribute<int64_t>("training");
          double momentum = readAttribute<float>("momentum");
          double eps = readAttribute<float>("eps");
          bool cudnn_enabled = readAttribute<int64_t>("cudnn_enabled");
          run_op = [=] {
              auto input = peek(0, 5);
              auto weight = peek(1, 5);
              auto bias = peek(2, 5);
              auto running_mean = peek(3, 5);
              auto running_var = peek(4, 5);
              auto the_result = at::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 73: { // _batch_norm_impl_index
          bool training = readAttribute<int64_t>("training");
          double momentum = readAttribute<float>("momentum");
          double eps = readAttribute<float>("eps");
          bool cudnn_enabled = readAttribute<int64_t>("cudnn_enabled");
          run_op = [=] {
              auto input = peek(0, 5);
              auto weight = peek(1, 5);
              auto bias = peek(2, 5);
              auto running_mean = peek(3, 5);
              auto running_var = peek(4, 5);
              auto the_result = at::_batch_norm_impl_index(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignToValue<int64_t>(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 74: { // _batch_norm_impl_index_backward
          int64_t impl_index = readAttribute<int64_t>("impl_index");
          bool train = readAttribute<int64_t>("train");
          double eps = readAttribute<float>("eps");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              auto input = peek(0, 7);
              auto grad_output = peek(1, 7);
              auto weight = peek(2, 7);
              auto running_mean = peek(3, 7);
              auto running_var = peek(4, 7);
              auto save_mean = peek(5, 7);
              auto save_var_transform = peek(6, 7);
              auto the_result = at::_batch_norm_impl_index_backward(impl_index, input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, train, eps, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 75: { // bernoulli
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::bernoulli(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 76: { // bernoulli
          double p = readAttribute<float>("p");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::bernoulli(self, p);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 77: { // bilinear
      
          run_op = [=] {
              auto input1 = peek(0, 4);
              auto input2 = peek(1, 4);
              auto weight = peek(2, 4);
              auto bias = peek(3, 4);
              auto the_result = at::bilinear(input1, input2, weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 78: { // binary_cross_entropy_with_logits
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto self = peek(0, 4);
              auto target = peek(1, 4);
              auto weight = peek(2, 4);
              auto pos_weight = peek(3, 4);
              auto the_result = at::binary_cross_entropy_with_logits(self, target, weight, pos_weight, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 79: { // binary_cross_entropy_with_logits
      
          run_op = [=] {
              auto self = peek(0, 4);
              auto target = peek(1, 4);
              auto weight = peek(2, 4);
              auto pos_weight = peek(3, 4);
              auto the_result = at::binary_cross_entropy_with_logits(self, target, weight, pos_weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 80: { // binary_cross_entropy_with_logits
      
          run_op = [=] {
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::binary_cross_entropy_with_logits(self, target, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 81: { // binary_cross_entropy_with_logits
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::binary_cross_entropy_with_logits(self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 82: { // binary_cross_entropy_with_logits_backward
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto grad_output = peek(0, 5);
              auto self = peek(1, 5);
              auto target = peek(2, 5);
              auto weight = peek(3, 5);
              auto pos_weight = peek(4, 5);
              auto the_result = at::binary_cross_entropy_with_logits_backward(grad_output, self, target, weight, pos_weight, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 83: { // binary_cross_entropy_with_logits_backward
      
          run_op = [=] {
              auto grad_output = peek(0, 5);
              auto self = peek(1, 5);
              auto target = peek(2, 5);
              auto weight = peek(3, 5);
              auto pos_weight = peek(4, 5);
              auto the_result = at::binary_cross_entropy_with_logits_backward(grad_output, self, target, weight, pos_weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 84: { // binary_cross_entropy_with_logits_backward
      
          run_op = [=] {
              auto grad_output = peek(0, 4);
              auto self = peek(1, 4);
              auto target = peek(2, 4);
              auto weight = peek(3, 4);
              auto the_result = at::binary_cross_entropy_with_logits_backward(grad_output, self, target, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 85: { // binary_cross_entropy_with_logits_backward
      
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::binary_cross_entropy_with_logits_backward(grad_output, self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 86: { // bincount
          int64_t minlength = readAttribute<int64_t>("minlength");
          run_op = [=] {
              auto self = peek(0, 2);
              auto weights = peek(1, 2);
              auto the_result = at::bincount(self, weights, minlength);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 87: { // bincount
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto weights = peek(1, 2);
              auto the_result = at::bincount(self, weights);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 88: { // bincount
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::bincount(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 89: { // bitwise_not
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::bitwise_not(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 90: { // logical_not
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::logical_not(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 91: { // logical_xor
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::logical_xor(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 92: { // bmm
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto mat2 = peek(1, 2);
              auto the_result = at::bmm(self, mat2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 93: { // broadcast_tensors
      
          run_op = [=] {
              auto tensors = peekSlice(0, InputSize() - 0, InputSize());
              auto the_result = at::broadcast_tensors(tensors);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 94: { // cat
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto tensors = peekSlice(0, InputSize() - 0, InputSize());
              auto the_result = at::cat(tensors, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 95: { // cat
      
          run_op = [=] {
              auto tensors = peekSlice(0, InputSize() - 0, InputSize());
              auto the_result = at::cat(tensors);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 96: { // ceil
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::ceil(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 97: { // chain_matmul
      
          run_op = [=] {
              auto matrices = peekSlice(0, InputSize() - 0, InputSize());
              auto the_result = at::chain_matmul(matrices);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 98: { // chunk
          int64_t chunks = readAttribute<int64_t>("chunks");
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::chunk(self, chunks, dim);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 99: { // chunk
          int64_t chunks = readAttribute<int64_t>("chunks");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::chunk(self, chunks);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 100: { // clamp
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::clamp(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 101: { // clamp_max
          at::Scalar max = readScalarAttribute("max");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::clamp_max(self, max);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 102: { // clamp_min
          at::Scalar min = readScalarAttribute("min");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::clamp_min(self, min);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 103: { // cudnn_is_acceptable
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::cudnn_is_acceptable(self);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 104: { // constant_pad_nd
          auto pad = readIntArrayRef("pad");
          at::Scalar value = readScalarAttribute("value");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::constant_pad_nd(self, pad, value);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 105: { // constant_pad_nd
          auto pad = readIntArrayRef("pad");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::constant_pad_nd(self, pad);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 106: { // contiguous
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.contiguous();
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 107: { // convolution
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool transposed = readAttribute<int64_t>("transposed");
          auto output_padding = readIntArrayRef("output_padding");
          int64_t groups = readAttribute<int64_t>("groups");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 108: { // convolution_overrideable
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool transposed = readAttribute<int64_t>("transposed");
          auto output_padding = readIntArrayRef("output_padding");
          int64_t groups = readAttribute<int64_t>("groups");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::convolution_overrideable(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 109: { // convolution_backward_overrideable
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool transposed = readAttribute<int64_t>("transposed");
          auto output_padding = readIntArrayRef("output_padding");
          int64_t groups = readAttribute<int64_t>("groups");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto input = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::convolution_backward_overrideable(grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 110: { // _convolution
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool transposed = readAttribute<int64_t>("transposed");
          auto output_padding = readIntArrayRef("output_padding");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          bool cudnn_enabled = readAttribute<int64_t>("cudnn_enabled");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 111: { // _convolution_nogroup
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool transposed = readAttribute<int64_t>("transposed");
          auto output_padding = readIntArrayRef("output_padding");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::_convolution_nogroup(input, weight, bias, stride, padding, dilation, transposed, output_padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 112: { // _convolution_double_backward
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool transposed = readAttribute<int64_t>("transposed");
          auto output_padding = readIntArrayRef("output_padding");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          bool cudnn_enabled = readAttribute<int64_t>("cudnn_enabled");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              auto ggI = peek(0, 6);
              auto ggW = peek(1, 6);
              auto ggb = peek(2, 6);
              auto gO = peek(3, 6);
              auto weight = peek(4, 6);
              auto self = peek(5, 6);
              auto the_result = at::_convolution_double_backward(ggI, ggW, ggb, gO, weight, self, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 113: { // conv1d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv1d(input, weight, bias, stride, padding, dilation, groups);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 114: { // conv1d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv1d(input, weight, bias, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 115: { // conv1d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv1d(input, weight, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 116: { // conv1d
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv1d(input, weight, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 117: { // conv1d
      
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv1d(input, weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 118: { // conv1d
      
          run_op = [=] {
              auto input = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::conv1d(input, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 119: { // conv2d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv2d(input, weight, bias, stride, padding, dilation, groups);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 120: { // conv2d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv2d(input, weight, bias, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 121: { // conv2d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv2d(input, weight, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 122: { // conv2d
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv2d(input, weight, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 123: { // conv2d
      
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv2d(input, weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 124: { // conv2d
      
          run_op = [=] {
              auto input = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::conv2d(input, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 125: { // conv3d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv3d(input, weight, bias, stride, padding, dilation, groups);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 126: { // conv3d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv3d(input, weight, bias, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 127: { // conv3d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv3d(input, weight, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 128: { // conv3d
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv3d(input, weight, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 129: { // conv3d
      
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv3d(input, weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 130: { // conv3d
      
          run_op = [=] {
              auto input = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::conv3d(input, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 131: { // conv_tbc
          int64_t pad = readAttribute<int64_t>("pad");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_tbc(self, weight, bias, pad);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 132: { // conv_tbc
      
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_tbc(self, weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 133: { // conv_tbc_backward
          int64_t pad = readAttribute<int64_t>("pad");
          run_op = [=] {
              auto self = peek(0, 4);
              auto input = peek(1, 4);
              auto weight = peek(2, 4);
              auto bias = peek(3, 4);
              auto the_result = at::conv_tbc_backward(self, input, weight, bias, pad);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 134: { // conv_transpose1d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          int64_t groups = readAttribute<int64_t>("groups");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 135: { // conv_transpose1d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          int64_t groups = readAttribute<int64_t>("groups");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 136: { // conv_transpose1d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose1d(input, weight, bias, stride, padding, output_padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 137: { // conv_transpose1d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose1d(input, weight, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 138: { // conv_transpose1d
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose1d(input, weight, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 139: { // conv_transpose1d
      
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose1d(input, weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 140: { // conv_transpose1d
      
          run_op = [=] {
              auto input = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::conv_transpose1d(input, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 141: { // conv_transpose2d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          int64_t groups = readAttribute<int64_t>("groups");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 142: { // conv_transpose2d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          int64_t groups = readAttribute<int64_t>("groups");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 143: { // conv_transpose2d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose2d(input, weight, bias, stride, padding, output_padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 144: { // conv_transpose2d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose2d(input, weight, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 145: { // conv_transpose2d
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose2d(input, weight, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 146: { // conv_transpose2d
      
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose2d(input, weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 147: { // conv_transpose2d
      
          run_op = [=] {
              auto input = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::conv_transpose2d(input, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 148: { // conv_transpose3d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          int64_t groups = readAttribute<int64_t>("groups");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 149: { // conv_transpose3d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          int64_t groups = readAttribute<int64_t>("groups");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 150: { // conv_transpose3d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose3d(input, weight, bias, stride, padding, output_padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 151: { // conv_transpose3d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose3d(input, weight, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 152: { // conv_transpose3d
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose3d(input, weight, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 153: { // conv_transpose3d
      
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose3d(input, weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 154: { // conv_transpose3d
      
          run_op = [=] {
              auto input = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::conv_transpose3d(input, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 155: { // _copy_from
          bool non_blocking = readAttribute<int64_t>("non_blocking");
          run_op = [=] {
              auto self = peek(0, 2);
              auto dst = peek(1, 2);
              auto the_result = at::_copy_from(self, dst, non_blocking);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 156: { // _copy_from
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto dst = peek(1, 2);
              auto the_result = at::_copy_from(self, dst);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 157: { // cos
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::cos(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 158: { // cosh
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::cosh(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 159: { // cosine_embedding_loss
          double margin = readAttribute<float>("margin");
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto input1 = peek(0, 3);
              auto input2 = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::cosine_embedding_loss(input1, input2, target, margin, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 160: { // cosine_embedding_loss
          double margin = readAttribute<float>("margin");
          run_op = [=] {
              auto input1 = peek(0, 3);
              auto input2 = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::cosine_embedding_loss(input1, input2, target, margin);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 161: { // cosine_embedding_loss
      
          run_op = [=] {
              auto input1 = peek(0, 3);
              auto input2 = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::cosine_embedding_loss(input1, input2, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 162: { // cudnn_affine_grid_generator
          int64_t N = readAttribute<int64_t>("N");
          int64_t C = readAttribute<int64_t>("C");
          int64_t H = readAttribute<int64_t>("H");
          int64_t W = readAttribute<int64_t>("W");
          run_op = [=] {
              auto theta = peek(0, 1);
              auto the_result = at::cudnn_affine_grid_generator(theta, N, C, H, W);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 163: { // cudnn_affine_grid_generator_backward
          int64_t N = readAttribute<int64_t>("N");
          int64_t C = readAttribute<int64_t>("C");
          int64_t H = readAttribute<int64_t>("H");
          int64_t W = readAttribute<int64_t>("W");
          run_op = [=] {
              auto grad = peek(0, 1);
              auto the_result = at::cudnn_affine_grid_generator_backward(grad, N, C, H, W);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 164: { // cudnn_batch_norm
          bool training = readAttribute<int64_t>("training");
          double exponential_average_factor = readAttribute<float>("exponential_average_factor");
          double epsilon = readAttribute<float>("epsilon");
          run_op = [=] {
              auto input = peek(0, 5);
              auto weight = peek(1, 5);
              auto bias = peek(2, 5);
              auto running_mean = peek(3, 5);
              auto running_var = peek(4, 5);
              auto the_result = at::cudnn_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 165: { // cudnn_batch_norm_backward
          double epsilon = readAttribute<float>("epsilon");
          run_op = [=] {
              auto input = peek(0, 7);
              auto grad_output = peek(1, 7);
              auto weight = peek(2, 7);
              auto running_mean = peek(3, 7);
              auto running_var = peek(4, 7);
              auto save_mean = peek(5, 7);
              auto save_var = peek(6, 7);
              auto the_result = at::cudnn_batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 166: { // cudnn_convolution
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::cudnn_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 167: { // cudnn_convolution_backward_input
          auto self_size = readIntArrayRef("self_size");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::cudnn_convolution_backward_input(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 168: { // cudnn_convolution_backward
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              auto self = peek(0, 3);
              auto grad_output = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::cudnn_convolution_backward(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 169: { // cudnn_convolution_backward_bias
      
          run_op = [=] {
              auto grad_output = peek(0, 1);
              auto the_result = at::cudnn_convolution_backward_bias(grad_output);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 170: { // cudnn_convolution_backward_weight
          auto weight_size = readIntArrayRef("weight_size");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::cudnn_convolution_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 171: { // cudnn_convolution_transpose
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::cudnn_convolution_transpose(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 172: { // cudnn_convolution_transpose_backward
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              auto self = peek(0, 3);
              auto grad_output = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::cudnn_convolution_transpose_backward(self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 173: { // cudnn_convolution_transpose_backward_bias
      
          run_op = [=] {
              auto grad_output = peek(0, 1);
              auto the_result = at::cudnn_convolution_transpose_backward_bias(grad_output);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 174: { // cudnn_convolution_transpose_backward_input
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::cudnn_convolution_transpose_backward_input(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 175: { // cudnn_convolution_transpose_backward_weight
          auto weight_size = readIntArrayRef("weight_size");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::cudnn_convolution_transpose_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 176: { // cudnn_grid_sampler
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto grid = peek(1, 2);
              auto the_result = at::cudnn_grid_sampler(self, grid);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 177: { // cudnn_grid_sampler_backward
      
          run_op = [=] {
              auto self = peek(0, 3);
              auto grid = peek(1, 3);
              auto grad_output = peek(2, 3);
              auto the_result = at::cudnn_grid_sampler_backward(self, grid, grad_output);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 178: { // cumsum
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::cumsum(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 179: { // cumprod
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::cumprod(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 180: { // ctc_loss
          auto input_lengths = readIntArrayRef("input_lengths");
          auto target_lengths = readIntArrayRef("target_lengths");
          int64_t blank = readAttribute<int64_t>("blank");
          int64_t reduction = readAttribute<int64_t>("reduction");
          bool zero_infinity = readAttribute<int64_t>("zero_infinity");
          run_op = [=] {
              auto log_probs = peek(0, 2);
              auto targets = peek(1, 2);
              auto the_result = at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 181: { // ctc_loss
          auto input_lengths = readIntArrayRef("input_lengths");
          auto target_lengths = readIntArrayRef("target_lengths");
          int64_t blank = readAttribute<int64_t>("blank");
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto log_probs = peek(0, 2);
              auto targets = peek(1, 2);
              auto the_result = at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 182: { // ctc_loss
          auto input_lengths = readIntArrayRef("input_lengths");
          auto target_lengths = readIntArrayRef("target_lengths");
          int64_t blank = readAttribute<int64_t>("blank");
          run_op = [=] {
              auto log_probs = peek(0, 2);
              auto targets = peek(1, 2);
              auto the_result = at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 183: { // ctc_loss
          auto input_lengths = readIntArrayRef("input_lengths");
          auto target_lengths = readIntArrayRef("target_lengths");
          run_op = [=] {
              auto log_probs = peek(0, 2);
              auto targets = peek(1, 2);
              auto the_result = at::ctc_loss(log_probs, targets, input_lengths, target_lengths);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 184: { // ctc_loss
          int64_t blank = readAttribute<int64_t>("blank");
          int64_t reduction = readAttribute<int64_t>("reduction");
          bool zero_infinity = readAttribute<int64_t>("zero_infinity");
          run_op = [=] {
              auto log_probs = peek(0, 4);
              auto targets = peek(1, 4);
              auto input_lengths = peek(2, 4);
              auto target_lengths = peek(3, 4);
              auto the_result = at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 185: { // ctc_loss
          int64_t blank = readAttribute<int64_t>("blank");
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto log_probs = peek(0, 4);
              auto targets = peek(1, 4);
              auto input_lengths = peek(2, 4);
              auto target_lengths = peek(3, 4);
              auto the_result = at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 186: { // ctc_loss
          int64_t blank = readAttribute<int64_t>("blank");
          run_op = [=] {
              auto log_probs = peek(0, 4);
              auto targets = peek(1, 4);
              auto input_lengths = peek(2, 4);
              auto target_lengths = peek(3, 4);
              auto the_result = at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 187: { // ctc_loss
      
          run_op = [=] {
              auto log_probs = peek(0, 4);
              auto targets = peek(1, 4);
              auto input_lengths = peek(2, 4);
              auto target_lengths = peek(3, 4);
              auto the_result = at::ctc_loss(log_probs, targets, input_lengths, target_lengths);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 188: { // _ctc_loss
          auto input_lengths = readIntArrayRef("input_lengths");
          auto target_lengths = readIntArrayRef("target_lengths");
          int64_t blank = readAttribute<int64_t>("blank");
          bool zero_infinity = readAttribute<int64_t>("zero_infinity");
          run_op = [=] {
              auto log_probs = peek(0, 2);
              auto targets = peek(1, 2);
              auto the_result = at::_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 189: { // _ctc_loss
          auto input_lengths = readIntArrayRef("input_lengths");
          auto target_lengths = readIntArrayRef("target_lengths");
          int64_t blank = readAttribute<int64_t>("blank");
          run_op = [=] {
              auto log_probs = peek(0, 2);
              auto targets = peek(1, 2);
              auto the_result = at::_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 190: { // _ctc_loss
          auto input_lengths = readIntArrayRef("input_lengths");
          auto target_lengths = readIntArrayRef("target_lengths");
          run_op = [=] {
              auto log_probs = peek(0, 2);
              auto targets = peek(1, 2);
              auto the_result = at::_ctc_loss(log_probs, targets, input_lengths, target_lengths);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 191: { // _ctc_loss_backward
          auto input_lengths = readIntArrayRef("input_lengths");
          auto target_lengths = readIntArrayRef("target_lengths");
          int64_t blank = readAttribute<int64_t>("blank");
          bool zero_infinity = readAttribute<int64_t>("zero_infinity");
          run_op = [=] {
              auto grad = peek(0, 5);
              auto log_probs = peek(1, 5);
              auto targets = peek(2, 5);
              auto neg_log_likelihood = peek(3, 5);
              auto log_alpha = peek(4, 5);
              auto the_result = at::_ctc_loss_backward(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 192: { // _ctc_loss_backward
          auto input_lengths = readIntArrayRef("input_lengths");
          auto target_lengths = readIntArrayRef("target_lengths");
          int64_t blank = readAttribute<int64_t>("blank");
          run_op = [=] {
              auto grad = peek(0, 5);
              auto log_probs = peek(1, 5);
              auto targets = peek(2, 5);
              auto neg_log_likelihood = peek(3, 5);
              auto log_alpha = peek(4, 5);
              auto the_result = at::_ctc_loss_backward(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 193: { // det
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::det(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 194: { // diag_embed
          int64_t offset = readAttribute<int64_t>("offset");
          int64_t dim1 = readAttribute<int64_t>("dim1");
          int64_t dim2 = readAttribute<int64_t>("dim2");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::diag_embed(self, offset, dim1, dim2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 195: { // diag_embed
          int64_t offset = readAttribute<int64_t>("offset");
          int64_t dim1 = readAttribute<int64_t>("dim1");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::diag_embed(self, offset, dim1);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 196: { // diag_embed
          int64_t offset = readAttribute<int64_t>("offset");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::diag_embed(self, offset);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 197: { // diag_embed
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::diag_embed(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 198: { // diagflat
          int64_t offset = readAttribute<int64_t>("offset");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::diagflat(self, offset);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 199: { // diagflat
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::diagflat(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 200: { // diagonal
          int64_t offset = readAttribute<int64_t>("offset");
          int64_t dim1 = readAttribute<int64_t>("dim1");
          int64_t dim2 = readAttribute<int64_t>("dim2");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::diagonal(self, offset, dim1, dim2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 201: { // diagonal
          int64_t offset = readAttribute<int64_t>("offset");
          int64_t dim1 = readAttribute<int64_t>("dim1");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::diagonal(self, offset, dim1);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 202: { // diagonal
          int64_t offset = readAttribute<int64_t>("offset");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::diagonal(self, offset);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 203: { // diagonal
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::diagonal(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 204: { // div
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::div(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 205: { // div
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::div(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 206: { // dot
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto tensor = peek(1, 2);
              auto the_result = at::dot(self, tensor);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 207: { // embedding
          int64_t padding_idx = readAttribute<int64_t>("padding_idx");
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          bool sparse = readAttribute<int64_t>("sparse");
          run_op = [=] {
              auto weight = peek(0, 2);
              auto indices = peek(1, 2);
              auto the_result = at::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 208: { // embedding
          int64_t padding_idx = readAttribute<int64_t>("padding_idx");
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          run_op = [=] {
              auto weight = peek(0, 2);
              auto indices = peek(1, 2);
              auto the_result = at::embedding(weight, indices, padding_idx, scale_grad_by_freq);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 209: { // embedding
          int64_t padding_idx = readAttribute<int64_t>("padding_idx");
          run_op = [=] {
              auto weight = peek(0, 2);
              auto indices = peek(1, 2);
              auto the_result = at::embedding(weight, indices, padding_idx);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 210: { // embedding
      
          run_op = [=] {
              auto weight = peek(0, 2);
              auto indices = peek(1, 2);
              auto the_result = at::embedding(weight, indices);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 211: { // embedding_backward
          int64_t num_weights = readAttribute<int64_t>("num_weights");
          int64_t padding_idx = readAttribute<int64_t>("padding_idx");
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          bool sparse = readAttribute<int64_t>("sparse");
          run_op = [=] {
              auto grad = peek(0, 2);
              auto indices = peek(1, 2);
              auto the_result = at::embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 212: { // embedding_dense_backward
          int64_t num_weights = readAttribute<int64_t>("num_weights");
          int64_t padding_idx = readAttribute<int64_t>("padding_idx");
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto indices = peek(1, 2);
              auto the_result = at::embedding_dense_backward(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 213: { // embedding_sparse_backward
          int64_t num_weights = readAttribute<int64_t>("num_weights");
          int64_t padding_idx = readAttribute<int64_t>("padding_idx");
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          run_op = [=] {
              auto grad = peek(0, 2);
              auto indices = peek(1, 2);
              auto the_result = at::embedding_sparse_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 214: { // embedding_bag
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          int64_t mode = readAttribute<int64_t>("mode");
          bool sparse = readAttribute<int64_t>("sparse");
          run_op = [=] {
              auto weight = peek(0, 4);
              auto indices = peek(1, 4);
              auto offsets = peek(2, 4);
              auto per_sample_weights = peek(3, 4);
              auto the_result = at::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 215: { // embedding_bag
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          int64_t mode = readAttribute<int64_t>("mode");
          bool sparse = readAttribute<int64_t>("sparse");
          run_op = [=] {
              auto weight = peek(0, 3);
              auto indices = peek(1, 3);
              auto offsets = peek(2, 3);
              auto the_result = at::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 216: { // embedding_bag
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          int64_t mode = readAttribute<int64_t>("mode");
          run_op = [=] {
              auto weight = peek(0, 3);
              auto indices = peek(1, 3);
              auto offsets = peek(2, 3);
              auto the_result = at::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 217: { // embedding_bag
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          run_op = [=] {
              auto weight = peek(0, 3);
              auto indices = peek(1, 3);
              auto offsets = peek(2, 3);
              auto the_result = at::embedding_bag(weight, indices, offsets, scale_grad_by_freq);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 218: { // embedding_bag
      
          run_op = [=] {
              auto weight = peek(0, 3);
              auto indices = peek(1, 3);
              auto offsets = peek(2, 3);
              auto the_result = at::embedding_bag(weight, indices, offsets);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 219: { // _embedding_bag
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          int64_t mode = readAttribute<int64_t>("mode");
          bool sparse = readAttribute<int64_t>("sparse");
          run_op = [=] {
              auto weight = peek(0, 4);
              auto indices = peek(1, 4);
              auto offsets = peek(2, 4);
              auto per_sample_weights = peek(3, 4);
              auto the_result = at::_embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 220: { // _embedding_bag
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          int64_t mode = readAttribute<int64_t>("mode");
          bool sparse = readAttribute<int64_t>("sparse");
          run_op = [=] {
              auto weight = peek(0, 3);
              auto indices = peek(1, 3);
              auto offsets = peek(2, 3);
              auto the_result = at::_embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 221: { // _embedding_bag
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          int64_t mode = readAttribute<int64_t>("mode");
          run_op = [=] {
              auto weight = peek(0, 3);
              auto indices = peek(1, 3);
              auto offsets = peek(2, 3);
              auto the_result = at::_embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 222: { // _embedding_bag
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          run_op = [=] {
              auto weight = peek(0, 3);
              auto indices = peek(1, 3);
              auto offsets = peek(2, 3);
              auto the_result = at::_embedding_bag(weight, indices, offsets, scale_grad_by_freq);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 223: { // _embedding_bag
      
          run_op = [=] {
              auto weight = peek(0, 3);
              auto indices = peek(1, 3);
              auto offsets = peek(2, 3);
              auto the_result = at::_embedding_bag(weight, indices, offsets);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 224: { // _embedding_bag_backward
          int64_t num_weights = readAttribute<int64_t>("num_weights");
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          int64_t mode = readAttribute<int64_t>("mode");
          bool sparse = readAttribute<int64_t>("sparse");
          run_op = [=] {
              auto grad = peek(0, 7);
              auto indices = peek(1, 7);
              auto offsets = peek(2, 7);
              auto offset2bag = peek(3, 7);
              auto bag_size = peek(4, 7);
              auto maximum_indices = peek(5, 7);
              auto per_sample_weights = peek(6, 7);
              auto the_result = at::_embedding_bag_backward(grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 225: { // _embedding_bag_sparse_backward
          int64_t num_weights = readAttribute<int64_t>("num_weights");
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          int64_t mode = readAttribute<int64_t>("mode");
          run_op = [=] {
              auto grad = peek(0, 6);
              auto indices = peek(1, 6);
              auto offsets = peek(2, 6);
              auto offset2bag = peek(3, 6);
              auto bag_size = peek(4, 6);
              auto per_sample_weights = peek(5, 6);
              auto the_result = at::_embedding_bag_sparse_backward(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode, per_sample_weights);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 226: { // _embedding_bag_dense_backward
          int64_t num_weights = readAttribute<int64_t>("num_weights");
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          int64_t mode = readAttribute<int64_t>("mode");
          run_op = [=] {
              auto grad = peek(0, 7);
              auto indices = peek(1, 7);
              auto offsets = peek(2, 7);
              auto offset2bag = peek(3, 7);
              auto bag_size = peek(4, 7);
              auto maximum_indices = peek(5, 7);
              auto per_sample_weights = peek(6, 7);
              auto the_result = at::_embedding_bag_dense_backward(grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, per_sample_weights);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 227: { // _embedding_bag_per_sample_weights_backward
          int64_t mode = readAttribute<int64_t>("mode");
          run_op = [=] {
              auto grad = peek(0, 5);
              auto weight = peek(1, 5);
              auto indices = peek(2, 5);
              auto offsets = peek(3, 5);
              auto offset2bag = peek(4, 5);
              auto the_result = at::_embedding_bag_per_sample_weights_backward(grad, weight, indices, offsets, offset2bag, mode);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 228: { // erf
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::erf(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 229: { // erfc
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::erfc(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 230: { // exp
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::exp(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 231: { // expm1
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::expm1(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 232: { // expand
          auto size = readIntArrayRef("size");
          bool implicit = readAttribute<int64_t>("implicit");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.expand(size, implicit);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 233: { // expand
          auto size = readIntArrayRef("size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.expand(size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 234: { // expand_as
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = self.expand_as(other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 235: { // flatten
          int64_t start_dim = readAttribute<int64_t>("start_dim");
          int64_t end_dim = readAttribute<int64_t>("end_dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::flatten(self, start_dim, end_dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 236: { // flatten
          int64_t start_dim = readAttribute<int64_t>("start_dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::flatten(self, start_dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 237: { // flatten
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::flatten(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 238: { // floor
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::floor(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 239: { // frac
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::frac(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 240: { // grid_sampler
          int64_t interpolation_mode = readAttribute<int64_t>("interpolation_mode");
          int64_t padding_mode = readAttribute<int64_t>("padding_mode");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              auto input = peek(0, 2);
              auto grid = peek(1, 2);
              auto the_result = at::grid_sampler(input, grid, interpolation_mode, padding_mode, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 241: { // grid_sampler_2d
          int64_t interpolation_mode = readAttribute<int64_t>("interpolation_mode");
          int64_t padding_mode = readAttribute<int64_t>("padding_mode");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              auto input = peek(0, 2);
              auto grid = peek(1, 2);
              auto the_result = at::grid_sampler_2d(input, grid, interpolation_mode, padding_mode, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 242: { // grid_sampler_2d_backward
          int64_t interpolation_mode = readAttribute<int64_t>("interpolation_mode");
          int64_t padding_mode = readAttribute<int64_t>("padding_mode");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto input = peek(1, 3);
              auto grid = peek(2, 3);
              auto the_result = at::grid_sampler_2d_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 243: { // grid_sampler_3d
          int64_t interpolation_mode = readAttribute<int64_t>("interpolation_mode");
          int64_t padding_mode = readAttribute<int64_t>("padding_mode");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              auto input = peek(0, 2);
              auto grid = peek(1, 2);
              auto the_result = at::grid_sampler_3d(input, grid, interpolation_mode, padding_mode, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 244: { // grid_sampler_3d_backward
          int64_t interpolation_mode = readAttribute<int64_t>("interpolation_mode");
          int64_t padding_mode = readAttribute<int64_t>("padding_mode");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto input = peek(1, 3);
              auto grid = peek(2, 3);
              auto the_result = at::grid_sampler_3d_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 245: { // hinge_embedding_loss
          double margin = readAttribute<float>("margin");
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::hinge_embedding_loss(self, target, margin, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 246: { // hinge_embedding_loss
          double margin = readAttribute<float>("margin");
          run_op = [=] {
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::hinge_embedding_loss(self, target, margin);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 247: { // hinge_embedding_loss
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::hinge_embedding_loss(self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 248: { // ger
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto vec2 = peek(1, 2);
              auto the_result = at::ger(self, vec2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 249: { // group_norm
          int64_t num_groups = readAttribute<int64_t>("num_groups");
          double eps = readAttribute<float>("eps");
          bool cudnn_enabled = readAttribute<int64_t>("cudnn_enabled");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::group_norm(input, num_groups, weight, bias, eps, cudnn_enabled);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 250: { // group_norm
          int64_t num_groups = readAttribute<int64_t>("num_groups");
          double eps = readAttribute<float>("eps");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::group_norm(input, num_groups, weight, bias, eps);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 251: { // group_norm
          int64_t num_groups = readAttribute<int64_t>("num_groups");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::group_norm(input, num_groups, weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 252: { // group_norm
          int64_t num_groups = readAttribute<int64_t>("num_groups");
          run_op = [=] {
              auto input = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::group_norm(input, num_groups, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 253: { // group_norm
          int64_t num_groups = readAttribute<int64_t>("num_groups");
          run_op = [=] {
              auto input = peek(0, 1);
              auto the_result = at::group_norm(input, num_groups);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 254: { // fft
          int64_t signal_ndim = readAttribute<int64_t>("signal_ndim");
          bool normalized = readAttribute<int64_t>("normalized");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::fft(self, signal_ndim, normalized);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 255: { // fft
          int64_t signal_ndim = readAttribute<int64_t>("signal_ndim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::fft(self, signal_ndim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 256: { // ifft
          int64_t signal_ndim = readAttribute<int64_t>("signal_ndim");
          bool normalized = readAttribute<int64_t>("normalized");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::ifft(self, signal_ndim, normalized);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 257: { // ifft
          int64_t signal_ndim = readAttribute<int64_t>("signal_ndim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::ifft(self, signal_ndim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 258: { // rfft
          int64_t signal_ndim = readAttribute<int64_t>("signal_ndim");
          bool normalized = readAttribute<int64_t>("normalized");
          bool onesided = readAttribute<int64_t>("onesided");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::rfft(self, signal_ndim, normalized, onesided);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 259: { // rfft
          int64_t signal_ndim = readAttribute<int64_t>("signal_ndim");
          bool normalized = readAttribute<int64_t>("normalized");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::rfft(self, signal_ndim, normalized);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 260: { // rfft
          int64_t signal_ndim = readAttribute<int64_t>("signal_ndim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::rfft(self, signal_ndim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 261: { // irfft
          int64_t signal_ndim = readAttribute<int64_t>("signal_ndim");
          bool normalized = readAttribute<int64_t>("normalized");
          bool onesided = readAttribute<int64_t>("onesided");
          auto signal_sizes = readIntArrayRef("signal_sizes");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::irfft(self, signal_ndim, normalized, onesided, signal_sizes);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 262: { // irfft
          int64_t signal_ndim = readAttribute<int64_t>("signal_ndim");
          bool normalized = readAttribute<int64_t>("normalized");
          bool onesided = readAttribute<int64_t>("onesided");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::irfft(self, signal_ndim, normalized, onesided);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 263: { // irfft
          int64_t signal_ndim = readAttribute<int64_t>("signal_ndim");
          bool normalized = readAttribute<int64_t>("normalized");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::irfft(self, signal_ndim, normalized);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 264: { // irfft
          int64_t signal_ndim = readAttribute<int64_t>("signal_ndim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::irfft(self, signal_ndim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 265: { // _fft_with_size
          int64_t signal_ndim = readAttribute<int64_t>("signal_ndim");
          bool complex_input = readAttribute<int64_t>("complex_input");
          bool complex_output = readAttribute<int64_t>("complex_output");
          bool inverse = readAttribute<int64_t>("inverse");
          auto checked_signal_sizes = readIntArrayRef("checked_signal_sizes");
          bool normalized = readAttribute<int64_t>("normalized");
          bool onesided = readAttribute<int64_t>("onesided");
          auto output_sizes = readIntArrayRef("output_sizes");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_fft_with_size(self, signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 266: { // _cufft_get_plan_cache_size
          int64_t device_index = readAttribute<int64_t>("device_index");
          run_op = [=] {
      
              auto the_result = at::_cufft_get_plan_cache_size(device_index);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 267: { // _cufft_get_plan_cache_max_size
          int64_t device_index = readAttribute<int64_t>("device_index");
          run_op = [=] {
      
              auto the_result = at::_cufft_get_plan_cache_max_size(device_index);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 268: { // index
      
          run_op = [=] {
              auto self = peek(0, InputSize());
              auto indices = peekSlice(1, InputSize() - 1, InputSize());
              auto the_result = at::index(self, indices);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 269: { // index_copy
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 3);
              auto index = peek(1, 3);
              auto source = peek(2, 3);
              auto the_result = at::index_copy(self, dim, index, source);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 270: { // index_put
          bool accumulate = readAttribute<int64_t>("accumulate");
          run_op = [=] {
              auto self = peek(0, InputSize());
              auto indices = peekSlice(1, InputSize() - 2, InputSize());
              auto values = peek(1, 2);
              auto the_result = at::index_put(self, indices, values, accumulate);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 271: { // index_put
      
          run_op = [=] {
              auto self = peek(0, InputSize());
              auto indices = peekSlice(1, InputSize() - 2, InputSize());
              auto values = peek(1, 2);
              auto the_result = at::index_put(self, indices, values);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 272: { // instance_norm
          bool use_input_stats = readAttribute<int64_t>("use_input_stats");
          double momentum = readAttribute<float>("momentum");
          double eps = readAttribute<float>("eps");
          bool cudnn_enabled = readAttribute<int64_t>("cudnn_enabled");
          run_op = [=] {
              auto input = peek(0, 5);
              auto weight = peek(1, 5);
              auto bias = peek(2, 5);
              auto running_mean = peek(3, 5);
              auto running_var = peek(4, 5);
              auto the_result = at::instance_norm(input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 273: { // inverse
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::inverse(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 274: { // _inverse_helper
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_inverse_helper(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 275: { // isclose
          double rtol = readAttribute<float>("rtol");
          double atol = readAttribute<float>("atol");
          bool equal_nan = readAttribute<int64_t>("equal_nan");
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::isclose(self, other, rtol, atol, equal_nan);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 276: { // isclose
          double rtol = readAttribute<float>("rtol");
          double atol = readAttribute<float>("atol");
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::isclose(self, other, rtol, atol);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 277: { // isclose
          double rtol = readAttribute<float>("rtol");
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::isclose(self, other, rtol);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 278: { // isclose
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::isclose(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 279: { // isnan
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::isnan(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 280: { // is_distributed
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::is_distributed(self);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 281: { // is_floating_point
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::is_floating_point(self);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 282: { // is_complex
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::is_complex(self);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 283: { // is_nonzero
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::is_nonzero(self);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 284: { // is_same_size
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::is_same_size(self, other);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 285: { // is_signed
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::is_signed(self);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 286: { // kl_div
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::kl_div(self, target, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 287: { // kl_div
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::kl_div(self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 288: { // kl_div_backward
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::kl_div_backward(grad_output, self, target, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 289: { // kl_div_backward
      
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::kl_div_backward(grad_output, self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 290: { // kthvalue
          int64_t k = readAttribute<int64_t>("k");
          int64_t dim = readAttribute<int64_t>("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::kthvalue(self, k, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 291: { // kthvalue
          int64_t k = readAttribute<int64_t>("k");
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::kthvalue(self, k, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 292: { // kthvalue
          int64_t k = readAttribute<int64_t>("k");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::kthvalue(self, k);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 293: { // layer_norm
          auto normalized_shape = readIntArrayRef("normalized_shape");
          double eps = readAttribute<float>("eps");
          bool cudnn_enable = readAttribute<int64_t>("cudnn_enable");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enable);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 294: { // layer_norm
          auto normalized_shape = readIntArrayRef("normalized_shape");
          double eps = readAttribute<float>("eps");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::layer_norm(input, normalized_shape, weight, bias, eps);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 295: { // layer_norm
          auto normalized_shape = readIntArrayRef("normalized_shape");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::layer_norm(input, normalized_shape, weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 296: { // layer_norm
          auto normalized_shape = readIntArrayRef("normalized_shape");
          run_op = [=] {
              auto input = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::layer_norm(input, normalized_shape, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 297: { // layer_norm
          auto normalized_shape = readIntArrayRef("normalized_shape");
          run_op = [=] {
              auto input = peek(0, 1);
              auto the_result = at::layer_norm(input, normalized_shape);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 298: { // native_layer_norm
          int64_t M = readAttribute<int64_t>("M");
          int64_t N = readAttribute<int64_t>("N");
          double eps = readAttribute<float>("eps");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::native_layer_norm(input, weight, bias, M, N, eps);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 299: { // native_layer_norm_backward
          int64_t M = readAttribute<int64_t>("M");
          int64_t N = readAttribute<int64_t>("N");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              auto grad_out = peek(0, 5);
              auto input = peek(1, 5);
              auto mean = peek(2, 5);
              auto rstd = peek(3, 5);
              auto weight = peek(4, 5);
              auto the_result = at::native_layer_norm_backward(grad_out, input, mean, rstd, weight, M, N, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 300: { // native_layer_norm_double_backward
          int64_t M = readAttribute<int64_t>("M");
          int64_t N = readAttribute<int64_t>("N");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              auto ggI = peek(0, 8);
              auto ggW = peek(1, 8);
              auto ggb = peek(2, 8);
              auto gO = peek(3, 8);
              auto input = peek(4, 8);
              auto mean = peek(5, 8);
              auto rstd = peek(6, 8);
              auto weight = peek(7, 8);
              auto the_result = at::native_layer_norm_double_backward(ggI, ggW, ggb, gO, input, mean, rstd, weight, M, N, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 301: { // linear
      
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::linear(input, weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 302: { // linear
      
          run_op = [=] {
              auto input = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::linear(input, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 303: { // mkldnn_linear
      
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::mkldnn_linear(input, weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 304: { // mkldnn_linear
      
          run_op = [=] {
              auto input = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::mkldnn_linear(input, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 305: { // fbgemm_linear_int8_weight_fp32_activation
          at::Scalar weight_scale = readScalarAttribute("weight_scale");
          at::Scalar weight_zero_point = readScalarAttribute("weight_zero_point");
          run_op = [=] {
              auto input = peek(0, 5);
              auto weight = peek(1, 5);
              auto packed = peek(2, 5);
              auto col_offsets = peek(3, 5);
              auto bias = peek(4, 5);
              auto the_result = at::fbgemm_linear_int8_weight_fp32_activation(input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 306: { // fbgemm_linear_int8_weight
          at::Scalar weight_scale = readScalarAttribute("weight_scale");
          at::Scalar weight_zero_point = readScalarAttribute("weight_zero_point");
          run_op = [=] {
              auto input = peek(0, 5);
              auto weight = peek(1, 5);
              auto packed = peek(2, 5);
              auto col_offsets = peek(3, 5);
              auto bias = peek(4, 5);
              auto the_result = at::fbgemm_linear_int8_weight(input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 307: { // fbgemm_pack_gemm_matrix_fp16
      
          run_op = [=] {
              auto input = peek(0, 1);
              auto the_result = at::fbgemm_pack_gemm_matrix_fp16(input);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 308: { // fbgemm_linear_fp16_weight_fp32_activation
      
          run_op = [=] {
              auto input = peek(0, 3);
              auto packed_weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::fbgemm_linear_fp16_weight_fp32_activation(input, packed_weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 309: { // fbgemm_linear_fp16_weight
      
          run_op = [=] {
              auto input = peek(0, 3);
              auto packed_weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::fbgemm_linear_fp16_weight(input, packed_weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 310: { // fbgemm_pack_quantized_matrix
      
          run_op = [=] {
              auto input = peek(0, 1);
              auto the_result = at::fbgemm_pack_quantized_matrix(input);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 311: { // fbgemm_pack_quantized_matrix
          int64_t K = readAttribute<int64_t>("K");
          int64_t N = readAttribute<int64_t>("N");
          run_op = [=] {
              auto input = peek(0, 1);
              auto the_result = at::fbgemm_pack_quantized_matrix(input, K, N);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 312: { // fbgemm_is_cpu_supported
      
          run_op = [=] {
      
              auto the_result = at::fbgemm_is_cpu_supported();
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 313: { // log
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::log(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 314: { // log10
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::log10(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 315: { // log1p
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::log1p(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 316: { // log2
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::log2(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 317: { // logdet
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::logdet(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 318: { // log_softmax
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::log_softmax(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 319: { // _log_softmax
          int64_t dim = readAttribute<int64_t>("dim");
          bool half_to_float = readAttribute<int64_t>("half_to_float");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_log_softmax(self, dim, half_to_float);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 320: { // _log_softmax_backward_data
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto output = peek(1, 3);
              auto self = peek(2, 3);
              auto the_result = at::_log_softmax_backward_data(grad_output, output, dim, self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 321: { // logsumexp
          auto dim = readIntArrayRef("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::logsumexp(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 322: { // logsumexp
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::logsumexp(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 323: { // margin_ranking_loss
          double margin = readAttribute<float>("margin");
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto input1 = peek(0, 3);
              auto input2 = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::margin_ranking_loss(input1, input2, target, margin, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 324: { // margin_ranking_loss
          double margin = readAttribute<float>("margin");
          run_op = [=] {
              auto input1 = peek(0, 3);
              auto input2 = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::margin_ranking_loss(input1, input2, target, margin);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 325: { // margin_ranking_loss
      
          run_op = [=] {
              auto input1 = peek(0, 3);
              auto input2 = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::margin_ranking_loss(input1, input2, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 326: { // matmul
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::matmul(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 327: { // matrix_rank
          double tol = readAttribute<float>("tol");
          bool symmetric = readAttribute<int64_t>("symmetric");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::matrix_rank(self, tol, symmetric);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 328: { // matrix_rank
          double tol = readAttribute<float>("tol");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::matrix_rank(self, tol);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 329: { // matrix_rank
          bool symmetric = readAttribute<int64_t>("symmetric");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::matrix_rank(self, symmetric);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 330: { // matrix_rank
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::matrix_rank(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 331: { // matrix_power
          int64_t n = readAttribute<int64_t>("n");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::matrix_power(self, n);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 332: { // max
          int64_t dim = readAttribute<int64_t>("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 333: { // max
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 334: { // max_values
          auto dim = readIntArrayRef("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_values(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 335: { // max_values
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_values(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 336: { // max_pool1d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool1d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 337: { // max_pool1d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool1d_with_indices(self, kernel_size, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 338: { // max_pool1d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool1d_with_indices(self, kernel_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 339: { // max_pool1d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool1d_with_indices(self, kernel_size, stride);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 340: { // max_pool1d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool1d_with_indices(self, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 341: { // max_pool1d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool1d(self, kernel_size, stride, padding, dilation, ceil_mode);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 342: { // max_pool1d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool1d(self, kernel_size, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 343: { // max_pool1d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool1d(self, kernel_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 344: { // max_pool1d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool1d(self, kernel_size, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 345: { // max_pool1d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool1d(self, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 346: { // max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 347: { // max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool2d(self, kernel_size, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 348: { // max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool2d(self, kernel_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 349: { // max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool2d(self, kernel_size, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 350: { // max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool2d(self, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 351: { // mkldnn_max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::mkldnn_max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 352: { // mkldnn_max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::mkldnn_max_pool2d(self, kernel_size, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 353: { // mkldnn_max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::mkldnn_max_pool2d(self, kernel_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 354: { // mkldnn_max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::mkldnn_max_pool2d(self, kernel_size, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 355: { // mkldnn_max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::mkldnn_max_pool2d(self, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 356: { // quantized_max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::quantized_max_pool2d(self, kernel_size, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 357: { // quantized_max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::quantized_max_pool2d(self, kernel_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 358: { // quantized_max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::quantized_max_pool2d(self, kernel_size, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 359: { // quantized_max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::quantized_max_pool2d(self, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 360: { // max_pool3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool3d(self, kernel_size, stride, padding, dilation, ceil_mode);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 361: { // max_pool3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool3d(self, kernel_size, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 362: { // max_pool3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool3d(self, kernel_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 363: { // max_pool3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool3d(self, kernel_size, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 364: { // max_pool3d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool3d(self, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 365: { // mean
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::mean(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 366: { // mean
          auto dim = readIntArrayRef("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::mean(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 367: { // mean
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::mean(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 368: { // median
          int64_t dim = readAttribute<int64_t>("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::median(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 369: { // median
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::median(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 370: { // min
          int64_t dim = readAttribute<int64_t>("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::min(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 371: { // min
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::min(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 372: { // min_values
          auto dim = readIntArrayRef("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::min_values(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 373: { // min_values
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::min_values(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 374: { // mkldnn_convolution
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::mkldnn_convolution(self, weight, bias, padding, stride, dilation, groups);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 375: { // mkldnn_convolution_backward_input
          auto self_size = readIntArrayRef("self_size");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool bias_defined = readAttribute<int64_t>("bias_defined");
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::mkldnn_convolution_backward_input(self_size, grad_output, weight, padding, stride, dilation, groups, bias_defined);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 376: { // mkldnn_convolution_backward_weights
          auto weight_size = readIntArrayRef("weight_size");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool bias_defined = readAttribute<int64_t>("bias_defined");
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::mkldnn_convolution_backward_weights(weight_size, grad_output, self, padding, stride, dilation, groups, bias_defined);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 377: { // mkldnn_convolution_backward
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              auto self = peek(0, 3);
              auto grad_output = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::mkldnn_convolution_backward(self, grad_output, weight, padding, stride, dilation, groups, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 378: { // miopen_batch_norm
          bool training = readAttribute<int64_t>("training");
          double exponential_average_factor = readAttribute<float>("exponential_average_factor");
          double epsilon = readAttribute<float>("epsilon");
          run_op = [=] {
              auto input = peek(0, 5);
              auto weight = peek(1, 5);
              auto bias = peek(2, 5);
              auto running_mean = peek(3, 5);
              auto running_var = peek(4, 5);
              auto the_result = at::miopen_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 379: { // miopen_batch_norm_backward
          double epsilon = readAttribute<float>("epsilon");
          run_op = [=] {
              auto input = peek(0, 7);
              auto grad_output = peek(1, 7);
              auto weight = peek(2, 7);
              auto running_mean = peek(3, 7);
              auto running_var = peek(4, 7);
              auto save_mean = peek(5, 7);
              auto save_var = peek(6, 7);
              auto the_result = at::miopen_batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 380: { // miopen_convolution
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::miopen_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 381: { // miopen_convolution_backward_input
          auto self_size = readIntArrayRef("self_size");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::miopen_convolution_backward_input(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 382: { // miopen_convolution_backward
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              auto self = peek(0, 3);
              auto grad_output = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::miopen_convolution_backward(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 383: { // miopen_convolution_backward_bias
      
          run_op = [=] {
              auto grad_output = peek(0, 1);
              auto the_result = at::miopen_convolution_backward_bias(grad_output);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 384: { // miopen_convolution_backward_weight
          auto weight_size = readIntArrayRef("weight_size");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::miopen_convolution_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 385: { // miopen_convolution_transpose
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::miopen_convolution_transpose(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 386: { // miopen_convolution_transpose_backward
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              auto self = peek(0, 3);
              auto grad_output = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::miopen_convolution_transpose_backward(self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 387: { // miopen_convolution_transpose_backward_input
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::miopen_convolution_transpose_backward_input(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 388: { // miopen_convolution_transpose_backward_weight
          auto weight_size = readIntArrayRef("weight_size");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::miopen_convolution_transpose_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 389: { // miopen_depthwise_convolution
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::miopen_depthwise_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 390: { // miopen_depthwise_convolution_backward_input
          auto self_size = readIntArrayRef("self_size");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::miopen_depthwise_convolution_backward_input(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 391: { // miopen_depthwise_convolution_backward
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              auto self = peek(0, 3);
              auto grad_output = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::miopen_depthwise_convolution_backward(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 392: { // miopen_depthwise_convolution_backward_weight
          auto weight_size = readIntArrayRef("weight_size");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::miopen_depthwise_convolution_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 393: { // miopen_rnn
          int64_t weight_stride0 = readAttribute<int64_t>("weight_stride0");
          int64_t mode = readAttribute<int64_t>("mode");
          int64_t hidden_size = readAttribute<int64_t>("hidden_size");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          bool batch_first = readAttribute<int64_t>("batch_first");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          auto batch_sizes = readIntArrayRef("batch_sizes");
          run_op = [=] {
              auto input = peek(0, InputSize());
              auto weight = peekSlice(1, InputSize() - 4, InputSize());
              auto hx = peek(1, 4);
              auto cx = peek(2, 4);
              auto dropout_state = peek(3, 4);
              auto the_result = at::miopen_rnn(input, weight, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
                if(OutputSize() > 4) {assignTo(Output(4),std::get<4>(the_result));}
              return true;
          };
      } break;
      case 394: { // mm
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto mat2 = peek(1, 2);
              auto the_result = at::mm(self, mat2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 395: { // _sparse_mm
      
          run_op = [=] {
              auto sparse = peek(0, 2);
              auto dense = peek(1, 2);
              auto the_result = at::_sparse_mm(sparse, dense);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 396: { // mode
          int64_t dim = readAttribute<int64_t>("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::mode(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 397: { // mode
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::mode(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 398: { // mode
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::mode(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 399: { // mul
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::mul(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 400: { // mul
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::mul(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 401: { // mv
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto vec = peek(1, 2);
              auto the_result = at::mv(self, vec);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 402: { // mvlgamma
          int64_t p = readAttribute<int64_t>("p");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::mvlgamma(self, p);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 403: { // narrow_copy
          int64_t dim = readAttribute<int64_t>("dim");
          int64_t start = readAttribute<int64_t>("start");
          int64_t length = readAttribute<int64_t>("length");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.narrow_copy(dim, start, length);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 404: { // narrow
          int64_t dim = readAttribute<int64_t>("dim");
          int64_t start = readAttribute<int64_t>("start");
          int64_t length = readAttribute<int64_t>("length");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::narrow(self, dim, start, length);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 405: { // native_batch_norm
          bool training = readAttribute<int64_t>("training");
          double momentum = readAttribute<float>("momentum");
          double eps = readAttribute<float>("eps");
          run_op = [=] {
              auto input = peek(0, 5);
              auto weight = peek(1, 5);
              auto bias = peek(2, 5);
              auto running_mean = peek(3, 5);
              auto running_var = peek(4, 5);
              auto the_result = at::native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 406: { // batch_norm_stats
          double eps = readAttribute<float>("eps");
          run_op = [=] {
              auto input = peek(0, 1);
              auto the_result = at::batch_norm_stats(input, eps);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 407: { // batch_norm_elemt
          double eps = readAttribute<float>("eps");
          run_op = [=] {
              auto input = peek(0, 5);
              auto weight = peek(1, 5);
              auto bias = peek(2, 5);
              auto mean = peek(3, 5);
              auto invstd = peek(4, 5);
              auto the_result = at::batch_norm_elemt(input, weight, bias, mean, invstd, eps);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 408: { // batch_norm_gather_stats
          double momentum = readAttribute<float>("momentum");
          double eps = readAttribute<float>("eps");
          int64_t count = readAttribute<int64_t>("count");
          run_op = [=] {
              auto input = peek(0, 5);
              auto mean = peek(1, 5);
              auto invstd = peek(2, 5);
              auto running_mean = peek(3, 5);
              auto running_var = peek(4, 5);
              auto the_result = at::batch_norm_gather_stats(input, mean, invstd, running_mean, running_var, momentum, eps, count);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 409: { // batch_norm_gather_stats_with_counts
          double momentum = readAttribute<float>("momentum");
          double eps = readAttribute<float>("eps");
          auto counts = readIntArrayRef("counts");
          run_op = [=] {
              auto input = peek(0, 5);
              auto mean = peek(1, 5);
              auto invstd = peek(2, 5);
              auto running_mean = peek(3, 5);
              auto running_var = peek(4, 5);
              auto the_result = at::batch_norm_gather_stats_with_counts(input, mean, invstd, running_mean, running_var, momentum, eps, counts);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 410: { // native_batch_norm_backward
          bool train = readAttribute<int64_t>("train");
          double eps = readAttribute<float>("eps");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              auto grad_out = peek(0, 7);
              auto input = peek(1, 7);
              auto weight = peek(2, 7);
              auto running_mean = peek(3, 7);
              auto running_var = peek(4, 7);
              auto save_mean = peek(5, 7);
              auto save_invstd = peek(6, 7);
              auto the_result = at::native_batch_norm_backward(grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 411: { // batch_norm_backward_reduce
          bool input_g = readAttribute<int64_t>("input_g");
          bool weight_g = readAttribute<int64_t>("weight_g");
          bool bias_g = readAttribute<int64_t>("bias_g");
          run_op = [=] {
              auto grad_out = peek(0, 5);
              auto input = peek(1, 5);
              auto mean = peek(2, 5);
              auto invstd = peek(3, 5);
              auto weight = peek(4, 5);
              auto the_result = at::batch_norm_backward_reduce(grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 412: { // batch_norm_backward_elemt
      
          run_op = [=] {
              auto grad_out = peek(0, 7);
              auto input = peek(1, 7);
              auto mean = peek(2, 7);
              auto invstd = peek(3, 7);
              auto weight = peek(4, 7);
              auto mean_dy = peek(5, 7);
              auto mean_dy_xmu = peek(6, 7);
              auto the_result = at::batch_norm_backward_elemt(grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 413: { // batch_norm_update_stats
          double momentum = readAttribute<float>("momentum");
          run_op = [=] {
              auto input = peek(0, 3);
              auto running_mean = peek(1, 3);
              auto running_var = peek(2, 3);
              auto the_result = at::batch_norm_update_stats(input, running_mean, running_var, momentum);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 414: { // _nnpack_available
      
          run_op = [=] {
      
              auto the_result = at::_nnpack_available();
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 415: { // _nnpack_spatial_convolution
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::_nnpack_spatial_convolution(input, weight, bias, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 416: { // _nnpack_spatial_convolution_backward
          auto padding = readIntArrayRef("padding");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              auto input = peek(0, 3);
              auto grad_output = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::_nnpack_spatial_convolution_backward(input, grad_output, weight, padding, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 417: { // _nnpack_spatial_convolution_backward_input
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto input = peek(0, 3);
              auto grad_output = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::_nnpack_spatial_convolution_backward_input(input, grad_output, weight, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 418: { // _nnpack_spatial_convolution_backward_weight
          auto weightsize = readIntArrayRef("weightsize");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto input = peek(0, 2);
              auto grad_output = peek(1, 2);
              auto the_result = at::_nnpack_spatial_convolution_backward_weight(input, weightsize, grad_output, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 419: { // pairwise_distance
          double p = readAttribute<float>("p");
          double eps = readAttribute<float>("eps");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              auto x1 = peek(0, 2);
              auto x2 = peek(1, 2);
              auto the_result = at::pairwise_distance(x1, x2, p, eps, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 420: { // pairwise_distance
          double p = readAttribute<float>("p");
          double eps = readAttribute<float>("eps");
          run_op = [=] {
              auto x1 = peek(0, 2);
              auto x2 = peek(1, 2);
              auto the_result = at::pairwise_distance(x1, x2, p, eps);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 421: { // pairwise_distance
          double p = readAttribute<float>("p");
          run_op = [=] {
              auto x1 = peek(0, 2);
              auto x2 = peek(1, 2);
              auto the_result = at::pairwise_distance(x1, x2, p);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 422: { // pairwise_distance
      
          run_op = [=] {
              auto x1 = peek(0, 2);
              auto x2 = peek(1, 2);
              auto the_result = at::pairwise_distance(x1, x2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 423: { // cdist
          double p = readAttribute<float>("p");
          run_op = [=] {
              auto x1 = peek(0, 2);
              auto x2 = peek(1, 2);
              auto the_result = at::cdist(x1, x2, p);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 424: { // cdist
      
          run_op = [=] {
              auto x1 = peek(0, 2);
              auto x2 = peek(1, 2);
              auto the_result = at::cdist(x1, x2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 425: { // _cdist_backward
          double p = readAttribute<float>("p");
          run_op = [=] {
              auto grad = peek(0, 4);
              auto x1 = peek(1, 4);
              auto x2 = peek(2, 4);
              auto cdist = peek(3, 4);
              auto the_result = at::_cdist_backward(grad, x1, x2, p, cdist);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 426: { // pdist
          double p = readAttribute<float>("p");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::pdist(self, p);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 427: { // pdist
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::pdist(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 428: { // _pdist_forward
          double p = readAttribute<float>("p");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_pdist_forward(self, p);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 429: { // _pdist_forward
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_pdist_forward(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 430: { // _pdist_backward
          double p = readAttribute<float>("p");
          run_op = [=] {
              auto grad = peek(0, 3);
              auto self = peek(1, 3);
              auto pdist = peek(2, 3);
              auto the_result = at::_pdist_backward(grad, self, p, pdist);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 431: { // cosine_similarity
          int64_t dim = readAttribute<int64_t>("dim");
          double eps = readAttribute<float>("eps");
          run_op = [=] {
              auto x1 = peek(0, 2);
              auto x2 = peek(1, 2);
              auto the_result = at::cosine_similarity(x1, x2, dim, eps);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 432: { // cosine_similarity
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto x1 = peek(0, 2);
              auto x2 = peek(1, 2);
              auto the_result = at::cosine_similarity(x1, x2, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 433: { // cosine_similarity
      
          run_op = [=] {
              auto x1 = peek(0, 2);
              auto x2 = peek(1, 2);
              auto the_result = at::cosine_similarity(x1, x2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 434: { // permute
          auto dims = readIntArrayRef("dims");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.permute(dims);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 435: { // numpy_T
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.numpy_T();
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 436: { // pixel_shuffle
          int64_t upscale_factor = readAttribute<int64_t>("upscale_factor");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::pixel_shuffle(self, upscale_factor);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 437: { // is_pinned
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.is_pinned();
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 438: { // pin_memory
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.pin_memory();
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 439: { // pinverse
          double rcond = readAttribute<float>("rcond");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::pinverse(self, rcond);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 440: { // pinverse
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::pinverse(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 441: { // poisson_nll_loss
          bool log_input = readAttribute<int64_t>("log_input");
          bool full = readAttribute<int64_t>("full");
          double eps = readAttribute<float>("eps");
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto input = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::poisson_nll_loss(input, target, log_input, full, eps, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 442: { // reciprocal
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::reciprocal(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 443: { // neg
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::neg(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 444: { // repeat
          auto repeats = readIntArrayRef("repeats");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.repeat(repeats);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 445: { // repeat_interleave
      
          run_op = [=] {
              auto repeats = peek(0, 1);
              auto the_result = at::repeat_interleave(repeats);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 446: { // repeat_interleave
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto repeats = peek(1, 2);
              auto the_result = at::repeat_interleave(self, repeats);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 447: { // repeat_interleave
          int64_t repeats = readAttribute<int64_t>("repeats");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::repeat_interleave(self, repeats);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 448: { // reshape
          auto shape = readIntArrayRef("shape");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::reshape(self, shape);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 449: { // _mkldnn_reshape
          auto shape = readIntArrayRef("shape");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_mkldnn_reshape(self, shape);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 450: { // reshape_as
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = self.reshape_as(other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 451: { // round
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::round(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 452: { // rrelu
          at::Scalar lower = readScalarAttribute("lower");
          at::Scalar upper = readScalarAttribute("upper");
          bool training = readAttribute<int64_t>("training");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::rrelu(self, lower, upper, training);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 453: { // rrelu
          at::Scalar lower = readScalarAttribute("lower");
          at::Scalar upper = readScalarAttribute("upper");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::rrelu(self, lower, upper);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 454: { // rrelu
          at::Scalar lower = readScalarAttribute("lower");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::rrelu(self, lower);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 455: { // rrelu
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::rrelu(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 456: { // relu
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::relu(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 457: { // prelu
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::prelu(self, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 458: { // prelu_backward
      
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::prelu_backward(grad_output, self, weight);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 459: { // gelu
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::gelu(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 460: { // gelu_backward
      
          run_op = [=] {
              auto grad = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::gelu_backward(grad, self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 461: { // hardshrink
          at::Scalar lambd = readScalarAttribute("lambd");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::hardshrink(self, lambd);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 462: { // hardshrink
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::hardshrink(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 463: { // hardshrink_backward
          at::Scalar lambd = readScalarAttribute("lambd");
          run_op = [=] {
              auto grad_out = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::hardshrink_backward(grad_out, self, lambd);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 464: { // rsqrt
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::rsqrt(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 465: { // select
          int64_t dim = readAttribute<int64_t>("dim");
          int64_t index = readAttribute<int64_t>("index");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::select(self, dim, index);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 466: { // selu
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::selu(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 467: { // celu
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::celu(self, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 468: { // celu
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::celu(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 469: { // sigmoid
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::sigmoid(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 470: { // sin
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::sin(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 471: { // sinh
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::sinh(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 472: { // detach
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::detach(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 473: { // size
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::size(self, dim);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 474: { // slice
          int64_t dim = readAttribute<int64_t>("dim");
          int64_t start = readAttribute<int64_t>("start");
          int64_t end = readAttribute<int64_t>("end");
          int64_t step = readAttribute<int64_t>("step");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::slice(self, dim, start, end, step);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 475: { // slice
          int64_t dim = readAttribute<int64_t>("dim");
          int64_t start = readAttribute<int64_t>("start");
          int64_t end = readAttribute<int64_t>("end");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::slice(self, dim, start, end);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 476: { // slice
          int64_t dim = readAttribute<int64_t>("dim");
          int64_t start = readAttribute<int64_t>("start");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::slice(self, dim, start);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 477: { // slice
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::slice(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 478: { // slice
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::slice(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 479: { // slogdet
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::slogdet(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 480: { // smm
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto mat2 = peek(1, 2);
              auto the_result = at::smm(self, mat2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 481: { // softmax
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::softmax(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 482: { // _softmax
          int64_t dim = readAttribute<int64_t>("dim");
          bool half_to_float = readAttribute<int64_t>("half_to_float");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_softmax(self, dim, half_to_float);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 483: { // _softmax_backward_data
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto output = peek(1, 3);
              auto self = peek(2, 3);
              auto the_result = at::_softmax_backward_data(grad_output, output, dim, self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 484: { // split
          int64_t split_size = readAttribute<int64_t>("split_size");
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::split(self, split_size, dim);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 485: { // split
          int64_t split_size = readAttribute<int64_t>("split_size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::split(self, split_size);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 486: { // split_with_sizes
          auto split_sizes = readIntArrayRef("split_sizes");
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::split_with_sizes(self, split_sizes, dim);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 487: { // split_with_sizes
          auto split_sizes = readIntArrayRef("split_sizes");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::split_with_sizes(self, split_sizes);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 488: { // squeeze
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::squeeze(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 489: { // squeeze
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::squeeze(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 490: { // sspaddmm
          at::Scalar beta = readScalarAttribute("beta");
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              auto self = peek(0, 3);
              auto mat1 = peek(1, 3);
              auto mat2 = peek(2, 3);
              auto the_result = at::sspaddmm(self, mat1, mat2, beta, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 491: { // sspaddmm
          at::Scalar beta = readScalarAttribute("beta");
          run_op = [=] {
              auto self = peek(0, 3);
              auto mat1 = peek(1, 3);
              auto mat2 = peek(2, 3);
              auto the_result = at::sspaddmm(self, mat1, mat2, beta);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 492: { // sspaddmm
      
          run_op = [=] {
              auto self = peek(0, 3);
              auto mat1 = peek(1, 3);
              auto mat2 = peek(2, 3);
              auto the_result = at::sspaddmm(self, mat1, mat2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 493: { // stack
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto tensors = peekSlice(0, InputSize() - 0, InputSize());
              auto the_result = at::stack(tensors, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 494: { // stack
      
          run_op = [=] {
              auto tensors = peekSlice(0, InputSize() - 0, InputSize());
              auto the_result = at::stack(tensors);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 495: { // stft
          int64_t n_fft = readAttribute<int64_t>("n_fft");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::stft(self, n_fft);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 496: { // stride
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::stride(self, dim);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 497: { // sum
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::sum(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 498: { // sum
          auto dim = readIntArrayRef("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::sum(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 499: { // sum
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::sum(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 500: { // sum_to_size
          auto size = readIntArrayRef("size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.sum_to_size(size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 501: { // sqrt
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::sqrt(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 502: { // std
          bool unbiased = readAttribute<int64_t>("unbiased");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::std(self, unbiased);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 503: { // std
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::std(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 504: { // std
          auto dim = readIntArrayRef("dim");
          bool unbiased = readAttribute<int64_t>("unbiased");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::std(self, dim, unbiased, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 505: { // std
          auto dim = readIntArrayRef("dim");
          bool unbiased = readAttribute<int64_t>("unbiased");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::std(self, dim, unbiased);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 506: { // std
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::std(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 507: { // std_mean
          bool unbiased = readAttribute<int64_t>("unbiased");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::std_mean(self, unbiased);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 508: { // std_mean
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::std_mean(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 509: { // std_mean
          auto dim = readIntArrayRef("dim");
          bool unbiased = readAttribute<int64_t>("unbiased");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::std_mean(self, dim, unbiased, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 510: { // std_mean
          auto dim = readIntArrayRef("dim");
          bool unbiased = readAttribute<int64_t>("unbiased");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::std_mean(self, dim, unbiased);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 511: { // std_mean
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::std_mean(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 512: { // prod
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::prod(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 513: { // prod
          int64_t dim = readAttribute<int64_t>("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::prod(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 514: { // prod
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::prod(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 515: { // t
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::t(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 516: { // tan
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::tan(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 517: { // tanh
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::tanh(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 518: { // tensordot
          auto dims_self = readIntArrayRef("dims_self");
          auto dims_other = readIntArrayRef("dims_other");
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::tensordot(self, other, dims_self, dims_other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 519: { // threshold
          at::Scalar threshold = readScalarAttribute("threshold");
          at::Scalar value = readScalarAttribute("value");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::threshold(self, threshold, value);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 520: { // threshold_backward
          at::Scalar threshold = readScalarAttribute("threshold");
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::threshold_backward(grad_output, self, threshold);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 521: { // transpose
          int64_t dim0 = readAttribute<int64_t>("dim0");
          int64_t dim1 = readAttribute<int64_t>("dim1");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::transpose(self, dim0, dim1);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 522: { // _mkldnn_transpose
          int64_t dim0 = readAttribute<int64_t>("dim0");
          int64_t dim1 = readAttribute<int64_t>("dim1");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_mkldnn_transpose(self, dim0, dim1);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 523: { // one_hot
          int64_t num_classes = readAttribute<int64_t>("num_classes");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::one_hot(self, num_classes);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 524: { // one_hot
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::one_hot(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 525: { // flip
          auto dims = readIntArrayRef("dims");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::flip(self, dims);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 526: { // roll
          auto shifts = readIntArrayRef("shifts");
          auto dims = readIntArrayRef("dims");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::roll(self, shifts, dims);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 527: { // roll
          auto shifts = readIntArrayRef("shifts");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::roll(self, shifts);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 528: { // rot90
          int64_t k = readAttribute<int64_t>("k");
          auto dims = readIntArrayRef("dims");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::rot90(self, k, dims);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 529: { // rot90
          int64_t k = readAttribute<int64_t>("k");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::rot90(self, k);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 530: { // rot90
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::rot90(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 531: { // trapz
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto y = peek(0, 2);
              auto x = peek(1, 2);
              auto the_result = at::trapz(y, x, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 532: { // trapz
      
          run_op = [=] {
              auto y = peek(0, 2);
              auto x = peek(1, 2);
              auto the_result = at::trapz(y, x);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 533: { // trapz
          double dx = readAttribute<float>("dx");
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto y = peek(0, 1);
              auto the_result = at::trapz(y, dx, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 534: { // trapz
          double dx = readAttribute<float>("dx");
          run_op = [=] {
              auto y = peek(0, 1);
              auto the_result = at::trapz(y, dx);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 535: { // trapz
      
          run_op = [=] {
              auto y = peek(0, 1);
              auto the_result = at::trapz(y);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 536: { // _trilinear
          auto expand1 = readIntArrayRef("expand1");
          auto expand2 = readIntArrayRef("expand2");
          auto expand3 = readIntArrayRef("expand3");
          auto sumdim = readIntArrayRef("sumdim");
          int64_t unroll_dim = readAttribute<int64_t>("unroll_dim");
          run_op = [=] {
              auto i1 = peek(0, 3);
              auto i2 = peek(1, 3);
              auto i3 = peek(2, 3);
              auto the_result = at::_trilinear(i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 537: { // _trilinear
          auto expand1 = readIntArrayRef("expand1");
          auto expand2 = readIntArrayRef("expand2");
          auto expand3 = readIntArrayRef("expand3");
          auto sumdim = readIntArrayRef("sumdim");
          run_op = [=] {
              auto i1 = peek(0, 3);
              auto i2 = peek(1, 3);
              auto i3 = peek(2, 3);
              auto the_result = at::_trilinear(i1, i2, i3, expand1, expand2, expand3, sumdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 538: { // triplet_margin_loss
          double margin = readAttribute<float>("margin");
          double p = readAttribute<float>("p");
          double eps = readAttribute<float>("eps");
          bool swap = readAttribute<int64_t>("swap");
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto anchor = peek(0, 3);
              auto positive = peek(1, 3);
              auto negative = peek(2, 3);
              auto the_result = at::triplet_margin_loss(anchor, positive, negative, margin, p, eps, swap, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 539: { // triplet_margin_loss
          double margin = readAttribute<float>("margin");
          double p = readAttribute<float>("p");
          double eps = readAttribute<float>("eps");
          bool swap = readAttribute<int64_t>("swap");
          run_op = [=] {
              auto anchor = peek(0, 3);
              auto positive = peek(1, 3);
              auto negative = peek(2, 3);
              auto the_result = at::triplet_margin_loss(anchor, positive, negative, margin, p, eps, swap);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 540: { // triplet_margin_loss
          double margin = readAttribute<float>("margin");
          double p = readAttribute<float>("p");
          double eps = readAttribute<float>("eps");
          run_op = [=] {
              auto anchor = peek(0, 3);
              auto positive = peek(1, 3);
              auto negative = peek(2, 3);
              auto the_result = at::triplet_margin_loss(anchor, positive, negative, margin, p, eps);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 541: { // triplet_margin_loss
          double margin = readAttribute<float>("margin");
          double p = readAttribute<float>("p");
          run_op = [=] {
              auto anchor = peek(0, 3);
              auto positive = peek(1, 3);
              auto negative = peek(2, 3);
              auto the_result = at::triplet_margin_loss(anchor, positive, negative, margin, p);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 542: { // triplet_margin_loss
          double margin = readAttribute<float>("margin");
          run_op = [=] {
              auto anchor = peek(0, 3);
              auto positive = peek(1, 3);
              auto negative = peek(2, 3);
              auto the_result = at::triplet_margin_loss(anchor, positive, negative, margin);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 543: { // triplet_margin_loss
      
          run_op = [=] {
              auto anchor = peek(0, 3);
              auto positive = peek(1, 3);
              auto negative = peek(2, 3);
              auto the_result = at::triplet_margin_loss(anchor, positive, negative);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 544: { // trunc
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::trunc(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 545: { // type_as
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = self.type_as(other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 546: { // _has_compatible_shallow_copy_type
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto from = peek(1, 2);
              auto the_result = at::_has_compatible_shallow_copy_type(self, from);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 547: { // _unique
          bool sorted = readAttribute<int64_t>("sorted");
          bool return_inverse = readAttribute<int64_t>("return_inverse");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_unique(self, sorted, return_inverse);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 548: { // _unique
          bool sorted = readAttribute<int64_t>("sorted");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_unique(self, sorted);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 549: { // _unique
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_unique(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 550: { // unique_dim
          int64_t dim = readAttribute<int64_t>("dim");
          bool sorted = readAttribute<int64_t>("sorted");
          bool return_inverse = readAttribute<int64_t>("return_inverse");
          bool return_counts = readAttribute<int64_t>("return_counts");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::unique_dim(self, dim, sorted, return_inverse, return_counts);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 551: { // unique_dim
          int64_t dim = readAttribute<int64_t>("dim");
          bool sorted = readAttribute<int64_t>("sorted");
          bool return_inverse = readAttribute<int64_t>("return_inverse");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::unique_dim(self, dim, sorted, return_inverse);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 552: { // unique_dim
          int64_t dim = readAttribute<int64_t>("dim");
          bool sorted = readAttribute<int64_t>("sorted");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::unique_dim(self, dim, sorted);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 553: { // unique_dim
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::unique_dim(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 554: { // unique_consecutive
          bool return_inverse = readAttribute<int64_t>("return_inverse");
          bool return_counts = readAttribute<int64_t>("return_counts");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::unique_consecutive(self, return_inverse, return_counts);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 555: { // unique_consecutive
          bool return_inverse = readAttribute<int64_t>("return_inverse");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::unique_consecutive(self, return_inverse);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 556: { // unique_consecutive
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::unique_consecutive(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 557: { // unique_dim_consecutive
          int64_t dim = readAttribute<int64_t>("dim");
          bool return_inverse = readAttribute<int64_t>("return_inverse");
          bool return_counts = readAttribute<int64_t>("return_counts");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::unique_dim_consecutive(self, dim, return_inverse, return_counts);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 558: { // unique_dim_consecutive
          int64_t dim = readAttribute<int64_t>("dim");
          bool return_inverse = readAttribute<int64_t>("return_inverse");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::unique_dim_consecutive(self, dim, return_inverse);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 559: { // unique_dim_consecutive
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::unique_dim_consecutive(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 560: { // _unique2
          bool sorted = readAttribute<int64_t>("sorted");
          bool return_inverse = readAttribute<int64_t>("return_inverse");
          bool return_counts = readAttribute<int64_t>("return_counts");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_unique2(self, sorted, return_inverse, return_counts);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 561: { // _unique2
          bool sorted = readAttribute<int64_t>("sorted");
          bool return_inverse = readAttribute<int64_t>("return_inverse");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_unique2(self, sorted, return_inverse);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 562: { // _unique2
          bool sorted = readAttribute<int64_t>("sorted");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_unique2(self, sorted);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 563: { // _unique2
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_unique2(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 564: { // _unsafe_view
          auto size = readIntArrayRef("size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_unsafe_view(self, size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 565: { // unsqueeze
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::unsqueeze(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 566: { // var
          bool unbiased = readAttribute<int64_t>("unbiased");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::var(self, unbiased);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 567: { // var
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::var(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 568: { // var
          auto dim = readIntArrayRef("dim");
          bool unbiased = readAttribute<int64_t>("unbiased");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::var(self, dim, unbiased, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 569: { // var
          auto dim = readIntArrayRef("dim");
          bool unbiased = readAttribute<int64_t>("unbiased");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::var(self, dim, unbiased);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 570: { // var
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::var(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 571: { // var_mean
          bool unbiased = readAttribute<int64_t>("unbiased");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::var_mean(self, unbiased);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 572: { // var_mean
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::var_mean(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 573: { // var_mean
          auto dim = readIntArrayRef("dim");
          bool unbiased = readAttribute<int64_t>("unbiased");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::var_mean(self, dim, unbiased, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 574: { // var_mean
          auto dim = readIntArrayRef("dim");
          bool unbiased = readAttribute<int64_t>("unbiased");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::var_mean(self, dim, unbiased);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 575: { // var_mean
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::var_mean(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 576: { // view_as
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = self.view_as(other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 577: { // where
      
          run_op = [=] {
              auto condition = peek(0, 3);
              auto self = peek(1, 3);
              auto other = peek(2, 3);
              auto the_result = at::where(condition, self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 578: { // where
      
          run_op = [=] {
              auto condition = peek(0, 1);
              auto the_result = at::where(condition);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 579: { // _s_where
      
          run_op = [=] {
              auto condition = peek(0, 3);
              auto self = peek(1, 3);
              auto other = peek(2, 3);
              auto the_result = at::_s_where(condition, self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 580: { // norm_except_dim
          int64_t pow = readAttribute<int64_t>("pow");
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto v = peek(0, 1);
              auto the_result = at::norm_except_dim(v, pow, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 581: { // norm_except_dim
          int64_t pow = readAttribute<int64_t>("pow");
          run_op = [=] {
              auto v = peek(0, 1);
              auto the_result = at::norm_except_dim(v, pow);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 582: { // norm_except_dim
      
          run_op = [=] {
              auto v = peek(0, 1);
              auto the_result = at::norm_except_dim(v);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 583: { // _weight_norm
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto v = peek(0, 2);
              auto g = peek(1, 2);
              auto the_result = at::_weight_norm(v, g, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 584: { // _weight_norm
      
          run_op = [=] {
              auto v = peek(0, 2);
              auto g = peek(1, 2);
              auto the_result = at::_weight_norm(v, g);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 585: { // _weight_norm_cuda_interface
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto v = peek(0, 2);
              auto g = peek(1, 2);
              auto the_result = at::_weight_norm_cuda_interface(v, g, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 586: { // _weight_norm_cuda_interface
      
          run_op = [=] {
              auto v = peek(0, 2);
              auto g = peek(1, 2);
              auto the_result = at::_weight_norm_cuda_interface(v, g);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 587: { // _weight_norm_cuda_interface_backward
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto grad_w = peek(0, 4);
              auto saved_v = peek(1, 4);
              auto saved_g = peek(2, 4);
              auto saved_norms = peek(3, 4);
              auto the_result = at::_weight_norm_cuda_interface_backward(grad_w, saved_v, saved_g, saved_norms, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 588: { // _weight_norm_differentiable_backward
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto grad_w = peek(0, 4);
              auto saved_v = peek(1, 4);
              auto saved_g = peek(2, 4);
              auto saved_norms = peek(3, 4);
              auto the_result = at::_weight_norm_differentiable_backward(grad_w, saved_v, saved_g, saved_norms, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 589: { // _standard_gamma_grad
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto output = peek(1, 2);
              auto the_result = at::_standard_gamma_grad(self, output);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 590: { // _standard_gamma
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_standard_gamma(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 591: { // _dirichlet_grad
      
          run_op = [=] {
              auto x = peek(0, 3);
              auto alpha = peek(1, 3);
              auto total = peek(2, 3);
              auto the_result = at::_dirichlet_grad(x, alpha, total);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 592: { // _sample_dirichlet
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_sample_dirichlet(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 593: { // poisson
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::poisson(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 594: { // native_norm
          at::Scalar p = readScalarAttribute("p");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::native_norm(self, p);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 595: { // native_norm
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::native_norm(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 596: { // _sparse_sum
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_sparse_sum(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 597: { // _sparse_sum
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_sparse_sum(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 598: { // _sparse_sum_backward
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              auto grad = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::_sparse_sum_backward(grad, self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 599: { // norm
          at::Scalar p = readScalarAttribute("p");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::norm(self, p);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 600: { // norm
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::norm(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 601: { // frobenius_norm
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::frobenius_norm(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 602: { // frobenius_norm
          auto dim = readIntArrayRef("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::frobenius_norm(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 603: { // frobenius_norm
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::frobenius_norm(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 604: { // nuclear_norm
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::nuclear_norm(self, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 605: { // nuclear_norm
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::nuclear_norm(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 606: { // nuclear_norm
          auto dim = readIntArrayRef("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::nuclear_norm(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 607: { // nuclear_norm
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::nuclear_norm(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 608: { // clone
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::clone(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 609: { // pow
          at::Scalar exponent = readScalarAttribute("exponent");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::pow(self, exponent);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 610: { // sub
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::sub(self, other, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 611: { // sub
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::sub(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 612: { // sub
          at::Scalar other = readScalarAttribute("other");
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::sub(self, other, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 613: { // sub
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::sub(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 614: { // rsub
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::rsub(self, other, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 615: { // rsub
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::rsub(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 616: { // rsub
          at::Scalar other = readScalarAttribute("other");
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::rsub(self, other, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 617: { // rsub
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::rsub(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 618: { // _sparse_addmm
          at::Scalar beta = readScalarAttribute("beta");
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              auto self = peek(0, 3);
              auto sparse = peek(1, 3);
              auto dense = peek(2, 3);
              auto the_result = at::_sparse_addmm(self, sparse, dense, beta, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 619: { // _sparse_addmm
          at::Scalar beta = readScalarAttribute("beta");
          run_op = [=] {
              auto self = peek(0, 3);
              auto sparse = peek(1, 3);
              auto dense = peek(2, 3);
              auto the_result = at::_sparse_addmm(self, sparse, dense, beta);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 620: { // _sparse_addmm
      
          run_op = [=] {
              auto self = peek(0, 3);
              auto sparse = peek(1, 3);
              auto dense = peek(2, 3);
              auto the_result = at::_sparse_addmm(self, sparse, dense);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 621: { // addmm
          at::Scalar beta = readScalarAttribute("beta");
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              auto self = peek(0, 3);
              auto mat1 = peek(1, 3);
              auto mat2 = peek(2, 3);
              auto the_result = at::addmm(self, mat1, mat2, beta, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 622: { // addmm
          at::Scalar beta = readScalarAttribute("beta");
          run_op = [=] {
              auto self = peek(0, 3);
              auto mat1 = peek(1, 3);
              auto mat2 = peek(2, 3);
              auto the_result = at::addmm(self, mat1, mat2, beta);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 623: { // addmm
      
          run_op = [=] {
              auto self = peek(0, 3);
              auto mat1 = peek(1, 3);
              auto mat2 = peek(2, 3);
              auto the_result = at::addmm(self, mat1, mat2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 624: { // sparse_mask
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto mask = peek(1, 2);
              auto the_result = self.sparse_mask(mask);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 625: { // to_dense
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.to_dense();
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 626: { // to_dense_backward
      
          run_op = [=] {
              auto grad = peek(0, 2);
              auto input = peek(1, 2);
              auto the_result = at::to_dense_backward(grad, input);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 627: { // sparse_dim
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.sparse_dim();
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 628: { // _dimI
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self._dimI();
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 629: { // dense_dim
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.dense_dim();
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 630: { // _dimV
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self._dimV();
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 631: { // _nnz
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self._nnz();
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 632: { // coalesce
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.coalesce();
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 633: { // is_coalesced
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.is_coalesced();
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 634: { // _indices
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self._indices();
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 635: { // _values
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self._values();
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 636: { // indices
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.indices();
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 637: { // values
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.values();
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 638: { // hspmm
      
          run_op = [=] {
              auto mat1 = peek(0, 2);
              auto mat2 = peek(1, 2);
              auto the_result = at::hspmm(mat1, mat2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 639: { // numel
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::numel(self);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 640: { // unbind
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::unbind(self, dim);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 641: { // unbind
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::unbind(self);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 642: { // to_sparse
          int64_t sparse_dim = readAttribute<int64_t>("sparse_dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.to_sparse(sparse_dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 643: { // to_sparse
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.to_sparse();
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 644: { // to_mkldnn
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.to_mkldnn();
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 645: { // mkldnn_reorder_conv2d_weight
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::mkldnn_reorder_conv2d_weight(self, padding, stride, dilation, groups);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 646: { // mkldnn_reorder_conv2d_weight
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::mkldnn_reorder_conv2d_weight(self, padding, stride, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 647: { // mkldnn_reorder_conv2d_weight
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::mkldnn_reorder_conv2d_weight(self, padding, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 648: { // mkldnn_reorder_conv2d_weight
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::mkldnn_reorder_conv2d_weight(self, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 649: { // mkldnn_reorder_conv2d_weight
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::mkldnn_reorder_conv2d_weight(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 650: { // to_mkldnn_backward
      
          run_op = [=] {
              auto grad = peek(0, 2);
              auto input = peek(1, 2);
              auto the_result = at::to_mkldnn_backward(grad, input);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 651: { // dequantize
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::dequantize(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 652: { // q_zero_point
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::q_zero_point(self);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 653: { // q_per_channel_scales
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::q_per_channel_scales(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 654: { // q_per_channel_zero_points
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::q_per_channel_zero_points(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 655: { // q_per_channel_axis
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::q_per_channel_axis(self);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 656: { // int_repr
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::int_repr(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 657: { // _make_per_tensor_quantized_tensor
          double scale = readAttribute<float>("scale");
          int64_t zero_point = readAttribute<int64_t>("zero_point");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_make_per_tensor_quantized_tensor(self, scale, zero_point);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 658: { // _make_per_channel_quantized_tensor
          int64_t axis = readAttribute<int64_t>("axis");
          run_op = [=] {
              auto self = peek(0, 3);
              auto scale = peek(1, 3);
              auto zero_point = peek(2, 3);
              auto the_result = at::_make_per_channel_quantized_tensor(self, scale, zero_point, axis);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 659: { // fake_quantize_per_tensor_affine
          double scale = readAttribute<float>("scale");
          int64_t zero_point = readAttribute<int64_t>("zero_point");
          int64_t quant_min = readAttribute<int64_t>("quant_min");
          int64_t quant_max = readAttribute<int64_t>("quant_max");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::fake_quantize_per_tensor_affine(self, scale, zero_point, quant_min, quant_max);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 660: { // fake_quantize_per_tensor_affine_backward
          double scale = readAttribute<float>("scale");
          int64_t zero_point = readAttribute<int64_t>("zero_point");
          int64_t quant_min = readAttribute<int64_t>("quant_min");
          int64_t quant_max = readAttribute<int64_t>("quant_max");
          run_op = [=] {
              auto grad = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::fake_quantize_per_tensor_affine_backward(grad, self, scale, zero_point, quant_min, quant_max);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 661: { // meshgrid
      
          run_op = [=] {
              auto tensors = peekSlice(0, InputSize() - 0, InputSize());
              auto the_result = at::meshgrid(tensors);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 662: { // cartesian_prod
      
          run_op = [=] {
              auto tensors = peekSlice(0, InputSize() - 0, InputSize());
              auto the_result = at::cartesian_prod(tensors);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 663: { // combinations
          int64_t r = readAttribute<int64_t>("r");
          bool with_replacement = readAttribute<int64_t>("with_replacement");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::combinations(self, r, with_replacement);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 664: { // combinations
          int64_t r = readAttribute<int64_t>("r");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::combinations(self, r);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 665: { // combinations
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::combinations(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 666: { // item
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.item();
                if(OutputSize() > 0) {assignTo(Output(0),self.scalar_type(), the_result);}
              return true;
          };
      } break;
      case 667: { // _local_scalar_dense
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_local_scalar_dense(self);
                if(OutputSize() > 0) {assignTo(Output(0),self.scalar_type(), the_result);}
              return true;
          };
      } break;
      case 668: { // _thnn_fused_lstm_cell
      
          run_op = [=] {
              auto input_gates = peek(0, 5);
              auto hidden_gates = peek(1, 5);
              auto cx = peek(2, 5);
              auto input_bias = peek(3, 5);
              auto hidden_bias = peek(4, 5);
              auto the_result = at::_thnn_fused_lstm_cell(input_gates, hidden_gates, cx, input_bias, hidden_bias);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 669: { // _thnn_fused_lstm_cell
      
          run_op = [=] {
              auto input_gates = peek(0, 4);
              auto hidden_gates = peek(1, 4);
              auto cx = peek(2, 4);
              auto input_bias = peek(3, 4);
              auto the_result = at::_thnn_fused_lstm_cell(input_gates, hidden_gates, cx, input_bias);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 670: { // _thnn_fused_lstm_cell
      
          run_op = [=] {
              auto input_gates = peek(0, 3);
              auto hidden_gates = peek(1, 3);
              auto cx = peek(2, 3);
              auto the_result = at::_thnn_fused_lstm_cell(input_gates, hidden_gates, cx);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 671: { // _thnn_fused_lstm_cell_backward
          bool has_bias = readAttribute<int64_t>("has_bias");
          run_op = [=] {
              auto grad_hy = peek(0, 5);
              auto grad_cy = peek(1, 5);
              auto cx = peek(2, 5);
              auto cy = peek(3, 5);
              auto workspace = peek(4, 5);
              auto the_result = at::_thnn_fused_lstm_cell_backward(grad_hy, grad_cy, cx, cy, workspace, has_bias);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
                if(OutputSize() > 4) {assignTo(Output(4),std::get<4>(the_result));}
              return true;
          };
      } break;
      case 672: { // _thnn_fused_gru_cell
      
          run_op = [=] {
              auto input_gates = peek(0, 5);
              auto hidden_gates = peek(1, 5);
              auto hx = peek(2, 5);
              auto input_bias = peek(3, 5);
              auto hidden_bias = peek(4, 5);
              auto the_result = at::_thnn_fused_gru_cell(input_gates, hidden_gates, hx, input_bias, hidden_bias);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 673: { // _thnn_fused_gru_cell
      
          run_op = [=] {
              auto input_gates = peek(0, 4);
              auto hidden_gates = peek(1, 4);
              auto hx = peek(2, 4);
              auto input_bias = peek(3, 4);
              auto the_result = at::_thnn_fused_gru_cell(input_gates, hidden_gates, hx, input_bias);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 674: { // _thnn_fused_gru_cell
      
          run_op = [=] {
              auto input_gates = peek(0, 3);
              auto hidden_gates = peek(1, 3);
              auto hx = peek(2, 3);
              auto the_result = at::_thnn_fused_gru_cell(input_gates, hidden_gates, hx);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 675: { // _thnn_fused_gru_cell_backward
          bool has_bias = readAttribute<int64_t>("has_bias");
          run_op = [=] {
              auto grad_hy = peek(0, 2);
              auto workspace = peek(1, 2);
              auto the_result = at::_thnn_fused_gru_cell_backward(grad_hy, workspace, has_bias);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
                if(OutputSize() > 4) {assignTo(Output(4),std::get<4>(the_result));}
              return true;
          };
      } break;
      case 676: { // lstm
          bool has_biases = readAttribute<int64_t>("has_biases");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          bool batch_first = readAttribute<int64_t>("batch_first");
          run_op = [=] {
              auto input = peek(0, InputSize());
              auto hx = peekSlice(1, InputSize() - 1, InputSize());
              auto params = peekSlice(1, InputSize() - 1, InputSize());
              auto the_result = at::lstm(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 677: { // lstm
          bool has_biases = readAttribute<int64_t>("has_biases");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          run_op = [=] {
              auto data = peek(0, InputSize());
              auto batch_sizes = peek(1, InputSize());
              auto hx = peekSlice(2, InputSize() - 2, InputSize());
              auto params = peekSlice(2, InputSize() - 2, InputSize());
              auto the_result = at::lstm(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 678: { // gru
          bool has_biases = readAttribute<int64_t>("has_biases");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          bool batch_first = readAttribute<int64_t>("batch_first");
          run_op = [=] {
              auto input = peek(0, InputSize());
              auto hx = peek(1, InputSize());
              auto params = peekSlice(2, InputSize() - 2, InputSize());
              auto the_result = at::gru(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 679: { // gru
          bool has_biases = readAttribute<int64_t>("has_biases");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          run_op = [=] {
              auto data = peek(0, InputSize());
              auto batch_sizes = peek(1, InputSize());
              auto hx = peek(2, InputSize());
              auto params = peekSlice(3, InputSize() - 3, InputSize());
              auto the_result = at::gru(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 680: { // rnn_tanh
          bool has_biases = readAttribute<int64_t>("has_biases");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          bool batch_first = readAttribute<int64_t>("batch_first");
          run_op = [=] {
              auto input = peek(0, InputSize());
              auto hx = peek(1, InputSize());
              auto params = peekSlice(2, InputSize() - 2, InputSize());
              auto the_result = at::rnn_tanh(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 681: { // rnn_tanh
          bool has_biases = readAttribute<int64_t>("has_biases");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          run_op = [=] {
              auto data = peek(0, InputSize());
              auto batch_sizes = peek(1, InputSize());
              auto hx = peek(2, InputSize());
              auto params = peekSlice(3, InputSize() - 3, InputSize());
              auto the_result = at::rnn_tanh(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 682: { // rnn_relu
          bool has_biases = readAttribute<int64_t>("has_biases");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          bool batch_first = readAttribute<int64_t>("batch_first");
          run_op = [=] {
              auto input = peek(0, InputSize());
              auto hx = peek(1, InputSize());
              auto params = peekSlice(2, InputSize() - 2, InputSize());
              auto the_result = at::rnn_relu(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 683: { // rnn_relu
          bool has_biases = readAttribute<int64_t>("has_biases");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          run_op = [=] {
              auto data = peek(0, InputSize());
              auto batch_sizes = peek(1, InputSize());
              auto hx = peek(2, InputSize());
              auto params = peekSlice(3, InputSize() - 3, InputSize());
              auto the_result = at::rnn_relu(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 684: { // lstm_cell
      
          run_op = [=] {
              auto input = peek(0, InputSize());
              auto hx = peekSlice(1, InputSize() - 5, InputSize());
              auto w_ih = peek(1, 5);
              auto w_hh = peek(2, 5);
              auto b_ih = peek(3, 5);
              auto b_hh = peek(4, 5);
              auto the_result = at::lstm_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 685: { // gru_cell
      
          run_op = [=] {
              auto input = peek(0, 6);
              auto hx = peek(1, 6);
              auto w_ih = peek(2, 6);
              auto w_hh = peek(3, 6);
              auto b_ih = peek(4, 6);
              auto b_hh = peek(5, 6);
              auto the_result = at::gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 686: { // gru_cell
      
          run_op = [=] {
              auto input = peek(0, 5);
              auto hx = peek(1, 5);
              auto w_ih = peek(2, 5);
              auto w_hh = peek(3, 5);
              auto b_ih = peek(4, 5);
              auto the_result = at::gru_cell(input, hx, w_ih, w_hh, b_ih);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 687: { // gru_cell
      
          run_op = [=] {
              auto input = peek(0, 4);
              auto hx = peek(1, 4);
              auto w_ih = peek(2, 4);
              auto w_hh = peek(3, 4);
              auto the_result = at::gru_cell(input, hx, w_ih, w_hh);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 688: { // rnn_tanh_cell
      
          run_op = [=] {
              auto input = peek(0, 6);
              auto hx = peek(1, 6);
              auto w_ih = peek(2, 6);
              auto w_hh = peek(3, 6);
              auto b_ih = peek(4, 6);
              auto b_hh = peek(5, 6);
              auto the_result = at::rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 689: { // rnn_tanh_cell
      
          run_op = [=] {
              auto input = peek(0, 5);
              auto hx = peek(1, 5);
              auto w_ih = peek(2, 5);
              auto w_hh = peek(3, 5);
              auto b_ih = peek(4, 5);
              auto the_result = at::rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 690: { // rnn_tanh_cell
      
          run_op = [=] {
              auto input = peek(0, 4);
              auto hx = peek(1, 4);
              auto w_ih = peek(2, 4);
              auto w_hh = peek(3, 4);
              auto the_result = at::rnn_tanh_cell(input, hx, w_ih, w_hh);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 691: { // rnn_relu_cell
      
          run_op = [=] {
              auto input = peek(0, 6);
              auto hx = peek(1, 6);
              auto w_ih = peek(2, 6);
              auto w_hh = peek(3, 6);
              auto b_ih = peek(4, 6);
              auto b_hh = peek(5, 6);
              auto the_result = at::rnn_relu_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 692: { // rnn_relu_cell
      
          run_op = [=] {
              auto input = peek(0, 5);
              auto hx = peek(1, 5);
              auto w_ih = peek(2, 5);
              auto w_hh = peek(3, 5);
              auto b_ih = peek(4, 5);
              auto the_result = at::rnn_relu_cell(input, hx, w_ih, w_hh, b_ih);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 693: { // rnn_relu_cell
      
          run_op = [=] {
              auto input = peek(0, 4);
              auto hx = peek(1, 4);
              auto w_ih = peek(2, 4);
              auto w_hh = peek(3, 4);
              auto the_result = at::rnn_relu_cell(input, hx, w_ih, w_hh);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 694: { // quantized_lstm
          bool has_biases = readAttribute<int64_t>("has_biases");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          bool batch_first = readAttribute<int64_t>("batch_first");
          run_op = [=] {
              auto input = peek(0, InputSize());
              auto hx = peekSlice(1, InputSize() - 1, InputSize());
              auto params = peekSlice(1, InputSize() - 1, InputSize());
              auto the_result = at::quantized_lstm(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 695: { // quantized_gru
          bool has_biases = readAttribute<int64_t>("has_biases");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          bool batch_first = readAttribute<int64_t>("batch_first");
          run_op = [=] {
              auto input = peek(0, InputSize());
              auto hx = peek(1, InputSize());
              auto params = peekSlice(2, InputSize() - 2, InputSize());
              auto the_result = at::quantized_gru(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 696: { // quantized_gru
          bool has_biases = readAttribute<int64_t>("has_biases");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          run_op = [=] {
              auto data = peek(0, InputSize());
              auto batch_sizes = peek(1, InputSize());
              auto hx = peek(2, InputSize());
              auto params = peekSlice(3, InputSize() - 3, InputSize());
              auto the_result = at::quantized_gru(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 697: { // quantized_lstm_cell
          at::Scalar scale_ih = readScalarAttribute("scale_ih");
          at::Scalar scale_hh = readScalarAttribute("scale_hh");
          at::Scalar zero_point_ih = readScalarAttribute("zero_point_ih");
          at::Scalar zero_point_hh = readScalarAttribute("zero_point_hh");
          run_op = [=] {
              auto input = peek(0, InputSize());
              auto hx = peekSlice(1, InputSize() - 9, InputSize());
              auto w_ih = peek(1, 9);
              auto w_hh = peek(2, 9);
              auto b_ih = peek(3, 9);
              auto b_hh = peek(4, 9);
              auto packed_ih = peek(5, 9);
              auto packed_hh = peek(6, 9);
              auto col_offsets_ih = peek(7, 9);
              auto col_offsets_hh = peek(8, 9);
              auto the_result = at::quantized_lstm_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 698: { // quantized_gru_cell
          at::Scalar scale_ih = readScalarAttribute("scale_ih");
          at::Scalar scale_hh = readScalarAttribute("scale_hh");
          at::Scalar zero_point_ih = readScalarAttribute("zero_point_ih");
          at::Scalar zero_point_hh = readScalarAttribute("zero_point_hh");
          run_op = [=] {
              auto input = peek(0, 10);
              auto hx = peek(1, 10);
              auto w_ih = peek(2, 10);
              auto w_hh = peek(3, 10);
              auto b_ih = peek(4, 10);
              auto b_hh = peek(5, 10);
              auto packed_ih = peek(6, 10);
              auto packed_hh = peek(7, 10);
              auto col_offsets_ih = peek(8, 10);
              auto col_offsets_hh = peek(9, 10);
              auto the_result = at::quantized_gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 699: { // quantized_rnn_relu_cell
          at::Scalar scale_ih = readScalarAttribute("scale_ih");
          at::Scalar scale_hh = readScalarAttribute("scale_hh");
          at::Scalar zero_point_ih = readScalarAttribute("zero_point_ih");
          at::Scalar zero_point_hh = readScalarAttribute("zero_point_hh");
          run_op = [=] {
              auto input = peek(0, 10);
              auto hx = peek(1, 10);
              auto w_ih = peek(2, 10);
              auto w_hh = peek(3, 10);
              auto b_ih = peek(4, 10);
              auto b_hh = peek(5, 10);
              auto packed_ih = peek(6, 10);
              auto packed_hh = peek(7, 10);
              auto col_offsets_ih = peek(8, 10);
              auto col_offsets_hh = peek(9, 10);
              auto the_result = at::quantized_rnn_relu_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 700: { // quantized_rnn_tanh_cell
          at::Scalar scale_ih = readScalarAttribute("scale_ih");
          at::Scalar scale_hh = readScalarAttribute("scale_hh");
          at::Scalar zero_point_ih = readScalarAttribute("zero_point_ih");
          at::Scalar zero_point_hh = readScalarAttribute("zero_point_hh");
          run_op = [=] {
              auto input = peek(0, 10);
              auto hx = peek(1, 10);
              auto w_ih = peek(2, 10);
              auto w_hh = peek(3, 10);
              auto b_ih = peek(4, 10);
              auto b_hh = peek(5, 10);
              auto packed_ih = peek(6, 10);
              auto packed_hh = peek(7, 10);
              auto col_offsets_ih = peek(8, 10);
              auto col_offsets_hh = peek(9, 10);
              auto the_result = at::quantized_rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 701: { // _pack_padded_sequence
          bool batch_first = readAttribute<int64_t>("batch_first");
          run_op = [=] {
              auto input = peek(0, 2);
              auto lengths = peek(1, 2);
              auto the_result = at::_pack_padded_sequence(input, lengths, batch_first);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 702: { // _pack_padded_sequence_backward
          auto input_size = readIntArrayRef("input_size");
          bool batch_first = readAttribute<int64_t>("batch_first");
          run_op = [=] {
              auto grad = peek(0, 2);
              auto batch_sizes = peek(1, 2);
              auto the_result = at::_pack_padded_sequence_backward(grad, input_size, batch_sizes, batch_first);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 703: { // _pad_packed_sequence
          bool batch_first = readAttribute<int64_t>("batch_first");
          at::Scalar padding_value = readScalarAttribute("padding_value");
          int64_t total_length = readAttribute<int64_t>("total_length");
          run_op = [=] {
              auto data = peek(0, 2);
              auto batch_sizes = peek(1, 2);
              auto the_result = at::_pad_packed_sequence(data, batch_sizes, batch_first, padding_value, total_length);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 704: { // is_set_to
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto tensor = peek(1, 2);
              auto the_result = self.is_set_to(tensor);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 705: { // masked_fill
          at::Scalar value = readScalarAttribute("value");
          run_op = [=] {
              auto self = peek(0, 2);
              auto mask = peek(1, 2);
              auto the_result = at::masked_fill(self, mask, value);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 706: { // masked_fill
      
          run_op = [=] {
              auto self = peek(0, 3);
              auto mask = peek(1, 3);
              auto value = peek(2, 3);
              auto the_result = at::masked_fill(self, mask, value);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 707: { // masked_scatter
      
          run_op = [=] {
              auto self = peek(0, 3);
              auto mask = peek(1, 3);
              auto source = peek(2, 3);
              auto the_result = at::masked_scatter(self, mask, source);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 708: { // view
          auto size = readIntArrayRef("size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.view(size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 709: { // index_add
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 3);
              auto index = peek(1, 3);
              auto source = peek(2, 3);
              auto the_result = at::index_add(self, dim, index, source);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 710: { // index_fill
          int64_t dim = readAttribute<int64_t>("dim");
          at::Scalar value = readScalarAttribute("value");
          run_op = [=] {
              auto self = peek(0, 2);
              auto index = peek(1, 2);
              auto the_result = at::index_fill(self, dim, index, value);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 711: { // index_fill
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 3);
              auto index = peek(1, 3);
              auto value = peek(2, 3);
              auto the_result = at::index_fill(self, dim, index, value);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 712: { // scatter
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 3);
              auto index = peek(1, 3);
              auto src = peek(2, 3);
              auto the_result = at::scatter(self, dim, index, src);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 713: { // scatter
          int64_t dim = readAttribute<int64_t>("dim");
          at::Scalar value = readScalarAttribute("value");
          run_op = [=] {
              auto self = peek(0, 2);
              auto index = peek(1, 2);
              auto the_result = at::scatter(self, dim, index, value);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 714: { // scatter_add
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 3);
              auto index = peek(1, 3);
              auto src = peek(2, 3);
              auto the_result = at::scatter_add(self, dim, index, src);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 715: { // __and__
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::__and__(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 716: { // __and__
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::__and__(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 717: { // __or__
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::__or__(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 718: { // __or__
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::__or__(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 719: { // __xor__
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::__xor__(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 720: { // __xor__
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::__xor__(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 721: { // __lshift__
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::__lshift__(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 722: { // __lshift__
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::__lshift__(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 723: { // __rshift__
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::__rshift__(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 724: { // __rshift__
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::__rshift__(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 725: { // addbmm
          at::Scalar beta = readScalarAttribute("beta");
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              auto self = peek(0, 3);
              auto batch1 = peek(1, 3);
              auto batch2 = peek(2, 3);
              auto the_result = at::addbmm(self, batch1, batch2, beta, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 726: { // addbmm
          at::Scalar beta = readScalarAttribute("beta");
          run_op = [=] {
              auto self = peek(0, 3);
              auto batch1 = peek(1, 3);
              auto batch2 = peek(2, 3);
              auto the_result = at::addbmm(self, batch1, batch2, beta);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 727: { // addbmm
      
          run_op = [=] {
              auto self = peek(0, 3);
              auto batch1 = peek(1, 3);
              auto batch2 = peek(2, 3);
              auto the_result = at::addbmm(self, batch1, batch2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 728: { // diag
          int64_t diagonal = readAttribute<int64_t>("diagonal");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::diag(self, diagonal);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 729: { // diag
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::diag(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 730: { // cross
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::cross(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 731: { // triu
          int64_t diagonal = readAttribute<int64_t>("diagonal");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::triu(self, diagonal);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 732: { // triu
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::triu(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 733: { // tril
          int64_t diagonal = readAttribute<int64_t>("diagonal");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::tril(self, diagonal);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 734: { // tril
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::tril(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 735: { // trace
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::trace(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 736: { // ne
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::ne(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 737: { // ne
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::ne(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 738: { // eq
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::eq(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 739: { // eq
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::eq(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 740: { // ge
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::ge(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 741: { // ge
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::ge(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 742: { // le
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::le(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 743: { // le
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::le(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 744: { // gt
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::gt(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 745: { // gt
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::gt(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 746: { // lt
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::lt(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 747: { // lt
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::lt(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 748: { // take
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto index = peek(1, 2);
              auto the_result = at::take(self, index);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 749: { // index_select
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 2);
              auto index = peek(1, 2);
              auto the_result = at::index_select(self, dim, index);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 750: { // masked_select
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto mask = peek(1, 2);
              auto the_result = at::masked_select(self, mask);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 751: { // nonzero
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::nonzero(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 752: { // nonzero_numpy
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::nonzero_numpy(self);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 753: { // gather
          int64_t dim = readAttribute<int64_t>("dim");
          bool sparse_grad = readAttribute<int64_t>("sparse_grad");
          run_op = [=] {
              auto self = peek(0, 2);
              auto index = peek(1, 2);
              auto the_result = at::gather(self, dim, index, sparse_grad);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 754: { // gather
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 2);
              auto index = peek(1, 2);
              auto the_result = at::gather(self, dim, index);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 755: { // _gather_sparse_backward
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 3);
              auto index = peek(1, 3);
              auto grad = peek(2, 3);
              auto the_result = at::_gather_sparse_backward(self, dim, index, grad);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 756: { // addcmul
          at::Scalar value = readScalarAttribute("value");
          run_op = [=] {
              auto self = peek(0, 3);
              auto tensor1 = peek(1, 3);
              auto tensor2 = peek(2, 3);
              auto the_result = at::addcmul(self, tensor1, tensor2, value);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 757: { // addcmul
      
          run_op = [=] {
              auto self = peek(0, 3);
              auto tensor1 = peek(1, 3);
              auto tensor2 = peek(2, 3);
              auto the_result = at::addcmul(self, tensor1, tensor2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 758: { // addcdiv
          at::Scalar value = readScalarAttribute("value");
          run_op = [=] {
              auto self = peek(0, 3);
              auto tensor1 = peek(1, 3);
              auto tensor2 = peek(2, 3);
              auto the_result = at::addcdiv(self, tensor1, tensor2, value);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 759: { // addcdiv
      
          run_op = [=] {
              auto self = peek(0, 3);
              auto tensor1 = peek(1, 3);
              auto tensor2 = peek(2, 3);
              auto the_result = at::addcdiv(self, tensor1, tensor2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 760: { // lstsq
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto A = peek(1, 2);
              auto the_result = at::lstsq(self, A);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 761: { // triangular_solve
          bool upper = readAttribute<int64_t>("upper");
          bool transpose = readAttribute<int64_t>("transpose");
          bool unitriangular = readAttribute<int64_t>("unitriangular");
          run_op = [=] {
              auto self = peek(0, 2);
              auto A = peek(1, 2);
              auto the_result = at::triangular_solve(self, A, upper, transpose, unitriangular);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 762: { // triangular_solve
          bool upper = readAttribute<int64_t>("upper");
          bool transpose = readAttribute<int64_t>("transpose");
          run_op = [=] {
              auto self = peek(0, 2);
              auto A = peek(1, 2);
              auto the_result = at::triangular_solve(self, A, upper, transpose);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 763: { // triangular_solve
          bool upper = readAttribute<int64_t>("upper");
          run_op = [=] {
              auto self = peek(0, 2);
              auto A = peek(1, 2);
              auto the_result = at::triangular_solve(self, A, upper);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 764: { // triangular_solve
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto A = peek(1, 2);
              auto the_result = at::triangular_solve(self, A);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 765: { // _triangular_solve_helper
          bool upper = readAttribute<int64_t>("upper");
          bool transpose = readAttribute<int64_t>("transpose");
          bool unitriangular = readAttribute<int64_t>("unitriangular");
          run_op = [=] {
              auto self = peek(0, 2);
              auto A = peek(1, 2);
              auto the_result = at::_triangular_solve_helper(self, A, upper, transpose, unitriangular);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 766: { // symeig
          bool eigenvectors = readAttribute<int64_t>("eigenvectors");
          bool upper = readAttribute<int64_t>("upper");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::symeig(self, eigenvectors, upper);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 767: { // symeig
          bool eigenvectors = readAttribute<int64_t>("eigenvectors");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::symeig(self, eigenvectors);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 768: { // symeig
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::symeig(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 769: { // _symeig_helper
          bool eigenvectors = readAttribute<int64_t>("eigenvectors");
          bool upper = readAttribute<int64_t>("upper");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_symeig_helper(self, eigenvectors, upper);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 770: { // eig
          bool eigenvectors = readAttribute<int64_t>("eigenvectors");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::eig(self, eigenvectors);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 771: { // eig
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::eig(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 772: { // svd
          bool some = readAttribute<int64_t>("some");
          bool compute_uv = readAttribute<int64_t>("compute_uv");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::svd(self, some, compute_uv);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 773: { // svd
          bool some = readAttribute<int64_t>("some");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::svd(self, some);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 774: { // svd
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::svd(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 775: { // _svd_helper
          bool some = readAttribute<int64_t>("some");
          bool compute_uv = readAttribute<int64_t>("compute_uv");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_svd_helper(self, some, compute_uv);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 776: { // cholesky
          bool upper = readAttribute<int64_t>("upper");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::cholesky(self, upper);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 777: { // cholesky
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::cholesky(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 778: { // _cholesky_helper
          bool upper = readAttribute<int64_t>("upper");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_cholesky_helper(self, upper);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 779: { // cholesky_solve
          bool upper = readAttribute<int64_t>("upper");
          run_op = [=] {
              auto self = peek(0, 2);
              auto input2 = peek(1, 2);
              auto the_result = at::cholesky_solve(self, input2, upper);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 780: { // cholesky_solve
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto input2 = peek(1, 2);
              auto the_result = at::cholesky_solve(self, input2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 781: { // _cholesky_solve_helper
          bool upper = readAttribute<int64_t>("upper");
          run_op = [=] {
              auto self = peek(0, 2);
              auto A = peek(1, 2);
              auto the_result = at::_cholesky_solve_helper(self, A, upper);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 782: { // solve
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto A = peek(1, 2);
              auto the_result = at::solve(self, A);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 783: { // _solve_helper
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto A = peek(1, 2);
              auto the_result = at::_solve_helper(self, A);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 784: { // cholesky_inverse
          bool upper = readAttribute<int64_t>("upper");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::cholesky_inverse(self, upper);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 785: { // cholesky_inverse
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::cholesky_inverse(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 786: { // qr
          bool some = readAttribute<int64_t>("some");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::qr(self, some);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 787: { // qr
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::qr(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 788: { // _qr_helper
          bool some = readAttribute<int64_t>("some");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_qr_helper(self, some);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 789: { // geqrf
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::geqrf(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 790: { // orgqr
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto input2 = peek(1, 2);
              auto the_result = at::orgqr(self, input2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 791: { // ormqr
          bool left = readAttribute<int64_t>("left");
          bool transpose = readAttribute<int64_t>("transpose");
          run_op = [=] {
              auto self = peek(0, 3);
              auto input2 = peek(1, 3);
              auto input3 = peek(2, 3);
              auto the_result = at::ormqr(self, input2, input3, left, transpose);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 792: { // ormqr
          bool left = readAttribute<int64_t>("left");
          run_op = [=] {
              auto self = peek(0, 3);
              auto input2 = peek(1, 3);
              auto input3 = peek(2, 3);
              auto the_result = at::ormqr(self, input2, input3, left);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 793: { // ormqr
      
          run_op = [=] {
              auto self = peek(0, 3);
              auto input2 = peek(1, 3);
              auto input3 = peek(2, 3);
              auto the_result = at::ormqr(self, input2, input3);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 794: { // _lu_with_info
          bool pivot = readAttribute<int64_t>("pivot");
          bool check_errors = readAttribute<int64_t>("check_errors");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_lu_with_info(self, pivot, check_errors);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 795: { // _lu_with_info
          bool pivot = readAttribute<int64_t>("pivot");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_lu_with_info(self, pivot);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 796: { // _lu_with_info
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_lu_with_info(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 797: { // lu_solve
      
          run_op = [=] {
              auto self = peek(0, 3);
              auto LU_data = peek(1, 3);
              auto LU_pivots = peek(2, 3);
              auto the_result = at::lu_solve(self, LU_data, LU_pivots);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 798: { // _lu_solve_helper
      
          run_op = [=] {
              auto self = peek(0, 3);
              auto LU_data = peek(1, 3);
              auto LU_pivots = peek(2, 3);
              auto the_result = at::_lu_solve_helper(self, LU_data, LU_pivots);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 799: { // multinomial
          int64_t num_samples = readAttribute<int64_t>("num_samples");
          bool replacement = readAttribute<int64_t>("replacement");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::multinomial(self, num_samples, replacement);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 800: { // multinomial
          int64_t num_samples = readAttribute<int64_t>("num_samples");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::multinomial(self, num_samples);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 801: { // _multinomial_alias_setup
      
          run_op = [=] {
              auto probs = peek(0, 1);
              auto the_result = at::_multinomial_alias_setup(probs);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 802: { // _multinomial_alias_draw
          int64_t num_samples = readAttribute<int64_t>("num_samples");
          run_op = [=] {
              auto J = peek(0, 2);
              auto q = peek(1, 2);
              auto the_result = at::_multinomial_alias_draw(J, q, num_samples);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 803: { // lgamma
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::lgamma(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 804: { // digamma
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::digamma(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 805: { // polygamma
          int64_t n = readAttribute<int64_t>("n");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::polygamma(n, self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 806: { // erfinv
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::erfinv(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 807: { // sign
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::sign(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 808: { // dist
          at::Scalar p = readScalarAttribute("p");
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::dist(self, other, p);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 809: { // dist
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::dist(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 810: { // atan2
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::atan2(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 811: { // lerp
          at::Scalar weight = readScalarAttribute("weight");
          run_op = [=] {
              auto self = peek(0, 2);
              auto end = peek(1, 2);
              auto the_result = at::lerp(self, end, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 812: { // lerp
      
          run_op = [=] {
              auto self = peek(0, 3);
              auto end = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::lerp(self, end, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 813: { // histc
          int64_t bins = readAttribute<int64_t>("bins");
          at::Scalar min = readScalarAttribute("min");
          at::Scalar max = readScalarAttribute("max");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::histc(self, bins, min, max);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 814: { // histc
          int64_t bins = readAttribute<int64_t>("bins");
          at::Scalar min = readScalarAttribute("min");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::histc(self, bins, min);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 815: { // histc
          int64_t bins = readAttribute<int64_t>("bins");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::histc(self, bins);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 816: { // histc
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::histc(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 817: { // fmod
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::fmod(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 818: { // fmod
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::fmod(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 819: { // remainder
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::remainder(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 820: { // remainder
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::remainder(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 821: { // min
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::min(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 822: { // min
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::min(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 823: { // max
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::max(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 824: { // max
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 825: { // median
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::median(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 826: { // sort
          int64_t dim = readAttribute<int64_t>("dim");
          bool descending = readAttribute<int64_t>("descending");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::sort(self, dim, descending);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 827: { // sort
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::sort(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 828: { // sort
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::sort(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 829: { // argsort
          int64_t dim = readAttribute<int64_t>("dim");
          bool descending = readAttribute<int64_t>("descending");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::argsort(self, dim, descending);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 830: { // argsort
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::argsort(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 831: { // argsort
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::argsort(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 832: { // topk
          int64_t k = readAttribute<int64_t>("k");
          int64_t dim = readAttribute<int64_t>("dim");
          bool largest = readAttribute<int64_t>("largest");
          bool sorted = readAttribute<int64_t>("sorted");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::topk(self, k, dim, largest, sorted);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 833: { // topk
          int64_t k = readAttribute<int64_t>("k");
          int64_t dim = readAttribute<int64_t>("dim");
          bool largest = readAttribute<int64_t>("largest");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::topk(self, k, dim, largest);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 834: { // topk
          int64_t k = readAttribute<int64_t>("k");
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::topk(self, k, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 835: { // topk
          int64_t k = readAttribute<int64_t>("k");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::topk(self, k);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 836: { // all
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::all(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 837: { // any
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::any(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 838: { // renorm
          at::Scalar p = readScalarAttribute("p");
          int64_t dim = readAttribute<int64_t>("dim");
          at::Scalar maxnorm = readScalarAttribute("maxnorm");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::renorm(self, p, dim, maxnorm);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 839: { // unfold
          int64_t dimension = readAttribute<int64_t>("dimension");
          int64_t size = readAttribute<int64_t>("size");
          int64_t step = readAttribute<int64_t>("step");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = self.unfold(dimension, size, step);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 840: { // equal
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::equal(self, other);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 841: { // pow
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto exponent = peek(1, 2);
              auto the_result = at::pow(self, exponent);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 842: { // pow
          at::Scalar self = readScalarAttribute("self");
          run_op = [=] {
              auto exponent = peek(0, 1);
              auto the_result = at::pow(self, exponent);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 843: { // alias
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::alias(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 844: { // _addr
          at::Scalar beta = readScalarAttribute("beta");
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              auto self = peek(0, 3);
              auto vec1 = peek(1, 3);
              auto vec2 = peek(2, 3);
              auto the_result = at::_addr(self, vec1, vec2, beta, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 845: { // _addr
          at::Scalar beta = readScalarAttribute("beta");
          run_op = [=] {
              auto self = peek(0, 3);
              auto vec1 = peek(1, 3);
              auto vec2 = peek(2, 3);
              auto the_result = at::_addr(self, vec1, vec2, beta);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 846: { // _addr
      
          run_op = [=] {
              auto self = peek(0, 3);
              auto vec1 = peek(1, 3);
              auto vec2 = peek(2, 3);
              auto the_result = at::_addr(self, vec1, vec2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 847: { // _cumsum
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_cumsum(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 848: { // _cumprod
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_cumprod(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 849: { // _var
          bool unbiased = readAttribute<int64_t>("unbiased");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_var(self, unbiased);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 850: { // _var
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_var(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 851: { // _std
          bool unbiased = readAttribute<int64_t>("unbiased");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_std(self, unbiased);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 852: { // _std
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_std(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 853: { // _cat
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto tensors = peekSlice(0, InputSize() - 0, InputSize());
              auto the_result = at::_cat(tensors, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 854: { // _cat
      
          run_op = [=] {
              auto tensors = peekSlice(0, InputSize() - 0, InputSize());
              auto the_result = at::_cat(tensors);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 855: { // _mode
          int64_t dim = readAttribute<int64_t>("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_mode(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 856: { // _mode
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_mode(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 857: { // _mode
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_mode(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 858: { // _max
          int64_t dim = readAttribute<int64_t>("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_max(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 859: { // _max
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_max(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 860: { // _min
          int64_t dim = readAttribute<int64_t>("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_min(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 861: { // _min
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_min(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 862: { // binary_cross_entropy
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::binary_cross_entropy(self, target, weight, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 863: { // binary_cross_entropy
      
          run_op = [=] {
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::binary_cross_entropy(self, target, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 864: { // binary_cross_entropy
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::binary_cross_entropy(self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 865: { // binary_cross_entropy_backward
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto grad_output = peek(0, 4);
              auto self = peek(1, 4);
              auto target = peek(2, 4);
              auto weight = peek(3, 4);
              auto the_result = at::binary_cross_entropy_backward(grad_output, self, target, weight, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 866: { // binary_cross_entropy_backward
      
          run_op = [=] {
              auto grad_output = peek(0, 4);
              auto self = peek(1, 4);
              auto target = peek(2, 4);
              auto weight = peek(3, 4);
              auto the_result = at::binary_cross_entropy_backward(grad_output, self, target, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 867: { // binary_cross_entropy_backward
      
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::binary_cross_entropy_backward(grad_output, self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 868: { // mse_loss
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::mse_loss(self, target, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 869: { // mse_loss
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::mse_loss(self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 870: { // mse_loss_backward
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::mse_loss_backward(grad_output, self, target, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 871: { // l1_loss
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::l1_loss(self, target, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 872: { // l1_loss
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::l1_loss(self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 873: { // l1_loss_backward
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::l1_loss_backward(grad_output, self, target, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 874: { // multi_margin_loss
          at::Scalar p = readScalarAttribute("p");
          at::Scalar margin = readScalarAttribute("margin");
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::multi_margin_loss(self, target, p, margin, weight, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 875: { // multi_margin_loss
          at::Scalar p = readScalarAttribute("p");
          at::Scalar margin = readScalarAttribute("margin");
          run_op = [=] {
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::multi_margin_loss(self, target, p, margin, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 876: { // multi_margin_loss
          at::Scalar p = readScalarAttribute("p");
          at::Scalar margin = readScalarAttribute("margin");
          run_op = [=] {
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::multi_margin_loss(self, target, p, margin);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 877: { // multi_margin_loss
          at::Scalar p = readScalarAttribute("p");
          run_op = [=] {
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::multi_margin_loss(self, target, p);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 878: { // multi_margin_loss
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::multi_margin_loss(self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 879: { // multi_margin_loss_backward
          at::Scalar p = readScalarAttribute("p");
          at::Scalar margin = readScalarAttribute("margin");
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto grad_output = peek(0, 4);
              auto self = peek(1, 4);
              auto target = peek(2, 4);
              auto weight = peek(3, 4);
              auto the_result = at::multi_margin_loss_backward(grad_output, self, target, p, margin, weight, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 880: { // multi_margin_loss_backward
          at::Scalar p = readScalarAttribute("p");
          at::Scalar margin = readScalarAttribute("margin");
          run_op = [=] {
              auto grad_output = peek(0, 4);
              auto self = peek(1, 4);
              auto target = peek(2, 4);
              auto weight = peek(3, 4);
              auto the_result = at::multi_margin_loss_backward(grad_output, self, target, p, margin, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 881: { // multi_margin_loss_backward
          at::Scalar p = readScalarAttribute("p");
          at::Scalar margin = readScalarAttribute("margin");
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::multi_margin_loss_backward(grad_output, self, target, p, margin);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 882: { // multilabel_margin_loss
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::multilabel_margin_loss(self, target, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 883: { // multilabel_margin_loss
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::multilabel_margin_loss(self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 884: { // multilabel_margin_loss_forward
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::multilabel_margin_loss_forward(self, target, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 885: { // multilabel_margin_loss_backward
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto grad_output = peek(0, 4);
              auto self = peek(1, 4);
              auto target = peek(2, 4);
              auto is_target = peek(3, 4);
              auto the_result = at::multilabel_margin_loss_backward(grad_output, self, target, reduction, is_target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 886: { // nll_loss
          int64_t reduction = readAttribute<int64_t>("reduction");
          int64_t ignore_index = readAttribute<int64_t>("ignore_index");
          run_op = [=] {
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::nll_loss(self, target, weight, reduction, ignore_index);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 887: { // nll_loss
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::nll_loss(self, target, weight, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 888: { // nll_loss
      
          run_op = [=] {
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::nll_loss(self, target, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 889: { // nll_loss
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::nll_loss(self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 890: { // nll_loss_forward
          int64_t reduction = readAttribute<int64_t>("reduction");
          int64_t ignore_index = readAttribute<int64_t>("ignore_index");
          run_op = [=] {
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::nll_loss_forward(self, target, weight, reduction, ignore_index);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 891: { // nll_loss_backward
          int64_t reduction = readAttribute<int64_t>("reduction");
          int64_t ignore_index = readAttribute<int64_t>("ignore_index");
          run_op = [=] {
              auto grad_output = peek(0, 5);
              auto self = peek(1, 5);
              auto target = peek(2, 5);
              auto weight = peek(3, 5);
              auto total_weight = peek(4, 5);
              auto the_result = at::nll_loss_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 892: { // nll_loss2d
          int64_t reduction = readAttribute<int64_t>("reduction");
          int64_t ignore_index = readAttribute<int64_t>("ignore_index");
          run_op = [=] {
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::nll_loss2d(self, target, weight, reduction, ignore_index);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 893: { // nll_loss2d
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::nll_loss2d(self, target, weight, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 894: { // nll_loss2d
      
          run_op = [=] {
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::nll_loss2d(self, target, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 895: { // nll_loss2d
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::nll_loss2d(self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 896: { // nll_loss2d_forward
          int64_t reduction = readAttribute<int64_t>("reduction");
          int64_t ignore_index = readAttribute<int64_t>("ignore_index");
          run_op = [=] {
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::nll_loss2d_forward(self, target, weight, reduction, ignore_index);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 897: { // nll_loss2d_backward
          int64_t reduction = readAttribute<int64_t>("reduction");
          int64_t ignore_index = readAttribute<int64_t>("ignore_index");
          run_op = [=] {
              auto grad_output = peek(0, 5);
              auto self = peek(1, 5);
              auto target = peek(2, 5);
              auto weight = peek(3, 5);
              auto total_weight = peek(4, 5);
              auto the_result = at::nll_loss2d_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 898: { // smooth_l1_loss
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::smooth_l1_loss(self, target, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 899: { // smooth_l1_loss
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::smooth_l1_loss(self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 900: { // smooth_l1_loss_backward
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::smooth_l1_loss_backward(grad_output, self, target, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 901: { // soft_margin_loss
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::soft_margin_loss(self, target, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 902: { // soft_margin_loss
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::soft_margin_loss(self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 903: { // soft_margin_loss_backward
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::soft_margin_loss_backward(grad_output, self, target, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 904: { // elu
          at::Scalar alpha = readScalarAttribute("alpha");
          at::Scalar scale = readScalarAttribute("scale");
          at::Scalar input_scale = readScalarAttribute("input_scale");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::elu(self, alpha, scale, input_scale);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 905: { // elu
          at::Scalar alpha = readScalarAttribute("alpha");
          at::Scalar scale = readScalarAttribute("scale");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::elu(self, alpha, scale);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 906: { // elu
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::elu(self, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 907: { // elu
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::elu(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 908: { // elu_backward
          at::Scalar alpha = readScalarAttribute("alpha");
          at::Scalar scale = readScalarAttribute("scale");
          at::Scalar input_scale = readScalarAttribute("input_scale");
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto output = peek(1, 2);
              auto the_result = at::elu_backward(grad_output, alpha, scale, input_scale, output);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 909: { // glu
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::glu(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 910: { // glu
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::glu(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 911: { // glu_backward
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::glu_backward(grad_output, self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 912: { // hardtanh
          at::Scalar min_val = readScalarAttribute("min_val");
          at::Scalar max_val = readScalarAttribute("max_val");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::hardtanh(self, min_val, max_val);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 913: { // hardtanh
          at::Scalar min_val = readScalarAttribute("min_val");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::hardtanh(self, min_val);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 914: { // hardtanh
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::hardtanh(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 915: { // hardtanh_backward
          at::Scalar min_val = readScalarAttribute("min_val");
          at::Scalar max_val = readScalarAttribute("max_val");
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::hardtanh_backward(grad_output, self, min_val, max_val);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 916: { // leaky_relu
          at::Scalar negative_slope = readScalarAttribute("negative_slope");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::leaky_relu(self, negative_slope);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 917: { // leaky_relu
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::leaky_relu(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 918: { // leaky_relu_backward
          at::Scalar negative_slope = readScalarAttribute("negative_slope");
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::leaky_relu_backward(grad_output, self, negative_slope);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 919: { // log_sigmoid
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::log_sigmoid(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 920: { // log_sigmoid_forward
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::log_sigmoid_forward(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 921: { // log_sigmoid_backward
      
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto buffer = peek(2, 3);
              auto the_result = at::log_sigmoid_backward(grad_output, self, buffer);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 922: { // rrelu_with_noise
          at::Scalar lower = readScalarAttribute("lower");
          at::Scalar upper = readScalarAttribute("upper");
          bool training = readAttribute<int64_t>("training");
          run_op = [=] {
              auto self = peek(0, 2);
              auto noise = peek(1, 2);
              auto the_result = at::rrelu_with_noise(self, noise, lower, upper, training);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 923: { // rrelu_with_noise
          at::Scalar lower = readScalarAttribute("lower");
          at::Scalar upper = readScalarAttribute("upper");
          run_op = [=] {
              auto self = peek(0, 2);
              auto noise = peek(1, 2);
              auto the_result = at::rrelu_with_noise(self, noise, lower, upper);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 924: { // rrelu_with_noise
          at::Scalar lower = readScalarAttribute("lower");
          run_op = [=] {
              auto self = peek(0, 2);
              auto noise = peek(1, 2);
              auto the_result = at::rrelu_with_noise(self, noise, lower);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 925: { // rrelu_with_noise
      
          run_op = [=] {
              auto self = peek(0, 2);
              auto noise = peek(1, 2);
              auto the_result = at::rrelu_with_noise(self, noise);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 926: { // rrelu_with_noise_backward
          at::Scalar lower = readScalarAttribute("lower");
          at::Scalar upper = readScalarAttribute("upper");
          bool training = readAttribute<int64_t>("training");
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto noise = peek(2, 3);
              auto the_result = at::rrelu_with_noise_backward(grad_output, self, noise, lower, upper, training);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 927: { // softplus
          at::Scalar beta = readScalarAttribute("beta");
          at::Scalar threshold = readScalarAttribute("threshold");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::softplus(self, beta, threshold);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 928: { // softplus
          at::Scalar beta = readScalarAttribute("beta");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::softplus(self, beta);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 929: { // softplus
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::softplus(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 930: { // softplus_backward
          at::Scalar beta = readScalarAttribute("beta");
          at::Scalar threshold = readScalarAttribute("threshold");
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto output = peek(2, 3);
              auto the_result = at::softplus_backward(grad_output, self, beta, threshold, output);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 931: { // softshrink
          at::Scalar lambd = readScalarAttribute("lambd");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::softshrink(self, lambd);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 932: { // softshrink
      
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::softshrink(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 933: { // softshrink_backward
          at::Scalar lambd = readScalarAttribute("lambd");
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::softshrink_backward(grad_output, self, lambd);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 934: { // adaptive_avg_pool2d
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::adaptive_avg_pool2d(self, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 935: { // mkldnn_adaptive_avg_pool2d
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::mkldnn_adaptive_avg_pool2d(self, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 936: { // _adaptive_avg_pool2d
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::_adaptive_avg_pool2d(self, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 937: { // _adaptive_avg_pool2d_backward
      
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::_adaptive_avg_pool2d_backward(grad_output, self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 938: { // adaptive_avg_pool3d
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::adaptive_avg_pool3d(self, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 939: { // adaptive_avg_pool3d_backward
      
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::adaptive_avg_pool3d_backward(grad_output, self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 940: { // adaptive_max_pool2d
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::adaptive_max_pool2d(self, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 941: { // adaptive_max_pool2d_backward
      
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto indices = peek(2, 3);
              auto the_result = at::adaptive_max_pool2d_backward(grad_output, self, indices);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 942: { // adaptive_max_pool3d
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::adaptive_max_pool3d(self, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 943: { // adaptive_max_pool3d_backward
      
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto indices = peek(2, 3);
              auto the_result = at::adaptive_max_pool3d_backward(grad_output, self, indices);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 944: { // avg_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          bool count_include_pad = readAttribute<int64_t>("count_include_pad");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 945: { // avg_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::avg_pool2d(self, kernel_size, stride, padding, ceil_mode);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 946: { // avg_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::avg_pool2d(self, kernel_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 947: { // avg_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::avg_pool2d(self, kernel_size, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 948: { // avg_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::avg_pool2d(self, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 949: { // avg_pool3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          bool count_include_pad = readAttribute<int64_t>("count_include_pad");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::avg_pool3d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 950: { // avg_pool3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::avg_pool3d(self, kernel_size, stride, padding, ceil_mode);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 951: { // avg_pool3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::avg_pool3d(self, kernel_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 952: { // avg_pool3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::avg_pool3d(self, kernel_size, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 953: { // avg_pool3d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::avg_pool3d(self, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 954: { // fractional_max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              auto self = peek(0, 2);
              auto random_samples = peek(1, 2);
              auto the_result = at::fractional_max_pool2d(self, kernel_size, output_size, random_samples);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 955: { // fractional_max_pool2d_backward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto indices = peek(2, 3);
              auto the_result = at::fractional_max_pool2d_backward(grad_output, self, kernel_size, output_size, indices);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 956: { // fractional_max_pool3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              auto self = peek(0, 2);
              auto random_samples = peek(1, 2);
              auto the_result = at::fractional_max_pool3d(self, kernel_size, output_size, random_samples);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 957: { // fractional_max_pool3d_backward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto indices = peek(2, 3);
              auto the_result = at::fractional_max_pool3d_backward(grad_output, self, kernel_size, output_size, indices);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 958: { // max_pool2d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 959: { // max_pool2d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 960: { // max_pool2d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool2d_with_indices(self, kernel_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 961: { // max_pool2d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool2d_with_indices(self, kernel_size, stride);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 962: { // max_pool2d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool2d_with_indices(self, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 963: { // max_pool2d_with_indices_backward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto indices = peek(2, 3);
              auto the_result = at::max_pool2d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 964: { // max_pool3d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool3d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 965: { // max_pool3d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool3d_with_indices(self, kernel_size, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 966: { // max_pool3d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool3d_with_indices(self, kernel_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 967: { // max_pool3d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool3d_with_indices(self, kernel_size, stride);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 968: { // max_pool3d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::max_pool3d_with_indices(self, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 969: { // max_pool3d_with_indices_backward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto indices = peek(2, 3);
              auto the_result = at::max_pool3d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 970: { // max_unpool2d
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              auto self = peek(0, 2);
              auto indices = peek(1, 2);
              auto the_result = at::max_unpool2d(self, indices, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 971: { // max_unpool2d_backward
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto indices = peek(2, 3);
              auto the_result = at::max_unpool2d_backward(grad_output, self, indices, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 972: { // max_unpool3d
          auto output_size = readIntArrayRef("output_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 2);
              auto indices = peek(1, 2);
              auto the_result = at::max_unpool3d(self, indices, output_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 973: { // max_unpool3d_backward
          auto output_size = readIntArrayRef("output_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto indices = peek(2, 3);
              auto the_result = at::max_unpool3d_backward(grad_output, self, indices, output_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 974: { // reflection_pad1d
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::reflection_pad1d(self, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 975: { // reflection_pad1d_backward
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::reflection_pad1d_backward(grad_output, self, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 976: { // reflection_pad2d
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::reflection_pad2d(self, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 977: { // reflection_pad2d_backward
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::reflection_pad2d_backward(grad_output, self, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 978: { // replication_pad1d
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::replication_pad1d(self, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 979: { // replication_pad1d_backward
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::replication_pad1d_backward(grad_output, self, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 980: { // replication_pad2d
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::replication_pad2d(self, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 981: { // replication_pad2d_backward
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::replication_pad2d_backward(grad_output, self, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 982: { // replication_pad3d
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::replication_pad3d(self, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 983: { // replication_pad3d_backward
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::replication_pad3d_backward(grad_output, self, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 984: { // upsample_linear1d
          auto output_size = readIntArrayRef("output_size");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::upsample_linear1d(self, output_size, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 985: { // upsample_linear1d_backward
          auto output_size = readIntArrayRef("output_size");
          auto input_size = readIntArrayRef("input_size");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              auto grad_output = peek(0, 1);
              auto the_result = at::upsample_linear1d_backward(grad_output, output_size, input_size, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 986: { // upsample_bilinear2d
          auto output_size = readIntArrayRef("output_size");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::upsample_bilinear2d(self, output_size, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 987: { // upsample_bilinear2d_backward
          auto output_size = readIntArrayRef("output_size");
          auto input_size = readIntArrayRef("input_size");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              auto grad_output = peek(0, 1);
              auto the_result = at::upsample_bilinear2d_backward(grad_output, output_size, input_size, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 988: { // upsample_bicubic2d
          auto output_size = readIntArrayRef("output_size");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::upsample_bicubic2d(self, output_size, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 989: { // upsample_bicubic2d_backward
          auto output_size = readIntArrayRef("output_size");
          auto input_size = readIntArrayRef("input_size");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              auto grad_output = peek(0, 1);
              auto the_result = at::upsample_bicubic2d_backward(grad_output, output_size, input_size, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 990: { // upsample_trilinear3d
          auto output_size = readIntArrayRef("output_size");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::upsample_trilinear3d(self, output_size, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 991: { // upsample_trilinear3d_backward
          auto output_size = readIntArrayRef("output_size");
          auto input_size = readIntArrayRef("input_size");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              auto grad_output = peek(0, 1);
              auto the_result = at::upsample_trilinear3d_backward(grad_output, output_size, input_size, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 992: { // upsample_nearest1d
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::upsample_nearest1d(self, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 993: { // upsample_nearest1d_backward
          auto output_size = readIntArrayRef("output_size");
          auto input_size = readIntArrayRef("input_size");
          run_op = [=] {
              auto grad_output = peek(0, 1);
              auto the_result = at::upsample_nearest1d_backward(grad_output, output_size, input_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 994: { // upsample_nearest2d
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::upsample_nearest2d(self, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 995: { // upsample_nearest2d_backward
          auto output_size = readIntArrayRef("output_size");
          auto input_size = readIntArrayRef("input_size");
          run_op = [=] {
              auto grad_output = peek(0, 1);
              auto the_result = at::upsample_nearest2d_backward(grad_output, output_size, input_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 996: { // upsample_nearest3d
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::upsample_nearest3d(self, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 997: { // upsample_nearest3d_backward
          auto output_size = readIntArrayRef("output_size");
          auto input_size = readIntArrayRef("input_size");
          run_op = [=] {
              auto grad_output = peek(0, 1);
              auto the_result = at::upsample_nearest3d_backward(grad_output, output_size, input_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 998: { // sigmoid_backward
      
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto output = peek(1, 2);
              auto the_result = at::sigmoid_backward(grad_output, output);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 999: { // tanh_backward
      
          run_op = [=] {
              auto grad_output = peek(0, 2);
              auto output = peek(1, 2);
              auto the_result = at::tanh_backward(grad_output, output);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1000: { // slow_conv_transpose2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_transpose2d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1001: { // slow_conv_transpose2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_transpose2d(self, weight, kernel_size, bias, stride, padding, output_padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1002: { // slow_conv_transpose2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_transpose2d(self, weight, kernel_size, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1003: { // slow_conv_transpose2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_transpose2d(self, weight, kernel_size, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1004: { // slow_conv_transpose2d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_transpose2d(self, weight, kernel_size, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1005: { // slow_conv_transpose2d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::slow_conv_transpose2d(self, weight, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1006: { // slow_conv_transpose2d_backward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          auto dilation = readIntArrayRef("dilation");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              auto grad_output = peek(0, 5);
              auto self = peek(1, 5);
              auto weight = peek(2, 5);
              auto columns = peek(3, 5);
              auto ones = peek(4, 5);
              auto the_result = at::slow_conv_transpose2d_backward(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 1007: { // slow_conv_transpose3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_transpose3d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1008: { // slow_conv_transpose3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_transpose3d(self, weight, kernel_size, bias, stride, padding, output_padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1009: { // slow_conv_transpose3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_transpose3d(self, weight, kernel_size, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1010: { // slow_conv_transpose3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_transpose3d(self, weight, kernel_size, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1011: { // slow_conv_transpose3d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_transpose3d(self, weight, kernel_size, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1012: { // slow_conv_transpose3d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::slow_conv_transpose3d(self, weight, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1013: { // slow_conv_transpose3d_backward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          auto dilation = readIntArrayRef("dilation");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              auto grad_output = peek(0, 5);
              auto self = peek(1, 5);
              auto weight = peek(2, 5);
              auto finput = peek(3, 5);
              auto fgrad_input = peek(4, 5);
              auto the_result = at::slow_conv_transpose3d_backward(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 1014: { // thnn_conv2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::thnn_conv2d(self, weight, kernel_size, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1015: { // thnn_conv2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::thnn_conv2d(self, weight, kernel_size, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1016: { // thnn_conv2d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::thnn_conv2d(self, weight, kernel_size, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1017: { // thnn_conv2d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::thnn_conv2d(self, weight, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1018: { // thnn_conv2d_forward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::thnn_conv2d_forward(self, weight, kernel_size, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 1019: { // thnn_conv2d_backward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              auto grad_output = peek(0, 5);
              auto self = peek(1, 5);
              auto weight = peek(2, 5);
              auto finput = peek(3, 5);
              auto fgrad_input = peek(4, 5);
              auto the_result = at::thnn_conv2d_backward(grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 1020: { // thnn_conv_depthwise2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::thnn_conv_depthwise2d(self, weight, kernel_size, bias, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1021: { // thnn_conv_depthwise2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::thnn_conv_depthwise2d(self, weight, kernel_size, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1022: { // thnn_conv_depthwise2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::thnn_conv_depthwise2d(self, weight, kernel_size, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1023: { // thnn_conv_depthwise2d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::thnn_conv_depthwise2d(self, weight, kernel_size, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1024: { // thnn_conv_depthwise2d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::thnn_conv_depthwise2d(self, weight, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1025: { // thnn_conv_depthwise2d_forward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::thnn_conv_depthwise2d_forward(self, weight, kernel_size, bias, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1026: { // thnn_conv_depthwise2d_backward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          auto output_mask = readBoolMask<2>("output_mask");
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::thnn_conv_depthwise2d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 1027: { // thnn_conv3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::thnn_conv3d(self, weight, kernel_size, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1028: { // thnn_conv3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::thnn_conv3d(self, weight, kernel_size, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1029: { // thnn_conv3d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::thnn_conv3d(self, weight, kernel_size, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1030: { // thnn_conv3d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::thnn_conv3d(self, weight, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1031: { // thnn_conv3d_forward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::thnn_conv3d_forward(self, weight, kernel_size, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 1032: { // thnn_conv3d_backward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              auto grad_output = peek(0, 5);
              auto self = peek(1, 5);
              auto weight = peek(2, 5);
              auto finput = peek(3, 5);
              auto fgrad_input = peek(4, 5);
              auto the_result = at::thnn_conv3d_backward(grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 1033: { // slow_conv_dilated2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_dilated2d(self, weight, kernel_size, bias, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1034: { // slow_conv_dilated2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_dilated2d(self, weight, kernel_size, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1035: { // slow_conv_dilated2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_dilated2d(self, weight, kernel_size, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1036: { // slow_conv_dilated2d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_dilated2d(self, weight, kernel_size, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1037: { // slow_conv_dilated2d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::slow_conv_dilated2d(self, weight, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1038: { // slow_conv_dilated2d_backward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::slow_conv_dilated2d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 1039: { // slow_conv_dilated3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_dilated3d(self, weight, kernel_size, bias, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1040: { // slow_conv_dilated3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_dilated3d(self, weight, kernel_size, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1041: { // slow_conv_dilated3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_dilated3d(self, weight, kernel_size, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1042: { // slow_conv_dilated3d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_dilated3d(self, weight, kernel_size, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1043: { // slow_conv_dilated3d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              auto self = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::slow_conv_dilated3d(self, weight, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1044: { // slow_conv_dilated3d_backward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::slow_conv_dilated3d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 1045: { // col2im
          auto output_size = readIntArrayRef("output_size");
          auto kernel_size = readIntArrayRef("kernel_size");
          auto dilation = readIntArrayRef("dilation");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::col2im(self, output_size, kernel_size, dilation, padding, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1046: { // col2im_backward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto dilation = readIntArrayRef("dilation");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto grad_output = peek(0, 1);
              auto the_result = at::col2im_backward(grad_output, kernel_size, dilation, padding, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1047: { // im2col
          auto kernel_size = readIntArrayRef("kernel_size");
          auto dilation = readIntArrayRef("dilation");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto self = peek(0, 1);
              auto the_result = at::im2col(self, kernel_size, dilation, padding, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1048: { // im2col_backward
          auto input_size = readIntArrayRef("input_size");
          auto kernel_size = readIntArrayRef("kernel_size");
          auto dilation = readIntArrayRef("dilation");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              auto grad_output = peek(0, 1);
              auto the_result = at::im2col_backward(grad_output, input_size, kernel_size, dilation, padding, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      default:
        CAFFE_THROW("Unexpected key value for aten operator");
    }
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    return run_op();
  }
private:
  // actual operator implementation is initialized in ctor.
  std::function<bool()> run_op;
  at::Backend backend() const;

  TypeMeta typeMetaFor(const at::Tensor & t) {
    return typeMetaFor(t.scalar_type());
  }
  TypeMeta typeMetaFor(at::ScalarType st) {
    #define DEFINE_CASE(ctype,aten_name) \
      case at::k##aten_name: \
        return TypeMeta::Make<ctype>();
    switch(st) {
      AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, DEFINE_CASE)
    default:
      CAFFE_THROW("Unknown ATen Type");
    }
    #undef DEFINE_CASE
  }

  at::TensorOptions optionsFor(const Tensor& ten) {
    at::Device device = ten.GetDevice();
#ifdef __HIP_PLATFORM_HCC__
    if (backend() == at::Backend::HIP) {
      device = at::Device(kCUDA, device.index());
    }
#endif
    return at::TensorOptions(device).dtype(ten.dtype());
  }

  at::Tensor tensorWrapping(const Tensor& ten_) {
    auto& ten = const_cast<Tensor&>(ten_);
    return at::from_blob(
        ten.raw_mutable_data(),
        ten.sizes(),
        optionsFor(ten));
  }

  at::Tensor peek(size_t i, size_t N) {
    auto real_idx = InputSize() - N + i;
    return tensorWrapping(Input(real_idx));
  }

  std::vector<at::Tensor> peekSlice(size_t i, size_t len, size_t N) {
    std::vector<at::Tensor> results;
    for (size_t ii = i; ii < i + len; ++ii) {
      results.push_back(peek(ii, N));
    }
    return results;
  }

  void assignTo(Tensor* dst, const at::Tensor& src_) {
    at::Tensor src = src_.contiguous();
    auto at_sizes = src.sizes();
    caffe2::TypeMeta type_meta = typeMetaFor(src);
    at::Device device = src.device();
#ifdef __HIP_PLATFORM_HCC__
    if (device.type() == at::DeviceType::CUDA) {
      device = at::Device(at::DeviceType::HIP, device.index());
    }
#endif
    at::TensorImpl* src_impl = src.unsafeReleaseTensorImpl();
    std::vector<int64_t> dims(at_sizes.begin(), at_sizes.end());
    dst->Resize(dims);
    dst->ShareExternalPointer(
        at::DataPtr(
            src_impl->data(),
            static_cast<void*>(src_impl),
            [](void* t_ptr) -> void {
              at::TensorImpl* local_impl = static_cast<at::TensorImpl*>(t_ptr);
              c10::raw::intrusive_ptr::decref(local_impl);
            },
            device),
        type_meta,
        0);
  }
  void assignListStartingAt(
      size_t offset,
      const std::vector<at::Tensor>& tensors) {
    for (size_t i = 0; i < tensors.size(); i++) {
      assignTo(Output(offset + i), tensors[i]);
    }
  }

  template<typename T,
          typename std::enable_if<std::numeric_limits<T>::is_integer, bool>::type* =
              nullptr>
  int64_t extract(const at::Scalar &s) {
    return s.toLong();
  }

  template<typename T,
          typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type* =
              nullptr>
  int64_t extract(const at::Scalar &s) {
    return s.toDouble();
  }

  void assignTo(Tensor* dst, at::ScalarType scalar_type, at::Scalar scalar) {
    switch(scalar_type) {
      #define DEFINE_CASE(ctype,aten_name) \
        case at::k##aten_name: { \
          auto value = extract<ctype>(scalar); \
          assignToValue<ctype>(dst, at::convert<ctype,decltype(value)>(value)); \
        } break;
      AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, DEFINE_CASE)
#undef DEFINE_CASE
      default:
        CAFFE_THROW("Unknown ATen Type");
    }
  }
  template <typename T>
  void assignToValue(Tensor* dst, T v) {
    dst->Resize(std::vector<int64_t>());
    math::Set(1, v, dst->template mutable_data<T>(), &context_);
  }
  int findImplementation(const OperatorDef& operator_def) {
    CAFFE_ENFORCE(HasArgument("operator"));
    std::string op = OperatorBase::GetSingleArgument<std::string>("operator", "");
    // construct descriptor string ([DESCRIPTORS]) given the attributes
    // and inputs of this operator_def, and look up the implementation key
    // for this variant
    std::stringstream descriptor;
    descriptor << op;
    std::vector<std::string> attrs;
    for(size_t i = 0; i < operator_def.arg_size(); i++) {
      auto & attr = operator_def.arg(i);
      if(attr.name() == "operator" || attr.name() == "type" )
        continue;
      attrs.push_back(attr.name());
    }
    std::sort(attrs.begin(), attrs.end());
    for(auto & a : attrs)
      descriptor << "-" << a;

    std::string descriptor_sized =
        descriptor.str() + "-" + c10::to_string(InputSize());
    std::string descriptor_var_args = descriptor.str() + "-*";
    if (op_to_key.count(descriptor_sized) > 0) {
      return op_to_key[descriptor_sized];
    }
    if (op_to_key.count(descriptor_var_args) > 0) {
      return op_to_key[descriptor_var_args];
    }
    std::stringstream ss;
    ss << "Attempting to run unknown ATen operator configuration: "
       << descriptor_sized;
    CAFFE_THROW(ss.str());
  }
  at::Scalar readScalarAttribute(const std::string & name) {
    if(OperatorBase::HasSingleArgumentOfType<int64_t>(name)) {
      return OperatorBase::GetSingleArgument<int64_t>(name, 0);
    } else {
      CAFFE_ENFORCE(OperatorBase::HasSingleArgumentOfType<float>(name));
      return OperatorBase::GetSingleArgument<float>(name, 0);
    }
  }
  template<typename T>
  T readAttribute(const std::string & name) {
    CAFFE_ENFORCE(OperatorBase::HasSingleArgumentOfType<T>(name));
    return OperatorBase::GetSingleArgument<T>(name, 0);
  }
  std::vector<int64_t> readIntArrayRef(const std::string & name) {
    CAFFE_ENFORCE(OperatorBase::HasArgument(name));
    return OperatorBase::GetRepeatedArgument<int64_t>(name, {});
  }
  template <int N>
  std::array<bool, N> readBoolMask(const std::string& name) {
    CAFFE_ENFORCE(OperatorBase::HasArgument(name));
    std::vector<int64_t> ints =
        OperatorBase::GetRepeatedArgument<int64_t>(name, {});
    std::array<bool, N> result;
    for (size_t i = 0; i < N; ++i) {
      result[i] = ints.at(i);
    }
    return result;
  }
};

}
