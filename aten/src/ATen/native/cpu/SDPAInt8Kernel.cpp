#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/Utils.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/transformers/SDPAInt8.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
// #include <ATen/ops/_scaled_dot_product_int8.h>
// #include <ATen/ops/_scaled_dot_product_int8_native.h>
// #include <ATen/ops/clamp_max.h>
// #include <ATen/ops/clamp_min.h>
// #include <ATen/ops/round.h>
// #include <ATen/ops/softmax.h>
// #include <ATen/ops/linear_native.h>
// #include <ATen/ops/matmul.h>
// #include <ATen/ops/matmul_native.h>
// #include <ATen/ops/all.h>
#endif

namespace at::native {

namespace {

template <typename T>
struct is_reduced_floating_point:
    std::integral_constant<bool,
      std::is_same_v<T, at::Half> ||
      std::is_same_v<T, at::BFloat16>> {
};

template <typename T>
constexpr bool is_reduced_floating_point_v = is_reduced_floating_point<T>::value;

// inline double calculate_scale(
//     const at::Tensor& query,
//     double scale) {
//   return scale == 0.0 ? 1.0 / std::sqrt(query.size(-1)) : scale;
// }

// #ifdef CPU_CAPABILITY_AVX512

template <typename scalar_t>
inline void fill_stub(scalar_t* data, scalar_t val, int64_t size) {
  using Vec = Vectorized<scalar_t>;
  Vec data_vec = Vec(val);
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    data_vec.store(data + d);
  }
  #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
  # pragma unroll
  #endif
  for (; d < size; d++) {
    data[d] = val;
  }
}

void reshape_attn_mask_to_4d(
    at::Tensor& attn_mask,
    int64_t batchSize,
    int64_t num_head,
    int64_t qSize,
    int64_t kvSize) {
  // Support mask shapes:
  // 2d: ({Q_seq_len, 1}  x {KV_seq_len, 1})
  // 4d: ({Batch, 1} x {Num_heads, 1} x {Q_seq_len, 1}  x {KV_seq_len, 1})
  // Guaranteed in check_attn_mask_shape
  int64_t attn_mask_size_0 = 1;
  int64_t attn_mask_size_1 = 1;
  if (attn_mask.dim() == 4) {
    if (attn_mask.size(0) == batchSize) {
      attn_mask_size_0 = batchSize;
    }
    if (attn_mask.size(1) == num_head) {
      attn_mask_size_1 = num_head;
    }
  }
  attn_mask = attn_mask
                .view({attn_mask_size_0, attn_mask_size_1, attn_mask.size(-2), attn_mask.size(-1)})
                .expand({attn_mask_size_0, attn_mask_size_1, qSize, kvSize});
}

// TODO: Use at::native::_store instead when it supports Half.
template <typename scalar_t>
inline void _store(scalar_t* dst, at::vec::Vectorized<scalar_t> src, int size=at::vec::Vectorized<scalar_t>::size()) {
  src.store(dst, size);
}

template <typename scalar_t>
inline typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, void>
_store(scalar_t* dst, at::vec::Vectorized<float> src) {
  auto res = at::vec::convert_from_float<scalar_t>(src, src);
  res.store(dst, at::vec::Vectorized<float>::size());
}

template <typename scalar_t>
inline typename std::enable_if_t<std::is_same_v<scalar_t, unsigned char> || std::is_same_v<scalar_t, signed char>, void>
_store(scalar_t* dst, at::vec::Vectorized<float> src) {
  auto res = at::vec::convert<scalar_t>(src);
  res.store(dst, at::vec::Vectorized<float>::size());
}

/*
1. dequant
2. add mask
3. max reduce for softmax
*/
template <typename mask_t>
inline void _dequant_mask_max_fusion_kernel(
    const int32_t* in,
    const mask_t* mask_ptr,
    const int32_t* sum_a_ptr,
    const int32_t* sum_b_ptr,
    const int& M,
    const int& N,
    const int& ldi,
    const int& ldm, // leading dimension mask
    const int& ldo,
    const int32_t& beta, // zp_a*zp_b*k
    const float& alpha, // scale_a*scale_b*scale_sdpa
    float* out,
    float* sfm_max_ptr) {
  const int32_t vec_size = at::vec::Vectorized<float>::size();
  auto vec_beta = at::vec::Vectorized<int32_t>(beta);
  auto vec_alpha = at::vec::Vectorized<float>(alpha);
  for (long row = 0; row < M; row += 1) {
    auto sum_a = sum_a_ptr[row];
    auto vec_sum_a = at::vec::Vectorized<int32_t>(sum_a);
    const int32_t* tmp_in = in + row * ldi;
    float* tmp_out = out + row * ldo;
    const mask_t* mask_data_ptr = mask_ptr + row * ldm;
    float tmp_max = -std::numeric_limits<float>::infinity();
    auto vec_tmp_max = at::vec::Vectorized<float>(tmp_max);
    for (long col = 0; col < vec_size * (N / vec_size); col += vec_size) {
      auto vec_sum_b = at::vec::Vectorized<int32_t>::loadu(sum_b_ptr + col);
      auto tmp0 = at::vec::Vectorized<int32_t>::loadu(tmp_in + col);
      auto tmp1 = tmp0 - vec_sum_b;
      auto tmp2 = tmp1 - vec_sum_a;
      auto tmp3 = tmp2 + vec_beta;
      auto tmp4 = at::vec::convert<float>(tmp3);
      auto tmp5 = tmp4 * vec_alpha;
      auto tmp6 = at::vec::Vectorized<mask_t>::loadu(mask_data_ptr + col);
      auto tmp7 = at::vec::convert<float>(tmp6);
      auto tmp8 = tmp5 + tmp7;
      vec_tmp_max = at::vec::clamp_min(vec_tmp_max, tmp8);
      _store(tmp_out + col, tmp8);
    }
    tmp_max = std::max(tmp_max, vec_tmp_max.reduce_max());
    for (long col = vec_size * (N / vec_size); col < N; col++) {
      auto sum_b = sum_b_ptr[col];
      auto tmp0 = tmp_in[col];
      auto tmp1 = tmp0 - sum_b;
      auto tmp2 = tmp1 - sum_a;
      auto tmp3 = tmp2 + beta;
      auto tmp4 = (float) tmp3;
      auto tmp5 = tmp4 * alpha;
      auto tmp6 = mask_data_ptr[col];
      auto tmp7 = (float) tmp6;
      auto tmp8 = tmp5 + tmp7;
      tmp_max = std::max(tmp_max, tmp8);
      tmp_out[col] = tmp8;
    }
    sfm_max_ptr[row] = std::max(sfm_max_ptr[row], tmp_max);
  }
}

/*
1. dequant
2. max reduce for softmax
*/
inline void _dequant_max_fusion_kernel(
    const int32_t* in,
    const int32_t* sum_a_ptr,
    const int32_t* sum_b_ptr,
    const int& M,
    const int& N,
    const int& ldi,
    const int& ldo,
    const int32_t& beta, // zp_a*zp_b*k
    const float& alpha, // scale_a*scale_b*scale_sdpa
    float* out,
    float* sfm_max_ptr) {
  const int32_t vec_size = at::vec::Vectorized<float>::size();
  auto vec_beta = at::vec::Vectorized<int32_t>(beta);
  auto vec_alpha = at::vec::Vectorized<float>(alpha);
  for (long row = 0; row < M; row += 1) {
    auto sum_a = sum_a_ptr[row];
    auto vec_sum_a = at::vec::Vectorized<int32_t>(sum_a);
    const int32_t* tmp_in = in + row * ldi;
    float* tmp_out = out + row * ldo;
    float tmp_max = -std::numeric_limits<float>::infinity();
    auto vec_tmp_max = at::vec::Vectorized<float>(tmp_max);
    for (long col = 0; col < vec_size * (N / vec_size); col += vec_size) {
      auto vec_sum_b = at::vec::Vectorized<int32_t>::loadu(sum_b_ptr + col);
      auto tmp0 = at::vec::Vectorized<int32_t>::loadu(tmp_in + col);
      auto tmp1 = tmp0 - vec_sum_b;
      auto tmp2 = tmp1 - vec_sum_a;
      auto tmp3 = tmp2 + vec_beta;
      auto tmp4 = at::vec::convert<float>(tmp3);
      auto tmp5 = tmp4 * vec_alpha;
      vec_tmp_max = at::vec::clamp_min(vec_tmp_max, tmp5);
      _store(tmp_out + col, tmp5);
    }
    tmp_max = std::max(tmp_max, vec_tmp_max.reduce_max());
    for (long col = vec_size * (N / vec_size); col < N; col++) {
      auto sum_b = sum_b_ptr[col];
      auto tmp0 = tmp_in[col];
      auto tmp1 = tmp0 - sum_b;
      auto tmp2 = tmp1 - sum_a;
      auto tmp3 = tmp2 + beta;
      auto tmp4 = (float) tmp3;
      auto tmp5 = tmp4 * alpha;
      tmp_max = std::max(tmp_max, tmp5);
      tmp_out[col] = tmp5;
    }
    sfm_max_ptr[row] = std::max(sfm_max_ptr[row], tmp_max);
  }
}

/*
1. Softmax: sub max, exp, sum reduce, div sum
2. quant
3. sum for attention
*/
template <typename scalar_t>
inline void _sub_exp_sum_div_quant_sum_fusion_kernel(
    const float* in,
    const int64_t& M,
    const int64_t& N_step,
    const int64_t& NSlice,
    const int& ldi,
    const int& ldo,
    const int& kvSize,
    const int& rndkvSplitSize,
    const int& av_gemm_K,
    const int32_t& beta1, // zp_a
    const int32_t& beta2, // zp_b
    const float& alpha, // scale_a
    float* local,
    scalar_t* out,
    float* sfm_max_ptr,
    float* sfm_sum_ptr,
    int32_t* sum_a_ptr) {
  const int32_t vec_size = at::vec::Vectorized<float>::size();
  float min_val = 0;
  float max_val = 255;
  auto vec_min_val = at::vec::Vectorized<float>(min_val);
  auto vec_max_val = at::vec::Vectorized<float>(max_val);
  scalar_t zero = 0;
  auto vec_zero = at::vec::Vectorized<scalar_t>(zero);
  float beta1_float = (float) beta1;
  auto vec_beta1 = at::vec::Vectorized<float>(beta1_float);
  for (int64_t row = 0; row < M; ++row) {
    auto sfm_max = sfm_max_ptr[row];
    auto vec_max = at::vec::Vectorized<float>(sfm_max);
    // sub max, exp, sum reduce
    const float* qk_block_data = in + row * rndkvSplitSize;
    for (int64_t l = 0; l < NSlice; l ++) {
      int64_t n = l * N_step;
      int64_t kvBlockSize = std::min(N_step, kvSize - n);
      const float* tmp_in = qk_block_data + l * ldi;
      float tmp_sum = 0;
      auto vec_tmp_sum = at::vec::Vectorized<float>(tmp_sum);
      float* tmp_out = local + n;
      for (long col = 0; col < vec_size * (kvBlockSize / vec_size); col += vec_size) {
        auto tmp0 = at::vec::Vectorized<float>::loadu(tmp_in + col);
        auto tmp1 = tmp0 - vec_max;
        auto tmp2 = tmp1.exp_u20();
        vec_tmp_sum += tmp2;
        _store(tmp_out + col, tmp2);
      }
      tmp_sum += vec_tmp_sum.reduce_add();
      for (long col = vec_size * (kvBlockSize / vec_size); col < kvBlockSize; col++) {
        auto tmp0 = tmp_in[col];
        auto tmp1 = tmp0 - sfm_max;
        auto tmp2 = exp(tmp1);
        tmp_sum += tmp2;
        tmp_out[col] = tmp2;
      }
      sfm_sum_ptr[row] += tmp_sum;
    }
    // div sum, sum for attention
    auto sum_scale = 1 / sfm_sum_ptr[row] / alpha;
    auto vec_sum_scale = at::vec::Vectorized<float>(sum_scale);
    scalar_t* qk_reduced_block_data = out + row * av_gemm_K;
    for (int64_t l = 0; l < NSlice; l ++) {
      int64_t n = l * N_step;
      int64_t kvBlockSize = std::min(N_step, kvSize - n);
      int32_t tmp_sum = 0;
      auto vec_tmp_sum = at::vec::Vectorized<int32_t>(tmp_sum);
      float* tmp_in = local + n;
      scalar_t* tmp_out = qk_reduced_block_data + l * ldo;
      for (long col = 0; col < vec_size * (kvBlockSize / vec_size); col += vec_size) {
        auto tmp0 = at::vec::Vectorized<float>::loadu(tmp_in + col);
        auto tmp1 = tmp0 * vec_sum_scale;
        auto tmp2 = tmp1.round();
        auto tmp3 = tmp2 + vec_beta1;
        auto tmp4 = at::vec::clamp(tmp3, vec_min_val, vec_max_val);
        _store(tmp_out + col, tmp4);
        auto tmp6 = at::vec::convert<int32_t>(tmp4);
        vec_tmp_sum += tmp6;
      }
      tmp_sum += vec_tmp_sum.reduce_add();
      for (long col = vec_size * (kvBlockSize / vec_size); col < kvBlockSize; col++) {
        auto tmp0 = tmp_in[col];
        auto tmp1 = tmp0 * sum_scale;
        auto tmp2 = std::nearbyint(tmp1);
        auto tmp3 = tmp2 + beta1_float;
        auto tmp4 = std::clamp(tmp3, min_val, max_val);
        tmp_out[col] = tmp4;
        auto tmp6 = (int32_t) tmp4;
        tmp_sum += tmp6;
      }
      sum_a_ptr[row] += tmp_sum * beta2;
      // set zero
      for (long col = kvBlockSize; col <  vec_size * (av_gemm_K / vec_size); col += vec_size) {
        _store(tmp_out + col, vec_zero);
      }
      for (long col = vec_size * (av_gemm_K / vec_size); col < av_gemm_K; col++) {
        tmp_out[col] = zero;
      }
    }
  }
}

template <typename scalar_t>
inline void _sub_exp_sum_div_quant_fusion_kernel(
    const float* in,
    const int64_t& M,
    const int64_t& N_step,
    const int64_t& NSlice,
    const int& ldi,
    const int& ldo,
    const int& kvSize,
    const int& rndkvSplitSize,
    const int& av_gemm_K,
    const int32_t& beta1, // zp_a
    const float& alpha, // scale_a
    float* local,
    scalar_t* out,
    float* sfm_max_ptr,
    float* sfm_sum_ptr) {
  const int32_t vec_size = at::vec::Vectorized<float>::size();
  float min_val = 0;
  float max_val = 255;
  auto vec_min_val = at::vec::Vectorized<float>(min_val);
  auto vec_max_val = at::vec::Vectorized<float>(max_val);
  scalar_t zero = 0;
  auto vec_zero = at::vec::Vectorized<scalar_t>(zero);
  float beta1_float = (float) beta1;
  auto vec_beta1 = at::vec::Vectorized<float>(beta1_float);
  for (int64_t row = 0; row < M; ++row) {
    auto sfm_max = sfm_max_ptr[row];
    auto vec_max = at::vec::Vectorized<float>(sfm_max);
    // sub max, exp, sum reduce
    const float* qk_block_data = in + row * rndkvSplitSize;
    for (int64_t l = 0; l < NSlice; l ++) {
      int64_t n = l * N_step;
      int64_t kvBlockSize = std::min(N_step, kvSize - n);
      const float* tmp_in = qk_block_data + l * ldi;
      float tmp_sum = 0;
      auto vec_tmp_sum = at::vec::Vectorized<float>(tmp_sum);
      float* tmp_out = local + n;
      for (long col = 0; col < vec_size * (kvBlockSize / vec_size); col += vec_size) {
        auto tmp0 = at::vec::Vectorized<float>::loadu(tmp_in + col);
        auto tmp1 = tmp0 - vec_max;
        auto tmp2 = tmp1.exp_u20();
        vec_tmp_sum += tmp2;
        _store(tmp_out + col, tmp2);
      }
      tmp_sum += vec_tmp_sum.reduce_add();
      for (long col = vec_size * (kvBlockSize / vec_size); col < kvBlockSize; col++) {
        auto tmp0 = tmp_in[col];
        auto tmp1 = tmp0 - sfm_max;
        auto tmp2 = exp(tmp1);
        tmp_sum += tmp2;
        tmp_out[col] = tmp2;
      }
      sfm_sum_ptr[row] += tmp_sum;
    }
    // div sum, sum for attention
    auto sum_scale = 1 / sfm_sum_ptr[row] / alpha;
    auto vec_sum_scale = at::vec::Vectorized<float>(sum_scale);
    scalar_t* qk_reduced_block_data = out + row * av_gemm_K;
    for (int64_t l = 0; l < NSlice; l ++) {
      int64_t n = l * N_step;
      int64_t kvBlockSize = std::min(N_step, kvSize - n);
      float* tmp_in = local + n;
      scalar_t* tmp_out = qk_reduced_block_data + l * ldo;
      for (long col = 0; col < vec_size * (kvBlockSize / vec_size); col += vec_size) {
        auto tmp0 = at::vec::Vectorized<float>::loadu(tmp_in + col);
        auto tmp1 = tmp0 * vec_sum_scale;
        auto tmp2 = tmp1.round();
        auto tmp3 = tmp2 + vec_beta1;
        auto tmp4 = at::vec::clamp(tmp3, vec_min_val, vec_max_val);
        _store(tmp_out + col, tmp4);
      }
      for (long col = vec_size * (kvBlockSize / vec_size); col < kvBlockSize; col++) {
        auto tmp0 = tmp_in[col];
        auto tmp1 = tmp0 * sum_scale;
        auto tmp2 = std::nearbyint(tmp1);
        auto tmp3 = tmp2 + beta1_float;
        auto tmp4 = std::clamp(tmp3, min_val, max_val);
        tmp_out[col] = tmp4;
      }
      // set zero
      for (long col = kvBlockSize; col <  vec_size * (av_gemm_K / vec_size); col += vec_size) {
        _store(tmp_out + col, vec_zero);
      }
      for (long col = vec_size * (av_gemm_K / vec_size); col < av_gemm_K; col++) {
        tmp_out[col] = zero;
      }
    }
  }
}

/*
1. dequant
2. quant
*/
template <typename scalar_t>
inline void _dequant_quant_fusion_kernel(
    const int32_t* in,
    const int32_t* sum_a_ptr,
    const int32_t* sum_b_ptr,
    const int& M,
    const int& N,
    const int& ldi,
    const int& ldo,
    const int32_t& beta1, // zp_a*zp_b*k
    const int32_t& beta2, // zp_c
    const float& alpha, // scale_a*scale_b/scale_c
    scalar_t* out) {
  const int32_t vec_size = at::vec::Vectorized<float>::size();
  float min_val = 0;
  float max_val = 255;
  auto vec_min_val = at::vec::Vectorized<float>(min_val);
  auto vec_max_val = at::vec::Vectorized<float>(max_val);
  auto vec_beta1 = at::vec::Vectorized<int32_t>(beta1);
  auto vec_alpha = at::vec::Vectorized<float>(alpha);
  float beta2_float = (float) beta2;
  auto vec_beta2 = at::vec::Vectorized<float>(beta2_float);
  for (long row = 0; row < M; row += 1) {
    auto sum_a = sum_a_ptr[row];
    auto vec_sum_a = at::vec::Vectorized<int32_t>(sum_a);
    const int32_t* tmp_in = in + row * ldi;
    scalar_t* tmp_out = out + row * ldo;
    for (long col = 0; col < vec_size * (N / vec_size); col += vec_size) {
      auto vec_sum_b = at::vec::Vectorized<int32_t>::loadu(sum_b_ptr + col);
      auto tmp0 = at::vec::Vectorized<int32_t>::loadu(tmp_in + col);
      auto tmp1 = tmp0 - vec_sum_b;
      auto tmp2 = tmp1 - vec_sum_a;
      auto tmp3 = tmp2 + vec_beta1;
      auto tmp4 = at::vec::convert<float>(tmp3);
      auto tmp5 = tmp4 * vec_alpha;
      auto tmp6 = tmp5.round();
      auto tmp7 = tmp6 + vec_beta2;
      auto tmp8 = at::vec::clamp(tmp7, vec_min_val, vec_max_val);
      _store(tmp_out + col, tmp8);
    }
    for (long col = vec_size * (N / vec_size); col < N; col++) {
      auto sum_b = sum_b_ptr[col];
      auto tmp0 = tmp_in[col];
      auto tmp1 = tmp0 - sum_b;
      auto tmp2 = tmp1 - sum_a;
      auto tmp3 = tmp2 + beta1;
      auto tmp4 = (float) tmp3;
      auto tmp5 = tmp4 * alpha;
      auto tmp6 = std::nearbyint(tmp5);
      auto tmp7 = tmp6 + beta2_float;
      auto tmp8 = std::clamp(tmp7, min_val, max_val);
      tmp_out[col] = tmp8;
    }
  }
}

template <typename scalar_t>
inline void _int_sum_b_contiguous_kernel_helper(
    const scalar_t* in,
    int32_t* out,
    const int& N,
    const int32_t& scale) {
  const int32_t vec_size = at::vec::Vectorized<int32_t>::size();
  int32_t tmp_sum = 0;
  auto vec_tmp_sum = at::vec::Vectorized<int32_t>(tmp_sum);
  for (long i = 0; i < vec_size * (N / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(in + i);
    auto tmp1 = at::vec::convert<int32_t>(tmp0);
    vec_tmp_sum = vec_tmp_sum + tmp1;
  }
  tmp_sum += vec_tmp_sum.reduce_add();
  for (long i = vec_size * (N / vec_size); i < N; i++) {
    tmp_sum += static_cast<int32_t>(in[i]);
  }
  out[0] = tmp_sum * scale;
}

template <typename scalar_t>
inline void _int_sum_b_contiguous_kernel(
    const scalar_t* in,
    int32_t* out,
    const int& M,
    const int& N,
    const int& ld,
    const int32_t& scale) {
  for (long r = 0; r < M; r += 1) {
    _int_sum_b_contiguous_kernel_helper(in + r * ld, out + r, N, scale);
  }
}

template <typename scalar_t>
inline void _int_sum_a_contiguous_kernel(
    const scalar_t* in,
    int32_t* out,
    const int& M,
    const int& N,
    const int& ld,
    const int32_t& scale) {
  const int32_t vec_size = at::vec::Vectorized<int32_t>::size();
  auto vec_scale = at::vec::Vectorized<int32_t>(scale);
  // initialization with 0
  int32_t zero = 0;
  auto vec_zero = at::vec::Vectorized<int32_t>(zero);
  for (long i = 0; i < vec_size * (M / vec_size); i += vec_size) {
    _store(out + i, vec_zero);
  }
  for (long i = vec_size * (M / vec_size); i < M; i++) {
    out[i] = zero;
  }
  // sum
  for (long j = 0; j < N; j++) {
    const scalar_t* tmp_in = in + j * ld;
    for (long i = 0; i < vec_size * (M / vec_size); i += vec_size) {
      auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(tmp_in + i);
      auto tmp1 = at::vec::Vectorized<int32_t>::loadu(out + i);
      auto tmp2 = at::vec::convert<int32_t>(tmp0);
      auto tmp3 = tmp1 + tmp2;
      _store(out + i, tmp3);
    }
    for (long i = vec_size * (M / vec_size); i < M; i++) {
      auto tmp0 = tmp_in[i];
      auto tmp1 = out[i];
      auto tmp2 = static_cast<int32_t>(tmp0);
      auto tmp3 = tmp1 + tmp2;
      out[i] = tmp3;
    }
  }
  // scale
  for (long i = 0; i < vec_size * (M / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<int32_t>::loadu(out + i);
    auto tmp1 = tmp0 * vec_scale;
    _store(out + i, tmp1);
  }
  for (long i = vec_size * (M / vec_size); i < M; i++) {
    auto tmp0 = out[i];
    auto tmp1 = tmp0 * scale;
    out[i] = tmp1;
  }
}

template <typename scalar_t>
inline void do_transpose(
    scalar_t* src,
    scalar_t* dst,
    int64_t in_rows,
    int64_t in_cols,
    int64_t ldi,
    int64_t ldo) {
  for (int64_t r=0; r<in_rows; r++) {
    for (int64_t c=0; c<in_cols; c++) {
      *(dst + c * ldo + r) = *(src + r * ldi + c);
    }
  }
}

template <typename scalar_t>
inline void pad_remain_row_col(
    scalar_t* value_ptr,
    int rows,
    int cols,
    int prows,
    int pcols,
    int ldi,
    scalar_t pad_val=0) {
  auto psize = pcols - cols;
  if (psize == 0 && prows == rows) {
    return;
  }
  const int32_t vec_size = at::vec::Vectorized<scalar_t>::size();
  auto pad = at::vec::Vectorized<scalar_t>(pad_val);
  if (psize > 0) {
    for (int i = 0; i < rows; i++) {
      int j = 0;
      for (; j < psize - (psize % vec_size); j += vec_size) {
        pad.store(value_ptr + i * ldi + cols + j);
      }
      if (j < psize) {
        pad.store(value_ptr + i * ldi + cols + j, psize - j);
      }
    }
  }

  for (int i = rows; i < prows; i++) {
    int j = 0;
    for (; j < pcols - (pcols % vec_size); j += vec_size) {
      pad.store(value_ptr + i * ldi + j);
    }
    if (j < pcols) {
      pad.store(value_ptr + i * ldi + j, pcols - j);
    }
  }
}

template <typename scalar_t>
inline void copy_value_with_pad(
    scalar_t* value_ptr,
    scalar_t* dst_ptr,
    int rows,
    int cols,
    int prows,
    int pcols,
    int ldi,
    scalar_t pad_val=0) {
  const int32_t vec_size = at::vec::Vectorized<scalar_t>::size();
  auto pad = at::vec::Vectorized<scalar_t>(pad_val);
  int i = 0;
  for (; i < rows; i++) {
    int j = 0;
    for (; j < cols - (cols % vec_size); j += vec_size) {
      auto vec_v =
          at::vec::Vectorized<scalar_t>::loadu(value_ptr + i * ldi + j);
      vec_v.store(dst_ptr + i * pcols + j);
    }

    if (j < cols) {
      auto vec_v = at::vec::Vectorized<scalar_t>::loadu(
          value_ptr + i * ldi + j, cols - j);
      vec_v.store(dst_ptr + i * pcols + j, cols - j);
    }

    // col padding
    auto psize = pcols - cols;
    if (psize > 0) {
      int pj = 0;
      for (; pj < psize - (psize % vec_size); pj += vec_size) {
        pad.store(dst_ptr + i * pcols + cols + pj);
      }
      if (pj < psize) {
        pad.store(dst_ptr + i * pcols + cols + pj, psize - pj);
      }
    }
  }

  // row padding
  for (; i < prows; i++) {
    int j = 0;
    for (; j < pcols - (pcols % vec_size); j += vec_size) {
      pad.store(dst_ptr + i * pcols + j);
    }
    if (j < pcols) {
      pad.store(dst_ptr + i * pcols + j, pcols - j);
    }

  }

}

// UINT8 - one parallel loop with u8u8s32 GEMM 
template <typename scalar_t, typename mask_t, int64_t q_split_size, int64_t kv_split_size>
inline typename std::enable_if_t<std::is_same_v<scalar_t, unsigned char>, void>
sdpa_int8_kernel_one_loop_impl(
    at::Tensor& output,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    std::optional<Tensor> attn_mask,
    std::optional<double> scale,
    double dropout_p,
    bool is_causal,
    int64_t q_zp,
    float q_scale,
    int64_t k_zp,
    float k_scale,
    int64_t v_zp,
    float v_scale,
    int64_t a_zp,
    float a_scale,
    int64_t o_zp,
    float o_scale) {
  // Query (Batch x Num_heads  x Q_seq_len  x Dim_per_head)
  //    -> (Batch x Q_seq_len  x Num_heads  x Dim_per_head)
  // Key   (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  // Value (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  at::Tensor query = q.transpose(1, 2);
  at::Tensor key = k.transpose(1, 2);
  at::Tensor value = v.transpose(1, 2);

  using accum_t = float;
  accum_t scaling_factor =
      sdp::calculate_scale(query, scale).expect_float();
  int block_64 = 64;
  auto u8_dt = at::ScalarType::Byte;

  // Sizes
  TORCH_CHECK(
      (query.size(3) == value.size(3)) && (key.size(3) == value.size(3)),
      "scaled_dot_product_attention_sdpa: Q/K/V should have the same head size");
  TORCH_CHECK(
      kv_split_size % block_64 == 0, "kv_split_size is not divisble by ", block_64);

  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(1);
  int64_t kvSize = value.size(1);
  int64_t num_head = query.size(2);
  int64_t headSize = query.size(3);

  bool has_attn_mask = attn_mask.has_value() && attn_mask.value().numel();
  if (has_attn_mask) {
    reshape_attn_mask_to_4d(attn_mask.value(), batchSize, num_head, qSize, kvSize);
  }

  // Strides
  int64_t qStrideB = query.stride(0);
  int64_t qStrideM = query.stride(1);
  int64_t qStrideH = query.stride(2);
  int64_t kStrideB = key.stride(0);
  int64_t kStrideN = key.stride(1);
  int64_t kStrideH = key.stride(2);
  int64_t vStrideB = value.stride(0);
  int64_t vStrideN = value.stride(1);
  int64_t vStrideH = value.stride(2);
  int64_t oStrideB = output.stride(0);
  int64_t oStrideM = output.stride(1);
  int64_t oStrideH = output.stride(2);
  int64_t mStrideB =
      (has_attn_mask && attn_mask.value().size(0) > 1)
      ? attn_mask.value().stride(0)
      : 0;
  int64_t mStrideH =
      (has_attn_mask && attn_mask.value().size(1) > 1)
      ? attn_mask.value().stride(1)
      : 0;
  int64_t mStrideM =
      (has_attn_mask && attn_mask.value().size(2) > 1)
      ? attn_mask.value().stride(2)
      : 0;
  int64_t mStrideN =
      (has_attn_mask && attn_mask.value().size(3) > 1)
      ? attn_mask.value().stride(3)
      : 0;

  int64_t qSplitSize = q_split_size > qSize ? qSize : q_split_size;
  int64_t kvSplitSize = kv_split_size > kvSize ? kvSize : kv_split_size;
  int64_t qSlice = (qSize - 1) / qSplitSize + 1;
  int64_t qTail = (qSize - 1) % qSplitSize + 1;
  int64_t kvSlice = (kvSize - 1) / kvSplitSize + 1;
  int64_t kvTail = (kvSize - 1) % kvSplitSize + 1;
  int64_t num_thread = at::get_num_threads();

  int64_t rndHeadSize = (headSize + block_64 - 1L) / block_64 * block_64;
  // one of 16, 32, 48, 64
  auto select_tail_tail_block_size = [](int64_t size) -> int64_t {
    if (size == 0) {
      return 0;
    } else if (size <= 16) {
      return 16;
    } else if (size <= 32) {
      return 32;
    } else if (size <= 48) {
      return 48;
    } else {
      return 64;
    }
  };
  int64_t kv_tail_tail_block_size = select_tail_tail_block_size(kvTail % block_64);
  int64_t rndkvSplitSize = (kvSplitSize + block_64 - 1L) / block_64 * block_64;
  int64_t rndkvTail = (kvTail + block_64 - 1L) / block_64 * block_64;
  int64_t rndkvSize = kv_split_size > kvSize ? rndkvTail : rndkvSplitSize * kvSlice + rndkvTail;

  bool av_gemm_K_mul4 = kvSplitSize % 4 == 0;
  int av_gemm_K_padding = av_gemm_K_mul4 ? 0 : 4 - kvSplitSize % 4;
  // // If K of Gemm is not even, use mkl gemm instead of tpp for BF16
  int av_gemm_K = kvSplitSize + av_gemm_K_padding;

  // Data ptrs
  scalar_t* q_data = query.data_ptr<scalar_t>();
  scalar_t* k_data = key.data_ptr<scalar_t>();
  scalar_t* v_data = value.data_ptr<scalar_t>();
  mask_t* mask_data = attn_mask.has_value()
      ? attn_mask.value().data_ptr<mask_t>()
      : nullptr;
  scalar_t* out_data = output.data_ptr<scalar_t>();

  // Create tpp kernels for Query @ Key
  bool headSize_mul4 = headSize % 4 == 0;
  // If K of Gemm is not even, use mkl gemm instead of tpp for BF16
  int qk_gemm_K_padding = headSize_mul4 ? 0 : 4 - headSize % 4;
  int qk_gemm_K = headSize + qk_gemm_K_padding;

  int64_t qk_reduce_strideL = qSplitSize * av_gemm_K;
  int64_t v_reorder_strideL = av_gemm_K * rndHeadSize;

  int64_t total_size_uint8_per_thread =
    /* qk */ kvSlice * qSplitSize * rndkvSplitSize * 4 +
    /* qk_local  */ kvSlice * av_gemm_K * 4 +
    /* qk_reduce  */ kvSlice * qk_reduce_strideL +
    /* qk_s32   */ qSplitSize * rndkvSplitSize * 4 +
    /* dst_s32  */ qSplitSize * rndHeadSize * 4 +
    /* softmax_sum   */ qSplitSize * 4 +
    /* query_sum     */ qSplitSize * 4 +
    /* attention_sum */ qSplitSize * 4 +
    /* softmax max */ qSplitSize * 4 +
    /* query_padding_data */ qSplitSize * qk_gemm_K +
    /* key_sum */ kvSize * 4 +
    /* value_sum */ headSize * 4 +
    /* key_t_reorder */ qk_gemm_K * rndkvSize +
    /* value_t_reorder */ kvSlice * v_reorder_strideL;

  at::Tensor total_buf = at::empty(
      {num_thread, total_size_uint8_per_thread},
      query.options());
  scalar_t* total_buf_data = total_buf.data_ptr<scalar_t>();

  at::parallel_for(
      0, batchSize * num_head, 1, [&](int64_t begin, int64_t end) {
        int64_t i = 0, j = 0;
        at::native::data_index_init(
            begin, i, batchSize, j, num_head);
        int ompIdx = at::get_thread_num();
        scalar_t* total_buf_ptr = total_buf_data + ompIdx * total_size_uint8_per_thread;
        int32_t offset = 0;
        accum_t* qk_data = reinterpret_cast<accum_t*>(total_buf_ptr);
        offset += kvSlice * qSplitSize * rndkvSplitSize * 4;
        accum_t* qk_local_data = reinterpret_cast<accum_t*>(total_buf_ptr + offset);
        offset += kvSlice * av_gemm_K * 4;
        scalar_t* qk_reduced_data = reinterpret_cast<scalar_t*>(total_buf_ptr + offset);
        offset += kvSlice * qk_reduce_strideL;
        int32_t* qk_s32_data = reinterpret_cast<int32_t*>(total_buf_ptr + offset);
        offset += qSplitSize * rndkvSplitSize * 4;
        int32_t* dst_s32_data = reinterpret_cast<int32_t*>(total_buf_ptr + offset);
        offset += qSplitSize * rndHeadSize * 4;
        accum_t* sfm_sum_ptr = reinterpret_cast<accum_t*>(total_buf_ptr + offset);
        offset += qSplitSize * 4;
        int32_t* q_sum_ptr = reinterpret_cast<int32_t*>(total_buf_ptr + offset);
        offset += qSplitSize * 4;
        int32_t* a_sum_ptr = reinterpret_cast<int32_t*>(total_buf_ptr + offset);
        offset += qSplitSize * 4;
        accum_t* sfm_max_ptr = reinterpret_cast<accum_t*>(total_buf_ptr + offset);
        offset += qSplitSize * 4;
        scalar_t* query_t_padding_ptr = reinterpret_cast<scalar_t*>(total_buf_ptr + offset);
        offset += qSplitSize * qk_gemm_K;

        int32_t* k_sum_ptr = reinterpret_cast<int32_t*>(total_buf_ptr + offset);
        offset += kvSize * 4;
        int32_t* v_sum_ptr = reinterpret_cast<int32_t*>(total_buf_ptr + offset);
        offset += headSize * 4;
        scalar_t* key_reorder_ptr = reinterpret_cast<scalar_t*>(total_buf_ptr + offset);
        offset += qk_gemm_K * rndkvSize;
        scalar_t* value_reorder_ptr = reinterpret_cast<scalar_t*>(total_buf_ptr + offset);

        uint8_t* B_blocked_xform_u8 = new uint8_t[std::max(qk_gemm_K, av_gemm_K) * block_64];

        for (const auto z : c10::irange(begin, end)) {
          (void)z; // Suppress unused variable

          // sum k and v
          if (q_zp == 0) {
            fill_stub(k_sum_ptr, static_cast<int32_t>(0), kvSize);
          } else {
            _int_sum_b_contiguous_kernel(k_data + i * kStrideB + j * kStrideH,
              k_sum_ptr,
              kvSize, headSize, kStrideN, q_zp);
          }
          if (a_zp == 0) {
            fill_stub(v_sum_ptr, static_cast<int32_t>(0), headSize);
          } else {
            _int_sum_a_contiguous_kernel(v_data + i * vStrideB + j * vStrideH,
              v_sum_ptr,
              headSize, kvSize, vStrideN, a_zp);
          }

          // pack
          for (int64_t n = 0; n < kvSize; n += kvSplitSize) {
            if (n + kvSplitSize < kvSize) {
              for (int64_t b = 0; b < rndkvSplitSize; b += block_64) {
                do_transpose(
                    k_data + i * kStrideB + j * kStrideH + n * kStrideN + b * kStrideN,
                    B_blocked_xform_u8,
                    std::min(int(kvSplitSize - b), block_64),
                    headSize,
                    kStrideN,
                    block_64);
                if (!headSize_mul4) {
                  pad_remain_row_col(
                      B_blocked_xform_u8,
                      headSize,
                      block_64,
                      qk_gemm_K,
                      block_64,
                      block_64
                    );
                }
                // Pack
                at::native::cpublas::pack(
                    qk_gemm_K, // K
                    block_64, // N
                    block_64, // ld_in
                    block_64, // ld_out
                    u8_dt, // dt_in
                    u8_dt, // dt_out
                    B_blocked_xform_u8,
                    key_reorder_ptr + n * qk_gemm_K +
                        b * qk_gemm_K);
              }
              // split headSize to block_64, block_64, block_64 ...
              // [kvSplitSize, headSize] -> [av_gemm_K,  block_64 ...]
              for (int64_t b = 0; b < rndHeadSize; b += block_64) {
                at::native::cpublas::pack(
                    av_gemm_K,
                    block_64,
                    vStrideN, // block_64,
                    block_64,
                    u8_dt,
                    u8_dt,
                    v_data + i * vStrideB + j * vStrideH + n * vStrideN + b,
                    value_reorder_ptr + n * rndHeadSize +
                      av_gemm_K * b);
              }
            } else {
              // tail
              auto block_size = kvTail >= block_64 ? block_64 : kv_tail_tail_block_size;
              int64_t b = 0;
              while (b < rndkvTail) {
                do_transpose(
                    k_data + i * kStrideB + j * kStrideH + n * kStrideN + b * kStrideN,
                    B_blocked_xform_u8,
                    std::min(kvTail - b, block_size),
                    headSize,
                    kStrideN,
                    block_size);
                if (!headSize_mul4) {
                  pad_remain_row_col(
                      B_blocked_xform_u8,
                      headSize,
                      block_size,
                      qk_gemm_K,
                      block_size,
                      block_size
                    );
                }
                // Pack
                at::native::cpublas::pack(
                    qk_gemm_K,
                    block_size,
                    block_size,
                    block_size,
                    u8_dt,
                    u8_dt,
                    B_blocked_xform_u8,
                    key_reorder_ptr + n * qk_gemm_K +
                        b * qk_gemm_K);
                b += block_size;
                block_size = (kvTail - b) >= block_64 ? block_64 : kv_tail_tail_block_size;
              }
              // split headSize to block_64, block_64, block_64 ...
              // [kvTail, headSize] -> [av_gemm_K_tail,  block_64 ...]
              for (int64_t b = 0; b < headSize; b += block_64) {
                at::native::cpublas::pack(
                    av_gemm_K,
                    block_64,
                    vStrideN, // block_64,
                    block_64,
                    u8_dt,
                    u8_dt,
                    v_data + i * vStrideB + j * vStrideH + n * vStrideN + b,
                    value_reorder_ptr + n * rndHeadSize +
                      av_gemm_K * b);
              }
            }
          }

          // sdpa core
          for (int64_t k = 0; k < qSlice; k++) {
            int64_t m = k * qSplitSize;
            int64_t qBlockSize = std::min(qSplitSize, qSize - m);
            // Initialize sum and max
            fill_stub(
                sfm_sum_ptr, static_cast<accum_t>(0), qSplitSize);
            fill_stub(
                a_sum_ptr, static_cast<int32_t>(0), qSplitSize);
            fill_stub(
                sfm_max_ptr, static_cast<accum_t>(-std::numeric_limits<accum_t>::infinity()), qSplitSize);
            int64_t num_keys =
                is_causal ? std::min(m + qBlockSize, kvSize) : kvSize;
            copy_value_with_pad(
                q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                query_t_padding_ptr,
                qBlockSize,
                headSize,
                qBlockSize,
                qk_gemm_K,
                qStrideM);

            if (k_zp != 0) {
              _int_sum_b_contiguous_kernel(q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                    q_sum_ptr, qBlockSize, headSize, qStrideM, k_zp);
            } else {
              fill_stub(
                q_sum_ptr, static_cast<int32_t>(0), qSplitSize);
            }
            const int64_t rkvSlice = (num_keys - 1) / kvSplitSize + 1;
            for (int64_t l = 0; l < rkvSlice; l++) {
              int64_t n = l * kvSplitSize;
              int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
              // Calculate sums for dequant compensation item
              if (qBlockSize == qSplitSize) {
                // q main
                if (n + kvSplitSize < kvSize) {
                  // k main
                  for (int64_t b = 0; b < kvSplitSize; b += block_64) {
                    at::native::cpublas::brgemm(
                          qSplitSize, block_64, qk_gemm_K,
                          qk_gemm_K, // lda
                          block_64, //ldb
                          rndkvSplitSize, //ldc,
                          false,
                          query_t_padding_ptr,
                          key_reorder_ptr + n * qk_gemm_K +
                            b * qk_gemm_K,
                          qk_s32_data + b);
                  }
                } else {
                  auto block_size = kvTail >= block_64 ? block_64 : kv_tail_tail_block_size;
                  int64_t b = 0;
                  while (b < kvTail) {
                    at::native::cpublas::brgemm(
                          qSplitSize, block_size, qk_gemm_K,
                          qk_gemm_K, // lda
                          block_size, //ldb
                          rndkvTail, //ldc
                          false,
                          query_t_padding_ptr,
                          key_reorder_ptr + n * qk_gemm_K +
                              b * qk_gemm_K,
                          qk_s32_data + b);
                    b += block_size;
                    block_size = (kvTail - b) >= block_64 ? block_64 : kv_tail_tail_block_size;
                  }
                }
              } else {
                if (n + kvSplitSize < kvSize) {
                  for (int64_t b = 0; b < kvSplitSize; b += block_64) {
                    at::native::cpublas::brgemm(
                          qTail, block_64, qk_gemm_K,
                          qk_gemm_K,// lda
                          block_64, //ldb
                          rndkvSplitSize, //ldc
                          false,
                          query_t_padding_ptr,
                          key_reorder_ptr + n * qk_gemm_K +
                            b * qk_gemm_K,
                          qk_s32_data + b);
                  }
                } else {
                  // k tail
                  auto block_size = kvTail >= block_64 ? block_64 : kv_tail_tail_block_size;
                  int64_t b = 0;
                  while (b < kvTail) {
                    at::native::cpublas::brgemm(
                          qTail, block_size, qk_gemm_K,
                          qk_gemm_K, // lda
                          block_size, //ldb
                          rndkvTail, //ldc
                          false,
                          query_t_padding_ptr,
                          key_reorder_ptr + n * qk_gemm_K +
                              b * qk_gemm_K,
                          qk_s32_data + b);
                  b += block_size;
                  block_size = (kvTail - b) >= block_64 ? block_64 : kv_tail_tail_block_size;
                  }
                }
              }

              // do dequant compensation, add mask, max reduce for softmax, and convert qk from s32 to fp32
              int64_t rndkvBlockSize = kvBlockSize == kvSplitSize ? rndkvSplitSize : rndkvTail;
              accum_t* qk_block_data = qk_data + l * qSplitSize * rndkvSplitSize;
              if (has_attn_mask) {
                mask_t* mask_data_offset = mask_data + i * mStrideB + j * mStrideH + m * mStrideM + (mStrideN == 0 ? 0 : n);
                _dequant_mask_max_fusion_kernel(
                  qk_s32_data, //in
                  mask_data_offset, //mask_ptr
                  q_sum_ptr, //sum_a_ptr
                  k_sum_ptr + n, //sum_b_ptr
                  qBlockSize, //M
                  kvBlockSize, //N
                  rndkvBlockSize, //ldi
                  mStrideM, //ldm
                  rndkvSplitSize,//kvBlockSize, //ldo
                  q_zp * k_zp * headSize, //zp_a*zp_b*k=beta
                  q_scale * k_scale * scaling_factor, //scale_a*scale_b*scale_sdpa=alpha
                  qk_block_data, //out
                  sfm_max_ptr // sfm_max_ptr
                );
              } else {
                _dequant_max_fusion_kernel(
                  qk_s32_data, //in
                  q_sum_ptr, //sum_a_ptr
                  k_sum_ptr + n, //sum_b_ptr
                  qBlockSize, //M
                  kvBlockSize, //N
                  rndkvBlockSize, //ldi
                  rndkvSplitSize,//kvBlockSize, //ldo
                  q_zp * k_zp * headSize, //zp_a*zp_b*k=beta
                  q_scale * k_scale * scaling_factor, //scale_a*scale_b*scale_sdpa=alpha
                  qk_block_data, //out
                  sfm_max_ptr // sfm_max_ptr
                );
              }
            }
            // sub max, exp, sum reduce, div sum for softmax
            // and quant
            // and sum for attention
            if (v_zp == 0) {
              _sub_exp_sum_div_quant_fusion_kernel(
                qk_data, //in
                qBlockSize, //M
                kvSplitSize, //N_step
                rkvSlice, //NSlices
                qSplitSize * rndkvSplitSize, //ldi
                qk_reduce_strideL, //ldo
                kvSize, //kvSize
                rndkvSplitSize, //rndkvSplitSize
                av_gemm_K, //av_gemm_K
                a_zp, // zp_a=beta1
                a_scale, // scale_a=alpha
                qk_local_data, //local
                qk_reduced_data, //out
                sfm_max_ptr, //sfm_max_ptr
                sfm_sum_ptr //sfm_sum_ptr
              );
            } else {
              _sub_exp_sum_div_quant_sum_fusion_kernel(
                qk_data, //in
                qBlockSize, //M
                kvSplitSize, //N_step
                rkvSlice, //NSlice
                qSplitSize * rndkvSplitSize, //ldi
                qk_reduce_strideL, //ldo
                kvSize, //kvSize
                rndkvSplitSize, //rndkvSplitSize
                av_gemm_K, //av_gemm_K
                a_zp, // zp_a=beta1
                v_zp, // zp_b=beta2
                a_scale, // scale_a=alpha
                qk_local_data, //local
                qk_reduced_data, //out
                sfm_max_ptr, //sfm_max_ptr
                sfm_sum_ptr, //sfm_sum_ptr
                a_sum_ptr //a_sum_ptr
              );
            }
            // Calculate Softmax(q @ k.T) @ v
            for (int64_t b = 0; b < headSize; b += block_64) {
              auto value_reorder_b = value_reorder_ptr + b * av_gemm_K;
              auto dst_s32_b = dst_s32_data + b;
              for (int64_t s = 0; s < kvSlice; s++) {
                at::native::cpublas::brgemm(
                    qSplitSize, block_64, av_gemm_K,
                    av_gemm_K, // lda
                    rndHeadSize, //block_64, //ldb
                    rndHeadSize, //ldc
                    s != 0,
                    qk_reduced_data + s * qk_reduce_strideL,
                    value_reorder_b + s * v_reorder_strideL,
                    dst_s32_b);
              }
            }

            // After the last gemm,
            // do dequant compensation, quant and convert from s32 to int8
            _dequant_quant_fusion_kernel(
              dst_s32_data, //in
              a_sum_ptr, //sum_a_ptr
              v_sum_ptr, //sum_b_ptr
              qBlockSize, //M
              headSize, //N
              rndHeadSize, //ldi
              oStrideM, //ldo
              a_zp * v_zp * kvSize, //zp_a*zp_b*k=beta1
              o_zp, //zp_c=beta2
              a_scale * v_scale / o_scale, //scale_a*scale_b/scale_c=alpha
              out_data + i * oStrideB + j * oStrideH + m * oStrideM //out
            );
          }
          // Move to the next query
          at::native::data_index_step(i, batchSize, j, num_head);
        }
      });
  // Once all computations are done, need to release HW context.
  at::native::cpublas::brgemm_release();
}

// UINT8 - several parallel loops with u8u8s32 GEMM
template <typename scalar_t, typename mask_t, int64_t q_split_size, int64_t kv_split_size>
inline typename std::enable_if_t<std::is_same_v<scalar_t, unsigned char>, void>
sdpa_int8_kernel_several_loops_impl(
    at::Tensor& output,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    std::optional<Tensor> attn_mask,
    std::optional<double> scale,
    double dropout_p,
    bool is_causal,
    int64_t q_zp,
    float q_scale,
    int64_t k_zp,
    float k_scale,
    int64_t v_zp,
    float v_scale,
    int64_t a_zp,
    float a_scale,
    int64_t o_zp,
    float o_scale) {
  // Query (Batch x Num_heads  x Q_seq_len  x Dim_per_head)
  //    -> (Batch x Q_seq_len  x Num_heads  x Dim_per_head)
  // Key   (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  // Value (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  at::Tensor query = q.transpose(1, 2);
  at::Tensor key = k.transpose(1, 2);
  at::Tensor value = v.transpose(1, 2);

  using accum_t = float;
  accum_t scaling_factor =
      sdp::calculate_scale(query, scale).expect_float();
  int block_64 = 64;
  auto u8_dt = at::ScalarType::Byte;

  // Sizes
  TORCH_CHECK(
      (query.size(3) == value.size(3)) && (key.size(3) == value.size(3)),
      "scaled_dot_product_attention_sdpa: Q/K/V should have the same head size");
  TORCH_CHECK(
      kv_split_size % block_64 == 0, "kv_split_size is not divisble by ", block_64);

  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(1);
  int64_t kvSize = value.size(1);
  int64_t num_head = query.size(2);
  int64_t headSize = query.size(3);

  bool has_attn_mask = attn_mask.has_value() && attn_mask.value().numel();
  if (has_attn_mask) {
    reshape_attn_mask_to_4d(attn_mask.value(), batchSize, num_head, qSize, kvSize);
  }

  // Strides
  int64_t qStrideB = query.stride(0);
  int64_t qStrideM = query.stride(1);
  int64_t qStrideH = query.stride(2);
  int64_t kStrideB = key.stride(0);
  int64_t kStrideN = key.stride(1);
  int64_t kStrideH = key.stride(2);
  int64_t vStrideB = value.stride(0);
  int64_t vStrideN = value.stride(1);
  int64_t vStrideH = value.stride(2);
  int64_t oStrideB = output.stride(0);
  int64_t oStrideM = output.stride(1);
  int64_t oStrideH = output.stride(2);
  int64_t mStrideB =
      (has_attn_mask && attn_mask.value().size(0) > 1)
      ? attn_mask.value().stride(0)
      : 0;
  int64_t mStrideH =
      (has_attn_mask && attn_mask.value().size(1) > 1)
      ? attn_mask.value().stride(1)
      : 0;
  int64_t mStrideM =
      (has_attn_mask && attn_mask.value().size(2) > 1)
      ? attn_mask.value().stride(2)
      : 0;
  int64_t mStrideN =
      (has_attn_mask && attn_mask.value().size(3) > 1)
      ? attn_mask.value().stride(3)
      : 0;

  int64_t qSplitSize = q_split_size > qSize ? qSize : q_split_size;
  int64_t kvSplitSize = kv_split_size > kvSize ? kvSize : kv_split_size;
  int64_t qSlice = (qSize - 1) / qSplitSize + 1;
  int64_t qTail = (qSize - 1) % qSplitSize + 1;
  int64_t kvSlice = (kvSize - 1) / kvSplitSize + 1;
  int64_t kvTail = (kvSize - 1) % kvSplitSize + 1;
  int64_t num_thread = at::get_num_threads();

  int64_t rndHeadSize = (headSize + block_64 - 1L) / block_64 * block_64;
  // one of 16, 32, 48, 64
  auto select_tail_tail_block_size = [](int64_t size) -> int64_t {
    if (size == 0) {
      return 0;
    } else if (size <= 16) {
      return 16;
    } else if (size <= 32) {
      return 32;
    } else if (size <= 48) {
      return 48;
    } else {
      return 64;
    }
  };
  int64_t kv_tail_tail_block_size = select_tail_tail_block_size(kvTail % block_64);
  int64_t rndkvSplitSize = (kvSplitSize + block_64 - 1L) / block_64 * block_64;
  int64_t rndkvTail = (kvTail + block_64 - 1L) / block_64 * block_64;
  int64_t rndkvSize = kv_split_size > kvSize ? rndkvTail : rndkvSplitSize * kvSlice + rndkvTail;

  bool av_gemm_K_mul4 = kvSplitSize % 4 == 0;
  int av_gemm_K_padding = av_gemm_K_mul4 ? 0 : 4 - kvSplitSize % 4;
  // // If K of Gemm is not even, use mkl gemm instead of tpp for BF16
  int av_gemm_K = kvSplitSize + av_gemm_K_padding;

  // Data ptrs
  scalar_t* q_data = query.data_ptr<scalar_t>();
  scalar_t* k_data = key.data_ptr<scalar_t>();
  scalar_t* v_data = value.data_ptr<scalar_t>();
  mask_t* mask_data = attn_mask.has_value()
      ? attn_mask.value().data_ptr<mask_t>()
      : nullptr;
  scalar_t* out_data = output.data_ptr<scalar_t>();

  // Create tpp kernels for Query @ Key
  bool headSize_mul4 = headSize % 4 == 0;
  // If K of Gemm is not even, use mkl gemm instead of tpp for BF16
  int qk_gemm_K_padding = headSize_mul4 ? 0 : 4 - headSize % 4;
  int qk_gemm_K = headSize + qk_gemm_K_padding;

  int64_t qk_reduce_strideL = qSplitSize * av_gemm_K;
  int64_t v_reorder_strideL = av_gemm_K * rndHeadSize;

  int64_t total_size_uint8_per_thread =
    /* qk */ kvSlice * qSplitSize * rndkvSplitSize * 4 +
    /* qk_local  */ kvSlice * av_gemm_K * 4 +
    /* qk_reduce  */ kvSlice * qk_reduce_strideL +
    /* qk_s32   */ qSplitSize * rndkvSplitSize * 4 +
    /* dst_s32  */ qSplitSize * rndHeadSize * 4 +
    /* softmax_sum   */ qSplitSize * 4 +
    /* query_sum     */ qSplitSize * 4 +
    /* attention_sum */ qSplitSize * 4 +
    /* softmax max */ qSplitSize * 4 +
    /* query_padding_data */ qSplitSize * qk_gemm_K;

  at::Tensor total_buf = at::empty(
      {num_thread, total_size_uint8_per_thread},
      query.options());
  scalar_t* total_buf_data = total_buf.data_ptr<scalar_t>();

  int64_t kv_sum_size_per_BH =
    /* key_sum */ kvSize +
    /* value_sum */ headSize;

  at::Tensor kv_sum_buf = at::empty(
      {batchSize, num_head, kv_sum_size_per_BH},
      query.options().dtype(at::kInt));
  int32_t* kv_sum_buf_data = kv_sum_buf.data_ptr<int32_t>();

  int64_t kv_reorder_size_per_BH =
    /* key_t_reorder */ qk_gemm_K * rndkvSize +
    /* value_t_reorder */ kvSlice * v_reorder_strideL;

  at::Tensor kv_reorder_buf = at::empty(
      {batchSize, num_head, kv_reorder_size_per_BH},
      query.options());
  scalar_t* kv_reorder_buf_data = kv_reorder_buf.data_ptr<scalar_t>();
  scalar_t* key_reorder_ptr = kv_reorder_buf_data;
  scalar_t* value_reorder_ptr = kv_reorder_buf_data + batchSize * num_head * qk_gemm_K * rndkvSize;

  // sum k and v
  at::parallel_for(
      0, batchSize * num_head, 1, [&](int64_t begin, int64_t end) {
        int64_t i = 0, j = 0;
        at::native::data_index_init(
            begin, i, batchSize, j, num_head);
        for (const auto z : c10::irange(begin, end)) {
          (void)z; // Suppress unused variable
          int32_t* kv_sum_ptr = kv_sum_buf_data
              + i * num_head * kv_sum_size_per_BH
              + j * kv_sum_size_per_BH;
          int32_t* k_sum_ptr = kv_sum_ptr;
          int32_t* v_sum_ptr = kv_sum_ptr + kvSize;
          if (q_zp == 0) {
            fill_stub(k_sum_ptr, static_cast<int32_t>(0), kvSize);
          } else {
            _int_sum_b_contiguous_kernel(k_data + i * kStrideB + j * kStrideH,
              k_sum_ptr,
              kvSize, headSize, kStrideN, q_zp);
          }
          if (a_zp == 0) {
            fill_stub(v_sum_ptr, static_cast<int32_t>(0), headSize);
          } else {
            _int_sum_a_contiguous_kernel(v_data + i * vStrideB + j * vStrideH,
              v_sum_ptr,
              headSize, kvSize, vStrideN, a_zp);
          }
        // Move to the next query
        at::native::data_index_step(i, batchSize, j, num_head);
      }
    });

  // packing
  at::parallel_for(
    0, batchSize * num_head * kvSlice, 1, [&](int64_t begin, int64_t end) {
      int64_t i = 0, j = 0, l = 0, n = 0;
      at::native::data_index_init(
          begin, i, batchSize, j, num_head, l, kvSlice);
      uint8_t* B_blocked_xform_u8 = new uint8_t[std::max(qk_gemm_K, av_gemm_K) * block_64];
      for (const auto z : c10::irange(begin, end)) {
        (void)z; // Suppress unused variable
        n = l * kvSplitSize;
        auto k_reorder = key_reorder_ptr + i * num_head * qk_gemm_K * rndkvSize +
                      j * qk_gemm_K * rndkvSize + n * qk_gemm_K;
        auto v_reorder = value_reorder_ptr +
                      i * num_head * kvSlice * v_reorder_strideL +
                      j * kvSlice * v_reorder_strideL + n * rndHeadSize;
        if (n + kvSplitSize < kvSize) {
          for (int64_t b = 0; b < rndkvSplitSize; b += block_64) {
            do_transpose(
                k_data + i * kStrideB + j * kStrideH + n * kStrideN + b * kStrideN,
                B_blocked_xform_u8,
                std::min(int(kvSplitSize - b), block_64),
                headSize,
                kStrideN,
                block_64);
            if (!headSize_mul4) {
              pad_remain_row_col(
                  B_blocked_xform_u8,
                  headSize,
                  block_64,
                  qk_gemm_K,
                  block_64,
                  block_64
                );
            }
            at::native::cpublas::pack(
                    qk_gemm_K, // K
                    block_64, // N
                    block_64, // ld_in
                    block_64, // ld_out
                    u8_dt, // dt_in
                    u8_dt, // dt_out
                    B_blocked_xform_u8,
                    k_reorder + b * qk_gemm_K);
          }
          // split headSize to block_64, block_64, block_64 ...
          // [kvSplitSize, headSize] -> [av_gemm_K,  block_64 ...]
          for (int64_t b = 0; b < rndHeadSize; b += block_64) {
            at::native::cpublas::pack(
                    av_gemm_K,
                    block_64,
                    vStrideN, // block_64,
                    block_64,
                    u8_dt,
                    u8_dt,
                    v_data + i * vStrideB + j * vStrideH + n * vStrideN + b,
                    v_reorder + av_gemm_K * b);
          }
        } else {
          // tail
          auto block_size = kvTail >= block_64 ? block_64 : kv_tail_tail_block_size;
          int64_t b = 0;
          while (b < rndkvTail) {
            do_transpose(
                k_data + i * kStrideB + j * kStrideH + n * kStrideN + b * kStrideN,
                B_blocked_xform_u8,
                std::min(kvTail - b, block_size),
                headSize,
                kStrideN,
                block_size);
            if (!headSize_mul4) {
              pad_remain_row_col(
                  B_blocked_xform_u8,
                  headSize,
                  block_size,
                  qk_gemm_K,
                  block_size,
                  block_size
                );
            }
            // Pack
            at::native::cpublas::pack(
                    qk_gemm_K,
                    block_size,
                    block_size,
                    block_size,
                    u8_dt,
                    u8_dt,
                    B_blocked_xform_u8,
                    k_reorder + b * qk_gemm_K);
            b += block_size;
            block_size = (kvTail - b) >= block_64 ? block_64 : kv_tail_tail_block_size;
          }
          // split headSize to block_64, block_64, block_64 ...
          // [kvTail, headSize] -> [av_gemm_K_tail,  block_64 ...]
          for (int64_t b = 0; b < headSize; b += block_64) {
            at::native::cpublas::pack(
                    av_gemm_K,
                    block_64,
                    vStrideN, // block_64,
                    block_64,
                    u8_dt,
                    u8_dt,
                    v_data + i * vStrideB + j * vStrideH + n * vStrideN + b,
                    v_reorder + av_gemm_K * b);
          }
        }
        // Move to the next query
        at::native::data_index_step(i, batchSize, j, num_head, l, kvSlice);
      }
    });

  at::parallel_for(
      0, batchSize * num_head * qSlice, 1, [&](int64_t begin, int64_t end) {
        int64_t i = 0, j = 0, k = 0;
        at::native::data_index_init(
            begin, i, batchSize, j, num_head, k, qSlice);
        int ompIdx = at::get_thread_num();
        scalar_t* total_buf_ptr = total_buf_data + ompIdx * total_size_uint8_per_thread;
        int32_t offset = 0;
        accum_t* qk_data = reinterpret_cast<accum_t*>(total_buf_ptr);
        offset += kvSlice * qSplitSize * rndkvSplitSize * 4;
        accum_t* qk_local_data = reinterpret_cast<accum_t*>(total_buf_ptr + offset);
        offset += kvSlice * av_gemm_K * 4;
        scalar_t* qk_reduced_data = reinterpret_cast<scalar_t*>(total_buf_ptr + offset);
        offset += kvSlice * qk_reduce_strideL;
        int32_t* qk_s32_data = reinterpret_cast<int32_t*>(total_buf_ptr + offset);
        offset += qSplitSize * rndkvSplitSize * 4;
        int32_t* dst_s32_data = reinterpret_cast<int32_t*>(total_buf_ptr + offset);
        offset += qSplitSize * rndHeadSize * 4;
        accum_t* sfm_sum_ptr = reinterpret_cast<accum_t*>(total_buf_ptr + offset);
        offset += qSplitSize * 4;
        int32_t* q_sum_ptr = reinterpret_cast<int32_t*>(total_buf_ptr + offset);
        offset += qSplitSize * 4;
        int32_t* a_sum_ptr = reinterpret_cast<int32_t*>(total_buf_ptr + offset);
        offset += qSplitSize * 4;
        accum_t* sfm_max_ptr = reinterpret_cast<accum_t*>(total_buf_ptr + offset);
        offset += qSplitSize * 4;
        scalar_t* query_t_padding_ptr = reinterpret_cast<scalar_t*>(total_buf_ptr + offset);

        for (const auto z : c10::irange(begin, end)) {
          (void)z; // Suppress unused variable

          int32_t* kv_sum_ptr = kv_sum_buf_data
              + i * num_head * kv_sum_size_per_BH
              + j * kv_sum_size_per_BH;
          int32_t* k_sum_ptr = kv_sum_ptr;
          int32_t* v_sum_ptr = kv_sum_ptr + kvSize;

          // sdpa core
          int64_t m = k * qSplitSize;
          int64_t qBlockSize = std::min(qSplitSize, qSize - m);
          // Initialize sum and max
          fill_stub(
              sfm_sum_ptr, static_cast<accum_t>(0), qSplitSize);
          fill_stub(
              a_sum_ptr, static_cast<int32_t>(0), qSplitSize);
          fill_stub(
              sfm_max_ptr, static_cast<accum_t>(-std::numeric_limits<accum_t>::infinity()), qSplitSize);
          int64_t num_keys =
              is_causal ? std::min(m + qBlockSize, kvSize) : kvSize;
          copy_value_with_pad(
              q_data + i * qStrideB + j * qStrideH + m * qStrideM,
              query_t_padding_ptr,
              qBlockSize,
              headSize,
              qBlockSize,
              qk_gemm_K,
              qStrideM);

          if (k_zp != 0) {
            _int_sum_b_contiguous_kernel(q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                  q_sum_ptr, qBlockSize, headSize, qStrideM, k_zp);
          } else {
            fill_stub(
              q_sum_ptr, static_cast<int32_t>(0), qSplitSize);
          }
          const int64_t rkvSlice = (num_keys - 1) / kvSplitSize + 1;
          for (int64_t l = 0; l < rkvSlice; l++) {
            int64_t n = l * kvSplitSize;
            int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
            auto k_reorder = key_reorder_ptr + i * num_head * qk_gemm_K * rndkvSize +
                      j * qk_gemm_K * rndkvSize + n * qk_gemm_K;
            // Calculate sums for dequant compensation item
            if (qBlockSize == qSplitSize) {
              // q main
              if (n + kvSplitSize < kvSize) {
                // k main
                for (int64_t b = 0; b < kvSplitSize; b += block_64) {
                  at::native::cpublas::brgemm(
                        qSplitSize, block_64, qk_gemm_K,
                        qk_gemm_K, // lda
                        block_64, //ldb
                        rndkvSplitSize, //ldc,
                        false,
                        query_t_padding_ptr,
                        k_reorder + b * qk_gemm_K,
                        qk_s32_data + b);
                }
              } else {
                auto block_size = kvTail >= block_64 ? block_64 : kv_tail_tail_block_size;
                int64_t b = 0;
                while (b < kvTail) {
                  at::native::cpublas::brgemm(
                        qSplitSize, block_size, qk_gemm_K,
                        qk_gemm_K, // lda
                        block_size, //ldb
                        rndkvTail, //ldc
                        false,
                        query_t_padding_ptr,
                        k_reorder + b * qk_gemm_K,
                        qk_s32_data + b);
                  b += block_size;
                  block_size = (kvTail - b) >= block_64 ? block_64 : kv_tail_tail_block_size;
                }
              }
            } else {
              if (n + kvSplitSize < kvSize) {
                for (int64_t b = 0; b < kvSplitSize; b += block_64) {
                  at::native::cpublas::brgemm(
                        qTail, block_64, qk_gemm_K,
                        qk_gemm_K,// lda
                        block_64, //ldb
                        rndkvSplitSize, //ldc
                        false,
                        query_t_padding_ptr,
                        k_reorder + b * qk_gemm_K,
                        qk_s32_data + b);
                }
              } else {
                // k tail
                auto block_size = kvTail >= block_64 ? block_64 : kv_tail_tail_block_size;
                int64_t b = 0;
                while (b < kvTail) {
                  at::native::cpublas::brgemm(
                        qTail, block_size, qk_gemm_K,
                        qk_gemm_K, // lda
                        block_size, //ldb
                        rndkvTail, //ldc
                        false,
                        query_t_padding_ptr,
                        k_reorder + b * qk_gemm_K,
                        qk_s32_data + b);
                  b += block_size;
                  block_size = (kvTail - b) >= block_64 ? block_64 : kv_tail_tail_block_size;
                }
              }
            }

            // do dequant compensation, add mask, max reduce for softmax, and convert qk from s32 to fp32
            int64_t rndkvBlockSize = kvBlockSize == kvSplitSize ? rndkvSplitSize : rndkvTail;
            accum_t* qk_block_data = qk_data + l * qSplitSize * rndkvSplitSize;
            if (has_attn_mask) {
              mask_t* mask_data_offset = mask_data + i * mStrideB + j * mStrideH + m * mStrideM + (mStrideN == 0 ? 0 : n);
              _dequant_mask_max_fusion_kernel(
                qk_s32_data, //in
                mask_data_offset, //mask_ptr
                q_sum_ptr, //sum_a_ptr
                k_sum_ptr + n, //sum_b_ptr
                qBlockSize, //M
                kvBlockSize, //N
                rndkvBlockSize, //ldi
                mStrideM, //ldm
                rndkvSplitSize,//kvBlockSize, //ldo
                q_zp * k_zp * headSize, //zp_a*zp_b*k=beta
                q_scale * k_scale * scaling_factor, //scale_a*scale_b*scale_sdpa=alpha
                qk_block_data, //out
                sfm_max_ptr // sfm_max_ptr
              );
            } else {
              _dequant_max_fusion_kernel(
                qk_s32_data, //in
                q_sum_ptr, //sum_a_ptr
                k_sum_ptr + n, //sum_b_ptr
                qBlockSize, //M
                kvBlockSize, //N
                rndkvBlockSize, //ldi
                rndkvSplitSize,//kvBlockSize, //ldo
                q_zp * k_zp * headSize, //zp_a*zp_b*k=beta
                q_scale * k_scale * scaling_factor, //scale_a*scale_b*scale_sdpa=alpha
                qk_block_data, //out
                sfm_max_ptr // sfm_max_ptr
              );
            }
          }
          // sub max, exp, sum reduce, div sum for softmax
          // and quant
          // and sum for attention
          if (v_zp == 0) {
            _sub_exp_sum_div_quant_fusion_kernel(
              qk_data, //in
              qBlockSize, //M
              kvSplitSize, //N_step
              rkvSlice, //NSlices
              qSplitSize * rndkvSplitSize, //ldi
              qk_reduce_strideL, //ldo
              kvSize, //kvSize
              rndkvSplitSize, //rndkvSplitSize
              av_gemm_K, //av_gemm_K
              a_zp, // zp_a=beta1
              a_scale, // scale_a=alpha
              qk_local_data, //local
              qk_reduced_data, //out
              sfm_max_ptr, //sfm_max_ptr
              sfm_sum_ptr //sfm_sum_ptr
            );
          } else {
            _sub_exp_sum_div_quant_sum_fusion_kernel(
              qk_data, //in
              qBlockSize, //M
              kvSplitSize, //N_step
              rkvSlice, //NSlice
              qSplitSize * rndkvSplitSize, //ldi
              qk_reduce_strideL, //ldo
              kvSize, //kvSize
              rndkvSplitSize, //rndkvSplitSize
              av_gemm_K, //av_gemm_K
              a_zp, // zp_a=beta1
              v_zp, // zp_b=beta2
              a_scale, // scale_a=alpha
              qk_local_data, //local
              qk_reduced_data, //out
              sfm_max_ptr, //sfm_max_ptr
              sfm_sum_ptr, //sfm_sum_ptr
              a_sum_ptr //a_sum_ptr
            );
          }
          // Calculate Softmax(q @ k.T) @ v
          auto v_reorder = value_reorder_ptr +
                  i * num_head * kvSlice * v_reorder_strideL +
                  j * kvSlice * v_reorder_strideL;
          for (int64_t b = 0; b < headSize; b += block_64) {
            auto value_reorder_b = v_reorder + b * av_gemm_K;
            auto dst_s32_b = dst_s32_data + b;
            for (int64_t s = 0; s < kvSlice; s++) {
              at::native::cpublas::brgemm(
                  qSplitSize, block_64, av_gemm_K,
                  av_gemm_K, // lda
                  rndHeadSize, //block_64, //ldb
                  rndHeadSize, //ldc
                  s != 0,
                  qk_reduced_data + s * qk_reduce_strideL,
                  value_reorder_b + s * v_reorder_strideL,
                  dst_s32_b);
            }
          }

          // After the last gemm,
          // do dequant compensation, quant and convert from s32 to int8
          _dequant_quant_fusion_kernel(
            dst_s32_data, //in
            a_sum_ptr, //sum_a_ptr
            v_sum_ptr, //sum_b_ptr
            qBlockSize, //M
            headSize, //N
            rndHeadSize, //ldi
            oStrideM, //ldo
            a_zp * v_zp * kvSize, //zp_a*zp_b*k=beta1
            o_zp, //zp_c=beta2
            a_scale * v_scale / o_scale, //scale_a*scale_b/scale_c=alpha
            out_data + i * oStrideB + j * oStrideH + m * oStrideM //out
          );
          // Move to the next query
          at::native::data_index_step(i, batchSize, j, num_head, k, qSlice);
        }
      });
  // Once all computations are done, need to release HW context.
  at::native::cpublas::brgemm_release();
}

#define AT_DISPATCH_MASK_TYPES(TYPE, NAME, ...)            \
  AT_DISPATCH_SWITCH(                                      \
      TYPE,                                                \
      NAME,                                                \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                     \
          at::ScalarType::Bool, mask_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                     \
          at::ScalarType::Float, mask_t, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                     \
          at::ScalarType::Double, mask_t, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                     \
          at::ScalarType::BFloat16, mask_t, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                     \
          at::ScalarType::Half, mask_t, __VA_ARGS__))

void sdpa_int8_fused_kernel(
    at::Tensor& output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    std::optional<Tensor> attn_mask,
    std::optional<double> scale,
    double dropout_p,
    bool is_causal,
    int64_t q_zp,
    double q_scale,
    int64_t k_zp,
    double k_scale,
    int64_t v_zp,
    double v_scale,
    int64_t a_zp,
    double a_scale,
    int64_t o_zp,
    double o_scale) {
  TORCH_CHECK(query.scalar_type() == c10::kByte);
  int64_t batchSize = query.size(0);
  int64_t num_head = query.size(1);
  int64_t q_seq_len = query.size(2);
  int64_t kv_seq_len = key.size(2);
  int64_t q_split_size = 32;
  if (q_seq_len >= 768) {
    q_split_size = 256;
  } else if (q_seq_len >= 192) {
    q_split_size = 64;
  }
  // Heuristic to decide whether to use one parallel loop or not
  uint32_t l2_cache_size = at::cpu::L2_cache_size();
  int64_t num_thread = at::get_num_threads();
  int64_t attn_size = q_split_size * kv_seq_len * sizeof(int32_t) * num_thread;
  bool use_one_parallel_loop = (batchSize * num_head > num_thread) &&
      (attn_size > l2_cache_size);
  if (use_one_parallel_loop) {
    if (!attn_mask.has_value()) {
      if (q_split_size == 256) {
        sdpa_int8_kernel_one_loop_impl<unsigned char, float, 256, 64>(
          output, query, key, value,
          attn_mask, scale, dropout_p, is_causal,
          q_zp, q_scale,
          k_zp, k_scale,
          v_zp, v_scale,
          a_zp, a_scale,
          o_zp, o_scale);
      } else if (q_split_size == 64) {
        sdpa_int8_kernel_one_loop_impl<unsigned char, float, 64, 64>(
          output, query, key, value,
          attn_mask, scale, dropout_p, is_causal,
          q_zp, q_scale,
          k_zp, k_scale,
          v_zp, v_scale,
          a_zp, a_scale,
          o_zp, o_scale);
      } else {
        sdpa_int8_kernel_one_loop_impl<unsigned char, float, 32, 64>(
          output, query, key, value,
          attn_mask, scale, dropout_p, is_causal,
          q_zp, q_scale,
          k_zp, k_scale,
          v_zp, v_scale,
          a_zp, a_scale,
          o_zp, o_scale);
      }
    } else {
      AT_DISPATCH_MASK_TYPES(attn_mask.value().scalar_type(), "sdpa_mask", [&]() {
        if (q_split_size == 256) {
          sdpa_int8_kernel_one_loop_impl<unsigned char, mask_t, 256, 64>(
            output, query, key, value,
            attn_mask, scale, dropout_p, is_causal,
            q_zp, q_scale,
            k_zp, k_scale,
            v_zp, v_scale,
            a_zp, a_scale,
            o_zp, o_scale);
        } else if (q_split_size == 64) {
          sdpa_int8_kernel_one_loop_impl<unsigned char, mask_t, 64, 64>(
            output, query, key, value,
            attn_mask, scale, dropout_p, is_causal,
            q_zp, q_scale,
            k_zp, k_scale,
            v_zp, v_scale,
            a_zp, a_scale,
            o_zp, o_scale);
        } else {
          sdpa_int8_kernel_one_loop_impl<unsigned char, mask_t, 32, 64>(
            output, query, key, value,
            attn_mask, scale, dropout_p, is_causal,
            q_zp, q_scale,
            k_zp, k_scale,
            v_zp, v_scale,
            a_zp, a_scale,
            o_zp, o_scale);
        }
      });
    }
  } else {
    if (!attn_mask.has_value()) {
      if (q_split_size == 256) {
        sdpa_int8_kernel_several_loops_impl<unsigned char, float, 256, 64>(
          output, query, key, value,
          attn_mask, scale, dropout_p, is_causal,
          q_zp, q_scale,
          k_zp, k_scale,
          v_zp, v_scale,
          a_zp, a_scale,
          o_zp, o_scale);
      } else if (q_split_size == 64) {
        sdpa_int8_kernel_several_loops_impl<unsigned char, float, 64, 64>(
          output, query, key, value,
          attn_mask, scale, dropout_p, is_causal,
          q_zp, q_scale,
          k_zp, k_scale,
          v_zp, v_scale,
          a_zp, a_scale,
          o_zp, o_scale);
      } else {
        sdpa_int8_kernel_several_loops_impl<unsigned char, float, 32, 64>(
          output, query, key, value,
          attn_mask, scale, dropout_p, is_causal,
          q_zp, q_scale,
          k_zp, k_scale,
          v_zp, v_scale,
          a_zp, a_scale,
          o_zp, o_scale);
      }
    } else {
      AT_DISPATCH_MASK_TYPES(attn_mask.value().scalar_type(), "sdpa_mask", [&]() {
        if (q_split_size == 256) {
          sdpa_int8_kernel_several_loops_impl<unsigned char, mask_t, 256, 64>(
            output, query, key, value,
            attn_mask, scale, dropout_p, is_causal,
            q_zp, q_scale,
            k_zp, k_scale,
            v_zp, v_scale,
            a_zp, a_scale,
            o_zp, o_scale);
        } else if (q_split_size == 64) {
          sdpa_int8_kernel_several_loops_impl<unsigned char, mask_t, 64, 64>(
            output, query, key, value,
            attn_mask, scale, dropout_p, is_causal,
            q_zp, q_scale,
            k_zp, k_scale,
            v_zp, v_scale,
            a_zp, a_scale,
            o_zp, o_scale);
        } else {
          sdpa_int8_kernel_several_loops_impl<unsigned char, mask_t, 32, 64>(
            output, query, key, value,
            attn_mask, scale, dropout_p, is_causal,
            q_zp, q_scale,
            k_zp, k_scale,
            v_zp, v_scale,
            a_zp, a_scale,
            o_zp, o_scale);
        }
      });
    }
  }
}
// #endif // CPU_CAPABILITY_AVX512

// at::Tensor sdpa_int8_math_kernel(
//     const at::Tensor& query_,
//     const at::Tensor& key,
//     const at::Tensor& value,
//     double dropout_p,
//     bool is_causal,
//     c10::optional<at::Tensor> attn_mask_,
//     c10::optional<double> scale,
//     int32_t q_zp,
//     float q_scale,
//     int32_t k_zp,
//     float k_scale,
//     int32_t v_zp,
//     float v_scale,
//     int32_t a_zp,
//     float a_scale,
//     int32_t o_zp,
//     float o_scale) {
//   // dequant q/k/v
//   auto q = (query_.to(at::kFloat) - q_zp) * q_scale;
//   auto k = (key.to(at::kFloat) - k_zp) * k_scale;
//   auto v = (value.to(at::kFloat) - v_zp) * v_scale;
//   auto attn_mask = attn_mask_;
//   if (attn_mask.has_value()) {
//     *attn_mask = (*attn_mask).to(at::kFloat);
//   }
//   // Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
//   bool is_negative_scaling = scale.has_value() && scale.value() < 0.0;
//   const auto scaling_factor = sdp::calculate_scale(q, is_negative_scaling ? std::abs(scale.value()) : scale).sqrt();
//   q = q * (is_negative_scaling ? c10::SymFloat(0.0) - scaling_factor: scaling_factor);
//   auto attn = at::matmul(q, k.transpose(-2, -1) * scaling_factor);
//   if (attn_mask.has_value()) {
//     if (at::areAnyTensorSubclassLike({attn, *attn_mask})) {
//       attn = attn.add(*attn_mask);
//     } else {
//       attn.add_(*attn_mask);
//     }
//   }
//   attn = at::softmax(attn, -1);
//   // quant attn
//   attn = at::clamp_max(
//       at::clamp_min(at::round(attn / a_scale) + a_zp, 0), 255
//   );
//   // dequant attn
//   attn = (attn - a_zp) * a_scale;
//   auto output = at::matmul(attn, v);
//   // quant output
//   output = at::clamp_max(
//       at::clamp_min(at::round(output / o_scale) + o_zp, 0), 255
//   ).to(at::kByte);
//   return output;
// }

// void _scaled_dot_product_int8_cpu(
//     const Tensor& output,
//     const Tensor& query,
//     const Tensor& key,
//     const Tensor& value,
//     const std::optional<Tensor>& attn_mask,
//     std::optional<double> scale,
//     double dropout_p,
//     bool is_causal,
//     int64_t q_zp,
//     double q_scale,
//     int64_t k_zp,
//     double k_scale,
//     int64_t v_zp,
//     double v_scale,
//     int64_t a_zp,
//     double a_scale,
//     int64_t o_zp,
//     double o_scale) {
//   // fallback math path
//   // at::Tensor output = sdpa_int8_math_impl(query, key, value,
//   //   attn_mask, scale, dropout_p, is_causal,
//   //   q_zp, q_scale,
//   //   k_zp, k_scale,
//   //   v_zp, v_scale,
//   //   a_zp, a_scale,
//   //   o_zp, o_scale).transpose(1, 2).contiguous().transpose(1, 2);

//   // TODO @Valentine233: add flash attention int8 impl

//   // #ifdef CPU_CAPABILITY_AVX512
//   if (at::native::cpublas::could_pack(dtype)) {
//     std::cout << "CPU_CAPABILITY_AVX512 fused" << std::endl;  
//     at::Tensor output = at::empty_like(query, query.options()).transpose(1, 2);
//       sdpa_int8_fused_kernel(output, query, key, value,
//           dropout_p, is_causal, attn_mask, scale,
//           q_zp, q_scale,
//           k_zp, k_scale,
//           v_zp, v_scale,
//           a_zp, a_scale,
//           o_zp, o_scale);
//       return output.transpose(1, 2);
//   } else {
//     std::cout << "CPU_CAPABILITY_AVX512 math" << std::endl;  
//     return sdpa_int8_math_kernel(query, key, value,
//           dropout_p, is_causal, attn_mask, scale,
//           q_zp, q_scale,
//           k_zp, k_scale,
//           v_zp, v_scale,
//           a_zp, a_scale,
//           o_zp, o_scale).transpose(1, 2).contiguous().transpose(1, 2);
//   }
//   // #else
//   //   std::cout << "CPU_CAPABILITY_AVX2 math" << std::endl;  
//   //   return sdpa_int8_math_kernel(query, key, value,
//   //       dropout_p, is_causal, attn_mask, scale,
//   //       q_zp, q_scale,
//   //       k_zp, k_scale,
//   //       v_zp, v_scale,
//   //       a_zp, a_scale,
//   //       o_zp, o_scale).transpose(1, 2).contiguous().transpose(1, 2);
//   // #endif // CPU_CAPABILITY_AVX512

// }

} // anonymous namespace
ALSO_REGISTER_AVX512_DISPATCH(sdpa_int8_kernel, &sdpa_int8_fused_kernel)

} // namespace at::native
