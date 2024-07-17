#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <c10/util/irange.h>
#include <oneapi/dnnl/dnnl_ukernel.hpp>
#include <iostream>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

using dt = dnnl::memory::data_type;

struct BrgemmKey {
  int64_t M;
  int64_t N;
  int64_t K;
  int64_t batch_size;
  int lda;
  int ldb;
  int ldc;
  dt dt_a;
  dt dt_b;
  dt dt_c;
  float alpha;
  float beta;
  BrgemmKey(int64_t M,
    int64_t N,
    int64_t K,
    int64_t batch_size,
    int lda,
    int ldb,
    int ldc,
    dt dt_a,
    dt dt_b,
    dt dt_c,
    float alpha,
    float beta)
      : M(M), N(N), K(K), batch_size(batch_size), lda(lda), ldb(ldb), ldc(ldc), dt_a(dt_a), dt_b(dt_b),
      dt_c(dt_c), alpha(alpha), beta(beta) {}
  bool operator==(const BrgemmKey& other) const {
    return M == other.M && N == other.N && K == other.K && batch_size == other.batch_size &&
        lda == other.lda && ldb == other.ldb && ldc == other.ldc &&
        dt_a == other.dt_a && dt_b == other.dt_b && dt_c == other.dt_c &&
        alpha == other.alpha && beta == other.beta;
  }
};

struct PackBKey {
  int64_t K;
  int64_t N;
  int ld_in;
  int ld_out;
  dt dt_in;
  dt dt_out;
  PackBKey(
    int64_t K,
    int64_t N,
    int ld_in,
    int ld_out,
    dt dt_in,
    dt dt_out)
      : K(K), N(N), ld_in(ld_in), ld_out(ld_out), dt_in(dt_in), dt_out(dt_out) {}
  bool operator==(const PackBKey& other) const {
    return N == other.N && K == other.K &&
        ld_in == other.ld_in && ld_out == other.ld_out &&
        dt_in == other.dt_in && dt_out == other.dt_out;
  }
};

namespace std {
template <>
struct hash<BrgemmKey> {
  std::size_t operator()(const BrgemmKey& key) const {
    std::size_t h = std::hash<bool>()(key.alpha);
    h = std::hash<bool>()(key.beta) ^ (h << 1);
    h = std::hash<int>()(key.batch_size) ^ (h << 1);
    // h = std::hash<int>()(key.dt_a) ^ (h << 1);
    // h = std::hash<int>()(key.dt_b) ^ (h << 1);
    // h = std::hash<int>()(key.dt_c) ^ (h << 1);
    h = std::hash<int>()(key.lda) ^ (h << 1);
    h = std::hash<int>()(key.ldb) ^ (h << 1);
    h = std::hash<int>()(key.ldc) ^ (h << 1);
    h = std::hash<int>()(key.M) ^ (h << 1);
    h = std::hash<int>()(key.N) ^ (h << 1);
    h = std::hash<int>()(key.K) ^ (h << 1);
    return h;
  }
};

template <>
struct hash<PackBKey> {
  std::size_t operator()(const PackBKey& key) const {
    std::size_t h = std::hash<int>()(key.K);
    h = std::hash<int>()(key.N) ^ (h << 1);
    h = std::hash<int>()(key.ld_in) ^ (h << 1);
    h = std::hash<int>()(key.ld_out) ^ (h << 1);
    return h;
  }
};
} // namespace std

namespace at::native {

namespace {

// out = val * a + b
// is_b_stride_zero: If the stride of b is 0 (mask broadcasting case),
//                take b as a scalar pointer.
template <bool is_b_stride_zero, typename T1, typename T2>
inline void _scale_attn_mask_fusion_kernel(
    T1* a,
    T2* b,
    const int& size,
    T1* out,
    T1& val) {
  const auto vec_size1 = at::vec::Vectorized<T1>::size();
  const auto vec_size2 = at::vec::Vectorized<T2>::size();
  constexpr int64_t T1_n =
      (vec_size2 == vec_size1 * 2 && is_reduced_floating_point_v<T2>) ? 2 : 1;
  constexpr int64_t T2_n = 1;
  auto vec_scale = at::vec::VectorizedN<T1, T1_n>(val);
  int64_t i = 0;
  for (; i < size - (size % vec_size2); i += vec_size2) {
    auto a_n = at::vec::VectorizedN<T1, T1_n>::loadu(a + i);
    at::vec::VectorizedN<T2, T2_n> b_n;
    if constexpr(is_b_stride_zero) {
      b_n = at::vec::VectorizedN<T2, T2_n>((T1)b[0]);
    } else {
      b_n = at::vec::VectorizedN<T2, T2_n>::loadu(b + i);
    }
    auto b_n_convert = at::vec::convert<T1, T1_n, T2, T2_n, true>(b_n);
    auto res = a_n * vec_scale + b_n_convert;
    res.store(out + i);
  }
  for (; i < size; i++) {
    auto tmp0 = a[i];
    T1 tmp1;
    if constexpr(is_b_stride_zero) {
      tmp1 = (T1)b[0];
    } else {
      tmp1 = (T1)b[i];
    }
    out[i] = tmp0 * val + tmp1;
  }
}

// 1) out = exp(a - val)
// 2) val = sum(out)
template <typename T1, typename T2>
inline void _exp_reduce_sum_fusion_kernel(
    T1* a,
    const int& size,
    T2* out,
    T1& val) {
  auto vec_size = vec::Vectorized<T1>::size();
  auto vec_max = vec::Vectorized<T1>(val);
  T1 tmp_sum = 0;
  auto vec_tmp_sum = vec::Vectorized<T1>(tmp_sum);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = vec::Vectorized<T1>::loadu(a + i);
    auto tmp1 = tmp0 - vec_max;
    auto tmp2 = tmp1.exp_u20();
    vec_tmp_sum += tmp2;
    _store(out + i, tmp2);
  }
  tmp_sum = vec::vec_reduce_all<T1>(
      [](vec::Vectorized<T1>& x, vec::Vectorized<T1>& y) {
        return x + y;
      },
      vec_tmp_sum);
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 - val;
    auto tmp2 = exp(tmp1);
    tmp_sum += tmp2;
    out[i] = tmp2;
  }
  val = tmp_sum;
}

// 1) out = a * scale
// 2) max = max(out)
template <typename scalar_t>
inline void _mul_reduce_max_fusion_kernel(
    const scalar_t* a,
    const scalar_t& scale,
    const int& size,
    scalar_t* out,
    scalar_t& max) {
  auto vec_size = vec::Vectorized<scalar_t>::size();
  auto vec_scale = vec::Vectorized<scalar_t>(scale);
  scalar_t tmp_max = -std::numeric_limits<scalar_t>::infinity();
  auto vec_tmp_max = vec::Vectorized<scalar_t>(tmp_max);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = vec::Vectorized<scalar_t>::loadu(a + i);
    auto tmp1 = tmp0 * vec_scale;
    vec_tmp_max = vec::maximum(vec_tmp_max, tmp1);
    _store(out + i, tmp1);
  }
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 * scale;
    tmp_max = std::max(tmp_max, tmp1);
    out[i] = tmp1;
  }
  max = std::max(
      tmp_max,
      vec::vec_reduce_all<scalar_t>(
          [](vec::Vectorized<scalar_t>& x, vec::Vectorized<scalar_t>& y) {
            return vec::maximum(x, y);
          },
          vec_tmp_max));
}

// 1) out = a * scale
// 2) out = out + mask
// 3) max = max(out)
template <typename scalar_t>
inline void _mul_add_reduce_max_fusion_kernel(
    const scalar_t* a,
    const scalar_t* mask,
    const scalar_t& scale,
    const int& size,
    scalar_t* out,
    scalar_t& max) {
  auto vec_size = vec::Vectorized<scalar_t>::size();
  auto vec_scale = vec::Vectorized<scalar_t>(scale);
  scalar_t tmp_max = -std::numeric_limits<scalar_t>::infinity();
  auto vec_tmp_max = vec::Vectorized<scalar_t>(tmp_max);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = vec::Vectorized<scalar_t>::loadu(a + i);
    auto tmp1 = tmp0 * vec_scale;
    auto tmp2 = vec::Vectorized<scalar_t>::loadu(mask + i);
    auto tmp3 = tmp1 + tmp2;
    vec_tmp_max = vec::maximum(vec_tmp_max, tmp3);
    _store(out + i, tmp3);
  }
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 * scale;
    auto tmp2 = mask[i];
    auto tmp3 = tmp1 + tmp2;
    tmp_max = std::max(tmp_max, tmp3);
    out[i] = tmp3;
  }
  max = std::max(
      tmp_max,
      vec::vec_reduce_all<scalar_t>(
          [](vec::Vectorized<scalar_t>& x, vec::Vectorized<scalar_t>& y) {
            return vec::maximum(x, y);
          },
          vec_tmp_max));
}

template <typename scalar_t>
static inline scalar_t* conditional_data_ptr(scalar_t* ptr, scalar_t* ptr2) {
  TORCH_CHECK(ptr2 == nullptr);
  return ptr;
}

template <typename scalar_t,
          typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
static inline scalar_t* conditional_data_ptr(float* ptr, scalar_t* ptr2) {
  return ptr2;
}

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
    Tensor& attn_mask,
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

template <typename scalar_t>
inline void pad_row_zero(
    scalar_t* value_ptr,
    scalar_t* padding_value_ptr,
    int rows,
    int cols,
    int ldi) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  int i = 0;
  for (; i < rows - 1; i++) {
    int j = 0;
    for (; j < cols - (cols % vec_size); j += vec_size) {
      auto vec_v =
          at::vec::Vectorized<scalar_t>::loadu(value_ptr + i * ldi + j);
      vec_v.store(padding_value_ptr + i * cols + j);
    }

    if (j < cols) {
      auto vec_v = at::vec::Vectorized<scalar_t>::loadu(
          value_ptr + i * ldi + j, cols - j);
      vec_v.store(padding_value_ptr + i * cols + j, cols - j);
    }
  }

  // zero padding
  int j = 0;
  for (; j < cols - (cols % vec_size); j += vec_size) {
    auto vec_v = at::vec::Vectorized<scalar_t>(0);
    vec_v.store(padding_value_ptr + i * cols + j);
  }

  if (j < cols) {
    auto vec_v = at::vec::Vectorized<scalar_t>(0);
    vec_v.store(padding_value_ptr + i * cols + j, cols - j);
  }
}

template <typename scalar_t>
inline void pad_row_128_padding(
    scalar_t* value_ptr,
    scalar_t* padding_value_ptr,
    int rows,
    int cols,
    int ldi,
    int padding) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  int i = 0;
  for (; i < rows - padding; i++) {
    int j = 0;
    for (; j < cols - (cols % vec_size); j += vec_size) {
      auto vec_v =
          at::vec::Vectorized<scalar_t>::loadu(value_ptr + i * ldi + j);
      vec_v.store(padding_value_ptr + i * cols + j);
    }

    if (j < cols) {
      auto vec_v = at::vec::Vectorized<scalar_t>::loadu(
          value_ptr + i * ldi + j, cols - j);
      vec_v.store(padding_value_ptr + i * cols + j, cols - j);
    }
  }

  // 128 padding
  for (; i < rows; i++) {
    int j = 0;
    for (; j < cols - (cols % vec_size); j += vec_size) {
      auto vec_v = at::vec::Vectorized<scalar_t>(128);
      vec_v.store(padding_value_ptr + i * cols + j);
    }

    if (j < cols) {
      auto vec_v = at::vec::Vectorized<scalar_t>(128);
      vec_v.store(padding_value_ptr + i * cols + j, cols - j);
    }
  }
}

template <typename scalar_t>
inline void pad_col_zero(
    scalar_t* value_ptr,
    scalar_t* padding_value_ptr,
    int rows,
    int cols,
    int ldi) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  for (int i = 0; i < rows; i++) {
    int j = 0;
    for (; j < cols - 1 - ((cols - 1) % vec_size); j += vec_size) {
      auto vec_v =
          at::vec::Vectorized<scalar_t>::loadu(value_ptr + i * ldi + j);
      vec_v.store(padding_value_ptr + i * cols + j);
    }
    if (j < cols - 1) {
      auto vec_v = at::vec::Vectorized<scalar_t>::loadu(
          value_ptr + i * ldi + j, cols - 1 - j);
      vec_v.store(padding_value_ptr + i * cols + j, cols - 1 - j);
      *(padding_value_ptr + i * cols + cols - 1) = scalar_t(0);
    }
  }
}

template <typename scalar_t>
inline void pad_col_zero_padding(
    scalar_t* value_ptr,
    scalar_t* padding_value_ptr,
    int rows,
    int cols,
    int ldi,
    int padding) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  for (int i = 0; i < rows; i++) {
    int j = 0;
    for (; j < cols - padding - ((cols - padding) % vec_size); j += vec_size) {
      auto vec_v =
          at::vec::Vectorized<scalar_t>::loadu(value_ptr + i * ldi + j);
      vec_v.store(padding_value_ptr + i * cols + j);
    }
    if (j < cols - padding) {
      auto vec_v = at::vec::Vectorized<scalar_t>::loadu(
          value_ptr + i * ldi + j, cols - padding - j);
      vec_v.store(padding_value_ptr + i * cols + j, cols - padding - j);
      *(padding_value_ptr + i * cols + cols - padding) = scalar_t(0);
    }
  }
}

// 1) out = a - max
// 2) out = exp(out)
// 3) sum = sum(out)
template <typename scalar_t>
inline void _sub_exp_sum_fusion_kernel(
    const scalar_t* a,
    const int& size,
    scalar_t* out,
    scalar_t& max,
    scalar_t& sum) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  auto vec_max = at::vec::Vectorized<scalar_t>(max);
  scalar_t tmp_sum = 0;
  auto vec_tmp_sum = at::vec::Vectorized<scalar_t>(tmp_sum);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(a + i);
    auto tmp1 = tmp0 - vec_max;
    auto tmp2 = tmp1.exp_u20();
    vec_tmp_sum = vec_tmp_sum + tmp2;
    _store(out + i, tmp2);
  }
  tmp_sum += at::vec::vec_reduce_all<scalar_t>(
        [](at::vec::Vectorized<scalar_t>& x, at::vec::Vectorized<scalar_t>& y) {
          return x + y;
        },
        vec_tmp_sum);
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 - max;
    auto tmp2 = exp(tmp1);
    tmp_sum += tmp2;
    out[i] = tmp2;
  }
  sum += tmp_sum;
}

// The input is ab format contiguous on b
// Do float sum on b dim and output size is M
template <typename scalar_t>
inline void _sum_b_contiguous_kernel(
    const scalar_t* in,
    float* out,
    const int& M,
    const int& N,
    const int& ld,
    const bool& beta) {
  auto vec_size = at::vec::Vectorized<float>::size();
  for (long r = 0; r < M; r += 1) {
    const scalar_t* tmp_in = in + r * ld;
    float tmp_sum = 0;
    auto vec_tmp_sum = at::vec::Vectorized<float>(tmp_sum);
    for (long i = 0; i < vec_size * (N / vec_size); i += vec_size) {
      auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(tmp_in + i);
      auto tmp1 = at::vec::convert<float>(tmp0);
      vec_tmp_sum = vec_tmp_sum + tmp1;
    }
    tmp_sum += at::vec::vec_reduce_all<float>(
        [](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) {
          return x + y;
        },
        vec_tmp_sum);
    for (long i = vec_size * (N / vec_size); i < N; i++) {
    // for (long i = 0; i < N; i++) {
      tmp_sum += (float) tmp_in[i];
    }
    out[r] = beta ? out[r] + tmp_sum : tmp_sum;
  }
}

// The input is ab format contiugous on a
// Do float sum on b dim and output size is M
template <typename scalar_t>
inline void _sum_a_contiguous_kernel(
    const scalar_t* in,
    float* out,
    const int& M,
    const int& N,
    const int& ld) {
  auto vec_size = at::vec::Vectorized<float>::size();
  for (long j = 0; j < N; j++) {
    const scalar_t* tmp_in = in + j * ld;
    for (long i = 0; i < vec_size * (M / vec_size); i += vec_size) {
      auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(tmp_in + i);
      auto tmp1 = at::vec::Vectorized<float>::loadu(out + i);
      auto tmp2 = at::vec::convert<float>(tmp0);
      auto tmp3 = tmp1 + tmp2;
      _store(out + i, tmp3);
    }
    for (long i = vec_size * (M / vec_size); i < M; i++) {
    // for (long i = 0; i < M; i++) {
      auto tmp0 = tmp_in[i];
      auto tmp1 = out[i];
      auto tmp2 = (float) tmp0;
      auto tmp3 = tmp1 + tmp2;
      out[i] = tmp3;
    }
  }
}

/*
  Do dequantization with compensation items:
  s_A * s_B * A.(B - 128)
  - s_A * s_B * zp_A * row_sum(B - 128)
  - s_A * s_B * (zp_B - 128) * col_sum(A)
  + s_A * zp_A * s_B * (zp_B - 128) * K
  with A:u8, B:u8->s8

  do the subtraction of 128
  and convert from int32 to fp32
*/
inline void _dequant_kernel_u8_s8(
    const int32_t* in,
    float* out,
    const float* col_sum_a,
    const float* row_sum_b,
    const int& M,
    const int& N,
    const int32_t& K,
    const int& ld_in,
    const int& ld_out,
    const float& scale_a,
    const float& scale_b,
    const int32_t& zp_a,
    const int32_t& zp_b) {
  auto vec_size = at::vec::Vectorized<float>::size();
  float scale_ab = scale_a * scale_b;
  auto vec_scale_ab = at::vec::Vectorized<float>(scale_ab);
  float k_float = (float) K;
  float k128_float = 128 * k_float;
  float zp_a_float = (float) zp_a;
  float zp_b_float = (float) (zp_b - 128);
  auto vec_zp_a = at::vec::Vectorized<float>(zp_a_float);
  auto vec_zp_b = at::vec::Vectorized<float>(zp_b_float);
  float zp_ab_float = zp_a_float * zp_b_float;
  auto vec_zp_ab = at::vec::Vectorized<float>(zp_ab_float);
  auto vec_k = at::vec::Vectorized<float>(k_float);
  auto vec_k128 = at::vec::Vectorized<float>(k128_float);
  for (long r = 0; r < M; r += 1) {
    auto sum_a = col_sum_a[r];
    auto vec_sum_a = at::vec::Vectorized<float>(sum_a);
    const int32_t* tmp_in = in + r * ld_in;
    float* tmp_out = out + r * ld_out;
    for (long i = 0; i < vec_size * (N / vec_size); i += vec_size) {
      auto vec_sum_b1 = at::vec::Vectorized<float>::loadu(row_sum_b + i);
      auto vec_sum_b2 = vec_sum_b1 - vec_k128;
      auto tmp0 = at::vec::Vectorized<int32_t>::loadu(tmp_in + i);
      auto tmp1 = at::vec::convert<float>(tmp0);
      auto tmp2 = tmp1 - vec_zp_a * vec_sum_b2;
      auto tmp3 = tmp2 - vec_zp_b * vec_sum_a;
      auto tmp4 = tmp3 + vec_zp_ab * vec_k;
      auto tmp5 = tmp4 * vec_scale_ab;
      _store(tmp_out + i, tmp5);
    }
    for (long i = vec_size * (N / vec_size); i < N; i++) {
    // for (long i = 0; i < N; i++) {
      auto sum_b1 = row_sum_b[i];
      auto sum_b2 = sum_b1 - k128_float;
      auto tmp0 = tmp_in[i];
      auto tmp1 = (float) tmp0;
      auto tmp2 = tmp1 - zp_a_float * sum_b2;
      auto tmp3 = tmp2 - zp_b_float * sum_a;
      auto tmp4 = tmp3 + zp_ab_float * k_float;
      auto tmp5 = tmp4 * scale_ab;
      tmp_out[i] = tmp5;
    }
  }
}

/*
  Do quantization with scaling:
  (X / scale + zp) * alpha

  reorder and convert from fp32 to uint8/int8
*/
template <typename scalar_t>
inline void _quant_scale_reorder_kernel(
    const float* in,
    scalar_t* out,
    const float& alpha,
    const int& size,
    const float& scale,
    const int32_t& zp) {
  auto vec_size = at::vec::Vectorized<float>::size();
  auto vec_alpha = at::vec::Vectorized<float>(alpha);
  auto vec_scale = at::vec::Vectorized<float>(scale);
  float zp_float = (float) zp;
  auto vec_zp = at::vec::Vectorized<float>(zp_float);
  float min_val = 0;
  float max_val = 255;
  auto vec_min_val = at::vec::Vectorized<float>(min_val);
  auto vec_max_val = at::vec::Vectorized<float>(max_val);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<float>::loadu(in + i);
    auto tmp1 = tmp0 / vec_scale;
    // auto tmp2 = tmp1.ceil();
    auto tmp2 = tmp1.round();
    auto tmp3 = tmp2 + vec_zp;
    auto tmp4 = at::vec::maximum(tmp3, vec_min_val);
    auto tmp5 = at::vec::minimum(tmp4, vec_max_val);
    auto tmp6 = tmp5 * vec_alpha;
    _store(out + i, tmp6);
  }
  for (long i = vec_size * (size / vec_size); i < size; i++) {
  // for (long i = 0; i < size; i++) {
    auto tmp0 = in[i];
    auto tmp1 = tmp0 / scale;
    // auto tmp2 = std::ceil(tmp1);
    auto tmp2 = std::nearbyint(tmp1);
    auto tmp3 = tmp2 + zp_float;
    auto tmp4 = std::max(tmp3, min_val);
    auto tmp5 = std::min(tmp4, max_val);
    auto tmp6 = tmp5 * alpha;
    out[i] = (scalar_t) tmp6;
  }
}

/*
  Do quantization:
  X / scale + zp
  convert from fp32 to uint8/int8
*/
template <typename scalar_t>
inline void _quant_kernel(
    const float* in,
    scalar_t* out,
    const int& M,
    const int& N,
    const int& ld_in,
    const int& ld_out,
    const float& scale,
    const int32_t& zp) {
  for (long r = 0; r < M; r += 1) {
    _quant_scale_reorder_kernel(
        in + r * ld_in,
        out + r * ld_out,
        1.0,
        N,
        scale,
        zp);
  }
}

inline void do_convert_u8_s8(
    unsigned char* src,
    signed char* dst,
    int64_t in_rows,
    int64_t in_cols,
    int64_t ldi,
    int64_t ldo) {
  auto vec_size = at::vec::Vectorized<int16_t>::size();
  auto vec_128 = at::vec::Vectorized<int16_t>(128);
  for (int64_t r = 0; r < in_rows; r++) {
    const unsigned char* tmp_src = src + r * ldi;
    signed char* tmp_dst = dst + r * ldo;
    for (int64_t c = 0; c < vec_size * (in_cols / vec_size); c += vec_size) {
      auto tmp0 = at::vec::Vectorized<unsigned char>::loadu(tmp_src + c, vec_size);
      auto tmp1 = at::vec::convert<int16_t>(tmp0);
      auto tmp2 = tmp1 - vec_128;
      auto tmp3 = at::vec::convert<signed char>(tmp2);
      _store(tmp_dst + c, tmp3, vec_size);
    }
    for (int64_t c = vec_size * (in_cols / vec_size); c < in_cols; c++) {
    // for (int64_t c = 0; c < in_cols; c++) {
      auto tmp0 = tmp_src[c];
      auto tmp1 = (int16_t) tmp0;
      auto tmp2 = tmp1 - 128;
      auto tmp3 = (signed char) tmp2;
      tmp_dst[c] = tmp3;
    }
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
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
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
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
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

std::unordered_map<
      BrgemmKey,
      std::shared_ptr<dnnl::ukernel::brgemm>> cache_brgemm_kernels;

std::unordered_map<
      PackBKey,
      std::shared_ptr<dnnl::ukernel::brgemm_pack_B>> cache_packb_kernels;

std::shared_ptr<dnnl::ukernel::brgemm> create_or_get_microkernel(
  int64_t M,
  int64_t N,
  int64_t K,
  int64_t batch_size,
  int lda,
  int ldb,
  int ldc,
  dt dt_a,
  dt dt_b,
  dt dt_c,
  float alpha,
  float beta) {
    BrgemmKey key_brgemm(M, N, K, batch_size, lda, ldb, ldc, dt_a, dt_b, dt_c, alpha, beta);
    auto search = cache_brgemm_kernels.find(key_brgemm);
    if (search != cache_brgemm_kernels.end()) {
      return search->second;
    } else {
      cache_brgemm_kernels.insert(
          {key_brgemm,
          std::make_shared<dnnl::ukernel::brgemm>(
              M, N, K, batch_size, lda, ldb, ldc, dt_a, dt_b, dt_c, alpha, beta)});
      return cache_brgemm_kernels[key_brgemm];
    }
  }

std::shared_ptr<dnnl::ukernel::brgemm_pack_B> create_or_get_packb_microkernel(
  int64_t K,
  int64_t N,
  int ld_in,
  int ld_out,
  dt dt_in,
  dt dt_out) {
    PackBKey key_packb(K, N, ld_in, ld_out, dt_in, dt_out);
    auto search = cache_packb_kernels.find(key_packb);
    if (search != cache_packb_kernels.end()) {
      return search->second;
    } else {
      cache_packb_kernels.insert(
          {key_packb,
          std::make_shared<dnnl::ukernel::brgemm_pack_B>(
              K, N, ld_in, ld_out, dt_in, dt_out)});
      return cache_packb_kernels[key_packb];
    }
  }

// UINT8 unfused - u8s8s32
template <typename scalar_t, int64_t q_split_size, int64_t kv_split_size>
inline typename std::enable_if_t<std::is_same_v<scalar_t, unsigned char>, void>
cpu_flash_attention_u8(
    const at::Tensor& output,
    const at::Tensor& logsumexp,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    double dropout_p,
    bool is_causal,
    c10::optional<at::Tensor> attention_mask,
    c10::optional<double> scale,
    int32_t q_zp,
    float q_scale,
    int32_t k_zp,
    float k_scale,
    int32_t v_zp,
    float v_scale,
    int32_t a_zp,
    float a_scale,
    int32_t o_zp,
    float o_scale) {
  // std::cout << "enter cpu_flash_attention_u8" << std::endl;
  // using dt = dnnl::memory::data_type;
  using namespace dnnl;
  using namespace dnnl::ukernel;
  // auto starts = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
  // Query (Batch x Num_heads  x Q_seq_len  x Dim_per_head)
  //    -> (Batch x Q_seq_len  x Num_heads  x Dim_per_head)
  // Key   (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  // Value (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  at::Tensor query = q.transpose(1, 2);
  at::Tensor key = k.transpose(1, 2);
  at::Tensor value = v.transpose(1, 2);

  const auto accumulate_dtype = at::kFloat; // at::toOpMathType(dtype);

  using accum_t = float; // at::opmath_type<scalar_t>;
  using Vec = at::vec::Vectorized<accum_t>;
  accum_t scaling_factor =
      sdp::calculate_scale(query, scale).as_float_unchecked();
  if (attention_mask.has_value() && attention_mask.value().scalar_type() != ScalarType::Float) {
    attention_mask.value() = attention_mask.value().to(at::kFloat);
  }
  int block_64 = 64;
  // Sizes
  TORCH_CHECK(
      (query.size(3) == value.size(3)) && (key.size(3) == value.size(3)),
      "scaled_dot_product_attention_flash_attention: Q/K/V should have the same head size");
  TORCH_CHECK(
      kv_split_size % block_64 == 0, "kv_split_size is not divisble by ", block_64);

  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(1);
  int64_t kvSize = value.size(1);
  int64_t num_head = query.size(2);
  int64_t headSize = query.size(3);


  bool has_attn_mask = attention_mask.has_value() && attention_mask.value().numel();
  if (has_attn_mask) {
    reshape_attn_mask_to_4d(attention_mask.value(), batchSize, num_head, qSize, kvSize);
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
  // int64_t lStrideB = logsumexp.stride(0);
  // int64_t lStrideM = logsumexp.stride(1);
  // int64_t lStrideH = logsumexp.stride(2);
  int64_t mStrideB =
      (attention_mask.has_value() && attention_mask.value().size(0) > 1)
      ? attention_mask.value().stride(0)
      : 0;
  int64_t mStrideH =
      (attention_mask.has_value() && attention_mask.value().size(1) > 1)
      ? attention_mask.value().stride(1)
      : 0;
  int64_t mStrideM =
      (attention_mask.has_value() && attention_mask.value().size(2) > 1)
      ? attention_mask.value().stride(2)
      : 0;
  int64_t mStrideN =
      (attention_mask.has_value() && attention_mask.value().size(3) > 1)
      ? attention_mask.value().stride(3)
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
  bool av_gemm_K_tail_mul4 = kvTail % 4 == 0;
  int av_gemm_K_tail_padding = av_gemm_K_tail_mul4 ? 0 : 4 - kvTail % 4;
  int av_gemm_K_tail = kvTail + av_gemm_K_tail_padding;

  // allocate per thread temp buf (accumulate type)
  // int64_t size_per_thread =
  //     /* qk       */ kvSlice * qSplitSize * kvSplitSize + // qSplitSize * kvSize +
  //     /* dst      */ qSplitSize * headSize;
  int64_t size_per_thread =
      /* qk       */ kvSlice * qSplitSize * rndkvSplitSize + // qSplitSize * kvSize +
      /* dst      */ qSplitSize * rndHeadSize;

  // int64_t size_s32_per_thread =
  //     /* qk_s32   */ qSplitSize * kvSplitSize +
  //     /* dst      */ qSplitSize * headSize;
  int64_t size_s32_per_thread =
      /* qk_s32   */ qSplitSize * rndkvSplitSize +
      /* dst_s32  */ qSplitSize * rndHeadSize;

  at::Tensor buf = at::empty(
      {num_thread, size_per_thread}, query.options().dtype(accumulate_dtype));
  at::Tensor buf_reduced = at::empty(
      {num_thread,
       qSplitSize,
       av_gemm_K},
      query.options());
  at::Tensor buf_s32 = at::empty(
      {num_thread, size_s32_per_thread}, query.options().dtype(at::kInt));

  // allocate per thread temp buf (accumulate type)
  int64_t sum_size_per_thread =
      /* query_sum     */ qSplitSize +
      /* key_sum       */ kvSplitSize +
      /* attention_sum */ qSplitSize +
      /* value_sum     */ headSize +
      /* softmax_sum   */ qSplitSize;

  at::Tensor sum_buf = at::empty(
      {num_thread, sum_size_per_thread},
      query.options().dtype(accumulate_dtype));

  int64_t max_size_per_thread = /* softmax max */ qSplitSize;

  at::Tensor max_buf = at::empty(
      {num_thread, max_size_per_thread},
      query.options().dtype(accumulate_dtype));

  dt u8_dt = dt::u8;
  dt s8_dt = dt::s8;
  // dt f32_dt = dt::f32;
  dt s32_dt = dt::s32;

  // Data ptrs
  scalar_t* q_data = query.data_ptr<scalar_t>();
  scalar_t* k_data = key.data_ptr<scalar_t>();
  scalar_t* v_data = value.data_ptr<scalar_t>();
  accum_t* mask_data = attention_mask.has_value()
      ? attention_mask.value().data_ptr<accum_t>()
      : nullptr;
  scalar_t* out_data = output.data_ptr<scalar_t>();
  // accum_t* lse_data = logsumexp.data_ptr<accum_t>();
  accum_t* buf_data = buf.data_ptr<accum_t>();
  scalar_t* buf_reduced_data = buf_reduced.data_ptr<scalar_t>();
  int32_t* buf_s32_data = buf_s32.data_ptr<int32_t>();
  accum_t* sum_data = sum_buf.data_ptr<accum_t>();
  accum_t* max_data = max_buf.data_ptr<accum_t>();

  // Create tpp kernels for Query @ Key
  bool headSize_mul4 = headSize % 4 == 0;
  // If K of Gemm is not even, use mkl gemm instead of tpp for BF16
  int qk_gemm_K_padding = headSize_mul4 ? 0 : 4 - headSize % 4;
  int qk_gemm_K = headSize + qk_gemm_K_padding;

  auto && qk_gemm = create_or_get_microkernel(
    qSplitSize, block_64, qk_gemm_K,
            1, //batch_size
            headSize_mul4 ? qStrideM : qk_gemm_K, // lda
            block_64, //ldb
            rndkvSplitSize, //ldc
            u8_dt, //a dtype
            s8_dt, //b dtype
            s32_dt, //c dtype
            1.f, //alpha
            0.f //beta
            );
  (*qk_gemm).generate();
  size_t qk_scratchpad_size = (*qk_gemm).get_scratchpad_size();
  // std::vector<uint8_t> scratchpad_qk_gemm(scratchpad_size);
  (*qk_gemm).generate();

  auto && qk_gemm_ktail = create_or_get_microkernel(
    qSplitSize, block_64, qk_gemm_K,
            1, //batch_size
            headSize_mul4 ? qStrideM : qk_gemm_K, // lda
            block_64, //ldb
            rndkvTail, //ldc
            u8_dt, //a dtype
            s8_dt, //b dtype
            s32_dt, //c dtype
            1.0, //alpha
            0.0 //beta
            );
  size_t qk_ktail_scratchpad_size = (*qk_gemm_ktail).get_scratchpad_size();
  // std::vector<uint8_t> scratchpad_qk_gemm_ktail(scratchpad_size);
  (*qk_gemm_ktail).generate();

  std::shared_ptr<dnnl::ukernel::brgemm> qk_gemm_ktail_tail;
  if (kvTail % block_64 != 0) {
    qk_gemm_ktail_tail = create_or_get_microkernel(
      qSplitSize, kv_tail_tail_block_size, qk_gemm_K,
              1, //batch_size
              headSize_mul4 ? qStrideM : qk_gemm_K, // lda
              kv_tail_tail_block_size, //ldb
              rndkvTail, //ldc
              u8_dt, //a dtype
              s8_dt, //b dtype
              s32_dt, //c dtype
              1.0, //alpha
              0.0 //beta
              );
    (*qk_gemm_ktail_tail).generate();
  }

  auto && qk_gemm_qtail = create_or_get_microkernel(
    qTail, block_64, qk_gemm_K,
            1, //batch_size
            headSize_mul4 ? qStrideM : qk_gemm_K, // lda
            block_64, //ldb
            rndkvSplitSize, //ldc
            u8_dt, //a dtype
            s8_dt, //b dtype
            s32_dt, //c dtype
            1.0, //alpha
            0.0 //beta
            );
  size_t qk_qtail_scratchpad_size = (*qk_gemm_qtail).get_scratchpad_size();
  (*qk_gemm_qtail).generate();
  auto && qk_gemm_qktail = create_or_get_microkernel(
    qTail, block_64, qk_gemm_K,
            1, //batch_size
            headSize_mul4 ? qStrideM : qk_gemm_K, // lda
            block_64, //ldb
            rndkvTail, //ldc
            u8_dt, //a dtype
            s8_dt, //b dtype
            s32_dt, //c dtype
            1.0, //alpha
            0.0 //beta
            );
  size_t qk_qktail_scratchpad_size = (*qk_gemm_qktail).get_scratchpad_size();
  (*qk_gemm_qktail).generate();

  std::shared_ptr<dnnl::ukernel::brgemm> qk_gemm_qktail_tail;
  if (kvTail % block_64 != 0) {
    qk_gemm_qktail_tail = create_or_get_microkernel(
      qSplitSize, kv_tail_tail_block_size, qk_gemm_K,
              1, //batch_size
              headSize_mul4 ? qStrideM : qk_gemm_K, // lda
              kv_tail_tail_block_size, //ldb
              rndkvTail, //ldc
              u8_dt, //a dtype
              s8_dt, //b dtype
              s32_dt, //c dtype
              1.0, //alpha
              0.0 //beta
              );
    (*qk_gemm_qktail_tail).generate();
  }

  std::vector<std::pair<memory::dim, memory::dim>> A_B_offsets(1);
  A_B_offsets[0] = std::make_pair(0, 0);

  // Create tpp kernels for Attention @ Value
  auto && av_gemm = create_or_get_microkernel(
    qSplitSize, block_64, av_gemm_K,
            1, //batch_size
            av_gemm_K, // lda
            block_64, //ldb
            rndHeadSize, //ldc
            u8_dt, //a dtype
            s8_dt, //b dtype
            s32_dt, //c dtype
            1.0, //alpha
            0.0 //beta
            );
  size_t av_scratchpad_size = (*av_gemm).get_scratchpad_size();
  (*av_gemm).generate();
  auto && av_gemm_tail = create_or_get_microkernel(
    qSplitSize, block_64, av_gemm_K_tail,
            1, //batch_size
            av_gemm_K_tail, // lda
            block_64, //ldb
            rndHeadSize, //ldc
            u8_dt, //a dtype
            s8_dt, //b dtype
            s32_dt, //c dtype
            1.0, //alpha
            0.0 //beta
            );
  size_t av_tail_scratchpad_size = (*av_gemm_tail).get_scratchpad_size();
  (*av_gemm_tail).generate();

  auto && av_gemm_bias = create_or_get_microkernel(
    qSplitSize, block_64, av_gemm_K,
            1, //batch_size
            av_gemm_K, // lda
            block_64, //ldb
            rndHeadSize, //ldc
            u8_dt, //a dtype
            s8_dt, //b dtype
            s32_dt, //c dtype
            1.0, //alpha
            1.0 //beta
            );
  size_t av_bias_scratchpad_size = (*av_gemm_bias).get_scratchpad_size();
  (*av_gemm_bias).generate();

  auto && av_gemm_bias_tail = create_or_get_microkernel(
    qSplitSize, block_64, av_gemm_K_tail,
            1, //batch_size
            av_gemm_K_tail, // lda
            block_64, //ldb
            rndHeadSize, //ldc
            u8_dt, //a dtype
            s8_dt, //b dtype
            s32_dt, //c dtype
            1.0, //alpha
            1.0 //beta
            );
  size_t av_bias_tail_scratchpad_size = (*av_gemm_bias_tail).get_scratchpad_size();
  (*av_gemm_bias_tail).generate();

  // Buffer to store Key and Value after transforms
  at::Tensor key_t_reorder = at::empty(
      {batchSize,
       num_head,
       qk_gemm_K,
       rndkvSize},
      c10::CppTypeToScalarType<signed char>::value);
  // Buffer to store padding query
  scalar_t* query_padding_ptr = nullptr;
  std::unique_ptr<unsigned short[]> query_padding_data;
  if (!headSize_mul4) {
    query_padding_data = std::make_unique<unsigned short[]>(
        num_thread * qSplitSize * qk_gemm_K);
    query_padding_ptr = reinterpret_cast<scalar_t*>(query_padding_data.get());
  }
  auto key_reorder_ptr = key_t_reorder.data_ptr<signed char>();
  int kv_padding_size = (kvSize - 1) / kvSplitSize * av_gemm_K + av_gemm_K_tail;

  at::Tensor value_t_reorder = at::empty(
      {batchSize, num_head, kv_padding_size, rndHeadSize},
      c10::CppTypeToScalarType<signed char>::value);
  auto value_reorder_ptr = value_t_reorder.data_ptr<signed char>();

  // Create transforms for Key
  auto && brgemm_k_xform = create_or_get_packb_microkernel(
      qk_gemm_K, // K
      block_64, // N
      block_64, // ld_in
      block_64, // ld_out
      s8_dt, // dt_in
      s8_dt // dt_out
    );
  (*brgemm_k_xform).generate();
  auto && brgemm_k_xform_tail = create_or_get_packb_microkernel(
      qk_gemm_K,
      block_64,
      block_64,
      block_64,
      s8_dt,
      s8_dt
    );
  (*brgemm_k_xform_tail).generate();
  std::shared_ptr<dnnl::ukernel::brgemm_pack_B> brgemm_k_xform_tail_tail;
  if (kvTail % block_64 != 0) {
    brgemm_k_xform_tail_tail = create_or_get_packb_microkernel(
      qk_gemm_K,
      kv_tail_tail_block_size,
      kv_tail_tail_block_size,
      kv_tail_tail_block_size,
      s8_dt,
      s8_dt
    );
    (*brgemm_k_xform_tail_tail).generate();
  }

  // Create transforms for Value
  auto && brgemm_v_xform = create_or_get_packb_microkernel(
      av_gemm_K,
      block_64,
      block_64,
      block_64,
      s8_dt,
      s8_dt
    );
  (*brgemm_v_xform).generate();
  auto && brgemm_v_xform_tail = create_or_get_packb_microkernel(
      av_gemm_K_tail,
      block_64,
      block_64,
      block_64,
      s8_dt,
      s8_dt
    );
  (*brgemm_v_xform_tail).generate();

  // std::cout << "before order\n";
  // Reorder K, V
  // std::cout << "-- [start reorder] --" << std::endl;
  // std::vector<long> pack_time(num_thread, 0);
  // std::vector<long> transpose_time(num_thread, 0);
  // std::vector<long> other_time(num_thread, 0);
  // auto start = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
  at::parallel_for(
      0, batchSize * num_head * kvSlice, 1, [&](int64_t begin, int64_t end) {
        int64_t i = 0, j = 0, l = 0, n = 0;
        at::native::data_index_init(
            begin, i, batchSize, j, num_head, l, kvSlice);
        uint8_t* B_blocked_xform_u8 = new uint8_t[std::max(qk_gemm_K, av_gemm_K) * block_64];
        int8_t* B_blocked_xform_s8 = reinterpret_cast<signed char*>(B_blocked_xform_u8);
        for (const auto z : c10::irange(begin, end)) {
          (void)z; // Suppress unused variable
          n = l * kvSplitSize;
          // long ss, ee;
          if (n + kvSplitSize < kvSize) {
            for (int64_t b = 0; b < rndkvSplitSize; b += block_64) {
              bool tail = kvSplitSize - b < block_64;
              do_transpose(
                  k_data + i * kStrideB + j * kStrideH + n * kStrideN + b * kStrideN,
                  B_blocked_xform_u8,
                  tail ? kvSplitSize - b : block_64,
                  headSize,
                  kStrideN,
                  block_64);
              do_convert_u8_s8(
                  B_blocked_xform_u8,
                  B_blocked_xform_s8,
                  headSize,
                  block_64,
                  block_64,
                  block_64
              );
              if (!headSize_mul4) {
                pad_remain_row_col(
                    B_blocked_xform_s8,
                    headSize,
                    block_64,
                    qk_gemm_K,
                    block_64,
                    block_64
                  );
              }
              // Pack
              (*brgemm_k_xform).execute(
                B_blocked_xform_s8,
                key_reorder_ptr + i * num_head * qk_gemm_K * rndkvSize +
                      j * qk_gemm_K * rndkvSize + n * qk_gemm_K +
                      b * qk_gemm_K
              );
            }
            // split headSize to block_64, block_64, block_64 ...
            // [kvSplitSize, headSize] -> [av_gemm_K,  block_64 ...]
            for (int64_t b = 0; b < rndHeadSize; b += block_64) {
              copy_value_with_pad(
                v_data + i * vStrideB + j * vStrideH + n * vStrideN + b,
                B_blocked_xform_u8,
                kvSplitSize,
                headSize - b < block_64 ? headSize - b : block_64,
                av_gemm_K,
                block_64,
                vStrideN,
                /* pad val */ static_cast<scalar_t>(128));
              do_convert_u8_s8(
                B_blocked_xform_u8,
                B_blocked_xform_s8,
                av_gemm_K,
                block_64,
                block_64,
                block_64);
              (*brgemm_v_xform).execute(
                // v_xform(
                B_blocked_xform_s8,
                value_reorder_ptr +
                    i * num_head * kv_padding_size * rndHeadSize +
                    j * kv_padding_size * rndHeadSize + n * rndHeadSize +
                    av_gemm_K * b);
            }
          } else {
            // tail
            auto block_size = kvTail >= block_64 ? block_64 : kv_tail_tail_block_size;
            int64_t b = 0;
            while (b < rndkvTail) {
              bool tail = kvTail - b < block_size;
              do_transpose(
                  k_data + i * kStrideB + j * kStrideH + n * kStrideN + b * kStrideN,
                  B_blocked_xform_u8,
                  tail ? kvTail - b : block_size,
                  headSize,
                  kStrideN,
                  block_size);
              do_convert_u8_s8(
                  B_blocked_xform_u8,
                  B_blocked_xform_s8,
                  headSize,
                  block_size,
                  block_size,
                  block_size
              );
              if (!headSize_mul4) {
                pad_remain_row_col(
                    B_blocked_xform_s8,
                    headSize,
                    block_size,
                    qk_gemm_K,
                    block_size,
                    block_size
                  );
              }
              // Pack
              if (block_size == block_64) {
                (*brgemm_k_xform_tail).execute(
                  B_blocked_xform_s8,
                  key_reorder_ptr + i * num_head * qk_gemm_K * rndkvSize +
                        j * qk_gemm_K * rndkvSize + n * qk_gemm_K +
                        b * qk_gemm_K
                );
              } else {
                (*brgemm_k_xform_tail_tail).execute(
                  B_blocked_xform_s8,
                  key_reorder_ptr + i * num_head * qk_gemm_K * rndkvSize +
                        j * qk_gemm_K * rndkvSize + n * qk_gemm_K +
                        b * qk_gemm_K
                );
              }
              b += block_size;
              block_size = (kvTail - b) >= block_64 ? block_64 : kv_tail_tail_block_size;
            }
            // split headSize to block_64, block_64, block_64 ...
            // [kvTail, headSize] -> [av_gemm_K_tail,  block_64 ...]
            for (int64_t b = 0; b < headSize; b += block_64) {
              copy_value_with_pad(
                v_data + i * vStrideB + j * vStrideH + n * vStrideN + b,
                B_blocked_xform_u8,
                kvTail,
                headSize - b < block_64 ? headSize - b : block_64,
                av_gemm_K_tail,
                block_64,
                vStrideN,
                /* pad val */ static_cast<scalar_t>(128));
              do_convert_u8_s8(
                B_blocked_xform_u8,
                B_blocked_xform_s8,
                av_gemm_K_tail,
                block_64,
                block_64,
                block_64);
              (*brgemm_v_xform_tail).execute(
                B_blocked_xform_s8,
                value_reorder_ptr +
                    i * num_head * kv_padding_size * rndHeadSize +
                    j * kv_padding_size * rndHeadSize + n * rndHeadSize +
                    av_gemm_K_tail * b);
            }
          }
          // Move to the next query
          at::native::data_index_step(i, batchSize, j, num_head, l, kvSlice);
        }
      });
  // auto end = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
  // std::cout << "\n*** transpose pack: " << end-start << " ***" << std::endl;
  // std::cout << "[pack]:";
  // for (int i = 0; i < num_thread; i++) {
  //   std::cout << " " << pack_time[i];
  // }
  // std::cout << std::endl;
  // std::cout << "[transpose]:";
  // for (int i = 0; i < num_thread; i++) {
  //   std::cout << " " << transpose_time[i];
  // }
  // std::cout << std::endl;
  // std::cout << "[other]:";
  // for (int i = 0; i < num_thread; i++) {
  //   std::cout << " " << other_time[i];
  // }
  // std::cout << std::endl;
  // std::cout << "-- [after reorder] --" << std::endl;
  // std::cout << "value: " << value << "\n";
  // std::cout << "key: " << key << "\n";
  // std::cout << "value_t_reorder: " << value_t_reorder << "\n";
  // std::cout << "key_t_reorder: " << key_t_reorder << "\n";
  // exit(0);
  // std::cout << "before computing\n";
  // std::vector<long> pre_gemm1_time(num_thread, 0);
  // std::vector<long> gemm1_time(num_thread, 0);
  // std::vector<long> post_gemm1_time(num_thread, 0);
  // std::vector<long> softmax_time(num_thread, 0);
  // std::vector<long> pre_gemm2_time(num_thread, 0);
  // std::vector<long> gemm2_time(num_thread, 0);
  // std::vector<long> post_gemm2_time(num_thread, 0);
  // start = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
  at::parallel_for(
      0, batchSize * num_head * qSlice, 1, [&](int64_t begin, int64_t end) {
        int64_t i = 0, j = 0, k = 0;
        at::native::data_index_init(
            begin, i, batchSize, j, num_head, k, qSlice);
        int ompIdx = at::get_thread_num();
        accum_t* sum_ptr = sum_data + ompIdx * sum_size_per_thread;
        accum_t* q_sum_ptr = sum_ptr;
        accum_t* k_sum_ptr = q_sum_ptr + qSplitSize;
        accum_t* a_sum_ptr = k_sum_ptr + kvSplitSize;
        accum_t* v_sum_ptr = a_sum_ptr + qSplitSize;
        accum_t* sfm_sum_ptr = v_sum_ptr + headSize;
        accum_t* max_ptr = max_data + ompIdx * max_size_per_thread;
        accum_t* sfm_max_ptr = max_ptr;
        accum_t* buf_ptr = buf_data + ompIdx * size_per_thread;
        accum_t* qk_data = buf_ptr;
        accum_t* dst_data = qk_data + kvSlice * qSplitSize * rndkvSplitSize; // qSplitSize * kvSize;
        scalar_t* qk_reduced_data =
            buf_reduced_data + ompIdx * qSplitSize * av_gemm_K;
        int32_t* qk_s32_data =
            buf_s32_data + ompIdx * size_s32_per_thread;
        int32_t* dst_s32_data = qk_s32_data + qSplitSize * rndkvSplitSize;
        scalar_t* query_t_padding_ptr = !headSize_mul4
            ? query_padding_ptr + ompIdx * qSplitSize * qk_gemm_K
            : nullptr;
        std::vector<uint8_t> scratchpad_qk_gemm(qk_scratchpad_size);
        std::vector<uint8_t> scratchpad_qk_gemm_ktail(qk_ktail_scratchpad_size);
        std::vector<uint8_t> scratchpad_qk_gemm_qtail(qk_qtail_scratchpad_size);
        std::vector<uint8_t> scratchpad_qk_gemm_qktail(qk_qktail_scratchpad_size);
        std::vector<uint8_t> scratchpad_av_gemm(av_scratchpad_size);
        std::vector<uint8_t> scratchpad_av_gemm_tail(av_tail_scratchpad_size);
        std::vector<uint8_t> scratchpad_av_gemm_bias(av_bias_scratchpad_size);
        std::vector<uint8_t> scratchpad_av_gemm_bias_tail(av_bias_tail_scratchpad_size);

        for (const auto z : c10::irange(begin, end)) {
          (void)z; // Suppress unused variable
          int64_t m = k * qSplitSize;
          int64_t qBlockSize = std::min(qSplitSize, qSize - m);
          // Initialize sum and max
          fill_stub(
              sum_ptr, static_cast<accum_t>(0), sum_size_per_thread);
          fill_stub(
              max_ptr, static_cast<accum_t>(-std::numeric_limits<accum_t>::infinity()), max_size_per_thread);
          int64_t num_keys =
              is_causal ? std::min(m + qBlockSize, kvSize) : kvSize;
          if (!headSize_mul4) {
            // pad query if headSize is not even
            // [qBlockSize, headSize] -> [qBlockSize, headSize + 1]
            copy_value_with_pad(
                q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                query_t_padding_ptr,
                qBlockSize,
                headSize,
                qBlockSize,
                qk_gemm_K,
                qStrideM);
          }
          // auto s = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
          // std::cout << "after pad_col_zero\n";
          for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
            // auto ss = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
            // Calculate sums for dequant compensation item
            // std::cout << "[sum]: before gemm1 query: " << query << std::endl;
            _sum_b_contiguous_kernel(q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                q_sum_ptr, qBlockSize, headSize, qStrideM, /* accum */ false);
            // std::cout << "[sum]: before gemm1 key: " << key << std::endl;
            _sum_b_contiguous_kernel(k_data + i * kStrideB + j * kStrideH + n * kStrideN,
                k_sum_ptr, kvBlockSize, headSize, kStrideN, /* accum */ false);
            // auto ee = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            // pre_gemm1_time[ompIdx] += ee-ss;
            // std::cout << "[sum]: before gemm1 sum_buf: " << sum_buf << std::endl;
            // Calculate scale * q @ k.T
            // std::cout << "before qk_gemm\n";
            // ss = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            if (qBlockSize == qSplitSize) {
              // q main
              if (n + kvSplitSize < kvSize) {
                // k main
                for (int64_t b = 0; b < kvSplitSize; b += block_64) {
                  dnnl::ukernel::brgemm::release_hw_context();
                  (*qk_gemm).set_hw_context();
                  (*qk_gemm).execute(
                  // qk_gemm(
                    !headSize_mul4
                        ? query_t_padding_ptr
                        : q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                    key_reorder_ptr + i * num_head * qk_gemm_K * rndkvSize +
                        j * qk_gemm_K * rndkvSize + n * qk_gemm_K +
                        b * qk_gemm_K,
                    A_B_offsets,
                    qk_s32_data + b,
                    scratchpad_qk_gemm.data()
                    // 1
                    );
                }
              } else {
                auto block_size = kvTail >= block_64 ? block_64 : kv_tail_tail_block_size;
                int64_t b = 0;
                while (b < kvTail) {
                  dnnl::ukernel::brgemm::release_hw_context();
                  if (block_size == block_64) {    
                    dnnl::ukernel::brgemm::release_hw_context();
                    (*qk_gemm_ktail).set_hw_context();
                    (*qk_gemm_ktail).execute(
                        !headSize_mul4
                            ? query_t_padding_ptr
                            : q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                        key_reorder_ptr + i * num_head * qk_gemm_K * rndkvSize +
                            j * qk_gemm_K * rndkvSize + n * qk_gemm_K +
                            b * qk_gemm_K,
                        A_B_offsets,
                        qk_s32_data + b,
                        scratchpad_qk_gemm_ktail.data()
                        // 1
                        );
                  } else {
                    (*qk_gemm_ktail_tail).set_hw_context();
                    (*qk_gemm_ktail_tail).execute(
                        !headSize_mul4
                            ? query_t_padding_ptr
                            : q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                        key_reorder_ptr + i * num_head * qk_gemm_K * rndkvSize +
                            j * qk_gemm_K * rndkvSize + n * qk_gemm_K +
                            b * qk_gemm_K,
                        A_B_offsets,
                        qk_s32_data + b,
                        scratchpad_qk_gemm_ktail.data()
                        // 1
                        );
                  }
                  b += block_size;
                  block_size = (kvTail - b) >= block_64 ? block_64 : kv_tail_tail_block_size;
                }
              }
            } else {
              if (n + kvSplitSize < kvSize) {
                for (int64_t b = 0; b < kvSplitSize; b += block_64) {
                  dnnl::ukernel::brgemm::release_hw_context();
                  (*qk_gemm_qtail).set_hw_context();
                  (*qk_gemm_qtail).execute(
                    !headSize_mul4
                        ? query_t_padding_ptr
                        : q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                    key_reorder_ptr + i * num_head * qk_gemm_K * rndkvSize +
                        j * qk_gemm_K * rndkvSize + n * qk_gemm_K +
                        b * qk_gemm_K,
                    A_B_offsets,
                    qk_s32_data + b,
                    scratchpad_qk_gemm_qtail.data()
                    // 1
                    );
                }
              } else {
                // k tail
                auto block_size = kvTail >= block_64 ? block_64 : kv_tail_tail_block_size;
                int64_t b = 0;
                while (b < kvTail) {
                  dnnl::ukernel::brgemm::release_hw_context();
                  if (block_size == block_64) {
                    (*qk_gemm_qktail).set_hw_context();
                    (*qk_gemm_qktail).execute(
                        !headSize_mul4
                            ? query_t_padding_ptr
                            : q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                        key_reorder_ptr + i * num_head * qk_gemm_K * rndkvSize +
                            j * qk_gemm_K * rndkvSize + n * qk_gemm_K +
                            b * qk_gemm_K,
                        A_B_offsets,
                        qk_s32_data + b,
                        scratchpad_qk_gemm_qktail.data()
                        // 1
                        );
                  } else {
                    (*qk_gemm_qktail_tail).set_hw_context();
                    (*qk_gemm_qktail_tail).execute(
                        !headSize_mul4
                            ? query_t_padding_ptr
                            : q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                        key_reorder_ptr + i * num_head * qk_gemm_K * rndkvSize +
                            j * qk_gemm_K * rndkvSize + n * qk_gemm_K +
                            b * qk_gemm_K,
                        A_B_offsets,
                        qk_s32_data + b,
                        scratchpad_qk_gemm_qktail.data()
                        // 1
                        );
                  }
                b += block_size;
                block_size = (kvTail - b) >= block_64 ? block_64 : kv_tail_tail_block_size;
                }
              }
            }
            // ee = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            // gemm1_time[ompIdx] += ee-ss;
            // std::cout << "after qk_gemm" << std::endl;

            // do dequant compensation and convert qk from s32 to fp32
            // ss = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            int64_t rndkvBlockSize = kvBlockSize == kvSplitSize ? rndkvSplitSize : rndkvTail;
            accum_t* qk_block_data = qk_data + n * qSplitSize;
            _dequant_kernel_u8_s8(qk_s32_data, qk_block_data,
                q_sum_ptr, k_sum_ptr,
                qBlockSize, rndkvBlockSize, headSize, rndkvBlockSize, kvBlockSize,
                q_scale, k_scale, q_zp, k_zp);

            // Apply causal mask, fill unused with -inf
            if (is_causal && num_keys - n <= kvSplitSize) {
              // std::cout << "[enter is_causal]" << std::endl;
              for (const auto row : c10::irange(qBlockSize)) {
                int64_t last_col = m + row - n;
                accum_t* row_ptr = qk_block_data + row * kvBlockSize; // + n + row * kvSize;
                fill_stub(
                    row_ptr + last_col + 1,
                    -std::numeric_limits<accum_t>::infinity(),
                    kvBlockSize - last_col - 1);
              }
            }
            // Update attention weights with attention mask
            // And apply scaling factor
            // ee = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            // post_gemm1_time[ompIdx] += ee-ss;
            // ss = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            // calculate max for softmax
            if (attention_mask.has_value()) {
              for (const auto row : c10::irange(qBlockSize)) {
                accum_t prev_max = sfm_max_ptr[row];
                accum_t curr_max = 0;
                _mul_add_reduce_max_fusion_kernel(
                      qk_block_data + row * kvBlockSize,
                      mask_data + i * mStrideB + j * mStrideH +
                        (m + row) * mStrideM + (mStrideN == 0 ? 0 : n),
                      scaling_factor,
                      kvBlockSize,
                      qk_block_data + row * kvBlockSize,
                      curr_max);
                sfm_max_ptr[row] = std::max(prev_max, curr_max);
              }
            } else {
              for (const auto row : c10::irange(qBlockSize)) {
                accum_t prev_max = sfm_max_ptr[row];
                accum_t curr_max = 0;
                _mul_reduce_max_fusion_kernel(
                      qk_block_data + row * kvBlockSize,
                      scaling_factor,
                      kvBlockSize,
                      qk_block_data + row * kvBlockSize,
                      curr_max);
                sfm_max_ptr[row] = std::max(prev_max, curr_max);
              }
            }
          }
          // auto e = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
          // gemm1_time[ompIdx] += e-s;
          // std::cout << "[gemm1]: " << e-s << std::endl;
          // s = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();

          // Softmax
          for (int64_t row = 0; row < qBlockSize; ++row) {
            // x - max, exp
            for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
              int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
              accum_t* qk_block_data = qk_data + n * qSplitSize;
              _sub_exp_sum_fusion_kernel(
                    qk_block_data + row * kvBlockSize,
                    kvBlockSize,
                    qk_block_data + row * kvBlockSize,
                    sfm_max_ptr[row],
                    sfm_sum_ptr[row]);
            }
            // x / sum
            for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
              int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
              accum_t* qk_block_data = qk_data + n * qSplitSize;
              accum_t sum_reciprocal = 1 / sfm_sum_ptr[row];
              at::vec::map<accum_t>(
                [sum_reciprocal](Vec x) { return x * Vec(sum_reciprocal); },
                qk_block_data + row * kvBlockSize,
                qk_block_data + row * kvBlockSize,
                kvBlockSize);
            }
          }

          // e = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
          // softmax_time[ompIdx] += e-s;
          // s = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
          // long ss, ee;
          for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
            // ss = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
            accum_t* qk_block_data = qk_data + n * qSplitSize;

            // do quant and convert qk from fp32 to u8
            _quant_kernel(
                qk_block_data, qk_reduced_data,
                qBlockSize, kvBlockSize, kvBlockSize,
                n + kvSplitSize >= kvSize ? av_gemm_K_tail : av_gemm_K,
                a_scale, a_zp);

            // Calculate sums for dequant compensation item
            _sum_a_contiguous_kernel(
                v_data + i * vStrideB + j * vStrideH + n * vStrideN,
                v_sum_ptr, headSize, kvBlockSize, vStrideN);
            _sum_b_contiguous_kernel(
                qk_reduced_data, a_sum_ptr,
                qBlockSize, kvBlockSize,
                n + kvSplitSize >= kvSize ? av_gemm_K_tail : av_gemm_K,
                /* accum */ true);
            // ee = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            // pre_gemm2_time[ompIdx] += ee-ss;

            // Calculate Softmax(q @ k.T) @ v
            // ss = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            // std::cout << "before av gemm\n";
            if (n + kvSplitSize < kvSize) {
              // main
              if (n == 0) {
                for (int64_t b = 0; b < headSize; b += block_64) {
                  dnnl::ukernel::brgemm::release_hw_context();
                  (*av_gemm).set_hw_context();
                  (*av_gemm).execute(
                    qk_reduced_data,
                    value_reorder_ptr +
                        i * num_head * kv_padding_size * rndHeadSize +
                        j * kv_padding_size * rndHeadSize + n * rndHeadSize
                        + b * av_gemm_K,
                    A_B_offsets,
                    dst_s32_data + b,
                    scratchpad_av_gemm.data()
                    // 1
                    );
                }
              } else {
                // bias
                for (int64_t b = 0; b < headSize; b += block_64) {
                  dnnl::ukernel::brgemm::release_hw_context();
                  (*av_gemm_bias).set_hw_context();
                  (*av_gemm_bias).execute(
                  // av_gemm_bias(
                      qk_reduced_data,
                      value_reorder_ptr +
                          i * num_head * kv_padding_size * rndHeadSize +
                          j * kv_padding_size * rndHeadSize + n * rndHeadSize
                          + b * av_gemm_K,
                      A_B_offsets,
                      dst_s32_data + b,
                      scratchpad_av_gemm_bias.data()
                      // 1
                      );
                }
              }
            } else if (n + kvSplitSize >= kvSize) {
              // tail
              if (n == 0) {
                for (int64_t b = 0; b < headSize; b += block_64) {
                  dnnl::ukernel::brgemm::release_hw_context();
                  (*av_gemm_tail).set_hw_context();
                  (*av_gemm_tail).execute(
                    qk_reduced_data,
                    value_reorder_ptr +
                        i * num_head * kv_padding_size * rndHeadSize +
                        j * kv_padding_size * rndHeadSize + n * rndHeadSize
                        + b * av_gemm_K_tail,
                    A_B_offsets,
                    dst_s32_data + b,
                    scratchpad_av_gemm_tail.data()
                    // 1
                    );
                }
              } else {
                // bias
                for (int64_t b = 0; b < headSize; b += block_64) {
                  dnnl::ukernel::brgemm::release_hw_context();
                  (*av_gemm_bias_tail).set_hw_context();
                  (*av_gemm_bias_tail).execute(
                    qk_reduced_data,
                    value_reorder_ptr +
                        i * num_head * kv_padding_size * rndHeadSize +
                        j * kv_padding_size * rndHeadSize + n * rndHeadSize
                        + b * av_gemm_K_tail,
                    A_B_offsets,
                    dst_s32_data + b,
                    scratchpad_av_gemm_bias_tail.data()
                    // 1
                    );
                }
              }
            }
            // ee = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            // gemm2_time[ompIdx] += ee-ss;
          }

          // After the last gemm,
          // do dequant compensation and convert dst from s32 to fp32
          // ss = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
          _dequant_kernel_u8_s8(dst_s32_data, dst_data,
                a_sum_ptr, v_sum_ptr,
                qBlockSize, rndHeadSize, kvSize, rndHeadSize, headSize,
                a_scale, v_scale, a_zp, v_zp);

          // dst <- dst / sum[row]
          // reorder MHA output with strides
          // do quant and convert qk from fp32 to u8
          for (int64_t row = 0; row < qBlockSize; ++row) {
            // accum_t sum_reciprocal = 1 / qk_sum_data[row];
            _quant_scale_reorder_kernel(
                dst_data + row * headSize,
                out_data + i * oStrideB + j * oStrideH + m * oStrideM
                + row * oStrideM,
                1.0,
                headSize,
                o_scale,
                o_zp);
          }
          // ee = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
          // post_gemm2_time[ompIdx] += ee-ss;
          // Store logsumexp for backward
          // accum_t* lse_ptr =
          //     lse_data + i * lStrideB + j * lStrideH + m * lStrideM;
          // for (const auto row : c10::irange(qBlockSize)) {
          //   lse_ptr[row * lStrideM] =
          //       qk_max_data[row] + std::log(qk_sum_data[row]);
          // }
          // Move to the next query
          at::native::data_index_step(i, batchSize, j, num_head, k, qSlice);
        }
      });
  // end = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
  // std::cout << "\n*** sdpa: " << end-start << " ***" << std::endl;
  // std::cout << "[pre_gemm1]:";
  // for (int i = 0; i < num_thread; i++) {
  //   std::cout << " " << pre_gemm1_time[i];
  // }
  // std::cout << std::endl;
  // std::cout << "[gemm1]:";
  // for (int i = 0; i < num_thread; i++) {
  //   std::cout << " " << gemm1_time[i];
  // }
  // std::cout << std::endl;
  // std::cout << "[post_gemm1]:";
  // for (int i = 0; i < num_thread; i++) {
  //   std::cout << " " << post_gemm1_time[i];
  // }
  // std::cout << std::endl;
  // std::cout << "[softmax]:";
  // for (int i = 0; i < num_thread; i++) {
  //   std::cout << " " << softmax_time[i];
  // }
  // std::cout << std::endl;
  // std::cout << "[pre_gemm2]:";
  // for (int i = 0; i < num_thread; i++) {
  //   std::cout << " " << pre_gemm2_time[i];
  // }
  // std::cout << std::endl;
  // std::cout << "[gemm2]:";
  // for (int i = 0; i < num_thread; i++) {
  //   std::cout << " " << gemm2_time[i];
  // }
  // std::cout << std::endl;
  // std::cout << "[post_gemm2]:";
  // for (int i = 0; i < num_thread; i++) {
  //   std::cout << " " << post_gemm2_time[i];
  // }
  // std::cout << std::endl;
  // Once all computations are done, need to release HW context.
  brgemm::release_hw_context();
}

template <typename scalar_t, typename mask_t, int64_t q_split_size, int64_t kv_split_size>
void cpu_flash_attention(
    const Tensor& output,
    const Tensor& logsumexp,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    double dropout_p,
    bool is_causal,
    std::optional<Tensor> attn_mask,
    std::optional<double> scale) {
  // Query (Batch x Num_heads  x Q_seq_len  x Dim_per_head)
  //    -> (Batch x Q_seq_len  x Num_heads  x Dim_per_head)
  // Key   (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  // Value (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  at::Tensor query = q.transpose(1, 2);
  at::Tensor key = k.transpose(1, 2);
  at::Tensor value = v.transpose(1, 2);

  constexpr bool is_reduced_type = is_reduced_floating_point_v<scalar_t>;
  using accum_t = at::opmath_type<scalar_t>;
  using Vec = vec::Vectorized<accum_t>;
  accum_t scaling_factor =
      sdp::calculate_scale(query, scale).as_float_unchecked();

  // Sizes
  TORCH_CHECK((query.size(3) == value.size(3)) && (key.size(3) == value.size(3)),
        "scaled_dot_product_attention_flash_attention: Q/K/V should have the same head size");
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
  int64_t lStrideB = logsumexp.stride(0);
  int64_t lStrideM = logsumexp.stride(1);
  int64_t lStrideH = logsumexp.stride(2);
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
  int64_t num_thread = at::get_num_threads();

  const auto dtype = query.scalar_type();
  const auto accumulate_dtype = toOpMathType(dtype);

  // allocate per thread temp buf (accumulate type)
  int64_t size_per_thread =
      /* qk     */ qSplitSize * kvSplitSize +
      /* qk_max */ qSplitSize +
      /* qk_sum */ qSplitSize +
      /* dst    */ qSplitSize * headSize;

  at::Tensor buf = at::empty({num_thread, size_per_thread}, query.options().dtype(accumulate_dtype));
  at::Tensor buf_reduced = at::empty({num_thread, qSplitSize, is_reduced_type ? kvSplitSize : 0}, query.options());

  // Data ptrs
  const scalar_t* q_data = query.const_data_ptr<scalar_t>();
  const scalar_t* k_data = key.const_data_ptr<scalar_t>();
  const scalar_t* v_data = value.const_data_ptr<scalar_t>();
  mask_t* mask_data = has_attn_mask
      ? attn_mask.value().data_ptr<mask_t>()
      : nullptr;
  scalar_t* out_data = output.data_ptr<scalar_t>();
  accum_t* lse_data = logsumexp.data_ptr<accum_t>();
  accum_t* buf_data = buf.data_ptr<accum_t>();
  scalar_t* buf_reduced_data = is_reduced_type ? buf_reduced.data_ptr<scalar_t>() : nullptr;

  at::parallel_for(0, batchSize * num_head * qSlice, 1, [&](int64_t begin, int64_t end) {
    int64_t i = 0, j = 0, k = 0;
    data_index_init(begin, i, batchSize, j, num_head, k, qSlice);
    int ompIdx = at::get_thread_num();
    accum_t* buf_ptr = buf_data + ompIdx * size_per_thread;
    accum_t* qk_data = buf_ptr;
    accum_t* qk_max_data = qk_data + qSplitSize * kvSplitSize;
    accum_t* qk_sum_data = qk_max_data + qSplitSize;
    accum_t* dst_data = qk_sum_data + qSplitSize;
    scalar_t* qk_reduced_data = is_reduced_type ? buf_reduced_data + ompIdx * qSplitSize * kvSplitSize : nullptr;

    for (const auto z : c10::irange(begin, end)) {
      (void)z; // Suppress unused variable
      int64_t m = k * qSplitSize;
      int64_t qBlockSize = std::min(qSplitSize, qSize - m);
      // Initialize max and sum
      fill_stub(qk_max_data,
          -std::numeric_limits<accum_t>::infinity(), qBlockSize);
      fill_stub(qk_sum_data,
          static_cast<accum_t>(0), qBlockSize);
      int64_t num_keys = is_causal ? std::min(m + qBlockSize, kvSize) : kvSize;
      for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
        int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
        // Calculate scale * q @ k.T
        cpublas::gemm(
            TransposeType::Transpose,
            TransposeType::NoTranspose,
            kvBlockSize,
            qBlockSize,
            headSize,
            static_cast<accum_t>(1),
            k_data + i * kStrideB + j * kStrideH +
                n * kStrideN,
            kStrideN,
            q_data + i * qStrideB + j * qStrideH +
                m * qStrideM,
            qStrideM,
            static_cast<accum_t>(0),
            qk_data,
            kvBlockSize);
        // Apply causal mask, fill unused with -inf
        if (is_causal && num_keys - n <= kvSplitSize) {
          for (const auto row : c10::irange(qBlockSize)) {
            int64_t last_col = m + row - n;
            accum_t* row_ptr = qk_data + row * kvBlockSize;
            fill_stub(row_ptr + last_col + 1,
                -std::numeric_limits<accum_t>::infinity(),
                kvBlockSize - last_col - 1);
          }
        }
        // Update attention weights with attention mask
        // And apply scaling factor
        // qk <- qk * scaling + attn_mask
        if (has_attn_mask) {
          for (int64_t row = 0; row < qBlockSize; ++row) {
            if (mStrideN == 0) {
              _scale_attn_mask_fusion_kernel</*is_stride_0*/ true>(
                qk_data + row * kvBlockSize,
                mask_data + i * mStrideB + j * mStrideH +
                    (m + row) * mStrideM,
                kvBlockSize,
                qk_data + row * kvBlockSize,
                scaling_factor);
            } else {
              _scale_attn_mask_fusion_kernel</*is_stride_0*/ false>(
                qk_data + row * kvBlockSize,
                mask_data + i * mStrideB + j * mStrideH +
                    (m + row) * mStrideM + n,
                kvBlockSize,
                qk_data + row * kvBlockSize,
                scaling_factor);
            }
          }
        }
        // Update coefficients with Softmax
        accum_t tmp_max = 0, tmp_sum = 0, exp_tmp = 0;
        for (int64_t row = 0; row < qBlockSize; ++row) {
          if (has_attn_mask) {
            // max per row
            tmp_max = at::vec::reduce_all<accum_t>(
                [](Vec& x, Vec& y) { return at::vec::maximum(x, y); },
                qk_data + row * kvBlockSize,
                kvBlockSize);
          } else {
            // apply scaling factor and max per row in fusion
            _mul_reduce_max_fusion_kernel(
                qk_data + row * kvBlockSize,
                scaling_factor,
                kvBlockSize,
                qk_data + row * kvBlockSize,
                tmp_max);
          }
          tmp_max = qk_max_data[row] > tmp_max ? qk_max_data[row] : tmp_max;
          // qk <- exp(qk - max) and sum per row
          tmp_sum = tmp_max;
          _exp_reduce_sum_fusion_kernel(
              qk_data + row * kvBlockSize, kvBlockSize,
              conditional_data_ptr(qk_data, qk_reduced_data) + row * kvBlockSize,
              tmp_sum);
          // exp_tmp <- exp(max[row] - max)
          exp_tmp = std::exp(qk_max_data[row] - tmp_max);
          // sum[row] <- sum + exp_tmp * sum[row]
          qk_sum_data[row] = tmp_sum + exp_tmp * qk_sum_data[row];
          // max[row] <- max
          qk_max_data[row] = tmp_max;
          // dst <- dst * exp_tmp
          if (n > 0) {
            vec::map<accum_t>(
              [exp_tmp](Vec x) { return x * Vec(exp_tmp); },
              dst_data + row * headSize, dst_data + row * headSize, headSize);
          }
        }
        // Calculate Softmax(q @ k.T) @ v
        cpublas::gemm(
            TransposeType::NoTranspose,
            TransposeType::NoTranspose,
            headSize,
            qBlockSize,
            kvBlockSize,
            static_cast<accum_t>(1),
            v_data + i * vStrideB + j * vStrideH +
                n * vStrideN,
            vStrideN,
            conditional_data_ptr(qk_data, qk_reduced_data),
            kvBlockSize,
            n == 0 ? static_cast<accum_t>(0) : static_cast<accum_t>(1),
            dst_data,
            headSize);
      }
      // dst <- dst / sum[row]
      // reorder MHA output with strides
      for (int64_t row = 0; row < qBlockSize; ++row) {
        accum_t sum_reciprocal = 1 / qk_sum_data[row];
        vec::map<scalar_t>(
          [sum_reciprocal](Vec x) { return x * Vec(sum_reciprocal); },
          out_data + i * oStrideB + j * oStrideH + m * oStrideM + row * oStrideM,
          dst_data + row * headSize,
          headSize);
      }
      // Store logsumexp for backward
      accum_t* lse_ptr = lse_data + i * lStrideB + j * lStrideH + m * lStrideM;
      for (const auto row : c10::irange(qBlockSize)) {
        lse_ptr[row * lStrideM] = qk_max_data[row]
            + std::log(qk_sum_data[row]);
      }
      // Move to the next query
      data_index_step(i, batchSize, j, num_head, k, qSlice);
    }
  });

}

template <typename scalar_t, typename mask_t, int64_t q_split_size, int64_t kv_split_size>
void cpu_flash_attention_backward(
    const at::Tensor& grad_q,
    const at::Tensor& grad_k,
    const at::Tensor& grad_v,
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    double dropout_p,
    bool is_causal,
    std::optional<Tensor> attn_mask,
    std::optional<double> scale) {
  constexpr bool is_reduced_type = is_reduced_floating_point_v<scalar_t>;
  using accum_t = at::opmath_type<scalar_t>;
  using Vec = vec::Vectorized<accum_t>;
  accum_t scaling_factor =
      sdp::calculate_scale(query, scale).as_float_unchecked();

  // Sizes
  TORCH_CHECK((query.size(3) == value.size(3)) && (key.size(3) == value.size(3)),
        "scaled_dot_product_attention_flash_attention_backward: Q/K/V should have the same head size");
  // Query (Batch x Q_seq_len  x Num_heads x Dim_per_head)
  // Key   (Batch x KV_seq_len x Num_heads x Dim_per_head)
  // Value (Batch x KV_seq_len x Num_heads x Dim_per_head)
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
  int64_t oStrideB = out.stride(0);
  int64_t oStrideM = out.stride(1);
  int64_t oStrideH = out.stride(2);
  int64_t lStrideB = logsumexp.stride(0);
  int64_t lStrideM = logsumexp.stride(1);
  int64_t lStrideH = logsumexp.stride(2);
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

  int64_t grad_qStrideB = grad_q.stride(0);
  int64_t grad_qStrideM = grad_q.stride(1);
  int64_t grad_qStrideH = grad_q.stride(2);
  int64_t grad_kStrideB = grad_k.stride(0);
  int64_t grad_kStrideN = grad_k.stride(1);
  int64_t grad_kStrideH = grad_k.stride(2);
  int64_t grad_vStrideB = grad_v.stride(0);
  int64_t grad_vStrideN = grad_v.stride(1);
  int64_t grad_vStrideH = grad_v.stride(2);
  int64_t grad_oStrideB = grad_out.stride(0);
  int64_t grad_oStrideM = grad_out.stride(1);
  int64_t grad_oStrideH = grad_out.stride(2);

  int64_t qSplitSize = q_split_size > qSize ? qSize : q_split_size;
  int64_t kvSplitSize = kv_split_size > kvSize ? kvSize : kv_split_size;
  int64_t num_thread = at::get_num_threads();

  const auto dtype = query.scalar_type();
  const auto accumulate_dtype = toOpMathType(dtype);

  // allocate per thread temp buf (accumulate type)
  int64_t size_per_thread =
      /* attn      */ qSplitSize * kvSplitSize +
      /* grad_attn */ qSplitSize * kvSplitSize;

  at::Tensor buf = at::empty({num_thread, size_per_thread}, query.options().dtype(accumulate_dtype));

  // allocate per thread temp buf_reduced (scalar type)
  // buf2 is only needed for bfloat16 and float16
  int64_t size_per_thread_reduced =
      /* attn_reduced      */ qSplitSize * kvSplitSize +
      /* grad_attn_reduced */ qSplitSize * kvSplitSize;

  at::Tensor buf_reduced = at::empty({num_thread, is_reduced_type ? size_per_thread_reduced : 0}, query.options());

  scalar_t* grad_q_data = grad_q.data_ptr<scalar_t>();
  scalar_t* grad_k_data = grad_k.data_ptr<scalar_t>();
  scalar_t* grad_v_data = grad_v.data_ptr<scalar_t>();
  const scalar_t* grad_out_data = grad_out.const_data_ptr<scalar_t>();
  const scalar_t* q_data = query.const_data_ptr<scalar_t>();
  const scalar_t* k_data = key.const_data_ptr<scalar_t>();
  const scalar_t* v_data = value.const_data_ptr<scalar_t>();
  mask_t* mask_data = has_attn_mask
      ? attn_mask.value().data_ptr<mask_t>()
      : nullptr;
  const scalar_t* out_data = out.const_data_ptr<scalar_t>();
  const accum_t* lse_data = logsumexp.const_data_ptr<accum_t>();
  accum_t* buf_data = buf.data_ptr<accum_t>();
  scalar_t* buf_reduced_data = is_reduced_type ? buf_reduced.data_ptr<scalar_t>() : nullptr;

  at::parallel_for(0, batchSize * num_head, 1, [&](int64_t begin, int64_t end) {
    int64_t i = 0, j = 0;
    data_index_init(begin, i, batchSize, j, num_head);
    int ompIdx = at::get_thread_num();
    accum_t* buf_ptr = buf_data + ompIdx * size_per_thread;
    accum_t* attn_data = buf_ptr;
    accum_t* grad_attn_data = attn_data + qSplitSize * kvSplitSize;
    scalar_t* buf_reduced_ptr = is_reduced_type ? buf_reduced_data + ompIdx * size_per_thread_reduced : nullptr;
    scalar_t* attn_reduced_data = is_reduced_type ? buf_reduced_ptr : nullptr;
    scalar_t* grad_attn_reduced_data = is_reduced_type ? attn_reduced_data + qSplitSize * kvSplitSize : nullptr;

    at::Tensor dsum = at::empty({qSplitSize}, query.options().dtype(accumulate_dtype));
    accum_t* dsum_data = dsum.data_ptr<accum_t>();
    for (const auto z : c10::irange(begin, end)) {
      (void)z; // Suppress unused variable
      // rowsum of grad_out * out
      for (int64_t m = 0; m < qSize; m += qSplitSize) {
        int64_t qBlockSize = std::min(qSplitSize, qSize - m);
        // dsum <- rowsum(grad_out * out)
        for (const auto row : c10::irange(qBlockSize)) {
          *(dsum_data + row) = vec::map2_reduce_all<scalar_t>(
            [](Vec x, Vec y) { return x * y; },
            [](Vec x, Vec y) { return x + y; },
            grad_out_data + i * grad_oStrideB + j * grad_oStrideH + (m + row) * grad_oStrideM,
            out_data + i * oStrideB + j * oStrideH + (m + row) * oStrideM,
            headSize);
        }
        int64_t num_keys = is_causal ? std::min(m + qBlockSize, kvSize) : kvSize;
        for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
          int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
          // attn <- scale * q @ k.T
          cpublas::gemm(
            TransposeType::Transpose,
            TransposeType::NoTranspose,
            kvBlockSize,
            qBlockSize,
            headSize,
            scaling_factor,
            k_data + i * kStrideB + j * kStrideH +
                n * kStrideN,
            kStrideN,
            q_data + i * qStrideB + j * qStrideH +
                m * qStrideM,
            qStrideM,
            static_cast<accum_t>(0),
            attn_data,
            kvBlockSize);
          // attn <- attn + mask
          if (has_attn_mask) {
            accum_t one = accum_t(1);
            for (const auto row : c10::irange(qBlockSize)) {
              if (mStrideN == 0) {
                _scale_attn_mask_fusion_kernel</*is_stride_0*/ true>(
                  attn_data + row * kvBlockSize,
                  mask_data + i * mStrideB + j * mStrideH +
                      (m + row) * mStrideM,
                  kvBlockSize,
                  attn_data + row * kvBlockSize,
                  one);
              } else {
                _scale_attn_mask_fusion_kernel</*is_stride_0*/ false>(
                  attn_data + row * kvBlockSize,
                  mask_data + i * mStrideB + j * mStrideH +
                      (m + row) * mStrideM + n,
                  kvBlockSize,
                  attn_data + row * kvBlockSize,
                  one);
              }
            }
          }
          // restore self attention after softmax from logsumexp
          // attn <- exp(attn - normalizer)
          for (const auto row : c10::irange(qBlockSize)) {
            accum_t normalizer = lse_data[i * lStrideB + j * lStrideH + (m + row) * lStrideM];
            vec::map<accum_t>(
              [normalizer](Vec x) { return (x - Vec(normalizer)).exp(); },
              attn_data + row * kvBlockSize,
              attn_data + row * kvBlockSize,
              kvBlockSize);
          }
          // Apply causal mask, filled unused with 0
          if (is_causal && num_keys - n <= kvSplitSize) {
            for (const auto row : c10::irange(qBlockSize)) {
              int64_t last_col = m + row - n;
              accum_t* row_ptr = attn_data + row * kvBlockSize;
              fill_stub(row_ptr + last_col + 1, static_cast<accum_t>(0), kvBlockSize - last_col - 1);
            }
          }
          if (is_reduced_type) {
            for (const auto row : c10::irange(qBlockSize)) {
              convert<accum_t, scalar_t>(
                attn_data + row * kvBlockSize,
                attn_reduced_data + row * kvBlockSize,
                kvBlockSize);
            }
          }
          // grad_v <- grad_v + attn.T @ grad_out
          cpublas::gemm(
            TransposeType::NoTranspose,
            TransposeType::Transpose,
            headSize,
            kvBlockSize,
            qBlockSize,
            static_cast<accum_t>(1),
            grad_out_data + i * grad_oStrideB + j * grad_oStrideH +
                m * grad_oStrideM,
            grad_oStrideM,
            conditional_data_ptr(attn_data, attn_reduced_data),
            kvBlockSize,
            static_cast<accum_t>(1),
            grad_v_data + i * grad_vStrideB + j * grad_vStrideH +
                n * grad_vStrideN,
            grad_vStrideN);
          // grad_attn <- grad_out @ v.T
          cpublas::gemm(
            TransposeType::Transpose,
            TransposeType::NoTranspose,
            kvBlockSize,
            qBlockSize,
            headSize,
            static_cast<accum_t>(1),
            v_data + i * vStrideB + j * vStrideH +
                n * vStrideN,
            vStrideN,
            grad_out_data + i * grad_oStrideB + j * grad_oStrideH +
                m * grad_oStrideM,
            grad_oStrideM,
            static_cast<accum_t>(0),
            grad_attn_data,
            kvBlockSize);
          // grad_attn <- attn * (grad_attn - dsum)
          for (const auto row : c10::irange(qBlockSize)) {
            accum_t d = *(dsum_data + row);
            vec::map2<accum_t>(
              [d](Vec attn, Vec grad_attn) { return attn * (grad_attn - Vec(d)); },
              grad_attn_data + row * kvBlockSize,
              attn_data + row * kvBlockSize,
              grad_attn_data + row * kvBlockSize,
              kvBlockSize);
          }
          if (is_reduced_type) {
            for (const auto row : c10::irange(qBlockSize)) {
              convert<accum_t, scalar_t>(
                grad_attn_data + row * kvBlockSize,
                grad_attn_reduced_data + row * kvBlockSize,
                kvBlockSize);
            }
          }
          // grad_q <- grad_q + scale * grad_attn @ k
          cpublas::gemm(
            TransposeType::NoTranspose,
            TransposeType::NoTranspose,
            headSize,
            qBlockSize,
            kvBlockSize,
            scaling_factor,
            k_data + i * kStrideB + j * kStrideH +
                n * kStrideN,
            kStrideN,
            conditional_data_ptr(grad_attn_data, grad_attn_reduced_data),
            kvBlockSize,
            static_cast<accum_t>(1),
            grad_q_data + i * grad_qStrideB + j * grad_qStrideH +
                m * grad_qStrideM,
            grad_qStrideM);
          // grad_k <- grad_k + scale * grad_attn.T @ q
          cpublas::gemm(
            TransposeType::NoTranspose,
            TransposeType::Transpose,
            headSize,
            kvBlockSize,
            qBlockSize,
            scaling_factor,
            q_data + i * qStrideB + j * qStrideH +
                m * qStrideM,
            qStrideM,
            conditional_data_ptr(grad_attn_data, grad_attn_reduced_data),
            kvBlockSize,
            static_cast<accum_t>(1),
            grad_k_data + i * grad_kStrideB + j * grad_kStrideH +
                n * grad_kStrideN,
            grad_kStrideN);
        }
      }
      // Move to the next query
      data_index_step(i, batchSize, j, num_head);
    }
  });
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

void flash_attention_kernel_impl(
    const Tensor& output,
    const Tensor& logsumexp,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    std::optional<Tensor> attn_mask,
    std::optional<double> scale,
    long q_zp,
    double q_scale,
    long k_zp,
    double k_scale,
    long v_zp,
    double v_scale,
    long a_zp,
    double a_scale,
    long o_zp,
    double o_scale) {
  auto q_seq_len = query.size(2);

  if (query.scalar_type() == kByte) {
    if (q_seq_len >= 768) {
      cpu_flash_attention_u8<unsigned char, 256, 64>(
        output, logsumexp, query, key, value,
        dropout_p, is_causal, attn_mask, scale,
        q_zp, q_scale,
        k_zp, k_scale,
        v_zp, v_scale,
        a_zp, a_scale,
        o_zp, o_scale);
    } else if (q_seq_len >= 192) {
      cpu_flash_attention_u8<unsigned char, 64, 64>(
        output, logsumexp, query, key, value,
        dropout_p, is_causal, attn_mask, scale,
        q_zp, q_scale,
        k_zp, k_scale,
        v_zp, v_scale,
        a_zp, a_scale,
        o_zp, o_scale);
    } else {
      cpu_flash_attention_u8<unsigned char, 32, 64>(
        output, logsumexp, query, key, value,
        dropout_p, is_causal, attn_mask, scale,
        q_zp, q_scale,
        k_zp, k_scale,
        v_zp, v_scale,
        a_zp, a_scale,
        o_zp, o_scale);
    }
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, query.scalar_type(), "flash_attention", [&] {
      if (!attn_mask.has_value()) {
        if (q_seq_len >= 768) {
          cpu_flash_attention<scalar_t, scalar_t, 256, 512>(
            output, logsumexp, query, key, value,
            dropout_p, is_causal, attn_mask, scale);
        } else if (q_seq_len >= 192) {
          cpu_flash_attention<scalar_t, scalar_t, 64, 512>(
            output, logsumexp, query, key, value,
            dropout_p, is_causal, attn_mask, scale);
        } else {
          cpu_flash_attention<scalar_t, scalar_t, 32, 512>(
            output, logsumexp, query, key, value,
            dropout_p, is_causal, attn_mask, scale);
        }
      } else {
        AT_DISPATCH_MASK_TYPES(attn_mask.value().scalar_type(), "flash_attention_mask", [&]() {
          if (q_seq_len >= 768) {
            cpu_flash_attention<scalar_t, mask_t, 256, 512>(
              output, logsumexp, query, key, value,
              dropout_p, is_causal, attn_mask, scale);
          } else if (q_seq_len >= 192) {
            cpu_flash_attention<scalar_t, mask_t, 64, 512>(
              output, logsumexp, query, key, value,
              dropout_p, is_causal, attn_mask, scale);
          } else {
            cpu_flash_attention<scalar_t, mask_t, 32, 512>(
              output, logsumexp, query, key, value,
              dropout_p, is_causal, attn_mask, scale);
          }
        });
      }
    });
  }
}

void flash_attention_backward_kernel_impl(
    const at::Tensor& grad_q,
    const at::Tensor& grad_k,
    const at::Tensor& grad_v,
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    double dropout_p,
    bool is_causal,
    std::optional<Tensor> attn_mask,
    std::optional<double> scale) {
  // make sure grad_out has no zero strides (broadcasted dimensions)
  // since we are going to call gemm next
  // zero stride in leading dimension would lead to slow impl for gemm
  auto grad_out_contig = grad_out.contiguous();
  auto q_seq_len = query.size(1);

  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, query.scalar_type(), "flash_attention_backward", [&] {
    if (!attn_mask.has_value() || !attn_mask.value().defined()) {
      using accum_t = at::opmath_type<scalar_t>;
      if (q_seq_len >= 768) {
        cpu_flash_attention_backward<scalar_t, accum_t, 256, 512>(
          grad_q, grad_k, grad_v, grad_out_contig,
          query, key, value, out, logsumexp,
          dropout_p, is_causal, attn_mask, scale);
      } else if (q_seq_len >= 192) {
        cpu_flash_attention_backward<scalar_t, accum_t, 64, 512>(
          grad_q, grad_k, grad_v, grad_out_contig,
          query, key, value, out, logsumexp,
          dropout_p, is_causal, attn_mask, scale);
      } else {
        cpu_flash_attention_backward<scalar_t, accum_t, 32, 512>(
          grad_q, grad_k, grad_v, grad_out_contig,
          query, key, value, out, logsumexp,
          dropout_p, is_causal, attn_mask, scale);
      }
    } else {
      AT_DISPATCH_MASK_TYPES(attn_mask.value().scalar_type(), "flash_attention_mask_backward", [&]() {
        if (q_seq_len >= 768) {
          cpu_flash_attention_backward<scalar_t, mask_t, 256, 512>(
            grad_q, grad_k, grad_v, grad_out_contig,
            query, key, value, out, logsumexp,
            dropout_p, is_causal, attn_mask, scale);
        } else if (q_seq_len >= 192) {
          cpu_flash_attention_backward<scalar_t, mask_t, 64, 512>(
            grad_q, grad_k, grad_v, grad_out_contig,
            query, key, value, out, logsumexp,
            dropout_p, is_causal, attn_mask, scale);
        } else {
          cpu_flash_attention_backward<scalar_t, mask_t, 32, 512>(
            grad_q, grad_k, grad_v, grad_out_contig,
            query, key, value, out, logsumexp,
            dropout_p, is_causal, attn_mask, scale);
        }
      });
    }
  });
}

} // anonymous namespace

ALSO_REGISTER_AVX512_DISPATCH(flash_attention_kernel, &flash_attention_kernel_impl);
ALSO_REGISTER_AVX512_DISPATCH(flash_attention_backward_kernel, &flash_attention_backward_kernel_impl);

} // at::native
