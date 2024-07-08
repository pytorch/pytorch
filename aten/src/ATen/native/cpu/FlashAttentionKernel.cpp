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

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

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
    std::optional<double> scale) {
  auto q_seq_len = query.size(2);

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
