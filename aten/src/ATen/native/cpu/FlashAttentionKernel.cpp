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

#if AT_MKL_ENABLED()
#include <mkl.h>
#endif

namespace at::native {

namespace {

using fVec = vec::Vectorized<float>;
static auto f_vec_size = fVec::size();

inline void _store(
    float* dst,
    fVec src,
    int64_t l=fVec::size()) {
  src.store(dst, l);
}

inline void _store(
    at::BFloat16* dst,
    fVec src,
    int64_t l=fVec::size()) {
  auto res = vec::convert_float_bfloat16(src, src);
  res.store(dst, l);
}

inline void _mkl_gemm(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE transa,
    const CBLAS_TRANSPOSE transb,
    const int& m,
    const int& n,
    const int& k,
    const float& alpha,
    float* a,
    const int& lda,
    float* b,
    const int& ldb,
    const float& beta,
    float* c,
    const int& ldc) {
  cblas_sgemm(
      layout,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      a,
      lda,
      b,
      ldb,
      beta,
      c,
      ldc);
}

inline void _mkl_gemm(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE transa,
    const CBLAS_TRANSPOSE transb,
    const int& m,
    const int& n,
    const int& k,
    const float& alpha,
    at::BFloat16* a,
    const int& lda,
    at::BFloat16* b,
    const int& ldb,
    const float& beta,
    float* c,
    const int& ldc) {
  cblas_gemm_bf16bf16f32(
      layout,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      (MKL_BF16*)(a),
      lda,
      (MKL_BF16*)(b),
      ldb,
      beta,
      c,
      ldc);
}

inline void _exp_reduce_sum_fusion_kernel(
    float* a,
    const int& size,
    float* out,
    float& val) {
  vec::map<float>(
    [val](fVec x) { return (x - fVec(val)).exp(); }, out, a, size);
  val = vec::reduce_all<float>(
    [](fVec& x, fVec& y) { return x + y; }, out, size);
}

template <typename scalar_t>
inline void _normalization_kernel(
    const float* a,
    const float& sum,
    const int& size,
    scalar_t* out) {
  auto vec_sum = fVec(sum);
  int64_t i = 0;
  for (i = 0; i < f_vec_size * (size / f_vec_size); i += f_vec_size) {
    auto tmp0 = fVec::loadu(a + i);
    auto tmp1 = tmp0 / vec_sum;
    _store(out + i, tmp1);
  }
  if (size - i > 0) {
    auto tmp0 = fVec::loadu(a + i, size - i);
    auto tmp1 = tmp0 / vec_sum;
    _store(out + i, tmp1, size - i);
  }
}

inline void _mul_reduce_max_fusion_kernel(
    const float* a,
    const float& scale,
    const int& size,
    float* out,
    float& max) {
  vec::map<float>(
    [scale](fVec x) { return x * fVec(scale); }, out, a, size);
  max = vec::reduce_all<float>(
    [](fVec& x, fVec& y) { return vec::maximum(x, y); }, out, size);
}

inline void _init_mha_buffer_kernel(float* max, float* sum, const int& size) {
  float tmp_max = -std::numeric_limits<float>::infinity();
  auto vec_tmp_max = fVec(tmp_max);
  float tmp_zero = 0;
  auto vec_tmp_zero = fVec(tmp_zero);
  int64_t i = 0;
  for (i = 0; i < f_vec_size * (size / f_vec_size); i += f_vec_size) {
    _store(max + i, vec_tmp_max);
    _store(sum + i, vec_tmp_zero);
  }
  if (size - i > 0) {
    _store(max + i, vec_tmp_max, size - i);
    _store(sum + i, vec_tmp_zero, size - i);
  }
}

/**
 * This kernel is used to reorder the MHA output
 * with strides.
 * src: MKL GEMM output buffer
 * dst: Final MHA output
 */
template <typename scalar_t>
inline void _reorder_mha_output_kernel(
    float* src,
    scalar_t* dst,
    const int& rows,
    const int& cols,
    const int& dst_stride) {
  for (int64_t i = 0; i < rows; ++i) {
    int64_t j = 0;
    for (j = 0; j < f_vec_size * (cols / f_vec_size); j += f_vec_size) {
      auto tmp0 = fVec::loadu(src + i * cols + j);
      _store(dst + i * dst_stride + j, tmp0);
    }
    if (cols - j > 0) {
      auto tmp0 = fVec::loadu(src + i * cols + j, cols - j);
      _store(dst + i * dst_stride + j, tmp0, cols - j);
    }
  }
}

/**
 * This kernel is used to update the MHA output with the latest MAX
 * and SUM values block by block.
 * exp_val: exp(max_old - max_new)
 * In the i th block, the softmax(qk - i th) * v - i th was calculated
 * with the old MAX and SUM values, max_old and sum_old. When moving to
 * the i + 1 th block, since softmax(qk - i + 1 th) will be calculated
 * with the new MAX and SUM values, max_new and sum_new, thus the MHA
 * buffer which stores the summation of blocked softmax(qk) * v should
 * be also updated using max_new and sum_new:
 * a = a * sum_old / sum_new
 * a = a * exp(max_old) / exp(max_new) = a * exp_val
 */
inline void _mha_update_sum_max_kernel(
    const float* a,
    const float& sum_old,
    const float& sum_new,
    const float& exp_val,
    const int& size,
    float* out) {
  float sum_cor = sum_old / sum_new;
  vec::map<float>(
    [sum_cor, exp_val](fVec x)
      { return x * fVec(sum_cor) * fVec(exp_val); },
      out, a, size);
}

const std::vector<int64_t> qsplit_range{767, 191, 31};
const std::vector<int64_t> qsplit_size{256, 64, 32};
const int64_t kvsplit_size = 512;

template <typename scalar_t>
at::ScalarType ConvertTypeToKType() {
  if (std::is_same<scalar_t, at::BFloat16>::value) {
    return at::kBFloat16;
  } else if (std::is_same<scalar_t, float>::value) {
    return at::kFloat;
  }
}

template <typename scalar_t>
void _mha_mul_softmax_kernel(
    float* a,
    scalar_t* b,
    float* dst,
    float* max,
    float* sum,
    const float& scale,
    const int& qsize,
    const int& kvsize,
    const int& headsize) {
  float tmp_max = 0.f, tmp_sum = 0.f, sum_old = 0.f, exp_tmp = 0.f;

  for (int i = 0; i < qsize; ++i) {
    sum_old = sum[i];

    _mul_reduce_max_fusion_kernel(
        a + i * kvsize, scale, kvsize, a + i * kvsize, tmp_max);

    tmp_max = max[i] > tmp_max ? max[i] : tmp_max;

    tmp_sum = tmp_max;
    _exp_reduce_sum_fusion_kernel(
        a + i * kvsize, kvsize, a + i * kvsize, tmp_sum);
    exp_tmp = exp(max[i] - tmp_max);
    sum[i] = tmp_sum + exp_tmp * sum[i];
    max[i] = tmp_max;

    _normalization_kernel<scalar_t>(
        a + i * kvsize, sum[i], kvsize, b + i * kvsize);

    _mha_update_sum_max_kernel(
        dst + i * headsize,
        sum_old,
        sum[i],
        exp_tmp,
        headsize,
        dst + i * headsize);
  }
}

template <typename scalar_t>
void sd_mha_base_kernel(
    scalar_t* output,
    scalar_t* query,
    scalar_t* key,
    scalar_t* value,
    const int64_t& qStride,
    const int64_t& kStride,
    const int64_t& vStride,
    const int64_t& batchSize,
    const int64_t& qSize,
    const int64_t& kvSize,
    const int64_t& num_head,
    const int64_t& headSize,
    const int64_t& hiddenSize,
    const float& scale) {

  int64_t qSplitSize = qSize;
  for (size_t i = 0; i < qsplit_range.size(); ++i) {
    if (qSize > qsplit_range[i]) {
      qSplitSize = qsplit_size[i];
      break;
    }
  }
  int64_t kvSplitSize = kvSize >= kvsplit_size ? kvsplit_size : kvSize;

  int64_t qSlice = (qSize - 1) / qSplitSize + 1;
  int64_t qTail = (qSize - 1) % qSplitSize + 1;
  int64_t kvSlice = (kvSize - 1) / kvSplitSize + 1;
  int64_t kvTail = (kvSize - 1) % kvSplitSize + 1;

  int64_t num_thread = at::get_num_threads();

  at::Tensor qk = at::empty({num_thread, qSplitSize, kvSplitSize}, at::kFloat);
  at::Tensor qk_norm = at::empty(
      {num_thread, qSplitSize, kvSplitSize}, ConvertTypeToKType<scalar_t>());
  at::Tensor qk_max = at::empty({num_thread, qSplitSize}, at::kFloat);
  at::Tensor qk_sum = at::empty({num_thread, qSplitSize}, at::kFloat);
  at::Tensor dst_fp32 =
      at::empty({num_thread, qSplitSize, headSize}, at::kFloat);

  at::parallel_for(0, batchSize * num_head * qSlice, 0, [&](int64_t begin, int64_t end) {
    int64_t i = 0;
    int64_t j = 0;
    int64_t k = 0;
    data_index_init(begin, i, batchSize, j, num_head, k, qSlice);

    for (int64_t x = begin; x < end; x++) {
      int qBlockSize = (k == qSlice - 1) ? qTail : qSplitSize;
      int ompIdx = at::get_thread_num();
      _init_mha_buffer_kernel(
          qk_max.data_ptr<float>() + ompIdx * qSplitSize,
          qk_sum.data_ptr<float>() + ompIdx * qSplitSize,
          qBlockSize);
      for (int l = 0; l < kvSlice; ++l) {
        int kvBlockSize = (l == kvSlice - 1) ? kvTail : kvSplitSize;
        _mkl_gemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            qBlockSize,
            kvBlockSize,
            headSize,
            1.f,
            query + i * qSize * qStride + headSize * j +
                k * qSplitSize * qStride,
            qStride,
            key + i * kvSize * kStride + headSize * j +
                l * kvSplitSize * kStride,
            kStride,
            0.f,
            qk.data_ptr<float>() + ompIdx * qSplitSize * kvSplitSize,
            kvBlockSize);
        _mha_mul_softmax_kernel<scalar_t>(
            qk.data_ptr<float>() + ompIdx * qSplitSize * kvSplitSize,
            qk_norm.data_ptr<scalar_t>() + ompIdx * qSplitSize * kvSplitSize,
            dst_fp32.data_ptr<float>() + ompIdx * qSplitSize * headSize,
            qk_max.data_ptr<float>() + ompIdx * qSplitSize,
            qk_sum.data_ptr<float>() + ompIdx * qSplitSize,
            scale,
            qBlockSize,
            kvBlockSize,
            headSize);
        _mkl_gemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            qBlockSize,
            headSize,
            kvBlockSize,
            1.f,
            qk_norm.data_ptr<scalar_t>() + ompIdx * qSplitSize * kvSplitSize,
            kvBlockSize,
            value + i * kvSize * vStride + headSize * j +
                l * kvSplitSize * vStride,
            vStride,
            l == 0 ? 0.f : 1.f,
            dst_fp32.data_ptr<float>() + ompIdx * qSplitSize * headSize,
            headSize);
      }
      _reorder_mha_output_kernel<scalar_t>(
          dst_fp32.data_ptr<float>() + ompIdx * qSplitSize * headSize,
          output + i * qSize * hiddenSize +
              headSize * j + k * qSplitSize * hiddenSize,
          qBlockSize,
          headSize,
          hiddenSize);
      data_index_step(i, batchSize, j, num_head, k, qSlice);
    }
  });

}

// Return the first output with other outputs arbitrary because they are used
// for backward.
// TODO: Calculate other outputs when adding cpu backward.
// template <typename scalar_t>
void cpu_flash_attention(
    const Tensor& output,
    const Tensor& logsumexp,
    const Tensor& cum_seq_q,
    const Tensor& cum_seq_k,
    int64_t& max_q,
    int64_t& max_k,
    const Tensor& philox_seed,
    const Tensor& philox_offset,
    const Tensor& debug_attn_mask,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    c10::optional<double> scale) {
  // // Check validation
  // TORCH_CHECK(at::hasMKL());
  // TORCH_CHECK(query.scalar_type() == at::kFloat
  //     || query.scalar_type() == at::kBFloat16);
  // TORCH_CHECK(query.stride(1) == query.size(-1));
  // TORCH_CHECK(query.dim() == 4);
  // TORCH_CHECK(dropout_p == 0.0);
  // TORCH_CHECK(!is_causal);

  // Query (Batch x Num_heads x Q_seq_len  x Dim_per_head)
  // Key   (Batch x Num_heads x KV_seq_len x Dim_per_head)
  // Value (Batch x Num_heads x KV_seq_len x Dim_per_head)
  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(2);
  int64_t kvSize = value.size(2);
  int64_t num_head = query.size(1);
  int64_t headSize = query.size(3);
  int64_t hiddenSize = num_head * headSize;

  // Query -> Query(Batch x Q_seq_len x hiddenSize)
  // Key   -> Key(Batch x KV_seq_len x hiddenSize)
  // Value -> Value(Batch x KV_seq_len x hiddenSize)
  Tensor q = query.transpose(1, 2).reshape(
      {batchSize, qSize, hiddenSize});
  Tensor k = key.transpose(1, 2).reshape(
      {batchSize, kvSize, hiddenSize});
  Tensor v = value.transpose(1, 2).reshape(
      {batchSize, kvSize, hiddenSize});

  int64_t qStride = q.size(-1);
  int64_t kStride = k.size(-1);
  int64_t vStride = v.size(-1);

  const auto softmax_scale =
      sdp::calculate_scale(query, scale).as_float_unchecked();

  if (query.scalar_type() == at::kFloat) {
    sd_mha_base_kernel<float>(
        output.data_ptr<float>(),
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        qStride,
        kStride,
        vStride,
        batchSize,
        qSize,
        kvSize,
        num_head,
        headSize,
        hiddenSize,
        softmax_scale);
  } else if (query.scalar_type() == at::kBFloat16) {
    sd_mha_base_kernel<at::BFloat16>(
        output.data_ptr<at::BFloat16>(),
        q.data_ptr<at::BFloat16>(),
        k.data_ptr<at::BFloat16>(),
        v.data_ptr<at::BFloat16>(),
        qStride,
        kStride,
        vStride,
        batchSize,
        qSize,
        kvSize,
        num_head,
        headSize,
        hiddenSize,
        softmax_scale);
  }
}

void flash_attention_kernel_impl(
    const Tensor& output,
    const Tensor& logsumexp,
    const Tensor& cum_seq_q,
    const Tensor& cum_seq_k,
    int64_t& max_q,
    int64_t& max_k,
    const Tensor& philox_seed,
    const Tensor& philox_offset,
    const Tensor& debug_attn_mask,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    c10::optional<double> scale) {
//   AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "flash_attention", [&] {
//     cpu_flash_attention<scalar_t>(output, logsumexp, cum_seq_q, cum_seq_k,
//         max_q, max_k, philox_seed, philox_offset, debug_attn_mask,
//         query, key, value, dropout_p, is_causal, return_debug_mask, scale);
//   });
  cpu_flash_attention(output, logsumexp, cum_seq_q, cum_seq_k,
      max_q, max_k, philox_seed, philox_offset, debug_attn_mask,
      query, key, value, dropout_p, is_causal, return_debug_mask, scale);
}

} // anonymous namespace

REGISTER_DISPATCH(flash_attention_kernel, &flash_attention_kernel_impl);

} // at::native