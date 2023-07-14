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

template <typename scalar_t>
static inline scalar_t* conditional_data_ptr(scalar_t* ptr, scalar_t* ptr2) {
  TORCH_INTERNAL_ASSERT(ptr2 == nullptr);
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

template <typename scalar_t, int64_t qSplitSize, int64_t kvSplitSize>
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
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    c10::optional<double> scale) {
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
  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(1);
  int64_t kvSize = value.size(1);
  int64_t num_head = query.size(2);
  int64_t headSize = query.size(3);

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

  int64_t qSlice = (qSize - 1) / qSplitSize + 1;
  int64_t num_thread = at::get_num_threads();

  const auto dtype = query.scalar_type();
  const auto accumulate_dtype = toOpMathType(dtype);

  at::Tensor qk = at::empty({num_thread, qSplitSize, kvSplitSize}, query.options().dtype(accumulate_dtype));
  at::Tensor qk_reduced = at::empty({num_thread, qSplitSize, is_reduced_type ? kvSplitSize : 0}, query.options());
  at::Tensor qk_max = at::empty({num_thread, qSplitSize}, query.options().dtype(accumulate_dtype));
  at::Tensor qk_sum = at::empty({num_thread, qSplitSize}, query.options().dtype(accumulate_dtype));
  at::Tensor dst = at::empty({num_thread, qSplitSize, headSize}, query.options().dtype(accumulate_dtype));

  // Data ptrs
  scalar_t* q_data = query.data_ptr<scalar_t>();
  scalar_t* k_data = key.data_ptr<scalar_t>();
  scalar_t* v_data = value.data_ptr<scalar_t>();
  scalar_t* out_data = output.data_ptr<scalar_t>();
  accum_t* lse_data = logsumexp.data_ptr<accum_t>();
  accum_t* qk_data = qk.data_ptr<accum_t>();
  scalar_t* qk_reduced_data = is_reduced_type ? qk_reduced.data_ptr<scalar_t>() : nullptr;
  accum_t* qk_max_data = qk_max.data_ptr<accum_t>();
  accum_t* qk_sum_data = qk_sum.data_ptr<accum_t>();
  accum_t* dst_data = dst.data_ptr<accum_t>();

  at::parallel_for(0, batchSize * num_head * qSlice, 1, [&](int64_t begin, int64_t end) {
    int64_t i = 0, j = 0, k = 0;
    data_index_init(begin, i, batchSize, j, num_head, k, qSlice);
    int ompIdx = at::get_thread_num();
    for (const auto x : c10::irange(begin, end)) {
      (void)x; // Suppress unused variable
      int64_t m = k * qSplitSize;
      int64_t qBlockSize = std::min(qSplitSize, qSize - m);
      // Initialize max and sum
      fill_stub(qk_max_data + ompIdx * qSplitSize,
          -std::numeric_limits<accum_t>::infinity(), qBlockSize);
      fill_stub(qk_sum_data + ompIdx * qSplitSize,
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
            scaling_factor,
            k_data + i * kStrideB + j * kStrideH +
                n * kStrideN,
            kStrideN,
            q_data + i * qStrideB + j * qStrideH +
                m * qStrideM,
            qStrideM,
            static_cast<accum_t>(0),
            qk_data + ompIdx * qSplitSize * kvSplitSize,
            kvBlockSize);
        // Apply causal mask, fill unused with -inf
        if (is_causal && num_keys - n <= kvSplitSize) {
          for (const auto row : c10::irange(qBlockSize)) {
            int64_t last_col = m + row - n;
            accum_t* row_ptr = qk_data + ompIdx * qSplitSize * kvSplitSize + row * kvBlockSize;
            fill_stub(row_ptr + last_col + 1,
                -std::numeric_limits<accum_t>::infinity(),
                kvBlockSize - last_col - 1);
          }
        }
        // Update coefficients with Softmax
        accum_t tmp_max = 0, tmp_sum = 0, sum_old = 0, exp_tmp = 0;
        accum_t* qk_block = qk_data + ompIdx * qSplitSize * kvSplitSize;
        scalar_t* qk_reduced_block = is_reduced_type ? qk_reduced_data + ompIdx * qSplitSize * kvSplitSize : nullptr;
        accum_t* dst_block = dst_data + ompIdx * qSplitSize * headSize;
        accum_t* max_block = qk_max_data + ompIdx * qSplitSize;
        accum_t* sum_block = qk_sum_data + ompIdx * qSplitSize;
        for (int64_t row = 0; row < qBlockSize; ++row) {
          sum_old = sum_block[row];
          // max per row
          tmp_max = vec::reduce_all<accum_t>(
            [](Vec& x, Vec& y) { return vec::maximum(x, y); },
            qk_block + row * kvBlockSize, kvBlockSize);
          tmp_max = max_block[row] > tmp_max ? max_block[row] : tmp_max;
          // qk <- exp(qk - max)
          vec::map<accum_t>(
            [tmp_max](Vec x) { return (x - Vec(tmp_max)).exp(); },
            qk_block + row * kvBlockSize, qk_block + row * kvBlockSize, kvBlockSize);
          // sum per row
          tmp_sum = vec::reduce_all<accum_t>(
            [](Vec& x, Vec& y) { return x + y; },  qk_block + row * kvBlockSize, kvBlockSize);
          // exp_tmp <- exp(max[row] - max)
          exp_tmp = std::exp(max_block[row] - tmp_max);
          // sum[row] <- sum + exp_tmp * sum[row]
          sum_block[row] = tmp_sum + exp_tmp * sum_block[row];
          // max[row] <- max
          max_block[row] = tmp_max;
          // qk <- qk / sum[row]
          accum_t sum_new = sum_block[row];
          vec::map<accum_t>(
            [sum_new](Vec x) { return x / Vec(sum_new); },
            qk_block + row * kvBlockSize, qk_block + row * kvBlockSize, kvBlockSize);
          if (is_reduced_type) {
            convert<accum_t, scalar_t>(
              qk_block + row * kvBlockSize,
              qk_reduced_block + row * kvBlockSize,
              kvBlockSize);
          }
          // dst <- dst * sum_old / sum_new * exp_tmp
          if (n > 0) {
            accum_t sum_cor = sum_old / sum_new;
            vec::map<accum_t>(
              [sum_cor, exp_tmp](Vec x)
              { return x * Vec(sum_cor) * Vec(exp_tmp); },
              dst_block + row * headSize, dst_block + row * headSize, headSize);
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
            conditional_data_ptr(qk_block, qk_reduced_block),
            kvBlockSize,
            n == 0 ? static_cast<accum_t>(0) : static_cast<accum_t>(1),
            dst_block,
            headSize);
      }
      // reorder MHA output with strides
      for (int64_t row = 0; row < qBlockSize; ++row) {
        vec::map<scalar_t>(
          [](Vec x) { return x; },
          out_data + i * oStrideB + j * oStrideH + m * oStrideM + row * oStrideM,
          dst_data + ompIdx * qSplitSize * headSize + row * headSize,
          headSize);
      }
      // Store logsumexp for backward
      accum_t* lse_ptr = lse_data + i * lStrideB + j * lStrideH + m * lStrideM;
      for (const auto row : c10::irange(qBlockSize)) {
        lse_ptr[row * lStrideM] = qk_max_data[ompIdx * qSplitSize + row]
            + std::log(qk_sum_data[ompIdx * qSplitSize + row]);
      }
      // Move to the next query
      data_index_step(i, batchSize, j, num_head, k, qSlice);
    }
  });

}

template <typename scalar_t, int64_t qSplitSize, int64_t kvSplitSize>
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
    const Tensor& cumulative_sequence_length_q,
    const Tensor& cumulative_sequence_length_k,
    const int64_t max_seqlen_batch_q,
    const int64_t max_seqlen_batch_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    c10::optional<double> scale) {
  constexpr bool is_reduced_type = is_reduced_floating_point_v<scalar_t>;
  using accum_t = at::opmath_type<scalar_t>;
  using Vec = vec::Vectorized<accum_t>;
  accum_t scaling_factor =
      sdp::calculate_scale(query, scale).as_float_unchecked();

  // Sizes
  // Query (Batch x Q_seq_len  x Num_heads x Dim_per_head)
  // Key   (Batch x KV_seq_len x Num_heads x Dim_per_head)
  // Value (Batch x KV_seq_len x Num_heads x Dim_per_head)
  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(1);
  int64_t kvSize = value.size(1);
  int64_t num_head = query.size(2);
  int64_t headSize = query.size(3);

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

  int64_t num_thread = at::get_num_threads();

  const auto dtype = query.scalar_type();
  const auto accumulate_dtype = toOpMathType(dtype);

  at::Tensor attn = at::empty({num_thread, qSplitSize, kvSplitSize}, query.options().dtype(accumulate_dtype));
  at::Tensor attn_reduced = at::empty({num_thread, qSplitSize, is_reduced_type ? kvSplitSize : 0}, query.options());
  at::Tensor grad_attn = at::empty({num_thread, qSplitSize, kvSplitSize}, query.options().dtype(accumulate_dtype));
  at::Tensor grad_attn_reduced = at::empty({num_thread, qSplitSize, is_reduced_type ? kvSplitSize : 0}, query.options());

  scalar_t* grad_q_data = grad_q.data_ptr<scalar_t>();
  scalar_t* grad_k_data = grad_k.data_ptr<scalar_t>();
  scalar_t* grad_v_data = grad_v.data_ptr<scalar_t>();
  scalar_t* grad_out_data = grad_out.data_ptr<scalar_t>();
  scalar_t* q_data = query.data_ptr<scalar_t>();
  scalar_t* k_data = key.data_ptr<scalar_t>();
  scalar_t* v_data = value.data_ptr<scalar_t>();
  scalar_t* out_data = out.data_ptr<scalar_t>();
  accum_t* lse_data = logsumexp.data_ptr<accum_t>();
  accum_t* attn_data = attn.data_ptr<accum_t>();
  scalar_t* attn_reduced_data = is_reduced_type ? attn_reduced.data_ptr<scalar_t>() : nullptr;
  accum_t* grad_attn_data = grad_attn.data_ptr<accum_t>();
  scalar_t* grad_attn_reduced_data = is_reduced_type ? grad_attn_reduced.data_ptr<scalar_t>() : nullptr;

  at::parallel_for(0, batchSize * num_head, 1, [&](int64_t begin, int64_t end) {
    int64_t i = 0, j = 0;
    data_index_init(begin, i, batchSize, j, num_head);
    int ompIdx = at::get_thread_num();
    accum_t dsum[qSplitSize];
    accum_t* attn_block = attn_data + ompIdx * qSplitSize * kvSplitSize;
    scalar_t* attn_reduced_block = is_reduced_type ? attn_reduced_data + ompIdx * qSplitSize * kvSplitSize : nullptr;
    accum_t* grad_attn_block = grad_attn_data + ompIdx * qSplitSize * kvSplitSize;
    scalar_t* grad_attn_reduced_block = is_reduced_type ? grad_attn_reduced_data + ompIdx * qSplitSize * kvSplitSize : nullptr;
    for (const auto x : c10::irange(begin, end)) {
      (void)x; // Suppress unused variable
      // rowsum of grad_out * out
      for (int64_t m = 0; m < qSize; m += qSplitSize) {
        int64_t qBlockSize = std::min(qSplitSize, qSize - m);
        // dsum <- rowsum(grad_out * out)
        for (const auto row : c10::irange(qBlockSize)) {
          dsum[row] = vec::map2_reduce_all<scalar_t>(
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
            attn_block,
            kvBlockSize);
          // restore self attention after softmax from logsumexp
          // attn <- exp(attn - normalizer)
          for (const auto row : c10::irange(qBlockSize)) {
            accum_t normalizer = lse_data[i * lStrideB + j * lStrideH + (m + row) * lStrideM];
            vec::map<accum_t>(
              [normalizer](Vec x) { return (x - Vec(normalizer)).exp(); },
              attn_block + row * kvBlockSize,
              attn_block + row * kvBlockSize,
              kvBlockSize);
          }
          // Apply causal mask, filled unused with 0
          if (is_causal && num_keys - n <= kvSplitSize) {
            for (const auto row : c10::irange(qBlockSize)) {
              int64_t last_col = m + row - n;
              accum_t* row_ptr = attn_block + row * kvBlockSize;
              fill_stub(row_ptr + last_col + 1, static_cast<accum_t>(0), kvBlockSize - last_col - 1);
            }
          }
          if (is_reduced_type) {
            for (const auto row : c10::irange(qBlockSize)) {
              convert<accum_t, scalar_t>(
                attn_block + row * kvBlockSize,
                attn_reduced_block + row * kvBlockSize,
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
            conditional_data_ptr(attn_block, attn_reduced_block),
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
            grad_attn_block,
            kvBlockSize);
          // grad_attn <- attn * (grad_attn - dsum)
          for (const auto row : c10::irange(qBlockSize)) {
            accum_t d = dsum[row];
            vec::map2<accum_t>(
              [d](Vec attn, Vec grad_attn) { return attn * (grad_attn - Vec(d)); },
              grad_attn_block + row * kvBlockSize,
              attn_block + row * kvBlockSize,
              grad_attn_block + row * kvBlockSize,
              kvBlockSize);
          }
          if (is_reduced_type) {
            for (const auto row : c10::irange(qBlockSize)) {
              convert<accum_t, scalar_t>(
                grad_attn_block + row * kvBlockSize,
                grad_attn_reduced_block + row * kvBlockSize,
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
            conditional_data_ptr(grad_attn_block, grad_attn_reduced_block),
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
            conditional_data_ptr(grad_attn_block, grad_attn_reduced_block),
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
  AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, query.scalar_type(), "flash_attention", [&] {
    cpu_flash_attention<scalar_t, 128, 256>(
        output, logsumexp, cum_seq_q, cum_seq_k,
        max_q, max_k, philox_seed, philox_offset, debug_attn_mask,
        query, key, value, dropout_p, is_causal, return_debug_mask, scale);
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
    const Tensor& cum_seq_q,
    const Tensor& cum_seq_k,
    const int64_t max_q,
    const int64_t max_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    c10::optional<double> scale) {
  // make sure grad_out has no zero strides (broadcasted dimensions)
  // since we are going to call gemm next
  // zero stride in leading dimension would lead to slow impl for gemm
  auto grad_out_contig = grad_out.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, query.scalar_type(), "flash_attention_backward", [&] {
    cpu_flash_attention_backward<scalar_t, 128, 256>(
        grad_q, grad_k, grad_v, grad_out_contig,
        query, key, value, out, logsumexp,
        cum_seq_q, cum_seq_k, max_q, max_k, dropout_p,
        is_causal, philox_seed, philox_offset, scale);
  });
}

} // anonymous namespace

REGISTER_DISPATCH(flash_attention_kernel, &flash_attention_kernel_impl);
REGISTER_DISPATCH(flash_attention_backward_kernel, &flash_attention_backward_kernel_impl);

} // at::native
