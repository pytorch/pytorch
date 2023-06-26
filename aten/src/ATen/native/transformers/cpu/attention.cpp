#include <ATen/native/transformers/cpu/attention.h>
#include <ATen/native/transformers/cpu/add_softmax.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIndexing.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/DispatchStub.h>
#include <c10/core/DeviceType.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <type_traits>
#include <utility>
#if AT_MKL_ENABLED()
#include <mkl.h>

namespace at {
namespace native {

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
at::Tensor sd_mha_base_kernel(
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
  at::Tensor output =
      at::empty({batchSize, qSize, hiddenSize}, ConvertTypeToKType<scalar_t>());

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

    for (int64_t l = begin; l < end; l++) {
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
          output.data_ptr<scalar_t>() + i * qSize * hiddenSize +
              headSize * j + k * qSplitSize * hiddenSize,
          qBlockSize,
          headSize,
          hiddenSize);
      data_index_step(i, batchSize, j, num_head, k, qSlice);
    }
  });

  return output;
}

// Return the first output with other outputs arbitrary because they are used
// for backward.
// TODO: Calculate other outputs when adding cpu backward.
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    int64_t,
    int64_t,
    at::Tensor,
    at::Tensor,
    at::Tensor>
_scaled_dot_product_flash_attention_cpu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    c10::optional<double> scale) {
  // Check validation
  TORCH_CHECK(at::hasMKL());
  TORCH_CHECK(query.scalar_type() == at::kFloat
      || query.scalar_type() == at::kBFloat16);
  TORCH_CHECK(query.stride(1) == query.size(-1));
  TORCH_CHECK(query.dim() == 4);
  TORCH_CHECK(dropout_p == 0.0);
  TORCH_CHECK(!is_causal);

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
  Tensor query_reshaped = query.transpose(1, 2).reshape(
      {batchSize, qSize, hiddenSize});
  Tensor key_reshaped = key.transpose(1, 2).reshape(
      {batchSize, kvSize, hiddenSize});
  Tensor value_reshaped = value.transpose(1, 2).reshape(
      {batchSize, kvSize, hiddenSize});

  int64_t qStride = query_reshaped.size(-1);
  int64_t kStride = key_reshaped.size(-1);
  int64_t vStride = value_reshaped.size(-1);

  const auto softmax_scale =
      sdp::calculate_scale(query, scale).as_float_unchecked();
  auto s = std::to_string(scale.has_value() ? scale.value() : 1.0);

  at::Tensor output;
  if (query.scalar_type() == at::kFloat) {
    output = sd_mha_base_kernel<float>(
        query_reshaped.data_ptr<float>(),
        key_reshaped.data_ptr<float>(),
        value_reshaped.data_ptr<float>(),
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
    output = sd_mha_base_kernel<at::BFloat16>(
        query_reshaped.data_ptr<at::BFloat16>(),
        key_reshaped.data_ptr<at::BFloat16>(),
        value_reshaped.data_ptr<at::BFloat16>(),
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

  output =
      output.view({batchSize, qSize, num_head, headSize}).transpose(1, 2);

  return std::make_tuple(
      output,
      Tensor(),
      Tensor(),
      Tensor(),
      0,
      0,
      Tensor(),
      Tensor(),
      Tensor());
}

// Backward to be implemented
std::tuple<at::Tensor, at::Tensor, at::Tensor> _scaled_dot_product_flash_attention_backward_cpu(
    const at::Tensor& grad_out_,
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
    c10::optional<double> scale){
  TORCH_CHECK(false, "FLASH ATTENTION BACKWARD CPU is not implemented now.");
  return std::make_tuple(Tensor(), Tensor(), Tensor());
}

} // namespace native
} // namespace at

#endif // AT_MKL_ENABLED
