#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/TensorAccessor.h>
#include <ATen/native/DispatchStub.h>
#include <c10/macros/Export.h>
#include <c10/util/Logging.h>
#include <c10/util/Optional.h>
#include <c10/util/bit_cast.h>

#include <ATen/native/nested/NestedTensorTransformerFunctions.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <ATen/native/transformers/sdp_utils.h>

namespace at {

namespace native {

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
    const int& headsize);

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
    const float& scale);

std::tuple<
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    int64_t,
    int64_t,
    Tensor,
    Tensor,
    Tensor>
_scaled_dot_product_flash_attention_cpu(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    float scale);

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
    c10::optional<double> scale);


} // namespace native
} // namespace at
