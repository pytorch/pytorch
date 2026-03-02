#include <ATen/OpMathType.h>
#include <ATen/native/mkldnn/xpu/detail/Attr.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <oneapi/dnnl/dnnl.hpp>

namespace {

using namespace at::native::onednn;
using logical_tensor = dnnl::graph::logical_tensor;
using data_type = logical_tensor::data_type;
using dims = logical_tensor::dims;
using op = dnnl::graph::op;
using partition = dnnl::graph::partition;

constexpr logical_tensor::data_type sdpa_intermediate_dtype =
    logical_tensor::data_type::f32;

inline data_type to_logical_tensor_data_type(c10::ScalarType scalar_type) {
  return scalar_type == c10::ScalarType::Float   ? data_type::f32
      : scalar_type == c10::ScalarType::Half     ? data_type::f16
      : scalar_type == c10::ScalarType::BFloat16 ? data_type::bf16
                                                 : data_type::undef;
}

namespace sdpa_forward {

struct SDPALogicalParams {
  enum class TensorID {
    query,
    key,
    scale,
    neg_inf,
    attn_mask,
    value,
    attention,
    logsumexp,
    end,
  };

  logical_tensor query{};
  logical_tensor key{};
  logical_tensor scale{};
  std::optional<logical_tensor> neg_inf;
  std::optional<logical_tensor> attn_mask;
  logical_tensor value{};
  logical_tensor attention{};
  std::optional<logical_tensor> logsumexp;

  SDPALogicalParams(
      const at::Tensor& query_,
      const at::Tensor& key_,
      const at::Tensor& value_,
      const std::optional<at::Tensor>& attn_mask_,
      const at::Tensor& attention_,
      const at::Tensor& logsumexp_,
      int batch_size,
      int seq_len_q,
      int seq_len_kv,
      int num_head_q,
      int num_head_kv,
      int head_dim_qk,
      int head_dim_v,
      bool is_causal,
      bool compute_logsumexp) {
    const data_type dtype = to_logical_tensor_data_type(query_.scalar_type());
    TORCH_INTERNAL_ASSERT(
        (dtype != data_type::undef),
        "Only FP16/BF16/FP32 datatypes are currently supported");
    TORCH_INTERNAL_ASSERT(
        query_.scalar_type() == attention_.scalar_type(),
        "scaled_dot_product_attention_xpu: query and attention tensors should have the same data type.");

    at::Tensor reshaped_query = query_;
    at::Tensor reshaped_key = key_;
    at::Tensor reshaped_value = value_;
    at::Tensor reshaped_attention = attention_;
    at::Tensor reshaped_logsumexp =
        compute_logsumexp ? logsumexp_.unsqueeze(-1) : logsumexp_;
    at::Tensor reshaped_attn_mask = attn_mask_.value_or(at::Tensor());

    // handle broadcasted input tensors for OneDNN
    if (at::native::onednn::is_broadcast(reshaped_query)) {
      at::native::onednn::undo_broadcast(reshaped_query);
    }
    if (at::native::onednn::is_broadcast(reshaped_key)) {
      at::native::onednn::undo_broadcast(reshaped_key);
    }
    if (at::native::onednn::is_broadcast(reshaped_value)) {
      at::native::onednn::undo_broadcast(reshaped_value);
    }
    if (attn_mask_.has_value() &&
        at::native::onednn::is_broadcast(reshaped_attn_mask)) {
      at::native::onednn::undo_broadcast(reshaped_attn_mask);
    }

    if (num_head_q != num_head_kv) { // Check whether the attention is a
                                     // Grouped-Query Attention (GQA)
      int group_num = num_head_kv;
      int group_size = num_head_q / num_head_kv;
      // oneDNN requires the shape of the query tensor to be represented as
      // [batch_size, num_head_q / num_head_kv, num_head_kv, seq_len_q,
      // head_dim_qk]. Please refer to
      // https://uxlfoundation.github.io/oneDNN/dev_guide_graph_gqa.html#gqa-pattern
      reshaped_query = query_.view(
          {batch_size, group_num, group_size, seq_len_q, head_dim_qk});
      reshaped_key = key_.unsqueeze(2);
      reshaped_value = value_.unsqueeze(2);
      reshaped_attention = attention_.view(
          {batch_size, group_num, group_size, seq_len_q, head_dim_v});
      if (attn_mask_.has_value() && attn_mask_.value().dim() == 4) {
        reshaped_attn_mask = attn_mask_.value().unsqueeze(2);
      }
    }

#define LOGIC_TENSOR_DESC(name, dtype)     \
  name = {                                 \
      static_cast<size_t>(TensorID::name), \
      dtype,                               \
      reshaped_##name.sizes().vec(),       \
      reshaped_##name.strides().vec()}

    LOGIC_TENSOR_DESC(query, dtype);
    LOGIC_TENSOR_DESC(key, dtype);
    scale = {
        static_cast<size_t>(TensorID::scale),
        logical_tensor::data_type::f32,
        0,
        logical_tensor::layout_type::strided,
        logical_tensor::property_type::host_scalar};
    if (is_causal) {
      neg_inf = {
          static_cast<size_t>(TensorID::neg_inf),
          logical_tensor::data_type::f32,
          0,
          logical_tensor::layout_type::strided,
          logical_tensor::property_type::host_scalar};
    }
    if (attn_mask_.has_value()) {
      const data_type mask_dtype =
          to_logical_tensor_data_type(attn_mask_->scalar_type());
      TORCH_INTERNAL_ASSERT(
          (mask_dtype != data_type::undef),
          "Only FP16/BF16/FP32 datatypes are currently supported for attn_mask");
      LOGIC_TENSOR_DESC(attn_mask, mask_dtype);
    }
    LOGIC_TENSOR_DESC(value, dtype);
    LOGIC_TENSOR_DESC(attention, dtype);
    if (compute_logsumexp) {
      TORCH_INTERNAL_ASSERT(
          logsumexp_.scalar_type() == at::kFloat,
          "scaled_dot_product_attention: Expected logsumexp data type in FP32, but got ",
          logsumexp_.scalar_type(),
          " instead.");
      LOGIC_TENSOR_DESC(logsumexp, sdpa_intermediate_dtype);
    }
#undef LOGIC_TENSOR_DESC
  }
  std::vector<logical_tensor> get_input() const {
    std::vector<logical_tensor> input = {query, key, scale};
    if (neg_inf.has_value()) {
      input.push_back(neg_inf.value());
    }
    if (attn_mask.has_value()) {
      input.push_back(attn_mask.value());
    }
    input.push_back(value);
    return input;
  }
  std::vector<logical_tensor> get_output() const {
    std::vector<logical_tensor> output;
    output.push_back(attention);
    if (logsumexp.has_value()) {
      output.push_back(logsumexp.value());
    }
    return output;
  }
};

partition create_sdpa_graph_partition(
    bool is_causal,
    bool compute_logsumexp,
    data_type dtype,
    const SDPALogicalParams& params) {
  // graph building and partitioning

  size_t lt_id = static_cast<size_t>(SDPALogicalParams::TensorID::end);
  size_t op_id = 0;

  // OneDNN graph has optimized implementation for `f16` or `bf16` SDPA with
  // `f32` intermediate data type on Intel Graphics Products with Intel(R) Xe
  // Matrix Extensions (Intel(R) XMX) support, which means the
  // Q/K/V tensors have bf16 or f16 data type while the output of the first
  // MatMul, Scale, Mask, and the input of SoftMax are in f32 data type.
  logical_tensor matmul_qk_out{lt_id++, sdpa_intermediate_dtype};
  op matmul_qk{
      op_id++,
      op::kind::MatMul,
      {params.query, params.key},
      {matmul_qk_out},
      "matmul_qk"};
  matmul_qk.set_attr<bool>(op::attr::transpose_b, true);

  logical_tensor scaled_qk_out{lt_id++, sdpa_intermediate_dtype};
  op scale_mul{
      op_id++,
      op::kind::Multiply,
      {matmul_qk_out, params.scale},
      {scaled_qk_out},
      "scale_mul"};

  std::optional<logical_tensor> masked_qk_out;

  // For optional additive mask
  std::optional<op> mask_add;

  // For optional implicit causal mask
  std::optional<op> mask_gen_idx_row;
  std::optional<logical_tensor> mask_row_idx;
  std::optional<op> mask_gen_idx_col;
  std::optional<logical_tensor> mask_col_idx;
  std::optional<op> mask_gt;
  std::optional<logical_tensor> mask_gt_out;
  std::optional<op> mask_select;

  if (params.attn_mask.has_value()) {
    TORCH_INTERNAL_ASSERT(
        !is_causal, "Additive mask cannot use with is_causal.");
    masked_qk_out = {lt_id++, sdpa_intermediate_dtype};
    mask_add = {
        op_id++,
        op::kind::Add,
        {scaled_qk_out, params.attn_mask.value()},
        {masked_qk_out.value()},
        "mask_add"};
  } else if (is_causal) {
#if (DNNL_VERSION_MAJOR >= 3 && DNNL_VERSION_MINOR >= 7)
    mask_row_idx = {lt_id++, data_type::s32};
    mask_gen_idx_row = {
        op_id++,
        op::kind::GenIndex,
        {scaled_qk_out},
        {mask_row_idx.value()},
        "mask_gen_idx_row"};
    mask_gen_idx_row->set_attr<int64_t>(op::attr::axis, -2);

    mask_col_idx = {lt_id++, data_type::s32};
    mask_gen_idx_col = {
        op_id++,
        op::kind::GenIndex,
        {scaled_qk_out},
        {mask_col_idx.value()},
        "mask_gen_idx_col"};
    mask_gen_idx_col->set_attr<int64_t>(op::attr::axis, -1);

    mask_gt_out = {lt_id++, data_type::boolean};
    mask_gt = {
        op_id++,
        op::kind::GreaterEqual,
        {mask_row_idx.value(), mask_col_idx.value()},
        {mask_gt_out.value()},
        "mask_gt"};

    masked_qk_out = {lt_id++, sdpa_intermediate_dtype};
    mask_select = {
        op_id++,
        op::kind::Select,
        {mask_gt_out.value(), scaled_qk_out, params.neg_inf.value()},
        {masked_qk_out.value()},
        "mask_select"};
#else
    TORCH_CHECK(
        false,
        "OneDNN v3.7 or later is required for implicit causal mask support.");
#endif
  }

  op softmax{op_id++, op::kind::SoftMax, "softmax"};
  softmax.set_attr<int64_t>(op::attr::axis, -1);
  softmax.set_attr<std::string>(op::attr::mode, "inf_as_zero");

  logical_tensor softmax_out{lt_id++, dtype};
  softmax.add_input(masked_qk_out.value_or(scaled_qk_out));
  softmax.add_output(softmax_out);
  if (compute_logsumexp) {
    softmax.add_output(params.logsumexp.value());
  }

  op matmul_v{
      op_id++,
      op::kind::MatMul,
      {softmax_out, params.value},
      {params.attention},
      "matmul_v"};

  constexpr auto ekind = dnnl::engine::kind::gpu;
  dnnl::graph::graph g(ekind);
  g.add_op(matmul_qk);
  g.add_op(scale_mul);
  if (mask_add.has_value()) {
    g.add_op(mask_add.value());
  }
  if (is_causal) {
    g.add_op(mask_gen_idx_row.value());
    g.add_op(mask_gen_idx_col.value());
    g.add_op(mask_gt.value());
    g.add_op(mask_select.value());
  }

  g.add_op(softmax);
  g.add_op(matmul_v);
  g.finalize();
  auto partitions = g.get_partitions();
  TORCH_INTERNAL_ASSERT(
      (partitions.size() == 1) && partitions[0].is_supported(),
      "oneDNN doesn't support this fusion pattern. If you'd like its support, please submit a issue.");
  return partitions[0];
}

partition& find_or_create_graph_partition(
    bool is_causal,
    bool compute_logsumexp,
    const SDPALogicalParams& params) {
  thread_local PartitionCache cache;
  const data_type dtype = params.query.get_data_type();

  // cache key creation
  // patternID is determined on the basis of the arguments provided
  std::bitset<32> patternID;
  if (dtype == data_type::f32) {
    patternID.set(static_cast<uint8_t>(PartitionCache::BitType::Float32), 1);
  }
  if (dtype == data_type::bf16) {
    patternID.set(static_cast<uint8_t>(PartitionCache::BitType::Bfloat16), 1);
  }
  // sdp pattern
  patternID.set(static_cast<uint8_t>(PartitionCache::BitType::SdpaPattern), 1);

  // Refer to comments in Utils.h. The first 8 bits are reserved
  int pos = 8;
  // attn_mask
  patternID.set(pos++, params.attn_mask.has_value());
  patternID.set(pos++, is_causal);
  // compute_logsumexp
  patternID.set(pos++, compute_logsumexp);

  auto partition_ = cache.find_partition(patternID);
  if (!partition_.has_value()) {
    // partition cache no hit
    // graph building and partitioning
    partition sdp_partition = create_sdpa_graph_partition(
        is_causal, compute_logsumexp, dtype, params);
    partition_ = cache.insert_partition_cache(patternID, sdp_partition);
  }
  return *partition_;
}
} // namespace sdpa_forward

namespace sdpa_backward {

struct SDPABackwardLogicalParams {
  enum class TensorID {
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    scale,
    neg_inf,
    attn_mask,
    grad_query,
    grad_key,
    grad_value,
    end,
  };

  logical_tensor grad_out{};
  logical_tensor query{};
  logical_tensor key{};
  logical_tensor value{};
  logical_tensor out{};
  logical_tensor logsumexp{};
  logical_tensor scale{};
  std::optional<logical_tensor> neg_inf;
  std::optional<logical_tensor> attn_mask;
  logical_tensor grad_query{};
  logical_tensor grad_key{};
  logical_tensor grad_value{};

  SDPABackwardLogicalParams(
      const at::Tensor& grad_out_,
      const at::Tensor& query_,
      const at::Tensor& key_,
      const at::Tensor& value_,
      const at::Tensor& out_,
      const at::Tensor& logsumexp_,
      const std::optional<at::Tensor>& attn_mask_,
      const at::Tensor& grad_query_,
      const at::Tensor& grad_key_,
      const at::Tensor& grad_value_,
      int batch_size,
      int num_head_q,
      int num_head_kv,
      int seq_len_q,
      int seq_len_kv,
      int head_dim_qk,
      int head_dim_v,
      bool is_causal) {
    const data_type dtype = to_logical_tensor_data_type(query_.scalar_type());
    TORCH_INTERNAL_ASSERT(
        (dtype != data_type::undef),
        "Only FP16/BF16/FP32 datatypes are currently supported");
    TORCH_INTERNAL_ASSERT(
        grad_out_.scalar_type() == query_.scalar_type() &&
            grad_out_.scalar_type() == key_.scalar_type() &&
            grad_out_.scalar_type() == value_.scalar_type() &&
            grad_out_.scalar_type() == out_.scalar_type(),
        "scaled_dot_product_attention_backward_xpu: Expected grad_out, q, k, v and out to have the same data type, but got ",
        " grad_out: ",
        grad_out_.scalar_type(),
        ", q: ",
        query_.scalar_type(),
        ", k: ",
        key_.scalar_type(),
        ", v: ",
        value_.scalar_type(),
        ", out: ",
        out_.scalar_type());
    TORCH_INTERNAL_ASSERT(
        logsumexp_.defined() && logsumexp_.scalar_type() == at::kFloat,
        "scaled_dot_product_attention_backward_xpu: Expected logsumexp to be defined and have FP32 data type");

    at::Tensor reshaped_grad_out = grad_out_;
    at::Tensor reshaped_query = query_;
    at::Tensor reshaped_key = key_;
    at::Tensor reshaped_value = value_;
    at::Tensor reshaped_out = out_;
    at::Tensor reshaped_logsumexp = logsumexp_.unsqueeze(-1);
    at::Tensor reshaped_attn_mask = attn_mask_.value_or(at::Tensor());
    at::Tensor reshaped_grad_query = grad_query_;
    at::Tensor reshaped_grad_key = grad_key_;
    at::Tensor reshaped_grad_value = grad_value_;

    // handle broadcasted input tensors for OneDNN
    if (at::native::onednn::is_broadcast(reshaped_grad_out)) {
      at::native::onednn::undo_broadcast(reshaped_grad_out);
    }
    if (at::native::onednn::is_broadcast(reshaped_query)) {
      at::native::onednn::undo_broadcast(reshaped_query);
    }
    if (at::native::onednn::is_broadcast(reshaped_key)) {
      at::native::onednn::undo_broadcast(reshaped_key);
    }
    if (at::native::onednn::is_broadcast(reshaped_value)) {
      at::native::onednn::undo_broadcast(reshaped_value);
    }
    if (attn_mask_.has_value() &&
        at::native::onednn::is_broadcast(reshaped_attn_mask)) {
      at::native::onednn::undo_broadcast(reshaped_attn_mask);
    }

    // TODO: Support GQA in backward pass once OneDNN supports it.

#define LOGIC_TENSOR_DESC(name, dtype)     \
  name = {                                 \
      static_cast<size_t>(TensorID::name), \
      dtype,                               \
      reshaped_##name.sizes().vec(),       \
      reshaped_##name.strides().vec()}

    LOGIC_TENSOR_DESC(grad_out, dtype);
    LOGIC_TENSOR_DESC(query, dtype);
    LOGIC_TENSOR_DESC(key, dtype);
    LOGIC_TENSOR_DESC(value, dtype);
    LOGIC_TENSOR_DESC(out, dtype);
    LOGIC_TENSOR_DESC(logsumexp, sdpa_intermediate_dtype);
    scale = {
        static_cast<size_t>(TensorID::scale),
        logical_tensor::data_type::f32,
        0,
        logical_tensor::layout_type::strided,
        logical_tensor::property_type::host_scalar};
    if (is_causal) {
      neg_inf = {
          static_cast<size_t>(TensorID::neg_inf),
          logical_tensor::data_type::f32,
          0,
          logical_tensor::layout_type::strided,
          logical_tensor::property_type::host_scalar};
    }
    if (attn_mask_.has_value()) {
      const data_type mask_dtype =
          to_logical_tensor_data_type(attn_mask_->scalar_type());
      TORCH_INTERNAL_ASSERT(
          (mask_dtype != data_type::undef),
          "Only FP16/BF16/FP32 datatypes are currently supported for attn_mask");
      LOGIC_TENSOR_DESC(attn_mask, mask_dtype);
    }
    LOGIC_TENSOR_DESC(grad_query, dtype);
    LOGIC_TENSOR_DESC(grad_key, dtype);
    LOGIC_TENSOR_DESC(grad_value, dtype);
#undef LOGIC_TENSOR_DESC
  }
  std::vector<logical_tensor> get_input() const {
    std::vector<logical_tensor> input = {
        grad_out, query, key, value, out, logsumexp, scale};
    if (neg_inf.has_value()) {
      input.push_back(neg_inf.value());
    }
    if (attn_mask.has_value()) {
      input.push_back(attn_mask.value());
    }
    return input;
  }
  std::vector<logical_tensor> get_output() const {
    std::vector<logical_tensor> output = {grad_query, grad_key, grad_value};
    return output;
  }
};

partition create_sdpa_backward_graph_partition(
    bool is_causal,
    data_type dtype,
    const SDPABackwardLogicalParams& params) {
  // graph building and partitioning
  size_t lt_id = static_cast<size_t>(SDPABackwardLogicalParams::TensorID::end);
  size_t op_id = 0;

  // OneDNN graph has optimized implementation for `f16` or `bf16` SDPA with
  // `f32` intermediate data type on Intel Graphics Products with Intel(R) Xe
  // Matrix Extensions (Intel(R) XMX) support, which means the
  // Q/K/V tensors have bf16 or f16 data type while the output of the first
  // MatMul, Scale, Mask, and the input of SoftMax are in f32 data type.
  logical_tensor matmul_qk_out{lt_id++, sdpa_intermediate_dtype};
  op matmul_qk{
      op_id++,
      op::kind::MatMul,
      {params.query, params.key},
      {matmul_qk_out},
      "matmul_qk"};
  matmul_qk.set_attr<bool>(op::attr::transpose_b, true);

  logical_tensor scaled_qk_out{lt_id++, sdpa_intermediate_dtype};
  op scale_mul{
      op_id++,
      op::kind::Multiply,
      {matmul_qk_out, params.scale},
      {scaled_qk_out},
      "scale_mul"};

  std::optional<logical_tensor> masked_qk_out;

  // For optional additive mask
  std::optional<op> mask_add;

  // For optional implicit causal mask
  std::optional<op> mask_gen_idx_row;
  std::optional<logical_tensor> mask_row_idx;
  std::optional<op> mask_gen_idx_col;
  std::optional<logical_tensor> mask_col_idx;
  std::optional<op> mask_gt;
  std::optional<logical_tensor> mask_gt_out;
  std::optional<op> mask_select;

  if (params.attn_mask.has_value()) {
    TORCH_INTERNAL_ASSERT(
        !is_causal, "Additive mask cannot use with is_causal.");
    masked_qk_out = {lt_id++, sdpa_intermediate_dtype};
    mask_add = {
        op_id++,
        op::kind::Add,
        {scaled_qk_out, params.attn_mask.value()},
        {masked_qk_out.value()},
        "mask_add"};
  } else if (is_causal) {
    mask_row_idx = {lt_id++, data_type::s32};
    mask_gen_idx_row = {
        op_id++,
        op::kind::GenIndex,
        {scaled_qk_out},
        {mask_row_idx.value()},
        "mask_gen_idx_row"};
    mask_gen_idx_row->set_attr<int64_t>(op::attr::axis, -2);

    mask_col_idx = {lt_id++, data_type::s32};
    mask_gen_idx_col = {
        op_id++,
        op::kind::GenIndex,
        {scaled_qk_out},
        {mask_col_idx.value()},
        "mask_gen_idx_col"};
    mask_gen_idx_col->set_attr<int64_t>(op::attr::axis, -1);

    mask_gt_out = {lt_id++, data_type::boolean};
    mask_gt = {
        op_id++,
        op::kind::GreaterEqual,
        {mask_row_idx.value(), mask_col_idx.value()},
        {mask_gt_out.value()},
        "mask_gt"};

    masked_qk_out = {lt_id++, sdpa_intermediate_dtype};
    mask_select = {
        op_id++,
        op::kind::Select,
        {mask_gt_out.value(), scaled_qk_out, params.neg_inf.value()},
        {masked_qk_out.value()},
        "mask_select"};
  }

  // attention_probs = softmax(masked_score) = exp(masked_score - logsumexp)
  logical_tensor sub_out{lt_id++, sdpa_intermediate_dtype};
  op subtract{
      op_id++,
      op::kind::Subtract,
      {masked_qk_out.value_or(scaled_qk_out), params.logsumexp},
      {sub_out},
      "subtract"};
  logical_tensor prob{lt_id++, sdpa_intermediate_dtype};
  op exp{op_id++, op::kind::Exp, {sub_out}, {prob}, "exp"};

  // The following matmul doesn't support different input dtypes, insert a
  // typecast
  logical_tensor prob_casted = prob;
  op typecast = op(op_id++, op::kind::TypeCast, "typecast");
  if (dtype != sdpa_intermediate_dtype) {
    prob_casted = logical_tensor(lt_id++, dtype);
    typecast.add_inputs({prob});
    typecast.add_outputs({prob_casted});
  }

  // grad_value = prob^T * grad_out
  // TODO: handle GQA headnum because (batch_size, num_head_kv, seq_len_kv,
  // head_dim_v) != (batch_size, num_head_q, seqlen_kv, seq_len_q) *
  // (batch_size, num_head_q, seqlen_q, head_dim_v)
  op matmul_grad_value{
      op_id++,
      op::kind::MatMul,
      {prob_casted, params.grad_out},
      {params.grad_value},
      "matmul_grad_value"};
  matmul_grad_value.set_attr<bool>(op::attr::transpose_a, true);

  // grad_prop = grad_out * value^T
  // TODO: handle GQA headnum because (batch_size, num_head_q, seq_len_q,
  // seq_len_kv) != (batch_size, num_head_q, seq_len_q, head_dim_v) *
  // (batch_size, num_head_kv, head_dim_v, seq_len_kv)
  logical_tensor grad_prop{lt_id++, sdpa_intermediate_dtype};
  op matmul_grad_prop{
      op_id++,
      op::kind::MatMul,
      {params.grad_out, params.value},
      {grad_prop},
      "matmul_grad_prop"};
  matmul_grad_prop.set_attr<bool>(op::attr::transpose_b, true);

  // grad_masked_score = softmaxbackward(grad_prop)
  logical_tensor grad_masked_score{lt_id++, sdpa_intermediate_dtype};
  op softmax_backward{
      op_id++,
      op::kind::SoftMaxBackward,
      {grad_prop, prob},
      {grad_masked_score},
      "softmax_backward"};
  softmax_backward.set_attr<int64_t>(op::attr::axis, -1);

  // TODO: add output tensor grad_attn_mask = grad_masked_score once OneDNN
  // supports output grad_attn_mask.

  // grad_scaled_score = grad_masked_score * scale
  logical_tensor grad_scaled_score{lt_id++, sdpa_intermediate_dtype};
  op grad_scale_mul{
      op_id++,
      op::kind::Multiply,
      {grad_masked_score, params.scale},
      {grad_scaled_score},
      "grad_scale_mul"};

  // The following matmul doesn't support different input dtypes, insert a
  // typecast
  logical_tensor grad_scaled_score_cast = grad_scaled_score;
  op typecast2 = op(op_id++, op::kind::TypeCast, "typecast2");
  if (dtype != sdpa_intermediate_dtype) {
    grad_scaled_score_cast = logical_tensor(lt_id++, dtype);
    typecast2.add_inputs({grad_scaled_score});
    typecast2.add_outputs({grad_scaled_score_cast});
  }

  // grad_query = grad_scaled_score_cast * key
  // TODO: handle GQA headnum because (batch_size, num_head_q, seq_len_q,
  // head_dim_qk) != (batch_size, num_head_q, seq_len_q, seq_len_kv) *
  // (batch_size, num_head_kv, seq_len_kv, head_dim_qk)
  op matmul_grad_query{
      op_id++,
      op::kind::MatMul,
      {grad_scaled_score_cast, params.key},
      {params.grad_query},
      "matmul_grad_query"};

  // grad_key = grad_scaled_score_cast^T * query
  op matmul_grad_key{
      op_id++,
      op::kind::MatMul,
      {grad_scaled_score_cast, params.query},
      {params.grad_key},
      "matmul_grad_key"};
  matmul_grad_key.set_attr<bool>(op::attr::transpose_a, true);

  constexpr auto ekind = dnnl::engine::kind::gpu;
  dnnl::graph::graph g(ekind);
  g.add_op(matmul_qk);
  g.add_op(scale_mul);
  if (mask_add.has_value()) {
    g.add_op(mask_add.value());
  }
  if (is_causal) {
    g.add_op(mask_gen_idx_row.value());
    g.add_op(mask_gen_idx_col.value());
    g.add_op(mask_gt.value());
    g.add_op(mask_select.value());
  }
  g.add_op(subtract);
  g.add_op(exp);
  g.add_op(matmul_grad_value);
  g.add_op(matmul_grad_prop);
  g.add_op(softmax_backward);
  g.add_op(grad_scale_mul);
  g.add_op(matmul_grad_query);
  g.add_op(matmul_grad_key);
  if (dtype != sdpa_intermediate_dtype) {
    g.add_op(typecast);
    g.add_op(typecast2);
  }
  g.finalize();
  auto partitions = g.get_partitions();
  TORCH_INTERNAL_ASSERT(
      (partitions.size() == 1) && partitions[0].is_supported(),
      "oneDNN doesn't support this fusion pattern. If you'd like its support, please submit a issue.");
  return partitions[0];
}

partition& find_or_create_backward_graph_partition(
    bool is_causal,
    const SDPABackwardLogicalParams& params) {
  thread_local PartitionCache cache;
  const data_type dtype = params.query.get_data_type();

  // cache key creation
  // patternID is determined on the basis of the arguments provided
  std::bitset<32> patternID;
  if (dtype == data_type::f32) {
    patternID.set(static_cast<uint8_t>(PartitionCache::BitType::Float32), 1);
  }
  if (dtype == data_type::bf16) {
    patternID.set(static_cast<uint8_t>(PartitionCache::BitType::Bfloat16), 1);
  }
  // sdpa backward pattern
  patternID.set(
      static_cast<uint8_t>(PartitionCache::BitType::SdpaBwdPattern), 1);

  // Refer to comments in Utils.h. The first 8 bits are reserved
  int pos = 8;
  // attn_mask
  patternID.set(pos++, params.attn_mask.has_value());
  patternID.set(pos++, is_causal);

  auto partition_ = cache.find_partition(patternID);
  if (!partition_.has_value()) {
    // partition cache no hit
    // graph building and partitioning
    partition sdpa_backward_partition =
        create_sdpa_backward_graph_partition(is_causal, dtype, params);
    partition_ =
        cache.insert_partition_cache(patternID, sdpa_backward_partition);
  }
  return *partition_;
}
} // namespace sdpa_backward
} // namespace

namespace at::native::onednn {
void sdpa(
    int batch_size,
    int seq_len_q,
    int seq_len_kv,
    int num_head_q,
    int num_head_kv,
    int head_dim_qk,
    int head_dim_v,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    std::optional<at::Tensor> attn_mask,
    bool is_causal,
    float softmax_scale,
    const Tensor& attention,
    bool compute_logsumexp,
    const Tensor& logsumexp) {
  auto& eng = GpuEngineManager::Instance().get_engine();
  auto& strm = GpuStreamManager::Instance().get_stream();

  const auto get_tril_mask = [&]() {
    auto opts = query.options();
    auto bool_tril =
        at::ones_symint({seq_len_q, seq_len_kv}, opts.dtype(at::kBool)).tril();
    return at::where(
        bool_tril,
        0.f,
        at::scalar_tensor(-std::numeric_limits<float>::infinity(), opts));
  };

  // OneDNN doesn't support fp32 ukernel for implicit causal mask,
  // and the reference implementation is worse than aten math + explicit causal
  // mask. Fall back to explicit causal mask until OneDNN v3.9 which has fp32
  // ukernel for implicit causal mask.
  if (is_causal && query.dtype() == at::kFloat) {
    attn_mask = get_tril_mask();
    is_causal = false;
  }

  std::vector<dnnl::graph::logical_tensor> l_inputs, l_outputs;
  std::optional<dnnl::graph::compiled_partition> compiled_partition;

  const sdpa_forward::SDPALogicalParams logical_params(
      query,
      key,
      value,
      attn_mask,
      attention,
      logsumexp,
      batch_size,
      seq_len_q,
      seq_len_kv,
      num_head_q,
      num_head_kv,
      head_dim_qk,
      head_dim_v,
      is_causal,
      compute_logsumexp);
  auto& partition = sdpa_forward::find_or_create_graph_partition(
      is_causal, compute_logsumexp, logical_params);
  l_inputs = std::move(logical_params.get_input());
  l_outputs = std::move(logical_params.get_output());
  compiled_partition = partition.compile(l_inputs, l_outputs, eng);

  std::vector<dnnl::graph::tensor> outputs = {
      {l_outputs[0], eng, attention.data_ptr()},
  };
  if (compute_logsumexp) {
    outputs.emplace_back(l_outputs[1], eng, logsumexp.data_ptr());
  }

  size_t i = 0;
  std::vector<dnnl::graph::tensor> inputs;
  inputs.reserve(l_inputs.size());

#define ADD_INPUT(variable) \
  inputs.emplace_back(l_inputs[i++], eng, variable.data_ptr())

  ADD_INPUT(query);
  ADD_INPUT(key);
  inputs.emplace_back(
      dnnl::graph::tensor::make_scalar_tensor(l_inputs[i++], &softmax_scale));
  if (is_causal) {
    constexpr float neg_inf_val = -std::numeric_limits<float>::infinity();
    inputs.emplace_back(dnnl::graph::tensor::make_scalar_tensor(
        l_inputs[i++], const_cast<float*>(&neg_inf_val)));
  }
  if (attn_mask.has_value()) {
    ADD_INPUT((*attn_mask));
  }
  ADD_INPUT(value);
#undef ADD_INPUT

  compiled_partition->execute(strm, inputs, outputs);
}

void sdpa_backward(
    int batch_size,
    int num_head_q,
    int num_head_kv,
    int seq_len_q,
    int seq_len_kv,
    int head_dim_qk,
    int head_dim_v,
    const Tensor& grad_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& out,
    const Tensor& logsumexp,
    std::optional<at::Tensor> attn_mask,
    bool is_causal,
    float softmax_scale,
    Tensor& grad_query,
    Tensor& grad_key,
    Tensor& grad_value) {
  auto& eng = GpuEngineManager::Instance().get_engine();
  auto& strm = GpuStreamManager::Instance().get_stream();

  const auto get_tril_mask = [&]() {
    auto opts = query.options();
    auto bool_tril =
        at::ones_symint({seq_len_q, seq_len_kv}, opts.dtype(at::kBool)).tril();
    return at::where(
        bool_tril,
        0.f,
        at::scalar_tensor(-std::numeric_limits<float>::infinity(), opts));
  };

  // OneDNN doesn't support fp32 ukernel for implicit causal mask,
  // and the reference implementation is worse than aten math + explicit causal
  // mask. Fall back to explicit causal mask until OneDNN v3.9 which has fp32
  // ukernel for implicit causal mask.
  if (is_causal && query.dtype() == at::kFloat) {
    attn_mask = get_tril_mask();
    is_causal = false;
  }

  std::vector<dnnl::graph::logical_tensor> l_inputs, l_outputs;
  std::optional<dnnl::graph::compiled_partition> compiled_partition;

  const sdpa_backward::SDPABackwardLogicalParams logical_params(
      grad_out,
      query,
      key,
      value,
      out,
      logsumexp,
      attn_mask,
      grad_query,
      grad_key,
      grad_value,
      batch_size,
      num_head_q,
      num_head_kv,
      seq_len_q,
      seq_len_kv,
      head_dim_qk,
      head_dim_v,
      is_causal);
  auto& partition = sdpa_backward::find_or_create_backward_graph_partition(
      is_causal, logical_params);
  l_inputs = std::move(logical_params.get_input());
  l_outputs = std::move(logical_params.get_output());
  compiled_partition = partition.compile(l_inputs, l_outputs, eng);

  std::vector<dnnl::graph::tensor> outputs = {
      {l_outputs[0], eng, grad_query.data_ptr()},
      {l_outputs[1], eng, grad_key.data_ptr()},
      {l_outputs[2], eng, grad_value.data_ptr()},
  };

  size_t i = 0;
  std::vector<dnnl::graph::tensor> inputs;
  inputs.reserve(l_inputs.size());

#define ADD_INPUT(variable) \
  inputs.emplace_back(l_inputs[i++], eng, variable.data_ptr())

  ADD_INPUT(grad_out);
  ADD_INPUT(query);
  ADD_INPUT(key);
  ADD_INPUT(value);
  ADD_INPUT(out);
  ADD_INPUT(logsumexp);
  inputs.emplace_back(
      dnnl::graph::tensor::make_scalar_tensor(l_inputs[i++], &softmax_scale));
  if (is_causal) {
    constexpr float neg_inf_val = -std::numeric_limits<float>::infinity();
    inputs.emplace_back(dnnl::graph::tensor::make_scalar_tensor(
        l_inputs[i++], const_cast<float*>(&neg_inf_val)));
  }
  if (attn_mask.has_value()) {
    ADD_INPUT((*attn_mask));
  }
#undef ADD_INPUT

  compiled_partition->execute(strm, inputs, outputs);
}
} // namespace at::native::onednn
