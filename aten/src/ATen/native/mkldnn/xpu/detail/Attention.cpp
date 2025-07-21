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

inline data_type to_logical_tensor_data_type(c10::ScalarType scalar_type) {
  return scalar_type == c10::ScalarType::Float   ? data_type::f32
      : scalar_type == c10::ScalarType::Half     ? data_type::f16
      : scalar_type == c10::ScalarType::BFloat16 ? data_type::bf16
                                                 : data_type::undef;
}

struct SDPALogicalParams {
  enum class TensorID {
    query,
    key,
    scale,
    neg_inf,
    attn_mask,
    value,
    output,
    end,
  };

  logical_tensor query{};
  logical_tensor key{};
  logical_tensor scale{};
  std::optional<logical_tensor> neg_inf;
  std::optional<logical_tensor> attn_mask;
  logical_tensor value{};
  logical_tensor output{};

  SDPALogicalParams(
      const at::Tensor& query_,
      const at::Tensor& key_,
      const at::Tensor& value_,
      const std::optional<at::Tensor>& attn_mask_,
      const at::Tensor& output_,
      int batch_size,
      int seq_len_q,
      int seq_len_kv,
      int num_head_q,
      int num_head_kv,
      int head_dim_qk,
      int head_dim_v,
      bool is_causal) {
    const data_type dtype = to_logical_tensor_data_type(query_.scalar_type());
    TORCH_INTERNAL_ASSERT(
        (dtype != data_type::undef),
        "Only FP16/BF16/FP32 datatypes are currently supported");
    const dims scalar_shape = {1};
    std::vector<logical_tensor> inputLogicalTensors;

    at::Tensor reshaped_query = query_;
    at::Tensor reshaped_key = key_;
    at::Tensor reshaped_value = value_;
    at::Tensor reshaped_output = output_;
    at::Tensor reshaped_attn_mask = attn_mask_.value_or(at::Tensor());
    if (at::native::onednn::is_broadcast(reshaped_query)) {
      at::native::onednn::undo_broadcast(reshaped_query);
    }
    if (at::native::onednn::is_broadcast(reshaped_key)) {
      at::native::onednn::undo_broadcast(reshaped_key);
    }
    if (at::native::onednn::is_broadcast(reshaped_value)) {
      at::native::onednn::undo_broadcast(reshaped_value);
    }
    if (at::native::onednn::is_broadcast(reshaped_output)) {
      at::native::onednn::undo_broadcast(reshaped_output);
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
      reshaped_output = output_.view(
          {batch_size, group_num, group_size, seq_len_q, head_dim_v});
      if (attn_mask_.has_value() && attn_mask_.value().dim() == 4) {
        reshaped_attn_mask = attn_mask_.value().unsqueeze(2);
      }
    }

    query = {
        static_cast<size_t>(TensorID::query),
        dtype,
        reshaped_query.sizes().vec(),
        reshaped_query.strides().vec()};
    key = {
        static_cast<size_t>(TensorID::key),
        dtype,
        reshaped_key.sizes().vec(),
        reshaped_key.strides().vec()};
    scale = {
        static_cast<size_t>(TensorID::scale),
        to_logical_tensor_data_type(at::toOpMathType(query_.scalar_type())),
        scalar_shape,
        logical_tensor::layout_type::strided,
        logical_tensor::property_type::constant};
    if (is_causal) {
      neg_inf = {
          static_cast<size_t>(TensorID::neg_inf),
          to_logical_tensor_data_type(at::toOpMathType(query_.scalar_type())),
          scalar_shape,
          logical_tensor::layout_type::strided,
          logical_tensor::property_type::constant};
    }
    if (attn_mask_.has_value()) {
      const data_type mask_dtype =
          to_logical_tensor_data_type(attn_mask_->scalar_type());
      TORCH_INTERNAL_ASSERT(
          (mask_dtype != data_type::undef),
          "Only FP16/BF16/FP32 datatypes are currently supported for attn_mask");
      attn_mask = {
          static_cast<size_t>(TensorID::attn_mask),
          mask_dtype,
          reshaped_attn_mask.sizes().vec(),
          reshaped_attn_mask.strides().vec()};
    }
    value = {
        static_cast<size_t>(TensorID::value),
        dtype,
        reshaped_value.sizes().vec(),
        reshaped_value.strides().vec()};
    output = {
        static_cast<size_t>(TensorID::output),
        dtype,
        reshaped_output.sizes().vec(),
        reshaped_output.strides().vec()};
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
    return {output};
  }
};

partition create_sdpa_graph_partition(
    bool is_causal,
    data_type dtype,
    const SDPALogicalParams& params) {
  // graph building and partitioning
  // currently, we assume that Q and K have same sequence length

  size_t lt_id = static_cast<size_t>(SDPALogicalParams::TensorID::end);
  size_t op_id = 0;

  // OneDNN graph has optimized implementation for `f16` or `bf16` SDPA with
  // `f32` intermediate data type on Intel Graphics Products with Intel(R) Xe
  // Matrix Extensions (Intel(R) XMX) support, which means the
  // Q/K/V tensors have bf16 or f16 data type while the output of the first
  // MatMul, Scale, Mask, and the input of SoftMax are in f32 data type.
  logical_tensor matmul_qk_out{lt_id++, data_type::f32};
  op matmul_qk{
      op_id++,
      op::kind::MatMul,
      {params.query, params.key},
      {matmul_qk_out},
      "matmul_qk"};
  matmul_qk.set_attr<bool>(op::attr::transpose_b, true);

  logical_tensor scaled_qk_out{lt_id++, data_type::f32};
  op scale_mul{
      op_id++,
      op::kind::Multiply,
      {matmul_qk_out, params.scale},
      {scaled_qk_out},
      "scale_mul"};

  std::optional<logical_tensor> masked_qk_out;

  // For optional additive mask
  std::optional<op> mask_add;

  // For optional implicite causal mask
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
    masked_qk_out = {lt_id++, data_type::f32};
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

    masked_qk_out = {lt_id++, data_type::f32};
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

  op matmul_v{
      op_id++,
      op::kind::MatMul,
      {softmax_out, params.value},
      {params.output},
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
    const SDPALogicalParams& params) {
  thread_local static PartitionCache cache;
  const data_type dtype = params.query.get_data_type();

  // cache key creation
  // patternID is determined on the basis of the arguments provided
  std::bitset<32> patternID;
  if (dtype == data_type::f32) {
    // bit 3 corresponds to float32 dtype
    patternID.set(3, 1);
  }
  if (dtype == data_type::bf16) {
    // bit 2 corresponds to fp16/bf16 dtype
    patternID.set(2, 1);
  }
  // sdp pattern
  patternID.set(4, 1);

  // Refer to comments in Utils.h. The first 8 bits are reserved
  int pos = 8;
  // attn_mask
  patternID.set(pos++, params.attn_mask.has_value());
  patternID.set(pos++, is_causal);

  auto partition_ = cache.find_partition(patternID);
  if (!partition_.has_value()) {
    // partition cache no hit
    // graph building and partitioning
    partition sdp_partition =
        create_sdpa_graph_partition(is_causal, dtype, params);
    partition_ = cache.insert_partition_cache(patternID, sdp_partition);
  }
  return *partition_;
}
} // namespace

namespace at::native::onednn {
void gpu_float_sdpa(
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
    const Tensor& output) {
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
  // and the reference implementation is worse than aten math + explict causal
  // mask. Fall back to explict causal mask until OneDNN v3.9 which has fp32
  // ukernel for implicit causal mask.
  if (is_causal && query.dtype() == at::kFloat) {
    attn_mask = get_tril_mask();
    is_causal = false;
  }

  std::vector<dnnl::graph::logical_tensor> l_inputs, l_outputs;
  std::optional<dnnl::graph::compiled_partition> compiled_partition;

  auto get_compiled_partition = [&]() {
    const SDPALogicalParams logical_params(
        query,
        key,
        value,
        attn_mask,
        output,
        batch_size,
        seq_len_q,
        seq_len_kv,
        num_head_q,
        num_head_kv,
        head_dim_qk,
        head_dim_v,
        is_causal);
    auto& partition_ =
        find_or_create_graph_partition(is_causal, logical_params);
    auto i = logical_params.get_input();
    auto o = logical_params.get_output();
    auto compiled_partition = partition_.compile(i, o, eng);
    l_inputs = std::move(i);
    l_outputs = std::move(o);
    return compiled_partition;
  };

  compiled_partition = get_compiled_partition();

  Tensor softmax_scale1 = at::full(
      {},
      softmax_scale,
      query.options().dtype(at::toOpMathType(query.scalar_type())));
  std::optional<at::Tensor> neg_inf;
  if (is_causal) {
    neg_inf = at::full(
        {},
        -INFINITY,
        query.options().dtype(at::toOpMathType(query.scalar_type())));
  }

  std::vector<dnnl::graph::tensor> outputs = {
      {l_outputs[0], eng, output.data_ptr()},
  };
  size_t i = 0;
  std::vector<dnnl::graph::tensor> inputs;
  inputs.reserve(l_inputs.size());
  inputs.emplace_back(l_inputs[i++], eng, query.data_ptr());
  inputs.emplace_back(l_inputs[i++], eng, key.data_ptr());
  inputs.emplace_back(l_inputs[i++], eng, softmax_scale1.data_ptr());
  if (neg_inf.has_value()) {
    inputs.emplace_back(l_inputs[i++], eng, neg_inf->data_ptr());
  }
  if (attn_mask.has_value()) {
    inputs.emplace_back(l_inputs[i++], eng, attn_mask->data_ptr());
  }
  inputs.emplace_back(l_inputs[i++], eng, value.data_ptr());
  compiled_partition->execute(strm, inputs, outputs);
}
} // namespace at::native::onednn
