#include <ATen/native/mkldnn/xpu/detail/Attr.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>

#include <omp.h>
#include <oneapi/dnnl/dnnl.hpp>

using namespace at::native::onednn;
using logical_tensor = dnnl::graph::logical_tensor;
using data_type = logical_tensor::data_type;
using dims = logical_tensor::dims;
using op = dnnl::graph::op;
using partition = dnnl::graph::partition;

namespace {
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

  logical_tensor query;
  logical_tensor key;
  logical_tensor scale;
  std::optional<logical_tensor> neg_inf;
  std::optional<logical_tensor> attn_mask;
  logical_tensor value;
  logical_tensor output;

  SDPALogicalParams(
      const at::Tensor& query_,
      const at::Tensor& key_,
      const at::Tensor& value_,
      const std::optional<at::Tensor>& attn_mask_,
      const at::Tensor& output_,
      data_type dtype,
      bool is_causal) {
    const dims scalar_shape = {1};
    std::vector<logical_tensor> inputLogicalTensors;
    query = {
        static_cast<size_t>(TensorID::query),
        dtype,
        query_.sizes().vec(),
        query_.strides().vec()};
    key = {
        static_cast<size_t>(TensorID::key),
        dtype,
        key_.sizes().vec(),
        key_.strides().vec()};
    scale = {
        static_cast<size_t>(TensorID::scale),
        dtype,
        scalar_shape,
        logical_tensor::layout_type::strided,
        logical_tensor::property_type::constant};
    if (is_causal) {
      neg_inf = {
          static_cast<size_t>(TensorID::neg_inf),
          dtype,
          scalar_shape,
          logical_tensor::layout_type::strided,
          logical_tensor::property_type::constant};
    }
    if (attn_mask_.has_value()) {
      attn_mask = {
          static_cast<size_t>(TensorID::attn_mask),
          dtype,
          attn_mask_->sizes().vec(),
          attn_mask_->strides().vec()};
    }
    value = {
        static_cast<size_t>(TensorID::value),
        dtype,
        value_.sizes().vec(),
        value_.strides().vec()};
    output = {
        static_cast<size_t>(TensorID::output),
        dtype,
        output_.sizes().vec(),
        output_.strides().vec()};
  }
  std::vector<logical_tensor> get_input() const {
    if (attn_mask.has_value()) {
      return {query, key, scale, attn_mask.value(), value};
    } else if (neg_inf.has_value()) {
      return {query, key, scale, neg_inf.value(), value};
    } else {
      return {query, key, scale, value};
    }
  }
  std::vector<logical_tensor> get_output() const {
    return {output};
  }
};

partition create_sdpa_graph_partition(
    int batch_size,
    int seq_len_q,
    int seq_len_k,
    int num_head,
    int head_dim,
    bool is_causal,
    data_type dtype,
    const SDPALogicalParams& params) {
  // graph building and partitioning
  // currently, we assume that Q and K have same sequence length

  dims qk_output_shape = {batch_size, num_head, seq_len_q, seq_len_k};
  dims scale_shape = {1};
  size_t lt_id = static_cast<size_t>(SDPALogicalParams::TensorID::end);
  size_t op_id = 0;

  logical_tensor matmul_qk_out{lt_id++, dtype};
  op matmul_qk{
      op_id++,
      op::kind::MatMul,
      {params.query, params.key},
      {matmul_qk_out},
      "matmul_qk"};
  matmul_qk.set_attr<bool>(op::attr::transpose_b, true);

  logical_tensor scaled_qk_out{lt_id++, dtype};
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
    masked_qk_out = {lt_id++, dtype};
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
    mask_gen_idx_row->set_attr<int64_t>(op::attr::axis, 2);

    mask_col_idx = {lt_id++, data_type::s32};
    mask_gen_idx_col = {
        op_id++,
        op::kind::GenIndex,
        {scaled_qk_out},
        {mask_col_idx.value()},
        "mask_gen_idx_col"};
    mask_gen_idx_col->set_attr<int64_t>(op::attr::axis, 3);

    mask_gt_out = {lt_id++, data_type::boolean};
    mask_gt = {
        op_id++,
        op::kind::GreaterEqual,
        {mask_row_idx.value(), mask_col_idx.value()},
        {mask_gt_out.value()},
        "mask_gt"};

    masked_qk_out = {lt_id++, dtype};
    mask_select = {
        op_id++,
        op::kind::Select,
        {mask_gt_out.value(), scaled_qk_out, params.neg_inf.value()},
        {masked_qk_out.value()},
        "mask_select"};
  }

  op softmax{op_id++, op::kind::SoftMax, "softmax"};
  softmax.set_attr<int64_t>(op::attr::axis, -1);

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
} // namespace

namespace at::native::onednn {
void gpu_float_sdpa(
    int batch_size,
    int seq_len_q,
    int seq_len_k,
    int num_head,
    int num_head_kv,
    int head_dim,
    int head_dim_v,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    std::optional<at::Tensor> attn_mask,
    bool is_causal,
    float softmax_scale,
    const Tensor& output) {
  auto eng = GpuEngineManager::Instance().get_engine(
      {c10::kXPU, c10::xpu::current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  Tensor softmax_scale1 = at::full({}, softmax_scale, query.options());

  std::optional<at::Tensor> neg_inf;
  if (is_causal) {
    neg_inf = at::full({}, -INFINITY, query.options());
  }
  const data_type logical_tensor_dtype =
      query.scalar_type() == c10::ScalarType::Float      ? data_type::f32
      : query.scalar_type() == c10::ScalarType::Half     ? data_type::f16
      : query.scalar_type() == c10::ScalarType::BFloat16 ? data_type::bf16
                                                         : data_type::undef;
  TORCH_INTERNAL_ASSERT(
      (logical_tensor_dtype != data_type::undef),
      "Only FP16/BF16/FP32 datatypes are currently supported");

  // align mask dim for OneDNN ukernel
  if (attn_mask.has_value()) {
    for (size_t i = attn_mask->dim(); i < 4; i++) {
      attn_mask = at::unsqueeze(*attn_mask, 0);
    }
  }

  thread_local static PartitionCache cache;

  // cache key creation
  // patternID is determined on the basis of the arguments provided
  std::bitset<32> patternID;
  if (logical_tensor_dtype == data_type::f32) {
    // bit 3 corresponds to float32 dtype
    patternID.set(3, 1);
  }
  if (logical_tensor_dtype == data_type::bf16) {
    // bit 2 corresponds to fp16/bf16 dtype
    patternID.set(2, 1);
  }
  // sdp pattern
  patternID.set(4, 1);

  // Refer to comments in Graph.cpp. The first 8 bits are reserved
  int pos = 8;
  // attn_mask
  patternID.set(pos++, attn_mask.has_value());
  patternID.set(pos++, is_causal);

  const SDPALogicalParams logical_params(
      query, key, value, attn_mask, output, logical_tensor_dtype, is_causal);
  auto partition_ = cache.find_partition(patternID);
  if (!partition_.has_value()) {
    // partition cache no hit
    // graph building and partitioning
    partition sdp_partition = create_sdpa_graph_partition(
        batch_size,
        seq_len_q,
        seq_len_k,
        num_head,
        head_dim,
        is_causal,
        logical_tensor_dtype,
        logical_params);
    partition_ = cache.insert_partition_cache(patternID, sdp_partition);
  }
  const auto l_inputs = logical_params.get_input();
  const auto l_outputs = logical_params.get_output();
  // partition compilation
  auto compiled_partition = partition_->get().compile(l_inputs, l_outputs, eng);

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
  compiled_partition.execute(strm, inputs, outputs);
}
} // namespace at::native::onednn
