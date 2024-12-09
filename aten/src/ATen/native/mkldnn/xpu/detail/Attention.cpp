#include <ATen/native/mkldnn/xpu/detail/Attr.h>
#include <ATen/native/mkldnn/xpu/detail/Graph.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>

#include <omp.h>
#include <oneapi/dnnl/dnnl.hpp>

using namespace at::native::onednn::graph;
namespace {
struct SDPALogicalParams {
  enum class TensorID {
    query,
    key,
    scale,
    attn_mask,
    value,
    output,
    end,
  };

  logical_tensor query;
  logical_tensor key;
  logical_tensor scale;
  std::optional<logical_tensor> attn_mask;
  logical_tensor value;
  logical_tensor output;

  SDPALogicalParams(
      const at::Tensor& query_,
      const at::Tensor& key_,
      const at::Tensor& value_,
      const std::optional<at::Tensor>& attn_mask_,
      const at::Tensor& output_,
      data_type dtype) {
    dims scale_shape = {1};
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
        scale_shape,
        logical_tensor::layout_type::strided,
        logical_tensor::property_type::constant};
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
  op scale_div{
      op_id++,
      op::kind::Divide,
      {matmul_qk_out, params.scale},
      {scaled_qk_out},
      "scale_div"};

  std::optional<op> mask_add;
  std::optional<logical_tensor> masked_qk_out;
  if (params.attn_mask.has_value()) {
    masked_qk_out = {lt_id++, dtype};
    mask_add = {
        op_id++,
        op::kind::Add,
        {scaled_qk_out, params.attn_mask.value()},
        {masked_qk_out.value()},
        "mask_add"};
  } else if (is_causal) {
    TORCH_CHECK(false, "Causal mask must use fallback mask for now.");
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

  engine::kind ekind = engine::kind::gpu;
  graph g(ekind);
  g.add_op(matmul_qk);
  g.add_op(scale_div);
  if (mask_add.has_value()) {
    g.add_op(mask_add.value());
  }
  g.add_op(softmax);
  g.add_op(matmul_v);
  g.finalize();
  auto partitions = g.get_partitions();
  TORCH_CHECK(
      (partitions.size() == 1) && partitions[0].is_supported(),
      "oneDNN Graph doesn't support this fusion pattern. If you'd like its support, please submit a issue.");
  return partitions[0];
}
} // namespace

namespace at::native::onednn::graph {
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
    const std::optional<at::Tensor>& attn_mask,
    bool is_causal,
    float softmax_scale,
    const Tensor& output) {
  auto eng = GpuEngineManager::Instance().get_engine(
      {c10::kXPU, c10::xpu::current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  Tensor softmax_scale1 = at::full({}, 1 / softmax_scale, query.options());

  const data_type logical_tensor_dtype =
      query.scalar_type() == c10::ScalarType::Float      ? data_type::f32
      : query.scalar_type() == c10::ScalarType::Half     ? data_type::f16
      : query.scalar_type() == c10::ScalarType::BFloat16 ? data_type::bf16
                                                         : data_type::undef;
  TORCH_CHECK(
      (logical_tensor_dtype != data_type::undef),
      "Only FP16/BF16/FP32 datatypes are currently supported");

  thread_local static GraphCache cache;

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

  // first check cache
  // The key has a pattern ID, as well as the shapes of input tenors
  std::vector<int64_t> map_key;
  map_key.reserve(1024);
  // We use this because different thread-pools may be used
  map_key.push_back(omp_get_max_threads());
  map_key.push_back(static_cast<int64_t>(patternID.to_ullong()));
  map_key.insert(map_key.end(), query.sizes().begin(), query.sizes().end());
  map_key.insert(map_key.end(), query.strides().begin(), query.strides().end());
  map_key.insert(map_key.end(), key.sizes().begin(), key.sizes().end());
  map_key.insert(map_key.end(), key.strides().begin(), key.strides().end());
  map_key.insert(map_key.end(), value.sizes().begin(), value.sizes().end());
  map_key.insert(map_key.end(), value.strides().begin(), value.strides().end());
  map_key.insert(
      map_key.end(),
      softmax_scale1.sizes().begin(),
      softmax_scale1.sizes().end());
  if (attn_mask.has_value()) {
    map_key.insert(
        map_key.end(), attn_mask->sizes().begin(), attn_mask->sizes().end());
    map_key.insert(
        map_key.end(),
        attn_mask->strides().begin(),
        attn_mask->strides().end());
  }

  auto cp_entry_ref = cache.find_kernel(map_key);
  if (!cp_entry_ref.has_value()) {
    SDPALogicalParams logical_params(
        query, key, value, attn_mask, output, logical_tensor_dtype);

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
    cp_entry sdp_cp_entry{
        /*.partition_ = */ partition_->get(),
        /*.input_logical_tensors = */ logical_params.get_input(),
        /*.output_logical_tensors = */ logical_params.get_output(),
    };
    // partition compilation
    sdp_cp_entry.cp = sdp_cp_entry.partition_.compile(
        sdp_cp_entry.input_logical_tensors,
        sdp_cp_entry.output_logical_tensors,
        eng);
    cp_entry_ref = cache.insert_fused_kernel_cache(map_key, sdp_cp_entry);
  }

  // partition execution
  auto& sdp_cp_entry = cp_entry_ref->get();
  const auto& l_inputs = sdp_cp_entry.input_logical_tensors;
  const auto& l_outputs = sdp_cp_entry.output_logical_tensors;

  std::vector<dnnl::graph::tensor> outputs = {
      {l_outputs[0], eng, output.data_ptr()},
  };
  size_t i = 0;
  std::vector<dnnl::graph::tensor> inputs;
  inputs.emplace_back(l_inputs[i++], eng, query.data_ptr());
  inputs.emplace_back(l_inputs[i++], eng, key.data_ptr());
  inputs.emplace_back(l_inputs[i++], eng, softmax_scale1.data_ptr());
  if (attn_mask.has_value()) {
    inputs.emplace_back(l_inputs[i++], eng, attn_mask->data_ptr());
  }
  inputs.emplace_back(l_inputs[i++], eng, value.data_ptr());
  sdp_cp_entry.cp.execute(strm, inputs, outputs);
}
} // namespace at::native::onednn::graph
