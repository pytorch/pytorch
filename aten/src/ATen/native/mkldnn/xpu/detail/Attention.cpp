#include <ATen/native/mkldnn/xpu/detail/Attr.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>

#include <oneapi/dnnl/dnnl.hpp>
namespace {
using namespace at::native::onednn::graph;

using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;
using RunArg = dnnl::graph::tensor;
using RunArgs = std::vector<RunArg>;
using LogicalTensors = std::vector<logical_tensor>;
using engine = dnnl::engine;

void allocate_sycl_graph_mem(
    std::vector<dnnl::graph::tensor>& tensors,
    const logical_tensor& lt,
    const engine& eng,
    const at::Tensor& input) {
  dnnl::graph::tensor new_ts{lt, eng, input.data_ptr()};
  tensors.push_back(new_ts);
}

std::vector<partition> graph_building_and_partitioning(
    int batch_size,
    int seq_len_q,
    int seq_len_k,
    int num_head,
    int size_per_head,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& attn_mask,
    const at::Tensor& output,
    data_type dtype) {
  // graph building and partitioning
  // currently, we assume that Q and K have same sequence length

  dims q_input_shape = {batch_size, num_head, seq_len_q, size_per_head};
  dims kv_input_shape = {batch_size, num_head, seq_len_k, size_per_head};
  dims qk_output_shape = {batch_size, num_head, seq_len_q, seq_len_k};
  dims scale_shape = {1};
  dims attention_mask_shape = {attn_mask.sizes().vec()};
  size_t lt_id = 0;

  logical_tensor query_input{
      lt_id++, dtype, q_input_shape, query.strides().vec()};
  logical_tensor key_input{lt_id++, dtype, kv_input_shape, key.strides().vec()};

  logical_tensor matmul_qk_out{
      lt_id++, dtype, qk_output_shape, logical_tensor::layout_type::strided};
  op matmul_qk{
      0,
      op::kind::MatMul,
      {query_input, key_input},
      {matmul_qk_out},
      "matmul_qk"};
  matmul_qk.set_attr<bool>(op::attr::transpose_b, true);

  logical_tensor scale_factor{
      lt_id++,
      dtype,
      scale_shape,
      logical_tensor::layout_type::strided,
      logical_tensor::property_type::constant};
  logical_tensor scaled_qk_out{
      lt_id++, dtype, qk_output_shape, logical_tensor::layout_type::strided};
  op scale_div{
      1,
      op::kind::Divide,
      {matmul_qk_out, scale_factor},
      {scaled_qk_out},
      "scale_div"};

  logical_tensor attention_mask{
      lt_id++, dtype, attention_mask_shape, attn_mask.strides().vec()};
  logical_tensor masked_qk_out{
      lt_id++, dtype, qk_output_shape, logical_tensor::layout_type::strided};
  op mask_add{
      2,
      op::kind::Add,
      {scaled_qk_out, attention_mask},
      {masked_qk_out},
      "mask_add"};

  op softmax{3, op::kind::SoftMax, "softmax"};
  softmax.set_attr<int64_t>(op::attr::axis, -1);

  logical_tensor softmax_out{
      lt_id++, dtype, qk_output_shape, logical_tensor::layout_type::strided};
  softmax.add_input(masked_qk_out);
  softmax.add_output(softmax_out);

  logical_tensor value_input{
      lt_id++, dtype, kv_input_shape, value.strides().vec()};
  logical_tensor matmul_v_out{
      lt_id++, dtype, q_input_shape, output.strides().vec()};

  op matmul_v{
      4,
      op::kind::MatMul,
      {softmax_out, value_input},
      {matmul_v_out},
      "matmul_v"};

  engine::kind ekind = engine::kind::gpu;
  graph g(ekind);
  g.add_op(matmul_qk);
  g.add_op(scale_div);
  g.add_op(mask_add);
  g.add_op(softmax);
  g.add_op(matmul_v);
  g.finalize();

  return g.get_partitions();
}
} // namespace

namespace at::native::onednn::graph {

TORCH_API void gpu_float_sdpa(
    int batch_size,
    int seq_len_q,
    int seq_len_k,
    int num_head,
    int size_per_head,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& attn_mask,
    const float& softmax_scale,
    const Tensor& output) {
  auto eng = GpuEngineManager::Instance().get_engine(
      {c10::kXPU, c10::xpu::current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  Tensor softmax_scale1 = at::full(
      {},
      1 / softmax_scale,
      TensorOptions().dtype(c10::ScalarType::Half).device(DeviceType::XPU));

  const data_type logical_tensor_dtype =
      query.scalar_type() == c10::ScalarType::Float      ? data_type::f32
      : query.scalar_type() == c10::ScalarType::Half     ? data_type::f16
      : query.scalar_type() == c10::ScalarType::BFloat16 ? data_type::bf16
                                                         : data_type::undef;
  TORCH_CHECK(
      (logical_tensor_dtype != data_type::undef),
      "Only F16 & BF16 & FP32 datatypes are currently supported");
  // graph building and partitioning
  std::vector<partition> partitions = graph_building_and_partitioning(
      batch_size,
      seq_len_q,
      seq_len_k,
      num_head,
      size_per_head,
      query,
      key,
      value,
      attn_mask,
      output,
      logical_tensor_dtype);

  TORCH_CHECK(partitions.size() == 1);
  partition sdp_partition = partitions[0];

  std::vector<logical_tensor> inputs = sdp_partition.get_input_ports();
  std::vector<logical_tensor> outputs = sdp_partition.get_output_ports();
  compiled_partition cp = sdp_partition.compile(inputs, outputs, eng);

  // partition execution
  std::vector<dnnl::graph::tensor> inputs_ts, outputs_ts;
  inputs_ts.reserve(inputs.size());
  outputs_ts.reserve(outputs.size());
  allocate_sycl_graph_mem(inputs_ts, inputs[0], eng, query);
  allocate_sycl_graph_mem(inputs_ts, inputs[1], eng, key);
  allocate_sycl_graph_mem(inputs_ts, inputs[2], eng, softmax_scale1);
  allocate_sycl_graph_mem(inputs_ts, inputs[3], eng, attn_mask);
  allocate_sycl_graph_mem(inputs_ts, inputs[4], eng, value);
  allocate_sycl_graph_mem(inputs_ts, inputs[3], eng, value);
  allocate_sycl_graph_mem(outputs_ts, outputs[0], eng, output);

  cp.execute(strm, inputs_ts, outputs_ts);
}
} // namespace at::native::onednn::graph
