#include <ATen/native/mkldnn/Attention.h>

#if AT_ONEDNN_GRAPH_ENABLED()

namespace at {
namespace native {
namespace onednn_graph {

void create_partition(
    std::bitset<32>& patternID,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& scale,
    const c10::optional<Tensor>& attn_mask) {
  dnnl::graph::graph g{dnnl::graph::engine::kind::cpu};
  size_t op_idx = 0;
  size_t logical_tensor_id = 0;
  auto dtype = aten_to_onednn_graph_dtype(query.scalar_type());
  // input tensors
  logical_tensor query_src_desc(logical_tensor_id++, dtype);
  logical_tensor key_src_desc(logical_tensor_id++, dtype);
  logical_tensor value_src_desc(logical_tensor_id++, dtype);
  logical_tensor scale_desc(logical_tensor_id++, dtype);
  logical_tensor attn_mask_desc;
  if (attn_mask.has_value()) {
    attn_mask_desc = logical_tensor(logical_tensor_id++, dtype);
  }

  // output tensors
  logical_tensor output_dst_desc(logical_tensor_id++, dtype);

  // first matmul
  logical_tensor matmul_qk_dst_desc(logical_tensor_id++, dtype);
  op matmul_qk(
      op_idx++,
      op::kind::MatMul,
      {query_src_desc, key_src_desc},
      {matmul_qk_dst_desc},
      "matmul_qk");
  matmul_qk.set_attr<bool>(op::attr::transpose_b, true);
  g.add_op(matmul_qk);

  logical_tensor post_scale_desc(logical_tensor_id++, dtype);
  op fscore_mul(
      op_idx++,
      op::kind::Multiply,
      {matmul_qk_dst_desc,
        scale_desc},
      {post_scale_desc},
      "fscore_mul");
  g.add_op(fscore_mul);

  logical_tensor fscore_add_desc(logical_tensor_id++, dtype);
  if (attn_mask.has_value()) {
    op fscore_add(
        op_idx++,
        op::kind::Add,
        {post_scale_desc,
         attn_mask_desc},
        {fscore_add_desc},
        "fscore_add");
    g.add_op(fscore_add);
  }

  logical_tensor softmax_out_dst_desc(logical_tensor_id++, dtype);
  op softmax_out(
      op_idx++,
      op::kind::SoftMax,
      {attn_mask.has_value()
           ? fscore_add_desc
           : post_scale_desc},
      {softmax_out_dst_desc},
      "softmax_out");
  softmax_out.set_attr<dim>(op::attr::axis, -1);
  g.add_op(softmax_out);

  logical_tensor matmul_v_dst_desc(logical_tensor_id++, dtype);
  op matmul_v(
      op_idx++,
      op::kind::MatMul,
      {softmax_out_dst_desc,
       value_src_desc},
      {output_dst_desc},
      "matmul_value");
  g.add_op(matmul_v);

  g.finalize();
  auto partitions = g.get_partitions();
  auto partition = partitions[0];
  TORCH_CHECK(
      (partitions.size() == 1) && partition.is_supported(),
      "oneDNN Graph doesn't support this fusion pattern. If you'd like its support, please submit a ticket.");

  insert_in_partition_cache(patternID, partition);
}

void compile_and_cache_sdpa_fusion(
    std::vector<Tensor>& input_tensors,
    Tensor& output_tensor,
    cp_entry& cp,
    std::bitset<32>& patternID) {
  // assuming all inputs have the same dtype. Might revisit this assumption
  // later
  int i = 0;
  for (auto& each_tensor : input_tensors) {
    cp.inputLogicalTensors_.emplace_back(logical_tensor(
        i,
        aten_to_onednn_graph_dtype(each_tensor.scalar_type()),
        each_tensor.sizes().vec(),
        each_tensor.strides().vec()));
    cp.inputLLGATensors_.emplace_back(RunArg(
        cp.inputLogicalTensors_[i++],
        onednn_graph::Engine::getEngine(),
        each_tensor.data_ptr()));
  }

  std::vector<int64_t> output_sizes(4);
  std::vector<int64_t> output_strides(4);
  output_sizes = output_tensor.sizes().vec();
  output_strides = output_tensor.strides().vec();
  cp.outputTensorShapes_.push_back(output_sizes);
  cp.outputTensorStrides_.push_back(output_strides);
  cp.outputLogicalTensors_.emplace_back(logical_tensor(
      i,
      aten_to_onednn_graph_dtype(input_tensors[0].scalar_type()),
      output_sizes,
      output_strides));

  // provide any data pointer because we'd change these at runtime, anyway
  cp.outputLLGATensors_.emplace_back(RunArg(
      cp.outputLogicalTensors_[0],
      onednn_graph::Engine::getEngine(),
      output_tensor.data_ptr()));

  cp.cp_ = compile_partition(
      cp.partition_, cp.inputLogicalTensors_, cp.outputLogicalTensors_);
}

} // end namespace onednn_graph
} // end namespace native
} // end namespace at

#endif // AT_ONEDNN_GRAPH_ENABLED()
