#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/util/ArrayRef.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_to_dense_native.h>
#endif

#if AT_ONEDNN_GRAPH_ENABLED()

#include <ATen/native/mkldnn/Graph.h>
#include <ATen/native/mkldnn/Utils.h>
#include <c10/util/irange.h>
#include <omp.h>
#include <oneapi/dnnl/dnnl_graph.hpp>

namespace at {
namespace native {

namespace {

using namespace onednn_graph;

/*

GPT2 MHA kernel

Attention-mask is non-optional but would be
optional in oneDNN v3.5 for better performance.

 [query]    [key]
      \     /
[cond] MatMul [mask]
     \   |   /
       Select   [scale]
            \   /
             Div   [add in]
               \   /
                Add
                 |
               Softmax   [value]
                    \     /
                     MatMul
                       |
                StaticTranspose
                       |
                    Reorder
                       |
                    [output]
*/
void create_graph_sdpa_pattern_18(data_type dtype) {
  dnnl::graph::graph g{dnnl::graph::engine::kind::cpu};
  size_t op_idx = 0;
  size_t logical_tensor_id = 0;

  // input tensors
  logical_tensor q_trans_src_desc(logical_tensor_id++, dtype);
  logical_tensor k_trans_src_desc(logical_tensor_id++, dtype);
  logical_tensor v_trans_src_desc(logical_tensor_id++, dtype);
  logical_tensor fscore_scale_desc(logical_tensor_id++, dtype);
  logical_tensor attn_mask_desc(logical_tensor_id++, dtype);
  logical_tensor causal_mask_desc(logical_tensor_id++, data_type::boolean);
  logical_tensor causal_mask_value_desc(logical_tensor_id++, dtype);

  // output tensors
  logical_tensor output_dst_desc(logical_tensor_id++, dtype);

  logical_tensor matmul_qk_dst_desc(logical_tensor_id++, dtype);
  op matmul_qk(
      op_idx++,
      op::kind::MatMul,
      {q_trans_src_desc, k_trans_src_desc},
      {matmul_qk_dst_desc},
      "matmul_qk");

  logical_tensor attn_scores_desc(logical_tensor_id++, dtype);
  op select(
      op_idx++,
      op::kind::Select,
      {causal_mask_desc, matmul_qk_dst_desc, causal_mask_value_desc},
      {attn_scores_desc},
      "fscore_where");

  logical_tensor fscore_div_dst_desc(logical_tensor_id++, dtype);
  op fscore_div(
      op_idx++,
      op::kind::Divide,
      {attn_scores_desc, fscore_scale_desc},
      {fscore_div_dst_desc},
      "fscore_div");

  logical_tensor fscore_add_dst_desc(logical_tensor_id++, dtype);
  op fscore_add(
      op_idx++,
      op::kind::Add,
      {fscore_div_dst_desc, attn_mask_desc},
      {fscore_add_dst_desc},
      "fscore_add");

  logical_tensor softmax_out_dst_desc(logical_tensor_id++, dtype);
  op softmax_out(
      op_idx++,
      op::kind::SoftMax,
      {fscore_add_dst_desc},
      {softmax_out_dst_desc},
      "softmax_out");
  softmax_out.set_attr<dim>(op::attr::axis, -1);

  logical_tensor matmul_v_dst_desc(logical_tensor_id++, dtype);
  op matmul_v(
      op_idx++,
      op::kind::MatMul,
      {softmax_out_dst_desc, v_trans_src_desc},
      {matmul_v_dst_desc},
      "matmul_value");

  logical_tensor matmul_v_trans_dst_desc(logical_tensor_id++, dtype);
  op matmul_v_transpose(
      op_idx++,
      op::kind::StaticTranspose,
      {matmul_v_dst_desc},
      {matmul_v_trans_dst_desc},
      "matmul_value_transpose");
  std::vector<int64_t> transpose_dims{0, 2, 1, 3};
  matmul_v_transpose.set_attr<dims>(
      dnnl::graph::op::attr::order, transpose_dims);

  op reorder_output(
      op_idx++,
      op::kind::Reorder,
      {matmul_v_trans_dst_desc},
      {output_dst_desc},
      "reorder_output");

  g.add_op(matmul_qk);
  g.add_op(select);
  g.add_op(fscore_div);
  g.add_op(fscore_add);
  g.add_op(softmax_out);
  g.add_op(matmul_v);
  g.add_op(matmul_v_transpose);
  g.add_op(reorder_output);

  g.finalize();
  auto partitions = g.get_partitions();
  auto partition = partitions[0];
  TORCH_CHECK(
      (partitions.size() == 1) && partition.is_supported(),
      " only one fusion group allowed");
  int patternID = dtype == data_type::bf16 ? ONEDNN_GRAPH_SDPA_PATTERN_18_BF16
                                           : ONEDNN_GRAPH_SDPA_PATTERN_18_FP32;
  insert_in_partition_cache(patternID, partition);
}

void create_partition(int64_t patternID) {
  switch (patternID) {
    case ONEDNN_GRAPH_SDPA_PATTERN_18_FP32:
      create_graph_sdpa_pattern_18(data_type::f32);
      break;
    case ONEDNN_GRAPH_SDPA_PATTERN_18_BF16:
      create_graph_sdpa_pattern_18(data_type::bf16);
      break;
    default:
      TORCH_CHECK(false, "Unsupported patternID");
  }
}

void compile_and_cache_pattern(
    int64_t patternID,
    dnnl::graph::partition& partition,
    std::vector<Tensor>& input_tensors,
    cp_entry& cp) {
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

  switch (patternID) {
    case ONEDNN_GRAPH_SDPA_PATTERN_18_FP32:
    case ONEDNN_GRAPH_SDPA_PATTERN_18_BF16: {
      auto output_size = input_tensors[0].sizes().vec();
      std::swap(output_size[1], output_size[2]);
      std::vector<int64_t> output_strides{
          output_size[1] * output_size[2] * output_size[3],
          output_size[2] * output_size[3],
          output_size[3],
          1};
      cp.outputLogicalTensors_.emplace_back(logical_tensor(
          i,
          aten_to_onednn_graph_dtype(input_tensors[0].scalar_type()),
          output_size,
          output_strides));
      break;
    }
  }
  // output is of the same size as query or value
  // In this pattern, key is of a different shape
  cp.outputLLGATensors_.emplace_back(RunArg(
      cp.outputLogicalTensors_[0],
      onednn_graph::Engine::getEngine(),
      input_tensors[0].data_ptr()));

  cp.partition_ = partition;
  cp.cp_ = compile_partition(
      partition, cp.inputLogicalTensors_, cp.outputLogicalTensors_);
}

// Execute SDPA partition
// TODO: save output shape in cp_entry in meta kernel
at::Tensor execute_sdpa_partition(
    int64_t patternID,
    std::vector<Tensor>& input_tensors,
    cp_entry& cp,
    bool inplace = false) {
  int i = 0;

  for (auto& each_tensor : input_tensors) {
    cp.inputLLGATensors_[i++].set_data_handle(each_tensor.data_ptr());
  }

  at::Tensor output_tensor;
  if (inplace) {
    // there's no copy, so it's fine
    output_tensor = input_tensors[0];
  } else {
    switch (patternID) {
      case ONEDNN_GRAPH_SDPA_PATTERN_18_FP32:
      case ONEDNN_GRAPH_SDPA_PATTERN_18_BF16: {
        auto output_size = input_tensors[0].sizes().vec();
        std::swap(output_size[1], output_size[2]);
        std::vector<int64_t> output_strides{
            output_size[1] * output_size[2] * output_size[3],
            output_size[2] * output_size[3],
            output_size[3],
            1};
        output_tensor = at::detail::empty_strided_cpu(
            output_size, output_strides, input_tensors[1].scalar_type());
        break;
      }
    }
  }
  cp.outputLLGATensors_[0].set_data_handle(output_tensor.data_ptr());
  cp.cp_.execute(
      onednn_graph::Stream::getStream(),
      cp.inputLLGATensors_,
      cp.outputLLGATensors_);
  return output_tensor;
}

Tensor mkldnn_graph_sdpa_pattern(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& scale,
    const c10::optional<Tensor>& inverse_scale,
    const c10::optional<Tensor>& attn_mask,
    const c10::optional<Tensor>& causal_mask,
    const c10::optional<Tensor>& causal_mask_value,
    bool query_requires_transpose = false,
    bool key_requires_transpose_twice = false,
    bool key_requires_transpose_once = false,
    bool value_requires_transpose = false,
    bool output_requires_transpose_and_reorder = false) {
  // first check cache
  // The key has a pattern ID, as well as the shapes of input tenors
  std::vector<int64_t> map_key;
  map_key.reserve(1024);
  // We use this because different thread-pools may be used
  map_key.push_back(omp_get_max_threads());
  // Algo ID
  int64_t patternID = -1;
  std::vector<Tensor> input_tensors;
  if (output_requires_transpose_and_reorder && causal_mask.has_value() &&
      attn_mask.has_value()) {
    patternID = query.scalar_type() == c10::ScalarType::Float
        ? ONEDNN_GRAPH_SDPA_PATTERN_18_FP32
        : ONEDNN_GRAPH_SDPA_PATTERN_18_BF16;
    input_tensors.reserve(7);
  }
  input_tensors.push_back(query);
  input_tensors.push_back(key);
  input_tensors.push_back(value);

  map_key.push_back(patternID);

  map_key.insert(map_key.end(), key.sizes().begin(), key.sizes().end());
  map_key.insert(map_key.end(), query.sizes().begin(), query.sizes().end());
  map_key.insert(map_key.end(), value.sizes().begin(), value.sizes().end());
  if (scale != c10::nullopt) {
    auto scale_val = scale.value();
    input_tensors.push_back(scale_val);
    map_key.insert(
        map_key.end(), scale_val.sizes().begin(), scale_val.sizes().end());
  }
  if (inverse_scale != c10::nullopt) {
    auto scale_val = inverse_scale.value();
    input_tensors.push_back(scale_val);
    map_key.insert(
        map_key.end(), scale_val.sizes().begin(), scale_val.sizes().end());
  }
  if (attn_mask != c10::nullopt) {
    auto attn_mask_val = attn_mask.value();
    input_tensors.push_back(attn_mask_val);
    map_key.insert(
        map_key.end(),
        attn_mask_val.sizes().begin(),
        attn_mask_val.sizes().end());
  }
  if (causal_mask != c10::nullopt) {
    auto causal_mask_val = causal_mask.value();
    input_tensors.push_back(causal_mask_val);
    map_key.insert(
        map_key.end(),
        causal_mask_val.sizes().begin(),
        causal_mask_val.sizes().end());
    // if causal mask is present, its value must be present as well
    auto causal_mask_value_val = causal_mask_value.value();
    input_tensors.push_back(causal_mask_value_val);
  }
  auto iter = cache_lookup(map_key);
  if (iter == cache_end()) {
    cp_entry compiledPartitionEntry;
    auto graph_partition_iter = partition_map_lookup(patternID);
    partition graph_partition;
    if (graph_partition_iter == partition_map_end()) {
      auto dtype = query.scalar_type();
      TORCH_CHECK(
          ((dtype == at::ScalarType::Float) ||
           (dtype == at::ScalarType::BFloat16)),
          "Only BF16 & FP32 datatypes are currently supported");
      create_partition(patternID);
      graph_partition_iter = partition_map_lookup(patternID);
    }
    graph_partition = graph_partition_iter->second;
    compile_and_cache_pattern(
        patternID, graph_partition, input_tensors, compiledPartitionEntry);
    auto retVal = execute_sdpa_partition(
        patternID, input_tensors, compiledPartitionEntry);
    insert_in_fused_kernel_cache(map_key, compiledPartitionEntry);
    return retVal;
  } else {
    change_pos_in_list(iter->second);
    cp_entry& cp = iter->second->second;
    return execute_sdpa_partition(patternID, input_tensors, cp, false);
  }
}

} // end anonymous namespace
} // namespace native

namespace meta {
namespace {

using namespace at::native;
using namespace at::native::onednn_graph;

bool is_any_shape_symbolic(SymIntArrayRef& shape) {
  auto shape_vec = shape.vec();
  for (auto& shape_symbol : shape_vec) {
    if (shape_symbol.is_symbolic()) {
      return true;
    }
  }
  return false;
}

// Compile fused kernels in Inductor compilation stage
Tensor mkldnn_graph_sdpa_pattern_meta(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& scale,
    const c10::optional<Tensor>& inverse_scale,
    const c10::optional<Tensor>& attn_mask,
    const c10::optional<Tensor>& causal_mask,
    const c10::optional<Tensor>& causal_mask_value,
    bool query_requires_transpose = false,
    bool key_requires_transpose_twice = false,
    bool key_requires_transpose_once = false,
    bool value_requires_transpose = false,
    bool output_requires_transpose_and_reorder = false) {
  // first check cache
  // The key has a pattern ID, as well as the shapes of input tenors
  std::vector<int64_t> map_key;
  map_key.reserve(1024);
  // We use this because different thread-pools may be used
  map_key.push_back(omp_get_max_threads());
  // Algo ID
  int64_t patternID = -1;
  std::vector<Tensor> input_tensors;
  if (output_requires_transpose_and_reorder && causal_mask.has_value() &&
      attn_mask.has_value()) {
    patternID = query.scalar_type() == c10::ScalarType::Float
        ? ONEDNN_GRAPH_SDPA_PATTERN_18_FP32
        : ONEDNN_GRAPH_SDPA_PATTERN_18_BF16;
    input_tensors.reserve(7);
  }
  input_tensors.push_back(query);
  input_tensors.push_back(key);
  input_tensors.push_back(value);

  map_key.push_back(patternID);
  auto key_sym_sizes = key.sym_sizes();
  if (is_any_shape_symbolic(key_sym_sizes)) {
    return query;
  }
  auto key_sym_sizes_vec = asIntArrayRefUnchecked(key_sym_sizes).vec();
  std::vector<int64_t> key_sizes(
      key_sym_sizes_vec.begin(), key_sym_sizes_vec.end());

  auto query_sym_sizes = query.sym_sizes();
  if (is_any_shape_symbolic(query_sym_sizes)) {
    return query;
  }
  auto query_sym_sizes_vec = asIntArrayRefUnchecked(query_sym_sizes).vec();
  std::vector<int64_t> query_sizes(
      query_sym_sizes_vec.begin(), query_sym_sizes_vec.end());

  auto value_sym_sizes = value.sym_sizes();
  if (is_any_shape_symbolic(value_sym_sizes)) {
    return query;
  }
  auto value_sym_sizes_vec = asIntArrayRefUnchecked(value_sym_sizes).vec();
  std::vector<int64_t> value_sizes(
      value_sym_sizes_vec.begin(), value_sym_sizes_vec.end());
  map_key.insert(map_key.end(), key_sizes.begin(), key_sizes.end());
  map_key.insert(map_key.end(), query_sizes.begin(), query_sizes.end());
  map_key.insert(map_key.end(), value_sizes.begin(), value_sizes.end());

  if (scale.has_value()) {
    input_tensors.push_back(scale.value());
    auto scale_sym_sizes = scale.value().sym_sizes();
    if (is_any_shape_symbolic(scale_sym_sizes)) {
      return query;
    }
    auto scale_sym_sizes_vec = asIntArrayRefUnchecked(scale_sym_sizes).vec();
    std::vector<int64_t> scale_sizes(
        scale_sym_sizes_vec.begin(), scale_sym_sizes_vec.end());
    map_key.insert(map_key.end(), scale_sizes.begin(), scale_sizes.end());
  } else if (inverse_scale.has_value()) {
    input_tensors.push_back(inverse_scale.value());
    auto scale_sym_sizes = inverse_scale.value().sym_sizes();
    if (is_any_shape_symbolic(scale_sym_sizes)) {
      return query;
    }
    auto scale_sym_sizes_vec = asIntArrayRefUnchecked(scale_sym_sizes).vec();
    std::vector<int64_t> scale_sizes(
        scale_sym_sizes_vec.begin(), scale_sym_sizes_vec.end());
    map_key.insert(map_key.end(), scale_sizes.begin(), scale_sizes.end());
  }

  if (attn_mask.has_value()) {
    input_tensors.push_back(attn_mask.value());
    auto attn_mask_sym_sizes = attn_mask.value().sym_sizes();
    if (is_any_shape_symbolic(attn_mask_sym_sizes)) {
      return query;
    }
    auto attn_mask_sym_sizes_vec =
        asIntArrayRefUnchecked(attn_mask_sym_sizes).vec();
    std::vector<int64_t> attn_mask_sizes(
        attn_mask_sym_sizes_vec.begin(), attn_mask_sym_sizes_vec.end());
    map_key.insert(
        map_key.end(), attn_mask_sizes.begin(), attn_mask_sizes.end());
  }

  if (causal_mask.has_value()) {
    input_tensors.push_back(causal_mask.value());
    auto causal_mask_sym_sizes = causal_mask.value().sym_sizes();
    if (is_any_shape_symbolic(causal_mask_sym_sizes)) {
      return query;
    }
    auto causal_mask_sym_sizes_vec =
        asIntArrayRefUnchecked(causal_mask_sym_sizes).vec();
    std::vector<int64_t> causal_mask_sizes(
        causal_mask_sym_sizes_vec.begin(), causal_mask_sym_sizes_vec.end());
    map_key.insert(
        map_key.end(), causal_mask_sizes.begin(), causal_mask_sizes.end());
  }

  if (causal_mask_value.has_value()) {
    input_tensors.push_back(causal_mask_value.value());
    auto causal_mask_val_sym_sizes = causal_mask_value.value().sym_sizes();
    if (is_any_shape_symbolic(causal_mask_val_sym_sizes)) {
      return query;
    }
    auto causal_mask_val_sym_sizes_vec =
        asIntArrayRefUnchecked(causal_mask_val_sym_sizes).vec();
    std::vector<int64_t> causal_mask_val_sizes(
        causal_mask_val_sym_sizes_vec.begin(),
        causal_mask_val_sym_sizes_vec.end());
    map_key.insert(
        map_key.end(),
        causal_mask_val_sizes.begin(),
        causal_mask_val_sizes.end());
  }

  auto iter = cache_lookup(map_key);
  if (iter == cache_end()) {
    cp_entry compiledPartitionEntry;
    auto graph_partition_iter = partition_map_lookup(patternID);
    partition graph_partition;
    if (graph_partition_iter == partition_map_end()) {
      auto dtype = query.scalar_type();
      TORCH_CHECK(
          ((dtype == at::ScalarType::Float) ||
           (dtype == at::ScalarType::BFloat16)),
          "Only BF16 & FP32 datatypes are currently supported");
      create_partition(patternID);
      graph_partition_iter = partition_map_lookup(patternID);
    }
    graph_partition = graph_partition_iter->second;
    compile_and_cache_pattern(
        patternID, graph_partition, input_tensors, compiledPartitionEntry);
    insert_in_fused_kernel_cache(map_key, compiledPartitionEntry);
  } else {
    change_pos_in_list(iter->second);
  }
  at::Tensor output_tensor;
  switch (patternID) {
    case ONEDNN_GRAPH_SDPA_PATTERN_18_FP32:
    case ONEDNN_GRAPH_SDPA_PATTERN_18_BF16: {
      auto output_size = input_tensors[0].sizes().vec();
      std::swap(output_size[1], output_size[2]);
      std::vector<int64_t> output_strides{
          output_size[1] * output_size[2] * output_size[3],
          output_size[2] * output_size[3],
          output_size[3],
          1};
      output_tensor = at::detail::empty_strided_meta(
          output_size, output_strides, input_tensors[1].scalar_type());
      break;
    }
  }
  return output_tensor;
}

} // end anonymous namespace

TORCH_LIBRARY_IMPL(mkldnn, Meta, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_graph_sdpa_pattern"),
      c10::DispatchKey::Meta,
      TORCH_FN(mkldnn_graph_sdpa_pattern_meta));
}
} // namespace meta

namespace native {
TORCH_LIBRARY_IMPL(mkldnn, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_graph_sdpa_pattern"),
      TORCH_FN(mkldnn_graph_sdpa_pattern));
}
} // namespace native
} // namespace at
#endif // AT_ONEDNN_GRAPH_ENABLED
