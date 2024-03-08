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
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
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

void create_partition(
    std::bitset<32>& patternID,
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
    bool apply_mask_before_scale = false,
    bool choose_causal_mask_over_attn_score = false,
    bool output_requires_transpose_and_reorder = false) {
  dnnl::graph::graph g{dnnl::graph::engine::kind::cpu};
  bool is_scale_present = scale.has_value() || inverse_scale.has_value();
  size_t op_idx = 0;
  size_t logical_tensor_id = 0;
  auto dtype = aten_to_onednn_graph_dtype(query.scalar_type());
  // input tensors
  logical_tensor query_src_desc(logical_tensor_id++, dtype);
  logical_tensor key_src_desc(logical_tensor_id++, dtype);
  logical_tensor value_src_desc(logical_tensor_id++, dtype);
  logical_tensor scale_desc;
  if (scale.has_value()) {
    scale_desc = logical_tensor(logical_tensor_id++, dtype);
  }
  logical_tensor inverse_scale_desc;
  if (inverse_scale.has_value()) {
    inverse_scale_desc = logical_tensor(logical_tensor_id++, dtype);
  }
  logical_tensor attn_mask_desc;
  if (attn_mask.has_value()) {
    attn_mask_desc = logical_tensor(logical_tensor_id++, dtype);
  }
  logical_tensor causal_mask_desc;
  if (causal_mask.has_value()) {
    causal_mask_desc = logical_tensor(logical_tensor_id++, data_type::boolean);
  }
  logical_tensor causal_mask_value_desc;
  if (causal_mask_value.has_value()) {
    causal_mask_value_desc = logical_tensor(logical_tensor_id++, dtype);
  }

  // output tensors
  logical_tensor output_dst_desc(logical_tensor_id++, dtype);

  std::vector<int64_t> query_transpose_dims{0, 2, 1, 3};
  std::vector<int64_t> key_transpose_dims{0, 1, 3, 2};

  logical_tensor transposed_query_desc(logical_tensor_id++, dtype);
  if (query_requires_transpose) {
    op transpose_query_op(
        op_idx++,
        op::kind::StaticTranspose,
        {query_src_desc},
        {transposed_query_desc},
        "query_transpose");
    transpose_query_op.set_attr<dims>(
        dnnl::graph::op::attr::order, query_transpose_dims);
    g.add_op(transpose_query_op);
  }

  logical_tensor transpose_key_first_time_desc(logical_tensor_id++, dtype);
  if (key_requires_transpose_twice) {
    op transpose_once_key_op(
        op_idx++,
        op::kind::StaticTranspose,
        {key_src_desc},
        {transpose_key_first_time_desc},
        "transpose_key_first_time");
    transpose_once_key_op.set_attr<dims>(
        dnnl::graph::op::attr::order, query_transpose_dims);
    g.add_op(transpose_once_key_op);
  }

  logical_tensor transposed_key_before_qk_desc(logical_tensor_id++, dtype);
  if (key_requires_transpose_twice || key_requires_transpose_once) {
    op transpose_key_before_qk_op(
        op_idx++,
        op::kind::StaticTranspose,
        {key_requires_transpose_once ? key_src_desc
                                     : transpose_key_first_time_desc},
        {transposed_key_before_qk_desc},
        "transpose_key");
    transpose_key_before_qk_op.set_attr<dims>(
        dnnl::graph::op::attr::order, key_transpose_dims);
    g.add_op(transpose_key_before_qk_op);
  }

  logical_tensor transposed_value_desc(logical_tensor_id++, dtype);
  if (value_requires_transpose) {
    op transpose_value_op(
        op_idx++,
        op::kind::StaticTranspose,
        {value_src_desc},
        {transposed_value_desc},
        "value_transpose");
    transpose_value_op.set_attr<dims>(
        dnnl::graph::op::attr::order, query_transpose_dims);
    g.add_op(transpose_value_op);
  }

  // first matmul
  logical_tensor matmul_qk_dst_desc(logical_tensor_id++, dtype);
  op matmul_qk(
      op_idx++,
      op::kind::MatMul,
      {query_requires_transpose ? transposed_query_desc : query_src_desc,
       key_requires_transpose_twice || key_requires_transpose_once
           ? transposed_key_before_qk_desc
           : key_src_desc},
      {matmul_qk_dst_desc},
      "matmul_qk");
  g.add_op(matmul_qk);

  // optionally apply causal mask before applying scale
  // this is because GPT2 MHA pattern in oneDNN does so
  // It doesn't bring any performance benefit because the shape is same
  logical_tensor post_qk_causal_mask_desc(logical_tensor_id++, dtype);
  bool mask_before_scale = apply_mask_before_scale && causal_mask.has_value();
  if (mask_before_scale) {
    op post_qk_where(
        op_idx++,
        op::kind::Select,
        {causal_mask_desc,
         choose_causal_mask_over_attn_score ? causal_mask_value_desc
                                            : matmul_qk_dst_desc,
         choose_causal_mask_over_attn_score ? matmul_qk_dst_desc
                                            : causal_mask_value_desc},
        {post_qk_causal_mask_desc},
        "select_after_qk");
    g.add_op(post_qk_where);
  }

  logical_tensor post_scale_desc(logical_tensor_id++, dtype);
  if (scale.has_value()) {
    op fscore_mul(
        op_idx++,
        op::kind::Multiply,
        {mask_before_scale ? post_qk_causal_mask_desc : matmul_qk_dst_desc,
         scale_desc},
        {post_scale_desc},
        "fscore_mul");
    g.add_op(fscore_mul);
  } else if (inverse_scale.has_value()) {
    op fscore_div(
        op_idx++,
        op::kind::Divide,
        {apply_mask_before_scale ? post_qk_causal_mask_desc
                                 : matmul_qk_dst_desc,
         inverse_scale_desc},
        {post_scale_desc},
        "fscore_div");
    g.add_op(fscore_div);
  }

  bool mask_applied_after_scale =
      !apply_mask_before_scale && causal_mask.has_value();
  // in DistilBERT, for example, causal mask is applied after scale
  logical_tensor attn_scores_desc(logical_tensor_id++, dtype);
  if (mask_applied_after_scale) {
    op select(
        op_idx++,
        op::kind::Select,
        {causal_mask_desc,
         causal_mask_value_desc,
         (is_scale_present)
             ? post_scale_desc
             : (mask_before_scale ? choose_causal_mask_over_attn_score
                        ? post_qk_causal_mask_desc
                        : matmul_qk_dst_desc
                                  : choose_causal_mask_over_attn_score
                        ? matmul_qk_dst_desc
                        : post_qk_causal_mask_desc)},
        {attn_scores_desc},
        "fscore_where");
    g.add_op(select);
  }

  logical_tensor fscore_add_desc(logical_tensor_id++, dtype);
  if (attn_mask.has_value()) {
    op fscore_add(
        op_idx++,
        op::kind::Add,
        {mask_applied_after_scale
             ? attn_scores_desc
             : (is_scale_present ? post_scale_desc
                                 : (mask_before_scale ? post_qk_causal_mask_desc
                                                      : matmul_qk_dst_desc)),
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
           : (mask_applied_after_scale
                  ? attn_scores_desc
                  : (is_scale_present
                         ? post_scale_desc
                         : (mask_before_scale ? post_qk_causal_mask_desc
                                              : matmul_qk_dst_desc)))},
      {softmax_out_dst_desc},
      "softmax_out");
  softmax_out.set_attr<dim>(op::attr::axis, -1);
  g.add_op(softmax_out);

  logical_tensor matmul_v_dst_desc(logical_tensor_id++, dtype);
  op matmul_v(
      op_idx++,
      op::kind::MatMul,
      {softmax_out_dst_desc,
       value_requires_transpose ? transposed_value_desc : value_src_desc},
      {output_requires_transpose_and_reorder ? matmul_v_dst_desc
                                             : output_dst_desc},
      "matmul_value");
  g.add_op(matmul_v);

  logical_tensor matmul_v_trans_dst_desc(logical_tensor_id++, dtype);
  if (output_requires_transpose_and_reorder) {
    op matmul_v_transpose(
        op_idx++,
        op::kind::StaticTranspose,
        {matmul_v_dst_desc},
        {matmul_v_trans_dst_desc},
        "matmul_value_transpose");
    matmul_v_transpose.set_attr<dims>(
        dnnl::graph::op::attr::order, query_transpose_dims);
    g.add_op(matmul_v_transpose);
    op reorder_output(
        op_idx++,
        op::kind::Reorder,
        {matmul_v_trans_dst_desc},
        {output_dst_desc},
        "reorder_output");
    g.add_op(reorder_output);
  }

  g.finalize();
  auto partitions = g.get_partitions();
  auto partition = partitions[0];
  TORCH_CHECK(
      (partitions.size() == 1) && partition.is_supported(),
      "oneDNN Graph doesn't support this fusion pattern. If you'd like its support, please submit a ticket.");

  insert_in_partition_cache(patternID, partition);
}

void update_output_shapes(
    Tensor& output_tensor,
    cp_entry& cp,
    c10::ScalarType dtype) {
  output_tensor = at::detail::empty_strided_meta(
      cp.outputTensorShapes_[0], cp.outputTensorStrides_[0], dtype);
}

void compile_and_cache_sdpa_fusion(
    std::vector<Tensor>& input_tensors,
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

  bool is_output_transposed_and_reordered = patternID.test(11);
  bool query_requires_transpose = patternID.test(5);
  std::vector<int64_t> output_sizes(4);
  std::vector<int64_t> output_strides(4);
  if (((is_output_transposed_and_reordered) && !(query_requires_transpose)) ||
      ((!is_output_transposed_and_reordered) && (query_requires_transpose))) {
    output_sizes = input_tensors[0].sizes().vec();
    std::swap(output_sizes[1], output_sizes[2]);
    output_strides = {
        output_sizes[1] * output_sizes[2] * output_sizes[3],
        output_sizes[2] * output_sizes[3],
        output_sizes[3],
        1};
  } else {
    output_sizes = input_tensors[0].sizes().vec();
    output_strides = input_tensors[0].strides().vec();
  }
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
      input_tensors[0].data_ptr()));

  cp.cp_ = compile_partition(
      cp.partition_, cp.inputLogicalTensors_, cp.outputLogicalTensors_);
}

void _handle_sdpa_fusions(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& scale,
    const c10::optional<Tensor>& inverse_scale,
    const c10::optional<Tensor>& attn_mask,
    const c10::optional<Tensor>& causal_mask,
    const c10::optional<Tensor>& causal_mask_value,
    at::Tensor& output_tensor,
    bool query_requires_transpose = false,
    bool key_requires_transpose_twice = false,
    bool key_requires_transpose_once = false,
    bool value_requires_transpose = false,
    bool apply_mask_before_scale = false,
    bool choose_causal_mask_over_attn_score = false,
    bool output_requires_transpose_and_reorder = false,
    bool execute = true) {
  // patternID is determined on the basis of the arguments provided
  std::bitset<32> patternID;
  if (query.scalar_type() == c10::ScalarType::Float) {
    // bit 3 corresponds to float dtype
    patternID.set(3, 1);
  } else {
    // bit 2 corresponds to bfloat16 dtype
    // the dtype can be either float or bfloat16.
    // This logic is checked in torch/_inductor/fx_passes/fuse_attention.py
    patternID.set(2, 1);
  }
  // MHA pattern
  patternID.set(4, 1);
  // Refer to comments in Graph.cpp. The first 8 bits are reserved
  int pos = 8;
  patternID.set(pos++, scale.has_value());
  patternID.set(pos++, inverse_scale.has_value());
  patternID.set(pos++, attn_mask.has_value());
  patternID.set(pos++, causal_mask.has_value());
  patternID.set(pos++, causal_mask_value.has_value());
  patternID.set(pos++, query_requires_transpose);
  patternID.set(pos++, key_requires_transpose_twice);
  patternID.set(pos++, key_requires_transpose_once);
  patternID.set(pos++, value_requires_transpose);
  patternID.set(pos++, apply_mask_before_scale);
  patternID.set(pos++, choose_causal_mask_over_attn_score);
  patternID.set(pos++, output_requires_transpose_and_reorder);
  // first check cache
  // The key has a pattern ID, as well as the shapes of input tenors
  std::vector<int64_t> map_key;
  map_key.reserve(1024);
  // We use this because different thread-pools may be used
  map_key.push_back(omp_get_max_threads());
  // Algo ID
  std::vector<Tensor> input_tensors;
  input_tensors.push_back(query);
  input_tensors.push_back(key);
  input_tensors.push_back(value);

  map_key.push_back(static_cast<int64_t>(patternID.to_ullong()));

  map_key.insert(map_key.end(), key.sizes().begin(), key.sizes().end());
  map_key.insert(map_key.end(), query.sizes().begin(), query.sizes().end());
  map_key.insert(map_key.end(), value.sizes().begin(), value.sizes().end());
  if (scale != c10::nullopt) {
    auto scale_val = scale.value();
    input_tensors.push_back(scale_val);
    map_key.insert(
        map_key.end(), scale_val.sizes().begin(), scale_val.sizes().end());
  } else if (inverse_scale != c10::nullopt) {
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
      create_partition(
          patternID,
          query,
          key,
          value,
          scale,
          inverse_scale,
          attn_mask,
          causal_mask,
          causal_mask_value,
          query_requires_transpose,
          key_requires_transpose_twice,
          key_requires_transpose_once,
          value_requires_transpose,
          apply_mask_before_scale,
          choose_causal_mask_over_attn_score,
          output_requires_transpose_and_reorder);
      graph_partition_iter = partition_map_lookup(patternID);
    }
    graph_partition = graph_partition_iter->second;
    compiledPartitionEntry.partition_ = graph_partition;
    compile_and_cache_sdpa_fusion(
        input_tensors, compiledPartitionEntry, patternID);
    if (C10_LIKELY(execute)) {
      execute_partition(input_tensors, output_tensor, compiledPartitionEntry);
    } else {
      update_output_shapes(
          output_tensor, compiledPartitionEntry, query.scalar_type());
    }
    insert_in_fused_kernel_cache(map_key, compiledPartitionEntry);
  } else {
    change_pos_in_list(iter->second);
    cp_entry& cp = iter->second->second;
    if (C10_LIKELY(execute)) {
      execute_partition(input_tensors, output_tensor, cp);
    } else {
      update_output_shapes(output_tensor, cp, query.scalar_type());
    }
  }
}

Tensor mkldnn_graph_sdpa_fusion(
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
    bool apply_mask_before_scale = false,
    bool choose_causal_mask_over_attn_score = false,
    bool output_requires_transpose_and_reorder = false) {
  at::Tensor output;
  _handle_sdpa_fusions(
      query,
      key,
      value,
      scale,
      inverse_scale,
      attn_mask,
      causal_mask,
      causal_mask_value,
      output,
      query_requires_transpose,
      key_requires_transpose_twice,
      key_requires_transpose_once,
      value_requires_transpose,
      apply_mask_before_scale,
      choose_causal_mask_over_attn_score,
      output_requires_transpose_and_reorder);
  return output;
}

} // end anonymous namespace
} // namespace native

namespace meta {
namespace {

using namespace at::native;
using namespace at::native::onednn_graph;

// Compile fused kernels in Inductor compilation stage
Tensor mkldnn_graph_sdpa_fusion_meta(
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
    bool apply_mask_before_scale = false,
    bool choose_causal_mask_over_attn_score = false,
    bool output_requires_transpose_and_reorder = false) {
  // check if shapes are static
  if (query.unsafeGetTensorImpl()->has_symbolic_sizes_strides() ||
      key.unsafeGetTensorImpl()->has_symbolic_sizes_strides() ||
      value.unsafeGetTensorImpl()->has_symbolic_sizes_strides()) {
    TORCH_CHECK(false, "Dynamic shapes are currently not supported");
  }

  at::Tensor output;
  _handle_sdpa_fusions(
      query,
      key,
      value,
      scale,
      inverse_scale,
      attn_mask,
      causal_mask,
      causal_mask_value,
      output,
      query_requires_transpose,
      key_requires_transpose_twice,
      key_requires_transpose_once,
      value_requires_transpose,
      apply_mask_before_scale,
      choose_causal_mask_over_attn_score,
      output_requires_transpose_and_reorder,
      /* execute = */ false);
  return output;
}

} // end anonymous namespace

TORCH_LIBRARY_IMPL(mkldnn, Meta, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_graph_sdpa_fusion"),
      c10::DispatchKey::Meta,
      TORCH_FN(mkldnn_graph_sdpa_fusion_meta));
}
} // namespace meta

namespace native {
TORCH_LIBRARY_IMPL(mkldnn, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_graph_sdpa_fusion"),
      TORCH_FN(mkldnn_graph_sdpa_fusion));
}
} // namespace native
} // namespace at
#endif // AT_ONEDNN_GRAPH_ENABLED
