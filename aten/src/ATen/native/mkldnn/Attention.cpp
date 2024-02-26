#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/utils/ParamUtils.h>
#include <omp.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_to_dense_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#endif

#if !AT_ONEDNN_GRAPH_ENABLED()

namespace at {
namespace native {

namespace {

Tensor mkldnn_graph_sdpa_pattern(
    const int64_t uniqueID,
    const Tensor& key,
    const Tensor& query,
    const Tensor& value,
    const c10::optional<Tensor>& scale,
    const c10::optional<Tensor>& attn_mask) {
  TORCH_CHECK(
      false,
      "mkldnn_graph_sdpa_pattern: ATen not compiled with oneDNN Graph support");
}

} // end anonymous namespace

#else

// custom hash function when key is a vector of int64 values
// c10::get_hash() does not provide unique outputs, so can't reuse it.
namespace std {
template <>
struct hash<std::vector<int64_t>> {
  size_t operator()(const std::vector<int64_t>& key) const {
    size_t total = key.size();
    size_t sum = 0;
    if (total < 64) {
      for (size_t i = 0; i < total; i++) {
        sum += key[i] << i;
      }
    } else {
      size_t batch = total / 64;
      size_t remaining = total % 64;
      for (size_t bs = 0; bs < batch; bs++) {
        for (size_t i = 0; i < 64; i++) {
          sum += key[bs * 64 + i] << i;
        }
      }
      for (size_t i = 0; i < remaining; i++) {
        sum += key[batch * 64 + i] << i;
      }
    }
    return sum;
  }
};

} // namespace std

#include <ATen/native/mkldnn/Utils.h>
#include <c10/util/irange.h>
#include <oneapi/dnnl/dnnl_graph.hpp>

namespace at {
namespace native {

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;

namespace {

// assign a unique ID to each pattern here
// using an enum is troublesome with using std::unordered_map
// These should be used in torch/_inductor/fx_passes/fuse_attention.py
// Instead of using magic numbers in the python code there,
// we could even use custom classes.
#define ONEDNN_GRAPH_SDPA_PATTERN_5_FP32 0
#define ONEDNN_GRAPH_SDPA_PATTERN_5_BF16 1

using RunArg = dnnl::graph::tensor;
using RunArgs = std::vector<RunArg>;
using LogicalTensors = std::vector<logical_tensor>;

// Compiled Partition entry
struct cp_entry {
  partition partition_;
  compiled_partition cp_;
  RunArgs constantInputTensors_;
  RunArgs inputLLGATensors_;
  RunArgs outputLLGATensors_;
  LogicalTensors inputLogicalTensors_;
  LogicalTensors outputLogicalTensors_;
};

// Thread local data-structures are required if multiple thread-pools
// of a PyTorch process would be used for inference.
thread_local std::unordered_map<int32_t, dnnl::graph::partition> partition_map_;
// Compiled partition (fused kernel) cache
// Adopted from
// https://github.com/lamerman/cpp-lru-cache/blob/master/include/lrucache.hpp
using key_value_pair_t = std::pair<std::vector<int64_t>, cp_entry>;
using list_iterator_t = std::list<key_value_pair_t>::iterator;
thread_local std::list<key_value_pair_t> cache_items_list_;
thread_local std::unordered_map<std::vector<int64_t>, list_iterator_t>
    cache_items_map_;
thread_local size_t capacity_ = 75000;

// Compile a partition
compiled_partition compile_partition(
    const partition& partition,
    const std::vector<logical_tensor>& inputs,
    const std::vector<logical_tensor>& outputs) {
  compiled_partition compilation;
  compilation =
      partition.compile(inputs, outputs, onednn_graph::Engine::getEngine());
  return compilation;
}

/*
   (f32/bf16)[Query]     [Key](f32/bf16)
              \     /
               MatMul
                 |
            Divide  [attn mask]
                 |  /
                 | /
                Add
                 |
              Softmax    [Value](f32/bf16)
                    \     /
                     MatMul
                       |
                    [output](f32/bf16)
*/
void create_graph_sdpa_pattern_5(data_type dtype) {
  dnnl::graph::graph g{dnnl::graph::engine::kind::cpu};
  size_t op_idx = 0;
  size_t logical_tensor_id = 0;

  // input tensors
  logical_tensor q_trans_src_desc(logical_tensor_id++, dtype);
  logical_tensor k_trans_src_desc(logical_tensor_id++, dtype);
  logical_tensor v_trans_src_desc(logical_tensor_id++, dtype);
  logical_tensor fscore_scale_desc(logical_tensor_id++, dtype);
  logical_tensor attension_mask_desc(logical_tensor_id++, dtype);

  // output tensors
  logical_tensor matmul_v_dst_desc(logical_tensor_id++, dtype);

  // intermediate outputs & corresponding ops
  logical_tensor matmul_qk_dst_desc(logical_tensor_id++, dtype);
  op matmul_qk(
      op_idx++,
      op::kind::MatMul,
      {q_trans_src_desc, k_trans_src_desc},
      {matmul_qk_dst_desc},
      "matmul_qk");

  logical_tensor fscore_div_dst_desc(logical_tensor_id++, dtype);
  op fscore_div(
      op_idx++,
      op::kind::Divide,
      {matmul_qk_dst_desc, fscore_scale_desc},
      {fscore_div_dst_desc},
      "fscore_div");

  logical_tensor fscore_add_dst_desc(logical_tensor_id++, dtype);
  op fscore_add(
      op_idx++,
      op::kind::Add,
      {fscore_div_dst_desc, attension_mask_desc},
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

  op matmul_v(
      op_idx++,
      op::kind::MatMul,
      {softmax_out_dst_desc, v_trans_src_desc},
      {matmul_v_dst_desc},
      "matmul_value");

  g.add_op(matmul_qk);
  g.add_op(fscore_div);
  g.add_op(fscore_add);
  g.add_op(softmax_out);
  g.add_op(matmul_v);

  g.finalize();
  auto partitions = g.get_partitions();
  auto partition = partitions[0];
  TORCH_CHECK(
      (partitions.size() == 1) && partition.is_supported(),
      " only one fusion group allowed");
  int patternID = dtype == data_type::bf16 ? ONEDNN_GRAPH_SDPA_PATTERN_5_BF16
                                           : ONEDNN_GRAPH_SDPA_PATTERN_5_FP32;
  partition_map_[patternID] = std::move(partition);
}

// Execute SDPA partition
// TODO: creating a vector of input tensors wouldn't create extra copies of
// tensors. So, we can make this function generic. Also, we need to be able to
// handle multiple outputs anyway
at::Tensor execute_sdpa_partition(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& scale,
    const c10::optional<Tensor>& attn_mask,
    cp_entry& cp,
    bool inplace = false) {
  int i = 0;

  cp.inputLLGATensors_[i++].set_data_handle(query.data_ptr());
  cp.inputLLGATensors_[i++].set_data_handle(key.data_ptr());
  cp.inputLLGATensors_[i++].set_data_handle(value.data_ptr());
  if (scale.has_value()) {
    cp.inputLLGATensors_[i++].set_data_handle(scale.value().data_ptr());
  }
  if (attn_mask.has_value()) {
    cp.inputLLGATensors_[i++].set_data_handle(attn_mask.value().data_ptr());
  }
  at::Tensor output_tensor;
  if (inplace) {
    // there's no copy, so it's fine
    output_tensor = query;
  } else {
    output_tensor = at::empty_like(query);
  }
  cp.outputLLGATensors_[0].set_data_handle(output_tensor.data_ptr());
  cp.cp_.execute(
      onednn_graph::Stream::getStream(),
      cp.inputLLGATensors_,
      cp.outputLLGATensors_);
  return output_tensor;
}

// Compile SDPA pattern 5 corresponding to
// torch/_inductor/fx_passes/fuse_attention.py
// TODO: we could use this function generic by adding shape information in
// cp_entry before this function is called. Also, handle synchronization in case
// multiple thread-pools of the same PyTorch process are used for inference.
void compile_and_cache_sdpa_pattern_5(
    const partition& partition,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& scale,
    const Tensor& attn_mask,
    cp_entry& cp) {
  // assuming all inputs have the same dtype. Might revisit this assumption
  // later
  data_type dtype = query.scalar_type() == at::ScalarType::Float
      ? data_type::f32
      : data_type::bf16;
  int i = 0;
  cp.inputLogicalTensors_.emplace_back(
      logical_tensor(i, dtype, query.sizes().vec(), query.strides().vec()));
  cp.inputLLGATensors_.emplace_back(RunArg(
      cp.inputLogicalTensors_[i++],
      onednn_graph::Engine::getEngine(),
      query.data_ptr()));

  cp.inputLogicalTensors_.emplace_back(
      logical_tensor(i, dtype, key.sizes().vec(), key.strides().vec()));
  cp.inputLLGATensors_.emplace_back(RunArg(
      cp.inputLogicalTensors_[i++],
      onednn_graph::Engine::getEngine(),
      key.data_ptr()));

  cp.inputLogicalTensors_.emplace_back(
      logical_tensor(i, dtype, value.sizes().vec(), value.strides().vec()));
  cp.inputLLGATensors_.emplace_back(RunArg(
      cp.inputLogicalTensors_[i++],
      onednn_graph::Engine::getEngine(),
      value.data_ptr()));

  cp.inputLogicalTensors_.emplace_back(
      logical_tensor(i, dtype, scale.sizes().vec(), scale.strides().vec()));
  cp.inputLLGATensors_.emplace_back(RunArg(
      cp.inputLogicalTensors_[i++],
      onednn_graph::Engine::getEngine(),
      scale.data_ptr()));

  cp.inputLogicalTensors_.emplace_back(logical_tensor(
      i, dtype, attn_mask.sizes().vec(), attn_mask.strides().vec()));
  cp.inputLLGATensors_.emplace_back(RunArg(
      cp.inputLogicalTensors_[i++],
      onednn_graph::Engine::getEngine(),
      attn_mask.data_ptr()));

  // output is of the same size as query or value
  // In this pattern, key is of a different shape
  cp.outputLogicalTensors_.emplace_back(
      logical_tensor(i, dtype, query.sizes().vec(), query.strides().vec()));
  cp.outputLLGATensors_.emplace_back(RunArg(
      cp.outputLogicalTensors_[0],
      onednn_graph::Engine::getEngine(),
      query.data_ptr()));

  cp.partition_ = partition;
  cp.cp_ = compile_partition(
      partition, cp.inputLogicalTensors_, cp.outputLogicalTensors_);
}

Tensor mkldnn_graph_sdpa_pattern(
    const int64_t patternID,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& scale,
    const c10::optional<Tensor>& attn_mask) {
  // first check cache
  // The key has a pattern ID, as well as the shapes of input tenors
  std::vector<int64_t> map_key;
  map_key.reserve(1024);
  // We use this because different thread-pools may be used
  map_key.push_back(omp_get_max_threads());
  // Algo ID
  map_key.push_back(patternID);

  map_key.insert(map_key.end(), key.sizes().begin(), key.strides().end());
  map_key.insert(map_key.end(), query.sizes().begin(), query.strides().end());
  map_key.insert(map_key.end(), value.sizes().begin(), value.strides().end());
  if (scale.has_value()) {
    auto scale_val = scale.value();
    map_key.insert(
        map_key.end(), scale_val.sizes().begin(), scale_val.strides().end());
  }
  if (attn_mask.has_value()) {
    auto attn_mask_val = attn_mask.value();
    map_key.insert(
        map_key.end(),
        attn_mask_val.sizes().begin(),
        attn_mask_val.strides().end());
  }

  auto iter = cache_items_map_.find(map_key);
  if (iter == cache_items_map_.end()) {
    cp_entry compiledPartitionEntry;
    auto graph_partition_iter = partition_map_.find(patternID);
    partition graph_partition;
    if (graph_partition_iter == partition_map_.end()) {
      auto dtype = query.scalar_type();
      TORCH_CHECK(((dtype == at::ScalarType::Float) || (dtype == at::ScalarType::BFloat16)),
          "Only BF16 & FP32 datatypes are currently supported");
      switch (patternID) {
        case ONEDNN_GRAPH_SDPA_PATTERN_5_FP32:
          create_graph_sdpa_pattern_5(data_type::f32);
          break;
        case ONEDNN_GRAPH_SDPA_PATTERN_5_BF16:
          create_graph_sdpa_pattern_5(data_type::bf16);
          break;
      }
      graph_partition_iter = partition_map_.find(patternID);
    }
    graph_partition = graph_partition_iter->second;
    switch (patternID) {
      case ONEDNN_GRAPH_SDPA_PATTERN_5_FP32:
      case ONEDNN_GRAPH_SDPA_PATTERN_5_BF16:
        compile_and_cache_sdpa_pattern_5(
            graph_partition,
            query,
            key,
            value,
            scale.value(),
            attn_mask.value(),
            compiledPartitionEntry);
    }
    auto retVal = execute_sdpa_partition(
        query, key, value, scale, attn_mask, compiledPartitionEntry);
    cache_items_list_.push_front(
        key_value_pair_t(map_key, std::move(compiledPartitionEntry)));
    cache_items_map_[map_key] = cache_items_list_.begin();
    if (cache_items_map_.size() > capacity_) {
      auto last = cache_items_list_.end();
      last--;
      cache_items_map_.erase(last->first);
      cache_items_list_.pop_back();
    }
    return retVal;
  } else {
    cache_items_list_.splice(
        cache_items_list_.begin(), cache_items_list_, iter->second);
    cp_entry& cp = iter->second->second;
    return execute_sdpa_partition(query, key, value, scale, attn_mask, cp);
  }
}

} // end anonymous namespace

#endif // !AT_ONEDNN_GRAPH_ENABLED()

TORCH_LIBRARY_IMPL(mkldnn, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_graph_sdpa_pattern"),
      TORCH_FN(mkldnn_graph_sdpa_pattern));
}

} // namespace native
} // namespace at
