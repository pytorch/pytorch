#include <ATen/native/mkldnn/Graph.h>

#if AT_ONEDNN_GRAPH_ENABLED()

namespace at {
namespace native {
namespace onednn_graph {

// Thread local data-structures are required if multiple thread-pools
// of a PyTorch process would be used for inference.
thread_local std::unordered_map<std::bitset<32>, dnnl::graph::partition>
    partition_map_;
// Compiled partition (fused kernel) cache
// Adopted from
// https://github.com/lamerman/cpp-lru-cache/blob/master/include/lrucache.hpp

thread_local std::list<key_value_pair_t> cache_items_list_;
thread_local std::unordered_map<std::vector<int64_t>, list_iterator_t>
    fused_kernel_cache_map_;
// cache capacity is arbitrary
// TODO: Add an API to manipulate cache capacity
thread_local size_t capacity_ = 1024;

void insert_in_fused_kernel_cache(std::vector<int64_t>& map_key, cp_entry& cp) {
  cache_items_list_.push_front(key_value_pair_t(map_key, std::move(cp)));
  fused_kernel_cache_map_[map_key] = cache_items_list_.begin();
  if (fused_kernel_cache_map_.size() > capacity_) {
    auto last = cache_items_list_.end();
    last--;
    fused_kernel_cache_map_.erase(last->first);
    cache_items_list_.pop_back();
  }
}

void change_pos_in_list(list_iterator_t& kvpair) {
  cache_items_list_.splice(
      cache_items_list_.begin(), cache_items_list_, kvpair);
}

std::unordered_map<std::vector<int64_t>, list_iterator_t>::iterator cache_lookup(
    std::vector<int64_t>& map_key) {
  return fused_kernel_cache_map_.find(map_key);
}

std::unordered_map<std::vector<int64_t>, list_iterator_t>::iterator cache_end() {
  return fused_kernel_cache_map_.end();
}

std::unordered_map<std::bitset<32>, dnnl::graph::partition>::iterator
partition_map_lookup(std::bitset<32>& patternID) {
  return partition_map_.find(patternID);
}

std::unordered_map<std::bitset<32>, dnnl::graph::partition>::iterator
partition_map_end() {
  return partition_map_.end();
}

// The first 8 bits are reserved
// bit 0: is int8_fp32
// bit 1: is uint8_fp32
// bit 2: is int8_bf16
// bit 3: is uint8_bf16
// bit 4: is bf16
// bit 5: is fp32
// bit 6: is MHA pattern
// The rest of the bits depend upon the arguments provided
// However, down the line, we might have different bitsets for different
// patterns
void insert_in_partition_cache(std::bitset<32>& patternID, partition& p) {
  partition_map_[patternID] = std::move(p);
}

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

data_type aten_to_onednn_graph_dtype(at::ScalarType dt) {
  switch (dt) {
    case at::ScalarType::Float:
      return data_type::f32;
    case at::ScalarType::BFloat16:
      return data_type::bf16;
    case at::ScalarType::Bool:
      return data_type::boolean;
    case at::kInt:
      return data_type::s32;
    case at::ScalarType::QInt8:
      return data_type::s8;
    case at::ScalarType::QUInt8:
      return data_type::u8;
    default:
      return data_type::undef;
  }
}

// Execute fused kernel
// Can be extended to create multiple outputs
void execute_partition(
    std::vector<Tensor>& input_tensors,
    at::Tensor& output_tensor,
    cp_entry& cp,
    bool inplace) {
  int i = 0;

  for (auto& each_tensor : input_tensors) {
    cp.inputLLGATensors_[i++].set_data_handle(each_tensor.data_ptr());
  }

  cp.outputLLGATensors_[0].set_data_handle(output_tensor.data_ptr());
  cp.cp_.execute(
      onednn_graph::Stream::getStream(),
      cp.inputLLGATensors_,
      cp.outputLLGATensors_);
}

} // end namespace onednn_graph
} // end namespace native
} // end namespace at

#endif // AT_ONEDNN_GRAPH_ENABLED()
