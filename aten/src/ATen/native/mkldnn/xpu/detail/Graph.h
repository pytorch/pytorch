#include <ATen/native/mkldnn/xpu/detail/Attr.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_graph_sycl.hpp>
#include <bitset>
#include <list>

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

namespace at::native::onednn::graph {

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;
using RunArg = dnnl::graph::tensor;
using RunArgs = std::vector<RunArg>;
using LogicalTensors = std::vector<logical_tensor>;

using engine = dnnl::engine;

// Compiled Partition entry
struct cp_entry {
  partition partition_;
  compiled_partition cp_;
  RunArgs inputLLGATensors_;
  RunArgs outputLLGATensors_;
  LogicalTensors inputLogicalTensors_;
  LogicalTensors outputLogicalTensors_;
};

using key_value_pair_t = std::pair<std::vector<int64_t>, cp_entry>;
using list_iterator_t = std::list<key_value_pair_t>::iterator;

// Thread local data-structures are required if multiple thread-pools
// of a PyTorch process would be used for inference.
thread_local std::unordered_map<std::vector<int64_t>, dnnl::graph::partition> partition_map_;
thread_local std::list<key_value_pair_t> cache_items_list_;
thread_local std::unordered_map<std::vector<int64_t>, list_iterator_t>
    fused_kernel_cache_map_;
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

// The first 8 bits are reserved
// bit 0: is int8
// bit 1: is uint8
// bit 2: is fp16
// bit 3: is fp32
// bit 4: is sdp pattern
// bit 5-7: N/A
// The rest of the bits depend upon the arguments provided
// However, down the line, we might have different bitsets for different
// patterns
void insert_in_partition_cache(std::vector<int64_t>& patternID, partition& p) {
  partition_map_[patternID] = std::move(p);
}

void change_pos_in_list(list_iterator_t& kvpair) {
  cache_items_list_.splice(
      cache_items_list_.begin(), cache_items_list_, kvpair);
}

std::unordered_map<std::vector<int64_t>, dnnl::graph::partition>::iterator
partition_map_lookup(const std::vector<int64_t>& partition_key) {
  return partition_map_.find(partition_key);
}
std::unordered_map<std::vector<int64_t>, dnnl::graph::partition>::iterator
partition_map_end() {
  return partition_map_.end();
}

std::unordered_map<std::vector<int64_t>, list_iterator_t>::iterator cache_lookup(
    std::vector<int64_t>& map_key) {
  return fused_kernel_cache_map_.find(map_key);
}

std::unordered_map<std::vector<int64_t>, list_iterator_t>::iterator cache_end() {
  return fused_kernel_cache_map_.end();
}
void compile_partition(cp_entry& cp, const engine& eng) {
  // RECORD_FUNCTION("compile_partition", c10::ArrayRef<c10::IValue>({}));
  cp.inputLogicalTensors_ = cp.partition_.get_input_ports();
  cp.outputLogicalTensors_ = cp.partition_.get_output_ports();
  cp.cp_ = cp.partition_.compile(
      cp.inputLogicalTensors_, cp.outputLogicalTensors_, eng);
}

} // namespace at::native::onednn::graph
