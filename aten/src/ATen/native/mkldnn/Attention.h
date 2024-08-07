#pragma once
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
namespace onednn_graph {

void create_partition(
    std::bitset<32>& patternID,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& scale,
    const c10::optional<Tensor>& attn_mask);

void compile_and_cache_sdpa_fusion(
    std::vector<Tensor>& input_tensors,
    Tensor& output_tensor,
    cp_entry& cp,
    std::bitset<32>& patternID);

} // end namespace onednn_graph
} // end namespace native
} // end namespace at

#endif // AT_ONEDNN_GRAPH_ENABLED()
