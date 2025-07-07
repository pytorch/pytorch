#pragma once

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace c10d {

C10_EXPORT at::Tensor& all_reduce_(
    at::Tensor& input,
    std::string reduce_op,
    std::string group_name);

C10_EXPORT at::Tensor all_reduce(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name);

C10_EXPORT std::vector<at::Tensor> all_reduce_coalesced_(
    std::vector<at::Tensor> inputs,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string reduce_op,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string group_name);

C10_EXPORT std::vector<at::Tensor> all_reduce_coalesced(
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::vector<at::Tensor> inputs,
    std::string reduce_op,
    std::string group_name);

C10_EXPORT std::vector<at::Tensor> all_gather_into_tensor_coalesced(
    std::vector<at::Tensor> inputs,
    int64_t group_size,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string group_name);

C10_EXPORT at::Tensor all_gather_into_tensor(
    const at::Tensor& input,
    int64_t group_size,
    std::string group_name);

C10_EXPORT at::Tensor& all_gather_into_tensor_out(
    at::Tensor& input,
    int64_t group_size,
    const std::string& group_name,
    at::Tensor& output);

C10_EXPORT std::vector<at::Tensor> reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor> inputs,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string reduce_op,
    int64_t group_size,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string group_name);

C10_EXPORT at::Tensor reduce_scatter_tensor(
    const at::Tensor& input,
    std::string reduce_op,
    int64_t group_size,
    std::string group_name);

C10_EXPORT at::Tensor all_to_all_single(
    const at::Tensor& input,
    std::vector<int64_t> output_split_sizes,
    std::vector<int64_t> input_split_sizes,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string group_name);

C10_EXPORT at::Tensor& broadcast_(
    at::Tensor& input,
    int64_t src,
    std::string group_name);

C10_EXPORT at::Tensor broadcast(
    const at::Tensor& input,
    int64_t src,
    std::string group_name);

} // namespace c10d
