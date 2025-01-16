#pragma once

#include <torch/csrc/profiler/stubs/base.h>
#include <torch/csrc/profiler/util.h>
#include <cstdint>

namespace torch::profiler::impl::vulkan {

// Using function pointer i.e. [std::tuple<std::string, uint64_t> (*)(int64_t)]
// doesn't work because we need to capture the QueryPool in the lambda context
// https://stackoverflow.com/a/28746827
using GetShaderNameAndDurationNsFn =
    std::function<std::tuple<std::string, uint64_t>(int64_t)>;
TORCH_API void registerGetShaderNameAndDurationNs(
    GetShaderNameAndDurationNsFn get_shader_name_and_duration_ns);

TORCH_API void deregisterGetShaderNameAndDurationNs();

std::tuple<std::string, uint64_t> getShaderNameAndDurationNs(
    const vulkan_id_t& vulkan_id);

} // namespace torch::profiler::impl::vulkan
