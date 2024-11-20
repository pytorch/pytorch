#include <torch/csrc/profiler/orchestration/vulkan.h>

#include <utility>

namespace torch::profiler::impl::vulkan {
namespace {

GetShaderNameAndDurationNsFn get_shader_name_and_duration_ns_fn;

} // namespace

void registerGetShaderNameAndDurationNs(
    GetShaderNameAndDurationNsFn get_shader_name_and_duration_ns) {
  get_shader_name_and_duration_ns_fn =
      std::move(get_shader_name_and_duration_ns);
}

void deregisterGetShaderNameAndDurationNs() {
  get_shader_name_and_duration_ns_fn = nullptr;
}

std::tuple<std::string, uint64_t> getShaderNameAndDurationNs(
    const vulkan_id_t& vulkan_id) {
  /*
    We don't need to worry about a race condition with
    deregisterGetShaderNameAndDurationNs here currently because
    deregisterGetShaderNameAndDurationNs is only called within the destructor
    of QueryPool, which would only be called after we're done calling
    getShaderNameAndDurationNs
  */
  TORCH_CHECK(
      get_shader_name_and_duration_ns_fn != nullptr,
      "Attempting to get shader duration in ",
      "torch::profiler::impl::vulkan::getShaderNameAndDurationNs, but "
      "get_shader_duration_fn is unregistered. Use "
      "torch::profiler::impl::vulkan::registerGetShaderNameAndDurationNs to register "
      "it first");
  return get_shader_name_and_duration_ns_fn(vulkan_id.value_of());
}

} // namespace torch::profiler::impl::vulkan
