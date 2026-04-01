#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vulkan {

class Pipeline {
public:
    Pipeline(VkDevice device,
             const uint32_t* spirv_code,
             size_t spirv_size,
             uint32_t num_buffers,
             uint32_t push_constant_size = 0);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    VkPipeline pipeline() const { return pipeline_; }
    VkPipelineLayout layout() const { return layout_; }
    VkDescriptorSetLayout descriptor_set_layout() const { return desc_set_layout_; }

private:
    VkDevice device_;
    VkShaderModule shader_module_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

class PipelineCache {
public:
    static PipelineCache& instance();

    Pipeline* get_or_create(
        VkDevice device,
        const std::string& key,
        const uint32_t* spirv_code,
        size_t spirv_size,
        uint32_t num_buffers,
        uint32_t push_constant_size = 0);

    void clear();

private:
    PipelineCache() = default;
    std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<Pipeline>> cache_;
};

} // namespace vulkan
