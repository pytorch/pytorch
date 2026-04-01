#include "Pipeline.h"
#include <stdexcept>

namespace vulkan {

Pipeline::Pipeline(VkDevice device,
                   const uint32_t* spirv_code,
                   size_t spirv_size,
                   uint32_t num_buffers,
                   uint32_t push_constant_size)
    : device_(device) {

    // Create shader module
    VkShaderModuleCreateInfo sm_ci{};
    sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm_ci.codeSize = spirv_size;
    sm_ci.pCode = spirv_code;

    VkResult result = vkCreateShaderModule(device_, &sm_ci, nullptr, &shader_module_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }

    // Create descriptor set layout
    std::vector<VkDescriptorSetLayoutBinding> bindings(num_buffers);
    for (uint32_t i = 0; i < num_buffers; i++) {
        bindings[i] = {};
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dsl_ci{};
    dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_ci.bindingCount = num_buffers;
    dsl_ci.pBindings = bindings.data();

    result = vkCreateDescriptorSetLayout(device_, &dsl_ci, nullptr, &desc_set_layout_);
    if (result != VK_SUCCESS) {
        vkDestroyShaderModule(device_, shader_module_, nullptr);
        throw std::runtime_error("Failed to create descriptor set layout");
    }

    // Create pipeline layout
    VkPipelineLayoutCreateInfo pl_ci{};
    pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount = 1;
    pl_ci.pSetLayouts = &desc_set_layout_;

    VkPushConstantRange pc_range{};
    if (push_constant_size > 0) {
        pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc_range.offset = 0;
        pc_range.size = push_constant_size;
        pl_ci.pushConstantRangeCount = 1;
        pl_ci.pPushConstantRanges = &pc_range;
    }

    result = vkCreatePipelineLayout(device_, &pl_ci, nullptr, &layout_);
    if (result != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device_, desc_set_layout_, nullptr);
        vkDestroyShaderModule(device_, shader_module_, nullptr);
        throw std::runtime_error("Failed to create pipeline layout");
    }

    // Create compute pipeline
    VkComputePipelineCreateInfo cp_ci{};
    cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp_ci.stage.module = shader_module_;
    cp_ci.stage.pName = "main";
    cp_ci.layout = layout_;

    result = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &pipeline_);
    if (result != VK_SUCCESS) {
        vkDestroyPipelineLayout(device_, layout_, nullptr);
        vkDestroyDescriptorSetLayout(device_, desc_set_layout_, nullptr);
        vkDestroyShaderModule(device_, shader_module_, nullptr);
        throw std::runtime_error("Failed to create compute pipeline");
    }
}

Pipeline::~Pipeline() {
    if (pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, pipeline_, nullptr);
    if (layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, layout_, nullptr);
    if (desc_set_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, desc_set_layout_, nullptr);
    if (shader_module_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, shader_module_, nullptr);
}

// ── PipelineCache ────────────────────────────────────────────────
PipelineCache& PipelineCache::instance() {
    static PipelineCache cache;
    return cache;
}

Pipeline* PipelineCache::get_or_create(
    VkDevice device,
    const std::string& key,
    const uint32_t* spirv_code,
    size_t spirv_size,
    uint32_t num_buffers,
    uint32_t push_constant_size) {

    // Fast path: check without lock (safe because cache_ is never modified
    // after initial population, and pointer reads are atomic on x86/ARM)
    {
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second.get();
        }
    }

    // Slow path: acquire lock and create pipeline
    std::lock_guard<std::mutex> lock(mutex_);

    // Double-check after acquiring lock
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return it->second.get();
    }

    auto pipeline = std::make_unique<Pipeline>(
        device, spirv_code, spirv_size, num_buffers, push_constant_size);
    auto* ptr = pipeline.get();
    cache_[key] = std::move(pipeline);
    return ptr;
}

void PipelineCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
}

} // namespace vulkan
