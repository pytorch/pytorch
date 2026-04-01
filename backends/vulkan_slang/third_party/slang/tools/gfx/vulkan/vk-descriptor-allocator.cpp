#include "vk-descriptor-allocator.h"

#include "vk-util.h"

namespace gfx
{
VkDescriptorPool DescriptorSetAllocator::newPool()
{
    VkDescriptorPoolCreateInfo descriptorPoolInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    Slang::Array<VkDescriptorPoolSize, 32> poolSizes;
    poolSizes.add(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, 1024});
    poolSizes.add(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1024});
    poolSizes.add(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4096});
    poolSizes.add(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024});
    poolSizes.add(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 256});
    poolSizes.add(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 256});
    poolSizes.add(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4096});
    poolSizes.add(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096});
    poolSizes.add(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 4096});
    poolSizes.add(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 4096});
    poolSizes.add(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 16});
    if (m_api->m_extendedFeatures.inlineUniformBlockFeatures.inlineUniformBlock)
    {
        poolSizes.add(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT, 16});
    }
    if (m_api->m_extendedFeatures.accelerationStructureFeatures.accelerationStructure)
    {
        poolSizes.add(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 256});
    }
    descriptorPoolInfo.maxSets = 4096;
    descriptorPoolInfo.poolSizeCount = (uint32_t)poolSizes.getCount();
    descriptorPoolInfo.pPoolSizes = poolSizes.getBuffer();
    descriptorPoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    VkDescriptorPoolInlineUniformBlockCreateInfo inlineUniformBlockInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_INLINE_UNIFORM_BLOCK_CREATE_INFO};
    inlineUniformBlockInfo.maxInlineUniformBlockBindings = 16;
    descriptorPoolInfo.pNext = &inlineUniformBlockInfo;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    SLANG_VK_CHECK(m_api->vkCreateDescriptorPool(
        m_api->m_device,
        &descriptorPoolInfo,
        nullptr,
        &descriptorPool));
    pools.add(descriptorPool);
    return descriptorPool;
}

VulkanDescriptorSet DescriptorSetAllocator::allocate(VkDescriptorSetLayout layout)
{
    VulkanDescriptorSet rs = {};
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = getPool();
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &layout;
    if (m_api->vkAllocateDescriptorSets(m_api->m_device, &allocInfo, &rs.handle) == VK_SUCCESS)
    {
        rs.pool = allocInfo.descriptorPool;
        return rs;
    }
    // If allocation from last pool fails, try all existing pools.
    for (Slang::Index i = 0; i < pools.getCount() - 1; i++)
    {
        allocInfo.descriptorPool = pools[i];
        if (m_api->vkAllocateDescriptorSets(m_api->m_device, &allocInfo, &rs.handle) == VK_SUCCESS)
        {
            rs.pool = allocInfo.descriptorPool;
            return rs;
        }
    }
    // If we still cannot allocate the descriptor set, add a new pool.
    auto pool = newPool();
    allocInfo.descriptorPool = pool;
    if (m_api->vkAllocateDescriptorSets(m_api->m_device, &allocInfo, &rs.handle) == VK_SUCCESS)
    {
        rs.pool = allocInfo.descriptorPool;
        return rs;
    }
    // Failed to allocate from a new pool, we are in trouble.
    assert(!"descriptor set allocation failed.");
    return rs;
}
} // namespace gfx
