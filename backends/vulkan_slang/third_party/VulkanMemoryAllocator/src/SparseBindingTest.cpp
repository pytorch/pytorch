//
// Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#include "Common.h"
#include "SparseBindingTest.h"

#ifdef _WIN32

////////////////////////////////////////////////////////////////////////////////
// External imports

extern VkDevice g_hDevice;
extern VmaAllocator g_hAllocator;
extern uint32_t g_FrameIndex;
extern bool g_SparseBindingEnabled;
extern VkQueue g_hSparseBindingQueue;
extern VkFence g_ImmediateFence;
extern VkCommandBuffer g_hTemporaryCommandBuffer;

void BeginSingleTimeCommands();
void EndSingleTimeCommands();
void SaveAllocatorStatsToFile(const wchar_t* filePath, bool detailed = true);
void LoadShader(std::vector<char>& out, const char* fileName);

////////////////////////////////////////////////////////////////////////////////
// Class definitions

static uint32_t CalculateMipMapCount(uint32_t width, uint32_t height, uint32_t depth)
{
    uint32_t mipMapCount = 1;
    while(width > 1 || height > 1 || depth > 1)
    {
        ++mipMapCount;
        width  /= 2;
        height /= 2;
        depth  /= 2;
    }
    return mipMapCount;
}

class BaseImage
{
public:
    virtual void Init(RandomNumberGenerator& rand) = 0;
    virtual ~BaseImage();

    const VkImageCreateInfo& GetCreateInfo() const { return m_CreateInfo; }

    void TestContent(RandomNumberGenerator& rand);

protected:
    VkImageCreateInfo m_CreateInfo = {};
    VkImage m_Image = VK_NULL_HANDLE;

    void FillImageCreateInfo(RandomNumberGenerator& rand);
    void UploadContent();
    void ValidateContent(RandomNumberGenerator& rand);
};

class TraditionalImage : public BaseImage
{
public:
    virtual void Init(RandomNumberGenerator& rand);
    virtual ~TraditionalImage();

private:
    VmaAllocation m_Allocation = VK_NULL_HANDLE;
};

class SparseBindingImage : public BaseImage
{
public:
    virtual void Init(RandomNumberGenerator& rand);
    virtual ~SparseBindingImage();

private:
    std::vector<VmaAllocation> m_Allocations;
};

////////////////////////////////////////////////////////////////////////////////
// class BaseImage

BaseImage::~BaseImage()
{
    if(m_Image)
    {
        vkDestroyImage(g_hDevice, m_Image, nullptr);
    }
}

void BaseImage::TestContent(RandomNumberGenerator& rand)
{
    printf("Validating content of %u x %u texture...\n",
        m_CreateInfo.extent.width, m_CreateInfo.extent.height);
    UploadContent();
    ValidateContent(rand);
}

void BaseImage::FillImageCreateInfo(RandomNumberGenerator& rand)
{
    constexpr uint32_t imageSizeMin = 8;
    constexpr uint32_t imageSizeMax = 2048;

    const bool useMipMaps = rand.Generate() % 2 != 0;

    ZeroMemory(&m_CreateInfo, sizeof(m_CreateInfo));
    m_CreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    m_CreateInfo.imageType = VK_IMAGE_TYPE_2D;
    m_CreateInfo.extent.width = rand.Generate() % (imageSizeMax - imageSizeMin) + imageSizeMin;
    m_CreateInfo.extent.height = rand.Generate() % (imageSizeMax - imageSizeMin) + imageSizeMin;
    m_CreateInfo.extent.depth = 1;
    m_CreateInfo.mipLevels = useMipMaps ?
        CalculateMipMapCount(m_CreateInfo.extent.width, m_CreateInfo.extent.height, m_CreateInfo.extent.depth) : 1;
    m_CreateInfo.arrayLayers = 1;
    m_CreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    m_CreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    m_CreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    m_CreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    m_CreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    m_CreateInfo.flags = 0;
}

void BaseImage::UploadContent()
{
    VkBufferCreateInfo srcBufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    srcBufCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    srcBufCreateInfo.size = 4 * m_CreateInfo.extent.width * m_CreateInfo.extent.height;

    VmaAllocationCreateInfo srcBufAllocCreateInfo = {};
    srcBufAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    srcBufAllocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer srcBuf = nullptr;
    VmaAllocation srcBufAlloc = nullptr;
    VmaAllocationInfo srcAllocInfo = {};
    TEST( vmaCreateBuffer(g_hAllocator, &srcBufCreateInfo, &srcBufAllocCreateInfo, &srcBuf, &srcBufAlloc, &srcAllocInfo) == VK_SUCCESS );

    // Fill texels with: r = x % 255, g = u % 255, b = 13, a = 25
    uint32_t* srcBufPtr = (uint32_t*)srcAllocInfo.pMappedData;
    for(uint32_t y = 0, sizeY = m_CreateInfo.extent.height; y < sizeY; ++y)
    {
        for(uint32_t x = 0, sizeX = m_CreateInfo.extent.width; x < sizeX; ++x, ++srcBufPtr)
        {
            const uint8_t r = (uint8_t)x;
            const uint8_t g = (uint8_t)y;
            const uint8_t b = 13;
            const uint8_t a = 25;
            *srcBufPtr = (uint32_t)r << 24 | (uint32_t)g << 16 |
                (uint32_t)b << 8 | (uint32_t)a;
        }
    }

    BeginSingleTimeCommands();

    // Barrier undefined to transfer dst.
    {
        VkImageMemoryBarrier barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = m_Image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.levelCount = 1;

        vkCmdPipelineBarrier(g_hTemporaryCommandBuffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, // srcStageMask
            VK_PIPELINE_STAGE_TRANSFER_BIT, // dstStageMask
            0, // dependencyFlags
            0, nullptr, // memoryBarriers
            0, nullptr, // bufferMemoryBarriers
            1, &barrier); // imageMemoryBarriers
    }

    // CopyBufferToImage
    {
        VkBufferImageCopy region = {};
        region.bufferOffset = 0;
        region.bufferRowLength = 0; // Zeros mean tightly packed.
        region.bufferImageHeight = 0; // Zeros mean tightly packed.
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = m_CreateInfo.extent;
        vkCmdCopyBufferToImage(g_hTemporaryCommandBuffer, srcBuf, m_Image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    }

    // Barrier transfer dst to fragment shader read only.
    {
        VkImageMemoryBarrier barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = m_Image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.levelCount = 1;

        vkCmdPipelineBarrier(g_hTemporaryCommandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, // srcStageMask
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, // dstStageMask
            0, // dependencyFlags
            0, nullptr, // memoryBarriers
            0, nullptr, // bufferMemoryBarriers
            1, &barrier); // imageMemoryBarriers
    }

    EndSingleTimeCommands();

    vmaDestroyBuffer(g_hAllocator, srcBuf, srcBufAlloc);
}

void BaseImage::ValidateContent(RandomNumberGenerator& rand)
{
    /*
    dstBuf has following layout:
    For each of texels to be sampled, [0..valueCount):
    struct {
        in uint32_t pixelX;
        in uint32_t pixelY;
        out uint32_t pixelColor;
    }
    */

    const uint32_t valueCount = 128;

    VkBufferCreateInfo dstBufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    dstBufCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    dstBufCreateInfo.size = valueCount * sizeof(uint32_t) * 3;

    VmaAllocationCreateInfo dstBufAllocCreateInfo = {};
    dstBufAllocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    dstBufAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

    VkBuffer dstBuf = nullptr;
    VmaAllocation dstBufAlloc = nullptr;
    VmaAllocationInfo dstBufAllocInfo = {};
    TEST( vmaCreateBuffer(g_hAllocator, &dstBufCreateInfo, &dstBufAllocCreateInfo, &dstBuf, &dstBufAlloc, &dstBufAllocInfo) == VK_SUCCESS );

    // Fill dstBuf input data.
    {
        uint32_t* dstBufContent = (uint32_t*)dstBufAllocInfo.pMappedData;
        for(uint32_t i = 0; i < valueCount; ++i)
        {
            const uint32_t x = rand.Generate() % m_CreateInfo.extent.width;
            const uint32_t y = rand.Generate() % m_CreateInfo.extent.height;
            dstBufContent[i * 3    ] = x;
            dstBufContent[i * 3 + 1] = y;
            dstBufContent[i * 3 + 2] = 0;
        }
    }

    VkSamplerCreateInfo samplerCreateInfo = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    samplerCreateInfo.magFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.minFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.unnormalizedCoordinates = VK_TRUE;

    VkSampler sampler = nullptr;
    TEST( vkCreateSampler( g_hDevice, &samplerCreateInfo, nullptr, &sampler) == VK_SUCCESS );

    VkDescriptorSetLayoutBinding bindings[2] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[0].pImmutableSamplers = &sampler;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo descSetLayoutCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    descSetLayoutCreateInfo.bindingCount = 2;
    descSetLayoutCreateInfo.pBindings = bindings;

    VkDescriptorSetLayout descSetLayout = nullptr;
    TEST( vkCreateDescriptorSetLayout(g_hDevice, &descSetLayoutCreateInfo, nullptr, &descSetLayout) == VK_SUCCESS );

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descSetLayout;

    VkPipelineLayout pipelineLayout = nullptr;
    TEST( vkCreatePipelineLayout(g_hDevice, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout) == VK_SUCCESS );

    std::vector<char> shaderCode;
    LoadShader(shaderCode, "SparseBindingTest.comp.spv");

    VkShaderModuleCreateInfo shaderModuleCreateInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    shaderModuleCreateInfo.codeSize = shaderCode.size();
    shaderModuleCreateInfo.pCode = (const uint32_t*)shaderCode.data();

    VkShaderModule shaderModule = nullptr;
    TEST( vkCreateShaderModule(g_hDevice, &shaderModuleCreateInfo, nullptr, &shaderModule) == VK_SUCCESS );

    VkComputePipelineCreateInfo pipelineCreateInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    pipelineCreateInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineCreateInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineCreateInfo.stage.module = shaderModule;
    pipelineCreateInfo.stage.pName = "main";
    pipelineCreateInfo.layout = pipelineLayout;

    VkPipeline pipeline = nullptr;
    TEST( vkCreateComputePipelines(g_hDevice, nullptr, 1, &pipelineCreateInfo, nullptr, &pipeline) == VK_SUCCESS );

    VkDescriptorPoolSize poolSizes[2] = {};
    poolSizes[0].type = bindings[0].descriptorType;
    poolSizes[0].descriptorCount = bindings[0].descriptorCount;
    poolSizes[1].type = bindings[1].descriptorType;
    poolSizes[1].descriptorCount = bindings[1].descriptorCount;

    VkDescriptorPoolCreateInfo descPoolCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    descPoolCreateInfo.maxSets = 1;
    descPoolCreateInfo.poolSizeCount = 2;
    descPoolCreateInfo.pPoolSizes = poolSizes;

    VkDescriptorPool descPool = nullptr;
    TEST( vkCreateDescriptorPool(g_hDevice, &descPoolCreateInfo, nullptr, &descPool) == VK_SUCCESS );

    VkDescriptorSetAllocateInfo descSetAllocInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    descSetAllocInfo.descriptorPool = descPool;
    descSetAllocInfo.descriptorSetCount = 1;
    descSetAllocInfo.pSetLayouts = &descSetLayout;

    VkDescriptorSet descSet = nullptr;
    TEST( vkAllocateDescriptorSets(g_hDevice, &descSetAllocInfo, &descSet) == VK_SUCCESS );

    VkImageViewCreateInfo imageViewCreateInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    imageViewCreateInfo.image = m_Image;
    imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageViewCreateInfo.format = m_CreateInfo.format;
    imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageViewCreateInfo.subresourceRange.layerCount = 1;
    imageViewCreateInfo.subresourceRange.levelCount = 1;

    VkImageView imageView = nullptr;
    TEST( vkCreateImageView(g_hDevice, &imageViewCreateInfo, nullptr, &imageView) == VK_SUCCESS );

    VkDescriptorImageInfo descImageInfo = {};
    descImageInfo.imageView = imageView;
    descImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorBufferInfo descBufferInfo = {};
    descBufferInfo.buffer = dstBuf;
    descBufferInfo.offset = 0;
    descBufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet descWrites[2] = {};
    descWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descWrites[0].dstSet = descSet;
    descWrites[0].dstBinding = bindings[0].binding;
    descWrites[0].dstArrayElement = 0;
    descWrites[0].descriptorCount = 1;
    descWrites[0].descriptorType = bindings[0].descriptorType;
    descWrites[0].pImageInfo = &descImageInfo;
    descWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descWrites[1].dstSet = descSet;
    descWrites[1].dstBinding = bindings[1].binding;
    descWrites[1].dstArrayElement = 0;
    descWrites[1].descriptorCount = 1;
    descWrites[1].descriptorType = bindings[1].descriptorType;
    descWrites[1].pBufferInfo = &descBufferInfo;
    vkUpdateDescriptorSets(g_hDevice, 2, descWrites, 0, nullptr);

    BeginSingleTimeCommands();
    vkCmdBindPipeline(g_hTemporaryCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(g_hTemporaryCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, nullptr);
    vkCmdDispatch(g_hTemporaryCommandBuffer, valueCount, 1, 1);
    EndSingleTimeCommands();

    // Validate dstBuf output data.
    {
        const uint32_t* dstBufContent = (const uint32_t*)dstBufAllocInfo.pMappedData;
        for(uint32_t i = 0; i < valueCount; ++i)
        {
            const uint32_t x     = dstBufContent[i * 3    ];
            const uint32_t y     = dstBufContent[i * 3 + 1];
            const uint32_t color = dstBufContent[i * 3 + 2];
            const uint8_t a = (uint8_t)(color >> 24);
            const uint8_t b = (uint8_t)(color >> 16);
            const uint8_t g = (uint8_t)(color >>  8);
            const uint8_t r = (uint8_t)color;
            TEST(r == (uint8_t)x && g == (uint8_t)y && b == 13 && a == 25);
        }
    }

    vkDestroyImageView(g_hDevice, imageView, nullptr);
    vkDestroyDescriptorPool(g_hDevice, descPool, nullptr);
    vmaDestroyBuffer(g_hAllocator, dstBuf, dstBufAlloc);
    vkDestroyPipeline(g_hDevice, pipeline, nullptr);
    vkDestroyShaderModule(g_hDevice, shaderModule, nullptr);
    vkDestroyPipelineLayout(g_hDevice, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(g_hDevice, descSetLayout, nullptr);
    vkDestroySampler(g_hDevice, sampler, nullptr);
}

////////////////////////////////////////////////////////////////////////////////
// class TraditionalImage

void TraditionalImage::Init(RandomNumberGenerator& rand)
{
    FillImageCreateInfo(rand);

    VmaAllocationCreateInfo allocCreateInfo = {};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    // Default BEST_FIT is clearly better.
    //allocCreateInfo.flags |= VMA_ALLOCATION_CREATE_STRATEGY_WORST_FIT_BIT;

    ERR_GUARD_VULKAN( vmaCreateImage(g_hAllocator, &m_CreateInfo, &allocCreateInfo,
        &m_Image, &m_Allocation, nullptr) );
}

TraditionalImage::~TraditionalImage()
{
    if(m_Allocation)
    {
        vmaFreeMemory(g_hAllocator, m_Allocation);
    }
}

////////////////////////////////////////////////////////////////////////////////
// class SparseBindingImage

void SparseBindingImage::Init(RandomNumberGenerator& rand)
{
    assert(g_SparseBindingEnabled && g_hSparseBindingQueue);

    // Create image.
    FillImageCreateInfo(rand);
    m_CreateInfo.flags |= VK_IMAGE_CREATE_SPARSE_BINDING_BIT;
    ERR_GUARD_VULKAN( vkCreateImage(g_hDevice, &m_CreateInfo, nullptr, &m_Image) );

    // Get memory requirements.
    VkMemoryRequirements imageMemReq;
    vkGetImageMemoryRequirements(g_hDevice, m_Image, &imageMemReq);

    // According to Vulkan specification, for sparse resources memReq.alignment is also page size.
    const VkDeviceSize pageSize = imageMemReq.alignment;
    const uint32_t pageCount = (uint32_t)ceil_div<VkDeviceSize>(imageMemReq.size, pageSize);

    VmaAllocationCreateInfo allocCreateInfo = {};
    allocCreateInfo.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkMemoryRequirements pageMemReq = imageMemReq;
    pageMemReq.size = pageSize;

    // Allocate and bind memory pages.
    m_Allocations.resize(pageCount);
    std::fill(m_Allocations.begin(), m_Allocations.end(), nullptr);
    std::vector<VkSparseMemoryBind> binds{pageCount};
    std::vector<VmaAllocationInfo> allocInfo{pageCount};
    ERR_GUARD_VULKAN( vmaAllocateMemoryPages(g_hAllocator, &pageMemReq, &allocCreateInfo, pageCount, m_Allocations.data(), allocInfo.data()) );

    for(uint32_t i = 0; i < pageCount; ++i)
    {
        binds[i] = {};
        binds[i].resourceOffset = pageSize * i;
        binds[i].size = pageSize;
        binds[i].memory = allocInfo[i].deviceMemory;
        binds[i].memoryOffset = allocInfo[i].offset;
    }

    VkSparseImageOpaqueMemoryBindInfo imageBindInfo;
    imageBindInfo.image = m_Image;
    imageBindInfo.bindCount = pageCount;
    imageBindInfo.pBinds = binds.data();

    VkBindSparseInfo bindSparseInfo = { VK_STRUCTURE_TYPE_BIND_SPARSE_INFO };
    bindSparseInfo.pImageOpaqueBinds = &imageBindInfo;
    bindSparseInfo.imageOpaqueBindCount = 1;

    ERR_GUARD_VULKAN( vkResetFences(g_hDevice, 1, &g_ImmediateFence) );
    ERR_GUARD_VULKAN( vkQueueBindSparse(g_hSparseBindingQueue, 1, &bindSparseInfo, g_ImmediateFence) );
    ERR_GUARD_VULKAN( vkWaitForFences(g_hDevice, 1, &g_ImmediateFence, VK_TRUE, UINT64_MAX) );
}

SparseBindingImage::~SparseBindingImage()
{
    vmaFreeMemoryPages(g_hAllocator, m_Allocations.size(), m_Allocations.data());
}

////////////////////////////////////////////////////////////////////////////////
// Private functions

////////////////////////////////////////////////////////////////////////////////
// Public functions

void TestSparseBinding()
{
    wprintf(L"TESTING SPARSE BINDING:\n");

    struct ImageInfo
    {
        std::unique_ptr<BaseImage> image;
        uint32_t endFrame;
    };
    std::vector<ImageInfo> images;

    constexpr uint32_t frameCount = 1000;
    constexpr uint32_t imageLifeFramesMin = 1;
    constexpr uint32_t imageLifeFramesMax = 400;

    RandomNumberGenerator rand(4652467);

    for(uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex)
    {
        // Bump frame index.
        ++g_FrameIndex;
        vmaSetCurrentFrameIndex(g_hAllocator, g_FrameIndex);

        // Create one new, random image.
        ImageInfo imageInfo;
        //imageInfo.image = std::make_unique<TraditionalImage>();
        imageInfo.image = std::make_unique<SparseBindingImage>();
        imageInfo.image->Init(rand);
        imageInfo.endFrame = g_FrameIndex + rand.Generate() % (imageLifeFramesMax - imageLifeFramesMin) + imageLifeFramesMin;
        images.push_back(std::move(imageInfo));

        // Delete all images that expired.
        for(size_t imageIndex = images.size(); imageIndex--; )
        {
            if(g_FrameIndex >= images[imageIndex].endFrame)
            {
                images.erase(images.begin() + imageIndex);
            }
        }
    }

    SaveAllocatorStatsToFile(L"SparseBindingTest.json");

    // Choose biggest image. Test uploading and sampling.
    BaseImage* biggestImage = nullptr;
    for(size_t i = 0, count = images.size(); i < count; ++i)
    {
        if(!biggestImage ||
            images[i].image->GetCreateInfo().extent.width * images[i].image->GetCreateInfo().extent.height >
                biggestImage->GetCreateInfo().extent.width * biggestImage->GetCreateInfo().extent.height)
        {
            biggestImage = images[i].image.get();
        }
    }
    assert(biggestImage);

    biggestImage->TestContent(rand);

    // Free remaining images.
    images.clear();

    wprintf(L"Done.\n");
}

#endif // #ifdef _WIN32
