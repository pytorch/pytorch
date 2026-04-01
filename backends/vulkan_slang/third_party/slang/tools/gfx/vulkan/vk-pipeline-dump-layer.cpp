#include "vk-pipeline-dump-layer.h"

#include "core/slang-basic.h"
#include "core/slang-stream.h"
namespace gfx
{
using namespace Slang;

struct PipelineDumpContext
{
    Dictionary<VkPipelineLayout, Index> pipelineLayouts;
    Dictionary<VkShaderModule, Index> shaderModules;
    Dictionary<VkDescriptorSetLayout, Index> descriptorSets;
    Dictionary<VkPipeline, Index> computePipelines;

    List<uint8_t> serializedBytes;

    VulkanApi api;

    template<typename T>
    void writeRaw(T v)
    {
        auto startIndex = serializedBytes.getCount();
        serializedBytes.growToCount(startIndex + sizeof(T));
        memcpy(serializedBytes.getBuffer() + startIndex, &v, sizeof(T));
    }

    template<typename T>
    void writeArray(uint32_t elementCount, const T* data)
    {
        writeRaw(elementCount);

        auto startIndex = serializedBytes.getCount();
        serializedBytes.growToCount(startIndex + sizeof(T) * elementCount);
        memcpy(serializedBytes.getBuffer() + startIndex, data, sizeof(T) * elementCount);
    }

    void writeStr(const char* str)
    {
        auto len = (uint32_t)strlen(str) + 1;
        writeRaw(len);

        auto startIndex = serializedBytes.getCount();
        serializedBytes.growToCount(startIndex + len);
        memcpy(serializedBytes.getBuffer() + startIndex, str, len - 1);
        serializedBytes[startIndex + len - 1] = 0;
    }

    void writePipelineLayout(VkPipelineLayout layout, const VkPipelineLayoutCreateInfo* createInfo)
    {
        auto startIndex = serializedBytes.getCount();
        writeRaw(createInfo->sType);
        writeRaw(createInfo->flags);
        writeRaw(createInfo->setLayoutCount);
        for (uint32_t i = 0; i < createInfo->setLayoutCount; i++)
            writeRaw(descriptorSets.getValue(createInfo->pSetLayouts[i]));
        writeArray(createInfo->pushConstantRangeCount, createInfo->pPushConstantRanges);
        pipelineLayouts[layout] = startIndex;
    }

    void writeShaderModule(VkShaderModule module, const VkShaderModuleCreateInfo* createInfo)
    {
        auto startIndex = serializedBytes.getCount();
        writeRaw(createInfo->sType);
        writeRaw(createInfo->flags);
        writeArray((uint32_t)(createInfo->codeSize / sizeof(uint32_t)), createInfo->pCode);
        shaderModules[module] = startIndex;
    }

    void writeDescriptorSetLayout(
        VkDescriptorSetLayout layout,
        const VkDescriptorSetLayoutCreateInfo* createInfo)
    {
        auto startIndex = serializedBytes.getCount();
        writeRaw(createInfo->sType);
        writeRaw(createInfo->flags);
        writeArray(createInfo->bindingCount, createInfo->pBindings);
        descriptorSets[layout] = startIndex;
    }

    void writePipeline(VkPipeline pipeline, const VkComputePipelineCreateInfo* createInfo)
    {
        auto startIndex = serializedBytes.getCount();
        writeRaw(createInfo->sType);
        writeRaw(createInfo->flags);
        writeRaw(createInfo->stage.sType);
        writeRaw(createInfo->stage.flags);
        writeRaw(createInfo->stage.stage);
        writeRaw(shaderModules.getValue(createInfo->stage.module));
        writeStr(createInfo->stage.pName);
        writeRaw(pipelineLayouts.getValue(createInfo->layout));
        computePipelines[pipeline] = startIndex;
    }

    void writeToFile(UnownedStringSlice path)
    {
        RefPtr<FileStream> fs = new FileStream();
        fs->init(path, FileMode::Create);
        uint32_t pipelineCount = (uint32_t)computePipelines.getCount();
        fs->write(&pipelineCount, sizeof(uint32_t));
        for (auto& pair : computePipelines)
        {
            fs->write(KeyValueDetail::getValue(&pair), sizeof(Index));
        }
        Index blobSize = serializedBytes.getCount();
        fs->write(&blobSize, sizeof(blobSize));
        fs->write(serializedBytes.getBuffer(), serializedBytes.getCount());
        fs->close();
    }
};

PipelineDumpContext dumpContext;

VkResult SLANG_MCALL createPipelineLayout(
    VkDevice device,
    const VkPipelineLayoutCreateInfo* createInfo,
    const VkAllocationCallbacks* callbacks,
    VkPipelineLayout* outLayout)
{
    auto result = dumpContext.api.vkCreatePipelineLayout(device, createInfo, callbacks, outLayout);
    dumpContext.writePipelineLayout(*outLayout, createInfo);
    return result;
}

VkResult SLANG_MCALL createComputePipelines(
    VkDevice device,
    VkPipelineCache cache,
    uint32_t createInfoCount,
    const VkComputePipelineCreateInfo* createInfos,
    const VkAllocationCallbacks* callbacks,
    VkPipeline* outPipelines)
{
    auto result = dumpContext.api.vkCreateComputePipelines(
        device,
        cache,
        createInfoCount,
        createInfos,
        callbacks,
        outPipelines);
    for (uint32_t i = 0; i < createInfoCount; i++)
        dumpContext.writePipeline(outPipelines[i], createInfos + i);
    return result;
}

VkResult SLANG_MCALL createShaderModule(
    VkDevice device,
    const VkShaderModuleCreateInfo* createInfo,
    const VkAllocationCallbacks* callbacks,
    VkShaderModule* outShaderModule)
{
    auto result =
        dumpContext.api.vkCreateShaderModule(device, createInfo, callbacks, outShaderModule);
    dumpContext.writeShaderModule(*outShaderModule, createInfo);
    return result;
}

VkResult SLANG_MCALL createDescriptorSetLayout(
    VkDevice device,
    const VkDescriptorSetLayoutCreateInfo* createInfo,
    const VkAllocationCallbacks* callbacks,
    VkDescriptorSetLayout* outDescSetLayout)
{
    auto result = dumpContext.api
                      .vkCreateDescriptorSetLayout(device, createInfo, callbacks, outDescSetLayout);
    dumpContext.writeDescriptorSetLayout(*outDescSetLayout, createInfo);
    return result;
}

void installPipelineDumpLayer(VulkanApi& api)
{
    dumpContext.api = api;
    api.vkCreatePipelineLayout = createPipelineLayout;
    api.vkCreateComputePipelines = createComputePipelines;
    api.vkCreateShaderModule = createShaderModule;
    api.vkCreateDescriptorSetLayout = createDescriptorSetLayout;
}

void writePipelineDump(UnownedStringSlice path)
{
    dumpContext.writeToFile(path);
}
} // namespace gfx
