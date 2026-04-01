// vk-shader-program.cpp
#include "vk-shader-program.h"

#include "external/spirv-tools/include/spirv-tools/linker.hpp"
#include "vk-device.h"
#include "vk-util.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

ShaderProgramImpl::ShaderProgramImpl(DeviceImpl* device)
    : m_device(device)
{
    for (auto& shaderModule : m_modules)
        shaderModule = VK_NULL_HANDLE;
}

ShaderProgramImpl::~ShaderProgramImpl()
{
    for (auto shaderModule : m_modules)
    {
        if (shaderModule != VK_NULL_HANDLE)
        {
            m_device->m_api.vkDestroyShaderModule(m_device->m_api.m_device, shaderModule, nullptr);
        }
    }
}

void ShaderProgramImpl::comFree()
{
    m_device.breakStrongReference();
}

VkPipelineShaderStageCreateInfo ShaderProgramImpl::compileEntryPoint(
    const char* entryPointName,
    ISlangBlob* code,
    VkShaderStageFlagBits stage,
    VkShaderModule& outShaderModule)
{
    char const* dataBegin = (char const*)code->getBufferPointer();
    char const* dataEnd = (char const*)code->getBufferPointer() + code->getBufferSize();

    // We need to make a copy of the code, since the Slang compiler
    // will free the memory after a compile request is closed.

    VkShaderModuleCreateInfo moduleCreateInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleCreateInfo.pCode = (uint32_t*)code->getBufferPointer();
    moduleCreateInfo.codeSize = code->getBufferSize();

    VkShaderModule module;
    SLANG_VK_CHECK(m_device->m_api.vkCreateShaderModule(
        m_device->m_device,
        &moduleCreateInfo,
        nullptr,
        &module));
    outShaderModule = module;

    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    shaderStageCreateInfo.stage = stage;

    shaderStageCreateInfo.module = module;
    shaderStageCreateInfo.pName = entryPointName;

    return shaderStageCreateInfo;
}

Result ShaderProgramImpl::createShaderModule(
    slang::EntryPointReflection* entryPointInfo,
    List<ComPtr<ISlangBlob>>& kernelCodes)
{
    ComPtr<ISlangBlob> linkedKernel;
    ComPtr<slang::ISession> slangSession;
    m_device->getSlangSession(slangSession.writeRef());
    if (kernelCodes.getCount() == 1)
    {
        linkedKernel = kernelCodes[0];
    }
    else
    {
        linkedKernel = m_device->m_glslang.linkSPIRV(kernelCodes);
        if (!linkedKernel)
        {
            return SLANG_FAIL;
        }
    }

    m_codeBlobs.add(linkedKernel);

    VkShaderModule shaderModule;
    auto realEntryPointName = entryPointInfo->getNameOverride();
    const char* spirvBinaryEntryPointName = "main";
    m_stageCreateInfos.add(compileEntryPoint(
        spirvBinaryEntryPointName,
        linkedKernel,
        (VkShaderStageFlagBits)VulkanUtil::getShaderStage(entryPointInfo->getStage()),
        shaderModule));
    m_entryPointNames.add(realEntryPointName);
    m_modules.add(shaderModule);
    return SLANG_OK;
}

} // namespace vk
} // namespace gfx
