// vk-shader-program.h
#pragma once

#include "vk-base.h"
#include "vk-shader-object-layout.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

class ShaderProgramImpl : public ShaderProgramBase
{
public:
    ShaderProgramImpl(DeviceImpl* device);

    ~ShaderProgramImpl();

    virtual void comFree() override;

    BreakableReference<DeviceImpl> m_device;

    List<VkPipelineShaderStageCreateInfo> m_stageCreateInfos;
    List<String> m_entryPointNames;
    List<ComPtr<ISlangBlob>> m_codeBlobs; //< To keep storage of code in scope
    List<VkShaderModule> m_modules;
    RefPtr<RootShaderObjectLayout> m_rootObjectLayout;

    VkPipelineShaderStageCreateInfo compileEntryPoint(
        const char* entryPointName,
        ISlangBlob* code,
        VkShaderStageFlagBits stage,
        VkShaderModule& outShaderModule);

    virtual Result createShaderModule(
        slang::EntryPointReflection* entryPointInfo,
        List<ComPtr<ISlangBlob>>& kernelCodes) override;
};


} // namespace vk
} // namespace gfx
