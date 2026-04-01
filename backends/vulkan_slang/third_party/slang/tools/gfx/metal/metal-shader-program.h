// metal-shader-program.h
#pragma once

#include "metal-base.h"
#include "metal-shader-object-layout.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

class ShaderProgramImpl : public ShaderProgramBase
{
public:
    DeviceImpl* m_device;
    RefPtr<RootShaderObjectLayoutImpl> m_rootObjectLayout;

    struct Module
    {
        SlangStage stage;
        String entryPointName;
        ComPtr<ISlangBlob> code;
        NS::SharedPtr<MTL::Library> library;
    };

    List<Module> m_modules;

    ShaderProgramImpl(DeviceImpl* device);
    ~ShaderProgramImpl();

    virtual Result createShaderModule(
        slang::EntryPointReflection* entryPointInfo,
        List<ComPtr<ISlangBlob>>& kernelCodes) override;
};


} // namespace metal
} // namespace gfx
