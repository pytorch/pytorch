// d3d12-shader-program.h
#pragma once

#include "d3d12-base.h"
#include "d3d12-shader-object-layout.h"

namespace gfx
{
namespace d3d12
{

using namespace Slang;

struct ShaderBinary
{
    SlangStage stage;
    slang::EntryPointReflection* entryPointInfo;
    String actualEntryPointNameInAPI;
    List<uint8_t> code;
};

class ShaderProgramImpl : public ShaderProgramBase
{
public:
    RefPtr<RootShaderObjectLayoutImpl> m_rootObjectLayout;
    List<ShaderBinary> m_shaders;

    virtual Result createShaderModule(
        slang::EntryPointReflection* entryPointInfo,
        List<ComPtr<ISlangBlob>>& kernelCodes) override;
};

} // namespace d3d12
} // namespace gfx
