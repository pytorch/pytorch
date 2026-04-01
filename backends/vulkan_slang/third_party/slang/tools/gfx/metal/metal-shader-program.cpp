// metal-shader-program.cpp
#include "metal-shader-program.h"

#include "metal-device.h"
#include "metal-util.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

ShaderProgramImpl::ShaderProgramImpl(DeviceImpl* device)
    : m_device(device)
{
}

ShaderProgramImpl::~ShaderProgramImpl() {}

Result ShaderProgramImpl::createShaderModule(
    slang::EntryPointReflection* entryPointInfo,
    List<ComPtr<ISlangBlob>>& kernelCodes)
{
    Module module;
    module.stage = entryPointInfo->getStage();
    module.entryPointName = entryPointInfo->getNameOverride();
    module.code = kernelCodes[0];

    dispatch_data_t data = dispatch_data_create(
        kernelCodes[0]->getBufferPointer(),
        kernelCodes[0]->getBufferSize(),
        dispatch_get_main_queue(),
        NULL);
    NS::Error* error;
    module.library = NS::TransferPtr(m_device->m_device->newLibrary(data, &error));
    if (!module.library)
    {
        // TODO use better mechanism for reporting errors
        std::cout << error->localizedDescription()->utf8String() << std::endl;
        return SLANG_E_INVALID_ARG;
    }

    m_modules.add(module);
    return SLANG_OK;
}

} // namespace metal
} // namespace gfx
