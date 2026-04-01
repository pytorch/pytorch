// metal-pipeline-state.cpp
#include "metal-pipeline-state.h"

#include "metal-device.h"
#include "metal-shader-object-layout.h"
#include "metal-shader-program.h"
#include "metal-util.h"
#include "metal-vertex-layout.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

PipelineStateImpl::PipelineStateImpl(DeviceImpl* device)
    : m_device(device)
{
}

PipelineStateImpl::~PipelineStateImpl() {}

void PipelineStateImpl::init(const GraphicsPipelineStateDesc& desc)
{
    PipelineStateDesc pipelineDesc;
    pipelineDesc.type = PipelineType::Graphics;
    pipelineDesc.graphics = desc;
    initializeBase(pipelineDesc);
}

void PipelineStateImpl::init(const ComputePipelineStateDesc& desc)
{
    PipelineStateDesc pipelineDesc;
    pipelineDesc.type = PipelineType::Compute;
    pipelineDesc.compute = desc;
    initializeBase(pipelineDesc);
}

void PipelineStateImpl::init(const RayTracingPipelineStateDesc& desc)
{
    PipelineStateDesc pipelineDesc;
    pipelineDesc.type = PipelineType::RayTracing;
    pipelineDesc.rayTracing.set(desc);
    initializeBase(pipelineDesc);
}

Result PipelineStateImpl::createMetalRenderPipelineState()
{
    auto programImpl = static_cast<ShaderProgramImpl*>(m_program.Ptr());
    if (!programImpl)
        return SLANG_FAIL;

    NS::SharedPtr<MTL::RenderPipelineDescriptor> pd =
        NS::TransferPtr(MTL::RenderPipelineDescriptor::alloc()->init());

    for (const ShaderProgramImpl::Module& module : programImpl->m_modules)
    {
        auto functionName = MetalUtil::createString(module.entryPointName.getBuffer());
        NS::SharedPtr<MTL::Function> function =
            NS::TransferPtr(module.library->newFunction(functionName.get()));
        if (!function)
            return SLANG_FAIL;

        switch (module.stage)
        {
        case SLANG_STAGE_VERTEX:
            pd->setVertexFunction(function.get());
            break;
        case SLANG_STAGE_FRAGMENT:
            pd->setFragmentFunction(function.get());
            break;
        default:
            return SLANG_FAIL;
        }
    }

    // Create a vertex descriptor with the vertex buffer binding indices being offset.
    // They need to be in a range not used by any buffers in the root object layout.
    // The +1 is to account for a potential constant buffer at index 0.
    m_vertexBufferOffset = programImpl->m_rootObjectLayout->getBufferCount() + 1;
    auto inputLayoutImpl = static_cast<InputLayoutImpl*>(desc.graphics.inputLayout);
    NS::SharedPtr<MTL::VertexDescriptor> vertexDescriptor =
        inputLayoutImpl->createVertexDescriptor(m_vertexBufferOffset);
    pd->setVertexDescriptor(vertexDescriptor.get());
    pd->setInputPrimitiveTopology(
        MetalUtil::translatePrimitiveTopologyClass(desc.graphics.primitiveType));

    // Set rasterization state
    auto framebufferLayoutImpl =
        static_cast<FramebufferLayoutImpl*>(desc.graphics.framebufferLayout);
    const auto& blend = desc.graphics.blend;
    GfxCount sampleCount = 1;

    pd->setAlphaToCoverageEnabled(blend.alphaToCoverageEnable);
    // pd->setAlphaToOneEnabled(); // Currently not supported by gfx
    // pd->setRasterizationEnabled(true); // Enabled by default

    for (Index i = 0; i < framebufferLayoutImpl->m_renderTargets.getCount(); ++i)
    {
        const IFramebufferLayout::TargetLayout& targetLayout =
            framebufferLayoutImpl->m_renderTargets[i];
        MTL::RenderPipelineColorAttachmentDescriptor* colorAttachment =
            pd->colorAttachments()->object(i);
        colorAttachment->setPixelFormat(MetalUtil::translatePixelFormat(targetLayout.format));
        if (i < blend.targetCount)
        {
            const TargetBlendDesc& targetBlendDesc = blend.targets[i];
            colorAttachment->setBlendingEnabled(targetBlendDesc.enableBlend);
            colorAttachment->setSourceRGBBlendFactor(
                MetalUtil::translateBlendFactor(targetBlendDesc.color.srcFactor));
            colorAttachment->setDestinationRGBBlendFactor(
                MetalUtil::translateBlendFactor(targetBlendDesc.color.dstFactor));
            colorAttachment->setRgbBlendOperation(
                MetalUtil::translateBlendOperation(targetBlendDesc.color.op));
            colorAttachment->setSourceAlphaBlendFactor(
                MetalUtil::translateBlendFactor(targetBlendDesc.alpha.srcFactor));
            colorAttachment->setDestinationAlphaBlendFactor(
                MetalUtil::translateBlendFactor(targetBlendDesc.alpha.dstFactor));
            colorAttachment->setAlphaBlendOperation(
                MetalUtil::translateBlendOperation(targetBlendDesc.alpha.op));
            colorAttachment->setWriteMask(
                MetalUtil::translateColorWriteMask(targetBlendDesc.writeMask));
        }
        sampleCount = Math::Max(sampleCount, targetLayout.sampleCount);
    }
    if (framebufferLayoutImpl->m_depthStencil.format != Format::Unknown)
    {
        const IFramebufferLayout::TargetLayout& depthStencil =
            framebufferLayoutImpl->m_depthStencil;
        MTL::PixelFormat pixelFormat = MetalUtil::translatePixelFormat(depthStencil.format);
        if (MetalUtil::isDepthFormat(pixelFormat))
        {
            pd->setDepthAttachmentPixelFormat(MetalUtil::translatePixelFormat(depthStencil.format));
        }
        if (MetalUtil::isStencilFormat(pixelFormat))
        {
            pd->setStencilAttachmentPixelFormat(
                MetalUtil::translatePixelFormat(depthStencil.format));
        }
    }

    pd->setRasterSampleCount(sampleCount);

    NS::Error* error;
    m_renderPipelineState =
        NS::TransferPtr(m_device->m_device->newRenderPipelineState(pd.get(), &error));
    if (!m_renderPipelineState)
    {
        std::cout << error->localizedDescription()->utf8String() << std::endl;
        return SLANG_E_INVALID_ARG;
    }

    // Create depth stencil state
    auto createStencilDesc = [](const DepthStencilOpDesc& desc,
                                uint32_t readMask,
                                uint32_t writeMask) -> NS::SharedPtr<MTL::StencilDescriptor>
    {
        NS::SharedPtr<MTL::StencilDescriptor> stencilDesc =
            NS::TransferPtr(MTL::StencilDescriptor::alloc()->init());
        stencilDesc->setStencilCompareFunction(
            MetalUtil::translateCompareFunction(desc.stencilFunc));
        stencilDesc->setStencilFailureOperation(
            MetalUtil::translateStencilOperation(desc.stencilFailOp));
        stencilDesc->setDepthFailureOperation(
            MetalUtil::translateStencilOperation(desc.stencilDepthFailOp));
        stencilDesc->setDepthStencilPassOperation(
            MetalUtil::translateStencilOperation(desc.stencilPassOp));
        stencilDesc->setReadMask(readMask);
        stencilDesc->setWriteMask(writeMask);
        return stencilDesc;
    };

    const auto& depthStencil = desc.graphics.depthStencil;
    NS::SharedPtr<MTL::DepthStencilDescriptor> depthStencilDesc =
        NS::TransferPtr(MTL::DepthStencilDescriptor::alloc()->init());
    m_depthStencilState =
        NS::TransferPtr(m_device->m_device->newDepthStencilState(depthStencilDesc.get()));
    if (!m_depthStencilState)
    {
        return SLANG_FAIL;
    }
    if (depthStencil.depthTestEnable)
    {
        depthStencilDesc->setDepthCompareFunction(
            MetalUtil::translateCompareFunction(depthStencil.depthFunc));
    }
    depthStencilDesc->setDepthWriteEnabled(depthStencil.depthWriteEnable);
    if (depthStencil.stencilEnable)
    {
        depthStencilDesc->setFrontFaceStencil(createStencilDesc(
                                                  depthStencil.frontFace,
                                                  depthStencil.stencilReadMask,
                                                  depthStencil.stencilWriteMask)
                                                  .get());
        depthStencilDesc->setBackFaceStencil(createStencilDesc(
                                                 depthStencil.backFace,
                                                 depthStencil.stencilReadMask,
                                                 depthStencil.stencilWriteMask)
                                                 .get());
    }

    return SLANG_OK;
}

Result PipelineStateImpl::createMetalComputePipelineState()
{
    auto programImpl = static_cast<ShaderProgramImpl*>(m_program.Ptr());
    if (!programImpl)
        return SLANG_FAIL;

    const ShaderProgramImpl::Module& module = programImpl->m_modules[0];
    auto functionName = MetalUtil::createString(module.entryPointName.getBuffer());
    NS::SharedPtr<MTL::Function> function =
        NS::TransferPtr(module.library->newFunction(functionName.get()));
    if (!function)
        return SLANG_FAIL;

    NS::Error* error;
    m_computePipelineState =
        NS::TransferPtr(m_device->m_device->newComputePipelineState(function.get(), &error));

    // Query thread group size for use during dispatch.
    SlangUInt threadGroupSize[3];
    programImpl->linkedProgram->getLayout()->getEntryPointByIndex(0)->getComputeThreadGroupSize(
        3,
        threadGroupSize);
    m_threadGroupSize = MTL::Size(threadGroupSize[0], threadGroupSize[1], threadGroupSize[2]);

    return m_computePipelineState ? SLANG_OK : SLANG_FAIL;
}

Result PipelineStateImpl::ensureAPIPipelineStateCreated()
{
    AUTORELEASEPOOL

    switch (desc.type)
    {
    case PipelineType::Compute:
        return m_computePipelineState ? SLANG_OK : createMetalComputePipelineState();
    case PipelineType::Graphics:
        return m_renderPipelineState ? SLANG_OK : createMetalRenderPipelineState();
    default:
        SLANG_UNREACHABLE("Unknown pipeline type.");
        return SLANG_FAIL;
    }
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL PipelineStateImpl::getNativeHandle(InteropHandle* outHandle)
{
    switch (desc.type)
    {
    case PipelineType::Compute:
        outHandle->api = InteropHandleAPI::Metal;
        outHandle->handleValue = reinterpret_cast<intptr_t>(m_computePipelineState.get());
        return SLANG_OK;
    case PipelineType::Graphics:
        outHandle->api = InteropHandleAPI::Metal;
        outHandle->handleValue = reinterpret_cast<intptr_t>(m_renderPipelineState.get());
        return SLANG_OK;
    }
    return SLANG_FAIL;
}

RayTracingPipelineStateImpl::RayTracingPipelineStateImpl(DeviceImpl* device)
    : PipelineStateImpl(device)
{
}

Result RayTracingPipelineStateImpl::ensureAPIPipelineStateCreated()
{
    return SLANG_E_NOT_IMPLEMENTED;
}

Result RayTracingPipelineStateImpl::getNativeHandle(InteropHandle* outHandle)
{
    return SLANG_E_NOT_IMPLEMENTED;
}


} // namespace metal
} // namespace gfx
