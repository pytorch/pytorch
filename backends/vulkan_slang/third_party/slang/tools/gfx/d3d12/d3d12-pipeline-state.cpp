// d3d12-ray-tracing.cpp
#include "d3d12-pipeline-state.h"

#ifdef GFX_NVAPI
#include "../nvapi/nvapi-include.h"
#endif

#include "../nvapi/nvapi-util.h"
#include "d3d12-device.h"
#include "d3d12-framebuffer.h"
#include "d3d12-pipeline-state-stream.h"
#include "d3d12-shader-program.h"
#include "d3d12-vertex-layout.h"

#include <climits>

namespace gfx
{
namespace d3d12
{

using namespace Slang;

void PipelineStateImpl::init(const GraphicsPipelineStateDesc& inDesc)
{
    PipelineStateDesc pipelineDesc;
    pipelineDesc.type = PipelineType::Graphics;
    pipelineDesc.graphics = inDesc;
    initializeBase(pipelineDesc);
}

void PipelineStateImpl::init(const ComputePipelineStateDesc& inDesc)
{
    PipelineStateDesc pipelineDesc;
    pipelineDesc.type = PipelineType::Compute;
    pipelineDesc.compute = inDesc;
    initializeBase(pipelineDesc);
}

Result PipelineStateImpl::getNativeHandle(InteropHandle* outHandle)
{
    SLANG_RETURN_ON_FAIL(ensureAPIPipelineStateCreated());
    outHandle->api = InteropHandleAPI::D3D12;
    outHandle->handleValue = reinterpret_cast<uint64_t>(m_pipelineState.get());
    return SLANG_OK;
}

Result PipelineStateImpl::ensureAPIPipelineStateCreated()
{
    if (m_pipelineState)
        return SLANG_OK;

    auto programImpl = static_cast<ShaderProgramImpl*>(m_program.Ptr());
    if (programImpl->m_shaders.getCount() == 0)
    {
        SLANG_RETURN_ON_FAIL(programImpl->compileShaders(m_device));
    }
    if (desc.type == PipelineType::Graphics)
    {
        // Only actually create a D3D12 pipeline state if the pipeline is fully specialized.
        auto inputLayoutImpl = (InputLayoutImpl*)desc.graphics.inputLayout;

        // A helper to fill common fields between graphics and mesh pipeline descs
        const auto fillCommonGraphicsState = [&](auto& psoDesc)
        {
            psoDesc.pRootSignature = programImpl->m_rootObjectLayout->m_rootSignature;

            psoDesc.PrimitiveTopologyType = D3DUtil::getPrimitiveType(desc.graphics.primitiveType);

            {
                auto framebufferLayout =
                    static_cast<FramebufferLayoutImpl*>(desc.graphics.framebufferLayout);
                const int numRenderTargets = int(framebufferLayout->m_renderTargets.getCount());

                if (framebufferLayout->m_hasDepthStencil)
                {
                    psoDesc.DSVFormat =
                        D3DUtil::getMapFormat(framebufferLayout->m_depthStencil.format);
                    psoDesc.SampleDesc.Count = framebufferLayout->m_depthStencil.sampleCount;
                }
                else
                {
                    psoDesc.DSVFormat = DXGI_FORMAT_UNKNOWN;
                    if (framebufferLayout->m_renderTargets.getCount())
                    {
                        psoDesc.SampleDesc.Count =
                            framebufferLayout->m_renderTargets[0].sampleCount;
                    }
                }
                psoDesc.NumRenderTargets = numRenderTargets;
                for (Int i = 0; i < numRenderTargets; i++)
                {
                    psoDesc.RTVFormats[i] =
                        D3DUtil::getMapFormat(framebufferLayout->m_renderTargets[i].format);
                }

                psoDesc.SampleDesc.Quality = 0;
                psoDesc.SampleMask = UINT_MAX;
            }

            {
                auto& rs = psoDesc.RasterizerState;
                rs.FillMode = D3DUtil::getFillMode(desc.graphics.rasterizer.fillMode);
                rs.CullMode = D3DUtil::getCullMode(desc.graphics.rasterizer.cullMode);
                rs.FrontCounterClockwise =
                    desc.graphics.rasterizer.frontFace == gfx::FrontFaceMode::CounterClockwise
                        ? TRUE
                        : FALSE;
                rs.DepthBias = desc.graphics.rasterizer.depthBias;
                rs.DepthBiasClamp = desc.graphics.rasterizer.depthBiasClamp;
                rs.SlopeScaledDepthBias = desc.graphics.rasterizer.slopeScaledDepthBias;
                rs.DepthClipEnable = desc.graphics.rasterizer.depthClipEnable ? TRUE : FALSE;
                rs.MultisampleEnable = desc.graphics.rasterizer.multisampleEnable ? TRUE : FALSE;
                rs.AntialiasedLineEnable =
                    desc.graphics.rasterizer.antialiasedLineEnable ? TRUE : FALSE;
                rs.ForcedSampleCount = desc.graphics.rasterizer.forcedSampleCount;
                rs.ConservativeRaster = desc.graphics.rasterizer.enableConservativeRasterization
                                            ? D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON
                                            : D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
            }

            {
                D3D12_BLEND_DESC& blend = psoDesc.BlendState;
                blend.IndependentBlendEnable = FALSE;
                blend.AlphaToCoverageEnable =
                    desc.graphics.blend.alphaToCoverageEnable ? TRUE : FALSE;
                blend.RenderTarget[0].RenderTargetWriteMask =
                    (uint8_t)RenderTargetWriteMask::EnableAll;
                for (GfxIndex i = 0; i < desc.graphics.blend.targetCount; i++)
                {
                    auto& d3dDesc = blend.RenderTarget[i];
                    d3dDesc.BlendEnable = desc.graphics.blend.targets[i].enableBlend ? TRUE : FALSE;
                    d3dDesc.BlendOp = D3DUtil::getBlendOp(desc.graphics.blend.targets[i].color.op);
                    d3dDesc.BlendOpAlpha =
                        D3DUtil::getBlendOp(desc.graphics.blend.targets[i].alpha.op);
                    d3dDesc.DestBlend =
                        D3DUtil::getBlendFactor(desc.graphics.blend.targets[i].color.dstFactor);
                    d3dDesc.DestBlendAlpha =
                        D3DUtil::getBlendFactor(desc.graphics.blend.targets[i].alpha.dstFactor);
                    d3dDesc.LogicOp = D3D12_LOGIC_OP_NOOP;
                    d3dDesc.LogicOpEnable = FALSE;
                    d3dDesc.RenderTargetWriteMask = desc.graphics.blend.targets[i].writeMask;
                    d3dDesc.SrcBlend =
                        D3DUtil::getBlendFactor(desc.graphics.blend.targets[i].color.srcFactor);
                    d3dDesc.SrcBlendAlpha =
                        D3DUtil::getBlendFactor(desc.graphics.blend.targets[i].alpha.srcFactor);
                }
                for (GfxIndex i = 1; i < desc.graphics.blend.targetCount; i++)
                {
                    if (memcmp(
                            &desc.graphics.blend.targets[i],
                            &desc.graphics.blend.targets[0],
                            sizeof(desc.graphics.blend.targets[0])) != 0)
                    {
                        blend.IndependentBlendEnable = TRUE;
                        break;
                    }
                }
                for (uint32_t i = (uint32_t)desc.graphics.blend.targetCount;
                     i < D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT;
                     ++i)
                {
                    blend.RenderTarget[i] = blend.RenderTarget[0];
                }
            }

            {
                auto& ds = psoDesc.DepthStencilState;

                ds.DepthEnable = desc.graphics.depthStencil.depthTestEnable;
                ds.DepthWriteMask = desc.graphics.depthStencil.depthWriteEnable
                                        ? D3D12_DEPTH_WRITE_MASK_ALL
                                        : D3D12_DEPTH_WRITE_MASK_ZERO;
                ds.DepthFunc = D3DUtil::getComparisonFunc(desc.graphics.depthStencil.depthFunc);
                ds.StencilEnable = desc.graphics.depthStencil.stencilEnable;
                ds.StencilReadMask = (UINT8)desc.graphics.depthStencil.stencilReadMask;
                ds.StencilWriteMask = (UINT8)desc.graphics.depthStencil.stencilWriteMask;
                ds.FrontFace =
                    D3DUtil::translateStencilOpDesc(desc.graphics.depthStencil.frontFace);
                ds.BackFace = D3DUtil::translateStencilOpDesc(desc.graphics.depthStencil.backFace);
            }

            psoDesc.PrimitiveTopologyType = D3DUtil::getPrimitiveType(desc.graphics.primitiveType);
        };

        if (m_program->isMeshShaderProgram())
        {
            D3DX12_MESH_SHADER_PIPELINE_STATE_DESC meshDesc = {};
            for (auto& shaderBin : programImpl->m_shaders)
            {
                switch (shaderBin.stage)
                {
                case SLANG_STAGE_FRAGMENT:
                    meshDesc.PS = {shaderBin.code.getBuffer(), SIZE_T(shaderBin.code.getCount())};
                    break;
                case SLANG_STAGE_AMPLIFICATION:
                    meshDesc.AS = {shaderBin.code.getBuffer(), SIZE_T(shaderBin.code.getCount())};
                    break;
                case SLANG_STAGE_MESH:
                    meshDesc.MS = {shaderBin.code.getBuffer(), SIZE_T(shaderBin.code.getCount())};
                    break;
                default:
                    getDebugCallback()->handleMessage(
                        DebugMessageType::Error,
                        DebugMessageSource::Layer,
                        "Unsupported shader stage.");
                    return SLANG_E_NOT_AVAILABLE;
                }
            }
            fillCommonGraphicsState(meshDesc);
            if (m_device->m_pipelineCreationAPIDispatcher)
            {
                SLANG_RETURN_ON_FAIL(
                    m_device->m_pipelineCreationAPIDispatcher->createMeshPipelineState(
                        m_device,
                        programImpl->linkedProgram.get(),
                        &meshDesc,
                        (void**)m_pipelineState.writeRef()));
            }
            else
            {
                CD3DX12_PIPELINE_STATE_STREAM2 meshStateStream{meshDesc};
                D3D12_PIPELINE_STATE_STREAM_DESC streamDesc{
                    sizeof(meshStateStream),
                    &meshStateStream};

                SLANG_RETURN_ON_FAIL(m_device->m_device5->CreatePipelineState(
                    &streamDesc,
                    IID_PPV_ARGS(m_pipelineState.writeRef())));
            }
        }
        else
        {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC graphicsDesc = {};
            for (auto& shaderBin : programImpl->m_shaders)
            {
                switch (shaderBin.stage)
                {
                case SLANG_STAGE_VERTEX:
                    graphicsDesc.VS = {
                        shaderBin.code.getBuffer(),
                        SIZE_T(shaderBin.code.getCount())};
                    break;
                case SLANG_STAGE_FRAGMENT:
                    graphicsDesc.PS = {
                        shaderBin.code.getBuffer(),
                        SIZE_T(shaderBin.code.getCount())};
                    break;
                case SLANG_STAGE_DOMAIN:
                    graphicsDesc.DS = {
                        shaderBin.code.getBuffer(),
                        SIZE_T(shaderBin.code.getCount())};
                    break;
                case SLANG_STAGE_HULL:
                    graphicsDesc.HS = {
                        shaderBin.code.getBuffer(),
                        SIZE_T(shaderBin.code.getCount())};
                    break;
                case SLANG_STAGE_GEOMETRY:
                    graphicsDesc.GS = {
                        shaderBin.code.getBuffer(),
                        SIZE_T(shaderBin.code.getCount())};
                    break;
                default:
                    getDebugCallback()->handleMessage(
                        DebugMessageType::Error,
                        DebugMessageSource::Layer,
                        "Unsupported shader stage.");
                    return SLANG_E_NOT_AVAILABLE;
                }
            }

            if (inputLayoutImpl)
            {
                graphicsDesc.InputLayout = {
                    inputLayoutImpl->m_elements.getBuffer(),
                    UINT(inputLayoutImpl->m_elements.getCount())};
            }

            fillCommonGraphicsState(graphicsDesc);

            if (m_device->m_pipelineCreationAPIDispatcher)
            {
                SLANG_RETURN_ON_FAIL(
                    m_device->m_pipelineCreationAPIDispatcher->createGraphicsPipelineState(
                        m_device,
                        programImpl->linkedProgram.get(),
                        &graphicsDesc,
                        (void**)m_pipelineState.writeRef()));
            }
            else
            {
                SLANG_RETURN_ON_FAIL(m_device->m_device->CreateGraphicsPipelineState(
                    &graphicsDesc,
                    IID_PPV_ARGS(m_pipelineState.writeRef())));
            }
        }
    }
    else
    {

        // Only actually create a D3D12 pipeline state if the pipeline is fully specialized.
        ComPtr<ID3D12PipelineState> pipelineState;
        if (!programImpl->isSpecializable())
        {
            // Describe and create the compute pipeline state object
            D3D12_COMPUTE_PIPELINE_STATE_DESC computeDesc = {};
            computeDesc.pRootSignature =
                desc.compute.d3d12RootSignatureOverride
                    ? static_cast<ID3D12RootSignature*>(desc.compute.d3d12RootSignatureOverride)
                    : programImpl->m_rootObjectLayout->m_rootSignature;
            computeDesc.CS = {
                programImpl->m_shaders[0].code.getBuffer(),
                SIZE_T(programImpl->m_shaders[0].code.getCount())};

#ifdef GFX_NVAPI
            if (m_device->m_nvapi)
            {
                // Also fill the extension structure.
                // Use the same UAV slot index and register space that are declared in the shader.

                // For simplicities sake we just use u0
                NVAPI_D3D12_PSO_SET_SHADER_EXTENSION_SLOT_DESC extensionDesc;
                extensionDesc.baseVersion = NV_PSO_EXTENSION_DESC_VER;
                extensionDesc.version = NV_SET_SHADER_EXTENSION_SLOT_DESC_VER;
                extensionDesc.uavSlot = 0;
                extensionDesc.registerSpace = 0;

                // Put the pointer to the extension into an array - there can be multiple extensions
                // enabled at once.
                const NVAPI_D3D12_PSO_EXTENSION_DESC* extensions[] = {&extensionDesc};

                // Now create the PSO.
                const NvAPI_Status nvapiStatus = NvAPI_D3D12_CreateComputePipelineState(
                    m_device->m_device,
                    &computeDesc,
                    SLANG_COUNT_OF(extensions),
                    extensions,
                    m_pipelineState.writeRef());

                if (nvapiStatus != NVAPI_OK)
                {
                    return SLANG_FAIL;
                }
            }
            else
#endif
            {
                if (m_device->m_pipelineCreationAPIDispatcher)
                {
                    SLANG_RETURN_ON_FAIL(
                        m_device->m_pipelineCreationAPIDispatcher->createComputePipelineState(
                            m_device,
                            programImpl->linkedProgram.get(),
                            &computeDesc,
                            (void**)m_pipelineState.writeRef()));
                }
                else
                {
                    SLANG_RETURN_ON_FAIL(m_device->m_device->CreateComputePipelineState(
                        &computeDesc,
                        IID_PPV_ARGS(m_pipelineState.writeRef())));
                }
            }
        }
    }

    return SLANG_OK;
}

#if SLANG_GFX_HAS_DXR_SUPPORT

RayTracingPipelineStateImpl::RayTracingPipelineStateImpl(DeviceImpl* device)
    : m_device(device)
{
}

void RayTracingPipelineStateImpl::init(const RayTracingPipelineStateDesc& inDesc)
{
    PipelineStateDesc pipelineDesc;
    pipelineDesc.type = PipelineType::RayTracing;
    pipelineDesc.rayTracing.set(inDesc);
    initializeBase(pipelineDesc);
}

Result RayTracingPipelineStateImpl::getNativeHandle(InteropHandle* outHandle)
{
    SLANG_RETURN_ON_FAIL(ensureAPIPipelineStateCreated());
    outHandle->api = InteropHandleAPI::D3D12;
    outHandle->handleValue = reinterpret_cast<uint64_t>(m_stateObject.get());
    return SLANG_OK;
}

Result RayTracingPipelineStateImpl::ensureAPIPipelineStateCreated()
{
    if (m_stateObject)
        return SLANG_OK;

    auto program = static_cast<ShaderProgramImpl*>(m_program.Ptr());
    auto slangGlobalScope = program->linkedProgram;
    auto programLayout = slangGlobalScope->getLayout();

    List<D3D12_STATE_SUBOBJECT> subObjects;
    ChunkedList<D3D12_DXIL_LIBRARY_DESC> dxilLibraries;
    ChunkedList<D3D12_HIT_GROUP_DESC> hitGroups;
    ChunkedList<ComPtr<ISlangBlob>> codeBlobs;
    ChunkedList<D3D12_EXPORT_DESC> exports;
    ChunkedList<const wchar_t*> strPtrs;
    ComPtr<ISlangBlob> diagnostics;
    ChunkedList<OSString> stringPool;
    auto getWStr = [&](const char* name)
    {
        String str = String(name);
        auto wstr = str.toWString();
        return stringPool.add(wstr)->begin();
    };

    D3D12_RAYTRACING_PIPELINE_CONFIG1 pipelineConfig = {};
    pipelineConfig.MaxTraceRecursionDepth = desc.rayTracing.maxRecursion;
    if (desc.rayTracing.flags & RayTracingPipelineFlags::SkipTriangles)
        pipelineConfig.Flags |= D3D12_RAYTRACING_PIPELINE_FLAG_SKIP_TRIANGLES;
    if (desc.rayTracing.flags & RayTracingPipelineFlags::SkipProcedurals)
        pipelineConfig.Flags |= D3D12_RAYTRACING_PIPELINE_FLAG_SKIP_PROCEDURAL_PRIMITIVES;

    D3D12_STATE_SUBOBJECT pipelineConfigSubobject = {};
    pipelineConfigSubobject.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG1;
    pipelineConfigSubobject.pDesc = &pipelineConfig;
    subObjects.add(pipelineConfigSubobject);

    auto compileShader = [&](slang::EntryPointLayout* entryPointInfo,
                             slang::IComponentType* component,
                             SlangInt entryPointIndex)
    {
        ComPtr<ISlangBlob> codeBlob;
        auto compileResult = m_device->getEntryPointCodeFromShaderCache(
            component,
            entryPointIndex,
            0,
            codeBlob.writeRef(),
            diagnostics.writeRef());
        if (diagnostics.get())
        {
            getDebugCallback()->handleMessage(
                compileResult == SLANG_OK ? DebugMessageType::Warning : DebugMessageType::Error,
                DebugMessageSource::Slang,
                (char*)diagnostics->getBufferPointer());
        }
        SLANG_RETURN_ON_FAIL(compileResult);
        codeBlobs.add(codeBlob);
        D3D12_DXIL_LIBRARY_DESC library = {};
        library.DXILLibrary.BytecodeLength = codeBlob->getBufferSize();
        library.DXILLibrary.pShaderBytecode = codeBlob->getBufferPointer();
        library.NumExports = 1;
        D3D12_EXPORT_DESC exportDesc = {};
        exportDesc.Name = getWStr(entryPointInfo->getNameOverride());
        exportDesc.ExportToRename = nullptr;
        exportDesc.Flags = D3D12_EXPORT_FLAG_NONE;
        library.pExports = exports.add(exportDesc);

        D3D12_STATE_SUBOBJECT dxilSubObject = {};
        dxilSubObject.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
        dxilSubObject.pDesc = dxilLibraries.add(library);
        subObjects.add(dxilSubObject);
        return SLANG_OK;
    };
    if (program->linkedEntryPoints.getCount() == 0)
    {
        for (SlangUInt i = 0; i < programLayout->getEntryPointCount(); i++)
        {
            SLANG_RETURN_ON_FAIL(compileShader(
                programLayout->getEntryPointByIndex(i),
                program->linkedProgram,
                (SlangInt)i));
        }
    }
    else
    {
        for (auto& entryPoint : program->linkedEntryPoints)
        {
            SLANG_RETURN_ON_FAIL(
                compileShader(entryPoint->getLayout()->getEntryPointByIndex(0), entryPoint, 0));
        }
    }

    for (Index i = 0; i < desc.rayTracing.hitGroupDescs.getCount(); i++)
    {
        auto& hitGroup = desc.rayTracing.hitGroups[i];
        D3D12_HIT_GROUP_DESC hitGroupDesc = {};
        hitGroupDesc.Type = hitGroup.intersectionEntryPoint.getLength() == 0
                                ? D3D12_HIT_GROUP_TYPE_TRIANGLES
                                : D3D12_HIT_GROUP_TYPE_PROCEDURAL_PRIMITIVE;

        if (hitGroup.anyHitEntryPoint.getLength())
        {
            hitGroupDesc.AnyHitShaderImport = getWStr(hitGroup.anyHitEntryPoint.getBuffer());
        }
        if (hitGroup.closestHitEntryPoint.getLength())
        {
            hitGroupDesc.ClosestHitShaderImport =
                getWStr(hitGroup.closestHitEntryPoint.getBuffer());
        }
        if (hitGroup.intersectionEntryPoint.getLength())
        {
            hitGroupDesc.IntersectionShaderImport =
                getWStr(hitGroup.intersectionEntryPoint.getBuffer());
        }
        hitGroupDesc.HitGroupExport = getWStr(hitGroup.hitGroupName.getBuffer());

        D3D12_STATE_SUBOBJECT hitGroupSubObject = {};
        hitGroupSubObject.Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
        hitGroupSubObject.pDesc = hitGroups.add(hitGroupDesc);
        subObjects.add(hitGroupSubObject);
    }

    D3D12_RAYTRACING_SHADER_CONFIG shaderConfig = {};
    // According to DXR spec, fixed function triangle intersections must use float2 as ray
    // attributes that defines the barycentric coordinates at intersection.
    shaderConfig.MaxAttributeSizeInBytes = (UINT)desc.rayTracing.maxAttributeSizeInBytes;
    shaderConfig.MaxPayloadSizeInBytes = (UINT)desc.rayTracing.maxRayPayloadSize;
    D3D12_STATE_SUBOBJECT shaderConfigSubObject = {};
    shaderConfigSubObject.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
    shaderConfigSubObject.pDesc = &shaderConfig;
    subObjects.add(shaderConfigSubObject);

    D3D12_GLOBAL_ROOT_SIGNATURE globalSignatureDesc = {};
    globalSignatureDesc.pGlobalRootSignature = program->m_rootObjectLayout->m_rootSignature.get();
    D3D12_STATE_SUBOBJECT globalSignatureSubobject = {};
    globalSignatureSubobject.Type = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;
    globalSignatureSubobject.pDesc = &globalSignatureDesc;
    subObjects.add(globalSignatureSubobject);

    if (m_device->m_pipelineCreationAPIDispatcher)
    {
        m_device->m_pipelineCreationAPIDispatcher->beforeCreateRayTracingState(
            m_device,
            slangGlobalScope);
    }

    D3D12_STATE_OBJECT_DESC rtpsoDesc = {};
    rtpsoDesc.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
    rtpsoDesc.NumSubobjects = (UINT)subObjects.getCount();
    rtpsoDesc.pSubobjects = subObjects.getBuffer();
    SLANG_RETURN_ON_FAIL(
        m_device->m_device5->CreateStateObject(&rtpsoDesc, IID_PPV_ARGS(m_stateObject.writeRef())));

    if (m_device->m_pipelineCreationAPIDispatcher)
    {
        m_device->m_pipelineCreationAPIDispatcher->afterCreateRayTracingState(
            m_device,
            slangGlobalScope);
    }
    return SLANG_OK;
}

#endif

} // namespace d3d12
} // namespace gfx
