// vk-pipeline-state.cpp
#include "vk-pipeline-state.h"

#include "vk-device.h"
#include "vk-helper-functions.h"
#include "vk-shader-object-layout.h"
#include "vk-shader-program.h"
#include "vk-vertex-layout.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

PipelineStateImpl::PipelineStateImpl(DeviceImpl* device)
{
    // Only weakly reference `device` at start.
    // We make it a strong reference only when the pipeline state is exposed to the user.
    // Note that `PipelineState`s may also be created via implicit specialization that
    // happens behind the scenes, and the user will not have access to those specialized
    // pipeline states. Only those pipeline states that are returned to the user needs to
    // hold a strong reference to `device`.
    m_device.setWeakReference(device);
}

PipelineStateImpl::~PipelineStateImpl()
{
    if (m_pipeline != VK_NULL_HANDLE)
    {
        m_device->m_api.vkDestroyPipeline(m_device->m_api.m_device, m_pipeline, nullptr);
    }
}

void PipelineStateImpl::establishStrongDeviceReference()
{
    m_device.establishStrongReference();
}

void PipelineStateImpl::comFree()
{
    m_device.breakStrongReference();
}

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

void PipelineStateImpl::init(const RayTracingPipelineStateDesc& inDesc)
{
    PipelineStateDesc pipelineDesc;
    pipelineDesc.type = PipelineType::RayTracing;
    pipelineDesc.rayTracing.set(inDesc);
    initializeBase(pipelineDesc);
}

Result PipelineStateImpl::createVKGraphicsPipelineState()
{
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;

    auto inputLayoutImpl = (InputLayoutImpl*)desc.graphics.inputLayout;

    // VertexBuffer/s
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 0;
    vertexInputInfo.vertexAttributeDescriptionCount = 0;

    if (inputLayoutImpl)
    {
        const auto& srcAttributeDescs = inputLayoutImpl->m_attributeDescs;
        const auto& srcStreamDescs = inputLayoutImpl->m_streamDescs;

        vertexInputInfo.vertexBindingDescriptionCount = (uint32_t)srcStreamDescs.getCount();
        vertexInputInfo.pVertexBindingDescriptions = srcStreamDescs.getBuffer();

        vertexInputInfo.vertexAttributeDescriptionCount = (uint32_t)srcAttributeDescs.getCount();
        vertexInputInfo.pVertexAttributeDescriptions = srcAttributeDescs.getBuffer();
    }

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    // All other forms of primitive toplogies are specified via dynamic state.
    inputAssembly.topology =
        VulkanUtil::translatePrimitiveTypeToListTopology(desc.graphics.primitiveType);
    inputAssembly.primitiveRestartEnable = VK_FALSE; // TODO: Currently unsupported

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    // We are using dynamic viewport and scissor state.
    // Here we specify an arbitrary size, actual viewport will be set at `beginRenderPass`
    // time.
    viewport.width = 16.0f;
    viewport.height = 16.0f;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = {uint32_t(16), uint32_t(16)};

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    auto rasterizerDesc = desc.graphics.rasterizer;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable =
        VK_TRUE; // TODO: Depth clipping and clamping are different between Vk and D3D12
    rasterizer.rasterizerDiscardEnable = VK_FALSE; // TODO: Currently unsupported
    rasterizer.polygonMode = VulkanUtil::translateFillMode(rasterizerDesc.fillMode);
    rasterizer.cullMode = VulkanUtil::translateCullMode(rasterizerDesc.cullMode);
    rasterizer.frontFace = VulkanUtil::translateFrontFaceMode(rasterizerDesc.frontFace);
    rasterizer.depthBiasEnable = (rasterizerDesc.depthBias == 0) ? VK_FALSE : VK_TRUE;
    rasterizer.depthBiasConstantFactor = (float)rasterizerDesc.depthBias;
    rasterizer.depthBiasClamp = rasterizerDesc.depthBiasClamp;
    rasterizer.depthBiasSlopeFactor = rasterizerDesc.slopeScaledDepthBias;
    rasterizer.lineWidth = 1.0f; // TODO: Currently unsupported

    VkPipelineRasterizationConservativeStateCreateInfoEXT conservativeRasterInfo = {};
    conservativeRasterInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT;
    conservativeRasterInfo.conservativeRasterizationMode =
        VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT;
    if (desc.graphics.rasterizer.enableConservativeRasterization)
    {
        rasterizer.pNext = &conservativeRasterInfo;
    }

    auto framebufferLayoutImpl =
        static_cast<FramebufferLayoutImpl*>(desc.graphics.framebufferLayout);
    auto forcedSampleCount = rasterizerDesc.forcedSampleCount;
    auto blendDesc = desc.graphics.blend;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = (forcedSampleCount == 0)
                                             ? framebufferLayoutImpl->m_sampleCount
                                             : VulkanUtil::translateSampleCount(forcedSampleCount);
    multisampling.sampleShadingEnable =
        VK_FALSE; // TODO: Should check if fragment shader needs this
    // TODO: Sample mask is dynamic in D3D12 but PSO state in Vulkan
    multisampling.alphaToCoverageEnable = blendDesc.alphaToCoverageEnable;
    multisampling.alphaToOneEnable = VK_FALSE;

    auto targetCount = GfxCount(
        Math::Min(framebufferLayoutImpl->m_renderTargetCount, (uint32_t)blendDesc.targetCount));
    List<VkPipelineColorBlendAttachmentState> colorBlendTargets;

    // Regardless of whether blending is enabled, Vulkan always applies the color write mask
    // operation, so if there is no blending then we need to add an attachment that defines
    // the color write mask to ensure colors are actually written.
    if (targetCount == 0)
    {
        colorBlendTargets.setCount(1);
        auto& vkBlendDesc = colorBlendTargets[0];
        memset(&vkBlendDesc, 0, sizeof(vkBlendDesc));
        vkBlendDesc.blendEnable = VK_FALSE;
        vkBlendDesc.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        vkBlendDesc.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        vkBlendDesc.colorBlendOp = VK_BLEND_OP_ADD;
        vkBlendDesc.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        vkBlendDesc.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        vkBlendDesc.alphaBlendOp = VK_BLEND_OP_ADD;
        vkBlendDesc.colorWriteMask = (VkColorComponentFlags)RenderTargetWriteMask::EnableAll;
    }
    else
    {
        colorBlendTargets.setCount(targetCount);
        for (GfxIndex i = 0; i < targetCount; ++i)
        {
            auto& gfxBlendDesc = blendDesc.targets[i];
            auto& vkBlendDesc = colorBlendTargets[i];

            vkBlendDesc.blendEnable = gfxBlendDesc.enableBlend;
            vkBlendDesc.srcColorBlendFactor =
                VulkanUtil::translateBlendFactor(gfxBlendDesc.color.srcFactor);
            vkBlendDesc.dstColorBlendFactor =
                VulkanUtil::translateBlendFactor(gfxBlendDesc.color.dstFactor);
            vkBlendDesc.colorBlendOp = VulkanUtil::translateBlendOp(gfxBlendDesc.color.op);
            vkBlendDesc.srcAlphaBlendFactor =
                VulkanUtil::translateBlendFactor(gfxBlendDesc.alpha.srcFactor);
            vkBlendDesc.dstAlphaBlendFactor =
                VulkanUtil::translateBlendFactor(gfxBlendDesc.alpha.dstFactor);
            vkBlendDesc.alphaBlendOp = VulkanUtil::translateBlendOp(gfxBlendDesc.alpha.op);
            vkBlendDesc.colorWriteMask = (VkColorComponentFlags)gfxBlendDesc.writeMask;
        }
    }

    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE; // TODO: D3D12 has per attachment logic op (and
                                            // both have way more than one op)
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = (uint32_t)colorBlendTargets.getCount();
    colorBlending.pAttachments = colorBlendTargets.getBuffer();
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    Array<VkDynamicState, 8> dynamicStates;
    dynamicStates.add(VK_DYNAMIC_STATE_VIEWPORT);
    dynamicStates.add(VK_DYNAMIC_STATE_SCISSOR);
    dynamicStates.add(VK_DYNAMIC_STATE_STENCIL_REFERENCE);
    dynamicStates.add(VK_DYNAMIC_STATE_BLEND_CONSTANTS);
    // It's not valid to specify VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY_EXT when
    // the pipeline contains a mesh shader.
    if (!m_program->isMeshShaderProgram() &&
        m_device->m_api.m_extendedFeatures.extendedDynamicStateFeatures.extendedDynamicState)

    {
        dynamicStates.add(VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY_EXT);
    }
    VkPipelineDynamicStateCreateInfo dynamicStateInfo = {};
    dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateInfo.dynamicStateCount = (uint32_t)dynamicStates.getCount();
    dynamicStateInfo.pDynamicStates = dynamicStates.getBuffer();

    VkPipelineDepthStencilStateCreateInfo depthStencilStateInfo = {};
    depthStencilStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencilStateInfo.depthTestEnable = desc.graphics.depthStencil.depthTestEnable ? 1 : 0;
    depthStencilStateInfo.back =
        VulkanUtil::translateStencilState(desc.graphics.depthStencil.backFace);
    depthStencilStateInfo.front =
        VulkanUtil::translateStencilState(desc.graphics.depthStencil.frontFace);
    depthStencilStateInfo.back.compareMask = desc.graphics.depthStencil.stencilReadMask;
    depthStencilStateInfo.back.writeMask = desc.graphics.depthStencil.stencilWriteMask;
    depthStencilStateInfo.front.compareMask = desc.graphics.depthStencil.stencilReadMask;
    depthStencilStateInfo.front.writeMask = desc.graphics.depthStencil.stencilWriteMask;
    depthStencilStateInfo.depthBoundsTestEnable = 0; // TODO: Currently unsupported
    depthStencilStateInfo.depthCompareOp =
        VulkanUtil::translateComparisonFunc(desc.graphics.depthStencil.depthFunc);
    depthStencilStateInfo.depthWriteEnable = desc.graphics.depthStencil.depthWriteEnable ? 1 : 0;
    depthStencilStateInfo.stencilTestEnable = desc.graphics.depthStencil.stencilEnable ? 1 : 0;

    VkGraphicsPipelineCreateInfo pipelineInfo = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};

    auto programImpl = static_cast<ShaderProgramImpl*>(m_program.Ptr());
    if (programImpl->m_stageCreateInfos.getCount() == 0)
    {
        SLANG_RETURN_ON_FAIL(programImpl->compileShaders(m_device));
    }

    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = (uint32_t)programImpl->m_stageCreateInfos.getCount();
    pipelineInfo.pStages = programImpl->m_stageCreateInfos.getBuffer();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDepthStencilState = &depthStencilStateInfo;
    pipelineInfo.layout = programImpl->m_rootObjectLayout->m_pipelineLayout;
    pipelineInfo.renderPass = framebufferLayoutImpl->m_renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.pDynamicState = &dynamicStateInfo;

    if (m_device->m_pipelineCreationAPIDispatcher)
    {
        SLANG_RETURN_ON_FAIL(m_device->m_pipelineCreationAPIDispatcher->createGraphicsPipelineState(
            m_device,
            programImpl->linkedProgram.get(),
            &pipelineInfo,
            (void**)&m_pipeline));
    }
    else
    {
        SLANG_VK_RETURN_ON_FAIL(m_device->m_api.vkCreateGraphicsPipelines(
            m_device->m_device,
            pipelineCache,
            1,
            &pipelineInfo,
            nullptr,
            &m_pipeline));
    }

    return SLANG_OK;
}

Result PipelineStateImpl::createVKComputePipelineState()
{
    auto programImpl = static_cast<ShaderProgramImpl*>(m_program.Ptr());
    if (programImpl->m_stageCreateInfos.getCount() == 0)
    {
        SLANG_RETURN_ON_FAIL(programImpl->compileShaders(m_device));
    }

    VkComputePipelineCreateInfo computePipelineInfo = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    computePipelineInfo.stage = programImpl->m_stageCreateInfos[0];
    computePipelineInfo.layout = programImpl->m_rootObjectLayout->m_pipelineLayout;

    if (m_device->m_pipelineCreationAPIDispatcher)
    {
        SLANG_RETURN_ON_FAIL(m_device->m_pipelineCreationAPIDispatcher->createComputePipelineState(
            m_device,
            programImpl->linkedProgram.get(),
            &computePipelineInfo,
            (void**)&m_pipeline));
    }
    else
    {
        VkPipelineCache pipelineCache = VK_NULL_HANDLE;
        SLANG_VK_RETURN_ON_FAIL(m_device->m_api.vkCreateComputePipelines(
            m_device->m_device,
            pipelineCache,
            1,
            &computePipelineInfo,
            nullptr,
            &m_pipeline));
    }
    return SLANG_OK;
}

Result PipelineStateImpl::ensureAPIPipelineStateCreated()
{
    if (m_pipeline)
        return SLANG_OK;

    switch (desc.type)
    {
    case PipelineType::Compute:
        return createVKComputePipelineState();
    case PipelineType::Graphics:
        return createVKGraphicsPipelineState();
    default:
        SLANG_UNREACHABLE("Unknown pipeline type.");
        return SLANG_FAIL;
    }
}
SLANG_NO_THROW Result SLANG_MCALL PipelineStateImpl::getNativeHandle(InteropHandle* outHandle)
{
    SLANG_RETURN_ON_FAIL(ensureAPIPipelineStateCreated());
    outHandle->api = InteropHandleAPI::Vulkan;
    outHandle->handleValue = 0;
    memcpy(&outHandle->handleValue, &m_pipeline, sizeof(m_pipeline));
    return SLANG_OK;
}

RayTracingPipelineStateImpl::RayTracingPipelineStateImpl(DeviceImpl* device)
    : PipelineStateImpl(device)
{
}
uint32_t RayTracingPipelineStateImpl::findEntryPointIndexByName(
    const Dictionary<String, Index>& entryPointNameToIndex,
    const char* name)
{
    if (!name)
        return VK_SHADER_UNUSED_KHR;

    auto indexPtr = entryPointNameToIndex.tryGetValue(String(name));
    if (indexPtr)
        return (uint32_t)*indexPtr;
    // TODO: Error reporting?
    return VK_SHADER_UNUSED_KHR;
}
Result RayTracingPipelineStateImpl::createVKRayTracingPipelineState()
{
    auto programImpl = static_cast<ShaderProgramImpl*>(m_program.Ptr());
    if (programImpl->m_stageCreateInfos.getCount() == 0)
    {
        SLANG_RETURN_ON_FAIL(programImpl->compileShaders(m_device));
    }

    VkRayTracingPipelineCreateInfoKHR raytracingPipelineInfo = {
        VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    raytracingPipelineInfo.pNext = nullptr;
    raytracingPipelineInfo.flags = translateRayTracingPipelineFlags(desc.rayTracing.flags);

    raytracingPipelineInfo.stageCount = (uint32_t)programImpl->m_stageCreateInfos.getCount();
    raytracingPipelineInfo.pStages = programImpl->m_stageCreateInfos.getBuffer();

    // Build Dictionary from entry point name to entry point index (stageCreateInfos index)
    // for all hit shaders - findShaderIndexByName
    Dictionary<String, Index> entryPointNameToIndex;

    List<VkRayTracingShaderGroupCreateInfoKHR> shaderGroupInfos;
    for (uint32_t i = 0; i < raytracingPipelineInfo.stageCount; ++i)
    {
        auto stageCreateInfo = programImpl->m_stageCreateInfos[i];
        auto entryPointName = programImpl->m_entryPointNames[i];
        entryPointNameToIndex.add(entryPointName, i);
        if (stageCreateInfo.stage &
            (VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
             VK_SHADER_STAGE_INTERSECTION_BIT_KHR))
            continue;

        VkRayTracingShaderGroupCreateInfoKHR shaderGroupInfo = {
            VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
        shaderGroupInfo.pNext = nullptr;
        shaderGroupInfo.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        shaderGroupInfo.generalShader = i;
        shaderGroupInfo.closestHitShader = VK_SHADER_UNUSED_KHR;
        shaderGroupInfo.anyHitShader = VK_SHADER_UNUSED_KHR;
        shaderGroupInfo.intersectionShader = VK_SHADER_UNUSED_KHR;
        shaderGroupInfo.pShaderGroupCaptureReplayHandle = nullptr;

        // For groups with a single entry point, the group name is the entry point name.
        auto shaderGroupName = entryPointName;
        auto shaderGroupIndex = shaderGroupInfos.getCount();
        shaderGroupInfos.add(shaderGroupInfo);
        shaderGroupNameToIndex.add(shaderGroupName, shaderGroupIndex);
    }

    for (int32_t i = 0; i < desc.rayTracing.hitGroupDescs.getCount(); ++i)
    {
        VkRayTracingShaderGroupCreateInfoKHR shaderGroupInfo = {
            VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
        auto& groupDesc = desc.rayTracing.hitGroupDescs[i];

        shaderGroupInfo.pNext = nullptr;
        shaderGroupInfo.type = (groupDesc.intersectionEntryPoint)
                                   ? VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR
                                   : VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        shaderGroupInfo.generalShader = VK_SHADER_UNUSED_KHR;
        shaderGroupInfo.closestHitShader =
            findEntryPointIndexByName(entryPointNameToIndex, groupDesc.closestHitEntryPoint);
        shaderGroupInfo.anyHitShader =
            findEntryPointIndexByName(entryPointNameToIndex, groupDesc.anyHitEntryPoint);
        shaderGroupInfo.intersectionShader =
            findEntryPointIndexByName(entryPointNameToIndex, groupDesc.intersectionEntryPoint);
        shaderGroupInfo.pShaderGroupCaptureReplayHandle = nullptr;

        auto shaderGroupIndex = shaderGroupInfos.getCount();
        shaderGroupInfos.add(shaderGroupInfo);
        shaderGroupNameToIndex.add(String(groupDesc.hitGroupName), shaderGroupIndex);
    }

    raytracingPipelineInfo.groupCount = (uint32_t)shaderGroupInfos.getCount();
    raytracingPipelineInfo.pGroups = shaderGroupInfos.getBuffer();

    raytracingPipelineInfo.maxPipelineRayRecursionDepth = (uint32_t)desc.rayTracing.maxRecursion;

    raytracingPipelineInfo.pLibraryInfo = nullptr;
    raytracingPipelineInfo.pLibraryInterface = nullptr;

    raytracingPipelineInfo.pDynamicState = nullptr;

    raytracingPipelineInfo.layout = programImpl->m_rootObjectLayout->m_pipelineLayout;
    raytracingPipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    raytracingPipelineInfo.basePipelineIndex = 0;

    if (m_device->m_pipelineCreationAPIDispatcher)
    {
        m_device->m_pipelineCreationAPIDispatcher->beforeCreateRayTracingState(
            m_device,
            programImpl->linkedProgram.get());
    }

    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    SLANG_VK_RETURN_ON_FAIL(m_device->m_api.vkCreateRayTracingPipelinesKHR(
        m_device->m_device,
        VK_NULL_HANDLE,
        pipelineCache,
        1,
        &raytracingPipelineInfo,
        nullptr,
        &m_pipeline));
    shaderGroupCount = shaderGroupInfos.getCount();

    if (m_device->m_pipelineCreationAPIDispatcher)
    {
        m_device->m_pipelineCreationAPIDispatcher->afterCreateRayTracingState(
            m_device,
            programImpl->linkedProgram.get());
    }
    return SLANG_OK;
}
Result RayTracingPipelineStateImpl::ensureAPIPipelineStateCreated()
{
    if (m_pipeline)
        return SLANG_OK;

    switch (desc.type)
    {
    case PipelineType::RayTracing:
        return createVKRayTracingPipelineState();
    default:
        SLANG_UNREACHABLE("Unknown pipeline type.");
        return SLANG_FAIL;
    }
}
Result RayTracingPipelineStateImpl::getNativeHandle(InteropHandle* outHandle)
{
    SLANG_RETURN_ON_FAIL(ensureAPIPipelineStateCreated());
    outHandle->api = InteropHandleAPI::Vulkan;
    outHandle->handleValue = 0;
    memcpy(&outHandle->handleValue, &m_pipeline, sizeof(m_pipeline));
    return SLANG_OK;
}

} // namespace vk
} // namespace gfx
