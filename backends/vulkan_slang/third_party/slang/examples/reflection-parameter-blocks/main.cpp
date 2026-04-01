// main.cpp

// Using Parameter Blocks With Reflection
// ======================================
//
// This example program is a companion to the article
// Using Slang Parameter Blocks, and specifically
// the section of that article called Using Parameter
// Blocks With Reflection.
//
// Where possible, the code is presented in the
// same order as the code in the article, so that the
// two can be read in parallel. When code relates to
// a sub-section of the article, a comment will be used
// to reference the relevant section.
//
// Boilerplate
// ===========
//
// As is typical for our example programs, this one starts
// with a certain amount of boilerplate that isn't especially
// interesting to discuss.

#include "slang-com-ptr.h"
#include "slang.h"
typedef SlangResult Result;

#include "core/slang-basic.h"
#include "examples/example-base/example-base.h"
using Slang::ComPtr;
using Slang::String;
using Slang::List;

// The example code currently only supports Vulkan, but the
// code is factored with the intention that it could be extended
// to support D3D12 as well.

#define ENABLE_VULKAN 1
#define ENABLE_D3D12 0

#if ENABLE_VULKAN
#include "vulkan-api.h"
#endif

static const ExampleResources resourceBase("reflection-parameter-blocks");
static const char* kSourceFileName = "shader.slang";

struct PipelineLayoutReflectionContext
{
    gfx::IDevice* _gfxDevice = nullptr;
    slang::ISession* _slangSession = nullptr;
    slang::ProgramLayout* _slangProgramLayout = nullptr;
    slang::IBlob* _slangCompiledProgramBlob = nullptr;
};

struct PipelineLayoutReflectionContext_Vulkan : PipelineLayoutReflectionContext
{
    // What Goes Into a Pipeline Layout?
    // =================================

    struct PipelineLayoutBuilder
    {
        std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
        std::vector<VkPushConstantRange> pushConstantRanges;
    };

    // Unlike how things are presented in the document, we do not
    // nest most of the functions under the `*Builder` types, in
    // order to allow for more flexibility in the order of
    // presentation. For example, instead of a
    // `PipelineLayoutBuilder::finishBuilding()` method, we instead
    // have a `finishBuildingPipelineLayout` function:

    Result finishBuildingPipelineLayout(
        PipelineLayoutBuilder& builder,
        VkPipelineLayout* outPipelineLayout)
    {
        filterOutEmptyDescriptorSets(builder);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};

        pipelineLayoutInfo.setLayoutCount = builder.descriptorSetLayouts.size();
        pipelineLayoutInfo.pSetLayouts = builder.descriptorSetLayouts.data();

        pipelineLayoutInfo.pushConstantRangeCount = builder.pushConstantRanges.size();
        pipelineLayoutInfo.pPushConstantRanges = builder.pushConstantRanges.data();

        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        vkAPI.vkCreatePipelineLayout(vkAPI.device, &pipelineLayoutInfo, nullptr, &pipelineLayout);

        *outPipelineLayout = pipelineLayout;
        return SLANG_OK;
    }

    // What Goes Into a Descriptor Set Layout?
    // =======================================

    struct DescriptorSetLayoutBuilder
    {
        std::vector<VkDescriptorSetLayoutBinding> descriptorRanges;

        int setIndex = -1;
    };


    // Once we are done traversing the contents of a parameter
    // block to collect bindings into a `DescriptorSetLayoutBuilder`,
    // it is a simple matter to create a descriptor set layout using
    // the Vulkan API, and to install it into the `setLayouts` array
    // at the index that was reserved.
    //
    void finishBuildingDescriptorSetLayout(
        PipelineLayoutBuilder& pipelineLayoutBuilder,
        DescriptorSetLayoutBuilder& descriptorSetLayoutBuilder)
    {
        if (descriptorSetLayoutBuilder.descriptorRanges.empty())
            return;

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};

        descriptorSetLayoutInfo.bindingCount = descriptorSetLayoutBuilder.descriptorRanges.size();
        descriptorSetLayoutInfo.pBindings = descriptorSetLayoutBuilder.descriptorRanges.data();

        VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
        vkAPI.vkCreateDescriptorSetLayout(
            vkAPI.device,
            &descriptorSetLayoutInfo,
            nullptr,
            &descriptorSetLayout);

        pipelineLayoutBuilder.descriptorSetLayouts[descriptorSetLayoutBuilder.setIndex] =
            descriptorSetLayout;
    }

    // Parameter Blocks
    // ================

    void addDescriptorSetForParameterBlock(
        PipelineLayoutBuilder& pipelineLayoutBuilder,
        slang::TypeLayoutReflection* parameterBlockTypeLayout)
    {
        DescriptorSetLayoutBuilder descriptorSetLayoutBuilder;
        startBuildingDescriptorSetLayout(pipelineLayoutBuilder, descriptorSetLayoutBuilder);

        addRangesForParameterBlockElement(
            pipelineLayoutBuilder,
            descriptorSetLayoutBuilder,
            parameterBlockTypeLayout->getElementTypeLayout());

        finishBuildingDescriptorSetLayout(pipelineLayoutBuilder, descriptorSetLayoutBuilder);
    }

    // Automatically-Introduced Uniform Buffer
    // ---------------------------------------

    void addRangesForParameterBlockElement(
        PipelineLayoutBuilder& pipelineLayoutBuilder,
        DescriptorSetLayoutBuilder& descriptorSetLayoutBuilder,
        slang::TypeLayoutReflection* elementTypeLayout)
    {
        if (elementTypeLayout->getSize() > 0)
        {
            addAutomaticallyIntroducedUniformBuffer(descriptorSetLayoutBuilder);
        }

        // Once we have accounted for the possibility of an implicitly-introduced
        // constant buffer, we can move on and add bindings based on whatever
        // non-ordinary data (textures, buffers, etc.) is in the element type:
        //
        addRanges(pipelineLayoutBuilder, descriptorSetLayoutBuilder, elementTypeLayout);
    }

    void addAutomaticallyIntroducedUniformBuffer(
        DescriptorSetLayoutBuilder& descriptorSetLayoutBuilder)
    {
        auto vulkanBindingIndex = descriptorSetLayoutBuilder.descriptorRanges.size();

        VkDescriptorSetLayoutBinding binding = {};
        binding.stageFlags = VK_SHADER_STAGE_ALL;
        binding.binding = vulkanBindingIndex;
        binding.descriptorCount = 1;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

        descriptorSetLayoutBuilder.descriptorRanges.push_back(binding);
    }

    // Ordering of Nested Parameter Blocks
    // -----------------------------------

    void startBuildingDescriptorSetLayout(
        PipelineLayoutBuilder& pipelineLayoutBuilder,
        DescriptorSetLayoutBuilder& descriptorSetLayoutBuilder)
    {
        descriptorSetLayoutBuilder.setIndex = pipelineLayoutBuilder.descriptorSetLayouts.size();
        pipelineLayoutBuilder.descriptorSetLayouts.push_back(VK_NULL_HANDLE);
    }

    // Empty Ranges
    // ------------

    void filterOutEmptyDescriptorSets(PipelineLayoutBuilder& builder)
    {
        std::vector<VkDescriptorSetLayout> filteredDescriptorSetLayouts;
        for (auto descriptorSetLayout : builder.descriptorSetLayouts)
        {
            if (!descriptorSetLayout)
                continue;
            filteredDescriptorSetLayouts.push_back(descriptorSetLayout);
        }
        std::swap(builder.descriptorSetLayouts, filteredDescriptorSetLayouts);
    }

    // Descritpor Ranges
    // =================

    void addDescriptorRanges(
        DescriptorSetLayoutBuilder& descriptorSetLayoutBuilder,
        slang::TypeLayoutReflection* typeLayout)
    {
        int relativeSetIndex = 0;
        int rangeCount = typeLayout->getDescriptorSetDescriptorRangeCount(relativeSetIndex);

        for (int rangeIndex = 0; rangeIndex < rangeCount; ++rangeIndex)
        {
            addDescriptorRange(
                descriptorSetLayoutBuilder,
                typeLayout,
                relativeSetIndex,
                rangeIndex);
        }
    }

    void addDescriptorRange(
        DescriptorSetLayoutBuilder& descriptorSetLayoutBuilder,
        slang::TypeLayoutReflection* typeLayout,
        int relativeSetIndex,
        int rangeIndex)
    {
        slang::BindingType bindingType =
            typeLayout->getDescriptorSetDescriptorRangeType(relativeSetIndex, rangeIndex);
        auto descriptorCount = typeLayout->getDescriptorSetDescriptorRangeDescriptorCount(
            relativeSetIndex,
            rangeIndex);

        // Some Ranges Need to Be Skipped
        // ------------------------------
        //
        switch (bindingType)
        {
        default:
            break;

        case slang::BindingType::PushConstant:
            return;
        }

        auto bindingIndex = descriptorSetLayoutBuilder.descriptorRanges.size();

        VkDescriptorSetLayoutBinding vulkanBindingRange = {};
        vulkanBindingRange.binding = bindingIndex;
        vulkanBindingRange.descriptorCount = descriptorCount;
        vulkanBindingRange.stageFlags = _currentStageFlags;
        vulkanBindingRange.descriptorType = mapSlangBindingTypeToVulkanDescriptorType(bindingType);

        descriptorSetLayoutBuilder.descriptorRanges.push_back(vulkanBindingRange);
    }

    VkDescriptorType mapSlangBindingTypeToVulkanDescriptorType(slang::BindingType bindingType)
    {
        switch (bindingType)
        {
#define CASE(FROM, TO)             \
    case slang::BindingType::FROM: \
        return VK_DESCRIPTOR_TYPE_##TO

            CASE(Sampler, SAMPLER);
            CASE(CombinedTextureSampler, COMBINED_IMAGE_SAMPLER);
            CASE(Texture, SAMPLED_IMAGE);
            CASE(MutableTexture, STORAGE_IMAGE);
            CASE(TypedBuffer, UNIFORM_TEXEL_BUFFER);
            CASE(MutableTypedBuffer, STORAGE_TEXEL_BUFFER);
            CASE(ConstantBuffer, UNIFORM_BUFFER);
            CASE(RawBuffer, STORAGE_BUFFER);
            CASE(MutableRawBuffer, STORAGE_BUFFER);
            CASE(InputRenderTarget, INPUT_ATTACHMENT);
            CASE(InlineUniformData, INLINE_UNIFORM_BLOCK);
            CASE(RayTracingAccelerationStructure, ACCELERATION_STRUCTURE_KHR);

#undef CASE

        default:
            return VkDescriptorType(-1);
        }
    }

    // Sub-Object Ranges
    // =================

    void addRanges(
        PipelineLayoutBuilder& pipelineLayoutBuilder,
        DescriptorSetLayoutBuilder& descriptorSetLayoutBuilder,
        slang::TypeLayoutReflection* typeLayout)
    {
        addDescriptorRanges(descriptorSetLayoutBuilder, typeLayout);
        addSubObjectRanges(pipelineLayoutBuilder, typeLayout);
    }

    void addSubObjectRanges(
        PipelineLayoutBuilder& pipelineLayoutBuilder,
        slang::TypeLayoutReflection* typeLayout)
    {
        int subObjectRangeCount = typeLayout->getSubObjectRangeCount();
        for (int subObjectRangeIndex = 0; subObjectRangeIndex < subObjectRangeCount;
             ++subObjectRangeIndex)
        {
            addSubObjectRange(pipelineLayoutBuilder, typeLayout, subObjectRangeIndex);
        }
    }

    void addSubObjectRange(
        PipelineLayoutBuilder& pipelineLayoutBuilder,
        slang::TypeLayoutReflection* typeLayout,
        int subObjectRangeIndex)
    {
        auto bindingRangeIndex =
            typeLayout->getSubObjectRangeBindingRangeIndex(subObjectRangeIndex);
        auto bindingType = typeLayout->getBindingRangeType(bindingRangeIndex);
        switch (bindingType)
        {
        default:
            return;

            // Nested Parameter Blocks
            // -----------------------

        case slang::BindingType::ParameterBlock:
            {
                auto parameterBlockTypeLayout =
                    typeLayout->getBindingRangeLeafTypeLayout(bindingRangeIndex);
                addDescriptorSetForParameterBlock(pipelineLayoutBuilder, parameterBlockTypeLayout);
            }
            break;

            // Push-Constant Ranges
            // --------------------

        case slang::BindingType::PushConstant:
            {
                auto constantBufferTypeLayout =
                    typeLayout->getBindingRangeLeafTypeLayout(bindingRangeIndex);
                addPushConstantRangeForConstantBuffer(
                    pipelineLayoutBuilder,
                    constantBufferTypeLayout);
            }
            break;
        }
    }

    void addPushConstantRangeForConstantBuffer(
        PipelineLayoutBuilder& pipelineLayoutBuilder,
        slang::TypeLayoutReflection* pushConstantBufferTypeLayout)
    {
        auto elementTypeLayout = pushConstantBufferTypeLayout->getElementTypeLayout();
        auto elementSize = elementTypeLayout->getSize();

        if (elementSize == 0)
            return;

        VkPushConstantRange pushConstantRange = {};
        pushConstantRange.stageFlags = _currentStageFlags;
        pushConstantRange.offset = 0;
        pushConstantRange.size = elementSize;

        pipelineLayoutBuilder.pushConstantRanges.push_back(pushConstantRange);
    }

    // Creating a Pipeline Layout for a Program
    // ========================================

    Result createPipelineLayout(
        slang::ProgramLayout* programLayout,
        VkPipelineLayout* outPipelineLayout)
    {
        PipelineLayoutBuilder pipelineLayoutBuilder;

        DescriptorSetLayoutBuilder defaultDescriptorSetLayoutBuilder;
        startBuildingDescriptorSetLayout(pipelineLayoutBuilder, defaultDescriptorSetLayoutBuilder);

        addGlobalScopeParameters(
            pipelineLayoutBuilder,
            defaultDescriptorSetLayoutBuilder,
            programLayout);

        addEntryPointParameters(
            pipelineLayoutBuilder,
            defaultDescriptorSetLayoutBuilder,
            programLayout);

        finishBuildingDescriptorSetLayout(pipelineLayoutBuilder, defaultDescriptorSetLayoutBuilder);
        finishBuildingPipelineLayout(pipelineLayoutBuilder, outPipelineLayout);

        return SLANG_OK;
    }

    // Global Scope
    // ------------

    void addGlobalScopeParameters(
        PipelineLayoutBuilder& pipelineLayoutBuilder,
        DescriptorSetLayoutBuilder& descriptorSetLayoutBuilder,
        slang::ProgramLayout* programLayout)
    {
        _currentStageFlags = VK_SHADER_STAGE_ALL;
        addRangesForParameterBlockElement(
            pipelineLayoutBuilder,
            descriptorSetLayoutBuilder,
            programLayout->getGlobalParamsTypeLayout());
    }

    // Entry Points
    // ------------

    void addEntryPointParameters(
        PipelineLayoutBuilder& pipelineLayoutBuilder,
        DescriptorSetLayoutBuilder& descriptorSetLayoutBuilder,
        slang::ProgramLayout* programLayout)
    {
        int entryPointCount = _slangProgramLayout->getEntryPointCount();
        for (int i = 0; i < entryPointCount; ++i)
        {
            auto entryPointLayout = _slangProgramLayout->getEntryPointByIndex(i);
            addEntryPointParameters(
                pipelineLayoutBuilder,
                descriptorSetLayoutBuilder,
                entryPointLayout);
        }
    }

    void addEntryPointParameters(
        PipelineLayoutBuilder& pipelineLayoutBuilder,
        DescriptorSetLayoutBuilder& descriptorSetLayoutBuilder,
        slang::EntryPointLayout* entryPointLayout)
    {
        _currentStageFlags = getShaderStageFlags(entryPointLayout->getStage());
        addRangesForParameterBlockElement(
            pipelineLayoutBuilder,
            descriptorSetLayoutBuilder,
            entryPointLayout->getTypeLayout());
    }

    VkShaderStageFlags _currentStageFlags = VK_SHADER_STAGE_ALL;
    VkShaderStageFlags getShaderStageFlags(SlangStage stage)
    {
        switch (stage)
        {
#define CASE(FROM, TO)       \
    case SLANG_STAGE_##FROM: \
        return VK_SHADER_STAGE_##TO

            CASE(VERTEX, VERTEX_BIT);
            CASE(HULL, TESSELLATION_CONTROL_BIT);
            CASE(DOMAIN, TESSELLATION_EVALUATION_BIT);
            CASE(GEOMETRY, GEOMETRY_BIT);
            CASE(FRAGMENT, FRAGMENT_BIT);
            CASE(COMPUTE, COMPUTE_BIT);
            CASE(RAY_GENERATION, RAYGEN_BIT_KHR);
            CASE(ANY_HIT, ANY_HIT_BIT_KHR);
            CASE(CLOSEST_HIT, CLOSEST_HIT_BIT_KHR);
            CASE(MISS, MISS_BIT_KHR);
            CASE(INTERSECTION, INTERSECTION_BIT_KHR);
            CASE(CALLABLE, CALLABLE_BIT_KHR);
            CASE(MESH, MESH_BIT_EXT);
            CASE(AMPLIFICATION, TASK_BIT_EXT);

#undef CASE
        default:
            return VK_SHADER_STAGE_ALL;
        }
    }

    // Validation
    // ==========
    //
    // The published article covers how to create a pipeline layout
    // using the reflection API, but for the purposes of an example
    // program, we should make sure that we validate that the layout
    // that results from that code is *actually* compatible with the
    // shader program.
    //
    // The remaining operations inside this type provide the support
    // code to create and validate a pipeline layout based on a
    // particular compiled compute program. Mismatches between the
    // pipeline layout and the program should be diagnosed by the
    // Vulkan validation layer when we attempt to create a pipeline
    // that uses the two together.

    Result validatePipelineLayout(VkPipelineLayout pipelineLayout)
    {
        VkShaderModuleCreateInfo shaderModuleInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shaderModuleInfo.pCode = (uint32_t const*)_slangCompiledProgramBlob->getBufferPointer();
        shaderModuleInfo.codeSize = _slangCompiledProgramBlob->getBufferSize();

        VkShaderModule vkShaderModule;
        vkAPI.vkCreateShaderModule(vkAPI.device, &shaderModuleInfo, nullptr, &vkShaderModule);

        VkComputePipelineCreateInfo pipelineInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.module = vkShaderModule;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;

        VkPipeline pipeline;
        vkAPI.vkCreateComputePipelines(
            vkAPI.device,
            VK_NULL_HANDLE,
            1,
            &pipelineInfo,
            nullptr,
            &pipeline);

        vkAPI.vkDestroyPipeline(vkAPI.device, pipeline, nullptr);

        return SLANG_OK;
    }

    Result createAndValidatePipelineLayout()
    {
        // Here we do a little bit of complicated interaction with
        // the `gfx` library to allow us to call raw Vulkan API
        // functions on the same device that `gfx` kindly set up
        // for us.
        //
        gfx::IDevice::InteropHandles handles;
        SLANG_RETURN_ON_FAIL(_gfxDevice->getNativeDeviceHandles(&handles));

        vkAPI.instance = (VkInstance)handles.handles[0].handleValue;
        vkAPI.physicalDevice = (VkPhysicalDevice)handles.handles[1].handleValue;
        vkAPI.device = (VkDevice)handles.handles[2].handleValue;

        vkAPI.initGlobalProcs();
        vkAPI.initInstanceProcs();
        vkAPI.initDeviceProcs();

        // Once the setup is dealt with, we can go ahead and
        // create the pipeline layout, before validating that
        // it can be used together with the compiled SPIR-V
        // binary for the program.
        //
        VkPipelineLayout pipelineLayout;
        SLANG_RETURN_ON_FAIL(createPipelineLayout(_slangProgramLayout, &pipelineLayout));
        SLANG_RETURN_ON_FAIL(validatePipelineLayout(pipelineLayout));

        vkAPI.vkDestroyPipelineLayout(vkAPI.device, pipelineLayout, nullptr);

        return SLANG_OK;
    }

    VulkanAPI vkAPI;
};

// More Boilerplate
// ================
//
// The logic below this point is just about setting up the necessary state
// in the example application for the code above to be run on a simple
// shader. Nothing here is especially relevant to the task of creating
// a pipeline layout from Slang reflection information.

struct ReflectionParameterBlocksExampleApp : public TestBase
{
    Result execute(int argc, char** argv)
    {
        parseOption(argc, argv);

        // We start by initializing the `gfx` system, so that
        // it can handle most of the details of getting a
        // Vulkan device up and running.

#ifdef _DEBUG
        gfx::gfxEnableDebugLayer();
#endif
        gfx::IDevice::Desc deviceDesc = {};
        deviceDesc.deviceType = gfx::DeviceType::Vulkan;

        ComPtr<gfx::IDevice> gfxDevice;
        SLANG_RETURN_ON_FAIL(gfxCreateDevice(&deviceDesc, gfxDevice.writeRef()));

        // The `gfx` library also creates a Slang session as
        // part of its startup, so we will use the session
        // it already created for the compilation in
        // this example.
        //
        auto slangSession = gfxDevice->getSlangSession();

        // Next we go through the fairly routine steps needed to
        // compile a Slang program from source.
        //
        ComPtr<slang::IBlob> diagnostics;
        Result result = SLANG_OK;

        // We load the source file as a module of Slang code.
        //
        String sourceFilePath = resourceBase.resolveResource(kSourceFileName);
        ComPtr<slang::IModule> module;
        module = slangSession->loadModule(sourceFilePath.getBuffer(), diagnostics.writeRef());
        diagnoseIfNeeded(diagnostics);
        if (!module)
            return SLANG_FAIL;

        // Next we will collect all of the entry points defined in the module,
        // to form the list of components we want to link together to form
        // a program.
        //
        List<ComPtr<slang::IComponentType>> componentsToLink;
        int definedEntryPointCount = module->getDefinedEntryPointCount();
        for (int i = 0; i < definedEntryPointCount; i++)
        {
            ComPtr<slang::IEntryPoint> entryPoint;
            SLANG_RETURN_ON_FAIL(module->getDefinedEntryPoint(i, entryPoint.writeRef()));
            componentsToLink.add(ComPtr<slang::IComponentType>(entryPoint.get()));
        }

        // Once we've collected the list of entry points we want to compose,
        // we use the Slang compilation API to compose them.
        //
        ComPtr<slang::IComponentType> composed;
        result = slangSession->createCompositeComponentType(
            (slang::IComponentType**)componentsToLink.getBuffer(),
            componentsToLink.getCount(),
            composed.writeRef(),
            diagnostics.writeRef());
        diagnoseIfNeeded(diagnostics);
        SLANG_RETURN_ON_FAIL(result);

        // As the final compilation step, we will use the compilation API
        // to link the composed code. Think of this as equivalent to
        // applying the linker to a bunch of `.o` and/or `.a` files to
        // produce a binary (executable or shared library).
        //
        ComPtr<slang::IComponentType> program;
        result = composed->link(program.writeRef(), diagnostics.writeRef());
        diagnoseIfNeeded(diagnostics);
        SLANG_RETURN_ON_FAIL(result);

        // Once the program has been compiled succcessfully, we can
        // go ahead and grab reflection data from the program.
        //
        int targetIndex = 0;
        slang::ProgramLayout* programLayout =
            program->getLayout(targetIndex, diagnostics.writeRef());
        diagnoseIfNeeded(diagnostics);
        if (!programLayout)
        {
            return SLANG_FAIL;
        }

        // The compiled program can also have binary code (either
        // for individual entry points, or the entire program)
        // generated for it.
        //
        ComPtr<slang::IBlob> programBinary;
        result = program->getEntryPointCode(0, 0, programBinary.writeRef(), diagnostics.writeRef());
        diagnoseIfNeeded(diagnostics);
        if (SLANG_FAILED(result))
            return result;

        // Finally, once all of the initialization work is dealt with,
        // we hand control over to the actual logic of the example.
        //
        PipelineLayoutReflectionContext_Vulkan context;

        context._gfxDevice = gfxDevice;
        context._slangSession = slangSession;
        context._slangProgramLayout = programLayout;
        context._slangCompiledProgramBlob = programBinary;

        SLANG_RETURN_ON_FAIL(context.createAndValidatePipelineLayout());

        return SLANG_OK;
    }
};

int main(int argc, char* argv[])
{
    ReflectionParameterBlocksExampleApp app;
    if (SLANG_FAILED(app.execute(argc, argv)))
    {
        return -1;
    }
    return 0;
}
