// main.cpp

// This file provides the application code for the `hello-world` example.
//

// This example uses Vulkan to run a simple compute shader written in Slang.
// The goal is to demonstrate how to use the Slang API to cross compile
// shader code.
//
#include "core/slang-string-util.h"
#include "examples/example-base/example-base.h"
#include "examples/example-base/test-base.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "vulkan-api.h"

using Slang::ComPtr;

static const ExampleResources resourceBase("hello-world");

struct HelloWorldExample : public TestBase
{
    // The Vulkan functions pointers result from loading the vulkan library.
    VulkanAPI vkAPI;

    // Vulkan objects used in this example.
    VkQueue queue;
    VkCommandPool commandPool = VK_NULL_HANDLE;

    // Input and output buffers.
    VkBuffer inOutBuffers[3] = {};
    VkDeviceMemory bufferMemories[3] = {};

    const size_t inputElementCount = 16;
    const size_t bufferSize = sizeof(float) * inputElementCount;

    // We use a staging buffer allocated on host-visible memory to
    // upload/download data from GPU.
    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    // Initializes the Vulkan instance and device.
    int initVulkanInstanceAndDevice();

    // This function contains the most interesting part of this example.
    // It loads the `hello-world.slang` shader and compile it using the Slang API
    // into a SPIRV module, then create a Vulkan pipeline from the compiled shader.
    int createComputePipelineFromShader();

    // Creates the input and output buffers.
    int createInOutBuffers();

    // Sets up descriptor set bindings and dispatches the compute task.
    int dispatchCompute();

    // Reads back and prints the result of the compute task.
    int printComputeResults();

    // Main logic of this example.
    int run();

    ~HelloWorldExample();
};


int exampleMain(int argc, char** argv)
{
    initDebugCallback();
    HelloWorldExample example;
    example.parseOption(argc, argv);
    return example.run();
}

/************************************************************/
/* HelloWorldExample Implementation */
/************************************************************/

int HelloWorldExample::run()
{
    // If VK failed to initialize, skip running but return success anyway.
    // This allows our automated testing to distinguish between essential failures and the
    // case where the application is just not supported.
    if (int result = initVulkanInstanceAndDevice())
        return (vkAPI.device == VK_NULL_HANDLE) ? 0 : result;
    RETURN_ON_FAIL(createComputePipelineFromShader());
    RETURN_ON_FAIL(createInOutBuffers());
    RETURN_ON_FAIL(dispatchCompute());
    RETURN_ON_FAIL(printComputeResults());
    return 0;
}

int HelloWorldExample::initVulkanInstanceAndDevice()
{
    if (initializeVulkanDevice(vkAPI) != 0)
    {
        printf("Failed to load Vulkan.\n");
        return -1;
    }

    VkCommandPoolCreateInfo poolCreateInfo = {};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolCreateInfo.queueFamilyIndex = vkAPI.queueFamilyIndex;
    RETURN_ON_FAIL(vkAPI.vkCreateCommandPool(vkAPI.device, &poolCreateInfo, nullptr, &commandPool));

    vkAPI.vkGetDeviceQueue(vkAPI.device, vkAPI.queueFamilyIndex, 0, &queue);
    return 0;
}

int HelloWorldExample::createComputePipelineFromShader()
{
    // First we need to create slang global session with work with the Slang API.
    ComPtr<slang::IGlobalSession> slangGlobalSession;
    RETURN_ON_FAIL(slang::createGlobalSession(slangGlobalSession.writeRef()));

    // Next we create a compilation session to generate SPIRV code from Slang source.
    slang::SessionDesc sessionDesc = {};
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_SPIRV;
    targetDesc.profile = slangGlobalSession->findProfile("spirv_1_5");
    targetDesc.flags = 0;


    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;

    std::vector<slang::CompilerOptionEntry> options;
    options.push_back(
        {slang::CompilerOptionName::EmitSpirvDirectly,
         {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr}});
    sessionDesc.compilerOptionEntries = options.data();
    sessionDesc.compilerOptionEntryCount = options.size();

    ComPtr<slang::ISession> session;
    RETURN_ON_FAIL(slangGlobalSession->createSession(sessionDesc, session.writeRef()));

    // Once the session has been obtained, we can start loading code into it.
    //
    // The simplest way to load code is by calling `loadModule` with the name of a Slang
    // module. A call to `loadModule("hello-world")` will behave more or less as if you
    // wrote:
    //
    //      import hello_world;
    //
    // In a Slang shader file. The compiler will use its search paths to try to locate
    // `hello-world.slang`, then compile and load that file. If a matching module had
    // already been loaded previously, that would be used directly.
    slang::IModule* slangModule = nullptr;
    {
        ComPtr<slang::IBlob> diagnosticBlob;
        Slang::String path = resourceBase.resolveResource("hello-world.slang");
        slangModule = session->loadModule(path.getBuffer(), diagnosticBlob.writeRef());
        diagnoseIfNeeded(diagnosticBlob);
        if (!slangModule)
            return -1;
    }

    // Loading the `hello-world` module will compile and check all the shader code in it,
    // including the shader entry points we want to use. Now that the module is loaded
    // we can look up those entry points by name.
    //
    // Note: If you are using this `loadModule` approach to load your shader code it is
    // important to tag your entry point functions with the `[shader("...")]` attribute
    // (e.g., `[shader("compute")] void computeMain(...)`). Without that information there
    // is no umambiguous way for the compiler to know which functions represent entry
    // points when it parses your code via `loadModule()`.
    //
    ComPtr<slang::IEntryPoint> entryPoint;
    slangModule->findEntryPointByName("computeMain", entryPoint.writeRef());

    // At this point we have a few different Slang API objects that represent
    // pieces of our code: `module`, `vertexEntryPoint`, and `fragmentEntryPoint`.
    //
    // A single Slang module could contain many different entry points (e.g.,
    // four vertex entry points, three fragment entry points, and two compute
    // shaders), and before we try to generate output code for our target API
    // we need to identify which entry points we plan to use together.
    //
    // Modules and entry points are both examples of *component types* in the
    // Slang API. The API also provides a way to build a *composite* out of
    // other pieces, and that is what we are going to do with our module
    // and entry points.
    //
    Slang::List<slang::IComponentType*> componentTypes;
    componentTypes.add(slangModule);
    componentTypes.add(entryPoint);

    // Actually creating the composite component type is a single operation
    // on the Slang session, but the operation could potentially fail if
    // something about the composite was invalid (e.g., you are trying to
    // combine multiple copies of the same module), so we need to deal
    // with the possibility of diagnostic output.
    //
    ComPtr<slang::IComponentType> composedProgram;
    {
        ComPtr<slang::IBlob> diagnosticsBlob;
        SlangResult result = session->createCompositeComponentType(
            componentTypes.getBuffer(),
            componentTypes.getCount(),
            composedProgram.writeRef(),
            diagnosticsBlob.writeRef());
        diagnoseIfNeeded(diagnosticsBlob);
        RETURN_ON_FAIL(result);
    }

    // Now we can call `composedProgram->getEntryPointCode()` to retrieve the
    // compiled SPIRV code that we will use to create a vulkan compute pipeline.
    // This will trigger the final Slang compilation and spirv code generation.
    ComPtr<slang::IBlob> spirvCode;
    {
        ComPtr<slang::IBlob> diagnosticsBlob;
        SlangResult result = composedProgram->getEntryPointCode(
            0,
            0,
            spirvCode.writeRef(),
            diagnosticsBlob.writeRef());
        diagnoseIfNeeded(diagnosticsBlob);
        RETURN_ON_FAIL(result);

        if (isTestMode())
        {
            printEntrypointHashes(1, 1, composedProgram);
        }
    }

    // The following steps are all Vulkan API calls to create a pipeline.

    // First we need to create a descriptor set layout and a pipeline layout.
    // In this example, the pipeline layout is simple: we have a single descriptor
    // set with three buffer descriptors for our input/output storage buffers.
    // General applications typically has much more complicated pipeline layouts,
    // and should consider using Slang's reflection API to learn about the shader
    // parameter layout of a shader program. However, Slang's reflection API is
    // out of scope of this example.
    VkDescriptorSetLayoutCreateInfo descSetLayoutCreateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descSetLayoutCreateInfo.bindingCount = 3;
    VkDescriptorSetLayoutBinding bindings[3];
    for (int i = 0; i < 3; i++)
    {
        auto& binding = bindings[i];
        binding.binding = i;
        binding.descriptorCount = 1;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.stageFlags = VK_SHADER_STAGE_ALL;
        binding.pImmutableSamplers = nullptr;
    }
    descSetLayoutCreateInfo.pBindings = bindings;
    RETURN_ON_FAIL(vkAPI.vkCreateDescriptorSetLayout(
        vkAPI.device,
        &descSetLayoutCreateInfo,
        nullptr,
        &descriptorSetLayout));
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
    RETURN_ON_FAIL(vkAPI.vkCreatePipelineLayout(
        vkAPI.device,
        &pipelineLayoutCreateInfo,
        nullptr,
        &pipelineLayout));

    // Next we create a shader module from the compiled SPIRV code.
    VkShaderModuleCreateInfo shaderCreateInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shaderCreateInfo.codeSize = spirvCode->getBufferSize();
    shaderCreateInfo.pCode = static_cast<const uint32_t*>(spirvCode->getBufferPointer());
    VkShaderModule vkShaderModule;
    RETURN_ON_FAIL(
        vkAPI.vkCreateShaderModule(vkAPI.device, &shaderCreateInfo, nullptr, &vkShaderModule));

    // Now we have all we need to create a compute pipeline.
    VkComputePipelineCreateInfo pipelineCreateInfo = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineCreateInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineCreateInfo.stage.module = vkShaderModule;
    pipelineCreateInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineCreateInfo.stage.pName = "main";
    pipelineCreateInfo.layout = pipelineLayout;
    RETURN_ON_FAIL(vkAPI.vkCreateComputePipelines(
        vkAPI.device,
        VK_NULL_HANDLE,
        1,
        &pipelineCreateInfo,
        nullptr,
        &pipeline));

    // We can destroy shader module now since it will no longer be used.
    vkAPI.vkDestroyShaderModule(vkAPI.device, vkShaderModule, nullptr);

    return 0;
}

int HelloWorldExample::createInOutBuffers()
{
    // Create input and output buffers that resides in device-local memory.
    for (int i = 0; i < 3; i++)
    {
        VkBufferCreateInfo bufferCreateInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferCreateInfo.size = bufferSize;
        bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        RETURN_ON_FAIL(
            vkAPI.vkCreateBuffer(vkAPI.device, &bufferCreateInfo, nullptr, &inOutBuffers[i]));
        VkMemoryRequirements memoryReqs = {};
        vkAPI.vkGetBufferMemoryRequirements(vkAPI.device, inOutBuffers[i], &memoryReqs);

        int memoryTypeIndex = vkAPI.findMemoryTypeIndex(
            memoryReqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        assert(memoryTypeIndex >= 0);

        VkMemoryPropertyFlags actualMemoryProperites =
            vkAPI.deviceMemoryProperties.memoryTypes[memoryTypeIndex].propertyFlags;

        VkMemoryAllocateInfo allocateInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocateInfo.allocationSize = memoryReqs.size;
        allocateInfo.memoryTypeIndex = memoryTypeIndex;
        RETURN_ON_FAIL(
            vkAPI.vkAllocateMemory(vkAPI.device, &allocateInfo, nullptr, &bufferMemories[i]));
        RETURN_ON_FAIL(
            vkAPI.vkBindBufferMemory(vkAPI.device, inOutBuffers[i], bufferMemories[i], 0));
    }

    // Create the device memory and buffer object used for reading/writing
    // data to/from the device local buffers.
    {
        VkBufferCreateInfo bufferCreateInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferCreateInfo.size = bufferSize;
        bufferCreateInfo.usage =
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        RETURN_ON_FAIL(
            vkAPI.vkCreateBuffer(vkAPI.device, &bufferCreateInfo, nullptr, &stagingBuffer));
        VkMemoryRequirements memoryReqs = {};
        vkAPI.vkGetBufferMemoryRequirements(vkAPI.device, stagingBuffer, &memoryReqs);

        int memoryTypeIndex = vkAPI.findMemoryTypeIndex(
            memoryReqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        assert(memoryTypeIndex >= 0);

        VkMemoryPropertyFlags actualMemoryProperites =
            vkAPI.deviceMemoryProperties.memoryTypes[memoryTypeIndex].propertyFlags;

        VkMemoryAllocateInfo allocateInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocateInfo.allocationSize = memoryReqs.size;
        allocateInfo.memoryTypeIndex = memoryTypeIndex;
        RETURN_ON_FAIL(
            vkAPI.vkAllocateMemory(vkAPI.device, &allocateInfo, nullptr, &stagingMemory));
        RETURN_ON_FAIL(vkAPI.vkBindBufferMemory(vkAPI.device, stagingBuffer, stagingMemory, 0));
    }

    // Map staging buffer and writes in the initial input content.
    float* stagingBufferData = nullptr;
    vkAPI.vkMapMemory(vkAPI.device, stagingMemory, 0, bufferSize, 0, (void**)&stagingBufferData);
    if (!stagingBufferData)
        return -1;
    for (size_t i = 0; i < inputElementCount; i++)
        stagingBufferData[i] = static_cast<float>(i);
    vkAPI.vkUnmapMemory(vkAPI.device, stagingMemory);

    // Create a temporary command buffer for recording commands that writes initial
    // data into the input buffers.
    VkCommandBuffer uploadCommandBuffer;
    VkCommandBufferAllocateInfo commandBufferAllocInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    commandBufferAllocInfo.commandBufferCount = 1;
    commandBufferAllocInfo.commandPool = commandPool;
    commandBufferAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    RETURN_ON_FAIL(vkAPI.vkAllocateCommandBuffers(
        vkAPI.device,
        &commandBufferAllocInfo,
        &uploadCommandBuffer));

    VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkAPI.vkBeginCommandBuffer(uploadCommandBuffer, &beginInfo);
    VkBufferCopy bufferCopy = {};
    bufferCopy.size = bufferSize;
    vkAPI.vkCmdCopyBuffer(uploadCommandBuffer, stagingBuffer, inOutBuffers[0], 1, &bufferCopy);
    vkAPI.vkCmdCopyBuffer(uploadCommandBuffer, stagingBuffer, inOutBuffers[1], 1, &bufferCopy);
    vkAPI.vkEndCommandBuffer(uploadCommandBuffer);
    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &uploadCommandBuffer;
    vkAPI.vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkAPI.vkQueueWaitIdle(queue);
    vkAPI.vkFreeCommandBuffers(vkAPI.device, commandPool, 1, &uploadCommandBuffer);
    return 0;
}

int HelloWorldExample::dispatchCompute()
{
    // Create a descriptor pool.
    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    VkDescriptorPoolSize poolSizes[] = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16}};
    descriptorPoolCreateInfo.maxSets = 4;
    descriptorPoolCreateInfo.poolSizeCount = sizeof(poolSizes) / sizeof(VkDescriptorPoolSize);
    descriptorPoolCreateInfo.pPoolSizes = poolSizes;
    descriptorPoolCreateInfo.flags = 0;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    RETURN_ON_FAIL(vkAPI.vkCreateDescriptorPool(
        vkAPI.device,
        &descriptorPoolCreateInfo,
        nullptr,
        &descriptorPool));

    // Allocate descriptor set.
    VkDescriptorSetAllocateInfo descSetAllocInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    descSetAllocInfo.descriptorPool = descriptorPool;
    descSetAllocInfo.descriptorSetCount = 1;
    descSetAllocInfo.pSetLayouts = &descriptorSetLayout;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    RETURN_ON_FAIL(vkAPI.vkAllocateDescriptorSets(vkAPI.device, &descSetAllocInfo, &descriptorSet));

    // Write descriptor set.
    VkWriteDescriptorSet descriptorSetWrites[3] = {};
    VkDescriptorBufferInfo bufferInfo[3];
    for (int i = 0; i < 3; i++)
    {
        bufferInfo[i].buffer = inOutBuffers[i];
        bufferInfo[i].offset = 0;
        bufferInfo[i].range = bufferSize;

        descriptorSetWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorSetWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetWrites[i].descriptorCount = 1;
        descriptorSetWrites[i].dstBinding = i;
        descriptorSetWrites[i].dstSet = descriptorSet;
        descriptorSetWrites[i].pBufferInfo = &bufferInfo[i];
    }
    vkAPI.vkUpdateDescriptorSets(vkAPI.device, 3, descriptorSetWrites, 0, nullptr);

    // Allocate command buffer and record dispatch commands.
    VkCommandBuffer commandBuffer;
    VkCommandBufferAllocateInfo commandBufferAllocInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    commandBufferAllocInfo.commandBufferCount = 1;
    commandBufferAllocInfo.commandPool = commandPool;
    commandBufferAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    RETURN_ON_FAIL(
        vkAPI.vkAllocateCommandBuffers(vkAPI.device, &commandBufferAllocInfo, &commandBuffer));
    VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkAPI.vkBeginCommandBuffer(commandBuffer, &beginInfo);
    vkAPI.vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkAPI.vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipelineLayout,
        0,
        1,
        &descriptorSet,
        0,
        nullptr);
    vkAPI.vkCmdDispatch(commandBuffer, (uint32_t)inputElementCount, 1, 1);
    vkAPI.vkEndCommandBuffer(commandBuffer);

    // Submit command buffer and wait.
    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkAPI.vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkAPI.vkQueueWaitIdle(queue);
    vkAPI.vkFreeCommandBuffers(vkAPI.device, commandPool, 1, &commandBuffer);

    // Clean up.
    vkAPI.vkDestroyDescriptorPool(vkAPI.device, descriptorPool, nullptr);
    return 0;
}

int HelloWorldExample::printComputeResults()
{
    // Allocate command buffer to read back data.
    VkCommandBuffer commandBuffer;
    VkCommandBufferAllocateInfo commandBufferAllocInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    commandBufferAllocInfo.commandBufferCount = 1;
    commandBufferAllocInfo.commandPool = commandPool;
    commandBufferAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    RETURN_ON_FAIL(
        vkAPI.vkAllocateCommandBuffers(vkAPI.device, &commandBufferAllocInfo, &commandBuffer));

    // Record commands to copy output buffer into staging buffer.
    VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkAPI.vkBeginCommandBuffer(commandBuffer, &beginInfo);
    VkBufferCopy bufferCopy = {};
    bufferCopy.size = bufferSize;
    vkAPI.vkCmdCopyBuffer(commandBuffer, inOutBuffers[2], stagingBuffer, 1, &bufferCopy);
    vkAPI.vkEndCommandBuffer(commandBuffer);

    // Execute command buffer and wait.
    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkAPI.vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkAPI.vkQueueWaitIdle(queue);
    vkAPI.vkFreeCommandBuffers(vkAPI.device, commandPool, 1, &commandBuffer);

    // Map and read back staging buffer.
    float* stagingBufferData = nullptr;
    vkAPI.vkMapMemory(vkAPI.device, stagingMemory, 0, bufferSize, 0, (void**)&stagingBufferData);
    if (!stagingBufferData)
        return -1;
    for (size_t i = 0; i < inputElementCount; i++)
    {
        printf("%f\n", stagingBufferData[i]);
    }
    return 0;
}

HelloWorldExample::~HelloWorldExample()
{
    if (vkAPI.device == VK_NULL_HANDLE)
        return;

    vkAPI.vkDestroyPipeline(vkAPI.device, pipeline, nullptr);
    for (int i = 0; i < 3; i++)
    {
        vkAPI.vkDestroyBuffer(vkAPI.device, inOutBuffers[i], nullptr);
        vkAPI.vkFreeMemory(vkAPI.device, bufferMemories[i], nullptr);
    }
    vkAPI.vkDestroyBuffer(vkAPI.device, stagingBuffer, nullptr);
    vkAPI.vkFreeMemory(vkAPI.device, stagingMemory, nullptr);
    vkAPI.vkDestroyPipelineLayout(vkAPI.device, pipelineLayout, nullptr);
    vkAPI.vkDestroyDescriptorSetLayout(vkAPI.device, descriptorSetLayout, nullptr);
    vkAPI.vkDestroyCommandPool(vkAPI.device, commandPool, nullptr);
}
