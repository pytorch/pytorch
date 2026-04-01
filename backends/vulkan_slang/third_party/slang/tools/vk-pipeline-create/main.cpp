// main.cpp

// This tools reads a gfx pipeline dump file and replays the pipeline creation to trigger
// shader compilation in the driver.
//
#include "../../source/core/slang-stream.h"
#include "../../source/core/slang-string-util.h"
#include "examples/hello-world/vulkan-api.h"
#include "slang-com-ptr.h"
#include "slang-gfx.h"
#include "slang.h"

#include <chrono>

#if SLANG_WINDOWS_FAMILY
#include <windows.h>
#else
#include <dlfcn.h>
#endif

using namespace Slang;

struct PipelineCreationReplay
{
    // The Vulkan functions pointers result from loading the vulkan library.
    VulkanAPI vkAPI;

    Dictionary<Index, VkPipelineLayout> pipelineLayouts;
    Dictionary<Index, VkDescriptorSetLayout> descSetLayouts;
    Dictionary<Index, VkShaderModule> shaderModules;
    Dictionary<Index, VkPipeline> pipelines;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    int initVulkanInstanceAndDevice();

    List<uint8_t> fileBlob;
    List<Index> pipelineOffsets;

    struct Reader
    {
        Index position;
        List<uint8_t>& fileBlob;
        Reader(List<uint8_t>& blob, Index pos)
            : fileBlob(blob), position(pos)
        {
        }
        template<typename T>
        void readRaw(T& val)
        {
            memcpy(&val, fileBlob.getBuffer() + position, sizeof(T));
            position += sizeof(T);
        }

        Index readIndex()
        {
            Index index;
            readRaw(index);
            return index;
        }

        uint32_t readUInt32()
        {
            uint32_t index;
            readRaw(index);
            return index;
        }

        const char* readString()
        {
            uint32_t len = readUInt32();
            auto result = (const char*)fileBlob.getBuffer() + position;
            position += len;
            return result;
        }

        const char* getPtr() { return (const char*)fileBlob.getBuffer() + position; }
    };

    VkShaderModule loadShaderModule(Index offset)
    {
        VkShaderModule shader = VK_NULL_HANDLE;
        if (shaderModules.tryGetValue(offset, shader))
            return shader;

        Reader reader(fileBlob, offset);
        VkShaderModuleCreateInfo createInfo = {};
        reader.readRaw(createInfo.sType);
        reader.readRaw(createInfo.flags);
        createInfo.codeSize = reader.readUInt32();
        createInfo.codeSize *= sizeof(uint32_t);
        createInfo.pCode = (uint32_t*)reader.getPtr();
        vkAPI.vkCreateShaderModule(vkAPI.device, &createInfo, nullptr, &shader);
        shaderModules[offset] = shader;

        return shader;
    }

    VkDescriptorSetLayout loadDescriptorSetLayout(Index offset)
    {
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        if (descSetLayouts.tryGetValue(offset, layout))
            return layout;
        Reader reader(fileBlob, offset);
        VkDescriptorSetLayoutCreateInfo createInfo = {};
        reader.readRaw(createInfo.sType);
        reader.readRaw(createInfo.flags);
        reader.readRaw(createInfo.bindingCount);
        List<VkDescriptorSetLayoutBinding> bindings;
        bindings.setCount(createInfo.bindingCount);
        memcpy(
            bindings.getBuffer(),
            reader.getPtr(),
            sizeof(VkDescriptorSetLayoutBinding) * bindings.getCount());
        createInfo.pBindings = bindings.getBuffer();

        vkAPI.vkCreateDescriptorSetLayout(vkAPI.device, &createInfo, nullptr, &layout);
        descSetLayouts[offset] = layout;
        return layout;
    }

    VkPipelineLayout loadPipelineLayout(Index offset)
    {
        VkPipelineLayout layout = VK_NULL_HANDLE;
        if (pipelineLayouts.tryGetValue(offset, layout))
            return layout;

        Reader reader(fileBlob, offset);
        VkPipelineLayoutCreateInfo createInfo = {};
        reader.readRaw(createInfo.sType);
        reader.readRaw(createInfo.flags);
        reader.readRaw(createInfo.setLayoutCount);
        List<VkDescriptorSetLayout> setLayouts;
        for (uint32_t i = 0; i < createInfo.setLayoutCount; i++)
        {
            setLayouts.add(loadDescriptorSetLayout(reader.readIndex()));
        }
        createInfo.pSetLayouts = setLayouts.getBuffer();
        reader.readRaw(createInfo.pushConstantRangeCount);
        List<VkPushConstantRange> pushConstants;
        pushConstants.setCount(createInfo.pushConstantRangeCount);
        memcpy(
            pushConstants.getBuffer(),
            reader.getPtr(),
            sizeof(VkPushConstantRange) * createInfo.pushConstantRangeCount);
        createInfo.pPushConstantRanges = pushConstants.getBuffer();

        vkAPI.vkCreatePipelineLayout(vkAPI.device, &createInfo, nullptr, &layout);
        pipelineLayouts[offset] = layout;
        return layout;
    }

    void loadPipeline(Index id, Index offset)
    {
        printf("Creating pipeline %d...", (int)id);

        Reader reader(fileBlob, offset);
        VkComputePipelineCreateInfo createInfo = {};
        reader.readRaw(createInfo.sType);
        reader.readRaw(createInfo.flags);
        reader.readRaw(createInfo.stage.sType);
        reader.readRaw(createInfo.stage.flags);
        reader.readRaw(createInfo.stage.stage);
        createInfo.stage.module = loadShaderModule(reader.readIndex());
        createInfo.stage.pName = reader.readString();
        createInfo.layout = loadPipelineLayout(reader.readIndex());

        VkPipeline pipeline = VK_NULL_HANDLE;

        auto startTime = std::chrono::high_resolution_clock::now();

        if (vkAPI.vkCreateComputePipelines(
                vkAPI.device,
                VK_NULL_HANDLE,
                1,
                &createInfo,
                nullptr,
                &pipeline) == 0)
            printf("done");
        else
            printf("failed");

        auto endTime = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        printf(" in %.2fs.\n", elapsed.count() / 1000.0);

        vkAPI.vkDestroyPipeline(vkAPI.device, pipeline, nullptr);
    }

    int createComputePipelineFromShader(UnownedStringSlice path, Int pipelineIndex)
    {
        RefPtr<FileStream> f = new FileStream();
        f->init(path, FileMode::Open);
        uint32_t pipelineCount;
        size_t readBytes;
        f->read(&pipelineCount, sizeof(uint32_t), readBytes);
        for (uint32_t i = 0; i < pipelineCount; ++i)
        {
            Index offset;
            f->read(&offset, sizeof(Index), readBytes);
            pipelineOffsets.add(offset);
        }
        Index blobSize;
        f->read(&blobSize, sizeof(Index), readBytes);
        fileBlob.setCount(blobSize);
        f->read(fileBlob.getBuffer(), sizeof(uint8_t) * blobSize, readBytes);

        if (pipelineIndex == -1)
        {
            for (Index i = 0; i < pipelineOffsets.getCount(); ++i)
            {
                loadPipeline(i, pipelineOffsets[i]);
            }
        }
        else if (pipelineIndex < pipelineOffsets.getCount())
        {
            loadPipeline(pipelineIndex, pipelineOffsets[pipelineIndex]);
        }

        for (auto p : descSetLayouts)
            vkAPI.vkDestroyDescriptorSetLayout(
                vkAPI.device,
                *KeyValueDetail::getValue(&p),
                nullptr);
        for (auto p : pipelineLayouts)
            vkAPI.vkDestroyPipelineLayout(vkAPI.device, *KeyValueDetail::getValue(&p), nullptr);
        for (auto p : shaderModules)
            vkAPI.vkDestroyShaderModule(vkAPI.device, *KeyValueDetail::getValue(&p), nullptr);

        return 0;
    }

    int run(int argc, const char** argv);

    void initVulkanAPI(gfx::IDevice* device);
};

int main(int argc, const char** argv)
{
    PipelineCreationReplay app;
    return app.run(argc, argv);
}

int PipelineCreationReplay::run(int argc, const char** argv)
{
    gfx::IDevice::Desc deviceDesc = {};
    deviceDesc.deviceType = gfx::DeviceType::Vulkan;
    ComPtr<gfx::IDevice> device;
    gfx::gfxCreateDevice(&deviceDesc, device.writeRef());
    initVulkanAPI(device);

    if (argc < 2)
    {
        printf("Usage: vk-pipeline-create <path-to-pipeline-file> [pipeline-index]\n");
        return -1;
    }
    UnownedStringSlice path = UnownedStringSlice(argv[1]);
    Int pipelineIndex = -1;
    if (argc > 2)
    {
        StringUtil::parseInt(UnownedStringSlice(argv[2]), pipelineIndex);
    }

    RETURN_ON_FAIL(createComputePipelineFromShader(path, pipelineIndex));

    vkAPI.vkDestroyDevice = nullptr;
    vkAPI.vkDestroyDebugReportCallbackEXT = nullptr;
    vkAPI.vkDestroyInstance = nullptr;
    return 0;
}

void PipelineCreationReplay::initVulkanAPI(gfx::IDevice* device)
{
    gfx::IDevice::InteropHandles handle;
    device->getNativeDeviceHandles(&handle);
    vkAPI.device = (VkDevice)(handle.handles[2].handleValue);
    vkAPI.instance = (VkInstance)(handle.handles[0].handleValue);
#if SLANG_WINDOWS_FAMILY
    auto dynamicLibraryName = "vulkan-1.dll";
    HMODULE module = ::LoadLibraryA(dynamicLibraryName);
    vkAPI.vulkanLibraryHandle = (void*)module;
#define VK_API_GET_GLOBAL_PROC(x) vkAPI.x = (PFN_##x)GetProcAddress(module, #x);
#else
    auto dynamicLibraryName = "libvulkan.so.1";
    vkAPI.vulkanLibraryHandle = dlopen(dynamicLibraryName, RTLD_NOW);
#define VK_API_GET_GLOBAL_PROC(x) vkAPI.x = (PFN_##x)dlsym(vkAPI.vulkanLibraryHandle, #x);
#endif

    // Initialize all the global functions.
    VK_API_ALL_GLOBAL_PROCS(VK_API_GET_GLOBAL_PROC);

    vkAPI.initInstanceProcs();
    vkAPI.initDeviceProcs();
}

int PipelineCreationReplay::initVulkanInstanceAndDevice()
{
    if (initializeVulkanDevice(vkAPI) != 0)
    {
        printf("Failed to load Vulkan.\n");
        return -1;
    }
    return 0;
}
