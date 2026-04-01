// main.cpp

// This file implements an example of hardware ray-tracing using
// Slang shaders and the `gfx` graphics API.

#include "core/slang-basic.h"
#include "examples/example-base/example-base.h"
#include "gfx-util/shader-cursor.h"
#include "platform/vector-math.h"
#include "platform/window.h"
#include "slang-com-ptr.h"
#include "slang-gfx.h"
#include "slang.h"

using namespace gfx;
using namespace Slang;

static const ExampleResources resourceBase("ray-tracing-pipeline");

struct Uniforms
{
    float screenWidth, screenHeight;
    float focalLength = 24.0f, frameHeight = 24.0f;
    float cameraDir[4];
    float cameraUp[4];
    float cameraRight[4];
    float cameraPosition[4];
    float lightDir[4];
};

struct Vertex
{
    float position[3];
};

// Define geometry data for our test scene.
// The scene contains a floor plane, and a cube placed on top of it at the center.
static const int kVertexCount = 24;
static const Vertex kVertexData[kVertexCount] = {
    // Floor plane
    {{-100.0f, 0, 100.0f}},
    {{100.0f, 0, 100.0f}},
    {{100.0f, 0, -100.0f}},
    {{-100.0f, 0, -100.0f}},
    // Cube face (+y).
    {{-1.0f, 2.0, 1.0f}},
    {{1.0f, 2.0, 1.0f}},
    {{1.0f, 2.0, -1.0f}},
    {{-1.0f, 2.0, -1.0f}},
    // Cube face (+z).
    {{-1.0f, 0.0, 1.0f}},
    {{1.0f, 0.0, 1.0f}},
    {{1.0f, 2.0, 1.0f}},
    {{-1.0f, 2.0, 1.0f}},
    // Cube face (-z).
    {{-1.0f, 0.0, -1.0f}},
    {{-1.0f, 2.0, -1.0f}},
    {{1.0f, 2.0, -1.0f}},
    {{1.0f, 0.0, -1.0f}},
    // Cube face (-x).
    {{-1.0f, 0.0, -1.0f}},
    {{-1.0f, 0.0, 1.0f}},
    {{-1.0f, 2.0, 1.0f}},
    {{-1.0f, 2.0, -1.0f}},
    // Cube face (+x).
    {{1.0f, 2.0, -1.0f}},
    {{1.0f, 2.0, 1.0f}},
    {{1.0f, 0.0, 1.0f}},
    {{1.0f, 0.0, -1.0f}},
};
static const int kIndexCount = 36;
static const int kIndexData[kIndexCount] = {0,  1,  2,  0,  2,  3,  4,  5,  6,  4,  6,  7,
                                            8,  9,  10, 8,  10, 11, 12, 13, 14, 12, 14, 15,
                                            16, 17, 18, 16, 18, 19, 20, 21, 22, 20, 22, 23};

struct Primitive
{
    float data[4];
    float color[4];
};
static const int kPrimitiveCount = 12;
static const Primitive kPrimitiveData[kPrimitiveCount] = {
    {{0.0f, 1.0f, 0.0f, 0.0f}, {0.75f, 0.8f, 0.85f, 1.0f}},
    {{0.0f, 1.0f, 0.0f, 0.0f}, {0.75f, 0.8f, 0.85f, 1.0f}},
    {{0.0f, 1.0f, 0.0f, 0.0f}, {0.95f, 0.85f, 0.05f, 1.0f}},
    {{0.0f, 1.0f, 0.0f, 0.0f}, {0.95f, 0.85f, 0.05f, 1.0f}},
    {{0.0f, 0.0f, 1.0f, 0.0f}, {0.95f, 0.85f, 0.05f, 1.0f}},
    {{0.0f, 0.0f, 1.0f, 0.0f}, {0.95f, 0.85f, 0.05f, 1.0f}},
    {{0.0f, 0.0f, -1.0f, 0.0f}, {0.95f, 0.85f, 0.05f, 1.0f}},
    {{0.0f, 0.0f, -1.0f, 0.0f}, {0.95f, 0.85f, 0.05f, 1.0f}},
    {{-1.0f, 0.0f, 0.0f, 0.0f}, {0.95f, 0.85f, 0.05f, 1.0f}},
    {{-1.0f, 0.0f, 0.0f, 0.0f}, {0.95f, 0.85f, 0.05f, 1.0f}},
    {{1.0f, 0.0f, 0.0f, 0.0f}, {0.95f, 0.85f, 0.05f, 1.0f}},
    {{1.0f, 0.0f, 0.0f, 0.0f}, {0.95f, 0.85f, 0.05f, 1.0f}},
};


// We need to use a rasterization pipeline to copy the ray-traced image
// to the swapchain. To do so we need to render a full-screen triangle.
// We will define a small helper type that defines the data for such a triangle.
//
struct FullScreenTriangle
{
    struct Vertex
    {
        float position[2];
    };

    enum
    {
        kVertexCount = 3
    };

    static const Vertex kVertices[kVertexCount];
};
const FullScreenTriangle::Vertex FullScreenTriangle::kVertices[FullScreenTriangle::kVertexCount] = {
    {{-1, -1}},
    {{-1, 3}},
    {{3, -1}},
};

// The example application will be implemented as a `struct`, so that
// we can scope the resources it allocates without using global variables.
//
struct RayTracing : public WindowedAppBase
{


    Uniforms gUniforms = {};


    // Many Slang API functions return detailed diagnostic information
    // (error messages, warnings, etc.) as a "blob" of data, or return
    // a null blob pointer instead if there were no issues.
    //
    // For convenience, we define a subroutine that will dump the information
    // in a diagnostic blob if one is produced, and skip it otherwise.
    //
    void diagnoseIfNeeded(slang::IBlob* diagnosticsBlob)
    {
        if (diagnosticsBlob != nullptr)
        {
            printf("%s", (const char*)diagnosticsBlob->getBufferPointer());
#ifdef _WIN32
            _Win32OutputDebugString((const char*)diagnosticsBlob->getBufferPointer());
#endif
        }
    }

    // Load and compile shader code from souce.
    gfx::Result loadShaderProgram(
        gfx::IDevice* device,
        bool isRayTracingPipeline,
        gfx::IShaderProgram** outProgram)
    {
        ComPtr<slang::ISession> slangSession;
        slangSession = device->getSlangSession();

        ComPtr<slang::IBlob> diagnosticsBlob;
        Slang::String path = resourceBase.resolveResource("shaders.slang");
        slang::IModule* module =
            slangSession->loadModule(path.getBuffer(), diagnosticsBlob.writeRef());
        diagnoseIfNeeded(diagnosticsBlob);
        if (!module)
            return SLANG_FAIL;

        Slang::List<slang::IComponentType*> componentTypes;
        componentTypes.add(module);
        if (isRayTracingPipeline)
        {
            ComPtr<slang::IEntryPoint> entryPoint;
            SLANG_RETURN_ON_FAIL(
                module->findEntryPointByName("rayGenShader", entryPoint.writeRef()));
            componentTypes.add(entryPoint);
            SLANG_RETURN_ON_FAIL(module->findEntryPointByName("missShader", entryPoint.writeRef()));
            componentTypes.add(entryPoint);
            SLANG_RETURN_ON_FAIL(
                module->findEntryPointByName("closestHitShader", entryPoint.writeRef()));
            componentTypes.add(entryPoint);
            SLANG_RETURN_ON_FAIL(
                module->findEntryPointByName("shadowRayHitShader", entryPoint.writeRef()));
            componentTypes.add(entryPoint);
        }
        else
        {
            ComPtr<slang::IEntryPoint> entryPoint;
            SLANG_RETURN_ON_FAIL(module->findEntryPointByName("vertexMain", entryPoint.writeRef()));
            componentTypes.add(entryPoint);
            SLANG_RETURN_ON_FAIL(
                module->findEntryPointByName("fragmentMain", entryPoint.writeRef()));
            componentTypes.add(entryPoint);
        }

        ComPtr<slang::IComponentType> linkedProgram;
        SlangResult result = slangSession->createCompositeComponentType(
            componentTypes.getBuffer(),
            componentTypes.getCount(),
            linkedProgram.writeRef(),
            diagnosticsBlob.writeRef());
        diagnoseIfNeeded(diagnosticsBlob);
        SLANG_RETURN_ON_FAIL(result);

        if (isTestMode())
        {
            printEntrypointHashes(componentTypes.getCount() - 1, 1, linkedProgram);
        }

        gfx::IShaderProgram::Desc programDesc = {};
        programDesc.slangGlobalScope = linkedProgram;
        SLANG_RETURN_ON_FAIL(device->createProgram(programDesc, outProgram));

        return SLANG_OK;
    }

    ComPtr<gfx::IPipelineState> gPresentPipelineState;
    ComPtr<gfx::IPipelineState> gRenderPipelineState;
    ComPtr<gfx::IBufferResource> gFullScreenVertexBuffer;
    ComPtr<gfx::IBufferResource> gVertexBuffer;
    ComPtr<gfx::IBufferResource> gIndexBuffer;
    ComPtr<gfx::IBufferResource> gPrimitiveBuffer;
    ComPtr<gfx::IBufferResource> gTransformBuffer;
    ComPtr<gfx::IResourceView> gPrimitiveBufferSRV;
    ComPtr<gfx::IBufferResource> gInstanceBuffer;
    ComPtr<gfx::IBufferResource> gBLASBuffer;
    ComPtr<gfx::IAccelerationStructure> gBLAS;
    ComPtr<gfx::IBufferResource> gTLASBuffer;
    ComPtr<gfx::IAccelerationStructure> gTLAS;
    ComPtr<gfx::ITextureResource> gResultTexture;
    ComPtr<gfx::IResourceView> gResultTextureUAV;
    ComPtr<gfx::IShaderTable> gShaderTable;

    uint64_t lastTime = 0;

    // glm::vec3 lightDir = normalize(glm::vec3(10, 10, 10));
    // glm::vec3 lightColor = glm::vec3(1, 1, 1);

    glm::vec3 cameraPosition = glm::vec3(-2.53f, 2.72f, 4.3f);
    float cameraOrientationAngles[2] = {-0.475f, -0.35f}; // Spherical angles (theta, phi).

    float translationScale = 0.5f;
    float rotationScale = 0.01f;

    // In order to control camera movement, we will
    // use good old WASD
    bool wPressed = false;
    bool aPressed = false;
    bool sPressed = false;
    bool dPressed = false;

    bool isMouseDown = false;
    float lastMouseX = 0.0f;
    float lastMouseY = 0.0f;

    void setKeyState(platform::KeyCode key, bool state)
    {
        switch (key)
        {
        default:
            break;
        case platform::KeyCode::W:
            wPressed = state;
            break;
        case platform::KeyCode::A:
            aPressed = state;
            break;
        case platform::KeyCode::S:
            sPressed = state;
            break;
        case platform::KeyCode::D:
            dPressed = state;
            break;
        }
    }
    void onKeyDown(platform::KeyEventArgs args) { setKeyState(args.key, true); }
    void onKeyUp(platform::KeyEventArgs args) { setKeyState(args.key, false); }

    void onMouseDown(platform::MouseEventArgs args)
    {
        isMouseDown = true;
        lastMouseX = (float)args.x;
        lastMouseY = (float)args.y;
    }

    void onMouseMove(platform::MouseEventArgs args)
    {
        if (isMouseDown)
        {
            float deltaX = args.x - lastMouseX;
            float deltaY = args.y - lastMouseY;

            cameraOrientationAngles[0] += -deltaX * rotationScale;
            cameraOrientationAngles[1] += -deltaY * rotationScale;
            lastMouseX = (float)args.x;
            lastMouseY = (float)args.y;
        }
    }
    void onMouseUp(platform::MouseEventArgs args) { isMouseDown = false; }

    Slang::Result initialize()
    {
        SLANG_RETURN_ON_FAIL(initializeBase("Ray Tracing Pipeline", 1024, 768));
        if (!isTestMode())
        {
            gWindow->events.mouseMove = [this](const platform::MouseEventArgs& e)
            { onMouseMove(e); };
            gWindow->events.mouseUp = [this](const platform::MouseEventArgs& e) { onMouseUp(e); };
            gWindow->events.mouseDown = [this](const platform::MouseEventArgs& e)
            { onMouseDown(e); };
            gWindow->events.keyDown = [this](const platform::KeyEventArgs& e) { onKeyDown(e); };
            gWindow->events.keyUp = [this](const platform::KeyEventArgs& e) { onKeyUp(e); };
        }

        IBufferResource::Desc vertexBufferDesc;
        vertexBufferDesc.type = IResource::Type::Buffer;
        vertexBufferDesc.sizeInBytes = kVertexCount * sizeof(Vertex);
        vertexBufferDesc.defaultState = ResourceState::ShaderResource;
        gVertexBuffer = gDevice->createBufferResource(vertexBufferDesc, &kVertexData[0]);
        if (!gVertexBuffer)
            return SLANG_FAIL;

        IBufferResource::Desc indexBufferDesc;
        indexBufferDesc.type = IResource::Type::Buffer;
        indexBufferDesc.sizeInBytes = kIndexCount * sizeof(int32_t);
        indexBufferDesc.defaultState = ResourceState::ShaderResource;
        gIndexBuffer = gDevice->createBufferResource(indexBufferDesc, &kIndexData[0]);
        if (!gIndexBuffer)
            return SLANG_FAIL;

        IBufferResource::Desc primitiveBufferDesc;
        primitiveBufferDesc.type = IResource::Type::Buffer;
        primitiveBufferDesc.sizeInBytes = kPrimitiveCount * sizeof(Primitive);
        primitiveBufferDesc.elementSize = sizeof(Primitive);
        primitiveBufferDesc.defaultState = ResourceState::ShaderResource;
        gPrimitiveBuffer = gDevice->createBufferResource(primitiveBufferDesc, &kPrimitiveData[0]);
        if (!gPrimitiveBuffer)
            return SLANG_FAIL;

        IResourceView::Desc primitiveSRVDesc = {};
        primitiveSRVDesc.format = Format::Unknown;
        primitiveSRVDesc.type = IResourceView::Type::ShaderResource;
        gPrimitiveBufferSRV =
            gDevice->createBufferView(gPrimitiveBuffer, nullptr, primitiveSRVDesc);

        IBufferResource::Desc transformBufferDesc;
        transformBufferDesc.type = IResource::Type::Buffer;
        transformBufferDesc.sizeInBytes = sizeof(float) * 12;
        transformBufferDesc.defaultState = ResourceState::ShaderResource;
        float transformData[12] =
            {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
        gTransformBuffer = gDevice->createBufferResource(transformBufferDesc, &transformData);
        if (!gTransformBuffer)
            return SLANG_FAIL;
        // Build bottom level acceleration structure.
        {
            IAccelerationStructure::BuildInputs accelerationStructureBuildInputs;
            IAccelerationStructure::PrebuildInfo accelerationStructurePrebuildInfo;
            accelerationStructureBuildInputs.descCount = 1;
            accelerationStructureBuildInputs.kind = IAccelerationStructure::Kind::BottomLevel;
            accelerationStructureBuildInputs.flags =
                IAccelerationStructure::BuildFlags::AllowCompaction;
            IAccelerationStructure::GeometryDesc geomDesc;
            geomDesc.flags = IAccelerationStructure::GeometryFlags::Opaque;
            geomDesc.type = IAccelerationStructure::GeometryType::Triangles;
            geomDesc.content.triangles.indexCount = kIndexCount;
            geomDesc.content.triangles.indexData = gIndexBuffer->getDeviceAddress();
            geomDesc.content.triangles.indexFormat = Format::R32_UINT;
            geomDesc.content.triangles.vertexCount = kVertexCount;
            geomDesc.content.triangles.vertexData = gVertexBuffer->getDeviceAddress();
            geomDesc.content.triangles.vertexFormat = Format::R32G32B32_FLOAT;
            geomDesc.content.triangles.vertexStride = sizeof(Vertex);
            geomDesc.content.triangles.transform3x4 = gTransformBuffer->getDeviceAddress();
            accelerationStructureBuildInputs.geometryDescs = &geomDesc;

            // Query buffer size for acceleration structure build.
            SLANG_RETURN_ON_FAIL(gDevice->getAccelerationStructurePrebuildInfo(
                accelerationStructureBuildInputs,
                &accelerationStructurePrebuildInfo));
            // Allocate buffers for acceleration structure.
            IBufferResource::Desc asDraftBufferDesc;
            asDraftBufferDesc.type = IResource::Type::Buffer;
            asDraftBufferDesc.defaultState = ResourceState::AccelerationStructure;
            asDraftBufferDesc.sizeInBytes =
                (size_t)accelerationStructurePrebuildInfo.resultDataMaxSize;
            ComPtr<IBufferResource> draftBuffer = gDevice->createBufferResource(asDraftBufferDesc);
            if (!draftBuffer)
                return SLANG_FAIL;
            IBufferResource::Desc scratchBufferDesc;
            scratchBufferDesc.type = IResource::Type::Buffer;
            scratchBufferDesc.defaultState = ResourceState::UnorderedAccess;
            scratchBufferDesc.sizeInBytes =
                (size_t)accelerationStructurePrebuildInfo.scratchDataSize;
            ComPtr<IBufferResource> scratchBuffer =
                gDevice->createBufferResource(scratchBufferDesc);
            if (!scratchBuffer)
                return SLANG_FAIL;

            // Build acceleration structure.
            ComPtr<IQueryPool> compactedSizeQuery;
            IQueryPool::Desc queryPoolDesc;
            queryPoolDesc.count = 1;
            queryPoolDesc.type = QueryType::AccelerationStructureCompactedSize;
            SLANG_RETURN_ON_FAIL(
                gDevice->createQueryPool(queryPoolDesc, compactedSizeQuery.writeRef()));

            ComPtr<IAccelerationStructure> draftAS;
            IAccelerationStructure::CreateDesc draftCreateDesc;
            draftCreateDesc.buffer = draftBuffer;
            draftCreateDesc.kind = IAccelerationStructure::Kind::BottomLevel;
            draftCreateDesc.offset = 0;
            draftCreateDesc.size = accelerationStructurePrebuildInfo.resultDataMaxSize;
            SLANG_RETURN_ON_FAIL(
                gDevice->createAccelerationStructure(draftCreateDesc, draftAS.writeRef()));

            compactedSizeQuery->reset();

            auto commandBuffer = gTransientHeaps[0]->createCommandBuffer();
            auto encoder = commandBuffer->encodeRayTracingCommands();
            IAccelerationStructure::BuildDesc buildDesc = {};
            buildDesc.dest = draftAS;
            buildDesc.inputs = accelerationStructureBuildInputs;
            buildDesc.scratchData = scratchBuffer->getDeviceAddress();
            AccelerationStructureQueryDesc compactedSizeQueryDesc = {};
            compactedSizeQueryDesc.queryPool = compactedSizeQuery;
            compactedSizeQueryDesc.queryType = QueryType::AccelerationStructureCompactedSize;
            encoder->buildAccelerationStructure(buildDesc, 1, &compactedSizeQueryDesc);
            encoder->endEncoding();
            commandBuffer->close();
            gQueue->executeCommandBuffer(commandBuffer);
            gQueue->waitOnHost();

            uint64_t compactedSize = 0;
            compactedSizeQuery->getResult(0, 1, &compactedSize);
            IBufferResource::Desc asBufferDesc;
            asBufferDesc.type = IResource::Type::Buffer;
            asBufferDesc.defaultState = ResourceState::AccelerationStructure;
            asBufferDesc.sizeInBytes = (size_t)compactedSize;
            gBLASBuffer = gDevice->createBufferResource(asBufferDesc);
            IAccelerationStructure::CreateDesc createDesc;
            createDesc.buffer = gBLASBuffer;
            createDesc.kind = IAccelerationStructure::Kind::BottomLevel;
            createDesc.offset = 0;
            createDesc.size = (size_t)compactedSize;
            gDevice->createAccelerationStructure(createDesc, gBLAS.writeRef());

            commandBuffer = gTransientHeaps[0]->createCommandBuffer();
            encoder = commandBuffer->encodeRayTracingCommands();
            encoder->copyAccelerationStructure(
                gBLAS,
                draftAS,
                AccelerationStructureCopyMode::Compact);
            encoder->endEncoding();
            commandBuffer->close();
            gQueue->executeCommandBuffer(commandBuffer);
            gQueue->waitOnHost();
        }

        // Build top level acceleration structure.
        {
            List<IAccelerationStructure::InstanceDesc> instanceDescs;
            instanceDescs.setCount(1);
            instanceDescs[0].accelerationStructure = gBLAS->getDeviceAddress();
            instanceDescs[0].flags =
                IAccelerationStructure::GeometryInstanceFlags::TriangleFacingCullDisable;
            instanceDescs[0].instanceContributionToHitGroupIndex = 0;
            instanceDescs[0].instanceID = 0;
            instanceDescs[0].instanceMask = 0xFF;
            float transformMatrix[] =
                {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
            memcpy(&instanceDescs[0].transform[0][0], transformMatrix, sizeof(float) * 12);

            IBufferResource::Desc instanceBufferDesc;
            instanceBufferDesc.type = IResource::Type::Buffer;
            instanceBufferDesc.sizeInBytes =
                instanceDescs.getCount() * sizeof(IAccelerationStructure::InstanceDesc);
            instanceBufferDesc.defaultState = ResourceState::ShaderResource;
            gInstanceBuffer =
                gDevice->createBufferResource(instanceBufferDesc, instanceDescs.getBuffer());
            if (!gInstanceBuffer)
                return SLANG_FAIL;

            IAccelerationStructure::BuildInputs accelerationStructureBuildInputs = {};
            IAccelerationStructure::PrebuildInfo accelerationStructurePrebuildInfo = {};
            accelerationStructureBuildInputs.descCount = 1;
            accelerationStructureBuildInputs.kind = IAccelerationStructure::Kind::TopLevel;
            accelerationStructureBuildInputs.instanceDescs = gInstanceBuffer->getDeviceAddress();

            // Query buffer size for acceleration structure build.
            SLANG_RETURN_ON_FAIL(gDevice->getAccelerationStructurePrebuildInfo(
                accelerationStructureBuildInputs,
                &accelerationStructurePrebuildInfo));

            IBufferResource::Desc asBufferDesc;
            asBufferDesc.type = IResource::Type::Buffer;
            asBufferDesc.defaultState = ResourceState::AccelerationStructure;
            asBufferDesc.sizeInBytes = (size_t)accelerationStructurePrebuildInfo.resultDataMaxSize;
            gTLASBuffer = gDevice->createBufferResource(asBufferDesc);

            IBufferResource::Desc scratchBufferDesc;
            scratchBufferDesc.type = IResource::Type::Buffer;
            scratchBufferDesc.defaultState = ResourceState::UnorderedAccess;
            scratchBufferDesc.sizeInBytes =
                (size_t)accelerationStructurePrebuildInfo.scratchDataSize;
            ComPtr<IBufferResource> scratchBuffer =
                gDevice->createBufferResource(scratchBufferDesc);

            IAccelerationStructure::CreateDesc createDesc;
            createDesc.buffer = gTLASBuffer;
            createDesc.kind = IAccelerationStructure::Kind::TopLevel;
            createDesc.offset = 0;
            createDesc.size = (size_t)accelerationStructurePrebuildInfo.resultDataMaxSize;
            SLANG_RETURN_ON_FAIL(
                gDevice->createAccelerationStructure(createDesc, gTLAS.writeRef()));

            auto commandBuffer = gTransientHeaps[0]->createCommandBuffer();
            auto encoder = commandBuffer->encodeRayTracingCommands();
            IAccelerationStructure::BuildDesc buildDesc = {};
            buildDesc.dest = gTLAS;
            buildDesc.inputs = accelerationStructureBuildInputs;
            buildDesc.scratchData = scratchBuffer->getDeviceAddress();
            encoder->buildAccelerationStructure(buildDesc, 0, nullptr);
            encoder->endEncoding();
            commandBuffer->close();
            gQueue->executeCommandBuffer(commandBuffer);
            gQueue->waitOnHost();
        }

        IBufferResource::Desc fullScreenVertexBufferDesc;
        fullScreenVertexBufferDesc.type = IResource::Type::Buffer;
        fullScreenVertexBufferDesc.sizeInBytes =
            FullScreenTriangle::kVertexCount * sizeof(FullScreenTriangle::Vertex);
        fullScreenVertexBufferDesc.defaultState = ResourceState::VertexBuffer;
        gFullScreenVertexBuffer = gDevice->createBufferResource(
            fullScreenVertexBufferDesc,
            &FullScreenTriangle::kVertices[0]);
        if (!gFullScreenVertexBuffer)
            return SLANG_FAIL;

        InputElementDesc inputElements[] = {
            {"POSITION", 0, Format::R32G32_FLOAT, offsetof(FullScreenTriangle::Vertex, position)},
        };
        auto inputLayout = gDevice->createInputLayout(
            sizeof(FullScreenTriangle::Vertex),
            &inputElements[0],
            SLANG_COUNT_OF(inputElements));
        if (!inputLayout)
            return SLANG_FAIL;

        ComPtr<IShaderProgram> shaderProgram;
        SLANG_RETURN_ON_FAIL(loadShaderProgram(gDevice, false, shaderProgram.writeRef()));
        GraphicsPipelineStateDesc desc;
        desc.inputLayout = inputLayout;
        desc.program = shaderProgram;
        desc.framebufferLayout = gFramebufferLayout;
        gPresentPipelineState = gDevice->createGraphicsPipelineState(desc);
        if (!gPresentPipelineState)
            return SLANG_FAIL;

        const char* hitgroupNames[] = {"hitgroup0", "hitgroup1"};

        ComPtr<IShaderProgram> rayTracingProgram;
        SLANG_RETURN_ON_FAIL(loadShaderProgram(gDevice, true, rayTracingProgram.writeRef()));
        RayTracingPipelineStateDesc rtpDesc = {};
        rtpDesc.program = rayTracingProgram;
        rtpDesc.hitGroupCount = 2;
        HitGroupDesc hitGroups[2];
        hitGroups[0].closestHitEntryPoint = "closestHitShader";
        hitGroups[0].hitGroupName = hitgroupNames[0];
        hitGroups[1].closestHitEntryPoint = "shadowRayHitShader";
        hitGroups[1].hitGroupName = hitgroupNames[1];
        rtpDesc.hitGroups = hitGroups;
        rtpDesc.maxRayPayloadSize = 64;
        rtpDesc.maxRecursion = 2;
        SLANG_RETURN_ON_FAIL(
            gDevice->createRayTracingPipelineState(rtpDesc, gRenderPipelineState.writeRef()));
        if (!gRenderPipelineState)
            return SLANG_FAIL;

        IShaderTable::Desc shaderTableDesc = {};
        const char* raygenName = "rayGenShader";
        const char* missName = "missShader";
        shaderTableDesc.program = rayTracingProgram;
        shaderTableDesc.hitGroupCount = 2;
        shaderTableDesc.hitGroupNames = hitgroupNames;
        shaderTableDesc.rayGenShaderCount = 1;
        shaderTableDesc.rayGenShaderEntryPointNames = &raygenName;
        shaderTableDesc.missShaderCount = 1;
        shaderTableDesc.missShaderEntryPointNames = &missName;
        SLANG_RETURN_ON_FAIL(gDevice->createShaderTable(shaderTableDesc, gShaderTable.writeRef()));

        createResultTexture();
        return SLANG_OK;
    }

    void createResultTexture()
    {
        ITextureResource::Desc resultTextureDesc = {};
        resultTextureDesc.type = IResource::Type::Texture2D;
        resultTextureDesc.numMipLevels = 1;
        resultTextureDesc.size.width = windowWidth;
        resultTextureDesc.size.height = windowHeight;
        resultTextureDesc.size.depth = 1;
        resultTextureDesc.defaultState = ResourceState::UnorderedAccess;
        resultTextureDesc.format = Format::R16G16B16A16_FLOAT;
        gResultTexture = gDevice->createTextureResource(resultTextureDesc);
        IResourceView::Desc resultUAVDesc = {};
        resultUAVDesc.format = resultTextureDesc.format;
        resultUAVDesc.type = IResourceView::Type::UnorderedAccess;
        gResultTextureUAV = gDevice->createTextureView(gResultTexture, resultUAVDesc);
    }

    virtual void windowSizeChanged() override
    {
        WindowedAppBase::windowSizeChanged();
        createResultTexture();
    }

    glm::vec3 getVectorFromSphericalAngles(float theta, float phi)
    {
        auto sinTheta = sin(theta);
        auto cosTheta = cos(theta);
        auto sinPhi = sin(phi);
        auto cosPhi = cos(phi);
        return glm::vec3(-sinTheta * cosPhi, sinPhi, -cosTheta * cosPhi);
    }
    void updateUniforms()
    {
        gUniforms.screenWidth = (float)windowWidth;
        gUniforms.screenHeight = (float)windowHeight;
        if (!lastTime)
            lastTime = getCurrentTime();
        uint64_t currentTime = getCurrentTime();
        float deltaTime = float(double(currentTime - lastTime) / double(getTimerFrequency()));
        lastTime = currentTime;

        auto camDir =
            getVectorFromSphericalAngles(cameraOrientationAngles[0], cameraOrientationAngles[1]);
        auto camUp = getVectorFromSphericalAngles(
            cameraOrientationAngles[0],
            cameraOrientationAngles[1] + glm::pi<float>() * 0.5f);
        auto camRight = glm::cross(camDir, camUp);

        glm::vec3 movement = glm::vec3(0);
        if (wPressed)
            movement += camDir;
        if (sPressed)
            movement -= camDir;
        if (aPressed)
            movement -= camRight;
        if (dPressed)
            movement += camRight;

        cameraPosition += deltaTime * translationScale * movement;

        memcpy(gUniforms.cameraDir, &camDir, sizeof(float) * 3);
        memcpy(gUniforms.cameraUp, &camUp, sizeof(float) * 3);
        memcpy(gUniforms.cameraRight, &camRight, sizeof(float) * 3);
        memcpy(gUniforms.cameraPosition, &cameraPosition, sizeof(float) * 3);
        auto lightDir = glm::normalize(glm::vec3(1.0f, 3.0f, 2.0f));
        memcpy(gUniforms.lightDir, &lightDir, sizeof(float) * 3);
    }

    virtual void renderFrame(int frameBufferIndex) override
    {
        updateUniforms();
        {
            ComPtr<ICommandBuffer> renderCommandBuffer =
                gTransientHeaps[frameBufferIndex]->createCommandBuffer();
            auto renderEncoder = renderCommandBuffer->encodeRayTracingCommands();
            IShaderObject* rootObject = nullptr;
            renderEncoder->bindPipeline(gRenderPipelineState, &rootObject);
            auto cursor = ShaderCursor(rootObject);
            cursor["resultTexture"].setResource(gResultTextureUAV);
            cursor["uniforms"].setData(&gUniforms, sizeof(Uniforms));
            cursor["sceneBVH"].setResource(gTLAS);
            cursor["primitiveBuffer"].setResource(gPrimitiveBufferSRV);
            renderEncoder->dispatchRays(0, gShaderTable, windowWidth, windowHeight, 1);
            renderEncoder->endEncoding();
            renderCommandBuffer->close();
            gQueue->executeCommandBuffer(renderCommandBuffer);
        }

        {
            ComPtr<ICommandBuffer> presentCommandBuffer =
                gTransientHeaps[frameBufferIndex]->createCommandBuffer();
            auto presentEncoder = presentCommandBuffer->encodeRenderCommands(
                gRenderPass,
                gFramebuffers[frameBufferIndex]);
            gfx::Viewport viewport = {};
            viewport.maxZ = 1.0f;
            viewport.extentX = (float)windowWidth;
            viewport.extentY = (float)windowHeight;
            presentEncoder->setViewportAndScissor(viewport);
            auto rootObject = presentEncoder->bindPipeline(gPresentPipelineState);
            auto cursor = ShaderCursor(rootObject->getEntryPoint(1));
            cursor["t"].setResource(gResultTextureUAV);
            presentEncoder->setVertexBuffer(0, gFullScreenVertexBuffer);
            presentEncoder->setPrimitiveTopology(PrimitiveTopology::TriangleList);
            presentEncoder->draw(3);
            presentEncoder->endEncoding();
            presentCommandBuffer->close();
            gQueue->executeCommandBuffer(presentCommandBuffer);
        }

        if (!isTestMode())
        {
            // With that, we are done drawing for one frame, and ready for the next.
            //
            gSwapchain->present();
        }
    }
};

// This macro instantiates an appropriate main function to
// run the application defined above.
EXAMPLE_MAIN(innerMain<RayTracing>);
