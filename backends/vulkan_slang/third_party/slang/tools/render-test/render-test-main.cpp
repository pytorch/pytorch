// render-test-main.cpp

#define _CRT_SECURE_NO_WARNINGS 1

#include "../../source/core/slang-test-tool-util.h"
#include "../source/core/slang-io.h"
#include "../source/core/slang-string-util.h"
#include "core/slang-token-reader.h"
#include "options.h"
#include "png-serialize-util.h"
#include "shader-input-layout.h"
#include "shader-renderer-util.h"
#include "slang-support.h"
#include "window.h"

#if defined(_WIN32)
#include <d3d12.h>
#endif

#include <slang-rhi.h>
#include <slang-rhi/acceleration-structure-utils.h>
#include <slang-rhi/shader-cursor.h>
#include <stdio.h>
#include <stdlib.h>
#define ENABLE_RENDERDOC_INTEGRATION 0

#if ENABLE_RENDERDOC_INTEGRATION
#include "external/renderdoc_app.h"

#include <windows.h>
#endif

namespace renderer_test
{

using Slang::Result;

int gWindowWidth = 1024;
int gWindowHeight = 768;

//
// For the purposes of a small example, we will define the vertex data for a
// single triangle directly in the source file. It should be easy to extend
// this example to load data from an external source, if desired.
//

struct Vertex
{
    float position[3];
    float color[3];
    float uv[2];
    float customData0[4];
    float customData1[4];
    float customData2[4];
    float customData3[4];
};

static const Vertex kVertexData[] = {
    {{0, 0, 0.5}, {1, 0, 0}, {0, 0}, {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}},
    {{0, 1, 0.5}, {0, 0, 1}, {1, 0}, {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}},
    {{1, 0, 0.5}, {0, 1, 0}, {1, 1}, {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}},
};
static const int kVertexCount = SLANG_COUNT_OF(kVertexData);

using namespace Slang;

static void _outputProfileTime(uint64_t startTicks, uint64_t endTicks)
{
    WriterHelper out = StdWriters::getOut();
    double time = double(endTicks - startTicks) / Process::getClockFrequency();
    out.print("profile-time=%g\n", time);
}

class ProgramVars;

struct ShaderOutputPlan
{
    struct Item
    {
        ComPtr<IResource> resource;
        slang::TypeLayoutReflection* typeLayout = nullptr;
    };

    List<Item> items;
};

// A context for hodling resources allocated for a test.
struct TestResourceContext
{
    List<ComPtr<IResource>> resources;
};

class RenderTestApp
{
public:
    Result update();

    // At initialization time, we are going to load and compile our Slang shader
    // code, and then create the API objects we need for rendering.
    Result initialize(
        SlangSession* session,
        IDevice* device,
        const Options& options,
        const ShaderCompilerUtil::Input& input);
    void finalize();

    Result applyBinding(IShaderObject* rootObject);
    void setProjectionMatrix(IShaderObject* rootObject);
    Result writeBindingOutput(const String& fileName);

    Result writeScreen(const String& filename);

protected:
    /// Called in initialize
    Result _initializeShaders(
        SlangSession* session,
        IDevice* device,
        Options::ShaderProgramType shaderType,
        const ShaderCompilerUtil::Input& input);
    void _initializeRenderPass();
    void _initializeAccelerationStructure();

    uint64_t m_startTicks;

    // variables for state to be used for rendering...
    uintptr_t m_constantBufferSize;

    IDevice* m_device;
    ComPtr<ICommandQueue> m_queue;
    ComPtr<IInputLayout> m_inputLayout;
    ComPtr<IBuffer> m_vertexBuffer;
    ComPtr<IShaderProgram> m_shaderProgram;
    ComPtr<IPipeline> m_pipeline;
    ComPtr<IShaderTable> m_shaderTable;
    ComPtr<ITexture> m_depthBuffer;
    ComPtr<ITextureView> m_depthBufferView;
    ComPtr<ITexture> m_colorBuffer;
    ComPtr<ITextureView> m_colorBufferView;

    ComPtr<IBuffer> m_blasBuffer;
    ComPtr<IAccelerationStructure> m_bottomLevelAccelerationStructure;
    ComPtr<IBuffer> m_tlasBuffer;
    ComPtr<IAccelerationStructure> m_topLevelAccelerationStructure;

    ShaderCompilerUtil::OutputAndLayout m_compilationOutput;

    ShaderInputLayout m_shaderInputLayout; ///< The binding layout

    Options m_options;

    ShaderOutputPlan m_outputPlan;
    TestResourceContext m_resourceContext;
};

struct AssignValsFromLayoutContext
{
    IDevice* device;
    slang::IComponentType* slangComponent;
    ShaderOutputPlan& outputPlan;
    TestResourceContext& resourceContext;
    IAccelerationStructure* accelerationStructure;

    AssignValsFromLayoutContext(
        IDevice* device,
        slang::IComponentType* slangComponent,
        ShaderOutputPlan& outputPlan,
        TestResourceContext& resourceContext,
        IAccelerationStructure* accelerationStructure)
        : device(device)
        , slangComponent(slangComponent)
        , outputPlan(outputPlan)
        , resourceContext(resourceContext)
        , accelerationStructure(accelerationStructure)
    {
    }

    slang::ProgramLayout* slangReflection() { return slangComponent->getLayout(); }
    slang::ISession* slangSession() { return slangComponent->getSession(); }

    void maybeAddOutput(
        ShaderCursor const& dstCursor,
        ShaderInputLayout::Val* srcVal,
        IResource* resource)
    {
        if (srcVal->isOutput)
        {
            ShaderOutputPlan::Item item;
            item.resource = resource;
            item.typeLayout = dstCursor.getTypeLayout();
            outputPlan.items.add(item);
        }
    }

    SlangResult assignData(ShaderCursor const& dstCursor, ShaderInputLayout::DataVal* srcVal)
    {
        const size_t bufferSize = srcVal->bufferData.getCount() * sizeof(uint32_t);

        ShaderCursor dataCursor = dstCursor;
        switch (dataCursor.getTypeLayout()->getKind())
        {
        case slang::TypeReflection::Kind::ConstantBuffer:
        case slang::TypeReflection::Kind::ParameterBlock:
            dataCursor = dataCursor.getDereferenced();
            break;

        default:
            break;
        }

        SLANG_RETURN_ON_FAIL(dataCursor.setData(srcVal->bufferData.getBuffer(), bufferSize));
        return SLANG_OK;
    }

    SlangResult assignBuffer(ShaderCursor const& dstCursor, ShaderInputLayout::BufferVal* srcVal)
    {
        const InputBufferDesc& srcBuffer = srcVal->bufferDesc;
        auto& bufferData = srcVal->bufferData;
        const size_t bufferSize = Math::Max(
            (size_t)bufferData.getCount() * sizeof(uint32_t),
            (size_t)(srcBuffer.elementCount * srcBuffer.stride));
        bufferData.reserve(bufferSize / sizeof(uint32_t));
        for (size_t i = bufferData.getCount(); i < bufferSize / sizeof(uint32_t); i++)
            bufferData.add(0);

        ComPtr<IBuffer> bufferResource;

        SLANG_RETURN_ON_FAIL(ShaderRendererUtil::createBuffer(
            srcBuffer,
            /*entry.isOutput,*/ bufferSize,
            bufferData.getBuffer(),
            device,
            bufferResource));

        if ((dstCursor.getTypeLayout()->getType()->getKind() ==
                 slang::TypeReflection::Kind::Scalar &&
             dstCursor.getTypeLayout()->getType()->getScalarType() ==
                 slang::TypeReflection::ScalarType::UInt64) ||
            dstCursor.getTypeLayout()->getType()->getKind() == slang::TypeReflection::Kind::Pointer)
        {
            // dstCursor is pointer to an ordinary uniform data field,
            // we should write bufferResource as a pointer.
            uint64_t addr = bufferResource->getDeviceAddress();
            dstCursor.setData(&addr, sizeof(addr));
            resourceContext.resources.add(ComPtr<IResource>(bufferResource.get()));
            maybeAddOutput(dstCursor, srcVal, bufferResource);
            return SLANG_OK;
        }

        ComPtr<IBuffer> counterResource;
        const auto explicitCounterCursor = dstCursor.getExplicitCounter();
        if (srcBuffer.counter != ~0u)
        {
            if (explicitCounterCursor.isValid())
            {
                // If this cursor has a full buffer object associated with the
                // resource, then assign to that.
                ShaderInputLayout::BufferVal counterVal;
                counterVal.bufferData.add(srcBuffer.counter);
                assignBuffer(explicitCounterCursor, &counterVal);
            }
            else
            {
                // Otherwise, this API (D3D) must be handling the buffer object
                // specially, in which case create the buffer resource to pass
                // into `createBufferView`
                const InputBufferDesc& counterBufferDesc{
                    InputBufferType::StorageBuffer,
                    sizeof(uint32_t),
                    1,
                    Format::Undefined,
                };
                SLANG_RETURN_ON_FAIL(ShaderRendererUtil::createBuffer(
                    counterBufferDesc,
                    sizeof(srcBuffer.counter),
                    &srcBuffer.counter,
                    device,
                    counterResource));
            }
        }
        else if (explicitCounterCursor.isValid())
        {
            // If we know we require a counter for this resource but haven't
            // been given one, error
            return SLANG_E_INVALID_ARG;
        }

        if (counterResource)
        {
            dstCursor.setBinding(Binding(bufferResource, counterResource));
        }
        else
        {
            dstCursor.setBinding(bufferResource);
        }
        maybeAddOutput(dstCursor, srcVal, bufferResource);

        return SLANG_OK;
    }

    SlangResult assignCombinedTextureSampler(
        ShaderCursor const& dstCursor,
        ShaderInputLayout::CombinedTextureSamplerVal* srcVal)
    {
        auto& textureEntry = srcVal->textureVal;
        auto& samplerEntry = srcVal->samplerVal;

        ComPtr<ITexture> texture;
        SLANG_RETURN_ON_FAIL(ShaderRendererUtil::generateTexture(
            textureEntry->textureDesc,
            ResourceState::ShaderResource,
            device,
            texture));

        auto sampler = _createSampler(device, samplerEntry->samplerDesc);

        dstCursor.setBinding(Binding(texture, sampler));
        maybeAddOutput(dstCursor, srcVal, texture);

        return SLANG_OK;
    }

    SlangResult assignTexture(ShaderCursor const& dstCursor, ShaderInputLayout::TextureVal* srcVal)
    {
        ComPtr<ITexture> texture;
        ResourceState defaultState = srcVal->textureDesc.isRWTexture
                                         ? ResourceState::UnorderedAccess
                                         : ResourceState::ShaderResource;

        SLANG_RETURN_ON_FAIL(ShaderRendererUtil::generateTexture(
            srcVal->textureDesc,
            defaultState,
            device,
            texture));

        dstCursor.setBinding(texture);
        maybeAddOutput(dstCursor, srcVal, texture);
        return SLANG_OK;
    }

    SlangResult assignSampler(ShaderCursor const& dstCursor, ShaderInputLayout::SamplerVal* srcVal)
    {
        auto sampler = _createSampler(device, srcVal->samplerDesc);

        dstCursor.setBinding(sampler);
        return SLANG_OK;
    }

    SlangResult assignAggregate(ShaderCursor const& dstCursor, ShaderInputLayout::AggVal* srcVal)
    {
        Index fieldCount = srcVal->fields.getCount();
        for (Index fieldIndex = 0; fieldIndex < fieldCount; ++fieldIndex)
        {
            auto& field = srcVal->fields[fieldIndex];

            if (field.name.getLength() == 0)
            {
                // If no name was given, assume by-indexing matching is requested
                auto fieldCursor = dstCursor.getElement((uint32_t)fieldIndex);
                if (!fieldCursor.isValid())
                {
                    StdWriters::getError().print(
                        "error: could not find shader parameter at index %d\n",
                        (int)fieldIndex);
                    return SLANG_E_INVALID_ARG;
                }
                SLANG_RETURN_ON_FAIL(assign(fieldCursor, field.val));
            }
            else
            {
                auto fieldCursor = dstCursor.getPath(field.name.getBuffer());
                if (!fieldCursor.isValid())
                {
                    StdWriters::getError().print(
                        "error: could not find shader parameter matching '%s'\n",
                        field.name.begin());
                    return SLANG_E_INVALID_ARG;
                }
                SLANG_RETURN_ON_FAIL(assign(fieldCursor, field.val));
            }
        }
        return SLANG_OK;
    }

    SlangResult assignObject(ShaderCursor const& dstCursor, ShaderInputLayout::ObjectVal* srcVal)
    {
        auto typeName = srcVal->typeName;
        slang::TypeReflection* slangType = nullptr;
        if (typeName.getLength() != 0)
        {
            // If the input line specified the name of the type
            // to allocate, then we use it directly.
            //
            slangType = slangReflection()->findTypeByName(typeName.getBuffer());
        }
        else
        {
            // if the user did not specify what type to allocate,
            // then we will infer the type from the type of the
            // value pointed to by `entryCursor`.
            //
            auto slangTypeLayout = dstCursor.getTypeLayout();
            switch (slangTypeLayout->getKind())
            {
            default:
                break;

            case slang::TypeReflection::Kind::ConstantBuffer:
            case slang::TypeReflection::Kind::ParameterBlock:
                // If the cursor is pointing at a constant buffer
                // or parameter block, then we assume the user
                // actually means to allocate an object based on
                // the element type of the block.
                //
                slangTypeLayout = slangTypeLayout->getElementTypeLayout();
                break;
            }
            slangType = slangTypeLayout->getType();
        }

        ComPtr<IShaderObject> shaderObject;
        device->createShaderObject(
            slangSession(),
            slangType,
            ShaderObjectContainerType::None,
            shaderObject.writeRef());

        SLANG_RETURN_ON_FAIL(assign(ShaderCursor(shaderObject), srcVal->contentVal));
        shaderObject->finalize();
        dstCursor.setObject(shaderObject);
        return SLANG_OK;
    }

    SlangResult assignValWithSpecializationArg(
        ShaderCursor const& dstCursor,
        ShaderInputLayout::SpecializeVal* srcVal)
    {
        assign(dstCursor, srcVal->contentVal);
        List<slang::SpecializationArg> args;
        for (auto& typeName : srcVal->typeArgs)
        {
            auto slangType = slangReflection()->findTypeByName(typeName.getBuffer());
            if (!slangType)
            {
                StdWriters::getError().print(
                    "error: could not find shader type '%s'\n",
                    typeName.getBuffer());
                return SLANG_E_INVALID_ARG;
            }
            args.add(slang::SpecializationArg::fromType(slangType));
        }
        return dstCursor.setSpecializationArgs(args.getBuffer(), (uint32_t)args.getCount());
    }

    SlangResult assignArray(ShaderCursor const& dstCursor, ShaderInputLayout::ArrayVal* srcVal)
    {
        Index elementCounter = 0;
        for (auto elementVal : srcVal->vals)
        {
            Index elementIndex = elementCounter++;
            SLANG_RETURN_ON_FAIL(assign(dstCursor[elementIndex], elementVal));
        }
        return SLANG_OK;
    }

    SlangResult assignAccelerationStructure(
        ShaderCursor const& dstCursor,
        ShaderInputLayout::AccelerationStructureVal* srcVal)
    {
        dstCursor.setBinding(accelerationStructure);
        return SLANG_OK;
    }

    SlangResult assign(ShaderCursor const& dstCursor, ShaderInputLayout::ValPtr const& srcVal)
    {
        auto& entryCursor = dstCursor;
        switch (srcVal->kind)
        {
        case ShaderInputType::UniformData:
            return assignData(dstCursor, (ShaderInputLayout::DataVal*)srcVal.Ptr());

        case ShaderInputType::Buffer:
            return assignBuffer(dstCursor, (ShaderInputLayout::BufferVal*)srcVal.Ptr());

        case ShaderInputType::CombinedTextureSampler:
            return assignCombinedTextureSampler(
                dstCursor,
                (ShaderInputLayout::CombinedTextureSamplerVal*)srcVal.Ptr());

        case ShaderInputType::Texture:
            return assignTexture(dstCursor, (ShaderInputLayout::TextureVal*)srcVal.Ptr());

        case ShaderInputType::Sampler:
            return assignSampler(dstCursor, (ShaderInputLayout::SamplerVal*)srcVal.Ptr());

        case ShaderInputType::Object:
            return assignObject(dstCursor, (ShaderInputLayout::ObjectVal*)srcVal.Ptr());

        case ShaderInputType::Specialize:
            return assignValWithSpecializationArg(
                dstCursor,
                (ShaderInputLayout::SpecializeVal*)srcVal.Ptr());

        case ShaderInputType::Aggregate:
            return assignAggregate(dstCursor, (ShaderInputLayout::AggVal*)srcVal.Ptr());

        case ShaderInputType::Array:
            return assignArray(dstCursor, (ShaderInputLayout::ArrayVal*)srcVal.Ptr());

        case ShaderInputType::AccelerationStructure:
            return assignAccelerationStructure(
                dstCursor,
                (ShaderInputLayout::AccelerationStructureVal*)srcVal.Ptr());
        default:
            assert(!"Unhandled type");
            return SLANG_FAIL;
        }
    }
};

static SlangResult _assignVarsFromLayout(
    IDevice* device,
    slang::IComponentType* slangComponent,
    IShaderObject* shaderObject,
    ShaderInputLayout const& layout,
    ShaderOutputPlan& ioOutputPlan,
    TestResourceContext& ioResourceContext,
    IAccelerationStructure* accelerationStructure)
{
    AssignValsFromLayoutContext
        context(device, slangComponent, ioOutputPlan, ioResourceContext, accelerationStructure);
    ShaderCursor rootCursor = ShaderCursor(shaderObject);
    return context.assign(rootCursor, layout.rootVal);
}

Result RenderTestApp::applyBinding(IShaderObject* rootObject)
{
    return _assignVarsFromLayout(
        m_device,
        m_compilationOutput.output.slangProgram,
        rootObject,
        m_compilationOutput.layout,
        m_outputPlan,
        m_resourceContext,
        m_topLevelAccelerationStructure);
}

SlangResult RenderTestApp::initialize(
    SlangSession* session,
    IDevice* device,
    const Options& options,
    const ShaderCompilerUtil::Input& input)
{
    m_options = options;

    // We begin by compiling the shader file and entry points that specified via the options.
    //
    SLANG_RETURN_ON_FAIL(ShaderCompilerUtil::compileWithLayout(
        device->getSlangSession()->getGlobalSession(),
        options,
        input,
        m_compilationOutput));
    m_shaderInputLayout = m_compilationOutput.layout;

    // Once the shaders have been compiled we load them via the underlying API.
    //
    ComPtr<ISlangBlob> outDiagnostics;
    auto result = device->createShaderProgram(
        m_compilationOutput.output.desc,
        m_shaderProgram.writeRef(),
        outDiagnostics.writeRef());

    // If there was a failure creating a program, we can't continue
    // Special case SLANG_E_NOT_AVAILABLE error code to make it a failure,
    // as it is also used to indicate an attempt setup something failed gracefully (because it
    // couldn't be supported) but that's not this.
    if (SLANG_FAILED(result))
    {
        result = (result == SLANG_E_NOT_AVAILABLE) ? SLANG_FAIL : result;
        return result;
    }

    m_device = device;

    _initializeRenderPass();
    _initializeAccelerationStructure();

    {
        switch (m_options.shaderType)
        {
        default:
            assert(!"unexpected test shader type");
            return SLANG_FAIL;

        case Options::ShaderProgramType::Compute:
            {
                ComputePipelineDesc desc;
                desc.program = m_shaderProgram;

                m_pipeline = device->createComputePipeline(desc);
            }
            break;

        case Options::ShaderProgramType::Graphics:
        case Options::ShaderProgramType::GraphicsCompute:
            {
                // TODO: We should conceivably be able to match up the "available" vertex
                // attributes, as defined by the vertex stream(s) on the model being
                // renderer, with the "required" vertex attributes as defiend on the
                // shader.
                //
                // For now we just create a fixed input layout for all graphics tests
                // since at present they all draw the same single triangle with a
                // fixed/known set of attributes.
                //
                const InputElementDesc inputElements[] = {
                    {"A", 0, Format::RGB32Float, offsetof(Vertex, position)},
                    {"A", 1, Format::RGB32Float, offsetof(Vertex, color)},
                    {"A", 2, Format::RG32Float, offsetof(Vertex, uv)},
                    {"A", 3, Format::RGBA32Float, offsetof(Vertex, customData0)},
                    {"A", 4, Format::RGBA32Float, offsetof(Vertex, customData1)},
                    {"A", 5, Format::RGBA32Float, offsetof(Vertex, customData2)},
                    {"A", 6, Format::RGBA32Float, offsetof(Vertex, customData3)},
                };

                ComPtr<IInputLayout> inputLayout;
                SLANG_RETURN_ON_FAIL(device->createInputLayout(
                    sizeof(Vertex),
                    inputElements,
                    SLANG_COUNT_OF(inputElements),
                    inputLayout.writeRef()));

                BufferDesc vertexBufferDesc;
                vertexBufferDesc.size = kVertexCount * sizeof(Vertex);
                vertexBufferDesc.memoryType = MemoryType::DeviceLocal;
                vertexBufferDesc.usage = BufferUsage::VertexBuffer;
                vertexBufferDesc.defaultState = ResourceState::VertexBuffer;

                SLANG_RETURN_ON_FAIL(
                    device->createBuffer(vertexBufferDesc, kVertexData, m_vertexBuffer.writeRef()));

                ColorTargetDesc colorTarget;
                colorTarget.format = Format::RGBA8Unorm;
                RenderPipelineDesc desc;
                desc.program = m_shaderProgram;
                desc.inputLayout = inputLayout;
                desc.targets = &colorTarget;
                desc.targetCount = 1;
                desc.depthStencil.format = Format::D32Float;
                m_pipeline = device->createRenderPipeline(desc);
            }
            break;

        case Options::ShaderProgramType::GraphicsMeshCompute:
        case Options::ShaderProgramType::GraphicsTaskMeshCompute:
            {
                ColorTargetDesc colorTarget;
                colorTarget.format = Format::RGBA8Unorm;
                RenderPipelineDesc desc;
                desc.program = m_shaderProgram;
                desc.targets = &colorTarget;
                desc.targetCount = 1;
                desc.depthStencil.format = Format::D32Float;
                m_pipeline = device->createRenderPipeline(desc);
            }
            break;

        case Options::ShaderProgramType::RayTracing:
            {
                RayTracingPipelineDesc desc;
                desc.program = m_shaderProgram;

                m_pipeline = device->createRayTracingPipeline(desc);

                const char* raygenNames[] = {"raygenMain"};

                // We don't define a miss shader for this test. OptiX allows
                // passing nullptr to indicate no miss shader, but something in
                // slang-rhi assumes that the miss shader always has a name. To
                // work around that, use a dummy name.
                const char* missNames[] = {"missNull"};

                ShaderTableDesc shaderTableDesc = {};
                shaderTableDesc.program = m_shaderProgram;
                shaderTableDesc.rayGenShaderCount = 1;
                shaderTableDesc.rayGenShaderEntryPointNames = raygenNames;
                shaderTableDesc.missShaderCount = 1;
                shaderTableDesc.missShaderEntryPointNames = missNames;
                SLANG_RETURN_ON_FAIL(
                    device->createShaderTable(shaderTableDesc, m_shaderTable.writeRef()));
            }
            break;
        }
    }
    // If success must have a pipeline state
    return m_pipeline ? SLANG_OK : SLANG_FAIL;
}

Result RenderTestApp::_initializeShaders(
    SlangSession* session,
    IDevice* device,
    Options::ShaderProgramType shaderType,
    const ShaderCompilerUtil::Input& input)
{
    SLANG_RETURN_ON_FAIL(ShaderCompilerUtil::compileWithLayout(
        device->getSlangSession()->getGlobalSession(),
        m_options,
        input,
        m_compilationOutput));
    m_shaderInputLayout = m_compilationOutput.layout;
    m_shaderProgram = device->createShaderProgram(m_compilationOutput.output.desc);
    return m_shaderProgram ? SLANG_OK : SLANG_FAIL;
}

void RenderTestApp::_initializeRenderPass()
{
    m_queue = m_device->getQueue(QueueType::Graphics);
    SLANG_ASSERT(m_queue);

    rhi::TextureDesc depthBufferDesc;
    depthBufferDesc.type = TextureType::Texture2D;
    depthBufferDesc.size.width = gWindowWidth;
    depthBufferDesc.size.height = gWindowHeight;
    depthBufferDesc.size.depth = 1;
    depthBufferDesc.mipCount = 1;
    depthBufferDesc.format = Format::D32Float;
    depthBufferDesc.usage = TextureUsage::DepthStencil;
    depthBufferDesc.defaultState = ResourceState::DepthWrite;
    m_depthBuffer = m_device->createTexture(depthBufferDesc, nullptr);
    SLANG_ASSERT(m_depthBuffer);
    m_depthBufferView = m_device->createTextureView(m_depthBuffer, {});
    SLANG_ASSERT(m_depthBufferView);

    rhi::TextureDesc colorBufferDesc;
    colorBufferDesc.type = TextureType::Texture2D;
    colorBufferDesc.size.width = gWindowWidth;
    colorBufferDesc.size.height = gWindowHeight;
    colorBufferDesc.size.depth = 1;
    colorBufferDesc.mipCount = 1;
    colorBufferDesc.format = Format::RGBA8Unorm;
    colorBufferDesc.usage = TextureUsage::RenderTarget | TextureUsage::CopySource;
    colorBufferDesc.defaultState = ResourceState::RenderTarget;
    m_colorBuffer = m_device->createTexture(colorBufferDesc, nullptr);
    SLANG_ASSERT(m_colorBuffer);
    m_colorBufferView = m_device->createTextureView(m_colorBuffer, {});
    SLANG_ASSERT(m_colorBufferView);
}

void RenderTestApp::_initializeAccelerationStructure()
{
    if (!m_device->hasFeature("ray-tracing"))
        return;
    BufferDesc vertexBufferDesc = {};
    vertexBufferDesc.size = kVertexCount * sizeof(Vertex);
    vertexBufferDesc.usage = BufferUsage::AccelerationStructureBuildInput;
    vertexBufferDesc.defaultState = ResourceState::AccelerationStructureBuildInput;
    ComPtr<IBuffer> vertexBuffer = m_device->createBuffer(vertexBufferDesc, &kVertexData[0]);

    BufferDesc transformBufferDesc = {};
    transformBufferDesc.size = sizeof(float) * 12;
    transformBufferDesc.usage = BufferUsage::AccelerationStructureBuildInput;
    transformBufferDesc.defaultState = ResourceState::AccelerationStructureBuildInput;
    float transformData[12] =
        {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    ComPtr<IBuffer> transformBuffer = m_device->createBuffer(transformBufferDesc, &transformData);

    // Build bottom level acceleration structure.
    {
        AccelerationStructureBuildInput buildInput = {};
        buildInput.type = AccelerationStructureBuildInputType::Triangles;
        buildInput.triangles.vertexBuffers[0] = vertexBuffer;
        buildInput.triangles.vertexBufferCount = 1;
        buildInput.triangles.vertexFormat = Format::RGB32Float;
        buildInput.triangles.vertexCount = kVertexCount;
        buildInput.triangles.vertexStride = sizeof(Vertex);
        buildInput.triangles.preTransformBuffer = transformBuffer;
        buildInput.triangles.flags = AccelerationStructureGeometryFlags::Opaque;
        AccelerationStructureBuildDesc buildDesc = {};
        buildDesc.inputs = &buildInput;
        buildDesc.inputCount = 1;
        buildDesc.flags = AccelerationStructureBuildFlags::AllowCompaction;

        // Query buffer size for acceleration structure build.
        AccelerationStructureSizes accelerationStructureSizes = {};
        m_device->getAccelerationStructureSizes(buildDesc, &accelerationStructureSizes);

        BufferDesc scratchBufferDesc = {};
        scratchBufferDesc.usage = BufferUsage::UnorderedAccess;
        scratchBufferDesc.defaultState = ResourceState::UnorderedAccess;
        scratchBufferDesc.size = accelerationStructureSizes.scratchSize;
        ComPtr<IBuffer> scratchBuffer = m_device->createBuffer(scratchBufferDesc);

        ComPtr<IQueryPool> compactedSizeQuery;
        QueryPoolDesc queryPoolDesc = {};
        queryPoolDesc.count = 1;
        queryPoolDesc.type = QueryType::AccelerationStructureCompactedSize;
        m_device->createQueryPool(queryPoolDesc, compactedSizeQuery.writeRef());

        // Build acceleration structure.
        ComPtr<IAccelerationStructure> draftAS;
        AccelerationStructureDesc draftDesc = {};
        draftDesc.size = accelerationStructureSizes.accelerationStructureSize;
        m_device->createAccelerationStructure(draftDesc, draftAS.writeRef());

        compactedSizeQuery->reset();

        auto encoder = m_queue->createCommandEncoder();
        AccelerationStructureQueryDesc compactedSizeQueryDesc = {};
        compactedSizeQueryDesc.queryPool = compactedSizeQuery;
        compactedSizeQueryDesc.queryType = QueryType::AccelerationStructureCompactedSize;
        encoder->buildAccelerationStructure(
            buildDesc,
            draftAS,
            nullptr,
            scratchBuffer,
            1,
            &compactedSizeQueryDesc);
        m_queue->submit(encoder->finish());
        m_queue->waitOnHost();

        uint64_t compactedSize = 0;
        compactedSizeQuery->getResult(0, 1, &compactedSize);
        AccelerationStructureDesc finalDesc;
        finalDesc.size = compactedSize;
        m_device->createAccelerationStructure(
            finalDesc,
            m_bottomLevelAccelerationStructure.writeRef());

        encoder = m_queue->createCommandEncoder();
        encoder->copyAccelerationStructure(
            m_bottomLevelAccelerationStructure,
            draftAS,
            AccelerationStructureCopyMode::Compact);
        m_queue->submit(encoder->finish());
        m_queue->waitOnHost();
    }

    // Build top level acceleration structure.
    {
        AccelerationStructureInstanceDescType nativeInstanceDescType =
            getAccelerationStructureInstanceDescType(m_device);
        Size nativeInstanceDescSize =
            getAccelerationStructureInstanceDescSize(nativeInstanceDescType);

        List<AccelerationStructureInstanceDescGeneric> genericInstanceDescs;
        genericInstanceDescs.setCount(1);
        genericInstanceDescs[0].accelerationStructure =
            m_bottomLevelAccelerationStructure->getHandle();
        genericInstanceDescs[0].flags =
            AccelerationStructureInstanceFlags::TriangleFacingCullDisable;
        genericInstanceDescs[0].instanceContributionToHitGroupIndex = 0;
        genericInstanceDescs[0].instanceID = 0;
        genericInstanceDescs[0].instanceMask = 0xFF;
        float transformMatrix[] =
            {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
        memcpy(&genericInstanceDescs[0].transform[0][0], transformMatrix, sizeof(float) * 12);

        List<unsigned char> nativeInstanceDescs;
        nativeInstanceDescs.setCount(genericInstanceDescs.getCount() * nativeInstanceDescSize);
        convertAccelerationStructureInstanceDescs(
            genericInstanceDescs.getCount(),
            nativeInstanceDescType,
            nativeInstanceDescs.getBuffer(),
            nativeInstanceDescSize,
            genericInstanceDescs.getBuffer(),
            sizeof(AccelerationStructureInstanceDescGeneric));

        BufferDesc instanceBufferDesc = {};
        instanceBufferDesc.size = nativeInstanceDescs.getCount();
        instanceBufferDesc.usage = BufferUsage::AccelerationStructureBuildInput;
        instanceBufferDesc.defaultState = ResourceState::AccelerationStructureBuildInput;
        ComPtr<IBuffer> instanceBuffer =
            m_device->createBuffer(instanceBufferDesc, nativeInstanceDescs.getBuffer());

        AccelerationStructureBuildInput buildInput = {};
        buildInput.type = AccelerationStructureBuildInputType::Instances;
        buildInput.instances.instanceBuffer = instanceBuffer;
        buildInput.instances.instanceCount = 1;
        buildInput.instances.instanceStride = nativeInstanceDescSize;
        AccelerationStructureBuildDesc buildDesc = {};
        buildDesc.inputs = &buildInput;
        buildDesc.inputCount = 1;

        // Query buffer size for acceleration structure build.
        AccelerationStructureSizes accelerationStructureSizes = {};
        m_device->getAccelerationStructureSizes(buildDesc, &accelerationStructureSizes);

        BufferDesc scratchBufferDesc = {};
        scratchBufferDesc.usage = BufferUsage::UnorderedAccess;
        scratchBufferDesc.defaultState = ResourceState::UnorderedAccess;
        scratchBufferDesc.size = (size_t)accelerationStructureSizes.scratchSize;
        ComPtr<IBuffer> scratchBuffer = m_device->createBuffer(scratchBufferDesc);

        AccelerationStructureDesc createDesc = {};
        createDesc.size = accelerationStructureSizes.accelerationStructureSize;
        m_device->createAccelerationStructure(
            createDesc,
            m_topLevelAccelerationStructure.writeRef());

        auto encoder = m_queue->createCommandEncoder();
        encoder->buildAccelerationStructure(
            buildDesc,
            m_topLevelAccelerationStructure,
            nullptr,
            scratchBuffer,
            0,
            nullptr);
        m_queue->submit(encoder->finish());
        m_queue->waitOnHost();
    }
}

void RenderTestApp::setProjectionMatrix(IShaderObject* rootObject)
{
    auto info = m_device->getDeviceInfo();
    ShaderCursor(rootObject)
        .getField("Uniforms")
        .getDereferenced()
        .setData(info.identityProjectionMatrix, sizeof(float) * 16);
}

void RenderTestApp::finalize()
{
    m_compilationOutput.output.reset();
}

Result RenderTestApp::writeBindingOutput(const String& fileName)
{
    // Wait until everything is complete
    m_queue->waitOnHost();

    FILE* f = fopen(fileName.getBuffer(), "wb");
    if (!f)
    {
        return SLANG_FAIL;
    }
    FileWriter writer(f, WriterFlags(0));

    for (auto outputItem : m_outputPlan.items)
    {
        auto resource = outputItem.resource;
        IBuffer* buffer = nullptr;
        resource->queryInterface(IBuffer::getTypeGuid(), (void**)&buffer);
        if (buffer)
        {
            const BufferDesc& bufferDesc = buffer->getDesc();
            const size_t bufferSize = bufferDesc.size;

            ComPtr<ISlangBlob> blob;
            m_device->readBuffer(buffer, 0, bufferSize, blob.writeRef());
            buffer->release();

            if (!blob)
            {
                return SLANG_FAIL;
            }
            const SlangResult res = ShaderInputLayout::writeBinding(
                m_options.outputUsingType ? outputItem.typeLayout
                                          : nullptr, // TODO: always output using type
                blob->getBufferPointer(),
                bufferSize,
                &writer);
            SLANG_RETURN_ON_FAIL(res);
        }
        else
        {
            auto typeName = outputItem.typeLayout->getName();
            printf("invalid output type '%s'.\n", typeName ? typeName : "UNKNOWN");
        }
    }
    return SLANG_OK;
}

Result RenderTestApp::writeScreen(const String& filename)
{
    size_t rowPitch, pixelSize;
    ComPtr<ISlangBlob> blob;
    SLANG_RETURN_ON_FAIL(
        m_device->readTexture(m_colorBuffer, blob.writeRef(), &rowPitch, &pixelSize));
    auto bufferSize = blob->getBufferSize();
    uint32_t width = static_cast<uint32_t>(rowPitch / pixelSize);
    uint32_t height = static_cast<uint32_t>(bufferSize / rowPitch);
    return PngSerializeUtil::write(filename.getBuffer(), blob, width, height);
}

Result RenderTestApp::update()
{
    auto encoder = m_queue->createCommandEncoder();
    if (m_options.shaderType == Options::ShaderProgramType::Compute)
    {
        auto passEncoder = encoder->beginComputePass();
        auto rootObject =
            passEncoder->bindPipeline(static_cast<IComputePipeline*>(m_pipeline.get()));
        applyBinding(rootObject);
        passEncoder->dispatchCompute(
            m_options.computeDispatchSize[0],
            m_options.computeDispatchSize[1],
            m_options.computeDispatchSize[2]);
        passEncoder->end();
    }
    else if (m_options.shaderType == Options::ShaderProgramType::RayTracing)
    {
        auto passEncoder = encoder->beginRayTracingPass();
        auto rootObject = passEncoder->bindPipeline(
            static_cast<IRayTracingPipeline*>(m_pipeline.get()),
            m_shaderTable);
        applyBinding(rootObject);
        passEncoder->dispatchRays(
            0,
            m_options.computeDispatchSize[0],
            m_options.computeDispatchSize[1],
            m_options.computeDispatchSize[2]);
        passEncoder->end();
    }
    else
    {
        RenderPassColorAttachment colorAttachment = {};
        colorAttachment.view = m_colorBufferView;
        colorAttachment.loadOp = LoadOp::Clear;
        colorAttachment.storeOp = StoreOp::Store;
        RenderPassDepthStencilAttachment depthStencilAttachment = {};
        depthStencilAttachment.view = m_depthBufferView;
        depthStencilAttachment.depthLoadOp = LoadOp::Clear;
        depthStencilAttachment.depthStoreOp = StoreOp::Store;
        RenderPassDesc renderPass = {};
        renderPass.colorAttachments = &colorAttachment;
        renderPass.colorAttachmentCount = 1;
        renderPass.depthStencilAttachment = &depthStencilAttachment;

        auto passEncoder = encoder->beginRenderPass(renderPass);
        auto rootObject =
            passEncoder->bindPipeline(static_cast<IRenderPipeline*>(m_pipeline.get()));
        applyBinding(rootObject);
        setProjectionMatrix(rootObject);

        RenderState state;
        state.viewports[0] = Viewport::fromSize(gWindowWidth, gWindowHeight);
        state.viewportCount = 1;
        state.scissorRects[0] = ScissorRect::fromSize(gWindowWidth, gWindowHeight);
        state.scissorRectCount = 1;

        if (m_options.shaderType == Options::ShaderProgramType::GraphicsMeshCompute ||
            m_options.shaderType == Options::ShaderProgramType::GraphicsTaskMeshCompute)
        {
            passEncoder->setRenderState(state);
            passEncoder->drawMeshTasks(
                m_options.computeDispatchSize[0],
                m_options.computeDispatchSize[1],
                m_options.computeDispatchSize[2]);
        }
        else
        {
            state.vertexBuffers[0] = m_vertexBuffer;
            state.vertexBufferCount = 1;
            passEncoder->setRenderState(state);
            DrawArguments args;
            args.vertexCount = 3;
            passEncoder->draw(args);
        }
        passEncoder->end();
    }
    m_startTicks = Process::getClockTick();
    m_queue->submit(encoder->finish());
    m_queue->waitOnHost();

    // If we are in a mode where output is requested, we need to snapshot the back buffer here
    if (m_options.outputPath.getLength() || m_options.performanceProfile)
    {
        // Wait until everything is complete

        if (m_options.performanceProfile)
        {
#if 0
            // It might not be enough on some APIs to 'waitForGpu' to mean the computation has completed. Let's lock an output
            // buffer to be sure
            if (m_bindingState->outputBindings.getCount() > 0)
            {
                const auto& binding = m_bindingState->outputBindings[0];
                auto i = binding.entryIndex;
                const auto& layoutBinding = m_shaderInputLayout.entries[i];

                assert(layoutBinding.isOutput);
                
                if (binding.resource && binding.resource->isBuffer())
                {
                    BufferResource* bufferResource = static_cast<BufferResource*>(binding.resource.Ptr());
                    const size_t bufferSize = bufferResource->getDesc().size;
                    unsigned int* ptr = (unsigned int*)m_renderer->map(bufferResource, MapFlavor::HostRead);
                    if (!ptr)
                    {                            
                        return SLANG_FAIL;
                    }
                    m_renderer->unmap(bufferResource);
                }
            }
#endif

            // Note we don't do the same with screen rendering -> as that will do a lot of work,
            // which may swamp any computation so can only really profile compute shaders at the
            // moment

            const uint64_t endTicks = Process::getClockTick();

            _outputProfileTime(m_startTicks, endTicks);
        }

        if (m_options.outputPath.getLength())
        {
            if (m_options.shaderType == Options::ShaderProgramType::Compute ||
                m_options.shaderType == Options::ShaderProgramType::GraphicsCompute ||
                m_options.shaderType == Options::ShaderProgramType::GraphicsMeshCompute ||
                m_options.shaderType == Options::ShaderProgramType::GraphicsTaskMeshCompute ||
                m_options.shaderType == Options::ShaderProgramType::RayTracing)
            {
                SLANG_RETURN_ON_FAIL(writeBindingOutput(m_options.outputPath));
            }
            else
            {
                SlangResult res = writeScreen(m_options.outputPath);
                if (SLANG_FAILED(res))
                {
                    fprintf(stderr, "ERROR: failed to write screen capture to file\n");
                    return res;
                }
            }
        }
        return SLANG_OK;
    }
    return SLANG_OK;
}


static SlangResult _setSessionPrelude(
    const Options& options,
    const char* exePath,
    SlangSession* session)
{
    // Let's see if we need to set up special prelude for HLSL
    if (options.nvapiExtnSlot.getLength())
    {
#if !SLANG_WINDOWS_FAMILY
        // NVAPI is currently only available on Windows
        return SLANG_E_NOT_AVAILABLE;
#else
        // We want to set the path to NVAPI
        String rootPath;
        SLANG_RETURN_ON_FAIL(TestToolUtil::getRootPath(exePath, rootPath));
        String includePath;
        SLANG_RETURN_ON_FAIL(
            TestToolUtil::getIncludePath(rootPath, "external/nvapi/nvHLSLExtns.h", includePath))

        StringBuilder buf;
        // We have to choose a slot that NVAPI will use.
        buf << "#define NV_SHADER_EXTN_SLOT " << options.nvapiExtnSlot << "\n";

        // Include the NVAPI header
        buf << "#include ";
        StringEscapeUtil::appendQuoted(
            StringEscapeUtil::getHandler(StringEscapeUtil::Style::Cpp),
            includePath.getUnownedSlice(),
            buf);
        buf << "\n\n";

        session->setLanguagePrelude(SLANG_SOURCE_LANGUAGE_HLSL, buf.getBuffer());
#endif
    }
    else
    {
        session->setLanguagePrelude(SLANG_SOURCE_LANGUAGE_HLSL, "");
    }

    return SLANG_OK;
}

} //  namespace renderer_test

#if ENABLE_RENDERDOC_INTEGRATION
static RENDERDOC_API_1_1_2* rdoc_api = NULL;
static void initializeRenderDoc()
{
    if (HMODULE mod = GetModuleHandleA("renderdoc.dll"))
    {
        pRENDERDOC_GetAPI RENDERDOC_GetAPI =
            (pRENDERDOC_GetAPI)GetProcAddress(mod, "RENDERDOC_GetAPI");
        int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_1_2, (void**)&rdoc_api);
        assert(ret == 1);
    }
}
static void renderDocBeginFrame()
{
    if (rdoc_api)
        rdoc_api->StartFrameCapture(nullptr, nullptr);
}
static void renderDocEndFrame()
{
    if (rdoc_api)
        rdoc_api->EndFrameCapture(nullptr, nullptr);
    _fgetchar();
}
#else
static void initializeRenderDoc() {}
static void renderDocBeginFrame() {}
static void renderDocEndFrame() {}
#endif

class StdWritersDebugCallback : public rhi::IDebugCallback
{
public:
    Slang::StdWriters* writers;
    virtual SLANG_NO_THROW void SLANG_MCALL handleMessage(
        rhi::DebugMessageType type,
        rhi::DebugMessageSource source,
        const char* message) override
    {
        SLANG_UNUSED(source);
        if (type == rhi::DebugMessageType::Error)
        {
            writers->getOut().print("%s\n", message);
        }
    }
};

static SlangResult _innerMain(
    Slang::StdWriters* stdWriters,
    SlangSession* session,
    int argcIn,
    const char* const* argvIn)
{
    using namespace renderer_test;
    using namespace Slang;

    initializeRenderDoc();

    StdWriters::setSingleton(stdWriters);

    Options options;

    // Parse command-line options
    SLANG_RETURN_ON_FAIL(Options::parse(argcIn, argvIn, StdWriters::getError(), options));
    if (options.deviceType == DeviceType::Default)
    {
        return SLANG_OK;
    }

    ShaderCompilerUtil::Input input;

    input.profile = "";
    input.target = SLANG_TARGET_NONE;

    SlangSourceLanguage nativeLanguage = SLANG_SOURCE_LANGUAGE_UNKNOWN;
    SlangPassThrough slangPassThrough = SLANG_PASS_THROUGH_NONE;
    char const* profileName = "";
    switch (options.deviceType)
    {
    case DeviceType::D3D11:
        input.target = SLANG_DXBC;
        input.profile = "sm_5_0";
        nativeLanguage = SLANG_SOURCE_LANGUAGE_HLSL;
        slangPassThrough = SLANG_PASS_THROUGH_FXC;

        break;

    case DeviceType::D3D12:
        input.target = SLANG_DXBC;
        input.profile = "sm_5_0";
        nativeLanguage = SLANG_SOURCE_LANGUAGE_HLSL;
        slangPassThrough = SLANG_PASS_THROUGH_FXC;

        if (options.useDXIL)
        {
            input.target = SLANG_DXIL;
            input.profile = "sm_6_5";
            slangPassThrough = SLANG_PASS_THROUGH_DXC;
        }
        break;

    case DeviceType::Vulkan:
        input.target = SLANG_SPIRV;
        input.profile = "";
        nativeLanguage = SLANG_SOURCE_LANGUAGE_GLSL;
        slangPassThrough = SLANG_PASS_THROUGH_GLSLANG;
        break;
    case DeviceType::Metal:
        input.target = SLANG_METAL_LIB;
        input.profile = "";
        nativeLanguage = SLANG_SOURCE_LANGUAGE_METAL;
        slangPassThrough = SLANG_PASS_THROUGH_METAL;
        break;
    case DeviceType::CPU:
        input.target = SLANG_SHADER_HOST_CALLABLE;
        input.profile = "";
        nativeLanguage = SLANG_SOURCE_LANGUAGE_CPP;
        slangPassThrough = SLANG_PASS_THROUGH_GENERIC_C_CPP;
        break;
    case DeviceType::CUDA:
        input.target = SLANG_PTX;
        input.profile = "";
        nativeLanguage = SLANG_SOURCE_LANGUAGE_CUDA;
        slangPassThrough = SLANG_PASS_THROUGH_NVRTC;
        break;
    case DeviceType::WGPU:
        input.target = SLANG_WGSL;
        input.profile = "";
        nativeLanguage = SLANG_SOURCE_LANGUAGE_WGSL;
        slangPassThrough = SLANG_PASS_THROUGH_NONE;
        break;

    default:
        fprintf(stderr, "error: unexpected\n");
        return SLANG_FAIL;
    }

    switch (options.inputLanguageID)
    {
    case Options::InputLanguageID::Slang:
        input.sourceLanguage = SLANG_SOURCE_LANGUAGE_SLANG;
        input.passThrough = SLANG_PASS_THROUGH_NONE;
        break;

    case Options::InputLanguageID::Native:
        input.sourceLanguage = nativeLanguage;
        input.passThrough = slangPassThrough;
        break;

    default:
        break;
    }

    if (options.sourceLanguage != SLANG_SOURCE_LANGUAGE_UNKNOWN)
    {
        input.sourceLanguage = options.sourceLanguage;

        if (input.sourceLanguage == SLANG_SOURCE_LANGUAGE_C ||
            input.sourceLanguage == SLANG_SOURCE_LANGUAGE_CPP)
        {
            input.passThrough = SLANG_PASS_THROUGH_GENERIC_C_CPP;
        }
    }

    StdWritersDebugCallback debugCallback;
    debugCallback.writers = stdWriters;

    // Use the profile name set on options if set
    input.profile = options.profileName.getLength() ? options.profileName : input.profile;

    StringBuilder rendererName;
    auto info = rendererName << "[" << getRHI()->getDeviceTypeName(options.deviceType) << "] ";

    if (options.onlyStartup)
    {
        switch (options.deviceType)
        {
        case DeviceType::CUDA:
            {
#if RENDER_TEST_CUDA
                if (SLANG_FAILED(
                        spSessionCheckPassThroughSupport(session, SLANG_PASS_THROUGH_NVRTC)))
                    return SLANG_FAIL;
#else
                return SLANG_FAIL;
#endif
            }
        case DeviceType::CPU:
            {
                // As long as we have CPU, then this should work
                return spSessionCheckPassThroughSupport(session, SLANG_PASS_THROUGH_GENERIC_C_CPP);
            }
        default:
            break;
        }
    }

    Index nvapiExtnSlot = -1;

    // Let's see if we need to set up special prelude for HLSL
    if (options.nvapiExtnSlot.getLength() && options.nvapiExtnSlot[0] == 'u')
    {
        //
        Slang::Int value;
        UnownedStringSlice slice = options.nvapiExtnSlot.getUnownedSlice();
        UnownedStringSlice indexText(slice.begin() + 1, slice.end());
        if (SLANG_SUCCEEDED(StringUtil::parseInt(indexText, value)))
        {
            nvapiExtnSlot = Index(value);
        }
    }

    // If can't set up a necessary prelude make not available (which will lead to the test being
    // ignored)
    if (SLANG_FAILED(_setSessionPrelude(options, argvIn[0], session)))
    {
        return SLANG_E_NOT_AVAILABLE;
    }

    Slang::ComPtr<IDevice> device;
    {
        DeviceDesc desc = {};
        desc.deviceType = options.deviceType;

#if _DEBUG
        desc.enableValidation = true;
        desc.debugCallback = &debugCallback;
#endif

        desc.slang.lineDirectiveMode = SLANG_LINE_DIRECTIVE_MODE_NONE;
        if (options.generateSPIRVDirectly)
            desc.slang.targetFlags = SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY;
        else
            desc.slang.targetFlags = 0;

        List<const char*> requiredFeatureList;
        for (auto& name : options.renderFeatures)
            requiredFeatureList.add(name.getBuffer());

        desc.requiredFeatures = requiredFeatureList.getBuffer();
        desc.requiredFeatureCount = (int)requiredFeatureList.getCount();

#if defined(_WIN32)
        // When the experimental feature is enabled, things become unstable.
        // It is enabled only when requested.
        D3D12ExperimentalFeaturesDesc experimentalFD = {};
        UUID features[1] = {D3D12ExperimentalShaderModels};
        experimentalFD.featureCount = 1;
        experimentalFD.featureIIDs = features;
        experimentalFD.configurationStructs = nullptr;
        experimentalFD.configurationStructSizes = nullptr;

        if (options.dx12Experimental)
            desc.next = &experimentalFD;
#endif

        // Look for args going to slang
        {
            const auto& args = options.downstreamArgs.getArgsByName("slang");
            for (const auto& arg : args)
            {
                if (arg.value == "-matrix-layout-column-major")
                {
                    desc.slang.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;
                    break;
                }
            }
        }

        desc.nvapiExtUavSlot = uint32_t(nvapiExtnSlot);
        desc.slang.slangGlobalSession = session;
        desc.slang.targetProfile = options.profileName.getBuffer();
        {
            if (options.enableDebugLayers)
            {
                getRHI()->enableDebugLayers();
            }
            SlangResult res = getRHI()->createDevice(desc, device.writeRef());
            if (SLANG_FAILED(res))
            {
                // We need to be careful here about SLANG_E_NOT_AVAILABLE. This return value means
                // that the renderer couldn't be created because it required *features* that were
                // *not available*. It does not mean the renderer in general couldn't be
                // constructed.
                //
                // Returning SLANG_E_NOT_AVAILABLE will lead to the test infrastructure ignoring
                // this test.
                //
                // We also don't want to output the 'Unable to create renderer' error, as this isn't
                // an error.
                if (res == SLANG_E_NOT_AVAILABLE)
                {
                    return res;
                }

                if (!options.onlyStartup)
                {
                    fprintf(stderr, "Unable to create renderer %s\n", rendererName.getBuffer());
                }

                return res;
            }
            SLANG_ASSERT(device);
        }

        for (const auto& feature : requiredFeatureList)
        {
            // If doesn't have required feature... we have to give up
            if (!device->hasFeature(feature))
            {
                return SLANG_E_NOT_AVAILABLE;
            }
        }
    }

    // Print adapter info after device creation but before any other operations
    if (options.showAdapterInfo)
    {
        auto info = device->getDeviceInfo();
        auto out = stdWriters->getOut();
        out.print("Using graphics adapter: %s\n", info.adapterName);
    }

    // If the only test is we can startup, then we are done
    if (options.onlyStartup)
    {
        return SLANG_OK;
    }

    {
        RenderTestApp app;
        renderDocBeginFrame();
        SLANG_RETURN_ON_FAIL(app.initialize(session, device, options, input));
        app.update();
        renderDocEndFrame();
        app.finalize();
    }
    return SLANG_OK;
}

SLANG_TEST_TOOL_API SlangResult innerMain(
    Slang::StdWriters* stdWriters,
    SlangSession* sharedSession,
    int inArgc,
    const char* const* inArgv)
{
    using namespace Slang;

    // Assume we will used the shared session
    ComPtr<slang::IGlobalSession> session(sharedSession);

    // The sharedSession always has a pre-loaded core module.
    // This differed test checks if the command line has an option to setup the core module.
    // If so we *don't* use the sharedSession, and create a new session without the core module just
    // for this compilation.
    if (TestToolUtil::hasDeferredCoreModule(Index(inArgc - 1), inArgv + 1))
    {
        SLANG_RETURN_ON_FAIL(
            slang_createGlobalSessionWithoutCoreModule(SLANG_API_VERSION, session.writeRef()));
    }

    SlangResult res = SLANG_FAIL;
    try
    {
        res = _innerMain(stdWriters, session, inArgc, inArgv);
    }
    catch (const Slang::Exception& exception)
    {
        stdWriters->getOut().put(exception.Message.getUnownedSlice());
        return SLANG_FAIL;
    }
    catch (...)
    {
        stdWriters->getOut().put(UnownedStringSlice::fromLiteral("Unhandled exception"));
        return SLANG_FAIL;
    }

    return res;
}

int main(int argc, char** argv)
{
    using namespace Slang;
    SlangSession* session = spCreateSession(nullptr);

    TestToolUtil::setSessionDefaultPreludeFromExePath(argv[0], session);

    auto stdWriters = StdWriters::initDefaultSingleton();

    SlangResult res = innerMain(stdWriters, session, argc, argv);
    spDestroySession(session);

    slang::shutdown();
    return (int)TestToolUtil::getReturnCode(res);
}
