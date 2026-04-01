// main.cpp

// This file provides the application code for the `shader-toy` example.
//
// Much of the logic here is identical to the simpler `hello-world` example,
// so we will not spend time commenting those parts that are identical or
// nearly identical. Readers who want detailed comments on a simpler example
// using Slang should look there.

// This example uses the Slang C/C++ API, alonmg with its optional type
// for managing COM-style reference-counted pointers.
//
#include "slang-com-ptr.h"
#include "slang.h"
using Slang::ComPtr;

// This example uses a graphics API abstraction layer that is implemented inside
// the Slang codebase for use in our sample programs and test cases. Use of
// this layer is *not* required or assumed when using the Slang language,
// compiler, and API.
//
#include "core/slang-basic.h"
#include "examples/example-base/example-base.h"
#include "gfx-util/shader-cursor.h"
#include "platform/performance-counter.h"
#include "platform/window.h"
#include "slang-gfx.h"

#include <chrono>

static const ExampleResources resourceBase("shader-toy");

using namespace gfx;

// In order to display a shader toy effect using rasterization-based shader
// execution we need to render a full-screen triangle. We will define a
// small helper type that defines the data for such a triangle.
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

// The application itself will be encapsulated in a C++ `struct` type
// so that it can easily scope its state without use of global variables.
//
struct ShaderToyApp : public WindowedAppBase
{

    // The uniform data used by the shader is defined here as a simple
    // POD ("plain old data") type.
    //
    // Note: This type must match the declaration of `ShaderToyUniforms`
    // in the file `shader-toy.slang`.
    //
    // An application could instead use a shared header file to define
    // this type, or use Slang's reflection capabilities to allocate
    // and set parameters at runtime. For this simple example we did
    // the expedient thing of having distinct Slang and C++ declarations.
    //
    struct Uniforms
    {
        float iMouse[4];
        float iResolution[2];
        float iTime;
    };

    // The main interesting part of the host application code is where we
    // load, compile, inspect, and compose the Slang shader code.
    //
    Result loadShaderProgram(gfx::IDevice* device, ComPtr<gfx::IShaderProgram>& outShaderProgram)
    {
        // We need to obatin a compilation session (`slang::ISession`) that will provide
        // a scope to all the compilation and loading of code we do.
        //
        // Our example application uses the `gfx` graphics API abstraction layer, which already
        // creates a Slang compilation session for us, so we just grab and use it here.
        ComPtr<slang::ISession> slangSession;
        SLANG_RETURN_ON_FAIL(device->getSlangSession(slangSession.writeRef()));

        // Once the session has been obtained, we can start loading code into it.
        //
        // The simplest way to load code is by calling `loadModule` with the name of a Slang
        // module. A call to `loadModule("MyStuff")` will behave more or less as if you
        // wrote:
        //
        //      import MyStuff;
        //
        // In a Slang shader file. The compiler will use its search paths to try to locate
        // `MyModule.slang`, then compile and load that file. If a matching module had
        // already been loaded previously, that would be used directly.
        //
        // Note: The only interesting wrinkle here is that our file is named `shader-toy` with
        // a hyphen in it, so the name is not directly usable as an identifier in Slang code.
        // Instead, when trying to import this module in the context of Slang code, a user
        // needs to replace the hyphens with underscores:
        //
        //      import shader_toy;
        //
        ComPtr<slang::IBlob> diagnosticsBlob;
        Slang::String shaderToyPath = resourceBase.resolveResource("shader-toy.slang");
        slang::IModule* module =
            slangSession->loadModule(shaderToyPath.getBuffer(), diagnosticsBlob.writeRef());
        diagnoseIfNeeded(diagnosticsBlob);
        if (!module)
            return SLANG_FAIL;

        // Loading the `shader-toy` module will compile and check all the shader code in it,
        // including the shader entry points we want to use. Now that the module is loaded
        // we can look up those entry points by name.
        //
        // Note: If you are using this `loadModule` approach to load your shader code it is
        // important to tag your entry point functions with the `[shader("...")]` attribute
        // (e.g., `[shader("vertex")] void vertexMain(...)`). Without that information there
        // is no umambiguous way for the compiler to know which functions represent entry
        // points when it parses your code via `loadModule()`.
        //
        char const* vertexEntryPointName = "vertexMain";
        char const* fragmentEntryPointName = "fragmentMain";
        //
        ComPtr<slang::IEntryPoint> vertexEntryPoint;
        SLANG_RETURN_ON_FAIL(
            module->findEntryPointByName(vertexEntryPointName, vertexEntryPoint.writeRef()));
        //
        ComPtr<slang::IEntryPoint> fragmentEntryPoint;
        SLANG_RETURN_ON_FAIL(
            module->findEntryPointByName(fragmentEntryPointName, fragmentEntryPoint.writeRef()));

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
        componentTypes.add(module);

        // Later on when we go to extract compiled kernel code for our vertex
        // and fragment shaders, we will need to make use of their order within
        // the composition, so we will record the relative ordering of the entry
        // points here as we add them.
        int entryPointCount = 0;
        int vertexEntryPointIndex = entryPointCount++;
        componentTypes.add(vertexEntryPoint);

        int fragmentEntryPointIndex = entryPointCount++;
        componentTypes.add(fragmentEntryPoint);

        // Actually creating the composite component type is a single operation
        // on the Slang session, but the operation could potentially fail if
        // something about the composite was invalid (e.g., you are trying to
        // combine multiple copies of the same module), so we need to deal
        // with the possibility of diagnostic output.
        //
        ComPtr<slang::IComponentType> composedProgram;
        SlangResult result = slangSession->createCompositeComponentType(
            componentTypes.getBuffer(),
            componentTypes.getCount(),
            composedProgram.writeRef(),
            diagnosticsBlob.writeRef());
        diagnoseIfNeeded(diagnosticsBlob);
        SLANG_RETURN_ON_FAIL(result);

        // At this point, `composedProgram` represents the shader program
        // we want to run, and the vertex and fragment shader there have
        // been checked.
        //
        // We could use the Slang reflection API on `composedProgram` at this
        // point to query things like the locations and offsets of the
        // various uniform parameters, textures, etc.
        //
        // What *cannot* be done yet at this point is actually generating
        // kernel code, because `composedProgram` includes a generic type
        // parameter as part of the `fragmentMain` entry point:
        //
        //      void fragmentMain<T : IShaderToyImageShader>(...)
        //
        // Our next task is to load code for a type we'd like to plug in
        // for `T` there.
        //
        // Because Slang supports modular programming, there is no requirement
        // that a type we want to plug in for `T` has to come from the
        // same module, and to demonstrate that we will load a different
        // module to provide the effect type we will plug in.
        //
        const char* effectTypeName = "ExampleEffect";
        Slang::String effectModulePath = resourceBase.resolveResource("example-effect.slang");
        slang::IModule* effectModule =
            slangSession->loadModule(effectModulePath.getBuffer(), diagnosticsBlob.writeRef());
        diagnoseIfNeeded(diagnosticsBlob);
        if (!module)
            return SLANG_FAIL;

        // Once we've loaded the code module that defines out effect type,
        // we can look it up by name using the reflection information on
        // the module.
        //
        // Note: A future version of the Slang API will support enumerating
        // the types declared in a module so that we do not have to hard-code
        // the name here.
        //
        auto effectType = effectModule->getLayout()->findTypeByName(effectTypeName);

        // Now that we have the `effectType` we want to plug in to our generic
        // shader, we need to specialize the shader to that type.
        //
        // Because a shader program could have zero or more specialization parameters,
        // we need to build up an array of specialization arguments.
        //
        Slang::List<slang::SpecializationArg> specializationArgs;

        {
            // In our case, we only have a single specialization argument we plan
            // to use, and it is a type argument.
            //
            slang::SpecializationArg effectTypeArg;
            effectTypeArg.kind = slang::SpecializationArg::Kind::Type;
            effectTypeArg.type = effectType;
            specializationArgs.add(effectTypeArg);
        }

        // Specialization of a component type is a single Slang API call, but
        // we need to deal with the possibility of diagnostic output on failure.
        // For example, if we tried to specialize the shader program to a
        // type like `int` that doesn't support the `IShaderToyImageShader` interface,
        // this is the step where we'd get an error message saying so.
        //
        ComPtr<slang::IComponentType> specializedProgram;
        result = composedProgram->specialize(
            specializationArgs.getBuffer(),
            specializationArgs.getCount(),
            specializedProgram.writeRef(),
            diagnosticsBlob.writeRef());
        diagnoseIfNeeded(diagnosticsBlob);
        SLANG_RETURN_ON_FAIL(result);

        // At this point we have a specialized shader program that represents our
        // intention to run the `vertexMain` and `fragmentMain` entry points,
        // specialized to the `ExampleEffect` type we loaded.
        //
        // We can now *link* the program, which ensures that all of the code that
        // it transitively depends on has been pulled together into a single
        // component type.
        //
        ComPtr<slang::IComponentType> linkedProgram;
        result = specializedProgram->link(linkedProgram.writeRef(), diagnosticsBlob.writeRef());
        diagnoseIfNeeded(diagnosticsBlob);
        SLANG_RETURN_ON_FAIL(result);

        gfx::IShaderProgram::Desc programDesc = {};
        programDesc.slangGlobalScope = linkedProgram.get();
        auto shaderProgram = device->createProgram(programDesc);
        outShaderProgram = shaderProgram;
        return SLANG_OK;
    }

    ComPtr<IShaderProgram> gShaderProgram;
    ComPtr<gfx::IPipelineState> gPipelineState;
    ComPtr<gfx::IBufferResource> gVertexBuffer;

    Result initialize()
    {
        SLANG_RETURN_ON_FAIL(initializeBase("Shader Toy", 1024, 768));

        // We may not have a window if we're running in test mode
        SLANG_ASSERT(isTestMode() || gWindow);
        if (gWindow)
        {
            gWindow->events.mouseMove = [this](const platform::MouseEventArgs& e)
            { handleEvent(e); };
            gWindow->events.mouseUp = [this](const platform::MouseEventArgs& e) { handleEvent(e); };
            gWindow->events.mouseDown = [this](const platform::MouseEventArgs& e)
            { handleEvent(e); };
        }

        InputElementDesc inputElements[] = {
            {"POSITION", 0, Format::R32G32_FLOAT, offsetof(FullScreenTriangle::Vertex, position)},
        };
        auto inputLayout = gDevice->createInputLayout(
            sizeof(FullScreenTriangle::Vertex),
            &inputElements[0],
            SLANG_COUNT_OF(inputElements));
        if (!inputLayout)
            return SLANG_FAIL;

        IBufferResource::Desc vertexBufferDesc;
        vertexBufferDesc.type = IResource::Type::Buffer;
        vertexBufferDesc.sizeInBytes =
            FullScreenTriangle::kVertexCount * sizeof(FullScreenTriangle::Vertex);
        vertexBufferDesc.defaultState = ResourceState::VertexBuffer;
        gVertexBuffer =
            gDevice->createBufferResource(vertexBufferDesc, &FullScreenTriangle::kVertices[0]);
        if (!gVertexBuffer)
            return SLANG_FAIL;

        SLANG_RETURN_ON_FAIL(loadShaderProgram(gDevice, gShaderProgram));

        // Create pipeline.
        GraphicsPipelineStateDesc desc;
        desc.inputLayout = inputLayout;
        desc.program = gShaderProgram;
        desc.framebufferLayout = gFramebufferLayout;
        auto pipelineState = gDevice->createGraphicsPipelineState(desc);
        if (!pipelineState)
            return SLANG_FAIL;

        gPipelineState = pipelineState;

        return SLANG_OK;
    }

    bool wasMouseDown = false;
    bool isMouseDown = false;
    float lastMouseX = 0.0f;
    float lastMouseY = 0.0f;
    float clickMouseX = 0.0f;
    float clickMouseY = 0.0f;

    bool firstTime = true;
    platform::TimePoint startTime;

    virtual void renderFrame(int frameIndex) override
    {
        auto commandBuffer = gTransientHeaps[frameIndex]->createCommandBuffer();
        if (firstTime)
        {
            startTime = platform::PerformanceCounter::now();
            firstTime = false;
        }

        // Update uniform buffer.

        Uniforms uniforms = {};
        {
            bool isMouseClick = isMouseDown && !wasMouseDown;
            wasMouseDown = isMouseDown;

            if (isMouseClick)
            {
                clickMouseX = lastMouseX;
                clickMouseY = lastMouseY;
            }

            uniforms.iMouse[0] = lastMouseX;
            uniforms.iMouse[1] = lastMouseY;
            uniforms.iMouse[2] = isMouseDown ? clickMouseX : -clickMouseX;
            uniforms.iMouse[3] = isMouseClick ? clickMouseY : -clickMouseY;
            uniforms.iTime = platform::PerformanceCounter::getElapsedTimeInSeconds(startTime);
            uniforms.iResolution[0] = float(windowWidth);
            uniforms.iResolution[1] = float(windowHeight);
        }

        // Encode render commands.
        auto encoder = commandBuffer->encodeRenderCommands(gRenderPass, gFramebuffers[frameIndex]);

        gfx::Viewport viewport = {};
        viewport.maxZ = 1.0f;
        viewport.extentX = (float)windowWidth;
        viewport.extentY = (float)windowHeight;
        encoder->setViewportAndScissor(viewport);
        auto rootObject = encoder->bindPipeline(gPipelineState);
        auto constantBuffer = rootObject->getObject(ShaderOffset());
        constantBuffer->setData(ShaderOffset(), &uniforms, sizeof(uniforms));

        encoder->setVertexBuffer(0, gVertexBuffer);
        encoder->setPrimitiveTopology(PrimitiveTopology::TriangleList);
        encoder->draw(3);
        encoder->endEncoding();
        commandBuffer->close();

        gQueue->executeCommandBuffer(commandBuffer);

        // We may not have a swapchain if we're running in test mode
        SLANG_ASSERT(isTestMode() || gSwapchain);
        if (gSwapchain)
            gSwapchain->present();
    }

    void handleEvent(const platform::MouseEventArgs& event)
    {
        isMouseDown = ((int)event.buttons & (int)platform::ButtonState::Enum::LeftButton) != 0;
        lastMouseX = (float)event.x;
        lastMouseY = (float)event.y;
    }
};

// This macro instantiates an appropriate main function to
// run the application defined above.
EXAMPLE_MAIN(innerMain<ShaderToyApp>);
