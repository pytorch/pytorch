// This example is out of date and currently disabled from build.
// The `gfx` layer has been refactored with a new shader-object model
// that will greatly simplify shader binding and specialization.
// This example should be updated to use the shader-object API in `gfx`.

// main.cpp

//
// This example is much more involved than the `hello-world` example,
// so readers are encouraged to work through the simpler code first
// before diving into this application. We will gloss over parts of
// the code that are similar to the code in `hello-world`, and
// instead focus on the new code that is required to use Slang in
// more advanced ways.
//

// We still need to include the Slang header to use the Slang API
//
#include "slang-com-helper.h"
#include "slang.h"

// We will again make use of a graphics API abstraction
// layer that implements the shader-object idiom based on Slang's
// `ParameterBlock` and `interface` features to simplify shader specialization
// and parameter binding.
//
#include "examples/example-base/example-base.h"
#include "gfx-util/shader-cursor.h"
#include "platform/gui.h"
#include "platform/model.h"
#include "platform/vector-math.h"
#include "platform/window.h"
#include "slang-gfx.h"

#include <map>
#include <sstream>

using namespace gfx;
using Slang::RefObject;
using Slang::RefPtr;

static const ExampleResources resourceBase("model-viewer");

struct RendererContext
{
    IDevice* device;
    slang::IModule* shaderModule;
    slang::ShaderReflection* slangReflection;
    ComPtr<IShaderProgram> shaderProgram;

    slang::TypeReflection* perViewShaderType;
    slang::TypeReflection* perModelShaderType;

    TestBase* pTestBase;

    Result init(IDevice* inDevice, TestBase* inTestBase)
    {
        device = inDevice;
        ComPtr<ISlangBlob> diagnostic;
        pTestBase = inTestBase;

        Slang::String path = resourceBase.resolveResource("shaders.slang").getBuffer();
        shaderModule =
            device->getSlangSession()->loadModule(path.getBuffer(), diagnostic.writeRef());
        diagnoseIfNeeded(diagnostic);
        if (!shaderModule)
            return SLANG_FAIL;

        // Compose the shader program for drawing models by combining the shader module
        // and entry points ("vertexMain" and "fragmentMain").
        char const* vertexEntryPointName = "vertexMain";
        ComPtr<slang::IEntryPoint> vertexEntryPoint;
        SLANG_RETURN_ON_FAIL(
            shaderModule->findEntryPointByName(vertexEntryPointName, vertexEntryPoint.writeRef()));

        char const* fragEntryPointName = "fragmentMain";
        ComPtr<slang::IEntryPoint> fragEntryPoint;
        SLANG_RETURN_ON_FAIL(
            shaderModule->findEntryPointByName(fragEntryPointName, fragEntryPoint.writeRef()));

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
        componentTypes.add(shaderModule);
        componentTypes.add(vertexEntryPoint);
        componentTypes.add(fragEntryPoint);

        // Actually creating the composite component type is a single operation
        // on the Slang session, but the operation could potentially fail if
        // something about the composite was invalid (e.g., you are trying to
        // combine multiple copies of the same module), so we need to deal
        // with the possibility of diagnostic output.
        //
        ComPtr<slang::IComponentType> composedProgram;
        ComPtr<ISlangBlob> diagnosticsBlob;
        SlangResult result = device->getSlangSession()->createCompositeComponentType(
            componentTypes.getBuffer(),
            componentTypes.getCount(),
            composedProgram.writeRef(),
            diagnosticsBlob.writeRef());
        diagnoseIfNeeded(diagnosticsBlob);
        SLANG_RETURN_ON_FAIL(result);

        if (pTestBase && pTestBase->isTestMode())
        {
            pTestBase->printEntrypointHashes(componentTypes.getCount() - 1, 1, composedProgram);
        }

        slangReflection = composedProgram->getLayout();

        // At this point, `composedProgram` represents the shader program
        // we want to run, and the compute shader there have been checked.
        // We can create a `gfx::IShaderProgram` object from `composedProgram`
        // so it may be used by the graphics layer.
        gfx::IShaderProgram::Desc programDesc = {};
        programDesc.slangGlobalScope = composedProgram.get();

        shaderProgram = device->createProgram(programDesc);

        // Get other shader types that we will use for creating shader objects.
        perViewShaderType = slangReflection->findTypeByName("PerView");
        perModelShaderType = slangReflection->findTypeByName("PerModel");

        return SLANG_OK;
    }
};

// Our application code has a rudimentary material system,
// to match the `IMaterial` abstraction used in the shade code.
//
struct Material : RefObject
{
    // The key feature of a matrial in our application is that
    // it can provide a shader object that describes it and
    // its parameters. The contents of the shader object will
    // be any colors, textures, etc. that the material needs,
    // while the Slang type that was used to allocate the
    // block will be an implementation of `IMaterial` that
    // provides the evaluation logic for the material.

    // Each subclass of `Material` will provide a routine to
    // create a shader object that stores its shader parameters.
    virtual IShaderObject* createShaderObject(RendererContext* context) = 0;

    // The shader object for a material will be stashed here
    // after it is created.
    ComPtr<IShaderObject> shaderObject;
};

// For now we have only a single implementation of `Material`,
// which corresponds to the `SimpleMaterial` type in our shader
// code.
//
struct SimpleMaterial : Material
{
    glm::vec3 diffuseColor;
    glm::vec3 specularColor;
    float specularity = 1.0f;

    // Create a shader object that contains the type info and parameter values
    // that represent an instance of `SimpleMaterial`.
    IShaderObject* createShaderObject(RendererContext* context) override
    {
        auto program = context->slangReflection;
        auto shaderType = program->findTypeByName("SimpleMaterial");
        shaderObject = context->device->createShaderObject(shaderType);
        gfx::ShaderCursor cursor(shaderObject);
        cursor["diffuseColor"].setData(&diffuseColor, sizeof(diffuseColor));
        cursor["specularColor"].setData(&specularColor, sizeof(specularColor));
        cursor["specularity"].setData(&specularity, sizeof(specularity));
        return shaderObject.get();
    }
};

// With the `Material` abstraction defined, we can go on to define
// the representation for loaded models that we will use.
//
// A `Model` will own vertex/index buffers, along with a list of meshes,
// while each `Mesh` will own a material and a range of indices.
// For this example we will be loading models from `.obj` files, but
// that is just a simple lowest-common-denominator choice.
//
struct Mesh : RefObject
{
    RefPtr<Material> material;
    int firstIndex;
    int indexCount;
};
struct Model : RefObject
{
    typedef platform::ModelLoader::Vertex Vertex;

    ComPtr<IBufferResource> vertexBuffer;
    ComPtr<IBufferResource> indexBuffer;
    PrimitiveTopology primitiveTopology;
    int vertexCount;
    int indexCount;
    std::vector<RefPtr<Mesh>> meshes;
};
//
// Loading a model from disk is done with the help of some utility
// code for parsing the `.obj` file format, so that the application
// mostly just registers some callbacks to allocate the objects
// used for its representation.
//
RefPtr<Model> loadModel(
    RendererContext* context,
    char const* inputPath,
    platform::ModelLoader::LoadFlags loadFlags = 0,
    float scale = 1.0f)
{
    // The model loading interface using a C++ interface of
    // callback functions to handle creating the application-specific
    // representation of meshes, materials, etc.
    //
    struct Callbacks : platform::ModelLoader::ICallbacks
    {
        RendererContext* context;
        // Hold a reference to all material and mesh objects
        // created during loading so that they can be properly
        // freed.
        std::vector<RefPtr<Material>> materials;
        std::vector<RefPtr<Mesh>> meshes;
        void* createMaterial(MaterialData const& data) override
        {
            SimpleMaterial* material = new SimpleMaterial();
            material->diffuseColor = data.diffuseColor;
            material->specularColor = data.specularColor;
            material->specularity = data.specularity;
            material->createShaderObject(context);
            materials.push_back(material);
            return material;
        }

        void* createMesh(MeshData const& data) override
        {
            Mesh* mesh = new Mesh();
            mesh->firstIndex = data.firstIndex;
            mesh->indexCount = data.indexCount;
            mesh->material = (Material*)data.material;
            meshes.push_back(mesh);
            return mesh;
        }

        void* createModel(ModelData const& data) override
        {
            Model* model = new Model();
            model->vertexBuffer = data.vertexBuffer;
            model->indexBuffer = data.indexBuffer;
            model->primitiveTopology = data.primitiveTopology;
            model->vertexCount = data.vertexCount;
            model->indexCount = data.indexCount;

            int meshCount = data.meshCount;
            for (int ii = 0; ii < meshCount; ++ii)
                model->meshes.push_back((Mesh*)data.meshes[ii]);

            return model;
        }
    };
    Callbacks callbacks;
    callbacks.context = context;

    // We instantiate a model loader object and then use it to
    // try and load a model from the chosen path.
    //
    platform::ModelLoader loader;
    loader.device = context->device;
    loader.loadFlags = loadFlags;
    loader.scale = scale;
    loader.callbacks = &callbacks;
    Model* model = nullptr;
    if (SLANG_FAILED(loader.load(inputPath, (void**)&model)))
    {
        log("failed to load '%s'\n", inputPath);
        return nullptr;
    }

    return model;
}

// Along with materials, our application needs to be able to represent
// multiple light sources in the scene. For this task we will use a C++
// inheritance hierarchy rooted at `Light` to match the `ILight`
// interface in Slang.

struct Light : RefObject
{
    // A light must be able to write its state into a shader parameters
    // of the matching Slang type.
    //
    virtual void writeTo(ShaderCursor const& cursor) = 0;

    // Retrieves the shader type for this light object.
    virtual slang::TypeReflection* getShaderType(RendererContext* context) = 0;

    // The shader object for a light will be stashed here
    // after it is created.
    //    ComPtr<IShaderObject> shaderObject;
};

// Helper function to retrieve the underlying shader type of `T`.
template<typename T>
slang::TypeReflection* getShaderType(RendererContext* context)
{
    auto program = context->slangReflection;
    auto shaderType = program->findTypeByName(T::getTypeName());
    return shaderType;
}

// We will provide two nearly trivial implementations of `Light` for now,
// to show the kind of application code needed to line up with the corresponding
// types defined in the Slang shader code for this application.

struct DirectionalLight : Light
{
    glm::vec3 direction = normalize(glm::vec3(1));
    glm::vec3 intensity = glm::vec3(1);

    static const char* getTypeName() { return "DirectionalLight"; }

    virtual void writeTo(ShaderCursor const& cursor) override
    {
        cursor["direction"].setData(&direction, sizeof(direction));
        cursor["intensity"].setData(&intensity, sizeof(intensity));
    }

    virtual slang::TypeReflection* getShaderType(RendererContext* context) override
    {
        return ::getShaderType<DirectionalLight>(context);
    }
};

struct PointLight : Light
{
    glm::vec3 position = glm::vec3(0);
    glm::vec3 intensity = glm::vec3(1);

    static const char* getTypeName() { return "PointLight"; }

    virtual void writeTo(ShaderCursor const& cursor) override
    {
        cursor["position"].setData(&position, sizeof(position));
        cursor["intensity"].setData(&intensity, sizeof(intensity));
    }

    virtual slang::TypeReflection* getShaderType(RendererContext* context) override
    {
        return ::getShaderType<PointLight>(context);
    }
};

// Rendering is usually done with collections of lights rather than single
// lights. This application will use a concept of "light environments" to
// group together lights for rendering.
//
// We want to be *able* to specialize our shader code based on the particular
// types of lights in a scene, but we also do not want to over-specialize
// and, e.g., use differnt specialized shaders for a scene with 99 point
// lights vs. 100.
//
// This particular application will use a notion of a "layout" for a lighting
// environment, which specifies the allowed types of lights, and the maximum
// number of lights of each type. Different lighting environment layouts
// will yield different specialized code.

struct LightEnvLayout : public RefObject
{
    // Our lighting environment layout will track layout
    // information for several different arrays: one
    // for each supported light type.
    //
    struct LightArrayLayout : RefObject
    {
        Int maximumCount = 0;
        std::string typeName;
    };
    std::vector<LightArrayLayout> lightArrayLayouts;
    std::map<slang::TypeReflection*, Int> mapLightTypeToArrayIndex;
    slang::TypeReflection* shaderType = nullptr;

    void addLightType(RendererContext* context, slang::TypeReflection* lightType, Int maximumCount)
    {
        Int arrayIndex = (Int)lightArrayLayouts.size();
        LightArrayLayout layout;
        layout.maximumCount = maximumCount;

        // When the user adds a light type `X` to a light-env layout,
        // we need to compute the corresponding Slang type and
        // layout information to use. If only a single light is
        // supported, this will just be the type `X`, while for
        // any other count this will be a `LightArray<X, maximumCount>`
        //
        if (maximumCount <= 1)
        {
            layout.typeName = lightType->getName();
        }
        else
        {
            auto program = context->slangReflection;
            std::stringstream typeNameBuilder;
            typeNameBuilder << "LightArray<" << lightType->getName() << "," << maximumCount << ">";
            layout.typeName = typeNameBuilder.str();
        }

        lightArrayLayouts.push_back(layout);
        mapLightTypeToArrayIndex.insert(std::make_pair(lightType, arrayIndex));
    }

    template<typename T>
    void addLightType(RendererContext* context, Int maximumCount)
    {
        addLightType(context, getShaderType<T>(context), maximumCount);
    }

    Int getArrayIndexForType(slang::TypeReflection* lightType)
    {
        auto iter = mapLightTypeToArrayIndex.find(lightType);
        if (iter != mapLightTypeToArrayIndex.end())
            return iter->second;

        return -1;
    }
};

// A `LightEnv` follows the structure of a `LightEnvLayout`,
// and provides storage for zero or more lights of various
// different types (up to the limits imposed by the layout).
//
struct LightEnv : public RefObject
{
    // A light environment is always created from a fixed layout
    // in this application, so the constructor allocates an array
    // for the per-light-type data.
    //
    // A more complex example might dynamically determine the
    // layout based on the number of lights of each type active
    // in the scene, with some quantization applied to avoid
    // generating too many shader specializations.
    //
    // Note: the kind of specialization going on here would also
    // be applicable to a deferred or "forward+" renderer, insofar
    // as it sets the bounds on the total set of lights for
    // a scene/frame, while per-tile/-cluster light lists would
    // probably just be indices into the global structure.
    //
    RefPtr<LightEnvLayout> layout;
    RendererContext* context;
    LightEnv(RefPtr<LightEnvLayout> layout, RendererContext* inContext)
        : layout(layout), context(inContext)
    {
        for (auto arrayLayout : layout->lightArrayLayouts)
        {
            RefPtr<LightArray> lightArray = new LightArray();
            lightArray->layout = arrayLayout;
            lightArrays.push_back(lightArray);
        }
    }

    // For each light type, we track the layout information,
    // plus the list of active lights of that type.
    //
    struct LightArray : RefObject
    {
        LightEnvLayout::LightArrayLayout layout;
        std::vector<RefPtr<Light>> lights;
    };
    std::vector<RefPtr<LightArray>> lightArrays;

    RefPtr<LightArray> getArrayForType(slang::TypeReflection* type)
    {
        auto index = layout->getArrayIndexForType(type);
        return lightArrays[index];
    }

    void add(RefPtr<Light> light)
    {
        auto array = getArrayForType(light->getShaderType(context));
        array->lights.push_back(light);
    }

    // Get the proper shader type that represents this lighting environment.
    slang::TypeReflection* getShaderType()
    {
        // Given a lighting environment with N light types:
        //
        // L0, L1, ... LN
        //
        // We want to compute the Slang type:
        //
        // LightPair<L0, LightPair<L1, ... LightPair<LN-1, LN>>>
        //
        // This is most easily accomplished by doing a "fold" while
        // walking the array in reverse order.

        std::string currentEnvTypeName;
        auto arrayCount = layout->lightArrayLayouts.size();
        for (size_t ii = arrayCount; ii--;)
        {
            auto arrayInfo = layout->lightArrayLayouts[ii];

            if (!currentEnvTypeName.size())
            {
                // The is the right-most entry, so it is the base case for our "fold".
                currentEnvTypeName = arrayInfo.typeName;
            }
            else
            {
                // Fold one entry: `envLayout = LightPair<a, envLayout>`
                std::stringstream typeBuilder;
                typeBuilder << "LightPair<" << arrayInfo.typeName << "," << currentEnvTypeName
                            << ">";
                currentEnvTypeName = typeBuilder.str();
            }
        }

        if (!currentEnvTypeName.size())
        {
            // Handle the special case of *zero* light types.
            currentEnvTypeName = "EmptyLightEnv";
        }
        return context->slangReflection->findTypeByName(currentEnvTypeName.c_str());
    }

    // Because the lighting environment will often change between frames,
    // we will not try to optimize for the case where it doesn't change,
    // and will instead create a "transient" shader object from
    // scratch every frame.
    //
    ComPtr<IShaderObject> createShaderObject()
    {
        auto specializedType = getShaderType();

        auto shaderObject = context->device->createShaderObject(specializedType);
        ShaderCursor cursor(shaderObject);
        // When filling in the shader object for a lighting
        // environment, we mostly follow the structure of
        // the type that was computed by the `LightEnv::getShaderType`:
        //
        //      LightPair<A, LightPair<B, ... LightPair<Y, Z>>>
        //
        // we will keep `encoder` pointed at the "spine" of this
        // structure (so at an element that represents a `LightPair`,
        // except for the special case of the last item like `Z` above).
        //
        // For each light type, we will then encode the data as
        // needed for the light type (`A` then `B` then ...)
        //
        size_t lightTypeCount = lightArrays.size();
        for (size_t tt = 0; tt < lightTypeCount; ++tt)
        {
            // The encoder for the very last item will
            // just be the one on the "spine" of the list.
            auto lightTypeCursor = cursor;
            if (tt != lightTypeCount - 1)
            {
                // In the common case `encoder` is set up
                // for writing to a `LightPair<X, Y>` so
                // we ant to set up the `lightTypeEncoder`
                // for writing to an `X` (which is the first
                // field of `LightPair`, and then have
                // `encoder` move on to the `Y` (the rest
                // of the list of light types).
                //
                lightTypeCursor = cursor["first"];
                cursor = cursor["second"];
            }

            auto& lightTypeArray = lightArrays[tt];
            size_t lightCount = lightTypeArray->lights.size();
            size_t maxLightCount = lightTypeArray->layout.maximumCount;

            // Recall that we are representing the data for a single
            // light type `L` as either an instance of type `L` (if
            // only a single light is supported), or as an instance
            // of the type `LightArray<L,N>`.
            //
            if (maxLightCount == 1)
            {
                // This is the case where the maximu number of lights of
                // the given type was set as one, so we just have a value
                // of type `L`, and can tell the first light in our application-side
                // array to encode itself into that location.

                if (lightCount > 0)
                {
                    lightTypeArray->lights[0]->writeTo(lightTypeCursor);
                }
                else
                {
                    // We really ought to zero out the entry in this case
                    // (under the assumption that all zeros will represent
                    // an inactive light).
                }
            }
            else
            {
                // The more interesting case is when we have a `LightArray<L,N>`,
                // in which case we need to fill in the first field (the light count)...
                //
                int32_t lightCount = int32_t(lightTypeArray->lights.size());
                lightTypeCursor["count"].setData(&lightCount, sizeof(lightCount));
                //
                // ... followed by an array of values of type `L` in the second field.
                // We will only write to the first `lightCount` entries, which may be
                // less than `N`. We will rely on dynamic looping in the shader to
                // not access the entries past that point.
                //
                auto arrayCursor = lightTypeCursor["lights"];
                for (int32_t ii = 0; ii < lightCount; ++ii)
                {
                    lightTypeArray->lights[ii]->writeTo(arrayCursor[ii]);
                }
            }
        }
        return shaderObject;
    }
};

// Now that we've written all the required infrastructure code for
// the application's renderer and shader library, we can move on
// to the main logic.
//
// We will again structure our example application as a C++ `struct`,
// so that we can scope its allocations for easy cleanup, rather than
// use global variables.
//
struct ModelViewer : WindowedAppBase
{
    RendererContext context;

    // Most of the application state is stored in the list of loaded models,
    // as well as the active light source (a single light for now).
    //
    std::vector<RefPtr<Model>> gModels;
    RefPtr<LightEnv> lightEnv;

    // The pipeline state object we will use to draw models.
    ComPtr<IPipelineState> gPipelineState;

    // During startup the application will load one or more models and
    // add them to the `gModels` list.
    //
    void loadAndAddModel(
        char const* inputPath,
        platform::ModelLoader::LoadFlags loadFlags = 0,
        float scale = 1.0f)
    {
        auto model = loadModel(&context, inputPath, loadFlags, scale);
        if (!model)
            return;
        gModels.push_back(model);
    }

    // Our "simulation" state consists of just a few values.
    //
    uint64_t lastTime = 0;

    // glm::vec3 lightDir = normalize(glm::vec3(10, 10, 10));
    // glm::vec3 lightColor = glm::vec3(1, 1, 1);

    glm::vec3 cameraPosition = glm::vec3(1.75, 1.25, 5);
    glm::quat cameraOrientation = glm::quat(1, glm::vec3(0));

    float translationScale = 0.5f;
    float rotationScale = 0.025f;

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

            cameraOrientation =
                glm::rotate(cameraOrientation, -deltaX * rotationScale, glm::vec3(0, 1, 0));
            cameraOrientation =
                glm::rotate(cameraOrientation, -deltaY * rotationScale, glm::vec3(1, 0, 0));

            cameraOrientation = normalize(cameraOrientation);

            lastMouseX = (float)args.x;
            lastMouseY = (float)args.y;
        }
    }
    void onMouseUp(platform::MouseEventArgs args) { isMouseDown = false; }

    // The overall initialization logic is quite similar to
    // the earlier example. The biggest difference is that we
    // create instances of our application-specific parameter
    // block layout and effect types instead of just creating
    // raw graphics API objects.
    //
    Result initialize()
    {
        SLANG_RETURN_ON_FAIL(initializeBase("Model Viewer", 1024, 768));
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

        // Initialize `RendererContext`, which loads the shader module from file.
        SLANG_RETURN_ON_FAIL(context.init(gDevice, this));


        InputElementDesc inputElements[] = {
            {"POSITION", 0, Format::R32G32B32_FLOAT, offsetof(Model::Vertex, position)},
            {"NORMAL", 0, Format::R32G32B32_FLOAT, offsetof(Model::Vertex, normal)},
            {"UV", 0, Format::R32G32_FLOAT, offsetof(Model::Vertex, uv)},
        };
        auto inputLayout = gDevice->createInputLayout(sizeof(Model::Vertex), &inputElements[0], 3);
        if (!inputLayout)
            return SLANG_FAIL;

        // Create the pipeline state object for drawing models.
        GraphicsPipelineStateDesc pipelineStateDesc = {};
        pipelineStateDesc.program = context.shaderProgram;
        pipelineStateDesc.framebufferLayout = gFramebufferLayout;
        pipelineStateDesc.inputLayout = inputLayout;
        pipelineStateDesc.primitiveType = PrimitiveType::Triangle;
        pipelineStateDesc.depthStencil.depthFunc = ComparisonFunc::LessEqual;
        pipelineStateDesc.depthStencil.depthTestEnable = true;
        gPipelineState = gDevice->createGraphicsPipelineState(pipelineStateDesc);

        // We will create a lighting environment layout that can hold a few point
        // and directional lights, and then initialize a lighting environment
        // with just a single point light.
        //
        RefPtr<LightEnvLayout> lightEnvLayout = new LightEnvLayout();
        lightEnvLayout->addLightType<PointLight>(&context, 10);
        lightEnvLayout->addLightType<DirectionalLight>(&context, 2);

        lightEnv = new LightEnv(lightEnvLayout, &context);

        RefPtr<PointLight> pointLight = new PointLight();
        pointLight->position = glm::vec3(5, 3, 1);
        pointLight->intensity = glm::vec3(10);
        lightEnv->add(pointLight);

        // Once we have created all our graphcis API and application resources,
        // we can start to load models. For now we are keeping things extremely
        // simple by using a trivial `.obj` file that can be checked into source
        // control.
        //
        // Support for loading more interesting/complex models will be added
        // to this example over time (although model loading is *not* the focus).
        //
        Slang::String path = resourceBase.resolveResource("cube.obj").getBuffer();
        loadAndAddModel(path.getBuffer());

        return SLANG_OK;
    }

    // With the setup work done, we can look at the per-frame rendering
    // logic to see how the application will drive the `RenderContext`
    // type to perform both shader parameter binding and code specialization.
    //
    void renderFrame(int frameIndex) override
    {
        // In order to see that things are rendering properly we need some
        // kind of animation, so we will compute a crude delta-time value here.
        //
        if (!lastTime)
            lastTime = getCurrentTime();
        uint64_t currentTime = getCurrentTime();
        float deltaTime = float(double(currentTime - lastTime) / double(getTimerFrequency()));
        lastTime = currentTime;

        // We will use the GLM library to do the matrix math required
        // to set up our various transformation matrices.
        //
        glm::mat4x4 identity = glm::mat4x4(1.0f);

        platform::Rect clientRect{};
        if (isTestMode())
        {
            clientRect.width = 1024;
            clientRect.height = 768;
        }
        else
        {
            clientRect = getWindow()->getClientRect();
        }
        if (clientRect.height == 0)
            return;
        glm::mat4x4 projection = glm::perspectiveRH_ZO(
            glm::radians(60.0f),
            float(clientRect.width) / float(clientRect.height),
            0.1f,
            1000.0f);

        // We are implementing a *very* basic 6DOF first-person
        // camera movement model.
        //
        glm::mat3x3 cameraOrientationMat(cameraOrientation);
        glm::vec3 forward = -cameraOrientationMat[2];
        glm::vec3 right = cameraOrientationMat[0];

        glm::vec3 movement = glm::vec3(0);
        if (wPressed)
            movement += forward;
        if (sPressed)
            movement -= forward;
        if (aPressed)
            movement -= right;
        if (dPressed)
            movement += right;

        cameraPosition += deltaTime * translationScale * movement;

        glm::mat4x4 view = identity;
        view *= glm::mat4x4(inverse(cameraOrientation));
        view = glm::translate(view, -cameraPosition);

        glm::mat4x4 viewProjection = projection * view;
        auto deviceInfo = gDevice->getDeviceInfo();
        glm::mat4x4 correctionMatrix;
        memcpy(&correctionMatrix, deviceInfo.identityProjectionMatrix, sizeof(float) * 16);
        viewProjection = correctionMatrix * viewProjection;
        // glm uses column-major layout, we need to translate it to row-major.
        viewProjection = glm::transpose(viewProjection);

        auto drawCommandBuffer = gTransientHeaps[frameIndex]->createCommandBuffer();
        auto drawCommandEncoder =
            drawCommandBuffer->encodeRenderCommands(gRenderPass, gFramebuffers[frameIndex]);
        gfx::Viewport viewport = {};
        viewport.maxZ = 1.0f;
        viewport.extentX = (float)clientRect.width;
        viewport.extentY = (float)clientRect.height;
        drawCommandEncoder->setViewportAndScissor(viewport);
        drawCommandEncoder->setPrimitiveTopology(PrimitiveTopology::TriangleList);

        // We are only rendering one view, so we can fill in a per-view
        // shader object once and use it across all draw calls.
        //

        auto viewShaderObject = gDevice->createShaderObject(context.perViewShaderType);
        {
            ShaderCursor cursor(viewShaderObject);
            cursor["viewProjection"].setData(&viewProjection, sizeof(viewProjection));
            cursor["eyePosition"].setData(&cameraPosition, sizeof(cameraPosition));
        }
        // The majority of our rendering logic is handled as a loop
        // over the models in the scene, and their meshes.
        //
        for (auto& model : gModels)
        {
            drawCommandEncoder->setVertexBuffer(0, model->vertexBuffer);
            drawCommandEncoder->setIndexBuffer(model->indexBuffer, Format::R32_UINT);
            // For each model we provide a parameter
            // block that holds the per-model transformation
            // parameters, corresponding to the `PerModel` type
            // in the shader code.
            glm::mat4x4 modelTransform = identity;
            glm::mat4x4 inverseTransposeModelTransform = inverse(transpose(modelTransform));
            auto modelShaderObject = gDevice->createShaderObject(context.perModelShaderType);
            {
                ShaderCursor cursor(modelShaderObject);
                cursor["modelTransform"].setData(&modelTransform, sizeof(modelTransform));
                cursor["inverseTransposeModelTransform"].setData(
                    &inverseTransposeModelTransform,
                    sizeof(inverseTransposeModelTransform));
            }

            auto lightShaderObject = lightEnv->createShaderObject();

            // Now we loop over the meshes in the model.
            //
            // A more advanced rendering loop would sort things by material
            // rather than by model, to avoid overly frequent state changes.
            // We are just doing something simple for the purposes of an
            // exmple program.
            //
            for (auto& mesh : model->meshes)
            {
                // Set the pipeline and binding state for drawing each mesh.
                auto rootObject = drawCommandEncoder->bindPipeline(gPipelineState);
                ShaderCursor rootCursor(rootObject);
                rootCursor["gViewParams"].setObject(viewShaderObject);
                rootCursor["gModelParams"].setObject(modelShaderObject);
                rootCursor["gLightEnv"].setObject(lightShaderObject);

                // Each mesh has a material, and each material has its own
                // parameter block that was created at load time, so we
                // can just re-use the persistent parameter block for the
                // chosen material.
                //
                // Note that binding the material parameter block here is
                // both selecting the values to use for various material
                // parameters as well as the *code* to use for material
                // evaluation (based on the concrete shader type that
                // is implementing the `IMaterial` interface).
                //
                rootCursor["gMaterial"].setObject(mesh->material->shaderObject);

                // All the shader parameters and pipeline states have been set up,
                // we can now issue a draw call for the mesh.
                drawCommandEncoder->drawIndexed(mesh->indexCount, mesh->firstIndex);
            }
        }
        drawCommandEncoder->endEncoding();
        drawCommandBuffer->close();
        gQueue->executeCommandBuffer(drawCommandBuffer);

        if (!isTestMode())
        {
            gSwapchain->present();
        }
    }
};

// This macro instantiates an appropriate main function to
// run the application defined above.
EXAMPLE_MAIN(innerMain<ModelViewer>);
