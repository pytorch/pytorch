#include "core/slang-blob.h"
#include "gfx-test-util.h"
#include "slang-gfx.h"
#include "unit-test/slang-unit-test.h"

using namespace gfx;

namespace gfx_test
{

static void diagnoseIfNeeded(Slang::ComPtr<slang::IBlob>& diagnosticsBlob)
{
    if (diagnosticsBlob && diagnosticsBlob->getBufferSize() > 0)
    {
        fprintf(stderr, "%s\n", (const char*)diagnosticsBlob->getBufferPointer());
    }
}

static Slang::Result loadSpirvProgram(
    gfx::IDevice* device,
    Slang::ComPtr<gfx::IShaderProgram>& outShaderProgram,
    slang::ProgramLayout*& slangReflection)
{
    // main.slang: declares the interface and extern struct S, and the vertex shader.
    const char* mainSrc = R"(
        public interface IFoo
        {
            public float4 getFoo();
        };
        public extern struct S : IFoo;

        [shader("vertex")]
        float4 vertexMain(S params) : SV_Position
        {
            return params.getFoo();
        }
    )";

    // foo.slang: defines S with its field layout and its implementation of getFoo().
    const char* fooSrc = R"(
        import main;

        export public struct S : IFoo
        {
            public float4 getFoo() { return this.foo; }
            float4 foo;
        }
    )";

    Slang::ComPtr<slang::ISession> slangSession;
    SLANG_RETURN_ON_FAIL(device->getSlangSession(slangSession.writeRef()));
    Slang::ComPtr<slang::IBlob> diagnosticsBlob;

    // Create blobs for the two modules.
    auto mainBlob = Slang::UnownedRawBlob::create(mainSrc, strlen(mainSrc));
    auto fooBlob = Slang::UnownedRawBlob::create(fooSrc, strlen(fooSrc));

    // Load modules from source.
    slang::IModule* mainModule = slangSession->loadModuleFromSource("main", "main.slang", mainBlob);
    slang::IModule* fooModule = slangSession->loadModuleFromSource("foo", "foo.slang", fooBlob);

    // Find the entry point from main.slang
    Slang::ComPtr<slang::IEntryPoint> vsEntryPoint;
    SLANG_RETURN_ON_FAIL(mainModule->findEntryPointByName("vertexMain", vsEntryPoint.writeRef()));

    // Compose the program from both modules and the entry point.
    Slang::List<slang::IComponentType*> componentTypes;
    componentTypes.add(mainModule);
    componentTypes.add(fooModule);
    componentTypes.add(vsEntryPoint);

    Slang::ComPtr<slang::IComponentType> composedProgram;
    SLANG_RETURN_ON_FAIL(slangSession->createCompositeComponentType(
        componentTypes.getBuffer(),
        componentTypes.getCount(),
        composedProgram.writeRef(),
        diagnosticsBlob.writeRef()));
    diagnoseIfNeeded(diagnosticsBlob);

    // Link the composite program.
    Slang::ComPtr<slang::IComponentType> linkedProgram;
    SLANG_RETURN_ON_FAIL(
        composedProgram->link(linkedProgram.writeRef(), diagnosticsBlob.writeRef()));
    diagnoseIfNeeded(diagnosticsBlob);

    // Retrieve the reflection information.
    composedProgram = linkedProgram;
    slangReflection = composedProgram->getLayout();

    // Create a shader program that will generate SPIRV code.
    gfx::IShaderProgram::Desc programDesc = {};
    programDesc.slangGlobalScope = composedProgram.get();
    auto shaderProgram = device->createProgram(programDesc);
    outShaderProgram = shaderProgram;

    // Force SPIRV generation by explicitly requesting it
    Slang::ComPtr<slang::IBlob> spirvBlob;
    Slang::ComPtr<slang::IBlob> spirvDiagnostics;

    // Request SPIRV code generation for the vertex shader entry point
    auto targetIndex = 0;     // Assuming this is the first/only target
    auto entryPointIndex = 0; // Assuming this is the first/only entry point

    auto result = composedProgram->getEntryPointCode(
        entryPointIndex,
        targetIndex,
        spirvBlob.writeRef(),
        spirvDiagnostics.writeRef());

    if (SLANG_FAILED(result))
    {
        if (spirvDiagnostics && spirvDiagnostics->getBufferSize() > 0)
        {
            fprintf(
                stderr,
                "SPIRV generation failed: %s\n",
                (const char*)spirvDiagnostics->getBufferPointer());
        }
        return result;
    }

    // Verify we actually got SPIRV code
    if (!spirvBlob || spirvBlob->getBufferSize() == 0)
    {
        return SLANG_FAIL;
    }

    return SLANG_OK;
}

// Function to validate the type layout of struct S
static void validateStructSLayout(UnitTestContext* context, slang::ProgramLayout* slangReflection)
{
    // Check reflection is available
    SLANG_CHECK(slangReflection != nullptr);

    // Get the entry point layout for vertexMain
    auto entryPointCount = slangReflection->getEntryPointCount();
    slang::EntryPointLayout* entryPointLayout = nullptr;

    for (unsigned int i = 0; i < entryPointCount; i++)
    {
        auto currentEntryPoint = slangReflection->getEntryPointByIndex(i);
        const char* name = currentEntryPoint->getName();

        if (strcmp(name, "vertexMain") == 0)
        {
            entryPointLayout = currentEntryPoint;
            break;
        }
    }

    SLANG_CHECK_MSG(entryPointLayout != nullptr, "Could not find vertexMain entry point");

    // Get the parameter count for the entry point
    auto paramCount = entryPointLayout->getParameterCount();
    SLANG_CHECK_MSG(paramCount >= 1, "Entry point has no parameters");

    // Get the first parameter, which should be of type S
    auto paramLayout = entryPointLayout->getParameterByIndex(0);
    SLANG_CHECK_MSG(paramLayout != nullptr, "Could not get first parameter layout");

    // Get the type layout of the parameter
    auto typeLayout = paramLayout->getTypeLayout();
    SLANG_CHECK_MSG(typeLayout != nullptr, "Parameter has no type layout");

    // Check if it's a struct type
    auto kind = typeLayout->getKind();
    SLANG_CHECK_MSG(kind == slang::TypeReflection::Kind::Struct, "Parameter is not a struct type");

    // Get the field count
    auto fieldCount = typeLayout->getFieldCount();
    SLANG_CHECK_MSG(fieldCount >= 1, "Struct has no fields");

    // Check for the 'foo' field
    bool foundFooField = false;
    for (unsigned int i = 0; i < fieldCount; i++)
    {
        auto fieldLayout = typeLayout->getFieldByIndex(i);
        const char* fieldName = fieldLayout->getName();

        if (fieldName && strcmp(fieldName, "foo") == 0)
        {
            foundFooField = true;

            // Check that it's a float4 type
            auto fieldTypeLayout = fieldLayout->getTypeLayout();
            auto fieldTypeKind = fieldTypeLayout->getKind();

            SLANG_CHECK_MSG(
                fieldTypeKind == slang::TypeReflection::Kind::Vector,
                "Field 'foo' is not a vector type");

            auto elementCount = fieldTypeLayout->getElementCount();
            SLANG_CHECK_MSG(elementCount == 4, "Field 'foo' is not a 4-element vector");

            break;
        }
    }

    SLANG_CHECK_MSG(foundFooField, "Could not find field 'foo' in struct S");
}

void linkTimeTypeLayoutImpl(gfx::IDevice* device, UnitTestContext* context)
{
    Slang::ComPtr<gfx::IShaderProgram> shaderProgram;
    slang::ProgramLayout* slangReflection = nullptr;

    auto result = loadSpirvProgram(device, shaderProgram, slangReflection);
    SLANG_CHECK(SLANG_SUCCEEDED(result));

    // Validate the struct S layout
    validateStructSLayout(context, slangReflection);

    // Create a graphics pipeline to verify SPIRV code generation works
    GraphicsPipelineStateDesc pipelineDesc = {};
    pipelineDesc.program = shaderProgram.get();

    // We need to set up a minimal pipeline state for a vertex shader
    pipelineDesc.primitiveType = PrimitiveType::Triangle;

    ComPtr<gfx::IPipelineState> pipelineState;
    auto pipelineResult =
        device->createGraphicsPipelineState(pipelineDesc, pipelineState.writeRef());
    SLANG_CHECK(SLANG_SUCCEEDED(pipelineResult));
}

//
// This test verifies that type layout information correctly propagates through
// the Slang compilation pipeline when types are defined in modules other than where they are used.
// Specifically, it tests
// that when using an extern struct that's defined in a separate module:
//
// 1. The struct definition is properly linked across module boundaries
// 2. The complete type layout information is available in the reflection data
// 3. SPIRV code generation succeeds with the linked type information (this
// failed before when layout information was required during code generation)
//

SLANG_UNIT_TEST(linkTimeTypeLayout)
{
    runTestImpl(linkTimeTypeLayoutImpl, unitTestContext, Slang::RenderApiFlag::Vulkan);
}

} // namespace gfx_test
