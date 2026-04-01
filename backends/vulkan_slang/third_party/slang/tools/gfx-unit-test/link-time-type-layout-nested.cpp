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

static Slang::Result loadProgram(
    gfx::IDevice* device,
    Slang::ComPtr<gfx::IShaderProgram>& outShaderProgram,
    slang::ProgramLayout*& slangReflection)
{
    // main.slang: declares the interface, extern struct Inner, and Outer struct with Inner field
    const char* mainSrc = R"(
        // Define an interface
        public interface IFoo
        {
            public float4 getFoo();
        };

        // Define an extern struct that implements the interface
        public extern struct Inner : IFoo;

        // Define a regular struct that contains an Inner field
        public struct Outer
        {
            float2 position;
            Inner innerData;
            float2 texCoord;
        };

        // Vertex shader entry point that takes an Outer parameter
        [shader("vertex")]
        float4 vertexMain(Outer params) : SV_Position
        {
            return float4(params.position, 0.0f, 1.0f) + params.innerData.getFoo();
        }
    )";

    // inner.slang: defines Inner with its field layout and its implementation of getFoo()
    const char* innerSrc = R"(
        import main;

        // Define the implementation of Inner with its field layout
        export public struct Inner : IFoo
        {
            public float4 getFoo() { return this.data; }
            float4 data;
        }
    )";

    Slang::ComPtr<slang::ISession> slangSession;
    SLANG_RETURN_ON_FAIL(device->getSlangSession(slangSession.writeRef()));
    Slang::ComPtr<slang::IBlob> diagnosticsBlob;

    // Create blobs for the two modules
    auto mainBlob = Slang::UnownedRawBlob::create(mainSrc, strlen(mainSrc));
    auto innerBlob = Slang::UnownedRawBlob::create(innerSrc, strlen(innerSrc));

    // Load modules from source
    slang::IModule* mainModule = slangSession->loadModuleFromSource("main", "main.slang", mainBlob);
    slang::IModule* innerModule =
        slangSession->loadModuleFromSource("inner", "inner.slang", innerBlob);

    // Find the entry point from main.slang
    Slang::ComPtr<slang::IEntryPoint> vsEntryPoint;
    SLANG_RETURN_ON_FAIL(mainModule->findEntryPointByName("vertexMain", vsEntryPoint.writeRef()));

    // Compose the program from both modules and the entry point
    Slang::List<slang::IComponentType*> componentTypes;
    componentTypes.add(mainModule);
    componentTypes.add(innerModule);
    componentTypes.add(vsEntryPoint);

    Slang::ComPtr<slang::IComponentType> composedProgram;
    SLANG_RETURN_ON_FAIL(slangSession->createCompositeComponentType(
        componentTypes.getBuffer(),
        componentTypes.getCount(),
        composedProgram.writeRef(),
        diagnosticsBlob.writeRef()));
    diagnoseIfNeeded(diagnosticsBlob);

    // Link the composite program
    Slang::ComPtr<slang::IComponentType> linkedProgram;
    SLANG_RETURN_ON_FAIL(
        composedProgram->link(linkedProgram.writeRef(), diagnosticsBlob.writeRef()));
    diagnoseIfNeeded(diagnosticsBlob);

    // Retrieve the reflection information
    composedProgram = linkedProgram;
    slangReflection = composedProgram->getLayout();

    // Create a shader program
    gfx::IShaderProgram::Desc programDesc = {};
    programDesc.slangGlobalScope = composedProgram.get();
    auto shaderProgram = device->createProgram(programDesc);
    outShaderProgram = shaderProgram;

    return SLANG_OK;
}

// Function to validate the type layout of Outer struct with nested Inner struct
static void validateNestedExternStructLayout(
    UnitTestContext* context,
    slang::ProgramLayout* slangReflection)
{
    // Check reflection is available
    SLANG_CHECK(slangReflection != nullptr);

    // Get the entry point layout for vertexMain
    slang::EntryPointLayout* entryPointLayout = slangReflection->findEntryPointByName("vertexMain");

    SLANG_CHECK_MSG(entryPointLayout != nullptr, "Could not find vertexMain entry point");

    // Get the parameter count for the entry point
    auto paramCount = entryPointLayout->getParameterCount();
    SLANG_CHECK_MSG(paramCount >= 1, "Entry point has no parameters");

    // Get the first parameter, which should be of type Outer
    auto paramLayout = entryPointLayout->getParameterByIndex(0);
    SLANG_CHECK_MSG(paramLayout != nullptr, "Could not get first parameter layout");

    // Get the type layout of the parameter
    auto outerTypeLayout = paramLayout->getTypeLayout();
    SLANG_CHECK_MSG(outerTypeLayout != nullptr, "Parameter has no type layout");

    // Check if it's a struct type
    auto kind = outerTypeLayout->getKind();
    SLANG_CHECK_MSG(kind == slang::TypeReflection::Kind::Struct, "Parameter is not a struct type");

    // Verify Outer has 3 fields: position, innerData, texCoord
    auto fieldCount = outerTypeLayout->getFieldCount();
    SLANG_CHECK_MSG(fieldCount == 3, "Outer struct does not have 3 fields");

    // Find and check the innerData field
    slang::VariableLayoutReflection* innerDataField = nullptr;
    for (unsigned int i = 0; i < fieldCount; i++)
    {
        auto fieldLayout = outerTypeLayout->getFieldByIndex(i);
        const char* fieldName = fieldLayout->getName();

        if (fieldName && strcmp(fieldName, "innerData") == 0)
        {
            innerDataField = fieldLayout;
            break;
        }
    }

    SLANG_CHECK_MSG(innerDataField != nullptr, "Could not find innerData field in Outer struct");

    // Get the type layout of the innerData field
    auto innerTypeLayout = innerDataField->getTypeLayout();
    SLANG_CHECK_MSG(innerTypeLayout != nullptr, "innerData field has no type layout");

    // Verify Inner is a struct type
    kind = innerTypeLayout->getKind();
    SLANG_CHECK_MSG(kind == slang::TypeReflection::Kind::Struct, "Inner is not a struct type");

    // Verify Inner has 1 field (data)
    fieldCount = innerTypeLayout->getFieldCount();
    SLANG_CHECK_MSG(fieldCount == 1, "Inner struct does not have 1 field");

    // Find and check the data field in Inner
    bool foundDataField = false;
    for (unsigned int i = 0; i < fieldCount; i++)
    {
        auto fieldLayout = innerTypeLayout->getFieldByIndex(i);
        const char* fieldName = fieldLayout->getName();

        if (fieldName && strcmp(fieldName, "data") == 0)
        {
            foundDataField = true;

            // Check that it's a float4 type
            auto fieldTypeLayout = fieldLayout->getTypeLayout();
            auto fieldTypeKind = fieldTypeLayout->getKind();

            SLANG_CHECK_MSG(
                fieldTypeKind == slang::TypeReflection::Kind::Vector,
                "Field 'data' is not a vector type");

            auto elementCount = fieldTypeLayout->getElementCount();
            SLANG_CHECK_MSG(elementCount == 4, "Field 'data' is not a 4-element vector");

            break;
        }
    }

    SLANG_CHECK_MSG(foundDataField, "Could not find field 'data' in Inner struct");
}

void linkTimeTypeLayoutNestedImpl(gfx::IDevice* device, UnitTestContext* context)
{
    Slang::ComPtr<gfx::IShaderProgram> shaderProgram;
    slang::ProgramLayout* slangReflection = nullptr;

    auto result = loadProgram(device, shaderProgram, slangReflection);
    SLANG_CHECK(SLANG_SUCCEEDED(result));

    // Validate the nested struct layout
    validateNestedExternStructLayout(context, slangReflection);

    // Create a graphics pipeline to verify everything works
    GraphicsPipelineStateDesc pipelineDesc = {};
    pipelineDesc.program = shaderProgram.get();
    pipelineDesc.primitiveType = PrimitiveType::Triangle;

    ComPtr<gfx::IPipelineState> pipelineState;
    auto pipelineResult =
        device->createGraphicsPipelineState(pipelineDesc, pipelineState.writeRef());
    SLANG_CHECK(SLANG_SUCCEEDED(pipelineResult));
}

//
// This test verifies that type layout information correctly propagates through
// the Slang compilation pipeline when a regular struct contains a field whose type
// is an extern struct defined in another module.
// Specifically, it tests that:
//
// 1. The Outer struct correctly includes the Inner extern struct as a field
// 2. After linking, the Inner struct's layout is properly resolved with its field
// 3. The complete type layout information is available in the reflection data
//

SLANG_UNIT_TEST(linkTimeTypeLayoutNested)
{
    runTestImpl(linkTimeTypeLayoutNestedImpl, unitTestContext, Slang::RenderApiFlag::Vulkan);
}

} // namespace gfx_test
