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

// Function to find and validate the struct S type layout
static void validateStructSLayout(
    UnitTestContext* context,
    slang::ProgramLayout* slangReflection,
    int expectedFieldCount)
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
    SLANG_CHECK_MSG(fieldCount == expectedFieldCount, "Struct has unexpected number of fields");

    // If we expect fields, check for the 'foo' field
    if (expectedFieldCount > 0)
    {
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
}

void linkTimeTypeLayoutCacheImpl(gfx::IDevice* device, UnitTestContext* context)
{
    // main.slang: declares the interface and extern struct S
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

    // foo.slang: defines S with its field layout and its implementation of getFoo()
    const char* fooSrc = R"(
        import main;

        export public struct S : IFoo
        {
            public float4 getFoo() { return this.foo; }
            float4 foo;
        }
    )";

    Slang::ComPtr<slang::ISession> slangSession;
    SLANG_CHECK(SLANG_SUCCEEDED(device->getSlangSession(slangSession.writeRef())));
    Slang::ComPtr<slang::IBlob> diagnosticsBlob;

    // Create blobs for the two modules
    auto mainBlob = Slang::UnownedRawBlob::create(mainSrc, strlen(mainSrc));
    auto fooBlob = Slang::UnownedRawBlob::create(fooSrc, strlen(fooSrc));

    // STEP 1: Load just the main module
    slang::IModule* mainModule = slangSession->loadModuleFromSource("main", "main.slang", mainBlob);
    SLANG_CHECK_MSG(mainModule != nullptr, "Failed to load main module");

    // Find the entry point from main.slang
    Slang::ComPtr<slang::IEntryPoint> vsEntryPoint;
    SLANG_CHECK(
        SLANG_SUCCEEDED(mainModule->findEntryPointByName("vertexMain", vsEntryPoint.writeRef())));

    // Create a program with just the main module
    Slang::List<slang::IComponentType*> componentTypes;
    componentTypes.add(mainModule);
    componentTypes.add(vsEntryPoint);

    Slang::ComPtr<slang::IComponentType> composedProgram;
    SLANG_CHECK(SLANG_SUCCEEDED(slangSession->createCompositeComponentType(
        componentTypes.getBuffer(),
        componentTypes.getCount(),
        composedProgram.writeRef(),
        diagnosticsBlob.writeRef())));
    diagnoseIfNeeded(diagnosticsBlob);

    // Link the main-only program
    Slang::ComPtr<slang::IComponentType> linkedProgram;
    SLANG_CHECK(SLANG_SUCCEEDED(
        composedProgram->link(linkedProgram.writeRef(), diagnosticsBlob.writeRef())));
    diagnoseIfNeeded(diagnosticsBlob);

    // Get the reflection information
    auto mainOnlyReflection = linkedProgram->getLayout();

    // Verify that struct S has no fields in the main-only program
    validateStructSLayout(context, mainOnlyReflection, 0);

    // STEP 2: Load the foo module and link it into the same program
    slang::IModule* fooModule = slangSession->loadModuleFromSource("foo", "foo.slang", fooBlob);
    SLANG_CHECK_MSG(fooModule != nullptr, "Failed to load foo module");

    // Create a new composite program that includes the foo module
    componentTypes.clear();
    componentTypes.add(mainModule);
    componentTypes.add(fooModule);
    componentTypes.add(vsEntryPoint);

    composedProgram = nullptr;
    SLANG_CHECK(SLANG_SUCCEEDED(slangSession->createCompositeComponentType(
        componentTypes.getBuffer(),
        componentTypes.getCount(),
        composedProgram.writeRef(),
        diagnosticsBlob.writeRef())));
    diagnoseIfNeeded(diagnosticsBlob);

    // Link the updated program
    linkedProgram = nullptr;
    SLANG_CHECK(SLANG_SUCCEEDED(
        composedProgram->link(linkedProgram.writeRef(), diagnosticsBlob.writeRef())));
    diagnoseIfNeeded(diagnosticsBlob);

    // Get the updated reflection information
    auto updatedReflection = linkedProgram->getLayout();

    // Verify that struct S now has one field in the updated program
    validateStructSLayout(context, updatedReflection, 1);
}

SLANG_UNIT_TEST(linkTimeTypeLayoutCache)
{
    runTestImpl(linkTimeTypeLayoutCacheImpl, unitTestContext, Slang::RenderApiFlag::Vulkan);
}

} // namespace gfx_test
