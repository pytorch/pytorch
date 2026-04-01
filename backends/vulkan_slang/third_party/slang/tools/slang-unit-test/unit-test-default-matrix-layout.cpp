// unit-test-default-matrix-layout.cpp

#include "../../source/core/slang-list.h"
#include "slang-com-helper.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

#include <stdio.h>
#include <stdlib.h>

namespace
{

using namespace Slang;

struct DefaultMatrixLayoutTestContext
{
    DefaultMatrixLayoutTestContext(UnitTestContext* context)
        : m_unitTestContext(context)
    {
        slang::IGlobalSession* slangSession = m_unitTestContext->slangGlobalSession;
    }

    SlangResult runTests()
    {
        slang::IGlobalSession* slangSession = m_unitTestContext->slangGlobalSession;
        ComPtr<slang::ISession> session;
        slang::SessionDesc sessionDesc{};
        sessionDesc.targetCount = 1;
        slang::TargetDesc targetDesc{};
        targetDesc.format = SLANG_GLSL;
        targetDesc.profile = slangSession->findProfile("glsl_460");
        sessionDesc.targets = &targetDesc;
        sessionDesc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;
        SLANG_RETURN_ON_FAIL(slangSession->createSession(sessionDesc, session.writeRef()));

        auto module = session->loadModuleFromSourceString(
            "mymodule",
            "mymodule.slang",
            R"(
            RWStructuredBuffer<float> output;
            [numthreads(1,1,1)] [shader("compute")]
            void main(uniform float3x4 m)
            {
                output[0] = m[0][0];
            })");
        if (!module)
            return SLANG_FAIL;

        ComPtr<slang::IEntryPoint> entryPoint;
        SLANG_RETURN_ON_FAIL(module->findEntryPointByName("main", entryPoint.writeRef()));

        if (!entryPoint)
            return SLANG_FAIL;

        slang::IComponentType* components[] = {module, entryPoint.get()};
        ComPtr<slang::IComponentType> composedProgram;
        SLANG_RETURN_ON_FAIL(
            session->createCompositeComponentType(components, 2, composedProgram.writeRef()));

        ComPtr<slang::IComponentType> linkedProgram;
        SLANG_RETURN_ON_FAIL(composedProgram->link(linkedProgram.writeRef()));

        ComPtr<slang::IBlob> outCode;
        SLANG_RETURN_ON_FAIL(linkedProgram->getEntryPointCode(0, 0, outCode.writeRef()));

        const char* code = (const char*)outCode->getBufferPointer();
        if (strstr(code, "row_major") != nullptr)
            return SLANG_OK;
        return SLANG_FAIL;
    }

    UnitTestContext* m_unitTestContext;
};

} // namespace

SLANG_UNIT_TEST(defaultMatrixLayout)
{
    DefaultMatrixLayoutTestContext context(unitTestContext);

    const auto result = context.runTests();

    SLANG_CHECK(SLANG_SUCCEEDED(result));
}
