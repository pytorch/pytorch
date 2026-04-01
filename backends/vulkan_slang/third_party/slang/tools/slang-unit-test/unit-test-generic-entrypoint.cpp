// unit-test-generic-entrypoint.cpp

#include "../../source/core/slang-io.h"
#include "../../source/core/slang-process.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

#include <stdio.h>
#include <stdlib.h>

using namespace Slang;

// Test the compilation API for compiling a specialized generic entrypoint.

SLANG_UNIT_TEST(genericEntryPointCompile)
{
    const char* userSourceBody = R"(
            interface I { int getValue(); }
            struct X : I { int getValue() { return 100; } }
            float4 vertMain<T:I>(uniform T o) {
                return float4(o.getValue(), 0, 0, 1);
            }
        )";
    ComPtr<slang::IGlobalSession> globalSession;
    SLANG_CHECK(slang_createGlobalSession(SLANG_API_VERSION, globalSession.writeRef()) == SLANG_OK);
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_GLSL;
    slang::SessionDesc sessionDesc = {};
    sessionDesc.targetCount = 1;
    sessionDesc.targets = &targetDesc;
    ComPtr<slang::ISession> session;
    SLANG_CHECK(globalSession->createSession(sessionDesc, session.writeRef()) == SLANG_OK);

    ComPtr<slang::IBlob> diagnosticBlob;
    auto module = session->loadModuleFromSourceString(
        "m",
        "m.slang",
        userSourceBody,
        diagnosticBlob.writeRef());
    SLANG_CHECK(module != nullptr);

    ComPtr<slang::IEntryPoint> entryPoint;
    module->findAndCheckEntryPoint(
        "vertMain<X>",
        SLANG_STAGE_VERTEX,
        entryPoint.writeRef(),
        diagnosticBlob.writeRef());

    slang::IComponentType* componentTypes[2] = {module, entryPoint.get()};
    ComPtr<slang::IComponentType> composedProgram;
    session->createCompositeComponentType(
        componentTypes,
        2,
        composedProgram.writeRef(),
        diagnosticBlob.writeRef());

    ComPtr<slang::IComponentType> linkedProgram;
    composedProgram->link(linkedProgram.writeRef(), diagnosticBlob.writeRef());

    ComPtr<slang::IBlob> code;
    linkedProgram->getEntryPointCode(0, 0, code.writeRef(), diagnosticBlob.writeRef());

    SLANG_CHECK(
        UnownedStringSlice((char*)code->getBufferPointer())
            .indexOf(toSlice("vec4(float(X_getValue")) != -1);
}
