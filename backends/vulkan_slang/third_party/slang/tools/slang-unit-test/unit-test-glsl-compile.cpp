// unit-test-glsl-compile.cpp

#include "../../source/core/slang-io.h"
#include "../../source/core/slang-process.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

#include <stdio.h>
#include <stdlib.h>

using namespace Slang;

// Test the compilation API for cross-compiling glsl source to SPIRV.

SLANG_UNIT_TEST(glslCompile)
{
    const char* userSourceBody = R"(
            #version 450 core
            layout(location = 0) in vec2 aPosition;
            layout(location = 1) in vec4 aColor;
            layout(location = 0) out vec4 vColor;
            void main() {
                vColor = aColor;
                gl_Position = vec4(aPosition, 0, 1);
            }
        )";
    ComPtr<slang::IGlobalSession> globalSession;
    SlangGlobalSessionDesc globalDesc = {};
    globalDesc.enableGLSL = true;
    SLANG_CHECK(slang_createGlobalSession2(&globalDesc, globalSession.writeRef()) == SLANG_OK);
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_SPIRV;
    targetDesc.profile = globalSession->findProfile("spirv_1_5");
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
        "main",
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

    SLANG_CHECK(code != nullptr);
}
