// unit-test-translation-unit-import.cpp

#include "../../source/core/slang-io.h"
#include "../../source/core/slang-process.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

#include <stdio.h>
#include <stdlib.h>

using namespace Slang;

// Test that the IComponentType::getTargetCode API supports
// compiling a program with multiple entrypoints and retrieving a single
// compiled module that contains all the entrypoints.
//
SLANG_UNIT_TEST(getTargetCode)
{
    // Source for a module that contains an undecorated entrypoint.
    const char* userSourceBody = R"(
        [shader("fragment")]
        float4 fragMain(float4 pos:SV_Position) : SV_Target
        {
            return pos;
        }
        [shader("vertex")]
        float4 vertMain(float4 pos) : SV_Position
        {
            return pos;
        }
        )";

    String userSource = userSourceBody;
    ComPtr<slang::IGlobalSession> globalSession;
    SLANG_CHECK(slang_createGlobalSession(SLANG_API_VERSION, globalSession.writeRef()) == SLANG_OK);
    slang::TargetDesc targetDesc = {};
    // Request SPIR-V disassembly so we can check the content.
    targetDesc.format = SLANG_SPIRV_ASM;
    targetDesc.profile = globalSession->findProfile("sm_5_0");
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

    ComPtr<slang::IComponentType> linkedProgram;
    module->link(linkedProgram.writeRef(), diagnosticBlob.writeRef());
    SLANG_CHECK(linkedProgram != nullptr);

    ComPtr<slang::IBlob> code;
    linkedProgram->getTargetCode(0, code.writeRef(), diagnosticBlob.writeRef());
    SLANG_CHECK(code != nullptr);

    SLANG_CHECK(code->getBufferSize() != 0);

    UnownedStringSlice resultStr = UnownedStringSlice((char*)code->getBufferPointer());

    // Make sure the spirv disassembly contains both entrypoint names.
    SLANG_CHECK(resultStr.indexOf(toSlice("fragMain")) != -1);
    SLANG_CHECK(resultStr.indexOf(toSlice("vertMain")) != -1);
}
