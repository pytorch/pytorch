// unit-test-geometry-shader.cpp

#include "../../source/core/slang-io.h"
#include "../../source/core/slang-process.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

#include <stdio.h>
#include <stdlib.h>

using namespace Slang;

// Test the compilation API for compiling geometry shaders to DXIL.

#if SLANG_WINDOWS_FAMILY

SLANG_UNIT_TEST(geometryShader)
{
    const char* userSourceBody = R"(
        struct GS_INPUT
        {
            float4 PosSS : TEXTURE0;     // [Screen Space] Position
        };

        struct PS_INPUT
        {
            float4 PosSS : SV_POSITION;  // [Screen Space] Position
        };

        [maxvertexcount(3)] 
        void main(triangle GS_INPUT input[3], inout TriangleStream<PS_INPUT> outStream)
        {
            PS_INPUT output;

            output.PosSS = input[0].PosSS;
            outStream.Append(output);
     
            output.PosSS = input[1].PosSS;
            outStream.Append(output);

            output.PosSS = input[2].PosSS;
            outStream.Append(output);

            outStream.RestartStrip();
        }
        )";
    ComPtr<slang::IGlobalSession> globalSession;
    SlangGlobalSessionDesc globalDesc = {};
    globalDesc.enableGLSL = true;
    SLANG_CHECK(slang_createGlobalSession2(&globalDesc, globalSession.writeRef()) == SLANG_OK);
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_DXIL;
    targetDesc.profile = globalSession->findProfile("sm_6_0");
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
        SLANG_STAGE_GEOMETRY,
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

#endif
