// unit-test-translation-unit-import.cpp

#include "../../source/core/slang-io.h"
#include "../../source/core/slang-process.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

#include <stdio.h>
#include <stdlib.h>

using namespace Slang;

// Test that the IModule::findAndCheckEntryPoint API supports discovering
// entrypoints without a [shader] attribute.

SLANG_UNIT_TEST(genericInterfaceConformance)
{
    // Source for a module that contains an undecorated entrypoint.
    const char* userSourceBody = R"(
        public interface ITestInterface<Real : IFloat> {
            Real sample();
        }

        struct TestInterfaceImpl<Real : IFloat> : ITestInterface<Real> {
            Real sample() {
                return x;
            }
            Real x;
        }

        //TEST_INPUT: set data = new StructuredBuffer<ITestInterface<float> >[new TestInterfaceImpl<float>{1.0}];
        StructuredBuffer<ITestInterface<float>> data;

        //TEST_INPUT: set outputBuffer = out ubuffer(data=[0 0 0 0], stride=4);
        RWStructuredBuffer<int> outputBuffer;

        //TEST_INPUT: type_conformance TestInterfaceImpl<float>:ITestInterface<float> = 3

        [numthreads(1, 1, 1)]
        void computeMain()
        {
            let obj = data[0];
            // CHECK: 1
            outputBuffer[0] = int(obj.sample());
        }
        )";

    ComPtr<slang::IGlobalSession> globalSession;
    SLANG_CHECK(slang_createGlobalSession(SLANG_API_VERSION, globalSession.writeRef()) == SLANG_OK);
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_HLSL;

    slang::SessionDesc sessionDesc = {};
    sessionDesc.targetCount = 1;
    sessionDesc.targets = &targetDesc;
    sessionDesc.allowGLSLSyntax = true;

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
        "computeMain",
        SLANG_STAGE_COMPUTE,
        entryPoint.writeRef(),
        diagnosticBlob.writeRef());
    SLANG_CHECK(entryPoint != nullptr);

    ComPtr<slang::IComponentType> compositeProgram;
    slang::IComponentType* components[] = {module, entryPoint.get()};
    session->createCompositeComponentType(
        components,
        2,
        compositeProgram.writeRef(),
        diagnosticBlob.writeRef());
    SLANG_CHECK(compositeProgram != nullptr);

    ComPtr<slang::ITypeConformance> typeConformance;
    auto result = session->createTypeConformanceComponentType(
        compositeProgram->getLayout()->findTypeByName("TestInterfaceImpl<float>"),
        compositeProgram->getLayout()->findTypeByName("ITestInterface<float>"),
        typeConformance.writeRef(),
        3,
        diagnosticBlob.writeRef());
    SLANG_CHECK(result == SLANG_OK);
    SLANG_CHECK(typeConformance != nullptr);

    ComPtr<slang::IComponentType> compositeProgram2;
    slang::IComponentType* components2[] = {compositeProgram.get(), typeConformance.get()};
    session->createCompositeComponentType(
        components2,
        2,
        compositeProgram2.writeRef(),
        diagnosticBlob.writeRef());

    ComPtr<slang::IComponentType> linkedProgram;
    compositeProgram2->link(linkedProgram.writeRef(), diagnosticBlob.writeRef());
    SLANG_CHECK(linkedProgram != nullptr);

    ComPtr<slang::IBlob> code;
    linkedProgram->getEntryPointCode(0, 0, code.writeRef(), diagnosticBlob.writeRef());
    SLANG_CHECK(code != nullptr);

    auto codeSrc = UnownedStringSlice((const char*)code->getBufferPointer());
    SLANG_CHECK(codeSrc.indexOf(toSlice("computeMain")) != -1);
}
