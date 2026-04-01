// unit-test-argument-buffer-tier-2-reflection.cpp

#include "../../source/core/slang-io.h"
#include "../../source/core/slang-process.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

#include <stdio.h>
#include <stdlib.h>

using namespace Slang;

// Test metal argument buffer tier2 layout rules.

SLANG_UNIT_TEST(metalArgumentBufferTier2Reflection)
{
    const char* userSourceBody = R"(
        struct A
        {
          float3 one;
          float3 two;
          float three;
        }

        struct Args{
          ParameterBlock<A> a;
        }
        ParameterBlock<Args> argument_buffer;
        RWStructuredBuffer<float> outputBuffer;

        [numthreads(1,1,1)]
        void computeMain()
        {
            outputBuffer[0] = argument_buffer.a.two.x;
        }
        )";

    auto moduleName = "moduleG" + String(Process::getId());
    String userSource = "import " + moduleName + ";\n" + userSourceBody;
    ComPtr<slang::IGlobalSession> globalSession;
    SLANG_CHECK(slang_createGlobalSession(SLANG_API_VERSION, globalSession.writeRef()) == SLANG_OK);
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

    auto layout = module->getLayout();

    auto type = layout->findTypeByName("A");
    auto typeLayout = layout->getTypeLayout(type, slang::LayoutRules::MetalArgumentBufferTier2);
    SLANG_CHECK(typeLayout->getFieldByIndex(0)->getOffset() == 0);
    SLANG_CHECK(typeLayout->getFieldByIndex(0)->getTypeLayout()->getSize() == 16);
    SLANG_CHECK(typeLayout->getFieldByIndex(1)->getOffset() == 16);
    SLANG_CHECK(typeLayout->getFieldByIndex(1)->getTypeLayout()->getSize() == 16);
    SLANG_CHECK(typeLayout->getFieldByIndex(2)->getOffset() == 32);
    SLANG_CHECK(typeLayout->getFieldByIndex(2)->getTypeLayout()->getSize() == 4);
}
