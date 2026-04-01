// unit-test-ptr-layout.cpp

#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

#include <stdlib.h>

using namespace Slang;

SLANG_UNIT_TEST(metalConstantBufferInParameterBlockLayout)
{
    const char* testSource = R"(
        struct T 
        {
            float4 m0;
            float m1;
            float3 m2;
        };

        ParameterBlock<ConstantBuffer<T>> params;
    )";

    ComPtr<slang::IGlobalSession> globalSession;
    SLANG_CHECK(slang_createGlobalSession(SLANG_API_VERSION, globalSession.writeRef()) == SLANG_OK);

    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_METAL;
    targetDesc.profile = globalSession->findProfile("metal");

    slang::SessionDesc sessionDesc = {};
    sessionDesc.targetCount = 1;
    sessionDesc.targets = &targetDesc;

    ComPtr<slang::ISession> session;
    SLANG_CHECK(globalSession->createSession(sessionDesc, session.writeRef()) == SLANG_OK);

    ComPtr<slang::IBlob> diagnosticBlob;
    auto module = session->loadModuleFromSourceString(
        "test",
        "test.slang",
        testSource,
        diagnosticBlob.writeRef());
    SLANG_CHECK(module != nullptr);

    auto testBody = [&]()
    {
        auto reflection = module->getLayout();

        // Collect our layouts
        auto paramBlockType = reflection->findTypeByName("ParameterBlock<ConstantBuffer<T>>");
        SLANG_CHECK(paramBlockType != nullptr);
        auto paramBlockLayout = reflection->getTypeLayout(paramBlockType);
        SLANG_CHECK(paramBlockLayout != nullptr);
        auto cbufferLayout = paramBlockLayout->getElementTypeLayout();
        SLANG_CHECK(cbufferLayout != nullptr);
        auto structLayout = cbufferLayout->getElementTypeLayout();
        SLANG_CHECK(structLayout != nullptr);

        // Check offsets follow constant buffer rules (uniform alignment)
        // m0 : float4 should be at offset 0
        // m1 : float  should be at offset 16 (after float4)
        // m2 : float3 should be at offset 32 (aligned to 16-byte boundary)
        SLANG_CHECK(structLayout->getFieldCount() == 3);
        SLANG_CHECK(structLayout->getFieldByIndex(0)->getOffset() == 0);
        SLANG_CHECK(structLayout->getFieldByIndex(1)->getOffset() == 16);
        SLANG_CHECK(structLayout->getFieldByIndex(2)->getOffset() == 32);
    };

    testBody();
}

SLANG_UNIT_TEST(metalArgumentBufferLayout)
{
    const char* testSource = R"(
        struct T 
        {
            float4 m0;
            float m1;
            float3 m2;
        };

        // Using ParameterBlock directly without ConstantBuffer wrapper
        ParameterBlock<T> params;
    )";

    ComPtr<slang::IGlobalSession> globalSession;
    SLANG_CHECK(slang_createGlobalSession(SLANG_API_VERSION, globalSession.writeRef()) == SLANG_OK);

    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_METAL;
    targetDesc.profile = globalSession->findProfile("metal");

    slang::SessionDesc sessionDesc = {};
    sessionDesc.targetCount = 1;
    sessionDesc.targets = &targetDesc;

    ComPtr<slang::ISession> session;
    SLANG_CHECK(globalSession->createSession(sessionDesc, session.writeRef()) == SLANG_OK);

    ComPtr<slang::IBlob> diagnosticBlob;
    auto module = session->loadModuleFromSourceString(
        "test",
        "test.slang",
        testSource,
        diagnosticBlob.writeRef());
    SLANG_CHECK(module != nullptr);

    auto testBody = [&]()
    {
        auto reflection = module->getLayout();

        // Collect our layouts
        auto paramBlockType = reflection->findTypeByName("ParameterBlock<T>");
        SLANG_CHECK(paramBlockType != nullptr);
        auto paramBlockLayout = reflection->getTypeLayout(paramBlockType);
        SLANG_CHECK(paramBlockLayout != nullptr);
        auto structLayout = paramBlockLayout->getElementTypeLayout();
        SLANG_CHECK(structLayout != nullptr);

        // Check that offsets follow Metal argument buffer rules
        // Fields should have 0 offset and meaningful binding indices
        SLANG_CHECK(structLayout->getFieldCount() == 3);
        SLANG_CHECK(structLayout->getFieldByIndex(0)->getOffset() == 0);
        SLANG_CHECK(structLayout->getFieldByIndex(1)->getOffset() == 0);
        SLANG_CHECK(structLayout->getFieldByIndex(2)->getOffset() == 0);
        SLANG_CHECK(structLayout->getFieldByIndex(0)->getBindingIndex() == 0);
        SLANG_CHECK(structLayout->getFieldByIndex(1)->getBindingIndex() == 1);
        SLANG_CHECK(structLayout->getFieldByIndex(2)->getBindingIndex() == 2);
    };

    testBody();
}
