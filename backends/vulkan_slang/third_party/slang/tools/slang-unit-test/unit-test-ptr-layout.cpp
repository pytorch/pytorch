// unit-test-ptr-layout.cpp

#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

#include <stdio.h>
#include <stdlib.h>

using namespace Slang;

SLANG_UNIT_TEST(pointerTypeLayout)
{
    const char* testSource = "struct TestStruct {"
                             "   int3 member0;"
                             "   float member1;"
                             "};";

    ComPtr<slang::IGlobalSession> globalSession;
    SLANG_CHECK(slang_createGlobalSession(SLANG_API_VERSION, globalSession.writeRef()) == SLANG_OK);
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_SPIRV;
    targetDesc.profile = globalSession->findProfile("spirv_1_5");
    slang::SessionDesc sessionDesc = {};
    sessionDesc.targetCount = 1;
    sessionDesc.targets = &targetDesc;
    sessionDesc.compilerOptionEntryCount = 1;
    slang::CompilerOptionEntry compilerOptionEntry = {};
    compilerOptionEntry.name = slang::CompilerOptionName::EmitSpirvViaGLSL;
    compilerOptionEntry.value.kind = slang::CompilerOptionValueKind::Int;
    compilerOptionEntry.value.intValue0 = 1;
    sessionDesc.compilerOptionEntries = &compilerOptionEntry;

    ComPtr<slang::ISession> session;
    SLANG_CHECK(globalSession->createSession(sessionDesc, session.writeRef()) == SLANG_OK);

    ComPtr<slang::IBlob> diagnosticBlob;
    auto module =
        session->loadModuleFromSourceString("m", "m.slang", testSource, diagnosticBlob.writeRef());
    SLANG_CHECK(module != nullptr);


    auto testBody = [&]()
    {
        auto reflection = module->getLayout();
        auto testStruct = reflection->findTypeByName("Ptr<TestStruct>");
        auto ptrLayout = reflection->getTypeLayout(testStruct);
        auto valueLayout = ptrLayout->getElementTypeLayout();
        SLANG_CHECK_ABORT(valueLayout->getFieldCount() == 2);
        SLANG_CHECK_ABORT(valueLayout->getFieldByIndex(1)->getOffset() == 12);
    };

    testBody();
}
