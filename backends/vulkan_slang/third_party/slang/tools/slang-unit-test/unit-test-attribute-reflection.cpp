// unit-test-translation-unit-import.cpp

#include "../../source/core/slang-io.h"
#include "../../source/core/slang-process.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

#include <stdio.h>
#include <stdlib.h>

using namespace Slang;

// Test that the reflection API provides correct info about attributes.

SLANG_UNIT_TEST(attributeReflection)
{
    const char* userSourceBody = R"(
        public enum E
        {
            V0,
            V1,
        };

        [__AttributeUsage(_AttributeTargets.Struct)]
        public struct NormalTextureAttribute
        {
            public E Type;
            public float x;
        };

        [COM("042BE50B-CB01-4DBB-8367-3A9CDCBE2F49")]
        interface IInterface { void f(); }

        [NormalTexture(E.V1, 6)]
        struct TS {};
        )";
    String userSource = userSourceBody;
    ComPtr<slang::IGlobalSession> globalSession;
    SLANG_CHECK(slang_createGlobalSession(SLANG_API_VERSION, globalSession.writeRef()) == SLANG_OK);
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_HLSL;
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

    auto reflection = module->getLayout();

    auto interfaceType = reflection->findTypeByName("IInterface");
    SLANG_CHECK(interfaceType != nullptr);

    auto comAttribute = interfaceType->findAttributeByName("COM");
    SLANG_CHECK(comAttribute != nullptr);

    size_t size = 0;
    auto guid = comAttribute->getArgumentValueString(0, &size);
    UnownedStringSlice stringSlice = UnownedStringSlice(guid, size);
    SLANG_CHECK(stringSlice == "042BE50B-CB01-4DBB-8367-3A9CDCBE2F49");

    auto testType = reflection->findTypeByName("TS");
    SLANG_CHECK(testType != nullptr);

    auto normalTextureAttribute = testType->findAttributeByName("NormalTexture");
    SLANG_CHECK(normalTextureAttribute != nullptr);

    int value = 0;
    normalTextureAttribute->getArgumentValueInt(0, &value);
    SLANG_CHECK(value == 1);

    float fvalue = 0;
    normalTextureAttribute->getArgumentValueFloat(1, &fvalue);
    SLANG_CHECK(fvalue == 6.0);
}
