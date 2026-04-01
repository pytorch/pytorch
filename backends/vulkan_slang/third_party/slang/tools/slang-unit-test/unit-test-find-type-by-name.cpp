// unit-test-find-type-by-name.cpp

#include "slang.h"
#include "unit-test/slang-unit-test.h"

#include <stdio.h>
#include <stdlib.h>

using namespace Slang;

SLANG_UNIT_TEST(findTypeByName)
{
    const char* testSource = "struct TestStruct {"
                             "   int member0;"
                             "   Texture2D texture1;"
                             "};";
    auto session = spCreateSession();
    auto request = spCreateCompileRequest(session);
    spAddCodeGenTarget(request, SLANG_DXBC);
    int tuIndex = spAddTranslationUnit(request, SLANG_SOURCE_LANGUAGE_SLANG, "tu1");
    spAddTranslationUnitSourceString(request, tuIndex, "internalFile", testSource);
    spCompile(request);

    auto testBody = [&]()
    {
        auto reflection = slang::ShaderReflection::get(request);
        auto testStruct = reflection->findTypeByName("TestStruct");
        SLANG_CHECK_ABORT(testStruct->getFieldCount() == 2);
        auto field0Name = testStruct->getFieldByIndex(0)->getName();
        SLANG_CHECK_ABORT(field0Name != nullptr && strcmp(field0Name, "member0") == 0);
        auto field1Name = testStruct->getFieldByIndex(1)->getName();
        SLANG_CHECK_ABORT(field1Name != nullptr && strcmp(field1Name, "texture1") == 0);

        auto intType = reflection->findTypeByName("int");
        auto intTypeName = intType->getName();
        SLANG_CHECK_ABORT(intTypeName && strcmp(intTypeName, "int") == 0);

        auto paramBlockType = reflection->findTypeByName("ParameterBlock<TestStruct>");
        SLANG_CHECK_ABORT(paramBlockType != nullptr);
        auto paramBlockTypeName = paramBlockType->getName();
        SLANG_CHECK_ABORT(paramBlockTypeName && strcmp(paramBlockTypeName, "ParameterBlock") == 0);
        auto paramBlockElementType = paramBlockType->getElementType();
        SLANG_CHECK_ABORT(paramBlockElementType != nullptr);
        auto paramBlockElementTypeName = paramBlockElementType->getName();
        SLANG_CHECK_ABORT(
            paramBlockElementTypeName && strcmp(paramBlockElementTypeName, "TestStruct") == 0);
    };

    testBody();

    spDestroyCompileRequest(request);
    spDestroySession(session);
}
