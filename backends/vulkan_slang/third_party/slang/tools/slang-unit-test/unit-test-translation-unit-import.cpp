// unit-test-translation-unit-import.cpp

#include "../../source/core/slang-io.h"
#include "../../source/core/slang-process.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

#include <stdio.h>
#include <stdlib.h>

using namespace Slang;

// Test that the API supports discovering previously checked translation unit in the same
// FrontEndCompileRequest.
SLANG_UNIT_TEST(translationUnitImport)
{
    // Source for the first translation unit.
    const char* generatedSource = "public int f() {"
                                  "   return 5;"
                                  "};";

    // Source for the a file that imports the first translation unit.
    // The import should succeed and `f` should be visible to this module.
    const char* fileSource =
        R"(
        import generatedUnit;

        public int g(){ return f(); }
        )";

    // Source for a module that transitively uses the generated source via a file.
    const char* userSourceBody = R"(
        [shader("compute")]
        [numthreads(4,1,1)]
        void computeMain(
            uint3 sv_dispatchThreadID : SV_DispatchThreadID,
            uniform RWStructuredBuffer<int> buffer)
        {
            buffer[sv_dispatchThreadID.x] = g();
        })";

    auto moduleName = "moduleG" + String(Process::getId());
    String userSource = "import " + moduleName + ";\n" + userSourceBody;
    auto session = spCreateSession();
    auto request = spCreateCompileRequest(session);

    File::writeAllText(moduleName + ".slang", fileSource);

    spAddCodeGenTarget(request, SLANG_HLSL);
    int generatedTranslationUnitIndex =
        spAddTranslationUnit(request, SLANG_SOURCE_LANGUAGE_SLANG, "generatedUnit");
    spAddTranslationUnitSourceString(
        request,
        generatedTranslationUnitIndex,
        "generatedFile",
        generatedSource);

    int entryPointTranslationUnitIndex =
        spAddTranslationUnit(request, SLANG_SOURCE_LANGUAGE_SLANG, "userUnit");
    spAddTranslationUnitSourceString(
        request,
        entryPointTranslationUnitIndex,
        "userFile",
        userSource.getUnownedSlice().begin());
    spAddEntryPoint(request, entryPointTranslationUnitIndex, "computeMain", SLANG_STAGE_COMPUTE);

    auto compileResult = spCompile(request);
    SLANG_CHECK(compileResult == SLANG_OK);

    Slang::ComPtr<ISlangBlob> outBlob;
    spGetEntryPointCodeBlob(request, 0, 0, outBlob.writeRef());
    SLANG_CHECK(outBlob && outBlob->getBufferSize() != 0);

    spDestroyCompileRequest(request);
    spDestroySession(session);
    File::remove(moduleName + ".slang");
}
