// unit-test-module-ptr.cpp

#include "core/slang-memory-file-system.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

#include <stdio.h>
#include <stdlib.h>

using namespace Slang;

SLANG_UNIT_TEST(modulePtr)
{
    const char* testModuleSource = R"(
        module test_module;

        public void atomicFunc(__ref Atomic<int> ptr) {
            ptr.add(1);
        }
    )";

    const char* testSource = R"(
        import "test_module";

        RWStructuredBuffer<Atomic<int>> input0;

        [shader("compute")]
        [numthreads(1,1,1)]
        void computeMain(uint3 workGroup : SV_GroupID)
        {
            atomicFunc(input0[0]);
        }
    )";
    ComPtr<ISlangMutableFileSystem> memoryFileSystem =
        ComPtr<ISlangMutableFileSystem>(new Slang::MemoryFileSystem());

    ComPtr<slang::IGlobalSession> globalSession;
    SLANG_CHECK(slang_createGlobalSession(SLANG_API_VERSION, globalSession.writeRef()) == SLANG_OK);
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_SPIRV;
    targetDesc.profile = globalSession->findProfile("spirv_1_5");
    slang::SessionDesc sessionDesc = {};
    sessionDesc.targetCount = 1;
    sessionDesc.targets = &targetDesc;
    sessionDesc.compilerOptionEntryCount = 0;
    sessionDesc.fileSystem = memoryFileSystem;

    // Precompile test_module to file.
    {
        ComPtr<slang::ISession> session;
        SLANG_CHECK(globalSession->createSession(sessionDesc, session.writeRef()) == SLANG_OK);

        ComPtr<slang::IBlob> diagnosticBlob;
        auto module = session->loadModuleFromSourceString(
            "test_module",
            "test_module.slang",
            testModuleSource,
            diagnosticBlob.writeRef());
        SLANG_CHECK(module != nullptr);

        ComPtr<slang::IBlob> moduleBlob;
        module->serialize(moduleBlob.writeRef());
        memoryFileSystem->saveFile(
            "test_module.slang-module",
            moduleBlob->getBufferPointer(),
            moduleBlob->getBufferSize());
    }

    // compile test.
    {
        ComPtr<slang::ISession> session;
        SLANG_CHECK(globalSession->createSession(sessionDesc, session.writeRef()) == SLANG_OK);

        ComPtr<slang::IBlob> diagnosticBlob;
        auto module = session->loadModuleFromSourceString(
            "test",
            "test.slang",
            testSource,
            diagnosticBlob.writeRef());
        SLANG_CHECK(module != nullptr);

        ComPtr<slang::IComponentType> linkedProgram;
        module->link(linkedProgram.writeRef());

        ComPtr<slang::IBlob> code;

        linkedProgram->getTargetCode(0, code.writeRef(), diagnosticBlob.writeRef());

        SLANG_CHECK(code->getBufferSize() > 0);
    }
}
