// slang-reflection-test-main.cpp

#include "../../source/compiler-core/slang-pretty-writer.h"
#include "../../source/core/slang-char-util.h"
#include "../../source/core/slang-string-escape-util.h"
#include "../../source/core/slang-string-util.h"
#include "../../source/core/slang-test-tool-util.h"
#include "slang-com-helper.h"
#include "slang.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace Slang;

void emitReflectionJSON(SlangCompileRequest* request, SlangReflection* reflection)
{
    ComPtr<ISlangBlob> b;

    spReflection_ToJson(reflection, request, b.writeRef());

    // Output the writer content to out stream
    StdWriters::getOut().write((const char*)b->getBufferPointer(), b->getBufferSize());
}

static SlangResult maybeDumpDiagnostic(SlangResult res, SlangCompileRequest* request)
{
    const char* diagnostic;
    if (SLANG_FAILED(res) && (diagnostic = spGetDiagnosticOutput(request)))
    {
        StdWriters::getError().put(diagnostic);
    }
    return res;
}

SlangResult performCompilationAndReflection(
    SlangCompileRequest* request,
    int argc,
    const char* const* argv)
{
    SLANG_RETURN_ON_FAIL(
        maybeDumpDiagnostic(spProcessCommandLineArguments(request, &argv[1], argc - 1), request));
    SLANG_RETURN_ON_FAIL(maybeDumpDiagnostic(spCompile(request), request));

    // Okay, let's go through and emit reflection info on whatever
    // we have.

    SlangReflection* reflection = spGetReflection(request);
    emitReflectionJSON(request, reflection);

    return SLANG_OK;
}

SLANG_TEST_TOOL_API SlangResult
innerMain(Slang::StdWriters* stdWriters, SlangSession* session, int argc, const char* const* argv)
{
    Slang::StdWriters::setSingleton(stdWriters);

    SlangCompileRequest* request = spCreateCompileRequest(session);
    for (int i = 0; i < SLANG_WRITER_CHANNEL_COUNT_OF; ++i)
    {
        const auto channel = SlangWriterChannel(i);
        spSetWriter(request, channel, stdWriters->getWriter(channel));
    }

    char const* appName = "slang-reflection-test";
    if (argc > 0)
        appName = argv[0];

    SlangResult res = performCompilationAndReflection(request, argc, argv);

    spDestroyCompileRequest(request);

    return res;
}

int main(int argc, char** argv)
{
    using namespace Slang;

    SlangSession* session = spCreateSession(nullptr);

    auto stdWriters = StdWriters::initDefaultSingleton();

    SlangResult res = innerMain(stdWriters, session, argc, argv);
    spDestroySession(session);
    slang::shutdown();
    return SLANG_FAILED(res) ? 1 : 0;
}
