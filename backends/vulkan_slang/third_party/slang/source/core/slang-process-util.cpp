// slang-process-util.cpp

#include "slang-process-util.h"

#include "slang-com-helper.h"
#include "slang-string-escape-util.h"
#include "slang-string-util.h"
#include "slang-string.h"

namespace Slang
{

/* static */ SlangResult ProcessUtil::execute(
    const CommandLine& commandLine,
    ExecuteResult& outExecuteResult)
{
    RefPtr<Process> process;
    SLANG_RETURN_ON_FAIL(Process::create(commandLine, 0, process));
    SLANG_RETURN_ON_FAIL(readUntilTermination(process, outExecuteResult));
    return SLANG_OK;
}

static Index _getCount(List<Byte>* buf)
{
    return buf ? buf->getCount() : 0;
}

// We may want something more sophisticated here, if bytes is something other than ascii/utf8
static String _getText(const ConstArrayView<Byte>& bytes)
{
    StringBuilder buf;
    StringUtil::appendStandardLines(
        UnownedStringSlice((const char*)bytes.begin(), (const char*)bytes.end()),
        buf);
    return buf.produceString();
}

/* static */ SlangResult ProcessUtil::readUntilTermination(
    Process* process,
    ExecuteResult& outExecuteResult)
{
    List<Byte> stdOut;
    List<Byte> stdError;

    SLANG_RETURN_ON_FAIL(readUntilTermination(process, &stdOut, &stdError));

    // Get the return code
    outExecuteResult.resultCode = ExecuteResult::ResultCode(process->getReturnValue());

    outExecuteResult.standardOutput = _getText(stdOut.getArrayView());
    outExecuteResult.standardError = _getText(stdError.getArrayView());

    return SLANG_OK;
}

/* static */ SlangResult ProcessUtil::readUntilTermination(
    Process* process,
    List<Byte>* outStdOut,
    List<Byte>* outStdError)
{
    Stream* stdOutStream = process->getStream(StdStreamType::Out);
    Stream* stdErrorStream = process->getStream(StdStreamType::ErrorOut);

    while (!process->isTerminated())
    {
        const auto preCount = _getCount(outStdOut) + _getCount(outStdError);

        SLANG_RETURN_ON_FAIL(StreamUtil::readOrDiscard(stdOutStream, 0, outStdOut));
        if (stdErrorStream)
            SLANG_RETURN_ON_FAIL(StreamUtil::readOrDiscard(stdErrorStream, 0, outStdError));

        const auto postCount = _getCount(outStdOut) + _getCount(outStdError);

        // If nothing was read, we can yield
        if (preCount == postCount)
        {
            Process::sleepCurrentThread(0);
        }
    }

    // Read anything remaining
    for (;;)
    {
        const auto preCount = _getCount(outStdOut) + _getCount(outStdError);
        StreamUtil::readOrDiscard(stdOutStream, 0, outStdOut);
        if (stdErrorStream)
            StreamUtil::readOrDiscard(stdErrorStream, 0, outStdError);
        const auto postCount = _getCount(outStdOut) + _getCount(outStdError);
        if (preCount == postCount)
            break;
    }
    return SLANG_OK;
}

} // namespace Slang
