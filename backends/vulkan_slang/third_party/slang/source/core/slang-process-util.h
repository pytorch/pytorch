// slang-process-util.h
#ifndef SLANG_PROCESS_UTIL_H
#define SLANG_PROCESS_UTIL_H

#include "slang-process.h"

namespace Slang
{

struct ExecuteResult
{
    typedef int ResultCode;

    void init()
    {
        resultCode = 0;
        standardOutput = String();
        standardError = String();
    }

    ResultCode resultCode;
    String standardOutput;
    String standardError;
};

struct ProcessUtil
{
    /// Execute the command line
    static SlangResult execute(const CommandLine& commandLine, ExecuteResult& outExecuteResult);

    /// Read from read from streams until process terminates.
    /// Passing nullptr for a stream, will just discard what's in the stream
    static SlangResult readUntilTermination(
        Process* process,
        List<Byte>* outStdOut,
        List<Byte>* stdError);

    /// Read streams from process.
    static SlangResult readUntilTermination(Process* process, ExecuteResult& outExecuteResult);
};

} // namespace Slang

#endif // SLANG_PROCESS_UTIL_H
