#ifndef SLANG_CORE_TEST_TOOL_UTIL_H
#define SLANG_CORE_TEST_TOOL_UTIL_H

#include "slang-std-writers.h"

namespace Slang
{

#ifdef SLANG_SHARED_LIBRARY_TOOL
#define SLANG_TEST_TOOL_API SLANG_EXTERN_C SLANG_DLL_EXPORT
#else
#define SLANG_TEST_TOOL_API
#endif

/* When a tool is run as an executable the return code is the code returned from
the last return of main. On unix this can be up to 8 bits.
By normal command line tool conventions returning 0 means success. */
enum class ToolReturnCode
{
    CompilationFailed = -1, ///< Compilation failure (-1 to maintain compatibility). This may still
                            ///< produce output and may mean a test was successful.
    Success = 0,            ///< Tool ran normally
    Failed,                 ///< Tool failed
    Ignored, ///< The run was ignored because it couldn't be run (because some optional feature was
             ///< not present for example)
    FailedToRun, ///< Could not even run the test
};

enum class ToolReturnCodeSpan
{
    // Span of all valid values
    First = int(ToolReturnCode::CompilationFailed),
    Last = int(ToolReturnCode::FailedToRun),
    // Span of all values that indicate the test is 'done'
    FirstIsDone = int(ToolReturnCode::Ignored),
    LastIsDone = int(ToolReturnCode::FailedToRun)
};

/* Utility functions for 'test tools' */
struct TestToolUtil
{
    typedef SlangResult (*InnerMainFunc)(
        Slang::StdWriters* stdWriters,
        SlangSession* session,
        int argc,
        const char* const* argv);

    /// If the test failed to run or was ignored then we are done
    static bool isDone(ToolReturnCode code)
    {
        return int(code) >= int(ToolReturnCodeSpan::FirstIsDone) &&
               int(code) <= int(ToolReturnCodeSpan::LastIsDone);
    }

    /// Convert from an int
    static ToolReturnCode getReturnCodeFromInt(int code);

    /// Given a slang result, returns a return code that can be returned from an executable
    static ToolReturnCode getReturnCode(SlangResult res);

    /// Given the executable path (as located in Slang directory hierarchy), works out the absolute
    /// path to the root
    static SlangResult getRootPath(const char* exePath, String& outRootPath);

    /// Given the exePath, give return the absolute path to the directory the exe is in
    static SlangResult getExeDirectoryPath(const char* exePath, String& outExeDirectoryPath);

    /// Sets the default preludes on the session based on an explicit path
    static SlangResult setSessionDefaultPreludeFromRootPath(
        const String& rootPath,
        slang::IGlobalSession* session);

    /// Calculates the path that is the combination of parentPath, and relPath
    /// And converts such that can be used as an include path (handling slashes)
    static SlangResult getIncludePath(
        const String& parentPath,
        const char* relPath,
        String& outIncludePath);


    /// Sets the default preludes on the session based on the executable path
    static SlangResult setSessionDefaultPreludeFromExePath(
        const char* exePath,
        slang::IGlobalSession* session);

    /// Returns true if the core module should not be initialized immediately (eg when doing a
    /// -load-core-module).
    static bool hasDeferredCoreModule(Index numArgs, const char* const* args);

    static SlangResult getDllDirectoryPath(const char* exePath, String& outDllDirectoryPath);
};

} // namespace Slang

#endif // SLANG_TEST_TOOL_H
