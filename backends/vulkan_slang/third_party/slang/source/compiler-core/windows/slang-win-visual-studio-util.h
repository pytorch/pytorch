#ifndef SLANG_WIN_VISUAL_STUDIO_UTIL_H
#define SLANG_WIN_VISUAL_STUDIO_UTIL_H

#include "../../core/slang-list.h"
#include "../../core/slang-process-util.h"
#include "../../core/slang-string.h"
#include "../slang-downstream-compiler-util.h"

namespace Slang
{

struct WinVisualStudioUtil
{
    struct VersionPath
    {
        SemanticVersion version; ///< The visual studio version
        String vcvarsPath; ///< The path to `vcvars.bat` files, that need to be executed before
                           ///< executing the compiler
    };

    ///  Find all the installations
    static SlangResult find(List<VersionPath>& outVersionPaths);

    /// Find and add to the set (if not already there)
    static SlangResult find(DownstreamCompilerSet* set);

    /// Create the cmdLine to start compiler for specified path
    static void calcExecuteCompilerArgs(const VersionPath& versionPath, CommandLine& outCmdLine);

    /// Run visual studio on specified path with the parameters specified on the command line.
    /// Output placed in outResult.
    static SlangResult executeCompiler(
        const VersionPath& versionPath,
        const CommandLine& commandLine,
        ExecuteResult& outResult);

    /// Gets the msc compiler used to compile this version.
    static DownstreamCompilerMatchVersion getCompiledVersion();
};

} // namespace Slang

#endif
