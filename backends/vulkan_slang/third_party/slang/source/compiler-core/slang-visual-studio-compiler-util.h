#ifndef SLANG_VISUAL_STUDIO_COMPILER_UTIL_H
#define SLANG_VISUAL_STUDIO_COMPILER_UTIL_H

#include "slang-downstream-compiler-util.h"

namespace Slang
{


struct VisualStudioCompilerUtil : public DownstreamCompilerUtilBase
{
    /// Calculate Visual Studio family compilers cmdLine arguments from options
    static SlangResult calcArgs(const CompileOptions& options, CommandLine& cmdLine);
    /// Parse Visual Studio exeRes into CPPCompiler::Output
    static SlangResult parseOutput(const ExecuteResult& exeRes, IArtifactDiagnostics* outOutput);

    static SlangResult calcCompileProducts(
        const CompileOptions& options,
        ProductFlags flags,
        IOSFileArtifactRepresentation* lockFile,
        List<ComPtr<IArtifact>>& outArtifacts);

    static SlangResult locateCompilers(
        const String& path,
        ISlangSharedLibraryLoader* loader,
        DownstreamCompilerSet* set);
};

class VisualStudioDownstreamCompiler : public CommandLineDownstreamCompiler
{
public:
    typedef CommandLineDownstreamCompiler Super;
    typedef VisualStudioCompilerUtil Util;

    // CommandLineDownstreamCompiler impl  - just forwards to the Util
    virtual SlangResult calcArgs(const CompileOptions& options, CommandLine& cmdLine) SLANG_OVERRIDE
    {
        return Util::calcArgs(options, cmdLine);
    }
    virtual SlangResult parseOutput(
        const ExecuteResult& exeResult,
        IArtifactDiagnostics* diagnostics) SLANG_OVERRIDE
    {
        return Util::parseOutput(exeResult, diagnostics);
    }
    virtual SlangResult calcCompileProducts(
        const CompileOptions& options,
        DownstreamProductFlags productFlags,
        IOSFileArtifactRepresentation* lockFile,
        List<ComPtr<IArtifact>>& outArtifacts) SLANG_OVERRIDE
    {
        return Util::calcCompileProducts(options, productFlags, lockFile, outArtifacts);
    }

    VisualStudioDownstreamCompiler(const Desc& desc)
        : Super(desc)
    {
    }
};


} // namespace Slang

#endif
