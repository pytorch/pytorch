#ifndef SLANG_GCC_COMPILER_UTIL_H
#define SLANG_GCC_COMPILER_UTIL_H

#include "slang-downstream-compiler-util.h"

namespace Slang
{

/* Utility for processing input and output of gcc-like compilers, including clang */
struct GCCDownstreamCompilerUtil : public DownstreamCompilerUtilBase
{
    /// Extracts version number into desc from text (assumes gcc/clang -v layout with a line with
    /// version)
    static SlangResult parseVersion(
        const UnownedStringSlice& text,
        const UnownedStringSlice& prefix,
        DownstreamCompilerDesc& outDesc);

    /// Runs the exe, and extracts the version info into outDesc
    static SlangResult calcVersion(const ExecutableLocation& exe, DownstreamCompilerDesc& outDesc);

    /// Calculate gcc family compilers (including clang) cmdLine arguments from options
    static SlangResult calcArgs(const CompileOptions& options, CommandLine& cmdLine);

    /// Parse ExecuteResult into diagnostics
    static SlangResult parseOutput(const ExecuteResult& exeRes, IArtifactDiagnostics* diagnostics);

    /// Given options, calculate paths to products/files produced for a compilation
    static SlangResult calcCompileProducts(
        const CompileOptions& options,
        ProductFlags flags,
        IOSFileArtifactRepresentation* lockFile,
        List<ComPtr<IArtifact>>& outArtifacts);

    /// Given the exe location, creates a DownstreamCompiler.
    /// Note! Invoke/s the compiler  to determine the compiler version number.
    static SlangResult createCompiler(
        const ExecutableLocation& exe,
        ComPtr<IDownstreamCompiler>& outCompiler);

    /// Finds GCC compiler/s and adds them to the set
    static SlangResult locateGCCCompilers(
        const String& path,
        ISlangSharedLibraryLoader* loader,
        DownstreamCompilerSet* set);

    /// Finds clang compiler/s and adds them to the set
    static SlangResult locateClangCompilers(
        const String& path,
        ISlangSharedLibraryLoader* loader,
        DownstreamCompilerSet* set);
};

class GCCDownstreamCompiler : public CommandLineDownstreamCompiler
{
public:
    typedef CommandLineDownstreamCompiler Super;
    typedef GCCDownstreamCompilerUtil Util;

    // CommandLineCPPCompiler impl  - just forwards to the Util
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
        DownstreamProductFlags flags,
        IOSFileArtifactRepresentation* lockFile,
        List<ComPtr<IArtifact>>& outArtifacts) SLANG_OVERRIDE
    {
        return Util::calcCompileProducts(options, flags, lockFile, outArtifacts);
    }

    GCCDownstreamCompiler(const Desc& desc)
        : Super(desc)
    {
    }
};

} // namespace Slang

#endif
