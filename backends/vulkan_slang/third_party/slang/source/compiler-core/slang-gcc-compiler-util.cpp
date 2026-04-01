// slang-gcc-compiler-util.cpp
#include "slang-gcc-compiler-util.h"

#include "../core/slang-char-util.h"
#include "../core/slang-common.h"
#include "../core/slang-io.h"
#include "../core/slang-shared-library.h"
#include "../core/slang-string-slice-pool.h"
#include "../core/slang-string-util.h"
#include "slang-artifact-desc-util.h"
#include "slang-artifact-diagnostic-util.h"
#include "slang-artifact-representation-impl.h"
#include "slang-artifact-util.h"
#include "slang-com-helper.h"

namespace Slang
{

static Index _findVersionEnd(const UnownedStringSlice& in)
{
    Index numDots = 0;
    const Index len = in.getLength();

    for (Index i = 0; i < len; ++i)
    {
        const char c = in[i];
        if (CharUtil::isDigit(c))
        {
            continue;
        }
        if (c == '.')
        {
            if (numDots >= 2)
            {
                return i;
            }
            numDots++;
            continue;
        }
        return i;
    }
    return len;
}

/* static */ SlangResult GCCDownstreamCompilerUtil::parseVersion(
    const UnownedStringSlice& text,
    const UnownedStringSlice& prefix,
    DownstreamCompilerDesc& outDesc)
{
    List<UnownedStringSlice> lines;
    StringUtil::calcLines(text, lines);

    for (auto line : lines)
    {
        Index prefixIndex = line.indexOf(prefix);
        if (prefixIndex < 0)
        {
            continue;
        }

        const UnownedStringSlice remainingSlice =
            UnownedStringSlice(line.begin() + prefixIndex + prefix.getLength(), line.end()).trim();

        const Index versionEndIndex = _findVersionEnd(remainingSlice);
        if (versionEndIndex < 0)
        {
            return SLANG_FAIL;
        }

        const UnownedStringSlice versionSlice(
            remainingSlice.begin(),
            remainingSlice.begin() + versionEndIndex);

        // Version is in format 0.0.0
        List<UnownedStringSlice> split;
        StringUtil::split(versionSlice, '.', split);
        List<Int> digits;

        for (auto v : split)
        {
            Int version;
            SLANG_RETURN_ON_FAIL(StringUtil::parseInt(v, version));
            digits.add(version);
        }

        if (digits.getCount() < 2)
        {
            return SLANG_FAIL;
        }

        outDesc.version.set(int(digits[0]), int(digits[1]));
        return SLANG_OK;
    }

    return SLANG_FAIL;
}

SlangResult GCCDownstreamCompilerUtil::calcVersion(
    const ExecutableLocation& exe,
    DownstreamCompilerDesc& outDesc)
{
    CommandLine cmdLine;
    cmdLine.setExecutableLocation(exe);
    cmdLine.addArg("-v");

    ExecuteResult exeRes;
    SLANG_RETURN_ON_FAIL(ProcessUtil::execute(cmdLine, exeRes));

    // Note we now have builds that add other words in front of the version
    // such as "Ubuntu clang version"
    const UnownedStringSlice prefixes[] = {
        UnownedStringSlice::fromLiteral("clang version"),
        UnownedStringSlice::fromLiteral("gcc version"),
        UnownedStringSlice::fromLiteral("Apple LLVM version"),
        UnownedStringSlice::fromLiteral("Apple metal version"),

    };
    const SlangPassThrough types[] = {
        SLANG_PASS_THROUGH_CLANG,
        SLANG_PASS_THROUGH_GCC,
        SLANG_PASS_THROUGH_CLANG,
        SLANG_PASS_THROUGH_METAL,
    };

    SLANG_COMPILE_TIME_ASSERT(SLANG_COUNT_OF(prefixes) == SLANG_COUNT_OF(types));

    for (Index i = 0; i < SLANG_COUNT_OF(prefixes); ++i)
    {
        // Set the type
        outDesc.type = types[i];
        // Extract the version
        if (SLANG_SUCCEEDED(
                parseVersion(exeRes.standardError.getUnownedSlice(), prefixes[i], outDesc)))
        {
            return SLANG_OK;
        }
    }

    return SLANG_FAIL;
}

static SlangResult _parseSeverity(
    const UnownedStringSlice& in,
    ArtifactDiagnostic::Severity& outSeverity)
{
    typedef ArtifactDiagnostic::Severity Severity;

    if (in == "error" || in == "fatal error")
    {
        outSeverity = Severity::Error;
    }
    else if (in == "warning")
    {
        outSeverity = Severity::Warning;
    }
    else if (in == "info" || in == "note")
    {
        outSeverity = Severity::Info;
    }
    else
    {
        return SLANG_FAIL;
    }
    return SLANG_OK;
}

namespace
{ // anonymous

enum class LineParseResult
{
    Single,       ///< It's a single line
    Start,        ///< Line was the start of a message
    Continuation, ///< Not totally clear, add to previous line if nothing else hit
    Ignore,       ///< Ignore the line
};

} // namespace

static SlangResult _parseGCCFamilyLine(
    SliceAllocator& allocator,
    const UnownedStringSlice& line,
    LineParseResult& outLineParseResult,
    ArtifactDiagnostic& outDiagnostic)
{
    typedef ArtifactDiagnostic Diagnostic;
    typedef Diagnostic::Severity Severity;

    // Set to default case
    outLineParseResult = LineParseResult::Ignore;

    /* example error output from different scenarios */

    /*
        tests/cpp-compiler/c-compile-error.c: In function 'int main(int, char**)':
        tests/cpp-compiler/c-compile-error.c:8:13: error: 'b' was not declared in this scope
        int a = b + c;
        ^
        tests/cpp-compiler/c-compile-error.c:8:17: error: 'c' was not declared in this scope
        int a = b + c;
        ^
    */

    /* /tmp/ccS0JCWe.o:c-compile-link-error.c:(.rdata$.refptr.thing[.refptr.thing]+0x0): undefined
       reference to `thing' collect2: error: ld returned 1 exit status*/

    /*
     clang: warning: treating 'c' input as 'c++' when in C++ mode, this behavior is deprecated
     [-Wdeprecated] Undefined symbols for architecture x86_64:
     "_thing", referenced from:
     _main in c-compile-link-error-a83ace.o
     ld: symbol(s) not found for architecture x86_64
     clang: error: linker command failed with exit code 1 (use -v to see invocation) */

    /* /tmp/c-compile-link-error-ccf151.o: In function `main':
     c-compile-link-error.c:(.text+0x19): undefined reference to `thing'
    clang: error: linker command failed with exit code 1 (use -v to see invocation)
    */

    /* /tmp/c-compile-link-error-301c8c.o: In function `main':
       /home/travis/build/shader-slang/slang/tests/cpp-compiler/c-compile-link-error.c:10: undefined
       reference to `thing' clang-7: error: linker command failed with exit code 1 (use -v to see
       invocation)*/

    /*  /path/slang-cpp-prelude.h:4:10: fatal error: ../slang.h: No such file or directory
        #include "slang.h"
        ^~~~~~~~~~~~
        compilation terminated.*/

    /* g++: error: unrecognized command line option ‘-std=c++14’ */

    outDiagnostic.stage = Diagnostic::Stage::Compile;

    List<UnownedStringSlice> split;
    StringUtil::split(line, ':', split);

    // On windows we can have paths that are a: etc... if we detect this we can combine 0 - 1 to
    // be 1.
    if (split.getCount() > 1 && split[0].getLength() == 1)
    {
        const char c = split[0][0];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
        {
            // We'll assume it's a path
            UnownedStringSlice path(split[0].begin(), split[1].end());
            split.removeAt(0);
            split[0] = path;
        }
    }

    if (split.getCount() == 2)
    {
        const auto split0 = split[0].trim();
        if (split0 == UnownedStringSlice::fromLiteral("ld"))
        {
            // We'll ignore for now
            outDiagnostic.stage = Diagnostic::Stage::Link;
            outDiagnostic.severity = Severity::Info;
            outDiagnostic.text = allocator.allocate(split[1].trim());
            outLineParseResult = LineParseResult::Start;
            return SLANG_OK;
        }

        if (SLANG_SUCCEEDED(_parseSeverity(split0, outDiagnostic.severity)))
        {
            // Command line errors can be just contain 'error:' etc. Can be seen on apple/clang
            outDiagnostic.stage = Diagnostic::Stage::Compile;
            outDiagnostic.text = allocator.allocate(split[1].trim());
            outLineParseResult = LineParseResult::Single;
            return SLANG_OK;
        }

        outLineParseResult = LineParseResult::Ignore;
        return SLANG_OK;
    }
    else if (split.getCount() == 3)
    {
        const auto split0 = split[0].trim();
        const auto split1 = split[1].trim();
        const auto text = split[2].trim();

        // Check for special handling for clang or metal
        if (split0.startsWith(UnownedStringSlice::fromLiteral("clang")) ||
            split0.startsWith(UnownedStringSlice::fromLiteral("metal")) ||
            split0.startsWith(UnownedStringSlice::fromLiteral("Clang")) ||
            split0 == UnownedStringSlice::fromLiteral("g++") ||
            split0 == UnownedStringSlice::fromLiteral("gcc"))
        {
            // Extract the type
            SLANG_RETURN_ON_FAIL(_parseSeverity(split[1].trim(), outDiagnostic.severity));

            if (text.startsWith("linker command failed"))
            {
                outDiagnostic.stage = Diagnostic::Stage::Link;
            }

            outDiagnostic.text = allocator.allocate(text);
            outLineParseResult = LineParseResult::Start;
            return SLANG_OK;
        }
        else if (split1.startsWith("(.text"))
        {
            // This is a little weak... but looks like it's a link error
            outDiagnostic.filePath = allocator.allocate(split[0]);
            outDiagnostic.severity = Severity::Error;
            outDiagnostic.stage = Diagnostic::Stage::Link;
            outDiagnostic.text = allocator.allocate(text);
            outLineParseResult = LineParseResult::Single;
            return SLANG_OK;
        }
        else if (text.startsWith("ld returned"))
        {
            outDiagnostic.stage = ArtifactDiagnostic::Stage::Link;
            SLANG_RETURN_ON_FAIL(_parseSeverity(split[1].trim(), outDiagnostic.severity));
            outDiagnostic.text = allocator.allocate(line);
            outLineParseResult = LineParseResult::Single;
            return SLANG_OK;
        }
        else if (text == "")
        {
            // This is probably a prelude line, we'll just ignore it
            outLineParseResult = LineParseResult::Ignore;
            return SLANG_OK;
        }
    }
    else if (split.getCount() == 4)
    {
        // Probably a link error, give the source line
        String ext = Path::getPathExt(split[0]);

        // Maybe a bit fragile -> but probably okay for now
        if (ext != "o" && ext != "obj")
        {
            outLineParseResult = LineParseResult::Ignore;
            return SLANG_OK;
        }
        else
        {
            outDiagnostic.filePath = allocator.allocate(split[1]);
            outDiagnostic.location.line = 0;
            outDiagnostic.location.column = 0;
            outDiagnostic.severity = Diagnostic::Severity::Error;
            outDiagnostic.stage = Diagnostic::Stage::Link;
            outDiagnostic.text = allocator.allocate(split[3]);

            outLineParseResult = LineParseResult::Start;
            return SLANG_OK;
        }
    }
    else if (split.getCount() >= 5)
    {
        // Probably a regular error line
        SLANG_RETURN_ON_FAIL(_parseSeverity(split[3].trim(), outDiagnostic.severity));

        outDiagnostic.filePath = allocator.allocate(split[0]);
        SLANG_RETURN_ON_FAIL(StringUtil::parseInt(split[1], outDiagnostic.location.line));

        // Everything from 4 to the end is the error
        outDiagnostic.text = allocator.allocate(split[4].begin(), split.getLast().end());

        outLineParseResult = LineParseResult::Start;
        return SLANG_OK;
    }

    // Assume it's a continuation
    outLineParseResult = LineParseResult::Continuation;
    return SLANG_OK;
}

/* static */ SlangResult GCCDownstreamCompilerUtil::parseOutput(
    const ExecuteResult& exeRes,
    IArtifactDiagnostics* diagnostics)
{
    LineParseResult prevLineResult = LineParseResult::Ignore;

    SliceAllocator allocator;

    diagnostics->reset();
    diagnostics->setRaw(SliceUtil::asCharSlice(exeRes.standardError));

    // We hold in workDiagnostics so as it is more convenient to append to the last with a
    // continuation also means we don't hold the allocations of building up continuations, just the
    // results when finally allocated at the end
    List<ArtifactDiagnostic> workDiagnostics;

    for (auto line : LineParser(exeRes.standardError.getUnownedSlice()))
    {
        ArtifactDiagnostic diagnostic;

        LineParseResult lineRes;

        SLANG_RETURN_ON_FAIL(_parseGCCFamilyLine(allocator, line, lineRes, diagnostic));

        switch (lineRes)
        {
        case LineParseResult::Start:
            {
                // It's start of a new message
                workDiagnostics.add(diagnostic);
                prevLineResult = LineParseResult::Start;
                break;
            }
        case LineParseResult::Single:
            {
                // It's a single message, without anything following
                workDiagnostics.add(diagnostic);
                prevLineResult = LineParseResult::Ignore;
                break;
            }
        case LineParseResult::Continuation:
            {
                if (prevLineResult == LineParseResult::Start ||
                    prevLineResult == LineParseResult::Continuation)
                {
                    if (workDiagnostics.getCount() > 0)
                    {
                        auto& last = workDiagnostics.getLast();

                        // TODO(JS): Note that this is somewhat wasteful as every time we append we
                        // just allocate more memory to hold the result. If we had an allocator
                        // dedicated to 'text' we could perhaps just append to the end of the last
                        // allocation
                        //
                        // We are now in a continuation, add to the last
                        StringBuilder buf;
                        buf.append(asStringSlice(last.text));
                        buf.append("\n");
                        buf.append(line);

                        last.text = allocator.allocate(buf);
                    }
                    prevLineResult = LineParseResult::Continuation;
                }
                break;
            }
        case LineParseResult::Ignore:
            {
                prevLineResult = lineRes;
                break;
            }
        default:
            return SLANG_FAIL;
        }
    }

    for (const auto& diagnostic : workDiagnostics)
    {
        diagnostics->add(diagnostic);
    }

    if (diagnostics->hasOfAtLeastSeverity(ArtifactDiagnostic::Severity::Error) ||
        exeRes.resultCode != 0)
    {
        diagnostics->setResult(SLANG_FAIL);
    }

    return SLANG_OK;
}

/* static */ SlangResult GCCDownstreamCompilerUtil::calcCompileProducts(
    const CompileOptions& options,
    ProductFlags flags,
    IOSFileArtifactRepresentation* lockFile,
    List<ComPtr<IArtifact>>& outArtifacts)
{
    SLANG_ASSERT(options.modulePath.count);

    outArtifacts.clear();

    if (flags & ProductFlag::Execution)
    {
        StringBuilder builder;
        const auto desc = ArtifactDescUtil::makeDescForCompileTarget(options.targetType);
        SLANG_RETURN_ON_FAIL(
            ArtifactDescUtil::calcPathForDesc(desc, asStringSlice(options.modulePath), builder));

        auto fileRep = OSFileArtifactRepresentation::create(
            IOSFileArtifactRepresentation::Kind::Owned,
            builder.getUnownedSlice(),
            lockFile);
        auto artifact = ArtifactUtil::createArtifact(desc);
        artifact->addRepresentation(fileRep);

        outArtifacts.add(artifact);
    }

    return SLANG_OK;
}

/* static */ SlangResult GCCDownstreamCompilerUtil::calcArgs(
    const CompileOptions& options,
    CommandLine& cmdLine)
{
    SLANG_ASSERT(options.modulePath.count);

    PlatformKind platformKind = (options.platform == PlatformKind::Unknown)
                                    ? PlatformUtil::getPlatformKind()
                                    : options.platform;

    const auto targetDesc = ArtifactDescUtil::makeDescForCompileTarget(options.targetType);

    if (options.sourceLanguage == SLANG_SOURCE_LANGUAGE_CPP)
    {
        cmdLine.addArg("-fvisibility=hidden");

        // C++17 since we share headers with slang itself (which uses c++17)
        cmdLine.addArg("-std=c++17");
    }

    if (targetDesc.payload == ArtifactDesc::Payload::MetalAIR)
    {
        cmdLine.addArg("-std=metal3.1");
    }

    // Our generated code very often casts between dissimilar types with the
    // knowledge that they have the same representation. This is strictly
    // speaking UB, and GCC 10+ is happy to take advantage of this, stop it.
    cmdLine.addArg("-fno-strict-aliasing");

    // TODO(JS): Here we always set -m32 on x86. It could be argued it is only necessary when
    // creating a shared library but if we create an object file, we don't know what to choose
    // because we don't know what final usage is. It could also be argued that the platformKind
    // could define the actual desired target - but as it stands we only have a target of 'Linux'
    // (as opposed to Win32/64). Really it implies we need an arch enumeration too.
    //
    // For now we just make X86 binaries try and produce x86 compatible binaries as fixes the
    // immediate problems.
#if SLANG_PROCESSOR_X86
    /* Used to specify the processor more broadly. For a x86 binary we need to make sure we build
    x86 builds even when on an x64 system. -m32 -m64*/
    cmdLine.addArg("-m32");
#endif

    switch (options.optimizationLevel)
    {
    case OptimizationLevel::None:
        {
            // No optimization
            cmdLine.addArg("-O0");
            break;
        }
    case OptimizationLevel::Default:
        {
            cmdLine.addArg("-Os");
            break;
        }
    case OptimizationLevel::High:
        {
            cmdLine.addArg("-O2");
            break;
        }
    case OptimizationLevel::Maximal:
        {
            cmdLine.addArg("-O3");
            break;
        }
    default:
        break;
    }

    if (options.debugInfoType != DebugInfoType::None)
    {
        cmdLine.addArg("-g");
    }

    if (options.flags & CompileOptions::Flag::Verbose)
    {
        cmdLine.addArg("-v");
    }

    switch (options.floatingPointMode)
    {
    case FloatingPointMode::Default:
        break;
    case FloatingPointMode::Precise:
        {
            // cmdLine.addArg("-fno-unsafe-math-optimizations");
            break;
        }
    case FloatingPointMode::Fast:
        {
            // We could enable SSE with -mfpmath=sse
            // But that would only make sense on a x64/x86 type processor and only if that feature
            // is present (it is on all x64)
            cmdLine.addArg("-ffast-math");
            break;
        }
    }

    StringBuilder moduleFilePath;
    SLANG_RETURN_ON_FAIL(ArtifactDescUtil::calcPathForDesc(
        targetDesc,
        asStringSlice(options.modulePath),
        moduleFilePath));

    cmdLine.addArg("-o");
    cmdLine.addArg(moduleFilePath);

    switch (options.targetType)
    {
    case SLANG_SHADER_SHARED_LIBRARY:
    case SLANG_HOST_SHARED_LIBRARY:
        {
            // Shared library
            cmdLine.addArg("-shared");

            if (PlatformUtil::isFamily(PlatformFamily::Unix, platformKind))
            {
                // Position independent
                cmdLine.addArg("-fPIC");
            }
            break;
        }
    case SLANG_HOST_EXECUTABLE:
        {
            cmdLine.addArg("-rdynamic");
            break;
        }
    case SLANG_OBJECT_CODE:
        {
            // Don't link, just produce object file
            cmdLine.addArg("-c");
            break;
        }
    default:
        break;
    }

    // Add defines
    for (const auto& define : options.defines)
    {
        StringBuilder builder;

        builder << "-D";
        builder << define.nameWithSig;
        if (define.value.count)
        {
            builder << "=" << asStringSlice(define.value);
        }

        cmdLine.addArg(builder);
    }

    // Add includes
    for (const auto& include : options.includePaths)
    {
        cmdLine.addArg("-I");
        cmdLine.addArg(asString(include));
    }

    // Link options
    if (0) // && options.targetType != TargetType::Object)
    {
        // linkOptions << "-Wl,";
        // cmdLine.addArg(linkOptions);
    }

    if (options.targetType == SLANG_SHADER_SHARED_LIBRARY)
    {
        if (!PlatformUtil::isFamily(PlatformFamily::Apple, platformKind))
        {
            // On MacOS, this linker option is not supported. That's ok though in
            // so far as on MacOS it does report any unfound symbols without the option.

            // Linker flag to report any undefined symbols as a link error
            cmdLine.addArg("-Wl,--no-undefined");
        }
    }

    // Files to compile, need to be on the file system.
    for (IArtifact* sourceArtifact : options.sourceArtifacts)
    {
        ComPtr<IOSFileArtifactRepresentation> fileRep;

        // TODO(JS):
        // Do we want to keep the file on the file system? It's probably reasonable to do so.
        SLANG_RETURN_ON_FAIL(sourceArtifact->requireFile(ArtifactKeep::Yes, fileRep.writeRef()));
        cmdLine.addArg(fileRep->getPath());
    }

    // Add the library paths

    if (options.libraryPaths.count && (options.targetType == SLANG_HOST_EXECUTABLE))
    {
        if (PlatformUtil::isFamily(PlatformFamily::Apple, platformKind))
            cmdLine.addArg("-Wl,-rpath,@loader_path,-rpath,@loader_path/../lib");
        else
            cmdLine.addArg("-Wl,-rpath,$ORIGIN,-rpath,$ORIGIN/../lib");
    }

    StringSlicePool libPathPool(StringSlicePool::Style::Default);

    for (const auto& libPath : options.libraryPaths)
    {
        libPathPool.add(libPath);
    }

    // Artifacts might add library paths
    for (IArtifact* artifact : options.libraries)
    {
        const auto artifactDesc = artifact->getDesc();
        // If it's a library for CPU types, try and use it
        if (ArtifactDescUtil::isCpuBinary(artifactDesc) &&
            artifactDesc.kind == ArtifactKind::Library)
        {
            ComPtr<IOSFileArtifactRepresentation> fileRep;

            // Get the name and path (can be empty) to the library
            SLANG_RETURN_ON_FAIL(artifact->requireFile(ArtifactKeep::Yes, fileRep.writeRef()));

            const UnownedStringSlice path(fileRep->getPath());
            libPathPool.add(Path::getParentDirectory(path));

            cmdLine.addPrefixPathArg(
                "-l",
                ArtifactDescUtil::getBaseNameFromPath(artifact->getDesc(), path));
        }
    }

    if (options.sourceLanguage == SLANG_SOURCE_LANGUAGE_CPP &&
        !PlatformUtil::isFamily(PlatformFamily::Windows, platformKind))
    {
        // Make STD libs available
        cmdLine.addArg("-lstdc++");
        // Make maths lib available
        cmdLine.addArg("-lm");
    }

    for (const auto& libPath : libPathPool.getAdded())
    {
        // Note that any escaping of the path is handled in the ProcessUtil::
        cmdLine.addArg("-L");
        cmdLine.addArg(libPath);
        cmdLine.addArg("-F");
        cmdLine.addArg(libPath);
    }

    // Add compiler specific options from user.
    for (auto compilerSpecificArg : options.compilerSpecificArguments)
    {
        const char* const arg = compilerSpecificArg;
        cmdLine.addArg(arg);
    }

    return SLANG_OK;
}

/* static */ SlangResult GCCDownstreamCompilerUtil::createCompiler(
    const ExecutableLocation& exe,
    ComPtr<IDownstreamCompiler>& outCompiler)
{
    DownstreamCompilerDesc desc;
    SLANG_RETURN_ON_FAIL(GCCDownstreamCompilerUtil::calcVersion(exe, desc));

    auto compiler = new GCCDownstreamCompiler(desc);
    ComPtr<IDownstreamCompiler> compilerIntf(compiler);
    compiler->m_cmdLine.setExecutableLocation(exe);

    outCompiler.swap(compilerIntf);
    return SLANG_OK;
}

/* static */ SlangResult GCCDownstreamCompilerUtil::locateGCCCompilers(
    const String& path,
    ISlangSharedLibraryLoader* loader,
    DownstreamCompilerSet* set)
{
    SLANG_UNUSED(loader);

    ComPtr<IDownstreamCompiler> compiler;
    if (SLANG_SUCCEEDED(createCompiler(ExecutableLocation(path, "g++"), compiler)))
    {
        // A downstream compiler for Slang must currently support C++17 - such that
        // the prelude and generated code works.
        //
        // The first version of gcc that supports stable `-std=c++17` is 9.0
        // https://gcc.gnu.org/projects/cxx-status.html

        auto desc = compiler->getDesc();
        if (desc.version.m_major < 9)
        {
            // If the version isn't 9 or higher, we don't add this version of the compiler.
            return SLANG_OK;
        }

        set->addCompiler(compiler);
    }
    return SLANG_OK;
}

/* static */ SlangResult GCCDownstreamCompilerUtil::locateClangCompilers(
    const String& path,
    ISlangSharedLibraryLoader* loader,
    DownstreamCompilerSet* set)
{
    SLANG_UNUSED(loader);

    ComPtr<IDownstreamCompiler> compiler;
    if (SLANG_SUCCEEDED(createCompiler(ExecutableLocation(path, "clang"), compiler)))
    {
        set->addCompiler(compiler);
    }
    return SLANG_OK;
}

} // namespace Slang
