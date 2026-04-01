// slang-visual-studio-compiler-util.cpp
#include "slang-visual-studio-compiler-util.h"

#include "../core/slang-common.h"
#include "../core/slang-string-slice-pool.h"
#include "../core/slang-string-util.h"
#include "slang-com-helper.h"

// if Visual Studio import the visual studio platform specific header
#if SLANG_VC
#include "windows/slang-win-visual-studio-util.h"
#endif

#include "../core/slang-io.h"
#include "slang-artifact-desc-util.h"
#include "slang-artifact-diagnostic-util.h"
#include "slang-artifact-representation-impl.h"
#include "slang-artifact-util.h"

namespace Slang
{

static void _addFile(
    const String& path,
    const ArtifactDesc& desc,
    IOSFileArtifactRepresentation* lockFile,
    List<ComPtr<IArtifact>>& outArtifacts)
{
    auto fileRep = OSFileArtifactRepresentation::create(
        IOSFileArtifactRepresentation::Kind::Owned,
        path.getUnownedSlice(),
        lockFile);
    auto artifact = ArtifactUtil::createArtifact(desc);
    artifact->addRepresentation(fileRep);

    outArtifacts.add(artifact);
}

/* static */ SlangResult VisualStudioCompilerUtil::calcCompileProducts(
    const CompileOptions& options,
    ProductFlags flags,
    IOSFileArtifactRepresentation* lockFile,
    List<ComPtr<IArtifact>>& outArtifacts)
{
    SLANG_ASSERT(options.modulePath.count);

    const String modulePath = asString(options.modulePath);

    const auto targetDesc = ArtifactDescUtil::makeDescForCompileTarget(options.targetType);

    outArtifacts.clear();

    if (flags & ProductFlag::Execution)
    {
        StringBuilder builder;
        const auto desc = ArtifactDescUtil::makeDescForCompileTarget(options.targetType);
        SLANG_RETURN_ON_FAIL(
            ArtifactDescUtil::calcPathForDesc(desc, modulePath.getUnownedSlice(), builder));

        _addFile(builder, desc, lockFile, outArtifacts);
    }
    if (flags & ProductFlag::Miscellaneous)
    {

        _addFile(
            modulePath + ".ilk",
            ArtifactDesc::make(
                ArtifactKind::BinaryFormat,
                ArtifactPayload::Unknown,
                ArtifactStyle::None),
            lockFile,
            outArtifacts);

        if (options.targetType == SLANG_SHADER_SHARED_LIBRARY)
        {
            _addFile(
                modulePath + ".exp",
                ArtifactDesc::make(
                    ArtifactKind::BinaryFormat,
                    ArtifactPayload::Unknown,
                    ArtifactStyle::None),
                lockFile,
                outArtifacts);
            _addFile(
                modulePath + ".lib",
                ArtifactDesc::make(ArtifactKind::Library, ArtifactPayload::HostCPU, targetDesc),
                lockFile,
                outArtifacts);
        }
    }
    if (flags & ProductFlag::Compile)
    {
        _addFile(
            modulePath + ".obj",
            ArtifactDesc::make(ArtifactKind::ObjectCode, ArtifactPayload::HostCPU, targetDesc),
            lockFile,
            outArtifacts);
    }
    if (flags & ProductFlag::Debug)
    {
        // TODO(JS): Could try and determine based on debug information
        _addFile(
            modulePath + ".pdb",
            ArtifactDesc::make(
                ArtifactKind::BinaryFormat,
                ArtifactPayload::PdbDebugInfo,
                targetDesc),
            lockFile,
            outArtifacts);
    }

    return SLANG_OK;
}

/* static */ SlangResult VisualStudioCompilerUtil::calcArgs(
    const CompileOptions& options,
    CommandLine& cmdLine)
{
    SLANG_ASSERT(options.modulePath.count);

    // https://docs.microsoft.com/en-us/cpp/build/reference/compiler-options-listed-alphabetically?view=vs-2019

    cmdLine.addArg("/nologo");

    // Display full path of source files in diagnostics
    cmdLine.addArg("/FC");

    if (options.sourceLanguage == SLANG_SOURCE_LANGUAGE_CPP)
    {
        if (options.flags & CompileOptions::Flag::EnableExceptionHandling)
        {
            // https://docs.microsoft.com/en-us/cpp/build/reference/eh-exception-handling-model?view=vs-2019
            // Assumes c functions cannot throw
            cmdLine.addArg("/EHsc");
        }

        // To maintain parity with the slang compiler headers which are shared
        cmdLine.addArg("/std:c++17");
    }

    if (options.flags & CompileOptions::Flag::Verbose)
    {
        // Doesn't appear to be a VS equivalent
    }

    if (options.flags & CompileOptions::Flag::EnableSecurityChecks)
    {
        cmdLine.addArg("/GS");
    }
    else
    {
        cmdLine.addArg("/GS-");
    }

    switch (options.debugInfoType)
    {
    default:
        {
            // Multithreaded statically linked runtime library
            cmdLine.addArg("/MD");
            break;
        }
    case DebugInfoType::None:
        {
            break;
        }
    case DebugInfoType::Maximal:
        {
            // Multithreaded statically linked *debug* runtime library
            cmdLine.addArg("/MDd");
            break;
        }
    }

    // /Fd - followed by name of the pdb file
    if (options.debugInfoType != DebugInfoType::None)
    {
        // Generate complete debugging information
        cmdLine.addArg("/Zi");
        cmdLine.addPrefixPathArg("/Fd", asString(options.modulePath), ".pdb");
    }

    switch (options.optimizationLevel)
    {
    case OptimizationLevel::None:
        {
            // No optimization
            cmdLine.addArg("/Od");
            break;
        }
    case OptimizationLevel::Default:
        {
            break;
        }
    case OptimizationLevel::High:
        {
            cmdLine.addArg("/O2");
            break;
        }
    case OptimizationLevel::Maximal:
        {
            cmdLine.addArg("/Ox");
            break;
        }
    default:
        break;
    }

    switch (options.floatingPointMode)
    {
    case FloatingPointMode::Default:
        break;
    case FloatingPointMode::Precise:
        {
            // precise is default behavior, VS also has 'strict'
            //
            // ```/fp:strict has behavior similar to /fp:precise, that is, the compiler preserves
            // the source ordering and rounding properties of floating-point code when it generates
            // and optimizes object code for the target machine, and observes the standard when
            // handling special values. In addition, the program may safely access or modify the
            // floating-point environment at runtime.```

            cmdLine.addArg("/fp:precise");
            break;
        }
    case FloatingPointMode::Fast:
        {
            cmdLine.addArg("/fp:fast");
            break;
        }
    }

    const auto modulePath = asString(options.modulePath);

    switch (options.targetType)
    {
    case SLANG_SHADER_SHARED_LIBRARY:
    case SLANG_HOST_SHARED_LIBRARY:
        {
            // Create dynamic link library
            if (options.debugInfoType == DebugInfoType::None)
            {
                cmdLine.addArg("/LDd");
            }
            else
            {
                cmdLine.addArg("/LD");
            }

            cmdLine.addPrefixPathArg("/Fe", modulePath, ".dll");
            break;
        }
    case SLANG_HOST_EXECUTABLE:
        {
            cmdLine.addPrefixPathArg("/Fe", modulePath, ".exe");
            break;
        }
    default:
        break;
    }

    // Object file specify it's location - needed if we are out
    cmdLine.addPrefixPathArg("/Fo", modulePath, ".obj");

    // Add defines
    for (const auto& define : options.defines)
    {
        StringBuilder builder;
        builder << "/D";
        builder << asStringSlice(define.nameWithSig);
        if (define.value.count)
        {
            builder << "=" << asStringSlice(define.value);
        }

        cmdLine.addArg(builder);
    }

    // Add includes
    for (const auto& include : options.includePaths)
    {
        cmdLine.addArg("/I");
        cmdLine.addArg(asString(include));
    }

    // https://docs.microsoft.com/en-us/cpp/build/reference/eh-exception-handling-model?view=vs-2019
    // /Eha - Specifies the model of exception handling. (a, s, c, r are options)

    // Files to compile, need to be on the file system.
    for (IArtifact* sourceArtifact : options.sourceArtifacts)
    {
        ComPtr<IOSFileArtifactRepresentation> fileRep;

        // TODO(JS):
        // Do we want to keep the file on the file system? It's probably reasonable to do so.
        SLANG_RETURN_ON_FAIL(sourceArtifact->requireFile(ArtifactKeep::Yes, fileRep.writeRef()));
        cmdLine.addArg(fileRep->getPath());
    }

    // Link options (parameters past /link go to linker)
    cmdLine.addArg("/link");

    StringSlicePool libPathPool(StringSlicePool::Style::Default);

    for (const auto& libPath : options.libraryPaths)
    {
        libPathPool.add(libPath);
    }

    // Link libraries.
    for (IArtifact* artifact : options.libraries)
    {
        auto desc = artifact->getDesc();

        if (ArtifactDescUtil::isCpuBinary(desc) && desc.kind == ArtifactKind::Library)
        {
            // Get the libray name and path
            ComPtr<IOSFileArtifactRepresentation> fileRep;
            SLANG_RETURN_ON_FAIL(artifact->requireFile(ArtifactKeep::Yes, fileRep.writeRef()));

            const UnownedStringSlice path(fileRep->getPath());
            libPathPool.add(Path::getParentDirectory(path));
            // We need the extension for windows
            cmdLine.addArg(ArtifactDescUtil::getBaseNameFromPath(desc, path) + ".lib");
        }
    }

    // Add all the library paths
    for (const auto& libPath : libPathPool.getAdded())
    {
        // Note that any escaping of the path is handled in the ProcessUtil::
        cmdLine.addPrefixPathArg("/LIBPATH:", libPath);
    }

    // Add compiler specific options from user.
    for (auto compilerSpecificArg : options.compilerSpecificArguments)
    {
        const char* const arg = compilerSpecificArg;
        cmdLine.addArg(arg);
    }

    return SLANG_OK;
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
    else if (in == "info")
    {
        outSeverity = Severity::Info;
    }
    else
    {
        return SLANG_FAIL;
    }
    return SLANG_OK;
}

static SlangResult _parseVisualStudioLine(
    SliceAllocator& allocator,
    const UnownedStringSlice& line,
    ArtifactDiagnostic& outDiagnostic)
{
    typedef IArtifactDiagnostics::Diagnostic Diagnostic;

    UnownedStringSlice linkPrefix = UnownedStringSlice::fromLiteral("LINK :");
    if (line.startsWith(linkPrefix))
    {
        outDiagnostic.stage = ArtifactDiagnostic::Stage::Link;
        outDiagnostic.severity = ArtifactDiagnostic::Severity::Info;

        outDiagnostic.text = allocator.allocate(line.begin() + linkPrefix.getLength(), line.end());

        return SLANG_OK;
    }

    outDiagnostic.stage = ArtifactDiagnostic::Stage::Compile;

    const char* const start = line.begin();
    const char* const end = line.end();

    UnownedStringSlice postPath;
    // Handle the path and line no
    {
        const char* cur = start;

        // We have to assume it is a path up to the first : that isn't part of a drive specification

        if ((end - cur > 2) && Path::isDriveSpecification(UnownedStringSlice(start, start + 2)))
        {
            // Skip drive spec
            cur += 2;
        }

        // Find the first colon after this
        Index colonIndex = UnownedStringSlice(cur, end).indexOf(':');
        if (colonIndex < 0)
        {
            return SLANG_FAIL;
        }

        // Looks like we have a line number
        if (cur[colonIndex - 1] == ')')
        {
            const char* lineNoEnd = cur + colonIndex - 1;
            const char* lineNoStart = lineNoEnd;
            while (lineNoStart > start && *lineNoStart != '(')
            {
                lineNoStart--;
            }
            // Check this appears plausible
            if (*lineNoStart != '(' || *lineNoEnd != ')')
            {
                return SLANG_FAIL;
            }
            Int numDigits = 0;
            Int lineNo = 0;
            for (const char* digitCur = lineNoStart + 1; digitCur < lineNoEnd; ++digitCur)
            {
                char c = *digitCur;
                if (c >= '0' && c <= '9')
                {
                    lineNo = lineNo * 10 + (c - '0');
                    numDigits++;
                }
                else
                {
                    return SLANG_FAIL;
                }
            }
            if (numDigits == 0)
            {
                return SLANG_FAIL;
            }

            outDiagnostic.filePath = allocator.allocate(start, lineNoStart);
            outDiagnostic.location.line = lineNo;
        }
        else
        {
            outDiagnostic.filePath = allocator.allocate(start, cur + colonIndex);
            outDiagnostic.location.line = 0;
        }

        // Save the remaining text in 'postPath'
        postPath = UnownedStringSlice(cur + colonIndex + 1, end);
    }

    // Split up the error section
    UnownedStringSlice postError;
    {
        // tests/cpp-compiler/c-compile-link-error.exe : fatal error LNK1120: 1 unresolved externals

        const Index errorColonIndex = postPath.indexOf(':');
        if (errorColonIndex < 0)
        {
            return SLANG_FAIL;
        }

        const UnownedStringSlice errorSection =
            UnownedStringSlice(postPath.begin(), postPath.begin() + errorColonIndex);
        Index errorCodeIndex = errorSection.lastIndexOf(' ');
        if (errorCodeIndex < 0)
        {
            return SLANG_FAIL;
        }

        // Extract the code
        outDiagnostic.code =
            allocator.allocate(errorSection.begin() + errorCodeIndex + 1, errorSection.end());
        if (asStringSlice(outDiagnostic.code).startsWith(UnownedStringSlice::fromLiteral("LNK")))
        {
            outDiagnostic.stage = Diagnostic::Stage::Link;
        }

        // Extract the bit before the code
        SLANG_RETURN_ON_FAIL(_parseSeverity(
            UnownedStringSlice(errorSection.begin(), errorSection.begin() + errorCodeIndex).trim(),
            outDiagnostic.severity));

        // Link codes start with LNK prefix
        postError = UnownedStringSlice(postPath.begin() + errorColonIndex + 1, end);
    }

    outDiagnostic.text = allocator.allocate(postError);

    return SLANG_OK;
}

/* static */ SlangResult VisualStudioCompilerUtil::parseOutput(
    const ExecuteResult& exeRes,
    IArtifactDiagnostics* diagnostics)
{
    diagnostics->reset();

    diagnostics->setRaw(SliceUtil::asTerminatedCharSlice(exeRes.standardOutput));

    SliceAllocator allocator;

    for (auto line : LineParser(exeRes.standardOutput.getUnownedSlice()))
    {
#if 0
        fwrite(line.begin(), 1, line.size(), stdout);
        fprintf(stdout, "\n");
#endif

        ArtifactDiagnostic diagnostic;
        if (SLANG_SUCCEEDED(_parseVisualStudioLine(allocator, line, diagnostic)))
        {
            diagnostics->add(diagnostic);
        }
    }

    // if it has a compilation error.. set on output
    if (diagnostics->hasOfAtLeastSeverity(ArtifactDiagnostic::Severity::Error))
    {
        diagnostics->setResult(SLANG_FAIL);
    }

    return SLANG_OK;
}

/* static */ SlangResult VisualStudioCompilerUtil::locateCompilers(
    const String& path,
    ISlangSharedLibraryLoader* loader,
    [[maybe_unused]] DownstreamCompilerSet* set)
{
    SLANG_UNUSED(loader);

    // TODO(JS): We don't support fixed path for visual studio just yet
    if (path.getLength() == 0)
    {
#if SLANG_VC
        return WinVisualStudioUtil::find(set);
#endif
    }

    return SLANG_OK;
}

} // namespace Slang
