// parse-diagnostic-util.cpp

#include "parse-diagnostic-util.h"

#include "../../source/compiler-core/slang-artifact-associated-impl.h"
#include "../../source/compiler-core/slang-artifact-diagnostic-util.h"
#include "../../source/compiler-core/slang-downstream-compiler.h"
#include "../../source/core/slang-byte-encode-util.h"
#include "../../source/core/slang-char-util.h"
#include "../../source/core/slang-hex-dump-util.h"
#include "../../source/core/slang-string-util.h"
#include "../../source/core/slang-type-text-util.h"
#include "slang-com-helper.h"

using namespace Slang;

/* static */ SlangResult ParseDiagnosticUtil::parseGenericLine(
    SliceAllocator& allocator,
    const UnownedStringSlice& line,
    List<UnownedStringSlice>& lineSlices,
    ArtifactDiagnostic& outDiagnostic)
{
    /* e:\git\somewhere\tests\diagnostics\syntax-error-intrinsic.slang(13): error C2018:  unknown
     * character '0x40' */
    if (lineSlices.getCount() < 3)
    {
        return SLANG_FAIL;
    }

    {
        const UnownedStringSlice severityAndCodeSlice = lineSlices[1].trim();
        // Get the code
        outDiagnostic.code =
            allocator.allocate(StringUtil::getAtInSplit(severityAndCodeSlice, ' ', 1).trim());

        const UnownedStringSlice severitySlice =
            StringUtil::getAtInSplit(severityAndCodeSlice, ' ', 0);

        outDiagnostic.severity = ArtifactDiagnostic::Severity::Error;
        if (severitySlice == UnownedStringSlice::fromLiteral("warning"))
        {
            outDiagnostic.severity = ArtifactDiagnostic::Severity::Warning;
        }
        else if (severitySlice == UnownedStringSlice::fromLiteral("info"))
        {
            outDiagnostic.severity = ArtifactDiagnostic::Severity::Info;
        }
    }

    // Get the location info
    SLANG_RETURN_ON_FAIL(
        ArtifactDiagnosticUtil::splitPathLocation(allocator, lineSlices[0], outDiagnostic));

    outDiagnostic.text = allocator.allocate(lineSlices[2].begin(), line.end());
    return SLANG_OK;
}

static SlangResult _getSlangDiagnosticSeverity(
    const UnownedStringSlice& inText,
    ArtifactDiagnostic::Severity& outSeverity,
    Int& outCode)
{
    UnownedStringSlice text(inText.trim());

    static const UnownedStringSlice prefixes[] = {
        UnownedStringSlice::fromLiteral("note"),
        UnownedStringSlice::fromLiteral("warning"),
        UnownedStringSlice::fromLiteral("error"),
        UnownedStringSlice::fromLiteral("fatal error"),
        UnownedStringSlice::fromLiteral("internal error"),
        UnownedStringSlice::fromLiteral("unknown error")};

    Int index = -1;

    for (Index i = 0; i < SLANG_COUNT_OF(prefixes); ++i)
    {
        const auto& prefix = prefixes[i];
        if (text.startsWith(prefix))
        {
            index = i;
            break;
        }
    }

    switch (index)
    {
    case -1:
        return SLANG_FAIL;
    case 0:
        outSeverity = ArtifactDiagnostic::Severity::Info;
        break;
    case 1:
        outSeverity = ArtifactDiagnostic::Severity::Warning;
        break;
    default:
        outSeverity = ArtifactDiagnostic::Severity::Error;
        break;
    }

    outCode = 0;

    UnownedStringSlice tail = text.tail(prefixes[index].getLength()).trim();
    if (tail.getLength() > 0)
    {
        SLANG_RETURN_ON_FAIL(StringUtil::parseInt(tail, outCode));
    }

    return SLANG_OK;
}

static bool _isSlangDiagnostic(const UnownedStringSlice& line)
{
    /*
    tests/diagnostics/accessors.slang(11): error 31101: accessors other than 'set' must not have
    parameters
    */

    UnownedStringSlice initial = StringUtil::getAtInSplit(line, ':', 0);

    // Handle if path has :
    const Index typeIndex = (initial.getLength() == 1 && CharUtil::isAlpha(initial[0])) ? 2 : 1;
    // Extract the type/code slice
    UnownedStringSlice typeSlice = StringUtil::getAtInSplit(line, ':', typeIndex);

    ArtifactDiagnostic::Severity type;
    Int code;
    return SLANG_SUCCEEDED(_getSlangDiagnosticSeverity(typeSlice, type, code));
}

/* static */ SlangResult ParseDiagnosticUtil::parseSlangLine(
    SliceAllocator& allocator,
    const UnownedStringSlice& line,
    List<UnownedStringSlice>& lineSlices,
    ArtifactDiagnostic& outDiagnostic)
{
    /*
    tests/diagnostics/accessors.slang(11): error 31101: accessors other than 'set' must not have
    parameters
    */

    // Can be larger than 3, because might be : in the actual error text
    if (lineSlices.getCount() < 3)
    {
        return SLANG_FAIL;
    }

    SLANG_RETURN_ON_FAIL(
        ArtifactDiagnosticUtil::splitPathLocation(allocator, lineSlices[0], outDiagnostic));
    Int code;
    SLANG_RETURN_ON_FAIL(_getSlangDiagnosticSeverity(lineSlices[1], outDiagnostic.severity, code));

    if (code != 0)
    {
        StringBuilder buf;
        buf << code;
        outDiagnostic.code = allocator.allocate(buf);
    }

    outDiagnostic.text = allocator.allocate(lineSlices[2].begin(), line.end());
    return SLANG_OK;
}

/* static */ SlangResult ParseDiagnosticUtil::splitDiagnosticLine(
    const CompilerIdentity& compilerIdentity,
    const UnownedStringSlice& line,
    const UnownedStringSlice& linePrefix,
    List<UnownedStringSlice>& outSlices)
{
    StringUtil::split(line, ':', outSlices);

    // If we have a prefix (typically identifying the compiler), remove so same code can be used for
    // output with prefixes and without
    if (linePrefix.getLength())
    {
        SLANG_ASSERT(outSlices[0].startsWith(linePrefix));
        outSlices.removeAt(0);
    }

    /*
    glslang: ERROR: tests/diagnostics/syntax-error-intrinsic.slang:13: '@' : unexpected token
    dxc: tests/diagnostics/syntax-error-intrinsic.slang:14:2: error: expected expression
    fxc: tests/diagnostics/syntax-error-intrinsic.slang(14,2): error X3000: syntax error: unexpected
    token '@' Visual Studio 14.0:
    e:\git\somewhere\tests\diagnostics\syntax-error-intrinsic.slang(13): error C2018:  unknown
    character '0x40' NVRTC 11.0: tests/diagnostics/syntax-error-intrinsic.slang(13): error :
    unrecognized token tests/diagnostics/accessors.slang(11): error 31101: accessors other than
    'set' must not have parameters
    */

    // The index where the path starts
    const Int pathIndex = 0;

    // Now we want to fix up a path as might have drive letter, and therefore :
    // If this is the situation then we need to have a slice after the one at the index
    if (outSlices.getCount() > pathIndex + 1)
    {
        const UnownedStringSlice pathStart = outSlices[pathIndex].trim();
        if (pathStart.getLength() == 1 && CharUtil::isAlpha(pathStart[0]))
        {
            // Splice back together
            outSlices[pathIndex] =
                UnownedStringSlice(outSlices[pathIndex].begin(), outSlices[pathIndex + 1].end());
            outSlices.removeAt(pathIndex + 1);
        }
    }

    return SLANG_OK;
}

static SlangResult _findDownstreamCompiler(
    const UnownedStringSlice& slice,
    SlangPassThrough& outDownstreamCompiler)
{
    for (Index i = SLANG_PASS_THROUGH_NONE + 1; i < SLANG_PASS_THROUGH_COUNT_OF; ++i)
    {
        const SlangPassThrough downstreamCompiler = SlangPassThrough(i);
        UnownedStringSlice name = TypeTextUtil::getPassThroughAsHumanText(downstreamCompiler);

        if (slice.startsWith(name))
        {
            outDownstreamCompiler = downstreamCompiler;
            return SLANG_OK;
        }
    }
    return SLANG_FAIL;
}

/* static */ SlangResult ParseDiagnosticUtil::identifyCompiler(
    const UnownedStringSlice& inText,
    CompilerIdentity& outIdentity)
{
    outIdentity = CompilerIdentity();

    // This might be overkill - we should be able to identify the compiler from the first line, of
    // the diagnostics. Here, we go through each line trying to identify the compiler. For
    // downstream compilers, the only way to identify unambiguously is via the compiler name prefix.
    // For Slang we *assume* if there isn't such a prefix, and it 'looks like' a Slang diagnostic
    // that it is

    UnownedStringSlice text(inText), line;
    while (StringUtil::extractLine(text, line))
    {
        UnownedStringSlice initial = StringUtil::getAtInSplit(line, ':', 0);

        if (_isSlangDiagnostic(line))
        {
            outIdentity = CompilerIdentity::makeSlang();
            return SLANG_OK;
        }
        else
        {
            SlangPassThrough downstreamCompiler;
            // First entry that begins with a numeral indicates the version number
            if (SLANG_SUCCEEDED(_findDownstreamCompiler(initial, downstreamCompiler)))
            {
                outIdentity = CompilerIdentity::make(downstreamCompiler);
                return SLANG_OK;
            }
        }
    }

    return SLANG_FAIL;
}

/* static */ ParseDiagnosticUtil::LineParser ParseDiagnosticUtil::getLineParser(
    const CompilerIdentity& compilerIdentity)
{
    switch (compilerIdentity.m_type)
    {
    case CompilerIdentity::Slang:
        return &parseSlangLine;
    case CompilerIdentity::DownstreamCompiler:
        return &parseGenericLine;
    default:
        return nullptr;
    }
}

static bool _isWhitespace(const UnownedStringSlice& slice)
{
    for (const char c : slice)
    {
        if (!CharUtil::isWhitespace(c))
        {
            return false;
        }
    }
    return true;
}

/* static */ SlangResult ParseDiagnosticUtil::parseDiagnostics(
    const UnownedStringSlice& inText,
    IArtifactDiagnostics* diagnostics)
{
    if (_isWhitespace(inText))
    {
        // If it's empty, then there are no diagnostics to add.
        return SLANG_OK;
    }

    CompilerIdentity compilerIdentity;
    SLANG_RETURN_ON_FAIL(ParseDiagnosticUtil::identifyCompiler(inText, compilerIdentity));

    UnownedStringSlice linePrefix;
    if (compilerIdentity.m_type == CompilerIdentity::Type::DownstreamCompiler)
    {
        linePrefix = TypeTextUtil::getPassThroughAsHumanText(compilerIdentity.m_downstreamCompiler);
    }
    else
    {
        // For Slang there isn't *currently* a prefix ever used, but that might change in the future
        // For now we assume no prefix.
    }

    return parseDiagnostics(inText, compilerIdentity, linePrefix, diagnostics);
}

/* static */ SlangResult ParseDiagnosticUtil::parseDiagnostics(
    const UnownedStringSlice& inText,
    const CompilerIdentity& compilerIdentity,
    const UnownedStringSlice& linePrefix,
    IArtifactDiagnostics* diagnostics)
{
    auto lineParser = getLineParser(compilerIdentity);
    if (!lineParser)
    {
        return SLANG_FAIL;
    }

    List<UnownedStringSlice> splitLine;

    SliceAllocator allocator;

    UnownedStringSlice text(inText), line;
    while (StringUtil::extractLine(text, line))
    {
        bool isValidSplit = false;
        // And the first entry must contain the prefix, else assume it's a note
        if (linePrefix.getLength() > 0 && line.startsWith(linePrefix))
        {
            // Try with the line prefix
            isValidSplit =
                SLANG_SUCCEEDED(splitDiagnosticLine(compilerIdentity, line, linePrefix, splitLine));
        }

        if (!isValidSplit)
        {
            // Try without the prefix, as some output output's only some lines with the prefix (GLSL
            // for example)
            isValidSplit = SLANG_SUCCEEDED(
                splitDiagnosticLine(compilerIdentity, line, UnownedStringSlice(), splitLine));
        }

        // If we don't have a valid split then just assume it's a note
        if (!isValidSplit)
        {
            diagnostics->maybeAddNote(asCharSlice(line));
            continue;
        }

        ArtifactDiagnostic diagnostic;
        diagnostic.severity = ArtifactDiagnostic::Severity::Error;
        diagnostic.stage = ArtifactDiagnostic::Stage::Compile;
        diagnostic.location.line = 0;

        if (SLANG_SUCCEEDED(lineParser(allocator, line, splitLine, diagnostic)))
        {
            diagnostics->add(diagnostic);
        }
        else
        {
            // If couldn't parse, just add as a note
            ArtifactDiagnosticUtil::maybeAddNote(line, diagnostics);
        }
    }

    return SLANG_OK;
}

static UnownedStringSlice _getEquals(const UnownedStringSlice& in)
{
    Index equalsIndex = in.indexOf('=');
    if (equalsIndex < 0)
    {
        return UnownedStringSlice();
    }
    return in.tail(equalsIndex + 1).trim();
}

static bool _isAtEnd(const UnownedStringSlice& text, const UnownedStringSlice& line)
{
    if (line != "}")
    {
        return false;
    }
    // We need to get the *next* line. If it is "}" then this isn't the final closing
    UnownedStringSlice remaining(text);
    UnownedStringSlice nextLine;
    StringUtil::extractLine(remaining, nextLine);

    return (nextLine != toSlice("}"));
}

/* static */ SlangResult ParseDiagnosticUtil::parseOutputInfo(
    const UnownedStringSlice& inText,
    OutputInfo& out)
{
    enum State
    {
        Normal,
        InStdError,
        InStdOut,
    };

    UnownedStringSlice resultCodePrefix = UnownedStringSlice::fromLiteral("result code");
    UnownedStringSlice stdErrorPrefix = UnownedStringSlice::fromLiteral("standard error");
    UnownedStringSlice stdOutputPrefix = UnownedStringSlice::fromLiteral("standard output");


    List<UnownedStringSlice> lines;

    State state = State::Normal;

    UnownedStringSlice text(inText), line;
    while (StringUtil::extractLine(text, line))
    {
        switch (state)
        {
        case State::Normal:
            {
                if (line.startsWith(resultCodePrefix))
                {
                    // Split past the equal
                    const UnownedStringSlice valueSlice =
                        _getEquals(line.tail(resultCodePrefix.getLength()));
                    Int value;
                    SLANG_RETURN_ON_FAIL(StringUtil::parseInt(valueSlice, value));
                    out.resultCode = int(value);
                }
                else
                {
                    UnownedStringSlice* startsWith = nullptr;
                    if (line.startsWith(stdErrorPrefix))
                    {
                        startsWith = &stdErrorPrefix;
                    }
                    else if (line.startsWith(stdOutputPrefix))
                    {
                        startsWith = &stdOutputPrefix;
                    }

                    if (startsWith)
                    {
                        // Clear the lines buffer
                        lines.clear();

                        UnownedStringSlice valueSlice =
                            _getEquals(line.tail(startsWith->getLength()));
                        if (!valueSlice.isChar('{'))
                        {
                            return SLANG_FAIL;
                        }
                        // Okay we now inside std out or std error, so update the state
                        state =
                            (startsWith == &stdErrorPrefix) ? State::InStdError : State::InStdOut;
                    }
                }
                break;
            }
        case State::InStdError:
        case State::InStdOut:
            {
                if (_isAtEnd(text, line))
                {
                    String& dst = (state == State::InStdError) ? out.stdError : out.stdOut;
                    if (lines.getCount() > 0)
                    {
                        dst = UnownedStringSlice(lines[0].begin(), lines.getLast().end());
                    }
                    state = State::Normal;
                }
                else
                {
                    lines.add(line);
                }
            }
        }
    }

    return (state == State::Normal) ? SLANG_OK : SLANG_FAIL;
}


/* static */ bool ParseDiagnosticUtil::areEqual(
    const UnownedStringSlice& a,
    const UnownedStringSlice& b,
    EqualityFlags flags)
{
    auto diagsA = ArtifactDiagnostics::create();
    auto diagsB = ArtifactDiagnostics::create();

    SlangResult resA = ParseDiagnosticUtil::parseDiagnostics(a, diagsA);
    SlangResult resB = ParseDiagnosticUtil::parseDiagnostics(b, diagsB);

    /*
        TODO(JS): In the past we needed special handling of the core module, when
        in some builds the path contains the core module.

        For now we don't seem to need this, this is for future reference, if there
        is an issue with needing to specially handle this.

       static const UnownedStringSlice coreModuleNames[] =
        {
            UnownedStringSlice::fromLiteral("core.meta.slang"),
            UnownedStringSlice::fromLiteral("hlsl.meta.slang"),
            UnownedStringSlice::fromLiteral("slang-core-module.cpp"),
        };
        */

    // Must have both succeeded, and have the same amount of lines
    if (SLANG_SUCCEEDED(resA) && SLANG_SUCCEEDED(resB) && diagsA->getCount() == diagsB->getCount())
    {
        const auto count = diagsA->getCount();
        for (Index i = 0; i < count; ++i)
        {
            ArtifactDiagnostic diagA = *diagsA->getAt(i);
            ArtifactDiagnostic diagB = *diagsB->getAt(i);

            // Check if we need to ignore line numbers
            if (flags & EqualityFlag::IgnoreLineNos)
            {
                const ArtifactDiagnostic::Location loc;

                diagA.location = loc;
                diagB.location = loc;
            }

            if (diagA != diagB)
            {
                return false;
            }
        }

        return true;
    }

    return false;
}
