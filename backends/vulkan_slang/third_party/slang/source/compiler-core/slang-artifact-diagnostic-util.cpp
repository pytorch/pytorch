// slang-artifact-diagnostic-util.cpp
#include "slang-artifact-diagnostic-util.h"

#include "../core/slang-char-util.h"
#include "../core/slang-string-util.h"

namespace Slang
{

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ArtifactDiagnosticsUtil !!!!!!!!!!!!!!!!!!!!!!!!!!! */

/* static */ UnownedStringSlice ArtifactDiagnosticUtil::getSeverityText(Severity severity)
{
    switch (severity)
    {
    default:
        return UnownedStringSlice::fromLiteral("Unknown");
    case Severity::Info:
        return UnownedStringSlice::fromLiteral("Info");
    case Severity::Warning:
        return UnownedStringSlice::fromLiteral("Warning");
    case Severity::Error:
        return UnownedStringSlice::fromLiteral("Error");
    }
}

/* static */ SlangResult ArtifactDiagnosticUtil::splitPathLocation(
    SliceAllocator& allocator,
    const UnownedStringSlice& pathLocation,
    ArtifactDiagnostic& outDiagnostic)
{
    const Index lineStartIndex = pathLocation.lastIndexOf('(');
    if (lineStartIndex >= 0)
    {
        outDiagnostic.filePath = allocator.allocate(pathLocation.head(lineStartIndex).trim());

        const UnownedStringSlice tail = pathLocation.tail(lineStartIndex + 1);
        const Index lineEndIndex = tail.indexOf(')');

        if (lineEndIndex >= 0)
        {
            // Extract the location info
            UnownedStringSlice locationSlice(tail.begin(), tail.begin() + lineEndIndex);

            UnownedStringSlice slices[2];
            const Index numSlices = StringUtil::split(locationSlice, ',', 2, slices);

            // NOTE! FXC actually outputs a range of columns in the form of START-END in the column
            // position We don't need to parse here, because we only care about the line number

            Int lineNumber = 0;
            if (numSlices > 0)
            {
                SLANG_RETURN_ON_FAIL(StringUtil::parseInt(slices[0], lineNumber));
            }

            // Store the line
            outDiagnostic.location.line = lineNumber;
        }
    }
    else
    {
        outDiagnostic.filePath = allocator.allocate(pathLocation);
    }
    return SLANG_OK;
}

/* static */ SlangResult ArtifactDiagnosticUtil::splitColonDelimitedLine(
    const UnownedStringSlice& line,
    Int pathIndex,
    List<UnownedStringSlice>& outSlices)
{
    StringUtil::split(line, ':', outSlices);

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

/* static */ SlangResult ArtifactDiagnosticUtil::parseColonDelimitedDiagnostics(
    SliceAllocator& allocator,
    const UnownedStringSlice& inText,
    Int pathIndex,
    LineParser lineParser,
    IArtifactDiagnostics* diagnostics)
{
    List<UnownedStringSlice> splitLine;

    UnownedStringSlice text(inText), line;
    while (StringUtil::extractLine(text, line))
    {
        SLANG_RETURN_ON_FAIL(splitColonDelimitedLine(line, pathIndex, splitLine));

        ArtifactDiagnostic diagnostic;
        diagnostic.severity = Severity::Error;
        diagnostic.stage = ArtifactDiagnostic::Stage::Compile;
        diagnostic.location.line = 0;
        diagnostic.location.column = 0;

        if (SLANG_SUCCEEDED(lineParser(allocator, line, splitLine, diagnostic)))
        {
            diagnostics->add(diagnostic);
        }
        else
        {
            // If couldn't parse, just add as a note
            maybeAddNote(line, diagnostics);
        }
    }

    return SLANG_OK;
}

/* static */ void ArtifactDiagnosticUtil::maybeAddNote(
    const UnownedStringSlice& in,
    IArtifactDiagnostics* diagnostics)
{
    // Don't bother adding an empty line
    if (in.trim().getLength() == 0)
    {
        return;
    }

    // If there's nothing previous, we'll ignore too, as note should be in addition to
    // a pre-existing error/warning
    if (diagnostics->getCount() == 0)
    {
        return;
    }

    // Make it a note on the output
    ArtifactDiagnostic diagnostic;

    String text(in);

    diagnostic.severity = ArtifactDiagnostic::Severity::Info;
    diagnostic.text = SliceUtil::asTerminatedCharSlice(text);
    diagnostics->add(diagnostic);
}


} // namespace Slang
