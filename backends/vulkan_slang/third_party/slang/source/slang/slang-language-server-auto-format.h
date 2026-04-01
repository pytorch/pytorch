#pragma once

#include "../core/slang-basic.h"
#include "slang-workspace-version.h"
#include "slang.h"

namespace Slang
{
struct Edit
{
    Index offset;
    Index length;
    String text;
};

struct TextRange
{
    Index offsetStart;
    Index offsetEnd;
};

enum class FormatBehavior
{
    Standard,
    PreserveLineBreak,
};

struct FormatOptions
{
    String clangFormatLocation;
    String style = "file";
    String fallbackStyle = "{BasedOnStyle: Microsoft}";
    String fileName;
    bool allowLineBreakInOnTypeFormatting = false;
    bool allowLineBreakInRangeFormatting = false;
    FormatBehavior behavior = FormatBehavior::Standard;
};

String findClangFormatTool();

List<TextRange> extractFormattingExclusionRanges(UnownedStringSlice text);

List<Edit> formatSource(
    UnownedStringSlice text,
    Index lineStart,
    Index lineEnd,
    Index cursorOffset,
    const List<TextRange>& exclusionRanges,
    const FormatOptions& options);

} // namespace Slang
