#pragma once

#include "diagnostics.h"

namespace CppParse
{
using namespace Slang;

// A macro to define a single indent as a string
#define CPP_EXTRACT_INDENT_STRING "    "

struct FileUtil
{
    /// Read text into outRead. Any failures written to sink (can be passed as nullptr, for no
    /// output)
    static SlangResult readAllText(
        const Slang::String& fileName,
        DiagnosticSink* sink,
        String& outRead);
    /// Write text to filename. Any failures written to sink. (can be passed as nullptr, for no
    /// output)
    static SlangResult writeAllText(
        const Slang::String& fileName,
        DiagnosticSink* sink,
        const UnownedStringSlice& text);

    /// Appends CPP_EXTRACT_INDENT_STRING indentCount number of times to out
    static void indent(Index indentCount, StringBuilder& out);
};

} // namespace CppParse
