#pragma once

#include "slang/slang-diagnostics.h"

namespace CppParse
{
using namespace Slang;


// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Options !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

struct Options
{
    void reset() { *this = Options(); }

    Options()
    {
        m_markPrefix = "SLANG_";
        m_markSuffix = "_CLASS";
    }

    bool m_defs = false; ///< If set will output a '-defs.h' file for each of the input files, that
                         ///< corresponds to previous defs files (although doesn't have fields/RAW)
    bool m_dump =
        false; ///< If true will dump to stderr the types/fields and hierarchy it extracted
    bool m_runUnitTests = false; ///< If true will run internal unit tests
    bool m_extractDoc = true;    ///< If set will try to extract documentation associated with nodes

    bool m_outputFields = false; ///< When dumping macros also dump field definitions
    bool m_requireMark = true;

    List<String> m_inputPaths; ///< The input paths to the files to be processed

    String m_outputPath; ///< The output path. Note that the extractor can generate multiple output
                         ///< files, and this will actually be the 'stem' of several files

    String m_inputDirectory;  ///< The input directory that is by default used for reading
                              ///< m_inputPaths from.
    String m_markPrefix;      ///< The prefix of the 'marker' used to identify a reflected type
    String m_markSuffix;      ///< The postfix of the 'marker' used to identify a reflected type
    String m_stripFilePrefix; ///< Used for the 'origin' information, this is stripped from the
                              ///< source filename, and the remainder of the filename (without
                              ///< extension) is 'macroized'
};

struct OptionsParser
{
    /// Parse the parameters. NOTE! Must have the program path removed
    SlangResult parse(int argc, const char* const* argv, DiagnosticSink* sink, Options& outOptions);

    SlangResult _parseArgWithValue(const char* option, String& outValue);
    SlangResult _parseArgReplaceValue(const char* option, String& outValue);
    SlangResult _parseArgFlag(const char* option, bool& outFlag);

    String m_reflectType;

    Index m_index;
    Int m_argCount;
    const char* const* m_args;
    DiagnosticSink* m_sink;
};

} // namespace CppParse
