#include "options.h"

#include "diagnostics.h"

namespace CppParse
{

SlangResult OptionsParser::_parseArgFlag(const char* option, bool& outFlag)
{
    SLANG_ASSERT(UnownedStringSlice(m_args[m_index]) == option);
    SLANG_ASSERT(m_index < m_argCount);

    m_index++;
    outFlag = true;
    return SLANG_OK;
}

SlangResult OptionsParser::_parseArgWithValue(const char* option, String& ioValue)
{
    SLANG_ASSERT(UnownedStringSlice(m_args[m_index]) == option);
    if (m_index + 1 < m_argCount)
    {
        // Next parameter is the output path, there can only be one
        if (ioValue.getLength())
        {
            // There already is output
            m_sink->diagnose(SourceLoc(), CPPDiagnostics::optionAlreadyDefined, option, ioValue);
            return SLANG_FAIL;
        }
    }
    else
    {
        m_sink->diagnose(SourceLoc(), CPPDiagnostics::requireValueAfterOption, option);
        return SLANG_FAIL;
    }

    ioValue = m_args[m_index + 1];
    m_index += 2;
    return SLANG_OK;
}

SlangResult OptionsParser::_parseArgReplaceValue(const char* option, String& ioValue)
{
    SLANG_ASSERT(UnownedStringSlice(m_args[m_index]) == option);
    if (m_index + 1 >= m_argCount)
    {
        m_sink->diagnose(SourceLoc(), CPPDiagnostics::requireValueAfterOption, option);
        return SLANG_FAIL;
    }

    ioValue = m_args[m_index + 1];
    m_index += 2;
    return SLANG_OK;
}

SlangResult OptionsParser::parse(
    int argc,
    const char* const* argv,
    DiagnosticSink* sink,
    Options& outOptions)
{
    outOptions.reset();

    m_index = 0;
    m_argCount = argc;
    m_args = argv;
    m_sink = sink;

    outOptions.reset();

    while (m_index < m_argCount)
    {
        const UnownedStringSlice arg = UnownedStringSlice(argv[m_index]);

        if (arg.getLength() > 0 && arg[0] == '-')
        {
            if (arg == "-d")
            {
                SLANG_RETURN_ON_FAIL(_parseArgWithValue("-d", outOptions.m_inputDirectory));
                continue;
            }
            else if (arg == "-o")
            {
                SLANG_RETURN_ON_FAIL(_parseArgWithValue("-o", outOptions.m_outputPath));
                continue;
            }
            else if (arg == "-dump")
            {
                SLANG_RETURN_ON_FAIL(_parseArgFlag("-dump", outOptions.m_dump));
                continue;
            }
            else if (arg == "-mark-prefix")
            {
                SLANG_RETURN_ON_FAIL(
                    _parseArgReplaceValue("-mark-prefix", outOptions.m_markPrefix));
                continue;
            }
            else if (arg == "-mark-suffix")
            {
                SLANG_RETURN_ON_FAIL(
                    _parseArgReplaceValue("-mark-suffix", outOptions.m_markSuffix));
                continue;
            }
            else if (arg == "-defs")
            {
                SLANG_RETURN_ON_FAIL(_parseArgFlag("-defs", outOptions.m_defs));
                continue;
            }
            else if (arg == "-output-fields")
            {
                SLANG_RETURN_ON_FAIL(_parseArgFlag("-output-fields", outOptions.m_outputFields));
                continue;
            }
            else if (arg == "-strip-prefix")
            {
                SLANG_RETURN_ON_FAIL(
                    _parseArgWithValue("-strip-prefix", outOptions.m_stripFilePrefix));
                continue;
            }
            else if (arg == "-unit-test")
            {
                SLANG_RETURN_ON_FAIL(_parseArgFlag("-unit-test", outOptions.m_runUnitTests));
                continue;
            }
            else if (arg == "-unmarked")
            {
                bool unmarked;
                SLANG_RETURN_ON_FAIL(_parseArgFlag("-unmarked", unmarked));
                outOptions.m_requireMark = !unmarked;
                continue;
            }

            m_sink->diagnose(SourceLoc(), CPPDiagnostics::unknownOption, arg);
            return SLANG_FAIL;
        }
        else
        {
            // If it doesn't start with - then it's assumed to be an input path
            outOptions.m_inputPaths.add(arg);
            m_index++;
        }
    }

    if (outOptions.m_inputPaths.getCount() < 0)
    {
        m_sink->diagnose(SourceLoc(), CPPDiagnostics::noInputPathsSpecified);
        return SLANG_FAIL;
    }

    return SLANG_OK;
}

} // namespace CppParse
