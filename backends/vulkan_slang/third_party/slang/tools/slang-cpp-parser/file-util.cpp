#include "file-util.h"

#include "core/slang-io.h"

namespace CppParse
{
using namespace Slang;

namespace
{ // anonymous
struct DiagnosticReporter
{
    SlangResult report(SlangResult res)
    {
        if (SLANG_FAILED(res))
        {
            if (m_sink)
            {
                if (res == SLANG_E_CANNOT_OPEN)
                {
                    m_sink->diagnose(SourceLoc(), CPPDiagnostics::cannotOpenFile, m_filename);
                }
                else
                {
                    m_sink->diagnose(SourceLoc(), CPPDiagnostics::errorAccessingFile, m_filename);
                }
            }
        }
        return res;
    }

    DiagnosticReporter(const String& filename, DiagnosticSink* sink)
        : m_filename(filename), m_sink(sink)
    {
    }

    DiagnosticSink* m_sink;
    String m_filename;
};

} // namespace

/* static */ SlangResult FileUtil::readAllText(
    const Slang::String& fileName,
    DiagnosticSink* sink,
    String& outRead)
{
    DiagnosticReporter reporter(fileName, sink);

    RefPtr<FileStream> stream = new FileStream;
    SLANG_RETURN_ON_FAIL(reporter.report(
        stream->init(fileName, FileMode::Open, FileAccess::Read, FileShare::ReadWrite)));

    StreamReader reader;
    SLANG_RETURN_ON_FAIL(reporter.report(reader.init(stream)));
    SLANG_RETURN_ON_FAIL(reporter.report(reader.readToEnd(outRead)));

    return SLANG_OK;
}

/* static */ SlangResult FileUtil::writeAllText(
    const Slang::String& fileName,
    DiagnosticSink* sink,
    const UnownedStringSlice& text)
{
    // TODO(JS):
    // There is an optimization/behavior here,that checks if the contents has changed. It only
    // writes if it hasn't That might not be what you want (both because of extra work of read, the
    // file modified stamp or other reasons, file is write only etc) NOTE! That this also does the
    // work of the comparison after it is decoded, but the *bytes* might actually be different.

    if (File::exists(fileName))
    {
        String existingText;
        if (SLANG_SUCCEEDED(readAllText(fileName, nullptr, existingText)))
        {
            if (existingText == text)
                return SLANG_OK;
        }
    }

    DiagnosticReporter reporter(fileName, sink);

    RefPtr<FileStream> stream = new FileStream;
    SLANG_RETURN_ON_FAIL(reporter.report(stream->init(fileName, FileMode::Create)));

    StreamWriter writer;
    SLANG_RETURN_ON_FAIL(reporter.report(writer.init(stream)));
    SLANG_RETURN_ON_FAIL(reporter.report(writer.write(text)))
    return SLANG_OK;
}

/* static */ void FileUtil::indent(Index indentCount, StringBuilder& out)
{
    for (Index i = 0; i < indentCount; ++i)
    {
        out << CPP_EXTRACT_INDENT_STRING;
    }
}

} // namespace CppParse
