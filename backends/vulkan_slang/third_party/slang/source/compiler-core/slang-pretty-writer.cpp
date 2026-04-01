#include "slang-pretty-writer.h"

#include "../core/slang-string-escape-util.h"

namespace Slang
{

void PrettyWriter::writeRaw(char const* begin, char const* end)
{
    SLANG_ASSERT(end >= begin);
    writeRaw(UnownedStringSlice(begin, end));
}

void PrettyWriter::adjust()
{
    // Only indent if at start of a line
    if (m_startOfLine)
    {
        // Output current indentation
        m_builder.appendRepeatedChar(' ', m_indent * 4);
        m_startOfLine = false;
    }
}

void PrettyWriter::dedent()
{
    SLANG_ASSERT(m_indent > 0);
    m_indent--;
}

void PrettyWriter::write(const UnownedStringSlice& slice)
{
    const auto end = slice.end();
    auto start = slice.begin();

    while (start < end)
    {
        const char* cur = start;

        // Search for \n if there is one
        while (cur < end && *cur != '\n')
            cur++;

        // If there were some chars, adjust and write
        if (cur > start)
        {
            adjust();
            writeRaw(UnownedStringSlice(start, cur));
        }

        if (cur < end && *cur == '\n')
        {
            writeRawChar('\n');
            // Skip the CR
            cur++;
            // Mark we are at the start of a line
            m_startOfLine = true;
        }

        start = cur;
    }
}

void PrettyWriter::writeEscapedString(const UnownedStringSlice& slice)
{
    adjust();
    auto handler = StringEscapeUtil::getHandler(StringEscapeUtil::Style::Cpp);
    StringEscapeUtil::appendQuoted(handler, slice, m_builder);
}

void PrettyWriter::maybeComma()
{
    if (auto state = m_commaState)
    {
        if (!state->needComma)
        {
            state->needComma = true;
            return;
        }
    }

    write(toSlice(",\n"));
}

} // namespace Slang
