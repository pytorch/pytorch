#ifndef SLANG_CORE_PRETTY_WRITER_H
#define SLANG_CORE_PRETTY_WRITER_H


#include "../core/slang-char-util.h"
#include "../core/slang-string-util.h"
#include "../core/slang-string.h"

namespace Slang
{

struct PrettyWriter
{
    typedef PrettyWriter ThisType;

    friend struct CommaTrackerRAII;

    struct CommaState
    {
        bool needComma = false;
    };

    void writeRaw(const UnownedStringSlice& slice) { m_builder.append(slice); }
    void writeRaw(char const* begin, char const* end);
    void writeRaw(char const* begin) { writeRaw(UnownedStringSlice(begin)); }

    void writeRawChar(int c) { m_builder.appendChar(char(c)); }

    void writeHexChar(int c) { writeRawChar(CharUtil::getHexChar(Index(c))); }

    /// Adjusts indentation if at start of a line
    void adjust();

    /// Increase indentation
    void indent() { m_indent++; }
    /// Decreate indentation
    void dedent();

    /// Write taking into account any CR that might be in a slice
    void write(const UnownedStringSlice& slice);
    void write(char const* text) { write(UnownedStringSlice(text)); }
    void write(char const* text, size_t length) { write(UnownedStringSlice(text, length)); }

    /// Write the slice as an escaped string
    void writeEscapedString(const UnownedStringSlice& slice);

    /// Call before items in a comma-separated JSON list to emit the comma if/when needed
    void maybeComma();

    /// Get the builder the result is being constructed in
    StringBuilder& getBuilder() { return m_builder; }

    ThisType& operator<<(const UnownedStringSlice& slice)
    {
        write(slice);
        return *this;
    }
    ThisType& operator<<(const char* text)
    {
        write(text);
        return *this;
    }
    ThisType& operator<<(uint64_t val)
    {
        adjust();
        m_builder << val;
        return *this;
    }
    ThisType& operator<<(int64_t val)
    {
        adjust();
        m_builder << val;
        return *this;
    }
    ThisType& operator<<(int32_t val)
    {
        adjust();
        m_builder << val;
        return *this;
    }
    ThisType& operator<<(uint32_t val)
    {
        adjust();
        m_builder << val;
        return *this;
    }
    ThisType& operator<<(float val)
    {
        adjust();
        // We want to use a specific format, so we use the StringUtil to specify format, and not
        // just use <<
        StringUtil::appendFormat(m_builder, "%f", val);
        return *this;
    }

    bool m_startOfLine = true;
    int m_indent = 0;
    CommaState* m_commaState = nullptr;
    StringBuilder m_builder;
};

/// Type for tracking whether a comma is needed in a comma-separated JSON list
struct CommaTrackerRAII
{
    CommaTrackerRAII(PrettyWriter& writer)
        : m_writer(&writer), m_previousState(writer.m_commaState)
    {
        writer.m_commaState = &m_state;
    }

    ~CommaTrackerRAII() { m_writer->m_commaState = m_previousState; }

private:
    PrettyWriter::CommaState m_state;
    PrettyWriter* m_writer;
    PrettyWriter::CommaState* m_previousState;
};

} // namespace Slang


#endif
