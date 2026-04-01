#ifndef SLANG_CORE_TEXT_IO_H
#define SLANG_CORE_TEXT_IO_H

#include "slang-char-encode.h"
#include "slang-secure-crt.h"
#include "slang-stream.h"

namespace Slang
{
using Slang::List;
using Slang::_EndLine;

class TextReader
{
public:
    virtual void close() {}
    virtual SlangResult readToEnd(String& outString) = 0;
    virtual bool isEnd() = 0;

    char read()
    {
        if (m_decodedCharIndex == m_decodedCharSize)
            readChar();
        if (m_decodedCharIndex < m_decodedCharSize)
            return m_decodedChar[m_decodedCharIndex++];
        else
            return 0;
    }
    char peek()
    {
        if (m_decodedCharIndex == m_decodedCharSize)
            readChar();
        if (m_decodedCharIndex < m_decodedCharSize)
            return m_decodedChar[m_decodedCharIndex];
        else
            return 0;
    }

    virtual ~TextReader() { close(); }

protected:
    char m_decodedChar[5];
    Index m_decodedCharIndex = 0;
    Index m_decodedCharSize = 0;

    virtual void readChar() = 0;
};


class StreamReader : public TextReader
{
public:
    virtual SlangResult readToEnd(String& outString) SLANG_OVERRIDE;
    virtual bool isEnd() SLANG_OVERRIDE
    {
        return m_index == m_buffer.getCount() && m_stream->isEnd();
    }
    virtual void close() SLANG_OVERRIDE { m_stream->close(); }

    void releaseStream() { m_stream.setNull(); }

    StreamReader();

    SlangResult init(const String& path);
    SlangResult init(RefPtr<Stream> stream, CharEncoding* encoding = nullptr);

protected:
    virtual void readChar() SLANG_OVERRIDE
    {
        m_decodedCharIndex = 0;

        Char32 codePoint = 0;
        switch (m_encodingType)
        {
        case CharEncodeType::UTF8:
            {
                codePoint = getUnicodePointFromUTF8([&]() -> Byte { return readBufferChar(); });
                break;
            }
        case CharEncodeType::UTF16:
            {
                codePoint = getUnicodePointFromUTF16([&]() -> Byte { return readBufferChar(); });
                break;
            }
        case CharEncodeType::UTF16Reversed:
            {
                codePoint =
                    getUnicodePointFromUTF16Reversed([&]() -> Byte { return readBufferChar(); });
                break;
            }
        case CharEncodeType::UTF32:
            {
                codePoint = getUnicodePointFromUTF32([&]() -> Byte { return readBufferChar(); });
                break;
            }
        }

        m_decodedCharSize = encodeUnicodePointToUTF8(codePoint, m_decodedChar);
    }

private:
    char readBufferChar();
    SlangResult readBuffer();

    RefPtr<Stream> m_stream;
    List<char> m_buffer;

    CharEncodeType m_encodingType = CharEncodeType::UTF8;
    CharEncoding* m_encoding = nullptr;
    Index m_index = 0; ///< Index into buffer
};

class TextWriter
{
public:
    virtual SlangResult writeSlice(const UnownedStringSlice& slice) = 0;
    virtual void close() {}

    SlangResult write(const UnownedStringSlice& slice) { return writeSlice(slice); }
    SlangResult write(const char* str) { return writeSlice(UnownedStringSlice(str)); }
    SlangResult write(const String& str) { return writeSlice(str.getUnownedSlice()); }

    virtual ~TextWriter() { close(); }

    template<typename T>
    TextWriter& operator<<(const T& val)
    {
        write(val.ToString());
        return *this;
    }
    TextWriter& operator<<(int value)
    {
        write(String(value));
        return *this;
    }
    TextWriter& operator<<(float value)
    {
        write(String(value));
        return *this;
    }
    TextWriter& operator<<(double value)
    {
        write(String(value));
        return *this;
    }
    TextWriter& operator<<(const char* value)
    {
        writeSlice(UnownedStringSlice(value));
        return *this;
    }
    TextWriter& operator<<(const String& val)
    {
        writeSlice(val.getUnownedSlice());
        return *this;
    }
    TextWriter& operator<<(const _EndLine&)
    {
#ifdef _WIN32
        writeSlice(UnownedStringSlice::fromLiteral("\r\n"));
#else
        writeSlice(UnownedStringSlice::fromLiteral("\n"));
#endif
        return *this;
    }
};

class StreamWriter : public TextWriter
{
public:
    // TextWriter
    virtual SlangResult writeSlice(const UnownedStringSlice& slice) SLANG_OVERRIDE;
    virtual void close() SLANG_OVERRIDE { m_stream->close(); }

    void releaseStream() { m_stream.setNull(); }

    StreamWriter() {}

    SlangResult init(const String& path, CharEncoding* encoding = CharEncoding::UTF8);
    SlangResult init(RefPtr<Stream> stream, CharEncoding* encoding = CharEncoding::UTF8);

private:
    List<Byte> m_encodingBuffer;
    RefPtr<Stream> m_stream;
    CharEncoding* m_encoding = nullptr;
};

} // namespace Slang

#endif
