#include "slang-string-escape-util.h"

#include "slang-char-util.h"
#include "slang-com-helper.h"
#include "slang-memory-arena.h"
#include "slang-text-io.h"

namespace Slang
{

// !!!!!!!!!!!!!!!!!!!!!!!!!! SpaceStringEscapeHandler !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

class SpaceStringEscapeHandler : public StringEscapeHandler
{
public:
    typedef StringEscapeHandler Super;

    virtual bool isQuotingNeeded(const UnownedStringSlice& slice) SLANG_OVERRIDE
    {
        return isEscapingNeeded(slice);
    }

    virtual bool isEscapingNeeded(const UnownedStringSlice& slice) SLANG_OVERRIDE;
    virtual bool isUnescapingNeeeded(const UnownedStringSlice& slice) SLANG_OVERRIDE;

    virtual SlangResult appendEscaped(const UnownedStringSlice& slice, StringBuilder& out)
        SLANG_OVERRIDE;
    virtual SlangResult appendUnescaped(const UnownedStringSlice& slice, StringBuilder& out)
        SLANG_OVERRIDE;
    virtual SlangResult lexQuoted(const char* cursor, const char** outCursor) SLANG_OVERRIDE;

    SpaceStringEscapeHandler()
        : Super('"')
    {
    }
};

bool SpaceStringEscapeHandler::isEscapingNeeded(const UnownedStringSlice& slice)
{
    return slice.indexOf(' ') >= 0;
}

bool SpaceStringEscapeHandler::isUnescapingNeeeded(const UnownedStringSlice& slice)
{
    SLANG_UNUSED(slice);
    // As it stands we never have to unescape
    return false;
}

SlangResult SpaceStringEscapeHandler::appendUnescaped(
    const UnownedStringSlice& slice,
    StringBuilder& out)
{
    if (slice.indexOf('"') >= 0)
    {
        return SLANG_FAIL;
    }

    out.append(slice);
    return SLANG_OK;
}

SlangResult SpaceStringEscapeHandler::appendEscaped(
    const UnownedStringSlice& slice,
    StringBuilder& out)
{
    if (slice.indexOf('"') >= 0)
    {
        return SLANG_FAIL;
    }
    out.append(slice);
    return SLANG_OK;
}

/* static */ SlangResult SpaceStringEscapeHandler::lexQuoted(
    const char* cursor,
    const char** outCursor)
{
    *outCursor = cursor;

    if (*cursor != m_quoteChar)
    {
        return SLANG_FAIL;
    }
    cursor++;

    for (;;)
    {
        const char c = *cursor;
        if (c == m_quoteChar)
        {
            *outCursor = cursor + 1;
            return SLANG_OK;
        }
        switch (c)
        {
        case 0:
        case '\n':
        case '\r':
            {
                // Didn't hit closing quote!
                return SLANG_FAIL;
            }
        default:
            {
                ++cursor;
                break;
            }
        }
    }
}


// !!!!!!!!!!!!!!!!!!!!!!!!!! CppStringEscapeHandler !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

class CppStringEscapeHandler : public StringEscapeHandler
{
public:
    typedef StringEscapeHandler Super;

    virtual bool isQuotingNeeded(const UnownedStringSlice& slice) SLANG_OVERRIDE
    {
        SLANG_UNUSED(slice);
        return true;
    }
    virtual bool isEscapingNeeded(const UnownedStringSlice& slice) SLANG_OVERRIDE;
    virtual bool isUnescapingNeeeded(const UnownedStringSlice& slice) SLANG_OVERRIDE;
    virtual SlangResult appendEscaped(const UnownedStringSlice& slice, StringBuilder& out)
        SLANG_OVERRIDE;
    virtual SlangResult appendUnescaped(const UnownedStringSlice& slice, StringBuilder& out)
        SLANG_OVERRIDE;
    virtual SlangResult lexQuoted(const char* cursor, const char** outCursor) SLANG_OVERRIDE;

    CppStringEscapeHandler()
        : Super('"')
    {
    }
};

static char _getCppEscapedChar(char c)
{
    switch (c)
    {
    case '\b':
        return 'b';
    case '\f':
        return 'f';
    case '\n':
        return 'n';
    case '\r':
        return 'r';
    case '\a':
        return 'a';
    case '\t':
        return 't';
    case '\v':
        return 'v';
    case '\'':
        return '\'';
    case '\"':
        return '"';
    case '\\':
        return '\\';
    default:
        return 0;
    }
}

static char _getCppUnescapedChar(char c)
{
    switch (c)
    {
    case 'b':
        return '\b';
    case 'f':
        return '\f';
    case 'n':
        return '\n';
    case 'r':
        return '\r';
    case 'a':
        return '\a';
    case 't':
        return '\t';
    case 'v':
        return '\v';
    case '\'':
        return '\'';
    case '\"':
        return '"';
    case '\\':
        return '\\';
    default:
        return 0;
    }
}

bool CppStringEscapeHandler::isUnescapingNeeeded(const UnownedStringSlice& slice)
{
    return slice.indexOf('\\') >= 0;
}

/* static */ bool CppStringEscapeHandler::isEscapingNeeded(const UnownedStringSlice& slice)
{
    const char* cur = slice.begin();
    const char* const end = slice.end();

    for (; cur < end; ++cur)
    {
        const char c = *cur;

        switch (c)
        {
        case '\'':
        case '\"':
        case '\\':
            {
                // Strictly speaking ' shouldn't need a quote if in a C style string.
                return true;
            }
        default:
            {
                if (c < ' ' || c >= 0x7e)
                {
                    return true;
                }
                break;
            }
        }
    }
    return false;
}

SlangResult CppStringEscapeHandler::appendEscaped(
    const UnownedStringSlice& slice,
    StringBuilder& out)
{
    const char* start = slice.begin();
    const char* cur = start;
    const char* const end = slice.end();

    // TODO(JS): A cleverer implementation might support U and u prefixing for unicode characters.
    // For now we just stick with hex if it's not 'regular' ascii.

    for (; cur < end; ++cur)
    {
        const char c = *cur;
        const char escapedChar = _getCppEscapedChar(c);

        if (escapedChar)
        {
            // Flush
            if (start < cur)
            {
                out.append(start, cur);
            }

            out.appendChar('\\');
            out.appendChar(escapedChar);

            start = cur + 1;
        }
        else if (c < ' ' || c > 126)
        {
            // Flush
            if (start < cur)
            {
                out.append(start, cur);
            }

            // NOTE! There is a possible flaw around checking 'next' character (used for outputting
            // oct and hex) If a string is constructed appended in parts, the next character is not
            // available so the problem below can still occur.

            // Another solution to this problem would be to output "", but that makes some other
            // assumptions For example Slang doesn't support that style.

            // C++ greedily consumes hex/octal digits. This is a problem if we have bytes
            // 0, '1' as by default this will output as
            // "\x001" which is the single character byte 1.

            // Note this claims \x is followed with up to 3 hex digits
            // https://msdn.microsoft.com/en-us/library/69ze775t.aspx
            // But the following claims otherwise
            // https://en.cppreference.com/w/cpp/language/string_literal

            // On testing in Visual Studio hex can indeed be more than 3 digits

            // There is a problem outputting values in hex, because C++ allows *any* amount of hex
            // digits. We could work around with \u \U but they are later extensions (C++11) and
            // have other issue

            // The solution taken here is to always output as octal, because octal can be at most 3
            // digits.

            // Special case handling of 0
            if (c == 0 && !(cur + 1 < end && CharUtil::isOctalDigit(cur[1])))
            {
                // We can just output as (octal) "\0"
                out.append("\\0");
            }
            else
            {
                // A slightly more sophisticated implementation could output less digits if needed,
                // if not followed by an octal digit, but for now we go simple and output all 3
                // digits

                const uint32_t v = uint32_t(c);

                char buf[4];
                buf[0] = '\\';
                buf[1] = ((v >> 6) & 3) + '0';
                buf[2] = ((v >> 3) & 7) + '0';
                buf[3] = ((v >> 0) & 7) + '0';

                out.append(buf, buf + 4);
            }

            start = cur + 1;
        }
    }

    // Flush anything remaining
    if (start < end)
    {
        out.append(start, end);
    }
    return SLANG_OK;
}

SlangResult CppStringEscapeHandler::appendUnescaped(
    const UnownedStringSlice& slice,
    StringBuilder& out)
{
    const char* start = slice.begin();
    const char* cur = start;
    const char* const end = slice.end();

    while (cur < end)
    {
        const char c = *cur;

        if (c == '\\')
        {
            // Flush
            if (start < cur)
            {
                out.append(start, cur);
            }

            /// Next
            cur++;

            if (cur >= end)
            {
                // Missing character following '\'
                return SLANG_FAIL;
            }

            const char nextC = *cur++;

            // Need to handle various escape sequence cases
            switch (nextC)
            {
            case '\'':
            case '\"':
            case '\\':
            case '?':
            case 'a':
            case 'b':
            case 'f':
            case 'n':
            case 'r':
            case 't':
            case 'v':
                {
                    const char unescapedChar = _getCppUnescapedChar(nextC);
                    if (unescapedChar == 0)
                    {
                        // Don't know how to unescape that char
                        return SLANG_FAIL;
                    }
                    out.appendChar(unescapedChar);

                    start = cur;
                    break;
                }
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
                {
                    // Rewind back a character, as first digit is the 'nextC'
                    --cur;

                    // Don't need to check for enough characters, because there must be 1 - the
                    // nextC

                    // octal escape: up to 3 characters
                    int value = 0;

                    const char* octEnd = cur + 3;
                    octEnd = (octEnd > end) ? end : octEnd;

                    for (; cur < octEnd; ++cur)
                    {
                        const int digitValue = CharUtil::getOctalDigitValue(*cur);
                        if (digitValue < 0)
                        {
                            break;
                        }
                        value = (value << 3) | digitValue;
                    }
                    out.appendChar(char(value));

                    // Reset start
                    start = cur;
                    break;
                }
            case 'x':
                {
                    /// In the C++ standard we consume hex digits until we hit a non hex digit
                    uint32_t value = 0;
                    for (; cur < end && CharUtil::isHexDigit(*cur); ++cur)
                    {
                        const int digitValue = CharUtil::getHexDigitValue(*cur);
                        if (digitValue < 0)
                        {
                            return SLANG_FAIL;
                        }

                        value = (value << 4) | digitValue;
                    }

                    // If it's ascii, just output it
                    if (value < 0x80)
                    {
                        out.appendChar(char(value));
                    }
                    else
                    {
                        // It's arguable what is appropriate. We only decode/encode 4, which the
                        // current spec has, but 6 are possible, so lets go large.
                        const Index maxUtf8EncodeCount = 6;

                        char* chars = out.prepareForAppend(maxUtf8EncodeCount);
                        int numChars = encodeUnicodePointToUTF8(Char32(value), chars);
                        out.appendInPlace(chars, numChars);
                    }

                    // Reset start
                    start = cur;
                    break;
                }
            case 'u':
            case 'U':
                {
                    // u implies 4 hex digits
                    // U implies 6.

                    // Work out how many digits we need
                    const Count digitCount = (nextC == 'u') ? 4 : 6;

                    // Do we have enough?
                    if (end - cur < digitCount)
                    {
                        return SLANG_FAIL;
                    }

                    uint32_t value = 0;
                    for (Index i = 0; i < digitCount; ++i)
                    {
                        const int digitValue = CharUtil::getHexDigitValue(cur[i]);
                        if (digitValue < 0)
                        {
                            return SLANG_FAIL;
                        }
                        value = (value << 4) | digitValue;
                    }
                    cur += digitCount;

                    // Encode to Utf8
                    // If it's ascii, just output it
                    if (value < 0x80)
                    {
                        out.appendChar(char(value));
                    }
                    else
                    {
                        // It's arguable what is appropriate. We only decode/encode 4, which the
                        // current spec has, but 6 are possible, so lets go large.
                        const Index maxUtf8EncodeCount = 6;

                        char* chars = out.prepareForAppend(maxUtf8EncodeCount);
                        int numChars = encodeUnicodePointToUTF8(Char32(value), chars);
                        out.appendInPlace(chars, numChars);
                    }

                    // Reset start
                    start = cur;
                    break;
                }
            default:
                {
                    return SLANG_FAIL;
                }
            }
        }
        else
        {
            // Next char
            ++cur;
        }
    }

    if (start < end)
    {
        out.append(start, end);
    }

    return SLANG_OK;
}

SlangResult CppStringEscapeHandler::lexQuoted(const char* cursor, const char** outCursor)
{
    *outCursor = cursor;

    if (*cursor != m_quoteChar)
    {
        return SLANG_FAIL;
    }
    cursor++;

    for (;;)
    {
        const char c = *cursor;
        if (c == m_quoteChar)
        {
            *outCursor = cursor + 1;
            return SLANG_OK;
        }
        switch (c)
        {
        case 0:
        case '\n':
        case '\r':
            {
                // Didn't hit closing quote!
                return SLANG_FAIL;
            }
        case '\\':
            {
                ++cursor;
                // Need to handle various escape sequence cases
                switch (*cursor)
                {
                case '\'':
                case '\"':
                case '\\':
                case '?':
                case 'a':
                case 'b':
                case 'f':
                case 'n':
                case 'r':
                case 't':
                case 'v':
                    {
                        ++cursor;
                        break;
                    }
                case '0':
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                case '6':
                case '7':
                    {
                        // octal escape: up to 3 characters
                        ++cursor;
                        for (int ii = 0; ii < 3; ++ii)
                        {
                            const char d = *cursor;
                            if (('0' <= d) && (d <= '7'))
                            {
                                ++cursor;
                                continue;
                            }
                            else
                            {
                                break;
                            }
                        }
                        break;
                    }
                case 'x':
                    {
                        // hexadecimal escape: any number of characters
                        ++cursor;
                        for (; CharUtil::isHexDigit(*cursor); ++cursor)
                            ;

                        // TODO: Unicode escape sequences
                        break;
                    }
                }
                break;
            }
        default:
            {
                ++cursor;
                break;
            }
        }
    }
}

// !!!!!!!!!!!!!!!!!!!!!!!!!! JSONStringEscapeHandler !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

class JSONStringEscapeHandler : public StringEscapeHandler
{
public:
    typedef StringEscapeHandler Super;

    virtual bool isQuotingNeeded(const UnownedStringSlice& slice) SLANG_OVERRIDE
    {
        SLANG_UNUSED(slice);
        return true;
    }
    virtual bool isEscapingNeeded(const UnownedStringSlice& slice) SLANG_OVERRIDE;
    virtual bool isUnescapingNeeeded(const UnownedStringSlice& slice) SLANG_OVERRIDE;
    virtual SlangResult appendEscaped(const UnownedStringSlice& slice, StringBuilder& out)
        SLANG_OVERRIDE;
    virtual SlangResult appendUnescaped(const UnownedStringSlice& slice, StringBuilder& out)
        SLANG_OVERRIDE;
    virtual SlangResult lexQuoted(const char* cursor, const char** outCursor) SLANG_OVERRIDE;

    JSONStringEscapeHandler()
        : Super('"')
    {
    }
};

bool JSONStringEscapeHandler::isUnescapingNeeeded(const UnownedStringSlice& slice)
{
    return slice.indexOf('\\') >= 0;
}

bool JSONStringEscapeHandler::isEscapingNeeded(const UnownedStringSlice& slice)
{
    const char* cur = slice.begin();
    const char* const end = slice.end();

    for (; cur < end; ++cur)
    {
        const char c = *cur;

        switch (c)
        {
        case '\"':
        case '\\':
        case '/':
            {
                return true;
            }
        default:
            {
                if (c < ' ' || c >= 0x7e)
                {
                    return true;
                }
                break;
            }
        }
    }
    return false;
}

SlangResult JSONStringEscapeHandler::lexQuoted(const char* cursor, const char** outCursor)
{
    // We've skipped the first "
    while (true)
    {
        const char c = *cursor++;

        switch (c)
        {
        case 0:
            return SLANG_FAIL;
        case '"':
            {
                *outCursor = cursor;
                return SLANG_OK;
            }
        case '\\':
            {
                const char nextC = *cursor;
                switch (nextC)
                {
                case '"':
                case '\\':
                case '/':
                case 'b':
                case 'f':
                case 'n':
                case 'r':
                case 't':
                    {
                        ++cursor;
                        break;
                    }
                case 'u':
                    {
                        cursor++;
                        for (Index i = 0; i < 4; ++i)
                        {
                            if (!CharUtil::isHexDigit(cursor[i]))
                            {
                                return SLANG_FAIL;
                            }
                        }
                        cursor += 4;
                        break;
                    }
                }
            }
        // Somewhat surprisingly it appears it's valid to have \r\n inside of quotes.
        default:
            break;
        }
    }
}

static char _getJSONEscapedChar(char c)
{
    switch (c)
    {
    case '\b':
        return 'b';
    case '\f':
        return 'f';
    case '\n':
        return 'n';
    case '\r':
        return 'r';
    case '\t':
        return 't';
    case '\\':
        return '\\';
    case '/':
        return '/';
    case '"':
        return '"';
    default:
        return 0;
    }
}

static char _getJSONUnescapedChar(char c)
{
    switch (c)
    {
    case 'b':
        return '\b';
    case 'f':
        return '\f';
    case 'n':
        return '\n';
    case 'r':
        return '\r';
    case 't':
        return '\t';
    case '\\':
        return '\\';
    case '/':
        return '/';
    case '"':
        return '"';
    default:
        return 0;
    }
}

static const char s_hex[] = "0123456789abcdef";

// Outputs ioSlice with the chars remaining after utf8 encoded value
// Returns ~uint32_t(0) if can't decode
static uint32_t _getUnicodePointFromUTF8(UnownedStringSlice& ioSlice)
{
    const Index length = ioSlice.getLength();
    SLANG_ASSERT(length > 0);
    const char* cur = ioSlice.begin();

    uint32_t codePoint = 0;
    unsigned int leading = cur[0];
    unsigned int mask = 0x80;

    Index count = 0;
    while (leading & mask)
    {
        count++;
        mask >>= 1;
    }

    if (count > length)
    {
        SLANG_ASSERT(!"Can't decode");
        ioSlice = UnownedStringSlice(ioSlice.end(), ioSlice.end());
        return ~uint32_t(0);
    }

    codePoint = (leading & (mask - 1));
    for (Index i = 1; i <= count - 1; i++)
    {
        codePoint <<= 6;
        codePoint += (cur[i] & 0x3F);
    }

    ioSlice = UnownedStringSlice(cur + count, ioSlice.end());
    return codePoint;
}

static void _appendHex16(uint32_t value, StringBuilder& out)
{
    // Let's go with hex
    char buf[] = "\\u0000";

    buf[2] = s_hex[(value >> 12) & 0xf];
    buf[3] = s_hex[(value >> 8) & 0xf];
    buf[4] = s_hex[(value >> 4) & 0xf];
    buf[5] = s_hex[(value >> 0) & 0xf];

    out.append(UnownedStringSlice(buf, 6));
}

SlangResult JSONStringEscapeHandler::appendEscaped(
    const UnownedStringSlice& slice,
    StringBuilder& out)
{
    const char* start = slice.begin();
    const char* cur = start;
    const char* const end = slice.end();

    for (; cur < end; ++cur)
    {
        const char c = *cur;

        const char escapedChar = _getJSONEscapedChar(c);

        if (escapedChar)
        {
            // Flush
            if (start < cur)
            {
                out.append(start, cur);
            }
            out.appendChar('\\');
            out.appendChar(escapedChar);

            start = cur + 1;
        }
        else if (uint8_t(c) & 0x80)
        {
            // Flush
            if (start < cur)
            {
                out.append(start, cur);
            }

            // UTF8
            UnownedStringSlice remainingSlice(cur, end);
            uint32_t codePoint = _getUnicodePointFromUTF8(remainingSlice);

            // We only support up to 16 bit unicode values for now...
            SLANG_ASSERT(codePoint < 0x10000);

            _appendHex16(codePoint, out);

            cur = remainingSlice.begin() - 1;
            start = cur + 1;
        }
        else if (uint8_t(c) < ' ' || (c >= 0x7e))
        {
            if (start < cur)
            {
                out.append(start, cur);
            }

            _appendHex16(uint32_t(c), out);

            start = cur + 1;
        }
        else
        {
            // Can go out as it is
        }
    }

    // Flush at the end
    if (start < end)
    {
        out.append(start, end);
    }
    return SLANG_OK;
}

SlangResult JSONStringEscapeHandler::appendUnescaped(
    const UnownedStringSlice& slice,
    StringBuilder& out)
{
    const char* start = slice.begin();
    const char* cur = start;
    const char* const end = slice.end();

    for (; cur < end; ++cur)
    {
        const char c = *cur;

        if (c == '\\')
        {
            // Flush
            if (start < cur)
            {
                out.append(start, cur);
            }

            /// Next
            cur++;

            if (cur >= end)
            {
                return SLANG_FAIL;
            }

            // Need to handle various escape sequence cases
            switch (*cur)
            {
            case '\"':
            case '\\':
            case '/':
            case 'b':
            case 'f':
            case 'n':
            case 'r':
            case 't':
                {
                    const char unescapedChar = _getJSONUnescapedChar(*cur);
                    if (unescapedChar == 0)
                    {
                        // Don't know how to unescape that char
                        return SLANG_FAIL;
                    }
                    out.appendChar(unescapedChar);

                    start = cur + 1;
                    break;
                }
            case 'u':
                {
                    uint32_t value = 0;
                    cur++;

                    if (cur + 4 > end)
                    {
                        return SLANG_FAIL;
                    }

                    for (Index i = 0; i < 4; ++i)
                    {
                        const char digitC = cur[i];

                        uint32_t digitValue;
                        if (digitC >= '0' && digitC <= '9')
                        {
                            digitValue = digitC - '0';
                        }
                        else if (digitC >= 'a' && digitC <= 'f')
                        {
                            digitValue = digitC - 'a' + 10;
                        }
                        else if (digitC >= 'A' && digitC <= 'F')
                        {
                            digitValue = digitC - 'A' + 10;
                        }
                        else
                        {
                            return SLANG_FAIL;
                        }
                        SLANG_ASSERT(digitValue < 0x10);
                        value = (value << 4) | digitValue;
                    }
                    cur += 4;

                    // NOTE! Strictly speaking we may want to combine 2 UTF16 surrogates to make a
                    // single UTF8 encoded char.

                    // Need to encode in UTF8 to concat

                    char buf[8];
                    int len = encodeUnicodePointToUTF8(Char32(value), buf);

                    out.append(buf, buf + len);

                    start = cur;
                    cur--;
                    break;
                }
            default:
                {
                    // Can't decode
                    return SLANG_FAIL;
                }
            }
        }
    }

    // Flush
    if (start < end)
    {
        out.append(start, end);
    }

    return SLANG_OK;
}

// !!!!!!!!!!!!!!!!!!!!!!!!!! StringEscapeUtil !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

static CppStringEscapeHandler g_cppHandler;
static SpaceStringEscapeHandler g_spaceHandler;
static JSONStringEscapeHandler g_jsonHandler;

StringEscapeUtil::Handler* StringEscapeUtil::getHandler(Style style)
{
    switch (style)
    {
    case Style::Cpp:
        return &g_cppHandler;
    case Style::Space:
        return &g_spaceHandler;
    case Style::JSON:
        return &g_jsonHandler;
    // TODO(JS): For now we make Slang language string encoding/decoding the same as C++
    // That may not be desirable because C++ has a variety of surprising edge cases (for example
    // around \x)
    case Style::Slang:
        return &g_cppHandler;
    default:
        return nullptr;
    }
}

/* static */ SlangResult StringEscapeUtil::appendQuoted(
    Handler* handler,
    const UnownedStringSlice& slice,
    StringBuilder& out)
{
    const char quoteChar = handler->getQuoteChar();
    out.appendChar(quoteChar);
    SlangResult res = handler->appendEscaped(slice, out);
    out.appendChar(quoteChar);
    return res;
}

/* static */ SlangResult StringEscapeUtil::appendUnquoted(
    Handler* handler,
    const UnownedStringSlice& slice,
    StringBuilder& out)
{
    const Index len = slice.getLength();

    const char quoteChar = handler->getQuoteChar();
    SLANG_UNUSED(quoteChar);

    // Must have quote characters around if
    SLANG_ASSERT(len >= 2 && slice[0] == quoteChar && slice[len - 1] == quoteChar);

    return handler->appendUnescaped(slice.subString(1, len - 2), out);
}

/* static */ SlangResult StringEscapeUtil::appendMaybeQuoted(
    Handler* handler,
    const UnownedStringSlice& slice,
    StringBuilder& out)
{
    if (handler->isQuotingNeeded(slice))
    {
        return appendQuoted(handler, slice, out);
    }
    else
    {
        out.append(slice);
        return SLANG_OK;
    }
}

/* static */ bool StringEscapeUtil::isQuoted(char quoteChar, UnownedStringSlice& slice)
{
    const Index len = slice.getLength();
    return len >= 2 && slice[0] == quoteChar && slice[len - 1] == quoteChar;
}

/* static */ UnownedStringSlice StringEscapeUtil::unquote(
    char quoteChar,
    const UnownedStringSlice& slice)
{
    const Index len = slice.getLength();
    if (len >= 2 && slice[0] == quoteChar && slice[len - 1] == quoteChar)
    {
        return UnownedStringSlice(slice.begin() + 1, len - 2);
    }
    SLANG_ASSERT(!"Not quoted!");
    return UnownedStringSlice();
}

/* static */ SlangResult StringEscapeUtil::appendMaybeUnquoted(
    Handler* handler,
    const UnownedStringSlice& slice,
    StringBuilder& out)
{
    const char quoteChar = handler->getQuoteChar();

    const Index len = slice.getLength();

    if (len >= 2 && slice[0] == quoteChar && slice[len - 1] == quoteChar)
    {
        return appendUnquoted(handler, slice, out);
    }
    else
    {
        out.append(slice);
        return SLANG_OK;
    }
}

/* static */ SlangResult StringEscapeUtil::isUnescapeShellLikeNeeded(
    Handler* handler,
    const UnownedStringSlice& slice)
{
    return slice.indexOf(handler->getQuoteChar()) >= 0;
}

/* static */ SlangResult StringEscapeUtil::unescapeShellLike(
    Handler* handler,
    const UnownedStringSlice& slice,
    StringBuilder& out)
{
    StringBuilder buf;
    const char quoteChar = handler->getQuoteChar();

    UnownedStringSlice remaining(slice);

    while (remaining.getLength())
    {
        const Index index = remaining.indexOf(quoteChar);

        if (index < 0)
        {
            out.append(remaining);
            return SLANG_OK;
        }

        // Append the bit before
        out.append(remaining.head(index));

        // Okay we need to lex to the end

        const char* quotedEnd = nullptr;
        SLANG_RETURN_ON_FAIL(handler->lexQuoted(remaining.begin() + index, &quotedEnd));

        // Unescape it
        SLANG_RETURN_ON_FAIL(
            appendUnquoted(handler, UnownedStringSlice(remaining.begin() + index, quotedEnd), out));

        // Fix up remaining
        remaining = UnownedStringSlice(quotedEnd, remaining.end());
    }

    return SLANG_OK;
}

String StringEscapeUtil::escapeString(UnownedStringSlice input, StringEscapeUtil::Style style)
{
    StringBuilder sb;
    auto handler = StringEscapeUtil::getHandler(style);
    StringEscapeUtil::appendQuoted(handler, input, sb);
    return sb.produceString();
}

String StringEscapeUtil::unescapeString(UnownedStringSlice input, StringEscapeUtil::Style style)
{
    StringBuilder sb;
    auto handler = StringEscapeUtil::getHandler(style);
    StringEscapeUtil::appendUnquoted(handler, input, sb);
    return sb.produceString();
}
} // namespace Slang
