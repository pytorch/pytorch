#ifndef SLANG_CORE_CHAR_ENCODE_H
#define SLANG_CORE_CHAR_ENCODE_H

#include "slang-basic.h"
#include "slang-secure-crt.h"

namespace Slang
{

// NOTE! Order must be kept the same to match up with
enum class CharEncodeType
{
    UTF8,
    UTF16,
    UTF16Reversed,
    UTF32,
    CountOf,
};

template<typename ReadByteFunc>
Char32 getUnicodePointFromUTF8(const ReadByteFunc& readByte)
{
    Char32 codePoint = 0;
    uint32_t leading = Byte(readByte());
    uint32_t mask = 0x80;
    Index count = 0;
    while (leading & mask)
    {
        count++;
        mask >>= 1;
    }
    codePoint = (leading & (mask - 1));
    for (Index i = 1; i <= count - 1; i++)
    {
        codePoint <<= 6;
        codePoint += (readByte() & 0x3F);
    }
    return codePoint;
}

template<typename ReadByteFunc>
Char32 getUnicodePointFromUTF16(const ReadByteFunc& readByte)
{
    uint32_t byte0 = Byte(readByte());
    uint32_t byte1 = Byte(readByte());
    uint32_t word0 = byte0 + (byte1 << 8);
    if (word0 >= 0xD800 && word0 <= 0xDFFF)
    {
        uint32_t byte2 = Byte(readByte());
        uint32_t byte3 = Byte(readByte());
        uint32_t word1 = byte2 + (byte3 << 8);
        return Char32(((word0 & 0x3FF) << 10) + (word1 & 0x3FF) + 0x10000);
    }
    else
        return Char32(word0);
}

template<typename ReadByteFunc>
Char32 getUnicodePointFromUTF16Reversed(const ReadByteFunc& readByte)
{
    uint32_t byte0 = Byte(readByte());
    uint32_t byte1 = Byte(readByte());
    uint32_t word0 = (byte0 << 8) + byte1;
    if (word0 >= 0xD800 && word0 <= 0xDFFF)
    {
        uint32_t byte2 = Byte(readByte());
        uint32_t byte3 = Byte(readByte());
        uint32_t word1 = (byte2 << 8) + byte3;
        return Char32(((word0 & 0x3FF) << 10) + (word1 & 0x3FF));
    }
    else
        return Char32(word0);
}

template<typename ReadByteFunc>
Char32 getUnicodePointFromUTF32(const ReadByteFunc& readByte)
{
    uint32_t byte0 = Byte(readByte());
    uint32_t byte1 = Byte(readByte());
    uint32_t byte2 = Byte(readByte());
    uint32_t byte3 = Byte(readByte());
    return Char32(byte0 + (byte1 << 8) + (byte2 << 16) + (byte3 << 24));
}

// Encode functions return the amount of elements output to the buffer
inline int encodeUnicodePointToUTF8(Char32 codePoint, char* outBuffer)
{
    char* const dst = outBuffer;
    // TODO(JS): This supports 4 + 6 * 3 = 22 bits.
    // The standard allows up to 0x10FFFF.
    if (codePoint <= 0x7F)
    {
        dst[0] = char(codePoint);
        return 1;
    }
    else if (codePoint <= 0x7FF)
    {
        dst[0] = char(0xC0 + (codePoint >> 6));
        dst[1] = char(0x80 + (codePoint & 0x3F));
        return 2;
    }
    else if (codePoint <= 0xFFFF)
    {
        dst[0] = char(0xE0 + (codePoint >> 12));
        dst[1] = char(0x80 + ((codePoint >> 6) & (0x3F)));
        dst[2] = char(0x80 + (codePoint & 0x3F));
        return 3;
    }
    else
    {
        dst[0] = char(0xF0 + (codePoint >> 18));
        dst[1] = char(0x80 + ((codePoint >> 12) & 0x3F));
        dst[2] = char(0x80 + ((codePoint >> 6) & 0x3F));
        dst[3] = char(0x80 + (codePoint & 0x3F));
        return 4;
    }
}

inline int encodeUnicodePointToUTF16(Char32 codePoint, Char16* outBuffer)
{
    Char16* const dst = outBuffer;
    if (codePoint <= 0xD7FF || (codePoint >= 0xE000 && codePoint <= 0xFFFF))
    {
        dst[0] = Char16(codePoint);
        return 1;
    }
    else
    {
        const uint32_t sub = codePoint - 0x10000;
        dst[0] = Char16((sub >> 10) + 0xD800);
        dst[1] = Char16((sub & 0x3FF) + 0xDC00);
        return 2;
    }
}

SLANG_FORCE_INLINE Char16 reverseByteOrder(Char16 val)
{
    return (val >> 8) | (val << 8);
}

inline int encodeUnicodePointToUTF16Reversed(Char32 codePoint, Char16* outBuffer)
{
    Char16* const dst = outBuffer;
    if (codePoint <= 0xD7FF || (codePoint >= 0xE000 && codePoint <= 0xFFFF))
    {
        dst[0] = reverseByteOrder(Char16(codePoint));
        return 1;
    }
    else
    {
        const uint32_t sub = codePoint - 0x10000;
        const uint32_t high = (sub >> 10) + 0xD800;
        const uint32_t low = (sub & 0x3FF) + 0xDC00;
        dst[0] = reverseByteOrder(Char16(high));
        dst[1] = reverseByteOrder(Char16(low));
        return 2;
    }
}

static const Char16 kUTF16Header = 0xFEFF;
static const Char16 kUTF16ReversedHeader = 0xFFFE;

class CharEncoding
{
public:
    static CharEncoding *UTF8, *UTF16, *UTF16Reversed, *UTF32;

    /// Encode Utf8 held in slice append into ioBuffer
    virtual void encode(const UnownedStringSlice& str, List<Byte>& ioBuffer) = 0;
    /// Decode buffer into Utf8 held in ioBuffer
    virtual void decode(const Byte* buffer, int length, List<char>& ioBuffer) = 0;

    virtual ~CharEncoding() {}

    /// Get the encoding type
    CharEncodeType getEncodingType() const { return m_encodingType; }

    /// Given some bytes determines a character encoding type, based on the initial bytes.
    /// If can't be determined will assume UTF8.
    /// Outputs the offset to the first non mark in outOffset
    static CharEncodeType determineEncoding(
        const Byte* bytes,
        size_t bytesCount,
        size_t& outOffset);

    /// Get the
    static CharEncoding* getEncoding(CharEncodeType type) { return g_encoding[Index(type)]; }

    CharEncoding(CharEncodeType encodingType)
        : m_encodingType(encodingType)
    {
    }

protected:
    CharEncodeType m_encodingType;

    static CharEncoding* const g_encoding[Index(CharEncodeType::CountOf)];
};

struct UTF8Util
{
    /// Given a slice calculate the number of code points (unicode chars)
    ///
    /// NOTE! This doesn't check the *validity* of code points/encoding.
    /// Non valid utf8 input or ending starting in partial characters, will produce
    /// undefined results without error.
    static Index calcCodePointCount(const UnownedStringSlice& in);


    /// Given a slice in UTF8, calculate the number of UTF16 characters needed to represent the
    /// string.
    static Index calcUTF16CharCount(const UnownedStringSlice& in);
};

} // namespace Slang

#endif
