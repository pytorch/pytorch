#ifndef SLANG_CORE_CHAR_UTIL_H
#define SLANG_CORE_CHAR_UTIL_H

#include "slang-string.h"

namespace Slang
{

struct CharUtil
{
    typedef uint8_t Flags;
    struct Flag
    {
        enum Enum : Flags
        {
            Upper = 0x01, ///< A-Z
            Lower = 0x02, ///< a-z
            Digit = 0x04, ///< 0-9
            HorizontalWhitespace =
                0x08,        ///< Whitespace that can appear horizontally (ie excluding CR/LF)
            HexDigit = 0x10, ///< 0-9, a-f, A-F
            VerticalWhitespace = 0x20, ///< \n \r
        };
    };

    SLANG_FORCE_INLINE static bool isDigit(char c) { return c >= '0' && c <= '9'; }
    SLANG_FORCE_INLINE static bool isLower(char c) { return c >= 'a' && c <= 'z'; }
    SLANG_FORCE_INLINE static bool isUpper(char c) { return c >= 'A' && c <= 'Z'; }
    SLANG_FORCE_INLINE static bool isHorizontalWhitespace(char c) { return c == ' ' || c == '\t'; }
    SLANG_FORCE_INLINE static bool isVerticalWhitespace(char c) { return c == '\n' || c == '\r'; }
    SLANG_FORCE_INLINE static bool isWhitespace(char c)
    {
        return (getFlags(c) & (Flag::HorizontalWhitespace | Flag::VerticalWhitespace)) != 0;
    }

    /// True if it's alpha
    SLANG_FORCE_INLINE static bool isAlpha(char c)
    {
        return (getFlags(c) & (Flag::Upper | Flag::Lower)) != 0;
    }
    /// True if it's alpha or a digit
    SLANG_FORCE_INLINE static bool isAlphaOrDigit(char c)
    {
        return (getFlags(c) & (Flag::Upper | Flag::Lower | Flag::Digit)) != 0;
    }

    /// True if the character is a valid hex character
    SLANG_FORCE_INLINE static bool isHexDigit(char c)
    {
        return (getFlags(c) & Flag::HexDigit) != 0;
    }

    /// True if the character is an octal digit
    SLANG_FORCE_INLINE static bool isOctalDigit(char c) { return c >= '0' && c <= '7'; }

    /// For a given character get the associated flags
    SLANG_FORCE_INLINE static Flags getFlags(char c) { return g_charFlagMap.flags[size_t(c)]; }

    /// Given a character return the lower case equivalent
    SLANG_FORCE_INLINE static char toLower(char c)
    {
        return (c >= 'A' && c <= 'Z') ? (c - 'A' + 'a') : c;
    }
    /// Given a character return the upper case equivalent
    SLANG_FORCE_INLINE static char toUpper(char c)
    {
        return (c >= 'a' && c <= 'z') ? (c - 'a' + 'A') : c;
    }

    /// Given a value between 0-15 inclusive returns the hex digit. Uses lower case hex.
    SLANG_FORCE_INLINE static char getHexChar(Index i)
    {
        SLANG_ASSERT((i & ~Index(0xf)) == 0);
        return char(i >= 10 ? (i - 10 + 'a') : (i + '0'));
    }

    /// Returns the value if c interpretted as a decimal digit
    /// If c is not a valid digit returns -1
    inline static int getDecimalDigitValue(char c) { return isDigit(c) ? (c - '0') : -1; }

    /// Returns the value if c interpretted as a hex digit
    /// If c is not a valid hex returns -1
    inline static int getHexDigitValue(char c);

    /// Returns the value if c interpretted as a octal digit
    /// If c is not a valid octal returns -1
    inline static int getOctalDigitValue(char c) { return isOctalDigit(c) ? (c - '0') : -1; }

    struct CharFlagMap
    {
        Flags flags[0x100];
    };

    static CharFlagMap makeCharFlagMap();

    // HACK!
    // JS: Many of the inlined functions of CharUtil just access a global map. That referencing this
    // global is *NOT* enough to link correctly with CharUtil on linux for a shared library. Caling
    // this function can force linkage.
    static int _ensureLink();

    static const CharFlagMap g_charFlagMap;
};

// ------------------------------------------------------------------------------------
inline /* static */ int CharUtil::getHexDigitValue(char c)
{
    if (c >= '0' && c <= '9')
    {
        return c - '0';
    }
    else if (c >= 'a' && c <= 'f')
    {
        return c - 'a' + 10;
    }
    else if (c >= 'A' && c <= 'F')
    {
        return c - 'A' + 10;
    }
    return -1;
}

} // namespace Slang

#endif // SLANG_CHAR_UTIL_H
