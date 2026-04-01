// slang-mangled-lexer.h
#ifndef SLANG_MANGLED_LEXER_H_INCLUDED
#define SLANG_MANGLED_LEXER_H_INCLUDED

#include "../core/slang-basic.h"
#include "../core/slang-char-util.h"
#include "slang-compiler.h"

namespace Slang
{

/* A lexer like utility class used for decoding mangled names.
Expects names to be correctly constructed - any errors will cause asserts/failures */
class MangledLexer
{
public:
    /// Reads a count at current position
    UInt readCount();

    void readGenericParam();

    void readGenericParams();

    SLANG_INLINE void readSimpleIntVal();

    String readRawStringSegment();

    void readNamedType();

    void readType();

    void readVal();

    void readGenericArg() { readVal(); }

    void readGenericArgs();

    SLANG_INLINE void readExtensionSpec();

    UnownedStringSlice readSimpleName();

    UInt readParamCount();

    /// Returns the character at the current position
    char peekChar() { return *m_cursor; }
    // Returns the current character and moves to next character.
    char nextChar() { return *m_cursor++; }

    static String unescapeString(UnownedStringSlice str);

    /// Ctor
    SLANG_FORCE_INLINE MangledLexer(const UnownedStringSlice& slice);

private:
    // Call at the beginning of a mangled name,
    // to strip off the main prefix
    void _start() { _expect("_S"); }

    SLANG_INLINE void _expect(char c);

    void _expect(char const* str)
    {
        while (char c = *str++)
            _expect(c);
    }

    char const* m_cursor = nullptr;
    char const* m_begin = nullptr;
    char const* m_end = nullptr;
};

// -------------------------------------------------------------------------- -
SLANG_FORCE_INLINE MangledLexer::MangledLexer(const UnownedStringSlice& slice)
    : m_cursor(slice.begin()), m_begin(slice.begin()), m_end(slice.end())
{
    _start();
}

// ---------------------------------------------------------------------------
SLANG_INLINE void MangledLexer::readSimpleIntVal()
{
    int c = peekChar();
    if (CharUtil::isDigit((char)c))
    {
        nextChar();
    }
    else
    {
        readVal();
    }
}

// ---------------------------------------------------------------------------
SLANG_INLINE void MangledLexer::readNamedType()
{
    // TODO: handle types with more complicated names
    readRawStringSegment();
}

// ---------------------------------------------------------------------------
SLANG_INLINE void MangledLexer::readExtensionSpec()
{
    _expect("X");
    readType();
}

// ---------------------------------------------------------------------------
SLANG_INLINE void MangledLexer::_expect(char c)
{
    if (peekChar() == c)
    {
        nextChar();
    }
    else
    {
        // ERROR!
        SLANG_UNEXPECTED("mangled name error");
    }
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MangledNameParser !!!!!!!!!!!!!!!!!!!!!!!!!!

struct MangledNameParser
{
    /// Tries to extract the module name from this mangled name.
    static SlangResult parseModuleName(const UnownedStringSlice& in, String& outModuleName);
};

} // namespace Slang
#endif
