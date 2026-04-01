// slang-token.h
#ifndef SLANG_TOKEN_H_INCLUDED
#define SLANG_TOKEN_H_INCLUDED

#include "../core/slang-basic.h"
#include "slang-name.h"
#include "slang-source-loc.h"

namespace Slang
{

class Name;

enum class TokenType : uint8_t
{
#define TOKEN(NAME, DESC) NAME,
#include "slang-token-defs.h"
};

char const* TokenTypeToString(TokenType type);

typedef uint8_t TokenFlags;
struct TokenFlag
{
    enum Enum : TokenFlags
    {
        AtStartOfLine = 1 << 0,
        AfterWhitespace = 1 << 1,
        ScrubbingNeeded = 1 << 2,
        Name = 1 << 3, ///< Determines if 'name' is set or 'chars' in the charsNameUnion
    };
};

class Token
{
public:
    TokenType type = TokenType::Unknown;
    TokenFlags flags = 0;

    SourceLoc loc;
    uint32_t charsCount = 0; ///< Amount of characters. Is set if name or not.

    union CharsNameUnion
    {
        const char* chars;
        Name* name;
    };

    CharsNameUnion charsNameUnion;

    bool hasContent() const { return charsCount > 0; }
    Index getContentLength() const { return charsCount; }

    UnownedStringSlice getContent() const;
    /// Set content
    void setContent(const UnownedStringSlice& content);

    Name* getName() const;

    Name* getNameOrNull() const;

    SourceLoc getLoc() const { return loc; }

    /// Set the name
    SLANG_FORCE_INLINE void setName(Name* inName);

    Token() { charsNameUnion.chars = nullptr; }

    Token(
        TokenType inType,
        const UnownedStringSlice& inContent,
        SourceLoc inLoc,
        TokenFlags inFlags = 0)
        : flags(inFlags)
    {
        SLANG_ASSERT((inFlags & TokenFlag::Name) == 0);
        type = inType;
        charsNameUnion.chars = inContent.begin();
        charsCount = uint32_t(inContent.getLength());
        loc = inLoc;
    }
    Token(TokenType inType, Name* name, SourceLoc inLoc, TokenFlags inFlags = 0)
    {
        SLANG_ASSERT(name);
        type = inType;
        flags = inFlags | TokenFlag::Name;
        charsNameUnion.name = name;
        charsCount = uint32_t(name->text.getLength());
        loc = inLoc;
    }
};

// ---------------------------------------------------------------------------
SLANG_FORCE_INLINE UnownedStringSlice Token::getContent() const
{
    return (flags & TokenFlag::Name) ? charsNameUnion.name->text.getUnownedSlice()
                                     : UnownedStringSlice(charsNameUnion.chars, charsCount);
}

// ---------------------------------------------------------------------------
SLANG_FORCE_INLINE Name* Token::getName() const
{
    return getNameOrNull();
}

// ---------------------------------------------------------------------------
SLANG_FORCE_INLINE Name* Token::getNameOrNull() const
{
    return (flags & TokenFlag::Name) ? charsNameUnion.name : nullptr;
}

// ---------------------------------------------------------------------------
SLANG_FORCE_INLINE void Token::setContent(const UnownedStringSlice& content)
{
    flags &= ~TokenFlag::Name;
    charsNameUnion.chars = content.begin();
    charsCount = uint32_t(content.getLength());
}

// ---------------------------------------------------------------------------
SLANG_FORCE_INLINE void Token::setName(Name* inName)
{
    SLANG_ASSERT(inName);
    flags |= TokenFlag::Name;
    charsNameUnion.name = inName;
    charsCount = uint32_t(inName->text.getLength());
}


} // namespace Slang

#endif
