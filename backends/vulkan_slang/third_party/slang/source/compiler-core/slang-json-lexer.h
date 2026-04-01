// slang-json-lexer.h
#ifndef SLANG_JSON_LEXER_H
#define SLANG_JSON_LEXER_H

#include "../core/slang-basic.h"
#include "slang-diagnostic-sink.h"
#include "slang-source-loc.h"

namespace Slang
{

enum class JSONTokenType
{
    Invalid,
    IntegerLiteral,
    FloatLiteral,
    StringLiteral,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Colon,
    True,
    False,
    Null,
    EndOfFile,
    CountOf,
};

struct JSONToken
{
    JSONTokenType type; ///< The token type
    SourceLoc loc;      ///< Location in the source file
    uint32_t length;    ///< The length of the token in bytes
};

UnownedStringSlice getJSONTokenAsText(JSONTokenType type);

class JSONLexer
{
public:
    /// Peek the current token
    JSONToken& peekToken() { return m_token; }
    /// Peek the current type
    JSONTokenType peekType() { return m_token.type; }
    /// Peek the current SourceLoc
    SourceLoc peekLoc() { return m_token.loc; }

    /// Get the lexeme of JSONToken
    UnownedStringSlice getLexeme(const JSONToken& tok) const;
    /// Peek the lexeme at the current position
    UnownedStringSlice peekLexeme() const { return getLexeme(m_token); }

    JSONTokenType advance();

    /// Expects a token of type type. If found advances, if not returns an error and outputs to
    /// diagnostic sink
    SlangResult expect(JSONTokenType type);
    /// Same as expect except out will hold the token.
    SlangResult expect(JSONTokenType type, JSONToken& out);

    /// Returns true and advances if current token is type
    bool advanceIf(JSONTokenType type);
    bool advanceIf(JSONTokenType type, JSONToken& out);

    /// Must be called before use
    SlangResult init(SourceView* sourceView, DiagnosticSink* sink);

    /// Determines the first token from text. Useful for diagnostics on DiagnosticSink
    static UnownedStringSlice calcLexemeLocation(const UnownedStringSlice& text);

protected:
    struct LexResult
    {
        JSONTokenType type;
        const char* cursor;
    };

    /// Get the location of the cursor
    SLANG_FORCE_INLINE SourceLoc _getLoc(const char* cursor) const
    {
        return m_startLoc + (cursor - m_contentStart);
    }
    const char* _lexLineComment(const char* cursor);
    const char* _lexBlockComment(const char* cursor);
    const char* _lexWhitespace(const char* cursor);
    const char* _lexString(const char* cursor);
    LexResult _lexNumber(const char* cursor);

    SLANG_FORCE_INLINE JSONTokenType _setToken(JSONTokenType type, const char* cursor)
    {
        SLANG_ASSERT(cursor >= m_lexemeStart);
        m_token.type = type;
        m_token.loc = m_startLoc + (m_lexemeStart - m_contentStart);
        m_token.length = uint32_t(cursor - m_lexemeStart);
        m_cursor = cursor;
        return type;
    }
    JSONTokenType _setInvalidToken();

    JSONToken m_token;

    const char* m_cursor;
    const char* m_lexemeStart;

    const char* m_contentStart;

    SourceLoc m_startLoc;

    SourceView* m_sourceView;
    DiagnosticSink* m_sink;
};

} // namespace Slang

#endif
