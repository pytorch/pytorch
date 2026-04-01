#ifndef SLANG_LEXER_H
#define SLANG_LEXER_H

#include "../core/slang-basic.h"
#include "slang-diagnostic-sink.h"

namespace Slang
{
struct NamePool;

//

struct TokenList
{
    const Token* begin() const;
    const Token* end() const;

    SLANG_FORCE_INLINE void add(const Token& token) { m_tokens.add(token); }

    List<Token> m_tokens;
};

struct TokenSpan
{
    TokenSpan();
    TokenSpan(TokenList const& tokenList)
        : m_begin(tokenList.begin()), m_end(tokenList.end())
    {
    }

    const Token* begin() const { return m_begin; }
    const Token* end() const { return m_end; }

    int getCount() { return (int)(m_end - m_begin); }

    const Token* m_begin;
    const Token* m_end;
};

struct TokenReader
{
    Token m_nextToken;
    TokenReader();
    explicit TokenReader(TokenSpan const& tokens)
        : m_cursor(tokens.begin()), m_end(tokens.end())
    {
        _updateLookaheadToken();
    }
    explicit TokenReader(TokenList const& tokens)
        : m_cursor(tokens.begin()), m_end(tokens.end())
    {
        _updateLookaheadToken();
    }
    explicit TokenReader(Token const* begin, Token const* end)
        : m_cursor(begin), m_end(end)
    {
        _updateLookaheadToken();
    }
    struct ParsingCursor
    {
        bool operator==(const ParsingCursor& rhs) const
        {
            return tokenReaderCursor == rhs.tokenReaderCursor;
        }
        bool operator!=(const ParsingCursor& rhs) const { return !(*this == rhs); }

        bool isValid() const { return tokenReaderCursor != nullptr; }

        Token nextToken;
        const Token* tokenReaderCursor = nullptr;
    };
    ParsingCursor getCursor()
    {
        ParsingCursor rs;
        rs.nextToken = m_nextToken;
        rs.tokenReaderCursor = m_cursor;
        return rs;
    }
    void setCursor(ParsingCursor cursor)
    {
        m_cursor = cursor.tokenReaderCursor;
        m_nextToken = cursor.nextToken;
    }
    bool isAtCursor(const ParsingCursor& cursor) const
    {
        return cursor.tokenReaderCursor == m_cursor;
    }
    bool isAtEnd() const { return m_cursor == m_end; }
    Token& peekToken();
    TokenType peekTokenType() const;
    SourceLoc peekLoc() const;

    Token advanceToken();

    int getCount() { return (int)(m_end - m_cursor); }

    const Token* m_cursor;
    const Token* m_end;
    static Token getEndOfFileToken();

private:
    /// Update the lookahead token in `m_nextToken` to reflect the cursor state
    void _updateLookaheadToken();
};

typedef unsigned int LexerFlags;
enum
{
    kLexerFlag_SuppressDiagnostics = 1
                                     << 2, ///< Suppress errors about invalid/unsupported characters
};

struct Lexer
{
    void initialize(
        SourceView* sourceView,
        DiagnosticSink* sink,
        NamePool* namePool,
        MemoryArena* memoryArena);

    ~Lexer();

    /// Runs the lexer to try and extract a single token, which is returned.
    /// This can be used by the DiagnosticSink to be able to display more appropriate
    /// information when displaying a source location - such as underscoring the
    /// token at that location.
    ///
    /// NOTE! This function is relatively slow, and is designed for use around this specific
    /// purpose. It does not return a token or a token type, because that information is
    /// not needed by the DiagnosticSink.
    static UnownedStringSlice sourceLocationLexer(const UnownedStringSlice& in);

    /// Lex the next token in the input stream, returning an EOF token if at end.
    Token lexToken();

    /// Lex all tokens (up to the end of the stream) that are semantically relevant
    TokenList lexAllSemanticTokens();

    /// Lex all tokens (up to the end of the stream) that are relevant to things like markup
    TokenList lexAllMarkupTokens();

    /// Lex all tokens (up to the end of the stream) whether relevant or not.
    TokenList lexAllTokens();

    /// Get the diagnostic sink, taking into account flags. Will return null if suppressing
    /// diagnostics.
    DiagnosticSink* getDiagnosticSink()
    {
        return ((m_lexerFlags & kLexerFlag_SuppressDiagnostics) == 0) ? m_sink : nullptr;
    }

    SourceLoc findNextLineEnd(SourceLoc from, UInt& lineCount) const;

    SourceView* m_sourceView;
    DiagnosticSink* m_sink;
    NamePool* m_namePool;

    char const* m_cursor;

    char const* m_begin;
    char const* m_end;

    /// The starting sourceLoc (same as first location of SourceView)
    SourceLoc m_startLoc;

    TokenFlags m_tokenFlags;
    LexerFlags m_lexerFlags;

    MemoryArena* m_memoryArena;
};


// Helper routines for extracting values from tokens
String getStringLiteralTokenValue(Token const& token);
String getFileNameTokenValue(Token const& token);

typedef int64_t IntegerLiteralValue;
typedef double FloatingPointLiteralValue;

IntegerLiteralValue getIntegerLiteralValue(
    Token const& token,
    UnownedStringSlice* outSuffix = 0,
    bool* outIsDecimalBase = 0);
FloatingPointLiteralValue getFloatingPointLiteralValue(
    Token const& token,
    UnownedStringSlice* outSuffix = 0);

IntegerLiteralValue getCharLiteralValue(Token const& token);
} // namespace Slang

#endif
