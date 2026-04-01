// slang-json-lexer.cpp
#include "slang-json-lexer.h"

#include "../core/slang-char-util.h"
#include "slang-json-diagnostics.h"

/*
https://www.json.org/json-en.html
*/

namespace Slang
{

/* static */ UnownedStringSlice JSONLexer::calcLexemeLocation(const UnownedStringSlice& text)
{
    SourceManager sourceManager;
    sourceManager.initialize(nullptr, nullptr);
    DiagnosticSink sink;
    sink.init(&sourceManager, nullptr);

    String contents(text);
    SourceFile* sourceFile =
        sourceManager.createSourceFileWithString(PathInfo::makeUnknown(), contents);
    SourceView* sourceView = sourceManager.createSourceView(sourceFile, nullptr, SourceLoc());

    JSONLexer lexer;

    lexer.init(sourceView, &sink);

    if (lexer.peekType() != JSONTokenType::Invalid)
    {
        // Get the start offset
        auto offset = sourceView->getRange().getOffset(lexer.peekLoc());

        return text.subString(offset, lexer.peekLexeme().getLength());
    }
    else
    {
        return text.head(0);
    }
}

SlangResult JSONLexer::init(SourceView* sourceView, DiagnosticSink* sink)
{
    m_sourceView = sourceView;
    m_sink = sink;

    SourceFile* sourceFile = sourceView->getSourceFile();

    // Note that the content must be null terminated (because of other requirements)
    SLANG_ASSERT(sourceFile && sourceFile->hasContent());

    m_contentStart = sourceFile->getContent().begin();

    m_startLoc = sourceView->getRange().begin;

    m_lexemeStart = m_contentStart;
    m_cursor = m_lexemeStart;

    // We need to prime the first token
    advance();

    return SLANG_OK;
}

SLANG_FORCE_INLINE static const char* _handleEndOfLine(char c, const char* cursor)
{
    SLANG_ASSERT(c == '\n' || c == '\r');
    const char d = *cursor;
    return cursor + Index((c ^ d) == ('\n' ^ '\r'));
}

JSONTokenType JSONLexer::_setInvalidToken()
{
    return _setToken(JSONTokenType::Invalid, m_lexemeStart);
}

SlangResult JSONLexer::expect(JSONTokenType type)
{
    if (type != peekType())
    {
        m_sink->diagnose(
            m_token.loc,
            JSONDiagnostics::unexpectedTokenExpectedTokenType,
            getJSONTokenAsText(peekType()),
            getJSONTokenAsText(type));
        return SLANG_FAIL;
    }

    advance();
    return SLANG_OK;
}

SlangResult JSONLexer::expect(JSONTokenType type, JSONToken& out)
{
    if (type != peekType())
    {
        m_sink->diagnose(
            m_token.loc,
            JSONDiagnostics::unexpectedTokenExpectedTokenType,
            getJSONTokenAsText(peekType()),
            getJSONTokenAsText(type));
        return SLANG_FAIL;
    }

    out = m_token;
    advance();
    return SLANG_OK;
}

bool JSONLexer::advanceIf(JSONTokenType type)
{
    if (type == peekType())
    {
        advance();
        return true;
    }
    return false;
}

bool JSONLexer::advanceIf(JSONTokenType type, JSONToken& out)
{
    if (type == peekType())
    {
        out = m_token;
        advance();
        return true;
    }
    return false;
}

UnownedStringSlice JSONLexer::getLexeme(const JSONToken& tok) const
{
    auto offset = m_sourceView->getRange().getOffset(tok.loc);
    return UnownedStringSlice(m_sourceView->getContent().begin() + offset, tok.length);
}

JSONTokenType JSONLexer::advance()
{
    const char* cursor = m_cursor;

    while (true)
    {
        m_lexemeStart = cursor;

        const char c = *cursor++;

        switch (c)
        {
        case 0:
            return _setToken(JSONTokenType::EndOfFile, cursor - 1);
        case '"':
            {
                cursor = _lexString(cursor);
                if (cursor == nullptr)
                {
                    return _setInvalidToken();
                }
                return _setToken(JSONTokenType::StringLiteral, cursor);
            }
        case '/':
            {
                // We allow comments
                const char nextChar = *m_cursor;

                if (nextChar == '/')
                {
                    // Line comment
                    cursor = _lexLineComment(cursor);
                }
                else if (nextChar == '*')
                {
                    cursor = _lexBlockComment(cursor);
                    // Can fail...
                    if (cursor == nullptr)
                    {
                        return _setInvalidToken();
                    }
                }
                else
                {
                    return _setInvalidToken();
                }
                break;
            }
        case ' ':
        case '\t':
        case '\n':
        case '\r':
            {
                cursor = _lexWhitespace(cursor);
                break;
            }
        case ':':
            return _setToken(JSONTokenType::Colon, cursor);
        case ',':
            return _setToken(JSONTokenType::Comma, cursor);
        case '[':
            return _setToken(JSONTokenType::LBracket, cursor);
        case ']':
            return _setToken(JSONTokenType::RBracket, cursor);
        case '{':
            return _setToken(JSONTokenType::LBrace, cursor);
        case '}':
            return _setToken(JSONTokenType::RBrace, cursor);

        case '-':
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            {
                LexResult res = _lexNumber(cursor - 1);
                if (res.cursor == nullptr)
                {
                    return _setToken(JSONTokenType::Invalid, m_lexemeStart);
                }
                return _setToken(res.type, res.cursor);
            }
        case 't':
            {
                if (cursor[0] == 'r' && cursor[1] == 'u' && cursor[2] == 'e')
                {
                    return _setToken(JSONTokenType::True, cursor + 3);
                }
                m_sink->diagnose(_getLoc(m_lexemeStart), JSONDiagnostics::expectingValueName);
                return _setInvalidToken();
            }
        case 'f':
            {
                if (cursor[0] == 'a' && cursor[1] == 'l' && cursor[2] == 's' && cursor[3] == 'e')
                {
                    return _setToken(JSONTokenType::False, cursor + 4);
                }
                m_sink->diagnose(_getLoc(m_lexemeStart), JSONDiagnostics::expectingValueName);
                return _setInvalidToken();
            }
        case 'n':
            {
                if (cursor[0] == 'u' && cursor[1] == 'l' && cursor[2] == 'l')
                {
                    return _setToken(JSONTokenType::Null, cursor + 3);
                }
                m_sink->diagnose(_getLoc(m_lexemeStart), JSONDiagnostics::expectingValueName);
                return _setInvalidToken();
            }
        default:
            {
                StringBuilder buf;
                if (c <= ' ' || c >= 0x7e)
                {
                    static const char s_hex[] = "0123456789abcdef";

                    char hexBuf[5] = "0x";

                    uint32_t value = c;
                    hexBuf[2] = s_hex[((value >> 4) & 0xf)];
                    hexBuf[3] = s_hex[(value & 0xf)];
                    hexBuf[4] = 0;

                    buf << hexBuf;
                }
                else
                {
                    buf << c;
                }

                m_sink->diagnose(_getLoc(m_lexemeStart), JSONDiagnostics::unexpectedCharacter);
                return _setInvalidToken();
            }
        }
    }
}

JSONLexer::LexResult JSONLexer::_lexNumber(const char* cursor)
{
    JSONTokenType tokenType = JSONTokenType::IntegerLiteral;

    if (*cursor == '-')
    {
        cursor++;
    }

    if (*cursor == '0')
    {
        // Can only be followed by . exponent, or nothing
        cursor++;
    }
    else if (*cursor >= '1' && *cursor <= '9')
    {
        cursor++;
        while (CharUtil::isDigit(*cursor))
        {
            cursor++;
        }
    }

    // Theres a fraction
    if (*cursor == '.')
    {
        tokenType = JSONTokenType::FloatLiteral;
        // Skip the dot
        cursor++;
        // Must have at least one digit
        if (!CharUtil::isDigit(*cursor))
        {
            m_sink->diagnose(_getLoc(cursor), JSONDiagnostics::expectingADigit);
            return LexResult{JSONTokenType::Invalid, nullptr};
        }
        // Skip the digit
        cursor++;
        // Skip any more digits
        while (CharUtil::isDigit(*cursor))
            cursor++;
    }

    // Theres an exponent
    if (*cursor == 'e' || *cursor == 'E')
    {
        tokenType = JSONTokenType::FloatLiteral;

        // Has an exponent
        cursor++;

        // Skip +/- if has one
        if (*cursor == '+' || *cursor == '-')
        {
            cursor++;
        }

        // Must have one digit
        if (!CharUtil::isDigit(*cursor))
        {
            m_sink->diagnose(_getLoc(cursor), JSONDiagnostics::expectingADigit);
            return LexResult{JSONTokenType::Invalid, nullptr};
        }

        // Skip the digit
        cursor++;
        // Skip any more digits
        while (CharUtil::isDigit(*cursor))
            cursor++;
    }

    return LexResult{tokenType, cursor};
}

const char* JSONLexer::_lexString(const char* cursor)
{
    // We've skipped the first "
    while (true)
    {
        const char c = *cursor++;

        switch (c)
        {
        case 0:
            {
                m_sink->diagnose(_getLoc(cursor - 1), JSONDiagnostics::endOfFileInLiteral);
                return nullptr;
            }
        case '"':
            {
                return cursor;
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
                                m_sink->diagnose(
                                    _getLoc(cursor),
                                    JSONDiagnostics::expectingAHexDigit);
                                return nullptr;
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

const char* JSONLexer::_lexLineComment(const char* cursor)
{
    for (;;)
    {
        const char c = *cursor++;

        switch (c)
        {
        case '\n':
        case '\r':
            {
                // We need to skip to the next line
                return _handleEndOfLine(c, cursor);
            }
        case 0:
            {
                return cursor - 1;
            }
        }
    }
}

const char* JSONLexer::_lexBlockComment(const char* cursor)
{
    for (;;)
    {
        const char c = *cursor++;
        switch (c)
        {
        case 0:
            {
                m_sink->diagnose(_getLoc(cursor), JSONDiagnostics::endOfFileInComment);
                return nullptr;
            }
        case '*':
            {
                if (*cursor == '/')
                {
                    return cursor + 1;
                }
                break;
            }
        default:
            break;
        }
    }
}

const char* JSONLexer::_lexWhitespace(const char* cursor)
{
    while (true)
    {
        const char c = *cursor;

        // Might want to use CharUtil::isWhitespace...

        switch (c)
        {
        case ' ':
        case '\n':
        case '\r':
        case '\t':
            {
                cursor++;
                break;
            }
        default:
            {
                // Hit non white space
                return cursor;
            }
        }
    }
}

UnownedStringSlice getJSONTokenAsText(JSONTokenType type)
{
    switch (type)
    {
    case JSONTokenType::Invalid:
        return UnownedStringSlice::fromLiteral("invalid");
    case JSONTokenType::IntegerLiteral:
        return UnownedStringSlice::fromLiteral("integer literal");
    case JSONTokenType::FloatLiteral:
        return UnownedStringSlice::fromLiteral("float literal");
    case JSONTokenType::StringLiteral:
        return UnownedStringSlice::fromLiteral("string literal");
    case JSONTokenType::LBracket:
        return UnownedStringSlice::fromLiteral("[");
    case JSONTokenType::RBracket:
        return UnownedStringSlice::fromLiteral("]");
    case JSONTokenType::LBrace:
        return UnownedStringSlice::fromLiteral("{");
    case JSONTokenType::RBrace:
        return UnownedStringSlice::fromLiteral("}");
    case JSONTokenType::Comma:
        return UnownedStringSlice::fromLiteral(",");
    case JSONTokenType::Colon:
        return UnownedStringSlice::fromLiteral(":");
    case JSONTokenType::True:
        return UnownedStringSlice::fromLiteral("true");
    case JSONTokenType::False:
        return UnownedStringSlice::fromLiteral("false");
    case JSONTokenType::Null:
        return UnownedStringSlice::fromLiteral("null");
    case JSONTokenType::EndOfFile:
        return UnownedStringSlice::fromLiteral("end of file");
    default:
        break;
    }
    SLANG_UNEXPECTED("JSONTokenType not known");
}

} // namespace Slang
