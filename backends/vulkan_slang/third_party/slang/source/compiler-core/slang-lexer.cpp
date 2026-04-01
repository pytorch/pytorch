// slang-lexer.cpp
#include "slang-lexer.h"

// This file implements the lexer/scanner, which is responsible for taking a raw stream of
// input bytes and turning it into semantically useful tokens.
//

#include "core/slang-char-encode.h"
#include "core/slang-string-escape-util.h"
#include "slang-core-diagnostics.h"
#include "slang-name.h"
#include "slang-source-loc.h"

namespace Slang
{
Token TokenReader::getEndOfFileToken()
{
    return Token(TokenType::EndOfFile, UnownedStringSlice::fromLiteral(""), SourceLoc());
}

const Token* TokenList::begin() const
{
    SLANG_ASSERT(m_tokens.getCount());
    return &m_tokens[0];
}

const Token* TokenList::end() const
{
    SLANG_ASSERT(m_tokens.getCount());
    SLANG_ASSERT(m_tokens[m_tokens.getCount() - 1].type == TokenType::EndOfFile);
    return &m_tokens[m_tokens.getCount() - 1];
}

TokenSpan::TokenSpan()
    : m_begin(nullptr), m_end(nullptr)
{
}

TokenReader::TokenReader()
    : m_cursor(nullptr), m_end(nullptr)
{
    _updateLookaheadToken();
}

Token& TokenReader::peekToken()
{
    return m_nextToken;
}

TokenType TokenReader::peekTokenType() const
{
    return m_nextToken.type;
}

SourceLoc TokenReader::peekLoc() const
{
    return m_nextToken.loc;
}

Token TokenReader::advanceToken()
{
    Token result = m_nextToken;
    if (m_cursor != m_end)
        m_cursor++;
    _updateLookaheadToken();
    return result;
}

void TokenReader::_updateLookaheadToken()
{
    // We assume here that we can read a token from a non-null `m_cursor`
    // *even* in the case where `m_cursor == m_end`, because the invariant
    // for lists of tokens is that they should be terminated with and
    // end-of-file token, so that there is always a token "one past the end."
    //
    m_nextToken = m_cursor ? *m_cursor : getEndOfFileToken();

    // If the token we read came from the end of the sub-sequence we are
    // reading, then we will change the token type to an end-of-file token
    // so that code that reads from the sequence and expects a terminating
    // EOF will find it.
    //
    // TODO: We might eventually want a way to look at the actual token type
    // and not just use EOF in all cases: e.g., when emitting diagnostic
    // messages that include the token that is seen.
    //
    if (m_cursor == m_end)
        m_nextToken.type = TokenType::EndOfFile;
}

// Lexer

void Lexer::initialize(
    SourceView* sourceView,
    DiagnosticSink* sink,
    NamePool* namePool,
    MemoryArena* memoryArena)
{
    m_sourceView = sourceView;
    m_sink = sink;
    m_namePool = namePool;
    m_memoryArena = memoryArena;

    auto content = sourceView->getContent();

    m_begin = content.begin();
    m_cursor = content.begin();
    m_end = content.end();

    // Set the start location
    m_startLoc = sourceView->getRange().begin;

    // The first token read from a translation unit should be considered to be at
    // the start of a line, and *also* as coming after whitespace (conceptually
    // both the end-of-file and beginning-of-file pseudo-tokens are whitespace).
    //
    m_tokenFlags = TokenFlag::AtStartOfLine | TokenFlag::AfterWhitespace;
    m_lexerFlags = 0;
}

Lexer::~Lexer() {}

enum
{
    kEOF = -1
};

// Get the next input byte, without any handling of
// escaped newlines, non-ASCII code points, source locations, etc.
static int _peekRaw(Lexer* lexer)
{
    // If we are at the end of the input, return a designated end-of-file value
    if (lexer->m_cursor == lexer->m_end)
        return kEOF;

    // Otherwise, just look at the next byte
    return *lexer->m_cursor;
}

// Read one input byte without any special handling (similar to `peekRaw`)
static int _advanceRaw(Lexer* lexer)
{
    // The logic here is basically the same as for `peekRaw()`,
    // escape we advance `cursor` if we aren't at the end.

    if (lexer->m_cursor == lexer->m_end)
        return kEOF;

    return *lexer->m_cursor++;
}

// When the cursor is already at the first byte of an end-of-line sequence,
// consume one or two bytes that compose the sequence.
//
// Basically, a newline is one of:
//
//  "\n"
//  "\r"
//  "\r\n"
//  "\n\r"
//
// We always look for the longest match possible.
//
static void _handleNewLineInner(Lexer* lexer, int c)
{
    SLANG_ASSERT(c == '\n' || c == '\r');

    int d = _peekRaw(lexer);
    if ((c ^ d) == ('\n' ^ '\r'))
    {
        _advanceRaw(lexer);
    }
}

// Look ahead one code point, dealing with complications like
// escaped newlines.
static int _peek(Lexer* lexer, int offset = 0)
{
    int pos = 0;
    int c = kEOF;

    do
    {
        if (lexer->m_cursor + pos == lexer->m_end)
            return kEOF;

        c = lexer->m_cursor[pos++];

        while (c == '\\')
        {
            // We might have a backslash-escaped newline.
            // Look at the next byte (if any) to see.
            //
            // Note(tfoley): We are assuming a null-terminated input here,
            // so that we can safely look at the next byte without issue.
            int d = lexer->m_cursor[pos++];
            switch (d)
            {
            case '\r':
            case '\n':
                {
                    // The newline was escaped, so return the code point after *that*
                    int e = lexer->m_cursor[pos++];
                    if ((d ^ e) == ('\r' ^ '\n'))
                        c = lexer->m_cursor[pos++];
                    else
                        c = e;
                    continue;
                }
            default:
                break;
            }

            // Only continue this while loop in the case where we consumed
            // some newlines
            break;
        }
        if (isUtf8LeadingByte((Byte)c))
        {
            // Consume all unicode characters.
            pos--;
            c = getUnicodePointFromUTF8([&]() { return lexer->m_cursor[pos++]; });
        }
        // Default case is to just hand along the byte we read as an ASCII code point.
    } while (offset--);

    return c;
}

// Get the next code point from the input, and advance the cursor.
static int _advance(Lexer* lexer)
{
    // We are going to loop, but only as a way of handling
    // escaped line endings.
    for (;;)
    {
        // If we are at the end of the input, then the task is easy.
        if (lexer->m_cursor == lexer->m_end)
            return kEOF;

        // Look at the next raw byte, and decide what to do
        int c = *lexer->m_cursor++;

        if (c == '\\')
        {
            // We might have a backslash-escaped newline.
            // Look at the next byte (if any) to see.
            //
            // Note(tfoley): We are assuming a null-terminated input here,
            // so that we can safely look at the next byte without issue.
            int d = *lexer->m_cursor;
            switch (d)
            {
            case '\r':
            case '\n':
                // handle the end-of-line for our source location tracking
                lexer->m_cursor++;
                _handleNewLineInner(lexer, d);

                lexer->m_tokenFlags |= TokenFlag::ScrubbingNeeded;

                // Now try again, looking at the character after the
                // escaped newline.
                continue;

            default:
                break;
            }
        }

        // Consume all unicode characters.
        if (isUtf8LeadingByte((Byte)c))
        {
            lexer->m_cursor--;
            c = getUnicodePointFromUTF8([&]() { return *lexer->m_cursor++; });
        }

        // Default case is to return the raw byte we saw.
        return c;
    }
}

static void _handleNewLine(Lexer* lexer)
{
    int c = _advance(lexer);
    _handleNewLineInner(lexer, c);
}

static void _lexLineComment(Lexer* lexer)
{
    for (;;)
    {
        switch (_peek(lexer))
        {
        case '\n':
        case '\r':
        case kEOF:
            return;

        default:
            _advance(lexer);
            continue;
        }
    }
}

static void _lexBlockComment(Lexer* lexer)
{
    for (;;)
    {
        switch (_peek(lexer))
        {
        case kEOF:
            // TODO(tfoley) diagnostic!
            return;

        case '\n':
        case '\r':
            _handleNewLine(lexer);
            continue;

        case '*':
            _advance(lexer);
            switch (_peek(lexer))
            {
            case '/':
                _advance(lexer);
                return;

            default:
                continue;
            }

        default:
            _advance(lexer);
            continue;
        }
    }
}

static void _lexHorizontalSpace(Lexer* lexer)
{
    for (;;)
    {
        switch (_peek(lexer))
        {
        case ' ':
        case '\t':
            _advance(lexer);
            continue;

        default:
            return;
        }
    }
}

static bool isNonAsciiCodePoint(unsigned int codePoint)
{
    return codePoint != 0xFFFFFFFF && codePoint >= 0x80;
}

static void _lexIdentifier(Lexer* lexer)
{
    for (;;)
    {
        int c = _peek(lexer);
        if (('a' <= c) && (c <= 'z') || ('A' <= c) && (c <= 'Z') || ('0' <= c) && (c <= '9') ||
            (c == '_') || isNonAsciiCodePoint((unsigned int)c))
        {
            _advance(lexer);
            continue;
        }
        return;
    }
}

static SourceLoc _getSourceLoc(const Lexer& lexer, const char* it)
{
    return lexer.m_startLoc + (it - lexer.m_begin);
}

static SourceLoc _getSourceLoc(const Lexer* lexer)
{
    return _getSourceLoc(*lexer, lexer->m_cursor);
}

static void _lexDigits(Lexer* lexer, int base)
{
    for (;;)
    {
        int c = _peek(lexer);

        int digitVal = 0;
        switch (c)
        {
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
            digitVal = c - '0';
            break;

        case 'a':
        case 'b':
        case 'c':
        case 'd':
        case 'e':
        case 'f':
            if (base <= 10)
                return;
            digitVal = 10 + c - 'a';
            break;

        case 'A':
        case 'B':
        case 'C':
        case 'D':
        case 'E':
        case 'F':
            if (base <= 10)
                return;
            digitVal = 10 + c - 'A';
            break;

        default:
            // Not more digits!
            return;
        }

        if (digitVal >= base)
        {
            if (auto sink = lexer->getDiagnosticSink())
            {
                char buffer[] = {(char)c, 0};
                sink->diagnose(
                    _getSourceLoc(lexer),
                    LexerDiagnostics::invalidDigitForBase,
                    buffer,
                    base);
            }
        }

        _advance(lexer);
    }
}

static TokenType _maybeLexNumberSuffix(Lexer* lexer, TokenType tokenType)
{
    // Be liberal in what we accept here, so that figuring out
    // the semantics of a numeric suffix is left up to the parser
    // and semantic checking logic.
    //
    for (;;)
    {
        int c = _peek(lexer);

        // Accept any alphanumeric character, plus underscores.
        if (('a' <= c) && (c <= 'z') || ('A' <= c) && (c <= 'Z') || ('0' <= c) && (c <= '9') ||
            (c == '_'))
        {
            _advance(lexer);
            continue;
        }

        // Stop at the first character that isn't
        // alphanumeric.
        return tokenType;
    }
}

static bool _isNumberExponent(int c, int base)
{
    switch (c)
    {
    default:
        return false;

    case 'e':
    case 'E':
        if (base != 10)
            return false;
        break;

    case 'p':
    case 'P':
        if (base != 16)
            return false;
        break;
    }

    return true;
}

static bool _maybeLexNumberExponent(Lexer* lexer, int base)
{
    if (_peek(lexer) == '#')
    {
        // Special case #INF
        const auto inf = toSlice("#INF");
        for (auto c : inf)
        {
            if (_peek(lexer) != c)
            {
                return false;
            }
            _advance(lexer);
        }

        return true;
    }

    if (!_isNumberExponent(_peek(lexer), base))
        return false;

    // we saw an exponent marker
    _advance(lexer);

    // Now start to read the exponent
    switch (_peek(lexer))
    {
    case '+':
    case '-':
        _advance(lexer);
        break;
    }

    // TODO(tfoley): it would be an error to not see digits here...

    _lexDigits(lexer, 10);

    return true;
}

static TokenType _lexNumberAfterDecimalPoint(Lexer* lexer, int base)
{
    _lexDigits(lexer, base);
    _maybeLexNumberExponent(lexer, base);

    return _maybeLexNumberSuffix(lexer, TokenType::FloatingPointLiteral);
}

static TokenType _lexNumber(Lexer* lexer, int base)
{
    // TODO(tfoley): Need to consider whether to allow any kind of digit separator character.

    TokenType tokenType = TokenType::IntegerLiteral;

    // At the start of things, we just concern ourselves with digits
    _lexDigits(lexer, base);

    if (_peek(lexer) == '.')
    {
        switch (_peek(lexer, 1))
        {
            // 123.xxxx or 123.rrrr
        case 'x':
        case 'r':
            break;

        default:
            tokenType = TokenType::FloatingPointLiteral;

            _advance(lexer);
            _lexDigits(lexer, base);
        }
    }

    if (_maybeLexNumberExponent(lexer, base))
    {
        tokenType = TokenType::FloatingPointLiteral;
    }

    _maybeLexNumberSuffix(lexer, tokenType);
    return tokenType;
}

static int _maybeReadDigit(char const** ioCursor, int base)
{
    auto& cursor = *ioCursor;

    for (;;)
    {
        int c = *cursor;
        switch (c)
        {
        default:
            return -1;

        // TODO: need to decide on digit separator characters
        case '_':
            cursor++;
            continue;

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
            cursor++;
            return c - '0';

        case 'a':
        case 'b':
        case 'c':
        case 'd':
        case 'e':
        case 'f':
            if (base > 10)
            {
                cursor++;
                return 10 + c - 'a';
            }
            return -1;

        case 'A':
        case 'B':
        case 'C':
        case 'D':
        case 'E':
        case 'F':
            if (base > 10)
            {
                cursor++;
                return 10 + c - 'A';
            }
            return -1;
        }
    }
}

static int _readOptionalBase(char const** ioCursor)
{
    auto& cursor = *ioCursor;
    if (*cursor == '0')
    {
        cursor++;
        switch (*cursor)
        {
        case 'x':
        case 'X':
            cursor++;
            return 16;

        case 'b':
        case 'B':
            cursor++;
            return 2;

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
            return 8;

        default:
            return 10;
        }
    }

    return 10;
}


IntegerLiteralValue getIntegerLiteralValue(
    Token const& token,
    UnownedStringSlice* outSuffix,
    bool* outIsDecimalBase)
{
    IntegerLiteralValue value = 0;

    const UnownedStringSlice content = token.getContent();

    char const* cursor = content.begin();
    char const* end = content.end();

    int base = _readOptionalBase(&cursor);

    for (;;)
    {
        int digit = _maybeReadDigit(&cursor, base);
        if (digit < 0)
            break;

        value = value * base + digit;
    }

    if (outSuffix)
    {
        *outSuffix = UnownedStringSlice(cursor, end);
    }

    if (outIsDecimalBase)
    {
        *outIsDecimalBase = (base == 10);
    }

    return value;
}

FloatingPointLiteralValue getFloatingPointLiteralValue(
    Token const& token,
    UnownedStringSlice* outSuffix)
{
    FloatingPointLiteralValue value = 0;

    const UnownedStringSlice content = token.getContent();

    char const* cursor = content.begin();
    char const* end = content.end();

    int radix = _readOptionalBase(&cursor);

    bool seenDot = false;
    FloatingPointLiteralValue divisor = 1;
    for (;;)
    {
        if (*cursor == '.')
        {
            cursor++;
            seenDot = true;
            continue;
        }

        int digit = _maybeReadDigit(&cursor, radix);
        if (digit < 0)
            break;

        value = value * radix + digit;

        if (seenDot)
        {
            divisor *= radix;
        }
    }

    if (*cursor == '#')
    {
        // It must be INF
        const auto inf = toSlice("#INF");

        if (UnownedStringSlice(cursor, end).startsWith(inf))
        {
            if (outSuffix)
            {
                *outSuffix = UnownedStringSlice(cursor + inf.getLength(), end);
            }

            value = INFINITY;

            return value;
        }
    }

    // Now read optional exponent
    if (_isNumberExponent(*cursor, radix))
    {
        cursor++;

        bool exponentIsNegative = false;
        switch (*cursor)
        {
        default:
            break;

        case '-':
            exponentIsNegative = true;
            cursor++;
            break;

        case '+':
            cursor++;
            break;
        }

        int exponentRadix = 10;
        int exponent = 0;

        for (;;)
        {
            int digit = _maybeReadDigit(&cursor, exponentRadix);
            if (digit < 0)
                break;

            exponent = exponent * exponentRadix + digit;
        }

        FloatingPointLiteralValue exponentBase = 10;
        if (radix == 16)
        {
            exponentBase = 2;
        }

        FloatingPointLiteralValue exponentValue = pow(exponentBase, exponent);

        if (exponentIsNegative)
        {
            divisor *= exponentValue;
        }
        else
        {
            value *= exponentValue;
        }
    }

    value /= divisor;

    if (outSuffix)
    {
        *outSuffix = UnownedStringSlice(cursor, end);
    }

    return value;
}

IntegerLiteralValue getCharLiteralValue(Token const& token)
{
    String unquotedContent = StringEscapeUtil::unquote('\'', token.getContent());
    StringBuilder unescaped(4);
    auto escapeHandler = StringEscapeUtil::getHandler(StringEscapeUtil::Style::Cpp);
    escapeHandler->appendUnescaped(unquotedContent.getUnownedSlice(), unescaped);

    char const* cursor = unescaped.getBuffer();

    IntegerLiteralValue codepoint = getUnicodePointFromUTF8([&]() { return *cursor++; });
    return codepoint;
}

static void _lexStringLiteralBody(Lexer* lexer, char quote, bool singleChar)
{
    int len = 0;
    for (;;)
    {
        int c = _peek(lexer);
        if (c == quote)
        {
            if (singleChar && len == 0)
            { // Empty char literal - size must be exactly 1.
                if (auto sink = lexer->getDiagnosticSink())
                {
                    sink->diagnose(_getSourceLoc(lexer), LexerDiagnostics::illegalCharacterLiteral);
                }
            }
            _advance(lexer);
            return;
        }

        len++;

        if (singleChar && len == 2)
        { // Char literal about to have more than 1 char.
            if (auto sink = lexer->getDiagnosticSink())
            {
                sink->diagnose(_getSourceLoc(lexer), LexerDiagnostics::illegalCharacterLiteral);
            }
        }

        switch (c)
        {
        case kEOF:
            if (auto sink = lexer->getDiagnosticSink())
            {
                sink->diagnose(_getSourceLoc(lexer), LexerDiagnostics::endOfFileInLiteral);
            }
            return;

        case '\n':
        case '\r':
            if (auto sink = lexer->getDiagnosticSink())
            {
                sink->diagnose(_getSourceLoc(lexer), LexerDiagnostics::newlineInLiteral);
            }
            return;

        case '\\':
            // Need to handle various escape sequence cases
            _advance(lexer);
            switch (_peek(lexer))
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
                _advance(lexer);
                break;

            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
                // octal escape: up to 3 characters
                _advance(lexer);
                for (int ii = 0; ii < 3; ++ii)
                {
                    int d = _peek(lexer);
                    if (('0' <= d) && (d <= '7'))
                    {
                        _advance(lexer);
                        continue;
                    }
                    else
                    {
                        break;
                    }
                }
                break;

            case 'x':
                // hexadecimal escape: any number of characters
                _advance(lexer);
                for (;;)
                {
                    int d = _peek(lexer);
                    if (('0' <= d) && (d <= '9') || ('a' <= d) && (d <= 'f') ||
                        ('A' <= d) && (d <= 'F'))
                    {
                        _advance(lexer);
                        continue;
                    }
                    else
                    {
                        break;
                    }
                }
                break;

                // TODO: Unicode escape sequences
            }
            break;

        default:
            _advance(lexer);
            continue;
        }
    }
}

static void _lexRawStringLiteralBody(Lexer* lexer)
{
    const char* start = lexer->m_cursor;
    const char* endOfDelimiter = nullptr;
    for (;;)
    {
        int c = _peek(lexer);
        if (c == '(' && endOfDelimiter == nullptr)
            endOfDelimiter = lexer->m_cursor;
        if (c == '\"')
        {
            if (!endOfDelimiter)
            {
                if (auto sink = lexer->getDiagnosticSink())
                {
                    sink->diagnose(_getSourceLoc(lexer), LexerDiagnostics::quoteCannotBeDelimiter);
                }
            }
            else
            {
                auto testStart = lexer->m_cursor - (endOfDelimiter - start);
                if (testStart > endOfDelimiter)
                {
                    auto testDelimiter = UnownedStringSlice(testStart, lexer->m_cursor);
                    auto delimiter = UnownedStringSlice(start, endOfDelimiter);
                    if (*(testStart - 1) == ')' && testDelimiter == delimiter)
                    {
                        _advance(lexer);
                        return;
                    }
                }
            }
        }

        switch (c)
        {
        case kEOF:
            if (auto sink = lexer->getDiagnosticSink())
            {
                sink->diagnose(_getSourceLoc(lexer), LexerDiagnostics::endOfFileInLiteral);
            }
            return;
        default:
            _advance(lexer);
            continue;
        }
    }
}

UnownedStringSlice getRawStringLiteralTokenValue(Token const& token)
{
    auto content = token.getContent();
    if (content.getLength() <= 5)
        return UnownedStringSlice();
    auto start = content.begin() + 2;
    auto delimEnd = start;
    while (delimEnd < content.end() && *delimEnd != '(')
        delimEnd++;
    auto delimLength = delimEnd - start;
    auto contentEnd = content.end() - delimLength - 2;
    auto contentBegin = start + delimLength + 1;
    if (contentEnd <= contentBegin)
        return UnownedStringSlice();
    return UnownedStringSlice(contentBegin, contentEnd);
}

String getStringLiteralTokenValue(Token const& token)
{
    SLANG_ASSERT(token.type == TokenType::StringLiteral || token.type == TokenType::CharLiteral);

    if (token.getContent().startsWith("R"))
        return getRawStringLiteralTokenValue(token);

    const UnownedStringSlice content = token.getContent();

    char const* cursor = content.begin();
    char const* end = content.end();
    SLANG_UNREFERENCED_VARIABLE(end);

    auto quote = *cursor++;
    SLANG_ASSERT(quote == '\'' || quote == '"');

    StringBuilder valueBuilder;
    for (;;)
    {
        SLANG_ASSERT(cursor != end);

        auto c = *cursor++;

        // If we see a closing quote, then we are at the end of the string literal
        if (c == quote)
        {
            SLANG_ASSERT(cursor == end);
            return valueBuilder.produceString();
        }

        // Characters that don't being escape sequences are easy;
        // just append them to the buffer and move on.
        if (c != '\\')
        {
            valueBuilder.append(c);
            continue;
        }

        // Now we look at another character to figure out the kind of
        // escape sequence we are dealing with:

        char d = *cursor++;

        switch (d)
        {
        // Simple characters that just needed to be escaped
        case '\'':
        case '\"':
        case '\\':
        case '?':
            valueBuilder.append(d);
            continue;

        // Traditional escape sequences for special characters
        case 'a':
            valueBuilder.append('\a');
            continue;
        case 'b':
            valueBuilder.append('\b');
            continue;
        case 'f':
            valueBuilder.append('\f');
            continue;
        case 'n':
            valueBuilder.append('\n');
            continue;
        case 'r':
            valueBuilder.append('\r');
            continue;
        case 't':
            valueBuilder.append('\t');
            continue;
        case 'v':
            valueBuilder.append('\v');
            continue;

        // Octal escape: up to 3 characters
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
            {
                cursor--;
                int value = 0;
                for (int ii = 0; ii < 3; ++ii)
                {
                    d = *cursor;
                    if (('0' <= d) && (d <= '7'))
                    {
                        value = value * 8 + (d - '0');

                        cursor++;
                        continue;
                    }
                    else
                    {
                        break;
                    }
                }

                // TODO: add support for appending an arbitrary code point?
                valueBuilder.append((char)value);
            }
            continue;

        // Hexadecimal escape: any number of characters
        case 'x':
            {
                int value = 0;
                for (;;)
                {
                    d = *cursor++;
                    int digitValue = 0;
                    if (('0' <= d) && (d <= '9'))
                    {
                        digitValue = d - '0';
                    }
                    else if (('a' <= d) && (d <= 'f'))
                    {
                        digitValue = d - 'a';
                    }
                    else if (('A' <= d) && (d <= 'F'))
                    {
                        digitValue = d - 'A';
                    }
                    else
                    {
                        cursor--;
                        break;
                    }

                    value = value * 16 + digitValue;
                }

                // TODO: add support for appending an arbitrary code point?
                valueBuilder.append((char)value);
            }
            continue;

            // TODO: Unicode escape sequences
        }
    }
}

String getFileNameTokenValue(Token const& token)
{
    const UnownedStringSlice content = token.getContent();

    // A file name usually doesn't process escape sequences
    // (this is import on Windows, where `\\` is a valid
    // path separator character).

    // Just trim off the first and last characters to remove the quotes
    // (whether they were `""` or `<>`.
    return String(content.begin() + 1, content.end() - 1);
}

static TokenType _lexTokenImpl(Lexer* lexer)
{
    int nextCodePoint = _peek(lexer);
    switch (nextCodePoint)
    {
    default:
        break;

    case kEOF:
        return TokenType::EndOfFile;

    case '\r':
    case '\n':
        _handleNewLine(lexer);
        return TokenType::NewLine;

    case ' ':
    case '\t':
        _lexHorizontalSpace(lexer);
        return TokenType::WhiteSpace;

    case '.':
        _advance(lexer);
        switch (_peek(lexer))
        {
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
            return _lexNumberAfterDecimalPoint(lexer, 10);

        case '.':
            // Note: consuming the second `.` here means that
            // we cannot back up and return a `.` token by itself
            // any more. We thus end up having distinct tokens for
            // `.`, `..`, and `...` even though the `..` case is
            // not part of HLSL.
            //
            _advance(lexer);
            switch (_peek(lexer))
            {
            case '.':
                _advance(lexer);
                return TokenType::Ellipsis;

            default:
                return TokenType::DotDot;
            }

        default:
            return TokenType::Dot;
        }

    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
        return _lexNumber(lexer, 10);

    case '0':
        {
            auto loc = _getSourceLoc(lexer);
            _advance(lexer);
            switch (_peek(lexer))
            {
            default:
                return _maybeLexNumberSuffix(lexer, TokenType::IntegerLiteral);

            case '.':
                switch (_peek(lexer, 1))
                {
                    // 0.xxxx or 0.rrrr
                case 'x':
                case 'r':
                    return _maybeLexNumberSuffix(lexer, TokenType::IntegerLiteral);
                default:
                    _advance(lexer);
                    return _lexNumberAfterDecimalPoint(lexer, 10);
                }

            case 'x':
            case 'X':
                _advance(lexer);
                return _lexNumber(lexer, 16);

            case 'b':
            case 'B':
                _advance(lexer);
                return _lexNumber(lexer, 2);

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
                if (auto sink = lexer->getDiagnosticSink())
                {
                    sink->diagnose(loc, LexerDiagnostics::octalLiteral);
                }
                return _lexNumber(lexer, 8);
            }
        }

    case 'a':
    case 'b':
    case 'c':
    case 'd':
    case 'e':
    case 'f':
    case 'g':
    case 'h':
    case 'i':
    case 'j':
    case 'k':
    case 'l':
    case 'm':
    case 'n':
    case 'o':
    case 'p':
    case 'q':
    case 'r':
    case 's':
    case 't':
    case 'u':
    case 'v':
    case 'w':
    case 'x':
    case 'y':
    case 'z':
    case 'A':
    case 'B':
    case 'C':
    case 'D':
    case 'E':
    case 'F':
    case 'G':
    case 'H':
    case 'I':
    case 'J':
    case 'K':
    case 'L':
    case 'M':
    case 'N':
    case 'O':
    case 'P':
    case 'Q':
    case 'S':
    case 'T':
    case 'U':
    case 'V':
    case 'W':
    case 'X':
    case 'Y':
    case 'Z':
    case '_':
        _lexIdentifier(lexer);
        return TokenType::Identifier;
    case 'R':
        _advance(lexer);
        switch (_peek(lexer))
        {
        default:
            _lexIdentifier(lexer);
            return TokenType::Identifier;
        case '\"':
            _advance(lexer);
            _lexRawStringLiteralBody(lexer);
            return TokenType::StringLiteral;
        }

    case '\"':
        _advance(lexer);
        _lexStringLiteralBody(lexer, '\"', false);
        return TokenType::StringLiteral;

    case '\'':
        _advance(lexer);
        _lexStringLiteralBody(lexer, '\'', true);
        return TokenType::CharLiteral;


    case '+':
        _advance(lexer);
        switch (_peek(lexer))
        {
        case '+':
            _advance(lexer);
            return TokenType::OpInc;
        case '=':
            _advance(lexer);
            return TokenType::OpAddAssign;
        default:
            return TokenType::OpAdd;
        }

    case '-':
        _advance(lexer);
        switch (_peek(lexer))
        {
        case '-':
            _advance(lexer);
            return TokenType::OpDec;
        case '=':
            _advance(lexer);
            return TokenType::OpSubAssign;
        case '>':
            _advance(lexer);
            return TokenType::RightArrow;
        default:
            return TokenType::OpSub;
        }

    case '*':
        _advance(lexer);
        switch (_peek(lexer))
        {
        case '=':
            _advance(lexer);
            return TokenType::OpMulAssign;
        default:
            return TokenType::OpMul;
        }

    case '/':
        _advance(lexer);
        switch (_peek(lexer))
        {
        case '=':
            _advance(lexer);
            return TokenType::OpDivAssign;
        case '/':
            _advance(lexer);
            _lexLineComment(lexer);
            return TokenType::LineComment;
        case '*':
            _advance(lexer);
            _lexBlockComment(lexer);
            return TokenType::BlockComment;
        default:
            return TokenType::OpDiv;
        }

    case '%':
        _advance(lexer);
        switch (_peek(lexer))
        {
        case '=':
            _advance(lexer);
            return TokenType::OpModAssign;
        default:
            return TokenType::OpMod;
        }

    case '|':
        _advance(lexer);
        switch (_peek(lexer))
        {
        case '|':
            _advance(lexer);
            return TokenType::OpOr;
        case '=':
            _advance(lexer);
            return TokenType::OpOrAssign;
        default:
            return TokenType::OpBitOr;
        }

    case '&':
        _advance(lexer);
        switch (_peek(lexer))
        {
        case '&':
            _advance(lexer);
            return TokenType::OpAnd;
        case '=':
            _advance(lexer);
            return TokenType::OpAndAssign;
        default:
            return TokenType::OpBitAnd;
        }

    case '^':
        _advance(lexer);
        switch (_peek(lexer))
        {
        case '=':
            _advance(lexer);
            return TokenType::OpXorAssign;
        default:
            return TokenType::OpBitXor;
        }

    case '>':
        _advance(lexer);
        switch (_peek(lexer))
        {
        case '>':
            _advance(lexer);
            switch (_peek(lexer))
            {
            case '=':
                _advance(lexer);
                return TokenType::OpShrAssign;
            default:
                return TokenType::OpRsh;
            }
        case '=':
            _advance(lexer);
            return TokenType::OpGeq;
        default:
            return TokenType::OpGreater;
        }

    case '<':
        _advance(lexer);
        switch (_peek(lexer))
        {
        case '<':
            _advance(lexer);
            switch (_peek(lexer))
            {
            case '=':
                _advance(lexer);
                return TokenType::OpShlAssign;
            default:
                return TokenType::OpLsh;
            }
        case '=':
            _advance(lexer);
            return TokenType::OpLeq;
        default:
            return TokenType::OpLess;
        }

    case '=':
        _advance(lexer);
        switch (_peek(lexer))
        {
        case '=':
            _advance(lexer);
            return TokenType::OpEql;
        case '>':
            _advance(lexer);
            return TokenType::DoubleRightArrow;
        default:
            return TokenType::OpAssign;
        }

    case '!':
        _advance(lexer);
        switch (_peek(lexer))
        {
        case '=':
            _advance(lexer);
            return TokenType::OpNeq;
        default:
            return TokenType::OpNot;
        }

    case '#':
        _advance(lexer);
        switch (_peek(lexer))
        {
        case '#':
            _advance(lexer);
            return TokenType::PoundPound;

        case '?':
            _advance(lexer);
            return TokenType::CompletionRequest;

        default:
            return TokenType::Pound;
        }

    case '~':
        _advance(lexer);
        return TokenType::OpBitNot;

    case ':':
        {
            _advance(lexer);
            if (_peek(lexer) == ':')
            {
                _advance(lexer);
                return TokenType::Scope;
            }
            return TokenType::Colon;
        }
    case ';':
        _advance(lexer);
        return TokenType::Semicolon;
    case ',':
        _advance(lexer);
        return TokenType::Comma;

    case '{':
        _advance(lexer);
        return TokenType::LBrace;
    case '}':
        _advance(lexer);
        return TokenType::RBrace;
    case '[':
        _advance(lexer);
        return TokenType::LBracket;
    case ']':
        _advance(lexer);
        return TokenType::RBracket;
    case '(':
        _advance(lexer);
        return TokenType::LParent;
    case ')':
        _advance(lexer);
        return TokenType::RParent;

    case '?':
        _advance(lexer);
        return TokenType::QuestionMark;
    case '@':
        _advance(lexer);
        return TokenType::At;
    case '$':
        {
            _advance(lexer);
            if (_peek(lexer) == '$')
            {
                _advance(lexer);
                return TokenType::DollarDollar;
            }
            return TokenType::Dollar;
        }
    }

    // We treat all unicode characters as a part of an identifier.
    if (isNonAsciiCodePoint(nextCodePoint))
    {
        _lexIdentifier(lexer);
        return TokenType::Identifier;
    }

    {
        // If none of the above cases matched, then we have an
        // unexpected/invalid character.

        auto loc = _getSourceLoc(lexer);
        int c = _advance(lexer);

        if (auto sink = lexer->getDiagnosticSink())
        {
            if (c >= 0x20 && c <= 0x7E)
            {
                char buffer[] = {(char)c, 0};
                sink->diagnose(loc, LexerDiagnostics::illegalCharacterPrint, buffer);
            }
            else if (c == kEOF)
            {
                sink->diagnose(loc, LexerDiagnostics::unexpectedEndOfInput);
            }
            else
            {
                // Fallback: print as hexadecimal
                sink->diagnose(
                    loc,
                    LexerDiagnostics::illegalCharacterHex,
                    String((unsigned char)c, 16));
            }
        }

        return TokenType::Invalid;
    }
}

Token Lexer::lexToken()
{
    for (;;)
    {
        Token token;
        token.loc = _getSourceLoc(this);

        char const* textBegin = m_cursor;

        auto tokenType = _lexTokenImpl(this);

        // The flags on the token we just lexed will be based
        // on the current state of the lexer.
        //
        auto tokenFlags = m_tokenFlags;
        //
        // Depending on what kind of token we just lexed, the
        // flags that will be used for the *next* token might
        // need to be updated.
        //
        switch (tokenType)
        {
        case TokenType::NewLine:
            {
                // If we just reached the end of a line, then the next token
                // should count as being at the start of a line, and also after
                // whitespace.
                //
                m_tokenFlags = TokenFlag::AtStartOfLine | TokenFlag::AfterWhitespace;
                break;
            }

        case TokenType::WhiteSpace:
        case TokenType::BlockComment:
        case TokenType::LineComment:
            {
                // True horizontal whitespace and comments both count as whitespace.
                //
                // Note that a line comment does not include the terminating newline,
                // we do not need to set `AtStartOfLine` here.
                //
                m_tokenFlags |= TokenFlag::AfterWhitespace;
                break;
            }

        default:
            {
                // If we read some token other then the above cases, then we are
                // neither after whitespace nor at the start of a line.
                //
                m_tokenFlags = 0;
                break;
            }
        }

        token.type = tokenType;
        token.flags = tokenFlags;

        char const* textEnd = m_cursor;

        // Note(tfoley): `StringBuilder::Append()` seems to crash when appending zero bytes
        if (textEnd != textBegin)
        {
            // "scrubbing" token value here to remove escaped newlines...
            //
            // Only perform this work if we encountered an escaped newline
            // while lexing this token (e.g., keep a flag on the lexer), or
            // do it on-demand when the actual value of the token is needed.
            if (tokenFlags & TokenFlag::ScrubbingNeeded)
            {
                // Allocate space that will always be more than enough for stripped contents
                char* startDst = (char*)m_memoryArena->allocateUnaligned(textEnd - textBegin);
                char* dst = startDst;

                auto tt = textBegin;
                while (tt != textEnd)
                {
                    char c = *tt++;
                    if (c == '\\')
                    {
                        char d = *tt;
                        switch (d)
                        {
                        case '\r':
                        case '\n':
                            {
                                tt++;
                                char e = *tt;
                                if ((d ^ e) == ('\r' ^ '\n'))
                                {
                                    tt++;
                                }
                            }
                            continue;

                        default:
                            break;
                        }
                    }
                    *dst++ = c;
                }
                token.setContent(UnownedStringSlice(startDst, dst));
            }
            else
            {
                token.setContent(UnownedStringSlice(textBegin, textEnd));
            }
        }

        if (m_namePool)
        {
            if (tokenType == TokenType::Identifier || tokenType == TokenType::CompletionRequest)
            {
                token.setName(m_namePool->getName(token.getContent()));
            }
        }

        return token;
    }
}

TokenList Lexer::lexAllSemanticTokens()
{
    TokenList tokenList;
    for (;;)
    {
        Token token = lexToken();

        // We are only interested intokens that are semantically
        // significant, so we will skip over forms of whitespace
        // and comments.
        //
        switch (token.type)
        {
        default:
            break;

        case TokenType::WhiteSpace:
        case TokenType::BlockComment:
        case TokenType::LineComment:
        case TokenType::NewLine:
            continue;
        }

        tokenList.add(token);
        if (token.type == TokenType::EndOfFile)
            return tokenList;
    }
}

TokenList Lexer::lexAllMarkupTokens()
{
    TokenList tokenList;
    for (;;)
    {
        Token token = lexToken();
        switch (token.type)
        {
        default:
            break;

        case TokenType::WhiteSpace:
        case TokenType::NewLine:
            continue;
        }

        tokenList.add(token);
        if (token.type == TokenType::EndOfFile)
            return tokenList;
    }
}

TokenList Lexer::lexAllTokens()
{
    TokenList tokenList;
    for (;;)
    {
        Token token = lexToken();
        tokenList.add(token);
        if (token.type == TokenType::EndOfFile)
            return tokenList;
    }
}

/* static */ UnownedStringSlice Lexer::sourceLocationLexer(const UnownedStringSlice& in)
{
    Lexer lexer;

    SourceManager sourceManager;
    sourceManager.initialize(nullptr, nullptr);

    auto sourceFile = sourceManager.createSourceFileWithString(PathInfo::makeUnknown(), in);
    auto sourceView = sourceManager.createSourceView(sourceFile, nullptr, SourceLoc::fromRaw(0));

    DiagnosticSink sink(&sourceManager, nullptr);

    MemoryArena arena;

    RootNamePool rootNamePool;
    NamePool namePool;
    namePool.setRootNamePool(&rootNamePool);

    lexer.initialize(sourceView, &sink, &namePool, &arena);

    Token tok = lexer.lexToken();

    if (tok.type == TokenType::Invalid)
    {
        return UnownedStringSlice();
    }

    const int offset = sourceView->getRange().getOffset(tok.loc);

    SLANG_ASSERT(offset >= 0 && offset <= in.getLength());
    SLANG_ASSERT(Index(offset + tok.charsCount) <= in.getLength());

    return UnownedStringSlice(in.begin() + offset, in.begin() + offset + tok.charsCount);
}

SourceLoc Lexer::findNextLineEnd(SourceLoc from, UInt& lineCount) const
{
    const char* it = m_begin + (from.getRaw() - m_startLoc.getRaw());
    if (it >= m_begin && it < m_end)
    {
        while (it != m_end)
        {
            const char c = *it;
            if (c == '\n' || c == '\r')
            {
                const char next = ((it + 1) == m_end) ? char(kEOF) : *(it + 1);
                if ((next ^ c) == ('\n' ^ '\r'))
                {
                    ++it;
                }
                --lineCount;
                if (lineCount == 0)
                {
                    SourceLoc res = _getSourceLoc(*this, it);
                    return res;
                }
            }
            ++it;
        }
    }
    return {};
}

} // namespace Slang
