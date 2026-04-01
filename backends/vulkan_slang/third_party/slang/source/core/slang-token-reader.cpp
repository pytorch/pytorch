#include "slang-token-reader.h"

namespace Slang
{
namespace Misc
{

enum class TokenizeErrorType
{
    InvalidCharacter,
    InvalidEscapeSequence
};

enum class State
{
    Start,
    Identifier,
    Operator,
    Int,
    Hex,
    Fixed,
    Double,
    Char,
    String,
    MultiComment,
    SingleComment
};

enum class LexDerivative
{
    None,
    Line,
    File
};

inline bool IsLetter(char ch)
{
    return ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_');
}

inline bool IsDigit(char ch)
{
    return ch >= '0' && ch <= '9';
}

inline bool IsPunctuation(char ch)
{
    return ch == '+' || ch == '-' || ch == '*' || ch == '/' || ch == '%' || ch == '!' ||
           ch == '^' || ch == '&' || ch == '(' || ch == ')' || ch == '=' || ch == '{' ||
           ch == '}' || ch == '[' || ch == ']' || ch == '|' || ch == ';' || ch == ',' ||
           ch == '.' || ch == '<' || ch == '>' || ch == '~' || ch == '@' || ch == ':' ||
           ch == '?' || ch == '#';
}

inline bool IsWhiteSpace(char ch)
{
    return (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\v');
}

void ParseOperators(
    const String& str,
    List<Token>& tokens,
    TokenFlags& tokenFlags,
    int line,
    int col,
    int startPos,
    String fileName)
{
    Index pos = 0;
    while (pos < str.getLength())
    {
        wchar_t curChar = str[pos];
        wchar_t nextChar = (pos < str.getLength() - 1) ? str[pos + 1] : '\0';
        wchar_t nextNextChar = (pos < str.getLength() - 2) ? str[pos + 2] : '\0';
        auto InsertToken = [&](TokenType type, const String& ct)
        {
            tokens.add(
                Token(type, ct, line, int(col + pos), int(pos + startPos), fileName, tokenFlags));
            tokenFlags = 0;
        };
        switch (curChar)
        {
        case '+':
            if (nextChar == '+')
            {
                InsertToken(TokenType::OpInc, "++");
                pos += 2;
            }
            else if (nextChar == '=')
            {
                InsertToken(TokenType::OpAddAssign, "+=");
                pos += 2;
            }
            else
            {
                InsertToken(TokenType::OpAdd, "+");
                pos++;
            }
            break;
        case '-':
            if (nextChar == '-')
            {
                InsertToken(TokenType::OpDec, "--");
                pos += 2;
            }
            else if (nextChar == '=')
            {
                InsertToken(TokenType::OpSubAssign, "-=");
                pos += 2;
            }
            else if (nextChar == '>')
            {
                InsertToken(TokenType::RightArrow, "->");
                pos += 2;
            }
            else
            {
                InsertToken(TokenType::OpSub, "-");
                pos++;
            }
            break;
        case '*':
            if (nextChar == '=')
            {
                InsertToken(TokenType::OpMulAssign, "*=");
                pos += 2;
            }
            else
            {
                InsertToken(TokenType::OpMul, "*");
                pos++;
            }
            break;
        case '/':
            if (nextChar == '=')
            {
                InsertToken(TokenType::OpDivAssign, "/=");
                pos += 2;
            }
            else
            {
                InsertToken(TokenType::OpDiv, "/");
                pos++;
            }
            break;
        case '%':
            if (nextChar == '=')
            {
                InsertToken(TokenType::OpModAssign, "%=");
                pos += 2;
            }
            else
            {
                InsertToken(TokenType::OpMod, "%");
                pos++;
            }
            break;
        case '|':
            if (nextChar == '|')
            {
                InsertToken(TokenType::OpOr, "||");
                pos += 2;
            }
            else if (nextChar == '=')
            {
                InsertToken(TokenType::OpOrAssign, "|=");
                pos += 2;
            }
            else
            {
                InsertToken(TokenType::OpBitOr, "|");
                pos++;
            }
            break;
        case '&':
            if (nextChar == '&')
            {
                InsertToken(TokenType::OpAnd, "&&");
                pos += 2;
            }
            else if (nextChar == '=')
            {
                InsertToken(TokenType::OpAndAssign, "&=");
                pos += 2;
            }
            else
            {
                InsertToken(TokenType::OpBitAnd, "&");
                pos++;
            }
            break;
        case '^':
            if (nextChar == '=')
            {
                InsertToken(TokenType::OpXorAssign, "^=");
                pos += 2;
            }
            else
            {
                InsertToken(TokenType::OpBitXor, "^");
                pos++;
            }
            break;
        case '>':
            if (nextChar == '>')
            {
                if (nextNextChar == '=')
                {
                    InsertToken(TokenType::OpShrAssign, ">>=");
                    pos += 3;
                }
                else
                {
                    InsertToken(TokenType::OpRsh, ">>");
                    pos += 2;
                }
            }
            else if (nextChar == '=')
            {
                InsertToken(TokenType::OpGeq, ">=");
                pos += 2;
            }
            else
            {
                InsertToken(TokenType::OpGreater, ">");
                pos++;
            }
            break;
        case '<':
            if (nextChar == '<')
            {
                if (nextNextChar == '=')
                {
                    InsertToken(TokenType::OpShlAssign, "<<=");
                    pos += 3;
                }
                else
                {
                    InsertToken(TokenType::OpLsh, "<<");
                    pos += 2;
                }
            }
            else if (nextChar == '=')
            {
                InsertToken(TokenType::OpLeq, "<=");
                pos += 2;
            }
            else
            {
                InsertToken(TokenType::OpLess, "<");
                pos++;
            }
            break;
        case '=':
            if (nextChar == '=')
            {
                InsertToken(TokenType::OpEql, "==");
                pos += 2;
            }
            else
            {
                InsertToken(TokenType::OpAssign, "=");
                pos++;
            }
            break;
        case '!':
            if (nextChar == '=')
            {
                InsertToken(TokenType::OpNeq, "!=");
                pos += 2;
            }
            else
            {
                InsertToken(TokenType::OpNot, "!");
                pos++;
            }
            break;
        case '?':
            InsertToken(TokenType::QuestionMark, "?");
            pos++;
            break;
        case '@':
            InsertToken(TokenType::At, "@");
            pos++;
            break;
        case '#':
            if (nextChar == '#')
            {
                InsertToken(TokenType::PoundPound, "##");
                pos += 2;
            }
            else
            {
                InsertToken(TokenType::Pound, "#");
                pos++;
            }
            pos++;
            break;
        case ':':
            InsertToken(TokenType::Colon, ":");
            pos++;
            break;
        case '~':
            InsertToken(TokenType::OpBitNot, "~");
            pos++;
            break;
        case ';':
            InsertToken(TokenType::Semicolon, ";");
            pos++;
            break;
        case ',':
            InsertToken(TokenType::Comma, ",");
            pos++;
            break;
        case '.':
            InsertToken(TokenType::Dot, ".");
            pos++;
            break;
        case '{':
            InsertToken(TokenType::LBrace, "{");
            pos++;
            break;
        case '}':
            InsertToken(TokenType::RBrace, "}");
            pos++;
            break;
        case '[':
            InsertToken(TokenType::LBracket, "[");
            pos++;
            break;
        case ']':
            InsertToken(TokenType::RBracket, "]");
            pos++;
            break;
        case '(':
            InsertToken(TokenType::LParent, "(");
            pos++;
            break;
        case ')':
            InsertToken(TokenType::RParent, ")");
            pos++;
            break;
        }
    }
}

List<Token> TokenizeText(const String& fileName, const String& text)
{
    Index lastPos = 0, pos = 0;
    int line = 1, col = 0;
    String file = fileName;
    State state = State::Start;
    StringBuilder tokenBuilder;
    int tokenLine, tokenCol;
    List<Token> tokenList;
    LexDerivative derivative = LexDerivative::None;
    TokenFlags tokenFlags = TokenFlag::AtStartOfLine;
    auto InsertToken = [&](TokenType type)
    {
        derivative = LexDerivative::None;
        tokenList.add(
            Token(type, tokenBuilder.toString(), tokenLine, tokenCol, int(pos), file, tokenFlags));
        tokenFlags = 0;
        tokenBuilder.clear();
    };
    auto ProcessTransferChar = [&](char nextChar)
    {
        switch (nextChar)
        {
        case '\\':
        case '\"':
        case '\'':
            tokenBuilder.append(nextChar);
            break;
        case 't':
            tokenBuilder.append('\t');
            break;
        case 's':
            tokenBuilder.append(' ');
            break;
        case 'n':
            tokenBuilder.append('\n');
            break;
        case 'r':
            tokenBuilder.append('\r');
            break;
        case 'b':
            tokenBuilder.append('\b');
            break;
        }
    };
    while (pos <= text.getLength())
    {
        char curChar = (pos < text.getLength() ? text[pos] : ' ');
        char nextChar = (pos < text.getLength() - 1) ? text[pos + 1] : '\0';
        if (lastPos != pos)
        {
            if (curChar == '\n')
            {
                line++;
                col = 0;
            }
            else
                col++;
            lastPos = pos;
        }

        switch (state)
        {
        case State::Start:
            if (IsLetter(curChar))
            {
                state = State::Identifier;
                tokenLine = line;
                tokenCol = col;
            }
            else if (IsDigit(curChar))
            {
                state = State::Int;
                tokenLine = line;
                tokenCol = col;
            }
            else if (curChar == '\'')
            {
                state = State::Char;
                pos++;
                tokenLine = line;
                tokenCol = col;
            }
            else if (curChar == '"')
            {
                state = State::String;
                pos++;
                tokenLine = line;
                tokenCol = col;
            }
            else if (curChar == '\r' || curChar == '\n')
            {
                tokenFlags |= TokenFlag::AtStartOfLine | TokenFlag::AfterWhitespace;
                pos++;
            }
            else if (
                curChar == ' ' || curChar == '\t' || curChar == '\xC2' ||
                curChar == '\xA0') // -62/-96:non-break space
            {
                tokenFlags |= TokenFlag::AfterWhitespace;
                pos++;
            }
            else if (curChar == '/' && nextChar == '/')
            {
                state = State::SingleComment;
                pos += 2;
            }
            else if (curChar == '/' && nextChar == '*')
            {
                pos += 2;
                state = State::MultiComment;
            }
            else if (curChar == '.' && IsDigit(nextChar))
            {
                tokenBuilder.append("0.");
                state = State::Fixed;
                pos++;
            }
            else if (IsPunctuation(curChar))
            {
                state = State::Operator;
                tokenLine = line;
                tokenCol = col;
            }
            else
            {
                pos++;
            }
            break;
        case State::Identifier:
            if (IsLetter(curChar) || IsDigit(curChar))
            {
                tokenBuilder.append(curChar);
                pos++;
            }
            else
            {
                auto tokenStr = tokenBuilder.toString();
#if 0
                    if (tokenStr == "#line_reset#")
                    {
                        line = 0;
                        col = 0;
                        tokenBuilder.clear();
                    }
                    else if (tokenStr == "#line")
                    {
                        derivative = LexDerivative::Line;
                        tokenBuilder.clear();
                    }
                    else if (tokenStr == "#file")
                    {
                        derivative = LexDerivative::File;
                        tokenBuilder.clear();
                        line = 0;
                        col = 0;
                    }
                    else
#endif
                InsertToken(TokenType::Identifier);
                state = State::Start;
            }
            break;
        case State::Operator:
            if (IsPunctuation(curChar) &&
                !((curChar == '/' && nextChar == '/') || (curChar == '/' && nextChar == '*')))
            {
                tokenBuilder.append(curChar);
                pos++;
            }
            else
            {
                // do token analyze
                ParseOperators(
                    tokenBuilder.toString(),
                    tokenList,
                    tokenFlags,
                    tokenLine,
                    tokenCol,
                    (int)(pos - tokenBuilder.getLength()),
                    file);
                tokenBuilder.clear();
                state = State::Start;
            }
            break;
        case State::Int:
            if (IsDigit(curChar))
            {
                tokenBuilder.append(curChar);
                pos++;
            }
            else if (curChar == '.')
            {
                state = State::Fixed;
                tokenBuilder.append(curChar);
                pos++;
            }
            else if (curChar == 'e' || curChar == 'E')
            {
                state = State::Double;
                tokenBuilder.append(curChar);
                if (nextChar == '-' || nextChar == '+')
                {
                    tokenBuilder.append(nextChar);
                    pos++;
                }
                pos++;
            }
            else if (curChar == 'x')
            {
                state = State::Hex;
                tokenBuilder.append(curChar);
                pos++;
            }
            else if (curChar == 'u')
            {
                pos++;
                tokenBuilder.append(curChar);
                InsertToken(TokenType::IntLiteral);
                state = State::Start;
            }
            else
            {
                if (derivative == LexDerivative::Line)
                {
                    derivative = LexDerivative::None;
                    line = stringToInt(tokenBuilder.toString()) - 1;
                    col = 0;
                    tokenBuilder.clear();
                }
                else
                {
                    InsertToken(TokenType::IntLiteral);
                }
                state = State::Start;
            }
            break;
        case State::Hex:
            if (IsDigit(curChar) || (curChar >= 'a' && curChar <= 'f') ||
                (curChar >= 'A' && curChar <= 'F'))
            {
                tokenBuilder.append(curChar);
                pos++;
            }
            else
            {
                InsertToken(TokenType::IntLiteral);
                state = State::Start;
            }
            break;
        case State::Fixed:
            if (IsDigit(curChar))
            {
                tokenBuilder.append(curChar);
                pos++;
            }
            else if (curChar == 'e' || curChar == 'E')
            {
                state = State::Double;
                tokenBuilder.append(curChar);
                if (nextChar == '-' || nextChar == '+')
                {
                    tokenBuilder.append(nextChar);
                    pos++;
                }
                pos++;
            }
            else
            {
                if (curChar == 'f')
                    pos++;
                InsertToken(TokenType::DoubleLiteral);
                state = State::Start;
            }
            break;
        case State::Double:
            if (IsDigit(curChar))
            {
                tokenBuilder.append(curChar);
                pos++;
            }
            else
            {
                if (curChar == 'f')
                    pos++;
                InsertToken(TokenType::DoubleLiteral);
                state = State::Start;
            }
            break;
        case State::String:
            if (curChar != '"')
            {
                if (curChar == '\\')
                {
                    ProcessTransferChar(nextChar);
                    pos++;
                }
                else
                    tokenBuilder.append(curChar);
            }
            else
            {
                if (derivative == LexDerivative::File)
                {
                    derivative = LexDerivative::None;
                    file = tokenBuilder.toString();
                    tokenBuilder.clear();
                }
                else
                {
                    InsertToken(TokenType::StringLiteral);
                }
                state = State::Start;
            }
            pos++;
            break;
        case State::Char:
            if (curChar != '\'')
            {
                if (curChar == '\\')
                {
                    ProcessTransferChar(nextChar);
                    pos++;
                }
                else
                    tokenBuilder.append(curChar);
            }
            else
            {
                InsertToken(TokenType::CharLiteral);
                state = State::Start;
            }
            pos++;
            break;
        case State::SingleComment:
            if (curChar == '\n')
            {
                state = State::Start;
                tokenFlags |= TokenFlag::AtStartOfLine | TokenFlag::AfterWhitespace;
            }
            pos++;
            break;
        case State::MultiComment:
            if (curChar == '*' && nextChar == '/')
            {
                state = State::Start;
                tokenFlags |= TokenFlag::AfterWhitespace;
                pos += 2;
            }
            else
                pos++;
            break;
        }
    }
    return tokenList;
}
List<Token> TokenizeText(const String& text)
{
    return TokenizeText("", text);
}

TokenReader::TokenReader(String text)
{
    this->tokens = TokenizeText("", text);
    tokenPtr = 0;
}

} // namespace Misc
} // namespace Slang
