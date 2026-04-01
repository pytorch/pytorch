#ifndef SLANG_CORE_TOKEN_READER_H
#define SLANG_CORE_TOKEN_READER_H

#include "slang-basic.h"

namespace Slang
{
namespace Misc
{

/* NOTE! This TokenReader is NOT used by the main slang compiler !*/

enum class TokenType
{
    EndOfFile = -1,
    // illegal
    Unknown,
    // identifier
    Identifier,
    // constant
    IntLiteral,
    DoubleLiteral,
    StringLiteral,
    CharLiteral,
    // operators
    Semicolon,
    Comma,
    Dot,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    LParent,
    RParent,
    OpAssign,
    OpAdd,
    OpSub,
    OpMul,
    OpDiv,
    OpMod,
    OpNot,
    OpBitNot,
    OpLsh,
    OpRsh,
    OpEql,
    OpNeq,
    OpGreater,
    OpLess,
    OpGeq,
    OpLeq,
    OpAnd,
    OpOr,
    OpBitXor,
    OpBitAnd,
    OpBitOr,
    OpInc,
    OpDec,
    OpAddAssign,
    OpSubAssign,
    OpMulAssign,
    OpDivAssign,
    OpModAssign,
    OpShlAssign,
    OpShrAssign,
    OpOrAssign,
    OpAndAssign,
    OpXorAssign,

    QuestionMark,
    Colon,
    RightArrow,
    At,
    Pound,
    PoundPound,
    Scope,
};

class CodePosition
{
public:
    int Line = -1, Col = -1, Pos = -1;
    String FileName;
    String ToString()
    {
        StringBuilder sb(100);
        sb << FileName;
        if (Line != -1)
            sb << "(" << Line << ")";
        return sb.produceString();
    }
    CodePosition() = default;
    CodePosition(int line, int col, int pos, String fileName)
    {
        Line = line;
        Col = col;
        Pos = pos;
        this->FileName = fileName;
    }
    bool operator<(const CodePosition& pos) const
    {
        return FileName < pos.FileName || (FileName == pos.FileName && Line < pos.Line) ||
               (FileName == pos.FileName && Line == pos.Line && Col < pos.Col);
    }
    bool operator==(const CodePosition& pos) const
    {
        return FileName == pos.FileName && Line == pos.Line && Col == pos.Col;
    }
};

enum TokenFlag : unsigned int
{
    AtStartOfLine = 1 << 0,
    AfterWhitespace = 1 << 1,
};
typedef unsigned int TokenFlags;

class Token
{
public:
    TokenType Type = TokenType::Unknown;
    String Content;
    CodePosition Position;
    TokenFlags flags = 0;
    Token() = default;
    Token(
        TokenType type,
        const String& content,
        int line,
        int col,
        int pos,
        String fileName,
        TokenFlags flags = 0)
        : flags(flags)
    {
        Type = type;
        Content = content;
        Position = CodePosition(line, col, pos, fileName);
    }
};

class TextFormatException : public Exception
{
public:
    TextFormatException(String message)
        : Exception(message)
    {
    }
};

class TokenReader
{
private:
    bool legal;
    List<Token> tokens;
    int tokenPtr;

public:
    TokenReader(String text);
    int ReadInt()
    {
        auto token = ReadToken();
        bool neg = false;
        if (token.Content.getUnownedSlice().isChar('-'))
        {
            neg = true;
            token = ReadToken();
        }
        if (token.Type == TokenType::IntLiteral)
        {
            if (neg)
                return -stringToInt(token.Content);
            else
                return stringToInt(token.Content);
        }
        throw TextFormatException("Text parsing error: int expected.");
    }
    unsigned int ReadUInt()
    {
        auto token = ReadToken();
        if (token.Type == TokenType::IntLiteral)
        {
            return stringToUInt(token.Content);
        }
        throw TextFormatException("Text parsing error: int expected.");
    }
    double ReadDouble()
    {
        auto token = ReadToken();
        bool neg = false;
        if (token.Content.getUnownedSlice().isChar('-'))
        {
            neg = true;
            token = ReadToken();
        }
        if (token.Type == TokenType::DoubleLiteral || token.Type == TokenType::IntLiteral)
        {
            if (neg)
                return -stringToDouble(token.Content);
            else
                return stringToDouble(token.Content);
        }
        throw TextFormatException("Text parsing error: floating point value expected.");
    }
    float ReadFloat() { return (float)ReadDouble(); }
    String ReadWord()
    {
        auto token = ReadToken();
        if (token.Type == TokenType::Identifier)
        {
            return token.Content;
        }
        throw TextFormatException("Text parsing error: identifier expected.");
    }
    String Read(const char* expectedStr)
    {
        auto token = ReadToken();
        if (token.Content == expectedStr)
        {
            return token.Content;
        }
        throw TextFormatException("Text parsing error: \'" + String(expectedStr) + "\' expected.");
    }
    String Read(String expectedStr)
    {
        auto token = ReadToken();
        if (token.Content == expectedStr)
        {
            return token.Content;
        }
        throw TextFormatException("Text parsing error: \'" + expectedStr + "\' expected.");
    }
    bool Read(TokenType tokenType)
    {
        if (NextToken().Type == tokenType)
        {
            ReadToken();
            return true;
        }
        throw TextFormatException("Text parsing error: unexpected '" + NextToken().Content + "'.");
    }

    String ReadStringLiteral()
    {
        auto token = ReadToken();
        if (token.Type == TokenType::StringLiteral)
        {
            return token.Content;
        }
        throw TextFormatException("Text parsing error: string literal expected.");
    }
    void Back(int count) { tokenPtr -= count; }
    Token ReadMatchingToken(TokenType type)
    {
        auto token = ReadToken();
        if (token.Type != type)
        {
            throw TextFormatException("Text parsing error: unexpected token.");
        }
        return token;
    }
    Token ReadToken()
    {
        if (tokenPtr < (int)tokens.getCount())
        {
            auto& rs = tokens[tokenPtr];
            tokenPtr++;
            return rs;
        }
        throw TextFormatException("Unexpected ending.");
    }
    Token NextToken(int offset = 0)
    {
        if (tokenPtr + offset < (int)tokens.getCount())
            return tokens[tokenPtr + offset];
        else
        {
            Token rs;
            rs.Type = TokenType::Unknown;
            return rs;
        }
    }
    bool LookAhead(String token)
    {
        if (tokenPtr < (int)tokens.getCount())
        {
            auto next = NextToken();
            return next.Content == token;
        }
        else
        {
            return false;
        }
    }
    bool LookAhead(TokenType tokenType)
    {
        if (tokenPtr < (int)tokens.getCount())
        {
            auto next = NextToken();
            return next.Type == tokenType;
        }
        else
        {
            return false;
        }
    }
    bool AdvanceIf(String token)
    {
        if (LookAhead(token))
        {
            ReadToken();
            return true;
        }
        return false;
    }
    bool AdvanceIf(TokenType token)
    {
        if (LookAhead(token))
        {
            ReadToken();
            return true;
        }
        return false;
    }
    bool IsEnd() { return tokenPtr == (int)tokens.getCount(); }

public:
    bool IsLegalText() { return legal; }
};

inline List<String> Split(String text, char c)
{
    List<String> result;
    StringBuilder sb;
    for (Index i = 0; i < text.getLength(); i++)
    {
        if (text[i] == c)
        {
            auto str = sb.toString();
            if (str.getLength() != 0)
                result.add(str);
            sb.clear();
        }
        else
            sb << text[i];
    }
    auto lastStr = sb.toString();
    if (lastStr.getLength())
        result.add(lastStr);
    return result;
}

} // namespace Misc
} // namespace Slang


#endif
