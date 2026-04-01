// slang-token-defs.h

// This file is meant to be included multiple times, to produce different
// pieces of code related to tokens
//
// Each token is declared here with:
//
//      TOKEN(id, desc)
//
// where `id` is the identifier that will be used for the token in
// ordinary code, while `desc` is name we should print when
// referring to this token in diagnostic messages.


#ifndef TOKEN
#error Need to define TOKEN(ID, DESC) before including "token-defs.h"
#endif

TOKEN(Unknown, "<unknown>")
TOKEN(EndOfFile, "end of file")
TOKEN(Invalid, "invalid character")
TOKEN(Identifier, "identifier")
TOKEN(IntegerLiteral, "integer literal")
TOKEN(FloatingPointLiteral, "floating-point literal")
TOKEN(StringLiteral, "string literal")
TOKEN(CharLiteral, "character literal")
TOKEN(WhiteSpace, "whitespace")
TOKEN(NewLine, "end of line")
TOKEN(LineComment, "line comment")
TOKEN(BlockComment, "block comment")

#define PUNCTUATION(id, text) TOKEN(id, "'" text "'")

PUNCTUATION(Semicolon, ";")
PUNCTUATION(Comma, ",")
PUNCTUATION(Dot, ".")
PUNCTUATION(DotDot, "..")
PUNCTUATION(Ellipsis, "...")

PUNCTUATION(LBrace, "{")
PUNCTUATION(RBrace, "}")
PUNCTUATION(LBracket, "[")
PUNCTUATION(RBracket, "]")
PUNCTUATION(LParent, "(")
PUNCTUATION(RParent, ")")

PUNCTUATION(OpAssign, "=")
PUNCTUATION(OpAdd, "+")
PUNCTUATION(OpSub, "-")
PUNCTUATION(OpMul, "*")
PUNCTUATION(OpDiv, "/")
PUNCTUATION(OpMod, "%")
PUNCTUATION(OpNot, "!")
PUNCTUATION(OpBitNot, "~")
PUNCTUATION(OpLsh, "<<")
PUNCTUATION(OpRsh, ">>")
PUNCTUATION(OpEql, "==")
PUNCTUATION(OpNeq, "!=")
PUNCTUATION(OpGreater, ">")
PUNCTUATION(OpLess, "<")
PUNCTUATION(OpGeq, ">=")
PUNCTUATION(OpLeq, "<=")
PUNCTUATION(OpAnd, "&&")
PUNCTUATION(OpOr, "||")
PUNCTUATION(OpBitAnd, "&")
PUNCTUATION(OpBitOr, "|")
PUNCTUATION(OpBitXor, "^")
PUNCTUATION(OpInc, "++")
PUNCTUATION(OpDec, "--")

PUNCTUATION(OpAddAssign, "+=")
PUNCTUATION(OpSubAssign, "-=")
PUNCTUATION(OpMulAssign, "*=")
PUNCTUATION(OpDivAssign, "/=")
PUNCTUATION(OpModAssign, "%=")
PUNCTUATION(OpShlAssign, "<<=")
PUNCTUATION(OpShrAssign, ">>=")
PUNCTUATION(OpAndAssign, "&=")
PUNCTUATION(OpOrAssign, "|=")
PUNCTUATION(OpXorAssign, "^=")

PUNCTUATION(QuestionMark, "?")
PUNCTUATION(Colon, ":")
PUNCTUATION(RightArrow, "->")
PUNCTUATION(DoubleRightArrow, "=>")
PUNCTUATION(At, "@")
PUNCTUATION(Dollar, "$")
PUNCTUATION(DollarDollar, "$$")
PUNCTUATION(Pound, "#")
PUNCTUATION(PoundPound, "##")

PUNCTUATION(Scope, "::")

PUNCTUATION(CompletionRequest, "#?")

#undef PUNCTUATION

// Un-define the `TOKEN` macro so that client doesn't have to
#undef TOKEN
