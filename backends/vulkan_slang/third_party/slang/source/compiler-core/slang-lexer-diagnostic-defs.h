// The file is meant to be included multiple times, to produce different
// pieces of declaration/definition code related to diagnostic messages
//
// Each diagnostic is declared here with:
//
//     DIAGNOSTIC(id, severity, name, messageFormat)
//
// Where `id` is the unique diagnostic ID, `severity` is the default
// severity (from the `Severity` enum), `name` is a name used to refer
// to this diagnostic from code, and `messageFormat` is the default
// (non-localized) message for the diagnostic, with placeholders
// for any arguments.

#ifndef DIAGNOSTIC
#error Need to #define DIAGNOSTIC(...) before including
#define DIAGNOSTIC(id, severity, name, messageFormat) /* */
#endif

//
// 1xxxx - Lexical analysis
//

DIAGNOSTIC(10000, Error, illegalCharacterPrint, "illegal character '$0'")
DIAGNOSTIC(10000, Error, illegalCharacterHex, "illegal character (0x$0)")
DIAGNOSTIC(10001, Error, illegalCharacterLiteral, "illegal character literal")

DIAGNOSTIC(10002, Warning, octalLiteral, "'0' prefix indicates octal literal")
DIAGNOSTIC(10003, Error, invalidDigitForBase, "invalid digit for base-$1 literal: '$0'")

DIAGNOSTIC(10004, Error, endOfFileInLiteral, "end of file in literal")
DIAGNOSTIC(10005, Error, newlineInLiteral, "newline in literal")
DIAGNOSTIC(
    10010,
    Error,
    quoteCannotBeDelimiter,
    "'\"' encountered before '(' in raw string literal. '\"' cannot be a part of a delimiter.")

DIAGNOSTIC(10011, Error, unexpectedEndOfInput, "unexpected end of input")

#undef DIAGNOSTIC
