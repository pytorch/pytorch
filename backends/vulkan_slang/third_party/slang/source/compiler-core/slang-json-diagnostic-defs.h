//

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
// -1 - Notes that decorate another diagnostic.
//

//
// 2xxxx - JSON Lexical analysis
//

DIAGNOSTIC(20000, Error, unexpectedCharacter, "unexpected character '$0'")
DIAGNOSTIC(20001, Error, endOfFileInLiteral, "end of file in literal")
DIAGNOSTIC(20002, Error, newlineInLiteral, "newline in literal")
DIAGNOSTIC(20003, Error, endOfFileInComment, "end of file in comment")
DIAGNOSTIC(20004, Error, expectingAHexDigit, "expecting a hex digit")
DIAGNOSTIC(20005, Error, expectingADigit, "expecting a digit")
DIAGNOSTIC(20006, Error, expectingValueName, "expecting value name [null, true, false]")
DIAGNOSTIC(20007, Error, unexpectedTokenExpectedTokenType, "unexpected '$0', expected '$1'")
DIAGNOSTIC(20008, Error, unexpectedToken, "unexpected '$0'")

DIAGNOSTIC(20009, Error, unableToConvertField, "unable to convert field '$0' in type '$1'")
DIAGNOSTIC(20010, Error, fieldNotFound, "field '$0' not found in type '$1'")
DIAGNOSTIC(20011, Error, fieldNotDefinedOnType, "field '$0' not defined on type '$1'")
DIAGNOSTIC(20011, Error, fieldRequiredOnType, "field '$0' required on '$1'")
DIAGNOSTIC(
    20012,
    Error,
    tooManyElementsForArray,
    "too many elements ($0) for array array. Max allowed is $1")

//
// 3xxxx JSON-RPC
//

DIAGNOSTIC(30000, Error, argsAreInvalid, "Args for '%0' are invalid")


#undef DIAGNOSTIC
