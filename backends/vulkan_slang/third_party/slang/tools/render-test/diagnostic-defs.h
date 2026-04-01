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


DIAGNOSTIC(
    1001,
    Error,
    expectingCommaComputeDispatch,
    "expected 3 comma separated integers for compute dispatch size")
DIAGNOSTIC(
    1002,
    Error,
    expectingPositiveComputeDispatch,
    "expected 3 comma positive integers for compute dispatch size")
DIAGNOSTIC(1003, Error, unknownSourceLanguage, "unknown source language name")
DIAGNOSTIC(1003, Error, unknown, "unknown source language name")
DIAGNOSTIC(1004, Error, unknownCommandLineOption, "unknown command-line option '$0'")
DIAGNOSTIC(1005, Error, unexpectedPositionalArg, "unexpected positional arg")

#undef DIAGNOSTIC
