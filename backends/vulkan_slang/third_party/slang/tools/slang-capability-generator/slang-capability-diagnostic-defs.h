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
#error Need to #define DIAGNOSTIC(...) before including "DiagnosticDefs.h"
#define DIAGNOSTIC(id, severity, name, messageFormat) /* */
#endif

//
// -1 - Notes that decorate another diagnostic.
//

DIAGNOSTIC(-1, Note, seeDefinitionOf, "see definition of '$0'")

//
// 0xxxx -  Command line and interaction with host platform APIs.
//

DIAGNOSTIC(1, Error, cannotOpenFile, "cannot open file '$0'.")
DIAGNOSTIC(2, Error, cannotFindFile, "cannot find file '$0'.")
DIAGNOSTIC(4, Error, cannotWriteOutputFile, "cannot write output file '$0'.")
DIAGNOSTIC(5, Error, failedToLoadDynamicLibrary, "failed to load dynamic library '$0'")
DIAGNOSTIC(
    6,
    Error,
    tooManyOutputPathsSpecified,
    "$0 output paths specified, but only $1 entry points given")
DIAGNOSTIC(
    7,
    Warning,
    couldNotFindValidDocumentationOutputPath,
    "could not find valid documentation output path at $0")

//
// 2xxxx - Parsing
//

DIAGNOSTIC(20003, Error, unexpectedToken, "unexpected $0")
DIAGNOSTIC(20001, Error, unexpectedTokenExpectedTokenType, "unexpected $0, expected $1")
DIAGNOSTIC(20001, Error, unexpectedTokenExpectedTokenName, "unexpected $0, expected '$1'")
DIAGNOSTIC(20004, Warning, requiresDocComment, "'$0' requires a documentation comment \"///\"")
DIAGNOSTIC(
    20004,
    Warning,
    invalidDocCommentHeader,
    "got documentation comment '[$0]', expected one of: [Target] [Stage] [EXT] [Version] "
    "[Compound] [Other]")

DIAGNOSTIC(0, Error, tokenNameExpectedButEOF, "\"$0\" expected but end of file encountered.")
DIAGNOSTIC(0, Error, tokenTypeExpectedButEOF, "$0 expected but end of file encountered.")
DIAGNOSTIC(20001, Error, tokenNameExpected, "\"$0\" expected")
DIAGNOSTIC(20001, Error, tokenNameExpectedButEOF2, "\"$0\" expected but end of file encountered.")
DIAGNOSTIC(20001, Error, tokenTypeExpected, "$0 expected")
DIAGNOSTIC(20001, Error, tokenTypeExpectedButEOF2, "$0 expected but end of file encountered.")
DIAGNOSTIC(20001, Error, typeNameExpectedBut, "unexpected $0, expected type name")
DIAGNOSTIC(20001, Error, typeNameExpectedButEOF, "type name expected but end of file encountered.")
DIAGNOSTIC(20001, Error, unexpectedEOF, " Unexpected end of file.")
DIAGNOSTIC(20002, Error, syntaxError, "syntax error.")
DIAGNOSTIC(20003, Error, undefinedIdentifier, "undefined identifier \"$0\".")
DIAGNOSTIC(20004, Error, redefinition, "capability redefinition: '$0'.")
DIAGNOSTIC(
    20005,
    Error,
    unionWithSameKeyAtomButNotSubset,
    "unioning ('|') capability sets which have incompatible atoms but compatible 'key atoms', "
    "this: '$0', other: '$1'")
DIAGNOSTIC(
    20006,
    Error,
    invalidJoinInGenerator,
    "joining ('+') capability sets which have incompatible 'key atoms'")
DIAGNOSTIC(
    20007,
    Error,
    missingExternalInternalAtomPair,
    "All internal '_atom' require a corresponding external 'atom' atom meant for user's use. "
    "Offending atom: $0")
#undef DIAGNOSTIC
