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
#error Need to #define DIAGNOSTIC(...) before including "slang-cpp-parser/diagnostics-defs.h"
#define DIAGNOSTIC(id, severity, name, messageFormat) /* */
#endif

#if 0
DIAGNOSTIC(-1, Note, seeDeclarationOf, "see declaration of '$0'")
DIAGNOSTIC(-1, Note, seeOpen, "see open $0")
DIAGNOSTIC(-1, Note, commandLine, "Command line: $0")
DIAGNOSTIC(-1, Note, previousLocation, "previous location")
#endif

// Command Line

DIAGNOSTIC(100001, Error, unknownOption, "unknown option '$0'")

// Basic I/O

DIAGNOSTIC(200001, Error, couldNotReadInputFile, "could not read input file '$0'")
DIAGNOSTIC(200002, Error, couldNotOverwriteInputFile, "could not overwrite input file '$0'")
DIAGNOSTIC(200002, Error, couldNotWriteOutputFile, "could not write output file '$0'")

// Template Parsing

DIAGNOSTIC(
    300001,
    Error,
    expectedOutputStartMarker,
    "start line for template not followed by a line marking output with '$0'")
DIAGNOSTIC(300002, Error, expectedEndMarker, "expected a template end line ('$0')")

// Scraper: Parsing

DIAGNOSTIC(500001, Error, unexpected, "unexpected $0, expected $1")

DIAGNOSTIC(
    501001,
    Error,
    expectedFiddleEllipsisInvocation,
    "expected 'FIDDLE(...)' at start of body of '$0'")

DIAGNOSTIC(
    502001,
    Error,
    expectedIncludeOfOutputHeader,
    "expected a '#include' of generated output file '$0' in file containing 'FIDDLE(...)' "
    "invocations")

// Scraper: Semantic Checking

DIAGNOSTIC(600001, Error, undefinedIdentifier, "undefined identifier '$0'")


DIAGNOSTIC(999999, Fatal, internalError, "internal error in 'fiddle' tool")

#undef DIAGNOSTIC
