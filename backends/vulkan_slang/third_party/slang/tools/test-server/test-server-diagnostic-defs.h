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
#error Need to #define DIAGNOSTIC(...) before including "test-server-diagnostics-defs.h"
#define DIAGNOSTIC(id, severity, name, messageFormat) /* */
#endif

DIAGNOSTIC(100000, Error, unableToLoadSharedLibrary, "Unable to load shared library '$0'")
DIAGNOSTIC(
    100001,
    Error,
    unableToFindFunctionInSharedLibrary,
    "Unable to find function '$0' in shared library")
DIAGNOSTIC(100002, Error, unableToGetUnitTestModule, "Unable to get unit test module")
DIAGNOSTIC(100003, Error, unableToFindTest, "Unable to find test '$0'")
DIAGNOSTIC(100004, Error, unableToFindUnitTestModule, "Unable to find unit test module '$0'")

#undef DIAGNOSTIC
