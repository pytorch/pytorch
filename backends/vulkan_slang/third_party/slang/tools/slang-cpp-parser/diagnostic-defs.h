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

DIAGNOSTIC(-1, Note, seeDeclarationOf, "see declaration of '$0'")
DIAGNOSTIC(-1, Note, seeOpen, "see open $0")
DIAGNOSTIC(-1, Note, commandLine, "Command line: $0")
DIAGNOSTIC(-1, Note, previousLocation, "previous location")

DIAGNOSTIC(1, Error, cannotOpenFile, "cannot open file '$0'.")
DIAGNOSTIC(1, Error, errorAccessingFile, "error accessing file '$0'.")

DIAGNOSTIC(1, Error, extractorFailed, "C++ Extractor failed")
DIAGNOSTIC(1, Error, internalError, "Unknown internal error in C++ Extractor, aborted!")

// Parsing errors
DIAGNOSTIC(100000, Error, expectingToken, "Expecting token $0")
DIAGNOSTIC(100001, Error, typeAlreadyDeclared, "Type '$0' already declared")
DIAGNOSTIC(100002, Error, scopeNotClosed, "Scope not closed")
DIAGNOSTIC(100003, Error, typeNameDoesntMatch, "Type name doesn't match $0")
DIAGNOSTIC(100004, Error, didntFindMatchingBrace, "Didn't find brace matching $0")
DIAGNOSTIC(100005, Error, braceOpenAtEndOfFile, "Brace open at file end")
DIAGNOSTIC(100006, Error, unexpectedTemplateClose, "Unexpected template close")
DIAGNOSTIC(100007, Error, superTypeNotFound, "Super type not found for $0")
DIAGNOSTIC(100008, Error, superTypeNotAType, "Named super type is not a type $0")
DIAGNOSTIC(100009, Error, unexpectedUnbalancedToken, "Unexpected unbalanced token")
DIAGNOSTIC(100010, Error, unexpectedEndOfFile, "Unexpected end of file")
DIAGNOSTIC(
    100011,
    Error,
    expectingTypeKeyword,
    "Expecting type keyword - struct or class, found $0")

DIAGNOSTIC(
    100012,
    Error,
    typeInDifferentTypeSet,
    "Type $0 in different type set $1 from super class $2")
DIAGNOSTIC(100013, Error, expectingIdentifier, "Expecting an identifier, found $0")
DIAGNOSTIC(100014, Error, cannotDeclareTypeInScope, "Cannot declare types in this scope")
DIAGNOSTIC(100015, Error, identifierAlreadyDefined, "Identifier already defined '$0'")
DIAGNOSTIC(100016, Error, expectingType, "Expecting a type")
DIAGNOSTIC(100017, Error, cannotParseExpression, "Cannot parse expression")
DIAGNOSTIC(100018, Error, cannotOverload, "Cannot overload this declaration");
DIAGNOSTIC(
    100019,
    Error,
    classMarkerOutsideOfClass,
    "A class/struct marker is found outside of class or struct");
DIAGNOSTIC(
    100020,
    Error,
    classMarkerAlreadyFound,
    "A class/struct marker already found in type '$0'");
DIAGNOSTIC(100021, Error, cannotParseType, "Cannot parse type");
DIAGNOSTIC(
    100022,
    Error,
    destructorNameDoesntMatch,
    "Destructor name doesn't match class name '$0'");
DIAGNOSTIC(100023, Error, cannotParseCallable, "Cannot parse callable");
DIAGNOSTIC(
    100024,
    Error,
    cannoseUseArchDependentType,
    "Cannot use architecture dependent type '$0' for serializable data.")

// Command line errors 100100

DIAGNOSTIC(100101, Error, optionAlreadyDefined, "Option '$0' is already defined '$1'")
DIAGNOSTIC(100102, Error, requireValueAfterOption, "Require a value after $0 option")
DIAGNOSTIC(100103, Error, unknownOption, "Unknown option '$0'")
DIAGNOSTIC(100104, Error, noInputPathsSpecified, "No input paths specified")
DIAGNOSTIC(100105, Error, noOutputPathSpecified, "No -o output path specified")

#undef DIAGNOSTIC
