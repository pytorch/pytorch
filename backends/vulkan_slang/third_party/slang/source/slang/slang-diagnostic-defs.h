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

DIAGNOSTIC(-1, Note, alsoSeePipelineDefinition, "also see pipeline definition")
DIAGNOSTIC(
    -1,
    Note,
    implicitParameterMatchingFailedBecauseNameNotAccessible,
    "implicit parameter matching failed because the component of the same name is not accessible "
    "from '$0'.\ncheck if you have declared necessary requirements and properly used the 'public' "
    "qualifier.")
DIAGNOSTIC(
    -1,
    Note,
    implicitParameterMatchingFailedBecauseShaderDoesNotDefineComponent,
    "implicit parameter matching failed because shader '$0' does not define component '$1'.")
DIAGNOSTIC(
    -1,
    Note,
    implicitParameterMatchingFailedBecauseTypeMismatch,
    "implicit parameter matching failed because the component of the same name does not match "
    "parameter type '$0'.")
DIAGNOSTIC(-1, Note, noteShaderIsTargetingPipeine, "shader '$0' is targeting pipeline '$1'")
DIAGNOSTIC(-1, Note, seeDefinitionOf, "see definition of '$0'")
DIAGNOSTIC(-1, Note, seeConstantBufferDefinition, "see constant buffer definition.")
DIAGNOSTIC(-1, Note, seeInterfaceDefinitionOf, "see interface definition of '$0'")
DIAGNOSTIC(-1, Note, seeUsingOf, "see using of '$0'")
DIAGNOSTIC(-1, Note, seeDefinitionOfShader, "see definition of shader '$0'")
DIAGNOSTIC(-1, Note, seeInclusionOf, "see inclusion of '$0'")
DIAGNOSTIC(-1, Note, seeModuleBeingUsedIn, "see module '$0' being used in '$1'")
DIAGNOSTIC(-1, Note, seeCallOfFunc, "see call to '$0'")
DIAGNOSTIC(-1, Note, seePipelineRequirementDefinition, "see pipeline requirement definition")
DIAGNOSTIC(
    -1,
    Note,
    seePotentialDefinitionOfComponent,
    "see potential definition of component '$0'")
DIAGNOSTIC(-1, Note, seePreviousDefinition, "see previous definition")
DIAGNOSTIC(-1, Note, seePreviousDefinitionOf, "see previous definition of '$0'")
DIAGNOSTIC(-1, Note, seeRequirementDeclaration, "see requirement declaration")
DIAGNOSTIC(
    -1,
    Note,
    doYouForgetToMakeComponentAccessible,
    "do you forget to make component '$0' acessible from '$1' (missing public qualifier)?")

DIAGNOSTIC(-1, Note, seeDeclarationOf, "see declaration of '$0'")
DIAGNOSTIC(
    -1,
    Note,
    seeDeclarationOfInterfaceRequirement,
    "see interface requirement declaration of '$0'")

DIAGNOSTIC(
    -1,
    Note,
    genericSignatureDoesNotMatchRequirement,
    "generic signature of '$0' does not match interface requirement.")

DIAGNOSTIC(
    -1,
    Note,
    cannotResolveOverloadForMethodRequirement,
    "none of the overloads of '$0' match the interface requirement.")

DIAGNOSTIC(
    -1,
    Note,
    parameterDirectionDoesNotMatchRequirement,
    "parameter '$0' is '$1' in the implementing member, but the interface requires '$2'.")

// An alternate wording of the above note, emphasing the position rather than content of the
// declaration.
DIAGNOSTIC(-1, Note, declaredHere, "declared here")
DIAGNOSTIC(-1, Note, seeOtherDeclarationOf, "see other declaration of '$0'")
DIAGNOSTIC(-1, Note, seePreviousDeclarationOf, "see previous declaration of '$0'")
DIAGNOSTIC(-1, Note, includeOutput, "include $0")
DIAGNOSTIC(-1, Note, genericSignatureTried, "see declaration of $0")
DIAGNOSTIC(-1, Note, entryPointCandidate, "see candidate declaration for entry point '$0'")

//
// 0xxxx -  Command line and interaction with host platform APIs.
//

DIAGNOSTIC(1, Error, cannotOpenFile, "cannot open file '$0'.")
DIAGNOSTIC(2, Error, cannotFindFile, "cannot find file '$0'.")
DIAGNOSTIC(2, Error, unsupportedCompilerMode, "unsupported compiler mode.")
DIAGNOSTIC(4, Error, cannotWriteOutputFile, "cannot write output file '$0'.")
DIAGNOSTIC(5, Error, failedToLoadDynamicLibrary, "failed to load dynamic library '$0'")
DIAGNOSTIC(
    6,
    Error,
    tooManyOutputPathsSpecified,
    "$0 output paths specified, but only $1 entry points given")

DIAGNOSTIC(
    7,
    Error,
    noOutputPathSpecifiedForEntryPoint,
    "no output path specified for entry point '$0' (the '-o' option for an entry point must "
    "precede the corresponding '-entry')")

DIAGNOSTIC(
    8,
    Error,
    outputPathsImplyDifferentFormats,
    "the output paths '$0' and '$1' require different code-generation targets")

DIAGNOSTIC(
    10,
    Error,
    explicitOutputPathsAndMultipleTargets,
    "canot use both explicit output paths ('-o') and multiple targets ('-target')")
DIAGNOSTIC(12, Error, cannotDeduceSourceLanguage, "can't deduce language for input file '$0'")
DIAGNOSTIC(13, Error, unknownCodeGenerationTarget, "unknown code generation target '$0'")
DIAGNOSTIC(14, Error, unknownProfile, "unknown profile '$0'")
DIAGNOSTIC(15, Error, unknownStage, "unknown stage '$0'")
DIAGNOSTIC(16, Error, unknownPassThroughTarget, "unknown pass-through target '$0'")
DIAGNOSTIC(17, Error, unknownCommandLineOption, "unknown command-line option '$0'")
DIAGNOSTIC(19, Error, unknownSourceLanguage, "unknown source language '$0'")

DIAGNOSTIC(
    20,
    Error,
    entryPointsNeedToBeAssociatedWithTranslationUnits,
    "when using multiple source files, entry points must be specified after their corresponding "
    "source file(s)")
DIAGNOSTIC(22, Error, unknownDownstreamCompiler, "unknown downstream compiler '$0'")

DIAGNOSTIC(26, Error, unknownOptimiziationLevel, "unknown optimization level '$0'")

DIAGNOSTIC(28, Error, unableToGenerateCodeForTarget, "unable to generate code for target '$0'")

DIAGNOSTIC(
    30,
    Warning,
    sameStageSpecifiedMoreThanOnce,
    "the stage '$0' was specified more than once for entry point '$1'")
DIAGNOSTIC(
    31,
    Error,
    conflictingStagesForEntryPoint,
    "conflicting stages have been specified for entry point '$0'")
DIAGNOSTIC(
    32,
    Warning,
    explicitStageDoesntMatchImpliedStage,
    "the stage specified for entry point '$0' ('$1') does not match the stage implied by the "
    "source file name ('$2')")
DIAGNOSTIC(
    33,
    Error,
    stageSpecificationIgnoredBecauseNoEntryPoints,
    "one or more stages were specified, but no entry points were specified with '-entry'")
DIAGNOSTIC(
    34,
    Error,
    stageSpecificationIgnoredBecauseBeforeAllEntryPoints,
    "when compiling multiple entry points, any '-stage' options must follow the '-entry' option "
    "that they apply to")
DIAGNOSTIC(
    35,
    Error,
    noStageSpecifiedInPassThroughMode,
    "no stage was specified for entry point '$0'; when using the '-pass-through' option, stages "
    "must be fully specified on the command line")
DIAGNOSTIC(36, Error, expectingAnInteger, "expecting an integer value")

DIAGNOSTIC(
    40,
    Warning,
    sameProfileSpecifiedMoreThanOnce,
    "the '$0' was specified more than once for target '$0'")
DIAGNOSTIC(
    41,
    Error,
    conflictingProfilesSpecifiedForTarget,
    "conflicting profiles have been specified for target '$0'")

DIAGNOSTIC(
    42,
    Error,
    profileSpecificationIgnoredBecauseNoTargets,
    "a '-profile' option was specified, but no target was specified with '-target'")
DIAGNOSTIC(
    43,
    Error,
    profileSpecificationIgnoredBecauseBeforeAllTargets,
    "when using multiple targets, any '-profile' option must follow the '-target' it applies to")

DIAGNOSTIC(
    42,
    Error,
    targetFlagsIgnoredBecauseNoTargets,
    "target options were specified, but no target was specified with '-target'")
DIAGNOSTIC(
    43,
    Error,
    targetFlagsIgnoredBecauseBeforeAllTargets,
    "when using multiple targets, any target options must follow the '-target' they apply to")

DIAGNOSTIC(50, Error, duplicateTargets, "the target '$0' has been specified more than once")

DIAGNOSTIC(
    51,
    Error,
    unhandledLanguageForSourceEmbedding,
    "unhandled source language for source embedding")

DIAGNOSTIC(
    60,
    Error,
    cannotDeduceOutputFormatFromPath,
    "cannot infer an output format from the output path '$0'")
DIAGNOSTIC(
    61,
    Error,
    cannotMatchOutputFileToTarget,
    "no specified '-target' option matches the output path '$0', which implies the '$1' format")

DIAGNOSTIC(62, Error, unknownCommandLineValue, "unknown value for option. Valid values are '$0'")
DIAGNOSTIC(63, Error, unknownHelpCategory, "unknown help category")

DIAGNOSTIC(
    70,
    Error,
    cannotMatchOutputFileToEntryPoint,
    "the output path '$0' is not associated with any entry point; a '-o' option for a compiled "
    "kernel must follow the '-entry' option for its corresponding entry point")

DIAGNOSTIC(
    80,
    Error,
    duplicateOutputPathsForEntryPointAndTarget,
    "multiple output paths have been specified entry point '$0' on target '$1'")
DIAGNOSTIC(
    81,
    Error,
    duplicateOutputPathsForTarget,
    "multiple output paths have been specified for target '$0'")
DIAGNOSTIC(
    82,
    Error,
    duplicateDependencyOutputPaths,
    "the -dep argument can only be specified once")

DIAGNOSTIC(82, Error, unableToWriteReproFile, "unable to write repro file '%0'")
DIAGNOSTIC(83, Error, unableToWriteModuleContainer, "unable to write module container '%0'")
DIAGNOSTIC(84, Error, unableToReadModuleContainer, "unable to read module container '%0'")
DIAGNOSTIC(
    85,
    Error,
    unableToAddReferenceToModuleContainer,
    "unable to add a reference to a module container")
DIAGNOSTIC(86, Error, unableToCreateModuleContainer, "unable to create module container")

DIAGNOSTIC(
    87,
    Error,
    unableToSetDefaultDownstreamCompiler,
    "unable to set default downstream compiler for source language '%0' to '%1'")

DIAGNOSTIC(88, Error, unknownArchiveType, "archive type '%0' is unknown")
DIAGNOSTIC(89, Error, expectingSlangRiffContainer, "expecting a slang riff container")
DIAGNOSTIC(
    90,
    Error,
    incompatibleRiffSemanticVersion,
    "incompatible riff semantic version %0 expecting %1")
DIAGNOSTIC(91, Error, riffHashMismatch, "riff hash mismatch - incompatible riff")
DIAGNOSTIC(92, Error, unableToCreateDirectory, "unable to create directory '$0'")
DIAGNOSTIC(93, Error, unableExtractReproToDirectory, "unable to extract repro to directory '$0'")
DIAGNOSTIC(94, Error, unableToReadRiff, "unable to read as 'riff'/not a 'riff' file")

DIAGNOSTIC(95, Error, unknownLibraryKind, "unknown library kind '$0'")
DIAGNOSTIC(96, Error, kindNotLinkable, "not a known linkable kind '$0'")
DIAGNOSTIC(97, Error, libraryDoesNotExist, "library '$0' does not exist")
DIAGNOSTIC(98, Error, cannotAccessAsBlob, "cannot access as a blob")
DIAGNOSTIC(99, Error, unknownDebugOption, "unknown debug option, known options are ($0)")

//
// 001xx - Downstream Compilers
//

DIAGNOSTIC(100, Error, failedToLoadDownstreamCompiler, "failed to load downstream compiler '$0'")
DIAGNOSTIC(
    101,
    Error,
    downstreamCompilerDoesntSupportWholeProgramCompilation,
    "downstream compiler '$0' doesn't support whole program compilation")
DIAGNOSTIC(102, Note, downstreamCompileTime, "downstream compile time: $0s")
DIAGNOSTIC(103, Note, performanceBenchmarkResult, "compiler performance benchmark:\n$0")
DIAGNOSTIC(99999, Note, noteFailedToLoadDynamicLibrary, "failed to load dynamic library '$0'")

//
// 15xxx - Preprocessing
//

// 150xx - conditionals
DIAGNOSTIC(
    15000,
    Error,
    endOfFileInPreprocessorConditional,
    "end of file encountered during preprocessor conditional")
DIAGNOSTIC(15001, Error, directiveWithoutIf, "'$0' directive without '#if'")
DIAGNOSTIC(15002, Error, directiveAfterElse, "'$0' directive without '#if'")

DIAGNOSTIC(-1, Note, seeDirective, "see '$0' directive")

// 151xx - directive parsing
DIAGNOSTIC(15100, Error, expectedPreprocessorDirectiveName, "expected preprocessor directive name")
DIAGNOSTIC(15101, Error, unknownPreprocessorDirective, "unknown preprocessor directive '$0'")
DIAGNOSTIC(15102, Error, expectedTokenInPreprocessorDirective, "expected '$0' in '$1' directive")
DIAGNOSTIC(
    15102,
    Error,
    expected2TokensInPreprocessorDirective,
    "expected '$0' or '$1' in '$2' directive")
DIAGNOSTIC(
    15103,
    Error,
    unexpectedTokensAfterDirective,
    "unexpected tokens following '$0' directive")


// 152xx - preprocessor expressions
DIAGNOSTIC(
    15200,
    Error,
    expectedTokenInPreprocessorExpression,
    "expected '$0' in preprocessor expression")
DIAGNOSTIC(
    15201,
    Error,
    syntaxErrorInPreprocessorExpression,
    "syntax error in preprocessor expression")
DIAGNOSTIC(
    15202,
    Error,
    divideByZeroInPreprocessorExpression,
    "division by zero in preprocessor expression")
DIAGNOSTIC(15203, Error, expectedTokenInDefinedExpression, "expected '$0' in 'defined' expression")
DIAGNOSTIC(15204, Warning, directiveExpectsExpression, "'$0' directive requires an expression")
DIAGNOSTIC(
    15205,
    Warning,
    undefinedIdentifierInPreprocessorExpression,
    "undefined identifier '$0' in preprocessor expression will evaluate to zero")
DIAGNOSTIC(15206, Error, expectedIntegralVersionNumber, "Expected integer for #version number")

DIAGNOSTIC(-1, Note, seeOpeningToken, "see opening '$0'")

// 153xx - #include
DIAGNOSTIC(15300, Error, includeFailed, "failed to find include file '$0'")
DIAGNOSTIC(15301, Error, importFailed, "failed to find imported file '$0'")
DIAGNOSTIC(-1, Error, noIncludeHandlerSpecified, "no `#include` handler was specified")
DIAGNOSTIC(
    15302,
    Error,
    noUniqueIdentity,
    "`#include` handler didn't generate a unique identity for file '$0'")


// 154xx - macro definition
DIAGNOSTIC(15400, Warning, macroRedefinition, "redefinition of macro '$0'")
DIAGNOSTIC(15401, Warning, macroNotDefined, "macro '$0' is not defined")
DIAGNOSTIC(15403, Error, expectedTokenInMacroParameters, "expected '$0' in macro parameters")
DIAGNOSTIC(15404, Warning, builtinMacroRedefinition, "Redefinition of builtin macro '$0'")

DIAGNOSTIC(15405, Error, tokenPasteAtStart, "'##' is not allowed at the start of a macro body")
DIAGNOSTIC(15406, Error, tokenPasteAtEnd, "'##' is not allowed at the end of a macro body")
DIAGNOSTIC(
    15407,
    Error,
    expectedMacroParameterAfterStringize,
    "'#' in macro body must be followed by the name of a macro parameter")
DIAGNOSTIC(15408, Error, duplicateMacroParameterName, "redefinition of macro parameter '$0'")
DIAGNOSTIC(
    15409,
    Error,
    variadicMacroParameterMustBeLast,
    "a variadic macro parameter is only allowed at the end of the parameter list")

// 155xx - macro expansion
DIAGNOSTIC(15500, Warning, expectedTokenInMacroArguments, "expected '$0' in macro invocation")
DIAGNOSTIC(
    15501,
    Error,
    wrongNumberOfArgumentsToMacro,
    "wrong number of arguments to macro (expected $0, got $1)")
DIAGNOSTIC(
    15502,
    Error,
    errorParsingToMacroInvocationArgument,
    "error parsing macro '$0' invocation argument to '$1'")

DIAGNOSTIC(
    15503,
    Warning,
    invalidTokenPasteResult,
    "toking pasting with '##' resulted in the invalid token '$0'")

// 156xx - pragmas
DIAGNOSTIC(15600, Error, expectedPragmaDirectiveName, "expected a name after '#pragma'")
DIAGNOSTIC(15601, Warning, unknownPragmaDirectiveIgnored, "ignoring unknown directive '#pragma $0'")
DIAGNOSTIC(
    15602,
    Warning,
    pragmaOnceIgnored,
    "pragma once was ignored - this is typically because is not placed in an include")
DIAGNOSTIC(15610, Error, pragmaWarningGenericError, "Error in #pragma warning processing: $0")
DIAGNOSTIC(
    15611,
    Warning,
    pragmaWarningPopEmpty,
    "Detected #pragma warning(pop) with no corresponding #pragma warning(push)")
DIAGNOSTIC(
    15612,
    Warning,
    pragmaWarningPushNotPopped,
    "Detected #pragma warning(push) with no corresponding #pragma warning(pop)")
DIAGNOSTIC(15613, Warning, pragmaWarningUnknownSpecifier, "Unknown #pragma warning specifier '$0'")
DIAGNOSTIC(
    15614,
    Warning,
    pragmaWarningSuppressCannotIdentifyNextLine,
    "Cannot identify the next line to suppress in #pragma warning suppress")
DIAGNOSTIC(
    15615,
    Warning,
    pragmaWarningCannotInsertHere,
    "Cannot insert #pragma warning here for id '$0'")
DIAGNOSTIC(
    15616,
    Note,
    pragmaWarningPointSuppress,
    "#pragma warning for id '$0' was suppressed here")


// 159xx - user-defined error/warning
DIAGNOSTIC(15900, Error, userDefinedError, "#error: $0")
DIAGNOSTIC(15901, Warning, userDefinedWarning, "#warning: $0")

//
// 2xxxx - Parsing
//

DIAGNOSTIC(20003, Error, unexpectedToken, "unexpected $0")
DIAGNOSTIC(20001, Error, unexpectedTokenExpectedTokenType, "unexpected $0, expected $1")
DIAGNOSTIC(20001, Error, unexpectedTokenExpectedTokenName, "unexpected $0, expected '$1'")

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
DIAGNOSTIC(
    20004,
    Error,
    unexpectedTokenExpectedComponentDefinition,
    "unexpected token '$0', only component definitions are allowed in a shader scope.")
DIAGNOSTIC(
    20005,
    Error,
    invalidEmptyParenthesisExpr,
    "empty parenthesis '()' is not a valid expression.")
DIAGNOSTIC(20008, Error, invalidOperator, "invalid operator '$0'.")
DIAGNOSTIC(20011, Error, unexpectedColon, "unexpected ':'.")
DIAGNOSTIC(
    20012,
    Error,
    invalidSPIRVVersion,
    "Expecting SPIR-V version as either 'major.minor', or quoted if has patch (eg for SPIR-V 1.2, "
    "'1.2' or \"1.2\"')")
DIAGNOSTIC(
    20013,
    Error,
    invalidCUDASMVersion,
    "Expecting CUDA SM version as either 'major.minor', or quoted if has patch (eg for '7.0' or "
    "\"7.0\"')")
DIAGNOSTIC(
    20014,
    Error,
    classIsReservedKeyword,
    "'class' is a reserved keyword in this context; use 'struct' instead.")
DIAGNOSTIC(20015, Error, unknownSPIRVCapability, "unknown SPIR-V capability '$0'.")
DIAGNOSTIC(
    20016,
    Error,
    missingLayoutBindingModifier,
    "Expecting 'binding' modifier in the layout qualifier here")

DIAGNOSTIC(
    20101,
    Warning,
    unintendedEmptyStatement,
    "potentially unintended empty statement at this location; use {} instead.")
DIAGNOSTIC(
    20102,
    Error,
    unexpectedBodyAfterSemicolon,
    "unexpected function body after signature declaration, is this ';' a typo?")
DIAGNOSTIC(30102, Error, declNotAllowed, "$0 is not allowed here.")

// 29xxx - Snippet parsing and inline asm
DIAGNOSTIC(29000, Error, snippetParsingFailed, "unable to parse target intrinsic snippet: $0")

DIAGNOSTIC(29100, Error, unrecognizedSPIRVOpcode, "unrecognized spirv opcode: $0")
DIAGNOSTIC(
    29101,
    Error,
    misplacedResultIdMarker,
    "the result-id marker must only be used in the last instruction of a spriv_asm expression")
DIAGNOSTIC(
    29102,
    Note,
    considerOpCopyObject,
    "consider adding an OpCopyObject instruction to the end of the spirv_asm expression")
DIAGNOSTIC(29103, Note, noSuchAddress, "unable to take the address of this address-of asm operand")
DIAGNOSTIC(
    29104,
    Error,
    spirvInstructionWithoutResultId,
    "cannot use this 'x = $0...' syntax because $0 does not have a <result-id> operand")
DIAGNOSTIC(
    29105,
    Error,
    spirvInstructionWithoutResultTypeId,
    "cannot use this 'x : <type> = $0...' syntax because $0 does not have a <result-type-id> "
    "operand")
// This is a warning because we trust that people using the spirv_asm block know what they're doing
DIAGNOSTIC(
    29106,
    Warning,
    spirvInstructionWithTooManyOperands,
    "too many operands for $0 (expected max $1), did you forget a semicolon?")
DIAGNOSTIC(
    29107,
    Error,
    spirvUnableToResolveName,
    "unknown SPIR-V identifier $0, it's not a known enumerator or opcode")
DIAGNOSTIC(
    29108,
    Error,
    spirvNonConstantBitwiseOr,
    "only integer literals and enum names can appear in a bitwise or expression")
DIAGNOSTIC(29109, Error, spirvOperandRange, "Literal ints must be in the range 0 to 0xffffffff")

DIAGNOSTIC(29110, Error, unknownTargetName, "unknown target name '$0'")

DIAGNOSTIC(
    29111,
    Error,
    spirvInvalidTruncate,
    "__truncate has been given a source smaller than its target")
DIAGNOSTIC(29112, Error, spirvInstructionWithNotEnoughOperands, "not enough operands for $0")
DIAGNOSTIC(
    29113,
    Error,
    spirvIdRedefinition,
    "SPIRV id '%$0' is already defined in the current assembly block")
DIAGNOSTIC(
    29114,
    Error,
    spirvUndefinedId,
    "SPIRV id '%$0' is not defined in the current assembly block location")

DIAGNOSTIC(
    29115,
    Error,
    targetSwitchCaseCannotBeAStage,
    "cannot use a stage name in '__target_switch', use '__stage_switch' for stage-specific code.")

//
// 3xxxx - Semantic analysis
//
DIAGNOSTIC(30002, Error, divideByZero, "divide by zero")
DIAGNOSTIC(30003, Error, breakOutsideLoop, "'break' must appear inside loop or switch constructs.")
DIAGNOSTIC(30004, Error, continueOutsideLoop, "'continue' must appear inside loop constructs.")
DIAGNOSTIC(30005, Error, whilePredicateTypeError, "'while': expression must evaluate to int.")
DIAGNOSTIC(30006, Error, ifPredicateTypeError, "'if': expression must evaluate to int.")
DIAGNOSTIC(30006, Error, returnNeedsExpression, "'return' should have an expression.")
DIAGNOSTIC(
    30007,
    Error,
    componentReturnTypeMismatch,
    "expression type '$0' does not match component's type '$1'")
DIAGNOSTIC(
    30007,
    Error,
    functionReturnTypeMismatch,
    "expression type '$0' does not match function's return type '$1'")
DIAGNOSTIC(30008, Error, variableNameAlreadyDefined, "variable $0 already defined.")
DIAGNOSTIC(30009, Error, invalidTypeVoid, "invalid type 'void'.")
DIAGNOSTIC(30010, Error, whilePredicateTypeError2, "'while': expression must evaluate to int.")
DIAGNOSTIC(30011, Error, assignNonLValue, "left of '=' is not an l-value.")
DIAGNOSTIC(30012, Error, noApplicationUnaryOperator, "no overload found for operator $0 ($1).")
DIAGNOSTIC(
    30012,
    Error,
    noOverloadFoundForBinOperatorOnTypes,
    "no overload found for operator $0  ($1, $2).")
DIAGNOSTIC(30013, Error, subscriptNonArray, "no subscript operation found for type '$0'")
DIAGNOSTIC(30014, Error, subscriptIndexNonInteger, "index expression must evaluate to int.")
DIAGNOSTIC(30016, Error, callOperatorNotFound, "no call operation found for type '$0'")
DIAGNOSTIC(30015, Error, undefinedIdentifier2, "undefined identifier '$0'.")
DIAGNOSTIC(30019, Error, typeMismatch, "expected an expression of type '$0', got '$1'")
DIAGNOSTIC(30021, Error, noApplicationFunction, "$0: no overload takes arguments ($1)")
DIAGNOSTIC(30022, Error, invalidTypeCast, "invalid type cast between \"$0\" and \"$1\".")
DIAGNOSTIC(30023, Error, typeHasNoPublicMemberOfName, "\"$0\" does not have public member \"$1\".")
DIAGNOSTIC(
    30024,
    Error,
    cannotConvertArrayOfSmallerToLargerSize,
    "Cannot convert array of size $0 to array of size $1 as this would truncate data")
DIAGNOSTIC(30025, Error, invalidArraySize, "array size must be larger than zero.")
DIAGNOSTIC(
    30026,
    Error,
    returnInComponentMustComeLast,
    "'return' can only appear as the last statement in component definition.")
DIAGNOSTIC(30027, Error, noMemberOfNameInType, "'$0' is not a member of '$1'.")
DIAGNOSTIC(
    30028,
    Error,
    forPredicateTypeError,
    "'for': predicate expression must evaluate to bool.")
DIAGNOSTIC(
    30030,
    Error,
    projectionOutsideImportOperator,
    "'project': invalid use outside import operator.")
DIAGNOSTIC(
    30031,
    Error,
    projectTypeMismatch,
    "'project': expression must evaluate to record type '$0'.")
DIAGNOSTIC(
    30033,
    Error,
    invalidTypeForLocalVariable,
    "cannot declare a local variable of this type.")
DIAGNOSTIC(
    30035,
    Error,
    componentOverloadTypeMismatch,
    "'$0': type of overloaded component mismatches previous definition.")
DIAGNOSTIC(30041, Error, bitOperationNonIntegral, "bit operation: operand must be integral type.")
DIAGNOSTIC(
    30043,
    Error,
    getStringHashRequiresStringLiteral,
    "getStringHash parameter can only accept a string literal")
DIAGNOSTIC(
    30047,
    Error,
    argumentExpectedLValue,
    "argument passed to parameter '$0' must be l-value.")
DIAGNOSTIC(
    30048,
    Error,
    argumentHasMoreMemoryQualifiersThanParam,
    "argument passed in to parameter has a memory qualifier the parameter type is missing: '$0'")

DIAGNOSTIC(
    30049,
    Note,
    thisIsImmutableByDefault,
    "a 'this' parameter is an immutable parameter by default in Slang; apply the `[mutating]` "
    "attribute to the function declaration to opt in to a mutable `this`")
DIAGNOSTIC(
    30050,
    Error,
    mutatingMethodOnImmutableValue,
    "mutating method '$0' cannot be called on an immutable value")

DIAGNOSTIC(30051, Error, invalidValueForArgument, "invalid value for argument '$0'")
DIAGNOSTIC(30052, Error, invalidSwizzleExpr, "invalid swizzle pattern '$0' on type '$1'")
DIAGNOSTIC(30053, Error, breakLabelNotFound, "label '$0' used as break target is not found.")
DIAGNOSTIC(
    30054,
    Error,
    targetLabelDoesNotMarkBreakableStmt,
    "invalid break target: statement labeled '$0' is not breakable.")
DIAGNOSTIC(
    30055,
    Error,
    useOfNonShortCircuitingOperatorInDiffFunc,
    "non-short-circuiting `?:` operator is not allowed in a differentiable function, use `select` "
    "instead.")
DIAGNOSTIC(
    30056,
    Warning,
    useOfNonShortCircuitingOperator,
    "non-short-circuiting `?:` operator is deprecated, use 'select' instead.")
DIAGNOSTIC(
    30057,
    Error,
    assignmentInPredicateExpr,
    "use an assignment operation as predicate expression is not allowed, wrap the assignment with "
    "'()' to clarify the intent.")
DIAGNOSTIC(30058, Warning, danglingEqualityExpr, "result of '==' not used, did you intend '='?")

DIAGNOSTIC(30060, Error, expectedAType, "expected a type, got a '$0'")
DIAGNOSTIC(30061, Error, expectedANamespace, "expected a namespace, got a '$0'")

DIAGNOSTIC(
    30062,
    Note,
    implicitCastUsedAsLValueRef,
    "argument was implicitly cast from '$0' to '$1', and Slang does not support using an implicit "
    "cast as an l-value with a reference")
DIAGNOSTIC(
    30063,
    Note,
    implicitCastUsedAsLValueType,
    "argument was implicitly cast from '$0' to '$1', and Slang does not support using an implicit "
    "cast as an l-value with this type")
DIAGNOSTIC(
    30064,
    Note,
    implicitCastUsedAsLValue,
    "argument was implicitly cast from '$0' to '$1', and Slang does not support using an implicit "
    "cast as an l-value for this usage")

DIAGNOSTIC(
    30065,
    Error,
    newCanOnlyBeUsedToInitializeAClass,
    "`new` can only be used to initialize a class")
DIAGNOSTIC(
    30066,
    Error,
    classCanOnlyBeInitializedWithNew,
    "a class can only be initialized by a `new` clause")

DIAGNOSTIC(
    30067,
    Error,
    mutatingMethodOnFunctionInputParameterError,
    "mutating method '$0' called on `in` parameter '$1'; changes will not be visible to caller. "
    "copy the parameter into a local variable if this behavior is intended")
DIAGNOSTIC(
    30068,
    Warning,
    mutatingMethodOnFunctionInputParameterWarning,
    "mutating method '$0' called on `in` parameter '$1'; changes will not be visible to caller. "
    "copy the parameter into a local variable if this behavior is intended")

DIAGNOSTIC(
    30070,
    Error,
    unsizedMemberMustAppearLast,
    "member with unknown size at compile time can only appear as the last member in a composite "
    "type.")
DIAGNOSTIC(30071, Error, varCannotBeUnsized, "cannot instantiate a variable of unsized type.")
DIAGNOSTIC(30072, Error, paramCannotBeUnsized, "function parameter cannot be unsized.")

DIAGNOSTIC(
    30075,
    Error,
    cannotSpecializeGeneric,
    "cannot specialize generic '$0' with the provided arguments.")

DIAGNOSTIC(30076, Error, globalVarCannotHaveOpaqueType, "global variable cannot have opaque type.")
DIAGNOSTIC(
    30077,
    Error,
    concreteArgumentToOutputInterface,
    "argument passed to parameter '$0' is of concrete type '$1', but interface-typed output "
    "parameters require interface-typed arguments. To allow passing a concrete type to this "
    "function, you can replace '$2 $0' with a generic 'T $0' and a 'where T : $2' constraint.")
DIAGNOSTIC(-1, Note, doYouMeanStaticConst, "do you intend to define a `static const` instead?")
DIAGNOSTIC(-1, Note, doYouMeanUniform, "do you intend to define a `uniform` parameter instead?")

DIAGNOSTIC(
    30100,
    Error,
    staticRefToNonStaticMember,
    "type '$0' cannot be used to refer to non-static member '$1'")
DIAGNOSTIC(
    30101,
    Error,
    cannotDereferenceType,
    "cannot dereference type '$0', do you mean to use '.'?")

DIAGNOSTIC(30200, Error, redeclaration, "declaration of '$0' conflicts with existing declaration")
DIAGNOSTIC(30201, Error, functionRedefinition, "function '$0' already has a body")
DIAGNOSTIC(
    30202,
    Error,
    functionRedeclarationWithDifferentReturnType,
    "function '$0' declared to return '$1' was previously declared to return '$2'")

DIAGNOSTIC(
    30300,
    Error,
    isOperatorValueMustBeInterfaceType,
    "'is'/'as' operator requires an interface-typed expression.")

DIAGNOSTIC(33070, Error, expectedFunction, "expected a function, got '$0'")
DIAGNOSTIC(33071, Error, expectedAStringLiteral, "expected a string literal")

DIAGNOSTIC(
    -1,
    Note,
    noteExplicitConversionPossible,
    "explicit conversion from '$0' to '$1' is possible")
DIAGNOSTIC(
    30080,
    Error,
    ambiguousConversion,
    "more than one implicit conversion exists from '$0' to '$1'")
DIAGNOSTIC(
    30081,
    Warning,
    unrecommendedImplicitConversion,
    "implicit conversion from '$0' to '$1' is not recommended")
DIAGNOSTIC(
    30082,
    Warning,
    implicitConversionToDouble,
    " implicit float-to-double conversion may cause unexpected performance issues, use explicit "
    "cast if intended.")
DIAGNOSTIC(
    30090,
    Error,
    tryClauseMustApplyToInvokeExpr,
    "expression in a 'try' clause must be a call to a function or operator overload.")
DIAGNOSTIC(
    30091,
    Error,
    tryInvokeCalleeShouldThrow,
    "'$0' called from a 'try' clause does not throw an error, make sure the callee is marked as "
    "'throws'")
DIAGNOSTIC(30092, Error, calleeOfTryCallMustBeFunc, "callee in a 'try' clause must be a function")
DIAGNOSTIC(
    30093,
    Error,
    uncaughtTryCallInNonThrowFunc,
    "the current function or environment is not declared to throw any errors, but the 'try' clause "
    "is not caught")
DIAGNOSTIC(
    30094,
    Error,
    mustUseTryClauseToCallAThrowFunc,
    "the callee may throw an error, and therefore must be called within a 'try' clause")
DIAGNOSTIC(
    30095,
    Error,
    errorTypeOfCalleeIncompatibleWithCaller,
    "the error type `$1` of callee `$0` is not compatible with the caller's error type `$2`.")

DIAGNOSTIC(
    30096,
    Error,
    differentialTypeShouldServeAsItsOwnDifferentialType,
    "cannot use type '$0' a `Differential` type. A differential type's differential must be "
    "itself. However, '$0.Differential' is '$1'.")
DIAGNOSTIC(
    30097,
    Error,
    functionNotMarkedAsDifferentiable,
    "function '$0' is not marked as $1-differentiable.")
DIAGNOSTIC(
    30098,
    Error,
    nonStaticMemberFunctionNotAllowedAsDiffOperand,
    "non-static function reference '$0' is not allowed here.")

DIAGNOSTIC(30099, Error, sizeOfArgumentIsInvalid, "argument to sizeof is invalid")
DIAGNOSTIC(
    30083,
    Error,
    countOfArgumentIsInvalid,
    "argument to countof can only be a type pack or tuple")


DIAGNOSTIC(30101, Error, readingFromWriteOnly, "cannot read from writeonly, check modifiers.")
DIAGNOSTIC(
    30102,
    Error,
    differentiableMemberShouldHaveCorrespondingFieldInDiffType,
    "differentiable member '$0' should have a corresponding field in '$1'. Use "
    "[DerivativeMember($1.<field-name>)] or mark as no_diff")

DIAGNOSTIC(30103, Error, expectTypePackAfterEach, "expected a type pack or a tuple after 'each'.")
DIAGNOSTIC(
    30104,
    Error,
    eachExprMustBeInsideExpandExpr,
    "'each' expression must be inside 'expand' expression.")
DIAGNOSTIC(
    30105,
    Error,
    expandTermCapturesNoTypePacks,
    "'expand' term captures no type packs. At least one type pack must be referenced via an 'each' "
    "term inside an 'expand' term.")
DIAGNOSTIC(30106, Error, improperUseOfType, "type '$0' cannot be used in this context.")
DIAGNOSTIC(30107, Error, parameterPackMustBeConst, "a parameter pack must be declared as 'const'.")

DIAGNOSTIC(30108, Error, breakInsideDefer, "'break' must not appear inside a defer statement.")
DIAGNOSTIC(
    30109,
    Error,
    continueInsideDefer,
    "'continue' must not appear inside a defer statement.")
DIAGNOSTIC(30110, Error, returnInsideDefer, "'return' must not appear inside a defer statement.")
DIAGNOSTIC(
    30111,
    Error,
    returnTypeMismatchInsideLambda,
    "returned values must have the same type among all 'return' statements inside a lambda "
    "expression: returned '$0' here, but '$1' previously.")

DIAGNOSTIC(
    30112,
    Error,
    nonCopyableTypeCapturedInLambda,
    "cannot capture non-copyable type '$0' in a lambda expression.")

// Include
DIAGNOSTIC(
    30500,
    Error,
    includedFileMissingImplementing,
    "missing 'implementing' declaration in the included source file '$0'.")
DIAGNOSTIC(
    30501,
    Error,
    includedFileMissingImplementingDoYouMeanImport,
    "missing 'implementing' declaration in the included source file '$0'. The file declares that "
    "it defines module '$1', do you mean 'import' instead?")
DIAGNOSTIC(
    30502,
    Error,
    includedFileDoesNotImplementCurrentModule,
    "the included source file is expected to implement module '$0', but it is implementing '$1' "
    "instead.")
DIAGNOSTIC(
    30503,
    Error,
    primaryModuleFileCannotStartWithImplementingDecl,
    "a primary source file for a module cannot start with 'implementing'.")
DIAGNOSTIC(
    30504,
    Warning,
    primaryModuleFileMustStartWithModuleDecl,
    "a primary source file for a module should start with 'module'.")
DIAGNOSTIC(
    30505,
    Error,
    implementingMustReferencePrimaryModuleFile,
    "the source file referenced by 'implementing' must be a primary module file starting with a "
    "'module' declaration.")
DIAGNOSTIC(
    30506,
    Warning,
    moduleImplementationHasFileExtension,
    "implementing directive contains file extension in module name '$0'. Module names should not "
    "include extensions. The compiler will use '$1' as the module name.")

// Visibilty
DIAGNOSTIC(30600, Error, declIsNotVisible, "'$0' is not accessible from the current context.")
DIAGNOSTIC(
    30601,
    Error,
    declCannotHaveHigherVisibility,
    "'$0' cannot have a higher visibility than '$1'.")
DIAGNOSTIC(
    30602,
    Error,
    satisfyingDeclCannotHaveLowerVisibility,
    "'$0' is less visible than the interface requirement it satisfies.")
DIAGNOSTIC(
    30603,
    Error,
    invalidUseOfPrivateVisibility,
    "'$0' cannot have private visibility because it is not a member of a type.")
DIAGNOSTIC(30604, Error, useOfLessVisibleType, "'$0' references less visible type '$1'.")
DIAGNOSTIC(
    36005,
    Error,
    invalidVisibilityModifierOnTypeOfDecl,
    "visibility modifier is not allowed on '$0'.")

// Capability
DIAGNOSTIC(
    36100,
    Error,
    conflictingCapabilityDueToUseOfDecl,
    "'$0' requires capability '$1' that is conflicting with the '$2's current capability "
    "requirement '$3'.")
DIAGNOSTIC(
    36101,
    Error,
    conflictingCapabilityDueToStatement,
    "statement requires capability '$0' that is conflicting with the '$1's current capability "
    "requirement '$2'.")
DIAGNOSTIC(
    36102,
    Error,
    conflictingCapabilityDueToStatementEnclosingFunc,
    "statement requires capability '$0' that is conflicting with the current function's capability "
    "requirement '$1'.")
DIAGNOSTIC(
    36103,
    Warning,
    missingCapabilityRequirementOnPublicDecl,
    "public symbol '$0' is missing capability requirement declaration, the symbol is assumed to "
    "require inferred capabilities '$1'.")
DIAGNOSTIC(36104, Error, useOfUndeclaredCapability, "'$0' uses undeclared capability '$1'.")
DIAGNOSTIC(
    36104,
    Error,
    useOfUndeclaredCapabilityOfInterfaceRequirement,
    "'$0' uses capability '$1' that is missing from the interface requirement.")
DIAGNOSTIC(36105, Error, unknownCapability, "unknown capability name '$0'.")
DIAGNOSTIC(36106, Error, expectCapability, "expect a capability name.")
DIAGNOSTIC(
    36107,
    Error,
    entryPointUsesUnavailableCapability,
    "entrypoint '$0' uses features that are not available in '$2' stage for '$1' target.")
DIAGNOSTIC(
    36108,
    Error,
    declHasDependenciesNotCompatibleOnTarget,
    "'$0' has dependencies that are not compatible on the required target '$1'.")
DIAGNOSTIC(36109, Error, invalidTargetSwitchCase, "'$0' cannot be used as a target_switch case.")
DIAGNOSTIC(
    36110,
    Error,
    stageIsIncompatibleWithCapabilityDefinition,
    "'$0' is defined for stage '$1', which is incompatible with the declared capability set '$2'.")
DIAGNOSTIC(36111, Error, unexpectedCapability, "'$0' resolves into a disallowed `$1` Capability.")
DIAGNOSTIC(
    36112,
    Warning,
    entryPointAndProfileAreIncompatible,
    "'$0' is defined for stage '$1', which is incompatible with the declared profile '$2'.")
DIAGNOSTIC(
    36113,
    Warning,
    usingInternalCapabilityName,
    "'$0' resolves into a '_Internal' '_$1' Capability, use '$1' instead.")
DIAGNOSTIC(
    36114,
    Warning,
    incompatibleWithPrecompileLib,
    "Precompiled library requires '$0', has `$1`, implicitly upgrading capabilities.")
DIAGNOSTIC(
    36115,
    Error,
    incompatibleWithPrecompileLibRestrictive,
    "Precompiled library requires '$0', has `$1`.")
DIAGNOSTIC(
    36116,
    Error,
    capabilityHasMultipleStages,
    "Capability '$0' is targeting stages '$1', only allowed to use 1 unique stage here.")
DIAGNOSTIC(
    36117,
    Error,
    declHasDependenciesNotCompatibleOnStage,
    "'$0' uses features that are not available in '$1' stage.")

// Attributes
DIAGNOSTIC(31000, Warning, unknownAttributeName, "unknown attribute '$0'")
DIAGNOSTIC(
    31001,
    Error,
    attributeArgumentCountMismatch,
    "attribute '$0' expects $1 arguments ($2 provided)")
DIAGNOSTIC(31002, Error, attributeNotApplicable, "attribute '$0' is not valid here")

DIAGNOSTIC(
    31003,
    Error,
    badlyDefinedPatchConstantFunc,
    "hull shader '$0' has has badly defined 'patchconstantfunc' attribute.")

DIAGNOSTIC(31004, Error, expectedSingleIntArg, "attribute '$0' expects a single int argument")
DIAGNOSTIC(31005, Error, expectedSingleStringArg, "attribute '$0' expects a single string argument")

DIAGNOSTIC(
    31006,
    Error,
    attributeFunctionNotFound,
    "Could not find function '$0' for attribute'$1'")

DIAGNOSTIC(31007, Error, attributeExpectedIntArg, "attribute '$0' expects argument $1 to be int")
DIAGNOSTIC(
    31008,
    Error,
    attributeExpectedStringArg,
    "attribute '$0' expects argument $1 to be string")

DIAGNOSTIC(
    31009,
    Error,
    expectedSingleFloatArg,
    "attribute '$0' expects a single floating point argument")

DIAGNOSTIC(31100, Error, unknownStageName, "unknown stage name '$0'")
DIAGNOSTIC(31101, Error, unknownImageFormatName, "unknown image format '$0'")
DIAGNOSTIC(31101, Error, unknownDiagnosticName, "unknown diagnostic '$0'")
DIAGNOSTIC(
    31102,
    Error,
    nonPositiveNumThreads,
    "expected a positive integer in 'numthreads' attribute, got '$0'")
DIAGNOSTIC(
    31103,
    Error,
    invalidWaveSize,
    "expected a power of 2 between 4 and 128, inclusive, in 'WaveSize' attribute, got '$0'")
DIAGNOSTIC(
    31104,
    Warning,
    explicitUniformLocation,
    "Explicit binding of uniform locations is discouraged. Prefer 'ConstantBuffer<$0>' over "
    "'uniform $0'")
DIAGNOSTIC(
    31105,
    Warning,
    imageFormatUnsupportedByBackend,
    "Image format '$0' is not explicitly supported by the $1 backend, using supported format '$2' "
    "instead.")


DIAGNOSTIC(31120, Error, invalidAttributeTarget, "invalid syntax target for user defined attribute")

DIAGNOSTIC(31121, Error, anyValueSizeExceedsLimit, "'anyValueSize' cannot exceed $0")

DIAGNOSTIC(
    31122,
    Error,
    associatedTypeNotAllowInComInterface,
    "associatedtype not allowed in a [COM] interface")
DIAGNOSTIC(31123, Error, invalidGUID, "'$0' is not a valid GUID")
DIAGNOSTIC(
    31124,
    Error,
    structCannotImplementComInterface,
    "a struct type cannot implement a [COM] interface")
DIAGNOSTIC(
    31124,
    Error,
    interfaceInheritingComMustBeCom,
    "an interface type that inherits from a [COM] interface must itself be a [COM] interface")

DIAGNOSTIC(
    31130,
    Error,
    derivativeMemberAttributeMustNameAMemberInExpectedDifferentialType,
    "[DerivativeMember] must reference to a member in the associated differential type '$0'.")
DIAGNOSTIC(
    31131,
    Error,
    invalidUseOfDerivativeMemberAttributeParentTypeIsNotDifferentiable,
    "invalid use of [DerivativeMember], parent type is not differentiable.")
DIAGNOSTIC(
    31132,
    Error,
    derivativeMemberAttributeCanOnlyBeUsedOnMembers,
    "[DerivativeMember] is allowed on members only.")

DIAGNOSTIC(
    31140,
    Error,
    typeOfExternDeclMismatchesOriginalDefinition,
    "type of `extern` decl '$0' differs from its original definition. expected '$1'.")
DIAGNOSTIC(
    31141,
    Error,
    definitionOfExternDeclMismatchesOriginalDefinition,
    "`extern` decl '$0' is not consistent with its original definition.")
DIAGNOSTIC(
    31142,
    Error,
    ambiguousOriginalDefintionOfExternDecl,
    "`extern` decl '$0' has ambiguous original definitions.")
DIAGNOSTIC(
    31143,
    Error,
    missingOriginalDefintionOfExternDecl,
    "no original definition found for `extern` decl '$0'.")

DIAGNOSTIC(31145, Error, invalidCustomDerivative, "invalid custom derivative attribute.")
DIAGNOSTIC(31146, Error, declAlreadyHasAttribute, "'$0' already has attribute '[$1]'.")
DIAGNOSTIC(
    31147,
    Error,
    cannotResolveOriginalFunctionForDerivative,
    "cannot resolve the original function for the the custom derivative.")
DIAGNOSTIC(
    31148,
    Error,
    cannotResolveDerivativeFunction,
    "cannot resolve the custom derivative function")
DIAGNOSTIC(
    31149,
    Error,
    customDerivativeSignatureMismatchAtPosition,
    "invalid custom derivative. parameter type mismatch at position $0. expected '$1', got '$2'")
DIAGNOSTIC(
    31150,
    Error,
    customDerivativeSignatureMismatch,
    "invalid custom derivative. could not resolve function with expected signature '$0'")
DIAGNOSTIC(
    31151,
    Error,
    cannotResolveGenericArgumentForDerivativeFunction,
    "The generic arguments to the derivative function cannot be deduced from the parameter list of "
    "the original function. "
    "Consider using [ForwardDerivative], [BackwardDerivative] or [PrimalSubstitute] attributes on "
    "the primal function"
    " with explicit generic arguments to associate it with a generic derivative function. Note "
    "that [ForwardDerivativeOf], "
    "[BackwardDerivativeOf], and [PrimalSubstituteOf] attributes are not supported when the "
    "generic arguments to the derivatives cannot be automatically deduced.")
DIAGNOSTIC(
    31152,
    Error,
    cannotAssociateInterfaceRequirementWithDerivative,
    "cannot associate an interface requirement with a derivative.")
DIAGNOSTIC(
    31153,
    Error,
    cannotUseInterfaceRequirementAsDerivative,
    "cannot use an interface requirement as a derivative.")
DIAGNOSTIC(
    31154,
    Error,
    customDerivativeSignatureThisParamMismatch,
    "custom derivative does not match expected signature on `this`. Both original and derivative "
    "function must have the same `this` type.")
DIAGNOSTIC(
    31155,
    Error,
    customDerivativeNotAllowedForMemberFunctionsOfDifferentiableType,
    "custom derivative is not allowed for non-static member functions of a differentiable type.")
DIAGNOSTIC(
    31156,
    Error,
    customDerivativeExpectedStatic,
    "expected a static definition for the custom derivative.")
DIAGNOSTIC(
    31157,
    Error,
    overloadedFuncUsedWithDerivativeOfAttributes,
    "cannot resolve overloaded functions for derivative-of attributes.")
DIAGNOSTIC(
    31158,
    Error,
    primalSubstituteTargetMustHaveHigherDifferentiabilityLevel,
    "primal substitute function for differentiable method must also be differentiable. Use "
    "[Differentiable] or [TreatAsDifferentiable] (for empty derivatives)")

DIAGNOSTIC(31200, Warning, deprecatedUsage, "$0 has been deprecated: $1")
DIAGNOSTIC(31201, Error, modifierNotAllowed, "modifier '$0' is not allowed here.")
DIAGNOSTIC(
    31202,
    Error,
    duplicateModifier,
    "modifier '$0' is redundant or conflicting with existing modifier '$1'")
DIAGNOSTIC(31203, Error, cannotExportIncompleteType, "cannot export incomplete type '$0'")
DIAGNOSTIC(
    31204,
    Error,
    incompleteTypeCannotBeUsedInBuffer,
    "incomplete type '$0' cannot be used in a buffer")
DIAGNOSTIC(
    31205,
    Error,
    incompleteTypeCannotBeUsedInUniformParameter,
    "incomplete type '$0' cannot be used in a uniform parameter")
DIAGNOSTIC(
    31206,
    Error,
    memoryQualifierNotAllowedOnANonImageTypeParameter,
    "modifier $0 is not allowed on a non image type parameter.")
DIAGNOSTIC(
    31208,
    Error,
    requireInputDecoratedVarForParameter,
    "$0 expects for argument $1 a type which is a shader input (`in`) variable.")
DIAGNOSTIC(
    31210,
    Error,
    derivativeGroupQuadMustBeMultiple2ForXYThreads,
    "compute derivative group quad requires thread dispatch count of X and Y to each be at a "
    "multiple of 2")
DIAGNOSTIC(
    31211,
    Error,
    derivativeGroupLinearMustBeMultiple4ForTotalThreadCount,
    "compute derivative group linear requires total thread dispatch count to be at a multiple of 4")
DIAGNOSTIC(
    31212,
    Error,
    onlyOneOfDerivativeGroupLinearOrQuadCanBeSet,
    "cannot set compute derivative group linear and compute derivative group quad at the same time")
DIAGNOSTIC(
    31213,
    Error,
    cudaKernelMustReturnVoid,
    "return type of a CUDA kernel function cannot be non-void.")
DIAGNOSTIC(
    31214,
    Error,
    differentiableKernelEntryPointCannotHaveDifferentiableParams,
    "differentiable kernel entry point cannot have differentiable parameters. Consider using "
    "DiffTensorView to pass differentiable data, or marking this parameter with 'no_diff'")
DIAGNOSTIC(
    31215,
    Error,
    cannotUseUnsizedTypeInConstantBuffer,
    "cannot use unsized type '$0' in a constant buffer.")
DIAGNOSTIC(31216, Error, unrecognizedGLSLLayoutQualifier, "GLSL layout qualifier is unrecognized")
DIAGNOSTIC(
    31217,
    Error,
    unrecognizedGLSLLayoutQualifierOrRequiresAssignment,
    "GLSL layout qualifier is unrecognized or requires assignment")
DIAGNOSTIC(
    31218,
    Error,
    specializationConstantMustBeScalar,
    "specialization constant must be a scalar.")
DIAGNOSTIC(
    31219,
    Error,
    pushOrSpecializationConstantCannotBeStatic,
    "push or specialization constants cannot be 'static'.")
DIAGNOSTIC(
    31220,
    Error,
    variableCannotBePushAndSpecializationConstant,
    "'$0' cannot be a push constant and a specialization constant at the same time")
// Enums

DIAGNOSTIC(32000, Error, invalidEnumTagType, "invalid tag type for 'enum': '$0'")
DIAGNOSTIC(32003, Error, unexpectedEnumTagExpr, "unexpected form for 'enum' tag value expression")

// 303xx: interfaces and associated types
DIAGNOSTIC(
    30300,
    Error,
    assocTypeInInterfaceOnly,
    "'associatedtype' can only be defined in an 'interface'.")
DIAGNOSTIC(
    30301,
    Error,
    globalGenParamInGlobalScopeOnly,
    "'type_param' can only be defined global scope.")
DIAGNOSTIC(
    30302,
    Error,
    staticConstRequirementMustBeIntOrBool,
    "'static const' requirement can only have int or bool type.")
DIAGNOSTIC(
    30303,
    Error,
    valueRequirementMustBeCompileTimeConst,
    "requirement in the form of a simple value must be declared as 'static const'.")
DIAGNOSTIC(30310, Error, typeIsNotDifferentiable, "type '$0' is not differentiable.")

// Interop
DIAGNOSTIC(
    30400,
    Error,
    cannotDefinePtrTypeToManagedResource,
    "pointer to a managed resource is invalid, use `NativeRef<T>` instead")

// Control flow
DIAGNOSTIC(
    30500,
    Warning,
    forLoopSideEffectChangingDifferentVar,
    "the for loop initializes and checks variable '$0' but the side effect expression is modifying "
    "'$1'.")
DIAGNOSTIC(
    30501,
    Warning,
    forLoopPredicateCheckingDifferentVar,
    "the for loop initializes and modifies variable '$0' but the predicate expression is checking "
    "'$1'.")
DIAGNOSTIC(
    30502,
    Warning,
    forLoopChangingIterationVariableInOppsoiteDirection,
    "the for loop is modifiying variable '$0' in the opposite direction from loop exit condition.")
DIAGNOSTIC(
    30503,
    Warning,
    forLoopNotModifyingIterationVariable,
    "the for loop is not modifiying variable '$0' because the step size evaluates to 0.")
DIAGNOSTIC(
    30504,
    Warning,
    forLoopTerminatesInFewerIterationsThanMaxIters,
    "the for loop is statically determined to terminate within $0 iterations, which is less than "
    "what [MaxIters] specifies.")
DIAGNOSTIC(
    30505,
    Warning,
    loopRunsForZeroIterations,
    "the loop runs for 0 iterations and will be removed.")
DIAGNOSTIC(
    30510,
    Error,
    loopInDiffFuncRequireUnrollOrMaxIters,
    "loops inside a differentiable function need to provide either '[MaxIters(n)]' or "
    "'[ForceUnroll]' attribute.")

// Switch
DIAGNOSTIC(
    30600,
    Error,
    switchMultipleDefault,
    "multiple 'default' cases not allowed within a 'switch' statement")
DIAGNOSTIC(
    30601,
    Error,
    switchDuplicateCases,
    "duplicate cases not allowed within a 'switch' statement")

// TODO: need to assign numbers to all these extra diagnostics...
DIAGNOSTIC(39999, Fatal, cyclicReference, "cyclic reference '$0'.")
DIAGNOSTIC(
    39999,
    Error,
    cyclicReferenceInInheritance,
    "cyclic reference in inheritance graph '$0'.")

DIAGNOSTIC(
    39999,
    Error,
    localVariableUsedBeforeDeclared,
    "local variable '$0' is being used before its declaration.")
DIAGNOSTIC(
    39999,
    Error,
    variableUsedInItsOwnDefinition,
    "the initial-value expression for variable '$0' depends on the value of the variable itself")
DIAGNOSTIC(
    39901,
    Fatal,
    cannotProcessInclude,
    "internal compiler error: cannot process '__include' in the current semantic checking context.")

// 304xx: generics
DIAGNOSTIC(30400, Error, genericTypeNeedsArgs, "generic type '$0' used without argument")
DIAGNOSTIC(30401, Error, invalidTypeForConstraint, "type '$0' cannot be used as a constraint.")
DIAGNOSTIC(
    30402,
    Error,
    invalidConstraintSubType,
    "type '$0' is not a valid left hand side of a type constraint.")

// 305xx: initializer lists
DIAGNOSTIC(30500, Error, tooManyInitializers, "too many initializers (expected $0, got $1)")
DIAGNOSTIC(
    30501,
    Error,
    cannotUseInitializerListForArrayOfUnknownSize,
    "cannot use initializer list for array of statically unknown size '$0'")
DIAGNOSTIC(
    30502,
    Error,
    cannotUseInitializerListForVectorOfUnknownSize,
    "cannot use initializer list for vector of statically unknown size '$0'")
DIAGNOSTIC(
    30503,
    Error,
    cannotUseInitializerListForMatrixOfUnknownSize,
    "cannot use initializer list for matrix of statically unknown size '$0' rows")
DIAGNOSTIC(
    30504,
    Error,
    cannotUseInitializerListForType,
    "cannot use initializer list for type '$0'")
DIAGNOSTIC(
    30505,
    Error,
    cannotUseInitializerListForCoopVectorOfUnknownSize,
    "cannot use initializer list for CoopVector of statically unknown size '$0'")

// 3062x: variables
DIAGNOSTIC(
    30620,
    Error,
    varWithoutTypeMustHaveInitializer,
    "a variable declaration without an initial-value expression must be given an explicit type")
DIAGNOSTIC(
    30622,
    Error,
    ambiguousDefaultInitializerForType,
    "more than one default initializer was found for type '$0'")
DIAGNOSTIC(30623, Error, cannotHaveInitializer, "'$0' cannot have an initializer because it is $1")
DIAGNOSTIC(
    30623,
    Error,
    genericValueParameterMustHaveType,
    "a generic value parameter must be given an explicit type")

// 307xx: parameters
DIAGNOSTIC(
    30700,
    Error,
    outputParameterCannotHaveDefaultValue,
    "an 'out' or 'inout' parameter cannot have a default-value expression")

// 308xx: inheritance
DIAGNOSTIC(
    30810,
    Error,
    baseOfInterfaceMustBeInterface,
    "interface '$0' cannot inherit from non-interface type '$1'")
DIAGNOSTIC(
    30811,
    Error,
    baseOfStructMustBeStructOrInterface,
    "struct '$0' cannot inherit from type '$1' that is neither a struct nor an interface")
DIAGNOSTIC(
    30812,
    Error,
    baseOfEnumMustBeIntegerOrInterface,
    "enum '$0' cannot inherit from type '$1' that is neither an interface not a builtin integer "
    "type")
DIAGNOSTIC(
    30813,
    Error,
    baseOfExtensionMustBeInterface,
    "extension cannot inherit from non-interface type '$1'")
DIAGNOSTIC(
    30814,
    Error,
    baseOfClassMustBeClassOrInterface,
    "class '$0' cannot inherit from type '$1' that is neither a class nor an interface")
DIAGNOSTIC(30815, Error, circularityInExtension, "circular extension is not allowed.")

DIAGNOSTIC(
    30820,
    Error,
    baseStructMustBeListedFirst,
    "a struct type may only inherit from one other struct type, and that type must appear first in "
    "the list of bases")
DIAGNOSTIC(
    30821,
    Error,
    tagTypeMustBeListedFirst,
    "an unum type may only have a single tag type, and that type must be listed first in the list "
    "of bases")
DIAGNOSTIC(
    30822,
    Error,
    baseClassMustBeListedFirst,
    "a class type may only inherit from one other class type, and that type must appear first in "
    "the list of bases")

DIAGNOSTIC(
    30830,
    Error,
    cannotInheritFromExplicitlySealedDeclarationInAnotherModule,
    "cannot inherit from type '$0' marked 'sealed' in module '$1'")
DIAGNOSTIC(
    30831,
    Error,
    cannotInheritFromImplicitlySealedDeclarationInAnotherModule,
    "cannot inherit from type '$0' in module '$1' because it is implicitly 'sealed'; mark the base "
    "type 'open' to allow inheritance across modules")
DIAGNOSTIC(30832, Error, invalidTypeForInheritance, "type '$0' cannot be used for inheritance")

DIAGNOSTIC(
    30850,
    Error,
    invalidExtensionOnType,
    "type '$0' cannot be extended. `extension` can only be used to extend a nominal type.")
DIAGNOSTIC(30851, Error, invalidMemberTypeInExtension, "$0 cannot be a part of an `extension`")
DIAGNOSTIC(
    30852,
    Error,
    invalidExtensionOnInterface,
    "cannot extend interface type '$0'. consider using a generic extension: `extension<T:$0> T "
    "{...}`.")

// 309xx: subscripts
DIAGNOSTIC(
    30900,
    Error,
    multiDimensionalArrayNotSupported,
    "multi-dimensional array is not supported.")
// 310xx: properties

// 311xx: accessors

DIAGNOSTIC(
    31100,
    Error,
    accessorMustBeInsideSubscriptOrProperty,
    "an accessor declaration is only allowed inside a subscript or property declaration")

DIAGNOSTIC(
    31101,
    Error,
    nonSetAccessorMustNotHaveParams,
    "accessors other than 'set' must not have parameters")
DIAGNOSTIC(
    31102,
    Error,
    setAccessorMayNotHaveMoreThanOneParam,
    "a 'set' accessor may not have more than one parameter")
DIAGNOSTIC(
    31102,
    Error,
    setAccessorParamWrongType,
    "'set' parameter '$0' has type '$1' which does not match the expected type '$2'")

// 313xx: bit fields
DIAGNOSTIC(
    31300,
    Error,
    bitFieldTooWide,
    "bit-field size ($0) exceeds the width of its type $1 ($2)")
DIAGNOSTIC(31301, Error, bitFieldNonIntegral, "bit-field type ($0) must be an integral type")

// 39999 waiting to be placed in the right range

DIAGNOSTIC(
    39999,
    Error,
    expectedIntegerConstantWrongType,
    "expected integer constant (found: '$0')")
DIAGNOSTIC(
    39999,
    Error,
    expectedIntegerConstantNotConstant,
    "expression does not evaluate to a compile-time constant")
DIAGNOSTIC(
    39999,
    Error,
    expectedIntegerConstantNotLiteral,
    "could not extract value from integer constant")

DIAGNOSTIC(
    39999,
    Error,
    expectedRayTracingPayloadObjectAtLocationButMissing,
    "raytracing payload expected at location $0 but it is missing")

DIAGNOSTIC(
    39999,
    Error,
    noApplicableOverloadForNameWithArgs,
    "no overload for '$0' applicable to arguments of type $1")
DIAGNOSTIC(39999, Error, noApplicableWithArgs, "no overload applicable to arguments of type $0")

DIAGNOSTIC(
    39999,
    Error,
    ambiguousOverloadForNameWithArgs,
    "ambiguous call to '$0' with arguments of type $1")
DIAGNOSTIC(
    39999,
    Error,
    ambiguousOverloadWithArgs,
    "ambiguous call to overloaded operation with arguments of type $0")

DIAGNOSTIC(39999, Note, overloadCandidate, "candidate: $0")
DIAGNOSTIC(39999, Note, invisibleOverloadCandidate, "candidate (invisible): $0")

DIAGNOSTIC(39999, Note, moreOverloadCandidates, "$0 more overload candidates")

DIAGNOSTIC(39999, Error, caseOutsideSwitch, "'case' not allowed outside of a 'switch' statement")
DIAGNOSTIC(
    39999,
    Error,
    defaultOutsideSwitch,
    "'default' not allowed outside of a 'switch' statement")

DIAGNOSTIC(39999, Error, expectedAGeneric, "expected a generic when using '<...>' (found: '$0')")

DIAGNOSTIC(
    39999,
    Error,
    genericArgumentInferenceFailed,
    "could not specialize generic for arguments of type $0")

DIAGNOSTIC(39999, Error, ambiguousReference, "ambiguous reference to '$0'")
DIAGNOSTIC(39999, Error, ambiguousExpression, "ambiguous reference")

DIAGNOSTIC(39999, Error, declarationDidntDeclareAnything, "declaration does not declare anything")

DIAGNOSTIC(
    39999,
    Error,
    expectedPrefixOperator,
    "function called as prefix operator was not declared `__prefix`")
DIAGNOSTIC(
    39999,
    Error,
    expectedPostfixOperator,
    "function called as postfix operator was not declared `__postfix`")

DIAGNOSTIC(39999, Error, notEnoughArguments, "not enough arguments to call (got $0, expected $1)")
DIAGNOSTIC(39999, Error, tooManyArguments, "too many arguments to call (got $0, expected $1)")

DIAGNOSTIC(39999, Error, invalidIntegerLiteralSuffix, "invalid suffix '$0' on integer literal")
DIAGNOSTIC(
    39999,
    Error,
    invalidFloatingPointLiteralSuffix,
    "invalid suffix '$0' on floating-point literal")
DIAGNOSTIC(
    39999,
    Warning,
    integerLiteralTooLarge,
    "integer literal is too large to be represented in a signed integer type, interpreting as "
    "unsigned")

DIAGNOSTIC(
    39999,
    Warning,
    integerLiteralTruncated,
    "integer literal '$0' too large for type '$1' truncated to '$2'")
DIAGNOSTIC(
    39999,
    Warning,
    floatLiteralUnrepresentable,
    "$0 literal '$1' unrepresentable, converted to '$2'")
DIAGNOSTIC(
    39999,
    Warning,
    floatLiteralTooSmall,
    "'$1' is smaller than the smallest representable value for type $0, converted to '$2'")

DIAGNOSTIC(
    39999,
    Error,
    unableToFindSymbolInModule,
    "unable to find the mangled symbol '$0' in module '$1'")

DIAGNOSTIC(
    39999,
    Error,
    overloadedParameterToHigherOrderFunction,
    "passing overloaded functions to higher order functions is not supported")

DIAGNOSTIC(
    39999,
    Error,
    matrixColumnOrRowCountIsOne,
    "matrices with 1 column or row are not supported by the current code generation target")

// 38xxx

DIAGNOSTIC(
    38000,
    Error,
    entryPointFunctionNotFound,
    "no function found matching entry point name '$0'")
DIAGNOSTIC(
    38001,
    Error,
    ambiguousEntryPoint,
    "more than one function matches entry point name '$0'")
DIAGNOSTIC(
    38003,
    Error,
    entryPointSymbolNotAFunction,
    "entry point '$0' must be declared as a function")

DIAGNOSTIC(
    38004,
    Error,
    entryPointTypeParameterNotFound,
    "no type found matching entry-point type parameter name '$0'")
DIAGNOSTIC(
    38005,
    Error,
    expectedTypeForSpecializationArg,
    "expected a type as argument for specialization parameter '$0'")

DIAGNOSTIC(
    38006,
    Warning,
    specifiedStageDoesntMatchAttribute,
    "entry point '$0' being compiled for the '$1' stage has a '[shader(...)]' attribute that "
    "specifies the '$2' stage")
DIAGNOSTIC(
    38007,
    Error,
    entryPointHasNoStage,
    "no stage specified for entry point '$0'; use either a '[shader(\"name\")]' function attribute "
    "or the '-stage <name>' command-line option to specify a stage")

DIAGNOSTIC(
    38008,
    Error,
    specializationParameterOfNameNotSpecialized,
    "no specialization argument was provided for specialization parameter '$0'")
DIAGNOSTIC(
    38008,
    Error,
    specializationParameterNotSpecialized,
    "no specialization argument was provided for specialization parameter")

DIAGNOSTIC(
    38009,
    Error,
    expectedValueOfTypeForSpecializationArg,
    "expected a constant value of type '$0' as argument for specialization parameter '$1'")

DIAGNOSTIC(
    38100,
    Error,
    typeDoesntImplementInterfaceRequirement,
    "type '$0' does not provide required interface member '$1'")
DIAGNOSTIC(
    38105,
    Error,
    memberDoesNotMatchRequirementSignature,
    "member '$0' does not match interface requirement.")
DIAGNOSTIC(
    38101,
    Error,
    thisExpressionOutsideOfTypeDecl,
    "'this' expression can only be used in members of an aggregate type")
DIAGNOSTIC(
    38102,
    Error,
    initializerNotInsideType,
    "an 'init' declaration is only allowed inside a type or 'extension' declaration")
DIAGNOSTIC(
    38103,
    Error,
    thisTypeOutsideOfTypeDecl,
    "'This' type can only be used inside of an aggregate type")
DIAGNOSTIC(
    38104,
    Error,
    returnValNotAvailable,
    "cannot use '__return_val' here. '__return_val' is defined only in functions that return a "
    "non-copyable value.")
DIAGNOSTIC(
    38020,
    Error,
    mismatchEntryPointTypeArgument,
    "expecting $0 entry-point type arguments, provided $1.")
DIAGNOSTIC(
    38021,
    Error,
    typeArgumentForGenericParameterDoesNotConformToInterface,
    "type argument `$0` for generic parameter `$1` does not conform to interface `$2`.")

DIAGNOSTIC(
    38022,
    Error,
    cannotSpecializeGlobalGenericToItself,
    "the global type parameter '$0' cannot be specialized to itself")
DIAGNOSTIC(
    38023,
    Error,
    cannotSpecializeGlobalGenericToAnotherGenericParam,
    "the global type parameter '$0' cannot be specialized using another global type parameter "
    "('$1')")


DIAGNOSTIC(
    38024,
    Error,
    invalidDispatchThreadIDType,
    "parameter with SV_DispatchThreadID must be either scalar or vector (1 to 3) of uint/int but "
    "is $0")

DIAGNOSTIC(-1, Note, noteWhenCompilingEntryPoint, "when compiling entry point '$0'")

DIAGNOSTIC(
    38025,
    Error,
    mismatchSpecializationArguments,
    "expected $0 specialization arguments ($1 provided)")
DIAGNOSTIC(
    38026,
    Error,
    globalTypeArgumentDoesNotConformToInterface,
    "type argument `$1` for global generic parameter `$0` does not conform to interface `$2`.")

DIAGNOSTIC(
    38027,
    Error,
    mismatchExistentialSlotArgCount,
    "expected $0 existential slot arguments ($1 provided)")
DIAGNOSTIC(
    38029,
    Error,
    typeArgumentDoesNotConformToInterface,
    "type argument '$0' does not conform to the required interface '$1'")

DIAGNOSTIC(
    38031,
    Error,
    invalidUseOfNoDiff,
    "'no_diff' can only be used to decorate a call or a subscript operation")
DIAGNOSTIC(
    38032,
    Error,
    useOfNoDiffOnDifferentiableFunc,
    "use 'no_diff' on a call to a differentiable function has no meaning.")
DIAGNOSTIC(
    38033,
    Error,
    cannotUseNoDiffInNonDifferentiableFunc,
    "cannot use 'no_diff' in a non-differentiable function.")
DIAGNOSTIC(
    38034,
    Error,
    cannotUseConstRefOnDifferentiableParameter,
    "cannot use '__constref' on a differentiable parameter.")
DIAGNOSTIC(
    38034,
    Error,
    cannotUseConstRefOnDifferentiableMemberMethod,
    "cannot use '[constref]' on a differentiable member method of a differentiable type.")

DIAGNOSTIC(
    38040,
    Warning,
    nonUniformEntryPointParameterTreatedAsUniform,
    "parameter '$0' is treated as 'uniform' because it does not have a system-value semantic.")


DIAGNOSTIC(38200, Error, recursiveModuleImport, "module `$0` recursively imports itself")
DIAGNOSTIC(
    39999,
    Error,
    errorInImportedModule,
    "import of module '$0' failed because of a compilation error")

DIAGNOSTIC(
    38201,
    Error,
    glslModuleNotAvailable,
    "'glsl' module is not available from the current global session. To enable GLSL compatibility "
    "mode, specify 'SlangGlobalSessionDesc::enableGLSL' when creating the global session.")
DIAGNOSTIC(39999, Fatal, complationCeased, "compilation ceased")

DIAGNOSTIC(
    38202,
    Error,
    matrixWithDisallowedElementTypeEncountered,
    "matrix with disallowed element type '$0' encountered")

DIAGNOSTIC(
    38203,
    Error,
    vectorWithDisallowedElementTypeEncountered,
    "vector with disallowed element type '$0' encountered")

// 39xxx - Type layout and parameter binding.

DIAGNOSTIC(
    39000,
    Error,
    conflictingExplicitBindingsForParameter,
    "conflicting explicit bindings for parameter '$0'")
DIAGNOSTIC(
    39001,
    Warning,
    parameterBindingsOverlap,
    "explicit binding for parameter '$0' overlaps with parameter '$1'")


DIAGNOSTIC(
    39002,
    Error,
    shaderParameterDeclarationsDontMatch,
    "declarations of shader parameter '$0' in different translation units don't match")

DIAGNOSTIC(
    39003,
    Note,
    shaderParameterTypeMismatch,
    "type is declared as '$0' in one translation unit, and '$0' in another")
DIAGNOSTIC(
    39004,
    Note,
    fieldTypeMisMatch,
    "type of field '$0' is declared as '$1' in one translation unit, and '$2' in another")
DIAGNOSTIC(
    39005,
    Note,
    fieldDeclarationsDontMatch,
    "type '$0' is declared with different fields in each translation unit")
DIAGNOSTIC(39006, Note, usedInDeclarationOf, "used in declaration of '$0'")

DIAGNOSTIC(39007, Error, unknownRegisterClass, "unknown register class: '$0'")
DIAGNOSTIC(39008, Error, expectedARegisterIndex, "expected a register index after '$0'")
DIAGNOSTIC(39009, Error, expectedSpace, "expected 'space', got '$0'")
DIAGNOSTIC(39010, Error, expectedSpaceIndex, "expected a register space index after 'space'")
DIAGNOSTIC(39011, Error, invalidComponentMask, "invalid register component mask '$0'.")

DIAGNOSTIC(
    39013,
    Warning,
    registerModifierButNoVulkanLayout,
    "shader parameter '$0' has a 'register' specified for D3D, but no '[[vk::binding(...)]]` "
    "specified for Vulkan")
DIAGNOSTIC(
    39014,
    Error,
    unexpectedSpecifierAfterSpace,
    "unexpected specifier after register space: '$0'")
DIAGNOSTIC(
    39015,
    Error,
    wholeSpaceParameterRequiresZeroBinding,
    "shader parameter '$0' consumes whole descriptor sets, so the binding must be in the form "
    "'[[vk::binding(0, ...)]]'; the non-zero binding '$1' is not allowed")

DIAGNOSTIC(
    39016,
    Warning,
    hlslToVulkanMappingNotFound,
    "unable to infer Vulkan binding for '$0', automatic layout will be used")

DIAGNOSTIC(
    39017,
    Error,
    dontExpectOutParametersForStage,
    "the '$0' stage does not support `out` or `inout` entry point parameters")
DIAGNOSTIC(
    39018,
    Error,
    dontExpectInParametersForStage,
    "the '$0' stage does not support `in` entry point parameters")

DIAGNOSTIC(
    39019,
    Warning,
    globalUniformNotExpected,
    "'$0' is implicitly a global shader parameter, not a global variable. If a global variable is "
    "intended, add the 'static' modifier. If a uniform shader parameter is intended, add the "
    "'uniform' modifier to silence this warning.")

DIAGNOSTIC(
    39020,
    Error,
    tooManyShaderRecordConstantBuffers,
    "can have at most one 'shader record' attributed constant buffer; found $0.")

DIAGNOSTIC(
    39021,
    Error,
    typeParametersNotAllowedOnEntryPointGlobal,
    "local-root-signature shader parameter '$0' at global scope must not include "
    "existential/interface types")

DIAGNOSTIC(
    39022,
    Warning,
    vkIndexWithoutVkLocation,
    "ignoring '[[vk::index(...)]]` attribute without a corresponding '[[vk::location(...)]]' "
    "attribute")
DIAGNOSTIC(
    39023,
    Error,
    mixingImplicitAndExplicitBindingForVaryingParams,
    "mixing explicit and implicit bindings for varying parameters is not supported (see '$0' and "
    "'$1')")

DIAGNOSTIC(
    39024,
    Warning,
    cannotInferVulkanBindingWithoutRegisterModifier,
    "shader parameter '$0' doesn't have a 'register' specified, automatic layout will be used")

DIAGNOSTIC(
    39025,
    Error,
    conflictingVulkanInferredBindingForParameter,
    "conflicting vulkan inferred binding for parameter '$0' overlap is $1 and $2")

DIAGNOSTIC(
    39026,
    Error,
    matrixLayoutModifierOnNonMatrixType,
    "matrix layout modifier cannot be used on non-matrix type '$0'.")

DIAGNOSTIC(
    39027,
    Error,
    getAttributeAtVertexMustReferToPerVertexInput,
    "'GetAttributeAtVertex' must reference a vertex input directly, and the vertex input must be "
    "decorated with 'pervertex' or 'nointerpolation'.")

DIAGNOSTIC(
    39028,
    Error,
    notValidVaryingParameter,
    "parameter '$0' is not a valid varying parameter.")

DIAGNOSTIC(
    39029,
    Warning,
    registerModifierButNoVkBindingNorShift,
    "shader parameter '$0' has a 'register' specified for D3D, but no '[[vk::binding(...)]]` "
    "specified for Vulkan, nor is `-fvk-$1-shift` used.")

DIAGNOSTIC(
    39071,
    Warning,
    bindingAttributeIgnoredOnUniform,
    "binding attribute on uniform '$0' will be ignored since it will be packed into the default "
    "constant buffer at descriptor set 0 binding 0. To use explicit bindings, declare the uniform "
    "inside a constant buffer.")

//

// 4xxxx - IL code generation.
//
DIAGNOSTIC(
    40001,
    Error,
    bindingAlreadyOccupiedByComponent,
    "resource binding location '$0' is already occupied by component '$1'.")
DIAGNOSTIC(40002, Error, invalidBindingValue, "binding location '$0' is out of valid range.")
DIAGNOSTIC(
    40003,
    Error,
    bindingExceedsLimit,
    "binding location '$0' assigned to component '$1' exceeds maximum limit.")
DIAGNOSTIC(
    40004,
    Error,
    bindingAlreadyOccupiedByModule,
    "DescriptorSet ID '$0' is already occupied by module instance '$1'.")
DIAGNOSTIC(
    40005,
    Error,
    topLevelModuleUsedWithoutSpecifyingBinding,
    "top level module '$0' is being used without specifying binding location. Use [Binding: "
    "\"index\"] attribute to provide a binding location.")
DIAGNOSTIC(40006, Error, unimplementedSystemValueSemantic, "unknown system-value semantic '$0'")


DIAGNOSTIC(49999, Error, unknownSystemValueSemantic, "unknown system-value semantic '$0'")

DIAGNOSTIC(40006, Error, needCompileTimeConstant, "expected a compile-time constant")

DIAGNOSTIC(40007, Internal, irValidationFailed, "IR validation failed: $0")

DIAGNOSTIC(
    40008,
    Error,
    invalidLValueForRefParameter,
    "the form of this l-value argument is not valid for a `ref` parameter")

DIAGNOSTIC(
    40009,
    Error,
    dynamicInterfaceLacksAnyValueSizeAttribute,
    "interface '$0' is being used in dynamic dispatch code but has no [anyValueSize] attribute "
    "defined.")
DIAGNOSTIC(40010, Note, seeInterfaceUsage, "see usage of interface '$0'.")

DIAGNOSTIC(
    40011,
    Error,
    unconstrainedGenericParameterNotAllowedInDynamicFunction,
    "unconstrained generic paramter '$0' is not allowed in a dynamic function.")


DIAGNOSTIC(
    40020,
    Error,
    cannotUnrollLoop,
    "loop does not terminate within the limited number of iterations, unrolling is aborted.")

DIAGNOSTIC(
    40030,
    Fatal,
    functionNeverReturnsFatal,
    "function '$0' never returns, compilation ceased.")

// 41000 - IR-level validation issues

DIAGNOSTIC(41000, Warning, unreachableCode, "unreachable code detected")
DIAGNOSTIC(41001, Error, recursiveType, "type '$0' contains cyclic reference to itself.")

DIAGNOSTIC(
    41009,
    Error,
    missingReturnError,
    "non-void function must return in all cases for target '$0'")
DIAGNOSTIC(41010, Warning, missingReturn, "non-void function does not return in all cases")
DIAGNOSTIC(
    41011,
    Error,
    profileIncompatibleWithTargetSwitch,
    "__target_switch has no compatable target with current profile '$0'")
DIAGNOSTIC(
    41012,
    Warning,
    profileImplicitlyUpgraded,
    "entry point '$0' uses additional capabilities that are not part of the specified profile "
    "'$1'. The profile setting is automatically updated to include these capabilities: '$2'")
DIAGNOSTIC(
    41012,
    Error,
    profileImplicitlyUpgradedRestrictive,
    "entry point '$0' uses capabilities that are not part of the specified profile '$1'. Missing "
    "capabilities are: '$2'")
DIAGNOSTIC(41015, Warning, usingUninitializedOut, "use of uninitialized out parameter '$0'")
DIAGNOSTIC(41016, Warning, usingUninitializedVariable, "use of uninitialized variable '$0'")
DIAGNOSTIC(
    41017,
    Warning,
    usingUninitializedGlobalVariable,
    "use of uninitialized global variable '$0'")
DIAGNOSTIC(
    41018,
    Warning,
    returningWithUninitializedOut,
    "returning without initializing out parameter '$0'")
DIAGNOSTIC(
    41019,
    Warning,
    returningWithPartiallyUninitializedOut,
    "returning without fully initializing out parameter '$0'")
DIAGNOSTIC(
    41020,
    Warning,
    constructorUninitializedField,
    "exiting constructor without initializing field '$0'")
DIAGNOSTIC(
    41021,
    Warning,
    fieldNotDefaultInitialized,
    "default initializer for '$0' will not initialize field '$1'")
DIAGNOSTIC(41022, Warning, inOutNeverStoredInto, "inout parameter '$0' is never written to")
DIAGNOSTIC(
    41023,
    Warning,
    methodNeverMutates,
    "method marked `[mutable]` but never modifies `this`")

DIAGNOSTIC(
    41011,
    Error,
    typeDoesNotFitAnyValueSize,
    "type '$0' does not fit in the size required by its conforming interface.")
DIAGNOSTIC(-1, Note, typeAndLimit, "sizeof($0) is $1, limit is $2")
DIAGNOSTIC(
    41014,
    Error,
    typeCannotBePackedIntoAnyValue,
    "type '$0' contains fields that cannot be packed into ordinary bytes for dynamic dispatch.")
DIAGNOSTIC(
    41020,
    Error,
    lossOfDerivativeDueToCallOfNonDifferentiableFunction,
    "derivative cannot be propagated through call to non-$1-differentiable function `$0`, use "
    "'no_diff' to clarify intention.")
DIAGNOSTIC(
    41024,
    Error,
    lossOfDerivativeAssigningToNonDifferentiableLocation,
    "derivative is lost during assignment to non-differentiable location, use 'detach()' to "
    "clarify intention.")
DIAGNOSTIC(
    41025,
    Error,
    lossOfDerivativeUsingNonDifferentiableLocationAsOutArg,
    "derivative is lost when passing a non-differentiable location to an `out` or `inout` "
    "parameter, consider passing a temporary variable instead.")
DIAGNOSTIC(
    41021,
    Error,
    differentiableFuncMustHaveOutput,
    "a differentiable function must have at least one differentiable output.")
DIAGNOSTIC(
    41022,
    Error,
    differentiableFuncMustHaveInput,
    "a differentiable function must have at least one differentiable input.")
DIAGNOSTIC(
    41023,
    Error,
    getStringHashMustBeOnStringLiteral,
    "getStringHash can only be called when argument is statically resolvable to a string literal")

DIAGNOSTIC(
    41030,
    Warning,
    operatorShiftLeftOverflow,
    "left shift amount exceeds the number of bits and the result will be always zero, (`$0` << "
    "`$1`).")

DIAGNOSTIC(
    41901,
    Error,
    unsupportedUseOfLValueForAutoDiff,
    "unsupported use of L-value for auto differentiation.")
DIAGNOSTIC(
    41902,
    Error,
    cannotDifferentiateDynamicallyIndexedData,
    "cannot auto-differentiate mixed read/write access to dynamically indexed data in '$0'.")

DIAGNOSTIC(41903, Error, unableToSizeOf, "sizeof could not be performed for type '$0'.")
DIAGNOSTIC(41904, Error, unableToAlignOf, "alignof could not be performed for type '$0'.")

DIAGNOSTIC(
    42001,
    Error,
    invalidUseOfTorchTensorTypeInDeviceFunc,
    "invalid use of TorchTensor type in device/kernel functions. use `TensorView` instead.")

DIAGNOSTIC(
    42050,
    Warning,
    potentialIssuesWithPreferRecomputeOnSideEffectMethod,
    "$0 has [PreferRecompute] and may have side effects. side effects may execute multiple times. "
    "use [PreferRecompute(SideEffectBehavior.Allow)], or mark function with [__NoSideEffect]")

DIAGNOSTIC(45001, Error, unresolvedSymbol, "unresolved external symbol '$0'.")

DIAGNOSTIC(
    41201,
    Warning,
    expectDynamicUniformArgument,
    "argument for '$0' might not be a dynamic uniform, use `asDynamicUniform()` to silence this "
    "warning.")
DIAGNOSTIC(
    41201,
    Warning,
    expectDynamicUniformValue,
    "value stored at this location must be dynamic uniform, use `asDynamicUniform()` to silence "
    "this warning.")


DIAGNOSTIC(
    41202,
    Error,
    notEqualBitCastSize,
    "invalid to bit_cast differently sized types: '$0' with size '$1' casted into '$2' with size "
    "'$3'")

DIAGNOSTIC(
    41300,
    Error,
    byteAddressBufferUnaligned,
    "invalid alignment `$0` specified for the byte address buffer resource with the element size "
    "of `$1`")

DIAGNOSTIC(41400, Error, staticAssertionFailure, "static assertion failed, $0")
DIAGNOSTIC(41401, Error, staticAssertionFailureWithoutMessage, "static assertion failed.")
DIAGNOSTIC(
    41402,
    Error,
    staticAssertionConditionNotConstant,
    "condition for static assertion cannot be evaluated at compile time.")

DIAGNOSTIC(
    41402,
    Error,
    multiSampledTextureDoesNotAllowWrites,
    "cannot write to a multisampled texture with target '$0'.")

DIAGNOSTIC(
    41403,
    Error,
    invalidAtomicDestinationPointer,
    "cannot perform atomic operation because destination is neither groupshared nor from a device "
    "buffer.")

//
// 5xxxx - Target code generation.
//

DIAGNOSTIC(
    50010,
    Internal,
    missingExistentialBindingsForParameter,
    "missing argument for existential parameter slot")
DIAGNOSTIC(
    50011,
    Warning,
    spirvVersionNotSupported,
    "Slang's SPIR-V backend only supports SPIR-V version 1.3 and later."
    " Use `-emit-spirv-via-glsl` option to produce SPIR-V 1.0 through 1.2.")
DIAGNOSTIC(50020, Error, invalidTessCoordType, "TessCoord must have vec2 or vec3 type.")
DIAGNOSTIC(50020, Error, invalidFragCoordType, "FragCoord must be a vec4.")
DIAGNOSTIC(50020, Error, invalidInvocationIdType, "InvocationId must have int type.")
DIAGNOSTIC(50020, Error, invalidThreadIdType, "ThreadId must have int type.")
DIAGNOSTIC(50020, Error, invalidPrimitiveIdType, "PrimitiveId must have int type.")
DIAGNOSTIC(50020, Error, invalidPatchVertexCountType, "PatchVertexCount must have int type.")
DIAGNOSTIC(50022, Error, worldIsNotDefined, "world '$0' is not defined.")
DIAGNOSTIC(50023, Error, stageShouldProvideWorldAttribute, "'$0' should provide 'World' attribute.")
DIAGNOSTIC(
    50040,
    Error,
    componentHasInvalidTypeForPositionOutput,
    "'$0': component used as 'loc' output must be of vec4 type.")
DIAGNOSTIC(50041, Error, componentNotDefined, "'$0': component not defined.")

DIAGNOSTIC(
    50052,
    Error,
    domainShaderRequiresControlPointCount,
    "'DomainShader' requires attribute 'ControlPointCount'.")
DIAGNOSTIC(
    50052,
    Error,
    hullShaderRequiresControlPointCount,
    "'HullShader' requires attribute 'ControlPointCount'.")
DIAGNOSTIC(
    50052,
    Error,
    hullShaderRequiresControlPointWorld,
    "'HullShader' requires attribute 'ControlPointWorld'.")
DIAGNOSTIC(
    50052,
    Error,
    hullShaderRequiresCornerPointWorld,
    "'HullShader' requires attribute 'CornerPointWorld'.")
DIAGNOSTIC(50052, Error, hullShaderRequiresDomain, "'HullShader' requires attribute 'Domain'.")
DIAGNOSTIC(
    50052,
    Error,
    hullShaderRequiresInputControlPointCount,
    "'HullShader' requires attribute 'InputControlPointCount'.")
DIAGNOSTIC(
    50052,
    Error,
    hullShaderRequiresOutputTopology,
    "'HullShader' requires attribute 'OutputTopology'.")
DIAGNOSTIC(
    50052,
    Error,
    hullShaderRequiresPartitioning,
    "'HullShader' requires attribute 'Partitioning'.")
DIAGNOSTIC(
    50052,
    Error,
    hullShaderRequiresPatchWorld,
    "'HullShader' requires attribute 'PatchWorld'.")
DIAGNOSTIC(
    50052,
    Error,
    hullShaderRequiresTessLevelInner,
    "'HullShader' requires attribute 'TessLevelInner'.")
DIAGNOSTIC(
    50052,
    Error,
    hullShaderRequiresTessLevelOuter,
    "'HullShader' requires attribute 'TessLevelOuter'.")

DIAGNOSTIC(
    50053,
    Error,
    invalidTessellationDomian,
    "'Domain' should be either 'triangles' or 'quads'.")
DIAGNOSTIC(
    50053,
    Error,
    invalidTessellationOutputTopology,
    "'OutputTopology' must be one of: 'point', 'line', 'triangle_cw', or 'triangle_ccw'.")
DIAGNOSTIC(
    50053,
    Error,
    invalidTessellationPartitioning,
    "'Partitioning' must be one of: 'integer', 'pow2', 'fractional_even', or 'fractional_odd'.")
DIAGNOSTIC(
    50053,
    Error,
    invalidTessellationDomain,
    "'Domain' should be either 'triangles' or 'quads'.")

DIAGNOSTIC(
    50060,
    Error,
    invalidMeshStageOutputTopology,
    "Invalid mesh stage output topology '$0' for target '$1', must be one of: $2")

DIAGNOSTIC(
    50082,
    Error,
    importingFromPackedBufferUnsupported,
    "importing type '$0' from PackedBuffer is not supported by the GLSL backend.")
DIAGNOSTIC(
    51090,
    Error,
    cannotGenerateCodeForExternComponentType,
    "cannot generate code for extern component type '$0'.")
DIAGNOSTIC(51091, Error, typeCannotBePlacedInATexture, "type '$0' cannot be placed in a texture.")
DIAGNOSTIC(51092, Error, stageDoesntHaveInputWorld, "'$0' doesn't appear to have any input world")

DIAGNOSTIC(
    50100,
    Error,
    noTypeConformancesFoundForInterface,
    "No type conformances are found for interface '$0'. Code generation for current target "
    "requires at least one implementation type present in the linkage.")

DIAGNOSTIC(
    52000,
    Error,
    multiLevelBreakUnsupported,
    "control flow appears to require multi-level `break`, which Slang does not yet support")

DIAGNOSTIC(
    52001,
    Warning,
    dxilNotFound,
    "dxil shared library not found, so 'dxc' output cannot be signed! Shader code will not be "
    "runnable in non-development environments.")

DIAGNOSTIC(
    52002,
    Error,
    passThroughCompilerNotFound,
    "could not find a suitable pass-through compiler for '$0'.")
DIAGNOSTIC(52003, Error, cannotDisassemble, "cannot disassemble '$0'.")

DIAGNOSTIC(52004, Error, unableToWriteFile, "unable to write file '$0'")
DIAGNOSTIC(52005, Error, unableToReadFile, "unable to read file '$0'")

DIAGNOSTIC(
    52006,
    Error,
    compilerNotDefinedForTransition,
    "compiler not defined for transition '$0' to '$1'.")

DIAGNOSTIC(
    52007,
    Error,
    typeCannotBeUsedInDynamicDispatch,
    "failed to generate dynamic dispatch code for type '$0'.")
DIAGNOSTIC(
    52008,
    Error,
    dynamicDispatchOnSpecializeOnlyInterface,
    "type '$0' is marked for specialization only, but dynamic dispatch is needed for the call.")
DIAGNOSTIC(
    53001,
    Error,
    invalidTypeMarshallingForImportedDLLSymbol,
    "invalid type marshalling in imported func $0.")

DIAGNOSTIC(54001, Warning, meshOutputMustBeOut, "Mesh shader outputs must be declared with 'out'.")
DIAGNOSTIC(54002, Error, meshOutputMustBeArray, "HLSL style mesh shader outputs must be arrays")
DIAGNOSTIC(
    54003,
    Error,
    meshOutputArrayMustHaveSize,
    "HLSL style mesh shader output arrays must have a length specified")
DIAGNOSTIC(
    54004,
    Warning,
    unnecessaryHLSLMeshOutputModifier,
    "Unnecessary HLSL style mesh shader output modifier")

DIAGNOSTIC(
    55101,
    Error,
    invalidTorchKernelReturnType,
    "'$0' is not a valid return type for a pytorch kernel function.")
DIAGNOSTIC(
    55102,
    Error,
    invalidTorchKernelParamType,
    "'$0' is not a valid parameter type for a pytorch kernel function.")

DIAGNOSTIC(
    55200,
    Error,
    unsupportedBuiltinType,
    "'$0' is not a supported builtin type for the target.")
DIAGNOSTIC(
    55201,
    Error,
    unsupportedRecursion,
    "recursion detected in call to '$0', but the current code generation target does not allow "
    "recursion.")
DIAGNOSTIC(
    55202,
    Error,
    systemValueAttributeNotSupported,
    "system value semantic '$0' is not supported for the current target.")
DIAGNOSTIC(
    55203,
    Error,
    systemValueTypeIncompatible,
    "system value semantic '$0' should have type '$1' or be convertible to type '$1'.")
DIAGNOSTIC(
    55204,
    Error,
    unsupportedTargetIntrinsic,
    "intrinsic operation '$0' is not supported for the current target.")
DIAGNOSTIC(
    55205,
    Error,
    unsupportedSpecializationConstantForNumThreads,
    "Specialization constants are not supported in the 'numthreads' attribute for the current "
    "target.")
DIAGNOSTIC(
    56001,
    Error,
    unableToAutoMapCUDATypeToHostType,
    "Could not automatically map '$0' to a host type. Automatic binding generation failed for '$1'")
DIAGNOSTIC(
    56002,
    Error,
    attemptToQuerySizeOfUnsizedArray,
    "cannot obtain the size of an unsized array.")

DIAGNOSTIC(56003, Fatal, useOfUninitializedOpaqueHandle, "use of uninitialized opaque handle '$0'.")

// Metal
DIAGNOSTIC(
    56100,
    Error,
    constantBufferInParameterBlockNotAllowedOnMetal,
    "nested 'ConstantBuffer' inside a 'ParameterBlock' is not supported on Metal, use "
    "'ParameterBlock' instead.")
DIAGNOSTIC(
    56101,
    Error,
    resourceTypesInConstantBufferInParameterBlockNotAllowedOnMetal,
    "nesting a 'ConstantBuffer' containing resource types inside a 'ParameterBlock' is not "
    "supported on Metal, please use 'ParameterBlock' instead.")
DIAGNOSTIC(
    56102,
    Error,
    divisionByMatrixNotSupported,
    "division by matrix is not supported for Metal and WGSL targets.")

DIAGNOSTIC(57001, Warning, spirvOptFailed, "spirv-opt failed. $0")
DIAGNOSTIC(57002, Error, unknownPatchConstantParameter, "unknown patch constant parameter '$0'.")
DIAGNOSTIC(57003, Error, unknownTessPartitioning, "unknown tessellation partitioning '$0'.")
DIAGNOSTIC(
    57004,
    Error,
    outputSpvIsEmpty,
    "output SPIR-V contains no exported symbols. Please make sure to specify at least one "
    "entrypoint.")

// GLSL Compatibility
DIAGNOSTIC(
    58001,
    Error,
    entryPointMustReturnVoidWhenGlobalOutputPresent,
    "entry point must return 'void' when global output variables are present.")
DIAGNOSTIC(
    58002,
    Error,
    unhandledGLSLSSBOType,
    "Unhandled GLSL Shader Storage Buffer Object contents, unsized arrays as a final parameter "
    "must be the only parameter")

DIAGNOSTIC(
    58003,
    Error,
    inconsistentPointerAddressSpace,
    "'$0': use of pointer with inconsistent address space.")

// Autodiff checkpoint reporting
DIAGNOSTIC(
    -1,
    Note,
    reportCheckpointIntermediates,
    "checkpointing context of $1 bytes associated with function: '$0'")
DIAGNOSTIC(
    -1,
    Note,
    reportCheckpointVariable,
    "$0 bytes ($1) used to checkpoint the following item:")
DIAGNOSTIC(-1, Note, reportCheckpointCounter, "$0 bytes ($1) used for a loop counter here:")
DIAGNOSTIC(-1, Note, reportCheckpointNone, "no checkpoint contexts to report")

// 9xxxx - Documentation generation
DIAGNOSTIC(
    90001,
    Warning,
    ignoredDocumentationOnOverloadCandidate,
    "documentation comment on overload candidate '$0' is ignored")

//
// 8xxxx - Issues specific to a particular library/technology/platform/etc.
//

// 811xx - NVAPI

DIAGNOSTIC(
    81110,
    Error,
    nvapiMacroMismatch,
    "conflicting definitions for NVAPI macro '$0': '$1' and '$2'")

DIAGNOSTIC(
    81111,
    Error,
    opaqueReferenceMustResolveToGlobal,
    "could not determine register/space for a resource or sampler used with NVAPI")

// 99999 - Internal compiler errors, and not-yet-classified diagnostics.

DIAGNOSTIC(99999, Internal, unimplemented, "unimplemented feature in Slang compiler: $0")
DIAGNOSTIC(99999, Internal, unexpected, "unexpected condition encountered in Slang compiler: $0")
DIAGNOSTIC(99999, Internal, internalCompilerError, "Slang internal compiler error")
DIAGNOSTIC(99999, Error, compilationAborted, "Slang compilation aborted due to internal error")
DIAGNOSTIC(
    99999,
    Error,
    compilationAbortedDueToException,
    "Slang compilation aborted due to an exception of $0: $1")
DIAGNOSTIC(
    99999,
    Internal,
    serialDebugVerificationFailed,
    "Verification of serial debug information failed.")
DIAGNOSTIC(99999, Internal, spirvValidationFailed, "Validation of generated SPIR-V failed.")

DIAGNOSTIC(
    99999,
    Internal,
    noBlocksOrIntrinsic,
    "no blocks found for function definition, is there a '$0' intrinsic missing?")

//
// Ray tracing
//

DIAGNOSTIC(
    40000,
    Error,
    rayPayloadFieldMissingAccessQualifiers,
    "field '$0' in ray payload struct must have either 'read' OR 'write' access qualifiers")

#undef DIAGNOSTIC
