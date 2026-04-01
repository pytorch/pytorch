// Preprocessor.h
#ifndef SLANG_PREPROCESSOR_H_INCLUDED
#define SLANG_PREPROCESSOR_H_INCLUDED

#include "../compiler-core/slang-include-system.h"
#include "../compiler-core/slang-lexer.h"
#include "../core/slang-basic.h"

namespace Slang
{

class DiagnosticSink;
class Linkage;
struct PreprocessorContentAssistInfo;

enum class SourceLanguage : SlangSourceLanguageIntegral;

namespace preprocessor
{
struct Preprocessor;
}
using preprocessor::Preprocessor;

/// A handler for callbacks invoked by the preprocessor.
///
/// A client of the preprocessor can implement its own `PreprocessorHandler` subtype
/// in order to insert custom logic that implements higher-level policies
/// that the preprocessor shouldn't need to understand.
///
struct PreprocessorHandler
{
    virtual void handleEndOfTranslationUnit(Preprocessor* preprocessor);
    virtual void handleFileDependency(SourceFile* sourceFile);
};

/// Description of a preprocessor options/dependencies
struct PreprocessorDesc
{
    /// Required: sink to use when emitting preprocessor diagnostic messages
    DiagnosticSink* sink = nullptr;

    /// Required: name pool to use when creating `Name`s from strings
    NamePool* namePool = nullptr;

    /// Required: file system to use when looking up files
    ISlangFileSystemExt* fileSystem = nullptr;

    /// Required: source manager to use when loading source files
    SourceManager* sourceManager = nullptr;

    /// Optional: include system to use when resolving `#include` directives
    IncludeSystem* includeSystem = nullptr;

    /// Optional: preprocessor `#define`s to assume are set on input
    Dictionary<String, String> const* defines = nullptr;

    /// Optional: handler for callbacks invoked during preprocessing
    PreprocessorHandler* handler = nullptr;

    /// Optional: additional information for code assist.
    PreprocessorContentAssistInfo* contentAssistInfo = nullptr;
};

/// Take a source `file` and preprocess it into a list of tokens.
TokenList preprocessSource(
    SourceFile* file,
    PreprocessorDesc const& desc,
    SourceLanguage& outDetectedLanguage);

/// Convenience wrapper for `preprocessSource` when a `Linkage` is available
TokenList preprocessSource(
    SourceFile* file,
    DiagnosticSink* sink,
    IncludeSystem* includeSystem,
    Dictionary<String, String> const& defines,
    Linkage* linkage,
    SourceLanguage& outDetectedLanguage,
    PreprocessorHandler* handler = nullptr);

// The following functions are intended to be used inside of implementations
// of the `PreprocessorHandler` interface, in order to query the current
// state of the preprocessor.

/// Try to look up a macro with the given `macroName` and produce its value as a string
Result findMacroValue(
    Preprocessor* preprocessor,
    char const* macroName,
    String& outValue,
    SourceLoc& outLoc);

} // namespace Slang

#endif
