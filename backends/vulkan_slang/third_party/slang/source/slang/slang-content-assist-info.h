// slang-content-assist-info.h

#pragma once

#include "slang-syntax.h"
#include "slang.h"

namespace Slang
{

struct CompletionSuggestions
{
    enum class ScopeKind
    {
        Invalid,
        Member,
        Swizzle,
        Decl,
        Stmt,
        Expr,
        Attribute,
        HLSLSemantics,
        Capabilities
    };
    ScopeKind scopeKind = ScopeKind::Invalid;
    List<LookupResultItem> candidateItems;
    Type* swizzleBaseType = nullptr;
    IntegerLiteralValue elementCount[2] = {0, 0};

    void clear()
    {
        scopeKind = ScopeKind::Invalid;
        candidateItems.clear();
        elementCount[0] = 0;
        elementCount[1] = 0;
        swizzleBaseType = nullptr;
    }
};

struct MacroDefinitionContentAssistInfo
{
    struct Param
    {
        Name* name;
        bool isVariadic;
    };

    Name* name;
    SourceLoc loc;
    List<Param> params;
    List<Token> tokenList;
};

struct MacroInvocationContentAssistInfo
{
    Name* name;
    SourceLoc loc;
};

struct FileIncludeContentAssistInfo
{
    SourceLoc loc;
    int length;
    String path;
};

struct PreprocessorContentAssistInfo
{
    List<MacroDefinitionContentAssistInfo> macroDefinitions;
    List<MacroInvocationContentAssistInfo> macroInvocations;
    List<FileIncludeContentAssistInfo> fileIncludes;
};

enum class ContentAssistCheckingMode
{
    // Language server not enabled.
    None,

    // General full checking for semantic token/document symbol/goto-defintion features.
    General,

    // Checking for completion request only. Will ignore checking all function bodies
    // except for the function the user is editing.
    Completion
};

// This struct wraps all input/output data that is used by the language server to provide
// content assist support.
struct ContentAssistInfo
{
    // The mode the semantics checking should be operating on. Provided by the
    // language server.
    ContentAssistCheckingMode checkingMode = ContentAssistCheckingMode::None;
    // The primary module from which the current content assist request is made. Provided by the
    // language server.
    Name* primaryModuleName = nullptr;
    // The primary module path from which the current content assist request is made. Provided by
    // the language server.
    String primaryModulePath;
    // The cursor location at which a completion request is made. Provided by the language server.
    Index cursorLine = 0;
    // The cursor location at which a completion request is made. Provided by the language server.
    Index cursorCol = 0;

    // The result candidate items for a completion request. Filled in during semantics checking.
    CompletionSuggestions completionSuggestions;

    // The preprocessors definitions and invocations found during preprocessing. Filled in during
    // preprocessing.
    PreprocessorContentAssistInfo preprocessorInfo;
};

} // namespace Slang
