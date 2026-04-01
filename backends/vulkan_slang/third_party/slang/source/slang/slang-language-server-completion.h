// slang-language-server-completion.h
#pragma once

#include "../compiler-core/slang-language-server-protocol.h"
#include "slang-language-server-ast-lookup.h"
#include "slang-workspace-version.h"

namespace Slang
{
class LanguageServerCore;

enum class CommitCharacterBehavior
{
    Disabled,
    MembersOnly,
    All
};

struct CompletionResult
{
    List<LanguageServerProtocol::CompletionItem> items;
    List<LanguageServerProtocol::TextEditCompletionItem> textEditItems;
    CompletionResult() = default;
    CompletionResult(List<LanguageServerProtocol::CompletionItem>&& other)
        : items(_Move(other))
    {
    }
    CompletionResult(List<LanguageServerProtocol::TextEditCompletionItem>&& other)
        : textEditItems(_Move(other))
    {
    }
};

struct CompletionContext
{
    LanguageServerCore* server;
    Index cursorOffset;
    WorkspaceVersion* version;
    DocumentVersion* doc;
    Module* parsedModule;
    UnownedStringSlice canonicalPath;
    CommitCharacterBehavior commitCharacterBehavior;
    Int line;
    Int col;

    LanguageServerResult<CompletionResult> tryCompleteMemberAndSymbol();
    LanguageServerResult<CompletionResult> tryCompleteHLSLSemantic();
    LanguageServerResult<CompletionResult> tryCompleteAttributes();
    LanguageServerResult<CompletionResult> tryCompleteImport();
    LanguageServerResult<CompletionResult> tryCompleteInclude();
    LanguageServerResult<CompletionResult> tryCompleteRawFileName(
        UnownedStringSlice lineContent,
        Index fileNameStartPos,
        bool isImportString);


    CompletionResult collectMembersAndSymbols();
    void createSwizzleCandidates(
        List<LanguageServerProtocol::CompletionItem>& result,
        Type* type,
        IntegerLiteralValue elementCount[2]);
    CompletionResult createCapabilityCandidates();
    CompletionResult collectAttributes();
    LanguageServerProtocol::CompletionItem generateGUIDCompletionItem();
    CompletionResult gatherFileAndModuleCompletionItems(
        const String& prefixPath,
        bool translateModuleName,
        bool isImportString,
        Index lineIndex,
        Index fileNameEnd,
        Index sectionStart,
        Index sectionEnd,
        char closingChar);
};

} // namespace Slang
