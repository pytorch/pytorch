#pragma once
#include "../compiler-core/slang-json-rpc-connection.h"
#include "../compiler-core/slang-json-rpc.h"
#include "../core/slang-range.h"
#include "slang-language-server-auto-format.h"
#include "slang-language-server-completion.h"
#include "slang-language-server-inlay-hints.h"
#include "slang-workspace-version.h"
#include "slang.h"

#include <chrono>

namespace Slang
{
ArrayView<const char*> getCommitChars();

struct Command
{
    PersistentJSONValue id;
    String method;

    template<typename T>
    struct Optional
    {
    public:
        T* value = nullptr;
        bool isValid() { return value != nullptr; }
        T& operator=(const T& val)
        {
            delete value;
            value = new T(val);
            return *value;
        }
        T& operator=(Optional&& other)
        {
            if (other.isValid())
                *this = (other.get());
            other.value = nullptr;
            return *value;
        }
        T& get()
        {
            SLANG_ASSERT(isValid());
            return *value;
        }
        Optional() = default;
        Optional(const Optional& other)
        {
            if (other.isValid())
                *this = (other.get());
        }
        Optional(Optional&& other)
        {
            if (other.isValid())
                *this = (other.get());
            other.value = nullptr;
        }

        ~Optional() { delete value; }
    };

    Optional<LanguageServerProtocol::CompletionParams> completionArgs;
    Optional<LanguageServerProtocol::CompletionItem> completionResolveArgs;
    Optional<LanguageServerProtocol::TextEditCompletionItem> textEditCompletionResolveArgs;
    Optional<LanguageServerProtocol::DocumentSymbolParams> documentSymbolArgs;
    Optional<LanguageServerProtocol::InlayHintParams> inlayHintArgs;
    Optional<LanguageServerProtocol::DocumentFormattingParams> formattingArgs;
    Optional<LanguageServerProtocol::DocumentRangeFormattingParams> rangeFormattingArgs;
    Optional<LanguageServerProtocol::DocumentOnTypeFormattingParams> onTypeFormattingArgs;
    Optional<LanguageServerProtocol::DidChangeConfigurationParams> changeConfigArgs;
    Optional<LanguageServerProtocol::SignatureHelpParams> signatureHelpArgs;
    Optional<LanguageServerProtocol::DefinitionParams> definitionArgs;
    Optional<LanguageServerProtocol::SemanticTokensParams> semanticTokenArgs;
    Optional<LanguageServerProtocol::HoverParams> hoverArgs;
    Optional<LanguageServerProtocol::DidOpenTextDocumentParams> openDocArgs;
    Optional<LanguageServerProtocol::DidChangeTextDocumentParams> changeDocArgs;
    Optional<LanguageServerProtocol::DidCloseTextDocumentParams> closeDocArgs;
    Optional<LanguageServerProtocol::CancelParams> cancelArgs;
};

struct LanguageServerStartupOptions
{
    // Are we working with Visual Studio client?
    bool isVisualStudio = false;

    SLANG_API void parse(int argc, const char* const* argv);
};

class LanguageServerCore
{
public:
    enum class TraceOptions
    {
        Off,
        Messages,
        Verbose
    };
    CommitCharacterBehavior m_commitCharacterBehavior = CommitCharacterBehavior::MembersOnly;
    ComPtr<slang::IGlobalSession> m_session;
    RefPtr<Workspace> m_workspace;
    FormatOptions m_formatOptions;
    Slang::InlayHintOptions m_inlayHintOptions;
    List<LanguageServerProtocol::WorkspaceFolder> m_workspaceFolders;
    LanguageServerStartupOptions m_options;

    LanguageServerCore(LanguageServerStartupOptions options)
        : m_options(options)
    {
    }

    SlangResult init(const LanguageServerProtocol::InitializeParams& args);
    SlangResult didOpenTextDocument(const LanguageServerProtocol::DidOpenTextDocumentParams& args);
    SlangResult didCloseTextDocument(
        const LanguageServerProtocol::DidCloseTextDocumentParams& args);
    SlangResult didChangeTextDocument(
        const LanguageServerProtocol::DidChangeTextDocumentParams& args);
    LanguageServerResult<LanguageServerProtocol::Hover> hover(
        const LanguageServerProtocol::HoverParams& args);
    LanguageServerResult<List<LanguageServerProtocol::Location>> gotoDefinition(
        const LanguageServerProtocol::DefinitionParams& args);

    LanguageServerResult<CompletionResult> completion(
        const LanguageServerProtocol::CompletionParams& args);
    LanguageServerResult<LanguageServerProtocol::CompletionItem> completionResolve(
        const LanguageServerProtocol::CompletionItem& args,
        const LanguageServerProtocol::TextEditCompletionItem& editItem);
    LanguageServerResult<LanguageServerProtocol::SemanticTokens> semanticTokens(
        const LanguageServerProtocol::SemanticTokensParams& args);
    LanguageServerResult<LanguageServerProtocol::SignatureHelp> signatureHelp(
        const LanguageServerProtocol::SignatureHelpParams& args);
    LanguageServerResult<List<LanguageServerProtocol::DocumentSymbol>> documentSymbol(
        const LanguageServerProtocol::DocumentSymbolParams& args);
    LanguageServerResult<List<LanguageServerProtocol::InlayHint>> inlayHint(
        const LanguageServerProtocol::InlayHintParams& args);
    LanguageServerResult<List<LanguageServerProtocol::TextEdit>> formatting(
        const LanguageServerProtocol::DocumentFormattingParams& args);
    LanguageServerResult<List<LanguageServerProtocol::TextEdit>> rangeFormatting(
        const LanguageServerProtocol::DocumentRangeFormattingParams& args);
    LanguageServerResult<List<LanguageServerProtocol::TextEdit>> onTypeFormatting(
        const LanguageServerProtocol::DocumentOnTypeFormattingParams& args);
    String getExprDeclSignature(
        Expr* expr,
        String* outDocumentation,
        List<Slang::Range<Index>>* outParamRanges);
    String getDeclRefSignature(
        DeclRef<Decl> declRef,
        String* outDocumentation,
        List<Slang::Range<Index>>* outParamRanges);

private:
    slang::IGlobalSession* getOrCreateGlobalSession();

    FormatOptions getFormatOptions(Workspace* workspace, FormatOptions inOptions);
    LanguageServerResult<LanguageServerProtocol::Hover> tryGetMacroHoverInfo(
        WorkspaceVersion* version,
        DocumentVersion* doc,
        Index line,
        Index col);
    LanguageServerResult<List<LanguageServerProtocol::Location>> tryGotoMacroDefinition(
        WorkspaceVersion* version,
        DocumentVersion* doc,
        Index line,
        Index col);
    LanguageServerResult<List<LanguageServerProtocol::Location>> tryGotoFileInclude(
        WorkspaceVersion* version,
        DocumentVersion* doc,
        Index line);
};

class LanguageServer
{
private:
    static const int kConfigResponseId = 0x1213;

public:
    enum class TraceOptions
    {
        Off,
        Messages,
        Verbose
    };

    bool m_quit = false;
    LanguageServerCore m_core;
    RefPtr<JSONRPCConnection> m_connection;
    RttiTypeFuncsMap m_typeMap;
    bool m_initialized = false;
    TraceOptions m_traceOptions = TraceOptions::Off;
    std::chrono::time_point<std::chrono::system_clock> m_lastDiagnosticUpdateTime;
    Dictionary<String, String> m_lastPublishedDiagnostics;

    LanguageServer(LanguageServerStartupOptions options)
        : m_core(options)
    {
    }

    SlangResult init(const LanguageServerProtocol::InitializeParams& args);
    SlangResult execute();
    void update();
    void updateConfigFromJSON(const JSONValue& jsonVal);
    SlangResult didOpenTextDocument(const LanguageServerProtocol::DidOpenTextDocumentParams& args);
    SlangResult didCloseTextDocument(
        const LanguageServerProtocol::DidCloseTextDocumentParams& args);
    SlangResult didChangeTextDocument(
        const LanguageServerProtocol::DidChangeTextDocumentParams& args);
    SlangResult didChangeConfiguration(
        const LanguageServerProtocol::DidChangeConfigurationParams& args);
    SlangResult hover(const LanguageServerProtocol::HoverParams& args, const JSONValue& responseId);
    SlangResult gotoDefinition(
        const LanguageServerProtocol::DefinitionParams& args,
        const JSONValue& responseId);
    SlangResult completion(
        const LanguageServerProtocol::CompletionParams& args,
        const JSONValue& responseId);
    SlangResult completionResolve(
        const LanguageServerProtocol::CompletionItem& args,
        const LanguageServerProtocol::TextEditCompletionItem& editItem,
        const JSONValue& responseId);
    SlangResult semanticTokens(
        const LanguageServerProtocol::SemanticTokensParams& args,
        const JSONValue& responseId);
    SlangResult signatureHelp(
        const LanguageServerProtocol::SignatureHelpParams& args,
        const JSONValue& responseId);
    SlangResult documentSymbol(
        const LanguageServerProtocol::DocumentSymbolParams& args,
        const JSONValue& responseId);
    SlangResult inlayHint(
        const LanguageServerProtocol::InlayHintParams& args,
        const JSONValue& responseId);
    SlangResult formatting(
        const LanguageServerProtocol::DocumentFormattingParams& args,
        const JSONValue& responseId);
    SlangResult rangeFormatting(
        const LanguageServerProtocol::DocumentRangeFormattingParams& args,
        const JSONValue& responseId);
    SlangResult onTypeFormatting(
        const LanguageServerProtocol::DocumentOnTypeFormattingParams& args,
        const JSONValue& responseId);

private:
    SlangResult parseNextMessage();
    void resetDiagnosticUpdateTime();
    void publishDiagnostics();
    void updatePredefinedMacros(const JSONValue& macros);
    void updateSearchPaths(const JSONValue& value);
    void updateSearchInWorkspace(const JSONValue& value);
    void updateCommitCharacters(const JSONValue& value);
    void updateFormattingOptions(
        const JSONValue& clangFormatLoc,
        const JSONValue& clangFormatStyle,
        const JSONValue& clangFormatFallbackStyle,
        const JSONValue& allowLineBreakOnType,
        const JSONValue& allowLineBreakInRange);
    void updateInlayHintOptions(const JSONValue& deducedTypes, const JSONValue& parameterNames);
    void updateTraceOptions(const JSONValue& value);

    void sendConfigRequest();
    void registerCapability(const char* methodName);
    void logMessage(int type, String message);

    List<Command> commands;
    SlangResult queueJSONCall(JSONRPCCall call);
    SlangResult runCommand(Command& cmd);
    void processCommands();
};

inline bool _isIdentifierChar(char ch)
{
    return ch >= '0' && ch <= '9' || ch >= 'a' && ch <= 'z' || ch >= 'A' && ch <= 'Z' || ch == '_';
}

SLANG_API SlangResult runLanguageServer(LanguageServerStartupOptions options);
} // namespace Slang
