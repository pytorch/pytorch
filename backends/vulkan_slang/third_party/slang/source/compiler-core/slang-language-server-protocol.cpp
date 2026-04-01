#include "slang-language-server-protocol.h"

namespace Slang
{
namespace LanguageServerProtocol
{
static const StructRttiInfo _makeTextDocumentSyncOptionsRtti()
{
    TextDocumentSyncOptions obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::TextDocumentSyncOptions", nullptr);
    builder.addField("change", &obj.change);
    builder.addField("openClose", &obj.openClose);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo TextDocumentSyncOptions::g_rttiInfo = _makeTextDocumentSyncOptionsRtti();

static const StructRttiInfo _makeWorkDoneProgressParamsRtti()
{
    WorkDoneProgressParams obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::WorkDoneProgressParams", nullptr);
    builder.addField("workDoneToken", &obj.workDoneToken, StructRttiInfo::Flag::Optional);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo WorkDoneProgressParams::g_rttiInfo = _makeWorkDoneProgressParamsRtti();

static const StructRttiInfo _makeInlayHintOptionsRtti()
{
    InlayHintOptions obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::InlayHintOptions", nullptr);
    builder.addField("resolveProvider", &obj.resolveProvider);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo InlayHintOptions::g_rttiInfo = _makeInlayHintOptionsRtti();

static const StructRttiInfo _makeDocumentOnTypeFormattingOptionsRtti()
{
    DocumentOnTypeFormattingOptions obj;
    StructRttiBuilder builder(
        &obj,
        "LanguageServerProtocol::DocumentOnTypeFormattingOptions",
        nullptr);
    builder.addField("firstTriggerCharacter", &obj.firstTriggerCharacter);
    builder.addField("moreTriggerCharacter", &obj.moreTriggerCharacter);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo DocumentOnTypeFormattingOptions::g_rttiInfo =
    _makeDocumentOnTypeFormattingOptionsRtti();

static const StructRttiInfo _makeCompletionOptionsRtti()
{
    CompletionOptions obj;
    StructRttiBuilder builder(
        &obj,
        "LanguageServerProtocol::CompletionOptions",
        &WorkDoneProgressParams::g_rttiInfo);
    builder.addField("triggerCharacters", &obj.triggerCharacters);
    builder.addField("resolveProvider", &obj.resolveProvider);
    builder.addField("allCommitCharacters", &obj.allCommitCharacters);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo CompletionOptions::g_rttiInfo = _makeCompletionOptionsRtti();

static const StructRttiInfo _makeSemanticTokensLegendRtti()
{
    SemanticTokensLegend obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::SemanticTokensLegend", nullptr);
    builder.addField("tokenTypes", &obj.tokenTypes);
    builder.addField("tokenModifiers", &obj.tokenModifiers);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo SemanticTokensLegend::g_rttiInfo = _makeSemanticTokensLegendRtti();

static const StructRttiInfo _makeSemanticTokensOptionsRtti()
{
    SemanticTokensOptions obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::SemanticTokensOptions", nullptr);
    builder.addField("legend", &obj.legend);
    builder.addField("range", &obj.range);
    builder.addField("full", &obj.full);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo SemanticTokensOptions::g_rttiInfo = _makeSemanticTokensOptionsRtti();

static const StructRttiInfo _makeSignatureHelpOptionsRtti()
{
    SignatureHelpOptions obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::SignatureHelpOptions", nullptr);
    builder.addField("triggerCharacters", &obj.triggerCharacters);
    builder.addField("retriggerCharacters", &obj.retriggerCharacters);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo SignatureHelpOptions::g_rttiInfo = _makeSignatureHelpOptionsRtti();

static const StructRttiInfo _makeTextDocumentItemRtti()
{
    TextDocumentItem obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::TextDocumentItem", nullptr);
    builder.addField("uri", &obj.uri);
    builder.addField("version", &obj.version);
    builder.addField("languageId", &obj.languageId);
    builder.addField("text", &obj.text);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo TextDocumentItem::g_rttiInfo = _makeTextDocumentItemRtti();

static const StructRttiInfo _makeTextDocumentIdentifierRtti()
{
    TextDocumentIdentifier obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::TextDocumentIdentifier", nullptr);
    builder.addField("uri", &obj.uri);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo TextDocumentIdentifier::g_rttiInfo = _makeTextDocumentIdentifierRtti();

static const StructRttiInfo _makeVersionedTextDocumentIdentifierRtti()
{
    VersionedTextDocumentIdentifier obj;
    StructRttiBuilder builder(
        &obj,
        "LanguageServerProtocol::VersionedTextDocumentIdentifier",
        nullptr);
    builder.addField("uri", &obj.uri);
    builder.addField("version", &obj.version);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo VersionedTextDocumentIdentifier::g_rttiInfo =
    _makeVersionedTextDocumentIdentifierRtti();

static const StructRttiInfo _makePositionRtti()
{
    Position obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::Position", nullptr);
    builder.addField("line", &obj.line);
    builder.addField("character", &obj.character);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo Position::g_rttiInfo = _makePositionRtti();

static const StructRttiInfo _makeRangeRtti()
{
    Range obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::Range", nullptr);
    builder.addField("start", &obj.start);
    builder.addField("end", &obj.end);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo Range::g_rttiInfo = _makeRangeRtti();

static const StructRttiInfo _makeTextEditRtti()
{
    TextEdit obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::TextEdit", nullptr);
    builder.addField("range", &obj.range);
    builder.addField("newText", &obj.newText);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo TextEdit::g_rttiInfo = _makeTextEditRtti();

static const StructRttiInfo _makeDidOpenTextDocumentRtti()
{
    DidOpenTextDocumentParams obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::DidOpenTextDocumentParams", nullptr);
    builder.addField("textDocument", &obj.textDocument);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo DidOpenTextDocumentParams::g_rttiInfo = _makeDidOpenTextDocumentRtti();
const UnownedStringSlice DidOpenTextDocumentParams::methodName =
    UnownedStringSlice::fromLiteral("textDocument/didOpen");

static const StructRttiInfo _makeTextDocumentContentChangeEventRtti()
{
    TextDocumentContentChangeEvent obj;
    StructRttiBuilder builder(
        &obj,
        "LanguageServerProtocol::TextDocumentContentChangeEvent",
        nullptr);
    builder.addField("range", &obj.range, StructRttiInfo::Flag::Optional);
    builder.addField("text", &obj.text);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo TextDocumentContentChangeEvent::g_rttiInfo =
    _makeTextDocumentContentChangeEventRtti();

static const StructRttiInfo _makeDidChangeTextDocumentParamsRtti()
{
    DidChangeTextDocumentParams obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::DidChangeTextDocumentParams", nullptr);
    builder.addField("textDocument", &obj.textDocument);
    builder.addField("contentChanges", &obj.contentChanges);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo DidChangeTextDocumentParams::g_rttiInfo =
    _makeDidChangeTextDocumentParamsRtti();
const UnownedStringSlice DidChangeTextDocumentParams::methodName =
    UnownedStringSlice::fromLiteral("textDocument/didChange");


static const StructRttiInfo _makeDidCloseTextDocumentParamsRtti()
{
    DidCloseTextDocumentParams obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::DidCloseTextDocumentParams", nullptr);
    builder.addField("textDocument", &obj.textDocument);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo DidCloseTextDocumentParams::g_rttiInfo = _makeDidCloseTextDocumentParamsRtti();
const UnownedStringSlice DidCloseTextDocumentParams::methodName =
    UnownedStringSlice::fromLiteral("textDocument/didClose");

static const StructRttiInfo _makeWorkspaceFoldersServerCapabilitiesRtti()
{
    WorkspaceFoldersServerCapabilities obj;
    StructRttiBuilder builder(
        &obj,
        "LanguageServerProtocol::WorkspaceFoldersServerCapabilities",
        nullptr);
    builder.addField("supported", &obj.supported);
    builder.addField("changeNotifications", &obj.changeNotifications);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo WorkspaceFoldersServerCapabilities::g_rttiInfo =
    _makeWorkspaceFoldersServerCapabilitiesRtti();

static const StructRttiInfo _makeWorkspaceCapabilitiesRtti()
{
    WorkspaceCapabilities obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::WorkspaceCapabilities", nullptr);
    builder.addField("workspaceFolders", &obj.workspaceFolders);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo WorkspaceCapabilities::g_rttiInfo = _makeWorkspaceCapabilitiesRtti();

static const StructRttiInfo _makeServerCapabilitiesRtti()
{
    ServerCapabilities obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::ServerCapabilities", nullptr);
    builder.addField("positionEncoding", &obj.positionEncoding);
    builder.addField("textDocumentSync", &obj.textDocumentSync);
    builder.addField("workspace", &obj.workspace);
    builder.addField("hoverProvider", &obj.hoverProvider);
    builder.addField("inlayHintProvider", &obj.inlayHintProvider);
    builder.addField("documentOnTypeFormattingProvider", &obj.documentOnTypeFormattingProvider);
    builder.addField("documentFormattingProvider", &obj.documentFormattingProvider);
    builder.addField("documentRangeFormattingProvider", &obj.documentRangeFormattingProvider);
    builder.addField("definitionProvider", &obj.definitionProvider);
    builder.addField("completionProvider", &obj.completionProvider);
    builder.addField("semanticTokensProvider", &obj.semanticTokensProvider);
    builder.addField("signatureHelpProvider", &obj.signatureHelpProvider);
    builder.addField("documentSymbolProvider", &obj.documentSymbolProvider);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo ServerCapabilities::g_rttiInfo = _makeServerCapabilitiesRtti();

static const StructRttiInfo _makeVSServerCapabilitiesRtti()
{
    VSServerCapabilities obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::ServerCapabilities", nullptr);
    builder.addField("positionEncoding", &obj.positionEncoding);
    builder.addField("textDocumentSync", &obj.textDocumentSync);
    builder.addField("workspace", &obj.workspace);
    builder.addField("hoverProvider", &obj.hoverProvider);
    builder.addField("inlayHintProvider", &obj.inlayHintProvider);
    builder.addField("documentOnTypeFormattingProvider", &obj.documentOnTypeFormattingProvider);
    builder.addField("documentFormattingProvider", &obj.documentFormattingProvider);
    builder.addField("documentRangeFormattingProvider", &obj.documentRangeFormattingProvider);
    builder.addField("definitionProvider", &obj.definitionProvider);
    builder.addField("completionProvider", &obj.completionProvider);
    builder.addField("semanticTokensProvider", &obj.semanticTokensProvider);
    builder.addField("signatureHelpProvider", &obj.signatureHelpProvider);
    builder.addField("documentSymbolProvider", &obj.documentSymbolProvider);
    builder.addField("_vs_projectContextProvider", &obj._vs_projectContextProvider);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo VSServerCapabilities::g_rttiInfo = _makeVSServerCapabilitiesRtti();

static const StructRttiInfo _makeServerInfoRtti()
{
    ServerInfo obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::ServerInfo", nullptr);
    builder.addField("name", &obj.name);
    builder.addField("version", &obj.version);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo ServerInfo::g_rttiInfo = _makeServerInfoRtti();


static const StructRttiInfo _makeInitializeResultRtti()
{
    InitializeResult obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::InitializeResult", nullptr);
    builder.addField("capabilities", &obj.capabilities);
    builder.addField("serverInfo", &obj.serverInfo);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo InitializeResult::g_rttiInfo = _makeInitializeResultRtti();

static const StructRttiInfo _makeVSInitializeResultRtti()
{
    VSInitializeResult obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::VSInitializeResult", nullptr);
    builder.addField("capabilities", &obj.capabilities);
    builder.addField("serverInfo", &obj.serverInfo);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo VSInitializeResult::g_rttiInfo = _makeVSInitializeResultRtti();

const UnownedStringSlice InitializeParams::methodName =
    UnownedStringSlice::fromLiteral("initialize");

const UnownedStringSlice ShutdownParams::methodName = UnownedStringSlice::fromLiteral("shutdown");

const UnownedStringSlice ExitParams::methodName = UnownedStringSlice::fromLiteral("exit");

static const StructRttiInfo _makeWorkspaceFolderRtti()
{
    WorkspaceFolder obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::WorkspaceFolder", nullptr);
    builder.addField("uri", &obj.uri);
    builder.addField("name", &obj.name);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo WorkspaceFolder::g_rttiInfo = _makeWorkspaceFolderRtti();

static const StructRttiInfo _makeInitializeParamsRtti()
{
    InitializeParams obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::InitializeParams", nullptr);
    builder.addField("workspaceFolders", &obj.workspaceFolders, StructRttiInfo::Flag::Optional);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo InitializeParams::g_rttiInfo = _makeInitializeParamsRtti();

static const StructRttiInfo _makeNullResponseRtti()
{
    NullResponse obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::NullResponse", nullptr);
    return builder.make();
}
const StructRttiInfo NullResponse::g_rttiInfo = _makeNullResponseRtti();

NullResponse* NullResponse::get()
{
    static NullResponse result = {};
    return &result;
}

static const StructRttiInfo _makeLocationRtti()
{
    Location obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::Location", nullptr);
    builder.addField("uri", &obj.uri);
    builder.addField("range", &obj.range);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo Location::g_rttiInfo = _makeLocationRtti();

static const StructRttiInfo _makeDiagnosticRelatedInformationRtti()
{
    DiagnosticRelatedInformation obj;
    StructRttiBuilder builder(
        &obj,
        "LanguageServerProtocol::DiagnosticRelatedInformation",
        nullptr);
    builder.addField("location", &obj.location);
    builder.addField("message", &obj.message);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo DiagnosticRelatedInformation::g_rttiInfo =
    _makeDiagnosticRelatedInformationRtti();

static const StructRttiInfo _makeDiagnosticRtti()
{
    Diagnostic obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::Diagnostic", nullptr);
    builder.addField("code", &obj.code);
    builder.addField("message", &obj.message);
    builder.addField("range", &obj.range);
    builder.addField("relatedInformation", &obj.relatedInformation);
    builder.addField("severity", &obj.severity);
    builder.addField("source", &obj.source);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo Diagnostic::g_rttiInfo = _makeDiagnosticRtti();

static const StructRttiInfo _makePublishDiagnosticsParamsRtti()
{
    PublishDiagnosticsParams obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::PublishDiagnosticsParams", nullptr);
    builder.addField("uri", &obj.uri);
    builder.addField("diagnostics", &obj.diagnostics);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo PublishDiagnosticsParams::g_rttiInfo = _makePublishDiagnosticsParamsRtti();

static const StructRttiInfo _makeTextDocumentPositionParamsRtti()
{
    TextDocumentPositionParams obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::TextDocumentPositionParams", nullptr);
    builder.addField("textDocument", &obj.textDocument);
    builder.addField("position", &obj.position);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo TextDocumentPositionParams::g_rttiInfo = _makeTextDocumentPositionParamsRtti();

static const StructRttiInfo _makeHoverParamsRtti()
{
    HoverParams obj;
    StructRttiBuilder builder(
        &obj,
        "LanguageServerProtocol::HoverParams",
        &WorkDoneProgressParams::g_rttiInfo);
    builder.addField("textDocument", &obj.textDocument);
    builder.addField("position", &obj.position);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo HoverParams::g_rttiInfo = _makeHoverParamsRtti();
const UnownedStringSlice HoverParams::methodName =
    UnownedStringSlice::fromLiteral("textDocument/hover");

static const StructRttiInfo _makeMarkupContentRtti()
{
    MarkupContent obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::MarkupContent", nullptr);
    builder.addField("kind", &obj.kind);
    builder.addField("value", &obj.value);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo MarkupContent::g_rttiInfo = _makeMarkupContentRtti();

static const StructRttiInfo _makeHoverRtti()
{
    Hover obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::Hover", nullptr);
    builder.addField("contents", &obj.contents);
    builder.addField("range", &obj.range);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo Hover::g_rttiInfo = _makeHoverRtti();

static const StructRttiInfo _makeCompletionContextRtti()
{
    CompletionContext obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::CompletionContext", nullptr);
    builder.addField("triggerKind", &obj.triggerKind);
    builder.addField("triggerCharacter", &obj.triggerCharacter, StructRttiInfo::Flag::Optional);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo CompletionContext::g_rttiInfo = _makeCompletionContextRtti();

static const StructRttiInfo _makeDefinitionParamsRtti()
{
    DefinitionParams obj;
    StructRttiBuilder builder(
        &obj,
        "LanguageServerProtocol::DefinitionParams",
        &WorkDoneProgressParams::g_rttiInfo);
    builder.addField("textDocument", &obj.textDocument);
    builder.addField("position", &obj.position);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo DefinitionParams::g_rttiInfo = _makeDefinitionParamsRtti();
const UnownedStringSlice DefinitionParams::methodName =
    UnownedStringSlice::fromLiteral("textDocument/definition");

static const StructRttiInfo _makeCompletionParamsRtti()
{
    CompletionParams obj;
    StructRttiBuilder builder(
        &obj,
        "LanguageServerProtocol::CompletionParams",
        &WorkDoneProgressParams::g_rttiInfo);
    builder.addField("textDocument", &obj.textDocument);
    builder.addField("position", &obj.position);
    builder.addField("context", &obj.context, StructRttiInfo::Flag::Optional);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo CompletionParams::g_rttiInfo = _makeCompletionParamsRtti();
const UnownedStringSlice CompletionParams::methodName =
    UnownedStringSlice::fromLiteral("textDocument/completion");

static const StructRttiInfo _makeCompletionItemRtti()
{
    CompletionItem obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::CompletionItem", nullptr);
    builder.addField("label", &obj.label, StructRttiInfo::Flag::Optional);
    builder.addField("detail", &obj.detail, StructRttiInfo::Flag::Optional);
    builder.addField("kind", &obj.kind, StructRttiInfo::Flag::Optional);
    builder.addField("documentation", &obj.documentation, StructRttiInfo::Flag::Optional);
    builder.addField("data", &obj.data, StructRttiInfo::Flag::Optional);
    builder.addField("commitCharacters", &obj.commitCharacters, StructRttiInfo::Flag::Optional);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo CompletionItem::g_rttiInfo = _makeCompletionItemRtti();

static const StructRttiInfo _makeTextEditCompletionItemRtti()
{
    TextEditCompletionItem obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::TextEditCompletionItem", nullptr);
    builder.addField("label", &obj.label, StructRttiInfo::Flag::Optional);
    builder.addField("detail", &obj.detail, StructRttiInfo::Flag::Optional);
    builder.addField("kind", &obj.kind, StructRttiInfo::Flag::Optional);
    builder.addField("documentation", &obj.documentation, StructRttiInfo::Flag::Optional);
    builder.addField("data", &obj.data, StructRttiInfo::Flag::Optional);
    builder.addField("textEdit", &obj.textEdit, StructRttiInfo::Flag::Optional);
    builder.addField("commitCharacters", &obj.commitCharacters, StructRttiInfo::Flag::Optional);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo TextEditCompletionItem::g_rttiInfo = _makeTextEditCompletionItemRtti();

static const StructRttiInfo _makeSemanticTokensParamsRtti()
{
    SemanticTokensParams obj;
    StructRttiBuilder builder(
        &obj,
        "LanguageServerProtocol::SemanticTokensParams",
        &WorkDoneProgressParams::g_rttiInfo);
    builder.addField("textDocument", &obj.textDocument);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo SemanticTokensParams::g_rttiInfo = _makeSemanticTokensParamsRtti();
const UnownedStringSlice SemanticTokensParams::methodName =
    UnownedStringSlice::fromLiteral("textDocument/semanticTokens/full");

static const StructRttiInfo _makeSemanticTokensRtti()
{
    SemanticTokens obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::SemanticTokens", nullptr);
    builder.addField("resultId", &obj.resultId);
    builder.addField("data", &obj.data);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo SemanticTokens::g_rttiInfo = _makeSemanticTokensRtti();

static const StructRttiInfo _makeSignatureHelpParamsRtti()
{
    SignatureHelpParams obj;
    StructRttiBuilder builder(
        &obj,
        "LanguageServerProtocol::SignatureHelpParams",
        &WorkDoneProgressParams::g_rttiInfo);
    builder.addField("textDocument", &obj.textDocument);
    builder.addField("position", &obj.position);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo SignatureHelpParams::g_rttiInfo = _makeSignatureHelpParamsRtti();
const UnownedStringSlice SignatureHelpParams::methodName =
    UnownedStringSlice::fromLiteral("textDocument/signatureHelp");

static const StructRttiInfo _makeParameterInformationRtti()
{
    ParameterInformation obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::ParameterInformation", nullptr);
    builder.addField("label", &obj.label);
    builder.addField("documentation", &obj.documentation, StructRttiInfo::Flag::Optional);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo ParameterInformation::g_rttiInfo = _makeParameterInformationRtti();

static const StructRttiInfo _makeSignatureInformationRtti()
{
    SignatureInformation obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::SignatureInformation", nullptr);
    builder.addField("label", &obj.label);
    builder.addField("parameters", &obj.parameters);
    builder.addField("documentation", &obj.documentation, StructRttiInfo::Flag::Optional);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo SignatureInformation::g_rttiInfo = _makeSignatureInformationRtti();

static const StructRttiInfo _makeSignatureHelpRtti()
{
    SignatureHelp obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::SignatureHelp", nullptr);
    builder.addField("signatures", &obj.signatures);
    builder.addField("activeParameter", &obj.activeParameter);
    builder.addField("activeSignature", &obj.activeSignature);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo SignatureHelp::g_rttiInfo = _makeSignatureHelpRtti();

static const StructRttiInfo _makeDidChangeConfigurationParamsRtti()
{
    DidChangeConfigurationParams obj;
    StructRttiBuilder builder(
        &obj,
        "LanguageServerProtocol::DidChangeConfigurationParams",
        nullptr);
    builder.addField("settings", &obj.settings, StructRttiInfo::Flag::Optional);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo DidChangeConfigurationParams::g_rttiInfo =
    _makeDidChangeConfigurationParamsRtti();
const UnownedStringSlice DidChangeConfigurationParams::methodName =
    UnownedStringSlice::fromLiteral("workspace/didChangeConfiguration");

static const StructRttiInfo _makeConfigurationItemRtti()
{
    ConfigurationItem obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::ConfigurationItem", nullptr);
    builder.addField("section", &obj.section, StructRttiInfo::Flag::Optional);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo ConfigurationItem::g_rttiInfo = _makeConfigurationItemRtti();

static const StructRttiInfo _makeConfigurationParamsRtti()
{
    ConfigurationParams obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::ConfigurationParams", nullptr);
    builder.addField("items", &obj.items, StructRttiInfo::Flag::Optional);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo ConfigurationParams::g_rttiInfo = _makeConfigurationParamsRtti();
const UnownedStringSlice ConfigurationParams::methodName =
    UnownedStringSlice::fromLiteral("workspace/configuration");

static const StructRttiInfo _makeRegistrationRtti()
{
    Registration obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::Registration", nullptr);
    builder.addField("id", &obj.id, StructRttiInfo::Flag::Optional);
    builder.addField("method", &obj.method, StructRttiInfo::Flag::Optional);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo Registration::g_rttiInfo = _makeRegistrationRtti();

static const StructRttiInfo _makeRegistrationParamsRtti()
{
    RegistrationParams obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::RegistrationParams", nullptr);
    builder.addField("registrations", &obj.registrations, StructRttiInfo::Flag::Optional);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo RegistrationParams::g_rttiInfo = _makeRegistrationParamsRtti();

static const StructRttiInfo _makeCancelParamsRtti()
{
    CancelParams obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::CancelParams", nullptr);
    builder.addField("id", &obj.id, StructRttiInfo::Flag::Optional);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo CancelParams::g_rttiInfo = _makeCancelParamsRtti();

static const StructRttiInfo _makeLogMessageParamsRtti()
{
    LogMessageParams obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::LogMessageParams", nullptr);
    builder.addField("type", &obj.type, StructRttiInfo::Flag::Optional);
    builder.addField("message", &obj.message, StructRttiInfo::Flag::Optional);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo LogMessageParams::g_rttiInfo = _makeLogMessageParamsRtti();
const UnownedStringSlice LogMessageParams::methodName =
    UnownedStringSlice::fromLiteral("window/logMessage");

static const StructRttiInfo _makeDocumentSymbolParamsRtti()
{
    DocumentSymbolParams obj;
    StructRttiBuilder builder(
        &obj,
        "LanguageServerProtocol::DocumentSymbolParams",
        &WorkDoneProgressParams::g_rttiInfo);
    builder.addField("textDocument", &obj.textDocument);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo DocumentSymbolParams::g_rttiInfo = _makeDocumentSymbolParamsRtti();
const UnownedStringSlice DocumentSymbolParams::methodName =
    UnownedStringSlice::fromLiteral("textDocument/documentSymbol");

static const StructRttiInfo _makeDocumentSymbolRtti()
{
    DocumentSymbol obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::DocumentSymbol", nullptr);
    builder.addField("name", &obj.name);
    builder.addField("detail", &obj.detail);
    builder.addField("kind", &obj.kind);
    builder.addField("range", &obj.range);
    builder.addField("selectionRange", &obj.selectionRange);
    builder.addField("children", &obj.children);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo DocumentSymbol::g_rttiInfo = _makeDocumentSymbolRtti();

static const StructRttiInfo _makeInlayHintParamsRtti()
{
    InlayHintParams obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::InlayHintParams", nullptr);
    builder.addField("range", &obj.range);
    builder.addField("textDocument", &obj.textDocument);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo InlayHintParams::g_rttiInfo = _makeInlayHintParamsRtti();
const UnownedStringSlice InlayHintParams::methodName =
    UnownedStringSlice::fromLiteral("textDocument/inlayHint");

static const StructRttiInfo _makeInlayHintRtti()
{
    InlayHint obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::InlayHint", nullptr);
    builder.addField("position", &obj.position);
    builder.addField("label", &obj.label);
    builder.addField("kind", &obj.kind);
    builder.addField("paddingLeft", &obj.paddingLeft);
    builder.addField("paddingRight", &obj.paddingRight);
    builder.addField("textEdits", &obj.textEdits);

    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo InlayHint::g_rttiInfo = _makeInlayHintRtti();

static const StructRttiInfo _makeDocumentFormattingParamsRtti()
{
    DocumentFormattingParams obj;
    StructRttiBuilder builder(&obj, "LanguageServerProtocol::DocumentFormattingParams", nullptr);
    builder.addField("textDocument", &obj.textDocument);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo DocumentFormattingParams::g_rttiInfo = _makeDocumentFormattingParamsRtti();
const UnownedStringSlice DocumentFormattingParams::methodName =
    UnownedStringSlice::fromLiteral("textDocument/formatting");

static const StructRttiInfo _makeDocumentRangeFormattingParamsRtti()
{
    DocumentRangeFormattingParams obj;
    StructRttiBuilder builder(
        &obj,
        "LanguageServerProtocol::DocumentRangeFormattingParams",
        nullptr);
    builder.addField("textDocument", &obj.textDocument);
    builder.addField("range", &obj.range);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo DocumentRangeFormattingParams::g_rttiInfo =
    _makeDocumentRangeFormattingParamsRtti();
const UnownedStringSlice DocumentRangeFormattingParams::methodName =
    UnownedStringSlice::fromLiteral("textDocument/rangeFormatting");

static const StructRttiInfo _makeDocumentOnTypeFormattingParamsRtti()
{
    DocumentOnTypeFormattingParams obj;
    StructRttiBuilder builder(
        &obj,
        "LanguageServerProtocol::DocumentOnTypeFormattingParams",
        nullptr);
    builder.addField("textDocument", &obj.textDocument);
    builder.addField("position", &obj.position);
    builder.addField("ch", &obj.ch);
    builder.ignoreUnknownFields();
    return builder.make();
}
const StructRttiInfo DocumentOnTypeFormattingParams::g_rttiInfo =
    _makeDocumentOnTypeFormattingParamsRtti();
const UnownedStringSlice DocumentOnTypeFormattingParams::methodName =
    UnownedStringSlice::fromLiteral("textDocument/onTypeFormatting");

} // namespace LanguageServerProtocol

} // namespace Slang
