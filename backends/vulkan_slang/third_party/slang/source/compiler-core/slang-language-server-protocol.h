#pragma once

#include "../../source/compiler-core/slang-json-value.h"
#include "../../source/core/slang-rtti-info.h"
#include "slang-com-helper.h"
#include "slang-com-ptr.h"
#include "slang.h"

#include <optional>

namespace Slang
{
namespace LanguageServerProtocol
{
struct ServerInfo
{
    String name;
    String version;

    static const StructRttiInfo g_rttiInfo;
};

enum class TextDocumentSyncKind
{
    None = 0,
    Full = 1,
    Incremental = 2
};

struct TextDocumentSyncOptions
{
    bool openClose = false;
    int32_t change = int32_t(TextDocumentSyncKind::None); // TextDocumentSyncKind
    static const StructRttiInfo g_rttiInfo;
};

struct WorkDoneProgressParams
{
    /**
     * An optional token that a server can use to report work done progress.
     */
    String workDoneToken; // optional

    static const StructRttiInfo g_rttiInfo;
};

struct CompletionOptions : public WorkDoneProgressParams
{
    /**
     * Most tools trigger completion request automatically without explicitly
     * requesting it using a keyboard shortcut (e.g. Ctrl+Space). Typically they
     * do so when the user starts to type an identifier. For example if the user
     * types `c` in a JavaScript file code complete will automatically pop up
     * present `console` besides others as a completion item. Characters that
     * make up identifiers don't need to be listed here.
     *
     * If code complete should automatically be trigger on characters not being
     * valid inside an identifier (for example `.` in JavaScript) list them in
     * `triggerCharacters`.
     */
    List<String> triggerCharacters;

    /**
     * The list of all possible characters that commit a completion. This field
     * can be used if clients don't support individual commit characters per
     * completion item. See client capability
     * `completion.completionItem.commitCharactersSupport`.
     *
     * If a server provides both `allCommitCharacters` and commit characters on
     * an individual completion item the ones on the completion item win.
     *
     * @since 3.2.0
     */
    List<String> allCommitCharacters;

    /**
     * The server provides support to resolve additional
     * information for a completion item.
     */
    bool resolveProvider = false;

    static const StructRttiInfo g_rttiInfo;
};

struct SemanticTokensLegend
{
    /**
     * The token types a server uses.
     */
    List<String> tokenTypes;

    /**
     * The token modifiers a server uses.
     */
    List<String> tokenModifiers;

    static const StructRttiInfo g_rttiInfo;
};


struct SemanticTokensOptions
{
    /**
     * The legend used by the server
     */
    SemanticTokensLegend legend;

    /**
     * Server supports providing semantic tokens for a specific range
     * of a document.
     */
    bool range = false;

    /**
     * Server supports providing semantic tokens for a full document.
     */
    bool full = false;

    static const StructRttiInfo g_rttiInfo;
};

struct SignatureHelpOptions
{
    /**
     * The characters that trigger signature help
     * automatically.
     */
    List<String> triggerCharacters;

    /**
     * List of characters that re-trigger signature help.
     *
     * These trigger characters are only active when signature help is already
     * showing. All trigger characters are also counted as re-trigger
     * characters.
     *
     * @since 3.15.0
     */
    List<String> retriggerCharacters;

    static const StructRttiInfo g_rttiInfo;
};

struct TextDocumentItem
{
    String uri;
    String languageId;
    int version = 0;
    String text;
    static const StructRttiInfo g_rttiInfo;
};

struct TextDocumentIdentifier
{
    String uri;
    static const StructRttiInfo g_rttiInfo;
};

struct VersionedTextDocumentIdentifier
{
    String uri;
    int version = 0;
    static const StructRttiInfo g_rttiInfo;
};

struct Position
{
    int line = -1;
    int character = -1;
    static const StructRttiInfo g_rttiInfo;
};

struct Range
{
    Position start;
    Position end;
    static const StructRttiInfo g_rttiInfo;
};

struct TextEdit
{
    /**
     * The range of the text document to be manipulated. To insert
     * text into a document create a range where start === end.
     */
    Range range;

    /**
     * The string to be inserted. For delete operations use an
     * empty string.
     */
    String newText;

    static const StructRttiInfo g_rttiInfo;
};

struct DidOpenTextDocumentParams
{
    TextDocumentItem textDocument;
    static const StructRttiInfo g_rttiInfo;
    static const UnownedStringSlice methodName;
};

struct TextDocumentContentChangeEvent
{
    Range range; // optional
    String text;
    static const StructRttiInfo g_rttiInfo;
};

struct DidChangeTextDocumentParams
{
    VersionedTextDocumentIdentifier textDocument;
    List<TextDocumentContentChangeEvent> contentChanges;
    static const StructRttiInfo g_rttiInfo;
    static const UnownedStringSlice methodName;
};

struct DidCloseTextDocumentParams
{
    TextDocumentIdentifier textDocument;
    static const StructRttiInfo g_rttiInfo;
    static const UnownedStringSlice methodName;
};

struct WorkspaceFoldersServerCapabilities
{
    /**
     * The server has support for workspace folders
     */
    bool supported = false;

    /**
     * Whether the server wants to receive workspace folder
     * change notifications.
     *
     * If a string is provided, the string is treated as an ID
     * under which the notification is registered on the client
     * side. The ID can be used to unregister for these events
     * using the `client/unregisterCapability` request.
     */
    bool changeNotifications = false;

    static const StructRttiInfo g_rttiInfo;
};

struct WorkspaceCapabilities
{
    WorkspaceFoldersServerCapabilities workspaceFolders;
    static const StructRttiInfo g_rttiInfo;
};

/**
 * Inlay hint options used during static registration.
 *
 * @since 3.17.0
 */
struct InlayHintOptions
{
    /**
     * The server provides support to resolve additional
     * information for an inlay hint item.
     */
    bool resolveProvider = false;
    static const StructRttiInfo g_rttiInfo;
};

struct DocumentOnTypeFormattingOptions
{
    /**
     * A character on which formatting should be triggered, like `{`.
     */
    String firstTriggerCharacter;

    /**
     * More trigger characters.
     */
    List<String> moreTriggerCharacter;

    static const StructRttiInfo g_rttiInfo;
};

struct ServerCapabilities
{
    String positionEncoding;
    TextDocumentSyncOptions textDocumentSync;
    bool hoverProvider = false;
    bool definitionProvider = false;
    bool documentSymbolProvider = false;
    bool documentFormattingProvider = false;
    bool documentRangeFormattingProvider = false;
    DocumentOnTypeFormattingOptions documentOnTypeFormattingProvider;
    InlayHintOptions inlayHintProvider;
    CompletionOptions completionProvider;
    SemanticTokensOptions semanticTokensProvider;
    SignatureHelpOptions signatureHelpProvider;
    WorkspaceCapabilities workspace;
    static const StructRttiInfo g_rttiInfo;
};

struct VSServerCapabilities : ServerCapabilities
{
    bool _vs_projectContextProvider = false;
    static const StructRttiInfo g_rttiInfo;
};

struct WorkspaceFolder
{
    String uri;
    String name;
    static const StructRttiInfo g_rttiInfo;
};

struct InitializeParams
{
    List<WorkspaceFolder> workspaceFolders;
    static const UnownedStringSlice methodName;
    static const StructRttiInfo g_rttiInfo;
};

struct NullResponse
{
    static const StructRttiInfo g_rttiInfo;
    static NullResponse* get();
};

struct InitializeResult
{
    ServerCapabilities capabilities;
    ServerInfo serverInfo;

    static const StructRttiInfo g_rttiInfo;
};

struct VSInitializeResult
{
    VSServerCapabilities capabilities;
    ServerInfo serverInfo;

    static const StructRttiInfo g_rttiInfo;
};

struct ShutdownParams
{
    static const UnownedStringSlice methodName;
};

struct ExitParams
{
    static const UnownedStringSlice methodName;
};

typedef uint32_t DiagnosticSeverity;
/**
 * Reports an error.
 */
const DiagnosticSeverity kDiagnosticsSeverityError = 1;
/**
 * Reports a warning.
 */
const DiagnosticSeverity kDiagnosticsSeverityWarning = 2;
/**
 * Reports an information.
 */
const DiagnosticSeverity kDiagnosticsSeverityInformation = 3;
/**
 * Reports a hint.
 */
const DiagnosticSeverity kDiagnosticsSeverityHint = 4;


struct Location
{
    String uri;
    Range range;
    static const StructRttiInfo g_rttiInfo;
};

struct DiagnosticRelatedInformation
{
    /**
     * The location of this related diagnostic information.
     */
    Location location;

    /**
     * The message of this related diagnostic information.
     */
    String message;

    static const StructRttiInfo g_rttiInfo;
};

struct Diagnostic
{
    /**
     * The range at which the message applies.
     */
    Range range;

    /**
     * The diagnostic's severity. Can be omitted. If omitted it is up to the
     * client to interpret diagnostics as error, warning, info or hint.
     */
    DiagnosticSeverity severity = 1;

    /**
     * The diagnostic's code, which might appear in the user interface.
     */
    int32_t code = 0;

    /**
     * A human-readable string describing the source of this
     * diagnostic, e.g. 'typescript' or 'super lint'.
     */
    String source;

    /**
     * The diagnostic's message.
     */
    String message;

    /**
     * An array of related diagnostic information, e.g. when symbol-names within
     * a scope collide all definitions can be marked via this property.
     */
    List<DiagnosticRelatedInformation> relatedInformation;

    bool operator==(const Diagnostic& other) const
    {
        return code == other.code && range.start.line == other.range.start.line &&
               message == other.message;
    }

    HashCode getHashCode() const
    {
        return combineHash(code, combineHash(range.start.line, message.getHashCode()));
    }

    static const StructRttiInfo g_rttiInfo;
};

struct PublishDiagnosticsParams
{
    /**
     * The URI for which diagnostic information is reported.
     */
    String uri;

    /**
     * An array of diagnostic information items.
     */
    List<Diagnostic> diagnostics;

    static const StructRttiInfo g_rttiInfo;
};

struct TextDocumentPositionParams
{
    /**
     * The text document.
     */
    TextDocumentIdentifier textDocument;

    /**
     * The position inside the text document.
     */
    Position position;

    static const StructRttiInfo g_rttiInfo;
};

struct HoverParams : WorkDoneProgressParams, TextDocumentPositionParams
{
    static const StructRttiInfo g_rttiInfo;
    static const UnownedStringSlice methodName;
};

struct DefinitionParams : WorkDoneProgressParams, TextDocumentPositionParams
{
    static const StructRttiInfo g_rttiInfo;
    static const UnownedStringSlice methodName;
};

struct MarkupContent
{
    /**
     * The type of the Markup
     */
    String kind;

    /**
     * The content itself
     */
    String value;

    static const StructRttiInfo g_rttiInfo;
};

struct Hover
{
    /**
     * The hover's content
     */
    MarkupContent contents;

    /**
     * An optional range is a range inside a text document
     * that is used to visualize a hover, e.g. by changing the background color.
     */
    Range range;

    static const StructRttiInfo g_rttiInfo;
};

typedef int CompletionTriggerKind;
const CompletionTriggerKind kCompletionTriggerKindInvoked = 1;

/**
 * Completion was triggered by a trigger character specified by
 * the `triggerCharacters` properties of the
 * `CompletionRegistrationOptions`.
 */
const CompletionTriggerKind kCompletionTriggerKindTriggerCharacter = 2;

/**
 * Completion was re-triggered as the current completion list is incomplete.
 */
const CompletionTriggerKind kCompletionTriggerKindTriggerForIncompleteCompletions = 3;

/**
 * Contains additional information about the context in which a completion
 * request is triggered.
 */
struct CompletionContext
{
    /**
     * How the completion was triggered.
     */
    CompletionTriggerKind triggerKind = 1;

    /**
     * The trigger character (a single character) that has trigger code
     * complete. Is undefined if
     * `triggerKind !== CompletionTriggerKind.TriggerCharacter`
     */
    String triggerCharacter;

    static const StructRttiInfo g_rttiInfo;
};

struct CompletionParams : WorkDoneProgressParams, TextDocumentPositionParams
{
    CompletionContext context;

    static const StructRttiInfo g_rttiInfo;
    static const UnownedStringSlice methodName;
};

typedef int32_t CompletionItemKind;
const CompletionItemKind kCompletionItemKindText = 1;
const CompletionItemKind kCompletionItemKindMethod = 2;
const CompletionItemKind kCompletionItemKindFunction = 3;
const CompletionItemKind kCompletionItemKindConstructor = 4;
const CompletionItemKind kCompletionItemKindField = 5;
const CompletionItemKind kCompletionItemKindVariable = 6;
const CompletionItemKind kCompletionItemKindClass = 7;
const CompletionItemKind kCompletionItemKindInterface = 8;
const CompletionItemKind kCompletionItemKindModule = 9;
const CompletionItemKind kCompletionItemKindProperty = 10;
const CompletionItemKind kCompletionItemKindUnit = 11;
const CompletionItemKind kCompletionItemKindValue = 12;
const CompletionItemKind kCompletionItemKindEnum = 13;
const CompletionItemKind kCompletionItemKindKeyword = 14;
const CompletionItemKind kCompletionItemKindSnippet = 15;
const CompletionItemKind kCompletionItemKindColor = 16;
const CompletionItemKind kCompletionItemKindFile = 17;
const CompletionItemKind kCompletionItemKindReference = 18;
const CompletionItemKind kCompletionItemKindFolder = 19;
const CompletionItemKind kCompletionItemKindEnumMember = 20;
const CompletionItemKind kCompletionItemKindConstant = 21;
const CompletionItemKind kCompletionItemKindStruct = 22;
const CompletionItemKind kCompletionItemKindEvent = 23;
const CompletionItemKind kCompletionItemKindOperator = 24;
const CompletionItemKind kCompletionItemKindTypeParameter = 25;

struct CompletionItem
{
    /**
     * The label of this completion item.
     *
     * The label property is also by default the text that
     * is inserted when selecting this completion.
     *
     * If label details are provided the label itself should
     * be an unqualified name of the completion item.
     */
    String label;

    /**
     * The kind of this completion item. Based of the kind
     * an icon is chosen by the editor. The standardized set
     * of available values is defined in `CompletionItemKind`.
     */
    CompletionItemKind kind = CompletionItemKind(0);

    /**
     * A human-readable string with additional information
     * about this item, like type or symbol information.
     */
    String detail;

    /**
     * A human-readable string that represents a doc-comment.
     */
    MarkupContent documentation;

    /**
     * An optional set of characters that when pressed while this completion is
     * active will accept it first and then type that character. *Note* that all
     * commit characters should have `length=1` and that superfluous characters
     * will be ignored.
     */
    List<String> commitCharacters;

    // Additional data.
    String data;

    static const StructRttiInfo g_rttiInfo;
};

struct TextEditCompletionItem
{
    /**
     * The label of this completion item.
     *
     * The label property is also by default the text that
     * is inserted when selecting this completion.
     *
     * If label details are provided the label itself should
     * be an unqualified name of the completion item.
     */
    String label;

    /**
     * The kind of this completion item. Based of the kind
     * an icon is chosen by the editor. The standardized set
     * of available values is defined in `CompletionItemKind`.
     */
    CompletionItemKind kind = CompletionItemKind(0);

    /**
     * A human-readable string with additional information
     * about this item, like type or symbol information.
     */
    String detail;

    /**
     * A human-readable string that represents a doc-comment.
     */
    MarkupContent documentation;

    TextEdit textEdit;

    /**
     * An optional set of characters that when pressed while this completion is
     * active will accept it first and then type that character. *Note* that all
     * commit characters should have `length=1` and that superfluous characters
     * will be ignored.
     */
    List<String> commitCharacters;

    // Additional data.
    String data;

    static const StructRttiInfo g_rttiInfo;
};

struct SemanticTokensParams : WorkDoneProgressParams
{
    TextDocumentIdentifier textDocument;

    static const UnownedStringSlice methodName;

    static const StructRttiInfo g_rttiInfo;
};


struct SemanticTokens
{
    /**
     * An optional result id. If provided and clients support delta updating
     * the client will include the result id in the next semantic token request.
     * A server can then instead of computing all semantic tokens again simply
     * send a delta.
     */
    String resultId;

    /**
     * The actual tokens.
     */
    List<uint32_t> data;

    static const StructRttiInfo g_rttiInfo;
};

struct SignatureHelpParams : WorkDoneProgressParams, TextDocumentPositionParams
{
    static const UnownedStringSlice methodName;

    static const StructRttiInfo g_rttiInfo;
};

/**
 * Represents a parameter of a callable-signature. A parameter can
 * have a label and a doc-comment.
 */
struct ParameterInformation
{
    /**
     * The label of this parameter information.
     *
     * Either a string or an inclusive start and exclusive end offsets within
     * its containing signature label. (see SignatureInformation.label). The
     * offsets are based on a UTF-16 string representation as `Position` and
     * `Range` does.
     *
     * *Note*: a label of type string should be a substring of its containing
     * signature label. Its intended use case is to highlight the parameter
     * label part in the `SignatureInformation.label`.
     */
    uint32_t label[2] = {0, 0};

    /**
     * The human-readable doc-comment of this parameter. Will be shown
     * in the UI but can be omitted.
     */
    MarkupContent documentation;

    static const StructRttiInfo g_rttiInfo;
};

/**
 * Represents the signature of something callable. A signature
 * can have a label, like a function-name, a doc-comment, and
 * a set of parameters.
 */
struct SignatureInformation
{
    /**
     * The label of this signature. Will be shown in
     * the UI.
     */
    String label;

    /**
     * The human-readable doc-comment of this signature. Will be shown
     * in the UI but can be omitted.
     */
    MarkupContent documentation;

    /**
     * The parameters of this signature.
     */
    List<ParameterInformation> parameters;

    static const StructRttiInfo g_rttiInfo;
};

struct SignatureHelp
{
    /**
     * One or more signatures. If no signatures are available the signature help
     * request should return `null`.
     */
    List<SignatureInformation> signatures;

    /**
     * The active signature. If omitted or the value lies outside the
     * range of `signatures` the value defaults to zero or is ignore if
     * the `SignatureHelp` as no signatures.
     *
     * Whenever possible implementors should make an active decision about
     * the active signature and shouldn't rely on a default value.
     *
     * In future version of the protocol this property might become
     * mandatory to better express this.
     */
    uint32_t activeSignature = 0;

    /**
     * The active parameter of the active signature. If omitted or the value
     * lies outside the range of `signatures[activeSignature].parameters`
     * defaults to 0 if the active signature has parameters. If
     * the active signature has no parameters it is ignored.
     * In future version of the protocol this property might become
     * mandatory to better express the active parameter if the
     * active signature does have any.
     */
    uint32_t activeParameter = 0;

    static const StructRttiInfo g_rttiInfo;
};


struct DidChangeConfigurationParams
{
    /**
     * The actual changed settings
     */
    JSONValue settings = JSONValue::makeInvalid();

    static const StructRttiInfo g_rttiInfo;

    static const UnownedStringSlice methodName;
};

struct ConfigurationItem
{
    /**
     * The configuration section asked for.
     */
    String section;

    static const StructRttiInfo g_rttiInfo;
};

struct ConfigurationParams
{
    List<ConfigurationItem> items;

    static const StructRttiInfo g_rttiInfo;

    static const UnownedStringSlice methodName;
};

struct Registration
{
    /**
     * The id used to register the request. The id can be used to deregister
     * the request again.
     */
    String id;

    /**
     * The method / capability to register for.
     */
    String method;

    static const StructRttiInfo g_rttiInfo;
};

struct RegistrationParams
{
    List<Registration> registrations;

    static const StructRttiInfo g_rttiInfo;
};

struct CancelParams
{
    /**
     * The request id to cancel.
     */
    int64_t id = 0;

    static const StructRttiInfo g_rttiInfo;
};

struct LogMessageParams
{
    /**
     * The message type. See {@link MessageType}
     */
    int type = 0;

    /**
     * The actual message
     */
    String message;

    static const StructRttiInfo g_rttiInfo;
    static const UnownedStringSlice methodName;
};

struct DocumentSymbolParams : WorkDoneProgressParams
{
    /**
     * The text document.
     */
    TextDocumentIdentifier textDocument;

    static const StructRttiInfo g_rttiInfo;
    static const UnownedStringSlice methodName;
};

typedef int SymbolKind;
const int kSymbolKindFile = 1;
const int kSymbolKindModule = 2;
const int kSymbolKindNamespace = 3;
const int kSymbolKindPackage = 4;
const int kSymbolKindClass = 5;
const int kSymbolKindMethod = 6;
const int kSymbolKindProperty = 7;
const int kSymbolKindField = 8;
const int kSymbolKindConstructor = 9;
const int kSymbolKindEnum = 10;
const int kSymbolKindInterface = 11;
const int kSymbolKindFunction = 12;
const int kSymbolKindVariable = 13;
const int kSymbolKindConstant = 14;
const int kSymbolKindString = 15;
const int kSymbolKindNumber = 16;
const int kSymbolKindBoolean = 17;
const int kSymbolKindArray = 18;
const int kSymbolKindObject = 19;
const int kSymbolKindKey = 20;
const int kSymbolKindNull = 21;
const int kSymbolKindEnumMember = 22;
const int kSymbolKindStruct = 23;
const int kSymbolKindEvent = 24;
const int kSymbolKindOperator = 25;
const int kSymbolKindTypeParameter = 26;

/**
 * Represents programming constructs like variables, classes, interfaces etc.
 * that appear in a document. Document symbols can be hierarchical and they
 * have two ranges: one that encloses its definition and one that points to its
 * most interesting range, e.g. the range of an identifier.
 */
struct DocumentSymbol
{
    /**
     * The name of this symbol. Will be displayed in the user interface and
     * therefore must not be an empty string or a string only consisting of
     * white spaces.
     */
    String name;

    /**
     * More detail for this symbol, e.g the signature of a function.
     */
    String detail;

    /**
     * The kind of this symbol.
     */
    SymbolKind kind = 0;

    /**
     * The range enclosing this symbol not including leading/trailing whitespace
     * but everything else like comments. This information is typically used to
     * determine if the clients cursor is inside the symbol to reveal in the
     * symbol in the UI.
     */
    Range range;

    /**
     * The range that should be selected and revealed when this symbol is being
     * picked, e.g. the name of a function. Must be contained by the `range`.
     */
    Range selectionRange;

    /**
     * Children of this symbol, e.g. properties of a class.
     */
    List<DocumentSymbol> children;

    static const StructRttiInfo g_rttiInfo;
};

/**
 * A parameter literal used in inlay hint requests.
 *
 * @since 3.17.0
 */
struct InlayHintParams
{
    /**
     * The text document.
     */
    TextDocumentIdentifier textDocument;

    /**
     * The visible document range for which inlay hints should be computed.
     */
    Range range;

    static const StructRttiInfo g_rttiInfo;
    static const UnownedStringSlice methodName;
};

typedef int InlayHintKind;
const int kInlayHintKindType = 1;
const int kInlayHintKindParameter = 2;

/**
 * Inlay hint information.
 *
 * @since 3.17.0
 */
struct InlayHint
{
    /**
     * The position of this hint.
     */
    Position position;

    /**
     * The label of this hint. A human readable string or an array of
     * InlayHintLabelPart label parts.
     *
     * *Note* that neither the string nor the label part can be empty.
     */
    String label;

    /**
     * The kind of this hint. Can be omitted in which case the client
     * should fall back to a reasonable default.
     */
    InlayHintKind kind = 1;

    List<TextEdit> textEdits;

    /**
     * Render padding before the hint.
     *
     * Note: Padding should use the editor's background color, not the
     * background color of the hint itself. That means padding can be used
     * to visually align/separate an inlay hint.
     */
    bool paddingLeft = false;

    /**
     * Render padding after the hint.
     *
     * Note: Padding should use the editor's background color, not the
     * background color of the hint itself. That means padding can be used
     * to visually align/separate an inlay hint.
     */
    bool paddingRight = false;

    static const StructRttiInfo g_rttiInfo;
};

struct DocumentOnTypeFormattingParams
{
    /**
     * The document to format.
     */
    TextDocumentIdentifier textDocument;

    /**
     * The position around which the on type formatting should happen.
     * This is not necessarily the exact position where the character denoted
     * by the property `ch` got typed.
     */
    Position position;

    /**
     * The character that has been typed that triggered the formatting
     * on type request. That is not necessarily the last character that
     * got inserted into the document since the client could auto insert
     * characters as well (e.g. like automatic brace completion).
     */
    String ch;

    /**
     * The formatting options.
     */
    // FormattingOptions options;

    static const StructRttiInfo g_rttiInfo;
    static const UnownedStringSlice methodName;
};

struct DocumentRangeFormattingParams
{
    /**
     * The document to format.
     */
    TextDocumentIdentifier textDocument;

    /**
     * The range to format
     */
    Range range;

    /**
     * The format options
     */
    // FormattingOptions options;

    static const StructRttiInfo g_rttiInfo;
    static const UnownedStringSlice methodName;
};

struct DocumentFormattingParams
{
    /**
     * The document to format.
     */
    TextDocumentIdentifier textDocument;

    /**
     * The format options
     */
    // FormattingOptions options;

    static const StructRttiInfo g_rttiInfo;
    static const UnownedStringSlice methodName;
};

} // namespace LanguageServerProtocol
} // namespace Slang

namespace Slang
{
template<typename T>
struct LanguageServerResult
{
    SlangResult returnCode;
    bool isNull = true;
    T result;
    LanguageServerResult() { returnCode = SLANG_OK; }
    LanguageServerResult(std::nullopt_t) { returnCode = SLANG_OK; }
    LanguageServerResult(const T& value)
    {
        result = value;
        isNull = false;
        returnCode = SLANG_OK;
    }
    LanguageServerResult(SlangResult code) { returnCode = code; }
};
} // namespace Slang
