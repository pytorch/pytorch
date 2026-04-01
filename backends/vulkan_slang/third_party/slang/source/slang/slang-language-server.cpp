// language-server.cpp

// This file implements the language server for Slang, conforming to the Language Server Protocol.
// https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/


#include "slang-language-server.h"

#include "../../tools/platform/performance-counter.h"
#include "../compiler-core/slang-json-native.h"
#include "../compiler-core/slang-json-rpc-connection.h"
#include "../compiler-core/slang-language-server-protocol.h"
#include "../core/slang-char-util.h"
#include "../core/slang-secure-crt.h"
#include "../core/slang-string-util.h"
#include "slang-ast-print.h"
#include "slang-check-impl.h"
#include "slang-com-helper.h"
#include "slang-doc-markdown-writer.h"
#include "slang-language-server-ast-lookup.h"
#include "slang-language-server-completion.h"
#include "slang-language-server-document-symbols.h"
#include "slang-language-server-inlay-hints.h"
#include "slang-language-server-semantic-tokens.h"
#include "slang-mangle.h"
#include "slang-workspace-version.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SLANG_LS_RETURN_ON_SUCCESS(x)    \
    {                                    \
        auto _res = (x);                 \
        if (_res.returnCode == SLANG_OK) \
            return _res;                 \
    }
#define SLANG_LS_RETURN_ON_FAIL(x)         \
    {                                      \
        auto _res = (x);                   \
        if (SLANG_FAILED(_res.returnCode)) \
            return _res;                   \
    }

namespace Slang
{
using namespace LanguageServerProtocol;

ArrayView<const char*> getCommitChars()
{
    static const char* _commitCharsArray[] = {",", ".", ";", ":", "(", ")", "[", "]",
                                              "<", ">", "{", "}", "*", "&", "^", "%",
                                              "!", "-", "=", "+", "|", "/", "?", " "};
    return makeArrayView(_commitCharsArray, SLANG_COUNT_OF(_commitCharsArray));
}

SlangResult LanguageServerCore::init(const InitializeParams& args)
{
    m_workspaceFolders = args.workspaceFolders;
    m_workspace = new Workspace();
    List<URI> rootUris;
    for (auto& wd : m_workspaceFolders)
    {
        rootUris.add(URI::fromString(wd.uri.getUnownedSlice()));
    }
    m_workspace->init(rootUris, getOrCreateGlobalSession());
    return SLANG_OK;
}

SlangResult LanguageServer::init(const InitializeParams& args)
{
    SLANG_RETURN_ON_FAIL(m_connection->initWithStdStreams(JSONRPCConnection::CallStyle::Object));

    m_typeMap = JSONNativeUtil::getTypeFuncsMap();

    return m_core.init(args);
}

slang::IGlobalSession* LanguageServerCore::getOrCreateGlobalSession()
{
    if (!m_session)
    {
        // Just create the global session in the regular way if there isn't one set
        SlangGlobalSessionDesc desc = {};
        desc.enableGLSL = true;
        if (SLANG_FAILED(slang_createGlobalSession2(&desc, m_session.writeRef())))
        {
            return nullptr;
        }
    }

    return m_session;
}

void LanguageServer::resetDiagnosticUpdateTime()
{
    m_lastDiagnosticUpdateTime = std::chrono::system_clock::now();
}

String uriToCanonicalPath(const String& uri)
{
    String canonnicalPath;
    Path::getCanonical(URI::fromString(uri.getUnownedSlice()).getPath(), canonnicalPath);
    return canonnicalPath;
}

SlangResult LanguageServer::parseNextMessage()
{
    const JSONRPCMessageType msgType = m_connection->getMessageType();

    switch (msgType)
    {
    case JSONRPCMessageType::Call:
        {
            JSONRPCCall call;
            SLANG_RETURN_ON_FAIL(m_connection->getRPCOrSendError(&call));
            if (call.method == ExitParams::methodName)
            {
                m_quit = true;
                return SLANG_OK;
            }
            else if (call.method == ShutdownParams::methodName)
            {
                m_connection->sendResult(NullResponse::get(), call.id);
                return SLANG_OK;
            }
            else if (call.method == InitializeParams::methodName)
            {
                InitializeParams args;
                m_connection->toNativeArgsOrSendError(call.params, &args, call.id);
                init(args);
                auto fillCapability = [&](ServerCapabilities& caps)
                {
                    caps.positionEncoding = "utf-16";
                    caps.textDocumentSync.openClose = true;
                    caps.textDocumentSync.change = (int)TextDocumentSyncKind::Incremental;
                    caps.workspace.workspaceFolders.supported = true;
                    caps.workspace.workspaceFolders.changeNotifications = false;
                    caps.hoverProvider = true;
                    caps.definitionProvider = true;
                    caps.documentSymbolProvider = true;
                    caps.inlayHintProvider.resolveProvider = false;
                    caps.documentFormattingProvider = true;
                    caps.documentOnTypeFormattingProvider.firstTriggerCharacter = "}";
                    caps.documentOnTypeFormattingProvider.moreTriggerCharacter.add(";");
                    caps.documentOnTypeFormattingProvider.moreTriggerCharacter.add(":");
                    caps.documentOnTypeFormattingProvider.moreTriggerCharacter.add("{");
                    caps.documentRangeFormattingProvider = true;
                    caps.completionProvider.triggerCharacters.add(".");
                    caps.completionProvider.triggerCharacters.add(">");
                    caps.completionProvider.triggerCharacters.add(":");
                    caps.completionProvider.triggerCharacters.add("[");
                    caps.completionProvider.triggerCharacters.add(" ");
                    caps.completionProvider.triggerCharacters.add("(");
                    caps.completionProvider.triggerCharacters.add("\"");
                    caps.completionProvider.triggerCharacters.add("/");
                    caps.completionProvider.resolveProvider = true;
                    caps.completionProvider.workDoneToken = "";
                    caps.semanticTokensProvider.full = true;
                    caps.semanticTokensProvider.range = false;
                    caps.signatureHelpProvider.triggerCharacters.add("(");
                    caps.signatureHelpProvider.triggerCharacters.add(",");
                    caps.signatureHelpProvider.retriggerCharacters.add(",");
                    for (auto tokenType : kSemanticTokenTypes)
                        caps.semanticTokensProvider.legend.tokenTypes.add(tokenType);
                };
                ServerInfo serverInfo;
                serverInfo.name = "SlangLanguageServer";
                serverInfo.version = "1.8";

                if (m_core.m_options.isVisualStudio)
                {
                    VSInitializeResult vsResult;
                    vsResult.serverInfo = serverInfo;
                    fillCapability(vsResult.capabilities);
                    vsResult.capabilities._vs_projectContextProvider = true;
                    m_connection->sendResult(&vsResult, call.id);
                }
                else
                {
                    InitializeResult result;
                    result.serverInfo = serverInfo;
                    fillCapability(result.capabilities);
                    m_connection->sendResult(&result, call.id);
                }
                return SLANG_OK;
            }
            else if (call.method == "initialized")
            {
                registerCapability("workspace/didChangeConfiguration");
                sendConfigRequest();
                m_initialized = true;
                return SLANG_OK;
            }
            else
            {
                queueJSONCall(call);
                return SLANG_OK;
            }
        }
    case JSONRPCMessageType::Result:
        {
            JSONResultResponse response;
            SLANG_RETURN_ON_FAIL(m_connection->getRPCOrSendError(&response));
            auto responseId = (int)m_connection->getContainer()->asInteger(response.id);
            switch (responseId)
            {
            case kConfigResponseId:
                if (response.result.getKind() == JSONValue::Kind::Array)
                {
                    auto arr = m_connection->getContainer()->getArray(response.result);
                    if (arr.getCount() == 12)
                    {
                        updatePredefinedMacros(arr[0]);
                        updateSearchPaths(arr[1]);
                        updateSearchInWorkspace(arr[2]);
                        updateCommitCharacters(arr[3]);
                        updateFormattingOptions(arr[4], arr[5], arr[6], arr[7], arr[8]);
                        updateInlayHintOptions(arr[9], arr[10]);
                        updateTraceOptions(arr[11]);
                    }
                }
                break;
            }
            return SLANG_OK;
        }
    case JSONRPCMessageType::Error:
        {
#if 0 // Enable for debug only
            JSONRPCErrorResponse error;
            SLANG_RETURN_ON_FAIL(m_connection->getRPCOrSendError(&error));
#endif
            return SLANG_OK;
        }
        break;
    default:
        {
            return m_connection->sendError(
                JSONRPC::ErrorCode::InvalidRequest,
                m_connection->getCurrentMessageId());
        }
    }
}

SlangResult LanguageServerCore::didOpenTextDocument(const DidOpenTextDocumentParams& args)
{
    String canonicalPath = uriToCanonicalPath(args.textDocument.uri);
    m_workspace->openDoc(canonicalPath, args.textDocument.text);
    return SLANG_OK;
}

SlangResult LanguageServer::didOpenTextDocument(const DidOpenTextDocumentParams& args)
{
    return m_core.didOpenTextDocument(args);
}

static bool isBoolType(Type* t)
{
    auto basicType = as<BasicExpressionType>(t);
    if (!basicType)
        return false;
    return basicType->getBaseType() == BaseType::Bool;
}

String getDeclKindString(DeclRef<Decl> declRef)
{
    if (declRef.as<ParamDecl>())
    {
        return "(parameter) ";
    }
    else if (declRef.as<GenericTypeParamDecl>())
    {
        return "(generic type parameter) ";
    }
    else if (declRef.as<GenericTypePackParamDecl>())
    {
        return "(generic type pack parameter) ";
    }
    else if (declRef.as<GenericValueParamDecl>())
    {
        return "(generic value parameter) ";
    }
    else if (auto varDecl = declRef.as<VarDeclBase>())
    {
        auto parent = declRef.getDecl()->parentDecl;
        if (as<GenericDecl>(parent))
            parent = parent->parentDecl;
        if (as<InterfaceDecl>(parent))
        {
            return "(associated constant) ";
        }
        else if (as<AggTypeDeclBase>(parent))
        {
            return "(field) ";
        }
        const char* scopeKind = "";
        if (as<NamespaceDeclBase>(parent))
            scopeKind = "global ";
        else if (getParentDecl(declRef.getDecl()))
            scopeKind = "local ";
        StringBuilder sb;
        sb << "(";
        sb << scopeKind;
        if (varDecl.as<LetDecl>())
            sb << "value";
        else
            sb << "variable";
        sb << ") ";
        return sb.produceString();
    }
    return String();
}

String getDeclSignatureString(DeclRef<Decl> declRef, WorkspaceVersion* version)
{
    if (declRef.getDecl())
    {
        auto astBuilder = version->linkage->getASTBuilder();
        ASTPrinter printer(
            astBuilder,
            ASTPrinter::OptionFlag::ParamNames | ASTPrinter::OptionFlag::NoInternalKeywords |
                ASTPrinter::OptionFlag::SimplifiedBuiltinType |
                ASTPrinter::OptionFlag::DefaultParamValues);
        printer.getStringBuilder() << getDeclKindString(declRef);
        printer.addDeclSignature(declRef);
        auto printInitExpr = [&](Module* module, Type* declType, Expr* initExpr)
        {
            auto& sb = printer.getStringBuilder();

            if (auto litExpr = as<LiteralExpr>(initExpr))
            {
                if (litExpr->token.type != TokenType::Unknown)
                    sb << " = " << litExpr->token.getContent();
                else if (auto intLit = as<IntegerLiteralExpr>(litExpr))
                    sb << " = " << intLit->value;
            }
            else if (auto isTypeDecl = as<IsTypeExpr>(initExpr))
            {
                if (isTypeDecl->constantVal)
                {
                    sb << " = " << (isTypeDecl->constantVal->value ? "true" : "false");
                }
            }
            else if (initExpr)
            {
                DiagnosticSink sink;
                SharedSemanticsContext semanticContext(version->linkage, module, &sink);
                SemanticsVisitor semanticsVisitor(&semanticContext);
                if (auto intVal = semanticsVisitor.tryFoldIntegerConstantExpression(
                        declRef.substitute(version->linkage->getASTBuilder(), initExpr),
                        SemanticsVisitor::ConstantFoldingKind::LinkTime,
                        nullptr))
                {
                    if (auto constantInt = as<ConstantIntVal>(intVal))
                    {
                        sb << " = ";
                        if (isBoolType(declType))
                        {
                            sb << (constantInt->getValue() ? "true" : "false");
                        }
                        else
                        {
                            sb << constantInt->getValue();
                        }
                    }
                    else
                    {
                        sb << " = ";
                        intVal->toText(sb);
                    }
                }
            }
        };
        if (auto varDecl = as<VarDeclBase>(declRef.getDecl()))
        {
            if (!varDecl->findModifier<ConstModifier>() && !as<LetDecl>(declRef.getDecl()))
                return printer.getString();
            printInitExpr(getModule(varDecl), varDecl->type, varDecl->initExpr);
        }
        else if (auto enumCase = as<EnumCaseDecl>(declRef.getDecl()))
        {
            printInitExpr(getModule(enumCase), nullptr, enumCase->tagExpr);
        }
        return printer.getString();
    }
    return "unknown";
}


static String _formatDocumentation(String doc)
{
    bool hasDoxygen = false;
    // TODO: may want to use DocMarkdownWriter in the future to format the text.
    // For now just insert line breaks before `\param` and `\returns` markups.
    List<UnownedStringSlice> lines;
    StringUtil::split(doc.getUnownedSlice(), '\n', lines);
    StringBuilder result;
    StringBuilder returnDocSB;
    StringBuilder parameterDocSB;
    StringBuilder* currentSection = &result;
    bool isFirstParam = true;
    for (Index i = 0; i < lines.getCount(); i++)
    {
        auto trimedLine = lines[i].trimStart();
        if (trimedLine.startsWith("\\") || trimedLine.startsWith("@"))
        {
            hasDoxygen = true;
            trimedLine = trimedLine.tail(1);
            if (trimedLine.startsWith("returns "))
            {
                trimedLine = trimedLine.tail(8);
                currentSection = &returnDocSB;
                (*currentSection) << trimedLine << "  \n";
            }
            else if (trimedLine.startsWith("return "))
            {
                trimedLine = trimedLine.tail(7);
                currentSection = &returnDocSB;
                (*currentSection) << trimedLine << "  \n";
            }
            else if (trimedLine.startsWith("param"))
            {
                trimedLine = trimedLine.tail(5).trimStart();
                Index endOfParamName = 0;
                if (trimedLine.getLength() > 0 && trimedLine[0] == '[')
                {
                    endOfParamName = trimedLine.indexOf(']') + 1;
                    while (endOfParamName < trimedLine.getLength() &&
                           CharUtil::isWhitespace(trimedLine[endOfParamName]))
                        endOfParamName++;
                }
                while (endOfParamName < trimedLine.getLength() &&
                       !CharUtil::isWhitespace(trimedLine[endOfParamName]))
                    endOfParamName++;
                if (isFirstParam)
                {
                    isFirstParam = false;
                }
                else
                {
                    (*currentSection) << "  \n";
                }
                if (endOfParamName < trimedLine.getLength())
                {
                    parameterDocSB << "`" << trimedLine.head(endOfParamName) << "`";
                    trimedLine = trimedLine.tail(endOfParamName);
                }
                currentSection = &parameterDocSB;
                (*currentSection) << trimedLine;
            }
        }
        else
        {
            (*currentSection) << trimedLine << "\n";
        }
    }
    result << "\n";
    if (parameterDocSB.getLength())
    {
        result << "**Parameters**  \n";
        result << parameterDocSB.produceString() << "\n\n";
    }
    if (returnDocSB.getLength())
    {
        result << "**Returns**  \n";
        result << returnDocSB.produceString();
    }

    if (!hasDoxygen)
    {
        // For ordinary comments, we want to preserve line breaks in the original comment.
        result.clear();
        for (Index i = 0; i < lines.getCount(); i++)
        {
            result << lines[i] << "  \n";
        }
    }
    return result.produceString();
}

static void _tryGetDocumentation(StringBuilder& sb, WorkspaceVersion* workspace, Decl* decl)
{
    auto definingModule = getModuleDecl(decl);
    if (definingModule)
    {
        MarkupEntry* markupEntry = nullptr;
        if (decl->markup)
        {
            markupEntry = decl->markup;
        }
        else
        {
            auto markupAST = workspace->getOrCreateMarkupAST(definingModule);
            markupEntry = markupAST->getEntry(decl);
        }
        if (markupEntry)
        {
            sb << "\n";
            sb << _formatDocumentation(markupEntry->m_markup);
            sb << "\n";
        }
    }
}

void appendDefinitionLocation(StringBuilder& sb, Workspace* workspace, const HumaneSourceLoc& loc)
{
    auto path = loc.pathInfo.foundPath;
    Path::getCanonical(path, path);
    UnownedStringSlice pathSlice = path.getUnownedSlice();
    if (workspace)
    {
        for (auto& root : workspace->rootDirectories)
        {
            if (pathSlice.startsWith(root.getUnownedSlice()))
            {
                pathSlice = pathSlice.tail(root.getLength());
                if (pathSlice.startsWith("\\") || pathSlice.startsWith("/"))
                    pathSlice = pathSlice.tail(1);
                break;
            }
        }
    }
    sb << "Defined in " << pathSlice << "(" << loc.line << ")\n";
}

HumaneSourceLoc getModuleLoc(SourceManager* manager, ContainerDecl* moduleDecl)
{
    if (moduleDecl)
    {
        if (moduleDecl->members.getCount() && moduleDecl->members[0])
        {
            auto loc = moduleDecl->members[0]->loc;
            if (loc.isValid())
            {
                auto location = manager->getHumaneLoc(loc, SourceLocType::Actual);
                location.line = 1;
                location.column = 1;
                return location;
            }
        }
    }
    return HumaneSourceLoc();
}

SlangResult LanguageServer::hover(
    const LanguageServerProtocol::HoverParams& args,
    const JSONValue& responseId)
{
    auto result = m_core.hover(args);
    if (SLANG_FAILED(result.returnCode) || result.isNull)
    {
        m_connection->sendResult(NullResponse::get(), responseId);
        return SLANG_OK;
    }
    m_connection->sendResult(&result.result, responseId);
    return SLANG_OK;
}

LanguageServerResult<LanguageServerProtocol::Hover> LanguageServerCore::hover(
    const LanguageServerProtocol::HoverParams& args)
{
    String canonicalPath = uriToCanonicalPath(args.textDocument.uri);
    RefPtr<DocumentVersion> doc;
    if (!m_workspace->openedDocuments.tryGetValue(canonicalPath, doc))
    {
        return std::nullopt;
    }
    Index line, col;
    doc->zeroBasedUTF16LocToOneBasedUTF8Loc(args.position.line, args.position.character, line, col);

    auto version = m_workspace->getCurrentVersion();
    SLANG_AST_BUILDER_RAII(version->linkage->getASTBuilder());

    Module* parsedModule = version->getOrLoadModule(canonicalPath);
    if (!parsedModule)
    {
        return LanguageServerResult<LanguageServerProtocol::Hover>();
    }
    auto findResult = findASTNodesAt(
        doc.Ptr(),
        version->linkage->getSourceManager(),
        parsedModule->getModuleDecl(),
        ASTLookupType::Decl,
        canonicalPath.getUnownedSlice(),
        line,
        col);
    if (findResult.getCount() == 0 || findResult[0].path.getCount() == 0)
    {
        SLANG_LS_RETURN_ON_SUCCESS(tryGetMacroHoverInfo(version, doc, line, col));
        return std::nullopt;
    }
    StringBuilder sb;

    Hover hover;
    auto leafNode = findResult[0].path.getLast();

    auto maybeAppendAdditionalOverloadsHint = [&]()
    {
        Index numOverloads = 0;
        for (Index i = findResult[0].path.getCount() - 1; i >= 0; i--)
        {
            auto node = findResult[0].path[i];
            if (auto overloadExpr = as<OverloadedExpr>(node))
            {
                numOverloads = overloadExpr->lookupResult2.items.getCount();
            }
            else if (auto overloadedExpr2 = as<OverloadedExpr2>(node))
            {
                numOverloads = overloadedExpr2->candidiateExprs.getCount();
            }
        }
        if (numOverloads > 1)
        {
            sb << "\n +" << numOverloads - 1 << " overload";
            if (numOverloads > 2)
                sb << "s";
        }
    };

    auto fillDeclRefHoverInfo = [&](DeclRef<Decl> declRef, Name* name)
    {
        if (declRef.getDecl())
        {
            sb << "```\n" << getDeclSignatureString(declRef, version) << "\n```\n";

            _tryGetDocumentation(sb, version, declRef.getDecl());

            if (auto funcDecl = as<FunctionDeclBase>(declRef.getDecl()))
            {
                DiagnosticSink sink;
                SharedSemanticsContext semanticContext(
                    version->linkage,
                    getModule(funcDecl),
                    &sink);
                SemanticsVisitor semanticsVisitor(&semanticContext);

                auto assocDecls = semanticContext.getAssociatedDeclsForDecl(funcDecl);
                Decl* bwdDiff = nullptr;
                Decl* fwdDiff = nullptr;
                Decl* primalSubst = nullptr;
                auto getDeclFromExpr = [&](Expr* expr) -> Decl*
                {
                    if (auto declRefExpr = as<DeclRefExpr>(expr))
                        return declRefExpr->declRef.getDecl();
                    return nullptr;
                };
                for (auto& assocDecl : assocDecls)
                {
                    if (assocDecl->kind == DeclAssociationKind::ForwardDerivativeFunc)
                        fwdDiff = assocDecl->decl;
                    else if (assocDecl->kind == DeclAssociationKind::BackwardDerivativeFunc)
                        bwdDiff = assocDecl->decl;
                    else if (assocDecl->kind == DeclAssociationKind::PrimalSubstituteFunc)
                        primalSubst = assocDecl->decl;
                }
                bool isBackwardDifferentiable = false;
                bool isForwardDifferentiable = false;
                for (auto modifier : funcDecl->modifiers)
                {
                    if (auto bwdDiffModifier = as<BackwardDerivativeAttribute>(modifier))
                        bwdDiff = getDeclFromExpr(bwdDiffModifier->funcExpr);
                    else if (auto fwdDiffModifier = as<ForwardDerivativeAttribute>(modifier))
                        fwdDiff = getDeclFromExpr(fwdDiffModifier->funcExpr);
                    else if (auto primalSubstModifier = as<PrimalSubstituteAttribute>(modifier))
                        primalSubst = getDeclFromExpr(primalSubstModifier->funcExpr);
                    else if (as<ForwardDifferentiableAttribute>(modifier))
                        isForwardDifferentiable = true;
                    else if (as<BackwardDifferentiableAttribute>(modifier))
                        isBackwardDifferentiable = true;
                }
                if (primalSubst)
                {
                    for (auto modifier : primalSubst->modifiers)
                    {
                        if (as<ForwardDifferentiableAttribute>(modifier))
                            isForwardDifferentiable = true;
                        else if (as<BackwardDifferentiableAttribute>(modifier))
                            isBackwardDifferentiable = true;
                    }
                }
                if (isBackwardDifferentiable)
                {
                    sb << "\nForward and backward differentiable\n\n";
                }
                if (isForwardDifferentiable)
                {
                    sb << "\nForward differentiable\n\n";
                }
                if (fwdDiff && fwdDiff->getName())
                    sb << "Forward derivative: `" << fwdDiff->getName()->text << "`\n\n";
                if (bwdDiff && bwdDiff->getName())
                    sb << "Backward derivative: `" << bwdDiff->getName()->text << "`\n\n";
                if (primalSubst && primalSubst->getName())
                    sb << "Primal substitute: `" << primalSubst->getName()->text << "`\n\n";
            }

            auto humaneLoc = version->linkage->getSourceManager()->getHumaneLoc(
                declRef.getLoc(),
                SourceLocType::Actual);
            appendDefinitionLocation(sb, m_workspace, humaneLoc);
            maybeAppendAdditionalOverloadsHint();
            auto nodeHumaneLoc = version->linkage->getSourceManager()->getHumaneLoc(leafNode->loc);
            doc->oneBasedUTF8LocToZeroBasedUTF16Loc(
                nodeHumaneLoc.line,
                nodeHumaneLoc.column,
                hover.range.start.line,
                hover.range.start.character);
            hover.range.end = hover.range.start;
            if (!name)
                name = declRef.getName();
            if (auto ctorDecl = declRef.as<ConstructorDecl>())
            {
                auto parent = ctorDecl.getDecl()->parentDecl;
                if (parent)
                {
                    name = parent->getName();
                }
            }
            if (name)
            {
                hover.range.end.character =
                    hover.range.start.character +
                    (int)UTF8Util::calcUTF16CharCount(name->text.getUnownedSlice());
            }
        }
    };
    auto fillLoc = [&](SourceLoc loc)
    {
        auto humaneLoc =
            version->linkage->getSourceManager()->getHumaneLoc(loc, SourceLocType::Actual);
        doc->oneBasedUTF8LocToZeroBasedUTF16Loc(
            humaneLoc.line,
            humaneLoc.column,
            hover.range.start.line,
            hover.range.start.character);
        doc->oneBasedUTF8LocToZeroBasedUTF16Loc(
            humaneLoc.line,
            humaneLoc.column + doc->getTokenLength(humaneLoc.line, humaneLoc.column),
            hover.range.end.line,
            hover.range.end.character);
    };
    auto fillExprHoverInfo = [&](Expr* expr)
    {
        if (auto declRefExpr = as<DeclRefExpr>(expr))
            return fillDeclRefHoverInfo(declRefExpr->declRef, declRefExpr->name);
        else if (as<ThisExpr>(expr))
        {
            if (expr->type)
            {
                sb << "```\n"
                   << expr->type->toString() << " this"
                   << "\n```\n";
            }
            fillLoc(expr->loc);
        }
        else if (auto swizzleExpr = as<SwizzleExpr>(expr))
        {
            if (expr->type && swizzleExpr->base && swizzleExpr->base->type)
            {
                bool isTupleType = as<TupleType>(swizzleExpr->base->type) != nullptr;
                sb << "```\n";
                swizzleExpr->type->toText(sb);
                sb << " ";
                swizzleExpr->base->type->toText(sb);
                sb << ".";
                for (auto index : swizzleExpr->elementIndices)
                {
                    if (isTupleType || index > 4)
                        sb << "_" << index;
                    else
                        sb << "xyzw"[index];
                }
                sb << "\n```\n";
                fillLoc(expr->loc);
            }
        }
        else if (auto countOfExpr = as<CountOfExpr>(expr))
        {
            if (countOfExpr->sizedType)
            {
                if (auto foldedVal = as<ConstantIntVal>(CountOfIntVal::tryFoldOrNull(
                        version->linkage->getASTBuilder(),
                        expr->type.type,
                        countOfExpr->sizedType)))
                {
                    sb << "```\n"
                       << "countof(";
                    countOfExpr->sizedType->toText(sb);
                    sb << ") = " << foldedVal->getValue() << "\n```\n";
                    fillLoc(expr->loc);
                }
            }
        }
        if (const auto higherOrderExpr = as<HigherOrderInvokeExpr>(expr))
        {
            String documentation;
            String signature = getExprDeclSignature(expr, &documentation, nullptr);
            if (signature.getLength() == 0)
                return;
            sb << "```\n" << signature << "\n```\n";
            sb << documentation;
            maybeAppendAdditionalOverloadsHint();
            fillLoc(expr->loc);
        }
    };
    if (auto declRefExpr = as<DeclRefExpr>(leafNode))
    {
        fillDeclRefHoverInfo(declRefExpr->declRef, declRefExpr->name);
    }
    else if (auto overloadedExpr = as<OverloadedExpr>(leafNode))
    {
        LookupResultItem& item = overloadedExpr->lookupResult2.item;
        fillDeclRefHoverInfo(item.declRef, overloadedExpr->name);
    }
    else if (auto overloadedExpr2 = as<OverloadedExpr2>(leafNode))
    {
        if (overloadedExpr2->candidiateExprs.getCount() > 0)
        {
            auto candidateExpr = overloadedExpr2->candidiateExprs[0];
            fillExprHoverInfo(candidateExpr);
        }
    }
    else if (auto higherOrderExpr = as<HigherOrderInvokeExpr>(leafNode))
    {
        fillExprHoverInfo(higherOrderExpr);
    }
    else if (auto thisExprExpr = as<ThisExpr>(leafNode))
    {
        fillExprHoverInfo(thisExprExpr);
    }
    else if (auto countOfExpr = as<CountOfExpr>(leafNode))
    {
        fillExprHoverInfo(countOfExpr);
    }
    else if (auto swizzleExpr = as<SwizzleExpr>(leafNode))
    {
        fillExprHoverInfo(swizzleExpr);
    }
    else if (auto importDecl = as<ImportDecl>(leafNode))
    {
        auto moduleLoc =
            getModuleLoc(version->linkage->getSourceManager(), importDecl->importedModuleDecl);
        if (moduleLoc.pathInfo.hasFoundPath())
        {
            String path = moduleLoc.pathInfo.foundPath;
            Path::getCanonical(path, path);
            sb << path;
            auto humaneLoc = version->linkage->getSourceManager()->getHumaneLoc(
                importDecl->startLoc,
                SourceLocType::Actual);
            Index utf16Line, utf16Col;
            doc->oneBasedUTF8LocToZeroBasedUTF16Loc(
                humaneLoc.line,
                humaneLoc.column,
                utf16Line,
                utf16Col);
            hover.range.start.line = (int)utf16Line;
            hover.range.start.character = (int)utf16Col;
            humaneLoc = version->linkage->getSourceManager()->getHumaneLoc(
                importDecl->endLoc,
                SourceLocType::Actual);
            doc->oneBasedUTF8LocToZeroBasedUTF16Loc(
                humaneLoc.line,
                humaneLoc.column,
                utf16Line,
                utf16Col);
            hover.range.end.line = (int)utf16Line;
            hover.range.end.character = (int)utf16Col;
        }
    }
    else if (auto includeDeclBase = as<IncludeDeclBase>(leafNode))
    {
        auto moduleLoc =
            getModuleLoc(version->linkage->getSourceManager(), includeDeclBase->fileDecl);
        if (moduleLoc.pathInfo.hasFoundPath())
        {
            String path = moduleLoc.pathInfo.foundPath;
            Path::getCanonical(path, path);
            sb << path;
            auto humaneLoc = version->linkage->getSourceManager()->getHumaneLoc(
                includeDeclBase->startLoc,
                SourceLocType::Actual);
            Index utf16Line, utf16Col;
            doc->oneBasedUTF8LocToZeroBasedUTF16Loc(
                humaneLoc.line,
                humaneLoc.column,
                utf16Line,
                utf16Col);
            hover.range.start.line = (int)utf16Line;
            hover.range.start.character = (int)utf16Col;
            humaneLoc = version->linkage->getSourceManager()->getHumaneLoc(
                includeDeclBase->endLoc,
                SourceLocType::Actual);
            doc->oneBasedUTF8LocToZeroBasedUTF16Loc(
                humaneLoc.line,
                humaneLoc.column,
                utf16Line,
                utf16Col);
            hover.range.end.line = (int)utf16Line;
            hover.range.end.character = (int)utf16Col;
        }
    }
    else if (auto decl = as<Decl>(leafNode))
    {
        fillDeclRefHoverInfo(makeDeclRef(decl), nullptr);
    }
    else if (auto attr = as<Attribute>(leafNode))
    {
        fillDeclRefHoverInfo(makeDeclRef(attr->attributeDecl), nullptr);
        hover.range.end.character =
            hover.range.start.character + (int)attr->originalIdentifierToken.getContentLength();
    }
    if (sb.getLength() == 0)
    {
        return std::nullopt;
    }
    else
    {
        hover.contents.kind = "markdown";
        hover.contents.value = sb.produceString();
        return hover;
    }
}

SlangResult LanguageServer::gotoDefinition(
    const LanguageServerProtocol::DefinitionParams& args,
    const JSONValue& responseId)
{
    auto result = m_core.gotoDefinition(args);
    if (SLANG_FAILED(result.returnCode) || result.isNull)
    {
        m_connection->sendResult(NullResponse::get(), responseId);
        return SLANG_OK;
    }
    m_connection->sendResult(&result.result, responseId);
    return SLANG_OK;
}

LanguageServerResult<List<LanguageServerProtocol::Location>> LanguageServerCore::gotoDefinition(
    const LanguageServerProtocol::DefinitionParams& args)
{
    String canonicalPath = uriToCanonicalPath(args.textDocument.uri);
    RefPtr<DocumentVersion> doc;
    if (!m_workspace->openedDocuments.tryGetValue(canonicalPath, doc))
    {
        return std::nullopt;
    }
    Index line, col;
    doc->zeroBasedUTF16LocToOneBasedUTF8Loc(args.position.line, args.position.character, line, col);

    auto version = m_workspace->getCurrentVersion();
    SLANG_AST_BUILDER_RAII(version->linkage->getASTBuilder());

    Module* parsedModule = version->getOrLoadModule(canonicalPath);
    if (!parsedModule)
    {
        return std::nullopt;
    }
    auto findResult = findASTNodesAt(
        doc.Ptr(),
        version->linkage->getSourceManager(),
        parsedModule->getModuleDecl(),
        ASTLookupType::Decl,
        canonicalPath.getUnownedSlice(),
        line,
        col);
    if (findResult.getCount() == 0 || findResult[0].path.getCount() == 0)
    {
        SLANG_LS_RETURN_ON_SUCCESS(tryGotoMacroDefinition(version, doc, line, col));
        SLANG_LS_RETURN_ON_SUCCESS(tryGotoFileInclude(version, doc, line));
        return std::nullopt;
    }
    struct LocationResult
    {
        HumaneSourceLoc loc;
        int length;
    };
    List<LocationResult> locations;
    auto leafNode = findResult[0].path.getLast();
    if (auto declRefExpr = as<DeclRefExpr>(leafNode))
    {
        if (declRefExpr->declRef.getDecl())
        {
            auto location = version->linkage->getSourceManager()->getHumaneLoc(
                declRefExpr->declRef.getNameLoc().isValid() ? declRefExpr->declRef.getNameLoc()
                                                            : declRefExpr->declRef.getLoc(),
                SourceLocType::Actual);
            auto name = declRefExpr->declRef.getName();
            locations.add(LocationResult{
                location,
                name ? (int)UTF8Util::calcUTF16CharCount(name->text.getUnownedSlice()) : 0});
        }
    }
    else if (auto overloadedExpr = as<OverloadedExpr>(leafNode))
    {
        if (overloadedExpr->lookupResult2.items.getCount())
        {
            for (auto item : overloadedExpr->lookupResult2.items)
            {
                auto location = version->linkage->getSourceManager()->getHumaneLoc(
                    item.declRef.getNameLoc(),
                    SourceLocType::Actual);
                auto name = item.declRef.getName();
                locations.add(LocationResult{
                    location,
                    name ? (int)UTF8Util::calcUTF16CharCount(name->text.getUnownedSlice()) : 0});
            }
        }
        else
        {
            LookupResultItem& item = overloadedExpr->lookupResult2.item;
            if (item.declRef.getDecl() != nullptr)
            {
                auto location = version->linkage->getSourceManager()->getHumaneLoc(
                    item.declRef.getNameLoc(),
                    SourceLocType::Actual);
                auto name = item.declRef.getName();
                locations.add(LocationResult{
                    location,
                    name ? (int)UTF8Util::calcUTF16CharCount(name->text.getUnownedSlice()) : 0});
            }
        }
    }
    else if (auto importDecl = as<ImportDecl>(leafNode))
    {
        auto location =
            getModuleLoc(version->linkage->getSourceManager(), importDecl->importedModuleDecl);
        if (location.pathInfo.hasFoundPath())
        {
            locations.add(LocationResult{location, 0});
        }
    }
    else if (auto includeDeclBase = as<IncludeDeclBase>(leafNode))
    {
        auto location =
            getModuleLoc(version->linkage->getSourceManager(), includeDeclBase->fileDecl);
        if (location.pathInfo.hasFoundPath())
        {
            locations.add(LocationResult{location, 0});
        }
    }
    if (locations.getCount() == 0)
    {
        return std::nullopt;
    }
    else
    {
        List<Location> results;
        for (auto loc : locations)
        {
            Location result;
            if (File::exists(loc.loc.pathInfo.foundPath))
            {
                result.uri =
                    URI::fromLocalFilePath(loc.loc.pathInfo.foundPath.getUnownedSlice()).uri;
                doc->oneBasedUTF8LocToZeroBasedUTF16Loc(
                    loc.loc.line,
                    loc.loc.column,
                    result.range.start.line,
                    result.range.start.character);
                result.range.end = result.range.start;
                result.range.end.character += loc.length;
                results.add(result);
            }
        }
        return results;
    }
}

template<typename Func>
struct Deferred
{
    Func f;
    Deferred(const Func& func)
        : f(func)
    {
    }
    ~Deferred() { f(); }
};
template<typename Func>
Deferred<Func> makeDeferred(const Func& f)
{
    return Deferred<Func>(f);
}

SlangResult LanguageServer::completion(
    const LanguageServerProtocol::CompletionParams& args,
    const JSONValue& responseId)
{
    auto result = m_core.completion(args);
    if (SLANG_FAILED(result.returnCode) || result.isNull)
        m_connection->sendResult(NullResponse::get(), responseId);
    else if (result.result.items.getCount())
        m_connection->sendResult(&result.result.items, responseId);
    else
        m_connection->sendResult(&result.result.textEditItems, responseId);
    return SLANG_OK;
}

LanguageServerResult<CompletionResult> LanguageServerCore::completion(
    const LanguageServerProtocol::CompletionParams& args)
{
    String canonicalPath = uriToCanonicalPath(args.textDocument.uri);

    RefPtr<DocumentVersion> doc;
    if (!m_workspace->openedDocuments.tryGetValue(canonicalPath, doc))
    {
        return std::nullopt;
    }

    // Don't show completion at case label or after single '>' operator.
    if (args.context.triggerKind == LanguageServerProtocol::kCompletionTriggerKindTriggerCharacter)
    {
        char requiredPrevChar = 0;
        if (args.context.triggerCharacter == ":")
            requiredPrevChar = ':';
        else if (args.context.triggerCharacter == ">")
            requiredPrevChar = '-';
        if (requiredPrevChar != 0)
        {
            // Check if the previous character is the required character (':' or '-'
            auto line = doc->getLine((Int)args.position.line + 1);
            auto prevCharPos = args.position.character - 2;
            if (prevCharPos >= 0 && prevCharPos < line.getLength() &&
                line[prevCharPos] != requiredPrevChar)
            {
                return std::nullopt;
            }
        }
    }

    Index utf8Line, utf8Col;
    doc->zeroBasedUTF16LocToOneBasedUTF8Loc(
        args.position.line,
        args.position.character,
        utf8Line,
        utf8Col);
    auto cursorOffset = doc->getOffset(utf8Line, utf8Col);
    if (cursorOffset == -1 || doc->getText().getLength() == 0)
    {
        return std::nullopt;
    }

    // Ajust cursor position to the beginning of the current/last identifier.
    cursorOffset--;
    while (cursorOffset > 0 && _isIdentifierChar(doc->getText()[cursorOffset]))
    {
        cursorOffset--;
    }

    // Never show suggestions when the user is typing a number.
    if (cursorOffset + 1 >= 0 && cursorOffset + 1 < doc->getText().getLength() &&
        CharUtil::isDigit(doc->getText()[cursorOffset + 1]))
    {
        return std::nullopt;
    }

    // Always create a new workspace version for the completion request since we
    // will use a modified source.
    auto version = m_workspace->createVersionForCompletion();
    SLANG_AST_BUILDER_RAII(version->linkage->getASTBuilder());

    auto moduleName = getMangledNameFromNameString(canonicalPath.getUnownedSlice());
    version->linkage->contentAssistInfo.cursorLine = utf8Line;
    version->linkage->contentAssistInfo.cursorCol = utf8Col;
    Slang::CompletionContext context;
    context.server = this;
    context.cursorOffset = cursorOffset;
    context.version = version;
    context.doc = doc.Ptr();
    context.canonicalPath = canonicalPath.getUnownedSlice();
    context.line = utf8Line;
    context.col = utf8Col;
    context.commitCharacterBehavior = m_commitCharacterBehavior;
    if (args.context.triggerKind == kCompletionTriggerKindTriggerCharacter &&
        (args.context.triggerCharacter == " " || args.context.triggerCharacter == "[" ||
         args.context.triggerCharacter == "("))
    {
        // Never use commit character if completion request is triggerred by these characters to
        // prevent annoyance.
        context.commitCharacterBehavior = CommitCharacterBehavior::Disabled;
    }

    SLANG_LS_RETURN_ON_SUCCESS(context.tryCompleteInclude());
    SLANG_LS_RETURN_ON_SUCCESS(context.tryCompleteImport());

    if (args.context.triggerKind ==
            LanguageServerProtocol::kCompletionTriggerKindTriggerCharacter &&
        (args.context.triggerCharacter == "\"" || args.context.triggerCharacter == "/"))
    {
        // Trigger characters '"' and '/' are for include only.
        return std::nullopt;
    }


    // Insert a completion request token at cursor position.
    auto originalText = doc->getText();
    StringBuilder newText;
    newText << originalText.getUnownedSlice().head(cursorOffset + 1) << "#?"
            << originalText.getUnownedSlice().tail(cursorOffset + 1);
    doc->setText(newText.produceString());
    auto restoreDocText = makeDeferred([&]() { doc->setText(originalText); });

    Module* parsedModule = version->getOrLoadModule(canonicalPath);
    if (!parsedModule)
    {
        return std::nullopt;
    }

    context.parsedModule = parsedModule;
    SLANG_LS_RETURN_ON_SUCCESS(context.tryCompleteAttributes());

    // Don't generate completion suggestions after typing '['.
    if (args.context.triggerKind ==
            LanguageServerProtocol::kCompletionTriggerKindTriggerCharacter &&
        args.context.triggerCharacter == "[")
    {
        return std::nullopt;
    }

    SLANG_LS_RETURN_ON_SUCCESS(context.tryCompleteHLSLSemantic());
    SLANG_LS_RETURN_ON_SUCCESS(context.tryCompleteMemberAndSymbol());
    return std::nullopt;
}

SlangResult LanguageServer::completionResolve(
    const LanguageServerProtocol::CompletionItem& args,
    const LanguageServerProtocol::TextEditCompletionItem& editItem,
    const JSONValue& responseId)
{
    auto result = m_core.completionResolve(args, editItem);
    if (SLANG_FAILED(result.returnCode) || result.isNull)
    {
        m_connection->sendResult(NullResponse::get(), responseId);
        return SLANG_OK;
    }
    m_connection->sendResult(&result.result, responseId);
    return SLANG_OK;
}

LanguageServerResult<LanguageServerProtocol::CompletionItem> LanguageServerCore::completionResolve(
    const LanguageServerProtocol::CompletionItem& args,
    const LanguageServerProtocol::TextEditCompletionItem& editItem)
{
    if (args.data.getLength() == 0)
    {
        if (editItem.textEdit.newText.getLength())
        {
            return std::nullopt;
        }
        return args;
    }

    LanguageServerProtocol::CompletionItem resolvedItem = args;
    int itemId = stringToInt(args.data);
    auto version = m_workspace->getCurrentCompletionVersion();
    if (!version || !version->linkage)
    {
        return resolvedItem;
    }
    SLANG_AST_BUILDER_RAII(version->linkage->getASTBuilder());
    auto& candidateItems = version->linkage->contentAssistInfo.completionSuggestions.candidateItems;
    if (itemId >= 0 && itemId < candidateItems.getCount())
    {
        auto declRef = candidateItems[itemId].declRef;
        resolvedItem.detail = getDeclSignatureString(declRef, version);
        StringBuilder docSB;
        _tryGetDocumentation(docSB, version, declRef.getDecl());
        resolvedItem.documentation.value = docSB.produceString();
        resolvedItem.documentation.kind = "markdown";
    }
    return resolvedItem;
}

SlangResult LanguageServer::semanticTokens(
    const LanguageServerProtocol::SemanticTokensParams& args,
    const JSONValue& responseId)
{
    auto result = m_core.semanticTokens(args);
    if (SLANG_FAILED(result.returnCode) || result.isNull)
    {
        m_connection->sendResult(NullResponse::get(), responseId);
        return SLANG_OK;
    }
    m_connection->sendResult(&result.result, responseId);
    return SLANG_OK;
}

LanguageServerResult<LanguageServerProtocol::SemanticTokens> LanguageServerCore::semanticTokens(
    const LanguageServerProtocol::SemanticTokensParams& args)
{
    String canonicalPath = uriToCanonicalPath(args.textDocument.uri);

    RefPtr<DocumentVersion> doc;
    if (!m_workspace->openedDocuments.tryGetValue(canonicalPath, doc))
    {
        return std::nullopt;
    }

    auto version = m_workspace->getCurrentVersion();
    SLANG_AST_BUILDER_RAII(version->linkage->getASTBuilder());

    Module* parsedModule = version->getOrLoadModule(canonicalPath);
    if (!parsedModule)
    {
        return std::nullopt;
    }

    auto tokens = getSemanticTokens(
        version->linkage,
        parsedModule,
        canonicalPath.getUnownedSlice(),
        doc.Ptr());
    for (auto& token : tokens)
    {
        Index line, col;
        doc->oneBasedUTF8LocToZeroBasedUTF16Loc(token.line, token.col, line, col);
        Index lineEnd, colEnd;
        doc->oneBasedUTF8LocToZeroBasedUTF16Loc(
            token.line,
            token.col + token.length,
            lineEnd,
            colEnd);
        token.line = (int)line;
        token.col = (int)col;
        token.length = (int)(colEnd - col);
    }
    SemanticTokens response;
    response.resultId = "";
    response.data = getEncodedTokens(tokens);
    return response;
}

String LanguageServerCore::getExprDeclSignature(
    Expr* expr,
    String* outDocumentation,
    List<Slang::Range<Index>>* outParamRanges)
{
    if (auto declRefExpr = as<DeclRefExpr>(expr))
    {
        return getDeclRefSignature(declRefExpr->declRef, outDocumentation, outParamRanges);
    }

    auto higherOrderExpr = as<HigherOrderInvokeExpr>(expr);
    auto declRefExpr = as<DeclRefExpr>(getInnerMostExprFromHigherOrderExpr(higherOrderExpr));
    if (!declRefExpr)
        return String();
    if (!declRefExpr->declRef.getDecl())
        return String();
    auto funcType = as<FuncType>(higherOrderExpr->type);
    if (!funcType)
        return String();

    auto version = m_workspace->getCurrentVersion();
    SLANG_AST_BUILDER_RAII(version->linkage->getASTBuilder());

    SignatureInformation sigInfo;

    ASTPrinter printer(
        version->linkage->getASTBuilder(),
        ASTPrinter::OptionFlag::ParamNames | ASTPrinter::OptionFlag::NoInternalKeywords |
            ASTPrinter::OptionFlag::SimplifiedBuiltinType);

    printer.addDeclKindPrefix(declRefExpr->declRef.getDecl());
    auto inner = higherOrderExpr;
    int closingParentCount = 0;
    while (inner)
    {
        printer.getStringBuilder() << getHigherOrderOperatorName(inner) << "(";
        closingParentCount++;
        inner = as<HigherOrderInvokeExpr>(inner->baseFunction);
    }
    printer.addDeclPath(declRefExpr->declRef);
    for (int i = 0; i < closingParentCount; i++)
        printer.getStringBuilder() << ")";
    bool isFirst = true;
    printer.getStringBuilder() << "(";
    int paramIndex = 0;
    for (auto param : funcType->getParamTypes())
    {
        if (!isFirst)
            printer.getStringBuilder() << ", ";
        Slang::Range<Index> range;
        range.begin = printer.getStringBuilder().getLength();
        if (paramIndex < higherOrderExpr->newParameterNames.getCount())
        {
            if (higherOrderExpr->newParameterNames[paramIndex])
            {
                printer.getStringBuilder()
                    << higherOrderExpr->newParameterNames[paramIndex]->text << ": ";
            }
        }
        printer.addType(param);
        range.end = printer.getStringBuilder().getLength();
        if (outParamRanges)
            outParamRanges->add(range);
        isFirst = false;
        paramIndex++;
    }
    printer.getStringBuilder() << ") -> ";
    printer.addType(funcType->getResultType());

    if (outDocumentation)
    {
        StringBuilder docSB;
        auto humaneLoc = version->linkage->getSourceManager()->getHumaneLoc(
            declRefExpr->declRef.getLoc(),
            SourceLocType::Actual);
        _tryGetDocumentation(docSB, version, declRefExpr->declRef.getDecl());
        appendDefinitionLocation(docSB, m_workspace, humaneLoc);
        *outDocumentation = docSB.produceString();
    }

    return printer.getString();
}

String LanguageServerCore::getDeclRefSignature(
    DeclRef<Decl> declRef,
    String* outDocumentation,
    List<Slang::Range<Index>>* outParamRanges)
{
    auto version = m_workspace->getCurrentVersion();
    SLANG_AST_BUILDER_RAII(version->linkage->getASTBuilder());

    ASTPrinter printer(
        version->linkage->getASTBuilder(),
        ASTPrinter::OptionFlag::ParamNames | ASTPrinter::OptionFlag::NoInternalKeywords |
            ASTPrinter::OptionFlag::SimplifiedBuiltinType);

    printer.addDeclKindPrefix(declRef.getDecl());
    printer.addDeclPath(declRef);
    printer.addDeclParams(declRef, outParamRanges);
    printer.addDeclResultType(declRef);

    if (outDocumentation)
    {
        StringBuilder docSB;
        auto humaneLoc = version->linkage->getSourceManager()->getHumaneLoc(
            declRef.getLoc(),
            SourceLocType::Actual);
        _tryGetDocumentation(docSB, version, declRef.getDecl());
        appendDefinitionLocation(docSB, m_workspace, humaneLoc);
        *outDocumentation = docSB.produceString();
    }
    return printer.getString();
}

SlangResult LanguageServer::signatureHelp(
    const LanguageServerProtocol::SignatureHelpParams& args,
    const JSONValue& responseId)
{
    auto result = m_core.signatureHelp(args);
    if (SLANG_FAILED(result.returnCode) || result.isNull)
    {
        m_connection->sendResult(NullResponse::get(), responseId);
        return SLANG_OK;
    }
    m_connection->sendResult(&result.result, responseId);
    return SLANG_OK;
}

LanguageServerResult<LanguageServerProtocol::SignatureHelp> LanguageServerCore::signatureHelp(
    const LanguageServerProtocol::SignatureHelpParams& args)
{
    String canonicalPath = uriToCanonicalPath(args.textDocument.uri);
    RefPtr<DocumentVersion> doc;
    if (!m_workspace->openedDocuments.tryGetValue(canonicalPath, doc))
    {
        return std::nullopt;
    }
    Index line, col;
    doc->zeroBasedUTF16LocToOneBasedUTF8Loc(args.position.line, args.position.character, line, col);

    auto version = m_workspace->getCurrentVersion();
    SLANG_AST_BUILDER_RAII(version->linkage->getASTBuilder());

    Module* parsedModule = version->getOrLoadModule(canonicalPath);
    if (!parsedModule)
    {
        return std::nullopt;
    }

    auto findResult = findASTNodesAt(
        doc.Ptr(),
        version->linkage->getSourceManager(),
        parsedModule->getModuleDecl(),
        ASTLookupType::Invoke,
        canonicalPath.getUnownedSlice(),
        line,
        col);

    if (findResult.getCount() == 0)
    {
        return std::nullopt;
    }

    AppExprBase* appExpr = nullptr;
    auto& declPath = findResult[0].path;
    Loc currentLoc = {args.position.line + 1, args.position.character + 1};
    for (Index i = declPath.getCount() - 1; i >= 0; i--)
    {
        if (auto expr = as<AppExprBase>(declPath[i]))
        {
            // Find the inner most invoke expr that has source token info.
            // This allows us to skip the invoke expr nodes for operators/implcit casts.
            if (expr->argumentDelimeterLocs.getCount())
            {
                auto start = Loc::fromSourceLoc(
                    version->linkage->getSourceManager(),
                    expr->argumentDelimeterLocs.getFirst());
                auto end = Loc::fromSourceLoc(
                    version->linkage->getSourceManager(),
                    expr->argumentDelimeterLocs.getLast());
                if (start < currentLoc && currentLoc <= end)
                {
                    appExpr = expr;
                    break;
                }
            }
        }
    }
    if (!appExpr)
    {
        return std::nullopt;
    }

    if (appExpr->argumentDelimeterLocs.getCount() == 0)
    {
        return std::nullopt;
    }

    auto funcExpr = appExpr->functionExpr;
    if (appExpr->originalFunctionExpr)
    {
        bool useOriginalExpr = true;
        if (auto originalDeclRefExpr = as<DeclRefExpr>(appExpr->originalFunctionExpr))
        {
            if (!originalDeclRefExpr->declRef)
            {
                useOriginalExpr = false;
            }
        }
        if (useOriginalExpr)
            funcExpr = appExpr->originalFunctionExpr;
    }
    if (!funcExpr)
    {
        return std::nullopt;
    }

    SignatureHelp response;
    auto addDeclRef = [&](DeclRef<Decl> declRef)
    {
        if (!declRef.getDecl())
            return;

        SignatureInformation sigInfo;

        List<Slang::Range<Index>> paramRanges;
        String documentation;
        sigInfo.label = getDeclRefSignature(declRef, &documentation, &paramRanges);
        sigInfo.documentation.value = documentation;
        sigInfo.documentation.kind = "markdown";

        for (auto& range : paramRanges)
        {
            ParameterInformation paramInfo;
            paramInfo.label[0] = (uint32_t)range.begin;
            paramInfo.label[1] = (uint32_t)range.end;
            sigInfo.parameters.add(paramInfo);
        }
        response.signatures.add(sigInfo);
    };

    auto addExpr = [&](Expr* expr)
    {
        SignatureInformation sigInfo;
        List<Slang::Range<Index>> paramRanges;
        String documentation;
        sigInfo.label = getExprDeclSignature(expr, &documentation, &paramRanges);
        if (sigInfo.label.getLength() == 0)
            return;
        sigInfo.documentation.value = documentation;
        sigInfo.documentation.kind = "markdown";
        for (auto& range : paramRanges)
        {
            ParameterInformation paramInfo;
            paramInfo.label[0] = (uint32_t)range.begin;
            paramInfo.label[1] = (uint32_t)range.end;
            sigInfo.parameters.add(paramInfo);
        }
        response.signatures.add(sigInfo);
    };

    auto addFuncType = [&](FuncType* funcType)
    {
        SignatureInformation sigInfo;

        List<Slang::Range<Index>> paramRanges;
        ASTPrinter printer(
            version->linkage->getASTBuilder(),
            ASTPrinter::OptionFlag::ParamNames | ASTPrinter::OptionFlag::NoInternalKeywords |
                ASTPrinter::OptionFlag::SimplifiedBuiltinType);

        printer.getStringBuilder() << "func (";
        bool isFirst = true;
        for (auto param : funcType->getParamTypes())
        {
            if (!isFirst)
                printer.getStringBuilder() << ", ";
            Slang::Range<Index> range;
            range.begin = printer.getStringBuilder().getLength();
            printer.addType(param);
            range.end = printer.getStringBuilder().getLength();
            paramRanges.add(range);
            isFirst = false;
        }
        printer.getStringBuilder() << ") -> ";
        printer.addType(funcType->getResultType());
        sigInfo.label = printer.getString();
        for (auto& range : paramRanges)
        {
            ParameterInformation paramInfo;
            paramInfo.label[0] = (uint32_t)range.begin;
            paramInfo.label[1] = (uint32_t)range.end;
            sigInfo.parameters.add(paramInfo);
        }
        response.signatures.add(sigInfo);
    };

    if (auto declRefExpr = as<DeclRefExpr>(funcExpr))
    {
        if (auto aggDeclRef = as<AggTypeDecl>(declRefExpr->declRef))
        {
            // Look for initializers
            for (auto member :
                 getMembersOfType<ConstructorDecl>(version->linkage->getASTBuilder(), aggDeclRef))
            {
                addDeclRef(member);
            }
        }
        else
        {
            addDeclRef(declRefExpr->declRef);
        }
    }
    else if (auto overloadedExpr = as<OverloadedExpr>(funcExpr))
    {
        for (auto item : overloadedExpr->lookupResult2)
        {
            addDeclRef(item.declRef);
        }
    }
    else if (auto overloadedExpr2 = as<OverloadedExpr2>(funcExpr))
    {
        for (auto item : overloadedExpr2->candidiateExprs)
        {
            addExpr(item);
        }
    }
    else if (auto higherOrder = as<HigherOrderInvokeExpr>(funcExpr))
    {
        addExpr(higherOrder);
    }
    else if (auto funcType = as<FuncType>(funcExpr->type.type))
    {
        addFuncType(funcType);
    }
    response.activeSignature = 0;
    response.activeParameter = 0;
    for (int i = 1; i < appExpr->argumentDelimeterLocs.getCount(); i++)
    {
        auto delimLoc = version->linkage->getSourceManager()->getHumaneLoc(
            appExpr->argumentDelimeterLocs[i],
            SourceLocType::Actual);
        if (delimLoc.line > args.position.line + 1 ||
            delimLoc.line == args.position.line + 1 &&
                delimLoc.column >= args.position.character + 1)
        {
            response.activeParameter = i - 1;
            break;
        }
    }

    return response;
}

SlangResult LanguageServer::documentSymbol(
    const LanguageServerProtocol::DocumentSymbolParams& args,
    const JSONValue& responseId)
{
    auto result = m_core.documentSymbol(args);
    if (SLANG_FAILED(result.returnCode) || result.isNull)
    {
        m_connection->sendResult(NullResponse::get(), responseId);
        return SLANG_OK;
    }
    m_connection->sendResult(&result.result, responseId);
    return SLANG_OK;
}

LanguageServerResult<List<LanguageServerProtocol::DocumentSymbol>> LanguageServerCore::
    documentSymbol(const LanguageServerProtocol::DocumentSymbolParams& args)
{
    String canonicalPath = uriToCanonicalPath(args.textDocument.uri);
    RefPtr<DocumentVersion> doc;
    if (!m_workspace->openedDocuments.tryGetValue(canonicalPath, doc))
    {
        return std::nullopt;
    }
    auto version = m_workspace->getCurrentVersion();
    SLANG_AST_BUILDER_RAII(version->linkage->getASTBuilder());

    Module* parsedModule = version->getOrLoadModule(canonicalPath);
    if (!parsedModule)
    {
        return std::nullopt;
    }
    List<DocumentSymbol> symbols = getDocumentSymbols(
        version->linkage,
        parsedModule,
        canonicalPath.getUnownedSlice(),
        doc.Ptr());
    return symbols;
}

SlangResult LanguageServer::inlayHint(
    const LanguageServerProtocol::InlayHintParams& args,
    const JSONValue& responseId)
{
    auto result = m_core.inlayHint(args);
    if (SLANG_FAILED(result.returnCode) || result.isNull)
    {
        m_connection->sendResult(NullResponse::get(), responseId);
        return SLANG_OK;
    }
    m_connection->sendResult(&result.result, responseId);
    return SLANG_OK;
}

LanguageServerResult<List<LanguageServerProtocol::InlayHint>> LanguageServerCore::inlayHint(
    const LanguageServerProtocol::InlayHintParams& args)
{
    String canonicalPath = uriToCanonicalPath(args.textDocument.uri);
    RefPtr<DocumentVersion> doc;
    if (!m_workspace->openedDocuments.tryGetValue(canonicalPath, doc))
    {
        return std::nullopt;
    }
    auto version = m_workspace->getCurrentVersion();
    SLANG_AST_BUILDER_RAII(version->linkage->getASTBuilder());

    Module* parsedModule = version->getOrLoadModule(canonicalPath);
    if (!parsedModule)
    {
        return std::nullopt;
    }
    List<InlayHint> hints = getInlayHints(
        version->linkage,
        parsedModule,
        canonicalPath.getUnownedSlice(),
        doc.Ptr(),
        args.range,
        m_inlayHintOptions);
    return hints;
}

List<LanguageServerProtocol::TextEdit> translateTextEdits(DocumentVersion* doc, List<Edit>& edits)
{
    List<LanguageServerProtocol::TextEdit> result;
    for (auto& edit : edits)
    {
        LanguageServerProtocol::TextEdit tedit;
        Index line, col;
        Index zeroBasedLine, zeroBasedCol;
        doc->offsetToLineCol(edit.offset, line, col);
        doc->oneBasedUTF8LocToZeroBasedUTF16Loc(line, col, zeroBasedLine, zeroBasedCol);
        tedit.range.start.line = (int)zeroBasedLine;
        tedit.range.start.character = (int)zeroBasedCol;
        doc->offsetToLineCol(edit.offset + edit.length, line, col);
        doc->oneBasedUTF8LocToZeroBasedUTF16Loc(line, col, zeroBasedLine, zeroBasedCol);
        tedit.range.end.line = (int)zeroBasedLine;
        tedit.range.end.character = (int)zeroBasedCol;
        tedit.newText = edit.text;
        result.add(_Move(tedit));
    }
    return result;
}

SlangResult LanguageServer::formatting(
    const LanguageServerProtocol::DocumentFormattingParams& args,
    const JSONValue& responseId)
{
    auto result = m_core.formatting(args);
    if (SLANG_FAILED(result.returnCode) || result.isNull)
    {
        m_connection->sendResult(NullResponse::get(), responseId);
        return SLANG_OK;
    }
    m_connection->sendResult(&result.result, responseId);
    return SLANG_OK;
}

LanguageServerResult<List<LanguageServerProtocol::TextEdit>> LanguageServerCore::formatting(
    const LanguageServerProtocol::DocumentFormattingParams& args)
{
    String canonicalPath = uriToCanonicalPath(args.textDocument.uri);
    RefPtr<DocumentVersion> doc;
    if (!m_workspace->openedDocuments.tryGetValue(canonicalPath, doc))
    {
        return std::nullopt;
    }
    if (m_formatOptions.clangFormatLocation.getLength() == 0)
        m_formatOptions.clangFormatLocation = findClangFormatTool();
    auto options = getFormatOptions(m_workspace, m_formatOptions);
    options.fileName = canonicalPath;
    List<TextRange> exclusionRange =
        extractFormattingExclusionRanges(doc->getText().getUnownedSlice());
    auto edits =
        formatSource(doc->getText().getUnownedSlice(), -1, -1, -1, exclusionRange, options);
    auto textEdits = translateTextEdits(doc, edits);
    return textEdits;
}

SlangResult LanguageServer::rangeFormatting(
    const LanguageServerProtocol::DocumentRangeFormattingParams& args,
    const JSONValue& responseId)
{
    auto result = m_core.rangeFormatting(args);
    if (SLANG_FAILED(result.returnCode) || result.isNull)
    {
        m_connection->sendResult(NullResponse::get(), responseId);
        return SLANG_OK;
    }
    m_connection->sendResult(&result.result, responseId);
    return SLANG_OK;
}

LanguageServerResult<List<LanguageServerProtocol::TextEdit>> LanguageServerCore::rangeFormatting(
    const LanguageServerProtocol::DocumentRangeFormattingParams& args)
{
    String canonicalPath = uriToCanonicalPath(args.textDocument.uri);
    RefPtr<DocumentVersion> doc;
    if (!m_workspace->openedDocuments.tryGetValue(canonicalPath, doc))
    {
        return std::nullopt;
    }
    Index endLine, endCol;
    doc->zeroBasedUTF16LocToOneBasedUTF8Loc(
        args.range.end.line,
        args.range.end.character,
        endLine,
        endCol);
    Index endOffset = doc->getOffset(endLine, endCol);
    if (m_formatOptions.clangFormatLocation.getLength() == 0)
        m_formatOptions.clangFormatLocation = findClangFormatTool();
    auto options = getFormatOptions(m_workspace, m_formatOptions);
    if (!m_formatOptions.allowLineBreakInRangeFormatting)
        options.behavior = FormatBehavior::PreserveLineBreak;
    List<TextRange> exclusionRange =
        extractFormattingExclusionRanges(doc->getText().getUnownedSlice());
    auto edits = formatSource(
        doc->getText().getUnownedSlice(),
        args.range.start.line,
        args.range.end.line,
        endOffset,
        exclusionRange,
        options);
    auto textEdits = translateTextEdits(doc, edits);
    return textEdits;
}

SlangResult LanguageServer::onTypeFormatting(
    const LanguageServerProtocol::DocumentOnTypeFormattingParams& args,
    const JSONValue& responseId)
{
    auto result = m_core.onTypeFormatting(args);
    if (SLANG_FAILED(result.returnCode) || result.isNull)
    {
        m_connection->sendResult(NullResponse::get(), responseId);
        return SLANG_OK;
    }
    m_connection->sendResult(&result.result, responseId);
    return SLANG_OK;
}

LanguageServerResult<List<LanguageServerProtocol::TextEdit>> LanguageServerCore::onTypeFormatting(
    const LanguageServerProtocol::DocumentOnTypeFormattingParams& args)
{
    String canonicalPath = uriToCanonicalPath(args.textDocument.uri);
    RefPtr<DocumentVersion> doc;
    if (!m_workspace->openedDocuments.tryGetValue(canonicalPath, doc))
    {
        return std::nullopt;
    }
    if (args.ch == ":" && !doc->getLine((Int)args.position.line + 1).trim().startsWith("case "))
    {
        return std::nullopt;
    }
    if (m_formatOptions.clangFormatLocation.getLength() == 0)
        m_formatOptions.clangFormatLocation = findClangFormatTool();
    Index line, col;
    doc->zeroBasedUTF16LocToOneBasedUTF8Loc(args.position.line, args.position.character, line, col);
    auto cursorOffset = doc->getOffset(line, col);
    auto options = getFormatOptions(m_workspace, m_formatOptions);
    if (!m_formatOptions.allowLineBreakInOnTypeFormatting)
        options.behavior = FormatBehavior::PreserveLineBreak;
    List<TextRange> exclusionRange =
        extractFormattingExclusionRanges(doc->getText().getUnownedSlice());
    auto edits = formatSource(
        doc->getText().getUnownedSlice(),
        args.position.line,
        args.position.line,
        cursorOffset,
        exclusionRange,
        options);
    auto textEdits = translateTextEdits(doc, edits);
    return textEdits;
}

void LanguageServer::publishDiagnostics()
{
    if (std::chrono::system_clock::now() - m_lastDiagnosticUpdateTime <
        std::chrono::milliseconds(1000))
    {
        return;
    }
    m_lastDiagnosticUpdateTime = std::chrono::system_clock::now();

    auto version = m_core.m_workspace->getCurrentVersion();
    SLANG_AST_BUILDER_RAII(version->linkage->getASTBuilder());

    // Send updates to clear diagnostics for files that no longer have any messages.
    List<String> filesToRemove;
    for (const auto& [filepath, _] : m_lastPublishedDiagnostics)
    {
        if (!version->diagnostics.containsKey(filepath))
        {
            PublishDiagnosticsParams args;
            args.uri = URI::fromLocalFilePath(filepath.getUnownedSlice()).uri;
            m_connection->sendCall(UnownedStringSlice("textDocument/publishDiagnostics"), &args);
            filesToRemove.add(filepath);
        }
    }
    for (auto& toRemove : filesToRemove)
    {
        m_lastPublishedDiagnostics.remove(toRemove);
    }
    // Send updates for any files whose diagnostic messages has changed since last update.
    for (const auto& [listKey, listValue] : version->diagnostics)
    {
        auto lastPublished = m_lastPublishedDiagnostics.tryGetValue(listKey);
        if (!lastPublished || *lastPublished != listValue.originalOutput)
        {
            PublishDiagnosticsParams args;
            args.uri = URI::fromLocalFilePath(listKey.getUnownedSlice()).uri;
            for (auto& d : listValue.messages)
                args.diagnostics.add(d);
            m_connection->sendCall(UnownedStringSlice("textDocument/publishDiagnostics"), &args);
            m_lastPublishedDiagnostics[listKey] = listValue.originalOutput;
        }
    }
}

void sendRefreshRequests(JSONRPCConnection* connection)
{
    connection->sendCall(
        UnownedStringSlice("workspace/semanticTokens/refresh"),
        JSONValue::makeInt(0));
    connection->sendCall(UnownedStringSlice("workspace/inlayHint/refresh"), JSONValue::makeInt(0));
}

void LanguageServer::updatePredefinedMacros(const JSONValue& macros)
{
    if (macros.isValid())
    {
        auto container = m_connection->getContainer();
        JSONToNativeConverter converter(container, &m_typeMap, m_connection->getSink());
        List<String> predefinedMacros;
        if (SLANG_SUCCEEDED(converter.convert(macros, &predefinedMacros)))
        {
            if (m_core.m_workspace->updatePredefinedMacros(predefinedMacros))
            {
                sendRefreshRequests(m_connection);
            }
        }
    }
}

void LanguageServer::updateSearchPaths(const JSONValue& value)
{
    if (value.isValid())
    {
        auto container = m_connection->getContainer();
        JSONToNativeConverter converter(container, &m_typeMap, m_connection->getSink());
        List<String> searchPaths;
        if (SLANG_SUCCEEDED(converter.convert(value, &searchPaths)))
        {
            if (m_core.m_workspace->updateSearchPaths(searchPaths))
            {
                sendRefreshRequests(m_connection);
            }
        }
    }
}

void LanguageServer::updateSearchInWorkspace(const JSONValue& value)
{
    if (value.isValid())
    {
        auto container = m_connection->getContainer();
        JSONToNativeConverter converter(container, &m_typeMap, m_connection->getSink());
        bool searchPaths;
        if (SLANG_SUCCEEDED(converter.convert(value, &searchPaths)))
        {
            if (m_core.m_workspace->updateSearchInWorkspace(searchPaths))
            {
                sendRefreshRequests(m_connection);
            }
        }
    }
}

void LanguageServer::updateCommitCharacters(const JSONValue& jsonValue)
{
    if (jsonValue.isValid())
    {
        auto container = m_connection->getContainer();
        JSONToNativeConverter converter(container, &m_typeMap, m_connection->getSink());
        String value;
        if (SLANG_SUCCEEDED(converter.convert(jsonValue, &value)))
        {
            if (value == "on")
            {
                m_core.m_commitCharacterBehavior = CommitCharacterBehavior::All;
            }
            else if (value == "off")
            {
                m_core.m_commitCharacterBehavior = CommitCharacterBehavior::Disabled;
            }
            else
            {
                m_core.m_commitCharacterBehavior = CommitCharacterBehavior::MembersOnly;
            }
        }
    }
}

void LanguageServer::updateFormattingOptions(
    const JSONValue& clangFormatLoc,
    const JSONValue& clangFormatStyle,
    const JSONValue& clangFormatFallbackStyle,
    const JSONValue& allowLineBreakOnType,
    const JSONValue& allowLineBreakInRange)
{
    auto container = m_connection->getContainer();
    JSONToNativeConverter converter(container, &m_typeMap, m_connection->getSink());
    if (clangFormatLoc.isValid())
        converter.convert(clangFormatLoc, &m_core.m_formatOptions.clangFormatLocation);
    if (clangFormatStyle.isValid())
        converter.convert(clangFormatStyle, &m_core.m_formatOptions.style);
    if (clangFormatFallbackStyle.isValid())
        converter.convert(clangFormatFallbackStyle, &m_core.m_formatOptions.fallbackStyle);
    if (allowLineBreakOnType.isValid())
        converter.convert(
            allowLineBreakOnType,
            &m_core.m_formatOptions.allowLineBreakInOnTypeFormatting);
    if (allowLineBreakInRange.isValid())
        converter.convert(
            allowLineBreakInRange,
            &m_core.m_formatOptions.allowLineBreakInRangeFormatting);
    if (m_core.m_formatOptions.style.getLength() == 0)
        m_core.m_formatOptions.style = Slang::FormatOptions().style;
}

void LanguageServer::updateInlayHintOptions(
    const JSONValue& deducedTypes,
    const JSONValue& parameterNames)
{
    auto container = m_connection->getContainer();
    JSONToNativeConverter converter(container, &m_typeMap, m_connection->getSink());
    bool showDeducedType = false;
    bool showParameterNames = false;
    converter.convert(deducedTypes, &showDeducedType);
    converter.convert(parameterNames, &showParameterNames);
    if (showDeducedType != m_core.m_inlayHintOptions.showDeducedType ||
        showParameterNames != m_core.m_inlayHintOptions.showParameterNames)
    {
        m_connection->sendCall(
            UnownedStringSlice("workspace/inlayHint/refresh"),
            JSONValue::makeInt(0));
    }
    m_core.m_inlayHintOptions.showDeducedType = showDeducedType;
    m_core.m_inlayHintOptions.showParameterNames = showParameterNames;
}

void LanguageServer::updateTraceOptions(const JSONValue& value)
{
    if (value.isValid())
    {
        auto container = m_connection->getContainer();
        JSONToNativeConverter converter(container, &m_typeMap, m_connection->getSink());
        String str;
        if (SLANG_SUCCEEDED(converter.convert(value, &str)))
        {
            if (str == "messages")
                m_traceOptions = TraceOptions::Messages;
            else if (str == "verbose")
                m_traceOptions = TraceOptions::Verbose;
            else
                m_traceOptions = TraceOptions::Off;
        }
    }
}

void LanguageServer::sendConfigRequest()
{
    ConfigurationParams args;
    ConfigurationItem item;
    item.section = "slang.predefinedMacros";
    args.items.add(item);
    item.section = "slang.additionalSearchPaths";
    args.items.add(item);
    item.section = "slang.searchInAllWorkspaceDirectories";
    args.items.add(item);
    item.section = "slang.enableCommitCharactersInAutoCompletion";
    args.items.add(item);
    item.section = "slang.format.clangFormatLocation";
    args.items.add(item);
    item.section = "slang.format.clangFormatStyle";
    args.items.add(item);
    item.section = "slang.format.clangFormatFallbackStyle";
    args.items.add(item);
    item.section = "slang.format.allowLineBreakChangesInOnTypeFormatting";
    args.items.add(item);
    item.section = "slang.format.allowLineBreakChangesInRangeFormatting";
    args.items.add(item);
    item.section = "slang.inlayHints.deducedTypes";
    args.items.add(item);
    item.section = "slang.inlayHints.parameterNames";
    args.items.add(item);
    item.section = "slangLanguageServer.trace.server";
    args.items.add(item);
    m_connection->sendCall(
        ConfigurationParams::methodName,
        &args,
        JSONValue::makeInt(kConfigResponseId));
}

void LanguageServer::registerCapability(const char* methodName)
{
    RegistrationParams args;
    Registration reg;
    reg.method = methodName;
    reg.id = reg.method;
    args.registrations.add(reg);
    m_connection->sendCall(
        UnownedStringSlice("client/registerCapability"),
        &args,
        JSONValue::makeInt(999));
}

void LanguageServer::logMessage(int type, String message)
{
    LanguageServerProtocol::LogMessageParams args;
    args.type = type;
    args.message = message;
    m_connection->sendCall(LanguageServerProtocol::LogMessageParams::methodName, &args);
}

FormatOptions LanguageServerCore::getFormatOptions(Workspace* workspace, FormatOptions inOptions)
{
    FormatOptions result = inOptions;
    if (workspace->rootDirectories.getCount())
    {
        result.clangFormatLocation = StringUtil::replaceAll(
            result.clangFormatLocation.getUnownedSlice(),
            toSlice("${workspaceFolder}"),
            workspace->rootDirectories.getFirst().getUnownedSlice());
    }
    return result;
}

LanguageServerResult<LanguageServerProtocol::Hover> LanguageServerCore::tryGetMacroHoverInfo(
    WorkspaceVersion* version,
    DocumentVersion* doc,
    Index line,
    Index col)
{
    Index startOffset = 0;
    auto identifier = doc->peekIdentifier(line, col, startOffset);
    if (identifier.getLength() == 0)
        return SLANG_FAIL;
    auto def = version->tryGetMacroDefinition(identifier);
    if (!def)
        return SLANG_FAIL;
    LanguageServerProtocol::Hover hover;
    doc->offsetToLineCol(startOffset, line, col);
    Index outLine, outCol;
    doc->oneBasedUTF8LocToZeroBasedUTF16Loc(line, col, outLine, outCol);
    hover.range.start.line = (int)outLine;
    hover.range.start.character = (int)outCol;
    hover.range.end.line = (int)outLine;
    hover.range.end.character = (int)(outCol + identifier.getLength());
    StringBuilder sb;
    sb << "```\n#define " << identifier;
    if (def->params.getCount())
    {
        sb << "(";
        bool isFirst = true;
        for (auto param : def->params)
        {
            if (!isFirst)
                sb << ", ";
            if (param.isVariadic)
                sb << "...";
            else if (param.name)
                sb << param.name->text;
            isFirst = false;
        }
        sb << ")";
    }
    for (auto& token : def->tokenList)
    {
        sb << " ";
        sb << token.getContent();
    }
    sb << "\n```\n\n";
    auto humaneLoc =
        version->linkage->getSourceManager()->getHumaneLoc(def->loc, SourceLocType::Actual);
    appendDefinitionLocation(sb, m_workspace, humaneLoc);
    hover.contents.kind = "markdown";
    hover.contents.value = sb.produceString();
    return hover;
}

LanguageServerResult<List<Location>> LanguageServerCore::tryGotoMacroDefinition(
    WorkspaceVersion* version,
    DocumentVersion* doc,
    Index line,
    Index col)
{
    Index startOffset = 0;
    auto identifier = doc->peekIdentifier(line, col, startOffset);
    if (identifier.getLength() == 0)
        return SLANG_FAIL;
    auto def = version->tryGetMacroDefinition(identifier);
    if (!def)
        return SLANG_FAIL;
    auto humaneLoc =
        version->linkage->getSourceManager()->getHumaneLoc(def->loc, SourceLocType::Actual);
    List<LanguageServerProtocol::Location> results;
    results.setCount(1);
    auto& result = results[0];
    result.uri = URI::fromLocalFilePath(humaneLoc.pathInfo.foundPath.getUnownedSlice()).uri;
    Index outLine, outCol;
    doc->oneBasedUTF8LocToZeroBasedUTF16Loc(humaneLoc.line, humaneLoc.column, outLine, outCol);
    result.range.start.line = (int)outLine;
    result.range.start.character = (int)outCol;
    result.range.end.line = (int)outLine;
    result.range.end.character = (int)(outCol + identifier.getLength());
    return results;
}

LanguageServerResult<List<Location>> LanguageServerCore::tryGotoFileInclude(
    WorkspaceVersion* version,
    DocumentVersion* doc,
    Index line)
{
    auto lineContent = doc->getLine(line).trim();
    if (!lineContent.startsWith("#") || lineContent.indexOf(UnownedStringSlice("include")) == -1)
        return SLANG_FAIL;
    for (auto& include : version->linkage->contentAssistInfo.preprocessorInfo.fileIncludes)
    {
        auto includeLoc =
            version->linkage->getSourceManager()->getHumaneLoc(include.loc, SourceLocType::Actual);
        if (includeLoc.line == line && includeLoc.pathInfo.foundPath == doc->getPath())
        {
            List<LanguageServerProtocol::Location> results;
            results.setCount(1);
            auto& result = results[0];
            result.uri = URI::fromLocalFilePath(include.path.getUnownedSlice()).uri;
            result.range.start.line = 0;
            result.range.start.character = 0;
            result.range.end.line = 0;
            result.range.end.character = 0;
            return results;
        }
    }
    return SLANG_FAIL;
}

SlangResult LanguageServer::queueJSONCall(JSONRPCCall call)
{
    Command cmd;
    cmd.id = PersistentJSONValue(call.id, m_connection->getContainer());
    cmd.method = call.method;
    if (call.method == DidOpenTextDocumentParams::methodName)
    {
        DidOpenTextDocumentParams args;
        SLANG_RETURN_ON_FAIL(m_connection->toNativeArgsOrSendError(call.params, &args, call.id));
        cmd.openDocArgs = args;
    }
    else if (call.method == DidCloseTextDocumentParams::methodName)
    {
        DidCloseTextDocumentParams args;
        SLANG_RETURN_ON_FAIL(m_connection->toNativeArgsOrSendError(call.params, &args, call.id));
        cmd.closeDocArgs = args;
    }
    else if (call.method == DidChangeTextDocumentParams::methodName)
    {
        DidChangeTextDocumentParams args;
        SLANG_RETURN_ON_FAIL(m_connection->toNativeArgsOrSendError(call.params, &args, call.id));
        cmd.changeDocArgs = args;
    }
    else if (call.method == HoverParams::methodName)
    {
        HoverParams args;
        SLANG_RETURN_ON_FAIL(m_connection->toNativeArgsOrSendError(call.params, &args, call.id));
        cmd.hoverArgs = args;
    }
    else if (call.method == DefinitionParams::methodName)
    {
        DefinitionParams args;
        SLANG_RETURN_ON_FAIL(m_connection->toNativeArgsOrSendError(call.params, &args, call.id));
        cmd.definitionArgs = args;
    }
    else if (call.method == CompletionParams::methodName)
    {
        CompletionParams args;
        SLANG_RETURN_ON_FAIL(m_connection->toNativeArgsOrSendError(call.params, &args, call.id));
        cmd.completionArgs = args;
    }
    else if (call.method == SemanticTokensParams::methodName)
    {
        SemanticTokensParams args;
        SLANG_RETURN_ON_FAIL(m_connection->checkArrayObjectWrap(
            call.params,
            GetRttiInfo<SemanticTokensParams>::get(),
            &args,
            call.id));
        cmd.semanticTokenArgs = args;
    }
    else if (call.method == SignatureHelpParams::methodName)
    {
        SignatureHelpParams args;
        SLANG_RETURN_ON_FAIL(m_connection->toNativeArgsOrSendError(call.params, &args, call.id));
        cmd.signatureHelpArgs = args;
    }
    else if (call.method == "completionItem/resolve")
    {
        Slang::LanguageServerProtocol::CompletionItem args;
        SLANG_RETURN_ON_FAIL(m_connection->toNativeArgsOrSendError(call.params, &args, call.id));
        cmd.completionResolveArgs = args;
        Slang::LanguageServerProtocol::TextEditCompletionItem editArgs;
        SLANG_RETURN_ON_FAIL(
            m_connection->toNativeArgsOrSendError(call.params, &editArgs, call.id));
        cmd.textEditCompletionResolveArgs = editArgs;
    }
    else if (call.method == DocumentSymbolParams::methodName)
    {
        DocumentSymbolParams args;
        SLANG_RETURN_ON_FAIL(m_connection->checkArrayObjectWrap(
            call.params,
            GetRttiInfo<DocumentSymbolParams>::get(),
            &args,
            call.id));
        cmd.documentSymbolArgs = args;
    }
    else if (call.method == DocumentFormattingParams::methodName)
    {
        DocumentFormattingParams args;
        SLANG_RETURN_ON_FAIL(m_connection->toNativeArgsOrSendError(call.params, &args, call.id));
        cmd.formattingArgs = args;
    }
    else if (call.method == DocumentRangeFormattingParams::methodName)
    {
        DocumentRangeFormattingParams args;
        SLANG_RETURN_ON_FAIL(m_connection->toNativeArgsOrSendError(call.params, &args, call.id));
        cmd.rangeFormattingArgs = args;
    }
    else if (call.method == DocumentOnTypeFormattingParams::methodName)
    {
        DocumentOnTypeFormattingParams args;
        SLANG_RETURN_ON_FAIL(m_connection->toNativeArgsOrSendError(call.params, &args, call.id));
        cmd.onTypeFormattingArgs = args;
    }
    else if (call.method == DidChangeConfigurationParams::methodName)
    {
        DidChangeConfigurationParams args;
        SLANG_RETURN_ON_FAIL(m_connection->toNativeArgsOrSendError(call.params, &args, call.id));
        // We need to process it now instead of sending to queue.
        // This is because there is reference to JSONValue that is only available here.
        return didChangeConfiguration(args);
    }
    else if (call.method == InlayHintParams::methodName)
    {
        InlayHintParams args;
        SLANG_RETURN_ON_FAIL(m_connection->toNativeArgsOrSendError(call.params, &args, call.id));
        cmd.inlayHintArgs = args;
    }
    else if (call.method == "$/cancelRequest")
    {
        CancelParams args;
        SLANG_RETURN_ON_FAIL(m_connection->toNativeArgsOrSendError(call.params, &args, call.id));
        cmd.cancelArgs = args;
    }
    commands.add(_Move(cmd));
    return SLANG_OK;
}

SlangResult LanguageServer::runCommand(Command& call)
{
    try
    {
        // Do different things
        if (call.method == DidOpenTextDocumentParams::methodName)
        {
            return didOpenTextDocument(call.openDocArgs.get());
        }
        else if (call.method == DidCloseTextDocumentParams::methodName)
        {
            return didCloseTextDocument(call.closeDocArgs.get());
        }
        else if (call.method == DidChangeTextDocumentParams::methodName)
        {
            return didChangeTextDocument(call.changeDocArgs.get());
        }
        else if (call.method.startsWith("$/"))
        {
            // Ignore.
            return SLANG_OK;
        }
    }
    catch (...)
    {
        return SLANG_FAIL;
    }

    try
    {
        if (call.method == HoverParams::methodName)
        {
            return hover(call.hoverArgs.get(), call.id);
        }
        else if (call.method == DefinitionParams::methodName)
        {
            return gotoDefinition(call.definitionArgs.get(), call.id);
        }
        else if (call.method == CompletionParams::methodName)
        {
            return completion(call.completionArgs.get(), call.id);
        }
        else if (call.method == SemanticTokensParams::methodName)
        {
            return semanticTokens(call.semanticTokenArgs.get(), call.id);
        }
        else if (call.method == SignatureHelpParams::methodName)
        {
            return signatureHelp(call.signatureHelpArgs.get(), call.id);
        }
        else if (call.method == "completionItem/resolve")
        {
            return completionResolve(
                call.completionResolveArgs.get(),
                call.textEditCompletionResolveArgs.get(),
                call.id);
        }
        else if (call.method == DocumentSymbolParams::methodName)
        {
            return documentSymbol(call.documentSymbolArgs.get(), call.id);
        }
        else if (call.method == DidChangeConfigurationParams::methodName)
        {
            return didChangeConfiguration(call.changeConfigArgs.get());
        }
        else if (call.method == InlayHintParams::methodName)
        {
            return inlayHint(call.inlayHintArgs.get(), call.id);
        }
        else if (call.method == DocumentOnTypeFormattingParams::methodName)
        {
            return onTypeFormatting(call.onTypeFormattingArgs.get(), call.id);
        }
        else if (call.method == DocumentRangeFormattingParams::methodName)
        {
            return rangeFormatting(call.rangeFormattingArgs.get(), call.id);
        }
        else if (call.method == DocumentFormattingParams::methodName)
        {
            return formatting(call.formattingArgs.get(), call.id);
        }
        else if (call.method == "NotificationReceived")
        {
            return SLANG_OK;
        }
    }
    catch (...)
    {
        // If we encountered an internal compiler error, don't crash the language server.
        // Instead we just return a null response.
        return m_connection->sendResult(NullResponse::get(), call.id);
    }

    return m_connection->sendError(JSONRPC::ErrorCode::MethodNotFound, call.id);
}

void LanguageServer::processCommands()
{
    HashSet<int64_t> canceledIDs;
    for (auto& cmd : commands)
    {
        if (cmd.method == "$/cancelRequest")
        {
            auto id = cmd.cancelArgs.get().id;
            if (id > 0)
            {
                canceledIDs.add(id);
            }
        }
    }
    const int kErrorRequestCanceled = -32800;
    for (auto& cmd : commands)
    {
        if (cmd.id.getKind() == JSONValue::Kind::Integer &&
            canceledIDs.contains(cmd.id.asInteger()))
        {
            m_connection->sendError((JSONRPC::ErrorCode)kErrorRequestCanceled, cmd.id);
        }
        else
        {
            runCommand(cmd);
        }
    }
}

SlangResult LanguageServer::didCloseTextDocument(const DidCloseTextDocumentParams& args)
{
    resetDiagnosticUpdateTime();
    return m_core.didCloseTextDocument(args);
}

SlangResult LanguageServerCore::didCloseTextDocument(const DidCloseTextDocumentParams& args)
{
    String canonicalPath = uriToCanonicalPath(args.textDocument.uri);
    m_workspace->closeDoc(canonicalPath);
    return SLANG_OK;
}

SlangResult LanguageServer::didChangeTextDocument(const DidChangeTextDocumentParams& args)
{
    resetDiagnosticUpdateTime();
    return m_core.didChangeTextDocument(args);
}

SlangResult LanguageServerCore::didChangeTextDocument(const DidChangeTextDocumentParams& args)
{
    String canonicalPath = uriToCanonicalPath(args.textDocument.uri);
    for (auto change : args.contentChanges)
        m_workspace->changeDoc(canonicalPath, change.range, change.text);
    return SLANG_OK;
}

SlangResult LanguageServer::didChangeConfiguration(
    const LanguageServerProtocol::DidChangeConfigurationParams& args)
{
    if (args.settings.isValid())
    {
        updateConfigFromJSON(args.settings);
    }
    else
    {
        sendConfigRequest();
    }
    return SLANG_OK;
}

void LanguageServer::update()
{
    if (!m_core.m_workspace)
        return;
    publishDiagnostics();
}

void LanguageServer::updateConfigFromJSON(const JSONValue& jsonVal)
{
    if (!jsonVal.isObjectLike())
        return;
    auto obj = m_connection->getContainer()->getObject(jsonVal);
    if (obj.getCount() == 1 &&
        (m_connection->getContainer()->getStringFromKey(obj[0].key) == "settings" ||
         m_connection->getContainer()->getStringFromKey(obj[0].key) == "RootElement"))
    {
        updateConfigFromJSON(obj[0].value);
        return;
    }
    for (auto kv : obj)
    {
        auto key = m_connection->getContainer()->getStringFromKey(kv.key);
        if (key == "slang.predefinedMacros")
        {
            updatePredefinedMacros(kv.value);
        }
        else if (key == "slang.additionalSearchPaths")
        {
            updateSearchPaths(kv.value);
        }
        else if (key == "slang.enableCommitCharactersInAutoCompletion")
        {
            updateCommitCharacters(kv.value);
        }
        else if (key == "slang.format.clangFormatLocation")
        {
            updateFormattingOptions(kv.value, JSONValue(), JSONValue(), JSONValue(), JSONValue());
        }
        else if (key == "slang.format.clangFormatStyle")
        {
            updateFormattingOptions(JSONValue(), kv.value, JSONValue(), JSONValue(), JSONValue());
        }
        else if (key == "slang.format.clangFormatFallbackStyle")
        {
            updateFormattingOptions(JSONValue(), JSONValue(), kv.value, JSONValue(), JSONValue());
        }
        else if (key == "slang.format.allowLineBreakChangesInOnTypeFormatting")
        {
            updateFormattingOptions(JSONValue(), JSONValue(), JSONValue(), kv.value, JSONValue());
        }
        else if (key == "slang.format.allowLineBreakChangesInRangeFormatting")
        {
            updateFormattingOptions(JSONValue(), JSONValue(), JSONValue(), JSONValue(), kv.value);
        }
        else if (key == "slang.inlayHints.deducedTypes")
        {
            updateInlayHintOptions(kv.value, JSONValue());
        }
        else if (key == "slang.inlayHints.parameterNames")
        {
            updateInlayHintOptions(JSONValue(), kv.value);
        }
    }
}

SlangResult LanguageServer::execute()
{
    m_connection = new JSONRPCConnection();
    m_connection->initWithStdStreams();

    while (m_connection->isActive() && !m_quit)
    {
        // Consume all messages first.
        commands.clear();
        while (true)
        {
            m_connection->tryReadMessage();
            if (!m_connection->hasMessage())
                break;
            parseNextMessage();
        }

        auto workStart = platform::PerformanceCounter::now();

        processCommands();

        // Report diagnostics if it hasn't been updated for a while.
        update();

        auto workTime = platform::PerformanceCounter::getElapsedTimeInSeconds(workStart);

        if (commands.getCount() > 0 && m_initialized && m_traceOptions != TraceOptions::Off)
        {
            StringBuilder msgBuilder;
            msgBuilder << "Server processed " << commands.getCount() << " commands, executed in "
                       << String(int(workTime * 1000)) << "ms";
            logMessage(3, msgBuilder.produceString());
        }

        m_connection->getUnderlyingConnection()->waitForResult(1000);
    }

    return SLANG_OK;
}

SLANG_API void LanguageServerStartupOptions::parse(int argc, const char* const* argv)
{
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-vs") == 0)
            isVisualStudio = true;
    }
}

SLANG_API SlangResult runLanguageServer(Slang::LanguageServerStartupOptions options)
{
    Slang::LanguageServer server(options);
    SLANG_RETURN_ON_FAIL(server.execute());
    return SLANG_OK;
}

} // namespace Slang
