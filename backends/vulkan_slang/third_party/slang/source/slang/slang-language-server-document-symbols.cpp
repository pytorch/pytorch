#include "slang-language-server-document-symbols.h"

namespace Slang
{
struct GetDocumentSymbolContext
{
    HashSet<Decl*> processedDecls;
    DocumentVersion* doc;
    Linkage* linkage;
    UnownedStringSlice fileName;
};

static LanguageServerProtocol::SymbolKind _getSymbolKind(Decl* decl)
{
    if (as<StructDecl>(decl))
    {
        return LanguageServerProtocol::kSymbolKindStruct;
    }
    if (as<ClassDecl>(decl))
    {
        return LanguageServerProtocol::kSymbolKindClass;
    }
    if (as<InterfaceDecl>(decl))
    {
        return LanguageServerProtocol::kSymbolKindInterface;
    }
    if (as<FuncDecl>(decl))
    {
        return as<AggTypeDecl>(decl->parentDecl) ? LanguageServerProtocol::kSymbolKindMethod
                                                 : LanguageServerProtocol::kSymbolKindFunction;
    }
    if (as<PropertyDecl>(decl))
    {
        return LanguageServerProtocol::kSymbolKindProperty;
    }
    if (as<ConstructorDecl>(decl))
    {
        return LanguageServerProtocol::kSymbolKindConstructor;
    }
    if (as<AssocTypeDecl>(decl))
    {
        return LanguageServerProtocol::kSymbolKindTypeParameter;
    }
    if (as<VarDeclBase>(decl))
    {
        if (decl->findModifier<ConstModifier>())
            return LanguageServerProtocol::kSymbolKindConstant;
        return as<AggTypeDecl>(decl->parentDecl) ? LanguageServerProtocol::kSymbolKindField
                                                 : LanguageServerProtocol::kSymbolKindVariable;
    }
    if (as<TypeDefDecl>(decl))
    {
        return LanguageServerProtocol::kSymbolKindClass;
    }
    if (as<GenericTypeParamDecl>(decl))
    {
        return LanguageServerProtocol::kSymbolKindTypeParameter;
    }
    if (as<GenericValueParamDecl>(decl))
    {
        return LanguageServerProtocol::kSymbolKindConstant;
    }
    if (as<EnumDecl>(decl))
    {
        return LanguageServerProtocol::kSymbolKindEnum;
    }
    if (as<EnumCaseDecl>(decl))
    {
        return LanguageServerProtocol::kSymbolKindEnumMember;
    }
    if (as<NamespaceDecl>(decl))
    {
        return LanguageServerProtocol::kSymbolKindNamespace;
    }
    if (as<ExtensionDecl>(decl))
    {
        return LanguageServerProtocol::kSymbolKindClass;
    }
    if (as<SubscriptDecl>(decl))
    {
        return LanguageServerProtocol::kSymbolKindOperator;
    }
    return -1;
}
static SourceLoc _findClosingSourceLoc(Decl* decl)
{
    if (auto func = as<FunctionDeclBase>(decl))
    {
        if (auto block = as<BlockStmt>(func->body))
        {
            return block->closingSourceLoc;
        }
        else if (auto unparsedStmt = as<UnparsedStmt>(func->body))
        {
            if (unparsedStmt->tokens.getCount())
                return unparsedStmt->tokens.getLast().getLoc();
        }
        else if (func->body)
        {
            return func->body->loc;
        }
    }
    if (auto container = as<ContainerDecl>(decl))
    {
        return container->closingSourceLoc;
    }
    return SourceLoc();
}

static NameLoc _getDeclRefExprNameLoc(Expr* expr)
{
    if (auto varExpr = as<VarExpr>(expr))
    {
        return NameLoc(varExpr->name, varExpr->loc);
    }
    else if (auto appBase = as<AppExprBase>(expr))
    {
        return _getDeclRefExprNameLoc(appBase->functionExpr);
    }
    return NameLoc();
}
static NameLoc _getDeclNameLoc(Decl* decl)
{
    if (auto extDecl = as<ExtensionDecl>(decl))
    {
        return _getDeclRefExprNameLoc(extDecl->targetType.exp);
    }
    return decl->nameAndLoc;
}

static void _getDocumentSymbolsImpl(
    GetDocumentSymbolContext& context,
    Decl* parent,
    List<LanguageServerProtocol::DocumentSymbol>& childSymbols)
{
    auto containerDecl = as<ContainerDecl>(parent);
    if (!containerDecl)
        return;
    if (!context.processedDecls.add(parent))
        return;
    auto srcManager = context.linkage->getSourceManager();
    for (auto child : containerDecl->members)
    {
        if (auto genericDecl = as<GenericDecl>(child))
        {
            child = genericDecl->inner;
        }
        LanguageServerProtocol::SymbolKind kind = _getSymbolKind(child);
        if (kind <= 0)
            continue;
        NameLoc nameLoc = _getDeclNameLoc(child);
        if (!nameLoc.name)
            continue;
        if (nameLoc.name->text.getLength() == 0)
            continue;
        if (!nameLoc.loc.isValid())
            continue;
        if (child->hasModifier<SynthesizedModifier>() ||
            child->hasModifier<ToBeSynthesizedModifier>())
            continue;
        auto humaneLoc = srcManager->getHumaneLoc(nameLoc.loc, SourceLocType::Actual);
        if (humaneLoc.line == 0)
            continue;
        if (context.fileName.endsWithCaseInsensitive(
                Path::getFileName(humaneLoc.pathInfo.foundPath).getUnownedSlice()))
        {
            LanguageServerProtocol::DocumentSymbol sym;
            sym.name = nameLoc.name->text;
            sym.kind = kind;
            Index line, col;
            context.doc
                ->oneBasedUTF8LocToZeroBasedUTF16Loc(humaneLoc.line, humaneLoc.column, line, col);
            sym.selectionRange.start.line = (int)line;
            sym.selectionRange.start.character = (int)col;
            sym.selectionRange.end.line = (int)line;
            sym.selectionRange.end.character =
                (int)(col +
                      (int)UTF8Util::calcUTF16CharCount(nameLoc.name->text.getUnownedSlice()));
            sym.range.start.line = (int)line;
            sym.range.start.character = 0;
            sym.range.end.line = (int)line;
            sym.range.end.character = sym.selectionRange.end.character;
            // Now try to find the end of the decl.
            auto closingLoc = _findClosingSourceLoc(child);
            if (closingLoc.isValid())
            {
                auto closingHumaneLoc = srcManager->getHumaneLoc(closingLoc, SourceLocType::Actual);
                context.doc->oneBasedUTF8LocToZeroBasedUTF16Loc(
                    closingHumaneLoc.line,
                    closingHumaneLoc.column,
                    line,
                    col);
                // Due to macro replacements, the closing loc may be before the start loc,
                // and we need to make sure never returning such invalid ranges to the editor
                // client.
                if (closingHumaneLoc.line > sym.range.start.line ||
                    closingHumaneLoc.line == sym.range.start.line &&
                        closingHumaneLoc.column >= sym.range.start.character)
                {
                    sym.range.end.line = (int)line;
                    sym.range.end.character = (int)col;
                }
                if (sym.selectionRange.end.line == sym.range.end.line ||
                    sym.selectionRange.end.character >= sym.range.end.character)
                {
                    sym.selectionRange.end = sym.range.end;
                }
            }
            if (const auto childContainerDecl = as<ContainerDecl>(child))
            {
                // Recurse
                bool shouldRecurse = true;
                if (as<CallableDecl>(child))
                    shouldRecurse = false;
                if (as<PropertyDecl>(child))
                    shouldRecurse = false;
                if (shouldRecurse)
                {
                    _getDocumentSymbolsImpl(context, child, sym.children);
                }
            }
            childSymbols.add(_Move(sym));
        }
    }
}

List<LanguageServerProtocol::DocumentSymbol> getDocumentSymbols(
    Linkage* linkage,
    Module* module,
    UnownedStringSlice fileName,
    DocumentVersion* doc)
{
    GetDocumentSymbolContext context;
    context.fileName = fileName;
    context.doc = doc;
    context.linkage = linkage;
    List<LanguageServerProtocol::DocumentSymbol> result;
    _getDocumentSymbolsImpl(context, module->getModuleDecl(), result);
    return result;
}

} // namespace Slang
