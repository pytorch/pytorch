#include "slang-language-server-semantic-tokens.h"

#include "../core/slang-char-util.h"
#include "slang-ast-iterator.h"
#include "slang-ast-support-types.h"
#include "slang-visitor.h"

#include <algorithm>

namespace Slang
{

const char* kSemanticTokenTypes[] = {
    "type",
    "enumMember",
    "variable",
    "parameter",
    "function",
    "property",
    "namespace",
    "keyword",
    "macro",
    "string"};

static const int kInitTokenLegnth = 6;

static_assert(
    SLANG_COUNT_OF(kSemanticTokenTypes) == (int)SemanticTokenType::NormalText,
    "kSemanticTokenTypes must match SemanticTokenType");

SemanticToken _createSemanticToken(SourceManager* manager, SourceLoc loc, Name* name)
{
    SemanticToken token;
    auto humaneLoc = manager->getHumaneLoc(loc, SourceLocType::Actual);
    token.line = (int)(humaneLoc.line);
    token.col = (int)(humaneLoc.column);
    token.length = name ? (int)(name->text.getLength()) : 0;
    token.type = SemanticTokenType::NormalText;
    return token;
}

// We don't want to semantic highlight a synthetic name, like $init, () etc,
// because they aren't actually appearing in the source code.
bool isHighlightableName(Name* name)
{
    if (!name)
        return false;
    for (auto ch : name->text)
    {
        if (!CharUtil::isAlphaOrDigit(ch) && ch != '_')
            return false;
    }
    return true;
}

List<SemanticToken> getSemanticTokens(
    Linkage* linkage,
    Module* module,
    UnownedStringSlice fileName,
    DocumentVersion* doc)
{
    auto manager = linkage->getSourceManager();

    auto cbufferName = linkage->getNamePool()->getName(toSlice("ConstantBuffer"));

    List<SemanticToken> result;
    auto maybeInsertToken = [&](const SemanticToken& token)
    {
        if (token.line > 0 && token.col > 0 && token.length > 0 &&
            token.type != SemanticTokenType::NormalText)
            result.add(token);
    };
    auto handleDeclRef = [&](DeclRef<Decl> declRef, Expr* originalExpr, Name* name, SourceLoc loc)
    {
        if (!declRef)
            return;
        auto decl = declRef.getDecl();
        if (auto genDecl = as<GenericDecl>(decl))
            decl = genDecl->inner;
        if (!decl)
            return;
        if (!name)
            name = declRef.getDecl()->getName();
        if (!isHighlightableName(name))
            return;
        // Don't look at the expr if it is defined in a different file.
        if (!manager->getHumaneLoc(loc, SourceLocType::Actual)
                 .pathInfo.foundPath.getUnownedSlice()
                 .endsWithCaseInsensitive(fileName))
            return;
        if (decl->hasModifier<SynthesizedModifier>())
            return;
        SemanticToken token = _createSemanticToken(manager, loc, name);
        auto target = decl;
        if (as<AggTypeDecl>(target))
        {
            if (target->hasModifier<BuiltinTypeModifier>())
                return;
            token.type = SemanticTokenType::Type;
            if (name == cbufferName)
            {
                token.length = doc->getTokenLength(token.line, token.col);
            }
        }
        else if (as<ConstructorDecl>(target))
        {
            token.type = SemanticTokenType::Type;
            token.length = doc->getTokenLength(token.line, token.col);
        }
        else if (as<SimpleTypeDecl>(target))
        {
            token.type = SemanticTokenType::Type;
        }
        else if (as<PropertyDecl>(target))
        {
            token.type = SemanticTokenType::Property;
        }
        else if (as<ParamDecl>(target))
        {
            token.type = SemanticTokenType::Parameter;
        }
        else if (as<VarDecl>(target))
        {
            if (as<MemberExpr>(originalExpr) || as<StaticMemberExpr>(originalExpr))
            {
                return;
            }
            token.type = SemanticTokenType::Variable;
        }
        else if (as<FunctionDeclBase>(target))
        {
            token.type = SemanticTokenType::Function;
        }
        else if (as<EnumCaseDecl>(target))
        {
            token.type = SemanticTokenType::EnumMember;
        }
        else if (as<NamespaceDecl>(target))
        {
            token.type = SemanticTokenType::Namespace;
        }

        if (as<CallableDecl>(target))
        {
            if (target->hasModifier<ImplicitConversionModifier>())
                return;
        }
        maybeInsertToken(token);
    };
    iterateASTWithLanguageServerFilter(
        fileName,
        manager,
        module->getModuleDecl(),
        [&](SyntaxNode* node)
        {
            if (auto decl = as<Decl>(node))
            {
                for (auto modifier : decl->modifiers)
                {
                    if (as<SynthesizedModifier>(modifier))
                        return;
                    if (as<ImplicitParameterGroupElementTypeModifier>(modifier))
                        return;
                }
            }

            if (auto declRefExpr = as<DeclRefExpr>(node))
            {
                handleDeclRef(
                    declRefExpr->declRef,
                    declRefExpr->originalExpr,
                    declRefExpr->name,
                    declRefExpr->loc);
            }
            else if (auto overloadedExpr = as<OverloadedExpr>(node))
            {
                if (overloadedExpr->lookupResult2.items.getCount())
                {
                    handleDeclRef(
                        overloadedExpr->lookupResult2.items[0].declRef,
                        overloadedExpr->originalExpr,
                        overloadedExpr->name,
                        overloadedExpr->loc);
                }
            }
            else if (auto accessorDecl = as<AccessorDecl>(node))
            {
                SemanticToken token =
                    _createSemanticToken(manager, accessorDecl->loc, accessorDecl->getName());
                token.type = SemanticTokenType::Keyword;
                maybeInsertToken(token);
            }
            else if (auto typeDecl = as<SimpleTypeDecl>(node))
            {
                if (typeDecl->getName())
                {
                    SemanticToken token =
                        _createSemanticToken(manager, typeDecl->getNameLoc(), typeDecl->getName());
                    token.type = SemanticTokenType::Type;
                    maybeInsertToken(token);
                }
            }
            else if (auto aggTypeDeclBase = as<AggTypeDeclBase>(node))
            {
                if (!aggTypeDeclBase->getName())
                    return;
                SemanticToken token = _createSemanticToken(
                    manager,
                    aggTypeDeclBase->getNameLoc(),
                    aggTypeDeclBase->getName());
                token.type = SemanticTokenType::Type;
                maybeInsertToken(token);
            }
            else if (auto enumCase = as<EnumCaseDecl>(node))
            {
                if (enumCase->getName())
                {
                    SemanticToken token =
                        _createSemanticToken(manager, enumCase->getNameLoc(), enumCase->getName());
                    token.type = SemanticTokenType::EnumMember;
                    maybeInsertToken(token);
                }
            }
            else if (auto propertyDecl = as<PropertyDecl>(node))
            {
                if (propertyDecl->getName())
                {
                    SemanticToken token = _createSemanticToken(
                        manager,
                        propertyDecl->getNameLoc(),
                        propertyDecl->getName());
                    token.type = SemanticTokenType::Property;
                    maybeInsertToken(token);
                }
            }
            else if (auto funcDecl = as<FuncDecl>(node))
            {
                if (isHighlightableName(funcDecl->getName()))
                {
                    SemanticToken token =
                        _createSemanticToken(manager, funcDecl->getNameLoc(), funcDecl->getName());
                    token.type = SemanticTokenType::Function;
                    maybeInsertToken(token);
                }
            }
            else if (auto ctorDecl = as<ConstructorDecl>(node))
            {
                if (ctorDecl->getName())
                {
                    SemanticToken token =
                        _createSemanticToken(manager, ctorDecl->getNameLoc(), ctorDecl->getName());
                    token.type = SemanticTokenType::Function;
                    token.length = kInitTokenLegnth;
                    maybeInsertToken(token);
                }
            }
            else if (auto paramDecl = as<ParamDecl>(node))
            {
                if (paramDecl->getName())
                {
                    SemanticToken token = _createSemanticToken(
                        manager,
                        paramDecl->getNameLoc(),
                        paramDecl->getName());
                    token.type = SemanticTokenType::Parameter;
                    maybeInsertToken(token);
                }
            }
            else if (auto varDecl = as<VarDeclBase>(node))
            {
                if (varDecl->getName())
                {
                    SemanticToken token =
                        _createSemanticToken(manager, varDecl->getNameLoc(), varDecl->getName());
                    token.type = SemanticTokenType::Variable;
                    maybeInsertToken(token);
                }
            }
            else if (auto attr = as<Attribute>(node))
            {
                if (attr->getKeywordName())
                {
                    SemanticToken token =
                        _createSemanticToken(manager, attr->originalIdentifierToken.loc, nullptr);
                    token.length = (int)attr->originalIdentifierToken.getContentLength();
                    token.type = SemanticTokenType::Type;
                    maybeInsertToken(token);

                    // Insert capability names as enum cases.
                    if (as<RequireCapabilityAttribute>(attr))
                    {
                        for (auto arg : attr->args)
                        {
                            if (auto varExpr = as<VarExpr>(arg))
                            {
                                if (varExpr->name)
                                {
                                    SemanticToken capToken =
                                        _createSemanticToken(manager, varExpr->loc, nullptr);
                                    capToken.length = (int)varExpr->name->text.getLength();
                                    capToken.type = SemanticTokenType::EnumMember;
                                    maybeInsertToken(capToken);
                                }
                            }
                        }
                    }
                }
            }
            else if (auto targetCase = as<TargetCaseStmt>(node))
            {
                SemanticToken token = _createSemanticToken(
                    manager,
                    targetCase->capabilityToken.loc,
                    targetCase->capabilityToken.getName());
                token.type = SemanticTokenType::EnumMember;
                maybeInsertToken(token);
            }
            else if (auto spirvAsmExpr = as<SPIRVAsmExpr>(node))
            {
                // Highlight opcodes and enums.
                for (auto inst : spirvAsmExpr->insts)
                {
                    // OpCode
                    {
                        SemanticToken token = _createSemanticToken(
                            manager,
                            inst.opcode.token.loc,
                            inst.opcode.token.getName());
                        token.type = SemanticTokenType::Function;
                        maybeInsertToken(token);
                    }
                    // Other operands
                    for (auto operand : inst.operands)
                    {
                        SemanticTokenType operandTokenType = SemanticTokenType::NormalText;
                        switch (operand.flavor)
                        {
                        case SPIRVAsmOperand::Flavor::NamedValue:
                        case SPIRVAsmOperand::Flavor::BuiltinVar:
                            operandTokenType = SemanticTokenType::EnumMember;
                            break;
                        case SPIRVAsmOperand::Flavor::ResultMarker:
                        case SPIRVAsmOperand::Flavor::TruncateMarker:
                        case SPIRVAsmOperand::Flavor::GLSL450Set:
                            operandTokenType = SemanticTokenType::Macro;
                            break;
                        case SPIRVAsmOperand::Flavor::Id:
                            operandTokenType = SemanticTokenType::String;
                            break;
                        }
                        if (operandTokenType != SemanticTokenType::NormalText)
                        {
                            SemanticToken token = _createSemanticToken(
                                manager,
                                operand.token.loc,
                                operand.token.getName());
                            token.type = operandTokenType;
                            maybeInsertToken(token);
                        }
                    }
                }
            }
        });
    // Insert macro tokens.
    auto& preprocessorInfo = linkage->contentAssistInfo.preprocessorInfo;
    for (auto& invocation : preprocessorInfo.macroInvocations)
    {
        if (!invocation.name)
            continue;
        // Don't look at the expr if it is defined in a different file.
        auto humaneLoc = manager->getHumaneLoc(invocation.loc, SourceLocType::Actual);
        if (!humaneLoc.pathInfo.foundPath.getUnownedSlice().endsWithCaseInsensitive(fileName))
            continue;
        SemanticToken token;
        token.line = (int)(humaneLoc.line);
        token.col = (int)(humaneLoc.column);
        token.length = (int)(invocation.name->text.getLength());
        token.type = SemanticTokenType::Macro;
        maybeInsertToken(token);
    }
    return result;
}

List<uint32_t> getEncodedTokens(List<SemanticToken>& tokens)
{
    List<uint32_t> result;
    if (tokens.getCount() == 0)
        return result;

    std::sort(tokens.begin(), tokens.end());

    // Encode the first token as is.
    result.add((uint32_t)tokens[0].line);
    result.add((uint32_t)tokens[0].col);
    result.add((uint32_t)tokens[0].length);
    result.add((uint32_t)tokens[0].type);
    result.add(0);

    // Encode the rest tokens as deltas.
    uint32_t prevLine = (uint32_t)tokens[0].line;
    uint32_t prevCol = (uint32_t)tokens[0].col;
    for (Index i = 1; i < tokens.getCount(); i++)
    {
        uint32_t thisLine = (uint32_t)tokens[i].line;
        uint32_t thisCol = (uint32_t)tokens[i].col;
        if (thisLine == prevLine && thisCol == prevCol)
            continue;

        uint32_t deltaLine = thisLine - prevLine;
        uint32_t deltaCol = deltaLine == 0 ? thisCol - prevCol : thisCol;

        result.add(deltaLine);
        result.add(deltaCol);
        result.add((uint32_t)tokens[i].length);
        result.add((uint32_t)tokens[i].type);
        result.add(0);

        prevLine = thisLine;
        prevCol = thisCol;
    }

    return result;
}

} // namespace Slang
