#include "slang-language-server-ast-lookup.h"

#include "slang-visitor.h"
#include "slang-workspace-version.h"

namespace Slang
{
struct ASTLookupContext
{
    DocumentVersion* doc;
    SourceManager* sourceManager;
    List<SyntaxNode*> nodePath;
    ASTLookupType findType;
    Int line;
    Int col;
    Loc cursorLoc;
    UnownedStringSlice sourceFileName;
    List<ASTLookupResult> results;

    Loc getLoc(SourceLoc loc, String* outFileName)
    {
        return Loc::fromSourceLoc(sourceManager, loc, outFileName);
    }
};

Loc Loc::fromSourceLoc(SourceManager* manager, SourceLoc loc, String* outFileName)
{
    auto humaneLoc = manager->getHumaneLoc(loc, SourceLocType::Actual);
    if (outFileName)
        *outFileName = humaneLoc.pathInfo.foundPath;
    return Loc{humaneLoc.line, humaneLoc.column};
}

struct PushNode
{
    ASTLookupContext* context;
    PushNode(ASTLookupContext* ctx, SyntaxNode* node)
    {
        context = ctx;
        context->nodePath.add(node);
    }
    ~PushNode()
    {
        if (context)
            context->nodePath.removeLast();
    }
};

static Index _getDeclNameLength(Name* name, Decl* optionalDecl = nullptr)
{
    if (!name)
        return 0;
    if (name->text.startsWith("$"))
    {
        if (auto ctorDecl = as<ConstructorDecl>(optionalDecl))
        {
            if (ctorDecl->parentDecl && optionalDecl->parentDecl->getName())
            {
                return optionalDecl->parentDecl->getName()->text.getLength();
            }
        }
        return 0;
    }
    // HACK: our __subscript functions currently have a name "operator[]".
    // and our operator() functions have a name "()".
    // Since this isn't the name that actually appears in user's code,
    // we need to shorten its reported length to 1 for now.
    if (name->text.startsWith("operator") || name->text.startsWith("()"))
    {
        return 1;
    }
    return name->text.getLength();
}

bool _isLocInRange(ASTLookupContext* context, SourceLoc loc, Int length)
{
    auto humaneLoc = context->sourceManager->getHumaneLoc(loc, SourceLocType::Actual);
    return humaneLoc.line == context->line && context->col >= humaneLoc.column &&
           context->col < humaneLoc.column + length &&
           humaneLoc.pathInfo.foundPath.getUnownedSlice().endsWithCaseInsensitive(
               context->sourceFileName);
}
bool _isLocInRange(ASTLookupContext* context, SourceLoc start, SourceLoc end)
{
    auto startLoc = context->sourceManager->getHumaneLoc(start, SourceLocType::Actual);
    auto endLoc = context->sourceManager->getHumaneLoc(end, SourceLocType::Actual);

    Loc s{startLoc.line, startLoc.column};
    Loc e{endLoc.line, endLoc.column};
    Loc c{context->line, context->col};
    return s <= c && c < e &&
           startLoc.pathInfo.foundPath.getUnownedSlice().endsWithCaseInsensitive(
               context->sourceFileName);
}

bool _findAstNodeImpl(ASTLookupContext& context, SyntaxNode* node);

struct ASTLookupExprVisitor : public ExprVisitor<ASTLookupExprVisitor, bool>
{
public:
    ASTLookupContext* context;

    ASTLookupExprVisitor(ASTLookupContext* ctx)
        : context(ctx)
    {
    }
    bool dispatchIfNotNull(Expr* expr)
    {
        if (!expr)
            return false;
        return dispatch(expr);
    }
    bool visitExpr(Expr*) { return false; }
    bool visitBoolLiteralExpr(BoolLiteralExpr*) { return false; }
    bool visitNullPtrLiteralExpr(NullPtrLiteralExpr*) { return false; }
    bool visitIntegerLiteralExpr(IntegerLiteralExpr*) { return false; }
    bool visitFloatingPointLiteralExpr(FloatingPointLiteralExpr*) { return false; }
    bool visitStringLiteralExpr(StringLiteralExpr*) { return false; }
    bool visitIncompleteExpr(IncompleteExpr*) { return false; }
    bool visitIndexExpr(IndexExpr* subscriptExpr)
    {
        for (auto arg : subscriptExpr->indexExprs)
            if (dispatchIfNotNull(arg))
                return true;
        return dispatchIfNotNull(subscriptExpr->baseExpression);
    }

    bool visitSizeOfLikeExpr(SizeOfLikeExpr* expr)
    {
        int tokenLength = 0;
        if (as<CountOfExpr>(expr))
            tokenLength = 7; // strlen("countof");
        else if (as<SizeOfExpr>(expr))
            tokenLength = 6; // strlen("sizeof");
        else if (as<AlignOfExpr>(expr))
            tokenLength = 7; // strlen("alignof");

        if (_isLocInRange(context, expr->loc, tokenLength))
        {
            ASTLookupResult result;
            result.path = context->nodePath;
            result.path.add(expr);
            context->results.add(result);
            return true;
        }
        return dispatchIfNotNull(expr->value);
    }

    bool visitParenExpr(ParenExpr* expr) { return dispatchIfNotNull(expr->base); }

    bool visitBuiltinCastExpr(BuiltinCastExpr* expr) { return dispatchIfNotNull(expr->base); }

    bool visitAssignExpr(AssignExpr* expr)
    {
        if (dispatchIfNotNull(expr->left))
            return true;
        return dispatchIfNotNull(expr->right);
    }

    bool visitGenericAppExpr(GenericAppExpr* genericAppExpr)
    {
        if (dispatchIfNotNull(genericAppExpr->functionExpr))
            return true;
        for (auto arg : genericAppExpr->arguments)
            if (dispatchIfNotNull(arg))
                return true;
        return false;
    }

    bool visitSharedTypeExpr(SharedTypeExpr* expr) { return dispatchIfNotNull(expr->base.exp); }

    bool visitInvokeExpr(InvokeExpr* expr)
    {
        PushNode pushNodeRAII(context, expr);
        if (dispatchIfNotNull(expr->functionExpr))
            return true;
        if (dispatchIfNotNull(expr->originalFunctionExpr))
            return true;
        for (auto arg : expr->arguments)
            if (dispatchIfNotNull(arg))
                return true;
        if (context->findType == ASTLookupType::Invoke && expr->argumentDelimeterLocs.getCount())
        {
            String fileName;
            Loc start = context->getLoc(expr->argumentDelimeterLocs.getFirst(), &fileName);
            Loc end = context->getLoc(expr->argumentDelimeterLocs.getLast(), nullptr);
            if (fileName.getUnownedSlice().endsWithCaseInsensitive(context->sourceFileName) &&
                start < context->cursorLoc && context->cursorLoc <= end)
            {
                ASTLookupResult result;
                result.path = context->nodePath;
                result.path.add(expr);
                context->results.add(result);
                return true;
            }
        }
        return false;
    }

    bool visitVarExpr(VarExpr* expr)
    {
        if (expr->name && expr->declRef.getDecl())
        {
            if (expr->declRef.getDecl()->hasModifier<ImplicitConversionModifier>())
                return false;
            Int declLength = 0;
            if (const auto ctorDecl = as<ConstructorDecl>(expr->declRef.getDecl()))
            {
                auto humaneLoc =
                    context->sourceManager->getHumaneLoc(expr->loc, SourceLocType::Actual);
                declLength = context->doc->getTokenLength(humaneLoc.line, humaneLoc.column);
            }
            else
            {
                declLength = _getDeclNameLength(expr->name, expr->declRef.getDecl());
            }
            if (_isLocInRange(context, expr->loc, declLength))
            {
                ASTLookupResult result;
                result.path = context->nodePath;
                result.path.add(expr);
                context->results.add(result);
                return true;
            }
        }

        return dispatchIfNotNull(expr->originalExpr);
    }

    bool visitTypeCastExpr(TypeCastExpr* expr)
    {
        if (dispatchIfNotNull(expr->functionExpr))
            return true;
        for (auto arg : expr->arguments)
            if (dispatchIfNotNull(arg))
                return true;
        return false;
    }

    bool visitDerefExpr(DerefExpr* expr) { return dispatchIfNotNull(expr->base); }
    bool visitMatrixSwizzleExpr(MatrixSwizzleExpr* expr)
    {
        if (_isLocInRange(context, expr->memberOpLoc, 0))
        {
            ASTLookupResult result;
            result.path = context->nodePath;
            result.path.add(expr);
            context->results.add(result);
            return true;
        }
        return dispatchIfNotNull(expr->base);
    }
    bool visitSwizzleExpr(SwizzleExpr* expr)
    {
        Index tokenLength = expr->elementIndices.getCount();
        if (expr->base && as<TupleType>(expr->base->type))
            tokenLength *= 2;
        if (_isLocInRange(context, expr->loc, tokenLength))
        {
            ASTLookupResult result;
            result.path = context->nodePath;
            result.path.add(expr);
            context->results.add(result);
            return true;
        }
        return dispatchIfNotNull(expr->base);
    }
    bool visitOverloadedExpr(OverloadedExpr* expr)
    {
        {
            PushNode pushNode(context, expr);
            if (dispatchIfNotNull(expr->base))
                return true;
        }
        {
            PushNode pushNode(context, expr);
            if (dispatchIfNotNull(expr->originalExpr))
                return true;
        }
        if (expr->lookupResult2.getName() &&
            _isLocInRange(context, expr->loc, _getDeclNameLength(expr->lookupResult2.getName())))
        {
            ASTLookupResult result;
            result.path = context->nodePath;
            result.path.add(expr);
            context->results.add(result);
            return true;
        }
        return false;
    }
    bool visitOverloadedExpr2(OverloadedExpr2* expr)
    {
        if (dispatchIfNotNull(expr->base))
            return true;
        bool result = false;
        PushNode pushNode(context, expr);
        for (auto candidate : expr->candidiateExprs)
        {
            result |= dispatchIfNotNull(candidate);
        }
        return result;
    }
    bool visitAggTypeCtorExpr(AggTypeCtorExpr* expr)
    {
        if (dispatchIfNotNull(expr->base.exp))
            return true;
        for (auto arg : expr->arguments)
        {
            if (dispatchIfNotNull(arg))
                return true;
        }
        return false;
    }
    bool visitCastToSuperTypeExpr(CastToSuperTypeExpr* expr)
    {
        return dispatchIfNotNull(expr->valueArg);
    }
    bool visitModifierCastExpr(ModifierCastExpr* expr) { return dispatchIfNotNull(expr->valueArg); }
    bool visitLetExpr(LetExpr* expr)
    {
        if (dispatchIfNotNull(expr->body))
            return true;
        return _findAstNodeImpl(*context, expr->decl);
    }
    bool visitExtractExistentialValueExpr(ExtractExistentialValueExpr* expr)
    {
        if (expr->declRef.getDecl() && expr->declRef.getName() &&
            _isLocInRange(context, expr->loc, _getDeclNameLength(expr->declRef.getName())))
        {
            ASTLookupResult result;
            result.path = context->nodePath;
            result.path.add(expr);
            context->results.add(result);
            return true;
        }
        return false;
    }

    bool visitDeclRefExpr(DeclRefExpr* expr)
    {
        if (expr->declRef.getDecl() && expr->declRef.getDecl()->getName() &&
            _isLocInRange(
                context,
                expr->loc,
                _getDeclNameLength(expr->declRef.getDecl()->getName())))
        {
            if (expr->declRef.getDecl()->hasModifier<ImplicitConversionModifier>())
                return false;
            ASTLookupResult result;
            result.path = context->nodePath;
            result.path.add(expr);
            context->results.add(result);
            return true;
        }
        return dispatchIfNotNull(expr->originalExpr);
    }

    bool visitStaticMemberExpr(StaticMemberExpr* expr)
    {
        if (_isLocInRange(context, expr->memberOperatorLoc, 0))
        {
            ASTLookupResult result;
            result.path = context->nodePath;
            result.path.add(expr);
            context->results.add(result);
            return true;
        }
        if (visitDeclRefExpr(expr))
            return true;
        return dispatchIfNotNull(expr->baseExpression);
    }

    bool visitMemberExpr(MemberExpr* expr)
    {
        if (_isLocInRange(context, expr->memberOperatorLoc, 0))
        {
            ASTLookupResult result;
            result.path = context->nodePath;
            result.path.add(expr);
            context->results.add(result);
            return true;
        }
        if (visitDeclRefExpr(expr))
            return true;
        return dispatchIfNotNull(expr->baseExpression);
    }

    bool visitOpenRefExpr(OpenRefExpr* expr) { return dispatchIfNotNull(expr->innerExpr); }

    bool visitInitializerListExpr(InitializerListExpr* expr)
    {
        for (auto arg : expr->args)
        {
            if (dispatchIfNotNull(arg))
                return true;
        }
        return false;
    }

    bool visitThisExpr(ThisExpr* expr)
    {
        static const int thisTokenLength = 4;
        if (_isLocInRange(context, expr->loc, thisTokenLength))
        {
            ASTLookupResult result;
            result.path = context->nodePath;
            result.path.add(expr);
            context->results.add(result);
            return true;
        }
        return false;
    }

    bool visitThisTypeExpr(ThisTypeExpr*) { return false; }
    bool visitAndTypeExpr(AndTypeExpr* expr)
    {
        if (dispatchIfNotNull(expr->left.exp))
            return true;
        return dispatchIfNotNull(expr->right.exp);
    }
    bool visitPointerTypeExpr(PointerTypeExpr* expr)
    {
        if (_isLocInRange(context, expr->loc, 0))
        {
            ASTLookupResult result;
            result.path = context->nodePath;
            result.path.add(expr);
            context->results.add(result);
            return true;
        }
        return dispatchIfNotNull(expr->base.exp);
    }
    bool visitAsTypeExpr(AsTypeExpr* expr)
    {
        if (dispatchIfNotNull(expr->value))
            return true;
        return dispatchIfNotNull(expr->typeExpr);
    }
    bool visitIsTypeExpr(IsTypeExpr* expr)
    {
        if (dispatchIfNotNull(expr->value))
            return true;
        return dispatchIfNotNull(expr->typeExpr.exp);
    }
    bool visitMakeOptionalExpr(MakeOptionalExpr* expr)
    {
        if (dispatchIfNotNull(expr->typeExpr))
            return true;
        return dispatchIfNotNull(expr->value);
    }
    bool visitPartiallyAppliedGenericExpr(PartiallyAppliedGenericExpr* expr)
    {
        return dispatchIfNotNull(expr->originalExpr);
    }
    bool visitSPIRVAsmExpr(SPIRVAsmExpr* expr)
    {
        for (const auto& i : expr->insts)
        {
            if (dispatchIfNotNull(i.opcode.expr))
                return true;
            for (const auto& o : i.operands)
                if (dispatchIfNotNull(o.expr))
                    return true;
        }
        return false;
    }
    bool visitModifiedTypeExpr(ModifiedTypeExpr* expr) { return dispatchIfNotNull(expr->base.exp); }
    bool visitFuncTypeExpr(FuncTypeExpr* expr)
    {
        for (const auto& t : expr->parameters)
        {
            if (!dispatchIfNotNull(t.exp))
                return false;
        }
        return dispatchIfNotNull(expr->result.exp);
    }
    bool visitTupleTypeExpr(TupleTypeExpr* expr)
    {
        for (auto t : expr->members)
        {
            if (dispatchIfNotNull(t.exp))
                return true;
        }
        return false;
    }
    bool visitTryExpr(TryExpr* expr) { return dispatchIfNotNull(expr->base); }
    bool visitPackExpr(PackExpr* expr)
    {
        for (auto arg : expr->args)
        {
            if (dispatchIfNotNull(arg))
                return true;
        }
        return false;
    }
    bool reportLookupResultIfInExprLeadingIdentifierRange(Expr* expr)
    {
        auto humaneLoc = context->sourceManager->getHumaneLoc(expr->loc, SourceLocType::Actual);
        auto tokenLen = context->doc->getTokenLength(humaneLoc.line, humaneLoc.column);
        if (_isLocInRange(context, expr->loc, tokenLen))
        {
            ASTLookupResult result;
            result.path = context->nodePath;
            result.path.add(expr);
            context->results.add(result);
            return true;
        }
        return false;
    }
    bool visitExpandExpr(ExpandExpr* expr)
    {
        if (reportLookupResultIfInExprLeadingIdentifierRange(expr))
            return true;
        return dispatchIfNotNull(expr->baseExpr);
    }
    bool visitEachExpr(EachExpr* expr)
    {
        if (reportLookupResultIfInExprLeadingIdentifierRange(expr))
            return true;
        return dispatchIfNotNull(expr->baseExpr);
    }
    bool visitHigherOrderInvokeExpr(HigherOrderInvokeExpr* expr)
    {
        if (reportLookupResultIfInExprLeadingIdentifierRange(expr))
            return true;
        return dispatchIfNotNull(expr->baseFunction);
    }
    bool visitTreatAsDifferentiableExpr(TreatAsDifferentiableExpr* expr)
    {
        return dispatchIfNotNull(expr->innerExpr);
    }
};

struct ASTLookupStmtVisitor : public StmtVisitor<ASTLookupStmtVisitor, bool>
{
    ASTLookupContext* context;

    ASTLookupStmtVisitor(ASTLookupContext* ctx)
        : context(ctx)
    {
    }

    bool dispatchIfNotNull(Stmt* stmt)
    {
        if (!stmt)
            return false;
        return dispatch(stmt);
    }

    bool checkExpr(Expr* expr)
    {
        if (!expr)
            return false;
        ASTLookupExprVisitor visitor(context);
        return visitor.dispatch(expr);
    }

    bool visitDeclStmt(DeclStmt* stmt) { return _findAstNodeImpl(*context, stmt->decl); }

    bool visitBlockStmt(BlockStmt* stmt)
    {
        if (!_isLocInRange(context, stmt->loc, stmt->closingSourceLoc))
            return false;
        return dispatchIfNotNull(stmt->body);
    }

    bool visitSeqStmt(SeqStmt* seqStmt)
    {
        for (auto stmt : seqStmt->stmts)
            if (dispatchIfNotNull(stmt))
                return true;
        return false;
    }

    bool visitLabelStmt(LabelStmt* stmt)
    {
        if (_isLocInRange(context, stmt->label.loc, stmt->label.getContent().getLength()))
            return true;
        return dispatchIfNotNull(stmt->innerStmt);
    }

    bool visitBreakStmt(BreakStmt*) { return false; }

    bool visitContinueStmt(ContinueStmt*) { return false; }

    bool visitDoWhileStmt(DoWhileStmt* stmt)
    {
        if (checkExpr(stmt->predicate))
            return true;
        return dispatchIfNotNull(stmt->statement);
    }

    bool visitForStmt(ForStmt* stmt)
    {
        if (dispatchIfNotNull(stmt->initialStatement))
            return true;
        if (checkExpr(stmt->predicateExpression))
            return true;
        if (checkExpr(stmt->sideEffectExpression))
            return true;
        return dispatchIfNotNull(stmt->statement);
    }

    bool visitCompileTimeForStmt(CompileTimeForStmt*) { return false; }

    bool visitSwitchStmt(SwitchStmt* stmt)
    {
        if (checkExpr(stmt->condition))
            return true;
        return dispatchIfNotNull(stmt->body);
    }

    bool visitCaseStmt(CaseStmt* stmt) { return checkExpr(stmt->expr); }

    bool visitTargetSwitchStmt(TargetSwitchStmt* stmt)
    {
        for (auto targetCase : stmt->targetCases)
            if (dispatchIfNotNull(targetCase))
                return true;
        return false;
    }

    bool visitTargetCaseStmt(TargetCaseStmt* stmt) { return dispatchIfNotNull(stmt->body); }

    bool visitIntrinsicAsmStmt(IntrinsicAsmStmt*) { return false; }

    bool visitDefaultStmt(DefaultStmt*) { return false; }

    bool visitIfStmt(IfStmt* stmt)
    {
        if (checkExpr(stmt->predicate))
            return true;
        if (dispatchIfNotNull(stmt->positiveStatement))
            return true;
        return dispatchIfNotNull(stmt->negativeStatement);
    }

    bool visitUnparsedStmt(UnparsedStmt*) { return false; }

    bool visitEmptyStmt(EmptyStmt*) { return false; }

    bool visitDiscardStmt(DiscardStmt*) { return false; }

    bool visitReturnStmt(ReturnStmt* stmt) { return checkExpr(stmt->expression); }

    bool visitDeferStmt(DeferStmt* stmt) { return dispatchIfNotNull(stmt->statement); }

    bool visitWhileStmt(WhileStmt* stmt)
    {
        if (checkExpr(stmt->predicate))
            return true;
        return dispatchIfNotNull(stmt->statement);
    }

    bool visitGpuForeachStmt(GpuForeachStmt*) { return false; }

    bool visitExpressionStmt(ExpressionStmt* stmt) { return checkExpr(stmt->expression); }
};

bool _findAstNodeImpl(ASTLookupContext& context, SyntaxNode* node)
{
    if (!node)
        return false;
    PushNode pushNodeRAII(&context, node);
    if (auto decl = as<Decl>(node))
    {
        if (decl->getName())
        {
            if (_isLocInRange(&context, decl->nameAndLoc.loc, _getDeclNameLength(decl->getName())))
            {
                bool isRealDeclName = true;
                for (auto modifier : decl->modifiers)
                {
                    if (as<SynthesizedModifier>(modifier))
                    {
                        isRealDeclName = false;
                        break;
                    }
                    if (as<ImplicitParameterGroupElementTypeModifier>(modifier))
                    {
                        isRealDeclName = false;
                        break;
                    }
                }
                if (isRealDeclName)
                {
                    ASTLookupResult result;
                    result.path = context.nodePath;
                    context.results.add(_Move(result));
                    return true;
                }
            }
        }
        if (auto funcDecl = as<FunctionDeclBase>(node))
        {
            ASTLookupStmtVisitor visitor(&context);
            if (visitor.dispatchIfNotNull(funcDecl->body))
                return true;
            ASTLookupExprVisitor exprVisitor(&context);
            if (exprVisitor.dispatchIfNotNull(funcDecl->returnType.exp))
                return true;
        }
        else if (auto propertyDecl = as<PropertyDecl>(node))
        {
            ASTLookupExprVisitor exprVisitor(&context);
            if (exprVisitor.dispatchIfNotNull(propertyDecl->type.exp))
                return true;
        }
        else if (auto varDecl = as<VarDeclBase>(node))
        {
            ASTLookupExprVisitor visitor(&context);
            if (visitor.dispatchIfNotNull(varDecl->type.exp))
                return true;
            if (visitor.dispatchIfNotNull(varDecl->initExpr))
                return true;
        }
        else if (auto genericDecl = as<GenericDecl>(node))
        {
            if (_findAstNodeImpl(context, genericDecl->inner))
                return true;
        }
        else if (auto typeConstraint = as<TypeConstraintDecl>(node))
        {
            ASTLookupExprVisitor visitor(&context);
            if (visitor.dispatchIfNotNull(typeConstraint->getSup().exp))
                return true;
            if (auto genTypeConstraint = as<GenericTypeConstraintDecl>(node))
            {
                if (genTypeConstraint->whereTokenLoc.isValid())
                {
                    if (visitor.dispatchIfNotNull(genTypeConstraint->sub.exp))
                        return true;
                }
            }
        }
        else if (auto typedefDecl = as<TypeDefDecl>(node))
        {
            ASTLookupExprVisitor visitor(&context);
            if (visitor.dispatchIfNotNull(typedefDecl->type.exp))
                return true;
        }
        else if (auto extDecl = as<ExtensionDecl>(node))
        {
            ASTLookupExprVisitor visitor(&context);
            if (visitor.dispatchIfNotNull(extDecl->targetType.exp))
                return true;
        }
        else if (auto usingDecl = as<UsingDecl>(node))
        {
            ASTLookupExprVisitor visitor(&context);
            if (visitor.dispatchIfNotNull(usingDecl->arg))
                return true;
        }
        else if (auto importDecl = as<FileReferenceDeclBase>(node))
        {
            if (_isLocInRange(&context, importDecl->startLoc, importDecl->endLoc))
            {
                ASTLookupResult result;
                result.path = context.nodePath;
                context.results.add(_Move(result));
                return true;
            }
        }

        for (auto modifier : decl->modifiers)
        {
            if (auto hlslSemantic = as<HLSLSemantic>(modifier))
            {
                if (_isLocInRange(
                        &context,
                        hlslSemantic->loc,
                        hlslSemantic->name.getContentLength()))
                {
                    ASTLookupResult result;
                    result.path = context.nodePath;
                    result.path.add(hlslSemantic);
                    context.results.add(result);
                    return true;
                }
            }
            else if (auto attribute = as<AttributeBase>(modifier))
            {
                if (attribute->getKeywordName() &&
                    _isLocInRange(
                        &context,
                        attribute->originalIdentifierToken.loc,
                        attribute->originalIdentifierToken.getContentLength()))
                {
                    ASTLookupResult result;
                    result.path = context.nodePath;
                    result.path.add(attribute);
                    context.results.add(result);
                    return true;
                }
                for (auto arg : attribute->args)
                {
                    ASTLookupExprVisitor exprVisitor(&context);
                    if (exprVisitor.dispatchIfNotNull(arg))
                        return true;
                }
            }
        }
        if (auto container = as<ContainerDecl>(node))
        {
            bool shouldInspectChildren = true;
            if (const auto genericDecl = as<GenericDecl>(node))
            {
            }
            else if (container->closingSourceLoc.getRaw() >= container->loc.getRaw())
            {
                if (!_isLocInRange(&context, container->loc, container->closingSourceLoc) &&
                    !as<NamespaceDeclBase>(container))
                {
                    shouldInspectChildren = false;
                }
            }
            if (shouldInspectChildren)
            {
                for (auto member : container->members)
                {
                    if (_findAstNodeImpl(context, member))
                        return true;
                }
            }
            if (auto aggTypeDecl = as<AggTypeDecl>(container))
            {
                ASTLookupExprVisitor visitor(&context);
                if (visitor.dispatchIfNotNull(aggTypeDecl->wrappedType.exp))
                    return true;
            }
        }
    }
    return false;
}

List<ASTLookupResult> findASTNodesAt(
    DocumentVersion* doc,
    SourceManager* sourceManager,
    ModuleDecl* moduleDecl,
    ASTLookupType findType,
    UnownedStringSlice fileName,
    Int line,
    Int col)
{
    ASTLookupContext context;
    context.sourceManager = sourceManager;
    context.line = line;
    context.col = col;
    context.cursorLoc = Loc{line, col};
    context.findType = findType;
    context.sourceFileName = fileName;
    context.doc = doc;
    _findAstNodeImpl(context, moduleDecl);
    return context.results;
}

} // namespace Slang
