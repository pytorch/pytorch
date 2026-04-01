#pragma once
#include "slang-syntax.h"
#include "slang-visitor.h"

namespace Slang
{
template<typename Callback, typename Filter>
struct ASTIterator
{
    const Callback& callback;
    const Filter& filter;
    ASTIterator(const Callback& func, const Filter& filterFunc)
        : callback(func), filter(filterFunc)
    {
    }

    void visitDecl(DeclBase* decl);
    void visitExpr(Expr* expr);
    void visitStmt(Stmt* stmt);

    void maybeDispatchCallback(SyntaxNode* node)
    {
        if (node)
        {
            callback(node);
        }
    }

    struct ASTIteratorExprVisitor : public ExprVisitor<ASTIteratorExprVisitor>
    {
    public:
        ASTIterator* iterator;
        ASTIteratorExprVisitor(ASTIterator* iter)
            : iterator(iter)
        {
        }
        void dispatchIfNotNull(Expr* expr)
        {
            if (!expr)
                return;
            this->dispatch(expr);
        }
        void visitExpr(Expr*) {}
        void visitBoolLiteralExpr(BoolLiteralExpr* expr) { iterator->maybeDispatchCallback(expr); }
        void visitNullPtrLiteralExpr(NullPtrLiteralExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
        }
        void visitNoneLiteralExpr(NoneLiteralExpr* expr) { iterator->maybeDispatchCallback(expr); }
        void visitIntegerLiteralExpr(IntegerLiteralExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
        }
        void visitOpenRefExpr(OpenRefExpr* expr) { dispatchIfNotNull(expr->innerExpr); }
        void visitFloatingPointLiteralExpr(FloatingPointLiteralExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
        }
        void visitStringLiteralExpr(StringLiteralExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
        }
        void visitIncompleteExpr(IncompleteExpr* expr) { iterator->maybeDispatchCallback(expr); }
        void visitIndexExpr(IndexExpr* subscriptExpr)
        {
            iterator->maybeDispatchCallback(subscriptExpr);
            dispatchIfNotNull(subscriptExpr->baseExpression);
            for (auto arg : subscriptExpr->indexExprs)
                dispatchIfNotNull(arg);
        }

        void visitBuiltinCastExpr(BuiltinCastExpr* expr) { dispatchIfNotNull(expr->base); }
        void visitParenExpr(ParenExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->base);
        }

        void visitAssignExpr(AssignExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->left);
            dispatchIfNotNull(expr->right);
        }

        void visitGenericAppExpr(GenericAppExpr* genericAppExpr)
        {
            iterator->maybeDispatchCallback(genericAppExpr);

            dispatchIfNotNull(genericAppExpr->functionExpr);
            for (auto arg : genericAppExpr->arguments)
                dispatchIfNotNull(arg);
        }

        void visitSharedTypeExpr(SharedTypeExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->base.exp);
        }

        void visitInvokeExpr(InvokeExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);

            dispatchIfNotNull(expr->functionExpr);
            dispatchIfNotNull(expr->originalFunctionExpr);

            for (auto arg : expr->arguments)
                dispatchIfNotNull(arg);
        }

        void visitVarExpr(VarExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->originalExpr);
        }

        void visitTryExpr(TryExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->base);
        }

        void visitTypeCastExpr(TypeCastExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);

            dispatchIfNotNull(expr->functionExpr);
            for (auto arg : expr->arguments)
                dispatchIfNotNull(arg);
        }
        void visitPackExpr(PackExpr* expr)
        {
            for (auto arg : expr->args)
                dispatchIfNotNull(arg);
        }

        void visitExpandExpr(ExpandExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->baseExpr);
        }

        void visitEachExpr(EachExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->baseExpr);
        }

        void visitDerefExpr(DerefExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->base);
        }

        void visitMatrixSwizzleExpr(MatrixSwizzleExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->base);
        }
        void visitSwizzleExpr(SwizzleExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->base);
        }
        void visitOverloadedExpr(OverloadedExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->base);
            dispatchIfNotNull(expr->originalExpr);
        }
        void visitOverloadedExpr2(OverloadedExpr2* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->base);
            for (auto candidate : expr->candidiateExprs)
            {
                dispatchIfNotNull(candidate);
            }
        }
        void visitAggTypeCtorExpr(AggTypeCtorExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->base.exp);
            for (auto arg : expr->arguments)
            {
                dispatchIfNotNull(arg);
            }
        }
        void visitCastToSuperTypeExpr(CastToSuperTypeExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->valueArg);
        }
        void visitModifierCastExpr(ModifierCastExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->valueArg);
        }
        void visitLetExpr(LetExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            iterator->visitDecl(expr->decl);
            dispatchIfNotNull(expr->body);
        }
        void visitExtractExistentialValueExpr(ExtractExistentialValueExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->originalExpr);
        }

        void visitDeclRefExpr(DeclRefExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->originalExpr);
        }

        void visitStaticMemberExpr(StaticMemberExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->baseExpression);
        }

        void visitMemberExpr(MemberExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->baseExpression);
        }

        void visitInitializerListExpr(InitializerListExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            for (auto arg : expr->args)
            {
                dispatchIfNotNull(arg);
            }
        }

        void visitThisExpr(ThisExpr* expr) { iterator->maybeDispatchCallback(expr); }
        void visitThisTypeExpr(ThisTypeExpr* expr) { iterator->maybeDispatchCallback(expr); }
        void visitReturnValExpr(ReturnValExpr* expr) { iterator->maybeDispatchCallback(expr); }

        void visitAndTypeExpr(AndTypeExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->left.exp);
            dispatchIfNotNull(expr->right.exp);
        }
        void visitModifiedTypeExpr(ModifiedTypeExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->base.exp);
        }
        void visitFuncTypeExpr(FuncTypeExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            for (const auto& t : expr->parameters)
                dispatchIfNotNull(t.exp);
            dispatchIfNotNull(expr->result.exp);
        }
        void visitTupleTypeExpr(TupleTypeExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            for (auto t : expr->members)
                dispatchIfNotNull(t.exp);
        }
        void visitPointerTypeExpr(PointerTypeExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->base.exp);
        }
        void visitAsTypeExpr(AsTypeExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->value);
            dispatchIfNotNull(expr->typeExpr);
        }
        void visitIsTypeExpr(IsTypeExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->value);
            dispatchIfNotNull(expr->typeExpr.exp);
        }
        void visitMakeOptionalExpr(MakeOptionalExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->value);
            dispatchIfNotNull(expr->typeExpr);
        }
        void visitPartiallyAppliedGenericExpr(PartiallyAppliedGenericExpr* expr)
        {
            dispatchIfNotNull(expr->originalExpr);
        }

        void visitHigherOrderInvokeExpr(HigherOrderInvokeExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            dispatchIfNotNull(expr->baseFunction);
        }

        void visitTreatAsDifferentiableExpr(TreatAsDifferentiableExpr* expr)
        {
            dispatchIfNotNull(expr->innerExpr);
        }

        void visitSPIRVAsmExpr(SPIRVAsmExpr* expr)
        {
            iterator->maybeDispatchCallback(expr);
            for (const auto& i : expr->insts)
            {
                dispatchIfNotNull(i.opcode.expr);
                for (const auto& o : i.operands)
                    dispatchIfNotNull(o.expr);
            }
        }

        void visitDetachExpr(DetachExpr* expr) { iterator->maybeDispatchCallback(expr); }
    };

    struct ASTIteratorStmtVisitor : public StmtVisitor<ASTIteratorStmtVisitor>
    {
        ASTIterator* iterator;
        ASTIteratorStmtVisitor(ASTIterator* iter)
            : iterator(iter)
        {
        }

        void dispatchIfNotNull(Stmt* stmt)
        {
            if (!stmt)
                return;
            this->dispatch(stmt);
        }

        void visitDeclStmt(DeclStmt* stmt)
        {
            iterator->maybeDispatchCallback(stmt);
            iterator->visitDecl(stmt->decl);
        }

        void visitBlockStmt(BlockStmt* stmt)
        {
            iterator->maybeDispatchCallback(stmt);
            dispatchIfNotNull(stmt->body);
        }

        void visitSeqStmt(SeqStmt* seqStmt)
        {
            iterator->maybeDispatchCallback(seqStmt);
            for (auto stmt : seqStmt->stmts)
                dispatchIfNotNull(stmt);
        }

        void visitLabelStmt(LabelStmt* stmt)
        {
            iterator->maybeDispatchCallback(stmt);
            dispatchIfNotNull(stmt->innerStmt);
        }

        void visitBreakStmt(BreakStmt* stmt) { iterator->maybeDispatchCallback(stmt); }

        void visitContinueStmt(ContinueStmt* stmt) { iterator->maybeDispatchCallback(stmt); }

        void visitDoWhileStmt(DoWhileStmt* stmt)
        {
            iterator->maybeDispatchCallback(stmt);
            iterator->visitExpr(stmt->predicate);
            dispatchIfNotNull(stmt->statement);
        }

        void visitForStmt(ForStmt* stmt)
        {
            iterator->maybeDispatchCallback(stmt);
            dispatchIfNotNull(stmt->initialStatement);
            iterator->visitExpr(stmt->predicateExpression);
            iterator->visitExpr(stmt->sideEffectExpression);
            dispatchIfNotNull(stmt->statement);
        }

        void visitCompileTimeForStmt(CompileTimeForStmt* stmt)
        {
            iterator->maybeDispatchCallback(stmt);
        }

        void visitSwitchStmt(SwitchStmt* stmt)
        {
            iterator->maybeDispatchCallback(stmt);
            iterator->visitExpr(stmt->condition);
            dispatchIfNotNull(stmt->body);
        }

        void visitCaseStmt(CaseStmt* stmt)
        {
            iterator->maybeDispatchCallback(stmt);
            iterator->visitExpr(stmt->expr);
        }

        void visitTargetSwitchStmt(TargetSwitchStmt* stmt)
        {
            iterator->maybeDispatchCallback(stmt);
            for (auto c : stmt->targetCases)
                dispatchIfNotNull(c);
        }

        void visitTargetCaseStmt(TargetCaseStmt* stmt)
        {
            iterator->maybeDispatchCallback(stmt);
            iterator->visitStmt(stmt->body);
        }

        void visitIntrinsicAsmStmt(IntrinsicAsmStmt*) {}

        void visitDefaultStmt(DefaultStmt* stmt) { iterator->maybeDispatchCallback(stmt); }

        void visitIfStmt(IfStmt* stmt)
        {
            iterator->maybeDispatchCallback(stmt);
            iterator->visitExpr(stmt->predicate);
            dispatchIfNotNull(stmt->positiveStatement);
            dispatchIfNotNull(stmt->negativeStatement);
        }

        void visitUnparsedStmt(UnparsedStmt* stmt) { iterator->maybeDispatchCallback(stmt); }

        void visitEmptyStmt(EmptyStmt* stmt) { iterator->maybeDispatchCallback(stmt); }

        void visitDiscardStmt(DiscardStmt* stmt) { iterator->maybeDispatchCallback(stmt); }

        void visitReturnStmt(ReturnStmt* stmt)
        {
            iterator->maybeDispatchCallback(stmt);
            iterator->visitExpr(stmt->expression);
        }

        void visitDeferStmt(DeferStmt* stmt)
        {
            iterator->maybeDispatchCallback(stmt);
            dispatchIfNotNull(stmt->statement);
        }

        void visitWhileStmt(WhileStmt* stmt)
        {
            iterator->maybeDispatchCallback(stmt);
            iterator->visitExpr(stmt->predicate);
            dispatchIfNotNull(stmt->statement);
        }

        void visitGpuForeachStmt(GpuForeachStmt* stmt) { iterator->maybeDispatchCallback(stmt); }

        void visitExpressionStmt(ExpressionStmt* stmt)
        {
            iterator->maybeDispatchCallback(stmt);
            iterator->visitExpr(stmt->expression);
        }
    };
};

template<typename CallbackFunc, typename FilterFunc>
void ASTIterator<CallbackFunc, FilterFunc>::visitDecl(DeclBase* decl)
{
    // Don't look at the decl if it is defined in a different file.
    if (!filter(decl))
        return;

    maybeDispatchCallback(decl);
    if (auto funcDecl = as<FunctionDeclBase>(decl))
    {
        visitStmt(funcDecl->body);
        visitExpr(funcDecl->returnType.exp);
    }
    else if (auto propertyDecl = as<PropertyDecl>(decl))
    {
        visitExpr(propertyDecl->type.exp);
    }
    else if (auto varDecl = as<VarDeclBase>(decl))
    {
        visitExpr(varDecl->type.exp);
        visitExpr(varDecl->initExpr);
    }
    else if (auto genericDecl = as<GenericDecl>(decl))
    {
        visitDecl(genericDecl->inner);
    }
    else if (auto typeConstraint = as<TypeConstraintDecl>(decl))
    {
        if (auto genericTypeConstraint = as<GenericTypeConstraintDecl>(typeConstraint))
        {
            // A generic constraint decl has a left hand side and right hand side expression
            // for the base and super type of the constraint.
            // In the case of a folded-in constraint syntax as in `Foo<T:IBar>`,
            // the left hand side of the constraint is represented by the same token
            // as the parameter decl itself, so we don't need to traverse into it.
            // In the case of `Foo<T> where T:IBar`, the left hand side is its own
            // expression so we do want to traverse it.
            if (genericTypeConstraint->whereTokenLoc.isValid())
                visitExpr(genericTypeConstraint->sub.exp);
        }
        visitExpr(typeConstraint->getSup().exp);
    }
    else if (auto typedefDecl = as<TypeDefDecl>(decl))
    {
        visitExpr(typedefDecl->type.exp);
    }
    else if (auto extDecl = as<ExtensionDecl>(decl))
    {
        visitExpr(extDecl->targetType.exp);
    }
    else if (auto usingDecl = as<UsingDecl>(decl))
    {
        visitExpr(usingDecl->arg);
    }
    if (auto container = as<ContainerDecl>(decl))
    {
        for (auto member : container->members)
        {
            visitDecl(member);
        }
        if (auto aggTypeDecl = as<AggTypeDecl>(decl))
            visitExpr(aggTypeDecl->wrappedType.exp);
    }
    for (auto modifier : decl->modifiers)
    {
        if (auto attr = as<Attribute>(modifier))
        {
            maybeDispatchCallback(attr);
            for (auto arg : attr->args)
                visitExpr(arg);
        }
    }
}
template<typename CallbackFunc, typename FilterFunc>
void ASTIterator<CallbackFunc, FilterFunc>::visitExpr(Expr* expr)
{
    ASTIteratorExprVisitor visitor(this);
    visitor.dispatchIfNotNull(expr);
}
template<typename CallbackFunc, typename FilterFunc>
void ASTIterator<CallbackFunc, FilterFunc>::visitStmt(Stmt* stmt)
{
    ASTIteratorStmtVisitor visitor(this);
    visitor.dispatchIfNotNull(stmt);
}

template<typename Func, typename FilterFunc>
void iterateAST(SyntaxNode* node, const FilterFunc& filterFunc, const Func& f)
{
    ASTIterator<Func, FilterFunc> iter(f, filterFunc);
    if (auto decl = as<Decl>(node))
    {
        iter.visitDecl(decl);
    }
    else if (auto expr = as<Expr>(node))
    {
        iter.visitExpr(expr);
    }
    else if (auto stmt = as<Stmt>(node))
    {
        iter.visitStmt(stmt);
    }
}

template<typename Func>
void iterateASTWithLanguageServerFilter(
    UnownedStringSlice fileName,
    SourceManager* sourceManager,
    SyntaxNode* node,
    const Func& f)
{
    auto filter = [&](DeclBase* decl)
    {
        if (as<ConstructorDecl>(decl) && decl->findModifier<SynthesizedModifier>())
            return false;
        return as<NamespaceDeclBase>(decl) ||
               sourceManager->getHumaneLoc(decl->loc, SourceLocType::Actual)
                   .pathInfo.foundPath.getUnownedSlice()
                   .endsWithCaseInsensitive(fileName);
    };
    iterateAST(node, filter, f);
}
} // namespace Slang
