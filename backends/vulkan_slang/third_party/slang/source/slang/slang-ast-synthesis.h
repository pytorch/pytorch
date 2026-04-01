// slang-ast-synthesis.h

#pragma once

#include "slang-syntax.h"

namespace Slang
{

struct ASTEmitScope
{
    ContainerDecl* m_parent = nullptr;
    SeqStmt* m_parentSeqStmt = nullptr;
    Scope* m_scope = nullptr;
};
class ASTSynthesizer
{
private:
    ASTBuilder* m_builder;
    NamePool* m_namePool;
    List<ASTEmitScope> m_scopeStack;

public:
    ASTSynthesizer(ASTBuilder* builder, NamePool* namePool)
        : m_builder(builder), m_namePool(namePool)
    {
    }

    ASTBuilder* getBuilder() { return m_builder; }

    Scope* getScope(ContainerDecl* decl)
    {
        for (auto container = decl; container; container = container->parentDecl)
        {
            if (container->ownedScope)
            {
                return container->ownedScope;
            }
        }
        return nullptr;
    }

    // Create a scope for `decl` and push it to scope stack
    void pushScopeForContainer(ContainerDecl* decl)
    {
        if (decl->ownedScope)
        {
            // if decl already owns a scope, don't create a new one.
            pushContainerScope(decl);
            return;
        }

        auto parentScope = getScope(decl);
        decl->ownedScope = m_builder->create<Scope>();
        decl->ownedScope->parent = parentScope;
        decl->ownedScope->containerDecl = decl;
        pushContainerScope(decl);
    }

    // Push `decl` and its associated scope to scope stack
    void pushContainerScope(ContainerDecl* decl)
    {
        ASTEmitScope scope = getCurrentScope();
        scope.m_parent = decl;
        scope.m_scope = getScope(decl);
        m_scopeStack.add(scope);
    }

    Scope* pushVarScope()
    {
        ASTEmitScope scope = getCurrentScope();
        auto scopeDecl = m_builder->create<ScopeDecl>();
        auto newScope = m_builder->create<Scope>();
        ContainerDecl::setParent(scope.m_parent, scopeDecl);
        newScope->parent = scope.m_scope;
        newScope->containerDecl = scopeDecl;
        scope.m_scope = newScope;
        m_scopeStack.add(scope);
        return newScope;
    }

    void _addStmtToScope(Stmt* stmt)
    {
        auto scope = getCurrentScope();
        if (scope.m_parentSeqStmt)
        {
            scope.m_parentSeqStmt->stmts.add(stmt);
        }
    }

    SeqStmt* pushSeqStmtScope()
    {
        ASTEmitScope scope = getCurrentScope();
        scope.m_parentSeqStmt = m_builder->create<SeqStmt>();
        m_scopeStack.add(scope);
        return scope.m_parentSeqStmt;
    }

    void popScope() { m_scopeStack.removeLast(); }

    ASTEmitScope getCurrentScope()
    {
        if (m_scopeStack.getCount())
            return m_scopeStack.getLast();
        return ASTEmitScope();
    }

    ForStmt* emitFor(Expr* initVal, Expr* finalVal, VarDecl*& outIndexVar);

    Expr* emitBinaryExpr(UnownedStringSlice operatorToken, Expr* left, Expr* right);

    Expr* emitPrefixExpr(UnownedStringSlice operatorToken, Expr* base);

    Expr* emitPostfixExpr(UnownedStringSlice operatorToken, Expr* base);

    Expr* emitThisExpr();
    Expr* emitVarExpr(Name* name);
    Expr* emitVarExpr(VarDeclBase* var);
    Expr* emitVarExpr(VarDeclBase* var, Type* type);
    Expr* emitVarExpr(DeclStmt* varStmt, Type* type);
    Expr* emitStaticTypeExpr(Type* type);

    Expr* emitIntConst(int value);

    Expr* emitGetArrayLengthExpr(Expr* arrayExpr);

    Expr* emitMemberExpr(Expr* base, Name* name);
    Expr* emitMemberExpr(Type* base, Name* name);

    Expr* emitIndexExpr(Expr* base, Expr* index);

    Expr* emitAssignExpr(Expr* left, Expr* right);
    ExpressionStmt* emitAssignStmt(Expr* left, Expr* right)
    {
        return emitExprStmt(emitAssignExpr(left, right));
    }

    Expr* emitInvokeExpr(Expr* callee, List<Expr*>&& args);
    Expr* emitCtorInvokeExpr(Expr* callee, List<Expr*>&& args);

    Expr* emitGenericAppExpr(Expr* genericExpr, List<Expr*>&& args);

    DeclStmt* emitVarDeclStmt(Type* type, Name* name = nullptr, Expr* initVal = nullptr);

    ExpressionStmt* emitExprStmt(Expr* expr);

    ReturnStmt* emitReturnStmt(Expr* expr);
};

} // namespace Slang
