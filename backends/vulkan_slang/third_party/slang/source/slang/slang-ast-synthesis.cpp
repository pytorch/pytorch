#include "slang-ast-synthesis.h"

namespace Slang
{
Expr* ASTSynthesizer::emitBinaryExpr(UnownedStringSlice operatorToken, Expr* left, Expr* right)
{
    auto infixExpr = m_builder->create<InfixExpr>();
    infixExpr->functionExpr = emitVarExpr(m_namePool->getName(operatorToken));
    ;
    infixExpr->arguments.add(left);
    infixExpr->arguments.add(right);
    return infixExpr;
}

Expr* ASTSynthesizer::emitPrefixExpr(UnownedStringSlice operatorToken, Expr* base)
{
    auto prefixExpr = m_builder->create<PrefixExpr>();
    prefixExpr->functionExpr = emitVarExpr(m_namePool->getName(operatorToken));
    ;
    prefixExpr->arguments.add(base);
    return prefixExpr;
}

Expr* ASTSynthesizer::emitPostfixExpr(UnownedStringSlice operatorToken, Expr* base)
{
    auto postfixExpr = m_builder->create<PostfixExpr>();
    postfixExpr->functionExpr = emitVarExpr(m_namePool->getName(operatorToken));
    ;
    postfixExpr->arguments.add(base);
    return postfixExpr;
}

ForStmt* ASTSynthesizer::emitFor(Expr* initVal, Expr* finalVal, VarDecl*& outIndexVar)
{
    auto parentStmt = getCurrentScope().m_parentSeqStmt;
    auto seqStmt = m_builder->create<SeqStmt>();
    auto scopeDecl = pushVarScope()->containerDecl;
    auto stmt = m_builder->create<ForStmt>();
    stmt->statement = seqStmt;
    stmt->scopeDecl = (ScopeDecl*)scopeDecl;
    auto declStmt = emitVarDeclStmt(nullptr, m_namePool->getName("S_synth_loop_index"), initVal);
    stmt->initialStatement = declStmt;
    outIndexVar = (VarDecl*)declStmt->decl;
    auto predicateExpr =
        emitBinaryExpr(UnownedStringSlice("<"), emitVarExpr(outIndexVar), finalVal);
    stmt->predicateExpression = predicateExpr;
    stmt->sideEffectExpression = emitPrefixExpr(UnownedStringSlice("++"), emitVarExpr(outIndexVar));
    parentStmt->stmts.add(stmt);
    m_scopeStack.getLast().m_parentSeqStmt = seqStmt;
    return stmt;
}

Expr* ASTSynthesizer::emitThisExpr()
{
    auto varExpr = m_builder->create<ThisExpr>();
    varExpr->scope = getCurrentScope().m_scope;
    return varExpr;
}

Expr* ASTSynthesizer::emitVarExpr(Name* name)
{
    auto scope = getCurrentScope();
    SLANG_RELEASE_ASSERT(scope.m_scope);
    auto varExpr = m_builder->create<VarExpr>();
    varExpr->name = name;
    varExpr->scope = scope.m_scope;
    return varExpr;
}

Expr* ASTSynthesizer::emitVarExpr(VarDeclBase* varDecl)
{
    auto varExpr = m_builder->create<VarExpr>();
    varExpr->declRef = makeDeclRef<Decl>(varDecl);
    varExpr->type = varDecl->type.type;
    return varExpr;
}

Expr* ASTSynthesizer::emitVarExpr(VarDeclBase* var, Type* type)
{
    auto expr = m_builder->create<VarExpr>();
    expr->declRef = makeDeclRef<Decl>(var);
    expr->type.type = type;
    expr->type.isLeftValue = true;
    return expr;
}

Expr* ASTSynthesizer::emitVarExpr(DeclStmt* varStmt, Type* type)
{
    auto expr = m_builder->create<VarExpr>();
    expr->declRef = makeDeclRef<Decl>(as<Decl>(varStmt->decl));
    expr->type.type = type;
    expr->type.isLeftValue = true;
    return expr;
}

Expr* ASTSynthesizer::emitStaticTypeExpr(Type* type)
{
    auto expr = m_builder->create<SharedTypeExpr>();
    expr->type.type = m_builder->getTypeType(type);
    expr->checked = true;
    return expr;
}

Expr* ASTSynthesizer::emitIntConst(int value)
{
    auto expr = m_builder->create<IntegerLiteralExpr>();
    expr->type.type = m_builder->getIntType();
    expr->value = value;
    return expr;
}

Expr* ASTSynthesizer::emitGetArrayLengthExpr(Expr* arrayExpr)
{
    auto expr = m_builder->create<GetArrayLengthExpr>();
    expr->arrayExpr = arrayExpr;
    expr->type = m_builder->getIntType();
    return expr;
}

Expr* ASTSynthesizer::emitMemberExpr(Expr* arrayExpr, Name* name)
{
    auto rs = m_builder->create<MemberExpr>();
    rs->baseExpression = arrayExpr;
    rs->name = name;
    return rs;
}

Expr* ASTSynthesizer::emitAssignExpr(Expr* left, Expr* right)
{
    auto rs = m_builder->create<AssignExpr>();
    rs->left = left;
    rs->right = right;
    return rs;
}

Expr* ASTSynthesizer::emitInvokeExpr(Expr* callee, List<Expr*>&& args)
{
    auto rs = m_builder->create<InvokeExpr>();
    rs->functionExpr = callee;
    rs->arguments = _Move(args);
    return rs;
}

Expr* ASTSynthesizer::emitCtorInvokeExpr(Expr* callee, List<Expr*>&& args)
{
    auto rs = m_builder->create<ExplicitCtorInvokeExpr>();
    rs->functionExpr = callee;
    rs->arguments = _Move(args);
    return rs;
}

Expr* ASTSynthesizer::emitGenericAppExpr(Expr* genericExpr, List<Expr*>&& args)
{
    auto rs = m_builder->create<GenericAppExpr>();
    rs->functionExpr = genericExpr;
    rs->arguments = _Move(args);
    return rs;
}

Expr* ASTSynthesizer::emitMemberExpr(Type* type, Name* name)
{
    auto rs = m_builder->create<StaticMemberExpr>();
    auto typeExpr = m_builder->create<SharedTypeExpr>();
    auto typetype = m_builder->getOrCreate<TypeType>(type);
    typeExpr->type = typetype;
    rs->baseExpression = typeExpr;
    rs->name = name;
    return rs;
}

Expr* ASTSynthesizer::emitIndexExpr(Expr* base, Expr* index)
{
    auto rs = m_builder->create<IndexExpr>();
    rs->baseExpression = base;
    rs->indexExprs.add(index);
    return rs;
}

ExpressionStmt* ASTSynthesizer::emitExprStmt(Expr* expr)
{
    auto rs = m_builder->create<ExpressionStmt>();
    _addStmtToScope(rs);
    rs->expression = expr;
    return rs;
}

ReturnStmt* ASTSynthesizer::emitReturnStmt(Expr* expr)
{
    auto rs = m_builder->create<ReturnStmt>();
    rs->expression = expr;
    _addStmtToScope(rs);
    return rs;
}

DeclStmt* ASTSynthesizer::emitVarDeclStmt(Type* type, Name* name, Expr* initVal)
{
    auto scope = getCurrentScope();
    SLANG_RELEASE_ASSERT(scope.m_parentSeqStmt);
    SLANG_RELEASE_ASSERT(scope.m_scope);
    SLANG_RELEASE_ASSERT(scope.m_scope->containerDecl);
    auto varDecl = m_builder->create<VarDecl>();
    varDecl->type.type = type;
    varDecl->nameAndLoc.name = name;
    varDecl->initExpr = initVal;
    scope.m_scope->containerDecl->addMember(varDecl);
    auto stmt = m_builder->create<DeclStmt>();
    stmt->decl = varDecl;
    _addStmtToScope(stmt);
    return stmt;
}

} // namespace Slang
