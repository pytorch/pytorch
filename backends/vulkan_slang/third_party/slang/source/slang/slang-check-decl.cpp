// slang-check-decl.cpp
#include "slang-ast-modifier.h"
#include "slang-ast-support-types.h"
#include "slang-check-impl.h"

// This file constaints the semantic checking logic and
// related queries for declarations.
//
// Because declarations are the top-level construct
// of the AST (in turn containing all the statements,
// types, and expressions), the declaration-checking
// logic also orchestrates the overall flow and how
// and when things get checked.

#include "slang-ast-forward-declarations.h"
#include "slang-ast-iterator.h"
#include "slang-ast-synthesis.h"
#include "slang-lookup.h"
#include "slang-parser.h"
#include "slang-syntax.h"

#include <limits>

namespace Slang
{
static ConstructorDecl* _getDefaultCtor(StructDecl* structDecl);
static List<ConstructorDecl*> _getCtorList(
    ASTBuilder* m_astBuilder,
    SemanticsVisitor* visitor,
    StructDecl* structDecl,
    ConstructorDecl** defaultCtorOut);
static Expr* constructDefaultInitExprForType(SemanticsVisitor* visitor, VarDeclBase* varDecl);

/// Visitor to transition declarations to `DeclCheckState::CheckedModifiers`
struct SemanticsDeclModifiersVisitor : public SemanticsDeclVisitorBase,
                                       public DeclVisitor<SemanticsDeclModifiersVisitor>
{
    SemanticsDeclModifiersVisitor(SemanticsContext const& outer)
        : SemanticsDeclVisitorBase(outer)
    {
    }

    void visitDeclGroup(DeclGroup*) {}

    void visitDecl(Decl* decl) { checkModifiers(decl); }

    void visitStructDecl(StructDecl* structDecl);
};

struct SemanticsDeclScopeWiringVisitor : public SemanticsDeclVisitorBase,
                                         public DeclVisitor<SemanticsDeclScopeWiringVisitor>
{
    SemanticsDeclScopeWiringVisitor(SemanticsContext const& outer)
        : SemanticsDeclVisitorBase(outer)
    {
    }

    void visitDeclGroup(DeclGroup*) {}

    void visitDecl(Decl*) {}

    void visitUsingDecl(UsingDecl* decl);

    void visitImplementingDecl(ImplementingDecl* decl);

    void visitNamespaceDecl(NamespaceDecl* decl);
};

struct SemanticsDeclAttributesVisitor : public SemanticsDeclVisitorBase,
                                        public DeclVisitor<SemanticsDeclAttributesVisitor>
{
    SemanticsDeclAttributesVisitor(SemanticsContext const& outer)
        : SemanticsDeclVisitorBase(outer)
    {
    }

    void visitDecl(Decl*) {}
    void visitDeclGroup(DeclGroup*) {}

    void visitStructDecl(StructDecl* structDecl);

    void visitFunctionDeclBase(FunctionDeclBase* decl);

    void checkForwardDerivativeOfAttribute(
        FunctionDeclBase* funcDecl,
        ForwardDerivativeOfAttribute* attr);

    void checkBackwardDerivativeOfAttribute(
        FunctionDeclBase* funcDecl,
        BackwardDerivativeOfAttribute* attr);

    void checkPrimalSubstituteOfAttribute(
        FunctionDeclBase* funcDecl,
        PrimalSubstituteOfAttribute* attr);

    void checkVarDeclCommon(VarDeclBase* varDecl);

    void visitVarDecl(VarDecl* varDecl) { checkVarDeclCommon(varDecl); }

    // Synthesize the constructor declaration for a struct during header visit, as we
    // need to have such declaration first such that the overloading resolution can lookup
    // such constructor and complete the initialize list to constructor translation.
    //
    // We will defer the actual implementation of the constructor to the body visit, because
    // we will have full information about each field in the struct during that stage.
    bool _synthesizeCtorSignature(StructDecl* structDecl);
    bool collectInitializableMembers(
        StructDecl* structDecl,
        const DeclVisibility ctorVisibility,
        List<VarDeclBase*>& resultMembers);
};

struct SemanticsDeclHeaderVisitor : public SemanticsDeclVisitorBase,
                                    public DeclVisitor<SemanticsDeclHeaderVisitor>
{
    SemanticsDeclHeaderVisitor(SemanticsContext const& outer)
        : SemanticsDeclVisitorBase(outer)
    {
    }

    void visitDecl(Decl*) {}
    void visitDeclGroup(DeclGroup*) {}

    void checkDerivativeMemberAttributeParent(
        VarDeclBase* varDecl,
        DerivativeMemberAttribute* attr);
    void checkExtensionExternVarAttribute(VarDeclBase* varDecl, ExtensionExternVarModifier* m);
    void checkMeshOutputDecl(VarDeclBase* varDecl);
    void maybeApplyLayoutModifier(VarDeclBase* varDecl);
    void checkVarDeclCommon(VarDeclBase* varDecl);
    void checkPushConstantBufferType(VarDeclBase* varDecl);

    void visitVarDecl(VarDecl* varDecl) { checkVarDeclCommon(varDecl); }

    void visitGlobalGenericValueParamDecl(GlobalGenericValueParamDecl* decl)
    {
        checkVarDeclCommon(decl);
    }

    void visitImportDecl(ImportDecl* decl);

    void visitIncludeDecl(IncludeDecl* decl);

    void visitGenericTypeParamDecl(GenericTypeParamDecl* decl);

    void visitGenericValueParamDecl(GenericValueParamDecl* decl);

    void visitGenericTypeConstraintDecl(GenericTypeConstraintDecl* decl);

    void visitTypeCoercionConstraintDecl(TypeCoercionConstraintDecl* decl);

    void validateGenericConstraintSubType(GenericTypeConstraintDecl* decl, TypeExp type);

    void visitGenericDecl(GenericDecl* genericDecl);

    void visitTypeDefDecl(TypeDefDecl* decl);

    void visitGlobalGenericParamDecl(GlobalGenericParamDecl* decl);

    void visitAssocTypeDecl(AssocTypeDecl* decl);

    void checkDifferentiableCallableCommon(CallableDecl* decl);

    void checkCallableDeclCommon(CallableDecl* decl);

    void visitFuncDecl(FuncDecl* funcDecl);

    void visitParamDecl(ParamDecl* paramDecl);

    void visitConstructorDecl(ConstructorDecl* decl);

    void visitAbstractStorageDeclCommon(ContainerDecl* decl);

    void visitSubscriptDecl(SubscriptDecl* decl);

    void visitPropertyDecl(PropertyDecl* decl);

    void visitStructDecl(StructDecl* decl);

    void visitClassDecl(ClassDecl* decl);

    /// Get the type of the storage accessed by an accessor.
    ///
    /// The type of storage is determined by the parent declaration.
    Type* _getAccessorStorageType(AccessorDecl* decl);

    /// Perform checks common to all types of accessors.
    void _visitAccessorDeclCommon(AccessorDecl* decl);

    void visitAccessorDecl(AccessorDecl* decl);
    void visitSetterDecl(SetterDecl* decl);

    void cloneModifiers(Decl* dest, Decl* src);
    void setFuncTypeIntoRequirementDecl(CallableDecl* decl, FuncType* funcType);
};

struct SemanticsDeclRedeclarationVisitor : public SemanticsDeclVisitorBase,
                                           public DeclVisitor<SemanticsDeclRedeclarationVisitor>
{
    SemanticsDeclRedeclarationVisitor(SemanticsContext const& outer)
        : SemanticsDeclVisitorBase(outer)
    {
    }

    void visitDecl(Decl*) {}
    void visitDeclGroup(DeclGroup*) {}

#define CASE(TYPE) \
    void visit##TYPE(TYPE* decl) { checkForRedeclaration(decl); }

    CASE(EnumCaseDecl)
    CASE(FuncDecl)
    CASE(VarDeclBase)
    CASE(SimpleTypeDecl)
    CASE(AggTypeDecl)

#undef CASE
};

struct SemanticsDeclBasesVisitor : public SemanticsDeclVisitorBase,
                                   public DeclVisitor<SemanticsDeclBasesVisitor>
{
    SemanticsDeclBasesVisitor(SemanticsContext const& outer)
        : SemanticsDeclVisitorBase(outer)
    {
    }

    void visitDecl(Decl*) {}
    void visitDeclGroup(DeclGroup*) {}

    void visitInheritanceDecl(InheritanceDecl* inheritanceDecl);

    void visitThisTypeConstraintDecl(ThisTypeConstraintDecl* thisTypeConstraintDecl);

    /// Validate that `decl` isn't illegally inheriting from a type in another module.
    ///
    /// This call checks a single `inheritanceDecl` to make sure that it either
    ///     * names a base type from the same module as `decl`, or
    ///     * names a type that allows cross-module inheritance
    void _validateCrossModuleInheritance(AggTypeDeclBase* decl, InheritanceDecl* inheritanceDecl);

    void visitInterfaceDecl(InterfaceDecl* decl);

    void visitStructDecl(StructDecl* decl);

    void visitClassDecl(ClassDecl* decl);

    void visitEnumDecl(EnumDecl* decl);

    /// Validate that the target type of an extension `decl` is valid.
    void _validateExtensionDeclTargetType(ExtensionDecl* decl);
    void _validateExtensionDeclMembers(ExtensionDecl* decl);

    void visitExtensionDecl(ExtensionDecl* decl);
};

struct SemanticsDeclTypeResolutionVisitor : public SemanticsDeclVisitorBase,
                                            public DeclVisitor<SemanticsDeclTypeResolutionVisitor>
{
    SemanticsDeclTypeResolutionVisitor(SemanticsContext const& outer)
        : SemanticsDeclVisitorBase(outer)
    {
    }

    void visitDecl(Decl*) {}
    void visitDeclGroup(DeclGroup*) {}

    void visitTypeExp(TypeExp& exp) { exp.type = resolveType(exp.type); }

    void visitVarDeclBase(VarDeclBase* varDecl) { visitTypeExp(varDecl->type); }

    void visitGenericTypeConstraintDecl(GenericTypeConstraintDecl* decl)
    {
        visitTypeExp(decl->sup);
    }

    void visitTypeDefDecl(TypeDefDecl* decl) { visitTypeExp(decl->type); }

    void visitGenericTypeParamDecl(GenericTypeParamDecl* paramDecl)
    {
        visitTypeExp(paramDecl->initType);
    }

    void visitInheritanceDecl(InheritanceDecl* inheritanceDecl)
    {
        visitTypeExp(inheritanceDecl->base);
    }

    void visitCallableDecl(CallableDecl* decl)
    {
        for (auto paramDecl : decl->getMembersOfType<ParamDecl>())
            visitTypeExp(paramDecl->type);

        visitTypeExp(decl->returnType);
        visitTypeExp(decl->errorType);
    }

    void visitPropertyDecl(PropertyDecl* decl) { visitTypeExp(decl->type); }
};

struct SemanticsDeclBodyVisitor : public SemanticsDeclVisitorBase,
                                  public DeclVisitor<SemanticsDeclBodyVisitor>
{
    SemanticsDeclBodyVisitor(SemanticsContext const& outer)
        : SemanticsDeclVisitorBase(outer)
    {
    }

    void visitDecl(Decl*) {}
    void visitDeclGroup(DeclGroup*) {}

    void checkVarDeclCommon(VarDeclBase* varDecl);

    void visitVarDecl(VarDecl* varDecl) { checkVarDeclCommon(varDecl); }

    void visitGenericValueParamDecl(GenericValueParamDecl* genValDecl)
    {
        checkVarDeclCommon(genValDecl);
    }

    void visitGlobalGenericValueParamDecl(GlobalGenericValueParamDecl* decl)
    {
        checkVarDeclCommon(decl);
    }

    void visitEnumCaseDecl(EnumCaseDecl* decl);

    void visitEnumDecl(EnumDecl* decl);

    void visitFunctionDeclBase(FunctionDeclBase* funcDecl);

    void visitParamDecl(ParamDecl* paramDecl);

    void visitAggTypeDecl(AggTypeDecl* aggTypeDecl);

    SemanticsContext registerDifferentiableTypesForFunc(FunctionDeclBase* funcDecl);

private:
    struct DeclAndCtorInfo
    {
        StructDecl* parent = nullptr;
        ConstructorDecl* defaultCtor = nullptr;
        List<ConstructorDecl*> ctorList;
        Type* type = nullptr;
        DeclAndCtorInfo() {}
        DeclAndCtorInfo(
            ASTBuilder* m_astBuilder,
            SemanticsVisitor* visitor,
            StructDecl* inParent,
            Type* inType,
            const bool getOnlyDefault)
        {
            parent = inParent;
            type = inType;
            if (getOnlyDefault)
                defaultCtor = _getDefaultCtor(parent);
            else
                ctorList = _getCtorList(m_astBuilder, visitor, parent, &defaultCtor);
        }
    };

    void synthesizeCtorBody(
        DeclAndCtorInfo& structDeclInfo,
        List<DeclAndCtorInfo>& inheritanceDefaultCtorList,
        StructDecl* structDecl);
    void synthesizeCtorBodyForBases(
        ConstructorDecl* ctor,
        List<DeclAndCtorInfo>& inheritanceDefaultCtorList,
        ThisExpr* thisExpr,
        SeqStmt* seqStmtChild,
        bool isMemberInitCtor,
        Index& paramIndex);

    void synthesizeCtorBodyForMember(
        ConstructorDecl* ctor,
        Decl* member,
        ThisExpr* thisExpr,
        Dictionary<Decl*, Expr*>& cachedDeclToCheckedVar,
        SeqStmt* seqStmtChild,
        bool isMemberInitCtor,
        Index& paramIndex);

    MemberExpr* createMemberExpr(ThisExpr* thisExpr, Scope* scope, Decl* member);
    Expr* createCtorParamExpr(ConstructorDecl* ctor, Index paramIndex);
};

template<typename VisitorType>
struct SemanticsDeclReferenceVisitor : public SemanticsDeclVisitorBase,
                                       public StmtVisitor<VisitorType>,
                                       public ExprVisitor<VisitorType>,
                                       public ValVisitor<VisitorType>,
                                       public DeclVisitor<VisitorType>
{
    SemanticsDeclReferenceVisitor(SemanticsContext const& outer)
        : SemanticsDeclVisitorBase(outer)
    {
    }

    List<SourceLoc> sourceLocStack;

    struct PushSourceLocRAII
    {
        List<SourceLoc>& stack;
        bool shouldPop = false;
        PushSourceLocRAII(List<SourceLoc>& sourceLocStack, SourceLoc loc)
            : stack(sourceLocStack)
        {
            if (loc.isValid())
            {
                stack.add(loc);
                shouldPop = true;
            }
        }
        ~PushSourceLocRAII()
        {
            if (shouldPop)
            {
                stack.removeLast();
            }
        }
    };

    virtual void processReferencedDecl(Decl* decl) = 0;

    virtual void processDeclModifiers(Decl* decl, SourceLoc refLoc) = 0;

    void dispatchIfNotNull(Stmt* stmt)
    {
        if (!stmt)
            return;
        PushSourceLocRAII sourceLocRAII(sourceLocStack, stmt->loc);
        return StmtVisitor<VisitorType>::dispatch(stmt);
    }
    void dispatchIfNotNull(Expr* expr)
    {
        if (!expr)
            return;
        PushSourceLocRAII sourceLocRAII(sourceLocStack, expr->loc);
        return ExprVisitor<VisitorType>::dispatch(expr);
    }
    void dispatchIfNotNull(Val* val)
    {
        if (!val)
            return;
        return ValVisitor<VisitorType>::dispatch(val);
    }
    void dispatchIfNotNull(DeclBase* val)
    {
        if (!val)
            return;
        return DeclVisitor<VisitorType>::dispatch(val);
    }
    // Expr Visitor
    void visitExpr(Expr*) {}
    void visitIndexExpr(IndexExpr* subscriptExpr)
    {
        for (auto arg : subscriptExpr->indexExprs)
            dispatchIfNotNull(arg);
        dispatchIfNotNull(subscriptExpr->baseExpression);
    }

    void visitParenExpr(ParenExpr* expr) { dispatchIfNotNull(expr->base); }

    void visitAssignExpr(AssignExpr* expr)
    {
        dispatchIfNotNull(expr->left);
        dispatchIfNotNull(expr->right);
    }

    void visitGenericAppExpr(GenericAppExpr* genericAppExpr)
    {
        dispatchIfNotNull(genericAppExpr->functionExpr);
        for (auto arg : genericAppExpr->arguments)
            dispatchIfNotNull(arg);
    }

    void visitSharedTypeExpr(SharedTypeExpr* expr) { dispatchIfNotNull(expr->base.exp); }

    void visitInvokeExpr(InvokeExpr* expr)
    {
        dispatchIfNotNull(expr->functionExpr);
        for (auto arg : expr->arguments)
            dispatchIfNotNull(arg);
    }

    void visitTypeCastExpr(TypeCastExpr* expr)
    {
        dispatchIfNotNull(expr->functionExpr);
        for (auto arg : expr->arguments)
            dispatchIfNotNull(arg);
    }

    void visitDerefExpr(DerefExpr* expr) { dispatchIfNotNull(expr->base); }
    void visitMatrixSwizzleExpr(MatrixSwizzleExpr* expr) { dispatchIfNotNull(expr->base); }
    void visitSwizzleExpr(SwizzleExpr* expr) { dispatchIfNotNull(expr->base); }
    void visitOverloadedExpr(OverloadedExpr*) { return; }
    void visitOverloadedExpr2(OverloadedExpr2*) { return; }
    void visitAggTypeCtorExpr(AggTypeCtorExpr*) { return; }
    void visitCastToSuperTypeExpr(CastToSuperTypeExpr* expr) { dispatchIfNotNull(expr->valueArg); }
    void visitModifierCastExpr(ModifierCastExpr* expr) { dispatchIfNotNull(expr->valueArg); }
    void visitLetExpr(LetExpr* expr) { dispatchIfNotNull(expr->body); }
    void visitExtractExistentialValueExpr(ExtractExistentialValueExpr* expr)
    {
        dispatchIfNotNull(expr->declRef.declRefBase);
    }

    void visitDeclRefExpr(DeclRefExpr* expr)
    {
        dispatchIfNotNull(expr->type.type);
        dispatchIfNotNull(expr->declRef.declRefBase);

        // Pass down the callee location
        processDeclModifiers(expr->declRef.getDecl(), expr->loc);
    }
    void visitStaticMemberExpr(StaticMemberExpr* expr)
    {
        dispatchIfNotNull(expr->declRef.declRefBase);
    }
    void visitInitializerListExpr(InitializerListExpr* expr)
    {
        for (auto arg : expr->args)
        {
            dispatchIfNotNull(arg);
        }
    }

    void visitThisExpr(ThisExpr*) { return; }

    void visitThisTypeExpr(ThisTypeExpr*) { return; }
    void visitAndTypeExpr(AndTypeExpr* expr)
    {
        dispatchIfNotNull(expr->left.type);
        dispatchIfNotNull(expr->right.type);
    }
    void visitPointerTypeExpr(PointerTypeExpr* expr) { dispatchIfNotNull(expr->base.type); }
    void visitAsTypeExpr(AsTypeExpr* expr)
    {
        dispatchIfNotNull(expr->value);
        dispatchIfNotNull(expr->witnessArg);
    }
    void visitIsTypeExpr(IsTypeExpr* expr)
    {
        dispatchIfNotNull(expr->value);
        dispatchIfNotNull(expr->witnessArg);
    }
    void visitMakeOptionalExpr(MakeOptionalExpr* expr)
    {
        dispatchIfNotNull(expr->value);
        dispatchIfNotNull(expr->typeExpr);
    }
    void visitPartiallyAppliedGenericExpr(PartiallyAppliedGenericExpr*) { return; }
    void visitSPIRVAsmExpr(SPIRVAsmExpr*) { return; }
    void visitModifiedTypeExpr(ModifiedTypeExpr* expr) { dispatchIfNotNull(expr->base.type); }
    void visitFuncTypeExpr(FuncTypeExpr* expr)
    {
        for (const auto& t : expr->parameters)
        {
            dispatchIfNotNull(t.type);
        }
        dispatchIfNotNull(expr->result.type);
    }
    void visitTupleTypeExpr(TupleTypeExpr* expr)
    {
        for (auto t : expr->members)
        {
            dispatchIfNotNull(t.type);
        }
    }
    void visitTryExpr(TryExpr* expr) { dispatchIfNotNull(expr->base); }
    void visitHigherOrderInvokeExpr(HigherOrderInvokeExpr* expr)
    {
        dispatchIfNotNull(expr->baseFunction);
    }
    void visitTreatAsDifferentiableExpr(TreatAsDifferentiableExpr* expr)
    {
        dispatchIfNotNull(expr->innerExpr);
    }

    // Stmt Visitor

    void visitDeclStmt(DeclStmt* stmt) { dispatchIfNotNull(stmt->decl); }

    void visitBlockStmt(BlockStmt* stmt) { dispatchIfNotNull(stmt->body); }

    void visitSeqStmt(SeqStmt* seqStmt)
    {
        for (auto stmt : seqStmt->stmts)
            dispatchIfNotNull(stmt);
    }

    void visitLabelStmt(LabelStmt* stmt) { dispatchIfNotNull(stmt->innerStmt); }

    void visitBreakStmt(BreakStmt*) { return; }

    void visitContinueStmt(ContinueStmt*) { return; }

    void visitDoWhileStmt(DoWhileStmt* stmt)
    {
        dispatchIfNotNull(stmt->predicate);
        dispatchIfNotNull(stmt->statement);
    }

    void visitForStmt(ForStmt* stmt)
    {
        dispatchIfNotNull(stmt->initialStatement);
        dispatchIfNotNull(stmt->predicateExpression);
        dispatchIfNotNull(stmt->sideEffectExpression);
        dispatchIfNotNull(stmt->statement);
    }

    void visitCompileTimeForStmt(CompileTimeForStmt* stmt)
    {
        dispatchIfNotNull(stmt->rangeBeginExpr);
        dispatchIfNotNull(stmt->rangeEndExpr);
        dispatchIfNotNull(stmt->body);
    }

    void visitSwitchStmt(SwitchStmt* stmt)
    {
        dispatchIfNotNull(stmt->condition);
        dispatchIfNotNull(stmt->body);
    }

    void visitCaseStmt(CaseStmt* stmt) { dispatchIfNotNull(stmt->expr); }

    void visitTargetSwitchStmt(TargetSwitchStmt* stmt)
    {
        for (auto targetCase : stmt->targetCases)
            dispatchIfNotNull(targetCase);
    }

    void visitTargetCaseStmt(TargetCaseStmt* stmt) { dispatchIfNotNull(stmt->body); }

    void visitIntrinsicAsmStmt(IntrinsicAsmStmt*) { return; }

    void visitDefaultStmt(DefaultStmt*) { return; }

    void visitIfStmt(IfStmt* stmt)
    {
        dispatchIfNotNull(stmt->predicate);
        dispatchIfNotNull(stmt->positiveStatement);
        dispatchIfNotNull(stmt->negativeStatement);
    }

    void visitUnparsedStmt(UnparsedStmt*) { return; }

    void visitEmptyStmt(EmptyStmt*) { return; }

    void visitDiscardStmt(DiscardStmt*) { return; }

    void visitReturnStmt(ReturnStmt* stmt) { dispatchIfNotNull(stmt->expression); }

    void visitDeferStmt(DeferStmt* stmt) { dispatchIfNotNull(stmt->statement); }

    void visitWhileStmt(WhileStmt* stmt)
    {
        dispatchIfNotNull(stmt->predicate);
        dispatchIfNotNull(stmt->statement);
    }

    void visitGpuForeachStmt(GpuForeachStmt*) { return; }

    void visitExpressionStmt(ExpressionStmt* stmt) { dispatchIfNotNull(stmt->expression); }

    // Val Visitor

    void visitDirectDeclRef(DirectDeclRef* declRef)
    {
        // If we have already visited, return.
        // Otherwise add it to visited set.
        if (!visitedVals.add(declRef))
            return;

        processReferencedDecl(declRef->getDecl());
    }

    void visitVal(Val* val)
    {
        // If we have already visited, return.
        // Otherwise add it to visited set.
        if (!visitedVals.add(val))
            return;

        for (Index i = 0; i < val->getOperandCount(); i++)
        {
            auto& operand = val->m_operands[i];
            switch (operand.kind)
            {
            case ValNodeOperandKind::ValNode:
                dispatchIfNotNull(val->getOperand(i));
                break;
            default:
                break;
            }
        }
        return;
    }

    HashSet<Val*> visitedVals;

    // Decl visitor
    void visitDeclBase(DeclBase*) {}

    void visitContainerDecl(ContainerDecl* decl)
    {
        for (auto m : decl->members)
        {
            dispatchIfNotNull(m);
        }
    }

    void visitFunctionDeclBase(FunctionDeclBase* decl)
    {
        visitContainerDecl(decl);
        dispatchIfNotNull(decl->body);
    }

    void visitVarDeclBase(VarDeclBase* varDecl)
    {
        dispatchIfNotNull(varDecl->type.type);
        dispatchIfNotNull(varDecl->initExpr);
    }
};

struct SemanticsDeclCapabilityVisitor : public SemanticsDeclVisitorBase,
                                        public DeclVisitor<SemanticsDeclCapabilityVisitor>
{
    CapabilitySet m_anyPlatfromCapabilitySet;

    SemanticsDeclCapabilityVisitor(SemanticsContext const& outer)
        : SemanticsDeclVisitorBase(outer)
    {
    }

    CapabilitySet& getAnyPlatformCapabilitySet()
    {
        if (m_anyPlatfromCapabilitySet.isEmpty())
        {
            m_anyPlatfromCapabilitySet = CapabilitySet(CapabilityName::any_target);
        }
        return m_anyPlatfromCapabilitySet;
    }

    CapabilitySet getDeclaredCapabilitySet(Decl* decl);


    void visitDecl(Decl*) {}
    void visitDeclGroup(DeclGroup*) {}
    void checkVarDeclCommon(VarDeclBase* varDecl);
    void visitAggTypeDeclBase(AggTypeDeclBase* decl);
    void visitNamespaceDeclBase(NamespaceDeclBase* decl);

    void visitVarDecl(VarDecl* varDecl) { checkVarDeclCommon(varDecl); }

    void visitParamDecl(ParamDecl* paramDecl) { checkVarDeclCommon(paramDecl); }

    void visitFunctionDeclBase(FunctionDeclBase* funcDecl);

    void visitInheritanceDecl(InheritanceDecl* inheritanceDecl);

    void diagnoseUndeclaredCapability(
        Decl* decl,
        const DiagnosticInfo& diagnosticInfo,
        const CapabilityAtomSet& failedAtomsInsideAvailableSet);
};


/// Should the given `decl` nested in `parentDecl` be treated as a static rather than instance
/// declaration?
bool isEffectivelyStatic(Decl* decl, ContainerDecl* parentDecl)
{
    // Things at the global scope are always "members" of their module.
    //
    if (as<NamespaceDeclBase>(parentDecl))
        return false;
    if (as<FileDecl>(parentDecl))
        return false;

    // Anything explicitly marked `static` and not at module scope
    // counts as a static rather than instance declaration.
    //
    if (decl->hasModifier<HLSLStaticModifier>())
        return true;

    // Next we need to deal with cases where a declaration is
    // effectively `static` even if the language doesn't make
    // the user say so. Most languages make the default assumption
    // that nested types are `static` even if they don't say
    // so (Java is an exception here, perhaps due to some
    // influence from the Scandanavian OOP tradition of Beta/gbeta).
    //
    if (as<AggTypeDecl>(decl))
        return true;
    if (as<SimpleTypeDecl>(decl))
        return true;

    // Initializer/constructor declarations are effectively `static`
    // in Slang. They behave like functions that return an instance
    // of the enclosing type, rather than as functions that are
    // called on a pre-existing value.
    //
    if (as<ConstructorDecl>(decl))
        return true;

    if (as<EnumCaseDecl>(decl))
        return true;

    // Things nested inside functions may have dependencies
    // on values from the enclosing scope, but this needs to
    // be dealt with via "capture" so they are also effectively
    // `static`
    //
    if (as<FunctionDeclBase>(parentDecl))
        return true;

    // Type constraint declarations are used in member-reference
    // context as a form of casting operation, so we treat them
    // as if they are instance members. This is a bit of a hack,
    // but it achieves the result we want until we have an
    // explicit representation of up-cast operations in the
    // AST.
    //
    if (as<TypeConstraintDecl>(decl))
        return false;

    return false;
}

bool isEffectivelyStatic(Decl* decl)
{
    // For the purposes of an ordinary declaration, when determining if
    // it is static or per-instance, the "parent" declaration we really
    // care about is the next outer non-generic declaration.
    //
    // TODO: This idiom of getting the "next outer non-generic declaration"
    // comes up just enough that we should probably have a convenience
    // function for it.

    auto parentDecl = decl->parentDecl;
    if (auto genericDecl = as<GenericDecl>(parentDecl))
        parentDecl = genericDecl->parentDecl;

    return isEffectivelyStatic(decl, parentDecl);
}

bool isGlobalDecl(Decl* decl)
{
    if (!decl)
        return false;
    auto parentDecl = decl->parentDecl;
    if (auto genericDecl = as<GenericDecl>(parentDecl))
        parentDecl = genericDecl->parentDecl;
    return as<NamespaceDeclBase>(parentDecl) != nullptr || as<FileDecl>(parentDecl) != nullptr;
}

bool isUnsafeForceInlineFunc(FunctionDeclBase* funcDecl)
{
    return funcDecl->hasModifier<UnsafeForceInlineEarlyAttribute>();
}

/// Is `decl` a global shader parameter declaration?
bool isGlobalShaderParameter(VarDeclBase* decl)
{
    // If it's an *actual* global it is not a global shader parameter
    if (decl->hasModifier<ActualGlobalModifier>())
    {
        return false;
    }

    // A global shader parameter must be declared at global or namespace
    // scope, so that it has a single definition across the module.
    //
    if (!isGlobalDecl(decl))
        return false;

    for (auto modifier : decl->modifiers)
    {
        // A global variable marked `static` indicates a traditional
        // global variable (albeit one that is implicitly local to
        // the translation unit)
        //
        if (as<HLSLStaticModifier>(modifier))
            return false;

        // While not normally allowed, out variables are not constant
        // parameters, this can happen for example in GLSL mode
        if (as<OutModifier>(modifier))
            return false;
        if (as<InModifier>(modifier))
            return false;

        // The `groupshared` modifier indicates that a variable cannot
        // be a shader parameters, but is instead transient storage
        // allocated for the duration of a thread-group's execution.
        //
        if (as<HLSLGroupSharedModifier>(modifier))
            return false;
    }

    return true;
}

[[maybe_unused]] static bool _isUncheckedLocalVar(const Decl* decl)
{
    auto checkStateExt = decl->checkState;
    auto isUnchecked =
        checkStateExt.getState() == DeclCheckState::Unchecked || checkStateExt.isBeingChecked();
    return isUnchecked && isLocalVar(decl);
}

// Get the type to use when referencing a declaration
QualType getTypeForDeclRef(
    ASTBuilder* astBuilder,
    SemanticsVisitor* sema,
    DiagnosticSink* sink,
    DeclRef<Decl> declRef,
    Type** outTypeResult,
    SourceLoc loc)
{
    if (sema)
    {
        // If this is a local variable which hasn't been checked yet then
        // it's probably a declare-after-use which has incorrectly got
        // through declref resolution.
        SLANG_ASSERT(!_isUncheckedLocalVar(declRef.getDecl()));

        // Once we've ruled out the case of referencing a local declaration
        // before it has been checked, we will go ahead and ensure that
        // semantic checking has been performed on the chosen declaration,
        // at least up to the point where we can query its type.
        //
        sema->ensureDecl(declRef, DeclCheckState::CanUseTypeOfValueDecl);
    }

    // We need to insert an appropriate type for the expression, based on
    // what we found.
    if (auto varDeclRef = declRef.as<VarDeclBase>())
    {
        QualType qualType;
        qualType.type = getType(astBuilder, varDeclRef);

        bool isLValue = true;
        if (varDeclRef.getDecl()->findModifier<ConstModifier>())
            isLValue = false;

        // Global-scope shader parameters should not be writable,
        // since they are effectively program inputs.
        //
        // TODO: We could eventually treat a mutable global shader
        // parameter as a shorthand for an immutable parameter and
        // a global variable that gets initialized from that parameter,
        // but in order to do so we'd need to support global variables
        // with resource types better in the back-end.
        //
        if (isGlobalShaderParameter(varDeclRef.getDecl()))
            isLValue = false;

        // Variables declared with `let` are always immutable.
        if (varDeclRef.is<LetDecl>())
            isLValue = false;

        // Generic value parameters are always immutable
        if (varDeclRef.is<GenericValueParamDecl>())
            isLValue = false;

        // Function parameters declared in the "modern" style
        // are immutable unless they have an `out` or `inout` modifier.
        if (varDeclRef.is<ModernParamDecl>())
        {
            // Note: the `inout` modifier AST class inherits from
            // the class for the `out` modifier so that we can
            // make simple checks like this.
            //
            if (!varDeclRef.getDecl()->hasModifier<OutModifier>())
            {
                isLValue = false;
            }
        }

        // Ensures child of struct is set read-only or not
        bool isWriteOnly = false;
        if (auto collection = varDeclRef.getDecl()->findModifier<MemoryQualifierSetModifier>())
        {
            if (collection->getMemoryQualifierBit() & MemoryQualifierSetModifier::Flags::kReadOnly)
            {
                isLValue = false;
                qualType.hasReadOnlyOnTarget = true;
            }
            if (collection->getMemoryQualifierBit() & MemoryQualifierSetModifier::Flags::kWriteOnly)
                isWriteOnly = true;
        }

        qualType.isLeftValue = isLValue;
        qualType.isWriteOnly = isWriteOnly;
        return qualType;
    }
    else if (auto propertyDeclRef = declRef.as<PropertyDecl>())
    {
        // Access to a declared `property` is similar to
        // access to a variable/field, except that it
        // is mediated through accessors (getters, seters, etc.).

        QualType qualType;
        qualType.type = getType(astBuilder, propertyDeclRef);

        bool isLValue = false;

        // If the property has any declared accessors that
        // can be used to set the property, then the resulting
        // expression behaves as an l-value.
        //
        if (propertyDeclRef.getDecl()->getMembersOfType<SetterDecl>().isNonEmpty())
            isLValue = true;
        if (propertyDeclRef.getDecl()->getMembersOfType<RefAccessorDecl>().isNonEmpty())
            isLValue = true;

        qualType.isLeftValue = isLValue;
        return qualType;
    }
    else if (auto enumCaseDeclRef = declRef.as<EnumCaseDecl>())
    {
        sema->ensureDecl(declRef.declRefBase, DeclCheckState::DefinitionChecked);
        QualType qualType;
        qualType.type = getType(astBuilder, enumCaseDeclRef);
        qualType.isLeftValue = false;
        return qualType;
    }
    else if (auto typeAliasDeclRef = declRef.as<TypeDefDecl>())
    {
        auto type = getNamedType(astBuilder, typeAliasDeclRef);
        *outTypeResult = type;
        return QualType(astBuilder->getTypeType(type));
    }
    else if (auto aggTypeDeclRef = declRef.as<AggTypeDecl>())
    {
        auto type = DeclRefType::create(astBuilder, aggTypeDeclRef);
        *outTypeResult = type;
        return QualType(astBuilder->getTypeType(type));
    }
    else if (auto simpleTypeDeclRef = declRef.as<SimpleTypeDecl>())
    {
        auto type = DeclRefType::create(astBuilder, simpleTypeDeclRef);
        *outTypeResult = type;
        return QualType(astBuilder->getTypeType(type));
    }
    else if (auto genericDeclRef = declRef.as<GenericDecl>())
    {
        auto type = getGenericDeclRefType(astBuilder, genericDeclRef);
        *outTypeResult = type;
        return QualType(astBuilder->getTypeType(type));
    }
    else if (auto funcDeclRef = declRef.as<CallableDecl>())
    {
        auto type = getFuncType(astBuilder, funcDeclRef);
        return QualType(type);
    }
    else if (auto constraintDeclRef = declRef.as<TypeConstraintDecl>())
    {
        // When we access a constraint or an inheritance decl (as a member),
        // we are conceptually performing a "cast" to the given super-type,
        // with the declaration showing that such a cast is legal.
        auto type = getSup(astBuilder, constraintDeclRef);
        return QualType(type);
    }
    else if (auto namespaceDeclRef = declRef.as<NamespaceDeclBase>())
    {
        auto type = getNamespaceType(astBuilder, namespaceDeclRef);
        return QualType(type);
    }
    if (sink)
    {
        // The compiler is trying to form a reference to a declaration
        // that doesn't appear to be usable as an expression or type.
        //
        // In practice, this arises when user code has an undefined-identifier
        // error, but the name that was undefined in context also matches
        // a contextual keyword. Rather than confuse the user with the
        // details of contextual keywords in the compiler, we will diagnose
        // this as an undefined identifier.
        //
        // TODO: This code could break if we ever go down this path with
        // an identifier that doesn't have a name.
        //
        sink->diagnose(loc, Diagnostics::undefinedIdentifier2, declRef.getName());
    }
    return QualType(astBuilder->getErrorType());
}

QualType getTypeForDeclRef(ASTBuilder* astBuilder, DeclRef<Decl> declRef, SourceLoc loc)
{
    Type* typeResult = nullptr;
    return getTypeForDeclRef(astBuilder, nullptr, nullptr, declRef, &typeResult, loc);
}

DeclRef<ExtensionDecl> applyExtensionToType(
    SemanticsVisitor* semantics,
    ExtensionDecl* extDecl,
    Type* type,
    Dictionary<Type*, SubtypeWitness*>* additionalSubtypeWitness)
{
    if (!semantics)
        return DeclRef<ExtensionDecl>();

    return semantics->applyExtensionToType(extDecl, type, additionalSubtypeWitness);
}

bool SemanticsVisitor::isDeclUsableAsStaticMember(Decl* decl)
{
    if (m_allowStaticReferenceToNonStaticMember)
        return true;

    if (auto genericDecl = as<GenericDecl>(decl))
        decl = genericDecl->inner;

    if (decl->hasModifier<HLSLStaticModifier>())
        return true;

    if (as<ConstructorDecl>(decl))
        return true;

    if (as<EnumCaseDecl>(decl))
        return true;

    if (as<AggTypeDeclBase>(decl))
        return true;

    if (as<SimpleTypeDecl>(decl))
        return true;

    if (as<TypeConstraintDecl>(decl))
        return true;

    return false;
}

bool SemanticsVisitor::isUsableAsStaticMember(LookupResultItem const& item)
{
    if (m_allowStaticReferenceToNonStaticMember)
        return true;

    // There's a bit of a gotcha here, because a lookup result
    // item might include "breadcrumbs" that indicate more steps
    // along the lookup path. As a result it isn't always
    // valid to just check whether the final decl is usable
    // as a static member, because it might not even be a
    // member of the thing we are trying to work with.
    //

    Decl* decl = item.declRef.getDecl();
    for (auto bb = item.breadcrumbs; bb; bb = bb->next)
    {
        switch (bb->kind)
        {
        // In case lookup went through a `__transparent` member,
        // we are interested in the static-ness of that transparent
        // member, and *not* the static-ness of whatever was inside
        // of it.
        //
        // TODO: This would need some work if we ever had
        // transparent *type* members.
        //
        case LookupResultItem::Breadcrumb::Kind::Member:
            decl = bb->declRef.getDecl();
            break;

            // TODO: Are there any other cases that need special-case
            // handling here?

        default:
            break;
        }
    }

    // Okay, we've found the declaration we should actually
    // be checking, so lets validate that.

    return isDeclUsableAsStaticMember(decl);
}

/// Dispatch an appropriate visitor to check `decl` up to state `state`
///
/// The current state of `decl` must be `state-1`.
/// This call does *not* handle updating the state of `decl`; the
/// caller takes responsibility for doing so.
///
static void _dispatchDeclCheckingVisitor(
    Decl* decl,
    DeclCheckState state,
    SemanticsContext& shared);

// Make sure a declaration has been checked, so we can refer to it.
// Note that this may lead to us recursively invoking checking,
// so this may not be the best way to handle things.
void SemanticsVisitor::ensureDecl(Decl* decl, DeclCheckState state, SemanticsContext* baseContext)
{
    // If the `decl` has already been checked up to or beyond `state`
    // then there is nothing for us to do.
    //
    if (decl->isChecked(state))
        return;

    // Is the declaration already being checked, somewhere up the
    // call stack from us?
    //
    if (decl->checkState.isBeingChecked())
    {
        // We tried to reference the same declaration while checking it!
        //
        // TODO: we should ideally be tracking a "chain" of declarations
        // being checked on the stack, so that we can report the full
        // chain that leads from this declaration back to itself.
        //
        getSink()->diagnose(decl, Diagnostics::cyclicReference, decl);
        return;
    }

    // If we should skip the checking, return now.
    // A common case to skip checking is for the function bodies when we are in
    // the language server. In that case we only care about the function bodies in a
    // specific module and can skip checking the reference modules until they
    // are being opened/edited later.
    if (shouldSkipChecking(decl, state))
    {
        decl->setCheckState(state);
        return;
    }

    // Set the flag that indicates we are checking this declaration,
    // so that the cycle check above will catch us before we go
    // into any infinite loops.
    //
    decl->checkState.setIsBeingChecked(true);

    // Our task is to bring the `decl` up to `state` which may be
    // one or more steps ahead of where it currently is. We can
    // invoke a visitor designed to bring a declaration from state
    // N to state N+1, and in general we might need multiple such
    // passes to get `decl` to where we need it.
    //
    // The coding of this loop is somewhat defensive to deal
    // with special cases that will be described along the way.
    //
    auto outerScope = getScope(decl);
    for (;;)
    {
        // The first thing is to check what state the decl is
        // currently in at the start of this loop iteration,
        // and to bail out if it has been checked up to
        // (or beyond) our target state.
        //
        auto currentState = decl->checkState.getState();
        if (currentState >= state)
            break;

        // Because our visitors are only designed to go from state
        // N to N+1 in general, we will aspire to transition to
        // a state that is one greater than `currentState`.
        //
        auto nextState = DeclCheckState(Int(currentState) + 1);

        // We now dispatch an appropriate visitor based on `nextState`.
        //
        // Note that we always dispatch the visitor in a "fresh" semantic-checking
        // context, so that the state at the point where a declaration is *referenced*
        // cannot affect the state in which the declaration is *checked*.
        //
        SemanticsContext subContext =
            baseContext ? SemanticsContext(*baseContext) : SemanticsContext(getShared());
        if (outerScope)
            subContext = subContext.withOuterScope(outerScope);
        _dispatchDeclCheckingVisitor(decl, nextState, subContext);

        // In the common case, the visitor will have done the necessary
        // checking, but will *not* have updated the `checkState` on
        // `decl`. In that case we will do the update here, to save
        // us the complication of having to deal with state update in
        // every single visitor method.
        //
        // However, sometimes a visitor *will* want to manually update
        // the state of a declaration, and it may actually update it
        // *past* the `nextState` we asked for (or even past the
        // eventual target `state`). In those cases we don't want to
        // accidentally set the state of `decl` to something lower
        // than what has actually been checked, so we test for
        // such cases here.
        //
        if (nextState > decl->checkState.getState())
        {
            decl->setCheckState(nextState);
        }
    }

    // Once we are done here, the state of `decl` should have
    // been upgraded to (at least) `state`.
    //
    SLANG_ASSERT(decl->isChecked(state));

    // Now that we are done checking `decl` we need to restore
    // its "is being checked" flag so that we don't generate
    // errors the next time somebody calls `ensureDecl()` on it.
    //
    decl->checkState.setIsBeingChecked(false);
}

/// Recursively ensure the tree of declarations under `decl` is in `state`.
///
/// This function does *not* handle declarations nested in function bodies
/// because those cannot be meaningfully checked outside of the context
/// of their surrounding statement(s).
///
void SemanticsVisitor::ensureAllDeclsRec(Decl* decl, DeclCheckState state)
{
    // Ensure `decl` itself first.
    ensureDecl(decl, state);

    // If `decl` is a container, then we want to ensure its children.
    if (auto containerDecl = as<ContainerDecl>(decl))
    {
        // NOTE! We purposefully do not iterate with the for(auto childDecl :
        // containerDecl->members) here, because the visitor may add to `members` whilst iteration
        // takes place, invalidating the iterator and likely a crash.
        //
        // Accessing the members via index side steps the issue.
        const auto& members = containerDecl->members;
        for (Index i = 0; i < members.getCount(); ++i)
        {
            Decl* childDecl = members[i];

            // As an exception, if any of the child is a `ScopeDecl`,
            // then that indicates that it represents a scope for local
            // declarations under a statement (e.g., in a function body),
            // and we don't want to check such local declarations here.
            //

            if (as<ScopeDecl>(childDecl))
                continue;

            ensureAllDeclsRec(childDecl, state);
        }
    }

    // Note: the "inner" declaration of a `GenericDecl` is currently
    // not exposed as one of its children (despite a `GenericDecl`
    // being a `ContainerDecl`), so we need to handle the inner
    // declaration of a generic as another case here.
    //
    if (auto genericDecl = as<GenericDecl>(decl))
    {
        ensureAllDeclsRec(genericDecl->inner, state);
    }
}

bool isUnsizedArrayType(Type* type)
{
    // Not an array?
    auto arrayType = as<ArrayExpressionType>(type);
    if (!arrayType)
        return false;

    // Explicit element count given?
    return arrayType->isUnsized();
}

bool isInterfaceType(Type* type)
{
    if (auto declRefType = as<DeclRefType>(type))
    {
        if (auto interfaceDeclRef = declRefType->getDeclRef().as<InterfaceDecl>())
            return true;
    }
    return false;
}

EnumDecl* isEnumType(Type* type)
{
    if (auto declRefType = as<DeclRefType>(type))
    {
        return as<EnumDecl>(declRefType->getDeclRef().getDecl());
    }
    return nullptr;
}

bool SemanticsVisitor::shouldSkipChecking(Decl* decl, DeclCheckState state)
{
    if (state < DeclCheckState::DefinitionChecked)
        return false;
    // If we are in language server, we should skip checking all the function bodies
    // except for the module or function that the user cared about.
    // This optimization helps reduce the response time.
    if (!getLinkage()->isInLanguageServer())
    {
        return false;
    }
    if (auto funcDecl = as<FunctionDeclBase>(decl))
    {
        auto& assistInfo = getLinkage()->contentAssistInfo;
        // If this func is not defined in the primary module, skip checking its body.
        auto moduleDecl = getModuleDecl(decl);
        if (moduleDecl && moduleDecl->module->getNameObj() != assistInfo.primaryModuleName &&
            moduleDecl->getName() != assistInfo.primaryModuleName)
            return true;
        if (funcDecl->body)
        {
            auto humaneLoc =
                getLinkage()->getSourceManager()->getHumaneLoc(decl->loc, SourceLocType::Actual);
            if (humaneLoc.pathInfo.foundPath != assistInfo.primaryModulePath)
            {
                return true;
            }
            if (assistInfo.checkingMode == ContentAssistCheckingMode::Completion)
            {
                // For completion requests, we skip all funtion bodies except for the one
                // that the current cursor is in.
                auto startingLine = humaneLoc.line;
                for (auto modifier : funcDecl->modifiers)
                {
                    auto modifierLoc = getLinkage()->getSourceManager()->getHumaneLoc(
                        modifier->loc,
                        SourceLocType::Actual);
                    if (modifierLoc.line < startingLine)
                        startingLine = modifierLoc.line;
                }
                auto closingLoc = getLinkage()->getSourceManager()->getHumaneLoc(
                    funcDecl->closingSourceLoc,
                    SourceLocType::Actual);

                if (assistInfo.cursorLine < startingLine || assistInfo.cursorLine > closingLoc.line)
                    return true;
            }
        }
    }
    return false;
}

IntVal* SemanticsVisitor::_validateCircularVarDefinition(VarDeclBase* varDecl)
{
    // The easiest way to test if the declaration is circular is to
    // validate it as a constant.
    //
    // TODO: The logic here will only apply for `static const` declarations
    // of integer type, given that our constant folding currently only
    // applies to such types. A more robust fix would involve a truly
    // recursive walk of the AST declarations, and an even *more* robust
    // fix would wait until after IR linking to detect and diagnose circularity
    // in case it crosses module boundaries.
    //
    //
    if (!isScalarIntegerType(varDecl->type))
        return nullptr;
    return tryConstantFoldDeclRef(
        DeclRef<VarDeclBase>(varDecl),
        ConstantFoldingKind::LinkTime,
        nullptr);
}

void SemanticsDeclModifiersVisitor::visitStructDecl(StructDecl* structDecl)
{
    checkModifiers(structDecl);

    // Replace any bitfield member with a property, do this here before
    // name lookup to avoid the original var decl being referenced
    for (auto& m : structDecl->members)
    {
        const auto bfm = m->findModifier<BitFieldModifier>();
        if (!bfm)
            continue;

        auto property = m_astBuilder->create<PropertyDecl>();
        property->modifiers = m->modifiers;
        property->type = as<VarDecl>(m)->type;
        property->loc = m->loc;
        property->nameAndLoc = m->getNameAndLoc();
        property->parentDecl = structDecl;
        property->ownedScope = m_astBuilder->create<Scope>();
        property->ownedScope->containerDecl = property;
        property->ownedScope->parent = getScope(structDecl);
        m = property;

        const auto get = m_astBuilder->create<GetterDecl>();
        get->ownedScope = m_astBuilder->create<Scope>();
        get->ownedScope->containerDecl = get;
        get->ownedScope->parent = getScope(property);
        property->addMember(get);

        const auto set = m_astBuilder->create<SetterDecl>();
        addModifier(set, m_astBuilder->create<MutatingAttribute>());
        set->ownedScope = m_astBuilder->create<Scope>();
        set->ownedScope->containerDecl = set;
        set->ownedScope->parent = getScope(property);
        property->addMember(set);

        structDecl->invalidateMemberDictionary();
    }
    structDecl->buildMemberDictionary();
}

void SemanticsDeclHeaderVisitor::checkDerivativeMemberAttributeParent(
    VarDeclBase* varDecl,
    DerivativeMemberAttribute* derivativeMemberAttr)
{
    auto memberType = checkProperType(getLinkage(), varDecl->type, getSink());
    auto diffType = getDifferentialType(m_astBuilder, memberType, varDecl->loc);
    if (as<ErrorType>(diffType))
    {
        getSink()->diagnose(derivativeMemberAttr, Diagnostics::typeIsNotDifferentiable, memberType);
    }
    auto thisType = calcThisType(makeDeclRef(varDecl->parentDecl));
    if (!thisType)
    {
        getSink()->diagnose(
            derivativeMemberAttr,
            Diagnostics::derivativeMemberAttributeCanOnlyBeUsedOnMembers);
    }
    auto diffThisType = getDifferentialType(m_astBuilder, thisType, derivativeMemberAttr->loc);
    if (!diffThisType)
    {
        getSink()->diagnose(
            derivativeMemberAttr,
            Diagnostics::invalidUseOfDerivativeMemberAttributeParentTypeIsNotDifferentiable);
    }
}

void SemanticsDeclHeaderVisitor::checkExtensionExternVarAttribute(
    VarDeclBase* varDecl,
    ExtensionExternVarModifier* extensionExternMemberModifier)
{
    if (const auto parentExtension = as<ExtensionDecl>(varDecl->parentDecl))
    {
        if (auto originalVarDecl = extensionExternMemberModifier->originalDecl.as<VarDeclBase>())
        {
            auto originalType = GetTypeForDeclRef(originalVarDecl, originalVarDecl.getLoc());
            auto extVarType = varDecl->type;
            if (!extVarType.type || !extVarType.type->equals(originalType))
            {
                getSink()->diagnose(
                    varDecl,
                    Diagnostics::typeOfExternDeclMismatchesOriginalDefinition,
                    varDecl,
                    originalType);
            }
            else
            {
                return;
            }
        }
        else
        {
            getSink()->diagnose(
                varDecl,
                Diagnostics::definitionOfExternDeclMismatchesOriginalDefinition,
                varDecl);
        }
    }
}

ImageFormat inferImageFormatFromTextureType(
    VarDeclBase* varDecl,
    TextureTypeBase* textureType,
    bool& outIsInferred)
{
    outIsInferred = false;
    ImageFormat format = ImageFormat::unknown;
    if (auto formatVal = as<ConstantIntVal>(textureType->getFormat()))
    {
        format = (ImageFormat)formatVal->getValue();
    }
    if (format != ImageFormat::unknown)
        return format;

    if (auto formatAttrib = varDecl->findModifier<FormatAttribute>())
    {
        format = formatAttrib->format;
    }
    else
    {
        // If format is not specified explicitly through format attribute, we will derive a default
        // value from the element format.
        outIsInferred = true;
        auto elementType = textureType->getElementType();
        Int vectorWidth = 1;
        if (auto elementVecType = as<VectorExpressionType>(elementType))
        {
            if (auto intLitVal = as<ConstantIntVal>(elementVecType->getElementCount()))
            {
                vectorWidth = (Int)intLitVal->getValue();
            }
            else
            {
                vectorWidth = 1;
            }
            elementType = elementVecType->getElementType();
        }
        if (auto basicType = as<BasicExpressionType>(elementType))
        {
            switch (basicType->getBaseType())
            {
            case BaseType::UInt:
                switch (vectorWidth)
                {
                case 1:
                    format = ImageFormat::r32ui;
                    break;
                case 2:
                    format = ImageFormat::rg32ui;
                    break;
                case 4:
                    format = ImageFormat::rgba32ui;
                    break;
                }
                break;
            case BaseType::Int:
                switch (vectorWidth)
                {
                case 1:
                    format = ImageFormat::r32i;
                    break;
                case 2:
                    format = ImageFormat::rg32i;
                    break;
                case 4:
                    format = ImageFormat::rgba32i;
                    break;
                }
                break;
            case BaseType::UInt16:
                switch (vectorWidth)
                {
                case 1:
                    format = ImageFormat::r16ui;
                    break;
                case 2:
                    format = ImageFormat::rg16ui;
                    break;
                case 4:
                    format = ImageFormat::rgba16ui;
                    break;
                }
                break;
            case BaseType::Int16:
                switch (vectorWidth)
                {
                case 1:
                    format = ImageFormat::r16i;
                    break;
                case 2:
                    format = ImageFormat::rg16i;
                    break;
                case 4:
                    format = ImageFormat::rgba16i;
                    break;
                }
                break;
            case BaseType::UInt8:
                switch (vectorWidth)
                {
                case 1:
                    format = ImageFormat::r8ui;
                    break;
                case 2:
                    format = ImageFormat::rg8ui;
                    break;
                case 4:
                    format = ImageFormat::rgba8ui;
                    break;
                }
                break;
            case BaseType::Int8:
                switch (vectorWidth)
                {
                case 1:
                    format = ImageFormat::r8i;
                    break;
                case 2:
                    format = ImageFormat::rg8i;
                    break;
                case 4:
                    format = ImageFormat::rgba8i;
                    break;
                }
                break;
            case BaseType::Int64:
                switch (vectorWidth)
                {
                case 1:
                    format = ImageFormat::r64i;
                    break;
                default:
                    break;
                }
                break;
            case BaseType::UInt64:
                switch (vectorWidth)
                {
                case 1:
                    format = ImageFormat::r64ui;
                    break;
                default:
                    break;
                }
                break;
            case BaseType::Half:
                switch (vectorWidth)
                {
                case 1:
                    format = ImageFormat::r16f;
                    break;
                case 2:
                    format = ImageFormat::rg16f;
                    break;
                case 4:
                    format = ImageFormat::rgba16f;
                    break;
                }
                break;
            }
        }
    }
    return format;
}

void SemanticsDeclHeaderVisitor::maybeApplyLayoutModifier(VarDeclBase* varDecl)
{
    if (auto matrixType = as<MatrixExpressionType>(varDecl->type.type))
    {
        if (auto matrixLayoutModifier = varDecl->findModifier<MatrixLayoutModifier>())
        {
            auto matrixLayout = as<ColumnMajorLayoutModifier>(matrixLayoutModifier)
                                    ? SLANG_MATRIX_LAYOUT_COLUMN_MAJOR
                                    : SLANG_MATRIX_LAYOUT_ROW_MAJOR;
            auto newMatrixType = getASTBuilder()->getMatrixType(
                matrixType->getElementType(),
                matrixType->getRowCount(),
                matrixType->getColumnCount(),
                getASTBuilder()->getIntVal(getASTBuilder()->getIntType(), matrixLayout));
            varDecl->type.type = newMatrixType;
        }
    }
    else if (auto textureType = as<TextureTypeBase>(unwrapArrayType(varDecl->type.type)))
    {
        if (getLinkage()->m_optionSet.getBoolOption(CompilerOptionName::DefaultImageFormatUnknown))
            return;

        // For texture types, we will ensure there is a [format] attribute declared on the decl,
        // if not, we will infer the format from the texture type if it is not specified.
        //
        bool isInferred = false;
        auto format = inferImageFormatFromTextureType(varDecl, textureType, isInferred);
        if (format != ImageFormat::unknown && isInferred)
        {
            auto formatAttrib = m_astBuilder->create<FormatAttribute>();
            formatAttrib->format = format;
            addModifier(varDecl, formatAttrib);
        }
    }
    else
    {
        checkPushConstantBufferType(varDecl);
    }
}

bool isSpecializationConstant(VarDeclBase* varDecl)
{
    for (auto modifier : varDecl->modifiers)
    {
        if (as<SpecializationConstantAttribute>(modifier))
            return true;
        if (as<VkConstantIdAttribute>(modifier))
            return true;
    }
    return false;
}

void SemanticsDeclHeaderVisitor::checkPushConstantBufferType(VarDeclBase* varDecl)
{
    if (varDecl->findModifier<PushConstantAttribute>())
    {
        // If we see a ConstantBuffer<T, DefaultLayout> parameter marked as "push_constant", we need
        // to set its type to ConstantBuffer<T, Std430>.
        if (auto cbufferType = as<ConstantBufferType>(varDecl->type))
        {
            if (cbufferType->getLayoutType() == m_astBuilder->getDefaultLayoutType())
            {
                varDecl->type.type = getConstantBufferType(
                    cbufferType->getElementType(),
                    m_astBuilder->getStd430LayoutType());
            }
        }
        else if (isGlobalShaderParameter(varDecl))
        {
            // If this is a global variable with [vk::push_constant] attribute,
            // we need to make sure to wrap it in a `ConstantBuffer`.
            //
            varDecl->type.type =
                getConstantBufferType(varDecl->type, m_astBuilder->getStd430LayoutType());
        }
    }
}

void SemanticsDeclHeaderVisitor::checkVarDeclCommon(VarDeclBase* varDecl)
{
    // A variable that didn't have an explicit type written must
    // have its type inferred from the initial-value expression.
    //
    if (!varDecl->type.exp)
    {
        // In this case we need to perform all checking of the
        // variable (including semantic checking of the initial-value
        // expression) during the first phase of checking.

        auto initExpr = varDecl->initExpr;
        if (!initExpr)
        {
            if (!varDecl->type.type)
            {
                if (as<GenericValueParamDecl>(varDecl))
                {
                    getSink()->diagnose(varDecl, Diagnostics::genericValueParameterMustHaveType);
                }
                else
                {
                    getSink()->diagnose(varDecl, Diagnostics::varWithoutTypeMustHaveInitializer);
                }
                varDecl->type.type = m_astBuilder->getErrorType();
            }
        }
        else
        {
            SemanticsVisitor subVisitor(withDeclToExcludeFromLookup(varDecl));
            initExpr = subVisitor.CheckExpr(initExpr);

            // TODO: We might need some additional steps here to ensure
            // that the type of the expression is one we are okay with
            // inferring. E.g., if we ever decide that integer and floating-point
            // literals have a distinct type from the standard int/float types,
            // then we would need to "decay" a literal to an explicit type here.

            varDecl->initExpr = initExpr;
            varDecl->type.type = initExpr->type;
            _validateCircularVarDefinition(varDecl);
        }

        // If we've gone down this path, then the variable
        // declaration is actually pretty far along in checking
        varDecl->setCheckState(DeclCheckState::DefinitionChecked);
    }
    else
    {
        // A variable with an explicit type is simpler, for the
        // most part.
        SemanticsVisitor subVisitor(withDeclToExcludeFromLookup(varDecl));
        TypeExp typeExp = subVisitor.CheckUsableType(varDecl->type, varDecl);
        varDecl->type = typeExp;
        if (varDecl->type.equals(m_astBuilder->getVoidType()))
        {
            getSink()->diagnose(varDecl, Diagnostics::invalidTypeVoid);
        }

        // If this is an unsized array variable, then we first want to give
        // it a chance to infer an array size from its initializer
        //
        // TODO(tfoley): May need to extend this to handle the
        // multi-dimensional case...
        //
        if (isUnsizedArrayType(varDecl->type))
        {
            if (auto initExpr = varDecl->initExpr)
            {
                initExpr = CheckTerm(initExpr);
                initExpr = coerce(CoercionSite::Initializer, varDecl->type.Ptr(), initExpr);
                varDecl->initExpr = initExpr;

                maybeInferArraySizeForVariable(varDecl);

                varDecl->setCheckState(DeclCheckState::DefinitionChecked);
            }
        }
        //
        // Next we want to make sure that the declared (or inferred)
        // size for the array meets whatever language-specific
        // constraints we want to enforce (e.g., disallow empty
        // arrays in specific cases)
        //
        validateArraySizeForVariable(varDecl);
    }

    // If there is a matrix layout modifier or texture format modifier, we will modify the type now.
    maybeApplyLayoutModifier(varDecl);

    if (varDecl->initExpr)
    {
        if (as<BasicExpressionType>(varDecl->type.type))
        {
            auto parentDecl = getParentDecl(varDecl);
            if (varDecl->findModifier<ConstModifier>() &&
                (as<NamespaceDeclBase>(parentDecl) || as<FileDecl>(parentDecl) ||
                 varDecl->findModifier<HLSLStaticModifier>()))
            {
                varDecl->val =
                    tryConstantFoldExpr(varDecl->initExpr, ConstantFoldingKind::LinkTime, nullptr);
            }
        }
    }

    checkMeshOutputDecl(varDecl);

    // The NVAPI library allows user code to express extended operations
    // (not supported natively by D3D HLSL) by communicating with
    // a specially identified shader parameter called `g_NvidiaExt`.
    //
    // By default, that shader parameter would look like an ordinary
    // global shader parameter to Slang, but we want to be able to
    // associate special behavior with it to make downstream compilation
    // work nicely (especially in the case where certain cross-platform
    // operations in the Slang core module need to use NVAPI).
    //
    // We will detect a global variable declaration that appears to
    // be declaring `g_NvidiaExt` from NVAPI, and mark it with a special
    // modifier to allow downstream steps to detect it whether or
    // not it has an associated name.
    //
    if (as<ModuleDecl>(varDecl->parentDecl) && varDecl->getName() &&
        varDecl->getName()->text == "g_NvidiaExt")
    {
        addModifier(varDecl, m_astBuilder->create<NVAPIMagicModifier>());
    }
    //
    // One thing that the `NVAPIMagicModifier` is going to do is ensure
    // that `g_NvidiaExt` always gets emitted with *exactly* that name,
    // whether or not obfuscation or other steps are enabled.
    //
    // The `g_NvidiaExt` variable is declared as a:
    //
    //      RWStructuredBuffer<NvShaderExtnStruct>
    //
    // and we also want to make sure that the fields of that struct
    // retain their original names in output code. We will detect
    // variable declarations that represent fields of that struct
    // and flag them as "magic" as well.
    //
    // Note: The goal here is to make it so that generated HLSL output
    // can either use these declarations as they have been preocessed
    // by the Slang front-end *or* they can use declarations directly
    // from the NVAPI header during downstream compilation.
    //
    // TODO: It would be nice if we had a way to identify *all* of the
    // declarations that come from the NVAPI header and mark them, so
    // that the Slang front-end doesn't have to take responsibility
    // for generating code from them (and can instead rely on the downstream
    // compiler alone).
    //
    // The NVAPI header doesn't put any kind of macro-defined modifier
    // (defaulting to an empty macro) in front of its declarations,
    // so the most plausible way to add a modifier to all the declarations
    // would be to tag the `nvHLSLExtns.h` header in a list of "magic"
    // headers which should get all their declarations flagged during
    // front-end processing, and then use the same header again during
    // downstream compilation.
    //
    // For now, the current hackery seems a bit less complicated.
    //
    if (auto structDecl = as<StructDecl>(varDecl->parentDecl))
    {
        if (structDecl->getName() && structDecl->getName()->text == "NvShaderExtnStruct")
        {
            addModifier(varDecl, m_astBuilder->create<NVAPIMagicModifier>());
        }
    }

    if (const auto interfaceDecl = as<InterfaceDecl>(varDecl->parentDecl))
    {
        if (auto basicType = as<BasicExpressionType>(varDecl->getType()))
        {
            switch (basicType->getBaseType())
            {
            case BaseType::Bool:
            case BaseType::Int8:
            case BaseType::Int16:
            case BaseType::Int:
            case BaseType::Int64:
            case BaseType::IntPtr:
            case BaseType::UInt8:
            case BaseType::UInt16:
            case BaseType::UInt:
            case BaseType::UInt64:
            case BaseType::UIntPtr:
                break;
            default:
                getSink()->diagnose(varDecl, Diagnostics::staticConstRequirementMustBeIntOrBool);
                break;
            }
        }
        if (!varDecl->findModifier<HLSLStaticModifier>() || !varDecl->findModifier<ConstModifier>())
        {
            getSink()->diagnose(varDecl, Diagnostics::valueRequirementMustBeCompileTimeConst);
        }
    }

    // Check modifiers that can't be checked earlier during modifier checking stage.
    if (auto derivativeMemberAttr = varDecl->findModifier<DerivativeMemberAttribute>())
    {
        checkDerivativeMemberAttributeParent(varDecl, derivativeMemberAttr);
    }
    if (auto extensionExternAttr = varDecl->findModifier<ExtensionExternVarModifier>())
    {
        checkExtensionExternVarAttribute(varDecl, extensionExternAttr);
    }

    // If a var decl has no_diff type, move the no_diff modifier from the type to the var.
    if (auto modifiedType = as<ModifiedType>(varDecl->type.type))
    {
        if (auto nodiffModifier = modifiedType->findModifier<NoDiffModifierVal>())
        {
            varDecl->type.type = getRemovedModifierType(modifiedType, nodiffModifier);
            auto noDiffModifier = m_astBuilder->create<NoDiffModifier>();
            noDiffModifier->loc = varDecl->loc;
            addModifier(varDecl, noDiffModifier);
        }
    }

    if (as<NamespaceDeclBase>(varDecl->parentDecl))
    {
        if (getModuleDecl(varDecl)->hasModifier<GLSLModuleModifier>())
        {
            // If we are in GLSL compatiblity mode, we want to treat all global variables
            // without any `uniform` modifiers as true global variables by default.
            if (!varDecl->findModifier<HLSLUniformModifier>() &&
                !varDecl->findModifier<InModifier>() && !varDecl->findModifier<OutModifier>() &&
                !varDecl->findModifier<GLSLBufferModifier>() && !isSpecializationConstant(varDecl))
            {
                if (!isUniformParameterType(varDecl->type))
                {
                    auto staticModifier = m_astBuilder->create<HLSLStaticModifier>();
                    addModifier(varDecl, staticModifier);
                }
            }
        }
    }

    // Propagate type tags.
    if (auto parentAggTypeDecl = as<AggTypeDecl>(getParentDecl(varDecl)))
    {
        if (auto varDeclRefType = as<DeclRefType>(varDecl->type.type))
        {
            parentAggTypeDecl->unionTagsWith(getTypeTags(varDeclRefType));
        }
    }
    if (getOptionSet().getBoolOption(CompilerOptionName::NoMangle) && isGlobalDecl(varDecl))
    {
        // If -no-mangle option is set, we will add `ExternCpp` modifier to all
        // global variables and struct fields to prevent mangling.
        addModifier(varDecl, m_astBuilder->create<ExternCppModifier>());
    }
    checkVisibility(varDecl);
}

static void addAutoDiffModifiersToFunc(
    SemanticsDeclVisitorBase* visitor,
    ASTBuilder* m_astBuilder,
    FunctionDeclBase* func)
{
    if (visitor->isTypeDifferentiable(func->returnType.type))
    {
        addModifier(func, m_astBuilder->create<BackwardDifferentiableAttribute>());
        addModifier(func, m_astBuilder->create<ForwardDifferentiableAttribute>());
    }
    else
        addModifier(func, m_astBuilder->create<TreatAsDifferentiableAttribute>());
}

ConstructorDecl* SemanticsDeclVisitorBase::createCtor(
    AggTypeDecl* decl,
    DeclVisibility ctorVisibility)
{
    auto ctor = m_astBuilder->create<ConstructorDecl>();
    addModifier(ctor, m_astBuilder->create<SynthesizedModifier>());
    auto ctorName = getName("$init");
    ctor->ownedScope = m_astBuilder->create<Scope>();
    ctor->ownedScope->containerDecl = ctor;
    ctor->ownedScope->parent = getScope(decl);
    ctor->parentDecl = decl;
    ctor->loc = decl->loc;
    ctor->closingSourceLoc = ctor->loc;
    ctor->nameAndLoc.name = ctorName;
    ctor->nameAndLoc.loc = ctor->loc;
    ctor->returnType.type = calcThisType(makeDeclRef(decl));
    auto body = m_astBuilder->create<BlockStmt>();
    body->scopeDecl = m_astBuilder->create<ScopeDecl>();
    body->scopeDecl->ownedScope = m_astBuilder->create<Scope>();
    body->scopeDecl->ownedScope->parent = getScope(ctor);
    body->scopeDecl->parentDecl = ctor;
    body->scopeDecl->loc = ctor->loc;
    body->scopeDecl->closingSourceLoc = ctor->loc;
    body->closingSourceLoc = ctor->closingSourceLoc;
    ctor->body = body;
    body->body = m_astBuilder->create<SeqStmt>();
    ctor->addFlavor(ConstructorDecl::ConstructorFlavor::SynthesizedDefault);
    decl->addMember(ctor);
    addAutoDiffModifiersToFunc(this, m_astBuilder, ctor);
    addVisibilityModifier(ctor, ctorVisibility);
    return ctor;
}

static inline bool _isDefaultCtor(ConstructorDecl* ctor)
{
    auto allParamHaveInitExpr = [](ConstructorDecl* ctor)
    {
        for (auto i : ctor->getParameters())
            if (!i->initExpr)
                return false;
        return true;
    };

    // 1. default ctor must have no parameters
    // 2. default ctor can have parameters, but all parameters have init expr (Because we won't
    // differentiate this case from 2.)
    if (ctor->members.getCount() == 0 || allParamHaveInitExpr(ctor))
    {
        return true;
    }

    return false;
}

static ConstructorDecl* _getDefaultCtor(StructDecl* structDecl)
{
    for (auto ctor : structDecl->getMembersOfType<ConstructorDecl>())
    {
        if (_isDefaultCtor(ctor))
            return ctor;
    }
    return nullptr;
}


static List<ConstructorDecl*> _getCtorList(
    ASTBuilder* m_astBuilder,
    SemanticsVisitor* visitor,
    StructDecl* structDecl,
    ConstructorDecl** defaultCtorOut)
{
    List<ConstructorDecl*> ctorList;

    auto ctorLookupResult = lookUpMember(
        m_astBuilder,
        visitor,
        visitor->getName("$init"),
        DeclRefType::create(m_astBuilder, structDecl),
        structDecl->ownedScope,
        LookupMask::Function,
        (LookupOptions)((Index)LookupOptions::IgnoreInheritance | (Index)LookupOptions::NoDeref));

    if (!ctorLookupResult.isValid())
        return ctorList;

    auto lookupResultHandle = [&](LookupResultItem& item)
    {
        auto ctor = as<ConstructorDecl>(item.declRef.getDecl());
        if (!ctor || !ctor->body)
            return;
        ctorList.add(ctor);
        if (_isDefaultCtor(ctor))
            *defaultCtorOut = ctor;
    };
    if (ctorLookupResult.items.getCount() == 0)
    {
        lookupResultHandle(ctorLookupResult.item);
        return ctorList;
    }

    for (auto m : ctorLookupResult.items)
    {
        lookupResultHandle(m);
    }

    return ctorList;
}

void SemanticsDeclHeaderVisitor::visitStructDecl(StructDecl* structDecl)
{
    // As described above in `SemanticsDeclHeaderVisitor::checkVarDeclCommon`,
    // we want to identify and tag the "magic" declarations that make NVAPI
    // work, so that downstream passes can identify them and act accordingly.
    //
    // In this case, we are looking for the `NvShaderExtnStruct` type, which
    // is used by `g_NvidiaExt`.
    //
    if (structDecl->getName() && structDecl->getName()->text == "NvShaderExtnStruct")
    {
        addModifier(structDecl, m_astBuilder->create<NVAPIMagicModifier>());
    }

    if (structDecl->hasModifier<ExternModifier>())
    {
        structDecl->addTag(TypeTag::Incomplete);
    }

    // Slang supports a convenient syntax to create a wrapper type from
    // an existing type that implements a given interface. For example,
    // the user can write: struct FooWrapper:IFoo = Foo;
    // In this case we will synthesize the FooWrapper type with an inner
    // member of type `Foo`, and use it to implement all requirements of
    // IFoo.
    // If this is a wrapper struct, synthesize the inner member now.
    if (structDecl->wrappedType.exp)
    {
        structDecl->wrappedType = CheckProperType(structDecl->wrappedType);
        auto member = m_astBuilder->create<VarDecl>();
        member->type = structDecl->wrappedType;
        member->nameAndLoc.name = getName("inner");
        member->nameAndLoc.loc = structDecl->wrappedType.exp->loc;
        member->loc = member->nameAndLoc.loc;
        addModifier(member, m_astBuilder->create<SynthesizedModifier>());
        structDecl->addMember(member);
    }
    checkVisibility(structDecl);
}

void SemanticsDeclHeaderVisitor::visitClassDecl(ClassDecl* classDecl)
{
    if (classDecl->hasModifier<ExternModifier>())
    {
        classDecl->addTag(TypeTag::Incomplete);
    }
    checkVisibility(classDecl);
}

bool DiagnoseIsAllowedInitExpr(VarDeclBase* varDecl, DiagnosticSink* sink)
{
    // find groupshared modifier
    if (varDecl->findModifier<HLSLGroupSharedModifier>())
    {
        if (sink && varDecl->initExpr)
            sink->diagnose(varDecl, Diagnostics::cannotHaveInitializer, varDecl, "groupshared");
        return false;
    }

    return true;
}

bool isDefaultInitializable(VarDeclBase* varDecl)
{
    if (!DiagnoseIsAllowedInitExpr(varDecl, nullptr))
        return false;

    // Find struct and modifiers associated with varDecl
    StructDecl* structDecl = as<StructDecl>(varDecl);
    if (auto declRefType = as<DeclRefType>(varDecl->getType()))
    {
        if (auto genericAppRefDecl = as<GenericAppDeclRef>(declRefType->getDeclRefBase()))
        {
            auto baseGenericRefType = genericAppRefDecl->getBase()->getDecl();
            if (auto baseTypeStruct = as<StructDecl>(baseGenericRefType))
            {
                structDecl = baseTypeStruct;
            }
            else if (auto genericDecl = as<GenericDecl>(baseGenericRefType))
            {
                if (auto innerTypeStruct = as<StructDecl>(genericDecl->inner))
                    structDecl = innerTypeStruct;
            }
        }
    }
    if (structDecl)
    {
        // find if a type is non-copyable
        if (structDecl->findModifier<NonCopyableTypeAttribute>())
            return false;
    }

    return true;
}

static Expr* constructDefaultConstructorForType(SemanticsVisitor* visitor, Type* type)
{
    ConstructorDecl* defaultCtor = nullptr;
    auto declRefType = as<DeclRefType>(type);
    if (declRefType)
    {
        if (auto structDecl = as<StructDecl>(declRefType->getDeclRef().getDecl()))
        {
            defaultCtor = _getDefaultCtor(structDecl);
        }
    }
    if (defaultCtor)
    {
        auto* invoke = visitor->getASTBuilder()->create<InvokeExpr>();
        auto member =
            visitor->getASTBuilder()->getMemberDeclRef(declRefType->getDeclRef(), defaultCtor);
        invoke->functionExpr = visitor->ConstructDeclRefExpr(
            member,
            nullptr,
            defaultCtor->getName(),
            defaultCtor->loc,
            nullptr);
        return invoke;
    }

    return nullptr;
}

static Expr* constructDefaultInitExprForType(SemanticsVisitor* visitor, VarDeclBase* varDecl)
{
    if (!varDecl->type || !varDecl->type.type)
        return nullptr;

    if (!isDefaultInitializable(varDecl))
        return nullptr;

    if (auto defaultInitExpr = constructDefaultConstructorForType(visitor, varDecl->type.type))
    {
        return defaultInitExpr;
    }
    else
    {
        auto* defaultCall = visitor->getASTBuilder()->create<DefaultConstructExpr>();
        defaultCall->type = QualType(varDecl->type);
        return defaultCall;
    }
}

void SemanticsDeclBodyVisitor::checkVarDeclCommon(VarDeclBase* varDecl)
{
    DiagnoseIsAllowedInitExpr(varDecl, getSink());

    // if zero initialize is true, set everything to a default
    if (getOptionSet().hasOption(CompilerOptionName::ZeroInitialize) && !varDecl->initExpr &&
        as<VarDecl>(varDecl))
    {
        varDecl->initExpr = constructDefaultInitExprForType(this, varDecl);
    }

    if (auto initExpr = varDecl->initExpr)
    {
        // Disable the short-circuiting for static const variable init expression
        bool isStaticConst =
            varDecl->hasModifier<HLSLStaticModifier>() && varDecl->hasModifier<ConstModifier>();

        auto subVisitor =
            isStaticConst ? SemanticsVisitor(disableShortCircuitLogicalExpr()) : *this;
        // If the variable has an explicit initial-value expression,
        // then we simply need to check that expression and coerce
        // it to the type of the variable.
        //
        initExpr = subVisitor.CheckTerm(initExpr);

        if (initExpr->type.isWriteOnly)
            getSink()->diagnose(initExpr, Diagnostics::readingFromWriteOnly);

        initExpr = coerce(CoercionSite::Initializer, varDecl->type.Ptr(), initExpr);
        varDecl->initExpr = initExpr;

        // We need to ensure that any variable doesn't introduce
        // a constant with a circular definition.
        //
        varDecl->setCheckState(DeclCheckState::DefinitionChecked);

        // Update constant value
        //
        if (!varDecl->val)
        {
            varDecl->val = _validateCircularVarDefinition(varDecl);
        }
    }
    else
    {
        // If a variable doesn't have an explicit initial-value
        // expression, it is still possible that it should
        // be initialized implicitly, because the type of the
        // variable has a default (zero parameter) initializer.
        // That is, for types where it is possible, we will
        // treat a variable declared like this:
        //
        //      MyType myVar;
        //
        // as if it were declared as:
        //
        //      MyType myVar = MyType();
        //
        // Rather than try to code up an ad hoc search for an
        // appropriate initializer here, we will instead fall
        // back on the general-purpose overload-resolution
        // machinery, which can handle looking up initializers
        // and filtering them to ones that are applicable
        // to our "call site" with zero arguments.
        //
        OverloadResolveContext overloadContext;
        overloadContext.loc = varDecl->nameAndLoc.loc;
        overloadContext.mode = OverloadResolveContext::Mode::JustTrying;
        overloadContext.sourceScope = m_outerScope;

        auto type = varDecl->getType();
        ImplicitCastMethodKey key = ImplicitCastMethodKey(QualType(), type, nullptr);
        auto ctorMethod = getShared()->tryGetImplicitCastMethod(key);
        if (ctorMethod)
        {
            overloadContext.bestCandidateStorage = ctorMethod->conversionFuncOverloadCandidate;
            overloadContext.bestCandidate = &overloadContext.bestCandidateStorage;
        }
        else
        {
            AddTypeOverloadCandidates(type, overloadContext);
        }

        if (overloadContext.bestCandidates.getCount() != 0)
        {
            // If there were multiple equally-good candidates to call,
            // then might have an ambiguity.
            //
            // Before issuing any kind of diagnostic we need to check
            // if any of those candidates are actually applicable,
            // because if they aren't then we actually just have
            // an uninitialized varaible.
            //
            if (overloadContext.bestCandidates[0].status != OverloadCandidate::Status::Applicable)
            {
                getShared()->cacheImplicitCastMethod(key, ImplicitCastMethod{});
            }
            else
            {
                getSink()->diagnose(varDecl, Diagnostics::ambiguousDefaultInitializerForType, type);
            }
        }
        else if (overloadContext.bestCandidate)
        {
            // If we are in the single-candidate case, then we again
            // want to ignore the case where that candidate wasn't
            // actually applicable, because declaring a variable
            // of a type that *doesn't* have a default initializer
            // isn't actually an error.
            //
            if (overloadContext.bestCandidate->status != OverloadCandidate::Status::Applicable)
            {
                getShared()->cacheImplicitCastMethod(key, ImplicitCastMethod{});
            }
            else
            {
                // If we had a single best candidate *and* it was applicable,
                // then we use it to construct a new initial-value expression
                // for the variable, that will be used for all downstream
                // code generation.
                //
                auto constructorDecl =
                    as<ConstructorDecl>(overloadContext.bestCandidate->item.declRef).getDecl();
                // We don't allow implicit initialization of struct only have synthesized default
                // ctor.
                if ((constructorDecl &&
                     !constructorDecl->containsFlavor(
                         ConstructorDecl::ConstructorFlavor::SynthesizedDefault)) ||
                    !constructorDecl)
                {
                    varDecl->initExpr =
                        CompleteOverloadCandidate(overloadContext, *overloadContext.bestCandidate);
                    getShared()->cacheImplicitCastMethod(
                        key,
                        ImplicitCastMethod{*overloadContext.bestCandidate, 0});
                }
            }
        }
    }

    TypeTag varTypeTags = getTypeTags(varDecl->getType());
    auto parentDecl = as<AggTypeDecl>(getParentDecl(varDecl));
    if (parentDecl)
    {
        parentDecl->addTag(varTypeTags);
        auto unsizedMask = (int)TypeTag::Unsized;
        bool isUnknownSize = (((int)varTypeTags & unsizedMask) != 0);
        if (isUnknownSize)
        {
            // Unsized decl must appear as the last member of the struct.
            for (auto memberIdx = parentDecl->members.getCount() - 1; memberIdx >= 0; memberIdx--)
            {
                if (parentDecl->members[memberIdx] == varDecl)
                {
                    break;
                }
                if (auto memberVarDecl = as<VarDeclBase>(parentDecl->members[memberIdx]))
                {
                    if (!memberVarDecl->hasModifier<HLSLStaticModifier>())
                    {
                        getSink()->diagnose(varDecl, Diagnostics::unsizedMemberMustAppearLast);
                    }
                    break;
                }
            }
        }
    }
    bool isGlobalOrLocalVar = !isGlobalShaderParameter(varDecl) && !as<ParamDecl>(varDecl) &&
                              (!parentDecl || isEffectivelyStatic(varDecl));
    if (isGlobalOrLocalVar)
    {
        bool isUnsized = (((int)varTypeTags & (int)TypeTag::Unsized) != 0);
        if (isUnsized)
        {
            getSink()->diagnose(varDecl, Diagnostics::varCannotBeUnsized);
        }

        bool isOpaque = (((int)varTypeTags & (int)TypeTag::Opaque) != 0);
        if (isOpaque && isGlobalDecl(varDecl) && !varDecl->hasModifier<ConstModifier>() &&
            varDecl->hasModifier<HLSLStaticModifier>())
        {
            // Opaque type global variable must be const.
            getSink()->diagnose(varDecl, Diagnostics::globalVarCannotHaveOpaqueType);
            if (varDecl->initExpr)
                getSink()->diagnose(varDecl, Diagnostics::doYouMeanStaticConst);
            else
                getSink()->diagnose(varDecl, Diagnostics::doYouMeanUniform);
        }
    }

    if (auto elementType = getConstantBufferElementType(varDecl->getType()))
    {
        if (doesTypeHaveTag(elementType, TypeTag::Incomplete))
        {
            getSink()->diagnose(
                varDecl->type.exp->loc,
                Diagnostics::incompleteTypeCannotBeUsedInBuffer,
                elementType);
        }
        if (doesTypeHaveTag(elementType, TypeTag::Unsized))
        {
            // If the element type is unsized, it can only be an array of resource types that we can
            // legalize out. Ordinary unsized arrays are not allowed in a constant buffer since we
            // cannot translate it to valid HLSL or SPIRV.
            ArrayExpressionType* trailingArrayType = nullptr;
            VarDeclBase* trailingArrayField =
                getTrailingUnsizedArrayElement(elementType, varDecl, trailingArrayType);
            if (trailingArrayField && !isOpaqueHandleType(trailingArrayType->getElementType()))
            {
                getSink()->diagnose(
                    trailingArrayField->loc,
                    Diagnostics::cannotUseUnsizedTypeInConstantBuffer,
                    trailingArrayType);
                getSink()->diagnose(varDecl->loc, Diagnostics::seeConstantBufferDefinition);
            }
        }
    }
    else if (varDecl->findModifier<HLSLUniformModifier>())
    {
        auto varType = varDecl->getType();
        if (doesTypeHaveTag(varType, TypeTag::Incomplete))
        {
            getSink()->diagnose(
                varDecl->type.exp->loc,
                Diagnostics::incompleteTypeCannotBeUsedInUniformParameter,
                varType);
        }
    }
    maybeRegisterDifferentiableType(getASTBuilder(), varDecl->getType());
}

// Fill in default substitutions for the 'subtype' part of a type constraint decl
void SemanticsVisitor::CheckConstraintSubType(TypeExp& typeExp)
{
    if (auto sharedTypeExpr = as<SharedTypeExpr>(typeExp.exp))
    {
        if (auto declRefType = as<DeclRefType>(sharedTypeExpr->base))
        {
            auto newDeclRef =
                createDefaultSubstitutionsIfNeeded(m_astBuilder, this, declRefType->getDeclRef());
            auto newType = DeclRefType::create(m_astBuilder, newDeclRef);
            sharedTypeExpr->base.type = newType;
            if (as<TypeType>(typeExp.exp->type))
                typeExp.exp->type = m_astBuilder->getTypeType(newType);
        }
    }
}

void SemanticsVisitor::addVisibilityModifier(Decl* decl, DeclVisibility vis)
{
    switch (vis)
    {
    case DeclVisibility::Public:
        addModifier(decl, m_astBuilder->create<PublicModifier>());
        break;
    case DeclVisibility::Internal:
        addModifier(decl, m_astBuilder->create<InternalModifier>());
        break;
    case DeclVisibility::Private:
        addModifier(decl, m_astBuilder->create<PrivateModifier>());
        break;
    default:
        break;
    }
}

bool SemanticsVisitor::trySynthesizeDifferentialAssociatedTypeRequirementWitness(
    ConformanceCheckingContext* context,
    DeclRef<AssocTypeDecl> requirementDeclRef,
    RefPtr<WitnessTable> witnessTable)
{
    ASTSynthesizer synth(m_astBuilder, getNamePool());
    Decl* existingDecl = nullptr;
    AggTypeDecl* aggTypeDecl = nullptr;
    if (context->parentDecl->getMemberDictionary().tryGetValue(
            requirementDeclRef.getName(),
            existingDecl))
    {
        // Remove the `ToBeSynthesizedModifier`.
        if (as<ToBeSynthesizedModifier>(existingDecl->modifiers.first))
        {
            existingDecl->modifiers.first = existingDecl->modifiers.first->next;
        }
        else
        {
            // The user has defined an associatedtype explicitly but that we reach here because
            // that type failed to satisfy the `IDifferential` requirement.
            // We stop the synthesis and let the follow-up logic to report a diagnostic.
            return false;
        }

        aggTypeDecl = as<AggTypeDecl>(existingDecl);
        SLANG_RELEASE_ASSERT(aggTypeDecl);
        synth.pushContainerScope(aggTypeDecl);
    }

    // If we did not find an existing empty struct, we may need to synthesize one.
    // But first, we check if the parent type can be used as its own differential type.
    //
    if (!aggTypeDecl && as<AggTypeDecl>(context->parentDecl) &&
        canStructBeUsedAsSelfDifferentialType(as<AggTypeDecl>(context->parentDecl)))
    {
        // If the parent type can be used as its own differential type, we will create a typealias
        // to itself as the differential type.
        //
        auto assocTypeDef = m_astBuilder->create<TypeDefDecl>();
        assocTypeDef->nameAndLoc.name = getName("Differential");
        assocTypeDef->type.type = context->conformingType;
        context->parentDecl->addMember(assocTypeDef);
        assocTypeDef->setCheckState(DeclCheckState::DefinitionChecked);

        markSelfDifferentialMembersOfType(
            as<AggTypeDecl>(context->parentDecl),
            context->conformingType);

        witnessTable->add(
            requirementDeclRef.getDecl(),
            RequirementWitness(context->conformingType));
        if (doesTypeSatisfyAssociatedTypeConstraintRequirement(
                context->conformingType,
                requirementDeclRef,
                witnessTable))
        {

            // Increase the epoch so that future calls to Type::getCanonicalType will return the
            // up-to-date folded types.
            m_astBuilder->incrementEpoch();
            return true;
        }
        else
        {
            witnessTable->m_requirementDictionary.remove(requirementDeclRef.getDecl());
        }

        // Something went wrong.
        return false;
    }

    if (!aggTypeDecl)
    {
        aggTypeDecl = m_astBuilder->create<StructDecl>();
        context->parentDecl->addMember(aggTypeDecl);
        aggTypeDecl->nameAndLoc.name = requirementDeclRef.getName();
        aggTypeDecl->loc = context->parentDecl->nameAndLoc.loc;
        context->parentDecl->invalidateMemberDictionary();
        synth.pushScopeForContainer(aggTypeDecl);
    }

    // If `This` is nested inside a generic, we need to form a complete declref type to the
    // newly synthesized aggTypeDecl here. This can be done by obtaining the this type witness
    // from requirementDeclRef to get the generic arguments for the outer generic, and
    // apply it to the newly synthesized decl.
    SubstitutionSet substSet;
    Type* thisType = nullptr;
    if (auto thisWitness = findThisTypeWitness(
            SubstitutionSet(requirementDeclRef),
            as<InterfaceDecl>(requirementDeclRef.getParent()).getDecl()))
    {
        thisType = thisWitness->getSub();
        if (auto declRefType = as<DeclRefType>(thisType))
        {
            substSet = SubstitutionSet(declRefType->getDeclRef());
        }
    }
    if (!substSet.declRef)
        return false;
    Type* satisfyingType = nullptr;
    if (substSet.declRef->getDecl() == context->parentDecl)
    {
        // The type we are synthesizing conformance for is direct inside a type itself.
        // We need to copy the outer generic arguments to the synthesized type.
        satisfyingType = DeclRefType::create(
            m_astBuilder,
            m_astBuilder->getMemberDeclRef(substSet.declRef, aggTypeDecl));
    }
    else if (auto parentExtDecl = as<ExtensionDecl>(context->parentDecl))
    {
        // The type is defined in an extension, we need to form a declref to the parent
        // extension from the requirementDeclRef.
        auto extDeclRef = applyExtensionToType(parentExtDecl, thisType);
        satisfyingType = DeclRefType::create(
            m_astBuilder,
            m_astBuilder->getMemberDeclRef(extDeclRef, aggTypeDecl));
    }
    if (!satisfyingType)
        return false;

    // Helper function to add a `diffType` field into the synthesized type for the original
    // `member`.
    auto differentialType =
        DeclRefType::create(m_astBuilder, DeclRef<Decl>(makeDeclRef(aggTypeDecl)));
    auto addDiffMember = [&](Decl* member, Type* diffMemberType)
    {
        // If the field is differentiable, add a corresponding field in the associated Differential
        // type.
        auto diffField = m_astBuilder->create<VarDecl>();
        diffField->nameAndLoc = member->nameAndLoc;
        diffField->type.type = diffMemberType;
        diffField->checkState = DeclCheckState::SignatureChecked;
        aggTypeDecl->addMember(diffField);

        auto visibility = getDeclVisibility(member);
        addVisibilityModifier(diffField, visibility);

        aggTypeDecl->invalidateMemberDictionary();

        // Inject a `DerivativeMember` modifier on the differential field to point to itself.
        {
            auto derivativeMemberModifier = m_astBuilder->create<DerivativeMemberAttribute>();
            auto fieldLookupExpr = m_astBuilder->create<StaticMemberExpr>();
            fieldLookupExpr->type.type = diffMemberType;
            auto baseTypeExpr = m_astBuilder->create<SharedTypeExpr>();
            baseTypeExpr->base.type = differentialType;
            auto baseTypeType = m_astBuilder->getOrCreate<TypeType>(differentialType);
            baseTypeExpr->type.type = baseTypeType;
            fieldLookupExpr->baseExpression = baseTypeExpr;
            fieldLookupExpr->declRef = makeDeclRef(diffField);
            derivativeMemberModifier->memberDeclRef = fieldLookupExpr;
            addModifier(diffField, derivativeMemberModifier);
        }

        // Inject a `DerivativeMember` modifier on the original decl.
        {
            auto derivativeMemberModifier = m_astBuilder->create<DerivativeMemberAttribute>();
            auto fieldLookupExpr = m_astBuilder->create<StaticMemberExpr>();
            fieldLookupExpr->type.type = diffMemberType;
            auto baseTypeExpr = m_astBuilder->create<SharedTypeExpr>();
            baseTypeExpr->base.type = differentialType;
            auto baseTypeType = m_astBuilder->getOrCreate<TypeType>(differentialType);
            baseTypeExpr->type.type = baseTypeType;
            fieldLookupExpr->baseExpression = baseTypeExpr;
            fieldLookupExpr->declRef = makeDeclRef(diffField);
            derivativeMemberModifier->memberDeclRef = fieldLookupExpr;
            addModifier(member, derivativeMemberModifier);
        }
    };

    // Make the Differential type itself conform to `IDifferential` interface.
    bool hasDifferentialConformance = false;
    for (auto inheritanceDecl : aggTypeDecl->getMembersOfType<InheritanceDecl>())
    {
        if (auto declRefType = as<DeclRefType>(inheritanceDecl->base.type))
        {
            if (declRefType->getDeclRef() == m_astBuilder->getDifferentiableInterfaceDecl())
            {
                hasDifferentialConformance = true;
                break;
            }
        }
    }
    if (!hasDifferentialConformance)
    {
        auto inheritanceIDiffernetiable = m_astBuilder->create<InheritanceDecl>();
        inheritanceIDiffernetiable->base.type = m_astBuilder->getDiffInterfaceType();
        aggTypeDecl->addMember(inheritanceIDiffernetiable);
    }

    // The `Differential` type of a `Differential` type is always itself.
    bool hasDifferentialTypeDef = false;
    for (auto member : aggTypeDecl->members)
    {
        if (auto name = member->getName())
        {
            if (name->text == "Differential")
            {
                hasDifferentialTypeDef = true;
                break;
            }
        }
    }
    if (!hasDifferentialTypeDef)
    {
        auto assocTypeDef = m_astBuilder->create<TypeDefDecl>();
        assocTypeDef->nameAndLoc.name = getName("Differential");
        assocTypeDef->type.type = satisfyingType;
        aggTypeDecl->addMember(assocTypeDef);
        assocTypeDef->setCheckState(DeclCheckState::DefinitionChecked);
    }

    // Go through all members and collect their differential types.
    // Go through super types.
    for (auto inheritance : context->parentDecl->getMembersOfType<InheritanceDecl>())
    {
        if (auto baseDeclRefType = as<DeclRefType>(inheritance->base.type))
        {
            // Skip interface super types.
            if (baseDeclRefType->getDeclRef().as<InterfaceDecl>())
                continue;
            if (auto superDiffType = tryGetDifferentialType(m_astBuilder, baseDeclRefType))
            {
                addDiffMember(inheritance, superDiffType);
            }
        }
    }
    // Go through all var members.
    for (auto member : context->parentDecl->getMembersOfType<VarDeclBase>())
    {
        if (member->hasModifier<NoDiffModifier>())
            continue;
        auto diffType = tryGetDifferentialType(m_astBuilder, member->type.type);
        if (!diffType)
            continue;
        addDiffMember(member, diffType);
    }

    addModifier(aggTypeDecl, m_astBuilder->create<SynthesizedModifier>());

    // The visibility of synthesized decl should be the min of the parent decl and the requirement.
    if (requirementDeclRef.getDecl()->findModifier<VisibilityModifier>())
    {
        auto requirementVisibility = getDeclVisibility(requirementDeclRef.getDecl());
        auto thisVisibility = getDeclVisibility(context->parentDecl);
        auto visibility = Math::Min(thisVisibility, requirementVisibility);
        addVisibilityModifier(aggTypeDecl, visibility);
    }

    // Synthesize the rest of IDifferential method conformances by recursively checking
    // conformance on the synthesized decl.
    checkAggTypeConformance(aggTypeDecl);

    witnessTable->add(requirementDeclRef.getDecl(), RequirementWitness(satisfyingType));
    if (!doesTypeSatisfyAssociatedTypeConstraintRequirement(
            satisfyingType,
            requirementDeclRef,
            witnessTable))
    {
        // Note: the call to `doesTypeSatisfyAssociatedTypeConstraintRequirement` should always
        // succeed. If not, there is something wrong with the code synthesis logic. For now we just
        // return false instead of crashing so the user can work around the issues.
        witnessTable->m_requirementDictionary.remove(requirementDeclRef.getDecl());
        return false;
    }
    return true;
}

void SemanticsDeclHeaderVisitor::validateGenericConstraintSubType(
    GenericTypeConstraintDecl* decl,
    TypeExp type)
{
    // Validate that the sub type of a constraint is in valid form.
    //
    if (auto subDeclRef = isDeclRefTypeOf<Decl>(type.type))
    {
        if (subDeclRef.getDecl()->parentDecl == decl->parentDecl)
        {
            // OK, sub type is one of the generic parameter type.
            return;
        }
        if (as<GenericDecl>(decl->parentDecl))
        {
            // If the constraint is in a generic decl, then the sub type must be dependent on at
            // least one of the generic type parameters defined in the same generic decl. For
            // example, it is invalid to define a constraint like `void foo<T>() where int : float`
            // since `int` isn't dependent on any generic type parameter.
            auto dependentGeneric = getShared()->getDependentGenericParent(subDeclRef);
            if (dependentGeneric.getDecl() != decl->parentDecl)
            {
                getSink()->diagnose(type.exp, Diagnostics::invalidConstraintSubType, type);
                return;
            }
        }
        else if (as<AssocTypeDecl>(decl->parentDecl))
        {
            // If the constraint is on an associated type, then it should either be the associated
            // type itself, or a associated type of the associated type. For example,
            // ```
            // interface IFoo {
            //    associatedtype T
            //        where T : IFoo  // OK, constraint is on the associatedtype T itself.
            //        where T.T == X  // OK, constraint is on the associated type of T.
            //        where int == X; // Error, int is not a valid left hand side of a constraint.
            //  }
            // ```
            auto lookupDeclRef = as<LookupDeclRef>(subDeclRef.declRefBase);
            if (!lookupDeclRef)
            {
                getSink()->diagnose(type.exp, Diagnostics::invalidConstraintSubType, type);
                return;
            }

            // We allow `associatedtype T where This.T : ...`.
            // In this case, the left hand side will be in the form of
            // LookupDeclRef(ThisType, T). i.e. lookupDeclRef->getDecl() == T.
            //
            if (lookupDeclRef->getDecl()->parentDecl == decl->parentDecl ||
                lookupDeclRef->getDecl() == decl->parentDecl)
                return;
            auto baseType = as<Type>(lookupDeclRef->getLookupSource());
            if (!baseType)
            {
                getSink()->diagnose(type.exp, Diagnostics::invalidConstraintSubType, type);
                return;
            }
            type.type = baseType;
            validateGenericConstraintSubType(decl, type);
        }
    }
}

void SemanticsDeclHeaderVisitor::visitTypeCoercionConstraintDecl(TypeCoercionConstraintDecl* decl)
{
    CheckConstraintSubType(decl->toType);

    if (!decl->fromType.type)
        decl->fromType = TranslateTypeNodeForced(decl->fromType);
    if (!decl->toType.type)
        decl->toType = TranslateTypeNodeForced(decl->toType);
}

void SemanticsDeclHeaderVisitor::visitGenericTypeConstraintDecl(GenericTypeConstraintDecl* decl)
{
    // TODO: are there any other validations we can do at this point?
    //
    // There probably needs to be a kind of "occurs check" to make
    // sure that the constraint actually applies to at least one
    // of the parameters of the generic.
    //
    CheckConstraintSubType(decl->sub);

    if (!decl->sub.type)
        decl->sub = TranslateTypeNodeForced(decl->sub);
    if (!decl->sup.type)
        decl->sup = TranslateTypeNodeForced(decl->sup);

    if (getLinkage()->m_optionSet.shouldRunNonEssentialValidation())
    {
        validateGenericConstraintSubType(decl, decl->sub);
    }

    if (!decl->isEqualityConstraint && !isValidGenericConstraintType(decl->sup) &&
        !as<ErrorType>(decl->sub.type))
    {
        getSink()->diagnose(decl->sup.exp, Diagnostics::invalidTypeForConstraint, decl->sup);
    }
}

void SemanticsDeclHeaderVisitor::visitGenericTypeParamDecl(GenericTypeParamDecl* decl)
{
    // TODO: could probably push checking the default value
    // for a generic type parameter later.
    //
    decl->initType = CheckProperType(decl->initType);
}

void SemanticsDeclHeaderVisitor::visitGenericValueParamDecl(GenericValueParamDecl* decl)
{
    checkVarDeclCommon(decl);
}

void SemanticsDeclHeaderVisitor::visitGenericDecl(GenericDecl* genericDecl)
{
    genericDecl->setCheckState(DeclCheckState::ReadyForLookup);

    // NOTE! We purposefully do not iterate with the for(auto m : genericDecl->members) here,
    // because the visitor may add to `members` whilst iteration takes place, invalidating the
    // iterator and likely a crash.
    //
    // Accessing the members via index side steps the issue.

    int parameterIndex = 0;
    const auto& members = genericDecl->members;
    for (Index i = 0; i < members.getCount(); ++i)
    {
        Decl* m = members[i];

        if (auto typeParam = as<GenericTypeParamDeclBase>(m))
        {
            ensureDecl(typeParam, DeclCheckState::ReadyForReference);
            typeParam->parameterIndex = parameterIndex++;
        }
        else if (auto valParam = as<GenericValueParamDecl>(m))
        {
            ensureDecl(valParam, DeclCheckState::ReadyForReference);
            valParam->parameterIndex = parameterIndex++;
        }
        else if (auto constraint = as<GenericTypeConstraintDecl>(m))
        {
            ensureDecl(constraint, DeclCheckState::ReadyForReference);
        }
    }
}

void SemanticsDeclBasesVisitor::visitInheritanceDecl(InheritanceDecl* inheritanceDecl)
{
    // check the type being inherited from
    auto base = inheritanceDecl->base;
    Decl* parent = getParentDecl(inheritanceDecl);
    // We exclude transparent members in the case that a circular reference is
    // possible. This is when a parent is also a transparent decl.
    SemanticsContext context(*this);
    if (parent->findModifier<TransparentModifier>())
        context = context.excludeTransparentMembersFromLookup();
    SemanticsDeclVisitorBase baseVistor(context);
    baseVistor.CheckConstraintSubType(base);
    base = baseVistor.TranslateTypeNode(base);
    inheritanceDecl->base = base;

    // Note: we do not check whether the type being inherited from
    // is valid to use for inheritance here, because there could
    // be contextual factors that need to be taken into account
    // based on the declaration that is doing the inheriting.
}

void SemanticsDeclBasesVisitor::visitThisTypeConstraintDecl(
    ThisTypeConstraintDecl* thisTypeConstraintDecl)
{
    // Make sure IFoo<T>.This.ThisIsIFooConstraint.base.type is properly set
    // to DeclRefType(IFoo<T>) with default generic arguments.
    if (!thisTypeConstraintDecl->base.type)
    {
        auto parentTypeDecl = getParentDecl(getParentDecl(thisTypeConstraintDecl));
        thisTypeConstraintDecl->base.type = DeclRefType::create(
            m_astBuilder,
            createDefaultSubstitutionsIfNeeded(
                m_astBuilder,
                this,
                getDefaultDeclRef(parentTypeDecl)));
    }
}

// Concretize interface conformances so that we have witnesses as required for lookup.
// for lookup.
struct SemanticsDeclConformancesVisitor : public SemanticsDeclVisitorBase,
                                          public DeclVisitor<SemanticsDeclConformancesVisitor>
{
    SemanticsDeclConformancesVisitor(SemanticsContext const& outer)
        : SemanticsDeclVisitorBase(outer)
    {
    }

    void visitDecl(Decl*) {}
    void visitDeclGroup(DeclGroup*) {}

    // Any user-defined type may have declared interface conformances,
    // which we should check.
    //
    void visitAggTypeDecl(AggTypeDecl* aggTypeDecl) { checkAggTypeConformance(aggTypeDecl); }

    // Conformances can also come via `extension` declarations, and
    // we should check them against the type(s) being extended.
    //
    void visitExtensionDecl(ExtensionDecl* extensionDecl)
    {
        checkExtensionConformance(extensionDecl);
    }
};

// Check that types used as `Differential` type use themselves as their own `Differential` type.
struct SemanticsDeclDifferentialConformanceVisitor
    : public SemanticsDeclVisitorBase,
      public DeclVisitor<SemanticsDeclDifferentialConformanceVisitor>
{
    SemanticsDeclDifferentialConformanceVisitor(SemanticsContext const& outer)
        : SemanticsDeclVisitorBase(outer)
    {
    }
    void visitDecl(Decl*) {}
    void visitDeclGroup(DeclGroup*) {}

    void visitInheritanceDecl(InheritanceDecl* inheritanceDecl)
    {
        if (as<InterfaceDecl>(inheritanceDecl->parentDecl))
            return;

        if (!inheritanceDecl->witnessTable)
            return;
        auto baseType = as<DeclRefType>(inheritanceDecl->witnessTable->baseType);
        if (!baseType)
            return;
        if (baseType->getDeclRef().getDecl() !=
            m_astBuilder->getDifferentiableInterfaceDecl().getDecl())
            return;
        RequirementWitness witnessValue;
        auto requirementDecl = m_astBuilder->getSharedASTBuilder()->findBuiltinRequirementDecl(
            BuiltinRequirementKind::DifferentialType);
        if (!inheritanceDecl->witnessTable->getRequirementDictionary().tryGetValue(
                requirementDecl,
                witnessValue))
            return;

        if (witnessValue.getFlavor() != RequirementWitness::Flavor::val)
            return;
        auto differentialType = as<DeclRefType>(witnessValue.getVal());
        if (!differentialType)
            return;

        // Check that the type used as differential type must have itself as its own differential
        // type.
        auto diffDiffType = tryGetDifferentialType(m_astBuilder, differentialType);
        if (!differentialType->equals(diffDiffType))
        {
            SourceLoc sourceLoc = differentialType->getDeclRef().getDecl()->loc;
            getSink()->diagnose(
                inheritanceDecl,
                Diagnostics::differentialTypeShouldServeAsItsOwnDifferentialType,
                differentialType,
                diffDiffType);
            getSink()->diagnose(sourceLoc, Diagnostics::seeDefinitionOf, differentialType);
        }

        // Check that all [DerivativeMember(...)] attributes have their references checked.
        for (auto member : inheritanceDecl->parentDecl->getMembersOfType<VarDeclBase>())
        {
            if (member->findModifier<NoDiffModifier>())
                continue;
            auto derivativeMemberAttr = member->findModifier<DerivativeMemberAttribute>();
            if (!derivativeMemberAttr)
                continue;
            checkDerivativeMemberAttributeReferences(member, derivativeMemberAttr);
        }

        // Check that either the differential type is the same as the base type, or all fields of
        // the base type that are differentiable have a corresponding field in the differential type
        // through the [DerivativeMember(...)] attribute.
        //
        // We only need to check the fields of the base type that are differentiable.
        auto baseDecl = as<AggTypeDecl>(inheritanceDecl->parentDecl);
        if (!baseDecl)
            return;

        auto thisType = calcThisType(getDefaultDeclRef(baseDecl));

        bool typeIsSelfDifferential = thisType->equals(differentialType);

        for (auto member : baseDecl->getMembersOfType<VarDeclBase>())
        {
            if (member->findModifier<NoDiffModifier>())
                continue;
            auto diffType = tryGetDifferentialType(m_astBuilder, member->type.type);
            if (!diffType)
                continue;

            if (member->findModifier<DerivativeMemberAttribute>())
                continue;
            else if (!typeIsSelfDifferential)
                getSink()->diagnose(
                    member,
                    Diagnostics::differentiableMemberShouldHaveCorrespondingFieldInDiffType,
                    member->nameAndLoc.name,
                    differentialType);
            else
            {
                // If the type is its own differential type, we can infer the differential
                // members from the original type.
                //
                // Add a derivative member attribute referencing itself.
                //
                auto derivativeMemberModifier = m_astBuilder->create<DerivativeMemberAttribute>();
                auto fieldLookupExpr = m_astBuilder->create<StaticMemberExpr>();
                fieldLookupExpr->type.type = diffType;
                auto baseTypeExpr = m_astBuilder->create<SharedTypeExpr>();
                baseTypeExpr->base.type = differentialType;
                auto baseTypeType = m_astBuilder->getOrCreate<TypeType>(differentialType);
                baseTypeExpr->type.type = baseTypeType;
                fieldLookupExpr->baseExpression = baseTypeExpr;
                fieldLookupExpr->declRef = makeDeclRef(member);
                derivativeMemberModifier->memberDeclRef = fieldLookupExpr;
                addModifier(member, derivativeMemberModifier);
            }
        }
    }
};

/// Recursively register any builtin declarations that need to be attached to the `session`.
///
/// This function should only be needed for declarations in the core module.
///
static void _registerBuiltinDeclsRec(Session* session, Decl* decl)
{
    SharedASTBuilder* sharedASTBuilder = session->m_sharedASTBuilder;

    if (auto builtinMod = decl->findModifier<BuiltinTypeModifier>())
    {
        sharedASTBuilder->registerBuiltinDecl(decl, builtinMod);
    }
    if (auto magicMod = decl->findModifier<MagicTypeModifier>())
    {
        sharedASTBuilder->registerMagicDecl(decl, magicMod);
    }
    if (auto builtinRequirement = decl->findModifier<BuiltinRequirementModifier>())
    {
        sharedASTBuilder->registerBuiltinRequirementDecl(decl, builtinRequirement);
    }
    if (auto containerDecl = as<ContainerDecl>(decl))
    {
        for (auto childDecl : containerDecl->members)
        {
            if (as<ScopeDecl>(childDecl))
                continue;

            _registerBuiltinDeclsRec(session, childDecl);
        }
    }
    if (auto genericDecl = as<GenericDecl>(decl))
    {
        _registerBuiltinDeclsRec(session, genericDecl->inner);
    }
}

void registerBuiltinDecls(Session* session, Decl* decl)
{
    _registerBuiltinDeclsRec(session, decl);
}

Type* unwrapArrayType(Type* type)
{
    for (;;)
    {
        if (auto arrayType = as<ArrayExpressionType>(type))
            type = arrayType->getElementType();
        else
            return type;
    }
}

Type* unwrapModifiedType(Type* type)
{
    for (;;)
    {
        if (auto modType = as<ModifiedType>(type))
            type = modType->getBase();
        else
            return type;
    }
}

void discoverExtensionDecls(List<ExtensionDecl*>& decls, Decl* parent)
{
    if (auto extDecl = as<ExtensionDecl>(parent))
        decls.add(extDecl);
    if (auto containerDecl = as<ContainerDecl>(parent))
    {
        for (auto child : containerDecl->members)
        {
            discoverExtensionDecls(decls, child);
        }
    }
    if (auto genericDecl = as<GenericDecl>(parent))
    {
        discoverExtensionDecls(decls, genericDecl->inner);
    }
}

void SemanticsDeclVisitorBase::checkModule(ModuleDecl* moduleDecl)
{
    // When we are dealing with code from the core modules,
    // there is a potential problem where we might need to look
    // up built-in types like `Int` through the session (e.g.,
    // to determine the type for an integer literal), but those
    // types might not have been registered yet. We solve that
    // by doing a pre-process on the core module code to find
    // and register any built-in declarations.
    //
    // TODO: This could be factored into another visitor pass
    // that fits the more standard checking below, but that would
    // seemingly add overhead to checking things other than
    // the core module.
    //
    if (isFromCoreModule(moduleDecl))
    {
        _registerBuiltinDeclsRec(getSession(), moduleDecl);
    }

    if (moduleDecl->members.getCount() > 0)
    {
        auto firstMember = moduleDecl->members[0];
        if (as<ImplementingDecl>(firstMember))
        {
            if (!getShared()->isInLanguageServer())
            {
                // A primary module file can't start with an "implementing" declaration.
                getSink()->diagnose(
                    firstMember,
                    Diagnostics::primaryModuleFileCannotStartWithImplementingDecl);
            }
        }
        else if (!as<ModuleDeclarationDecl>(firstMember))
        {
            // A primary module file must start with a `module` declaration.
            // TODO: this warning is disabled for now to free users from massive change for now.
#if 0
                getSink()->diagnose(firstMember, Diagnostics::primaryModuleFileMustStartWithModuleDecl);
#endif
        }
    }

    if (moduleDecl->findModifier<PublicModifier>())
    {
        moduleDecl->defaultVisibility = DeclVisibility::Public;
    }

    // We need/want to visit any `import` declarations before
    // anything else, to make sure that scoping works.
    //
    // TODO: This could be factored into another visitor pass
    // that fits more with the standard checking below.
    //
    for (auto importDecl : moduleDecl->getMembersOfType<ImportDecl>())
    {
        ensureDecl(importDecl, DeclCheckState::DefinitionChecked);
    }

    // Next, make sure all `__include` decls are processed and the referenced
    // files are parsed.
    auto visitIncludeDecls = [&](ContainerDecl* fileDecl)
    {
        for (Index i = 0; i < fileDecl->members.getCount(); i++)
        {
            auto decl = fileDecl->members[i];
            if (auto includeDecl = as<IncludeDecl>(decl))
            {
                ensureDecl(includeDecl, DeclCheckState::DefinitionChecked);
            }
            else if (auto implementingDecl = as<ImplementingDecl>(decl))
            {
                ensureDecl(implementingDecl, DeclCheckState::DefinitionChecked);
            }
            else if (auto importDecl = as<ImportDecl>(decl))
            {
                ensureDecl(importDecl, DeclCheckState::DefinitionChecked);
            }
        }
    };
    visitIncludeDecls(moduleDecl);
    for (Index i = 0; i < moduleDecl->members.getCount(); i++)
    {
        if (auto fileDecl = as<FileDecl>(moduleDecl->members[i]))
            visitIncludeDecls(fileDecl);
    }

    // The entire goal of semantic checking is to get all of the
    // declarations in the module up to `DeclCheckState::DefinitionChecked`.
    //
    // The main catch is that checking one declaration A up to state M
    // may required that declaration B is checked up to state N.
    // A call to `ensureDecl(B, N)` can guarantee that things are checked
    // when and where we need them, but that runs the risk of creating
    // very deep recursion in the semantic checking.
    //
    // Instead, we would rather do more breadth-first checking,
    // where everything gets checked up to state 1, 2, ...
    // before anything gets too far ahead.
    // We will therefore enumerate the states/phases for checking,
    // and then iteratively try to update all declarations to each
    // state in turn.
    //
    // Note: for a simpler language we could eliminate `ensureDecl`
    // completely and *just* have these phases of checking.
    // Unfortunately, we have some circularity between the phases:
    //
    // * Checking an overloaded call requires knowing the parameter
    //   types of all candidate callees.
    //
    // * Checking the parameter type of a function requires being
    //   able to check type expressions.
    //
    // * A type expression like `vector<T, N>` may have an arbitary
    //   expression for `N`.
    //
    // * An arbitrary expression may include function calls, which
    //   may be to overloaded functions.
    //
    // Languages like C++ solve the apparent problem by making
    // restrictions on order of declaration/definition (and by
    // requiring forward declarations or the `template`/`typename`
    // keywrods in some cases).
    //
    // TODO: We could eventually eliminate the potential recursion
    // in checking by splitting each phase into a "requirements gathering"
    // step and an actual execution step.
    //
    // When checking a declaration D up to state S, the requirements
    // gathering step would produce a list of pairs `(someDecl, someState)`
    // indicating that `someDecl` must be in `someState` before the
    // actual execution of checking for `(D,S)` can proceeed. The checker
    // can then produce an elaborated dependency graph and select nodes
    // for execution in an order that satisfies all the dependencies.
    //
    // Such a more elaborate checking scheme will have to wait for another
    // day, but might be worth it (or even necessary) if/when we want to
    // support incremental compilation.
    //
    DeclCheckState states[] = {
        DeclCheckState::ScopesWired,
        DeclCheckState::ReadyForReference,
        DeclCheckState::ReadyForLookup,
        DeclCheckState::ReadyForConformances,
        DeclCheckState::DefinitionChecked,
        DeclCheckState::CapabilityChecked,
    };

    // Discover and check all extension decls before anything else.
    List<ExtensionDecl*> extensionDecls;
    discoverExtensionDecls(extensionDecls, moduleDecl);
    for (auto s : states)
    {
        for (auto extensionDecl : extensionDecls)
        {
            ensureDecl(extensionDecl, s);
        }
        // We only need to check extension decls up to ReadyForLookup
        // so they are properly registered in type inheritance infos.
        if (s == DeclCheckState::ReadyForLookup)
            break;
    }

    // With extensions taken care of, we can now check the remaining decls.
    for (auto s : states)
    {
        // When advancing to state `s` we will recursively
        // advance all declarations rooted in the module
        // up to `s`.
        //
        // TODO: In cases where a large module is split across files,
        // we could potentially parallelize front-end compilation by
        // having multiple instances of the front end where each is
        // only responsible for those declarations in a given file.
        //
        // Under that model, we might only apply later phases of
        // checking (notably the final push to `DeclState::Checked`)
        // to the subset of declarations coming from a given source
        // file.
        //
        ensureAllDeclsRec(moduleDecl, s);
    }

    // Once we have completed the above loop, all declarations not
    // nested in function bodies should be in `DeclState::Checked`.
    // Furthermore, because a fully checked function will have checked
    // its body, this also means that all function bodies and the
    // declarations they contain should be fully checked.
}

bool SemanticsVisitor::doesSignatureMatchRequirement(
    DeclRef<CallableDecl> satisfyingMemberDeclRef,
    DeclRef<CallableDecl> requiredMemberDeclRef,
    RefPtr<WitnessTable> witnessTable)
{
    if (satisfyingMemberDeclRef.getDecl()->hasModifier<MutatingAttribute>() !=
        requiredMemberDeclRef.getDecl()->hasModifier<MutatingAttribute>())
    {
        // A `[mutating]` method can't satisfy a non-`[mutating]` requirement.
        // The opposite direction is okay, but we will need to synthesize a wrapper
        // to ensure type matches, so we will return false here either way.
        return false;
    }

    if (satisfyingMemberDeclRef.getDecl()->hasModifier<ConstRefAttribute>() !=
        requiredMemberDeclRef.getDecl()->hasModifier<ConstRefAttribute>())
    {
        // A `[constref]` method can't satisfy a non-`[constref]` requirement.
        // The opposite direction is okay, but we will need to synthesize a wrapper
        // to ensure type matches, so we will return false here either way.
        return false;
    }

    if (satisfyingMemberDeclRef.getDecl()->hasModifier<RefAttribute>() !=
        requiredMemberDeclRef.getDecl()->hasModifier<RefAttribute>())
    {
        // A `[ref]` method can't satisfy a non-`[ref]` requirement.
        // The opposite direction is okay, but we will need to synthesize a wrapper
        // to ensure type matches, so we will return false here either way.
        return false;
    }

    if (satisfyingMemberDeclRef.getDecl()->hasModifier<HLSLStaticModifier>() !=
        requiredMemberDeclRef.getDecl()->hasModifier<HLSLStaticModifier>())
    {
        // A `static` method can't satisfy a non-`static` requirement and vice versa.
        return false;
    }

    bool hasBackwardDerivative = false;
    bool hasForwardDerivative = false;
    if (requiredMemberDeclRef.getDecl()->hasModifier<BackwardDifferentiableAttribute>())
    {
        auto funcDecl = as<FunctionDeclBase>(satisfyingMemberDeclRef.getDecl());
        if (!funcDecl)
            return false;

        if (getShared()->getFuncDifferentiableLevel(funcDecl) !=
            FunctionDifferentiableLevel::Backward)
        {
            // A non-`BackwardDifferentiable` method can't satisfy a `BackwardDifferentiable`
            // requirement and vice versa.
            return false;
        }
        hasBackwardDerivative = true;
        hasForwardDerivative = true;
    }
    else if (requiredMemberDeclRef.getDecl()->hasModifier<ForwardDifferentiableAttribute>())
    {
        auto funcDecl = as<FunctionDeclBase>(satisfyingMemberDeclRef.getDecl());
        if (!funcDecl)
            return false;
        if (getShared()->getFuncDifferentiableLevel(funcDecl) == FunctionDifferentiableLevel::None)
        {
            // A non-`BackwardDifferentiable` method can't satisfy a `BackwardDifferentiable`
            // requirement and vice versa.
            return false;
        }
        hasForwardDerivative = true;
    }

    // A signature matches the required one if it has the right number of parameters,
    // and those parameters have the right types, and also the result/return type
    // is the required one.
    //
    auto requiredParams = getParameters(m_astBuilder, requiredMemberDeclRef).toArray();
    auto satisfyingParams = getParameters(m_astBuilder, satisfyingMemberDeclRef).toArray();
    auto paramCount = requiredParams.getCount();
    if (satisfyingParams.getCount() != paramCount)
        return false;

    for (Index paramIndex = 0; paramIndex < paramCount; ++paramIndex)
    {
        auto requiredParam = requiredParams[paramIndex];
        auto satisfyingParam = satisfyingParams[paramIndex];
        if (getParameterDirection(requiredParam.getDecl()) !=
            getParameterDirection(satisfyingParam.getDecl()))
            return false;
        auto requiredParamType = getType(m_astBuilder, requiredParam);
        auto satisfyingParamType = getType(m_astBuilder, satisfyingParam);

        if (!requiredParamType->equals(satisfyingParamType))
            return false;
    }

    auto requiredResultType = getResultType(m_astBuilder, requiredMemberDeclRef);
    auto satisfyingResultType = getResultType(m_astBuilder, satisfyingMemberDeclRef);
    if (!requiredResultType->equals(satisfyingResultType))
        return false;

    if (hasForwardDerivative || hasBackwardDerivative)
    {
        auto parentInterfaceDecl =
            as<InterfaceDecl>(getParentDecl(requiredMemberDeclRef.getDecl()));
        if (parentInterfaceDecl)
        {
            bool noDiffThisSatisfying = !isTypeDifferentiable(witnessTable->witnessedType);
            bool noDiffThisRequirement =
                (requiredMemberDeclRef.getDecl()->findModifier<NoDiffThisAttribute>() != nullptr);
            if (noDiffThisRequirement != noDiffThisSatisfying)
                return false;
        }
    }

    _addMethodWitness(witnessTable, requiredMemberDeclRef, satisfyingMemberDeclRef);

    return true;
}

bool SemanticsVisitor::doesAccessorMatchRequirement(
    DeclRef<AccessorDecl> satisfyingMemberDeclRef,
    DeclRef<AccessorDecl> requiredMemberDeclRef)
{
    // We require the AST node class of the satisfying accessor
    // to be a subclass of the one from the required accessor.
    //
    // For our current accessor types, this amounts to requiring
    // an exact match, but using a subtype test means that if
    // we ever add an `ExtraSpecialGetDecl` that is a subclass
    // of `GetDecl`, then one of those would be able to satisfy
    // a `get` requirement.
    //
    auto satisfyingMemberClass = satisfyingMemberDeclRef.getDecl()->getClass();
    auto requiredMemberClass = requiredMemberDeclRef.getDecl()->getClass();
    if (!satisfyingMemberClass.isSubClassOf(requiredMemberClass))
        return false;

    // We do not check the parameters or return types of accessors
    // here, under the assumption that the validity checks for
    // the parent `property` declaration would already make sure
    // they are in order.

    // TODO: There are other checks we need to make here, like not letting
    // an ordinary `set` satisfy a `[nonmutating] set` requirement.

    return true;
}

bool SemanticsVisitor::doesPropertyMatchRequirement(
    DeclRef<PropertyDecl> satisfyingMemberDeclRef,
    DeclRef<PropertyDecl> requiredMemberDeclRef,
    RefPtr<WitnessTable> witnessTable)
{
    // The type of the satisfying member must match the type of the required member.
    //
    // Note: It is possible that a `get`-only property could be satisfied by
    // a declaration that uses a subtype of the requirement, but that would not
    // count as an "exact match" and we would rely on the logic to synthesize
    // a stub implementation in that case.
    //
    auto satisfyingType = getType(getASTBuilder(), satisfyingMemberDeclRef);
    auto requiredType = getType(getASTBuilder(), requiredMemberDeclRef);
    if (!satisfyingType->equals(requiredType))
        return false;

    // Each accessor in the requirement must be accounted for by an accessor
    // in the satisfying member.
    //
    // Note: it is fine for the satisfying member to provide *more* accessors
    // than the original declaration.
    //
    Dictionary<DeclRef<AccessorDecl>, DeclRef<AccessorDecl>> mapRequiredToSatisfyingAccessorDeclRef;
    for (auto requiredAccessorDeclRef :
         getMembersOfType<AccessorDecl>(m_astBuilder, requiredMemberDeclRef))
    {
        // We need to search for an accessor that can satisfy the requirement.
        //
        // For now we will do the simplest (and slowest) thing of a linear search,
        // which is mostly fine because the number of accessors is bounded.
        //
        bool found = false;
        for (auto satisfyingAccessorDeclRef :
             getMembersOfType<AccessorDecl>(m_astBuilder, satisfyingMemberDeclRef))
        {
            if (doesAccessorMatchRequirement(satisfyingAccessorDeclRef, requiredAccessorDeclRef))
            {
                // When we find a match on an accessor, we record it so that
                // we can set up the witness values later, but we do *not*
                // record it into the actual witness table yet, in case
                // a later accessor comes along that doesn't find a match.
                //
                mapRequiredToSatisfyingAccessorDeclRef.add(
                    requiredAccessorDeclRef,
                    satisfyingAccessorDeclRef);
                found = true;
                break;
            }
        }
        if (!found)
            return false;
    }

    // Once things are done, we will install the satisfying values
    // into the witness table for the requirements.
    //
    for (const auto& [key, value] : mapRequiredToSatisfyingAccessorDeclRef)
    {
        witnessTable->add(key.getDecl(), RequirementWitness(value));
    }
    //
    // Note: the property declaration itself isn't something that
    // has a useful value/representation in downstream passes, so
    // we are mostly just installing it into the witness table
    // as a way to mark this requirement as being satisfied.
    //
    // TODO: It is possible that having a witness table entry that
    // doesn't actually map to any IR value could create a problem
    // in downstream passes. If such propblems arise, we should
    // probably create a new `RequirementWitness` case that
    // represents a witness value that is only needed by the front-end,
    // and that can be ignored by IR and emit logic.
    //
    witnessTable->add(requiredMemberDeclRef.getDecl(), RequirementWitness(satisfyingMemberDeclRef));
    return true;
}

bool SemanticsVisitor::doesSubscriptMatchRequirement(
    DeclRef<SubscriptDecl> satisfyingMemberDeclRef,
    DeclRef<SubscriptDecl> requiredMemberDeclRef,
    RefPtr<WitnessTable> witnessTable)
{
    // The result type and parameters of the satisfying member must match the type of the required
    // member.
    //
    auto requiredParams = getParameters(m_astBuilder, requiredMemberDeclRef).toArray();
    auto satisfyingParams = getParameters(m_astBuilder, satisfyingMemberDeclRef).toArray();
    auto paramCount = requiredParams.getCount();
    if (satisfyingParams.getCount() != paramCount)
        return false;

    for (Index paramIndex = 0; paramIndex < paramCount; ++paramIndex)
    {
        auto requiredParam = requiredParams[paramIndex];
        auto satisfyingParam = satisfyingParams[paramIndex];

        auto requiredParamType = getType(m_astBuilder, requiredParam);
        auto satisfyingParamType = getType(m_astBuilder, satisfyingParam);

        if (!requiredParamType->equals(satisfyingParamType))
            return false;
    }

    auto requiredResultType = getResultType(m_astBuilder, requiredMemberDeclRef);
    auto satisfyingResultType = getResultType(m_astBuilder, satisfyingMemberDeclRef);
    if (!requiredResultType->equals(satisfyingResultType))
        return false;

    // Each accessor in the requirement must be accounted for by an accessor
    // in the satisfying member.
    //
    // Note: it is fine for the satisfying member to provide *more* accessors
    // than the original declaration.
    //
    Dictionary<DeclRef<AccessorDecl>, DeclRef<AccessorDecl>> mapRequiredToSatisfyingAccessorDeclRef;
    for (auto requiredAccessorDeclRef :
         getMembersOfType<AccessorDecl>(m_astBuilder, requiredMemberDeclRef))
    {
        // We need to search for an accessor that can satisfy the requirement.
        //
        // For now we will do the simplest (and slowest) thing of a linear search,
        // which is mostly fine because the number of accessors is bounded.
        //
        bool found = false;
        for (auto satisfyingAccessorDeclRef :
             getMembersOfType<AccessorDecl>(m_astBuilder, satisfyingMemberDeclRef))
        {
            if (doesAccessorMatchRequirement(satisfyingAccessorDeclRef, requiredAccessorDeclRef))
            {
                // When we find a match on an accessor, we record it so that
                // we can set up the witness values later, but we do *not*
                // record it into the actual witness table yet, in case
                // a later accessor comes along that doesn't find a match.
                //
                mapRequiredToSatisfyingAccessorDeclRef.add(
                    requiredAccessorDeclRef,
                    satisfyingAccessorDeclRef);
                found = true;
                break;
            }
        }
        if (!found)
            return false;
    }

    // Once things are done, we will install the satisfying values
    // into the witness table for the requirements.
    //
    for (const auto& [key, value] : mapRequiredToSatisfyingAccessorDeclRef)
    {
        witnessTable->add(key.getDecl(), RequirementWitness(value));
    }
    //
    // Note: the subscript declaration itself isn't something that
    // has a useful value/representation in downstream passes, so
    // we are mostly just installing it into the witness table
    // as a way to mark this requirement as being satisfied.
    //
    witnessTable->add(requiredMemberDeclRef.getDecl(), RequirementWitness(satisfyingMemberDeclRef));
    return true;
}

bool SemanticsVisitor::doesVarMatchRequirement(
    DeclRef<VarDeclBase> satisfyingMemberDeclRef,
    DeclRef<VarDeclBase> requiredMemberDeclRef,
    RefPtr<WitnessTable> witnessTable)
{
    // The type of the satisfying member must match the type of the required member.
    auto satisfyingType = getType(getASTBuilder(), satisfyingMemberDeclRef);
    auto requiredType = getType(getASTBuilder(), requiredMemberDeclRef);
    if (!satisfyingType->equals(requiredType))
        return false;

    for (auto modifier : requiredMemberDeclRef.getDecl()->modifiers)
    {
        bool found = false;
        for (auto satisfyingModifier : satisfyingMemberDeclRef.getDecl()->modifiers)
        {
            if (satisfyingModifier->astNodeType == modifier->astNodeType)
            {
                found = true;
                break;
            }
        }
        if (!found)
            return false;
    }

    auto satisfyingVal =
        tryConstantFoldDeclRef(satisfyingMemberDeclRef, ConstantFoldingKind::LinkTime, nullptr);
    if (satisfyingVal)
    {
        witnessTable->add(requiredMemberDeclRef.getDecl(), RequirementWitness(satisfyingVal));
    }
    else
    {
        witnessTable->add(
            requiredMemberDeclRef.getDecl(),
            RequirementWitness(satisfyingMemberDeclRef));
    }
    return true;
}

bool SemanticsVisitor::doesGenericSignatureMatchRequirement(
    DeclRef<GenericDecl> satisfyingGenericDeclRef,
    DeclRef<GenericDecl> requiredGenericDeclRef,
    RefPtr<WitnessTable> witnessTable)
{
    // The signature of a generic is defiend by its members, and we need the
    // satisfying value to have the same number of members for it to be an
    // exact match.
    //
    auto memberCount = requiredGenericDeclRef.getDecl()->members.getCount();
    if (satisfyingGenericDeclRef.getDecl()->members.getCount() != memberCount)
        return false;

    // We then want to check that pairwise members match, in order.
    //
    auto requiredMemberDeclRefs = getMembers(m_astBuilder, requiredGenericDeclRef);
    auto satisfyingMemberDeclRefs = getMembers(m_astBuilder, satisfyingGenericDeclRef);
    //
    // We start by performing a superficial "structural" match of the parameters
    // to ensure that the two generics have an equivalent mix of type, value,
    // and constraint parameters in the same order.
    //
    // Note that in this step we do *not* make any checks on the actual types
    // involved in constraints, or on the types of value parameters. The reason
    // for this is that the types on those parameters could be dependent on
    // type parameters in the generic parameter list, and thus there could be
    // a mismatch at this point. For example, if we have:
    //
    //      interface IBase         { void doThing<T, U : IThing<T>>(); }
    //      struct Derived : IBase  { void doThing<X, Y : IThing<X>>(); }
    //
    // We clearly have a signature match here, but the constraint parameters for
    // `U : IThing<T>` and `Y : IThing<X>` have the problem that both the sub-type
    // and super-type they reference are not equivalent without substititions.
    //
    // We will deal with this issue after the structural matching is checked, at
    // which point we can actually verify things like types.
    //
    for (Index i = 0; i < memberCount; i++)
    {
        auto requiredMemberDeclRef = requiredMemberDeclRefs[i];
        auto satisfyingMemberDeclRef = satisfyingMemberDeclRefs[i];

        if (as<GenericTypeParamDecl>(requiredMemberDeclRef))
        {
            if (as<GenericTypeParamDecl>(satisfyingMemberDeclRef))
            {
            }
            else
                return false;
        }
        else if (auto requiredValueParamDeclRef = requiredMemberDeclRef.as<GenericValueParamDecl>())
        {
            if (auto satisfyingValueParamDeclRef =
                    satisfyingMemberDeclRef.as<GenericValueParamDecl>())
            {
            }
            else
                return false;
        }
        else if (
            auto requiredConstraintDeclRef = requiredMemberDeclRef.as<GenericTypeConstraintDecl>())
        {
            if (auto satisfyingConstraintDeclRef =
                    satisfyingMemberDeclRef.as<GenericTypeConstraintDecl>())
            {
            }
            else
                return false;
        }
    }

    // In order to compare the inner declarations of the two generics, we need to
    // align them so that they are expressed in terms of consistent type parameters.
    //
    // For example, we might have:
    //
    //      interface IBase           { void doThing<T>(T val); }
    //      struct    Derived : IBase { void doThing<U>(U val); }
    //
    // If we directly compare the signatures of the inner `doThing` function declarations,
    // we'd find a mismatch between the `T` and `U` types of the `val` parameter.
    //
    // We can get around this mismatch by constructing a specialized reference and
    // then doing the comparison. For example `IBase::doThing<X>` and `Derived::doThing<X>`
    // should both have the signature `X -> void`.
    //
    // The one big detail that we need to be careful about here is that when we
    // recursively call `doesMemberSatisfyRequirement`, that will eventually store
    // the satisfying `DeclRef` as the value for the given requirement key, and we don't
    // want to store a specialized reference like `Derived::doThing<X>` - we need to
    // somehow store the original declaration.
    //
    // The solution here is to specialize the *required* declaration to the parameters
    // of the satisfying declaration. In the example above that means we are going to
    // compare `Derived::doThing` against `IBase::doThing<U>` where the `U` there is
    // the parameter of `Dervived::doThing`.
    //
    List<Val*> requiredSubstArgs;

    for (Index i = 0; i < memberCount; i++)
    {
        auto requiredMemberDeclRef = requiredMemberDeclRefs[i];
        auto satisfyingMemberDeclRef = satisfyingMemberDeclRefs[i];

        if (auto requiredTypeParamDeclRef = requiredMemberDeclRef.as<GenericTypeParamDecl>())
        {
            auto satisfyingTypeParamDeclRef = satisfyingMemberDeclRef.as<GenericTypeParamDecl>();
            SLANG_ASSERT(satisfyingTypeParamDeclRef);
            auto satisfyingType = DeclRefType::create(m_astBuilder, satisfyingTypeParamDeclRef);

            requiredSubstArgs.add(satisfyingType);
        }
        else if (auto requiredValueParamDeclRef = requiredMemberDeclRef.as<GenericValueParamDecl>())
        {
            auto satisfyingValueParamDeclRef = satisfyingMemberDeclRef.as<GenericValueParamDecl>();
            SLANG_ASSERT(satisfyingValueParamDeclRef);

            auto satisfyingVal = m_astBuilder->getOrCreate<GenericParamIntVal>(
                requiredValueParamDeclRef.getDecl()->getType(),
                satisfyingValueParamDeclRef);
            satisfyingVal->getDeclRef() = satisfyingValueParamDeclRef;

            requiredSubstArgs.add(satisfyingVal);
        }
    }
    for (Index i = 0; i < memberCount; i++)
    {
        auto requiredMemberDeclRef = requiredMemberDeclRefs[i];
        auto satisfyingMemberDeclRef = satisfyingMemberDeclRefs[i];

        if (auto requiredConstraintDeclRef = requiredMemberDeclRef.as<GenericTypeConstraintDecl>())
        {
            auto satisfyingConstraintDeclRef =
                satisfyingMemberDeclRef.as<GenericTypeConstraintDecl>();
            SLANG_ASSERT(satisfyingConstraintDeclRef);

            auto satisfyingWitness = m_astBuilder->getDeclaredSubtypeWitness(
                getSub(m_astBuilder, satisfyingConstraintDeclRef),
                getSup(m_astBuilder, satisfyingConstraintDeclRef),
                satisfyingConstraintDeclRef);

            requiredSubstArgs.add(satisfyingWitness);
        }
    }

    // Now that we have computed a set of specialization arguments that will
    // specialize the generic requirement at the type parameters of the satisfying
    // generic, we can construct a reference to that declaration and re-run some
    // of the earlier checking logic with more type information usable.
    //
    auto specializedRequiredGenericInnerDeclRef = m_astBuilder->getGenericAppDeclRef(
        requiredGenericDeclRef,
        requiredSubstArgs.getArrayView());
    for (Index i = 0; i < memberCount; i++)
    {
        auto requiredMemberDeclRef = requiredMemberDeclRefs[i];
        auto satisfyingMemberDeclRef = satisfyingMemberDeclRefs[i];

        if (auto requiredTypeParamDeclRef = requiredMemberDeclRef.as<GenericTypeParamDecl>())
        {
            [[maybe_unused]] auto satisfyingTypeParamDeclRef =
                satisfyingMemberDeclRef.as<GenericTypeParamDecl>();
            SLANG_ASSERT(satisfyingTypeParamDeclRef);

            // There are no additional checks we need to make on plain old
            // type parameters at this point.
            //
            // TODO: If we ever support having type parameters of higher kinds,
            // then this is possibly where we'd want to check that the kinds of
            // the two parameters match.
            //
        }
        else if (auto requiredValueParamDeclRef = requiredMemberDeclRef.as<GenericValueParamDecl>())
        {
            auto satisfyingValueParamDeclRef = satisfyingMemberDeclRef.as<GenericValueParamDecl>();
            SLANG_ASSERT(satisfyingValueParamDeclRef);

            // For a generic value parameter, we need to check that the required
            // and satisfying declaration both agree on the type of the parameter.
            //
            auto requiredParamType = getType(m_astBuilder, requiredValueParamDeclRef);
            auto satisfyingParamType = getType(m_astBuilder, satisfyingValueParamDeclRef);
            if (!satisfyingParamType->equals(requiredParamType))
                return false;
        }
        else if (
            auto requiredConstraintDeclRef = requiredMemberDeclRef.as<GenericTypeConstraintDecl>())
        {
            auto satisfyingConstraintDeclRef =
                satisfyingMemberDeclRef.as<GenericTypeConstraintDecl>();
            SLANG_ASSERT(satisfyingConstraintDeclRef);

            // For a generic constraint parameter, we need to check that the sub-type
            // and super-type in the constraint both match.
            //
            // In current code the sub type will always be one of the generic type parameters,
            // and the super-type will always be an interface, but there should be no
            // need to make use of those additional details here.
            auto specializedRequiredConstraintDeclRef = m_astBuilder
                                                            ->getGenericAppDeclRef(
                                                                requiredGenericDeclRef,
                                                                requiredSubstArgs.getArrayView(),
                                                                requiredConstraintDeclRef.getDecl())
                                                            .as<GenericTypeConstraintDecl>();
            auto requiredSubType = getSub(m_astBuilder, specializedRequiredConstraintDeclRef);
            auto satisfyingSubType = getSub(m_astBuilder, satisfyingConstraintDeclRef);
            if (!satisfyingSubType->equals(requiredSubType))
                return false;

            auto requiredSuperType = getSup(m_astBuilder, specializedRequiredConstraintDeclRef);
            auto satisfyingSuperType = getSup(m_astBuilder, satisfyingConstraintDeclRef);
            if (!satisfyingSuperType->equals(requiredSuperType))
                return false;
        }
    }

    // Note: the above logic really only applies to the case of an exact match on signature,
    // even down to the way that constraints were declared. We could potentially be more
    // relaxed by taking advantage of the way that various different generic signatures will
    // actually lower to the same IR generic signature.
    //
    // In theory, all we really care about when it comes to constraints is that the constraints
    // on the required and satisfying declaration are *equivalent*.
    //
    // More generally, a satisfying generic could actually provide *looser* constraints and
    // still work; all that matters is that it can be instantiated at any argument values/types
    // that are valid for the requirement.
    //
    // We leave both of those issues up to the synthesis path: if we do not find a member that
    // provides an exact match, then the compiler should try to synthesize one that is an exact
    // match and makes use of existing declarations that might have require defaulting of arguments
    // or type conversations to fit.

    // Once we've validated that the generic signatures are in an exact match, and devised type
    // arguments for the requirement to make the two align, we can recursively check the inner
    // declaration (whatever it is) for an exact match.
    //
    return doesMemberSatisfyRequirement(
        m_astBuilder->getMemberDeclRef(
            satisfyingGenericDeclRef,
            getInner(satisfyingGenericDeclRef)),
        specializedRequiredGenericInnerDeclRef,
        witnessTable);
}

bool SemanticsVisitor::doesTypeSatisfyAssociatedTypeConstraintRequirement(
    Type* satisfyingType,
    DeclRef<AssocTypeDecl> requiredAssociatedTypeDeclRef,
    RefPtr<WitnessTable> witnessTable)
{
    SLANG_UNUSED(satisfyingType);

    // We will enumerate the type constraints placed on the
    // associated type and see if they can be satisfied.
    //
    bool conformance = true;
    Val* witness = nullptr;
    for (auto requiredConstraintDeclRef :
         getMembersOfType<GenericTypeConstraintDecl>(m_astBuilder, requiredAssociatedTypeDeclRef))
    {
        // Grab the type we expect to conform to from the constraint.
        auto requiredSuperType = getSup(m_astBuilder, requiredConstraintDeclRef);

        auto subType = getSub(m_astBuilder, requiredConstraintDeclRef);

        // Perform a search for a witness to the subtype relationship.
        witness = tryGetSubtypeWitness(subType, requiredSuperType);
        if (witness)
        {
            auto genConstraint = as<GenericTypeConstraintDecl>(requiredConstraintDeclRef.getDecl());
            if (genConstraint && genConstraint->isEqualityConstraint &&
                !isTypeEqualityWitness(witness))
                witness = nullptr;
        }
        if (witness)
        {
            // If a subtype witness was found, then the conformance
            // appears to hold, and we can satisfy that requirement.
            witnessTable->add(requiredConstraintDeclRef.getDecl(), RequirementWitness(witness));
        }
        else
        {
            // If a witness couldn't be found, then the conformance
            // seems like it will fail.
            conformance = false;
        }
    }
    return conformance;
}

bool SemanticsVisitor::doesTypeSatisfyAssociatedTypeRequirement(
    Type* satisfyingType,
    DeclRef<AssocTypeDecl> requiredAssociatedTypeDeclRef,
    RefPtr<WitnessTable> witnessTable)
{
    if (auto declRefType = as<DeclRefType>(satisfyingType))
    {
        // If we are seeing a placeholder that awaits synthesis, return false now to trigger
        // auto synthesis.
        if (declRefType->getDeclRef().getDecl()->hasModifier<ToBeSynthesizedModifier>())
            return false;
    }

    // Register the satisfying type to the witness table
    // before checking the constraints, since the subtype of
    // the constraints maybe referencing the satisfying type via
    // witness lookups.
    auto requirementWitness = RequirementWitness(satisfyingType->getCanonicalType());
    witnessTable->m_requirementDictionary[requiredAssociatedTypeDeclRef.getDecl()] =
        requirementWitness;

    // We need to confirm that the chosen type `satisfyingType`,
    // meets all the constraints placed on the associated type
    // requirement `requiredAssociatedTypeDeclRef`.
    //
    // We will enumerate the type constraints placed on the
    // associated type and see if they can be satisfied.
    //
    bool conformance = doesTypeSatisfyAssociatedTypeConstraintRequirement(
        satisfyingType,
        requiredAssociatedTypeDeclRef,
        witnessTable);

    // TODO: if any conformance check failed, we should probably include
    // that in an error message produced about not satisfying the requirement.

    if (!conformance)
    {
        witnessTable->m_requirementDictionary.remove(requiredAssociatedTypeDeclRef.getDecl());
    }

    return conformance;
}

bool SemanticsVisitor::doesMemberSatisfyRequirement(
    DeclRef<Decl> memberDeclRef,
    DeclRef<Decl> requiredMemberDeclRef,
    RefPtr<WitnessTable> witnessTable)
{
    // Sanity check: if are checking whether a type `T`
    // implements, say, `IFoo::bar` and lookup of `bar`
    // in type `T` yielded `IFoo::bar`, then that shouldn't
    // be treated as a valid satisfaction of the requirement.
    //
    // TODO: Ideally this check should be comparing the `DeclRef`s
    // and not just the `Decl`s, but we currently don't get exactly
    // the same substitutions when we see the inherited `IFoo::bar`.
    //
    if (memberDeclRef.getDecl() == requiredMemberDeclRef.getDecl())
        return false;

    // At a high level, we want to check that the
    // `memberDecl` and the `requiredMemberDeclRef`
    // have the same AST node class, and then also
    // check that their signatures match.
    //
    // There are a bunch of detailed decisions that
    // have to be made, though, because we might, e.g.,
    // allow a function with more general parameter
    // types to satisfy a requirement with more
    // specific parameter types.
    //
    // If we ever allow for "property" declarations,
    // then we would probably need to allow an
    // ordinary field to satisfy a property requirement.
    //
    // An associated type requirement should be allowed
    // to be satisfied by any type declaration:
    // a typedef, a `struct`, etc.
    //
    if (auto memberFuncDecl = memberDeclRef.as<FuncDecl>())
    {
        if (auto requiredFuncDeclRef = requiredMemberDeclRef.as<FuncDecl>())
        {
            // Check signature match.
            return doesSignatureMatchRequirement(memberFuncDecl, requiredFuncDeclRef, witnessTable);
        }
    }
    else if (auto memberInitDecl = memberDeclRef.as<ConstructorDecl>())
    {
        if (auto requiredInitDecl = requiredMemberDeclRef.as<ConstructorDecl>())
        {
            // Check signature match.
            return doesSignatureMatchRequirement(memberInitDecl, requiredInitDecl, witnessTable);
        }
    }
    else if (auto genDecl = memberDeclRef.as<GenericDecl>())
    {
        // For a generic member, we will check if it can satisfy
        // a generic requirement in the interface.
        //
        // TODO: we could also conceivably check that the generic
        // could be *specialized* to satisfy the requirement,
        // and then install a specialization of the generic into
        // the witness table. Actually doing this would seem
        // to require performing something akin to overload
        // resolution as part of requirement satisfaction.
        //
        if (auto requiredGenDeclRef = requiredMemberDeclRef.as<GenericDecl>())
        {
            return doesGenericSignatureMatchRequirement(genDecl, requiredGenDeclRef, witnessTable);
        }
    }
    else if (auto subAggTypeDeclRef = memberDeclRef.as<AggTypeDecl>())
    {
        if (auto requiredTypeDeclRef = requiredMemberDeclRef.as<AssocTypeDecl>())
        {
            ensureDecl(subAggTypeDeclRef, DeclCheckState::CanUseAsType);

            auto satisfyingType = DeclRefType::create(m_astBuilder, subAggTypeDeclRef);
            return doesTypeSatisfyAssociatedTypeRequirement(
                satisfyingType,
                requiredTypeDeclRef,
                witnessTable);
        }
    }
    else if (auto typedefDeclRef = memberDeclRef.as<TypeDefDecl>())
    {
        // this is a type-def decl in an aggregate type
        // check if the specified type satisfies the constraints defined by the associated type
        if (auto requiredTypeDeclRef = requiredMemberDeclRef.as<AssocTypeDecl>())
        {
            ensureDecl(typedefDeclRef, DeclCheckState::ReadyForLookup);

            auto satisfyingType = getNamedType(m_astBuilder, typedefDeclRef);
            return doesTypeSatisfyAssociatedTypeRequirement(
                satisfyingType,
                requiredTypeDeclRef,
                witnessTable);
        }
    }
    else if (auto propertyDeclRef = memberDeclRef.as<PropertyDecl>())
    {
        if (auto requiredPropertyDeclRef = requiredMemberDeclRef.as<PropertyDecl>())
        {
            ensureDecl(propertyDeclRef, DeclCheckState::CanUseFuncSignature);
            return doesPropertyMatchRequirement(
                propertyDeclRef,
                requiredPropertyDeclRef,
                witnessTable);
        }
    }
    else if (auto varDeclRef = memberDeclRef.as<VarDeclBase>())
    {
        if (auto requiredVarDeclRef = requiredMemberDeclRef.as<VarDeclBase>())
        {
            ensureDecl(varDeclRef, DeclCheckState::SignatureChecked);
            return doesVarMatchRequirement(varDeclRef, requiredVarDeclRef, witnessTable);
        }
    }
    else if (auto subscriptDeclRef = memberDeclRef.as<SubscriptDecl>())
    {
        if (auto requiredSubscriptDeclRef = requiredMemberDeclRef.as<SubscriptDecl>())
        {
            ensureDecl(subscriptDeclRef, DeclCheckState::CanUseFuncSignature);
            return doesSubscriptMatchRequirement(
                subscriptDeclRef,
                requiredSubscriptDeclRef,
                witnessTable);
        }
    }
    // Default: just assume that thing aren't being satisfied.
    return false;
}

GenericDecl* SemanticsVisitor::synthesizeGenericSignatureForRequirementWitness(
    ConformanceCheckingContext* context,
    DeclRef<GenericDecl> requiredMemberDeclRef,
    List<Expr*>& synArgs,
    List<Expr*>& synGenericArgs,
    ThisExpr*& synThis)
{
    auto synGenericDecl = m_astBuilder->create<GenericDecl>();
    synGenericDecl->parentDecl = context->parentDecl;
    synGenericDecl->ownedScope = m_astBuilder->create<Scope>();
    synGenericDecl->ownedScope->containerDecl = synGenericDecl;
    synGenericDecl->ownedScope->parent = getScope(context->parentDecl);

    // For now our synthesized method will use the name and source
    // location of the requirement we are trying to satisfy.
    //
    // TODO: as it stands right now our syntesized method will
    // get a mangled name, which we don't actually want. Leaving
    // out the name here doesn't help matters, because then *all*
    // snthesized methods on a given type would share the same
    // mangled name!
    //
    synGenericDecl->nameAndLoc = requiredMemberDeclRef.getDecl()->nameAndLoc;
    if (synGenericDecl->nameAndLoc.name)
    {
        synGenericDecl->nameAndLoc.name =
            getSession()->getNameObj("$__syn_" + synGenericDecl->nameAndLoc.name->text);
    }

    // Dictionary to map from the original type parameters to the synthesized ones.
    Dictionary<Decl*, Decl*> mapOrigToSynTypeParams;

    // Our synthesized method will have parameters matching the names
    // and types of those on the requirement, and it will use expressions
    // that reference those parametesr as arguments for the call expresison
    // that makes up the body.
    //
    for (auto member : requiredMemberDeclRef.getDecl()->members)
    {
        if (auto typeParamDeclBase = as<GenericTypeParamDeclBase>(member))
        {
            auto synTypeParamDeclBase = (GenericTypeParamDeclBase*)m_astBuilder->createByNodeType(
                typeParamDeclBase->astNodeType);
            synTypeParamDeclBase->nameAndLoc = typeParamDeclBase->getNameAndLoc();
            synTypeParamDeclBase->parameterIndex = typeParamDeclBase->parameterIndex;
            synGenericDecl->addMember(synTypeParamDeclBase);

            // Note: we intentionally do not copy GenericTypeParamDecl::initType here,
            // because initType maybe dependent on the original type parameters,
            // and if we copy we must also substitute all the original type parameters with the
            // synthesized ones. It shouldn't be required for the implementing declaration to define
            // initType anyways, so we'll just save ourselves from the trouble.
            //

            mapOrigToSynTypeParams.add(typeParamDeclBase, synTypeParamDeclBase);

            // Construct a DeclRefExpr from the type parameter.
            auto synTypeParamDeclRef = makeDeclRef(synTypeParamDeclBase);

            auto synTypeParamDeclRefExpr = m_astBuilder->create<VarExpr>();
            synTypeParamDeclRefExpr->declRef = synTypeParamDeclRef;
            synTypeParamDeclRefExpr->type =
                getTypeForDeclRef(m_astBuilder, synTypeParamDeclRef, SourceLoc());

            synGenericArgs.add(synTypeParamDeclRefExpr);
        }
        else if (auto valParamDecl = as<GenericValueParamDecl>(member))
        {
            auto synValParamDecl = m_astBuilder->create<GenericValueParamDecl>();
            synValParamDecl->nameAndLoc = valParamDecl->nameAndLoc;
            synGenericDecl->addMember(synValParamDecl);
            synValParamDecl->parameterIndex = valParamDecl->parameterIndex;
            synValParamDecl->type = valParamDecl->type;

            // Note: we intentionally do not copy GenericValueParamDecl::initExpr here,
            // because initType maybe dependent on the original type/value parameters,
            // and if we copy we must also substitute all the original type parameters with the
            // synthesized ones. It shouldn't be required for the implementing declaration to define
            // initType anyways, so we'll just save ourselves from the trouble.
            //

            mapOrigToSynTypeParams.add(valParamDecl, synGenericDecl);

            // Construct a DeclRefExpr from the value parameter.
            auto synValParamDeclRef = makeDeclRef(synValParamDecl);

            auto synValParamDeclRefExpr = m_astBuilder->create<VarExpr>();
            synValParamDeclRefExpr->declRef = synValParamDeclRef;
            synValParamDeclRefExpr->type = synValParamDecl->type.type;

            synGenericArgs.add(synValParamDeclRefExpr);
        }
    }

    // With all generic parameters in place, we can now form a partial substitution argument list
    // without taking into account all the generic constraints.

    // Given `requiredMemberDeclRef` that is `Lookup(ConcreteType:IFoo<int>, IFoo::bar)`, we can now
    // form a partial specialized declref to `IFoo<int>::bar` with substitution args comming
    // from the synthesized generic decl, i.e. we want to form:
    // `Lookup(ConcreteType:IFoo<int>, IFoo::bar)<UImpl>` where `UImpl` is a synthesized generic
    // parameter.
    //
    auto partialDefaultArgs = getDefaultSubstitutionArgs(m_astBuilder, this, synGenericDecl);
    DeclRef<CallableDecl> partiallySpecializedRequiredGenericDeclRef =
        m_astBuilder->getGenericAppDeclRef(requiredMemberDeclRef, partialDefaultArgs.getArrayView())
            .as<CallableDecl>();

    // With `partiallySpecializedRequiredGenericDeclRef`, we can obtain the right specialized types
    // from the original requirement decl. For example, we can simply apply declref substituion on
    // the original type constraint `U:IDerived` to get `UImpl : IDerived`.
    //
    for (auto member : requiredMemberDeclRef.getDecl()->members)
    {
        if (auto constraintDecl = as<GenericTypeConstraintDecl>(member))
        {
            auto synConstraintDecl = m_astBuilder->create<GenericTypeConstraintDecl>();
            synConstraintDecl->nameAndLoc = constraintDecl->getNameAndLoc();
            synConstraintDecl->parentDecl = synGenericDecl;

            // For generic constraint Sub : Sup, we need to substitute them with
            // synthesized generic parameters.
            //
            synConstraintDecl->sub = TypeExp((Type*)constraintDecl->sub.type->substitute(
                m_astBuilder,
                SubstitutionSet(partiallySpecializedRequiredGenericDeclRef)));
            synConstraintDecl->sup = TypeExp((Type*)constraintDecl->sup.type->substitute(
                m_astBuilder,
                SubstitutionSet(partiallySpecializedRequiredGenericDeclRef)));
            synGenericDecl->members.add(synConstraintDecl);
        }
    }

    // Override generic pointer to point to the original generic container.
    // This will create a substitution of the synthesized parameters for the
    // original parameters.
    //
    auto defaultArgs = getDefaultSubstitutionArgs(m_astBuilder, this, synGenericDecl);
    DeclRef<CallableDecl> requiredFuncDeclRef =
        m_astBuilder->getGenericAppDeclRef(requiredMemberDeclRef, defaultArgs.getArrayView())
            .as<CallableDecl>();

    SLANG_ASSERT(requiredFuncDeclRef);
    ConformanceCheckingContext subContext = *context;
    subContext.parentDecl = synGenericDecl;

    synGenericDecl->inner = synthesizeMethodSignatureForRequirementWitnessInner(
        &subContext,
        requiredFuncDeclRef,
        synArgs,
        synThis);
    return synGenericDecl;
}

void SemanticsVisitor::addModifiersToSynthesizedDecl(
    ConformanceCheckingContext* context,
    DeclRef<Decl> requiredMemberDeclRef,
    CallableDecl* synthesized,
    ThisExpr*& synThis)
{
    // Required interface methods can be `static` or non-`static`,
    // and non-`static` methods can be `[mutating]` or non-`[mutating]`.
    // All of these details affect how we introduce our `this` parameter,
    // if any.
    //
    if (requiredMemberDeclRef.getDecl()->hasModifier<HLSLStaticModifier>())
    {
        auto synStaticModifier = m_astBuilder->create<HLSLStaticModifier>();
        synthesized->modifiers.first = synStaticModifier;
    }
    else
    {
        // For a non-`static` requirement, we need a `this` parameter.
        //
        synThis = m_astBuilder->create<ThisExpr>();
        synThis->scope = synthesized->ownedScope;

        // The type of `this` in our method will be the type for
        // which we are synthesizing a conformance.
        //
        synThis->type.type = context->conformingType;

        if (requiredMemberDeclRef.getDecl()->hasModifier<MutatingAttribute>())
        {
            // If the interface requirement is `[mutating]` then our
            // synthesized method should be too, and also the `this`
            // parameter should be an l-value.
            //
            synThis->type.isLeftValue = true;

            auto synMutatingAttr = m_astBuilder->create<MutatingAttribute>();
            addModifier(synthesized, synMutatingAttr);
        }
        if (requiredMemberDeclRef.getDecl()->hasModifier<ConstRefAttribute>())
        {
            // If the interface requirement is `[constref]` then our
            // synthesized method should be too.
            //
            auto synConstRefAttr = m_astBuilder->create<ConstRefAttribute>();
            addModifier(synthesized, synConstRefAttr);
        }
        if (requiredMemberDeclRef.getDecl()->hasModifier<RefAttribute>())
        {
            // If the interface requirement is `[ref]` then our
            // synthesized method should be too.
            //
            synThis->type.isLeftValue = true;

            auto synConstRefAttr = m_astBuilder->create<RefAttribute>();
            addModifier(synthesized, synConstRefAttr);
        }
        if (requiredMemberDeclRef.getDecl()->hasModifier<NoDiffThisAttribute>())
        {
            auto noDiffThisAttr = m_astBuilder->create<NoDiffThisAttribute>();
            addModifier(synthesized, noDiffThisAttr);
        }
    }
    if (requiredMemberDeclRef.getDecl()->hasModifier<ForwardDifferentiableAttribute>())
    {
        auto attr = m_astBuilder->create<ForwardDifferentiableAttribute>();
        addModifier(synthesized, attr);
    }
    if (requiredMemberDeclRef.getDecl()->hasModifier<BackwardDifferentiableAttribute>())
    {
        auto attr = m_astBuilder->create<BackwardDifferentiableAttribute>();
        addModifier(synthesized, attr);
    }
    // The visibility of synthesized decl should be the min of the parent decl and the requirement.
    if (requiredMemberDeclRef.getDecl()->findModifier<VisibilityModifier>())
    {
        auto requirementVisibility = getDeclVisibility(requiredMemberDeclRef.getDecl());
        auto thisVisibility = getDeclVisibility(context->parentDecl);
        auto visibility = Math::Min(thisVisibility, requirementVisibility);
        addVisibilityModifier(synthesized, visibility);
    }
}

void SemanticsVisitor::addRequiredParamsToSynthesizedDecl(
    DeclRef<CallableDecl> requirement,
    CallableDecl* synthesized,
    List<Expr*>& synArgs)
{
    // Our synthesized method will have parameters matching the names
    // and types of those on the requirement, and it will use expressions
    // that reference those parameters as arguments for the call expresison
    // that makes up the body.
    //
    for (auto paramDeclRef : getParameters(m_astBuilder, requirement))
    {
        auto paramType = QualType(getType(m_astBuilder, paramDeclRef));

        // For each parameter of the requirement, we create a matching
        // parameter (same name and type) for the synthesized method.
        //
        auto synParamDecl = m_astBuilder->create<ParamDecl>();
        synParamDecl->nameAndLoc = paramDeclRef.getDecl()->nameAndLoc;
        synParamDecl->type.type = paramType.type;

        // We need to add the parameter as a child declaration of
        // the method we are building.
        //
        synthesized->addMember(synParamDecl);

        // Add modifiers
        paramType.isLeftValue = true;
        for (auto modifier : paramDeclRef.getDecl()->modifiers)
        {
            if (as<NoDiffModifier>(modifier))
            {
                auto noDiffModifier = m_astBuilder->create<NoDiffModifier>();
                noDiffModifier->keywordName = getSession()->getNameObj("no_diff");
                addModifier(synParamDecl, noDiffModifier);
            }
            else if (
                as<InOutModifier>(modifier) || as<OutModifier>(modifier) ||
                as<ConstRefModifier>(modifier) || as<RefModifier>(modifier))
            {
                auto clonedModifier =
                    (Modifier*)m_astBuilder->createByNodeType(modifier->astNodeType);
                clonedModifier->keywordName = modifier->keywordName;
                addModifier(synParamDecl, clonedModifier);
                if (as<ConstRefModifier>(modifier))
                    paramType.isLeftValue = false;
            }
        }

        // Create an expression that references the parameter for use in arguments.
        auto synArg = m_astBuilder->create<VarExpr>();
        synArg->declRef = makeDeclRef(synParamDecl);
        synArg->type = paramType;

        if (auto typePack = as<ConcreteTypePack>(paramType))
        {
            // If paramType is a concrete type pack, we want to expand it out into
            // individual arguments.
            for (Index i = 0; i < typePack->getTypeCount(); i++)
            {
                auto elementType = typePack->getElementType(i);
                auto synMemberExpr = m_astBuilder->create<SwizzleExpr>();
                synMemberExpr->base = synArg;
                synMemberExpr->elementIndices.add((uint32_t)i);
                synMemberExpr->type = elementType;
                synMemberExpr->type.isLeftValue = paramType.isLeftValue;
                synMemberExpr->checked = true;
                synArgs.add(synMemberExpr);
            }
        }
        else
        {
            // For ordinary non-pack paramters, we will use synArg directly to
            // referencing the parameter for the call in the function body.
            //
            synArgs.add(synArg);
        }
    }
}

CallableDecl* SemanticsVisitor::synthesizeMethodSignatureForRequirementWitness(
    ConformanceCheckingContext* context,
    DeclRef<CallableDecl> requiredMemberDeclRef,
    List<Expr*>& synArgs,
    ThisExpr*& synThis)
{
    if (auto genericDeclRef = as<GenericDecl>(requiredMemberDeclRef.getParent()))
    {
        List<Expr*> synGenericArgs;
        auto genericDecl = synthesizeGenericSignatureForRequirementWitness(
            context,
            genericDeclRef,
            synArgs,
            synGenericArgs,
            synThis);
        return (CallableDecl*)genericDecl->inner;
    }
    return synthesizeMethodSignatureForRequirementWitnessInner(
        context,
        requiredMemberDeclRef,
        synArgs,
        synThis);
}

CallableDecl* SemanticsVisitor::synthesizeMethodSignatureForRequirementWitnessInner(
    ConformanceCheckingContext* context,
    DeclRef<CallableDecl> requiredMemberDeclRef,
    List<Expr*>& synArgs,
    ThisExpr*& synThis)
{
    CallableDecl* synFuncDecl = as<CallableDecl>(
        m_astBuilder->createByNodeType(requiredMemberDeclRef.getDecl()->astNodeType));
    SLANG_ASSERT(synFuncDecl);

    synFuncDecl->ownedScope = m_astBuilder->create<Scope>();
    synFuncDecl->ownedScope->containerDecl = synFuncDecl;
    synFuncDecl->ownedScope->parent = getScope(context->parentDecl);
    synFuncDecl->parentDecl = context->parentDecl;

    // For now our synthesized method will use the name and source
    // location of the requirement we are trying to satisfy.
    //
    // TODO: as it stands right now our syntesized method will
    // get a mangled name, which we don't actually want. Leaving
    // out the name here doesn't help matters, because then *all*
    // snthesized methods on a given type would share the same
    // mangled name!
    //
    synFuncDecl->nameAndLoc = requiredMemberDeclRef.getDecl()->nameAndLoc;
    if (synFuncDecl->nameAndLoc.name)
    {
        synFuncDecl->nameAndLoc.name =
            getSession()->getNameObj("$__syn_" + synFuncDecl->nameAndLoc.name->text);
    }

    // The result type of our synthesized method will be the expected
    // result type from the interface requirement.
    //
    // TODO: This logic can/will run into problems if the return type
    // is an associated type.
    //
    // The ideal solution is that we should be solving for interface
    // conformance in two phases: a first phase to solve for how
    // associated types are satisfied, and then a second phase to solve
    // for how other requirements are satisfied (where we can substitute
    // in the associated type witnesses for the abstract associated
    // types as part of `requiredMemberDeclRef`).
    //
    // TODO: We should also double-check that this logic will work
    // with a method that returns `This`.
    //
    auto resultType = getResultType(m_astBuilder, requiredMemberDeclRef);
    synFuncDecl->returnType.type = resultType;

    addRequiredParamsToSynthesizedDecl(requiredMemberDeclRef, synFuncDecl, synArgs);
    addModifiersToSynthesizedDecl(context, requiredMemberDeclRef, synFuncDecl, synThis);

    return synFuncDecl;
}

void SemanticsVisitor::_addMethodWitness(
    WitnessTable* witnessTable,
    DeclRef<CallableDecl> requiredMemberDeclRef,
    DeclRef<CallableDecl> satisfyingMemberDeclRef)
{
    for (auto reqRefDecl :
         requiredMemberDeclRef.getDecl()->getMembersOfType<DerivativeRequirementReferenceDecl>())
    {
        if (auto fwdReq = as<ForwardDerivativeRequirementDecl>(reqRefDecl->referencedDecl))
        {
            ForwardDifferentiateVal* val =
                m_astBuilder->getOrCreate<ForwardDifferentiateVal>(satisfyingMemberDeclRef);
            witnessTable->add(fwdReq, RequirementWitness(val));
        }
        else if (auto bwdReq = as<BackwardDerivativeRequirementDecl>(reqRefDecl->referencedDecl))
        {
            DifferentiateVal* val =
                m_astBuilder->getOrCreate<BackwardDifferentiateVal>(satisfyingMemberDeclRef);
            witnessTable->add(bwdReq, RequirementWitness(val));
        }
    }
    witnessTable->add(requiredMemberDeclRef.getDecl(), RequirementWitness(satisfyingMemberDeclRef));
}

static bool isWrapperTypeDecl(Decl* decl)
{
    if (auto aggTypeDecl = as<AggTypeDecl>(decl))
    {
        if (aggTypeDecl->wrappedType)
            return true;
    }
    return false;
}

// Is it allowed to have an interface method parameter whose direction is `reqDir`, and an
// implementing method parameter whose direction is `implDir`?
//
static bool matchParamDirection(ParameterDirection implDir, ParameterDirection reqDir)
{
    // If the parameter directions match exactly, then we are good.
    if (implDir == reqDir)
        return true;
    // Otherwise, we only allow the cases where reqDir is `InOut` and implDir is `In` or `Out`.
    if (implDir == kParameterDirection_In && reqDir == kParameterDirection_InOut)
        return true;
    if (implDir == kParameterDirection_Out && reqDir == kParameterDirection_InOut)
        return true;
    return false;
}

static void removeNonStaticLookupItems(LookupResult& lookupResult)
{
    List<LookupResultItem> newItems;
    for (auto item : lookupResult)
    {
        if (item.declRef.getDecl()->hasModifier<HLSLStaticModifier>())
        {
            newItems.add(item);
        }
    }

    lookupResult.items = newItems;
    lookupResult.item = LookupResultItem();
    if (lookupResult.items.getCount() > 0)
    {
        lookupResult.item = lookupResult.items[0];
    }
}

bool SemanticsVisitor::trySynthesizeMethodRequirementWitness(
    ConformanceCheckingContext* context,
    LookupResult const& lookupResult,
    DeclRef<FuncDecl> requiredMemberDeclRef,
    RefPtr<WitnessTable> witnessTable)
{
    // The situation here is that the context of an inheritance
    // declaration didn't provide an exact match for a required
    // method. E.g.:
    //
    //      interface ICounter { [mutating] int increment(); }
    //      struct MyCounter : ICounter
    //      {
    //          [mutating] int increment(int val = 1) { ... }
    //      }
    //
    // It is clear in this case that the `MyCounter` type *can*
    // satisfy the signature required by `ICounter`, but it has
    // no explicit method declaration that is a perfect match.
    //
    // The approach in this function will be to construct a
    // synthesized method along the lines of:
    //
    //      struct MyCounter ...
    //      {
    //          ...
    //          [murtating] int synthesized()
    //          {
    //              return this.increment();
    //          }
    //      }
    //
    // That is, we construct a method with the exact signature
    // of the requirement (same parameter and result types),
    // and then provide it with a body that simple `return`s
    // the result of applying the desired requirement name
    // (`increment` in this case) to those parameters.
    //
    // If the synthesized method type-checks, then we can say
    // that the type must satisfy the requirement structurally,
    // even if there isn't an exact signature match. More
    // importantly, the method we just synthesized can be
    // used as a witness to the fact that the requirement is
    // satisfied.

    // With the big picture spelled out, we can settle into
    // the work of constructing our synthesized method.
    //

    bool isInWrapperType = isWrapperTypeDecl(context->parentDecl);

    // First, we check that the differentiabliity of the method matches the requirement,
    // and we don't attempt to synthesize a method if they don't match.
    if (!isInWrapperType && getShared()->getFuncDifferentiableLevel(
                                as<FunctionDeclBase>(lookupResult.item.declRef.getDecl())) <
                                getShared()->getFuncDifferentiableLevel(
                                    as<FunctionDeclBase>(requiredMemberDeclRef.getDecl())))
    {
        return false;
    }

    ThisExpr* synThis = nullptr;
    List<Expr*> synArgs;
    auto synFuncDecl = as<FunctionDeclBase>(synthesizeMethodSignatureForRequirementWitness(
        context,
        requiredMemberDeclRef,
        synArgs,
        synThis));

    auto resultType = synFuncDecl->returnType.type;

    // The body of our synthesized method is going to try to
    // make a call using the name of the method requirement (e.g.,
    // the name `increment` in our example at the top of this function).
    //
    // The caller already passed in a `LookupResult` that represents
    // an attempt to look up the given name in the type of `this`,
    // and we really just need to wrap that result up as an overloaded
    // expression.
    //
    auto baseOverloadedExpr = m_astBuilder->create<OverloadedExpr>();
    baseOverloadedExpr->name = requiredMemberDeclRef.getDecl()->getName();

    if (isInWrapperType)
    {
        auto aggTypeDecl = as<AggTypeDecl>(context->parentDecl);
        baseOverloadedExpr->lookupResult2 = lookUpMember(
            m_astBuilder,
            this,
            baseOverloadedExpr->name,
            aggTypeDecl->wrappedType.type,
            aggTypeDecl->ownedScope,
            LookupMask::Default,
            LookupOptions::IgnoreBaseInterfaces);
        addModifier(synFuncDecl, m_astBuilder->create<ForceInlineAttribute>());

        synFuncDecl->parentDecl = aggTypeDecl;
    }
    else
    {
        baseOverloadedExpr->lookupResult2 = lookupResult;
    }

    // Non-static methods cannot implement static methods, remove them.
    if (requiredMemberDeclRef.getDecl()->hasModifier<HLSLStaticModifier>())
    {
        removeNonStaticLookupItems(baseOverloadedExpr->lookupResult2);
    }

    // If `synThis` is non-null, then we will use it as the base of
    // the overloaded expression, so that we have an overloaded
    // member reference, and not just an overloaded reference to some
    // static definitions.
    //
    if (synThis)
    {
        if (isInWrapperType)
        {
            // If this is a wrapper type, then use the inner
            // object as the actual this parameter for the redirected
            // call.
            auto innerExpr = m_astBuilder->create<VarExpr>();
            innerExpr->scope = synThis->scope;
            innerExpr->name = getName("inner");
            baseOverloadedExpr->base = CheckExpr(innerExpr);
            SemanticsDeclBodyVisitor bodyVisitor(withParentFunc(synFuncDecl));
            bodyVisitor.maybeRegisterDifferentiableType(
                m_astBuilder,
                baseOverloadedExpr->base->type);
        }
        else
        {
            baseOverloadedExpr->base = synThis;
        }
    }


    // In order to know if our call is well-formed, we need to run
    // the semantic checking logic for overload resolution. If it
    // runs into an error, we don't want that being reported back
    // to the user as some kind of overload-resolution failure.
    //
    // In order to protect the user from whatever errors might
    // occur, we will perform the checking in the context of
    // a temporary diagnostic sink.
    //
    DiagnosticSink tempSink(getSourceManager(), nullptr);
    ExprLocalScope localScope;
    SemanticsVisitor subVisitor(
        withSink(&tempSink).withParentFunc(synFuncDecl).withExprLocalScope(&localScope));

    Expr* synBase = baseOverloadedExpr;

    // If the requirement is a generic decl, fill in all generic arguments explicitly.
    if (auto genericDeclRef = as<GenericDecl>(synFuncDecl->parentDecl))
    {
        auto genericAppExpr = m_astBuilder->create<GenericAppExpr>();
        genericAppExpr->functionExpr = synBase;
        for (auto member : genericDeclRef->members)
        {
            if (auto typeParamDecl = as<GenericTypeParamDeclBase>(member))
            {
                auto synTypeParamDeclRef = makeDeclRef(typeParamDecl);
                auto synTypeParamDeclRefExpr = m_astBuilder->create<VarExpr>();
                synTypeParamDeclRefExpr->declRef = synTypeParamDeclRef;
                synTypeParamDeclRefExpr->type =
                    getTypeForDeclRef(m_astBuilder, synTypeParamDeclRef, SourceLoc());
                genericAppExpr->arguments.add(synTypeParamDeclRefExpr);
            }
            else if (auto valParamDecl = as<GenericValueParamDecl>(member))
            {
                auto synValParamDeclRef = makeDeclRef(valParamDecl);
                auto synValParamDeclRefExpr = m_astBuilder->create<VarExpr>();
                synValParamDeclRefExpr->declRef = synValParamDeclRef;
                synValParamDeclRefExpr->type = getType(m_astBuilder, synValParamDeclRef);
                genericAppExpr->arguments.add(synValParamDeclRefExpr);
            }
        }
        synBase = subVisitor.checkGenericAppWithCheckedArgs(genericAppExpr);

        // If checking the generic app failed, we can't synthesize the witness.
        //
        if (tempSink.getErrorCount() != 0)
        {
            context->innerSink.diagnose(
                SourceLoc(),
                Diagnostics::genericSignatureDoesNotMatchRequirement,
                baseOverloadedExpr->name);
            return false;
        }
    }

    // We now have the reference to the overload group we plan to call,
    // and we already built up the argument list, so we can construct
    // an `InvokeExpr` that represents the call we want to make.
    //
    auto synCall = m_astBuilder->create<InvokeExpr>();
    synCall->functionExpr = synBase;
    synCall->arguments = synArgs;

    // With our temporary diagnostic sink soaking up any messages
    // from overload resolution, we can now try to resolve
    // the call to see what happens.
    //
    auto checkedCall = subVisitor.ResolveInvoke(synCall);

    // Of course, it is possible that the call went through fine,
    // but the result isn't of the type we expect/require,
    // so we also need to coerce the result of the call to
    // the expected type.
    //
    auto coercedCall = subVisitor.coerce(CoercionSite::Return, resultType, checkedCall);

    // If our overload resolution or type coercion failed,
    // then we have not been able to synthesize a witness
    // for the requirement.
    //
    // TODO: We might want to detect *why* overload resolution
    // or type coercion failed, and report errors accordingly.
    //
    // More detailed diagnostics could help users understand
    // what they did wrong, e.g.:
    //
    // * "We tried to use `foo(int)` but the interface requires `foo(String)`
    //
    // * "You have two methods that can apply as `bar()` and we couldn't tell which one you meant
    //
    // For now we just bail out here and rely on the caller to
    // diagnose a generic "failed to satisfying requirement" error.
    //
    if (tempSink.getErrorCount() != 0)
    {
        context->innerSink.diagnose(
            SourceLoc(),
            Diagnostics::cannotResolveOverloadForMethodRequirement,
            baseOverloadedExpr->name);
        return false;
    }

    // If we were able to type-check the call, we also need to make
    // sure that the resolved callee member has consistent parameter
    // direction as the requirement method.
    //
    // For example, if there is a requirement:
    // ```
    // interface IFoo { void method(out int x); }
    // ```
    // and a type:
    // ```
    // struct X : IFoo { void method(int x) { ... } }
    // ```
    // After we synthesize:
    // ```
    // void X::synthesized_method(out int x) { this.method(x); }
    // ```
    // The synthesized method will pass all type check just fine,
    // but we don't want to allow this method to be used as a witness
    // for the requirement due to inconsistent parameter direction.
    // So let's check for this now.
    //
    if (auto checkedInvoke = as<InvokeExpr>(checkedCall))
    {
        if (auto declRefExpr = as<DeclRefExpr>(checkedInvoke->functionExpr))
        {
            if (auto callee = as<CallableDecl>(declRefExpr->declRef))
            {
                auto synParams = synFuncDecl->getParameters();
                auto calleeParams = callee.getDecl()->getParameters();
                auto synParamIter = synParams.begin();
                auto calleeParamIter = calleeParams.begin();
                for (; synParamIter != synParams.end() && calleeParamIter != calleeParams.end();
                     ++synParamIter, ++calleeParamIter)
                {
                    auto synParam = *synParamIter;
                    auto calleeParam = *calleeParamIter;
                    if (!matchParamDirection(
                            getParameterDirection(calleeParam),
                            getParameterDirection(synParam)))
                    {
                        context->innerSink.diagnose(
                            calleeParam,
                            Diagnostics::parameterDirectionDoesNotMatchRequirement,
                            calleeParam,
                            getParameterDirection(calleeParam),
                            getParameterDirection(synParam));
                        return false;
                    }
                }
            }
        }
    }

    // We've already created the outer declaration (including its
    // parameters), and the inner expression, so the main work
    // that is left is defining the body of the new function,
    // which comprises a single `return` statement.
    //
    auto synReturn = m_astBuilder->create<ReturnStmt>();
    synReturn->expression = coercedCall;

    synFuncDecl->body = synReturn;

    // Note: we set the parent of the synthesized declaration
    // to the parent of the inheritance declaration being
    // validated (which is either a type declaration or
    // an `extension`), but we do *not* add the syntehsized
    // declaration to the list of child declarations at
    // this point.
    //
    // The synthesized decl already has its parent set to
    // the current parent decl, so we don't need more actions
    // to wire it up to the AST hierarchy.
    //
    // By leaving the synthesized declaration off of the list
    // of members, we ensure that it doesn't get found
    // by lookup (e.g., in a module that `import`s this type).
    // Unfortunately, we may also break invariants in other parts
    // of the code if they assume that all declarations have
    // to appear in the parent/child hierarchy of the module.
    //
    // TODO: We may need to properly wire the synthesized
    // declaration into the hierarchy, but then attach a modifier
    // to it to indicate that it should be ignored by things like lookup.
    //

    // If the synthesized func is differentiable, make sure to populate its
    // differential type dictionary.
    SemanticsDeclBodyVisitor bodyVisitor(withParentFunc(synFuncDecl));
    bodyVisitor.registerDifferentiableTypesForFunc(synFuncDecl);

    // Once our synthesized declaration is complete, we need
    // to install it as the witness that satifies the given
    // requirement.
    //
    // Subsequent code generation should not be able to tell the
    // difference between our synthetic method and a hand-written
    // one with the same behavior.
    //
    auto containerDecl = getParentDecl(synFuncDecl);
    auto containerDeclRef = getDefaultDeclRef(containerDecl);
    auto synDeclRef = m_astBuilder->getMemberDeclRef(containerDeclRef, synFuncDecl);
    _addMethodWitness(witnessTable, requiredMemberDeclRef, synDeclRef);
    return true;
}

bool SemanticsVisitor::trySynthesizeConstructorRequirementWitness(
    ConformanceCheckingContext* context,
    LookupResult const& satisfyingMemberLookupResult,
    DeclRef<ConstructorDecl> requiredMemberDeclRef,
    RefPtr<WitnessTable> witnessTable)
{
    SLANG_UNUSED(satisfyingMemberLookupResult);

    if (as<EnumDecl>(context->parentDecl))
    {
        if (auto builtinRequirement =
                requiredMemberDeclRef.getDecl()->findModifier<BuiltinRequirementModifier>())
        {
            return trySynthesizeEnumTypeMethodRequirementWitness(
                context,
                requiredMemberDeclRef,
                witnessTable,
                builtinRequirement->kind);
        }
    }

    bool isDefaultInitializableType = requiredMemberDeclRef.getParent() ==
                                      getASTBuilder()->getDefaultInitializableTypeInterfaceDecl();
    bool isInWrapperType = isWrapperTypeDecl(context->parentDecl);
    if (!isInWrapperType && !isDefaultInitializableType && !satisfyingMemberLookupResult.isValid())
    {
        return false;
    }

    List<Expr*> synArgs;
    ThisExpr* synThis = nullptr;

    auto ctorDecl = (ConstructorDecl*)synthesizeMethodSignatureForRequirementWitness(
        context,
        requiredMemberDeclRef,
        synArgs,
        synThis);
    ctorDecl->loc = context->parentDecl->loc;
    ctorDecl->closingSourceLoc = ctorDecl->loc;
    auto ctorName = getName("$init");
    ctorDecl->nameAndLoc.name = ctorName;
    ctorDecl->nameAndLoc.loc = context->parentDecl->loc;

    auto seqStmt = m_astBuilder->create<SeqStmt>();
    ctorDecl->body = seqStmt;


    if (isInWrapperType)
    {
        SemanticsDeclBodyVisitor bodyVisitor(withParentFunc(ctorDecl));
        bodyVisitor.maybeRegisterDifferentiableType(m_astBuilder, context->conformingType);

        for (auto member : context->parentDecl->members)
        {
            if (auto varDecl = as<VarDeclBase>(member))
            {
                auto varExpr = m_astBuilder->create<VarExpr>();
                varExpr->scope = ctorDecl->ownedScope;
                varExpr->name = varDecl->getName();
                auto checkedVarExpr = CheckTerm(varExpr);
                if (!checkedVarExpr)
                    return false;
                if (as<ErrorType>(checkedVarExpr->type.type))
                    return false;
                auto assign = m_astBuilder->create<AssignExpr>();
                assign->left = checkedVarExpr;
                auto temp = m_astBuilder->create<InvokeExpr>();
                auto lookupResult = lookUpMember(
                    m_astBuilder,
                    this,
                    ctorName,
                    varDecl->type.type,
                    ctorDecl->ownedScope,
                    LookupMask::Function,
                    LookupOptions::IgnoreBaseInterfaces);
                temp->functionExpr = createLookupResultExpr(
                    ctorName,
                    lookupResult,
                    nullptr,
                    context->parentDecl->loc,
                    nullptr);
                temp->arguments.addRange(synArgs);
                auto resolvedVar = ResolveInvoke(temp);
                if (!resolvedVar)
                    return false;
                assign->right = resolvedVar;
                assign->type = m_astBuilder->getVoidType();
                bodyVisitor.maybeRegisterDifferentiableType(m_astBuilder, varDecl->type.type);

                auto stmt = m_astBuilder->create<ExpressionStmt>();
                stmt->expression = assign;
                seqStmt->stmts.add(stmt);
                break;
            }
        }
    }
    else if (synArgs.getCount())
    {
        // The body of our synthesized method is going to try to
        // make a ctor call with the specified arguments (e.g.,
        // the name `increment` in our example at the top of this function).
        //
        auto synBase = m_astBuilder->create<OverloadedExpr>();
        synBase->name = requiredMemberDeclRef.getDecl()->getName();

        synBase->lookupResult2 = satisfyingMemberLookupResult;

        // We now have the reference to the overload group we plan to call,
        // and we already built up the argument list, so we can construct
        // an `InvokeExpr` that represents the call we want to make.
        //
        auto synCall = m_astBuilder->create<InvokeExpr>();
        synCall->functionExpr = synBase;
        synCall->arguments = synArgs;

        // In order to know if our call is well-formed, we need to run
        // the semantic checking logic for overload resolution. If it
        // runs into an error, we don't want that being reported back
        // to the user as some kind of overload-resolution failure.
        //
        // In order to protect the user from whatever errors might
        // occur, we will perform the checking in the context of
        // a temporary diagnostic sink.
        //
        DiagnosticSink tempSink(getSourceManager(), nullptr);
        ExprLocalScope localScope;
        SemanticsVisitor subVisitor(
            withSink(&tempSink).withParentFunc(ctorDecl).withExprLocalScope(&localScope));

        // With our temporary diagnostic sink soaking up any messages
        // from overload resolution, we can now try to resolve
        // the call to see what happens.
        //
        auto checkedCall = subVisitor.ResolveInvoke(synCall);

        // If any error occurs during overload resolution, we can't synthesize the witness.
        if (tempSink.getErrorCount() != 0)
            return false;

        // If we were able to type-check the call, then we should
        // be able to finish construction of a suitable ctor witness,
        // by emitting `this = resolvedCtorCall()`.
        //
        AssignExpr* assignExpr = m_astBuilder->create<AssignExpr>();
        assignExpr->left = synThis;
        assignExpr->right = checkedCall;
        assignExpr->type = m_astBuilder->getVoidType();
        ExpressionStmt* exprStmt = m_astBuilder->create<ExpressionStmt>();
        exprStmt->expression = assignExpr;
        seqStmt->stmts.add(exprStmt);
    }

    if (isDefaultInitializableType)
        context->parentDecl->addMember(ctorDecl);

    auto containerDecl = getParentDecl(ctorDecl);
    auto containerDeclRef = getDefaultDeclRef(containerDecl);
    auto synDeclRef = m_astBuilder->getMemberDeclRef(containerDeclRef, ctorDecl);
    _addMethodWitness(witnessTable, requiredMemberDeclRef, synDeclRef);

    return true;
}

bool SemanticsVisitor::trySynthesizePropertyRequirementWitness(
    ConformanceCheckingContext* context,
    LookupResult const& lookupResult,
    DeclRef<PropertyDecl> requiredMemberDeclRef,
    RefPtr<WitnessTable> witnessTable)
{
    if (isWrapperTypeDecl(context->parentDecl))
        return trySynthesizeWrapperTypePropertyRequirementWitness(
            context,
            requiredMemberDeclRef,
            witnessTable);

    // The situation here is that the context of an inheritance
    // declaration didn't provide an exact match for a required
    // property. E.g.:
    //
    //      interface ICell { property value : int { get; set; } }
    //      struct MyCell : ICell
    //      {
    //          int value;
    //      }
    //
    // It is clear in this case that the `MyCell` type *can*
    // satisfy the signature required by `ICell`, but it has
    // no explicit `property` declaration, and instead just
    // a field with the right name and type.
    //
    // The approach in this function will be to construct a
    // synthesized `preoperty` along the lines of:
    //
    //      struct MyCounter ...
    //      {
    //          ...
    //          property value_synthesized : int
    //          {
    //              get { return this.value; }
    //              set(newValue) { this.value = newValue; }
    //          }
    //      }
    //
    // That is, we construct a `property` with the correct type
    // and with an accessor for each requirement, where the accesors
    // all try to read or write `this.value`.
    //
    // If those synthesized accessors all type-check, then we can
    // say that the type must satisfy the requirement structurally,
    // even if there isn't an exact signature match. More
    // importantly, the `property` we just synthesized can be
    // used as a witness to the fact that the requirement is
    // satisfied.
    //
    // The big-picture flow of the logic here is similar to
    // `trySynthesizeMethodRequirementWitness()` above, and we
    // will not comment this code as exhaustively, under the
    // assumption that readers of the code don't benefit from
    // having the exact same information stated twice.

    // With the introduction out of the way, let's get started
    // constructing a synthesized `PropertyDecl`.
    //
    auto synPropertyDecl = m_astBuilder->create<PropertyDecl>();

    // Synthesize the property name with a prefix to avoid name clashing.
    synPropertyDecl->nameAndLoc = requiredMemberDeclRef.getDecl()->nameAndLoc;
    synPropertyDecl->nameAndLoc.name =
        getName(String("$syn_property_") + getText(requiredMemberDeclRef.getName()));
    synPropertyDecl->parentDecl = context->parentDecl;
    synPropertyDecl->ownedScope = m_astBuilder->create<Scope>();
    synPropertyDecl->ownedScope->containerDecl = synPropertyDecl;
    synPropertyDecl->ownedScope->parent = context->parentDecl->ownedScope;

    // The type of our synthesized property can be derived from the
    // specialized declref to the requirement decl.
    //
    auto propertyType = getType(m_astBuilder, requiredMemberDeclRef);
    synPropertyDecl->type.type = propertyType;


    // We start by constructing an expression that represents
    // `this.name` where `name` is the name of the required
    // member. The caller already passed in a `lookupResult`
    // that should indicate all the declarations found by
    // looking up `name`, so we can start with that.
    //
    // TODO: Note that there are many cases for member lookup
    // that are not handled just by using `createLookupResultExpr`
    // because they are currently being special-cased (the most
    // notable cases are swizzles, as well as lookup of static
    // members in types).
    //
    // The main result here is that we will not be able to synthesize
    // a requirement for a built-in scalar/vector/matrix type to
    // a property with a name like `.xy` based on the presence of
    // swizles, even though it seems like such a thing should Just Work.
    //
    // If this is important we could "fix" it by allowing this
    // code to dispatch to the special-case logic used when doing
    // semantic checking for member expressions.
    //
    // Note: an alternative would be to change the core module declarations
    // of vectors/matrices so that all the swizzles are defined as
    // `property` declarations. There are some C++ math libraries (like GLM)
    // that implement swizzle syntax by a similar approach of statically
    // enumerating all possible swizzles. The down-side to such an
    // approach is that the combinatorial space of swizzles is quite
    // large (especially for matrices) so that supporting them via
    // general-purpose language features is unlikely to be as efficient
    // as special-case logic.
    //
    // We are going to synthesize an expression and then perform
    // semantic checking on it, but if there are semantic errors
    // we do *not* want to report them to the user as such, and
    // instead want the result to be a failure to synthesize
    // a valid witness.
    //
    // We will buffer up diagnostics into a temporary sink and
    // then throw them away when we are done.
    //
    // TODO: This behavior might be something we want to make
    // into a more fundamental capability of `DiagnosticSink` and/or
    // `SemanticsVisitor` so that code can push/pop the emission
    // of diagnostics more easily.
    //
    DiagnosticSink tempSink(getSourceManager(), nullptr);
    SemanticsVisitor subVisitor(withSink(&tempSink));

    // We need to create a `this` expression to be used in the body
    // of the synthesized accessor.
    //
    // TODO: if we ever allow `static` properties or subscripts,
    // we will need to handle that case here, by *not* creating
    // a `this` expression.
    //
    ThisExpr* synThis = m_astBuilder->create<ThisExpr>();
    synThis->scope = synPropertyDecl->ownedScope;

    // The type of `this` in our accessor will be the type for
    // which we are synthesizing a conformance.
    //
    synThis->type.type = context->conformingType;
    synThis->type.isLeftValue = true;
    auto synMemberRef = subVisitor.createLookupResultExpr(
        requiredMemberDeclRef.getName(),
        lookupResult,
        synThis,
        requiredMemberDeclRef.getLoc(),
        nullptr);
    synMemberRef->loc = requiredMemberDeclRef.getLoc();

    bool canSynAccessors = synthesizeAccessorRequirements(
        context,
        requiredMemberDeclRef,
        propertyType,
        synMemberRef,
        synPropertyDecl,
        witnessTable);
    if (!canSynAccessors)
        return false;


    // The visibility of synthesized decl should be the min of the parent decl and the requirement.
    if (requiredMemberDeclRef.getDecl()->findModifier<VisibilityModifier>())
    {
        auto requirementVisibility = getDeclVisibility(requiredMemberDeclRef.getDecl());
        auto thisVisibility = getDeclVisibility(context->parentDecl);
        auto visibility = Math::Min(thisVisibility, requirementVisibility);
        addVisibilityModifier(synPropertyDecl, visibility);
    }
    return true;
}

bool SemanticsVisitor::trySynthesizeWrapperTypePropertyRequirementWitness(
    ConformanceCheckingContext* context,
    DeclRef<PropertyDecl> requiredMemberDeclRef,
    RefPtr<WitnessTable> witnessTable)
{
    // We are synthesizing a property requirement for a wrapper type:
    //
    //      interface IFoo { property value : int { get; set; } }
    //      struct Foo : IFoo = FooImpl;
    //
    // We need to synthesize Foo to:
    //
    //      struct Foo : IFoo
    //      {
    //          FooImpl inner;
    //          property value : int { get { return inner.value; }
    //                                 set { inner.value = newValue; }
    //                               }
    //      }
    //
    // To do so, we need to grab the witness table of FooImpl:IFoo, and create
    // wrapper property in Foo that forwards the accessors to the inner object.
    //
    // We get started by constructing a synthesized `PropertyDecl`.
    //
    auto synPropertyDecl = m_astBuilder->create<PropertyDecl>();
    synPropertyDecl->parentDecl = context->parentDecl;

    // Synthesize the property name with a prefix to avoid name clashing.
    //
    synPropertyDecl->nameAndLoc = requiredMemberDeclRef.getDecl()->nameAndLoc;
    synPropertyDecl->nameAndLoc.name =
        getName(String("$syn_property_") + getText(requiredMemberDeclRef.getName()));

    // Find the witness that FooImpl : IFoo.
    auto aggTypeDecl = as<AggTypeDecl>(context->parentDecl);
    auto innerType = aggTypeDecl->wrappedType.type;
    DeclRef<Decl> innerProperty;
    auto innerWitness = tryGetSubtypeWitness(innerType, witnessTable->baseType);
    if (!innerWitness)
        return false;

    for (auto requiredAccessorDeclRef :
         getMembersOfType<AccessorDecl>(m_astBuilder, requiredMemberDeclRef))
    {
        auto innerEntry = tryLookUpRequirementWitness(
            m_astBuilder,
            innerWitness,
            requiredAccessorDeclRef.getDecl());
        if (innerEntry.getFlavor() != RequirementWitness::Flavor::declRef)
            return false;
        auto innerAccessorDeclRef = as<AccessorDecl>(innerEntry.getDeclRef());
        if (!innerAccessorDeclRef)
            return false;

        // The synthesized accessor will be an AST node of the same class as
        // the required accessor.
        //
        auto synAccessorDecl = (AccessorDecl*)m_astBuilder->createByNodeType(
            requiredAccessorDeclRef.getDecl()->astNodeType);
        synAccessorDecl->ownedScope = m_astBuilder->create<Scope>();
        synAccessorDecl->ownedScope->containerDecl = synAccessorDecl;
        synAccessorDecl->ownedScope->parent = getScope(context->parentDecl);

        // The return type should be the same as the inner object's accessor return type.
        //
        synAccessorDecl->returnType.type = getResultType(m_astBuilder, innerAccessorDeclRef);

        // Similarly, our synthesized accessor will have parameters matching those of the inner
        // accessor.
        //
        List<Expr*> synArgs;
        for (auto innerParamDeclRef : getParameters(m_astBuilder, innerAccessorDeclRef))
        {
            auto paramType = getType(m_astBuilder, innerParamDeclRef);

            // The synthesized parameter will ahve the same name and
            // type as the parameter of the requirement.
            //
            auto synParamDecl = m_astBuilder->create<ParamDecl>();
            synParamDecl->nameAndLoc = innerParamDeclRef.getDecl()->nameAndLoc;
            synParamDecl->type.type = paramType;

            // We need to add the parameter as a child declaration of
            // the accessor we are building.
            //
            synAccessorDecl->addMember(synParamDecl);

            // For each paramter, we will create an argument expression
            // to represent it in the body of the accessor.
            //
            auto synArg = m_astBuilder->create<VarExpr>();
            synArg->declRef = makeDeclRef(synParamDecl);
            synArg->type = paramType;
            synArgs.add(synArg);
        }

        // Now synthesize the body of the property accessor.
        // The body of the accessor will depend on the class of the accessor
        // we are synthesizing (e.g., `get` vs. `set`).
        //
        Stmt* synBodyStmt = nullptr;
        auto propertyRef = m_astBuilder->create<MemberExpr>();
        propertyRef->scope = synAccessorDecl->ownedScope;
        auto base = m_astBuilder->create<VarExpr>();
        base->scope = propertyRef->scope;
        base->name = getName("inner");
        propertyRef->baseExpression = base;
        innerProperty = innerAccessorDeclRef.getParent();
        propertyRef->name = requiredMemberDeclRef.getName();
        auto checkedPropertyRefExpr = CheckExpr(propertyRef);

        if (as<GetterDecl>(requiredAccessorDeclRef))
        {
            auto synReturn = m_astBuilder->create<ReturnStmt>();
            synReturn->expression = checkedPropertyRefExpr;

            synBodyStmt = synReturn;
        }
        else if (as<SetterDecl>(requiredAccessorDeclRef))
        {
            auto synAssign = m_astBuilder->create<AssignExpr>();
            synAssign->left = checkedPropertyRefExpr;
            synAssign->right = synArgs[0];

            auto synCheckedAssign = checkAssignWithCheckedOperands(synAssign);

            auto synExprStmt = m_astBuilder->create<ExpressionStmt>();
            synExprStmt->expression = synCheckedAssign;

            synBodyStmt = synExprStmt;
        }
        else
        {
            // While there are other kinds of accessors than `get` and `set`,
            // those are currently only reserved for the internal use in the core module.
            // We will not bother with synthesis for those cases.
            //
            return false;
        }

        addModifier(synAccessorDecl, m_astBuilder->create<ForceInlineAttribute>());
        synAccessorDecl->body = synBodyStmt;

        synPropertyDecl->addMember(synAccessorDecl);

        // Register the synthesized accessor.
        //
        witnessTable->add(
            requiredAccessorDeclRef.getDecl(),
            RequirementWitness(makeDeclRef(synAccessorDecl)));
    }

    // The type of our synthesized property will be the same as the inner property.
    //
    auto propertyType = getType(m_astBuilder, as<PropertyDecl>(innerProperty));
    synPropertyDecl->type.type = propertyType;

    // The visibility of synthesized decl should be the same as the inner requirement
    if (innerProperty.getDecl()->findModifier<VisibilityModifier>())
    {
        auto vis = getDeclVisibility(innerProperty.getDecl());
        addVisibilityModifier(synPropertyDecl, vis);
    }

    context->parentDecl->addMember(synPropertyDecl);
    witnessTable->add(
        requiredMemberDeclRef.getDecl(),
        RequirementWitness(makeDeclRef(synPropertyDecl)));
    return true;
}

bool SemanticsVisitor::trySynthesizeAssociatedTypeRequirementWitness(
    ConformanceCheckingContext* context,
    LookupResult const& inLookupResult,
    DeclRef<AssocTypeDecl> requiredMemberDeclRef,
    RefPtr<WitnessTable> witnessTable)
{
    SLANG_UNUSED(inLookupResult);

    // The only case we can synthesize for now is when the conformant type
    // is a wrapper type.
    if (!isWrapperTypeDecl(context->parentDecl))
        return false;
    auto aggTypeDecl = as<AggTypeDecl>(context->parentDecl);
    auto lookupResult = lookUpMember(
        m_astBuilder,
        this,
        requiredMemberDeclRef.getName(),
        aggTypeDecl->wrappedType.type,
        aggTypeDecl->ownedScope,
        LookupMask::Default,
        LookupOptions::IgnoreBaseInterfaces);
    if (!lookupResult.isValid() || lookupResult.isOverloaded())
        return false;
    auto assocType = DeclRefType::create(m_astBuilder, lookupResult.item.declRef);
    witnessTable->add(requiredMemberDeclRef.getDecl(), assocType);
    for (auto typeConstraintDecl :
         getMembersOfType<TypeConstraintDecl>(m_astBuilder, requiredMemberDeclRef))
    {
        auto witness = tryGetSubtypeWitness(assocType, getSup(m_astBuilder, typeConstraintDecl));
        if (!witness)
            return false;
        witnessTable->add(typeConstraintDecl.getDecl(), witness);
    }
    return true;
}

bool SemanticsVisitor::trySynthesizeAssociatedConstantRequirementWitness(
    ConformanceCheckingContext* context,
    LookupResult const& inLookupResult,
    DeclRef<VarDeclBase> requiredMemberDeclRef,
    RefPtr<WitnessTable> witnessTable)
{
    SLANG_UNUSED(inLookupResult);

    // The only case we can synthesize for now is when the conformant type
    // is a wrapper type, i.e.
    // struct Foo:IFoo = FooImpl;
    if (!isWrapperTypeDecl(context->parentDecl))
        return false;

    // Find the witness that FooImpl : IFoo.
    auto aggTypeDecl = as<AggTypeDecl>(context->parentDecl);
    auto innerType = aggTypeDecl->wrappedType.type;
    DeclRef<Decl> innerProperty;
    auto innerWitness = tryGetSubtypeWitness(innerType, witnessTable->baseType);
    if (!innerWitness)
        return false;

    auto witness =
        tryLookUpRequirementWitness(m_astBuilder, innerWitness, requiredMemberDeclRef.getDecl());
    if (witness.getFlavor() != RequirementWitness::Flavor::val)
        return false;
    witnessTable->add(requiredMemberDeclRef.getDecl(), witness.getVal());
    return true;
}

bool SemanticsVisitor::synthesizeAccessorRequirements(
    ConformanceCheckingContext* context,
    DeclRef<ContainerDecl> requiredMemberDeclRef,
    Type* resultType,
    Expr* synBoundStorageExpr,
    ContainerDecl* synAccesorContainer,
    RefPtr<WitnessTable> witnessTable)
{
    Dictionary<DeclRef<AccessorDecl>, AccessorDecl*> mapRequiredAccessorToSynAccessor;
    for (auto requiredAccessorDeclRef :
         getMembersOfType<AccessorDecl>(m_astBuilder, requiredMemberDeclRef))
    {
        // The synthesized accessor will be an AST node of the same class as
        // the required accessor.
        //
        auto synAccessorDecl = (AccessorDecl*)m_astBuilder->createByNodeType(
            requiredAccessorDeclRef.getDecl()->astNodeType);
        synAccessorDecl->ownedScope = m_astBuilder->create<Scope>();
        synAccessorDecl->ownedScope->containerDecl = synAccessorDecl;
        synAccessorDecl->ownedScope->parent = getScope(context->parentDecl);

        // Whatever the required accessor returns, that is what our synthesized accessor will
        // return.
        //
        synAccessorDecl->returnType.type = resultType;

        // Similarly, our synthesized accessor will have parameters matching those of the
        // requirement.
        //
        // Note: in practice we expect that only `set` accessors will have any parameters,
        // and they will only have a single parameter.
        //
        List<Expr*> synArgs;
        for (auto requiredParamDeclRef : getParameters(m_astBuilder, requiredAccessorDeclRef))
        {
            auto paramType = getType(m_astBuilder, requiredParamDeclRef);

            // The synthesized parameter will ahve the same name and
            // type as the parameter of the requirement.
            //
            auto synParamDecl = m_astBuilder->create<ParamDecl>();
            synParamDecl->nameAndLoc = requiredParamDeclRef.getDecl()->nameAndLoc;
            synParamDecl->type.type = paramType;

            // We need to add the parameter as a child declaration of
            // the accessor we are building.
            //
            synAccessorDecl->addMember(synParamDecl);

            // For each paramter, we will create an argument expression
            // to represent it in the body of the accessor.
            //
            auto synArg = m_astBuilder->create<VarExpr>();
            synArg->declRef = makeDeclRef(synParamDecl);
            synArg->type = paramType;
            synArgs.add(synArg);
        }

        // We need to create a `this` expression to be used in the body
        // of the synthesized accessor.
        //
        // TODO: if we ever allow `static` properties or subscripts,
        // we will need to handle that case here, by *not* creating
        // a `this` expression.
        //
        ThisExpr* synThis = m_astBuilder->create<ThisExpr>();
        synThis->scope = synAccessorDecl->ownedScope;

        // The type of `this` in our accessor will be the type for
        // which we are synthesizing a conformance.
        //
        synThis->type.type = context->conformingType;

        // A `get` accessor should default to an immutable `this`,
        // while other accessors default to mutable `this`.
        //
        // TODO: If we ever add other kinds of accessors, we will
        // need to check that this assumption stays valid.
        //
        synThis->type.isLeftValue = true;
        if (as<GetterDecl>(requiredAccessorDeclRef))
            synThis->type.isLeftValue = false;

        // If the accessor requirement is `[nonmutating]` then our
        // synthesized accessor should be too, and also the `this`
        // parameter should *not* be an l-value.
        //
        if (requiredAccessorDeclRef.getDecl()->hasModifier<NonmutatingAttribute>())
        {
            synThis->type.isLeftValue = false;

            auto synAttr = m_astBuilder->create<NonmutatingAttribute>();
            synAccessorDecl->modifiers.first = synAttr;
        }
        //
        // Note: we don't currently support `[mutating] get` accessors,
        // but the desired behavior in that case is clear, so we go
        // ahead and future-proof this code a bit:
        //
        else if (requiredAccessorDeclRef.getDecl()->hasModifier<MutatingAttribute>())
        {
            synThis->type.isLeftValue = true;

            auto synAttr = m_astBuilder->create<MutatingAttribute>();
            synAccessorDecl->modifiers.first = synAttr;
        }
        else if (requiredAccessorDeclRef.getDecl()->hasModifier<RefAttribute>())
        {
            synThis->type.isLeftValue = true;

            auto synAttr = m_astBuilder->create<RefAttribute>();
            synAccessorDecl->modifiers.first = synAttr;
        }
        else if (requiredAccessorDeclRef.getDecl()->hasModifier<ConstRefAttribute>())
        {
            auto synAttr = m_astBuilder->create<ConstRefAttribute>();
            synAccessorDecl->modifiers.first = synAttr;
        }
        // We are going to synthesize an expression and then perform
        // semantic checking on it, but if there are semantic errors
        // we do *not* want to report them to the user as such, and
        // instead want the result to be a failure to synthesize
        // a valid witness.
        //
        // We will buffer up diagnostics into a temporary sink and
        // then throw them away when we are done.
        //
        // TODO: This behavior might be something we want to make
        // into a more fundamental capability of `DiagnosticSink` and/or
        // `SemanticsVisitor` so that code can push/pop the emission
        // of diagnostics more easily.
        //
        DiagnosticSink tempSink(getSourceManager(), nullptr);
        SemanticsVisitor subVisitor(withSink(&tempSink));

        // The body of the accessor will depend on the class of the accessor
        // we are synthesizing (e.g., `get` vs. `set`).
        //
        Stmt* synBodyStmt = nullptr;
        if (as<GetterDecl>(requiredAccessorDeclRef))
        {
            // A `get` accessor will simply perform:
            //
            //      return this.name;
            //
            // which involves coercing the member access `this.name` to
            // the expected type of the property.
            //
            auto coercedMemberRef =
                subVisitor.coerce(CoercionSite::Return, resultType, synBoundStorageExpr);
            auto synReturn = m_astBuilder->create<ReturnStmt>();
            synReturn->expression = coercedMemberRef;

            synBodyStmt = synReturn;
        }
        else if (as<SetterDecl>(requiredAccessorDeclRef))
        {
            // We expect all `set` accessors to have a single argument,
            // but we will defensively bail out if that is somehow
            // not the case.
            //
            SLANG_ASSERT(synArgs.getCount() == 1);
            if (synArgs.getCount() != 1)
                return false;

            // A `set` accessor will simply perform:
            //
            //      this.name = newValue;
            //
            // which involves creating and checking an assignment
            // expression.

            auto synAssign = m_astBuilder->create<AssignExpr>();
            synAssign->left = synBoundStorageExpr;
            synAssign->right = synArgs[0];

            auto synCheckedAssign = subVisitor.checkAssignWithCheckedOperands(synAssign);

            auto synExprStmt = m_astBuilder->create<ExpressionStmt>();
            synExprStmt->expression = synCheckedAssign;

            synBodyStmt = synExprStmt;
        }
        else
        {
            // While there are other kinds of accessors than `get` and `set`,
            // those are currently only reserved for the internal use in the core module.
            // We will not bother with synthesis for those cases.
            //
            return false;
        }

        // We bail out if we ran into any errors (meaning that the synthesized
        // accessor is not usable).
        //
        // TODO: If there were *warnings* emitted to the sink, it would probably
        // be good to show those warnings to the user, since they might indicate
        // real issues. E.g., with the current logic a `float` field could
        // satisfying an `int` property requirement, but the user would probably
        // want to be warned when they do such a thing.
        //
        if (tempSink.getErrorCount() != 0)
            return false;

        synAccessorDecl->body = synBodyStmt;

        synAccesorContainer->addMember(synAccessorDecl);

        // If synthesis of an accessor worked, then we will record it into
        // a local dictionary. We do *not* install the accessor into the
        // witness table yet, because it is possible that synthesis will
        // succeed for some accessors but not others, and we don't want
        // to leave the witness table in a state where a requirement is
        // "partially satisfied."
        //
        mapRequiredAccessorToSynAccessor.add(requiredAccessorDeclRef, synAccessorDecl);
    }

    // Once our synthesized declaration is complete, we need
    // to install it as the witness that satifies the given
    // requirement.
    //
    // Subsequent code generation should not be able to tell the
    // difference between our synthetic property and a hand-written
    // one with the same behavior.
    //
    auto containerDecl = getParentDecl(synAccesorContainer);
    auto containerDeclRef = getDefaultDeclRef(containerDecl);
    for (auto& [key, value] : mapRequiredAccessorToSynAccessor)
    {
        witnessTable->add(
            key.getDecl(),
            RequirementWitness(m_astBuilder->getMemberDeclRef(containerDeclRef, value)));
    }

    witnessTable->add(
        requiredMemberDeclRef.getDecl(),
        RequirementWitness(m_astBuilder->getMemberDeclRef(containerDeclRef, synAccesorContainer)));
    return true;
}

bool SemanticsVisitor::trySynthesizeWrapperTypeSubscriptRequirementWitness(
    ConformanceCheckingContext* context,
    DeclRef<SubscriptDecl> requiredMemberDeclRef,
    RefPtr<WitnessTable> witnessTable)
{
    // We are synthesizing the subscript requirement for a wrapper type:
    // struct Wrapper
    // {
    //      Inner inner;
    //      subscript(int index)->int { get { return inner[index]; }
    //                                  set { inner[index] = newValue; }
    //                                }
    // }
    //
    // // Find the witness that FooImpl : IFoo.
    auto aggTypeDecl = as<AggTypeDecl>(context->parentDecl);
    auto innerType = aggTypeDecl->wrappedType.type;
    DeclRef<Decl> innerProperty;
    auto innerWitness = tryGetSubtypeWitness(innerType, witnessTable->baseType);
    if (!innerWitness)
        return false;
    //
    List<Expr*> synArgs;
    ThisExpr* synThis;
    auto synSubscriptDecl = synthesizeMethodSignatureForRequirementWitness(
        context,
        requiredMemberDeclRef,
        synArgs,
        synThis);
    auto declType = getType(m_astBuilder, getDefaultDeclRef(synSubscriptDecl).as<SubscriptDecl>());
    synThis->checked = true;

    // Form a `this[args...]` expression that we will use to coerce from
    // in the synthesized subscript accessors.
    //
    DiagnosticSink tempSink(getSourceManager(), nullptr);
    SemanticsVisitor subVisitor(withSink(&tempSink));
    auto base = m_astBuilder->create<VarExpr>();
    base->scope = synThis->scope;
    base->name = getName("inner");

    IndexExpr* indexExpr = m_astBuilder->create<IndexExpr>();
    indexExpr->baseExpression = base;
    indexExpr->indexExprs = _Move(synArgs);
    auto synBaseStorageExpr = subVisitor.CheckTerm(indexExpr);

    if (tempSink.getErrorCount() != 0)
        return false;

    // Our synthesized subscript will have an accessor declaration for
    // each accessor of the requirement.
    //
    bool canSynAccessors = synthesizeAccessorRequirements(
        context,
        requiredMemberDeclRef,
        declType,
        synBaseStorageExpr,
        synSubscriptDecl,
        witnessTable);
    if (!canSynAccessors)
        return false;

    // The visibility of synthesized decl should be the min of the parent decl and the requirement.
    if (requiredMemberDeclRef.getDecl()->findModifier<VisibilityModifier>())
    {
        auto requirementVisibility = getDeclVisibility(requiredMemberDeclRef.getDecl());
        auto thisVisibility = getDeclVisibility(context->parentDecl);
        auto visibility = Math::Min(thisVisibility, requirementVisibility);
        addVisibilityModifier(synSubscriptDecl, visibility);
    }

    return true;
}

bool SemanticsVisitor::trySynthesizeSubscriptRequirementWitness(
    ConformanceCheckingContext* context,
    const LookupResult& lookupResult,
    DeclRef<SubscriptDecl> requiredMemberDeclRef,
    RefPtr<WitnessTable> witnessTable)
{
    if (isWrapperTypeDecl(context->parentDecl))
        return trySynthesizeWrapperTypeSubscriptRequirementWitness(
            context,
            requiredMemberDeclRef,
            witnessTable);

    // The situation here is that the context of an inheritance
    // declaration didn't provide an exact match for a required
    // subscript. E.g.:
    //
    //      interface ICell { subscript(int index)->int {get;} }
    //      struct MyCell : ICell
    //      {
    //          subscript(uint index)->int {ref;}
    //      }
    //
    // It is clear in this case that the `MyCell` type *can*
    // satisfy the signature required by `ICell`, if we consider
    // all the allowed type coercion rules, and use `ref` accessor
    // to implement `get`.
    //
    // The approach in this function will be to construct a
    // synthesized `subscript` along the lines of:
    //
    //      struct MyCell ...
    //      {
    //          ...
    //          subscript(int index)->int {get;}
    //          {
    //              get { return this.origianl_subscript[index]; }
    //          }
    //      }
    //
    // That is, we construct a `subscript` with the correct type
    // and with an accessor for each requirement, where the accesors
    // all try to dispatch to the original subscript decl.
    //
    // If those synthesized accessors all type-check, then we can
    // say that the type must satisfy the requirement structurally,
    // even if there isn't an exact signature match. More
    // importantly, the `property` we just synthesized can be
    // used as a witness to the fact that the requirement is
    // satisfied.
    //
    // The big-picture flow of the logic here is similar to
    // `trySynthesizePropertyRequirementWitness()` above, and we
    // will not comment this code as exhaustively, under the
    // assumption that readers of the code don't benefit from
    // having the exact same information stated twice.
    //

    List<Expr*> synArgs;
    ThisExpr* synThis;
    auto synSubscriptDecl = synthesizeMethodSignatureForRequirementWitness(
        context,
        requiredMemberDeclRef,
        synArgs,
        synThis);
    synThis->type.isLeftValue = true;
    synThis->checked = true;

    auto declType = getType(m_astBuilder, getDefaultDeclRef(synSubscriptDecl).as<SubscriptDecl>());

    // Form a `this[args...]` expression that we will use to coerce from
    // in the synthesized subscript accessors.
    //
    DiagnosticSink tempSink(getSourceManager(), nullptr);
    SemanticsVisitor subVisitor(withSink(&tempSink));
    Expr* synBaseStorageExpr = nullptr;
    if (lookupResult.isValid())
    {
        auto calleeExpr = m_astBuilder->create<OverloadedExpr>();
        calleeExpr->base = synThis;
        calleeExpr->lookupResult2 = lookupResult;
        auto invokeExpr = m_astBuilder->create<InvokeExpr>();
        invokeExpr->functionExpr = calleeExpr;
        invokeExpr->arguments = _Move(synArgs);
        synBaseStorageExpr = subVisitor.ResolveInvoke(invokeExpr);
    }
    else
    {
        IndexExpr* indexExpr = m_astBuilder->create<IndexExpr>();
        indexExpr->baseExpression = synThis;
        indexExpr->indexExprs = _Move(synArgs);
        synBaseStorageExpr = subVisitor.CheckTerm(indexExpr);
    }
    if (tempSink.getErrorCount() != 0)
        return false;

    // Our synthesized subscript will have an accessor declaration for
    // each accessor of the requirement.
    //
    bool canSynAccessors = synthesizeAccessorRequirements(
        context,
        requiredMemberDeclRef,
        declType,
        synBaseStorageExpr,
        synSubscriptDecl,
        witnessTable);
    if (!canSynAccessors)
        return false;


    // The visibility of synthesized decl should be the min of the parent decl and the requirement.
    if (requiredMemberDeclRef.getDecl()->findModifier<VisibilityModifier>())
    {
        auto requirementVisibility = getDeclVisibility(requiredMemberDeclRef.getDecl());
        auto thisVisibility = getDeclVisibility(context->parentDecl);
        auto visibility = Math::Min(thisVisibility, requirementVisibility);
        addVisibilityModifier(synSubscriptDecl, visibility);
    }

    return true;
}

bool SemanticsVisitor::trySynthesizeRequirementWitness(
    ConformanceCheckingContext* context,
    LookupResult const& lookupResult,
    DeclRef<Decl> requiredMemberDeclRef,
    RefPtr<WitnessTable> witnessTable)
{
    SLANG_UNUSED(lookupResult);
    SLANG_UNUSED(requiredMemberDeclRef);
    SLANG_UNUSED(witnessTable);

    if (auto requiredFuncDeclRef = requiredMemberDeclRef.as<FuncDecl>())
    {
        // Check signature match.
        if (trySynthesizeMethodRequirementWitness(
                context,
                lookupResult,
                requiredFuncDeclRef,
                witnessTable))
            return true;

        if (auto builtinAttr =
                requiredFuncDeclRef.getDecl()->findModifier<BuiltinRequirementModifier>())
        {
            switch (builtinAttr->kind)
            {
            case BuiltinRequirementKind::DAddFunc:
            case BuiltinRequirementKind::DZeroFunc:
                return trySynthesizeDifferentialMethodRequirementWitness(
                    context,
                    requiredFuncDeclRef,
                    witnessTable,
                    SynthesisPattern::AllInductive);
            case BuiltinRequirementKind::And:
            case BuiltinRequirementKind::Or:
            case BuiltinRequirementKind::Not:
            case BuiltinRequirementKind::BitAnd:
            case BuiltinRequirementKind::BitNot:
            case BuiltinRequirementKind::BitOr:
            case BuiltinRequirementKind::BitXor:
            case BuiltinRequirementKind::Shl:
            case BuiltinRequirementKind::Shr:
            case BuiltinRequirementKind::Equals:
            case BuiltinRequirementKind::LessThan:
            case BuiltinRequirementKind::LessThanOrEquals:
                if (isEnumType(context->conformingType))
                    return trySynthesizeEnumTypeMethodRequirementWitness(
                        context,
                        requiredFuncDeclRef,
                        witnessTable,
                        builtinAttr->kind);
                break;
            }
        }
        return false;
    }

    // For generic decl, check if we match DMulFunc, and synthesize the method.
    if (auto requiredGenericDeclRef = requiredMemberDeclRef.as<GenericDecl>())
    {
        auto inner = getInner(requiredGenericDeclRef);

        // TODO: we should be able to remove DMul synthesis logic.
        if (auto builtinAttr = inner->findModifier<BuiltinRequirementModifier>())
        {
            switch (builtinAttr->kind)
            {
            case BuiltinRequirementKind::DMulFunc:
                return trySynthesizeDifferentialMethodRequirementWitness(
                    context,
                    requiredGenericDeclRef,
                    witnessTable,
                    SynthesisPattern::FixedFirstArg);
            }
        }

        if (as<CallableDecl>(inner))
        {
            return trySynthesizeRequirementWitness(
                context,
                lookupResult,
                m_astBuilder->getMemberDeclRef(requiredGenericDeclRef, inner),
                witnessTable);
        }
        return false;
    }

    if (auto requiredPropertyDeclRef = requiredMemberDeclRef.as<PropertyDecl>())
    {
        return trySynthesizePropertyRequirementWitness(
            context,
            lookupResult,
            requiredPropertyDeclRef,
            witnessTable);
    }

    if (auto requiredSubscriptDeclRef = requiredMemberDeclRef.as<SubscriptDecl>())
    {
        return trySynthesizeSubscriptRequirementWitness(
            context,
            lookupResult,
            requiredSubscriptDeclRef,
            witnessTable);
    }

    if (auto requiredAssocTypeDeclRef = requiredMemberDeclRef.as<AssocTypeDecl>())
    {
        if (auto builtinAttr =
                requiredAssocTypeDeclRef.getDecl()->findModifier<BuiltinRequirementModifier>())
        {
            switch (builtinAttr->kind)
            {
            case BuiltinRequirementKind::DifferentialType:
                return trySynthesizeDifferentialAssociatedTypeRequirementWitness(
                    context,
                    requiredAssocTypeDeclRef,
                    witnessTable);
            }
        }
        else
        {
            return trySynthesizeAssociatedTypeRequirementWitness(
                context,
                lookupResult,
                requiredAssocTypeDeclRef,
                witnessTable);
        }
    }

    if (auto requiredConstantDeclRef = requiredMemberDeclRef.as<VarDeclBase>())
    {
        return trySynthesizeAssociatedConstantRequirementWitness(
            context,
            lookupResult,
            requiredConstantDeclRef,
            witnessTable);
    }

    if (auto requiredCtor = requiredMemberDeclRef.as<ConstructorDecl>())
    {
        return trySynthesizeConstructorRequirementWitness(
            context,
            lookupResult,
            requiredCtor,
            witnessTable);
    }

    // TODO: There are other kinds of requirements for which synthesis should
    // be possible:
    //
    // * It should be possible to synthesize required initializers
    //   using an approach similar to what is used for methods.
    //
    // * We should be able to synthesize subscripts with different
    //   signatures (taking into account default parameters).
    //
    // * For specific kinds of generic requirements, we should be able
    //   to wrap the synthesis of the inner declaration in synthesis
    //   of an outer generic with a matching signature.
    //
    // All of these cases can/should use similar logic to
    // `trySynthesizeMethodRequirementWitness` where they construct an AST
    // in the form of what the use site ought to look like, and then
    // apply existing semantic checking logic to generate the code.

    return false;
}

Stmt* _synthesizeMemberAssignMemberHelper(
    ASTSynthesizer& synth,
    Name* funcName,
    Type* leftType,
    Expr* leftValue,
    List<Expr*>&& args,
    List<Expr*>&& genericArgs,
    List<bool>&& inductiveArgMask,
    int nestingLevel = 0)
{
    if (nestingLevel > 16)
        return nullptr;

    // If field type is an array, assign each element individually.
    if (auto arrayType = as<ArrayExpressionType>(leftType))
    {
        VarDecl* indexVar = nullptr;
        auto forStmt =
            synth.emitFor(synth.emitIntConst(0), synth.emitGetArrayLengthExpr(leftValue), indexVar);
        addModifier(forStmt, synth.getBuilder()->create<ForceUnrollAttribute>());
        auto innerLeft = synth.emitIndexExpr(leftValue, synth.emitVarExpr(indexVar));

        for (auto ii = 0; ii < args.getCount(); ++ii)
        {
            auto& arg = args[ii];
            if (inductiveArgMask[ii])
                arg = synth.emitIndexExpr(arg, synth.emitVarExpr(indexVar));
        }

        auto assignStmt = _synthesizeMemberAssignMemberHelper(
            synth,
            funcName,
            arrayType->getElementType(),
            innerLeft,
            _Move(args),
            _Move(genericArgs),
            _Move(inductiveArgMask),
            nestingLevel + 1);

        synth.popScope();
        if (!assignStmt)
            return nullptr;
        return forStmt;
    }

    auto callee = synth.emitMemberExpr(leftType, funcName);

    if (genericArgs.getCount() > 0)
        callee = synth.emitGenericAppExpr(callee, _Move(genericArgs));

    return synth.emitAssignStmt(leftValue, synth.emitInvokeExpr(callee, _Move(args)));
}

bool SemanticsVisitor::trySynthesizeEnumTypeMethodRequirementWitness(
    ConformanceCheckingContext* context,
    DeclRef<FunctionDeclBase> funcDeclRef,
    RefPtr<WitnessTable> witnessTable,
    BuiltinRequirementKind requirementKind)
{
    List<Expr*> synArgs;
    ThisExpr* synThis = nullptr;
    auto synFunc =
        synthesizeMethodSignatureForRequirementWitness(context, funcDeclRef, synArgs, synThis);
    auto intrinsicOpModifier = getASTBuilder()->create<IntrinsicOpModifier>();
    switch (requirementKind)
    {
    case BuiltinRequirementKind::And:
        intrinsicOpModifier->op = kIROp_And;
        break;
    case BuiltinRequirementKind::Or:
        intrinsicOpModifier->op = kIROp_Or;
        break;
    case BuiltinRequirementKind::Not:
        intrinsicOpModifier->op = kIROp_Not;
        break;
    case BuiltinRequirementKind::BitAnd:
        intrinsicOpModifier->op = kIROp_BitAnd;
        break;
    case BuiltinRequirementKind::BitNot:
        intrinsicOpModifier->op = kIROp_BitNot;
        break;
    case BuiltinRequirementKind::BitOr:
        intrinsicOpModifier->op = kIROp_BitOr;
        break;
    case BuiltinRequirementKind::BitXor:
        intrinsicOpModifier->op = kIROp_BitXor;
        break;
    case BuiltinRequirementKind::Shl:
        intrinsicOpModifier->op = kIROp_Lsh;
        break;
    case BuiltinRequirementKind::Shr:
        intrinsicOpModifier->op = kIROp_Rsh;
        break;
    case BuiltinRequirementKind::Equals:
        intrinsicOpModifier->op = kIROp_Eql;
        break;
    case BuiltinRequirementKind::LessThan:
        intrinsicOpModifier->op = kIROp_Less;
        break;
    case BuiltinRequirementKind::LessThanOrEquals:
        intrinsicOpModifier->op = kIROp_Leq;
        break;
    case BuiltinRequirementKind::InitLogicalFromInt:
        intrinsicOpModifier->op = kIROp_IntCast;
        break;
    default:
        SLANG_UNEXPECTED("unknown builtin requirement kind.");
    }
    synFunc->loc = context->parentDecl->closingSourceLoc;
    synFunc->nameAndLoc.loc = synFunc->loc;
    // synFunc already has its parent set
    SLANG_ASSERT(context->parentDecl == synFunc->parentDecl);
    context->parentDecl->addMember(synFunc);
    context->parentDecl->invalidateMemberDictionary();
    addModifier(synFunc, intrinsicOpModifier);
    witnessTable->add(
        funcDeclRef.getDecl(),
        RequirementWitness(m_astBuilder->getDirectDeclRef(synFunc)));
    return true;
}

bool SemanticsVisitor::trySynthesizeDifferentialMethodRequirementWitness(
    ConformanceCheckingContext* context,
    DeclRef<Decl> requirementDeclRef,
    RefPtr<WitnessTable> witnessTable,
    SynthesisPattern pattern)
{
    // We support two cases of synthesis here.
    // Case 1 is that there the associated Differential type is defined to be `DifferentialBottom`.
    // In this case we just trivially return `DifferentialBottom` in all synthesized methods.
    // Case 2 is that the `Differential` type contains members corresponding to each primal member.
    // We will apply a general code synthesis pattern to reflect that structure.
    // For requirement of the form:
    // ```
    // static TResult requiredMethod(TParam1 p0, TParam2 p1, ...)
    // ```
    // Where TResult,TParam1, TParam2 is either `This` or `Differential`,
    // We synthesize a memberwise dispatch to compute each field of `TResult`.
    // Multiple patterns are supported (see SemanticsVisitor::SynthesisPattern for a full list)
    // For AllInductive, we synthesize an implementation of the form:
    // ```
    // [BackwardDifferentiable]
    // static TResult requiredMethod(TParam1 p0, TParam2 p1, ...)
    // {
    //     TResult result;
    //     result.member0 = decltype(result.member0).requiredMethod(p0.member0, p1.member0);
    //     result.member1 = decltype(result.member1).requiredMethod(p0.member1, p1.member1);
    //     ...
    //     return result;
    // }
    // ```

    // First we need to make sure the associated `Differential` type requirement is satisfied.
    bool hasDifferentialAssocType = false;
    for (auto& existingEntry : witnessTable->getRequirementDictionary())
    {
        if (auto builtinReqAttr = existingEntry.key->findModifier<BuiltinRequirementModifier>())
        {
            if (builtinReqAttr->kind == BuiltinRequirementKind::DifferentialType &&
                existingEntry.value.getFlavor() != RequirementWitness::Flavor::none)
            {
                hasDifferentialAssocType = true;
            }
        }
    }
    if (!hasDifferentialAssocType)
        return false;

    ASTSynthesizer synth(m_astBuilder, getNamePool());
    List<Expr*> synArgs;
    List<Expr*> synGenericArgs;
    ThisExpr* synThis = nullptr;
    FuncDecl* synFunc = nullptr;
    GenericDecl* synGeneric = nullptr;

    if (auto genericDeclRef = requirementDeclRef.as<GenericDecl>())
    {
        synGeneric = synthesizeGenericSignatureForRequirementWitness(
            context,
            genericDeclRef,
            synArgs,
            synGenericArgs,
            synThis);
        synFunc = as<FuncDecl>(synGeneric->inner);
    }
    else if (auto funcDeclRef = requirementDeclRef.as<FuncDecl>())
    {
        synFunc = as<FuncDecl>(
            synthesizeMethodSignatureForRequirementWitness(context, funcDeclRef, synArgs, synThis));
    }

    SLANG_ASSERT(synFunc);

    addModifier(synFunc, m_astBuilder->create<BackwardDifferentiableAttribute>());


    synth.pushContainerScope(synFunc);
    auto blockStmt = m_astBuilder->create<BlockStmt>();
    synFunc->body = blockStmt;
    auto seqStmt = synth.pushSeqStmtScope();
    blockStmt->body = seqStmt;

    // Create a variable for return value.
    synth.pushVarScope();
    auto varStmt = synth.emitVarDeclStmt(synFunc->returnType.type, getName("result"));
    auto resultVarExpr = synth.emitVarExpr(varStmt, synFunc->returnType.type);

    for (auto member : context->parentDecl->members)
    {
        auto derivativeAttr = member->findModifier<DerivativeMemberAttribute>();
        if (!derivativeAttr)
            continue;

        auto varMember = as<VarDeclBase>(member);
        if (!varMember)
            continue;
        ensureDecl(varMember, DeclCheckState::ReadyForReference);
        auto memberType = varMember->getType();
        auto diffMemberType = tryGetDifferentialType(m_astBuilder, memberType);
        if (!diffMemberType)
            continue;

        // Pull up the derivative member name from the attribute
        auto derivMemberName = derivativeAttr->memberDeclRef->declRef.getName();

        // Construct reference exprs to the member's corresponding fields in each parameter.
        List<Expr*> paramFields;
        List<bool> inductiveArgMask;

        switch (pattern)
        {
        case SynthesisPattern::AllInductive:
            {
                for (auto arg : synArgs)
                {
                    auto memberExpr = m_astBuilder->create<MemberExpr>();
                    memberExpr->baseExpression = arg;

                    memberExpr->name = derivMemberName;

                    paramFields.add(memberExpr);
                    inductiveArgMask.add(true);
                }
                break;
            }
        case SynthesisPattern::FixedFirstArg:
            {
                int paramIndex = 0;
                for (auto arg : synArgs)
                {
                    if (paramIndex == 0)
                    {
                        paramFields.add(arg);
                        inductiveArgMask.add(false);

                        paramIndex++;
                    }
                    else
                    {
                        auto memberExpr = m_astBuilder->create<MemberExpr>();
                        memberExpr->baseExpression = arg;

                        memberExpr->name = derivMemberName;
                        paramFields.add(memberExpr);
                        inductiveArgMask.add(true);

                        paramIndex++;
                    }
                }
                break;
            }
        default:
            SLANG_UNIMPLEMENTED_X("unhandled synthesis pattern");
            break;
        }

        // Invoke the method for the field and assign the value to resultVar.
        auto leftVal = synth.emitMemberExpr(resultVarExpr, derivMemberName);
        if (!_synthesizeMemberAssignMemberHelper(
                synth,
                requirementDeclRef.getName(),
                memberType,
                leftVal,
                _Move(paramFields),
                _Move(synGenericArgs),
                _Move(inductiveArgMask)))
            return false;
    }

    // TODO: synthesize assignments for inherited members here.

    auto synReturn = m_astBuilder->create<ReturnStmt>();
    synReturn->expression = resultVarExpr;
    seqStmt->stmts.add(synReturn);

    Decl* witnessDecl = synGeneric ? (Decl*)synGeneric : synFunc;
    SLANG_ASSERT(context->parentDecl == witnessDecl->parentDecl);
    context->parentDecl->addMember(witnessDecl);
    context->parentDecl->invalidateMemberDictionary();
    addModifier(synFunc, m_astBuilder->create<SynthesizedModifier>());

    // If `This` is nested inside a generic, we need to form a complete declref type to the
    // newly synthesized method here in order to fill into the witness table.
    // This can be done by obtaining the ThisType witness from requirementDeclRef to get the
    // generic substitution for outer generic parameters, and apply it here.
    SubstitutionSet substSet;
    if (auto thisTypeWitness = findThisTypeWitness(
            SubstitutionSet(requirementDeclRef),
            as<InterfaceDecl>(requirementDeclRef.getDecl()->parentDecl)))
    {
        if (auto declRefType = as<DeclRefType>(thisTypeWitness->getSub()))
        {
            substSet = SubstitutionSet(declRefType->getDeclRef());
        }
    }
    if (!substSet.declRef)
        return false;
    DeclRef<Decl> synthesizedWitnessDeclRef;
    if (auto parentExtDecl = as<ExtensionDecl>(context->parentDecl))
    {
        // If the conformance is declared on an extension to ThisType,
        // we need to form a new proper decl ref to the parent extension decl
        // with the correct specialization arguments.
        //
        if (GetOuterGeneric(context->parentDecl))
        {

            auto extDeclRef = applyExtensionToType(parentExtDecl, context->conformingType);
            synthesizedWitnessDeclRef = m_astBuilder->getMemberDeclRef(extDeclRef, witnessDecl);
        }
    }
    else
    {
        synthesizedWitnessDeclRef = m_astBuilder->getMemberDeclRef(substSet.declRef, witnessDecl);
    }
    if (!synthesizedWitnessDeclRef)
        synthesizedWitnessDeclRef = m_astBuilder->getDirectDeclRef(witnessDecl);
    witnessTable->add(requirementDeclRef.getDecl(), RequirementWitness(synthesizedWitnessDeclRef));
    return true;
}

bool SemanticsVisitor::findWitnessForInterfaceRequirement(
    ConformanceCheckingContext* context,
    Type* subType,
    Type* superInterfaceType,
    InheritanceDecl* inheritanceDecl,
    DeclRef<InterfaceDecl> superInterfaceDeclRef,
    DeclRef<Decl> requiredMemberDeclRef,
    RefPtr<WitnessTable> witnessTable,
    SubtypeWitness* subTypeConformsToSuperInterfaceWitness)
{
    SLANG_UNUSED(superInterfaceDeclRef)

    // The goal of this function is to find a suitable
    // value to satisfy the requirement.
    //
    // The 99% case is that the requirement is a named member
    // of the interface, and we need to search for a member
    // with the same name in the type declaration and
    // its (known) extensions.

    // The exception to that is when the requiredMemberDeclRef is already
    // resolved to the actual satisfying decl, in which case we simply return
    // true without any further lookup.
    if (!as<InterfaceDecl>(requiredMemberDeclRef.getParent().getDecl()))
        return true;

    // If `requiredMemberDeclRef` is a lookup decl ref for an interface requirement
    // we attempt to do the loopkup through witness tables.
    //
    // As a first pass, lets check if we already have a
    // witness in the table for the requirement, so
    // that we can bail out early.
    //
    if (witnessTable->getRequirementDictionary().containsKey(requiredMemberDeclRef.getDecl()))
    {
        return true;
    }

    // The ThisType requirement is always satisfied.
    if (as<ThisTypeDecl>(requiredMemberDeclRef.getDecl()))
    {
        return true;
    }

    // An important exception to the above is that an
    // inheritance declaration in the interface is not going
    // to be satisfied by an inheritance declaration in the
    // conforming type, but rather by a full "witness table"
    // full of the satisfying values for each requirement
    // in the inherited-from interface.
    //
    if (auto requiredInheritanceDeclRef = requiredMemberDeclRef.as<InheritanceDecl>())
    {
        // Recursively check that the type conforms
        // to the inherited interface.
        //
        // TODO: we *really* need a linearization step here!!!!

        auto reqType = getBaseType(m_astBuilder, requiredInheritanceDeclRef);

        auto interfaceIsReqWitness = m_astBuilder->getDeclaredSubtypeWitness(
            superInterfaceType,
            reqType,
            requiredInheritanceDeclRef);
        // ...

        auto subIsReqWitness = m_astBuilder->getTransitiveSubtypeWitness(
            subTypeConformsToSuperInterfaceWitness,
            interfaceIsReqWitness);
        // ...

        RefPtr<WitnessTable> satisfyingWitnessTable = new WitnessTable();
        satisfyingWitnessTable->witnessedType = subType;
        satisfyingWitnessTable->baseType = reqType;

        witnessTable->add(
            requiredInheritanceDeclRef.getDecl(),
            RequirementWitness(satisfyingWitnessTable));

        if (!checkConformanceToType(
                context,
                subType,
                requiredInheritanceDeclRef.getDecl(),
                reqType,
                subIsReqWitness,
                satisfyingWitnessTable))
        {
            return false;
        }

        return true;
    }

    // We will look up members with the same name,
    // since only same-name members will be able to
    // satisfy the requirement.
    //
    Name* name = requiredMemberDeclRef.getName();

    // We start by looking up members of the same
    // name, on the type that is claiming to conform.
    //
    // This lookup step could include members that
    // we might not actually want to consider:
    //
    // * Lookup through a type `Foo` where `Foo : IBar`
    //   will be able to find members of `IBar`, which
    //   somewhat obviously shouldn't apply when
    //   determining if `Foo` satisfies the requirements
    //   of `IBar`.
    //
    // * Lookup in the presence of `__transparent` members
    //   may produce references to declarations on a *field*
    //   of the type rather than the type. Conformance through
    //   transparent members could be supported in theory,
    //   but would require synthesizing proxy/forwarding
    //   implementations in the type itself.
    //
    // For the first issue, we will use a flag to influence
    // lookup so that it doesn't include results looked up
    // through interface inheritance clauses (but it *will*
    // look up result through inheritance clauses corresponding
    // to concrete types).
    //
    // The second issue of members that require us to proxy/forward
    // requests will be handled further down. For now we include
    // lookup results that might be usable, but not as-is.
    //
    LookupResult lookupResult;
    if (!isWrapperTypeDecl(context->parentDecl))
    {
        lookupResult = lookUpMember(
            m_astBuilder,
            this,
            name,
            subType,
            nullptr,
            LookupMask::Default,
            LookupOptions::IgnoreBaseInterfaces);

        if (!lookupResult.isValid())
        {
            // If we failed to look up a member with the name of the
            // requirement, it may be possible that we can still synthesis the
            // implementation if this is one of the known builtin requirements.
            // Otherwise, report diagnostic now.

            if (requiredMemberDeclRef.getDecl()->hasModifier<BuiltinRequirementModifier>() ||
                (requiredMemberDeclRef.as<GenericDecl>() &&
                 getInner(requiredMemberDeclRef.as<GenericDecl>())
                     ->hasModifier<BuiltinRequirementModifier>()))
            {
            }
            else if (
                requiredMemberDeclRef.as<SubscriptDecl>() &&
                (as<ArrayExpressionType>(context->conformingType) ||
                 as<VectorExpressionType>(context->conformingType) ||
                 as<MatrixExpressionType>(context->conformingType)))
            {
            }
            else
            {
                getSink()->diagnose(
                    inheritanceDecl,
                    Diagnostics::typeDoesntImplementInterfaceRequirement,
                    subType,
                    requiredMemberDeclRef);
                getSink()->diagnose(
                    requiredMemberDeclRef,
                    Diagnostics::seeDeclarationOf,
                    requiredMemberDeclRef);
                return false;
            }
        }
    }
    if (lookupResult.isOverloaded())
    {
        // If we found multiple members with the same name,
        // we want to move the declarations in the same parent as inheritanceDecl
        // to the front of the list, so that we always consider them first instead of
        // the members declared in other extension decls.
        //
        Index front = 0;
        auto parentOfInheritanceDecl = getParentAggTypeDeclBase(inheritanceDecl);
        for (Index i = 0; i < lookupResult.items.getCount(); i++)
        {
            if (getParentAggTypeDeclBase(lookupResult.items[i].declRef.getDecl()) ==
                parentOfInheritanceDecl)
            {
                lookupResult.items.swapElements(i, front);
                front++;
            }
        }
    }
    // Iterate over the members and look for one that matches
    // the expected signature for the requirement.
    for (auto member : lookupResult)
    {
        // To a first approximation, any lookup result that required a "breadcrumb"
        // will not be usable to directly satisfy an interface requirement, since
        // each breadcrumb will amount to a manipulation of `this` that is required
        // to make the declaration usable (e.g., casting to a base type).
        //
        if (member.breadcrumbs != nullptr)
            continue;

        if (doesMemberSatisfyRequirement(member.declRef, requiredMemberDeclRef, witnessTable))
        {
            // The member satisfies the requirement in every other way except that
            // it may have a lower visibility than min(parentVisibility, requirementVisibilty),
            // in that case we will treat it as an error.
            auto minRequiredVisibility = Math::Min(
                getDeclVisibility(requiredMemberDeclRef.getDecl()),
                getTypeVisibility(subType));
            if (getDeclVisibility(member.declRef.getDecl()) < minRequiredVisibility)
            {
                getSink()->diagnose(
                    member.declRef,
                    Diagnostics::satisfyingDeclCannotHaveLowerVisibility,
                    member.declRef);
                getSink()->diagnose(
                    requiredMemberDeclRef,
                    Diagnostics::seeDeclarationOf,
                    QualifiedDeclPath(requiredMemberDeclRef));
                return false;
            }
            return true;
        }
    }

    // If we reach this point then there were no members suitable
    // for satisfying the interface requirement *diretly*.
    //
    // It is possible that one of the items in `lookupResult` could be
    // used to synthesize an exact-match witness, by generating the
    // code required to handle all the conversions that might be
    // required on `this`.
    //
    // Another situation that will get us here is that we are dealing with
    // a wrapper type (struct Foo:IFoo=FooImpl), and we will synthesize
    // wrappers that redirects the call into the inner element.
    //
    context->innerSink.reset();
    if (trySynthesizeRequirementWitness(context, lookupResult, requiredMemberDeclRef, witnessTable))
    {
        return true;
    }

    // We failed to find a member of the type that can be used
    // to satisfy the requirement (even via synthesis), so we
    // need to report the failure to the user.
    //
    // TODO: Eventually we might want something akin to the current
    // overload resolution logic, where we keep track of a list
    // of "candidates" for satisfaction of the requirement,
    // and if nothing is found we print the candidates that made it
    // furthest in checking.
    //
    if (!lookupResult.isOverloaded() && lookupResult.isValid())
    {
        getSink()->diagnose(
            lookupResult.item.declRef,
            Diagnostics::memberDoesNotMatchRequirementSignature,
            lookupResult.item.declRef);
    }
    else
    {
        getSink()->diagnose(
            inheritanceDecl,
            Diagnostics::typeDoesntImplementInterfaceRequirement,
            subType,
            requiredMemberDeclRef);
    }
    if (context->innerSink.outputBuffer.getLength())
    {
        getSink()->diagnoseRaw(Severity::Note, context->innerSink.outputBuffer.getUnownedSlice());
    }
    getSink()->diagnose(
        requiredMemberDeclRef,
        Diagnostics::seeDeclarationOfInterfaceRequirement,
        requiredMemberDeclRef);
    return false;
}

RefPtr<WitnessTable> SemanticsVisitor::checkInterfaceConformance(
    ConformanceCheckingContext* context,
    Type* subType,
    Type* superInterfaceType,
    InheritanceDecl* inheritanceDecl,
    DeclRef<InterfaceDecl> superInterfaceDeclRef,
    SubtypeWitness* subTypeConformsToSuperInterfaceWitnes)
{
    // Has somebody already checked this conformance,
    // and/or is in the middle of checking it?
    RefPtr<WitnessTable> witnessTable;
    if (context->mapInterfaceToWitnessTable.tryGetValue(superInterfaceDeclRef, witnessTable))
        return witnessTable;

    // We need to check the declaration of the interface
    // before we can check that we conform to it.
    //
    ensureDecl(superInterfaceDeclRef, DeclCheckState::CanReadInterfaceRequirements);

    // We will construct the witness table, and register it
    // *before* we go about checking fine-grained requirements,
    // in order to short-circuit any potential for infinite recursion.

    // Note: we will re-use the witnes table attached to the inheritance decl,
    // if there is one. This catches cases where semantic checking might
    // have synthesized some of the conformance witnesses for us.
    //
    witnessTable = inheritanceDecl->witnessTable;
    if (!witnessTable)
    {
        witnessTable = new WitnessTable();
        witnessTable->baseType = DeclRefType::create(m_astBuilder, superInterfaceDeclRef);
        witnessTable->witnessedType = subType;
    }
    context->mapInterfaceToWitnessTable.add(superInterfaceDeclRef, witnessTable);

    if (!checkInterfaceConformance(
            context,
            subType,
            superInterfaceType,
            inheritanceDecl,
            superInterfaceDeclRef,
            subTypeConformsToSuperInterfaceWitnes,
            witnessTable))
        return nullptr;

    return witnessTable;
}

static bool isAssociatedTypeDecl(Decl* decl)
{
    auto d = decl;
    while (auto genericDecl = as<GenericDecl>(d))
        d = genericDecl->inner;
    if (as<AssocTypeDecl>(d))
        return true;
    return false;
}

bool SemanticsVisitor::checkInterfaceConformance(
    ConformanceCheckingContext* context,
    Type* subType,
    Type* superInterfaceType,
    InheritanceDecl* inheritanceDecl,
    DeclRef<InterfaceDecl> superInterfaceDeclRef,
    SubtypeWitness* subTypeConformsToSuperInterfaceWitness,
    WitnessTable* witnessTable)
{
    // We need to check the declaration of the interface
    // before we can check that we conform to it.
    //
    ensureDecl(superInterfaceDeclRef, DeclCheckState::CanReadInterfaceRequirements);

    // When comparing things like signatures, we need to do so in the context
    // of a LookupDeclRef that aligns the signatures in the interface
    // with those in the concrete type. For example, we need to treat any uses
    // of `This` in the interface as equivalent to the concrete type for the
    // purpose of signature matching (and similarly for associated types).
    //
    auto thisTypeDeclRef = m_astBuilder->getLookupDeclRef(
        subTypeConformsToSuperInterfaceWitness,
        superInterfaceDeclRef.getDecl()->getThisTypeDecl());

    bool result = true;

    // TODO: If we ever allow for implementation inheritance,
    // then we will need to consider the case where a type
    // declares that it conforms to an interface, but one of
    // its (non-interface) base types already conforms to
    // that interface, so that all of the requirements are
    // already satisfied with inherited implementations...

    // Note: we break this logic into two loops, where we first
    // check conformance for all associated-type requirements
    // and *then* check conformance for all other requirements.
    //
    // Checking associated-type requirements first ensures that
    // we can make use of the identity of the associated types
    // when checking other members.
    //
    // TODO: There could in theory be subtle cases involving
    // circular or recursive dependency chains that make such
    // a simple ordering impractical (e.g., associated type `A`
    // is constrained to `IThing<This>` where `IThing<T>` requires
    // that `T : IOtherThing where T.B == int` for another associated
    // type `B`).
    //
    // The only robust solution long-term is probably to treat this
    // as a type-inference problem by creating type variables to
    // stand in for the associated-type requirements and then to discover
    // constraints and solve for those type variables as part of the
    // conformance-checking process.
    //
    for (auto requiredMemberDecl : getMembers(m_astBuilder, superInterfaceDeclRef))
    {
        if (!isAssociatedTypeDecl(requiredMemberDecl.getDecl()))
            continue;
        ensureDecl(requiredMemberDecl, DeclCheckState::ReadyForReference);
        auto requiredMemberDeclRef = m_astBuilder->getLookupDeclRef(
            subTypeConformsToSuperInterfaceWitness,
            requiredMemberDecl.getDecl());
        auto requirementSatisfied = findWitnessForInterfaceRequirement(
            context,
            subType,
            superInterfaceType,
            inheritanceDecl,
            superInterfaceDeclRef,
            requiredMemberDeclRef,
            witnessTable,
            subTypeConformsToSuperInterfaceWitness);

        result = result && requirementSatisfied;
    }
    for (auto requiredMemberDecl : getMembers(m_astBuilder, superInterfaceDeclRef))
    {
        if (isAssociatedTypeDecl(requiredMemberDecl.getDecl()))
            continue;
        if (requiredMemberDecl.as<DerivativeRequirementDecl>())
            continue;
        ensureDecl(requiredMemberDecl, DeclCheckState::ReadyForReference);
        auto requiredMemberDeclRef = m_astBuilder->getLookupDeclRef(
            subTypeConformsToSuperInterfaceWitness,
            requiredMemberDecl.getDecl());
        auto requirementSatisfied = findWitnessForInterfaceRequirement(
            context,
            subType,
            superInterfaceType,
            inheritanceDecl,
            superInterfaceDeclRef,
            requiredMemberDeclRef,
            witnessTable,
            subTypeConformsToSuperInterfaceWitness);

        result = result && requirementSatisfied;
    }

    // Extensions that apply to the interface type can create new conformances
    // for the concrete types that inherit from the interface.
    //
    // These new conformances should not be able to introduce new *requirements*
    // for an implementing interface (although they currently can), but we
    // still need to go through this logic to find the appropriate value
    // that will satisfy the requirement in these cases, and also to put
    // the required entry into the witness table for the interface itself.
    //
    // TODO: This logic is a bit slippery, and we need to figure out what
    // it means in the context of separate compilation. If module A defines
    // an interface IA, module B defines a type C that conforms to IA, and then
    // module C defines an extension that makes IA conform to IC, then it is
    // unreasonable to expect the {B:IA} witness table to contain an entry
    // corresponding to {IA:IC}.
    //
    // The simple answer then would be that the {IA:IC} conformance should be
    // fixed, with a single witness table for {IA:IC}, but then what should
    // happen in B explicitly conformed to IC already?
    //
    // For now we will just walk through the extensions that are known at
    // the time we are compiling and handle those, and punt on the larger issue
    // for a bit longer.
    //
    auto candidateExtensions = getCandidateExtensions(superInterfaceDeclRef, this);
    for (const auto& candidateExt : candidateExtensions)
    {
        // We need to apply the extension to the interface type that our
        // concrete type is inheriting from.
        //
        Type* targetType = DeclRefType::create(m_astBuilder, thisTypeDeclRef);
        auto parentDeclRef = applyExtensionToType(candidateExt, targetType);
        if (!parentDeclRef)
            continue;

        // Only inheritance clauses from the extension matter right now.
        for (auto requiredInheritanceDecl :
             getMembersOfType<InheritanceDecl>(m_astBuilder, candidateExt))
        {
            auto requiredInheritanceDeclRef = m_astBuilder->getLookupDeclRef(
                subTypeConformsToSuperInterfaceWitness,
                requiredInheritanceDecl.getDecl());
            auto requirementSatisfied = findWitnessForInterfaceRequirement(
                context,
                subType,
                superInterfaceType,
                inheritanceDecl,
                superInterfaceDeclRef,
                requiredInheritanceDeclRef,
                witnessTable,
                subTypeConformsToSuperInterfaceWitness);

            result = result && requirementSatisfied;
        }
    }

    // The conformance was satisfied if all the requirements were satisfied.
    //
    return result;
}

bool SemanticsVisitor::checkConformanceToType(
    ConformanceCheckingContext* context,
    Type* subType,
    InheritanceDecl* inheritanceDecl,
    Type* superType,
    SubtypeWitness* subIsSuperWitness,
    WitnessTable* witnessTable)
{
    if (witnessTable->isExtern)
        return true;

    if (auto supereclRefType = as<DeclRefType>(superType))
    {
        auto superTypeDeclRef = supereclRefType->getDeclRef();
        if (auto superInterfaceDeclRef = superTypeDeclRef.as<InterfaceDecl>())
        {
            // The type is stating that it conforms to an interface.
            // We need to check that it provides all of the members
            // required by that interface.
            return checkInterfaceConformance(
                context,
                subType,
                superType,
                inheritanceDecl,
                superInterfaceDeclRef,
                subIsSuperWitness,
                witnessTable);
        }
        else if (auto superStructDeclRef = superTypeDeclRef.as<StructDecl>())
        {
            // The type is saying it inherits from a `struct`,
            // which doesn't require any checking at present
            return true;
        }
    }
    if (!as<ErrorType>(superType))
    {
        getSink()->diagnose(inheritanceDecl, Diagnostics::invalidTypeForInheritance, superType);
    }
    return false;
}

static bool _doesTypeDeclHaveDefinition(ContainerDecl* decl)
{
    if (auto aggTypeDecl = as<AggTypeDecl>(decl))
        return aggTypeDecl->hasBody;
    return false;
}

bool SemanticsVisitor::checkConformance(
    Type* subType,
    InheritanceDecl* inheritanceDecl,
    ContainerDecl* parentDecl)
{
    auto superType = inheritanceDecl->base.type;

    if (auto declRefType = as<DeclRefType>(subType))
    {
        auto declRef = declRefType->getDeclRef();

        if (auto superDeclRefType = as<DeclRefType>(superType))
        {
            auto superTypeDecl = superDeclRefType->getDeclRef().getDecl();
            if (superTypeDecl->findModifier<ComInterfaceAttribute>())
            {
                // A struct cannot implement a COM Interface.
                if (auto classDecl = as<ClassDecl>(superTypeDecl))
                {
                    // OK.
                    SLANG_UNUSED(classDecl);
                }
                else if (auto subInterfaceDecl = as<InterfaceDecl>(superTypeDecl))
                {
                    if (!subInterfaceDecl->findModifier<ComInterfaceAttribute>())
                    {
                        getSink()->diagnose(
                            inheritanceDecl,
                            Diagnostics::interfaceInheritingComMustBeCom);
                    }
                }
                else if (const auto structDecl = as<StructDecl>(superTypeDecl))
                {
                    getSink()->diagnose(
                        inheritanceDecl,
                        Diagnostics::structCannotImplementComInterface);
                }
            }
        }

        // Don't check conformances for abstract types that
        // are being used to express *required* conformances.
        if (auto assocTypeDeclRef = declRef.as<AssocTypeDecl>())
        {
            // An associated type declaration represents a requirement
            // in an outer interface declaration, and its members
            // (type constraints) represent additional requirements.
            return true;
        }
        else if (auto interfaceDeclRef = declRef.as<InterfaceDecl>())
        {
            // HACK: Our semantics as they stand today are that an
            // `extension` of an interface that adds a new inheritance
            // clause acts *as if* that inheritnace clause had been
            // attached to the original `interface` decl: that is,
            // it adds additional requirements.
            //
            // This is *not* a reasonable semantic to keep long-term,
            // but it is required for some of our current example
            // code to work.
            return true;
        }
    }

    // Look at the type being inherited from, and validate
    // appropriately.

    DeclaredSubtypeWitness* subIsSuperWitness =
        m_astBuilder->getDeclaredSubtypeWitness(subType, superType, makeDeclRef(inheritanceDecl));

    ConformanceCheckingContext context;
    context.conformingType = subType;
    context.parentDecl = parentDecl;


    RefPtr<WitnessTable> witnessTable = inheritanceDecl->witnessTable;
    if (!witnessTable)
    {
        witnessTable = new WitnessTable();
        witnessTable->baseType = superType;
        witnessTable->witnessedType = subType;
        witnessTable->isExtern =
            (!_doesTypeDeclHaveDefinition(parentDecl) && parentDecl->hasModifier<ExternModifier>());
        inheritanceDecl->witnessTable = witnessTable;
    }

    if (!checkConformanceToType(
            &context,
            subType,
            inheritanceDecl,
            superType,
            subIsSuperWitness,
            witnessTable))
    {
        return false;
    }

    return true;
}

void SemanticsVisitor::checkExtensionConformance(ExtensionDecl* decl)
{
    auto declRef = createDefaultSubstitutionsIfNeeded(m_astBuilder, this, makeDeclRef(decl))
                       .as<ExtensionDecl>();
    auto targetType = getTargetType(m_astBuilder, declRef);

    for (auto inheritanceDecl : decl->getMembersOfType<InheritanceDecl>())
    {
        checkConformance(targetType, inheritanceDecl, decl);
    }
}

void SemanticsVisitor::checkDifferentiableMembersInType(AggTypeDecl* decl)
{
    for (auto member : decl->getMembersOfType<VarDeclBase>())
    {
        if (auto derivativeAttr = member->findModifier<DerivativeMemberAttribute>())
        {
            checkDerivativeMemberAttributeReferences(member, derivativeAttr);
        }
    }
}

void SemanticsVisitor::checkAggTypeConformance(AggTypeDecl* decl)
{
    // After we've checked members, we need to go through
    // any inheritance clauses on the type itself, and
    // confirm that the type actually provides whatever
    // those clauses require.

    if (const auto interfaceDecl = as<InterfaceDecl>(decl))
    {
        // Don't check that an interface conforms to the
        // things it inherits from.
    }
    else if (const auto assocTypeDecl = as<AssocTypeDecl>(decl))
    {
        // Don't check that an associated type decl conforms to the
        // things it inherits from.
    }
    else
    {
        // For non-interface types we need to check conformance.
        //

        auto astBuilder = getASTBuilder();

        auto declRef = createDefaultSubstitutionsIfNeeded(astBuilder, this, makeDeclRef(decl))
                           .as<AggTypeDeclBase>();
        auto type = DeclRefType::create(astBuilder, declRef);

        // TODO: Need to figure out what this should do for
        // `abstract` types if we ever add them. Should they
        // be required to implement all interface requirements,
        // just with `abstract` methods that replicate things?
        // (That's what C# does).

        // Make a copy of inhertanceDecls firstsince `checkConformance` may modify decl->members.
        auto inheritanceDecls = decl->getMembersOfType<InheritanceDecl>().toList();
        for (auto inheritanceDecl : inheritanceDecls)
        {
            // Special handling for when we check for conformance against `IDifferentiable`
            // We will reference-checking for the [DerivativeMember(DiffType.member)]
            // attributes here, since they have to be performed after types can be referenced
            // and before conformance checking, where this information can be used to synthesize
            // member methods (such as `dzero`, `dadd`, etc..)
            //
            if (inheritanceDecl->getSup().type->equals(
                    astBuilder->getDifferentiableInterfaceType()))
                checkDifferentiableMembersInType(decl);

            checkConformance(type, inheritanceDecl, decl);
        }

        // Successful conformance checking may have created new witness tables.
        // Increment epoch to invalidate the cache, so subsequent canonical types are
        // re-calculated.
        //
        // TODO: Is it really necessary to invalidate globally? Maybe there's a way to invalidate
        // only the types that are affected by these interface decls.
        //
        astBuilder->incrementEpoch();
    }
}

void SemanticsDeclBasesVisitor::_validateCrossModuleInheritance(
    AggTypeDeclBase* decl,
    InheritanceDecl* inheritanceDecl)
{
    // Within a single module, users should be allowed to inherit
    // one type from another more or less freely, so long as they
    // don't violate fundamental validity conditions around
    // inheritance.
    //
    // When an inheritance relationship is declared in one module,
    // and the base type is in another module, we may want to
    // enforce more restrictions. As a strong example, we probably
    // don't want people to declare their own subtype of `int`
    // or `Texture2D<float4>`.
    //
    // We start by checking if the type being inherited from is
    // a decl-ref type, since that means it refers to a declaration
    // that can be localized to its original module.
    //
    auto baseType = inheritanceDecl->base.type;
    auto baseDeclRefType = as<DeclRefType>(baseType);
    if (!baseDeclRefType)
    {
        return;
    }
    auto baseDecl = baseDeclRefType->getDeclRef().getDecl();

    // Using the parent/child hierarchy baked into `Decl`s we
    // can find the modules that contain both the `decl` doing
    // the inheriting, and the `baseDeclRefType` that is being
    // inherited from.
    //
    // If those modules are the same, then we aren't seeing any
    // kind of cross-module inheritance here, and there is nothing
    // that needs enforcing.
    //
    auto moduleWithInheritance = getModule(decl);
    auto moduleWithBaseType = getModule(baseDecl);
    if (moduleWithInheritance == moduleWithBaseType)
    {
        return;
    }

    if (baseDecl->hasModifier<SealedAttribute>())
    {
        // If the original declaration had the `[sealed]` attribute on it,
        // then it explicitly does *not* allow inheritance from other
        // modules.
        //
        getSink()->diagnose(
            inheritanceDecl,
            Diagnostics::cannotInheritFromExplicitlySealedDeclarationInAnotherModule,
            baseType,
            moduleWithBaseType->getModuleDecl()->getName());
        return;
    }
    else if (baseDecl->hasModifier<OpenAttribute>())
    {
        // Conversely, if the original declaration had the `[open]` attribute
        // on it, then it explicit *does* allow inheritance from other
        // modules.
        //
        // In this case we don't need to check anything: the inheritance
        // is allowed.
    }
    else if (as<InterfaceDecl>(baseDecl))
    {
        // If an interface isn't explicitly marked `[open]` or `[sealed]`,
        // then the default behavior is to treat it as `[open]`, since
        // interfaces are most often used to define protocols that
        // users of a module can opt into.
    }
    else
    {
        // For any non-interface type, if the declaration didn't specify
        // `[open]` or `[sealed]` then we assume `[sealed]` is the default.
        //
        getSink()->diagnose(
            inheritanceDecl,
            Diagnostics::cannotInheritFromImplicitlySealedDeclarationInAnotherModule,
            baseType,
            moduleWithBaseType->getModuleDecl()->getName());
        return;
    }
}

void SemanticsDeclBasesVisitor::visitInterfaceDecl(InterfaceDecl* decl)
{
    SLANG_OUTER_SCOPE_CONTEXT_DECL_RAII(this, decl);
    checkVisibility(decl);
    for (auto inheritanceDecl : decl->getMembersOfType<InheritanceDecl>())
    {
        ensureDecl(inheritanceDecl, DeclCheckState::CanUseBaseOfInheritanceDecl);
        auto baseType = inheritanceDecl->base.type;

        // It is possible that there was an error in checking the base type
        // expression, and in such a case we shouldn't emit a cascading error.
        //
        if (const auto baseErrorType = as<ErrorType>(baseType))
        {
            continue;
        }

        // An `interface` type can only inherit from other `interface` types.
        //
        // TODO: In the long run it might make sense for an interface to support
        // an inheritance clause naming a non-interface type, with the meaning
        // that any type that implements the interface must be a sub-type of the
        // type named in the inheritance clause.
        //
        auto baseDeclRefType = as<DeclRefType>(baseType);
        if (!baseDeclRefType)
        {
            getSink()->diagnose(
                inheritanceDecl,
                Diagnostics::baseOfInterfaceMustBeInterface,
                decl,
                baseType);
            continue;
        }

        auto baseDeclRef = baseDeclRefType->getDeclRef();
        auto baseInterfaceDeclRef = baseDeclRef.as<InterfaceDecl>();
        if (!baseInterfaceDeclRef)
        {
            getSink()->diagnose(
                inheritanceDecl,
                Diagnostics::baseOfInterfaceMustBeInterface,
                decl,
                baseType);
            continue;
        }

        // TODO: At this point we have the `baseInterfaceDeclRef`
        // and could use it to perform further validity checks,
        // and/or to build up a more refined representation of
        // the inheritance graph for this type (e.g., a "class
        // precedence list").
        //
        // E.g., we can/should check that we aren't introducing
        // a circular inheritance relationship.

        _validateCrossModuleInheritance(decl, inheritanceDecl);
    }

    if (decl->findModifier<ComInterfaceAttribute>())
    {
        // `associatedtype` declaration is not allowed in a COM interface declaration.
        for (auto associatedType : decl->getMembersOfType<AssocTypeDecl>())
        {
            getSink()->diagnose(associatedType, Diagnostics::associatedTypeNotAllowInComInterface);
        }
    }
}

void SemanticsDeclBasesVisitor::visitStructDecl(StructDecl* decl)
{
    // A `struct` type can only inherit from `struct` or `interface` types.
    //
    // Furthermore, only the first inheritance clause (in source
    // order) is allowed to declare a base `struct` type.
    //
    SLANG_OUTER_SCOPE_CONTEXT_DECL_RAII(this, decl);

    Index inheritanceClauseCounter = 0;
    for (auto inheritanceDecl : decl->getMembersOfType<InheritanceDecl>())
    {
        Index inheritanceClauseIndex = inheritanceClauseCounter++;

        ensureDecl(inheritanceDecl, DeclCheckState::CanUseBaseOfInheritanceDecl);
        auto baseType = inheritanceDecl->base.type;

        // It is possible that there was an error in checking the base type
        // expression, and in such a case we shouldn't emit a cascading error.
        //
        if (const auto baseErrorType = as<ErrorType>(baseType))
        {
            continue;
        }

        auto baseDeclRefType = as<DeclRefType>(baseType);
        if (!baseDeclRefType)
        {
            getSink()->diagnose(
                inheritanceDecl,
                Diagnostics::baseOfStructMustBeStructOrInterface,
                decl,
                baseType);
            continue;
        }

        auto baseDeclRef = baseDeclRefType->getDeclRef();
        if (auto baseInterfaceDeclRef = baseDeclRef.as<InterfaceDecl>())
        {
        }
        else if (auto baseStructDeclRef = baseDeclRef.as<StructDecl>())
        {
            // To simplify the task of reading and maintaining code,
            // we require that when a `struct` inherits from another
            // `struct`, the base `struct` is the first item in
            // the list of bases (before any interfaces).
            //
            // This constraint also has the secondary effect of restricting
            // it so that a `struct` cannot multiply inherit from other
            // `struct` types.
            //
            if (inheritanceClauseIndex != 0)
            {
                getSink()->diagnose(
                    inheritanceDecl,
                    Diagnostics::baseStructMustBeListedFirst,
                    decl,
                    baseType);
            }
        }
        else
        {
            getSink()->diagnose(
                inheritanceDecl,
                Diagnostics::baseOfStructMustBeStructOrInterface,
                decl,
                baseType);
            continue;
        }

        if (this->getOptionSet().getBoolOption(CompilerOptionName::ZeroInitialize) &&
            !isFromCoreModule(decl))
        {
            // Force add IDefaultInitializable to any struct missing (transitively)
            // `IDefaultInitializable`.
            auto* defaultInitializableType = m_astBuilder->getDefaultInitializableType();
            if (!isSubtype(
                    DeclRefType::create(m_astBuilder, decl),
                    defaultInitializableType,
                    IsSubTypeOptions::NoCaching))
            {
                InheritanceDecl* conformanceDecl = m_astBuilder->create<InheritanceDecl>();
                conformanceDecl->loc = decl->loc;
                conformanceDecl->base.type = defaultInitializableType;
                conformanceDecl->nameAndLoc.name = getName("$inheritance");
                decl->addMember(conformanceDecl);
            }
        }

        // TODO: At this point we have the `baseDeclRef`
        // and could use it to perform further validity checks,
        // and/or to build up a more refined representation of
        // the inheritance graph for this type (e.g., a "class
        // precedence list").
        //
        // E.g., we can/should check that we aren't introducing
        // a circular inheritance relationship.

        _validateCrossModuleInheritance(decl, inheritanceDecl);
    }
}

void SemanticsDeclBasesVisitor::visitClassDecl(ClassDecl* decl)
{
    // A `class` type can only inherit from `class` or `interface` types.
    //
    // Furthermore, only the first inheritance clause (in source
    // order) is allowed to declare a base `class` type.
    //
    SLANG_OUTER_SCOPE_CONTEXT_DECL_RAII(this, decl);

    Index inheritanceClauseCounter = 0;
    for (auto inheritanceDecl : decl->getMembersOfType<InheritanceDecl>())
    {
        Index inheritanceClauseIndex = inheritanceClauseCounter++;

        ensureDecl(inheritanceDecl, DeclCheckState::CanUseBaseOfInheritanceDecl);
        auto baseType = inheritanceDecl->base.type;

        // It is possible that there was an error in checking the base type
        // expression, and in such a case we shouldn't emit a cascading error.
        //
        if (const auto baseErrorType = as<ErrorType>(baseType))
        {
            continue;
        }

        auto baseDeclRefType = as<DeclRefType>(baseType);
        if (!baseDeclRefType)
        {
            getSink()->diagnose(
                inheritanceDecl,
                Diagnostics::baseOfClassMustBeClassOrInterface,
                decl,
                baseType);
            continue;
        }

        auto baseDeclRef = baseDeclRefType->getDeclRef();
        if (auto baseInterfaceDeclRef = baseDeclRef.as<InterfaceDecl>())
        {
        }
        else if (auto baseStructDeclRef = baseDeclRef.as<ClassDecl>())
        {
            // To simplify the task of reading and maintaining code,
            // we require that when a `class` inherits from another
            // `class`, the base `class` is the first item in
            // the list of bases (before any interfaces).
            //
            // This constraint also has the secondary effect of restricting
            // it so that a `struct` cannot multiply inherit from other
            // `struct` types.
            //
            if (inheritanceClauseIndex != 0)
            {
                getSink()->diagnose(
                    inheritanceDecl,
                    Diagnostics::baseClassMustBeListedFirst,
                    decl,
                    baseType);
            }
        }
        else
        {
            getSink()->diagnose(
                inheritanceDecl,
                Diagnostics::baseOfClassMustBeClassOrInterface,
                decl,
                baseType);
            continue;
        }

        // TODO: At this point we have the `baseDeclRef`
        // and could use it to perform further validity checks,
        // and/or to build up a more refined representation of
        // the inheritance graph for this type (e.g., a "class
        // precedence list").
        //
        // E.g., we can/should check that we aren't introducing
        // a circular inheritance relationship.

        _validateCrossModuleInheritance(decl, inheritanceDecl);
    }
}

bool SemanticsVisitor::isIntegerBaseType(BaseType baseType)
{
    return (BaseTypeInfo::getInfo(baseType).flags & BaseTypeInfo::Flag::Integer) != 0;
}

bool SemanticsVisitor::isScalarIntegerType(Type* type)
{
    auto basicType = as<BasicExpressionType>(type);
    if (!basicType)
        return false;
    auto baseType = basicType->getBaseType();
    return isIntegerBaseType(baseType) || baseType == BaseType::Bool;
}

bool SemanticsVisitor::isHalfType(Type* type)
{
    auto basicType = as<BasicExpressionType>(type);
    if (!basicType)
        return false;
    auto baseType = basicType->getBaseType();
    return baseType == BaseType::Half;
}

bool SemanticsVisitor::isValidCompileTimeConstantType(Type* type)
{
    return isScalarIntegerType(type) || isEnumType(type);
}

bool SemanticsVisitor::isIntValueInRangeOfType(IntegerLiteralValue value, Type* type)
{
    auto basicType = as<BasicExpressionType>(type);
    if (!basicType)
        return false;

    switch (basicType->getBaseType())
    {
    case BaseType::UInt8:
        return (value >= 0 && value <= std::numeric_limits<uint8_t>::max()) || (value == -1);
    case BaseType::UInt16:
        return (value >= 0 && value <= std::numeric_limits<uint16_t>::max()) || (value == -1);
    case BaseType::UInt:
#if SLANG_PTR_IS_32
    case BaseType::UIntPtr:
#endif
        return (value >= 0 && value <= std::numeric_limits<uint32_t>::max()) || (value == -1);
    case BaseType::UInt64:
#if SLANG_PTR_IS_64
    case BaseType::UIntPtr:
#endif
        return true;
    case BaseType::Int8:
        return value >= std::numeric_limits<int8_t>::min() &&
               value <= std::numeric_limits<int8_t>::max();
    case BaseType::Int16:
        return value >= std::numeric_limits<int16_t>::min() &&
               value <= std::numeric_limits<int16_t>::max();
    case BaseType::Int:
#if SLANG_PTR_IS_32
    case BaseType::IntPtr:
#endif
        return value >= std::numeric_limits<int32_t>::min() &&
               value <= std::numeric_limits<int32_t>::max();
    case BaseType::Int64:
#if SLANG_PTR_IS_64
    case BaseType::IntPtr:
#endif
        return value >= std::numeric_limits<int64_t>::min() &&
               value <= std::numeric_limits<int64_t>::max();

    case BaseType::Half:
        return value >= -2048 && value <= 2048;
    default:
        return false;
    }
}

void SemanticsVisitor::validateEnumTagType(Type* type, SourceLoc const& loc)
{
    // Allow the built-in integer types.
    //
    if (isScalarIntegerType(type))
        return;

    // By default, don't allow other types to be used
    // as an `enum` tag type.
    //
    getSink()->diagnose(loc, Diagnostics::invalidEnumTagType, type);
}

bool SemanticsVisitor::_hasExplicitConstructor(StructDecl* structDecl, bool checkBaseType)
{
    if (!structDecl)
        return false;

    auto _hasExplicitCtor = [](AggTypeDecl* aggDecl) -> bool
    {
        // First check if the extension of this struct defines an explicit constructor.
        for (auto ctor : aggDecl->getMembersOfType<ConstructorDecl>())
        {
            // constructor that is not synthesized must be user defined.
            if (ctor->findModifier<SynthesizedModifier>() == nullptr)
            {
                return true;
            }
        }
        return false;
    };

    if (_hasExplicitCtor(structDecl))
        return true;

    if (!checkBaseType)
        return false;

    for (auto inheritanceMember : structDecl->getMembersOfType<InheritanceDecl>())
    {
        auto baseTypeDecl = isDeclRefTypeOf<AggTypeDecl>(inheritanceMember->base.type);
        if (baseTypeDecl && !as<InterfaceDecl>(baseTypeDecl))
        {
            if (_hasExplicitCtor(baseTypeDecl.getDecl()))
                return true;
        }
    }
    return false;
}

void SemanticsDeclBasesVisitor::visitEnumDecl(EnumDecl* decl)
{
    SLANG_OUTER_SCOPE_CONTEXT_DECL_RAII(this, decl);
    checkVisibility(decl);

    // An `enum` type can inherit from interfaces, and also
    // from a single "tag" type that must:
    //
    // * be a built-in integer type
    // * come first in the list of base types
    //
    Index inheritanceClauseCounter = 0;
    Type* tagType = nullptr;
    InheritanceDecl* tagTypeInheritanceDecl = nullptr;
    for (auto inheritanceDecl : decl->getMembersOfType<InheritanceDecl>())
    {
        Index inheritanceClauseIndex = inheritanceClauseCounter++;

        ensureDecl(inheritanceDecl, DeclCheckState::CanUseBaseOfInheritanceDecl);
        auto baseType = inheritanceDecl->base.type;

        // It is possible that there was an error in checking the base type
        // expression, and in such a case we shouldn't emit a cascading error.
        //
        if (const auto baseErrorType = as<ErrorType>(baseType))
        {
            continue;
        }

        auto baseDeclRefType = as<DeclRefType>(baseType);
        if (!baseDeclRefType)
        {
            getSink()->diagnose(
                inheritanceDecl,
                Diagnostics::baseOfEnumMustBeIntegerOrInterface,
                decl,
                baseType);
            continue;
        }

        auto baseDeclRef = baseDeclRefType->getDeclRef();
        if (auto baseInterfaceDeclRef = baseDeclRef.as<InterfaceDecl>())
        {
            _validateCrossModuleInheritance(decl, inheritanceDecl);
        }
        else if (auto baseStructDeclRef = baseDeclRef.as<StructDecl>())
        {
            // To simplify the task of reading and maintaining code,
            // we require that when an `enum` declares an explicit
            // underlying tag type using an inheritance clause, that
            // type must be the first item in the list of bases.
            //
            // This constraint also has the secondary effect of restricting
            // it so that an `enum` can't possibly have multiple tag
            // types declared.
            //
            if (inheritanceClauseIndex != 0)
            {
                getSink()->diagnose(
                    inheritanceDecl,
                    Diagnostics::tagTypeMustBeListedFirst,
                    decl,
                    baseType);
            }
            else
            {
                tagType = baseType;
                tagTypeInheritanceDecl = inheritanceDecl;
            }

            // Note: we do *not* apply the code that validates
            // cross-module inheritance to a base that represnts
            // a tag type, because declaring a tag type for an
            // `enum` doesn't actually make it into a subtype
            // of the tag type, and thus doesn't violate the
            // rules when the tag type is `sealed`.
        }
        else
        {
            getSink()->diagnose(
                inheritanceDecl,
                Diagnostics::baseOfEnumMustBeIntegerOrInterface,
                decl,
                baseType);
            continue;
        }
    }

    // If a tag type has not been set, then we
    // default it to the built-in `int` type.
    //
    // TODO: In the far-flung future we may want to distinguish
    // `enum` types that have a "raw representation" like this from
    // ones that are purely abstract and don't expose their
    // type of their tag.
    //
    if (!tagType)
    {
        tagType = m_astBuilder->getIntType();
    }
    else
    {
        // TODO: Need to establish that the tag
        // type is suitable. (e.g., if we are going
        // to allow raw values for case tags to be
        // derived automatically, then the tag
        // type needs to be some kind of integer type...)
        //
        // For now we will just be harsh and require it
        // to be one of a few builtin types.
        validateEnumTagType(tagType, tagTypeInheritanceDecl->loc);

        // Note: The `InheritanceDecl` that introduces a tag
        // type isn't actually representing a super-type of
        // the `enum`, and things like name lookup need to
        // know to ignore that "inheritance" relationship.
        //
        // We add a modifier to the `InheritanceDecl` to ensure
        // that it can be detected and ignored by such steps.
        //
        addModifier(tagTypeInheritanceDecl, m_astBuilder->create<IgnoreForLookupModifier>());
    }
    decl->tagType = tagType;


    // An `enum` type should automatically conform to the `__EnumType` interface.
    // The compiler needs to insert this conformance behind the scenes, and this
    // seems like the best place to do it.
    {
        // First, look up the type of the `__EnumType` interface.
        Type* enumTypeType = getASTBuilder()->getEnumTypeType();

        InheritanceDecl* enumConformanceDecl = m_astBuilder->create<InheritanceDecl>();
        enumConformanceDecl->loc = decl->loc;
        enumConformanceDecl->base.type = getASTBuilder()->getEnumTypeType();
        decl->addMember(enumConformanceDecl);

        // The `__EnumType` interface has one required member, the `__Tag` type.
        // We need to satisfy this requirement automatically, rather than require
        // the user to actually declare a member with this name (otherwise we wouldn't
        // let them define a tag value with the name `__Tag`).
        //
        RefPtr<WitnessTable> witnessTable = new WitnessTable();
        witnessTable->baseType = enumConformanceDecl->base.type;
        witnessTable->witnessedType = enumTypeType;
        enumConformanceDecl->witnessTable = witnessTable;

        Name* tagAssociatedTypeName = getSession()->getNameObj("__Tag");
        Decl* tagAssociatedTypeDecl = nullptr;
        if (auto enumTypeTypeDeclRefType = dynamicCast<DeclRefType>(enumTypeType))
        {
            if (auto enumTypeTypeInterfaceDecl =
                    as<InterfaceDecl>(enumTypeTypeDeclRefType->getDeclRef().getDecl()))
            {
                for (auto memberDecl : enumTypeTypeInterfaceDecl->members)
                {
                    if (memberDecl->getName() == tagAssociatedTypeName)
                    {
                        tagAssociatedTypeDecl = memberDecl;
                        break;
                    }
                }
            }
        }
        if (!tagAssociatedTypeDecl)
        {
            SLANG_DIAGNOSE_UNEXPECTED(
                getSink(),
                decl,
                "failed to find built-in declaration '__Tag'");
        }

        // Okay, add the conformance witness for `__Tag` being satisfied by `tagType`
        witnessTable->add(tagAssociatedTypeDecl, RequirementWitness(tagType));

        // TODO: we actually also need to synthesize a witness for the conformance of `tagType`
        // to the `__BuiltinIntegerType` interface, because that is a constraint on the
        // associated type `__Tag`.

        // TODO: eventually we should consider synthesizing other requirements for
        // the min/max tag values, or the total number of tags, so that people don't
        // have to declare these as additional cases.

        enumConformanceDecl->setCheckState(DeclCheckState::DefinitionChecked);
    }
}

void SemanticsDeclBodyVisitor::visitEnumDecl(EnumDecl* decl)
{
    SLANG_OUTER_SCOPE_CONTEXT_DECL_RAII(this, decl);

    auto enumType = DeclRefType::create(m_astBuilder, makeDeclRef(decl));

    auto tagType = decl->tagType;

    auto isEnumFlags = decl->hasModifier<FlagsAttribute>();

    // Check the enum cases in order.
    for (auto caseDecl : decl->getMembersOfType<EnumCaseDecl>())
    {
        // Each case defines a value of the enum's type.
        //
        // TODO: If we ever support enum cases with payloads,
        // then they would probably have a type that is a
        // `FunctionType` from the payload types to the
        // enum type.
        //
        // TODO(tfoley): the case should grab its type  when
        // doing its own header checking, rather than rely on this...
        caseDecl->type.type = enumType;

        ensureDecl(caseDecl, DeclCheckState::DefinitionChecked);
    }

    // For any enum case that didn't provide an explicit
    // tag value, derived an appropriate tag value.
    IntegerLiteralValue defaultTag = isEnumFlags ? 1 : 0;
    for (auto caseDecl : decl->getMembersOfType<EnumCaseDecl>())
    {
        if (auto explicitTagValExpr = caseDecl->tagExpr)
        {
            // This tag has an initializer, so it should establish
            // the tag value for a successor case that doesn't
            // provide an explicit tag.

            IntVal* explicitTagVal =
                tryConstantFoldExpr(explicitTagValExpr, ConstantFoldingKind::CompileTime, nullptr);
            if (explicitTagVal)
            {
                if (auto constIntVal = as<ConstantIntVal>(explicitTagVal))
                {
                    defaultTag = constIntVal->getValue();
                }
                else
                {
                    // TODO: need to handle other possibilities here
                    getSink()->diagnose(explicitTagValExpr, Diagnostics::unexpectedEnumTagExpr);
                }
            }
            else
            {
                // If this happens, then the explicit tag value expression
                // doesn't seem to be a constant after all. In this case
                // we expect the checking logic to have applied already.
            }
        }
        else
        {
            // This tag has no initializer, so it should use
            // the default tag value we are tracking.
            IntegerLiteralExpr* tagValExpr = m_astBuilder->create<IntegerLiteralExpr>();
            tagValExpr->loc = caseDecl->loc;
            tagValExpr->type = QualType(tagType);
            tagValExpr->value = defaultTag;

            caseDecl->tagExpr = tagValExpr;
        }

        // Default tag for the next case will be one more than
        // for the most recent case.
        //
        if (!isEnumFlags)
            defaultTag++;
        else
        {
            if (defaultTag == 0)
                defaultTag = 1;
            else
                defaultTag <<= 1;
        }
    }
}

void SemanticsDeclBodyVisitor::visitEnumCaseDecl(EnumCaseDecl* decl)
{
    // An enum case had better appear inside an enum!
    //
    // TODO: Do we need/want to support generic cases some day?
    auto parentEnumDecl = as<EnumDecl>(decl->parentDecl);
    SLANG_ASSERT(parentEnumDecl);

    decl->type.type = DeclRefType::create(m_astBuilder, makeDeclRef(parentEnumDecl));

    // The tag type should have already been set by
    // the surrounding `enum` declaration.
    auto tagType = parentEnumDecl->tagType;
    SLANG_ASSERT(tagType);

    // Need to check the init expression, if present, since
    // that represents the explicit tag for this case.
    if (auto initExpr = decl->tagExpr)
    {
        initExpr = CheckTerm(initExpr);
        initExpr = coerce(CoercionSite::General, tagType, initExpr);

        // We want to enforce that this is an integer constant
        // expression.
        decl->tagVal = CheckIntegerConstantExpression(
            initExpr,
            IntegerConstantExpressionCoercionType::AnyInteger,
            nullptr,
            ConstantFoldingKind::CompileTime);

        decl->tagExpr = initExpr;
    }
}

void SemanticsVisitor::ensureDeclBase(
    DeclBase* declBase,
    DeclCheckState state,
    SemanticsContext* baseContext)
{
    if (auto decl = as<Decl>(declBase))
    {
        ensureDecl(decl, state, baseContext);
    }
    else if (auto declGroup = as<DeclGroup>(declBase))
    {
        for (auto dd : declGroup->decls)
        {
            ensureDecl(dd, state, baseContext);
        }
    }
    else
    {
        SLANG_UNEXPECTED("unknown case for declaration");
    }
}

void SemanticsDeclHeaderVisitor::visitTypeDefDecl(TypeDefDecl* decl)
{
    SemanticsVisitor visitor(withDeclToExcludeFromLookup(decl));
    decl->type = visitor.CheckProperType(decl->type);
    checkVisibility(decl);
}

void SemanticsDeclHeaderVisitor::visitGlobalGenericParamDecl(GlobalGenericParamDecl* decl)
{
    // global generic param only allowed in global scope
    auto program = as<ModuleDecl>(decl->parentDecl);
    if (!program)
        getSink()->diagnose(decl, Slang::Diagnostics::globalGenParamInGlobalScopeOnly);
}

void SemanticsDeclHeaderVisitor::visitAssocTypeDecl(AssocTypeDecl* decl)
{
    // assoctype only allowed in an interface
    auto interfaceDecl = as<InterfaceDecl>(decl->parentDecl);
    if (!interfaceDecl)
        getSink()->diagnose(decl, Slang::Diagnostics::assocTypeInInterfaceOnly);
    checkVisibility(decl);
}

SemanticsContext SemanticsDeclBodyVisitor::registerDifferentiableTypesForFunc(
    FunctionDeclBase* decl)
{
    auto newContext = withParentFunc(decl);
    if (newContext.getParentDifferentiableAttribute())
    {
        // Register additional types outside the function body first.
        auto oldAttr = m_parentDifferentiableAttr;
        m_parentDifferentiableAttr = newContext.getParentDifferentiableAttribute();
        for (auto param : decl->getParameters())
            maybeRegisterDifferentiableType(m_astBuilder, param->type.type);
        maybeRegisterDifferentiableType(m_astBuilder, decl->returnType.type);
        if (as<ConstructorDecl>(decl) || !isEffectivelyStatic(decl))
        {
            auto parentDecl = getParentDecl(decl);
            auto parentDeclRef =
                createDefaultSubstitutionsIfNeeded(m_astBuilder, this, makeDeclRef(parentDecl));
            auto thisType = calcThisType(parentDeclRef);
            maybeRegisterDifferentiableType(m_astBuilder, thisType);
        }
        m_parentDifferentiableAttr = oldAttr;
    }
    return newContext;
}

void SemanticsDeclBodyVisitor::visitFunctionDeclBase(FunctionDeclBase* decl)
{
    auto newContext = registerDifferentiableTypesForFunc(decl);
    decl->body = maybeParseStmt(decl->body, newContext);
    if (const auto body = decl->body)
    {
        checkStmt(decl->body, newContext);
    }
}

void SemanticsVisitor::getGenericParams(
    GenericDecl* decl,
    List<Decl*>& outParams,
    List<GenericTypeConstraintDecl*>& outConstraints)
{
    for (auto dd : decl->members)
    {
        if (dd == decl->inner)
            continue;

        if (auto typeParamDecl = as<GenericTypeParamDecl>(dd))
            outParams.add(typeParamDecl);
        else if (auto valueParamDecl = as<GenericValueParamDecl>(dd))
            outParams.add(valueParamDecl);
        else if (auto constraintDecl = as<GenericTypeConstraintDecl>(dd))
            outConstraints.add(constraintDecl);
    }
}

bool SemanticsVisitor::doGenericSignaturesMatch(
    GenericDecl* left,
    GenericDecl* right,
    DeclRef<Decl>* outSpecializedRightInner)
{
    // Our first goal here is to determine if `left` and
    // `right` have equivalent lists of explicit
    // generic parameters.
    //
    // Once we have determined that the explicit generic
    // parameters match, we will look at the constraints
    // placed on those parameters to see if they are
    // equivalent.
    //
    // We thus start by extracting the explicit parameters
    // and the constraints from each declaration.
    //
    List<Decl*> leftParams;
    List<GenericTypeConstraintDecl*> leftConstraints;
    getGenericParams(left, leftParams, leftConstraints);

    List<Decl*> rightParams;
    List<GenericTypeConstraintDecl*> rightConstraints;
    getGenericParams(right, rightParams, rightConstraints);

    // For there to be any hope of a match, the two decls
    // need to have the same number of explicit parameters.
    //
    Index paramCount = leftParams.getCount();
    if (paramCount != rightParams.getCount())
        return false;

    // Next we will walk through the parameters and look
    // for a pair-wise match.
    //
    for (Index pp = 0; pp < paramCount; ++pp)
    {
        Decl* leftParam = leftParams[pp];
        Decl* rightParam = rightParams[pp];

        if (const auto leftTypeParam = as<GenericTypeParamDecl>(leftParam))
        {
            if (const auto rightTypeParam = as<GenericTypeParamDecl>(rightParam))
            {
                // Right now any two type parameters are a match.
                // Names are irrelevant to matching, and any constraints
                // on the type parameters are represented as implicit
                // extra parameters of the generic.
                //
                // TODO: If we ever supported type parameters with
                // higher kinds we might need to make a check here
                // that the kind of each parameter matches (which
                // would in a sense be a kind of recursive check
                // of the generic signature of the parameter).
                //
                continue;
            }
        }
        else if (auto leftValueParam = as<GenericValueParamDecl>(leftParam))
        {
            if (auto rightValueParam = as<GenericValueParamDecl>(rightParam))
            {
                // In this case we have two generic value parameters,
                // and they should only be considered to match if
                // they have the same type.
                //
                // Note: We are assuming here that the type of a value
                // parameter cannot be dependent on any of the type
                // parameters in the same signature. This is a reasonable
                // assumption for now, but could get thorny down the road.
                //
                if (!leftValueParam->getType()->equals(rightValueParam->getType()))
                {
                    // If the value parameters have non-matching types,
                    // then the full generic signatures do not match.
                    //
                    return false;
                }

                // Generic value parameters with the same type are
                // always considered to match.
                //
                continue;
            }
        }

        // If we get to this point, then we have two parameters that
        // were of different syntatic categories (e.g., one type parameter
        // and one value parameter), so the signatures clearly don't match.
        //
        return false;
    }

    // At this point we know that the explicit generic parameters
    // of `left` and `right` are aligned, but we need to check
    // that the constraints that each declaration places on
    // its parameters match.
    //
    // A first challenge that arises is that `left` and `right`
    // will each express the constraints in terms of their
    // own parameters. For example, consider the following
    // declarations:
    //
    //      void foo1<T : IFoo>(T value);
    //      void foo2<U : IFoo>(U value);
    //
    // It is "obvious" to a human that the signatures here
    // match, but `foo1` has a constraint `T : IFoo` while
    // `foo2` has a constraint `U : IFoo`, and since `T`
    // and `U` are distinct `Decl`s, those constraints
    // are not obviously equivalent.
    //
    // We will work around this first issue by creating
    // a substitution taht lists all the parameters of
    // `left`, which we can use to specialize `right`
    // so that it aligns.
    //
    // In terms of the example above, this is like constructing
    // `foo2<T>` so that its constraint, after specialization,
    // looks like `T : IFoo`.
    //
    auto& substInnerRightToLeft = *outSpecializedRightInner;
    List<Val*> leftArgs = getDefaultSubstitutionArgs(m_astBuilder, this, left);
    substInnerRightToLeft =
        m_astBuilder->getGenericAppDeclRef(makeDeclRef(right), leftArgs.getArrayView());

    // We should now be able to enumerate the constraints
    // on `right` in a way that uses the same type parameters
    // as `left`, using `rightDeclRef`.
    //
    // At this point a second problem arises: if/when we support
    // more flexibility in how generic parameter constraints are
    // specified, it will be possible for two declarations to
    // list the "same" constraints in very different ways.
    //
    // For example, if we support a `where` clause for separating
    // the constraints from the parameters, then the following
    // two declarations should have equivalent signatures:
    //
    //      void foo1<T>(T value)
    //          where T : IFoo
    //      { ... }
    //
    //      void foo2<T : IFoo>(T value)
    //      { ... }
    //
    // Similarly, if we allow for general compositions of interfaces
    // to be used as constraints, then there can be more than one
    // way to specify the same constraints:
    //
    //      void foo1<T : IFoo&IBar>(T value);
    //      void foo2<T : IBar&IFoo>(T value);
    //
    // Adding support for equality constraints in `where` clauses
    // also creates opportunities for multiple equivalent expressions:
    //
    //      void foo1<T,U>(...) where T.A == U.A;
    //      void foo2<T,U>(...) where U.A == T.A;
    //
    // A robsut version of the checking logic here should attempt
    // to *canonicalize* all of the constraints. Canonicalization
    // should involve putting constraints into a deterministic
    // order (e.g., for a generic with `<T,U>` all the constraints
    // on `T` should come before those on `U`), rewriting individual
    // constraints into a canonical form (e.g., `T : IFoo & IBar`
    // should turn into two constraints: `T : IFoo` and `T : IBar`),
    // etc.
    //
    // Once the constraints are in a canonical form we should be able
    // to test them for pairwise equivalent. As a safety measure we
    // could also try to test whether one set of constraints implies
    // the other (since implication in both directions should imply
    // equivalence, in which case our canonicalization had better
    // have produced the same result).
    //
    // For now we are taking a simpler short-cut by assuming
    // that constraints are already in a canonical form, which
    // is reasonable for now as the syntax only allows a single
    // constraint per parameter, specified on the parameter itself.
    //
    // Under the assumption of canonical constraints, we can
    // assume that different numbers of constraints must indicate
    // a signature mismatch.
    //
    Index constraintCount = leftConstraints.getCount();
    if (constraintCount != rightConstraints.getCount())
        return false;

    for (Index cc = 0; cc < constraintCount; ++cc)
    {
        // Note that we use a plain `Decl` pointer for the left
        // constraint, but need to use a `DeclRef` for the right
        // constraint so that we can take the substitution
        // arguments into account.
        //
        GenericTypeConstraintDecl* leftConstraint = leftConstraints[cc];
        auto unspecializedRightConstarintDeclRef = createDefaultSubstitutionsIfNeeded(
            m_astBuilder,
            this,
            makeDeclRef(rightConstraints[cc]));
        DeclRef<GenericTypeConstraintDecl> rightConstraint =
            substInnerRightToLeft.substitute(m_astBuilder, unspecializedRightConstarintDeclRef)
                .as<GenericTypeConstraintDecl>();

        // For now, every constraint has the form `sub : sup`
        // to indicate that `sub` must be a subtype of `sup`.
        //
        // Two such constraints are equivalent if their `sub`
        // and `sup` types are pairwise equivalent.
        //
        auto leftSub = leftConstraint->sub.type;
        auto rightSub =
            substInnerRightToLeft.substitute(m_astBuilder, rightConstraint.getDecl()->sub.type);
        if (!leftSub->equals(rightSub))
            return false;

        auto leftSup = leftConstraint->sup.type;
        auto rightSup =
            substInnerRightToLeft.substitute(m_astBuilder, rightConstraint.getDecl()->sup.type);
        if (!leftSup->equals(rightSup))
            return false;
    }

    // If we have checked all of the (canonicalized) constraints
    // and found them to be pairwise equivalent then the two
    // generic signatures seem to match.
    //
    return true;
}

bool SemanticsVisitor::doFunctionSignaturesMatch(DeclRef<FuncDecl> fst, DeclRef<FuncDecl> snd)
{

    // TODO(tfoley): This copies the parameter array, which is bad for performance.
    auto fstParams = getParameters(m_astBuilder, fst).toArray();
    auto sndParams = getParameters(m_astBuilder, snd).toArray();

    // If the functions have different numbers of parameters, then
    // their signatures trivially don't match.
    auto fstParamCount = fstParams.getCount();
    auto sndParamCount = sndParams.getCount();
    if (fstParamCount != sndParamCount)
        return false;

    for (Index ii = 0; ii < fstParamCount; ++ii)
    {
        auto fstParam = fstParams[ii];
        auto sndParam = sndParams[ii];

        // If a given parameter type doesn't match, then signatures don't match
        if (!getType(m_astBuilder, fstParam)->equals(getType(m_astBuilder, sndParam)))
            return false;

        // If one parameter is `out` and the other isn't, then they don't match
        //
        // Note(tfoley): we don't consider `out` and `inout` as distinct here,
        // because there is no way for overload resolution to pick between them.
        if (fstParam.getDecl()->hasModifier<OutModifier>() !=
            sndParam.getDecl()->hasModifier<OutModifier>())
            return false;

        // If one parameter is `ref` and the other isn't, then they don't match.
        //
        if (fstParam.getDecl()->hasModifier<RefModifier>() !=
            sndParam.getDecl()->hasModifier<RefModifier>())
            return false;

        // If one parameter is `constref` and the other isn't, then they don't match.
        //
        if (fstParam.getDecl()->hasModifier<ConstRefModifier>() !=
            sndParam.getDecl()->hasModifier<ConstRefModifier>())
            return false;
    }

    // Note(tfoley): return type doesn't enter into it, because we can't take
    // calling context into account during overload resolution.

    return true;
}

List<Val*> getDefaultSubstitutionArgs(
    ASTBuilder* astBuilder,
    SemanticsVisitor* semantics,
    GenericDecl* genericDecl)
{
    List<Val*> args;
    if (astBuilder->m_cachedGenericDefaultArgs.tryGetValue(genericDecl, args))
        return args;

    for (auto mm : genericDecl->members)
    {
        if (auto genericTypeParamDecl = as<GenericTypeParamDecl>(mm))
        {
            args.add(DeclRefType::create(
                astBuilder,
                astBuilder->getDirectDeclRef(genericTypeParamDecl)));
        }
        else if (auto genericTypePackParamDecl = as<GenericTypePackParamDecl>(mm))
        {
            auto packType = DeclRefType::create(
                astBuilder,
                astBuilder->getDirectDeclRef(genericTypePackParamDecl));
            args.add(packType);
        }
        else if (auto genericValueParamDecl = as<GenericValueParamDecl>(mm))
        {
            if (semantics)
                semantics->ensureDecl(genericValueParamDecl, DeclCheckState::ReadyForLookup);

            args.add(astBuilder->getOrCreate<GenericParamIntVal>(
                genericValueParamDecl->getType(),
                astBuilder->getDirectDeclRef(genericValueParamDecl)));
        }
    }

    bool shouldCache = true;

    // create default substitution arguments for constraints
    for (auto mm : genericDecl->members)
    {
        if (auto genericTypeConstraintDecl = as<GenericTypeConstraintDecl>(mm))
        {
            if (semantics)
                semantics->ensureDecl(genericTypeConstraintDecl, DeclCheckState::ReadyForReference);
            auto constraintDeclRef =
                astBuilder->getDirectDeclRef<GenericTypeConstraintDecl>(genericTypeConstraintDecl);
            auto supType = getSup(astBuilder, constraintDeclRef);
            if (!supType)
            {
                args.add(astBuilder->getErrorType());
                shouldCache = false;
                continue;
            }
            auto witness = astBuilder->getDeclaredSubtypeWitness(
                getSub(astBuilder, constraintDeclRef),
                getSup(astBuilder, constraintDeclRef),
                constraintDeclRef);
            // TODO: this is an ugly hack to prevent crashing.
            // In early stages of compilation witness->sub and witness->sup may not be checked yet.
            // When semanticVisitor is present we have used that to ensure the type is checked.
            // However due to how the code is written we cannot guarantee semanticVisitor is always
            // available here, and if we can't get the checked sup/sub type this subst is incomplete
            // and should not be cached.
            if (!witness->getSub())
                shouldCache = false;
            args.add(witness);
        }
    }

    if (shouldCache)
        astBuilder->m_cachedGenericDefaultArgs[genericDecl] = args;

    return args;
}

typedef Dictionary<Name*, CallableDecl*> TargetDeclDictionary;

static void _addTargetModifiers(CallableDecl* decl, TargetDeclDictionary& ioDict)
{
    if (auto specializedModifier = decl->findModifier<SpecializedForTargetModifier>())
    {
        // If it's specialized for target it should have a body...
        if (auto funcDecl = as<FunctionDeclBase>(decl))
        {
            // Normally if we have specialization for target it must have a body.
            if (funcDecl->body == nullptr)
            {
                // If it doesn't have a body but does have a target intrinsic/SPIRVInstructionOp
                // it's probably ok

                SLANG_ASSERT(
                    funcDecl->findModifier<SPIRVInstructionOpAttribute>() ||
                    funcDecl->findModifier<TargetIntrinsicModifier>());
            }
        }
        Name* targetName = specializedModifier->targetToken.getName();

        ioDict.addIfNotExists(targetName, decl);
    }
    else
    {
        for (auto modifier : decl->getModifiersOfType<TargetIntrinsicModifier>())
        {
            Name* targetName = modifier->targetToken.getName();
            ioDict.addIfNotExists(targetName, decl);
        }

        auto funcDecl = as<FunctionDeclBase>(decl);
        if (funcDecl && funcDecl->body)
        {
            // Should only be one body if it isn't specialized for target.
            // Use nullptr for this scenario
            ioDict.addIfNotExists(nullptr, decl);
        }
    }
}

Result SemanticsVisitor::checkFuncRedeclaration(FuncDecl* newDecl, FuncDecl* oldDecl)
{
    // There are a few different cases that this function needs
    // to check for:
    //
    // * If `newDecl` and `oldDecl` have different signatures such
    //   that they can always be distinguished at call sites, then
    //   they don't conflict and don't count as redeclarations.
    //
    // * If `newDecl` and `oldDecl` have matching signatures, but
    //   differ in return type (or other details that would affect
    //   compatibility), then the declarations conflict and an
    //   error needs to be diagnosed.
    //
    // * If `newDecl` and `oldDecl` have matching/compatible sigantures,
    //   but differ when it comes to target-specific overloading,
    //   then they can co-exist.
    //
    // * If `newDecl` and `oldDecl` have matching/compatible signatures
    //   and are specialized for the same target(s), then only
    //   one can have a body (in which case the other is a forward declaration),
    //   or else we have a redefinition error.

    auto newGenericDecl = as<GenericDecl>(newDecl->parentDecl);
    auto oldGenericDecl = as<GenericDecl>(oldDecl->parentDecl);

    // If one declaration is a prefix/postfix operator, and the
    // other is not a matching operator, then don't consider these
    // to be re-declarations.
    //
    // Note(tfoley): Any attempt to call such an operator using
    // ordinary function-call syntax (if we decided to allow it)
    // would be ambiguous in such a case, of course.
    //
    if (newDecl->hasModifier<PrefixModifier>() != oldDecl->hasModifier<PrefixModifier>())
        return SLANG_OK;
    if (newDecl->hasModifier<PostfixModifier>() != oldDecl->hasModifier<PostfixModifier>())
        return SLANG_OK;

    // If one is generic and the other isn't, then there is no match.
    if ((newGenericDecl != nullptr) != (oldGenericDecl != nullptr))
        return SLANG_OK;

    // We are going to be comparing the signatures of the
    // two functions, but if they are *generic* functions
    // then we will need to compare them with consistent
    // specializations in place.
    //
    // We'll go ahead and create some (unspecialized) declaration
    // references here, just to be prepared.
    //
    DeclRef<FuncDecl> newDeclRef(newDecl);
    DeclRef<FuncDecl> oldDeclRef(oldDecl);

    // If we are working with generic functions, then we need to
    // consider if their generic signatures match.
    //
    if (newGenericDecl)
    {
        // If one declaration is generic, the other must be.
        // (This condition was already checked above)
        //
        SLANG_ASSERT(oldGenericDecl);

        // As part of checking if the generic signatures match,
        // we will produce a substitution that can be used to
        // reference `oldGenericDecl` with the generic parameters
        // substituted for those of `newDecl`.
        //
        // One way to think about it is that if we have these
        // declarations (ignore the name differences...):
        //
        //     // oldDecl:
        //     void foo1<T>(T x);
        //
        //     // newDecl:
        //     void foo2<U>(U x);
        //
        // Then we will compare the parameter types of `foo2`
        // against the specialization `foo1<U>`.
        //
        DeclRef<Decl> specializedOldDeclInner;
        if (!doGenericSignaturesMatch(newGenericDecl, oldGenericDecl, &specializedOldDeclInner))
            return SLANG_OK;

        oldDeclRef = specializedOldDeclInner.as<FuncDecl>();
    }

    // If the parameter signatures don't match, then don't worry
    if (!doFunctionSignaturesMatch(newDeclRef, oldDeclRef))
        return SLANG_OK;

    // If the declatation is declared by 'extern', and new definition is with 'export', then
    // we should let overload resolution to handle it.
    if (oldDecl->hasModifier<ExternModifier>() && newDecl->hasModifier<HLSLExportModifier>())
    {
        return SLANG_OK;
    }

    // If we get this far, then we've got two declarations in the same
    // scope, with the same name and signature, so they appear
    // to be redeclarations.
    //
    // We will track that redeclaration occured, so that we can
    // take it into account for overload resolution.
    //
    // A huge complication that we'll need to deal with is that
    // multiple declarations might introduce default values for
    // (different) parameters, and we might need to merge across
    // all of them (which could get complicated if defaults for
    // parameters can reference earlier parameters).

    // If the previous declaration wasn't already recorded
    // as being part of a redeclaration family, then make
    // it the primary declaration of a new family.
    if (!oldDecl->primaryDecl)
    {
        oldDecl->primaryDecl = oldDecl;
    }

    // The new declaration will belong to the family of
    // the previous one, and so it will share the same
    // primary declaration.
    newDecl->primaryDecl = oldDecl->primaryDecl;
    newDecl->nextDecl = nullptr;

    // Next we want to chain the new declaration onto
    // the linked list of redeclarations.
    auto link = &oldDecl->nextDecl;
    while (*link)
        link = &(*link)->nextDecl;
    *link = newDecl;

    // Now that we've added things to a group of redeclarations,
    // we can do some additional validation.

    // First, we will ensure that the return types match
    // between the declarations, so that they are truly
    // interchangeable.
    //
    // Note(tfoley): If we ever decide to add a beefier type
    // system to Slang, we might allow overloads like this,
    // so long as the desired result type can be disambiguated
    // based on context at the call type. In that case we would
    // consider result types earlier, as part of the signature
    // matching step.
    //
    auto resultType = getResultType(m_astBuilder, newDeclRef);
    auto prevResultType = getResultType(m_astBuilder, oldDeclRef);
    if (!resultType->equals(prevResultType))
    {
        // Bad redeclaration
        getSink()->diagnose(
            newDecl,
            Diagnostics::functionRedeclarationWithDifferentReturnType,
            newDecl->getName(),
            resultType,
            prevResultType);
        getSink()->diagnose(oldDecl, Diagnostics::seePreviousDeclarationOf, newDecl->getName());

        // Don't bother emitting other errors at this point
        return SLANG_FAIL;
    }

    // TODO: Enforce that the new declaration had better
    // not specify a default value for any parameter that
    // already had a default value in a prior declaration.

    // We are going to want to enforce that we cannot have
    // two declarations of a function both specify bodies.
    // Before we make that check, however, we need to deal
    // with the case where the two function declarations
    // might represent different target-specific versions
    // of a function.

    // If both of the declarations have a body, then there
    // is trouble, because we wouldn't know which one to
    // use during code generation.

    // Here to cover the 'bodies'/target_intrinsics, we find all the targets that
    // that are previously defined, and make sure the new definition
    // doesn't try and define what is already defined.
    {
        TargetDeclDictionary currentTargets;
        {
            CallableDecl* curDecl = newDecl->primaryDecl;
            while (curDecl)
            {
                if (curDecl != newDecl)
                {
                    _addTargetModifiers(curDecl, currentTargets);
                }
                curDecl = curDecl->nextDecl;
            }
        }

        // Add the targets for this new decl
        TargetDeclDictionary newTargets;
        _addTargetModifiers(newDecl, newTargets);

        bool hasConflict = false;
        for (auto& [target, value] : newTargets)
        {
            auto found = currentTargets.tryGetValue(target);
            if (found)
            {
                // Redefinition
                if (!hasConflict)
                {
                    getSink()->diagnose(
                        newDecl,
                        Diagnostics::functionRedefinition,
                        newDecl->getName());
                    hasConflict = true;
                }

                auto prevDecl = *found;
                getSink()->diagnose(
                    prevDecl,
                    Diagnostics::seePreviousDefinitionOf,
                    prevDecl->getName());
            }
        }

        if (hasConflict)
        {
            return SLANG_FAIL;
        }
    }

    // At this point we've processed the redeclaration and
    // put it into a group, so there is no reason to keep
    // looping and looking at prior declarations.
    //
    // While no diagnostics have been emitted, we return
    // a failure result from the operation to indicate
    // to the caller that they should stop looping over
    // declarations at this point.
    //
    return SLANG_FAIL;
}

Result SemanticsVisitor::checkRedeclaration(Decl* newDecl, Decl* oldDecl)
{
    // If either of the declarations being looked at is generic, then
    // we want to consider the "inner" declaration instead when
    // making decisions about what to allow or not.
    //
    if (auto newGenericDecl = as<GenericDecl>(newDecl))
        newDecl = newGenericDecl->inner;
    if (auto oldGenericDecl = as<GenericDecl>(oldDecl))
        oldDecl = oldGenericDecl->inner;

    // Functions are special in that we can have many declarations
    // with the same name in a given scope, and it is possible
    // for them to co-exist as overloads, or even just be multiple
    // declarations of the same function (thanks to the inherited
    // legacy of C forward declarations).
    //
    // If both declarations are functions, we will check that
    // they are allowed to co-exist using these more nuanced rules.
    //
    if (auto newFuncDecl = as<FuncDecl>(newDecl))
    {
        if (auto oldFuncDecl = as<FuncDecl>(oldDecl))
        {
            // Both new and old declarations are functions,
            // so redeclaration may be valid.
            return checkFuncRedeclaration(newFuncDecl, oldFuncDecl);
        }
    }

    if (as<ModuleDeclarationDecl>(oldDecl) || as<ModuleDeclarationDecl>(newDecl))
    {
        // It is allowed to have a decl whose name is the same as the module.
        return SLANG_OK;
    }


    // For all other flavors of declaration, we do not
    // allow duplicate declarations with the same name.
    //
    // TODO: We might consider allowing some other cases
    // of overloading that can be safely disambiguated:
    //
    // * A type and a value (function/variable/etc.) of the same name can usually
    //   co-exist because we can distinguish which is needed by context.
    //
    // * Multiple generic types with the same name can co-exist
    //   if their generic parameter lists are sufficient to
    //   tell them apart at a use site.

    // We will diagnose a redeclaration error at the new declaration,
    // and point to the old declaration for context.
    //
    getSink()->diagnose(newDecl, Diagnostics::redeclaration, newDecl->getName());
    getSink()->diagnose(oldDecl, Diagnostics::seePreviousDeclarationOf, oldDecl->getName());
    return SLANG_FAIL;
}


void SemanticsVisitor::checkForRedeclaration(Decl* decl)
{
    // We want to consider a "new" declaration in the context
    // of some parent/container declaration, and compare it
    // to pre-existing "old" declarations of the same name
    // in the same container.
    //
    auto newDecl = decl;
    auto parentDecl = decl->parentDecl;

    // Sanity check: there should always be a parent declaration.
    //
    SLANG_ASSERT(parentDecl);
    if (!parentDecl)
        return;

    // If the declaration is the "inner" declaration of a generic,
    // then we actually want to look one level up, because the
    // peers/siblings of the declaration will belong to the same
    // parent as the generic, not to the generic.
    //
    if (auto genericParentDecl = as<GenericDecl>(parentDecl))
    {
        // Note: we need to check here to be sure `newDecl`
        // is the "inner" declaration and not one of the
        // generic parameters, or else we will end up
        // checking them at the wrong scope.
        //
        if (newDecl == genericParentDecl->inner)
        {
            newDecl = parentDecl;
            parentDecl = genericParentDecl->parentDecl;
        }
    }

    // We will now look for other declarations with
    // the same name in the same parent/container.
    //
    parentDecl->buildMemberDictionary();
    for (auto oldDecl = newDecl->nextInContainerWithSameName; oldDecl;
         oldDecl = oldDecl->nextInContainerWithSameName)
    {
        // For each matching declaration, we will check
        // whether the redeclaration should be allowed,
        // and emit an appropriate diagnostic if not.
        //
        Result checkResult = checkRedeclaration(newDecl, oldDecl);

        // The `checkRedeclaration` function will return a failure
        // status (whether or not it actually emitted a diagnostic)
        // if we should stop checking further redeclarations, because
        // the declaration in question has been dealt with fully.
        //
        if (SLANG_FAILED(checkResult))
            break;
    }
}


void SemanticsDeclHeaderVisitor::visitParamDecl(ParamDecl* paramDecl)
{
    // TODO: This logic should be shared with the other cases of
    // variable declarations. The main reason I am not doing it
    // yet is that we use a `ParamDecl` with a null type as a
    // special case in attribute declarations, and that could
    // trip up the ordinary variable checks.

    auto typeExpr = paramDecl->type;
    if (typeExpr.exp)
    {
        SemanticsVisitor subVisitor(withDeclToExcludeFromLookup(paramDecl));
        typeExpr = subVisitor.CheckUsableType(typeExpr, paramDecl);
        paramDecl->type = typeExpr;
        checkMeshOutputDecl(paramDecl);
    }

    if (auto declRefType = as<DeclRefType>(paramDecl->type.type))
    {
        if (declRefType->getDeclRef().getDecl()->findModifier<NonCopyableTypeAttribute>())
        {
            // Always pass a non-copyable type by reference.
            // Remove all existing direction modifiers, and replace them with a single Ref modifier.
            List<Modifier*> newModifiers;
            bool hasRefModifier = false;
            bool isMutable = false;
            for (auto modifier : paramDecl->modifiers)
            {
                if (as<InModifier>(modifier))
                {
                    continue;
                }
                else if (as<InOutModifier>(modifier) || as<OutModifier>(modifier))
                {
                    isMutable = true;
                    continue;
                }
                if (as<RefModifier>(modifier) || as<ConstRefModifier>(modifier))
                {
                    hasRefModifier = true;
                }
                newModifiers.add(modifier);
            }
            if (!hasRefModifier)
            {
                if (isMutable)
                    newModifiers.add(this->getASTBuilder()->create<RefModifier>());
                else
                    newModifiers.add(this->getASTBuilder()->create<ConstRefModifier>());
            }
            paramDecl->modifiers.first = newModifiers.getFirst();
            for (Index i = 0; i < newModifiers.getCount(); i++)
            {
                if (i < newModifiers.getCount() - 1)
                    newModifiers[i]->next = newModifiers[i + 1];
                else
                    newModifiers[i]->next = nullptr;
            }
        }
    }
    else if (isTypePack(paramDecl->type.type))
    {
        // For now, we only allow parameter packs to be `const`.
        bool hasConstModifier = false;
        for (auto modifier : paramDecl->modifiers)
        {
            if (as<OutModifier>(modifier) || as<InOutModifier>(modifier) ||
                as<RefModifier>(modifier) || as<ConstRefModifier>(modifier))
            {
                getSink()->diagnose(modifier, Diagnostics::parameterPackMustBeConst);
            }
            else if (as<ConstModifier>(modifier))
            {
                hasConstModifier = true;
            }
        }
        if (!hasConstModifier)
        {
            auto constModifier = this->getASTBuilder()->create<ConstModifier>();
            addModifier(paramDecl, constModifier);
        }
    }

    maybeApplyLayoutModifier(paramDecl);

    // Only texture types are allowed to have memory qualifiers on parameters
    if (!paramDecl->type || paramDecl->type->astNodeType != ASTNodeType::TextureType)
    {
        auto MemoryQualifierSet = paramDecl->findModifier<MemoryQualifierSetModifier>();
        if (!MemoryQualifierSet)
            return;
        for (auto mod : MemoryQualifierSet->getModifiers())
            getSink()->diagnose(
                paramDecl,
                Diagnostics::memoryQualifierNotAllowedOnANonImageTypeParameter,
                mod);
    }
}

// This checks that the declaration is marked as "out" and changes the hlsl
// modifier based syntax into a proper type.
void SemanticsDeclHeaderVisitor::checkMeshOutputDecl(VarDeclBase* varDecl)
{
    auto modifier = varDecl->findModifier<HLSLMeshShaderOutputModifier>();
    auto meshOutputType = as<MeshOutputType>(varDecl->type.type);
    bool isMeshOutput = modifier || meshOutputType;

    if (!isMeshOutput)
    {
        return;
    }
    // HLSL requires an 'out' modifier here, but since we don't operate
    // under such strict compatability we can just not warn here.
    if (!varDecl->findModifier<OutModifier>() && modifier)
    {
        getSink()->diagnose(varDecl, Diagnostics::meshOutputMustBeOut);
    }

    //
    // If necessary, convert to our typed representation
    //
    if (!modifier)
    {
        return;
    }
    if (meshOutputType)
    {
        getSink()->diagnose(modifier, Diagnostics::unnecessaryHLSLMeshOutputModifier);
        varDecl->type.type = m_astBuilder->getErrorType();
        return;
    }
    auto indexExpr = as<IndexExpr>(varDecl->type.exp);
    if (!indexExpr)
    {
        getSink()->diagnose(varDecl, Diagnostics::meshOutputMustBeArray);
        varDecl->type.type = m_astBuilder->getErrorType();
        return;
    }
    if (indexExpr->indexExprs.getCount() != 1)
    {
        getSink()->diagnose(varDecl, Diagnostics::meshOutputArrayMustHaveSize);
        varDecl->type.type = m_astBuilder->getErrorType();
        return;
    }
    auto base = ExpectAType(indexExpr->baseExpression);
    auto index = CheckIntegerConstantExpression(
        indexExpr->indexExprs[0],
        IntegerConstantExpressionCoercionType::AnyInteger,
        nullptr,
        ConstantFoldingKind::LinkTime,
        getSink());
    if (!index)
        return;
    Type* d = m_astBuilder->getMeshOutputTypeFromModifier(modifier, base, index);
    varDecl->type.type = d;
}

void SemanticsDeclBodyVisitor::visitParamDecl(ParamDecl* paramDecl)
{
    auto typeExpr = paramDecl->type;

    if (!as<ArrayExpressionType>(paramDecl->type) &&
        doesTypeHaveTag(paramDecl->type, TypeTag::Unsized))
    {
        getSink()->diagnose(paramDecl, Diagnostics::paramCannotBeUnsized, paramDecl);
    }

    // The "initializer" expression for a parameter represents
    // a default argument value to use if an explicit one is
    // not supplied.
    if (auto initExpr = paramDecl->initExpr)
    {
        // We must check the expression and coerce it to the
        // actual type of the parameter.
        //
        initExpr = CheckTerm(initExpr);
        initExpr = coerce(CoercionSite::Initializer, typeExpr.type, initExpr);
        paramDecl->initExpr = initExpr;

        // TODO: a default argument expression needs to
        // conform to other constraints to be valid.
        // For example, it should not be allowed to refer
        // to other parameters of the same function (or maybe
        // only the parameters to its left...).

        // A default argument value should not be allowed on an
        // `out` or `inout` parameter.
        //
        // TODO: we could relax this by requiring the expression
        // to yield an lvalue, but that seems like a feature
        // with limited practical utility (and an easy source
        // of confusing behavior).
        //
        // Note: the `InOutModifier` class inherits from `OutModifier`,
        // so we only need to check for the base case.
        //
        if (paramDecl->findModifier<OutModifier>())
        {
            getSink()->diagnose(initExpr, Diagnostics::outputParameterCannotHaveDefaultValue);
        }
    }
}


static SeqStmt* _ensureCtorBodyIsSeqStmt(ASTBuilder* m_astBuilder, ConstructorDecl* decl)
{
    // It is possible BlockStmt has a child with the type of
    // `ExpressionStmt` if an existing constructor has only 1
    // expression. This would be a senario we need to
    // put the `ExpressionStmt` inside a `SeqStmt`.
    auto stmt = as<BlockStmt>(decl->body);
    if (!stmt)
    {
        auto tmpExpr = decl->body;
        auto blockStmt = m_astBuilder->create<BlockStmt>();
        blockStmt->body = tmpExpr;
        decl->body = blockStmt;
        stmt = blockStmt;
    }
    if (!as<SeqStmt>(stmt->body))
    {
        auto tmpExpr = stmt->body;
        auto seqStmt = m_astBuilder->create<SeqStmt>();
        seqStmt->stmts.add(tmpExpr);
        stmt->body = seqStmt;
        return seqStmt;
    }
    return as<SeqStmt>(stmt->body);
}

MemberExpr* SemanticsDeclBodyVisitor::createMemberExpr(
    ThisExpr* thisExpr,
    Scope* scope,
    Decl* member)
{
    MemberExpr* memberExpr = m_astBuilder->create<MemberExpr>();
    memberExpr->baseExpression = thisExpr;
    memberExpr->declRef = member->getDefaultDeclRef();
    memberExpr->scope = scope;
    memberExpr->loc = member->loc;
    memberExpr->name = member->getName();
    memberExpr->type = GetTypeForDeclRef(member->getDefaultDeclRef(), member->loc);

    return memberExpr;
}

Expr* SemanticsDeclBodyVisitor::createCtorParamExpr(ConstructorDecl* ctor, Index paramIndex)
{
    if (paramIndex < ctor->members.getCount())
    {
        if (auto param = as<ParamDecl>(ctor->members[paramIndex]))
        {
            auto paramType = param->getType();
            auto paramExpr = m_astBuilder->create<VarExpr>();
            paramExpr->scope = ctor->ownedScope;
            paramExpr->declRef = param;
            paramExpr->type = paramType;
            paramExpr->loc = param->loc;
            return paramExpr;
        }
    }
    return nullptr;
}

void SemanticsDeclBodyVisitor::synthesizeCtorBodyForBases(
    ConstructorDecl* ctor,
    List<DeclAndCtorInfo>& inheritanceDefaultCtorList,
    ThisExpr* thisExpr,
    SeqStmt* seqStmtChild,
    bool isMemberInitCtor,
    Index& ioParamIndex)
{
    for (auto& declInfo : inheritanceDefaultCtorList)
    {
        ConstructorDecl* baseCtor = nullptr;
        List<Expr*> argumentList;

        if (isMemberInitCtor)
        {
            // Pick the parameters from the member initialize ctor, and use them to invoke the
            // base's member initialize ctor. e.g. base->init(...);
            baseCtor = _getSynthesizedConstructor(
                declInfo.parent,
                ConstructorDecl::ConstructorFlavor::SynthesizedMemberInit);
            if (baseCtor)
            {
                Index idx = 0;
                for (; idx < baseCtor->getParameters().getCount(); idx++)
                {
                    auto paramExpr = createCtorParamExpr(ctor, idx);
                    argumentList.add(paramExpr);
                }
                ioParamIndex += idx;
            }
        }

        // It's possible that the base type doesn't have a member initialize ctor, in this case, we
        // should use the default ctor.
        if (!baseCtor)
        {
            // If the base type has no default constructor, it means that it's not default
            // initializable, e.g. unsized array, resource type, etc. We will not synthesize code to
            // initialize it.
            if (!declInfo.defaultCtor)
                continue;
            baseCtor = declInfo.defaultCtor;
        }

        auto declRefType = as<DeclRefType>(declInfo.type);

        auto ctorToInvoke = m_astBuilder->create<VarExpr>();
        ctorToInvoke->declRef = declRefType->getDeclRef();
        ctorToInvoke->name = baseCtor->getName();
        ctorToInvoke->loc = baseCtor->loc;
        ctorToInvoke->type = m_astBuilder->getFuncType(ArrayView<Type*>(), declRefType);

        auto invoke = m_astBuilder->create<InvokeExpr>();
        invoke->functionExpr = ctorToInvoke;
        invoke->arguments.addRange(argumentList);

        auto assign = m_astBuilder->create<AssignExpr>();
        assign->left = coerce(CoercionSite::Initializer, declRefType, thisExpr);
        assign->right = invoke;

        auto stmt = m_astBuilder->create<ExpressionStmt>();
        stmt->expression = assign;
        stmt->loc = ctor->loc;

        seqStmtChild->stmts.add(stmt);
    }
}

void SemanticsDeclBodyVisitor::synthesizeCtorBodyForMember(
    ConstructorDecl* ctor,
    Decl* member,
    ThisExpr* thisExpr,
    Dictionary<Decl*, Expr*>& cachedDeclToCheckedVar,
    SeqStmt* seqStmtChild,
    bool isMemberInitCtor,
    Index& paramIndex)
{
    auto varDeclBase = as<VarDeclBase>(member);

    // Static variables are initialized at start of runtime, not inside a constructor
    // Once thing to notice is that if a member variable doesn't have name, it must be synthesized
    // instead of defined by user, we should not put it into the constructor because it's not a real
    // member.
    if (!varDeclBase || varDeclBase->hasModifier<HLSLStaticModifier>() ||
        varDeclBase->getName() == nullptr)
        return;

    Expr* initExpr = nullptr;
    auto structDecl = as<StructDecl>(member->parentDecl);
    bool useParamList = isMemberInitCtor;
    useParamList = isMemberInitCtor && structDecl->m_membersVisibleInCtor.contains(varDeclBase);

    if (!useParamList)
    {
        // If this is not a synthesized constructor (e.g. explicit ctor), or
        // the member has no visibility, we can only use it's init expression to initialize it.
        if (!varDeclBase->initExpr)
            return;
        initExpr = varDeclBase->initExpr;
    }
    else
    {
        // Find the corresponding parameter, if we can't find it, there
        // must be something wrong, it indicates that the ctor signature
        // is incorrect that the parameter list doesn't match the member list.
        initExpr = createCtorParamExpr(ctor, paramIndex++);
        if (!initExpr)
        {
            const char* structName =
                (structDecl->getName() ? structDecl->getName()->text.begin() : "unknown");
            StringBuilder msg;
            msg << "Fail to synthesize the member initialize constructor for struct '" << structName
                << "', the parameter list doesn't match the member list.";
            SLANG_ABORT_COMPILATION(msg.produceString().begin());
        }
    }

    MemberExpr* memberExpr = createMemberExpr(thisExpr, ctor->ownedScope, member);
    if (!memberExpr->type.isLeftValue)
        return;

    auto assign = m_astBuilder->create<AssignExpr>();
    assign->left = memberExpr;
    assign->right = initExpr;
    assign->loc = member->loc;

    auto stmt = m_astBuilder->create<ExpressionStmt>();
    stmt->expression = assign;
    stmt->loc = member->loc;

    Expr* checkedMemberVarExpr;
    if (cachedDeclToCheckedVar.containsKey(member))
        checkedMemberVarExpr = cachedDeclToCheckedVar[member];
    else
    {
        checkedMemberVarExpr = CheckTerm(memberExpr);
        cachedDeclToCheckedVar.add({member, checkedMemberVarExpr});
    }

    seqStmtChild->stmts.add(stmt);
}


void SemanticsDeclBodyVisitor::synthesizeCtorBody(
    DeclAndCtorInfo& structDeclInfo,
    List<DeclAndCtorInfo>& inheritanceDefaultCtorList,
    StructDecl* structDecl)
{
    Dictionary<Decl*, Expr*> cachedDeclToCheckedVar;
    for (auto ctor : structDeclInfo.ctorList)
    {
        auto seqStmt = _ensureCtorBodyIsSeqStmt(m_astBuilder, ctor);
        auto seqStmtChild = m_astBuilder->create<SeqStmt>();
        seqStmtChild->stmts.reserve(
            inheritanceDefaultCtorList.getCount() + structDecl->members.getCount());

        ThisExpr* thisExpr = m_astBuilder->create<ThisExpr>();
        thisExpr->scope = ctor->ownedScope;
        thisExpr->type = ctor->returnType.type;

        // We treat the ctor with parameters and all parameters have default value as default ctor
        // as well, but the method to synthesize them are totally different, therefore, we need to
        // differentiate them here.
        bool isMemberInitCtor =
            ctor->containsFlavor(ConstructorDecl::ConstructorFlavor::SynthesizedMemberInit);

        // When we synthesize the member initialize constructor, we need to use the parameters in
        // the function body, so this inout parameter is used to keep track of the index of the
        // parameters.
        Index ioParamIndex = 0;

        // The first step is to synthesize the initialization of the base member.
        synthesizeCtorBodyForBases(
            ctor,
            inheritanceDefaultCtorList,
            thisExpr,
            seqStmtChild,
            isMemberInitCtor,
            ioParamIndex);

        // Then synthesize the initialization of the other members.
        for (auto& m : structDecl->members)
        {
            synthesizeCtorBodyForMember(
                ctor,
                m,
                thisExpr,
                cachedDeclToCheckedVar,
                seqStmtChild,
                isMemberInitCtor,
                ioParamIndex);
        }

        if (seqStmtChild->stmts.getCount() != 0)
        {
            seqStmt->stmts.insert(0, seqStmtChild);
        }
    }
}

void SemanticsDeclBodyVisitor::visitAggTypeDecl(AggTypeDecl* aggTypeDecl)
{
    if (aggTypeDecl->hasTag(TypeTag::Incomplete) && aggTypeDecl->hasModifier<HLSLExportModifier>())
    {
        getSink()->diagnose(aggTypeDecl->loc, Diagnostics::cannotExportIncompleteType, aggTypeDecl);
    }

    auto structDecl = as<StructDecl>(aggTypeDecl);
    if (!structDecl)
        return;

    List<DeclAndCtorInfo> inheritanceDefaultCtorList{};
    for (auto inheritanceMember : structDecl->getMembersOfType<InheritanceDecl>())
    {
        auto declRefType = as<DeclRefType>(inheritanceMember->base.type);
        if (!declRefType)
            continue;
        auto structOfInheritance = as<StructDecl>(declRefType->getDeclRef().getDecl());
        if (!structOfInheritance)
            continue;
        inheritanceDefaultCtorList.add(
            DeclAndCtorInfo(m_astBuilder, this, structOfInheritance, declRefType, true));
    }
    DeclAndCtorInfo structDeclInfo = DeclAndCtorInfo(
        m_astBuilder,
        this,
        structDecl,
        calcThisType(makeDeclRef(structDecl)),
        false);

    // ensure all varDecl members are processed up to SemanticsBodyVisitor so we can be sure that if
    // init expressions of members are to be synthisised, they are.
    bool isDefaultInitializableType = isSubtype(
        DeclRefType::create(m_astBuilder, structDecl),
        m_astBuilder->getDefaultInitializableType(),
        IsSubTypeOptions::None);
    for (auto m : structDecl->members)
    {
        auto varDeclBase = as<VarDeclBase>(m);
        if (!varDeclBase)
            continue;
        ensureDecl(m->getDefaultDeclRef(), DeclCheckState::DefaultConstructorReadyForUse);
        if (!isDefaultInitializableType || varDeclBase->initExpr)
            continue;
        varDeclBase->initExpr = constructDefaultInitExprForType(this, varDeclBase);
    }

    synthesizeCtorBody(structDeclInfo, inheritanceDefaultCtorList, structDecl);

    if (structDeclInfo.defaultCtor)
    {
        auto seqStmt = as<SeqStmt>(as<BlockStmt>(structDeclInfo.defaultCtor->body)->body);
        if (seqStmt && seqStmt->stmts.getCount() == 0)
        {
            structDecl->members.remove(structDeclInfo.defaultCtor);
            structDecl->invalidateMemberDictionary();
            structDecl->buildMemberDictionary();
        }
    }
}

Stmt* SemanticsVisitor::maybeParseStmt(Stmt* stmt, const SemanticsContext& context)
{
    if (auto unparsedStmt = as<UnparsedStmt>(stmt))
    {
        // Parse the statement now, and check it.
        SemanticsVisitor subVisitor(context);
        TokenList tokenList;
        tokenList.m_tokens = _Move(unparsedStmt->tokens);
        return parseUnparsedStmt(
            m_astBuilder,
            &subVisitor,
            getShared()->getTranslationUnitRequest(),
            unparsedStmt->sourceLanguage,
            unparsedStmt->isInVariadicGenerics,
            tokenList,
            getShared()->getSink(),
            unparsedStmt->currentScope,
            unparsedStmt->outerScope);
    }
    return stmt;
}

void SemanticsDeclHeaderVisitor::cloneModifiers(Decl* dest, Decl* src)
{
    dest->modifiers = src->modifiers;
}
void SemanticsDeclHeaderVisitor::setFuncTypeIntoRequirementDecl(
    CallableDecl* decl,
    FuncType* funcType)
{
    if (!funcType)
        return;
    decl->returnType.type = funcType->getResultType();
    decl->errorType.type = funcType->getErrorType();
    for (Index i = 0; i < funcType->getParamCount(); i++)
    {
        auto paramType = funcType->getParamType(i);
        if (auto dirType = as<ParamDirectionType>(paramType))
            paramType = dirType->getValueType();
        auto param = m_astBuilder->create<ParamDecl>();
        param->type.type = paramType;
        auto paramDir = funcType->getParamDirection(i);
        switch (paramDir)
        {
        case ParameterDirection::kParameterDirection_InOut:
            addModifier(param, m_astBuilder->create<InOutModifier>());
            break;
        case ParameterDirection::kParameterDirection_Out:
            addModifier(param, m_astBuilder->create<OutModifier>());
            break;
        case ParameterDirection::kParameterDirection_Ref:
            addModifier(param, m_astBuilder->create<RefModifier>());
            break;
        case ParameterDirection::kParameterDirection_ConstRef:
            addModifier(param, m_astBuilder->create<ConstRefModifier>());
            break;
        default:
            break;
        }
        decl->addMember(param);
    }
}

void SemanticsDeclHeaderVisitor::checkDifferentiableCallableCommon(CallableDecl* decl)
{
    if (auto interfaceDecl = findParentInterfaceDecl(decl))
    {
        bool isDiffFunc = false;
        if (decl->hasModifier<ForwardDifferentiableAttribute>() ||
            decl->hasModifier<BackwardDifferentiableAttribute>())
        {
            auto reqDecl = m_astBuilder->create<ForwardDerivativeRequirementDecl>();
            reqDecl->originalRequirementDecl = decl;
            cloneModifiers(reqDecl, decl);
            auto declRef = createDefaultSubstitutionsIfNeeded(m_astBuilder, this, makeDeclRef(decl))
                               .as<CallableDecl>();
            auto diffFuncType = getForwardDiffFuncType(getFuncType(m_astBuilder, declRef));
            setFuncTypeIntoRequirementDecl(reqDecl, as<FuncType>(diffFuncType));
            interfaceDecl->addMember(reqDecl);

            if (!decl->hasModifier<NoDiffThisAttribute>())
            {
                // Build decl-ref-type from interface.
                auto thisType = DeclRefType::create(
                    m_astBuilder,
                    createDefaultSubstitutionsIfNeeded(
                        m_astBuilder,
                        this,
                        makeDeclRef(interfaceDecl->getThisTypeDecl())));

                // If the interface is differentiable, make the this type a pair.
                if (tryGetDifferentialType(getASTBuilder(), thisType))
                    reqDecl->diffThisType = getDifferentialPairType(thisType);
            }

            auto reqRef = m_astBuilder->create<DerivativeRequirementReferenceDecl>();
            reqRef->referencedDecl = reqDecl;
            decl->addMember(reqRef);
            isDiffFunc = true;
        }
        if (decl->hasModifier<BackwardDifferentiableAttribute>())
        {
            // Requirement for backward derivative.
            auto declRef = createDefaultSubstitutionsIfNeeded(m_astBuilder, this, makeDeclRef(decl))
                               .as<CallableDecl>();
            auto originalFuncType = getFuncType(m_astBuilder, declRef);
            auto diffFuncType = as<FuncType>(getBackwardDiffFuncType(originalFuncType));
            {
                auto reqDecl = m_astBuilder->create<BackwardDerivativeRequirementDecl>();
                reqDecl->originalRequirementDecl = decl;
                cloneModifiers(reqDecl, decl);
                setFuncTypeIntoRequirementDecl(reqDecl, diffFuncType);
                interfaceDecl->addMember(reqDecl);
                if (!decl->hasModifier<NoDiffThisAttribute>())
                {
                    // Build decl-ref-type for this-type.
                    auto thisType = DeclRefType::create(
                        m_astBuilder,
                        createDefaultSubstitutionsIfNeeded(
                            m_astBuilder,
                            this,
                            makeDeclRef(interfaceDecl->getThisTypeDecl())));

                    // If the interface is differentiable, make the this type a pair.
                    if (tryGetDifferentialType(getASTBuilder(), thisType))
                        reqDecl->diffThisType = getDifferentialPairType(thisType);
                }

                auto reqRef = m_astBuilder->create<DerivativeRequirementReferenceDecl>();
                reqRef->referencedDecl = reqDecl;
                decl->addMember(reqRef);
            }
            isDiffFunc = true;
        }
        if (isDiffFunc)
        {
            auto interfaceDeclRef =
                createDefaultSubstitutionsIfNeeded(m_astBuilder, this, makeDeclRef(interfaceDecl));
            auto interfaceType = DeclRefType::create(m_astBuilder, interfaceDeclRef);
            bool noDiffThisRequirement = !isTypeDifferentiable(interfaceType);
            if (noDiffThisRequirement)
            {
                auto noDiffThisModifier = m_astBuilder->create<NoDiffThisAttribute>();
                addModifier(decl, noDiffThisModifier);
            }
        }
    }
    if (decl->findModifier<DifferentiableAttribute>())
    {
        // Add `no_diff` modifiers to parameters.
        // This is necessary to preserve no-diff-ness for generic function before and after
        // specialization.
        for (auto paramDecl : decl->getParameters())
        {
            if (!paramDecl->type.type)
                continue;
            if (!isTypeDifferentiable(paramDecl->type.type))
            {
                if (!paramDecl->hasModifier<NoDiffModifier>())
                {
                    auto noDiffModifier = m_astBuilder->create<NoDiffModifier>();
                    noDiffModifier->keywordName = getSession()->getNameObj("no_diff");
                    addModifier(paramDecl, noDiffModifier);
                }
            }
            if (!paramDecl->hasModifier<NoDiffModifier>())
            {
                if (auto modifier = paramDecl->findModifier<ConstRefModifier>())
                {
                    getSink()->diagnose(
                        modifier,
                        Diagnostics::cannotUseConstRefOnDifferentiableParameter);
                }
            }
        }
        if (!isEffectivelyStatic(decl))
        {
            auto constrefAttr = decl->findModifier<ConstRefAttribute>();
            auto refAttr = decl->findModifier<RefAttribute>();
            if (constrefAttr || refAttr)
            {
                if (isTypeDifferentiable(calcThisType(getParentDecl(decl))))
                {
                    getSink()->diagnose(
                        constrefAttr,
                        Diagnostics::cannotUseConstRefOnDifferentiableMemberMethod);
                }
            }
        }
    }
}

void SemanticsDeclHeaderVisitor::checkCallableDeclCommon(CallableDecl* decl)
{
    for (auto paramDecl : decl->getParameters())
    {
        ensureDecl(paramDecl, DeclCheckState::ReadyForReference);
    }

    auto errorType = decl->errorType;
    if (errorType.exp)
    {
        errorType = CheckProperType(errorType);
    }
    else
    {
        errorType = TypeExp(m_astBuilder->getBottomType());
    }
    decl->errorType = errorType;

    checkDifferentiableCallableCommon(decl);

    // If this method is intended to be a CUDA kernel, verify that the return type is void.
    if (decl->findModifier<CudaKernelAttribute>())
    {
        if (decl->returnType.type && !decl->returnType.type->equals(m_astBuilder->getVoidType()))
        {
            getSink()->diagnose(decl, Diagnostics::cudaKernelMustReturnVoid);
        }
    }

    checkVisibility(decl);
}

void SemanticsDeclHeaderVisitor::visitFuncDecl(FuncDecl* funcDecl)
{
    auto resultType = funcDecl->returnType;
    if (resultType.exp)
    {
        resultType = CheckProperType(resultType);
    }
    else if (!funcDecl->returnType.type)
    {
        resultType = TypeExp(m_astBuilder->getVoidType());
    }
    funcDecl->returnType = resultType;

    checkCallableDeclCommon(funcDecl);
}

IntegerLiteralValue SemanticsVisitor::GetMinBound(IntVal* val)
{
    if (auto constantVal = as<ConstantIntVal>(val))
        return constantVal->getValue();

    // TODO(tfoley): Need to track intervals so that this isn't just a lie...
    return 1;
}

void SemanticsVisitor::maybeInferArraySizeForVariable(VarDeclBase* varDecl)
{
    // Not an array?
    auto arrayType = as<ArrayExpressionType>(varDecl->type);
    if (!arrayType)
        return;

    // Explicit element count given?
    if (!isUnsizedArrayType(arrayType))
        return;

    // No initializer?
    auto initExpr = varDecl->initExpr;
    if (!initExpr)
        return;

    IntVal* elementCount = nullptr;

    // Is the type of the initializer an array type?
    if (auto arrayInitType = as<ArrayExpressionType>(initExpr->type))
    {
        elementCount = arrayInitType->getElementCount();
    }
    else
    {
        // Nothing to do: we couldn't infer a size
        return;
    }

    // Create a new array type based on the size we found,
    // and install it into our type.
    varDecl->type.type = getArrayType(m_astBuilder, arrayType->getElementType(), elementCount);
}

void SemanticsVisitor::validateArraySizeForVariable(VarDeclBase* varDecl)
{
    auto arrayType = as<ArrayExpressionType>(varDecl->type);
    if (!arrayType)
        return;

    if (arrayType->isUnsized())
    {
        // Note(tfoley): For now we allow arrays of unspecified size
        // everywhere, because some source languages (e.g., GLSL)
        // allow them in specific cases.
#if 0
            getSink()->diagnose(varDecl, Diagnostics::invalidArraySize);
#endif
        return;
    }

    // TODO(tfoley): How to handle the case where bound isn't known?
    auto elementCount = arrayType->getElementCount();
    if (GetMinBound(elementCount) <= 0)
    {
        getSink()->diagnose(varDecl, Diagnostics::invalidArraySize);
        return;
    }
}

bool getExtensionTargetDeclList(
    ASTBuilder* astBuilder,
    DeclRefType* targetDeclRefType,
    ExtensionDecl* extDecl,
    ShortList<AggTypeDecl*>& targetDecls)
{
    if (auto aggTypeDeclRef = targetDeclRefType->getDeclRef().as<AggTypeDecl>())
    {
        auto aggTypeDecl = aggTypeDeclRef.getDecl();

        targetDecls.add(aggTypeDecl);
        return true;
    }

    auto genericParamDeclRef = targetDeclRefType->getDeclRef().as<GenericTypeParamDeclBase>();
    if (!genericParamDeclRef)
        return false;

    auto genericParent = as<GenericDecl>(genericParamDeclRef.getParent().getDecl());
    if (!genericParent)
        return false;

    if (genericParent != extDecl->parentDecl)
        return false;

    for (auto member : getMembersOfType<GenericTypeConstraintDecl>(astBuilder, genericParent))
    {
        if (getSub(astBuilder, member) == targetDeclRefType)
        {
            auto baseType = getSup(astBuilder, member);
            if (auto baseTypeDecl = isDeclRefTypeOf<AggTypeDecl>(baseType))
            {
                targetDecls.add(baseTypeDecl.getDecl());
            }
        }
    }
    return targetDecls.getCount() != 0;
}


void SemanticsDeclBasesVisitor::_validateExtensionDeclTargetType(ExtensionDecl* decl)
{
    if (auto targetDeclRefType = as<DeclRefType>(decl->targetType))
    {
        // Attach our extension to that type as a candidate...
        if (targetDeclRefType->getDeclRef().as<InterfaceDecl>())
        {
            getSink()->diagnose(
                decl->targetType.exp,
                Diagnostics::invalidExtensionOnInterface,
                decl->targetType);
            return;
        }
        else if (auto aggTypeDeclRef = targetDeclRefType->getDeclRef().as<AggTypeDecl>())
        {
            auto aggTypeDecl = aggTypeDeclRef.getDecl();

            getShared()->registerCandidateExtension(aggTypeDecl, decl);

            return;
        }
        else if (
            auto genericTypeParamDecl = targetDeclRefType->getDeclRef().as<GenericTypeParamDecl>())
        {
            // If we are extending a generic type parameter as in `extension<T:IFoo> T`,
            // we want to register the extension with the interface type `IFoo` instead.
            auto genericDecl = as<GenericDecl>(genericTypeParamDecl.getDecl()->parentDecl);
            if (!genericDecl)
                goto error;
            if (genericDecl != decl->parentDecl)
                goto error;
            bool isTypeConstrained = false;
            for (auto constraintDecl : genericDecl->getMembersOfType<GenericTypeConstraintDecl>())
            {
                ensureDecl(constraintDecl, DeclCheckState::ReadyForReference);
                if (targetDeclRefType == constraintDecl->sub.type)
                {
                    auto supTypeDeclRef = isDeclRefTypeOf<AggTypeDecl>(constraintDecl->sup.type);
                    getShared()->registerCandidateExtension(supTypeDeclRef.getDecl(), decl);
                    isTypeConstrained = true;
                }
            }
            if (isTypeConstrained)
                return;
        }
    }
error:;
    if (!as<ErrorType>(decl->targetType.type))
    {
        getSink()->diagnose(
            decl->targetType.exp,
            Diagnostics::invalidExtensionOnType,
            decl->targetType);
    }
}

void SemanticsDeclBasesVisitor::_validateExtensionDeclMembers(ExtensionDecl* decl)
{
    for (auto m : decl->members)
    {
        auto ctor = as<ConstructorDecl>(m);
        if (!ctor || !ctor->body || ctor->members.getCount() != 0)
            continue;
        getSink()->diagnose(m->loc, Diagnostics::invalidMemberTypeInExtension, m->astNodeType);
    }
}

void SemanticsDeclBasesVisitor::visitExtensionDecl(ExtensionDecl* decl)
{
    // We check the target type expression and members, and then validate
    // that the type it names is one that it makes sense
    // to extend.
    //
    decl->targetType = CheckProperType(decl->targetType);

    _validateExtensionDeclTargetType(decl);

    _validateExtensionDeclMembers(decl);

    for (auto inheritanceDecl : decl->getMembersOfType<InheritanceDecl>())
    {
        ensureDecl(inheritanceDecl, DeclCheckState::CanUseBaseOfInheritanceDecl);
        auto baseType = inheritanceDecl->base.type;

        // It is possible that there was an error in checking the base type
        // expression, and in such a case we shouldn't emit a cascading error.
        //
        if (const auto baseErrorType = as<ErrorType>(baseType))
        {
            continue;
        }

        // An `extension` can only introduce inheritance from `interface` types.
        //
        // TODO: It might in theory make sense to allow an `extension` to
        // introduce a non-`interface` base if we decide that an `extension`
        // within the same module as the type it extends counts as just
        // a continuation of the type's body (like a `partial class` in C#).
        //
        auto baseDeclRefType = as<DeclRefType>(baseType);
        if (!baseDeclRefType)
        {
            getSink()->diagnose(
                inheritanceDecl,
                Diagnostics::baseOfExtensionMustBeInterface,
                decl,
                baseType);
            continue;
        }

        auto baseDeclRef = baseDeclRefType->getDeclRef();
        auto baseInterfaceDeclRef = baseDeclRef.as<InterfaceDecl>();
        if (!baseInterfaceDeclRef)
        {
            getSink()->diagnose(
                inheritanceDecl,
                Diagnostics::baseOfExtensionMustBeInterface,
                decl,
                baseType);
            continue;
        }

        // TODO: At this point we have the `baseInterfaceDeclRef`
        // and could use it to perform further validity checks,
        // and/or to build up a more refined representation of
        // the inheritance graph for this extension (e.g., a "class
        // precedence list").
        //
        // E.g., we can/should check that we aren't introducing
        // an inheritance relationship that already existed
        // on the type as originally declared.

        _validateCrossModuleInheritance(decl, inheritanceDecl);
    }
}

Type* SemanticsVisitor::calcThisType(DeclRef<Decl> declRef)
{
    if (auto interfaceDeclRef = declRef.as<InterfaceDecl>())
    {
        // In the body of an `interface`, a `This` type
        // refers to the concrete type that will eventually
        // conform to the interface and fill in its
        // requirements.
        //
        return DeclRefType::create(
            m_astBuilder,
            m_astBuilder->getDirectDeclRef(interfaceDeclRef.getDecl()->getThisTypeDecl()));
    }
    else if (auto aggTypeDeclRef = declRef.as<AggTypeDecl>())
    {
        // In the body of an ordinary aggregate type,
        // such as a `struct`, the `This` type just
        // refers to the type itself.
        //
        // TODO: If/when we support `class` types
        // with inheritance, then `This` inside a class
        // would need to refer to the eventual concrete
        // type, much like the `interface` case above.
        //
        return DeclRefType::create(m_astBuilder, aggTypeDeclRef);
    }
    else if (auto genTypeParam = declRef.as<GenericTypeParamDecl>())
    {
        // We will reach here when we are checking `extension<T> T {...}`,
        // where inside the extension, `This` type is the target type
        // of the extension, in this case this is a DeclRefType to
        // a GenericTypeParamDecl.
        //
        return DeclRefType::create(m_astBuilder, declRef);
    }
    else if (auto extDeclRef = declRef.as<ExtensionDecl>())
    {
        // In the body of an `extension`, the `This`
        // type refers to the type being extended.
        //
        // Note: we currently have this loop back
        // around through `calcThisType` for the
        // type being extended, rather than just
        // using it directly. This makes a difference
        // for polymorphic types like `interface`s,
        // and there are reasonable arguments for
        // the validity of either option.
        //
        // Does `extension IFoo` mean extending
        // exactly the type `IFoo` (an existential,
        // which could at runtime be a value of
        // any type conforming to `IFoo`), or does
        // it implicitly extend every type that
        // conforms to `IFoo`? The difference is
        // significant, and we need to make a choice
        // sooner or later.
        //
        ensureDecl(extDeclRef, DeclCheckState::CanUseExtensionTargetType);
        auto targetType = getTargetType(m_astBuilder, extDeclRef);
        return calcThisType(targetType);
    }
    else
    {
        return nullptr;
    }
}

Type* SemanticsVisitor::calcThisType(Type* type)
{
    if (auto declRefType = as<DeclRefType>(type))
    {
        return calcThisType(declRefType->getDeclRef());
    }
    else
    {
        return type;
    }
}

Type* SemanticsVisitor::findResultTypeForConstructorDecl(ConstructorDecl* decl)
{
    // We want to look at the parent of the declaration,
    // but if the declaration is generic, the parent will be
    // the `GenericDecl` and we need to skip past that to
    // the grandparent.
    //
    auto parent = decl->parentDecl;
    auto genericParent = as<GenericDecl>(parent);
    if (genericParent)
    {
        parent = genericParent->parentDecl;
    }

    // The result type for a constructor is whatever `This` would
    // refer to in the body of the outer declaration.
    //
    auto thisType = calcThisType(makeDeclRef(parent));
    if (!thisType)
    {
        getSink()->diagnose(decl, Diagnostics::initializerNotInsideType);
        thisType = m_astBuilder->getErrorType();
    }
    return thisType;
}

void SemanticsDeclHeaderVisitor::visitConstructorDecl(ConstructorDecl* decl)
{
    // We need to compute the result type for this declaration,
    // since it wasn't filled in for us.
    decl->returnType.type = findResultTypeForConstructorDecl(decl);

    checkCallableDeclCommon(decl);
}

void SemanticsDeclHeaderVisitor::visitAbstractStorageDeclCommon(ContainerDecl* decl)
{
    // If we have a subscript or property declaration with no accessor declarations,
    // then we should create a single `GetterDecl` to represent
    // the implicit meaning of their declaration, so:
    //
    //      subscript(uint index) -> T;
    //      property x : Y;
    //
    // becomes:
    //
    //      subscript(uint index) -> T { get; }
    //      property x : Y { get; }
    //

    bool anyAccessors = decl->getMembersOfType<AccessorDecl>().isNonEmpty();

    if (!anyAccessors)
    {
        GetterDecl* getterDecl = m_astBuilder->create<GetterDecl>();
        getterDecl->loc = decl->loc;

        decl->addMember(getterDecl);
    }
}

void SemanticsDeclHeaderVisitor::visitSubscriptDecl(SubscriptDecl* decl)
{
    decl->returnType = CheckUsableType(decl->returnType, decl);

    visitAbstractStorageDeclCommon(decl);

    checkCallableDeclCommon(decl);
}

void SemanticsDeclHeaderVisitor::visitPropertyDecl(PropertyDecl* decl)
{
    SemanticsVisitor subVisitor(withDeclToExcludeFromLookup(decl));
    decl->type = subVisitor.CheckUsableType(decl->type, decl);
    visitAbstractStorageDeclCommon(decl);
    checkVisibility(decl);
}

Type* SemanticsDeclHeaderVisitor::_getAccessorStorageType(AccessorDecl* decl)
{
    auto parentDecl = decl->parentDecl;
    if (auto parentSubscript = as<SubscriptDecl>(parentDecl))
    {
        ensureDecl(parentSubscript, DeclCheckState::CanUseTypeOfValueDecl);
        return parentSubscript->returnType;
    }
    else if (auto parentProperty = as<PropertyDecl>(parentDecl))
    {
        ensureDecl(parentProperty, DeclCheckState::CanUseTypeOfValueDecl);
        return parentProperty->type.type;
    }
    else
    {
        return getASTBuilder()->getErrorType();
    }
}

void SemanticsDeclHeaderVisitor::_visitAccessorDeclCommon(AccessorDecl* decl)
{
    // An accessor must appear nested inside a subscript or property declaration.
    //
    auto parentDecl = decl->parentDecl;
    if (as<SubscriptDecl>(parentDecl))
    {
    }
    else if (as<PropertyDecl>(parentDecl))
    {
    }
    else
    {
        getSink()->diagnose(decl, Diagnostics::accessorMustBeInsideSubscriptOrProperty);
    }
}

void SemanticsDeclHeaderVisitor::visitAccessorDecl(AccessorDecl* decl)
{
    _visitAccessorDeclCommon(decl);

    // Note: This subroutine is used by both `get`
    // and `ref` accessors, but is bypassed by
    // `set` accessors (which use `visitSetterDecl`
    // intead).

    // Accessors (other than setters) don't support
    // parameters.
    //
    if (decl->getParameters().getCount() != 0)
    {
        getSink()->diagnose(decl, Diagnostics::nonSetAccessorMustNotHaveParams);
    }

    // By default, the return type of an accessor is treated as
    // the type of the abstract storage location being accessed.
    //
    // A `ref`  accessor currently relies on this logic even though
    // it isn't quite correct, because we don't have support
    // for by-reference return values today. This is a non-issue
    // for now because we don't support user-defined `ref`
    // accessors yet.
    //
    // TODO: Once we can support the by-reference return value
    // correctly *or* we can move to something like a coroutine-based
    // `modify` accessor (a la Swift), we should split out
    // handling of `RefAccessorDecl` and only use this routine
    // for `GetterDecl`s.
    //
    decl->returnType.type = _getAccessorStorageType(decl);

    checkDifferentiableCallableCommon(decl);
}

void SemanticsDeclHeaderVisitor::visitSetterDecl(SetterDecl* decl)
{
    // Make sure to invoke the common checking logic for all accessors.
    _visitAccessorDeclCommon(decl);

    // A `set` accessor always returns `void`.
    //
    decl->returnType.type = getASTBuilder()->getVoidType();

    // A setter always receives a single value representing
    // the new value to set into the storage.
    //
    // The user may declare that parameter explicitly and
    // thereby control its name, or they can declare no
    // parmaeters and allow the compiler to synthesize one
    // names `newValue`.
    //
    ParamDecl* newValueParam = nullptr;
    auto params = decl->getParameters();
    if (params.getCount() >= 1)
    {
        // If the user declared an explicit parameter
        // then that is the one that will represent
        // the new value.
        //
        newValueParam = params.getFirst();

        if (params.getCount() > 1)
        {
            // If the user declared more than one explicit
            // parameter, then that is an error.
            //
            getSink()->diagnose(params[1], Diagnostics::setAccessorMayNotHaveMoreThanOneParam);
        }
    }
    else
    {
        // If the user didn't declare any explicit parameters,
        // then we create an implicit one and add it into
        // the AST.
        //
        newValueParam = m_astBuilder->create<ParamDecl>();
        newValueParam->nameAndLoc.name = getName("newValue");
        newValueParam->nameAndLoc.loc = decl->loc;

        decl->addMember(newValueParam);
    }

    // The new-value parameter is expected to have the
    // same type as the abstract storage that the
    // accessor is setting.
    //
    auto newValueType = _getAccessorStorageType(decl);

    // It is allowed and encouraged for the programmer
    // to leave off the type on the new-value parameter,
    // in which case we will set it to the expected
    // type automatically.
    //
    if (!newValueParam->type.exp)
    {
        newValueParam->type.type = newValueType;
    }
    else
    {
        // If the user *did* give the new-value parameter
        // an explicit type, then we need to check it
        // and then enforce that it matches what we expect.
        //
        auto actualType = CheckProperType(newValueParam->type);

        if (as<ErrorType>(actualType))
        {
        }
        else if (actualType->equals(newValueType))
        {
        }
        else
        {
            getSink()->diagnose(
                newValueParam,
                Diagnostics::setAccessorParamWrongType,
                newValueParam,
                actualType,
                newValueType);
        }
    }
    checkDifferentiableCallableCommon(decl);
}

GenericDecl* SemanticsVisitor::GetOuterGeneric(Decl* decl)
{
    auto parentDecl = decl->parentDecl;
    if (!parentDecl)
        return nullptr;
    auto parentGeneric = as<GenericDecl>(parentDecl);
    return parentGeneric;
}

Decl* SemanticsVisitor::getOuterGenericOrSelf(Decl* decl)
{
    auto parentDecl = decl->parentDecl;
    if (!parentDecl)
        return decl;
    auto parentGeneric = as<GenericDecl>(parentDecl);
    if (!parentGeneric)
        return decl;
    return parentGeneric;
}

GenericDecl* SemanticsVisitor::findNextOuterGeneric(Decl* decl)
{
    for (auto p = decl->parentDecl; p; p = p->parentDecl)
    {
        if (auto genDecl = as<GenericDecl>(p))
            return genDecl;
    }
    return nullptr;
}

DeclRef<ExtensionDecl> SemanticsVisitor::applyExtensionToType(
    ExtensionDecl* extDecl,
    Type* type,
    Dictionary<Type*, SubtypeWitness*>* additionalSubtypeWitnessesForType)
{
    DeclRef<ExtensionDecl> extDeclRef = makeDeclRef(extDecl);

    // If the extension is a generic extension, then we
    // need to infer type arguments that will give
    // us a target type that matches `type`.
    //
    if (auto extGenericDecl = GetOuterGeneric(extDecl))
    {
        ConstraintSystem constraints;
        constraints.loc = extDecl->loc;
        constraints.genericDecl = extGenericDecl;
        if (additionalSubtypeWitnessesForType)
        {
            constraints.subTypeForAdditionalWitnesses = type;
            constraints.additionalSubtypeWitnesses = additionalSubtypeWitnessesForType;
        }

        // Inside the body of an extension declaration, we may end up trying to apply that
        // extension to its own target type.
        // If we see that we are in that case, we can apply the extension declaration as - is,
        // without any additional substitutions.
        if (extDecl->targetType->equals(type))
        {
            return createDefaultSubstitutionsIfNeeded(m_astBuilder, this, extDeclRef)
                .as<ExtensionDecl>();
        }

        if (!TryUnifyTypes(constraints, ValUnificationContext(), extDecl->targetType.Ptr(), type))
            return DeclRef<ExtensionDecl>();

        ConversionCost baseCost;
        auto solvedDeclRef = trySolveConstraintSystem(
            &constraints,
            makeDeclRef(extGenericDecl),
            ArrayView<Val*>(),
            baseCost);
        if (!solvedDeclRef)
        {
            return DeclRef<ExtensionDecl>();
        }

        // Construct a reference to the extension with our constraint variables
        // set as they were found by solving the constraint system.
        extDeclRef = solvedDeclRef.as<ExtensionDecl>();
    }

    // Now extract the target type from our (possibly specialized) extension decl-ref.
    Type* targetType = getTargetType(m_astBuilder, extDeclRef);

    // As a bit of a kludge here, if the target type of the extension is
    // an interface, and the `type` we are trying to match up has a this-type
    // substitution for that interface, then we want to attach a matching
    // substitution to the extension decl-ref.
    if (auto targetDeclRefType = as<DeclRefType>(targetType))
    {
        if (auto targetInterfaceDeclRef = targetDeclRefType->getDeclRef().as<InterfaceDecl>())
        {
            // Okay, the target type is an interface.
            //
            // Is the type we want to apply to a ThisType?
            if (auto appDeclRefType = as<ThisType>(type))
            {
                if (auto thisTypeLookupDeclRef =
                        SubstitutionSet(appDeclRefType->getDeclRef()).findLookupDeclRef())
                {
                    if (thisTypeLookupDeclRef->getDecl() == targetInterfaceDeclRef.getDecl())
                    {
                        // Looks like we have a match in the types,
                        // now let's see if `type`'s declref starts with a Lookup.
                        targetType = type;
                        extDeclRef = m_astBuilder
                                         ->getLookupDeclRef(
                                             thisTypeLookupDeclRef->getWitness(),
                                             extDeclRef.getDecl())
                                         .as<ExtensionDecl>();
                    }
                }
            }
        }
    }

    // In order for this extension to apply to the given type, we
    // need to have a match on the target types.
    if (!type->equals(targetType))
        return DeclRef<ExtensionDecl>();


    return extDeclRef;
}

QualType SemanticsVisitor::GetTypeForDeclRef(DeclRef<Decl> declRef, SourceLoc loc)
{
    Type* typeResult = nullptr;
    return getTypeForDeclRef(m_astBuilder, this, getSink(), declRef, &typeResult, loc);
}

void SemanticsVisitor::importFileDeclIntoScope(Scope* scope, FileDecl* fileDecl)
{
    // Create a new sub-scope to wire the module
    // into our lookup chain.
    if (!fileDecl)
        return;
    addSiblingScopeForContainerDecl(getASTBuilder(), scope, fileDecl);
}

void SemanticsVisitor::importModuleIntoScope(Scope* scope, ModuleDecl* moduleDecl)
{
    if (!moduleDecl)
        return;

    // If we've imported this one already, then
    // skip the step where we modify the current scope.
    auto& importedModulesList = getShared()->importedModulesList;
    auto& importedModulesSet = getShared()->importedModulesSet;
    if (importedModulesSet.contains(moduleDecl))
    {
        return;
    }

    if (getText(moduleDecl->getName()) == "glsl")
    {
        getShared()->glslModuleDecl = moduleDecl;
    }

    importedModulesList.add(moduleDecl);
    importedModulesSet.add(moduleDecl);

    // Create a new sub-scope to wire the module's scope and its nested FileDecl's scopes
    // into our lookup chain.
    for (auto moduleScope = moduleDecl->ownedScope; moduleScope;
         moduleScope = moduleScope->nextSibling)
    {
        if (moduleScope->containerDecl != moduleDecl &&
            moduleScope->containerDecl->parentDecl != moduleDecl)
            continue;

        addSiblingScopeForContainerDecl(getASTBuilder(), scope, moduleScope->containerDecl);
    }

    // Also import any modules from nested `import` declarations
    // with the `__exported` modifier
    for (auto importDecl : moduleDecl->getMembersOfType<ImportDecl>())
    {
        if (!importDecl->hasModifier<ExportedModifier>())
            continue;

        importModuleIntoScope(scope, importDecl->importedModuleDecl);
    }
}

void SemanticsDeclHeaderVisitor::visitImportDecl(ImportDecl* decl)
{
    // We need to look for a module with the specified name
    // (whether it has already been loaded, or needs to
    // be loaded), and then put its declarations into
    // the module's scope.

    auto name = decl->moduleNameAndLoc.name;
    auto scope = getModuleDecl(decl)->ownedScope;

    // Try to load a module matching the name
    auto importedModule = findOrImportModule(
        getLinkage(),
        name,
        decl->moduleNameAndLoc.loc,
        getSink(),
        getShared()->m_environmentModules);

    // If we didn't find a matching module, then bail out
    if (!importedModule)
        return;

    // Record the module that was imported, so that we can use
    // it later during code generation.
    auto importedModuleDecl = importedModule->getModuleDecl();
    decl->importedModuleDecl = importedModuleDecl;

    // Add the declarations from the imported module into the scope
    // that the `import` declaration is set to extend.
    //
    importModuleIntoScope(scope, importedModuleDecl);

    // Record the `import`ed module (and everything it depends on)
    // as a dependency of the module we are compiling.
    if (auto module = getModule(decl))
    {
        module->addModuleDependency(importedModule);
    }
}

String getSimpleModuleName(Name* name)
{
    auto text = getText(name);
    auto dirPos = Math::Max(text.indexOf('/'), text.indexOf('\\'));
    if (dirPos < 0)
        return text;
    auto slice = text.getUnownedSlice().tail(dirPos + 1);
    auto dotPos = slice.indexOf('.');
    if (dotPos < 0)
        return slice;
    return String(slice.head(dotPos));
}

ModuleDeclarationDecl* findExistingModuleDeclarationDecl(ModuleDecl* decl)
{
    if (decl->members.getCount() == 0)
        return nullptr;
    if (auto rs = as<ModuleDeclarationDecl>(decl->members[0]))
        return rs;
    for (auto fileDecl : decl->getMembersOfType<FileDecl>())
    {
        if (fileDecl->members.getCount() == 0)
            continue;
        if (auto rs = as<ModuleDeclarationDecl>(fileDecl->members[0]))
            return rs;
    }
    return nullptr;
}

void SemanticsDeclHeaderVisitor::visitIncludeDecl(IncludeDecl* decl)
{
    auto name = decl->moduleNameAndLoc.name;

    if (!getShared()->getTranslationUnitRequest())
        getSink()->diagnose(decl->moduleNameAndLoc.loc, Diagnostics::cannotProcessInclude);

    auto parentModule = getModule(decl);
    auto moduleDecl = parentModule->getModuleDecl();

    auto [fileDecl, isNew] = getLinkage()->findAndIncludeFile(
        getModule(decl),
        getShared()->getTranslationUnitRequest(),
        name,
        decl->moduleNameAndLoc.loc,
        getSink());

    if (!fileDecl)
        return;

    decl->fileDecl = fileDecl;

    if (!isNew)
        return;

    if (fileDecl->members.getCount() == 0)
        return;

    auto firstMember = fileDecl->members[0];
    if (auto moduleDeclaration = as<ModuleDeclarationDecl>(firstMember))
    {
        // We are trying to include a file that defines a module, the user could mean "import"
        // instead.
        getSink()->diagnose(
            decl->moduleNameAndLoc.loc,
            Diagnostics::includedFileMissingImplementingDoYouMeanImport,
            name,
            moduleDeclaration->getName());
        return;
    }

    importFileDeclIntoScope(moduleDecl->ownedScope, fileDecl);

    if (auto implementing = as<ImplementingDecl>(firstMember))
    {
        // The file we are including must be implementing the current module.
        auto moduleName = getSimpleModuleName(implementing->moduleNameAndLoc.name);
        auto expectedModuleName = moduleDecl->getName();
        bool shouldSkipDiagnostic = false;

        if (moduleDecl->members.getCount())
        {
            if (auto moduleDeclarationDecl = as<ModuleDeclarationDecl>(moduleDecl->members[0]))
            {
                expectedModuleName = moduleDeclarationDecl->getName();
            }
            else if (getShared()->isInLanguageServer())
            {
                auto moduleDeclarationDecls = findExistingModuleDeclarationDecl(moduleDecl);
                if (moduleDeclarationDecls)
                {
                    expectedModuleName = moduleDeclarationDecls->getName();
                }
                else
                {
                    shouldSkipDiagnostic = true;
                }
            }
        }

        if (!shouldSkipDiagnostic)
        {
            // First check for the case when the user has put a file extension
            // in the include path
            String moduleNameStr = moduleName.getUnownedSlice();
            String expectedModuleNameStr = getText(expectedModuleName).getUnownedSlice();

            // Check if module name has a source file extension
            if (moduleNameStr.endsWith(".slang"))
            {
                String normalizedName = moduleNameStr.subString(0, moduleNameStr.getLength() - 6);

                // If the normalized name would match, emit warning but continue
                if (normalizedName.getUnownedSlice().caseInsensitiveEquals(
                        expectedModuleNameStr.getUnownedSlice()))
                {
                    getSink()->diagnose(
                        implementing->moduleNameAndLoc.loc,
                        Diagnostics::moduleImplementationHasFileExtension,
                        moduleNameStr,
                        normalizedName);
                    return;
                }
            }

            if (!moduleNameStr.getUnownedSlice().caseInsensitiveEquals(
                    expectedModuleNameStr.getUnownedSlice()))
            {
                getSink()->diagnose(
                    decl->moduleNameAndLoc.loc,
                    Diagnostics::includedFileDoesNotImplementCurrentModule,
                    expectedModuleName,
                    moduleName);
                return;
            }
        }

        return;
    }

    getSink()->diagnose(
        decl->moduleNameAndLoc.loc,
        Diagnostics::includedFileMissingImplementing,
        name);
}


void SemanticsDeclScopeWiringVisitor::visitImplementingDecl(ImplementingDecl* decl)
{
    // Don't need to do anything unless we are in a language server context.
    if (!getShared()->isInLanguageServer())
        return;

    // Treat an `implementing` declaration as an `include` declaration when
    // we are in a language server context.

    auto name = decl->moduleNameAndLoc.name;

    if (!getShared()->getTranslationUnitRequest())
        getSink()->diagnose(decl->moduleNameAndLoc.loc, Diagnostics::cannotProcessInclude);

    auto [fileDecl, isNew] = getLinkage()->findAndIncludeFile(
        getModule(decl),
        getShared()->getTranslationUnitRequest(),
        name,
        decl->moduleNameAndLoc.loc,
        getSink());

    decl->fileDecl = fileDecl;

    if (!isNew)
        return;

    if (!fileDecl || fileDecl->members.getCount() == 0)
    {
        return;
    }

    auto firstMember = fileDecl->members[0];
    if (as<ModuleDeclarationDecl>(firstMember))
    {
        // We are trying to implement a file that defines a module, this is expected.
    }
    else if (as<ImplementingDecl>(firstMember))
    {
        getSink()->diagnose(
            decl->moduleNameAndLoc.loc,
            Diagnostics::implementingMustReferencePrimaryModuleFile);
        return;
    }

    if (auto moduleDecl = getModuleDecl(decl))
        importFileDeclIntoScope(moduleDecl->ownedScope, fileDecl);
}

void SemanticsDeclScopeWiringVisitor::visitUsingDecl(UsingDecl* decl)
{
    // First, we need to look up whatever the argument of the `using`
    // declaration names.
    //
    decl->arg = CheckTerm(decl->arg);

    // Next, we want to ensure that whatever is being named by `decl->arg`
    // is a namespace (or a module, since modules are namespace-like).
    //
    // If a user `import`s multiple modules that all have namespaces
    // of the same name, it would be possible for `decl->arg` to be overloaded.
    // To handle that case, we will iterate over all the entities that are
    // named and import any that are namespace-like.
    //
    bool scopesAdded = false;
    bool hasValidNamespace = false;

    // TODO: consider caching the scope set in NamespaceDecl.
    HashSet<ContainerDecl*> addedScopes;
    for (auto s = decl->scope; s; s = s->nextSibling)
        addedScopes.add(s->containerDecl);

    auto addAllSiblingScopesFromDecl = [&](Scope* scope, ContainerDecl* containerDecl)
    {
        for (auto s = containerDecl->ownedScope; s; s = s->nextSibling)
        {
            if (addedScopes.add(s->containerDecl))
            {
                scopesAdded = true;
                addSiblingScopeForContainerDecl(getASTBuilder(), scope, s->containerDecl);
            }
        }
    };

    if (auto declRefExpr = as<DeclRefExpr>(decl->arg))
    {
        if (auto namespaceDeclRef = declRefExpr->declRef.as<NamespaceDeclBase>())
        {
            auto namespaceDecl = namespaceDeclRef.getDecl();
            addAllSiblingScopesFromDecl(decl->scope, namespaceDecl);
            hasValidNamespace = true;
        }
    }
    else if (auto overloadedExpr = as<OverloadedExpr>(decl->arg))
    {
        for (auto item : overloadedExpr->lookupResult2)
        {
            if (auto namespaceDeclRef = item.declRef.as<NamespaceDeclBase>())
            {
                addAllSiblingScopesFromDecl(decl->scope, namespaceDeclRef.getDecl());
                hasValidNamespace = true;
            }
        }
    }

    if (!scopesAdded)
    {
        if (!hasValidNamespace)
            getSink()->diagnose(decl->arg, Diagnostics::expectedANamespace, decl->arg->type);
        return;
    }
}

void SemanticsDeclScopeWiringVisitor::visitNamespaceDecl(NamespaceDecl* decl)
{
    // We need to wire up the scope of namespaces with other namespace decls of the same name
    // that is accessible from the current context.
    auto parent = as<ContainerDecl>(getParentDecl(decl));
    if (!parent)
        return;
    for (auto parentScope = parent->ownedScope; parentScope; parentScope = parentScope->parent)
    {
        for (auto scope = parentScope; scope; scope = scope->nextSibling)
        {
            auto container = scope->containerDecl;
            auto nsDeclPtr = container->getMemberDictionary().tryGetValue(decl->getName());
            if (!nsDeclPtr)
                continue;
            auto nsDecl = *nsDeclPtr;
            for (auto ns = nsDecl; ns; ns = ns->nextInContainerWithSameName)
            {
                if (ns == decl)
                    continue;
                auto otherNamespace = as<NamespaceDeclBase>(ns);
                if (!otherNamespace)
                    continue;

                if (!ns->checkState.isBeingChecked())
                {
                    ensureDecl(ns, DeclCheckState::ScopesWired);
                }
                addSiblingScopeForContainerDecl(getASTBuilder(), decl, otherNamespace);
            }
        }
        // For file decls, we need to continue searching up in the parent module scope.
        if (!as<FileDecl>(parentScope->containerDecl))
            break;
    }
    for (auto usingDecl : decl->getMembersOfType<UsingDecl>())
    {
        ensureDecl(usingDecl, DeclCheckState::ScopesWired);
    }
}

/// Get a reference to the candidate extension list for `typeDecl` in the given dictionary
///
/// Note: this function creates an empty list of candidates for the given type if
/// a matching entry doesn't exist already.
///
static List<ExtensionDecl*>& _getCandidateExtensionList(
    AggTypeDecl* typeDecl,
    Dictionary<AggTypeDecl*, RefPtr<CandidateExtensionList>>& mapTypeToCandidateExtensions)
{
    RefPtr<CandidateExtensionList> entry;
    if (!mapTypeToCandidateExtensions.tryGetValue(typeDecl, entry))
    {
        entry = new CandidateExtensionList();
        mapTypeToCandidateExtensions.add(typeDecl, entry);
    }
    return entry->candidateExtensions;
}

List<ExtensionDecl*> const& SharedSemanticsContext::getCandidateExtensionsForTypeDecl(
    AggTypeDecl* decl)
{
    // We are caching the lists of candidate extensions on the shared
    // context, so we will only build the lists if they either have
    // not been built before, or if some code caused the lists to
    // be invalidated.
    //
    // TODO: Similar to the rebuilding of lookup tables in `ContainerDecl`s,
    // we probably want to optimize this logic to gracefully handle new
    // extensions encountered during checking instead of tearing the whole
    // thing down. For now this potentially-quadratic behavior is acceptable
    // because there just aren't that many extension declarations being used.
    //
    if (!m_candidateExtensionListsBuilt)
    {
        m_candidateExtensionListsBuilt = true;

        // We need to make sure that all extensions that were declared
        // as parts of our core module are always visible,
        // even if they are not explicit `import`ed into user code.
        //
        for (auto module : getSession()->coreModules)
        {
            _addCandidateExtensionsFromModule(module->getModuleDecl());
        }

        // There are two primary modes in which the `SharedSemanticsContext`
        // gets used.
        //
        // In the first mode, we are checking an entire `ModuelDecl`, and we
        // need to always check things from the "point of view" of that module
        // (so that the extensions that should be visible are based on what
        // that module can access via `import`s).
        //
        // In the second mode, we are checking code related to API interactions
        // by the user (e.g., parsing a type from a string, specializing an
        // entry point to type arguments, etc.). In these cases there is no
        // clear module that should determine the point of view for looking
        // up extensions, and we instead need/want to consider any extensions
        // from all modules loaded into the linkage.
        //
        // We differentiate these cases based on whether a "primary" module
        // was set at the time the `SharedSemanticsContext` was constructed.
        //
        if (m_module)
        {
            // We have a "primary" module that is being checked, and we should
            // look up extensions based on what would be visible to that
            // module.
            //
            // Extensions declared in the module itself should have already
            // been registered when we check them, but we still need to bring
            // along with everything the module imported.
            //
            // Note: there is an implicit assumption here that the `importedModules`
            // member on the `SharedSemanticsContext` is accurate in this case.
            //
            for (auto moduleDecl : this->importedModulesList)
            {
                _addCandidateExtensionsFromModule(moduleDecl);
            }
        }
        else
        {
            // We are in one of the many ad hoc checking modes where we really
            // want to resolve things based on the totality of what is
            // available/defined within the current linkage.
            //
            for (auto module : m_linkage->loadedModulesList)
            {
                _addCandidateExtensionsFromModule(module->getModuleDecl());
            }
        }
    }

    // Once we are sure that the dictionary-of-arrays of extensions
    // has been populated, we return to the user the entry they
    // asked for.
    //
    return _getCandidateExtensionList(decl, m_mapTypeDeclToCandidateExtensions);
}

void SharedSemanticsContext::registerCandidateExtension(
    AggTypeDecl* typeDecl,
    ExtensionDecl* extDecl)
{
    // The primary cache of extension declarations is on the `ModuleDecl`.
    // We will add the `extDecl` to the cache for the module it belongs to.
    //
    // We can be sure that the resulting cache won't have lifetime issues,
    // because all the extensions it contains are owned by the module itself,
    // and the types used as keys had to be reachable/referenceable from the
    // code inside the module for the given `extDecl` to extend them.
    //
    auto moduleDecl = getModuleDecl(extDecl);
    _getCandidateExtensionList(typeDecl, moduleDecl->mapTypeToCandidateExtensions).add(extDecl);

    // Because we've loaded a new extension, we need to invalidate whatever
    // information the `SharedSemanticsContext` had cached about loaded
    // extensions, and force it to rebuild its cache to include the
    // new extension we just added.
    //
    _getCandidateExtensionList(typeDecl, m_mapTypeDeclToCandidateExtensions).add(extDecl);

    // Remove the cached inheritanceInfo about typeDecl, if `extDecl` inherits new types.
    bool invalidateSubtypes = false;
    if (as<InterfaceDecl>(typeDecl))
    {
        // If we are extending an interface, we are effectively extending all types
        // that inherits the interface. So we need to remove all inheritance info
        // that is related to the interface.
        invalidateSubtypes = true;
    }
    bool hasInheritanceMember = false;
    bool hasImplicitCastMember = false;
    for (auto member : extDecl->members)
    {
        if (as<InheritanceDecl>(member))
        {
            hasInheritanceMember = true;
        }
        else if (auto ctorDecl = as<ConstructorDecl>(member))
        {
            if (ctorDecl->hasModifier<ImplicitConversionModifier>())
                hasImplicitCastMember = true;
        }
    }
    auto isTypeUpToDate = [this](Type* type)
    {
        if (auto declRefType = as<DeclRefType>(type))
        {
            return m_mapDeclRefToInheritanceInfo.containsKey(declRefType->getDeclRef());
        }
        return m_mapTypeToInheritanceInfo.containsKey(type);
    };
    auto isInheritanceInfoAffected = [typeDecl](InheritanceInfo& info)
    {
        for (auto f : info.facets)
            if (f.getImpl()->getDeclRef().getDecl() == typeDecl)
            {
                return true;
            }
        return false;
    };
    if (invalidateSubtypes)
    {
        decltype(m_mapTypeToInheritanceInfo) newMapTypeToInheritanceInfo;
        for (auto& kv : m_mapTypeToInheritanceInfo)
        {
            if (!isInheritanceInfoAffected(kv.second))
            {
                newMapTypeToInheritanceInfo.add(kv.first, kv.second);
            }
        }
        m_mapTypeToInheritanceInfo = _Move(newMapTypeToInheritanceInfo);
    }

    ShortList<DeclRef<Decl>, 16> keysToRemove;
    for (auto& kv : m_mapDeclRefToInheritanceInfo)
    {
        // We can confirm the type is affected by the new extension,
        // if the declref type points to typeDecl.
        if (kv.first.getDecl() == typeDecl)
        {
            keysToRemove.add(kv.first);
            continue;
        }

        // If we are extending interface types (and in the future any struct type
        // if we decide to have full inheritance support),
        // we also need to account for conformant that implements the interface.
        if (invalidateSubtypes && isInheritanceInfoAffected(kv.second))
        {
            keysToRemove.add(kv.first);
        }
    }
    for (auto& key : keysToRemove)
    {
        m_mapDeclRefToInheritanceInfo.remove(key);
    }

    if (hasInheritanceMember || invalidateSubtypes)
    {
        ShortList<TypePair, 16> typePairsToRemove;
        for (auto& kv : m_mapTypePairToSubtypeWitness)
        {
            if (!isTypeUpToDate(kv.first.type0) || !isTypeUpToDate(kv.first.type1))
            {
                typePairsToRemove.add(kv.first);
            }
        }
        for (auto& key : typePairsToRemove)
        {
            m_mapTypePairToSubtypeWitness.remove(key);
        }
    }

    if (hasImplicitCastMember)
    {
        ShortList<ImplicitCastMethodKey> entriesToRemove;
        for (auto& kv : m_mapTypePairToImplicitCastMethod)
        {
            // Since implicit casts are defined as constructors on the toType,
            // we only need to check if the toType is affected by the new extension.
            auto declRefType = as<DeclRefType>(kv.first.toType);

            if (!declRefType || declRefType->getDeclRef().getDecl() == typeDecl)
            {
                entriesToRemove.add(kv.first);
            }
        }
        for (auto& key : entriesToRemove)
        {
            m_mapTypePairToImplicitCastMethod.remove(key);
        }
    }
}

void SharedSemanticsContext::_addCandidateExtensionsFromModule(ModuleDecl* moduleDecl)
{
    for (auto& [entryKey, entryValue] : moduleDecl->mapTypeToCandidateExtensions)
    {
        auto& list = _getCandidateExtensionList(entryKey, m_mapTypeDeclToCandidateExtensions);
        list.addRange(entryValue->candidateExtensions);
    }
}

/// Get a reference to the associated decl list for `decl` in the given dictionary
///
/// Note: this function creates an empty list of candidates for the given type if
/// a matching entry doesn't exist already.
///
static List<RefPtr<DeclAssociation>>& _getDeclAssociationList(
    Decl* decl,
    OrderedDictionary<Decl*, RefPtr<DeclAssociationList>>& mapDeclToDeclarations)
{
    RefPtr<DeclAssociationList> entry;
    if (!mapDeclToDeclarations.tryGetValue(decl, entry))
    {
        entry = new DeclAssociationList();
        mapDeclToDeclarations.add(decl, entry);
    }
    return entry->associations;
}

void SharedSemanticsContext::_addDeclAssociationsFromModule(ModuleDecl* moduleDecl)
{
    for (auto& entry : moduleDecl->mapDeclToAssociatedDecls)
    {
        auto& list = _getDeclAssociationList(entry.key, m_mapDeclToAssociatedDecls);
        list.addRange(entry.value->associations);
    }
}

void SharedSemanticsContext::registerAssociatedDecl(
    Decl* original,
    DeclAssociationKind kind,
    Decl* associated)
{
    auto moduleDecl = getModuleDecl(associated);
    RefPtr<DeclAssociation> assoc = new DeclAssociation();
    assoc->kind = kind;
    assoc->decl = associated;
    _getDeclAssociationList(original, moduleDecl->mapDeclToAssociatedDecls).add(assoc);

    m_associatedDeclListsBuilt = false;
    m_mapDeclToAssociatedDecls.clear();
}

List<RefPtr<DeclAssociation>> const& SharedSemanticsContext::getAssociatedDeclsForDecl(Decl* decl)
{
    // This duplicates the exact same logic from `getCandidateExtensionsForTypeDecl`.
    // Consider refactoring them into the same framework.
    if (!m_associatedDeclListsBuilt)
    {
        m_associatedDeclListsBuilt = true;

        for (auto module : getSession()->coreModules)
        {
            _addDeclAssociationsFromModule(module->getModuleDecl());
        }

        if (m_module)
        {
            _addDeclAssociationsFromModule(m_module->getModuleDecl());
            for (auto moduleDecl : this->importedModulesList)
            {
                _addDeclAssociationsFromModule(moduleDecl);
            }
        }
        else
        {
            for (auto module : m_linkage->loadedModulesList)
            {
                _addDeclAssociationsFromModule(module->getModuleDecl());
            }
        }
    }
    return _getDeclAssociationList(decl, m_mapDeclToAssociatedDecls);
}

bool SharedSemanticsContext::isDifferentiableFunc(FunctionDeclBase* func)
{
    return getFuncDifferentiableLevel(func) != FunctionDifferentiableLevel::None;
}

bool SharedSemanticsContext::isBackwardDifferentiableFunc(FunctionDeclBase* func)
{
    return getFuncDifferentiableLevel(func) == FunctionDifferentiableLevel::Backward;
}

FunctionDifferentiableLevel SharedSemanticsContext::getFuncDifferentiableLevel(
    FunctionDeclBase* func)
{
    return _getFuncDifferentiableLevelImpl(func, 1);
}

FunctionDifferentiableLevel SharedSemanticsContext::_getFuncDifferentiableLevelImpl(
    FunctionDeclBase* func,
    int recurseLimit)
{
    if (!func)
        return FunctionDifferentiableLevel::None;

    if (recurseLimit > 0)
    {
        if (auto primalSubst = func->findModifier<PrimalSubstituteAttribute>())
        {
            if (auto declRefExpr = as<DeclRefExpr>(primalSubst->funcExpr))
            {
                if (auto primalSubstFunc = declRefExpr->declRef.as<FunctionDeclBase>())
                    return _getFuncDifferentiableLevelImpl(
                        primalSubstFunc.getDecl(),
                        recurseLimit - 1);
            }
        }
    }

    if (func->findModifier<BackwardDifferentiableAttribute>())
        return FunctionDifferentiableLevel::Backward;
    if (func->findModifier<BackwardDerivativeAttribute>())
        return FunctionDifferentiableLevel::Backward;

    if (func->findModifier<TreatAsDifferentiableAttribute>())
        return FunctionDifferentiableLevel::Backward;

    FunctionDifferentiableLevel diffLevel = FunctionDifferentiableLevel::None;
    if (func->findModifier<DifferentiableAttribute>())
        diffLevel = FunctionDifferentiableLevel::Forward;

    for (auto assocDecl : getAssociatedDeclsForDecl(func))
    {
        switch (assocDecl->kind)
        {
        case DeclAssociationKind::BackwardDerivativeFunc:
            return FunctionDifferentiableLevel::Backward;
        case DeclAssociationKind::ForwardDerivativeFunc:
            diffLevel = FunctionDifferentiableLevel::Forward;
            break;
        case DeclAssociationKind::PrimalSubstituteFunc:
            if (auto assocFunc = as<FunctionDeclBase>(assocDecl->decl))
            {
                return _getFuncDifferentiableLevelImpl(assocFunc, recurseLimit - 1);
            }
            break;
        default:
            break;
        }
    }
    if (auto builtinReq = func->findModifier<BuiltinRequirementModifier>())
    {
        switch (builtinReq->kind)
        {
        case BuiltinRequirementKind::DAddFunc:
        case BuiltinRequirementKind::DMulFunc:
        case BuiltinRequirementKind::DZeroFunc:
            return FunctionDifferentiableLevel::Backward;
        default:
            break;
        }
    }
    return diffLevel;
}

List<ExtensionDecl*> const& getCandidateExtensions(
    DeclRef<AggTypeDecl> const& declRef,
    SemanticsVisitor* semantics)
{
    auto decl = declRef.getDecl();
    auto shared = semantics->getShared();
    return shared->getCandidateExtensionsForTypeDecl(decl);
}

void _foreachDirectOrExtensionMemberOfType(
    SemanticsVisitor* semantics,
    DeclRef<ContainerDecl> const& containerDeclRef,
    SyntaxClassBase const& syntaxClass,
    void (*callback)(DeclRefBase*, void*),
    void const* userData)
{
    // We are being asked to invoke the given callback on
    // each direct member of `containerDeclRef`, along with
    // any members added via `extension` declarations, that
    // have the correct AST node class (`syntaxClass`).
    //
    // We start with the direct members.
    //
    for (auto memberDeclRef : getMembers(semantics->getASTBuilder(), containerDeclRef))
    {
        if (memberDeclRef.getDecl()->getClass().isSubClassOf(syntaxClass))
        {
            callback(memberDeclRef, (void*)userData);
        }
    }

    // Next, in the case wher ethe type can be subject to extensions,
    // we loop over the applicable extensions and their member.s
    //
    if (auto aggTypeDeclRef = containerDeclRef.as<AggTypeDecl>())
    {
        auto aggType = DeclRefType::create(semantics->getASTBuilder(), aggTypeDeclRef);
        auto candidateExtensions = getCandidateExtensions(aggTypeDeclRef, semantics);
        for (auto extDecl : candidateExtensions)
        {
            // Note that `extDecl` may have been declared for a type
            // base on the declaration that `aggTypeDeclRef` refers
            // to, but that does not guarantee that it applies to
            // the type itself. E.g., we might have an extension of
            // `vector<float, N>` for any `N`, but the current type is
            // `vector<int, 2>` so that the extension doesn't match.
            //
            // In order to make sure that we don't enumerate members
            // that don't make sense in context, we must apply
            // the extension to the type and see if we succeed in
            // making a match.
            //
            auto extDeclRef = applyExtensionToType(semantics, extDecl, aggType);
            if (!extDeclRef)
                continue;

            for (auto memberDeclRef : getMembers(semantics->getASTBuilder(), extDeclRef))
            {
                if (memberDeclRef.getDecl()->getClass().isSubClassOf(syntaxClass))
                {
                    callback(memberDeclRef, (void*)userData);
                }
            }
        }
    }
}

static void _dispatchDeclCheckingVisitor(Decl* decl, DeclCheckState state, SemanticsContext& shared)
{
    switch (state)
    {
    case DeclCheckState::ReadyForParserLookup:
        // We don't need to do anything to make a decl ready for parser lookup.
        break;
    case DeclCheckState::ModifiersChecked:
        SemanticsDeclModifiersVisitor(shared).dispatch(decl);
        break;
    case DeclCheckState::ScopesWired:
        SemanticsDeclScopeWiringVisitor(shared).dispatch(decl);
        break;

    case DeclCheckState::SignatureChecked:
        SemanticsDeclHeaderVisitor(shared).dispatch(decl);
        break;

    case DeclCheckState::ReadyForReference:
        SemanticsDeclRedeclarationVisitor(shared).dispatch(decl);
        break;

    case DeclCheckState::ReadyForLookup:
        SemanticsDeclBasesVisitor(shared).dispatch(decl);
        break;

    case DeclCheckState::ReadyForConformances:
        SemanticsDeclConformancesVisitor(shared).dispatch(decl);
        break;

    case DeclCheckState::TypesFullyResolved:
        SemanticsDeclTypeResolutionVisitor(shared).dispatch(decl);
        SemanticsDeclDifferentialConformanceVisitor(shared).dispatch(decl);
        break;

    case DeclCheckState::AttributesChecked:
        SemanticsDeclAttributesVisitor(shared).dispatch(decl);
        break;

    case DeclCheckState::DefinitionChecked:
        SemanticsDeclBodyVisitor(shared).dispatch(decl);
        break;

    case DeclCheckState::CapabilityChecked:
        if (!shared.getOptionSet().getBoolOption(CompilerOptionName::IgnoreCapabilities))
        {
            SemanticsDeclCapabilityVisitor(shared).dispatch(decl);
        }
        break;
    }
}


// Replace with <=> in C++20
template<typename T>
int compareThreeWays(T a, T b)
{
    if (a > b)
        return -1;
    else if (b > a)
        return 1;
    else
        return 0;
}

// lhs and rhs cannot be nullptr
int compareDecls(Decl& lhs, Decl& rhs);

// lhs and rhs cannot be nullptr
int compareVals(Val& lhs, Val& rhs);

template<typename T, class Compare>
int comparePtrs(T* lhs, T* rhs, Compare const& compare)
{
    int res = 0;
    if (lhs == rhs)
        res = 0;
    else if (!lhs)
        res = -1;
    else if (!rhs)
        res = 1;
    else
        res = compare(*lhs, *rhs);
    return res;
}

// lhs or rhs might be nullptr
int compareDecls(Decl* lhs, Decl* rhs)
{
    return comparePtrs(lhs, rhs, [&](Decl& lhs, Decl& rhs) { return compareDecls(lhs, rhs); });
}

// lhs or rhs might be nullptr
int compareVals(Val* lhs, Val* rhs)
{
    return comparePtrs(lhs, rhs, [&](Val& lhs, Val& rhs) { return compareVals(lhs, rhs); });
}

// Compare operands of lhs and rhs from offset,
// and at most count operands, if the capacity allows it.
int compareValOperands(Val& lhs, Val& rhs, Index offset, Index count)
{
    const Index lN = std::clamp<Index>(lhs.getOperandCount() - offset, 0, count);
    const Index rN = std::clamp<Index>(rhs.getOperandCount() - offset, 0, count);
    int res = compareThreeWays(lN, rN);
    if (res)
        return res;
    for (Index i = 0; i < lN; ++i)
    {
        auto lOp = lhs.m_operands[offset + i];
        auto rOp = rhs.m_operands[offset + i];
        res = compareThreeWays(lOp.kind, rOp.kind);
        if (res)
        {
            break;
        }
        switch (lOp.kind)
        {
        case ValNodeOperandKind::ConstantValue:
            res = compareThreeWays(lOp.getIntConstant(), rOp.getIntConstant());
            break;
        case ValNodeOperandKind::ValNode:
            res = compareVals(lOp.getVal(), rOp.getVal());
            break;
        case ValNodeOperandKind::ASTNode:
            res = compareDecls(lOp.getDecl(), rOp.getDecl());
            break;
        }
        if (res)
        {
            break;
        }
    }
    return res;
}

// Compare operands of lhs and rhs from offset to the end
int compareValOperands(Val& lhs, Val& rhs, Index offset)
{
    return compareValOperands(lhs, rhs, offset, std::numeric_limits<Index>::max());
}

// Find the lowest common ancestor (LCA) of nodes a and b
// Uppon return, a and b are modified to be direct children of the LCA
// Returns nullptr if a and b have no common ancestor
// (e.g. a and b are not in the same module)
// a and b are set to each respective module
ContainerDecl* findDeclsLowestCommonAncestor(Decl*& a, Decl*& b)
{
    auto ascendToRoot = [](Decl*& d)
    {
        UIndex depth = 0;
        while (d->parentDecl)
        {
            ++depth;
            d = d->parentDecl;
        }
        return depth;
    };

    Decl* aRoot = a;
    Decl* bRoot = b;
    auto aDepth = ascendToRoot(aRoot);
    auto bDepth = ascendToRoot(bRoot);
    if (aRoot != bRoot) // Not in the same tree / module
    {
        a = aRoot;
        b = bRoot;
        return nullptr;
    }
    // Level nodes
    Decl** toAscend = nullptr;
    Decl** reference = nullptr;
    UIndex n = 0;
    if (aDepth > bDepth)
    {
        toAscend = &a;
        reference = &b;
        n = aDepth - bDepth;
    }
    else if (bDepth > aDepth)
    {
        toAscend = &b;
        reference = &a;
        n = bDepth - aDepth;
    }
    if (n)
    {
        // Level until toAscend is one level under reference
        while (n > UIndex(1))
        {
            *toAscend = (*toAscend)->parentDecl;
            --n;
        }
        // If toAscend was a child of reference
        if ((*toAscend)->parentDecl == *reference)
        {
            return (*toAscend)->parentDecl;
        }
        else
        {
            *toAscend = (*toAscend)->parentDecl;
        }
    }
    while (a->parentDecl != b->parentDecl)
    {
        a = a->parentDecl;
        b = b->parentDecl;
    }
    return a->parentDecl;
}

int compareDecls(Decl& lhs, Decl& rhs)
{
    int res = compareThreeWays(lhs.astNodeType, rhs.astNodeType);
    if (res)
        return res;
    Decl* lLCAChild = &lhs;
    Decl* rLCAChild = &rhs;
    if (ContainerDecl* lca = findDeclsLowestCommonAncestor(lLCAChild, rLCAChild))
    {
        res = compareThreeWays(lca->getDeclIndex(lLCAChild), lca->getDeclIndex(rLCAChild));
    }
    else
    {
        res = comparePtrs(
            lLCAChild->getName(),
            rLCAChild->getName(),
            [](Name const& lName, Name const& rName)
            { return strcmp(lName.text.begin(), rName.text.begin()); });
    }
    return res;
}

int compareVals(Val& lhs, Val& rhs)
{
    int res = compareThreeWays(lhs.astNodeType, rhs.astNodeType);
    if (res)
        return res;
    res = compareValOperands(lhs, rhs, 0);
    return res;
}

int compareTypes(Type* lhs, Type* rhs)
{
    return compareVals(lhs, rhs);
}

int compareTypes(Type& lhs, Type& rhs)
{
    return compareVals(lhs, rhs);
}

static void _getCanonicalConstraintTypes(List<Type*>& outTypeList, Type* type)
{
    if (auto andType = as<AndType>(type))
    {
        _getCanonicalConstraintTypes(outTypeList, andType->getLeft());
        _getCanonicalConstraintTypes(outTypeList, andType->getRight());
    }
    else
    {
        outTypeList.add(type);
    }
}
OrderedDictionary<GenericTypeParamDeclBase*, List<Type*>> getCanonicalGenericConstraints(
    ASTBuilder* astBuilder,
    DeclRef<ContainerDecl> genericDecl)
{
    OrderedDictionary<GenericTypeParamDeclBase*, List<Type*>> genericConstraints;
    for (auto mm : getMembersOfType<GenericTypeParamDeclBase>(astBuilder, genericDecl))
    {
        genericConstraints[mm.getDecl()] = List<Type*>();
    }
    for (auto genericTypeConstraintDecl :
         getMembersOfType<GenericTypeConstraintDecl>(astBuilder, genericDecl))
    {
        if (genericTypeConstraintDecl.getDecl()->sub.type->astNodeType == ASTNodeType::DeclRefType)
        {
            auto typeParamDecl = as<DeclRefType>(genericTypeConstraintDecl.getDecl()->sub.type)
                                     ->getDeclRef()
                                     .getDecl();
            List<Type*>* constraintTypes = genericConstraints.tryGetValue(typeParamDecl);
            if (!constraintTypes)
                continue;
            constraintTypes->add(genericTypeConstraintDecl.getDecl()->getSup().type);
        }
        else
        {
            SLANG_UNEXPECTED("Cannot extract Cannonical Generic Constraints on non DeclRefTypes. "
                             "Use getCanonicalGenericConstraints2(...) instead.");
        }
    }
    OrderedDictionary<GenericTypeParamDeclBase*, List<Type*>> result;
    for (auto& constraints : genericConstraints)
    {
        List<Type*> typeList;
        for (auto type : constraints.value)
        {
            _getCanonicalConstraintTypes(typeList, type);
        }
        const auto typeComparator = [&](Type* lhs, Type* rhs)
        { return compareTypes(*lhs, *rhs) < 0; };
        typeList.sort(typeComparator);
        result[constraints.key] = std::move(typeList);
    }
    return result;
}


OrderedDictionary<Type*, List<Type*>> getCanonicalGenericConstraints2(
    ASTBuilder* astBuilder,
    DeclRef<ContainerDecl> genericDecl)
{
    Dictionary<Type*, HashSet<Type*>> genericConstraints;
    for (auto genericTypeConstraintDecl :
         getMembersOfType<GenericTypeConstraintDecl>(astBuilder, genericDecl))
    {
        auto subExpr = genericTypeConstraintDecl.getDecl()->sub;
        auto supExpr = genericTypeConstraintDecl.getDecl()->sup;
        Type* typeToAdd = subExpr.type;
        if (typeToAdd)
        {
            if (!genericConstraints.containsKey(typeToAdd))
            {
                genericConstraints[typeToAdd] = HashSet<Type*>();
            }
            genericConstraints[typeToAdd].add(supExpr.type);
        }
    }
    const auto typeComparator = [&](Type* lhs, Type* rhs) { return compareTypes(lhs, rhs) < 0; };
    const List<Type*> sortedKeys = [&]()
    {
        List<Type*> res;
        res.reserve(genericConstraints.getCount());
        for (auto& t : genericConstraints)
        {
            res.add(t.first);
        }
        res.sort(typeComparator);
        return res;
    }();
    OrderedDictionary<Type*, List<Type*>> result;
    for (auto& key : sortedKeys)
    {
        List<Type*> typeList;
        for (auto type : genericConstraints[key])
        {
            _getCanonicalConstraintTypes(typeList, type);
        }
        typeList.sort(typeComparator);
        result[key] = std::move(typeList);
    }
    return result;
}

bool areTypesCompatibile(SemanticsVisitor* visitor, Type* fst, Type* snd)
{
    if (fst->equals(snd))
        return true;

    if (auto declRefType = as<DeclRefType>(fst))
    {
        auto decl = declRefType->getDeclRef().getDecl();
        if (auto extGenericDecl = visitor->GetOuterGeneric(decl))
        {
            SemanticsVisitor::ConstraintSystem constraints;
            constraints.loc = decl->loc;
            constraints.genericDecl = extGenericDecl;

            if (!visitor->TryUnifyTypes(
                    constraints,
                    SemanticsVisitor::ValUnificationContext(),
                    fst,
                    snd))
                return false;

            ConversionCost baseCost;
            if (!visitor->trySolveConstraintSystem(
                    &constraints,
                    makeDeclRef(extGenericDecl),
                    ArrayView<Val*>(),
                    baseCost))
                return false;

            // If we reach here, it means we have a valid unification.
            return true;
        }
    }
    return false;
}

Type* getTypeForThisExpr(SemanticsVisitor* visitor, FunctionDeclBase* funcDecl)
{
    ThisExpr* expr = visitor->getASTBuilder()->create<ThisExpr>();
    expr->scope = funcDecl->ownedScope;
    expr->loc = funcDecl->loc;

    DiagnosticSink dummySink;
    auto tempVisitor = SemanticsVisitor(visitor->withSink(&dummySink));

    auto checkedExpr = tempVisitor.CheckTerm(expr);

    return !(as<ErrorType>(checkedExpr->type.type)) ? (checkedExpr->type.type) : nullptr;
}

Type* getTypeForThisExpr(SemanticsVisitor* visitor, DeclRef<FunctionDeclBase> funcDeclRef)
{
    auto type = getTypeForThisExpr(visitor, funcDeclRef.getDecl());
    if (type)
        return substituteType(
            SubstitutionSet(funcDeclRef.declRefBase),
            visitor->getASTBuilder(),
            type);
    return nullptr;
}


struct ArgsWithDirectionInfo
{
    List<Expr*> args;
    List<ParameterDirection> directions;

    Expr* thisArg;
    ParameterDirection thisArgDirection;
};

template<typename TDerivativeAttr>
void checkDerivativeAttributeImpl(
    SemanticsVisitor* visitor,
    Decl* funcDecl,
    TDerivativeAttr* attr,
    const List<Expr*>& imaginaryArguments,
    const List<ParameterDirection>& expectedParamDirections,
    Expr* expectedThisArg,
    ParameterDirection expectedThisArgDirection)
{
    if (isInterfaceRequirement(funcDecl))
    {
        visitor->getSink()->diagnose(
            attr,
            Diagnostics::cannotAssociateInterfaceRequirementWithDerivative);
        return;
    }

    SemanticsContext::ExprLocalScope scope;
    auto ctx = visitor->withExprLocalScope(&scope);
    auto subVisitor = SemanticsVisitor(ctx);

    auto exprToCheck = attr->funcExpr;

    // If this is a generic, we want to wrap the call to the derivative method
    // with the generic parameters of the source.
    //
    if (as<GenericDecl>(funcDecl->parentDecl) && !as<GenericAppExpr>(attr->funcExpr))
    {
        auto genericDecl = as<GenericDecl>(funcDecl->parentDecl);
        auto substArgs = getDefaultSubstitutionArgs(ctx.getASTBuilder(), visitor, genericDecl);
        auto appExpr = ctx.getASTBuilder()->create<GenericAppExpr>();

        Index count = 0;
        for (auto member : genericDecl->members)
        {
            if (as<GenericTypeParamDecl>(member) || as<GenericValueParamDecl>(member) ||
                as<GenericTypePackParamDecl>(member))
                count++;
        }

        appExpr->functionExpr = attr->funcExpr;

        for (auto arg : substArgs)
        {
            if (count == 0)
                break;

            if (auto declRefType = as<DeclRefType>(arg))
            {
                auto baseTypeExpr = ctx.getASTBuilder()->create<SharedTypeExpr>();
                baseTypeExpr->base.type = declRefType;
                auto baseTypeType = ctx.getASTBuilder()->getOrCreate<TypeType>(declRefType);
                baseTypeExpr->type.type = baseTypeType;

                appExpr->arguments.add(baseTypeExpr);
            }
            else if (auto genericValParam = as<GenericParamIntVal>(arg))
            {
                auto declRef = genericValParam->getDeclRef();
                appExpr->arguments.add(
                    subVisitor
                        .ConstructDeclRefExpr(declRef, nullptr, nullptr, SourceLoc(), nullptr));
            }
            else
            {
                SLANG_UNEXPECTED("Unhandled substitution arg type");
            }

            count--;
        }

        exprToCheck = appExpr;
    }

    auto checkedFuncExpr = visitor->dispatchExpr(exprToCheck, ctx);
    attr->funcExpr = checkedFuncExpr;
    if (attr->args.getCount())
        attr->args[0] = attr->funcExpr;
    if (auto declRefExpr = as<DeclRefExpr>(checkedFuncExpr))
    {
        if (declRefExpr->declRef)
            visitor->ensureDecl(declRefExpr->declRef, DeclCheckState::TypesFullyResolved);
        else
        {
            visitor->getSink()->diagnose(attr, Diagnostics::cannotResolveDerivativeFunction);
            return;
        }
    }
    else if (auto overloadedExpr = as<OverloadedExpr>(checkedFuncExpr))
    {
        for (auto candidate : overloadedExpr->lookupResult2.items)
        {
            visitor->ensureDecl(candidate.declRef, DeclCheckState::TypesFullyResolved);
        }
    }
    else
    {
        visitor->getSink()->diagnose(attr, Diagnostics::cannotResolveDerivativeFunction);
        return;
    }

    // If left value is true, then convert the
    // inner type to an InOutType.
    //
    auto qualTypeToString = [&](QualType qualType) -> String
    {
        Type* type = qualType.type;
        if (qualType.isLeftValue)
        {
            type = ctx.getASTBuilder()->getInOutType(type);
        }
        return type->toString();
    };

    List<Expr*> argList = imaginaryArguments;
    List<ParameterDirection> paramDirections = expectedParamDirections;
    bool expectStaticFunc = false;

    if (expectedThisArg)
    {
        argList.insert(0, expectedThisArg);
        paramDirections.insert(0, expectedThisArgDirection);
        expectStaticFunc = true;
    }

    auto invokeExpr = subVisitor.constructUncheckedInvokeExpr(checkedFuncExpr, argList);
    auto resolved = subVisitor.ResolveInvoke(invokeExpr);

    if (auto resolvedInvoke = as<InvokeExpr>(resolved))
    {
        if (auto calleeDeclRef = as<DeclRefExpr>(resolvedInvoke->functionExpr))
        {
            // There are two ways to make it to this point.. a proper resolution, and a
            // resolution that has failed due to type mismatch.
            // Further, a proper resolution can still be invalid due to incorrect parameter
            // directionality.
            // We'll detect both these incorrect cases here and issue an appropriate diagnostic.
            //
            auto funcType = as<FuncType>(calleeDeclRef->type);
            if (!funcType)
            {
                // The best candidate does not have a function type.
                // If we reach here, it means the function is a generic and we can't deduce the
                // generic arguments from imaginary argument list.
                // In this case we issue a diagnostic to ask the user to explicitly provide the
                // arguments.
                visitor->getSink()->diagnose(
                    attr,
                    Diagnostics::cannotResolveGenericArgumentForDerivativeFunction);
                return;
            }
            if (isInterfaceRequirement(calleeDeclRef->declRef.getDecl()))
            {
                visitor->getSink()->diagnose(
                    attr,
                    Diagnostics::cannotUseInterfaceRequirementAsDerivative);
                return;
            }
            if (funcType->getParamCount() != argList.getCount())
            {
                goto error;
            }
            for (Index ii = 0; ii < argList.getCount(); ++ii)
            {
                // Check if the resolved invoke argument type is an error type.
                // If so, then we have a type mismatch.
                //
                if (resolvedInvoke->arguments[ii]->type.type->equals(
                        ctx.getASTBuilder()->getErrorType()) ||
                    funcType->getParamDirection(ii) != paramDirections[ii])
                {
                    visitor->getSink()->diagnose(
                        attr,
                        Diagnostics::customDerivativeSignatureMismatchAtPosition,
                        ii,
                        qualTypeToString(argList[ii]->type),
                        funcType->getParamType(ii)->toString());
                }
            }
            // The `imaginaryArguments` list does not include the `this` parameter.
            // So we need to check that `this` type matches.
            bool funcIsStatic = isEffectivelyStatic(funcDecl);
            if (funcIsStatic)
                expectStaticFunc = true;

            bool derivativeFuncIsStatic = isEffectivelyStatic(calleeDeclRef->declRef.getDecl());

            if (expectStaticFunc && !derivativeFuncIsStatic)
            {
                visitor->getSink()->diagnose(attr, Diagnostics::customDerivativeExpectedStatic);
                return;
            }

            if (!derivativeFuncIsStatic)
            {
                auto defaultFuncDeclRef = createDefaultSubstitutionsIfNeeded(
                    visitor->getASTBuilder(),
                    visitor,
                    makeDeclRef(funcDecl));

                DeclRef<FunctionDeclBase> funcDeclRef = defaultFuncDeclRef.as<FunctionDeclBase>();
                auto funcThisType = getTypeForThisExpr(visitor, funcDeclRef);
                DeclRef<FunctionDeclBase> calleeFuncDeclRef =
                    calleeDeclRef->declRef.template as<FunctionDeclBase>();
                auto derivativeFuncThisType = getTypeForThisExpr(visitor, calleeFuncDeclRef);

                // If the function is a member function, we need to check that the
                // `this` type matches the expected type. This will ensure that after lowering
                // to IR, the two functions are compatible.
                //
                if (!areTypesCompatibile(visitor, funcThisType, derivativeFuncThisType))
                {
                    visitor->getSink()->diagnose(
                        attr,
                        Diagnostics::customDerivativeSignatureThisParamMismatch);
                    return;
                }
            }

            // If the two decls are under different generic contexts, we'll need to check that
            // they agree and specialize the attribute's decl-ref accordingly.
            //

            auto originalNextGeneric =
                visitor->findNextOuterGeneric(visitor->getOuterGenericOrSelf(funcDecl));
            auto derivativeNextGeneric = visitor->findNextOuterGeneric(
                visitor->getOuterGenericOrSelf(calleeDeclRef->declRef.getDecl()));

            if ((!originalNextGeneric) != (!derivativeNextGeneric))
            {
                // Diagnostic for when one is generic and the other is not.
                visitor->getSink()->diagnose(
                    attr,
                    Diagnostics::cannotResolveGenericArgumentForDerivativeFunction);
                return;
            }

            if (originalNextGeneric != derivativeNextGeneric)
            {
                // If the two generic containers are not the same, but are compatible, we can
                // unify them.
                //

                DeclRef<Decl> specializedDecl;
                if (!visitor->doGenericSignaturesMatch(
                        originalNextGeneric,
                        derivativeNextGeneric,
                        &specializedDecl))
                {
                    visitor->getSink()->diagnose(
                        attr,
                        Diagnostics::customDerivativeSignatureMismatch);
                    return;
                }

                calleeDeclRef->declRef = substituteDeclRef(
                    SubstitutionSet(specializedDecl),
                    visitor->getASTBuilder(),
                    calleeDeclRef->declRef);
                calleeDeclRef->type = substituteType(
                    SubstitutionSet(specializedDecl),
                    visitor->getASTBuilder(),
                    calleeDeclRef->type);
            }

            attr->funcExpr = calleeDeclRef;
            if (attr->args.getCount())
                attr->args[0] = attr->funcExpr;
            return;
        }
    }
error:;
    // Build the expected signature from imaginary args to diagnose
    // when no matching function is found (this excludes the case handled above)
    //
    StringBuilder builder;
    builder << "(";
    for (Index ii = 0; ii < argList.getCount(); ++ii)
    {
        if (ii != 0)
            builder << ", ";
        if (argList[ii]->type)
            builder << qualTypeToString(argList[ii]->type);
        else
            builder << "<error>";
    }
    builder << ")";

    visitor->getSink()->diagnose(
        attr,
        Diagnostics::customDerivativeSignatureMismatch,
        builder.produceString());
}

template<typename TDerivativeAttr>
const char* getDerivativeAttrName()
{
    SLANG_UNREACHABLE("");
}

template<>
const char* getDerivativeAttrName<ForwardDerivativeAttribute>()
{
    return "ForwardDerivative";
}
template<>
const char* getDerivativeAttrName<BackwardDerivativeAttribute>()
{
    return "BackwardDerivative";
}
template<>
const char* getDerivativeAttrName<PrimalSubstituteAttribute>()
{
    return "PrimalSubstitute";
}

ArgsWithDirectionInfo getImaginaryArgsToFunc(
    ASTBuilder* astBuilder,
    FunctionDeclBase* func,
    SourceLoc loc)
{
    List<Expr*> imaginaryArguments;
    List<ParameterDirection> directions;
    for (auto param : func->getParameters())
    {
        auto arg = astBuilder->create<VarExpr>();
        arg->declRef = makeDeclRef(param);
        arg->type.isLeftValue = param->findModifier<OutModifier>() ? true : false;
        arg->type.type = param->getType();
        arg->loc = loc;
        imaginaryArguments.add(arg);
        directions.add(getParameterDirection(param));
    }
    return {imaginaryArguments, directions, nullptr, ParameterDirection::kParameterDirection_In};
}

ArgsWithDirectionInfo getImaginaryArgsToForwardDerivative(
    SemanticsVisitor* visitor,
    FunctionDeclBase* originalFuncDecl,
    SourceLoc loc)
{
    Expr* thisArgExpr = nullptr;
    if (auto thisType = getTypeForThisExpr(visitor, originalFuncDecl))
    {
        thisArgExpr = visitor->getASTBuilder()->create<VarExpr>();
        thisArgExpr->type = thisType;
        thisArgExpr->loc = loc;

        if (visitor->isTypeDifferentiable(thisType) &&
            !originalFuncDecl->findModifier<NoDiffThisAttribute>() &&
            !isEffectivelyStatic(originalFuncDecl))
        {
            auto pairType = visitor->getDifferentialPairType(thisType);
            thisArgExpr->type.type = pairType;
        }
        else
        {
            thisArgExpr = nullptr;
        }
    }

    ParameterDirection thisTypeDirection = (thisArgExpr && !thisArgExpr->type.isLeftValue)
                                               ? ParameterDirection::kParameterDirection_In
                                               : ParameterDirection::kParameterDirection_InOut;

    List<Expr*> imaginaryArguments;
    for (auto param : originalFuncDecl->getParameters())
    {
        auto arg = visitor->getASTBuilder()->create<VarExpr>();
        arg->declRef = makeDeclRef(param);
        arg->type.isLeftValue = param->findModifier<OutModifier>() ? true : false;
        arg->type.type = param->getType();
        arg->loc = loc;
        if (!param->findModifier<NoDiffModifier>())
        {
            if (auto pairType = visitor->getDifferentialPairType(param->getType()))
            {
                arg->type.type = pairType;
            }
        }
        imaginaryArguments.add(arg);
    }

    // Copy parameter directions as is.
    List<ParameterDirection> expectedParamDirections;
    for (auto param : originalFuncDecl->getParameters())
    {
        expectedParamDirections.add(getParameterDirection(param));
    }

    return {imaginaryArguments, expectedParamDirections, thisArgExpr, thisTypeDirection};
}

ArgsWithDirectionInfo getImaginaryArgsToBackwardDerivative(
    SemanticsVisitor* visitor,
    FunctionDeclBase* originalFuncDecl,
    SourceLoc loc)
{
    Expr* thisArgExpr = nullptr;
    if (auto thisType = getTypeForThisExpr(visitor, originalFuncDecl))
    {
        thisArgExpr = visitor->getASTBuilder()->create<VarExpr>();
        thisArgExpr->type = thisType;
        thisArgExpr->loc = loc;

        if (visitor->isTypeDifferentiable(thisType) &&
            !originalFuncDecl->findModifier<NoDiffThisAttribute>() &&
            !isEffectivelyStatic(originalFuncDecl))
        {
            auto pairType = visitor->getDifferentialPairType(thisType);
            thisArgExpr->type.type = pairType;

            // TODO: for ptr pair types, no need to set isLeftValue to true.
            if (as<DifferentialPairType>(thisArgExpr->type.type))
                thisArgExpr->type.isLeftValue = true;
        }
        else
        {
            thisArgExpr = nullptr;
        }
    }

    ParameterDirection thisTypeDirection = (thisArgExpr && !thisArgExpr->type.isLeftValue)
                                               ? ParameterDirection::kParameterDirection_In
                                               : ParameterDirection::kParameterDirection_InOut;

    List<Expr*> imaginaryArguments;
    List<ParameterDirection> expectedParamDirections;

    auto isOutParam = [&](ParamDecl* param)
    {
        return param->findModifier<OutModifier>() != nullptr &&
               param->findModifier<InModifier>() == nullptr &&
               param->findModifier<InOutModifier>() == nullptr;
    };

    for (auto param : originalFuncDecl->getParameters())
    {
        auto arg = visitor->getASTBuilder()->create<VarExpr>();
        arg->declRef = makeDeclRef(param);
        arg->type.isLeftValue = param->findModifier<OutModifier>() ? true : false;
        arg->type.type = param->getType();
        arg->loc = loc;

        ParameterDirection direction = getParameterDirection(param);

        bool isDiffParam = (!param->findModifier<NoDiffModifier>());
        if (isDiffParam)
        {
            auto diffPair = visitor->getDifferentialPairType(param->getType());
            if (auto pairType = as<DifferentialPairType>(diffPair))
            {
                arg->type.type = pairType;
                arg->type.isLeftValue = true;

                if (isOutParam(param))
                {
                    // out T : IDifferentiable -> in T.Differential
                    arg->type.isLeftValue = false;
                    arg->type.type = visitor->tryGetDifferentialType(
                        visitor->getASTBuilder(),
                        pairType->getPrimalType());

                    direction = ParameterDirection::kParameterDirection_In;
                }
                else
                {
                    // in T : IDifferentiable -> inout DifferentialPair<T>
                    // inout T : IDifferentiable -> inout DifferentialPair<T>
                    direction = ParameterDirection::kParameterDirection_InOut;
                }
            }
            else if (auto refPairType = as<DifferentialPtrPairType>(diffPair))
            {
                // no need to change direction of ref-pairs.
                arg->type.type = refPairType;
            }
            else
            {
                isDiffParam = false;
            }
        }
        if (!isDiffParam)
        {
            if (isOutParam(param))
            {
                // Skip non-differentiable out params.
                continue;
            }

            // no_diff inout T -> in T
            // no_diff in T -> in T
            //
            direction = ParameterDirection::kParameterDirection_In;
        }

        imaginaryArguments.add(arg);
        expectedParamDirections.add(direction);
    }
    if (auto diffReturnType = visitor->tryGetDifferentialType(
            visitor->getASTBuilder(),
            originalFuncDecl->returnType.type))
    {
        auto arg = visitor->getASTBuilder()->create<InitializerListExpr>();
        arg->type.isLeftValue = false;
        arg->type.type = diffReturnType;
        arg->loc = loc;
        imaginaryArguments.add(arg);
        expectedParamDirections.add(ParameterDirection::kParameterDirection_In);
    }

    return {imaginaryArguments, expectedParamDirections, thisArgExpr, thisTypeDirection};
}

// This helper function is needed to workaround a gcc bug.
// Remove when we upgrade to a newer version of gcc.
template<typename T>
static T* _findModifier(Decl* decl)
{
    return decl->findModifier<T>();
}

template<typename TDerivativeAttr, typename TDifferentiateExpr, typename TDerivativeOfAttr>
void checkDerivativeOfAttributeImpl(
    SemanticsVisitor* visitor,
    FunctionDeclBase* funcDecl,
    TDerivativeOfAttr* derivativeOfAttr,
    DeclAssociationKind assocKind)
{
    auto astBuilder = visitor->getASTBuilder();
    DeclRef<Decl> calleeDeclRef;
    DeclRefExpr* calleeDeclRefExpr = nullptr;
    HigherOrderInvokeExpr* higherOrderFuncExpr = astBuilder->create<TDifferentiateExpr>();
    higherOrderFuncExpr->baseFunction = derivativeOfAttr->funcExpr;
    if (derivativeOfAttr->args.getCount() > 0)
        higherOrderFuncExpr->loc = derivativeOfAttr->args[0]->loc;

    Expr* checkedHigherOrderFuncExpr = visitor->dispatchExpr(
        higherOrderFuncExpr,
        visitor->allowStaticReferenceToNonStaticMember());

    if (!checkedHigherOrderFuncExpr)
    {
        visitor->getSink()->diagnose(
            derivativeOfAttr,
            Diagnostics::cannotResolveOriginalFunctionForDerivative);
        return;
    }
    List<Expr*> imaginaryArgs =
        getImaginaryArgsToFunc(astBuilder, funcDecl, derivativeOfAttr->loc).args;
    auto invokeExpr =
        visitor->constructUncheckedInvokeExpr(checkedHigherOrderFuncExpr, imaginaryArgs);
    SemanticsContext::ExprLocalScope scope;
    auto ctx = visitor->withExprLocalScope(&scope);
    auto subVisitor = SemanticsVisitor(ctx);
    auto resolved = subVisitor.ResolveInvoke(invokeExpr);
    if (auto resolvedInvoke = as<InvokeExpr>(resolved))
    {
        auto resolvedFuncExpr = as<HigherOrderInvokeExpr>(resolvedInvoke->functionExpr);
        if (resolvedFuncExpr)
        {
            calleeDeclRefExpr = as<DeclRefExpr>(resolvedFuncExpr->baseFunction);
            if (!calleeDeclRef && as<OverloadedExpr>(resolvedFuncExpr->baseFunction))
            {
                visitor->getSink()->diagnose(
                    derivativeOfAttr,
                    Diagnostics::overloadedFuncUsedWithDerivativeOfAttributes);
            }
        }
    }

    if (!calleeDeclRefExpr)
    {
        visitor->getSink()->diagnose(
            derivativeOfAttr,
            Diagnostics::cannotResolveOriginalFunctionForDerivative);
        return;
    }

    calleeDeclRefExpr->loc = higherOrderFuncExpr->loc;
    if (derivativeOfAttr->args.getCount() > 0)
        derivativeOfAttr->args[0] = calleeDeclRefExpr;

    calleeDeclRef = calleeDeclRefExpr->declRef;

    auto calleeFunc = as<FunctionDeclBase>(calleeDeclRef.getDecl());

    if (!calleeFunc)
    {
        // If we couldn't find a direct function, it might be a generic.
        if (auto genericDecl = as<GenericDecl>(calleeDeclRef.getDecl()))
        {
            calleeFunc = as<FunctionDeclBase>(genericDecl->inner);

            if (as<ErrorType>(resolved->type.type))
            {
                // If we can't resolve a type, something went wrong. If we're working with a
                // generic decl, the most likely cause is a failure of generic argument
                // inference.
                //
                visitor->getSink()->diagnose(
                    derivativeOfAttr,
                    Diagnostics::cannotResolveGenericArgumentForDerivativeFunction);
            }
        }
    }

    if (!calleeFunc)
    {
        visitor->getSink()->diagnose(
            derivativeOfAttr,
            Diagnostics::cannotResolveOriginalFunctionForDerivative);
        return;
    }

    // For now, if calleeFunc or funcDecl is nested inside some generic aggregate,
    // they must be the same generic decl. For example, using B<T>.f() as the original function
    // for C<T>.derivative() is not allowed.
    // We may relax this restriction in the future by solving the "inverse" generic arguments
    // from the `calleeDeclRef`, and use them to create a declRef to funcDecl from the original
    // func.

    if (isInterfaceRequirement(calleeFunc))
    {
        visitor->getSink()->diagnose(
            derivativeOfAttr,
            Diagnostics::cannotAssociateInterfaceRequirementWithDerivative);
        return;
    }
    if (isInterfaceRequirement(funcDecl))
    {
        visitor->getSink()->diagnose(
            derivativeOfAttr,
            Diagnostics::cannotUseInterfaceRequirementAsDerivative);
        return;
    }

    if (auto existingModifier = _findModifier<TDerivativeAttr>(calleeFunc))
    {
        // The primal function already has a `[*Derivative]` attribute, this is invalid.
        visitor->getSink()->diagnose(
            derivativeOfAttr,
            Diagnostics::declAlreadyHasAttribute,
            calleeDeclRef,
            getDerivativeAttrName<TDerivativeAttr>());
        visitor->getSink()->diagnose(
            existingModifier->loc,
            Diagnostics::seeDeclarationOf,
            calleeDeclRef.getDecl());
    }

    derivativeOfAttr->funcExpr = calleeDeclRefExpr;
    auto derivativeAttr = astBuilder->create<TDerivativeAttr>();
    derivativeAttr->loc = derivativeOfAttr->loc;
    auto outterGeneric = visitor->GetOuterGeneric(funcDecl);
    auto declRef = makeDeclRef<Decl>((outterGeneric ? (Decl*)outterGeneric : funcDecl));

    // If both the derivative and the original function are defined in the same outer generic
    // aggregate type, we want to form a full declref with default arguments.
    declRef = createDefaultSubstitutionsIfNeeded(astBuilder, visitor, declRef);

    auto declRefExpr = visitor->ConstructDeclRefExpr(
        declRef,
        nullptr,
        declRef.getName(),
        derivativeOfAttr->loc,
        nullptr);
    declRefExpr->type.type = nullptr;
    derivativeAttr->args.add(declRefExpr);
    derivativeAttr->funcExpr = declRefExpr;
    checkDerivativeAttribute(visitor, calleeFunc, derivativeAttr);
    derivativeOfAttr->backDeclRef = derivativeAttr->funcExpr;
    derivativeAttr->funcExpr = nullptr;
    visitor->getShared()->registerAssociatedDecl(calleeDeclRef.getDecl(), assocKind, funcDecl);
}

static void checkDerivativeAttribute(
    SemanticsVisitor* visitor,
    FunctionDeclBase* funcDecl,
    ForwardDerivativeAttribute* attr)
{
    if (!attr->funcExpr)
        return;
    if (attr->funcExpr->type.type)
        return;

    ArgsWithDirectionInfo imaginaryArguments =
        getImaginaryArgsToForwardDerivative(visitor, funcDecl, attr->loc);
    checkDerivativeAttributeImpl(
        visitor,
        funcDecl,
        attr,
        imaginaryArguments.args,
        imaginaryArguments.directions,
        imaginaryArguments.thisArg,
        imaginaryArguments.thisArgDirection);
}

static void checkDerivativeAttribute(
    SemanticsVisitor* visitor,
    FunctionDeclBase* funcDecl,
    BackwardDerivativeAttribute* attr)
{
    if (!attr->funcExpr)
        return;
    if (attr->funcExpr->type.type)
        return;

    ArgsWithDirectionInfo imaginaryArguments =
        getImaginaryArgsToBackwardDerivative(visitor, funcDecl, attr->loc);
    checkDerivativeAttributeImpl(
        visitor,
        funcDecl,
        attr,
        imaginaryArguments.args,
        imaginaryArguments.directions,
        imaginaryArguments.thisArg,
        imaginaryArguments.thisArgDirection);
}

static void checkDerivativeAttribute(
    SemanticsVisitor* visitor,
    FunctionDeclBase* funcDecl,
    PrimalSubstituteAttribute* attr)
{
    if (!attr->funcExpr)
        return;
    if (attr->funcExpr->type.type)
        return;

    ArgsWithDirectionInfo imaginaryArguments =
        getImaginaryArgsToFunc(visitor->getASTBuilder(), funcDecl, attr->loc);
    checkDerivativeAttributeImpl(
        visitor,
        funcDecl,
        attr,
        imaginaryArguments.args,
        imaginaryArguments.directions,
        imaginaryArguments.thisArg,
        imaginaryArguments.thisArgDirection);

    // For primal-substitute we'd also want to make sure that the differentiability
    // level of the target is as high as the funcDecl itself
    //
    if (auto declRefExpr = as<DeclRefExpr>(attr->funcExpr))
    {
        if (auto declRef = declRefExpr->declRef)
        {
            auto targetDiffLevel = visitor->getShared()->getFuncDifferentiableLevel(
                declRef.as<FunctionDeclBase>().getDecl());
            auto currDiffLevel = visitor->getShared()->getFuncDifferentiableLevel(funcDecl);
            if (targetDiffLevel < currDiffLevel)
            {
                visitor->getSink()->diagnose(
                    attr->loc,
                    Diagnostics::primalSubstituteTargetMustHaveHigherDifferentiabilityLevel,
                    declRefExpr->declRef.getDecl(),
                    funcDecl);
            }
        }
    }
}

static void checkCudaKernelAttribute(
    SemanticsVisitor* visitor,
    FunctionDeclBase* funcDecl,
    CudaKernelAttribute*)
{
    // If the method is also marked differentiable, check that the data types are either
    // non-differentiable or marked with no_diff.
    //
    // Note: This is a temporary restriction until we have a more complete story for
    // differentiability.
    //
    if (funcDecl->findModifier<DifferentiableAttribute>())
    {
        for (auto paramDecl : funcDecl->getParameters())
        {
            auto paramType = paramDecl->type;

            if (visitor->isTypeDifferentiable(paramType))
            {
                if (!paramDecl->hasModifier<NoDiffModifier>())
                {
                    visitor->getSink()->diagnose(
                        paramDecl,
                        Diagnostics::differentiableKernelEntryPointCannotHaveDifferentiableParams);
                }
            }
        }
    }
}

template<typename TDerivativeAttr, typename TDerivativeOfAttr>
bool tryCheckDerivativeOfAttributeImpl(
    SemanticsVisitor* visitor,
    FunctionDeclBase* funcDecl,
    TDerivativeOfAttr* derivativeOfAttr,
    DeclAssociationKind assocKind,
    const List<Expr*>& imaginaryArgsToOriginal)
{
    DiagnosticSink tempSink(visitor->getSourceManager(), nullptr);
    SemanticsVisitor subVisitor(visitor->withSink(&tempSink));
    checkDerivativeOfAttributeImpl<TDerivativeAttr>(
        &subVisitor,
        funcDecl,
        derivativeOfAttr,
        assocKind,
        imaginaryArgsToOriginal);
    return tempSink.getErrorCount() == 0;
}

void SemanticsDeclAttributesVisitor::checkVarDeclCommon(VarDeclBase* varDecl)
{
    bool hasSpecConstAttr = false;
    bool hasPushConstAttr = false;
    for (auto modifier : varDecl->modifiers)
    {
        if (as<SpecializationConstantAttribute>(modifier) || as<VkConstantIdAttribute>(modifier))
        {
            // Specialization constant.
            // Check that type is basic type.
            if (!as<BasicExpressionType>(varDecl->getType()) && !as<ErrorType>(varDecl->getType()))
            {
                getSink()->diagnose(modifier, Diagnostics::specializationConstantMustBeScalar);
            }
            hasSpecConstAttr = true;
        }
        else if (as<PushConstantAttribute>(modifier))
        {
            hasPushConstAttr = true;
        }
    }
    if (hasSpecConstAttr && hasPushConstAttr)
    {
        getSink()->diagnose(
            varDecl,
            Diagnostics::variableCannotBePushAndSpecializationConstant,
            varDecl->getName());
    }
    if (hasSpecConstAttr || hasPushConstAttr)
    {
        if (varDecl->findModifier<HLSLStaticModifier>())
        {
            getSink()->diagnose(varDecl, Diagnostics::pushOrSpecializationConstantCannotBeStatic);
        }
    }
}

void SemanticsDeclAttributesVisitor::checkForwardDerivativeOfAttribute(
    FunctionDeclBase* funcDecl,
    ForwardDerivativeOfAttribute* attr)
{
    checkDerivativeOfAttributeImpl<ForwardDerivativeAttribute, ForwardDifferentiateExpr>(
        this,
        funcDecl,
        attr,
        DeclAssociationKind::ForwardDerivativeFunc);
}

void SemanticsDeclAttributesVisitor::checkBackwardDerivativeOfAttribute(
    FunctionDeclBase* funcDecl,
    BackwardDerivativeOfAttribute* attr)
{
    checkDerivativeOfAttributeImpl<BackwardDerivativeAttribute, BackwardDifferentiateExpr>(
        this,
        funcDecl,
        attr,
        DeclAssociationKind::BackwardDerivativeFunc);
}

void SemanticsDeclAttributesVisitor::checkPrimalSubstituteOfAttribute(
    FunctionDeclBase* funcDecl,
    PrimalSubstituteOfAttribute* attr)
{
    checkDerivativeOfAttributeImpl<PrimalSubstituteAttribute, PrimalSubstituteExpr>(
        this,
        funcDecl,
        attr,
        DeclAssociationKind::PrimalSubstituteFunc);
}

bool SemanticsDeclAttributesVisitor::collectInitializableMembers(
    StructDecl* structDecl,
    const DeclVisibility ctorVisibility,
    List<VarDeclBase*>& resultMembers)
{
    auto findMembers = [&](StructDecl* structDecl)
    {
        for (auto varDeclRef : getMembersOfType<VarDeclBase>(
                 getASTBuilder(),
                 structDecl,
                 MemberFilterStyle::Instance))
        {
            auto varDecl = varDeclRef.getDecl();
            if (getDeclVisibility(varDecl) < ctorVisibility)
                continue;

            auto type = GetTypeForDeclRef(varDeclRef, varDecl->loc);
            if (!type.isLeftValue)
                continue;

            resultMembers.add(varDecl);
            structDecl->m_membersVisibleInCtor.add(varDecl);
        }
    };

    // Find the base type's members first
    for (auto inheritanceMember : structDecl->getMembersOfType<InheritanceDecl>())
    {
        // For base types, we need to pick their parameters of the constructor to the derived
        // type's constructor
        if (auto baseTypeDeclRef = isDeclRefTypeOf<StructDecl>(inheritanceMember->base.type))
        {
            // We should only find the member initialization constructor because it is the
            // constructor has parameters
            ConstructorDecl* ctor = _getSynthesizedConstructor(
                baseTypeDeclRef.getDecl(),
                ConstructorDecl::ConstructorFlavor::SynthesizedMemberInit);

            // The constructor has to have higher or equal visibility level than the struct
            // itself, otherwise, it's not accessible so we will not pick up.
            if (ctor && getDeclVisibility(ctor) >= ctorVisibility)
            {
                for (ParamDecl* param : ctor->getParameters())
                {
                    // Because the parameters in the ctor must have the higher or equal
                    // visibility than the ctor itself, we don't need to check the visibility
                    // level of the parameter.
                    resultMembers.add(param);
                }
            }
        }
    }

    // Find the struct's members
    findMembers(structDecl);
    return (resultMembers.getCount() > 0);
}

// If a struct's member:
// 1. has an initialize expression: Struct S {int a = 1;}; or
// 2. is a default initializable type
//    Note, If a type is not default initializable, it doesn't have default value.
// it can be associated with default value expression in the constructor signature.
// This function helps to check whether either of those 2 conditions are met and create
// a default value for the parameter.
// It's totally fine that there is no default value for the parameter, in this case, user
// code has to provide the argument for this parameter.
static Expr* _getParamDefaultValue(SemanticsVisitor* visitor, VarDeclBase* varDecl)
{
    // 1st condition is easy, we can just use the init expression as the default value.
    if (varDecl->initExpr)
    {
        return varDecl->initExpr;
    }

    if (!varDecl->type || !varDecl->type.type)
        return nullptr;

    if (!isDefaultInitializable(varDecl))
        return nullptr;

    return constructDefaultConstructorForType(visitor, varDecl->type.type);
}

bool SemanticsDeclAttributesVisitor::_synthesizeCtorSignature(StructDecl* structDecl)
{
    // If a type or its base type already defines any explicit constructors, do not synthesize
    // any constructors. see:
    // https://github.com/shader-slang/slang/blob/master/docs/proposals/004-initialization.md#inheritance-initialization
    if (_hasExplicitConstructor(structDecl, true))
        return true;

    // synthesize the signature first.
    // The constructor's visibility level is the same as the struct itself.
    // See
    // https://github.com/shader-slang/spec/blob/main/proposals/004-initialization.md#synthesis-of-constructors-for-member-initialization
    DeclVisibility ctorVisibility = getDeclVisibility(structDecl);

    // Only the members whose visibility level is higher or equal than the
    // constructor's visibility level will appear in the constructor's parameter list.
    List<VarDeclBase*> resultMembers;
    if (!collectInitializableMembers(structDecl, ctorVisibility, resultMembers))
        return false;

    // synthesize the constructor signature:
    // 1. The constructor's name is always `$init`, we create one without parameters now.
    ConstructorDecl* ctor = createCtor(structDecl, ctorVisibility);
    ctor->addFlavor(ConstructorDecl::ConstructorFlavor::SynthesizedMemberInit);

    ctor->members.reserve(resultMembers.getCount());

    // 2. Add the parameter list
    bool stopProcessingDefaultValues = false;
    for (SlangInt i = resultMembers.getCount() - 1; i >= 0; i--)
    {
        auto member = resultMembers[i];
        auto parentAggDecl = getParentAggTypeDecl(member);

        auto ctorParam = m_astBuilder->create<ParamDecl>();
        ctorParam->type = (TypeExp)member->type;
        if (auto atomicType = as<AtomicType>(ctorParam->type))
        {
            ctorParam->type.type = atomicType->getElementType();
        }

        if (!stopProcessingDefaultValues)
            ctorParam->initExpr = _getParamDefaultValue(this, member);

        if (!ctorParam->initExpr)
            stopProcessingDefaultValues = true;

        ctorParam->parentDecl = ctor;

        Name* paramName =
            (parentAggDecl == structDecl)
                ? member->getName()
                : getName(parentAggDecl->getName()->text + "_" + member->getName()->text);

        ctorParam->nameAndLoc = NameLoc(paramName, ctor->loc);

        ctorParam->loc = ctor->loc;
        ctor->members.add(ctorParam);

        // We need to ensure member is `no_diff` if it cannot be differentiated, `ctor`
        // modifiers do not matter in this case since member-wise ctor is always differentiable
        // or "treat as differentiable".
        if (!isTypeDifferentiable(member->getType()) || member->hasModifier<NoDiffModifier>())
        {
            auto noDiffMod = m_astBuilder->create<NoDiffModifier>();
            noDiffMod->loc = ctorParam->loc;
            addModifier(ctorParam, noDiffMod);
        }
    }
    ctor->members.reverse();
    return true;
}

void SemanticsDeclAttributesVisitor::visitStructDecl(StructDecl* structDecl)
{
    // add the member initialize constructor here to avoid circular checking logic
    if (!_synthesizeCtorSignature(structDecl))
    {
        // add a default CTor if missing; checking in attributes
        // to avoid circular checking logic
        auto defaultCtor = _getDefaultCtor(structDecl);
        if (!defaultCtor)
        {
            DeclVisibility ctorVisibility = getDeclVisibility(structDecl);
            createCtor(structDecl, ctorVisibility);
        }
    }

    // Check if this is a ray payload struct and validate field access qualifiers
    if (structDecl->findModifier<RayPayloadAttribute>())
    {
        checkRayPayloadStructFields(structDecl);
    }

    int backingWidth = 0;
    [[maybe_unused]] int totalWidth = 0;
    struct BitFieldInfo
    {
        int memberIndex;
        int bitWidth;
        Type* memberType;
        BitFieldModifier* bitFieldModifier;
    };
    List<BitFieldInfo> groupInfo;

    int memberIndex = 0;
    int backing_nonce = 0;
    const auto dispatchSomeBitPackedMembers = [&]()
    {
        SLANG_ASSERT(totalWidth <= backingWidth);
        SLANG_ASSERT(backingWidth <= 64);

        // We're going to insert a backing member to be referenced in
        // all the bitfield properties
        if (groupInfo.getCount())
        {
            const auto backingMemberBasicType = backingWidth <= 8    ? BaseType::UInt8
                                                : backingWidth <= 16 ? BaseType::UInt16
                                                : backingWidth <= 32 ? BaseType::UInt
                                                                     : BaseType::UInt64;
            auto backingMember = m_astBuilder->create<VarDecl>();
            backingMember->type.type = m_astBuilder->getBuiltinType(backingMemberBasicType);
            backingMember->nameAndLoc.name =
                getName(String("$bit_field_backing_") + String(backing_nonce));
            backing_nonce++;
            backingMember->initExpr = nullptr;
            backingMember->parentDecl = structDecl;
            const auto backingMemberDeclRef = DeclRef<VarDecl>(backingMember->getDefaultDeclRef());

            int bottomOfMember = 0;
            for (const auto m : groupInfo)
            {
                SLANG_ASSERT(bottomOfMember <= backingWidth);

                m.bitFieldModifier->backingDeclRef = backingMemberDeclRef;
                m.bitFieldModifier->offset = bottomOfMember;

                bottomOfMember += m.bitWidth;
            }

            const auto backingMemberIndex = groupInfo[0].memberIndex;
            structDecl->members.insert(backingMemberIndex, backingMember);
            structDecl->invalidateMemberDictionary();
            ++memberIndex;
        }
        structDecl->buildMemberDictionary();

        // Reset everything
        backingWidth = 0;
        totalWidth = 0;
        groupInfo.clear();
    };
    for (; memberIndex < structDecl->members.getCount(); ++memberIndex)
    {
        const auto& m = structDecl->members[memberIndex];

        // We can trivially skip any non-property decls
        const auto v = as<PropertyDecl>(m);
        if (!v)
        {
            // If this is a non-bitfield member then finish the current group
            if (as<VarDecl>(m))
                dispatchSomeBitPackedMembers();
            continue;
        }

        const auto bfm = m->findModifier<BitFieldModifier>();
        // If there isn't a bit field modifier, then dispatch the
        // current group and continue
        if (!bfm)
        {
            dispatchSomeBitPackedMembers();
            continue;
        }

        // Verify that this makes sense as a bitfield
        const auto t = v->type.type->getCanonicalType();
        SLANG_ASSERT(t);
        const auto b = as<BasicExpressionType>(t);
        if (!b)
        {
            getSink()->diagnose(v->loc, Diagnostics::bitFieldNonIntegral, t);
            continue;
        }
        const auto baseType = b->getBaseType();
        const bool isIntegerType = isIntegerBaseType(baseType);
        if (!isIntegerType)
        {
            getSink()->diagnose(v->loc, Diagnostics::bitFieldNonIntegral, t);
            continue;
        }

        // The bit width of this member, and the member type width
        const auto thisFieldWidth = bfm->width;
        const auto thisFieldTypeWidth = getTypeBitSize(b);
        SLANG_ASSERT(thisFieldTypeWidth != 0);
        if (thisFieldWidth > thisFieldTypeWidth)
        {
            getSink()->diagnose(
                v->loc,
                Diagnostics::bitFieldTooWide,
                thisFieldWidth,
                t,
                thisFieldTypeWidth);
            // Not much we can do with this field, just ignore it
            continue;
        }

        // At this point we're sure that we have a bit field,
        // update our bit packing state

        // If there's a 0 width type, dispatch the current group
        if (thisFieldWidth == 0)
            dispatchSomeBitPackedMembers();

        // If this member wouldn't fit into the current group, dispatch
        // everything so far;
        if (totalWidth + thisFieldWidth > std::max(thisFieldTypeWidth, backingWidth))
            dispatchSomeBitPackedMembers();

        // Add this member to the group,
        // Grow the backing width if necessary
        backingWidth = std::max(thisFieldTypeWidth, backingWidth);
        // Grow the total width
        totalWidth += int(thisFieldWidth);
        groupInfo.add({memberIndex, int(thisFieldWidth), t, bfm});
    }
    // If the struct ended with a bitpacked member, then make sure we don't forget the last
    // group
    dispatchSomeBitPackedMembers();
}

void SemanticsDeclAttributesVisitor::visitFunctionDeclBase(FunctionDeclBase* decl)
{
    // Run checking on attributes that can't be fully checked in header checking stage.
    for (auto attr : decl->modifiers)
    {
        if (auto fwdDerivativeOfAttr = as<ForwardDerivativeOfAttribute>(attr))
            checkForwardDerivativeOfAttribute(decl, fwdDerivativeOfAttr);
        else if (auto bwdDerivativeOfAttr = as<BackwardDerivativeOfAttribute>(attr))
            checkBackwardDerivativeOfAttribute(decl, bwdDerivativeOfAttr);
        else if (auto primalOfAttr = as<PrimalSubstituteOfAttribute>(attr))
            checkPrimalSubstituteOfAttribute(decl, primalOfAttr);
        else if (auto fwdDerivativeAttr = as<ForwardDerivativeAttribute>(attr))
            checkDerivativeAttribute(this, decl, fwdDerivativeAttr);
        else if (auto bwdDerivativeAttr = as<BackwardDerivativeAttribute>(attr))
            checkDerivativeAttribute(this, decl, bwdDerivativeAttr);
        else if (auto primalAttr = as<PrimalSubstituteAttribute>(attr))
            checkDerivativeAttribute(this, decl, primalAttr);
        else if (auto cudaKernelAttr = as<CudaKernelAttribute>(attr))
            checkCudaKernelAttribute(this, decl, cudaKernelAttr);
    }
}

static void _propagateSeeDefinitionOf(
    SemanticsVisitor* visitor,
    Decl* funcDecl,
    DiagnosticCategory diagnosticCategory)
{
    maybeDiagnose(
        visitor->getSink(),
        visitor->getOptionSet(),
        diagnosticCategory,
        funcDecl,
        Diagnostics::seeDefinitionOf,
        funcDecl);
}

static void _propagateRequirement(
    SemanticsVisitor* visitor,
    CapabilitySet& resultCaps,
    SyntaxNode* userNode,
    SyntaxNode* referencedNode,
    const CapabilitySet& nodeCaps,
    SourceLoc referenceLoc)
{
    auto referencedDecl = as<Decl>(referencedNode);

    // Ignore cyclic references.
    if (referencedDecl)
    {
        if (referencedDecl->checkState.isBeingChecked())
            return;

        ensureDecl(visitor, referencedDecl, DeclCheckState::CapabilityChecked);
    }

    if (resultCaps.implies(nodeCaps))
        return;

    auto oldCaps = resultCaps;
    bool isAnyInvalid = resultCaps.isInvalid() || nodeCaps.isInvalid();
    resultCaps.join(nodeCaps);

    auto decl = as<Decl>(userNode);

    if (!isAnyInvalid && resultCaps.isInvalid())
    {
        // If joining the referenced decl's requirements results an invalid capability set,
        // then the decl is using things that require conflicting set of capabilities, and we
        // should diagnose an error.
        if (referencedDecl && decl)
        {
            maybeDiagnose(
                visitor->getSink(),
                visitor->getOptionSet(),
                DiagnosticCategory::Capability,
                referenceLoc,
                Diagnostics::conflictingCapabilityDueToUseOfDecl,
                referencedDecl,
                nodeCaps,
                decl,
                oldCaps);
        }
        else if (decl)
        {
            maybeDiagnose(
                visitor->getSink(),
                visitor->getOptionSet(),
                DiagnosticCategory::Capability,
                referenceLoc,
                Diagnostics::conflictingCapabilityDueToStatement,
                nodeCaps,
                decl,
                oldCaps);
        }
        else
        {
            maybeDiagnose(
                visitor->getSink(),
                visitor->getOptionSet(),
                DiagnosticCategory::Capability,
                referenceLoc,
                Diagnostics::conflictingCapabilityDueToStatementEnclosingFunc,
                nodeCaps,
                oldCaps);
        }
    }

    // if stmt inside parent, set the provenance tracker to the calling function
    if (!decl)
        decl = visitor->getParentFuncOfVisitor();
    if (referencedNode && decl)
    {
        // Here we store a childDecl that added/removed capabilities from a parentDecl
        decl->capabilityRequirementProvenance.add(
            ProvenenceNodeWithLoc{referencedNode, referenceLoc});
    }
};

CapabilitySet getStatementCapabilityUsage(SemanticsVisitor* visitor, Stmt* stmt);

template<typename ProcessFunc, typename ParentDiagnosticFunc>
struct CapabilityDeclReferenceVisitor
    : public SemanticsDeclReferenceVisitor<
          CapabilityDeclReferenceVisitor<ProcessFunc, ParentDiagnosticFunc>>
{
    typedef SemanticsDeclReferenceVisitor<
        CapabilityDeclReferenceVisitor<ProcessFunc, ParentDiagnosticFunc>>
        Base;

    const ProcessFunc handleProcessFunc;
    const ParentDiagnosticFunc handleParentDiagnosticFunc;
    RequireCapabilityAttribute* maybeRequireCapability;
    SemanticsContext& outerContext;
    CapabilityDeclReferenceVisitor(
        const ProcessFunc& processFunc,
        const ParentDiagnosticFunc& parentDiagnosticFunc,
        RequireCapabilityAttribute* maybeRequireCapability,
        SemanticsContext& outer)
        : handleProcessFunc(processFunc)
        , handleParentDiagnosticFunc(parentDiagnosticFunc)
        , maybeRequireCapability(maybeRequireCapability)
        , outerContext(outer)
        , SemanticsDeclReferenceVisitor<
              CapabilityDeclReferenceVisitor<ProcessFunc, ParentDiagnosticFunc>>(outer)
    {
    }
    virtual void processReferencedDecl(Decl* decl) override
    {
        SourceLoc loc = SourceLoc();
        if (Base::sourceLocStack.getCount())
            loc = Base::sourceLocStack.getLast();
        handleProcessFunc(decl, decl->inferredCapabilityRequirements, loc);
    }
    virtual void processDeclModifiers(Decl* decl, SourceLoc refLoc) override
    {
        if (decl)
            handleProcessFunc(decl, decl->inferredCapabilityRequirements, refLoc);
    }
    void visitDiscardStmt(DiscardStmt* stmt)
    {
        handleProcessFunc(stmt, CapabilitySet(CapabilityName::fragment), stmt->loc);
    }
    void visitTargetSwitchStmt(TargetSwitchStmt* stmt)
    {
        CapabilitySet set;
        auto targetCaseCount = stmt->targetCases.getCount();
        for (Index targetCaseIndex = 0; targetCaseIndex < targetCaseCount; targetCaseIndex++)
        {
            // We may recieve a `default:` case for a `__target_switch`. If this is the case,
            // we must resolve the target capability for a non empty set of
            // `calling_functions_targets`:
            //      ``` default_target = calling_functions_targets-{other_case_targets} ```
            //
            // * `calling_functions_capability` = `requirement attribute` of the calling
            // function; if missing
            //    we can assume it is `any_target`
            //
            // * `{other_case_targets}` = set of all capabilities all `case` statments target
            // inside the `__target_switch`

            // If we do not handle `default:`, the codegen will fail when trying to find a
            // specific codegen target not handled explicitly by a `case` statment. We must also
            // ensure the `default` case is last so we have priority to hit `case` statments and
            // can preprocess `case` statments before the `default` case.
            CapabilitySet targetCap;
            if (CapabilityName(stmt->targetCases[targetCaseIndex]->capability) ==
                CapabilityName::Invalid)
            {
                if (targetCaseCount - 1 != targetCaseIndex)
                {
                    for (Index i = targetCaseIndex; i < targetCaseCount - 1; i++)
                        std::swap(stmt->targetCases[i], stmt->targetCases[i + 1]);
                    continue;
                }
                if (as<StageSwitchStmt>(stmt))
                {
                    if (!maybeRequireCapability)
                        targetCap = (CapabilitySet(CapabilityName::any_target)
                                         .getStagesThisHasButOtherDoesNot(set));
                    else
                        targetCap =
                            (maybeRequireCapability->capabilitySet.getStagesThisHasButOtherDoesNot(
                                set));
                }
                else
                {
                    if (!maybeRequireCapability)
                        targetCap = (CapabilitySet(CapabilityName::any_target)
                                         .getTargetsThisHasButOtherDoesNot(set));
                    else
                        targetCap =
                            (maybeRequireCapability->capabilitySet.getTargetsThisHasButOtherDoesNot(
                                set));
                }
            }
            else
            {
                targetCap =
                    CapabilitySet(CapabilityName(stmt->targetCases[targetCaseIndex]->capability));

                if (maybeRequireCapability)
                {
                    CapabilitySet testingForInvalid = maybeRequireCapability->capabilitySet;
                    // Ensure case statement is valid with parent `[require(...)]`
                    testingForInvalid.join(targetCap);
                    if (testingForInvalid.isInvalid())
                    {
                        maybeDiagnose(
                            Base::getSink(),
                            outerContext.getOptionSet(),
                            DiagnosticCategory::Capability,
                            stmt->targetCases[targetCaseIndex]->loc,
                            Diagnostics::conflictingCapabilityDueToStatement,
                            targetCap,
                            maybeRequireCapability,
                            maybeRequireCapability->capabilitySet);
                        handleParentDiagnosticFunc(DiagnosticCategory::Capability);
                    }
                }
            }
            auto targetCase = stmt->targetCases[targetCaseIndex];
            auto oldCap = targetCap;
            auto bodyCap = getStatementCapabilityUsage(this, targetCase->body);
            targetCap.join(bodyCap);
            if (targetCap.isInvalid())
            {
                maybeDiagnose(
                    Base::getSink(),
                    outerContext.getOptionSet(),
                    DiagnosticCategory::Capability,
                    targetCase->body->loc,
                    Diagnostics::conflictingCapabilityDueToStatement,
                    bodyCap,
                    "target_switch",
                    oldCap);
                handleParentDiagnosticFunc(DiagnosticCategory::Capability);
            }
            set.unionWith(targetCap);
        }
        handleProcessFunc(stmt, set, stmt->loc);
    }

    void visitRequireCapabilityDecl(RequireCapabilityDecl* decl)
    {
        handleProcessFunc(decl, decl->inferredCapabilityRequirements, decl->loc);
    }
};

template<typename ProcessFunc, typename ParentDiagnosticFunc>
void visitReferencedDecls(
    SemanticsContext& context,
    NodeBase* node,
    SourceLoc initialLoc,
    RequireCapabilityAttribute* maybeRequireCapability,
    const ProcessFunc& processFunc,
    const ParentDiagnosticFunc& parentDiagnosticFunc)
{
    CapabilityDeclReferenceVisitor<ProcessFunc, ParentDiagnosticFunc> visitor(
        processFunc,
        parentDiagnosticFunc,
        maybeRequireCapability,
        context);
    visitor.sourceLocStack.add(initialLoc);

    if (auto val = as<Val>(node))
        visitor.dispatchIfNotNull(val);
    if (auto stmt = as<Stmt>(node))
        visitor.dispatchIfNotNull(stmt);
    if (auto expr = as<Expr>(node))
        visitor.dispatchIfNotNull(expr);
    if (auto decl = as<Decl>(node))
        visitor.dispatchIfNotNull(decl);
}

CapabilitySet getStatementCapabilityUsage(SemanticsVisitor* visitor, Stmt* stmt)
{
    if (stmt == nullptr)
        return CapabilitySet();

    CapabilitySet inferredRequirements;
    visitReferencedDecls(
        *visitor,
        stmt,
        stmt->loc,
        nullptr,
        [&](SyntaxNode* node, const CapabilitySet& nodeCaps, SourceLoc refLoc)
        { _propagateRequirement(visitor, inferredRequirements, stmt, node, nodeCaps, refLoc); },
        [](DiagnosticCategory category) { SLANG_UNUSED(category); });
    return inferredRequirements;
}

void SemanticsDeclCapabilityVisitor::checkVarDeclCommon(VarDeclBase* varDecl)
{
    visitReferencedDecls(
        *this,
        varDecl->type.type,
        varDecl->loc,
        varDecl->findModifier<RequireCapabilityAttribute>(),
        [this, varDecl](SyntaxNode* node, const CapabilitySet& nodeCaps, SourceLoc refLoc)
        {
            _propagateRequirement(
                this,
                varDecl->inferredCapabilityRequirements,
                varDecl,
                node,
                nodeCaps,
                refLoc);
        },
        [this, varDecl](DiagnosticCategory category)
        { _propagateSeeDefinitionOf(this, varDecl, category); });
}

CapabilitySet SemanticsDeclCapabilityVisitor::getDeclaredCapabilitySet(Decl* decl)
{
    // Merge a decls's declared capability set with all parent declarations.
    // For every existing target, we want to join their requirements together.
    // If the the parent defines additional targets, we want to add them to the disjunction set.
    // For example:
    //    [require(glsl)] struct Parent { [require(glsl, glsl_ext_1)] [require(spirv)] void
    //    foo(); }
    // The requirement for `foo` should be glsl+glsl_ext_1 | spirv.
    //
    CapabilitySet declaredCaps;
    CapabilityAtom stageToJoin = CapabilityAtom::Invalid;
    for (Decl* parent = decl; parent; parent = getParentDecl(parent))
    {
        CapabilitySet localDeclaredCaps;
        bool shouldBreak = false;
        if (!as<AggTypeDeclBase>(parent) || parent->inferredCapabilityRequirements.isEmpty())
        {
            for (auto mod : parent->modifiers)
            {
                if (auto decoration = as<RequireCapabilityAttribute>(mod))
                    localDeclaredCaps.unionWith(decoration->capabilitySet);
                else if (auto entrypoint = as<EntryPointAttribute>(mod))
                    stageToJoin = entrypoint->capabilitySet.getTargetStage();
            }
        }
        else
        {
            localDeclaredCaps = parent->inferredCapabilityRequirements;
            shouldBreak = true;
        }
        // Merge decl's capability declaration with the parent.
        declaredCaps.nonDestructiveJoin(localDeclaredCaps);

        // If the parent already has inferred capability requirements, we should stop now
        // since that already covers transitive parents.
        if (shouldBreak)
            break;
    }
    if (!declaredCaps.isEmpty() && stageToJoin != CapabilityAtom::Invalid)
        declaredCaps.join(CapabilitySet((CapabilityName)stageToJoin));
    return declaredCaps;
}

void SemanticsDeclCapabilityVisitor::visitAggTypeDeclBase(AggTypeDeclBase* decl)
{
    decl->inferredCapabilityRequirements = getDeclaredCapabilitySet(decl);
}

void SemanticsDeclCapabilityVisitor::visitNamespaceDeclBase(NamespaceDeclBase* decl)
{
    decl->inferredCapabilityRequirements = getDeclaredCapabilitySet(decl);
}

template<typename ProcessFunc, typename ParentDiagnosticFunc>
static inline void _dispatchCapabilitiesVisitorOfFunctionDecl(
    SemanticsVisitor* visitor,
    FunctionDeclBase* funcDecl,
    const ProcessFunc& processFunc,
    const ParentDiagnosticFunc& parentDiagnosticFunc)
{
    visitor->setParentFuncOfVisitor(funcDecl);

    for (auto member : funcDecl->members)
    {
        visitor->ensureDecl(member, DeclCheckState::CapabilityChecked);
        _propagateRequirement(
            visitor,
            funcDecl->inferredCapabilityRequirements,
            funcDecl,
            member,
            member->inferredCapabilityRequirements,
            member->loc);
    }

    visitReferencedDecls(
        *visitor,
        funcDecl->body,
        funcDecl->loc,
        funcDecl->findModifier<RequireCapabilityAttribute>(),
        processFunc,
        parentDiagnosticFunc);

    if (!isEffectivelyStatic(funcDecl))
    {
        auto parentAggTypeDecl = getParentAggTypeDecl(funcDecl);
        if (parentAggTypeDecl)
        {
            visitor->ensureDecl(parentAggTypeDecl, DeclCheckState::CapabilityChecked);
            _propagateRequirement(
                visitor,
                funcDecl->inferredCapabilityRequirements,
                funcDecl,
                parentAggTypeDecl,
                parentAggTypeDecl->inferredCapabilityRequirements,
                funcDecl->loc);
        }
    }
}

void SemanticsDeclCapabilityVisitor::visitFunctionDeclBase(FunctionDeclBase* funcDecl)
{
    // If the function is an entrypoint and specifies a target stage, add the capabilities to
    // our function capabilities.
    _dispatchCapabilitiesVisitorOfFunctionDecl(
        this,
        funcDecl,
        [this, funcDecl](SyntaxNode* node, const CapabilitySet& nodeCaps, SourceLoc refLoc)
        {
            _propagateRequirement(
                this,
                funcDecl->inferredCapabilityRequirements,
                funcDecl,
                node,
                nodeCaps,
                refLoc);
        },
        [this, funcDecl](DiagnosticCategory category)
        { _propagateSeeDefinitionOf(this, funcDecl, category); });

    auto declaredCaps = getDeclaredCapabilitySet(funcDecl);

    auto vis = getDeclVisibility(funcDecl);

    // If 0 capabilities were annotated on a function, capabilities are inferred from the
    // function body
    if (declaredCaps.isEmpty())
    {
        declaredCaps = funcDecl->inferredCapabilityRequirements;
    }
    else
    {
        if (vis == DeclVisibility::Public)
        {
            // For public decls, we need to enforce that the function
            // only uses capabilities that it declares.
            // At a minimum we will propagate shader requirements to our
            // function from calling children in all cases so the parent
            // can enforce shader targets correctly and propagate to `main`
            CapabilityAtomSet failedAvailableCapabilityConjunction;
            if (!CapabilitySet::checkCapabilityRequirement(
                    declaredCaps,
                    funcDecl->inferredCapabilityRequirements,
                    failedAvailableCapabilityConjunction))
            {
                diagnoseUndeclaredCapability(
                    funcDecl,
                    Diagnostics::useOfUndeclaredCapability,
                    failedAvailableCapabilityConjunction);
                funcDecl->inferredCapabilityRequirements = declaredCaps;
            }
            else
                funcDecl->inferredCapabilityRequirements.nonDestructiveJoin(declaredCaps);
        }
        else
        {
            // For internal decls, their inferred capability should be joined
            // with the declared capabilities.
            funcDecl->inferredCapabilityRequirements.join(declaredCaps);
        }
    }
}

void SemanticsDeclCapabilityVisitor::visitInheritanceDecl(InheritanceDecl* inheritanceDecl)
{
    // Check that the implementation of an interface requirement is not using more capabilities
    // than what's declared on the interface method.
    if (inheritanceDecl->witnessTable)
    {
        for (auto& kv : inheritanceDecl->witnessTable->m_requirementDictionary)
        {
            if (kv.value.getFlavor() != RequirementWitness::Flavor::declRef)
                continue;
            auto requirementDecl = kv.key;
            auto implDecl = kv.value.getDeclRef();
            if (!implDecl)
                continue;

            if (getModuleDecl(implDecl.getDecl())->isInLegacyLanguage)
                break;

            ensureDecl(requirementDecl, DeclCheckState::CapabilityChecked);
            ensureDecl(implDecl.declRefBase, DeclCheckState::CapabilityChecked);

            CapabilityAtomSet failedAvailableCapabilityConjunction;
            if (!CapabilitySet::checkCapabilityRequirement(
                    requirementDecl->inferredCapabilityRequirements,
                    implDecl.getDecl()->inferredCapabilityRequirements,
                    failedAvailableCapabilityConjunction))
            {
                diagnoseUndeclaredCapability(
                    implDecl.getDecl(),
                    Diagnostics::useOfUndeclaredCapabilityOfInterfaceRequirement,
                    failedAvailableCapabilityConjunction);
            }
        }
    }
}

DeclVisibility getDeclVisibility(Decl* decl)
{
    if (as<GenericTypeParamDeclBase>(decl) || as<GenericValueParamDecl>(decl) ||
        as<GenericTypeConstraintDecl>(decl))
    {
        auto genericDecl = as<GenericDecl>(decl->parentDecl);
        if (!genericDecl)
            return DeclVisibility::Default;
        if (genericDecl->inner)
            return getDeclVisibility(genericDecl->inner);
        return DeclVisibility::Default;
    }
    if (auto genericDecl = as<GenericDecl>(decl))
        decl = genericDecl->inner;
    for (; decl; decl = getParentDecl(decl))
    {
        if (as<AccessorDecl>(decl))
            continue;
        if (as<EnumCaseDecl>(decl))
            continue;
        break;
    }
    if (!decl)
        return DeclVisibility::Public;

    for (auto modifier : decl->modifiers)
    {
        if (as<PublicModifier>(modifier))
            return DeclVisibility::Public;
        else if (as<InternalModifier>(modifier))
            return DeclVisibility::Internal;
        else if (as<PrivateModifier>(modifier))
            return DeclVisibility::Private;
    }
    // Interface members will always have the same visibility as the interface itself.
    if (auto interfaceDecl = findParentInterfaceDecl(decl))
    {
        return getDeclVisibility(interfaceDecl);
    }
    auto defaultVis = DeclVisibility::Default;
    if (auto parentModule = getModuleDecl(decl))
    {
        defaultVis = parentModule->isInLegacyLanguage ? DeclVisibility::Public
                                                      : parentModule->defaultVisibility;
    }

    // Members of other agg type decls will have their default visibility capped to the
    // parents'.
    if (as<NamespaceDecl>(decl))
    {
        return DeclVisibility::Public;
    }
    return defaultVis;
}

VarDeclBase* getTrailingUnsizedArrayElement(
    Type* type,
    VarDeclBase* parentVar,
    ArrayExpressionType*& outArrayType)
{
    while (auto modifiedType = as<ModifiedType>(type))
        type = modifiedType->getBase();
    HashSet<Type*> seenTypes;
    for (;;)
    {
        if (auto arrayType = as<ArrayExpressionType>(type))
        {
            if (arrayType->isUnsized())
            {
                outArrayType = arrayType;
                return parentVar;
            }
            else
                return nullptr;
        }
        else if (auto declRefType = as<DeclRefType>(type))
        {
            if (auto aggTypeDecl = declRefType->getDeclRef().as<AggTypeDecl>())
            {
                auto varDecls = aggTypeDecl.getDecl()->getMembersOfType<VarDeclBase>();
                if (varDecls.getCount() == 0)
                    return nullptr;
                VarDeclBase* lastVarDecl = nullptr;
                for (auto varDecl : varDecls)
                {
                    if (isEffectivelyStatic(varDecl))
                        continue;
                    lastVarDecl = varDecl;
                }
                auto lastMember =
                    _getMemberDeclRef(getCurrentASTBuilder(), aggTypeDecl, lastVarDecl)
                        .as<VarDeclBase>();
                auto varType = getType(getCurrentASTBuilder(), lastMember);
                if (!varType)
                    return nullptr;
                if (!seenTypes.add(type))
                    return nullptr;
                type = varType;
                parentVar = lastMember.getDecl();
                continue;
            }
        }
    }
    return nullptr;
}

bool isOpaqueHandleType(Type* type)
{
    while (auto modifiedType = as<ModifiedType>(type))
        type = modifiedType->getBase();
    if (as<ResourceType>(type))
        return true;
    if (as<SamplerStateType>(type))
        return true;
    if (as<UniformParameterGroupType>(type))
        return true;
    if (as<HLSLStructuredBufferTypeBase>(type))
        return true;
    if (as<UntypedBufferResourceType>(type))
        return true;
    if (as<GLSLShaderStorageBufferType>(type))
        return true;
    if (as<FeedbackType>(type))
        return true;
    if (as<HLSLPatchType>(type))
        return true;
    if (as<HLSLStreamOutputType>(type))
        return true;
    if (as<MeshOutputType>(type))
        return true;
    return false;
}

void diagnoseMissingCapabilityProvenance(
    CompilerOptionSet& optionSet,
    DiagnosticSink* sink,
    Decl* decl,
    CapabilitySet& setToFind)
{
    HashSet<NodeBase*> checkedDecls;
    ProvenenceNodeWithLoc provNode;
    provNode.referencedNode = decl;
    provNode.referenceLoc = (decl) ? decl->loc : SourceLoc();
    bool bottomOfProvenanceStack = false;
    // Find the bottom of the atom provenance stack which fails to contain `setToFind`
    while (!bottomOfProvenanceStack && provNode.referencedNode)
    {
        bottomOfProvenanceStack = true;
        if (auto referencedDecl = as<Decl>(provNode.referencedNode))
        {
            for (auto& i : referencedDecl->capabilityRequirementProvenance)
            {
                if (checkedDecls.contains(i.referencedNode))
                    continue;
                checkedDecls.add(i.referencedNode);
                auto innerReferencedDecl = as<Decl>(i.referencedNode);
                if (!innerReferencedDecl ||
                    !innerReferencedDecl->inferredCapabilityRequirements.implies(setToFind))
                {
                    // We found a source of the incompatible capability, follow this
                    // element inside the provenance stack until we are at the bottom
                    provNode = i;
                    bottomOfProvenanceStack = false;
                    break;
                }
            }
        }
        else
        {
            bottomOfProvenanceStack = true;
        }
    }

    if (!provNode.referencedNode)
        return;

    if (auto referencedDecl = as<Decl>(provNode.referencedNode))
    {
        // Diagnose the use-site
        maybeDiagnose(
            sink,
            optionSet,
            DiagnosticCategory::Capability,
            provNode.referenceLoc,
            Diagnostics::seeUsingOf,
            referencedDecl);
        // Diagnose the definition as the problem
        maybeDiagnose(
            sink,
            optionSet,
            DiagnosticCategory::Capability,
            referencedDecl->loc,
            Diagnostics::seeDefinitionOf,
            referencedDecl);
        // If we find a 'require' modifier, this is contributing to the overall capability
        // incompatibility. We should hint to the user that this declaration is problematic.
        if (auto requireCapabilityAttribute =
                referencedDecl->findModifier<RequireCapabilityAttribute>())
            maybeDiagnose(
                sink,
                optionSet,
                DiagnosticCategory::Capability,
                requireCapabilityAttribute->loc,
                Diagnostics::seeDeclarationOf,
                requireCapabilityAttribute);
    }
    else
    {
        maybeDiagnose(
            sink,
            optionSet,
            DiagnosticCategory::Capability,
            provNode.referenceLoc,
            Diagnostics::seeUsingOf,
            provNode.referencedNode->astNodeType);
    }
}

void diagnoseCapabilityProvenance(
    CompilerOptionSet& optionSet,
    DiagnosticSink* sink,
    Decl* decl,
    CapabilityAtom atomToFind,
    HashSet<Decl*>& printedDecls)
{
    auto thisModule = getModuleDecl(decl);
    Decl* declToPrint = decl;
    while (declToPrint)
    {
        Decl* previousDecl = declToPrint;
        printedDecls.add(declToPrint);
        for (auto& provenance : declToPrint->capabilityRequirementProvenance)
        {
            auto referencedDecl = as<Decl>(provenance.referencedNode);
            if (!referencedDecl)
            {
                maybeDiagnose(
                    sink,
                    optionSet,
                    DiagnosticCategory::Capability,
                    provenance.referenceLoc,
                    Diagnostics::seeUsingOf,
                    provenance.referencedNode->astNodeType);
                break;
            }

            if (!referencedDecl->inferredCapabilityRequirements.implies(atomToFind))
                continue;
            maybeDiagnose(
                sink,
                optionSet,
                DiagnosticCategory::Capability,
                provenance.referenceLoc,
                Diagnostics::seeUsingOf,
                referencedDecl);
            declToPrint = referencedDecl;
            if (printedDecls.contains(declToPrint))
                break;
            if (declToPrint->findModifier<RequireCapabilityAttribute>())
                break;
            auto moduleDecl = getModuleDecl(declToPrint);
            if (thisModule != moduleDecl)
                break;
        }
        if (previousDecl == declToPrint)
            break;
    }
    if (declToPrint)
    {
        maybeDiagnose(
            sink,
            optionSet,
            DiagnosticCategory::Capability,
            declToPrint->loc,
            Diagnostics::seeDefinitionOf,
            declToPrint);
    }
}

void SemanticsDeclCapabilityVisitor::diagnoseUndeclaredCapability(
    Decl* decl,
    const DiagnosticInfo& diagnosticInfo,
    const CapabilityAtomSet& failedAtomsInsideAvailableSet)
{
    if (decl->inferredCapabilityRequirements.isEmpty())
        return;
    if (failedAtomsInsideAvailableSet.isEmpty() ||
        failedAtomsInsideAvailableSet.contains((UInt)CapabilityAtom::Invalid))
        return;

    // There are two causes for why type checking failed on failedAvailableSet.
    // The first scenario is that failedAvailableSet defines a set of capabilities on a
    // compilation target (e.g. hlsl) that isn't defined by some callees, for example, if we
    // have a function:
    //    [require(hlsl)]  // <-- failedAvailableSet
    //    [require(cpp)]
    //    void caller()
    //    {
    //        printf(); // assume this is defined for (cpp | cuda).
    //    }
    // In this case we should diagnose error reporting printf isn't defined on a required
    // target.
    //
    // Now, we detect if we are case 1.

    {
        CapabilityAtom outFailedAtom{};
        if (hasTargetAtom(failedAtomsInsideAvailableSet, outFailedAtom))
        {
            maybeDiagnose(
                getSink(),
                this->getOptionSet(),
                DiagnosticCategory::Capability,
                decl->loc,
                Diagnostics::declHasDependenciesNotCompatibleOnTarget,
                decl,
                outFailedAtom);

            // Anything defined on a non-failed target atom may be the culprit to why we fail
            // having a target capability. Print out all possible culprits.
            CapabilityAtomSet failedAtomSet;
            failedAtomSet.add((UInt)outFailedAtom);
            CapabilityAtomSet targetsNotUsedSet;
            CapabilityAtomSet::calcSubtract(
                targetsNotUsedSet,
                getAtomSetOfTargets(),
                failedAtomSet);

            HashSet<Decl*> printedDecls;
            for (auto atom : targetsNotUsedSet)
            {
                CapabilityAtom formattedAtom = asAtom(atom);
                diagnoseCapabilityProvenance(
                    this->getOptionSet(),
                    getSink(),
                    decl,
                    formattedAtom,
                    printedDecls);
            }
            return;
        }
    }

    //// The second scenario is when the callee is using a capability that is not provided by
    /// the
    /// requirement. / For example: /     [require(hlsl,b,c)] /     void caller() /     { /
    /// useD();
    ///// require capability (hlsl,d) /     } / In this case we should report that useD() is
    /// using a
    /// capability that is not declared by caller.
    ////

    //// If we reach here, we are case 2.

    // We will produce all failed atoms. This is important since provenance of multiple atoms
    // can come from multiple referenced items in a function body.
    HashSet<Decl*> printedDecls;
    auto simplifiedFailedAtomsSet = failedAtomsInsideAvailableSet.newSetWithoutImpliedAtoms();
    for (auto i : simplifiedFailedAtomsSet)
    {
        CapabilityAtom formattedAtom = asAtom(i);
        CapabilityName canonicalName;
        if (isStageAtom((CapabilityName)formattedAtom, canonicalName))
        {
            // Provide a more friendly message if atom is a stage.
            maybeDiagnose(
                getSink(),
                this->getOptionSet(),
                DiagnosticCategory::Capability,
                decl->loc,
                Diagnostics::declHasDependenciesNotCompatibleOnStage,
                decl,
                formattedAtom);
        }
        else
        {
            maybeDiagnose(
                getSink(),
                this->getOptionSet(),
                DiagnosticCategory::Capability,
                decl->loc,
                diagnosticInfo,
                decl,
                formattedAtom);
        }
        // Print provenances.
        diagnoseCapabilityProvenance(
            this->getOptionSet(),
            getSink(),
            decl,
            formattedAtom,
            printedDecls);
    }
}

} // namespace Slang
