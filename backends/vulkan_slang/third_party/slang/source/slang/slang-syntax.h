#ifndef SLANG_SYNTAX_H
#define SLANG_SYNTAX_H

#include "slang-ast-builder.h"

namespace Slang
{

inline Type* getSub(ASTBuilder* astBuilder, DeclRef<GenericTypeConstraintDecl> const& declRef)
{
    return declRef.substitute(astBuilder, declRef.getDecl()->sub.Ptr());
}

inline Type* getSup(ASTBuilder* astBuilder, DeclRef<TypeConstraintDecl> const& declRef)
{
    return declRef.substitute(astBuilder, declRef.getDecl()->getSup().type);
}


// `Val`

inline bool areValsEqual(Val* left, Val* right)
{
    if (!left || !right)
        return left == right;
    return left->equals(right);
}

//

inline BaseType getVectorBaseType(VectorExpressionType* vecType)
{
    auto basicExprType = as<BasicExpressionType>(vecType->getElementType());
    return basicExprType->getBaseType();
}

inline int getVectorSize(VectorExpressionType* vecType)
{
    auto constantVal = as<ConstantIntVal>(vecType->getElementCount());
    if (constantVal)
        return (int)constantVal->getValue();
    // TODO: what to do in this case?
    return 0;
}

//
// Declarations
//

struct SemanticsVisitor;

List<ExtensionDecl*> const& getCandidateExtensions(
    DeclRef<AggTypeDecl> const& declRef,
    SemanticsVisitor* semantics);

// Returns the members of `genericInnerDecl`'s enclosing generic decl.
inline FilteredMemberRefList<Decl> getGenericMembers(
    ASTBuilder* astBuilder,
    DeclRef<Decl> genericInnerDecl,
    MemberFilterStyle filterStyle = MemberFilterStyle::All)
{
    return FilteredMemberRefList<Decl>(
        astBuilder,
        genericInnerDecl.getParent().getDecl()->members,
        genericInnerDecl,
        filterStyle);
}

inline FilteredMemberRefList<Decl> getMembers(
    ASTBuilder* astBuilder,
    DeclRef<ContainerDecl> declRef,
    MemberFilterStyle filterStyle = MemberFilterStyle::All)
{
    return FilteredMemberRefList<Decl>(
        astBuilder,
        declRef.getDecl()->members,
        declRef,
        filterStyle);
}

template<typename T>
inline FilteredMemberRefList<T> getMembersOfType(
    ASTBuilder* astBuilder,
    DeclRef<ContainerDecl> declRef,
    MemberFilterStyle filterStyle = MemberFilterStyle::All)
{
    return FilteredMemberRefList<T>(astBuilder, declRef.getDecl()->members, declRef, filterStyle);
}

void _foreachDirectOrExtensionMemberOfType(
    SemanticsVisitor* semantics,
    DeclRef<ContainerDecl> const& declRef,
    SyntaxClassBase const& syntaxClass,
    void (*callback)(DeclRefBase*, void*),
    void const* userData);

DeclRef<Decl> _getMemberDeclRef(ASTBuilder* builder, DeclRef<Decl> parent, Decl* decl);

template<typename T, typename F>
inline void foreachDirectOrExtensionMemberOfType(
    SemanticsVisitor* semantics,
    DeclRef<ContainerDecl> const& declRef,
    F const& func)
{
    struct Helper
    {
        const F* userFunc;
        SemanticsVisitor* semanticsVisitor;
        static void callback(DeclRefBase* declRef, void* userData)
        {
            (*((*(Helper*)userData).userFunc))(DeclRef<T>(declRef));
        }
    };
    Helper helper;
    helper.userFunc = &func;
    helper.semanticsVisitor = semantics;
    _foreachDirectOrExtensionMemberOfType(
        semantics,
        declRef,
        getSyntaxClass<T>(),
        &Helper::callback,
        &helper);
}

/// The the user-level name for a variable that might be a shader parameter.
///
/// In most cases this is just the name of the variable declaration itself,
/// but in the specific case of a `cbuffer`, the name that the user thinks
/// of is really metadata. For example:
///
///     cbuffer C { int x; }
///
/// In this example, error messages relating to the constant buffer should
/// really use the name `C`, but that isn't the name of the declaration
/// (it is in practice anonymous, and `C` can be used for a different
/// declaration in the same file).
///
Name* getReflectionName(VarDeclBase* varDecl);

inline Type* getType(ASTBuilder* astBuilder, DeclRef<VarDeclBase> declRef)
{
    return declRef.substitute(astBuilder, declRef.getDecl()->type.Ptr());
}

/// same as getType, but take into account the additional type modifiers from the parameter's
/// modifier list and return a ModifiedType if such modifiers exist.
Type* getParamType(ASTBuilder* astBuilder, DeclRef<VarDeclBase> paramDeclRef);

/// Get the parameter type, wrapped with `Out<>`, `InOut<>` or `Ref<>` if the parameter has
/// an non-trivial direction.
Type* getParamTypeWithDirectionWrapper(ASTBuilder* astBuilder, DeclRef<VarDeclBase> paramDeclRef);


inline SubstExpr<Expr> getInitExpr(ASTBuilder* astBuilder, DeclRef<VarDeclBase> declRef)
{
    return declRef.substitute(astBuilder, declRef.getDecl()->initExpr);
}

inline Type* getType(ASTBuilder* astBuilder, DeclRef<PropertyDecl> declRef)
{
    return declRef.substitute(astBuilder, declRef.getDecl()->type.Ptr());
}

inline Type* getType(ASTBuilder* astBuilder, DeclRef<SubscriptDecl> declRef)
{
    return declRef.substitute(astBuilder, declRef.getDecl()->returnType.Ptr());
}

inline Type* getType(ASTBuilder* astBuilder, DeclRef<EnumCaseDecl> declRef)
{
    return declRef.substitute(astBuilder, declRef.getDecl()->type.Ptr());
}

inline SubstExpr<Expr> getTagExpr(ASTBuilder* astBuilder, DeclRef<EnumCaseDecl> declRef)
{
    return declRef.substitute(astBuilder, declRef.getDecl()->tagExpr);
}

inline Type* getTargetType(ASTBuilder* astBuilder, DeclRef<ExtensionDecl> declRef)
{
    return declRef.substitute(astBuilder, declRef.getDecl()->targetType.Ptr());
}

inline FilteredMemberRefList<VarDecl> getFields(
    ASTBuilder* astBuilder,
    DeclRef<StructDecl> declRef,
    MemberFilterStyle filterStyle)
{
    return getMembersOfType<VarDecl>(astBuilder, declRef, filterStyle);
}

/// If the given `structTypeDeclRef` inherits from another struct type, return that base type
DeclRefType* findBaseStructType(ASTBuilder* astBuilder, DeclRef<StructDecl> structTypeDeclRef);

/// If the given `structTypeDeclRef` inherits from another struct type, return that base struct decl
DeclRef<StructDecl> findBaseStructDeclRef(
    ASTBuilder* astBuilder,
    DeclRef<StructDecl> structTypeDeclRef);

SubtypeWitness* findThisTypeWitness(SubstitutionSet substs, InterfaceDecl* interfaceDecl);

RequirementWitness tryLookUpRequirementWitness(
    ASTBuilder* astBuilder,
    SubtypeWitness* subtypeWitness,
    Decl* requirementKey);

DeclRef<Decl> createDefaultSubstitutionsIfNeeded(
    ASTBuilder* astBuilder,
    SemanticsVisitor* semantics,
    DeclRef<Decl> declRef);

List<Val*> getDefaultSubstitutionArgs(
    ASTBuilder* astBuilder,
    SemanticsVisitor* semantics,
    GenericDecl* genericDecl);

SubstitutionSet makeSubstitutionFromIncompleteSet(
    ASTBuilder* astBuilder,
    SemanticsVisitor* semantics,
    DeclRef<GenericDecl> genericDeclRef,
    Dictionary<Decl*, Val*> paramArgMap,
    DiagnosticSink* sink);

Val::OperandView<Val> findInnerMostGenericArgs(SubstitutionSet subst);

ParameterDirection getParameterDirection(VarDeclBase* varDecl);

inline Type* getTagType(ASTBuilder* astBuilder, DeclRef<EnumDecl> declRef)
{
    return declRef.substitute(astBuilder, declRef.getDecl()->tagType);
}

inline Type* getBaseType(ASTBuilder* astBuilder, DeclRef<InheritanceDecl> declRef)
{
    return declRef.substitute(astBuilder, declRef.getDecl()->base.type);
}

inline Type* getType(ASTBuilder* astBuilder, DeclRef<TypeDefDecl> declRef)
{
    return declRef.substitute(astBuilder, declRef.getDecl()->type.Ptr());
}

inline Type* getResultType(ASTBuilder* astBuilder, DeclRef<CallableDecl> declRef)
{
    return declRef.substitute(astBuilder, declRef.getDecl()->returnType.type);
}

inline Type* getErrorCodeType(ASTBuilder* astBuilder, DeclRef<CallableDecl> declRef)
{
    if (declRef.getDecl()->errorType.type)
    {
        return declRef.substitute(astBuilder, declRef.getDecl()->errorType.type);
    }
    else
    {
        return astBuilder->getBottomType();
    }
}

inline FilteredMemberRefList<ParamDecl> getParameters(
    ASTBuilder* astBuilder,
    DeclRef<CallableDecl> declRef)
{
    return getMembersOfType<ParamDecl>(astBuilder, declRef);
}

inline Decl* getInner(DeclRef<GenericDecl> declRef)
{
    return declRef.getDecl()->inner;
}

//

inline Type* getType(ASTBuilder* astBuilder, SubstExpr<Expr> expr)
{
    if (!expr)
        return astBuilder->getErrorType();
    return substituteType(expr.getSubsts(), astBuilder, expr.getExpr()->type);
}

inline SubstExpr<Expr> getBaseExpr(SubstExpr<ParenExpr> expr)
{
    return substituteExpr(expr.getSubsts(), expr.getExpr()->base);
}

inline SubstExpr<Expr> getBaseExpr(SubstExpr<BuiltinCastExpr> expr)
{
    return substituteExpr(expr.getSubsts(), expr.getExpr()->base);
}

inline SubstExpr<Expr> getBaseExpr(SubstExpr<InvokeExpr> expr)
{
    return substituteExpr(expr.getSubsts(), expr.getExpr()->functionExpr);
}

inline Index getArgCount(SubstExpr<InvokeExpr> expr)
{
    return expr.getExpr()->arguments.getCount();
}

inline SubstExpr<Expr> getArg(SubstExpr<InvokeExpr> expr, Index index)
{
    return substituteExpr(expr.getSubsts(), expr.getExpr()->arguments[index]);
}

inline DeclRef<Decl> getDeclRef(ASTBuilder* astBuilder, SubstExpr<DeclRefExpr> expr)
{
    return substituteDeclRef(expr.getSubsts(), astBuilder, expr.getExpr()->declRef);
}

//

ArrayExpressionType* getArrayType(ASTBuilder* astBuilder, Type* elementType, IntVal* elementCount);

ArrayExpressionType* getArrayType(ASTBuilder* astBuilder, Type* elementType);

NamedExpressionType* getNamedType(ASTBuilder* astBuilder, DeclRef<TypeDefDecl> const& declRef);

FuncType* getFuncType(ASTBuilder* astBuilder, DeclRef<CallableDecl> const& declRef);

GenericDeclRefType* getGenericDeclRefType(
    ASTBuilder* astBuilder,
    DeclRef<GenericDecl> const& declRef);

NamespaceType* getNamespaceType(ASTBuilder* astBuilder, DeclRef<NamespaceDeclBase> const& declRef);

SamplerStateType* getSamplerStateType(ASTBuilder* astBuilder);


// Definitions that can't come earlier despite
// being in templates, because gcc/clang get angry.
//
template<typename T>
Modifier* FilteredModifierList<T>::adjust(Modifier* modifier)
{
    Modifier* m = modifier;
    for (;;)
    {
        if (!m)
            return m;
        if (as<T>(m))
        {
            return m;
        }
        m = m->next;
    }
}

template<typename T>
void FilteredModifierList<T>::Iterator::operator++()
{
    current = FilteredModifierList<T>::adjust(current->next);
}
//

enum class UserDefinedAttributeTargets
{
    None = 0,
    Struct = 1,
    Var = 2,
    Function = 4,
    Param = 8,
    All = 0x0F
};

const int kUnsizedArrayMagicLength = 0x7FFFFFFF;

/// Get the module dclaration that a declaration is associated with, if any.
ModuleDecl* getModuleDecl(Decl* decl);
ModuleDecl* getModuleDecl(Scope* scope);

/// Get the module that a declaration is associated with, if any.
Module* getModule(Decl* decl);

/// Get the parent decl, skipping any generic decls in between.
Decl* getParentDecl(Decl* decl);
Decl* getParentAggTypeDecl(Decl* decl);
Decl* getParentAggTypeDeclBase(Decl* decl);
Decl* getParentFunc(Decl* decl);

} // namespace Slang

#endif
