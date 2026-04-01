// slang-ast-val.h
#pragma once

#include "slang-ast-base.h"
#include "slang-ast-decl.h"
#include "slang-ast-val.h.fiddle"

FIDDLE()
namespace Slang
{

// Syntax class definitions for compile-time values.

FIDDLE()
class DirectDeclRef : public DeclRefBase
{
    FIDDLE(...)
public:
    DirectDeclRef(Decl* decl) { setOperands(decl); }

    DeclRefBase* _substituteImplOverride(
        ASTBuilder* astBuilder,
        SubstitutionSet subst,
        int* ioDiff);
    void _toTextOverride(StringBuilder& out);
    Val* _resolveImplOverride();
    DeclRefBase* _getBaseOverride();
};

// Represent an static member of a base decl.
// Note that we automatically fold the DeclRef if the path is known to be static.
// For example, MemberDeclRef(DirectDeclRef(A), B) ==> DirectDeclRef(B),
// and MemberDeclRef(MemberDeclRef(A, B), C) ==> MemberDeclRef(A, C).
//
FIDDLE()
class MemberDeclRef : public DeclRefBase
{
    FIDDLE(...)
public:
    DeclRefBase* getParentOperand() { return as<DeclRefBase>(getOperand(1)); }

    MemberDeclRef(Decl* decl, DeclRefBase* parent) { setOperands(decl, parent); }

    DeclRefBase* _substituteImplOverride(
        ASTBuilder* astBuilder,
        SubstitutionSet subst,
        int* ioDiff);

    void _toTextOverride(StringBuilder& out);

    Val* _resolveImplOverride();

    DeclRefBase* _getBaseOverride();
};


// Represent a lookup of SuperType::`m_decl` from `lookupSourceType` type that we know conforms to
// SuperType.
FIDDLE()
class LookupDeclRef : public DeclRefBase
{
    FIDDLE(...)
public:
    // m_decl represents the decl in SuperType that we want to lookup.

    // The source type that we are looking up from.
    Type* getLookupSource() { return as<Type>(getOperand(1)); }

    // Witness that `lookupSourceType`:SuperType.
    SubtypeWitness* getWitness() { return as<SubtypeWitness>(getOperand(2)); }

    LookupDeclRef(Decl* declToLookup, Type* lookupSource, SubtypeWitness* witness)
    {
        setOperands(declToLookup, lookupSource, witness);
    }

    Decl* getSupDecl();

    DeclRefBase* _substituteImplOverride(
        ASTBuilder* astBuilder,
        SubstitutionSet subst,
        int* ioDiff);

    void _toTextOverride(StringBuilder& out);

    Val* _resolveImplOverride();

    DeclRefBase* _getBaseOverride();

private:
    Val* tryResolve(SubtypeWitness* newWitness, Type* newLookupSource);
};

// Represents a specialization of a generic decl.
FIDDLE()
class GenericAppDeclRef : public DeclRefBase
{
    FIDDLE(...)
public:
    DeclRefBase* getGenericDeclRef() { return as<DeclRefBase>(getOperand(1)); }
    Index getArgCount() { return getOperandCount() - 2; }
    Val* getArg(Index index) { return getOperand(index + 2); }

    GenericAppDeclRef(Decl* innerDecl, DeclRefBase* genericDeclRef, OperandView<Val> args)
    {
        m_operands.add(ValNodeOperand(innerDecl));
        m_operands.add(ValNodeOperand(genericDeclRef));
        for (auto arg : args)
        {
            m_operands.add(ValNodeOperand(arg));
        }
    }

    GenericAppDeclRef(Decl* innerDecl, DeclRefBase* genericDeclRef, ConstArrayView<Val*> args)
    {
        m_operands.add(ValNodeOperand(innerDecl));
        m_operands.add(ValNodeOperand(genericDeclRef));
        for (auto arg : args)
        {
            m_operands.add(ValNodeOperand(arg));
        }
    }

    GenericDecl* getGenericDecl();

    OperandView<Val> getArgs() { return OperandView<Val>(this, 2, getArgCount()); }

    DeclRefBase* _substituteImplOverride(
        ASTBuilder* astBuilder,
        SubstitutionSet subst,
        int* ioDiff);

    void _toTextOverride(StringBuilder& out);

    Val* _resolveImplOverride();

    DeclRefBase* _getBaseOverride();
};

// A compile-time integer (may not have a specific concrete value)
FIDDLE(abstract)
class IntVal : public Val
{
    FIDDLE(...)
    Type* getType() { return as<Type>(getOperand(0)); }

    Val* _resolveImplOverride() { return this; }

    bool isLinkTimeVal();
    bool _isLinkTimeValOverride() { return false; }
    Val* linkTimeResolve(Dictionary<String, IntVal*>& mapMangledNameToVal);
    Val* _linkTimeResolveOverride(Dictionary<String, IntVal*>&) { return this; }
};

// Trivial case of a value that is just a constant integer
FIDDLE()
class ConstantIntVal : public IntVal
{
    FIDDLE(...)
    IntegerLiteralValue getValue() { return getIntConstOperand(1); }

    // Overrides should be public so base classes can access
    void _toTextOverride(StringBuilder& out);

    ConstantIntVal(Type* inType, IntegerLiteralValue inValue) { setOperands(inType, inValue); }
    bool _isLinkTimeValOverride() { return false; }
};

// The logical "value" of a reference to a generic value parameter
FIDDLE()
class GenericParamIntVal : public IntVal
{
    FIDDLE(...)
    DeclRef<VarDeclBase> getDeclRef() { return as<DeclRefBase>(getOperand(1)); }

    // Overrides should be public so base classes can access
    void _toTextOverride(StringBuilder& out);
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);

    GenericParamIntVal(Type* inType, DeclRef<VarDeclBase> inDeclRef)
    {
        setOperands(inType, inDeclRef);
    }

    bool _isLinkTimeValOverride();
    Val* _linkTimeResolveOverride(Dictionary<String, IntVal*>& map);
};

FIDDLE()
class TypeCastIntVal : public IntVal
{
    FIDDLE(...)
    void _toTextOverride(StringBuilder& out);
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
    Val* _resolveImplOverride();

    Val* getBase() { return getOperand(1); }
    TypeCastIntVal(Type* inType, Val* inBase) { setOperands(inType, inBase); }

    static Val* tryFoldImpl(
        ASTBuilder* astBuilder,
        Type* resultType,
        Val* base,
        DiagnosticSink* sink);

    bool _isLinkTimeValOverride()
    {
        if (auto intBase = as<IntVal>(getBase()))
            return intBase->isLinkTimeVal();
        return false;
    }

    Val* _linkTimeResolveOverride(Dictionary<String, IntVal*>& map);
};

// An compile time int val as result of some general computation.
FIDDLE()
class FuncCallIntVal : public IntVal
{
    FIDDLE(...)
    void _toTextOverride(StringBuilder& out);
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
    Val* _resolveImplOverride();

    DeclRef<Decl> getFuncDeclRef() { return as<DeclRefBase>(getOperand(1)); }
    Type* getFuncType() { return as<Type>(getOperand(2)); }
    OperandView<IntVal> getArgs() { return OperandView<IntVal>(this, 3, getOperandCount() - 3); }
    Index getArgCount() { return getOperandCount() - 3; }

    FuncCallIntVal(
        Type* inType,
        DeclRef<Decl> inFuncDeclRef,
        Type* inFuncType,
        ArrayView<IntVal*> inArgs)
    {
        setOperands(inType, inFuncDeclRef, inFuncType);
        for (auto arg : inArgs)
            m_operands.add(ValNodeOperand(arg));
    }

    static Val* tryFoldImpl(
        ASTBuilder* astBuilder,
        Type* resultType,
        DeclRef<Decl> newFuncDecl,
        List<IntVal*>& newArgs,
        DiagnosticSink* sink);

    bool _isLinkTimeValOverride()
    {
        for (auto arg : getArgs())
        {
            if (arg->isLinkTimeVal())
                return true;
        }
        return false;
    }

    Val* _linkTimeResolveOverride(Dictionary<String, IntVal*>& map);
};

FIDDLE()
class CountOfIntVal : public IntVal
{
    FIDDLE(...)
    CountOfIntVal(Type* inType, Type* typeArg) { setOperands(inType, typeArg); }

    Val* getTypeArg() { return getOperand(1); }

    void _toTextOverride(StringBuilder& out);
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
    Val* _resolveImplOverride();
    bool _isLinkTimeValOverride() { return false; }

    static Val* tryFoldOrNull(ASTBuilder* astBuilder, Type* intType, Type* newType);

    static Val* tryFold(ASTBuilder* astBuilder, Type* intType, Type* newType);
};

FIDDLE()
class WitnessLookupIntVal : public IntVal
{
    FIDDLE(...)
    void _toTextOverride(StringBuilder& out);
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
    Val* _resolveImplOverride();

    SubtypeWitness* getWitness() { return as<SubtypeWitness>(getOperand(1)); }
    Decl* getKey() { return as<Decl>(getDeclOperand(2)); }

    WitnessLookupIntVal(Type* inType, SubtypeWitness* witness, Decl* key)
    {
        setOperands(inType, witness, key);
    }

    static Val* tryFoldOrNull(ASTBuilder* astBuilder, SubtypeWitness* witness, Decl* key);

    static Val* tryFold(ASTBuilder* astBuilder, SubtypeWitness* witness, Decl* key, Type* type);

    bool _isLinkTimeValOverride() { return false; }
};

// polynomial expression "2*a*b^3 + 1" will be represented as:
// { constantTerm:1, terms: [ { constFactor:2, paramFactors:[{"a", 1}, {"b", 3}] } ] }
FIDDLE()
class PolynomialIntValFactor : public Val
{
    FIDDLE(...)
public:
    IntVal* getParam() const { return as<IntVal>(getOperand(0)); }
    IntegerLiteralValue getPower() const { return getIntConstOperand(1); }

    PolynomialIntValFactor(IntVal* inParam, IntegerLiteralValue inPower)
    {
        setOperands(inParam, inPower);
    }

    Val* _resolveImplOverride();

    // for sorting only.
    bool operator<(const PolynomialIntValFactor& other) const
    {
        if (auto thisGenParam = as<GenericParamIntVal>(getParam()))
        {
            if (auto thatGenParam = as<GenericParamIntVal>(other.getParam()))
            {
                if (thisGenParam->equals(thatGenParam))
                    return getPower() < other.getPower();
                else
                    return thisGenParam->getDeclRef().getDecl() <
                           thatGenParam->getDeclRef().getDecl();
            }
            else
            {
                return true;
            }
        }
        else
        {
            if (const auto thatGenParam = as<GenericParamIntVal>(other.getParam()))
            {
                return false;
            }
            return getParam() == other.getParam() ? getPower() < other.getPower()
                                                  : getParam() < other.getParam();
        }
    }
    // for sorting only.
    bool operator==(const PolynomialIntValFactor& other) const
    {
        if (auto thisGenParam = as<GenericParamIntVal>(getParam()))
        {
            if (auto thatGenParam = as<GenericParamIntVal>(other.getParam()))
            {
                if (thisGenParam->equals(thatGenParam) && getPower() == other.getPower())
                    return true;
            }
            return false;
        }
        return getPower() == other.getPower() && getParam() == other.getParam();
    }
    bool equals(const PolynomialIntValFactor& other) const
    {
        return getPower() == other.getPower() && getParam()->equals(other.getParam());
    }
};

FIDDLE()
class PolynomialIntValTerm : public Val
{
    FIDDLE(...)
public:
    IntegerLiteralValue getConstFactor() const { return getIntConstOperand(0); }
    OperandView<PolynomialIntValFactor> getParamFactors() const
    {
        return OperandView<PolynomialIntValFactor>(this, 1, getOperandCount() - 1);
    }

    Val* _resolveImplOverride();

    PolynomialIntValTerm(
        IntegerLiteralValue inConstFactor,
        ArrayView<PolynomialIntValFactor*> inParamFactors)
    {
        setOperands(inConstFactor);
        addOperands(inParamFactors);
    }

    PolynomialIntValTerm(
        IntegerLiteralValue inConstFactor,
        OperandView<PolynomialIntValFactor> inParamFactors)
    {
        setOperands(inConstFactor);
        addOperands(inParamFactors);
    }

    bool canCombineWith(const PolynomialIntValTerm& other) const
    {
        auto paramFactors = getParamFactors();
        if (paramFactors.getCount() != other.getParamFactors().getCount())
            return false;
        for (Index i = 0; i < getParamFactors().getCount(); i++)
        {
            if (!paramFactors[i]->equals(*other.getParamFactors()[i]))
                return false;
        }
        return true;
    }
    bool operator<(const PolynomialIntValTerm& other) const
    {
        auto constFactor = getConstFactor();
        auto paramFactors = getParamFactors();

        if (constFactor < other.getConstFactor())
            return true;
        else if (constFactor == other.getConstFactor())
        {
            auto otherParamFactors = other.getParamFactors();
            for (Index i = 0; i < paramFactors.getCount(); i++)
            {
                if (i >= otherParamFactors.getCount())
                    return false;
                if (*(paramFactors[i]) < *(otherParamFactors[i]))
                    return true;
                if (*(paramFactors[i]) == *(otherParamFactors[i]))
                {
                }
                else
                {
                    return false;
                }
            }
        }
        return false;
    }

    bool isLinkTimeVal()
    {
        for (auto factor : getParamFactors())
        {
            if (factor->getParam()->isLinkTimeVal())
                return true;
        }
        return false;
    }
};

FIDDLE()
class PolynomialIntVal : public IntVal
{
    FIDDLE(...)
public:
    IntegerLiteralValue getConstantTerm() { return getIntConstOperand(1); };
    OperandView<PolynomialIntValTerm> getTerms()
    {
        return OperandView<PolynomialIntValTerm>(this, 2, getOperandCount() - 2);
    };

    bool isConstant() { return getOperandCount() == 1; }

    // Overrides should be public so base classes can access
    void _toTextOverride(StringBuilder& out);
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
    Val* _resolveImplOverride();

    static IntVal* neg(ASTBuilder* astBuilder, IntVal* base);
    static IntVal* add(ASTBuilder* astBuilder, IntVal* op0, IntVal* op1);
    static IntVal* sub(ASTBuilder* astBuilder, IntVal* op0, IntVal* op1);
    static IntVal* mul(ASTBuilder* astBuilder, IntVal* op0, IntVal* op1);
    PolynomialIntVal(
        Type* inType,
        IntegerLiteralValue inConstantTerm,
        ArrayView<PolynomialIntValTerm*> inTerms)
    {
        setOperands(inType, inConstantTerm);
        addOperands(inTerms);
    }

    bool _isLinkTimeValOverride()
    {
        for (auto factor : getTerms())
        {
            if (factor->isLinkTimeVal())
                return true;
        }
        return false;
    }
};

/// An unknown integer value indicating an erroneous sub-expression
FIDDLE()
class ErrorIntVal : public IntVal
{
    FIDDLE(...)
    ErrorIntVal(Type* inType) { setOperands(inType); }

    // TODO: We should probably eventually just have an `ErrorVal` here
    // and have all `Val`s that represent ordinary values hold their
    // `Type` so that we can have an `ErrorVal` of any type.

    // Overrides should be public so base classes can access
    void _toTextOverride(StringBuilder& out);
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
    Val* _resolveImplOverride() { return this; }
    bool _isLinkTimeValOverride() { return false; }
};

// A witness to the fact that some proposition is true, encoded
// at the level of the type system.
//
// Given a generic like:
//
//     void example<L>(L light)
//          where L : ILight
//     { ... }
//
// a call to `example()` needs two things for us to be sure
// it is valid:
//
// 1. We need a type `X` to use as the argument for the
//    parameter `L`. We might supply this explicitly, or
//    via inference.
//
// 2. We need a *proof* that whatever `X` we chose conforms
//    to the `ILight` interface.
//
// The easiest way to make such a proof is by construction,
// and a `Witness` represents such a constructive proof.
// Conceptually a proposition like `X : ILight` can be
// seen as a type, and witness prooving that proposition
// is a value of that type.
//
// We construct and store witnesses explicitly during
// semantic checking because they can help us with
// generating downstream code. By following the structure
// of a witness (the structure of a proof) we can, e.g.,
// navigate from the knowledge that `X : ILight` to
// the concrete declarations that provide the implementation
// of `ILight` for `X`.
//
FIDDLE(abstract)
class Witness : public Val
{
    FIDDLE(...)
};

// A witness that one type is a subtype of another
// (where by "subtype" we include both inheritance
// relationships and type-conforms-to-interface relationships)
//
// TODO: we may need to tease those apart.
FIDDLE(abstract)
class SubtypeWitness : public Witness
{
    FIDDLE(...)
    Val* _resolveImplOverride();

    Type* getSub() { return as<Type>(getOperand(0)); }
    Type* getSup() { return as<Type>(getOperand(1)); }

    ConversionCost _getOverloadResolutionCostOverride();
    ConversionCost getOverloadResolutionCost();
};

FIDDLE()
class TypePackSubtypeWitness : public SubtypeWitness
{
    FIDDLE(...)
    Type* getSub() { return as<Type>(getOperand(0)); }
    Type* getSup() { return as<Type>(getOperand(1)); }

    Index getCount() { return getOperandCount() - 2; }
    SubtypeWitness* getWitness(Index index) { return as<SubtypeWitness>(getOperand(index + 2)); }

    TypePackSubtypeWitness(Type* sub, Type* sup, ArrayView<SubtypeWitness*> witnesses)
    {
        setOperands(sub);
        addOperands(sup);
        for (auto w : witnesses)
            addOperands(ValNodeOperand(w));
    }

    void _toTextOverride(StringBuilder& out);
    Val* _resolveImplOverride();
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
};

FIDDLE()
class EachSubtypeWitness : public SubtypeWitness
{
    FIDDLE(...)
    EachSubtypeWitness(Type* sub, Type* sup, SubtypeWitness* patternWitness)
    {
        setOperands(sub, sup, patternWitness);
    }
    Type* getSub() { return as<Type>(getOperand(0)); }
    Type* getSup() { return as<Type>(getOperand(1)); }
    SubtypeWitness* getPatternTypeWitness() { return as<SubtypeWitness>(getOperand(2)); }
    void _toTextOverride(StringBuilder& out);
    Val* _resolveImplOverride();
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
};

FIDDLE()
class ExpandSubtypeWitness : public SubtypeWitness
{
    FIDDLE(...)
    ExpandSubtypeWitness(Type* sub, Type* sup, SubtypeWitness* patternWitness)
    {
        setOperands(sub, sup, patternWitness);
    }
    Type* getSub() { return as<Type>(getOperand(0)); }
    Type* getSup() { return as<Type>(getOperand(1)); }
    SubtypeWitness* getPatternTypeWitness() { return as<SubtypeWitness>(getOperand(2)); }
    void _toTextOverride(StringBuilder& out);
    Val* _resolveImplOverride();
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
};

FIDDLE()
class TypeEqualityWitness : public SubtypeWitness
{
    FIDDLE(...)
    TypeEqualityWitness(Type* subType, Type* supType) { setOperands(subType, supType); }

    // Overrides should be public so base classes can access
    void _toTextOverride(StringBuilder& out);
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
};

FIDDLE()
class TypeCoercionWitness : public Witness
{
    FIDDLE(...)
    Type* getFromType() { return as<Type>(getOperand(0)); }
    Type* getToType() { return as<Type>(getOperand(1)); }

    DeclRef<Decl> getDeclRef() { return as<DeclRefBase>(getOperand(2)); }

    void _toTextOverride(StringBuilder& out);
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
    Val* _resolveImplOverride();
};

// A witness that one type is a subtype of another
// because some in-scope declaration says so
FIDDLE()
class DeclaredSubtypeWitness : public SubtypeWitness
{
    FIDDLE(...)
    DeclRef<Decl> getDeclRef() { return as<DeclRefBase>(getOperand(2)); }

    bool isEquality()
    {
        if (auto declRef = getDeclRef().as<GenericTypeConstraintDecl>())
            return declRef.getDecl()->isEqualityConstraint;
        return false;
    }

    // Overrides should be public so base classes can access
    void _toTextOverride(StringBuilder& out);
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
    Val* _resolveImplOverride();

    DeclaredSubtypeWitness(Type* inSub, Type* inSup, DeclRef<Decl> inDeclRef)
    {
        setOperands(inSub, inSup, inDeclRef);
    }

    ConversionCost _getOverloadResolutionCostOverride();
};

// A witness that `sub : sup` because `sub : mid` and `mid : sup`
FIDDLE()
class TransitiveSubtypeWitness : public SubtypeWitness
{
    FIDDLE(...)
    // Witness that `sub : mid`
    SubtypeWitness* getSubToMid() { return as<SubtypeWitness>(getOperand(2)); }

    // Witness that `mid : sup`
    SubtypeWitness* getMidToSup() { return as<SubtypeWitness>(getOperand(3)); }

    // Overrides should be public so base classes can access
    void _toTextOverride(StringBuilder& out);
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);

    TransitiveSubtypeWitness(
        Type* subType,
        Type* supType,
        SubtypeWitness* inSubToMid,
        SubtypeWitness* inMidToSup)
    {
        setOperands(subType, supType, inSubToMid, inMidToSup);
    }

    ConversionCost _getOverloadResolutionCostOverride();
};

// A witness that `sub : sup` because `sub` was wrapped into
// an existential of type `sup`.
FIDDLE()
class ExtractExistentialSubtypeWitness : public SubtypeWitness
{
    FIDDLE(...)
    // The declaration of the existential value that has been opened
    DeclRef<VarDeclBase> getDeclRef() { return as<DeclRefBase>(getOperand(2)); }

    ExtractExistentialSubtypeWitness(Type* inSub, Type* inSup, DeclRef<Decl> inDeclRef)
    {
        setOperands(inSub, inSup, inDeclRef);
    }

    // Overrides should be public so base classes can access
    void _toTextOverride(StringBuilder& out);
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
};

/// A witness of the fact that a user provided "__Dynamic" type argument is a
/// subtype to the existential type parameter.
FIDDLE()
class DynamicSubtypeWitness : public SubtypeWitness
{
    FIDDLE(...)
    DynamicSubtypeWitness(Type* inSub, Type* inSup) { setOperands(inSub, inSup); }
};

/// A witness that `T : L & R` because `T : L` and `T : R`
FIDDLE()
class ConjunctionSubtypeWitness : public SubtypeWitness
{
    FIDDLE(...)
    // At the operational level, this class of witness is
    // an operation that takes two witness tables `leftWitness`
    // and `rightWitness`, and forms a pair/tuple of
    // `(leftWitness, rightWitness)`.
    static const int kComponentCount = 2;

    ConjunctionSubtypeWitness(Type* inSub, Type* inSup, SubtypeWitness* left, SubtypeWitness* right)
    {
        setOperands(inSub, inSup, left, right);
    }

    SubtypeWitness* getLeftWitness() const { return as<SubtypeWitness>(getOperand(2)); }
    SubtypeWitness* getRightWitness() const { return as<SubtypeWitness>(getOperand(3)); }

    Count getComponentCount() const { return 2; }
    SubtypeWitness* getComponentWitness(Index index) const
    {
        SLANG_ASSERT(index >= 0 && index < kComponentCount);
        return as<SubtypeWitness>(getOperand(2 + index));
    }

    void _toTextOverride(StringBuilder& out);
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);

    ConversionCost _getOverloadResolutionCostOverride();
};

/// A witness that `T <: L` or `T <: R` because `T <: L&R`
FIDDLE()
class ExtractFromConjunctionSubtypeWitness : public SubtypeWitness
{
    FIDDLE(...)
    // At the operational level, this class of witness is
    // an operation that takes a pair/tuple of witness tables
    // `(leftWtiness, rightWitness)` and extracts one of the
    // elements of it.

    /// Witness that `T < L & R`
    SubtypeWitness* getConjunctionWitness() { return as<SubtypeWitness>(getOperand(2)); };

    ExtractFromConjunctionSubtypeWitness(
        Type* inSub,
        Type* inSup,
        SubtypeWitness* witness,
        int index)
    {
        setOperands(inSub, inSup, witness, index);
    }

    /// The zero-based index of the super-type we care about in the conjunction
    ///
    /// If `conjunctionWitness` is `T < L & R` then this index should be zero if
    /// we want to represent `T < L` and one if we want `T < R`.
    ///
    int getIndexInConjunction() { return (int)getIntConstOperand(3); };

    void _toTextOverride(StringBuilder& out);
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);

    ConversionCost _getOverloadResolutionCostOverride();
};

/// A value that represents a modifier attached to some other value
FIDDLE()
class ModifierVal : public Val
{
    FIDDLE(...)
    Val* _resolveImplOverride() { return this; }
};

FIDDLE()
class TypeModifierVal : public ModifierVal
{
    FIDDLE(...)
};

FIDDLE()
class ResourceFormatModifierVal : public TypeModifierVal
{
    FIDDLE(...)
};

FIDDLE()
class UNormModifierVal : public ResourceFormatModifierVal
{
    FIDDLE(...)
    void _toTextOverride(StringBuilder& out);
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
};

FIDDLE()
class SNormModifierVal : public ResourceFormatModifierVal
{
    FIDDLE(...)
    void _toTextOverride(StringBuilder& out);
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
};

FIDDLE()
class NoDiffModifierVal : public TypeModifierVal
{
    FIDDLE(...)
    void _toTextOverride(StringBuilder& out);
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
};

/// Represents the result of differentiating a function.
FIDDLE()
class DifferentiateVal : public Val
{
    FIDDLE(...)
    DifferentiateVal(DeclRef<Decl> inFunc) { setOperands(inFunc); }

    DeclRef<Decl> getFunc() { return as<DeclRefBase>(getOperand(0)); }

    void _toTextOverride(StringBuilder& out);
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
    Val* _resolveImplOverride();
};

FIDDLE()
class ForwardDifferentiateVal : public DifferentiateVal
{
    FIDDLE(...)
    ForwardDifferentiateVal(DeclRef<Decl> inFunc)
        : DifferentiateVal(inFunc)
    {
    }
};

FIDDLE()
class BackwardDifferentiateVal : public DifferentiateVal
{
    FIDDLE(...)
    BackwardDifferentiateVal(DeclRef<Decl> inFunc)
        : DifferentiateVal(inFunc)
    {
    }
};

FIDDLE()
class BackwardDifferentiateIntermediateTypeVal : public DifferentiateVal
{
    FIDDLE(...)
    BackwardDifferentiateIntermediateTypeVal(DeclRef<Decl> inFunc)
        : DifferentiateVal(inFunc)
    {
    }
};

FIDDLE()
class BackwardDifferentiatePrimalVal : public DifferentiateVal
{
    FIDDLE(...)
    BackwardDifferentiatePrimalVal(DeclRef<Decl> inFunc)
        : DifferentiateVal(inFunc)
    {
    }
};

FIDDLE()
class BackwardDifferentiatePropagateVal : public DifferentiateVal
{
    FIDDLE(...)
    BackwardDifferentiatePropagateVal(DeclRef<Decl> inFunc)
        : DifferentiateVal(inFunc)
    {
    }
};


template<typename F>
void SubstitutionSet::forEachGenericSubstitution(F func) const
{
    if (!declRef)
        return;
    for (auto subst = declRef; subst; subst = subst->getBase())
    {
        if (auto genSubst = as<GenericAppDeclRef>(subst))
            func(genSubst->getGenericDecl(), genSubst->getArgs());
    }
}

template<typename F>
void SubstitutionSet::forEachSubstitutionArg(F func) const
{
    if (!declRef)
        return;
    for (auto subst = declRef; subst; subst = subst->getBase())
    {
        if (auto genSubst = as<GenericAppDeclRef>(subst))
        {
            for (auto arg : genSubst->getArgs())
                func(arg);
        }
        else if (auto thisSubst = as<LookupDeclRef>(subst))
        {
            func(thisSubst->getWitness()->getSub());
        }
    }
}

inline bool isTypeEqualityWitness(Val* witness)
{
    if (auto declaredWitness = as<DeclaredSubtypeWitness>(witness))
    {
        return declaredWitness->isEquality();
    }
    else if (as<TypeEqualityWitness>(witness))
    {
        return true;
    }
    else if (auto eachWitness = as<EachSubtypeWitness>(witness))
    {
        return isTypeEqualityWitness(eachWitness->getPatternTypeWitness());
    }
    else if (auto typePackWitness = as<TypePackSubtypeWitness>(witness))
    {
        for (Index i = 0; i < typePackWitness->getCount(); i++)
        {
            if (!isTypeEqualityWitness(typePackWitness->getWitness(i)))
                return false;
        }
        return true;
    }
    return false;
}

} // namespace Slang
