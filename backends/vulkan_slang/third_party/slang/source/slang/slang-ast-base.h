// slang-ast-base.h

#pragma once

#include "slang-ast-base.h.fiddle"
#include "slang-ast-forward-declarations.h"
#include "slang-ast-support-types.h"
#include "slang-capability.h"

// This file defines the primary base classes for the hierarchy of
// AST nodes and related objects. For example, this is where the
// basic `Decl`, `Stmt`, `Expr`, `type`, etc. definitions come from.

FIDDLE()
namespace Slang
{

class ASTBuilder;
struct SemanticsVisitor;

FIDDLE(abstract)
class NodeBase
{
    FIDDLE(...)

    // MUST be called before used. Called automatically via the ASTBuilder.
    // Note that the astBuilder is not stored in the NodeBase derived types by default.
    SLANG_FORCE_INLINE void init(ASTNodeType inAstNodeType, ASTBuilder* inAstBuilder)
    {
        SLANG_UNUSED(inAstBuilder);
        astNodeType = inAstNodeType;
#ifdef _DEBUG
        _initDebug(inAstNodeType, inAstBuilder);
#endif
    }

    void _initDebug(ASTNodeType inAstNodeType, ASTBuilder* inAstBuilder);

    SyntaxClass<NodeBase> getClass() const { return SyntaxClass<NodeBase>(astNodeType); }

    /// The type of the node. ASTNodeType(-1) is an invalid node type, and shouldn't appear on any
    /// correctly constructed (through ASTBuilder) NodeBase derived class.
    /// The actual type is set when constructed on the ASTBuilder.
    FIDDLE() ASTNodeType astNodeType = ASTNodeType(-1);

#ifdef _DEBUG
    SLANG_UNREFLECTED int32_t _debugUID = 0;
#endif
};

// Casting of NodeBase

template<typename T>
SLANG_FORCE_INLINE T* dynamicCast(NodeBase* node)
{
    return (node && node->getClass().isSubClassOf<T>()) ? static_cast<T*>(node) : nullptr;
}

template<typename T>
SLANG_FORCE_INLINE const T* dynamicCast(const NodeBase* node)
{
    return (node && node->getClass().isSubClassOf<T>()) ? static_cast<const T*>(node) : nullptr;
}

template<typename T>
SLANG_FORCE_INLINE T* as(NodeBase* node)
{
    return (node && node->getClass().isSubClassOf<T>()) ? static_cast<T*>(node) : nullptr;
}

template<typename T>
SLANG_FORCE_INLINE const T* as(const NodeBase* node)
{
    return (node && node->getClass().isSubClassOf<T>()) ? static_cast<const T*>(node) : nullptr;
}

// Because DeclRefBase is now a `Val`, we prevent casting it directly into other nodes
// to avoid confusion and bugs. Instead, use the `as<>()` method on `DeclRefBase` to
// get a `DeclRef<T>` for a specific node type.
template<typename T>
T* as(
    DeclRefBase* declRefBase,
    typename EnableIf<!IsBaseOf<DeclRefBase, T>::Value, void*>::type arg = nullptr) = delete;

template<typename T>
T* as(
    DeclRefBase* declRefBase,
    typename EnableIf<IsBaseOf<DeclRefBase, T>::Value, void*>::type arg = nullptr)
{
    SLANG_UNUSED(arg);
    return dynamicCast<T>(declRefBase);
}

template<typename T, typename U>
DeclRef<T> as(DeclRef<U> declRef)
{
    return DeclRef<T>(declRef);
}

FIDDLE()
class Scope : public NodeBase
{
    FIDDLE(...)

    // The container to use for lookup
    //
    // Note(tfoley): This is kept as an unowned pointer
    // so that a scope can't keep parts of the AST alive,
    // but the opposite it allowed.
    ContainerDecl* containerDecl = nullptr;

    // The parent of this scope (where lookup should go if nothing is found locally)
    Scope* parent = nullptr;

    SLANG_UNREFLECTED
    // The next sibling of this scope (a peer for lookup)
    Scope* nextSibling = nullptr;
};

// Base class for all nodes representing actual syntax
// (thus having a location in the source code)
FIDDLE(abstract)
class SyntaxNodeBase : public NodeBase
{
    FIDDLE(...)

    // The primary source location associated with this AST node
    FIDDLE() SourceLoc loc;
};

enum class ValNodeOperandKind
{
    ConstantValue,
    ValNode,
    ASTNode,
};

struct ValNodeOperand
{
    ValNodeOperandKind kind = ValNodeOperandKind::ConstantValue;
    union
    {
        NodeBase* nodeOperand;
        int64_t intOperand;
    } values;

    ValNodeOperand() { values.intOperand = 0; }

    int64_t getIntConstant() const
    {
        SLANG_ASSERT(kind == ValNodeOperandKind::ConstantValue);
        return values.intOperand;
    }

    Val* getVal() const
    {
        SLANG_ASSERT(kind == ValNodeOperandKind::ValNode);
        return (Val*)values.nodeOperand;
    }

    Decl* getDecl() const
    {
        SLANG_ASSERT(kind == ValNodeOperandKind::ASTNode);
        return (Decl*)values.nodeOperand;
    }

    explicit ValNodeOperand(NodeBase* node)
    {
        if constexpr (sizeof(values.nodeOperand) < sizeof(values.intOperand))
            values.intOperand = 0;

        if (as<Val>(node))
        {
            values.nodeOperand = (NodeBase*)node;
            kind = ValNodeOperandKind::ValNode;
        }
        else
        {
            values.nodeOperand = node;
            kind = ValNodeOperandKind::ASTNode;
        }
    }

    template<typename T>
    explicit ValNodeOperand(DeclRef<T> declRef)
    {
        if constexpr (sizeof(values.nodeOperand) < sizeof(values.intOperand))
            values.intOperand = 0;
        values.nodeOperand = declRef.declRefBase;
        kind = ValNodeOperandKind::ValNode;
    }

    template<typename T>
    explicit ValNodeOperand(T* node)
    {
        if constexpr (sizeof(values.nodeOperand) < sizeof(values.intOperand))
            values.intOperand = 0;
        if constexpr (std::is_base_of<Val, T>::value)
        {
            values.nodeOperand = (NodeBase*)node;
            kind = ValNodeOperandKind::ValNode;
        }
        else if constexpr (std::is_base_of<NodeBase, T>::value)
        {
            values.nodeOperand = node;
            kind = ValNodeOperandKind::ASTNode;
        }
        else
        {
            static_assert(
                std::is_base_of<Val, T>::value || std::is_base_of<NodeBase, T>::value,
                "pointer used as Val operand must be an AST node.");
        }
    }

    template<typename EnumType>
    explicit ValNodeOperand(EnumType intVal)
    {
        static_assert(
            std::is_trivial<EnumType>::value,
            "Type to construct NodeOperand must be trivial.");
        static_assert(
            sizeof(EnumType) <= sizeof(values),
            "size of operand must be less than pointer size.");
        values.intOperand = 0;
        memcpy(&values, &intVal, sizeof(intVal));
        kind = ValNodeOperandKind::ConstantValue;
    }
};

struct ValNodeDesc
{
private:
    HashCode hashCode = 0;

public:
    SyntaxClass<NodeBase> type;
    ShortList<ValNodeOperand, 8> operands;

    inline bool operator==(ValNodeDesc const& that) const
    {
        if (hashCode != that.hashCode)
            return false;
        if (type != that.type)
            return false;
        if (operands.getCount() != that.operands.getCount())
            return false;
        for (Index i = 0; i < operands.getCount(); ++i)
        {
            // Note: we are comparing the operands directly for identity
            // (pointer equality) rather than doing the `Val`-level
            // equality check.
            //
            // The rationale here is that nodes that will be created
            // via a `NodeDesc` *should* all be going through the
            // deduplication path anyway, as should their operands.
            //
            if (operands[i].values.intOperand != that.operands[i].values.intOperand)
                return false;
        }
        return true;
    }
    HashCode getHashCode() const { return hashCode; }
    void init();
};

template<int N>
static void addOrAppendToNodeList(ShortList<ValNodeOperand, N>&)
{
}

template<int N, typename... Ts>
static void addOrAppendToNodeList(
    ShortList<ValNodeOperand, N>& list,
    ExpandedSpecializationArgs e,
    Ts... ts)
{
    for (auto arg : e)
    {
        list.add(ValNodeOperand(arg.val));
        list.add(ValNodeOperand(arg.witness));
    }
    addOrAppendToNodeList(list, ts...);
}

template<int N, typename T, typename... Ts>
static void addOrAppendToNodeList(ShortList<ValNodeOperand, N>& list, T t, Ts... ts)
{
    list.add(ValNodeOperand(t));
    addOrAppendToNodeList(list, ts...);
}

template<int N, typename T, typename... Ts>
static void addOrAppendToNodeList(ShortList<ValNodeOperand, N>& list, const List<T>& l, Ts... ts)
{
    for (auto t : l)
        list.add(ValNodeOperand(t));
    addOrAppendToNodeList(list, ts...);
}

template<int N, typename T, typename... Ts>
static void addOrAppendToNodeList(ShortList<ValNodeOperand, N>& list, ConstArrayView<T> l, Ts... ts)
{
    for (auto t : l)
        list.add(ValNodeOperand(t));
    addOrAppendToNodeList(list, ts...);
}

template<int N, typename T, typename... Ts>
static void addOrAppendToNodeList(ShortList<ValNodeOperand, N>& list, ArrayView<T> l, Ts... ts)
{
    for (auto t : l)
        list.add(ValNodeOperand(t));
    addOrAppendToNodeList(list, ts...);
}

inline void addOrAppendToNodeList(List<ValNodeOperand>&) {}

template<typename... Ts>
static void addOrAppendToNodeList(
    List<ValNodeOperand>& list,
    ExpandedSpecializationArgs e,
    Ts... ts)
{
    for (auto arg : e)
    {
        list.add(ValNodeOperand(arg.val));
        list.add(ValNodeOperand(arg.witness));
    }
    addOrAppendToNodeList(list, ts...);
}

template<typename T, typename... Ts>
static void addOrAppendToNodeList(List<ValNodeOperand>& list, T t, Ts... ts)
{
    list.add(ValNodeOperand(t));
    addOrAppendToNodeList(list, ts...);
}

template<typename T, typename... Ts>
static void addOrAppendToNodeList(List<ValNodeOperand>& list, const List<T>& l, Ts... ts)
{
    for (auto t : l)
        list.add(ValNodeOperand(t));
    addOrAppendToNodeList(list, ts...);
}

template<typename T, typename... Ts>
static void addOrAppendToNodeList(List<ValNodeOperand>& list, ConstArrayView<T> l, Ts... ts)
{
    for (auto t : l)
        list.add(ValNodeOperand(t));
    addOrAppendToNodeList(list, ts...);
}

template<typename T, typename... Ts>
static void addOrAppendToNodeList(List<ValNodeOperand>& list, ArrayView<T> l, Ts... ts)
{
    for (auto t : l)
        list.add(ValNodeOperand(t));
    addOrAppendToNodeList(list, ts...);
}

// Base class for compile-time values (most often a type).
// These are *not* syntax nodes, because they do not have
// a unique location, and any two `Val`s representing
// the same value should be conceptually equal.

FIDDLE(abstract)
class Val : public NodeBase
{
    FIDDLE(...)

    template<typename T>
    struct OperandView
    {
        const Val* val;
        Index offset;
        Index count;
        OperandView()
        {
            val = nullptr;
            offset = 0;
            count = 0;
        }
        OperandView(const Val* val, Index offset, Index count)
            : val(val), offset(offset), count(count)
        {
        }
        Index getCount() { return count; }
        T* operator[](Index index) const { return as<T>(val->getOperand(index + offset)); }
        struct ConstIterator
        {
            const Val* val;
            Index i;
            bool operator==(ConstIterator other) const { return val == other.val && i == other.i; }
            bool operator!=(ConstIterator other) const { return val != other.val || i != other.i; }
            T* const& operator*() const { return *(this->operator->()); }
            T* const* operator->() const
            {
                return reinterpret_cast<T* const*>(&val->m_operands[i].values.nodeOperand);
            }
            ConstIterator& operator++()
            {
                i++;
                return *this;
            }
        };
        ConstIterator begin() const { return ConstIterator{val, offset}; }
        ConstIterator end() const { return ConstIterator{val, offset + count}; }
    };

    // construct a new value by applying a set of parameter
    // substitutions to this one
    Val* substitute(ASTBuilder* astBuilder, SubstitutionSet subst);

    // Lower-level interface for substitution. Like the basic
    // `Substitute` above, but also takes a by-reference
    // integer parameter that should be incremented when
    // returning a modified value (this can help the caller
    // decide whether they need to do anything).
    Val* substituteImpl(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);

    bool equals(Val* val) const
    {
        return this == val || (val && const_cast<Val*>(this)->resolve() == val->resolve());
    }

    // Appends as text to the end of the builder
    void toText(StringBuilder& out);
    String toString();

    HashCode getHashCode();
    bool operator==(const Val& v) const { return equals(const_cast<Val*>(&v)); }

    // Overrides should be public so base classes can access
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
    void _toTextOverride(StringBuilder& out);

    Val* _resolveImplOverride();

    Val* resolveImpl();
    Val* resolve();

    Val* getOperand(Index index) const { return m_operands[index].getVal(); }

    Decl* getDeclOperand(Index index) const { return m_operands[index].getDecl(); }

    int64_t getIntConstOperand(Index index) const { return m_operands[index].getIntConstant(); }

    Index getOperandCount() const { return m_operands.getCount(); }

    template<typename... TArgs>
    void setOperands(TArgs... args)
    {
        m_operands.clear();
        addOrAppendToNodeList(m_operands, args...);
    }
    template<typename... TArgs>
    void addOperands(TArgs... args)
    {
        addOrAppendToNodeList(m_operands, args...);
    }
    template<typename T>
    void addOperands(OperandView<T> operands)
    {
        for (auto v : operands)
            m_operands.add(ValNodeOperand(v));
    }
    FIDDLE() List<ValNodeOperand> m_operands;

    // Private use by the core module deserialization only. Since we know the Vals serialized into
    // the core module is already unique, we can just use `this` pointer as the `m_resolvedVal` so
    // we don't need to resolve them again.
    void _setUnique();

protected:
    Val* defaultResolveImpl();

private:
    mutable Val* m_resolvedVal = nullptr;
    SLANG_UNREFLECTED mutable Index m_resolvedValEpoch = 0;
};

template<int N, typename T, typename... Ts>
static void addOrAppendToNodeList(
    ShortList<ValNodeOperand, N>& list,
    Val::OperandView<T> l,
    Ts... ts)
{
    for (auto t : l)
        list.add(ValNodeOperand(t));
    addOrAppendToNodeList(list, ts...);
}

struct ValSet
{
    struct ValItem
    {
        Val* val = nullptr;
        ValItem() = default;
        ValItem(Val* v)
            : val(v)
        {
        }

        HashCode getHashCode() const { return val ? val->getHashCode() : 0; }
        bool operator==(const ValItem other) const
        {
            if (val == other.val)
                return true;
            if (val)
                return val->equals(other.val);
            else if (other.val)
                return other.val->equals(val);
            return false;
        }
    };
    HashSet<ValItem> set;
    bool add(Val* val) { return set.add(ValItem(val)); }
    bool contains(Val* val) { return set.contains(ValItem(val)); }
};


SLANG_FORCE_INLINE StringBuilder& operator<<(StringBuilder& io, Val* val)
{
    SLANG_ASSERT(val);
    val->toText(io);
    return io;
}

/// Given a `value` that refers to a `param` of some generic, attempt to apply
/// the `subst` to it and produce a new `Val` as a result.
///
/// If the `subst` does not include anything to replace `value`, then this function
/// returns null.
///
Val* maybeSubstituteGenericParam(Val* value, Decl* param, SubstitutionSet subst, int* ioDiff);

class Type;

template<typename T>
SLANG_FORCE_INLINE T* as(Type* obj);
template<typename T>
SLANG_FORCE_INLINE const T* as(const Type* obj);

// A type, representing a classifier for some term in the AST.
//
// Types can include "sugar" in that they may refer to a
// `typedef` which gives them a good name when printed as
// part of diagnostic messages.
//
// In order to operate on types, though, we often want
// to look past any sugar, and operate on an underlying
// "canonical" type. The representation caches a pointer to
// a canonical type on every type, so we can easily
// operate on the raw representation when needed.
FIDDLE(abstract)
class Type : public Val
{
    FIDDLE(...)

    /// Type derived types store the AST builder they were constructed on. The builder calls this
    /// function after constructing.
    SLANG_FORCE_INLINE void init(ASTNodeType inAstNodeType, ASTBuilder* inAstBuilder)
    {
        Val::init(inAstNodeType, inAstBuilder);
        m_astBuilderForReflection = inAstBuilder;
    }

    // Overrides should be public so base classes can access
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
    Type* _createCanonicalTypeOverride();
    Val* _resolveImplOverride();

    Type* getCanonicalType() { return as<Type>(resolve()); }

    ASTBuilder* getASTBuilderForReflection() const { return m_astBuilderForReflection; }

protected:
    Type* createCanonicalType();

    // We store the ASTBuilder to support reflection API only.
    // It should not be used for anything else, especially not for constructing new AST nodes during
    // semantic checking, since Val deduplication requires the entire semantic checking process to
    // stick with one ASTBuilder.
    // Call getCurrentASTBuilder() to obtain the right ASTBuilder for semantic checking.
    SLANG_UNREFLECTED ASTBuilder* m_astBuilderForReflection;
};

template<typename T>
SLANG_FORCE_INLINE T* as(Type* obj)
{
    return obj ? dynamicCast<T>(obj->getCanonicalType()) : nullptr;
}
template<typename T>
SLANG_FORCE_INLINE const T* as(const Type* obj)
{
    return obj ? dynamicCast<T>(const_cast<Type*>(obj)->getCanonicalType()) : nullptr;
}

class Decl;

// A reference to a declaration, which may include
// substitutions for generic parameters.
FIDDLE(abstract)
class DeclRefBase : public Val
{
    FIDDLE(...)

    Decl* getDecl() const { return getDeclOperand(0); }

    // Apply substitutions to this declaration reference
    DeclRefBase* substituteImpl(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);

    DeclRefBase* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff)
    {
        SLANG_UNUSED(astBuilder);
        SLANG_UNUSED(subst);
        SLANG_UNUSED(ioDiff);
        SLANG_UNREACHABLE("DeclRefBase::_substituteImplOverride not overrided.");
    }

    void _toTextOverride(StringBuilder& out)
    {
        SLANG_UNUSED(out);
        SLANG_UNREACHABLE("DeclRefBase::_toTextOverride not overrided.");
    }

    Val* _resolveImplOverride()
    {
        SLANG_UNREACHABLE("DeclRefBase::_resolveImplOverride not overrided.");
    }

    DeclRefBase* _getBaseOverride()
    {
        SLANG_UNREACHABLE("DeclRefBase::_getBaseOverride not overrided.");
    }

    // Returns true if 'as' will return a valid cast
    template<typename T>
    bool is() const
    {
        return Slang::as<T>(getDecl()) != nullptr;
    }

    // Convenience accessors for common properties of declarations
    Name* getName() const;
    SourceLoc getNameLoc() const;
    SourceLoc getLoc() const;
    DeclRefBase* getParent();
    String toString() const
    {
        StringBuilder sb;
        const_cast<DeclRefBase*>(this)->toText(sb);
        return sb.produceString();
    }
    DeclRefBase* getBase();
    void toText(StringBuilder& out);
};

SLANG_FORCE_INLINE StringBuilder& operator<<(StringBuilder& io, const DeclRefBase* declRef)
{
    if (declRef)
        const_cast<DeclRefBase*>(declRef)->toText(io);
    return io;
}

SLANG_FORCE_INLINE StringBuilder& operator<<(StringBuilder& io, Decl* decl)
{
    if (decl)
        makeDeclRef(decl).declRefBase->toText(io);
    return io;
}

FIDDLE(abstract)
class SyntaxNode : public SyntaxNodeBase
{
    FIDDLE(...)
};

//
// All modifiers are represented as full-fledged objects in the AST
// (that is, we don't use a bitfield, even for simple/common flags).
// This ensures that we can track source locations for all modifiers.
//
FIDDLE(abstract)
class Modifier : public SyntaxNode
{
    FIDDLE(...)

    // Next modifier in linked list of modifiers on same piece of syntax
    Modifier* next = nullptr;

    // The keyword that was used to introduce t that was used to name this modifier.
    FIDDLE() Name* keywordName = nullptr;

    Name* getKeywordName() { return keywordName; }
    NameLoc getKeywordNameAndLoc() { return NameLoc(keywordName, loc); }
};

// A syntax node which can have modifiers applied
FIDDLE(abstract)
class ModifiableSyntaxNode : public SyntaxNode
{
    FIDDLE(...)

    FIDDLE() Modifiers modifiers;

    template<typename T>
    FilteredModifierList<T> getModifiersOfType()
    {
        return FilteredModifierList<T>(modifiers.first);
    }

    // Find the first modifier of a given type, or return `nullptr` if none is found.
    template<typename T>
    T* findModifier()
    {
        return *getModifiersOfType<T>().begin();
    }

    template<typename T>
    bool hasModifier()
    {
        return findModifier<T>() != nullptr;
    }
};

struct ProvenenceNodeWithLoc
{
    NodeBase* referencedNode;
    SourceLoc referenceLoc;
};

// An intermediate type to represent either a single declaration, or a group of declarations
FIDDLE(abstract)
class DeclBase : public ModifiableSyntaxNode
{
    FIDDLE(...)
};

FIDDLE(abstract)
class Decl : public DeclBase
{
    FIDDLE(...)
public:
    FIDDLE() ContainerDecl* parentDecl = nullptr;

    DeclRefBase* getDefaultDeclRef();

    FIDDLE() NameLoc nameAndLoc;
    FIDDLE() CapabilitySet inferredCapabilityRequirements;

    FIDDLE() RefPtr<MarkupEntry> markup;

    Name* getName() const { return nameAndLoc.name; }
    SourceLoc getNameLoc() const { return nameAndLoc.loc; }
    NameLoc getNameAndLoc() const { return nameAndLoc; }

    DeclCheckStateExt checkState = DeclCheckState::Unchecked;

    // The next declaration defined in the same container with the same name
    Decl* nextInContainerWithSameName = nullptr;

    bool isChecked(DeclCheckState state) const { return checkState >= state; }
    void setCheckState(DeclCheckState state)
    {
        SLANG_RELEASE_ASSERT(state >= checkState.getState());
        checkState.setState(state);
    }
    bool isChildOf(Decl* other) const;

    // Track the decl reference that caused the requirement of a capability atom.
    SLANG_UNREFLECTED List<ProvenenceNodeWithLoc> capabilityRequirementProvenance;

    SLANG_UNREFLECTED bool hiddenFromLookup = false;

private:
    SLANG_UNREFLECTED DeclRefBase* m_defaultDeclRef = nullptr;
};

FIDDLE(abstract)
class Expr : public SyntaxNode
{
    FIDDLE(...)

    FIDDLE() QualType type;

    bool checked = false;
};

FIDDLE(abstract)
class Stmt : public ModifiableSyntaxNode
{
    FIDDLE(...)
};

template<typename T>
void DeclRef<T>::init(DeclRefBase* base)
{
    if (base && !Slang::as<T>(base->getDecl()))
        declRefBase = nullptr;
    else
        declRefBase = base;
}

template<typename T>
DeclRef<T>::DeclRef(Decl* decl)
{
    DeclRefBase* declRef = nullptr;
    if (decl)
    {
        declRef = decl->getDefaultDeclRef();
    }
    init(declRef);
}

template<typename T>
T* DeclRef<T>::getDecl() const
{
    return declRefBase ? (T*)declRefBase->getDecl() : nullptr;
}

template<typename T>
Name* DeclRef<T>::getName() const
{
    if (declRefBase)
        return declRefBase->getName();
    return nullptr;
}

template<typename T>
SourceLoc DeclRef<T>::getNameLoc() const
{
    if (declRefBase)
        return declRefBase->getNameLoc();
    return SourceLoc();
}

template<typename T>
SourceLoc DeclRef<T>::getLoc() const
{
    if (declRefBase)
        return declRefBase->getLoc();
    return SourceLoc();
}

template<typename T>
DeclRef<ContainerDecl> DeclRef<T>::getParent() const
{
    if (declRefBase)
        return DeclRef<ContainerDecl>(declRefBase->getParent());
    return DeclRef<ContainerDecl>((DeclRefBase*)nullptr);
}

template<typename T>
HashCode DeclRef<T>::getHashCode() const
{
    if (declRefBase)
        return declRefBase->getHashCode();
    return 0;
}

template<typename T>
Type* DeclRef<T>::substitute(ASTBuilder* astBuilder, Type* type) const
{
    SLANG_UNUSED(astBuilder);
    if (!declRefBase)
        return type;
    return SubstitutionSet(*this).applyToType(astBuilder, type);
}

template<typename T>
SubstExpr<Expr> DeclRef<T>::substitute(ASTBuilder* astBuilder, Expr* expr) const
{
    SLANG_UNUSED(astBuilder);
    if (!declRefBase)
        return expr;
    return applySubstitutionToExpr(SubstitutionSet(*this), expr);
}

// Apply substitutions to a type or declaration
template<typename T>
template<typename U>
DeclRef<U> DeclRef<T>::substitute(ASTBuilder* astBuilder, DeclRef<U> declRef) const
{
    SLANG_UNUSED(astBuilder);
    if (!declRefBase)
        return declRef;
    return DeclRef<U>(SubstitutionSet(*this).applyToDeclRef(astBuilder, declRef.declRefBase));
}

// Apply substitutions to this declaration reference
template<typename T>
DeclRef<T> DeclRef<T>::substituteImpl(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff)
    const
{
    SLANG_UNUSED(astBuilder);
    if (!declRefBase)
        return *this;
    return DeclRef<T>(declRefBase->substituteImpl(astBuilder, subst, ioDiff));
}

Val::OperandView<Val> tryGetGenericArguments(SubstitutionSet substSet, Decl* genericDecl);


} // namespace Slang
