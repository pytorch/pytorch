#pragma once

#include "compiler-core/slang-doc-extractor.h"
#include "diagnostics.h"

namespace CppParse
{
using namespace Slang;

enum class ReflectionType : uint8_t
{
    NotReflected,
    Reflected,
};

// Pre-declare
class TypeSet;
class SourceOrigin;

struct ScopeNode;

class Node : public RefObject
{
public:
    enum class Kind : uint8_t
    {
        Invalid,

        StructType,
        ClassType,

        Enum,
        EnumClass,

        Namespace,
        AnonymousNamespace,

        Field,
        EnumCase,

        TypeDef,

        Callable, ///< Functions/methods

        Other,   ///< Used 'other' parsing like for TYPE
        Unknown, ///< Used for marking tokens consumed but usage is not known

        CountOf,
    };

    enum class KindRange
    {
        ScopeStart = int(Kind::StructType),
        ScopeEnd = int(Kind::AnonymousNamespace),

        ClassLikeStart = int(Kind::StructType),
        ClassLikeEnd = int(Kind::ClassType),

        ScopeTypeStart = int(Kind::StructType),
        ScopeTypeEnd = int(Kind::EnumClass),

        OtherTypeStart = int(Kind::TypeDef),
        OtherTypeEnd = int(Kind::TypeDef),

        EnumStart = int(Kind::Enum),
        EnumEnd = int(Kind::EnumClass),
    };

    /// Returns true if kind can cast to this type
    /// Used for implementing as<T> casting
    static bool isOfKind(Kind kind)
    {
        SLANG_UNUSED(kind);
        return true;
    }

    static bool isKindScope(Kind kind)
    {
        return int(kind) >= int(KindRange::ScopeStart) && int(kind) <= int(KindRange::ScopeEnd);
    }
    static bool isKindClassLike(Kind kind)
    {
        return int(kind) >= int(KindRange::ClassLikeStart) &&
               int(kind) <= int(KindRange::ClassLikeEnd);
    }
    static bool isKindEnumLike(Kind kind)
    {
        return int(kind) >= int(KindRange::EnumStart) && int(kind) <= int(KindRange::EnumEnd);
    }

    /// It a type, but doesn't have a scope
    static bool isKindOtherType(Kind kind)
    {
        return int(kind) >= int(KindRange::OtherTypeStart) &&
               int(kind) <= int(KindRange::OtherTypeEnd);
    }
    /// Is a type and has a scope
    static bool isKindScopeType(Kind kind)
    {
        return int(kind) >= int(KindRange::ScopeTypeStart) &&
               int(kind) <= int(KindRange::ScopeTypeEnd);
    }

    /// True if the kind is any type
    static bool isKindType(Kind kind) { return isKindOtherType(kind) || isKindScopeType(kind); }

    /// True if the kind can accept contained types
    static bool canKindContainTypes(Kind type)
    {
        switch (type)
        {
        case Kind::StructType:
        case Kind::ClassType:
        case Kind::Namespace:
        case Kind::AnonymousNamespace:
            {
                return true;
            }
        default:
            break;
        }
        return false;
    }

    bool isNamespace() const { return m_kind == Kind::Namespace; }
    bool isClassLike() const { return isKindClassLike(m_kind); }
    bool isScope() const { return isKindScope(m_kind); }
    bool isEnumLike() const { return isKindEnumLike(m_kind); }

    /// These are useful for the filter
    static bool isClassLikeAndReflected(Node* node)
    {
        return node->isClassLike() && node->isReflected();
    }
    static bool isClassLike(Node* node) { return isKindClassLike(node->m_kind); }

    virtual void dump(int indent, StringBuilder& out) = 0;

    /// Do depth first traversal of nodes in scopes
    virtual void calcScopeDepthFirst(List<Node*>& outNodes);

    /// Calculate the absolute name for this namespace/type
    void calcAbsoluteName(StringBuilder& outName) const;

    /// Get the absolute name
    String getAbsoluteName() const
    {
        StringBuilder buf;
        calcAbsoluteName(buf);
        return buf.produceString();
    }

    /// Calculate the scope path to this node, from the root
    void calcScopePath(List<Node*>& outPath) { calcScopePath(this, outPath); }

    /// True if reflected
    bool isReflected() const { return m_reflectionType == ReflectionType::Reflected; }

    SourceLoc getSourceLoc() const { return m_name.getLoc(); }

    ScopeNode* getRootScope();

    typedef bool (*Filter)(Node* node);

    template<typename T>
    static void filter(Filter filter, List<T*>& io)
    {
        const Node* _isNodeDerived = (T*)nullptr;
        SLANG_UNUSED(_isNodeDerived);
        filterImpl(filter, reinterpret_cast<List<Node*>&>(io));
    }

    static void filterImpl(Filter filter, List<Node*>& io);

    static void calcScopePath(Node* node, List<Node*>& outPath);

    /// Lookup a name in just the specified scope
    /// Handles anonymous namespaces, or name lookups that are in the parents space
    static Node* lookupNameInScope(ScopeNode* scope, const UnownedStringSlice& name);

    /// Lookup from a path
    static Node* lookupFromScope(ScopeNode* scope, const UnownedStringSlice* path, Index pathCount);
    /// Looks up *just* from the specified scope.
    static Node* lookupFromScope(ScopeNode* scope, const UnownedStringSlice& slice);

    /// Look up name (which can contain ::)
    static Node* lookup(ScopeNode* scope, const UnownedStringSlice& name);

    static void splitPath(const UnownedStringSlice& slice, List<UnownedStringSlice>& outSplitPath);

    /// If markup is specified dump it
    void dumpMarkup(int indent, StringBuilder& out);

    Node(Kind type)
        : m_kind(type), m_parentScope(nullptr), m_reflectionType(ReflectionType::NotReflected)
    {
    }

    Kind m_kind;                     ///< The kind of node this is
    ReflectionType m_reflectionType; ///< Classes can be traversed, but not reflected. To be
                                     ///< reflected they have to contain the marker

    MarkupVisibility m_markupVisibility =
        MarkupVisibility::Public; ///< The visibility of the markup
    String m_markup;              ///< Documentation associated with this node

    Token m_name; ///< The name of this scope/type

    ScopeNode* m_parentScope; ///< The scope this type/scope is defined in
};

struct ScopeNode : public Node
{
    typedef Node Super;

    static bool isOfKind(Kind kind) { return isKindScope(kind); }

    virtual void dump(int indent, StringBuilder& out) SLANG_OVERRIDE;
    virtual void calcScopeDepthFirst(List<Node*>& outNodes) SLANG_OVERRIDE;

    /// True if can contain callable entries
    bool canContainCallable() const { return isClassLike() || isNamespace(); }

    /// True if can accept fields (class like types can)
    bool canContainFields() const { return isClassLike(); }

    /// True if the scope can accept types
    bool canContainTypes() const { return canKindContainTypes(m_kind); }

    /// Gets the reflection for any contained types
    ReflectionType getContainedReflectionType() const
    {
        return m_reflectionType == ReflectionType::NotReflected ? ReflectionType::NotReflected
                                                                : m_reflectionOverride;
    }

    /// Add a child node to this nodes scope
    void addChild(Node* child);
    /// Adds the child but does not add the name to the map
    void addChildIgnoringName(Node* child);

    /// Find a child node in this scope with the specified name. Return nullptr if not found
    Node* findChild(const UnownedStringSlice& name) const;

    /// Gets the anonymous namespace associated with this scope
    ScopeNode* getAnonymousNamespace();

    ScopeNode(Kind kind)
        : Super(kind)
        , m_reflectionOverride(ReflectionType::Reflected)
        , m_anonymousNamespace(nullptr)
    {
    }

    /// For child types, fields, how reflection is handled. If this type is not reflected
    ReflectionType m_reflectionOverride;

    /// All of the types and namespaces in this *scope*
    List<RefPtr<Node>> m_children;

    /// Map from a name (in this scope) to the Node
    Dictionary<UnownedStringSlice, Node*> m_childMap;

    /// There can only be one anonymousNamespace for a scope. If there is one it's held here
    ScopeNode* m_anonymousNamespace;
};

struct FieldNode : public Node
{
    typedef Node Super;

    static bool isOfKind(Kind kind) { return kind == Kind::Field; }

    virtual void dump(int indent, StringBuilder& out) SLANG_OVERRIDE;

    FieldNode()
        : Super(Kind::Field)
    {
    }

    UnownedStringSlice m_fieldType;

    bool m_isStatic = false;

    /// TODO(JS): We may want to add initializer tokens
};

struct ClassLikeNode : public ScopeNode
{
    typedef ScopeNode Super;

    static bool isOfKind(Kind kind) { return isKindClassLike(kind); }

    /// Add a node that is derived from this
    void addDerived(ClassLikeNode* derived);

    /// Dump all of the derived types
    void dumpDerived(int indentCount, StringBuilder& out);

    /// Calculates the derived depth
    Index calcDerivedDepth() const;

    /// Find the last (reflected) derived type
    ClassLikeNode* findLastDerived();

    /// Traverse the hierarchy of derived nodes, in depth first order
    void calcDerivedDepthFirst(List<ClassLikeNode*>& outNodes);

    /// True if has a derived type that is reflected
    bool hasReflectedDerivedType() const;

    /// Stores in out any reflected derived types
    void getReflectedDerivedTypes(List<ClassLikeNode*>& out) const;

    // Node Impl
    virtual void dump(int indent, StringBuilder& out) SLANG_OVERRIDE;

    ClassLikeNode(Kind kind)
        : Super(kind), m_origin(nullptr), m_typeSet(nullptr), m_superNode(nullptr)
    {
        SLANG_ASSERT(kind == Kind::ClassType || kind == Kind::StructType);
    }

    SourceOrigin* m_origin; ///< Defines where this was uniquely defined.

    Token m_marker; ///< The marker associated with this scope (typically the marker is SLANG_CLASS
                    ///< etc, that is used to identify reflectedType)

    List<RefPtr<ClassLikeNode>> m_derivedTypes; ///< All of the types derived from this type

    TypeSet* m_typeSet; ///< The typeset this type belongs to.

    Token m_super;              ///< Super class name
    ClassLikeNode* m_superNode; ///< If this is a class/struct, the type it is derived from (or
                                ///< nullptr if base)
};

struct CallableNode : public Node
{
    typedef Node Super;

    static bool isOfKind(Kind kind) { return kind == Kind::Callable; }

    virtual void dump(int indent, StringBuilder& out) SLANG_OVERRIDE;

    CallableNode()
        : Super(Kind::Callable)
    {
    }

    struct Param
    {
        UnownedStringSlice m_type;
        Token m_name;
    };

    UnownedStringSlice m_returnType;

    CallableNode* m_nextOverload = nullptr;

    List<Param> m_params;

    bool m_isStatic = false;
    bool m_isVirtual = false;
    bool m_isPure = false;
};

struct EnumCaseNode : public Node
{
    typedef Node Super;

    static bool isOfKind(Kind kind) { return kind == Kind::EnumCase; }

    virtual void dump(int indent, StringBuilder& out) SLANG_OVERRIDE;

    EnumCaseNode()
        : Super(Kind::EnumCase)
    {
    }

    // Tokens that make up the value. If not defined will be empty
    List<Token> m_valueTokens;
};

struct EnumNode : public ScopeNode
{
    typedef ScopeNode Super;
    static bool isOfKind(Kind kind) { return isKindEnumLike(kind); }

    virtual void dump(int indent, StringBuilder& out) SLANG_OVERRIDE;

    EnumNode(Kind kind)
        : Super(kind)
    {
        SLANG_ASSERT(isKindEnumLike(kind));
    }

    List<Token> m_backingTokens;
};

struct TypeDefNode : public Node
{
    typedef Node Super;
    static bool isOfKind(Kind kind) { return kind == Kind::TypeDef; }

    virtual void dump(int indent, StringBuilder& out) SLANG_OVERRIDE;

    TypeDefNode()
        : Super(Kind::TypeDef)
    {
    }

    List<Token> m_targetTypeTokens;
};

template<typename T>
T* as(Node* node)
{
    return (node && T::isOfKind(node->m_kind)) ? static_cast<T*>(node) : nullptr;
}

} // namespace CppParse
