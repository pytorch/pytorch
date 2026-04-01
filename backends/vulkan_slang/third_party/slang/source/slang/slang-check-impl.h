// slang-check-impl.h
#pragma once

// This file provides the private interfaces used by
// the various `slang-check-*` files that provide
// the semantic checking infrastructure.

#include "slang-check.h"
#include "slang-compiler.h"
#include "slang-visitor.h"

namespace Slang
{
template<typename P, typename... Args>
bool diagnoseCapabilityErrors(
    DiagnosticSink* sink,
    CompilerOptionSet& optionSet,
    P const& pos,
    DiagnosticInfo const& info,
    Args const&... args)
{
    if (optionSet.getBoolOption(CompilerOptionName::IgnoreCapabilities))
        return false;
    return sink->diagnose(pos, info, args...);
}

enum class IsSubTypeOptions
{
    None = 0,

    /// A type may not be finished 'DeclCheckState::ReadyForLookup` while `isSubType` is called.
    /// We should not cache any negative results when this flag is set.
    NoCaching = 1 << 0,
};

/// Should the given `decl` be treated as a static rather than instance declaration?
bool isEffectivelyStatic(Decl* decl);

bool isGlobalDecl(Decl* decl);

bool isUnsafeForceInlineFunc(FunctionDeclBase* funcDecl);

bool isUniformParameterType(Type* type);

/// Create a new component type based on `inComponentType`, but with all its requiremetns filled.
RefPtr<ComponentType> fillRequirements(ComponentType* inComponentType);

Type* checkProperType(Linkage* linkage, TypeExp typeExp, DiagnosticSink* sink);

/// Get the element type if `type` is Ptr or PtrLike type, otherwise returns null.
/// Note: this currently does not include PtrTypeBase.
Type* getPointedToTypeIfCanImplicitDeref(Type* type);

inline int getIntValueBitSize(IntegerLiteralValue val)
{
    uint64_t v = val > 0 ? (uint64_t)val : (uint64_t)-val;
    int result = 1;
    while (v >>= 1)
    {
        result++;
    }
    return result;
}

int getTypeBitSize(Type* t);

// A flat representation of basic types (scalars, vectors and matrices)
// that can be used as lookup key in caches
struct BasicTypeKey
{
    uint32_t baseType : 8;
    uint32_t dim1 : 4;
    uint32_t dim2 : 4;
    uint32_t knownConstantBitCount : 8;
    uint32_t knownNegative : 1;
    uint32_t isLValue : 1;
    uint32_t reserved : 6;
    uint32_t getRaw() const
    {
        uint32_t val;
        memcpy(&val, this, sizeof(uint32_t));
        return val;
    }
    bool operator==(BasicTypeKey other) const { return getRaw() == other.getRaw(); }
    static BasicTypeKey invalid() { return BasicTypeKey{0xff, 0, 0, 0, 0, 0, 0}; }
};

SLANG_FORCE_INLINE BasicTypeKey makeBasicTypeKey(
    BaseType baseType,
    IntegerLiteralValue dim1 = 0,
    IntegerLiteralValue dim2 = 0,
    bool inIsLValue = false)
{
    SLANG_ASSERT(dim1 >= 0 && dim2 >= 0);
    return BasicTypeKey{
        uint8_t(baseType),
        uint8_t(dim1),
        uint8_t(dim2),
        0,
        0,
        (inIsLValue ? 1u : 0u),
        0};
}

inline BasicTypeKey makeBasicTypeKey(QualType typeIn, Expr* exprIn = nullptr)
{
    if (auto basicType = as<BasicExpressionType>(typeIn))
    {
        auto rs = makeBasicTypeKey(basicType->getBaseType());
        if (auto constInt = as<IntegerLiteralExpr>(exprIn))
        {
            if (constInt->value < 0)
            {
                rs.knownNegative = 1;
            }
            rs.knownConstantBitCount = getIntValueBitSize(constInt->value);
        }
        rs.isLValue = typeIn.isLeftValue ? 1u : 0u;
        return rs;
    }
    else if (auto vectorType = as<VectorExpressionType>(typeIn))
    {
        if (auto elemCount = as<ConstantIntVal>(vectorType->getElementCount()))
        {
            if (auto elemBasicType = as<BasicExpressionType>(vectorType->getElementType()))
            {
                return makeBasicTypeKey(
                    elemBasicType->getBaseType(),
                    elemCount->getValue(),
                    0,
                    typeIn.isLeftValue);
            }
        }
    }
    else if (auto matrixType = as<MatrixExpressionType>(typeIn))
    {
        if (auto elemCount1 = as<ConstantIntVal>(matrixType->getRowCount()))
        {
            if (auto elemCount2 = as<ConstantIntVal>(matrixType->getColumnCount()))
            {
                if (auto elemBasicType = as<BasicExpressionType>(matrixType->getElementType()))
                {
                    return makeBasicTypeKey(
                        elemBasicType->getBaseType(),
                        elemCount1->getValue(),
                        elemCount2->getValue(),
                        typeIn.isLeftValue);
                }
            }
        }
    }
    return BasicTypeKey::invalid();
}

struct BasicTypeKeyPair
{
    BasicTypeKey type1, type2;
    bool operator==(const BasicTypeKeyPair& rhs) const
    {
        return type1 == rhs.type1 && type2 == rhs.type2;
    }
    bool operator!=(const BasicTypeKeyPair& rhs) const { return !(*this == rhs); }

    bool isValid() const
    {
        return type1.getRaw() != BasicTypeKey::invalid().getRaw() &&
               type2.getRaw() != BasicTypeKey::invalid().getRaw();
    }

    HashCode getHashCode() const { return combineHash(type1.getRaw(), type2.getRaw()); }
};

struct OperatorOverloadCacheKey
{
    int32_t operatorName;
    bool isGLSLMode;
    BasicTypeKey args[2];
    bool operator==(OperatorOverloadCacheKey key) const
    {
        return operatorName == key.operatorName && args[0] == key.args[0] &&
               args[1] == key.args[1] && isGLSLMode == key.isGLSLMode;
    }
    HashCode getHashCode() const
    {
        return combineHash(operatorName, args[0].getRaw(), args[1].getRaw(), isGLSLMode ? 1 : 0);
    }
    bool fromOperatorExpr(OperatorExpr* opExpr)
    {
        // First, lets see if the argument types are ones
        // that we can encode in our space of keys.
        args[0] = BasicTypeKey::invalid();
        args[1] = BasicTypeKey::invalid();
        if (opExpr->arguments.getCount() > 2)
            return false;

        for (Index i = 0; i < opExpr->arguments.getCount(); i++)
        {
            auto key = makeBasicTypeKey(opExpr->arguments[i]->type, opExpr->arguments[i]);
            if (key.getRaw() == BasicTypeKey::invalid().getRaw())
            {
                return false;
            }
            args[i] = key;
        }

        // Next, lets see if we can find an intrinsic opcode
        // attached to an overloaded definition (filtered for
        // definitions that could conceivably apply to us).
        //
        // TODO: This should really be parsed on the operator name
        // plus fixity, rather than the intrinsic opcode...
        //
        // We will need to reject postfix definitions for prefix
        // operators, and vice versa, to ensure things work.
        //
        auto prefixExpr = as<PrefixExpr>(opExpr);
        auto postfixExpr = as<PostfixExpr>(opExpr);

        if (auto overloadedBase = as<OverloadedExpr>(opExpr->functionExpr))
        {
            for (auto item : overloadedBase->lookupResult2)
            {
                // Look at a candidate definition to be called and
                // see if it gives us a key to work with.
                //
                Decl* funcDecl = item.declRef.getDecl();
                if (auto genDecl = as<GenericDecl>(funcDecl))
                    funcDecl = genDecl->inner;

                // Reject definitions that have the wrong fixity.
                //
                if (prefixExpr && !funcDecl->findModifier<PrefixModifier>())
                    continue;
                if (postfixExpr && !funcDecl->findModifier<PostfixModifier>())
                    continue;

                if (auto intrinsicOp = funcDecl->findModifier<IntrinsicOpModifier>())
                {
                    operatorName = intrinsicOp->op;
                    return true;
                }
            }
        }
        return false;
    }
};

struct OverloadCandidate
{
    enum class Flavor
    {
        Func,
        Generic,
        UnspecializedGeneric,
        Expr,
    };
    Flavor flavor;

    enum class Status
    {
        GenericArgumentInferenceFailed,
        Unchecked,
        ArityChecked,
        FixityChecked,
        TypeChecked,
        DirectionChecked,
        VisibilityChecked,
        Applicable,
    };
    Status status = Status::Unchecked;

    typedef unsigned int Flags;
    enum Flag : Flags
    {
        IsPartiallyAppliedGeneric = 1 << 0,
    };
    Flags flags = 0;

    // Reference to the declaration being applied
    LookupResultItem item;

    // The expression when flavor is Expr.
    Expr* exprVal = nullptr;

    // Type of function being applied (for cases where `item` is not used)
    FuncType* funcType = nullptr;

    // The type of the result expression if this candidate is selected
    Type* resultType = nullptr;

    // A system for tracking constraints introduced on generic parameters
    //            ConstraintSystem constraintSystem;

    // How much conversion cost should be considered for this overload,
    // when ranking candidates.
    ConversionCost conversionCostSum = kConversionCost_None;

    // When required, a candidate can store a pre-checked list of
    // arguments so that we don't have to repeat work across checking
    // phases. Currently this is only needed for generics.
    SubstitutionSet subst;
};

struct ResolvedOperatorOverload
{
    // The resolved decl.
    Decl* decl;

    // The cached overload candidate in the current TypeCheckingCache.
    // Note that a `OverloadCandidate` object is not migratable over different
    // Linkages (compile sessions), so we will need to use `cacheVersion` to track
    // if this `candidate` is valid for the current session. If not, we will
    // recreate it from `decl`.
    OverloadCandidate candidate;
    // The version of the TypeCheckingCache for which the cached candidate is valid.
    int cacheVersion;
};

struct TypeCheckingCache : public RefObject
{
    Dictionary<OperatorOverloadCacheKey, ResolvedOperatorOverload> resolvedOperatorOverloadCache;
    Dictionary<BasicTypeKeyPair, ConversionCost> conversionCostCache;

    // The version used to invalidate the cached declRefs in ResolvedOperatorOverload entries.
    int version = 0;
};

enum class CoercionSite
{
    General,
    Assignment,
    Argument,
    Return,
    Initializer,
    ExplicitCoercion
};

struct FacetImpl;

/// Information about one "facet" of a type or declaration
///
/// In the simplest terms, a facet represents a grouping of
/// member declarations that were all originally declared
/// as part of the same `{}`-enclosed body.
///
/// A given *entity* (a type, type declaration, or `extension`
/// declaration) may have multiple facets, depending on what it
/// declares, what it inherits from, or what `extension`s apply to it.
///
/// Broadly, an entity will have:
///
/// * A *self facet*, if it has a body, that contains the members
///   the entity directly declares.
///
/// * An *inherited facet* for each base type that it (transitively)
///   inherits from. Inherited facets are either *direct*, if the
///   original entity stated the inheritance relationship, or
///   *indirect* if they arise from the transitive closure of the
///   inheritance relationship. Each inherited facet contains the
///   members of the entity that was inherited from.
///
/// * An *extension facet* for each `extension` declaration that
///   is known to apply to the entity in the context where semantic
///   checking is being performed. Each extension facet contains the
///   members of the `extension` that applied.
///
struct Facet
{
public:
    /// Kinds of facets that can occur
    enum class Kind
    {
        Type,
        Extension,
    };

    /// How many indirections away from the self facet?
    typedef unsigned int DirectnessVal;
    enum class Directness : DirectnessVal
    {
        Self = 0,
        Direct = 1,
    };

    /// The *origin* of a facet is the type and/or declaration
    /// that the facet's members belong to.
    ///
    struct Origin
    {
        /// A `DeclRef` to the declaration this facet corresponds to, if any.
        ///
        /// This might be a type declaration, an `extension` declaration,
        /// or nothing.
        ///
        DeclRef<Decl> declRef;

        /// The type that this facet corresponds to, if any
        Type* type = nullptr;

        Origin() {}

        explicit Origin(DeclRef<Decl> declRef, Type* type = nullptr)
            : declRef(declRef), type(type)
        {
        }
    };

    Facet() {}

    typedef FacetImpl Impl;

    Facet(Impl* impl)
        : _impl(impl)
    {
    }

    Impl* getImpl() const { return _impl; }
    Impl* operator->() const { return _impl; }

private:
    Impl* _impl = nullptr;
};


/// Do the origins of `left` and `right` match,
/// such that they are both facets for the same
/// base type or `extension`?
///
bool originsMatch(Facet left, Facet right);

inline bool operator!(Facet facet)
{
    return !facet.getImpl();
}

bool operator==(Facet::Origin left, Facet::Origin right);

inline bool operator!=(Facet::Origin left, Facet::Origin right)
{
    return !(left == right);
}

/// Heap-allocated implementation of a single facet.
struct FacetImpl
{
    /// The kind of this facet
    Facet::Kind kind = Facet::Kind::Type;

    /// How many indirections away from the self facet?
    Facet::Directness directness = Facet::Directness::Self;

    /// The origin of this facet.
    ///
    /// This is the type or declaration that the facet
    /// corresponds to.
    ///
    Facet::Origin origin;

    Type* getType() const { return origin.type; }
    DeclRef<Decl> getDeclRef() const { return origin.declRef; }

    /// A witness that the type this facet belongs to
    /// is a subtype of `origin.type` (if both of those
    /// types exist).
    ///
    SubtypeWitness* subtypeWitness = nullptr;

    /// The next facet in the linearized inheritance list of the entity.
    Facet next;

    FacetImpl() {}

    FacetImpl(
        Facet::Kind kind,
        Facet::Directness directness,
        DeclRef<Decl> declRef,
        Type* type,
        SubtypeWitness* subtypeWitness)
        : kind(kind), directness(directness), origin(declRef, type), subtypeWitness(subtypeWitness)
    {
    }
};

struct FacetListBuilder;

/// A singly linked list of facets.
struct FacetList
{
public:
    FacetList() {}

    explicit FacetList(Facet head)
        : _head(head)
    {
    }

    Facet getHead() const { return _head; }
    Facet& getHead() { return _head; }

    Facet advanceHead()
    {
        SLANG_ASSERT(_head.getImpl());
        auto facet = _head;
        _head = facet->next;
        return facet;
    }

    Facet popHead()
    {
        auto facet = advanceHead();
        facet->next = nullptr;
        return facet;
    }

    FacetList getTail() const
    {
        SLANG_ASSERT(_head.getImpl());
        return FacetList(_head->next);
    }

    bool containsMatchFor(Facet facet) const;

    bool isEmpty() const { return _head.getImpl() == nullptr; }

    struct Iterator
    {
    public:
        Iterator() {}

        Iterator(Facet::Impl* cursor)
            : _cursor(cursor)
        {
        }

        bool operator!=(Iterator const& that) const { return this->_cursor != that._cursor; }

        void operator++()
        {
            SLANG_ASSERT(_cursor);
            _cursor = _cursor->next.getImpl();
        }

        Facet operator*() const { return _cursor; }

    private:
        Facet::Impl* _cursor = nullptr;
    };

    Iterator begin() const { return Iterator(_head.getImpl()); }
    Iterator end() const { return Iterator(); }

    struct Appender
    {
    public:
        Appender(FacetList& list) { _link = &list._head; }

        void add(Facet facet)
        {
            *_link = facet;
            _link = &facet->next;
        }

    protected:
        Appender() {}

        Facet* _link = nullptr;
    };

    typedef FacetListBuilder Builder;

protected:
    Facet _head;
};

struct FacetListBuilder : FacetList, FacetList::Appender
{
public:
    FacetListBuilder() { _link = &_head; }
};

/// Information about the inheritance of an entity (type or declaration)
///
/// Currently this is only used to store a linearized list of the
/// `Facet`s that the type/declaration transitively inherits.
///
struct InheritanceInfo
{
    FacetList facets;
};

/// Cached information about how to convert between two types.
struct ImplicitCastMethod
{
    OverloadCandidate conversionFuncOverloadCandidate = OverloadCandidate();
    ConversionCost cost = kConversionCost_Impossible;
    bool isAmbiguous = false;
};

struct ImplicitCastMethodKey
{
    Type* fromType; // nullptr means default construct.
    bool isLValue;
    Type* toType;
    uint64_t constantVal;
    bool isConstant;
    HashCode getHashCode() const
    {
        return combineHash(
            Slang::getHashCode(fromType),
            Slang::getHashCode(toType),
            Slang::getHashCode(constantVal),
            (HashCode32)isConstant,
            (HashCode32)isLValue);
    }
    bool operator==(const ImplicitCastMethodKey& other) const
    {
        return fromType == other.fromType && toType == other.toType &&
               isConstant == other.isConstant && constantVal == other.constantVal &&
               isLValue == other.isLValue;
    }
    ImplicitCastMethodKey() = default;
    ImplicitCastMethodKey(QualType fromType, Type* toType, Expr* fromExpr)
        : fromType(fromType)
        , toType(toType)
        , constantVal(0)
        , isConstant(false)
        , isLValue(fromType.isLeftValue)
    {
        if (auto constInt = as<IntegerLiteralExpr>(fromExpr))
        {
            constantVal = constInt->value;
            isConstant = true;
        }
    }
};

/// Used to track offsets for atomic counter storage qualifiers.
struct GLSLBindingOffsetTracker
{
public:
    void setBindingOffset(int binding, int64_t byteOffset);
    int64_t getNextBindingOffset(int binding);

private:
    Dictionary<int, int64_t> bindingToByteOffset;
};

/// Shared state for a semantics-checking session.
struct SharedSemanticsContext : public RefObject
{
    Linkage* m_linkage = nullptr;

    /// The (optional) "primary" module that is the parent to everything that will be checked.
    Module* m_module = nullptr;

    DiagnosticSink* m_sink = nullptr;

    // Whether the current module has imported the GLSL module.
    ModuleDecl* glslModuleDecl = nullptr;

    /// (optional) modules that comes from previously processed translation units in the
    /// front-end request that are made visible to the module being checked. This allows
    /// `import` to use them instead of trying to find the files in file system.
    LoadedModuleDictionary* m_environmentModules = nullptr;

    /// (optional) The translation unit that is being checked.
    /// Needed for handling `__include`s.
    TranslationUnitRequest* m_translationUnitRequest = nullptr;

    DiagnosticSink* getSink() { return m_sink; }

    CompilerOptionSet& getOptionSet() { return m_linkage->m_optionSet; }

    // We need to track what has been `import`ed into
    // the scope of this semantic checking session,
    // and also to avoid importing the same thing more
    // than once.
    //
    List<ModuleDecl*> importedModulesList;
    HashSet<ModuleDecl*> importedModulesSet;

    GLSLBindingOffsetTracker m_glslBindingOffsetTracker;

public:
    SharedSemanticsContext(
        Linkage* linkage,
        Module* module,
        DiagnosticSink* sink,
        LoadedModuleDictionary* environmentModules = nullptr,
        TranslationUnitRequest* translationUnit = nullptr)
        : m_linkage(linkage)
        , m_module(module)
        , m_sink(sink)
        , m_environmentModules(environmentModules)
        , m_translationUnitRequest(translationUnit)
    {
    }

    Session* getSession() { return m_linkage->getSessionImpl(); }

    Linkage* getLinkage() { return m_linkage; }

    Module* getModule() { return m_module; }

    TranslationUnitRequest* getTranslationUnitRequest() { return m_translationUnitRequest; }

    bool isInLanguageServer()
    {
        if (m_linkage)
            return m_linkage->isInLanguageServer();
        return false;
    }
    /// Get the list of extension declarations that appear to apply to `decl` in this context
    List<ExtensionDecl*> const& getCandidateExtensionsForTypeDecl(AggTypeDecl* decl);

    /// Register a candidate extension `extDecl` for `typeDecl` encountered during checking.
    void registerCandidateExtension(AggTypeDecl* typeDecl, ExtensionDecl* extDecl);

    void registerAssociatedDecl(Decl* original, DeclAssociationKind assoc, Decl* declaration);

    List<RefPtr<DeclAssociation>> const& getAssociatedDeclsForDecl(Decl* decl);

    bool isDifferentiableFunc(FunctionDeclBase* func);
    bool isBackwardDifferentiableFunc(FunctionDeclBase* func);
    FunctionDifferentiableLevel _getFuncDifferentiableLevelImpl(
        FunctionDeclBase* func,
        int recurseLimit);
    FunctionDifferentiableLevel getFuncDifferentiableLevel(FunctionDeclBase* func);

    struct InheritanceCircularityInfo
    {
        InheritanceCircularityInfo(Decl* decl, InheritanceCircularityInfo* next)
            : decl(decl), next(next)
        {
        }

        /// A declaration whose inheritance is being calculated
        Decl* decl = nullptr;

        /// The rest of the links in the chain of declarations being processed
        InheritanceCircularityInfo* next = nullptr;
    };

    GLSLBindingOffsetTracker* getGLSLBindingOffsetTracker() { return &m_glslBindingOffsetTracker; }

    /// Get the processed inheritance information for `type`, including all its facets
    InheritanceInfo getInheritanceInfo(
        Type* type,
        InheritanceCircularityInfo* circularityInfo = nullptr);

    /// Get the processed inheritance information for `extension`, including all its facets
    InheritanceInfo getInheritanceInfo(
        DeclRef<ExtensionDecl> const& extension,
        InheritanceCircularityInfo* circularityInfo = nullptr);

    /// Prevent an unsupported case of
    /// ```
    ///     extension<T:IFoo> : IBar{};
    ///     extesnion<T:IBar> : IFoo{};
    /// ```
    /// from causing infinite recursion.
    bool _checkForCircularityInExtensionTargetType(
        Decl* decl,
        InheritanceCircularityInfo* circularityInfo);

    /// Try get subtype witness from cache, returns true if cache contains a result for the query.
    bool tryGetSubtypeWitnessFromCache(Type* sub, Type* sup, SubtypeWitness*& outWitness)
    {
        auto pair = TypePair{sub, sup};
        return m_mapTypePairToSubtypeWitness.tryGetValue(pair, outWitness);
    }
    void cacheSubtypeWitness(Type* sub, Type* sup, SubtypeWitness*& outWitness)
    {
        auto pair = TypePair{sub, sup};
        m_mapTypePairToSubtypeWitness[pair] = outWitness;
    }
    ImplicitCastMethod* tryGetImplicitCastMethod(ImplicitCastMethodKey key)
    {
        return m_mapTypePairToImplicitCastMethod.tryGetValue(key);
    }
    void cacheImplicitCastMethod(ImplicitCastMethodKey key, ImplicitCastMethod candidate)
    {
        m_mapTypePairToImplicitCastMethod[key] = candidate;
    }

    bool* isCStyleType(Type* type) { return m_isCStyleTypeCache.tryGetValue(type); }

    void cacheCStyleType(Type* type, bool isCStyle)
    {
        m_isCStyleTypeCache.addIfNotExists(type, isCStyle);
    }
    // Get the inner most generic decl that a decl-ref is dependent on.
    // For example, `Foo<T>` depends on the generic decl that defines `T`.
    //
    DeclRef<GenericDecl> getDependentGenericParent(DeclRef<Decl> declRef);

private:
    /// Mapping from type declarations to the known extensiosn that apply to them
    Dictionary<AggTypeDecl*, RefPtr<CandidateExtensionList>> m_mapTypeDeclToCandidateExtensions;

    /// Is the `m_mapTypeDeclToCandidateExtensions` dictionary valid and up to date?
    bool m_candidateExtensionListsBuilt = false;

    /// Add candidate extensions declared in `moduleDecl` to `m_mapTypeDeclToCandidateExtensions`
    void _addCandidateExtensionsFromModule(ModuleDecl* moduleDecl);

    /// Mapping from a decl to additional declarations of the same decl.
    /// The additional declarations provide a location to hold extra decorations.
    OrderedDictionary<Decl*, RefPtr<DeclAssociationList>> m_mapDeclToAssociatedDecls;

    /// Is the `m_mapDeclToAssociatedDecls` dictionary valid and up to date?
    bool m_associatedDeclListsBuilt = false;

    /// Add associated decls declared in `moduleDecl` to `m_mapDeclToAssociatedDecls`
    void _addDeclAssociationsFromModule(ModuleDecl* moduleDecl);

    ASTBuilder* _getASTBuilder() { return m_linkage->getASTBuilder(); }

    InheritanceInfo _getInheritanceInfo(
        DeclRef<Decl> declRef,
        Type* selfType,
        InheritanceCircularityInfo* circularityInfo);
    InheritanceInfo _calcInheritanceInfo(Type* type, InheritanceCircularityInfo* circularityInfo);
    InheritanceInfo _calcInheritanceInfo(
        DeclRef<Decl> declRef,
        Type* selfType,
        InheritanceCircularityInfo* circularityInfo);

    void getDependentGenericParentImpl(DeclRef<GenericDecl>& genericParent, DeclRef<Decl> declRef);

    struct DirectBaseInfo
    {
        FacetList facets;

        Facet::Impl facetImpl;

        DirectBaseInfo* next = nullptr;
    };

    struct DirectBaseListBuilder;

    struct DirectBaseList
    {
    public:
        struct Iterator
        {
        public:
            Iterator() {}

            Iterator(DirectBaseInfo* cursor)
                : _cursor(cursor)
            {
            }

            bool operator!=(Iterator that) const { return _cursor != that._cursor; }

            void operator++()
            {
                SLANG_ASSERT(_cursor);
                _cursor = _cursor->next;
            }

            DirectBaseInfo* operator*() { return _cursor; }

        private:
            DirectBaseInfo* _cursor = nullptr;
        };

        Iterator begin() const { return Iterator(_head); }
        Iterator end() const { return Iterator(); }

        bool isEmpty() const { return _head == nullptr; }

        bool doesAnyTailContainMatchFor(Facet facet) const;

        void removeEmptyLists();

        typedef DirectBaseListBuilder Builder;

    public:
        DirectBaseInfo* _head = nullptr;
    };

    struct DirectBaseListBuilder : DirectBaseList
    {
    public:
        DirectBaseListBuilder() { _link = &_head; }

        void add(DirectBaseInfo* base)
        {
            *_link = base;
            _link = &base->next;
        }

    private:
        DirectBaseInfo** _link = nullptr;
    };

    void _mergeFacetLists(
        DirectBaseList bases,
        FacetList baseFacets,
        FacetList::Builder& ioMergedFacets);

    struct TypePair
    {
        Type* type0;
        Type* type1;
        HashCode getHashCode() const
        {
            return combineHash(Slang::getHashCode(type0), Slang::getHashCode(type1));
        }
        bool operator==(const TypePair& other) const
        {
            return type0 == other.type0 && type1 == other.type1;
        }
    };
    Dictionary<Type*, InheritanceInfo> m_mapTypeToInheritanceInfo;
    Dictionary<DeclRef<Decl>, InheritanceInfo> m_mapDeclRefToInheritanceInfo;
    Dictionary<TypePair, SubtypeWitness*> m_mapTypePairToSubtypeWitness;
    Dictionary<ImplicitCastMethodKey, ImplicitCastMethod> m_mapTypePairToImplicitCastMethod;
    Dictionary<Type*, bool> m_isCStyleTypeCache;
};

/// Local/scoped state of the semantic-checking system
///
/// This type is kept distinct from `SharedSemanticsContext` so that we
/// can avoid unncessary mutable state being propagated through the
/// checking process.
///
/// Semantic-checking code should make a new local `SemanticsContext`
/// in cases where it want to check a sub-entity (expression, statement,
/// declaration, etc.) in a modified or extended context.
///
struct SemanticsContext
{
public:
    friend struct OuterScopeContextRAII;

    explicit SemanticsContext(SharedSemanticsContext* shared)
        : m_shared(shared)
        , m_sink(shared->getSink())
        , m_astBuilder(shared->getLinkage()->getASTBuilder())
    {
        if (shared->getLinkage()->m_optionSet.hasOption(CompilerOptionName::DisableShortCircuit))
        {
            m_shouldShortCircuitLogicExpr = !shared->getLinkage()->m_optionSet.getBoolOption(
                CompilerOptionName::DisableShortCircuit);
        }
    }

    SharedSemanticsContext* getShared() { return m_shared; }
    CompilerOptionSet& getOptionSet() { return getShared()->getOptionSet(); }
    ASTBuilder* getASTBuilder() { return m_astBuilder; }

    DiagnosticSink* getSink() { return m_sink; }

    Session* getSession() { return m_shared->getSession(); }

    Linkage* getLinkage() { return m_shared->m_linkage; }
    NamePool* getNamePool() { return getLinkage()->getNamePool(); }
    SourceManager* getSourceManager() { return getLinkage()->getSourceManager(); }

    SemanticsContext withSink(DiagnosticSink* sink)
    {
        SemanticsContext result(*this);
        result.m_sink = sink;
        return result;
    }

    FunctionDeclBase* getParentFuncOfVisitor() { return m_parentFunc; }
    void setParentFuncOfVisitor(FunctionDeclBase* funcDecl) { m_parentFunc = funcDecl; }

    SemanticsContext withParentFunc(FunctionDeclBase* parentFunc)
    {
        SemanticsContext result(*this);
        result.m_parentFunc = parentFunc;
        result.m_outerStmts = nullptr;
        result.m_parentDifferentiableAttr = parentFunc->findModifier<DifferentiableAttribute>();
        if (parentFunc->ownedScope)
            result.m_outerScope = parentFunc->ownedScope;
        return result;
    }

    SemanticsContext withParentExpandExpr(ExpandExpr* expr, OrderedHashSet<Type*>* capturedTypes)
    {
        SemanticsContext result(*this);
        result.m_parentExpandExpr = expr;
        result.m_capturedTypePacks = capturedTypes;
        return result;
    }

    SemanticsContext withParentLambdaExpr(
        LambdaExpr* expr,
        LambdaDecl* decl,
        Dictionary<Decl*, VarDeclBase*>* mapSrcDeclToCapturedLambdaDecl)
    {
        SemanticsContext result(*this);
        result.m_parentLambdaExpr = expr;
        result.m_mapSrcDeclToCapturedLambdaDecl = mapSrcDeclToCapturedLambdaDecl;
        result.m_parentLambdaDecl = decl;
        return result;
    }

    /// Information for tracking one or more outer statements.
    ///
    /// During checking of statements, we need to track what
    /// outer statements are in scope, so that we can resolve
    /// the target for a `break` or `continue` statement (and
    /// validate that such statements are only used in contexts
    /// where such a target exists).
    ///
    /// We use a linked list of `OuterStmtInfo` threaded up
    /// through the recursive call stack to track the statements
    /// that are lexically surrounding the one we are checking.
    ///
    struct OuterStmtInfo
    {
        Stmt* stmt = nullptr;
        OuterStmtInfo* next;
    };

    OuterStmtInfo* getOuterStmts() { return m_outerStmts; }

    SemanticsContext withOuterStmts(OuterStmtInfo* outerStmts)
    {
        SemanticsContext result(*this);
        result.m_outerStmts = outerStmts;
        return result;
    }

    // Setup the flag to indicate disabling the short-circuiting evaluation
    // for the logical expressions associted with the subcontext
    SemanticsContext disableShortCircuitLogicalExpr()
    {
        SemanticsContext result(*this);
        result.m_shouldShortCircuitLogicExpr = false;
        return result;
    }

    TryClauseType getEnclosingTryClauseType() { return m_enclosingTryClauseType; }

    SemanticsContext withEnclosingTryClauseType(TryClauseType tryClauseType)
    {
        SemanticsContext result(*this);
        result.m_enclosingTryClauseType = tryClauseType;
        return result;
    }

    DifferentiableAttribute* getParentDifferentiableAttribute()
    {
        return m_parentDifferentiableAttr;
    }

    /// A scope that is local to a particular expression, and
    /// that can be used to allocate temporary bindings that
    /// might be needed by that expression or its sub-expressions.
    ///
    /// The scope is represented as a sequence of nested `LetExpr`s
    /// that introduce the bindings needed in the scope.
    ///
    struct ExprLocalScope
    {
    public:
        void addBinding(LetExpr* binding);

        LetExpr* getOuterMostBinding() const { return m_outerMostBinding; }

    private:
        LetExpr* m_outerMostBinding = nullptr;
        LetExpr* m_innerMostBinding = nullptr;
    };

    ExprLocalScope* getExprLocalScope() { return m_exprLocalScope; }
    Scope* getOuterScope() { return m_outerScope; }

    SemanticsContext withExprLocalScope(ExprLocalScope* exprLocalScope)
    {
        SemanticsContext result(*this);
        result.m_exprLocalScope = exprLocalScope;
        return result;
    }

    SemanticsContext withOuterScope(Scope* scope)
    {
        SemanticsContext result(*this);
        result.m_outerScope = scope;
        return result;
    }

    SemanticsContext withTreatAsDifferentiable(TreatAsDifferentiableExpr* expr)
    {
        SemanticsContext result(*this);
        result.m_treatAsDifferentiableExpr = expr;
        return result;
    }

    SemanticsContext allowStaticReferenceToNonStaticMember()
    {
        SemanticsContext result(*this);
        result.m_allowStaticReferenceToNonStaticMember = true;
        return result;
    }

    SemanticsContext withDeclToExcludeFromLookup(Decl* decl)
    {
        SemanticsContext result(*this);
        result.m_declToExcludeFromLookup = decl;
        return result;
    }

    Decl* getDeclToExcludeFromLookup() { return m_declToExcludeFromLookup; }

    SemanticsContext excludeTransparentMembersFromLookup()
    {
        SemanticsContext result(*this);
        result.m_excludeTransparentMembersFromLookup = true;
        return result;
    }

    bool getExcludeTransparentMembersFromLookup() { return m_excludeTransparentMembersFromLookup; }

    OrderedHashSet<Type*>* getCapturedTypePacks() { return m_capturedTypePacks; }

    GLSLBindingOffsetTracker* getGLSLBindingOffsetTracker()
    {
        return m_shared->getGLSLBindingOffsetTracker();
    }

private:
    SharedSemanticsContext* m_shared = nullptr;

    DiagnosticSink* m_sink = nullptr;

    ExprLocalScope* m_exprLocalScope = nullptr;

    Decl* m_declToExcludeFromLookup = nullptr;

    bool m_excludeTransparentMembersFromLookup = false;

protected:
    // TODO: consider making more of this state `private`...

    /// The parent function (if any) that surrounds the statement being checked.
    FunctionDeclBase* m_parentFunc = nullptr;

    DifferentiableAttribute* m_parentDifferentiableAttr = nullptr;

    /// The linked list of lexically surrounding statements.
    OuterStmtInfo* m_outerStmts = nullptr;

    /// The type of a try clause (if any) enclosing current expr.
    TryClauseType m_enclosingTryClauseType = TryClauseType::None;

    /// Whether an expr referencing to a non-static member in static style (e.g. `Type.member`)
    /// is considered valid in the current context.
    bool m_allowStaticReferenceToNonStaticMember = false;

    /// Whether or not we are in a `no_diff` environment (and therefore should treat the call to
    /// a non-differentiable function as differentiable and not issue a diagnostic).
    TreatAsDifferentiableExpr* m_treatAsDifferentiableExpr = nullptr;

    ASTBuilder* m_astBuilder = nullptr;

    Scope* m_outerScope = nullptr;

    // By default, we will support short-circuit evaluation for the logic expression.
    // However, there are few exceptions where we will disable it:
    // 1. the logic expression is inside the generic parameter list.
    // 2. the logic expression is in the init expression of a static const variable.
    // 3. the logic expression is in an array size declaration.
    bool m_shouldShortCircuitLogicExpr = true;

    ExpandExpr* m_parentExpandExpr = nullptr;

    OrderedHashSet<Type*>* m_capturedTypePacks = nullptr;

    // If we are checking inside a lambda expression, we need
    // to track the referenced variables that should be captured
    // by the lambda.
    LambdaExpr* m_parentLambdaExpr = nullptr;
    LambdaDecl* m_parentLambdaDecl = nullptr;
    Dictionary<Decl*, VarDeclBase*>* m_mapSrcDeclToCapturedLambdaDecl = nullptr;
};

struct OuterScopeContextRAII
{
    SemanticsContext* m_context;
    Scope* m_oldOuterScope;

    OuterScopeContextRAII(SemanticsContext* context, Scope* outerScope)
        : m_context(context), m_oldOuterScope(context->getOuterScope())
    {
        context->m_outerScope = outerScope;
    }

    ~OuterScopeContextRAII() { m_context->m_outerScope = m_oldOuterScope; }
};

#define SLANG_OUTER_SCOPE_CONTEXT_RAII(context, scope) \
    OuterScopeContextRAII _outerScopeContextRAII(context, scope)
#define SLANG_OUTER_SCOPE_CONTEXT_DECL_RAII(context, decl) \
    OuterScopeContextRAII _outerScopeContextRAII(          \
        context,                                           \
        decl->ownedScope ? decl->ownedScope : context->getOuterScope())

struct RequirementSynthesisResult
{
    bool suceeded = false;
    operator bool() const { return suceeded; }
};

struct SemanticsVisitor : public SemanticsContext
{
    typedef SemanticsContext Super;

    explicit SemanticsVisitor(SharedSemanticsContext* shared)
        : Super(shared)
    {
    }

    SemanticsVisitor(SemanticsContext const& context)
        : Super(context)
    {
    }

    CompilerOptionSet& getOptionSet() { return getShared()->getOptionSet(); }

public:
    // Translate Types


    Expr* TranslateTypeNodeImpl(Expr* node);
    Type* ExtractTypeFromTypeRepr(Expr* typeRepr);
    Type* TranslateTypeNode(Expr* node);
    TypeExp TranslateTypeNodeForced(TypeExp const& typeExp);
    TypeExp TranslateTypeNode(TypeExp const& typeExp);
    Type* getRemovedModifierType(ModifiedType* type, ModifierVal* modifier);
    Type* getConstantBufferType(Type* elementType, Type* layoutType);

    DeclRefType* getExprDeclRefType(Expr* expr);

    /// Is `decl` usable as a static member?
    bool isDeclUsableAsStaticMember(Decl* decl);

    /// Is `item` usable as a static member?
    bool isUsableAsStaticMember(LookupResultItem const& item);

    /// Move `expr` into a temporary variable and execute `func` on that variable.
    ///
    /// Returns an expression that wraps both the creation and initialization of
    /// the temporary, and the computation created by `func`.
    ///
    template<typename F>
    Expr* moveTemp(Expr* const& expr, F const& func);

    /// Execute `func` on a variable with the value of `expr`.
    ///
    /// If `expr` is just a reference to an immutable (e.g., `let`) variable
    /// then this might use the existing variable. Otherwise it will create
    /// a new variable to hold `expr`, using `moveTemp()`.
    ///
    template<typename F>
    Expr* maybeMoveTemp(Expr* const& expr, F const& func);

    /// Return an expression that represents "opening" the existential `expr`.
    ///
    /// The type of `expr` must be an interface type, matching `interfaceDeclRef`.
    ///
    /// If we scope down the PL theory to just the case that Slang cares about,
    /// a value of an existential type like `IMover` is a tuple of:
    ///
    ///  * a concrete type `X`
    ///  * a witness `w` of the fact that `X` implements `IMover`
    ///  * a value `v` of type `X`
    ///
    /// "Opening" an existential value is the process of decomposing a single
    /// value `e : IMover` into the pieces `X`, `w`, and `v`.
    ///
    /// Rather than return all those pieces individually, this operation
    /// returns an expression that logically corresponds to `v`: an expression
    /// of type `X`, where the type carries the knowledge that `X` implements `IMover`.
    ///
    Expr* openExistential(Expr* expr, DeclRef<InterfaceDecl> interfaceDeclRef);

    /// If `expr` has existential type, then open it.
    ///
    /// Returns an expression that opens `expr` if it had existential type.
    /// Otherwise returns `expr` itself.
    ///
    /// See `openExistential` for a discussion of what "opening" an
    /// existential-type value means.
    ///
    Expr* maybeOpenExistential(Expr* expr);

    /// If `expr` has Ref<T> Type, convert it into an l-value expr that has T type.
    Expr* maybeOpenRef(Expr* expr);

    Scope* getScope(SyntaxNode* node);

    void diagnoseDeprecatedDeclRefUsage(DeclRef<Decl> declRef, SourceLoc loc, Expr* originalExpr);

    DeclRef<Decl> getDefaultDeclRef(Decl* decl)
    {
        return createDefaultSubstitutionsIfNeeded(m_astBuilder, this, makeDeclRef(decl));
    }

    DeclRef<Decl> getSpecializedDeclRef(
        DeclRef<Decl> declToSpecialize,
        DeclRef<Decl> declRefWithSpecializationArgs)
    {
        return declRefWithSpecializationArgs.substitute(m_astBuilder, declToSpecialize);
    }

    DeclRef<Decl> getSpecializedDeclRef(
        Decl* declToSpecialize,
        DeclRef<Decl> declRefWithSpecializationArgs)
    {
        return declRefWithSpecializationArgs.substitute(
            m_astBuilder,
            getDefaultDeclRef(declToSpecialize));
    }

    DeclRefExpr* ConstructDeclRefExpr(
        DeclRef<Decl> declRef,
        Expr* baseExpr,
        Name* name,
        SourceLoc loc,
        Expr* originalExpr);

    Expr* ConstructDerefExpr(Expr* base, SourceLoc loc);
    Expr* constructDerefExpr(Expr* base, QualType elementType, SourceLoc loc);

    InvokeExpr* constructUncheckedInvokeExpr(Expr* callee, const List<Expr*>& arguments);

    Expr* maybeUseSynthesizedDeclForLookupResult(LookupResultItem const& item, Expr* orignalExpr);

    Expr* ConstructLookupResultExpr(
        LookupResultItem const& item,
        Expr* baseExpr,
        Name* name,
        SourceLoc loc,
        Expr* originalExpr);

    Expr* createLookupResultExpr(
        Name* name,
        LookupResult const& lookupResult,
        Expr* baseExpr,
        SourceLoc loc,
        Expr* originalExpr);

    DeclVisibility getTypeVisibility(Type* type);
    bool isDeclVisibleFromScope(DeclRef<Decl> declRef, Scope* scope);
    LookupResult filterLookupResultByVisibility(const LookupResult& lookupResult);
    LookupResult filterLookupResultByVisibilityAndDiagnose(
        const LookupResult& lookupResult,
        SourceLoc loc,
        bool& outDiagnosed);

    Val* resolveVal(Val* val)
    {
        if (!val)
            return nullptr;
        return val->resolve();
    }
    Type* resolveType(Type* type) { return (Type*)resolveVal(type); }
    DeclRef<Decl> resolveDeclRef(DeclRef<Decl> declRef);

    /// Attempt to "resolve" an overloaded `LookupResult` to only include the "best" results
    LookupResult resolveOverloadedLookup(LookupResult const& lookupResult);

    /// Attempt to resolve `expr` into an expression that refers to a single declaration/value.
    /// If `expr` isn't overloaded, then it will be returned as-is.
    ///
    /// The provided `mask` is used to filter items down to those that are applicable in a given
    /// context (e.g., just types).
    ///
    /// If the expression cannot be resolved to a single value then *if* `diagSink` is non-null an
    /// appropriate "ambiguous reference" error will be reported, and an error expression will be
    /// returned. Otherwise, the original expression is returned if resolution fails.
    ///
    Expr* maybeResolveOverloadedExpr(Expr* expr, LookupMask mask, DiagnosticSink* diagSink);

    /// Attempt to resolve `overloadedExpr` into an expression that refers to a single
    /// declaration/value.
    ///
    /// Equivalent to `maybeResolveOverloadedExpr` with `diagSink` bound to the sink for the
    /// `SemanticsVisitor`.
    Expr* resolveOverloadedExpr(OverloadedExpr* overloadedExpr, LookupMask mask);

    /// Worker reoutine for `maybeResolveOverloadedExpr` and `resolveOverloadedExpr`.
    Expr* _resolveOverloadedExprImpl(
        OverloadedExpr* overloadedExpr,
        LookupMask mask,
        DiagnosticSink* diagSink);

    void diagnoseAmbiguousReference(
        OverloadedExpr* overloadedExpr,
        LookupResult const& lookupResult);
    void diagnoseAmbiguousReference(Expr* overloadedExpr);


    Expr* ExpectATypeRepr(Expr* expr);

    Type* ExpectAType(Expr* expr);

    Type* ExtractGenericArgType(Expr* exp);

    IntVal* ExtractGenericArgInteger(Expr* exp, Type* genericParamType, DiagnosticSink* sink);
    IntVal* ExtractGenericArgInteger(Expr* exp, Type* genericParamType);

    Val* ExtractGenericArgVal(Expr* exp);

    // Construct a type representing the instantiation of
    // the given generic declaration for the given arguments.
    // The arguments should already be checked against
    // the declaration.
    Type* InstantiateGenericType(DeclRef<GenericDecl> genericDeclRef, List<Expr*> const& args);

    // These routines are bottlenecks for semantic checking,
    // so that we can add some quality-of-life features for users
    // in cases where the compiler crashes
    //
    void dispatchStmt(Stmt* stmt, SemanticsContext const& context);
    Expr* dispatchExpr(Expr* expr, SemanticsContext const& context);

    /// Ensure that a declaration has been checked up to some state
    /// (aka, a phase of semantic checking) so that we can safely
    /// perform certain operations on it.
    ///
    /// Calling `ensureDecl` may cause the type-checker to recursively
    /// start checking `decl` on top of the stack that is already
    /// doing other semantic checking. Care should be taken when relying
    /// on this function to avoid blowing out the stack or (even worse
    /// creating a circular dependency).
    ///
    void ensureDecl(Decl* decl, DeclCheckState state, SemanticsContext* baseContext = nullptr);

    /// Helper routine allowing `ensureDecl` to be called on a `DeclRef`
    void ensureDecl(DeclRefBase* declRef, DeclCheckState state)
    {
        ensureDecl(declRef->getDecl(), state);
    }

    void ensureAllDeclsRec(Decl* decl, DeclCheckState state);

    /// Helper routine allowing `ensureDecl` to be used on a `DeclBase`
    ///
    /// `DeclBase` is the base clas of `Decl` and `DeclGroup`. When
    /// called on a `DeclGroup` this function just calls `ensureDecl()`
    /// on each declaration in the group.
    ///
    void ensureDeclBase(DeclBase* decl, DeclCheckState state, SemanticsContext* baseContext);

    // A "proper" type is one that can be used as the type of an expression.
    // Put simply, it can be a concrete type like `int`, or a generic
    // type that is applied to arguments, like `Texture2D<float4>`.
    // The type `void` is also a proper type, since we can have expressions
    // that return a `void` result (e.g., many function calls).
    //
    // A "non-proper" type is any type that can't actually have values.
    // A simple example of this in C++ is `std::vector` - you can't have
    // a value of this type.
    //
    // Part of what this function does is give errors if somebody tries
    // to use a non-proper type as the type of a variable (or anything
    // else that needs a proper type).
    //
    // The other thing it handles is the fact that HLSL lets you use
    // the name of a non-proper type, and then have the compiler fill
    // in the default values for its type arguments (e.g., a variable
    // given type `Texture2D` will actually have type `Texture2D<float4>`).
    bool CoerceToProperTypeImpl(
        TypeExp const& typeExp,
        Type** outProperType,
        DiagnosticSink* diagSink);

    TypeExp CoerceToProperType(TypeExp const& typeExp);

    TypeExp tryCoerceToProperType(TypeExp const& typeExp);

    // Check a type, and coerce it to be proper
    TypeExp CheckProperType(TypeExp typeExp);

    // For our purposes, a "usable" type is one that can be
    // used to declare a function parameter, variable, etc.
    // These turn out to be all the proper types except
    // `void`.
    //
    // TODO(tfoley): consider just allowing `void` as a
    // simple example of a "unit" type, and get rid of
    // this check.
    TypeExp CoerceToUsableType(TypeExp const& typeExp, Decl* decl);

    // Check a type, and coerce it to be usable
    TypeExp CheckUsableType(TypeExp typeExp, Decl* decl);

    Expr* CheckTerm(Expr* term);

    Expr* _CheckTerm(Expr* term);

    Expr* CreateErrorExpr(Expr* expr);

    bool IsErrorExpr(Expr* expr);

    // Capture the "base" expression in case this is a member reference
    Expr* GetBaseExpr(Expr* expr);

    /// Validate a declaration to ensure that it doesn't introduce a circularly-defined constant
    ///
    /// Circular definition in a constant may lead to infinite looping or stack overflow in
    /// the compiler, so it needs to be protected against.
    ///
    /// Note that this function does *not* protect against circular definitions in general,
    /// and a program that indirectly initializes a global variable using its own value (e.g.,
    /// by calling a function that indirectly reads the variable) will be allowed and then
    /// exhibit undefined behavior at runtime.
    ///
    IntVal* _validateCircularVarDefinition(VarDeclBase* varDecl);

    bool shouldSkipChecking(Decl* decl, DeclCheckState state);

    // Auto-diff convenience functions for translating primal types to differential types.
    Type* _toDifferentialParamType(Type* primalType);

    Type* getDifferentialPairType(Type* primalType);

    // Convert a function's original type to it's forward/backward diff'd type.
    Type* getForwardDiffFuncType(FuncType* originalType);
    Type* getBackwardDiffFuncType(FuncType* originalType);

    /// Registers a type as conforming to IDifferentiable, along with a witness
    /// describing the relationship.
    ///
    void addDifferentiableTypeToDiffTypeRegistry(Type* type, SubtypeWitness* witness);
    void maybeRegisterDifferentiableTypeImplRecursive(ASTBuilder* builder, Type* type);

    // Construct the differential for 'type', if it exists.
    Type* getDifferentialType(ASTBuilder* builder, Type* type, SourceLoc loc);
    Type* tryGetDifferentialType(ASTBuilder* builder, Type* type);

    // Helper function to check if a struct can be used as its own differential type.
    bool canStructBeUsedAsSelfDifferentialType(AggTypeDecl* aggTypeDecl);
    void markSelfDifferentialMembersOfType(AggTypeDecl* parent, Type* type);

    void checkDerivativeMemberAttributeReferences(
        VarDeclBase* varDecl,
        DerivativeMemberAttribute* derivativeMemberAttr);

public:
    bool ValuesAreEqual(IntVal* left, IntVal* right);

    // Compute the cost of using a particular declaration to
    // perform implicit type conversion.
    ConversionCost getImplicitConversionCost(Decl* decl);

    ConversionCost getImplicitConversionCostWithKnownArg(
        DeclRef<Decl> decl,
        Type* toType,
        Expr* arg);


    BuiltinConversionKind getImplicitConversionBuiltinKind(Decl* decl);

    bool isEffectivelyScalarForInitializerLists(Type* type);

    /// Should the provided expression (from an initializer list) be used directly to initialize
    /// `toType`?
    bool shouldUseInitializerDirectly(Type* toType, Expr* fromExpr);

    /// Read a value from an initializer list expression.
    ///
    /// This reads one or more argument from the initializer list
    /// given as `fromInitializerListExpr` to initialize a value
    /// of type `toType`. This may involve reading one or
    /// more arguments from the initializer list, depending
    /// on whether `toType` is an aggregate or not, and on
    /// whether the next argument in the initializer list is
    /// itself an initializer list.
    ///
    /// This routine returns `true` if it was able to read
    /// arguments that can form a value of type `toType`,
    /// and `false` otherwise.
    ///
    /// If the routine succeeds and `outToExpr` is non-null,
    /// then it will be filled in with an expression
    /// representing the value (or type `toType`) that was read,
    /// or it will be left null to indicate that a default
    /// value should be used.
    ///
    /// If the routine fails and `outToExpr` is non-null,
    /// then a suitable diagnostic will be emitted.
    ///
    bool _readValueFromInitializerList(
        Type* toType,
        Expr** outToExpr,
        InitializerListExpr* fromInitializerListExpr,
        UInt& ioInitArgIndex);

    /// Read an aggregate value from an initializer list expression.
    ///
    /// This reads one or more arguments from the initializer list
    /// given as `fromInitializerListExpr` to initialize the
    /// fields/elements of a value of type `toType`.
    ///
    /// This routine returns `true` if it was able to read
    /// arguments that can form a value of type `toType`,
    /// and `false` otherwise.
    ///
    /// If the routine succeeds and `outToExpr` is non-null,
    /// then it will be filled in with an expression
    /// representing the value (or type `toType`) that was read,
    /// or it will be left null to indicate that a default
    /// value should be used.
    ///
    /// If the routine fails and `outToExpr` is non-null,
    /// then a suitable diagnostic will be emitted.
    ///
    bool _readAggregateValueFromInitializerList(
        Type* inToType,
        Expr** outToExpr,
        InitializerListExpr* fromInitializerListExpr,
        UInt& ioArgIndex);

    /// Coerce an initializer-list expression to a specific type.
    ///
    /// This reads one or more arguments from the initializer list
    /// given as `fromInitializerListExpr` to initialize the
    /// fields/elements of a value of type `toType`.
    ///
    /// This routine returns `true` if it was able to read
    /// arguments that can form a value of type `toType`,
    /// with no arguments left over, and `false` otherwise.
    ///
    /// If the routine succeeds and `outToExpr` is non-null,
    /// then it will be filled in with an expression
    /// representing the value (or type `toType`) that was read,
    /// or it will be left null to indicate that a default
    /// value should be used.
    ///
    /// If the routine fails and `outToExpr` is non-null,
    /// then a suitable diagnostic will be emitted.
    ///
    bool _coerceInitializerList(
        Type* toType,
        Expr** outToExpr,
        InitializerListExpr* fromInitializerListExpr);

    /// Report that implicit type coercion is not possible.
    bool _failedCoercion(Type* toType, Expr** outToExpr, Expr* fromExpr);

    /// Central engine for implementing implicit coercion logic
    ///
    /// This function tries to find an implicit conversion path from
    /// `fromType` to `toType`. It returns `true` if a conversion
    /// is found, and `false` if not.
    ///
    /// If a conversion is found, then its cost will be written to `outCost`.
    ///
    /// If a `fromExpr` is provided, it must be of type `fromType`,
    /// and represent a value to be converted.
    ///
    /// If `outToExpr` is non-null, and if a conversion is found, then
    /// `*outToExpr` will be set to an expression that performs the
    /// implicit conversion of `fromExpr` (which must be non-null
    /// to `toType`).
    ///
    /// The case where `outToExpr` is non-null is used to identify
    /// when a conversion is being done "for real" so that diagnostics
    /// should be emitted on failure.
    ///
    bool _coerce(
        CoercionSite site,
        Type* toType,
        Expr** outToExpr,
        QualType fromType,
        Expr* fromExpr,
        ConversionCost* outCost);

    /// Check whether implicit type coercion from `fromType` to `toType` is possible.
    ///
    /// If conversion is possible, returns `true` and sets `outCost` to the cost
    /// of the conversion found (if `outCost` is non-null).
    ///
    /// If conversion is not possible, returns `false`.
    ///
    bool canCoerce(Type* toType, QualType fromType, Expr* fromExpr, ConversionCost* outCost = 0);

    TypeCastExpr* createImplicitCastExpr();

    Expr* CreateImplicitCastExpr(Type* toType, Expr* fromExpr);

    /// Create an "up-cast" from a value to an interface type
    ///
    /// This operation logically constructs an "existential" value,
    /// which packages up the value, its type, and the witness
    /// of its conformance to the interface.
    ///
    Expr* createCastToInterfaceExpr(Type* toType, Expr* fromExpr, Val* witness);

    /// Implicitly coerce `fromExpr` to `toType` and diagnose errors if it isn't possible
    Expr* coerce(CoercionSite site, Type* toType, Expr* fromExpr);

    // Fill in default substitutions for the 'subtype' part of a type constraint decl
    void CheckConstraintSubType(TypeExp& typeExp);

    void checkGenericDeclHeader(GenericDecl* genericDecl);

    IntVal* checkLinkTimeConstantIntVal(Expr* expr);

    ConstantIntVal* checkConstantIntVal(Expr* expr);

    ConstantIntVal* checkConstantEnumVal(Expr* expr);

    // Check an expression, coerce it to the `String` type, and then
    // ensure that it has a literal (not just compile-time constant) value.
    bool checkLiteralStringVal(Expr* expr, String* outVal);

    bool checkCapabilityName(Expr* expr, CapabilityName& outCapabilityName);

    void visitModifier(Modifier*);

    DeclRef<VarDeclBase> tryGetIntSpecializationConstant(Expr* expr);

    AttributeDecl* lookUpAttributeDecl(Name* attributeName, Scope* scope);

    bool hasFloatArgs(Attribute* attr, int numArgs);
    bool hasIntArgs(Attribute* attr, int numArgs);
    bool hasStringArgs(Attribute* attr, int numArgs);

    bool getAttributeTargetSyntaxClasses(SyntaxClass<NodeBase>& cls, uint32_t typeFlags);

    // Check an attribute, and return a checked modifier that represents it.
    //
    Modifier* validateAttribute(
        Attribute* attr,
        AttributeDecl* attribClassDecl,
        ModifiableSyntaxNode* attrTarget);

    AttributeBase* checkAttribute(
        UncheckedAttribute* uncheckedAttr,
        ModifiableSyntaxNode* attrTarget);

    AttributeBase* checkGLSLLayoutAttribute(
        UncheckedGLSLLayoutAttribute* uncheckedAttr,
        ModifiableSyntaxNode* attrTarget);

    Modifier* checkModifier(
        Modifier* m,
        ModifiableSyntaxNode* syntaxNode,
        bool ignoreUnallowedModifier);

    void checkModifiers(ModifiableSyntaxNode* syntaxNode);
    void checkVisibility(Decl* decl);

    bool doesSignatureMatchRequirement(
        DeclRef<CallableDecl> satisfyingMemberDeclRef,
        DeclRef<CallableDecl> requiredMemberDeclRef,
        RefPtr<WitnessTable> witnessTable);

    bool doesAccessorMatchRequirement(
        DeclRef<AccessorDecl> satisfyingMemberDeclRef,
        DeclRef<AccessorDecl> requiredMemberDeclRef);

    bool doesPropertyMatchRequirement(
        DeclRef<PropertyDecl> satisfyingMemberDeclRef,
        DeclRef<PropertyDecl> requiredMemberDeclRef,
        RefPtr<WitnessTable> witnessTable);

    bool doesSubscriptMatchRequirement(
        DeclRef<SubscriptDecl> satisfyingMemberDeclRef,
        DeclRef<SubscriptDecl> requiredMemberDeclRef,
        RefPtr<WitnessTable> witnessTable);

    bool doesVarMatchRequirement(
        DeclRef<VarDeclBase> satisfyingMemberDeclRef,
        DeclRef<VarDeclBase> requiredMemberDeclRef,
        RefPtr<WitnessTable> witnessTable);

    bool doesGenericSignatureMatchRequirement(
        DeclRef<GenericDecl> genDecl,
        DeclRef<GenericDecl> requirementGenDecl,
        RefPtr<WitnessTable> witnessTable);

    bool doesTypeSatisfyAssociatedTypeConstraintRequirement(
        Type* satisfyingType,
        DeclRef<AssocTypeDecl> requiredAssociatedTypeDeclRef,
        RefPtr<WitnessTable> witnessTable);

    bool doesTypeSatisfyAssociatedTypeRequirement(
        Type* satisfyingType,
        DeclRef<AssocTypeDecl> requiredAssociatedTypeDeclRef,
        RefPtr<WitnessTable> witnessTable);

    // Does the given `memberDecl` work as an implementation
    // to satisfy the requirement `requiredMemberDeclRef`
    // from an interface?
    //
    // If it does, then inserts a witness into `witnessTable`
    // and returns `true`, otherwise returns `false`
    bool doesMemberSatisfyRequirement(
        DeclRef<Decl> memberDeclRef,
        DeclRef<Decl> requiredMemberDeclRef,
        RefPtr<WitnessTable> witnessTable);

    // State used while checking if a declaration (either a type declaration
    // or an extension of that type) conforms to the interfaces it claims
    // via its inheritance clauses.
    //
    struct ConformanceCheckingContext
    {
        /// The type for which conformances are being checked
        Type* conformingType;

        /// The outer declaration for the conformances being checked (either a type or `extension`
        /// declaration)
        ContainerDecl* parentDecl;

        // An inner diagnostic sink to store diagnostics about why requirement synthesis failed.
        DiagnosticSink innerSink;

        Dictionary<DeclRef<InterfaceDecl>, RefPtr<WitnessTable>> mapInterfaceToWitnessTable;
    };

    void addModifiersToSynthesizedDecl(
        ConformanceCheckingContext* context,
        DeclRef<Decl> requirement,
        CallableDecl* synthesized,
        ThisExpr*& synThis);

    void addRequiredParamsToSynthesizedDecl(
        DeclRef<CallableDecl> requirement,
        CallableDecl* synthesized,
        List<Expr*>& synArgs);

    CallableDecl* synthesizeMethodSignatureForRequirementWitnessInner(
        ConformanceCheckingContext* context,
        DeclRef<CallableDecl> requiredMemberDeclRef,
        List<Expr*>& synArgs,
        ThisExpr*& synThis);

    CallableDecl* synthesizeMethodSignatureForRequirementWitness(
        ConformanceCheckingContext* context,
        DeclRef<CallableDecl> requiredMemberDeclRef,
        List<Expr*>& synArgs,
        ThisExpr*& synThis);

    GenericDecl* synthesizeGenericSignatureForRequirementWitness(
        ConformanceCheckingContext* context,
        DeclRef<GenericDecl> requiredMemberDeclRef,
        List<Expr*>& synArgs,
        List<Expr*>& synGenericArgs,
        ThisExpr*& synThis);

    bool synthesizeAccessorRequirements(
        ConformanceCheckingContext* context,
        DeclRef<ContainerDecl> requiredMemberDeclRef,
        Type* resultType,
        Expr* synBoundStorageExpr,
        ContainerDecl* synAccesorContainer,
        RefPtr<WitnessTable> witnessTable);

    void _addMethodWitness(
        WitnessTable* witnessTable,
        DeclRef<CallableDecl> requirement,
        DeclRef<CallableDecl> method);

    /// Attempt to synthesize a method that can satisfy `requiredMemberDeclRef` using
    /// `lookupResult`.
    ///
    /// On success, installs the syntethesized method in `witnessTable` and returns `true`.
    /// Otherwise, returns `false`.
    bool trySynthesizeMethodRequirementWitness(
        ConformanceCheckingContext* context,
        LookupResult const& lookupResult,
        DeclRef<FuncDecl> requiredMemberDeclRef,
        RefPtr<WitnessTable> witnessTable);

    bool trySynthesizeConstructorRequirementWitness(
        ConformanceCheckingContext* context,
        LookupResult const& lookupResult,
        DeclRef<ConstructorDecl> requiredMemberDeclRef,
        RefPtr<WitnessTable> witnessTable);

    /// Attempt to synthesize a property that can satisfy `requiredMemberDeclRef` using
    /// `lookupResult`.
    ///
    /// On success, installs the syntethesized method in `witnessTable` and returns `true`.
    /// Otherwise, returns `false`.
    ///
    bool trySynthesizePropertyRequirementWitness(
        ConformanceCheckingContext* context,
        LookupResult const& lookupResult,
        DeclRef<PropertyDecl> requiredMemberDeclRef,
        RefPtr<WitnessTable> witnessTable);

    bool trySynthesizeWrapperTypePropertyRequirementWitness(
        ConformanceCheckingContext* context,
        DeclRef<PropertyDecl> requiredMemberDeclRef,
        RefPtr<WitnessTable> witnessTable);

    bool trySynthesizeSubscriptRequirementWitness(
        ConformanceCheckingContext* context,
        const LookupResult& lookupResult,
        DeclRef<SubscriptDecl> requiredMemberDeclRef,
        RefPtr<WitnessTable> witnessTable);

    bool trySynthesizeWrapperTypeSubscriptRequirementWitness(
        ConformanceCheckingContext* context,
        DeclRef<SubscriptDecl> requiredMemberDeclRef,
        RefPtr<WitnessTable> witnessTable);

    bool trySynthesizeAssociatedTypeRequirementWitness(
        ConformanceCheckingContext* context,
        LookupResult const& lookupResult,
        DeclRef<AssocTypeDecl> requiredMemberDeclRef,
        RefPtr<WitnessTable> witnessTable);

    bool trySynthesizeAssociatedConstantRequirementWitness(
        ConformanceCheckingContext* context,
        LookupResult const& lookupResult,
        DeclRef<VarDeclBase> requiredMemberDeclRef,
        RefPtr<WitnessTable> witnessTable);

    /// Attempt to synthesize a declartion that can satisfy `requiredMemberDeclRef` using
    /// `lookupResult`.
    ///
    /// On success, installs the syntethesized declaration in `witnessTable` and returns `true`.
    /// Otherwise, returns `false`.
    bool trySynthesizeRequirementWitness(
        ConformanceCheckingContext* context,
        LookupResult const& lookupResult,
        DeclRef<Decl> requiredMemberDeclRef,
        RefPtr<WitnessTable> witnessTable);


    enum SynthesisPattern
    {
        // Synthesized method inducts over all arguments.
        // T fn(T x, T y, T z, ...)
        // {
        //     typeof(T::member0)::fn(x.member0, y.member0, z.member0, ...);
        //     typeof(T::member1)::fn(x.member1, y.member1, z.member1, ...);
        //     ...
        // }
        //
        AllInductive,

        // Synthesized method inducts over all arguments except the first.
        // T fn(U x, T y, T z)
        // {
        //     typeof(T::member0)::fn(x, y.member0, z.member0, ...);
        //     typeof(T::member1)::fn(x, y.member1, z.member1, ...);
        //     ...
        // }
        FixedFirstArg
    };

    /// Attempt to synthesize `zero`, `dadd` & `dmul` methods for a type that conforms to
    /// `IDifferentiable`.
    /// On success, installs the syntethesized functions and returns `true`.
    /// Otherwise, returns `false`.
    bool trySynthesizeDifferentialMethodRequirementWitness(
        ConformanceCheckingContext* context,
        DeclRef<Decl> requirementDeclRef,
        RefPtr<WitnessTable> witnessTable,
        SynthesisPattern pattern);

    /// Attempt to synthesize an associated `Differential` type for a type that conforms to
    /// `IDifferentiable`.
    ///
    /// On success, installs the syntethesized type in `witnessTable`, injects `[DerivativeMember]`
    /// modifiers on differentiable fields to point to the corresponding field in the synthesized
    /// differential type, and returns `true`.
    /// Otherwise, returns `false`.
    bool trySynthesizeDifferentialAssociatedTypeRequirementWitness(
        ConformanceCheckingContext* context,
        DeclRef<AssocTypeDecl> requirementDeclRef,
        RefPtr<WitnessTable> witnessTable);

    /// Attempt to synthesize function requirements for enum types to make them conform to
    /// `ILogical`.
    bool trySynthesizeEnumTypeMethodRequirementWitness(
        ConformanceCheckingContext* context,
        DeclRef<FunctionDeclBase> requirementDeclRef,
        RefPtr<WitnessTable> witnessTable,
        BuiltinRequirementKind requirementKind);

    /// Check references from`[DerivativeMember(...)]` attributes on members of the agg-decl.
    /// this is typically deferred until after types are ready for reference.
    void checkDifferentiableMembersInType(AggTypeDecl* decl);

    struct DifferentiableMemberInfo
    {
        Decl* memberDecl;
        Type* diffType;
    };

    /// Gather differentiable members from decl.
    List<DifferentiableMemberInfo> collectDifferentiableMemberInfo(ContainerDecl* decl);

    // Check and register a type if it is differentiable.
    void maybeRegisterDifferentiableType(ASTBuilder* builder, Type* type);

    // Find the appropriate member of a declared type to
    // satisfy a requirement of an interface the type
    // claims to conform to.
    //
    // The type declaration `typeDecl` has declared that it
    // conforms to the interface `interfaceDeclRef`, and
    // `requiredMemberDeclRef` is a required member of
    // the interface.
    //
    // If a satisfying value is found, registers it in
    // `witnessTable` and returns `true`, otherwise
    // returns `false`.
    //
    bool findWitnessForInterfaceRequirement(
        ConformanceCheckingContext* context,
        Type* subType,
        Type* superInterfaceType,
        InheritanceDecl* inheritanceDecl,
        DeclRef<InterfaceDecl> superInterfaceDeclRef,
        DeclRef<Decl> requiredMemberDeclRef,
        RefPtr<WitnessTable> witnessTable,
        SubtypeWitness* subTypeConformsToSuperInterfaceWitness);

    // Check that the type declaration `typeDecl`, which
    // declares conformance to the interface `interfaceDeclRef`,
    // (via the given `inheritanceDecl`) actually provides
    // members to satisfy all the requirements in the interface.
    bool checkInterfaceConformance(
        ConformanceCheckingContext* context,
        Type* subType,
        Type* superInterfaceType,
        InheritanceDecl* inheritanceDecl,
        DeclRef<InterfaceDecl> superInterfaceDeclRef,
        SubtypeWitness* subTypeConformsToSuperInterfaceWitness,
        WitnessTable* witnessTable);

    RefPtr<WitnessTable> checkInterfaceConformance(
        ConformanceCheckingContext* context,
        Type* subType,
        Type* superInterfaceType,
        InheritanceDecl* inheritanceDecl,
        DeclRef<InterfaceDecl> superInterfaceDeclRef,
        SubtypeWitness* subTypeConformsToSuperInterfaceWitness);

    bool checkConformanceToType(
        ConformanceCheckingContext* context,
        Type* subType,
        InheritanceDecl* inheritanceDecl,
        Type* superType,
        SubtypeWitness* subIsSuperWitness,
        WitnessTable* witnessTable);

    /// Check that `type` which has declared that it inherits from (and/or implements)
    /// another type via `inheritanceDecl` actually does what it needs to for that
    /// inheritance to be valid.
    bool checkConformance(Type* type, InheritanceDecl* inheritanceDecl, ContainerDecl* parentDecl);

    void checkExtensionConformance(ExtensionDecl* decl);

    void checkAggTypeConformance(AggTypeDecl* decl);

    bool isIntegerBaseType(BaseType baseType);

    /// Is `type` a scalar integer type.
    bool isScalarIntegerType(Type* type);

    /// Is `type` a scalar half type.
    bool isHalfType(Type* type);

    /// Is `type` something we allow as compile time constants, i.e. scalar integer and enum types.
    bool isValidCompileTimeConstantType(Type* type);

    bool isIntValueInRangeOfType(IntegerLiteralValue value, Type* type);

    // Validate that `type` is a suitable type to use
    // as the tag type for an `enum`
    void validateEnumTagType(Type* type, SourceLoc const& loc);

    void checkStmt(Stmt* stmt, SemanticsContext const& context);

    Stmt* maybeParseStmt(Stmt* stmt, const SemanticsContext& context);

    void getGenericParams(
        GenericDecl* decl,
        List<Decl*>& outParams,
        List<GenericTypeConstraintDecl*>& outConstraints);

    /// Determine if `left` and `right` have matching generic signatures.
    /// If they do, then outputs a specialized declRef to `ioSubstRightToLeft` that
    /// represents a reference to `right` with the parameters of `left`.
    bool doGenericSignaturesMatch(
        GenericDecl* left,
        GenericDecl* right,
        DeclRef<Decl>* outSpecializedRightInner);

    // Check if two functions have the same signature for the purposes
    // of overload resolution.
    bool doFunctionSignaturesMatch(DeclRef<FuncDecl> fst, DeclRef<FuncDecl> snd);

    Result checkRedeclaration(Decl* newDecl, Decl* oldDecl);
    Result checkFuncRedeclaration(FuncDecl* newDecl, FuncDecl* oldDecl);
    void checkForRedeclaration(Decl* decl);

    Expr* checkPredicateExpr(Expr* expr);

    enum class ConstantFoldingKind
    {
        CompileTime,
        LinkTime,
    };
    Expr* checkExpressionAndExpectIntegerConstant(
        Expr* expr,
        IntVal** outIntVal,
        ConstantFoldingKind kind);

    IntegerLiteralValue GetMinBound(IntVal* val);

    void maybeInferArraySizeForVariable(VarDeclBase* varDecl);

    void validateArraySizeForVariable(VarDeclBase* varDecl);

    IntVal* getIntVal(IntegerLiteralExpr* expr);

    inline IntVal* getIntVal(SubstExpr<IntegerLiteralExpr> expr)
    {
        return getIntVal(expr.getExpr());
    }

    Name* getName(String const& text) { return getNamePool()->getName(text); }

    /// Helper type to detect and catch circular definitions when folding constants,
    /// to prevent the compiler from going into infinite loops or overflowing the stack.
    struct ConstantFoldingCircularityInfo
    {
        ConstantFoldingCircularityInfo(Decl* decl, ConstantFoldingCircularityInfo* next)
            : decl(decl), next(next)
        {
        }

        /// A declaration whose value is contributing to the constant being folded
        Decl* decl = nullptr;

        /// The rest of the links in the chain of declarations being folded
        ConstantFoldingCircularityInfo* next = nullptr;
    };
    /// Try to apply front-end constant folding to determine the value of `invokeExpr`.
    IntVal* tryConstantFoldExpr(
        SubstExpr<InvokeExpr> invokeExpr,
        ConstantFoldingKind kind,
        ConstantFoldingCircularityInfo* circularityInfo);

    /// Try to apply front-end constant folding to determine the value of `expr`.
    IntVal* tryConstantFoldExpr(
        SubstExpr<Expr> expr,
        ConstantFoldingKind kind,
        ConstantFoldingCircularityInfo* circularityInfo);

    bool _checkForCircularityInConstantFolding(
        Decl* decl,
        ConstantFoldingCircularityInfo* circularityInfo);

    /// Try to resolve a compile-time constant `IntVal` from the given `declRef`.
    IntVal* tryConstantFoldDeclRef(
        DeclRef<VarDeclBase> const& declRef,
        ConstantFoldingKind kind,
        ConstantFoldingCircularityInfo* circularityInfo);

    /// Try to extract the value of an integer constant expression, either
    /// returning the `IntVal` value, or null if the expression isn't recognized
    /// as an integer constant.
    ///
    IntVal* tryFoldIntegerConstantExpression(
        SubstExpr<Expr> expr,
        ConstantFoldingKind kind,
        ConstantFoldingCircularityInfo* circularityInfo);

    IntVal* tryFoldIndexExpr(
        SubstExpr<IndexExpr> expr,
        ConstantFoldingKind kind,
        ConstantFoldingCircularityInfo* circularityInfo);

    // Enforce that an expression resolves to an integer constant, and get its value
    enum class IntegerConstantExpressionCoercionType
    {
        SpecificType,
        AnyInteger
    };
    IntVal* CheckIntegerConstantExpression(
        Expr* inExpr,
        IntegerConstantExpressionCoercionType coercionType,
        Type* expectedType,
        ConstantFoldingKind kind);
    IntVal* CheckIntegerConstantExpression(
        Expr* inExpr,
        IntegerConstantExpressionCoercionType coercionType,
        Type* expectedType,
        ConstantFoldingKind kind,
        DiagnosticSink* sink);

    IntVal* CheckEnumConstantExpression(Expr* expr, ConstantFoldingKind kind);


    Expr* CheckSimpleSubscriptExpr(IndexExpr* subscriptExpr, Type* elementType);

    // The way that we have designed out type system, pretyt much *every*
    // type is a reference to some declaration in the core module.
    // That means that when we construct a new type on the fly, we need
    // to make sure that it is wired up to reference the appropriate
    // declaration, or else it won't compare as equal to other types
    // that *do* reference the declaration.
    //
    // This function is used to construct a `vector<T,N>` type
    // programmatically, so that it will work just like a type of
    // that form constructed by the user.
    VectorExpressionType* createVectorType(Type* elementType, IntVal* elementCount);

    //

    /// Given an immutable `expr` used as an l-value emit a special diagnostic if it was derived
    /// from `this`.
    void maybeDiagnoseThisNotLValue(Expr* expr);

    // Figure out what type an initializer/constructor declaration
    // is supposed to return. In most cases this is just the type
    // declaration that its declaration is nested inside.
    Type* findResultTypeForConstructorDecl(ConstructorDecl* decl);

    /// Determine what type `This` should refer to in the context of the given parent `decl`.
    Type* calcThisType(DeclRef<Decl> decl);

    /// Determine what type `This` should refer to in an extension of `type`.
    Type* calcThisType(Type* type);


    //

    struct Constraint
    {
        Decl* decl = nullptr;  // the declaration of the thing being constraints
        Index indexInPack = 0; // If the constraint is for a type parameter pack, which index in the
                               // pack is this constraint for?

        Val* val = nullptr;          // the value to which we are constraining it
        bool isUsedAsLValue = false; // If this constraint is for a type parameter, is the type used
                                     // in an l-value parameter?
        bool satisfied = false;      // Has this constraint been met?

        // Is this constraint optional? An optional constraint provides a hint value to a parameter
        // if it is otherwise unconstrained, but doesn't take precedence over a constraint that is
        // not optional.
        bool isOptional = false;
    };

    // A collection of constraints that will need to be satisfied (solved)
    // in order for checking to succeed.
    struct ConstraintSystem
    {
        // A source location to use in reporting any issues
        SourceLoc loc;

        // The generic declaration whose parameters we
        // are trying to solve for.
        GenericDecl* genericDecl = nullptr;

        // Constraints we have accumulated, which constrain
        // the possible arguments for those parameters.
        List<Constraint> constraints;

        // Additional subtype witnesses available to the currentt constraint solving context.
        Type* subTypeForAdditionalWitnesses = nullptr;
        Dictionary<Type*, SubtypeWitness*>* additionalSubtypeWitnesses = nullptr;
    };

    Type* TryJoinVectorAndScalarType(
        ConstraintSystem* constraints,
        VectorExpressionType* vectorType,
        BasicExpressionType* scalarType);

    /// Is the given interface one that a tagged-union type can conform to?
    ///
    /// If a tagged union type `__TaggedUnion(A,B)` is going to be
    /// plugged in for a type parameter `T : IFoo` then we need to
    /// be sure that the interface `IFoo` doesn't have anything
    /// that could lead to unsafe/unsound behavior. This function
    /// checks that all the requirements on the interfaceare safe ones.
    ///
    bool isInterfaceSafeForTaggedUnion(DeclRef<InterfaceDecl> interfaceDeclRef);

    /// Is the given interface requirement one that a tagged-union type can satisfy?
    ///
    /// Unsafe requirements include any `static` requirements,
    /// any associated types, and also any requirements that make
    /// use of the `This` type (once we support it).
    ///
    bool isInterfaceRequirementSafeForTaggedUnion(
        DeclRef<InterfaceDecl> interfaceDeclRef,
        DeclRef<Decl> requirementDeclRef);

    /// Check whether `subType` is a subtype of `superType`
    ///
    /// If `subType` is a subtype of `superType`, returns
    /// a witness value for the subtype relationship.
    ///
    /// If `subType` is *not* a subtype of `superType`, returns null.
    ///
    SubtypeWitness* isSubtype(Type* subType, Type* superType, IsSubTypeOptions isSubTypeOptions);

    SubtypeWitness* checkAndConstructSubtypeWitness(
        Type* subType,
        Type* superType,
        IsSubTypeOptions isSubTypeOptions);

    bool isValidGenericConstraintType(Type* type);

    SubtypeWitness* isTypeDifferentiable(Type* type);

    bool doesTypeHaveTag(Type* type, TypeTag tag);

    TypeTag getTypeTags(Type* type);

    Type* getConstantBufferElementType(Type* type);

    /// Check whether `subType` is a sub-type of `superTypeDeclRef`,
    /// and return a witness to the sub-type relationship if it holds
    /// (return null otherwise).
    ///
    SubtypeWitness* tryGetSubtypeWitness(Type* subType, Type* superType)
    {
        return isSubtype(subType, superType, IsSubTypeOptions::None);
    }

    /// Check whether `type` conforms to `interfaceDeclRef`,
    /// and return a witness to the conformance if it holds
    /// (return null otherwise).
    ///
    /// This function is equivalent to `tryGetSubtypeWitness()`.
    ///
    SubtypeWitness* tryGetInterfaceConformanceWitness(Type* type, Type* interfaceType);

    Expr* createCastToSuperTypeExpr(Type* toType, Expr* fromExpr, Val* witness);

    Expr* createModifierCastExpr(Type* toType, Expr* fromExpr);

    /// Does there exist an implicit conversion from `fromType` to `toType`?
    bool canConvertImplicitly(Type* toType, QualType fromType);

    bool canConvertImplicitly(ConversionCost cost);

    ConversionCost getConversionCost(Type* toType, QualType fromType);

    Type* _tryJoinTypeWithInterface(ConstraintSystem* constraints, Type* type, Type* interfaceType);

    // Try to compute the "join" between two types
    Type* TryJoinTypes(ConstraintSystem* constraints, QualType left, QualType right);

    // Try to solve a system of generic constraints.
    // The `system` argument provides the constraints.
    // The `varSubst` argument provides the list of constraint
    // variables that were created for the system.
    //
    // Returns a new declref to the inner decl of `genericDeclRef`,
    // representing the specialized generic with the values
    // we solved for along the way.
    DeclRef<Decl> trySolveConstraintSystem(
        ConstraintSystem* system,
        DeclRef<GenericDecl> genericDeclRef,
        ArrayView<Val*> knownGenericArgs,
        ConversionCost& outBaseCost);


    // State related to overload resolution for a call
    // to an overloaded symbol
    struct OverloadResolveContext
    {
        enum class Mode
        {
            // We are just checking if a candidate works or not
            JustTrying,

            // We want to actually update the AST for a chosen candidate
            ForReal,
        };

        // Location to use when reporting overload-resolution errors.
        SourceLoc loc;

        // The original expression (if any) that triggered things
        AppExprBase* originalExpr = nullptr;

        // Source location of the "function" part of the expression, if any
        SourceLoc funcLoc;

        // The source scope of the lookup for performing visibiliity tests.
        Scope* sourceScope = nullptr;

        // The original arguments to the call
        Index argCount = 0;
        List<Expr*>* args = nullptr;
        Type** argTypes = nullptr;

        Index getArgCount() { return argCount; }
        Expr*& getArg(Index index) { return (*args)[index]; }
        Type* getArgType(Index index)
        {
            if (argTypes)
                return argTypes[index];
            else
                return getArg(index)->type.type;
        }
        Type* getArgTypeForInference(Index index, SemanticsVisitor* semantics)
        {
            if (argTypes)
                return argTypes[index];
            else
                return semantics
                    ->maybeResolveOverloadedExpr(getArg(index), LookupMask::Default, nullptr)
                    ->type;
        }
        struct MatchedArg
        {
            Expr* argExpr = nullptr;
            Type* argType = nullptr;
        };
        bool matchArgumentsToParams(
            SemanticsVisitor* semantics,
            const List<QualType>& params,
            bool computeTypes,
            ShortList<MatchedArg>& outMatchedArgs);

        bool disallowNestedConversions = false;

        Expr* baseExpr = nullptr;

        // Are we still trying out candidates, or are we
        // checking the chosen one for real?
        Mode mode = Mode::JustTrying;

        // We store one candidate directly, so that we don't
        // need to do dynamic allocation on the list every time
        OverloadCandidate bestCandidateStorage;
        OverloadCandidate* bestCandidate = nullptr;

        // Full list of all candidates being considered, in the ambiguous case
        List<OverloadCandidate> bestCandidates;
    };

    struct ParamCounts
    {
        Count required;
        Count allowed;
    };

    // count the number of parameters required/allowed for a callable
    ParamCounts CountParameters(FilteredMemberRefList<ParamDecl> params);

    // count the number of parameters required/allowed for a generic
    ParamCounts CountParameters(DeclRef<GenericDecl> genericRef);

    bool TryCheckOverloadCandidateClassNewMatchUp(
        OverloadResolveContext& context,
        OverloadCandidate const& candidate);

    bool TryCheckOverloadCandidateArity(
        OverloadResolveContext& context,
        OverloadCandidate const& candidate);

    bool TryCheckOverloadCandidateFixity(
        OverloadResolveContext& context,
        OverloadCandidate const& candidate);

    bool TryCheckOverloadCandidateVisibility(
        OverloadResolveContext& context,
        OverloadCandidate const& candidate);

    bool TryCheckGenericOverloadCandidateTypes(
        OverloadResolveContext& context,
        OverloadCandidate& candidate);

    bool TryCheckOverloadCandidateTypes(
        OverloadResolveContext& context,
        OverloadCandidate& candidate);

    bool TryCheckOverloadCandidateDirections(
        OverloadResolveContext& /*context*/,
        OverloadCandidate const& /*candidate*/
    );

    /// Check if the given `expr` refers to an `in` function
    /// parameter, or part of one (through field reference, etc.).
    ///
    /// If the expression refers into a parameter, returns
    /// the declaration of the parameter. Otherwise returns
    /// null.
    ///
    ParamDecl* isReferenceIntoFunctionInputParameter(Expr* expr);

    // Create a witness that attests to the fact that `type`
    // is equal to itself.
    TypeEqualityWitness* createTypeEqualityWitness(Type* type);

    // In the case where we are explicitly applying a generic
    // to arguments (e.g., `G<A,B>`) check that the constraints
    // on those parameters are satisfied.
    //
    // Note: the constraints actually work as additional parameters/arguments
    // of the generic, and so we need to reify them into the final
    // argument list.
    //
    bool TryCheckOverloadCandidateConstraints(
        OverloadResolveContext& context,
        OverloadCandidate& candidate);

    // Try to check an overload candidate, but bail out
    // if any step fails
    void TryCheckOverloadCandidate(OverloadResolveContext& context, OverloadCandidate& candidate);

    // Create the representation of a given generic applied to some arguments
    Expr* createGenericDeclRef(Expr* baseExpr, Expr* originalExpr, SubstitutionSet substSet);

    // Take an overload candidate that previously got through
    // `TryCheckOverloadCandidate` above, and try to finish
    // up the work and turn it into a real expression.
    //
    // If the candidate isn't actually applicable, this is
    // where we'd start reporting the issue(s).
    Expr* CompleteOverloadCandidate(OverloadResolveContext& context, OverloadCandidate& candidate);

    // Implement a comparison operation between overload candidates,
    // so that the better candidate compares as less-than the other
    int CompareOverloadCandidates(OverloadCandidate* left, OverloadCandidate* right);

    /// If `declRef` representations a specialization of a generic, returns the number of
    /// specialized generic arguments. Otherwise, returns zero.
    ///
    Int getSpecializedParamCount(DeclRef<Decl> const& declRef);

    /// Compare items `left` and `right` produced by lookup, to see if one should be favored for
    /// overloading.
    int CompareLookupResultItems(LookupResultItem const& left, LookupResultItem const& right);

    /// Compare items `left` and `right` being considered as overload candidates, and determine if
    /// one should be favored for structural reasons.
    int compareOverloadCandidateSpecificity(
        LookupResultItem const& left,
        LookupResultItem const& right);

    void AddOverloadCandidateInner(OverloadResolveContext& context, OverloadCandidate& candidate);

    void AddOverloadCandidate(
        OverloadResolveContext& context,
        OverloadCandidate& candidate,
        ConversionCost baseCost);

    void AddHigherOrderOverloadCandidates(
        Expr* funcExpr,
        OverloadResolveContext& context,
        ConversionCost baseCost);

    void AddFuncOverloadCandidate(
        LookupResultItem item,
        DeclRef<CallableDecl> funcDeclRef,
        OverloadResolveContext& context,
        ConversionCost baseCost);

    void AddFuncOverloadCandidate(
        FuncType* /*funcType*/,
        OverloadResolveContext& /*context*/,
        ConversionCost baseCost);

    void AddFuncExprOverloadCandidate(
        FuncType* funcType,
        OverloadResolveContext& context,
        Expr* expr,
        ConversionCost baseCost);

    // Add a candidate callee for overload resolution, based on
    // calling a particular `ConstructorDecl`.
    void AddCtorOverloadCandidate(
        LookupResultItem typeItem,
        Type* type,
        DeclRef<ConstructorDecl> ctorDeclRef,
        OverloadResolveContext& context,
        Type* resultType,
        ConversionCost baseCost);

    // If the given declaration has generic parameters, then
    // return the corresponding `GenericDecl` that holds the
    // parameters, etc. This returns the immediate generic parent
    // of `decl`, e.g. the generic for f<T>, and *not* any indirect
    // generic parents, such as P<T>.f().
    GenericDecl* GetOuterGeneric(Decl* decl);

    // If `decl` is inside a generic, return that outer generic,
    // otherwise returns `decl`.
    Decl* getOuterGenericOrSelf(Decl* decl);

    // Find the next outer generic parent of `decl`, including
    // indirect parents.
    GenericDecl* findNextOuterGeneric(Decl* decl);

    struct ValUnificationContext
    {
        Index indexInTypePack = 0;
    };

    // Try to find a unification for two values
    bool TryUnifyVals(
        ConstraintSystem& constraints,
        ValUnificationContext unificationContext,
        Val* fst,
        bool fstLVal,
        Val* snd,
        bool sndLVal);

    bool tryUnifyDeclRef(
        ConstraintSystem& constraints,
        ValUnificationContext unificationContext,
        DeclRefBase* fst,
        bool fstLVal,
        DeclRefBase* snd,
        bool sndLVal);

    bool tryUnifyGenericAppDeclRef(
        ConstraintSystem& constraints,
        ValUnificationContext unificationContext,
        GenericAppDeclRef* fst,
        bool fstLVal,
        GenericAppDeclRef* snd,
        bool sndLVal);

    bool TryUnifyTypeParam(
        ConstraintSystem& constraints,
        ValUnificationContext unificationContext,
        GenericTypeParamDeclBase* typeParamDecl,
        QualType type);

    bool TryUnifyIntParam(
        ConstraintSystem& constraints,
        ValUnificationContext unificationContext,
        GenericValueParamDecl* paramDecl,
        IntVal* val);

    bool TryUnifyIntParam(
        ConstraintSystem& constraints,
        ValUnificationContext unificationContext,
        DeclRef<VarDeclBase> const& varRef,
        IntVal* val);

    bool TryUnifyTypesByStructuralMatch(
        ConstraintSystem& constraints,
        ValUnificationContext unificationContext,
        QualType fst,
        QualType snd);

    bool TryUnifyTypes(
        ConstraintSystem& constraints,
        ValUnificationContext unificationContext,
        QualType fst,
        QualType snd);

    bool TryUnifyConjunctionType(
        ConstraintSystem& constraints,
        ValUnificationContext unificationContext,
        QualType fst,
        QualType snd);

    void maybeUnifyUnconstraintIntParam(
        ConstraintSystem& constraints,
        ValUnificationContext unificationContext,
        IntVal* param,
        IntVal* arg,
        bool paramIsLVal);

    // Is the candidate extension declaration actually applicable to the given type
    DeclRef<ExtensionDecl> applyExtensionToType(
        ExtensionDecl* extDecl,
        Type* type,
        Dictionary<Type*, SubtypeWitness*>* additionalSubtypeWitnessesForType = nullptr);

    // Take a generic declaration that is being applied
    // in a context and attempt to infer any missing generic
    // arguments to form a `DeclRef` to the inner declaration
    // that could be applicable in the context of the given
    // overloaded call.
    // Also computes a `baseCost` for the inferred arguments,
    // so that we can prefer a more specialized generic candidate
    // when there is ambiguity. For example, given
    // ```
    //     interface IBase;
    //     interface IDerived : IBase;
    //     struct Derived : IDerived {}
    //     void f1<T:IBase>(T b)
    //     void f2<T:IDerived>(T b);
    // ```
    // We will prefer f2 when seeing f(Derived()), because it takes
    // less steps to upcast `Derived` to  `IDerived` than it does
    // to `IBase`.
    //
    DeclRef<Decl> inferGenericArguments(
        DeclRef<GenericDecl> genericDeclRef,
        OverloadResolveContext& context,
        ArrayView<Val*> knownGenericArgs,
        ConversionCost& outBaseCost,
        List<QualType>* innerParameterTypes = nullptr);

    void AddTypeOverloadCandidates(Type* type, OverloadResolveContext& context);

    void AddDeclRefOverloadCandidates(
        LookupResultItem item,
        OverloadResolveContext& context,
        ConversionCost baseCost);

    void AddOverloadCandidates(LookupResult const& result, OverloadResolveContext& context);

    void AddOverloadCandidates(Expr* funcExpr, OverloadResolveContext& context);

    String getCallSignatureString(OverloadResolveContext& context);

    Expr* ResolveInvoke(InvokeExpr* expr);

    void AddGenericOverloadCandidate(LookupResultItem baseItem, OverloadResolveContext& context);

    void AddGenericOverloadCandidates(Expr* baseExpr, OverloadResolveContext& context);

    template<class T>
    void trySetGenericToRayTracingWithParamAttribute(
        LookupResultItem genericItem,
        DeclRef<GenericDecl> genericDeclRef,
        OverloadResolveContext& context);

    // Add overload candidates based on use of `genericDeclRef`
    // in an ordinary function-call context (that is, where it
    // has been applied to arguments using `()` and not `<>`).
    //
    // If some or all of the generic arguments to `genericDeclRef`
    // are known at the call site, they should be passed in via
    // `substWithKnownGenericArgs`.
    //
    void addOverloadCandidatesForCallToGeneric(
        LookupResultItem genericItem,
        OverloadResolveContext& context,
        ArrayView<Val*> knownGenericArgs);

    /// Check a generic application where the operands have already been checked.
    Expr* checkGenericAppWithCheckedArgs(GenericAppExpr* genericAppExpr);

    Expr* CheckExpr(Expr* expr);


    void compareMemoryQualifierOfParamToArgument(ParamDecl* paramIn, Expr* argIn);
    Expr* CheckInvokeExprWithCheckedOperands(InvokeExpr* expr);
    // Get the type to use when referencing a declaration
    QualType GetTypeForDeclRef(DeclRef<Decl> declRef, SourceLoc loc);

    //
    //
    //

    Expr* CheckMatrixSwizzleExpr(
        MemberExpr* memberRefExpr,
        Type* baseElementType,
        IntegerLiteralValue baseElementRowCount,
        IntegerLiteralValue baseElementColCount);

    Expr* CheckMatrixSwizzleExpr(
        MemberExpr* memberRefExpr,
        Type* baseElementType,
        IntVal* baseElementRowCount,
        IntVal* baseElementColCount);

    Expr* checkTupleSwizzleExpr(MemberExpr* memberExpr, TupleType* baseTupleType);

    Expr* CheckSwizzleExpr(
        MemberExpr* memberRefExpr,
        Type* baseElementType,
        IntegerLiteralValue baseElementCount);

    Expr* CheckSwizzleExpr(
        MemberExpr* memberRefExpr,
        Type* baseElementType,
        IntVal* baseElementCount);

    // Check a member expr as a general member lookup.
    // This is the default/fallback behavior if the base type isn't swizzlable.
    Expr* checkGeneralMemberLookupExpr(MemberExpr* expr, Type* baseType);

    /// Perform semantic checking of an assignment where the operands have already been checked.
    Expr* checkAssignWithCheckedOperands(AssignExpr* expr);

    // Look up a static member
    // @param expr Can be StaticMemberExpr or MemberExpr
    // @param baseExpression Is the underlying type expression determined from resolving expr
    Expr* _lookupStaticMember(DeclRefExpr* expr, Expr* baseExpression);

    Expr* visitStaticMemberExpr(StaticMemberExpr* expr);

    /// Perform checking operations required for the "base" expression of a member-reference like
    /// `base.someField`
    enum class CheckBaseContext
    {
        Member,
        Subscript,
    };
    Expr* checkBaseForMemberExpr(
        Expr* baseExpr,
        CheckBaseContext checkBaseContext,
        bool& outNeedDeref);

    Expr* maybeDereference(Expr* inExpr, CheckBaseContext checkBaseContext);

    /// Prepare baseExpr for use as the base of a member expr.
    /// This include inserting implicit open-existential operations as needed.
    Expr* maybeInsertImplicitOpForMemberBase(
        Expr* baseExpr,
        CheckBaseContext checkBaseContext,
        bool& outNeedDeref);

    Expr* lookupMemberResultFailure(
        DeclRefExpr* expr,
        QualType const& baseType,
        bool supressDiagnostic = false);

    SharedSemanticsContext& operator=(const SharedSemanticsContext&) = delete;


    //

    void importModuleIntoScope(Scope* scope, ModuleDecl* moduleDecl);
    void importFileDeclIntoScope(Scope* scope, FileDecl* fileDecl);


    void suggestCompletionItems(
        CompletionSuggestions::ScopeKind scopeKind,
        LookupResult const& lookupResult);

    bool createInvokeExprForExplicitCtor(
        Type* toType,
        InitializerListExpr* fromInitializerListExpr,
        Expr** outExpr);

    bool createInvokeExprForSynthesizedCtor(
        Type* toType,
        InitializerListExpr* fromInitializerListExpr,
        Expr** outExpr);

    Expr* _createCtorInvokeExpr(Type* toType, const SourceLoc& loc, const List<Expr*>& coercedArgs);
    bool _hasExplicitConstructor(StructDecl* structDecl, bool checkBaseType);
    ConstructorDecl* _getSynthesizedConstructor(
        StructDecl* structDecl,
        ConstructorDecl::ConstructorFlavor flavor);
    bool isCStyleType(Type* type, HashSet<Type*>& isVisit);

    void addVisibilityModifier(Decl* decl, DeclVisibility vis);

    void checkRayPayloadStructFields(StructDecl* structDecl);
};


inline void ensureDecl(SemanticsVisitor* visitor, Decl* decl, DeclCheckState state)
{
    visitor->ensureDecl(decl, state);
}

DeclRef<ExtensionDecl> applyExtensionToType(
    SemanticsVisitor* semantics,
    ExtensionDecl* extDecl,
    Type* type,
    Dictionary<Type*, SubtypeWitness*>* additionalSubtypeWitness = nullptr);


struct SemanticsExprVisitor : public SemanticsVisitor, ExprVisitor<SemanticsExprVisitor, Expr*>
{
public:
    SemanticsExprVisitor(SemanticsContext const& outer)
        : SemanticsVisitor(outer)
    {
    }

    Expr* visitSizeOfLikeExpr(SizeOfLikeExpr* expr);

    Expr* visitIncompleteExpr(IncompleteExpr* expr);
    Expr* visitBoolLiteralExpr(BoolLiteralExpr* expr);
    Expr* visitNullPtrLiteralExpr(NullPtrLiteralExpr* expr);
    Expr* visitNoneLiteralExpr(NoneLiteralExpr* expr);
    Expr* visitIntegerLiteralExpr(IntegerLiteralExpr* expr);
    Expr* visitFloatingPointLiteralExpr(FloatingPointLiteralExpr* expr);
    Expr* visitStringLiteralExpr(StringLiteralExpr* expr);

    Expr* visitIndexExpr(IndexExpr* subscriptExpr);

    Expr* visitParenExpr(ParenExpr* expr);

    Expr* visitAssignExpr(AssignExpr* expr);

    Expr* visitGenericAppExpr(GenericAppExpr* genericAppExpr);

    Expr* visitSharedTypeExpr(SharedTypeExpr* expr);

    Expr* visitInvokeExpr(InvokeExpr* expr);

    Expr* visitSelectExpr(SelectExpr* expr);

    Expr* visitVarExpr(VarExpr* expr);

    Expr* visitTypeCastExpr(TypeCastExpr* expr);

    Expr* visitBuiltinCastExpr(BuiltinCastExpr* expr);

    Expr* visitTryExpr(TryExpr* expr);

    Expr* visitIsTypeExpr(IsTypeExpr* expr);

    Expr* visitAsTypeExpr(AsTypeExpr* expr);

    Expr* visitExpandExpr(ExpandExpr* expr);

    Expr* visitEachExpr(EachExpr* expr);

    Expr* visitLambdaExpr(LambdaExpr* expr);

    void maybeCheckKnownBuiltinInvocation(Expr* invokeExpr);

    Expr* maybeRegisterLambdaCapture(Expr* exprIn);
    //
    // Some syntax nodes should not occur in the concrete input syntax,
    // and will only appear *after* checking is complete. We need to
    // deal with this cases here, even if they are no-ops.
    //

#define CASE(NAME) \
    Expr* visit##NAME(NAME* expr) { return expr; }

    CASE(DerefExpr)
    CASE(MakeRefExpr)
    CASE(MatrixSwizzleExpr)
    CASE(SwizzleExpr)
    CASE(OverloadedExpr)
    CASE(OverloadedExpr2)
    CASE(AggTypeCtorExpr)
    CASE(ModifierCastExpr)
    CASE(LetExpr)
    CASE(ExtractExistentialValueExpr)
    CASE(OpenRefExpr)
    CASE(MakeOptionalExpr)
    CASE(PartiallyAppliedGenericExpr)
    CASE(PackExpr)
#undef CASE

    Expr* visitStaticMemberExpr(StaticMemberExpr* expr);

    Expr* visitMemberExpr(MemberExpr* expr);

    Expr* visitInitializerListExpr(InitializerListExpr* expr);

    Expr* visitThisExpr(ThisExpr* expr);
    Expr* visitThisTypeExpr(ThisTypeExpr* expr);
    Expr* visitCastToSuperTypeExpr(CastToSuperTypeExpr* expr);
    Expr* visitReturnValExpr(ReturnValExpr* expr);
    Expr* visitAndTypeExpr(AndTypeExpr* expr);
    Expr* visitPointerTypeExpr(PointerTypeExpr* expr);
    Expr* visitModifiedTypeExpr(ModifiedTypeExpr* expr);
    Expr* visitFuncTypeExpr(FuncTypeExpr* expr);
    Expr* visitTupleTypeExpr(TupleTypeExpr* expr);

    Expr* visitForwardDifferentiateExpr(ForwardDifferentiateExpr* expr);
    Expr* visitBackwardDifferentiateExpr(BackwardDifferentiateExpr* expr);
    Expr* visitPrimalSubstituteExpr(PrimalSubstituteExpr* expr);
    Expr* visitDispatchKernelExpr(DispatchKernelExpr* expr);

    Expr* visitTreatAsDifferentiableExpr(TreatAsDifferentiableExpr* expr);

    Expr* visitGetArrayLengthExpr(GetArrayLengthExpr* expr);

    Expr* visitDefaultConstructExpr(DefaultConstructExpr* expr);

    Expr* visitDetachExpr(DetachExpr* expr);

    Expr* visitSPIRVAsmExpr(SPIRVAsmExpr*);

    /// Perform semantic checking on a `modifier` that is being applied to the given `type`
    Val* checkTypeModifier(Modifier* modifier, Type* type);

private:
    // Convert the logic operator expression to not use 'InvokeExpr' type
    Expr* convertToLogicOperatorExpr(InvokeExpr* expr);
};

struct SemanticsStmtVisitor : public SemanticsVisitor, StmtVisitor<SemanticsStmtVisitor>
{
    SemanticsStmtVisitor(SemanticsContext const& outer)
        : SemanticsVisitor(outer)
    {
    }

    FunctionDeclBase* getParentFunc() { return m_parentFunc; }

    void checkStmt(Stmt* stmt);

    template<typename T>
    T* FindOuterStmt(Stmt* searchUntil = nullptr);

    Stmt* findOuterStmtWithLabel(Name* label);

    void visitDeclStmt(DeclStmt* stmt);

    void visitBlockStmt(BlockStmt* stmt);

    void visitSeqStmt(SeqStmt* stmt);

    void visitLabelStmt(LabelStmt* stmt);

    void visitBreakStmt(BreakStmt* stmt);

    void visitContinueStmt(ContinueStmt* stmt);

    void visitDoWhileStmt(DoWhileStmt* stmt);

    void visitForStmt(ForStmt* stmt);

    void visitCompileTimeForStmt(CompileTimeForStmt* stmt);

    void visitSwitchStmt(SwitchStmt* stmt);

    void visitCaseStmt(CaseStmt* stmt);

    void visitTargetSwitchStmt(TargetSwitchStmt* stmt);

    void visitTargetCaseStmt(TargetCaseStmt* stmt);

    void visitIntrinsicAsmStmt(IntrinsicAsmStmt*);

    void visitDefaultStmt(DefaultStmt* stmt);

    void visitIfStmt(IfStmt* stmt);

    void visitUnparsedStmt(UnparsedStmt*);

    void visitEmptyStmt(EmptyStmt*);

    void visitDiscardStmt(DiscardStmt*);

    void visitReturnStmt(ReturnStmt* stmt);

    void visitDeferStmt(DeferStmt* stmt);

    void visitWhileStmt(WhileStmt* stmt);

    void visitGpuForeachStmt(GpuForeachStmt* stmt);

    void visitExpressionStmt(ExpressionStmt* stmt);

    // Try to infer the max number of iterations the loop will run.
    void tryInferLoopMaxIterations(ForStmt* stmt);

    void checkLoopInDifferentiableFunc(Stmt* stmt);

private:
    void validateCaseStmts(SwitchStmt* stmt, DiagnosticSink* sink);

    void generateUniqueIDForStmt(BreakableStmt* stmt);
};

struct SemanticsDeclVisitorBase : public SemanticsVisitor
{
    SemanticsDeclVisitorBase(SemanticsContext const& outer)
        : SemanticsVisitor(outer)
    {
    }

    void checkBodyStmt(Stmt* stmt, FunctionDeclBase* parentDecl)
    {
        checkStmt(stmt, withParentFunc(parentDecl));
    }

    void checkModule(ModuleDecl* programNode);

    ConstructorDecl* createCtor(AggTypeDecl* decl, DeclVisibility ctorVisibility);
};

bool isUnsizedArrayType(Type* type);

bool isInterfaceType(Type* type);

EnumDecl* isEnumType(Type* type);

DeclVisibility getDeclVisibility(Decl* decl);

// If `type` is unsized, return the trailing unsized array field that makes it so.
VarDeclBase* getTrailingUnsizedArrayElement(
    Type* type,
    VarDeclBase* rootObject,
    ArrayExpressionType*& outArrayType);

// Test if `type` can be an opaque handle on certain targets, this includes
// texture, buffer, sampler, acceleration structure, etc.
bool isOpaqueHandleType(Type* type);

void diagnoseMissingCapabilityProvenance(
    CompilerOptionSet& optionSet,
    DiagnosticSink* sink,
    Decl* decl,
    CapabilitySet& setToFind);
void diagnoseCapabilityProvenance(
    CompilerOptionSet& optionSet,
    DiagnosticSink* sink,
    Decl* decl,
    CapabilityAtom atomToFind,
    HashSet<Decl*>& printedDecls);

void _ensureAllDeclsRec(SemanticsDeclVisitorBase* visitor, Decl* decl, DeclCheckState state);

RefPtr<EntryPoint> findAndValidateEntryPoint(FrontEndEntryPointRequest* entryPointReq);

bool resolveStageOfProfileWithEntryPoint(
    Profile& entryPointProfile,
    CompilerOptionSet& optionSet,
    const List<RefPtr<TargetRequest>>& targets,
    FuncDecl* entryPointFuncDecl,
    DiagnosticSink* sink);

// For an extensions decl, collect a list of decls on which the extension might be applying to.
// For example, if we see a `extension Foo`, return a `Decl*` that represents `struct Foo`.
// In the case of free-form generic extensions i.e. `extension<T:IFoo> T : IBar`, return `IFoo`.
// These are the decls that we need to register the extension with in
// `mapTypeToCandidateExtensions`.
// Returns true when any base decls are found.
bool getExtensionTargetDeclList(
    ASTBuilder* astBuilder,
    DeclRefType* targetDeclRefType,
    ExtensionDecl* extDeclRef,
    ShortList<AggTypeDecl*>& targetDecls);

} // namespace Slang
