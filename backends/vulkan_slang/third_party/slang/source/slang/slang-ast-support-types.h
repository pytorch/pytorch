#ifndef SLANG_AST_SUPPORT_TYPES_H
#define SLANG_AST_SUPPORT_TYPES_H

#include "../compiler-core/slang-doc-extractor.h"
#include "../compiler-core/slang-lexer.h"
#include "../compiler-core/slang-name.h"
#include "../core/slang-basic.h"
#include "../core/slang-semantic-version.h"
#include "slang-ast-forward-declarations.h"
#include "slang-ast-support-types.h.fiddle"
#include "slang-profile.h"
#include "slang-type-system-shared.h"
#include "slang.h"

#include <assert.h>
#include <type_traits>

#define SLANG_UNREFLECTED /* empty */

FIDDLE(hidden class RefObject;)

FIDDLE() namespace Slang
{
#define SLANG_AST_NODE_VIRTUAL_CALL(CLASS, METHOD, ARGS)                    \
    return ASTNodeDispatcher<CLASS, decltype(this->METHOD ARGS)>::dispatch( \
        this,                                                               \
        [&](auto _this) -> decltype(this->METHOD ARGS)                      \
        { return _this->_##METHOD##Override ARGS; });

    class Module;
    class Name;
    class Session;
    class SyntaxVisitor;
    class FuncDecl;
    class Layout;

    class Parser;
    class SyntaxNode;

    class Decl;
    struct QualType;
    class Type;
    struct TypeExp;
    class Val;

    class DeclRefBase;
    class NodeBase;
    class LookupDeclRef;
    class GenericAppDeclRef;
    struct CapabilitySet;

    template<typename T>
    T* as(NodeBase * node);

    template<typename T>
    const T* as(const NodeBase* node);

    void printDiagnosticArg(StringBuilder & sb, Decl * decl);
    void printDiagnosticArg(StringBuilder & sb, Type * type);
    void printDiagnosticArg(StringBuilder & sb, TypeExp const& type);
    void printDiagnosticArg(StringBuilder & sb, QualType const& type);
    void printDiagnosticArg(StringBuilder & sb, Val * val);
    void printDiagnosticArg(StringBuilder & sb, DeclRefBase * declRefBase);
    void printDiagnosticArg(StringBuilder & sb, ASTNodeType nodeType);
    void printDiagnosticArg(StringBuilder & sb, const CapabilitySet& set);
    void printDiagnosticArg(StringBuilder & sb, List<CapabilityAtom> & set);

    struct QualifiedDeclPath
    {
        DeclRefBase* declRef;
        QualifiedDeclPath() = default;
        QualifiedDeclPath(DeclRefBase* declRef)
            : declRef(declRef)
        {
        }
    };
    // Prints the fully qualified decl name.
    void printDiagnosticArg(StringBuilder & sb, QualifiedDeclPath path);


    class SyntaxNode;
    SourceLoc getDiagnosticPos(SyntaxNode const* syntax);
    SourceLoc getDiagnosticPos(TypeExp const& typeExp);
    SourceLoc getDiagnosticPos(DeclRefBase * declRef);
    SourceLoc getDiagnosticPos(Decl * decl);

    typedef NodeBase* (*SyntaxParseCallback)(Parser* parser, void* userData);

    typedef unsigned int ConversionCost;
    enum : ConversionCost
    {
        // No conversion at all
        kConversionCost_None = 0,

        kConversionCost_GenericParamUpcast = 1,
        kConversionCost_UnconstraintGenericParam = 20,
        kConversionCost_SizedArrayToUnsizedArray = 30,

        // Convert between matrices of different layout
        kConversionCost_MatrixLayout = 5,

        // Conversion from a buffer to the type it carries needs to add a minimal
        // extra cost, just so we can distinguish an overload on `ConstantBuffer<Foo>`
        // from one on `Foo`
        kConversionCost_GetRef = 5,
        kConversionCost_ImplicitDereference = 10,
        kConversionCost_InRangeIntLitConversion = 23,
        kConversionCost_InRangeIntLitSignedToUnsignedConversion = 32,
        kConversionCost_InRangeIntLitUnsignedToSignedConversion = 81,

        kConversionCost_MutablePtrToConstPtr = 20,

        // Conversions based on explicit sub-typing relationships are the cheapest
        //
        // TODO(tfoley): We will eventually need a discipline for ranking
        // when two up-casts are comparable.
        kConversionCost_CastToInterface = 50,

        // Conversion that is lossless and keeps the "kind" of the value the same
        kConversionCost_BoolToInt = 120, // Converting bool to int has lower cost than other integer
                                         // types to prevent ambiguity.
        kConversionCost_RankPromotion = 150,
        kConversionCost_NoneToOptional = 150,
        kConversionCost_ValToOptional = 150,
        kConversionCost_NullPtrToPtr = 150,
        kConversionCost_PtrToVoidPtr = 150,

        // Conversions that are lossless, but change "kind"
        kConversionCost_UnsignedToSignedPromotion = 200,

        // Same-size size unsigned->signed conversions are potentially lossy, but they are commonly
        // allowed silently.
        kConversionCost_SameSizeUnsignedToSignedConversion = 300,

        // Conversion from signed->unsigned integer of same or greater size
        kConversionCost_SignedToUnsignedConversion = 250,

        // Cost of converting an integer to a floating-point type
        kConversionCost_IntegerToFloatConversion = 400,

        // Cost of converting a pointer to bool
        kConversionCost_PtrToBool = 400,

        // Cost of converting an integer to int16_t
        kConversionCost_IntegerTruncate = 450,

        // Cost of converting an integer to a half type
        kConversionCost_IntegerToHalfConversion = 500,

        // Cost of using a concrete argument pack
        kConversionCost_ParameterPack = 500,

        // Default case (usable for user-defined conversions)
        kConversionCost_Default = 500,

        // Catch-all for conversions that should be discouraged
        // (i.e., that really shouldn't be made implicitly)
        //
        // TODO: make these conversions not be allowed implicitly in "Slang mode"
        kConversionCost_GeneralConversion = 900,

        // This is the cost of an explicit conversion, which should
        // not actually be performed.
        kConversionCost_Explicit = 90000,

        // Additional conversion cost to add when promoting from a scalar to
        // a vector (this will be added to the cost, if any, of converting
        // the element type of the vector)
        kConversionCost_OneVectorToScalar = 1,
        kConversionCost_ScalarToVector = 2,
        kConversionCost_ScalarToMatrix = 10,
        kConversionCost_ScalarIntegerToFloatMatrix =
            kConversionCost_IntegerToFloatConversion + kConversionCost_ScalarToMatrix,

        // Additional conversion cost to add when promoting from a scalar to
        // a CoopVector (this will be added to the cost, if any, of converting
        // the element type of the CoopVector)
        kConversionCost_ScalarToCoopVector = 1,

        // Additional cost when casting an LValue.
        kConversionCost_LValueCast = 800,

        // The cost of this conversion is defined by the type coercion constraint.
        kConversionCost_TypeCoercionConstraint = 1000,
        kConversionCost_TypeCoercionConstraintPlusScalarToVector =
            kConversionCost_TypeCoercionConstraint + kConversionCost_ScalarToVector,

        // Conversion is impossible
        kConversionCost_Impossible = 0xFFFFFFFF,
    };

    typedef unsigned int BuiltinConversionKind;
    enum : BuiltinConversionKind
    {
        kBuiltinConversion_Unknown = 0,
        kBuiltinConversion_FloatToDouble = 1,
    };

    enum class ImageFormat
    {
#define SLANG_FORMAT(NAME, OTHER) NAME,
#include "slang-image-format-defs.h"
#undef SLANG_FORMAT
    };

    struct ImageFormatInfo
    {
        SlangScalarType scalarType; ///< If image format is not made up of channels of set sizes
                                    ///< this will be SLANG_SCALAR_TYPE_NONE
        uint8_t channelCount;       ///< The number of channels
        uint8_t sizeInBytes;        ///< Size in bytes
        UnownedStringSlice name;    ///< The name associated with this type. NOTE! Currently these
                                    ///< names *are* the GLSL format names.
    };

    const ImageFormatInfo& getImageFormatInfo(ImageFormat format);

    bool findImageFormatByName(const UnownedStringSlice& name, ImageFormat* outFormat);
    bool findVkImageFormatByName(const UnownedStringSlice& name, ImageFormat* outFormat);

    char const* getGLSLNameForImageFormat(ImageFormat format);

    // TODO(tfoley): We should ditch this enumeration
    // and just use the IR opcodes that represent these
    // types directly. The one major complication there
    // is that the order of the enum values currently
    // matters, since it determines promotion rank.
    // We either need to keep that restriction, or
    // look up promotion rank by some other means.
    //

    class Decl;
    class Val;

    // Helper type for pairing up a name and the location where it appeared
    struct NameLoc
    {
        Name* name;
        SourceLoc loc;

        NameLoc()
            : name(nullptr)
        {
        }

        explicit NameLoc(Name* inName)
            : name(inName)
        {
        }


        NameLoc(Name* inName, SourceLoc inLoc)
            : name(inName), loc(inLoc)
        {
        }

        NameLoc(Token const& token)
            : name(token.getNameOrNull()), loc(token.getLoc())
        {
        }
    };

    struct StringSliceLoc
    {
        UnownedStringSlice name;
        SourceLoc loc;

        StringSliceLoc()
            : name(nullptr)
        {
        }
        explicit StringSliceLoc(const UnownedStringSlice& inName)
            : name(inName)
        {
        }
        StringSliceLoc(const UnownedStringSlice& inName, SourceLoc inLoc)
            : name(inName), loc(inLoc)
        {
        }
        StringSliceLoc(Token const& token)
            : loc(token.getLoc())
        {
            Name* tokenName = token.getNameOrNull();
            if (tokenName)
            {
                name = tokenName->text.getUnownedSlice();
            }
        }
    };

    // Helper class for iterating over a list of heap-allocated modifiers
    struct ModifierList
    {
        struct Iterator
        {
            Modifier* current = nullptr;

            Modifier* operator*() { return current; }

            void operator++();

            bool operator!=(Iterator other) { return current != other.current; };

            Iterator()
                : current(nullptr)
            {
            }

            Iterator(Modifier* modifier)
                : current(modifier)
            {
            }
        };

        ModifierList()
            : modifiers(nullptr)
        {
        }

        ModifierList(Modifier* modifiers)
            : modifiers(modifiers)
        {
        }

        Iterator begin() { return Iterator(modifiers); }
        Iterator end() { return Iterator(nullptr); }

        Modifier* modifiers = nullptr;
    };

    // Helper class for iterating over heap-allocated modifiers
    // of a specific type.
    template<typename T>
    struct FilteredModifierList
    {
        struct Iterator
        {
            Modifier* current = nullptr;

            T* operator*() { return (T*)current; }

            void operator++();

            bool operator!=(Iterator other) { return current != other.current; };

            Iterator()
                : current(nullptr)
            {
            }

            Iterator(Modifier* modifier)
                : current(modifier)
            {
            }
        };

        FilteredModifierList()
            : modifiers(nullptr)
        {
        }

        FilteredModifierList(Modifier* modifiers)
            : modifiers(adjust(modifiers))
        {
        }

        Iterator begin() { return Iterator(modifiers); }
        Iterator end() { return Iterator(nullptr); }

        static Modifier* adjust(Modifier* modifier);

        Modifier* modifiers = nullptr;
    };

    // A set of modifiers attached to a syntax node
    struct Modifiers
    {
        // The first modifier in the linked list of heap-allocated modifiers
        Modifier* first = nullptr;

        template<typename T>
        FilteredModifierList<T> getModifiersOfType()
        {
            return FilteredModifierList<T>(first);
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

        /// True if has no modifiers
        bool isEmpty() const { return first == nullptr; }

        FilteredModifierList<Modifier>::Iterator begin()
        {
            return FilteredModifierList<Modifier>::Iterator(first);
        }
        FilteredModifierList<Modifier>::Iterator end()
        {
            return FilteredModifierList<Modifier>::Iterator(nullptr);
        }
    };

    class NamedExpressionType;
    class GenericDecl;
    class ContainerDecl;

    // Try to extract a simple integer value from an `IntVal`.
    // This fill assert-fail if the object doesn't represent a literal value.
    IntegerLiteralValue getIntVal(IntVal * val);

    /// Represents how much checking has been applied to a declaration.
    enum class DeclCheckState : uint8_t
    {
        /// The declaration has been parsed, but
        /// is otherwise completely unchecked.
        ///
        Unchecked,

        /// The declaration is parsed and inserted into the initial scope,
        /// ready for future lookups from within the parser for disambiguation purposes.
        ReadyForParserLookup,

        /// Basic checks on the modifiers of the declaration have been applied.
        ///
        /// For example, when a declaration has attributes, the transformation
        /// of an attribute from the parsed-but-unchecked form into a checked
        /// form (in which it has the appropriate C++ subclass) happens here.
        ///
        ModifiersChecked,

        /// Wiring up scopes of namespaces with their siblings defined in different
        /// files/modules, and other namespaces imported via `using`.
        ScopesWired,

        /// The type/signature of the declaration has been checked.
        ///
        /// For a value declaration like a variable or function, this means that
        /// the type of the declaration can be queried.
        ///
        /// For a type declaration like a `struct` or `typedef` this means
        /// that a `Type` referring to that declaration can be formed.
        ///
        SignatureChecked,

        /// The declaration's basic signature has been checked to the point that
        /// it is ready to be referenced in other places.
        ///
        /// For a function, this means that it has been organized into a
        /// "redeclration group" if there are multiple functions with the
        /// same name in a scope.
        ///
        ReadyForReference,

        /// The declaration is ready for lookup operations to be performed.
        ///
        /// For type declarations (e.g., aggregate types, generic type parameters)
        /// this means that any base type or constraint clauses have been
        /// sufficiently checked so that we can enumerate the inheritance
        /// hierarchy of the type and discover all its members.
        ///
        ReadyForLookup,

        /// Any conformance declared on the declaration have been validated.
        ///
        /// In particular, this step means that a "witness table" has been
        /// created to show  how a type satisfies the requirements of any
        /// interfaces it conforms to.
        ///
        ReadyForConformances,

        /// Any DeclRefTypes with substitutions have been fully resolved
        /// to concrete type. E.g. `T.X` with `T=A` should resolve to `A.X`.
        /// We need a separate pass to resolve these types because `A.X`
        /// maybe synthesized and made available only after conformance checking.
        TypesFullyResolved,

        /// All attributes are fully checked. This is the final step before
        /// checking the function body.
        AttributesChecked,

        /// The body/definition is checked.
        ///
        /// This step includes any validation of the declaration that is
        /// immaterial to clients code using the declaration, but that is
        /// nonetheless relevant to checking correctness.
        ///
        /// The canonical example here is checking the body of functions.
        /// Client code cannot depend on *how* a function is implemented,
        /// but we still need to (eventually) check the bodies of all
        /// functions, so it belongs in the last phase of checking.
        ///
        DefinitionChecked,
        DefaultConstructorReadyForUse = DefinitionChecked,

        /// The capabilities required by the decl is infered and validated.
        ///
        CapabilityChecked,

        // For convenience at sites that call `ensureDecl()`, we define
        // some aliases for the above states that are expressed in terms
        // of what client code needs to be able to do with a declaration.
        //
        // These aliases can be changed over time if we decide to add
        // more phases to semantic checking.

        CanEnumerateBases = ReadyForLookup,
        CanUseBaseOfInheritanceDecl = ReadyForLookup,
        CanUseTypeOfValueDecl = ReadyForReference,
        CanUseExtensionTargetType = ReadyForLookup,
        CanUseAsType = ReadyForReference,
        CanUseFuncSignature = ReadyForReference,
        CanSpecializeGeneric = ReadyForReference,
        CanReadInterfaceRequirements = ReadyForLookup,
    };

    /// A `DeclCheckState` plus a bit to track whether a declaration is currently being checked.
    struct DeclCheckStateExt
    {
        typedef uint8_t RawType;
        DeclCheckStateExt() {}
        DeclCheckStateExt(DeclCheckState state)
            : m_raw(uint8_t(state))
        {
        }

        enum : RawType
        {
            /// A flag to indicate that a declaration is being checked.
            ///
            /// The value of this flag is chosen so that it can be
            /// represented in the bits of a `DeclCheckState` without
            /// colliding with the bits that represent actual states.
            ///
            kBeingCheckedBit = 0x80,
        };

        DeclCheckState getState() const { return DeclCheckState(m_raw & ~kBeingCheckedBit); }
        void setState(DeclCheckState state) { m_raw = (m_raw & kBeingCheckedBit) | RawType(state); }

        bool isBeingChecked() const { return (m_raw & kBeingCheckedBit) != 0; }

        void setIsBeingChecked(bool isBeingChecked)
        {
            m_raw = (m_raw & ~kBeingCheckedBit) | (isBeingChecked ? kBeingCheckedBit : 0);
        }

        bool operator>=(DeclCheckState state) const { return getState() >= state; }

        RawType getRaw() const { return m_raw; }
        void setRaw(RawType raw) { m_raw = raw; }

        // TODO(JS):
        // Unfortunately for automatic serialization to see this member, it has to be public.
        // private:
        RawType m_raw = 0;
    };

    void addModifier(ModifiableSyntaxNode * syntax, Modifier * modifier);

    void removeModifier(ModifiableSyntaxNode * syntax, Modifier * modifier);

    FIDDLE()
    struct QualType
    {
        FIDDLE(...)
        Type* type = nullptr;
        bool isLeftValue = false;
        bool hasReadOnlyOnTarget = false;
        bool isWriteOnly = false;

        QualType() = default;

        QualType(Type* type);

        QualType(Type* type, bool isLVal)
            : QualType(type)
        {
            isLeftValue = isLVal;
        }


        Type* Ptr() { return type; }

        operator Type*() { return type; }
        Type* operator->() { return type; }
    };

    class ASTBuilder;

    struct SyntaxClassBase;
    typedef SyntaxClassBase ReflectClassInfo;
    typedef SyntaxClassBase ASTClassInfo;

    struct SyntaxClassInfo
    {
    public:
        char const* name;
        ASTNodeType firstTag;
        Count tagCount;
        void* (*createFunc)(ASTBuilder*);
        void (*destructFunc)(void*);

        template<typename T>
        static SyntaxClassInfo* get()
        {
            return const_cast<SyntaxClassInfo*>(&T::kSyntaxClassInfo);
        }
    };

    // A reference to a class of syntax node, that can be
    // used to create instances on the fly
    struct SyntaxClassBase
    {
        SyntaxClassBase() {}

        explicit SyntaxClassBase(ASTNodeType tag);

        SyntaxClassBase(SyntaxClassInfo const* info)
            : _info(info)
        {
        }


        ASTNodeType getTag() const { return getInfo()->firstTag; }
        UnownedTerminatedStringSlice getName() const;

        void* createInstanceImpl(ASTBuilder* astBuilder) const;
        void destructInstanceImpl(void* instance) const;

        bool isSubClassOf(SyntaxClassBase const& super) const;

        typedef SyntaxClassInfo Info;

        Info* getInfo() const { return const_cast<Info*>(_info); }
        operator Info*() const { return const_cast<Info*>(_info); }


        bool operator==(SyntaxClassBase const& other) const { return _info == other._info; }

        bool operator!=(SyntaxClassBase const& other) const { return _info != other._info; }

    private:
        Info const* _info = nullptr;
    };

    template<typename T>
    struct SyntaxClass;

    template<typename T>
    SyntaxClass<T> getSyntaxClass();

    template<typename T = NodeBase>
    struct SyntaxClass : SyntaxClassBase
    {
        SyntaxClass() {}

        template<typename U>
        SyntaxClass(
            SyntaxClass<U> const& other,
            typename EnableIf<IsConvertible<T*, U*>::Value, void>::type* = 0)
            : SyntaxClassBase(other)
        {
        }

        explicit SyntaxClass(SyntaxClassBase const& other)
            : SyntaxClassBase(other)
        {
        }

        explicit SyntaxClass(ASTNodeType tag)
            : SyntaxClassBase(tag)
        {
        }

        explicit SyntaxClass(SyntaxClassInfo const* info)
            : SyntaxClassBase(info)
        {
        }

        T* createInstance(ASTBuilder* astBuilder) const
        {
            return (T*)createInstanceImpl(astBuilder);
        }
        void destructInstance(T* instance) { destructInstanceImpl(instance); }

        bool isSubClassOf(SyntaxClassBase const& other)
        {
            return SyntaxClassBase::isSubClassOf(other);
        }

        template<typename U>
        bool isSubClassOf()
        {
            return SyntaxClassBase::isSubClassOf(getSyntaxClass<U>());
        }
    };

    template<typename T>
    SyntaxClass<T> getSyntaxClass()
    {
        return SyntaxClass<T>(SyntaxClassInfo::get<T>());
    }

    struct SubstitutionSet
    {
        DeclRefBase* declRef = nullptr;

        // The element index if the substitution is happening inside a pack expansion.
        // For example, if we are substituting the pattern type of `expand each T`, where
        // `T` is a type pack, then packExpansionIndex will have a value starting from 0
        // to the count of the type pack during expansion of the `expand` type when we
        // substitute `each T` with the element of `T` at index `packExpansionIndex`.
        int packExpansionIndex = -1;

        SubstitutionSet() = default;
        SubstitutionSet(DeclRefBase* declRefBase)
            : declRef(declRefBase)
        {
        }
        explicit operator bool() const;

        template<typename F>
        void forEachGenericSubstitution(F func) const;

        template<typename F>
        void forEachSubstitutionArg(F func) const;

        Type* applyToType(ASTBuilder* astBuilder, Type* type) const;
        DeclRefBase* applyToDeclRef(ASTBuilder* astBuilder, DeclRefBase* declRef) const;

        LookupDeclRef* findLookupDeclRef() const;
        GenericAppDeclRef* findGenericAppDeclRef(GenericDecl* genericDecl) const;
        GenericAppDeclRef* findGenericAppDeclRef() const;
        DeclRefBase* getInnerMostNodeWithSubstInfo() const;
    };

    /// An expression together with (optional) substutions to apply to it
    ///
    /// Under the hood this is a pair of an `Expr*` and a `SubstitutionSet`.
    /// Conceptually it represents the result of applying the substitutions,
    /// recursively, to the given expression.
    ///
    /// `SubstExprBase` exists primarily to provide a non-templated base type
    /// for `SubstExpr<T>`. Code should prefer to use `SubstExpr<Expr>` instead
    /// of `SubstExprBase` as often as possible.
    ///
    struct SubstExprBase
    {
    public:
        /// Initialize as a null expression
        SubstExprBase() {}

        /// Initialize as the given `expr` with no subsitutions applied
        SubstExprBase(Expr* expr)
            : m_expr(expr)
        {
        }

        /// Initialize as the given `expr` with the given `substs` applied
        SubstExprBase(Expr* expr, SubstitutionSet const& substs)
            : m_expr(expr), m_substs(substs)
        {
        }

        /// Get the underlying expression without any substitutions
        Expr* getExpr() const { return m_expr; }

        /// Get the subsitutions being applied, if any
        SubstitutionSet const& getSubsts() const { return m_substs; }

    private:
        Expr* m_expr = nullptr;
        SubstitutionSet m_substs;

        typedef void (SubstExprBase::*SafeBool)();
        void SafeBoolTrue() {}

    public:
        /// Test whether this is a non-null expression
        operator SafeBool() { return m_expr ? &SubstExprBase::SafeBoolTrue : nullptr; }

        /// Test whether this is a null expression
        bool operator!() const { return m_expr == nullptr; }
    };

    /// An expression together with (optional) substutions to apply to it
    ///
    /// Under the hood this is a pair of an `T*` (there `T: Expr`) and a `SubstitutionSet`.
    /// Conceptually it represents the result of applying the substitutions,
    /// recursively, to the given expression.
    ///
    template<typename T>
    struct SubstExpr : SubstExprBase
    {
    private:
        typedef SubstExprBase Super;

    public:
        /// Initialize as a null expression
        SubstExpr() {}

        /// Initialize as the given `expr` with no subsitutions applied
        SubstExpr(T* expr)
            : Super(expr)
        {
        }

        /// Initialize as the given `expr` with the given `substs` applied
        SubstExpr(T* expr, SubstitutionSet const& substs)
            : Super(expr, substs)
        {
        }

        /// Initialize as a copy of the given `other` expression
        template<typename U>
        SubstExpr(
            SubstExpr<U> const& other,
            typename EnableIf<IsConvertible<T*, U*>::Value, void>::type* = 0)
            : Super(other.getExpr(), other.getSubsts())
        {
        }

        /// Get the underlying expression without any substitutions
        T* getExpr() const { return (T*)Super::getExpr(); }

        /// Dynamic cast to an expression of type `U`
        ///
        /// Returns a null expression if the cast fails, or if this expression was null.
        template<typename U>
        SubstExpr<U> as()
        {
            return SubstExpr<U>(Slang::as<U>(getExpr()), getSubsts());
        }
    };

    SubstExpr<Expr> applySubstitutionToExpr(SubstitutionSet substSet, Expr * expr);

    class ASTBuilder;

    template<typename T>
    struct DeclRef;
    Module* getModule(Decl * decl);


    // If this is a declref to an associatedtype with a ThisTypeSubsitution,
    // try to find the concrete decl that satisfies the associatedtype requirement from the
    // concrete type supplied by ThisTypeSubstittution.
    Val* _tryLookupConcreteAssociatedTypeFromThisTypeSubst(
        ASTBuilder * builder,
        DeclRef<Decl> declRef);

    template<typename T = Decl>
    struct DeclRef
    {
        friend class ASTBuilder;

    public:
        typedef T DeclType;
        DeclRefBase* declRefBase;
        DeclRef()
            : declRefBase(nullptr)
        {
        }

        void init(DeclRefBase* base);

        DeclRef(Decl* decl);

        DeclRef(DeclRefBase* base) { init(base); }

        template<typename U, typename = typename EnableIf<IsConvertible<T*, U*>::Value, void>::type>
        DeclRef(DeclRef<U> const& other)
            : declRefBase(other.declRefBase)
        {
        }

        T* getDecl() const;

        Name* getName() const;

        SourceLoc getNameLoc() const;
        SourceLoc getLoc() const;
        DeclRef<ContainerDecl> getParent() const;
        HashCode getHashCode() const;
        Type* substitute(ASTBuilder* astBuilder, Type* type) const;

        SubstExpr<Expr> substitute(ASTBuilder* astBuilder, Expr* expr) const;

        // Apply substitutions to a type or declaration
        template<typename U>
        DeclRef<U> substitute(ASTBuilder* astBuilder, DeclRef<U> declRef) const;

        // Apply substitutions to this declaration reference
        DeclRef<T> substituteImpl(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff) const;

        template<typename U>
        DeclRef<U> as() const
        {
            DeclRef<U> result = DeclRef<U>(declRefBase);
            return result;
        }

        template<typename U>
        bool is() const
        {
            return Slang::as<U>(static_cast<NodeBase*>(getDecl())) != nullptr;
        }

        operator DeclRefBase*() const { return declRefBase; }

        operator DeclRef<Decl>() const { return DeclRef<Decl>(declRefBase); }

        template<typename U>
        bool equals(DeclRef<U> other) const
        {
            return declRefBase == other.declRefBase;
        }

        template<typename U>
        bool operator==(DeclRef<U> other) const
        {
            return equals(other);
        }

        template<typename U>
        bool operator!=(DeclRef<U> other) const
        {
            return !equals(other);
        }

        explicit operator bool() const { return declRefBase; }
    };

    template<typename T>
    inline DeclRef<T> makeDeclRef(T * decl)
    {
        return DeclRef<T>(decl);
    }

    SubstExpr<Expr> substituteExpr(SubstitutionSet const& substs, Expr* expr);
    DeclRef<Decl> substituteDeclRef(
        SubstitutionSet const& substs,
        ASTBuilder* astBuilder,
        DeclRef<Decl> const& declRef);
    Type* substituteType(SubstitutionSet const& substs, ASTBuilder* astBuilder, Type* type);

    enum class MemberFilterStyle
    {
        All,      ///< All members
        Instance, ///< Only instance members
        Static,   ///< Only static (ie non instance) members
    };

    Decl* const* adjustFilterCursorImpl(
        const ReflectClassInfo& clsInfo,
        MemberFilterStyle filterStyle,
        Decl* const* ptr,
        Decl* const* end);
    Decl* const* getFilterCursorByIndexImpl(
        const ReflectClassInfo& clsInfo,
        MemberFilterStyle filterStyle,
        Decl* const* ptr,
        Decl* const* end,
        Index index);
    Index getFilterCountImpl(
        const ReflectClassInfo& clsInfo,
        MemberFilterStyle filterStyle,
        Decl* const* ptr,
        Decl* const* end);


    template<typename T>
    Decl* const* adjustFilterCursor(
        MemberFilterStyle filterStyle,
        Decl* const* ptr,
        Decl* const* end)
    {
        return adjustFilterCursorImpl(getSyntaxClass<T>(), filterStyle, ptr, end);
    }

    /// Finds the element at index. If there is no element at the index (for example has too few
    /// elements), returns nullptr.
    template<typename T>
    Decl* const* getFilterCursorByIndex(
        MemberFilterStyle filterStyle,
        Decl* const* ptr,
        Decl* const* end,
        Index index)
    {
        return getFilterCursorByIndexImpl(getSyntaxClass<T>(), filterStyle, ptr, end, index);
    }

    template<typename T>
    Index getFilterCount(MemberFilterStyle filterStyle, Decl* const* ptr, Decl* const* end)
    {
        return getFilterCountImpl(getSyntaxClass<T>(), filterStyle, ptr, end);
    }

    template<typename T>
    bool isFilterNonEmpty(MemberFilterStyle filterStyle, Decl* const* ptr, Decl* const* end)
    {
        return adjustFilterCursorImpl(getSyntaxClass<T>(), filterStyle, ptr, end) != end;
    }

    template<typename T>
    struct FilteredMemberList
    {
        typedef Decl* Element;

        FilteredMemberList()
            : m_begin(nullptr), m_end(nullptr)
        {
        }

        explicit FilteredMemberList(
            List<Element> const& list,
            MemberFilterStyle filterStyle = MemberFilterStyle::All)
            : m_begin(adjustFilterCursor<T>(filterStyle, list.begin(), list.end()))
            , m_end(list.end())
            , m_filterStyle(filterStyle)
        {
        }

        struct Iterator
        {
            const Element* m_cursor;
            const Element* m_end;
            MemberFilterStyle m_filterStyle;

            bool operator!=(Iterator const& other) const { return m_cursor != other.m_cursor; }

            void operator++()
            {
                m_cursor = adjustFilterCursor<T>(m_filterStyle, m_cursor + 1, m_end);
            }

            T* operator*() { return static_cast<T*>(*m_cursor); }
        };

        Iterator begin()
        {
            Iterator iter = {m_begin, m_end, m_filterStyle};
            return iter;
        }

        Iterator end()
        {
            Iterator iter = {m_end, m_end, m_filterStyle};
            return iter;
        }

        // TODO(tfoley): It is ugly to have these.
        // We should probably fix the call sites instead.
        T* getFirst() { return *begin(); }
        Index getCount() { return getFilterCount<T>(m_filterStyle, m_begin, m_end); }

        T* operator[](Index index) const
        {
            Decl* const* ptr = getFilterCursorByIndex<T>(m_filterStyle, m_begin, m_end, index);
            SLANG_ASSERT(ptr);
            return static_cast<T*>(*ptr);
        }

        /// Returns true if empty (equivalent to getCount() == 0)
        bool isEmpty() const
        {
            /// Note we don't have to scan, because m_begin has already been adjusted, when the
            /// FilteredMemberList is constructed
            return m_begin == m_end;
        }
        /// Returns true if non empty (equivalent to getCount() != 0 but faster)
        bool isNonEmpty() const { return !isEmpty(); }

        List<T*> toList()
        {
            List<T*> result;
            for (auto element : (*this))
            {
                result.add(element);
            }
            return result;
        }

        const Element*
            m_begin; ///< Is either equal to m_end, or points to first *valid* filtered member
        const Element* m_end;
        MemberFilterStyle m_filterStyle;
    };

    struct TransparentMemberInfo
    {
        // The declaration of the transparent member
        Decl* decl = nullptr;
    };

    template<typename T>
    struct FilteredMemberRefList
    {
        List<Decl*> const& m_decls;
        DeclRef<Decl> m_parent;
        MemberFilterStyle m_filterStyle;
        ASTBuilder* m_astBuilder;

        FilteredMemberRefList(
            ASTBuilder* astBuilder,
            List<Decl*> const& decls,
            DeclRef<Decl> parent,
            MemberFilterStyle filterStyle = MemberFilterStyle::All)
            : m_decls(decls), m_parent(parent), m_filterStyle(filterStyle), m_astBuilder(astBuilder)
        {
        }

        Index getCount() const
        {
            return getFilterCount<T>(m_filterStyle, m_decls.begin(), m_decls.end());
        }

        /// True if empty (equivalent to getCount == 0, but faster)
        bool isEmpty() const { return !isNonEmpty(); }
        /// True if non empty (equivalent to getCount() != 0 but faster)
        bool isNonEmpty() const
        {
            return isFilterNonEmpty<T>(m_filterStyle, m_decls.begin(), m_decls.end());
        }

        DeclRef<T> getFirstOrNull() { return isEmpty() ? DeclRef<T>() : (*this)[0]; }

        DeclRef<T> operator[](Index index) const
        {
            Decl* const* decl =
                getFilterCursorByIndex<T>(m_filterStyle, m_decls.begin(), m_decls.end(), index);
            SLANG_ASSERT(decl);
            return _getMemberDeclRef(m_astBuilder, m_parent, (T*)*decl).template as<T>();
        }

        List<DeclRef<T>> toArray() const
        {
            List<DeclRef<T>> result;
            for (auto d : *this)
                result.add(d);
            return result;
        }

        struct Iterator
        {
            FilteredMemberRefList const* m_list;
            Decl* const* m_ptr;
            Decl* const* m_end;
            MemberFilterStyle m_filterStyle;

            Iterator()
                : m_list(nullptr), m_ptr(nullptr), m_filterStyle(MemberFilterStyle::All)
            {
            }
            Iterator(
                FilteredMemberRefList const* list,
                Decl* const* ptr,
                Decl* const* end,
                MemberFilterStyle filterStyle)
                : m_list(list), m_ptr(ptr), m_end(end), m_filterStyle(filterStyle)
            {
            }

            bool operator!=(const Iterator& other) const { return m_ptr != other.m_ptr; }

            void operator++() { m_ptr = adjustFilterCursor<T>(m_filterStyle, m_ptr + 1, m_end); }

            DeclRef<T> operator*()
            {
                return _getMemberDeclRef(m_list->m_astBuilder, m_list->m_parent, (T*)*m_ptr)
                    .template as<T>();
            }
        };

        Iterator begin() const
        {
            return Iterator(
                this,
                adjustFilterCursor<T>(m_filterStyle, m_decls.begin(), m_decls.end()),
                m_decls.end(),
                m_filterStyle);
        }
        Iterator end() const { return Iterator(this, m_decls.end(), m_decls.end(), m_filterStyle); }
    };

    //
    // type Expressions
    //

    // A "type expression" is a term that we expect to resolve to a type during checking.
    // We store both the original syntax and the resolved type here.
    FIDDLE()
    struct TypeExp
    {
        FIDDLE(...)
        typedef TypeExp ThisType;

        TypeExp() {}
        TypeExp(TypeExp const& other)
            : exp(other.exp), type(other.type)
        {
        }
        explicit TypeExp(Expr* exp)
            : exp(exp)
        {
        }
        explicit TypeExp(Type* type)
            : type(type)
        {
        }
        TypeExp(Expr* exp, Type* type)
            : exp(exp), type(type)
        {
        }

        Expr* exp = nullptr;
        Type* type = nullptr;

        bool equals(Type* other);

        Type* Ptr() { return type; }
        operator Type*() { return type; }
        Type* operator->() { return Ptr(); }

        ThisType& operator=(const ThisType& rhs) = default;

        /// A global immutable TypeExp, that has no type or exp set.
        static const TypeExp empty;
    };

    // Masks to be applied when lookup up declarations
    enum class LookupMask : uint8_t
    {
        type = 0x1,
        Function = 0x2,
        Value = 0x4,
        Attribute = 0x8,
        SyntaxDecl = 0x10,
        Default = type | Function | Value | SyntaxDecl,
    };

    /// Flags for options to be used when looking up declarations
    enum class LookupOptions : uint8_t
    {
        None = 0,
        IgnoreBaseInterfaces = 1 << 0,
        Completion = 1 << 1, ///< Lookup all applicable decls for code completion suggestions
        NoDeref = 1 << 2,
        ConsiderAllLocalNamesInScope = 1 << 3,
        ///^ Normally we rely on the checking state of local names to determine
        /// if they have been declared. If the scopes are currently
        /// "under-construction" and not being checked, then it's safe to
        /// consider all names we've inserted so far. This is used when
        /// checking to see if a keyword is shadowed.
        IgnoreInheritance =
            1 << 4, ///< Lookup only non inheritance children of a struct (including `extension`)
        IgnoreTransparentMembers = 1 << 5,
    };
    inline LookupOptions operator&(LookupOptions a, LookupOptions b)
    {
        return (LookupOptions)((std::underlying_type_t<LookupOptions>)a &
                               (std::underlying_type_t<LookupOptions>)b);
    }

    class LookupResultItem_Breadcrumb : public RefObject
    {
    public:
        enum class Kind : uint8_t
        {
            // The lookup process looked "through" an in-scope
            // declaration to the fields inside of it, so that
            // even if lookup started with a simple name `f`,
            // it needs to result in a member expression `obj.f`.
            Member,

            // The lookup process took a pointer(-like) value, and then
            // proceeded to derefence it and look at the thing(s)
            // it points to instead, so that the final expression
            // needs to have `(*obj)`
            Deref,

            // The lookup process saw a value `obj` of type `T` and
            // took into account an in-scope constraint that says
            // `T` is a subtype of some other type `U`, so that
            // lookup was able to find a member through type `U`
            // instead.
            SuperType,

            // The lookup process considered a member of an
            // enclosing type as being in scope, so that any
            // reference to that member needs to use a `this`
            // expression as appropriate.
            This,
        };

        // The kind of lookup step that was performed
        Kind kind;

        // For the `Kind::This` case, what does the implicit
        // `this` or `This` parameter refer to?
        //
        enum class ThisParameterMode : uint8_t
        {
            ImmutableValue, // An immutable `this` value
            MutableValue,   // A mutable `this` value
            Type,           // A `This` type

            Default = ImmutableValue,
        };
        ThisParameterMode thisParameterMode = ThisParameterMode::Default;

        // As needed, a reference to the declaration that faciliated
        // the lookup step.
        //
        // For a `Member` lookup step, this is the declaration whose
        // members were implicitly pulled into scope.
        //
        // For a `Constraint` lookup step, this is the `ConstraintDecl`
        // that serves to witness the subtype relationship.
        //
        DeclRef<Decl> declRef;

        Val* val = nullptr;

        // The next implicit step that the lookup process took to
        // arrive at a final value.
        RefPtr<LookupResultItem_Breadcrumb> next;

        LookupResultItem_Breadcrumb(
            Kind kind,
            DeclRef<Decl> declRef,
            Val* val,
            RefPtr<LookupResultItem_Breadcrumb> next,
            ThisParameterMode thisParameterMode = ThisParameterMode::Default)
            : kind(kind)
            , thisParameterMode(thisParameterMode)
            , declRef(declRef)
            , val(val)
            , next(next)
        {
        }

    protected:
        // Needed for serialization
        LookupResultItem_Breadcrumb() = default;
    };

    // Represents one item found during lookup
    struct LookupResultItem
    {
        typedef LookupResultItem_Breadcrumb Breadcrumb;

        // Sometimes lookup finds an item, but there were additional
        // "hops" taken to reach it. We need to remember these steps
        // so that if/when we consturct a full expression we generate
        // appropriate AST nodes for all the steps.
        //
        // We build up a list of these "breadcrumbs" while doing
        // lookup, and store them alongside each item found.
        //
        // As an example, suppose we have an HLSL `cbuffer` declaration:
        //
        //     cbuffer C { float4 f; }
        //
        // This is syntax sugar for a global-scope variable of
        // type `ConstantBuffer<T>` where `T` is a `struct` containing
        // all the members:
        //
        //     struct Anon0 { float4 f; };
        //     __transparent ConstantBuffer<Anon0> anon1;
        //
        // The `__transparent` modifier there captures the fact that
        // when somebody writes `f` in their code, they expect it to
        // "see through" the `cbuffer` declaration (or the global variable,
        // in this case) and find the member inside.
        //
        // But when the user writes `f` we can't just create a simple
        // `VarExpr` that refers directly to that field, because that
        // doesn't actually reflect the required steps in a way that
        // code generation can use.
        //
        // Instead we need to construct an expression like `(*anon1).f`,
        // where there is are two additional steps in the process:
        //
        // 1. We needed to dereference the pointer-like type `ConstantBuffer<Anon0>`
        //    to get at a value of type `Anon0`
        // 2. We needed to access a sub-field of the aggregate type `Anon0`
        //
        // We *could* just create these full-formed expressions during
        // lookup, but this might mean creating a large number of
        // AST nodes in cases where the user calls an overloaded function.
        // At the very least we'd rather not heap-allocate in the common
        // case where no "extra" steps need to be performed to get to
        // the declarations.
        //
        // This is where "breadcrumbs" come in. A breadcrumb represents
        // an extra "step" that must be performed to turn a declaration
        // found by lookup into a valid expression to splice into the
        // AST. Most of the time lookup result items don't have any
        // breadcrumbs, so that no extra heap allocation takes place.
        // When an item does have breadcrumbs, and it is chosen as
        // the unique result (perhaps by overload resolution), then
        // we can walk the list of breadcrumbs to create a full
        // expression.


        // A properly-specialized reference to the declaration that was found.
        DeclRef<Decl> declRef;

        // Any breadcrumbs needed in order to turn that declaration
        // reference into a well-formed expression.
        //
        // This is unused in the simple case where a declaration
        // is being referenced directly (rather than through
        // transparent members).
        RefPtr<LookupResultItem_Breadcrumb> breadcrumbs;

        LookupResultItem() = default;
        explicit LookupResultItem(DeclRef<Decl> declRef)
            : declRef(declRef)
        {
        }
        LookupResultItem(DeclRef<Decl> declRef, RefPtr<Breadcrumb> breadcrumbs)
            : declRef(declRef), breadcrumbs(breadcrumbs)
        {
        }
    };


    // Result of looking up a name in some lexical/semantic environment.
    // Can be used to enumerate all the declarations matching that name,
    // in the case where the result is overloaded.
    struct LookupResult
    {
        // The one item that was found, in the simple case
        LookupResultItem item;

        // All of the items that were found, in the complex case.
        // Note: if there was no overloading, then this list isn't
        // used at all, to avoid allocation.
        //
        // Additionally, if `items` is used, then `item` *must* hold an item that
        // is also in the items list (typically the first entry), as an invariant.
        // Otherwise isValid/begin will not function correctly.
        List<LookupResultItem> items;

        // Was at least one result found?
        bool isValid() const { return item.declRef.getDecl() != nullptr; }

        bool isOverloaded() const { return items.getCount() > 1; }

        Name* getName() const
        {
            return items.getCount() > 1 ? items[0].declRef.getName() : item.declRef.getName();
        }
        LookupResultItem* begin() const
        {
            if (isValid())
            {
                if (isOverloaded())
                    return const_cast<LookupResultItem*>(items.begin());
                else
                    return const_cast<LookupResultItem*>(&item);
            }
            else
                return nullptr;
        }
        LookupResultItem* end() const
        {
            if (isValid())
            {
                if (isOverloaded())
                    return const_cast<LookupResultItem*>(items.end());
                else
                    return const_cast<LookupResultItem*>(&item + 1);
            }
            else
                return nullptr;
        }
    };

    // A helper to avoid having to include slang-check-impl.h in slang-syntax.h
    struct SemanticsVisitor;
    ASTBuilder* semanticsVisitorGetASTBuilder(SemanticsVisitor*);

    struct LookupRequest
    {
        SemanticsVisitor* semantics = nullptr;
        Scope* scope = nullptr;
        Scope* endScope = nullptr;

        // A decl to exclude from the lookup, used to exclude the current decl being checked, such
        // as in typedef Foo Foo; to avoid finding itself.
        Decl* declToExclude = nullptr;
        LookupMask mask = LookupMask::Default;
        LookupOptions options = LookupOptions::None;

        bool isCompletionRequest() const
        {
            return (options & LookupOptions::Completion) != LookupOptions::None;
        }
        bool shouldConsiderAllLocalNames() const
        {
            return (options & LookupOptions::ConsiderAllLocalNamesInScope) != LookupOptions::None;
        }
    };

    class WitnessTable;

    // A value that witnesses the satisfaction of an interface
    // requirement by a particular declaration or value.
    struct RequirementWitness
    {
        RequirementWitness()
            : m_flavor(Flavor::none)
        {
        }

        RequirementWitness(DeclRefBase* declRef)
            : m_flavor(Flavor::declRef), m_declRef(declRef)
        {
        }

        RequirementWitness(Val* val);

        RequirementWitness(RefPtr<WitnessTable> witnessTable);

        enum class Flavor
        {
            none,
            declRef,
            val,
            witnessTable,
        };

        Flavor getFlavor() const { return m_flavor; }

        DeclRef<Decl> getDeclRef()
        {
            SLANG_ASSERT(getFlavor() == Flavor::declRef);
            return m_declRef;
        }

        Val* getVal()
        {
            SLANG_ASSERT(getFlavor() == Flavor::val);
            return m_val;
        }

        RefPtr<WitnessTable> getWitnessTable();

        RequirementWitness specialize(ASTBuilder* astBuilder, SubstitutionSet const& subst);

        Flavor m_flavor;
        DeclRef<Decl> m_declRef;
        RefPtr<RefObject> m_obj;
        Val* m_val = nullptr;
    };

    typedef OrderedDictionary<Decl*, RequirementWitness> RequirementDictionary;

    FIDDLE()
    class WitnessTable : public RefObject
    {
        FIDDLE(...)
        const RequirementDictionary& getRequirementDictionary() { return m_requirementDictionary; }

        void add(Decl* decl, RequirementWitness const& witness);

        // The type that the witness table witnesses conformance to (e.g. an Interface)
        Type* baseType;

        // The type witnessesd by the witness table (a concrete type).
        Type* witnessedType;

        // Whether or not this witness table is an extern declaration.
        bool isExtern = false;

        // Cached dictionary for looking up satisfying values.
        RequirementDictionary m_requirementDictionary;

        RefPtr<WitnessTable> specialize(ASTBuilder* astBuilder, SubstitutionSet const& subst);
    };

    struct SpecializationParam
    {
        enum class Flavor
        {
            GenericType,
            GenericValue,
            ExistentialType,
            ExistentialValue,
        };
        Flavor flavor;
        SourceLoc loc;
        NodeBase* object = nullptr;
    };
    typedef List<SpecializationParam> SpecializationParams;

    struct SpecializationArg
    {
        Val* val = nullptr;
    };
    typedef List<SpecializationArg> SpecializationArgs;

    struct ExpandedSpecializationArg : SpecializationArg
    {
        Val* witness = nullptr;
    };
    typedef List<ExpandedSpecializationArg> ExpandedSpecializationArgs;

    /// A reference-counted object to hold a list of candidate extensions
    /// that might be applicable to a type based on its declaration.
    ///
    FIDDLE()
    class CandidateExtensionList : public RefObject
    {
        FIDDLE(...)
        List<ExtensionDecl*> candidateExtensions;
    };


    enum class DeclAssociationKind
    {
        ForwardDerivativeFunc,
        BackwardDerivativeFunc,
        PrimalSubstituteFunc
    };

    FIDDLE()
    class DeclAssociation : public RefObject
    {
        FIDDLE(...)
        DeclAssociationKind kind;
        Decl* decl;
    };

    /// A reference-counted object to hold a list of associated decls for a decl.
    ///
    FIDDLE()
    class DeclAssociationList : public RefObject
    {
        FIDDLE(...)
        List<RefPtr<DeclAssociation>> associations;
    };

    /// Represents the "direction" that a parameter is being passed (e.g., `in` or `out`
    enum ParameterDirection
    {
        kParameterDirection_In,       ///< Copy in
        kParameterDirection_Out,      ///< Copy out
        kParameterDirection_InOut,    ///< Copy in, copy out
        kParameterDirection_Ref,      ///< By-reference
        kParameterDirection_ConstRef, ///< By-const-reference
    };

    void printDiagnosticArg(StringBuilder & sb, ParameterDirection direction);

    /// The kind of a builtin interface requirement that can be automatically synthesized.
    enum class BuiltinRequirementKind
    {
        DefaultInitializableConstructor, ///< The `IDefaultInitializable.__init()` method

        DifferentialType,    ///< The `IDifferentiable.Differential` associated type requirement
        DifferentialPtrType, ///< The `IDifferentiable.DifferentialPtr` associated type requirement
        DZeroFunc,           ///< The `IDifferentiable.dzero` function requirement
        DAddFunc,            ///< The `IDifferentiable.dadd` function requirement
        DMulFunc,            ///< The `IDifferentiable.dmul` function requirement

        InitLogicalFromInt, ///< The `ILogical.__init` mtehod.
        Equals,             ///< The `ILogical.equals` mtehod.
        LessThan,           ///< The `ILogical.lessThan` mtehod.
        LessThanOrEquals,   ///< The `ILogical.lessThanOrEquals` mtehod.
        Shl,                ///< The `ILogical.shl` mtehod.
        Shr,                ///< The `ILogical.shr` mtehod.
        BitAnd,             ///< The `ILogical.bitAnd` mtehod.
        BitOr,              ///< The `ILogical.bitOr` mtehod.
        BitXor,             ///< The `ILogical.bitXor` mtehod.
        BitNot,             ///< The `ILogical.bitNot` mtehod.
        And,                ///< The `ILogical.and` mtehod.
        Or,                 ///< The `ILogical.or` mtehod.
        Not,                ///< The `ILogical.not` mtehod.
    };

    enum class FunctionDifferentiableLevel
    {
        None,
        Forward,
        Backward
    };

    /// Represents a markup (documentation) associated with a decl.
    FIDDLE()
    class MarkupEntry : public RefObject
    {
        FIDDLE(...)
        NodeBase* m_node; ///< The node this documentation is associated with
        String m_markup;  ///< The raw contents of of markup associated with the decoration
        MarkupVisibility m_visibility = MarkupVisibility::Public; ///< How visible this decl is
    };

    /// Get the inner most expr from an higher order expr chain, e.g. `__fwd_diff(__fwd_diff(f))`'s
    /// inner most expr is `f`.
    Expr* getInnerMostExprFromHigherOrderExpr(
        Expr * expr,
        FunctionDifferentiableLevel & outDiffLevel);
    inline Expr* getInnerMostExprFromHigherOrderExpr(Expr * expr)
    {
        FunctionDifferentiableLevel level;
        return getInnerMostExprFromHigherOrderExpr(expr, level);
    }


    /// Get the operator name from the higher order invoke expr.
    UnownedStringSlice getHigherOrderOperatorName(HigherOrderInvokeExpr * expr);

    enum class DeclVisibility
    {
        Private,
        Internal,
        Public,
        Default = Internal,
    };

} // namespace Slang

#endif
