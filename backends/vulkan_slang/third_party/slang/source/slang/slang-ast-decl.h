// slang-decl-defs.h

#pragma once

#include "slang-ast-base.h"
#include "slang-ast-decl.h.fiddle"

FIDDLE()
namespace Slang
{

// Syntax class definitions for declarations.

// A group of declarations that should be treated as a unit
FIDDLE()
class DeclGroup : public DeclBase
{
    FIDDLE(...)
    FIDDLE() List<Decl*> decls;
};

FIDDLE()
class UnresolvedDecl : public Decl
{
    FIDDLE(...)
};

// A "container" decl is a parent to other declarations
FIDDLE(abstract)
class ContainerDecl : public Decl
{
    FIDDLE(...)

    FIDDLE() List<Decl*> members;
    SourceLoc closingSourceLoc;

    // The associated scope owned by this decl.
    Scope* ownedScope = nullptr;

    template<typename T>
    FilteredMemberList<T> getMembersOfType()
    {
        return FilteredMemberList<T>(members);
    }

    void buildMemberDictionary();

    bool isMemberDictionaryValid() const { return dictionaryLastCount == members.getCount(); }

    void invalidateMemberDictionary()
    {
        dictionaryLastCount = -1;
        mapDeclMemberToIndex.clear();
    }

    Dictionary<Name*, Decl*>& getMemberDictionary()
    {
        buildMemberDictionary();
        return memberDictionary;
    }

    List<TransparentMemberInfo>& getTransparentMembers()
    {
        buildMemberDictionary();
        return transparentMembers;
    }

    void addMember(Decl* member)
    {
        if (member)
        {
            member->parentDecl = this;
            auto index = members.getCount();
            members.add(member);
            mapDeclMemberToIndex[member] = index;
        }
    }

    static void setParent(ContainerDecl* parent, Decl* child)
    {
        if (child)
            child->parentDecl = parent;
        if (parent)
            parent->addMember(child);
    }

    Index getDeclIndex(Decl* d);

    SLANG_UNREFLECTED // We don't want to reflect the following fields

        private :
        // Denotes how much of Members has been placed into the dictionary/transparentMembers.
        // If this value equals the Members.getCount(), the dictionary is completely full and valid.
        // If it's >= 0, then the Members after dictionaryLastCount are all that need to be added.
        // If it < 0 it means that the dictionary/transparentMembers is invalid and needs to be
        // recreated.
        Index dictionaryLastCount = 0;

    // Dictionary for looking up members by name.
    // This is built on demand before performing lookup.
    Dictionary<Name*, Decl*> memberDictionary;

    Dictionary<Decl*, Index> mapDeclMemberToIndex;

    // A list of transparent members, to be used in lookup
    // Note: this is only valid if `memberDictionaryIsValid` is true
    List<TransparentMemberInfo> transparentMembers;
};

// Base class for all variable declarations
FIDDLE(abstract)
class VarDeclBase : public Decl
{
    FIDDLE(...)

    // type of the variable
    FIDDLE() TypeExp type;

    Type* getType() { return type.type; }

    // Initializer expression (optional)
    FIDDLE() Expr* initExpr = nullptr;

    // Folded IntVal if the initializer is a constant integer.
    FIDDLE() IntVal* val = nullptr;
};

// Ordinary potentially-mutable variables (locals, globals, and member variables)
FIDDLE()
class VarDecl : public VarDeclBase
{
    FIDDLE(...)
};

// A variable declaration that is always immutable (whether local, global, or member variable)
FIDDLE()
class LetDecl : public VarDecl
{
    FIDDLE(...)
};

// An `AggTypeDeclBase` captures the shared functionality
// between true aggregate type declarations and extension
// declarations:
//
// - Both can contain members (they are `ContainerDecl`s)
// - Both can have declared bases
// - Both expose a `this` variable in their body
//
FIDDLE(abstract)
class AggTypeDeclBase : public ContainerDecl
{
    FIDDLE(...)
};

// An extension to apply to an existing type
FIDDLE()
class ExtensionDecl : public AggTypeDeclBase
{
    FIDDLE(...)
    FIDDLE() TypeExp targetType;
};

enum class TypeTag
{
    None = 0,
    Unsized = 1,
    Incomplete = 2,
    LinkTimeSized = 4,
    Opaque = 8,
};

// Declaration of a type that represents some sort of aggregate
FIDDLE(abstract)
class AggTypeDecl : public AggTypeDeclBase
{
    FIDDLE(...)
    FIDDLE() TypeTag typeTags = TypeTag::None;

    // Used if this type declaration is a wrapper, i.e. struct FooWrapper:IFoo = Foo;
    TypeExp wrappedType;
    bool hasBody = true;

    void unionTagsWith(TypeTag other);
    void addTag(TypeTag tag);
    bool hasTag(TypeTag tag);

    FilteredMemberList<VarDecl> getFields() { return getMembersOfType<VarDecl>(); }
};

FIDDLE()
class StructDecl : public AggTypeDecl
{
    FIDDLE(...)
    SLANG_UNREFLECTED
    // We will use these auxiliary to help in synthesizing the member initialize constructor.
    Slang::HashSet<VarDeclBase*> m_membersVisibleInCtor;
};

FIDDLE()
class ClassDecl : public AggTypeDecl
{
    FIDDLE(...)
};

FIDDLE()
class GLSLInterfaceBlockDecl : public AggTypeDecl
{
    FIDDLE(...)
};

// TODO: Is it appropriate to treat an `enum` as an aggregate type?
// Most code that looks for, e.g., conformances assumes user-defined
// types are all `AggTypeDecl`, so this is the right choice for now
// if we want `enum` types to be able to implement interfaces, etc.
//
FIDDLE()
class EnumDecl : public AggTypeDecl
{
    FIDDLE(...)
    FIDDLE() Type* tagType = nullptr;
};

// A single case in an enum.
//
// E.g., in a declaration like:
//
//      enum Color { Red = 0, Green, Blue };
//
// The `Red = 0` is the declaration of the `Red`
// case, with `0` as an explicit expression for its
// _tag value_.
//
FIDDLE()
class EnumCaseDecl : public Decl
{
    FIDDLE(...)
    // type of the parent `enum`
    FIDDLE() TypeExp type;

    Type* getType() { return type.type; }

    // Tag value
    FIDDLE() Expr* tagExpr = nullptr;

    FIDDLE() IntVal* tagVal = nullptr;
};

// A member of InterfaceDecl representing the abstract ThisType.
FIDDLE()
class ThisTypeDecl : public AggTypeDecl
{
    FIDDLE(...)
};

// An interface which other types can conform to
FIDDLE()
class InterfaceDecl : public AggTypeDecl
{
    FIDDLE(...)
    ThisTypeDecl* getThisTypeDecl();
};

FIDDLE(abstract)
class TypeConstraintDecl : public Decl
{
    FIDDLE(...)
    const TypeExp& getSup() const;
    // Overrides should be public so base classes can access
    // Implement _getSupOverride on derived classes to change behavior of getSup, as if getSup is
    // virtual
    const TypeExp& _getSupOverride() const;
};

FIDDLE()
class ThisTypeConstraintDecl : public TypeConstraintDecl
{
    FIDDLE(...)
    FIDDLE() TypeExp base;
    const TypeExp& _getSupOverride() const { return base; }
    InterfaceDecl* getInterfaceDecl();
};

// A kind of pseudo-member that represents an explicit
// or implicit inheritance relationship.
//
FIDDLE()
class InheritanceDecl : public TypeConstraintDecl
{
    FIDDLE(...)
    // The type expression as written
    FIDDLE() TypeExp base;

    // After checking, this dictionary will map members
    // required by the base type to their concrete
    // implementations in the type that contains
    // this inheritance declaration.
    FIDDLE() RefPtr<WitnessTable> witnessTable;

    // Overrides should be public so base classes can access
    const TypeExp& _getSupOverride() const { return base; }
};

// TODO: may eventually need sub-classes for explicit/direct vs. implicit/indirect inheritance


// A declaration that represents a simple (non-aggregate) type
//
// TODO: probably all types will be aggregate decls eventually,
// so that we can easily store conformances/constraints on type variables
FIDDLE(abstract)
class SimpleTypeDecl : public Decl
{
    FIDDLE(...)
};

// A `typedef` declaration
FIDDLE()
class TypeDefDecl : public SimpleTypeDecl
{
    FIDDLE(...)
    FIDDLE() TypeExp type;
};

FIDDLE()
class TypeAliasDecl : public TypeDefDecl
{
    FIDDLE(...)
};

// An 'assoctype' declaration, it is a container of inheritance clauses
FIDDLE()
class AssocTypeDecl : public AggTypeDecl
{
    FIDDLE(...)
};

// A 'type_param' declaration, which defines a generic
// entry-point parameter. Is a container of GenericTypeConstraintDecl
FIDDLE()
class GlobalGenericParamDecl : public AggTypeDecl
{
    FIDDLE(...)
};

// A `__generic_value_param` declaration, which defines an existential
// value parameter (not a type parameter.
FIDDLE()
class GlobalGenericValueParamDecl : public VarDeclBase
{
    FIDDLE(...)
};

// A scope for local declarations (e.g., as part of a statement)
FIDDLE()
class ScopeDecl : public ContainerDecl
{
    FIDDLE(...)
};

// A function/initializer/subscript parameter (potentially mutable)
FIDDLE()
class ParamDecl : public VarDeclBase
{
    FIDDLE(...)
};

// A parameter of a function declared in "modern" types (immutable unless explicitly `out` or
// `inout`)
FIDDLE()
class ModernParamDecl : public ParamDecl
{
    FIDDLE(...)
};

// Base class for things that have parameter lists and can thus be applied to arguments ("called")
FIDDLE(abstract)
class CallableDecl : public ContainerDecl
{
    FIDDLE(...)
    FilteredMemberList<ParamDecl> getParameters() { return getMembersOfType<ParamDecl>(); }

    FIDDLE() TypeExp returnType;

    // If this callable throws an error code, `errorType` is the type of the error code.
    FIDDLE() TypeExp errorType;

    // Fields related to redeclaration, so that we
    // can support multiple specialized variations
    // of the "same" logical function.
    //
    // This should also help us to support redeclaration
    // of functions when handling HLSL/GLSL.

    // The "primary" declaration of the function, which will
    // be used whenever we need to unique things.
    FIDDLE() CallableDecl* primaryDecl = nullptr;

    // The next declaration of the "same" function (that is,
    // with the same `primaryDecl`).
    FIDDLE() CallableDecl* nextDecl = nullptr;
};

// Base class for callable things that may also have a body that is evaluated to produce their
// result
FIDDLE(abstract)
class FunctionDeclBase : public CallableDecl
{
    FIDDLE(...)
    FIDDLE() Stmt* body = nullptr;
};

// A constructor/initializer to create instances of a type
FIDDLE()
class ConstructorDecl : public FunctionDeclBase
{
    FIDDLE(...)
    enum class ConstructorFlavor : int
    {
        UserDefined = 0x00,
        // Indicates whether the declaration was synthesized by
        // Slang and not explicitly provided by the user
        SynthesizedDefault = 0x01,
        // Member initialize constructor is a synthesized ctor,
        // but it takes parameters.
        SynthesizedMemberInit = 0x02
    };

    FIDDLE() int m_flavor = (int)ConstructorFlavor::UserDefined;
    void addFlavor(ConstructorFlavor flavor) { m_flavor |= (int)flavor; }
    bool containsFlavor(ConstructorFlavor flavor) { return m_flavor & (int)flavor; }
};

FIDDLE()
class LambdaDecl : public StructDecl
{
    FIDDLE(...)

    FIDDLE() FunctionDeclBase* funcDecl;
};

// A subscript operation used to index instances of a type
FIDDLE()
class SubscriptDecl : public CallableDecl
{
    FIDDLE(...)
};

/// A property declaration that abstracts over storage with a getter/setter/etc.
FIDDLE()
class PropertyDecl : public ContainerDecl
{
    FIDDLE(...)
    FIDDLE() TypeExp type;
};

// An "accessor" for a subscript or property
FIDDLE(abstract)
class AccessorDecl : public FunctionDeclBase
{
    FIDDLE(...)
};

FIDDLE()
class GetterDecl : public AccessorDecl
{
    FIDDLE(...)
};

FIDDLE()
class SetterDecl : public AccessorDecl
{
    FIDDLE(...)
};

FIDDLE()
class RefAccessorDecl : public AccessorDecl
{
    FIDDLE(...)
};

FIDDLE()
class FuncDecl : public FunctionDeclBase
{
    FIDDLE(...)
};

FIDDLE(abstract)
class NamespaceDeclBase : public ContainerDecl
{
    FIDDLE(...)
};

// A `namespace` declaration inside some module, that provides
// a named scope for declarations inside it.
//
// Note: Multiple `namespace` declarations with the same name
// in a given module/file will be collapsed into a single
// `NamespaceDecl` during parsing, so this declaration does
// not directly represent what is present in the input syntax.
//
FIDDLE()
class NamespaceDecl : public NamespaceDeclBase
{
    FIDDLE(...)
};

// A "module" of code (essentially, a single translation unit)
// that provides a scope for some number of declarations.
FIDDLE()
class ModuleDecl : public NamespaceDeclBase
{
    FIDDLE(...)

    // The API-level module that this declaration belong to.
    //
    // This field allows lookup of the `Module` based on a
    // declaration nested under a `ModuleDecl` by following
    // its chain of parents.
    //
    Module* module = nullptr;

    /// Map a decl to the list of its associated decls.
    ///
    /// This mapping is filled in during semantic checking, as the decl declarations get checked or
    /// generated.
    ///
    FIDDLE() OrderedDictionary<Decl*, RefPtr<DeclAssociationList>> mapDeclToAssociatedDecls;

    /// Whether the module is defined in legacy language.
    /// The legacy Slang language does not have visibility modifiers and everything is treated as
    /// `public`. Newer version of the language introduces visibility and makes `internal` as the
    /// default. To prevent this from breaking existing code, we need to know whether a module is
    /// written in the legacy language. We detect this by checking whether the module has any
    /// visibility modifiers, or if the module uses new language constructs, e.g. `module`,
    /// `__include`,
    /// `__implementing` etc.
    FIDDLE() bool isInLegacyLanguage = true;

    FIDDLE() DeclVisibility defaultVisibility = DeclVisibility::Internal;

    SLANG_UNREFLECTED

    /// Map a type to the list of extensions of that type (if any) declared in this module
    ///
    /// This mapping is filled in during semantic checking, as `ExtensionDecl`s get checked.
    ///
    FIDDLE() Dictionary<AggTypeDecl*, RefPtr<CandidateExtensionList>> mapTypeToCandidateExtensions;
};

// Represents a transparent scope of declarations that are defined in a single source file.
FIDDLE()
class FileDecl : public ContainerDecl
{
    FIDDLE(...)
};

/// A declaration that brings members of another declaration or namespace into scope
FIDDLE()
class UsingDecl : public Decl
{
    FIDDLE(...)

    /// An expression that identifies the entity (e.g., a namespace) to be brought into `scope`
    Expr* arg = nullptr;

    SLANG_UNREFLECTED
    /// The scope that the entity named by `arg` will be brought into
    Scope* scope = nullptr;
};

FIDDLE()
class FileReferenceDeclBase : public Decl
{
    FIDDLE(...)

    // The name of the module we are trying to import
    NameLoc moduleNameAndLoc;

    SourceLoc startLoc;
    SourceLoc endLoc;

    SLANG_UNREFLECTED
    // The scope that we want to import into
    Scope* scope = nullptr;
};

FIDDLE()
class ImportDecl : public FileReferenceDeclBase
{
    FIDDLE(...)

    // The module that actually got imported
    FIDDLE() ModuleDecl* importedModuleDecl = nullptr;
};

FIDDLE(abstract)
class IncludeDeclBase : public FileReferenceDeclBase
{
    FIDDLE(...)
    FileDecl* fileDecl = nullptr;
};

FIDDLE()
class IncludeDecl : public IncludeDeclBase
{
    FIDDLE(...)
};

FIDDLE()
class ImplementingDecl : public IncludeDeclBase
{
    FIDDLE(...)
};

FIDDLE()
class ModuleDeclarationDecl : public Decl
{
    FIDDLE(...)
};

FIDDLE()
class RequireCapabilityDecl : public Decl
{
    FIDDLE(...)
};

// A generic declaration, parameterized on types/values
FIDDLE()
class GenericDecl : public ContainerDecl
{
    FIDDLE(...)
    // The decl that is genericized...
    FIDDLE() Decl* inner = nullptr;
};

FIDDLE(abstract)
class GenericTypeParamDeclBase : public SimpleTypeDecl
{
    FIDDLE(...)
    // The index of the generic parameter.
    int parameterIndex = -1;
};

FIDDLE()
class GenericTypeParamDecl : public GenericTypeParamDeclBase
{
    FIDDLE(...)
    // The bound for the type parameter represents a trait that any
    // type used as this parameter must conform to
    //            TypeExp bound;

    // The "initializer" for the parameter represents a default value
    FIDDLE() TypeExp initType;
};

FIDDLE()
class GenericTypePackParamDecl : public GenericTypeParamDeclBase
{
    FIDDLE(...)
};

// A constraint placed as part of a generic declaration
FIDDLE()
class GenericTypeConstraintDecl : public TypeConstraintDecl
{
    FIDDLE(...)
    // A type constraint like `T : U` is constraining `T` to be "below" `U`
    // on a lattice of types. This may not be a subtyping relationship
    // per se, but it makes sense to use that terminology here, so we
    // think of these fields as the sub-type and super-type, respectively.
    FIDDLE() TypeExp sub;
    FIDDLE() TypeExp sup;

    // If this decl is defined in a where clause, store the source location of the where token.
    SourceLoc whereTokenLoc = SourceLoc();

    FIDDLE() bool isEqualityConstraint = false;

    // Overrides should be public so base classes can access
    const TypeExp& _getSupOverride() const { return sup; }
};

FIDDLE()
class TypeCoercionConstraintDecl : public Decl
{
    FIDDLE(...)
    SourceLoc whereTokenLoc = SourceLoc();
    FIDDLE() TypeExp fromType;
    FIDDLE() TypeExp toType;
};

FIDDLE()
class GenericValueParamDecl : public VarDeclBase
{
    FIDDLE(...)
    // The index of the generic parameter.
    int parameterIndex = 0;
};

// An empty declaration (which might still have modifiers attached).
//
// An empty declaration is uncommon in HLSL, but
// in GLSL it is often used at the global scope
// to declare metadata that logically belongs
// to the entry point, e.g.:
//
//     layout(local_size_x = 16) in;
//
FIDDLE()
class EmptyDecl : public Decl
{
    FIDDLE(...)
};

// A declaration used by the implementation to put syntax keywords
// into the current scope.
//
FIDDLE()
class SyntaxDecl : public Decl
{
    FIDDLE(...)
    // What type of syntax node will be produced when parsing with this keyword?
    FIDDLE() SyntaxClass<NodeBase> syntaxClass;

    SLANG_UNREFLECTED

    // Callback to invoke in order to parse syntax with this keyword.
    SyntaxParseCallback parseCallback = nullptr;
    void* parseUserData = nullptr;
};

// A declaration of an attribute to be used with `[name(...)]` syntax.
//
FIDDLE()
class AttributeDecl : public ContainerDecl
{
    FIDDLE(...)
    // What type of syntax node will be produced to represent this attribute.
    FIDDLE() SyntaxClass<NodeBase> syntaxClass;
};

// A synthesized decl used as a placeholder for a differentiable function requirement. This decl
// will be a child of interface decl. This allows us to form an interface requirement key for the
// derivative of an interface function. The synthesized `DerivativeRequirementDecl` will be a child
// of the original function requirement decl after an interface type is checked.
FIDDLE()
class DerivativeRequirementDecl : public FunctionDeclBase
{
    FIDDLE(...)
    // The original requirement decl.
    FIDDLE() Decl* originalRequirementDecl = nullptr;

    // Type to use for 'ThisType'
    FIDDLE() Type* diffThisType;
};

// A reference to a synthesized decl representing a differentiable function requirement, this decl
// will be a child in the orignal function.
FIDDLE()
class DerivativeRequirementReferenceDecl : public FunctionDeclBase
{
    FIDDLE(...)
    FIDDLE() DerivativeRequirementDecl* referencedDecl;
};

FIDDLE()
class ForwardDerivativeRequirementDecl : public DerivativeRequirementDecl
{
    FIDDLE(...)
};

FIDDLE()
class BackwardDerivativeRequirementDecl : public DerivativeRequirementDecl
{
    FIDDLE(...)
};

bool isInterfaceRequirement(Decl* decl);
InterfaceDecl* findParentInterfaceDecl(Decl* decl);

bool isLocalVar(const Decl* decl);


// Add a sibling lookup scope for `dest` to refer to `source`.
void addSiblingScopeForContainerDecl(
    ASTBuilder* builder,
    ContainerDecl* dest,
    ContainerDecl* source);
void addSiblingScopeForContainerDecl(ASTBuilder* builder, Scope* destScope, ContainerDecl* source);

} // namespace Slang
