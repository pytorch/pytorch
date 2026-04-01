// slang-ast-expr.h
#pragma once

#include "slang-ast-base.h"
#include "slang-ast-expr.h.fiddle"

FIDDLE()
namespace Slang
{

using SpvWord = uint32_t;

// Syntax class definitions for expressions.
//
// A placeholder for where an Expr is expected but is missing from source.
FIDDLE()
class IncompleteExpr : public Expr
{
    FIDDLE(...)
};

// Base class for expressions that will reference declarations
FIDDLE(abstract)
class DeclRefExpr : public Expr
{
    FIDDLE(...)
    // The declaration of the symbol being referenced

    FIDDLE() DeclRef<Decl> declRef;
    // The name of the symbol being referenced
    Name* name = nullptr;

    // The original expr before DeclRef resolution.
    Expr* originalExpr = nullptr;

    SLANG_UNREFLECTED
    // The scope in which to perform lookup
    Scope* scope = nullptr;
};

FIDDLE()
class VarExpr : public DeclRefExpr
{
    FIDDLE(...)
};

FIDDLE()
class DefaultConstructExpr : public Expr
{
    FIDDLE(...)
};

// An expression that references an overloaded set of declarations
// having the same name.
FIDDLE()
class OverloadedExpr : public Expr
{
    FIDDLE(...)
    // The name that was looked up and found to be overloaded
    Name* name = nullptr;

    // Optional: the base expression is this overloaded result
    // arose from a member-reference expression.
    Expr* base = nullptr;

    Expr* originalExpr = nullptr;

    // The lookup result that was ambiguous
    LookupResult lookupResult2;
};

// An expression that references an overloaded set of declarations
// having the same name.
FIDDLE()
class OverloadedExpr2 : public Expr
{
    FIDDLE(...)
    // Optional: the base expression is this overloaded result
    // arose from a member-reference expression.
    Expr* base = nullptr;

    // The lookup result that was ambiguous
    List<Expr*> candidiateExprs;
};

FIDDLE(abstract)
class LiteralExpr : public Expr
{
    FIDDLE(...)
    // The token that was used to express the literal. This can be
    // used to get the raw text of the literal, including any suffix.
    Token token;
    FIDDLE() BaseType suffixType = BaseType::Void;
};

FIDDLE()
class IntegerLiteralExpr : public LiteralExpr
{
    FIDDLE(...)
    FIDDLE() IntegerLiteralValue value;
};

FIDDLE()
class FloatingPointLiteralExpr : public LiteralExpr
{
    FIDDLE(...)
    FIDDLE() FloatingPointLiteralValue value;
};

FIDDLE()
class BoolLiteralExpr : public LiteralExpr
{
    FIDDLE(...)
    FIDDLE() bool value;
};

FIDDLE()
class NullPtrLiteralExpr : public LiteralExpr
{
    FIDDLE(...)
};

FIDDLE()
class NoneLiteralExpr : public LiteralExpr
{
    FIDDLE(...)
};

FIDDLE()
class StringLiteralExpr : public LiteralExpr
{
    FIDDLE(...)
    // TODO: consider storing the "segments" of the string
    // literal, in the case where multiple literals were
    // lined up at the lexer level, e.g.:
    //
    //      "first" "second" "third"
    //
    FIDDLE() String value;
};

// An initializer list, e.g. `{ 1, 2, 3 }`
FIDDLE()
class InitializerListExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() List<Expr*> args;

    bool useCStyleInitialization = true;
};

FIDDLE()
class GetArrayLengthExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() Expr* arrayExpr = nullptr;
};

FIDDLE()
class ExpandExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() Expr* baseExpr = nullptr;
};

FIDDLE()
class EachExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() Expr* baseExpr = nullptr;
};

// A base class for expressions with arguments
FIDDLE(abstract)
class ExprWithArgsBase : public Expr
{
    FIDDLE(...)
    FIDDLE() List<Expr*> arguments;
};

// An aggregate type constructor
FIDDLE()
class AggTypeCtorExpr : public ExprWithArgsBase
{
    FIDDLE(...)
    FIDDLE() TypeExp base;
};


// A base expression being applied to arguments: covers
// both ordinary `()` function calls and `<>` generic application
FIDDLE(abstract)
class AppExprBase : public ExprWithArgsBase
{
    FIDDLE(...)
    FIDDLE() Expr* functionExpr = nullptr;

    // The original function expr before overload resolution.
    Expr* originalFunctionExpr = nullptr;

    // The source location of `(`, `)`, and `,` that marks the start/end of the application op and
    // each argument expr. This info is used by language server.
    List<SourceLoc> argumentDelimeterLocs;
};

FIDDLE()
class InvokeExpr : public AppExprBase
{
    FIDDLE(...)
};

FIDDLE()
class ExplicitCtorInvokeExpr : public InvokeExpr
{
    FIDDLE(...)
};

enum class TryClauseType
{
    None,
    Standard, // Normal `try` clause
    Optional, // (Not implemented) `try?` clause that returns an optional value.
    Assert, // (Not implemented) `try!` clause that should always succeed and triggers runtime error
            // if failed.
};

char const* getTryClauseTypeName(TryClauseType value);

FIDDLE()
class TryExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() Expr* base;

    FIDDLE() TryClauseType tryClauseType = TryClauseType::Standard;

    // The scope of this expr.
    Scope* scope = nullptr;
};

FIDDLE()
class NewExpr : public InvokeExpr
{
    FIDDLE(...)
};

FIDDLE()
class OperatorExpr : public InvokeExpr
{
    FIDDLE(...)
};

FIDDLE()
class InfixExpr : public OperatorExpr
{
    FIDDLE(...)
};

FIDDLE()
class PrefixExpr : public OperatorExpr
{
    FIDDLE(...)
};

FIDDLE()
class PostfixExpr : public OperatorExpr
{
    FIDDLE(...)
};

FIDDLE()
class IndexExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() Expr* baseExpression;
    FIDDLE() List<Expr*> indexExprs;

    // The source location of `(`, `)`, and `,` that marks the start/end of the application op and
    // each argument expr. This info is used by language server.
    List<SourceLoc> argumentDelimeterLocs;
};

FIDDLE()
class MemberExpr : public DeclRefExpr
{
    FIDDLE(...)
    FIDDLE() Expr* baseExpression = nullptr;
    SourceLoc memberOperatorLoc;
};

// Member expression that is dereferenced, e.g. `a->b`.
FIDDLE()
class DerefMemberExpr : public MemberExpr
{
    FIDDLE(...)
};

// Member looked up on a type, rather than a value
FIDDLE()
class StaticMemberExpr : public DeclRefExpr
{
    FIDDLE(...)
    Expr* baseExpression = nullptr;
    SourceLoc memberOperatorLoc;
};

struct MatrixCoord
{
    bool operator==(const MatrixCoord& rhs) const { return row == rhs.row && col == rhs.col; };
    bool operator!=(const MatrixCoord& rhs) const { return !(*this == rhs); };
    // Rows and columns are zero indexed
    int row;
    int col;
};

FIDDLE()
class MatrixSwizzleExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() Expr* base = nullptr;
    FIDDLE() int elementCount;
    FIDDLE() MatrixCoord elementCoords[4];
    SourceLoc memberOpLoc;
};

FIDDLE()
class SwizzleExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() Expr* base = nullptr;
    FIDDLE() ShortList<uint32_t, 4> elementIndices;
    SourceLoc memberOpLoc;
};

// An operation to convert an l-value to a reference type.
FIDDLE()
class MakeRefExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() Expr* base = nullptr;
};

// A dereference of a pointer or pointer-like type
FIDDLE()
class DerefExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() Expr* base = nullptr;
};

// Any operation that performs type-casting
FIDDLE()
class TypeCastExpr : public InvokeExpr
{
    FIDDLE(...)
    //    TypeExp TargetType;
    //    Expr* Expression = nullptr;
};

// An explicit type-cast that appear in the user's code with `(type) expr` syntax
FIDDLE()
class ExplicitCastExpr : public TypeCastExpr
{
    FIDDLE(...)
};

// An implicit type-cast inserted during semantic checking
FIDDLE()
class ImplicitCastExpr : public TypeCastExpr
{
    FIDDLE(...)
};

// A builtin cast expr generated during semantic checking, where there is
// no associated conversion function decl.
FIDDLE()
class BuiltinCastExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() Expr* base = nullptr;
};

FIDDLE()
class LValueImplicitCastExpr : public TypeCastExpr
{
    FIDDLE(...)
    explicit LValueImplicitCastExpr(const TypeCastExpr& rhs)
        : Super(rhs)
    {
    }
};

// To work around situations like int += uint
// where we want to allow an LValue to work with an implicit cast.
// The argument being cast *must* be an LValue.
FIDDLE()
class OutImplicitCastExpr : public LValueImplicitCastExpr
{
    FIDDLE(...)
    /// Allow explict construction from any TypeCastExpr
    explicit OutImplicitCastExpr(const TypeCastExpr& rhs)
        : Super(rhs)
    {
    }
};

FIDDLE()
class InOutImplicitCastExpr : public LValueImplicitCastExpr
{
    FIDDLE(...)
    /// Allow explict construction from any TypeCastExpr
    explicit InOutImplicitCastExpr(const TypeCastExpr& rhs)
        : Super(rhs)
    {
    }
};

/// A cast of a value to a super-type of its type.
///
/// The type being cast to is stored as this expression's `type`.
///
FIDDLE()
class CastToSuperTypeExpr : public Expr
{
    FIDDLE(...)
    /// The value being cast to a super type
    ///
    /// The type being cast from is `valueArg->type`.
    ///
    FIDDLE() Expr* valueArg = nullptr;

    /// A witness showing that `valueArg`'s type is a sub-type of this expression's `type`
    FIDDLE() Val* witnessArg = nullptr;
};

/// A `value is Type` expression that evaluates to `true` if type of `value` is a sub-type of
/// `Type`.
FIDDLE()
class IsTypeExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() Expr* value = nullptr;
    FIDDLE() TypeExp typeExpr;

    // A witness showing that `typeExpr.type` is a subtype of `typeof(value)`.
    FIDDLE() Val* witnessArg = nullptr;

    // non-null if evaluates to a constant.
    FIDDLE() BoolLiteralExpr* constantVal = nullptr;
};

/// A `value as Type` expression that casts `value` to `Type` within type hierarchy.
/// The result is undefined if `value` is not `Type`.
FIDDLE()
class AsTypeExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() Expr* value = nullptr;
    FIDDLE() Expr* typeExpr = nullptr;

    // A witness showing that `typeExpr` is a subtype of `typeof(value)`.
    FIDDLE() Val* witnessArg = nullptr;
};

FIDDLE(abstract)
class SizeOfLikeExpr : public Expr
{
    FIDDLE(...)
    // Set during the parse, could be an expression, a variable or a type
    FIDDLE() Expr* value = nullptr;

    // The type the size/alignment needs to operate on. Set during traversal of SemanticsExprVisitor
    FIDDLE() Type* sizedType = nullptr;
};

FIDDLE()
class SizeOfExpr : public SizeOfLikeExpr
{
    FIDDLE(...)
};

FIDDLE()
class AlignOfExpr : public SizeOfLikeExpr
{
    FIDDLE(...)
};

FIDDLE()
class CountOfExpr : public SizeOfLikeExpr
{
    FIDDLE(...)
};

FIDDLE()
class MakeOptionalExpr : public Expr
{
    FIDDLE(...)
    // If `value` is null, this constructs an `Optional<T>` that doesn't have a value.
    FIDDLE() Expr* value = nullptr;
    FIDDLE() Expr* typeExpr = nullptr;
};

/// A cast of a value to the same type, with different modifiers.
///
/// The type being cast to is stored as this expression's `type`.
///
FIDDLE()
class ModifierCastExpr : public Expr
{
    FIDDLE(...)
    /// The value being cast.
    ///
    /// The type being cast from is `valueArg->type`.
    ///
    FIDDLE() Expr* valueArg = nullptr;
};

FIDDLE()
class SelectExpr : public OperatorExpr
{
    FIDDLE(...)
};

FIDDLE()
class LogicOperatorShortCircuitExpr : public OperatorExpr
{
    FIDDLE(...)
public:
    enum Flavor
    {
        And, // &&
        Or,  // ||
    };
    FIDDLE() Flavor flavor;
};


FIDDLE()
class GenericAppExpr : public AppExprBase
{
    FIDDLE(...)
};

// An expression representing re-use of the syntax for a type in more
// than once conceptually-distinct declaration
FIDDLE()
class SharedTypeExpr : public Expr
{
    FIDDLE(...)
    // The underlying type expression that we want to share
    TypeExp base;
};

FIDDLE()
class AssignExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() Expr* left = nullptr;
    FIDDLE() Expr* right = nullptr;
};

// Just an expression inside parentheses `(exp)`
//
// We keep this around explicitly to be sure we don't lose any structure
// when we do rewriter stuff.
FIDDLE()
class ParenExpr : public Expr
{
    FIDDLE(...)
    Expr* base = nullptr;
};

// An object-oriented `this` expression, used to
// refer to the current instance of an enclosing type.
FIDDLE()
class ThisExpr : public Expr
{
    FIDDLE(...)
    SLANG_UNREFLECTED
    Scope* scope = nullptr;
};

// Represent a reference to the virtual __return_val object holding the return value of
// functions whose result type is non-copyable.
FIDDLE()
class ReturnValExpr : public Expr
{
    FIDDLE(...)
    SLANG_UNREFLECTED
    Scope* scope = nullptr;
};

// An expression that binds a temporary variable in a local expression context
FIDDLE()
class LetExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() VarDecl* decl = nullptr;
    FIDDLE() Expr* body = nullptr;
};

FIDDLE()
class ExtractExistentialValueExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() DeclRef<VarDeclBase> declRef;
    Expr* originalExpr;
};

FIDDLE()
class OpenRefExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() Expr* innerExpr = nullptr;
};

FIDDLE()
class DetachExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() Expr* inner = nullptr;
};

/// Base class for higher-order function application
/// Eg: foo(fn) where fn is a function expression.
///
FIDDLE(abstract)
class HigherOrderInvokeExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() Expr* baseFunction;
    FIDDLE() List<Name*> newParameterNames;
};

FIDDLE()
class PrimalSubstituteExpr : public HigherOrderInvokeExpr
{
    FIDDLE(...)
};

FIDDLE(abstract)
class DifferentiateExpr : public HigherOrderInvokeExpr
{
    FIDDLE(...)
};

/// An expression of the form `__fwd_diff(fn)` to access the
/// forward-mode derivative version of the function `fn`
///
FIDDLE()
class ForwardDifferentiateExpr : public DifferentiateExpr
{
    FIDDLE(...)
};

/// An expression of the form `__bwd_diff(fn)` to access the
/// forward-mode derivative version of the function `fn`
///
FIDDLE()
class BackwardDifferentiateExpr : public DifferentiateExpr
{
    FIDDLE(...)
};

/// An expression of the form `__dispatch_kernel(fn, threadGroupSize, dispatchSize)` to
/// dispatch a compute kernel from host.
///
FIDDLE()
class DispatchKernelExpr : public HigherOrderInvokeExpr
{
    FIDDLE(...)
    FIDDLE() Expr* threadGroupSize;
    FIDDLE() Expr* dispatchSize;
};

FIDDLE()
class LambdaExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() ScopeDecl* paramScopeDecl;
    FIDDLE() Stmt* bodyStmt;
};

/// An express to mark its inner expression as an intended non-differential call.
FIDDLE()
class TreatAsDifferentiableExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() Expr* innerExpr;
    Scope* scope;

    enum Flavor
    {
        /// Represents a no_diff wrapper over
        /// a non-differentiable method.
        /// i.e. no_diff(fn(...))
        ///
        NoDiff,

        /// Represents a call to a method that
        /// is either marked differentiable, or has
        /// a user-defined derivative in scope.
        ///
        Differentiable
    };

    FIDDLE() Flavor flavor;
};

/// A type expression of the form `This`
///
/// Refers to the type of `this` in the current context.
///
FIDDLE()
class ThisTypeExpr : public Expr
{
    FIDDLE(...)
    SLANG_UNREFLECTED
    Scope* scope = nullptr;
};

/// A type expression of the form `Left & Right`.
FIDDLE()
class AndTypeExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() TypeExp left;
    FIDDLE() TypeExp right;
};

/// A type exprssion that applies one or more modifiers to another type
FIDDLE()
class ModifiedTypeExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() Modifiers modifiers;
    FIDDLE() TypeExp base;
};

/// A type expression that rrepresents a pointer type, e.g. T*
FIDDLE()
class PointerTypeExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() TypeExp base;
};

/// A type expression that represents a function type, e.g. (bool, int) -> float
FIDDLE()
class FuncTypeExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() List<TypeExp> parameters;
    FIDDLE() TypeExp result;
};

FIDDLE()
class TupleTypeExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() List<TypeExp> members;
};

/// An expression that applies a generic to arguments for some,
/// but not all, of its explicit parameters.
///
FIDDLE()
class PartiallyAppliedGenericExpr : public Expr
{
    FIDDLE(...)
public:
    Expr* originalExpr = nullptr;

    /// The generic being applied
    DeclRef<GenericDecl> baseGenericDeclRef;

    /// A substitution that includes the generic arguments known so far
    List<Val*> knownGenericArgs;
};


/// An expression that holds a set of argument exprs that got matched to a pack parameter
/// during overload resolution.
///
FIDDLE()
class PackExpr : public Expr
{
    FIDDLE(...)
    FIDDLE() List<Expr*> args;
};

FIDDLE()
struct SPIRVAsmOperand
{
    FIDDLE(...)

public:
    enum Flavor
    {
        Literal,      // No prefix
        Id,           // Prefixed with %
        ResultMarker, // "result" (without quotes)
        NamedValue,   // Any other identifier
        SlangValue,
        SlangValueAddr,
        SlangImmediateValue,
        SlangType,
        SampledType, // __sampledType(T), this becomes a 4 vector of the component type of T
        ImageType,   // __imageType(texture), returns the equivalaent OpTypeImage of a given texture
                     // typed value.
        SampledImageType, // __sampledImageType(texture), returns the equivalent OpTypeSampledImage
                          // of a given texture typed value.
        ConvertTexel,     // __convertTexel(value), converts `value` to the native texel type of a
                          // texture.
        TruncateMarker,   // __truncate, an invented instruction which coerces to the result type by
                          // truncating the element count
        EntryPoint,       // __entryPoint, a placeholder for the id of a referencing entryPoint.
        BuiltinVar,
        GLSL450Set,
        NonSemanticDebugPrintfExtSet,
        RayPayloadFromLocation, // insert from scope of all payloads in the spir-v shader the
                                // payload identified by the integer value provided
        RayAttributeFromLocation,
        RayCallableFromLocation,
    };

    // The flavour and token describes how this was parsed
    Flavor flavor;
    // The single token this came from
    Token token;

    // If this was a SlangValue or SlangValueAddr or SlangType, then we also
    // store the expression, which should be a single VarExpr because we only
    // parse single idents at the moment
    Expr* expr = nullptr;

    // If this is part of a bitwise or expression, this will point to the
    // remaining operands values in such an expression must be of flavour
    // Literal or NamedValue
    List<SPIRVAsmOperand> bitwiseOrWith = List<SPIRVAsmOperand>();

    // If this is a named value then we calculate the value here during
    // checking. If this is an opcode, then the parser will populate this too
    // (or set it to 0xffffffff);
    SpvWord knownValue = 0xffffffff;
    // Although this might be a constant in the source we should actually pass
    // it as an id created with OpConstant
    bool wrapInId = false;

    // Once we've checked things, the SlangType and BuiltinVar flavour operands
    // will have this type populated.
    TypeExp type = TypeExp();
};

FIDDLE()
struct SPIRVAsmInst
{
    FIDDLE(...)
public:
    SPIRVAsmOperand opcode;
    List<SPIRVAsmOperand> operands;
};

FIDDLE()
class SPIRVAsmExpr : public Expr
{
    FIDDLE(...)
public:
    FIDDLE() List<SPIRVAsmInst> insts;
};

} // namespace Slang
