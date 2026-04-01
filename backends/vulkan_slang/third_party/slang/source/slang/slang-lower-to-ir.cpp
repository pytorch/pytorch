// lower.cpp
#include "slang-lower-to-ir.h"

#include "../core/slang-char-util.h"
#include "../core/slang-hash.h"
#include "../core/slang-performance-profiler.h"
#include "../core/slang-random-generator.h"
#include "slang-check.h"
#include "slang-ir-autodiff.h"
#include "slang-ir-bit-field-accessors.h"
#include "slang-ir-check-differentiability.h"
#include "slang-ir-check-recursion.h"
#include "slang-ir-clone.h"
#include "slang-ir-constexpr.h"
#include "slang-ir-dce.h"
#include "slang-ir-diff-call.h"
#include "slang-ir-entry-point-decorations.h"
#include "slang-ir-inline.h"
#include "slang-ir-insert-debug-value-store.h"
#include "slang-ir-insts.h"
#include "slang-ir-loop-inversion.h"
#include "slang-ir-lower-defer.h"
#include "slang-ir-lower-error-handling.h"
#include "slang-ir-lower-expand-type.h"
#include "slang-ir-missing-return.h"
#include "slang-ir-obfuscate-loc.h"
#include "slang-ir-operator-shift-overflow.h"
#include "slang-ir-peephole.h"
#include "slang-ir-sccp.h"
#include "slang-ir-simplify-cfg.h"
#include "slang-ir-ssa.h"
#include "slang-ir-string-hash.h"
#include "slang-ir-strip.h"
#include "slang-ir-use-uninitialized-values.h"
#include "slang-ir-util.h"
#include "slang-ir-validate.h"
#include "slang-ir.h"
#include "slang-mangle.h"
#include "slang-type-layout.h"
#include "slang-visitor.h"
#include "slang.h"

// Natural layout
#include "slang-ast-natural-layout.h"

namespace Slang
{

// This file implements lowering of the Slang AST to a simpler SSA
// intermediate representation.
//
// IR is generated in a context (`IRGenContext`), which tracks the current
// location in the IR where code should be emitted (e.g., what basic
// block to add instructions to). Lowering a statement will emit some
// number of instructions to the context, and possibly change the
// insertion point (because of control flow).
//
// When lowering an expression we have a more interesting challenge, for
// two main reasons:
//
// 1. There might be types that are representible in the AST, but which
//    we don't want to support natively in the IR. An example is a `struct`
//    type with both ordinary and resource-type members; we might want to
//    split values with such a type into distinct values during lowering.
//
// 2. We need to handle the difference between l-value and r-value expressions,
//    and in particular the fact that HLSL/Slang supports complicated sorts
//    of l-values (e.g., `someVector.zxy` is an l-value, even though it can't
//    be represented by a single pointer), and also allows l-values to appear
//    in multiple contexts (not just the left-hand side of assignment, but
//    also as an argument to match an `out` or `in out` parameter).
//
// Our solution to both of these problems is the same. Rather than having
// the lowering of an expression return a single IR-level value (`IRInst*`),
// we have it return a more complex type (`LoweredValInfo`) which can represent
// a wider range of conceptual "values" which might correspond to multiple IR-level
// values, and/or represent a pointer to an l-value rather than the r-value itself.

// We want to keep the representation of a `LoweringValInfo` relatively light
// - right now it is just a single pointer plus a "tag" to distinguish the cases.
//
// This means that cases that can't fit in a single pointer need a heap allocation
// to store their payload. For simplicity we represent all of these with a class
// hierarchy:
//
struct ExtendedValueInfo : RefObject
{
};

// This case is used to indicate a value that is a reference
// to an AST-level subscript declaration.
//
struct SubscriptInfo : ExtendedValueInfo
{
    DeclRef<SubscriptDecl> declRef;
};

// Some cases of `ExtendedValueInfo` need to
// recursively contain `LoweredValInfo`s, and
// so we forward declare them here and fill
// them in later.
//
struct BoundStorageInfo;
struct BoundMemberInfo;
struct SwizzledLValueInfo;
struct SwizzledMatrixLValueInfo;
struct CopiedValInfo;
struct ExtractedExistentialValInfo;
struct ImplicitCastLValueInfo;

// This type is our core representation of lowered values.
// In the simple case, it just wraps an `IRInst*`.
// More complex cases, representing l-values or aggregate
// values are also supported.
struct LoweredValInfo
{
    typedef LoweredValInfo ThisType;

    // Which of the cases of value are we looking at?
    enum class Flavor
    {
        // No value (akin to a null pointer)
        None,

        // A simple IR value
        Simple,

        // An l-value represented as an IR
        // pointer to the value
        Ptr,

        // A member declaration bound to a particular `this` value
        BoundMember,

        // A reference to an AST-level subscript operation
        Subscript,

        // An AST-level subscript operation bound to a particular
        // object and arguments.
        BoundStorage,

        // The result of applying swizzling to an l-value
        SwizzledLValue,

        // The result of applying swizzling to an l-value matrix
        SwizzledMatrixLValue,

        // The value extracted from an opened existential
        ExtractedExistential,

        // The L-Value that is an implicit cast.
        ImplicitCastedLValue,
    };

    union
    {
        IRInst* val;
        ExtendedValueInfo* ext;

        // We can compare any of the pointers above by comparing this pointer. If the union
        // ever becomes something other than a union of pointers, this would no longer be
        // applicable.
        void* aliasPtr;
    };
    Flavor flavor;

    // NOTE! This relies on the union, allowing the comparison of any of the pointer type in the
    // union. Assumes equality is the same as val pointer/or ext pointer being equal.
    bool operator==(const ThisType& rhs) const
    {
        return flavor == rhs.flavor && aliasPtr == rhs.aliasPtr;
    }
    bool operator!=(const ThisType& rhs) const { return !(*this == rhs); }

    LoweredValInfo()
    {
        flavor = Flavor::None;
        val = nullptr;
    }

    LoweredValInfo(IRType* t)
    {
        flavor = Flavor::Simple;
        val = t;
    }

    static LoweredValInfo simple(IRInst* v)
    {
        LoweredValInfo info;
        info.flavor = Flavor::Simple;
        info.val = v;
        return info;
    }

    static LoweredValInfo ptr(IRInst* v)
    {
        LoweredValInfo info;
        info.flavor = Flavor::Ptr;
        info.val = v;
        return info;
    }

    static LoweredValInfo boundMember(BoundMemberInfo* boundMemberInfo);

    BoundMemberInfo* getBoundMemberInfo()
    {
        SLANG_ASSERT(flavor == Flavor::BoundMember);
        return (BoundMemberInfo*)ext;
    }

    static LoweredValInfo subscript(SubscriptInfo* subscriptInfo);

    SubscriptInfo* getSubscriptInfo()
    {
        SLANG_ASSERT(flavor == Flavor::Subscript);
        return (SubscriptInfo*)ext;
    }

    static LoweredValInfo boundStorage(BoundStorageInfo* boundStorageInfo);

    BoundStorageInfo* getBoundStorageInfo()
    {
        SLANG_ASSERT(flavor == Flavor::BoundStorage);
        return (BoundStorageInfo*)ext;
    }

    static LoweredValInfo swizzledLValue(SwizzledLValueInfo* extInfo);

    static LoweredValInfo swizzledMatrixLValue(SwizzledMatrixLValueInfo* extInfo);

    static LoweredValInfo implicitCastedLValue(ImplicitCastLValueInfo* extInfo);

    SwizzledLValueInfo* getSwizzledLValueInfo()
    {
        SLANG_ASSERT(flavor == Flavor::SwizzledLValue);
        return (SwizzledLValueInfo*)ext;
    }

    SwizzledMatrixLValueInfo* getSwizzledMatrixLValueInfo()
    {
        SLANG_ASSERT(flavor == Flavor::SwizzledMatrixLValue);
        return (SwizzledMatrixLValueInfo*)ext;
    }

    static LoweredValInfo extractedExistential(ExtractedExistentialValInfo* extInfo);

    ExtractedExistentialValInfo* getExtractedExistentialValInfo()
    {
        SLANG_ASSERT(flavor == Flavor::ExtractedExistential);
        return (ExtractedExistentialValInfo*)ext;
    }

    ImplicitCastLValueInfo* getImplicitCastedLValue()
    {
        SLANG_ASSERT(flavor == Flavor::ImplicitCastedLValue);
        return (ImplicitCastLValueInfo*)ext;
    }
};

// This case is used to indicate a reference to an AST-level
// operation that accesses abstract storage.
//
// This could be an invocation of a `subscript` declaration,
// with argument representing an index or indices:
//
//     RWStructuredBuffer<Foo> gBuffer;
//     ... gBuffer[someIndex] ...
//
// the expression `gBuffer[someIndex]` will be lowered to
// a value that references `RWStructureBuffer<Foo>::operator[]`
// with arguments `(gBuffer, someIndex)`.
//
// This could also be an reference to a `property` declaration,
// with no arguments:
//
//      struct Sphere { property radius : int { get { ... } } }
//      Sphere sphere;
//      ... sphere.radius ...
//
// the expression `sphere.radius` will be lowered to a value
// that references `Sphere::radius` with arguments `(sphere)`.
//
// Such a value can be an l-value, and depending on the context
// where it is used, can lower into a call to either the getter
// or setter operations of the storage.
//
struct BoundStorageInfo : ExtendedValueInfo
{
    /// The declaration of the abstract storage (subscript or property)
    DeclRef<ContainerDecl> declRef;

    /// The IR-level type of the stored value
    IRType* type;

    /// The base value/object on which storage is being accessed
    LoweredValInfo base;

    /// Additional arguments required to reify a reference to the storage
    List<IRInst*> additionalArgs;
};


// Represents some declaration bound to a particular
// object. For example, if we had `obj.f` where `f`
// is a member function, we'd use a `BoundMemberInfo`
// to represnet this.
//
// Note: This case is largely avoided by special-casing
// in the handling of calls (like `obj.f(arg)`), but
// it is being left here as an example of what we might
// need/want to do in the long term.
struct BoundMemberInfo : ExtendedValueInfo
{
    // The base object
    LoweredValInfo base;

    // The (AST-level) declaration reference.
    DeclRef<Decl> declRef;

    // The type of this value
    IRType* type;
};

// Represents the result of a swizzle operation in
// an l-value context. A swizzle without duplicate
// elements is allowed as an l-value, even if the
// element are non-contiguous (`.xz`) or out of
// order (`.zxy`).
//
struct SwizzledLValueInfo : ExtendedValueInfo
{
    // The type of the expression.
    IRType* type;

    // The base expression (this should be an l-value)
    LoweredValInfo base;

    // The indices for the elements being swizzled
    ShortList<uint32_t, 4> elementIndices;
};

// Represents the result of a matrix swizzle operation in an l-value context.
// The same non-contiguous and no-duplicate rules as above apply.
struct SwizzledMatrixLValueInfo : ExtendedValueInfo
{
    // The type of the expression.
    IRType* type;

    // The base expression (this should be an l-value)
    LoweredValInfo base;

    // The number of elements in the swizzle
    UInt elementCount;

    // The coords for the elements being swizzled, zero indexed
    MatrixCoord elementCoords[4];
};

// Represents the results of extractng a value of
// some (statically unknown) concrete type from
// an existential, in an l-value context.
//
struct ExtractedExistentialValInfo : ExtendedValueInfo
{
    // The extracted value
    IRInst* extractedVal;

    // The original existential value
    LoweredValInfo existentialVal;

    // The type of `existentialVal`
    IRType* existentialType;

    // The IR witness table for the conformance of
    // the type of `extractedVal` to `existentialType`
    //
    IRInst* witnessTable;
};

struct ImplicitCastLValueInfo : ExtendedValueInfo
{
    // The type of the expression.
    IRType* type;

    // The base expression (this should be an l-value)
    LoweredValInfo base;

    // The type of the lvalue (inout, out, ref, etc.)
    ParameterDirection lValueType;
};

LoweredValInfo LoweredValInfo::boundMember(BoundMemberInfo* boundMemberInfo)
{
    LoweredValInfo info;
    info.flavor = Flavor::BoundMember;
    info.ext = boundMemberInfo;
    return info;
}

LoweredValInfo LoweredValInfo::subscript(SubscriptInfo* subscriptInfo)
{
    LoweredValInfo info;
    info.flavor = Flavor::Subscript;
    info.ext = subscriptInfo;
    return info;
}

LoweredValInfo LoweredValInfo::boundStorage(BoundStorageInfo* boundStorageInfo)
{
    LoweredValInfo info;
    info.flavor = Flavor::BoundStorage;
    info.ext = boundStorageInfo;
    return info;
}

LoweredValInfo LoweredValInfo::swizzledLValue(SwizzledLValueInfo* extInfo)
{
    LoweredValInfo info;
    info.flavor = Flavor::SwizzledLValue;
    info.ext = extInfo;
    return info;
}

LoweredValInfo LoweredValInfo::swizzledMatrixLValue(SwizzledMatrixLValueInfo* extInfo)
{
    LoweredValInfo info;
    info.flavor = Flavor::SwizzledMatrixLValue;
    info.ext = extInfo;
    return info;
}

LoweredValInfo LoweredValInfo::implicitCastedLValue(ImplicitCastLValueInfo* extInfo)
{
    LoweredValInfo info;
    info.flavor = Flavor::ImplicitCastedLValue;
    info.ext = extInfo;
    return info;
}

LoweredValInfo LoweredValInfo::extractedExistential(ExtractedExistentialValInfo* extInfo)
{
    LoweredValInfo info;
    info.flavor = Flavor::ExtractedExistential;
    info.ext = extInfo;
    return info;
}


// An "environment" for mapping AST declarations to IR values.
//
// This is required because in some cases we might lower the
// same AST declaration to the IR multiple times (e.g., when
// a generic transitively contains multiple functions, we
// will emit a distinct IR generic for each function, with
// its own copies of the generic parameters).
//
struct IRGenEnv
{
    // Map an AST-level declaration to the IR-level value that represents it.
    Dictionary<Decl*, LoweredValInfo> mapDeclToValue;

    // The next outer env around this one
    IRGenEnv* outer = nullptr;
};

struct SharedIRGenContext
{

    // The "global" environment for mapping declarations to their IR values.
    IRGenEnv globalEnv;

    // Map an AST-level declaration of an interface
    // requirement to the IR-level "key" that
    // is used to fetch that requirement from a
    // witness table.
    Dictionary<Decl*, IRStructKey*> interfaceRequirementKeys;

    // Arrays we keep around strictly for memory-management purposes:

    // Any extended values created during lowering need
    // to be cleaned up after the fact. We don't try
    // to reference-count these along the way because
    // they need to get stored into a `union` inside `LoweredValInfo`
    List<RefPtr<ExtendedValueInfo>> extValues;

    // Map from an AST-level statement that can be
    // used as the target of a `break` or `continue`
    // to the appropriate basic block to jump to.
    Dictionary<BreakableStmt::UniqueID, IRBlock*> breakLabels;
    Dictionary<BreakableStmt::UniqueID, IRBlock*> continueLabels;

    Dictionary<SourceFile*, IRInst*> mapSourceFileToDebugSourceInst;
    Dictionary<String, IRInst*> mapSourcePathToDebugSourceInst;

    void setGlobalValue(Decl* decl, LoweredValInfo value)
    {
        globalEnv.mapDeclToValue[decl] = value;
    }

    SharedIRGenContext(
        Session* session,
        DiagnosticSink* sink,
        bool obfuscateCode,
        ModuleDecl* mainModuleDecl,
        Linkage* linkage)
        : m_session(session)
        , m_sink(sink)
        , m_obfuscateCode(obfuscateCode)
        , m_mainModuleDecl(mainModuleDecl)
        , m_linkage(linkage)
    {
    }

    Session* m_session = nullptr;
    DiagnosticSink* m_sink = nullptr;
    bool m_obfuscateCode = false;
    ModuleDecl* m_mainModuleDecl = nullptr;
    Linkage* m_linkage = nullptr;

    // List of all string literals used in user code, regardless
    // of how they were used (i.e., whether or not they were hashed).
    //
    // This does *not* collect:
    // * String literals that were only used for attributes/modifiers in
    //   the user's code (e.g., `"compute"` in `[shader("compute")]`)
    // * Any IR string literals constructed for the purpose of decorations,
    //   reflection, or other meta-data that did not appear as a literal
    //   in the source code.
    //
    List<IRInst*> m_stringLiterals;
};

struct IRGenContext;

struct AstOrIRType
{
    Type* astType = nullptr;
    IRInst* irType = nullptr;
    IRInst* getIRType(IRGenContext* context);

    AstOrIRType& operator=(Type* t)
    {
        astType = t;
        irType = nullptr;
        return *this;
    }
    AstOrIRType& operator=(IRInst* t)
    {
        astType = nullptr;
        irType = t;
        return *this;
    }
    explicit operator bool() { return astType || irType; }
};

struct IRGenContext
{
    ASTBuilder* astBuilder;

    // Shared state for the IR generation process
    SharedIRGenContext* shared;

    // environment for mapping AST decls to IR values
    IRGenEnv* env;

    // IR builder to use when building code under this context
    IRBuilder* irBuilder;

    // The value to use for any `this` expressions
    // that appear in the current context.
    //
    // TODO: If we ever allow nesting of (non-static)
    // types, then we may need to support references
    // to an "outer `this`", and this representation
    // might be insufficient.
    LoweredValInfo thisVal;

    // The IRType value to lower into for `ThisType`.
    AstOrIRType thisType;

    // The IR witness value to use for `ThisType`
    IRInst* thisTypeWitness = nullptr;

    // The return destination parameter to write to at return sites.
    // (For use by functions that returns non-copyable types)
    LoweredValInfo returnDestination;

    // A reference to the Function decl to identify the parent function
    // that contains the Inst.
    FunctionDeclBase* funcDecl;

    bool includeDebugInfo = false;

    // The element index if we are inside an `expand` expression.
    IRInst* expandIndex = nullptr;

    // The current scope end for use with `defer`.
    IRBlock* scopeEndBlock = nullptr;

    // Callback function to call when after lowering a type.
    std::function<IRType*(IRGenContext* context, Type* type, IRType* irType)> lowerTypeCallback =
        nullptr;

    explicit IRGenContext(SharedIRGenContext* inShared, ASTBuilder* inAstBuilder)
        : shared(inShared), astBuilder(inAstBuilder), env(&inShared->globalEnv), irBuilder(nullptr)
    {
    }

    void registerTypeCallback(
        std::function<IRType*(IRGenContext* context, Type* type, IRType* irType)> callback)
    {
        lowerTypeCallback = callback;
    }

    void setGlobalValue(Decl* decl, LoweredValInfo value) { shared->setGlobalValue(decl, value); }

    void setValue(Decl* decl, LoweredValInfo value) { env->mapDeclToValue[decl] = value; }

    Session* getSession() { return shared->m_session; }

    DiagnosticSink* getSink() { return shared->m_sink; }

    ModuleDecl* getMainModuleDecl() { return shared->m_mainModuleDecl; }

    Linkage* getLinkage() { return shared->m_linkage; }

    LoweredValInfo* findLoweredDecl(Decl* decl)
    {
        IRGenEnv* envToFindIn = env;
        while (envToFindIn)
        {
            if (auto rs = envToFindIn->mapDeclToValue.tryGetValue(decl))
                return rs;
            envToFindIn = envToFindIn->outer;
        }
        return nullptr;
    }
};

ModuleDecl* findModuleDecl(Decl* decl)
{
    for (auto dd = decl; dd; dd = dd->parentDecl)
    {
        if (auto moduleDecl = as<ModuleDecl>(dd))
            return moduleDecl;
    }
    return nullptr;
}

bool isFromCoreModule(Decl* decl)
{
    for (auto dd = decl; dd; dd = dd->parentDecl)
    {
        if (dd->hasModifier<FromCoreModuleModifier>())
            return true;
    }
    return false;
}

bool isDeclInDifferentModule(IRGenContext* context, Decl* decl)
{
    return getModuleDecl(decl) != context->getMainModuleDecl();
}

bool isForceInlineEarly(Decl* decl)
{
    return decl->hasModifier<UnsafeForceInlineEarlyAttribute>();
}

bool isImportedDecl(IRGenContext* context, Decl* decl, bool& outIsExplicitExtern)
{
    // If the declaration has the extern attribute then it must be imported
    // from another module.
    // Note that `extern` declarations will have a mangled name that does not
    // include the module name so the linking step can resolve them correctly.
    //
    outIsExplicitExtern = false;
    if (decl->findModifier<ExternAttribute>() || decl->findModifier<ExternModifier>())
    {
        outIsExplicitExtern = true;
        return true;
    }

    for (auto parent = decl; parent; parent = parent->parentDecl)
    {
        if (as<ModuleDecl>(parent) && parent != context->getMainModuleDecl())
            return true;
        if (parent->findModifier<ExternAttribute>() || parent->findModifier<ExternModifier>())
        {
            outIsExplicitExtern = true;
            return true;
        }
    }
    return false;
}

/// Should the given `decl` nested in `parentDecl` be treated as a static rather than instance
/// declaration?
bool isEffectivelyStatic(Decl* decl, ContainerDecl* parentDecl);

bool isCoreModuleMemberFuncDecl(Decl* decl);

// Ensure that a version of the given declaration has been emitted to the IR
LoweredValInfo ensureDecl(IRGenContext* context, Decl* decl);

// Emit code as needed to construct a reference to the given declaration with
// any needed specializations in place.
LoweredValInfo emitDeclRef(IRGenContext* context, DeclRef<Decl> declRef, IRType* type);


bool isFunctionVarDecl(VarDeclBase* decl)
{
    // The immediate parent of a function-scope variable
    // declaration will be a `ScopeDecl`.
    //
    // TODO: right now the parent links for scopes are *not*
    // set correctly, so we can't just scan up and look
    // for a function in the parent chain...
    auto parent = decl->parentDecl;
    if (as<ScopeDecl>(parent))
    {
        return true;
    }
    return false;
}

bool isFunctionStaticVarDecl(VarDeclBase* decl)
{
    // Only a variable marked `static` can be static.
    if (!decl->findModifier<HLSLStaticModifier>())
        return false;
    return isFunctionVarDecl(decl);
}

IRInst* getSimpleVal(IRGenContext* context, LoweredValInfo lowered);

int32_t getIntrinsicOp(Decl* decl, IntrinsicOpModifier* intrinsicOpMod)
{
    int32_t op = intrinsicOpMod->op;
    if (op != 0)
        return op;

    // No specified modifier? Then we need to look it up
    // based on the name of the declaration...

    auto name = decl->getName();
    auto nameText = getUnownedStringSliceText(name);

    IROp irOp = findIROp(nameText);
    SLANG_ASSERT(irOp != kIROp_Invalid);
    SLANG_ASSERT(int32_t(irOp) >= 0);
    return int32_t(irOp);
}

struct TryClauseEnvironment
{
    TryClauseType clauseType = TryClauseType::None;
    IRBlock* catchBlock = nullptr;
};

// Given a `LoweredValInfo` for something callable, along with a
// bunch of arguments, emit an appropriate call to it.
LoweredValInfo emitCallToVal(
    IRGenContext* context,
    IRType* type,
    LoweredValInfo funcVal,
    UInt argCount,
    IRInst* const* args,
    const TryClauseEnvironment& tryEnv)
{
    auto builder = context->irBuilder;
    switch (funcVal.flavor)
    {
    case LoweredValInfo::Flavor::None:
        SLANG_UNEXPECTED("null function");
    default:
        switch (tryEnv.clauseType)
        {
        case TryClauseType::None:
            {
                auto callee = getSimpleVal(context, funcVal);
                if (auto dispatchKernel = as<IRDispatchKernel>(callee))
                {
                    // If callee is a dispatch kernel expr, don't emit call(dispatchKernel,
                    // ...), instead emit a dispatchKernel(high_order_args, actual_args).
                    auto result = LoweredValInfo::simple(builder->emitDispatchKernelInst(
                        type,
                        dispatchKernel->getBaseFn(),
                        dispatchKernel->getThreadGroupSize(),
                        dispatchKernel->getDispatchSize(),
                        argCount,
                        args));
                    SLANG_ASSERT(!dispatchKernel->hasUses());
                    dispatchKernel->removeAndDeallocate();
                    return result;
                }
                else
                {
                    return LoweredValInfo::simple(builder->emitCallInst(
                        type,
                        getSimpleVal(context, funcVal),
                        argCount,
                        args));
                }
            }

        case TryClauseType::Standard:
            {
                auto callee = getSimpleVal(context, funcVal);
                auto succBlock = builder->createBlock();
                auto failBlock = builder->createBlock();
                auto funcType = as<IRFuncType>(callee->getDataType());
                auto throwAttr = funcType->findAttr<IRFuncThrowTypeAttr>();
                assert(throwAttr);
                auto voidType = builder->getVoidType();
                builder->emitTryCallInst(voidType, succBlock, failBlock, callee, argCount, args);
                builder->insertBlock(failBlock);
                auto errParam = builder->emitParam(throwAttr->getErrorType());
                builder->emitThrow(errParam);
                builder->insertBlock(succBlock);
                auto value = builder->emitParam(type);
                return LoweredValInfo::simple(value);
            }
            break;
        default:
            SLANG_UNIMPLEMENTED_X("emitCallToVal(tryClauseType)");
        }
    }
}

LoweredValInfo lowerRValueExpr(IRGenContext* context, Expr* expr);

void lowerRValueExprWithDestination(IRGenContext* context, LoweredValInfo destination, Expr* expr);

IRType* lowerType(IRGenContext* context, Type* type);

static IRType* lowerType(IRGenContext* context, QualType const& type)
{
    return lowerType(context, type.type);
}

IRInst* AstOrIRType::getIRType(IRGenContext* context)
{
    if (irType)
        return irType;
    irType = lowerType(context, astType);
    return irType;
}

// Given a `DeclRef` for something callable, along with a bunch of
// arguments, emit an appropriate call to it.
LoweredValInfo emitCallToDeclRef(
    IRGenContext* context,
    IRType* type,
    DeclRef<Decl> funcDeclRef,
    IRType* funcType,
    UInt argCount,
    IRInst* const* args,
    const TryClauseEnvironment& tryEnv)
{
    SLANG_ASSERT(funcType);

    auto builder = context->irBuilder;

    auto funcDecl = funcDeclRef.getDecl();
    if (auto intrinsicOpModifier = funcDecl->findModifier<IntrinsicOpModifier>())
    {
        // The intrinsic op maps to a single IR instruction,
        // so we will emit an instruction with the chosen
        // opcode, and the arguments to the call as its operands.
        //
        if (intrinsicOpModifier->op == 0) // Identity, just pass operand 0 through.
        {
            SLANG_RELEASE_ASSERT(argCount == 1);
            return LoweredValInfo::simple(args[0]);
        }
        auto intrinsicOp = getIntrinsicOp(funcDecl, intrinsicOpModifier);
        switch (IROp(intrinsicOp))
        {
        case kIROp_GetOffsetPtr:
            SLANG_ASSERT(argCount == 2);
            return LoweredValInfo::simple(builder->emitGetOffsetPtr(args[0], args[1]));
        default:
            return LoweredValInfo::simple(
                builder->emitIntrinsicInst(type, IROp(intrinsicOp), argCount, args));
        }
    }

    // Fallback case is to emit an actual call.
    //
    LoweredValInfo funcVal = emitDeclRef(context, funcDeclRef, funcType);
    return emitCallToVal(context, type, funcVal, argCount, args, tryEnv);
}

LoweredValInfo emitCallToDeclRef(
    IRGenContext* context,
    IRType* type,
    DeclRef<Decl> funcDeclRef,
    IRType* funcType,
    List<IRInst*> const& args,
    const TryClauseEnvironment& tryEnv)
{
    return emitCallToDeclRef(
        context,
        type,
        funcDeclRef,
        funcType,
        args.getCount(),
        args.getBuffer(),
        tryEnv);
}

/// Emit a call to the given `accessorDeclRef`.
///
/// The `base` value represents the object on which the accessor is being invoked.
/// The `args` represent any additional arguments to the accessor. This could be
/// because we are invoking a subscript accessor (so the args include any index value(s)),
/// and/or because we are invoking a setter (so that the args include the new value
/// to be set).
///
static LoweredValInfo _emitCallToAccessor(
    IRGenContext* context,
    IRType* type,
    DeclRef<AccessorDecl> accessorDeclRef,
    LoweredValInfo base,
    UInt argCount,
    IRInst* const* args);

static LoweredValInfo _emitCallToAccessor(
    IRGenContext* context,
    IRType* type,
    DeclRef<AccessorDecl> accessorDeclRef,
    LoweredValInfo base,
    List<IRInst*> const& args)
{
    return _emitCallToAccessor(
        context,
        type,
        accessorDeclRef,
        base,
        args.getCount(),
        args.getBuffer());
}

/// Lower a reference to abstract storage (a property or subscript).
///
/// The given `storageDeclRef` is being accessed on some `base` value,
/// to yield a value of some expected `type`. The additional `args`
/// are only needed in the case of a subscript declaration (for
/// a property, `argCount` should be zero).
///
/// In the case where there is only a `get` accessor, this function
/// will go ahead and invoke it to produce a value here and now.
/// Otherwise, it will produce an abstract `LoweredValInfo` that
/// encapsulates the reference to the storage so that downstream
/// code can decide which accessor(s) to invoke.
///
[[maybe_unused]] static LoweredValInfo lowerStorageReference(
    IRGenContext* context,
    IRType* type,
    DeclRef<ContainerDecl> storageDeclRef,
    LoweredValInfo base,
    UInt argCount,
    IRInst* const* args)
{
    DeclRef<GetterDecl> getterDeclRef;
    bool justAGetter = true;
    for (auto accessorDeclRef : getMembersOfType<AccessorDecl>(
             context->astBuilder,
             storageDeclRef,
             MemberFilterStyle::Instance))
    {
        // We want to track whether this storage has any accessors other than
        // `get` (assuming that everything except `get` can be used for setting...).

        if (auto foundGetterDeclRef = accessorDeclRef.as<GetterDecl>())
        {
            // We found a getter.
            getterDeclRef = foundGetterDeclRef;
        }
        else
        {
            // There was something other than a getter, so we can't
            // invoke an accessor just now.
            justAGetter = false;
        }
    }

    if (!justAGetter || !getterDeclRef)
    {
        // We can't perform an actual call right now, because
        // this expression might appear in an r-value or l-value
        // position (or *both* if it is being passed as an argument
        // for an `in out` parameter!).
        //
        // Instead, we will construct a special-case value to
        // represent the latent access operation (abstractly
        // this is a reference to a storage location).

        // The abstract storage location will need to include
        // all the arguments being passed in the case of a subscript operation.

        RefPtr<BoundStorageInfo> boundStorage = new BoundStorageInfo();
        boundStorage->declRef = storageDeclRef;
        boundStorage->type = type;
        boundStorage->base = base;
        boundStorage->additionalArgs.addRange(args, argCount);

        context->shared->extValues.add(boundStorage);

        return LoweredValInfo::boundStorage(boundStorage);
    }

    return _emitCallToAccessor(context, type, getterDeclRef, base, argCount, args);
}

IRInst* getFieldKey(IRGenContext* context, DeclRef<Decl> field)
{
    return getSimpleVal(context, emitDeclRef(context, field, context->irBuilder->getKeyType()));
}

LoweredValInfo extractField(
    IRGenContext* context,
    IRType* fieldType,
    LoweredValInfo base,
    DeclRef<Decl> field)
{
    IRBuilder* builder = context->irBuilder;

    switch (base.flavor)
    {
    default:
        {
            IRInst* irBase = getSimpleVal(context, base);
            return LoweredValInfo::simple(
                builder->emitFieldExtract(fieldType, irBase, getFieldKey(context, field)));
        }
        break;

    case LoweredValInfo::Flavor::BoundMember:
    case LoweredValInfo::Flavor::BoundStorage:
        {
            // The base value is one that is trying to defer a get-vs-set
            // decision, so we will need to do the same.

            RefPtr<BoundMemberInfo> boundMemberInfo = new BoundMemberInfo();
            boundMemberInfo->type = fieldType;
            boundMemberInfo->base = base;
            boundMemberInfo->declRef = field;

            context->shared->extValues.add(boundMemberInfo);
            return LoweredValInfo::boundMember(boundMemberInfo);
        }
        break;

    case LoweredValInfo::Flavor::Ptr:
        {
            // We are "extracting" a field from an lvalue address,
            // which means we should just compute an lvalue
            // representing the field address.
            IRInst* irBasePtr = base.val;
            return LoweredValInfo::ptr(builder->emitFieldAddress(
                builder->getPtrType(fieldType),
                irBasePtr,
                getFieldKey(context, field)));
        }
        break;
    }
}

LoweredValInfo materialize(IRGenContext* context, LoweredValInfo lowered)
{
    auto builder = context->irBuilder;

top:
    switch (lowered.flavor)
    {
    case LoweredValInfo::Flavor::None:
    case LoweredValInfo::Flavor::Simple:
    case LoweredValInfo::Flavor::Ptr:
        return lowered;

    case LoweredValInfo::Flavor::BoundStorage:
        {
            auto boundStorageInfo = lowered.getBoundStorageInfo();

            // We are being asked to extract a value from a subscript call
            // (e.g., `base[index]`). We will first check if the subscript
            // declared a getter and use that if possible, and then fall
            // back to a `ref` accessor if one is defined.
            //
            // (Picking the `get` over the `ref` accessor simplifies things
            // in case the `get` operation has a natural translation for
            // a target, while the general `ref` case does not...)

            auto getters = getMembersOfType<GetterDecl>(
                context->astBuilder,
                boundStorageInfo->declRef,
                MemberFilterStyle::Instance);
            if (getters.getCount())
            {
                auto getter = *getters.begin();
                lowered = _emitCallToAccessor(
                    context,
                    boundStorageInfo->type,
                    getter,
                    boundStorageInfo->base,
                    boundStorageInfo->additionalArgs);
                goto top;
            }

            auto refAccessors = getMembersOfType<RefAccessorDecl>(
                context->astBuilder,
                boundStorageInfo->declRef,
                MemberFilterStyle::Instance);
            if (refAccessors.getCount())
            {
                auto refAccessor = *refAccessors.begin();

                // The `ref` accessor will return a pointer to the value, so
                // we need to reflect that in the type of our `call` instruction.
                IRType* ptrType = context->irBuilder->getPtrType(boundStorageInfo->type);

                LoweredValInfo refVal = _emitCallToAccessor(
                    context,
                    ptrType,
                    refAccessor,
                    boundStorageInfo->base,
                    boundStorageInfo->additionalArgs);

                // The result from the call needs to be implicitly dereferenced,
                // so that it can work as an l-value of the desired result type.
                lowered = LoweredValInfo::ptr(getSimpleVal(context, refVal));

                goto top;
            }

            // TODO: Ellie, Is this really unreachable? User code input can get here
            SLANG_UNEXPECTED("subscript had no getter");
            UNREACHABLE_RETURN(LoweredValInfo());
        }
        break;

    case LoweredValInfo::Flavor::BoundMember:
        {
            auto boundMemberInfo = lowered.getBoundMemberInfo();
            auto base = materialize(context, boundMemberInfo->base);

            auto declRef = boundMemberInfo->declRef;
            if (auto fieldDeclRef = declRef.as<VarDecl>())
            {
                lowered = extractField(context, boundMemberInfo->type, base, fieldDeclRef);
                goto top;
            }
            else
            {

                SLANG_UNEXPECTED("unexpected member flavor");
                UNREACHABLE_RETURN(LoweredValInfo());
            }
        }
        break;

    case LoweredValInfo::Flavor::SwizzledLValue:
        {
            auto swizzleInfo = lowered.getSwizzledLValueInfo();

            return LoweredValInfo::simple(builder->emitSwizzle(
                swizzleInfo->type,
                getSimpleVal(context, swizzleInfo->base),
                swizzleInfo->elementIndices.getCount(),
                swizzleInfo->elementIndices.getArrayView().getBuffer()));
        }

    case LoweredValInfo::Flavor::SwizzledMatrixLValue:
        {
            auto swizzleInfo = lowered.getSwizzledMatrixLValueInfo();
            auto base = getSimpleVal(context, swizzleInfo->base);
            if (const auto type = as<IRMatrixType>(base->getDataType()))
            {
                IRInst* components[4];
                for (UInt i = 0; i < swizzleInfo->elementCount; ++i)
                {
                    components[i] = builder->emitElementExtract(
                        builder->emitElementExtract(base, swizzleInfo->elementCoords[i].row),
                        swizzleInfo->elementCoords[i].col);
                }
                return swizzleInfo->elementCount == 1
                           ? LoweredValInfo::simple(components[0])
                           : LoweredValInfo::simple(builder->emitMakeVector(
                                 builder->getVectorType(
                                     type->getElementType(),
                                     swizzleInfo->elementCount),
                                 swizzleInfo->elementCount,
                                 components));
            }
            else
            {
                SLANG_UNEXPECTED("Expected a matrix type in matrix swizzle");
            }
        }

    case LoweredValInfo::Flavor::ExtractedExistential:
        {
            auto info = lowered.getExtractedExistentialValInfo();

            return LoweredValInfo::simple(info->extractedVal);
        }
    case LoweredValInfo::Flavor::ImplicitCastedLValue:
        {
            auto info = lowered.getImplicitCastedLValue();
            auto baseVal = materialize(context, info->base);
            auto result = builder->emitCast(info->type, getSimpleVal(context, baseVal));
            return LoweredValInfo::simple(result);
        }
    default:
        SLANG_UNEXPECTED("unhandled value flavor");
        UNREACHABLE_RETURN(LoweredValInfo());
    }
}

IRInst* getSimpleVal(IRGenContext* context, LoweredValInfo lowered)
{
    auto builder = context->irBuilder;

    // First, try to eliminate any "bound" operations along the chain,
    // so that we are dealing with an ordinary value, or an l-value pointer.
    lowered = materialize(context, lowered);

    switch (lowered.flavor)
    {
    case LoweredValInfo::Flavor::None:
        return nullptr;

    case LoweredValInfo::Flavor::Simple:
        return lowered.val;

    case LoweredValInfo::Flavor::Ptr:
        return builder->emitLoad(lowered.val);

    default:
        SLANG_UNEXPECTED("unhandled value flavor");
        UNREACHABLE_RETURN(nullptr);
    }
}

LoweredValInfo lowerVal(IRGenContext* context, Val* val);

IRInst* lowerSimpleVal(IRGenContext* context, Val* val)
{
    auto lowered = lowerVal(context, val);
    return getSimpleVal(context, lowered);
}

LoweredValInfo lowerLValueExpr(IRGenContext* context, Expr* expr);

void assign(IRGenContext* context, LoweredValInfo const& left, LoweredValInfo const& right);

void assignExpr(
    IRGenContext* context,
    const LoweredValInfo& inLeft,
    Expr* rightExpr,
    SourceLoc assignmentLoc);

IRInst* getAddress(
    IRGenContext* context,
    LoweredValInfo const& inVal,
    SourceLoc diagnosticLocation);

void lowerStmt(IRGenContext* context, Stmt* stmt);

LoweredValInfo lowerDecl(IRGenContext* context, DeclBase* decl);

IRType* getIntType(IRGenContext* context)
{
    return context->irBuilder->getBasicType(BaseType::Int);
}

static IRGeneric* getOuterGeneric(IRInst* gv)
{
    auto parentBlock = as<IRBlock>(gv->getParent());
    if (!parentBlock)
        return nullptr;

    auto parentGeneric = as<IRGeneric>(parentBlock->getParent());
    return parentGeneric;
}

static void addLinkageDecoration(
    IRGenContext* context,
    IRInst* inInst,
    Decl* decl,
    UnownedStringSlice const& mangledName)
{
    // If the instruction is nested inside one or more generics,
    // then the mangled name should really apply to the outer-most
    // generic, and not the declaration nested inside.

    auto builder = context->irBuilder;

    IRInst* inst = inInst;
    while (auto outerGeneric = getOuterGeneric(inst))
    {
        inst = outerGeneric;
    }

    bool explicitExtern = false;
    if (isImportedDecl(context, decl, explicitExtern))
    {
        builder->addImportDecoration(inst, mangledName);
        if (explicitExtern)
            builder->addUserExternDecoration(inst);
    }
    else
    {
        builder->addExportDecoration(inst, mangledName);
    }
    for (auto modifier : decl->modifiers)
    {
        if (as<PublicModifier>(modifier))
        {
            builder->addPublicDecoration(inst);
        }
        else if (as<HLSLExportModifier>(modifier))
        {
            builder->addHLSLExportDecoration(inst);
            builder->addKeepAliveDecoration(inst);
        }
        else if (as<ExternCppModifier>(modifier))
        {
            builder->addExternCppDecoration(inst, mangledName);
        }
        else if (auto dllImportModifier = as<DllImportAttribute>(modifier))
        {
            auto libraryName = dllImportModifier->modulePath;
            auto functionName = dllImportModifier->functionName.getLength()
                                    ? dllImportModifier->functionName.getUnownedSlice()
                                    : decl->getName()->text.getUnownedSlice();
            builder->addDllImportDecoration(inst, libraryName.getUnownedSlice(), functionName);
        }
        else if (as<DllExportAttribute>(modifier))
        {
            builder->addDllExportDecoration(inst, decl->getName()->text.getUnownedSlice());
            builder->addHLSLExportDecoration(inst);
            builder->addKeepAliveDecoration(inst);
        }
        else if (as<CudaDeviceExportAttribute>(modifier))
        {
            builder->addCudaDeviceExportDecoration(inst, decl->getName()->text.getUnownedSlice());
            builder->addHLSLExportDecoration(inst);
            builder->addExternCppDecoration(inst, decl->getName()->text.getUnownedSlice());
            builder->addKeepAliveDecoration(inst);
        }
        else if (as<CudaHostAttribute>(modifier))
        {
            builder->addCudaHostDecoration(inst);
            builder->addExternCppDecoration(inst, decl->getName()->text.getUnownedSlice());
            builder->addKeepAliveDecoration(inst);
        }
        else if (as<CudaKernelAttribute>(modifier))
        {
            builder->addCudaKernelDecoration(inst);
            builder->addExternCppDecoration(inst, decl->getName()->text.getUnownedSlice());

            // Temp decorations to get this function through the linker.
            builder->addKeepAliveDecoration(inst);
            builder->addHLSLExportDecoration(inst);
        }
        else if (as<TorchEntryPointAttribute>(modifier))
        {
            builder->addTorchEntryPointDecoration(inst, decl->getName()->text.getUnownedSlice());
            builder->addCudaHostDecoration(inst);
            builder->addExternCppDecoration(inst, decl->getName()->text.getUnownedSlice());

            // Temp decorations to get this function through the linker.
            builder->addKeepAliveDecoration(inst);
            builder->addHLSLExportDecoration(inst);
        }
        else if (as<AutoPyBindCudaAttribute>(modifier))
        {
            builder->addAutoPyBindCudaDecoration(inst, decl->getName()->text.getUnownedSlice());
            builder->addAutoPyBindExportInfoDecoration(inst);

            // Temp decorations to get this function through the linker.
            builder->addKeepAliveDecoration(inst);
            builder->addHLSLExportDecoration(inst);
        }
        else if (auto pyExportModifier = as<PyExportAttribute>(modifier))
        {
            builder->addPyExportDecoration(
                inst,
                pyExportModifier->name.getLength() ? pyExportModifier->name.getUnownedSlice()
                                                   : decl->getName()->text.getUnownedSlice());
            builder->addHLSLExportDecoration(inst);
        }
        else if (auto knownBuiltinModifier = as<KnownBuiltinAttribute>(modifier))
        {
            // We add this to the internal instruction, like other name-like
            // decorations, for instance "nameHint". This prevents it becoming
            // lost during specialization.
            builder->addKnownBuiltinDecoration(
                inInst,
                knownBuiltinModifier->name.getUnownedSlice());
        }
    }
    if (as<InterfaceDecl>(decl->parentDecl) &&
        decl->parentDecl->hasModifier<ComInterfaceAttribute>() &&
        !inst->findDecoration<IRExternCppDecoration>())
    {
        builder->addExternCppDecoration(inst, decl->getName()->text.getUnownedSlice());
    }
}

static void addLinkageDecoration(IRGenContext* context, IRInst* inst, Decl* decl)
{
    const String mangledName = getMangledName(context->astBuilder, decl);

    // Obfuscate the mangled names if necessary.
    //
    // Care is needed around the core module as it is only compiled once and *without* obfuscation,
    // so any linkage name to the core module *shouldn't* have obfuscation applied to it.
    if (context->shared->m_obfuscateCode && !isFromCoreModule(decl))
    {
        const auto obfuscatedName = getHashedName(mangledName.getUnownedSlice());

        addLinkageDecoration(context, inst, decl, obfuscatedName.getUnownedSlice());
    }
    else
    {
        addLinkageDecoration(context, inst, decl, mangledName.getUnownedSlice());
    }
}

bool shouldDeclBeTreatedAsInterfaceRequirement(Decl* requirementDecl)
{
    if (const auto funcDecl = as<CallableDecl>(requirementDecl))
    {
        // Subscript decl itself won't have a witness table entry.
        // But its accessors will.
        if (const auto subscriptDecl = as<SubscriptDecl>(requirementDecl))
            return false;
    }
    else if (const auto assocTypeDecl = as<AssocTypeDecl>(requirementDecl))
    {
    }
    else if (const auto typeConstraint = as<TypeConstraintDecl>(requirementDecl))
    {
    }
    else if (const auto varDecl = as<VarDeclBase>(requirementDecl))
    {
    }
    else if (const auto genericDecl = as<GenericDecl>(requirementDecl))
    {
        return shouldDeclBeTreatedAsInterfaceRequirement(genericDecl->inner);
    }
    else
    {
        // We will return false for PropertyDecl because the property decl itself
        // won't have a witness table entry. Instead there will be witness entries
        // for its accessors.
        return false;
    }
    return true;
}

IRStructKey* getInterfaceRequirementKey(IRGenContext* context, Decl* requirementDecl)
{
    // TODO: this special case logic can be removed if we also clean up
    // `doesGenericSignatureMatchRequirement` Currently `doesGenericSignatureMatchRequirement` will
    // use the inner func decl as the key in AST WitnessTable. Therefore we need to match this
    // behavior by always using the inner decl as the requirement key.
    if (auto genericDecl = as<GenericDecl>(requirementDecl))
        return getInterfaceRequirementKey(context, genericDecl->inner);

    // Only specific types of decls are treated as requirements, e.g. methods and asssociated types.
    // Other types of decls are allowed but not regarded as a requirement.
    if (!shouldDeclBeTreatedAsInterfaceRequirement(requirementDecl))
        return nullptr;

    IRStructKey* requirementKey = nullptr;
    if (context->shared->interfaceRequirementKeys.tryGetValue(requirementDecl, requirementKey))
    {
        return requirementKey;
    }

    IRBuilder builderStorage = *context->irBuilder;
    auto builder = &builderStorage;

    builder->setInsertInto(builder->getModule());

    // Construct a key to serve as the representation of
    // this requirement in the IR, and to allow lookup
    // into the declaration.
    requirementKey = builder->createStructKey();

    addLinkageDecoration(context, requirementKey, requirementDecl);

    context->shared->interfaceRequirementKeys.add(requirementDecl, requirementKey);

    return requirementKey;
}

void getGenericTypeConformances(
    IRGenContext* context,
    ShortList<IRType*>& supTypes,
    Decl* genericParamDecl)
{
    auto parent = genericParamDecl->parentDecl;
    if (parent)
    {
        for (auto typeConstraint : parent->getMembersOfType<GenericTypeConstraintDecl>())
        {
            if (auto declRefType = as<DeclRefType>(typeConstraint->sub.type))
            {
                if (declRefType->getDeclRef().getDecl() == genericParamDecl)
                {
                    supTypes.add(lowerType(context, typeConstraint->getSup().type));
                }
            }
        }
    }
}


// Check if declRef represents a witness that `ISomeInterface.This : ISomeInterface`.
static bool _isThisTypeSubtypeWitness(DeclRefBase* declRef)
{
    auto lookupDeclRef = as<LookupDeclRef>(declRef);
    if (!lookupDeclRef)
        return false;
    if (!as<ThisType>(lookupDeclRef->getLookupSource()))
        return false;
    auto declaredWitness = as<DeclaredSubtypeWitness>(lookupDeclRef->getWitness());
    if (!declaredWitness)
        return false;
    if (!as<ThisTypeConstraintDecl>(declaredWitness->getDeclRef()))
        return false;
    return true;
}

// Returns whether `declRef` represents a trivial lookup of an interface requirement
// through `ThisTypeDecl` made from within the same interface Decl.
static bool _isTrivialLookupFromInterfaceThis(IRGenContext* context, DeclRefBase* declRef)
{
    if (!_isThisTypeSubtypeWitness(declRef))
        return false;
    // This is a lookup from an interface's This type.
    // If the lookup is made from an interface type itself rather than an extension of it,
    // then it is a trivial lookup and we should lower it as a struct key.
    return context->thisTypeWitness == nullptr;
}


//

struct ValLoweringVisitor : ValVisitor<ValLoweringVisitor, LoweredValInfo, LoweredValInfo>
{
    IRGenContext* context;

    IRBuilder* getBuilder() { return context->irBuilder; }

    LoweredValInfo visitVal(Val* /*val*/)
    {
        SLANG_UNIMPLEMENTED_X("value lowering");
        UNREACHABLE_RETURN(LoweredValInfo());
    }

    LoweredValInfo visitGenericParamIntVal(GenericParamIntVal* val)
    {
        return emitDeclRef(
            context,
            val->getDeclRef(),
            lowerType(context, getType(context->astBuilder, val->getDeclRef())));
    }

    LoweredValInfo visitFuncCallIntVal(FuncCallIntVal* val)
    {
        TryClauseEnvironment tryEnv;
        List<IRInst*> args;
        for (auto arg : val->getArgs())
        {
            auto loweredArg = lowerVal(context, arg);
            args.add(loweredArg.val);
        }
        auto funcType = lowerType(context, val->getFuncType());
        return emitCallToDeclRef(
            context,
            as<IRFuncType>(funcType)->getResultType(),
            val->getFuncDeclRef(),
            funcType,
            args,
            tryEnv);
    }

    LoweredValInfo visitTypeCastIntVal(TypeCastIntVal* val)
    {
        auto baseVal = lowerVal(context, val->getBase());
        SLANG_ASSERT(baseVal.flavor == LoweredValInfo::Flavor::Simple);
        auto type = lowerType(context, val->getType());
        return LoweredValInfo::simple(getBuilder()->emitCast(type, baseVal.val));
    }

    LoweredValInfo visitWitnessLookupIntVal(WitnessLookupIntVal* val)
    {
        auto witnessVal = lowerVal(context, val->getWitness());
        auto key = getInterfaceRequirementKey(context, val->getKey());
        auto type = lowerType(context, val->getType());
        return LoweredValInfo::simple(
            getBuilder()->emitLookupInterfaceMethodInst(type, witnessVal.val, key));
    }

    LoweredValInfo visitPolynomialIntVal(PolynomialIntVal* val)
    {
        auto irBuilder = getBuilder();
        auto type = lowerType(context, val->getType());
        auto constTerm = irBuilder->getIntValue(type, val->getConstantTerm());
        auto resultVal = constTerm;
        for (auto term : val->getTerms())
        {
            auto termVal = irBuilder->getIntValue(type, term->getConstFactor());
            for (auto factor : term->getParamFactors())
            {
                auto factorVal = lowerVal(context, factor->getParam()).val;
                for (IntegerLiteralValue i = 0; i < factor->getPower(); i++)
                {
                    termVal = irBuilder->emitMul(factorVal->getDataType(), termVal, factorVal);
                }
            }
            resultVal = irBuilder->emitAdd(termVal->getDataType(), resultVal, termVal);
        }
        return LoweredValInfo::simple(resultVal);
    }

    LoweredValInfo visitCountOfIntVal(CountOfIntVal* val)
    {
        auto irBuilder = getBuilder();
        auto type = lowerType(context, val->getType());
        auto typeArg = lowerType(context, as<Type>(val->getTypeArg()));
        auto count = irBuilder->emitCountOf(type, typeArg);
        return LoweredValInfo::simple(count);
    }

    LoweredValInfo visitConcreteTypePack(ConcreteTypePack* typePack)
    {
        ShortList<IRType*> types;
        for (Index i = 0; i < typePack->getTypeCount(); i++)
        {
            auto loweredType = lowerType(context, typePack->getElementType(i));
            types.add(loweredType);
        }
        auto irBuilder = getBuilder();
        IRType* irTypePack =
            irBuilder->getTypePack((UInt)types.getCount(), types.getArrayView().getBuffer());
        return LoweredValInfo::simple(irTypePack);
    }

    LoweredValInfo visitEachType(EachType* eachType)
    {
        auto type = lowerType(context, eachType->getElementType());
        return LoweredValInfo::simple(
            getBuilder()->emitEachInst(getBuilder()->getTypeKind(), type));
    }

    LoweredValInfo visitExpandType(ExpandType* expandType)
    {
        auto irBuilder = getBuilder();
        auto type = lowerType(context, expandType->getPatternType());
        ShortList<IRInst*> capturedTypes;
        for (Index i = 0; i < expandType->getCapturedTypePackCount(); i++)
        {
            auto loweredType = lowerType(context, expandType->getCapturedTypePack(i));
            capturedTypes.add(loweredType);
        }
        return LoweredValInfo::simple(irBuilder->getExpandTypeOrVal(
            irBuilder->getTypeKind(),
            type,
            capturedTypes.getArrayView().arrayView));
    }

    LoweredValInfo visitTypePackSubtypeWitness(TypePackSubtypeWitness* witnessPack)
    {
        auto irBuilder = getBuilder();
        ShortList<IRInst*> witnesses;
        ShortList<IRType*> elementTypes;
        for (Index i = 0; i < witnessPack->getCount(); i++)
        {
            auto loweredWitness = lowerVal(context, witnessPack->getWitness(i));
            witnesses.add(loweredWitness.val);
            elementTypes.add(loweredWitness.val->getFullType());
        }
        auto irWitnessPack = irBuilder->emitMakeWitnessPack(
            irBuilder->getTupleType(
                (UInt)elementTypes.getCount(),
                elementTypes.getArrayView().getBuffer()),
            witnesses.getArrayView().arrayView);
        return LoweredValInfo::simple(irWitnessPack);
    }

    LoweredValInfo visitExpandSubtypeWitness(ExpandSubtypeWitness* witness)
    {
        auto irBuilder = getBuilder();

        auto patternWitnessVal = lowerVal(context, witness->getPatternTypeWitness());
        auto subType = lowerType(context, witness->getSub());
        auto supType = lowerType(context, witness->getSup());
        auto witnessTableType = irBuilder->getWitnessTableType(supType);
        ShortList<IRInst*> captures;
        if (auto expandType = as<IRExpandType>(subType))
        {
            for (UInt i = 0; i < expandType->getCaptureCount(); i++)
            {
                captures.add(expandType->getCaptureType(i));
            }
        }
        return LoweredValInfo::simple(irBuilder->getExpandTypeOrVal(
            witnessTableType,
            patternWitnessVal.val,
            captures.getArrayView().arrayView));
    }

    LoweredValInfo visitEachSubtypeWitness(EachSubtypeWitness* witness)
    {
        auto elementWitness = lowerVal(context, witness->getPatternTypeWitness());
        auto irBuilder = getBuilder();
        auto subType = lowerType(context, witness->getSub());
        auto witnessTableType = irBuilder->getWitnessTableType(subType);
        return LoweredValInfo::simple(
            irBuilder->emitEachInst(witnessTableType, getSimpleVal(context, elementWitness)));
    }

    LoweredValInfo visitDeclaredSubtypeWitness(DeclaredSubtypeWitness* val)
    {
        if (as<ThisTypeConstraintDecl>(val->getDeclRef()))
            return LoweredValInfo::simple(context->thisTypeWitness);

        return emitDeclRef(
            context,
            val->getDeclRef(),
            context->irBuilder->getWitnessTableType(lowerType(context, val->getSup())));
    }

    LoweredValInfo visitTypeEqualityWitness(TypeEqualityWitness* val)
    {
        auto subType = lowerType(context, val->getSub());
        auto supType = lowerType(context, val->getSup());
        auto witnessType =
            context->irBuilder->getWitnessTableType(lowerType(context, val->getSup()));
        return LoweredValInfo::simple(
            context->irBuilder->getTypeEqualityWitness(witnessType, subType, supType));
    }

    LoweredValInfo visitTypeCoercionWitness(TypeCoercionWitness*)
    {
        // When we fully support type coercion constraints, we should lower the witness into a
        // function that does the conversion.
        return LoweredValInfo();
    }

    LoweredValInfo visitTransitiveSubtypeWitness(TransitiveSubtypeWitness* val)
    {
        // The base (subToMid) will turn into a value with
        // witness-table type.
        IRInst* baseWitnessTable = lowerSimpleVal(context, val->getSubToMid());
        IRInst* midToSup = nullptr;

        // The next step should map to an interface requirement
        // that is itself an interface conformance, so the result
        // of lowering this value should be a "key" that we can
        // use to look up a witness table.
        //
        // TODO: There are some ugly cases here if `midToSup` is allowed
        // to be an arbitrary witness, rather than just a declared one,
        // and we probably need to change the logic here so that we
        // instead think in terms of applying a subtype witness to
        // either a value or a witness table, to perform the appropriate
        // casting/lookup logic.
        //
        // For now we rely on the fact that the front-end doesn't
        // produce transitive witnesses in shapes that will cuase us
        // problems here.
        //
        SLANG_RELEASE_ASSERT(baseWitnessTable);

        if (auto declaredMidToSup = as<DeclaredSubtypeWitness>(val->getMidToSup()))
        {
            midToSup =
                getInterfaceRequirementKey(context, declaredMidToSup->getDeclRef().getDecl());
        }
        else
        {
            midToSup = lowerSimpleVal(context, val->getMidToSup());
        }

        return LoweredValInfo::simple(getBuilder()->emitLookupInterfaceMethodInst(
            getBuilder()->getWitnessTableType(lowerType(context, val->getSup())),
            baseWitnessTable,
            midToSup));
    }

    LoweredValInfo visitForwardDifferentiateVal(ForwardDifferentiateVal* val)
    {
        // TODO: properly fill in type info here.
        // We should consider fold all cases of witness table entries to `Val`, and make the
        // `DeclRef` case a `DeclRefVal`. So that we can hold the type in `DeclRefVal`.
        auto funcVal = emitDeclRef(context, val->getFunc(), context->irBuilder->getTypeKind());
        SLANG_RELEASE_ASSERT(funcVal.flavor == LoweredValInfo::Flavor::Simple);

        auto diff =
            getBuilder()->emitForwardDifferentiateInst(getBuilder()->getTypeKind(), funcVal.val);
        return LoweredValInfo::simple(diff);
    }

    LoweredValInfo visitBackwardDifferentiateVal(BackwardDifferentiateVal* val)
    {
        auto funcVal = emitDeclRef(context, val->getFunc(), context->irBuilder->getTypeKind());
        SLANG_RELEASE_ASSERT(funcVal.flavor == LoweredValInfo::Flavor::Simple);

        auto diff =
            getBuilder()->emitBackwardDifferentiateInst(getBuilder()->getTypeKind(), funcVal.val);
        return LoweredValInfo::simple(diff);
    }

    LoweredValInfo visitBackwardDifferentiatePropagateVal(BackwardDifferentiatePropagateVal* val)
    {
        auto funcVal = emitDeclRef(context, val->getFunc(), context->irBuilder->getTypeKind());
        SLANG_RELEASE_ASSERT(funcVal.flavor == LoweredValInfo::Flavor::Simple);

        auto diff = getBuilder()->emitBackwardDifferentiatePropagateInst(
            getBuilder()->getTypeKind(),
            funcVal.val);
        return LoweredValInfo::simple(diff);
    }

    LoweredValInfo visitBackwardDifferentiatePrimalVal(BackwardDifferentiatePrimalVal* val)
    {
        auto funcVal = emitDeclRef(context, val->getFunc(), context->irBuilder->getTypeKind());
        SLANG_RELEASE_ASSERT(funcVal.flavor == LoweredValInfo::Flavor::Simple);

        auto diff = getBuilder()->emitBackwardDifferentiatePrimalInst(
            getBuilder()->getTypeKind(),
            funcVal.val);
        return LoweredValInfo::simple(diff);
    }

    LoweredValInfo visitBackwardDifferentiateIntermediateTypeVal(
        BackwardDifferentiateIntermediateTypeVal* val)
    {
        auto funcVal = emitDeclRef(context, val->getFunc(), context->irBuilder->getTypeKind());
        SLANG_RELEASE_ASSERT(funcVal.flavor == LoweredValInfo::Flavor::Simple);

        auto diff = getBuilder()->getBackwardDiffIntermediateContextType(funcVal.val);
        return LoweredValInfo::simple(diff);
    }

    LoweredValInfo visitDynamicSubtypeWitness(DynamicSubtypeWitness* /*val*/)
    {
        return LoweredValInfo::simple(nullptr);
    }

    LoweredValInfo visitConjunctionSubtypeWitness(ConjunctionSubtypeWitness* val)
    {
        // A witness `W = X & Y & ...` will lower as a tuple of the sub-witnesses
        // `X`, `Y`, etc.
        //
        // The AST representation of a conjunction of witnesses matches this
        // tuple-like encoding very closely, so we can simply lower each of
        // the component witnesses to produce our result.
        //
        List<IRInst*> componentWitnesses;
        auto componentCount = val->getComponentCount();
        for (Index i = 0; i < componentCount; ++i)
        {
            auto componentWitness = lowerSimpleVal(context, val->getComponentWitness(i));
            componentWitnesses.add(componentWitness);
        }
        return LoweredValInfo::simple(getBuilder()->emitMakeTuple(componentWitnesses));
    }

    LoweredValInfo visitExtractFromConjunctionSubtypeWitness(
        ExtractFromConjunctionSubtypeWitness* val)
    {
        auto builder = getBuilder();

        // We know from `visitConjunctionSubtypeWitness` that a witness for a relationship
        // like `T : L & R` will be a tuple `(w_l, w_r)` where `w_l` is a witness
        // for `T : L` and `w_r` will be a witness for `T : R`.
        //
        // An `ExtractFromConjunctionSubtypeWitness` represents the intention to
        // extract one of those two sub-witnesses. It directly stores the original
        // witness that `T : L & R`, so lower that first and expect it to be
        // a value of tuple type.
        //
        auto conjunctionWitness = lowerSimpleVal(context, val->getConjunctionWitness());
        auto conjunctionTupleType = as<IRTupleType>(conjunctionWitness->getDataType());
        SLANG_ASSERT(conjunctionTupleType);

        // The `ExtractFromConjunctionSubtypeWitness` also stores the index of
        // the witness/supertype we want in the conjunction `L & R`.
        //
        auto indexInConjunction = val->getIndexInConjunction();

        // We want to extract the appropriate element from the tuple based on
        // the index, but to know the type of the result we need to look up
        // the element type that corresponds to that index.
        //
        // TODO: `IRTupleType` should really have `getElementCount()` and
        // `getElementType(index)` accessors.
        //
        auto elementType = (IRType*)conjunctionTupleType->getOperand(indexInConjunction);

        // With the information we've extracted above, we now just need to
        // extract the appropriate element from the `(w_l, w_r)` tuple of
        // witnesses, and we will have our desired result.
        //
        return LoweredValInfo::simple(
            builder->emitGetTupleElement(elementType, conjunctionWitness, indexInConjunction));
    }


    LoweredValInfo visitConstantIntVal(ConstantIntVal* val)
    {
        auto type = lowerType(context, val->getType());
        return LoweredValInfo::simple(getBuilder()->getIntValue(type, val->getValue()));
    }

    IRType* visitDifferentialPairType(DifferentialPairType* pairType)
    {
        IRType* primalType = lowerType(context, pairType->getPrimalType());
        if (as<IRAssociatedType>(primalType) || as<IRThisType>(primalType))
        {
            List<IRInst*> operands;
            SubstitutionSet(pairType->getDeclRef())
                .forEachSubstitutionArg(
                    [&](Val* arg)
                    {
                        auto argVal = lowerVal(context, arg).val;
                        SLANG_ASSERT(argVal);
                        operands.add(argVal);
                    });

            auto undefined = getBuilder()->emitUndefined(operands[1]->getFullType());
            return getBuilder()->getDifferentialPairUserCodeType(primalType, undefined);
        }
        else
            return lowerSimpleIntrinsicType(pairType);
    }

    IRFuncType* visitFuncType(FuncType* type)
    {
        IRType* resultType = lowerType(context, type->getResultType());
        Index paramCount = type->getParamCount();
        List<IRType*> paramTypes;
        for (Index pp = 0; pp < paramCount; ++pp)
        {
            paramTypes.add(lowerType(context, type->getParamType(pp)));
        }
        if (type->getErrorType()->equals(context->astBuilder->getBottomType()))
        {
            return getBuilder()->getFuncType(paramCount, paramTypes.getBuffer(), resultType);
        }
        else
        {
            auto errorType = lowerType(context, type->getErrorType());
            auto irThrowFuncTypeAttribute =
                getBuilder()->getAttr(kIROp_FuncThrowTypeAttr, 1, (IRInst**)&errorType);
            return getBuilder()->getFuncType(
                paramCount,
                paramTypes.getBuffer(),
                resultType,
                irThrowFuncTypeAttribute);
        }
    }

    IRType* visitPtrType(PtrType* type)
    {
        auto astValueType = type->getValueType();

        IRType* irValueType = lowerType(context, astValueType);
        IRInst* addrSpace = nullptr;
        if (auto astAddrSpace = type->getAddressSpace())
        {
            addrSpace = getSimpleVal(context, lowerVal(context, astAddrSpace));
        }
        else
        {
            addrSpace = getBuilder()->getIntValue(
                getBuilder()->getUInt64Type(),
                (IRIntegerValue)AddressSpace::Generic);
        }
        return getBuilder()->getPtrType(kIROp_PtrType, irValueType, addrSpace);
    }

    IRType* visitDeclRefType(DeclRefType* type)
    {
        auto declRef = type->getDeclRef();
        auto decl = declRef.getDecl();

        // Check for types with teh `__intrinsic_type` modifier.
        if (decl->findModifier<IntrinsicTypeModifier>())
        {
            return lowerSimpleIntrinsicType(type);
        }


        return (IRType*)getSimpleVal(
            context,
            emitDeclRef(context, declRef, context->irBuilder->getTypeKind()));
    }

    IRType* visitTupleType(TupleType* type)
    {
        List<IRType*> elementTypes;
        if (as<ConcreteTypePack>(type->getTypePack()))
        {
            for (Index i = 0; i < type->getMemberCount(); i++)
            {
                elementTypes.add(lowerType(context, type->getMember(i)));
            }
            return context->irBuilder->getTupleType(elementTypes);
        }
        else
        {
            return context->irBuilder->getTupleType(lowerType(context, type->getTypePack()));
        }
    }

    IRType* visitNamedExpressionType(NamedExpressionType* type)
    {
        return (IRType*)getSimpleVal(context, dispatchType(type->getCanonicalType()));
    }

    IRType* visitBasicExpressionType(BasicExpressionType* type)
    {
        return getBuilder()->getBasicType(type->getBaseType());
    }

    IRType* visitVectorExpressionType(VectorExpressionType* type)
    {
        auto elementType = lowerType(context, type->getElementType());
        auto elementCount = lowerSimpleVal(context, type->getElementCount());

        return getBuilder()->getVectorType(elementType, elementCount);
    }

    IRType* visitMatrixExpressionType(MatrixExpressionType* type)
    {
        auto elementType = lowerType(context, type->getElementType());
        auto rowCount = lowerSimpleVal(context, type->getRowCount());
        auto columnCount = lowerSimpleVal(context, type->getColumnCount());
        auto layout = lowerSimpleVal(context, type->getLayout());
        return getBuilder()->getMatrixType(elementType, rowCount, columnCount, layout);
    }

    IRType* visitArrayExpressionType(ArrayExpressionType* type)
    {
        auto elementType = lowerType(context, type->getElementType());
        if (!type->isUnsized())
        {
            auto elementCount = lowerSimpleVal(context, type->getElementCount());
            return getBuilder()->getArrayType(elementType, elementCount);
        }
        else
        {
            return getBuilder()->getUnsizedArrayType(elementType);
        }
    }

    // Lower a type where the type declaration being referenced is assumed
    // to be an intrinsic type, which can thus be lowered to a simple IR
    // type with the appropriate opcode.
    IRType* lowerSimpleIntrinsicType(DeclRefType* type)
    {
        SLANG_ASSERT(getBuilder()->getInsertLoc().getMode() != IRInsertLoc::Mode::None);

        auto intrinsicTypeModifier =
            type->getDeclRef().getDecl()->findModifier<IntrinsicTypeModifier>();
        SLANG_ASSERT(intrinsicTypeModifier);
        IROp op = IROp(intrinsicTypeModifier->irOp);
        List<IRInst*> operands;
        // If there are any substitutions attached to the declRef,
        // add them as operands of the IR type.
        SubstitutionSet(type->getDeclRef())
            .forEachSubstitutionArg(
                [&](Val* arg)
                {
                    auto argVal = lowerVal(context, arg).val;
                    SLANG_ASSERT(argVal);
                    operands.add(argVal);
                });
        return getBuilder()->getType(
            op,
            static_cast<UInt>(operands.getCount()),
            operands.getBuffer());
    }

    IRType* visitResourceType(ResourceType* type) { return lowerSimpleIntrinsicType(type); }

    IRType* visitSamplerStateType(SamplerStateType* type) { return lowerSimpleIntrinsicType(type); }

    IRType* visitBuiltinGenericType(BuiltinGenericType* type)
    {
        return lowerSimpleIntrinsicType(type);
    }

    IRType* visitUntypedBufferResourceType(UntypedBufferResourceType* type)
    {
        return lowerSimpleIntrinsicType(type);
    }

    IRType* visitHLSLPatchType(HLSLPatchType* type) { return lowerSimpleIntrinsicType(type); }

    IRType* visitMeshOutputType(MeshOutputType* type) { return lowerSimpleIntrinsicType(type); }

    IRType* visitExtractExistentialType(ExtractExistentialType* type)
    {
        auto declRef = type->getDeclRef();
        auto existentialType = lowerType(context, getType(context->astBuilder, declRef));
        IRInst* existentialVal =
            getSimpleVal(context, emitDeclRef(context, declRef, existentialType));
        return getBuilder()->emitExtractExistentialType(existentialVal);
    }

    LoweredValInfo visitExtractExistentialSubtypeWitness(ExtractExistentialSubtypeWitness* witness)
    {
        auto declRef = witness->getDeclRef();
        auto existentialType = lowerType(context, getType(context->astBuilder, declRef));
        IRInst* existentialVal =
            getSimpleVal(context, emitDeclRef(context, declRef, existentialType));
        return LoweredValInfo::simple(
            getBuilder()->emitExtractExistentialWitnessTable(existentialVal));
    }

    LoweredValInfo visitExistentialSpecializedType(ExistentialSpecializedType* type)
    {
        auto irBaseType = lowerType(context, type->getBaseType());

        List<IRInst*> slotArgs;
        for (Index i = 0; i < type->getArgCount(); i++)
        {
            auto arg = type->getArg(i);
            auto irArgVal = lowerSimpleVal(context, arg.val);
            slotArgs.add(irArgVal);

            if (auto witness = arg.witness)
            {
                auto irArgWitness = lowerSimpleVal(context, witness);
                slotArgs.add(irArgWitness);
            }
        }

        auto irType = getBuilder()->getBindExistentialsType(
            irBaseType,
            slotArgs.getCount(),
            slotArgs.getBuffer());
        return LoweredValInfo::simple(irType);
    }

    LoweredValInfo visitThisType(ThisType* type)
    {
        // A `This` type in an interface decl should lower to `IRThisType`,
        // while `This` type in a concrete `struct` should lower to the `struct` type
        // itself. A `This` type reference in a concrete type is already translated to that
        // type in semantics checking in this setting.
        // If we see `This` type here, we are dealing with `This` inside an interface decl.
        // Therefore, `context->thisType` should have been set to `IRThisType`
        // in `visitInterfaceDecl`, and we can just use that value here.
        //
        if (context->thisType.irType)
        {
            return LoweredValInfo::simple(context->thisType.irType);
        }
        auto interfaceType =
            emitDeclRef(context, type->getInterfaceDeclRef(), getBuilder()->getTypeKind());
        auto result = LoweredValInfo::simple(
            getBuilder()->getThisType((IRType*)getSimpleVal(context, interfaceType)));
        if (context->thisType.astType == type)
        {
            context->thisType = getSimpleVal(context, result);
        }
        return result;
    }

    LoweredValInfo visitAndType(AndType* type)
    {
        auto left = lowerType(context, type->getLeft());
        auto right = lowerType(context, type->getRight());

        auto irType = getBuilder()->getConjunctionType(left, right);
        return LoweredValInfo::simple(irType);
    }

    LoweredValInfo visitModifiedType(ModifiedType* astType)
    {
        IRType* irBase = lowerType(context, astType->getBase());

        List<IRAttr*> irAttrs;
        for (Index i = 0; i < astType->getModifierCount(); i++)
        {
            auto astModifier = astType->getModifier(i);
            IRAttr* irAttr = (IRAttr*)lowerSimpleVal(context, astModifier);
            if (irAttr)
                irAttrs.add(irAttr);
        }

        auto irType = getBuilder()->getAttributedType(irBase, irAttrs);
        return LoweredValInfo::simple(irType);
    }

    LoweredValInfo visitUNormModifierVal(UNormModifierVal* astVal)
    {
        SLANG_UNUSED(astVal);
        return LoweredValInfo::simple(getBuilder()->getAttr(kIROp_UNormAttr));
    }

    LoweredValInfo visitSNormModifierVal(SNormModifierVal* astVal)
    {
        SLANG_UNUSED(astVal);
        return LoweredValInfo::simple(getBuilder()->getAttr(kIROp_SNormAttr));
    }

    LoweredValInfo visitNoDiffModifierVal(NoDiffModifierVal* astVal)
    {
        SLANG_UNUSED(astVal);
        return LoweredValInfo::simple(getBuilder()->getAttr(kIROp_NoDiffAttr));
    }

    // We do not expect to encounter the following types in ASTs that have
    // passed front-end semantic checking.
#define UNEXPECTED_CASE(NAME)        \
    IRType* visit##NAME(NAME*)       \
    {                                \
        SLANG_UNEXPECTED(#NAME);     \
        UNREACHABLE_RETURN(nullptr); \
    }
    UNEXPECTED_CASE(GenericDeclRefType)
    UNEXPECTED_CASE(TypeType)
    UNEXPECTED_CASE(ErrorType)
    UNEXPECTED_CASE(InitializerListType)
    UNEXPECTED_CASE(OverloadGroupType)
    UNEXPECTED_CASE(NamespaceType)
#undef UNEXPECTED_CASE
};

LoweredValInfo lowerVal(IRGenContext* context, Val* val)
{
    ValLoweringVisitor visitor;
    visitor.context = context;
    auto resolvedVal = val->resolve();
    return visitor.dispatch(resolvedVal);
}

IRType* lowerType(IRGenContext* context, Type* type)
{
    ValLoweringVisitor visitor;
    visitor.context = context;
    IRType* loweredType = (IRType*)getSimpleVal(context, visitor.dispatchType(type));

    if (context->lowerTypeCallback && loweredType)
        context->lowerTypeCallback(context, type, loweredType);

    return loweredType;
}

void addVarDecorations(IRGenContext* context, IRInst* inst, Decl* decl)
{
    auto builder = context->irBuilder;
    for (Modifier* mod : decl->modifiers)
    {
        if (as<HLSLNoInterpolationModifier>(mod))
        {
            builder->addInterpolationModeDecoration(inst, IRInterpolationMode::NoInterpolation);
        }
        else if (as<PerVertexModifier>(mod))
        {
            builder->addInterpolationModeDecoration(inst, IRInterpolationMode::PerVertex);
        }
        else if (as<HLSLNoPerspectiveModifier>(mod))
        {
            builder->addInterpolationModeDecoration(inst, IRInterpolationMode::NoPerspective);
        }
        else if (as<HLSLLinearModifier>(mod))
        {
            builder->addInterpolationModeDecoration(inst, IRInterpolationMode::Linear);
        }
        else if (as<HLSLSampleModifier>(mod))
        {
            builder->addInterpolationModeDecoration(inst, IRInterpolationMode::Sample);
        }
        else if (as<HLSLCentroidModifier>(mod))
        {
            builder->addInterpolationModeDecoration(inst, IRInterpolationMode::Centroid);
        }
        else if (auto rayPayloadAttr = as<VulkanRayPayloadAttribute>(mod))
        {
            builder->addVulkanRayPayloadDecoration(inst, rayPayloadAttr->location);
            // may not be referenced; adding HLSL export modifier force emits
            builder->addHLSLExportDecoration(inst);
        }
        else if (auto rayPayloadInAttr = as<VulkanRayPayloadInAttribute>(mod))
        {
            builder->addVulkanRayPayloadInDecoration(inst, rayPayloadInAttr->location);
            // may not be referenced; adding HLSL export modifier force emits
            builder->addHLSLExportDecoration(inst);
        }
        else if (auto callablePayloadAttr = as<VulkanCallablePayloadAttribute>(mod))
        {
            builder->addVulkanCallablePayloadDecoration(inst, callablePayloadAttr->location);
            // may not be referenced; adding HLSL export modifier force emits
            builder->addHLSLExportDecoration(inst);
        }
        else if (auto callablePayloadInAttr = as<VulkanCallablePayloadInAttribute>(mod))
        {
            builder->addVulkanCallablePayloadInDecoration(inst, callablePayloadInAttr->location);
            // may not be referenced; adding HLSL export modifier force emits
            builder->addHLSLExportDecoration(inst);
        }
        else if (auto hitObjectAttr = as<VulkanHitObjectAttributesAttribute>(mod))
        {
            builder->addVulkanHitObjectAttributesDecoration(inst, hitObjectAttr->location);
            // may not be referenced; adding HLSL export modifier force emits
            builder->addHLSLExportDecoration(inst);
        }
        else if (as<VulkanHitAttributesAttribute>(mod))
        {
            builder->addSimpleDecoration<IRVulkanHitAttributesDecoration>(inst);
            // may not be referenced; adding HLSL export modifier force emits
            builder->addHLSLExportDecoration(inst);
        }
        else if (as<PreciseModifier>(mod))
        {
            builder->addSimpleDecoration<IRPreciseDecoration>(inst);
        }
        else if (auto formatAttr = as<FormatAttribute>(mod))
        {
            builder->addFormatDecoration(inst, formatAttr->format);
        }
        else if (as<HLSLPayloadModifier>(mod))
        {
            builder->addSimpleDecoration<IRHLSLMeshPayloadDecoration>(inst);
        }
        else if (as<OutModifier>(mod))
        {
            builder->addSimpleDecoration<IRGlobalOutputDecoration>(inst);
        }
        else if (as<InModifier>(mod))
        {
            builder->addSimpleDecoration<IRGlobalInputDecoration>(inst);
        }
        else if (auto glslLocationMod = as<GLSLLocationAttribute>(mod))
        {
            builder->addDecoration(
                inst,
                kIROp_GLSLLocationDecoration,
                builder->getIntValue(builder->getIntType(), glslLocationMod->value));
        }
        else if (auto glslOffsetMod = as<GLSLOffsetLayoutAttribute>(mod))
        {
            builder->addDecoration(
                inst,
                kIROp_GLSLOffsetDecoration,
                builder->getIntValue(builder->getIntType(), glslOffsetMod->offset));
        }
        else if (auto glslStructOffsetMod = as<VkStructOffsetAttribute>(mod))
        {
            builder->addDecoration(
                inst,
                kIROp_VkStructOffsetDecoration,
                builder->getIntValue(builder->getIntType(), glslStructOffsetMod->value));
        }
        else if (auto hlslSemantic = as<HLSLSimpleSemantic>(mod))
        {
            builder->addSemanticDecoration(inst, hlslSemantic->name.getContent());
        }
        else if (as<DynamicUniformModifier>(mod))
        {
            builder->addDynamicUniformDecoration(inst);
        }
        else if (auto collection = as<MemoryQualifierSetModifier>(mod))
        {
            builder->addMemoryQualifierSetDecoration(
                inst,
                IRIntegerValue(collection->getMemoryQualifierBit()));
        }
        else if (auto geometryModifier = as<HLSLGeometryShaderInputPrimitiveTypeModifier>(mod))
        {
            IROp op = kIROp_Invalid;
            switch (geometryModifier->astNodeType)
            {
            case ASTNodeType::HLSLTriangleModifier:
                op = kIROp_TriangleInputPrimitiveTypeDecoration;
                break;
            case ASTNodeType::HLSLPointModifier:
                op = kIROp_PointInputPrimitiveTypeDecoration;
                break;
            case ASTNodeType::HLSLLineModifier:
                op = kIROp_LineInputPrimitiveTypeDecoration;
                break;
            case ASTNodeType::HLSLLineAdjModifier:
                op = kIROp_LineAdjInputPrimitiveTypeDecoration;
                break;
            case ASTNodeType::HLSLTriangleAdjModifier:
                op = kIROp_TriangleAdjInputPrimitiveTypeDecoration;
                break;
            }
            if (op != kIROp_Invalid)
                builder->addDecoration(inst, op);
        }
        // TODO: what are other modifiers we need to propagate through?
    }
    if (auto t =
            composeGetters<IRMeshOutputType>(inst->getFullType(), &IROutTypeBase::getValueType))
    {
        IROp op;
        switch (t->getOp())
        {
        case kIROp_VerticesType:
            op = kIROp_VerticesDecoration;
            break;
        case kIROp_IndicesType:
            op = kIROp_IndicesDecoration;
            break;
        case kIROp_PrimitivesType:
            op = kIROp_PrimitivesDecoration;
            break;
        default:
            SLANG_UNREACHABLE("Missing case for IRMeshOutputType");
            break;
        }
        builder->addMeshOutputDecoration(op, inst, t->getMaxElementCount());
    }
}

/// If `decl` has a modifier that should turn into a
/// rate qualifier, then apply it to `inst`.
void maybeSetRate(IRGenContext* context, IRInst* inst, Decl* decl)
{
    auto builder = context->irBuilder;

    if (decl->hasModifier<HLSLGroupSharedModifier>())
    {
        inst->setFullType(
            builder->getRateQualifiedType(builder->getGroupSharedRate(), inst->getFullType()));
    }
    else if (decl->hasModifier<ActualGlobalModifier>())
    {
        inst->setFullType(
            builder->getRateQualifiedType(builder->getActualGlobalRate(), inst->getFullType()));
    }
}

static String getNameForNameHint(IRGenContext* context, Decl* decl)
{
    // We will use a bit of an ad hoc convention here for now.

    Name* leafName = decl->getName();

    // Handle custom name for a global parameter group (e.g., a `cbuffer`)
    if (auto reflectionNameModifier = decl->findModifier<ParameterGroupReflectionName>())
    {
        leafName = reflectionNameModifier->nameAndLoc.name;
    }

    // There is no point in trying to provide a name hint for something with no name,
    // or with an empty name
    if (!leafName)
        return String();
    if (leafName->text.getLength() == 0)
        return String();


    if (const auto varDecl = as<VarDeclBase>(decl))
    {
        // For an ordinary local variable, global variable,
        // parameter, or field, we will just use the name
        // as declared, and now work in anything from
        // its parent declaration(s).
        //
        // TODO: consider whether global/static variables should
        // follow different rules.
        //
        return leafName->text;
    }

    // For other cases of declaration, we want to consider
    // merging its name with the name of its parent declaration.
    auto parentDecl = decl->parentDecl;

    // Skip past a generic parent, if we are a declaration nested in a generic.
    if (auto genericParentDecl = as<GenericDecl>(parentDecl))
        parentDecl = genericParentDecl->parentDecl;

    // Skip past a FileDecl parent.
    if (auto fileParentDecl = as<FileDecl>(parentDecl))
        parentDecl = fileParentDecl->parentDecl;

    // A `ModuleDecl` can have a name too, but in the common case
    // we don't want to generate name hints that include the module
    // name, simply because they would lead to every global symbol
    // getting a much longer name.
    //
    // TODO: We should probably include the module name for symbols
    // being `import`ed, and not for symbols being compiled directly
    // (those coming from a module that had no name given to it).
    //
    // For now we skip past a `ModuleDecl` parent.
    //
    if (auto moduleParentDecl = as<ModuleDecl>(parentDecl))
        parentDecl = moduleParentDecl->parentDecl;

    if (!parentDecl)
    {
        return leafName->text;
    }

    auto parentName = getNameForNameHint(context, parentDecl);
    if (parentName.getLength() == 0)
    {
        return leafName->text;
    }

    // We will now construct a new `Name` to use as the hint,
    // combining the name of the parent and the leaf declaration.

    StringBuilder sb;
    sb.append(parentName);
    sb.append(".");
    sb.append(leafName->text);

    return sb.produceString();
}

/// Try to add an appropriate name hint to the instruction,
/// that can be used for back-end code emission or debug info.
static void addNameHint(IRGenContext* context, IRInst* inst, Decl* decl)
{
    String name = getNameForNameHint(context, decl);
    if (name.getLength() == 0)
        return;
    context->irBuilder->addNameHintDecoration(inst, name.getUnownedSlice());
}

/// Add a name hint based on a fixed string.
static void addNameHint(IRGenContext* context, IRInst* inst, char const* text)
{
    if (context->shared->m_obfuscateCode)
    {
        return;
    }

    context->irBuilder->addNameHintDecoration(inst, UnownedTerminatedStringSlice(text));
}

LoweredValInfo createVar(IRGenContext* context, IRType* type, Decl* decl = nullptr)
{
    auto builder = context->irBuilder;
    auto irAlloc = builder->emitVar(type);

    if (decl)
    {
        maybeSetRate(context, irAlloc, decl);

        addVarDecorations(context, irAlloc, decl);

        builder->addHighLevelDeclDecoration(irAlloc, decl);

        addNameHint(context, irAlloc, decl);
    }

    return LoweredValInfo::ptr(irAlloc);
}

// When we try to turn a `LoweredValInfo` into an address of some temporary storage,
// we can either do it "aggressively" or not (what we'll call the "default" behavior,
// although it isn't strictly more common).
//
// The case that this is mostly there to address is when somebody writes an operation
// like:
//
//      foo[a] = b;
//
// In that case, we might as well just use the `set` accessor if there is one, rather
// than complicate things. However, in more complex cases like:
//
//      foo[a].x = b;
//
// there is no way to satisfy the semantics of the code the user wrote (in terms of
// only writing one vector component, and not a full vector) by using the `set`
// accessor, and we need to be "aggressive" in turning the lvalue `foo[a]` into
// an address.
//
// TODO: realistically IR lowering is too early to be binding to this choice,
// because different accessors might be supported on different targets.
//
enum class TryGetAddressMode
{
    Default,
    Aggressive,
};

/// Try to coerce `inVal` into a `LoweredValInfo::ptr()` with a simple address.
LoweredValInfo tryGetAddress(
    IRGenContext* context,
    LoweredValInfo const& inVal,
    TryGetAddressMode mode);

/// Add a single `in` argument value to a list of arguments
void addInArg(IRGenContext* context, List<IRInst*>* ioArgs, LoweredValInfo argVal)
{
    auto& args = *ioArgs;
    switch (argVal.flavor)
    {
    case LoweredValInfo::Flavor::Simple:
    case LoweredValInfo::Flavor::Ptr:
    case LoweredValInfo::Flavor::SwizzledLValue:
    case LoweredValInfo::Flavor::SwizzledMatrixLValue:
    case LoweredValInfo::Flavor::BoundStorage:
    case LoweredValInfo::Flavor::BoundMember:
    case LoweredValInfo::Flavor::ExtractedExistential:
        args.add(getSimpleVal(context, argVal));
        break;

    default:
        SLANG_UNIMPLEMENTED_X("addInArg case");
        break;
    }
}

// After a call to a function with `out` or `in out`
// parameters, we may need to copy data back into
// the l-value locations used for output arguments.
//
// During lowering of the argument list, we build
// up a list of these "fixup" assignments that need
// to be performed.
struct OutArgumentFixup
{
    LoweredValInfo dst;
    LoweredValInfo src;
};

/// Apply any fixups that have been created for `out` and `inout` arguments.
static void applyOutArgumentFixups(IRGenContext* context, List<OutArgumentFixup> const& fixups)
{
    for (auto fixup : fixups)
    {
        assign(context, fixup.dst, fixup.src);
    }
}

/// Add one argument value to the argument list for a call being constructed
void addArg(
    IRGenContext* context,
    List<IRInst*>* ioArgs,             //< The argument list being built
    List<OutArgumentFixup>* ioFixups,  //< "Fixup" logic to apply for `out` or `inout` arguments
    LoweredValInfo argVal,             //< The lowered value of the argument to add
    IRType* paramType,                 //< The type of the corresponding parameter
    ParameterDirection paramDirection, //< The direction of the parameter (`in`, `out`, etc.)
    Type* argType,                     //< The AST-level type of the argument
    SourceLoc loc)                     //< A location to use if we need to report an error
{
    switch (paramDirection)
    {
    case kParameterDirection_Ref:
        {
            // According to our "calling convention" we need to
            // pass a pointer into the callee. Unlike the case for
            // `out` and `inout` below, it is never valid to do
            // copy-in/copy-out for a `ref` parameter, so we just
            // pass in the actual pointer.
            //
            IRInst* argPtr = getAddress(context, argVal, loc);
            if (argPtr)
                addInArg(context, ioArgs, LoweredValInfo::simple(argPtr));
            else
            {
                // If arg can't be converted to a pointer, we have already
                // reported an error, so just pass a null pointer to allow
                // the remaining lowering steps to finish.
                addInArg(
                    context,
                    ioArgs,
                    LoweredValInfo::simple(context->irBuilder->getNullVoidPtrValue()));
            }
        }
        break;

    case kParameterDirection_Out:
    case kParameterDirection_InOut:
    case kParameterDirection_ConstRef:
        {
            // According to our "calling convention" we need to
            // pass a pointer into the callee.
            //
            // Ideally we would like to just pass the address of
            // `loweredArg`, and when that it possible we will do so.
            // It may happen, though, that `loweredArg` is not an
            // addressable l-value (e.g., it is `foo.xyz`, so that
            // the bytes of the l-value are not contiguous).
            //
            LoweredValInfo argPtr = tryGetAddress(context, argVal, TryGetAddressMode::Default);
            if (argPtr.flavor == LoweredValInfo::Flavor::Ptr)
            {
                addInArg(context, ioArgs, LoweredValInfo::simple(argPtr.val));
            }
            else
            {
                // If the value is not one that could yield a simple l-value
                // then we need to convert it into a temporary
                //
                if (as<IRThisType>(paramType))
                {
                    // When paramType is ThisType, we need to get the actual argument type
                    // from the arg.
                    paramType = lowerType(context, argType);
                }
                if (auto refType = as<IRConstRefType>(paramType))
                {
                    paramType = refType->getValueType();
                    argVal = LoweredValInfo::simple(
                        context->irBuilder->emitLoad(getSimpleVal(context, argPtr)));
                }

                LoweredValInfo tempVar = createVar(context, paramType);

                // If the parameter is `in out` or `inout`, then we need
                // to ensure that we pass in the original value stored
                // in the argument, which we accomplish by assigning
                // from the l-value to our temp.
                //
                if (paramDirection == kParameterDirection_InOut ||
                    paramDirection == kParameterDirection_ConstRef)
                {
                    assign(context, tempVar, argVal);
                }

                // Now we can pass the address of the temporary variable
                // to the callee as the actual argument for the `in out`
                SLANG_ASSERT(tempVar.flavor == LoweredValInfo::Flavor::Ptr);
                IRInst* tempPtr = getAddress(context, tempVar, loc);
                addInArg(context, ioArgs, LoweredValInfo::simple(tempPtr));

                // Finally, after the call we will need
                // to copy in the other direction: from our
                // temp back to the original l-value.
                if (paramDirection != kParameterDirection_ConstRef)
                {
                    OutArgumentFixup fixup;
                    fixup.src = tempVar;
                    fixup.dst = argVal;

                    (*ioFixups).add(fixup);
                }
            }
        }
        break;

    default:
        addInArg(context, ioArgs, argVal);
        break;
    }
}

/// Add argument(s) corresponding to one parameter to a call
///
/// The `argExpr` is the AST-level expression being passed as an argument to the call.
/// The `paramType` and `paramDirection` represent what is known about the receiving
/// parameter of the callee (e.g., if the parameter `in`, `inout`, etc.).
/// The `ioArgs` array receives the IR-level argument(s) that are added for the given
/// argument expression.
/// The `ioFixups` array receives any "fixup" code that needs to be run *after* the
/// call completes (e.g., to move from a scratch variable used for an `inout` argument back
/// into the original location).
///
void addCallArgsForParam(
    IRGenContext* context,
    IRType* paramType,
    ParameterDirection paramDirection,
    Expr* argExpr,
    List<IRInst*>* ioArgs,
    List<OutArgumentFixup>* ioFixups)
{
    switch (paramDirection)
    {
    case kParameterDirection_Ref:
    case kParameterDirection_ConstRef:
    case kParameterDirection_Out:
    case kParameterDirection_InOut:
        {
            LoweredValInfo loweredArg = lowerLValueExpr(context, argExpr);
            addArg(
                context,
                ioArgs,
                ioFixups,
                loweredArg,
                paramType,
                paramDirection,
                argExpr->type,
                argExpr->loc);
        }
        break;

    default:
        {
            LoweredValInfo loweredArg = lowerRValueExpr(context, argExpr);
            addInArg(context, ioArgs, loweredArg);
        }
        break;
    }
}


//

/// Compute the direction for a parameter based on its declaration
ParameterDirection getParameterDirection(VarDeclBase* paramDecl)
{
    if (paramDecl->hasModifier<RefModifier>())
    {
        return kParameterDirection_Ref;
    }
    if (paramDecl->hasModifier<ConstRefModifier>() || paramDecl->hasModifier<HLSLPayloadModifier>())
    {
        // The payload types are a groupshared variable, and we really don't
        // want to copy that into registers in every invocation on platforms
        // where this matters, so treat them as by-reference here.

        return kParameterDirection_ConstRef;
    }
    if (paramDecl->hasModifier<InOutModifier>())
    {
        // The AST specified `inout`:
        return kParameterDirection_InOut;
    }
    if (paramDecl->hasModifier<OutModifier>())
    {
        // We saw an `out` modifier, so now we need
        // to check if there was a paired `in`.
        if (paramDecl->hasModifier<InModifier>())
            return kParameterDirection_InOut;
        else
            return kParameterDirection_Out;
    }
    else
    {
        // No direction modifier, or just `in`:
        return kParameterDirection_In;
    }
}

/// Compute the direction for a `this` parameter based on the declaration of its parent function
///
/// If the given declaration doesn't care about the direction of a `this` parameter, then
/// it will return the provided `defaultDirection` instead.
///
ParameterDirection getThisParamDirection(Decl* parentDecl, ParameterDirection defaultDirection)
{
    auto parentParent = getParentAggTypeDecl(parentDecl);

    // The `this` parameter for a `class` is always `in`.
    if (as<ClassDecl>(parentParent))
    {
        return kParameterDirection_In;
    }

    if (parentParent && parentParent->findModifier<NonCopyableTypeAttribute>())
    {
        if (parentDecl->hasModifier<MutatingAttribute>())
            return kParameterDirection_Ref;
        else
            return kParameterDirection_ConstRef;
    }

    // Applications can opt in to a mutable `this` parameter,
    // by applying the `[mutating]` attribute to their
    // declaration.
    //
    if (parentDecl->hasModifier<MutatingAttribute>())
    {
        return kParameterDirection_InOut;
    }
    else if (parentDecl->hasModifier<ConstRefAttribute>())
    {
        return kParameterDirection_ConstRef;
    }
    else if (parentDecl->hasModifier<RefAttribute>())
    {
        return kParameterDirection_Ref;
    }

    // A `set` accessor on a property or subscript declaration
    // defaults to a mutable `this` parameter, but the programmer
    // can opt out of this behavior using `[nonmutating]`
    //
    if (parentDecl->hasModifier<NonmutatingAttribute>())
    {
        return kParameterDirection_In;
    }
    else if (as<SetterDecl>(parentDecl))
    {
        return kParameterDirection_InOut;
    }

    // Declarations that represent abstract storage (a property
    // or subscript) do not want to dictate anything about
    // the direction of an outer `this` parameter, since that
    // should be determined by their inner accessors.
    //
    if (as<PropertyDecl>(parentDecl))
    {
        return defaultDirection;
    }
    if (as<SubscriptDecl>(parentDecl))
    {
        return defaultDirection;
    }

    // A parent generic declaration should not change the
    // mutating-ness of the inner declaration.
    //
    if (as<GenericDecl>(parentDecl))
    {
        return defaultDirection;
    }

    // For now we make any `this` parameter default to `in`.
    //
    return kParameterDirection_In;
}

DeclRef<Decl> createDefaultSpecializedDeclRefImpl(
    IRGenContext* context,
    SemanticsVisitor* semantics,
    Decl* decl)
{
    DeclRef<Decl> declRef =
        createDefaultSubstitutionsIfNeeded(context->astBuilder, semantics, makeDeclRef(decl));
    return declRef;
}
//
// The client should actually call the templated wrapper, to preserve type information.
template<typename D>
DeclRef<D> createDefaultSpecializedDeclRef(
    IRGenContext* context,
    SemanticsVisitor* semantics,
    D* decl)
{
    DeclRef<Decl> declRef = createDefaultSpecializedDeclRefImpl(context, semantics, decl);
    return declRef.as<D>();
}

static Type* _findReplacementThisParamType(IRGenContext* context, DeclRef<Decl> parentDeclRef)
{
    if (auto extensionDeclRef = parentDeclRef.as<ExtensionDecl>())
    {
        auto targetType = getTargetType(context->astBuilder, extensionDeclRef);
        if (auto targetDeclRefType = as<DeclRefType>(targetType))
        {
            if (auto replacementType =
                    _findReplacementThisParamType(context, targetDeclRefType->getDeclRef()))
                return replacementType;
        }
        return targetType;
    }

    if (auto interfaceDeclRef = parentDeclRef.as<InterfaceDecl>())
    {
        auto thisType = DeclRefType::create(
            context->astBuilder,
            context->astBuilder->getMemberDeclRef(
                interfaceDeclRef,
                interfaceDeclRef.getDecl()->getThisTypeDecl()));
        return thisType;
    }

    return nullptr;
}

/// Get the type of the `this` parameter introduced by `parentDeclRef`, or null.
///
/// E.g., if `parentDeclRef` is a `struct` declaration, then this will
/// return the type of that `struct`.
///
/// If this function is called on a declaration that does not itself directly
/// introduce a notion of `this`, then null will be returned. Note that this
/// includes things like function declarations themselves, which inherit the
/// definition of `this` from their parent/outer declaration.
///
Type* getThisParamTypeForContainer(IRGenContext* context, DeclRef<Decl> parentDeclRef)
{
    if (auto replacementType = _findReplacementThisParamType(context, parentDeclRef))
        return replacementType;

    if (auto aggTypeDeclRef = parentDeclRef.as<AggTypeDecl>())
    {
        return DeclRefType::create(context->astBuilder, aggTypeDeclRef);
    }

    return nullptr;
}

Type* getThisParamTypeForCallable(IRGenContext* context, DeclRef<Decl> callableDeclRef)
{
    if (auto lookup = as<LookupDeclRef>((callableDeclRef.declRefBase)))
    {
        return lookup->getLookupSource();
    }

    auto parentDeclRef = callableDeclRef.getParent();

    if (auto subscriptDeclRef = parentDeclRef.as<SubscriptDecl>())
        parentDeclRef = subscriptDeclRef.getParent();

    if (auto genericDeclRef = parentDeclRef.as<GenericDecl>())
        parentDeclRef = genericDeclRef.getParent();

    return getThisParamTypeForContainer(context, parentDeclRef);
}

struct StmtLoweringVisitor;

void maybeEmitDebugLine(
    IRGenContext* context,
    StmtLoweringVisitor& visitor,
    Stmt* stmt,
    SourceLoc loc = SourceLoc());

// When lowering something callable (most commonly a function declaration),
// we need to construct an appropriate parameter list for the IR function
// that folds in any contributions from both the declaration itself *and*
// its parent declaration(s).
//
// For example, given code like:
//
//     struct Foo { int bar(float y) { ... } };
//
// we need to generate IR-level code something like:
//
//     func Foo_bar(Foo this, float y) -> int;
//
// that is, the `this` parameter has become explicit.
//
// The same applies to generic parameters, and these
// should apply even if the nested declaration is `static`:
//
//     struct Foo<T> { static int bar(T y) { ... } };
//
// becomes:
//
//     func Foo_bar<T>(T y) -> int;
//
// In order to implement this, we are going to do a recursive
// walk over a declaration and its parents, collecting separate
// lists of ordinary and generic parameters that will need
// to be included in the final declaration's parameter list.
//
// When doing code generation for an ordinary value parameter,
// we mostly care about its type, and then also its "direction"
// (`in`, `out`, `in out`). We sometimes need acess to the
// original declaration so that we can inspect it for meta-data,
// but in some cases there is no such declaration (e.g., a `this`
// parameter doesn't get an explicit declaration in the AST).
// To handle this we break out the relevant data into derived
// structures:
//
struct IRLoweringParameterInfo
{
    // This AST-level type of the parameter
    Type* type = nullptr;

    // The direction (`in` vs `out` vs `in out`)
    ParameterDirection direction;

    // The direction declared in user code.
    ParameterDirection declaredDirection = ParameterDirection::kParameterDirection_In;

    // The variable/parameter declaration for
    // this parameter (if any)
    VarDeclBase* decl = nullptr;

    // Is this the representation of a `this` parameter?
    bool isThisParam = false;

    // Is this the destination of address for non-copyable return val?
    bool isReturnDestination = false;
};
//
// We need a way to be able to create a `IRLoweringParameterInfo` given the declaration
// of a parameter:
//
IRLoweringParameterInfo getParameterInfo(
    IRGenContext* context,
    DeclRef<VarDeclBase> const& paramDecl)
{
    IRLoweringParameterInfo info;

    info.type = getParamType(context->astBuilder, paramDecl);
    info.decl = paramDecl.getDecl();
    info.direction = getParameterDirection(paramDecl.getDecl());
    info.declaredDirection = info.direction;
    info.isThisParam = false;
    return info;
}
//

// Here's the declaration for the type to hold the lists:
struct ParameterLists
{
    List<IRLoweringParameterInfo> params;
};
//
// Because there might be a `static` declaration somewhere
// along the lines, we need to be careful to prohibit adding
// non-generic parameters in some cases.
enum ParameterListCollectMode
{
    // Collect everything: ordinary and generic parameters.
    kParameterListCollectMode_Default,


    // Only collect generic parameters.
    kParameterListCollectMode_Static,
};
//
// We also need to be able to detect whether a declaration is
// either explicitly or implicitly treated as `static`:
ParameterListCollectMode getModeForCollectingParentParameters(Decl* decl, ContainerDecl* parentDecl)
{
    // If we have a `static` parameter, then it is obvious
    // that we should use the `static` mode
    if (isEffectivelyStatic(decl, parentDecl))
        return kParameterListCollectMode_Static;

    // Otherwise, let's default to collecting everything
    return kParameterListCollectMode_Default;
}
//
// When dealing with a member function, we need to be able to add the `this`
// parameter for the enclosing type:
//
void addThisParameter(ParameterDirection direction, Type* type, ParameterLists* ioParameterLists)
{
    IRLoweringParameterInfo info;
    info.type = type;
    info.decl = nullptr;
    info.direction = direction;
    info.declaredDirection = direction;
    info.isThisParam = true;

    ioParameterLists->params.add(info);
}

void maybeAddReturnDestinationParam(ParameterLists* ioParameterLists, Type* resultType)
{
    if (isNonCopyableType(resultType))
    {
        IRLoweringParameterInfo info;
        info.type = resultType;
        info.decl = nullptr;
        info.direction = kParameterDirection_Ref;
        info.declaredDirection = info.direction;
        info.isReturnDestination = true;
        ioParameterLists->params.add(info);
    }
}

void makeVaryingInputParamConstRef(IRLoweringParameterInfo& paramInfo)
{
    if (paramInfo.direction != kParameterDirection_In)
        return;
    if (paramInfo.decl->findModifier<HLSLUniformModifier>())
        return;
    if (as<HLSLPatchType>(paramInfo.type))
        return;
    paramInfo.direction = kParameterDirection_ConstRef;
}
//
// And here is our function that will do the recursive walk:
void collectParameterLists(
    IRGenContext* context,
    DeclRef<Decl> const& declRef,
    ParameterLists* ioParameterLists,
    ParameterListCollectMode mode,
    ParameterDirection thisParamDirection)
{
    // The parameters introduced by any "parent" declarations
    // will need to come first, so we'll deal with that
    // logic here.
    if (auto parentDeclRef = declRef.getParent())
    {
        // Compute the mode to use when collecting parameters from
        // the outer declaration. The most important question here
        // is whether parameters of the outer declaration should
        // also count as parameters of the inner declaration.
        ParameterListCollectMode innerMode =
            getModeForCollectingParentParameters(declRef.getDecl(), parentDeclRef.getDecl());

        // Don't down-grade our `static`-ness along the chain.
        if (innerMode < mode)
            innerMode = mode;

        ParameterDirection innerThisParamDirection =
            getThisParamDirection(declRef.getDecl(), thisParamDirection);


        // Now collect any parameters from the parent declaration itself
        collectParameterLists(
            context,
            parentDeclRef,
            ioParameterLists,
            innerMode,
            innerThisParamDirection);

        // We also need to consider whether the inner declaration needs to have a `this`
        // parameter corresponding to the outer declaration.
        if (innerMode != kParameterListCollectMode_Static)
        {
            auto thisType = getThisParamTypeForContainer(context, parentDeclRef);
            if (thisType)
            {
                if (declRef.getDecl()->findModifier<NoDiffThisAttribute>())
                {
                    auto noDiffAttr = context->astBuilder->getNoDiffModifierVal();
                    thisType = context->astBuilder->getModifiedType(thisType, 1, &noDiffAttr);
                }
                else if (auto fwdDerivDeclRef = declRef.as<ForwardDerivativeRequirementDecl>())
                {
                    thisType = fwdDerivDeclRef.getDecl()->diffThisType;
                }
                else if (auto bwdDerivDeclRef = declRef.as<BackwardDerivativeRequirementDecl>())
                {
                    thisType = bwdDerivDeclRef.getDecl()->diffThisType;
                    innerThisParamDirection = kParameterDirection_InOut;
                }

                addThisParameter(innerThisParamDirection, thisType, ioParameterLists);
            }
        }
    }

    // Once we've added any parameters based on parent declarations,
    // we can see if this declaration itself introduces parameters.
    //
    if (auto callableDeclRef = declRef.as<CallableDecl>())
    {
        // We need a special case here when lowering the varying parameters of an entrypoint
        // function. Due to the existence of `EvaluateAttributeAtSample` and friends, we need to
        // always lower the varying inputs as `__constref` parameters so we can pass pointers to
        // these intrinsics.
        // This means that although these parameters are declared as "in" parameters in the source,
        // we will actually treat them as __constref parameters when lowering to IR. A complication
        // result from this is that if the original source code actually modifies the input
        // parameter we still need to create a local var to hold the modified value. In the future
        // when we are able to update our language spec to always assume input parameters are
        // immutable, then we can remove this adhoc logic of introducing temporary variables. For
        // For now we will rely on a follow up pass to remove unnecessary temporary variables if
        // we can determine that they are never actually writtten to by the user.
        //
        bool lowerVaryingInputAsConstRef = declRef.getDecl()->hasModifier<EntryPointAttribute>() ||
                                           declRef.getDecl()->hasModifier<NumThreadsAttribute>();

        // Don't collect parameters from the outer scope if
        // we are in a `static` context.
        if (mode == kParameterListCollectMode_Default)
        {
            for (auto paramDeclRef : getParameters(context->astBuilder, callableDeclRef))
            {
                auto paramInfo = getParameterInfo(context, paramDeclRef);
                if (lowerVaryingInputAsConstRef)
                    makeVaryingInputParamConstRef(paramInfo);
                ioParameterLists->params.add(paramInfo);
            }
            maybeAddReturnDestinationParam(
                ioParameterLists,
                getResultType(context->astBuilder, callableDeclRef));
        }
    }
}

bool isConstExprVar(Decl* decl)
{
    if (decl->hasModifier<ConstExprModifier>())
    {
        return true;
    }
    else if (decl->hasModifier<HLSLStaticModifier>() && decl->hasModifier<ConstModifier>())
    {
        return true;
    }

    return false;
}


IRType* maybeGetConstExprType(IRBuilder* builder, IRType* type, Decl* decl)
{
    if (isConstExprVar(decl))
    {
        return builder->getRateQualifiedType(builder->getConstExprRate(), type);
    }

    return type;
}


struct FuncDeclBaseTypeInfo
{
    IRType* type;
    IRType* resultType;
    ParameterLists parameterLists;
    List<IRType*> paramTypes;
    // If the function returns a non-copyable value, this
    // flag is set to indicate that the result should be
    // returned via the last ref parameter.
    bool returnViaLastRefParam = false;
};

void _lowerFuncDeclBaseTypeInfo(
    IRGenContext* context,
    DeclRef<FunctionDeclBase> declRef,
    FuncDeclBaseTypeInfo& outInfo)
{
    auto builder = context->irBuilder;

    // Collect the parameter lists we will use for our new function.
    auto& parameterLists = outInfo.parameterLists;
    collectParameterLists(
        context,

        declRef,
        &parameterLists,
        kParameterListCollectMode_Default,
        kParameterDirection_In);

    auto& paramTypes = outInfo.paramTypes;

    for (auto paramInfo : parameterLists.params)
    {
        IRType* irParamType = lowerType(context, paramInfo.type);

        switch (paramInfo.direction)
        {
        case kParameterDirection_In:
            // Simple case of a by-value input parameter.
            break;

        // If the parameter is declared `out` or `inout`,
        // then we will represent it with a pointer type in
        // the IR, but we will use a specialized pointer
        // type that encodes the parameter direction information.
        case kParameterDirection_Out:
            irParamType = builder->getOutType(irParamType);
            break;
        case kParameterDirection_InOut:
            irParamType = builder->getInOutType(irParamType);
            break;
        case kParameterDirection_Ref:
            irParamType = builder->getRefType(irParamType, AddressSpace::Generic);
            break;
        case kParameterDirection_ConstRef:
            irParamType = builder->getConstRefType(irParamType);
            break;
        default:
            SLANG_UNEXPECTED("unknown parameter direction");
            break;
        }

        // If the parameter was explicitly marked as being a compile-time
        // constant (`constexpr`), then attach that information to its
        // IR-level type explicitly.
        if (paramInfo.decl)
        {
            irParamType = maybeGetConstExprType(builder, irParamType, paramInfo.decl);
        }

        if (paramInfo.decl && paramInfo.decl->hasModifier<HLSLGroupSharedModifier>())
        {
            irParamType = builder->getRateQualifiedType(builder->getGroupSharedRate(), irParamType);
        }

        // The 'payload' parameter is a read-only groupshared value
        if (paramInfo.decl && paramInfo.decl->hasModifier<HLSLPayloadModifier>())
        {
            irParamType = builder->getRateQualifiedType(builder->getGroupSharedRate(), irParamType);
        }

        paramTypes.add(irParamType);
    }

    auto& irResultType = outInfo.resultType;

    if (parameterLists.params.getCount() && parameterLists.params.getLast().isReturnDestination)
    {
        irResultType = context->irBuilder->getVoidType();
        outInfo.returnViaLastRefParam = true;
    }
    else
    {
        irResultType = lowerType(context, getResultType(context->astBuilder, declRef));


        if (auto setterDeclRef = declRef.as<SetterDecl>())
        {
            // A `set` accessor always returns `void`
            //
            // TODO: We should handle this by making the result
            // type of a `set` accessor be represented accurately
            // at the AST level (ditto for the `ref` case below).
            //
            irResultType = builder->getVoidType();
        }

        if (auto refAccessorDeclRef = declRef.as<RefAccessorDecl>())
        {
            // A `ref` accessor needs to return a *pointer* to the value
            // being accessed, rather than a simple value.
            irResultType = builder->getPtrType(irResultType);
        }
    }

    if (!getErrorCodeType(context->astBuilder, declRef)
             ->equals(context->astBuilder->getBottomType()))
    {
        auto errorType = lowerType(context, getErrorCodeType(context->astBuilder, declRef));
        IRAttr* throwTypeAttr = nullptr;
        throwTypeAttr = builder->getAttr(kIROp_FuncThrowTypeAttr, 1, (IRInst**)&errorType);
        outInfo.type = builder->getFuncType(
            paramTypes.getCount(),
            paramTypes.getBuffer(),
            irResultType,
            throwTypeAttr);
    }
    else
    {
        outInfo.type =
            builder->getFuncType(paramTypes.getCount(), paramTypes.getBuffer(), irResultType);
    }
}

static LoweredValInfo _emitCallToAccessor(
    IRGenContext* context,
    IRType* type,
    DeclRef<AccessorDecl> accessorDeclRef,
    LoweredValInfo base,
    UInt argCount,
    IRInst* const* args)
{
    FuncDeclBaseTypeInfo info;
    _lowerFuncDeclBaseTypeInfo(context, accessorDeclRef, info);

    List<IRInst*> allArgs;

    List<OutArgumentFixup> fixups;
    if (base.flavor != LoweredValInfo::Flavor::None)
    {
        SLANG_ASSERT(info.parameterLists.params.getCount() >= 1);
        SLANG_ASSERT(info.parameterLists.params[0].isThisParam);

        auto thisParam = info.parameterLists.params[0];
        auto thisParamType = lowerType(context, thisParam.type);

        addArg(
            context,
            &allArgs,
            &fixups,
            base,
            thisParamType,
            thisParam.direction,
            thisParam.type,
            SourceLoc());
    }

    allArgs.addRange(args, argCount);

    LoweredValInfo result = emitCallToDeclRef(
        context,
        type,
        accessorDeclRef,
        info.type,
        allArgs.getCount(),
        allArgs.getBuffer(),
        TryClauseEnvironment());

    applyOutArgumentFixups(context, fixups);

    return result;
}

template<typename Derived>
struct ExprLoweringContext
{
    static bool isLValueContext() { return Derived::_isLValueContext(); }

    IRGenContext* context;

    IRBuilder* getBuilder() { return context->irBuilder; }
    ASTBuilder* getASTBuilder() { return context->astBuilder; }


    struct ResolvedCallInfo
    {
        DeclRef<Decl> funcDeclRef;
        Expr* baseExpr = nullptr;
    };

    // Try to resolve a the function expression for a call
    // into a reference to a specific declaration, along
    // with some contextual information about the declaration
    // we are calling.
    bool tryResolveDeclRefForCall(Expr* funcExpr, ResolvedCallInfo* outInfo)
    {
        // TODO: unwrap any "identity" expressions that might
        // be wrapping the callee.

        // First look to see if the expression references a
        // declaration at all.
        auto declRefExpr = as<DeclRefExpr>(funcExpr);
        if (!declRefExpr)
            return false;

        // A little bit of future proofing here: if we ever
        // allow higher-order functions, then we might be
        // calling through a variable/field that has a function
        // type, but is not itself a function.
        // In such a case we should be careful to not statically
        // resolve things.
        //
        if (auto callableDecl = as<CallableDecl>(declRefExpr->declRef.getDecl()))
        {
            // Okay, the declaration is directly callable, so we can continue.
        }
        else
        {
            // The callee declaration isn't itself a callable (it must have
            // a function type, though).
            return false;
        }

        // Now we can look at the specific kinds of declaration references,
        // and try to tease them apart.
        if (auto memberFuncExpr = as<MemberExpr>(funcExpr))
        {
            outInfo->funcDeclRef = memberFuncExpr->declRef;
            outInfo->baseExpr = memberFuncExpr->baseExpression;
            return true;
        }
        else if (auto staticMemberFuncExpr = as<StaticMemberExpr>(funcExpr))
        {
            outInfo->funcDeclRef = staticMemberFuncExpr->declRef;
            return true;
        }
        else if (auto varExpr = as<VarExpr>(funcExpr))
        {
            outInfo->funcDeclRef = varExpr->declRef;
            return true;
        }
        else
        {
            // Seems to be a case of declaration-reference we don't know about.
            SLANG_UNEXPECTED("unknown declaration reference kind");
            // return false;
        }
    }

    /// Return `expr` with any outer casts to interface types stripped away
    Expr* maybeIgnoreCastToInterface(Expr* expr)
    {
        auto e = expr;
        while (auto castExpr = as<CastToSuperTypeExpr>(e))
        {
            if (auto declRefType = as<DeclRefType>(e->type))
            {
                if (declRefType->getDeclRef().as<InterfaceDecl>())
                {
                    e = castExpr->valueArg;
                    continue;
                }
            }
            else if (auto andType = as<AndType>(e->type))
            {
                // TODO: We might eventually need to tell the difference
                // between conjunctions of interfaces and conjunctions
                // that might include non-interface types.
                //
                // For now we assume that any case to a conjunction
                // is effectively a cast to an interface type.
                //
                e = castExpr->valueArg;
                continue;
            }
            break;
        }
        return e;
    }


    // Lower an expression that should have the same l-value-ness
    // as the visitor itself.
    LoweredValInfo lowerSubExpr(Expr* expr)
    {
        IRBuilderSourceLocRAII sourceLocInfo(getBuilder(), expr->loc);
        if (isLValueContext())
            return lowerLValueExpr(context, expr);
        return lowerRValueExpr(context, expr);
    }

    /// Create IR instructions for an argument at a call site, based on
    /// AST-level expressions plus function signature information.
    ///
    /// The `funcType` parameter is always required, and specifies the types
    /// of all the parameters. The `funcDeclRef` parameter is only required
    /// if there are parameter positions for which the matching argument is
    /// absent.
    ///
    void addDirectCallArgs(
        InvokeExpr* expr,
        Index argIndex,
        IRType* paramType,
        ParameterDirection paramDirection,
        DeclRef<ParamDecl> paramDeclRef,
        List<IRInst*>* ioArgs,
        List<OutArgumentFixup>* ioFixups)
    {
        Count argCount = expr->arguments.getCount();
        if (argIndex < argCount)
        {
            auto argExpr = expr->arguments[argIndex];
            addCallArgsForParam(context, paramType, paramDirection, argExpr, ioArgs, ioFixups);
        }
        else
        {
            // We have run out of arguments supplied at the call site,
            // but there are still parameters remaining. This must mean
            // that these parameters have default argument expressions
            // associated with them.
            //
            // Currently we simply extract the initial-value expression
            // from the parameter declaration and then lower it in
            // the context of the caller.
            //
            // Note that the expression could involve subsitutions because
            // in the general case it could depend on the generic parameters
            // used the specialize the callee. For now we do not handle that
            // case, and simply ignore generic arguments.
            //
            SubstExpr<Expr> argExpr = getInitExpr(getASTBuilder(), paramDeclRef);
            SLANG_ASSERT(argExpr);

            IRGenEnv subEnvStorage;
            IRGenEnv* subEnv = &subEnvStorage;
            subEnv->outer = context->env;

            IRGenContext subContextStorage = *context;
            IRGenContext* subContext = &subContextStorage;
            subContext->env = subEnv;

            _lowerSubstitutionEnv(
                subContext,
                argExpr.getSubsts() ? argExpr.getSubsts().declRef : nullptr);

            addCallArgsForParam(
                subContext,
                paramType,
                paramDirection,
                argExpr.getExpr(),
                ioArgs,
                ioFixups);

            // TODO: The approach we are taking here to default arguments
            // is simplistic, and has consequences for the front-end as
            // well as binary serialization of modules.
            //
            // We could consider some more refined approaches where, e.g.,
            // functions with default arguments generate multiple IR-level
            // functions, that compute and provide the default values.
            //
            // Alternatively, each parameter with defaults could be generated
            // into its own callable function that provides the default value,
            // so that calling modules can call into a pre-generated function.
            //
            // Each of these options involves trade-offs, and we need to
            // make a conscious decision at some point.

            // Assert that such an expression must have been present.
        }
    }

    void addDirectCallArgs(
        InvokeExpr* expr,
        FuncType* funcType,
        List<IRInst*>* ioArgs,
        List<OutArgumentFixup>* ioFixups)
    {
        Count argCount = expr->arguments.getCount();
        SLANG_ASSERT(argCount == funcType->getParamCount());

        for (Index i = 0; i < argCount; ++i)
        {
            IRType* paramType = lowerType(context, funcType->getParamType(i));
            ParameterDirection paramDirection = funcType->getParamDirection(i);
            addDirectCallArgs(
                expr,
                i,
                paramType,
                paramDirection,
                DeclRef<ParamDecl>(),
                ioArgs,
                ioFixups);
        }
    }

    void addDirectCallArgs(
        InvokeExpr* expr,
        DeclRef<CallableDecl> funcDeclRef,
        List<IRInst*>* ioArgs,
        List<OutArgumentFixup>* ioFixups)
    {
        Count argCounter = 0;
        for (auto paramDeclRef : getMembersOfType<ParamDecl>(getASTBuilder(), funcDeclRef))
        {
            auto paramDecl = paramDeclRef.getDecl();
            IRType* paramType = lowerType(context, getType(getASTBuilder(), paramDeclRef));
            auto paramDirection = getParameterDirection(paramDecl);

            Index argIndex = argCounter++;
            addDirectCallArgs(
                expr,
                argIndex,
                paramType,
                paramDirection,
                paramDeclRef,
                ioArgs,
                ioFixups);
        }
    }

    // Add arguments that appeared directly in an argument list
    // to the list of argument values for a call.
    void addDirectCallArgs(
        InvokeExpr* expr,
        DeclRef<Decl> funcDeclRef,
        List<IRInst*>* ioArgs,
        List<OutArgumentFixup>* ioFixups)
    {
        if (auto callableDeclRef = funcDeclRef.as<CallableDecl>())
        {
            addDirectCallArgs(expr, callableDeclRef, ioArgs, ioFixups);
        }
        else
        {
            SLANG_UNEXPECTED("callee was not a callable decl");
        }
    }

    void addFuncBaseArgs(LoweredValInfo funcVal, List<IRInst*>* /*ioArgs*/)
    {
        switch (funcVal.flavor)
        {
        default:
            return;
        }
    }


    void _lowerSubstitutionArg(
        IRGenContext* subContext,
        GenericAppDeclRef* subst,
        Decl* paramDecl,
        Index argIndex)
    {
        SLANG_ASSERT(argIndex < subst->getArgs().getCount());
        auto argVal = lowerVal(subContext, subst->getArgs()[argIndex]);
        subContext->setValue(paramDecl, argVal);
    }

    void _lowerSubstitutionEnv(IRGenContext* subContext, DeclRefBase* subst)
    {
        if (!subst)
            return;
        _lowerSubstitutionEnv(subContext, subst->getBase());

        if (auto genSubst = as<GenericAppDeclRef>(subst))
        {
            auto genDecl = genSubst->getGenericDecl();

            Index argCounter = 0;
            for (auto memberDecl : genDecl->members)
            {
                if (auto typeParamDecl = as<GenericTypeParamDecl>(memberDecl))
                {
                    _lowerSubstitutionArg(subContext, genSubst, typeParamDecl, argCounter++);
                }
                else if (auto valParamDecl = as<GenericValueParamDecl>(memberDecl))
                {
                    _lowerSubstitutionArg(subContext, genSubst, valParamDecl, argCounter++);
                }
            }
            for (auto memberDecl : genDecl->members)
            {
                if (auto constraintDecl = as<GenericTypeConstraintDecl>(memberDecl))
                {
                    _lowerSubstitutionArg(subContext, genSubst, constraintDecl, argCounter++);
                }
            }
        }
        // TODO: also need to handle this-type substitution here?
    }

    void validateInvokeExprArgsWithFunctionModifiers(
        InvokeExpr* expr,
        FunctionDeclBase* decl,
        List<IRInst*>& irArgs)
    {
        if (auto glslRequireShaderInputParameter =
                decl->findModifier<GLSLRequireShaderInputParameterAttribute>())
        {
            if (!irArgs[glslRequireShaderInputParameter->parameterNumber]
                     ->findDecoration<IRGlobalInputDecoration>())
            {
                this->context->getSink()->diagnose(
                    expr,
                    Diagnostics::requireInputDecoratedVarForParameter,
                    decl,
                    glslRequireShaderInputParameter->parameterNumber);
            }
            return;
        }
    }

    /// Lower an invoke expr, and attempt to fuse a store of the expr's result into destination.
    /// If the store is fused, returns LoweredValInfo::None. Otherwise, returns the IR val
    /// representing the RValue.
    LoweredValInfo visitInvokeExprImpl(
        InvokeExpr* expr,
        LoweredValInfo destination,
        const TryClauseEnvironment& tryEnv)
    {
        auto type = lowerType(context, expr->type);

        // We are going to look at the syntactic form of
        // the "function" expression, so that we can avoid
        // a lot of complexity that would come from lowering
        // it as a general expression first, and then trying
        // to apply it. For example, given `obj.f(a,b)` we
        // will try to detect that we are trying to compute
        // something like `ObjType::f(obj, a, b)` (in pseudo-code),
        // rather than trying to construct a meaningful
        // intermediate value for `obj.f` first.
        //
        // Note that this doe not preclude having support
        // for directly generating code from `obj.f` - it
        // just may be that such usage is more complicated.

        // Along the way, we may end up collecting additional
        // arguments that will be part of the call.
        List<IRInst*> irArgs;

        // We will also collect "fixup" actions that need
        // to be performed after the call, in order to
        // copy the final values for `out` parameters
        // back to their arguments.
        List<OutArgumentFixup> argFixups;

        auto funcExpr = expr->functionExpr;
        ResolvedCallInfo resolvedInfo;
        if (tryResolveDeclRefForCall(funcExpr, &resolvedInfo))
        {
            // In this case we know exactly what declaration we
            // are going to call, and so we can resolve things
            // appropriately.
            auto funcDeclRef = resolvedInfo.funcDeclRef;
            auto baseExpr = resolvedInfo.baseExpr;
            if (baseExpr)
            {
                // The base expression might be an "upcast" to a base interface, in
                // which case we don't want to emit the result of the cast, but instead
                // the source.
                //
                baseExpr = this->maybeIgnoreCastToInterface(baseExpr);
            }

            // If the thing being invoked is a subscript operation,
            // then we need to handle multiple extra details
            // that don't arise for other kinds of calls.
            //
            // TODO: subscript operations probably deserve to
            // be handled on their own path for this reason...
            //
            if (auto subscriptDeclRef = funcDeclRef.template as<SubscriptDecl>())
            {
                // A reference to a subscript declaration is a special case,
                // because it is not possible to call a subscript directly;
                // we must call one of its accessors.
                //
                auto loweredBase = lowerSubExpr(baseExpr);
                addDirectCallArgs(expr, funcDeclRef, &irArgs, &argFixups);
                auto result = lowerStorageReference(
                    context,
                    type,
                    subscriptDeclRef,
                    loweredBase,
                    irArgs.getCount(),
                    irArgs.getBuffer());

                // TODO: Applying the fixups for arguments to the subscript at this point
                // won't technically be correct, since the call to the subscript may
                // not have occured at this point.
                //
                // It seems like we need to either:
                //
                // * Capture the arguments to the subscript as `LoweredValInfo` instead of `IRInst*`
                //   so that we can deal with everything related to fixups around the actual call
                //   site.
                //
                // OR
                //
                // * Handle everything to do with "fixups" differently, by treating them as deferred
                // actions that gert queued up on the context itself and then flushed at certain
                // well-defined points, so that we don't have to be as careful around them.
                //
                // OR
                //
                // * Switch to a more "destination-driven" approach to code generation, where we
                // can determine on entry to the lowering of a sub-expression whether it will be
                // used for read, write, or read/write, and resolve things like the choice of
                // accessor at that point instead.
                //
                applyOutArgumentFixups(context, argFixups);
                return result;
            }

            // First comes the `this` argument if we are calling
            // a member function:
            if (baseExpr)
            {
                auto thisType = getThisParamTypeForCallable(context, funcDeclRef);
                auto irThisType = lowerType(context, thisType);
                addCallArgsForParam(
                    context,
                    irThisType,
                    getThisParamDirection(funcDeclRef.getDecl(), kParameterDirection_In),
                    baseExpr,
                    &irArgs,
                    &argFixups);
            }

            // Then we have the "direct" arguments to the call.
            // These may include `out` and `inout` arguments that
            // require "fixup" work on the other side.
            //
            FuncDeclBaseTypeInfo funcTypeInfo;
            _lowerFuncDeclBaseTypeInfo(
                context,
                funcDeclRef.template as<FunctionDeclBase>(),
                funcTypeInfo);

            auto funcType = funcTypeInfo.type;
            addDirectCallArgs(expr, funcDeclRef, &irArgs, &argFixups);

            validateInvokeExprArgsWithFunctionModifiers(
                expr,
                as<FunctionDeclBase>(funcDeclRef.getDecl()),
                irArgs);

            LoweredValInfo result;
            if (funcTypeInfo.returnViaLastRefParam)
            {
                // If the function returns a non-copyable type, then we need to
                // pass in the destination that receives the result value as an `__ref` parameter.
                //
                if (destination.flavor != LoweredValInfo::Flavor::None)
                {
                    // If we have a known destination, we can use it directly as argument to the
                    // call.
                    irArgs.add(destination.val);
                    result = LoweredValInfo();
                }
                else
                {
                    // Otherwise, we need to create a temporary variable to hold the result.
                    //
                    auto tempVar = context->irBuilder->emitVar(
                        tryGetPointedToType(context->irBuilder, funcTypeInfo.paramTypes.getLast()));
                    irArgs.add(tempVar);
                    result = LoweredValInfo::ptr(tempVar);
                }
            }

            auto callResult =
                emitCallToDeclRef(context, type, funcDeclRef, funcType, irArgs, tryEnv);
            applyOutArgumentFixups(context, argFixups);

            if (funcTypeInfo.returnViaLastRefParam)
                return result;
            return callResult;
        }
        else if (auto funcType = as<FuncType>(expr->functionExpr->type))
        {
            auto funcVal = lowerRValueExpr(context, expr->functionExpr);
            addDirectCallArgs(expr, funcType, &irArgs, &argFixups);

            auto result = emitCallToVal(
                context,
                type,
                funcVal,
                irArgs.getCount(),
                irArgs.getBuffer(),
                tryEnv);

            applyOutArgumentFixups(context, argFixups);
            return result;
        }


        // TODO: In this case we should be emitting code for the callee as
        // an ordinary expression, then emitting the arguments according
        // to the type information on the callee (e.g., which parameters
        // are `out` or `inout`, and then finally emitting the `call`
        // instruction.
        //
        // We don't currently have the case of emitting arguments according
        // to function type info (instead of declaration info), and really
        // this case can't occur unless we start adding first-class functions
        // to the source language.
        //
        // For now we just bail out with an error.
        //
        SLANG_UNEXPECTED("could not resolve target declaration for call");
        UNREACHABLE_RETURN(LoweredValInfo());
    }
};

template<typename Derived>
struct ExprLoweringVisitorBase : public ExprVisitor<Derived, LoweredValInfo>
{
    static bool isLValueContext() { return Derived::_isLValueContext(); }

    ExprLoweringContext<Derived> sharedLoweringContext;

    IRGenContext*& context;

    ExprLoweringVisitorBase()
        : context(sharedLoweringContext.context)
    {
    }

    IRBuilder* getBuilder() { return context->irBuilder; }
    ASTBuilder* getASTBuilder() { return context->astBuilder; }
    LoweredValInfo lowerSubExpr(Expr* expr) { return sharedLoweringContext.lowerSubExpr(expr); }

    LoweredValInfo visitIncompleteExpr(IncompleteExpr*)
    {
        SLANG_UNEXPECTED("a valid ast should not contain an IncompleteExpr.");
        UNREACHABLE_RETURN(LoweredValInfo());
    }

    LoweredValInfo visitVarExpr(VarExpr* expr)
    {
        auto lowerTypeOfExpr = lowerType(context, expr->type);
        auto declRef = expr->declRef;
        if (auto propertyDeclRef = declRef.as<PropertyDecl>())
        {
            // A reference to a property is a special case, because
            // we must translate the reference to the property
            // into a reference to one of its accessors.
            return lowerStorageReference(
                context,
                lowerTypeOfExpr,
                propertyDeclRef,
                LoweredValInfo(),
                0,
                nullptr);
        }
        LoweredValInfo info = emitDeclRef(context, declRef, lowerTypeOfExpr);
        return info;
    }

    // Emit IR to denote the forward-mode derivative
    // of the inner func-expr. This will be resolved
    // to a concrete function during the derivative
    // pass.
    LoweredValInfo visitForwardDifferentiateExpr(ForwardDifferentiateExpr* expr)
    {
        auto baseVal = lowerSubExpr(expr->baseFunction);
        SLANG_ASSERT(baseVal.flavor == LoweredValInfo::Flavor::Simple);

        return LoweredValInfo::simple(getBuilder()->emitForwardDifferentiateInst(
            lowerType(context, expr->type),
            baseVal.val));
    }

    LoweredValInfo visitDetachExpr(DetachExpr* expr)
    {
        auto baseVal = lowerRValueExpr(context, expr->inner);

        return LoweredValInfo::simple(getBuilder()->emitDetachDerivative(
            lowerType(context, expr->type),
            getSimpleVal(context, baseVal)));
    }

    LoweredValInfo visitPrimalSubstituteExpr(PrimalSubstituteExpr* expr)
    {
        auto baseVal = lowerSubExpr(expr->baseFunction);
        SLANG_ASSERT(baseVal.flavor == LoweredValInfo::Flavor::Simple);

        return LoweredValInfo::simple(
            getBuilder()->emitPrimalSubstituteInst(lowerType(context, expr->type), baseVal.val));
    }

    LoweredValInfo visitTreatAsDifferentiableExpr(TreatAsDifferentiableExpr* expr)
    {
        auto baseVal = lowerSubExpr(expr->innerExpr);

        IRInst* innerInst = nullptr;
        if (baseVal.flavor != LoweredValInfo::Flavor::Simple)
        {
            if (!isLValueContext())
            {
                auto materializedVal = materialize(context, baseVal);

                // TODO(Sai): We might be missing the case where a single materialize could create
                // multiple calls (multiple index operations?). Not quite sure what the right way
                // to handle that case might be.
                //
                if (as<IRCall>(materializedVal.val))
                {
                    if (expr->flavor == TreatAsDifferentiableExpr::Flavor::NoDiff)
                        getBuilder()->addDecoration(
                            materializedVal.val,
                            kIROp_TreatCallAsDifferentiableDecoration);
                    else if (expr->flavor == TreatAsDifferentiableExpr::Flavor::Differentiable)
                        getBuilder()->addDecoration(
                            materializedVal.val,
                            kIROp_DifferentiableCallDecoration);
                    else
                        SLANG_UNEXPECTED("Unknown TreatAsDifferentiableExpr::Flavor");
                }

                innerInst = getSimpleVal(context, materializedVal);

                // We'll special case handle 'loads' here in order to allow TreatAsDifferentiable to
                // be used on array index operations. (This is to avoid a discrepancy between using
                // no_diff on local variable indexing vs. resource indexing.)
                //
                if (as<IRLoad>(innerInst))
                    innerInst =
                        getBuilder()->emitDetachDerivative(innerInst->getDataType(), innerInst);
            }
            else
            {
                SLANG_UNEXPECTED(
                    "TreatAsDifferentiableExpr on non-simple l-values not properly defined.");
            }
        }
        else
        {
            if (auto callInst = as<IRCall>(baseVal.val))
                if (expr->flavor == TreatAsDifferentiableExpr::Flavor::NoDiff)
                    getBuilder()->addDecoration(
                        callInst,
                        kIROp_TreatCallAsDifferentiableDecoration);
                else if (expr->flavor == TreatAsDifferentiableExpr::Flavor::Differentiable)
                    getBuilder()->addDecoration(callInst, kIROp_DifferentiableCallDecoration);
                else
                    SLANG_UNEXPECTED("Unknown TreatAsDifferentiableExpr::Flavor");

            innerInst = baseVal.val;
        }

        SLANG_ASSERT(innerInst);

        return LoweredValInfo::simple(innerInst);
    }

    // Emit IR to denote the forward-mode derivative
    // of the inner func-expr. This will be resolved
    // to a concrete function during the derivative
    // pass.
    LoweredValInfo visitBackwardDifferentiateExpr(BackwardDifferentiateExpr* expr)
    {
        auto baseVal = lowerSubExpr(expr->baseFunction);
        SLANG_ASSERT(baseVal.flavor == LoweredValInfo::Flavor::Simple);

        return LoweredValInfo::simple(getBuilder()->emitBackwardDifferentiateInst(
            lowerType(context, expr->type),
            baseVal.val));
    }

    LoweredValInfo visitDispatchKernelExpr(DispatchKernelExpr* expr)
    {
        auto baseVal = lowerSubExpr(expr->baseFunction);
        SLANG_ASSERT(baseVal.flavor == LoweredValInfo::Flavor::Simple);
        auto threadSize = lowerRValueExpr(context, expr->threadGroupSize);
        auto groupSize = lowerRValueExpr(context, expr->dispatchSize);
        // Actual arguments to be filled in when we lower the actual call expr.
        // This is handled in `emitCallToVal`.
        return LoweredValInfo::simple(getBuilder()->emitDispatchKernelInst(
            lowerType(context, expr->type),
            baseVal.val,
            getSimpleVal(context, threadSize),
            getSimpleVal(context, groupSize),
            0,
            nullptr));
    }

    LoweredValInfo visitGetArrayLengthExpr(GetArrayLengthExpr* expr)
    {
        auto type = lowerType(context, expr->arrayExpr->type);
        auto arrayType = as<IRArrayType>(type);
        SLANG_ASSERT(arrayType);
        return LoweredValInfo::simple(arrayType->getElementCount());
    }

    LoweredValInfo visitSizeOfLikeExpr(SizeOfLikeExpr* sizeOfLikeExpr)
    {
        // Lets try and lower to a constant
        ASTNaturalLayoutContext naturalLayoutContext(getASTBuilder(), nullptr);

        const auto size = naturalLayoutContext.calcSize(sizeOfLikeExpr->sizedType);

        auto builder = getBuilder();
        auto resultType = lowerType(context, sizeOfLikeExpr->type);

        if (!size)
        {
            auto sizedType = lowerType(context, sizeOfLikeExpr->sizedType);

            // We can create an inst

            IRInst* inst = nullptr;

            if (as<AlignOfExpr>(sizeOfLikeExpr))
            {
                inst = builder->emitAlignOf(sizedType);
            }
            else if (as<SizeOfExpr>(sizeOfLikeExpr))
            {
                inst = builder->emitSizeOf(sizedType);
            }
            else
            {

                inst = builder->emitCountOf(resultType, sizedType);
            }

            return LoweredValInfo::simple(inst);
        }

        const auto value = as<SizeOfExpr>(sizeOfLikeExpr) ? size.size : size.alignment;

        return LoweredValInfo::simple(getBuilder()->getIntValue(resultType, value));
    }

    LoweredValInfo visitOverloadedExpr(OverloadedExpr* /*expr*/)
    {
        SLANG_UNEXPECTED("overloaded expressions should not occur in checked AST");
        UNREACHABLE_RETURN(LoweredValInfo());
    }

    LoweredValInfo visitOverloadedExpr2(OverloadedExpr2* /*expr*/)
    {
        SLANG_UNEXPECTED("overloaded expressions should not occur in checked AST");
        UNREACHABLE_RETURN(LoweredValInfo());
    }

    LoweredValInfo visitPartiallyAppliedGenericExpr(PartiallyAppliedGenericExpr* /*expr*/)
    {
        SLANG_UNEXPECTED("partially applied generics should not occur in checked AST");
        UNREACHABLE_RETURN(LoweredValInfo());
    }

    LoweredValInfo visitLambdaExpr(LambdaExpr*)
    {
        SLANG_UNEXPECTED("a valid ast should not contain an LambdaExpr.");
        UNREACHABLE_RETURN(LoweredValInfo());
    }

    LoweredValInfo visitSPIRVAsmExpr(SPIRVAsmExpr* expr)
    {
        // Although the surface syntax can have an empty ASM block, the IR asm
        // block must have at least one inst
        if (!expr->insts.getCount())
            return LoweredValInfo{};

        auto builder = context->irBuilder;

        const auto type = lowerType(context, expr->type);
        const auto spirvAsmInst = builder->emitSPIRVAsm(type);

        const auto lowerOperand = [&](const SPIRVAsmOperand& operand) -> IRSPIRVAsmOperand*
        {
            switch (operand.flavor)
            {
            case SPIRVAsmOperand::Literal:
                {
                    if (operand.token.type == TokenType::IntegerLiteral)
                    {
                        // TODO: we should sign-extend these where appropriate,
                        // difficult because it requires information on usage...
                        return builder->emitSPIRVAsmOperandLiteral(
                            builder->getIntValue(builder->getUIntType(), operand.knownValue));
                    }
                    else if (operand.token.type == TokenType::StringLiteral)
                    {
                        const auto v = getStringLiteralTokenValue(operand.token);
                        return builder->emitSPIRVAsmOperandLiteral(
                            builder->getStringValue(v.getUnownedSlice()));
                    }
                    SLANG_UNREACHABLE("Unhandled literal type in visitSPIRVAsmExpr");
                }
            case SPIRVAsmOperand::Id:
                {
                    const auto id = operand.token.getContent();
                    return builder->emitSPIRVAsmOperandId(builder->getStringValue(id));
                }
            case SPIRVAsmOperand::ResultMarker:
                {
                    return builder->emitSPIRVAsmOperandResult();
                }
            case SPIRVAsmOperand::NamedValue:
                {
                    const auto v = operand.knownValue;
                    const auto i = builder->getIntValue(builder->getUIntType(), v);
                    if (operand.wrapInId)
                        return builder->emitSPIRVAsmOperandEnum(i, builder->getUIntType());
                    else
                        return builder->emitSPIRVAsmOperandEnum(i);
                }
            case SPIRVAsmOperand::BuiltinVar:
                {
                    const auto kind = operand.knownValue;
                    auto kindInst = builder->getIntValue(builder->getIntType(), kind);
                    const auto type = lowerType(context, operand.type.type);
                    return builder->emitSPIRVAsmOperandBuiltinVar(type, kindInst);
                }
            case SPIRVAsmOperand::GLSL450Set:
                {
                    return builder->emitSPIRVAsmOperandGLSL450Set();
                }
            case SPIRVAsmOperand::NonSemanticDebugPrintfExtSet:
                {
                    return builder->emitSPIRVAsmOperandDebugPrintfSet();
                }
            case SPIRVAsmOperand::SlangValue:
                {
                    IRInst* i;
                    {
                        IRBuilderInsertLocScope insertScope(builder);
                        builder->setInsertBefore(spirvAsmInst);
                        i = getSimpleVal(context, lowerRValueExpr(context, operand.expr));
                    }
                    return builder->emitSPIRVAsmOperandInst(i);
                }
            case SPIRVAsmOperand::SlangImmediateValue:
                {
                    IRInst* i;
                    {
                        IRBuilderInsertLocScope insertScope(builder);
                        builder->setInsertBefore(spirvAsmInst);
                        i = getSimpleVal(context, lowerRValueExpr(context, operand.expr));
                    }
                    return builder->emitSPIRVAsmOperandEnum(i);
                }
            case SPIRVAsmOperand::SlangValueAddr:
                {
                    IRInst* i;
                    {
                        IRBuilderInsertLocScope insertScope(builder);
                        builder->setInsertBefore(spirvAsmInst);
                        const auto addr = tryGetAddress(
                            context,
                            lowerLValueExpr(context, operand.expr),
                            TryGetAddressMode::Default);
                        if (addr.flavor == LoweredValInfo::Flavor::Ptr)
                            i = addr.val;
                        else
                        {
                            context->getSink()->diagnose(operand.expr, Diagnostics::noSuchAddress);
                            return nullptr;
                        }
                    }
                    return builder->emitSPIRVAsmOperandInst(i);
                }
            case SPIRVAsmOperand::SlangType:
                {
                    IRInst* i;
                    {
                        IRBuilderInsertLocScope insertScope(builder);
                        builder->setInsertBefore(spirvAsmInst);
                        i = lowerType(context, operand.type.type);
                    }
                    return builder->emitSPIRVAsmOperandInst(i);
                }
            case SPIRVAsmOperand::SampledType:
                {
                    IRType* i;
                    {
                        IRBuilderInsertLocScope insertScope(builder);
                        builder->setInsertBefore(spirvAsmInst);
                        i = lowerType(context, operand.type.type);
                    }
                    return builder->emitSPIRVAsmOperandSampledType(i);
                }
            case SPIRVAsmOperand::ImageType:
                {
                    IRInst* i;
                    {
                        IRBuilderInsertLocScope insertScope(builder);
                        builder->setInsertBefore(spirvAsmInst);
                        i = getSimpleVal(context, lowerRValueExpr(context, operand.expr));
                    }
                    return builder->emitSPIRVAsmOperandImageType(i);
                }
            case SPIRVAsmOperand::SampledImageType:
                {
                    IRInst* i;
                    {
                        IRBuilderInsertLocScope insertScope(builder);
                        builder->setInsertBefore(spirvAsmInst);
                        i = getSimpleVal(context, lowerRValueExpr(context, operand.expr));
                    }
                    return builder->emitSPIRVAsmOperandSampledImageType(i);
                }
            case SPIRVAsmOperand::TruncateMarker:
                {
                    return builder->emitSPIRVAsmOperandTruncate();
                }
            case SPIRVAsmOperand::ConvertTexel:
                {
                    IRInst* i;
                    {
                        IRBuilderInsertLocScope insertScope(builder);
                        builder->setInsertBefore(spirvAsmInst);
                        i = getSimpleVal(context, lowerRValueExpr(context, operand.expr));
                    }
                    return builder->emitSPIRVAsmOperandConvertTexel(i);
                }
            case SPIRVAsmOperand::EntryPoint:
                {
                    return builder->emitSPIRVAsmOperandEntryPoint();
                }
            case SPIRVAsmOperand::RayPayloadFromLocation:
                {
                    IRInst* i;
                    {
                        IRBuilderInsertLocScope insertScope(builder);
                        builder->setInsertBefore(spirvAsmInst);
                        i = getSimpleVal(context, lowerRValueExpr(context, operand.expr));
                    }
                    return builder->emitSPIRVAsmOperandRayPayloadFromLocation(i);
                }
            case SPIRVAsmOperand::RayAttributeFromLocation:
                {
                    IRInst* i;
                    {
                        IRBuilderInsertLocScope insertScope(builder);
                        builder->setInsertBefore(spirvAsmInst);
                        i = getSimpleVal(context, lowerRValueExpr(context, operand.expr));
                    }
                    return builder->emitSPIRVAsmOperandRayAttributeFromLocation(i);
                }
            case SPIRVAsmOperand::RayCallableFromLocation:
                {
                    IRInst* i;
                    {
                        IRBuilderInsertLocScope insertScope(builder);
                        builder->setInsertBefore(spirvAsmInst);
                        i = getSimpleVal(context, lowerRValueExpr(context, operand.expr));
                    }
                    return builder->emitSPIRVAsmOperandRayCallableFromLocation(i);
                }
            }
            SLANG_UNREACHABLE("Unhandled case in visitSPIRVAsmExpr");
        };
        IRBuilderInsertLocScope insertScope(builder);
        builder->setInsertInto(spirvAsmInst);
        for (const auto& inst : expr->insts)
        {
            const auto opcode = lowerOperand(inst.opcode);
            List<IRInst*> operands;
            for (const auto& operand : inst.operands)
                operands.add(lowerOperand(operand));
            builder->emitSPIRVAsmInst(opcode, operands);
        }
        return LoweredValInfo::simple(spirvAsmInst);
    }

    LoweredValInfo visitIndexExpr(IndexExpr* expr)
    {
        auto type = lowerType(context, expr->type);
        auto baseVal = lowerSubExpr(expr->baseExpression);

        SLANG_RELEASE_ASSERT(expr->indexExprs.getCount() == 1);

        auto indexVal = getSimpleVal(context, lowerRValueExpr(context, expr->indexExprs[0]));

        return subscriptValue(type, baseVal, indexVal);
    }

    LoweredValInfo visitThisExpr(ThisExpr* /*expr*/) { return context->thisVal; }

    LoweredValInfo visitReturnValExpr(ReturnValExpr*) { return context->returnDestination; }

    LoweredValInfo visitMemberExpr(MemberExpr* expr)
    {
        auto loweredType = lowerType(context, expr->type);

        auto baseExpr = expr->baseExpression;
        baseExpr = sharedLoweringContext.maybeIgnoreCastToInterface(baseExpr);
        auto loweredBase = lowerSubExpr(baseExpr);

        auto declRef = expr->declRef;
        if (auto fieldDeclRef = declRef.as<VarDecl>())
        {
            // Okay, easy enough: we have a reference to a field of a struct type...
            return extractField(loweredType, loweredBase, fieldDeclRef);
        }
        else if (auto callableDeclRef = declRef.as<CallableDecl>())
        {
            RefPtr<BoundMemberInfo> boundMemberInfo = new BoundMemberInfo();
            boundMemberInfo->type = nullptr;
            boundMemberInfo->base = loweredBase;
            boundMemberInfo->declRef = callableDeclRef;
            return LoweredValInfo::boundMember(boundMemberInfo);
        }
        else if (auto propertyDeclRef = declRef.as<PropertyDecl>())
        {
            // A reference to a property is a special case, because
            // we must translate the reference to the property
            // into a reference to one of its accessors.
            //
            return lowerStorageReference(
                context,
                loweredType,
                propertyDeclRef,
                loweredBase,
                0,
                nullptr);
        }

        SLANG_UNIMPLEMENTED_X("codegen for member expression");
        UNREACHABLE_RETURN(LoweredValInfo());
    }

    // We will always lower a dereference expression (`*ptr`)
    // as an l-value, since that is the easiest way to handle it.
    LoweredValInfo visitDerefExpr(DerefExpr* expr)
    {
        auto loweredBase = lowerRValueExpr(context, expr->base);

        // TODO: handle tupel-type for `base`

        // The type of the lowered base must by some kind of pointer,
        // in order for a dereference to make senese, so we just
        // need to extract the value type from that pointer here.
        //
        IRInst* loweredBaseVal = getSimpleVal(context, loweredBase);
        IRType* loweredBaseType = loweredBaseVal->getDataType();

        if (as<IRPointerLikeType>(loweredBaseType) || as<IRPtrTypeBase>(loweredBaseType))
        {
            // Note that we do *not* perform an actual `load` operation
            // here, but rather just use the pointer value to construct
            // an appropriate `LoweredValInfo` representing the underlying
            // dereference.
            //
            // This is important so that an expression like `&((*foo).bar)`
            // (which is desugared from `&foo->bar`) can be handled; such
            // an expression does *not* perform a dereference at runtime,
            // and is just a bit of pointer math.
            //
            return LoweredValInfo::ptr(loweredBaseVal);
        }
        else
        {
            SLANG_UNIMPLEMENTED_X("codegen for deref expression");
            UNREACHABLE_RETURN(LoweredValInfo());
        }
    }

    LoweredValInfo visitMakeRefExpr(MakeRefExpr* expr)
    {
        auto loweredBase = lowerLValueExpr(context, expr->base);

        if (loweredBase.flavor != LoweredValInfo::Flavor::Ptr)
        {
            SLANG_ASSERT(as<ConstRefType>(expr->type));
            // If the base isn't a pointer, then we are trying to form
            // a const ref to a temporary value.
            // To do so we must copy it into a variable.
            auto baseVal = getSimpleVal(context, loweredBase);
            auto tempVar = context->irBuilder->emitVar(baseVal->getFullType());
            context->irBuilder->emitStore(tempVar, baseVal);
            loweredBase.val = tempVar;
        }

        loweredBase.flavor = LoweredValInfo::Flavor::Simple;
        return loweredBase;
    }

    LoweredValInfo visitParenExpr(ParenExpr* expr) { return lowerSubExpr(expr->base); }

    LoweredValInfo visitPackExpr(PackExpr* expr)
    {
        List<IRInst*> irArgs;
        for (auto arg : expr->args)
        {
            irArgs.add(getSimpleVal(context, lowerSubExpr(arg)));
        }
        auto irMakeTuple =
            getBuilder()->emitMakeValuePack((UInt)irArgs.getCount(), irArgs.getBuffer());
        return LoweredValInfo::simple(irMakeTuple);
    }

    LoweredValInfo visitEachExpr(EachExpr* expr)
    {
        auto subVal = lowerSubExpr(expr->baseExpr);
        SLANG_ASSERT(context->expandIndex);
        auto irEach = getBuilder()->emitGetTupleElement(
            lowerType(context, expr->type),
            getSimpleVal(context, subVal),
            context->expandIndex);
        return LoweredValInfo::simple(irEach);
    }

    LoweredValInfo visitExpandExpr(ExpandExpr* expr)
    {
        auto irBuilder = getBuilder();
        auto irType = lowerType(context, expr->type);
        List<IRInst*> irCapturedPacks;
        if (auto expandType = as<IRExpandType>(irType))
        {
            for (UInt i = 0; i < expandType->getCaptureCount(); i++)
            {
                irCapturedPacks.add(expandType->getCaptureType(i));
            }
        }
        else
        {
            // If the type of the expression is not an ExpandType, then it must be
            // a DeclRefType to a generic type pack parameter.
            // In this case, the captured type is just the DeclRefType itself.
            irCapturedPacks.add(irType);
        }
        auto expandInst = irBuilder->emitExpandInst(
            irType,
            (UInt)irCapturedPacks.getCount(),
            irCapturedPacks.getBuffer());
        irBuilder->setInsertInto(expandInst);
        irBuilder->emitBlock();
        auto eachIndex = irBuilder->emitParam(irBuilder->getIntType());
        IRInst* oldExpandIndex = context->expandIndex;
        context->expandIndex = eachIndex;
        SLANG_DEFER(context->expandIndex = oldExpandIndex);
        irBuilder->emitYield(getSimpleVal(context, lowerSubExpr(expr->baseExpr)));
        irBuilder->setInsertAfter(expandInst);
        return LoweredValInfo::simple(expandInst);
    }

    LoweredValInfo getSimpleDefaultVal(IRType* type)
    {
        type = (IRType*)unwrapAttributedType(type);
        if (auto basicType = as<IRBasicType>(type))
        {
            switch (basicType->getBaseType())
            {
            default:
                SLANG_UNEXPECTED("missing case for getting IR default value");
                UNREACHABLE_RETURN(LoweredValInfo());
                break;

            case BaseType::Bool:
                return LoweredValInfo::simple(getBuilder()->getBoolValue(false));

            case BaseType::Int8:
            case BaseType::Int16:
            case BaseType::Int:
            case BaseType::Int64:
            case BaseType::UInt8:
            case BaseType::UInt16:
            case BaseType::UInt:
            case BaseType::UInt64:
            case BaseType::UIntPtr:
            case BaseType::IntPtr:
                return LoweredValInfo::simple(getBuilder()->getIntValue(type, 0));

            case BaseType::Half:
            case BaseType::Float:
            case BaseType::Double:
                return LoweredValInfo::simple(getBuilder()->getFloatValue(type, 0.0));
            }
        }

        SLANG_UNEXPECTED("missing case for getting IR default value");
        UNREACHABLE_RETURN(LoweredValInfo());
    }

    Type* getOriginalTypeFromModifiedType(Type* type)
    {
        auto innerType = type;
        while (auto modifiedType = as<ModifiedType>(innerType))
            innerType = modifiedType->getBase();
        return innerType;
    }

    LoweredValInfo getDefaultVal(Type* type)
    {
        type = getOriginalTypeFromModifiedType(type);

        auto irType = lowerType(context, type);
        if (auto basicType = as<BasicExpressionType>(type))
        {
            return getSimpleDefaultVal(irType);
        }
        else if (auto vectorType = as<VectorExpressionType>(type))
        {
            UInt elementCount = (UInt)getIntVal(vectorType->getElementCount());

            auto irDefaultValue =
                getSimpleVal(context, getDefaultVal(vectorType->getElementType()));

            List<IRInst*> args;
            for (UInt ee = 0; ee < elementCount; ++ee)
            {
                args.add(irDefaultValue);
            }
            return LoweredValInfo::simple(
                getBuilder()->emitMakeVector(irType, args.getCount(), args.getBuffer()));
        }
        else if (auto matrixType = as<MatrixExpressionType>(type))
        {
            UInt rowCount = (UInt)getIntVal(matrixType->getRowCount());

            auto rowType = matrixType->getRowType();

            auto irDefaultValue = getSimpleVal(context, getDefaultVal(rowType));

            List<IRInst*> args;
            for (UInt rr = 0; rr < rowCount; ++rr)
            {
                args.add(irDefaultValue);
            }
            return LoweredValInfo::simple(
                getBuilder()->emitMakeMatrix(irType, args.getCount(), args.getBuffer()));
        }
        else if (auto arrayType = as<ArrayExpressionType>(type))
        {
            auto irDefaultElement =
                getSimpleVal(context, getDefaultVal(arrayType->getElementType()));

            return LoweredValInfo::simple(
                getBuilder()->emitMakeArrayFromElement(irType, irDefaultElement));
        }
        else if (auto ptrType = as<PtrType>(type))
        {
            return LoweredValInfo::simple(getBuilder()->getNullPtrValue(irType));
        }
        else if (auto tupleType = as<TupleType>(type))
        {
            List<IRInst*> args;
            for (Index i = 0; i < tupleType->getMemberCount(); i++)
            {
                args.add(getSimpleVal(context, getDefaultVal(tupleType->getMember(i))));
            }
            return LoweredValInfo::simple(
                getBuilder()->emitMakeTuple(irType, args.getCount(), args.getBuffer()));
        }
        else if (auto declRefType = as<DeclRefType>(type))
        {
            DeclRef<Decl> declRef = declRefType->getDeclRef();
            if (auto enumType = declRef.as<EnumDecl>())
            {
                return LoweredValInfo::simple(getBuilder()->getIntValue(irType, 0));
            }
            else if (declRef.as<InterfaceDecl>())
            {
                return LoweredValInfo::simple(getBuilder()->emitDefaultConstruct(irType));
            }
            else if (auto aggTypeDeclRef = declRef.as<AggTypeDecl>())
            {
                List<IRInst*> args;

                if (auto structTypeDeclRef = aggTypeDeclRef.as<StructDecl>())
                {
                    if (auto baseStructType =
                            findBaseStructType(getASTBuilder(), structTypeDeclRef))
                    {
                        auto irBaseVal = getSimpleVal(context, getDefaultVal(baseStructType));
                        args.add(irBaseVal);
                    }
                }

                for (auto ff : getMembersOfType<VarDecl>(
                         getASTBuilder(),
                         aggTypeDeclRef,
                         MemberFilterStyle::Instance))
                {
                    auto irFieldVal = getSimpleVal(context, getDefaultVal(ff));
                    args.add(irFieldVal);
                }

                return LoweredValInfo::simple(
                    getBuilder()->emitMakeStruct(irType, args.getCount(), args.getBuffer()));
            }
        }
        return LoweredValInfo::simple(getBuilder()->emitDefaultConstruct(irType));
    }

    LoweredValInfo visitDefaultConstructExpr(DefaultConstructExpr* expr)
    {
        return LoweredValInfo::simple(
            getBuilder()->emitDefaultConstruct(lowerType(context, expr->type)));
    }

    LoweredValInfo getDefaultVal(DeclRef<VarDeclBase> decl)
    {
        if (auto initExpr = decl.getDecl()->initExpr)
        {
            return lowerRValueExpr(context, initExpr);
        }
        else
        {
            Type* type = decl.substitute(getASTBuilder(), decl.getDecl()->type);
            SLANG_ASSERT(type);
            return getDefaultVal(type);
        }
    }

    LoweredValInfo visitInitializerListExpr(InitializerListExpr* expr)
    {
        // Allocate a temporary of the given type
        auto type = expr->type;
        IRType* irType = lowerType(context, type);
        List<IRInst*> args;

        UInt argCount = expr->args.getCount();

        // If the initializer list was empty, then the user was
        // asking for default initialization, which should apply
        // to (almost) any type.
        //
        if (argCount == 0)
        {
            return getDefaultVal(type.type);
        }

        // Now for each argument in the initializer list,
        // fill in the appropriate field of the result
        if (auto arrayType = as<ArrayExpressionType>(type))
        {
            UInt elementCount = (UInt)getIntVal(arrayType->getElementCount());

            for (UInt ee = 0; ee < argCount; ++ee)
            {
                auto argExpr = expr->args[ee];
                LoweredValInfo argVal = lowerRValueExpr(context, argExpr);
                args.add(getSimpleVal(context, argVal));
            }
            if (elementCount > argCount)
            {
                auto irDefaultValue =
                    getSimpleVal(context, getDefaultVal(arrayType->getElementType()));
                for (UInt ee = argCount; ee < elementCount; ++ee)
                {
                    args.add(irDefaultValue);
                }
            }

            return LoweredValInfo::simple(
                getBuilder()->emitMakeArray(irType, args.getCount(), args.getBuffer()));
        }
        else if (auto vectorType = as<VectorExpressionType>(type))
        {
            UInt elementCount = (UInt)getIntVal(vectorType->getElementCount());

            for (UInt ee = 0; ee < argCount; ++ee)
            {
                auto argExpr = expr->args[ee];
                LoweredValInfo argVal = lowerRValueExpr(context, argExpr);
                args.add(getSimpleVal(context, argVal));
            }
            if (elementCount > argCount)
            {
                auto irDefaultValue =
                    getSimpleVal(context, getDefaultVal(vectorType->getElementType()));
                for (UInt ee = argCount; ee < elementCount; ++ee)
                {
                    args.add(irDefaultValue);
                }
            }

            return LoweredValInfo::simple(
                getBuilder()->emitMakeVector(irType, args.getCount(), args.getBuffer()));
        }
        else if (auto matrixType = as<MatrixExpressionType>(type))
        {
            UInt rowCount = (UInt)getIntVal(matrixType->getRowCount());

            for (UInt rr = 0; rr < argCount; ++rr)
            {
                auto argExpr = expr->args[rr];
                LoweredValInfo argVal = lowerRValueExpr(context, argExpr);
                args.add(getSimpleVal(context, argVal));
            }
            if (rowCount > argCount)
            {
                auto rowType = matrixType->getRowType();
                auto irDefaultValue = getSimpleVal(context, getDefaultVal(rowType));

                for (UInt rr = argCount; rr < rowCount; ++rr)
                {
                    args.add(irDefaultValue);
                }
            }

            return LoweredValInfo::simple(
                getBuilder()->emitMakeMatrix(irType, args.getCount(), args.getBuffer()));
        }
        else if (auto coopVecType = as<CoopVectorExpressionType>(type))
        {
            UInt elementCount = (UInt)getIntVal(coopVecType->getElementCount());

            for (UInt ee = 0; ee < argCount; ++ee)
            {
                auto argExpr = expr->args[ee];
                LoweredValInfo argVal = lowerRValueExpr(context, argExpr);
                args.add(getSimpleVal(context, argVal));
            }
            if (elementCount > argCount)
            {
                auto irDefaultValue =
                    getSimpleVal(context, getDefaultVal(coopVecType->getElementType()));
                for (UInt ee = argCount; ee < elementCount; ++ee)
                {
                    args.add(irDefaultValue);
                }
            }

            return LoweredValInfo::simple(
                getBuilder()->emitMakeCoopVector(irType, args.getCount(), args.getBuffer()));
        }
        else if (auto declRefType = as<DeclRefType>(type))
        {
            DeclRef<Decl> declRef = declRefType->getDeclRef();
            if (auto aggTypeDeclRef = declRef.as<AggTypeDecl>())
            {
                UInt argCounter = 0;

                // If the type is a structure type that inherits from another
                // structure type, then we need to treat the base type as
                // an implicit first field.
                //
                if (auto structTypeDeclRef = aggTypeDeclRef.as<StructDecl>())
                {
                    if (auto baseStructType =
                            findBaseStructType(getASTBuilder(), structTypeDeclRef))
                    {
                        UInt argIndex = argCounter++;
                        if (argIndex < argCount)
                        {
                            auto argExpr = expr->args[argIndex];
                            LoweredValInfo argVal = lowerRValueExpr(context, argExpr);
                            args.add(getSimpleVal(context, argVal));
                        }
                        else
                        {
                            auto irDefaultValue =
                                getSimpleVal(context, getDefaultVal(baseStructType));
                            args.add(irDefaultValue);
                        }
                    }
                }

                for (auto ff : getMembersOfType<VarDecl>(
                         getASTBuilder(),
                         aggTypeDeclRef,
                         MemberFilterStyle::Instance))
                {
                    UInt argIndex = argCounter++;
                    if (argIndex < argCount)
                    {
                        auto argExpr = expr->args[argIndex];
                        LoweredValInfo argVal = lowerRValueExpr(context, argExpr);
                        args.add(getSimpleVal(context, argVal));
                    }
                    else
                    {
                        auto irDefaultValue = getSimpleVal(context, getDefaultVal(ff));
                        args.add(irDefaultValue);
                    }
                }
                if (as<TupleType>(type))
                {
                    return LoweredValInfo::simple(
                        getBuilder()->emitMakeTuple(irType, args.getCount(), args.getBuffer()));
                }
                else
                {
                    return LoweredValInfo::simple(
                        getBuilder()->emitMakeStruct(irType, args.getCount(), args.getBuffer()));
                }
            }
        }

        // If none of the above cases matched, then we had better
        // have zero arguments in the initializer list, in which
        // case we are just looking for default initialization.
        //
        SLANG_UNEXPECTED("unhandled case for initializer list codegen");
        UNREACHABLE_RETURN(LoweredValInfo());
    }

    LoweredValInfo visitBoolLiteralExpr(BoolLiteralExpr* expr)
    {
        return LoweredValInfo::simple(context->irBuilder->getBoolValue(expr->value));
    }

    LoweredValInfo visitNullPtrLiteralExpr(NullPtrLiteralExpr*)
    {
        return LoweredValInfo::simple(context->irBuilder->getNullVoidPtrValue());
    }

    LoweredValInfo visitNoneLiteralExpr(NoneLiteralExpr*)
    {
        return LoweredValInfo::simple(context->irBuilder->getVoidValue());
    }

    LoweredValInfo visitIntegerLiteralExpr(IntegerLiteralExpr* expr)
    {
        auto type = lowerType(context, expr->type);
        return LoweredValInfo::simple(context->irBuilder->getIntValue(type, expr->value));
    }

    LoweredValInfo visitFloatingPointLiteralExpr(FloatingPointLiteralExpr* expr)
    {
        auto type = lowerType(context, expr->type);
        return LoweredValInfo::simple(context->irBuilder->getFloatValue(type, expr->value));
    }

    LoweredValInfo visitStringLiteralExpr(StringLiteralExpr* expr)
    {
        auto irLit = context->irBuilder->getStringValue(expr->value.getUnownedSlice());
        context->shared->m_stringLiterals.add(irLit);
        return LoweredValInfo::simple(irLit);
    }

    LoweredValInfo visitMakeOptionalExpr(MakeOptionalExpr* expr)
    {
        if (expr->value)
        {
            auto val = lowerRValueExpr(context, expr->value);
            auto optType = lowerType(context, expr->type);
            auto irVal = context->irBuilder->emitMakeOptionalValue(optType, val.val);
            return LoweredValInfo::simple(irVal);
        }
        else
        {
            auto optType = lowerType(context, expr->type);
            auto defaultVal = getDefaultVal(as<OptionalType>(expr->type)->getValueType());
            auto irVal = context->irBuilder->emitMakeOptionalNone(optType, defaultVal.val);
            return LoweredValInfo::simple(irVal);
        }
    }

    LoweredValInfo visitAggTypeCtorExpr(AggTypeCtorExpr* /*expr*/)
    {
        SLANG_UNIMPLEMENTED_X("codegen for aggregate type constructor expression");
        UNREACHABLE_RETURN(LoweredValInfo());
    }

    LoweredValInfo visitSelectExpr(SelectExpr* expr)
    {
        // A vector typed `select` expr will turn into a normal `select` op.
        if (!as<BasicExpressionType>(expr->arguments[0]->type.type))
        {
            return visitInvokeExpr(expr);
        }

        // In global scope? This is a constant, and we should emit as `select` inst.
        if (!getParentFunc(context->irBuilder->getInsertLoc().getInst()))
        {
            return visitInvokeExpr(expr);
        }

        // A scalar typed `select` expr will turn into an if-else to implement short circuiting
        // semantics.
        auto builder = context->irBuilder;
        auto thenBlock = builder->createBlock();
        auto elseBlock = builder->createBlock();
        auto afterBlock = builder->createBlock();
        auto irCond = getSimpleVal(context, lowerRValueExpr(context, expr->arguments[0]));
        builder->emitIfElse(irCond, thenBlock, elseBlock, afterBlock);
        builder->insertBlock(thenBlock);
        builder->setInsertInto(thenBlock);
        auto trueVal = getSimpleVal(context, lowerRValueExpr(context, expr->arguments[1]));
        builder->emitBranch(afterBlock, 1, &trueVal);
        builder->insertBlock(elseBlock);
        builder->setInsertInto(elseBlock);
        auto falseVal = getSimpleVal(context, lowerRValueExpr(context, expr->arguments[2]));
        builder->emitBranch(afterBlock, 1, &falseVal);
        builder->insertBlock(afterBlock);
        builder->setInsertInto(afterBlock);
        auto paramType = lowerType(context, expr->type.type);
        auto result = builder->emitParam(paramType);
        return LoweredValInfo::simple(result);
    }

    LoweredValInfo visitLogicOperatorShortCircuitExpr(LogicOperatorShortCircuitExpr* expr)
    {
        auto builder = context->irBuilder;
        auto thenBlock = builder->createBlock();
        auto elseBlock = builder->createBlock();
        auto afterBlock = builder->createBlock();
        auto irCond = getSimpleVal(context, lowerRValueExpr(context, expr->arguments[0]));

        // ifElse(<first param>, %true-block, %false-block, %after-block)
        builder->emitIfElse(irCond, thenBlock, elseBlock, afterBlock);

        // true-block: nonconditionalBranch(%after-block, <second param> : Bool)
        // true-block: nonconditionalBranch(%after-block, true) for ||
        builder->insertBlock(thenBlock);
        builder->setInsertInto(thenBlock);
        auto trueVal = expr->flavor == LogicOperatorShortCircuitExpr::Flavor::And
                           ? getSimpleVal(context, lowerRValueExpr(context, expr->arguments[1]))
                           : LoweredValInfo::simple(context->irBuilder->getBoolValue(true)).val;

        builder->emitBranch(afterBlock, 1, &trueVal);

        // false-block: nonconditionalBranch(%after-block, false) for &&
        // false-block: nonconditionalBranch(%after-block, <second param>: Bool) for ||
        builder->insertBlock(elseBlock);
        builder->setInsertInto(elseBlock);
        auto falseVal = expr->flavor == LogicOperatorShortCircuitExpr::Flavor::And
                            ? LoweredValInfo::simple(context->irBuilder->getBoolValue(false)).val
                            : getSimpleVal(context, lowerRValueExpr(context, expr->arguments[1]));

        builder->emitBranch(afterBlock, 1, &falseVal);

        // after-block: return input parameter
        builder->insertBlock(afterBlock);
        builder->setInsertInto(afterBlock);
        auto paramType = lowerType(context, expr->type.type);
        auto result = builder->emitParam(paramType);

        return LoweredValInfo::simple(result);
    }

    LoweredValInfo visitInvokeExpr(InvokeExpr* expr)
    {
        return sharedLoweringContext.visitInvokeExprImpl(
            expr,
            LoweredValInfo(),
            TryClauseEnvironment());
    }

    LoweredValInfo visitBuiltinCastExpr(BuiltinCastExpr* expr)
    {
        auto irType = lowerType(context, expr->type);
        auto irVal = getSimpleVal(context, lowerRValueExpr(context, expr->base));
        return LoweredValInfo::simple(context->irBuilder->emitCast(irType, irVal));
    }

    /// Emit code for a `try` invoke.
    LoweredValInfo visitTryExpr(TryExpr* expr)
    {
        auto invokeExpr = as<InvokeExpr>(expr->base);
        assert(invokeExpr);
        TryClauseEnvironment tryEnv;
        tryEnv.clauseType = expr->tryClauseType;
        return sharedLoweringContext.visitInvokeExprImpl(invokeExpr, LoweredValInfo(), tryEnv);
    }

    /// Emit code to cast `value` to a concrete `superType` (e.g., a `struct`).
    ///
    /// The `subTypeWitness` is expected to witness the sub-type relationship
    /// by naming a field (or chain of fields) that leads from the type of
    /// `value` to the field that stores its members for `superType`.
    ///
    LoweredValInfo emitCastToConcreteSuperTypeRec(
        LoweredValInfo const& value,
        IRType* superType,
        Val* subTypeWitness)
    {
        if (auto declaredSubtypeWitness = as<DeclaredSubtypeWitness>(subTypeWitness))
        {
            // Drop the specialization info on inheritance decl struct keys, as it makes no
            // sense to specialize a key.
            return extractField(superType, value, declaredSubtypeWitness->getDeclRef().getDecl());
        }
        else if (auto transitiveSubtypeWitness = as<TransitiveSubtypeWitness>(subTypeWitness))
        {
            // Try to resolve the inheritance situation which may show-up with 2+ levels of
            // inheritance. We will recursivly follow through the subType->midType &
            // midType->superType witnesses until we resolve DeclaredSubtypeWitness's
            LoweredValInfo subToMid;
            if (auto witness = as<SubtypeWitness>(transitiveSubtypeWitness->getSubToMid()))
                subToMid = emitCastToConcreteSuperTypeRec(
                    value,
                    lowerType(context, witness->getSup()),
                    witness);
            else
            {
                SLANG_ASSERT(!"unhandled");
                return nullptr;
            }

            if (auto witness = as<SubtypeWitness>(transitiveSubtypeWitness->getMidToSup()))
                return emitCastToConcreteSuperTypeRec(subToMid, superType, witness);
            else
            {
                SLANG_ASSERT(!"unhandled");
                return nullptr;
            }
        }
        else
        {
            SLANG_ASSERT(!"unhandled");
            return nullptr;
        }
    }

    LoweredValInfo visitCastToSuperTypeExpr(CastToSuperTypeExpr* expr)
    {
        auto superType = lowerType(context, expr->type);
        auto value = lowerRValueExpr(context, expr->valueArg);

        // First, we check if the witness is a type equality witness.
        // If so, we can simply emit a bit cast to the target type that should eventually
        // fold out to a no-op.
        // Note: if we are going to equivalent but not identical types in the future,
        // then the cast between equivalent types shouldn't be as simple as a bit cast
        // and will require actual coercion logic between the two types.
        // For now, we don't support type equivalence witness so this is safe for
        // equal types.
        if (isTypeEqualityWitness(expr->witnessArg))
        {
            return LoweredValInfo::simple(
                getBuilder()->emitBitCast(superType, getSimpleVal(context, value)));
        }

        // The actual operation that we need to perform here
        // depends on the kind of subtype relationship we
        // are making use of.
        //
        // The first important case is when the super type is
        // an interface type, such that casting from a concrete
        // value to that type creates a value of existential
        // type that binds together the concrete value and the
        // witness table that represents the subtype relationship.
        //
        if (auto declRefType = as<DeclRefType>(expr->type))
        {
            auto declRef = declRefType->getDeclRef();
            if (auto interfaceDeclRef = declRef.as<InterfaceDecl>())
            {
                // We have an expression that is "up-casting" some concrete value
                // to an existential type (aka interface type), using a subtype witness
                // (which will lower as a witness table) to show that the conversion
                // is valid.
                //
                auto witnessTable = lowerSimpleVal(context, expr->witnessArg);

                // At the IR level, this will become a `makeExistential` instruction,
                // which collects the above information into a single IR-level value.
                // A dynamic CPU implementation of Slang might encode an existential
                // as a "fat pointer" representation, which includes a pointer to
                // data for the concrete value, plus a pointer to the witness table.
                //
                // Note: if/when Slang supports more general existential types, such
                // as compositions of interface (e.g., `IReadable & IWritable`), then
                // we should probably extend the AST and IR mechanism here to accept
                // a sequence of witness tables.
                //
                auto concreteValue = getSimpleVal(context, value);
                auto existentialValue =
                    getBuilder()->emitMakeExistential(superType, concreteValue, witnessTable);
                return LoweredValInfo::simple(existentialValue);
            }
            else if (auto structDeclRef = declRef.as<StructDecl>())
            {
                // We are up-casting to a concrete `struct` super-type,
                // such that the witness will represent a field of the super-type
                // that is stored in instances of the sub-type (or a chain
                // of such fields for a transitive witness).
                //
                return emitCastToConcreteSuperTypeRec(value, superType, expr->witnessArg);
            }
        }
        SLANG_UNEXPECTED("unexpected case of subtype relationship");
        UNREACHABLE_RETURN(LoweredValInfo());
    }

    LoweredValInfo visitAsTypeExpr(AsTypeExpr* expr)
    {
        auto value = lowerLValueExpr(context, expr->value);
        ExtractedExistentialValInfo* existentialInfo = nullptr;
        auto optType = lowerType(context, expr->type);
        SLANG_RELEASE_ASSERT(optType->getOp() == kIROp_OptionalType);
        auto targetType = optType->getOperand(0);
        auto builder = getBuilder();
        auto var = builder->emitVar(optType);
        IRInst* isType = nullptr;
        if (expr->witnessArg)
        {
            auto witness = lowerSimpleVal(context, expr->witnessArg);
            existentialInfo = value.getExtractedExistentialValInfo();
            isType = builder->emitIsType(
                existentialInfo->extractedVal,
                existentialInfo->witnessTable,
                targetType,
                witness);
        }
        else
        {
            SLANG_ASSERT(value.val);
            auto leftType = lowerType(context, expr->value->type);
            IRInst* args[] = {leftType, targetType};
            isType = builder->emitIntrinsicInst(builder->getBoolType(), kIROp_TypeEquals, 2, args);
        }
        IRBlock* trueBlock;
        IRBlock* falseBlock;
        IRBlock* afterBlock;
        builder->emitIfElseWithBlocks(isType, trueBlock, falseBlock, afterBlock);
        builder->setInsertInto(trueBlock);
        auto irVal = builder->emitReinterpret(
            targetType,
            existentialInfo ? existentialInfo->extractedVal : getSimpleVal(context, value));
        auto optionalVal = builder->emitMakeOptionalValue(optType, irVal);
        builder->emitStore(var, optionalVal);
        builder->emitBranch(afterBlock);
        builder->setInsertInto(falseBlock);
        auto defaultVal = getDefaultVal(as<OptionalType>(expr->type)->getValueType());
        auto noneVal = builder->emitMakeOptionalNone(optType, defaultVal.val);
        builder->emitStore(var, noneVal);
        builder->emitBranch(afterBlock);
        builder->setInsertInto(afterBlock);
        auto result = builder->emitLoad(var);
        return LoweredValInfo::simple(result);
    }

    LoweredValInfo visitIsTypeExpr(IsTypeExpr* expr)
    {
        if (expr->constantVal)
        {
            return LoweredValInfo::simple(getBuilder()->getBoolValue(expr->constantVal->value));
        }
        // If expr is a witness, then this is a run-time type check from for an existential type.
        if (expr->witnessArg)
        {
            auto value = lowerLValueExpr(context, expr->value);
            auto type = lowerType(context, expr->typeExpr.type);
            auto witness = lowerSimpleVal(context, expr->witnessArg);
            auto existentialInfo = value.getExtractedExistentialValInfo();
            auto irVal = getBuilder()->emitIsType(
                existentialInfo->extractedVal,
                existentialInfo->witnessTable,
                type,
                witness);
            return LoweredValInfo::simple(irVal);
        }
        // For all other cases, we map to a simple type equality check in the IR.
        IRType* leftType = nullptr;
        if (auto typeType = as<TypeType>(expr->value->type))
        {
            leftType = lowerType(context, typeType->getType());
        }
        else
        {
            leftType = lowerType(context, expr->value->type);
        }
        auto rightType = lowerType(context, expr->typeExpr.type);
        IRInst* args[] = {leftType, rightType};
        auto irVal =
            getBuilder()->emitIntrinsicInst(getBuilder()->getBoolType(), kIROp_TypeEquals, 2, args);
        return LoweredValInfo::simple(irVal);
    }

    LoweredValInfo visitModifierCastExpr(ModifierCastExpr* expr)
    {
        return this->dispatch(expr->valueArg);
    }

    LoweredValInfo subscriptValue(IRType* type, LoweredValInfo baseVal, IRInst* indexVal)
    {
        auto builder = getBuilder();

        // The `tryGetAddress` operation will take a complex value representation
        // and try to turn it into a single pointer, if possible.
        //
        baseVal = tryGetAddress(context, baseVal, TryGetAddressMode::Aggressive);

        // The `materialize` operation should ensure that we only have to deal
        // with the small number of base cases for lowered value representations.
        //
        baseVal = materialize(context, baseVal);

        switch (baseVal.flavor)
        {
        case LoweredValInfo::Flavor::Simple:
            return LoweredValInfo::simple(
                builder->emitElementExtract(type, getSimpleVal(context, baseVal), indexVal));

        case LoweredValInfo::Flavor::Ptr:
            return LoweredValInfo::ptr(builder->emitElementAddress(baseVal.val, indexVal));

        default:
            SLANG_UNIMPLEMENTED_X("subscript expr");
            UNREACHABLE_RETURN(LoweredValInfo());
        }
    }

    LoweredValInfo extractField(IRType* fieldType, LoweredValInfo base, DeclRef<Decl> field)
    {
        return Slang::extractField(context, fieldType, base, field);
    }

    LoweredValInfo visitStaticMemberExpr(StaticMemberExpr* expr)
    {
        return emitDeclRef(context, expr->declRef, lowerType(context, expr->type));
    }

    LoweredValInfo visitGenericAppExpr(GenericAppExpr* /*expr*/)
    {
        SLANG_UNIMPLEMENTED_X("generic application expression during code generation");
        UNREACHABLE_RETURN(LoweredValInfo());
    }

    LoweredValInfo visitSharedTypeExpr(SharedTypeExpr* /*expr*/)
    {
        SLANG_UNIMPLEMENTED_X("shared type expression during code generation");
        UNREACHABLE_RETURN(LoweredValInfo());
    }

    LoweredValInfo visitThisTypeExpr(ThisTypeExpr* /*expr*/)
    {
        SLANG_UNIMPLEMENTED_X("this-type expression during code generation");
        UNREACHABLE_RETURN(LoweredValInfo());
    }

    LoweredValInfo visitAndTypeExpr(AndTypeExpr* /*expr*/)
    {
        SLANG_UNIMPLEMENTED_X("'&' type expression during code generation");
        UNREACHABLE_RETURN(LoweredValInfo());
    }

    LoweredValInfo visitModifiedTypeExpr(ModifiedTypeExpr* /*expr*/)
    {
        SLANG_UNIMPLEMENTED_X("type expression during code generation");
        UNREACHABLE_RETURN(LoweredValInfo());
    }

    LoweredValInfo visitFuncTypeExpr(FuncTypeExpr* /*expr*/)
    {
        SLANG_UNIMPLEMENTED_X("type expression during code generation");
        UNREACHABLE_RETURN(LoweredValInfo());
    }

    LoweredValInfo visitTupleTypeExpr(TupleTypeExpr* /*expr*/)
    {
        SLANG_UNIMPLEMENTED_X("type expression during code generation");
        UNREACHABLE_RETURN(LoweredValInfo());
    }

    LoweredValInfo visitPointerTypeExpr(PointerTypeExpr* /*expr*/)
    {
        SLANG_UNIMPLEMENTED_X("'*' type expression during code generation");
        UNREACHABLE_RETURN(LoweredValInfo());
    }

    LoweredValInfo visitAssocTypeDecl(AssocTypeDecl* /*decl*/)
    {
        SLANG_UNIMPLEMENTED_X("associatedtype expression during code generation");
        UNREACHABLE_RETURN(LoweredValInfo());
    }

    LoweredValInfo visitAssignExpr(AssignExpr* expr)
    {
        // Because our representation of lowered "values"
        // can encompass l-values explicitly, we can
        // lower assignment easily. We just lower the left-
        // and right-hand sides, and then perform an assignment
        // based on the resulting values.
        //
        auto leftVal = lowerLValueExpr(context, expr->left);
        assignExpr(context, leftVal, expr->right, expr->loc);

        // The result value of the assignment expression is
        // the value of the left-hand side (and it is expected
        // to be an l-value).
        return leftVal;
    }

    LoweredValInfo visitLetExpr(LetExpr* expr)
    {
        // Note: The semantics here are annoyingly subtle.
        //
        // If `expr->decl->initExpr` is an l-value, then we will set things up
        // so that `expr->decl` is bound as an *alias* for that l-value.
        //
        // Otherwise, `expr->decl` will simply be bound to the r-value.
        //
        // The first case is necessary to make `maybeMoveTemp` operations that
        // produce l-value results work correctly, but seems slippery.
        //
        // TODO: We should probably have two AST node types to cover the two
        // different use cases of `LetExpr`: the definitely-immutable case that
        // actually behaves like a `let`, and this other mutable-alias case that
        // feels kind of messy and gross.

        auto initVal = lowerLValueExpr(context, expr->decl->initExpr);
        context->setGlobalValue(expr->decl, initVal);
        auto bodyVal = lowerSubExpr(expr->body);
        return bodyVal;
    }

    LoweredValInfo visitExtractExistentialValueExpr(ExtractExistentialValueExpr* expr)
    {
        // We are being asked to extract the value from an existential, which
        // is itself a single IR op. However, we also need to handle the case
        // where `expr` might be used as an l-value, in which case we need
        // additional information to allow any mutations through the extracted
        // value to be written back.

        auto existentialType = lowerType(context, getType(getASTBuilder(), expr->declRef));
        auto existentialVal = emitDeclRef(context, expr->declRef, existentialType);

        // Note that we make a *copy* of the existential value that is definitely
        // a simple r-value. This ensures that all the `extractExistential*()` operations
        // below work on the same consistent IR value.
        //
        auto existentialValCopy = getSimpleVal(context, existentialVal);

        auto openedType = lowerType(context, expr->type);

        auto extractedVal =
            getBuilder()->emitExtractExistentialValue(openedType, existentialValCopy);

        if (!isLValueContext())
        {
            // If we are in an r-value context, we can directly use the `extractExistentialValue`
            // instruction as the result, and life is simple.
            //
            return LoweredValInfo::simple(extractedVal);
        }

        // In an l-value context, we need to track the information necessary so that
        // if a new/modified value of `openedType` was produced, we could write it
        // back into the original `existentialVal`'s location.
        //
        // The write-back is actually pretty simple: it is just a `makeExisential` op.
        // In order to be able to emit that op later, we need to track the operands
        // that it would use. The first operand would be the new concrete value (which
        // would implicitly encode the concrete type via its IR type) while the second
        // is the witness table for the conformance to the existential.
        //
        // Note: We are assuming/requiring here that any value "written back" must have
        // the exact same concrete type as `extractedVal`, so taht it can use the same
        // IR witness table. The front-end should be enforcing that constraint, and we
        // have no way to check or enforce it at this point.

        auto witnessTable = getBuilder()->emitExtractExistentialWitnessTable(existentialValCopy);

        RefPtr<ExtractedExistentialValInfo> info = new ExtractedExistentialValInfo();
        info->extractedVal = extractedVal;
        info->existentialVal = existentialVal;
        info->existentialType = existentialType;
        info->witnessTable = witnessTable;

        context->shared->extValues.add(info);
        return LoweredValInfo::extractedExistential(info);
    }

    LoweredValInfo visitOpenRefExpr(OpenRefExpr* expr)
    {
        auto info = lowerRValueExpr(context, expr->innerExpr);
        SLANG_RELEASE_ASSERT(as<IRPtrTypeBase>(info.val->getFullType()));
        SLANG_RELEASE_ASSERT(info.flavor == LoweredValInfo::Flavor::Simple);
        info.flavor = LoweredValInfo::Flavor::Ptr;
        return info;
    }
};

struct LValueExprLoweringVisitor : ExprLoweringVisitorBase<LValueExprLoweringVisitor>
{
    static bool _isLValueContext() { return true; }

    LoweredValInfo visitLValueImplicitCastExpr(LValueImplicitCastExpr* expr)
    {
        auto irType = lowerType(context, expr->type);
        auto loweredArg = lowerLValueExpr(context, expr->arguments[0]);

        RefPtr<ImplicitCastLValueInfo> lValueInfo = new ImplicitCastLValueInfo();
        lValueInfo->type = irType;
        lValueInfo->base = loweredArg;
        lValueInfo->lValueType = kParameterDirection_InOut;
        if (as<OutImplicitCastExpr>(expr))
            lValueInfo->lValueType = kParameterDirection_Out;
        context->shared->extValues.add(lValueInfo);
        return LoweredValInfo::implicitCastedLValue(lValueInfo);
    }

    // When visiting a swizzle expression in an l-value context,
    // we need to construct a "swizzled l-value."
    LoweredValInfo visitMatrixSwizzleExpr(MatrixSwizzleExpr* expr)
    {
        auto irType = lowerType(context, expr->type);
        auto loweredBase = lowerRValueExpr(context, expr->base);

        RefPtr<SwizzledMatrixLValueInfo> swizzledLValue = new SwizzledMatrixLValueInfo();
        swizzledLValue->type = irType;

        UInt elementCount = (UInt)expr->elementCount;
        swizzledLValue->elementCount = elementCount;

        // In the default case, we can just copy the indices being
        // used for the swizzle over directly from the expression,
        // and use the base as-is.
        //
        swizzledLValue->base = loweredBase;
        for (UInt ii = 0; ii < elementCount; ++ii)
        {
            swizzledLValue->elementCoords[ii] = expr->elementCoords[ii];
        }

        context->shared->extValues.add(swizzledLValue);
        return LoweredValInfo::swizzledMatrixLValue(swizzledLValue);
    }

    // When visiting a swizzle expression in an l-value context,
    // we need to construct a "swizzled l-value."
    LoweredValInfo visitSwizzleExpr(SwizzleExpr* expr)
    {
        auto irType = lowerType(context, expr->type);
        auto loweredBase = lowerLValueExpr(context, expr->base);
        UInt elementCount = (UInt)expr->elementIndices.getCount();

        // Assign to 'bs' the elements from 'as' according to the first 'n' indices in 'is'
        auto backpermute = [](UInt n, const auto as, const auto is, auto bs)
        {
            for (UInt i = 0; i < n; ++i)
            {
                bs[i] = as[is[i]];
            }
        };

        LoweredValInfo result;

        // As required by the implementation of 'assign' and as a small
        // optimization, we will detect if the base expression has also lowered
        // into a swizzle and only return a single swizzle instead of nested
        // swizzles.
        //
        // E.g., if we have input like `foo[i].zw.y` we should optimize it
        // down to just `foo[i].w`.
        if (loweredBase.flavor == LoweredValInfo::Flavor::SwizzledLValue)
        {
            auto baseSwizzleInfo = loweredBase.getSwizzledLValueInfo();

            // Our new swizzle will use the same base expression (e.g.,
            // `foo[i]` in our example above), but will need to remap
            // the swizzle indices it uses.
            //

            RefPtr<SwizzledLValueInfo> swizzledLValue = new SwizzledLValueInfo;
            swizzledLValue->type = irType;
            swizzledLValue->base = baseSwizzleInfo->base;
            swizzledLValue->elementIndices.add((uint32_t)elementCount);

            // Take the swizzle element of the "outer" swizzle, as it was
            // written by the user. In our running example of `foo[i].zw.y`
            // this is the `y` element reference.
            //
            // Use that original element index to figure out which of the
            // elements of the original swizzle this should map to.
            backpermute(
                swizzledLValue->elementIndices.getCount(),
                baseSwizzleInfo->elementIndices,
                expr->elementIndices,
                swizzledLValue->elementIndices);

            context->shared->extValues.add(swizzledLValue);
            result = LoweredValInfo::swizzledLValue(swizzledLValue);
        }
        else if (loweredBase.flavor == LoweredValInfo::Flavor::SwizzledMatrixLValue)
        {
            auto baseSwizzleInfo = loweredBase.getSwizzledMatrixLValueInfo();

            RefPtr<SwizzledMatrixLValueInfo> swizzledLValue = new SwizzledMatrixLValueInfo();
            swizzledLValue->type = irType;
            swizzledLValue->base = baseSwizzleInfo->base;
            swizzledLValue->elementCount = elementCount;

            // Use the index of our swizzle to permute the index of the base
            // swizzle as above
            backpermute(
                swizzledLValue->elementCount,
                baseSwizzleInfo->elementCoords,
                expr->elementIndices,
                swizzledLValue->elementCoords);

            context->shared->extValues.add(swizzledLValue);
            result = LoweredValInfo::swizzledMatrixLValue(swizzledLValue);
        }
        else
        {
            RefPtr<SwizzledLValueInfo> swizzledLValue = new SwizzledLValueInfo;
            swizzledLValue->type = irType;
            swizzledLValue->base = loweredBase;
            swizzledLValue->elementIndices = expr->elementIndices;
            context->shared->extValues.add(swizzledLValue);
            result = LoweredValInfo::swizzledLValue(swizzledLValue);
        }

        // For a one-element swizzle on a tuple, we can just return the pointer to the member
        // instead of a SwizzledLValue because they can't follow the same folding logic as
        // vectors and matrices.
        //
        bool shouldUseSimpleLVal = elementCount == 1 && as<TupleType>(expr->base->type) != nullptr;
        if (shouldUseSimpleLVal)
        {
            auto addr = getAddress(context, result, expr->loc);
            return LoweredValInfo::ptr(addr);
        }
        return result;
    }
};

struct RValueExprLoweringVisitor : public ExprLoweringVisitorBase<RValueExprLoweringVisitor>
{
    static bool _isLValueContext() { return false; }

    LoweredValInfo visitMatrixSwizzleExpr(MatrixSwizzleExpr* expr)
    {
        auto resultType = lowerType(context, expr->type);
        auto base = lowerSubExpr(expr->base);
        auto matType = as<MatrixExpressionType>(expr->base->type.type);
        if (!matType)
            SLANG_UNEXPECTED("Expected a matrix type in matrix swizzle");
        auto subscript2 = lowerType(context, matType->getElementType());
        auto subscript1 = lowerType(context, matType->getRowType());

        auto builder = getBuilder();

        auto irIntType = getIntType(context);

        UInt elementCount = (UInt)expr->elementCount;
        IRInst* irExtracts[4];
        for (UInt ii = 0; ii < elementCount; ++ii)
        {
            auto index1 =
                builder->getIntValue(irIntType, (IRIntegerValue)expr->elementCoords[ii].row);
            auto index2 =
                builder->getIntValue(irIntType, (IRIntegerValue)expr->elementCoords[ii].col);
            // First index expression
            auto irExtract1 = subscriptValue(subscript1, base, index1);
            // Second index expression
            irExtracts[ii] = getSimpleVal(context, subscriptValue(subscript2, irExtract1, index2));
        }
        auto irVector = builder->emitMakeVector(resultType, elementCount, irExtracts);

        return LoweredValInfo::simple(irVector);
    }

    // A swizzle in an r-value context can save time by just
    // emitting the swizzle instructions directly.
    LoweredValInfo visitSwizzleExpr(SwizzleExpr* expr)
    {
        auto irType = lowerType(context, expr->type);
        auto irBase = getSimpleVal(context, lowerRValueExpr(context, expr->base));

        auto builder = getBuilder();

        auto irIntType = getIntType(context);

        ShortList<IRInst*, 4> irElementIndices;
        irElementIndices.setCount(expr->elementIndices.getCount());
        for (UInt ii = 0; ii < (UInt)expr->elementIndices.getCount(); ++ii)
        {
            irElementIndices[ii] =
                builder->getIntValue(irIntType, (IRIntegerValue)expr->elementIndices[ii]);
        }

        auto irSwizzle = builder->emitSwizzle(
            irType,
            irBase,
            (UInt)irElementIndices.getCount(),
            &irElementIndices[0]);

        return LoweredValInfo::simple(irSwizzle);
    }

    LoweredValInfo visitOpenRefExpr(OpenRefExpr* expr)
    {
        auto inner = lowerLValueExpr(context, expr->innerExpr);
        return LoweredValInfo::ptr(inner.val);
    }
};

// ExprLoweringVisitor that fuses the destination assignment.
//
struct DestinationDrivenRValueExprLoweringVisitor
    : ExprVisitor<DestinationDrivenRValueExprLoweringVisitor>
{
    ExprLoweringContext<DestinationDrivenRValueExprLoweringVisitor> sharedLoweringContext;
    LoweredValInfo destination;

    IRGenContext*& context;
    DestinationDrivenRValueExprLoweringVisitor()
        : context(sharedLoweringContext.context)
    {
    }

    static bool _isLValueContext() { return false; }

    // The default case is lower the rvalue expr independently and then assign to destination.
    void visitExpr(Expr* expr)
    {
        auto rValue = lowerRValueExpr(context, expr);
        assign(context, destination, rValue);
    }

    void visitSelectExpr(SelectExpr* expr)
    {
        auto rValue = lowerRValueExpr(context, expr);
        assign(context, destination, rValue);
    }

    void visitLogicOperatorShortCircuitExpr(LogicOperatorShortCircuitExpr* expr)
    {
        auto rValue = lowerRValueExpr(context, expr);
        assign(context, destination, rValue);
    }

    void visitInvokeExpr(InvokeExpr* expr)
    {
        LoweredValInfo resultRVal;
        {
            IRBuilderSourceLocRAII sourceLocInfo(context->irBuilder, expr->loc);
            resultRVal = sharedLoweringContext.visitInvokeExprImpl(
                expr,
                destination,
                TryClauseEnvironment{});
        }
        if (resultRVal.flavor != LoweredValInfo::Flavor::None)
        {
            // If we weren't able to fuse the destination write during lowering rvalue,
            // we should insert the assign operation now.
            assign(context, destination, resultRVal);
        }
    }

    /// Emit code for a `try` invoke.
    void visitTryExpr(TryExpr* expr)
    {
        auto invokeExpr = as<InvokeExpr>(expr->base);
        assert(invokeExpr);
        TryClauseEnvironment tryEnv;
        tryEnv.clauseType = expr->tryClauseType;
        auto rValue = sharedLoweringContext.visitInvokeExprImpl(invokeExpr, destination, tryEnv);
        if (rValue.flavor != LoweredValInfo::Flavor::None)
        {
            // If we weren't able to fuse the destination write during lowering rvalue,
            // we should insert the assign operation now.
            assign(context, destination, rValue);
        }
    }
};

LoweredValInfo lowerLValueExpr(IRGenContext* context, Expr* expr)
{
    IRBuilderSourceLocRAII sourceLocInfo(context->irBuilder, expr->loc);

    LValueExprLoweringVisitor visitor;
    visitor.context = context;
    auto info = visitor.dispatch(expr);
    return info;
}

LoweredValInfo lowerRValueExpr(IRGenContext* context, Expr* expr)
{
    IRBuilderSourceLocRAII sourceLocInfo(context->irBuilder, expr->loc);

    RValueExprLoweringVisitor visitor;
    visitor.context = context;
    auto info = visitor.dispatch(expr);
    return info;
}

void lowerRValueExprWithDestination(IRGenContext* context, LoweredValInfo destination, Expr* expr)
{
    DestinationDrivenRValueExprLoweringVisitor visitor;
    visitor.context = context;
    visitor.destination = destination;
    visitor.dispatch(expr);
}

struct StmtLoweringVisitor : StmtVisitor<StmtLoweringVisitor>
{
    IRGenContext* context;

    IRBuilder* getBuilder() { return context->irBuilder; }

    void visitEmptyStmt(EmptyStmt*)
    {
        // Nothing to do.
    }

    void visitUnparsedStmt(UnparsedStmt*) { SLANG_UNEXPECTED("UnparsedStmt not supported by IR"); }

    void visitCaseStmtBase(CaseStmtBase*)
    {
        SLANG_UNEXPECTED("`case` or `default` not under `switch`");
    }

    void visitLabelStmt(LabelStmt* stmt) { lowerStmt(context, stmt->innerStmt); }

    void visitCompileTimeForStmt(CompileTimeForStmt* stmt)
    {
        // The user is asking us to emit code for the loop
        // body for each value in the given integer range.
        // For now, we will handle this by repeatedly lowering
        // the body statement, with the loop variable bound
        // to a different integer literal value each time.
        //
        // TODO: eventually we might handle this as just an
        // ordinary loop, with an `[unroll]` attribute on
        // it that we would respect.

        auto rangeBeginVal = getIntVal(stmt->rangeBeginVal);
        auto rangeEndVal = getIntVal(stmt->rangeEndVal);

        if (rangeBeginVal >= rangeEndVal)
            return;

        auto varDecl = stmt->varDecl;
        auto varType = lowerType(context, varDecl->type);

        IRGenEnv subEnvStorage;
        IRGenEnv* subEnv = &subEnvStorage;
        subEnv->outer = context->env;

        IRGenContext subContextStorage = *context;
        IRGenContext* subContext = &subContextStorage;
        subContext->env = subEnv;


        for (IntegerLiteralValue ii = rangeBeginVal; ii < rangeEndVal; ++ii)
        {
            auto constVal = getBuilder()->getIntValue(varType, ii);

            subEnv->mapDeclToValue[varDecl] = LoweredValInfo::simple(constVal);

            lowerStmt(subContext, stmt->body);
        }
    }

    // Create a basic block in the current function,
    // so that it can be used for a label.
    IRBlock* createBlock() { return getBuilder()->createBlock(); }

    /// Does the given block have a terminator?
    bool isBlockTerminated(IRBlock* block) { return block->getTerminator() != nullptr; }

    /// Emit a branch to the target block if the current
    /// block being inserted into is not already terminated.
    void emitBranchIfNeeded(IRBlock* targetBlock)
    {
        auto builder = getBuilder();
        auto currentBlock = builder->getBlock();

        // Don't emit if there is no current block.
        if (!currentBlock)
            return;

        // Don't emit if the block already has a terminator.
        if (isBlockTerminated(currentBlock))
            return;

        // The block is unterminated, so cap it off with
        // a terminator that branches to the target.
        builder->emitBranch(targetBlock);
    }

    /// Insert a block at the current location (ending
    /// the previous block with an unconditional jump
    /// if needed).
    void insertBlock(IRBlock* block)
    {
        auto builder = getBuilder();

        auto prevBlock = builder->getBlock();
        auto parentFunc = prevBlock ? prevBlock->getParent() : builder->getFunc();

        // If the previous block doesn't already have
        // a terminator instruction, then be sure to
        // emit a branch to the new block.
        emitBranchIfNeeded(block);

        // Add the new block to the function we are building,
        // and setit as the block we will be inserting into.
        parentFunc->addBlock(block);
        builder->setInsertInto(block);
    }

    // Start a new block at the current location.
    // This is just the composition of `createBlock`
    // and `insertBlock`.
    IRBlock* startBlock()
    {
        auto block = createBlock();
        insertBlock(block);
        return block;
    }

    /// Start a new block if there isn't a current
    /// block that we can append to.
    ///
    /// The `stmt` parameter is the statement we
    /// are about to emit.
    void startBlockIfNeeded(Stmt* stmt)
    {
        auto builder = getBuilder();
        auto currentBlock = builder->getBlock();

        // If there is a current block and it hasn't
        // been terminated, then we can just use that.
        if (currentBlock && !isBlockTerminated(currentBlock))
        {
            return;
        }

        // We are about to emit code *after* a terminator
        // instruction, and there is no label to allow
        // branching into this code, so whatever we are
        // about to emit is going to be unreachable.
        //
        // Let's diagnose that here just to help the user.
        //
        // TODO: We might want to have a more robust check
        // for unreachable code based on IR analysis instead,
        // at which point we'd probably disable this check.
        //
        context->getSink()->diagnose(stmt, Diagnostics::unreachableCode);

        startBlock();
    }

    /// Create a new scope end block and return the previous one.
    ///
    /// This is needed for `defer` to be aware of scopes. `preallocated` can
    /// be specified if you already have a block at the end of the scope, like
    /// in `for` loops.
    IRBlock* pushScopeBlock(IRBlock* preallocated = nullptr)
    {
        IRBlock* prevScopeEndBlock = context->scopeEndBlock;

        auto builder = getBuilder();
        context->scopeEndBlock = preallocated ? preallocated : builder->createBlock();
        return prevScopeEndBlock;
    }

    /// Pop the current scope end block and restore the previous one.
    ///
    /// This is needed for `defer` to be aware of scopes. `previous` should be
    /// the block returned from the corresponding pushScopeBlock. `preallocated`
    /// should be true if the corresponding pushScopeBlock was given a block
    /// as a parameter.
    void popScopeBlock(IRBlock* previous, bool preallocated)
    {
        if (!preallocated)
        {
            // If pushScopeBlock actually created the block, we have to insert
            // or deallocate it here. Otherwise, we assume that the caller
            // handles the end block.
            auto builder = getBuilder();
            if (context->scopeEndBlock->hasUses())
            {
                // The end of the scope was referenced, so we need to actually
                // keep it around and jump through it.
                // Move the terminator to the scope end block.
                emitBranchIfNeeded(context->scopeEndBlock);
                builder->insertBlock(context->scopeEndBlock);
                builder->setInsertInto(context->scopeEndBlock);
            }
            else
            {
                // Scope end block was left unused, so we may as well delete it.
                context->scopeEndBlock->removeAndDeallocate();
            }
        }

        context->scopeEndBlock = previous;
    }

    void visitIfStmt(IfStmt* stmt)
    {
        auto builder = getBuilder();
        startBlockIfNeeded(stmt);

        auto condExpr = stmt->predicate;
        auto thenStmt = stmt->positiveStatement;
        auto elseStmt = stmt->negativeStatement;

        auto irCond = getSimpleVal(context, lowerRValueExpr(context, condExpr));

        IRInst* ifInst = nullptr;

        if (elseStmt)
        {
            auto thenBlock = createBlock();
            auto elseBlock = createBlock();
            auto afterBlock = createBlock();

            ifInst = builder->emitIfElse(irCond, thenBlock, elseBlock, afterBlock);

            insertBlock(thenBlock);
            IRBlock* prevScopeEndBlock = pushScopeBlock(afterBlock);
            lowerStmt(context, thenStmt);
            emitBranchIfNeeded(afterBlock);

            insertBlock(elseBlock);
            lowerStmt(context, elseStmt);
            popScopeBlock(prevScopeEndBlock, true);

            insertBlock(afterBlock);
        }
        else
        {
            auto thenBlock = createBlock();
            auto afterBlock = createBlock();

            ifInst = builder->emitIf(irCond, thenBlock, afterBlock);

            insertBlock(thenBlock);

            IRBlock* prevScopeEndBlock = pushScopeBlock(afterBlock);
            lowerStmt(context, thenStmt);
            popScopeBlock(prevScopeEndBlock, true);

            insertBlock(afterBlock);
        }

        if (stmt->findModifier<FlattenAttribute>())
        {
            builder->addDecoration(ifInst, kIROp_FlattenDecoration);
        }
        if (stmt->findModifier<BranchAttribute>())
        {
            builder->addDecoration(ifInst, kIROp_BranchDecoration);
        }
    }

    void addLoopDecorations(IRInst* inst, Stmt* stmt)
    {
        if (stmt->findModifier<UnrollAttribute>())
        {
            getBuilder()->addLoopControlDecoration(inst, kIRLoopControl_Unroll);
        }
        else if (stmt->findModifier<LoopAttribute>())
        {
            getBuilder()->addLoopControlDecoration(inst, kIRLoopControl_Loop);
        }

        if (auto maxItersAttr = stmt->findModifier<MaxItersAttribute>())
        {
            auto iters = lowerVal(context, maxItersAttr->value);
            getBuilder()->addLoopMaxItersDecoration(inst, getSimpleVal(context, iters));
        }
        else if (auto inferredMaxItersAttr = stmt->findModifier<InferredMaxItersAttribute>())
        {
            getBuilder()->addLoopMaxItersDecoration(inst, inferredMaxItersAttr->value);
        }

        if (auto forceUnrollAttr = stmt->findModifier<ForceUnrollAttribute>())
        {
            getBuilder()->addLoopForceUnrollDecoration(inst, forceUnrollAttr->maxIterations);
        }
        // TODO: handle other cases here
    }

    void visitForStmt(ForStmt* stmt)
    {
        auto builder = getBuilder();
        startBlockIfNeeded(stmt);

        // The initializer clause for the statement
        // can always safetly be emitted to the current block.
        if (auto initStmt = stmt->initialStatement)
        {
            lowerStmt(context, initStmt);
        }

        // We will create blocks for the various places
        // we need to jump to inside the control flow,
        // including the blocks that will be referenced
        // by `continue` or `break` statements.
        auto loopHead = createBlock();
        auto bodyLabel = createBlock();
        auto breakLabel = createBlock();
        auto continueLabel = createBlock();

        // Register the `break` and `continue` labels so
        // that we can find them for nested statements.
        context->shared->breakLabels.add(stmt->uniqueID, breakLabel);
        context->shared->continueLabels.add(stmt->uniqueID, continueLabel);

        // Emit the branch that will start out loop,
        // and then insert the block for the head.

        auto loopInst = builder->emitLoop(loopHead, breakLabel, continueLabel);

        insertBlock(loopHead);

        // Now that we are within the header block, we
        // want to emit the expression for the loop condition:
        if (const auto condExpr = stmt->predicateExpression)
        {
            maybeEmitDebugLine(context, *this, stmt, condExpr->loc);

            auto irCondition =
                getSimpleVal(context, lowerRValueExpr(context, stmt->predicateExpression));

            // Now we want to `break` if the loop condition is false.
            builder->emitLoopTest(irCondition, bodyLabel, breakLabel);
        }

        // Emit the body of the loop
        insertBlock(bodyLabel);
        IRBlock* prevScopeEndBlock = pushScopeBlock(continueLabel);
        lowerStmt(context, stmt->statement);
        popScopeBlock(prevScopeEndBlock, true);

        if (auto inferredMaxIters = stmt->findModifier<InferredMaxItersAttribute>())
        {
            // We only use inferred max iters attribute when the loop body
            // does not modify induction var.
            auto inductionVar =
                emitDeclRef(context, inferredMaxIters->inductionVar, builder->getIntType());
            if (inductionVar.val)
            {
                int writes = 0;
                traverseUsers(
                    inductionVar.val,
                    [&](IRInst* user)
                    {
                        if (user->getOp() != kIROp_Load)
                            writes++;
                    });
                if (writes > 1)
                {
                    removeModifier(stmt, inferredMaxIters);
                }
            }
        }
        if (auto inferredMaxIters = stmt->findModifier<InferredMaxItersAttribute>())
        {
            if (auto maxIters = stmt->findModifier<MaxItersAttribute>())
            {
                if (auto constIntVal = as<ConstantIntVal>(maxIters->value))
                {
                    if (inferredMaxIters->value < constIntVal->getValue())
                    {
                        context->getSink()->diagnose(
                            maxIters,
                            Diagnostics::forLoopTerminatesInFewerIterationsThanMaxIters,
                            inferredMaxIters->value);
                    }
                }
            }
        }
        addLoopDecorations(loopInst, stmt);


        // Insert the `continue` block
        insertBlock(continueLabel);
        if (auto incrExpr = stmt->sideEffectExpression)
        {
            maybeEmitDebugLine(context, *this, stmt, incrExpr->loc);
            lowerRValueExpr(context, incrExpr);
        }

        // At the end of the body we need to jump back to the top.
        emitBranchIfNeeded(loopHead);

        // Finally we insert the label that a `break` will jump to
        insertBlock(breakLabel);
    }

    void visitWhileStmt(WhileStmt* stmt)
    {
        // Generating IR for `while` statement is similar to a
        // `for` statement, but without a lot of the complications.

        auto builder = getBuilder();
        startBlockIfNeeded(stmt);

        // We will create blocks for the various places
        // we need to jump to inside the control flow,
        // including the blocks that will be referenced
        // by `continue` or `break` statements.
        auto loopHead = createBlock();
        auto bodyLabel = createBlock();
        auto breakLabel = createBlock();

        // A `continue` inside a `while` loop always
        // jumps to the head of hte loop.
        auto continueLabel = loopHead;

        // Register the `break` and `continue` labels so
        // that we can find them for nested statements.
        context->shared->breakLabels.add(stmt->uniqueID, breakLabel);
        context->shared->continueLabels.add(stmt->uniqueID, continueLabel);

        // Emit the branch that will start out loop,
        // and then insert the block for the head.

        auto loopInst = builder->emitLoop(loopHead, breakLabel, continueLabel);

        addLoopDecorations(loopInst, stmt);

        insertBlock(loopHead);

        // Now that we are within the header block, we
        // want to emit the expression for the loop condition:
        if (auto condExpr = stmt->predicate)
        {
            maybeEmitDebugLine(context, *this, stmt, condExpr->loc);

            auto irCondition = getSimpleVal(context, lowerRValueExpr(context, condExpr));

            // Now we want to `break` if the loop condition is false.
            builder->emitLoopTest(irCondition, bodyLabel, breakLabel);
        }

        // Emit the body of the loop
        insertBlock(bodyLabel);
        IRBlock* prevScopeEndBlock = pushScopeBlock(continueLabel);
        lowerStmt(context, stmt->statement);
        popScopeBlock(prevScopeEndBlock, true);

        // At the end of the body we need to jump back to the top.
        emitBranchIfNeeded(loopHead);

        // Finally we insert the label that a `break` will jump to
        insertBlock(breakLabel);
    }

    void visitDoWhileStmt(DoWhileStmt* stmt)
    {
        // Generating IR for `do {...} while` statement is similar to a
        // `while` statement, just with the test in a different place

        auto builder = getBuilder();
        startBlockIfNeeded(stmt);

        // We will create blocks for the various places
        // we need to jump to inside the control flow,
        // including the blocks that will be referenced
        // by `continue` or `break` statements.
        auto loopHead = createBlock();
        auto testLabel = createBlock();
        auto breakLabel = createBlock();

        // A `continue` inside a `do { ... } while ( ... )` loop always
        // jumps to the loop test.
        auto continueLabel = testLabel;

        // Register the `break` and `continue` labels so
        // that we can find them for nested statements.
        context->shared->breakLabels.add(stmt->uniqueID, breakLabel);
        context->shared->continueLabels.add(stmt->uniqueID, continueLabel);

        // Emit the branch that will start out loop,
        // and then insert the block for the head.

        auto loopInst = builder->emitLoop(loopHead, breakLabel, continueLabel);

        addLoopDecorations(loopInst, stmt);

        insertBlock(loopHead);

        // Emit the body of the loop
        IRBlock* prevScopeEndBlock = pushScopeBlock(continueLabel);
        lowerStmt(context, stmt->statement);
        popScopeBlock(prevScopeEndBlock, true);

        insertBlock(testLabel);

        // Now that we are within the header block, we
        // want to emit the expression for the loop condition:
        if (auto condExpr = stmt->predicate)
        {
            maybeEmitDebugLine(context, *this, stmt, stmt->predicate->loc);

            auto irCondition = getSimpleVal(context, lowerRValueExpr(context, condExpr));

            // One thing to be careful here is that lowering irCondition
            // may create additional blocks due to short circuiting, so
            // the block we are current inserting into is not necessarily
            // the same as `testLabel`.
            //
            auto invCondition = builder->emitNot(irCondition->getDataType(), irCondition);

            // Now we want to `break` if the loop condition is false,
            // otherwise we will jump back to the head of the loop.
            //
            // We need to make sure not to reuse the break block of the loop as
            // the break/merge block of the ifelse test.
            // Therefore, we introduce a separate merge block for the loop test.
            //
            // Emit the following structure:
            //
            // [merge(mergeBlock)]
            // if (cond) goto loopHead;
            // else goto mergeBlock;
            //
            // mergeBlock:
            //   goto breakLabel;
            auto mergeBlock = builder->createBlock();
            builder->emitIfElse(invCondition, breakLabel, mergeBlock, mergeBlock);

            insertBlock(mergeBlock);
            builder->emitBranch(loopHead);
        }

        // Finally we insert the label that a `break` will jump to
        insertBlock(breakLabel);
    }

    void visitGpuForeachStmt(GpuForeachStmt* stmt)
    {
        auto builder = getBuilder();
        startBlockIfNeeded(stmt);

        auto device = getSimpleVal(context, lowerRValueExpr(context, stmt->device));
        auto gridDims = getSimpleVal(context, lowerRValueExpr(context, stmt->gridDims));

        List<IRInst*> irArgs;
        if (auto callExpr = as<InvokeExpr>(stmt->kernelCall))
        {
            irArgs.add(device);
            irArgs.add(gridDims);
            auto fref = getSimpleVal(context, lowerRValueExpr(context, callExpr->functionExpr));
            irArgs.add(fref);
            for (auto arg : callExpr->arguments)
            {
                // if a reference to dispatchThreadID, don't emit
                if (auto declRefExpr = as<DeclRefExpr>(arg))
                {
                    if (declRefExpr->declRef.getDecl() == stmt->dispatchThreadID)
                    {
                        continue;
                    }
                }
                auto irArg = getSimpleVal(context, lowerRValueExpr(context, arg));
                irArgs.add(irArg);
            }
        }
        else
        {
            SLANG_UNEXPECTED("GPUForeach parsing produced an invalid result");
        }

        builder->emitGpuForeach(irArgs);
        return;
    }

    void visitExpressionStmt(ExpressionStmt* stmt)
    {
        startBlockIfNeeded(stmt);

        // The statement evaluates an expression
        // (for side effects, one assumes) and then
        // discards the result. As such, we simply
        // lower the expression, and don't use
        // the result.
        //
        // Note that we lower using the l-value path,
        // so that an expression statement that names
        // a location (but doesn't load from it)
        // will not actually emit a load.
        lowerLValueExpr(context, stmt->expression);
    }

    void visitDeclStmt(DeclStmt* stmt)
    {
        startBlockIfNeeded(stmt);

        // For now, we lower a declaration directly
        // into the current context.
        //
        // TODO: We may want to consider whether
        // nested type/function declarations should
        // be lowered into the global scope during
        // IR generation, or whether they should
        // be lifted later (pushing capture analysis
        // down to the IR).
        //
        lowerDecl(context, stmt->decl);
    }

    void visitSeqStmt(SeqStmt* stmt)
    {
        // To lower a sequence of statements,
        // just lower each in order
        for (auto ss : stmt->stmts)
        {
            lowerStmt(context, ss);
        }
    }

    void visitBlockStmt(BlockStmt* stmt)
    {
        IRBlock* prevScopeEndBlock = pushScopeBlock(nullptr);

        // To lower a block (scope) statement, just lower its body.
        lowerStmt(context, stmt->body);

        popScopeBlock(prevScopeEndBlock, false);
    }

    void visitReturnStmt(ReturnStmt* stmt)
    {
        startBlockIfNeeded(stmt);

        // Check if this return is within a constructor.
        auto constructorDecl = as<ConstructorDecl>(context->funcDecl);

        // A `return` statement turns into a `return` instruction,
        // but we have two kinds of `return`: one for returning
        // a (non-`void`) value, and one for returning "no value"
        // (which effectively returns a value of type `void`).
        //
        if (auto expr = stmt->expression)
        {
            if (context->returnDestination.flavor != LoweredValInfo::Flavor::None)
            {
                // If this function should return via a __ref parameter, do that and return void.
                lowerRValueExprWithDestination(context, context->returnDestination, expr);
                getBuilder()->emitReturn();
                return;
            }

            if (constructorDecl)
            {
                // If this function is a constructor, but returns a value, rewrite it as
                // this = val;
                // return this;
                lowerRValueExprWithDestination(context, context->thisVal, expr);
                getBuilder()->emitReturn(getSimpleVal(context, context->thisVal));
                return;
            }

            // If the AST `return` statement had an expression, then we
            // need to lower it to the IR at this point, both to
            // compute its value and (in case we are returning a
            // `void`-typed expression) to execute its side effects.
            //
            auto loweredExpr = lowerRValueExpr(context, expr);

            // If the AST `return` statement was returning a non-`void`
            // value, then we need to emit an IR `return` of that value.
            //
            if (!expr->type.type->equals(context->astBuilder->getVoidType()))
            {
                getBuilder()->emitReturn(getSimpleVal(context, loweredExpr));
            }
            else
            {
                // If the type of the value returned was `void`, then
                // we don't want to emit an IR-level `return` with a value,
                // because that could trip up some of our back-end.
                //
                // TODO: We should eventually have only a single IR-level
                // `return` operation that always takes a value (including
                // values of type `void`), and then treat an AST `return;`
                // as equivalent to something like `return void();`.
                //
                getBuilder()->emitReturn();
            }
        }
        else
        {
            // If we hit this case, then the AST `return` was a `return;`
            // with no value, which can only occur in a function with
            // a `void` result type.
            //
            if (constructorDecl)
            {
                // If this `return` is within a NonCopyableType or an ordinary constructor,
                // then we must either simply return or `return` the instance respectively.
                if (context->returnDestination.flavor != LoweredValInfo::Flavor::None)
                {
                    // If we have a NonCopyableType constructor of the form
                    //   void ctor(inout this) { return; }
                    getBuilder()->emitReturn();
                }
                else
                {
                    // If we have an ordinary constructor of the form
                    //   Type ctor() { return; }
                    getBuilder()->emitReturn(getSimpleVal(context, context->thisVal));
                }
                return;
            }
            getBuilder()->emitReturn();
        }
    }

    void visitDeferStmt(DeferStmt* stmt)
    {
        auto builder = getBuilder();
        startBlockIfNeeded(stmt);

        IRBlock* deferBlock = builder->createBlock();
        IRBlock* mergeBlock = builder->createBlock();

        builder->emitDefer(deferBlock, mergeBlock, context->scopeEndBlock);

        builder->insertBlock(deferBlock);
        builder->setInsertInto(deferBlock);

        IRBlock* prevScopeEndBlock = pushScopeBlock(mergeBlock);
        lowerStmt(context, stmt->statement);
        popScopeBlock(prevScopeEndBlock, true);

        builder->emitBranch(mergeBlock);

        builder->insertBlock(mergeBlock);
        builder->setInsertInto(mergeBlock);
    }

    void visitDiscardStmt(DiscardStmt* stmt)
    {
        startBlockIfNeeded(stmt);
        getBuilder()->emitDiscard();
    }

    void visitBreakStmt(BreakStmt* stmt)
    {
        startBlockIfNeeded(stmt);

        // Semantic checking is responsible for finding
        // the statement taht this `break` breaks out of
        auto targetStmtID = stmt->targetOuterStmtID;
        SLANG_ASSERT(targetStmtID != BreakableStmt::kInvalidUniqueID);

        // We just need to look up the basic block that
        // corresponds to the break label for that statement,
        // and then emit an instruction to jump to it.
        IRBlock* targetBlock = nullptr;
        context->shared->breakLabels.tryGetValue(targetStmtID, targetBlock);
        SLANG_ASSERT(targetBlock);
        getBuilder()->emitBreak(targetBlock);
    }

    void visitContinueStmt(ContinueStmt* stmt)
    {
        startBlockIfNeeded(stmt);

        // Semantic checking is responsible for finding
        // the loop that this `continue` statement continues
        auto targetStmtID = stmt->targetOuterStmtID;
        SLANG_ASSERT(targetStmtID != BreakableStmt::kInvalidUniqueID);


        // We just need to look up the basic block that
        // corresponds to the continue label for that statement,
        // and then emit an instruction to jump to it.
        IRBlock* targetBlock = nullptr;
        context->shared->continueLabels.tryGetValue(targetStmtID, targetBlock);
        SLANG_ASSERT(targetBlock);
        getBuilder()->emitContinue(targetBlock);
    }

    // Lowering a `switch` statement can get pretty involved,
    // so we need to track a bit of extra data:
    struct SwitchStmtInfo
    {
        // The block that will be made to contain the `switch` statement
        IRBlock* initialBlock = nullptr;

        // The label for the `default` case, if any.
        IRBlock* defaultLabel = nullptr;

        // The label of the current "active" case block.
        IRBlock* currentCaseLabel = nullptr;

        // Has anything been emitted to the current "active" case block?
        bool anythingEmittedToCurrentCaseBlock = false;

        // The collected (value, label) pairs for
        // all the `case` statements.
        List<IRInst*> cases;
    };

    // We need a label to use for a `case` or `default` statement,
    // so either create one here, or re-use the current one if
    // that is okay.
    IRBlock* getLabelForCase(SwitchStmtInfo* info)
    {
        // Look at the "current" label we are working with.
        auto currentCaseLabel = info->currentCaseLabel;

        // If there is a current block, and it is empty,
        // then it is still a viable target (we are in
        // a case of "trivial fall-through" from the previous
        // block).
        if (currentCaseLabel && !info->anythingEmittedToCurrentCaseBlock)
        {
            return currentCaseLabel;
        }

        // Othwerise, we need to start a new block and use that.
        IRBlock* newCaseLabel = createBlock();

        // Note: if the previous block failed
        // to end with a `break`, then inserting
        // this block will append an unconditional
        // branch to the end of it that will target
        // this block.
        insertBlock(newCaseLabel);

        info->currentCaseLabel = newCaseLabel;
        info->anythingEmittedToCurrentCaseBlock = false;
        return newCaseLabel;
    }

    bool hasSwitchCases(Stmt* inStmt)
    {
        Stmt* stmt = inStmt;
        // Unwrap any surrounding `{ ... }` so we can look
        // at the statement inside.
        while (auto blockStmt = as<BlockStmt>(stmt))
        {
            stmt = blockStmt->body;
            continue;
        }

        if (auto seqStmt = as<SeqStmt>(stmt))
        {
            // Walk through the children looking for cases
            for (auto childStmt : seqStmt->stmts)
            {
                if (hasSwitchCases(childStmt))
                {
                    return true;
                }
            }
        }
        else if (const auto caseStmt = as<CaseStmt>(stmt))
        {
            return true;
        }
        else if (const auto defaultStmt = as<DefaultStmt>(stmt))
        {
            // A 'default:' is a kind of case.
            return true;
        }

        return false;
    }

    // Given a statement that appears as (or in) the body
    // of a `switch` statement
    void lowerSwitchCases(Stmt* inStmt, SwitchStmtInfo* info)
    {
        // TODO: in the general case (e.g., if we were going
        // to eventual lower to an unstructured format like LLVM),
        // the Right Way to handle C-style `switch` statements
        // is just to emit the body directly as "normal" statements,
        // and then treat `case` and `default` as special statements
        // that start a new block and register a label with the
        // enclosing `switch`.
        //
        // For now we will assume that any `case` and `default`
        // statements need to be directly nested under the `switch`,
        // and so we can find them with a simpler walk.

        Stmt* stmt = inStmt;

        // Unwrap any surrounding `{ ... }` so we can look
        // at the statement inside.
        while (auto blockStmt = as<BlockStmt>(stmt))
        {
            stmt = blockStmt->body;
            continue;
        }

        if (auto seqStmt = as<SeqStmt>(stmt))
        {
            // Walk through teh children and process each.
            for (auto childStmt : seqStmt->stmts)
            {
                lowerSwitchCases(childStmt, info);
            }
        }
        else if (auto caseStmt = as<CaseStmt>(stmt))
        {
            // A full `case` statement has a value we need
            // to test against. It is expected to be a
            // compile-time constant, so we will emit
            // it like an expression here, and then hope
            // for the best.
            //
            // TODO: figure out something cleaner.

            // Actually, one gotcha is that if we ever allow non-constant
            // expressions here (or anything that requires instructions
            // to be emitted to yield its value), then those instructions
            // need to go into an appropriate block.

            IRGenContext subContext = *context;
            IRBuilder subBuilder = *getBuilder();
            subBuilder.setInsertInto(info->initialBlock);
            subContext.irBuilder = &subBuilder;
            auto caseVal = getSimpleVal(context, lowerRValueExpr(&subContext, caseStmt->expr));

            // Figure out where we are branching to.
            auto label = getLabelForCase(info);

            // Add this `case` to the list for the enclosing `switch`.
            info->cases.add(caseVal);
            info->cases.add(label);
        }
        else if (const auto defaultStmt = as<DefaultStmt>(stmt))
        {
            auto label = getLabelForCase(info);

            // We expect to only find a single `default` stmt.
            SLANG_ASSERT(!info->defaultLabel);

            info->defaultLabel = label;
        }
        else if (const auto emptyStmt = as<EmptyStmt>(stmt))
        {
            // Special-case empty statements so they don't
            // mess up our "trivial fall-through" optimization.
        }
        else
        {
            // We have an ordinary statement, that needs to get
            // emitted to the current case block.
            if (!info->currentCaseLabel)
            {
                // It possible in full C/C++ to have statements
                // before the first `case`. Usually these are
                // unreachable, unless they start with a label.
                //
                // We'll ignore them here, figuring they are
                // dead. If we ever add `LabelStmt` then we'd
                // need to emit these statements to a dummy
                // block just in case.
            }
            else
            {
                // Emit the code to our current case block,
                // and record that we've done so.
                lowerStmt(context, stmt);
                info->anythingEmittedToCurrentCaseBlock = true;
            }
        }
    }

    void visitStageSwitchStmt(StageSwitchStmt* stmt)
    {
        if (!stmt->targetCases.getCount())
            return;

        // We will lower stage switch as a normal switch statement, so they can participate in all
        // optimizations.
        auto builder = getBuilder();
        startBlockIfNeeded(stmt);

        // First emit code to get the current stage to switch on:
        auto conditionVal = builder->emitGetCurrentStage();

        // Remember the initial block so that we can add to it
        // after we've collected all the `case`s
        auto initialBlock = builder->getBlock();

        // Next, create a block to use as the target for any `break` statements
        auto breakLabel = createBlock();

        // Register the `break` label so
        // that we can find it for nested statements.
        context->shared->breakLabels.add(stmt->uniqueID, breakLabel);

        builder->setInsertInto(initialBlock->getParent());

        // Iterate over the body of the statement, looking
        // for `case` or `default` statements:
        SwitchStmtInfo info;
        info.initialBlock = initialBlock;
        info.defaultLabel = nullptr;

        Dictionary<Stmt*, IRBlock*> mapCaseStmtToBlock;
        for (auto targetCase : stmt->targetCases)
        {
            IRBlock* caseBlock = nullptr;
            if (!mapCaseStmtToBlock.tryGetValue(targetCase->body, caseBlock))
            {
                caseBlock = builder->emitBlock();
                lowerStmt(context, targetCase->body);
                mapCaseStmtToBlock.add(targetCase->body, caseBlock);
                if (!builder->getBlock()->getTerminator())
                    builder->emitBranch(breakLabel);
            }
            if (targetCase->capability == 0)
            {
                info.defaultLabel = caseBlock;
            }
            else
            {
                auto stage = getStageFromAtom((CapabilityAtom)targetCase->capability);
                info.cases.add(builder->getIntValue(builder->getIntType(), (IRIntegerValue)stage));
                info.cases.add(caseBlock);
            }
        }

        // If the current block (the end of the last
        // `case`) is not terminated, then terminate with a
        // `break` operation.
        //
        // Double check that we aren't in the initial
        // block, so we don't get tripped up on an
        // empty `switch`.
        auto curBlock = builder->getBlock();
        if (curBlock != initialBlock)
        {
            // Is the block already terminated?
            if (!curBlock->getTerminator())
            {
                // Not terminated, so add one.
                builder->emitBreak(breakLabel);
            }
        }

        // If there was no `default` statement, then the
        // default case will just branch directly to the end.
        auto defaultLabel = info.defaultLabel ? info.defaultLabel : breakLabel;

        // Now that we've collected the cases, we are
        // prepared to emit the `switch` instruction
        // itself.
        builder->setInsertInto(initialBlock);
        builder->emitSwitch(
            conditionVal,
            breakLabel,
            defaultLabel,
            info.cases.getCount(),
            info.cases.getBuffer());

        // Finally we insert the label that a `break` will jump to
        // (and that control flow will fall through to otherwise).
        // This is the block that subsequent code will go into.
        insertBlock(breakLabel);
        context->shared->breakLabels.remove(stmt->uniqueID);
    }

    void visitTargetSwitchStmt(TargetSwitchStmt* stmt)
    {
        if (!stmt->targetCases.getCount())
            return;

        auto builder = getBuilder();
        startBlockIfNeeded(stmt);
        auto initialBlock = builder->getBlock();
        auto breakLabel = builder->createBlock();
        context->shared->breakLabels.add(stmt->uniqueID, breakLabel);
        builder->setInsertInto(initialBlock->getParent());
        List<IRInst*> args;
        args.add(breakLabel);
        Dictionary<Stmt*, IRBlock*> mapCaseStmtToBlock;
        for (auto targetCase : stmt->targetCases)
        {
            IRBlock* caseBlock = nullptr;
            if (!mapCaseStmtToBlock.tryGetValue(targetCase->body, caseBlock))
            {
                caseBlock = builder->emitBlock();
                lowerStmt(context, targetCase->body);
                mapCaseStmtToBlock.add(targetCase->body, caseBlock);
                if (!builder->getBlock()->getTerminator())
                    builder->emitBranch(breakLabel);
            }
            args.add(builder->getIntValue(builder->getIntType(), targetCase->capability));
            args.add(caseBlock);
        }
        context->shared->breakLabels.remove(stmt->uniqueID);
        builder->setInsertInto(initialBlock);

        auto parentFunc = initialBlock->getParent();
        parentFunc->addBlock(breakLabel);

        builder->emitIntrinsicInst(
            nullptr,
            kIROp_TargetSwitch,
            (UInt)args.getCount(),
            args.getBuffer());

        builder->setInsertInto(breakLabel);
    }

    void visitTargetCaseStmt(TargetCaseStmt*) { SLANG_UNREACHABLE("lowering target case"); }

    void visitIntrinsicAsmStmt(IntrinsicAsmStmt* stmt)
    {
        auto builder = getBuilder();
        ShortList<IRInst*> args;
        args.add(builder->getStringValue(stmt->asmText.getUnownedSlice()));
        for (auto argExpr : stmt->args)
        {
            if (auto typetype = as<TypeType>(argExpr->type))
            {
                auto type = lowerType(context, typetype->getType());
                args.add(type);
            }
            else
            {
                auto argVal = lowerRValueExpr(context, argExpr);
                args.add(argVal.val);
            }
        }
        builder->emitIntrinsicInst(
            nullptr,
            kIROp_GenericAsm,
            args.getCount(),
            args.getArrayView().getBuffer());
    }

    void visitSwitchStmt(SwitchStmt* stmt)
    {
        auto builder = getBuilder();
        startBlockIfNeeded(stmt);

        // Given a statement:
        //
        //      switch( CONDITION )
        //      {
        //      case V0:
        //          S0;
        //          break;
        //
        //      case V1:
        //      default:
        //          S1;
        //          break;
        //      }
        //
        // we want to generate IR like:
        //
        //      let %c = <CONDITION>;
        //      switch %c,          // value to switch on
        //          %breakLabel,    // join point (and break target)
        //          %s1,            // default label
        //          %v0,            // first case value
        //          %s0,            // first case label
        //          %v1,            // second case value
        //          %s1             // second case label
        //  s0:
        //      <S0>
        //      break %breakLabel
        //  s1:
        //      <S1>
        //      break %breakLabel
        //  breakLabel:
        //

        // First emit code to compute the condition:
        auto conditionVal = getSimpleVal(context, lowerRValueExpr(context, stmt->condition));

        // Check for any cases or default.
        if (!hasSwitchCases(stmt->body))
        {
            // If we don't have any case/default then nothing inside switch can be executed (other
            // than condition) so we are done.
            return;
        }

        // Remember the initial block so that we can add to it
        // after we've collected all the `case`s
        auto initialBlock = builder->getBlock();

        // Next, create a block to use as the target for any `break` statements
        auto breakLabel = createBlock();

        // Register the `break` label so
        // that we can find it for nested statements.
        context->shared->breakLabels.add(stmt->uniqueID, breakLabel);

        builder->setInsertInto(initialBlock->getParent());

        // Iterate over the body of the statement, looking
        // for `case` or `default` statements:
        SwitchStmtInfo info;
        info.initialBlock = initialBlock;
        info.defaultLabel = nullptr;
        lowerSwitchCases(stmt->body, &info);

        // TODO: once we've discovered the cases, we should
        // be able to make a quick pass over the list and eliminate
        // any cases that have the exact same label as the `default`
        // case, since these don't actually need to be represented.

        // If the current block (the end of the last
        // `case`) is not terminated, then terminate with a
        // `break` operation.
        //
        // Double check that we aren't in the initial
        // block, so we don't get tripped up on an
        // empty `switch`.
        auto curBlock = builder->getBlock();
        if (curBlock != initialBlock)
        {
            // Is the block already terminated?
            if (!curBlock->getTerminator())
            {
                // Not terminated, so add one.
                builder->emitBreak(breakLabel);
            }
        }

        // If there was no `default` statement, then the
        // default case will just branch directly to the end.
        auto defaultLabel = info.defaultLabel ? info.defaultLabel : breakLabel;

        // Now that we've collected the cases, we are
        // prepared to emit the `switch` instruction
        // itself.
        builder->setInsertInto(initialBlock);
        auto switchInst = builder->emitSwitch(
            conditionVal,
            breakLabel,
            defaultLabel,
            info.cases.getCount(),
            info.cases.getBuffer());

        // Finally we insert the label that a `break` will jump to
        // (and that control flow will fall through to otherwise).
        // This is the block that subsequent code will go into.
        insertBlock(breakLabel);
        context->shared->breakLabels.remove(stmt->uniqueID);

        // If there is the branch attribute output the IR decoration
        if (stmt->hasModifier<BranchAttribute>())
        {
            builder->addDecoration(switchInst, kIROp_BranchDecoration);
        }
    }
};

IRInst* getOrEmitDebugSource(IRGenContext* context, PathInfo path)
{
    if (auto result = context->shared->mapSourcePathToDebugSourceInst.tryGetValue(path.foundPath))
        return *result;

    ComPtr<ISlangBlob> outBlob;
    if (path.hasFileFoundPath())
    {
        context->getLinkage()->getFileSystemExt()->loadFile(
            path.foundPath.getBuffer(),
            outBlob.writeRef());
    }
    UnownedStringSlice content;
    if (outBlob)
        content = UnownedStringSlice((char*)outBlob->getBufferPointer(), outBlob->getBufferSize());
    IRBuilder builder(*context->irBuilder);
    builder.setInsertInto(context->irBuilder->getModule());
    auto debugSrcInst = builder.emitDebugSource(path.foundPath.getUnownedSlice(), content);
    context->shared->mapSourcePathToDebugSourceInst[path.foundPath] = debugSrcInst;
    return debugSrcInst;
}

void maybeEmitDebugLine(
    IRGenContext* context,
    StmtLoweringVisitor& visitor,
    Stmt* stmt,
    SourceLoc loc)
{
    if (!context->includeDebugInfo)
        return;
    if (as<EmptyStmt>(stmt))
        return;
    if (!loc.isValid())
        loc = stmt->loc;
    auto sourceManager = context->getLinkage()->getSourceManager();
    auto sourceView = context->getLinkage()->getSourceManager()->findSourceView(loc);
    if (!sourceView)
        return;

    IRInst* debugSourceInst = nullptr;
    auto humaneLoc =
        context->getLinkage()->getSourceManager()->getHumaneLoc(loc, SourceLocType::Emit);

    // Do a best-effort attempt to retrieve the nominal source file.
    auto pathInfo = sourceView->getPathInfo(loc, SourceLocType::Emit);

    // If the source file path correspond to an existing SourceFile in the source manager, use it.
    auto source = sourceManager->findSourceFileByPathRecursively(pathInfo.foundPath);
    if (!source)
        source = sourceManager->findSourceFile(pathInfo.getMostUniqueIdentity());
    if (source)
    {
        context->shared->mapSourceFileToDebugSourceInst.tryGetValue(source, debugSourceInst);
    }
    // If the source manager does not have an entry for the corresponding file name, make sure we
    // still emit an source file entry in the spirv module.
    if (!debugSourceInst)
    {
        debugSourceInst = getOrEmitDebugSource(context, pathInfo);
    }
    visitor.startBlockIfNeeded(stmt);
    context->irBuilder->emitDebugLine(
        debugSourceInst,
        humaneLoc.line,
        humaneLoc.line,
        humaneLoc.column,
        humaneLoc.column + 1);
}

void maybeAddDebugLocationDecoration(IRGenContext* context, IRInst* inst)
{
    if (!context->includeDebugInfo)
        return;
    auto sourceView = context->getLinkage()->getSourceManager()->findSourceView(inst->sourceLoc);
    if (!sourceView)
        return;
    auto source = sourceView->getSourceFile();
    IRInst* debugSourceInst = nullptr;
    if (context->shared->mapSourceFileToDebugSourceInst.tryGetValue(source, debugSourceInst))
    {
        auto humaneLoc = context->getLinkage()->getSourceManager()->getHumaneLoc(
            inst->sourceLoc,
            SourceLocType::Emit);
        context->irBuilder
            ->addDebugLocationDecoration(inst, debugSourceInst, humaneLoc.line, humaneLoc.column);
    }
}

void lowerStmt(IRGenContext* context, Stmt* stmt)
{
    IRBuilderSourceLocRAII sourceLocInfo(context->irBuilder, stmt->loc);

    StmtLoweringVisitor visitor;
    visitor.context = context;

    try
    {
        maybeEmitDebugLine(context, visitor, stmt, stmt->loc);
        visitor.dispatch(stmt);
    }
    // Don't emit any context message for an explicit `AbortCompilationException`
    // because it should only happen when an error is already emitted.
    catch (const AbortCompilationException&)
    {
        throw;
    }
    catch (...)
    {
        context->getSink()->noteInternalErrorLoc(stmt->loc);
        throw;
    }
}

/// Create and return a mutable temporary initialized with `val`
static LoweredValInfo moveIntoMutableTemp(IRGenContext* context, LoweredValInfo const& val)
{
    IRInst* irVal = getSimpleVal(context, val);
    auto type = irVal->getDataType();
    auto var = createVar(context, type);

    assign(context, var, LoweredValInfo::simple(irVal));
    return var;
}

LoweredValInfo tryGetAddress(
    IRGenContext* context,
    LoweredValInfo const& inVal,
    TryGetAddressMode mode)
{
    LoweredValInfo val = inVal;

    switch (val.flavor)
    {
    case LoweredValInfo::Flavor::Ptr:
        // The `Ptr` case means that we already have an IR value with
        // the address of our value. Easy!
        return val;

    case LoweredValInfo::Flavor::BoundStorage:
        {
            // If we are are trying to turn a subscript operation like `buffer[index]`
            // into a pointer, then we need to find a `ref` accessor declared
            // as part of the subscript operation being referenced.
            //
            auto subscriptInfo = val.getBoundStorageInfo();

            // We don't want to immediately bind to a `ref` accessor if there is
            // a `set` accessor available, unless we are in an "aggressive" mode
            // where we really want/need a pointer to be able to make progress.
            //
            if (mode != TryGetAddressMode::Aggressive && getMembersOfType<SetterDecl>(
                                                             context->astBuilder,
                                                             subscriptInfo->declRef,
                                                             MemberFilterStyle::Instance)
                                                             .isNonEmpty())
            {
                // There is a setter that we should consider using,
                // so don't go and aggressively collapse things just yet.
                return val;
            }

            auto refAccessors = getMembersOfType<RefAccessorDecl>(
                context->astBuilder,
                subscriptInfo->declRef,
                MemberFilterStyle::Instance);
            if (refAccessors.isNonEmpty())
            {
                auto refAccessor = *refAccessors.begin();

                // The `ref` accessor will return a pointer to the value, so
                // we need to reflect that in the type of our `call` instruction.
                IRType* ptrType = context->irBuilder->getPtrType(subscriptInfo->type);

                LoweredValInfo refVal = _emitCallToAccessor(
                    context,
                    ptrType,
                    refAccessor,
                    subscriptInfo->base,
                    subscriptInfo->additionalArgs);

                // The result from the call should be a pointer, and it
                // is the address that we wanted in the first place.
                return LoweredValInfo::ptr(getSimpleVal(context, refVal));
            }

            // Otherwise, there was no `ref` accessor, and so it is not possible
            // to materialize this location into a pointer for whatever purpose
            // we have in mind (e.g., passing it to an atomic operation).
        }
        break;

    case LoweredValInfo::Flavor::BoundMember:
        {
            auto boundMemberInfo = val.getBoundMemberInfo();

            // If we hit this case, then it means that we have a reference
            // to a single field in something, but for whatever reason the
            // higher-level logic was not able to turn it into a pointer
            // already (maybe the base value for the field reference is
            // a `BoundStorage`, etc.).
            //
            // We need to read the entire base value out, modify the field
            // we care about, and then write it back.

            auto declRef = boundMemberInfo->declRef;
            if (auto fieldDeclRef = declRef.as<VarDecl>())
            {
                auto baseVal = boundMemberInfo->base;
                auto basePtr = tryGetAddress(context, baseVal, TryGetAddressMode::Aggressive);

                return extractField(context, boundMemberInfo->type, basePtr, fieldDeclRef);
            }
        }
        break;

    case LoweredValInfo::Flavor::SwizzledLValue:
        {
            auto originalSwizzleInfo = val.getSwizzledLValueInfo();
            auto originalBase = originalSwizzleInfo->base;

            UInt elementCount = (UInt)originalSwizzleInfo->elementIndices.getCount();

            auto newBase = tryGetAddress(context, originalBase, TryGetAddressMode::Aggressive);
            if (newBase.flavor == LoweredValInfo::Flavor::Ptr && elementCount == 1)
            {
                // A special case is when we have a single element swizzle,
                // we can just emit an element address.
                auto elementPtr = context->irBuilder->emitElementAddress(
                    newBase.val,
                    originalSwizzleInfo->elementIndices[0]);
                return LoweredValInfo::ptr(elementPtr);
            }

            RefPtr<SwizzledLValueInfo> newSwizzleInfo = new SwizzledLValueInfo();
            context->shared->extValues.add(newSwizzleInfo);

            newSwizzleInfo->base = newBase;
            newSwizzleInfo->type = originalSwizzleInfo->type;
            newSwizzleInfo->elementIndices.setCount(elementCount);
            for (UInt ee = 0; ee < elementCount; ++ee)
                newSwizzleInfo->elementIndices[ee] = originalSwizzleInfo->elementIndices[ee];

            return LoweredValInfo::swizzledLValue(newSwizzleInfo);
        }
        break;

    // TODO(Ellie): There's an uncomfortable level of duplication here...
    case LoweredValInfo::Flavor::SwizzledMatrixLValue:
        {
            auto originalSwizzleInfo = val.getSwizzledMatrixLValueInfo();
            auto originalBase = originalSwizzleInfo->base;

            UInt elementCount = originalSwizzleInfo->elementCount;

            auto newBase = tryGetAddress(context, originalBase, TryGetAddressMode::Aggressive);
            RefPtr<SwizzledMatrixLValueInfo> newSwizzleInfo = new SwizzledMatrixLValueInfo();
            context->shared->extValues.add(newSwizzleInfo);

            newSwizzleInfo->base = newBase;
            newSwizzleInfo->type = originalSwizzleInfo->type;
            newSwizzleInfo->elementCount = elementCount;
            for (UInt ee = 0; ee < elementCount; ++ee)
            {
                newSwizzleInfo->elementCoords[ee] = originalSwizzleInfo->elementCoords[ee];
            }

            return LoweredValInfo::swizzledMatrixLValue(newSwizzleInfo);
        }
        break;
    case LoweredValInfo::Flavor::ImplicitCastedLValue:
        {
            auto info = val.getImplicitCastedLValue();
            auto baseAddr = tryGetAddress(context, info->base, TryGetAddressMode::Aggressive);
            if (baseAddr.flavor == LoweredValInfo::Flavor::Ptr)
            {
                IRInst* result = nullptr;
                if (info->lValueType == kParameterDirection_InOut)
                    result = context->irBuilder->emitInOutImplicitCast(
                        context->irBuilder->getPtrType(info->type),
                        baseAddr.val);
                else
                    result = context->irBuilder->emitOutImplicitCast(
                        context->irBuilder->getPtrType(info->type),
                        baseAddr.val);
                return LoweredValInfo::ptr(result);
            }
        }
        break;
        // TODO: are there other cases we need to handled here?

    default:
        break;
    }

    // If none of the special cases above applied, then we werent' able to make
    // this value into a pointer, and we should just return it as-is.
    return val;
}

IRInst* getAddress(IRGenContext* context, LoweredValInfo const& inVal, SourceLoc diagnosticLocation)
{
    LoweredValInfo val = tryGetAddress(context, inVal, TryGetAddressMode::Aggressive);

    if (val.flavor == LoweredValInfo::Flavor::Ptr)
    {
        return val.val;
    }

    context->getSink()->diagnose(diagnosticLocation, Diagnostics::invalidLValueForRefParameter);
    return nullptr;
}

void assignExpr(
    IRGenContext* context,
    const LoweredValInfo& inLeft,
    Expr* rightExpr,
    SourceLoc assignmentLoc)
{
    auto left = tryGetAddress(context, inLeft, TryGetAddressMode::Default);
    IRBuilderSourceLocRAII locRAII(context->irBuilder, assignmentLoc);
    switch (left.flavor)
    {
    case LoweredValInfo::Flavor::Ptr:
        {
            lowerRValueExprWithDestination(context, left, rightExpr);
        }
        break;
    default:
        {
            auto right = lowerRValueExpr(context, rightExpr);
            assign(context, inLeft, right);
        }
        break;
    }
}

void assign(IRGenContext* context, LoweredValInfo const& inLeft, LoweredValInfo const& inRight)
{
    LoweredValInfo left = inLeft;
    LoweredValInfo right = inRight;

    // Before doing the case analysis on the shape of the `left` value,
    // we might as well go ahead and see if we can coerce it into
    // a simple pointer, since that would make our life a lot easier
    // when handling complex cases.
    //
    left = tryGetAddress(context, left, TryGetAddressMode::Default);

    auto builder = context->irBuilder;

    // If there's a single element, just emit a regular store, otherwise
    // proceed with a swizzle store
    auto swizzledStore =
        [builder](IRInst* dest, IRInst* source, UInt elementCount, uint32_t const* elementIndices)
    {
        if (elementCount == 1)
        {
            return builder->emitStore(builder->emitElementAddress(dest, elementIndices[0]), source);
        }
        return builder->emitSwizzledStore(dest, source, elementCount, elementIndices);
    };

top:
    switch (left.flavor)
    {
    case LoweredValInfo::Flavor::Ptr:
        {
            // The `left` value is just a pointer, so we can emit
            // a store to it directly.
            //
            if (as<IRAtomicType>(tryGetPointedToType(builder, left.val->getDataType())))
            {
                builder->emitAtomicStore(
                    left.val,
                    getSimpleVal(context, right),
                    builder->getIntValue(builder->getIntType(), kIRMemoryOrder_Relaxed));
            }
            else
            {
                builder->emitStore(left.val, getSimpleVal(context, right));
            }
        }
        break;

    case LoweredValInfo::Flavor::SwizzledLValue:
        {
            // The `left` value is of the form `<base>.<swizzleElements>`.
            // How we will handle this depends on what `base` looks like:
            auto swizzleInfo = left.getSwizzledLValueInfo();
            auto loweredBase = swizzleInfo->base;

            // Note that the call to `tryGetAddress` at the start should
            // ensure that `loweredBase` has been simplified as much as
            // possible (e.g., if it is possible to turn it into a
            // `LoweredValInfo::ptr()` then that will have been done).

            switch (loweredBase.flavor)
            {
            default:
                {
                    // Our fallback position is to lower via a temporary, e.g.:
                    //
                    //      float4 tmp = <base>;
                    //      tmp.xyz = float3(...);
                    //      <base> = tmp;
                    //

                    // Load from the base value
                    IRInst* irLeftVal = getSimpleVal(context, loweredBase);

                    // Extract a simple value for the right-hand side
                    IRInst* irRightVal = getSimpleVal(context, right);

                    // Apply the swizzle
                    IRInst* irSwizzled = builder->emitSwizzleSet(
                        irLeftVal->getDataType(),
                        irLeftVal,
                        irRightVal,
                        (UInt)swizzleInfo->elementIndices.getCount(),
                        swizzleInfo->elementIndices.getArrayView().getBuffer());

                    // And finally, store the value back where we got it.
                    //
                    // Note: this is effectively a recursive call to
                    // `assign()`, so we do a simple tail-recursive call here.
                    left = loweredBase;
                    right = LoweredValInfo::simple(irSwizzled);
                    goto top;
                }
                break;

            case LoweredValInfo::Flavor::Ptr:
                {
                    // We are writing through a pointer, which might be
                    // pointing into a UAV or other memory resource, so
                    // we can't introduce use a temporary like the case
                    // above, because then we would read and write bytes
                    // that are not strictly required for the store.
                    //
                    // Note that the messy case of a "swizzle of a swizzle"
                    // was handled already in lowering of a `SwizzleExpr`,
                    // so that we don't need to deal with that case here.
                    //
                    // TODO: we may need to consider whether there is
                    // enough value in a masked store like this to keep
                    // it around, in comparison to a simpler model where
                    // we simply form a pointer to each of the vector
                    // elements and write to them individually.
                    IRInst* irRightVal = getSimpleVal(context, right);
                    swizzledStore(
                        loweredBase.val,
                        irRightVal,
                        (UInt)swizzleInfo->elementIndices.getCount(),
                        swizzleInfo->elementIndices.getArrayView().getBuffer());
                }
                break;
            }
        }
        break;

    case LoweredValInfo::Flavor::SwizzledMatrixLValue:
        {
            // The `left` value is of the form `<base>.<swizzleElements>`.
            // How we will handle this depends on what `base` looks like:
            auto swizzleInfo = left.getSwizzledMatrixLValueInfo();
            auto loweredBase = swizzleInfo->base;

            IRInst* irRightVal = getSimpleVal(context, right);

            const UInt maxRowIndex = 4;
            const UInt maxCols = 4; // swizzleInfo->elementCount;

            // Sort the swizzle elements according to the row to which they
            // write.
            // Using row-major terminology

            // The number of element writes in each row
            UInt rowSizes[maxRowIndex] = {};
            // The columns being written to in each row
            uint32_t rowWrites[maxRowIndex][maxCols];
            // The RHS element indices being written in each row
            UInt rowIndices[maxRowIndex][maxCols];
            for (UInt i = 0; i < swizzleInfo->elementCount; ++i)
            {
                const auto& c = swizzleInfo->elementCoords[i];
                auto& rowSize = rowSizes[c.row];
                rowWrites[c.row][rowSize] = (uint32_t)c.col;
                rowIndices[c.row][rowSize] = i;
                ++rowSize;
            }

            const auto rElemType = composeGetters<IRType>(
                irRightVal,
                &IRInst::getDataType,
                &IRVectorType::getElementType);

            switch (loweredBase.flavor)
            {
            case LoweredValInfo::Flavor::Ptr:
                {
                    // Matrix swizzle writes are implemented as several vector swizzle writes
                    for (UInt r = 0; r < maxRowIndex; ++r)
                    {
                        // Skip if we have nothing in this row
                        if (rowSizes[r] == 0)
                        {
                            continue;
                        }
                        const auto rowAddr = builder->emitElementAddress(loweredBase.val, r);
                        // Only select the RHS elements if it's a vector
                        const auto rSwizzled =
                            rElemType ? builder->emitSwizzle(
                                            builder->getVectorType(rElemType, rowSizes[r]),
                                            irRightVal,
                                            rowSizes[r],
                                            rowIndices[r])
                                      : irRightVal;
                        swizzledStore(rowAddr, rSwizzled, rowSizes[r], rowWrites[r]);
                    }
                }
                break;
            default:
                {
                    // As above, our fallback position is to lower via a
                    // temporary, e.g.:
                    //
                    //      float4x3 tmp = <base>;
                    //      tmp[0].xzy = float3(...);
                    //      tmp[1].yxz = float3(...);
                    //      tmp[4].yzx = float3(...);
                    //      <base> = tmp;
                    //
                    // Create a variable, and use the ptr writing matrix
                    // swizzle assignment above to fill it, then write that back
                    // to the l value. This approach generates the neatest IR
                    const auto beforeLValue = getSimpleVal(context, loweredBase);
                    const auto type = beforeLValue->getDataType();

                    // Store our initial lvalue in tmp
                    const auto tmpVar = builder->emitVar(type);
                    builder->emitStore(tmpVar, beforeLValue);

                    // Make a new swizzle write to write into this pointer
                    auto nextSwizzleInfo = left.getSwizzledMatrixLValueInfo();
                    SwizzledMatrixLValueInfo nextInfo = *nextSwizzleInfo;
                    nextInfo.base = LoweredValInfo::ptr(tmpVar);

                    // Perform that swizzling assignment
                    assign(context, LoweredValInfo::swizzledMatrixLValue(&nextInfo), right);

                    // Write (non-swizzled) into the l value
                    left = loweredBase;
                    right = LoweredValInfo::ptr(tmpVar);
                    goto top;
                }
                break;
            }
        }
        break;

    case LoweredValInfo::Flavor::BoundStorage:
        {
            // The `left` value refers to a subscript operation on
            // a resource type, bound to particular arguments, e.g.:
            // `someStructuredBuffer[index]`.
            //
            // When storing to such a value, we need to emit a call
            // to the appropriate builtin "setter" accessor, if there
            // is one, and then fall back to a `ref` accessor if
            // there is no setter.
            //
            auto subscriptInfo = left.getBoundStorageInfo();

            // Search for an appropriate "setter" declaration
            auto setters = getMembersOfType<SetterDecl>(
                context->astBuilder,
                subscriptInfo->declRef,
                MemberFilterStyle::Instance);
            if (setters.isNonEmpty())
            {
                auto setter = *setters.begin();

                auto allArgs = subscriptInfo->additionalArgs;

                // Note: here we are assuming that all setters take
                // the new-value parameter as an `in` rather than
                // as any kind of reference.
                //
                // TODO: If we add support for something like `const&`
                // for input parameters, we might have to deal with
                // that here.
                //
                addInArg(context, &allArgs, right);

                _emitCallToAccessor(
                    context,
                    builder->getVoidType(),
                    setter,
                    subscriptInfo->base,
                    allArgs);
                return;
            }

            auto refAccessors = getMembersOfType<RefAccessorDecl>(
                context->astBuilder,
                subscriptInfo->declRef,
                MemberFilterStyle::Instance);
            if (refAccessors.isNonEmpty())
            {
                auto refAccessor = *refAccessors.begin();

                // The `ref` accessor will return a pointer to the value, so
                // we need to reflect that in the type of our `call` instruction.
                IRType* ptrType = context->irBuilder->getPtrType(subscriptInfo->type);

                LoweredValInfo refVal = _emitCallToAccessor(
                    context,
                    ptrType,
                    refAccessor,
                    subscriptInfo->base,
                    subscriptInfo->additionalArgs);

                // The result from the call needs to be implicitly dereferenced,
                // so that it can work as an l-value of the desired result type.
                left = LoweredValInfo::ptr(getSimpleVal(context, refVal));

                // Tail-recursively attempt assignment again on the new l-value.
                goto top;
            }

            // No setter found? Then we have an error!
            SLANG_UNEXPECTED("no setter found");
            break;
        }
        break;

    case LoweredValInfo::Flavor::BoundMember:
        {
            auto boundMemberInfo = left.getBoundMemberInfo();

            // If we hit this case, then it means that we are trying to set
            // a single field in someting that is not atomically set-able.
            // (e.g., an element of a value where the `subscript` operation
            // has `get` and `set` but not a `ref` accessor).
            //
            // We need to read the entire base value out, modify the field
            // we care about, and then write it back.

            auto declRef = boundMemberInfo->declRef;
            if (auto fieldDeclRef = declRef.as<VarDecl>())
            {
                // materialize the base value and move it into
                // a mutable temporary if needed
                auto baseVal = boundMemberInfo->base;
                auto tempVal = moveIntoMutableTemp(context, baseVal);

                // extract the field l-value out of the temporary
                auto tempFieldVal =
                    extractField(context, boundMemberInfo->type, tempVal, fieldDeclRef);

                // assign to the field of the temporary l-value
                assign(context, tempFieldVal, right);

                // write back the modified temporary to the base l-value
                assign(context, baseVal, tempVal);

                return;
            }
            else
            {
                SLANG_UNEXPECTED("handled member flavor");
            }
        }
        break;

    case LoweredValInfo::Flavor::ExtractedExistential:
        {
            // The `left` value is the result of opening an existential.
            //
            auto leftInfo = left.getExtractedExistentialValInfo();
            auto existentialVal = leftInfo->existentialVal;

            // The actual desitnation we need to store into is the
            // existential value itself.
            //
            left = existentialVal;

            // The `right` value must be of the same concrete type as
            // the opened value, but the new destination is of the
            // original existential type, so we need to wrap it up
            // appropriately.
            //
            right = LoweredValInfo::simple(builder->emitMakeExistential(
                leftInfo->existentialType,
                getSimpleVal(context, right),
                leftInfo->witnessTable));

            goto top;
        }
        break;

    case LoweredValInfo::Flavor::ImplicitCastedLValue:
        {
            auto leftInfo = left.getImplicitCastedLValue();
            left = leftInfo->base;
            auto rightVal = getSimpleVal(context, right);
            right = LoweredValInfo::simple(builder->emitCast(leftInfo->type, rightVal));
            goto top;
        }
        break;

    default:
        SLANG_UNIMPLEMENTED_X("assignment");
        break;
    }
}

struct DeclLoweringVisitor : DeclVisitor<DeclLoweringVisitor, LoweredValInfo>
{
    IRGenContext* context;

    DiagnosticSink* getSink() { return context->getSink(); }

    IRBuilder* getBuilder() { return context->irBuilder; }

    LoweredValInfo visitDeclBase(DeclBase* /*decl*/)
    {
        SLANG_UNIMPLEMENTED_X("decl catch-all");
        UNREACHABLE_RETURN(LoweredValInfo());
    }

    LoweredValInfo visitDecl(Decl* /*decl*/)
    {
        SLANG_UNIMPLEMENTED_X("decl catch-all");
        UNREACHABLE_RETURN(LoweredValInfo());
    }

    LoweredValInfo visitExtensionDecl(ExtensionDecl* decl)
    {
        for (auto& member : decl->members)
            ensureDecl(context, member);
        return LoweredValInfo();
    }

#define IGNORED_CASE(NAME) \
    LoweredValInfo visit##NAME(NAME*) { return LoweredValInfo(); }

    IGNORED_CASE(ImportDecl)
    IGNORED_CASE(IncludeDecl)
    IGNORED_CASE(ImplementingDecl)
    IGNORED_CASE(UsingDecl)
    IGNORED_CASE(SyntaxDecl)
    IGNORED_CASE(AttributeDecl)
    IGNORED_CASE(NamespaceDecl)
    IGNORED_CASE(ModuleDeclarationDecl)
    IGNORED_CASE(FileDecl)
    IGNORED_CASE(RequireCapabilityDecl)

#undef IGNORED_CASE

    void getAllEntryPointsNoOverride(List<IRInst*>& entryPoints)
    {
        if (entryPoints.getCount() != 0)
            return;
        for (const auto d : context->irBuilder->getModule()->getModuleInst()->getGlobalInsts())
            if (d->findDecoration<IREntryPointDecoration>())
                entryPoints.add(d);
    }

    LoweredValInfo visitEmptyDecl(EmptyDecl* decl)
    {
        bool verifyComputeDerivativeGroupModifier = false;
        List<IRInst*> entryPoints{};
        for (const auto modifier : decl->modifiers)
        {
            if (const auto layoutLocalSizeAttr = as<GLSLLayoutLocalSizeAttribute>(modifier))
            {
                verifyComputeDerivativeGroupModifier = true;
                getAllEntryPointsNoOverride(entryPoints);

                LoweredValInfo extents[3];

                for (int i = 0; i < 3; ++i)
                {
                    extents[i] = layoutLocalSizeAttr->specConstExtents[i]
                                     ? emitDeclRef(
                                           context,
                                           layoutLocalSizeAttr->specConstExtents[i],
                                           lowerType(
                                               context,
                                               getType(
                                                   context->astBuilder,
                                                   layoutLocalSizeAttr->specConstExtents[i])))
                                     : lowerVal(context, layoutLocalSizeAttr->extents[i]);
                }

                for (auto d : entryPoints)
                    as<IRNumThreadsDecoration>(getBuilder()->addNumThreadsDecoration(
                        d,
                        getSimpleVal(context, extents[0]),
                        getSimpleVal(context, extents[1]),
                        getSimpleVal(context, extents[2])));
            }
            else if (as<GLSLLayoutDerivativeGroupQuadAttribute>(modifier))
            {
                verifyComputeDerivativeGroupModifier = true;
                getAllEntryPointsNoOverride(entryPoints);
                for (auto d : entryPoints)
                    getBuilder()->addSimpleDecoration<IRDerivativeGroupQuadDecoration>(d);
            }
            else if (as<GLSLLayoutDerivativeGroupLinearAttribute>(modifier))
            {
                verifyComputeDerivativeGroupModifier = true;
                getAllEntryPointsNoOverride(entryPoints);
                for (auto d : entryPoints)
                    getBuilder()->addSimpleDecoration<IRDerivativeGroupLinearDecoration>(d);
            }
        }

        if (!verifyComputeDerivativeGroupModifier)
            return LoweredValInfo();
        for (auto d : entryPoints)
        {
            verifyComputeDerivativeGroupModifiers(
                getSink(),
                decl->loc,
                d->findDecoration<IRDerivativeGroupQuadDecoration>(),
                d->findDecoration<IRDerivativeGroupLinearDecoration>(),
                d->findDecoration<IRNumThreadsDecoration>());
        }

        return LoweredValInfo();
    }

    void ensureInsertAtGlobalScope(IRBuilder* builder)
    {
        auto inst = builder->getInsertLoc().getInst();
        if (inst->getOp() == kIROp_Module)
            return;

        while (inst && inst->getParent() && inst->getParent()->getOp() != kIROp_Module)
        {
            inst = inst->getParent();
        }
        if (inst)
        {
            builder->setInsertBefore(inst);
        }
    }

    LoweredValInfo visitTypeDefDecl(TypeDefDecl* decl)
    {
        // A type alias declaration may be generic, if it is
        // nested under a generic type/function/etc.
        //
        NestedContext nested(this);
        auto subBuilder = nested.getBuilder();
        auto subContext = nested.getContext();

        ensureInsertAtGlobalScope(nested.getBuilder());

        IRGeneric* outerGeneric = emitOuterGenerics(subContext, decl, decl);

        // TODO: if a type alias declaration can have linkage,
        // we will need to lower it to some kind of global
        // value in the IR so that we can attach a name to it.
        //
        // For now, we can only attach a name *if* the type
        // alias is somehow generic.
        if (outerGeneric)
        {
            addLinkageDecoration(context, outerGeneric, decl);
        }

        auto type = lowerType(subContext, decl->type.type);

        return LoweredValInfo::simple(finishOuterGenerics(subBuilder, type, outerGeneric));
    }

    LoweredValInfo visitGenericTypeParamDecl(GenericTypeParamDecl* /*decl*/)
    {
        return LoweredValInfo();
    }

    LoweredValInfo visitGenericTypeConstraintDecl(GenericTypeConstraintDecl* decl)
    {
        // This might be a type constraint on an associated type,
        // in which case it should lower as the key for that
        // interface requirement.
        if (auto assocTypeDecl = as<AssocTypeDecl>(decl->parentDecl))
        {
            // TODO: might need extra steps if we ever allow
            // generic associated types.


            if (const auto interfaceDecl = as<InterfaceDecl>(assocTypeDecl->parentDecl))
            {
                // Okay, this seems to be an interface rquirement, and
                // we should lower it as such.
                return LoweredValInfo::simple(getInterfaceRequirementKey(decl));
            }
        }

        if (const auto globalGenericParamDecl = as<GlobalGenericParamDecl>(decl->parentDecl))
        {
            // This is a constraint on a global generic type parameters,
            // and so it should lower as a parameter of its own.
            auto supType = lowerType(context, decl->getSup().type);
            auto inst = getBuilder()->emitGlobalGenericWitnessTableParam(supType);
            addLinkageDecoration(context, inst, decl);
            return LoweredValInfo::simple(inst);
        }

        // Otherwise we really don't expect to see a type constraint
        // declaration like this during lowering, because a generic
        // should have set up a parameter for any constraints as
        // part of being lowered.

        SLANG_UNEXPECTED("generic type constraint during lowering");
        UNREACHABLE_RETURN(LoweredValInfo());
    }

    LoweredValInfo visitGlobalGenericParamDecl(GlobalGenericParamDecl* decl)
    {
        auto inst = getBuilder()->emitGlobalGenericTypeParam();
        addLinkageDecoration(context, inst, decl);
        return LoweredValInfo::simple(inst);
    }

    LoweredValInfo visitGlobalGenericValueParamDecl(GlobalGenericValueParamDecl* decl)
    {
        auto builder = getBuilder();
        auto type = lowerType(context, decl->type);
        auto inst = builder->emitGlobalGenericParam(type);
        addLinkageDecoration(context, inst, decl);
        return LoweredValInfo::simple(inst);
    }

    bool isExportedType(Type* type)
    {
        if (auto declRefType = as<DeclRefType>(type))
        {
            if (declRefType->getDeclRef().getDecl()->findModifier<HLSLExportModifier>())
                return true;
        }
        return false;
    }

    void lowerWitnessTable(
        IRGenContext* subContext,
        WitnessTable* astWitnessTable,
        IRWitnessTable* irWitnessTable,
        Dictionary<WitnessTable*, IRWitnessTable*>& mapASTToIRWitnessTable)
    {
        auto subBuilder = subContext->irBuilder;

        for (auto entry : astWitnessTable->getRequirementDictionary())
        {
            auto requiredMemberDecl = entry.key;
            auto satisfyingWitness = entry.value;

            auto irRequirementKey = getInterfaceRequirementKey(requiredMemberDecl);
            if (!irRequirementKey)
                continue;

            IRInst* irSatisfyingVal = nullptr;

            switch (satisfyingWitness.getFlavor())
            {
            case RequirementWitness::Flavor::declRef:
                {
                    auto satisfyingDeclRef = satisfyingWitness.getDeclRef();
                    irSatisfyingVal = getSimpleVal(
                        subContext,
                        emitDeclRef(
                            subContext,
                            satisfyingDeclRef,
                            // TODO: we need to know what type to plug in here...
                            nullptr));
                }
                break;

            case RequirementWitness::Flavor::val:
                {
                    auto satisfyingVal = satisfyingWitness.getVal();
                    irSatisfyingVal = lowerSimpleVal(subContext, satisfyingVal);
                }
                break;

            case RequirementWitness::Flavor::witnessTable:
                {
                    auto astReqWitnessTable = satisfyingWitness.getWitnessTable();
                    IRWitnessTable* irSatisfyingWitnessTable = nullptr;
                    if (!mapASTToIRWitnessTable.tryGetValue(
                            astReqWitnessTable,
                            irSatisfyingWitnessTable))
                    {
                        // Need to construct a sub-witness-table
                        auto irWitnessTableBaseType =
                            lowerType(subContext, astReqWitnessTable->baseType);

                        auto concreteType = irWitnessTable->getConcreteType();

                        irSatisfyingWitnessTable =
                            subBuilder->createWitnessTable(irWitnessTableBaseType, concreteType);

                        // Avoid adding same decorations and child more than once.
                        if (!irSatisfyingWitnessTable->hasDecorationOrChild())
                        {
                            auto mangledName = getMangledNameForConformanceWitness(
                                subContext->astBuilder,
                                astReqWitnessTable->witnessedType,
                                astReqWitnessTable->baseType,
                                concreteType->getOp());

                            subBuilder->addExportDecoration(
                                irSatisfyingWitnessTable,
                                mangledName.getUnownedSlice());

                            if (isExportedType(astReqWitnessTable->witnessedType))
                            {
                                subBuilder->addHLSLExportDecoration(irSatisfyingWitnessTable);
                                subBuilder->addKeepAliveDecoration(irSatisfyingWitnessTable);
                            }

                            // Recursively lower the sub-table.
                            lowerWitnessTable(
                                subContext,
                                astReqWitnessTable,
                                irSatisfyingWitnessTable,
                                mapASTToIRWitnessTable);

                            irSatisfyingWitnessTable->moveToEnd();
                        }
                    }
                    irSatisfyingVal = irSatisfyingWitnessTable;
                }
                break;

            default:
                SLANG_UNEXPECTED("handled requirement witness case");
                break;
            }


            subBuilder->createWitnessTableEntry(irWitnessTable, irRequirementKey, irSatisfyingVal);
        }
    }

    LoweredValInfo visitInheritanceDecl(InheritanceDecl* inheritanceDecl)
    {
        // An inheritance clause inside of an `interface`
        // declaration should not give rise to a witness
        // table, because it represents something the
        // interface requires, and not what it provides.
        //
        auto parentDecl = inheritanceDecl->parentDecl;
        if (const auto parentInterfaceDecl = as<InterfaceDecl>(parentDecl))
        {
            return LoweredValInfo::simple(getInterfaceRequirementKey(inheritanceDecl));
        }
        //
        // We also need to cover the case where an `extension`
        // declaration is being used to add a conformance to
        // an existing `interface`:
        //
        if (auto parentExtensionDecl = as<ExtensionDecl>(parentDecl))
        {
            auto targetType = parentExtensionDecl->targetType;
            if (auto targetDeclRefType = as<DeclRefType>(targetType))
            {
                if (auto targetInterfaceDeclRef =
                        targetDeclRefType->getDeclRef().as<InterfaceDecl>())
                {
                    return LoweredValInfo::simple(getInterfaceRequirementKey(inheritanceDecl));
                }
            }
        }

        // Find the type that is doing the inheriting.
        // Under normal circumstances it is the type declaration that
        // is the parent for the inheritance declaration, but if
        // the inheritance declaration is on an `extension` declaration,
        // then we need to identify the type being extended.
        //
        Type* subType = nullptr;
        if (auto extParentDecl = as<ExtensionDecl>(parentDecl))
        {
            subType = extParentDecl->targetType.type;
        }
        else
        {
            subType = DeclRefType::create(context->astBuilder, makeDeclRef(parentDecl));
        }

        // What is the super-type that we have declared we inherit from?
        Type* superType = inheritanceDecl->base.type;

        if (auto superDeclRefType = as<DeclRefType>(superType))
        {
            if (superDeclRefType->getDeclRef().as<StructDecl>() ||
                superDeclRefType->getDeclRef().as<ClassDecl>())
            {
                // TODO: the witness that a type inherits from a `struct`
                // type should probably be a key that will be used for
                // a field that holds the base type...
                //
                auto irKey = getBuilder()->createStructKey();
                addLinkageDecoration(context, irKey, inheritanceDecl);
                getBuilder()->addNameHintDecoration(irKey, UnownedTerminatedStringSlice("base"));
                auto keyVal = LoweredValInfo::simple(irKey);
                context->setGlobalValue(inheritanceDecl, keyVal);
                return keyVal;
            }
        }

        // A witness table may need to be generic, if the outer
        // declaration (either a type declaration or an `extension`)
        // is generic.
        //
        NestedContext nested(this);
        auto subBuilder = nested.getBuilder();
        auto subContext = nested.getContext();
        auto outerGeneric = emitOuterGenerics(subContext, inheritanceDecl, inheritanceDecl);

        // Lower the super-type to force its declaration to be lowered.
        //
        // Note: we are using the "sub-context" here because the
        // type being inherited from could reference generic parameters,
        // and we need those parameters to lower as references to
        // the parameters of our IR-level generic.
        //
        auto irWitnessTableBaseType = lowerType(subContext, superType);

        // Register a dummy value to avoid infinite recursions.
        // Without this, the call to lowerType() can get into an infinite recursion.
        //
        context->setGlobalValue(
            inheritanceDecl,
            LoweredValInfo::simple(findOuterMostGeneric(subBuilder->getInsertLoc().getParent())));

        auto irSubType = lowerType(subContext, subType);

        // Create the IR-level witness table
        auto irWitnessTable = subBuilder->createWitnessTable(irWitnessTableBaseType, irSubType);

        // Override with the correct witness-table
        context->setGlobalValue(
            inheritanceDecl,
            LoweredValInfo::simple(findOuterMostGeneric(irWitnessTable)));

        // Avoid adding same decorations and child more than once.
        if (!irWitnessTable->hasDecorationOrChild())
        {
            // Construct the mangled name for the witness table, which depends
            // on the type that is conforming, and the type that it conforms to.
            //
            // TODO: This approach doesn't really make sense for generic `extension`
            // conformances.
            auto mangledName = getMangledNameForConformanceWitness(
                context->astBuilder,
                subType,
                superType,
                irSubType->getOp());

            // TODO(JS):
            // Should the mangled name take part in obfuscation if enabled?

            addLinkageDecoration(
                context,
                irWitnessTable,
                inheritanceDecl,
                mangledName.getUnownedSlice());

            // If the witness table is for a COM interface, always keep it alive.
            if (irWitnessTableBaseType->findDecoration<IRComInterfaceDecoration>())
            {
                subBuilder->addHLSLExportDecoration(irWitnessTable);
            }

            for (auto mod : parentDecl->modifiers)
            {
                if (as<HLSLExportModifier>(mod))
                {
                    subBuilder->addHLSLExportDecoration(irWitnessTable);
                    subBuilder->addKeepAliveDecoration(irWitnessTable);
                }
                else if (as<AutoDiffBuiltinAttribute>(mod))
                {
                    subBuilder->addAutoDiffBuiltinDecoration(irWitnessTable);
                }
            }

            // Make sure that all the entries in the witness table have been filled in,
            // including any cases where there are sub-witness-tables for conformances
            bool isExplicitExtern = false;
            bool isImported = isImportedDecl(context, parentDecl, isExplicitExtern);
            if (!isImported || isExplicitExtern)
            {
                Dictionary<WitnessTable*, IRWitnessTable*> mapASTToIRWitnessTable;
                lowerWitnessTable(
                    subContext,
                    inheritanceDecl->witnessTable,
                    irWitnessTable,
                    mapASTToIRWitnessTable);
            }

            irWitnessTable->moveToEnd();
        }

        return LoweredValInfo::simple(
            finishOuterGenerics(subBuilder, irWitnessTable, outerGeneric));
    }

    LoweredValInfo visitDeclGroup(DeclGroup* declGroup)
    {
        // To lower a group of declarations, we just
        // lower each one individually.
        //
        for (auto decl : declGroup->decls)
        {
            IRBuilderSourceLocRAII sourceLocInfo(context->irBuilder, decl->loc);

            // Note: I am directly invoking `dispatch` here,
            // instead of `ensureDecl` just to try and
            // make sure that we don't accidentally
            // emit things to an outer context.
            //
            // TODO: make sure that can't happen anyway.
            dispatch(decl);
        }

        return LoweredValInfo();
    }

    LoweredValInfo visitStorageDeclCommon(ContainerDecl* decl)
    {
        // A subscript operation may encompass one or more
        // accessors, and these are what should actually
        // get lowered (they are effectively functions).

        for (auto accessor : decl->getMembersOfType<AccessorDecl>())
        {
            if (accessor->hasModifier<IntrinsicOpModifier>())
                continue;

            ensureDecl(context, accessor);
        }

        // The subscript declaration itself won't correspond
        // to anything in the lowered program, so we don't
        // bother creating a representation here.
        //
        // Note: We may want to have a specific lowered value
        // that can represent the combination of callables
        // that make up the subscript operation.
        return LoweredValInfo();
    }

    LoweredValInfo visitSubscriptDecl(SubscriptDecl* decl) { return visitStorageDeclCommon(decl); }

    LoweredValInfo visitPropertyDecl(PropertyDecl* decl) { return visitStorageDeclCommon(decl); }

    bool isGlobalVarDecl(VarDecl* decl)
    {
        auto parent = decl->parentDecl;
        if (as<NamespaceDeclBase>(parent))
        {
            // Variable declared at global/namespace scope? -> Global.
            return true;
        }
        else if (as<FileDecl>(parent))
        {
            // Variable declared at file scope? -> Global.
            return true;
        }
        else if (as<AggTypeDeclBase>(parent))
        {
            if (decl->hasModifier<HLSLStaticModifier>())
            {
                // A `static` member variable is effectively global.
                return true;
            }
        }

        return false;
    }

    bool isMemberVarDecl(VarDecl* decl)
    {
        auto parent = decl->parentDecl;
        if (as<AggTypeDecl>(parent))
        {
            // A variable declared inside of an aggregate type declaration is a member.
            return true;
        }
        if (auto extDecl = as<ExtensionDecl>(parent))
        {
            if (const auto declRefType = as<DeclRefType>(extDecl->targetType.type))
            {
                return true;
            }
        }
        return false;
    }

    struct NestedContext
    {
        IRGenEnv subEnvStorage;
        IRBuilder subBuilderStorage;
        IRGenContext subContextStorage;

        NestedContext(DeclLoweringVisitor* outer)
            : subBuilderStorage(*outer->getBuilder()), subContextStorage(*outer->context)
        {
            auto outerContext = outer->context;

            subEnvStorage.outer = outerContext->env;

            subContextStorage.irBuilder = &subBuilderStorage;
            subContextStorage.env = &subEnvStorage;

            subContextStorage.thisType = outerContext->thisType;
            subContextStorage.thisTypeWitness = outerContext->thisTypeWitness;

            subContextStorage.returnDestination = LoweredValInfo();
            subContextStorage.lowerTypeCallback = nullptr;
        }

        IRBuilder* getBuilder() { return &subBuilderStorage; }
        IRGenContext* getContext() { return &subContextStorage; }
    };

    LoweredValInfo lowerGlobalShaderParam(VarDecl* decl)
    {
        IRType* paramType = lowerType(context, decl->getType());

        auto builder = getBuilder();

        auto irParam = builder->createGlobalParam(paramType);
        auto paramVal = LoweredValInfo::simple(irParam);

        addLinkageDecoration(context, irParam, decl);
        addNameHint(context, irParam, decl);
        maybeSetRate(context, irParam, decl);
        addVarDecorations(context, irParam, decl);
        maybeAddDebugLocationDecoration(context, irParam);

        if (decl)
        {
            builder->addHighLevelDeclDecoration(irParam, decl);
        }

        addTargetIntrinsicDecorations(nullptr, irParam, decl);

        bool hasLayoutSemantic = false;
        bool isSpecializationConstant = false;
        for (auto modifier : decl->modifiers)
        {
            if (as<HLSLLayoutSemantic>(modifier))
            {
                hasLayoutSemantic = true;
            }
            else if (
                as<SpecializationConstantAttribute>(modifier) ||
                as<VkConstantIdAttribute>(modifier))
            {
                isSpecializationConstant = true;
            }
        }
        if (hasLayoutSemantic)
            builder->addHasExplicitHLSLBindingDecoration(irParam);

        // A global variable's SSA value is a *pointer* to
        // the underlying storage.
        context->setGlobalValue(decl, paramVal);

        if (isSpecializationConstant && decl->initExpr)
        {
            auto initVal = getSimpleVal(context, lowerRValueExpr(context, decl->initExpr));
            builder->addDefaultValueDecoration(irParam, initVal);
        }

        irParam->moveToEnd();

        return paramVal;
    }

    LoweredValInfo lowerConstantDeclCommon(VarDeclBase* decl)
    {
        // It's constant, so shoul dhave this modifier
        SLANG_ASSERT(decl->hasModifier<ConstModifier>());

        NestedContext nested(this);
        auto subBuilder = nested.getBuilder();
        auto subContext = nested.getContext();

        IRGeneric* outerGeneric = nullptr;

        // If we are static, then we need to insert the declaration before the parent.
        // This tries to match the behavior of previous `lowerFunctionStaticConstVarDecl`
        // functionality
        if (isFunctionStaticVarDecl(decl))
        {
            // We need to insert the constant at a level above
            // the function being emitted. This will usually
            // be the global scope, but it might be an outer
            // generic if we are lowering a generic function.
            subBuilder->setInsertBefore(subBuilder->getFunc());
        }
        else if (!isFunctionVarDecl(decl))
        {
            outerGeneric = emitOuterGenerics(subContext, decl, decl);
        }

        auto initExpr = decl->initExpr;

        // We want to be able to support cases where a global constant is defined in
        // another module and we should not bind to its value at (front-end) compile
        // time. We handle this by adding a level of indirection where a global constant
        // is represented as an IR node with zero or one operands. In the zero-operand
        // case the node represents a global constant with an unknown value (perhaps
        // an imported constant), while in the one-operand case the operand gives us
        // the concrete value to use for the constant.
        //
        // Using a level of indirection also gives us a well-defined place to attach
        // annotation information like name hints, since otherwise two constants
        // with the same value would map to identical IR nodes.
        //
        // TODO: For now we detect whether or not to include the value operand based on
        // whether we see an initial-value expression in the AST declaration, but
        // eventually we might base this on whether or not the value should be accessible
        // to the module we are lowering.

        IRInst* irConstant = nullptr;
        if (!initExpr)
        {
            // If we don't know the value we want to use, then we just create
            // a global constant IR node with the right type.
            //
            auto irType = lowerType(subContext, decl->getType());
            irConstant = subBuilder->emitGlobalConstant(irType);
        }
        else
        {
            // We lower the value expression directly, which yields a
            // global instruction to represent the value. There is
            // no guarantee that this instruction is unique (e.g.,
            // if we have two different constants definitions both
            // with the value `5`, then we might have only a single
            // instruction to represent `5`.
            //
            auto irInitVal = getSimpleVal(subContext, lowerRValueExpr(subContext, initExpr));

            // We construct a distinct IR instruction to represent the
            // constant itself, with the value as an operand.
            //
            irConstant = subBuilder->emitGlobalConstant(irInitVal->getFullType(), irInitVal);
        }

        // All of the attributes/decorations we can attach
        // belong on the IR constant node.
        //

        addLinkageDecoration(context, irConstant, decl);

        addNameHint(context, irConstant, decl);
        addVarDecorations(context, irConstant, decl);

        getBuilder()->addHighLevelDeclDecoration(irConstant, decl);

        // Finish of generic

        auto loweredValue =
            LoweredValInfo::simple(finishOuterGenerics(subBuilder, irConstant, outerGeneric));

        // Register the value that was emitted as the value
        // for any references to the constant from elsewhere
        // in the code.
        //
        context->setGlobalValue(decl, loweredValue);

        return loweredValue;
    }

    LoweredValInfo lowerGlobalConstantDecl(VarDecl* decl) { return lowerConstantDeclCommon(decl); }

    LoweredValInfo lowerGlobalVarDecl(VarDecl* decl)
    {
        // A non-`static` global is actually a shader parameter in HLSL.
        //
        // TODO: We should probably make that case distinct at the AST
        // level as well, since global shader parameters are fairly
        // different from global variables.
        //
        if (isGlobalShaderParameter(decl))
        {
            return lowerGlobalShaderParam(decl);
        }

        // A `static const` global is actually a compile-time constant.
        //
        if (decl->hasModifier<HLSLStaticModifier>() && decl->hasModifier<ConstModifier>())
        {
            return lowerGlobalConstantDecl(decl);
        }

        NestedContext nested(this);
        auto subBuilder = nested.getBuilder();
        auto subContext = nested.getContext();

        IRGeneric* outerGeneric = nullptr;

        // If we are static, then we need to insert the declaration before the parent.
        // This tries to match the behavior of previous `lowerFunctionStaticConstVarDecl`
        // functionality
        if (isFunctionStaticVarDecl(decl))
        {
            // We need to insert the constant at a level above
            // the function being emitted. This will usually
            // be the global scope, but it might be an outer
            // generic if we are lowering a generic function.
            subBuilder->setInsertBefore(subBuilder->getFunc());
        }
        else if (!isFunctionVarDecl(decl))
        {
            outerGeneric = emitOuterGenerics(subContext, decl, decl);
        }

        IRType* varType = lowerType(subContext, decl->getType());

        // TODO(JS): Do we create something derived from IRGlobalVar? Or do we use
        // a decoration to identify an *actual* global?

        IRGlobalValueWithCode* irGlobal = subBuilder->createGlobalVar(varType);

        addLinkageDecoration(subContext, irGlobal, decl);
        addNameHint(subContext, irGlobal, decl);

        maybeSetRate(subContext, irGlobal, decl);

        addVarDecorations(subContext, irGlobal, decl);
        maybeAddDebugLocationDecoration(subContext, irGlobal);

        if (decl)
        {
            subBuilder->addHighLevelDeclDecoration(irGlobal, decl);
        }

        if (auto initExpr = decl->initExpr)
        {
            subBuilder->setInsertInto(irGlobal);

            IRBlock* entryBlock = subBuilder->emitBlock();
            subBuilder->setInsertInto(entryBlock);

            LoweredValInfo initVal = lowerLValueExpr(subContext, initExpr);
            subContext->irBuilder->emitReturn(getSimpleVal(subContext, initVal));
        }

        // A global variable's SSA value is a *pointer* to
        // the underlying storage.
        auto loweredValue =
            LoweredValInfo::ptr(finishOuterGenerics(subBuilder, irGlobal, outerGeneric));
        context->setGlobalValue(decl, loweredValue);

        return loweredValue;
    }

    LoweredValInfo lowerFunctionStaticConstVarDecl(VarDeclBase* decl)
    {
        return lowerConstantDeclCommon(decl);
    }

    LoweredValInfo lowerFunctionStaticVarDecl(VarDeclBase* decl)
    {
        // We know the variable is `static`, but it might also be `const.
        if (decl->hasModifier<ConstModifier>())
            return lowerFunctionStaticConstVarDecl(decl);

        // A function-scope `static` variable is effectively a global,
        // and a simple solution here would be to try to emit this
        // variable directly into the global scope.
        //
        // The one major wrinkle we need to deal with is the way that
        // a function-scope `static` variable could be nested under
        // a generic, leading to the situation that different instances
        // of that same generic would need distinct storage for that
        // variable declaration.
        //
        // We will handle that constraint by carefully nesting the
        // IR global variable under the parent of its containing
        // function.
        //
        auto parent = getBuilder()->getInsertLoc().getParent();
        if (auto block = as<IRBlock>(parent))
            parent = block->getParent();

        NestedContext nestedContext(this);
        auto subBuilder = nestedContext.getBuilder();
        auto subContext = nestedContext.getContext();
        subBuilder->setInsertBefore(parent);

        IRType* subVarType = lowerType(subContext, decl->getType());
        IRGlobalValueWithCode* irGlobal = subBuilder->createGlobalVar(subVarType);
        addVarDecorations(subContext, irGlobal, decl);

        addNameHint(context, irGlobal, decl);
        maybeSetRate(context, irGlobal, decl);

        subBuilder->addHighLevelDeclDecoration(irGlobal, decl);

        LoweredValInfo globalVal = LoweredValInfo::ptr(irGlobal);
        context->setValue(decl, globalVal);

        // A `static` variable with an initializer needs special handling,
        // at least if the initializer isn't a compile-time constant.
        if (auto initExpr = decl->initExpr)
        {
            // We must create another global `bool isInitialized = false`
            // to represent whether we've initialized this before.
            // Then emit code like:
            //
            //      if(!isInitialized) { <globalVal> = <initExpr>; isInitialized = true; }
            //
            // This will generate a lot of boilterplate code, but we optimize out the
            // boilerplate functions later during `moveGlobalVarInitializationToEntryPoints`
            // if we see the init function is just returning a global constant.
            //
            auto boolBuilder = subBuilder;

            auto irBoolType = boolBuilder->getBoolType();
            auto irBool = boolBuilder->createGlobalVar(irBoolType);
            boolBuilder->setInsertInto(irBool);
            boolBuilder->emitBlock();
            boolBuilder->emitReturn(boolBuilder->getBoolValue(false));

            auto boolVal = LoweredValInfo::ptr(irBool);

            // Okay, with our global Boolean created, we can move on to
            // generating the code we actually care about, back in the original function.

            auto builder = getBuilder();

            auto initBlock = builder->createBlock();
            auto afterBlock = builder->createBlock();

            builder->emitIfElse(getSimpleVal(context, boolVal), afterBlock, initBlock, afterBlock);

            builder->insertBlock(initBlock);
            LoweredValInfo initVal = lowerLValueExpr(context, initExpr);
            assign(context, globalVal, initVal);
            assign(context, boolVal, LoweredValInfo::simple(builder->getBoolValue(true)));
            builder->emitBranch(afterBlock);

            builder->insertBlock(afterBlock);
        }

        return globalVal;
    }

    LoweredValInfo visitGenericValueParamDecl(GenericValueParamDecl* decl)
    {
        return emitDeclRef(context, makeDeclRef(decl), lowerType(context, decl->type));
    }

    LoweredValInfo visitVarDecl(VarDecl* decl)
    {
        // Detect global (or effectively global) variables
        // and handle them differently.
        if (isGlobalVarDecl(decl))
        {
            return lowerGlobalVarDecl(decl);
        }

        if (isFunctionStaticVarDecl(decl))
        {
            return lowerFunctionStaticVarDecl(decl);
        }

        if (isMemberVarDecl(decl))
        {
            return lowerMemberVarDecl(decl);
        }

        // A user-defined variable declaration will usually turn into
        // an `alloca` operation for the variable's storage,
        // plus some code to initialize it and then store to the variable.

        IRType* varType = lowerType(context, decl->getType());

        // As a special case, an immutable local variable with an
        // initializer can just lower to the SSA value of its initializer.
        //
        if (as<LetDecl>(decl))
        {
            if (auto initExpr = decl->initExpr)
            {
                auto initVal = lowerRValueExpr(context, initExpr);
                initVal = LoweredValInfo::simple(getSimpleVal(context, initVal));
                context->setGlobalValue(decl, initVal);
                return initVal;
            }
        }


        LoweredValInfo varVal = createVar(context, varType, decl);
        maybeAddDebugLocationDecoration(context, varVal.val);

        if (auto initExpr = decl->initExpr)
        {
            assignExpr(context, varVal, initExpr, decl->loc);
        }

        context->setGlobalValue(decl, varVal);

        return varVal;
    }

    IRStructKey* getInterfaceRequirementKey(Decl* requirementDecl)
    {
        return Slang::getInterfaceRequirementKey(context, requirementDecl);
    }

    LoweredValInfo visitAssocTypeDecl(AssocTypeDecl* decl)
    {
        SLANG_ASSERT(decl->parentDecl != nullptr);
        ShortList<IRInterfaceType*> constraintInterfaces;
        for (auto constraintDecl : decl->getMembersOfType<GenericTypeConstraintDecl>())
        {
            auto baseType = lowerType(context, constraintDecl->sup.type);
            if (baseType && baseType->getOp() == kIROp_InterfaceType)
                constraintInterfaces.add((IRInterfaceType*)baseType);
        }
        auto assocType =
            context->irBuilder->getAssociatedType(constraintInterfaces.getArrayView().arrayView);
        context->setValue(decl, assocType);
        return LoweredValInfo::simple(assocType);
    }

    void insertRequirementKeyAssociation(
        Decl* requirementDecl,
        IRInst* originalKey,
        IRInst* associatedKey)
    {
        IROp op = kIROp_Nop;
        if (as<BackwardDerivativeRequirementDecl>(requirementDecl))
        {
            op = kIROp_BackwardDerivativeDecoration;
        }
        else if (as<ForwardDerivativeRequirementDecl>(requirementDecl))
        {
            op = kIROp_ForwardDerivativeDecoration;
        }
        else
        {
            return;
        }
        context->irBuilder->addDecoration(originalKey, op, associatedKey);
    }

    // Given `value` defined as an independent generic of `outerGeneric`, emit IR that specializes
    // it using the generic params defined in `outerGeneric`. For example:
    // ```
    //  interface IFoo<T> { void f(); }
    // ```
    // We will lower `IFoo<T>::f` into `%f = IRGeneric(T) { return IRFunc(...) }`
    // When we lower the interface type `IFoo`, it will become:
    // ```
    // %IFoo = IRGeneric(T1) { return IRInterfaceType(???); )
    // ```
    // We want the `???` to be `specialize(%f, T1)`.
    // To do so, we will call `specializeWithOuterGeneric` with `value` = `%f`, and `outerGeneric` =
    // %IFoo.
    //
    IRInst* specializeWithOuterGeneric(IRBuilder* irBuilder, IRInst* value, IRGeneric* outerGeneric)
    {
        if (!as<IRGeneric>(value))
            return value;
        if (!outerGeneric)
            return value;

        // If `outerGeneric` has a generic parent, we want to recursively specialize value
        // using the parent generic first.
        auto parentGeneric = getOuterGeneric(outerGeneric);
        if (parentGeneric)
            value = specializeWithOuterGeneric(irBuilder, value, parentGeneric);

        // Now we can specialize `value` using the params defined in `outerGeneric`.
        List<IRInst*> args;
        for (auto param : outerGeneric->getParams())
            args.add(param);
        return irBuilder->emitSpecializeInst(irBuilder->getGenericKind(), value, args);
    }

    LoweredValInfo visitInterfaceDecl(InterfaceDecl* decl)
    {
        // The members of an interface will turn into the keys that will
        // be used for lookup operations into witness
        // tables that promise conformance to the interface.
        //
        // TODO: we don't handle the case here of an interface
        // with concrete/default implementations for any
        // of its members.
        //
        // TODO: If we want to support using an interface as
        // an existential type, then we might need to emit
        // a witness table for the interface type's conformance
        // to its own interface.
        //
        NestedContext nestedContext(this);
        auto subBuilder = nestedContext.getBuilder();
        auto subContext = nestedContext.getContext();

        // Emit any generics that should wrap the actual type.
        auto outerGeneric = emitOuterGenerics(subContext, decl, decl);

        // First, compute the number of requirement entries that will be included in this
        // interface type.
        UInt operandCount = 0;
        for (auto requirementDecl : decl->members)
        {
            if (as<GenericDecl>(requirementDecl))
                requirementDecl = getInner(requirementDecl);

            if (as<SubscriptDecl>(requirementDecl) || as<PropertyDecl>(requirementDecl))
            {
                for (auto accessorDecl : as<ContainerDecl>(requirementDecl)->members)
                {
                    if (as<AccessorDecl>(accessorDecl))
                        operandCount++;
                }
            }
            if (!shouldDeclBeTreatedAsInterfaceRequirement(requirementDecl))
            {
                continue;
            }

            operandCount++;
            // As a special case, any type constraints placed
            // on an associated type will *also* need to be turned
            // into requirement keys for this interface.
            if (auto associatedTypeDecl = as<AssocTypeDecl>(requirementDecl))
            {
                operandCount +=
                    associatedTypeDecl->getMembersOfType<TypeConstraintDecl>().getCount();
            }
        }

        // Allocate an IRInterfaceType with the `operandCount` operands.
        IRInterfaceType* irInterface = subBuilder->createInterfaceType(operandCount, nullptr);
        auto finalVal = finishOuterGenerics(subBuilder, irInterface, outerGeneric);

        // Add `irInterface` to decl mapping now to prevent cyclic lowering.
        context->setGlobalValue(decl, LoweredValInfo::simple(finalVal));

        subBuilder->setInsertBefore(irInterface);

        // Setup subContext for proper lowering `ThisType`, associated types and
        // the interface decl's self reference.

        auto thisType = DeclRefType::create(
            context->astBuilder,
            createDefaultSpecializedDeclRef(subContext, nullptr, decl->getThisTypeDecl()));
        subContext->thisType = thisType;
        // Create a stand-in witness that represents `ThisType` conforms to the interface.
        subContext->thisTypeWitness = subBuilder->createThisTypeWitness((IRType*)finalVal);

        // Lower associated types first, so they can be referred to when lowering functions.
        for (auto assocTypeDecl : decl->getMembersOfType<AssocTypeDecl>())
        {
            ensureDecl(subContext, assocTypeDecl);
        }

        UInt entryIndex = 0;
        auto addEntry = [&](IRStructKey* requirementKey, DeclRef<Decl> requirementDeclRef)
        {
            auto entry = subBuilder->createInterfaceRequirementEntry(requirementKey, nullptr);
            if (auto inheritance = requirementDeclRef.as<InheritanceDecl>())
            {
                auto irBaseType =
                    lowerType(subContext, getSup(subContext->astBuilder, inheritance));
                auto irWitnessTableType = subBuilder->getWitnessTableType(irBaseType);
                entry->setRequirementVal(irWitnessTableType);
            }
            else
            {
                auto requirementVal = ensureDecl(subContext, requirementDeclRef.getDecl()).val;

                switch (requirementVal->getOp())
                {
                default:
                    // For the majority of requirements, we only care about its type in an
                    // interface definition, so we store only the type from the lowered IR
                    // in the interface entry.
                    // We need to make sure the type is specialized with the outer generic
                    // parameters in case the interface itself is inside a generic.
                    //
                    requirementVal = specializeWithOuterGeneric(
                        context->irBuilder,
                        requirementVal->getFullType(),
                        outerGeneric);
                    entry->setRequirementVal(requirementVal);
                    break;

                case kIROp_AssociatedType:
                    // For associated types, we will store it directly inside the interface
                    // type.
                    entry->setRequirementVal(requirementVal);
                    break;
                }
                if (requirementDeclRef.getDecl()->findModifier<HLSLStaticModifier>())
                {
                    getBuilder()->addStaticRequirementDecoration(requirementKey);
                }
            }
            irInterface->setOperand(entryIndex, entry);
            entryIndex++;
            // Add addtional requirements for type constraints placed
            // on an associated types.
            if (auto associatedTypeDeclRef = requirementDeclRef.as<AssocTypeDecl>())
            {
                for (auto constraintDeclRef : getMembersOfType<TypeConstraintDecl>(
                         subContext->astBuilder,
                         associatedTypeDeclRef))
                {
                    auto constraintKey = getInterfaceRequirementKey(constraintDeclRef.getDecl());
                    auto constraintInterfaceType =
                        lowerType(subContext, getSup(subContext->astBuilder, constraintDeclRef));
                    auto witnessTableType =
                        getBuilder()->getWitnessTableType(constraintInterfaceType);

                    auto constraintEntry = subBuilder->createInterfaceRequirementEntry(
                        constraintKey,
                        witnessTableType);
                    irInterface->setOperand(entryIndex, constraintEntry);
                    entryIndex++;

                    context->setValue(
                        constraintDeclRef.getDecl(),
                        LoweredValInfo::simple(constraintEntry));
                }
            }
            else
            {
                CallableDecl* callableDecl = nullptr;
                if (auto genDecl = as<GenericDecl>(requirementDeclRef.getDecl()))
                    callableDecl = as<CallableDecl>(genDecl->inner);
                else
                    callableDecl = as<CallableDecl>(requirementDeclRef.getDecl());
                if (callableDecl)
                {
                    // Differentiable functions has additional requirements for the derivatives.
                    for (auto diffDecl :
                         callableDecl->getMembersOfType<DerivativeRequirementReferenceDecl>())
                    {
                        auto diffKey = getInterfaceRequirementKey(diffDecl->referencedDecl);
                        insertRequirementKeyAssociation(
                            diffDecl->referencedDecl,
                            requirementKey,
                            diffKey);
                    }
                }
                // Add lowered requirement entry to current decl mapping to prevent
                // the function requirements from being lowered again when we get to
                // `ensureAllDeclsRec`.
                context->setValue(requirementDeclRef.getDecl(), LoweredValInfo::simple(entry));
            }
        };
        for (auto requirementDecl : decl->members)
        {
            auto requirementKey = getInterfaceRequirementKey(requirementDecl);
            if (!requirementKey)
            {
                if (auto genericDecl = as<GenericDecl>(requirementDecl))
                {
                    // We need to form a declref into the inner decls in case of a generic
                    // requirement.
                    requirementDecl = getInner(genericDecl);
                }

                if (as<PropertyDecl>(requirementDecl) || as<SubscriptDecl>(requirementDecl))
                {
                    for (auto member : as<ContainerDecl>(requirementDecl)->members)
                    {
                        if (auto accessorDecl = as<AccessorDecl>(member))
                        {
                            auto accessorKey = getInterfaceRequirementKey(accessorDecl);
                            if (accessorKey)
                            {
                                auto accessorDeclRef = createDefaultSpecializedDeclRef(
                                    subContext,
                                    nullptr,
                                    accessorDecl);
                                addEntry(accessorKey, accessorDeclRef);
                            }
                        }
                    }
                }
                continue;
            }
            else
            {
                if (auto genericDecl = as<GenericDecl>(requirementDecl))
                {
                    // We need to form a declref into the inner decls in case of a generic
                    // requirement.
                    requirementDecl = getInner(genericDecl);
                }
                auto requirementDeclRef =
                    createDefaultSpecializedDeclRef(subContext, nullptr, requirementDecl);
                addEntry(requirementKey, requirementDeclRef);
            }
        }

        addNameHint(context, irInterface, decl);
        addLinkageDecoration(context, irInterface, decl);
        if (auto anyValueSizeAttr = decl->findModifier<AnyValueSizeAttribute>())
        {
            subBuilder->addAnyValueSizeDecoration(irInterface, anyValueSizeAttr->size);
        }
        if (const auto specializeAttr = decl->findModifier<SpecializeAttribute>())
        {
            subBuilder->addSpecializeDecoration(irInterface);
        }
        if (auto comInterfaceAttr = decl->findModifier<ComInterfaceAttribute>())
        {
            subBuilder->addComInterfaceDecoration(
                irInterface,
                comInterfaceAttr->guid.getUnownedSlice());
        }
        if (const auto builtinAttr = decl->findModifier<BuiltinAttribute>())
        {
            subBuilder->addBuiltinDecoration(irInterface);
        }
        if (decl->hasModifier<TreatAsDifferentiableAttribute>())
        {
            subBuilder->addDecoration(irInterface, kIROp_TreatAsDifferentiableDecoration);
        }

        subBuilder->setInsertInto(irInterface);

        addTargetIntrinsicDecorations(subContext, irInterface, decl);

        return LoweredValInfo::simple(finalVal);
    }

    LoweredValInfo visitEnumCaseDecl(EnumCaseDecl* decl)
    {
        // A case within an `enum` decl will lower to a value
        // of the `enum`'s "tag" type.
        //
        // TODO: a bit more work will be needed if we allow for
        // enum cases that have payloads, because then we need
        // a function that constructs the value given arguments.
        //
        NestedContext nestedContext(this);
        auto subContext = nestedContext.getContext();

        // Emit any generics that should wrap the actual type.
        emitOuterGenerics(subContext, decl, decl);

        return lowerRValueExpr(subContext, decl->tagExpr);
    }

    LoweredValInfo visitEnumDecl(EnumDecl* decl)
    {
        // Given a declaration of a type, we need to make sure
        // to output "witness tables" for any interfaces this
        // type has declared conformance to.
        for (auto inheritanceDecl : decl->getMembersOfType<InheritanceDecl>())
        {
            ensureDecl(context, inheritanceDecl);
        }

        NestedContext nestedContext(this);
        auto subBuilder = nestedContext.getBuilder();
        auto subContext = nestedContext.getContext();
        auto outerGeneric = emitOuterGenerics(subContext, decl, decl);

        // TODO: if we ever support `enum` types with payloads, we would
        // need to make the `enum` lower to some kind of custom "tagged union"
        // type.

        IRType* loweredTagType = lowerType(subContext, decl->tagType);
        IRType* enumType = subBuilder->createEnumType(loweredTagType);
        addLinkageDecoration(context, enumType, decl);

        return LoweredValInfo::simple(finishOuterGenerics(subBuilder, enumType, outerGeneric));
    }

    LoweredValInfo visitThisTypeDecl(ThisTypeDecl* decl)
    {
        SLANG_UNUSED(decl);
        return LoweredValInfo();
    }

    LoweredValInfo visitThisTypeConstraintDecl(ThisTypeConstraintDecl* decl)
    {
        SLANG_UNUSED(decl);
        return LoweredValInfo();
    }

    LoweredValInfo visitAggTypeDecl(AggTypeDecl* decl)
    {
        // Don't generate an IR `struct` for intrinsic types
        if (decl->findModifier<IntrinsicTypeModifier>() ||
            decl->findModifier<BuiltinTypeModifier>())
        {
            return LoweredValInfo();
        }

        if (as<AssocTypeDecl>(decl))
        {
            SLANG_UNREACHABLE("associatedtype should have been handled by visitAssocTypeDecl.");
        }

        // TODO(JS):
        // Not clear what to do around HLSLExportModifier.
        // The HLSL spec says it only applies to functions, so we ignore for now.

        // We are going to create nested IR building state
        // to use when emitting the members of the type.
        //
        NestedContext nestedContext(this);
        auto subBuilder = nestedContext.getBuilder();
        auto subContext = nestedContext.getContext();

        // Emit any generics that should wrap the actual type.
        auto outerGeneric = emitOuterGenerics(subContext, decl, decl);

        IRType* irAggType = nullptr;
        if (as<StructDecl>(decl))
        {
            irAggType = subBuilder->createStructType();
        }
        else if (as<ClassDecl>(decl))
        {
            irAggType = subBuilder->createClassType();
        }
        else if (as<GLSLInterfaceBlockDecl>(decl))
        {
            if (decl->findModifier<GLSLBufferModifier>())
            {
                irAggType = subBuilder->createStructType();
            }
            else
            {
                return LoweredValInfo();
            }
        }
        else
        {
            getSink()->diagnose(
                decl->loc,
                Diagnostics::unimplemented,
                "lower unknown AggType to IR");
            return LoweredValInfo::simple(subBuilder->getVoidType());
        }

        maybeAddDebugLocationDecoration(context, irAggType);

        auto finalFinishedVal = finishOuterGenerics(subBuilder, irAggType, outerGeneric);

        // We add the decl now such that if there are Ptr or other references
        // to this type they can still complete
        context->setValue(decl, LoweredValInfo::simple(finalFinishedVal));

        subBuilder->setInsertBefore(irAggType);

        // Given a declaration of a type, we need to make sure
        // to output "witness tables" for any interfaces this
        // type has declared conformance to.
        for (auto inheritanceDecl : decl->getMembersOfType<InheritanceDecl>())
        {
            ensureDecl(subContext, inheritanceDecl);
        }

        addNameHint(context, irAggType, decl);
        addLinkageDecoration(context, irAggType, decl);

        if (const auto payloadAttribute = decl->findModifier<PayloadAttribute>())
        {
            subBuilder->addDecoration(irAggType, kIROp_PayloadDecoration);
        }

        if (const auto rayPayloadAttribute = decl->findModifier<RayPayloadAttribute>())
        {
            subBuilder->addDecoration(irAggType, kIROp_RayPayloadDecoration);
        }

        subBuilder->setInsertInto(irAggType);

        // A `struct` that inherits from another `struct` must start
        // with a member for the direct base type.
        //
        for (auto inheritanceDecl : decl->getMembersOfType<InheritanceDecl>())
        {
            auto superType = inheritanceDecl->base;
            if (auto superDeclRefType = as<DeclRefType>(superType))
            {
                if (superDeclRefType->getDeclRef().as<StructDecl>() ||
                    superDeclRefType->getDeclRef().as<ClassDecl>() ||
                    superDeclRefType->getDeclRef().as<GLSLInterfaceBlockDecl>())
                {
                    auto superKey =
                        (IRStructKey*)getSimpleVal(context, ensureDecl(context, inheritanceDecl));
                    auto irSuperType = lowerType(subContext, superType.type);
                    subBuilder->createStructField(irAggType, superKey, irSuperType);
                }
            }
        }


        for (auto fieldDecl : decl->getMembersOfType<VarDeclBase>())
        {
            if (fieldDecl->hasModifier<HLSLStaticModifier>())
            {
                // A `static` field is actually a global variable,
                // and we should emit it as such.
                ensureDecl(context, fieldDecl);
                continue;
            }

            // Each ordinary field will need to turn into a struct "key"
            // that is used for fetching the field.
            IRInst* fieldKeyInst = getSimpleVal(subContext, ensureDecl(subContext, fieldDecl));
            auto fieldKey = as<IRStructKey>(fieldKeyInst);
            SLANG_ASSERT(fieldKey);

            // Note: we lower the type of the field in the "sub"
            // context, so that any generic parameters that were
            // set up for the type can be referenced by the field type.
            IRType* fieldType = lowerType(subContext, fieldDecl->getType());

            // Then, the parent `struct` instruction itself will have
            // a "field" instruction.
            subBuilder->createStructField(irAggType, fieldKey, fieldType);

            for (auto mod : fieldDecl->modifiers)
            {
                if (auto packOffsetModifier = as<HLSLPackOffsetSemantic>(mod))
                {
                    lowerPackOffsetModifier(fieldKey, packOffsetModifier);
                }
                else if (as<DynamicUniformModifier>(mod))
                {
                    subBuilder->addDynamicUniformDecoration(fieldKey);
                }
            }
        }

        // There may be members not handled by the above logic (e.g.,
        // member functions), but we will not immediately force them
        // to be emitted here, so as not to risk a circular dependency.
        //
        // Instead we will force emission of all children of aggregate
        // type declarations later, from the top-level emit logic.

        addTargetIntrinsicDecorations(subContext, irAggType, decl);
        for (auto modifier : decl->modifiers)
        {
            if (as<NonCopyableTypeAttribute>(modifier))
                subBuilder->addNonCopyableTypeDecoration(irAggType);
            else if (as<AutoDiffBuiltinAttribute>(modifier))
                subBuilder->addAutoDiffBuiltinDecoration(irAggType);
        }

        addTargetRequirementDecorations(irAggType, decl);

        return LoweredValInfo::simple(finalFinishedVal);
    }

    void lowerPackOffsetModifier(IRInst* inst, HLSLPackOffsetSemantic* semantic)
    {
        auto builder = getBuilder();
        int registerOffset =
            stringToInt(semantic->registerName.getName()->text.getUnownedSlice().tail(1));
        int componentOffset = 0;
        if (semantic->componentMask.getContentLength() != 0)
        {
            switch (semantic->componentMask.getContent()[0])
            {
            case 'x':
                componentOffset = 0;
                break;
            case 'y':
                componentOffset = 1;
                break;
            case 'z':
                componentOffset = 2;
                break;
            case 'w':
                componentOffset = 3;
                break;
            }
        }
        builder->addDecoration(
            inst,
            kIROp_PackOffsetDecoration,
            builder->getIntValue(builder->getIntType(), registerOffset),
            builder->getIntValue(builder->getIntType(), componentOffset));
    }

    void lowerRayPayloadAccessModifier(IRInst* inst, RayPayloadAccessSemantic* semantic, IROp op)
    {
        auto builder = getBuilder();

        List<IRInst*> operands;
        for (auto stageNameToken : semantic->stageNameTokens)
        {
            IRInst* stageName = builder->getStringValue(stageNameToken.getContent());
            operands.add(stageName);
        }

        builder->addDecoration(inst, op, operands.getBuffer(), operands.getCount());
    }

    void lowerDerivativeMemberModifier(
        IRInst* inst,
        Decl* memberDecl,
        DerivativeMemberAttribute* derivativeMember)
    {
        IRInst* key = nullptr;
        if (derivativeMember->memberDeclRef->declRef.getDecl() == memberDecl)
        {
            key = inst;
        }
        else
        {
            ensureDecl(context, derivativeMember->memberDeclRef->declRef.getDecl()->parentDecl);
            key = lowerRValueExpr(context, derivativeMember->memberDeclRef).val;
        }
        SLANG_RELEASE_ASSERT(as<IRStructKey>(key));
        auto builder = getBuilder();
        builder->addDecoration(inst, kIROp_DerivativeMemberDecoration, key);
    }

    void lowerDifferentiableAttribute(
        IRGenContext* subContext,
        IRInst* inst,
        DifferentiableAttribute* attr)
    {
        auto irDict = getBuilder()->addDifferentiableTypeDictionaryDecoration(inst);
        for (auto& entry : attr->getMapTypeToIDifferentiableWitness())
        {
            // Lower type and witness.
            IRType* irType = lowerType(subContext, entry.value->getSub());
            IRInst* irWitness = lowerVal(subContext, entry.value).val;

            SLANG_ASSERT(irType);

            // If the witness can be lowered, and the differentiable type entry exists,
            // add an entry to the context.
            //
            if (irWitness)
            {
                getBuilder()->addDifferentiableTypeEntry(irDict, irType, irWitness);
            }
        }
    }

    LoweredValInfo lowerMemberVarDecl(VarDecl* fieldDecl)
    {
        // Each field declaration in the AST translates into
        // a "key" that can be used to extract field values
        // from instances of struct types that contain the field.
        //
        // It is correct to say struct *types* because a `struct`
        // nested under a generic can be used to realize a number
        // of different concrete types, but all of these types
        // will use the same space of keys.

        auto builder = getBuilder();
        IRInst* irFieldKey = nullptr;
        if (auto extVarModifier = fieldDecl->findModifier<ExtensionExternVarModifier>())
        {
            irFieldKey = ensureDecl(context, extVarModifier->originalDecl.getDecl()).val;
            SLANG_RELEASE_ASSERT(as<IRStructKey>(irFieldKey));
        }

        if (!irFieldKey)
        {
            irFieldKey = builder->createStructKey();

            addNameHint(context, irFieldKey, fieldDecl);
            addVarDecorations(context, irFieldKey, fieldDecl);
            addLinkageDecoration(context, irFieldKey, fieldDecl);
        }

        if (auto semanticModifier = fieldDecl->findModifier<HLSLSimpleSemantic>())
        {
            builder->addSemanticDecoration(
                irFieldKey,
                semanticModifier->name.getName()->text.getUnownedSlice());
        }

        if (auto readModifier = fieldDecl->findModifier<RayPayloadReadSemantic>())
        {
            lowerRayPayloadAccessModifier(
                irFieldKey,
                readModifier,
                kIROp_StageReadAccessDecoration);
        }
        if (auto writeModifier = fieldDecl->findModifier<RayPayloadWriteSemantic>())
        {
            lowerRayPayloadAccessModifier(
                irFieldKey,
                writeModifier,
                kIROp_StageWriteAccessDecoration);
        }
        if (auto derivativeMemberModifier = fieldDecl->findModifier<DerivativeMemberAttribute>())
        {
            lowerDerivativeMemberModifier(irFieldKey, fieldDecl, derivativeMemberModifier);
        }

        // We allow a field to be marked as a target intrinsic,
        // so that we can override its mangled name in the
        // output for the chosen target.
        addTargetIntrinsicDecorations(nullptr, irFieldKey, fieldDecl);

        return LoweredValInfo::simple(irFieldKey);
    }

    IRType* maybeGetConstExprType(IRType* type, Decl* decl)
    {
        return Slang::maybeGetConstExprType(getBuilder(), type, decl);
    }

    /// Emit appropriate generic parameters for a constraint, and return the value of that
    /// constraint.
    ///
    /// The `supType` paramete represents the super-type that a parameter is constrained to.
    IRInst* emitGenericConstraintValue(
        IRGenContext* subContext,
        GenericTypeConstraintDecl* constraintDecl,
        IRType* supType)
    {

        auto subBuilder = subContext->irBuilder;

        // There are two cases we care about here.
        //
        if (auto andType = as<IRConjunctionType>(supType))
        {
            // The non-trivial case is when the constraint on a generic parameter
            // was of the form `T : A & B`. In this case, we really want to
            // emit the function with parameters for each of the two independent
            // constraints `T : A` and `T : B`.
            //
            // We will loop over the "cases" of the conjunction (since
            // the `IRConunctionType` can support more than just binary
            // conjunctions) and recursively add constraints for each.
            //
            List<IRInst*> caseVals;
            auto caseCount = andType->getCaseCount();
            for (Int i = 0; i < caseCount; ++i)
            {
                auto caseType = andType->getCaseType(i);
                auto caseVal = emitGenericConstraintValue(subContext, constraintDecl, caseType);
                caseVals.add(caseVal);
            }

            return subBuilder->emitMakeTuple(caseVals);
        }
        else
        {
            // The case case is any other type being used as the constraint.
            //
            // The constraint will then map to a single generic parameter passing
            // a witness table for conformance to the given `supType`.
            //
            auto param = subBuilder->emitParam(subBuilder->getWitnessTableType(supType));
            addNameHint(context, param, constraintDecl);

            // In order to support some of the "any-value" work in dynamic dispatch
            // we have to attach the interface that was used as a constraint onto the
            // type that is being constrained (which we expect to be a generic type
            // parameter).
            //
            // TODO: It feels a bit gross to be doing this here; perhaps the front-end
            // should handle propgation of value-size information from constraints
            // back to generic parameters?
            //
            if (auto genParamDeclRef =
                    isDeclRefTypeOf<GenericTypeParamDeclBase>(constraintDecl->sub.type))
            {
                auto typeParamDeclVal = subContext->findLoweredDecl(genParamDeclRef.getDecl());
                SLANG_ASSERT(typeParamDeclVal && typeParamDeclVal->val);
                subBuilder->addTypeConstraintDecoration(typeParamDeclVal->val, supType);
            }

            return param;
        }
    }

    void emitGenericConstraintDecl(
        IRGenContext* subContext,
        GenericTypeConstraintDecl* constraintDecl)
    {
        auto supType = lowerType(subContext, constraintDecl->sup.type);
        auto value = emitGenericConstraintValue(subContext, constraintDecl, supType);
        subContext->setValue(constraintDecl, LoweredValInfo::simple(value));
    }

    IRGeneric* emitOuterGeneric(IRGenContext* subContext, GenericDecl* genericDecl, Decl* leafDecl)
    {
        auto subBuilder = subContext->irBuilder;

        // Of course, a generic might itself be nested inside of other generics...
        emitOuterGenerics(subContext, genericDecl, leafDecl);

        // We need to create an IR generic

        auto irGeneric = subBuilder->emitGeneric();
        subBuilder->setInsertInto(irGeneric);

        auto irBlock = subBuilder->emitBlock();
        subBuilder->setInsertInto(irBlock);

        // Now emit any parameters of the generic
        //
        // First we start with type and value parameters,
        // in the order they were declared.
        for (auto member : genericDecl->members)
        {
            if (auto typeParamDecl = as<GenericTypeParamDeclBase>(member))
            {
                IRType* typeKind = nullptr;
                if (as<GenericTypePackParamDecl>(member))
                    typeKind = subBuilder->getTypeParameterPackKind();
                else
                    typeKind = subBuilder->getTypeType();
                auto param = subBuilder->emitParam(typeKind);
                addNameHint(context, param, typeParamDecl);
                subContext->setValue(typeParamDecl, LoweredValInfo::simple(param));
            }
            else if (auto valDecl = as<GenericValueParamDecl>(member))
            {
                auto paramType = lowerType(subContext, valDecl->getType());
                auto param = subBuilder->emitParam(paramType);
                addNameHint(context, param, valDecl);
                subContext->setValue(valDecl, LoweredValInfo::simple(param));
            }
        }
        // Then we emit constraint parameters, again in
        // declaration order.
        for (auto member : genericDecl->members)
        {
            if (auto constraintDecl = as<GenericTypeConstraintDecl>(member))
            {
                emitGenericConstraintDecl(subContext, constraintDecl);
            }
        }

        return irGeneric;
    }

    IRGeneric* emitOuterInterfaceGeneric(
        IRGenContext* subContext,
        ContainerDecl* parentDecl,
        DeclRefType* interfaceType,
        Decl* leafDecl)
    {
        auto subBuilder = subContext->irBuilder;

        // Of course, a generic might itself be nested inside of other generics...
        emitOuterGenerics(subContext, parentDecl, leafDecl);

        // We need to create an IR generic

        auto irGeneric = subBuilder->emitGeneric();
        subBuilder->setInsertInto(irGeneric);

        auto irBlock = subBuilder->emitBlock();
        subBuilder->setInsertInto(irBlock);

        // The generic needs two parameters: one to represent the
        // `ThisType`, and one to represent a witness that the
        // `ThisType` conforms to the interface itself.
        //
        auto irThisTypeParam = subBuilder->emitParam(subBuilder->getTypeType());

        auto irInterfaceType = lowerType(context, interfaceType);
        auto irWitnessTableParam =
            subBuilder->emitParam(subBuilder->getWitnessTableType(irInterfaceType));
        subBuilder->addTypeConstraintDecoration(irThisTypeParam, irInterfaceType);

        // Now we need to wire up the IR parameters
        // we created to be used as the `ThisType` in
        // the body of the code.
        //
        subContext->thisType = irThisTypeParam;
        subContext->thisTypeWitness = irWitnessTableParam;

        return irGeneric;
    }

    // If the given `decl` is enclosed in any generic declarations, then
    // emit IR-level generics to represent them.
    // The `leafDecl` represents the inner-most declaration we are actually
    // trying to emit, which is the one that should receive the mangled name.
    //
    IRGeneric* emitOuterGenerics(IRGenContext* subContext, Decl* decl, Decl* leafDecl)
    {
        for (auto pp = decl->parentDecl; pp; pp = pp->parentDecl)
        {
            if (auto genericAncestor = as<GenericDecl>(pp))
            {
                return emitOuterGeneric(subContext, genericAncestor, leafDecl);
            }

            // We introduce IR generics in one other case, where the input
            // code wasn't visibly using generics: when a concrete member
            // is defined on an interface type. In that case, the resulting
            // definition needs to be generic on a parameter to represent
            // the `ThisType` of the interface.
            //
            if (auto extensionAncestor = as<ExtensionDecl>(pp))
            {
                if (auto targetDeclRefType = as<DeclRefType>(extensionAncestor->targetType))
                {
                    if (auto interfaceDeclRef = targetDeclRefType->getDeclRef().as<InterfaceDecl>())
                    {
                        return emitOuterInterfaceGeneric(
                            subContext,
                            extensionAncestor,
                            targetDeclRefType,
                            leafDecl);
                    }
                }
            }
        }

        return nullptr;
    }

    static bool isChildOf(IRInst* child, IRInst* parent)
    {
        while (child && child->getParent() != parent)
            child = child->getParent();
        return child != nullptr;
    }
    static void markInstsToClone(InstHashSet& valuesToClone, IRInst* parentBlock, IRInst* value)
    {
        if (!isChildOf(value, parentBlock))
            return;
        if (valuesToClone.add(value))
        {
            for (UInt i = 0; i < value->getOperandCount(); i++)
            {
                auto operand = value->getOperand(i);
                markInstsToClone(valuesToClone, parentBlock, operand);
            }
            if (value->getFullType())
                markInstsToClone(valuesToClone, parentBlock, value->getFullType());
            for (auto child : value->getDecorationsAndChildren())
                markInstsToClone(valuesToClone, parentBlock, child);
        }
        auto parent = parentBlock->getParent();
        while (parent && parent != parentBlock)
        {
            valuesToClone.add(parent);
            parent = parent->getParent();
        }
    }

    // If any generic declarations have been created by `emitOuterGenerics`,
    // then finish them off by emitting `return` instructions for the
    // values that they should produce.
    //
    // Return the outer-most generic (if there is one), or the original
    // value (if there were no generics), which should be the IR-level
    // representation of the original declaration.
    //
    IRInst* finishOuterGenerics(IRBuilder* subBuilder, IRInst* val, IRGeneric* parentGeneric)
    {
        IRInst* v = val;

        IRInst* returnType = v->getFullType();

        while (parentGeneric)
        {
            // Create a universal type in `outerBlock` that will be used
            // as the type of this generic inst. The return value of the
            // generic inst will have a specialized type.
            // For example, if we have a generic function
            // g0 = generic<T> { return f: T->int }
            // The type for `g0` should be:
            // g0Type = generic<T1> { return IRFuncType{T1->int} }
            // with `g0Type`, we can rewrite `g0` into:
            // ```
            //    g0 : g0Type = generic<T>
            //    {
            //       ftype = specialize(g0Type, T);
            //       return f : ftype;
            //    }
            // ```
            IRBuilder typeBuilder(subBuilder->getModule());
            IRCloneEnv cloneEnv = {};
            if (returnType)
            {
                InstHashSet valuesToClone(subBuilder->getModule());
                markInstsToClone(valuesToClone, parentGeneric->getFirstBlock(), returnType);
                // For Function Types, we always clone all generic parameters regardless of whether
                // the generic parameter appears in the function signature or not.
                if (returnType->getOp() == kIROp_FuncType || returnType->getOp() == kIROp_Generic)
                {
                    for (auto genericParam : parentGeneric->getParams())
                    {
                        markInstsToClone(
                            valuesToClone,
                            parentGeneric->getFirstBlock(),
                            genericParam);
                    }
                }
                if (valuesToClone.getCount() == 0)
                {
                    // If the new generic has no parameters, set
                    // the generic inst's type to just `returnType`.
                    parentGeneric->setFullType((IRType*)returnType);
                }
                else
                {
                    // In the general case, we need to construct a separate
                    // generic value for the return type, and set the generic's type
                    // to the newly construct generic value.
                    typeBuilder.setInsertBefore(parentGeneric);
                    auto typeGeneric = typeBuilder.emitGeneric();
                    typeGeneric->setFullType(typeBuilder.getGenericKind());
                    typeBuilder.setInsertInto(typeGeneric);
                    auto block = typeBuilder.emitBlock();

                    struct ParamCloneInfo
                    {
                        IRParam* originalParam;
                        IRParam* clonedParam;
                    };
                    ShortList<ParamCloneInfo> paramCloneInfos;

                    for (auto child : parentGeneric->getFirstBlock()->getChildren())
                    {
                        if (valuesToClone.contains(child))
                        {
                            if (child->getOp() == kIROp_Param)
                            {
                                // Params may have forward references in its type and
                                // decorations, so we just create a placeholder for it
                                // in this first pass.
                                IRParam* clonedParam = typeBuilder.emitParam(nullptr);
                                cloneEnv.mapOldValToNew[child] = clonedParam;
                                paramCloneInfos.add({(IRParam*)child, clonedParam});
                            }
                            else
                            {
                                cloneInst(&cloneEnv, &typeBuilder, child);
                            }
                        }
                    }

                    // In a second pass, clone the types and decorations on params which may
                    // contain forward references.
                    for (auto param : paramCloneInfos)
                    {
                        typeBuilder.setInsertInto(param.clonedParam);
                        param.clonedParam->setFullType((IRType*)cloneInst(
                            &cloneEnv,
                            &typeBuilder,
                            param.originalParam->getFullType()));
                        cloneInstDecorationsAndChildren(
                            &cloneEnv,
                            typeBuilder.getModule(),
                            param.originalParam,
                            param.clonedParam);
                    }

                    typeBuilder.setInsertInto(block);

                    IRInst* clonedReturnType = nullptr;
                    cloneEnv.mapOldValToNew.tryGetValue(returnType, clonedReturnType);
                    if (clonedReturnType)
                    {
                        // If the type has explicit dependency on generic parameters, use
                        // the cloned type.
                        typeBuilder.emitReturn(clonedReturnType);
                    }
                    else
                    {
                        // Otherwise just use the original type value directly.
                        typeBuilder.emitReturn(returnType);
                    }
                    parentGeneric->setFullType((IRType*)typeGeneric);
                    returnType = typeGeneric;
                }
            }

            subBuilder->setInsertInto(parentGeneric->getFirstBlock());
#if 0
            // TODO: we cannot enable this right now as it breaks too many existing code
            // that is assuming a generic function type is `IRFuncType` rather than `IRSpecialize`.
            if (v->getFullType() != returnType)
            {
                // We need to rewrite the type of the return value as
                // `specialize(returnType, ...)`.
                SLANG_ASSERT(returnType->getOp() == kIROp_Generic);
                auto oldType = v->getFullType();
                SLANG_ASSERT(isChildOf(oldType, parentGeneric->getFirstBlock()));

                List<IRInst*> specializeArgs;
                for (auto param : parentGeneric->getParams())
                {
                    IRInst* arg = nullptr;
                    if (cloneEnv.mapOldValToNew.tryGetValue(param, arg))
                    {
                        specializeArgs.add(arg);
                    }
                }
                auto specializedType = subBuilder->emitSpecializeInst(
                    subBuilder->getTypeKind(),
                    returnType,
                    (UInt)specializeArgs.getCount(),
                    specializeArgs.getBuffer());
                oldType->replaceUsesWith(specializedType);
            }
#endif
            subBuilder->emitReturn(v);
            parentGeneric->moveToEnd();

            // There might be more outer generics,
            // so we need to loop until we run out.
            v = parentGeneric;
            auto parentBlock = as<IRBlock>(v->getParent());
            if (!parentBlock)
                break;

            parentGeneric = as<IRGeneric>(parentBlock->getParent());
            if (!parentGeneric)
                break;
        }
        return v;
    }


    void addSpecializedForTargetDecorations(IRInst* inst, Decl* decl)
    {
        // If this declaration was marked as being an intrinsic for a particular
        // target, then we should reflect that here.
        for (auto targetMod : decl->getModifiersOfType<SpecializedForTargetModifier>())
        {
            // `targetMod` indicates that this particular declaration represents
            // a specialized definition of the particular function for the given
            // target, and we need to reflect that at the IR level.

            auto targetName = targetMod->targetToken.getContent();
            auto targetCap = findCapabilityName(targetName);

            getBuilder()->addTargetDecoration(inst, CapabilitySet(targetCap));
        }
    }

    // Attach target-intrinsic decorations to an instruction,
    // based on modifiers on an AST declaration.
    void addTargetIntrinsicDecorations(IRGenContext* subContext, IRInst* irInst, Decl* decl)
    {
        auto builder = getBuilder();

        for (auto targetMod : decl->getModifiersOfType<TargetIntrinsicModifier>())
        {
            String definition;
            if (targetMod->isString)
            {
                definition = targetMod->definitionString;
            }
            else if (targetMod->definitionIdent.type == TokenType::Identifier)
            {
                definition = targetMod->definitionIdent.getContent();
            }
            else
            {
                if (isCoreModuleMemberFuncDecl(decl))
                {
                    // We will mark member functions by appending a `.` to the
                    // start of their name.
                    //
                    definition.append(".");
                }

                definition.append(decl->getName()->text);
            }

            UnownedStringSlice targetName;
            auto& targetToken = targetMod->targetToken;
            if (targetToken.type != TokenType::Unknown)
            {
                targetName = targetToken.getContent();
            }

            CapabilitySet targetCaps;
            if (targetName.getLength() == 0)
            {
                targetCaps = CapabilitySet::makeEmpty();
            }
            else
            {
                CapabilityName targetCap = findCapabilityName(targetName);
                SLANG_ASSERT(targetCap != CapabilityName::Invalid);
                targetCaps = CapabilitySet(targetCap);
            }

            IRInst* scrutinee = nullptr;
            UnownedStringSlice predicate;
            if (targetMod->scrutineeDeclRef)
            {
                const auto s = subContext->findLoweredDecl(targetMod->scrutineeDeclRef.getDecl());
                if (s && s->flavor == LoweredValInfo::Flavor::Simple)
                {
                    scrutinee = s->val;
                    predicate = targetMod->predicateToken.getContent();
                }
            }

            builder->addTargetIntrinsicDecoration(
                irInst,
                targetCaps,
                definition.getUnownedSlice(),
                predicate,
                scrutinee);
        }

        if (const auto nvapiMod = decl->findModifier<NVAPIMagicModifier>())
        {
            builder->addNVAPIMagicDecoration(irInst, decl->getName()->text.getUnownedSlice());
        }
        if (const auto requirePrelude = decl->findModifier<RequirePreludeAttribute>())
        {
            builder->addRequirePreludeDecoration(
                irInst,
                requirePrelude->capabilitySet,
                requirePrelude->prelude.getUnownedSlice());
        }
    }

    void addTargetRequirementDecorations(IRInst* inst, Decl* decl)
    {
        // If this declaration requires certain GLSL extension (or a particular GLSL version)
        // for it to be usable, then declare that here. Similarly for SPIR-V or CUDA
        //
        // TODO: We should wrap this an `SpecializedForTargetModifier` together into a single
        // case for enumerating the "capabilities" that a declaration requires.
        //
        for (auto extensionMod : decl->getModifiersOfType<RequiredGLSLExtensionModifier>())
        {
            getBuilder()->addRequireGLSLExtensionDecoration(
                inst,
                extensionMod->extensionNameToken.getContent());
        }
        for (auto versionMod : decl->getModifiersOfType<RequiredGLSLVersionModifier>())
        {
            getBuilder()->addRequireGLSLVersionDecoration(
                inst,
                Int(getIntegerLiteralValue(versionMod->versionNumberToken)));
        }
        for (auto versionMod : decl->getModifiersOfType<RequiredSPIRVVersionModifier>())
        {
            getBuilder()->addRequireSPIRVVersionDecoration(inst, versionMod->version);
        }
        for (auto extensionMod : decl->getModifiersOfType<RequiredWGSLExtensionModifier>())
        {
            getBuilder()->addRequireWGSLExtensionDecoration(
                inst,
                extensionMod->extensionNameToken.getContent());
        }
        for (auto versionMod : decl->getModifiersOfType<RequiredCUDASMVersionModifier>())
        {
            getBuilder()->addRequireCUDASMVersionDecoration(inst, versionMod->version);
        }
    }

    void addBitFieldAccessorDecorations(IRInst* irFunc, Decl* decl)
    {
        // If this is an accessor and the parent is describing some bitfield,
        // we can move the bitfield modifiers to the accessor function.
        if (as<AccessorDecl>(decl))
        {
            if (const auto bfm = decl->parentDecl->findModifier<BitFieldModifier>())
            {
                getBuilder()->addDecoration(
                    irFunc,
                    kIROp_BitFieldAccessorDecoration,
                    getSimpleVal(context, ensureDecl(context, bfm->backingDeclRef.getDecl())),
                    getBuilder()->getIntValue(getBuilder()->getIntType(), bfm->width),
                    getBuilder()->getIntValue(getBuilder()->getIntType(), bfm->offset));
            }
        }
    }

    /// Is `decl` a member function (or effectively a member function) when considered as a core
    /// module declaration?
    bool isCoreModuleMemberFuncDecl(Decl* inDecl)
    {
        auto decl = as<CallableDecl>(inDecl);
        if (!decl)
            return false;

        // Constructors aren't really member functions, insofar
        // as they aren't called with a `this` parameter.
        if (as<ConstructorDecl>(decl))
            return false;

        // Exclude `static` functions for same reason.
        if (decl->findModifier<HLSLStaticModifier>())
        {
            return false;
        }

        auto dd = decl->parentDecl;
        for (;;)
        {
            if (auto genericDecl = as<GenericDecl>(dd))
            {
                dd = genericDecl->parentDecl;
                continue;
            }

            if (auto subscriptDecl = as<SubscriptDecl>(dd))
            {
                dd = subscriptDecl->parentDecl;
            }

            break;
        }

        // Note: the use of `AggTypeDeclBase` here instead of just
        // `AggTypeDecl` means that we consider a declaration that
        // is under a `struct` *or* an `extension` to be a member
        // function for our purposes.
        //
        if (as<AggTypeDeclBase>(dd))
            return true;

        return false;
    }

    /// Add a "catch-all" decoration for a core module function if it would be needed
    void addCatchAllIntrinsicDecorationIfNeeded(IRInst* irInst, FunctionDeclBase* decl)
    {
        // We don't need an intrinsic decoration on a function that has a body,
        // since the body can be used as the "catch-all" case.
        //
        if (decl->body)
            return;

        // Only core module declarations should get any kind of catch-all
        // treatment by default. Declarations in user case are responsible
        // for marking things as target intrinsics if they want to go down
        // that (unsupported) route.
        //
        if (!isFromCoreModule(decl))
            return;

        // No need to worry about functions that lower to intrinsic IR opcodes
        // (or pseudo-ops).
        //
        if (decl->findModifier<IntrinsicOpModifier>())
            return;

        // We also don't need an intrinsic decoration if the function already
        // had a catch-all case on one of its target overloads.
        //
        for (auto f = decl->primaryDecl; f; f = f->nextDecl)
        {
            for (auto targetMod : f->getModifiersOfType<TargetIntrinsicModifier>())
            {
                // If we find a catch-all case (marked as either *no* target
                // token or an empty target name), then we should bail out.
                //
                if (targetMod->targetToken.type == TokenType::Unknown)
                    return;
                else if (!targetMod->targetToken.hasContent())
                    return;
            }
        }

        String definition;

        // If we have a member function, then we want the default intrinsic
        // definition to reflect this fact so that we can emit it correctly
        // (the assumption is that a catch-all definition of a member function
        // is itself implemented as a member function).
        //
        if (isCoreModuleMemberFuncDecl(decl))
        {
            // We will mark member functions by appending a `.` to the
            // start of their name.
            //
            definition.append(".");
        }

        // We want to output the name of the declaration,
        // but in some cases the actual `decl` that has
        // to be emitted is not the one with the name.
        //
        // In particular, an accessor declaration (e.g.,
        // a `get`ter` in a subscript or property) doesn't
        // have a name, but its parent should.
        //
        Decl* declForName = decl;
        if (const auto accessorDecl = as<AccessorDecl>(decl))
            declForName = decl->parentDecl;

        definition.append(getText(declForName->getName()));

        getBuilder()->addTargetIntrinsicDecoration(
            irInst,
            CapabilitySet(CapabilityName::textualTarget),
            definition.getUnownedSlice());
    }

    void addParamNameHint(IRInst* inst, IRLoweringParameterInfo const& info)
    {
        if (auto decl = info.decl)
        {
            addNameHint(context, inst, decl);
        }
        else if (info.isThisParam)
        {
            addNameHint(context, inst, "this");
        }
    }

    IRFloatLit* _getFloatFromAttribute(IRBuilder* builder, Attribute* attrib, Index index = 0)
    {
        SLANG_ASSERT(attrib->args.getCount() > index);
        Expr* expr = attrib->args[index];

        if (auto floatLitExpr = as<FloatingPointLiteralExpr>(expr))
        {
            return as<IRFloatLit>(
                builder->getFloatValue(builder->getFloatType(), floatLitExpr->value));
        }

        auto intLitExpr = as<IntegerLiteralExpr>(expr);
        SLANG_ASSERT(intLitExpr);
        return as<IRFloatLit>(builder->getFloatValue(
            builder->getFloatType(),
            (IRFloatingPointValue)(intLitExpr->value)));
    }

    IRIntLit* _getIntLitFromAttribute(IRBuilder* builder, Attribute* attrib, Index index = 0)
    {
        SLANG_ASSERT(attrib->args.getCount() > index);
        Expr* expr = attrib->args[index];
        auto intLitExpr = as<IntegerLiteralExpr>(expr);
        SLANG_ASSERT(intLitExpr);
        return as<IRIntLit>(builder->getIntValue(builder->getIntType(), intLitExpr->value));
    }

    IRStringLit* _getStringLitFromAttribute(IRBuilder* builder, Attribute* attrib, Index index = 0)
    {
        SLANG_ASSERT(attrib->args.getCount() > index);
        Expr* expr = attrib->args[index];

        auto stringLitExpr = as<StringLiteralExpr>(expr);
        SLANG_ASSERT(stringLitExpr);
        return as<IRStringLit>(builder->getStringValue(stringLitExpr->value.getUnownedSlice()));
    }

    bool isClassType(IRType* type)
    {
        type = (IRType*)unwrapAttributedType(type);
        if (auto specialize = as<IRSpecialize>(type))
        {
            return findSpecializeReturnVal(specialize)->getOp() == kIROp_ClassType;
        }
        else if (auto genericInst = as<IRGeneric>(type))
        {
            return findGenericReturnVal(genericInst)->getOp() == kIROp_ClassType;
        }
        return type->getOp() == kIROp_ClassType;
    }

    LoweredValInfo lowerFuncDeclInContext(
        IRGenContext* subContext,
        IRBuilder* subBuilder,
        FunctionDeclBase* decl,
        bool emitBody = true)
    {
        IRGeneric* outerGeneric = nullptr;
        subContext->funcDecl = decl;

        if (auto derivativeRequirement = as<DerivativeRequirementDecl>(decl))
            outerGeneric = emitOuterGenerics(
                subContext,
                derivativeRequirement->originalRequirementDecl,
                derivativeRequirement->originalRequirementDecl);
        else
            outerGeneric = emitOuterGenerics(subContext, decl, decl);

        // If our function is differentiable, register a callback so the derivative
        // annotations for types can be lowered.
        //
        if (decl->findModifier<DifferentiableAttribute>() && !isInterfaceRequirement(decl))
        {
            auto diffAttr = decl->findModifier<DifferentiableAttribute>();

            auto diffTypeWitnessMap = diffAttr->getMapTypeToIDifferentiableWitness();
            OrderedDictionary<Type*, SubtypeWitness*> resolveddiffTypeWitnessMap;

            // Go through each entry in the map and resolve the key.
            for (auto& entry : diffTypeWitnessMap)
            {
                auto resolvedKey = as<Type>(entry.key->resolve());
                resolveddiffTypeWitnessMap[resolvedKey] =
                    as<SubtypeWitness>(as<Val>(entry.value)->resolve());
            }

            subContext->registerTypeCallback(
                [=](IRGenContext* context, Type* type, IRType* irType)
                {
                    if (resolveddiffTypeWitnessMap.containsKey(type))
                    {
                        auto irWitness = lowerVal(subContext, resolveddiffTypeWitnessMap[type]).val;
                        if (irWitness)
                        {
                            IRInst* args[] = {irType, irWitness};
                            context->irBuilder->emitIntrinsicInst(
                                context->irBuilder->getVoidType(),
                                kIROp_DifferentiableTypeAnnotation,
                                2,
                                args);
                        }
                    }

                    return irType;
                });
        }

        FuncDeclBaseTypeInfo info;
        _lowerFuncDeclBaseTypeInfo(
            subContext,
            createDefaultSpecializedDeclRef(context, nullptr, decl),
            info);

        // need to create an IR function here

        IRFunc* irFunc = subBuilder->createFunc();
        addNameHint(subContext, irFunc, decl);
        addLinkageDecoration(subContext, irFunc, decl);
        maybeAddDebugLocationDecoration(subContext, irFunc);

        // Register the value now, to avoid any possible infinite recursion when lowering the body
        // or attributes.
        context->setGlobalValue(decl, LoweredValInfo::simple(findOuterMostGeneric(irFunc)));

        // Always force inline diff setter accessor to prevent downstream compiler from complaining
        // fields are not fully initialized for the first `inout` parameter.
        if (as<SetterDecl>(decl))
        {
            if (!decl->findModifier<ForceInlineAttribute>())
            {
                getBuilder()->addForceInlineDecoration(irFunc);
            }
        }

        // For diagnostics
        if (as<StructDecl>(decl->parentDecl))
            getBuilder()->addSimpleDecoration<IRMethodDecoration>(irFunc);

        auto irFuncType = info.type;
        auto& irResultType = info.resultType;
        auto& parameterLists = info.parameterLists;
        auto& paramTypes = info.paramTypes;

        irFunc->setFullType(irFuncType);

        subBuilder->setInsertInto(irFunc);

        // If a function is imported from another module then
        // we usually don't want to emit it as a definition, and
        // will instead only emit a declaration for it with an
        // appropriate `[import(...)]` linkage decoration.
        //
        // However, if the function is marked with `[__unsafeForceInlineEarly]`
        // then we need to make sure the IR for its definition is available
        // to the mandatory optimization passes.
        //
        // TODO: The design here means that we will re-emit the inline
        // function from its AST in every module that uses it. We should
        // instead have logic to clone the target function in from the
        // pre-generated IR for the module that defines it (or do some kind
        // of minimal linking to bring in the inline functions).
        //
        if (!decl->body)
        {
            // This is a function declaration without a body.
            // In Slang we currently try not to support forward declarations
            // (although we might have to give in eventually), so
            // this case should really only occur for builtin declarations.
        }
        else if (isDeclInDifferentModule(context, decl) && !isForceInlineEarly(decl))
        {
        }
        else if (emitBody)
        {
            // This is a function definition, so we need to actually
            // construct IR for the body...
            IRBlock* entryBlock = subBuilder->emitBlock();
            subBuilder->setInsertInto(entryBlock);

            UInt paramTypeIndex = 0;
            for (auto paramInfo : parameterLists.params)
            {
                auto irParamType = paramTypes[paramTypeIndex++];

                LoweredValInfo paramVal;

                IRParam* irParam = nullptr;

                switch (paramInfo.direction)
                {
                default:
                    {
                        // The parameter is being used for input/output purposes,
                        // so it will lower to an actual parameter with a pointer type.
                        //
                        // TODO: Is this the best representation we can use?

                        irParam = subBuilder->emitParam(irParamType);
                        if (auto paramDecl = paramInfo.decl)
                        {
                            addVarDecorations(context, irParam, paramDecl);
                            subBuilder->addHighLevelDeclDecoration(irParam, paramDecl);
                            irParam->sourceLoc = paramDecl->loc;
                        }
                        addParamNameHint(irParam, paramInfo);

                        paramVal = LoweredValInfo::ptr(irParam);

                        if (paramInfo.isReturnDestination)
                            subContext->returnDestination = paramVal;

                        if (paramInfo.declaredDirection == kParameterDirection_In &&
                            paramInfo.direction == kParameterDirection_ConstRef)
                        {
                            // If the parameter is originally declared as "in", but we are
                            // lowering it as constref for any reason (e.g. it is a varying input),
                            // then we need to emit a local variable to hold the original value, so
                            // that we can still generate correct code when the user trys to mutate
                            // the variable.
                            // The local variable introduced here is cleaned up by the SSA pass, if
                            // we can determine that there are no actual writes into the local var.
                            auto irLocal =
                                subBuilder->emitVar(tryGetPointedToType(subBuilder, irParamType));
                            auto localVal = LoweredValInfo::ptr(irLocal);
                            assign(subContext, localVal, paramVal);
                            paramVal = localVal;
                        }
                        // TODO: We might want to copy the pointed-to value into
                        // a temporary at the start of the function, and then copy
                        // back out at the end, so that we don't have to worry
                        // about things like aliasing in the function body.
                        //
                        // For now we will just use the storage that was passed
                        // in by the caller, knowing that our current lowering
                        // at call sites will guarantee a fresh/unique location.
                    }
                    break;

                case kParameterDirection_In:
                    {
                        // Simple case of a by-value input parameter.
                        //
                        // We start by declaring an IR parameter of the same type.
                        //
                        auto paramDecl = paramInfo.decl;
                        irParam = subBuilder->emitParam(irParamType);
                        if (paramDecl)
                        {
                            addVarDecorations(context, irParam, paramDecl);
                            subBuilder->addHighLevelDeclDecoration(irParam, paramDecl);
                            irParam->sourceLoc = paramDecl->loc;
                        }
                        addParamNameHint(irParam, paramInfo);
                        paramVal = LoweredValInfo::simple(irParam);
                        //
                        // HLSL allows a function parameter to be used as a local
                        // variable in the function body (just like C/C++), so
                        // we need to support that case as well.
                        //
                        // However, if we notice that the parameter was marked
                        // `const`, then we can skip this step.
                        //
                        // TODO: we should consider having all parameter be implicitly
                        // immutable except in a specific "compatibility mode."
                        //
                        if (paramDecl && paramDecl->findModifier<ConstModifier>())
                        {
                            // This parameter was declared to be immutable,
                            // so there should be no assignment to it in the
                            // function body, and we don't need a temporary.
                        }
                        else
                        {
                            // The parameter migth get used as a temporary in
                            // the function body. We will allocate a mutable
                            // local variable for is value, and then assign
                            // from the parameter to the local at the start
                            // of the function.
                            //
                            auto irLocal = subBuilder->emitVar(irParamType);
                            auto localVal = LoweredValInfo::ptr(irLocal);
                            assign(subContext, localVal, paramVal);
                            //
                            // When code later in the body of the function refers
                            // to the parameter declaration, it will actually refer
                            // to the value stored in the local variable.
                            //
                            paramVal = localVal;
                        }
                    }
                    break;
                }

                if (auto paramDecl = paramInfo.decl)
                {
                    subContext->setValue(paramDecl, paramVal);
                }

                if (paramInfo.isThisParam)
                {
                    subContext->thisVal = paramVal;
                }
            }

            {

                auto attr = decl->findModifier<PatchConstantFuncAttribute>();

                // I needed to test for patchConstantFuncDecl here
                // because it is only set if validateEntryPoint is called with Hull as the required
                // stage If I just build domain shader, and then the attribute exists, but
                // patchConstantFuncDecl is not set and thus leads to a crash.
                if (attr && attr->patchConstantFuncDecl)
                {
                    // We need to lower the function
                    FuncDecl* patchConstantFunc = attr->patchConstantFuncDecl;
                    assert(patchConstantFunc);

                    // Convert the patch constant function into IRInst
                    IRInst* irPatchConstantFunc =
                        getSimpleVal(context, ensureDecl(subContext, patchConstantFunc));

                    // Attach a decoration so that our IR function references
                    // the patch constant function.
                    //
                    subContext->irBuilder->addPatchConstantFuncDecoration(
                        irFunc,
                        irPatchConstantFunc);
                }
            }


            // We will now set about emitting the code for the body of
            // the function/callable.
            //
            // In the case of an initializer ("constructor") declaration,
            // the `this` value is not a parameter, but rather a placeholder
            // for the value that will be returned. We thus need to set up
            // a local variable to represent this value.
            //
            auto constructorDecl = as<ConstructorDecl>(decl);
            if (constructorDecl)
            {
                if (subContext->returnDestination.flavor != LoweredValInfo::Flavor::None)
                    subContext->thisVal = subContext->returnDestination;
                else
                {
                    auto thisVar = subContext->irBuilder->emitVar(irResultType);
                    subContext->thisVal = LoweredValInfo::ptr(thisVar);

                    // For class-typed objects, we need to allocate it from heap.
                    if (isClassType(irResultType))
                    {
                        auto allocatedObj = subContext->irBuilder->emitAllocObj(irResultType);
                        subContext->irBuilder->emitStore(thisVar, allocatedObj);
                    }
                }

                // Used for diagnostics
                getBuilder()->addConstructorDecoration(
                    irFunc,
                    constructorDecl->containsFlavor(
                        ConstructorDecl::ConstructorFlavor::SynthesizedDefault));
            }

            // We lower whatever statement was stored on the declaration
            // as the body of the new IR function.
            //
            lowerStmt(subContext, decl->body);

            // We need to carefully add a terminator instruction to the end
            // of the body, in case the user didn't do so.
            //
            if (!subContext->irBuilder->getBlock()->getTerminator())
            {
                if (constructorDecl)
                {
                    // A constructor declaration should return the
                    // value of the `this` variable that was set
                    // up at the start.
                    //
                    // TODO: This should also apply if any code
                    // path in an initializer/constructor attempts
                    // to do an early `return;`.
                    //
                    if (subContext->returnDestination.flavor != LoweredValInfo::Flavor::None)
                        subContext->irBuilder->emitReturn();
                    else
                    {
                        subContext->irBuilder->emitReturn(
                            getSimpleVal(subContext, subContext->thisVal));
                    }
                }
                else if (as<IRVoidType>(irResultType))
                {
                    // `void`-returning function can get an implicit
                    // return on exit of the body statement.
                    IRInst* returnInst = subContext->irBuilder->emitReturn();

                    if (BlockStmt* blockStmt = as<BlockStmt>(decl->body))
                    {
                        returnInst->sourceLoc = blockStmt->closingSourceLoc;
                    }
                    else
                    {
                        returnInst->sourceLoc = SourceLoc();
                    }
                }
                else
                {
                    // Value-returning function is expected to `return`
                    // on every control-flow path. We need to enforce
                    // this by putting an `unreachable` terminator here,
                    // and then emit a dataflow error if this block
                    // can't be eliminated.
                    subContext->irBuilder->emitMissingReturn();
                }
            }
        }

        subContext->registerTypeCallback(nullptr);

        getBuilder()->addHighLevelDeclDecoration(irFunc, decl);

        addSpecializedForTargetDecorations(irFunc, decl);

        // If this declaration was marked as having a target-specific lowering
        // for a particular target, then handle that here.
        addTargetIntrinsicDecorations(subContext, irFunc, decl);

        addCatchAllIntrinsicDecorationIfNeeded(irFunc, decl);

        addTargetRequirementDecorations(irFunc, decl);

        bool isInline = false;

        addBitFieldAccessorDecorations(irFunc, decl);

        IRNumThreadsDecoration* numThreadsDecor = nullptr;
        IRDecoration* derivativeGroupQuadDecor = nullptr;
        IRDecoration* derivativeGroupLinearDecor = nullptr;
        for (auto modifier : decl->modifiers)
        {
            if (as<RequiresNVAPIAttribute>(modifier))
            {
                getBuilder()->addSimpleDecoration<IRRequiresNVAPIDecoration>(irFunc);
            }
            else if (as<AlwaysFoldIntoUseSiteAttribute>(modifier))
            {
                getBuilder()->addSimpleDecoration<IRAlwaysFoldIntoUseSiteDecoration>(irFunc);
            }
            else if (as<NoInlineAttribute>(modifier))
            {
                getBuilder()->addSimpleDecoration<IRNoInlineDecoration>(irFunc);
            }
            else if (as<DerivativeGroupQuadAttribute>(modifier))
            {
                derivativeGroupQuadDecor =
                    getBuilder()->addSimpleDecoration<IRDerivativeGroupQuadDecoration>(irFunc);
            }
            else if (as<DerivativeGroupLinearAttribute>(modifier))
            {
                derivativeGroupLinearDecor =
                    getBuilder()->addSimpleDecoration<IRDerivativeGroupLinearDecoration>(irFunc);
            }
            else if (as<MaximallyReconvergesAttribute>(modifier))
            {
                getBuilder()->addSimpleDecoration<IRMaximallyReconvergesDecoration>(irFunc);
            }
            else if (as<QuadDerivativesAttribute>(modifier))
            {
                getBuilder()->addSimpleDecoration<IRQuadDerivativesDecoration>(irFunc);
            }
            else if (as<RequireFullQuadsAttribute>(modifier))
            {
                getBuilder()->addSimpleDecoration<IRRequireFullQuadsDecoration>(irFunc);
            }
            else if (as<NoRefInlineAttribute>(modifier))
            {
                getBuilder()->addSimpleDecoration<IRNoRefInlineDecoration>(irFunc);
            }
            else if (auto instanceAttr = as<InstanceAttribute>(modifier))
            {
                IRIntLit* intLit = _getIntLitFromAttribute(getBuilder(), instanceAttr);
                getBuilder()->addDecoration(irFunc, kIROp_InstanceDecoration, intLit);
            }
            else if (auto maxVertCountAttr = as<MaxVertexCountAttribute>(modifier))
            {
                IRIntLit* intLit = _getIntLitFromAttribute(getBuilder(), maxVertCountAttr);
                getBuilder()->addDecoration(irFunc, kIROp_MaxVertexCountDecoration, intLit);
            }
            else if (auto numThreadsAttr = as<NumThreadsAttribute>(modifier))
            {
                LoweredValInfo extents[3];

                for (int i = 0; i < 3; ++i)
                {
                    extents[i] = numThreadsAttr->specConstExtents[i]
                                     ? emitDeclRef(
                                           context,
                                           numThreadsAttr->specConstExtents[i],
                                           lowerType(
                                               context,
                                               getType(
                                                   context->astBuilder,
                                                   numThreadsAttr->specConstExtents[i])))
                                     : lowerVal(context, numThreadsAttr->extents[i]);
                }

                numThreadsDecor = as<IRNumThreadsDecoration>(getBuilder()->addNumThreadsDecoration(
                    irFunc,
                    getSimpleVal(context, extents[0]),
                    getSimpleVal(context, extents[1]),
                    getSimpleVal(context, extents[2])));
                numThreadsDecor->sourceLoc = numThreadsAttr->loc;
            }
            else if (auto waveSizeAttr = as<WaveSizeAttribute>(modifier))
            {
                getBuilder()->addWaveSizeDecoration(
                    irFunc,
                    getSimpleVal(context, lowerVal(context, waveSizeAttr->numLanes)));
            }
            else if (as<ReadNoneAttribute>(modifier))
            {
                getBuilder()->addSimpleDecoration<IRReadNoneDecoration>(irFunc);
            }
            else if (as<NoSideEffectAttribute>(modifier))
            {
                getBuilder()->addSimpleDecoration<IRNoSideEffectDecoration>(irFunc);
            }
            else if (as<EarlyDepthStencilAttribute>(modifier))
            {
                getBuilder()->addSimpleDecoration<IREarlyDepthStencilDecoration>(irFunc);
            }
            else if (auto domainAttr = as<DomainAttribute>(modifier))
            {
                IRStringLit* stringLit = _getStringLitFromAttribute(getBuilder(), domainAttr);
                getBuilder()->addDecoration(irFunc, kIROp_DomainDecoration, stringLit);
            }
            else if (auto partitionAttr = as<PartitioningAttribute>(modifier))
            {
                IRStringLit* stringLit = _getStringLitFromAttribute(getBuilder(), partitionAttr);
                getBuilder()->addDecoration(irFunc, kIROp_PartitioningDecoration, stringLit);
            }
            else if (auto outputTopAttr = as<OutputTopologyAttribute>(modifier))
            {
                IRStringLit* stringLit = _getStringLitFromAttribute(getBuilder(), outputTopAttr);
                const auto topologyType =
                    convertOutputTopologyStringToEnum(stringLit->getStringSlice());
                IRInst* topologyTypeInst = getBuilder()->getIntValue(
                    getBuilder()->getIntType(),
                    IRIntegerValue(topologyType));

                auto outputTopologyDecoration = getBuilder()->addDecoration(
                    irFunc,
                    kIROp_OutputTopologyDecoration,
                    stringLit,
                    topologyTypeInst);
                outputTopologyDecoration->sourceLoc = outputTopAttr->loc;
            }
            else if (auto maxTessFactortAttr = as<MaxTessFactorAttribute>(modifier))
            {
                IRFloatLit* floatLit = _getFloatFromAttribute(getBuilder(), maxTessFactortAttr);
                getBuilder()->addDecoration(irFunc, kIROp_MaxTessFactorDecoration, floatLit);
            }
            else if (auto outputCtrlPtAttr = as<OutputControlPointsAttribute>(modifier))
            {
                IRIntLit* intLit = _getIntLitFromAttribute(getBuilder(), outputCtrlPtAttr);
                getBuilder()->addDecoration(irFunc, kIROp_OutputControlPointsDecoration, intLit);
            }
            else if (auto spvInstOpAttr = as<SPIRVInstructionOpAttribute>(modifier))
            {
                auto builder = getBuilder();
                IRIntLit* intLit = _getIntLitFromAttribute(builder, spvInstOpAttr, 0);

                IRStringLit* setStringLit = nullptr;
                if (spvInstOpAttr->args.getCount() > 1)
                {
                    IRStringLit* checkSetStringLit =
                        _getStringLitFromAttribute(builder, spvInstOpAttr, 1);
                    if (checkSetStringLit && checkSetStringLit->getStringSlice().getLength() > 0)
                    {
                        setStringLit = checkSetStringLit;
                    }
                }

                // If it has a `set` defined, set it on the decoration
                if (setStringLit)
                {
                    builder->addDecoration(irFunc, kIROp_SPIRVOpDecoration, intLit, setStringLit);
                }
                else
                {
                    builder->addDecoration(irFunc, kIROp_SPIRVOpDecoration, intLit);
                }
            }
            else if (as<UnsafeForceInlineEarlyAttribute>(modifier))
            {
                getBuilder()->addDecoration(irFunc, kIROp_UnsafeForceInlineEarlyDecoration);
                isInline = true;
            }
            else if (as<ForceInlineAttribute>(modifier))
            {
                getBuilder()->addDecoration(irFunc, kIROp_ForceInlineDecoration);
                isInline = true;
            }
            else if (as<TreatAsDifferentiableAttribute>(modifier))
            {
                getBuilder()->addDecoration(irFunc, kIROp_TreatAsDifferentiableDecoration);
            }
            else if (auto intrinsicOp = as<IntrinsicOpModifier>(modifier))
            {
                auto op = getBuilder()->getIntValue(getBuilder()->getIntType(), intrinsicOp->op);
                getBuilder()->addDecoration(irFunc, kIROp_IntrinsicOpDecoration, op);
                isInline = true;
            }
            else if (
                as<UserDefinedDerivativeAttribute>(modifier) ||
                as<PrimalSubstituteAttribute>(modifier))
            {
                // We need to lower the decl ref to the custom derivative function to IR.
                // The IR insts correspond to the decl ref is not part of the function we
                // are processing. If we emit it directly to within the function, it could
                // mess up the assumption on the form of the IR (e.g. having non decoration insts
                // appearing in the middle of decoration insts). so we emit the decl ref to the
                // function's parent for now.

                subContext->irBuilder->setInsertInto(irFunc->getParent());
                Expr* funcExpr = nullptr;
                if (auto udAttr = as<UserDefinedDerivativeAttribute>(modifier))
                    funcExpr = udAttr->funcExpr;
                else if (auto primalAttr = as<PrimalSubstituteAttribute>(modifier))
                    funcExpr = primalAttr->funcExpr;
                DeclRefExpr* declRefExpr = as<DeclRefExpr>(funcExpr);
                auto funcType = lowerType(subContext, funcExpr->type);
                auto loweredVal = emitDeclRef(subContext, declRefExpr->declRef, funcType);

                SLANG_RELEASE_ASSERT(loweredVal.flavor == LoweredValInfo::Flavor::Simple);

                IRInst* derivativeFunc = loweredVal.val;

                if (as<ForwardDerivativeAttribute>(modifier))
                    getBuilder()->addForwardDerivativeDecoration(irFunc, derivativeFunc);
                else if (as<BackwardDerivativeAttribute>(modifier))
                    getBuilder()->addUserDefinedBackwardDerivativeDecoration(
                        irFunc,
                        derivativeFunc);
                else
                    getBuilder()->addPrimalSubstituteDecoration(irFunc, derivativeFunc);

                // Reset cursor.
                subContext->irBuilder->setInsertInto(irFunc);
            }
            else if (as<ForwardDifferentiableAttribute>(modifier))
            {
                getBuilder()->addForwardDifferentiableDecoration(irFunc);
            }
            else if (as<BackwardDifferentiableAttribute>(modifier))
            {
                getBuilder()->addBackwardDifferentiableDecoration(irFunc);
            }
            else if (as<TreatAsDifferentiableAttribute>(modifier))
            {
                getBuilder()->addDecoration(irFunc, kIROp_TreatAsDifferentiableDecoration);
            }
            else if (as<PreferCheckpointAttribute>(modifier))
            {
                getBuilder()->addDecoration(irFunc, kIROp_PreferCheckpointDecoration);
            }
            else if (auto attr = as<PreferRecomputeAttribute>(modifier))
            {
                getBuilder()->addDecoration(
                    irFunc,
                    kIROp_PreferRecomputeDecoration,
                    getBuilder()->getIntValue(
                        getBuilder()->getIntType(),
                        attr->sideEffectBehavior));
            }
            else if (auto extensionMod = as<RequiredGLSLExtensionModifier>(modifier))
                getBuilder()->addRequireGLSLExtensionDecoration(
                    irFunc,
                    extensionMod->extensionNameToken.getContent());
            else if (auto versionMod = as<RequiredGLSLVersionModifier>(modifier))
                getBuilder()->addRequireGLSLVersionDecoration(
                    irFunc,
                    Int(getIntegerLiteralValue(versionMod->versionNumberToken)));
            else if (auto spvVersion = as<RequiredSPIRVVersionModifier>(modifier))
                getBuilder()->addRequireSPIRVVersionDecoration(irFunc, spvVersion->version);
            else if (auto wgslExtensionMod = as<RequiredWGSLExtensionModifier>(modifier))
                getBuilder()->addRequireWGSLExtensionDecoration(
                    irFunc,
                    wgslExtensionMod->extensionNameToken.getContent());
            else if (auto cudasmVersion = as<RequiredCUDASMVersionModifier>(modifier))
                getBuilder()->addRequireCUDASMVersionDecoration(irFunc, cudasmVersion->version);
            else if (as<NonDynamicUniformAttribute>(modifier))
                getBuilder()->addDecoration(irFunc, kIROp_NonDynamicUniformReturnDecoration);
        }

        verifyComputeDerivativeGroupModifiers(
            getSink(),
            decl->loc,
            derivativeGroupQuadDecor,
            derivativeGroupLinearDecor,
            numThreadsDecor);

        if (!isInline)
        {
            // If there are any constant expr rate parameters, we should inline this function.
            // TODO: consider specializing them instead of inlining.
            for (auto param : decl->getParameters())
            {
                if (param->hasModifier<ConstExprModifier>())
                {
                    getBuilder()->addDecoration(irFunc, kIROp_ForceInlineDecoration);
                    isInline = true;
                    break;
                }
            }
        }

        // For convenience, ensure that any additional global
        // values that were emitted while outputting the function
        // body appear before the function itself in the list
        // of global values.
        irFunc->moveToEnd();

        // If this function is defined inside an interface, add a reference to the IRFunc from
        // the interface's type definition.
        auto finalVal = finishOuterGenerics(subBuilder, irFunc, outerGeneric);

        for (auto modifier : decl->modifiers)
        {
            if (as<DerivativeOfAttribute>(modifier) || as<PrimalSubstituteOfAttribute>(modifier))
            {
                Expr* funcExpr = nullptr;
                Expr* backDeclRef = nullptr;
                if (auto attr = as<DerivativeOfAttribute>(modifier))
                {
                    funcExpr = attr->funcExpr;
                    backDeclRef = attr->backDeclRef;
                }
                else if (auto primalAttr = as<PrimalSubstituteOfAttribute>(modifier))
                {
                    funcExpr = primalAttr->funcExpr;
                    backDeclRef = primalAttr->backDeclRef;
                }

                if (auto originalDeclRefExpr = as<DeclRefExpr>(funcExpr))
                {
                    NestedContext originalContextFunc(this);
                    auto originalSubBuilder = originalContextFunc.getBuilder();
                    auto originalSubContext = originalContextFunc.getContext();
                    if (auto outterGeneric = getOuterGeneric(irFunc))
                        originalSubBuilder->setInsertBefore(outterGeneric);
                    else
                        originalSubBuilder->setInsertBefore(irFunc);
                    auto originalFuncDecl =
                        as<FunctionDeclBase>(originalDeclRefExpr->declRef.getDecl());
                    SLANG_RELEASE_ASSERT(originalFuncDecl);

                    auto originalFuncVal = lowerFuncDeclInContext(
                                               originalSubContext,
                                               originalSubBuilder,
                                               originalFuncDecl,
                                               false)
                                               .val;
                    if (auto originalFuncGeneric = as<IRGeneric>(originalFuncVal))
                    {
                        originalFuncVal = findGenericReturnVal(originalFuncGeneric);
                    }
                    originalSubBuilder->setInsertBefore(originalFuncVal);
                    auto derivativeFuncVal = lowerRValueExpr(originalSubContext, backDeclRef);
                    if (as<ForwardDerivativeOfAttribute>(modifier))
                    {
                        originalSubBuilder->addForwardDerivativeDecoration(
                            originalFuncVal,
                            derivativeFuncVal.val);
                        getBuilder()->addForwardDifferentiableDecoration(irFunc);
                    }
                    else if (as<BackwardDerivativeOfAttribute>(modifier))
                    {
                        originalSubBuilder->addUserDefinedBackwardDerivativeDecoration(
                            originalFuncVal,
                            derivativeFuncVal.val);
                    }
                    else
                    {
                        originalSubBuilder->addPrimalSubstituteDecoration(
                            originalFuncVal,
                            derivativeFuncVal.val);
                    }
                }
                subContext->irBuilder->setInsertInto(irFunc);
                finalVal->moveToEnd();
            }
        }
        return LoweredValInfo::simple(finalVal);
    }

    LoweredValInfo lowerFuncDecl(FunctionDeclBase* decl)
    {
        // We are going to use a nested builder, because we will
        // change the parent node that things get nested into.
        //
        NestedContext nestedContextFunc(this);
        auto subBuilder = nestedContextFunc.getBuilder();
        auto subContext = nestedContextFunc.getContext();
        return lowerFuncDeclInContext(subContext, subBuilder, decl);
    }

    LoweredValInfo visitGenericDecl(GenericDecl* genDecl)
    {
        // TODO: Should this just always visit/lower the inner decl?

        if (auto innerFuncDecl = as<FunctionDeclBase>(genDecl->inner))
            return ensureDecl(context, innerFuncDecl);
        else if (auto innerStructDecl = as<StructDecl>(genDecl->inner))
        {
            ensureDecl(context, innerStructDecl);
            return LoweredValInfo();
        }
        else if (auto extensionDecl = as<ExtensionDecl>(genDecl->inner))
        {
            return ensureDecl(context, extensionDecl);
        }
        else if (auto interfaceDecl = as<InterfaceDecl>(genDecl->inner))
        {
            return ensureDecl(context, interfaceDecl);
        }
        else if (auto typedefDecl = as<TypeDefDecl>(genDecl->inner))
        {
            return ensureDecl(context, typedefDecl);
        }
        else if (auto subscriptDecl = as<SubscriptDecl>(genDecl->inner))
        {
            return ensureDecl(context, subscriptDecl);
        }
        SLANG_RELEASE_ASSERT(false);
        UNREACHABLE_RETURN(LoweredValInfo());
    }

    LoweredValInfo visitFunctionDeclBase(FunctionDeclBase* decl)
    {
        // A function declaration may have multiple, target-specific
        // overloads, and we need to emit an IR version of each of these.

        // The front end will form a linked list of declarations with
        // the same signature, whenever there is any kind of redeclaration.
        // We will look to see if that linked list has been formed.
        auto primaryDecl = decl->primaryDecl;

        if (!primaryDecl)
        {
            // If there is no linked list then we are in the ordinary
            // case with a single declaration, and no special handling
            // is needed.
            return lowerFuncDecl(decl);
        }

        // Otherwise, we need to walk the linked list of declarations
        // and make sure to emit IR code for any targets that need it.

        // TODO: Need to be careful about how this is approached,
        // to avoid emitting a bunch of extra definitions in the IR.

        auto primaryFuncDecl = as<FunctionDeclBase>(primaryDecl);
        SLANG_ASSERT(primaryFuncDecl);
        LoweredValInfo result = lowerFuncDecl(primaryFuncDecl);
        for (auto dd = primaryDecl->nextDecl; dd; dd = dd->nextDecl)
        {
            auto funcDecl = as<FunctionDeclBase>(dd);
            SLANG_ASSERT(funcDecl);
            lowerFuncDecl(funcDecl);

            // Note: Because we are iterating over multiple declarations,
            // but only one will be registered as the value for `decl`
            // in the global mapping by `ensureDecl()`, we have to take
            // responsibility here for registering a lowered value
            // for the remaining (non-primary) declarations.
            //
            // It doesn't really matter which one we register here, because
            // they will all have the same mangled name in the IR, but we
            // default to the `result` that is returned from this visitor,
            // so that all the declarations share the same IR representative.
            //
            context->setGlobalValue(funcDecl, result);
        }
        return result;
    }
};

LoweredValInfo lowerDecl(IRGenContext* context, DeclBase* decl)
{
    IRBuilderSourceLocRAII sourceLocInfo(context->irBuilder, decl->loc);

    DeclLoweringVisitor visitor;
    visitor.context = context;

    try
    {
        return visitor.dispatch(decl);
    }
    // Don't emit any context message for an explicit `AbortCompilationException`
    // because it should only happen when an error is already emitted.
    catch (const AbortCompilationException&)
    {
        throw;
    }
    catch (...)
    {
        context->getSink()->noteInternalErrorLoc(decl->loc);
        throw;
    }
}

// We will probably want to put the

LoweredValInfo* _findLoweredValInfo(IRGenContext* context, Decl* decl)
{
    // Look for an existing value installed in this context
    auto env = context->env;
    while (env)
    {
        if (auto result = env->mapDeclToValue.tryGetValue(decl))
            return result;

        env = env->outer;
    }
    return nullptr;
}

// Ensure that a version of the given declaration has been emitted to the IR
LoweredValInfo ensureDecl(IRGenContext* context, Decl* decl)
{
    if (auto valInfoPtr = _findLoweredValInfo(context, decl))
    {
        return *valInfoPtr;
    }

    // If we have a decl that's a generic value/type decl then something has gone seriously
    // wrong
    if (as<GenericValueParamDecl>(decl) || as<GenericTypeParamDecl>(decl))
    {
        SLANG_UNEXPECTED("Generic type/value shouldn't be handled here!");
    }

    IRBuilder subIRBuilder(context->irBuilder->getModule());
    if (as<VarDecl>(decl) && decl->findModifier<LocalTempVarModifier>())
    {
        // Do not modify insert location.
        subIRBuilder.setInsertLoc(context->irBuilder->getInsertLoc());
    }
    else
    {
        subIRBuilder.setInsertInto(subIRBuilder.getModule());
    }

    IRGenEnv subEnv;
    subEnv.outer = context->env;

    IRGenContext subContext = *context;
    subContext.irBuilder = &subIRBuilder;
    subContext.env = &subEnv;

    auto result = lowerDecl(&subContext, decl);

    // By default assume that any value we are lowering represents
    // something that should be installed globally.
    context->setGlobalValue(decl, result);

    return result;
}

// Can the IR lowered version of this declaration ever be an `IRGeneric`?
bool canDeclLowerToAGeneric(Decl* decl)
{
    // A callable decl lowers to an `IRFunc`, and can be generic
    if (as<CallableDecl>(decl))
        return true;

    // An aggregate type decl lowers to an `IRStruct`, and can be generic
    if (as<AggTypeDecl>(decl))
        return true;

    // An inheritance decl lowers to an `IRWitnessTable`, and can be generic
    if (as<InheritanceDecl>(decl))
        return true;

    // A `typedef` declaration nested under a generic will turn into
    // a generic that returns a type (a simple type-level function).
    if (as<TypeDefDecl>(decl))
        return true;

    // A static member variable declaration can be lowered into a generic.
    if (auto varDecl = as<VarDecl>(decl))
    {
        if (varDecl->hasModifier<HLSLStaticModifier>())
        {
            return !isFunctionVarDecl(varDecl);
        }
    }

    return false;
}

/// Add flattened "leaf" elements from `val` to the `ioArgs` list
static void _addFlattenedTupleArgs(List<IRInst*>& ioArgs, IRInst* val)
{
    if (auto tupleVal = as<IRMakeTuple>(val))
    {
        // If the value is a tuple, we can add its element directly.
        auto elementCount = tupleVal->getOperandCount();
        for (UInt i = 0; i < elementCount; ++i)
        {
            _addFlattenedTupleArgs(ioArgs, tupleVal->getOperand(i));
        }
    }
    //
    // TODO: We should handle the case here where `val`
    // is not a `makeTuple` instruction, but still has
    // a tuple *type*. In that case we should apply `getTupleElement`
    // for each of its elements and then recurse on them.
    //
    else
    {
        ioArgs.add(val);
    }
}

bool isAbstractWitnessTable(IRInst* inst)
{
    if (as<IRThisTypeWitness>(inst) || as<IRInterfaceRequirementEntry>(inst))
        return true;
    if (auto lookup = as<IRLookupWitnessMethod>(inst))
        return isAbstractWitnessTable(lookup->getWitnessTable());
    return false;
}

LoweredValInfo emitDeclRef(IRGenContext* context, Decl* decl, DeclRefBase* subst, IRType* type)
{
    const auto initialSubst = subst;
    SLANG_UNUSED(initialSubst);


    if (as<ThisTypeDecl>(decl))
    {
        // A declref to ThisType decl should be lowered differently
        // from other decls. In general, IFoo<T>.ThisType should lower to
        // ThisType(specialize(IFoo,T)) instead of specialize(IFoo.ThisType, T).
        SLANG_ASSERT(subst->getDecl() == decl);
        IRType* parentInterfaceType = nullptr;
        if (auto lookupDeclRef = as<LookupDeclRef>(subst))
        {
            parentInterfaceType = lowerType(context, lookupDeclRef->getWitness()->getSup());
        }
        else
        {
            parentInterfaceType =
                lowerType(context, DeclRefType::create(context->astBuilder, subst->getParent()));
        }
        auto thisType = context->irBuilder->getThisType(parentInterfaceType);
        return LoweredValInfo::simple(thisType);
    }

    // We need to proceed by considering the specializations that
    // have been put in place.
    subst = SubstitutionSet(subst).getInnerMostNodeWithSubstInfo();

    // If the declaration would not get wrapped in a `IRGeneric`,
    // even if it is nested inside of an AST `GenericDecl`, then
    // we should also ignore any generic substitutions.
    if (!canDeclLowerToAGeneric(decl))
    {
        while (auto genericSubst = SubstitutionSet(subst).findGenericAppDeclRef())
            subst = genericSubst->getBase();
    }

    // In the simplest case, there is no specialization going
    // on, and the decl-ref turns into a reference to the
    // lowered IR value for the declaration.
    if (!SubstitutionSet(subst) || _isTrivialLookupFromInterfaceThis(context, subst))
    {
        LoweredValInfo loweredDecl = ensureDecl(context, decl);
        return loweredDecl;
    }

    // Otherwise, we look at the kind of substitution, and let it guide us.
    if (auto genericSubst = as<GenericAppDeclRef>(subst))
    {
        // A generic substitution means we will need to output
        // a `specialize` instruction to specialize the generic.
        //
        // First we want to emit the value without generic specialization
        // applied, to get a correct value for it.
        //
        // Note: we only "unwrap" a single layer from the
        // substitutions here, because the underlying declaration
        // might be nested in multiple generics, or it might
        // come from an interface.
        //
        LoweredValInfo genericVal = emitDeclRef(
            context,
            decl,
            genericSubst->getBase(),
            context->irBuilder->getGenericKind());

        // There's no reason to specialize something that maps to a NULL pointer.
        if (genericVal.flavor == LoweredValInfo::Flavor::None)
            return LoweredValInfo();

        // We can only really specialize things that map to single values.
        // It would be an error if we got a non-`None` value that
        // wasn't somehow a single value.
        genericVal = materialize(context, genericVal);
        auto irGenericVal = genericVal.val;
        SLANG_ASSERT(irGenericVal);

        // We have the IR value for the generic we'd like to specialize,
        // and now we need to get the value for the arguments.
        List<IRInst*> irArgs;
        for (auto argVal : genericSubst->getArgs())
        {
            auto irArgVal = lowerSimpleVal(context, argVal);
            if (!irArgVal)
                continue;

            // It is possible that some of the arguments to the generic
            // represent conformances to conjunction types like `A & B`.
            // These conjunction conformances will appear as tuples in
            // the IR, and we want to "flatten" them here so that we
            // pass each "leaf" witness table as its own argument (to
            // match the way that generic parameters are being emitted
            // to the IR).
            //
            // TODO: This isn't a robust strategy if we ever have to deal
            // with tuples as ordinary values.
            //
            _addFlattenedTupleArgs(irArgs, irArgVal);
        }

        // Once we have both the generic and its arguments,
        // we can emit a `specialize` instruction and use
        // its value as the result.
        auto irSpecializedVal = context->irBuilder->emitSpecializeInst(
            type,
            irGenericVal,
            irArgs.getCount(),
            irArgs.getBuffer());
        switch (genericVal.flavor)
        {
        case LoweredValInfo::Flavor::Simple:
            return LoweredValInfo::simple(irSpecializedVal);
        case LoweredValInfo::Flavor::Ptr:
            return LoweredValInfo::ptr(irSpecializedVal);
        default:
            SLANG_UNEXPECTED("unhandled lowered value flavor");
            UNREACHABLE_RETURN(LoweredValInfo());
        }
    }
    else if (auto thisTypeSubst = as<LookupDeclRef>(subst))
    {
        if (as<ThisTypeDecl>(decl))
        {
            // This is a reference to the ThisType from the interface,
            // therefore we should just lower it as the sub type.
            return lowerType(context, thisTypeSubst->getWitness()->getSub());
        }

        if (isInterfaceRequirement(decl))
        {
            // If we reach here, somebody is trying to look up an interface
            // requirement "through" some concrete type. We need to lower this
            // decl-ref as a lookup of the corresponding member in a witness
            // table.
            //
            // The witness table itself is referenced by the this-type
            // substitution, so we can just lower that.
            //
            // Note: unlike the case for generics above, in the interface-lookup
            // case, we don't end up caring about any further outer substitutions.
            // That is because even if we are naming `ISomething<Foo>.doIt()`,
            // a method inside a generic interface, we don't actually care
            // about the substitution of `Foo` for the parameter `T` of
            // `ISomething<T>`. That is because we really care about the
            // witness table for the concrete type that conforms to `ISomething<Foo>`.
            //
            auto irWitnessTable = lowerSimpleVal(context, thisTypeSubst->getWitness());
            if (isAbstractWitnessTable(irWitnessTable))
            {
                // If `thisTypeSubst` doesn't lower into a concrete IRWitnessTable,
                // this is a lookup of an interface requirement
                // defined in some base interface from an interface type.
                // For now we just lower that decl as if it is referenced
                // from the same interface directly, e.g. a reference to
                // IBase.AssocType from IDerived:IBase will be lowered as
                // IRAssocType(IBase).
                // We may want to consider unifying our IR representation to
                // represent associated types with lookupWitness inst even inside
                // interface definitions.
                return emitDeclRef(
                    context,
                    decl->getDefaultDeclRef(),
                    context->irBuilder->getTypeKind());
            }

            SLANG_RELEASE_ASSERT(irWitnessTable);

            //
            // The key to use for looking up the interface member is
            // derived from the declaration.
            //
            auto irRequirementKey = getInterfaceRequirementKey(context, decl);
            //
            // Those two pieces of information tell us what we need to
            // do in order to look up the value that satisfied the requirement.
            //
            auto irSatisfyingVal = context->irBuilder->emitLookupInterfaceMethodInst(
                type,
                irWitnessTable,
                irRequirementKey);
            return LoweredValInfo::simple(irSatisfyingVal);
        }
        else
        {
            // This case is a reference to a member declaration of the interface
            // (or added by an extension of the interface) that does *not*
            // represent a requirement of the interface.
            //
            // Our policy is that concrete methods/members on an interface type
            // are lowered as generics, where the generic parameter represents
            // the `ThisType`.
            //
            auto genericVal = emitDeclRef(
                context,
                decl,
                thisTypeSubst->getBase(),
                context->irBuilder->getGenericKind());
            auto irGenericVal = getSimpleVal(context, genericVal);

            // In order to reference the member for a particular type, we
            // specialize the generic for that type.
            //
            IRInst* irSubType = lowerType(context, thisTypeSubst->getWitness()->getSub());
            IRInst* irSubTypeWitness = lowerSimpleVal(context, thisTypeSubst->getWitness());

            IRInst* irSpecializeArgs[] = {irSubType, irSubTypeWitness};
            auto irSpecializedVal =
                context->irBuilder->emitSpecializeInst(type, irGenericVal, 2, irSpecializeArgs);
            return LoweredValInfo::simple(irSpecializedVal);
        }
    }
    else
    {
        SLANG_UNEXPECTED("uhandled substitution type");
        UNREACHABLE_RETURN(LoweredValInfo());
    }
}

LoweredValInfo emitDeclRef(IRGenContext* context, DeclRef<Decl> declRef, IRType* type)
{
    return emitDeclRef(context, declRef.getDecl(), declRef.declRefBase, type);
}

static void lowerFrontEndEntryPointToIR(
    IRGenContext* context,
    EntryPoint* entryPoint,
    String moduleName)
{
    // TODO: We should emit an entry point as a dedicated IR function
    // (distinct from the IR function used if it were called normally),
    // with a mangled name based on the original function name plus
    // the stage for which it is being compiled as an entry point (so
    // that entry points for distinct stages always have distinct names).
    //
    // For now we just have an (implicit) constraint that a given
    // function should only be used as an entry point for one stage,
    // and any such function should *not* be used as an ordinary function.

    auto entryPointFuncDecl = entryPoint->getFuncDecl();

    if (!entryPointFuncDecl->findModifier<EntryPointAttribute>())
    {
        // If the entry point doesn't have an explicit `[shader("...")]` attribute,
        // then we make sure to add one here, so the lowering logic knows it is an
        // entry point.
        auto entryPointAttr = context->astBuilder->create<EntryPointAttribute>();
        entryPointAttr->capabilitySet = entryPoint->getProfile().getCapabilityName();
        addModifier(entryPointFuncDecl, entryPointAttr);
    }

    auto builder = context->irBuilder;
    builder->setInsertInto(builder->getModule()->getModuleInst());

    auto loweredEntryPointFunc = getSimpleVal(context, ensureDecl(context, entryPointFuncDecl));

    // Attach a marker decoration so that we recognize
    // this as an entry point.
    //
    IRInst* instToDecorate = loweredEntryPointFunc;
    if (auto irGeneric = as<IRGeneric>(instToDecorate))
    {
        instToDecorate = findGenericReturnVal(irGeneric);
    }

    // If the entry-point decorations has already been created (because the user
    // specified duplicate entries in the entry point list), we can stop now.
    if (instToDecorate->findDecoration<IREntryPointDecoration>())
        return;

    {

        Name* entryPointName = entryPoint->getFuncDecl()->getName();
        builder->addEntryPointDecoration(
            instToDecorate,
            entryPoint->getProfile(),
            entryPointName->text.getUnownedSlice(),
            moduleName.getUnownedSlice());
    }
}

static void lowerProgramEntryPointToIR(
    IRGenContext* context,
    EntryPoint* entryPoint,
    EntryPoint::EntryPointSpecializationInfo* specializationInfo)
{
    auto entryPointFuncDeclRef = entryPoint->getFuncDeclRef();
    if (specializationInfo)
        entryPointFuncDeclRef = specializationInfo->specializedFuncDeclRef;

    // First, lower the entry point like an ordinary function

    auto entryPointFuncType =
        lowerType(context, getFuncType(context->astBuilder, entryPointFuncDeclRef));

    auto builder = context->irBuilder;
    builder->setInsertInto(builder->getModule()->getModuleInst());

    auto loweredEntryPointFunc =
        getSimpleVal(context, emitDeclRef(context, entryPointFuncDeclRef, entryPointFuncType));

    if (!loweredEntryPointFunc->findDecoration<IRLinkageDecoration>())
    {
        builder->addExportDecoration(
            loweredEntryPointFunc,
            getMangledName(context->astBuilder, entryPointFuncDeclRef).getUnownedSlice());
    }

    // We may have shader parameters of interface/existential type,
    // which need us to supply concrete type information for specialization.
    //
    if (specializationInfo && specializationInfo->existentialSpecializationArgs.getCount() != 0)
    {
        List<IRInst*> existentialSlotArgs;
        for (auto arg : specializationInfo->existentialSpecializationArgs)
        {

            auto irArgType = lowerSimpleVal(context, arg.val);
            existentialSlotArgs.add(irArgType);

            if (auto witness = arg.witness)
            {
                auto irWitnessTable = lowerSimpleVal(context, witness);
                existentialSlotArgs.add(irWitnessTable);
            }
        }

        builder->addBindExistentialSlotsDecoration(
            loweredEntryPointFunc,
            existentialSlotArgs.getCount(),
            existentialSlotArgs.getBuffer());
    }
}

/// Ensure that `decl` and all relevant declarations under it get emitted.
static void ensureAllDeclsRec(IRGenContext* context, Decl* decl)
{
    ensureDecl(context, decl);

    // Note: We are checking here for aggregate type declarations, and
    // not for `ContainerDecl`s in general. This is because many kinds
    // of container declarations will already take responsibility for emitting
    // their children directly (e.g., a function declaration is responsible
    // for emitting its own parameters).
    //
    // Aggregate types are the main case where we can emit an outer declaration
    // and not the stuff nested inside of it.
    //
    if (auto containerDecl = as<AggTypeDeclBase>(decl))
    {
        for (auto memberDecl : containerDecl->members)
        {
            ensureAllDeclsRec(context, memberDecl);
        }
    }
    else if (auto namespaceDecl = as<NamespaceDecl>(decl))
    {
        for (auto memberDecl : namespaceDecl->members)
        {
            ensureAllDeclsRec(context, memberDecl);
        }
    }
    else if (auto fileDecl = as<FileDecl>(decl))
    {
        for (auto memberDecl : fileDecl->members)
        {
            ensureAllDeclsRec(context, memberDecl);
        }
    }
    else if (auto genericDecl = as<GenericDecl>(decl))
    {
        ensureAllDeclsRec(context, genericDecl->inner);
    }
}

RefPtr<IRModule> generateIRForTranslationUnit(
    ASTBuilder* astBuilder,
    TranslationUnitRequest* translationUnit)
{
    SLANG_PROFILE;
    SLANG_AST_BUILDER_RAII(astBuilder);

    auto session = translationUnit->getSession();
    auto compileRequest = translationUnit->compileRequest;
    Linkage* linkage = compileRequest->getLinkage();

    SharedIRGenContext sharedContextStorage(
        session,
        translationUnit->compileRequest->getSink(),
        translationUnit->compileRequest->optionSet.shouldObfuscateCode(),
        translationUnit->getModuleDecl(),
        translationUnit->compileRequest->getLinkage());
    SharedIRGenContext* sharedContext = &sharedContextStorage;

    IRGenContext contextStorage(sharedContext, astBuilder);
    IRGenContext* context = &contextStorage;

    RefPtr<IRModule> module = IRModule::create(session);

    module->setName(translationUnit->getModuleDecl()->getName());

    IRBuilder builderStorage(module);
    IRBuilder* builder = &builderStorage;

    context->irBuilder = builder;
    context->includeDebugInfo =
        compileRequest->getLinkage()->m_optionSet.getDebugInfoLevel() != DebugInfoLevel::None;

    // We need to emit IR for all public/exported symbols
    // in the translation unit.
    //
    // If debug info is enabled, we emit the DebugSource insts for each source file into IR.
    if (context->includeDebugInfo)
    {
        builder->setInsertInto(module->getModuleInst());
        for (auto source : translationUnit->getSourceFiles())
        {
            auto debugSource = builder->emitDebugSource(
                source->getPathInfo().getMostUniqueIdentity().getUnownedSlice(),
                source->getContent());
            context->shared->mapSourceFileToDebugSourceInst.add(source, debugSource);
        }
    }

    // For now, we will assume that *all* global-scope declarations
    // represent public/exported symbols.

    // First, ensure that all entry points have been emitted,
    // in case they require special handling.
    for (auto entryPoint : translationUnit->getEntryPoints())
    {
        List<SourceFile*> sources = translationUnit->getSourceFiles();
        SourceFile* source = sources.getFirst();
        PathInfo pInfo = source->getPathInfo();
        String path = pInfo.getMostUniqueIdentity();
        lowerFrontEndEntryPointToIR(context, entryPoint, Path::getFileNameWithoutExt(path));
    }

    //
    // Next, ensure that all other global declarations have
    // been emitted.
    for (auto decl : translationUnit->getModuleDecl()->members)
    {
        ensureAllDeclsRec(context, decl);
    }

    // Build a global instruction to hold all the string
    // literals used in the module.
    {
        auto& stringLits = sharedContext->m_stringLiterals;
        auto stringLitCount = stringLits.getCount();
        if (stringLitCount != 0)
        {
            builder->setInsertInto(module->getModuleInst());
            builder->emitIntrinsicInst(
                builder->getVoidType(),
                kIROp_GlobalHashedStringLiterals,
                stringLitCount,
                stringLits.getBuffer());
        }
    }

    if (auto nvapiSlotModifier =
            translationUnit->getModuleDecl()->findModifier<NVAPISlotModifier>())
    {
        builder->addNVAPISlotDecoration(
            module->getModuleInst(),
            nvapiSlotModifier->registerName.getUnownedSlice(),
            nvapiSlotModifier->spaceName.getUnownedSlice());
    }

#if 0
    {
        DiagnosticSinkWriter writer(compileRequest->getSink());
        dumpIR(module, &writer, "GENERATED");
    }
#endif

    validateIRModuleIfEnabled(compileRequest, module);


    // We will perform certain "mandatory" optimization passes now.
    // These passes serve two purposes:
    //
    // 1. To simplify the code that we use in backend compilation,
    // or when serializing/deserializing modules, so that we can
    // amortize this effort when we compile multiple entry points
    // that use the same module(s).
    //
    // 2. To ensure certain semantic properties that can't be
    // validated without dataflow information. For example, we want
    // to detect when a variable might be used before it is initialized.

    // Note: if you need to debug the IR that is created before
    // any mandatory optimizations have been applied, then
    // uncomment this line while debugging.

    //      dumpIR(module);

    // First, lower error handling logic into normal control flow.
    // This includes lowering throwing functions into functions that
    // returns a `Result<T,E>` value, translating `tryCall` into
    // normal `call` + `ifElse`, etc.
    lowerErrorHandling(module, compileRequest->getSink());

    // Lower `defer` so that later passes need not be aware of it.
    lowerDefer(module, compileRequest->getSink());

    // Synthesize some code we want to make sure is inlined and simplified
    synthesizeBitFieldAccessors(module);

    // Lower `IRExpandType` types to use `IRExpand`, where the pattern type
    // is nested inside the `IRExpand` as its children, instead of being same
    // level entities as the ExpandType itself.
    // This will unify the specialization logic for both type and value level
    // expansion.
    lowerExpandType(module);

    // Generate DebugValue insts to store values into debug variables,
    // if debug symbols are enabled.
    if (context->includeDebugInfo)
    {
        insertDebugValueStore(module);
    }


    // Next, attempt to promote local variables to SSA
    // temporaries and do basic simplifications.
    //
    constructSSA(module);
    applySparseConditionalConstantPropagation(module, compileRequest->getSink());

    bool minimumOptimizations =
        linkage->m_optionSet.getBoolOption(CompilerOptionName::MinimumSlangOptimization);
    if (!minimumOptimizations)
    {
        simplifyCFG(module, CFGSimplificationOptions::getDefault());
        auto peepholeOptions = PeepholeOptimizationOptions::getPrelinking();
        peepholeOptimize(nullptr, module, peepholeOptions);
    }

    IRDeadCodeEliminationOptions dceOptions = IRDeadCodeEliminationOptions();
    dceOptions.keepExportsAlive = true;
    dceOptions.keepLayoutsAlive = true;
    dceOptions.useFastAnalysis = true;

    for (auto inst : module->getGlobalInsts())
    {
        if (auto func = as<IRGlobalValueWithCode>(inst))
            eliminateDeadCode(func, dceOptions);
    }

    // Where possible, move loop condition checks to the end of loops, and wrap
    // the loop in an 'if(condition)'.
    // This makes it so that if sccp can see that the loop will always loop
    // at least once it can record this information by removing the outer
    // conditional.
    // This has advantages:
    // - Uninitialized variable usage detection doesn't have to
    //   worry about a loop never being executed.
    // - The loop condition is evaluated one fewer times.
    // - Allegedly better performance on pipelined processors:
    //   https://en.wikipedia.org/wiki/Loop_inversion
    //
    // And disadvantages
    // - If sccp is unable to eliminate the outer 'if' then we end up with
    //   duplicated code the the conditional value. Users don't tend to put
    //   huge gobs of code in the conditional expression in loops however.
    if (compileRequest->getLinkage()->m_optionSet.getBoolOption(CompilerOptionName::LoopInversion))
    {
        invertLoops(module);
    }

    // Next, inline calls to any functions that have been
    // marked for mandatory "early" inlining.
    //
    // Note: We performed certain critical simplifications
    // above, before this step, so that the body of functions
    // subject to mandatory inlining can be simplified ahead
    // of time. By simplifying the body before inlining it,
    // we can make sure that things like superfluous temporaries
    // are eliminated from the callee, and not copied into
    // call sites.
    //
    InstHashSet modifiedFuncs(module);
    for (;;)
    {
        bool changed = false;
        modifiedFuncs.clear();
        changed = performMandatoryEarlyInlining(module, &modifiedFuncs.getHashSet());
        if (changed)
        {
            changed = peepholeOptimizeGlobalScope(nullptr, module);
            if (!minimumOptimizations)
            {
                for (auto func : modifiedFuncs.getHashSet())
                {
                    auto codeInst = as<IRGlobalValueWithCode>(func);
                    changed |= constructSSA(func);
                    changed |=
                        applySparseConditionalConstantPropagation(func, compileRequest->getSink());
                    changed |= peepholeOptimize(nullptr, func);
                    changed |= simplifyCFG(codeInst, CFGSimplificationOptions::getFast());
                    eliminateDeadCode(func, dceOptions);
                }
            }
        }
        if (!changed)
            break;
    }

    if (compileRequest->getLinkage()->m_optionSet.shouldRunNonEssentialValidation())
    {
        // We don't allow recursive types.
        checkForRecursiveTypes(module, compileRequest->getSink());

        if (compileRequest->getSink()->getErrorCount() != 0)
            return module;

        // Propagate `constexpr`-ness through the dataflow graph (and the
        // call graph) based on constraints imposed by different instructions.
        propagateConstExpr(module, compileRequest->getSink());

        // Check for using uninitialized values
        checkForUsingUninitializedValues(module, compileRequest->getSink());

        // TODO: give error messages if any `undefined` or
        // instructions remain.

        checkForMissingReturns(module, compileRequest->getSink(), CodeGenTarget::None, true);
        // Check for invalid differentiable function body.
        checkAutoDiffUsages(module, compileRequest->getSink());

        checkForOperatorShiftOverflow(module, linkage->m_optionSet, compileRequest->getSink());
    }

    // The "mandatory" optimization passes may make use of the
    // `IRHighLevelDeclDecoration` type to relate IR instructions
    // back to AST-level code in order to improve the quality
    // of diagnostics that are emitted.
    //
    // While it is important for these passes to have access
    // to AST-level information, allowing that information to
    // flow into later steps (e.g., code generation) could lead
    // to unclean layering of the parts of the compiler.
    // In principle, back-end steps should not need to know where
    // IR code came from.
    //
    // In order to avoid problems, we run a pass here to strip
    // out any decorations that should not be relied upon by
    // later passes.
    //
    {
        // Because we are already stripping out undesired decorations,
        // this is also a convenient place to remove any `IRNameHintDecoration`s
        // in the case where we are obfuscating code. We handle this
        // by setting up the options for the stripping pass appropriately.
        //
        IRStripOptions stripOptions;

        stripOptions.shouldStripNameHints = linkage->m_optionSet.shouldObfuscateCode();

        // If we are generating an obfuscated source map, we don't want to strip locs,
        // we want to generate *new* locs that can be mapped (via source map)
        // back to *actual* source.
        //
        // We don't do the obfuscation remapping here, because DCE and other passes may
        // change what locs are actually needed, we need to be sure
        // that if we have obfuscation enabled we don't forget to obfuscate.
        stripOptions.stripSourceLocs = false;
        stripFrontEndOnlyInstructions(module, stripOptions);

        stripImportedWitnessTable(module);

        // Stripping out decorations could leave some dead code behind
        // in the module, and in some cases that extra code is also
        // undesirable (e.g., the string literals referenced by name-hint
        // decorations are just as undesirable as the decorations themselves).
        // To clean up after ourselves we also run a dead-code elimination
        // pass here, but make sure to set our options so that we don't
        // eliminate anything that has been marked for export.
        //
        eliminateDeadCode(module, dceOptions);

        if (stripOptions.shouldStripNameHints && linkage->m_optionSet.shouldHaveSourceMap())
        {
            // The obfuscated source map is stored on the module
            obfuscateModuleLocs(module, compileRequest->getSourceManager());
        }
    }


    // TODO: consider doing some more aggressive optimizations
    // (in particular specialization of generics) here, so
    // that we can avoid doing them downstream.
    //
    // Note: doing specialization or inlining involving code
    // from other modules potentially makes the IR we generate
    // "fragile" in that we'd now need to recompile when
    // a module we depend on changes.

    validateIRModuleIfEnabled(compileRequest, module);

    // If we are being asked to dump IR during compilation,
    // then we can dump the initial IR for the module here.
    if (compileRequest->optionSet.shouldDumpIR())
    {
        DiagnosticSinkWriter writer(compileRequest->getSink());

        dumpIR(
            module,
            compileRequest->m_irDumpOptions,
            "LOWER-TO-IR",
            compileRequest->getSourceManager(),
            &writer);
    }

    module->buildMangledNameToGlobalInstMap();

    return module;
}

/// Context for generating IR code to represent a `SpecializedComponentType`
struct SpecializedComponentTypeIRGenContext : ComponentTypeVisitor
{
    DiagnosticSink* sink;
    Linkage* linkage;
    Session* session;
    IRGenContext* context;
    IRBuilder* builder;

    RefPtr<IRModule> process(SpecializedComponentType* componentType, DiagnosticSink* inSink)
    {
        sink = inSink;

        linkage = componentType->getLinkage();
        session = linkage->getSessionImpl();
        auto option = linkage->m_optionSet;
        option.overrideWith(componentType->getOptionSet());
        SharedIRGenContext
            sharedContextStorage(session, sink, option.shouldObfuscateCode(), nullptr, linkage);
        SharedIRGenContext* sharedContext = &sharedContextStorage;

        IRGenContext contextStorage(sharedContext, linkage->getASTBuilder());
        context = &contextStorage;

        RefPtr<IRModule> module = IRModule::create(session);

        IRBuilder builderStorage(module);
        builder = &builderStorage;

        builder->setInsertInto(module);

        context->irBuilder = builder;

        componentType->acceptVisitor(this, nullptr);
        module->buildMangledNameToGlobalInstMap();
        return module;
    }

    void visitEntryPoint(
        EntryPoint* entryPoint,
        EntryPoint::EntryPointSpecializationInfo* specializationInfo) SLANG_OVERRIDE
    {
        // We need to emit symbols for all of the entry
        // points in the program; this is especially
        // important in the case where a generic entry
        // point is being specialized.
        //
        lowerProgramEntryPointToIR(context, entryPoint, specializationInfo);
    }

    void visitRenamedEntryPoint(
        RenamedEntryPointComponentType* entryPoint,
        EntryPoint::EntryPointSpecializationInfo* specializationInfo) SLANG_OVERRIDE
    {
        entryPoint->getBase()->acceptVisitor(this, specializationInfo);
    }

    void visitModule(Module* module, Module::ModuleSpecializationInfo* specializationInfo)
        SLANG_OVERRIDE
    {
        // We've hit a leaf module, so we should be able to bind any global
        // generic type parameters here...
        //
        if (specializationInfo)
        {
            for (auto genericArgInfo : specializationInfo->genericArgs)
            {
                IRInst* irParam =
                    getSimpleVal(context, ensureDecl(context, genericArgInfo.paramDecl));
                IRInst* irVal = lowerSimpleVal(context, genericArgInfo.argVal);

                // bind `irParam` to `irVal`
                builder->emitBindGlobalGenericParam(irParam, irVal);
            }

            auto shaderParamCount = module->getShaderParamCount();
            Index existentialArgOffset = 0;

            for (Index ii = 0; ii < shaderParamCount; ++ii)
            {
                auto shaderParam = module->getShaderParam(ii);
                auto specializationArgCount = shaderParam.specializationParamCount;

                IRInst* irParam =
                    getSimpleVal(context, ensureDecl(context, shaderParam.paramDeclRef.getDecl()));
                List<IRInst*> irSlotArgs;
                // Tracks if there are any type args that is not an IRDynamicType.
                bool hasConcreteTypeArg = false;
                for (Index jj = 0; jj < specializationArgCount; ++jj)
                {
                    auto& specializationArg =
                        specializationInfo->existentialArgs[existentialArgOffset++];

                    auto irType = lowerSimpleVal(context, specializationArg.val);
                    auto irWitness = lowerSimpleVal(context, specializationArg.witness);

                    if (irType->getOp() != kIROp_DynamicType)
                        hasConcreteTypeArg = true;

                    irSlotArgs.add(irType);
                    irSlotArgs.add(irWitness);
                }
                // Only insert a `BindExistentialSlots` decoration when there are at least
                // one non-dynamic type argument.
                if (hasConcreteTypeArg)
                {
                    builder->addBindExistentialSlotsDecoration(
                        irParam,
                        irSlotArgs.getCount(),
                        irSlotArgs.getBuffer());
                }
            }
        }
    }

    void visitComposite(
        CompositeComponentType* composite,
        CompositeComponentType::CompositeSpecializationInfo* specializationInfo) SLANG_OVERRIDE
    {
        visitChildren(composite, specializationInfo);
    }

    void visitSpecialized(SpecializedComponentType* specialized) SLANG_OVERRIDE
    {
        visitChildren(specialized);
    }

    void visitTypeConformance(TypeConformance* conformance) SLANG_OVERRIDE
    {
        SLANG_UNUSED(conformance);
    }
};

RefPtr<IRModule> generateIRForSpecializedComponentType(
    SpecializedComponentType* componentType,
    DiagnosticSink* sink)
{
    SLANG_AST_BUILDER_RAII(componentType->getLinkage()->getASTBuilder());

    SpecializedComponentTypeIRGenContext context;
    return context.process(componentType, sink);
}

/// Context for generating IR code to represent a `TypeConformance`
struct TypeConformanceIRGenContext
{
    DiagnosticSink* sink;
    Linkage* linkage;
    Session* session;
    IRGenContext* context;
    IRBuilder* builder;

    RefPtr<IRModule> process(
        TypeConformance* typeConformance,
        Int conformanceIdOverride,
        DiagnosticSink* inSink)
    {
        sink = inSink;

        linkage = typeConformance->getLinkage();
        session = linkage->getSessionImpl();

        SharedIRGenContext sharedContextStorage(
            session,
            sink,
            linkage->m_optionSet.shouldObfuscateCode(),
            nullptr,
            linkage);
        SharedIRGenContext* sharedContext = &sharedContextStorage;

        IRGenContext contextStorage(sharedContext, linkage->getASTBuilder());
        context = &contextStorage;

        RefPtr<IRModule> module = IRModule::create(session);

        IRBuilder builderStorage(module);
        builder = &builderStorage;

        builder->setInsertInto(module);

        context->irBuilder = builder;

        auto witness = lowerSimpleVal(context, typeConformance->getSubtypeWitness());
        builder->addKeepAliveDecoration(witness);
        builder->addHLSLExportDecoration(witness);
        builder->addDynamicDispatchWitnessDecoration(witness);
        if (conformanceIdOverride != -1)
        {
            builder->addSequentialIDDecoration(witness, conformanceIdOverride);
        }
        module->buildMangledNameToGlobalInstMap();
        return module;
    }
};

RefPtr<IRModule> generateIRForTypeConformance(
    TypeConformance* typeConformance,
    Int conformanceIdOverride,
    DiagnosticSink* sink)
{
    SLANG_AST_BUILDER_RAII(typeConformance->getLinkage()->getASTBuilder());

    TypeConformanceIRGenContext context;
    return context.process(typeConformance, conformanceIdOverride, sink);
}

RefPtr<IRModule> TargetProgram::getOrCreateIRModuleForLayout(DiagnosticSink* sink)
{
    getOrCreateLayout(sink);
    return m_irModuleForLayout;
}

/// Specialized IR generation context for when generating IR for layouts.
struct IRLayoutGenContext : IRGenContext
{
    IRLayoutGenContext(SharedIRGenContext* shared, ASTBuilder* astBuilder)
        : IRGenContext(shared, astBuilder)
    {
    }

    /// Cache for custom key instructions used for entry-point parameter layout information.
    Dictionary<ParamDecl*, IRInst*> mapEntryPointParamToKey;
};

/// Lower an AST-level type layout to an IR-level type layout.
IRTypeLayout* lowerTypeLayout(IRLayoutGenContext* context, TypeLayout* typeLayout);

/// Lower an AST-level variable layout to an IR-level variable layout.
IRVarLayout* lowerVarLayout(IRLayoutGenContext* context, VarLayout* varLayout);

/// Shared code for most `lowerTypeLayout` cases.
///
/// Handles copying of resource usage and pending data type layout
/// from the AST `typeLayout` to the specified `builder`.
///
static IRTypeLayout* _lowerTypeLayoutCommon(
    IRLayoutGenContext* context,
    IRTypeLayout::Builder* builder,
    TypeLayout* typeLayout)
{
    for (auto resInfo : typeLayout->resourceInfos)
    {
        builder->addResourceUsage(resInfo.kind, resInfo.count);
    }

    if (auto pendingTypeLayout = typeLayout->pendingDataTypeLayout)
    {
        builder->setPendingTypeLayout(lowerTypeLayout(context, pendingTypeLayout));
    }

    return builder->build();
}

IRTypeLayout* lowerTypeLayout(IRLayoutGenContext* context, TypeLayout* typeLayout)
{
    // TODO: We chould consider caching the layouts we create based on `typeLayout`
    // and re-using them. This isn't strictly necessary because we emit the
    // instructions as "hoistable" which should give us de-duplication, and it wouldn't
    // help much until/unless the AST level gets less wasteful about how it computes layout.

    // We will use casting to detect if `typeLayout` is
    // one of the cases that requires a dedicated sub-type
    // of IR type layout.
    //
    if (auto paramGroupTypeLayout = as<ParameterGroupTypeLayout>(typeLayout))
    {
        IRParameterGroupTypeLayout::Builder builder(context->irBuilder);

        builder.setContainerVarLayout(
            lowerVarLayout(context, paramGroupTypeLayout->containerVarLayout));
        builder.setElementVarLayout(
            lowerVarLayout(context, paramGroupTypeLayout->elementVarLayout));
        builder.setOffsetElementTypeLayout(
            lowerTypeLayout(context, paramGroupTypeLayout->offsetElementTypeLayout));

        return _lowerTypeLayoutCommon(context, &builder, paramGroupTypeLayout);
    }
    else if (auto structuredBufferTypeLayout = as<StructuredBufferTypeLayout>(typeLayout))
    {
        auto irElementTypeLayout =
            lowerTypeLayout(context, structuredBufferTypeLayout->elementTypeLayout);
        IRStructuredBufferTypeLayout::Builder builder(context->irBuilder, irElementTypeLayout);
        return _lowerTypeLayoutCommon(context, &builder, structuredBufferTypeLayout);
    }
    else if (auto structTypeLayout = as<StructTypeLayout>(typeLayout))
    {
        IRStructTypeLayout::Builder builder(context->irBuilder);
        int fieldIndex = 0;
        for (auto fieldLayout : structTypeLayout->fields)
        {
            auto fieldDecl = fieldLayout->varDecl;

            IRInst* irFieldKey = nullptr;
            if (auto paramDecl = as<ParamDecl>(fieldDecl))
            {
                // There is a subtle special case here.
                //
                // A `StructTypeLayout` might be used to represent
                // the parameters of an entry point, and this is the
                // one and only case where the "fields" being used
                // might actually be `ParamDecl`s.
                //
                // The IR encoding of structure type layouts relies
                // on using field "key" instructions to identify
                // the fields, but these don't exist (by default)
                // for function parameters.
                //
                // To get around this problem we will create key
                // instructions to stand in for the entry-point parameters
                // as needed when generating layout.
                //
                // We need to cache the generated keys on the context,
                // so that if we run into another type layout for the
                // same entry point we will re-use the same keys.
                //
                if (!context->mapEntryPointParamToKey.tryGetValue(paramDecl.getDecl(), irFieldKey))
                {
                    irFieldKey = context->irBuilder->createStructKey();

                    // TODO: It might eventually be a good idea to attach a mangled
                    // name to the key we just generated (derived from the entry point
                    // and parameter name), even though parameters don't usually have
                    // linkage.
                    //
                    // Doing so would ensure that if we ever combined partial layout
                    // information from different modules they would agree on the key
                    // to use for entry-point parameters.
                    //
                    // For now this is a non-issue because both the creation and use
                    // of these keys will be local to a single `IREntryPointLayout`,
                    // and we don't support combination at a finer granularity than that.

                    context->mapEntryPointParamToKey.add(paramDecl.getDecl(), irFieldKey);
                }
            }
            else if (fieldDecl.getDecl())
            {
                irFieldKey = getSimpleVal(context, ensureDecl(context, fieldDecl.getDecl()));
            }
            else
            {
                // If we don't have a concrete field decl for the field in the layout,
                // it could be that the field in the layout is for a member of a tuple
                // type that hasn't been materialized into a struct decl yet.
                // We will use a `IndexFieldKey(type, memberIndex)` inst as a placeholder
                // for the field key.
                // This placeholder can be replaced with the actual field key when the
                // tuple type is materialized into a struct type.
                auto irType = lowerType(context, typeLayout->getType());
                irFieldKey = context->irBuilder->getIndexedFieldKey(irType, fieldIndex);
            }
            fieldIndex++;
            SLANG_ASSERT(irFieldKey);

            auto irFieldLayout = lowerVarLayout(context, fieldLayout);
            builder.addField(irFieldKey, irFieldLayout);
        }

        return _lowerTypeLayoutCommon(context, &builder, structTypeLayout);
    }
    else if (auto arrayTypeLayout = as<ArrayTypeLayout>(typeLayout))
    {
        auto irElementTypeLayout = lowerTypeLayout(context, arrayTypeLayout->elementTypeLayout);
        IRArrayTypeLayout::Builder builder(context->irBuilder, irElementTypeLayout);
        return _lowerTypeLayoutCommon(context, &builder, arrayTypeLayout);
    }
    else if (auto ptrTypeLayout = as<PointerTypeLayout>(typeLayout))
    {
        // TODO(JS):
        // For now we don't lower the value/target type because this could lead to inifinte
        // recursion in the way this is currently implemented.

        // auto irValueTypeLayout = lowerTypeLayout(context, ptrTypeLayout->valueTypeLayout);
        IRPointerTypeLayout::Builder builder(context->irBuilder);
        return _lowerTypeLayoutCommon(context, &builder, ptrTypeLayout);
    }
    else if (auto streamOutputTypeLayout = as<StreamOutputTypeLayout>(typeLayout))
    {
        auto irElementTypeLayout =
            lowerTypeLayout(context, streamOutputTypeLayout->elementTypeLayout);

        IRStreamOutputTypeLayout::Builder builder(context->irBuilder, irElementTypeLayout);
        return _lowerTypeLayoutCommon(context, &builder, streamOutputTypeLayout);
    }
    else if (auto matrixTypeLayout = as<MatrixTypeLayout>(typeLayout))
    {
        // TODO: Our support for explicit layouts on matrix types is minimal, so whether
        // or not we even include `IRMatrixTypeLayout` doesn't impact any behavior we
        // currently test.
        //
        // Our handling of matrix types and their layout needs a complete overhaul, but
        // that isn't something we can get to right away, so we'll just try to pass
        // along this data as best we can for now.

        IRMatrixTypeLayout::Builder builder(context->irBuilder, matrixTypeLayout->mode);
        return _lowerTypeLayoutCommon(context, &builder, matrixTypeLayout);
    }
    else if (auto existentialTypeLayout = as<ExistentialTypeLayout>(typeLayout))
    {
        IRExistentialTypeLayout::Builder builder(context->irBuilder);
        return _lowerTypeLayoutCommon(context, &builder, existentialTypeLayout);
    }
    else
    {
        // If no special case applies we will build a generic `IRTypeLayout`.
        //
        IRTypeLayout::Builder builder(context->irBuilder);
        return _lowerTypeLayoutCommon(context, &builder, typeLayout);
    }
}

IRVarLayout* lowerVarLayout(
    IRLayoutGenContext* context,
    VarLayout* varLayout,
    IRTypeLayout* irTypeLayout)
{
    IRVarLayout::Builder irLayoutBuilder(context->irBuilder, irTypeLayout);

    for (auto resInfo : varLayout->resourceInfos)
    {
        auto irResInfo = irLayoutBuilder.findOrAddResourceInfo(resInfo.kind);
        irResInfo->offset = resInfo.index;
        irResInfo->space = resInfo.space;
    }

    if (auto pendingVarLayout = varLayout->pendingVarLayout)
    {
        irLayoutBuilder.setPendingVarLayout(lowerVarLayout(context, pendingVarLayout));
    }

    // We will only generate layout information with *either* a system-value
    // semantic or a user-defined semantic, and we will always check for
    // the system-value semantic first because the AST-level representation
    // seems to encode both when a system-value semantic is present.
    //
    if (varLayout->systemValueSemantic.getLength())
    {
        irLayoutBuilder.setSystemValueSemantic(
            varLayout->systemValueSemantic,
            varLayout->systemValueSemanticIndex);
    }
    else if (varLayout->semanticName.getLength())
    {
        irLayoutBuilder.setUserSemantic(varLayout->semanticName, varLayout->semanticIndex);
    }

    if (varLayout->stage != Stage::Unknown)
    {
        irLayoutBuilder.setStage(varLayout->stage);
    }

    return irLayoutBuilder.build();
}

IRVarLayout* lowerVarLayout(IRLayoutGenContext* context, VarLayout* varLayout)
{
    auto irTypeLayout = lowerTypeLayout(context, varLayout->typeLayout);
    return lowerVarLayout(context, varLayout, irTypeLayout);
}

/// Handle the lowering of an entry-point result layout to the IR
IRVarLayout* lowerEntryPointResultLayout(IRLayoutGenContext* context, VarLayout* layout)
{
    // The easy case is when there is a non-null `layout`, because we
    // can handle it like any other var layout.
    //
    if (layout)
        return lowerVarLayout(context, layout);

    // Right now the AST-level layout logic will leave a null layout
    // for the result when an entry point has a `void` result type.
    //
    // TODO: We should fix this at the AST level instead of the IR,
    // but doing so would impact reflection, where clients could
    // be using a null check to test for a `void` result.
    //
    // As a workaround, we will create an empty type layout and
    // an empty var layout that represents it, consistent with the
    // way that a `void` value consumes no resources.
    //
    IRTypeLayout::Builder typeLayoutBuilder(context->irBuilder);
    auto irTypeLayout = typeLayoutBuilder.build();
    IRVarLayout::Builder varLayoutBuilder(context->irBuilder, irTypeLayout);
    return varLayoutBuilder.build();
}

/// Lower AST-level layout information for an entry point to the IR
IREntryPointLayout* lowerEntryPointLayout(
    IRLayoutGenContext* context,
    EntryPointLayout* entryPointLayout)
{
    auto irParamsLayout = lowerVarLayout(context, entryPointLayout->parametersLayout);
    auto irResultLayout = lowerEntryPointResultLayout(context, entryPointLayout->resultLayout);

    return context->irBuilder->getEntryPointLayout(irParamsLayout, irResultLayout);
}

RefPtr<IRModule> TargetProgram::createIRModuleForLayout(DiagnosticSink* sink)
{
    if (m_irModuleForLayout)
        return m_irModuleForLayout;


    // Okay, now we need to fill it in.

    auto programLayout = getOrCreateLayout(sink);
    if (!programLayout)
        return nullptr;

    auto program = getProgram();
    auto linkage = program->getLinkage();

    SLANG_AST_BUILDER_RAII(linkage->getASTBuilder());

    auto session = linkage->getSessionImpl();

    SharedIRGenContext sharedContextStorage(
        session,
        sink,
        linkage->m_optionSet.shouldObfuscateCode(),
        nullptr,
        linkage);
    auto sharedContext = &sharedContextStorage;

    ASTBuilder* astBuilder = linkage->getASTBuilder();

    IRLayoutGenContext contextStorage(sharedContext, astBuilder);
    auto context = &contextStorage;

    RefPtr<IRModule> irModule = IRModule::create(session);

    IRBuilder builderStorage(irModule);
    auto builder = &builderStorage;

    builder->setInsertInto(irModule);

    context->irBuilder = builder;


    // Okay, now we need to walk through and decorate everything.
    auto globalStructLayout = getScopeStructLayout(programLayout);

    IRStructTypeLayout::Builder globalStructTypeLayoutBuilder(builder);

    for (auto varLayout : globalStructLayout->fields)
    {
        auto varDecl = varLayout->varDecl;

        // Ensure that an `[import(...)]` declaration for the variable
        // has been emitted to this module, so that we will have something
        // to decorate.
        //
        auto irVar = materialize(context, ensureDecl(context, varDecl.getDecl())).val;
        if (!irVar)
            SLANG_UNEXPECTED("unhandled value flavor");

        auto irLayout = lowerVarLayout(context, varLayout);

        // Now attach the decoration to the variable.
        //
        builder->addLayoutDecoration(irVar, irLayout);

        // Also add this to our mapping for the global-scope structure type
        globalStructTypeLayoutBuilder.addField(irVar, irLayout);
    }
    auto irGlobalStructTypeLayout =
        _lowerTypeLayoutCommon(context, &globalStructTypeLayoutBuilder, globalStructLayout);

    auto globalScopeVarLayout = programLayout->parametersLayout;
    auto globalScopeTypeLayout = globalScopeVarLayout->typeLayout;
    IRTypeLayout* irGlobalScopeTypeLayout = irGlobalStructTypeLayout;
    if (auto paramGroupTypeLayout = as<ParameterGroupTypeLayout>(globalScopeTypeLayout))
    {
        IRParameterGroupTypeLayout::Builder globalParameterGroupTypeLayoutBuilder(builder);

        auto irElementTypeLayout = irGlobalStructTypeLayout;
        auto irElementVarLayout =
            lowerVarLayout(context, paramGroupTypeLayout->elementVarLayout, irElementTypeLayout);

        globalParameterGroupTypeLayoutBuilder.setContainerVarLayout(
            lowerVarLayout(context, paramGroupTypeLayout->containerVarLayout));
        globalParameterGroupTypeLayoutBuilder.setElementVarLayout(irElementVarLayout);
        globalParameterGroupTypeLayoutBuilder.setOffsetElementTypeLayout(
            lowerTypeLayout(context, paramGroupTypeLayout->offsetElementTypeLayout));

        auto irParamGroupTypeLayout = _lowerTypeLayoutCommon(
            context,
            &globalParameterGroupTypeLayoutBuilder,
            paramGroupTypeLayout);

        irGlobalScopeTypeLayout = irParamGroupTypeLayout;
    }

    auto irGlobalScopeVarLayout =
        lowerVarLayout(context, globalScopeVarLayout, irGlobalScopeTypeLayout);

    builder->addLayoutDecoration(irModule->getModuleInst(), irGlobalScopeVarLayout);

    auto latestSpirvAtom = getLatestSpirvAtom();
    auto latestMetalAtom = getLatestMetalAtom();

    for (auto entryPointLayout : programLayout->entryPoints)
    {
        auto funcDeclRef = entryPointLayout->entryPoint;

        // HACK: skip over entry points that came from deserialization,
        // and thus don't have AST-level information for us to work with.
        //
        if (!funcDeclRef)
            continue;

        auto irFuncType = lowerType(context, getFuncType(astBuilder, funcDeclRef));
        auto irFunc = getSimpleVal(context, emitDeclRef(context, funcDeclRef, irFuncType));

        if (!irFunc->findDecoration<IRLinkageDecoration>())
        {
            builder->addImportDecoration(
                irFunc,
                getMangledName(astBuilder, funcDeclRef).getUnownedSlice());
        }

        for (auto atomSet :
             as<FuncDecl>(funcDeclRef.getDecl())->inferredCapabilityRequirements.getAtomSets())
        {
            for (auto atomVal : atomSet)
            {
                auto atom = asAtom(atomVal);
                if (atom >= CapabilityAtom::_spirv_1_0 && atom <= latestSpirvAtom ||
                    atom >= CapabilityAtom::metallib_2_3 && atom <= latestMetalAtom)
                {
                    builder->addRequireCapabilityAtomDecoration(irFunc, (CapabilityName)atom);
                }
            }
        }

        auto irEntryPointLayout = lowerEntryPointLayout(context, entryPointLayout);

        builder->addLayoutDecoration(irFunc, irEntryPointLayout);
    }

    // Lets strip and run DCE here
    if (linkage->m_optionSet.shouldObfuscateCode())
    {
        IRStripOptions stripOptions;

        stripOptions.shouldStripNameHints = true;
        stripOptions.stripSourceLocs = true;
        ;

        stripFrontEndOnlyInstructions(irModule, stripOptions);

        IRDeadCodeEliminationOptions options;
        options.keepExportsAlive = true;
        options.keepLayoutsAlive = true;

        // Eliminate any dead code
        eliminateDeadCode(irModule, options);
    }
    irModule->buildMangledNameToGlobalInstMap();
    m_irModuleForLayout = irModule;
    return irModule;
}


} // namespace Slang
