// slang-ir-legalize-types.cpp

// This file implements type legalization for the IR.
// It uses the core legalization logic in
// `legalize-types.{h,cpp}` to decide what to do with
// the types, while this file handles the actual
// rewriting of the IR to use the new types.
//
// This pass should only be applied to IR that has been
// fully specialized (no more generics/interfaces), so
// that the concrete type of everything is known.

#include "../compiler-core/slang-name.h"
#include "../core/slang-performance-profiler.h"
#include "slang-ir-clone.h"
#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-ir.h"
#include "slang-legalize-types.h"
#include "slang-mangle.h"

namespace Slang
{

LegalVal LegalVal::tuple(RefPtr<TuplePseudoVal> tupleVal)
{
    SLANG_ASSERT(tupleVal->elements.getCount());

    LegalVal result;
    result.flavor = LegalVal::Flavor::tuple;
    result.obj = tupleVal;
    return result;
}

LegalVal LegalVal::pair(RefPtr<PairPseudoVal> pairInfo)
{
    LegalVal result;
    result.flavor = LegalVal::Flavor::pair;
    result.obj = pairInfo;
    return result;
}

LegalVal LegalVal::pair(
    LegalVal const& ordinaryVal,
    LegalVal const& specialVal,
    RefPtr<PairInfo> pairInfo)
{
    if (ordinaryVal.flavor == LegalVal::Flavor::none)
        return specialVal;

    if (specialVal.flavor == LegalVal::Flavor::none)
        return ordinaryVal;


    RefPtr<PairPseudoVal> obj = new PairPseudoVal();
    obj->ordinaryVal = ordinaryVal;
    obj->specialVal = specialVal;
    obj->pairInfo = pairInfo;

    return LegalVal::pair(obj);
}

LegalVal LegalVal::implicitDeref(LegalVal const& val)
{
    RefPtr<ImplicitDerefVal> implicitDerefVal = new ImplicitDerefVal();
    implicitDerefVal->val = val;

    LegalVal result;
    result.flavor = LegalVal::Flavor::implicitDeref;
    result.obj = implicitDerefVal;
    return result;
}

LegalVal LegalVal::getImplicitDeref() const
{
    SLANG_ASSERT(flavor == Flavor::implicitDeref);
    return as<ImplicitDerefVal>(obj)->val;
}

LegalVal LegalVal::wrappedBuffer(LegalVal const& baseVal, LegalElementWrapping const& elementInfo)
{
    RefPtr<WrappedBufferPseudoVal> obj = new WrappedBufferPseudoVal();
    obj->base = baseVal;
    obj->elementInfo = elementInfo;

    LegalVal result;
    result.flavor = LegalVal::Flavor::wrappedBuffer;
    result.obj = obj;
    return result;
}

//

IRTypeLegalizationContext::IRTypeLegalizationContext(
    TargetProgram* target,
    IRModule* inModule,
    DiagnosticSink* sink)
{
    targetProgram = target;

    session = inModule->getSession();
    module = inModule;

    builderStorage = IRBuilder(inModule);
    builder = &builderStorage;

    m_sink = sink;
}

static void registerLegalizedValue(
    IRTypeLegalizationContext* context,
    IRInst* irValue,
    LegalVal const& legalVal)
{
    context->mapValToLegalVal[irValue] = legalVal;
}

/// Structure to pass information from the original/old global param to
/// composite members during tuple flavored global param legalization.
struct IRGlobalParamInfo
{
    IRFunc* originatingEntryPoint = nullptr;
};

static LegalVal declareVars(
    IRTypeLegalizationContext* context,
    IROp op,
    LegalType type,
    IRTypeLayout* typeLayout,
    LegalVarChain const& varChain,
    UnownedStringSlice nameHint,
    IRInst* leafVar,
    IRGlobalParamInfo* globalParamInfo,
    bool isSpecial);

/// Unwrap a value with flavor `wrappedBuffer`
///
/// The original `legalPtrOperand` has a wrapped-buffer type
/// which encodes the way that, e.g., a `ConstantBuffer<Foo>`
/// where `Foo` includes interface types, got legalized
/// into a buffer that stores a `Foo` value plus addition
/// fields for the concrete types that got plugged in.
///
/// The `elementInfo` is the layout information for the
/// modified ("wrapped") buffer type, and specifies how
/// the logical element type was expanded into actual fields.
///
/// This function returns a new value that undoes all of
/// the wrapping and produces a new `LegalVal` that matches
/// the nominal type of the original buffer.
///
static LegalVal unwrapBufferValue(
    IRTypeLegalizationContext* context,
    LegalVal legalPtrOperand,
    LegalElementWrapping const& elementInfo);

/// Perform any actions required to materialize `val` into a usable value.
///
/// Certain case of `LegalVal` (currently just the `wrappedBuffer` case) are
/// suitable for use to represent a variable, but cannot be used directly
/// in computations, because their structured needs to be "unwrapped."
///
/// This function unwraps any `val` that needs it, which may involve
/// emitting additional IR instructions, and returns the unmodified
/// `val` otherwise.
///
static LegalVal maybeMaterializeWrappedValue(IRTypeLegalizationContext* context, LegalVal val)
{
    if (val.flavor != LegalVal::Flavor::wrappedBuffer)
        return val;

    auto wrappedBufferVal = val.getWrappedBuffer();
    return unwrapBufferValue(context, wrappedBufferVal->base, wrappedBufferVal->elementInfo);
}

// Take a value that is being used as an operand,
// and turn it into the equivalent legalized value.
static LegalVal legalizeOperand(IRTypeLegalizationContext* context, IRInst* irValue)
{
    LegalVal legalVal;
    if (context->mapValToLegalVal.tryGetValue(irValue, legalVal))
    {
        return maybeMaterializeWrappedValue(context, legalVal);
    }

    // For now, assume that anything not covered
    // by the mapping is legal as-is.

    return LegalVal::simple(irValue);
}

/// Helper type for legalization an IR `call` instruction
struct LegalCallBuilder
{
    LegalCallBuilder(IRTypeLegalizationContext* context, IRCall* call)
        : m_context(context), m_call(call)
    {
    }

    /// The context for legalization
    IRTypeLegalizationContext* m_context = nullptr;

    /// The `call` instruction we are legalizing
    IRCall* m_call = nullptr;

    /// The legalized arguments for the call
    ShortList<IRInst*> m_args;

    /// Add a logical argument to the call (which may map to zero or mmore actual arguments)
    void addArg(LegalVal const& val)
    {
        // In order to add the argument(s) for `val`,
        // we will recurse over its structure.

        switch (val.flavor)
        {
        case LegalVal::Flavor::none:
            break;

        case LegalVal::Flavor::simple:
            m_args.add(val.getSimple());
            break;

        case LegalVal::Flavor::implicitDeref:
            addArg(val.getImplicitDeref());
            break;

        case LegalVal::Flavor::pair:
            {
                auto pairVal = val.getPair();
                addArg(pairVal->ordinaryVal);
                addArg(pairVal->specialVal);
            }
            break;

        case LegalVal::Flavor::tuple:
            {
                auto tuplePsuedoVal = val.getTuple();
                for (auto elem : val.getTuple()->elements)
                {
                    addArg(elem.val);
                }
            }
            break;

        default:
            SLANG_UNEXPECTED("uhandled val flavor");
            break;
        }
    }

    /// Build a new call based on the original, given the expected `resultType`.
    ///
    /// Returns a value representing the result of the call.
    LegalVal build(LegalType const& resultType)
    {
        // We can recursively decompose the cases for
        // how to legalize a call based on the expected
        // result type.
        //
        switch (resultType.flavor)
        {
        case LegalType::Flavor::simple:
            // In the case where the result type is simple,
            // we can directly emit the `call` instruction
            // and use the result as our single result value.
            //
            return LegalVal::simple(_emitCall(resultType.getSimple()));

        case LegalType::Flavor::none:
            // In the case where there is no result type,
            // that is equivalent to the call returning `void`.
            //
            // We directly emit the call and then return an
            // empty value to represent the result.
            //
            _emitCall(m_context->builder->getVoidType());
            return LegalVal();

        case LegalVal::Flavor::implicitDeref:
            // An `implicitDeref` wraps a single value, so we can simply
            // unwrap, recurse on the innter value, and then wrap up
            // the result.
            //
            return LegalVal::implicitDeref(build(resultType.getImplicitDeref()->valueType));

        case LegalVal::Flavor::pair:
            {
                // A `pair` type consists of both an ordinary part and a special part.
                //
                auto pairType = resultType.getPair();

                // The ordinary part will be used as the direct result of the call,
                // while the special part will need to be returned via an `out`
                // argument.
                //
                // We will start by emitting the declaration(s) needed for those
                // `out` arguments that represent the special part, and adding
                // them to the argument list. Basically this step is declaring
                // local variables that will hold the special part of the result,
                // and it returns a value that repsents those variables.
                //
                auto specialVal = _addOutArg(pairType->specialType);

                // Once the argument values for the special part are set up,
                // we can recurse on the ordinary part and emit the actual
                // call operation (which will include our new arguments).
                //
                auto ordinaryVal = build(pairType->ordinaryType);

                // The resulting value will be a pair of the ordinary value
                // (returned from the `call` instruction) and the special value
                // (declared as zero or more local variables).
                //
                RefPtr<PairPseudoVal> pairVal = new PairPseudoVal();
                pairVal->pairInfo = pairType->pairInfo;
                pairVal->ordinaryVal = ordinaryVal;
                pairVal->specialVal = specialVal;
                return LegalVal::pair(pairVal);
            }
            break;

        case LegalVal::Flavor::tuple:
            {
                // A `tuple` value consists of zero or more elements
                // that are each of a "special" type. We will handle
                // *all* of those values as `out` arguments akin to
                // what we did for the special half of a pair type
                // above.
                //
                auto resultVal = _addOutArg(resultType);

                // In this case there was no "ordinary" part to the
                // result type of the function, so we know that
                // the legalization funciton/call will use a `void`
                // result type.
                //
                _emitCall(m_context->builder->getVoidType());
                return resultVal;
            }
            break;

        default:
            // TODO: implement legalization of non-simple return types
            SLANG_UNEXPECTED("unimplemented legalized return type for IRCall.");
        }
    }

private:
    /// Add an `out` argument to the call, to capture the given `resultType`.
    LegalVal _addOutArg(LegalType const& resultType)
    {
        switch (resultType.flavor)
        {
        case LegalType::Flavor::simple:
            {
                // In the leaf case we have a simple type, and
                // we just want to declare a local variable based on it.
                //
                auto simpleType = resultType.getSimple();
                auto builder = m_context->builder;

                // Recall that a local variable in our IR represents a *pointer*
                // to storage of the appropriate type.
                //
                auto varPtr = builder->emitVar(simpleType);

                // We need to pass that pointer as an argument to our new
                // `call` instruction, so that it can receive the value
                // written by the callee.
                //
                m_args.add(varPtr);

                // Note: Because `varPtr` is a pointer to the value we want,
                // we have the small problem of needing to return a `LegalVal`
                // that has dereferenced the value after the call.
                //
                // We solve this problem by inserting as `load` from our
                // new variable immediately after the call, before going
                // and resetting the insertion point to continue inserting
                // stuff before the call (which is where we wnat the local
                // variable declarations to go).
                //
                // TODO: Confirm that this logic can't go awry if (somehow)
                // there is no instruction after `m_call`. That should not
                // be possible inside of a function body, but it could in
                // theory be a problem if we ever have top-level module-scope
                // code representing initialization of constants and/or globals.
                //
                builder->setInsertBefore(m_call->getNextInst());
                auto val = builder->emitLoad(simpleType, varPtr);
                builder->setInsertBefore(m_call);

                return LegalVal::simple(val);
            }
            break;

            // The remaining cases are a straightforward structural recursion
            // on top of the base case above.

        case LegalType::Flavor::none:
            return LegalVal();

        case LegalVal::Flavor::implicitDeref:
            return LegalVal::implicitDeref(_addOutArg(resultType.getImplicitDeref()->valueType));

        case LegalVal::Flavor::pair:
            {
                auto pairType = resultType.getPair();
                auto specialVal = _addOutArg(pairType->specialType);
                auto ordinaryVal = _addOutArg(pairType->ordinaryType);

                RefPtr<PairPseudoVal> pairVal = new PairPseudoVal();
                pairVal->pairInfo = pairType->pairInfo;
                pairVal->ordinaryVal = ordinaryVal;
                pairVal->specialVal = specialVal;

                return LegalVal::pair(pairVal);
            }
            break;

        case LegalVal::Flavor::tuple:
            {
                auto tuplePsuedoType = resultType.getTuple();

                RefPtr<TuplePseudoVal> tupleVal = new TuplePseudoVal();
                for (auto typeElement : tuplePsuedoType->elements)
                {
                    TuplePseudoVal::Element valElement;
                    valElement.key = typeElement.key;
                    valElement.val = _addOutArg(typeElement.type);
                    tupleVal->elements.add(valElement);
                }

                return LegalVal::tuple(tupleVal);
            }
            break;

        default:
            // TODO: implement legalization of non-simple return types
            SLANG_UNEXPECTED("unimplemented legalized return type for IRCall.");
        }
    }

    /// Emit the actual `call` instruction given an IR result type
    IRInst* _emitCall(IRType* resultType)
    {
        // The generated call will include all of the arguments that have
        // been added up to this point, which includes those that were
        // added to represent legalized parts of the result type.
        //
        return m_context->builder->emitCallInst(
            resultType,
            m_call->getCallee(),
            m_args.getCount(),
            m_args.getArrayView().getBuffer());
    }
};


static LegalVal legalizeCall(IRTypeLegalizationContext* context, IRCall* callInst)
{
    LegalCallBuilder builder(context, callInst);

    auto argCount = callInst->getArgCount();
    for (UInt i = 0; i < argCount; i++)
    {
        auto legalArg = legalizeOperand(context, callInst->getArg(i));
        builder.addArg(legalArg);
    }

    auto legalResultType = legalizeType(context, callInst->getFullType());
    return builder.build(legalResultType);
}

/// Helper type for legalizing a `returnVal` instruction
struct LegalReturnBuilder
{
    LegalReturnBuilder(IRTypeLegalizationContext* context, IRReturn* returnInst)
        : m_context(context), m_returnInst(returnInst)
    {
    }

    /// Emit code to perform a return of `val`
    void returnVal(LegalVal val)
    {
        auto builder = m_context->builder;

        switch (val.flavor)
        {
        case LegalVal::Flavor::simple:
            // The case of a simple value is easy: just emit a `returnVal`.
            //
            builder->emitReturn(val.getSimple());
            break;

        case LegalVal::Flavor::none:
            // The case of an empty/void value is also easy: emit a `return`.
            //
            builder->emitReturn();
            break;

        case LegalVal::Flavor::implicitDeref:
            returnVal(val.getImplicitDeref());
            break;

        case LegalVal::Flavor::pair:
            {
                // The case for a pair value is the main interesting one.
                // We need to write the special part of the return value
                // to the `out` parameters that were declared to capture
                // it, and then return the ordinary part of the value
                // like normal.
                //
                // Note that the order here matters, because we need to
                // emit the code that writes to the `out` parameters
                // before the `return` instruction.
                //
                auto pairVal = val.getPair();
                _writeResultParam(pairVal->specialVal);
                returnVal(pairVal->ordinaryVal);
            }
            break;

        case LegalVal::Flavor::tuple:
            {
                // The tuple case is kind of a degenerate combination
                // of the `pair` and `none` cases: we need to emit
                // writes to the `out` parameters declared to capture
                // the tuple (all of it), and then we do a `return`
                // of `void` because there is no ordinary result to
                // capture.
                //
                _writeResultParam(val);
                builder->emitReturn();
            }
            break;

        default:
            // TODO: implement legalization of non-simple return types
            SLANG_UNEXPECTED("unimplemented legalized return type for IRReturn.");
        }
    }

private:
    /// Write `val` to the `out` parameters of the enclosing function
    void _writeResultParam(LegalVal const& val)
    {
        switch (val.flavor)
        {
        case LegalVal::Flavor::simple:
            {
                // The leaf case here is the interesting one.
                //
                // We know that if we are writing to `out` parameters to
                // represent the function result then the function must
                // have been legalized in a way that introduced those parameters.
                // We thus need to look up the information on how the
                // function got legalized so that we can identify the
                // new parameters.
                //
                // TODO: One detail worth confirming here is whether there
                // could ever be a case where a `return` instruction gets legalized
                // before its outer function does.
                //
                if (!m_parentFuncInfo)
                {
                    // We start by searching for the ancestor instruction
                    // that represents the function (or other code-bearing value)
                    // that holds this instruction.
                    //
                    auto p = m_returnInst->getParent();
                    while (p && !as<IRGlobalValueWithCode>(p))
                    {
                        p = p->parent;
                    }

                    // We expect that the parent is actually an IR function.
                    //
                    // TODO: What about the case where we have an `IRGlobalVar`
                    // of a type that needs legalization, and the variable has
                    // an initializer? For now, I believe that case is disallowed
                    // in the legalization for global variables.
                    //
                    auto parentFunc = as<IRFunc>(p);
                    SLANG_ASSERT(parentFunc);
                    if (!parentFunc)
                        return;

                    // We also expect that extended legalization information was
                    // recorded for the function.
                    //
                    RefPtr<LegalFuncInfo> parentFuncInfo;
                    if (!m_context->mapFuncToInfo.tryGetValue(parentFunc, parentFuncInfo))
                    {
                        // If we fail to find the extended information then either:
                        //
                        // * The parent function has not been legalized yet. This would
                        //   be a violation of our assumption about ordering of legalization.
                        //
                        // * The parent function was legalized, but didn't require any
                        //   additional IR parameters to represent its result. This would
                        //   be a violation of our assumption that the declared result type
                        //   of a function and the type at `return` sites inside the function
                        //   need to match.
                        //
                        SLANG_ASSERT(parentFuncInfo);
                        return;
                    }

                    // If we find the extended information, then this is the first
                    // leaf parameter we are dealing with, so we set up to read through
                    // the parameters starting at index zero.
                    //
                    m_parentFuncInfo = parentFuncInfo;
                    m_resultParamCounter = 0;
                }
                SLANG_ASSERT(m_parentFuncInfo);

                // The recursion through the result `val` will iterate over the
                // leaf parameters in the same order they should have been declared,
                // so the parameter we need to write to will be the next one in order.
                //
                // We expect that the parameter index must be in range, beacuse otherwise
                // the recursion here and the recursion that declared the parameters are
                // mismatched in terms of how they traversed the hierarchical representation
                // of `LegalVal` / `LegalType`.
                //
                Index resultParamIndex = m_resultParamCounter++;
                SLANG_ASSERT(resultParamIndex >= 0);
                SLANG_ASSERT(resultParamIndex < m_parentFuncInfo->resultParamVals.getCount());

                // Once we've identified the right parameter, we can emit a `store`
                // to write the value that the function wants to output.
                //
                // Note that an `out` parameter is represented with a pointer type
                // in the IR, so that the `IRParam` here represents a pointer to
                // the value that will receive the result.
                //
                auto resultParamPtr = m_parentFuncInfo->resultParamVals[resultParamIndex];
                m_context->builder->emitStore(resultParamPtr, val.getSimple());
            }
            break;

            // The remaining cases are just a straightforward recursion
            // over the structure of the `val`.

        case LegalVal::Flavor::none:
            break;

        case LegalVal::Flavor::implicitDeref:
            _writeResultParam(val.getImplicitDeref());
            break;

        case LegalVal::Flavor::pair:
            {
                auto pairVal = val.getPair();
                _writeResultParam(pairVal->ordinaryVal);
                _writeResultParam(pairVal->specialVal);
            }
            break;

        case LegalVal::Flavor::tuple:
            {
                auto tupleVal = val.getTuple();
                for (auto element : tupleVal->elements)
                {
                    _writeResultParam(element.val);
                }
            }
            break;

        default:
            // TODO: implement legalization of non-simple return types
            SLANG_UNEXPECTED("unimplemented legalized return type for IRReturn.");
        }
    }

    IRTypeLegalizationContext* m_context = nullptr;
    IRReturn* m_returnInst = nullptr;

    RefPtr<LegalFuncInfo> m_parentFuncInfo;
    Index m_resultParamCounter = 0;
};

static LegalVal legalizeRetVal(
    IRTypeLegalizationContext* context,
    LegalVal retVal,
    IRReturn* returnInst)
{
    LegalReturnBuilder builder(context, returnInst);
    builder.returnVal(retVal);
    return LegalVal();
}

static void _addVal(ShortList<IRInst*>& rs, const LegalVal& legalVal)
{
    switch (legalVal.flavor)
    {
    case LegalVal::Flavor::simple:
        rs.add(legalVal.getSimple());
        break;
    case LegalVal::Flavor::tuple:
        for (auto element : legalVal.getTuple()->elements)
            _addVal(rs, element.val);
        break;
    case LegalVal::Flavor::pair:
        _addVal(rs, legalVal.getPair()->ordinaryVal);
        _addVal(rs, legalVal.getPair()->specialVal);
        break;
    case LegalVal::Flavor::none:
        break;
    default:
        SLANG_UNEXPECTED("unhandled legalized val flavor");
    }
}

static LegalVal legalizeUnconditionalBranch(
    IRTypeLegalizationContext* context,
    ArrayView<LegalVal> args,
    IRUnconditionalBranch* branchInst)
{
    ShortList<IRInst*> newArgs;
    for (auto arg : args)
    {
        switch (arg.flavor)
        {
        case LegalVal::Flavor::none:
            break;
        case LegalVal::Flavor::simple:
            newArgs.add(arg.getSimple());
            break;
        case LegalVal::Flavor::pair:
            _addVal(newArgs, arg.getPair()->ordinaryVal);
            _addVal(newArgs, arg.getPair()->specialVal);
            break;
        case LegalVal::Flavor::tuple:
            for (auto element : arg.getTuple()->elements)
            {
                _addVal(newArgs, element.val);
            }
            break;
        default:
            SLANG_UNIMPLEMENTED_X("Unknown legalized val flavor.");
        }
    }
    context->builder->emitIntrinsicInst(
        nullptr,
        branchInst->getOp(),
        newArgs.getCount(),
        newArgs.getArrayView().getBuffer());
    return LegalVal();
}

static LegalVal legalizeLoad(IRTypeLegalizationContext* context, LegalVal legalPtrVal)
{
    switch (legalPtrVal.flavor)
    {
    case LegalVal::Flavor::none:
        return LegalVal();

    case LegalVal::Flavor::simple:
        {
            return LegalVal::simple(context->builder->emitLoad(legalPtrVal.getSimple()));
        }
        break;

    case LegalVal::Flavor::implicitDeref:
        // We have turne a pointer(-like) type into its pointed-to (value)
        // type, and so the operation of loading goes away; we just use
        // the underlying value.
        return legalPtrVal.getImplicitDeref();

    case LegalVal::Flavor::pair:
        {
            auto ptrPairVal = legalPtrVal.getPair();

            auto ordinaryVal = legalizeLoad(context, ptrPairVal->ordinaryVal);
            auto specialVal = legalizeLoad(context, ptrPairVal->specialVal);
            return LegalVal::pair(ordinaryVal, specialVal, ptrPairVal->pairInfo);
        }

    case LegalVal::Flavor::tuple:
        {
            // We need to emit a load for each element of
            // the tuple.
            auto ptrTupleVal = legalPtrVal.getTuple();
            RefPtr<TuplePseudoVal> tupleVal = new TuplePseudoVal();

            for (auto ee : legalPtrVal.getTuple()->elements)
            {
                TuplePseudoVal::Element element;
                element.key = ee.key;
                element.val = legalizeLoad(context, ee.val);

                tupleVal->elements.add(element);
            }
            return LegalVal::tuple(tupleVal);
        }
        break;

    default:
        SLANG_UNEXPECTED("unhandled case");
        break;
    }
}

static LegalVal legalizePrintf(IRTypeLegalizationContext* context, ArrayView<LegalVal> args)
{
    ShortList<IRInst*> legalArgs;
    for (auto arg : args)
    {
        switch (arg.flavor)
        {
        case LegalVal::Flavor::none:
            break;
        case LegalVal::Flavor::simple:
            legalArgs.add(arg.getSimple());
            break;
        case LegalVal::Flavor::pair:
            legalArgs.add(arg.getPair()->ordinaryVal.getSimple());
            break;
        default:
            SLANG_UNIMPLEMENTED_X("Unknown legalized val flavor for printf operand");
        }
    }
    return LegalVal::simple(context->builder->emitIntrinsicInst(
        context->builder->getVoidType(),
        kIROp_Printf,
        (UInt)legalArgs.getCount(),
        legalArgs.getArrayView().getBuffer()));
}

static LegalVal legalizeDebugVar(
    IRTypeLegalizationContext* context,
    LegalType type,
    IRDebugVar* originalInst)
{
    // For now we just discard any special part and keep the ordinary part.

    switch (type.flavor)
    {
    case LegalType::Flavor::simple:
        {
            auto legalVal = context->builder->emitDebugVar(
                tryGetPointedToType(context->builder, type.getSimple()),
                originalInst->getSource(),
                originalInst->getLine(),
                originalInst->getCol(),
                originalInst->getArgIndex());
            copyNameHintAndDebugDecorations(legalVal, originalInst);
            return LegalVal::simple(legalVal);
        }
    case LegalType::Flavor::none:
        return LegalVal();
    case LegalType::Flavor::pair:
        {
            auto pairType = type.getPair();
            auto ordinaryVal = legalizeDebugVar(context, pairType->ordinaryType, originalInst);
            return ordinaryVal;
        }
    case LegalType::Flavor::tuple:
        {
            auto tupleType = type.getTuple();
            for (auto ee : tupleType->elements)
            {
                auto innerResult = legalizeDebugVar(context, ee.type, originalInst);
                if (innerResult.flavor != LegalVal::Flavor::none)
                    return innerResult;
            }
            return LegalVal();
        }
    default:
        return LegalVal();
    }
}

static LegalVal legalizeDebugValue(
    IRTypeLegalizationContext* context,
    LegalVal debugVar,
    LegalVal debugValue,
    IRDebugValue* originalInst)
{
    // For now we just discard any special part and keep the ordinary part.
    switch (debugValue.flavor)
    {
    case LegalType::Flavor::simple:
        return LegalVal::simple(
            context->builder->emitDebugValue(debugVar.getSimple(), debugValue.getSimple()));
    case LegalType::Flavor::none:
        return LegalVal();
    case LegalType::Flavor::pair:
        {
            auto ordinaryVal = legalizeDebugValue(
                context,
                debugVar.getPair()->ordinaryVal,
                debugValue.getPair()->ordinaryVal,
                originalInst);
            return ordinaryVal;
        }
    case LegalType::Flavor::tuple:
        {
            auto tupleVal = debugValue.getTuple();
            for (auto ee : tupleVal->elements)
            {
                auto innerResult = legalizeDebugValue(context, debugVar, ee.val, originalInst);
                if (innerResult.flavor != LegalVal::Flavor::none)
                    return innerResult;
            }
            return LegalVal();
        }
    default:
        return LegalVal();
    }
}

static LegalVal legalizeStore(
    IRTypeLegalizationContext* context,
    LegalVal legalPtrVal,
    LegalVal legalVal)
{
    switch (legalPtrVal.flavor)
    {
    case LegalVal::Flavor::none:
        return LegalVal();

    case LegalVal::Flavor::simple:
        {
            if (legalVal.flavor == LegalVal::Flavor::none)
                return LegalVal();
            context->builder->emitStore(legalPtrVal.getSimple(), legalVal.getSimple());
            return legalVal;
        }
        break;

    case LegalVal::Flavor::implicitDeref:
        // TODO: what is the right behavior here?
        //
        // The crux of the problem is that we may legalize a pointer-to-pointer
        // type in cases where one of the two needs to become an implicit-deref,
        // so that we have `PtrA<PtrB<Thing>>` become, say, `PtrA<Thing>` with
        // an `implicitDeref` wrapper. When we encounter a store to that
        // wrapped value, we seemingly need to know whether the original code
        // meant to store to `*ptrPtr` or `**ptrPtr`, and need to legalize
        // the result accordingly...
        //
        if (legalVal.flavor == LegalVal::Flavor::implicitDeref)
            return legalizeStore(
                context,
                legalPtrVal.getImplicitDeref(),
                legalVal.getImplicitDeref());
        else
            return legalizeStore(context, legalPtrVal.getImplicitDeref(), legalVal);

    case LegalVal::Flavor::pair:
        {
            auto destPair = legalPtrVal.getPair();
            auto valPair = legalVal.getPair();
            legalizeStore(context, destPair->ordinaryVal, valPair->ordinaryVal);
            legalizeStore(context, destPair->specialVal, valPair->specialVal);
            return LegalVal();
        }

    case LegalVal::Flavor::tuple:
        {
            // We need to emit a store for each element of
            // the tuple.
            auto destTuple = legalPtrVal.getTuple();
            auto valTuple = legalVal.getTuple();
            SLANG_ASSERT(destTuple->elements.getCount() == valTuple->elements.getCount());

            for (Index i = 0; i < valTuple->elements.getCount(); i++)
            {
                legalizeStore(context, destTuple->elements[i].val, valTuple->elements[i].val);
            }
            return legalVal;
        }
        break;

    default:
        SLANG_UNEXPECTED("unhandled case");
        break;
    }
}

static LegalVal legalizeFieldExtract(
    IRTypeLegalizationContext* context,
    LegalType type,
    LegalVal legalStructOperand,
    IRStructKey* fieldKey)
{
    auto builder = context->builder;

    if (type.flavor == LegalType::Flavor::none)
        return LegalVal();

    switch (legalStructOperand.flavor)
    {
    case LegalVal::Flavor::none:
        return LegalVal();

    case LegalVal::Flavor::simple:
        return LegalVal::simple(
            builder->emitFieldExtract(type.getSimple(), legalStructOperand.getSimple(), fieldKey));

    case LegalVal::Flavor::pair:
        {
            // There are two sides, the ordinary and the special,
            // and we basically just dispatch to both of them.
            auto pairVal = legalStructOperand.getPair();
            auto pairInfo = pairVal->pairInfo;
            auto pairElement = pairInfo->findElement(fieldKey);
            if (!pairElement)
            {
                SLANG_UNEXPECTED("didn't find tuple element");
                UNREACHABLE_RETURN(LegalVal());
            }

            // If the field we are extracting has a pair type,
            // that means it exists on both the ordinary and
            // special sides.
            RefPtr<PairInfo> fieldPairInfo;
            LegalType ordinaryType = type;
            LegalType specialType = type;
            if (type.flavor == LegalType::Flavor::pair)
            {
                auto fieldPairType = type.getPair();
                fieldPairInfo = fieldPairType->pairInfo;
                ordinaryType = fieldPairType->ordinaryType;
                specialType = fieldPairType->specialType;
            }

            LegalVal ordinaryVal;
            LegalVal specialVal;

            if (pairElement->flags & PairInfo::kFlag_hasOrdinary)
            {
                ordinaryVal =
                    legalizeFieldExtract(context, ordinaryType, pairVal->ordinaryVal, fieldKey);
            }

            if (pairElement->flags & PairInfo::kFlag_hasSpecial)
            {
                specialVal =
                    legalizeFieldExtract(context, specialType, pairVal->specialVal, fieldKey);
            }
            return LegalVal::pair(ordinaryVal, specialVal, fieldPairInfo);
        }
        break;

    case LegalVal::Flavor::tuple:
        {
            // The operand is a tuple of pointer-like
            // values, we want to extract the element
            // corresponding to a field. We will handle
            // this by simply returning the corresponding
            // element from the operand.
            auto ptrTupleInfo = legalStructOperand.getTuple();
            for (auto ee : ptrTupleInfo->elements)
            {
                if (ee.key == fieldKey)
                {
                    return ee.val;
                }
            }

            // TODO: we can legally reach this case now
            // when the field is "ordinary".

            SLANG_UNEXPECTED("didn't find tuple element");
            UNREACHABLE_RETURN(LegalVal());
        }

    default:
        SLANG_UNEXPECTED("unhandled");
        UNREACHABLE_RETURN(LegalVal());
    }
}

static LegalVal legalizeFieldExtract(
    IRTypeLegalizationContext* context,
    LegalType type,
    LegalVal legalPtrOperand,
    LegalVal legalFieldOperand)
{
    // We don't expect any legalization to affect
    // the "field" argument.
    auto fieldKey = legalFieldOperand.getSimple();

    return legalizeFieldExtract(context, type, legalPtrOperand, (IRStructKey*)fieldKey);
}

/// Take a value of some buffer/pointer type and unwrap it according to provided info.
static LegalVal unwrapBufferValue(
    IRTypeLegalizationContext* context,
    LegalVal legalPtrOperand,
    LegalElementWrapping const& elementInfo)
{
    // The `elementInfo` tells us how a non-simple element
    // type was wrapped up into a new structure types used
    // as the element type of the buffer.
    //
    // This function will recurse through the structure of
    // `elementInfo` to pull out all the required data from
    // the buffer represented by `legalPtrOperand`.

    switch (elementInfo.flavor)
    {
    default:
        SLANG_UNEXPECTED("unhandled");
        UNREACHABLE_RETURN(LegalVal());
        break;

    case LegalElementWrapping::Flavor::none:
        return LegalVal();

    case LegalElementWrapping::Flavor::simple:
        {
            // In the leaf case, we just had to store some
            // data of a simple type in the buffer. We can
            // produce a valid result by computing the
            // address of the field used to represent the
            // element, and then returning *that* as if
            // it were the buffer type itself.
            //
            // (Basically instead of `someBuffer` we will
            // end up with `&(someBuffer->field)`.
            //
            auto builder = context->getBuilder();

            auto simpleElementInfo = elementInfo.getSimple();
            auto valPtr = builder->emitFieldAddress(
                builder->getPtrType(simpleElementInfo->type),
                legalPtrOperand.getSimple(),
                simpleElementInfo->key);

            return LegalVal::simple(valPtr);
        }

    case LegalElementWrapping::Flavor::implicitDeref:
        {
            // If the element type was logically `ImplicitDeref<T>`,
            // then we declared actual fields based on `T`, and
            // we need to extract references to those fields and
            // wrap them up in an `implicitDeref` value.
            //
            auto derefField = elementInfo.getImplicitDeref();
            auto baseVal = unwrapBufferValue(context, legalPtrOperand, derefField->field);
            return LegalVal::implicitDeref(baseVal);
        }

    case LegalElementWrapping::Flavor::pair:
        {
            // If the element type was logically a `Pair<O,S>`
            // then we encoded fields for both `O` and `S` into
            // the actual element type, and now we need to
            // extract references to both and pair them up.
            //
            auto pairField = elementInfo.getPair();
            auto pairInfo = pairField->pairInfo;

            auto ordinaryVal = unwrapBufferValue(context, legalPtrOperand, pairField->ordinary);
            auto specialVal = unwrapBufferValue(context, legalPtrOperand, pairField->special);
            return LegalVal::pair(ordinaryVal, specialVal, pairInfo);
        }

    case LegalElementWrapping::Flavor::tuple:
        {
            // If the element type was logically a `Tuple<E0, E1, ...>`
            // then we encoded fields for each of the `Ei` and
            // need to extract references to all of them and
            // encode them as a tuple.
            //
            auto tupleField = elementInfo.getTuple();

            RefPtr<TuplePseudoVal> obj = new TuplePseudoVal();
            for (auto ee : tupleField->elements)
            {
                auto elementVal = unwrapBufferValue(context, legalPtrOperand, ee.field);

                TuplePseudoVal::Element element;
                element.key = ee.key;
                element.val = unwrapBufferValue(context, legalPtrOperand, ee.field);
                obj->elements.add(element);
            }

            return LegalVal::tuple(obj);
        }
    }
}

static IRType* getPointedToType(IRTypeLegalizationContext* context, IRType* ptrType)
{
    auto valueType = tryGetPointedToType(context->builder, ptrType);
    if (!valueType)
    {
        SLANG_UNEXPECTED("expected a pointer type during type legalization");
    }
    return valueType;
}

static LegalType getPointedToType(IRTypeLegalizationContext* context, LegalType type)
{
    switch (type.flavor)
    {
    case LegalType::Flavor::none:
        return LegalType();

    case LegalType::Flavor::simple:
        return LegalType::simple(getPointedToType(context, type.getSimple()));

    case LegalType::Flavor::implicitDeref:
        return type.getImplicitDeref()->valueType;

    case LegalType::Flavor::pair:
        {
            auto pairType = type.getPair();
            auto ordinary = getPointedToType(context, pairType->ordinaryType);
            auto special = getPointedToType(context, pairType->specialType);
            return LegalType::pair(ordinary, special, pairType->pairInfo);
        }

    case LegalType::Flavor::tuple:
        {
            auto tupleType = type.getTuple();
            RefPtr<TuplePseudoType> resultTuple = new TuplePseudoType();
            for (auto ee : tupleType->elements)
            {
                TuplePseudoType::Element resultElement;
                resultElement.key = ee.key;
                resultElement.type = getPointedToType(context, ee.type);
                resultTuple->elements.add(resultElement);
            }
            return LegalType::tuple(resultTuple);
        }

    default:
        SLANG_UNEXPECTED("unhandled case in type legalization");
        UNREACHABLE_RETURN(LegalType());
    }
}

static LegalVal legalizeFieldAddress(
    IRTypeLegalizationContext* context,
    LegalType type,
    LegalVal legalPtrOperand,
    IRStructKey* fieldKey)
{
    auto builder = context->builder;
    if (type.flavor == LegalType::Flavor::none)
        return LegalVal();

    switch (legalPtrOperand.flavor)
    {
    case LegalVal::Flavor::none:
        return LegalVal();

    case LegalVal::Flavor::simple:
        switch (type.flavor)
        {
        case LegalType::Flavor::implicitDeref:
            // TODO: Should this case be needed?
            return legalizeFieldAddress(
                context,
                type.getImplicitDeref()->valueType,
                legalPtrOperand,
                fieldKey);

        default:
            return LegalVal::simple(
                builder->emitFieldAddress(legalPtrOperand.getSimple(), fieldKey));
        }

    case LegalVal::Flavor::pair:
        {
            // There are two sides, the ordinary and the special,
            // and we basically just dispatch to both of them.
            auto pairVal = legalPtrOperand.getPair();
            auto pairInfo = pairVal->pairInfo;
            auto pairElement = pairInfo->findElement(fieldKey);
            if (!pairElement)
            {
                SLANG_UNEXPECTED("didn't find tuple element");
                UNREACHABLE_RETURN(LegalVal());
            }

            // If the field we are extracting has a pair type,
            // that means it exists on both the ordinary and
            // special sides.
            RefPtr<PairInfo> fieldPairInfo;
            LegalType ordinaryType = type;
            LegalType specialType = type;
            if (type.flavor == LegalType::Flavor::pair)
            {
                auto fieldPairType = type.getPair();
                fieldPairInfo = fieldPairType->pairInfo;
                ordinaryType = fieldPairType->ordinaryType;
                specialType = fieldPairType->specialType;
            }

            LegalVal ordinaryVal;
            LegalVal specialVal;

            if (pairElement->flags & PairInfo::kFlag_hasOrdinary)
            {
                ordinaryVal =
                    legalizeFieldAddress(context, ordinaryType, pairVal->ordinaryVal, fieldKey);
            }

            if (pairElement->flags & PairInfo::kFlag_hasSpecial)
            {
                specialVal =
                    legalizeFieldAddress(context, specialType, pairVal->specialVal, fieldKey);
            }
            return LegalVal::pair(ordinaryVal, specialVal, fieldPairInfo);
        }
        break;

    case LegalVal::Flavor::tuple:
        {
            // The operand is a tuple of pointer-like
            // values, we want to extract the element
            // corresponding to a field. We will handle
            // this by simply returning the corresponding
            // element from the operand.
            auto ptrTupleInfo = legalPtrOperand.getTuple();
            for (auto ee : ptrTupleInfo->elements)
            {
                if (ee.key == fieldKey)
                {
                    return ee.val;
                }
            }

            // TODO: we can legally reach this case now
            // when the field is "ordinary".

            SLANG_UNEXPECTED("didn't find tuple element");
            UNREACHABLE_RETURN(LegalVal());
        }

    case LegalVal::Flavor::implicitDeref:
        {
            // The original value had a level of indirection
            // that is now being removed, so should not be
            // able to get at the *address* of the field any
            // more, and need to resign ourselves to just
            // getting at the field *value* and then
            // adding an implicit dereference on top of that.
            //
            auto implicitDerefVal = legalPtrOperand.getImplicitDeref();
            auto valueType = getPointedToType(context, type);
            return LegalVal::implicitDeref(
                legalizeFieldExtract(context, valueType, implicitDerefVal, fieldKey));
        }

    default:
        SLANG_UNEXPECTED("unhandled");
        UNREACHABLE_RETURN(LegalVal());
    }
}

static LegalVal legalizeFieldAddress(
    IRTypeLegalizationContext* context,
    LegalType type,
    LegalVal legalPtrOperand,
    LegalVal legalFieldOperand)
{
    // We don't expect any legalization to affect
    // the "field" argument.
    auto fieldKey = legalFieldOperand.getSimple();

    return legalizeFieldAddress(context, type, legalPtrOperand, (IRStructKey*)fieldKey);
}

static LegalVal legalizeGetElement(
    IRTypeLegalizationContext* context,
    LegalType type,
    LegalVal legalPtrOperand,
    IRInst* indexOperand)
{
    auto builder = context->builder;

    switch (legalPtrOperand.flavor)
    {
    case LegalVal::Flavor::none:
        return LegalVal();

    case LegalVal::Flavor::simple:
        return LegalVal::simple(builder->emitElementExtract(
            type.getSimple(),
            legalPtrOperand.getSimple(),
            indexOperand));

    case LegalVal::Flavor::pair:
        {
            // There are two sides, the ordinary and the special,
            // and we basically just dispatch to both of them.
            auto pairVal = legalPtrOperand.getPair();
            auto pairInfo = pairVal->pairInfo;

            LegalType ordinaryType = type;
            LegalType specialType = type;
            if (type.flavor == LegalType::Flavor::pair)
            {
                auto pairType = type.getPair();
                ordinaryType = pairType->ordinaryType;
                specialType = pairType->specialType;
            }

            LegalVal ordinaryVal =
                legalizeGetElement(context, ordinaryType, pairVal->ordinaryVal, indexOperand);

            LegalVal specialVal =
                legalizeGetElement(context, specialType, pairVal->specialVal, indexOperand);

            return LegalVal::pair(ordinaryVal, specialVal, pairInfo);
        }
        break;

    case LegalVal::Flavor::tuple:
        {
            // The operand is a tuple of pointer-like
            // values, we want to extract the element
            // corresponding to a field. We will handle
            // this by simply returning the corresponding
            // element from the operand.
            auto ptrTupleInfo = legalPtrOperand.getTuple();

            RefPtr<TuplePseudoVal> resTupleInfo = new TuplePseudoVal();

            auto tupleType = type.getTuple();
            SLANG_ASSERT(tupleType);

            auto elemCount = ptrTupleInfo->elements.getCount();
            SLANG_ASSERT(elemCount == tupleType->elements.getCount());

            for (Index ee = 0; ee < elemCount; ++ee)
            {
                auto ptrElem = ptrTupleInfo->elements[ee];
                auto elemType = tupleType->elements[ee].type;

                TuplePseudoVal::Element resElem;
                resElem.key = ptrElem.key;
                resElem.val = legalizeGetElement(context, elemType, ptrElem.val, indexOperand);

                resTupleInfo->elements.add(resElem);
            }

            return LegalVal::tuple(resTupleInfo);
        }

    default:
        SLANG_UNEXPECTED("unhandled");
        UNREACHABLE_RETURN(LegalVal());
    }
}

static LegalVal legalizeGetElement(
    IRTypeLegalizationContext* context,
    LegalType type,
    LegalVal legalPtrOperand,
    LegalVal legalIndexOperand)
{
    // We don't expect any legalization to affect
    // the "index" argument.
    auto indexOperand = legalIndexOperand.getSimple();

    return legalizeGetElement(context, type, legalPtrOperand, indexOperand);
}

static LegalVal legalizeGetElementPtr(
    IRTypeLegalizationContext* context,
    LegalType type,
    LegalVal legalPtrOperand,
    IRInst* indexOperand)
{
    auto builder = context->builder;

    switch (legalPtrOperand.flavor)
    {
    case LegalVal::Flavor::none:
        return LegalVal();

    case LegalVal::Flavor::simple:
        return LegalVal::simple(builder->emitElementAddress(
            type.getSimple(),
            legalPtrOperand.getSimple(),
            indexOperand));

    case LegalVal::Flavor::pair:
        {
            // There are two sides, the ordinary and the special,
            // and we basically just dispatch to both of them.
            auto pairVal = legalPtrOperand.getPair();
            auto pairInfo = pairVal->pairInfo;

            LegalType ordinaryType = type;
            LegalType specialType = type;
            if (type.flavor == LegalType::Flavor::pair)
            {
                auto pairType = type.getPair();
                ordinaryType = pairType->ordinaryType;
                specialType = pairType->specialType;
            }

            LegalVal ordinaryVal =
                legalizeGetElementPtr(context, ordinaryType, pairVal->ordinaryVal, indexOperand);

            LegalVal specialVal =
                legalizeGetElementPtr(context, specialType, pairVal->specialVal, indexOperand);

            return LegalVal::pair(ordinaryVal, specialVal, pairInfo);
        }
        break;

    case LegalVal::Flavor::tuple:
        {
            // The operand is a tuple of pointer-like
            // values, we want to extract the element
            // corresponding to a field. We will handle
            // this by simply returning the corresponding
            // element from the operand.
            auto ptrTupleInfo = legalPtrOperand.getTuple();

            RefPtr<TuplePseudoVal> resTupleInfo = new TuplePseudoVal();

            auto tupleType = type.getTuple();
            SLANG_ASSERT(tupleType);

            auto elemCount = ptrTupleInfo->elements.getCount();
            SLANG_ASSERT(elemCount == tupleType->elements.getCount());

            for (Index ee = 0; ee < elemCount; ++ee)
            {
                auto ptrElem = ptrTupleInfo->elements[ee];
                auto elemType = tupleType->elements[ee].type;

                TuplePseudoVal::Element resElem;
                resElem.key = ptrElem.key;
                resElem.val = legalizeGetElementPtr(context, elemType, ptrElem.val, indexOperand);

                resTupleInfo->elements.add(resElem);
            }

            return LegalVal::tuple(resTupleInfo);
        }

    case LegalVal::Flavor::implicitDeref:
        {
            // The original value used to be a pointer to an array,
            // and somebody is trying to get at an element pointer.
            // Now we just have an array (wrapped with an implicit
            // dereference) and need to just fetch the chosen element
            // instead (and then wrap the element value with an
            // implicit dereference).
            //
            // The result type for our `getElement` instruction needs
            // to be the type *pointed to* by `type`, and not `type.
            //
            auto valueType = getPointedToType(context, type);

            auto implicitDerefVal = legalPtrOperand.getImplicitDeref();
            return LegalVal::implicitDeref(
                legalizeGetElement(context, valueType, implicitDerefVal, indexOperand));
        }

    default:
        SLANG_UNEXPECTED("unhandled");
        UNREACHABLE_RETURN(LegalVal());
    }
}

static LegalVal legalizeGetElementPtr(
    IRTypeLegalizationContext* context,
    LegalType type,
    LegalVal legalPtrOperand,
    LegalVal legalIndexOperand)
{
    // We don't expect any legalization to affect
    // the "index" argument.
    auto indexOperand = legalIndexOperand.getSimple();

    return legalizeGetElementPtr(context, type, legalPtrOperand, indexOperand);
}

static LegalVal legalizeMakeStruct(
    IRTypeLegalizationContext* context,
    LegalType legalType,
    LegalVal const* legalArgs,
    UInt argCount)
{
    auto builder = context->builder;

    switch (legalType.flavor)
    {
    case LegalType::Flavor::none:
        return LegalVal();

    case LegalType::Flavor::simple:
        {
            List<IRInst*> args;
            for (UInt aa = 0; aa < argCount; ++aa)
            {
                // Ignore none values.
                if (legalArgs[aa].flavor == LegalVal::Flavor::none)
                    continue;

                // Note: we assume that all the arguments
                // must be simple here, because otherwise
                // the `struct` type with them as fields
                // would not be simple...
                //
                args.add(legalArgs[aa].getSimple());
            }
            return LegalVal::simple(
                builder->emitMakeStruct(legalType.getSimple(), args.getCount(), args.getBuffer()));
        }

    case LegalType::Flavor::pair:
        {
            // There are two sides, the ordinary and the special,
            // and we basically just dispatch to both of them.
            auto pairType = legalType.getPair();
            auto pairInfo = pairType->pairInfo;
            LegalType ordinaryType = pairType->ordinaryType;
            LegalType specialType = pairType->specialType;

            List<LegalVal> ordinaryArgs;
            List<LegalVal> specialArgs;
            UInt argCounter = 0;
            for (auto ee : pairInfo->elements)
            {
                UInt argIndex = argCounter++;
                LegalVal arg = legalArgs[argIndex];

                if (arg.flavor == LegalVal::Flavor::pair)
                {
                    // The argument is itself a pair
                    auto argPair = arg.getPair();
                    ordinaryArgs.add(argPair->ordinaryVal);
                    specialArgs.add(argPair->specialVal);
                }
                else if (ee.flags & Slang::PairInfo::kFlag_hasOrdinary)
                {
                    ordinaryArgs.add(arg);
                }
                else if (ee.flags & Slang::PairInfo::kFlag_hasSpecial)
                {
                    specialArgs.add(arg);
                }
            }

            LegalVal ordinaryVal = legalizeMakeStruct(
                context,
                ordinaryType,
                ordinaryArgs.getBuffer(),
                ordinaryArgs.getCount());

            LegalVal specialVal = legalizeMakeStruct(
                context,
                specialType,
                specialArgs.getBuffer(),
                specialArgs.getCount());

            return LegalVal::pair(ordinaryVal, specialVal, pairInfo);
        }
        break;

    case LegalType::Flavor::tuple:
        {
            // We are constructing a tuple of values from
            // the individual fields. We need to identify
            // for each tuple element what field it uses,
            // and then extract that field's value.

            auto tupleType = legalType.getTuple();

            RefPtr<TuplePseudoVal> resTupleInfo = new TuplePseudoVal();
            UInt argCounter = 0;
            for (auto typeElem : tupleType->elements)
            {
                auto elemKey = typeElem.key;
                UInt argIndex = argCounter++;
                SLANG_ASSERT(argIndex < argCount);

                LegalVal argVal = legalArgs[argIndex];

                TuplePseudoVal::Element resElem;
                resElem.key = elemKey;
                resElem.val = argVal;

                resTupleInfo->elements.add(resElem);
            }
            return LegalVal::tuple(resTupleInfo);
        }

    default:
        SLANG_UNEXPECTED("unhandled");
        UNREACHABLE_RETURN(LegalVal());
    }
}

static LegalVal legalizeMakeArray(
    IRTypeLegalizationContext* context,
    LegalType legalType,
    LegalVal const* legalArgs,
    UInt argCount,
    IROp constructOp)
{
    auto builder = context->builder;

    switch (legalType.flavor)
    {
    case LegalType::Flavor::none:
        return LegalVal();

    case LegalType::Flavor::simple:
        {
            List<IRInst*> args;
            // We need a valid default val for elements that are legalized to `none`.
            // We grab the first non-none value from the legalized args and use it.
            // If all args are none (althoguh this shouldn't happen, since the entire array
            // would have been legalized to none in this case.), we use defaultConstruct op.
            // Use of defaultConstruct may lead to invalid HLSL/GLSL code, so we want to
            // avoid that if possible.
            IRInst* defaultVal = nullptr;
            for (UInt aa = 0; aa < argCount; ++aa)
            {
                if (legalArgs[aa].flavor == LegalVal::Flavor::simple)
                {
                    defaultVal = legalArgs[aa].getSimple();
                    break;
                }
            }
            if (!defaultVal)
            {
                defaultVal = builder->emitDefaultConstruct(
                    as<IRArrayTypeBase>(legalType.getSimple())->getElementType());
            }
            for (UInt aa = 0; aa < argCount; ++aa)
            {
                if (legalArgs[aa].flavor == LegalVal::Flavor::none)
                    args.add(defaultVal);
                else
                    args.add(legalArgs[aa].getSimple());
            }
            return LegalVal::simple(builder->emitIntrinsicInst(
                legalType.getSimple(),
                constructOp,
                args.getCount(),
                args.getBuffer()));
        }

    case LegalType::Flavor::pair:
        {
            // There are two sides, the ordinary and the special,
            // and we basically just dispatch to both of them.
            auto pairType = legalType.getPair();
            auto pairInfo = pairType->pairInfo;
            LegalType ordinaryType = pairType->ordinaryType;
            LegalType specialType = pairType->specialType;

            List<LegalVal> ordinaryArgs;
            List<LegalVal> specialArgs;
            bool hasValidOrdinaryArgs = false;
            bool hasValidSpecialArgs = false;
            for (UInt argIndex = 0; argIndex < argCount; argIndex++)
            {
                LegalVal arg = legalArgs[argIndex];

                // The argument must be a pair.
                if (arg.flavor == LegalVal::Flavor::pair)
                {
                    auto argPair = arg.getPair();
                    ordinaryArgs.add(argPair->ordinaryVal);
                    specialArgs.add(argPair->specialVal);
                    hasValidOrdinaryArgs = true;
                    hasValidSpecialArgs = true;
                }
                else if (arg.flavor == LegalVal::Flavor::simple)
                {
                    if (arg.getSimple()->getFullType() == ordinaryType.irType)
                    {
                        ordinaryArgs.add(arg);
                        specialArgs.add(LegalVal());
                        hasValidOrdinaryArgs = true;
                    }
                    else
                    {
                        ordinaryArgs.add(LegalVal());
                        specialArgs.add(arg);
                        hasValidSpecialArgs = true;
                    }
                }
                else if (arg.flavor == LegalVal::Flavor::none)
                {
                    ordinaryArgs.add(arg);
                    specialArgs.add(arg);
                }
                else
                {
                    SLANG_UNEXPECTED("unhandled");
                }
            }

            LegalVal ordinaryVal = LegalVal();
            if (hasValidOrdinaryArgs)
                ordinaryVal = legalizeMakeArray(
                    context,
                    ordinaryType,
                    ordinaryArgs.getBuffer(),
                    ordinaryArgs.getCount(),
                    constructOp);

            LegalVal specialVal = LegalVal();
            if (hasValidSpecialArgs)
                specialVal = legalizeMakeArray(
                    context,
                    specialType,
                    specialArgs.getBuffer(),
                    specialArgs.getCount(),
                    constructOp);

            return LegalVal::pair(ordinaryVal, specialVal, pairInfo);
        }
        break;

    case LegalType::Flavor::tuple:
        {
            // For array types that are legalized as tuples,
            // we expect each element of the array to be legalized as the same tuples.
            // We want to return a tuple, where i-th element is an array containing
            // the i-th tuple-element of each legalized array-element.

            auto tupleType = legalType.getTuple();

            RefPtr<TuplePseudoVal> resTupleInfo = new TuplePseudoVal();
            UInt elementCounter = 0;
            for (auto typeElem : tupleType->elements)
            {
                auto elemKey = typeElem.key;
                UInt elementIndex = elementCounter++;
                List<LegalVal> subArray;
                for (UInt i = 0; i < argCount; i++)
                {
                    LegalVal argVal = legalArgs[i];
                    SLANG_RELEASE_ASSERT(argVal.flavor == LegalVal::Flavor::tuple);
                    auto argTuple = argVal.getTuple();
                    SLANG_RELEASE_ASSERT(
                        argTuple->elements.getCount() == tupleType->elements.getCount());
                    subArray.add(argTuple->elements[elementIndex].val);
                }

                auto legalSubArray = legalizeMakeArray(
                    context,
                    typeElem.type,
                    subArray.getBuffer(),
                    subArray.getCount(),
                    constructOp);

                TuplePseudoVal::Element resElem;
                resElem.key = elemKey;
                resElem.val = legalSubArray;
                resTupleInfo->elements.add(resElem);
            }
            return LegalVal::tuple(resTupleInfo);
        }

    default:
        SLANG_UNEXPECTED("unhandled");
        UNREACHABLE_RETURN(LegalVal());
    }
}

static LegalVal legalizeDefaultConstruct(IRTypeLegalizationContext* context, LegalType legalType)
{
    auto builder = context->builder;

    switch (legalType.flavor)
    {
    case LegalType::Flavor::none:
        return LegalVal();

    case LegalType::Flavor::simple:
        {
            return LegalVal::simple(builder->emitDefaultConstruct(legalType.getSimple()));
        }

    case LegalType::Flavor::pair:
        {
            auto pairType = legalType.getPair();
            auto pairInfo = pairType->pairInfo;
            LegalType ordinaryType = pairType->ordinaryType;
            LegalType specialType = pairType->specialType;

            LegalVal ordinaryVal = legalizeDefaultConstruct(context, ordinaryType);

            LegalVal specialVal = legalizeDefaultConstruct(context, specialType);

            return LegalVal::pair(ordinaryVal, specialVal, pairInfo);
        }
        break;

    case LegalType::Flavor::tuple:
        {
            auto tupleType = legalType.getTuple();

            RefPtr<TuplePseudoVal> resTupleInfo = new TuplePseudoVal();
            for (auto typeElem : tupleType->elements)
            {
                auto elemKey = typeElem.key;
                TuplePseudoVal::Element resElem;
                resElem.key = elemKey;
                resElem.val = legalizeDefaultConstruct(context, typeElem.type);
                resTupleInfo->elements.add(resElem);
            }
            return LegalVal::tuple(resTupleInfo);
        }

    default:
        SLANG_UNEXPECTED("unhandled");
        UNREACHABLE_RETURN(LegalVal());
    }
}

// If a legalized `val` has a different flavor than `type`, try to coerce it to `type`.
//
static LegalVal coerceToLegalType(IRTypeLegalizationContext* context, LegalType type, LegalVal val)
{
    switch (type.flavor)
    {
    case LegalType::Flavor::none:
        return LegalVal();
    case LegalType::Flavor::simple:
        {
            if (val.flavor != LegalVal::Flavor::simple)
                return val;
            auto simpleVal = val.getSimple();
            if (simpleVal->getDataType() == type.getSimple())
                return val;

            auto resultType = type.getSimple();
            auto structType = as<IRStructType>(resultType);
            if (!structType)
            {
                auto resultValueType = tryGetPointedToType(context->builder, resultType);
                if (!resultValueType)
                    return val;
                auto valValueType = tryGetPointedToType(context->builder, simpleVal->getDataType());
                if (!valValueType)
                    return val;
                if (resultValueType == valValueType)
                    return val;
                auto loadedVal = context->builder->emitLoad(val.getSimple());
                auto innerLegalVal = coerceToLegalType(
                    context,
                    LegalType::simple(resultValueType),
                    LegalVal::simple(loadedVal));
                return LegalVal::implicitDeref(innerLegalVal);
            }
            ShortList<IRInst*> fields;
            for (auto field : structType->getFields())
            {
                if (as<IRVoidType>(field->getFieldType()))
                    continue;
                auto fieldVal = coerceToLegalType(
                    context,
                    LegalType::simple(field->getFieldType()),
                    LegalVal::simple(
                        context->builder->emitFieldExtract(simpleVal, field->getKey())));
                fields.add(fieldVal.getSimple());
            }
            return LegalVal::simple(context->builder->emitMakeStruct(
                structType,
                (UInt)fields.getCount(),
                fields.getArrayView().getBuffer()));
        }
    case LegalType::Flavor::implicitDeref:
        {
            auto innerVal = val;
            if (innerVal.flavor == LegalVal::Flavor::implicitDeref)
                innerVal = innerVal.getImplicitDeref();
            else if (innerVal.flavor == LegalVal::Flavor::simple)
                innerVal = LegalVal::simple(context->builder->emitLoad(innerVal.getSimple()));
            innerVal = coerceToLegalType(context, type.getImplicitDeref()->valueType, innerVal);
            return LegalVal::implicitDeref(innerVal);
        }
    case LegalType::Flavor::pair:
        {
            if (val.flavor == LegalVal::Flavor::pair)
                return val;
            else if (val.flavor == LegalVal::Flavor::simple)
            {
                auto pairType = type.getPair();
                auto pairInfo = pairType->pairInfo;
                LegalVal ordinaryVal = coerceToLegalType(context, pairType->ordinaryType, val);
                LegalVal specialVal = coerceToLegalType(context, pairType->specialType, val);
                return LegalVal::pair(ordinaryVal, specialVal, pairInfo);
            }
            else if (val.flavor == LegalVal::Flavor::implicitDeref)
            {
                LegalVal innerVal = coerceToLegalType(context, type, val.getImplicitDeref());
                return LegalVal::implicitDeref(innerVal);
            }
            else
            {
                SLANG_UNEXPECTED("unhandled legal type coercion");
                UNREACHABLE_RETURN(LegalVal());
            }
        }
    case LegalType::Flavor::tuple:
        {
            if (val.flavor == LegalVal::Flavor::tuple)
                return val;
            else if (val.flavor == LegalVal::Flavor::simple)
            {
                auto tupleType = type.getTuple();
                RefPtr<TuplePseudoVal> tupleVal = new TuplePseudoVal();
                auto simpleVal = val.getSimple();
                for (auto elem : tupleType->elements)
                {
                    IRInst* elementVal = nullptr;
                    if (as<IRPtrTypeBase>(simpleVal->getDataType()) ||
                        as<IRPointerLikeType>(simpleVal->getDataType()))
                        elementVal = context->builder->emitFieldAddress(simpleVal, elem.key);
                    else
                        elementVal = context->builder->emitFieldExtract(simpleVal, elem.key);
                    LegalVal legalElementVal =
                        coerceToLegalType(context, elem.type, LegalVal::simple(elementVal));
                    TuplePseudoVal::Element tupleElem;
                    tupleElem.key = elem.key;
                    tupleElem.val = legalElementVal;
                    tupleVal->elements.add(tupleElem);
                }
                return LegalVal::tuple(tupleVal);
            }
            else if (val.flavor == LegalVal::Flavor::implicitDeref)
            {
                LegalVal innerVal = coerceToLegalType(context, type, val.getImplicitDeref());
                return LegalVal::implicitDeref(innerVal);
            }
            else
            {
                SLANG_UNEXPECTED("unhandled legal type coercion");
                UNREACHABLE_RETURN(LegalVal());
            }
        }
    default:
        return val;
    }
}

static LegalVal legalizeUndefined(IRTypeLegalizationContext* context, IRInst* inst)
{
    IRType* opaqueType = nullptr;
    if (isOpaqueType(inst->getFullType(), &opaqueType))
    {
        SourceLoc loc = findBestSourceLocFromUses(inst);

        if (!loc.isValid())
            loc = getDiagnosticPos(opaqueType);

        context->m_sink->diagnose(loc, Diagnostics::useOfUninitializedOpaqueHandle, opaqueType);
    }
    return LegalVal();
}

static LegalVal legalizeInst(
    IRTypeLegalizationContext* context,
    IRInst* inst,
    LegalType type,
    ArrayView<LegalVal> args)
{
    LegalVal result = LegalVal();
    switch (inst->getOp())
    {
    case kIROp_Load:
        result = legalizeLoad(context, args[0]);
        break;

    case kIROp_GetValueFromBoundInterface:
        result = args[0];
        break;

    case kIROp_FieldAddress:
        result = legalizeFieldAddress(context, type, args[0], args[1]);
        break;

    case kIROp_FieldExtract:
        result = legalizeFieldExtract(context, type, args[0], args[1]);
        break;

    case kIROp_GetElement:
        result = legalizeGetElement(context, type, args[0], args[1]);
        break;

    case kIROp_GetElementPtr:
        result = legalizeGetElementPtr(context, type, args[0], args[1]);
        break;

    case kIROp_Store:
        result = legalizeStore(context, args[0], args[1]);
        break;

    case kIROp_Call:
        result = legalizeCall(context, (IRCall*)inst);
        break;
    case kIROp_Return:
        result = legalizeRetVal(context, args[0], (IRReturn*)inst);
        break;
    case kIROp_CastDescriptorHandleToResource:
        result = LegalVal::simple(inst);
        break;
    case kIROp_DebugVar:
        result = legalizeDebugVar(context, type, (IRDebugVar*)inst);
        break;
    case kIROp_DebugValue:
        result = legalizeDebugValue(context, args[0], args[1], (IRDebugValue*)inst);
        break;

    case kIROp_MakeStruct:
        result = legalizeMakeStruct(context, type, args.getBuffer(), inst->getOperandCount());
        break;
    case kIROp_MakeArray:
    case kIROp_MakeArrayFromElement:
        result = legalizeMakeArray(
            context,
            type,
            args.getBuffer(),
            inst->getOperandCount(),
            inst->getOp());
        break;
    case kIROp_DefaultConstruct:
        result = legalizeDefaultConstruct(context, type);
        break;
    case kIROp_unconditionalBranch:
    case kIROp_loop:
        result = legalizeUnconditionalBranch(context, args, (IRUnconditionalBranch*)inst);
        break;
    case kIROp_Printf:
        result = legalizePrintf(context, args);
        break;
    case kIROp_undefined:
        return legalizeUndefined(context, inst);
    case kIROp_GpuForeach:
        // This case should only happen when compiling for a target that does not support
        // GpuForeach
        return LegalVal();
    case kIROp_StructuredBufferLoad:
        // empty types are removed, so we need to make sure that we're still
        // loading a none type when we try and load from a to-be-optimized
        // out structured buffer
        SLANG_ASSERT(type.flavor == LegalType::Flavor::none);
        return LegalVal();
    default:
        // TODO: produce a user-visible diagnostic here
        SLANG_UNEXPECTED("non-simple operand(s)!");
        break;
    }
    result = coerceToLegalType(context, type, result);
    return result;
}

static UnownedStringSlice findNameHint(IRInst* inst)
{
    if (auto nameHintDecoration = inst->findDecoration<IRNameHintDecoration>())
    {
        return nameHintDecoration->getName();
    }
    return UnownedStringSlice();
}

static LegalVal legalizeLocalVar(IRTypeLegalizationContext* context, IRVar* irLocalVar)
{
    // Legalize the type for the variable's value
    auto originalValueType = irLocalVar->getDataType()->getValueType();
    auto legalValueType = legalizeType(context, originalValueType);

    auto originalRate = irLocalVar->getRate();

    IRVarLayout* varLayout = findVarLayout(irLocalVar);
    IRTypeLayout* typeLayout = varLayout ? varLayout->getTypeLayout() : nullptr;

    // If we've decided to do implicit deref on the type,
    // then go ahead and declare a value of the pointed-to type.
    LegalType maybeSimpleType = legalValueType;
    while (maybeSimpleType.flavor == LegalType::Flavor::implicitDeref)
    {
        maybeSimpleType = maybeSimpleType.getImplicitDeref()->valueType;
    }

    switch (maybeSimpleType.flavor)
    {
    case LegalType::Flavor::simple:
        {
            // Easy case: the type is usable as-is, and we
            // should just do that.
            auto type = maybeSimpleType.getSimple();
            type = context->builder->getPtrTypeWithAddressSpace(type, irLocalVar->getDataType());
            if (originalRate)
            {
                type = context->builder->getRateQualifiedType(originalRate, type);
            }
            irLocalVar->setFullType(type);
            return LegalVal::simple(irLocalVar);
        }

    default:
        {
            // TODO: We don't handle rates in this path.

            context->insertBeforeLocalVar = irLocalVar;

            LegalVarChainLink varChain(LegalVarChain(), varLayout);

            UnownedStringSlice nameHint = findNameHint(irLocalVar);
            context->builder->setInsertBefore(irLocalVar);
            LegalVal newVal = declareVars(
                context,
                kIROp_Var,
                legalValueType,
                typeLayout,
                varChain,
                nameHint,
                irLocalVar,
                nullptr,
                context->isSpecialType(originalValueType));

            // Remove the old local var.
            irLocalVar->removeFromParent();
            // add old local var to list
            context->replacedInstructions.add(irLocalVar);
            return newVal;
        }
    }
    UNREACHABLE_RETURN(LegalVal());
}

static LegalVal legalizeParam(IRTypeLegalizationContext* context, IRParam* originalParam)
{
    auto legalParamType = legalizeType(context, originalParam->getFullType());
    if (legalParamType.flavor == LegalType::Flavor::simple)
    {
        // Simple case: things were legalized to a simple type,
        // so we can just use the original parameter as-is.
        originalParam->setFullType(legalParamType.getSimple());
        return LegalVal::simple(originalParam);
    }
    else
    {
        // Complex case: we need to insert zero or more new parameters,
        // which will replace the old ones.

        context->insertBeforeParam = originalParam;

        UnownedStringSlice nameHint = findNameHint(originalParam);

        context->builder->setInsertBefore(originalParam);
        auto newVal = declareVars(
            context,
            kIROp_Param,
            legalParamType,
            nullptr,
            LegalVarChain(),
            nameHint,
            originalParam,
            nullptr,
            context->isSpecialType(originalParam->getDataType()));

        originalParam->removeFromParent();
        context->replacedInstructions.add(originalParam);
        return newVal;
    }
}

static LegalVal legalizeFunc(IRTypeLegalizationContext* context, IRFunc* irFunc);

static LegalVal legalizeGlobalVar(IRTypeLegalizationContext* context, IRGlobalVar* irGlobalVar);

static LegalVal legalizeGlobalParam(
    IRTypeLegalizationContext* context,
    IRGlobalParam* irGlobalParam);

static LegalVal legalizeInst(IRTypeLegalizationContext* context, IRInst* inst)
{
    // Any additional instructions we need to emit
    // in the process of legalizing `inst` should
    // by default be insertied right before `inst`.
    //
    context->builder->setInsertBefore(inst);

    // Special-case certain operations
    switch (inst->getOp())
    {
    case kIROp_Var:
        return legalizeLocalVar(context, cast<IRVar>(inst));

    case kIROp_Param:
        return legalizeParam(context, cast<IRParam>(inst));

    case kIROp_WitnessTable:
        // Just skip these.
        break;

    case kIROp_Func:
        return legalizeFunc(context, cast<IRFunc>(inst));

    case kIROp_GlobalVar:
        return legalizeGlobalVar(context, cast<IRGlobalVar>(inst));

    case kIROp_GlobalParam:
        return legalizeGlobalParam(context, cast<IRGlobalParam>(inst));

    case kIROp_Block:
        return LegalVal::simple(inst);

    default:
        break;
    }

    if (as<IRAttr>(inst))
        return LegalVal::simple(inst);


    // We will iterate over all the operands, extract the legalized
    // value of each, and collect them in an array for subsequent use.
    //
    auto argCount = inst->getOperandCount();
    ShortList<LegalVal> legalArgs;
    //
    // Along the way we will also note whether there were any operands
    // with non-simple legalized values.
    //
    bool anyComplex = false;
    for (UInt aa = 0; aa < argCount; ++aa)
    {
        auto oldArg = inst->getOperand(aa);
        auto legalArg = legalizeOperand(context, oldArg);
        legalArgs.add(legalArg);

        if (legalArg.flavor != LegalVal::Flavor::simple)
            anyComplex = true;
    }

    // We must also legalize the type of the instruction, since that
    // is implicitly one of its operands.
    //
    LegalType legalType = legalizeType(context, inst->getFullType());

    // If there was nothing interesting that occured for the operands
    // then we can re-use this instruction as-is.
    //
    if (!anyComplex && legalType.flavor == LegalType::Flavor::simple)
    {
        // While the operands are all "simple," they might not necessarily
        // be equal to the operands we started with.
        //
        ShortList<IRInst*> newArgs;
        newArgs.setCount(argCount);
        bool recreate = false;
        for (UInt aa = 0; aa < argCount; ++aa)
        {
            auto legalArg = legalArgs[aa];
            newArgs[aa] = legalArg.getSimple();
            if (newArgs[aa] != inst->getOperand(aa))
                recreate = true;
        }
        if (inst->getFullType() != legalType.getSimple())
            recreate = true;
        if (recreate)
        {
            IRBuilder builder(inst->getModule());
            builder.setInsertBefore(inst);
            auto newInst = builder.emitIntrinsicInst(
                legalType.getSimple(),
                inst->getOp(),
                argCount,
                newArgs.getArrayView().getBuffer());
            inst->replaceUsesWith(newInst);
            inst->removeFromParent();
            context->replacedInstructions.add(inst);
            for (auto child : inst->getDecorationsAndChildren())
            {
                child->insertAtEnd(newInst);
            }
            return LegalVal::simple(newInst);
        }
        return LegalVal::simple(inst);
    }

    // We have at least one "complex" operand, and we
    // need to figure out what to do with it. The anwer
    // will, in general, depend on what we are doing.

    // We will set up the IR builder so that any new
    // instructions generated will be placed before
    // the location of the original instruction.
    auto builder = context->builder;
    builder->setInsertBefore(inst);

    LegalVal legalVal = legalizeInst(context, inst, legalType, legalArgs.getArrayView().arrayView);

    if (legalVal.flavor == LegalVal::Flavor::simple)
    {
        inst->replaceUsesWith(legalVal.getSimple());
    }
    // After we are done, we will eliminate the
    // original instruction by removing it from
    // the IR.
    //
    inst->removeFromParent();
    context->replacedInstructions.add(inst);

    // The value to be used when referencing
    // the original instruction will now be
    // whatever value(s) we created to replace it.
    return legalVal;
}

/// Helper type for legalizing the signature of an `IRFunc`
struct LegalFuncBuilder
{
    LegalFuncBuilder(IRTypeLegalizationContext* context)
        : m_context(context)
    {
    }

    /// Construct a legalized value to represent `oldFunc`
    LegalVal build(IRFunc* oldFunc)
    {
        // We can start by computing what the type signature of the
        // legalized function should be, based on the type signature
        // of the original.
        //
        IRFuncType* oldFuncType = oldFunc->getDataType();

        // Each parameter of the original function will translate into
        // zero or more parameters in the legalized function signature.
        //
        UInt oldParamCount = oldFuncType->getParamCount();
        for (UInt pp = 0; pp < oldParamCount; ++pp)
        {
            auto legalParamType = legalizeType(m_context, oldFuncType->getParamType(pp));
            _addParam(legalParamType);
        }

        // We will record how many parameters resulted from
        // legalization of the original / "base" parameter list.
        // This number will help us in computing how many parameters
        // were added to capture the result type of the function.
        //
        Index baseLegalParamCount = m_paramTypes.getCount();

        // Next we add a result type to the function based on the
        // legalized result type of the original function.
        //
        // It is possible that this process will had one or more
        // `out` parameters to represent parts of the result type
        // that couldn't be passed via the ordinary function result.
        //
        auto legalResultType = legalizeType(m_context, oldFuncType->getResultType());
        _addResult(legalResultType);

        // If any part of the result type required new function parameters
        // to be introduced, then we want to know how many there were.
        // These additional function paameters will always come after the original
        // parameters, so that they don't shift around call sites too much.
        //
        // TODO: Where we put the added `out` parameters in the signature may
        // have performance implications when it starts interacting with ABI
        // (e.g., most ABIs assign parameters to registers from left to right,
        // so parameters later in the list are more likely to be passed through
        // memory; we'd need to decide whether the base parameters or the
        // legalized result parameters should be prioritized for register
        // allocation).
        //
        Index resultParamCount = m_paramTypes.getCount() - baseLegalParamCount;

        // If we didn't bottom out on a result type for the legalized function,
        // then we should default to returning `void`.
        //
        auto irBuilder = m_context->builder;
        if (!m_resultType)
        {
            m_resultType = irBuilder->getVoidType();
        }

        // We will compute the new IR type for the function and install it
        // as the data type of original function.
        //
        // Note: This is one of the few cases where the legalization pass
        // prefers to modify an IR node in-place rather than create a distinct
        // legalized copy of it.
        //
        irBuilder->setInsertBefore(oldFunc);
        auto newFuncType =
            irBuilder->getFuncType(m_paramTypes.getCount(), m_paramTypes.getBuffer(), m_resultType);
        irBuilder->setDataType(oldFunc, newFuncType);

        // If the function required any new parameters to be created
        // to represent the result/return type, then we need to
        // actually add the appropriate IR parameters to represent
        // that stuff as well.
        //
        if (resultParamCount != 0)
        {
            // Only a function with a body will need this additonal
            // step, since the function parameters are stored on the
            // first block of the body.
            //
            auto firstBlock = oldFunc->getFirstBlock();
            if (firstBlock)
            {
                // Because legalization of this function required us
                // to introduce new parameters, we need to allocate
                // a data structure to record the identities of those
                // new parameters so that they can be looked up when
                // legalizing the body of the function.
                //
                // In particular, we will use this information when
                // legalizing `return` instructions in the function body,
                // since those will need to store at least part of
                // the reuslt value into the newly-declared parameter(s).
                //
                RefPtr<LegalFuncInfo> funcInfo = new LegalFuncInfo();
                m_context->mapFuncToInfo.add(oldFunc, funcInfo);

                // We know that our new parameters need to come after
                // those that were declared for the "base" parameters
                // of the original function.
                //
                auto firstResultParamIndex = baseLegalParamCount;
                auto firstOrdinaryInst = firstBlock->getFirstOrdinaryInst();
                for (Index i = 0; i < resultParamCount; ++i)
                {
                    // Note: The parameter types that were added to
                    // the `m_paramTypes` array already account for the
                    // fact that these are `out` parameters, since that
                    // impacts the function type signature as well.
                    // We do *not* need to wrap `paramType` in an `Out<...>`
                    // type here.
                    //
                    auto paramType = m_paramTypes[firstResultParamIndex + i];
                    auto param = irBuilder->createParam(paramType);
                    param->insertBefore(firstOrdinaryInst);

                    funcInfo->resultParamVals.add(param);
                }
            }
        }

        // Note: at this point we do *not* apply legalization to the parameters
        // of the function or its body; those are left for the recursive part
        // of the overall legalization pass to handle.

        return LegalVal::simple(oldFunc);
    }


private:
    IRTypeLegalizationContext* m_context = nullptr;
    ;

    /// The types of the parameters of the legalized function
    List<IRType*> m_paramTypes;

    /// The result type of the legalized function (can be null to represent `void`)
    IRType* m_resultType = nullptr;

    /// Add a parameter of type `t` to the function signature
    void _addParam(LegalType t)
    {
        // This logic is a simple recursion over the structure of `t`,
        // with the leaf case adding parameters of simple IR type.

        switch (t.flavor)
        {
        case LegalType::Flavor::none:
            break;

        case LegalType::Flavor::simple:
            m_paramTypes.add(t.getSimple());
            break;

        case LegalType::Flavor::implicitDeref:
            {
                auto imp = t.getImplicitDeref();
                _addParam(imp->valueType);
            }
            break;
        case LegalType::Flavor::pair:
            {
                auto pairInfo = t.getPair();
                _addParam(pairInfo->ordinaryType);
                _addParam(pairInfo->specialType);
            }
            break;
        case LegalType::Flavor::tuple:
            {
                auto tup = t.getTuple();
                for (auto& elem : tup->elements)
                    _addParam(elem.type);
            }
            break;
        default:
            SLANG_UNEXPECTED("unknown legalized type flavor");
        }
    }

    /// Set the logical result type of the legalized function to `t`
    void _addResult(LegalType t)
    {
        switch (t.flavor)
        {
        case LegalType::Flavor::simple:
            // The simple case is when the result type is a simple IR
            // type, and we can use it directly as the return type.
            //
            m_resultType = t.getSimple();
            break;


        case LegalType::Flavor::none:
            // The case where we have no result type is also simple,
            // becaues we can leave `m_resultType` as null to represent
            // a `void` result type.
            break;

        case LegalType::Flavor::implicitDeref:
            {
                // An `implicitDeref` is a wrapper around another legal
                // type, so we can simply set the result type to the
                // unwrapped inner type.
                //
                auto imp = t.getImplicitDeref();
                _addResult(imp->valueType);
            }
            break;

        case LegalType::Flavor::pair:
            {
                // The `pair` case is the first interesting one.
                //
                // We will set the actual result type of the operation
                // to the ordinary side of the pair, while any special
                // part of the pair will be returned via fresh `out`
                // parameters insteqad.
                //
                auto pairInfo = t.getPair();
                _addResult(pairInfo->ordinaryType);
                _addOutParam(pairInfo->specialType);
            }
            break;

        case LegalType::Flavor::tuple:
            {
                // In the `tuple` case we have zero or more types,
                // and there is no distinguished primary one that
                // should become the result type of the legalized function.
                //
                // We will instead declare fresh `out` parameters to
                // capture all the outputs in the tuple.
                //
                auto tup = t.getTuple();
                for (auto& elem : tup->elements)
                {
                    _addOutParam(elem.type);
                }
            }
            break;

        default:
            SLANG_UNEXPECTED("unknown legalized type flavor");
        }
    }

    /// Add a single `out` parameter based on type `t`.
    void _addOutParam(LegalType t)
    {
        switch (t.flavor)
        {
        case LegalType::Flavor::simple:
            // The simple case here is almost the same as `_addParam()`,
            // except we wrap the simple type in `Out<...>` to indicate
            // that we are producing an `out` parameter.
            //
            m_paramTypes.add(m_context->builder->getOutType(t.getSimple()));
            break;

            // The remaining cases are all simple recursion on the
            // structure of `t`.

        case LegalType::Flavor::none:
            break;

        case LegalType::Flavor::implicitDeref:
            {
                auto imp = t.getImplicitDeref();
                _addOutParam(imp->valueType);
            }
            break;
        case LegalType::Flavor::pair:
            {
                auto pairInfo = t.getPair();
                _addOutParam(pairInfo->ordinaryType);
                _addOutParam(pairInfo->specialType);
            }
            break;
        case LegalType::Flavor::tuple:
            {
                auto tup = t.getTuple();
                for (auto& elem : tup->elements)
                {
                    _addOutParam(elem.type);
                }
            }
            break;
        default:
            SLANG_UNEXPECTED("unknown legalized type flavor");
        }
    }
};

static LegalVal legalizeFunc(IRTypeLegalizationContext* context, IRFunc* irFunc)
{
    LegalFuncBuilder builder(context);
    return builder.build(irFunc);
}

static void cloneDecorationToVar(IRInst* srcInst, IRInst* varInst)
{
    for (auto decoration : srcInst->getDecorations())
    {
        switch (decoration->getOp())
        {
        case kIROp_FormatDecoration:
        case kIROp_UserTypeNameDecoration:
        case kIROp_SemanticDecoration:
        case kIROp_MemoryQualifierSetDecoration:
            cloneDecoration(decoration, varInst);
            break;

        default:
            break;
        }
    }
}

static LegalVal declareSimpleVar(
    IRTypeLegalizationContext* context,
    IROp op,
    IRType* type,
    IRTypeLayout* typeLayout,
    LegalVarChain const& varChain,
    UnownedStringSlice nameHint,
    IRInst* leafVar,
    IRGlobalParamInfo* globalParamInfo)
{
    IRVarLayout* varLayout = createVarLayout(context->builder, varChain, typeLayout);

    IRBuilder* builder = context->builder;

    IRInst* irVar = nullptr;
    LegalVal legalVarVal;

    switch (op)
    {
    case kIROp_GlobalVar:
        {
            auto globalVar = builder->createGlobalVar(type);
            globalVar->removeFromParent();
            globalVar->insertBefore(context->insertBeforeGlobal);

            irVar = globalVar;
            legalVarVal = LegalVal::simple(irVar);
        }
        break;

    case kIROp_GlobalParam:
        {
            auto globalParam = builder->createGlobalParam(type);
            globalParam->removeFromParent();
            globalParam->insertBefore(context->insertBeforeGlobal);

            // Add originating entry point decoration if original global param
            // comes from an entry point parameter. This is required in cases where the global
            // param has to be linked back to the originating entry point, such as when
            // emitting Metal where there global params have to be moved back to the
            // entry point parameter.
            SLANG_ASSERT(globalParamInfo);
            if (globalParamInfo->originatingEntryPoint)
            {
                builder->addEntryPointParamDecoration(
                    globalParam,
                    globalParamInfo->originatingEntryPoint);
            }

            irVar = globalParam;
            legalVarVal = LegalVal::simple(globalParam);
        }
        break;

    case kIROp_Var:
        {
            builder->setInsertBefore(context->insertBeforeLocalVar);
            auto localVar = builder->emitVar(type);

            irVar = localVar;
            legalVarVal = LegalVal::simple(irVar);
        }
        break;

    case kIROp_Param:
        {
            auto param = builder->emitParam(type);
            param->insertBefore(context->insertBeforeParam);

            irVar = param;
            legalVarVal = LegalVal::simple(irVar);
        }
        break;

    default:
        SLANG_UNEXPECTED("unexpected IR opcode");
        break;
    }

    if (irVar)
    {
        if (varLayout)
        {
            builder->addLayoutDecoration(irVar, varLayout);
        }

        if (nameHint.getLength())
        {
            context->builder->addNameHintDecoration(irVar, nameHint);
        }

        if (leafVar)
        {
            cloneDecorationToVar(leafVar, irVar);
            if (as<IRStructKey>(leafVar))
            {
                // Find the struct field and clone any decorations on the field over.
                for (auto use = leafVar->firstUse; use; use = use->nextUse)
                {
                    if (auto field = as<IRStructField>(use->getUser()))
                    {
                        cloneDecorationToVar(field, irVar);
                        break;
                    }
                }
            }
        }
    }

    return legalVarVal;
}

/// Add layout information for the fields of a wrapped buffer type.
///
/// A wrapped buffer type encodes a buffer like `ConstantBuffer<Foo>`
/// where `Foo` might have interface-type fields that have been
/// specialized to a concrete type. E.g.:
///
///     struct Car { IDriver driver; int mph; };
///     ConstantBuffer<Car> machOne;
///
/// In a case where the `machOne.driver` field has been specialized
/// to the type `SpeedRacer`, we need to generate a legalized
/// buffer layout something like:
///
///     struct Car_0 { int mph; }
///     struct Wrapped { Car_0 car; SpeedRacer card_d; }
///     ConstantBuffer<Wrapped> machOne;
///
/// The layout information for the existing `machOne` clearly
/// can't apply because we have a new element type with new fields.
///
/// This function is used to recursively fill in the layout for
/// the fields of the `Wrapped` type, using information recorded
/// when the legal wrapped buffer type was created.
///
static void _addFieldsToWrappedBufferElementTypeLayout(
    IRBuilder* irBuilder,
    IRTypeLayout* elementTypeLayout,            // layout of the original field type
    IRStructTypeLayout::Builder* newTypeLayout, // layout we are filling in
    LegalElementWrapping const& elementInfo,    // information on how the original type got wrapped
    LegalVarChain const& varChain,              // chain of variables that is leading to this field
    bool isSpecial) // should we assume a leaf field is a special (interface) type?
{
    // The way we handle things depends primary on the
    // `elementInfo`, because that tells us how things
    // were wrapped up when the type was legalized.

    switch (elementInfo.flavor)
    {
    case LegalElementWrapping::Flavor::none:
        // A leaf `none` value meant there was nothing
        // to encode for a particular field (probably
        // had a `void` or empty structure type).
        break;

    case LegalElementWrapping::Flavor::simple:
        {
            auto simpleInfo = elementInfo.getSimple();

            // A `simple` wrapping means we hit a leaf
            // field that can be encoded directly.
            // What we do here depends on whether we've
            // reached an ordinary field of the original
            // data type, or if we've reached a leaf
            // field of interface type.
            //
            // We've been tracking a `varChain` that
            // remembers all the parent `struct` fields
            // we've navigated through to get here, and
            // that information has been tracking two
            // different pieces of layout:
            //
            // * The "primary" layout represents the storage
            // of the buffer element type as we usually
            // think of its (e.g., the bytes starting at offset zero).
            //
            // * The "pending" layout tells us where all the
            // fields representing concrete types plugged in
            // for interface-type slots got placed.
            //
            // We have tunneled down info to tell us which case
            // we should use (`isSpecial`).
            //
            // Most of the logic is the same between the two
            // cases. We will be computing layout information
            // for a field of the new/wrapped buffer element type.
            //
            IRVarLayout* newFieldLayout = nullptr;
            if (isSpecial)
            {
                // In the special case, that field will be laid out
                // based on the "pending" var chain, and the type
                // of the pending data for the element.
                //
                newFieldLayout = createSimpleVarLayout(
                    irBuilder,
                    varChain.pendingChain,
                    elementTypeLayout->getPendingDataTypeLayout());
            }
            else
            {
                // The ordinary case just uses the primary layout
                // information and the primary/nominal type of
                // the field.
                //
                newFieldLayout =
                    createSimpleVarLayout(irBuilder, varChain.primaryChain, elementTypeLayout);
            }

            // Either way, we add the new field to the struct type
            // layout we are building, and also update the mapping
            // information so that we can find the field layout
            // based on the IR key for the struct field.
            //
            newTypeLayout->addField(simpleInfo->key, newFieldLayout);
        }
        break;

    case LegalElementWrapping::Flavor::implicitDeref:
        {
            // This is the case where a field in the element type
            // has been legalized from `SomePtrLikeType<T>` to
            // `T`, so there is a different in levels of indirection.
            //
            // We need to recurse and see how the type `T`
            // got laid out to know what field(s) it might comprise.
            //
            auto implicitDerefInfo = elementInfo.getImplicitDeref();
            _addFieldsToWrappedBufferElementTypeLayout(
                irBuilder,
                elementTypeLayout,
                newTypeLayout,
                implicitDerefInfo->field,
                varChain,
                isSpecial);
            return;
        }
        break;

    case LegalElementWrapping::Flavor::pair:
        {
            // The pair case is the first main workhorse where
            // if we had a type that mixed ordinary and interface-type
            // fields, it would get split into an ordinary part
            // and a "special" part, each of which might comprise
            // zero or more fields.
            //
            // Here we recurse on both the ordinary and special
            // sides, and the only interesting tidbit is that
            // we pass along appropriate values for the `isSpecial`
            // flag so that we act appropriately upon running
            // into a leaf field.
            //
            auto pairElementInfo = elementInfo.getPair();
            _addFieldsToWrappedBufferElementTypeLayout(
                irBuilder,
                elementTypeLayout,
                newTypeLayout,
                pairElementInfo->ordinary,
                varChain,
                false);
            _addFieldsToWrappedBufferElementTypeLayout(
                irBuilder,
                elementTypeLayout,
                newTypeLayout,
                pairElementInfo->special,
                varChain,
                true);
        }
        break;

    case LegalElementWrapping::Flavor::tuple:
        {
            auto tupleInfo = elementInfo.getTuple();

            // There is an extremely special case that we need to deal with here,
            // which is the case where the original element/field had an interface
            // type, which was subject to static specialization.
            //
            // In such a case, the layout will show a simple type layout for
            // the field/element itself (just uniform data) plus a "pending"
            // layout for any fields related to the static specialization.
            //
            // In contrast, the actual IR type structure will have been turned
            // into a `struct` type with multiple fields, one of which is a
            // pseudo-pointer to the "pending" data. That field would require
            // legalization, sending us down this path.
            //
            // The situation here is that we have an `elementTypeLayout` that
            // is for a single field of interface type, but an `elementInfo`
            // that corresponds to a struct with 3 or more fields (the tuple
            // that was introduced to represent the interface type).
            //
            // We expect that `elementInfo` will represent a tuple with
            // only a single element, and that element will reference the third
            // field of the tuple/struct (the payload).
            //
            // What we want to do in this case is instead add the fields
            // corresponding to the payload type, which are stored as
            // the pending type layout on `elementTypeLayout`.
            //
            if (isSpecial)
            {
                if (auto existentialTypeLayout = as<IRExistentialTypeLayout>(elementTypeLayout))
                {
                    if (const auto pendingTypeLayout =
                            existentialTypeLayout->getPendingDataTypeLayout())
                    {
                        SLANG_ASSERT(tupleInfo->elements.getCount() == 1);

                        for (auto ee : tupleInfo->elements)
                        {
                            _addFieldsToWrappedBufferElementTypeLayout(
                                irBuilder,
                                existentialTypeLayout,
                                newTypeLayout,
                                ee.field,
                                varChain,
                                true);
                        }

                        return;
                    }
                }
            }

            // A tuple comes up when we've turned an aggregate
            // with one or more interface-type fields into
            // distinct fields at the top level.
            //
            // For the most part we just recurse on each field,
            // but note that we set the `isSpecial` flag on
            // the recursive calls, since we never use tuples
            // to store anything that isn't special.

            for (auto ee : tupleInfo->elements)
            {
                auto oldFieldLayout = getFieldLayout(elementTypeLayout, ee.key);
                SLANG_ASSERT(oldFieldLayout);

                LegalVarChainLink fieldChain(varChain, oldFieldLayout);

                _addFieldsToWrappedBufferElementTypeLayout(
                    irBuilder,
                    oldFieldLayout->getTypeLayout(),
                    newTypeLayout,
                    ee.field,
                    fieldChain,
                    true);
            }
        }
        break;

    default:
        SLANG_UNEXPECTED("unhandled element wrapping flavor");
        break;
    }
}

/// Add offset information for `kind` to `resultVarLayout`,
/// if it doesn't already exist, and adjust the offset so
/// that it will represent an offset relative to the
/// "primary" data for the surrounding type, rather than
/// being relative to the "pending" data.
///
static void _addOffsetVarLayoutEntry(
    IRVarLayout::Builder* resultVarLayout,
    LegalVarChain const& varChain,
    LayoutResourceKind kind)
{
    // If the target already has an offset for this kind, bail out.
    //
    if (resultVarLayout->usesResourceKind(kind))
        return;

    // Add the `ResourceInfo` that will represent the offset for
    // this resource kind (it will be initialized to zero by default)
    //
    auto resultResInfo = resultVarLayout->findOrAddResourceInfo(kind);

    // Add in any contributions from the "pending" var chain, since
    // that chain of offsets will accumulate to get the leaf offset
    // within the pending data, which in this case we assume amounts
    // to an *absolute* offset.
    //
    for (auto vv = varChain.pendingChain; vv; vv = vv->next)
    {
        if (auto chainResInfo = vv->varLayout->findOffsetAttr(kind))
        {
            resultResInfo->offset += chainResInfo->getOffset();
            resultResInfo->space += chainResInfo->getSpace();
        }
    }

    // Subtract any contributions from the primary var chain, since
    // we want the resulting offset to be relative to the same
    // base as that chain.
    //
    for (auto vv = varChain.primaryChain; vv; vv = vv->next)
    {
        if (auto chainResInfo = vv->varLayout->findOffsetAttr(kind))
        {
            resultResInfo->offset -= chainResInfo->getOffset();
            resultResInfo->space -= chainResInfo->getSpace();
        }
    }
}

/// Create a variable layout for an field with "pending" type.
///
/// The given `typeLayout` should represent the type of a field
/// that is being stored in "pending" data, but that now needs
/// to be made relative to the "primary" data, because we are
/// legalizing the pending data out of the code.
///
static IRVarLayout* _createOffsetVarLayout(
    IRBuilder* irBuilder,
    LegalVarChain const& varChain,
    IRTypeLayout* typeLayout)
{
    IRVarLayout::Builder resultVarLayoutBuilder(irBuilder, typeLayout);

    // For every resource kind the type consumes, we will
    // compute an adjusted offset for the variable that
    // encodes the (absolute) offset of the pending data
    // in `varChain` relative to its primary data.
    //
    for (auto resInfo : typeLayout->getSizeAttrs())
    {
        _addOffsetVarLayoutEntry(&resultVarLayoutBuilder, varChain, resInfo->getResourceKind());
    }

    return resultVarLayoutBuilder.build();
}

/// Place offset information from `srcResInfo` onto `dstLayout`,
/// offset by whatever is in `offsetVarLayout`
static void addOffsetResInfo(
    IRVarLayout::Builder* dstLayout,
    IRVarOffsetAttr* srcResInfo,
    IRVarLayout* offsetVarLayout)
{
    auto kind = srcResInfo->getResourceKind();
    auto dstResInfo = dstLayout->findOrAddResourceInfo(kind);

    dstResInfo->offset = srcResInfo->getOffset();
    dstResInfo->space = srcResInfo->getSpace();

    if (auto offsetResInfo = offsetVarLayout->findOffsetAttr(kind))
    {
        dstResInfo->offset += offsetResInfo->getOffset();
        dstResInfo->space += offsetResInfo->getSpace();
    }
}

/// Create layout information for a wrapped buffer type.
///
/// A wrapped buffer type encodes a buffer like `ConstantBuffer<Foo>`
/// where `Foo` might have interface-type fields that have been
/// specialized to a concrete type.
///
/// Consider:
///
///     struct Car { IDriver driver; int mph; };
///     ConstantBuffer<Car> machOne;
///
/// In a case where the `machOne.driver` field has been specialized
/// to the type `SpeedRacer`, we need to generate a legalized
/// buffer layout something like:
///
///     struct Car_0 { int mph; }
///     struct Wrapped { Car_0 car; SpeedRacer card_d; }
///     ConstantBuffer<Wrapped> machOne;
///
/// The layout information for the existing `machOne` clearly
/// can't apply because we have a new element type with new fields.
///
/// This function is used to create a layout for a legalized
/// buffer type that requires wrapping, based on the original
/// type layout information and the variable layout information
/// of the surrounding context (e.g., the global shader parameter
/// that has this type).
///
static IRTypeLayout* _createWrappedBufferTypeLayout(
    IRBuilder* irBuilder,
    IRTypeLayout* oldTypeLayout,
    WrappedBufferPseudoType* wrappedBufferTypeInfo,
    LegalVarChain const& outerVarChain)
{
    // We shouldn't get invoked unless there was a parameter group type,
    // so we will sanity check for that just to be sure.
    //
    auto oldParameterGroupTypeLayout = as<IRParameterGroupTypeLayout>(oldTypeLayout);
    SLANG_ASSERT(oldParameterGroupTypeLayout);
    if (!oldParameterGroupTypeLayout)
        return oldTypeLayout;

    // The original type must have been split between the direct/primary
    // data and some amount of "pending" data to deal with interface-type
    // data in the element type of the parameter group.
    //
    // The legalization step will have already flattened the data inside of
    // the group to a single `struct` type, which places the primary data first,
    // and then any pending data into additional fields.
    //
    // Our job is to compute a type layout that we can apply to that new
    // element type, and to a parameter group surrounding it, that will
    // re-create the original intention of the split layout (both primary
    // and pending data) for a type that now only has the "primary" data.
    //

    IRParameterGroupTypeLayout::Builder newTypeLayoutBuilder(irBuilder);
    newTypeLayoutBuilder.addResourceUsageFrom(oldTypeLayout);

    // Any fields in the "pending" data will have offset information
    // that is relative to the pending data for their parent, and so on.
    // We need to compute layout information that only includes primary
    // data, so any offset information that is relative to the pending data
    // needs to instead be relative to the primary data. That amounts to
    // computing the absolute offset of each pending field, and then
    // subtracting off the absolute offset of the primary data.
    //
    // We will compute the offset that needs to be added up front,
    // and store it in the form of a `VarLayout`. The offsets we need
    // can be computed from the `outerVarChain`, and we only need to
    // store offset information for resource kinds actually consumed
    // by the pending data type for the buffer as a whole (e.g., we
    // don't need to apply offsetting to uniform bytes, because
    // those don't show up in the resource usage of a constant buffer
    // itself, and so the offsets already *are* relative to the start
    // of the buffer).
    //
    auto offsetVarLayout =
        _createOffsetVarLayout(irBuilder, outerVarChain, oldTypeLayout->getPendingDataTypeLayout());
    LegalVarChainLink offsetVarChain(LegalVarChain(), offsetVarLayout);

    // We will start our construction of the pieces of the output
    // type layout by looking at the "container" type/variable.
    //
    // A parameter block or constant buffer in Slang needs to
    // distinguish between the resource usage of the thing in
    // the block/buffer, vs. the resource usage of the block/buffer
    // itself. Consider:
    //
    //      struct Material { float4 color; Texture2D tex; }
    //      ConstantBuffer<Material> gMat;
    //
    // When compiling for Vulkan, the `gMat` constant buffer needs
    // a `binding`, and the `tex` field does too, so the overall
    // resource usage of `gMat` is two bindings, but we need a
    // way to encode which of those bindings goes to `gMat.tex`
    // and which to the constant buffer for `gMat` itself.
    //
    {
        // We will start by extracting the "primary" part of the old
        // container type/var layout, and constructing new objects
        // that will represent the layout for our wrapped buffer.
        //
        auto oldPrimaryContainerVarLayout = oldParameterGroupTypeLayout->getContainerVarLayout();
        auto oldPrimaryContainerTypeLayout = oldPrimaryContainerVarLayout->getTypeLayout();

        IRTypeLayout::Builder newContainerTypeLayoutBuilder(irBuilder);
        newContainerTypeLayoutBuilder.addResourceUsageFrom(oldPrimaryContainerTypeLayout);

        if (auto oldPendingContainerVarLayout = oldPrimaryContainerVarLayout->getPendingVarLayout())
        {
            // Whatever resources were allocated for the pending data type,
            // our new combined container type needs to account for them
            // (e.g., if we didn't have a constant buffer in the primary
            // data, but one got allocated in the pending data, we need
            // to end up with type layout information that includes a
            // constnat buffer).
            //
            auto oldPendingContainerTypeLayout = oldPendingContainerVarLayout->getTypeLayout();
            newContainerTypeLayoutBuilder.addResourceUsageFrom(oldPendingContainerTypeLayout);
        }
        auto newContainerTypeLayout = newContainerTypeLayoutBuilder.build();


        IRVarLayout::Builder newContainerVarLayoutBuilder(irBuilder, newContainerTypeLayout);

        // Whatever got allocated for the primary container should get copied
        // over to the new layout (e.g., if we allocated a constant buffer
        // for `gMat` then we need to retain that information).
        //
        for (auto resInfo : oldPrimaryContainerVarLayout->getOffsetAttrs())
        {
            auto newResInfo =
                newContainerVarLayoutBuilder.findOrAddResourceInfo(resInfo->getResourceKind());
            newResInfo->offset = resInfo->getOffset();
            newResInfo->space = resInfo->getSpace();
        }

        // It is possible that a constant buffer and/or space didn't get
        // allocated for the "primary" data, but ended up being required for
        // the "pending" data (this would happen if, e.g., a constant buffer
        // didn't appear to have any uniform data in it, but then once we
        // plugged in concrete types for interface fields it did...), so
        // we need to account for that case and copy over the relevant
        // resource usage from the pending data, if there is any.
        //
        if (auto oldPendingContainerVarLayout = oldPrimaryContainerVarLayout->getPendingVarLayout())
        {
            // We also need to add offset information based on the "pending"
            // var layout, but we need to deal with the fact that this information
            // is currently stored relative to the pending var layout for the surrounding
            // context (passed in as `outerVarChain.pendingChain`), but we need it to be
            // relative to the primary layout for the surrounding context
            // (`outerVarChain.primaryChain`). This is where the `offsetVarLayout` we computed above
            // comes in handy, because it represents the value(s) we need to add to each of the
            // per-resource-kind offsets.
            //
            for (auto resInfo : oldPendingContainerVarLayout->getOffsetAttrs())
            {
                addOffsetResInfo(&newContainerVarLayoutBuilder, resInfo, offsetVarLayout);
            }
        }

        auto newContainerVarLayout = newContainerVarLayoutBuilder.build();
        newTypeLayoutBuilder.setContainerVarLayout(newContainerVarLayout);
    }

    // Now that we've dealt with the container variable, we can turn
    // our attention to the element type. This is the part that
    // actually got legalized and required us to create a "wrapped"
    // buffer type in the first place, so we know that it will
    // have both primary and "pending" parts.
    //
    // Let's start by extracting the fields we care about from
    // the original element type/var layout, and constructing
    // the objects we'll use to represent the type/var layout for
    // the new element type.
    //
    auto oldElementVarLayout = oldParameterGroupTypeLayout->getElementVarLayout();
    auto oldElementTypeLayout = oldElementVarLayout->getTypeLayout();

    // Now matter what, the element type of a wrapped buffer
    // will always have a structure type.
    //
    IRStructTypeLayout::Builder newElementTypeLayoutBuilder(irBuilder);

    // The `wrappedBufferTypeInfo` that was passed in tells
    // us how the fields of the original type got turned into
    // zero or more fields in the new element type, so we
    // need to follow its recursive structure to build
    // layout information for each of the new fields.
    //
    // We will track a "chain" of parent variables that
    // determines how we got to each leaf field, and is
    // used to add up the offsets that will be stored
    // in the new `VarLayout`s that get created.
    // We know we need to add in some offsets (usually
    // negative) to any fields that were pending data,
    // so we will account for that in the initial
    // chain of outer variables that we pass in.
    //
    LegalVarChain varChainForElementType;
    varChainForElementType.primaryChain = nullptr;
    varChainForElementType.pendingChain = offsetVarChain.primaryChain;

    _addFieldsToWrappedBufferElementTypeLayout(
        irBuilder,
        oldElementTypeLayout,
        &newElementTypeLayoutBuilder,
        wrappedBufferTypeInfo->elementInfo,
        varChainForElementType,
        true);

    auto newElementTypeLayout = newElementTypeLayoutBuilder.build();

    // A parameter group type layout holds a `VarLayout` for the element type,
    // which encodes the offset of the element type with respect to the
    // start of the parameter group as a whole (e.g., to handle the case
    // where a constant buffer needs a `binding`, and so does its
    // element type, so the offset to the first `binding` for the element
    // type is one, not zero.
    //
    LegalVarChainLink elementVarChain(
        LegalVarChain(),
        oldParameterGroupTypeLayout->getElementVarLayout());
    auto newElementVarLayout = createVarLayout(irBuilder, elementVarChain, newElementTypeLayout);

    newTypeLayoutBuilder.setElementVarLayout(newElementVarLayout);

    // For legacy/API reasons, we also need to compute a version of the
    // element type where the offset stored in the `elementVarLayout`
    // gets "baked in" to the fields of the element type.
    //
    // TODO: For IR-based layout information the offset layout should
    // not really be required, and it is only being used in a few places
    // that could in principle be refactored. We need to make sure to
    // do that cleanup eventually.
    //
    newTypeLayoutBuilder.setOffsetElementTypeLayout(
        applyOffsetToTypeLayout(irBuilder, newElementTypeLayout, newElementVarLayout));

    return newTypeLayoutBuilder.build();
}

static LegalVal declareVars(
    IRTypeLegalizationContext* context,
    IROp op,
    LegalType type,
    IRTypeLayout* inTypeLayout,
    LegalVarChain const& inVarChain,
    UnownedStringSlice nameHint,
    IRInst* leafVar,
    IRGlobalParamInfo* globalParamInfo,
    bool isSpecial)
{
    LegalVarChain varChain = inVarChain;
    IRTypeLayout* typeLayout = inTypeLayout;
    if (isSpecial)
    {
        if (varChain.pendingChain)
        {
            varChain.primaryChain = varChain.pendingChain;
            varChain.pendingChain = nullptr;
        }
        if (typeLayout)
        {
            if (auto pendingTypeLayout = typeLayout->getPendingDataTypeLayout())
            {
                typeLayout = pendingTypeLayout;
            }
        }
    }

    switch (type.flavor)
    {
    case LegalType::Flavor::none:
        return LegalVal();

    case LegalType::Flavor::simple:
        return declareSimpleVar(
            context,
            op,
            type.getSimple(),
            typeLayout,
            varChain,
            nameHint,
            leafVar,
            globalParamInfo);
        break;

    case LegalType::Flavor::implicitDeref:
        {
            // Just declare a variable of the pointed-to type,
            // since we are removing the indirection.
            auto val = declareVars(
                context,
                op,
                type.getImplicitDeref()->valueType,
                typeLayout,
                varChain,
                nameHint,
                leafVar,
                globalParamInfo,
                isSpecial);
            return LegalVal::implicitDeref(val);
        }
        break;

    case LegalType::Flavor::pair:
        {
            auto pairType = type.getPair();
            auto ordinaryVal = declareVars(
                context,
                op,
                pairType->ordinaryType,
                typeLayout,
                varChain,
                nameHint,
                leafVar,
                globalParamInfo,
                false);
            auto specialVal = declareVars(
                context,
                op,
                pairType->specialType,
                typeLayout,
                varChain,
                nameHint,
                leafVar,
                globalParamInfo,
                true);
            return LegalVal::pair(ordinaryVal, specialVal, pairType->pairInfo);
        }

    case LegalType::Flavor::tuple:
        {
            // Declare one variable for each element of the tuple
            auto tupleType = type.getTuple();

            RefPtr<TuplePseudoVal> tupleVal = new TuplePseudoVal();

            for (auto ee : tupleType->elements)
            {
                auto fieldLayout = getFieldLayout(typeLayout, ee.key);
                IRTypeLayout* fieldTypeLayout =
                    fieldLayout ? fieldLayout->getTypeLayout() : nullptr;

                // If we have a type layout coming in, we really expect to have a layout for each
                // field.
                SLANG_ASSERT(fieldLayout || !typeLayout);

                // If we are processing layout information, then
                // we need to create a new link in the chain
                // of variables that will determine offsets
                // for the eventual leaf fields...
                //
                LegalVarChainLink newVarChain(varChain, fieldLayout);

                UnownedStringSlice fieldNameHint;
                String joinedNameHintStorage;
                if (nameHint.getLength())
                {
                    if (auto fieldNameHintDecoration =
                            ee.key->findDecoration<IRNameHintDecoration>())
                    {
                        joinedNameHintStorage.append(nameHint);
                        joinedNameHintStorage.append(".");
                        joinedNameHintStorage.append(fieldNameHintDecoration->getName());

                        fieldNameHint = joinedNameHintStorage.getUnownedSlice();
                    }
                }

                LegalVal fieldVal = declareVars(
                    context,
                    op,
                    ee.type,
                    fieldTypeLayout,
                    newVarChain,
                    fieldNameHint,
                    ee.key,
                    globalParamInfo,
                    true);

                TuplePseudoVal::Element element;
                element.key = ee.key;
                element.val = fieldVal;
                tupleVal->elements.add(element);
            }

            if (tupleVal->elements.getCount() == 2 && tupleVal->elements[0].key &&
                tupleVal->elements[0].key->findDecorationImpl(kIROp_CounterBufferDecoration))
            {
                // If this is a lowered struct from a structured buffer type that has an atomic
                // counter, insert decorations to each element var to associate the element buffer
                // with the atomic buffer. This decoration is inserted to all lowered structs in the
                // slang-ir-lower-append-consume-structured-buffer pass.
                //
                if (tupleVal->elements[0].val.flavor == LegalVal::Flavor::simple &&
                    tupleVal->elements[1].val.flavor == LegalVal::Flavor::simple)
                {
                    auto simpleElementVar = tupleVal->elements[0].val.getSimple();
                    auto simpleCounterVar = tupleVal->elements[1].val.getSimple();
                    IRBuilder builder(simpleElementVar);
                    builder.addDecoration(
                        simpleElementVar,
                        kIROp_CounterBufferDecoration,
                        simpleCounterVar);
                    // Clone decorations from leafVar to both element and counter var.
                    cloneDecorationToVar(leafVar, simpleElementVar);
                    cloneDecorationToVar(leafVar, simpleCounterVar);
                }
            }

            return LegalVal::tuple(tupleVal);
        }
        break;

    case LegalType::Flavor::wrappedBuffer:
        {
            auto wrappedBuffer = type.getWrappedBuffer();

            auto wrappedTypeLayout = _createWrappedBufferTypeLayout(
                context->builder,
                typeLayout,
                wrappedBuffer,
                varChain);

            auto innerVal = declareSimpleVar(
                context,
                op,
                wrappedBuffer->simpleType,
                wrappedTypeLayout,
                varChain,
                nameHint,
                leafVar,
                globalParamInfo);

            return LegalVal::wrappedBuffer(innerVal, wrappedBuffer->elementInfo);
        }

    default:
        SLANG_UNEXPECTED("unhandled");
        UNREACHABLE_RETURN(LegalVal());
        break;
    }
}

static LegalVal legalizeGlobalVar(IRTypeLegalizationContext* context, IRGlobalVar* irGlobalVar)
{
    // Legalize the type for the variable's value
    auto originalValueType = irGlobalVar->getDataType()->getValueType();
    auto legalValueType = legalizeType(context, originalValueType);
    auto varPtrType = as<IRPtrTypeBase>(irGlobalVar->getDataType());
    switch (legalValueType.flavor)
    {
    case LegalType::Flavor::simple:
        // Easy case: the type is usable as-is, and we
        // should just do that.
        context->builder->setDataType(
            irGlobalVar,
            context->builder->getPtrType(
                legalValueType.getSimple(),
                varPtrType ? varPtrType->getAddressSpace() : AddressSpace::Global));
        return LegalVal::simple(irGlobalVar);

    default:
        {
            context->insertBeforeGlobal = irGlobalVar;

            UnownedStringSlice nameHint = findNameHint(irGlobalVar);
            context->builder->setInsertBefore(irGlobalVar);
            LegalVal newVal = declareVars(
                context,
                kIROp_GlobalVar,
                legalValueType,
                nullptr,
                LegalVarChain(),
                nameHint,
                irGlobalVar,
                nullptr,
                context->isSpecialType(originalValueType));

            // Register the new value as the replacement for the old
            registerLegalizedValue(context, irGlobalVar, newVal);

            // Remove the old global from the module.
            irGlobalVar->removeFromParent();
            context->replacedInstructions.add(irGlobalVar);

            return newVal;
        }
        break;
    }
    UNREACHABLE_RETURN(LegalVal());
}

static LegalVal legalizeGlobalParam(
    IRTypeLegalizationContext* context,
    IRGlobalParam* irGlobalParam)
{
    // Legalize the type for the variable's value
    auto legalValueType = legalizeType(context, irGlobalParam->getFullType());

    IRVarLayout* varLayout = findVarLayout(irGlobalParam);
    IRTypeLayout* typeLayout = varLayout ? varLayout->getTypeLayout() : nullptr;

    switch (legalValueType.flavor)
    {
    case LegalType::Flavor::simple:
        // Easy case: the type is usable as-is, and we
        // should just do that.
        irGlobalParam->setFullType(legalValueType.getSimple());
        return LegalVal::simple(irGlobalParam);

    default:
        {
            context->insertBeforeGlobal = irGlobalParam;

            LegalVarChainLink varChain(LegalVarChain(), varLayout);

            IRGlobalParamInfo globalParamInfo;
            if (auto entryPointParamDecoration =
                    irGlobalParam->findDecoration<IREntryPointParamDecoration>())
            {
                globalParamInfo.originatingEntryPoint = entryPointParamDecoration->getEntryPoint();
            }

            // TODO: need to handle initializer here!

            UnownedStringSlice nameHint = findNameHint(irGlobalParam);
            context->builder->setInsertBefore(irGlobalParam);
            LegalVal newVal = declareVars(
                context,
                kIROp_GlobalParam,
                legalValueType,
                typeLayout,
                varChain,
                nameHint,
                irGlobalParam,
                &globalParamInfo,
                context->isSpecialType(irGlobalParam->getDataType()));

            // Register the new value as the replacement for the old
            registerLegalizedValue(context, irGlobalParam, newVal);

            // Remove the old global from the module.
            irGlobalParam->removeFromParent();
            context->replacedInstructions.add(irGlobalParam);

            return newVal;
        }
        break;
    }
    UNREACHABLE_RETURN(LegalVal());
}

static constexpr int kHasBeenAddedOrProcessedScratchBitIndex = 0;
static constexpr int kHasBeenAddedScratchBitIndex = 1;

struct IRTypeLegalizationPass
{
    IRTypeLegalizationContext* context = nullptr;

    // The goal of this pass is to ensure that legalization has been
    // applied to each instruction in a module. We also want to
    // ensure that an insturction doesn't get processed until after
    // all of its operands have been processed.
    //
    // The basic idea will be to maintain a work list of instructions
    // that are able to be processed, and also a set to track which
    // instructions have ever been added to the work list.

    List<IRInst*> workList;

    IRTypeLegalizationPass() { workList.reserve(8192); }

    bool hasBeenAddedOrProcessed(IRInst* inst)
    {
        if (!inst)
            return true;
        return (inst->scratchData & (1 << kHasBeenAddedOrProcessedScratchBitIndex)) != 0;
    }
    void setHasBeenAddedOrProcessed(IRInst* inst)
    {
        inst->scratchData |= (1 << kHasBeenAddedOrProcessedScratchBitIndex);
    }
    bool addedToWorkList(IRInst* inst)
    {
        return (inst->scratchData & (1 << kHasBeenAddedScratchBitIndex)) != 0;
    }
    void setAddedToWorkList(IRInst* inst)
    {
        inst->scratchData |= (1 << kHasBeenAddedScratchBitIndex);
    }

    bool hasBeenAddedToWorkListOrProcessed(IRInst* inst)
    {
        if (!inst)
            return true;
        return (inst->scratchData != 0);
    }

    // We will add a simple query to check whether an instruciton
    // has been put on the work list before (or if it should be
    // treated *as if* it has been placed on the work list).
    //
    bool hasBeenAddedToWorkList(IRInst* inst)
    {
        // Sometimes we end up with instructions that have a null
        // operand (mostly as the type field of key instructions
        // like the module itself).
        //
        // We want to treat such null pointers like we would an
        // already-processed instruction.
        //
        if (!inst)
            return true;

        // HACK(tfoley): In most cases it is structurally invalid for our
        // IR to have a cycle where following the operands (or type) of
        // instructions can lead back to the original instruction.
        //
        // (Note that circular dependencies are still possible, but they
        // must generally be from *children* of an instruction back
        // to the instruction itself. E.g., an instruction in the body
        // of a function can directly or indirectly depend on that function.)
        //
        // The one key counterexample is with interface types, because their
        // requirements and the expected types of those requirements are
        // encoded as operands. A typical method on inteface `I` will have a type
        // that involves a `ThisType<I>` parameter for `this`, and that creates
        // a cycle back to `I`.
        //
        // In our type legalization pass we are going to manually break that
        // cycle by treating all `IRInterfaceRequirementEntry` instructions
        // as having already been processed, since there is no particular
        // need for us to handle them as part of legalization.
        //
        if (inst->getOp() == kIROp_InterfaceRequirementEntry)
            return true;

        return addedToWorkList(inst);
    }

    // Next we define a convenience routine for adding something to the work list.
    //
    void addToWorkList(IRInst* inst)
    {
        // We want to avoid adding anything we've already added or processed.
        //
        if (addedToWorkList(inst))
            return;
        workList.add(inst);
        setAddedToWorkList(inst);
        setHasBeenAddedOrProcessed(inst);
    }

    void processModule(IRModule* module)
    {
        initializeScratchData(module->getModuleInst());

        // In order to process an entire module, we start by adding the
        // root module insturction to our work list, and then we will
        // proceed to process instructions until the work list goes dry.

        addToWorkList(module->getModuleInst());
        while (workList.getCount() != 0)
        {
            // The order of items in the work list is signficiant;
            // later entries could depend on earlier ones. As such, we
            // cannot just do something like the `fastRemoveAt(...)`
            // operation that could potentially lead to instructions
            // being processed in a different order than they were added.
            //
            // Instead, we will make a copy of the current work list
            // at each step, and swap in an empty work list to be added
            // to with any new instructions.
            //
            List<IRInst*> workListCopy = _Move(workList);

            resetScratchDataBit(module->getModuleInst(), kHasBeenAddedScratchBitIndex);

            // Now we simply process each instruction on the copy of
            // the work list, knowing that `processInst` may add additional
            // instructions to the original work list.
            //
            for (auto inst : workListCopy)
            {
                processInst(inst);
            }
        }

        // After we are done, there might be various instructions that
        // were marked for deletion, but have not yet been cleaned up.
        //
        // We will clean up all those unnecessary instructions now.
        //
        for (auto& lv : context->replacedInstructions)
        {
#if _DEBUG
            for (auto use = lv->firstUse; use; use = use->nextUse)
            {
                auto user = use->getUser();
                if (user->getModule() == nullptr)
                    continue;
                if (as<IRType>(user))
                    continue;
                if (!context->replacedInstructions.contains(user))
                    SLANG_UNEXPECTED("replaced inst still has use.");
                if (lv->getParent())
                    SLANG_UNEXPECTED("replaced inst still in a parent.");
            }
#endif
            lv->removeAndDeallocate();
        }
    }

    void processInst(IRInst* inst)
    {
        // It is possible that an insturction we
        // encounterer during the legalization process
        // will be one that was already removed or
        // otherwise made redundant.
        //
        // We want to skip such instructions since there
        // would not be a valid location at which to
        // store their replacements.
        //
        if (!inst->getParent() && inst->getOp() != kIROp_Module)
            return;

        // The main logic for legalizing an instruction is defined
        // earlier in this file.
        //
        // Our primary task here is to legalize the instruction, and then
        // register the result of legalization as the proper value
        // for that instruction.
        //
        LegalVal legalVal = legalizeInst(context, inst);
        registerLegalizedValue(context, inst, legalVal);

        // Once the instruction has been processed, we need to consider
        // whether any other instructions might now be ready to process.
        //
        // An instruction `i` might have been blocked by `inst` if:
        //
        // * `inst` was an operand (including the type operand) of `i`, or
        // * `inst` was the parent of `i`.
        //
        // To turn those relationships around, we want to check instructions
        // `i` where:
        //
        // * `i` is a user of `inst`, or
        // * `i` is a child of `inst`.
        //
        if (legalVal.flavor == LegalVal::Flavor::simple)
        {
            // The resulting inst may be different from the one we added to the
            // worklist, so ensure that the appropriate flags are set.
            //
            setHasBeenAddedOrProcessed(legalVal.irValue);

            inst = legalVal.irValue;
        }

        for (auto use = inst->firstUse; use; use = use->nextUse)
        {
            auto user = use->getUser();
            maybeAddToWorkList(user);
        }
        for (auto child : inst->getDecorationsAndChildren())
        {
            maybeAddToWorkList(child);
        }
    }

    void maybeAddToWorkList(IRInst* inst)
    {
        // Here we have an `inst` that might be ready to go on
        // the work list, but we need to check that it would
        // be valid to put it on there.
        //
        // First, we don't want to add something if it has
        // already been added.
        //
        if (hasBeenAddedToWorkList(inst))
            return;

        // Next, we don't want to add something if its parent
        // hasn't been added already.
        //
        if (!hasBeenAddedToWorkListOrProcessed(inst->getParent()))
            return;

        // Finally, we don't want to add something if its
        // type and/or operands haven't all been added.
        //
        if (!hasBeenAddedToWorkListOrProcessed(inst->getFullType()))
            return;
        Index operandCount = (Index)inst->getOperandCount();
        for (Index i = 0; i < operandCount; ++i)
        {
            auto operand = inst->getOperand(i);
            if (!hasBeenAddedToWorkListOrProcessed(operand))
                return;
        }

        // If all those checks pass, then we are ready to
        // process `inst`, and we will add it to our work list.
        //
        addToWorkList(inst);
    }
};

static void legalizeTypes(IRTypeLegalizationContext* context)
{
    IRTypeLegalizationPass pass;
    pass.context = context;

    pass.processModule(context->module);
}

// We use the same basic type legalization machinery for both simplifying
// away resource-type fields nested in `struct`s and for shuffling around
// exisential-box fields to get the layout right.
//
// The differences between the two passes come down to some very small
// distinctions about what types each pass considers "special" (e.g.,
// resources in one case and existential boxes in the other), along
// with what they want to do when a uniform/constant buffer needs to
// be made where the element type is non-simple (that is, includes
// some fields of "special" type).
//
// The resource case is then the simpler one:
//
struct IRResourceTypeLegalizationContext : IRTypeLegalizationContext
{
    IRResourceTypeLegalizationContext(TargetProgram* target, IRModule* module, DiagnosticSink* sink)
        : IRTypeLegalizationContext(target, module, sink)
    {
    }

    bool isSpecialType(IRType* type) override
    {
        // For resource type legalization, the "special" types
        // we are working with are resource types.
        //
        return isResourceType(type);
    }

    bool isSimpleType(IRType*) override { return false; }

    LegalType createLegalUniformBufferType(
        IROp op,
        LegalType legalElementType,
        IRInst* layoutOperand) override
    {
        // The appropriate strategy for legalizing uniform buffers
        // with resources inside already exists, so we can delegate to it.
        //
        return createLegalUniformBufferTypeForResources(this, op, legalElementType, layoutOperand);
    }
};

// The case for legalizing existential box types is then similar.
//
struct IRExistentialTypeLegalizationContext : IRTypeLegalizationContext
{
    IRExistentialTypeLegalizationContext(
        TargetProgram* target,
        IRModule* module,
        DiagnosticSink* sink)
        : IRTypeLegalizationContext(target, module, sink)
    {
    }

    bool isSpecialType(IRType* inType) override
    {
        // The "special" types for our purposes are existential
        // boxes, or arrays thereof.
        //
        auto type = unwrapArray(inType);
        return as<IRPseudoPtrType>(type) != nullptr;
    }

    bool isSimpleType(IRType*) override { return false; }

    LegalType createLegalUniformBufferType(
        IROp op,
        LegalType legalElementType,
        IRInst* layoutOperand) override
    {
        // We'll delegate the logic for creating uniform buffers
        // over a mix of ordinary and existential-box types to
        // a subroutine so it can live near the resource case.
        //
        // TODO: We should eventually try to refactor this code
        // so that related functionality is grouped together.
        //
        return createLegalUniformBufferTypeForExistentials(
            this,
            op,
            legalElementType,
            layoutOperand);
    }
};

// This customization of type legalization is used to remove empty
// structs from cpp/cuda programs if the empty type isn't used in
// a public function signature.
struct IREmptyTypeLegalizationContext : IRTypeLegalizationContext
{
    IREmptyTypeLegalizationContext(TargetProgram* target, IRModule* module, DiagnosticSink* sink)
        : IRTypeLegalizationContext(target, module, sink)
    {
    }

    bool isSpecialType(IRType*) override { return false; }

    bool isSimpleType(IRType* type) override
    {
        if (isMetalTarget(targetProgram->getTargetReq()))
        {
            return false;
        }

        // If type is used as public interface, then treat it as simple.
        for (auto decor : type->getDecorations())
        {
            switch (decor->getOp())
            {
            case kIROp_LayoutDecoration:
            case kIROp_PublicDecoration:
            case kIROp_ExternCppDecoration:
            case kIROp_DllImportDecoration:
            case kIROp_DllExportDecoration:
            case kIROp_HLSLExportDecoration:
            case kIROp_BinaryInterfaceTypeDecoration:
                return true;
            }
        }
        return false;
    }

    LegalType createLegalUniformBufferType(IROp, LegalType, IRInst*) override
    {
        return LegalType();
    }

    virtual bool shouldLegalizeParameterBlockElementType() override
    {
        return isMetalTarget(targetProgram->getTargetReq());
    }
};

// The main entry points that are used when transforming IR code
// to get it ready for lower-level codegen are then simple
// wrappers around `legalizeTypes()` that pick an appropriately
// specialized context type to use to get the job done.

void legalizeResourceTypes(TargetProgram* target, IRModule* module, DiagnosticSink* sink)
{
    SLANG_PROFILE;

    IRResourceTypeLegalizationContext context(target, module, sink);
    legalizeTypes(&context);
}

void legalizeExistentialTypeLayout(TargetProgram* target, IRModule* module, DiagnosticSink* sink)
{
    SLANG_PROFILE;

    IRExistentialTypeLegalizationContext context(target, module, sink);
    legalizeTypes(&context);
}

void legalizeEmptyTypes(TargetProgram* target, IRModule* module, DiagnosticSink* sink)
{
    IREmptyTypeLegalizationContext context(target, module, sink);
    legalizeTypes(&context);
}


} // namespace Slang
