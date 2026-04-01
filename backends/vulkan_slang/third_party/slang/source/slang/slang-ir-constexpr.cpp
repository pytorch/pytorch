// slang-ir-constexpr.cpp
#include "slang-ir-constexpr.h"

#include "slang-ir-dominators.h"
#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{

struct PropagateConstExprContext
{
    IRModule* module;
    IRModule* getModule() { return module; }

    DiagnosticSink* sink;

    IRBuilder builder;

    InstWorkList workList;
    InstHashSet onWorkList;

    PropagateConstExprContext(IRModule* module)
        : module(module), workList(module), onWorkList(module)
    {
    }

    IRBuilder* getBuilder() { return &builder; }

    Session* getSession() { return module->getSession(); }

    DiagnosticSink* getSink() { return sink; }
};

bool isConstExpr(IRType* fullType)
{
    if (auto rateQualifiedType = as<IRRateQualifiedType>(fullType))
    {
        auto rate = rateQualifiedType->getRate();
        if (const auto constExprRate = as<IRConstExprRate>(rate))
            return true;
    }

    return false;
}

bool isConstExpr(IRInst* value)
{
    // Certain IR value ops are implicitly `constexpr`
    //
    // TODO: should we just go ahead and make that explicit
    // in the type system?
    switch (value->getOp())
    {
    case kIROp_IntLit:
    case kIROp_FloatLit:
    case kIROp_BoolLit:
    case kIROp_Func:
    case kIROp_StructKey:
    case kIROp_WitnessTable:
    case kIROp_Generic:
        return true;

    default:
        break;
    }

    if (isConstExpr(value->getFullType()))
        return true;

    return false;
}

bool opCanBeConstExpr(IROp op)
{
    switch (op)
    {
    case kIROp_IntLit:
    case kIROp_FloatLit:
    case kIROp_BoolLit:
    case kIROp_Param:
    case kIROp_Add:
    case kIROp_Sub:
    case kIROp_Mul:
    case kIROp_Div:
    case kIROp_IRem:
    case kIROp_FRem:
    case kIROp_Neg:
    case kIROp_Geq:
    case kIROp_Leq:
    case kIROp_Greater:
    case kIROp_Less:
    case kIROp_Neq:
    case kIROp_Eql:
    case kIROp_BitAnd:
    case kIROp_BitOr:
    case kIROp_BitXor:
    case kIROp_BitNot:
    case kIROp_Lsh:
    case kIROp_Rsh:
    case kIROp_Select:
    case kIROp_MakeVectorFromScalar:
    case kIROp_MakeVector:
    case kIROp_MakeMatrix:
    case kIROp_MakeMatrixFromScalar:
    case kIROp_MatrixReshape:
    case kIROp_MakeCoopVector:
    case kIROp_VectorReshape:
    case kIROp_CastFloatToInt:
    case kIROp_CastIntToFloat:
    case kIROp_IntCast:
    case kIROp_FloatCast:
    case kIROp_CastIntToPtr:
    case kIROp_CastPtrToInt:
    case kIROp_CastPtrToBool:
    case kIROp_PtrCast:
    case kIROp_Reinterpret:
    case kIROp_BitCast:
    case kIROp_BuiltinCast:
    case kIROp_MakeTuple:
    case kIROp_MakeDifferentialPair:
    case kIROp_MakeExistential:
    case kIROp_MakeExistentialWithRTTI:
    case kIROp_MakeOptionalNone:
    case kIROp_MakeOptionalValue:
    case kIROp_MakeResultError:
    case kIROp_MakeResultValue:
    case kIROp_MakeString:
    case kIROp_MakeUInt64:
    case kIROp_MakeArray:
    case kIROp_MakeArrayFromElement:
    case kIROp_swizzle:
    case kIROp_GetElement:
    case kIROp_FieldExtract:
    case kIROp_UpdateElement:
    case kIROp_ExtractExistentialType:
    case kIROp_ExtractExistentialValue:
    case kIROp_ExtractExistentialWitnessTable:
    case kIROp_WrapExistential:
    case kIROp_GetResultError:
    case kIROp_GetResultValue:
    case kIROp_GetOptionalValue:
    case kIROp_DifferentialPairGetDifferential:
    case kIROp_DifferentialPairGetPrimal:
    case kIROp_LookupWitness:
    case kIROp_Specialize:
        // TODO: more cases
        return true;

    default:
        return false;
    }
}

bool opCanBeConstExprByForwardPass(IRInst* value)
{
    // TODO: handle call inst here.

    if (value->getOp() == kIROp_Param)
        return false;
    return opCanBeConstExpr(value->getOp());
}

IRLoop* isLoopPhi(IRParam* param)
{
    IRBlock* bb = cast<IRBlock>(param->getParent());
    for (auto pred : bb->getPredecessors())
    {
        auto loop = as<IRLoop>(pred->getTerminator());
        if (loop)
        {
            return loop;
        }
    }
    return nullptr;
}

bool opCanBeConstExprByBackwardPass(IRInst* value)
{
    if (value->getOp() == kIROp_Param)
        return isLoopPhi(as<IRParam, IRDynamicCastBehavior::NoUnwrap>(value));
    if (opCanBeConstExpr(value->getOp()))
        return true;
    if (auto callInst = as<IRCall>(value))
    {
        return !callInst->mightHaveSideEffects();
    }
    return false;
}

void markConstExpr(PropagateConstExprContext* context, IRInst* value)
{
    Slang::markConstExpr(context->getBuilder(), value);
}

void maybeAddToWorkList(PropagateConstExprContext* context, IRInst* gv)
{
    if (!context->onWorkList.contains(gv))
    {
        context->workList.add(gv);
        context->onWorkList.add(gv);
    }
}

bool maybeMarkConstExprBackwardPass(PropagateConstExprContext* context, IRInst* value)
{
    if (isConstExpr(value))
        return false;

    if (!opCanBeConstExprByBackwardPass(value))
        return false;

    markConstExpr(context, value);

    // TODO: we should only allow function parameters to be
    // changed to be `constexpr` when we are compiling "application"
    // code, and not library code.
    // (Or eventually we'd have a rule that only non-`public` symbols
    // can have this kind of propagation applied).

    if (value->getOp() == kIROp_Param)
    {
        auto param = (IRParam*)value;
        auto block = (IRBlock*)param->parent;
        auto code = block->getParent();

        if (block == code->getFirstBlock())
        {
            // We've just changed a function parameter to
            // be `constexpr`. We need to remember that
            // fact so taht we can mark callers of this
            // function as `constexpr` themselves.

            for (auto u = code->firstUse; u; u = u->nextUse)
            {
                auto user = u->getUser();

                switch (user->getOp())
                {
                case kIROp_Call:
                    {
                        auto inst = (IRCall*)user;
                        auto caller = as<IRGlobalValueWithCode>(inst->getParent()->getParent());
                        maybeAddToWorkList(context, caller);
                    }
                    break;

                default:
                    break;
                }
            }
        }
    }

    return true;
}

// Produce an estimate on whether a loop is unrollable, by checking
// if there is at least one exit path where all the conditions along
// the control path has a constexpr condition.
bool isUnrollableLoop(IRLoop* loop)
{
    // A loop is unrollable if all exit conditions are constexpr.
    auto breakBlock = loop->getBreakBlock();
    auto func = getParentFunc(loop);
    auto domTree = loop->getModule()->findOrCreateDominatorTree(func);
    List<IRBlock*> workList;
    bool result = false;
    for (auto pred : breakBlock->getPredecessors())
    {
        workList.clear();
        workList.add(pred);
        for (Index i = 0; i < workList.getCount(); i++)
        {
            auto block = workList[i];
            if (auto ifElse = as<IRConditionalBranch>(block->getTerminator()))
            {
                if (!isConstExpr(ifElse->getCondition()))
                    return false;
            }
            else if (as<IRSwitch>(block->getTerminator()))
            {
                if (!isConstExpr(ifElse->getCondition()))
                    return false;
            }
            auto idom = domTree->getImmediateDominator(block);
            if (idom && idom != loop->getParent())
                workList.add(idom);
        }
        // We found at least one exit path that is constexpr,
        // we will regard this loop as unrollable.
        result = true;
    }
    return result;
}

// Propagate `constexpr`-ness in a forward direction, from the
// operands of an instruction to the instruction itself.
bool propagateConstExprForward(PropagateConstExprContext* context, IRGlobalValueWithCode* code)
{
    bool anyChanges = false;
    for (;;)
    {
        bool changedThisIteration = false;
        for (auto bb = code->getFirstBlock(); bb; bb = bb->getNextBlock())
        {
            for (auto ii = bb->getFirstInst(); ii; ii = ii->getNextInst())
            {
                // Instruction already `constexpr`? Then skip it.
                if (isConstExpr(ii))
                    continue;

                // Is the operation one that we can actually make be constexpr?
                if (!opCanBeConstExprByForwardPass(ii))
                    continue;

                // Are all arguments `constexpr`?
                bool allArgsConstExpr = true;
                UInt argCount = ii->getOperandCount();
                for (UInt aa = 0; aa < argCount; ++aa)
                {
                    auto arg = ii->getOperand(aa);

                    if (!isConstExpr(arg))
                    {
                        allArgsConstExpr = false;
                        break;
                    }
                }
                if (!allArgsConstExpr)
                    continue;

                // Seems like this operation can/should be made constexpr
                markConstExpr(context, ii);
                changedThisIteration = true;
            }
        }

        if (!changedThisIteration)
            return anyChanges;

        anyChanges = true;
    }
}


// Propagate `constexpr`-ness in a backward direction, from an instruction
// to its operands.
bool propagateConstExprBackward(PropagateConstExprContext* context, IRGlobalValueWithCode* code)
{
    IRBuilder builder(context->getModule());
    builder.setInsertInto(code);

    bool anyChanges = false;
    for (;;)
    {
        // Note: we are walking the list of blocks and the instructions
        // in each block in reverse order, to maximize the chances that
        // we propagate multiple changes in a each pass.
        //
        // TODO: this should probably all be done with a work list instead,
        // but that requires being able to detect instructions vs. other
        // values.

        bool changedThisIteration = false;
        for (auto bb = code->getLastBlock(); bb; bb = bb->getPrevBlock())
        {
            for (auto ii = bb->getLastInst(); ii; ii = ii->getPrevInst())
            {
                if (isConstExpr(ii))
                {
                    // If this instruction is `constexpr`, then its operands should be too.
                    UInt argCount = ii->getOperandCount();
                    for (UInt aa = 0; aa < argCount; ++aa)
                    {
                        auto arg = ii->getOperand(aa);
                        if (isConstExpr(arg))
                            continue;

                        if (!opCanBeConstExprByBackwardPass(arg))
                            continue;

                        if (maybeMarkConstExprBackwardPass(context, arg))
                        {
                            changedThisIteration = true;
                        }
                    }
                }
                else if (ii->getOp() == kIROp_Call)
                {
                    // A non-constexpr call might be calling a function with one or
                    // more constexpr parameters. We should check if we can resolve
                    // the callee for this call statically, and if so try to propagate
                    // constexpr from the parameters back to the arguments.
                    auto callInst = (IRCall*)ii;

                    UInt operandCount = callInst->getOperandCount();

                    UInt firstCallArg = 1;
                    UInt callArgCount = operandCount - firstCallArg;

                    auto callee = callInst->getOperand(0);

                    // If we are calling a generic operation, then
                    // try to follow through the `specialize` chain
                    // and find the callee.
                    //
                    // TODO: This probably shouldn't be required,
                    // since we can hopefully use the type of the
                    // callee in all cases.
                    //
                    while (auto specInst = as<IRSpecialize>(callee))
                    {
                        auto genericInst = as<IRGeneric>(specInst->getBase());
                        if (!genericInst)
                            break;

                        auto returnVal = findGenericReturnVal(genericInst);
                        if (!returnVal)
                            break;

                        callee = returnVal;
                    }

                    auto calleeFunc = as<IRFunc>(callee);
                    if (calleeFunc && isDefinition(calleeFunc))
                    {
                        // We have an IR-level function definition we are calling,
                        // and thus we can propagate `constexpr` information
                        // through its `IRParam`s.

                        auto calleeFuncType = calleeFunc->getDataType();

                        UInt callParamCount = calleeFuncType->getParamCount();
                        SLANG_RELEASE_ASSERT(callParamCount == callArgCount);

                        // If the callee has a definition, then we can read `constexpr`
                        // information off of the parameters of its first IR block.
                        if (auto calleeFirstBlock = calleeFunc->getFirstBlock())
                        {
                            UInt paramCounter = 0;
                            for (auto pp = calleeFirstBlock->getFirstParam(); pp;
                                 pp = pp->getNextParam())
                            {
                                UInt paramIndex = paramCounter++;

                                auto param = pp;
                                auto arg = callInst->getOperand(firstCallArg + paramIndex);

                                if (isConstExpr(param))
                                {
                                    if (maybeMarkConstExprBackwardPass(context, arg))
                                    {
                                        changedThisIteration = true;
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        // If we don't have a concrete callee function
                        // definition, then we need to extract the
                        // type of the callee instruction, and try to work
                        // with that.
                        //
                        // Note that this does not allow us to propagate
                        // `constexpr` information from the body of a callee
                        // back to call sites.
                        auto calleeType = callee->getDataType();
                        if (auto caleeFuncType = as<IRFuncType>(calleeType))
                        {
                            auto paramCount = caleeFuncType->getParamCount();
                            for (UInt pp = 0; pp < paramCount; ++pp)
                            {
                                auto paramType = caleeFuncType->getParamType(pp);
                                auto arg = callInst->getOperand(firstCallArg + pp);
                                if (isConstExpr(paramType))
                                {
                                    if (maybeMarkConstExprBackwardPass(context, arg))
                                    {
                                        changedThisIteration = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (bb != code->getFirstBlock())
            {
                // A parameter in anything butr the first block is
                // conceptually a phi node, which means its operands
                // are the corresponding values from the terminating
                // branch in a predecessor block.

                UInt paramCounter = 0;
                for (auto pp = bb->getFirstParam(); pp; pp = pp->getNextParam())
                {
                    UInt paramIndex = paramCounter++;

                    if (!isConstExpr(pp))
                        continue;

                    for (auto pred : bb->getPredecessors())
                    {
                        auto terminator = as<IRUnconditionalBranch>(pred->getLastInst());
                        if (!terminator)
                            continue;

                        SLANG_RELEASE_ASSERT(paramIndex < terminator->getArgCount());

                        auto operand = terminator->getArg(paramIndex);
                        if (maybeMarkConstExprBackwardPass(context, operand))
                        {
                            changedThisIteration = true;
                        }
                    }
                }
            }
        }

        if (!changedThisIteration)
            return anyChanges;

        anyChanges = true;
    }
}
// Validate use of `constexpr` within a function (in particular,
// diagnose places where a value that must be contexpr depends
// on a value that cannot be)
void validateConstExpr(PropagateConstExprContext* context, IRGlobalValueWithCode* code)
{
    for (auto bb = code->getFirstBlock(); bb; bb = bb->getNextBlock())
    {
        for (auto ii = bb->getFirstInst(); ii; ii = ii->getNextInst())
        {
            if (isConstExpr(ii))
            {
                // For an instruction that must be `constexpr`, we need
                // to ensure that its argumenst are all `constexpr`

                UInt argCount = ii->getOperandCount();
                for (UInt aa = 0; aa < argCount; ++aa)
                {
                    auto arg = ii->getOperand(aa);
                    bool shouldDiagnose = !isConstExpr(arg);
                    if (!shouldDiagnose)
                    {
                        if (auto param = as<IRParam>(arg))
                        {
                            if (IRLoop* loopInst = isLoopPhi(param))
                            {
                                // If the param is a phi node in a loop that
                                // does not depend on non-constexpr values, we
                                // can make it constexpr by force unrolling the
                                // loop, if the loop is unrollable.
                                if (isUnrollableLoop(loopInst))
                                {
                                    if (!loopInst->findDecoration<IRForceUnrollDecoration>())
                                    {
                                        context->getBuilder()->addLoopForceUnrollDecoration(
                                            loopInst,
                                            0);
                                    }
                                    continue;
                                }
                                shouldDiagnose = true;
                            }
                        }
                    }
                    if (shouldDiagnose)
                    {

                        // Diagnose the failure.

                        context->getSink()->diagnose(
                            ii->sourceLoc,
                            Diagnostics::needCompileTimeConstant);

                        break;
                    }
                }
            }
        }
    }
}

void propagateInFunc(PropagateConstExprContext* context, IRGlobalValueWithCode* code)
{
    for (;;)
    {
        bool anyChange = false;
        if (propagateConstExprForward(context, code))
        {
            anyChange = true;
        }
        if (propagateConstExprBackward(context, code))
        {
            anyChange = true;
        }
        if (!anyChange)
            break;
    }
}

void propagateConstExpr(IRModule* module, DiagnosticSink* sink)
{
    PropagateConstExprContext context(module);
    context.sink = sink;
    context.builder = IRBuilder(module);

    // We need to propagate information both forward and backward.
    //
    // In the forward direction we need to check if all of the operands
    // to an instruction are `constexpr` *and* if the operation is
    // one that can conceptually be "promoted" to the constexpr rate.
    //
    // In the backward direction, if an instruction has already been
    // marked as needing to be `constexpr`, then its operands had
    // better be too.
    //
    // The backward direction needs to be interprocedural, because
    // a parameter to a function might be `constexpr`, so that callers
    // of that function would need to be marked too. If backwards
    // propagation in any of the callers leads to some of their
    // parameters being marked constexpr, then we would need to
    // revisit their callers.

    // We will build an initial work list with all of the global values in it.

    for (auto ii : module->getGlobalInsts())
    {
        maybeAddToWorkList(&context, ii);
    }

    // We will iterate applying propagation to one global value at a time
    // until we run out.
    while (context.workList.getCount())
    {
        auto gv = context.workList[0];
        context.workList.fastRemoveAt(0);
        context.onWorkList.remove(gv);

        switch (gv->getOp())
        {
        default:
            break;

        case kIROp_Generic:
            {
                auto gen = as<IRGeneric>(gv);
                gv = as<IRFunc>(findGenericReturnVal(gen));
                if (nullptr == gv)
                    break;
            }
            [[fallthrough]];
        case kIROp_Func:
        case kIROp_GlobalVar:
            {
                IRGlobalValueWithCode* code = (IRGlobalValueWithCode*)gv;
                propagateInFunc(&context, code);
            }
            break;
        }
    }

    // Okay, we've processed all our functions and found a steady state.
    // Now we need to try and issue diagnostics for any IR values where
    // we find that they are *required* to be `constexpr`, but *cannot*
    // be, for some reason.

    for (auto ii : module->getGlobalInsts())
    {
        switch (ii->getOp())
        {
        default:
            break;

        case kIROp_Func:
        case kIROp_GlobalVar:
            {
                IRGlobalValueWithCode* code = (IRGlobalValueWithCode*)ii;
                validateConstExpr(&context, code);
            }
            break;
        }
    }
}

} // namespace Slang
