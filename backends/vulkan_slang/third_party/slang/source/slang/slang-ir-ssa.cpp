// slang-ir-ssa.cpp
#include "slang-ir-ssa.h"

#include "slang-ir-clone.h"
#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-ir-validate.h"
#include "slang-ir.h"

namespace Slang
{

// Track information on a phi node we are in
// the process of constructing.
struct PhiInfo : RefObject
{
    // The phi node will be represented as a parameter
    // to a (non-entry) basic block.
    IRParam* phi;

    // The original variable that this phi will be replacing.
    IRVar* var;

    // The operands to the phi will be stored as uses here,
    // because our IR parameters don't have operands.
    //
    // Once we've collected all the values we plan to use,
    // we will turn this into argument in predecessor blocks
    // that branch to this one.
    //
    // The order of elements in this list must match the
    // order in which the predecessor blocks get enumerated.
    List<IRUse> operands;

    // If this phi ended up being removed as trivial, then
    // this will be the value that we replaced it with.
    IRInst* replacement = nullptr;
};

// Information about a basic block that we generate/use
// during SSA construction.
struct SSABlockInfo : RefObject
{
    // Map a promotable variable to the value to
    // use for that variable
    Dictionary<IRVar*, IRInst*> valueForVar;

    // The underlying basic block.
    IRBlock* block;

    // Have we processed all the instructions in the
    // body of this block (so that we would have
    // found any stores to SSA variables)?
    bool isFilled = false;

    // Have we filled all the predecessors of
    // this block, so that we can actually perform
    // look up in them?
    bool isSealed = false;

    // An IR builder to use when we want to construct
    // stuff in the context of this block
    IRBuilder builder;

    // Phi nodes we are creating for this block.
    List<PhiInfo*> phis;

    // Arguments that this block needs to pass along
    // to the phi nodes defined by is sucessor
    List<IRInst*> successorArgs;
};

// State for constructing SSA form for a global value
// with code (usually a function).
struct ConstructSSAContext
{
    // The value that we want to rewrite into SSA form
    // (usually an IR function)
    IRGlobalValueWithCode* globalVal;

    // Variables that we've identified for promotion
    // to SSA values.
    List<IRVar*> promotableVars;

    // Information about each basic block
    Dictionary<IRBlock*, RefPtr<SSABlockInfo>> blockInfos;

    IRModule* module;

    // Instructions to remove during cleanup
    List<IRInst*> instsToRemove;

    IRBuilder builder;
    IRBuilder* getBuilder() { return &builder; }


    Dictionary<IRParam*, RefPtr<PhiInfo>> phiInfos;

    PhiInfo* getPhiInfo(IRParam* phi)
    {
        if (auto found = phiInfos.tryGetValue(phi))
            return *found;
        return nullptr;
    }
};

/// Do all uses of this instruction lead to a `load`?
///
/// Checks if all uses of `inst` are either loads,
/// or get-element-address/get-field-address operations
/// that also lead to loads.
bool allUsesLeadToLoads(IRInst* inst)
{
    for (auto u = inst->firstUse; u; u = u->nextUse)
    {
        auto user = u->getUser();
        switch (user->getOp())
        {
        default:
            return false;

        case kIROp_Load:
            break;

        case kIROp_GetElementPtr:
        case kIROp_FieldAddress:
            {
                // Sanity check: the address being used should
                // be the base-address operand, and not the field
                // key or index (this should never be a problem).
                if (u != &user->getOperands()[0])
                    return false;

                if (!allUsesLeadToLoads(user))
                    return false;
            }
            break;
        }
    }

    // If all of the uses passed our checking, then
    // we are good to go.
    return true;
}

// Is the given variable one that we can promote to SSA form?
bool isPromotableVar(ConstructSSAContext* /*context*/, IRVar* var, HashSet<IRBlock*>& knownBlocks)
{
    // We want to identify variables such that we can always
    // determine what they will contain at a point in the
    // program by directly inspecting their uses.
    //
    // The simplest possible answer would be instructions
    // that are only ever used as the operand of "full"
    // load and store instructions (loads and stores that
    // write the entire variable). This is enough to
    // promote simple scalar variables to SSA temporaries,
    // but falls apart for aggregates and arrays.
    //
    // A slightly more powerful option (which is what we
    // implement for now) is to promote variables when
    // all of the stores are "full," and all other uses
    // are in the form of a "chain" of `getElmeentAddress`
    // or `getFieldAddress` operations that terminates
    // with a load.
    //
    // An even more powerful option (which we do not yet
    // implement) would be to handle cases where there are
    // "chains" that end with stores, and to treat these
    // as partial assignments (where we can still form
    // an SSA value by creating a new temporary with just
    // one element/field different). This kind of approach
    // would be best if it is combined with scalarization,
    // so that we don't need to construct aggregate temps.
    //

    for (auto u = var->firstUse; u; u = u->nextUse)
    {
        auto user = u->getUser();
        switch (user->getOp())
        {
        default:
            // If the variable gets used by any operation
            // we can't account for directly, then it isn't
            // promotable.
            return false;

        case kIROp_Load:
            {
                // A load has only a single argument, so
                // it had better be our pointer.
                SLANG_ASSERT(u == &((IRLoad*)user)->ptr);
            }
            break;

        case kIROp_Store:
            {
                auto storeInst = (IRStore*)user;

                // We don't want to promote a variable if
                // its address gets stored into another
                // variable, so check for that case.
                if (u == &storeInst->val)
                    return false;

                // Otherwise our variable is being used
                // as the destination for the store, and
                // that is okay by us.
                SLANG_ASSERT(u == &storeInst->ptr);
            }
            break;

        case kIROp_GetElementPtr:
        case kIROp_FieldAddress:
            {
                // Sanity check: the address being used should
                // be the base-address operand, and not the field
                // key or index (this should never be a problem).
                if (u != &user->getOperands()[0])
                    return false;

                if (!allUsesLeadToLoads(user))
                    return false;
            }
            break;
        }

        // If the use is outside of known blocks, then we can't promote it.
        if (!knownBlocks.contains(getBlock(user)))
            return false;
    }


    // If all of the uses passed our checking, then
    // we are good to go.
    return true;
}

// Identify local variables that can be promoted to SSA form
void identifyPromotableVars(ConstructSSAContext* context)
{
    HashSet<IRBlock*> knownBlocks;
    for (auto bb = context->globalVal->getFirstBlock(); bb; bb = bb->getNextBlock())
    {
        knownBlocks.add(bb);
    }

    for (auto bb = context->globalVal->getFirstBlock(); bb; bb = bb->getNextBlock())
    {
        for (auto ii = bb->getFirstInst(); ii; ii = ii->getNextInst())
        {
            if (ii->getOp() != kIROp_Var)
                continue;

            IRVar* var = (IRVar*)ii;

            if (isPromotableVar(context, var, knownBlocks))
            {
                context->promotableVars.add(var);
            }
        }
    }
}

/// If `value` is a promotable variable, then cast and return it.
IRVar* asPromotableVar(ConstructSSAContext* context, IRInst* value)
{
    if (value->getOp() != kIROp_Var)
        return nullptr;

    IRVar* var = (IRVar*)value;
    if (!context->promotableVars.contains(var))
        return nullptr;

    return var;
}

/// If `value` is a promotable variable or an access chain
/// based on one, then cast and return the variable.
IRVar* asPromotableVarAccessChain(ConstructSSAContext* context, IRInst* value)
{
    switch (value->getOp())
    {
    case kIROp_Var:
        return asPromotableVar(context, value);

    case kIROp_FieldAddress:
    case kIROp_GetElementPtr:
        return asPromotableVarAccessChain(context, value->getOperand(0));

    default:
        return nullptr;
    }
}

/// After looking up the SSA value of avariable in some context,
/// apply whatever "access chain" was applied at the original use site.
///
/// E.g., if the original operation was *((&a)->b) or *((&a) + i) and we've
/// resolved that the value of the variable `a` should be `v`, then
/// construct v.b or v[i].
///
IRInst* applyAccessChain(
    ConstructSSAContext* context,
    IRBuilder* builder,
    IRInst* accessChain,
    IRInst* leafVarValue)
{
    switch (accessChain->getOp())
    {
    default:
        SLANG_UNEXPECTED("unexpected op along access chain");
        UNREACHABLE_RETURN(leafVarValue);

    case kIROp_Var:
        return leafVarValue;

    case kIROp_FieldAddress:
        {
            SLANG_ASSERT(context->instsToRemove.contains(accessChain));

            auto baseChain = accessChain->getOperand(0);
            auto fieldKey = accessChain->getOperand(1);
            auto type = cast<IRPtrTypeBase>(accessChain->getDataType())->getValueType();
            auto baseValue = applyAccessChain(context, builder, baseChain, leafVarValue);
            auto extractInst = builder->emitFieldExtract(type, baseValue, fieldKey);

            for (auto decoration : accessChain->getDecorations())
            {
                cloneDecoration(decoration, extractInst);
            }

            return extractInst;
        }

    case kIROp_GetElementPtr:
        {
            SLANG_ASSERT(context->instsToRemove.contains(accessChain));

            auto baseChain = accessChain->getOperand(0);
            auto index = accessChain->getOperand(1);
            auto type = cast<IRPtrTypeBase>(accessChain->getDataType())->getValueType();
            auto baseValue = applyAccessChain(context, builder, baseChain, leafVarValue);
            return builder->emitElementExtract(type, baseValue, index);
        }
    }
}

// Try to read the value of an SSA variable
// in the context of the given block. If
// the variable is defined in the block, then
// that value will be used. If not, this all
// may recursively work its way up through
// the predecessors of the block.
IRInst* readVar(ConstructSSAContext* context, SSABlockInfo* blockInfo, IRVar* var);

/// Try to copy any relevant decorations from `var` over to `val`.
///
static void cloneRelevantDecorations(IRVar* var, IRInst* val)
{
    // Copy selected decorations over from the original
    // variable to the SSA variable, when doing so is
    // required for semantics.
    //
    for (auto decoration : var->getDecorations())
    {
        switch (decoration->getOp())
        {
        default:
            // Ignore most decorations.
            //
            // TODO: Should we include or exclude by default?
            break;

        case kIROp_PreciseDecoration:
            // Copy these decorations if the target doesn't already have them,
            // but don't make duplicate decorations on the target.
            //
            if (!val->findDecorationImpl(decoration->getOp()))
            {
                cloneDecoration(nullptr, decoration, val, var->getModule());
            }
            break;
        case kIROp_NameHintDecoration:
            // If the target already contains a linkage decoration, don't add
            // a name decoration to avoid issues with emit logic.
            if (!val->findDecorationImpl(decoration->getOp()) &&
                !val->findDecoration<IRLinkageDecoration>())
            {
                cloneDecoration(nullptr, decoration, val, var->getModule());
            }
            break;
        }
    }
}

// Add a phi node to represent the given variable
PhiInfo* addPhi(ConstructSSAContext* context, SSABlockInfo* blockInfo, IRVar* var)
{
    auto builder = &blockInfo->builder;

    auto valueType = var->getDataType()->getValueType();
    if (auto rate = var->getRate())
    {
        valueType = context->getBuilder()->getRateQualifiedType(rate, valueType);
    }
    IRParam* phi = builder->emitParam(valueType);
    cloneRelevantDecorations(var, phi);

    RefPtr<PhiInfo> phiInfo = new PhiInfo();
    context->phiInfos.add(phi, phiInfo);

    phi->sourceLoc = var->sourceLoc;
    phiInfo->phi = phi;
    phiInfo->var = var;

    blockInfo->phis.add(phiInfo);

    return phiInfo;
}

IRInst* tryRemoveTrivialPhi(ConstructSSAContext* context, PhiInfo* phiInfo)
{
    auto phi = phiInfo->phi;

    // We are going to check if all of the operands
    // to the phi are either the same, or are equal
    // to the phi itself.

    IRInst* same = nullptr;
    for (auto u : phiInfo->operands)
    {
        auto usedVal = u.get();
        SLANG_ASSERT(usedVal);

        if (usedVal == same || usedVal == phi)
        {
            // Either this is a self-reference, or it refers
            // to the same value we've seen already.
            continue;
        }
        if (same != nullptr)
        {
            // We've found at least two distinct values
            // other than the phi itself, so this phi
            // indeed appears to be non-trivial.
            //
            // We will keep the phi around.
            return phi;
        }
        else
        {
            // This value is distinct from the phi itself,
            // so we need to track its value.
            same = usedVal;
        }
    }

    if (!same)
    {
        // There were no operands other than the phi itself.
        // This implies that the value at the use sites should
        // actually be undefined.
        //
        // For now we will simply return in this case, without
        // removing the phi node.
        //
        // TODO: Construct a value for `same` that represents an
        // undefined value with the same type as `phi`.
        //
        return phi;
    }

    // Removing this phi as trivial may make other phi nodes
    // become trivial. We will recognize such candidates
    // by looking for phi nodes that use this node.
    List<PhiInfo*> otherPhis;
    for (auto u = phi->firstUse; u; u = u->nextUse)
    {
        auto user = u->user;
        if (!user)
            continue;
        if (user == phi)
            continue;

        if (user->getOp() == kIROp_Param)
        {
            auto maybeOtherPhi = (IRParam*)user;
            if (auto otherPhiInfo = context->getPhiInfo(maybeOtherPhi))
            {
                otherPhis.add(otherPhiInfo);
            }
        }
    }

    // replace uses of the phi (including its possible uses
    // of itself) with the unique non-phi value.
    phi->replaceUsesWith(same);
    phi->removeAndDeallocate();

    // Clear out the operands to the phi, since they won't
    // actually get used in the program any more.
    for (auto& u : phiInfo->operands)
    {
        u.clear();
    }

    // We will record the value that was used to replace this
    // phi, so that we can easily look it up later.
    phiInfo->replacement = same;

    // Now that we've cleaned up this phi, we need to consider
    // other phis that might have become  trivial.
    for (auto otherPhi : otherPhis)
    {
        // It is possible that between when we added a phi
        // node to `otherPhis` and here it might have been
        // replaced. This could happen if `otherPhis` had
        // two entries, `A` and `B`, and the recursive
        // `tryRemoveTrivialPhi` on `A` also ended up
        // eliminating `B`.
        //
        if (otherPhi->replacement)
            continue;

        tryRemoveTrivialPhi(context, otherPhi);
    }

    return same;
}

IRInst* addPhiOperands(ConstructSSAContext* context, SSABlockInfo* blockInfo, PhiInfo* phiInfo)
{
    auto var = phiInfo->var;

    auto block = blockInfo->block;

    List<IRInst*> operandValues;
    auto predecessorCount = block->getPredecessors().getCount();
    for (auto predBlock : block->getPredecessors())
    {
        // Precondition: if we have multiple predecessors, then
        // each must have only one successor (no critical edges).
        //
        SLANG_RELEASE_ASSERT(predecessorCount <= 1 || predBlock->getSuccessors().getCount() == 1);

        auto predInfo = *context->blockInfos.tryGetValue(predBlock);

        auto phiOperand = readVar(context, predInfo, var);
        SLANG_ASSERT(phiOperand != nullptr);

        operandValues.add(phiOperand);
    }

    // The `IRUse`  type needs to stay at a stable location
    // since they get threaded into lists. We allocate the
    // list with its final size so that we can preserve the
    // required invariant.

    UInt operandCount = operandValues.getCount();
    phiInfo->operands.setCount(operandCount);
    for (UInt ii = 0; ii < operandCount; ++ii)
    {
        phiInfo->operands[ii].init(phiInfo->phi, operandValues[ii]);
    }

    return tryRemoveTrivialPhi(context, phiInfo);
}

void writeVar(ConstructSSAContext* /*context*/, SSABlockInfo* blockInfo, IRVar* var, IRInst* val)
{
    blockInfo->valueForVar[var] = val;
}

void maybeSealBlock(ConstructSSAContext* context, SSABlockInfo* blockInfo)
{
    // We can't seal a block that has already been sealed.
    if (blockInfo->isSealed)
        return;

    // We can't seal a block until all of its predecessors
    // have been filled.
    for (auto pp : blockInfo->block->getPredecessors())
    {
        auto predInfo = *context->blockInfos.tryGetValue(pp);
        if (!predInfo->isFilled)
            return;
    }

    // All the checks passed, so it seems like we can be sealed.

    // We will loop over any incomplete phis that have been recoreded
    // for this block, and complete them here.
    //
    // Note that we are doing the "inefficient" loop where we compute
    // the count on each iteration to account for the possibility that
    // new incomplete phis will get added while we are working.
    for (Index ii = 0; ii < blockInfo->phis.getCount(); ++ii)
    {
        auto incompletePhi = blockInfo->phis[ii];
        addPhiOperands(context, blockInfo, incompletePhi);
    }

    // After we've completed all our incomplete phis, we can mark this
    // block as sealed and move along.
    blockInfo->isSealed = true;
}

// In some cases we may have a pointer to an IR value that
// represents a phi node that has been replaced with another
// IR value, because we discovered that the phi is no longer
// needed.
//
// The `maybeGetPhiReplacement` function will follow any
// chain of replacements that might be present, so that we
// don't end up referencing a dangling/unused value in
// the code that we generate.
//
IRInst* maybeGetPhiReplacement(ConstructSSAContext* context, IRInst* inVal)
{
    IRInst* val = inVal;

    while (val->getOp() == kIROp_Param)
    {
        // The value is a parameter, but is it a phi?
        IRParam* maybePhi = (IRParam*)val;
        RefPtr<PhiInfo> phiInfo = nullptr;
        if (!context->phiInfos.tryGetValue(maybePhi, phiInfo))
            break;

        // Okay, this is indeed a phi we are adding, but
        // is it one that got replaced?
        if (!phiInfo->replacement)
            break;

        // The phi we want to use got replaced, so we
        // had better use the replacement instead.
        val = phiInfo->replacement;
    }

    return val;
}

IRInst* readVarRec(ConstructSSAContext* context, SSABlockInfo* blockInfo, IRVar* var)
{
    IRInst* val = nullptr;
    if (!blockInfo->isSealed)
    {
        // If block isn't sealed, we need to
        // speculatively add a phi to it.
        // This phi may get removed later, once
        // we are able to seal this block.

        PhiInfo* phiInfo = addPhi(context, blockInfo, var);
        val = phiInfo->phi;
    }
    else
    {
        // If the block is sealed, then we are free to look at
        // it predecessor list, and use that to decide what to do.
        auto predecessors = blockInfo->block->getPredecessors();

        //
        IRBlock* firstPred = nullptr;
        bool multiplePreds = false;
        for (auto pp : predecessors)
        {
            if (!firstPred)
            {
                // A candidate for the sole predecessor
                firstPred = pp;
            }
            else if (pp == firstPred)
            {
                // Same as existing predecessor
            }
            else
            {
                // Multiple unique predecessors
                multiplePreds = true;
            }
        }

        if (!firstPred)
        {
            // The block had *no* predecssors. This will commonly
            // happen for the entry block, but could also conceivably
            // happen for a block that is somehow disconnected
            // from the CFG and thus unreachable.

            // We would only reach this function (`readVarRec`) if
            // a local lookup in the block had already failed, so
            // at this point we are dealing with an undefined value.

            auto type = var->getDataType()->getValueType();
            val = blockInfo->builder.emitUndefined(type);
            cloneRelevantDecorations(var, val);
        }
        else if (!multiplePreds)
        {
            // There is only a single predecessor for this block,
            // so there is no need to insert a phi. Instead, we
            // just perform the lookup step recursively in
            // the predecessor.
            auto predInfo = *context->blockInfos.tryGetValue(firstPred);
            val = readVar(context, predInfo, var);
        }
        else
        {
            // The default/fallback case requires us to create
            // a phi node in the current block, and then look
            // up the appropriate operands in the predecessor
            // blocks, which will eventually become the operands
            // that drive the phi.

            // Create the phi node for the given variable
            PhiInfo* phiInfo = addPhi(context, blockInfo, var);

            // Mark the phi as the value for the variable inside
            // this block
            writeVar(context, blockInfo, var, phiInfo->phi);

            // Now add operands to the phi and maybe simplify
            // it, based on what gets found.

            val = addPhiOperands(context, blockInfo, phiInfo);
        }
    }

    // Whatever value we find, we need to mark it as the
    // value for the given variable in this block
    writeVar(context, blockInfo, var, val);

    // If `val` represents a phi node (block parameter) then
    // it is possible that some of the operations above might
    // have caused it to be replaced with another value,
    // and in that case we had better not return it to
    // be referenced in user code.
    //
    // Note: it is okay for the `valueForVar` map that
    // we update in `writeVar` to use the old value, so long
    // as we do this replacement logic anywhere we might read
    // from that map.
    //
    val = maybeGetPhiReplacement(context, val);

    return val;
}


IRInst* readVar(ConstructSSAContext* context, SSABlockInfo* blockInfo, IRVar* var)
{
    // In the easy case, there will be a preceeding
    // store in the same block, so we can use
    // that local value.
    IRInst* val = nullptr;
    if (blockInfo->valueForVar.tryGetValue(var, val))
    {
        // Hooray, we found a value to use, and we
        // can proceed without too many complications.

        // Just like in the `readVarRec` case above, we need
        // to handle the case where `val` might represent
        // a phi node that has subsequently been replaced.
        //
        val = maybeGetPhiReplacement(context, val);

        return val;
    }

    if (blockInfo->block == var->parent)
    {
        // If this is the block that actually *introduces*
        // the variable, then there is no reason to keep
        // searching, because its value cannot have been
        // established in a predecessor block.
        //
        auto type = var->getDataType()->getValueType();
        val = blockInfo->builder.emitUndefined(type);
        cloneRelevantDecorations(var, val);
        writeVar(context, blockInfo, var, val);
        return val;
    }

    // Otherwise we need to try to non-trivial/recursive
    // case of lookup.
    return readVarRec(context, blockInfo, var);
}

void collectInstsToRemove(ConstructSSAContext* context, IRBlock* block)
{
    IRInst* next = nullptr;
    for (auto ii = block->getFirstInst(); ii; ii = next)
    {
        next = ii->getNextInst();

        switch (ii->getOp())
        {
        default:
            // Ordinary instruction -> leave as-is
            break;
        case kIROp_GetElementPtr:
        case kIROp_FieldAddress:
            {
                auto ptrArg = ii->getOperand(0);
                if (const auto var = asPromotableVarAccessChain(context, ptrArg))
                {
                    context->instsToRemove.add(ii);
                }
            }
            break;
        }
    }
}

void processBlock(ConstructSSAContext* context, IRBlock* block, SSABlockInfo* blockInfo)
{
    hoistInstOutOfASMBlocks(block);

    // Before starting, check if this block can be sealed
    maybeSealBlock(context, blockInfo);

    // Walk the instructions in the block, and either
    // leave them as-is, or replace them with a value
    // that we look up with local/global value numbering

    List<IRInst*> workList;
    for (auto ii = block->getFirstInst(); ii; ii = ii->getNextInst())
        workList.add(ii);

    for (auto& ii : workList)
    {
        // Any new instructions we create to represent
        // the new value will get inserted before whatever
        // instruction we are working with.
        blockInfo->builder.setInsertBefore(ii);

        switch (ii->getOp())
        {
        default:
            // Ordinary instruction -> leave as-is
            break;

        case kIROp_Store:
            {
                auto storeInst = (IRStore*)ii;
                auto ptrArg = storeInst->ptr.get();
                auto valArg = storeInst->val.get();

                if (auto var = asPromotableVar(context, ptrArg))
                {
                    // We are storing to a promotable variable,
                    // so we want to register the value being
                    // stored as the value for the given SSA
                    // variable.
                    writeVar(context, blockInfo, var, valArg);

                    // Also eliminate the store instruction,
                    // since it is no longer needed.
                    storeInst->removeAndDeallocate();
                }
            }
            break;

        case kIROp_Load:
            {
                IRLoad* loadInst = (IRLoad*)ii;
                auto ptrArg = loadInst->ptr.get();

                if (auto var = asPromotableVarAccessChain(context, ptrArg))
                {
                    // We are loading from a promotable variable.
                    // Look up the value in the context of this
                    // block.
                    auto val = readVar(context, blockInfo, var);

                    cloneRelevantDecorations(var, val);

                    val = applyAccessChain(context, &blockInfo->builder, ptrArg, val);

                    // We can just replace all uses of this
                    // load instruction with the given value.
                    loadInst->replaceUsesWith(val);

                    // Also eliminate the load instruction,
                    // since it is no longer needed.
                    loadInst->removeAndDeallocate();
                }
            }
            break;
        }
    }

    auto terminator = block->getTerminator();
    SLANG_ASSERT(terminator);
    blockInfo->builder.setInsertBefore(terminator);

    // Once we are done with all of the instructions
    // in a block, we can mark it as "filled," which
    // means we can actually consider lookups into
    // it.
    blockInfo->isFilled = true;

    // Having filled this block might allow us to seal some
    // of its successor(s)
    for (auto ss : block->getSuccessors())
    {
        auto successorInfo = *context->blockInfos.tryGetValue(ss);
        maybeSealBlock(context, successorInfo);
    }
}

IRBlock* IREdge::getPredecessor() const
{
    return cast<IRBlock>(getUse()->getUser()->parent);
}

IRBlock* IREdge::getSuccessor() const
{
    return cast<IRBlock>(getUse()->get());
}

void IRBuilder::insertBlockAlongEdge(IRModule* module, IREdge const& edge)
{
    auto pred = edge.getPredecessor();
    auto succ = edge.getSuccessor();
    auto edgeUse = edge.getUse();

    IRBuilder builder(module);
    builder.setInsertInto(pred);

    // Create a new block that will sit "along" the edge
    IRBlock* edgeBlock = builder.createBlock();

    edgeUse->debugValidate();

    // The predecessor block should now branch to
    // the edge block.
    edgeUse->set(edgeBlock);

    // The edge block should branch (unconditionally)
    // to the successor block.
    builder.setInsertInto(edgeBlock);
    builder.emitBranch(succ);

    // Insert the new block into the block list
    // for the function.
    //
    // In principle, the order of this list shouldn't
    // affect the semantics of a program, but we
    // might want to be careful about ordering anyway.
    edgeBlock->insertAfter(pred);
}

bool IREdge::isCritical() const
{
    // A critical edge is an edge P -> S where
    // P has multiple sucessors, and S has multiple
    // predecessors.
    //
    auto pred = getPredecessor();
    auto succ = getSuccessor();

    // If the predecessor block doesn't have multiple successors,
    // then this can't be a critical edge.
    //
    if (pred->getSuccessors().getCount() <= 1)
        return false;

    // For the edge to be critical, the successor must have
    // more than one predecessor.
    //
    // More than that, we require that it has more than one
    // *unique* predecessor, to handle the case where multiple
    // cases of a `switch` might lead to the same block.
    //
    // To implement this, we test if it has any predecessor
    // other than `pred` which we already know about.
    //
    bool multiplePreds = false;
    for (auto pp : succ->getPredecessors())
    {
        if (pp != pred)
        {
            multiplePreds = true;
            break;
        }
    }
    if (!multiplePreds)
        return false;

    // At this point we have confirmed that `edgeUse`
    // leads from a block with multiple successors
    // to one with multiple (distinct) predecessors,
    // so we are sure it is a critical edge.
    //
    return true;
}

static void breakCriticalEdges(ConstructSSAContext* context)
{
    auto globalVal = context->globalVal;

    // We will make a pass over the CFG to collect all the critical
    // edges, and then we will break them in a follow-up pass.
    //
    List<IREdge> criticalEdges;
    for (auto pred = globalVal->getFirstBlock(); pred; pred = pred->getNextBlock())
    {
        // As an early out: if the predecessor block
        // has one (or fewer) successors, then no
        // edge from that block can be critical.
        //
        auto successors = pred->getSuccessors();
        if (successors.getCount() <= 1)
            continue;

        // Otherwise, we iterate over its outgoing
        // edges (represented with its successor list),
        // and record those that are critical.
        //
        auto succIter = successors.begin();
        auto succEnd = successors.end();
        for (; succIter != succEnd; ++succIter)
        {
            auto edge = succIter.getEdge();
            if (edge.isCritical())
            {
                criticalEdges.add(edge);
            }
        }
    }

    // Now we will iterate over the critical edges and break each
    // one by inserting a new block. Note that we do not try
    // to break the edges while doing the initial walk, because
    // that would change the CFG while we are walking it.

    for (auto edge : criticalEdges)
    {
        IRBuilder::insertBlockAlongEdge(context->module, edge);
    }
}

// Construct SSA form for a global value with code
bool constructSSA(ConstructSSAContext* context)
{
    // First, detect and and break any critical edges in the CFG,
    // because our representation of SSA form doesn't allow for them.
    breakCriticalEdges(context);

    // Figure out what variables we can promote to
    // SSA temporaries.
    identifyPromotableVars(context);

    // If none of the variables are promote-able,
    // then we can exit without making any changes
    if (context->promotableVars.getCount() == 0)
        return false;

    // We are going to walk the blocks in order,
    // and try to process each, by replacing loads
    // and stores of promotable variables with simple values.

    auto globalVal = context->globalVal;
    for (auto bb : globalVal->getBlocks())
    {
        auto blockInfo = new SSABlockInfo();
        blockInfo->block = bb;

        blockInfo->builder = IRBuilder(context->module);
        blockInfo->builder.setInsertBefore(bb->getLastInst());

        context->blockInfos.add(bb, blockInfo);
    }

    for (auto bb : globalVal->getBlocks())
        collectInstsToRemove(context, bb);

    for (auto bb : globalVal->getBlocks())
    {
        auto blockInfo = *context->blockInfos.tryGetValue(bb);
        processBlock(context, bb, blockInfo);
    }

    // We need to transfer the logical arguments to our phi nodes
    // from the phi nodes back to the predecessor blocks that will
    // pass them in.
    for (auto bb : globalVal->getBlocks())
    {
        auto blockInfo = *context->blockInfos.tryGetValue(bb);

        // First remove phis from their parent blocks.
        for (auto phiInfo : blockInfo->phis)
            if (!phiInfo->replacement)
                phiInfo->phi->removeFromParent();

        // Then, add them back in a consistent order, and add predecessor
        // args in the same order.
        //
        for (auto phiInfo : blockInfo->phis)
        {
            // If we replaced this phi with another value,
            // then we had better not include it in the result.
            if (phiInfo->replacement)
                continue;

            // We should add the phi as an explicit parameter of
            // the given block.
            bb->addParam(phiInfo->phi);

            UInt predCounter = 0;
            for (auto pp : bb->getPredecessors())
            {
                UInt predIndex = predCounter++;
                auto predInfo = *context->blockInfos.tryGetValue(pp);

                IRInst* operandVal = phiInfo->operands[predIndex].get();

                phiInfo->operands[predIndex].clear();

                predInfo->successorArgs.add(operandVal);
            }
        }
    }

    // Some blocks may now need to pass along arguments to their successor,
    // which have been stored into the `SSABlockInfo::successorArgs` field.
    for (auto bb : globalVal->getBlocks())
    {
        auto blockInfo = *context->blockInfos.tryGetValue(bb);

        // Sanity check: all blocks should be filled and sealed.
        SLANG_ASSERT(blockInfo->isSealed);
        SLANG_ASSERT(blockInfo->isFilled);

        // Don't do any work for blocks that don't need to pass along
        // values to the sucessor block.
        auto addedArgCount = blockInfo->successorArgs.getCount();
        if (addedArgCount == 0)
            continue;

        // We need to replace the terminator instruction with one that
        // has additional arguments.

        IRTerminatorInst* oldTerminator = bb->getTerminator();
        SLANG_ASSERT(oldTerminator);

        blockInfo->builder.setInsertInto(bb);

        auto oldArgCount = oldTerminator->getOperandCount();
        auto newArgCount = oldArgCount + addedArgCount;

        List<IRInst*> newArgs;
        for (UInt aa = 0; aa < oldArgCount; ++aa)
        {
            newArgs.add(oldTerminator->getOperand(aa));
        }
        for (Index aa = 0; aa < addedArgCount; ++aa)
        {
            newArgs.add(blockInfo->successorArgs[aa]);
        }

        IRTerminatorInst* newTerminator = (IRTerminatorInst*)blockInfo->builder.emitIntrinsicInst(
            oldTerminator->getFullType(),
            oldTerminator->getOp(),
            newArgCount,
            newArgs.getBuffer());

        // Transfer decorations (a terminator should have no children) over to the new instruction.
        //
        oldTerminator->transferDecorationsTo(newTerminator);

        // Copy the source location of the old terminator to the new terminator
        newTerminator->sourceLoc = oldTerminator->sourceLoc;

        // A terminator better not have uses, so we shouldn't have
        // to replace them.
        SLANG_ASSERT(!oldTerminator->firstUse);


        // Okay, we should be clear to remove the old terminator
        oldTerminator->removeAndDeallocate();
    }

    // Remove all the instructions we marked for deletion along
    // the way.
    //
    // Currently these are "access chain" instructions for
    // loads from (parts of) variables that got promoted.
    for (auto inst : context->instsToRemove)
    {
        // TODO: do we need to be careful here in case one
        // of thes operations still has uses, as part of
        // another to-be-remvoed instruction?

        inst->removeAndDeallocate();
    }

    // Now we should be able to go through and remove
    // of of the variables
    for (auto var : context->promotableVars)
    {
        var->removeAndDeallocate();
    }
    return true;
}

// Construct SSA form for a global value with code and reuse
// an existing sharedBuilder
//
bool constructSSA(IRModule* module, IRGlobalValueWithCode* globalVal)
{
    ConstructSSAContext context;
    context.globalVal = globalVal;
    context.module = module;
    context.builder = IRBuilder(module);
    context.builder.setInsertInto(module);
    return constructSSA(&context);
}

bool constructSSA(IRModule* module, IRInst* globalVal)
{
    switch (globalVal->getOp())
    {
    case kIROp_Func:
    case kIROp_GlobalVar:
        return constructSSA(module, (IRGlobalValueWithCode*)globalVal);

    case kIROp_Generic:
        {
            // The above cases handle the actual code-bearing declarations
            // that can contian basic blocks with local variables, but
            // we would also like to perform SSA simplifications on
            // *generic* functions, and so we will also process any
            // instruction that is produced by an `IRGeneric`.
            //
            // TODO: At some point we may simply want to apply this pass
            // recursively to *all* instructions, in order to make it
            // robust to the presence of nested functions in general.

            auto generic = cast<IRGeneric>(globalVal);
            auto returnVal = findInnerMostGenericReturnVal(generic);
            if (!returnVal)
                return false;

            return constructSSA(module, returnVal);
        }

    default:
        break;
    }
    return false;
}

bool constructSSA(IRModule* module)
{
    bool changed = false;
    for (auto ii : module->getGlobalInsts())
    {
        changed |= constructSSA(module, ii);
    }
    return changed;
}

bool constructSSA(IRInst* globalVal)
{
    switch (globalVal->getOp())
    {
    case kIROp_Func:
    case kIROp_GlobalVar:
        return constructSSA(globalVal->getModule(), (IRGlobalValueWithCode*)globalVal);

    default:
        break;
    }
    return false;
}

} // namespace Slang
