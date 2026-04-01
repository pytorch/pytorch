#include "slang-ir-autodiff-primal-hoist.h"

#include "../core/slang-func-ptr.h"
#include "slang-ast-support-types.h"
#include "slang-ir-autodiff-loop-analysis.h"
#include "slang-ir-autodiff-region.h"
#include "slang-ir-insts.h"
#include "slang-ir-simplify-cfg.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{

void applyCheckpointSet(
    CheckpointSetInfo* checkpointInfo,
    IRGlobalValueWithCode* func,
    HoistedPrimalsInfo* hoistInfo,
    HashSet<IRUse*>& pendingUses,
    Dictionary<IRBlock*, IRBlock*>& mapPrimalBlockToRecomputeBlock,
    IROutOfOrderCloneContext* cloneCtx,
    Dictionary<IRBlock*, List<IndexTrackingInfo>>& blockIndexInfo);

bool containsOperand(IRInst* inst, IRInst* operand)
{
    for (UIndex ii = 0; ii < inst->getOperandCount(); ii++)
        if (inst->getOperand(ii) == operand)
            return true;

    return false;
}

static bool isDifferentialInst(IRInst* inst)
{
    auto parent = inst->getParent();
    if (parent->findDecoration<IRDifferentialInstDecoration>())
        return true;
    return inst->findDecoration<IRDifferentialInstDecoration>() != nullptr;
}

static bool isDifferentialBlock(IRBlock* block)
{
    return block->findDecoration<IRDifferentialInstDecoration>();
}

static IRBlock* getLoopConditionBlock(IRLoop* loop)
{
    auto condBlock = as<IRBlock>(loop->getTargetBlock());
    SLANG_ASSERT(as<IRIfElse>(condBlock->getTerminator()));
    return condBlock;
}

static IRBlock* getLoopRegionBodyBlock(IRLoop* loop)
{
    auto condBlock = getLoopConditionBlock(loop);
    // We assume the loop body always sit at the true side of the if-else.
    if (auto ifElse = as<IRIfElse>(condBlock->getTerminator()))
    {
        return ifElse->getTrueBlock();
    }
    return nullptr;
}

static IRBlock* tryGetSubRegionEndBlock(IRInst* terminator)
{
    auto loop = as<IRLoop>(terminator);
    if (!loop)
        return nullptr;
    return loop->getBreakBlock();
}

static Dictionary<IRBlock*, IRBlock*> createPrimalRecomputeBlocks(
    IRGlobalValueWithCode* func,
    Dictionary<IRBlock*, List<IndexTrackingInfo>>& indexedBlockInfo,
    IROutOfOrderCloneContext* cloneCtx)
{
    IRBlock* firstDiffBlock = nullptr;
    for (auto block : func->getBlocks())
    {
        if (isDifferentialBlock(block))
        {
            firstDiffBlock = block;
            break;
        }
    }
    if (!firstDiffBlock)
        return Dictionary<IRBlock*, IRBlock*>();

    Dictionary<IRLoop*, IRLoop*> mapPrimalLoopToDiffLoop;
    for (auto block : func->getBlocks())
    {
        if (isDifferentialBlock(block))
        {
            if (auto diffLoop = as<IRLoop>(block->getTerminator()))
            {
                if (auto diffDecor = diffLoop->findDecoration<IRDifferentialInstDecoration>())
                {
                    mapPrimalLoopToDiffLoop[as<IRLoop>(diffDecor->getPrimalInst())] = diffLoop;
                }
            }
        }
    }

    IRBuilder builder(func);
    Dictionary<IRBlock*, IRBlock*> recomputeBlockMap;

    // Create the first recompute block right before the first diff block,
    // and change all jumps into the diff block to the recompute block instead.
    auto createRecomputeBlock = [&](IRBlock* primalBlock)
    {
        auto recomputeBlock = builder.createBlock();
        recomputeBlock->insertAtEnd(func);
        builder.addDecoration(recomputeBlock, kIROp_RecomputeBlockDecoration);
        recomputeBlockMap.add(primalBlock, recomputeBlock);
        indexedBlockInfo.set(recomputeBlock, indexedBlockInfo.getValue(primalBlock));
        return recomputeBlock;
    };

    auto firstRecomputeBlock = createRecomputeBlock(func->getFirstBlock());
    firstRecomputeBlock->insertBefore(firstDiffBlock);
    moveParams(firstRecomputeBlock, firstDiffBlock);
    firstDiffBlock->replaceUsesWith(firstRecomputeBlock);

    struct WorkItem
    {
        // The first primal block in this region.
        IRBlock* primalBlock;

        // The recompute block created for the first primal block in this region.
        IRBlock* recomptueBlock;

        // The end of primal block in tihs region.
        IRBlock* regionEndBlock;

        // The first diff block in this region.
        IRBlock* firstDiffBlock;
    };

    List<WorkItem> workList;
    WorkItem firstWorkItem =
        {func->getFirstBlock(), firstRecomputeBlock, firstRecomputeBlock, firstDiffBlock};
    workList.add(firstWorkItem);

    recomputeBlockMap[func->getFirstBlock()] = firstRecomputeBlock;

    for (Index i = 0; i < workList.getCount(); i++)
    {
        auto workItem = workList[i];
        auto primalBlock = workItem.primalBlock;
        auto recomputeBlock = workItem.recomptueBlock;

        List<IndexTrackingInfo>* thisBlockIndexInfo = indexedBlockInfo.tryGetValue(primalBlock);
        if (!thisBlockIndexInfo)
            continue;

        builder.setInsertInto(recomputeBlock);
        if (auto subRegionEndBlock = tryGetSubRegionEndBlock(primalBlock->getTerminator()))
        {
            // The terminal inst of primalBlock marks the start of a sub loop region?
            // We need to queue work for both the next region after the loop at the current level,
            // and for the sub region for the next level.
            if (subRegionEndBlock == workItem.regionEndBlock)
            {
                // We have reached the end of top-level region, jump to first diff block.
                builder.emitBranch(workItem.firstDiffBlock);
            }
            else
            {
                // Have we already created a recompute block for this target?
                // If so, use it.
                IRBlock* existingRecomputeBlock = nullptr;
                if (recomputeBlockMap.tryGetValue(subRegionEndBlock, existingRecomputeBlock))
                {
                    builder.emitBranch(existingRecomputeBlock);
                }
                else
                {
                    // Queue work for the next region after the subregion at this level.
                    auto nextRegionRecomputeBlock = createRecomputeBlock(subRegionEndBlock);
                    nextRegionRecomputeBlock->insertAfter(recomputeBlock);
                    builder.emitBranch(nextRegionRecomputeBlock);

                    {
                        WorkItem newWorkItem = {
                            subRegionEndBlock,
                            nextRegionRecomputeBlock,
                            workItem.regionEndBlock,
                            workItem.firstDiffBlock};
                        workList.add(newWorkItem);
                    }
                }
            }
            // Queue work for the subregion.
            auto loop = as<IRLoop>(primalBlock->getTerminator());
            auto bodyBlock = getLoopRegionBodyBlock(loop);
            auto diffLoop = mapPrimalLoopToDiffLoop.getValue(loop);
            auto diffBodyBlock = getLoopRegionBodyBlock(diffLoop);
            auto bodyRecomputeBlock = createRecomputeBlock(bodyBlock);
            bodyRecomputeBlock->insertBefore(diffBodyBlock);
            diffBodyBlock->replaceUsesWith(bodyRecomputeBlock);

            // Map the primal condition block directly to the diff
            // conditon block (we won't create a recompute block for this)
            //
            recomputeBlockMap[getLoopConditionBlock(loop)] = getLoopConditionBlock(diffLoop);

            moveParams(bodyRecomputeBlock, diffBodyBlock);
            {
                // After CFG normalization, the loop body will contain only jumps to the
                // beginning of the loop.
                // If we see such a jump, it means we have reached the end of current
                // region in the loop.
                // Therefore, we set the regionEndBlock for the sub-region as loop's target
                // block.
                WorkItem newWorkItem =
                    {bodyBlock, bodyRecomputeBlock, loop->getTargetBlock(), diffBodyBlock};
                workList.add(newWorkItem);
            }
        }
        else
        {
            // This is a normal control flow, just copy the CFG structure.
            auto terminator = primalBlock->getTerminator();
            IRInst* newTerminator = nullptr;
            switch (terminator->getOp())
            {
            case kIROp_Switch:
            case kIROp_ifElse:
                newTerminator =
                    cloneCtx->cloneInstOutOfOrder(&builder, primalBlock->getTerminator());
                break;
            case kIROp_unconditionalBranch:
                newTerminator =
                    builder.emitBranch(as<IRUnconditionalBranch>(terminator)->getTargetBlock());
                break;
            default:
                SLANG_UNREACHABLE("terminator type");
            }

            // Modify jump targets in newTerminator to point to the right recompute block or
            // firstDiffBlock.
            for (UInt op = 0; op < newTerminator->getOperandCount(); op++)
            {
                auto target = as<IRBlock>(newTerminator->getOperand(op));
                if (!target)
                    continue;
                if (target == workItem.regionEndBlock)
                {
                    // This jump target is the end of the current region, we will jump to
                    // firstDiffBlock instead.
                    newTerminator->setOperand(op, workItem.firstDiffBlock);
                    continue;
                }

                // Have we already created a recompute block for this target?
                // If so, use it.
                IRBlock* existingRecomputeBlock = nullptr;
                if (recomputeBlockMap.tryGetValue(target, existingRecomputeBlock))
                {
                    newTerminator->setOperand(op, existingRecomputeBlock);
                    continue;
                }

                // This jump target is a normal part of control flow, clone the next block.
                auto targetRecomputeBlock = createRecomputeBlock(target);
                targetRecomputeBlock->insertBefore(workItem.firstDiffBlock);

                newTerminator->setOperand(op, targetRecomputeBlock);

                // Queue work for the successor.
                WorkItem newWorkItem = {
                    target,
                    targetRecomputeBlock,
                    workItem.regionEndBlock,
                    workItem.firstDiffBlock};
                workList.add(newWorkItem);
            }
        }
    }
    // After this pass, all primal blocks except the condition block and the false block of a loop
    // will have a corresponding recomputeBlock.
    return recomputeBlockMap;
}

// Checks if list A is a subset of list B by comparing their primal count parameters.
//
// Parameters:
//   indicesA - First list of IndexTrackingInfo to compare
//   indicesB - Second list of IndexTrackingInfo to compare
//
// Returns:
//   true if all indices in indicesA are present in indicesB, false otherwise
//
bool areIndicesSubsetOf(List<IndexTrackingInfo>& indicesA, List<IndexTrackingInfo>& indicesB)
{
    if (indicesA.getCount() > indicesB.getCount())
        return false;

    auto offset = (indicesB.getCount() - indicesA.getCount());
    for (Index ii = 0; ii < indicesA.getCount(); ii++)
    {
        if (indicesA[ii].primalCountParam != indicesB[ii + offset].primalCountParam)
            return false;
    }

    return true;
}

bool canInstBeStored(IRInst* inst)
{
    // Cannot store insts whose value is a type or a witness table, or a function.
    // These insts get lowered to target-specific logic, and cannot be
    // stored into variables or context structs as normal values.
    //
    if (as<IRTypeType>(inst->getDataType()) || as<IRWitnessTableType>(inst->getDataType()) ||
        as<IRTypeKind>(inst->getDataType()) || as<IRFuncType>(inst->getDataType()) ||
        !inst->getDataType())
        return false;

    return true;
}

// This is a helper that converts insts in a loop condition block into two if necessary,
// then replaces all uses 'outside' the loop region with the new insts. This is because
// insts in loop condition blocks can be used in two distinct regions (the loop body, and
// after the loop).
//
// We'll use CheckpointObject for the splitting, which is allowed on any value-typed inst.
//
void splitLoopConditionBlockInsts(
    IRGlobalValueWithCode* func,
    Dictionary<IRBlock*, List<IndexTrackingInfo>>& indexedBlockInfo)
{
    // RefPtr<IRDominatorTree> domTree = computeDominatorTree(func);

    // Collect primal loop condition blocks, and map differential blocks to their primal blocks.
    List<IRBlock*> loopConditionBlocks;
    Dictionary<IRBlock*, IRBlock*> diffBlockMap;
    for (auto block : func->getBlocks())
    {
        if (auto loop = as<IRLoop>(block->getTerminator()))
        {
            auto loopConditionBlock = getLoopConditionBlock(loop);
            if (isDifferentialBlock(loopConditionBlock))
            {
                auto diffDecor = loopConditionBlock->findDecoration<IRDifferentialInstDecoration>();
                diffBlockMap[cast<IRBlock>(diffDecor->getPrimalInst())] = loopConditionBlock;
            }
            else
                loopConditionBlocks.add(loopConditionBlock);
        }
    }

    // For each loop condition block, split the insts that are used in both the loop body and
    // after the loop.
    // Use the dominator tree to find uses of insts outside the loop body
    //
    // Essentially we want to split the uses dominated by the true block and the false block of the
    // condition.
    //
    IRBuilder builder(func->getModule());


    List<IRUse*> loopUses;
    List<IRUse*> afterLoopUses;

    for (auto condBlock : loopConditionBlocks)
    {
        // For each inst in the primal condition block, check if it has uses inside the loop body
        // as well as outside of it. (Use the indexedBlockInfo to perform the teets)
        //
        for (auto inst = condBlock->getFirstInst(); inst; inst = inst->getNextInst())
        {
            // Skip terminators and insts that can't be stored
            if (as<IRTerminatorInst>(inst) || !canInstBeStored(inst))
                continue;
            // Shouldn't see any vars.
            SLANG_ASSERT(!as<IRVar>(inst));

            // Get the indices for the condition block
            auto& condBlockIndices = indexedBlockInfo[condBlock];

            loopUses.clear();
            afterLoopUses.clear();

            // Check all uses of this inst
            for (auto use = inst->firstUse; use; use = use->nextUse)
            {
                auto userBlock = getBlock(use->getUser());
                auto& userBlockIndices = indexedBlockInfo[userBlock];

                // If all of the condBlock's indices are a subset of the userBlock's indices,
                // then the userBlock is inside the loop.
                //
                bool isInLoop = areIndicesSubsetOf(condBlockIndices, userBlockIndices);

                if (isInLoop)
                    loopUses.add(use);
                else
                    afterLoopUses.add(use);
            }

            // If inst has uses both inside and after the loop, create a copy for after-loop uses
            if (loopUses.getCount() > 0 && afterLoopUses.getCount() > 0)
            {
                setInsertAfterOrdinaryInst(&builder, inst);
                auto copy = builder.emitLoopExitValue(inst);

                // Copy source location so that checkpoint reporting is accurate
                copy->sourceLoc = inst->sourceLoc;

                // Replace after-loop uses with the copy
                for (auto use : afterLoopUses)
                {
                    builder.replaceOperand(use, copy);
                }
            }
        }
    }
}

RefPtr<HoistedPrimalsInfo> AutodiffCheckpointPolicyBase::processFunc(
    IRGlobalValueWithCode* func,
    Dictionary<IRBlock*, IRBlock*>& mapDiffBlockToRecomputeBlock,
    IROutOfOrderCloneContext* cloneCtx,
    Dictionary<IRBlock*, List<IndexTrackingInfo>>& blockIndexInfo)
{
    collectInductionValues(func);

    collectLoopExitConditions(func);

    RefPtr<CheckpointSetInfo> checkpointInfo = new CheckpointSetInfo();

    RefPtr<IRDominatorTree> domTree = computeDominatorTree(func);

    List<UseOrPseudoUse> workList;
    HashSet<UseOrPseudoUse> processedUses;
    HashSet<IRUse*> usesToReplace;

    auto addPrimalOperandsToWorkList = [&](IRInst* inst)
    {
        UIndex opIndex = 0;
        for (auto operand = inst->getOperands(); opIndex < inst->getOperandCount();
             operand++, opIndex++)
        {
            if (!isDifferentialInst(operand->get()) && !as<IRFunc>(operand->get()) &&
                !as<IRBlock>(operand->get()) && !(as<IRModuleInst>(operand->get()->getParent())) &&
                !isDifferentialBlock(getBlock(operand->get())))
                workList.add(operand);
        }

        // Is the type itself computed within our function?
        // If so, we'll need to consider that too (this is for existential types, specialize insts,
        // etc)
        // TODO: We might not really need to query the checkpointing algorithm for these
        // since they _have_ to be classified as 'recompute'
        //
        if (inst->getDataType() && (getParentFunc(inst->getDataType()) == func))
        {
            if (!isDifferentialBlock(getBlock(inst->getDataType())))
                workList.add(&inst->typeUse);
        }
    };

    // Populate recompute/store/invert sets with insts, by applying the policy
    // to them.
    //
    for (auto block : func->getBlocks())
    {
        // Skip parameter block.
        if (block == func->getFirstBlock())
            continue;

        if (!isDifferentialBlock(block))
            continue;

        for (auto child : block->getChildren())
        {
            // Special case: Ignore the primals used to construct the return pair.
            if (as<IRMakeDifferentialPair>(child) && as<IRReturn>(child->firstUse->getUser()))
            {
                // quick check
                SLANG_RELEASE_ASSERT(child->firstUse->nextUse == nullptr);
                continue;
            }

            // General case: we'll add all primal operands to the work list.
            addPrimalOperandsToWorkList(child);

            // Also add type annotations to the list, since these have to be made available to the
            // function context.
            //
            if (as<IRDifferentiableTypeAnnotation>(child))
            {
                checkpointInfo->recomputeSet.add(child);
                addPrimalOperandsToWorkList(child);
            }

            // We'll be conservative with the decorations we consider as differential uses
            // of a primal inst, in order to avoid weird behaviour with some decorations
            //
            for (auto decoration : child->getDecorations())
            {
                if (auto primalCtxDecoration =
                        as<IRBackwardDerivativePrimalContextDecoration>(decoration))
                    workList.add(&primalCtxDecoration->primalContextVar);
                else if (auto loopExitDecoration = as<IRLoopExitPrimalValueDecoration>(decoration))
                    workList.add(&loopExitDecoration->exitVal);
            }
        }

        addPrimalOperandsToWorkList(block->getTerminator());
    }

    while (workList.getCount() > 0)
    {
        while (workList.getCount() > 0)
        {
            auto use = workList.getLast();
            workList.removeLast();

            if (processedUses.contains(use))
                continue;

            processedUses.add(use);

            HoistResult result = HoistResult::none();

            // Sometimes, we already have a decision for this val.
            //
            // This is a workaround to some of the problems
            // with the multi-pass approach where we can see an
            // inst that was already classified, but through a
            // different use.
            //
            if (checkpointInfo->recomputeSet.contains(use.usedVal))
                result = HoistResult::recompute(use.usedVal);
            else if (checkpointInfo->storeSet.contains(use.usedVal))
                result = HoistResult::store(use.usedVal);
            else
                result = this->classify(use);

            if (result.mode == HoistResult::Mode::Store)
            {
                SLANG_ASSERT(!checkpointInfo->recomputeSet.contains(result.instToStore));
                checkpointInfo->storeSet.add(result.instToStore);
            }
            else if (result.mode == HoistResult::Mode::Recompute)
            {
                SLANG_ASSERT(!checkpointInfo->storeSet.contains(result.instToRecompute));
                checkpointInfo->recomputeSet.add(result.instToRecompute);

                if (isDifferentialInst(use.user) && use.irUse)
                    usesToReplace.add(use.irUse);

                if (auto param = as<IRParam>(result.instToRecompute))
                {
                    if (auto inductionInfo = inductionValueInsts.tryGetValue(param))
                    {
                        checkpointInfo->loopInductionInfo.addIfNotExists(param, *inductionInfo);
                        continue;
                    }

                    // Add in the branch-args of every predecessor block.
                    auto paramBlock = as<IRBlock>(param->getParent());
                    UIndex paramIndex = 0;
                    for (auto _param : paramBlock->getParams())
                    {
                        if (_param == param)
                            break;
                        paramIndex++;
                    }

                    for (auto predecessor : paramBlock->getPredecessors())
                    {
                        // If we hit this, the checkpoint policy is trying to recompute
                        // values across a loop region boundary (we don't currently support this,
                        // and in general this is quite inefficient in both compute & memory)
                        //
                        SLANG_RELEASE_ASSERT(!domTree->dominates(paramBlock, predecessor));

                        auto branchInst = as<IRUnconditionalBranch>(predecessor->getTerminator());
                        SLANG_ASSERT(branchInst->getOperandCount() > paramIndex);

                        workList.add(&branchInst->getArgs()[paramIndex]);
                    }
                }
                else if (auto exitValue = as<IRLoopExitValue>(result.instToRecompute))
                {
                    // If we also have an exit value (a stronger condition on the param), record
                    // it.
                    //
                    if (auto loopExitValueInst =
                            loopExitValueInsts.tryGetValue(exitValue->getVal()))
                    {
                        checkpointInfo->loopExitValueInsts.addIfNotExists(
                            exitValue->getVal(),
                            *loopExitValueInst);
                    }
                }
                else
                {
                    if (auto var = as<IRVar>(result.instToRecompute))
                    {
                        for (auto varUse = var->firstUse; varUse; varUse = varUse->nextUse)
                        {
                            switch (varUse->getUser()->getOp())
                            {
                            case kIROp_Store:
                            case kIROp_Call:
                                // When we have a var and a store/call insts that writes to the
                                // var, we treat as if there is a pseudo-use of the store/call
                                // to compute the var inst, i.e. the var depends on the
                                // store/call, despite the IR's def-use chain doesn't reflect
                                // this.
                                workList.add(UseOrPseudoUse(var, varUse->getUser()));
                                break;
                            }
                        }
                    }
                    else
                    {
                        addPrimalOperandsToWorkList(result.instToRecompute);
                    }
                }
            }
        }

        // If a var or call is in recomputeSet, move any var/calls associated with the same call to
        // recomputeSet.
        // This is a bit of a 'retro-active' analysis where we go back on processed insts and
        // correct them.
        //
        List<IRInst*> callVarWorkList;
        HashSet<IRInst*> callVarWorkListSet;
        for (auto inst : checkpointInfo->recomputeSet)
        {
            switch (inst->getOp())
            {
            case kIROp_Call:
            case kIROp_Var:
                callVarWorkList.add(inst);
                callVarWorkListSet.add(inst);
                break;
            }
        }

        for (Index i = 0; i < callVarWorkList.getCount(); i++)
        {
            auto inst = callVarWorkList[i];
            if (auto var = as<IRVar>(inst))
            {
                for (auto use = var->firstUse; use; use = use->nextUse)
                {
                    if (auto callUser = as<IRCall>(use->getUser()))
                    {
                        checkpointInfo->recomputeSet.add(callUser);
                        checkpointInfo->storeSet.remove(callUser);
                        if (callVarWorkListSet.add(callUser))
                            callVarWorkList.add(callUser);
                    }
                    else if (auto storeUser = as<IRStore>(use->getUser()))
                    {
                        checkpointInfo->recomputeSet.add(storeUser);
                        checkpointInfo->storeSet.remove(storeUser);
                        if (callVarWorkListSet.add(callUser))
                            callVarWorkList.add(callUser);
                    }
                }

                // This is a bit of a hack.. ideally we need to add the var to the worklist for
                // further processing rather than replicating those operations here.
                //
                for (auto use = var->firstUse; use; use = use->nextUse)
                {
                    if (isDifferentialInst(use->getUser()))
                        usesToReplace.add(use);
                }
            }
            else if (auto call = as<IRCall>(inst))
            {
                for (UInt j = 0; j < call->getArgCount(); j++)
                {
                    if (auto varArg = as<IRVar>(call->getArg(j)))
                    {
                        checkpointInfo->recomputeSet.add(varArg);
                        checkpointInfo->storeSet.remove(varArg);
                        if (callVarWorkListSet.add(varArg))
                            callVarWorkList.add(varArg);
                    }
                }

                // This next few lines are a bit of a hack.. ideally we need to add the call to the
                // main worklist for processing, so we don't have to repeat the recomputationn
                // actions.
                //
                auto calleeUse = &call->getOperands()[0];
                if (!as<IRModuleInst>(calleeUse->get()->getParent()) &&
                    !processedUses.contains(calleeUse))
                    addPrimalOperandsToWorkList(call);

                for (auto use = call->firstUse; use; use = use->nextUse)
                {
                    if (isDifferentialInst(use->getUser()))
                        usesToReplace.add(use);
                }
            }
        }
    }

    RefPtr<HoistedPrimalsInfo> hoistInfo = new HoistedPrimalsInfo();
    applyCheckpointSet(
        checkpointInfo,
        func,
        hoistInfo,
        usesToReplace,
        mapDiffBlockToRecomputeBlock,
        cloneCtx,
        blockIndexInfo);
    return hoistInfo;
}

struct ImplicationParams
{
    IRInst *condition, *induction, *block;
    SLANG_BYTEWISE_HASHABLE;
    bool operator==(const ImplicationParams& other) const
    {
        return condition == other.condition && induction == other.induction && block == other.block;
    }
};

struct ImplicationResult
{
    enum
    {
        // The value was not a constant offset from the induction variable and
        // the condition variable was true
        Falsified,
        // The condition variable is false, so the value's relationship to the
        // induction variable doesn't matter
        AntecedentHolds,
        // The value is a constant offset from the induction variable, stored
        // in 'factor'
        ConsequentHolds
    } e;
    IRIntegerValue factor;
};

static ImplicationResult join(const ImplicationResult& a, const ImplicationResult& b)
{
    if (a.e == ImplicationResult::Falsified || b.e == ImplicationResult::Falsified)
        return {ImplicationResult::Falsified, 0};
    if (a.e == ImplicationResult::AntecedentHolds)
        return b;
    if (b.e == ImplicationResult::AntecedentHolds)
        return a;
    if (a.factor != b.factor)
        return {ImplicationResult::Falsified, 0};
    return a;
}

static ImplicationResult inductionImplicationHolds(
    Dictionary<ImplicationParams, ImplicationResult>& memo,
    IRInst* const prevVal,
    IRInst* const conditionVal,
    IRInst* const inductiveVal,
    IRBlock* const block);

static bool unpackConstantAddition(IRInst* addOrSub, IRInst*& operand, IRIntegerValue& constant)
{
    if (addOrSub->getOp() != kIROp_Add && addOrSub->getOp() != kIROp_Sub)
        return false;
    const bool negate = addOrSub->getOp() == kIROp_Sub;

    auto o = addOrSub->getOperand(0);
    auto c = addOrSub->getOperand(1);
    if (!as<IRIntLit>(c))
        std::swap(o, c);
    const auto cLit = as<IRIntLit>(c);
    if (!cLit)
        return false;
    operand = o;
    constant = cLit->getValue();
    // Check that we can actually represent this!
    if (negate && constant == std::numeric_limits<IRIntegerValue>::min())
        return false;
    constant *= negate ? -1 : 1;
    return true;
}

// isAdditionOf(m, i, p, c, f) returns true if it can prove `c => i = p + f`
// for some constant factor f
static bool isAdditionOf(
    Dictionary<ImplicationParams, ImplicationResult>& memo,
    IRInst* inductiveVal,
    IRInst* prevVal,
    IRInst* conditionVal,
    IRIntegerValue& factor)
{
    IRInst* operand;
    IRIntegerValue constant;
    if (!unpackConstantAddition(inductiveVal, operand, constant))
        return false;

    const auto impRes = inductionImplicationHolds(
        memo,
        prevVal,
        conditionVal,
        operand,
        as<IRBlock>(inductiveVal->getParent()));

    if (impRes.e == ImplicationResult::ConsequentHolds)
    {
        // TODO: Check for overflow here (strictly speaking it shouldn't
        // matter in the end numerically (except that this could be UB)).
        factor = impRes.factor + constant;
        return true;
    }
    return false;
}

// Returns true if we can prove that in this block this value is
// always false
static bool isAlwaysFalseInBlock(IRInst* inst, IRBlock* block)
{
    const auto b = as<IRBoolLit>(inst);
    if (b)
        return !b->getValue();

    // At the moment we just check that the predecessors of this
    // block have us on the false path of a conditional branch on
    // the instruction under question.
    bool isFalse = true;
    for (const auto predecessor : block->getPredecessors())
    {
        const auto predConditionalBranch = as<IRConditionalBranch>(predecessor->getTerminator());
        isFalse &= predConditionalBranch && predConditionalBranch->getCondition() == inst &&
                   predConditionalBranch->getFalseBlock() == block;
        if (!isFalse)
            break;
    }
    return isFalse;
}

// This function takes:
// - A block with an unconditional branch with at least one parameter 'i'
// - The index of 'i'
// - A condition variable 'c'
// - The predecessor case in the induction 'p'
//
// It returns true if at the time of the branch: 'isTrue(c) => isInductiveValue(i, p)'
// It return false if it can't prove this implication holds.
static ImplicationResult inductionImplicationHolds(
    Dictionary<ImplicationParams, ImplicationResult>& memo,
    IRInst* const prevVal,
    IRInst* const conditionVal,
    IRInst* const inductiveVal,
    IRBlock* const block)
{
    // If we have a result memoized we can safely return that.
    const ImplicationParams i = {conditionVal, inductiveVal, block};
    const auto memoized = memo.tryGetValue(i);
    if (memoized)
        return *memoized;

    // While we are detemining if the implication holds at this position we set
    // the result to Falsified so as to fail if we require a self-referential
    // proof
    memo.add(i, {ImplicationResult::Falsified, 0});
    // A helper to record the solution as we're returning
    const auto andRemember = [&memo, i](ImplicationResult r)
    {
        memo.set(i, r);
        return r;
    };

    // Our most general solution is if the left hand side of the implication is
    // false, in which case we can return success without specifying a factor
    if (isAlwaysFalseInBlock(conditionVal, block))
        return andRemember({ImplicationResult::AntecedentHolds, 0});

    // Otherwise, we handle the additive case
    // One easy case is that this *is* the previous value, in which case it's a
    // trivial solution with an addition of 0
    if (prevVal == inductiveVal)
        return andRemember({ImplicationResult::ConsequentHolds, 0});

    // Otherwise is it a function over the inductive variable
    IRIntegerValue factor;
    if (isAdditionOf(memo, inductiveVal, prevVal, conditionVal, factor))
        return andRemember({ImplicationResult::ConsequentHolds, factor});

    // The last thing to try is to consider the case where the
    // inductive value under consideration is a parameter, in that case we can
    // recurse into the predecessors of this block, replacing the parameter and
    // condition variables with their arguments where appropriate.
    const auto inductiveParam = as<IRParam>(inductiveVal);

    // If it's not a parameter then we don't know how to continue
    // (in principle we could also hadle instructions such as loads here)
    if (!inductiveParam)
        return {ImplicationResult::Falsified, 0};

    const auto conditionParam = as<IRParam>(conditionVal);
    const auto inductiveParamIndex = block->getParamIndex(inductiveParam);
    const auto conditionParamIndex = block->getParamIndex(conditionParam);

    // If we have no predecessors, then all the possible values (none) of the
    // condition variable are false, so our antecedent holds
    ImplicationResult res = {ImplicationResult::AntecedentHolds, 0};

    for (const auto predecessor : block->getPredecessors())
    {
        const auto predTerminator = as<IRUnconditionalBranch>(predecessor->getTerminator());
        SLANG_ASSERT(inductiveParamIndex == -1 || predTerminator);
        SLANG_ASSERT(conditionParamIndex == -1 || predTerminator);

        const auto nextInductiveParam = inductiveParamIndex == -1
                                            ? inductiveParam
                                            : predTerminator->getArg(inductiveParamIndex);
        const auto nextConditionParam = conditionParamIndex == -1
                                            ? conditionParam
                                            : predTerminator->getArg(conditionParamIndex);

        const auto predResult = inductionImplicationHolds(
            memo,
            prevVal,
            nextConditionParam,
            nextInductiveParam,
            predecessor);
        res = join(res, predResult);

        if (res.e == ImplicationResult::Falsified)
            break;
    }

    return andRemember(res);
}

void AutodiffCheckpointPolicyBase::collectInductionValues(IRGlobalValueWithCode* func)
{
    // Collect loop induction values.
    // There are two special phi insts we want to handle differently in our
    // checkpointing policy:
    // 1. a bool execution flag inserted as the result of CFG normalization,
    //    that is always true as long as the loop is still active.
    // 2. the original induction variable that can be replaced with the loop
    //    counter we inserted during createPrimalRecomputeBlocks().

    for (auto block : func->getBlocks())
    {
        auto loopInst = as<IRLoop>(block->getTerminator());
        if (!loopInst)
            continue;
        auto targetBlock = loopInst->getTargetBlock();
        auto ifElse = as<IRIfElse>(targetBlock->getTerminator());
        Int paramIndex = -1;
        Int conditionParamIndex = -1;
        // First, we are going to collect all the bool execution flags from loops.
        // These are very easy to identify: they are a phi param defined in
        // targetBlock, and used as the condition value in the condtion block.
        for (auto param : targetBlock->getParams())
        {
            paramIndex++;
            if (!param->getDataType())
                continue;
            if (param->getDataType()->getOp() == kIROp_BoolType)
            {
                if (ifElse->getCondition() == param)
                {
                    // The bool param is used as the condition of the if-else inside the loop,
                    // this param will always be true during the loop, and we don't need to store
                    // it.
                    LoopInductionValueInfo info;
                    info.kind = LoopInductionValueInfo::Kind::AlwaysTrue;
                    inductionValueInsts[param] = info;
                    conditionParamIndex = paramIndex;
                }
            }
        }
        if (conditionParamIndex == -1)
            continue;

        // Next we try to identify any induction variables.
        //
        // An inductive parameter must:
        // - Be initialized as anything from a single predecessor, the base
        //   case
        // - Be passed a function of itself only on any other entries to
        //   the loop, the inductive case
        //
        // In terms of matching here, we allow the base case to be
        // anything, and the inductive case to be the successor function
        //
        // We also handle the case where something other than the base or
        // inductive case is passed to the top of the loop when the "condition
        // parameter" is false, in which case the "non-inductive" value isn't
        // actually used.

        paramIndex = -1;
        for (auto param : targetBlock->getParams())
        {
            paramIndex++;

            const auto t = param->getDataType();
            if (!t || !isScalarIntegerType(t))
                continue;

            // This *is* the loop counter!
            if (param->findDecoration<IRLoopCounterDecoration>())
                continue;

            auto predecessors = targetBlock->getPredecessors();
            Dictionary<ImplicationParams, ImplicationResult> memo;
            ImplicationResult impRes = {ImplicationResult::AntecedentHolds, 0};
            for (const auto predecessor : predecessors)
            {
                // Since this is branching with a parameter, it can only be an
                // unconditional branch.
                const auto predTerminator = as<IRUnconditionalBranch>(predecessor->getTerminator());
                SLANG_ASSERT(predTerminator);

                // Is this the base case?
                if (predTerminator == loopInst)
                    continue;

                const auto conditionArg = predTerminator->getArg(conditionParamIndex);
                const auto inductiveArg = predTerminator->getArg(paramIndex);

                // Check that the required implication holds for this block
                const auto predRes =
                    inductionImplicationHolds(memo, param, conditionArg, inductiveArg, predecessor);
                impRes = join(impRes, predRes);
                if (impRes.e == ImplicationResult::Falsified)
                    break;
            }

            switch (impRes.e)
            {
            // This wasn't an induction variable
            case ImplicationResult::Falsified:
                break;

            // The loop doesn't loop (because in every case the break flag is
            // true)
            case ImplicationResult::AntecedentHolds:
                break;

            case ImplicationResult::ConsequentHolds:
                {
                    // The use of the add inst matches all of our conditions as an induction value
                    // that is a constant offset from a multiple of the loop counter.
                    LoopInductionValueInfo info;
                    info.kind = LoopInductionValueInfo::Kind::AffineFunctionOfCounter;
                    info.loopInst = loopInst;
                    info.counterOffset = loopInst->getArg(paramIndex);
                    info.counterFactor = impRes.factor;
                    inductionValueInsts[param] = info;
                }
            }
        }
    }
}

static bool isValueInRange(IRIntegerValue value, IRType* type)
{
    IRInst* innerType = unwrapAttributedType(type);
    IRIntegerValue nBits;
    bool isSigned;

    switch (innerType->getOp())
    {
    case kIROp_IntType:
    case kIROp_UIntType:
        nBits = 32;
        break;
    case kIROp_Int16Type:
    case kIROp_UInt16Type:
        nBits = 16;
        break;
    case kIROp_Int8Type:
    case kIROp_UInt8Type:
        nBits = 8;
        break;
    case kIROp_Int64Type:
    case kIROp_UInt64Type:
        nBits = 64;
        break;
    default:
        return false;
    }

    switch (innerType->getOp())
    {
    case kIROp_IntType:
    case kIROp_Int16Type:
    case kIROp_Int8Type:
    case kIROp_Int64Type:
        isSigned = true;
        break;
    case kIROp_UIntType:
    case kIROp_UInt16Type:
    case kIROp_UInt8Type:
    case kIROp_UInt64Type:
        isSigned = false;
        break;
    default:
        return false;
    }

    if (nBits >= 64)
    {
        // IRIntegerValue is 64-bit, so we assume we can always represent the value.
        // TODO: Corner cases like loops that _rely_ on 64-bit integer overflow might not be handled
        // correctly.
        //
        return true;
    }

    if (isSigned)
    {
        IRIntegerValue maxValue = (1ULL << (nBits - 1)) - 1;
        return value >= -maxValue && value <= maxValue;
    }
    else
    {
        IRIntegerValue maxValue = (1ULL << nBits) - 1;
        return value >= 0 && value <= maxValue;
    }
}

void AutodiffCheckpointPolicyBase::collectLoopExitConditions(IRGlobalValueWithCode* func)
{
    // Assume that the InductionValueInfo is already collected.
    IRBuilder builder(func->getModule());
    RefPtr<IRDominatorTree> domTree = computeDominatorTree(func);
    for (auto block : func->getBlocks())
    {
        auto loopInst = as<IRLoop>(block->getTerminator());
        if (!loopInst)
            continue;
        auto targetBlock = loopInst->getTargetBlock();
        auto ifElse = as<IRIfElse>(targetBlock->getTerminator());
        if (!ifElse)
            continue;

        auto condParam = as<IRParam>(ifElse->getCondition());
        if (!condParam || condParam->getParent() != targetBlock)
            continue;

        // Locate the loop counter.
        IRInst* loopCounter = nullptr;
        for (auto param : targetBlock->getParams())
        {
            if (param->findDecoration<IRLoopCounterDecoration>())
            {
                loopCounter = param;
                break;
            }
        }

        if (!loopCounter)
            continue;

        // Go over all loop phi parameters for which we have induction value info,
        // and try to determine a relation on the exit value.
        //
        for (auto param : targetBlock->getParams())
        {
            auto inductionValueInfo = inductionValueInsts.tryGetValue(param);
            if (!inductionValueInfo ||
                inductionValueInfo->kind != LoopInductionValueInfo::AffineFunctionOfCounter)
                continue;

            // We need to have a known constant offset to be able to compute the loop exit value.
            if (!isIntegerConstantValue(inductionValueInfo->counterOffset))
                continue;

            StatementSet conditionIsFalse;
            conditionIsFalse.conjunct(condParam, SimpleRelation::boolRelation(false));

            // Collect a statement that holds when the loop condition is false.
            const auto implicationsForFalseCondition =
                collectImplications(domTree, targetBlock, conditionIsFalse);

            if (!implicationsForFalseCondition.statements.containsKey(param))
            {
                // The statement we collected says nothing about the parameter. No point continuing.
                continue;
            }

            // Collect statements for the inverse.. i.e. some relation that holds if the condition
            // is true.
            StatementSet conditionIsTrue;
            conditionIsTrue.conjunct(condParam, SimpleRelation::boolRelation(true));
            const auto implicationsForTrueCondition =
                collectImplications(domTree, targetBlock, conditionIsTrue);

            if (!implicationsForTrueCondition.statements.containsKey(param))
            {
                // The statement we collected says nothing about the parameter. No point continuing.
                continue;
            }

            // Extract A s.t. ~breakFlag => A.
            //
            // (Note that breakFlag == false is the case where the
            // loop exits)
            //
            SimpleRelation statement = implicationsForFalseCondition.statements.getValue(param);

            // Extract B s.t. breakFlag => B
            SimpleRelation inverseStatement =
                implicationsForTrueCondition.statements.getValue(param);

            // If A => ~B, then by using the contrapositive, we get A <=> ~breakFlag
            if (!doesRelationImply(statement, inverseStatement.negated()))
            {
                // If the above doesn't work, we can try using ~B instead.
                if (!doesRelationImply(inverseStatement.negated(), statement))
                    continue; // Neither works.. we can't infer anything about param.
                else
                    statement = inverseStatement.negated(); // Use ~B <=> ~breakFlag
            }

            // We found a relation on the parameter at the loop exit, and we also proved that
            // if the relation holds, the loop must exit.
            //
            // If we have an inequality + information that a value is an inductive (i.e. follows a
            // sequence of the form `start + i * step`), then we can use that to compute the exact
            // value at the loop exit.
            //
            // We can do this by solving the inequality for the parameter, using the inductive value
            // as the counter variable.
            //
            if (inductionValueInfo->kind == LoopInductionValueInfo::Kind::AffineFunctionOfCounter)
            {
                auto counterOffset = getConstantIntegerValue(inductionValueInfo->counterOffset);
                auto counterFactor = inductionValueInfo->counterFactor;

                SLANG_ASSERT(statement.type == SimpleRelation::Type::IntegerRelation);
                auto relationValue = statement.integerValue;

                auto recordExitValue = [&](IRIntegerValue exitIValue, IRIntegerValue exitParamValue)
                {
                    // TODO: Maybe we should warn if the inferred exit value is out of range?
                    if (isValueInRange(exitParamValue, param->getDataType()))
                    {
                        this->loopExitValueInsts[param] =
                            builder.getIntValue(param->getDataType(), exitParamValue);
                    }

                    // The interesting part is that since we know that this variable is an bijective
                    // function of the loop counter, we can also compute the loop counter's exit
                    // value.
                    //
                    // Since this can come from multiple parameters, we'll verify to make sure that
                    // there are no contradictions.
                    //
                    IRInst* loopCounterExitValue;
                    if (this->loopExitValueInsts.tryGetValue(loopCounter, loopCounterExitValue))
                    {
                        auto loopCounterExitIValue = getConstantIntegerValue(loopCounterExitValue);
                        if (loopCounterExitIValue != exitIValue)
                        {
                            SLANG_ASSERT(!"contradictory loop exit values");
                        }
                    }
                    else
                    {
                        // TODO: Maybe we should warn if the inferred exit value is out of range?
                        if (isValueInRange(exitIValue, loopCounter->getDataType()))
                        {
                            this->loopExitValueInsts[loopCounter] =
                                builder.getIntValue(loopCounter->getDataType(), exitIValue);
                        }
                    }
                };

                if (counterFactor > 0 && statement.comparator == SimpleRelation::GreaterThanEqual)
                {
                    // Find the smallest value that satisfies counterFactor * i + counterOffset >=
                    // relationValue
                    //
                    IRIntegerValue exitIValue =
                        (((relationValue - counterOffset) + counterFactor - 1) / counterFactor);
                    IRIntegerValue exitParamValue =
                        counterOffset + counterFactor * (exitIValue - 1);
                    recordExitValue(exitIValue, exitParamValue);
                }
                else if (counterFactor < 0 && statement.comparator == SimpleRelation::LessThanEqual)
                {
                    // Find the largest value that satisfies counterFactor * i + counterOffset <=
                    // relationValue
                    //
                    IRIntegerValue exitIValue =
                        ((relationValue - counterOffset) + (counterFactor + 1)) / counterFactor;
                    IRIntegerValue exitParamValue = counterOffset + counterFactor * exitIValue;
                    recordExitValue(exitIValue, exitParamValue);
                }
                // TODO: handle other cases
            }
        }
    }
}

void applyToInst(
    IRBuilder* builder,
    CheckpointSetInfo* checkpointInfo,
    HoistedPrimalsInfo* hoistInfo,
    IROutOfOrderCloneContext* cloneCtx,
    Dictionary<IRBlock*, List<IndexTrackingInfo>>& blockIndexInfo,
    IRInst* inst)
{
    // Early-out..
    if (checkpointInfo->storeSet.contains(inst))
    {
        hoistInfo->storeSet.add(inst);
        return;
    }


    bool isInstRecomputed = checkpointInfo->recomputeSet.contains(inst);
    if (isInstRecomputed)
    {
        if (auto loopExitValueInst = as<IRLoopExitValue>(inst))
        {
            if (auto loopExitValue =
                    checkpointInfo->loopExitValueInsts.tryGetValue(loopExitValueInst->getVal()))
            {
                cloneCtx->cloneEnv.mapOldValToNew[inst] = *loopExitValue;
                cloneCtx->registerClonedInst(builder, inst, *loopExitValue);
                return;
            }

            // Should never happen. (Can't mark a LoopExitValue inst as recomputed without having an
            // entry in loopExitValueInsts dict)
            //
            SLANG_ASSERT(!"no loop exit value found for inst");
        }

        if (as<IRParam>(inst))
        {
            // Can completely ignore first block parameters
            if (getBlock(inst) == getBlock(inst)->getParent()->getFirstBlock())
            {
                return;
            }
            // If this is loop condition, it is always true in reverse blocks.
            LoopInductionValueInfo inductionValueInfo;
            if (checkpointInfo->loopInductionInfo.tryGetValue(inst, inductionValueInfo))
            {
                IRInst* replacement = nullptr;
                if (inductionValueInfo.kind == LoopInductionValueInfo::Kind::AlwaysTrue)
                {
                    replacement = builder->getBoolValue(true);
                }
                else if (
                    inductionValueInfo.kind ==
                    LoopInductionValueInfo::Kind::AffineFunctionOfCounter)
                {
                    auto indexInfo =
                        blockIndexInfo.tryGetValue(inductionValueInfo.loopInst->getTargetBlock());
                    SLANG_ASSERT(indexInfo);
                    SLANG_ASSERT(indexInfo->getCount() != 0);
                    replacement = indexInfo->getFirst().diffCountParam;
                    if (inductionValueInfo.counterFactor != 1)
                    {
                        setInsertAfterOrdinaryInst(builder, replacement);
                        replacement = builder->emitMul(
                            replacement->getDataType(),
                            replacement,
                            builder->getIntValue(
                                replacement->getDataType(),
                                inductionValueInfo.counterFactor));
                    }
                    if (inductionValueInfo.counterOffset)
                    {
                        setInsertAfterOrdinaryInst(builder, replacement);
                        replacement = builder->emitAdd(
                            replacement->getDataType(),
                            replacement,
                            inductionValueInfo.counterOffset);
                    }
                }
                SLANG_ASSERT(replacement);

                // If the replacement and inst are not the exact same type, use an int-cast
                // (e.g. uint vs. int)
                //
                if (replacement->getDataType() != inst->getDataType())
                {
                    setInsertAfterOrdinaryInst(builder, replacement);
                    replacement = builder->emitCast(inst->getDataType(), replacement);
                }

                cloneCtx->cloneEnv.mapOldValToNew[inst] = replacement;
                cloneCtx->registerClonedInst(builder, inst, replacement);
                return;
            }
        }

        auto recomputeInst = cloneCtx->cloneInstOutOfOrder(builder, inst);
        hoistInfo->recomputeSet.add(recomputeInst);
    }

    bool isInstInverted = checkpointInfo->invertSet.contains(inst);
    if (isInstInverted)
    {
        InversionInfo info = checkpointInfo->invInfoMap[inst];
        auto clonedInstToInvert = cloneCtx->cloneInstOutOfOrder(builder, info.instToInvert);

        // Process operand set for the inverse inst.
        List<IRInst*> newOperands;
        for (auto operand : info.requiredOperands)
        {
            if (cloneCtx->cloneEnv.mapOldValToNew.containsKey(operand))
                newOperands.add(cloneCtx->cloneEnv.mapOldValToNew[operand]);
            else
                newOperands.add(operand);
        }

        info.requiredOperands = newOperands;

        hoistInfo->invertInfoMap[clonedInstToInvert] = info;
        hoistInfo->instsToInvert.add(clonedInstToInvert);
        hoistInfo->invertSet.add(cloneCtx->cloneInstOutOfOrder(builder, inst));
    }
}

static IRBlock* getParamPreludeBlock(IRGlobalValueWithCode* func)
{
    return func->getFirstBlock()->getNextBlock();
}

void applyCheckpointSet(
    CheckpointSetInfo* checkpointInfo,
    IRGlobalValueWithCode* func,
    HoistedPrimalsInfo* hoistInfo,
    HashSet<IRUse*>& pendingUses,
    Dictionary<IRBlock*, IRBlock*>& mapPrimalBlockToRecomputeBlock,
    IROutOfOrderCloneContext* cloneCtx,
    Dictionary<IRBlock*, List<IndexTrackingInfo>>& blockIndexInfo)
{
    for (auto use : pendingUses)
        cloneCtx->pendingUses.add(use);

    // Go back over the insts and move/clone them accoridngly.
    auto paramPreludeBlock = getParamPreludeBlock(func);
    for (auto block : func->getBlocks())
    {
        // Skip parameter block and the param prelude block.
        if (block == func->getFirstBlock() || block == paramPreludeBlock)
            continue;

        if (isDifferentialBlock(block))
            continue;

        if (block->findDecoration<IRRecomputeBlockDecoration>())
            continue;

        IRBlock* recomputeBlock = block;
        mapPrimalBlockToRecomputeBlock.tryGetValue(block, recomputeBlock);
        auto recomputeInsertBeforeInst = recomputeBlock->getFirstOrdinaryInst();

        IRBuilder builder(func->getModule());
        UIndex ii = 0;
        for (auto param : block->getParams())
        {
            builder.setInsertBefore(recomputeInsertBeforeInst);
            bool isRecomputed = checkpointInfo->recomputeSet.contains(param);
            bool isInverted = checkpointInfo->invertSet.contains(param);
            bool loopInductionInfo = checkpointInfo->loopInductionInfo.tryGetValue(param);
            if (!isRecomputed && !isInverted)
            {
                ii++;
                continue;
            }

            if (!loopInductionInfo)
            {
                SLANG_RELEASE_ASSERT(
                    recomputeBlock != block &&
                    "recomputed param should belong to block that has recompute block.");
            }

            // Apply checkpoint rule to the parameter itself.
            applyToInst(&builder, checkpointInfo, hoistInfo, cloneCtx, blockIndexInfo, param);

            if (loopInductionInfo)
            {
                ii++;
                continue;
            }

            // Copy primal branch-arg for predecessor blocks.
            HashSet<IRBlock*> predecessorSet;
            for (auto predecessor : block->getPredecessors())
            {
                if (predecessorSet.contains(predecessor))
                    continue;
                predecessorSet.add(predecessor);

                auto primalPhiArg =
                    as<IRUnconditionalBranch>(predecessor->getTerminator())->getArg(ii);
                auto recomputePredecessor = mapPrimalBlockToRecomputeBlock.getValue(predecessor);

                // For now, find the primal phi argument in this predecessor,
                // and stick it into the recompute predecessor's branch inst. We
                // will use a patch-up pass in the end to replace all these
                // arguments to their recomputed versions if they exist.

                if (isRecomputed)
                {
                    IRInst* terminator = recomputeBlock->getTerminator();
                    addPhiOutputArg(&builder, recomputePredecessor, terminator, primalPhiArg);
                }
                else if (isInverted)
                {
                    IRInst* terminator = recomputeBlock->getTerminator();
                    addPhiOutputArg(&builder, recomputePredecessor, terminator, primalPhiArg);
                }
            }
            ii++;
        }


        for (auto child : block->getChildren())
        {
            // Determine the insertion point for the recomputeInst.
            // Normally we insert recomputeInst into the block's corresponding recomputeBlock.
            // The exception is a load(inoutParam), in which case we insert the recomputed load
            // at the right beginning of the function to correctly receive the initial parameter
            // value. We can't just insert the load at recomputeBlock because at that point the
            // primal logic may have already updated the param with a new value, and instead we
            // want the original value.
            builder.setInsertBefore(recomputeInsertBeforeInst);
            applyToInst(&builder, checkpointInfo, hoistInfo, cloneCtx, blockIndexInfo, child);
        }
    }

    // Go through phi arguments in recompute blocks and replace them to
    // recomputed insts if they exist.
    for (auto block : func->getBlocks())
    {
        if (!block->findDecoration<IRRecomputeBlockDecoration>())
            continue;
        auto terminator = block->getTerminator();
        for (UInt i = 0; i < terminator->getOperandCount(); i++)
        {
            auto arg = terminator->getOperand(i);
            if (as<IRBlock>(arg))
                continue;
            if (auto recomputeArg = cloneCtx->cloneEnv.mapOldValToNew.tryGetValue(arg))
            {
                terminator->setOperand(i, *recomputeArg);
            }
        }
    }
}

IRType* getTypeForLocalStorage(
    IRBuilder* builder,
    IRType* storageType,
    const List<IndexTrackingInfo>& defBlockIndices)
{
    for (auto& index : defBlockIndices)
    {
        SLANG_ASSERT(index.status == IndexTrackingInfo::CountStatus::Static);
        SLANG_ASSERT(index.maxIters >= 0);

        storageType = builder->getArrayType(
            storageType,
            builder->getIntValue(builder->getUIntType(), index.maxIters + 1));
    }

    return storageType;
}

IRVar* emitIndexedLocalVar(
    IRBlock* varBlock,
    IRType* baseType,
    const List<IndexTrackingInfo>& defBlockIndices,
    SourceLoc location)
{
    // Cannot store pointers. Case should have been handled by now.
    SLANG_RELEASE_ASSERT(!asRelevantPtrType(baseType));

    // Cannot store types. Case should have been handled by now.
    SLANG_RELEASE_ASSERT(!as<IRTypeType>(baseType));

    IRBuilder varBuilder(varBlock->getModule());
    IRBuilderSourceLocRAII sourceLocationScope(&varBuilder, location);

    varBuilder.setInsertBefore(varBlock->getFirstOrdinaryInst());

    IRType* varType = getTypeForLocalStorage(&varBuilder, baseType, defBlockIndices);

    auto var = varBuilder.emitVar(varType);
    varBuilder.emitStore(var, varBuilder.emitDefaultConstruct(varType));

    return var;
}

IRInst* emitIndexedStoreAddressForVar(
    IRBuilder* builder,
    IRVar* localVar,
    const List<IndexTrackingInfo>& defBlockIndices)
{
    IRInst* storeAddr = localVar;
    for (auto& index : defBlockIndices)
    {
        storeAddr = builder->emitElementAddress(storeAddr, index.primalCountParam);
    }

    return storeAddr;
}


IRInst* emitIndexedLoadAddressForVar(
    IRBuilder* builder,
    IRVar* localVar,
    IRBlock* defBlock,
    const List<IndexTrackingInfo>& defBlockIndices,
    const List<IndexTrackingInfo>& useBlockIndices)
{
    IRInst* loadAddr = localVar;

    for (auto index : defBlockIndices)
    {
        if (useBlockIndices.contains(index))
        {
            // If the use-block is under the same region, use the
            // differential counter variable
            //
            auto diffCounterCurrValue = index.diffCountParam;

            loadAddr = builder->emitElementAddress(loadAddr, diffCounterCurrValue);
        }
        else
        {
            // If the use-block is outside this region, use the
            // last available value (by indexing with primal counter minus 1)
            // An exception is if the stored inst is in a loop header block where
            // we use counter directly (since that block runs N+1 times)
            //
            auto primalCounterCurrValue = index.primalCountParam;
            auto primalCounterLastValue = (index.loopHeaderBlock == defBlock)
                                              ? primalCounterCurrValue
                                              : builder->emitSub(
                                                    primalCounterCurrValue->getDataType(),
                                                    primalCounterCurrValue,
                                                    builder->getIntValue(builder->getIntType(), 1));

            loadAddr = builder->emitElementAddress(loadAddr, primalCounterLastValue);
        }
    }

    return loadAddr;
}

IRVar* storeIndexedValue(
    IRBuilder* builder,
    IRBlock* defaultVarBlock,
    IRInst* instToStore,
    const List<IndexTrackingInfo>& defBlockIndices)
{
    IRVar* localVar = emitIndexedLocalVar(
        defaultVarBlock,
        instToStore->getDataType(),
        defBlockIndices,
        instToStore->sourceLoc);

    IRInst* addr = emitIndexedStoreAddressForVar(builder, localVar, defBlockIndices);

    builder->emitStore(addr, instToStore);

    return localVar;
}

IRInst* loadIndexedValue(
    IRBuilder* builder,
    IRVar* localVar,
    IRBlock* defBlock,
    const List<IndexTrackingInfo>& defBlockIndices,
    const List<IndexTrackingInfo>& useBlockIndices)
{
    IRInst* addr =
        emitIndexedLoadAddressForVar(builder, localVar, defBlock, defBlockIndices, useBlockIndices);

    return builder->emitLoad(addr);
}

bool areIndicesEqual(
    const List<IndexTrackingInfo>& indicesA,
    const List<IndexTrackingInfo>& indicesB)
{
    if (indicesA.getCount() != indicesB.getCount())
        return false;

    for (Index ii = 0; ii < indicesA.getCount(); ii++)
    {
        if (indicesA[ii].primalCountParam != indicesB[ii].primalCountParam)
            return false;
    }

    return true;
}

static int getInstRegionNestLevel(
    Dictionary<IRBlock*, List<IndexTrackingInfo>>& indexedBlockInfo,
    IRBlock* defBlock,
    IRInst* inst)
{
    auto result = indexedBlockInfo.getValue(defBlock).getCount();
    // Loop counters are considered to not belong to the region started by the its loop.
    if (result > 0 && inst->findDecoration<IRLoopCounterDecoration>())
        result--;
    return (int)result;
}


struct UseChain
{
    // The chain of uses from the base use to the relevant use.
    // However, this is stored in reverse order (so that the last use is the 'base use')
    //
    List<IRUse*> chain;

    static List<UseChain> from(
        IRUse* baseUse,
        Func<bool, IRUse*> isRelevantUse,
        Func<bool, IRInst*> passthroughInst)
    {
        IRInst* inst = baseUse->getUser();

        // Base case 1: we hit a relevant use, return a single-element chain.
        if (isRelevantUse(baseUse))
        {
            UseChain baseUseChain;
            baseUseChain.chain.add(baseUse);

            return List<UseChain>(UseChain(baseUseChain));
        }

        // Base case 2: we hit an irrelevant use that is not also a passthrough.
        // so stop here.
        if (!passthroughInst(inst))
        {
            return List<UseChain>();
        }

        // Recurse.
        List<UseChain> result;
        for (auto use = inst->firstUse; use; use = use->nextUse)
        {
            List<UseChain> innerChain = from(use, isRelevantUse, passthroughInst);

            for (auto& useChain : innerChain)
            {
                useChain.chain.add(baseUse);
                result.add(useChain);
            }
        }

        return result;
    }

    // This function only replaces the inner links, not the base use.
    void replaceInnerLinks(IROutOfOrderCloneContext* ctx, IRBuilder* builder)
    {
        SLANG_ASSERT(chain.getCount() > 0);

        const UIndex count = chain.getCount();

        // Process the chain in reverse order (excluding the first and last elements).
        // That is, iterate from count - 2 down to 1 (inclusive).
        for (int i = ((int)count) - 2; i >= 1; i--)
        {
            IRUse* use = chain[i];
            ctx->cloneInstOutOfOrder(builder, use->get());
        }
    }

    IRInst* getUser() const
    {
        SLANG_ASSERT(chain.getCount() > 0);
        return chain.getFirst()->getUser();
    }
};

struct UseGraph
{
    // Set of linear paths to the base use.
    // Note that some nodes may be common to multiple paths.
    //
    OrderedDictionary<IRUse*, List<UseChain>> chainSets;

    // Create a UseGraph from a base inst.
    //
    // `isRelevantUse` is a predicate that determines if a use is relevant. Traversal will stop at
    // this use, and all chains to this use will be grouped together.
    //
    // `passthroughInst` is a predicate that determines if an inst should be looked through
    // for uses.
    //
    static UseGraph from(
        IRInst* baseInst,
        Func<bool, IRUse*> isRelevantUse,
        Func<bool, IRInst*> passthroughInst)
    {
        UseGraph result;
        for (auto use = baseInst->firstUse; use;)
        {
            auto nextUse = use->nextUse;

            auto chains = UseChain::from(use, isRelevantUse, passthroughInst);
            for (auto& chain : chains)
            {
                auto finalUse = chain.chain.getFirst();

                if (!result.chainSets.containsKey(finalUse))
                {
                    result.chainSets[finalUse] = List<UseChain>();
                }

                result.chainSets[finalUse].getValue().add(chain);
            }

            use = nextUse;
        }
        return result;
    }

    void replace(IRBuilder* builder, IRUse* relevantUse, IRInst* inst)
    {
        // Since we may have common nodes, we will use an out-of-order cloning context
        // that can retroactively correct the uses as needed.
        //
        IROutOfOrderCloneContext ctx;
        List<UseChain> chains = chainSets[relevantUse];

        // Link the first use of each chain to inst.
        for (auto& chain : chains)
            ctx.cloneEnv.mapOldValToNew[chain.chain.getLast()->get()] = inst;

        // Process the inner links of each chain using the replacement.
        for (auto& chain : chains)
        {
            IRBuilder chainBuilder(builder->getModule());
            setInsertAfterOrdinaryInst(&chainBuilder, inst);

            chain.replaceInnerLinks(&ctx, builder);
        }

        // Finally, replace the relevant use (i.e, "final use") with the new replacement inst.
        builder->setInsertBefore(relevantUse->getUser());
        auto lastInstInChain = ctx.cloneInstOutOfOrder(builder, relevantUse->get());

        // Replace the base use.
        builder->replaceOperand(relevantUse, lastInstInChain);
    }

    List<IRUse*> getUniqueUses() const
    {
        List<IRUse*> result;

        for (auto& pair : chainSets)
        {
            result.add(pair.key);
        }

        return result;
    }
};


// Trim defBlockIndices based on the indices of out of scope uses.
//
static List<IndexTrackingInfo> maybeTrimIndices(
    const List<IndexTrackingInfo>& defBlockIndices,
    const Dictionary<IRBlock*, List<IndexTrackingInfo>>& indexedBlockInfo,
    const List<IRUse*>& outOfScopeUses)
{
    // Go through uses, lookup the defBlockIndices, and remove any indices if they
    // are not present in any of the uses. (This is sort of slow...)
    //
    List<IndexTrackingInfo> result;
    for (const auto& index : defBlockIndices)
    {
        bool found = false;
        for (const auto& use : outOfScopeUses)
        {
            auto useInst = use->getUser();
            auto useBlock = useInst->getParent();
            auto useBlockIndices = indexedBlockInfo.getValue(as<IRBlock>(useBlock));
            if (useBlockIndices.contains(index))
            {
                found = true;
                break;
            }
        }
        if (found)
            result.add(index);
    }
    return result;
}

/// Legalizes all accesses to primal insts from recompute and diff blocks.
///
RefPtr<HoistedPrimalsInfo> ensurePrimalAvailability(
    HoistedPrimalsInfo* hoistInfo,
    IRGlobalValueWithCode* func,
    Dictionary<IRBlock*, List<IndexTrackingInfo>>& indexedBlockInfo)
{
    // In general, after checkpointing, we can have a function like the following:
    // ```
    // void func()
    // {
    // primal:
    //      for (int i = 0; i < 5; i++)
    //      {
    //            float x = g(i);
    //            use(x);
    //      }
    // recompute:
    //      ...
    // diff:
    //      for (int i = 5; i >= 0; i--)
    //      {
    //      recompute:
    //          ...
    //      diff:
    //          use_diff(x); // def of x is not dominating this location!
    //      }
    // }
    // ```
    // This function will legalize the access to x in the dff block by creating
    // a proper local variable and insert store/loads, so that the above function
    // will be transformed to:
    // ```
    // void func()
    // {
    // primal:
    //      float x_storage[5];
    //
    //      for (int i = 0; i < 5; i++)
    //      {
    //            float x = g(i);
    //            x_storage[i] = x;
    //            use(x);
    //      }
    // recompute:
    //      ...
    // diff:
    //      for (int i = 5; i >= 0; i--)
    //      {
    //      recompute:
    //          ...
    //      diff:
    //          use_diff(x_storage[i]);
    //      }
    // }
    //

    RefPtr<IRDominatorTree> domTree = computeDominatorTree(func);

    IRBlock* defaultVarBlock = func->getFirstBlock()->getNextBlock();

    IRBuilder builder(func->getModule());

    IRBlock* defaultRecomptueVarBlock = nullptr;
    for (auto block : func->getBlocks())
        if (isDifferentialOrRecomputeBlock(block))
        {
            defaultRecomptueVarBlock = block;
            break;
        }
    SLANG_RELEASE_ASSERT(defaultRecomptueVarBlock);

    OrderedHashSet<IRInst*> processedStoreSet;

    auto ensureInstAvailable = [&](OrderedHashSet<IRInst*>& instSet, bool isRecomputeInst)
    {
        SLANG_ASSERT(!isDifferentialBlock(defaultVarBlock));

        List<IRInst*> workList;
        for (auto inst : instSet)
            workList.add(inst);

        HashSet<IRInst*> seenInstSet;
        while (workList.getCount() != 0)
        {
            auto instToStore = workList.getLast();
            workList.removeLast();

            if (seenInstSet.contains(instToStore))
                continue;

            IRBlock* defBlock = nullptr;
            if (auto varInst = as<IRVar>(instToStore))
            {
                auto storeUse = findEarliestUniqueWriteUse(varInst);

                defBlock = getBlock(storeUse->getUser());
            }
            else
                defBlock = getBlock(instToStore);

            SLANG_RELEASE_ASSERT(defBlock);

            // Lambda to check if a use is relevant.
            auto isRelevantUse = [&](IRUse* use)
            {
                // Only consider uses in differential blocks.
                // This method is not responsible for other blocks.
                //
                IRBlock* userBlock = getBlock(use->getUser());
                if (isRecomputeInst)
                {
                    if (isDifferentialOrRecomputeBlock(userBlock))
                    {
                        if (!domTree->dominates(defBlock, userBlock))
                        {
                            return true;
                        }
                        else if (!areIndicesSubsetOf(
                                     indexedBlockInfo[defBlock],
                                     indexedBlockInfo[userBlock]))
                        {
                            return true;
                        }
                        else if (
                            getInstRegionNestLevel(indexedBlockInfo, defBlock, instToStore) > 0 &&
                            !isDifferentialOrRecomputeBlock(defBlock))
                        {
                            return true;
                        }
                        else if (
                            asRelevantPtrType(instToStore->getDataType()) &&
                            !isDifferentialOrRecomputeBlock(defBlock))
                        {
                            return true;
                        }
                    }
                }
                else
                {
                    if (isDifferentialOrRecomputeBlock(userBlock))
                        return true;
                }
                return false;
            };

            // Lambda to check if an inst is transparent. We lookup uses 'through' transparent
            // insts recursively.
            //
            auto isPassthroughInst = [&](IRInst* inst)
            {
                if (as<IRTerminatorInst>(inst))
                    return false;

                if (!canInstBeStored(inst))
                    return true;

                switch (inst->getOp())
                {
                case kIROp_GetSequentialID:
                case kIROp_ExtractExistentialValue:
                    return true;
                }

                return false;
            };

            UseGraph useGraph = UseGraph::from(instToStore, isRelevantUse, isPassthroughInst);

            List<IRUse*> outOfScopeUses = useGraph.getUniqueUses();

            if (outOfScopeUses.getCount() == 0)
            {
                if (!isRecomputeInst)
                    processedStoreSet.add(instToStore);
                seenInstSet.add(instToStore);
                continue;
            }

            auto defBlockIndices = indexedBlockInfo.getValue(defBlock);
            IRBlock* varBlock = defaultVarBlock;
            if (isRecomputeInst)
            {
                varBlock = defaultRecomptueVarBlock;
                if (defBlockIndices.getCount())
                {
                    varBlock = as<IRBlock>(defBlockIndices[0].diffCountParam->getParent());
                    defBlockIndices.clear();
                }
            }
            if (IRVar* varToStore = as<IRVar>(instToStore))
            {
                auto storeUse = findLatestUniqueWriteUse(varToStore);

                bool isIndexedStore = (storeUse && defBlockIndices.getCount() > 0);

                // TODO: There's a slight hackiness here. (Ideally we might just want to emit
                // additional vars when splitting a call)
                //
                if (!isIndexedStore && isDerivativeContextVar(varToStore))
                {
                    varToStore->insertBefore(defaultVarBlock->getFirstOrdinaryInst());

                    if (!isRecomputeInst)
                        processedStoreSet.add(varToStore);
                    continue;
                }

                setInsertAfterOrdinaryInst(&builder, getInstInBlock(storeUse->getUser()));

                // There is an edge-case optimization we apply here,
                // If none of the out-of-scope uses are actually within the indexed
                // region, that means there's no need to allocate a fully indexed var.
                //
                defBlockIndices =
                    maybeTrimIndices(defBlockIndices, indexedBlockInfo, outOfScopeUses);

                IRVar* localVar = nullptr;
                {
                    IRBuilderSourceLocRAII sourceLocationScope(&builder, varToStore->sourceLoc);
                    localVar = storeIndexedValue(
                        &builder,
                        varBlock,
                        builder.emitLoad(varToStore),
                        defBlockIndices);
                }

                for (auto use : outOfScopeUses)
                {
                    setInsertBeforeOrdinaryInst(&builder, getInstInBlock(use->getUser()));

                    List<IndexTrackingInfo>& useBlockIndices =
                        indexedBlockInfo[getBlock(use->getUser())];

                    IRInst* loadAddr = emitIndexedLoadAddressForVar(
                        &builder,
                        localVar,
                        defBlock,
                        defBlockIndices,
                        useBlockIndices);

                    useGraph.replace(&builder, use, loadAddr);
                }

                if (!isRecomputeInst)
                    processedStoreSet.add(localVar);
            }
            else if (isPassthroughInst(instToStore))
            {
                // We won't actually process these insts here. Instead we'll
                // simply make sure that their operands are either already present
                // in the worklist or add them to the worklist for legalization.
                //

                List<IRInst*> pendingOperands;
                for (UIndex ii = 0; ii < instToStore->getOperandCount(); ii++)
                {
                    auto operand = instToStore->getOperand(ii);
                    if (!instSet.contains(operand) && !seenInstSet.contains(operand))
                    {
                        if (getBlock(operand) &&
                            (getBlock(operand)->getParent() == getBlock(instToStore)->getParent()))
                            pendingOperands.add(operand);
                    }
                }

                if (pendingOperands.getCount() > 0)
                {
                    for (Index ii = pendingOperands.getCount() - 1; ii >= 0; --ii)
                        workList.add(pendingOperands[ii]);
                }
            }
            else
            {
                IRBuilderSourceLocRAII sourceLocationScope(&builder, instToStore->sourceLoc);

                // Handle the special case of loop counters.
                // The only case where there will be a reference of primal loop counter from rev
                // blocks is the start of a loop in the reverse code. Since loop counters are not
                // considered a part of their loop region, so we remove the first index info.
                bool isLoopCounter =
                    (instToStore->findDecoration<IRLoopCounterDecoration>() != nullptr);
                if (isLoopCounter)
                {
                    defBlockIndices.removeAt(0);
                }
                else
                {
                    // For all others, check out of scope uses and trim indices if possible.
                    defBlockIndices =
                        maybeTrimIndices(defBlockIndices, indexedBlockInfo, outOfScopeUses);
                }

                setInsertAfterOrdinaryInst(&builder, instToStore);
                auto localVar = storeIndexedValue(&builder, varBlock, instToStore, defBlockIndices);
                if (isLoopCounter)
                    builder.addLoopCounterDecoration(localVar);

                for (auto use : outOfScopeUses)
                {
                    // TODO: Prevent terminator insts from being treated as passthrough..
                    List<IndexTrackingInfo> useBlockIndices =
                        indexedBlockInfo[getBlock(use->getUser())];
                    setInsertBeforeOrdinaryInst(&builder, getInstInBlock(use->getUser()));
                    useGraph.replace(
                        &builder,
                        use,
                        loadIndexedValue(
                            &builder,
                            localVar,
                            defBlock,
                            defBlockIndices,
                            useBlockIndices));
                }

                if (!isRecomputeInst)
                    processedStoreSet.add(localVar);
            }

            // Put the inst back on the worklist since there's a possibility that we created more
            // uses for it in the process.
            //
            // workList.add(instToStore);
            seenInstSet.add(instToStore);
        }
    };

    // Pull any loop counter in the store set out to another list.
    //
    Dictionary<UIndex, OrderedHashSet<IRInst*>> loopCounters;
    {
        List<IRInst*> loopCounterInsts;
        for (auto inst : hoistInfo->storeSet)
        {
            if (inst->findDecoration<IRLoopCounterDecoration>())
            {
                auto block = cast<IRBlock>(inst->getParent());
                auto nestDepth = indexedBlockInfo.getValue(block).getCount() - 1;

                if (!loopCounters.containsKey(nestDepth))
                    loopCounters[nestDepth] = OrderedHashSet<IRInst*>();

                loopCounters[nestDepth].add(inst);
                loopCounterInsts.add(inst);
            }
        }

        for (auto inst : loopCounterInsts)
            hoistInfo->storeSet.remove(inst);
    }

    // First handle all non-loop-counter insts.
    ensureInstAvailable(hoistInfo->storeSet, false);

    // Then handle the loop counter insts in reverse-order of nest depth
    // This ordering is important because loop counters at level N _may_ depend on
    // the counters at the previous levels.
    //
    for (Index ii = (Index)loopCounters.getCount() - 1; ii >= 0; --ii)
    {
        ensureInstAvailable(loopCounters[(UIndex)ii], false);
    }

    // Next handle all recompute insts, from within
    ensureInstAvailable(hoistInfo->recomputeSet, true);

    // Replace the old store set with the processed one.
    hoistInfo->storeSet = processedStoreSet;

    return hoistInfo;
}

void tryInferMaxIndex(IndexedRegion* region, IndexTrackingInfo* info)
{
    if (info->status != IndexTrackingInfo::CountStatus::Unresolved)
        return;

    auto loop = as<IRLoop>(region->getInitializerBlock()->getTerminator());

    if (auto maxItersDecoration = loop->findDecoration<IRLoopMaxItersDecoration>())
    {
        info->maxIters = (Count)maxItersDecoration->getMaxIters();
        info->status = IndexTrackingInfo::CountStatus::Static;
    }
}

IRInst* addPhiInputParam(IRBuilder* builder, IRBlock* block, IRType* type)
{
    builder->setInsertInto(block);
    return builder->emitParam(type);
}

IRInst* addPhiInputParam(IRBuilder* builder, IRBlock* block, IRType* type, UIndex index)
{
    List<IRParam*> params;
    for (auto param : block->getParams())
        params.add(param);

    SLANG_RELEASE_ASSERT(index == (UCount)params.getCount());

    return addPhiInputParam(builder, block, type);
}

static IRBlock* getUpdateBlock(IRLoop* loop)
{
    auto initBlock = cast<IRBlock>(loop->getParent());

    auto condBlock = loop->getTargetBlock();

    IRBlock* lastLoopBlock = nullptr;

    for (auto predecessor : condBlock->getPredecessors())
    {
        if (predecessor != initBlock)
            lastLoopBlock = predecessor;
    }

    // Should find atleast one predecessor that is _not_ the
    // init block (that contains the loop info). This
    // predecessor would be the last block in the loop
    // before looping back to the condition.
    //
    SLANG_RELEASE_ASSERT(lastLoopBlock);

    return lastLoopBlock;
}

void lowerIndexedRegion(
    IRLoop*& primalLoop,
    IRLoop*& diffLoop,
    IRInst*& primalCountParam,
    IRInst*& diffCountParam)
{
    IRBuilder builder(primalLoop);
    IRBuilderSourceLocRAII sourceLocationScope(&builder, primalLoop->sourceLoc);

    primalCountParam = nullptr;

    // Grab first primal block.
    IRBlock* primalInitBlock = as<IRBlock>(primalLoop->getParent());
    builder.setInsertBefore(primalInitBlock->getTerminator());
    {
        auto primalCondBlock =
            as<IRUnconditionalBranch>(primalInitBlock->getTerminator())->getTargetBlock();
        builder.setInsertBefore(primalInitBlock->getTerminator());

        auto phiCounterArgLoopEntryIndex = addPhiOutputArg(
            &builder,
            primalInitBlock,
            *(IRInst**)&primalLoop,
            builder.getIntValue(builder.getIntType(), 0));

        builder.setInsertBefore(primalCondBlock->getTerminator());
        primalCountParam = addPhiInputParam(
            &builder,
            primalCondBlock,
            builder.getIntType(),
            phiCounterArgLoopEntryIndex);
        builder.addLoopCounterDecoration(primalCountParam);
        builder.addNameHintDecoration(primalCountParam, UnownedStringSlice("_pc"));
        builder.markInstAsPrimal(primalCountParam);

        IRBlock* primalUpdateBlock = getUpdateBlock(primalLoop);
        IRInst* terminator = primalUpdateBlock->getTerminator();
        builder.setInsertBefore(primalUpdateBlock->getTerminator());

        auto incCounterVal = builder.emitAdd(
            builder.getIntType(),
            primalCountParam,
            builder.getIntValue(builder.getIntType(), 1));
        builder.markInstAsPrimal(incCounterVal);

        auto phiCounterArgLoopCycleIndex =
            addPhiOutputArg(&builder, primalUpdateBlock, terminator, incCounterVal);

        SLANG_RELEASE_ASSERT(phiCounterArgLoopEntryIndex == phiCounterArgLoopCycleIndex);
    }

    {
        IRBlock* diffInitBlock = as<IRBlock>(diffLoop->getParent());

        auto diffCondBlock =
            as<IRUnconditionalBranch>(diffInitBlock->getTerminator())->getTargetBlock();
        builder.setInsertBefore(diffInitBlock->getTerminator());
        auto revCounterInitVal = builder.emitSub(
            builder.getIntType(),
            primalCountParam,
            builder.getIntValue(builder.getIntType(), 1));
        auto phiCounterArgLoopEntryIndex =
            addPhiOutputArg(&builder, diffInitBlock, *(IRInst**)&diffLoop, revCounterInitVal);

        builder.setInsertBefore(diffCondBlock->getTerminator());

        diffCountParam = addPhiInputParam(
            &builder,
            diffCondBlock,
            builder.getIntType(),
            phiCounterArgLoopEntryIndex);
        builder.addNameHintDecoration(diffCountParam, UnownedStringSlice("_dc"));
        builder.markInstAsPrimal(diffCountParam);

        IRBlock* diffUpdateBlock = getUpdateBlock(diffLoop);
        builder.setInsertBefore(diffUpdateBlock->getTerminator());
        IRInst* terminator = diffUpdateBlock->getTerminator();

        auto decCounterVal = builder.emitSub(
            builder.getIntType(),
            diffCountParam,
            builder.getIntValue(builder.getIntType(), 1));
        builder.markInstAsPrimal(decCounterVal);

        auto phiCounterArgLoopCycleIndex =
            addPhiOutputArg(&builder, diffUpdateBlock, terminator, decCounterVal);

        auto ifElse = as<IRIfElse>(diffCondBlock->getTerminator());
        builder.setInsertBefore(ifElse);
        auto exitCondition =
            builder.emitGeq(diffCountParam, builder.getIntValue(builder.getIntType(), 0));
        ifElse->condition.set(exitCondition);

        SLANG_RELEASE_ASSERT(phiCounterArgLoopEntryIndex == phiCounterArgLoopCycleIndex);
    }
}

// Insert iteration counters for all loops to form indexed regions. For loops in
// primal blocks, the counter is incremented from 0. For loops in reverse
// blocks, the counter is decremented from the final value in primal block
// downto 0. Returns a mapping from each block to a list of their enclosing loop
// regions. A loop region records the iteration counter for the corresponding
// loop in the primal block and the reverse block.
//
void buildIndexedBlocks(
    Dictionary<IRBlock*, List<IndexTrackingInfo>>& info,
    IRGlobalValueWithCode* func)
{
    Dictionary<IRLoop*, IndexTrackingInfo> mapLoopToTrackingInfo;

    for (auto block : func->getBlocks())
    {
        auto loop = as<IRLoop>(block->getTerminator());
        if (!loop)
            continue;
        auto diffDecor = loop->findDecoration<IRDifferentialInstDecoration>();
        if (!diffDecor)
            continue;
        auto primalLoop = as<IRLoop>(diffDecor->getPrimalInst());
        if (!primalLoop)
            continue;

        IndexTrackingInfo indexInfo = {};
        lowerIndexedRegion(primalLoop, loop, indexInfo.primalCountParam, indexInfo.diffCountParam);

        indexInfo.loopHeaderBlock = getLoopConditionBlock(primalLoop);

        SLANG_RELEASE_ASSERT(indexInfo.primalCountParam);
        SLANG_RELEASE_ASSERT(indexInfo.diffCountParam);

        mapLoopToTrackingInfo[loop] = indexInfo;
        mapLoopToTrackingInfo[primalLoop] = indexInfo;
    }

    auto regionMap = buildIndexedRegionMap(func);

    for (auto block : func->getBlocks())
    {
        List<IndexTrackingInfo> trackingInfos;
        for (auto region : regionMap->getAllAncestorRegions(block))
        {
            IndexTrackingInfo trackingInfo;
            if (mapLoopToTrackingInfo.tryGetValue(region->loop, trackingInfo))
            {
                tryInferMaxIndex(region, &trackingInfo);
                trackingInfos.add(trackingInfo);
            }
        }
        info[block] = trackingInfos;
    }
}

// This function simply turns all CheckpointObject insts into a 'no-op'.
// i.e. simply replaces all uses of CheckpointObject with the original value.
//
// This operation is 'correct' because if CheckpointObject's operand is visible
// in a block, then it is visible in all dominated blocks.
//
void lowerCheckpointObjectInsts(IRGlobalValueWithCode* func)
{
    // For each block in the function
    for (auto block : func->getBlocks())
    {
        // For each instruction in the block
        for (auto inst = block->getFirstInst(); inst;)
        {
            // Get next inst before potentially removing current one
            auto nextInst = inst->getNextInst();

            // Check if this is a CheckpointObject instruction
            if (auto copyInst = as<IRCheckpointObject>(inst))
            {
                // Replace all uses of the copy with the original value
                auto originalVal = copyInst->getVal();
                copyInst->replaceUsesWith(originalVal);

                // Remove the now unused copy instruction
                inst->removeAndDeallocate();
            }

            if (auto loopExitValueInst = as<IRLoopExitValue>(inst))
            {
                auto originalVal = loopExitValueInst->getVal();
                loopExitValueInst->replaceUsesWith(originalVal);
                loopExitValueInst->removeAndDeallocate();
            }

            inst = nextInst;
        }
    }
}

// For each primal inst that is used in reverse blocks, decide if we should recompute or store
// its value, then make them accessible in reverse blocks based the decision.
//
RefPtr<HoistedPrimalsInfo> applyCheckpointPolicy(IRGlobalValueWithCode* func)
{
    sortBlocksInFunc(func);

    // Insert loop counters and establish loop regions.
    // Also makes the reverse loops counting downwards from the final iteration count.
    //
    Dictionary<IRBlock*, List<IndexTrackingInfo>> indexedBlockInfo;
    buildIndexedBlocks(indexedBlockInfo, func);

    // Split loop condition insts into two if necessary.
    splitLoopConditionBlockInsts(func, indexedBlockInfo);

    // Create recompute blocks for each region following the same control flow structure
    // as in primal code.
    //
    RefPtr<IROutOfOrderCloneContext> cloneCtx = new IROutOfOrderCloneContext();
    auto recomputeBlockMap = createPrimalRecomputeBlocks(func, indexedBlockInfo, cloneCtx);

    sortBlocksInFunc(func);

    // Dump IR.
    /*IRDumpOptions options;
    options.flags = IRDumpOptions::Flag::DumpDebugIds;
    options.mode = IRDumpOptions::Mode::Detailed;
    DiagnosticSinkWriter writer(sink);
    writer.write("### BEFORE-PROCESS-FUNC\n", strlen("### BEFORE-PROCESS-FUNC\n"));
    dumpIR(func, options, sink->getSourceManager(), &writer);*/

    // Determine the strategy we should use to make a primal inst available.
    // If we decide to recompute the inst, emit the recompute inst in the corresponding
    // recompute block.
    //
    RefPtr<AutodiffCheckpointPolicyBase> chkPolicy = new DefaultCheckpointPolicy(func->getModule());
    chkPolicy->preparePolicy(func);
    auto primalsInfo = chkPolicy->processFunc(func, recomputeBlockMap, cloneCtx, indexedBlockInfo);

    // Legalize the primal inst accesses by introducing local variables / arrays and emitting
    // necessary load/store logic.
    //
    auto hoistedPrimalsInfo = ensurePrimalAvailability(primalsInfo, func, indexedBlockInfo);

    // Lower CheckpointObject insts to a no-op.
    lowerCheckpointObjectInsts(func);

    return hoistedPrimalsInfo;
}

void DefaultCheckpointPolicy::preparePolicy(IRGlobalValueWithCode* func)
{
    SLANG_UNUSED(func)
    return;
}

enum CheckpointPreference
{
    None,
    PreferCheckpoint,
    PreferRecompute
};

static CheckpointPreference getCheckpointPreference(IRInst* callee)
{
    callee = getResolvedInstForDecorations(callee, true);
    for (auto decor : callee->getDecorations())
    {
        switch (decor->getOp())
        {
        case kIROp_PreferCheckpointDecoration:
            return CheckpointPreference::PreferCheckpoint;
        case kIROp_PreferRecomputeDecoration:
        case kIROp_TargetIntrinsicDecoration:
            return CheckpointPreference::PreferRecompute;
        }
    }
    return CheckpointPreference::None;
}

static bool isInstInPrimalOrTransposedParameterBlocks(IRInst* inst)
{
    auto func = getParentFunc(inst);
    if (!func)
        return false;
    auto firstBlock = func->getFirstBlock();
    if (inst->getParent() == firstBlock)
        return true;
    auto branch = as<IRUnconditionalBranch>(firstBlock->getTerminator());
    if (!branch)
        return false;
    auto secondBlock = branch->getTargetBlock();
    if (inst->getParent() == secondBlock)
        return true;
    return false;
}

static bool shouldStoreInst(IRInst* inst)
{
    if (!inst->getDataType())
    {
        return false;
    }

    if (!canTypeBeStored(inst->getDataType()))
        return false;

    switch (inst->getOp())
    {
    // Never store these opcodes because they are not real computations.
    case kIROp_CastFloatToInt:
    case kIROp_CastIntToFloat:
    case kIROp_IntCast:
    case kIROp_FloatCast:
    case kIROp_MakeVectorFromScalar:
    case kIROp_MakeMatrixFromScalar:
    case kIROp_Reinterpret:
    case kIROp_BitCast:
    case kIROp_DefaultConstruct:
    case kIROp_MakeStruct:
    case kIROp_MakeTuple:
    case kIROp_MakeArray:
    case kIROp_MakeVector:
    case kIROp_MakeMatrix:
    case kIROp_MakeArrayFromElement:
    case kIROp_MakeDifferentialPair:
    case kIROp_MakeDifferentialPairUserCode:
    case kIROp_MakeDifferentialPtrPair:
    case kIROp_MakeOptionalNone:
    case kIROp_MakeOptionalValue:
    case kIROp_MakeExistential:
    case kIROp_DifferentialPairGetDifferential:
    case kIROp_DifferentialPairGetPrimal:
    case kIROp_DifferentialPairGetDifferentialUserCode:
    case kIROp_DifferentialPairGetPrimalUserCode:
    case kIROp_DifferentialPtrPairGetDifferential:
    case kIROp_DifferentialPtrPairGetPrimal:
    case kIROp_ExtractExistentialValue:
    case kIROp_ExtractExistentialType:
    case kIROp_ExtractExistentialWitnessTable:
    case kIROp_undefined:
    case kIROp_GetSequentialID:
    case kIROp_GetStringHash:
    case kIROp_Specialize:
    case kIROp_LookupWitness:
    case kIROp_Param:
    case kIROp_DetachDerivative:
        return false;

    // Never store these op codes because they are trivial to compute.
    case kIROp_Add:
    case kIROp_Sub:
    case kIROp_Mul:
    case kIROp_Div:
    case kIROp_Neg:
    case kIROp_Geq:
    case kIROp_FRem:
    case kIROp_IRem:
    case kIROp_Leq:
    case kIROp_Neq:
    case kIROp_Eql:
    case kIROp_Greater:
    case kIROp_Less:
    case kIROp_And:
    case kIROp_Or:
    case kIROp_Not:
    case kIROp_BitNot:
    case kIROp_BitAnd:
    case kIROp_BitOr:
    case kIROp_BitXor:
    case kIROp_Lsh:
    case kIROp_Rsh:
    case kIROp_Select:
        return false;

    case kIROp_GetElement:
    case kIROp_FieldExtract:
    case kIROp_swizzle:
    case kIROp_UpdateElement:
    case kIROp_OptionalHasValue:
    case kIROp_GetOptionalValue:
    case kIROp_MatrixReshape:
    case kIROp_VectorReshape:
    case kIROp_GetTupleElement:
    case kIROp_LoopExitValue:
        return false;

    case kIROp_Load:
        // In general, don't store loads, because:
        //  - Loads to constant data can just be reloaded.
        //  - Loads to local variables can only exist for the temp variables used for calls,
        //    those variables are written only once so we can always load them anytime.
        //  - Loads to global mutable variables are now allowed, but we will capture that
        //    case in canRecompute().
        //  - The only exception is the load of an inout param, in which case we do need
        //    to store it because the param may be modified by the func at exit. Similarly,
        //    this will be handled in canRecompute().
        return false;

    case kIROp_Call:
        {
            // If the callee has a preference, we should follow it.
            if (getCheckpointPreference(inst->getOperand(0)) ==
                CheckpointPreference::PreferRecompute)
            {
                return false;
            }
            else if (
                getCheckpointPreference(inst->getOperand(0)) ==
                CheckpointPreference::PreferCheckpoint)
            {
                return true;
            }

            // If not, we'll default to recomputing calls that don't have side effects & don't
            // load from non-local variables. A previous data-flow pass should have already tagged
            // functions with the appropriate decorations.
            //
            auto callee = getResolvedInstForDecorations(inst->getOperand(0), true);
            if (callee->findDecoration<IRReadNoneDecoration>())
                return false;

            break;
        }
    case kIROp_CheckpointObject:
        // Special inst for when a value must be stored.
        return true;
    default:
        break;
    }

    if (as<IRType>(inst))
        return false;

    return true;
}

static bool shouldStoreVar(IRVar* var)
{
    if (const auto typeDecor = var->findDecoration<IRBackwardDerivativePrimalContextDecoration>())
    {
        // If we are specializing a callee's intermediate context with types that can't be stored,
        // we can't store the entire context.
        if (auto spec = as<IRSpecialize>(as<IRPtrTypeBase>(var->getDataType())->getValueType()))
        {
            for (UInt i = 0; i < spec->getArgCount(); i++)
            {
                if (!canTypeBeStored(spec->getArg(i)))
                    return false;
            }
        }
    }

    auto storeUse = findLatestUniqueWriteUse(var);
    if (storeUse)
    {
        if (!canTypeBeStored(as<IRPtrTypeBase>(var->getDataType())->getValueType()))
            return false;
        if (auto callUser = as<IRCall>(storeUse->getUser()))
        {
            // If the var is being written to by a call, the decision
            // of the var will be the same as the decision for the call.
            return shouldStoreInst(callUser);
        }
        // Default behavior is to recompute stuff.
        return false;
    }
    // If the var has never been written to, don't store it.
    return false;
}

bool DefaultCheckpointPolicy::canRecompute(UseOrPseudoUse use)
{
    if (auto load = as<IRLoad>(use.usedVal))
    {
        auto ptr = load->getPtr();

        // We can't recompute a `load` is if it is a load from a global mutable
        // variable.
        if (isGlobalOrUnknownMutableAddress(getParentFunc(load), ptr))
            return false;

        // We can't recompute a 'load' from a mutable function parameter.
        if (as<IRParam>(ptr) || as<IRVar>(ptr))
        {
            // An exception is a load of a constref parameter, which should
            // remain constant throughout the function.
            if (as<IRConstRefType>(getRootAddr(ptr)->getDataType()))
                return true;
            if (isInstInPrimalOrTransposedParameterBlocks(ptr))
                return false;
        }
    }
    else if (auto param = as<IRParam>(use.usedVal))
    {
        if (inductionValueInsts.containsKey(param))
            return true;

        // We can recompute a phi param if it is not in a loop start block.
        auto parentBlock = as<IRBlock>(param->getParent());
        for (auto pred : parentBlock->getPredecessors())
        {
            if (auto loop = as<IRLoop>(pred->getTerminator()))
            {
                if (loop->getTargetBlock() == parentBlock)
                    return false;
            }
        }
    }
    else if (auto exitValue = as<IRLoopExitValue>(use.usedVal))
    {
        if (loopExitValueInsts.containsKey(exitValue->getVal()))
            return true;
        else
            return false;
    }
    return true;
}

HoistResult DefaultCheckpointPolicy::classify(UseOrPseudoUse use)
{
    // Store all that we can.. by default, classify will only be called on relevant differential
    // uses (or on uses in a 'recompute' inst)
    //
    if (auto var = as<IRVar>(use.usedVal))
    {
        if (shouldStoreVar(var))
            return HoistResult::store(var);
        else
            return HoistResult::recompute(var);
    }
    else
    {
        if (shouldStoreInst(use.usedVal))
        {
            return HoistResult::store(use.usedVal);
        }
        else
        {
            // We may not be able to recompute due to limitations of
            // the unzip pass. If so we will store the result.
            if (canRecompute(use))
                return HoistResult::recompute(use.usedVal);

            // The fallback is to store.
            return HoistResult::store(use.usedVal);
        }
    }
}
}; // namespace Slang
