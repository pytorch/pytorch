#include "slang-ir-simplify-cfg.h"

#include "slang-ir-dominators.h"
#include "slang-ir-insts.h"
#include "slang-ir-loop-unroll.h"
#include "slang-ir-reachability.h"
#include "slang-ir-restructure.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{

struct CFGSimplificationContext
{
    RefPtr<RegionTree> regionTree;
    RefPtr<IRDominatorTree> domTree;
    Dictionary<IRInst*, List<IRInst*>> relatedAddrMap;
};

static bool isBlockInRegion(
    IRDominatorTree* domTree,
    IRTerminatorInst* regionHeader,
    IRBlock* block)
{
    auto headerBlock = cast<IRBlock>(regionHeader->getParent());
    IRBlock* breakBlock = nullptr;
    if (auto loop = as<IRLoop>(regionHeader))
        breakBlock = loop->getBreakBlock();
    else if (auto switchInst = as<IRSwitch>(regionHeader))
        breakBlock = switchInst->getBreakLabel();

    auto parentBreakBlocks = getParentBreakBlockSet(domTree, headerBlock);

    if (!domTree->dominates(headerBlock, block))
        return false;

    if (domTree->dominates(breakBlock, block))
        return false;

    for (auto parentBreakBlock : parentBreakBlocks)
    {
        if (domTree->dominates(parentBreakBlock, block))
            return false;
    }

    return true;
}

static IRInst* findBreakableRegionHeaderInst(IRDominatorTree* domTree, IRBlock* block)
{
    for (auto idom = domTree->getImmediateDominator(block); idom;
         idom = domTree->getImmediateDominator(idom))
    {
        auto terminator = idom->getTerminator();
        switch (terminator->getOp())
        {
        case kIROp_Switch:
        case kIROp_loop:
            return terminator;
        }
    }
    return nullptr;
}

// Test if a loop is trivial: a trivial loop runs for a single iteration without any back edges, and
// there is only one break out of the loop at the very end. The function generates `regionTree` if
// it is needed and hasn't been generated yet.
bool isTrivialSingleIterationLoop(
    IRDominatorTree* domTree,
    IRGlobalValueWithCode* func,
    IRLoop* loop)
{
    auto targetBlock = loop->getTargetBlock();
    if (targetBlock->getPredecessors().getCount() != 1)
        return false;
    if (*targetBlock->getPredecessors().begin() != loop->getParent())
        return false;

    int useCount = 0;
    for (auto use = loop->getBreakBlock()->firstUse; use; use = use->nextUse)
    {
        if (use->getUser() == loop)
            continue;
        useCount++;
        if (useCount > 1)
            return false;
    }

    // The loop has passed simple test.
    //
    // We need to verify this is a trivial loop by checking if there is any multi-level breaks
    // that skips out of this loop.
    if (!domTree)
        domTree = computeDominatorTree(func);
    bool hasMultiLevelBreaks = false;
    auto loopBlocks = collectBlocksInRegion(domTree, loop, &hasMultiLevelBreaks);
    if (hasMultiLevelBreaks)
        return false;
    for (auto block : loopBlocks)
    {
        for (auto branchTarget : block->getSuccessors())
        {
            if (!domTree->dominates(loop->getParent(), branchTarget))
                return false;
            if (branchTarget != loop->getBreakBlock())
                continue;
            if (findBreakableRegionHeaderInst(domTree, block) != loop)
            {
                // If the break is initiated from a nested region, this is not trivial.
                return false;
            }
        }
    }

    // We'll also check if there's an inner loop that is breaking out into this loop's break block.
    // If so, we cannot remove it right away since it interferes with the multi-level break
    // elimination logic.
    //
    // Track the break block backwards through the dominator tree, and see if we find a loop block
    // that is not the current loop.
    //
    auto breakPredList = loop->getBreakBlock()->getPredecessors();

    if (breakPredList.getCount() > 0)
    {
        auto breakOriginBlock = *loop->getBreakBlock()->getPredecessors().begin();

        for (auto currBlock = breakOriginBlock; currBlock;
             currBlock = domTree->getImmediateDominator(currBlock))
        {
            auto terminator = currBlock->getTerminator();
            if (terminator == loop)
                break;

            // Check if the break originated from an inner breakable region.
            // If so, the outer loop cannot be trivially removed.
            //
            switch (terminator->getOp())
            {
            case kIROp_loop:
                if (isBlockInRegion(domTree, as<IRLoop>(terminator), breakOriginBlock))
                    return false;
                break;
            case kIROp_Switch:
                if (isBlockInRegion(domTree, as<IRSwitch>(terminator), breakOriginBlock))
                    return false;
                break;
            default:
                break;
            }
        }
    }

    return true;
}

static bool doesLoopHasSideEffect(
    CFGSimplificationContext& context,
    ReachabilityContext& reachability,
    IRGlobalValueWithCode* func,
    IRLoop* loopInst)
{
    bool hasMultiLevelBreaks = false;
    if (!context.domTree)
        context.domTree = computeDominatorTree(func);
    auto blocks = collectBlocksInRegion(context.domTree.get(), loopInst, &hasMultiLevelBreaks);

    // We'll currently not deal with loops that contain multi-level breaks.
    if (hasMultiLevelBreaks)
        return true;

    HashSet<IRBlock*> loopBlocks;
    for (auto b : blocks)
        loopBlocks.add(b);

    // Construct a map from a root address to all derived addresses.
    Dictionary<IRInst*, List<IRInst*>>& relatedAddrMap = context.relatedAddrMap;
    if (!relatedAddrMap.getCount())
    {
        for (auto b : func->getBlocks())
        {
            for (auto inst : b->getChildren())
            {
                if (as<IRPtrTypeBase>(inst->getDataType()))
                {
                    auto root = getRootAddr(inst);
                    if (!root)
                        continue;
                    auto list = relatedAddrMap.tryGetValue(root);
                    if (!list)
                    {
                        relatedAddrMap.add(root, List<IRInst*>());
                        list = relatedAddrMap.tryGetValue(root);
                    }
                    list->add(inst);
                }
            }
        }
    }

    auto addressHasOutOfLoopUses = [&](IRInst* addr)
    {
        auto rootAddr = getRootAddr(addr);
        if (isGlobalOrUnknownMutableAddress(func, rootAddr))
            return true;
        if (as<IRParam, IRDynamicCastBehavior::NoUnwrap>(rootAddr))
            return true;

        // If we can't find the address from our map, we conservatively assume it is an unknown
        // address.
        auto relatedAddrs = relatedAddrMap.tryGetValue(getRootAddr(addr));
        if (!relatedAddrs)
            return true;

        // For all related address of `addr` that may alias with it, we check their uses.
        for (auto relatedAddr : *relatedAddrs)
        {
            if (!canAddressesPotentiallyAlias(func, relatedAddr, addr))
                continue;
            for (auto use = relatedAddr->firstUse; use; use = use->nextUse)
            {
                if (!loopBlocks.contains(as<IRBlock>(use->getUser()->getParent())))
                {
                    // Is this use reachable from the loop header?
                    if (reachability.isInstReachable(loopInst, use->getUser()))
                        return true;
                }
            }
        }

        return false;
    };

    for (auto b : blocks)
    {
        for (auto inst : b->getChildren())
        {
            // Is this inst used anywhere outside the loop? If so the loop has side effect.
            for (auto use = inst->firstUse; use; use = use->nextUse)
            {
                if (!loopBlocks.contains(as<IRBlock>(use->getUser()->getParent())))
                    return true;
            }

            // This inst might have side effect, try to prove that the
            // side effect does not leak beyond the scope of the loop.
            if (auto call = as<IRCall>(inst))
            {
                auto callee = getResolvedInstForDecorations(call->getCallee());
                if (!callee || !(callee->findDecoration<IRNoSideEffectDecoration>() ||
                                 callee->findDecoration<IRReadNoneDecoration>()))
                    return true;
                // We are calling a pure function, check if any of the return
                // variables are used outside the loop.
                for (UInt i = 0; i < call->getArgCount(); i++)
                {
                    auto arg = call->getArg(i);
                    if (!isValueType(arg->getDataType()))
                    {
                        if (addressHasOutOfLoopUses(arg))
                            return true;
                    }
                }
            }
            else if (auto store = as<IRStore>(inst))
            {
                if (addressHasOutOfLoopUses(store->getPtr()))
                    return true;
            }
            else if (auto branch = as<IRUnconditionalBranch>(inst))
            {
                if (loopBlocks.contains(branch->getTargetBlock()))
                    continue;
                // Branching out of the loop with some argument is considered
                // having a side effect.
                if (branch->getArgCount() != 0)
                    return true;
            }
            else if (as<IRIfElse>(inst) || as<IRSwitch>(inst) || as<IRLoop>(inst))
            {
                // We are starting a sub control flow.
                // This is considered side effect free.
            }
            else
            {
                // The inst can't possibly have side effect? Skip it.
                if (!inst->mightHaveSideEffects())
                    continue;

                // For all other insts, we assume it has a global side effect.
                return true;
            }
        }
    }
    return false;
}

static bool removeDeadBlocks(IRGlobalValueWithCode* func)
{
    bool changed = false;
    List<IRBlock*> workList;
    auto firstBlock = func->getFirstBlock();
    if (!firstBlock)
        return false;

    for (auto block = firstBlock->getNextBlock(); block; block = block->getNextBlock())
    {
        workList.add(block);
    }

    HashSet<IRBlock*> workListSet;
    List<IRBlock*> nextWorkList;
    for (;;)
    {
        for (Index i = 0; i < workList.getCount(); i++)
        {
            auto block = workList[i];
            if (!block->hasUses() && as<IRTerminatorInst>(block->getFirstInst()))
            {
                for (auto succ : block->getSuccessors())
                {
                    if (workListSet.add(succ))
                    {
                        nextWorkList.add(succ);
                    }
                }
                block->removeAndDeallocate();
                changed = true;
            }
        }
        if (nextWorkList.getCount())
        {
            workList = _Move(nextWorkList);
            workListSet.clear();
        }
        else
        {
            break;
        }
    }
    return changed;
}

// Return the true of the if-else branch block if the branch is a trivial jump
// to after block with no other insts.
static bool isTrivialIfElseBranch(IRIfElse* condBranch, IRBlock* branchBlock)
{
    if (branchBlock != condBranch->getAfterBlock())
    {
        if (auto br = as<IRUnconditionalBranch>(branchBlock->getFirstOrdinaryInst()))
        {
            if (br->getTargetBlock() == condBranch->getAfterBlock() &&
                br->getOp() == kIROp_unconditionalBranch)
            {
                return true;
            }
        }
    }
    else
    {
        return true;
    }
    return false;
}

static bool arePhiArgsEquivalentInBranchesImpl(
    IRBlock* branch1,
    IRBlock* branch2,
    IRBlock* afterBlock)
{
    if (branch1 == afterBlock)
        return true;
    if (branch2 == afterBlock)
        return true;

    auto branchInst1 = as<IRUnconditionalBranch>(branch1->getTerminator());
    auto branchInst2 = as<IRUnconditionalBranch>(branch2->getTerminator());
    if (!branchInst1)
        return false;
    if (!branchInst2)
        return false;

    // If both branches are trivial blocks, we must compare the arguments.
    if (branchInst1->getArgCount() != branchInst2->getArgCount())
    {
        // This should never happen, return false now to be safe.
        return false;
    }

    for (UInt i = 0; i < branchInst1->getArgCount(); i++)
    {
        if (branchInst1->getArg(i) != branchInst2->getArg(i))
        {
            // argument is different, the if-else is non-trivial.
            return false;
        }
    }
    return true;
}

static bool arePhiArgsEquivalentInBranches(IRIfElse* ifElse)
{
    // If one of the branch target is afterBlock itself, and the other branch
    // is a trivial block that jumps into the afterBlock, this if-else is trivial.
    // In this case the argCount must be 0 because a block with phi parameters can't
    // be used as targets in a conditional branch.
    auto branch1 = ifElse->getTrueBlock();
    auto branch2 = ifElse->getFalseBlock();
    auto afterBlock = ifElse->getAfterBlock();

    return arePhiArgsEquivalentInBranchesImpl(branch1, branch2, afterBlock);
}

static bool isTrivialIfElse(
    IRIfElse* condBranch,
    bool& isTrueBranchTrivial,
    bool& isFalseBranchTrivial)
{
    isTrueBranchTrivial = isTrivialIfElseBranch(condBranch, condBranch->getTrueBlock());
    isFalseBranchTrivial = isTrivialIfElseBranch(condBranch, condBranch->getFalseBlock());
    if (isTrueBranchTrivial && isFalseBranchTrivial)
    {
        if (arePhiArgsEquivalentInBranches(condBranch))
            return true;
    }
    return false;
}

// Return the true of the switch branch block if the branch is a trivial jump
// to after block with no other insts.
static bool isTrivialSwitchBranch(IRSwitch* switchInst, IRBlock* branchBlock)
{
    if (branchBlock != switchInst->getBreakLabel())
    {
        if (auto br = as<IRUnconditionalBranch>(branchBlock->getFirstOrdinaryInst()))
        {
            if (br->getTargetBlock() == switchInst->getBreakLabel() &&
                br->getOp() == kIROp_unconditionalBranch)
            {
                return true;
            }
        }
    }
    else
    {
        return true;
    }
    return false;
}

static bool arePhiArgsEquivalentInBranches(IRSwitch* switchInst)
{
    ShortList<IRBlock*> jumpTargets;
    if (switchInst->getDefaultLabel())
        jumpTargets.add(switchInst->getDefaultLabel());
    for (UInt i = 0; i < switchInst->getCaseCount(); i++)
    {
        jumpTargets.add(switchInst->getCaseLabel(i));
    }
    if (jumpTargets.getCount() == 0)
        return true;
    for (Index i = 1; i < jumpTargets.getCount(); i++)
    {
        auto branch1 = jumpTargets[0];
        auto branch2 = jumpTargets[i];
        auto afterBlock = switchInst->getBreakLabel();

        if (!arePhiArgsEquivalentInBranchesImpl(branch1, branch2, afterBlock))
            return false;
    }
    return true;
}

static bool isTrivialSwitch(IRSwitch* switchBranch)
{
    for (UInt i = 0; i < switchBranch->getCaseCount(); i++)
    {
        if (!isTrivialSwitchBranch(switchBranch, switchBranch->getCaseLabel(i)))
            return false;
    }
    if (!isTrivialSwitchBranch(switchBranch, switchBranch->getDefaultLabel()))
        return false;
    return true;
}

static bool trySimplifyIfElse(IRBuilder& builder, IRIfElse* ifElseInst)
{
    bool isTrueBranchTrivial = false;
    bool isFalseBranchTrivial = false;
    if (isTrivialIfElse(ifElseInst, isTrueBranchTrivial, isFalseBranchTrivial))
    {
        // If either branch of `if-else` is a trivial jump into after block,
        // we can get rid of the entire conditional branch and replace it
        // with a jump into the after block.
        IRUnconditionalBranch* termInst =
            as<IRUnconditionalBranch>(ifElseInst->getTrueBlock()->getTerminator());
        if (!termInst || (termInst->getTargetBlock() != ifElseInst->getAfterBlock()))
        {
            termInst = as<IRUnconditionalBranch>(ifElseInst->getFalseBlock()->getTerminator());
        }

        if (termInst)
        {
            SLANG_ASSERT(termInst->getTargetBlock() == ifElseInst->getAfterBlock());
            List<IRInst*> args;
            for (UInt i = 0; i < termInst->getArgCount(); i++)
                args.add(termInst->getArg(i));
            builder.setInsertBefore(ifElseInst);
            builder.emitBranch(ifElseInst->getAfterBlock(), (Int)args.getCount(), args.getBuffer());
            ifElseInst->removeAndDeallocate();
            return true;
        }
    }
    else
    {
        // Otherwise, we can try to remove at least remove one of the trivial branches
        // Remove either the true or false block if it jumps to the after block
        // with no parameters.

        const auto afterBlock = ifElseInst->getAfterBlock();
        if (!afterBlock->getFirstParam())
        {
            const auto trueBlock = ifElseInst->getTrueBlock();
            const auto falseBlock = ifElseInst->getFalseBlock();

            if (isTrueBranchTrivial && trueBlock != afterBlock && !trueBlock->hasMoreThanOneUse())
            {
                trueBlock->replaceUsesWith(afterBlock);
                trueBlock->removeAndDeallocate();
            }
            else if (
                isFalseBranchTrivial && falseBlock != afterBlock &&
                !falseBlock->hasMoreThanOneUse())
            {
                falseBlock->replaceUsesWith(afterBlock);
                falseBlock->removeAndDeallocate();
            }
        }
    }
    return false;
}

static bool trySimplifySwitch(IRBuilder& builder, IRSwitch* switchInst)
{
    // First, we fuse switch case blocks that is a trivial branch.
    // If we see:
    // ```
    //     someBlock:
    //         switch(..., case_block_A, ...)
    //     case_block_A:
    //         branch blockB;
    // ```
    // Then we fold blockB into the switch case operand:
    // ```
    //     someBlock:
    //         switch(..., blockB, ...)
    // ```
    // We can do this if `blockB` is not a merge block.
    //
    bool changed = false;
    auto fuseSwitchCaseBlock = [&](IRUse* targetUse)
    {
        for (;;)
        {
            auto block = as<IRBlock>(targetUse->get());
            if (block->getFirstInst()->getOp() != kIROp_unconditionalBranch)
                return;
            auto branch = as<IRUnconditionalBranch>(block->getFirstInst());
            // We can't fuse the block if there are phi arguments.
            if (branch->getArgCount() != 0)
                return;
            auto target = branch->getTargetBlock();
            if (target == switchInst->getBreakLabel())
                return;
            // target must not be used as a merge block of other control flow constructs.
            for (auto use = target->firstUse; use; use = use->nextUse)
            {
                if (use->getUser() == switchInst || use->getUser() == branch)
                    continue;
                switch (use->getUser()->getOp())
                {
                case kIROp_loop:
                case kIROp_ifElse:
                case kIROp_Switch:
                    // If the target block is used by a special control flow inst,
                    // it is likely a merge block and we can't fuse it.
                    return;
                default:
                    break;
                }
            }
            targetUse->set(target);
            changed = true;
        }
    };

    fuseSwitchCaseBlock(&switchInst->defaultLabel);
    for (UInt i = 0; i < switchInst->getCaseCount(); i++)
        fuseSwitchCaseBlock(switchInst->getCaseLabelUse(i));

    // Next, we check if all switch cases are jumping to the same target.
    if (!isTrivialSwitch(switchInst))
        return changed;
    if (switchInst->getCaseCount() == 0)
        return changed;

    auto termInst = as<IRUnconditionalBranch>(switchInst->getCaseLabel(0)->getTerminator());
    if (!termInst)
        return changed;

    if (!arePhiArgsEquivalentInBranches(switchInst))
        return changed;

    List<IRInst*> args;
    for (UInt i = 0; i < termInst->getArgCount(); i++)
        args.add(termInst->getArg(i));
    builder.setInsertBefore(switchInst);
    builder.emitBranch(switchInst->getBreakLabel(), (Int)args.getCount(), args.getBuffer());
    switchInst->removeAndDeallocate();
    return true;
}

static bool isTrueLit(IRInst* lit)
{
    if (auto boolLit = as<IRBoolLit>(lit))
        return boolLit->getValue();
    return false;
}
static bool isFalseLit(IRInst* lit)
{
    if (auto boolLit = as<IRBoolLit>(lit))
        return !boolLit->getValue();
    return false;
}

static bool simplifyBoolPhiParam(
    IRIfElse* ifElse,
    Array<IRBlock*, 2>& preds,
    IRParam* param,
    UInt paramIndex)
{
    // For bool params where its value is assigned from the same `if-else` statement,
    // we can simplify it into an expression of the condition of the source `if-else`.

    if (!param->getDataType() || param->getDataType()->getOp() != kIROp_BoolType)
        return false;

    auto branch0 = as<IRUnconditionalBranch>(preds[0]->getTerminator());
    if (!branch0)
        return false;
    if (branch0->getArgCount() <= paramIndex)
        return false;
    auto branch1 = as<IRUnconditionalBranch>(preds[1]->getTerminator());
    if (!branch1)
        return false;
    if (branch1->getArgCount() <= paramIndex)
        return false;

    IRInst* replacement = nullptr;
    if (isTrueLit(branch0->getArg(paramIndex)) && isFalseLit(branch1->getArg(paramIndex)))
    {
        replacement = ifElse->getCondition();
    }
    else if (isFalseLit(branch0->getArg(paramIndex)) && isTrueLit(branch1->getArg(paramIndex)))
    {
        IRBuilder builder(param);
        setInsertBeforeOrdinaryInst(&builder, param);
        replacement = builder.emitNot(builder.getBoolType(), ifElse->getCondition());
    }
    if (replacement)
    {
        param->replaceUsesWith(replacement);
        param->removeAndDeallocate();
        branch0->removeArgument(paramIndex);
        branch1->removeArgument(paramIndex);
        return true;
    }
    return false;
}

static bool simplifyBoolPhiParams(IRBlock* block)
{
    if (!block)
        return false;

    if (block->getPredecessors().getCount() != 2)
        return false;

    Array<IRBlock*, 2> preds;
    for (auto pred : block->getPredecessors())
    {
        if (pred->getTerminator()->getOp() != kIROp_unconditionalBranch)
            return false;
        preds.add(pred);
    }

    IRBlock* ifElseBlock = nullptr;
    if (preds[0]->getPredecessors().getCount() != 1)
        return false;
    ifElseBlock = *(preds[0]->getPredecessors().begin());
    if (preds[1]->getPredecessors().getCount() != 1)
        return false;
    auto p = *(preds[1]->getPredecessors().begin());
    if (p != ifElseBlock)
        return false;

    auto ifElse = as<IRIfElse>(ifElseBlock->getTerminator());
    if (!ifElse)
        return false;

    if (ifElse->getTrueBlock() == preds[1])
    {
        Swap(preds[0], preds[1]);
    }
    SLANG_ASSERT(ifElse->getTrueBlock() == preds[0] && ifElse->getFalseBlock() == preds[1]);

    List<IRParam*> params;
    for (auto param : block->getParams())
        params.add(param);
    bool changed = false;
    for (Index i = params.getCount() - 1; i >= 0; i--)
    {
        changed |= simplifyBoolPhiParam(ifElse, preds, params[i], (UInt)i);
    }
    return changed;
}

static bool removeTrivialPhiParams(IRBlock* block)
{
    // We can remove a phi parameter if:
    // 1. all non-self-referential arguments to a parameter are the same (not really a phi).
    // 2. the arguments to the parameter are always the same as arguments to another existing
    // parameter (duplicate phi).

    bool changed = false;
    List<IRParam*> params;
    struct ParamState
    {
        bool areKnownValueSame = true;
        IRInst* knownValue = nullptr;
        OrderedHashSet<UInt> sameAsParamSet;
    };
    List<ParamState> args;
    List<IRUnconditionalBranch*> termInsts;
    for (auto param : block->getParams())
    {
        params.add(param);
        args.add(ParamState());
    }

    if (!params.getCount())
        return false;

    for (UInt i = 1; i < (UInt)args.getCount(); i++)
        for (UInt j = 0; j < i; j++)
            args[i].sameAsParamSet.add(j);

    for (auto pred : block->getPredecessors())
    {
        auto termInst = as<IRUnconditionalBranch>(pred->getTerminator());
        if (!termInst)
            return false;
        SLANG_ASSERT(termInst->getArgCount() == (UInt)args.getCount());
        termInsts.add(termInst);
        for (UInt i = 0; i < termInst->getArgCount(); i++)
        {
            // Self-referential parameters can be skipped, as they cannot
            // introduce a new value. The phi can only have multiple different
            // values if non-self-referential arguments differ.
            if (args[i].areKnownValueSame && termInst->getArg(i) != params[i])
            {
                if (args[i].knownValue == nullptr)
                    args[i].knownValue = termInst->getArg(i);
                else if (args[i].knownValue != termInst->getArg(i))
                    args[i].areKnownValueSame = false;
            }
            for (UInt j = 0; j < i; j++)
            {
                if (termInst->getArg(i) != termInst->getArg(j))
                {
                    args[i].sameAsParamSet.remove(j);
                }
            }
        }
    }
    for (Index i = args.getCount() - 1; i >= 0; i--)
    {
        IRInst* targetVal = nullptr;
        if (args[i].areKnownValueSame)
        {
            targetVal = args[i].knownValue;
        }
        else if (args[i].sameAsParamSet.getCount())
        {
            auto targetParamId = *args[i].sameAsParamSet.begin();
            targetVal = params[targetParamId];
        }
        if (targetVal)
        {
            params[i]->replaceUsesWith(targetVal);
            params[i]->removeAndDeallocate();
            for (auto termInst : termInsts)
                termInst->removeArgument((UInt)i);
            changed = true;
        }
    }
    return changed;
}

static bool processFunc(IRGlobalValueWithCode* func, CFGSimplificationOptions options)
{
    auto firstBlock = func->getFirstBlock();
    if (!firstBlock)
        return false;

    IRBuilder builder(func->getModule());

    bool isReachabilityContextValid = false;
    ReachabilityContext reachabilityContext;
    CFGSimplificationContext simplificationContext;

    bool changed = false;
    for (;;)
    {
        List<IRBlock*> workList;
        HashSet<IRBlock*> processedBlock;
        workList.add(func->getFirstBlock());
        while (workList.getCount())
        {
            auto block = workList.getFirst();
            workList.fastRemoveAt(0);
            while (block)
            {
                // If all arguments to a phi parameter are the known to be the same,
                // we can safely replace the phi parameter with the argument.
                if (block != func->getFirstBlock())
                {
                    changed |= simplifyBoolPhiParams(block);
                    changed |= removeTrivialPhiParams(block);
                }

                if (auto loop = as<IRLoop>(block->getTerminator()))
                {
                    // If continue block is unreachable, remove it.
                    auto continueBlock = loop->getContinueBlock();
                    if (continueBlock && !continueBlock->hasMoreThanOneUse())
                    {
                        loop->continueBlock.set(loop->getTargetBlock());
                        continueBlock->removeAndDeallocate();
                        simplificationContext = CFGSimplificationContext();
                        changed = true;
                    }

                    // If there isn't any actual back jumps into loop target and there is a trivial
                    // break at the end of the loop, we can remove the header and turn it into
                    // a normal branch.
                    auto targetBlock = loop->getTargetBlock();
                    if (!simplificationContext.domTree)
                        simplificationContext.domTree = computeDominatorTree(func);
                    if (options.removeTrivialSingleIterationLoops &&
                        isTrivialSingleIterationLoop(simplificationContext.domTree, func, loop))
                    {
                        builder.setInsertBefore(loop);
                        List<IRInst*> args;
                        for (UInt i = 0; i < loop->getArgCount(); i++)
                        {
                            args.add(loop->getArg(i));
                        }
                        builder.emitBranch(targetBlock, args.getCount(), args.getBuffer());
                        loop->removeAndDeallocate();
                        simplificationContext = CFGSimplificationContext();
                        changed = true;
                    }
                    else if (options.removeSideEffectFreeLoops)
                    {
                        if (!isReachabilityContextValid)
                        {
                            isReachabilityContextValid = true;
                            reachabilityContext = ReachabilityContext(func);
                        }
                        if (!doesLoopHasSideEffect(
                                simplificationContext,
                                reachabilityContext,
                                func,
                                loop))
                        {
                            // The loop isn't computing anything useful outside the loop.
                            // We can delete the entire loop.
                            builder.setInsertBefore(loop);
                            SLANG_ASSERT(loop->getBreakBlock()->getFirstParam() == nullptr);
                            builder.emitBranch(loop->getBreakBlock());
                            loop->removeAndDeallocate();
                            simplificationContext = CFGSimplificationContext();
                            changed = true;
                        }
                    }
                }
                else if (auto condBranch = as<IRIfElse>(block->getTerminator()))
                {
                    if (trySimplifyIfElse(builder, condBranch))
                    {
                        simplificationContext = CFGSimplificationContext();
                        changed = true;
                    }
                }
                else if (auto switchBranch = as<IRSwitch>(block->getTerminator()))
                {
                    if (trySimplifySwitch(builder, switchBranch))
                    {
                        simplificationContext = CFGSimplificationContext();
                        changed = true;
                    }
                }

                // If `block` does not end with an unconditional branch, bail.
                if (block->getTerminator()->getOp() != kIROp_unconditionalBranch)
                    break;
                auto branch = as<IRUnconditionalBranch>(block->getTerminator());
                auto successor = branch->getTargetBlock();
                // Only perform the merge if `block` is the only predecessor of `successor`.
                // We also need to make sure not to merge a block that serves as the
                // merge point in CFG. Such blocks will have more than one use.
                if (successor->hasMoreThanOneUse())
                    break;
                if (block->hasMoreThanOneUse())
                    break;
                changed = true;
                simplificationContext = CFGSimplificationContext();
                Index paramIndex = 0;
                auto inst = successor->getFirstDecorationOrChild();
                while (inst)
                {
                    auto next = inst->getNextInst();
                    if (inst->getOp() == kIROp_Param)
                    {
                        inst->replaceUsesWith(branch->getArg(paramIndex));
                        paramIndex++;
                    }
                    else
                    {
                        inst->removeFromParent();
                        inst->insertAtEnd(block);
                    }
                    inst = next;
                }
                branch->removeAndDeallocate();
                assert(!successor->hasUses());
                successor->removeAndDeallocate();

                break;
            }
            for (auto successor : block->getSuccessors())
            {
                if (processedBlock.add(successor))
                {
                    workList.add(successor);
                }
            }
        }
        bool blocksRemoved = removeDeadBlocks(func);
        changed |= blocksRemoved;
        if (!blocksRemoved)
            break;
    }
    if (changed)
    {
        auto module = func->getModule();
        if (module)
            module->invalidateAnalysisForInst(func);
    }
    return changed;
}

bool simplifyCFG(IRModule* module, CFGSimplificationOptions options)
{
    bool changed = false;
    for (auto inst : module->getGlobalInsts())
    {
        if (auto genericInst = as<IRGeneric>(inst))
        {
            inst = findGenericReturnVal(genericInst);
        }
        if (auto func = as<IRFunc>(inst))
        {
            changed |= processFunc(func, options);
        }
    }
    return changed;
}

bool simplifyCFG(IRGlobalValueWithCode* func, CFGSimplificationOptions options)
{
    if (auto genericFunc = as<IRGeneric>(func))
    {
        if (auto inner = as<IRFunc>(findGenericReturnVal(genericFunc)))
            processFunc(inner, options);
    }
    return processFunc(func, options);
}

} // namespace Slang
