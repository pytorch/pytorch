// slang-ir-autodiff-cfg-norm.cpp
#include "slang-ir-autodiff-cfg-norm.h"

#include "slang-ir-dominators.h"
#include "slang-ir-eliminate-phis.h"
#include "slang-ir-ssa.h"
#include "slang-ir-util.h"
#include "slang-ir-validate.h"

namespace Slang
{

struct RegionEndpoint
{
    bool inBreakRegion = false;
    bool inBaseRegion = false;

    IRBlock* exitBlock = nullptr;

    bool isRegionEmpty = false;

    RegionEndpoint(IRBlock* exitBlock, bool inBreakRegion, bool inBaseRegion)
        : exitBlock(exitBlock)
        , inBreakRegion(inBreakRegion)
        , inBaseRegion(inBaseRegion)
        , isRegionEmpty(false)
    {
    }

    RegionEndpoint(IRBlock* exitBlock, bool inBreakRegion, bool inBaseRegion, bool isRegionEmpty)
        : exitBlock(exitBlock)
        , inBreakRegion(inBreakRegion)
        , inBaseRegion(inBaseRegion)
        , isRegionEmpty(isRegionEmpty)
    {
    }

    RegionEndpoint() {}
};

struct BreakableRegionInfo
{
    IRVar* breakVar;
    IRBlock* breakBlock;
    IRBlock* headerBlock;
};

struct CFGNormalizationContext
{
    IRModule* module;
    DiagnosticSink* sink;
};

IRBlock* getOrCreateTopLevelCondition(IRLoop* loopInst)
{
    // For now, we're going to naively assume the next block is the condition block.
    // Add in more support for more cases as necessary.
    //

    auto firstBlock = loopInst->getTargetBlock();

    if (as<IRIfElse>(firstBlock->getTerminator()))
    {
        return firstBlock;
    }
    else
    {
        // If there isn't a condition we need to make one with a dummy condition that
        // always evaluates to true
        //

        IRBuilder condBuilder(loopInst->getModule());

        auto condBlock = condBuilder.emitBlock();
        condBlock->insertAfter(as<IRBlock>(loopInst->getParent()));

        // Make loop go into the condition block
        firstBlock->replaceUsesWith(condBlock);

        // Emit a condition: true side goes to the loop body, and
        // false side goes into the break block.
        //
        condBuilder.setInsertInto(condBlock);
        auto ifElse = as<IRIfElse>(condBuilder.emitIfElse(
            condBuilder.getBoolValue(true),
            firstBlock,
            loopInst->getBreakBlock(),
            firstBlock));

        // We'll insert a blank block between the condition and the
        // break block, since otherwise, we might trip up the later
        // parts of this pass.
        //
        condBuilder.insertBlockAlongEdge(loopInst->getModule(), IREdge(&ifElse->falseBlock));

        return condBlock;
    }
}

struct CFGNormalizationPass
{
    CFGNormalizationContext cfgContext;

    CFGNormalizationPass(CFGNormalizationContext ctx)
        : cfgContext(ctx)
    {
    }

    void replaceBreakWithAfterBlock(
        IRBuilder* builder,
        BreakableRegionInfo* info,
        IRBlock* currBlock,
        IRBlock* afterBlock,
        IRBlock* parentAfterBlock)
    {
        SLANG_ASSERT(as<IRUnconditionalBranch>(currBlock->getTerminator()));

        currBlock->getTerminator()->removeAndDeallocate();

        builder->setInsertInto(currBlock);

        builder->emitStore(info->breakVar, builder->getBoolValue(false));
        builder->emitBranch(afterBlock);

        // Is after-block unreachable?
        if (auto unreachInst = as<IRUnreachable>(afterBlock->getFirstOrdinaryInst()))
        {
            // Link it to the parentAfterBlock.
            builder->setInsertInto(afterBlock);
            unreachInst->removeAndDeallocate();

            builder->emitBranch(parentAfterBlock);
        }
    }

    IRBlock* getUnconditionalTarget(RegionEndpoint endpoint)
    {
        if (!endpoint.isRegionEmpty)
        {
            auto branchInst = as<IRUnconditionalBranch>(endpoint.exitBlock->getTerminator());
            SLANG_ASSERT(branchInst);

            return branchInst->getTargetBlock();
        }
        else
        {
            return endpoint.exitBlock;
        }
    }

    IRBlock* maybeGetUnconditionalTarget(IRBlock* block)
    {
        auto branchInst = as<IRUnconditionalBranch>(block->getTerminator());

        return branchInst ? branchInst->getTargetBlock() : nullptr;
    }

    bool isSuccessorBlock(IRBlock* baseBlock, IRBlock* succBlock)
    {
        for (auto successor : baseBlock->getSuccessors())
            if (successor == succBlock)
                return true;

        return false;
    }

    void _moveVarsToRegionHeader(BreakableRegionInfo* region, IRBlock* block)
    {
        for (auto child = block->getFirstChild(); child;)
        {
            auto nextChild = child->getNextInst();

            if (as<IRVar>(child))
            {
                if (auto loopInst = as<IRLoop>(region->headerBlock->getTerminator()))
                {
                    // In order to avoid introducing unnecessary loop state, we'll move vars
                    // to the loop's target (first loop block) instead of the loop header.
                    // (unless the var is already in the header or target)
                    //
                    if (block != region->headerBlock && block != loopInst->getTargetBlock())
                        child->insertBefore(loopInst->getTargetBlock()->getTerminator());
                }
                else
                    child->insertBefore(region->headerBlock->getTerminator());
            }

            child = nextChild;
        }
    }

    RegionEndpoint getNormalizedRegionEndpoint(
        BreakableRegionInfo* parentRegion,
        IRBlock* entryBlock,
        List<IRBlock*> afterBlocks)
    {
        IRBlock* currentBlock = entryBlock;
        _moveVarsToRegionHeader(parentRegion, currentBlock);

        // By default a region starts off with the 'base' control flow
        // and not in the 'break' control flow
        // It is the job of the *caller* to make sure the break flow
        // does not reach this point.
        //
        bool currBreakRegion = false;
        bool currBaseRegion = true;

        // Detect the trivial case. The current block is already
        // in the next region => this region is empty.
        //
        if (afterBlocks.contains(currentBlock))
            return RegionEndpoint(currentBlock, currBreakRegion, currBaseRegion, true);

        IRBuilder builder(cfgContext.module);

        List<IRBlock*> pendingAfterBlocks;

        IRBlock* parentAfterBlock = afterBlocks[0];

        auto addBreakBypassBranch = [&](IRBlock* block)
        {
            // We could arrive at the after-block before or
            // after encountering a break statement.
            // To handle this, we'll split the flow by checking the break flag
            //
            builder.setInsertAfter(block);

            auto preAfterSplitBlock = builder.emitBlock();
            preAfterSplitBlock->insertBefore(block);

            auto afterSplitBlock = builder.emitBlock();
            afterSplitBlock->insertBefore(block);

            block->replaceUsesWith(preAfterSplitBlock);

            builder.setInsertInto(preAfterSplitBlock);
            builder.emitBranch(afterSplitBlock);

            // Converging block for the split that we're making.
            auto afterSplitAfterBlock = builder.emitBlock();

            builder.setInsertInto(afterSplitBlock);
            auto breakFlagValue = builder.emitLoad(parentRegion->breakVar);

            builder.emitIfElse(breakFlagValue, block, afterSplitAfterBlock, afterSplitAfterBlock);

            // At this point, we need to place afterSplitAfterBlock between
            // at the _end_ of this region, but we aren't there yet (and
            // don't know which block is the end of this region)
            // Therefore, we'll defer this step and add it to a list for later.
            //
            pendingAfterBlocks.add(afterSplitAfterBlock);
        };

        // Follow this thread of execution till we hit an
        // acceptable after block.
        //
        while (!afterBlocks.contains(maybeGetUnconditionalTarget(currentBlock)))
        {
            // Check the terminator.
            auto terminator = currentBlock->getTerminator();
            switch (terminator->getOp())
            {
            case kIROp_unconditionalBranch:
                {
                    auto targetBlock = as<IRUnconditionalBranch>(terminator)->getTargetBlock();
                    currentBlock = targetBlock;
                    break;
                }

            case kIROp_ifElse:
                {
                    auto ifElse = as<IRIfElse>(terminator);

                    // Special case. One of the branches will
                    // lead back to the condition.
                    //
                    SLANG_ASSERT(ifElse->getAfterBlock() != parentRegion->breakBlock);

                    auto trueEndPoint = getNormalizedRegionEndpoint(
                        parentRegion,
                        ifElse->getTrueBlock(),
                        List<IRBlock*>(ifElse->getAfterBlock(), parentRegion->breakBlock));

                    auto falseEndPoint = getNormalizedRegionEndpoint(
                        parentRegion,
                        ifElse->getFalseBlock(),
                        List<IRBlock*>(ifElse->getAfterBlock(), parentRegion->breakBlock));

                    auto trueTargetBlock = getUnconditionalTarget(trueEndPoint);
                    auto falseTargetBlock = getUnconditionalTarget(falseEndPoint);

                    auto afterBlock = ifElse->getAfterBlock();

                    // Trivial case, both end-points branch into the after block
                    /*if (trueTargetBlock == afterBlock &&
                        falseTargetBlock == afterBlock)
                    {
                        if ()
                        addBreakBypassBranch(afterBlock);
                        currentBlock = afterBlock;
                        // TODO: Need to split block.
                        break;
                    }*/

                    auto afterBreakRegion = false;
                    auto afterBaseRegion = false;

                    if (trueTargetBlock == parentRegion->breakBlock)
                    {
                        // Branch into after block (and set break variable)
                        replaceBreakWithAfterBlock(
                            &builder,
                            parentRegion,
                            trueEndPoint.exitBlock,
                            afterBlock,
                            parentAfterBlock);

                        // If this branch breaks, then the after-block
                        // definitely has break-flow.
                        //
                        afterBreakRegion = true;
                    }
                    else
                    {
                        // If this branch naturally branches into our
                        // after-block, copy whatever flags the endpoints
                        // have.
                        //
                        afterBreakRegion = afterBreakRegion || trueEndPoint.inBreakRegion;
                        afterBaseRegion = afterBaseRegion || trueEndPoint.inBaseRegion;
                    }

                    if (falseTargetBlock == parentRegion->breakBlock)
                    {
                        // Branch into after block (and set break variable)
                        replaceBreakWithAfterBlock(
                            &builder,
                            parentRegion,
                            falseEndPoint.exitBlock,
                            afterBlock,
                            parentAfterBlock);

                        // If this branch breaks, then the after-block
                        // definitely has break-flow.
                        //
                        afterBreakRegion = true;
                    }
                    else
                    {
                        // If this branch naturally branches into our
                        // after-block, copy whatever flags the endpoints
                        // have.
                        //
                        afterBreakRegion = afterBreakRegion || falseEndPoint.inBreakRegion;
                        afterBaseRegion = afterBaseRegion || falseEndPoint.inBaseRegion;
                    }

                    // TODO: For now, we're being overly cautious and assuming
                    // the after region might have something to execute.
                    // Ideally, we should check if the block is empty, and
                    // hold off on splitting until we encounter non-empty
                    // blocks.
                    //
                    afterBaseRegion = true;

                    // One case we do check for is if the after block is 'unreachable'
                    // i.e. the terminator is an `unreachable` instruction.
                    // In this case, we can safely assume that the after block does not
                    // have anything to execute. Further, we need to re-wire the
                    // previously unreachable block to the parent break block.
                    // Note that this operation is safe because if the after block was
                    // originally unreachable, all potential paths to it must have
                    // broken out of the region.
                    //
                    if (auto unreachInst = as<IRUnreachable>(afterBlock->getTerminator()))
                    {
                        // Link it to the parentAfterBlock.
                        builder.setInsertInto(afterBlock);
                        unreachInst->removeAndDeallocate();

                        builder.emitBranch(parentAfterBlock);

                        // We can now safely assume that the after block is empty.
                        // Set 'afterBaseRegion' to false, which should lead the rest
                        // of the logic to avoid splitting the after block
                        //
                        afterBaseRegion = false;
                    }

                    // Do we need to split the after region?
                    if (afterBaseRegion && afterBreakRegion)
                    {
                        // Before we split the afterBlock, we
                        // want to make sure the afterBlock is
                        // firmly _inside_ the current region.
                        // If it's part of the parent, add a
                        // dummy block.
                        //
                        if (afterBlocks.contains(afterBlock))
                        {
                            auto newAfterBlock = builder.emitBlock();

                            // TODO: This is a hack. Ideally we should be putting
                            // the new after block 'before' the old after block,
                            // but if the latter is a loop condition block, it dominates
                            // the former, which may depend on parameters in the loop
                            // condition block. (This eventually causes cloneInst to fail,
                            // since it is currently order-dependent)
                            // Remove this once cloneInst is order-independent.
                            //
                            // newAfterBlock->insertBefore(afterBlock);
                            newAfterBlock->insertAfter(falseEndPoint.exitBlock);

                            builder.emitBranch(afterBlock);

                            ifElse->afterBlock.set(newAfterBlock);
                            as<IRUnconditionalBranch>(trueEndPoint.exitBlock->getTerminator())
                                ->block.set(newAfterBlock);
                            as<IRUnconditionalBranch>(falseEndPoint.exitBlock->getTerminator())
                                ->block.set(newAfterBlock);

                            afterBlock = newAfterBlock;
                        }

                        addBreakBypassBranch(afterBlock);

                        // Update current block.
                        currentBlock = afterBlock;
                        afterBreakRegion = false;
                        afterBaseRegion = true;
                    }

                    currentBlock = afterBlock;
                    currBreakRegion = afterBreakRegion;
                    currBaseRegion = afterBaseRegion;
                    break;
                }

            case kIROp_loop:
            case kIROp_Switch:
                {
                    auto breakBlock = normalizeBreakableRegion(terminator);

                    // Advance to the break block (no updates to the control flags)
                    currentBlock = breakBlock;
                    break;
                }

            default:
                // Do proper diagnosing
                SLANG_UNEXPECTED("Unhandled control flow inst");
                break;
            }

            _moveVarsToRegionHeader(parentRegion, currentBlock);
        }

        // Resolve all intermediate after-blocks
        pendingAfterBlocks.reverse();

        for (auto block : pendingAfterBlocks)
        {
            builder.setInsertInto(block);
            auto nextRegionBlock = maybeGetUnconditionalTarget(currentBlock);
            SLANG_ASSERT(nextRegionBlock);

            builder.emitBranch(nextRegionBlock);

            builder.setInsertInto(currentBlock);
            currentBlock->getTerminator()->removeAndDeallocate();
            builder.emitBranch(block);

            block->insertAfter(currentBlock);

            currentBlock = block;
            currBaseRegion = true;
            currBreakRegion = true;
        }

        return RegionEndpoint(currentBlock, currBreakRegion, currBaseRegion);
    }

    HashSet<IRBlock*> getPredecessorSet(IRBlock* block)
    {
        HashSet<IRBlock*> predecessorSet;
        for (auto predecessor : block->getPredecessors())
            predecessorSet.add(predecessor);

        return predecessorSet;
    }

    bool isLoopTrivial(IRLoop* loop)
    {
        // Get 'looping' block (first block in loop)
        auto firstLoopBlock = loop->getTargetBlock();

        // If we only have one predecessor, the loop is trivial.
        return (getPredecessorSet(firstLoopBlock).getCount() == 1);
    }

    IRBlock* normalizeBreakableRegion(IRInst* branchInst)
    {
        IRBuilder builder(cfgContext.module);

        switch (branchInst->getOp())
        {
        case kIROp_loop:
            {
                BreakableRegionInfo info;
                info.breakBlock = as<IRLoop>(branchInst)->getBreakBlock();
                info.headerBlock = as<IRBlock>(branchInst->getParent());

                // Emit var into parent block.
                builder.setInsertBefore(as<IRBlock>(branchInst->getParent())->getTerminator());

                // Create and initialize break var to true
                // true -> no break yet.
                // false -> atleast one break statement hit.
                //
                info.breakVar = builder.emitVar(builder.getBoolType());
                builder.addNameHintDecoration(info.breakVar, UnownedStringSlice("_bflag"));
                builder.emitStore(info.breakVar, builder.getBoolValue(true));

                // If the loop is trivial (i.e. single iteration, with no
                // edges actually in a loop), we're just going to remove
                // it.. (we can do this, because the normalization pass
                // will transform any break and continue statements)
                //
                if (isLoopTrivial(as<IRLoop>(branchInst)))
                {
                    auto firstLoopBlock = as<IRLoop>(branchInst)->getTargetBlock();

                    // Normalize the region from the first loop block till break.
                    auto preBreakEndPoint = getNormalizedRegionEndpoint(
                        &info,
                        firstLoopBlock,
                        List<IRBlock*>(info.breakBlock));

                    // Should not be empty.. but check anyway
                    SLANG_RELEASE_ASSERT(!preBreakEndPoint.isRegionEmpty);

                    // Quick consistency check.. preBreakEndPoint should be
                    // branching into break block.
                    SLANG_RELEASE_ASSERT(
                        as<IRUnconditionalBranch>(preBreakEndPoint.exitBlock->getTerminator())
                            ->getTargetBlock() == info.breakBlock);

                    auto currentBlock = branchInst->getParent();

                    // Now get rid of the loop inst and replace with unconditional branch.
                    branchInst->removeAndDeallocate();
                    builder.setInsertInto(currentBlock);
                    builder.emitBranch(firstLoopBlock);

                    return info.breakBlock;
                }

                auto condBlock = getOrCreateTopLevelCondition(as<IRLoop>(branchInst));

                auto ifElse = as<IRIfElse>(condBlock->getTerminator());

                auto trueEndPoint = getNormalizedRegionEndpoint(
                    &info,
                    ifElse->getTrueBlock(),
                    List<IRBlock*>(condBlock, info.breakBlock));

                auto falseEndPoint = getNormalizedRegionEndpoint(
                    &info,
                    ifElse->getFalseBlock(),
                    List<IRBlock*>(condBlock, info.breakBlock));

                RegionEndpoint loopEndPoint;
                bool isLoopOnTrueSide = true;

                // First figure out which side belongs to the loop body.
                if (isSuccessorBlock(trueEndPoint.exitBlock, condBlock))
                {
                    loopEndPoint = trueEndPoint;
                    isLoopOnTrueSide = true;
                }

                if (isSuccessorBlock(falseEndPoint.exitBlock, condBlock))
                {
                    loopEndPoint = falseEndPoint;
                    isLoopOnTrueSide = false;
                }

                // Right now, we only support loops where the loop is on the true side of
                // the condition. If we encounter the other case, flip the condition.
                //
                if (!isLoopOnTrueSide)
                {
                    IRBuilderInsertLocScope locScope{&builder};
                    // Invert the cond
                    builder.setInsertBefore(ifElse);
                    const auto c = ifElse->getCondition();
                    const auto negatedCond = c->getOp() == kIROp_Not
                                                 ? c->getOperand(0)
                                                 : builder.emitNot(builder.getBoolType(), c);
                    ifElse->condition.set(negatedCond);
                    const auto t = ifElse->getTrueBlock();
                    const auto f = ifElse->getFalseBlock();
                    ifElse->trueBlock.set(f);
                    ifElse->falseBlock.set(t);

                    // Invert our discovered state
                    std::swap(trueEndPoint, falseEndPoint);
                    isLoopOnTrueSide = true;
                }

                // Expect atleast one basic block (other than the condition block), in
                // the loop.
                //
                SLANG_RELEASE_ASSERT(loopEndPoint.exitBlock);
                SLANG_RELEASE_ASSERT(!loopEndPoint.isRegionEmpty);

                // Does the loop endpoint have both 'break' and 'base'
                // control flows?
                //
                if (loopEndPoint.inBaseRegion && loopEndPoint.inBreakRegion)
                {
                    // Add a test for the break variable into the condition.
                    auto cond = ifElse->getCondition();

                    builder.setInsertBefore(ifElse);
                    auto breakFlagVal = builder.emitLoad(info.breakVar);

                    // Need to invert the break flag if the loop is
                    // on the false side.
                    //
                    if (!isLoopOnTrueSide)
                    {
                        IRInst* args[1] = {breakFlagVal};
                        breakFlagVal =
                            builder.emitIntrinsicInst(builder.getBoolType(), kIROp_Not, 1, args);
                    }

                    IRInst* args[2] = {cond, breakFlagVal};

                    // If break-var = true, direct flow to the loop
                    // otherwise, direct flow to break
                    //
                    auto complexCond =
                        builder.emitIntrinsicInst(builder.getBoolType(), kIROp_And, 2, args);

                    ifElse->condition.set(complexCond);
                }

                return info.breakBlock;
            }
        case kIROp_Switch:
            {
                auto switchInst = as<IRSwitch>(branchInst);

                // SLANG_UNEXPECTED("Switch-case normalization not implemented yet.");
                BreakableRegionInfo info;
                info.breakBlock = as<IRSwitch>(branchInst)->getBreakLabel();
                info.headerBlock = as<IRBlock>(branchInst->getParent());

                // Emit var into parent block.
                builder.setInsertBefore(as<IRBlock>(branchInst->getParent())->getTerminator());

                // Create and initialize break var to true
                // true -> no break yet.
                // false -> atleast one break statement hit.
                //
                info.breakVar = builder.emitVar(builder.getBoolType());
                builder.emitStore(info.breakVar, builder.getBoolValue(true));

                // Go over case labels and normalize all sub-regions.
                for (UIndex ii = 0; ii < switchInst->getCaseCount(); ii++)
                {
                    auto caseBlock = switchInst->getCaseLabel(ii);
                    auto caseEndPoint = getNormalizedRegionEndpoint(
                                            &info,
                                            caseBlock,
                                            List<IRBlock*>(info.breakBlock))
                                            .exitBlock;

                    // Consistency check (if this case hits, it's probably
                    // because the switch has fall-through, which we don't support)
                    SLANG_RELEASE_ASSERT(
                        as<IRUnconditionalBranch>(caseEndPoint->getTerminator())
                            ->getTargetBlock() == info.breakBlock);
                }

                auto defaultEndPoint = getNormalizedRegionEndpoint(
                                           &info,
                                           switchInst->getDefaultLabel(),
                                           List<IRBlock*>(info.breakBlock))
                                           .exitBlock;

                // Consistency check (if this case hits, it's probably
                // because the switch has fall-through, which we don't support)
                SLANG_RELEASE_ASSERT(
                    as<IRUnconditionalBranch>(defaultEndPoint->getTerminator())->getTargetBlock() ==
                    info.breakBlock);

                return info.breakBlock;
            }
        default:
            break;
        }

        SLANG_UNEXPECTED("Unhandled control-flow inst");
    }
};

void normalizeCFG(
    IRModule* module,
    IRGlobalValueWithCode* func,
    IRCFGNormalizationPass const& options)
{
    // Remove phis to simplify our pass. We'll add them back in later
    // with constructSSA.
    //
    eliminatePhisInFunc(
        LivenessMode::Disabled,
        func->getModule(),
        func,
        PhiEliminationOptions::getFast());

    CFGNormalizationContext context = {module, options.sink};
    CFGNormalizationPass cfgPass(context);

    List<IRBlock*> workList;
    workList.add(func->getFirstBlock());

    while (workList.getCount() > 0)
    {
        auto block = workList.getLast();
        workList.removeLast();

        if (auto loop = as<IRLoop>(block->getTerminator()))
        {
            auto breakBlock = cfgPass.normalizeBreakableRegion(loop);
            workList.add(breakBlock);
        }
        else if (auto switchCase = as<IRSwitch>(block->getTerminator()))
        {
            auto breakBlock = cfgPass.normalizeBreakableRegion(switchCase);
            workList.add(breakBlock);
        }
        else
        {
            for (auto successor : block->getSuccessors())
                workList.add(successor);
        }
    }

    // After CFG normalization, there may be invalid uses of var/ssa values where the def
    // no longer dominate the use. We fix these up by going through the IR and create temp
    // vars for such uses.
    sortBlocksInFunc(func);
    legalizeDefUse(func);

    disableIRValidationAtInsert();
    constructSSA(module, func);
    enableIRValidationAtInsert();

    module->invalidateAnalysisForInst(func);
#if _DEBUG
    validateIRInst(maybeFindOuterGeneric(func));
#endif
}

} // namespace Slang
