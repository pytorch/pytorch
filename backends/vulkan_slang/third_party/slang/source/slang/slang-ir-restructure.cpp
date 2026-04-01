// ir-restructure.cpp
#include "slang-ir-restructure.h"

#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{
bool Region::isDescendentOf(Region* other)
{
    Region* rr = this;
    while (rr)
    {
        if (rr == other)
            return true;

        rr = rr->getParent();
    }
    return false;
}

bool Region::isDescendentOf(IRBlock* block)
{
    Region* rr = this;
    while (rr)
    {
        if (rr->getFlavor() == Region::Flavor::Simple)
        {
            SimpleRegion* simpleRegion = (SimpleRegion*)rr;
            if (simpleRegion->block == block)
                return true;
        }

        rr = rr->getParent();
    }
    return false;
}

/// An "active" label during control flow (re)structuring.
struct LabelStack
{
    /// Possible operations associated with labels.
    enum class Op
    {
        Break,
        Continue,

        CountOf,
    };

    /// What kind of operation does a branch to this label represent?
    Op op;

    /// The next label down on the stack
    LabelStack* parent;

    /// The block the represents this label in the IR control flow graph.
    IRBlock* block;

    /// The region that represents this label in the structured program
    Region* region;
};

/// State used when restructuring control flow.
struct ControlFlowRestructuringContext
{
    /// Sink to use when diagnosing errors in control-flow restructuring.
    ///
    /// The restructuring pass should be able to handle anything the front-end
    /// throws at it, so these errors will all be unexpected. Still, we need
    /// a way to report them cleanly without crashing the process.
    ///
    DiagnosticSink* sink = nullptr;
    DiagnosticSink* getSink() { return sink; }

    /// The region tree we are in the process of building.
    RegionTree* regionTree = nullptr;
};

/// Convert a range of blocks in the IR CFG into a region.
///
/// We want to generate a region that stands in for the
/// blocks that are logically in the internal [begin, end)
/// which we consider as representing a single-entry multiple-exit
/// sub-graph. Note that `end` is *not* part of the sub-graph,
/// but instead points to a block that is logically "after"
/// the sub-graph. `end` can be `null` to indicate that the
/// sub-graph extends as far as possible.
///
/// Because there can be multiple exits, control flow may
/// exit the sub-graph without branching to `end`, any
/// such "non-local" branching should be to one of the
/// blocks stored in the current `LabelStack`.
///
// TODO: Eventually we should replace all of this logic with
// a variation on the "Relooper" algorithm as it is used
// in Emscripten.
//
static RefPtr<Region> generateRegionsForIRBlocks(
    ControlFlowRestructuringContext* ctx,
    Region* inParentRegion,
    IRBlock* begin,
    IRBlock* end,
    LabelStack* initialLabels,    // Labels to use at the start
    LabelStack* labels = nullptr) // Labels to switch to after emitting first basic block
{
    if (!labels)
        labels = initialLabels;
    auto useLabels = initialLabels;

    //
    // We will try to build up as long of a sequential/simple region
    // as possible, to avoid deep recursion in this algorithm.
    //
    RefPtr<Region> resultRegion = nullptr;
    RefPtr<Region>* resultLink = &resultRegion;

    // As we move along, the parent region to use for regions
    // we create will shift, so we need a temporary to track
    // the current parent region.
    //
    Region* parentRegion = inParentRegion;

    //
    // We will start with the `begin` block, and try to proceed
    // sequentially until we see the `end` block, or run into
    // an edge that exits teh region.
    //
    IRBlock* block = begin;
    while (block != end)
    {
        // If the block we are trying to emit has been registered as a
        // destination label (e.g. for a loop or `switch`) then we
        // need to exit the current region, which amounts to generating
        // a `break` or `continue` operation.
        //
        // TODO: we eventually need to handle the possibility of
        // multi-level break/continue targets, which could be challenging.

        // Because we will only support single-level break/continue, we
        // want to resolve what is the most recent label that is "active"
        // for the given operation (`break` or `continue`).
        //
        // We will do this with a naive loop, just to keep things simple.
        // We start with no block "regsitered" as the target for each
        // operation.
        //
        IRBlock* registeredBlock[(int)LabelStack::Op::CountOf] = {};
        for (auto ll = useLabels; ll; ll = ll->parent)
        {
            // For each active label, see if it is the first one
            // we encounter for the given op.
            //
            if (!registeredBlock[(int)ll->op])
            {
                registeredBlock[(int)ll->op] = ll->block;
            }
        }

        // Next we will search through *all* of the registered labels,
        // and see if one of them matches the current `block`.
        //
        for (auto ll = useLabels; ll; ll = ll->parent)
        {
            // Does this label match the block we are trying to translate?
            if (ll->block != block)
                continue;

            // Okay, the block we are trying to generate code for is a label
            // that we should branch to (we shouldn't just emit the code here
            // and now...)
            //
            // We should first confirm that the block is the inner-most label
            // registered for the given control-flow op (`break` or `continue`)
            // because if it *isn't* we currently can't generate code.
            //
            if (block != registeredBlock[(int)ll->op])
            {
                if (ctx->getSink())
                    ctx->getSink()->diagnose(block, Diagnostics::multiLevelBreakUnsupported);
            }

            // Now we need to create a structured `break` or `continue` operation
            // to match the operation associated with the target.
            //
            switch (ll->op)
            {
            case LabelStack::Op::Break:
                {
                    auto outerRegion = (BreakableRegion*)ll->region;
                    RefPtr<BreakRegion> breakRegion = new BreakRegion(parentRegion, outerRegion);

                    *resultLink = breakRegion;
                    resultLink = nullptr;
                }
                break;

            case LabelStack::Op::Continue:
                {
                    auto outerRegion = (LoopRegion*)ll->region;
                    RefPtr<ContinueRegion> continueRegion =
                        new ContinueRegion(parentRegion, outerRegion);

                    *resultLink = continueRegion;
                    resultLink = nullptr;
                }
                break;
            }

            // If the `block` matched an active label, then we should have
            // created a branch, and there is nothing to be done here.
            return resultRegion;
        }

        // We now know that the given `block` is part of our control-flow region,
        // so we need to output a simple region that executes the code in that block.
        //
        RefPtr<SimpleRegion> simpleRegion = new SimpleRegion(parentRegion, block);

        // We need to register the mapping from `block` to this region, but in
        // general this isn't a one-to-one mapping, but rather one-to-many.
        // This is because a "continue clause" in a `for` loop might get duplicated
        // at each `continue` site in the output code. To deal with this
        // we build a singly-linked list of regions for each block.
        //
        // TODO: confirm that continue clauses are the only case that leads
        // to duplication.
        //
        // TODO: remove this workaround once we have a more powerful restructuring
        // pass that avoids duplicating blocks (by introducing new temporaries...)
        //
        SimpleRegion* nextSimpleRegionForSameBlock = nullptr;
        ctx->regionTree->mapBlockToRegion.tryGetValue(block, nextSimpleRegionForSameBlock);
        ctx->regionTree->mapBlockToRegion[block] = simpleRegion;

        *resultLink = simpleRegion;
        resultLink = &simpleRegion->nextRegion;
        parentRegion = simpleRegion;

        // The simple region we created will represent all of the non-terminator
        // instructions in the `block`, so now we need to figure out what to
        // create to represent that terminator.
        //
        auto terminator = block->getTerminator();
        SLANG_ASSERT(terminator != nullptr);
        switch (terminator->getOp())
        {
        default:
        case kIROp_conditionalBranch:
            // Note: we don't currently generate ordinary `conditionalBranch` instructions,
            // and instead only generate `ifElse` instructions, which include additional
            // information that can inform our control-flow restructuring pass.
            //
            SLANG_UNEXPECTED("unhandled terminator instruction opcode");
            [[fallthrough]];
        case kIROp_Unreachable:
        case kIROp_MissingReturn:
        case kIROp_Return:
        case kIROp_GenericAsm:
            // These cases are all simple terminators that can be handled as-is
            // without needing to construct a separate `Region` to encapsulate them.
            //
            // We will cap off the current sequence of simple regions and return.
            //
            *resultLink = nullptr;
            return resultRegion;

        case kIROp_ifElse:
            {
                // Here we have a two-way branch, so that we will construct a
                // region representing an `if` statement.
                //
                auto ifInst = (IRIfElse*)terminator;
                auto trueBlock = ifInst->getTrueBlock();
                auto falseBlock = ifInst->getFalseBlock();
                auto afterBlock = ifInst->getAfterBlock();


                RefPtr<IfRegion> ifRegion = new IfRegion(parentRegion, ifInst);

                // The region for the "then" part of things will consist of
                // the range of blocks `[trueBlock, afterBlock)`.
                //
                // This logic assumes that `afterBlock` is a valid structured
                // "join point" such that any branch out of the sub-region
                // either leads to `afterBlock` *or* one of the labels
                // that is already present on our label stack.
                //
                ifRegion->thenRegion =
                    generateRegionsForIRBlocks(ctx, ifRegion, trueBlock, afterBlock, labels);

                // Generating a region for the `else` part is similar.
                // Note that it is possible for this to be a `null`
                // region, if `falseBlock == afterBlock`.
                //
                ifRegion->elseRegion =
                    generateRegionsForIRBlocks(ctx, ifRegion, falseBlock, afterBlock, labels);

                *resultLink = ifRegion;
                resultLink = &ifRegion->nextRegion;
                parentRegion = ifRegion;

                // Continue with the block after the `ifElse` instruction.
                block = afterBlock;
            }
            break;

        case kIROp_loop:
            {
                // The terminator in this case is the header for a structured loop.
                //
                auto loopInst = (IRLoop*)terminator;
                auto bodyBlock = loopInst->getTargetBlock();
                auto afterBlock = loopInst->getBreakBlock();

                RefPtr<LoopRegion> loopRegion = new LoopRegion(parentRegion, loopInst);

                // We will need to set up entries on our label stack to
                // represent the targets for `break` or `continue`
                // operations inside the loop.
                //
                // First we set up the stack entry for the `break` label,
                // which will refer to the block *after* the loop.
                //
                // The region we specify for the label will still be
                // the loop region, though, because the loop is what
                // we are breaking out of.
                //
                LabelStack loopBreakLabelStack;
                loopBreakLabelStack.parent = labels;
                loopBreakLabelStack.block = afterBlock;
                loopBreakLabelStack.region = loopRegion;
                loopBreakLabelStack.op = LabelStack::Op::Break;

                //
                // The `continue` label warrants a bit more careful explanation,
                // because it will *not* refer to the block that was regsitered
                // as the continue target in the IR `loop` instruction. This
                // is because we will always emit our loops as `for(;;) { ... }`
                // with no continue clause at all, so that a `continue` in
                // the output code will always refer to the top of the loop.
                //
                // This means that the `continue` label for the purposes of
                // structured control flow will be the start of the loop body:
                //
                LabelStack loopContinueLabelStack;
                loopContinueLabelStack.parent = &loopBreakLabelStack;
                loopContinueLabelStack.block = bodyBlock;
                loopContinueLabelStack.region = loopRegion;
                loopContinueLabelStack.op = LabelStack::Op::Continue;
                //
                // Note: by ignoring the original continue block from the
                // high-level loop, we create a situation where that code
                // might get emitted more than once (once per implicit
                // or explicit `continue` site in the original program).
                //
                // That is an acceptable trade-off for now, because continue
                // blocks will usually be small (and fxc makes the same choice),
                // but it could lead to Bad Things if somebody were to call
                // a function in their continue clause, and that function does
                // a compute shader barrier operation.
                //
                // A better long-term fix is to take a high-level loop like:
                //
                //      for(A; B; C) { ... continue; ... break; ... }
                //
                // and translate it into something like the following (assuming
                // we have labeled statements and multi-level `break`):
                //
                //      A;
                //      Outer: for(;;) {
                //          Inner: for(;;) {
                //              if(B) {} else break Outer;
                //              ...
                //              break Inner; // `continue` becomes break of inner loop
                //              ...
                //              break Outer; // `break` becomes break of outer loop
                //              ...
                //              break; // inner loop unconditionally breaks at the end
                //          }
                //          C;  // continue clause comes after inner loop
                //      }
                //
                // If you draw up a control flow graph for that code, you'll find
                // it is equivalent to the orignal `for` loop, but now supports
                // arbitrary code (not just a single expression) for the continue clause.
                // Unlike the current code-duplication solution, `C` appears only once
                // in the output, and seems to clearly be at a "joint point" for control
                // flow so that it is clear that a barrier there is valid in GLSL.
                //
                // Anyway, back our regularly scheduled programming.
                //
                // With the label stack stuff set up, we want to take the region
                // of the CFG defined by `[bodyBlock, afterBlock)` and turn it into
                // the body region for our loop.
                //
                // The only thing we want to be a little bit careful about is
                // that we don't want the logic at the top of this function
                // that looks for a block it can translate into a `continue`
                // to trigger on `bodyBlock`, since that means we'd just turn
                // the whole body into a single `continue`.
                //
                // To avoid this problem, we pass in two different label stacks:
                // one to use for the first block, and one to use for subsequent
                // blocks.
                //
                loopRegion->body = generateRegionsForIRBlocks(
                    ctx,
                    loopRegion,
                    bodyBlock,
                    // TODO: should we pass `afterBlock` here instead of `null`?
                    nullptr,
                    // For the first block, we only want the `break` label active
                    &loopBreakLabelStack,
                    // After the first block, we can safely use the `continue` label too
                    &loopContinueLabelStack);

                *resultLink = loopRegion;
                resultLink = &loopRegion->nextRegion;
                parentRegion = loopRegion;

                // Continue with the block after the loop
                block = afterBlock;
            }
            break;

        case kIROp_unconditionalBranch:
            {
                // Here we have an unconditional branch that was
                // not covered by one of our labels for non-local
                // branches (`break` or `continue`).
                //
                // We will thus assume that the target of the
                // branch is part of the same region we are building,
                // and continue with the target block;
                //
                auto branchInst = (IRUnconditionalBranch*)terminator;
                block = branchInst->getTargetBlock();
            }
            break;

        case kIROp_Switch:
            {
                // A `switch` instruction will always translate
                // to a `SwitchRegion` and then to a `switch` statement.
                //
                // We will need to take care to emit `case`s in ways
                // that avoid code duplication.
                //
                // The logic here isn't going to be robust in edge cases
                // (please don't write Duff's Device in Slang just yet).
                // Doing significantly better than what is here would
                // require something like the Relooper algorithm, though.
                //
                auto switchInst = (IRSwitch*)terminator;
                auto breakLabel = switchInst->getBreakLabel();
                auto defaultLabel = switchInst->getDefaultLabel();

                RefPtr<SwitchRegion> switchRegion = new SwitchRegion(parentRegion, switchInst);

                // A direct branch to the block after the `switch` can
                // be emitted as a `break` statement, so we will register
                // the appropriate label on a label stack:
                //
                LabelStack switchBreakLabelStack;
                switchBreakLabelStack.parent = labels;
                switchBreakLabelStack.op = LabelStack::Op::Break;
                switchBreakLabelStack.block = breakLabel;
                switchBreakLabelStack.region = switchRegion;

                // We need to track whether we've dealt with
                // the `default` case already.
                //
                bool defaultLabelHandled = false;

                // If the `default` case just branches to
                // the join point, then we don't need to
                // do anything with it.
                //
                if (defaultLabel == breakLabel)
                    defaultLabelHandled = true;

                // We will now iterate over the different `case`s, and
                // try to group them together to minimize the number of
                // sub-regions we have to create.
                //
                UInt caseIndex = 0;
                UInt caseCount = switchInst->getCaseCount();
                while (caseIndex < caseCount)
                {
                    // We are going to extract one case here,
                    // but we might need to fold additional
                    // cases into it, if they share the
                    // same label.
                    //
                    // Note: this makes assumptions that the
                    // IR code generator orders cases such
                    // that: (1) cases with the same label
                    // are consecutive, and (2) any case
                    // that "falls through" to another must
                    // come right before it in the list.

                    auto caseVal = switchInst->getCaseValue(caseIndex);
                    auto caseLabel = switchInst->getCaseLabel(caseIndex);
                    caseIndex++;

                    RefPtr<SwitchRegion::Case> currentCase = new SwitchRegion::Case();
                    switchRegion->cases.add(currentCase);

                    // Add the case value for this case, and any
                    // others that share the same label
                    //
                    for (;;)
                    {
                        currentCase->values.add(caseVal);

                        // Are there any more `case`s left?
                        //
                        if (caseIndex >= caseCount)
                            break;

                        // Does the next `case` share the same target label?
                        auto nextCaseLabel = switchInst->getCaseLabel(caseIndex);
                        if (nextCaseLabel != caseLabel)
                            break;

                        // If those checks passed, then we will fold
                        // the next `case` into the same region, and
                        // keep looking.
                        caseVal = switchInst->getCaseValue(caseIndex);
                        caseIndex++;
                    }

                    // The label for the current `case` might also
                    // be the label used by the `default` case, so
                    // check for that here.
                    //
                    if (caseLabel == defaultLabel)
                    {
                        switchRegion->defaultCase = currentCase;
                        defaultLabelHandled = true;
                    }

                    // Now we need to generate a region for the instructions
                    // that make up this case. The 99% case will be that it
                    // will terminate with a `break` (or a `return`,
                    // `continue`, etc.) and so we can pass in `nullptr`
                    // for the ending block.
                    //
                    IRBlock* caseEndLabel = nullptr;

                    // However, there is also the possibility that
                    // this `case` will fall through to the next, and
                    // so we need to prepare for that possibility here.
                    //
                    // If there *is* a next `case`, then we will set its
                    // label up as the "end" label when emitting
                    // the statements inside the block.
                    if (caseIndex < caseCount)
                    {
                        caseEndLabel = switchInst->getCaseLabel(caseIndex);
                    }

                    // Now we can actually generate the region.
                    //
                    currentCase->body = generateRegionsForIRBlocks(
                        ctx,
                        switchRegion,
                        caseLabel,
                        caseEndLabel,
                        &switchBreakLabelStack);
                }

                // If we've gone through all the cases and haven't
                // managed to encounter the `default:` label,
                // then assume it is a distinct case and handle it here.
                if (!defaultLabelHandled)
                {
                    RefPtr<SwitchRegion::Case> defaultCase = new SwitchRegion::Case();
                    switchRegion->cases.add(defaultCase);

                    // Note: we use `null` instead of `breakLabel` as the end block
                    // here, to ensure that the `default` region will end with an
                    // explicit `break` rather than just falling off the end.

                    defaultCase->body = generateRegionsForIRBlocks(
                        ctx,
                        switchRegion,
                        defaultLabel,
                        nullptr,
                        &switchBreakLabelStack);

                    switchRegion->defaultCase = defaultCase;
                }

                *resultLink = switchRegion;
                resultLink = &switchRegion->nextRegion;
                parentRegion = switchRegion;

                // Continue with the block after the `switch`
                block = breakLabel;
            }
            break;
        }

        // After we've emitted the first block, we are safe from accidental
        // cases where we'd emit an entire loop body as a single `continue`,
        // so we can safely switch in whatever labels are intended to be used.
        useLabels = labels;

        // If we reach this point, then we've emitted
        // one block, and we have a new block where
        // control flow continues.
        //
        // We need to handle a special case here,
        // when control flow jumps back to the
        // starting block of the range we were
        // asked to work with:
        if (block == begin)
        {
            break;
        }
    }

    // We seem to have reached the rend of the region
    // without anything special happening. This means
    // we should cap off the current sequence of regions
    // and return what we have.
    //
    *resultLink = nullptr;
    return resultRegion;
}

RefPtr<RegionTree> generateRegionTreeForFunc(IRGlobalValueWithCode* code, DiagnosticSink* sink)
{
    RefPtr<RegionTree> regionTree = new RegionTree();
    regionTree->irCode = code;

    ControlFlowRestructuringContext restructuringContext;
    restructuringContext.sink = sink;
    restructuringContext.regionTree = regionTree;

    regionTree->rootRegion = generateRegionsForIRBlocks(
        &restructuringContext,
        nullptr,
        code->getFirstBlock(),
        nullptr,
        nullptr);

    return regionTree;
}
} // namespace Slang
