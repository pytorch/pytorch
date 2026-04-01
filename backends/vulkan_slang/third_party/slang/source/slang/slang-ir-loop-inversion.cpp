#include "slang-ir-loop-inversion.h"

#include "slang-ir-clone.h"
#include "slang-ir-dominators.h"
#include "slang-ir-insts.h"
#include "slang-ir-lower-witness-lookup.h"
#include "slang-ir-reachability.h"
#include "slang-ir-ssa-simplification.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{

static bool isSameBlockOrTrivialBranch(IRBlock* target, IRBlock* scrutinee)
{
    if (target == scrutinee)
        return true;
    const auto br = as<IRUnconditionalBranch>(scrutinee->getFirstOrdinaryInst());
    return br && br->getTargetBlock() == target && br->getArgCount() == 0 &&
           !scrutinee->hasMoreThanOneUse();
};

static bool isSmallBlock(IRBlock* c)
{
    // Somewhat arbitrarily, 4 instructions, enough for:
    // - Arith
    // - Comparison
    // - Negation
    // - Terminator
    Int n = 0;
    for ([[maybe_unused]] const auto i : c->getOrdinaryInsts())
        if (++n > 4)
            return false;
    return true;
}

static bool hasIrrelevantContinueBlock(IRLoop* loop)
{
    const auto c = loop->getContinueBlock();
    return c == loop->getTargetBlock() || c->getPredecessors().getCount() <= 1;
}

// Loops are suitable for inversion if:
// - The loop jumps to a conditional branch which has the break block as one of
//   its successors (or a trivial break block which we erase) and the other
//   successor is empty
// - The conditional block is "small", because we will be duplicating it
// - The loop's continue block is irrelevant, because we'll need to change it,
//   either by being the loop header already or by having only a single use
//   within the loop body
static bool isSuitableForInversion(IRLoop* loop)
{
    const auto nextBlock = loop->getTargetBlock();
    const auto breakBlock = loop->getBreakBlock();

    // The first thing a loop does must be a conditional
    const auto branch = as<IRIfElse>(nextBlock->getTerminator());
    if (!branch)
        return false;

    if (!isSmallBlock(nextBlock))
        return false;

    if (!hasIrrelevantContinueBlock(loop))
        return false;

    const auto t = branch->getTrueBlock();
    const auto f = branch->getFalseBlock();
    const auto a = branch->getAfterBlock();

    //
    // In principle we could perform this simplification in the cfg simplifier,
    // however it relies on slightly more context than is simple to insert
    // there, namely that the removed trivial branching block is branching to a
    // loop break block.
    //

    // Do we break on the 'true' side?
    if (isSameBlockOrTrivialBranch(breakBlock, t) && f == a)
    {
        if (t != breakBlock)
        {
            branch->trueBlock.set(breakBlock);
            t->removeAndDeallocate();
        }
        return true;
    }

    // ... or the false side
    if (isSameBlockOrTrivialBranch(breakBlock, f) && t == a)
    {
        if (f != breakBlock)
        {
            branch->falseBlock.set(breakBlock);
            f->removeAndDeallocate();
        }
        return true;
    }

    return false;
}

static IRParam* duplicateToParamWithDecorations(IRBuilder& builder, IRCloneEnv& cloneEnv, IRInst* i)
{
    const auto p = builder.emitParam(i->getFullType());
    for (const auto dec : i->getDecorations())
        cloneDecoration(&cloneEnv, dec, p, builder.getModule());
    return p;
}

// Given
// s: ...1 loop break=b next=c1
// c1: if x then goto b else goto d (merge at d)
// d: goto c1
// b: ...2
//
// Produce:
// s: ...1 goto c2
// c2: if x then goto e2 else goto l (merge at b)
// e2: goto b
// l: loop break=b2 next=d
// d: goto c1:
// c1: if x then goto e1 else goto e3 (merge at e3)
// e3: goto d
// e1: goto b2
// b2: goto b
// b: ...2
//
// s is the Start block
// c1, c2 are the Condition blocks
// e1, e2, e3 are the critical Edge breakers
// l is the loop entering block
// d is the loop body
// b is the merge point for the outer condition
// b2 is the new break block for the loop
//
static void invertLoop(IRBuilder& builder, IRLoop* loop)
{
    IRBuilderInsertLocScope builderScope(&builder);
    builder.setInsertInto(loop->getParent());

    const auto s = as<IRBlock>(loop->getParent());
    auto domTree = computeDominatorTree((IRGlobalValueWithCode*)s->getParent());
    SLANG_ASSERT(s);
    const auto c1 = loop->getTargetBlock();
    const auto c1Terminator = as<IRIfElse>(c1->getTerminator());
    SLANG_ASSERT(c1Terminator);
    const auto b = loop->getBreakBlock();
    auto& c1dUse =
        c1Terminator->getTrueBlock() == b ? c1Terminator->falseBlock : c1Terminator->trueBlock;
    auto& c1bUse =
        c1Terminator->getTrueBlock() == b ? c1Terminator->trueBlock : c1Terminator->falseBlock;
    const auto d = as<IRBlock>(c1dUse.get());
    SLANG_ASSERT(d);

    IRCloneEnv cloneEnv;
    cloneEnv.squashChildrenMapping = true;

    // We don't expect 'd' to have any parameters, because it used to be the
    // target of a conditional branch
    SLANG_ASSERT(d->getFirstParam() == nullptr);

    // Since we are duplicating the loop break condition block (c1) we must
    // introduce phi values for anything in it upon which the rest of the
    // program (inside the loop, and b onwards) uses. Lift the values from c1
    // used in b (and onwards) to parameters, the same for those used before b.
    // To avoid a critical edge, pass these via a new block, e1.
    // For any such values used within the loop we can pass directly to d.
    //
    // c1PostLoopParams are values form c1 used after the loop
    List<IRInst*> c1PostLoopParams;
    // c1LoopParams are values from c1 used within the loop itself
    List<IRInst*> c1LoopParams;
    for (auto i : IRInstList<IRInst>(c1->getFirstInst(), c1->getLastInst()))
    {
        IRParam* postLoopParam = nullptr;
        IRParam* loopParam = nullptr;
        traverseUses(
            i,
            [&](IRUse* u)
            {
                auto userBlock = u->getUser()->getParent();
                if (domTree->dominates(b, userBlock))
                {
                    // A new parameter to replace this 'i'
                    if (!postLoopParam)
                    {
                        postLoopParam = duplicateToParamWithDecorations(builder, cloneEnv, i);
                        b->addParam(postLoopParam);
                    }
                    u->set(postLoopParam);
                }
                else if (userBlock != c1)
                {
                    // A new parameter to replace this 'i'
                    if (!loopParam)
                    {
                        loopParam = duplicateToParamWithDecorations(builder, cloneEnv, i);
                        d->addParam(loopParam);
                    }
                    u->set(loopParam);
                }
            });
        if (postLoopParam)
            c1PostLoopParams.add(i);
        if (loopParam)
            c1LoopParams.add(i);
    }

    // Create another break block b2 that will act as the new break block for the
    // loop. The original break block b will become the merge point for the outer condition.
    //
    const auto b2 = builder.emitBlock();
    b2->insertBefore(b);

    // Create a copy of the parameters in b. b2 will simply pass these to b.
    List<IRInst*> b2Params;
    for (auto p : b->getParams())
    {
        auto q = duplicateToParamWithDecorations(builder, cloneEnv, p);
        b2Params.add(q);
    }
    builder.setInsertInto(b2);
    builder.emitBranch(b, b2Params.getCount(), b2Params.getBuffer());

    auto e1 = builder.emitBlock();
    e1->insertAfter(c1);
    builder.emitBranch(b2, c1PostLoopParams.getCount(), c1PostLoopParams.getBuffer());
    c1bUse.set(e1);
    c1Terminator->afterBlock.set(d);

    // We'll rewrite existing breaks to jump to b2, but via an intermediate jump block to
    // avoid a critical edge.
    //
    traverseUses(
        b,
        [&](IRUse* u)
        {
            auto userBlock = u->getUser()->getParent();
            // Restrict this to just those blocks within this loop
            if (userBlock != e1 && userBlock != b2 && userBlock != s &&
                domTree->dominates(s, userBlock) && !domTree->dominates(b, userBlock))
            {
                const auto jumpToB2Block = builder.emitBlock();
                jumpToB2Block->insertAfter(userBlock);
                builder.emitBranch(b2, c1PostLoopParams.getCount(), c1PostLoopParams.getBuffer());
                u->set(jumpToB2Block);
            }
        });

    // We now have
    // s: ...1 loop break=b next=c1
    // c1: if x then goto e1 else goto d (merge at d)
    // e1: goto b2
    // d: goto c1
    // b2: goto b
    // b: ...2

    // Duplicate c1 into c2, and using the same cloneEnv, duplicate e1 into e2
    builder.setInsertInto(builder.getFunc());
    const auto c2 = as<IRBlock>(cloneInst(&cloneEnv, &builder, c1));
    c2->insertBefore(c1);
    const auto e2 = as<IRBlock>(cloneInst(&cloneEnv, &builder, e1));
    e2->insertAfter(c2);
    const auto c2Terminator = as<IRConditionalBranch>(c2->getTerminator());
    auto& c2eUse =
        c2Terminator->getTrueBlock() == e1 ? c2Terminator->trueBlock : c2Terminator->falseBlock;
    c2eUse.set(e2);
    builder.setInsertAfter(c2Terminator);
    const auto newC2Terminator = builder.emitIfElse(
        c2Terminator->getCondition(),
        c2Terminator->getTrueBlock(),
        c2Terminator->getFalseBlock(),
        b);
    c2Terminator->removeAndDeallocate();
    // The cloned e2 will branch into b2 by default, rewrite it to branch to b, the correct merge
    // point.
    SLANG_ASSERT(cast<IRUnconditionalBranch>(e2->getTerminator())->getTargetBlock() == b2);
    cast<IRUnconditionalBranch>(e2->getTerminator())->block.set(b);

    // We now have
    // s: ...1 loop break=b next=c1
    // c2: if x then goto e2 else goto d (merge at b)
    // e2: goto b
    // c1: if x then goto e1 else goto d (merge at d)
    // e1: goto b2
    // d: goto c1
    // b2: goto b
    // b: ...2

    // move the loop instruction to its own block, l
    const auto l = builder.emitBlock();
    l->insertAfter(e2);
    loop->insertAtEnd(l);

    // We now have
    // s: ...1 no-termiator
    // c2: if x then goto e2 else goto d (merge at b)
    // e2: goto b
    // l: loop break=b next=c1
    // c1: if x then goto e1 else goto d (merge at d)
    // e1: goto b2
    // d: goto c1
    // b2: goto b
    // b: ...2

    // add a new terminator to s. A jump to c2, our outer conditional. retain
    // any parameters the loop instruction passed to c1
    builder.setInsertInto(s);
    List<IRInst*> as;
    for (UInt i = 0; i < loop->getArgCount(); ++i)
        as.add(loop->getArg(i));
    builder.emitBranch(c2, as.getCount(), as.getBuffer());
    // We now have
    // s: ...1, goto c2
    // c2: if x then goto e2 else goto d (merge at b)
    // e2: goto b
    // l: loop break=b next=c1
    // c1: if x then goto e1 else goto d (merge at d)
    // e1: goto b2
    // d: goto c1
    // b2: goto b
    // b: ...2

    // modify c2 to jump to the new loop
    auto& c2dUse = newC2Terminator->getTrueBlock() == e2 ? newC2Terminator->falseBlock
                                                         : newC2Terminator->trueBlock;
    c2dUse.set(l);
    // We now have
    // s: ...1, goto c2
    // c2: if x then goto e2 else goto l (merge at b)
    // e2: goto b
    // l: loop break=b next=c1
    // c1: if x then goto e1 else goto d (merge at d)
    // e1: goto b2
    // d: goto c1
    // b2: goto b
    // b: ...2

    //
    // Now we can modify the loop to jump to the block after the first
    // conditional, d, as we know that it won't break out of the loop on the
    // first iteration
    //
    // Since we're only here if the continue block is irrelevant (either the
    // target block already or has a single predecessor) we can set it to the
    // loop header.
    //
    // Beyond just retargeting the loop instruction, we need to make sure any
    // parameters the loop instruction is passing to c1 are instead passed to
    // 'd', and because we've added parameters to 'd' we need to forward them
    // from c1 also during the back-edge, which we will accomplish using a new
    // block, e3,
    //
    // Utilize the cloneenv to make sure that when entering the loop we use
    // c1's instructions as cloned into c2
    builder.setInsertAfter(loop);
    List<IRInst*> loopEntryArgs;
    for (const auto p : c1LoopParams)
        loopEntryArgs.add(cloneInst(&cloneEnv, &builder, p));
    for (UInt i = 0; i < loop->getArgCount(); ++i)
        loopEntryArgs.add(loop->getArg(i));
    const auto newLoop = builder.emitLoop(
        d,
        b2,
        loop->getContinueBlock(),
        loopEntryArgs.getCount(),
        loopEntryArgs.getBuffer());
    newLoop->sourceLoc = loop->sourceLoc;
    loop->transferDecorationsTo(newLoop);
    loop->removeAndDeallocate();

    // TODO: This really upsets a few later passes, why isn't it ok to do given
    // our "irrelevant continue" condition?
    // loop->continueBlock.set(loop->getTargetBlock());
    c1->insertBefore(b2);
    e1->insertAfter(c1);
    builder.setInsertInto(d);
    // Add the necessary parameters for the loop state to d, first the
    // paramters for instructions in the duplicated conditional block, then the
    // ones from the loop body.
    List<IRInst*> ps = c1LoopParams;
    for (const auto p : c1->getParams())
    {
        ps.add(p);
        const auto q = duplicateToParamWithDecorations(builder, cloneEnv, p);
        // Replace all uses, except for those in c1 and e1
        List<IRUse*> uses;
        traverseUses(
            p,
            [&](IRUse* u)
            {
                if (u->user->getParent() != c1 && u->user->getParent() != e1)
                    uses.add(u);
            });
        for (auto u : uses)
            u->set(q);
    }
    const auto e3 = builder.emitBlock();
    e3->insertAfter(c1);
    b2->insertBefore(b);
    builder.emitBranch(d, ps.getCount(), ps.getBuffer());
    c1dUse.set(e3);
    c1Terminator->afterBlock.set(e3);
    // We now have the desired output
    // s: ...1, goto c2
    // c2: if x then goto e2 else goto l (merge at b)
    // e2: goto b
    // l: loop break=b2 next=d
    // d: goto c1
    // c1: if x then goto e1 else goto e3 (merge at e3)
    // e3: goto d
    // e1: goto b2
    // b2: goto b
    // b: ...2
}

bool invertLoops(IRModule* module)
{
    IRBuilder builder(module);
    List<IRLoop*> toInvert;
    overAllBlocks(
        module,
        [&](auto b)
        {
            if (auto loop = as<IRLoop>(b->getTerminator()))
            {
                if (isSuitableForInversion(loop))
                    toInvert.add(loop);
            }
        });
    for (const auto loop : toInvert)
        invertLoop(builder, loop);
    return toInvert.getCount() > 0;
}

} // namespace Slang
