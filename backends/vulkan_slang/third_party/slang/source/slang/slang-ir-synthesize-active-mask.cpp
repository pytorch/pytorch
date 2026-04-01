// slang-ir-synthesize-active-mask.cpp
#include "slang-ir-synthesize-active-mask.h"

#include "slang-ir-dominators.h"
#include "slang-ir-insts.h"

namespace Slang
{

// This file implements a pass on the IR to make the "active mask" concept
// from HLSL explicit in the code, so that we can generate code for
// targets like CUDA where explicit masks are used for warp-/wave-level
// collective operations.
//
// At the time this pass was written, the exact semantics of the implicit
// active mask in HLSL (and GLSL/SPIR-V) had not been specified, so this
// file had to define what the semantics ought to be.
//
// The main guideline for these semantics is that it should conform to
// the user's view of control flow based on the high-level language.
//
// The intuitive version of the semantics provided by this pass are:
//
// * Code is conceptually executed by "shards" of threads/lanes, where
//   each shard consists of a subset of the threads/lanes in the warp/wave
//   that are executing "together."
//
// * When a shard encounters a structured control-flow statement, the
//   shard may fragment into sub-shards that partition its set of threads/lanes.
//   E.g. for a two-way `if` we end up with a sub-shard for threads/lanes
//   where the condition was `true`, and another sub-shard for the
//   threads/lanes where the condition was `false`.
//
// * When the threads in a shard exit some structured control-flow
//   statement normally (e.g., execution reaches the end of the "then"
//   statement for an `if`), those threads *re-converge* with all the other
//   threads that entered the structured statement together and which
//   exited normally.
//
// * When threads in a shard exit some structured control-flow statement
//   "abnormally" (bypassing the statements that logically come after),
//   those threads do *not* re-converge with the other threads that exited
//   normally, although they may still re-converge at the normal exit of
//   some lexically outer statement.
//
// Turning these rules into something formal mostly comes down to codifying:
//
// 1. when threads exit a structured control flow statement, and
// 2. which of those exits are "normal" vs. "abnormal."
//
// The Slang IR maintains structure information in its IR (similar to the
// information maintained by SPIR-V for Vulkan). For a structured control
// flow statement like an `if`, the IR-level representation encodes both
// the conditional branch for the `if`, but also the label/block representing
// the code that logically comes after the `if` in the high-level-language
// code. We can refer to a block like that as a "merge point."
//
// Our more formal divergence and re-convergence behavior then depends
// on two definitions:
//
// 1. Code is "inside" a control-flow structure if it is dominated by
// the entry block, but not dominated by the merge block. Control flow
// thus exits the structure whenever it branchs from a block inside the
// structure to some place not inside the structure.
//
// 2. An edge that exits a control-flow structure represents a "normal"
// exit if it branches to the merge block for the structure, and an
// "abnormal" exit otherwise.
//
// These definitions result in divergence and re-convergence behavior
// that closely matches what programmers expect from their high-level
// language code.
//
// (Note: because the definition of what code is inside a control-flow
// structure depends on dominance in the CFG, the existence of arbitrary
// `goto` operations that can jump "into" a control-flow statement would
// break our assumptions. This pass would only be compatible with
// `goto` operations that jump "out" of control-flow statements.)

// The translation of code to make masks explicit can be broadly
// broken up into:
//
// * Analysis and transformation of the entire module and its call
//   graph, to identify and update functions that need an explicit
//   active mask.
//
// * Analysis and transformation of a single function, that was
//   identified by the above logic.
//
// We will start with the whole-module logic.
//
// As it typical for our IR passes, we will wrap up the module-level
// synthesis work in a "context" type.
//
struct SynthesizeActiveMaskForModuleContext
{
    // The context needs to know the module it will process, and to
    // have a sink where it can report any errors it might run into
    // (e.g., any cases where a mask is needed but cannot be synthesized)
    //
    IRModule* m_module;
    DiagnosticSink* m_sink;

    // Because we are introducing explicit masks, it is important
    // for all the code to agree on the type of mask that will be
    // used.
    //
    IRType* m_maskType;

    void processModule()
    {
        // We use the plain 32-bit `uint` type masks we
        // generate here since it matches the current
        // definition of `WaveMask` in the core module.
        //
        // TODO: If/when the `WaveMask` type in the core module is
        // made opaque, this should use the opaque type instead,
        // so that the pass is compatible with all targets that
        // support  a wave mask.
        //
        IRBuilder builder(m_module);
        m_maskType = builder.getBasicType(BaseType::UInt);

        // With the setup out of the way, the job of the module
        // level pass is simple to describe:
        //
        // First we want to identify and mark all of the functions
        // in the module that use the active mask.
        //
        markFuncsUsingActiveMask();
        //
        // Then we want to transform all of those functions that
        // were marked so that they use an explicitly synthesized
        // value for the active mask.
        //
        transformFuncsUsingActiveMask();
    }

    // During the marking process we will build up a list of functions
    // that need the active mask, and we will also maintain a set of
    // those functions so that we don't waste time redundantly considering
    // functions we've already marked.
    //
    List<IRFunc*> m_funcsUsingActiveMask;
    HashSet<IRFunc*> m_funcsUsingActiveMaskSet;

    // When the code finds an IR function that seems to use the active
    // mask, we will mark it by adding it to the list and set, but
    // only if we haven't already done so previously.
    //
    void markFuncUsingActiveMask(IRFunc* func)
    {
        if (m_funcsUsingActiveMaskSet.contains(func))
            return;

        m_funcsUsingActiveMask.add(func);
        m_funcsUsingActiveMaskSet.add(func);
    }

    // The easiest way to know that a function uses the active
    // mask is if it contains an *instruction* that uses the active
    // mask.
    //
    // Because it is easiest to detect use of the active mask at
    // an instruciton level, it is convenient to have a utility
    // that, given an instruction which uses the active mask,
    // marks the function that contains it (if any) as using
    // the active mask.
    //
    void markInstUsingActiveMask(IRInst* inst)
    {
        // We expect the immedaite parent of an ordinary
        // instruction to be a basic block (although in
        // unlikely cases an instruction can be directly nested
        // in the `IRModule`).
        //
        auto parent = inst->getParent();
        if (auto block = as<IRBlock>(parent))
        {
            parent = block->getParent();
        }
        //
        // We expect the immediate parent of that block
        // to be an IR function (again, in unlikely cases
        // the block could be nested in an IR generic,
        // or a global variable with an initializer, etc.).
        //
        if (auto func = as<IRFunc>(parent))
        {
            markFuncUsingActiveMask(func);
        }
    }

    void markFuncsUsingActiveMask()
    {
        // We break down the task of identifying all functions
        // that use the active mask into two steps.
        //
        // First we identify those functions that *directly* use
        // the active mask, by virtuae of having a `getActiveMask`
        // instruction in their body.
        //
        markFuncsDirectlyUsingActiveMaskRec(m_module->getModuleInst());

        // Second, we identify any function that indirectly use
        // the active mask.
        //
        // The detecting of indirect use is handled by looking
        // at all the functions we've already marked, and identifying
        // unmarked functions that call marked functions.
        //
        // We also modify these call sites (from unmarked functions
        // to marked functions) so that they pass along the explicit
        // active mask value to the callee.
        //
        // Note: we are iterating over `m_funcsUsingActiveMask` here
        // while also invoking code that could add additional
        // functions to that list. As such it is *intentional* that
        // we are querying `getCount()` on the list on every iteration
        // of the loop, rather than querying it once and using a
        // variable (or using a range-based `for` loop).
        //
        for (Index i = 0; i < m_funcsUsingActiveMask.getCount(); ++i)
        {
            markAndModifyFuncsIndirectlyUsingActiveMask(m_funcsUsingActiveMask[i]);
        }

        // TODO: The logic here isn't robust against cases where we
        // might have indirect calls.
        //
        // One possible solution would be to always treat indirect
        // calls as if they require an active mask, and pass one along.
        // Then all functions that might be used as the callee of an
        // indirect call site (those whose values "escape") would need
        // to have a variant that takes the active mask generated (if
        // they didn't actually need the active mask based on their body).
        // All function pointer values would be based on the mask-taking
        // variant.
        //
        // Another alternative would be to make dependency on the
        // active mask an explicit part of the type/signature of functions
        // in the case where masks need to be passed, so that even indirect
        // call sites could be decomposed into mask-taking and non-mask-taking
        // cases just based on type information.
        //
        // There are probably other options to consider, each with
        // its own trade-offs. This is a design question that we aren't
        // really equipped to tackle right now, so the easiest solution
        // is to defer the decision until we actually need an answer.
    }

    void markFuncsDirectlyUsingActiveMaskRec(IRInst* inst)
    {
        // Detecting functions that directly use the active
        // mask is fairly simple: we recursively walk the
        // IR module and look for instances of `waveGetActiveMask`
        // instructions. When we find one we mark the function
        // that contains it.

        if (inst->getOp() == kIROp_WaveGetActiveMask)
        {
            markInstUsingActiveMask(inst);
        }

        for (auto child : inst->getChildren())
        {
            markFuncsDirectlyUsingActiveMaskRec(child);
        }
    }

    void markAndModifyFuncsIndirectlyUsingActiveMask(IRFunc* callee)
    {
        // This transform does not apply to host or kernel callees.
        if (callee->findDecoration<IRCudaHostDecoration>() ||
            callee->findDecoration<IRCudaKernelDecoration>())
            return;

        // In order to detect functions that indirectly use the active
        // mask through `callee`, we need to identify call sites.
        //
        // We will build up a list of call sites during the marking
        // step, so that we can also modify those call sites later
        // in this function.

        List<IRCall*> calls;
        for (IRUse* use = callee->firstUse; use; use = use->nextUse)
        {
            // We are looking for instructions that use `callee`,
            // where the instruction is a call, and `callee` is used
            // as, well, the callee.
            //
            IRInst* user = use->getUser();
            IRCall* call = as<IRCall>(user);
            if (!call)
                continue;
            if (call->getCallee() != callee)
                continue;

            // Once we have found a call site to `callee`,
            // we mark the function that contains the call
            // site as one we need to process, and also
            // add the call site to our list of call sites
            // we need to modify.
            //
            markInstUsingActiveMask(call);
            calls.add(call);
        }

        // Once we've discovered all the call sites for `callee`,
        // we will go ahead and modify them.
        //
        // (We don't mix up marking and modification because
        // changing the use/def list of `callee` while also
        // walking it would be dangerous)
        //
        for (auto call : calls)
        {
            // The basic situation here is that we have a call of the form:
            //
            //      callee(arg0, arg1, arg2, ...);
            //
            // and we want to transform it to:
            //
            //      mask = waveGetActiveMask();
            //      callee(arg0, arg1, arg2, ..., m);
            //
            IRBuilder builder(m_module);
            builder.setInsertBefore(call);

            // First we synthesize the mask to pass down by
            // directly invoking `waveGetActiveMask()`.
            //
            // Note that after this instruction has been inserted,
            // the function containing `call` is now a function
            // that directly uses the active mask.
            //
            auto mask = builder.emitIntrinsicInst(
                builder.getBasicType(BaseType::UInt),
                kIROp_WaveGetActiveMask,
                0,
                nullptr);

            // Next we synthesize the new argument list for the
            // call, which consists of the original arguments,
            // followed by the `mask`.
            //
            List<IRInst*> newArgs;
            Int originalArgCount = call->getArgCount();
            for (Int a = 0; a < originalArgCount; ++a)
            {
                newArgs.add(call->getArg(a));
            }
            newArgs.add(mask);

            // Finally we emit a new call instruction to the same
            // callee with our new arguments, and use the new call
            // to replace the original `call`.
            //
            // Note: We can't just modify the `call` in-place because
            // of the way that the operands to an `IRInst` are allocated
            // contiguously in the same storage as the `IRInst` itself.
            //
            auto newCall = builder.emitCallInst(call->getFullType(), call->getCallee(), newArgs);
            call->replaceUsesWith(newCall);
            call->removeAndDeallocate();
        }
    }

    void transformFuncsUsingActiveMask()
    {
        // Once all the functions that need the active mask have
        // been marked (and all call sites to them have
        // been modified), we can transform each of the marked
        // functions one by one.
        //
        for (auto func : m_funcsUsingActiveMask)
        {
            transformFuncUsingActiveMask(func);
        }
    }

    // We will get to the details of how we transform each
    // function shortly, but for now we just forward-declare
    // the operation that makes it happen.
    //
    void transformFuncUsingActiveMask(IRFunc* func);
};

// The module-level pass isn't too complicated, and its main job is
// to reduce the scope of the problem down to individual functions.
// As a result the function-level pass, which has to deal with much
// more complicated issues, at least doesn't have to deal with
// distractions related to the whole-module processing.
//
struct SynthesizeActiveMaskForFunctionContext
{
    // The funciton-level pass unsurprisngly operates on one IR function.
    //
    IRFunc* m_func;

    IRModule* m_module;
    IRType* m_maskType;

    void transformFunc()
    {
        // The first thing we will do to our function is preprocess it
        // so that it satisfies several properties that will make our
        // pass easier to implement correctly.
        //
        // Note that given a CFG edge P->S from a predecessor P to
        // a successor S, we can always rewrite the CFG by removing
        // the edge P->S, creating a new block E, and then inserting
        // edges P->E and E->S. That transformation is referred to
        // as "breaking" the edge P->S, and it always preserves the
        // semantics of the original program.
        //
        // We will start out by breaking several kinds of edges in
        // the CFG of the function. Specifically, we break:
        //
        // * *Critical* edges which are those from a block P with
        //  multiple successors to a block S with multiple predecessors.
        //
        // * What we are calling *pseudo-critical* edges, which are
        //   those from a block P with multiple successors to a block
        //   M that is used as the merge point of a structured control-flow
        //   construct.
        //
        // The benefit of these transformations is the invariants they
        // provide us for the rest of this pass:
        //
        // * When looking at a conditional branch instruction, we can assume:
        //
        //   * The block with the branch is the only predecessor of the target blocks
        //
        //   * The block with the branch is the immediate dominator of its target blocks
        //
        //   * None of the target blocks is the merge point for the conditional branch
        //
        // * As a corallary of the above, all branches to a merge block are uncondtiional branches
        //
        // * Any block with multiple predecessors is only targetted by unconditional branches
        //
        // We will hightlight our use of these assumptions in the code
        // that takes advantage of them.
        //
        breakCriticalAndPseudoCriticalEdges();

        // Given that we are processing a function that had a `waveGetActiveMask` instruction
        // in its body, we expect to find an entry block on the function.
        //
        auto funcEntryBlock = m_func->getFirstBlock();
        SLANG_ASSERT(funcEntryBlock);

        // We don't want to inject logic for computing and maintaining the
        // active mask into parts of a function that don't actually use it,
        // so one of the first things we do (once we've finished messing with
        // the CFG of the function) is to mark which blocks (transitively) need
        // the active mask, and which don't.
        //
        markBlocksNeedingActiveMask();
        //
        // We expect to find that the entry block to the function needs the
        // active mask, or else that would imply nothing else in the function
        // did, and we shouldn't be processing this function at all.
        //
        SLANG_ASSERT(m_blocksNeedingActiveMask.contains(funcEntryBlock));

        // Our basic approach will be to associate an `IRInst*` that represents
        // the active mask value to use with each basic block of the function.
        //
        // The active mask to use for the entry block of a function is a special
        // case, since it can't be derived from the value in other blocks.
        //
        createInitialActiveMask(funcEntryBlock);

        // For blocks wtih multiple predecessors, it is possible that the correct
        // active mask to use will depend on the predecessor.
        //
        // We will thus add an extra parameter (a phi node in more traditional
        // SSA representations) to any block with multiple distinct predecessors,
        // that can be used to represent the incoming active mask.
        //
        addActiveMaskParametersToBlocksWithMultiplePredecessors();

        // The remaining work of assigning active masks to blocks will
        // be handled by a recursive traversal of the function's CFG
        // in terms of control-flow regions, where the entry block
        // for the function serves as the starting point of the outer-most region.
        //
        transformRegions(funcEntryBlock);
    }

    void breakCriticalAndPseudoCriticalEdges()
    {
        // In order to break all the critical pseudo-critical
        // edges, we will first identify them and build a list
        // of `IREdge`s, and then go along and break them all.
        //
        // This approach ensures things don't get tripped up
        // by modifying the CFG while also walking its structure.
        //
        List<IREdge> edgesToBreak;
        //
        // We are going to consider each block in turn as a
        // candidate predecessor block, and look at its
        // outgoing edges.
        //
        for (auto pred = m_func->getFirstBlock(); pred; pred = pred->getNextBlock())
        {
            auto successors = pred->getSuccessors();
            auto succIter = successors.begin();
            auto succEnd = successors.end();
            for (; succIter != succEnd; ++succIter)
            {
                auto edge = succIter.getEdge();

                // We want to collect the edges that are critical
                // or psuedo-critical.
                //
                if (edge.isCritical() || isPseudoCriticalEdge(edge))
                {
                    edgesToBreak.add(edge);
                }
            }
        }

        // Once we've identified the edges we want to break,
        // we do so by inserting an edge along them.
        //
        for (auto& edge : edgesToBreak)
        {
            IRBuilder::insertBlockAlongEdge(m_module, edge);
        }
    }

    bool isPseudoCriticalEdge(IREdge const& edge)
    {
        // Our definition of a pseudo-critical edge is one
        // where the prececessors is a conditional branch
        // (meaning it has multiple outgoing edges), and
        // where the successor is a merge point for some
        // structured control-flow construct.
        //
        auto pred = edge.getPredecessor();
        auto succ = edge.getSuccessor();

        // The condition on the predecessor is easy
        // enough to check.
        //
        if (pred->getSuccessors().getCount() <= 1)
            return false;

        // To check if the successor is used as a merge
        // point, we must iterate over its uses.
        //
        for (auto use = succ->firstUse; use; use = use->nextUse)
        {
            // For each use, we need to consider if
            // the user represents a structured control-flow
            // construct, and then whether the `succ` block
            // is being used as a merge point in that construct.
            //
            auto user = use->getUser();
            if (auto ifElseInst = as<IRIfElse>(user))
            {
                // The merge point for an `if` or `if`/`else`
                // if the block *after* the statement.
                //
                if (ifElseInst->getAfterBlock() == succ)
                    return true;
            }
            else if (auto switchInst = as<IRSwitch>(user))
            {
                // The merge point for a `switch` is the block
                // after the statement, which is also the label
                // where control flow goes on a `break`.
                //
                if (switchInst->getBreakLabel() == succ)
                    return true;
            }
            else if (auto loopInst = as<IRLoop>(user))
            {
                // A loop construct can actually have two
                // merge points.
                //
                // First, the merge point for the entire loop
                // is the block after the loop, which is also
                // where control flow goes on a `break`.
                //
                if (loopInst->getBreakBlock() == succ)
                    return true;
                //
                // Second, for a given iteration of the loop,
                // the "continue clause" of a `for` loop
                // is a merge point, and is where control flow
                // goes on a `continue`.
                //
                if (loopInst->getContinueBlock() == succ)
                    return true;
            }
        }

        return false;
    }

    // We want to identify which blocks need the active mask,
    // and will use a set to keep track of them.
    //
    HashSet<IRBlock*> m_blocksNeedingActiveMask;

    // To make things convenient for downstream code, we will
    // define a simple accessor to check if a block needs the
    // active mask, as determined by our setup logic.
    //
    bool doesBlockNeedActiveMask(IRBlock* block)
    {
        return m_blocksNeedingActiveMask.contains(block);
    }

    void markBlocksNeedingActiveMask()
    {
        // We can start by identifying the blocks that directly
        // use the active because they contain a `waveGetActiveMask()`
        // instruciton.
        //
        // Note: the module-level pass will have modified any
        // function that indirectly uses the active mask (that is,
        // a function that calls another functio nneeding the mask)
        // so that it uses `waveGetActiveMask` at those call sites,
        // so there whould always be at least one block that gets
        // discovered this way, even in functions that did not
        // initially contain direct uses of `waveGetActiveMask`.
        //
        for (auto block : m_func->getBlocks())
        {
            for (auto inst : block->getOrdinaryInsts())
            {
                if (inst->getOp() == kIROp_WaveGetActiveMask)
                {
                    m_blocksNeedingActiveMask.add(block);
                    break;
                }
            }
        }

        // Once we've identified an initial set of blocks that need/use
        // the active mask, we want to mark any blocks that lead to
        // blocks we've already marked, since they will need to pass
        // along the active mask.
        //
        // We will do this in an iterative fashion, by looping over
        // all the blocks in the CFG and checking if they branch to
        // a block that needs the active mask.
        //
        {
            // Once we make a sweep over the CFG and don't see any change
            // in the decision about what blocks use the active mask, we
            // know we have converged.
            //
            bool change = true;
            while (change)
            {
                change = false;

                // Because we are solving a backwards dataflow problem,
                // we iterate over the blocks of the function in reverse
                // order to minimize the number of iterations required
                // in the common case.
                //
                for (auto block = m_func->getLastBlock(); block; block = block->getPrevBlock())
                {
                    if (m_blocksNeedingActiveMask.contains(block))
                        continue;

                    for (auto successor : block->getSuccessors())
                    {
                        if (!m_blocksNeedingActiveMask.contains(successor))
                            continue;

                        // If we get here then `block` has *not* been marked
                        // as needing the active mask, but it branches to
                        // `successor` which *has* been marked, so we need
                        // to mark `block` and keep looking for changes.
                        //
                        m_blocksNeedingActiveMask.add(block);
                        change = true;
                        break;
                    }
                }
            }
        }
    }

    // As described previously, our main goal in this pass is to associated an
    // `IRInst*` representing the active mask with each block in the CFG
    // of the function (or rather, with each block that *needs* the active mask).
    //
    // We will keep track of the active mask values that have been registered
    // so far in a dictionary.
    //
    Dictionary<IRBlock*, IRInst*> m_activeMaskForBlock;

    void createInitialActiveMask(IRBlock* funcEntryBlock)
    {
        // The active mask value to use on entry to a function depends
        // on whether the function is an entry point or not.
        //
        // The easy case is ordinary functions (ones that aren't entry
        // points).
        //
        if (!m_func->findDecoration<IREntryPointDecoration>() &&
            !m_func->findDecoration<IRCudaKernelDecoration>())
        {
            // We simplyu need to add a new parameter to the entry block
            // (which holds the parameters for the function itself).
            // That parameter will receive the initial of the active
            // mask as an argument at (modified) call sites to the function.
            //
            addActiveMaskParameter(funcEntryBlock);
        }
        else
        {
            // The case of a shader entry point is a bit trickier.
            //
            // We can't change the signature of a shader entry point (e.g., adding
            // new varying parameters), so we need to compute the initial active
            // mask value from first principles.
            //
            // We will insert the code we generate at the start of the entry block.
            //
            IRBuilder builder(m_module);
            builder.setInsertBefore(funcEntryBlock->getFirstOrdinaryInst());
            //
            // A naive approach would be to set the active mask to an all-ones
            // value representing the idea that all threads/lanes in the warp/wave
            // will be active at the start of execution.
            //
            auto allLanesMask = builder.getIntValue(m_maskType, -1);
            //
            // That all-ones mask won't actually be correct in any case where
            // a warp/wave is less than fully occupied (e.g., what if the thread
            // group size is smaller than the warp/wave size?).
            //
            // We can refine that all-ones mask down to a more accurate one by
            // using a ballot operation:
            //
            //      initialActiveMask = waveMaskBallot(allOnesMask, true);
            //
            auto initialActiveMask =
                builder.emitWaveMaskBallot(m_maskType, allLanesMask, builder.getBoolValue(true));
            //
            // Note: an important detail here is that we are invoking `waveMaskBallot`
            // with the all-ones mask and *assuming* that a target will properly ignore
            // the bits in `allOnesMask` that correspond to threads/lanes that aren't
            // actually executing. Fortunately CUDA at least guarantees this
            // behavior for `__ballot_sync()` and its other collective operations:
            // bits corresponding to threads that aren't executing are ignored.
            //
            // Once we've computed the mask value to start with, we add it to
            // our tracking structure so we can remember which value to use.
            //
            m_activeMaskForBlock.add(funcEntryBlock, initialActiveMask);
        }
    }

    void addActiveMaskParametersToBlocksWithMultiplePredecessors()
    {
        // Blocks with multiple (distinct) predecessors will
        // recieve their input active mask as a block parameter
        // (an SSA phi operation in more traditional representations).
        //
        for (auto block : m_func->getBlocks())
        {
            // We only care about blocks that want the active mask,
            // and that have multiple predecessors.
            //
            if (!doesBlockNeedActiveMask(block))
                continue;
            if (block->getPredecessors().getCount() <= 1)
                continue;

            // What is more, we only care about blocks with multiple
            // *distinct* predecessors, which means we need to look
            // out for things like a `switch` where multiple outgoing
            // edges from a single block could lead to the same
            // destination.
            //
            // A block with multiple distinct predecessors is one
            // where the predecessors aren't all the same.
            //
            bool allPredsTheSame = true;
            //
            // We will check if all the predecessors of `block`
            // are the same as the first predecessor (which we
            // know must exist because we dissmissed blocks
            // with <= 1 predecessor already).
            //
            IRBlock* firstPred = *block->getPredecessors().begin();
            for (auto pred : block->getPredecessors())
            {
                if (pred != firstPred)
                {
                    allPredsTheSame = false;
                    break;
                }
            }
            if (allPredsTheSame)
                continue;

            // If we have identified a block with multiple distinct
            // predecessors, then we know that it needs a new
            // block parameter to be added to represent the active mask.
            //
            addActiveMaskParameter(block);
        }
    }

    void addActiveMaskParameter(IRBlock* block)
    {
        // Adding a paramter to a block to represent the
        // active mask is a straightforward application
        // of `IRBuilder`.
        //
        // As a small sanity check, we make sure that the
        // block we are modifying actually wants the active
        // mask.
        //
        SLANG_ASSERT(doesBlockNeedActiveMask(block));

        IRBuilder builder(m_module);
        builder.setInsertBefore(block->getFirstOrdinaryInst());

        auto activeMaskParam = builder.emitParam(m_maskType);

        m_activeMaskForBlock.add(block, activeMaskParam);
    }

    // The remainder of the work in this pass is going to be based
    // on a recursive walk of the CFG for the function in terms of
    // regions.
    //
    // Our definition of regions will rely on having built a dominator
    // tree for the function.
    //
    RefPtr<IRDominatorTree> m_dominatorTree;
    //
    // Briefly, a node (basic block) A dominates block B if every
    // possible path through the graph that ends in B goes through A.
    //
    // The dominator tree encodes dominance relationships by making
    // it so that A is an ancestor node of B in the tree if and only if
    // A dominates B. Furthermore, if A is the direct parent of B in
    // the dominator tree, we say that A *immediately dominates* B.
    //
    // We will use a few wrapper/helper functions to make working
    // with the dominator tree more convenient.
    //
    // First is a query to check if one block dominates another:
    //
    bool dominates(IRBlock* dominator, IRBlock* dominated)
    {
        return m_dominatorTree->dominates(dominator, dominated);
    }
    //
    // Next is an operation to check if one block domiantes
    // another *or* the child region is unreachable in the CFG.
    //
    // TODO: The `IRDominatorTree` type should now make this
    // operation redundant, since it will treat an unreachable
    // block as being domianted by any other block.
    //
    bool dominatesOrIsUnreachable(IRBlock* dominator, IRBlock* dominated)
    {
        return m_dominatorTree->isUnreachable(dominated) || dominates(dominator, dominated);
    }

    // With the definition of dominance in hand, we are now ready
    // to define the notion of regions that we will use.
    //
    // There are many ways to define regions on a CFG, and we are
    // specifically interested in single-entry, multiple-exit
    // regions.
    //
    struct RegionInfo
    {
        // Each region will have a dedicated entry block, which
        // is the only way into the region (it is single-entry,
        // after all). This means that the entry block must
        // dominate everything inside the region.
        //
        IRBlock* entryBlock = nullptr;

        // Each region will also have a dedicated *merge* block,
        // which is where control flow goes when exiting the
        // region normally. Code after the merge block is considered
        // to be outside the region.
        //
        IRBlock* mergeBlock = nullptr;

        // Regions in the CFG will nest with a parent-child relationship,
        // and during our processing it will be important to track
        // the chain of ancestor regions that enclosue a piece of code.
        //
        // Note: it is possible that `parentRegion` will have the same
        // `mergeBlock` as this region, and code that handles regions
        // needs to deal with that case.
        //
        RegionInfo* parentRegion = nullptr;

        // As a convenience, we also make our region type hold
        // the active mask for the entry block of the region.
        // This could technically be accessed using `m_activeMaskForBlock`,
        // but storing it here can avoid some extra lookups and
        // error checks around them.
        //
        IRInst* activeMaskOnEntry = nullptr;
    };

    // Given the definition of a region, one of the most important
    // queries to be able to answer is: given a block B and a region R,
    // is B inside R?
    //
    bool isBlockInRegion(IRBlock* block, RegionInfo* region)
    {
        // By our definition, a region only contains blocks
        // that are dominated by its entry block.
        //
        // Thus, any block not dominated by the entry block
        // is outside the region.
        //
        if (!m_dominatorTree->dominates(region->entryBlock, block))
            return false;

        // In addition, if `block` is dominated by the merge
        // block of `region` then it can only be reached by
        // going through the merge block, which implies leaving
        // the region.
        //
        // Thus any block that is dominated by the merge block
        // of `region` (or any of its parents) is not in the region.
        //
        // TODO: It may not be necessary to check the mrege block
        // of parent regions, so that checking just the one merge
        // block would suffice. We need to double-check that
        // nothing would be changed before removing that logic...
        //
        for (auto r = region; r; r = r->parentRegion)
        {
            // Note: It is possible for the merge block of a region to
            // be null (specifically for the entry block of a function,
            // where the logical merge point is after the function
            // returns).
            //
            if (r->mergeBlock && dominates(r->mergeBlock, block))
                return false;
        }

        return true;
    }

    void transformRegions(IRBlock* funcEntryBlock)
    {
        // Given the definition of regiosn we will use,
        // we can process the entire function by recursively
        // carving it up into regions.
        //
        // We start by computing a dominator tree for
        // the function, since that will help us
        // identify the regions.
        //
        m_dominatorTree = computeDominatorTree(m_func);

        // Next we look up th active mask for the function's
        // entry region, which had better be set before
        // we run this code.
        //
        IRInst* activeMaskOnFuncEntry = nullptr;
        m_activeMaskForBlock.tryGetValue(funcEntryBlock, activeMaskOnFuncEntry);
        SLANG_ASSERT(activeMaskOnFuncEntry);

        // The root region of our tree of regions will
        // start at the entry block of the function, and
        // its merge block will be left as null (because
        // we merge upon return from the function, which isn't
        // represented as an explicit block in the CFG).
        //
        RegionInfo rootRegion;
        rootRegion.entryBlock = funcEntryBlock;
        rootRegion.activeMaskOnEntry = activeMaskOnFuncEntry;

        // We then kick off transformation of the whole
        // CFG by transforming the root region.
        //
        transformRegion(&rootRegion);
    }

    void transformRegion(RegionInfo* region)
    {
        // The task of transforming a region with a given
        // entry block and initial active mask comprises
        // two steps.
        //
        IRBlock* regionEntry = region->entryBlock;
        IRInst* activeMaskOnRegionEntry = region->activeMaskOnEntry;
        //
        // The first step is the easy one: any uses of
        // the `waveGetActiveMask` instruction in the entry
        // block of the region can be replaced with the value
        // of `actvieMaskOnRegionEntry`.
        //
        // The only wrinkle here is carefully fetching the
        // next instruction before processing each instruction
        // so that we can continue to iterative while also
        // potentially removing and replacing instructions along
        // the way.
        //
        IRInst* nextInst = nullptr;
        for (IRInst* inst = regionEntry->getFirstInst(); inst; inst = nextInst)
        {
            nextInst = inst->getNextInst();

            if (inst->getOp() == kIROp_WaveGetActiveMask)
            {
                inst->replaceUsesWith(activeMaskOnRegionEntry);
                inst->removeAndDeallocate();
            }
        }

        // The second and much more involved step is to process all
        // the other blocks in the region, recursively.
        //
        // The way to proceed will be determined by the terminator
        // instruction of the entry block.
        //
        auto terminator = regionEntry->getTerminator();
        SLANG_ASSERT(terminator);
        switch (terminator->getOp())
        {
            // There are some cases of terminator instructions that
            // we explicit do not or cannot handle.
            //
            // A `conditionalBranch` instruction represents a two-way
            // conditional branch *without* structured control-flow
            // information. Right now our front-end doesn't produce
            // such instructions.
            //
            // TODO: We could theoretically handle unstructured control
            // flow either by running a restructuring pass, or by saying
            // that unstructured branches can split the active mask,
            // without any option to reconverge it.
            //
        case kIROp_conditionalBranch:
            //
            // Finally, we also don't handle any control-flow op we might not
            // have introduced at the time this pass was created.
            //
        default:
            SLANG_UNEXPECTED("unhandled terminator op");
            break;

            // Next, there are cases of terminators that represent code that
            // is or must be be unreachable. The runtime semantics of executing
            // one of these instructions would be some kind of undefined behavior,
            // so we can elect to simply do nothing.
            //
        case kIROp_MissingReturn:
        case kIROp_Unreachable:
            break;

        case kIROp_unconditionalBranch:
            {
                auto branch = cast<IRUnconditionalBranch>(terminator);

                // An `unconditionalBranch` instruction represents any kind of
                // unconditonal branch in the code. This includes cases like:
                //
                // * Leaving a control-flow structure normally (e.g., ending the "then"
                //   block of an `if` statement and moving on the code after the `if`;
                //   jumping out of a loop with a `break`)
                //
                // * Leaving a control-flow structure abnormally (e.g., leaving the "then"
                //   part of an `if` statement by `break`ing out of an outer loop).
                //
                // * Ordinary control flow that doesn't leave any region(s) (e.g., cycling
                //   back to the top of a `while` loop; progressing through straight-line
                //   non-branching control flow that happens to use multiple blocks)
                //
                // We factor most of the handling of unconditional branches out into
                // a subroutine, so we just invoke it here and leave the explanation
                // to later.
                //
                transformUnconditionalEdge(
                    region,
                    terminator,
                    branch->getTargetBlock(),
                    activeMaskOnRegionEntry);

                // Once we've handled the control-flow edge at the end of the entry
                // block of our region, we need to process any child regions that
                // might exist.
                //
                // E.g. if we had straight-line control flow from block A->B->C->... and
                // we were processing the region that starts with A, then the previous
                // code would have handled all work related to the A->B edge, and then
                // this step would handle recursively processing the B->C->... region.
                //
                transformChildRegions(region, region);
            }
            break;

        case kIROp_Return:
            {
                // A `return` instruction is akin to an unconditional branch,
                // except that it is guaranteed to exit any structured control
                // flow regions that we are nested under.
                //
                // We thus handle a `return` as a special case of an unconditional
                // branch where the target block is null (which also happens to
                // be the value used to represent the merge block for the region
                // representing the entire function body).
                //
                transformUnconditionalEdge(region, terminator, nullptr, activeMaskOnRegionEntry);

                // We also make a call to recusrively process any child regions here,
                // although in practice there should be no child regions if the
                // entry block ended with a `return`.
                //
                // TODO: Consider eliminating this call if it really isn't needed.
                //
                transformChildRegions(region, region);
            }
            break;

        case kIROp_ifElse:
            {
                // A structured `ifElse` instruction is a two-way branch on a
                // Boolean coniditon, along with a specific block representing
                // the code after the high-level-language `if` statement.
                //
                auto ifElse = cast<IRIfElse>(terminator);
                auto condition = ifElse->getCondition();
                auto trueBlock = ifElse->getTrueBlock();
                auto falseBlock = ifElse->getFalseBlock();
                auto afterBlock = ifElse->getAfterBlock();
                //
                // We can picture the situation as something like:
                //
                //          I       // block with if/else           |
                //         / \                                      |
                //        T   F     // true and false blocks        |
                //       /|\ /|\                                    |
                //                                                  |
                //         ...                                      |
                //                                                  |
                //         \|/                                      |
                //          A       // "after" block                |
                //                                                  |
                //         ...                                      |
                //
                // Note that very few assumptions can or should be
                // made about the code under T and F. In particular:
                //
                // * Code under T or F could exit the region under
                //   consideration without going through A
                //
                // * There could exist some block X such that X is in
                //   the region of our if/else, and control flow can reach
                //   A from both T and F *without* going through A
                //
                // Our construction doesn't rely on many assumptions.
                // All we need is that we can construct a sub-region
                // representing the code dominated by I but not by A
                // (which serves as the merge point).
                //
                RegionInfo ifElseRegion;
                ifElseRegion.entryBlock = regionEntry;
                ifElseRegion.activeMaskOnEntry = activeMaskOnRegionEntry;
                ifElseRegion.mergeBlock = afterBlock;
                ifElseRegion.parentRegion = region;
                //
                // This sub-region will be used as a parent region
                // when recursively processing code reachable through
                // the `ifElse`.

                // Because we broke all critical edges as a pre-process, we
                // can be certain that the entry block for `region` dominates
                // the blocks for the `true` and `false` cases (it should
                // be their only predecessor, and thus the immediate
                // dominator).
                //
                SLANG_ASSERT(m_dominatorTree->dominates(regionEntry, trueBlock));
                SLANG_ASSERT(m_dominatorTree->dominates(regionEntry, falseBlock));

                // Because we also broke all pseudo-critical edges, we also
                // expect taht the blocks we branch to on `true` or `false`
                // are not the same as the merge block.
                //
                SLANG_ASSERT(trueBlock != afterBlock);
                SLANG_ASSERT(falseBlock != afterBlock);

                // Because of the above assumptions, we know that this
                // code can take responsibility for computing the mask
                // value to use for both `trueBlock` and `falseBlock`.
                //
                IRBuilder builder(m_module);

                // To establish the mask value for `trueBlock` we will
                // insert a `waveMaskBallot` before the branch:
                //
                //      trueMask = waveMaskBallot(activeMaskOnRegionEntry, condition);
                //
                // That mask weill consist of all threads that entered the
                // parent region and computed a `condition` value of `true`.
                //
                builder.setInsertBefore(ifElse);
                auto trueMask =
                    builder.emitWaveMaskBallot(m_maskType, activeMaskOnRegionEntry, condition);

                // To establish the mask value for `falseBlock`, we will
                // insert code into the false block that computes an inverted mask as:
                //
                //      falseMask = activeMaskOnRegionEntry & ~trueMask;
                //
                builder.setInsertBefore(falseBlock->getFirstOrdinaryInst());
                auto falseMask = builder.emitBitAnd(
                    m_maskType,
                    activeMaskOnRegionEntry,
                    builder.emitBitNot(m_maskType, trueMask));

                // The task of associating the mask value comptued by a conditional branch
                // with the successor block(s) is delegated to a subroutine, which we will
                // detail later.
                //
                // For now it suffices that we need to transform each of the outgoing
                // edges from our conditional branch.
                //
                // TODO: There is a pathological corner case that is not handled here,
                // when `trueBlock == falseBlock`. Right now we have no optimizations
                // that could make this case arise, but we should be careful about
                // it if we ever add such passes.
                //
                transformConditionalEdge(&ifElseRegion, terminator, trueBlock, trueMask);
                transformConditionalEdge(&ifElseRegion, terminator, falseBlock, falseMask);

                // Next we need to transform any and all child regions of our `if`-`else`
                // construct as well as the children of the overall `region`.
                //
                // Those children chould (at least) include child regions where `trueBlock`
                // and `falseBlock` are the entry blocks.
                //
                transformChildRegions(&ifElseRegion, region);
            }
            break;

        case kIROp_loop:
            {
                // At the most basic level, a `loop` instruction is just an uncondtional
                // branch. What is stores above and beyond an `unconditionalBranch`
                // instruction is the strucrutal information about the loop, including
                // where the code "after" the loop starts (the same as the `break`
                // target), and where the code that starts a new loop iteration goes
                // (the `continue` target).
                //
                auto loopInst = cast<IRLoop>(terminator);
                auto loopHeader = loopInst->getTargetBlock();
                auto breakBlock = loopInst->getBreakBlock();
                auto continueBlock = loopInst->getContinueBlock();
                //
                // We can visualize it as something like this:
                //
                //          L           // block with `loop` instruction    |
                //          |                                               |
                //          |  ___      // back edge(s) that start a        |
                //          |/    \     //     new iteration                |
                //          H     |     // loop header, where control flow  |
                //         /|\    |     //     starts on each iteration     |
                //         ...                                              |
                //                                                          |
                //               \|/                                        |
                //                C     // continue label, which leads      |
                //               ...    //     to any/all back edges        |
                //                                                          |
                //         \|/                                              |
                //          B           // break label, which is the start  |
                //         ...          //     of code after the loop       |
                //
                // Much like the case for `ifElse`, we can't make a lot
                // of definition statements about the structure of the control
                // flow after the `loop` instruction, so we need to be careful
                // and construct regions that don't rely on unsafe assumptions.
                //
                // We can construct a region that represents all the code that
                // is logically inside the loop. This regions starts with the
                // loop header (*not* the block with the `loop` instruction),
                // and ends at the block B where a `break` goes (to exit
                // the loop).
                //
                RegionInfo loopRegion;
                loopRegion.entryBlock = loopHeader;
                loopRegion.activeMaskOnEntry = activeMaskOnRegionEntry;
                loopRegion.mergeBlock = breakBlock;
                loopRegion.parentRegion = region;

                // In order for our structured control-flow information to
                // mean anything, the `loop` instruction must dominate
                // the loop body (starting with the header).
                //
                SLANG_ASSERT(m_dominatorTree->dominates(regionEntry, loopHeader));
                //
                // We also require that the region with the `loop` either
                // dominates the `break` block (you can't exit the loop without
                // first enterting it), *or* that block is unreachable (the
                // loop never exits normally).
                //
                SLANG_ASSERT(dominatesOrIsUnreachable(regionEntry, breakBlock));
                //
                // Simlarly, we assume that the `continue` label is also either
                // dominated by the `loop` instruction or is unreachable (the
                // loop never actually iterates).
                //
                SLANG_ASSERT(dominatesOrIsUnreachable(regionEntry, continueBlock));
                //
                // Finally, our pre-pass on the CFG will have ruled out very
                // silly cases like a `loop` that branches directly to the `break`
                // label.
                //
                SLANG_ASSERT(loopHeader != breakBlock);

                // Each iteration of the loop will execute under an active
                // mask, and the mask may vary per iteration.
                //
                IRInst* iterEntryMask = nullptr;
                //
                // For a loop that actually *loops*, the loop header block
                // must have multiple distinct predecessors: one for the
                // `loop` instruction and one or more for back edges that
                // continue execution of the loop.
                //
                if (loopHeader->getPredecessors().getCount() > 1)
                {
                    iterEntryMask = loopHeader->getLastParam();
                    SLANG_ASSERT(iterEntryMask);
                }
                else
                {
                    // It technically possible that a `loop` instruction
                    // doesn't actually yield a loop. For example, one
                    // can write:
                    //
                    //      for(;;)
                    //      {
                    //          doThings();
                    //          if(someCondition) break;
                    //          doOtherThings();
                    //          break;
                    //      }
                    //
                    // There are no cases where this code actually loops,
                    // but the use of a `for` loop is still enabling some
                    // non-trivial control flow with the early `break`.
                    //
                    // We cannot in general eliminate all `loop`s that don't
                    // actually loop as a pre-process, without adding some
                    // alternative kind of control-flow construct that
                    // represents this kind of alternative use of looping
                    // constructs.
                    //
                    // Fortunately, handling this case isn't complicated,
                    // because the mask to use on entry to the (only)
                    // iteration is well known.
                    //
                    iterEntryMask = activeMaskOnRegionEntry;
                }

                // Along with the outer region that represents the
                // entire loop (which is exited with `break`) there
                // is conceptually a nested inner region that represents
                // the body of a single loop iteration, and which is
                // exited with a `continue`.
                //
                // This detail primarily matters for a `for` loop where
                // there can be non-trivial code (including code that
                // depends on the active mask) that executes on a
                // `continue`.
                //
                // The nested region thus begins at the loop header,
                // and ends at the `continue` label. Note that it
                // is possible for this region to be empty, in the
                // case where `loopHeader == continueBlock` (which
                // will happen for any non-`for` loop).
                //
                RegionInfo iterRegion;
                iterRegion.entryBlock = loopHeader;
                iterRegion.activeMaskOnEntry = iterEntryMask;
                iterRegion.mergeBlock = continueBlock;
                iterRegion.parentRegion = &loopRegion;

                // Once we've computed the child regions that are introduced
                // by the loop structure, we can move on to processing
                // its single outgoing edge, and the child control flow
                // regions.
                //
                transformUnconditionalEdge(region, terminator, loopHeader, activeMaskOnRegionEntry);
                transformChildRegions(&iterRegion, region);
            }
            break;


        case kIROp_Switch:
            {
                // A `switch` instruction represents a structured N-way
                // branch on an integer condition. The structural information
                // gives us a block that represents code after the `switch`
                // statement, and which is also the target for any `break`s
                // inside the `switch`.
                //
                auto switchInst = cast<IRSwitch>(terminator);
                auto condition = switchInst->getCondition();
                auto mergeBlock = switchInst->getBreakLabel();

                // A `switch` is mostly just a more involved and complicated
                // version of an `if`-`else`, so the overall logical is similar
                // but with a lot more details to handle.
                //
                // We can construct a child region that represents the code
                // that is logically inside the `switch`, consisting of all
                // code from teh entry block up to the merge (`break`) block.
                //
                RegionInfo switchRegion;
                switchRegion.entryBlock = regionEntry;
                switchRegion.activeMaskOnEntry = activeMaskOnRegionEntry;
                switchRegion.mergeBlock = mergeBlock;
                switchRegion.parentRegion = region;

                // Next, we need to establish a mask value that will
                // represent the active mask on entry to a given `case`.
                //
                IRBuilder builder(m_module);
                builder.setInsertBefore(switchInst);

                // For now we are computing a simple-but-inaccurate version
                // of the active mask. We are using:
                //
                //      matchingMask = waveMaskMatch(activeMaskOnRegionEntry, condition);
                //
                // This value will yield a mask for each thread that consists of
                // the threads who had exactly the same value for `condition`.
                //
                auto matchingMask =
                    builder.emitWaveMaskMatch(m_maskType, activeMaskOnRegionEntry, condition);
                //
                // TODO: this mask computation yields a surprising value in cases where
                // multiple values go to the same code:
                //
                //      switch(condition)
                //      {
                //      case 0: case 1:  A(WaveGetActiveMask()); break;
                //
                //      case 2: default: B(WaveGetActiveMask()); break;
                //      }
                //
                // In this case we want the mask seen by `A()` to be all the threads
                // where `condition` was `0` *or* `1` (and not to see two
                // different masks, one for each value).
                //
                // Similarly, if we have threads where `condition` is `2`, `3`, and `4`,
                // we expec them to all see a common mask at `B()`, despite
                // representing multiple distinct values.
                //
                // The desired mask could be synthesized in different ways.
                //
                // We could execute an extra `switch` before this one, to reduce
                // the actual values of `condition` down to an artificial `caseNumber`,
                // which only represents the unique code blocks being branched to:
                //
                //      int caseNumber;
                //      switch(condition)
                //      {
                //      case 0: case 1:  caseNumber = 0xA; break;
                //
                //      case 2: default: caseNumber = 0xB; break;
                //      }
                //      switch( caseNumber ) // NOTE: changed condition here
                //      {
                //      case 0xA: A(WaveGetActiveMask()); break;
                //
                //      case 0xB: B(WaveGetActiveMask()); break;
                //      }
                //
                // The main downside of that approach is that it adds additional
                // control flow that wasn't in the input program (and that new
                // control flow could complicate our interactions with the
                // dominator tree and CFG edges).
                //
                // Another alternatie would be to move the synthesis of the mask
                // down into the individual cases:
                //
                //      switch(condition)
                //      {
                //      case 0: case 1:
                //          mask = WaveMaskMatch(activeMaskOnRegionEntry, 0xA);
                //          A(mask);
                //          break;
                //
                //      case 2: default:
                //          mask = WaveMaskMatch(activeMaskOnRegionEntry, 0xB);
                //          B(mask);
                //          break;
                //      }
                //
                // This second approach avoids adding additional control flow
                // operations before the `switch`, but does so at the cost of
                // having distinct textual call sites for a collective that
                // need to work together.
                //
                // We need to pick and implement one of these strategies
                // (or something else entirely) in a future change.

                // A `switch` instruction always has a `default` label, which
                // is where control flow goes for values of the condition
                // that weren't explicitly handled.
                //
                // Note: even if the high-level-language `switch` didn't have
                // a `default` label, the IR one will. During initial IR generation,
                // the `default` label for such a `switch` will be the same as
                // the `break` label, but our pass to break pseudo-critical edges
                // will guarantee that every `switch` has a `default` label distinct
                // from its `break` label.
                //
                // We will handle the conditional edge leading the `default` label
                // before looking at any of the explicit `case`s.
                //
                auto defaultLabel = switchInst->getDefaultLabel();
                transformConditionalEdge(&switchRegion, terminator, defaultLabel, matchingMask);

                // One wrinkle that arises when processing a `switch` statement is
                // that multiple `case`s may branch to same target label, and the
                // label for `case`s can be the same as the `default` label.
                //
                // Our approach in `transformConditionalEdge` currently assumes that
                // for each block reached via a conditional edge, the `transformConditionalEdge`
                // function is only called once.
                //
                // To avoid violating this assumption, we will make sure to only
                // call `transformConditionalEdge` once for each unique target block
                // of the `switch`.
                //
                // In order to make that guarantee, we rely on the invariant (provided
                // by our IR generation, maintained by our IR passes, and currently already
                // assumed by our emit logic) that the `case`s of a `switch` are stored
                // in an order such that:
                //
                // * All cases that branch to the same label are contiguous in the list of cases
                //
                // * If the cases with label A can fall through to the cases with label B, then
                //   the A cases immediately precede the B cases in the list of cases.
                //
                // We can thus simply track the label of the last case we processed,
                // and detect duplicates by comparing to it (and the `default` label).
                //
                IRBlock* prevLabel = nullptr;
                UInt caseCount = switchInst->getCaseCount();
                for (UInt ii = 0; ii < caseCount; ++ii)
                {
                    auto caseLabel = switchInst->getCaseLabel(ii);

                    // As discussed above, we only process the first `case` with
                    // a given label, since all of the edges will pass  along the
                    // same mask.
                    //
                    if (caseLabel == prevLabel)
                        continue;
                    prevLabel = caseLabel;

                    // Similarly, we skip `case`s that branch to the same label
                    // as a `default`, since the `default` label was already
                    // processed above.
                    //
                    if (caseLabel == defaultLabel)
                        continue;

                    transformConditionalEdge(&switchRegion, terminator, caseLabel, matchingMask);

                    // TODO: One issue that is getting ignored in the current
                    // code is "non-trivial fall-through," where one `case`
                    // executes some amount of code before falling through to
                    // another:
                    //
                    //      switch(x)
                    //      {
                    //      case 0:
                    //          A(WaveGetActiveMask());
                    //      case 1:
                    //          B(WaveGetActiveMask());
                    //          break;
                    //      ...
                    //      }
                    //
                    // In the scenario above it is clear what the desired
                    // active mask is for the call to `A()` (the set of lanes
                    // that had `x==0`), but the active mask when calling
                    // `B()` is a bit thornier question.
                    //
                    // The easiest implementation choice is to just let
                    // `B()` see two active masks: one for the `x==0` lanes,
                    // and another for the `x==1` lanes, and it could be
                    // argued that this matches the programmer's view of
                    // how things diverged at the `switch`.
                    //
                    // The alternative view is that the lanes with `x==0`
                    // should re-converge with the lanes with `x==1` before
                    // executing the call to `B()`.
                    //
                    // Achieving the more intuitive behavior for fall-through
                    // naively seems to require rewriting the control flow
                    // and introducing nested conditionals (where the nesting
                    // depth relates to the length of the chain of fall-throughs).
                    //
                    // More work is required in order to figure out a good
                    // strategy for dealing with fall-through in the general case.
                    //
                    // For now we can ignore this issue since Slang
                    // does not support fall-through in `switch` statements,
                    // but if/when we do, we will need to make a policy
                    // decision on this issue.
                }

                // Just like for `ifElse`, once we have processed the outgoing
                // conditional edges, we need to process all child regions.
                //
                transformChildRegions(&switchRegion, region);
            }
            break;
        }
    }

    // During the presentation of `transformRegion` we deferred
    // the details of how to deal with edges and child regions to
    // subroutines, which we will begin to work through now.
    //
    // The case of conditional edges is the easiest.
    //
    void transformConditionalEdge(
        RegionInfo* fromRegion,
        IRTerminatorInst* terminator,
        IRBlock* toBlock,
        IRInst* fromActiveMask)
    {
        SLANG_UNUSED(fromRegion);
        SLANG_UNUSED(terminator);

        // Because of the way that we broke critical edges, block with the
        // conditional branch (the entry block to `fromRegion`) had better
        // be the immediate dominator of the block being branched to.
        //
        SLANG_ASSERT(m_dominatorTree->immediatelyDominates(fromRegion->entryBlock, toBlock));

        // If the block being branched from immediately dominates the block being
        // branched to, that makes it the only predecessor of the `toBlock`.
        //
        // As such, the `toBlock` can't have had an SSA parameter/phi introduced
        // to receive the block, and the only value needed to represent the
        // active mask on input to `toBlock` is the `fromActiveMask` being
        // provided as part of the conditional branch.
        //
        m_activeMaskForBlock.add(toBlock, fromActiveMask);
    }

    // Unconditional edges are more complicated than conditional
    // edges, because they may branch out of a control-flow
    // region and/or branch to a block with multiple predecessors.
    //
    void transformUnconditionalEdge(
        RegionInfo* fromRegion,
        IRTerminatorInst* terminator,
        IRBlock* toBlock,
        IRInst* fromActiveMask)
    {
        IRBuilder builder(m_module);
        builder.setInsertBefore(terminator);

        // The context here is that the `terminator` instruction,
        // at the end of the first block of `fromRegion` is
        // branching to `toBlock`. The `fromActiveMask` is the
        // active mask that was in place when executing the
        // first block of `fromRegion`, and thus the active
        // mask used when executing the `terminator`.
        //
        // One task we need to deal with is setting up
        // the active mask that should be used when executing
        // the `toBlock`. We start by assuming that the active
        // mask does not change when control flow moves from
        // one block to the next (we will then deal with all
        // the cases where this assuption isn't true).
        //
        IRInst* toActiveMask = fromActiveMask;

        // The most important thing we need to deal with is
        // that an unconditional edge may exit a control-flow
        // region. We know that the block being branched from
        // is inside `fromRegion`, and thus recursively inside
        // all the ancestors of that region.
        //
        // We will walk through the regions from the inner-most
        // to the outer-most and check if we are exiting each
        // region in turn.
        //
        for (RegionInfo* r = fromRegion; r; r = r->parentRegion)
        {
            // To know if we are exiting the region `r`, we
            // need to look at its merge block.
            //
            auto mergeBlock = r->mergeBlock;

            // One subtle detail here is that we might technically
            // be exiting a region A with merge block M, but the
            // parent of A is a region B that *also* has M as
            // its merge block.
            //
            // In such a case we really don't want/need to deal
            // with any issues around reconvergence at M for A,
            // since we can instead rely on the reconvergence logic
            // for B to do all the work.
            //
            // In practical terms, this means that we skipp any
            // region that has a parent region with the same merge
            // block.
            //
            // TODO: We could try to find ways to eliminate this
            // situation as part of the structure of regions,
            // so that code like this doesn't need the ad hoc check.
            //
            auto parentRegion = r->parentRegion;
            if (parentRegion && parentRegion->mergeBlock == mergeBlock)
                continue;

            // We need to know if the branch exits the region `r`
            // and, if it does, we need to know whether it exits
            // the region normally or not.
            //
            // If the block being branched to is *inside* region `r`
            // (which can only be the case for a non-null target
            // block), then we clearly aren't exiting region `r`.
            //
            if (toBlock && isBlockInRegion(toBlock, r))
            {
                // Furthermore, if we aren't exiting region `r`,
                // then we must not be exiting any of its parent
                // regions, since our regions are strictly
                // nested.
                //
                // We thus don't need to process any more
                // regions to check if we are exiting them.
                //
                break;
            }
            //
            // A normal exit from the region can be detected easily:
            // it is any case where the destination of the branch is
            // the merge block for the region.
            //
            else if (toBlock == mergeBlock)
            {
                // In this case we are jumping to the dedicated
                // merge point of region `r`, which means we need
                // to re-converge with all the other threads/lanes
                // that entered the region together, and which are
                // also exiting normally.
                //
                // It is possible that the `mergeBlock` because
                // it representing the merge point upon return from
                // the function, so we guard against that case.
                //
                if (mergeBlock)
                {
                    // Otherwise, if we are branching to a non-null
                    // merge point, then we emit a ballot operation
                    // that will detect all the other threads/lanes:
                    //
                    //      toActiveMask = waveMaskBallot(activeMaskOnEntry, true);
                    //
                    // This call will synchronize with all the threads/lanes
                    // that entered region `r`, and will collect the
                    // mask representing the threads/lanes that passed
                    // in `true` because they want to re-converge. That
                    // mask should be used as the new active mask when
                    // executing the merge block.
                    //
                    // TODO: Instead of emitting this `waveMaskBallot`
                    // on every edge that branches to the merge point,
                    // we could instead emit a single `waveMaskBallot`
                    // at the start of the merge point instead. This
                    // would be a good optimization to make, but requires
                    // extra logic throughout this algorithm to manage.
                    //
                    toActiveMask = builder.emitWaveMaskBallot(
                        m_maskType,
                        r->activeMaskOnEntry,
                        builder.getBoolValue(true));
                }

                // If we are exiting a region normally, then we can't
                // also be exiting its parent region(s), because we
                // eliminated the case of regions that have the same
                // merge point as their parent at the top of our loop.
                //
                // Thus we are done checking if regions have been
                // exited.
                //
                break;
            }
            else
            {
                // Finally, if we aren't staying in the region,
                // but we aren't jumping to the merge point, then
                // we must be exiting the region "abnormally."
                //
                // In this case, we need to coordinate with the
                // other threads/lanes that entered the region
                // to let them know we won't be reconverging
                // with them.
                //
                // The other threads will be executing a `waveMaskBallot`
                // to compute the active mask to use, so we can
                // participate in that same ballot:
                //
                //      waveMaskBallot(activeMaskOnEntry, false);
                //
                // We don't care about the mask that is returned from
                // the ballot operation, since it will only represent
                // the active mask for threads that wanted to re-converge.
                //
                builder.emitWaveMaskBallot(
                    m_maskType,
                    r->activeMaskOnEntry,
                    builder.getBoolValue(false));
            }
        }

        // After we've checked all the outer regions to see
        // which (if any) we are exiting, we should have
        // a ussable value for `toActiveMask` that we want
        // to pass along to the target block.
        //
        // The way that we pass things along to the target
        // block will vary a lot based on its form.
        //
        if (!toBlock)
        {
            // First, if the target block is null, then
            // the branch is exiting the current function
            // completely, so we don't need to wire up an
            // active mask at all.
        }
        else if (toBlock->getPredecessors().getCount() > 1)
        {
            if (doesBlockNeedActiveMask(toBlock))
            {
                // If the target block is one with multiple
                // predecessors, such that it will have an
                // added block parameter (phi node) to select
                // the corect mask value, then we need to
                // pass along the mask value to use as an
                // additional argument on the unconditional branch.
                //
                // If the old unconditional branch was:
                //
                //      <op>(arg0, arg1, arg2, ...);
                //
                // Then our new branch will be:
                //
                //      <op>(arg0, arg1, arg2, ..., toActiveMask);
                //
                List<IRInst*> newOperands;
                UInt oldOperandCount = terminator->getOperandCount();
                for (UInt i = 0; i < oldOperandCount; ++i)
                {
                    newOperands.add(terminator->getOperand(i));
                }
                newOperands.add(toActiveMask);

                IRInst* newTerminator = builder.emitIntrinsicInst(
                    terminator->getFullType(),
                    terminator->getOp(),
                    newOperands.getCount(),
                    newOperands.getBuffer());

                terminator->replaceUsesWith(newTerminator);
                terminator->removeAndDeallocate();
            }
        }
        else
        {
            // If the target block has only a single predecessor,
            // then it means it must be immediately dominated
            // by the block we are branching from.
            //
            // As such, this is the only branch that wants to
            // establish an active mask value for the target
            // block, and we can just bind the comptue value
            // directly.
            //
            m_activeMaskForBlock.add(toBlock, toActiveMask);
        }
    }

    // During the course of transforming a control-flow region,
    // we deferred the processing of its child regions to
    // a subroutine.
    //
    void transformChildRegions(RegionInfo* innerRegion, RegionInfo* outerRegion)
    {
        // The input to this function represents a (closed) range of regions
        // between `innerRegion` and `outerMergePoint`.
        //
        // The `innerRegion` represents the body of some control-flow construct,
        // and can be used to decide which basic blocks are inside the construct
        // and thus need to be processed as child regions.
        //
        // The `outerRegion` represents the outer context in which the
        // control-flow construct was found, and determines how control-flow
        // entered the construct.
        //
        // If `innerRegion == outerRegion` then there is only one region
        // in the range, while if `innerRegion != outerREgion` then the range
        // comprises: `innerRegion`, `innerRegion->parent`, ... `outerRegion`.
        //
        // There are two kinds of child regions we need to concern ourselves
        // with, and they relate to two kinds of blocks:
        //
        // * Blocks that are inside the inner region represent the start
        //   of their own nested single-entry multi-exit regions. We need
        //   to process these while building up the correct parent/child hierarchy.
        //
        // * The merge block(s) for our range of regions (from inner to
        //   outer), which represent code "after" each of the regions,
        //   that is still nested in the same parent region.
        //
        // We start by enumerating the first kind of child region.
        // To make sure that we only consider blocks that represent the
        // immediate children of `innerRegion`, and not other descendents,
        // we will only look for regions that start with blocks that
        // are children in the dominator tree for the entry block of
        // our overall range of regions.
        //
        IRBlock* entryBlock = outerRegion->entryBlock;
        for (auto childBlock : m_dominatorTree->getImmediatelyDominatedBlocks(entryBlock))
        {
            // We don't want to consider blocks that are outside of
            // the inner region here (they could be, e.g., the merge
            // point of the region, or just blocks inside some
            // nested control-flow construct).
            //
            if (!isBlockInRegion(childBlock, innerRegion))
                continue;

            // If we do find a child region, then we want to process it as
            // a direct child of our inner region.
            //
            transformChildRegion(childBlock, innerRegion);
        }

        // Once we've dealt with the child regions that involve
        // traversing deeper down the tree of regions, we need
        // to deal with merge blocks, which represent siblings
        // of regions in our range.
        //
        // We will walk the regions in our range from the inner-most
        // to the outer-most and look at their merge blocks.
        //
        for (auto mergeRegion = innerRegion; mergeRegion; mergeRegion = mergeRegion->parentRegion)
        {
            // The merge point of a region forms the start of a
            // sibling region that shares the same parent region.
            //
            auto parentRegion = mergeRegion->parentRegion;

            // Well, actually there is one special case we need to
            // deal with: if the parent region of our region has
            // the same merge point, then  the merge region can't
            // actually be one of its children.
            //
            if (parentRegion && parentRegion->mergeBlock == mergeRegion->mergeBlock)
            {
                // In the case where the parent region has the same
                // merge point, we choose not to process it here
                // (since that would be inaccurate), and assume that
                // it will be handled when this loop processes that
                // parent region (perhaps as part of the same call to
                // `transformChildRegions()`, and perhaps as part of
                // another call).
            }
            else
            {
                // In the ordinary case where the parent region has a different
                // merge point, then we know that the merge block for our region
                // needs to be processed as a child of that parent region.
                //
                transformChildRegion(mergeRegion->mergeBlock, parentRegion);
            }

            // We bail out of this loop once we've processed all the regions
            // in the range we've been asked to handle.
            //
            // Note: This check can't just be folded into the `for` loop
            // condition because we need to make sure that we process
            // at least one region in the case where `innerRegion == outerRegion`.
            //
            if (mergeRegion == outerRegion)
                break;
        }
    }

    void transformChildRegion(IRBlock* regionEntry, RegionInfo* parentRegion)
    {
        // This function gets called when we've disocvered a child region
        // that should be processed recursively. The region that starts
        // with `regionEntry` needs to be processed as a child of the
        // given `parentRegion`.
        //
        // It is posible for this routine to be called to process the
        // merge point for the entire function body, which is a null
        // region. There is nothing that needs to be done in that case.
        //
        if (!regionEntry)
            return;

        // Similarly, we don't need to do anything for regions that don't
        // need or use the active mask (even to pass along to their
        // successors).
        //
        if (!doesBlockNeedActiveMask(regionEntry))
            return;

        // An important invariant of our approach is that before a child
        // region is processed, an active mask value will have been
        // established for it. The logic can be thought of as follows:
        //
        // * If the child region is one with multiple predecessors, then
        //   its active mask value comes in as a block parameter (phi node),
        //   and was set up before the recursive walk even started.
        //
        // * If the child region has only one predecessor, then that predecessor
        //   is its immediate dominator, and would be part of a parent region
        //   that has already been processed up the call stack. That parent
        //   region would have established the active mask value to use on input
        //   to each of its successors.
        //
        IRInst* activeMaskOnRegionEntry = nullptr;
        if (!m_activeMaskForBlock.tryGetValue(regionEntry, activeMaskOnRegionEntry))
        {
            SLANG_UNEXPECTED("no active mask registered for block");
        }

        // Once we've done the sanity checks and looked up
        // the mask value to use on entry to this region, we
        // can recursively process it (and by extension, its children).
        //
        RegionInfo region;
        region.entryBlock = regionEntry;
        region.activeMaskOnEntry = activeMaskOnRegionEntry;
        region.mergeBlock = parentRegion->mergeBlock;
        region.parentRegion = parentRegion;
        transformRegion(&region);
    }
};

// Now that we've defined the context for function-level transformation,
// we can finally circle back and fill in the definition of the function
// that the module-level pass uses to transform each function.
//
void SynthesizeActiveMaskForModuleContext::transformFuncUsingActiveMask(IRFunc* func)
{
    SynthesizeActiveMaskForFunctionContext context;
    context.m_func = func;
    context.m_module = m_module;
    context.m_maskType = m_maskType;

    context.transformFunc();
}

// The public entry point for this pass is just a wrapper around
// the context type for the module-level pass.
//
void synthesizeActiveMask(IRModule* module, DiagnosticSink* sink)
{
    SynthesizeActiveMaskForModuleContext context;
    context.m_module = module;
    context.m_sink = sink;
    context.processModule();
}

} // namespace Slang
