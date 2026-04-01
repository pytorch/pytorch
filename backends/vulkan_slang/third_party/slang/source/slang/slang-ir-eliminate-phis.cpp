// slang-ir-eliminate-phis.cpp
#include "slang-ir-eliminate-phis.h"

#include "slang-ir-ssa-register-allocate.h"
#include "slang-ir-util.h"

// This file implements a pass to take code in the Slang IR out out SSA form
// by eliminating all "phi nodes."
//
// The Slang IR does not represent phi operations in the "textbook" way, and
// it is important to understand how it *does* represent those operations in
// order to understand this pass.
//
// In a textbook encoding of SSA, a basic block begins with zero or more `phi`
// instructions:
//
//      block B:
//          let x = phi(a, b);
//          let y = phi(c, d);
//          ...
//
// Each `phi` operation has a number of operands equal to the number of
// predecessors of `B`, with one operand corresponding to each predecessor.
// Semantically, we usually think of a `phi` as auto-magically selecting between
// its operands beased on the control-flow path that was used to arrive at `B`.
// Note, however, that all of the `phi` operations at the start of a block must
// be understood as executing *simultaneously*, so that if `x` or `y` above was
// used as a `phi` operand, they would yield their value from *before* the binding
// not after. Oh, and also the operands of `phi` instructions are the *one* place
// where the rule about how "defs must dominate uses" isn't upheld.
//
// Our encoding of the same thing is equivalent, more consistent, and in most ways
// easier to understand. A basic block in the Slang IR can have *parameters*,
// and a branch to such a block must pass *arguments* for those parameters:
//
//      block B(x,y):
//          ...
//
//      block M:
//          ...
//          br B(a, c);
//
//      block N:
//          ...
//          br B(b, d);
//
// Note how in this formulation there is no auto-magical behavior. The relationship
// between a predecessor block and the values that `x, y` take on is clear. The
// way that `x, y` take on their new values simulataneously is also more apparent
// (since it intuitively matches a recursive function call). Further, our IR is
// able to follow the "def dominates use" rule more completely.
//
// With that out of the way, let's dive into the pass itself.

#include "slang-ir-dominators.h"
#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{

struct PhiEliminationContext
{
    // We are going to make some effort to re-use intermediate structures across
    // different functions that we process, so that we don't do more allocation
    // than necessary.
    //
    // At the top level, our pas needs to have access to the IR module, and needs
    // a builder it can use to generate code.
    //
    IRModule* m_module = nullptr;
    IRBuilder m_builder;
    LivenessMode m_livenessMode;
    PhiEliminationOptions m_options;

    PhiEliminationContext(LivenessMode livenessMode, IRModule* module)
        : m_module(module), m_builder(module), m_livenessMode(livenessMode), m_options()
    {
    }

    PhiEliminationContext(
        LivenessMode livenessMode,
        IRModule* module,
        PhiEliminationOptions options)
        : m_module(module), m_builder(module), m_livenessMode(livenessMode), m_options(options)
    {
    }

    // We start with the top-down logic of the pass, which is to process
    // the functions in the module one at a time.
    //
    // Note that global variables are also included here, since they are
    // also `IRGlobalValueWithCode`s, but we don't typically expect them
    // to have more than one basic block, much less phis.
    //
    void eliminatePhisInModule()
    {
        for (auto inst : m_module->getGlobalInsts())
        {
            switch (inst->getOp())
            {
            default:
                continue;

            case kIROp_Func:
            case kIROp_GlobalVar:
                break;
            }

            auto code = (IRGlobalValueWithCode*)inst;
            eliminatePhisInFunc(code);
        }
    }

    // Within a single function, we are primarily concerned with processing
    // each of its blocks.
    //
    void eliminatePhisInFunc(IRGlobalValueWithCode* func)
    {
        // Perform initialization and register allocation
        // for Phi parameters and other insts that benefit from
        // converting to memory.
        initializePerFuncState(func);

        // First, we eliminate all the phi instructions (params)
        // using the result of register allocation.
        // The first block in a function is always the entry block,
        // and its parameters are different than those of the other blocks;
        // they represent the parameters of the *function*. We therefore
        // need to skip the first block.
        //
        auto firstBlock = func->getFirstBlock();
        for (auto block : func->getBlocks())
        {
            if (block == firstBlock)
            {
                continue;
            }

            eliminatePhisInBlock(block);
        }

        // Next, convert the definition of other ordinary insts to assignments.
        convertInstDefToRegisterAssignment();

        // Finally, replaces the uses of other ordinary insts to loads from registers.
        replaceInstUseWithRegisterLoad();
    }

    void convertInstDefToRegisterAssignment()
    {
        IRBuilder builder(m_module);

        for (const auto& [inst, reg] : m_registerAllocation.mapInstToRegister)
        {
            IRInst* registerVar = nullptr;
            m_mapRegToTempVar.tryGetValue(reg, registerVar);
            SLANG_RELEASE_ASSERT(registerVar);

            switch (inst->getOp())
            {
            case kIROp_Param:
                continue;
            case kIROp_UpdateElement:
                {
                    auto updateInst = as<IRUpdateElement>(inst);
                    builder.setInsertBefore(updateInst);
                    RefPtr<RegisterInfo> oldReg;
                    m_registerAllocation.mapInstToRegister.tryGetValue(
                        updateInst->getOldValue(),
                        oldReg);
                    // If the original value is not assigned to the same register as this inst,
                    // we need to insert a copy.
                    if (reg != oldReg)
                    {
                        builder.emitStore(registerVar, updateInst->getOldValue());
                    }
                    // Perform update on the register var.
                    auto elementAddr = builder.emitElementAddress(
                        registerVar,
                        updateInst->getAccessChain().getArrayView());
                    builder.emitStore(elementAddr, updateInst->getElementValue());
                }
                break;
            default:
                break;
            }
        }
    }

    void replaceInstUseWithRegisterLoad()
    {
        IRBuilder builder(m_module);

        for (const auto& [inst, reg] : m_registerAllocation.mapInstToRegister)
        {
            IRInst* registerVar = nullptr;
            m_mapRegToTempVar.tryGetValue(reg, registerVar);
            SLANG_RELEASE_ASSERT(registerVar);
            while (auto use = inst->firstUse)
            {
                auto user = use->getUser();
                m_builder.setInsertBefore(user);
                auto newVal = m_builder.emitLoad(registerVar);
                use->set(newVal);
            }
            inst->removeAndDeallocate();
        }
    }

    // In order to facilitate breaking things down into subroutines, we use a
    // member variable to track the function currently being processed, and
    // also a dominator tree for that function.
    //
    IRGlobalValueWithCode* m_func = nullptr;
    RefPtr<IRDominatorTree> m_dominatorTree;
    RegisterAllocationResult m_registerAllocation;
    Dictionary<RegisterInfo*, IRInst*> m_mapRegToTempVar;

    // Because we use the same `PhiEliminationContext` to process all of
    // the functions in a module, we need to set up these per-function
    // state variables for each new function encountered.
    //
    void initializePerFuncState(IRGlobalValueWithCode* func)
    {
        m_func = func;
        m_dominatorTree = nullptr;

        if (m_options.useRegisterAllocation)
        {
            m_registerAllocation = allocateRegistersForFunc(
                func,
                m_dominatorTree,
                m_options.eliminateCompositeTypedPhiOnly);
            m_mapRegToTempVar = createTempVarForInsts(func);
        }
    }

    Dictionary<RegisterInfo*, IRInst*> createTempVarForInsts(IRGlobalValueWithCode* func)
    {
        Dictionary<RegisterInfo*, IRInst*> mapRegToVar;
        for (auto& regList : m_registerAllocation.mapTypeToRegisterList)
        {
            auto type = regList.key;
            for (auto reg : regList.value)
            {
                // Find the common dominator for all the insts, and determine the latest insertion
                // point of the tempVar inst.
                IRBlock* dom = nullptr;
                IRInst* insertionPoint = nullptr;
                for (auto inst : reg->insts)
                {
                    // Determine where the temp register var should be inserted if
                    // it represents only `inst`.
                    IRBlock* thisDom = as<IRBlock>(inst->getParent());
                    IRInst* thisInsertionPoint = inst;
                    if (inst->getOp() == kIROp_Param)
                    {
                        thisDom = getDominatorTree()->getImmediateDominator(thisDom);
                        thisInsertionPoint = thisDom->getTerminator();
                    }

                    // Push the insertionPoint early enough to dominate `thisInsertionPoint`.
                    if (dom == nullptr)
                    {
                        dom = thisDom;
                        insertionPoint = thisInsertionPoint;
                    }
                    else
                    {
                        auto domTree = getDominatorTree();
                        while (!domTree->dominates(dom, thisDom) && dom != func->getFirstBlock())
                        {
                            dom = domTree->getImmediateDominator(dom);
                            insertionPoint = dom->getTerminator();
                        }
                    }
                    // Move insertion point to before thisInsertionPoint.
                    if (dom == thisDom)
                    {
                        bool isInsertionPointBeforeCurrentInst = false;
                        for (auto current = insertionPoint; current;
                             current = current->getNextInst())
                        {
                            if (current == thisInsertionPoint)
                            {
                                isInsertionPointBeforeCurrentInst = true;
                                break;
                            }
                        }
                        if (!isInsertionPointBeforeCurrentInst)
                            insertionPoint = thisInsertionPoint;
                    }
                }
                SLANG_ASSERT(dom);
                SLANG_ASSERT(insertionPoint && insertionPoint->getParent() == dom);
                m_builder.setInsertBefore(insertionPoint);

                // Note that the `emitVar` operation expects to be passed the
                // type *stored* in the variable, but the IR `var` instruction
                // itself will have a pointer type. Thus if `param` has type
                // `T`, then `temp` will have type `T*`.
                //
                auto temp = m_builder.emitVar(type);
                for (auto inst : reg->insts)
                {
                    inst->transferDecorationsTo(temp);
                }
                mapRegToVar[reg] = temp;
            }
        }
        return mapRegToVar;
    }

    // The dominator tree for the function is computed on demand and
    // cached. We do this to avoid the expensive of allocating the
    // `IRDominatorTree` structure in cases where a function doesn't
    // end up having any phis that need elimination. Note that any
    // "straight-line" function taht doesn't involve control flow
    // will never have any phis, so we expect that case to be common.
    //
    IRDominatorTree* getDominatorTree()
    {
        if (!m_dominatorTree)
        {
            m_dominatorTree = computeDominatorTree(m_func);
        }
        return m_dominatorTree;
    }

    // The meat of the work happens on a per-basic-block basis.
    //
    void eliminatePhisInBlock(IRBlock* block)
    {
        // We start by checking if the block has any parameters.
        // If it doesn't then there is nothing to eliminate.
        //
        if (!block->getFirstParam())
        {
            return;
        }

        // Once the early-exit case has been dealt with, the overall
        // process amounts to three simple steps.
        //
        // 1. Create a temporary corresponding to each of the
        // parameters of `block`.
        //
        collectPhiInfoForParams(block);
        //
        // 2. For each predecessor of `block`, eliminate the arguments
        // it passes, by assigning them to the temporaries.
        //
        emitAssignmentsInPredecessors(block);
        //
        // 3. Replace all (remaining) uses of the block parameters with
        // loads from the temporaries.
        //
        replaceParamUsesWithTemps();
    }

    // We need to record information about the parameters and the temporaries
    // we create for them, so that subsequent steps can easily access them.
    //
    struct ParamInfo
    {
        IRParam* param = nullptr;
        IRVar* temp = nullptr;

        // We track one additional field for each parameter, which is
        // used to record its "current" value for the purposes of
        // emitting assignments at the end of a predecessor block.
        //
        // The `currentVal` field will either be the same as `param`
        // itself (if using the parameter directly is still safe) or
        // a value loaded from `temp`, after the point where this
        // parameter has been assigned to.
        //
        IRInst* currentVal = nullptr;
    };

    // We build a  mapping from block parameters to their indices,
    // which makes it convenient to look up whether a given `IRInst*`
    // is a parameter and, if so, get its index in a single operation.
    //
    Dictionary<IRInst*, Index> mapParamToIndex;

    void collectPhiInfoForParams(IRBlock* block)
    {
        // The temporaries used to replace the parameters of `block`
        // must be read-able any where that the parameters were accessible.
        // They must also be write-able at every point that branches
        // *into* `block`. The most narrowly-scoepd place that meets both
        // of those criteria is the *immediate dominator* of `block`.
        //
        if (auto blockForTemps = getDominatorTree()->getImmediateDominator(block))
        {
            // We will insert our new teporary variables at the *end* of the
            // immediate dominator block, right before the terminator. In
            // the case where the immediate dominator is also one of the
            // predecessors of `block`, that terminator will branch to `block`,
            // and we need the temporary variables to be in scope there, but
            // no earlier.
            //
            auto terminator = blockForTemps->getTerminator();
            SLANG_ASSERT(terminator);
            m_builder.setInsertBefore(terminator);
        }
        else
        {
            // There are two cases where a `block` would not have an immedidate
            // dominator. The first is that is that it is the entry block of
            // its function, but we already skipped over such blocks earlier.
            // The second case is that `block` is unreachable.
            //
            // In the case of an unreachable block, it doesn't especially
            // matter what we do. In principle we could leave such blocks
            // as-is and expect later steps to ignore tham and/or their
            // parameters.
            //
            // In an attempt to make this code as robust as possible, we
            // will handle any unreachable blocks by inserting the
            // temporary variables right after the parameters (which means
            // right *before* the ordinary body instructions).
            //
            // Note that nothing in the code here will *initialize* those
            // temporaries, so if the unreachable code were to somehow
            // get executed, the values would be undefined.
            //
            auto firstOrdinaryInst = block->getFirstOrdinaryInst();
            SLANG_ASSERT(firstOrdinaryInst);
            m_builder.setInsertBefore(firstOrdinaryInst);
        }

        // Now that we've set up the IR builder for inserting our
        // temporaries, we are going to iterate over the parameters
        // and create a temporary for each. Along the way we will
        // be building up auxilliary data structures that the
        // subsequent steps will make use of.
        //
        mapParamToIndex.clear();
        phiInfos.clear();
        Count paramCounter = 0;
        for (auto param : block->getParams())
        {
            Index paramIndex = paramCounter++;
            mapParamToIndex.add(param, paramIndex);

            IRInst* temp = nullptr;

            // Have we already allocated a register for this inst?
            // If so we use the var for that register.
            if (auto registerInfo = m_registerAllocation.mapInstToRegister.tryGetValue(param))
            {
                m_mapRegToTempVar.tryGetValue(registerInfo->get(), temp);
            }

            bool shouldAllocTemp =
                !m_options.eliminateCompositeTypedPhiOnly || isCompositeType(param->getFullType());

            if (!temp && shouldAllocTemp)
            {
                // Note that the `emitVar` operation expects to be passed the
                // type *stored* in the variable, but the IR `var` instruction
                // itself will have a pointer type. Thus if `param` has type
                // `T`, then `temp` will have type `T*`.
                //
                temp = m_builder.emitVar(param->getDataType());
                //
                // Because we will be eliminating the paramter, we can transfer
                // any decorations that were added to it (notably any name hint)
                // to the temporary that will replace it.
                //
                param->transferDecorationsTo(temp);
                temp->sourceLoc = param->sourceLoc;
            }

            // The other main auxilliary sxtructure is used to track
            // both per-parameter information (which we can fill in
            // here) and information about each value *assigned* to
            // a parameter at a branch site. Both kinds of information
            // are stored in the same array, but we only initialize the
            // relevant fields here.
            //
            PhiInfo phiInfo;
            auto& paramInfo = phiInfo.param;
            paramInfo.param = param;
            paramInfo.temp = as<IRVar>(temp);
            phiInfos.add(phiInfo);
        }
    }


    // The work of emitting assignments to our temporaries in
    // the predecessors of `block` is really a per-predecessor task.
    //
    void emitAssignmentsInPredecessors(IRBlock* block)
    {
        // The only interesting detail at this level is that
        // we need to work with a *copy* of the predecessor
        // list. Our manipulation replaces the branch instruction
        // at the end of each predecessor by adding a new one and
        // removing the old. The addition/removing of branch
        // instructions causes the predecessor list of `block` to
        // be mutated, even if its contents end up the same.
        //
        List<IRBlock*> predecessors;
        for (auto pred : block->getPredecessors())
        {
            predecessors.add(pred);
        }
        for (auto pred : predecessors)
        {
            emitAssignmentsInPredecessor(pred);
        }
    }

    // We will put off discussion of `emitAssignmentsInPredecessor()`
    // for now, because it is the thorniest part of the problem.
    //
    // Instead, let us look at the far simpler task of eliminating
    // all the *other* uses of block parameters, after the branches
    // have been dealt with.
    //
    void replaceParamUsesWithTemps()
    {
        for (auto& phiInfo : phiInfos)
        {
            auto& paramInfo = phiInfo.param;
            auto param = paramInfo.param;
            auto temp = paramInfo.temp;

            if (!temp)
                continue;

            // We will repeatedly replace whatever the *first*
            // use of `param` is, until there are no more uses
            // left. Iterating in this fashion avoids any
            // problems that would arise from trying to traverse
            // the list of uses while also modifying it.
            //
            while (auto use = param->firstUse)
            {
                // We emit a distinct `load` from the temporary
                // right before each instruction that uses the
                // parameter. We do this to minimize the number
                // of temporaries/copies that are created in
                // the emit logic for high-level-language targets.
                //
                // We have logic that can "fold" a `load` instruction
                // into a use site such that it shows up as an ordinary
                // variable reference, but this logic currently only
                // applies if there are no `store`s or other operations
                // with possible side effects between the `load` and
                // the place where it gets used.
                //
                // An alternative implementation of this pass might
                // `load` each of our temporaries once, at the top of
                // `block`, and then use that same value at all use sites.
                //
                auto user = use->getUser();
                m_builder.setInsertBefore(user);
                auto newVal = m_builder.emitLoad(temp);
                newVal->sourceLoc = param->sourceLoc;
                m_builder.replaceOperand(use, newVal);
            }

            // Once we've replaced all its uses, there is no need
            // to keep `param` around.
            //
            param->removeAndDeallocate();
        }
    }

    // Now it is time to get back to `emitAssignmentsInPredecessor()`.
    //
    // As discussed in `replaceParamUsesWithTemps()`, we want to avoid
    // emitting high-level-language output code with unnecessary copies.
    // That goal makes the process of emitting assignments to our
    // temporarites in the predecessors of `block` more challenging.
    //
    // To understand the challenge, consider a block like:
    //
    //      block B(x,y):
    //          ...
    //
    // and a branch to that block of the form:
    //
    //      br B(y, x);
    //
    // The phi operations here are effectively swapping `x` and `y`,
    // so we know that output code will need at least *one*
    // intermediate copy to do the job.
    //
    // We don't want to see output like:
    //
    //      // br B(y,x);
    //      x = y;
    //      y = x;
    //
    // but we also don't want to see more copies then necessary:
    //
    //      // br B(y,x);
    //      auto tmp_x = x;
    //      auto tmp_y = y;
    //      x = tmp_y;
    //      y = tmp_x;
    //
    // Our goal is emit a strictly *minimal* number of copies.
    //
    // In order to solve the problem, we need to track some information
    // on a per-branch-site basis. The most obvious of this is information
    // about each argument that the branch pasess to a corresponding
    // block parameter.
    //
    struct ArgInfo
    {
        // We track the original argument value that was passed at the branch.
        //
        IRInst* originalVal = nullptr;

        // At a branch site, we can consider that the goal is to assign
        // each argument (`ArgInfo`) to the temporary for a corresponding
        // parameter (`ParamInfo`).
        //
        // The problematic cases arise when an argument value is itself
        // a reference to a block parameter (e.g., in the `(y,x) -> (x,y)`
        // case). We thus track whether or not `originalVal` above is
        // itself a block parameter and, if so, what the index of that
        // parameter is.
        //
        Index paramIndex = kInvalidIndex;

        // When there is a cyclic dependency between arguments at
        // a branch site, no sequence of plain assignments (without
        // additional copies) will suffice.
        //
        // When we end up having to break cycles, we do so by loading
        // a copy of the value in one of the per-parameter temporaries.
        // Any subsequent branch arguments that referenced the parameter
        // will need to use that copy instead.
        //
        // In order to make sure that we properly reference the loaded
        // copy instead of the original argument in such cases, we use
        // a pointer field for each argument that points to a location
        // where the up-to-date value can be found.
        //
        // For most arguments this will always just point to `originalVal`,
        // but for arguments that refer to a block parameter, this will
        // point to the `currentVal` field of the corresponding `ParamInfo`.
        //
        IRInst** currentValPtr = nullptr;
    };
    enum
    {
        kInvalidIndex = -1
    };

    // A lot of the logic in this pass is concerned with the process of
    // emitting *assignments* from branch arguments to block parameters.
    // Those assignments are implicit in our SSA IR, but this pass needs
    // to make them explicit.
    //
    // In order to emit assignments in an order that minimizes the number
    // of temporaries/copies that appear in output code, we need a way
    // to track which assignments have been done, which are ready to be done,
    // and which are "blocked" for some reason. We thus associate each
    // assignment with an integer _state_ which is in one of three cases:
    //
    // * _done_ (`-1`): any instructions needed for the assignment have been emitted
    //
    // * _ready_ (`0`): the assignment can be emitted without causing any problems
    //
    // * _blocked_ (`N > 0`): the assignment cannot be emitted yet, because there
    // are `N` other not-yet-done assignments that need to read the value of the
    // parameter that this assignment wants to write. The reads need to be able
    // to proceed before the write can go through.
    //
    enum
    {
        kState_Done = -1,
        kState_Ready = 0,
    };

    // There is a one-to-one correspondance between:
    //
    // * The phis/parameters of a particular `block`
    // * The arguments passed at some branch to `block`
    // * The assignments that need to be performed at that branch site
    //
    // We thus use a single structure to track all of that information,
    // which is handy but also requires careful thought at use sites
    // about which version of the information is relevant.
    //
    struct PhiInfo
    {
        ParamInfo param;
        ArgInfo arg;
        Count state = kState_Ready;
    };
    List<PhiInfo> phiInfos;
    Count getParamCount() { return phiInfos.getCount(); }

    void initializeAssignmentInfo(IRUnconditionalBranch* branch, Index assignmentIndex)
    {
        // Each assignment is a request to write the value of
        // some `srcArg` to some `dstParam`.
        //
        auto& assignment = phiInfos[assignmentIndex];
        auto& srcArg = assignment.arg;
        auto& dstParam = assignment.param;

        // The actual argument values can be read off of
        // the branch instruction.
        //
        auto srcArgVal = branch->getArg(assignmentIndex);
        srcArg.originalVal = srcArgVal;
        srcArg.currentValPtr = &srcArg.originalVal;

        // The parameters have largely been initialized in the
        // per-block logic, but we do need to (re-)initialize
        // the `currentVal` field to get it ready for a new
        // sequence of assignments.
        //
        dstParam.currentVal = dstParam.param;

        // The main challenges arise when the argument value
        // for an assignment is itself one of the parameters
        // of the destination block.
        //
        // We can check if `srcArgVal` is a parameter using
        // the map we pre-computed.
        //
        Index srcParamIndex = kInvalidIndex;
        mapParamToIndex.tryGetValue(srcArgVal, srcParamIndex);
        srcArg.paramIndex = srcParamIndex;

        if (srcParamIndex != kInvalidIndex)
        {
            // In the case where the source *is* a parameter,
            // we may not be able to use the source parameter value
            // directly for this assignment, since we might
            // need to make a scratch copy of it.
            //
            // To be able to keep up with such changes if they
            // occur, we will update `currentValPtr` to point
            // to the `currentVal` of the source parameter.
            //
            auto& srcParamInfo = phiInfos[srcParamIndex].param;
            srcArg.currentValPtr = &srcParamInfo.currentVal;
        }

        // One very special case is when the source value to be assigned
        // to a destination phi/parameter is the exact same phi/parameter.
        // In such a case there is nothing that needs to be done, and we
        // can consider the assignment fully handled.
        //
        if (srcParamIndex == assignmentIndex)
        {
            assignment.state = kState_Done;
        }
        else if (!dstParam.temp)
        {
            assignment.state = kState_Done;
        }
        else
        {
            // Otherwise we start out assuming that the assignment is ready
            // to proceed, and then check to see whether it should be
            // blocked as part of the next loop.
            //
            assignment.state = kState_Ready;
        }
    }

    void checkIfAssignmentBlocksAnother(Index assignmentIndex)
    {
        // We can skip any case where this assignment has already been
        // performed/resolved, since it cannot lead to anything being
        // blocked waiting on it.
        //
        auto& assignment = phiInfos[assignmentIndex];
        if (assignment.state == kState_Done)
        {
            return;
        }

        // We only care about cases where the source/argument for this
        // assignment is another parameter.
        //
        auto& srcArg = assignment.arg;
        auto srcParamIndex = srcArg.paramIndex;
        if (srcParamIndex == kInvalidIndex)
        {
            return;
        }

        // Note that the sticky case of a parameter that refers to itself
        // was already detected and handled in the previous loop.
        //
        SLANG_ASSERT(srcParamIndex != assignmentIndex);
        //
        // In fact, it is possible that the assignment for `srcParamIndex`
        // has already been completed, since it was one of the self-referential
        // cases. If that's true and the assignment is already marked as
        // done, there is no reason to try and block it.
        //
        auto& srcParamAssignment = phiInfos[srcParamIndex];
        if (srcParamAssignment.state == kState_Done)
        {
            return;
        }

        // Otherwise we are in exactly the case we are looking for.
        // This `assignment` is of the form:
        //
        //      temps[...] = temps[srcParamIndex];
        //
        // and there is another assignment of the form:
        //
        //      temps[srcParamIndex] = ...;
        //
        // That we cannot allow to proceed until after *this* assignment
        // has been allowed to read from the temporary for `srcParamIndex`.
        //
        // The representation of both the _ready_ and _blocked_ states
        // is equal to the number of "blockers," so in either case we can
        // increment the `state` in order to add in a(nother) blocker.
        //
        srcParamAssignment.state++;
    }

    void emitAssignmentsInPredecessor(IRBlock* pred)
    {
        // Given the way our IR is structured, we have an invariant that the
        // predecessor block *must* end with an unconditional branch (as they
        // are the only kinds of branches allowed to carry arguments).
        //
        auto branch = cast<IRUnconditionalBranch>(pred->getTerminator());
        SLANG_ASSERT(branch);

        // The predecessor block must pass the expected number of arguments
        // to `block`, or the IR has been invalidated in some previous pass.
        //
        auto paramCount = getParamCount();
        Count argCount = branch->getArgCount();
        SLANG_ASSERT(argCount == paramCount);

        // We are going to emit a sequence of assignments that write the
        // arguments of `branch` to the temporaries for the parameters of
        // `block`. All of these assignments have to be the last thing we
        // do before the branch.
        //
        m_builder.setInsertBefore(branch);

        // Our first order of business is to initialize the per-branch-site
        // and per-argument/-assignment information, so that it is correct
        // for `branch`.
        //
        for (Index assignmentIndex = 0; assignmentIndex < argCount; ++assignmentIndex)
        {
            initializeAssignmentInfo(branch, assignmentIndex);
        }

        // We can now scan through our assignments and try to determine
        // which of them are _blocked_.
        //
        // A assignment of `param_i <- arg_i` is blocked if there
        // exists some other not-yet-done assignment of the form
        // `param_j <- param_i`. That is, an assignment is blocked
        // when the parameter it wants to write to is being used as
        // the source/argument for some other assignment (that has not
        // yet been completed).
        //
        // We can thus loop over the assignments and for each one see
        // if it is in the form `param_j <- param_i` for some `i` and,
        // if so, tell the assignment for `param_i <- ....` that it is
        // blocked.
        //
        for (Index assignmentIndex = 0; assignmentIndex < paramCount; ++assignmentIndex)
        {
            checkIfAssignmentBlocksAnother(assignmentIndex);
        }

        // Once we have identified the cases where one assignment is
        // blocked on another, we can scan through the list of assignments
        // and try to perform any assignmenst that are in the _ready_ state,
        // as well as any additional assignments that become _ready_ as a result.
        //
        for (Index assignmentIndex = 0; assignmentIndex < paramCount; ++assignmentIndex)
        {
            tryPerformParamAssignment(assignmentIndex);
        }

        // The only assignments that could not be completed in the previous
        // loop would be those that are part of a dependency cycle.
        // We make one more loop over the assignments, and this time we will
        // ensure that the assignment gets to the _done_ state, even if doing
        // so requires loading a copy of one of our temporaries.
        //
        for (Index assignmentIndex = 0; assignmentIndex < paramCount; ++assignmentIndex)
        {
            completeAssignmentUsingCopyIfNeeded(assignmentIndex);
        }

        // Once we are sure all the assignment operations have been performed,
        // we can set about replacing the unconditional branch itself.
        //
        replaceBranch(branch);
    }

    // Replacing the branch instruction at the end of a predecessor block
    // is relatively simple, and just a bit of busy-work.
    //
    void replaceBranch(IRUnconditionalBranch* oldBranch)
    {
        // When creating a replacement instruction here, we need to make sure
        // that we keep all the operands that weren't phi arguments.
        //
        Count oldOperandCount = oldBranch->getOperandCount();
        Count paramCount = getParamCount();
        Count newOperandCount = oldOperandCount - paramCount;

        // There are currently two different opcodes that map to unconditional
        // branches, with different numbers of operands before the phi-related
        // arguments:
        //
        //      unconditionalBranch(TargetBlock, arg0, arg1, arg2, ...);
        //      loop(TargetBlock, BreakBlock, ContinueBlock, arg0, arg1, arg2, ...);
        //
        // In either case, there is a constant bound on the number of non-phi
        // operands.
        //
        static const Count kMaxNewOperandCount = 3;
        SLANG_ASSERT(newOperandCount <= kMaxNewOperandCount);

        ShortList<IRInst*> newOperands;
        for (Index i = 0; i < newOperandCount; ++i)
        {
            newOperands.add(oldBranch->getOperand(i));
        }

        // Add operands for any remaining phi parameters that has not been eliminated.
        for (UInt i = 0; i < (UInt)phiInfos.getCount(); i++)
        {
            if (!phiInfos[i].param.temp)
                newOperands.add(oldBranch->getArg(i));
        }

        auto newBranch = m_builder.emitIntrinsicInst(
            oldBranch->getFullType(),
            oldBranch->getOp(),
            newOperands.getCount(),
            newOperands.getArrayView().getBuffer());
        oldBranch->transferDecorationsTo(newBranch);
        newBranch->sourceLoc = oldBranch->sourceLoc;

        // TODO: We could consider just modifying `branch` in-place by clearing
        // the relevant operands for the phi arguments and setting its operand
        // count to a lower value.
        //
        oldBranch->removeAndDeallocate();
    }

    bool canLoadBeFoldedAtInst(IRInst* load, IRInst* useSite)
    {
        if (load->getParent() != useSite->getParent())
            return false;

        auto addr = load->getOperand(0);
        switch (addr->getOp())
        {
        case kIROp_Var:
        case kIROp_Param:
            break;
        default:
            return false;
        }
        for (auto inst = load; inst; inst = inst->getNextInst())
        {
            if (inst == useSite)
            {
                return true;
            }
            switch (inst->getOp())
            {
            case kIROp_Store:
            case kIROp_GetElementPtr:
            case kIROp_FieldAddress:
                if (inst->getOperand(0) == addr)
                    return false;
                break;
            default:
                if (inst->mightHaveSideEffects())
                    return false;
                break;
            }
        }
        // Should never reach here if useSite appears after inst.
        // Return false to be safe.
        return false;
    }

    // The most subtle bit of logic, which relies on the data structures
    // we have built so far, is the way we attempt to perform assignments
    // that have become ready.
    //
    void tryPerformParamAssignment(Index firstAssignmentIndex)
    {
        // Performing one assignment may lead to another being unblocked,
        // and entering the _ready_ state, in which case we wnat to perform
        // *that* assignment, which may unblock another, etc.
        //
        // We thus use a loop and keep processing assignments as they
        // become unblocked, until we run out of work.
        //
        Index assignmentIndex = firstAssignmentIndex;
        for (;;)
        {
            auto& assignment = phiInfos[assignmentIndex];
            if (assignment.state != kState_Ready)
            {
                // if the assignment we are looking at isn't ready to
                // perform, we have reached the end of a chain of
                // unblocked depdendencies (perhaps even a chain of zero).
                //
                return;
            }

            auto& dstParam = assignment.param;
            auto& srcArg = assignment.arg;

            // If we have liveness tracking add the start location.
            if (isEnabled(m_livenessMode))
            {
                // A store could (perhaps?) consist of multiple instructions
                // If we make liveness *after* the store, then it implies anything stored
                // into the location might be lost.
                //
                // Therefore is seems appropriate to say the variable is *live* *before* the store
                // instruction.
                m_builder.emitLiveRangeStart(dstParam.temp);
            }

            // When we have an assignment that is ready to perform,
            // we do so by storing the value of the corresponding
            // argument into the temporary for the coresponding
            // parameter.
            //
            // Note that we use `actualValPtr` here instead of `originalVal`,
            // so that any logic that might have moved another parameter
            // into a temporary will influence our result.
            //
            if ((*srcArg.currentValPtr)->getOp() != kIROp_undefined)
            {
                // If we are trying to emit a store directly after a load from the same var,
                // skip the store.
                SLANG_ASSERT(m_builder.getInsertLoc().getMode() == IRInsertLoc::Mode::Before);
                auto srcLoad = as<IRLoad>(*srcArg.currentValPtr);
                if (srcLoad && srcLoad->getOperand(0) == dstParam.temp &&
                    canLoadBeFoldedAtInst(srcLoad, m_builder.getInsertLoc().getInst()))
                {
                }
                else
                {
                    m_builder.emitStore(dstParam.temp, *srcArg.currentValPtr);
                }
            }

            //
            // Once the store is emitted, the assignment has been performed,
            // and it can move to the _done_ state.
            //
            assignment.state = kState_Done;

            // If the source of this assignment as itself a block
            // parameter, then we may need to unblock the assignment
            // for that parameter.
            //
            // If the source *isn't* a parameter, then there is nothing
            // to unblock, and we've reached the end of a chain.
            //
            auto srcParamIndex = srcArg.paramIndex;
            if (srcParamIndex == kInvalidIndex)
            {
                return;
            }

            // If the source *is* a parameter, but its assignment
            // has already been performed, then we cannot unblock it
            // (we certainly don't want to perform it again).
            //
            auto& srcParamAssignment = phiInfos[srcParamIndex];
            if (srcParamAssignment.state == kState_Done)
            {
                return;
            }

            // If the source parameter's assignment hasn't been
            // done yet, then we expect that it *must* be blocked
            // (at the very least blocked on the assignment we
            // have just performed).
            //
            SLANG_ASSERT(srcParamAssignment.state != kState_Ready);

            // We remove one blocker from the source.
            //
            srcParamAssignment.state--;

            // It is possible that removing this one blocker has
            // moved the source parameter into the _ready_ state.
            // Rather than check that here, we can simply move
            // back to the top of this loop and consider the
            // assignment corresponding to the source parameter
            // as the next in our chain.
            //
            assignmentIndex = srcParamIndex;
        }
    }

    // When we want to make sure that all our assignments *definitely*
    // get completed, we need to be willing to make a temporary copy
    // of a branch parameter in order to unblock an assignment.
    //
    void completeAssignmentUsingCopyIfNeeded(Index assignmentIndex)
    {
        // If the assignemtn is _blocked_ because of one or more
        // other assignments, we will unblock it by making a copy.
        //
        auto& assignment = phiInfos[assignmentIndex];
        if (assignment.state > 0)
        {
            auto& dstParam = assignment.param;

            // The assignment to `dstParam` is blocked because there
            // exist one or more other not-yet-completed assignments
            // where `dstParam` is being used as a source.
            //
            // We emit a `load` from the temporary for `dstParam` in
            // order to make a copy, at which point we can safely
            // perform the assignment for to the original, and allow
            // the not-yet-completed assignments to use that copy
            // instead, as the current value for `dstParam`.
            //
            dstParam.currentVal = m_builder.emitLoad(dstParam.temp);
            assignment.state = kState_Ready;
        }

        // If this assignment *was* blocked, we have made it so
        // that it isn't blocked any more. We thus expect that
        // trying to perform the assignment again will definitely
        // result it the assignment being in the _done_ state.
        //
        tryPerformParamAssignment(assignmentIndex);
        SLANG_ASSERT(assignment.state == kState_Done);
    }
};

void eliminatePhis(LivenessMode livenessMode, IRModule* module, PhiEliminationOptions options)
{
    PhiEliminationContext context(livenessMode, module, options);
    context.eliminatePhisInModule();
}

void eliminatePhisInFunc(
    LivenessMode livenessMode,
    IRModule* module,
    IRGlobalValueWithCode* func,
    PhiEliminationOptions options)
{
    PhiEliminationContext context(livenessMode, module, options);
    context.eliminatePhisInFunc(func);
}

} // namespace Slang
