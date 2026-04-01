// slang-ir-autodiff-propagate.h
#pragma once

#include "slang-compiler.h"
#include "slang-ir-autodiff.h"
#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{

inline bool isDifferentialInst(IRInst* inst)
{
    return inst->findDecoration<IRDifferentialInstDecoration>();
}

inline bool isPrimalInst(IRInst* inst)
{
    return inst->findDecoration<IRPrimalInstDecoration>() || (as<IRConstant>(inst) != nullptr);
}

inline bool isMixedDifferentialInst(IRInst* inst)
{
    return inst->findDecoration<IRMixedDifferentialInstDecoration>();
}

struct DiffPropagationPass : InstPassBase
{
    AutoDiffSharedContext* autodiffContext;

    DiffPropagationPass(AutoDiffSharedContext* autodiffContext)
        : autodiffContext(autodiffContext), InstPassBase(autodiffContext->moduleInst->getModule())
    {
    }


    bool shouldInstBeMarkedDifferential(IRInst* inst)
    {
        for (UIndex ii = 0; ii < inst->getOperandCount(); ii++)
        {
            if (isDifferentialInst(inst->getOperand(ii)))
            {
                return true;
            }
        }

        return false;
    }

    void addPendingUsersToWorkList(IRInst* inst)
    {
        auto use = inst->firstUse;
        while (use)
        {
            if (!isDifferentialInst(use->getUser()))
            {
                addToWorkList(use->getUser());
            }
            use = use->nextUse;
        }
    }

    // Propagate IRDifferentialInstDecoration for all children of instWithChildren.
    void propagateDiffInstDecoration(IRBuilder* builder, IRInst* instWithChildren)
    {
        List<IRInst*> initialList;
        // Mark 'GetDifferential' insts as differential.
        processChildInstsOfType<IRDifferentialPairGetDifferential>(
            kIROp_DifferentialPairGetDifferential,
            instWithChildren,
            [&](IRDifferentialPairGetDifferential* getDifferentialInst)
            {
                builder->markInstAsDifferential(getDifferentialInst);
                initialList.add(getDifferentialInst);
            });


        workList.clear();
        workListSet.clear();

        // Add the marked insts to the work list.
        for (auto inst : initialList)
        {
            // Look for insts marked as differential.
            if (isDifferentialInst(inst))
                addPendingUsersToWorkList(inst);
        }

        // Propagate to all users..
        while (workList.getCount() != 0)
        {
            IRInst* inst = pop();

            // Skip if this is already a differential inst.
            if (isDifferentialInst(inst))
            {
                continue;
            }

            if (shouldInstBeMarkedDifferential(inst))
            {
                builder->markInstAsDifferential(inst);
                addPendingUsersToWorkList(inst);
            }
        }
    }
};

} // namespace Slang
