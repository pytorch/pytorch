// slang-ir-ssa-register-allocate.cpp
#include "slang-ir-ssa-register-allocate.h"

#include "slang-ir-dominators.h"
#include "slang-ir-insts.h"
#include "slang-ir-reachability.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{

struct RegisterAllocateContext
{
    OrderedDictionary<IRType*, List<RefPtr<RegisterInfo>>> mapTypeToRegisterList;
    bool allocateForCompositeTypeOnly;

    RegisterAllocateContext(bool compositeTypeOnly)
        : allocateForCompositeTypeOnly(compositeTypeOnly)
    {
    }

    List<RefPtr<RegisterInfo>>& getRegisterListForType(IRType* type)
    {
        if (auto list = mapTypeToRegisterList.tryGetValue(type))
        {
            return *list;
        }
        mapTypeToRegisterList[type] = List<RefPtr<RegisterInfo>>();
        return mapTypeToRegisterList[type].getValue();
    }

    void assignInstToNewRegister(List<RefPtr<RegisterInfo>>& regList, IRInst* inst)
    {
        auto reg = new RegisterInfo();
        reg->type = inst->getFullType();
        reg->insts.add(inst);
        regList.add(reg);
    }

    bool areInstsPreferredToBeCoalescedImpl(IRInst* inst0, IRInst* inst1)
    {
        switch (inst1->getOp())
        {
        case kIROp_UpdateElement:
            if (inst0 == inst1->getOperand(0))
                return true;
            break;
        default:
            break;
        }

        // If insts have the same name, prefer to coalesce them.
        auto name1 = inst0->findDecoration<IRNameHintDecoration>();
        auto name2 = inst1->findDecoration<IRNameHintDecoration>();
        if (name1 && name2 && name1->getName() == name2->getName())
            return true;

        return false;
    }
    bool areInstsPreferredToBeCoalesced(IRInst* inst0, IRInst* inst1)
    {
        return areInstsPreferredToBeCoalescedImpl(inst0, inst1) ||
               areInstsPreferredToBeCoalescedImpl(inst1, inst0);
    }

    bool isRegisterPreferred(
        RegisterInfo* existingRegister,
        RegisterInfo* newRegister,
        IRInst* inst)
    {
        int preferredCountExistingReg = 0;
        int preferredCountNewReg = 0;
        for (auto existingInst : existingRegister->insts)
        {
            if (areInstsPreferredToBeCoalesced(existingInst, inst))
                preferredCountExistingReg++;
        }
        for (auto existingInst : newRegister->insts)
        {
            if (areInstsPreferredToBeCoalesced(existingInst, inst))
                preferredCountNewReg++;
        }
        return preferredCountNewReg > preferredCountExistingReg;
    }

    bool canCoalesce(IRInst* inst1, IRInst* inst2)
    {
        // If two insts are Phis from the same block, don't coalesce.
        // This logic should not be needed in most cases because params from the same block should
        // always interfere anyways. However if a param is never used for for
        // some reason not DCE'd out, we don't want it to share the same register as another
        // param to avoid problems during phi elimination.
        if (inst1->getParent() == inst2->getParent() && inst1->getOp() == kIROp_Param &&
            inst2->getOp() == kIROp_Param)
            return false;

        // If two insts are coming from two separate user defined names, don't coalesce them into
        // the same register.
        auto name1 = inst1->findDecoration<IRNameHintDecoration>();
        auto name2 = inst2->findDecoration<IRNameHintDecoration>();

        if (name1 && !name2 || !name1 && name2)
            return true;

        if (!name1 || !name2)
            return true;
        if (name1->getName() != name2->getName())
            return false;

        return true;
    }

    bool isUseOfParamAfterPhiAssignment(
        IRDominatorTree* dom,
        IRUse* useToTest,
        IRInst* phiParam,
        IRInst* phiArg)
    {
        IRParam* param = as<IRParam, IRDynamicCastBehavior::NoUnwrap>(phiParam);
        if (!param)
            return false;
        IRUse* branchUse = nullptr;
        for (auto use = phiArg->firstUse; use; use = use->nextUse)
        {
            if (use->getUser()->getOp() == kIROp_unconditionalBranch)
            {
                if (!branchUse)
                    branchUse = use;
                else
                {
                    // If arg is being used in more than one branch, don't handle it.
                    return false;
                }
            }
        }
        if (!branchUse)
            return false;
        auto branch = as<IRUnconditionalBranch>(branchUse->getUser());
        auto branchTarget = branch->getTargetBlock();

        if (param->getParent() != branchTarget)
            return false;
        auto paramIndex = getParamIndexInBlock(param);
        if (paramIndex >= (int)branch->getArgCount() || paramIndex == -1)
            return false;
        if (branch->getArg(paramIndex) != phiArg)
            return false;

        // If we reach here, then phiArg is indeed used as arg for phiParam.
        // We will allow any use of phiParam when phiArg isn't live.
        if (dom->dominates(phiArg, useToTest->getUser()))
            return false;
        return true;
    }

    RegisterAllocationResult allocateRegisters(
        IRGlobalValueWithCode* func,
        RefPtr<IRDominatorTree>& inOutDom)
    {
        ReachabilityContext reachabilityContext(func);
        mapTypeToRegisterList.clear();

        auto dom = computeDominatorTree(func);
        inOutDom = dom;

        // Note that if inst A does not dominate inst B, then A can't be alive at B.
        // Therefore we only need to test interference against insts that dominates the
        // current inst.
        //
        // We can visit the dominance tree in pre-order and assign insts to registers.
        // This order allows us to easily track what is dominating the current inst.

        // We track the insts dominating the current location in a stack.
        InstWorkList dominatingInsts(func->getModule());
        InstHashSet dominatingInstSet(func->getModule());

        struct WorkStackItem
        {
            IRBlock* block;
            Index dominatingInstCount;
            WorkStackItem() = default;
            WorkStackItem(IRBlock* inBlock, Index inDominatingInstCount)
            {
                block = inBlock;
                dominatingInstCount = inDominatingInstCount;
            }
        };
        List<WorkStackItem> workStack;
        workStack.add(WorkStackItem(func->getFirstBlock(), 0));

        while (workStack.getCount())
        {
            auto item = workStack.getLast();
            workStack.removeLast();

            // Pop dominatingInst stack to correct location.
            for (Index i = item.dominatingInstCount; i < dominatingInsts.getCount(); i++)
                dominatingInstSet.remove(dominatingInsts[i]);
            dominatingInsts.setCount(item.dominatingInstCount);

            for (auto inst : item.block->getChildren())
            {
                if (!instNeedsProcessing(func, inst))
                    continue;
                // This is an inst we need to allocate register for.
                // Find register list for this type.
                auto& registers = getRegisterListForType(inst->getFullType());
                RegisterInfo* allocatedReg = nullptr;
                for (auto reg : registers)
                {
                    // Can we assign inst to this reg?
                    // We answer this by checking if any insts already assigned
                    // to this register is alive. If none are alive we can assign
                    // the register.
                    bool hasInterference = false;
                    for (auto existingInst : reg->insts)
                    {
                        // If `existingInst` does not dominate `inst`, it
                        // can't be alive here and during the entire life-time of the `inst`.
                        // This means that `inst` and `existingInst` won't interfere.
                        if (!dominatingInstSet.contains(existingInst))
                            continue;

                        // In the general case, we need to check all its use
                        // sites U to see if there is a path from `inst` to U.
                        // The idea is that is `existingInst` is never used
                        // anywhere after `inst`, then its lifetime ended before
                        // `inst` is defined, so it is still fine to place them
                        // in the same register.
                        for (auto use = existingInst->firstUse; use; use = use->nextUse)
                        {
                            if (use->getUser() == inst)
                                continue;

                            if (!canCoalesce(existingInst, inst) ||
                                reachabilityContext.isInstReachable(inst, use->getUser()))
                            {
                                // We found a use of `existingInst` (U) where
                                // there is a path from `inst` to U.
                                // Generally this means that existingInst and inst interfere.
                                // However, an exception is that existingInst is a PhiParam,
                                // and inst is an arg to that param, and use happens after
                                // the phi assignment.
                                if (isUseOfParamAfterPhiAssignment(dom, use, existingInst, inst))
                                    continue;
                                hasInterference = true;
                                goto endRegInstCheck;
                            }
                        }
                    }
                endRegInstCheck:;
                    if (!hasInterference)
                    {
                        if (!allocatedReg || isRegisterPreferred(allocatedReg, reg, inst))
                        {
                            allocatedReg = reg;
                        }
                    }
                }
                if (!allocatedReg)
                {
                    assignInstToNewRegister(registers, inst);
                }
                else
                {
                    allocatedReg->insts.add(inst);
                }
                dominatingInsts.add(inst);
                dominatingInstSet.add(inst);
            }

            // Recursively visit idom children.
            for (auto idomChild : dom->getImmediatelyDominatedBlocks(item.block))
            {
                workStack.add(WorkStackItem(idomChild, dominatingInsts.getCount()));
            }
        }

        RegisterAllocationResult result;
        result.mapTypeToRegisterList = _Move(mapTypeToRegisterList);
        for (auto& regList : result.mapTypeToRegisterList)
        {
            for (auto reg : regList.value)
            {
                for (auto inst : reg->insts)
                {
                    result.mapInstToRegister[inst] = reg;
                }
            }
        }
        return result;
    }

    bool instNeedsProcessing(IRGlobalValueWithCode* func, IRInst* inst)
    {
        switch (inst->getOp())
        {
        case kIROp_Param:
            if (inst->getParent() == func->getFirstBlock())
                return false;
            if (allocateForCompositeTypeOnly && !isCompositeType(inst->getFullType()))
                return false;
            return true;
        case kIROp_UpdateElement:
            return true;
        default:
            return false;
        }
    }
    bool needProcessing(IRGlobalValueWithCode* func)
    {
        for (auto block : func->getBlocks())
        {
            for (auto inst : block->getChildren())
            {
                if (instNeedsProcessing(func, inst))
                    return true;
            }
        }
        return false;
    }
};

RegisterAllocationResult allocateRegistersForFunc(
    IRGlobalValueWithCode* func,
    RefPtr<IRDominatorTree>& inOutDom,
    bool allocateForCompositeTypeOnly)
{
    RegisterAllocateContext context(allocateForCompositeTypeOnly);
    if (context.needProcessing(func))
        return context.allocateRegisters(func, inOutDom);
    return RegisterAllocationResult();
}

} // namespace Slang
