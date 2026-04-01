#include "slang-ir-address-analysis.h"

#include "slang-ir-insts.h"
#include "slang-ir-util.h"

namespace Slang
{
void moveInstToEarliestPoint(IRDominatorTree* domTree, IRGlobalValueWithCode* func, IRInst* inst)
{
    if (!as<IRBlock>(inst->getParent()))
        return;
    if (domTree->isUnreachable(as<IRBlock>(inst->getParent())))
        return;

    InstWorkList blocks(func->getModule());
    InstHashSet operandInsts(func->getModule());
    for (UInt i = 0; i < inst->getOperandCount(); i++)
    {
        operandInsts.add(inst->getOperand(i));
        auto parentBlock = as<IRBlock>(inst->getOperand(i)->getParent());
        if (parentBlock)
        {
            if (!domTree->isUnreachable(parentBlock))
                blocks.add(parentBlock);
        }
    }
    {
        operandInsts.add(inst->getFullType());
        auto parentBlock = as<IRBlock>(inst->getFullType()->getParent());
        if (parentBlock)
        {
            if (!domTree->isUnreachable(parentBlock))
                blocks.add(parentBlock);
        }
    }
    // Find earliest block that is dominated by all operand blocks.
    IRBlock* earliestBlock = as<IRBlock>(inst->getParent());
    for (auto block : func->getBlocks())
    {
        bool dominated = true;
        for (auto opBlock : blocks)
        {
            if (!domTree->dominates(opBlock, block))
            {
                dominated = false;
                break;
            }
        }
        if (dominated)
        {
            earliestBlock = block;
            break;
        }
    }

    if (!earliestBlock)
        return;

    IRInst* latestOperand = nullptr;
    for (auto childInst : earliestBlock->getChildren())
    {
        if (operandInsts.contains(childInst))
        {
            latestOperand = childInst;
        }
    }

    if (!latestOperand || as<IRParam, IRDynamicCastBehavior::NoUnwrap>(latestOperand))
        inst->insertBefore(earliestBlock->getFirstOrdinaryInst());
    else
        inst->insertAfter(latestOperand);
}

AddressAccessInfo analyzeAddressUse(IRDominatorTree* dom, IRGlobalValueWithCode* func)
{
    DeduplicateContext deduplicateContext;

    AddressAccessInfo info;

    // Deduplicate and move known address insts.
    for (auto block : func->getBlocks())
    {
        for (auto inst : block->getModifiableChildren())
        {
            switch (inst->getOp())
            {
            case kIROp_Var:
                {
                    RefPtr<AddressInfo> addrInfo = new AddressInfo();
                    addrInfo->addrInst = inst;
                    addrInfo->isConstant = true;
                    addrInfo->parentAddress = nullptr;
                    info.addressInfos[inst] = addrInfo;
                }
                break;
            case kIROp_Param:
                if (as<IRPtrTypeBase>(inst->getFullType()))
                {
                    RefPtr<AddressInfo> addrInfo = new AddressInfo();
                    addrInfo->addrInst = inst;
                    addrInfo->isConstant = (block == func->getFirstBlock() ? true : false);
                    addrInfo->parentAddress = nullptr;
                    info.addressInfos[inst] = addrInfo;
                }
                break;
            case kIROp_GetElementPtr:
            case kIROp_FieldAddress:
                {
                    moveInstToEarliestPoint(dom, func, inst->getFullType());
                    moveInstToEarliestPoint(dom, func, inst);
                    auto deduplicated = deduplicateContext.deduplicate(
                        inst,
                        [func](IRInst* inst)
                        {
                            if (!inst->getParent())
                                return false;
                            if (inst->getParent()->getParent() != func)
                                return false;
                            switch (inst->getOp())
                            {
                            case kIROp_GetElementPtr:
                            case kIROp_FieldAddress:
                                return true;
                            default:
                                return false;
                            }
                        });

                    if (deduplicated != inst)
                    {
                        SLANG_RELEASE_ASSERT(dom->dominates(
                            as<IRBlock>(deduplicated->getParent()),
                            as<IRBlock>(inst->getParent())));

                        inst->replaceUsesWith(deduplicated);
                        inst->removeAndDeallocate();
                    }
                    else
                    {
                        RefPtr<AddressInfo> addrInfo = new AddressInfo();
                        addrInfo->addrInst = inst;
                        if (inst->getOp() == kIROp_FieldAddress)
                        {
                            addrInfo->isConstant = true;
                        }
                        else
                        {
                            addrInfo->isConstant =
                                as<IRConstant>(inst->getOperand(1)) ? true : false;
                        }
                        info.addressInfos[inst] = addrInfo;
                    }
                }
                break;
            }
        }
    }

    // Construct address info tree.
    for (auto& addr : info.addressInfos)
    {
        RefPtr<AddressInfo> parentInfo;
        if (addr.value->addrInst->getOperandCount() > 1 &&
            info.addressInfos.tryGetValue(addr.value->addrInst->getOperand(0), parentInfo))
        {
            addr.value->parentAddress = parentInfo;
            parentInfo->children.add(addr.value);
            if (!parentInfo->isConstant)
                addr.value->isConstant = false;
        }
    }
    return info;
}
} // namespace Slang
