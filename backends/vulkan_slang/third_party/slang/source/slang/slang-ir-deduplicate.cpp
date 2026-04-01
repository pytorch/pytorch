#include "slang-ir-insts.h"

namespace Slang
{
void IRDeduplicationContext::init(IRModule* module)
{
    m_module = module;
    m_session = module->getSession();

    m_globalValueNumberingMap.clear();
    m_constantMap.clear();
}

void IRDeduplicationContext::removeHoistableInstFromGlobalNumberingMap(IRInst* instToRemove)
{
    InstHashSet userWorkListSet(instToRemove->getModule());
    InstWorkList userWorkList(instToRemove->getModule());
    auto addToWorkList = [&](IRInst* i)
    {
        if (userWorkListSet.add(i))
            userWorkList.add(i);
    };
    addToWorkList(instToRemove);
    for (Index i = 0; i < userWorkList.getCount(); i++)
    {
        auto inst = userWorkList[i];
        if (getIROpInfo(inst->getOp()).isHoistable())
        {
            _removeGlobalNumberingEntry(inst);
            for (auto use = inst->firstUse; use; use = use->nextUse)
            {
                addToWorkList(use->getUser());
            }
        }
    }
}

void IRDeduplicationContext::tryHoistInst(IRInst* inst)
{
    InstWorkList workList(inst->getModule());
    InstHashSet workListSet(inst->getModule());
    workList.add(inst);
    workListSet.add(inst);
    IRBuilder builder(inst->getModule());

    for (Index i = 0; i < workList.getCount(); i++)
    {
        auto item = workList[i];

        // Does inst no longer depend on anything defined locally?
        // If so we should hoist it.
        bool shouldHoist = false;
        for (UInt a = 0; a < item->getOperandCount(); a++)
        {
            auto opParent = item->getOperand(a)->getParent();
            if (opParent != item->getParent())
            {
                shouldHoist = true;
                break;
            }
        }

        // Hoisting this inst
        if (shouldHoist)
        {
            item->removeFromParent();
            addHoistableInst(&builder, item);

            // Continue to consider all users for hoisting.
            for (auto use = item->firstUse; use; use = use->nextUse)
            {
                if (getIROpInfo(use->getUser()->getOp()).isHoistable())
                {
                    if (workListSet.add(use->getUser()))
                        workList.add(use->getUser());
                }
            }
        }
    }
}
} // namespace Slang
