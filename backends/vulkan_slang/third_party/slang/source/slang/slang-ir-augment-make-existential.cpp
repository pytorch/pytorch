#include "slang-ir-augment-make-existential.h"

#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{
struct AugmentMakeExistentialContext
{
    IRModule* module;

    InstWorkList workList;
    InstHashSet workListSet;

    AugmentMakeExistentialContext(IRModule* inModule)
        : module(inModule), workList(inModule), workListSet(inModule)
    {
    }

    void addToWorkList(IRInst* inst)
    {
        if (workListSet.contains(inst))
            return;

        workList.add(inst);
        workListSet.add(inst);
    }

    void processMakeExistential(IRMakeExistential* inst)
    {
        IRBuilder builderStorage(module);
        auto builder = &builderStorage;
        builder->setInsertBefore(inst);

        auto augInst = builder->emitMakeExistentialWithRTTI(
            inst->getFullType(),
            inst->getWrappedValue(),
            inst->getWitnessTable(),
            inst->getWrappedValue()->getDataType());
        inst->replaceUsesWith(augInst);
        inst->removeAndDeallocate();
    }

    void processInst(IRInst* inst)
    {
        switch (inst->getOp())
        {
        case kIROp_MakeExistential:
            processMakeExistential((IRMakeExistential*)inst);
            break;
        default:
            break;
        }
    }

    void processModule()
    {
        addToWorkList(module->getModuleInst());

        while (workList.getCount() != 0)
        {
            IRInst* inst = workList.getLast();

            workList.removeLast();
            workListSet.remove(inst);

            processInst(inst);

            for (auto child = inst->getLastChild(); child; child = child->getPrevInst())
            {
                addToWorkList(child);
            }
        }
    }
};

void augmentMakeExistentialInsts(IRModule* module)
{
    AugmentMakeExistentialContext context(module);
    context.processModule();
}
} // namespace Slang
