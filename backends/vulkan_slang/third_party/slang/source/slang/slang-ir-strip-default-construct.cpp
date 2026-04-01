// slang-ir-strip-default-construct.cpp
#include "slang-ir-strip-default-construct.h"

#include "slang-ir-inst-pass-base.h"
#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{

struct RemoveDefaultConstructInsts : InstPassBase
{
    RemoveDefaultConstructInsts(IRModule* module)
        : InstPassBase(module)
    {
    }
    void processModule()
    {
        processInstsOfType<IRDefaultConstruct>(
            kIROp_DefaultConstruct,
            [&](IRDefaultConstruct* defaultConstruct)
            {
                List<IRInst*> instsToRemove;
                for (auto use = defaultConstruct->firstUse; use; use = use->nextUse)
                {
                    if (as<IRStore>(use->getUser()))
                        instsToRemove.add(use->getUser());
                    else
                        return; // Ignore this inst if there are non-store uses.
                }

                for (auto inst : instsToRemove)
                    inst->removeAndDeallocate();

                defaultConstruct->removeAndDeallocate();
            });
    }
};

void removeRawDefaultConstructors(IRModule* module)
{
    RemoveDefaultConstructInsts(module).processModule();
}

} // namespace Slang
