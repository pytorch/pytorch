#include "slang-ir-strip-debug-info.h"

#include "slang-ir-insts.h"

namespace Slang
{
static void findDebugInfo(IRInst* inst, List<IRInst*>& debugInstructions)
{
    switch (inst->getOp())
    {
    case kIROp_DebugValue:
    case kIROp_DebugVar:
    case kIROp_DebugLine:
    case kIROp_DebugLocationDecoration:
    case kIROp_DebugSource:
        debugInstructions.add(inst);
        break;
    default:
        break;
    }

    for (auto child : inst->getChildren())
        findDebugInfo(child, debugInstructions);
}

void stripDebugInfo(IRModule* irModule)
{
    List<IRInst*> debugInstructions;
    findDebugInfo(irModule->getModuleInst(), debugInstructions);
    for (auto debugInst : debugInstructions)
        debugInst->removeAndDeallocate();
}
} // namespace Slang
