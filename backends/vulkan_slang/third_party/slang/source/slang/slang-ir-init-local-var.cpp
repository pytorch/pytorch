// slang-ir-init-local-var.cpp
#include "slang-ir-init-local-var.h"

#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{

void initializeLocalVariables(IRModule* module, IRGlobalValueWithCode* func)
{
    IRBuilder builder(module);
    InstHashSet userSet(module);
    for (auto block : func->getBlocks())
    {
        for (auto inst : block->getChildren())
        {
            if (inst->getOp() == kIROp_Var)
            {
                bool initialized = false;
                userSet.clear();
                for (auto use = inst->firstUse; use; use = use->nextUse)
                    userSet.add(use->getUser());

                // Check if the variable is initialized in the same block.
                for (auto nextInst = inst->next; nextInst; nextInst = nextInst->next)
                {
                    switch (nextInst->getOp())
                    {
                    case kIROp_Store:
                        if (nextInst->getOperand(0) == inst)
                            initialized = true;
                        break;
                    case kIROp_GetElementPtr:
                    case kIROp_FieldAddress:
                        continue;
                    default:
                        if (userSet.contains(nextInst))
                        {
                            // We encountered a user of the variable before it was initialized.
                            // Break out of the loop and insert the initialization code.
                            goto breakLabel;
                        }
                    }
                    if (initialized)
                        break;
                }
            breakLabel:;
                if (initialized)
                    continue;

                IRBuilderSourceLocRAII sourceLocationScope(&builder, inst->sourceLoc);

                builder.setInsertAfter(inst);
                builder.emitStore(
                    inst,
                    builder.emitDefaultConstruct(
                        as<IRPtrTypeBase>(inst->getFullType())->getValueType()));
            }
        }
    }
}

} // namespace Slang
