#include "slang-ir-resolve-varying-input-ref.h"

#include "slang-ir-insts.h"
#include "slang-ir-util.h"

namespace Slang
{
void resolveVaryingInputRef(IRFunc* func)
{
    List<IRInst*> toRemove;
    for (auto bb = func->getFirstBlock(); bb; bb = bb->getNextBlock())
    {
        for (auto inst : bb->getChildren())
        {
            switch (inst->getOp())
            {
            case kIROp_ResolveVaryingInputRef:
                {
                    // Resolve a reference to varying input to the actual global param
                    // representing the varying input.
                    auto operand = inst->getOperand(0);
                    List<IRInst*> accessChain;
                    List<IRInst*> types;
                    auto rootAddr = getRootAddr(operand, accessChain, &types);
                    if (rootAddr->getOp() == kIROp_Param || rootAddr->getOp() == kIROp_GlobalParam)
                    {
                        // If the referred operand is already a global param, use it directly.
                        inst->replaceUsesWith(operand);
                        toRemove.add(inst);
                        break;
                    }
                    // If the referred operand is a local var,
                    // and there is a store(var, load(globalParam)),
                    // replace `inst` with `globalParam`.
                    IRInst* srcPtr = nullptr;
                    for (auto use = rootAddr->firstUse; use; use = use->nextUse)
                    {
                        auto user = use->getUser();
                        if (auto store = as<IRStore>(user))
                        {
                            if (store->getPtrUse() == use)
                            {
                                if (auto load = as<IRLoad>(store->getVal()))
                                {
                                    auto ptr = load->getPtr();
                                    if (ptr->getOp() == kIROp_Param ||
                                        ptr->getOp() == kIROp_GlobalParam)
                                    {
                                        if (!srcPtr)
                                            srcPtr = ptr;
                                        else
                                        {
                                            srcPtr = nullptr;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (srcPtr)
                    {
                        IRBuilder builder(inst);
                        builder.setInsertBefore(inst);
                        auto resolvedPtr = builder.emitElementAddress(
                            srcPtr,
                            accessChain.getArrayView(),
                            types.getArrayView());
                        inst->replaceUsesWith(resolvedPtr);
                        toRemove.add(inst);
                    }
                }
                break;
            }
        }
    }
    for (auto inst : toRemove)
    {
        inst->removeAndDeallocate();
    }
}

void resolveVaryingInputRef(IRModule* module)
{
    for (auto globalInst : module->getGlobalInsts())
    {
        if (globalInst->findDecoration<IREntryPointDecoration>())
            resolveVaryingInputRef((IRFunc*)globalInst);
    }
}

} // namespace Slang
