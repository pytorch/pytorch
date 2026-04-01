#include "slang-ir-composite-reg-to-mem.h"

#include "slang-ir-dce.h"
#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{
struct RegisterReplacementWorkItem
{
    IRInst* ssaValue;
    IRInst* addr;
    IRInst* initialStore;
};

void replaceRegisterUseWithAddrUse(
    List<RegisterReplacementWorkItem>& workList,
    IRInst* ssaValue,
    IRInst* addr,
    IRInst* initialStore)
{
    IRBuilder builder(ssaValue);
    traverseUses(
        ssaValue,
        [&](IRUse* use)
        {
            auto user = use->getUser();
            if (user == initialStore)
                return;
            builder.setInsertBefore(user);
            IRInst* newAddr = nullptr;
            // If the user is itself a getElement/getField inst,
            // we want to follow that chain and recursively replace
            // their users.
            if (auto getElementUser = as<IRGetElement>(user))
            {
                if (getElementUser->getOperands() == use)
                {
                    newAddr = builder.emitElementAddress(addr, getElementUser->getIndex());
                }
            }
            else if (auto getFieldUser = as<IRFieldExtract>(user))
            {
                if (getFieldUser->getOperands() == use)
                {
                    newAddr = builder.emitFieldAddress(
                        builder.getPtrType(user->getFullType()),
                        addr,
                        getFieldUser->getField());
                }
            }
            if (newAddr)
            {
                workList.add(RegisterReplacementWorkItem{user, newAddr, nullptr});
            }
            else
            {
                // For all other uses, we emit a load from addr and use it.
                auto val = builder.emitLoad(addr);
                builder.replaceOperand(use, val);
            }
        });
}

void replaceRegisterUseWithAddrUse(IRInst* ssaValue, IRInst* addr, IRInst* initialStore)
{
    List<RegisterReplacementWorkItem> workList, pendingWorkList;
    workList.add(RegisterReplacementWorkItem{ssaValue, addr, initialStore});

    while (workList.getCount())
    {
        for (auto item : workList)
        {
            replaceRegisterUseWithAddrUse(
                pendingWorkList,
                item.ssaValue,
                item.addr,
                item.initialStore);
        }
        workList.swapWith(pendingWorkList);
        pendingWorkList.clear();
    }
}

void convertCompositeTypeParametersToPointers(IRFunc* func)
{
    IRBuilder builder(func);
    List<UInt> compositeParamIds;
    UInt idx = 0;
    List<IRParam*> paramWorkList;
    if (!func->findDecoration<IREntryPointDecoration>())
    {
        // Only translate function parameters for non entry points.
        for (auto param : func->getParams())
        {
            if (as<IRArrayTypeBase>(param->getFullType()) || as<IRStructType>(param->getFullType()))
            {
                paramWorkList.add(param);
                compositeParamIds.add(idx);
            }
            idx++;
        }
    }
    for (auto param : paramWorkList)
    {
        // We have a composite type parameter, so we need to replace it with a pointer.
        //

        auto ptrCompositeType = builder.getPtrType(param->getFullType());
        auto newParam = builder.createParam(ptrCompositeType);
        newParam->insertBefore(param);
        replaceRegisterUseWithAddrUse(param, newParam, nullptr);
        param->removeAndDeallocate();
    }
    if (paramWorkList.getCount())
    {
        // The function is modified, we need to also update its type.
        List<IRType*> paramTypes;
        for (auto param : func->getParams())
        {
            paramTypes.add(param->getFullType());
        }
        auto newFuncType = builder.getFuncType(
            (UInt)paramTypes.getCount(),
            paramTypes.getBuffer(),
            func->getResultType());
        func->setFullType(newFuncType);

        // Update all the call sites to pass the composite by pointer.
        traverseUses(
            func,
            [&](IRUse* use)
            {
                if (const auto call = as<IRCall>(use->getUser()))
                {
                    builder.setInsertBefore(call);
                    for (auto paramId : compositeParamIds)
                    {
                        auto arg = call->getArg(paramId);
                        SLANG_ASSERT(as<IRPtrTypeBase>(paramTypes[paramId]));
                        auto var =
                            builder.emitVar(as<IRPtrTypeBase>(paramTypes[paramId])->getValueType());
                        builder.emitStore(var, arg);
                        call->setArg(paramId, var);
                    }
                }
            });
    }

    // Now work through all the local values and process uses of `Load(composite)`.
    for (auto block : func->getBlocks())
    {
        for (auto inst : block->getModifiableChildren())
        {
            if (!as<IRArrayTypeBase>(inst->getDataType()) && !as<IRStructType>(inst->getDataType()))
                continue;
            if (inst->getParent() != block)
                continue;
            IRInst* tempVar = nullptr;
            IRInst* initialStore = nullptr;
            builder.setInsertAfter(inst);
            switch (inst->getOp())
            {
            case kIROp_Load:
                {
                    auto ptr = inst->getOperand(0);
                    auto rootPtr = getRootAddr(ptr);
                    if (as<IRConstantBufferType>(rootPtr->getDataType()) ||
                        as<IRParameterBlockType>(rootPtr->getDataType()))
                    {
                        tempVar = ptr;
                    }
                    else
                    {
                        tempVar = builder.emitVar(inst->getFullType());
                        initialStore = builder.emitStore(tempVar, inst);
                    }
                    break;
                }
            case kIROp_Call:
                {
                    tempVar = builder.emitVar(inst->getFullType());
                    initialStore = builder.emitStore(tempVar, inst);
                    break;
                }
            default:
                break;
            }

            if (!tempVar)
                continue;
            replaceRegisterUseWithAddrUse(inst, tempVar, initialStore);
        }
    }
    eliminateDeadCode(func);
}

void convertCompositeTypeParametersToPointers(IRModule* module)
{
    for (auto inst : module->getGlobalInsts())
    {
        if (auto func = as<IRFunc>(inst))
        {
            convertCompositeTypeParametersToPointers(func);
        }
    }
}
} // namespace Slang
