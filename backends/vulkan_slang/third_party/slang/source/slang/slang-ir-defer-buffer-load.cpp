#include "slang-ir-defer-buffer-load.h"

#include "slang-ir-clone.h"
#include "slang-ir-dominators.h"
#include "slang-ir-insts.h"
#include "slang-ir-redundancy-removal.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{
struct DeferBufferLoadContext
{
    // Map an original SSA value to a pointer that can be used to load the value.
    Dictionary<IRInst*, IRInst*> mapValueToPtr;

    // Map an ptr to its loaded value.
    Dictionary<IRInst*, IRInst*> mapPtrToValue;

    IRFunc* currentFunc = nullptr;
    IRDominatorTree* dominatorTree = nullptr;

    // Ensure that for an original SSA value, we have formed a pointer that can be used to load the
    // value.
    IRInst* ensurePtr(IRInst* valueInst)
    {
        IRInst* result = nullptr;
        if (mapValueToPtr.tryGetValue(valueInst, result))
            return result;

        IRBuilder b(valueInst);
        b.setInsertBefore(valueInst);

        switch (valueInst->getOp())
        {
        case kIROp_StructuredBufferLoad:
        case kIROp_StructuredBufferLoadStatus:
            {
                result = b.emitRWStructuredBufferGetElementPtr(
                    valueInst->getOperand(0),
                    valueInst->getOperand(1));
                break;
            }
        case kIROp_GetElement:
            {
                auto ptr = ensurePtr(valueInst->getOperand(0));
                if (!ptr)
                    return nullptr;
                result = b.emitElementAddress(ptr, valueInst->getOperand(1));
                break;
            }
        case kIROp_FieldExtract:
            {
                auto ptr = ensurePtr(valueInst->getOperand(0));
                if (!ptr)
                    return nullptr;
                result = b.emitFieldAddress(ptr, valueInst->getOperand(1));
                break;
            }
        }
        if (result)
        {
            mapValueToPtr[valueInst] = result;
        }
        return result;
    }

    static bool isStructuredBufferLoad(IRInst* inst)
    {
        // Note: we cannot defer loads from RWStructuredBuffer because there can be other
        // instructions that modify the buffer.
        switch (inst->getOp())
        {
        case kIROp_StructuredBufferLoad:
        case kIROp_StructuredBufferLoadStatus:
            return true;
        default:
            return false;
        }
    }

    // Ensure that for a pointer value, we have created a load instruction to materialize the value.
    IRInst* materializePointer(IRBuilder& builder, IRInst* loadInst)
    {
        auto ptr = ensurePtr(loadInst);
        if (!ptr)
            return nullptr;
        IRInst* result = nullptr;
        if (mapPtrToValue.tryGetValue(ptr, result))
            return result;
        builder.setInsertAfter(ptr);
        result = builder.emitLoad(ptr);
        mapPtrToValue[ptr] = result;
        return result;
    }

    static bool isSimpleType(IRInst* type)
    {
        if (as<IRBasicType>(type))
            return true;
        if (as<IRVectorType>(type))
            return true;
        if (as<IRMatrixType>(type))
            return true;
        return false;
    }

    void deferBufferLoadInst(IRBuilder& builder, List<IRInst*>& workList, IRInst* loadInst)
    {
        // Don't defer the load anymore if the type is simple.
        if (isSimpleType(loadInst->getDataType()))
        {
            if (!isStructuredBufferLoad(loadInst))
            {
                auto materializedVal = materializePointer(builder, loadInst);
                loadInst->replaceUsesWith(materializedVal);
            }
            return;
        }

        // Otherwise, look for all uses and try to defer the load before actual use of the value.
        ShortList<IRInst*> pendingWorkList;
        bool needMaterialize = false;
        traverseUses(
            loadInst,
            [&](IRUse* use)
            {
                if (needMaterialize)
                    return;

                auto user = use->getUser();
                switch (user->getOp())
                {
                case kIROp_GetElement:
                case kIROp_FieldExtract:
                    {
                        auto basePtr = ensurePtr(loadInst);
                        if (!basePtr)
                            return;
                        pendingWorkList.add(user);
                    }
                    break;
                default:
                    if (!isStructuredBufferLoad(loadInst))
                    {
                        needMaterialize = true;
                        return;
                    }
                    break;
                }
            });

        if (needMaterialize)
        {
            auto val = materializePointer(builder, loadInst);
            loadInst->replaceUsesWith(val);
            loadInst->removeAndDeallocate();
        }
        else
        {
            // Append to worklist in reverse order so we process the uses in natural appearance
            // order.
            for (Index i = pendingWorkList.getCount() - 1; i >= 0; i--)
                workList.add(pendingWorkList[i]);
        }
    }

    void deferBufferLoadInFunc(IRFunc* func)
    {
        removeRedundancyInFunc(func, false);

        currentFunc = func;
        dominatorTree = func->getModule()->findOrCreateDominatorTree(func);

        List<IRInst*> workList;

        for (auto block : func->getBlocks())
        {
            for (auto inst : block->getChildren())
            {
                if (isStructuredBufferLoad(inst))
                {
                    workList.add(inst);
                }
            }
        }

        IRBuilder builder(func);
        for (Index i = 0; i < workList.getCount(); i++)
        {
            auto inst = workList[i];
            deferBufferLoadInst(builder, workList, inst);
        }
    }

    void deferBufferLoad(IRGlobalValueWithCode* inst)
    {
        if (auto func = as<IRFunc>(inst))
        {
            deferBufferLoadInFunc(func);
        }
        else if (auto generic = as<IRGeneric>(inst))
        {
            auto inner = findGenericReturnVal(generic);
            if (auto innerFunc = as<IRFunc>(inner))
                deferBufferLoadInFunc(innerFunc);
        }
    }
};

void deferBufferLoad(IRModule* module)
{
    DeferBufferLoadContext context;
    for (auto childInst : module->getGlobalInsts())
    {
        if (auto code = as<IRGlobalValueWithCode>(childInst))
        {
            context.deferBufferLoad(code);
        }
    }
}

} // namespace Slang
