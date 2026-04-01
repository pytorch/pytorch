#include "slang-ir-addr-inst-elimination.h"

#include "slang-ir-insts.h"
#include "slang-ir-util.h"

namespace Slang
{

// Rewrites address load/store into value extract/updates to allow SSA transform to apply to struct
// and array elements. For example,
//  load(elementPtr(arr, 1)) ==> elementExtract(load(arr), 1)
//  store(fieldAddr(s, field_key), val) ==> store(s, updateField(load(s), fieldKey, val))
// After this transform, all address operands of `load` and `store` insts will be either a var or a
// param.

struct AddressInstEliminationContext
{
    IRModule* module;
    DiagnosticSink* sink;

    IRInst* getValue(IRBuilder& builder, IRInst* addr)
    {
        switch (addr->getOp())
        {
        default:
            return builder.emitLoad(addr);
        case kIROp_GetElementPtr:
        case kIROp_FieldAddress:
            {
                IRInst* args[] = {getValue(builder, addr->getOperand(0)), addr->getOperand(1)};
                return builder.emitIntrinsicInst(
                    cast<IRPtrTypeBase>(addr->getFullType())->getValueType(),
                    (addr->getOp() == kIROp_GetElementPtr ? kIROp_GetElement : kIROp_FieldExtract),
                    2,
                    args);
            }
        }
    }

    void storeValue(IRBuilder& builder, IRInst* addr, IRInst* val)
    {
        List<IRInst*> accessChain;

        for (auto inst = addr; inst;)
        {
            switch (inst->getOp())
            {
            default:
                accessChain.add(inst);
                goto endLoop;
            case kIROp_GetElementPtr:
            case kIROp_FieldAddress:
                accessChain.add(inst->getOperand(1));
                inst = inst->getOperand(0);
                break;
            }
        }
    endLoop:;
        auto lastAddr = accessChain.getLast();
        accessChain.removeLast();
        accessChain.reverse();
        if (accessChain.getCount())
        {
            auto lastVal = builder.emitLoad(lastAddr);
            auto update = builder.emitUpdateElement(lastVal, accessChain.getArrayView(), val);
            builder.emitStore(lastAddr, update);
        }
        else
        {
            builder.emitStore(lastAddr, val);
        }
    }

    void transformLoadAddr(IRBuilder& builder, IRUse* use)
    {
        auto addr = use->get();
        auto load = as<IRLoad>(use->getUser());

        builder.setInsertBefore(use->getUser());
        auto value = getValue(builder, addr);
        load->replaceUsesWith(value);
        load->removeAndDeallocate();
    }

    void transformStoreAddr(IRBuilder& builder, IRUse* use)
    {
        auto addr = use->get();
        auto store = as<IRStore>(use->getUser());

        builder.setInsertBefore(use->getUser());
        storeValue(builder, addr, store->getVal());
        store->removeAndDeallocate();
    }

    void transformCallAddr(IRBuilder& builder, IRUse* use)
    {
        auto addr = use->get();
        auto call = as<IRCall>(use->getUser());

        // Don't change the use if addr is a non mutable address.
        if (as<IRConstRefType>(getRootAddr(addr)->getDataType()))
        {
            return;
        }

        builder.setInsertBefore(call);
        auto tempVar = builder.emitVar(cast<IRPtrTypeBase>(addr->getFullType())->getValueType());

        // Store the initial value of the mutable argument into temp var.
        // If this is an `out` var, the initial value will be undefined,
        // which will get cleaned up later into a `defaultConstruct`.
        builder.emitStore(tempVar, getValue(builder, addr));
        builder.setInsertAfter(call);
        storeValue(builder, addr, builder.emitLoad(tempVar));
        use->set(tempVar);
    }

    SlangResult eliminateAddressInstsImpl(IRFunc* func, DiagnosticSink* inSink)
    {
        sink = inSink;

        IRBuilder builder(func->getModule());

        List<IRInst*> workList;
        for (auto block : func->getBlocks())
        {
            for (auto inst : block->getChildren())
            {
                if (as<IRConstRefType>(getRootAddr(inst)->getDataType()))
                    continue;
                if (auto ptrType = as<IRPtrTypeBase>(inst->getDataType()))
                {
                    auto valType = unwrapAttributedType(ptrType->getValueType());
                    if (!getResolvedInstForDecorations(valType)
                             ->findDecoration<IRNonCopyableTypeDecoration>())
                    {
                        workList.add(inst);
                    }
                }
            }
        }

        for (Index workListIndex = 0; workListIndex < workList.getCount(); workListIndex++)
        {
            auto addrInst = workList[workListIndex];

            for (auto use = addrInst->firstUse; use;)
            {
                auto nextUse = use->nextUse;

                if (as<IRDecoration>(use->getUser()))
                {
                    use = nextUse;
                    continue;
                }

                IRBuilder transformBuilder(module);
                IRBuilderSourceLocRAII sourceLocationScope(
                    &transformBuilder,
                    use->getUser()->sourceLoc);

                switch (use->getUser()->getOp())
                {
                case kIROp_Load:
                    transformLoadAddr(transformBuilder, use);
                    break;
                case kIROp_Store:
                    transformStoreAddr(transformBuilder, use);
                    break;
                case kIROp_Call:
                    transformCallAddr(transformBuilder, use);
                    break;
                case kIROp_GetElementPtr:
                case kIROp_FieldAddress:
                case kIROp_Unmodified:
                case kIROp_DebugValue:
                case kIROp_GetOffsetPtr:
                    break;
                default:
                    sink->diagnose(
                        use->getUser()->sourceLoc,
                        Diagnostics::unsupportedUseOfLValueForAutoDiff);
                    break;
                }
                use = nextUse;
            }
        }

        return SLANG_OK;
    }
};

SlangResult eliminateAddressInsts(IRFunc* func, DiagnosticSink* sink)
{
    AddressInstEliminationContext ctx;
    ctx.module = func->getModule();
    ctx.sink = sink;
    return ctx.eliminateAddressInstsImpl(func, sink);
}

} // namespace Slang
