#include "slang-ir-remove-unused-generic-param.h"

#include "slang-ir-inst-pass-base.h"
#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{
struct RemoveUnusedGenericParamContext : InstPassBase
{
    RemoveUnusedGenericParamContext(IRModule* inModule)
        : InstPassBase(inModule)
    {
    }

    bool processModule()
    {
        IRBuilder builder(module);
        bool changed = false;
        for (auto inst : module->getModuleInst()->getChildren())
        {
            if (auto genInst = as<IRGeneric>(inst))
            {
                auto returnVal = findGenericReturnVal(genInst);
                switch (returnVal->getOp())
                {
                case kIROp_StructType:
                case kIROp_ClassType:
                    break;
                case kIROp_Func:
                case kIROp_FuncType:
                default:
                    // Don't simplify functions since this can break signature compatiblity with
                    // the interface. For example, if we have interface IFoo { void
                    // genFunc<T>(int x); } We can't simplify this by removing `T` even when the
                    // function type here does not depend on T.
                    continue;
                }
                if (returnVal->findDecoration<IRTargetIntrinsicDecoration>())
                    continue;

                List<UInt> paramToPreserve;
                UInt id = 0;
                List<IRInst*> paramsToRemove;
                for (auto param : genInst->getParams())
                {
                    if (param->hasUses())
                    {
                        paramToPreserve.add(id);
                    }
                    else
                    {
                        paramsToRemove.add(param);
                    }
                    id++;
                }
                if (paramsToRemove.getCount() == 0)
                    continue;
                changed = true;
                if (paramToPreserve.getCount() == 0)
                {
                    // Special case: the generic return value is not dependent on the generic param,
                    // we can hoist to global scope safely.
                    for (auto child = genInst->getFirstBlock()->getFirstOrdinaryInst(); child;)
                    {
                        auto next = child->getNextInst();
                        if (child->getOp() == kIROp_Return)
                        {
                            break;
                        }
                        child->insertBefore(genInst);
                        child = next;
                    }
                    SLANG_ASSERT(returnVal);
                    List<IRUse*> uses;
                    for (auto use = genInst->firstUse; use; use = use->nextUse)
                        uses.add(use);
                    for (auto use : uses)
                    {
                        if (use->getUser()->getOp() == kIROp_Specialize &&
                            use == use->getUser()->getOperands())
                        {
                            use->getUser()->replaceUsesWith(returnVal);
                        }
                    }
                    genInst->replaceUsesWith(returnVal);
                    genInst->removeAndDeallocate();
                }
                else
                {
                    // General case: remove unnecessary specialization arguments.
                    // Disabled this optimization for now since we still need to take care
                    // of the type of the generic, or change other passes to not
                    // use type info on a generic at all.
                    List<IRUse*> uses;
                    for (auto use = genInst->firstUse; use; use = use->nextUse)
                        uses.add(use);
                    for (auto use : uses)
                    {
                        if (use->getUser()->getOp() == kIROp_Specialize &&
                            use == use->getUser()->getOperands())
                        {
                            auto specialize = as<IRSpecialize>(use->getUser());
                            builder.setInsertBefore(specialize);
                            List<IRInst*> newArgs;
                            for (auto i : paramToPreserve)
                                newArgs.add(specialize->getArg(i));
                            auto newSpecialize = builder.emitSpecializeInst(
                                specialize->getFullType(),
                                specialize->getBase(),
                                newArgs.getCount(),
                                newArgs.getBuffer());
                            specialize->transferDecorationsTo(newSpecialize);
                            specialize->replaceUsesWith(newSpecialize);
                            specialize->removeAndDeallocate();
                        }
                    }
                    for (auto param : paramsToRemove)
                        param->removeAndDeallocate();
                }
            }
        }
        return changed;
    }
};

bool removeUnusedGenericParam(IRModule* module)
{
    RemoveUnusedGenericParamContext context = RemoveUnusedGenericParamContext(module);
    return context.processModule();
}

} // namespace Slang
