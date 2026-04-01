#include "slang-ir-fix-entrypoint-callsite.h"

#include "slang-ir-clone.h"
#include "slang-ir-insts.h"
#include "slang-ir-util.h"

namespace Slang
{
// If the entrypoint is called by some other function, we need to clone the
// entrypoint and replace the callsites to call the cloned entrypoint instead.
// This is because we will be modifying the signature of the entrypoint during
// entrypoint legalization to rewrite the way system values are passed in.
// By replacing the callsites to call the cloned entrypoint that act as ordinary
// functions, we will no longer need to worry about changing the callsites when we
// legalize the entry-points.
//
void fixEntryPointCallsites(IRFunc* entryPoint)
{
    IRFunc* clonedEntryPointForCall = nullptr;
    auto ensureClonedEntryPointForCall = [&]() -> IRFunc*
    {
        if (clonedEntryPointForCall)
            return clonedEntryPointForCall;
        IRCloneEnv cloneEnv;
        IRBuilder builder(entryPoint);
        builder.setInsertBefore(entryPoint);
        clonedEntryPointForCall = (IRFunc*)cloneInst(&cloneEnv, &builder, entryPoint);
        // Remove entrypoint and linkage decorations from the cloned callee.
        List<IRInst*> decorsToRemove;
        for (auto decor : clonedEntryPointForCall->getDecorations())
        {
            switch (decor->getOp())
            {
            case kIROp_ExportDecoration:
            case kIROp_UserExternDecoration:
            case kIROp_HLSLExportDecoration:
            case kIROp_EntryPointDecoration:
            case kIROp_LayoutDecoration:
            case kIROp_NumThreadsDecoration:
            case kIROp_ImportDecoration:
            case kIROp_ExternCDecoration:
            case kIROp_ExternCppDecoration:
                decorsToRemove.add(decor);
                break;
            }
        }
        for (auto decor : decorsToRemove)
            decor->removeAndDeallocate();
        return clonedEntryPointForCall;
    };
    traverseUses(
        entryPoint,
        [&](IRUse* use)
        {
            auto user = use->getUser();
            auto call = as<IRCall>(user);
            if (!call)
                return;
            auto callee = ensureClonedEntryPointForCall();
            call->setOperand(0, callee);

            // Fix up argument types: if the callee entrypoint is expecting a constref
            // and the caller is passing a value, we need to wrap the value in a temporary var
            // and pass the temporary var.
            //
            auto funcType = as<IRFuncType>(callee->getDataType());
            SLANG_ASSERT(funcType);
            IRBuilder builder(call);
            builder.setInsertBefore(call);
            List<IRParam*> params;
            for (auto param : callee->getParams())
                params.add(param);
            if ((UInt)params.getCount() != call->getArgCount())
                return;
            for (UInt i = 0; i < call->getArgCount(); i++)
            {
                auto paramType = params[i]->getDataType();
                auto arg = call->getArg(i);
                if (auto refType = as<IRConstRefType>(paramType))
                {
                    if (!as<IRPtrTypeBase>(arg->getDataType()))
                    {
                        auto tempVar = builder.emitVar(refType->getValueType());
                        builder.emitStore(tempVar, arg);
                        call->setArg(i, tempVar);
                    }
                }
            }
        });
}

void fixEntryPointCallsites(IRModule* module)
{
    for (auto globalInst : module->getGlobalInsts())
    {
        if (globalInst->findDecoration<IREntryPointDecoration>())
            fixEntryPointCallsites((IRFunc*)globalInst);
    }
}

} // namespace Slang
