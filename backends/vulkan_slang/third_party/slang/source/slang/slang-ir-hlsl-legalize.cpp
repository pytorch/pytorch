// slang-ir-hlsl-legalize.cpp
#include "slang-ir-hlsl-legalize.h"

#include "slang-ir-inst-pass-base.h"
#include "slang-ir-insts.h"
#include "slang-ir-specialize-function-call.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

#include <functional>

namespace Slang
{

void searchChildrenForForceVarIntoStructTemporarily(IRModule* module, IRInst* inst)
{
    for (auto child : inst->getChildren())
    {
        switch (child->getOp())
        {
        case kIROp_Block:
            {
                searchChildrenForForceVarIntoStructTemporarily(module, child);
                break;
            }
        case kIROp_Call:
            {
                auto call = as<IRCall>(child);
                for (UInt i = 0; i < call->getArgCount(); i++)
                {
                    auto arg = call->getArg(i);
                    const bool isForcedStruct = arg->getOp() == kIROp_ForceVarIntoStructTemporarily;
                    const bool isForcedRayPayloadStruct =
                        arg->getOp() == kIROp_ForceVarIntoRayPayloadStructTemporarily;
                    if (!(isForcedStruct || isForcedRayPayloadStruct))
                        continue;
                    auto forceStructArg = arg->getOperand(0);
                    auto forceStructBaseType =
                        (IRType*)(forceStructArg->getDataType()->getOperand(0));
                    IRBuilder builder(call);
                    if (forceStructBaseType->getOp() == kIROp_StructType)
                    {
                        call->setArg(i, arg->getOperand(0));
                        if (isForcedRayPayloadStruct)
                            builder.addRayPayloadDecoration(forceStructBaseType);
                        continue;
                    }

                    // When `__forceVarIntoStructTemporarily` is called with a non-struct type
                    // parameter, we create a temporary struct and copy the parameter into the
                    // struct. This struct is then subsituted for the return of
                    // `__forceVarIntoStructTemporarily`. Optionally, if
                    // `__forceVarIntoStructTemporarily` is a parameter to a side effect type
                    // (`ref`, `out`, `inout`) we copy the struct back into our original non-struct
                    // parameter.

                    const auto typeNameHint = isForcedRayPayloadStruct
                                                  ? "RayPayload_t"
                                                  : "ForceVarIntoStructTemporarily_t";
                    const auto varNameHint =
                        isForcedRayPayloadStruct ? "rayPayload" : "forceVarIntoStructTemporarily";

                    builder.setInsertBefore(call->getCallee());
                    auto structType = builder.createStructType();
                    StringBuilder structName;
                    builder.addNameHintDecoration(structType, UnownedStringSlice(typeNameHint));
                    if (isForcedRayPayloadStruct)
                        builder.addRayPayloadDecoration(structType);

                    auto elementBufferKey = builder.createStructKey();
                    builder.addNameHintDecoration(elementBufferKey, UnownedStringSlice("data"));
                    auto _dataField = builder.createStructField(
                        structType,
                        elementBufferKey,
                        forceStructBaseType);

                    builder.setInsertBefore(call);
                    auto structVar = builder.emitVar(structType);
                    builder.addNameHintDecoration(structVar, UnownedStringSlice(varNameHint));
                    builder.emitStore(
                        builder.emitFieldAddress(
                            builder.getPtrType(_dataField->getFieldType()),
                            structVar,
                            _dataField->getKey()),
                        builder.emitLoad(forceStructArg));

                    arg->replaceUsesWith(structVar);
                    arg->removeAndDeallocate();

                    auto argType = call->getCallee()->getDataType()->getOperand(i + 1);
                    if (!isPtrLikeOrHandleType(argType))
                        continue;

                    builder.setInsertAfter(call);
                    builder.emitStore(
                        forceStructArg,
                        builder.emitFieldAddress(
                            builder.getPtrType(_dataField->getFieldType()),
                            structVar,
                            _dataField->getKey()));
                }
                break;
            }
        }
    }
}

void legalizeNonStructParameterToStructForHLSL(IRModule* module)
{
    for (auto globalInst : module->getGlobalInsts())
    {
        if (globalInst->getOp() != kIROp_Func)
            continue;
        searchChildrenForForceVarIntoStructTemporarily(module, globalInst);
    }
}

} // namespace Slang
