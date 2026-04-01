#include "slang-ir-metal-legalize.h"

#include "slang-ir-clone.h"
#include "slang-ir-insts.h"
#include "slang-ir-legalize-binary-operator.h"
#include "slang-ir-legalize-varying-params.h"
#include "slang-ir-specialize-address-space.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{

// metal textures only support writing 4-component values, even if the texture is only 1, 2, or
// 3-component in this case the other channels get ignored, but the signature still doesnt match so
// now we have to replace the value being written with a 4-component vector where the new components
// get ignored, nice
void legalizeImageStoreValue(IRBuilder& builder, IRImageStore* imageStore)
{
    builder.setInsertBefore(imageStore);
    auto originalValue = imageStore->getValue();
    auto valueBaseType = originalValue->getDataType();
    IRType* elementType = nullptr;
    List<IRInst*> components;
    if (auto valueVectorType = as<IRVectorType>(valueBaseType))
    {
        if (auto originalElementCount = as<IRIntLit>(valueVectorType->getElementCount()))
        {
            if (originalElementCount->getValue() == 4)
            {
                return;
            }
        }
        elementType = valueVectorType->getElementType();
        auto vectorValue = as<IRMakeVector>(originalValue);
        for (UInt i = 0; i < vectorValue->getOperandCount(); i++)
        {
            components.add(vectorValue->getOperand(i));
        }
    }
    else
    {
        elementType = valueBaseType;
        components.add(originalValue);
    }
    for (UInt i = components.getCount(); i < 4; i++)
    {
        components.add(builder.getIntValue(builder.getIntType(), 0));
    }
    auto fourComponentVectorType = builder.getVectorType(elementType, 4);
    imageStore->setOperand(2, builder.emitMakeVector(fourComponentVectorType, components));
}

void legalizeFuncBody(IRFunc* func)
{
    IRBuilder builder(func);
    for (auto block : func->getBlocks())
    {
        for (auto inst : block->getModifiableChildren())
        {
            if (auto call = as<IRCall>(inst))
            {
                ShortList<IRUse*> argsToFixup;
                // Metal doesn't support taking the address of a vector element.
                // If such an address is used as an argument to a call, we need to replace it with a
                // temporary. for example, if we see:
                // ```
                //     void foo(inout float x) { x = 1; }
                //     float4 v;
                //     foo(v.x);
                // ```
                // We need to transform it into:
                // ```
                //     float4 v;
                //     float temp = v.x;
                //     foo(temp);
                //     v.x = temp;
                // ```
                //
                for (UInt i = 0; i < call->getArgCount(); i++)
                {
                    if (auto addr = as<IRGetElementPtr>(call->getArg(i)))
                    {
                        auto ptrType = addr->getBase()->getDataType();
                        auto valueType = tryGetPointedToType(&builder, ptrType);
                        if (!valueType)
                            continue;
                        if (as<IRVectorType>(valueType))
                            argsToFixup.add(call->getArgs() + i);
                    }
                }
                if (argsToFixup.getCount() == 0)
                    continue;

                // Define temp vars for all args that need fixing up.
                for (auto arg : argsToFixup)
                {
                    auto addr = as<IRGetElementPtr>(arg->get());
                    auto ptrType = addr->getDataType();
                    auto valueType = tryGetPointedToType(&builder, ptrType);
                    builder.setInsertBefore(call);
                    auto temp = builder.emitVar(valueType);
                    auto initialValue = builder.emitLoad(valueType, addr);
                    builder.emitStore(temp, initialValue);
                    builder.setInsertAfter(call);
                    builder.emitStore(addr, builder.emitLoad(valueType, temp));
                    arg->set(temp);
                }
            }
            if (auto write = as<IRImageStore>(inst))
            {
                legalizeImageStoreValue(builder, write);
            }
        }
    }
}

struct MetalAddressSpaceAssigner : InitialAddressSpaceAssigner
{
    virtual bool tryAssignAddressSpace(IRInst* inst, AddressSpace& outAddressSpace) override
    {
        switch (inst->getOp())
        {
        case kIROp_Var:
            outAddressSpace = AddressSpace::ThreadLocal;
            return true;
        case kIROp_RWStructuredBufferGetElementPtr:
            outAddressSpace = AddressSpace::Global;
            return true;
        default:
            return false;
        }
    }

    virtual AddressSpace getAddressSpaceFromVarType(IRInst* type) override
    {
        if (as<IRUniformParameterGroupType>(type))
        {
            return AddressSpace::Uniform;
        }
        if (as<IRByteAddressBufferTypeBase>(type))
        {
            return AddressSpace::Global;
        }
        if (as<IRHLSLStructuredBufferTypeBase>(type))
        {
            return AddressSpace::Global;
        }
        if (as<IRGLSLShaderStorageBufferType>(type))
        {
            return AddressSpace::Global;
        }
        if (auto ptrType = as<IRPtrTypeBase>(type))
        {
            if (ptrType->hasAddressSpace())
                return ptrType->getAddressSpace();
            return AddressSpace::Global;
        }
        return AddressSpace::Generic;
    }

    virtual AddressSpace getLeafInstAddressSpace(IRInst* inst) override
    {
        if (as<IRGroupSharedRate>(inst->getRate()))
            return AddressSpace::GroupShared;
        switch (inst->getOp())
        {
        case kIROp_RWStructuredBufferGetElementPtr:
            return AddressSpace::Global;
        case kIROp_Var:
            if (as<IRBlock>(inst->getParent()))
                return AddressSpace::ThreadLocal;
            break;
        default:
            break;
        }
        auto type = unwrapAttributedType(inst->getDataType());
        if (!type)
            return AddressSpace::Generic;
        return getAddressSpaceFromVarType(type);
    }
};

static void processInst(IRInst* inst, DiagnosticSink* sink)
{
    switch (inst->getOp())
    {
    case kIROp_Add:
    case kIROp_Sub:
    case kIROp_Mul:
    case kIROp_Div:
    case kIROp_FRem:
    case kIROp_IRem:
    case kIROp_And:
    case kIROp_Or:
    case kIROp_BitAnd:
    case kIROp_BitOr:
    case kIROp_BitXor:
    case kIROp_Lsh:
    case kIROp_Rsh:
    case kIROp_Eql:
    case kIROp_Neq:
    case kIROp_Greater:
    case kIROp_Less:
    case kIROp_Geq:
    case kIROp_Leq:
        legalizeBinaryOp(inst, sink, CodeGenTarget::Metal);
        break;
    case kIROp_MetalCastToDepthTexture:
        {
            // If the operand is already a depth texture, don't do anything.
            auto textureType = as<IRTextureTypeBase>(inst->getOperand(0)->getDataType());
            if (textureType && getIntVal(textureType->getIsShadowInst()) == 1)
            {
                inst->replaceUsesWith(inst->getOperand(0));
                inst->removeAndDeallocate();
            }
            break;
        }
    default:
        for (auto child : inst->getModifiableChildren())
        {
            processInst(child, sink);
        }
    }
}

void legalizeIRForMetal(IRModule* module, DiagnosticSink* sink)
{
    List<EntryPointInfo> entryPoints;
    for (auto inst : module->getGlobalInsts())
    {
        if (auto func = as<IRFunc>(inst))
        {
            if (auto entryPointDecor = func->findDecoration<IREntryPointDecoration>())
            {
                EntryPointInfo info;
                info.entryPointDecor = entryPointDecor;
                info.entryPointFunc = func;
                entryPoints.add(info);
            }
            legalizeFuncBody(func);
        }
    }

    legalizeEntryPointVaryingParamsForMetal(module, sink, entryPoints);

    MetalAddressSpaceAssigner metalAddressSpaceAssigner;
    specializeAddressSpace(module, &metalAddressSpaceAssigner);

    processInst(module->getModuleInst(), sink);
}

} // namespace Slang
