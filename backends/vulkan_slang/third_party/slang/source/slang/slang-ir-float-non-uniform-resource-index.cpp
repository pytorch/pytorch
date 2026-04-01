#include "slang-ir-float-non-uniform-resource-index.h"

#include "slang-ir-util.h"

namespace Slang
{
void processNonUniformResourceIndex(
    IRInst* nonUniformResourceIndexInst,
    NonUniformResourceIndexFloatMode floatMode)
{
    // float `NonUniformResourceIndex()` to right before the access operation
    // by walking up the use-def chain
    // from nonUniformResource inst of an index to an array of buffer or
    // texture def all the way to the leaf operations. To be precise:
    // - go through GEP and see if it calls an intrinsic function,
    //   then decorate the address itself (GetElementPtr)
    // - go through GEP to identify the pointer access and the Loads that it
    //   accesses (GetElementPtr -> Load), then decorate the load instruction.
    // - go through IntCasts to deal with u32 -> i32 / vice-versa (IntCast)
    List<IRInst*> resWorkList;

    // Handle cases when `nonUniformResourceIndexInst` inst is wrapped around
    // an index in a nested fashion, i.e. nonUniform(nonUniform(index)) by
    // only adding the inner-most inst in the worklist, and work our way out.
    auto insti = nonUniformResourceIndexInst;
    while (insti->getOp() == kIROp_NonUniformResourceIndex)
    {
        if (resWorkList.getCount() != 0)
            resWorkList.removeLast();
        resWorkList.add(insti);
        insti = insti->getOperand(0);
    }

    // For all the users of a `nonUniformResourceIndexInst`, make them directly
    // use the underlying base inst that is wrapped by `nonUniformResourceIndex`
    // and finally wrap them with a `nonUniformResourceIndex`, and add back to the
    // worklist, and keep bubbling them up until it can.
    for (Index i = 0; i < resWorkList.getCount(); i++)
    {
        auto inst = resWorkList[i];
        traverseUses(
            inst,
            [&](IRUse* use)
            {
                auto user = use->getUser();
                IRBuilder builder(user);
                builder.setInsertBefore(user);

                IRInst* newUser = nullptr;
                switch (user->getOp())
                {
                case kIROp_IntCast:
                    // Replace intCast(nonUniformRes(x)), into nonUniformRes(intCast(x))
                    newUser = builder.emitCast(user->getFullType(), inst->getOperand(0));
                    break;
                case kIROp_CastDescriptorHandleToUInt2:
                    {
                        // Replace castBindlessToInt(nonUniformRes(x)), into
                        // nonUniformRes(castBindlessToInt(x))
                        auto operand = inst->getOperand(0);
                        newUser = builder.emitIntrinsicInst(
                            user->getFullType(),
                            kIROp_CastDescriptorHandleToUInt2,
                            1,
                            &operand);
                    }
                    break;
                case kIROp_GetElementPtr:
                    // Ignore when `NonUniformResourceIndex` is not on the index
                    if (floatMode != NonUniformResourceIndexFloatMode::SPIRV)
                        break;
                    if (user->getOperand(1) == inst)
                    {
                        // Replace gep(pArray, nonUniformRes(x)), into
                        // nonUniformRes(gep(pArray, x))
                        newUser = builder.emitElementAddress(
                            user->getFullType(),
                            user->getOperand(0),
                            inst->getOperand(0));
                    }
                    break;
                case kIROp_GetElement:
                    // Ignore when `NonUniformResourceIndex` is not on base
                    if (user->getOperand(0) == inst)
                    {
                        // Replace getElement(nonuniformRes(obj), i), into
                        // nonUniformRes(getElement(obj, i))
                        newUser = builder.emitElementExtract(
                            user->getFullType(),
                            inst->getOperand(0),
                            user->getOperand(1));
                    }
                    break;
                case kIROp_swizzle:
                    // Ignore when `NonUniformResourceIndex` is not on base
                    if (user->getOperand(0) == inst)
                    {
                        // Replace getElement(nonuniformRes(obj), i), into
                        // nonUniformRes(getElement(obj, i))
                        ShortList<IRInst*> operands;
                        for (UInt i = 0; i < user->getOperandCount(); i++)
                            operands.add(user->getOperand(i));
                        operands[0] = inst->getOperand(0);
                        newUser = builder.emitIntrinsicInst(
                            user->getFullType(),
                            kIROp_swizzle,
                            operands.getCount(),
                            operands.getArrayView().getBuffer());
                    }
                    break;
                case kIROp_NonUniformResourceIndex:
                    // Replace nonUniformRes(nonUniformRes(x)), into nonUniformRes(x)
                    newUser = inst->getOperand(0);
                    break;
                case kIROp_Load:
                    if (floatMode != NonUniformResourceIndexFloatMode::SPIRV)
                        break;
                    // Replace load(nonUniformRes(x)), into nonUniformRes(load(x))
                    newUser = builder.emitLoad(user->getFullType(), inst->getOperand(0));
                    break;
                default:
                    // Ignore for all other unknown insts.
                    break;
                };

                // Early exit when we could not process the `NonUniformResourceIndex` inst.
                if (!newUser)
                    return;

                auto nonuniformUser = builder.emitNonUniformResourceIndexInst(newUser);
                user->replaceUsesWith(nonuniformUser);

                // Update the worklist with the newly added `NonUniformResourceIndex` inst,
                // based on the base inst it was constructed around, in case we need to further
                // bubble up the `NonUniformResourceIndex` inst.
                switch (user->getOp())
                {
                case kIROp_IntCast:
                case kIROp_GetElementPtr:
                case kIROp_Load:
                case kIROp_NonUniformResourceIndex:
                case kIROp_CastDescriptorHandleToUInt2:
                case kIROp_GetElement:
                case kIROp_swizzle:
                    resWorkList.add(nonuniformUser);
                    break;
                };

                // Clean up the base inst from the IR module, to avoid duplicate decorations.
                user->removeAndDeallocate();
            });
    }

    if (floatMode != NonUniformResourceIndexFloatMode::SPIRV)
        return;
    // Once all the `NonUniformResourceIndex` insts are visited, and the inst type is bubbled up
    // to the parent, a decoration is added to the operands of the insts.
    for (int i = 0; i < resWorkList.getCount(); ++i)
    {
        // It is only required to decorate the base inst, if the `NonUniformResourceIndex` inst
        // around it has any active uses.
        auto inst = resWorkList[i];
        if (!inst->hasUses())
        {
            inst->removeAndDeallocate();
            continue;
        }
        // For each of the `NonUniformResourceIndex` inst that remain, decorate the base inst
        // with a [NonUniformResource] decoration, which is the operand0 of the inst, only
        // when the type is a resource type, or a pointer to a resource type, or a pointer
        // in the Physical Storage buffer address space.
        auto operand = inst->getOperand(0);
        auto type = operand->getDataType();
        if (isResourceType(type) || isPointerToResourceType(type))
        {
            IRBuilder builder(operand);
            builder.addSPIRVNonUniformResourceDecoration(operand);
            if (operand->getOp() == kIROp_Load)
            {
                // If the inst is a load, then the addr inst itself should also be decorated
                // with the [NonUniformResource] decoration.
                auto addr = operand->getOperand(0);
                if (!addr->findDecoration<IRSPIRVNonUniformResourceDecoration>())
                    builder.addSPIRVNonUniformResourceDecoration(addr);
            }
        }
        inst->replaceUsesWith(operand);
        inst->removeAndDeallocate();
    }
}

void floatNonUniformResourceIndex(IRModule* module, NonUniformResourceIndexFloatMode floatMode)
{
    // Walk through all the instructions in the module, and float the `NonUniformResourceIndex`
    // insts to the right place in the IR module.

    List<IRInst*> workList;
    for (auto globalInst : module->getGlobalInsts())
    {
        auto func = as<IRGlobalValueWithCode>(getGenericReturnVal(globalInst));
        if (!func)
            continue;
        workList.clear();
        for (auto block : func->getBlocks())
        {
            for (auto inst : block->getChildren())
            {
                if (inst->getOp() == kIROp_NonUniformResourceIndex)
                    workList.add(inst);
            }
        }
        for (auto inst : workList)
        {
            if (inst->getParent() != nullptr)
                processNonUniformResourceIndex(inst, floatMode);
        }
    }
}
} // namespace Slang
