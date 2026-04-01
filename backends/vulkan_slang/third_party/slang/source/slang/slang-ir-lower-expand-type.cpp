#include "slang-ir-lower-expand-type.h"

#include "slang-ir-clone.h"
#include "slang-ir-insts.h"
#include "slang-ir-util.h"

namespace Slang
{
IRInst* clonePatternVal(IRCloneEnv& cloneEnv, IRBuilder* builder, IRInst* val, IRInst* eachIndex);

IRInst* clonePatternValImpl(
    IRCloneEnv& cloneEnv,
    IRBuilder* builder,
    IRInst* val,
    IRInst* eachIndex)
{
    if (!val)
        return val;

    switch (val->getOp())
    {
    case kIROp_ExpandTypeOrVal:
        return val;
    case kIROp_Each:
        {
            auto eachInst = as<IREach>(val);
            auto packInst = eachInst->getElement();
            auto type =
                (IRType*)clonePatternVal(cloneEnv, builder, packInst->getFullType(), eachIndex);
            packInst = clonePatternValImpl(cloneEnv, builder, packInst, eachIndex);
            auto result = builder->emitGetTupleElement(type, packInst, eachIndex);
            return result;
        }
    case kIROp_Specialize:
    case kIROp_LookupWitness:
    case kIROp_ExtractExistentialType:
    case kIROp_ExtractExistentialWitnessTable:
        break;
    default:
        // If the value is not a type, and it is not in a block, then it is some global inst
        // that shouldn't be deep copied into current block, such as a IRFunc.
        if (!as<IRType>(val) && getBlock(val->getParent()) == nullptr)
            return val;
        break;
    }
    bool anyChange = false;
    ShortList<IRInst*> operands;
    for (UInt i = 0; i < val->getOperandCount(); i++)
    {
        auto newOperand = clonePatternVal(cloneEnv, builder, val->getOperand(i), eachIndex);
        if (newOperand != val->getOperand(i))
            anyChange = true;
        operands.add(newOperand);
    }
    auto newType = clonePatternVal(cloneEnv, builder, val->getFullType(), eachIndex);
    if (newType != val->getFullType())
        anyChange = true;
    if (!anyChange)
        return val;

    auto newVal = builder->emitIntrinsicInst(
        (IRType*)newType,
        val->getOp(),
        operands.getCount(),
        operands.getArrayView().getBuffer());
    if (newVal != val)
    {
        cloneInstDecorationsAndChildren(&cloneEnv, builder->getModule(), val, newVal);
    }
    return newVal;
}

IRInst* clonePatternVal(IRCloneEnv& cloneEnv, IRBuilder* builder, IRInst* val, IRInst* eachIndex)
{
    if (auto clonedVal = cloneEnv.mapOldValToNew.tryGetValue(val))
        return *clonedVal;
    cloneEnv.mapOldValToNew[val] = val;
    auto result = clonePatternValImpl(cloneEnv, builder, val, eachIndex);
    cloneEnv.mapOldValToNew[val] = result;
    return result;
}

// Translate a `IRExpandType` into an `IRExpand` where the `PatternType` is defined
// inside the `IRExpand` body.
//
IRInst* lowerExpandTypeImpl(IRExpandType* expandType)
{
    // Turn `IRExpandType` into an `IRExpand` instruction.
    IRBuilder builder(expandType);
    builder.setInsertBefore(expandType);
    List<IRInst*> capturedArgs;
    IRCloneEnv cloneEnv;
    for (UInt i = 0; i < expandType->getCaptureCount(); i++)
    {
        auto capturedArg = expandType->getCaptureType(i);
        capturedArgs.add(capturedArg);
    }
    auto result = builder.emitExpandInst(
        expandType->getFullType(),
        expandType->getCaptureCount(),
        capturedArgs.getBuffer());
    builder.setInsertInto(result);
    builder.emitBlock();
    auto eachIndex = builder.emitParam(builder.getIntType());
    auto newPatternType =
        clonePatternVal(cloneEnv, &builder, expandType->getPatternType(), eachIndex);
    builder.emitYield(newPatternType);
    return result;
}

// Process the body of an `IRExpand` instruction, and replace the type of children insts if it
// is an `IRExpandType`.
//
void processExpandVal(IRExpand* expandVal)
{
    IRBuilder builder(expandVal);
    IRCloneEnv cloneEnv;
    auto eachIndex = expandVal->getFirstBlock()->getFirstParam();
    for (auto block : expandVal->getBlocks())
    {
        for (auto inst : block->getModifiableChildren())
        {
            builder.setInsertBefore(inst);
            auto newType = clonePatternVal(cloneEnv, &builder, inst->getFullType(), eachIndex);
            if (newType != inst->getFullType())
            {
                inst = builder.replaceOperand(&inst->typeUse, newType);
            }
            for (UInt i = 0; i < inst->getOperandCount(); i++)
            {
                auto oldOperand = inst->getOperand(i);
                if (!oldOperand)
                    continue;
                if (isChildInstOf(oldOperand, expandVal))
                    continue;
                auto newOperand = clonePatternVal(cloneEnv, &builder, oldOperand, eachIndex);
                if (newOperand != inst->getOperand(i))
                {
                    inst = builder.replaceOperand(inst->getOperands() + i, newOperand);
                }
            }
        }
    }
}

void lowerExpandType(IRModule* module)
{
    // Use a work list to process all instructions in the module, and lower any `IRExpandType` we
    // see along the way.

    List<IRInst*> workList;
    for (auto type : module->getGlobalInsts())
    {
        workList.add(type);
    }

    while (workList.getCount() != 0)
    {
        auto inst = workList.getLast();
        workList.removeLast();

        if (auto expandType = as<IRExpandType>(inst))
        {
            inst = lowerExpandTypeImpl(expandType);
            if (inst != expandType)
            {
                expandType->replaceUsesWith(inst);
                expandType->removeAndDeallocate();
            }
        }
        else if (auto expandVal = as<IRExpand>(inst))
        {
            processExpandVal(expandVal);
        }
        for (auto child : inst->getChildren())
        {
            workList.add(child);
        }
    }
}
} // namespace Slang
