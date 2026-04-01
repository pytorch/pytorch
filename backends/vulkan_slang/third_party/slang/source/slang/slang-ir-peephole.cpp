#include "slang-ir-peephole.h"

#include "slang-ir-dominators.h"
#include "slang-ir-inst-pass-base.h"
#include "slang-ir-layout.h"
#include "slang-ir-sccp.h"
#include "slang-ir-util.h"

namespace Slang
{
struct PeepholeContext : InstPassBase
{
    PeepholeContext(IRModule* inModule)
        : InstPassBase(inModule)
    {
    }

    bool changed = false;
    FloatingPointMode floatingPointMode = FloatingPointMode::Precise;
    bool removeOldInst = true;
    bool isInGeneric = false;
    bool isPrelinking = false;
    bool useFastAnalysis = false;

    TargetProgram* targetProgram;

    void maybeRemoveOldInst(IRInst* inst)
    {
        if (removeOldInst)
            inst->removeAndDeallocate();
    }

    bool tryFoldElementExtractFromUpdateInst(IRInst* inst)
    {
        bool isAccessChainEqual = false;
        bool isAccessChainNotEqual = false;
        List<IRInst*> chainKey;
        IRInst* chainNode = inst;
        for (;;)
        {
            switch (chainNode->getOp())
            {
            case kIROp_FieldExtract:
            case kIROp_GetElement:
                chainKey.add(chainNode->getOperand(1));
                chainNode = chainNode->getOperand(0);
                continue;
            }
            break;
        }
        chainKey.reverse();
        if (auto updateInst = as<IRUpdateElement>(chainNode))
        {
            // If we see an extract(updateElement(x, accessChain, val), accessChain), then
            // we can replace the inst with val.

            if (updateInst->getAccessKeyCount() > (UInt)chainKey.getCount())
                return false;

            isAccessChainEqual = true;
            for (UInt i = 0; i < updateInst->getAccessKeyCount(); i++)
            {
                if (updateInst->getAccessKey(i) != chainKey[i])
                {
                    isAccessChainEqual = false;
                    if (as<IRStructKey>(chainKey[i]))
                    {
                        isAccessChainNotEqual = true;
                        break;
                    }
                    else
                    {
                        if (auto constIndex1 = as<IRIntLit>(updateInst->getAccessKey(i)))
                        {
                            if (auto constIndex2 = as<IRIntLit>(chainKey[i]))
                            {
                                if (constIndex1->getValue() != constIndex2->getValue())
                                {
                                    isAccessChainNotEqual = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            if (isAccessChainEqual)
            {
                auto remainingKeys = chainKey.getArrayView(
                    updateInst->getAccessKeyCount(),
                    chainKey.getCount() - updateInst->getAccessKeyCount());
                if (remainingKeys.getCount() == 0)
                {
                    inst->replaceUsesWith(updateInst->getElementValue());
                    maybeRemoveOldInst(inst);
                    return true;
                }
                else if (remainingKeys.getCount() > 0)
                {
                    IRBuilder builder(module);
                    IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);
                    builder.setInsertBefore(inst);
                    auto newValue =
                        builder.emitElementExtract(updateInst->getElementValue(), remainingKeys);
                    inst->replaceUsesWith(newValue);
                    maybeRemoveOldInst(inst);
                    return true;
                }
            }
            else if (isAccessChainNotEqual)
            {
                // If we see an extract(updateElement(x, accessChain, val), accessChain2), where
                // accessChain!=accessChain2, then we can replace the inst with extract(x,
                // accessChain2).
                IRBuilder builder(module);
                IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);
                builder.setInsertBefore(inst);
                auto newInst =
                    builder.emitElementExtract(updateInst->getOldValue(), chainKey.getArrayView());
                inst->replaceUsesWith(newInst);
                maybeRemoveOldInst(inst);
                return true;
            }
        }
        return false;
    }

    bool tryOptimizeArithmeticInst(IRInst* inst)
    {
        bool allowUnsafeOptimizations =
            (floatingPointMode == FloatingPointMode::Fast ||
             isIntegralScalarOrCompositeType(inst->getDataType()));

        auto tryReplace = [&](IRInst* replacement) -> bool
        {
            if (replacement->getFullType() != inst->getFullType())
            {
                // If the operand type is different from result type,
                // we try to convert for some known cases.
                if (auto vectorType = as<IRVectorType>(inst->getFullType()))
                {
                    if (vectorType->getElementType() != replacement->getFullType())
                        return false;
                    IRBuilder builder(module);
                    IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);

                    builder.setInsertBefore(inst);
                    replacement =
                        builder.emitMakeVectorFromScalar(inst->getFullType(), replacement);
                }
                else
                {
                    return false;
                }
            }

            inst->replaceUsesWith(replacement);
            maybeRemoveOldInst(inst);
            return true;
        };

        switch (inst->getOp())
        {
        case kIROp_Add:
            if (isZero(inst->getOperand(0)))
            {
                return tryReplace(inst->getOperand(1));
            }
            else if (isZero(inst->getOperand(1)))
            {
                return tryReplace(inst->getOperand(0));
            }
            break;
        case kIROp_Sub:
            if (isZero(inst->getOperand(1)))
            {
                return tryReplace(inst->getOperand(0));
            }
            else if (inst->getOperand(0) == inst->getOperand(1))
            {
                IRBuilder builder(inst);
                IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);
                builder.setInsertBefore(inst);
                return tryReplace(builder.emitDefaultConstruct(inst->getDataType()));
            }
            break;
        case kIROp_Mul:
            if (isOne(inst->getOperand(0)))
            {
                return tryReplace(inst->getOperand(1));
            }
            else if (isOne(inst->getOperand(1)))
            {
                return tryReplace(inst->getOperand(0));
            }
            else if (allowUnsafeOptimizations && isZero(inst->getOperand(0)))
            {
                return tryReplace(inst->getOperand(0));
            }
            else if (allowUnsafeOptimizations && isZero(inst->getOperand(1)))
            {
                return tryReplace(inst->getOperand(1));
            }
            break;
        case kIROp_Div:
            if (allowUnsafeOptimizations && isZero(inst->getOperand(0)))
            {
                return tryReplace(inst->getOperand(0));
            }
            else if (isOne(inst->getOperand(1)))
            {
                return tryReplace(inst->getOperand(0));
            }
            break;
        case kIROp_And:
            if (isZero(inst->getOperand(0)))
            {
                return tryReplace(inst->getOperand(0));
            }
            else if (isZero(inst->getOperand(1)))
            {
                return tryReplace(inst->getOperand(1));
            }
            else if (isOne(inst->getOperand(1)))
            {
                return tryReplace(inst->getOperand(0));
            }
            else if (isOne(inst->getOperand(0)))
            {
                return tryReplace(inst->getOperand(1));
            }
            break;
        case kIROp_Or:
            if (isZero(inst->getOperand(0)))
            {
                return tryReplace(inst->getOperand(1));
            }
            else if (isZero(inst->getOperand(1)))
            {
                return tryReplace(inst->getOperand(0));
            }
            else if (isOne(inst->getOperand(1)))
            {
                return tryReplace(inst->getOperand(1));
            }
            else if (isOne(inst->getOperand(0)))
            {
                return tryReplace(inst->getOperand(0));
            }
            break;
        }
        return false;
    }

    void processInst(IRInst* inst)
    {
        if (as<IRGlobalValueWithCode>(inst))
        {
            if (auto fpModeDecor = inst->findDecoration<IRFloatingModeOverrideDecoration>())
                floatingPointMode = fpModeDecor->getFloatingPointMode();
        }

        switch (inst->getOp())
        {
        case kIROp_AlignOf:
        case kIROp_SizeOf:
            {
                if (!targetProgram)
                    break;

                // Save the alignment information and exit early if it is invalid
                IRSizeAndAlignment sizeAlignment;
                IRType* baseType = nullptr;
                if (auto t = as<IRType>(inst->getOperand(0)))
                    baseType = t;
                else
                    baseType = inst->getOperand(0)->getDataType();

                if (SLANG_FAILED(getNaturalSizeAndAlignment(
                        targetProgram->getOptionSet(),
                        baseType,
                        &sizeAlignment)))
                    break;
                if (sizeAlignment.size == 0)
                    break;

                IRBuilder builder(module);
                IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);

                builder.setInsertBefore(inst);
                IRInst* resultVal = nullptr;
                if (inst->getOp() == kIROp_AlignOf)
                    resultVal = builder.getIntValue(inst->getDataType(), sizeAlignment.alignment);
                else
                    resultVal = builder.getIntValue(inst->getDataType(), sizeAlignment.size);
                inst->replaceUsesWith(resultVal);
                maybeRemoveOldInst(inst);
                changed = true;
            }
            break;
        case kIROp_GetArrayLength:
            if (auto arrayType = as<IRArrayType>(inst->getOperand(0)->getDataType()))
            {
                inst->replaceUsesWith(arrayType->getElementCount());
                maybeRemoveOldInst(inst);
                changed = true;
            }
            break;
        case kIROp_GetResultError:
            if (inst->getOperand(0)->getOp() == kIROp_MakeResultError)
            {
                inst->replaceUsesWith(inst->getOperand(0)->getOperand(0));
                maybeRemoveOldInst(inst);
                changed = true;
            }
            break;
        case kIROp_GetResultValue:
            if (inst->getOperand(0)->getOp() == kIROp_MakeResultValue)
            {
                inst->replaceUsesWith(inst->getOperand(0)->getOperand(0));
                maybeRemoveOldInst(inst);
                changed = true;
            }
            break;
        case kIROp_IsResultError:
            if (inst->getOperand(0)->getOp() == kIROp_MakeResultError)
            {
                IRBuilder builder(module);
                IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);

                inst->replaceUsesWith(builder.getBoolValue(true));
                maybeRemoveOldInst(inst);
                changed = true;
            }
            else if (inst->getOperand(0)->getOp() == kIROp_MakeResultValue)
            {
                IRBuilder builder(module);
                IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);

                inst->replaceUsesWith(builder.getBoolValue(false));
                maybeRemoveOldInst(inst);
                changed = true;
            }
            break;
        case kIROp_GetTupleElement:
            switch (inst->getOperand(0)->getOp())
            {
            case kIROp_MakeTuple:
            case kIROp_MakeValuePack:
            case kIROp_MakeWitnessPack:
            case kIROp_TypePack:
                {
                    auto element = inst->getOperand(1);
                    if (auto intLit = as<IRIntLit>(element))
                    {
                        inst->replaceUsesWith(
                            inst->getOperand(0)->getOperand((UInt)intLit->value.intVal));
                        maybeRemoveOldInst(inst);
                        changed = true;
                    }
                    break;
                }
            default:
                break;
            }
            break;
        case kIROp_MakeCoopVectorFromValuePack:
            {
                const auto pack = inst->getOperand(0);
                if (const auto packType = as<IRTypePack>(pack->getDataType()))
                {
                    IRBuilder builder(inst);
                    IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);

                    builder.setInsertBefore(inst);
                    List<IRInst*> args;
                    for (UInt j = 0; j < packType->getOperandCount(); ++j)
                    {
                        const auto e = builder.emitGetTupleElement(
                            cast<IRType>(packType->getOperand(j)),
                            pack,
                            j);
                        args.add(e);
                    }
                    const auto cvt = builder.getCoopVectorType(
                        args[0]->getDataType(),
                        builder.getIntValue(builder.getIntType(), args.getCount()));
                    const auto v = builder.emitMakeCoopVector(cvt, args.getCount(), args.begin());
                    inst->replaceUsesWith(v);
                    inst->removeAndDeallocate();
                }
            }
            break;
        case kIROp_FieldExtract:
            if (inst->getOperand(0)->getOp() == kIROp_MakeStruct)
            {
                auto field = as<IRFieldExtract>(inst)->field.get();
                Index fieldIndex = -1;
                auto structType = as<IRStructType>(inst->getOperand(0)->getDataType());
                if (structType)
                {
                    Index i = 0;
                    for (auto sfield : structType->getFields())
                    {
                        // skip the void field
                        if (as<IRVoidType>(sfield->getFieldType()))
                        {
                            continue;
                        }

                        if (sfield->getKey() == field)
                        {
                            fieldIndex = i;
                            break;
                        }
                        i++;
                    }
                    if (fieldIndex != -1 &&
                        fieldIndex < (Index)inst->getOperand(0)->getOperandCount())
                    {
                        inst->replaceUsesWith(inst->getOperand(0)->getOperand((UInt)fieldIndex));
                        maybeRemoveOldInst(inst);
                        changed = true;
                    }
                }
            }
            else
            {
                changed |= tryFoldElementExtractFromUpdateInst(inst);
            }
            break;
        case kIROp_GetElement:
            if (inst->getOperand(0)->getOp() == kIROp_MakeArray)
            {
                auto index = as<IRIntLit>(as<IRGetElement>(inst)->getIndex());
                if (!index)
                    break;
                auto opCount = inst->getOperand(0)->getOperandCount();
                if ((UInt)index->getValue() < opCount)
                {
                    inst->replaceUsesWith(inst->getOperand(0)->getOperand((UInt)index->getValue()));
                    maybeRemoveOldInst(inst);
                    changed = true;
                }
            }
            else if (inst->getOperand(0)->getOp() == kIROp_MakeVector)
            {
                auto index = as<IRIntLit>(as<IRGetElement>(inst)->getIndex());
                if (!index)
                    break;
                auto opCount = inst->getOperand(0)->getOperandCount();
                IRIntegerValue startIndex = 0;
                for (UInt i = 0; i < opCount; i++)
                {
                    auto element = inst->getOperand(0)->getOperand(i);
                    if (auto elementVecType = as<IRVectorType>(element->getDataType()))
                    {
                        auto vecSize = as<IRIntLit>(elementVecType->getElementCount());
                        if (!vecSize)
                            break;
                        if (index->getValue() >= startIndex &&
                            index->getValue() < startIndex + vecSize->getValue())
                        {
                            IRBuilder builder(module);
                            IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);

                            builder.setInsertBefore(inst);
                            auto newElement = builder.emitElementExtract(
                                element,
                                builder.getIntValue(
                                    builder.getIntType(),
                                    index->getValue() - startIndex));
                            inst->replaceUsesWith(newElement);
                            maybeRemoveOldInst(inst);
                            changed = true;
                            break;
                        }
                        startIndex += vecSize->getValue();
                    }
                    else
                    {
                        if (startIndex == index->getValue())
                        {
                            inst->replaceUsesWith(element);
                            maybeRemoveOldInst(inst);
                            changed = true;
                            break;
                        }
                        startIndex++;
                    }
                }
            }
            else if (
                inst->getOperand(0)->getOp() == kIROp_MakeArrayFromElement ||
                inst->getOperand(0)->getOp() == kIROp_MakeVectorFromScalar)
            {
                inst->replaceUsesWith(inst->getOperand(0)->getOperand(0));
                maybeRemoveOldInst(inst);
                changed = true;
            }
            else
            {
                changed |= tryFoldElementExtractFromUpdateInst(inst);
            }
            break;
        case kIROp_UpdateElement:
            {
                auto updateInst = as<IRUpdateElement>(inst);
                if (updateInst->getAccessKeyCount() != 1)
                    break;
                auto key = updateInst->getAccessKey(0);
                if (auto constIndex = as<IRIntLit>(key))
                {
                    auto oldVal = inst->getOperand(0);
                    if (oldVal->getOp() == kIROp_MakeArray ||
                        oldVal->getOp() == kIROp_MakeArrayFromElement)
                    {
                        auto arrayType = as<IRArrayType>(inst->getDataType());
                        if (!arrayType)
                            break;
                        auto arraySize = as<IRIntLit>(arrayType->getElementCount());
                        if (!arraySize)
                            break;
                        List<IRInst*> args;
                        for (IRIntegerValue i = 0; i < arraySize->getValue(); i++)
                        {
                            IRInst* arg = nullptr;
                            if (i < (IRIntegerValue)oldVal->getOperandCount())
                                arg = oldVal->getOperand((UInt)i);
                            else if (oldVal->getOperandCount() != 0)
                                arg = oldVal->getOperand(0);
                            else
                                break;
                            if (i == (IRIntegerValue)constIndex->getValue())
                                arg = updateInst->getElementValue();
                            args.add(arg);
                        }
                        if (args.getCount() == arraySize->getValue())
                        {
                            IRBuilder builder(module);
                            IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);

                            builder.setInsertBefore(inst);
                            auto makeArray = builder.emitMakeArray(
                                arrayType,
                                (UInt)args.getCount(),
                                args.getBuffer());
                            inst->replaceUsesWith(makeArray);
                            maybeRemoveOldInst(inst);
                            changed = true;
                        }
                    }
                    else
                    {
                        // Check if the updated value is a chain of `updateElement` instructions
                        // that updates every element in the same array, and if so we can replace
                        // the whole chain with a single `makeArray` instruction.
                        auto arrayType = as<IRArrayType>(inst->getDataType());
                        if (!arrayType)
                            break;
                        auto arraySize = as<IRIntLit>(arrayType->getElementCount());
                        if (!arraySize)
                            break;

                        List<IRInst*> args;
                        args.setCount((UInt)arraySize->getValue());
                        for (Index i = 0; i < args.getCount(); i++)
                            args[i] = nullptr;

                        for (auto updateElement = updateInst; updateElement;
                             updateElement = as<IRUpdateElement>(updateElement->getOldValue()))
                        {
                            auto subKey = updateElement->getAccessKey(0);
                            auto subConstIndex = as<IRIntLit>(subKey);
                            if (!subConstIndex)
                                break;
                            auto index = (Index)subConstIndex->getValue();
                            if (index >= args.getCount())
                                break;
                            // If we have already seen an update for this index, then we can't
                            // override it with an earlier update.
                            if (args[index])
                                continue;
                            args[index] = updateElement->getElementValue();
                        }

                        bool isComplete = true;
                        for (auto arg : args)
                        {
                            if (!arg)
                            {
                                isComplete = false;
                                break;
                            }
                        }
                        if (isComplete)
                        {
                            IRBuilder builder(module);
                            IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);

                            builder.setInsertBefore(inst);
                            auto makeArray = builder.emitMakeArray(
                                arrayType,
                                (UInt)args.getCount(),
                                args.getBuffer());
                            inst->replaceUsesWith(makeArray);
                            maybeRemoveOldInst(inst);
                            changed = true;
                        }
                    }
                }
                else if (const auto structKey = as<IRStructKey>(key))
                {
                    auto oldVal = inst->getOperand(0);
                    if (oldVal->getOp() == kIROp_MakeStruct)
                    {
                        // If we see updateElement(makeStruct(...), structKey, ...), we can
                        // replace it with a makeStruct that has the updated value.
                        auto structType = as<IRStructType>(inst->getDataType());
                        if (!structType)
                            break;
                        List<IRInst*> args;
                        UInt i = 0;
                        bool isValid = true;
                        for (auto field : structType->getFields())
                        {
                            IRInst* arg = nullptr;
                            if (i < oldVal->getOperandCount())
                                arg = oldVal->getOperand(i);
                            if (field->getKey() == key)
                                arg = updateInst->getElementValue();
                            if (arg)
                            {
                                args.add(arg);
                            }
                            else
                            {
                                isValid = false;
                                break;
                            }
                            i++;
                        }
                        if (isValid)
                        {
                            IRBuilder builder(module);
                            IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);

                            builder.setInsertBefore(inst);
                            auto makeStruct = builder.emitMakeStruct(
                                structType,
                                (UInt)args.getCount(),
                                args.getBuffer());
                            inst->replaceUsesWith(makeStruct);
                            maybeRemoveOldInst(inst);
                            changed = true;
                        }
                    }
                    else
                    {
                        // Check if the updated `oldVal` is a chain of updateElement insts that
                        // assigns values to every field of the struct, if so, we can just emit a
                        // makeStruct instead.
                        Dictionary<IRStructKey*, IRInst*> mapFieldKeyToVal;
                        for (auto updateElement = as<IRUpdateElement>(inst); updateElement;
                             updateElement = as<IRUpdateElement>(updateElement->getOldValue()))
                        {
                            if (updateElement->getAccessKeyCount() != 1)
                                break;
                            auto subStructKey = as<IRStructKey>(updateElement->getAccessKey(0));
                            if (!subStructKey)
                                break;

                            // If the key already exists, it means there is already a later update
                            // at this key. We need to be careful not to override it with an earlier
                            // value. AddIfNotExists will ensure this does not happen.
                            mapFieldKeyToVal.addIfNotExists(
                                subStructKey,
                                updateElement->getElementValue());
                        }

                        // Check if every field of the struct has a value assigned to it,
                        // while build up arguments for makeStruct inst at the same time.
                        auto structType = as<IRStructType>(inst->getDataType());
                        if (!structType)
                            break;
                        List<IRInst*> args;
                        bool isComplete = true;
                        for (auto field : structType->getFields())
                        {
                            IRInst* arg = nullptr;
                            if (mapFieldKeyToVal.tryGetValue(field->getKey(), arg))
                            {
                                args.add(arg);
                            }
                            else
                            {
                                isComplete = false;
                                break;
                            }
                        }

                        if (!isComplete)
                            break;

                        // Create a makeStruct inst using args.

                        IRBuilder builder(module);
                        IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);

                        builder.setInsertBefore(inst);
                        auto makeStruct = builder.emitMakeStruct(
                            structType,
                            (UInt)args.getCount(),
                            args.getBuffer());
                        inst->replaceUsesWith(makeStruct);
                        maybeRemoveOldInst(inst);
                        changed = true;
                    }
                }
            }
            break;
        case kIROp_CastPtrToBool:
            {
                auto ptr = inst->getOperand(0);
                IRBuilder builder(module);
                IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);

                builder.setInsertBefore(inst);
                auto neq = builder.emitNeq(ptr, builder.getNullPtrValue(ptr->getDataType()));
                inst->replaceUsesWith(neq);
                maybeRemoveOldInst(inst);
                changed = true;
            }
            break;
        case kIROp_IsType:
            {
                auto isTypeInst = as<IRIsType>(inst);
                auto actualType = isTypeInst->getValue()->getDataType();
                if (isTypeEqual(actualType, (IRType*)isTypeInst->getTypeOperand()))
                {
                    IRBuilder builder(module);
                    IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);

                    builder.setInsertBefore(inst);
                    auto trueVal = builder.getBoolValue(true);
                    inst->replaceUsesWith(trueVal);
                    maybeRemoveOldInst(inst);
                    changed = true;
                }
            }
            break;
        case kIROp_Reinterpret:
        case kIROp_BitCast:
        case kIROp_IntCast:
        case kIROp_FloatCast:
            {
                if (isTypeEqual(inst->getOperand(0)->getDataType(), inst->getDataType()))
                {
                    inst->replaceUsesWith(inst->getOperand(0));
                    maybeRemoveOldInst(inst);
                    changed = true;
                }
            }
            break;
        case kIROp_UnpackAnyValue:
            {
                if (inst->getOperand(0)->getOp() == kIROp_PackAnyValue)
                {
                    if (isTypeEqual(
                            inst->getOperand(0)->getOperand(0)->getDataType(),
                            inst->getDataType()))
                    {
                        inst->replaceUsesWith(inst->getOperand(0)->getOperand(0));
                        maybeRemoveOldInst(inst);
                        changed = true;
                    }
                }
            }
            break;
        case kIROp_PackAnyValue:
            {
                // Pack(obj: anyValueN) : anyValueN --> obj
                if (isTypeEqual(inst->getOperand(0)->getDataType(), inst->getDataType()))
                {
                    inst->replaceUsesWith(inst->getOperand(0));
                    maybeRemoveOldInst(inst);
                    changed = true;
                }
            }
            break;
        case kIROp_GetOptionalValue:
            {
                if (inst->getOperand(0)->getOp() == kIROp_MakeOptionalValue)
                {
                    inst->replaceUsesWith(inst->getOperand(0)->getOperand(0));
                    maybeRemoveOldInst(inst);
                    changed = true;
                }
            }
            break;
        case kIROp_OptionalHasValue:
            {
                if (inst->getOperand(0)->getOp() == kIROp_MakeOptionalValue)
                {
                    IRBuilder builder(module);
                    IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);
                    builder.setInsertBefore(inst);
                    auto trueVal = builder.getBoolValue(true);
                    inst->replaceUsesWith(trueVal);
                    maybeRemoveOldInst(inst);
                    changed = true;
                }
                else if (inst->getOperand(0)->getOp() == kIROp_MakeOptionalNone)
                {
                    IRBuilder builder(module);
                    IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);

                    builder.setInsertBefore(inst);
                    auto falseVal = builder.getBoolValue(false);
                    inst->replaceUsesWith(falseVal);
                    maybeRemoveOldInst(inst);
                    changed = true;
                }
            }
            break;
        case kIROp_GetNativePtr:
            {
                if (inst->getOperand(0)->getOp() == kIROp_PtrLit)
                {
                    inst->replaceUsesWith(inst->getOperand(0));
                    maybeRemoveOldInst(inst);
                    changed = true;
                }
            }
            break;
        case kIROp_MakeExistential:
            {
                if (inst->getOperand(0)->getOp() == kIROp_ExtractExistentialValue)
                {
                    inst->replaceUsesWith(inst->getOperand(0)->getOperand(0));
                    maybeRemoveOldInst(inst);
                    changed = true;
                }
            }
            break;
        case kIROp_LookupWitness:
            {
                if (inst->getOperand(0)->getOp() == kIROp_WitnessTable)
                {
                    // Don't fold witness lookups prelinking if the witness table is `extern`.
                    // These witness tables provides `default`s in case they are not
                    // explicitly specialized via other linked modules, therefore we don't want
                    // to resolve them too soon before linking.
                    if (isPrelinking &&
                        inst->getOperand(0)->findDecoration<IRUserExternDecoration>())
                        break;

                    auto wt = as<IRWitnessTable>(inst->getOperand(0));
                    auto key = inst->getOperand(1);
                    for (auto item : wt->getChildren())
                    {
                        if (auto entry = as<IRWitnessTableEntry>(item))
                        {
                            if (entry->getRequirementKey() == key)
                            {
                                auto value = entry->getSatisfyingVal();
                                inst->replaceUsesWith(value);
                                inst->removeAndDeallocate();
                                changed = true;
                                break;
                            }
                        }
                    }
                }
            }
            break;
        case kIROp_DefaultConstruct:
            {
                IRBuilder builder(module);
                IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);
                builder.setInsertBefore(inst);
                // See if we can replace the default construct inst with concrete values.
                if (auto newCtor = builder.emitDefaultConstruct(inst->getFullType(), false))
                {
                    inst->replaceUsesWith(newCtor);
                    maybeRemoveOldInst(inst);
                    changed = true;
                }
            }
            break;
        case kIROp_BuiltinCast:
            {
                IRBuilder builder(module);
                IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);
                builder.setInsertBefore(inst);
                // See if we can replace the default construct inst with concrete values.
                if (auto newCast =
                        builder.emitCast(inst->getFullType(), inst->getOperand(0), false))
                {
                    inst->replaceUsesWith(newCast);
                    maybeRemoveOldInst(inst);
                    changed = true;
                }
            }
            break;
        case kIROp_VectorReshape:
            {
                auto fromType = as<IRVectorType>(inst->getOperand(0)->getDataType());
                if (!fromType)
                    break;
                auto resultType = as<IRVectorType>(inst->getDataType());
                if (!resultType)
                {
                    if (!fromType)
                    {
                        inst->replaceUsesWith(inst->getOperand(0));
                        maybeRemoveOldInst(inst);
                        changed = true;
                        break;
                    }
                    IRBuilder builder(inst);
                    IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);
                    builder.setInsertBefore(inst);
                    UInt index = 0;
                    auto newInst = builder.emitSwizzle(resultType, inst->getOperand(0), 1, &index);
                    inst->replaceUsesWith(newInst);
                    maybeRemoveOldInst(inst);
                    changed = true;
                    break;
                }
                auto fromCount = as<IRIntLit>(fromType->getElementCount());
                if (!fromCount)
                    break;
                auto toCount = as<IRIntLit>(resultType->getElementCount());
                if (!toCount)
                    break;
                IRBuilder builder(inst);
                IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);

                builder.setInsertBefore(inst);
                auto newInst = builder.emitVectorReshape(resultType, inst->getOperand(0));
                if (newInst != inst)
                {
                    inst->replaceUsesWith(newInst);
                    maybeRemoveOldInst(inst);
                    changed = true;
                }
            }
            break;
        case kIROp_MatrixReshape:
            {
                auto fromType = as<IRMatrixType>(inst->getOperand(0)->getDataType());
                auto resultType = as<IRMatrixType>(inst->getDataType());
                SLANG_ASSERT(fromType && resultType);
                auto fromRows = as<IRIntLit>(fromType->getRowCount());
                if (!fromRows)
                    break;
                auto fromCols = as<IRIntLit>(fromType->getColumnCount());
                if (!fromCols)
                    break;
                auto toRows = as<IRIntLit>(resultType->getRowCount());
                if (!toRows)
                    break;
                auto toCols = as<IRIntLit>(resultType->getColumnCount());
                if (!toCols)
                    break;
                List<IRInst*> rows;
                IRBuilder builder(inst);
                IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);
                builder.setInsertBefore(inst);
                auto toRowType = builder.getVectorType(
                    resultType->getElementType(),
                    resultType->getColumnCount());
                for (IRIntegerValue i = 0; i < toRows->getValue(); i++)
                {
                    if (i < fromRows->getValue())
                    {
                        auto originalRow = builder.emitElementExtract(inst->getOperand(0), i);
                        auto resizedRow = builder.emitVectorReshape(toRowType, originalRow);
                        rows.add(resizedRow);
                    }
                    else
                    {
                        auto zero = builder.emitDefaultConstruct(resultType->getElementType());
                        auto row = builder.emitMakeVectorFromScalar(toRowType, zero);
                        rows.add(row);
                    }
                }
                auto newInst =
                    builder.emitMakeMatrix(resultType, (UInt)rows.getCount(), rows.getBuffer());
                inst->replaceUsesWith(newInst);
                maybeRemoveOldInst(inst);
                changed = true;
            }
            break;
        case kIROp_Add:
        case kIROp_Mul:
        case kIROp_Sub:
        case kIROp_Div:
        case kIROp_And:
        case kIROp_Or:
            changed |= tryOptimizeArithmeticInst(inst);
            break;
        case kIROp_Param:
            {
                auto block = as<IRBlock>(inst->parent);
                if (!block)
                    break;
                UInt paramIndex = 0;
                auto prevParam = inst->getPrevInst();
                while (as<IRParam, IRDynamicCastBehavior::NoUnwrap>(prevParam))
                {
                    prevParam = prevParam->getPrevInst();
                    paramIndex++;
                }
                IRInst* argValue = nullptr;
                for (auto pred : block->getPredecessors())
                {
                    auto terminator = as<IRUnconditionalBranch>(pred->getTerminator());
                    if (!terminator)
                        continue;
                    SLANG_ASSERT(terminator->getArgCount() > paramIndex);
                    auto arg = terminator->getArg(paramIndex);
                    if (arg->getOp() == kIROp_undefined)
                        continue;
                    if (argValue == nullptr)
                        argValue = arg;
                    else if (argValue == arg)
                    {
                    }
                    else
                    {
                        argValue = nullptr;
                        break;
                    }
                }
                if (argValue)
                {
                    if (inst->hasUses())
                    {
                        // Is argValue not a local value, i.e. it's not a child
                        // of a block, and it's 'visible' from inst because
                        // inst is a descendent of argValue's parent
                        if (!as<IRBlock>(argValue->getParent()) &&
                            isChildInstOf(inst, argValue->getParent()))
                        {
                            inst->replaceUsesWith(argValue);
                            // Never remove param inst.
                            changed = true;
                        }
                        else if (!useFastAnalysis)
                        {
                            // If argValue is defined locally,
                            // we can replace only if argVal dominates inst.
                            auto parentFunc = getParentFunc(inst);
                            if (!parentFunc)
                                break;

                            auto domTree =
                                parentFunc->getModule()->findOrCreateDominatorTree(parentFunc);

                            if (domTree->dominates(argValue, inst))
                            {
                                inst->replaceUsesWith(argValue);
                                // Never remove param inst.
                                changed = true;
                            }
                        }
                    }
                }
            }
            break;
        case kIROp_swizzle:
            {
                // If we see a swizzle(scalar), we replace it with makeVectorFromScalar.
                if (as<IRBasicType>(inst->getOperand(0)->getDataType()))
                {
                    auto vectorType = as<IRVectorType>(inst->getDataType());
                    IRIntegerValue vectorSize = 1;
                    if (vectorType)
                    {
                        auto sizeLit = as<IRIntLit>(vectorType->getElementCount());
                        if (!sizeLit)
                            vectorSize = 0;
                        vectorSize = sizeLit->getValue();
                    }
                    if (vectorSize == 1)
                    {
                        inst->replaceUsesWith(inst->getOperand(0));
                        maybeRemoveOldInst(inst);
                        break;
                    }
                    IRBuilder builder(module);
                    IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);

                    builder.setInsertBefore(inst);
                    auto newInst =
                        builder.emitMakeVectorFromScalar(vectorType, inst->getOperand(0));
                    inst->replaceUsesWith(newInst);
                    maybeRemoveOldInst(inst);
                    break;
                }
                // If we see a swizzle(makeVector) then we can replace it with the values from
                // makeVector.
                auto makeVector = inst->getOperand(0);
                if (makeVector->getOp() != kIROp_MakeVector)
                    break;
                auto swizzle = as<IRSwizzle>(inst);
                List<IRInst*> vals;
                auto vectorType = as<IRVectorType>(makeVector->getDataType());
                auto vectorSize = as<IRIntLit>(vectorType->getElementCount());
                if (!vectorSize)
                    break;
                if (makeVector->getOperandCount() != (UInt)vectorSize->getValue())
                    break;
                for (UInt i = 0; i < swizzle->getElementCount(); i++)
                {
                    auto index = swizzle->getElementIndex(i);
                    auto intLitIndex = as<IRIntLit>(index);
                    if (!intLitIndex)
                        return;
                    if (intLitIndex->getValue() < (Int)makeVector->getOperandCount())
                        vals.add(makeVector->getOperand((UInt)intLitIndex->getValue()));
                    else
                        return;
                }
                if (vals.getCount() == 1)
                {
                    inst->replaceUsesWith(vals[0]);
                    maybeRemoveOldInst(inst);
                    changed = true;
                }
                else
                {
                    IRBuilder builder(module);
                    IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);

                    builder.setInsertBefore(inst);
                    auto newMakeVector = builder.emitMakeVector(
                        swizzle->getDataType(),
                        (UInt)vals.getCount(),
                        vals.getBuffer());
                    inst->replaceUsesWith(newMakeVector);
                    maybeRemoveOldInst(inst);
                    changed = true;
                }
                break;
            }
        case kIROp_TypeEquals:
            {
                auto getTypeFromOperand = [](IRInst* operand) -> IRType*
                {
                    if (as<IRTypeType>(operand->getFullType()) || !operand->getFullType() ||
                        as<IRTypeKind>(operand->getFullType()))
                        return (IRType*)operand;
                    return operand->getFullType();
                };
                auto left = getTypeFromOperand(inst->getOperand(0));
                auto right = getTypeFromOperand(inst->getOperand(1));
                if (isConcreteType(left) && isConcreteType(right))
                {
                    IRBuilder builder(module);
                    IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);

                    builder.setInsertBefore(inst);
                    bool result = left == right;
                    inst->replaceUsesWith(builder.getBoolValue(result));
                    maybeRemoveOldInst(inst);
                    changed = true;
                }
                break;
            }
        case kIROp_GetNaturalStride:
            {
                if (targetProgram)
                {
                    if (isInGeneric)
                        break;
                    auto type = inst->getOperand(0)->getDataType();
                    IRSizeAndAlignment sizeAlignment;
                    const auto res = getNaturalSizeAndAlignment(
                        targetProgram->getOptionSet(),
                        type,
                        &sizeAlignment);
                    if (!SLANG_SUCCEEDED(res))
                        break;
                    IRBuilder builder(module);
                    IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);

                    builder.setInsertBefore(inst);
                    auto stride =
                        builder.getIntValue(inst->getDataType(), sizeAlignment.getStride());
                    inst->replaceUsesWith(stride);
                    maybeRemoveOldInst(inst);
                    changed = true;
                }
                break;
            }
        case kIROp_IsInt:
        case kIROp_IsFloat:
        case kIROp_IsHalf:
        case kIROp_IsUnsignedInt:
        case kIROp_IsSignedInt:
        case kIROp_IsBool:
        case kIROp_IsVector:
            {
                auto type = inst->getOperand(0)->getDataType();
                if (auto vectorType = as<IRVectorType>(type))
                    type = vectorType->getElementType();
                if (auto matType = as<IRMatrixType>(type))
                    type = matType->getElementType();
                if (isConcreteType(type))
                {
                    IRBuilder builder(module);
                    IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);

                    builder.setInsertBefore(inst);
                    bool result = false;
                    switch (inst->getOp())
                    {
                    case kIROp_IsInt:
                        result = isIntegralType(type);
                        break;
                    case kIROp_IsBool:
                        result = type->getOp() == kIROp_BoolType;
                        break;
                    case kIROp_IsFloat:
                        result = isFloatingType(type);
                        break;
                    case kIROp_IsHalf:
                        result = type->getOp() == kIROp_HalfType;
                        break;
                    case kIROp_IsUnsignedInt:
                        result = isIntegralType(type) && !getIntTypeInfo(type).isSigned;
                        break;
                    case kIROp_IsSignedInt:
                        result = isIntegralType(type) && getIntTypeInfo(type).isSigned;
                        break;
                    case kIROp_IsVector:
                        result = as<IRVectorType>(type);
                        break;
                    }
                    inst->replaceUsesWith(builder.getBoolValue(result));
                    maybeRemoveOldInst(inst);
                    changed = true;
                }
                break;
            }
        case kIROp_Load:
            {
                // Load from undef is undef.
                if (as<IRLoad>(inst)->getPtr()->getOp() == kIROp_undefined)
                {
                    IRBuilder builder(module);
                    IRBuilderSourceLocRAII srcLocRAII(&builder, inst->sourceLoc);

                    builder.setInsertBefore(inst);
                    auto undef = builder.emitUndefined(inst->getDataType());
                    inst->replaceUsesWith(undef);
                    maybeRemoveOldInst(inst);
                    changed = true;
                }
                break;
            }
        case kIROp_Store:
            {
                // Store undef is no-op.
                if (as<IRStore>(inst)->getVal()->getOp() == kIROp_undefined)
                {
                    maybeRemoveOldInst(inst);
                    changed = true;
                }
                break;
            }
        case kIROp_DebugValue:
            {
                // Update debug value with undef is no-op.
                if (as<IRDebugValue>(inst)->getValue()->getOp() == kIROp_undefined)
                {
                    maybeRemoveOldInst(inst);
                    changed = true;
                }
                break;
            }
        default:
            break;
        }
    }

    bool isConcreteType(IRType* type)
    {
        return type->parent->getOp() == kIROp_Module && !as<IRGlobalGenericParam>(type);
    }

    bool processFunc(IRInst* func)
    {
        if (!useFastAnalysis)
            func->getModule()->invalidateAllAnalysis();

        bool lastIsInGeneric = isInGeneric;
        if (!isInGeneric)
            isInGeneric = as<IRGeneric>(func) != nullptr;

        bool result = false;
        for (;;)
        {
            changed = false;
            processChildInsts(func, [this](IRInst* inst) { processInst(inst); });
            if (changed)
                result = true;
            else
                break;
        }

        isInGeneric = lastIsInGeneric;

        return result;
    }

    bool processModule() { return processFunc(module->getModuleInst()); }
};

bool peepholeOptimize(TargetProgram* target, IRModule* module, PeepholeOptimizationOptions options)
{
    PeepholeContext context = PeepholeContext(module);
    context.targetProgram = target;
    context.isPrelinking = options.isPrelinking;
    context.useFastAnalysis =
        target ? target->getOptionSet().getBoolOption(CompilerOptionName::MinimumSlangOptimization)
               : true;
    return context.processModule();
}

bool peepholeOptimize(TargetProgram* target, IRInst* func)
{
    PeepholeContext context = PeepholeContext(func->getModule());
    context.targetProgram = target;
    context.useFastAnalysis =
        target ? target->getOptionSet().getBoolOption(CompilerOptionName::MinimumSlangOptimization)
               : true;
    return context.processFunc(func);
}

bool peepholeOptimizeInst(TargetProgram* target, IRModule* module, IRInst* inst)
{
    PeepholeContext context = PeepholeContext(module);
    context.targetProgram = target;
    context.useFastAnalysis = true;
    context.processInst(inst);
    return context.changed;
}

bool peepholeOptimizeGlobalScope(TargetProgram* target, IRModule* module)
{
    PeepholeContext context = PeepholeContext(module);
    context.targetProgram = target;
    context.useFastAnalysis = true;
    bool result = false;
    for (;;)
    {
        context.changed = false;
        for (auto globalInst : module->getGlobalInsts())
            context.processInst(globalInst);
        result |= context.changed;
        if (!context.changed)
            break;
    }
    return result;
}

bool tryReplaceInstUsesWithSimplifiedValue(TargetProgram* target, IRModule* module, IRInst* inst)
{
    if (inst != tryConstantFoldInst(module, inst))
        return true;

    PeepholeContext context = PeepholeContext(inst->getModule());
    context.targetProgram = target;
    context.removeOldInst = false;
    context.useFastAnalysis = true;
    context.processInst(inst);
    return context.changed;
}

} // namespace Slang
