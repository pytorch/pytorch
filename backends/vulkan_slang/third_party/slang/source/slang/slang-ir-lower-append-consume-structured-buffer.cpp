#include "slang-ir-lower-append-consume-structured-buffer.h"

#include "slang-ir-insts.h"
#include "slang-ir-layout.h"
#include "slang-ir-lower-buffer-element-type.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{
static void lowerStructuredBufferType(TargetProgram* target, IRHLSLStructuredBufferTypeBase* type)
{
    IRBuilder builder(type);
    builder.setInsertBefore(type);

    auto elementType = type->getElementType();

    // Type.
    auto structType = builder.createStructType();
    StringBuilder nameSb;
    if (type->getOp() == kIROp_HLSLAppendStructuredBufferType)
        nameSb << "AppendStructuredBuffer<";
    else
        nameSb << "ConsumeStructuredBuffer<";
    getTypeNameHint(nameSb, elementType);
    nameSb << ">";
    builder.addNameHintDecoration(structType, nameSb.produceString().getUnownedSlice());

    auto elementBufferKey = builder.createStructKey();
    builder.addNameHintDecoration(elementBufferKey, UnownedStringSlice("elements"));

    auto counterBufferKey = builder.createStructKey();
    builder.addNameHintDecoration(counterBufferKey, UnownedStringSlice("counter"));

    builder.addDecoration(elementBufferKey, kIROp_CounterBufferDecoration, counterBufferKey);

    IRInst* operands[2] = {elementType, type->getDataLayout()};
    auto elementBufferType = builder.getType(kIROp_HLSLRWStructuredBufferType, 2, operands);

    operands[0] = builder.getIntType();
    operands[1] = builder.getType(kIROp_DefaultBufferLayoutType);
    auto counterBufferType = builder.getType(kIROp_HLSLRWStructuredBufferType, 2, operands);

    builder.createStructField(structType, elementBufferKey, elementBufferType);
    builder.createStructField(structType, counterBufferKey, counterBufferType);

    // Type layout.
    auto layoutRules = getTypeLayoutRuleForBuffer(target, type);

    IRTypeLayout::Builder elementTypeLayoutBuilder(&builder);
    IRSizeAndAlignment elementSize;
    getSizeAndAlignment(target->getOptionSet(), layoutRules, elementType, &elementSize);
    elementTypeLayoutBuilder.addResourceUsage(
        LayoutResourceKind::Uniform,
        LayoutSize((LayoutSize::RawValue)elementSize.getStride()));
    auto elementTypeLayout = elementTypeLayoutBuilder.build();

    IRStructuredBufferTypeLayout::Builder elementBufferTypeLayoutBuilder(
        &builder,
        elementTypeLayout);
    elementBufferTypeLayoutBuilder.addResourceUsage(LayoutResourceKind::DescriptorTableSlot, 1);
    auto elementBufferTypeLayout = elementBufferTypeLayoutBuilder.build();

    IRTypeLayout::Builder counterTypeLayoutBuilder(&builder);
    counterTypeLayoutBuilder.addResourceUsage(LayoutResourceKind::Uniform, LayoutSize(4));
    auto counterTypeLayout = counterTypeLayoutBuilder.build();

    IRStructuredBufferTypeLayout::Builder counterBufferTypeLayoutBuilder(
        &builder,
        counterTypeLayout);
    counterBufferTypeLayoutBuilder.addResourceUsage(LayoutResourceKind::DescriptorTableSlot, 1);
    auto counterBufferTypeLayout = counterBufferTypeLayoutBuilder.build();

    IRVarLayout::Builder elementBufferVarLayoutBuilder(&builder, elementBufferTypeLayout);
    elementBufferVarLayoutBuilder.findOrAddResourceInfo(LayoutResourceKind::DescriptorTableSlot)
        ->offset = 0;
    auto elementBufferVarLayout = elementBufferVarLayoutBuilder.build();

    IRVarLayout::Builder counterBufferVarLayoutBuilder(&builder, counterBufferTypeLayout);
    counterBufferVarLayoutBuilder.findOrAddResourceInfo(LayoutResourceKind::DescriptorTableSlot)
        ->offset = 1;
    auto counterBufferVarLayout = counterBufferVarLayoutBuilder.build();

    IRStructTypeLayout::Builder layoutBuilder(&builder);
    layoutBuilder.addField(elementBufferKey, elementBufferVarLayout);
    layoutBuilder.addField(counterBufferKey, counterBufferVarLayout);
    auto typeLayout = layoutBuilder.build();

    builder.addLayoutDecoration(structType, typeLayout);

    IRFunc* appendFunc = nullptr;
    IRFunc* consumeFunc = nullptr;
    IRFunc* getDimensionsFunc = nullptr;

    if (type->getOp() == kIROp_HLSLAppendStructuredBufferType)
    {
        // Append method.
        appendFunc = builder.createFunc();
        builder.addNameHintDecoration(
            appendFunc,
            UnownedStringSlice("AppendStructuredBuffer_Append"));
        IRType* paramTypes[] = {structType, elementType};
        auto funcType = builder.getFuncType(2, paramTypes, builder.getVoidType());
        appendFunc->setFullType(funcType);
        builder.setInsertInto(appendFunc);
        builder.emitBlock();
        auto bufferParam = builder.emitParam(structType);
        auto elementParam = builder.emitParam(elementType);
        auto elementBuffer =
            builder.emitFieldExtract(elementBufferType, bufferParam, elementBufferKey);
        auto counterBuffer =
            builder.emitFieldExtract(counterBufferType, bufferParam, counterBufferKey);
        IRInst* getCounterPtrArgs[] = {counterBuffer, builder.getIntValue(builder.getIntType(), 0)};
        auto counterBufferPtr = builder.emitIntrinsicInst(
            builder.getPtrType(builder.getIntType()),
            kIROp_RWStructuredBufferGetElementPtr,
            2,
            getCounterPtrArgs);
        IRInst* atomicIncArgs[] = {
            counterBufferPtr,
            builder.getIntValue(builder.getIntType(), kIRMemoryOrder_Relaxed)};
        auto oldCounter =
            builder.emitIntrinsicInst(builder.getIntType(), kIROp_AtomicInc, 2, atomicIncArgs);

        IRInst* getElementPtrArgs[] = {elementBuffer, oldCounter};
        auto elementBufferPtr = builder.emitIntrinsicInst(
            builder.getPtrType(elementType),
            kIROp_RWStructuredBufferGetElementPtr,
            2,
            getElementPtrArgs);

        builder.emitStore(elementBufferPtr, elementParam);
        builder.emitReturn();
    }
    else
    {
        // Consume method.
        consumeFunc = builder.createFunc();
        builder.addNameHintDecoration(
            consumeFunc,
            UnownedStringSlice("ConsumeStructuredBuffer_Consume"));
        IRType* paramTypes[] = {structType};
        auto funcType = builder.getFuncType(1, paramTypes, elementType);
        consumeFunc->setFullType(funcType);
        builder.setInsertInto(consumeFunc);
        auto firstBlock = builder.emitBlock();
        auto bufferParam = builder.emitParam(structType);
        auto elementBuffer =
            builder.emitFieldExtract(elementBufferType, bufferParam, elementBufferKey);
        auto counterBuffer =
            builder.emitFieldExtract(counterBufferType, bufferParam, counterBufferKey);
        IRInst* getCounterPtrArgs[] = {counterBuffer, builder.getIntValue(builder.getIntType(), 0)};
        auto counterBufferPtr = builder.emitIntrinsicInst(
            builder.getPtrType(builder.getIntType()),
            kIROp_RWStructuredBufferGetElementPtr,
            2,
            getCounterPtrArgs);
        IRInst* atomicDecArgs[] = {
            counterBufferPtr,
            builder.getIntValue(builder.getIntType(), kIRMemoryOrder_Relaxed)};
        auto oldCounter =
            builder.emitIntrinsicInst(builder.getIntType(), kIROp_AtomicDec, 2, atomicDecArgs);
        auto index = builder.emitSub(
            builder.getIntType(),
            oldCounter,
            builder.getIntValue(builder.getIntType(), 1));

        // Test if index is greater or equal than 0.
        auto geq = builder.emitGeq(index, builder.getIntValue(builder.getIntType(), 0));
        auto trueBlock = builder.emitBlock();

        auto falseBlock = builder.emitBlock();
        auto mergeBlock = builder.emitBlock();

        builder.setInsertInto(firstBlock);
        builder.emitIfElse(geq, trueBlock, falseBlock, mergeBlock);

        builder.setInsertInto(trueBlock);
        IRInst* getElementPtrArgs[] = {elementBuffer, index};
        auto elementBufferPtr = builder.emitIntrinsicInst(
            builder.getPtrType(elementType),
            kIROp_RWStructuredBufferGetElementPtr,
            2,
            getElementPtrArgs);
        auto val = builder.emitLoad(elementBufferPtr);
        builder.emitReturn(val);

        builder.setInsertInto(falseBlock);
        auto defaultVal = builder.emitDefaultConstruct(elementType);
        builder.emitReturn(defaultVal);

        builder.setInsertInto(mergeBlock);
        builder.emitUnreachable();
    }

    // GetDimensions method.
    {
        getDimensionsFunc = builder.createFunc();
        builder.addNameHintDecoration(
            getDimensionsFunc,
            UnownedStringSlice("StructuredBuffer_GetDimensions"));
        IRType* paramTypes[] = {structType};
        auto uint2Type = builder.getVectorType(builder.getUIntType(), 2);
        auto funcType = builder.getFuncType(1, paramTypes, uint2Type);
        getDimensionsFunc->setFullType(funcType);
        builder.setInsertInto(getDimensionsFunc);
        builder.emitBlock();
        auto bufferParam = builder.emitParam(structType);
        auto elementBuffer =
            builder.emitFieldExtract(elementBufferType, bufferParam, elementBufferKey);

        const auto dim = builder.emitIntrinsicInst(
            uint2Type,
            kIROp_StructuredBufferGetDimensions,
            1,
            &elementBuffer);
        builder.emitReturn(dim);
    }

    // Replace all insts with synthesized functions.
    traverseUsers(
        type,
        [&](IRInst* typeUser)
        {
            if (typeUser->getFullType() != type)
                return;
            if (auto layoutDecor = typeUser->findDecoration<IRLayoutDecoration>())
            {
                // Replace the original StructuredBufferVarLayout with the new StructTypeVarLayout.
                if (auto varLayout = as<IRVarLayout>(layoutDecor->getLayout()))
                {
                    IRBuilder subBuilder(typeUser);
                    IRVarLayout::Builder newVarLayoutBuilder(&subBuilder, typeLayout);
                    newVarLayoutBuilder.cloneEverythingButOffsetsFrom(varLayout);
                    IRVarOffsetAttr* uavOffsetAttr = nullptr;
                    IRVarOffsetAttr* descriptorTableSlotOffsetAttr = nullptr;

                    for (auto offsetAttr : varLayout->getOffsetAttrs())
                    {
                        if (offsetAttr->getResourceKind() == LayoutResourceKind::UnorderedAccess)
                            uavOffsetAttr = offsetAttr;
                        else if (
                            offsetAttr->getResourceKind() ==
                            LayoutResourceKind::DescriptorTableSlot)
                            descriptorTableSlotOffsetAttr = offsetAttr;
                        auto info = newVarLayoutBuilder.findOrAddResourceInfo(
                            offsetAttr->getResourceKind());
                        info->offset = offsetAttr->getOffset();
                        info->space = offsetAttr->getSpace();
                        info->kind = offsetAttr->getResourceKind();
                    }
                    // If the user provided an layout offset for UAV but not for descriptor table
                    // slot, then we use the UAV offset for the descriptor table slot offset.
                    if (uavOffsetAttr && !descriptorTableSlotOffsetAttr)
                    {
                        auto info = newVarLayoutBuilder.findOrAddResourceInfo(
                            LayoutResourceKind::DescriptorTableSlot);
                        info->offset = uavOffsetAttr->getOffset();
                        info->space = uavOffsetAttr->getSpace();
                        info->kind = LayoutResourceKind::DescriptorTableSlot;
                    }
                    auto newVarLayout = newVarLayoutBuilder.build();
                    subBuilder.addLayoutDecoration(typeUser, newVarLayout);
                    varLayout->removeAndDeallocate();
                }
            }
            traverseUses(
                typeUser,
                [&](IRUse* use)
                {
                    auto user = use->getUser();
                    switch (user->getOp())
                    {
                    case kIROp_StructuredBufferAppend:
                        {
                            IRBuilder subBuilder(user);
                            subBuilder.setInsertBefore(user);
                            IRInst* args[] = {user->getOperand(0), user->getOperand(1)};
                            auto call =
                                subBuilder.emitCallInst(user->getFullType(), appendFunc, 2, args);
                            user->replaceUsesWith(call);
                            user->removeAndDeallocate();
                            break;
                        }
                    case kIROp_StructuredBufferConsume:
                        {
                            IRBuilder subBuilder(user);
                            subBuilder.setInsertBefore(user);
                            IRInst* args[] = {user->getOperand(0)};
                            auto call =
                                subBuilder.emitCallInst(user->getFullType(), consumeFunc, 1, args);
                            user->replaceUsesWith(call);
                            user->removeAndDeallocate();
                            break;
                        }
                    case kIROp_StructuredBufferGetDimensions:
                        {
                            IRBuilder subBuilder(user);
                            subBuilder.setInsertBefore(user);
                            IRInst* args[] = {user->getOperand(0)};
                            auto call = subBuilder.emitCallInst(
                                user->getFullType(),
                                getDimensionsFunc,
                                1,
                                args);
                            user->replaceUsesWith(call);
                            user->removeAndDeallocate();
                            break;
                        }
                    }
                });
        });
    type->replaceUsesWith(structType);
}

void lowerAppendConsumeStructuredBuffers(
    TargetProgram* target,
    IRModule* module,
    DiagnosticSink* sink)
{
    SLANG_UNUSED(sink);
    for (auto globalInst : module->getGlobalInsts())
    {
        switch (globalInst->getOp())
        {
        case kIROp_HLSLAppendStructuredBufferType:
        case kIROp_HLSLConsumeStructuredBufferType:
            lowerStructuredBufferType(target, as<IRHLSLStructuredBufferTypeBase>(globalInst));
            break;
        }
    }
}
} // namespace Slang
