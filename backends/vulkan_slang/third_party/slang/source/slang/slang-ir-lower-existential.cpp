// slang-ir-lower-generic-existential.cpp

#include "slang-ir-lower-existential.h"

#include "slang-ir-generics-lowering-context.h"
#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{
bool isCPUTarget(TargetRequest* targetReq);
bool isCUDATarget(TargetRequest* targetReq);

struct ExistentialLoweringContext
{
    SharedGenericsLoweringContext* sharedContext;

    void processMakeExistential(IRMakeExistentialWithRTTI* inst)
    {
        IRBuilder builderStorage(sharedContext->module);
        auto builder = &builderStorage;
        builder->setInsertBefore(inst);
        auto value = inst->getWrappedValue();
        auto valueType = sharedContext->lowerType(builder, value->getDataType());
        if (valueType->getOp() == kIROp_ComPtrType)
            return;
        auto witnessTableType =
            cast<IRWitnessTableTypeBase>(inst->getWitnessTable()->getDataType());
        auto interfaceType = witnessTableType->getConformanceType();
        if (interfaceType->findDecoration<IRComInterfaceDecoration>())
            return;
        auto witnessTableIdType = builder->getWitnessTableIDType((IRType*)interfaceType);
        auto anyValueSize = sharedContext->getInterfaceAnyValueSize(interfaceType, inst->sourceLoc);
        auto anyValueType = builder->getAnyValueType(anyValueSize);
        auto rttiType = builder->getRTTIHandleType();
        auto tupleType = builder->getTupleType(rttiType, witnessTableIdType, anyValueType);

        IRInst* rttiObject = inst->getRTTI();
        if (auto type = as<IRType>(rttiObject))
        {
            rttiObject = sharedContext->maybeEmitRTTIObject(type);
            rttiObject = builder->emitGetAddress(rttiType, rttiObject);
        }
        IRInst* packedValue = value;
        if (valueType->getOp() != kIROp_AnyValueType)
            packedValue = builder->emitPackAnyValue(anyValueType, value);
        IRInst* tupleArgs[] = {rttiObject, inst->getWitnessTable(), packedValue};
        auto tuple = builder->emitMakeTuple(tupleType, 3, tupleArgs);
        inst->replaceUsesWith(tuple);
        inst->removeAndDeallocate();
    }

    // Translates `createExistentialObject` insts, which takes a user defined
    // type id and user defined value and turns into an existential value,
    // into a `makeTuple` inst that makes the tuple representing the lowered
    // existential value.
    void processCreateExistentialObject(IRCreateExistentialObject* inst)
    {
        IRBuilder builderStorage(sharedContext->module);
        auto builder = &builderStorage;
        builder->setInsertBefore(inst);

        // The result type of this `createExistentialObject` inst should already
        // be lowered into a `TupleType(rttiType, WitnessTableIDType, AnyValueType)`
        // in the previous `lowerGenericType` pass.
        auto tupleType = inst->getDataType();
        auto witnessTableIdType = cast<IRWitnessTableIDType>(tupleType->getOperand(1));
        auto anyValueType = cast<IRAnyValueType>(tupleType->getOperand(2));

        // Create a null value for `rttiObject` for now since it will not be used.
        auto uint2Type = builder->getVectorType(
            builder->getUIntType(),
            builder->getIntValue(builder->getIntType(), 2));
        IRInst* zero = builder->getIntValue(builder->getUIntType(), 0);
        IRInst* zeroVectorArgs[] = {zero, zero};
        IRInst* rttiObject = builder->emitMakeVector(uint2Type, 2, zeroVectorArgs);

        // Pack the user provided value into `AnyValue`.
        IRInst* packedValue = inst->getValue();
        if (packedValue->getDataType()->getOp() != kIROp_AnyValueType)
            packedValue = builder->emitPackAnyValue(anyValueType, packedValue);

        // Use the user provided `typeID` value as the witness table ID field in the
        // newly constructed tuple.
        // All `WitnessTableID` types are lowered into `uint2`s, so we need to create
        // a `uint2` value from `typeID` to stay consistent with the convention.
        IRInst* vectorArgs[2] = {
            inst->getTypeID(),
            builder->getIntValue(builder->getUIntType(), 0)};

        IRInst* typeIdValue = builder->emitMakeVector(uint2Type, 2, vectorArgs);
        typeIdValue = builder->emitBitCast(witnessTableIdType, typeIdValue);
        IRInst* tupleArgs[] = {rttiObject, typeIdValue, packedValue};
        auto tuple = builder->emitMakeTuple(tupleType, 3, tupleArgs);
        inst->replaceUsesWith(tuple);
        inst->removeAndDeallocate();
    }

    IRInst* extractTupleElement(IRBuilder* builder, IRInst* value, UInt index)
    {
        auto tupleType = cast<IRTupleType>(sharedContext->lowerType(builder, value->getDataType()));
        auto getElement =
            builder->emitGetTupleElement((IRType*)tupleType->getOperand(index), value, index);
        return getElement;
    }

    void processExtractExistentialElement(IRInst* extractInst, UInt elementId)
    {
        IRBuilder builderStorage(sharedContext->module);
        auto builder = &builderStorage;
        builder->setInsertBefore(extractInst);

        IRInst* element = nullptr;
        if (isComInterfaceType(extractInst->getOperand(0)->getDataType()))
        {
            // If this is an COM interface, the elements (witness table/rtti) are just the interface
            // value itself.
            element = extractInst->getOperand(0);
        }
        else
        {
            element = extractTupleElement(builder, extractInst->getOperand(0), elementId);
        }
        extractInst->replaceUsesWith(element);
        extractInst->removeAndDeallocate();
    }

    void processExtractExistentialValue(IRExtractExistentialValue* inst)
    {
        processExtractExistentialElement(inst, 2);
    }

    void processExtractExistentialWitnessTable(IRExtractExistentialWitnessTable* inst)
    {
        processExtractExistentialElement(inst, 1);
    }

    void processExtractExistentialType(IRExtractExistentialType* extractInst)
    {
        IRBuilder builderStorage(sharedContext->module);
        auto builder = &builderStorage;
        builder->setInsertBefore(extractInst);

        IRInst* element = nullptr;
        IRInst* anyValueType = nullptr;
        if (isComInterfaceType(extractInst->getOperand(0)->getDataType()))
        {
            // If this is an COM interface, the elements (witness table/rtti) are just the interface
            // value itself.
            element = extractInst->getOperand(0);
        }
        else
        {
            element = extractTupleElement(builder, extractInst->getOperand(0), 0);
            if (auto tupleType = as<IRTupleType>(extractInst->getOperand(0)->getDataType()))
            {
                anyValueType = tupleType->getOperand(2);
            }
        }

        // If this instruction is used as a type, we need to replace it with the lowered type,
        // which should be an AnyValueType.
        // If it is used as a value, then we can replace it with the extracted element.
        auto isTypeUse = [](IRUse* use) -> bool
        {
            auto user = use->getUser();
            if (as<IRType>(user))
                return true;
            if (use == &use->getUser()->typeUse)
                return true;
            return false;
        };
        traverseUses(
            extractInst,
            [&](IRUse* use)
            {
                if (anyValueType && isTypeUse(use))
                {
                    builder->replaceOperand(use, anyValueType);
                    return;
                }
                builder->replaceOperand(use, element);
            });
    }

    void processGetValueFromBoundInterface(IRGetValueFromBoundInterface* inst)
    {
        IRBuilder builderStorage(sharedContext->module);
        auto builder = &builderStorage;
        builder->setInsertBefore(inst);
        if (inst->getDataType()->getOp() == kIROp_ClassType)
        {
            return;
        }
        // A value of interface will lower as a tuple, and
        // the third element of that tuple represents the
        // concrete value that was put into the existential.
        //
        auto element = extractTupleElement(builder, inst->getOperand(0), 2);
        auto elementType = element->getDataType();

        // There are two cases we expect to see for that
        // tuple element.
        //
        IRInst* replacement = nullptr;
        if (as<IRPseudoPtrType>(elementType))
        {
            // The first case is when legacy static specialization
            // is applied, and the element is a "pseudo-pointer."
            //
            // Semantically, we should emit a (pseudo-)load from the pseudo-pointer
            // to go from `PseudoPtr<T>` to `T`.
            //
            // TODO: Actually introduce and emit a "psedudo-load" instruction
            // here. For right now we are just using the value directly and
            // downstream passes seem okay with it, but it isn't really
            // type-correct to be doing this.
            //
            replacement = element;
        }
        else
        {
            // The second case is when the dynamic-dispatch layout is
            // being used, and the element is an "any-value."
            //
            // In this case we need to emit an unpacking operation
            // to get from `AnyValue` to `T`.
            //
            SLANG_ASSERT(as<IRAnyValueType>(elementType));
            replacement = builder->emitUnpackAnyValue(inst->getFullType(), element);
        }

        inst->replaceUsesWith(replacement);
        inst->removeAndDeallocate();
    }

    void processInst(IRInst* inst)
    {
        if (auto makeExistential = as<IRMakeExistentialWithRTTI>(inst))
        {
            processMakeExistential(makeExistential);
        }
        else if (auto createExistentialObject = as<IRCreateExistentialObject>(inst))
        {
            processCreateExistentialObject(createExistentialObject);
        }
        else if (auto getValueFromBoundInterface = as<IRGetValueFromBoundInterface>(inst))
        {
            processGetValueFromBoundInterface(getValueFromBoundInterface);
        }
        else if (auto extractExistentialVal = as<IRExtractExistentialValue>(inst))
        {
            processExtractExistentialValue(extractExistentialVal);
        }
        else if (auto extractExistentialType = as<IRExtractExistentialType>(inst))
        {
            processExtractExistentialType(extractExistentialType);
        }
        else if (auto extractExistentialWitnessTable = as<IRExtractExistentialWitnessTable>(inst))
        {
            processExtractExistentialWitnessTable(extractExistentialWitnessTable);
        }
    }

    void processModule()
    {
        sharedContext->addToWorkList(sharedContext->module->getModuleInst());

        while (sharedContext->workList.getCount() != 0)
        {
            IRInst* inst = sharedContext->workList.getLast();

            sharedContext->workList.removeLast();
            sharedContext->workListSet.remove(inst);

            processInst(inst);

            for (auto child = inst->getLastChild(); child; child = child->getPrevInst())
            {
                sharedContext->addToWorkList(child);
            }
        }
    }
};


void lowerExistentials(SharedGenericsLoweringContext* sharedContext)
{
    ExistentialLoweringContext context;
    context.sharedContext = sharedContext;
    context.processModule();
}
} // namespace Slang
