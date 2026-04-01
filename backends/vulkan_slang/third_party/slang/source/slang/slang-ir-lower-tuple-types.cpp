// slang-ir-lower-tuple-types.cpp

#include "slang-ir-lower-tuple-types.h"

#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{
struct TupleLoweringContext
{
    IRModule* module;
    DiagnosticSink* sink;

    InstWorkList workList;
    InstHashSet workListSet;

    TupleLoweringContext(IRModule* inModule)
        : module(inModule), workList(inModule), workListSet(inModule)
    {
    }

    struct LoweredTupleInfo : public RefObject
    {
        IRType* tupleType;
        IRStructType* structType;
        List<IRStructField*> fields;
    };
    Dictionary<IRInst*, RefPtr<LoweredTupleInfo>> mapLoweredStructToTupleInfo;
    Dictionary<IRInst*, RefPtr<LoweredTupleInfo>> loweredTuples;

    IRType* maybeLowerTupleType(IRBuilder* builder, IRType* type)
    {
        if (auto info = getLoweredTupleType(builder, type))
            return info->structType;
        else
            return type;
    }

    LoweredTupleInfo* getLoweredTupleType(IRBuilder* builder, IRInst* type)
    {
        if (auto loweredInfo = loweredTuples.tryGetValue(type))
            return loweredInfo->Ptr();
        if (auto loweredInfo = mapLoweredStructToTupleInfo.tryGetValue(type))
            return loweredInfo->Ptr();

        if (!type)
            return nullptr;
        if (type->getOp() != kIROp_TupleType)
            return nullptr;

        RefPtr<LoweredTupleInfo> info = new LoweredTupleInfo();
        info->tupleType = (IRType*)type;
        auto structType = builder->createStructType();
        info->structType = structType;
        builder->addNameHintDecoration(structType, UnownedStringSlice("Tuple"));

        StringBuilder fieldNameSb;
        for (UInt i = 0; i < type->getOperandCount(); i++)
        {
            auto elementType = maybeLowerTupleType(builder, (IRType*)(type->getOperand(i)));
            auto key = builder->createStructKey();
            fieldNameSb.clear();
            fieldNameSb << "value" << i;
            builder->addNameHintDecoration(key, fieldNameSb.getUnownedSlice());
            auto field = builder->createStructField(structType, key, (IRType*)elementType);
            info->fields.add(field);
        }
        mapLoweredStructToTupleInfo[structType] = info;
        loweredTuples[type] = info;
        return info.Ptr();
    }

    void addToWorkList(IRInst* inst)
    {
        for (auto ii = inst->getParent(); ii; ii = ii->getParent())
        {
            if (as<IRGeneric>(ii))
                return;
        }

        if (workListSet.contains(inst))
            return;

        workList.add(inst);
        workListSet.add(inst);
    }

    void processMakeTuple(IRInst* inst)
    {
        IRBuilder builderStorage(module);
        auto builder = &builderStorage;
        builder->setInsertBefore(inst);

        auto info = getLoweredTupleType(builder, inst->getDataType());
        List<IRInst*> operands;
        for (Index i = 0; i < info->fields.getCount(); i++)
        {
            SLANG_ASSERT(i < (Index)inst->getOperandCount());
            operands.add(inst->getOperand(i));
        }
        auto makeStruct = builder->emitMakeStruct(info->structType, operands);
        inst->replaceUsesWith(makeStruct);
        inst->removeAndDeallocate();
    }

    void processGetTupleElement(IRGetTupleElement* inst)
    {
        IRBuilder builderStorage(module);
        auto builder = &builderStorage;
        builder->setInsertBefore(inst);

        auto base = inst->getTuple();
        auto loweredTupleInfo = getLoweredTupleType(builder, base->getDataType());
        SLANG_ASSERT(loweredTupleInfo);
        auto elementIndex = getIntVal(inst->getElementIndex());
        SLANG_ASSERT((Index)elementIndex < loweredTupleInfo->fields.getCount());

        auto field = loweredTupleInfo->fields[(Index)elementIndex];
        auto getElement = builder->emitFieldExtract(field->getFieldType(), base, field->getKey());
        inst->replaceUsesWith(getElement);
        inst->removeAndDeallocate();
    }

    void processGetElementPtr(IRGetElementPtr* inst)
    {
        IRBuilder builder(module);
        builder.setInsertBefore(inst);

        auto base = inst->getBase();
        auto baseValueType = tryGetPointedToType(&builder, base->getDataType());
        auto loweredTupleInfo = getLoweredTupleType(&builder, baseValueType);
        if (!loweredTupleInfo)
            return;

        auto elementIndex = getIntVal(inst->getIndex());
        SLANG_ASSERT((Index)elementIndex < loweredTupleInfo->fields.getCount());

        auto field = loweredTupleInfo->fields[(Index)elementIndex];
        auto getElement = builder.emitFieldAddress(
            builder.getPtrType(field->getFieldType()),
            base,
            field->getKey());
        inst->replaceUsesWith(getElement);
        inst->removeAndDeallocate();
    }

    void processSwizzle(IRSwizzle* inst)
    {
        IRBuilder builder(module);
        builder.setInsertBefore(inst);

        auto base = inst->getBase();
        auto loweredTupleInfo = getLoweredTupleType(&builder, base->getDataType());

        if (!loweredTupleInfo)
            return;

        if (inst->getElementCount() == 1)
        {
            auto elementIndex = getIntVal(inst->getElementIndex(0));
            SLANG_ASSERT((Index)elementIndex < loweredTupleInfo->fields.getCount());

            auto field = loweredTupleInfo->fields[(Index)elementIndex];
            auto getElement =
                builder.emitFieldExtract(field->getFieldType(), base, field->getKey());
            inst->replaceUsesWith(getElement);
            inst->removeAndDeallocate();
        }
        else
        {
            List<IRInst*> elements;
            for (UInt i = 0; i < inst->getElementCount(); i++)
            {
                auto elementIndex = getIntVal(inst->getElementIndex(i));
                SLANG_ASSERT((Index)elementIndex < loweredTupleInfo->fields.getCount());

                auto field = loweredTupleInfo->fields[(Index)elementIndex];
                auto getElement =
                    builder.emitFieldExtract(field->getFieldType(), base, field->getKey());
                elements.add(getElement);
            }
            auto resultTypeInfo = getLoweredTupleType(&builder, inst->getDataType());
            auto makeStruct = builder.emitMakeStruct(resultTypeInfo->structType, elements);
            inst->replaceUsesWith(makeStruct);
            inst->removeAndDeallocate();
        }
    }

    void processSwizzleSet(IRSwizzleSet* inst)
    {
        IRBuilder builder(module);
        builder.setInsertBefore(inst);

        auto base = inst->getBase();
        auto loweredTupleInfo = getLoweredTupleType(&builder, base->getDataType());
        auto sourceTupleInfo = getLoweredTupleType(&builder, inst->getSource()->getDataType());
        if (!loweredTupleInfo)
            return;

        List<IRInst*> elements;
        for (Index i = 0; i < loweredTupleInfo->fields.getCount(); i++)
        {
            auto field = loweredTupleInfo->fields[i];
            auto getElement =
                builder.emitFieldExtract(field->getFieldType(), base, field->getKey());
            elements.add(getElement);
        }

        for (UInt i = 0; i < inst->getElementCount(); i++)
        {
            auto baseIndex = getIntVal(inst->getElementIndex(i));
            auto sourceElement = sourceTupleInfo ? builder.emitFieldExtract(
                                                       sourceTupleInfo->fields[i]->getFieldType(),
                                                       inst->getSource(),
                                                       sourceTupleInfo->fields[i]->getKey())
                                                 : inst->getSource();
            elements[baseIndex] = sourceElement;
        }
        auto resultTypeInfo = getLoweredTupleType(&builder, inst->getDataType());
        auto makeStruct = builder.emitMakeStruct(resultTypeInfo->structType, elements);
        inst->replaceUsesWith(makeStruct);
        inst->removeAndDeallocate();
    }

    void processSwizzledStore(IRSwizzledStore* inst)
    {
        IRBuilder builder(module);
        builder.setInsertBefore(inst);

        auto dest = inst->getDest();
        auto destValueType = tryGetPointedToType(&builder, dest->getDataType());
        auto loweredTupleInfo = getLoweredTupleType(&builder, destValueType);
        auto sourceTupleInfo = getLoweredTupleType(&builder, inst->getSource()->getDataType());
        if (!loweredTupleInfo)
            return;

        for (UInt i = 0; i < inst->getElementCount(); i++)
        {
            auto baseIndex = getIntVal(inst->getElementIndex(i));
            auto destField = loweredTupleInfo->fields[baseIndex];
            auto destFieldPtr = builder.emitFieldAddress(
                builder.getPtrType(destField->getFieldType()),
                dest,
                destField->getKey());
            auto sourceElement = sourceTupleInfo ? builder.emitFieldExtract(
                                                       sourceTupleInfo->fields[i]->getFieldType(),
                                                       inst->getSource(),
                                                       sourceTupleInfo->fields[i]->getKey())
                                                 : inst->getSource();
            builder.emitStore(destFieldPtr, sourceElement);
        }
        inst->removeAndDeallocate();
    }

    void processTupleType(IRTupleType* inst)
    {
        IRBuilder builderStorage(module);
        auto builder = &builderStorage;
        builder->setInsertBefore(inst);

        auto loweredTupleInfo = getLoweredTupleType(builder, inst);
        SLANG_ASSERT(loweredTupleInfo);
        SLANG_UNUSED(loweredTupleInfo);
    }

    void processIndexedFieldKey(IRIndexedFieldKey* inst)
    {
        IRBuilder builder(module);
        auto loweredTupleInfo = getLoweredTupleType(&builder, inst->getBaseType());
        if (!loweredTupleInfo)
            return;
        auto fieldIndex = getIntVal(inst->getIndex());
        SLANG_ASSERT(fieldIndex >= 0 && (Index)fieldIndex < loweredTupleInfo->fields.getCount());
        inst->replaceUsesWith(loweredTupleInfo->fields[fieldIndex]->getKey());
        inst->removeAndDeallocate();
    }

    void processUpdateElement(IRUpdateElement* inst)
    {
        // For UpdateElement insts, we need to figure out all the intermediate types on the access
        // chain, and if any of them are lowered tuples, we need to replace the access key with the
        // new struct key for the lowered tuple struct.
        //
        ShortList<IRInst*> newAccessChain;
        bool accessChainChanged = false;
        auto baseType = inst->getOldValue()->getDataType();
        IRBuilder builder(inst);
        builder.setInsertBefore(inst);

        for (UInt i = 0; i < inst->getAccessKeyCount(); i++)
        {
            auto key = inst->getAccessKey(i);
            if (auto structKey = as<IRStructKey>(key))
            {
                if (auto structType = as<IRStructType>(baseType))
                {
                    auto field = findStructField(structType, structKey);
                    baseType = field->getFieldType();
                    newAccessChain.add(structKey);
                }
                else
                {
                    // If we see anything not supported, just bail out.
                    return;
                }
            }
            else if (auto arrayType = as<IRArrayTypeBase>(baseType))
            {
                baseType = arrayType->getElementType();
                newAccessChain.add(key);
            }
            else if (auto loweredTupleInfo = getLoweredTupleType(&builder, baseType))
            {
                auto fieldIndex = getIntVal(key);
                if (fieldIndex >= 0 && (Index)fieldIndex < loweredTupleInfo->fields.getCount())
                {
                    auto field = loweredTupleInfo->fields[fieldIndex];
                    baseType = field->getFieldType();
                    newAccessChain.add(field->getKey());
                    accessChainChanged = true;
                }
                else
                {
                    // If we see anything not supported, just bail out.
                    break;
                }
            }
            else
            {
                // If we see anything not supported, just bail out.
                break;
            }
        }

        if (accessChainChanged)
        {
            auto newInst = builder.emitUpdateElement(
                inst->getOldValue(),
                newAccessChain.getArrayView().arrayView,
                inst->getElementValue());
            inst->replaceUsesWith(newInst);
            inst->removeAndDeallocate();
        }
    }

    void processInst(IRInst* inst)
    {
        switch (inst->getOp())
        {
        case kIROp_MakeTuple:
        case kIROp_MakeValuePack:
            processMakeTuple((IRMakeTuple*)inst);
            break;
        case kIROp_GetTupleElement:
            processGetTupleElement((IRGetTupleElement*)inst);
            break;
        case kIROp_GetElementPtr:
            processGetElementPtr((IRGetElementPtr*)inst);
            break;
        case kIROp_swizzle:
            processSwizzle((IRSwizzle*)inst);
            break;
        case kIROp_swizzleSet:
            processSwizzleSet((IRSwizzleSet*)inst);
            break;
        case kIROp_SwizzledStore:
            processSwizzledStore((IRSwizzledStore*)inst);
            break;
        case kIROp_TupleType:
            processTupleType((IRTupleType*)inst);
            break;
        case kIROp_IndexedFieldKey:
            processIndexedFieldKey((IRIndexedFieldKey*)inst);
            break;
        case kIROp_UpdateElement:
            processUpdateElement((IRUpdateElement*)inst);
            break;
        default:
            break;
        }
    }

    void processModule()
    {
        // First, we want to replace all TypePack with TupleType.

        List<IRInst*> typePacks;
        for (auto inst : module->getGlobalInsts())
        {
            if (inst->getOp() == kIROp_TypePack)
            {
                typePacks.add(inst);
            }
        }
        IRBuilder builder(module);
        for (auto inst : typePacks)
        {
            builder.setInsertBefore(inst);
            ShortList<IRType*> types;
            for (UInt i = 0; i < inst->getOperandCount(); i++)
            {
                types.add((IRType*)inst->getOperand(i));
            }
            auto tupleType =
                builder.getTupleType((UInt)types.getCount(), types.getArrayView().getBuffer());
            inst->replaceUsesWith(tupleType);
            inst->removeAndDeallocate();
        }

        // Next, lower all tuples to structs.

        addToWorkList(module->getModuleInst());

        while (workList.getCount() != 0)
        {
            IRInst* inst = workList.getLast();

            workList.removeLast();
            workListSet.remove(inst);

            processInst(inst);

            for (auto child = inst->getLastChild(); child; child = child->getPrevInst())
            {
                addToWorkList(child);
            }
        }

        // Replace all tuple types with lowered struct types.
        for (const auto& [key, value] : loweredTuples)
            key->replaceUsesWith(value->structType);
    }
};

void lowerTuples(IRModule* module, DiagnosticSink* sink)
{
    TupleLoweringContext context(module);
    context.sink = sink;
    context.processModule();
}
} // namespace Slang
