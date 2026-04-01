// slang-ir-lower-enum-type.cpp

#include "slang-ir-lower-enum-type.h"

#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{
struct EnumTypeLoweringContext
{
    IRModule* module;
    DiagnosticSink* sink;

    InstWorkList workList;
    InstHashSet workListSet;

    IRGeneric* genericOptionalStructType = nullptr;
    IRStructKey* valueKey = nullptr;
    IRStructKey* hasValueKey = nullptr;

    EnumTypeLoweringContext(IRModule* inModule)
        : module(inModule), workList(inModule), workListSet(inModule)
    {
    }

    struct LoweredEnumTypeInfo : public RefObject
    {
        IRType* enumType = nullptr;
        IRType* loweredType = nullptr;
    };
    Dictionary<IRInst*, RefPtr<LoweredEnumTypeInfo>> loweredEnumTypes;

    void addToWorkList(IRInst* inst)
    {
        if (workListSet.contains(inst))
            return;

        workList.add(inst);
        workListSet.add(inst);
    }

    LoweredEnumTypeInfo* getLoweredEnumType(IRInst* type)
    {
        if (auto loweredInfo = loweredEnumTypes.tryGetValue(type))
            return loweredInfo->Ptr();

        if (!type)
            return nullptr;

        if (type->getOp() != kIROp_EnumType)
            return nullptr;

        RefPtr<LoweredEnumTypeInfo> info = new LoweredEnumTypeInfo();
        auto enumType = cast<IREnumType>(type);
        auto valueType = enumType->getTagType();
        info->enumType = (IRType*)type;
        info->loweredType = valueType;
        loweredEnumTypes[type] = info;
        return info.Ptr();
    }

    void processEnumType(IREnumType* inst)
    {
        auto loweredEnumTypeInfo = getLoweredEnumType(inst);
        SLANG_ASSERT(loweredEnumTypeInfo);
        SLANG_UNUSED(loweredEnumTypeInfo);
    }

    void processEnumCast(IRInst* inst)
    {
        IRBuilder builderStorage(module);
        auto builder = &builderStorage;
        builder->setInsertBefore(inst);

        auto value = inst->getOperand(0);
        if (auto enumType = getLoweredEnumType(value->getDataType()))
        {
            auto rate = value->getRate();
            auto type = enumType->loweredType;
            if (rate)
            {
                type = builder->getRateQualifiedType(rate, type);
            }

            value->setFullType(type);
        }

        auto type = inst->getDataType();
        if (auto enumType = getLoweredEnumType(type))
        { // Cast was into enum, so use tag type instead.
            type = enumType->loweredType;
        }

        auto cast = builder->emitCast(type, value);

        inst->replaceUsesWith(cast);
        inst->removeAndDeallocate();
    }

    void processInst(IRInst* inst)
    {
        switch (inst->getOp())
        {
        case kIROp_EnumType:
            processEnumType((IREnumType*)inst);
            break;
        case kIROp_CastEnumToInt:
        case kIROp_CastIntToEnum:
        case kIROp_EnumCast:
            processEnumCast(inst);
            break;
        default:
            break;
        }
    }

    void processModule()
    {
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

        // Replace all enum types with their lowered equivalent types.
        for (const auto& [key, value] : loweredEnumTypes)
            key->replaceUsesWith(value->loweredType);
    }
};

void lowerEnumType(IRModule* module, DiagnosticSink* sink)
{
    EnumTypeLoweringContext context(module);
    context.sink = sink;
    context.processModule();
}
} // namespace Slang
