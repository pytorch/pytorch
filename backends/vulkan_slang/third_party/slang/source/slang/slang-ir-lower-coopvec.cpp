#include "slang-ir-lower-coopvec.h"

#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{
struct CoopVecLoweringContext
{
    IRModule* module;
    DiagnosticSink* sink;

    InstWorkList workList;
    InstHashSet workListSet;

    CoopVecLoweringContext(IRModule* inModule)
        : module(inModule), workList(inModule), workListSet(inModule)
    {
    }

    struct LoweredCoopVecInfo : public RefObject
    {
        IRType* coopvecType;
        IRArrayType* arrayType;
    };
    Dictionary<IRInst*, RefPtr<LoweredCoopVecInfo>> mapLoweredArrayToCoopVecInfo;
    Dictionary<IRInst*, RefPtr<LoweredCoopVecInfo>> loweredCoopVecs;

    IRType* maybeLowerCoopVecType(IRBuilder* builder, IRType* type)
    {
        if (const auto cvt = as<IRCoopVectorType>(type))
        {
            if (auto info = getLoweredCoopVecType(builder, cvt))
                return info->arrayType;
        }
        return type;
    }

    LoweredCoopVecInfo* getLoweredCoopVecType(IRBuilder* builder, IRCoopVectorType* type)
    {
        if (auto loweredInfo = loweredCoopVecs.tryGetValue(type))
            return loweredInfo->Ptr();
        if (auto loweredInfo = mapLoweredArrayToCoopVecInfo.tryGetValue(type))
            return loweredInfo->Ptr();

        if (!type)
            return nullptr;

        RefPtr<LoweredCoopVecInfo> info = new LoweredCoopVecInfo();
        info->coopvecType = (IRType*)type;
        info->arrayType = builder->getArrayType(type->getElementType(), type->getElementCount());
        builder->addNameHintDecoration(info->arrayType, UnownedStringSlice("CoopVec"));

        mapLoweredArrayToCoopVecInfo[info->arrayType] = info;
        loweredCoopVecs[type] = info;
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

    void processMakeCoopVec(IRInst* inst)
    {
        IRBuilder builderStorage(module);
        auto builder = &builderStorage;
        builder->setInsertBefore(inst);

        const auto cvt = as<IRCoopVectorType>(inst->getDataType());
        SLANG_ASSERT(cvt);
        auto info = getLoweredCoopVecType(builder, cvt);
        List<IRInst*> operands;
        operands.setCount(Index(inst->getOperandCount()));
        UIndex i = 0;
        for (auto operand = inst->getOperands(); i < inst->getOperandCount(); operand++, i++)
            operands[Index(i)] = operand->get();
        auto makeArray =
            builder->emitMakeArray(info->arrayType, operands.getCount(), operands.begin());
        inst->replaceUsesWith(makeArray);
        inst->removeAndDeallocate();
    }

    void processGetCoopVecElement(IRGetElement*) {}

    void processGetElementPtr(IRGetElementPtr*) {}

    void processGetElement(IRGetElement*) {}

    void processCoopVecType(IRCoopVectorType* inst)
    {
        IRBuilder builderStorage(module);
        auto builder = &builderStorage;
        builder->setInsertBefore(inst);

        auto loweredCoopVecInfo = getLoweredCoopVecType(builder, inst);
        SLANG_ASSERT(loweredCoopVecInfo);
        SLANG_UNUSED(loweredCoopVecInfo);
    }

    void processUpdateElement(IRUpdateElement*) {}

    void processEntrywiseOp(IRInst* inst)
    {
        SLANG_ASSERT(inst->getOperandCount());
        if (!as<IRCoopVectorType>(inst->getDataType()))
            return;
        List<IRInst*> operands;
        IRIntegerValue width = 0;
        IRType* resultElementType = nullptr;
        UIndex opIndex = 0;
        for (auto operand = inst->getOperands(); opIndex < inst->getOperandCount();
             operand++, opIndex++)
        {
            operands.add(operand->get());
            if (const auto cv = as<IRCoopVectorType>(operand->get()->getDataType()))
            {
                width = getIntVal(cv->getElementCount());
                resultElementType = cv->getElementType();
            }
        }
        if (width == 0)
            return;
        IRBuilder builder(module);
        IRType* resultElementPtrType = builder.getPtrType(resultElementType);
        builder.setInsertBefore(inst);
        const auto result = builder.emitVar(inst->getFullType());
        List<IRInst*> entrywiseOperands;
        entrywiseOperands.setCount(operands.getCount());
        for (IRIntegerValue i = 0; i < width; ++i)
        {
            for (int j = 0; j < operands.getCount(); ++j)
            {
                if (const auto cv = as<IRCoopVectorType>(operands[j]->getDataType()))
                {
                    SLANG_ASSERT(getIntVal(cv->getElementCount()) == width);
                    const auto elementType = cv->getElementType();
                    entrywiseOperands[j] = builder.emitGetElement(elementType, operands[j], i);
                }
                else
                {
                    entrywiseOperands[j] = operands[j];
                }
            }
            const auto x = builder.emitIntrinsicInst(
                resultElementType,
                inst->getOp(),
                entrywiseOperands.getCount(),
                entrywiseOperands.begin());
            const auto d = builder.emitGetElementPtr(resultElementPtrType, result, i);
            builder.emitStore(d, x);
        }
        const auto v = builder.emitLoad(result);
        inst->replaceUsesWith(v);
        inst->removeAndDeallocate();
    }

    void processInst(IRInst* inst)
    {
        switch (inst->getOp())
        {
        case kIROp_CoopVectorType:
            processCoopVecType((IRCoopVectorType*)inst);
            break;
        case kIROp_MakeCoopVector:
            processMakeCoopVec((IRMakeCoopVector*)inst);
            break;
        case kIROp_GetElement:
            processGetElement((IRGetElement*)inst);
            break;
        case kIROp_GetElementPtr:
            processGetElementPtr((IRGetElementPtr*)inst);
            break;
        case kIROp_UpdateElement:
            processUpdateElement((IRUpdateElement*)inst);
            break;
        case kIROp_Neg:
        case kIROp_Add:
        case kIROp_Sub:
        case kIROp_Mul:
        case kIROp_Div:
        case kIROp_IntCast:
        case kIROp_FloatCast:
        case kIROp_CastIntToFloat:
        case kIROp_CastFloatToInt:
            processEntrywiseOp(inst);
            break;
        default:
            break;
        }
    }

    void processModule()
    {
        IRBuilder builder(module);

        addToWorkList(module->getModuleInst());

        while (workList.getCount() != 0)
        {
            IRInst* inst = workList.getLast();

            workList.removeLast();
            workListSet.remove(inst);

            processInst(inst);

            for (auto child = inst->getLastChild(); child; child = child->getPrevInst())
                addToWorkList(child);
        }

        // Replace all coopvec types with sized array types
        for (const auto& [key, value] : loweredCoopVecs)
            key->replaceUsesWith(value->arrayType);
    }
};

void lowerCooperativeVectors(IRModule* module, DiagnosticSink* sink)
{
    CoopVecLoweringContext context(module);
    context.sink = sink;
    context.processModule();
}
} // namespace Slang
