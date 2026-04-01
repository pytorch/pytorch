// slang-ir-cleanup-void.cpp

#include "slang-ir-cleanup-void.h"

#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{
struct CleanUpVoidContext
{
    IRModule* module;

    InstWorkList workList;
    InstHashSet workListSet;

    CleanUpVoidContext(IRModule* inModule)
        : module(inModule), workList(inModule), workListSet(inModule)
    {
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

    void processInst(IRInst* inst)
    {
        switch (inst->getOp())
        {
        case kIROp_Call:
        case kIROp_MakeStruct:
            {
                // Remove void argument.
                List<IRInst*> newArgs;
                for (UInt i = 0; i < inst->getOperandCount(); i++)
                {
                    auto arg = inst->getOperand(i);
                    if (arg->getDataType() && arg->getDataType()->getOp() == kIROp_VoidType)
                    {
                        continue;
                    }
                    newArgs.add(arg);
                }
                if (newArgs.getCount() != (Index)inst->getOperandCount())
                {
                    IRBuilder builder(module);
                    builder.setInsertBefore(inst);
                    auto newCall = builder.emitIntrinsicInst(
                        inst->getFullType(),
                        inst->getOp(),
                        newArgs.getCount(),
                        newArgs.getBuffer());
                    inst->replaceUsesWith(newCall);
                    inst->removeAndDeallocate();
                    inst = newCall;
                }
            }
            break;
        case kIROp_Func:
            {
                // Remove void parameter.
                List<IRParam*> paramsToRemove;
                auto func = as<IRFunc>(inst);
                for (auto param : func->getParams())
                {
                    if (param->getDataType()->getOp() == kIROp_VoidType)
                    {
                        paramsToRemove.add(param);
                    }
                }
                IRBuilder builder(module);
                builder.setInsertBefore(func);
                for (auto param : paramsToRemove)
                {
                    auto voidVal = builder.getVoidValue();
                    param->replaceUsesWith(voidVal);
                    param->removeAndDeallocate();
                }
            }
            break;
        case kIROp_FuncType:
            {
                auto funcType = as<IRFuncType>(inst);
                List<IRInst*> newOperands;
                for (UInt i = 1; i < funcType->getOperandCount(); i++)
                {
                    auto operand = funcType->getOperand(i);
                    if (operand->getOp() == kIROp_VoidType)
                    {
                        continue;
                    }
                    newOperands.add(operand);
                }
                if (newOperands.getCount() != (Index)funcType->getParamCount())
                {
                    IRBuilder builder(module);
                    builder.setInsertBefore(funcType);
                    auto newFuncType = builder.getFuncType(
                        newOperands.getCount(),
                        (IRType**)newOperands.getBuffer(),
                        funcType->getResultType());
                    if (newFuncType != funcType)
                    {
                        funcType->replaceUsesWith(newFuncType);
                        funcType->removeAndDeallocate();
                    }
                    inst = newFuncType;
                }
            }
            break;
        case kIROp_StructType:
            {
                List<IRInst*> toRemove;
                for (auto child : inst->getChildren())
                {
                    if (auto field = as<IRStructField>(child))
                    {
                        if (field->getFieldType()->getOp() == kIROp_VoidType)
                        {
                            toRemove.add(field);
                        }
                    }
                }
                for (auto ii : toRemove)
                    ii->removeAndDeallocate();
            }
            break;
        default:
            break;
        }

        // If inst has void type, all uses of it should be replaced with void val.
        // We should do this only for a subset of opcodes known to be safe.
        switch (inst->getOp())
        {
        case kIROp_Load:
        case kIROp_GetElement:
        case kIROp_GetOptionalValue:
        case kIROp_FieldExtract:
        case kIROp_GetTupleElement:
        case kIROp_GetResultError:
        case kIROp_GetResultValue:
        case kIROp_Call:
        case kIROp_UpdateElement:
        case kIROp_GetTargetTupleElement:
            if (inst->getDataType()->getOp() == kIROp_VoidType)
            {
                IRBuilder builder(module);
                builder.setInsertBefore(inst);
                inst->replaceUsesWith(builder.getVoidValue());
            }
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
    }
};

void cleanUpVoidType(IRModule* module)
{
    CleanUpVoidContext context(module);
    context.processModule();
}
} // namespace Slang
