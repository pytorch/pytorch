// slang-ir-lower-com-methods.cpp

#include "slang-ir-lower-com-methods.h"

#include "slang-ir-inst-pass-base.h"
#include "slang-ir-insts.h"
#include "slang-ir-marshal-native-call.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{

struct ComMethodLoweringContext : public InstPassBase
{
    DiagnosticSink* diagnosticSink = nullptr;

    NativeCallMarshallingContext marshal;

    OrderedHashSet<IRLookupWitnessMethod*> comCallees;

    ComMethodLoweringContext(IRModule* inModule)
        : InstPassBase(inModule)
    {
    }

    void processComCall(IRCall* comCall)
    {
        IRBuilder builder(module);
        builder.setInsertBefore(comCall);
        auto callee = as<IRLookupWitnessMethod>(comCall->getCallee());
        SLANG_ASSERT(callee);

        IRLookupWitnessMethod* innerMostCallee = callee;
        while (innerMostCallee->getOperand(0)->getOp() == kIROp_LookupWitness)
        {
            innerMostCallee = as<IRLookupWitnessMethod>(innerMostCallee->getOperand(0));
        }
        if (callee != innerMostCallee)
        {
            callee = (IRLookupWitnessMethod*)builder.emitLookupInterfaceMethodInst(
                callee->getDataType(),
                innerMostCallee->getWitnessTable(),
                callee->getRequirementKey());
        }
        comCallees.add(callee);

        auto calleeType = as<IRFuncType>(callee->getDataType());
        SLANG_ASSERT(calleeType);

        auto nativeFuncType = marshal.getNativeFuncType(builder, calleeType);
        ShortList<IRInst*> args;
        for (UInt i = 0; i < comCall->getArgCount(); i++)
            args.add(comCall->getArg(i));
        auto currentBlock = builder.getBlock();
        auto nextInst = comCall->getNextInst();
        auto newResult = marshal.marshalNativeCall(
            builder,
            calleeType,
            nativeFuncType,
            callee,
            args.getCount(),
            args.getArrayView().getBuffer());

        comCall->replaceUsesWith(newResult);
        if (builder.getBlock() != currentBlock)
        {
            // `marshalNativeCall` may have replaced the original call with branch insts.
            // If this is the case, we need to move all insts after the original call in the
            // original basic block to the new basic block.
            while (nextInst)
            {
                auto next = nextInst->getNextInst();
                nextInst->removeFromParent();
                nextInst->insertAtEnd(builder.getBlock());
                nextInst = next;
            }
        }
        comCall->removeAndDeallocate();
    }

    void processCall(IRCall* inst)
    {
        auto funcValue = inst->getOperand(0);

        // Detect if this is a call into a COM interface method.
        if (funcValue->getOp() == kIROp_LookupWitness)
        {
            const auto operand0TypeOp = funcValue->getOperand(0)->getDataType();
            if (auto tableType = as<IRWitnessTableTypeBase>(operand0TypeOp))
            {
                if (tableType->getConformanceType()->findDecoration<IRComInterfaceDecoration>())
                {
                    processComCall(inst);
                    return;
                }
            }
        }
    }

    void processInterfaceType(IRInterfaceType* interfaceType)
    {
        if (!interfaceType->findDecoration<IRComInterfaceDecoration>())
            return;
        IRBuilder builder(module);
        for (UInt i = 0; i < interfaceType->getOperandCount(); i++)
        {
            auto entry = as<IRInterfaceRequirementEntry>(interfaceType->getOperand(i));
            if (!entry)
                continue;
            if (auto funcType = as<IRFuncType>(entry->getRequirementVal()))
            {
                builder.setInsertBefore(funcType);
                entry->setRequirementVal(marshal.getNativeFuncType(builder, funcType));
            }
        }
    }

    void processWitnessTable(IRWitnessTable* witnessTable)
    {
        auto interfaceType = as<IRInterfaceType>(witnessTable->getConformanceType());
        if (!interfaceType)
            return;
        if (!interfaceType->findDecoration<IRComInterfaceDecoration>())
            return;
        auto interfaceReqDict = buildInterfaceRequirementDict(interfaceType);

        IRBuilder builder(module);
        NativeCallMarshallingContext marshalContext;
        marshalContext.diagnosticSink = diagnosticSink;
        for (auto entry : witnessTable->getEntries())
        {
            IRInst* interfaceRequirement = nullptr;
            if (!interfaceReqDict.tryGetValue(entry->getRequirementKey(), interfaceRequirement))
                continue;
            auto implFunc = as<IRFunc>(entry->getSatisfyingVal());
            if (!implFunc)
                continue;
            // If the function already has the same signature as the lowered COM interface method,
            // we don't need to do anything.
            if (isTypeEqual(
                    entry->getSatisfyingVal()->getDataType(),
                    (IRType*)interfaceRequirement))
                continue;
            // Now we need to generate a wrapper function that calls into the original one.
            auto nativeFunc = marshalContext.generateDLLExportWrapperFunc(builder, implFunc);
            entry->setOperand(1, nativeFunc);
        }

        auto classType = witnessTable->getConcreteType();
        builder.addCOMWitnessDecoration(classType, witnessTable);
    }

    void processModule()
    {
        // Translate all Calls to interface methods.
        processInstsOfType<IRCall>(kIROp_Call, [this](IRCall* inst) { processCall(inst); });

        // Update functypes of com callees.
        for (auto callee : comCallees)
        {
            IRBuilder builder(module);
            builder.setInsertBefore(callee);
            auto nativeType =
                marshal.getNativeFuncType(builder, as<IRFuncType>(callee->getDataType()));
            callee->setFullType(nativeType);
        }

        // Update func types of COM interfaces.
        processInstsOfType<IRInterfaceType>(
            kIROp_InterfaceType,
            [this](IRInterfaceType* inst) { processInterfaceType(inst); });

        // Update witness tables of classes that implement COM interfaces.
        // Generate native-to-managed wrappers for each witness table entry.
        processInstsOfType<IRWitnessTable>(
            kIROp_WitnessTable,
            [this](IRWitnessTable* table) { processWitnessTable(table); });
    }
};

void lowerComMethods(IRModule* module, DiagnosticSink* sink)
{
    ComMethodLoweringContext context(module);
    context.diagnosticSink = sink;
    context.marshal.diagnosticSink = sink;

    return context.processModule();
}
} // namespace Slang
