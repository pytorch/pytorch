#include "slang-ir-generics-lowering-context.h"
#include "slang-ir-insts.h"
#include "slang-ir-specialize-dispatch.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{

struct AssociatedTypeLookupSpecializationContext
{
    SharedGenericsLoweringContext* sharedContext;

    IRFunc* createWitnessTableLookupFunc(IRInterfaceType* interfaceType, IRInst* key)
    {
        IRBuilder builder(sharedContext->module);
        builder.setInsertBefore(interfaceType);

        auto inputWitnessTableIDType = builder.getWitnessTableIDType(interfaceType);
        auto requirementEntry = sharedContext->findInterfaceRequirementVal(interfaceType, key);

        auto resultWitnessTableType = cast<IRWitnessTableType>(requirementEntry);
        auto resultWitnessTableIDType =
            builder.getWitnessTableIDType((IRType*)resultWitnessTableType->getConformanceType());

        auto funcType =
            builder.getFuncType(1, (IRType**)&inputWitnessTableIDType, resultWitnessTableIDType);
        auto func = builder.createFunc();
        func->setFullType(funcType);

        if (auto linkage = key->findDecoration<IRLinkageDecoration>())
            builder.addNameHintDecoration(func, linkage->getMangledName());

        builder.setInsertInto(func);

        auto block = builder.emitBlock();
        auto witnessTableParam = builder.emitParam(inputWitnessTableIDType);

        // `witnessTableParam` is expected to have `IRWitnessTableID` type, which
        // will later lower into a `uint2`. We only use the first element of the uint2
        // to store the sequential ID and reserve the second 32-bit value for future
        // pointer-compatibility. We insert a member extract inst right now
        // to obtain the first element and use it in our switch statement.
        UInt elemIdx = 0;
        auto witnessTableSequentialID =
            builder.emitSwizzle(builder.getUIntType(), witnessTableParam, 1, &elemIdx);

        // Collect all witness tables of `witnessTableType` in current module.
        List<IRWitnessTable*> witnessTables =
            sharedContext->getWitnessTablesFromInterfaceType(interfaceType);

        // Generate case blocks for each possible witness table.
        IRBlock* defaultBlock = nullptr;
        List<IRInst*> caseBlocks;
        for (Index i = 0; i < witnessTables.getCount(); i++)
        {
            auto witnessTable = witnessTables[i];
            auto seqIdDecoration = witnessTable->findDecoration<IRSequentialIDDecoration>();
            SLANG_ASSERT(seqIdDecoration);

            if (i != witnessTables.getCount() - 1)
            {
                // Create a case block if we are not the last case.
                caseBlocks.add(seqIdDecoration->getSequentialIDOperand());
                builder.setInsertInto(func);
                auto caseBlock = builder.emitBlock();
                caseBlocks.add(caseBlock);
            }
            else
            {
                // Generate code for the last possible value in the `default` block.
                builder.setInsertInto(func);
                defaultBlock = builder.emitBlock();
                builder.setInsertInto(defaultBlock);
            }

            auto resultWitnessTable = findWitnessTableEntry(witnessTable, key);
            auto resultWitnessTableIDDecoration =
                resultWitnessTable->findDecoration<IRSequentialIDDecoration>();
            SLANG_ASSERT(resultWitnessTableIDDecoration);
            // Pack the resulting witness table ID into a `uint2`.
            auto uint2Type = builder.getVectorType(
                builder.getUIntType(),
                builder.getIntValue(builder.getIntType(), 2));
            IRInst* uint2Args[] = {
                resultWitnessTableIDDecoration->getSequentialIDOperand(),
                builder.getIntValue(builder.getUIntType(), 0)};
            auto resultID = builder.emitMakeVector(uint2Type, 2, uint2Args);
            builder.emitReturn(resultID);
        }

        builder.setInsertInto(func);

        if (witnessTables.getCount() == 1)
        {
            // If there is only 1 case, no switch statement is necessary.
            builder.setInsertInto(block);
            builder.emitBranch(defaultBlock);
        }
        else
        {
            // If there are more than 1 cases,
            // emit a switch statement to return the correct witness table ID based on
            // the witness table ID passed in.
            auto breakBlock = builder.emitBlock();
            builder.setInsertInto(breakBlock);
            builder.emitUnreachable();

            builder.setInsertInto(block);
            builder.emitSwitch(
                witnessTableSequentialID,
                breakBlock,
                defaultBlock,
                caseBlocks.getCount(),
                caseBlocks.getBuffer());
        }

        return func;
    }

    void processLookupInterfaceMethodInst(IRLookupWitnessMethod* inst)
    {
        if (isComInterfaceType(inst->getWitnessTable()->getDataType()))
        {
            return;
        }

        // Ignore lookups for RTTI objects for now, since they are not used anywhere.
        if (!as<IRWitnessTableType>(inst->getDataType()))
        {
            IRBuilder builder(sharedContext->module);
            builder.setInsertBefore(inst);
            auto uint2Type = builder.getVectorType(
                builder.getUIntType(),
                builder.getIntValue(builder.getIntType(), 2));
            auto zero = builder.getIntValue(builder.getUIntType(), 0);
            IRInst* args[] = {zero, zero};
            auto zeroUint2 = builder.emitMakeVector(uint2Type, 2, args);
            inst->replaceUsesWith(zeroUint2);
            return;
        }

        // Replace all witness table lookups with calls to specialized functions that directly
        // returns the sequential ID of the resulting witness table, effectively getting rid
        // of actual witness table objects in the target code (they all become IDs).
        auto witnessTableType = inst->getWitnessTable()->getDataType();
        IRInterfaceType* interfaceType = cast<IRInterfaceType>(
            cast<IRWitnessTableTypeBase>(witnessTableType)->getConformanceType());
        if (!interfaceType)
            return;
        auto key = inst->getRequirementKey();
        IRFunc* func = nullptr;
        if (!sharedContext->mapInterfaceRequirementKeyToDispatchMethods.tryGetValue(key, func))
        {
            func = createWitnessTableLookupFunc(interfaceType, key);
            sharedContext->mapInterfaceRequirementKeyToDispatchMethods[key] = func;
        }
        IRBuilder builder(sharedContext->module);
        builder.setInsertBefore(inst);
        auto witnessTableArg = inst->getWitnessTable();
        auto callInst = builder.emitCallInst(func->getResultType(), func, witnessTableArg);
        inst->replaceUsesWith(callInst);
        inst->removeAndDeallocate();
    }

    void processGetSequentialIDInst(IRGetSequentialID* inst)
    {
        // If the operand is a witness table, it is already replaced with a uint2
        // at this point, where the first element in the uint2 is the id of the
        // witness table.
        IRBuilder builder(sharedContext->module);
        builder.setInsertBefore(inst);
        UInt index = 0;
        auto id = builder.emitSwizzle(builder.getUIntType(), inst->getRTTIOperand(), 1, &index);
        inst->replaceUsesWith(id);
        inst->removeAndDeallocate();
    }

    void processModule()
    {
        // Replace all `lookup_interface_method():IRWitnessTable` with call to specialized
        // functions.
        workOnModule(
            sharedContext,
            [this](IRInst* inst)
            {
                if (inst->getOp() == kIROp_LookupWitness)
                {
                    processLookupInterfaceMethodInst(cast<IRLookupWitnessMethod>(inst));
                }
            });

        // Replace all direct uses of IRWitnessTables with its sequential ID.
        workOnModule(
            sharedContext,
            [this](IRInst* inst)
            {
                if (inst->getOp() == kIROp_WitnessTable)
                {
                    auto seqId = inst->findDecoration<IRSequentialIDDecoration>();
                    if (!seqId)
                        return;
                    // Insert code to pack sequential ID into an uint2 at all use sites.
                    traverseUses(
                        inst,
                        [&](IRUse* use)
                        {
                            if (as<IRCOMWitnessDecoration>(use->getUser()))
                            {
                                return;
                            }
                            IRBuilder builder(sharedContext->module);
                            builder.setInsertBefore(use->getUser());
                            auto uint2Type = builder.getVectorType(
                                builder.getUIntType(),
                                builder.getIntValue(builder.getIntType(), 2));
                            IRInst* uint2Args[] = {
                                seqId->getSequentialIDOperand(),
                                builder.getIntValue(builder.getUIntType(), 0)};
                            auto uint2seqID = builder.emitMakeVector(uint2Type, 2, uint2Args);
                            builder.replaceOperand(use, uint2seqID);
                        });
                }
            });

        // Replace all `IRWitnessTableType`s with `IRWitnessTableIDType`.
        for (auto globalInst : sharedContext->module->getGlobalInsts())
        {
            if (globalInst->getOp() == kIROp_WitnessTableType)
            {
                IRBuilder builder(sharedContext->module);
                builder.setInsertBefore(globalInst);
                auto witnessTableIDType = builder.getWitnessTableIDType(
                    (IRType*)cast<IRWitnessTableType>(globalInst)->getConformanceType());
                traverseUses(
                    globalInst,
                    [&](IRUse* use)
                    {
                        if (use->getUser()->getOp() == kIROp_WitnessTable)
                            return;
                        builder.replaceOperand(use, witnessTableIDType);
                    });
            }
        }

        // `GetSequentialID(WitnessTableIDOperand)` becomes just `WitnessTableIDOperand`.
        workOnModule(
            sharedContext,
            [this](IRInst* inst)
            {
                if (inst->getOp() == kIROp_GetSequentialID)
                {
                    processGetSequentialIDInst(cast<IRGetSequentialID>(inst));
                }
            });
    }
};

void specializeDynamicAssociatedTypeLookup(SharedGenericsLoweringContext* sharedContext)
{
    AssociatedTypeLookupSpecializationContext context;
    context.sharedContext = sharedContext;
    context.processModule();
}

} // namespace Slang
