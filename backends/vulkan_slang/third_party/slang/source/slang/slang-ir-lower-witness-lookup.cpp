// slang-ir-lower-generic-existential.cpp

#include "slang-ir-lower-witness-lookup.h"

#include "slang-ir-clone.h"
#include "slang-ir-generics-lowering-context.h"
#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{

struct WitnessLookupLoweringContext
{
    IRModule* module;
    DiagnosticSink* sink;

    Dictionary<IRStructKey*, IRInst*> witnessDispatchFunctions;

    void init()
    {
        // Reconstruct the witness dispatch functions map.
        for (auto inst : module->getGlobalInsts())
        {
            if (auto key = as<IRStructKey>(inst))
            {
                for (auto decor : key->getDecorations())
                {
                    if (auto witnessDispatchFunc = as<IRDispatchFuncDecoration>(decor))
                    {
                        witnessDispatchFunctions.add(key, witnessDispatchFunc->getFunc());
                    }
                }
            }
        }
    }

    bool hasAssocType(IRInst* type)
    {
        if (!type)
            return false;

        InstHashSet processedSet(type->getModule());
        InstWorkList workList(type->getModule());
        workList.add(type);
        processedSet.add(type);
        for (Index i = 0; i < workList.getCount(); i++)
        {
            auto inst = workList[i];
            if (inst->getOp() == kIROp_AssociatedType)
                return true;

            for (UInt j = 0; j < inst->getOperandCount(); j++)
            {
                if (!inst->getOperand(j))
                    continue;
                if (processedSet.add(inst->getOperand(j)))
                    workList.add(inst->getOperand(j));
            }
        }
        return false;
    }

    IRType* translateType(IRBuilder builder, IRInst* type)
    {
        if (!type)
            return nullptr;
        if (auto genType = as<IRGeneric>(type))
        {
            IRCloneEnv cloneEnv;
            builder.setInsertBefore(genType);
            auto newGeneric = as<IRGeneric>(cloneInst(&cloneEnv, &builder, genType));
            newGeneric->setFullType(builder.getGenericKind());
            auto retVal = findGenericReturnVal(newGeneric);
            builder.setInsertBefore(retVal);
            auto translated = translateType(builder, retVal);
            retVal->replaceUsesWith(translated);
            return (IRType*)newGeneric;
        }
        else if (auto thisType = as<IRThisType>(type))
        {
            return (IRType*)thisType->getConstraintType();
        }
        else if (auto assocType = as<IRAssociatedType>(type))
        {
            return assocType;
        }

        if (as<IRBasicType>(type))
            return (IRType*)type;

        switch (type->getOp())
        {
        case kIROp_Param:
        case kIROp_VectorType:
        case kIROp_MatrixType:
        case kIROp_StructType:
        case kIROp_ClassType:
        case kIROp_InterfaceType:
            return (IRType*)type;
        default:
            {
                List<IRInst*> translatedOperands;
                for (UInt i = 0; i < type->getOperandCount(); i++)
                {
                    translatedOperands.add(translateType(builder, type->getOperand(i)));
                }
                auto translated = builder.emitIntrinsicInst(
                    type->getFullType(),
                    type->getOp(),
                    (UInt)translatedOperands.getCount(),
                    translatedOperands.getBuffer());
                return (IRType*)translated;
            }
        }
    }

    IRInst* findOrCreateDispatchFunc(IRLookupWitnessMethod* lookupInst)
    {
        IRInst* func = nullptr;
        auto requirementKey = cast<IRStructKey>(lookupInst->getRequirementKey());
        if (witnessDispatchFunctions.tryGetValue(requirementKey, func))
        {
            return func;
        }

        auto witnessTableOperand = lookupInst->getWitnessTable();
        auto witnessTableType = as<IRWitnessTableTypeBase>(witnessTableOperand->getDataType());
        SLANG_RELEASE_ASSERT(witnessTableType);
        auto interfaceType =
            as<IRInterfaceType>(unwrapAttributedType(witnessTableType->getConformanceType()));
        SLANG_RELEASE_ASSERT(interfaceType);
        if (interfaceType->findDecoration<IRComInterfaceDecoration>())
            return nullptr;
        auto requirementType = findInterfaceRequirement(interfaceType, requirementKey);
        SLANG_RELEASE_ASSERT(requirementType);

        // We only lower non-static function requirement lookups for now.
        // Our front end will stick a StaticRequirementDecoration on the IRStructKey for static
        // member requirements.
        if (lookupInst->getRequirementKey()->findDecoration<IRStaticRequirementDecoration>())
            return nullptr;
        auto interfaceMethodFuncType =
            as<IRFuncType>(getResolvedInstForDecorations(requirementType));
        if (interfaceMethodFuncType)
        {
            // Detect cases that we currently does not support and exit.

            // If this is a non static function requirement, we should
            // make sure the first parameter is the interface type. If not, something has gone
            // wrong.
            if (interfaceMethodFuncType->getParamCount() == 0)
                return nullptr;
            if (!as<IRThisType>(unwrapAttributedType(interfaceMethodFuncType->getParamType(0))))
                return nullptr;

            // The function has any associated type parameter, we currently can't lower it early in
            // this pass. We will lower it in the catch all generic lowering pass.
            for (UInt i = 1; i < interfaceMethodFuncType->getParamCount(); i++)
            {
                if (hasAssocType(interfaceMethodFuncType->getParamType(i)))
                    return nullptr;
            }

            // If return type is a composite type containing an assoc type, we won't lower it now.
            // Supporting general use of assoc type is possible, but would require more complex
            // logic in this pass to marshal things to and from existential types.
            if (interfaceMethodFuncType->getResultType()->getOp() != kIROp_AssociatedType &&
                hasAssocType(interfaceMethodFuncType->getResultType()))
                return nullptr;
        }
        else
        {
            return nullptr;
        }


        IRBuilder builder(module);
        builder.setInsertBefore(getParentFunc(lookupInst));

        // Create a dispatch func.
        IRFunc* dispatchFunc = nullptr;
        IRFuncType* dispatchFuncType = nullptr;
        IRGeneric* parentGeneric = nullptr;

        // If requirementType is a generic, we need to create a new generic that has the same
        // parameters.
        if (auto genericRequirement = as<IRGeneric>(requirementType))
        {
            IRCloneEnv cloneEnv;
            parentGeneric = as<IRGeneric>(cloneInst(&cloneEnv, &builder, genericRequirement));

            auto returnInst = as<IRReturn>(parentGeneric->getFirstBlock()->getLastInst());
            SLANG_RELEASE_ASSERT(returnInst);
            builder.setInsertBefore(returnInst);
            auto oldDispatchFuncType = as<IRFuncType>(returnInst->getVal());
            if (!oldDispatchFuncType)
                return nullptr;

            dispatchFuncType = as<IRFuncType>(translateType(builder, oldDispatchFuncType));

            SLANG_RELEASE_ASSERT(dispatchFuncType);

            dispatchFunc = builder.createFunc();
            dispatchFunc->setFullType(dispatchFuncType);
            builder.emitReturn(dispatchFunc);
            returnInst->removeAndDeallocate();

            parentGeneric->setFullType(translateType(builder, requirementType));
        }
        else
        {
            dispatchFuncType = as<IRFuncType>(translateType(builder, requirementType));
            dispatchFunc = builder.createFunc();
            dispatchFunc->setFullType(dispatchFuncType);
        }

        // We need to inline this function if the requirement is differentiable,
        // so that the autodiff pass doesn't need to handle the dispatch function.
        if (requirementKey->findDecoration<IRForwardDerivativeDecoration>() ||
            requirementKey->findDecoration<IRBackwardDerivativeDecoration>())
        {
            builder.addForceInlineDecoration(dispatchFunc);
        }

        // Collect generic params.
        List<IRInst*> genericParams;
        if (parentGeneric)
        {
            for (auto param : parentGeneric->getParams())
                genericParams.add(param);
        }

        // Emit the body of the dispatch func.
        builder.setInsertInto(dispatchFunc);
        auto firstBlock = builder.emitBlock();
        auto firstBlockBuilder = builder;
        // Emit parameters.
        List<IRInst*> params;

        for (UInt i = 0; i < dispatchFuncType->getParamCount(); i++)
        {
            params.add(builder.emitParam(dispatchFuncType->getParamType(i)));
        }
        auto witness = builder.emitExtractExistentialWitnessTable(params[0]);

        auto witnessTables = getWitnessTablesFromInterfaceType(module, interfaceType);
        if (witnessTables.getCount() == 0)
        {
            // If there is no witness table, we should emit an error.
            sink->diagnose(
                lookupInst,
                Diagnostics::noTypeConformancesFoundForInterface,
                interfaceType);
            return nullptr;
        }
        else
        {
            List<IRInst*> cases;
            for (auto witnessTable : witnessTables)
            {
                IRBlock* block = builder.emitBlock();
                auto caseValue = firstBlockBuilder.emitGetSequentialIDInst(witnessTable);
                cases.add(caseValue);
                cases.add(block);
                auto entry = findWitnessTableEntry(witnessTable, requirementKey);
                SLANG_RELEASE_ASSERT(entry);
                // If the entry is a generic, we need to specialize it.
                if (const auto genericEntry = as<IRGeneric>(entry))
                {
                    auto specializedFuncType = builder.emitSpecializeInst(
                        builder.getTypeKind(),
                        entry->getFullType(),
                        (UInt)genericParams.getCount(),
                        genericParams.getBuffer());
                    entry = builder.emitSpecializeInst(
                        (IRType*)specializedFuncType,
                        entry,
                        (UInt)genericParams.getCount(),
                        genericParams.getBuffer());
                }
                auto args = params;
                // Reinterpret the first arg into the concrete type.
                args[0] = builder.emitReinterpret(
                    witnessTable->getConcreteType(),
                    builder.emitExtractExistentialValue(
                        builder.emitExtractExistentialType(args[0]),
                        args[0]));

                auto calleeFuncType =
                    as<IRFuncType>(getResolvedInstForDecorations(entry)->getFullType());
                auto callReturnType = calleeFuncType->getResultType();
                if (callReturnType->getParent() != module->getModuleInst())
                {
                    // the return type is dependent on generic parameter, use the type from
                    // dispatchFuncType instead.
                    callReturnType = dispatchFuncType->getResultType();
                }

                IRInst* ret = builder.emitCallInst(
                    callReturnType,
                    entry,
                    (UInt)args.getCount(),
                    args.getBuffer());
                // If result type is an associated type, we need to pack it into an anyValue.
                if (as<IRAssociatedType>(dispatchFuncType->getResultType()))
                {
                    ret = builder.emitPackAnyValue(dispatchFuncType->getResultType(), ret);
                }
                builder.emitReturn(ret);
            }
            builder.setInsertInto(firstBlock);
            if (witnessTables.getCount() == 1)
            {
                builder.emitBranch((IRBlock*)cases[1]);
            }
            else
            {
                auto witnessId = firstBlockBuilder.emitGetSequentialIDInst(witness);
                auto breakLabel = builder.emitBlock();
                builder.emitUnreachable();
                firstBlockBuilder.emitSwitch(
                    witnessId,
                    breakLabel,
                    (IRBlock*)cases.getLast(),
                    (UInt)(cases.getCount() - 2),
                    cases.getBuffer());
            }
        }

        // Stick a decoration on the requirement key so we can find the dispatch func later.
        IRInst* resultValue = parentGeneric ? (IRInst*)parentGeneric : dispatchFunc;
        builder.addDispatchFuncDecoration(requirementKey, resultValue);

        // Register the dispatch func to witnessDispatchFunctions dictionary.
        witnessDispatchFunctions[requirementKey] = resultValue;

        return resultValue;
    }

    void rewriteCallSite(IRCall* call, IRInst* dispatchFunc, IRInst* initialExistentialObject)
    {
        SLANG_RELEASE_ASSERT(call->getArgCount() != 0);
        call->setOperand(0, dispatchFunc);
        IRBuilder builder(call);
        builder.setInsertBefore(call);
        auto witnessTable = builder.emitExtractExistentialWitnessTable(initialExistentialObject);
        auto newExistentialObject = builder.emitMakeExistential(
            initialExistentialObject->getDataType(),
            call->getOperand(1),
            witnessTable);
        call->setOperand(1, newExistentialObject);
    }

    bool processWitnessLookup(IRLookupWitnessMethod* lookupInst)
    {
        auto witnessTableOperand = lookupInst->getWitnessTable();
        auto extractInst = as<IRExtractExistentialWitnessTable>(witnessTableOperand);
        if (!extractInst)
            return false;
        auto dispatchFunc = findOrCreateDispatchFunc(lookupInst);
        if (!dispatchFunc)
            return false;
        bool changed = false;
        auto existentialObject = extractInst->getOperand(0);

        IRBuilder builder(lookupInst);
        builder.setInsertBefore(lookupInst);
        traverseUses(
            lookupInst,
            [&](IRUse* use)
            {
                if (auto specialize = as<IRSpecialize>(use->getUser()))
                {
                    builder.setInsertBefore(use->getUser());
                    List<IRInst*> args;
                    for (UInt i = 0; i < specialize->getArgCount(); i++)
                        args.add(specialize->getArg(i));
                    auto specializedType = builder.emitSpecializeInst(
                        builder.getTypeKind(),
                        dispatchFunc->getFullType(),
                        (UInt)args.getCount(),
                        args.getBuffer());
                    auto newSpecialize = builder.emitSpecializeInst(
                        (IRType*)specializedType,
                        dispatchFunc,
                        (UInt)args.getCount(),
                        args.getBuffer());
                    traverseUses(
                        specialize,
                        [&](IRUse* specializeUse)
                        {
                            if (auto call = as<IRCall>(specializeUse->getUser()))
                            {
                                changed = true;
                                rewriteCallSite(call, newSpecialize, existentialObject);
                            }
                        });
                }
                else if (auto call = as<IRCall>(use->getUser()))
                {
                    changed = true;
                    rewriteCallSite(call, dispatchFunc, existentialObject);
                }
            });
        return changed;
    }

    bool processFunc(IRFunc* func)
    {
        bool changed = false;
        for (auto bb : func->getBlocks())
        {
            for (auto inst : bb->getModifiableChildren())
            {
                if (auto witnessLookupInst = as<IRLookupWitnessMethod>(inst))
                {
                    changed |= processWitnessLookup(witnessLookupInst);
                }
            }
        }
        return changed;
    }
};

bool lowerWitnessLookup(IRModule* module, DiagnosticSink* sink)
{
    bool changed = false;
    WitnessLookupLoweringContext context;
    context.module = module;
    context.sink = sink;
    context.init();

    for (auto inst : module->getGlobalInsts())
    {
        // Process all fully specialized functions and look for
        // witness lookup instructions. If we see a lookup for a non-static function,
        // create a dispatch function and replace the lookup with a call to the dispatch function.
        if (auto func = as<IRFunc>(inst))
            changed |= context.processFunc(func);
    }
    return changed;
}
} // namespace Slang
