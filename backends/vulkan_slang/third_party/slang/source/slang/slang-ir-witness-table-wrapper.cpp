// slang-ir-witness-table-wrapper.cpp
#include "slang-ir-witness-table-wrapper.h"

#include "slang-ir-clone.h"
#include "slang-ir-generics-lowering-context.h"
#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{
struct GenericsLoweringContext;

struct GenerateWitnessTableWrapperContext
{
    SharedGenericsLoweringContext* sharedContext;

    // Represents a work item for packing `inout` or `out` arguments after a concrete call.
    struct ArgumentPackWorkItem
    {
        // A `AnyValue` typed destination.
        IRInst* dstArg = nullptr;
        // A concrete value to be packed.
        IRInst* concreteArg = nullptr;
    };

    // Unpack an `arg` of `IRAnyValue` into concrete type if necessary, to make it feedable into the
    // parameter. If `arg` represents a AnyValue typed variable passed in to a concrete `out`
    // parameter, this function indicates that it needs to be packed after the call by setting
    // `packAfterCall`.
    IRInst* maybeUnpackArg(
        IRBuilder* builder,
        IRType* paramType,
        IRInst* arg,
        ArgumentPackWorkItem& packAfterCall)
    {
        packAfterCall.dstArg = nullptr;
        packAfterCall.concreteArg = nullptr;

        // If either paramType or argType is a pointer type
        // (because of `inout` or `out` modifiers), we extract
        // the underlying value type first.
        IRType* paramValType = paramType;
        IRType* argValType = arg->getDataType();
        IRInst* argVal = arg;
        if (auto ptrType = as<IRPtrTypeBase>(paramType))
        {
            paramValType = ptrType->getValueType();
        }
        auto argType = arg->getDataType();
        if (auto argPtrType = as<IRPtrTypeBase>(argType))
        {
            argValType = argPtrType->getValueType();
            argVal = builder->emitLoad(arg);
        }

        // Unpack `arg` if the parameter expects concrete type but
        // `arg` is an AnyValue.
        if (!as<IRAnyValueType>(paramValType) && as<IRAnyValueType>(argValType))
        {
            auto unpackedArgVal = builder->emitUnpackAnyValue(paramValType, argVal);
            // if parameter expects an `out` pointer, store the unpacked val into a
            // variable and pass in a pointer to that variable.
            if (as<IRPtrTypeBase>(paramType))
            {
                auto tempVar = builder->emitVar(paramValType);
                builder->emitStore(tempVar, unpackedArgVal);
                // tempVar needs to be unpacked into original var after the call.
                packAfterCall.dstArg = arg;
                packAfterCall.concreteArg = tempVar;
                return tempVar;
            }
            else
            {
                return unpackedArgVal;
            }
        }
        return arg;
    }

    IRStringLit* _getWitnessTableWrapperFuncName(IRFunc* func)
    {
        IRBuilder builderStorage(sharedContext->module);
        auto builder = &builderStorage;
        builder->setInsertBefore(func);
        if (auto linkageDecoration = func->findDecoration<IRLinkageDecoration>())
        {
            return builder->getStringValue(
                (String(linkageDecoration->getMangledName()) + "_wtwrapper").getUnownedSlice());
        }
        if (auto namehintDecoration = func->findDecoration<IRNameHintDecoration>())
        {
            return builder->getStringValue(
                (String(namehintDecoration->getName()) + "_wtwrapper").getUnownedSlice());
        }
        return nullptr;
    }

    IRFunc* emitWitnessTableWrapper(IRFunc* func, IRInst* interfaceRequirementVal)
    {
        auto funcTypeInInterface = cast<IRFuncType>(interfaceRequirementVal);

        IRBuilder builderStorage(sharedContext->module);
        auto builder = &builderStorage;
        builder->setInsertBefore(func);

        auto wrapperFunc = builder->createFunc();
        wrapperFunc->setFullType((IRType*)interfaceRequirementVal);
        if (auto name = _getWitnessTableWrapperFuncName(func))
            builder->addNameHintDecoration(wrapperFunc, name);

        builder->setInsertInto(wrapperFunc);
        auto block = builder->emitBlock();
        builder->setInsertInto(block);

        ShortList<IRParam*> params;
        for (UInt i = 0; i < funcTypeInInterface->getParamCount(); i++)
        {
            params.add(builder->emitParam(funcTypeInInterface->getParamType(i)));
        }

        List<IRInst*> args;
        List<ArgumentPackWorkItem> argsToPack;

        SLANG_ASSERT(params.getCount() == (Index)func->getParamCount());
        for (UInt i = 0; i < func->getParamCount(); i++)
        {
            auto wrapperParam = params[i];
            // Type of the parameter in the callee.
            auto funcParamType = func->getParamType(i);

            // If the implementation expects a concrete type
            // (either in the form of a pointer for `out`/`inout` parameters,
            // or in the form a value for `in` parameters, while
            // the interface exposes an AnyValue type,
            // we need to unpack the AnyValue argument to the appropriate
            // concerete type.
            ArgumentPackWorkItem packWorkItem;
            auto newArg = maybeUnpackArg(builder, funcParamType, wrapperParam, packWorkItem);
            args.add(newArg);
            if (packWorkItem.concreteArg)
                argsToPack.add(packWorkItem);
        }
        auto call = builder->emitCallInst(func->getResultType(), func, args);

        // Pack all `out` arguments.
        for (auto item : argsToPack)
        {
            auto anyValType = cast<IRPtrTypeBase>(item.dstArg->getDataType())->getValueType();
            auto concreteVal = builder->emitLoad(item.concreteArg);
            auto packedVal = builder->emitPackAnyValue(anyValType, concreteVal);
            builder->emitStore(item.dstArg, packedVal);
        }

        // Pack return value if necessary.
        if (!as<IRAnyValueType>(call->getDataType()) &&
            as<IRAnyValueType>(funcTypeInInterface->getResultType()))
        {
            auto pack = builder->emitPackAnyValue(funcTypeInInterface->getResultType(), call);
            builder->emitReturn(pack);
        }
        else
        {
            if (call->getDataType()->getOp() == kIROp_VoidType)
                builder->emitReturn();
            else
                builder->emitReturn(call);
        }
        return wrapperFunc;
    }

    void lowerWitnessTable(IRWitnessTable* witnessTable)
    {
        auto interfaceType = cast<IRInterfaceType>(witnessTable->getConformanceType());
        if (isBuiltin(interfaceType))
            return;
        if (isComInterfaceType(interfaceType))
            return;

        // We need to consider whether the concrete type that is conforming
        // in this witness table actually fits within the declared any-value
        // size for the interface.
        //
        // If the type doesn't fit then it would be invalid to use for dynamic
        // dispatch, and the packing/unpacking operations we emit would fail
        // to generate valid code.
        //
        // Such a type might still be useful for static specialization, so
        // we can't consider this case a hard error.
        //
        auto concreteType = witnessTable->getConcreteType();
        IRIntegerValue typeSize, sizeLimit;
        bool isTypeOpaque = false;
        if (!sharedContext->doesTypeFitInAnyValue(
                concreteType,
                interfaceType,
                &typeSize,
                &sizeLimit,
                &isTypeOpaque))
        {
            HashSet<IRType*> visited;
            if (isTypeOpaque)
            {
                sharedContext->sink->diagnose(
                    concreteType,
                    Diagnostics::typeCannotBePackedIntoAnyValue,
                    concreteType);
            }
            else
            {
                sharedContext->sink->diagnose(
                    concreteType,
                    Diagnostics::typeDoesNotFitAnyValueSize,
                    concreteType);
                sharedContext->sink->diagnoseWithoutSourceView(
                    concreteType,
                    Diagnostics::typeAndLimit,
                    concreteType,
                    typeSize,
                    sizeLimit);
            }
            return;
        }

        for (auto child : witnessTable->getChildren())
        {
            auto entry = as<IRWitnessTableEntry>(child);
            if (!entry)
                continue;
            auto interfaceRequirementVal = sharedContext->findInterfaceRequirementVal(
                interfaceType,
                entry->getRequirementKey());
            if (auto ordinaryFunc = as<IRFunc>(entry->getSatisfyingVal()))
            {
                auto wrapper = emitWitnessTableWrapper(ordinaryFunc, interfaceRequirementVal);
                entry->satisfyingVal.set(wrapper);
                sharedContext->addToWorkList(wrapper);
            }
        }
    }

    void processInst(IRInst* inst)
    {
        if (auto witnessTable = as<IRWitnessTable>(inst))
        {
            lowerWitnessTable(witnessTable);
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

void generateWitnessTableWrapperFunctions(SharedGenericsLoweringContext* sharedContext)
{
    GenerateWitnessTableWrapperContext context;
    context.sharedContext = sharedContext;
    context.processModule();
}

} // namespace Slang
