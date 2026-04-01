// slang-ir-lower-generic-call.cpp
#include "slang-ir-lower-generic-call.h"

#include "slang-ir-generics-lowering-context.h"
#include "slang-ir-util.h"

namespace Slang
{
struct GenericCallLoweringContext
{
    SharedGenericsLoweringContext* sharedContext;

    // Represents a work item for unpacking `inout` or `out` arguments after a generic call.
    struct ArgumentUnpackWorkItem
    {
        // Concrete typed destination.
        IRInst* dstArg = nullptr;
        // Packed argument.
        IRInst* packedArg = nullptr;
    };

    // Packs `arg` into a `IRAnyValue` if necessary, to make it feedable into the parameter.
    // If `arg` represents a concrete typed variable passed in to a generic `out` parameter,
    // this function indicates that it needs to be unpacked after the call by setting
    // `unpackAfterCall`.
    IRInst* maybePackArgument(
        IRBuilder* builder,
        IRType* paramType,
        IRInst* arg,
        ArgumentUnpackWorkItem& unpackAfterCall)
    {
        unpackAfterCall.dstArg = nullptr;
        unpackAfterCall.packedArg = nullptr;

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

        // Pack `arg` if the parameter expects AnyValue but
        // `arg` is not an AnyValue.
        if (as<IRAnyValueType>(paramValType) && !as<IRAnyValueType>(argValType))
        {
            auto packedArgVal = builder->emitPackAnyValue(paramValType, argVal);
            // if parameter expects an `out` pointer, store the packed val into a
            // variable and pass in a pointer to that variable.
            if (as<IRPtrTypeBase>(paramType))
            {
                auto tempVar = builder->emitVar(paramValType);
                builder->emitStore(tempVar, packedArgVal);
                // tempVar needs to be unpacked into original var after the call.
                unpackAfterCall.dstArg = arg;
                unpackAfterCall.packedArg = tempVar;
                return tempVar;
            }
            else
            {
                return packedArgVal;
            }
        }
        return arg;
    }

    IRInst* maybeUnpackValue(
        IRBuilder* builder,
        IRType* expectedType,
        IRType* actualType,
        IRInst* value)
    {
        if (as<IRAnyValueType>(actualType) && !as<IRAnyValueType>(expectedType))
        {
            auto unpack = builder->emitUnpackAnyValue(expectedType, value);
            return unpack;
        }
        return value;
    }

    // Create a dispatch function for a interface method.
    // On CPU, the dispatch function is implemented as a witness table lookup followed by
    // a function-pointer call.
    // On GPU targets, we can modify the body of the dispatch function in a follow-up
    // pass to implement it with a `switch` statement based on the type ID.
    IRFunc* _createInterfaceDispatchMethod(
        IRBuilder* builder,
        IRInterfaceType* interfaceType,
        IRInst* requirementKey,
        IRInst* requirementVal)
    {
        auto func = builder->createFunc();
        if (auto linkage = requirementKey->findDecoration<IRLinkageDecoration>())
        {
            builder->addNameHintDecoration(func, linkage->getMangledName());
        }

        auto reqFuncType = cast<IRFuncType>(requirementVal);
        List<IRType*> paramTypes;
        paramTypes.add(builder->getWitnessTableType(interfaceType));
        for (UInt i = 0; i < reqFuncType->getParamCount(); i++)
        {
            paramTypes.add(reqFuncType->getParamType(i));
        }
        auto dispatchFuncType = builder->getFuncType(paramTypes, reqFuncType->getResultType());
        func->setFullType(dispatchFuncType);
        builder->setInsertInto(func);
        builder->emitBlock();
        List<IRInst*> params;
        IRParam* witnessTableParam = builder->emitParam(paramTypes[0]);
        for (Index i = 1; i < paramTypes.getCount(); i++)
        {
            params.add(builder->emitParam(paramTypes[i]));
        }
        auto callee =
            builder->emitLookupInterfaceMethodInst(reqFuncType, witnessTableParam, requirementKey);
        auto call = (IRCall*)builder->emitCallInst(reqFuncType->getResultType(), callee, params);
        if (call->getDataType()->getOp() == kIROp_VoidType)
            builder->emitReturn();
        else
            builder->emitReturn(call);
        return func;
    }

    // If an interface dispatch method is already created, return it.
    // Otherwise, create the method.
    IRFunc* getOrCreateInterfaceDispatchMethod(
        IRBuilder* builder,
        IRInterfaceType* interfaceType,
        IRInst* requirementKey,
        IRInst* requirementVal)
    {
        if (auto func = sharedContext->mapInterfaceRequirementKeyToDispatchMethods.tryGetValue(
                requirementKey))
            return *func;
        auto dispatchFunc =
            _createInterfaceDispatchMethod(builder, interfaceType, requirementKey, requirementVal);
        sharedContext->mapInterfaceRequirementKeyToDispatchMethods.addIfNotExists(
            requirementKey,
            dispatchFunc);
        return dispatchFunc;
    }

    // Translate `callInst` into a call of `newCallee`, and respect the new `funcType`.
    // If `newCallee` is a lowered generic function, `specializeInst` contains the type
    // arguments used to specialize the callee.
    void translateCallInst(
        IRCall* callInst,
        IRFuncType* funcType,
        IRInst* newCallee,
        IRSpecialize* specializeInst)
    {
        List<IRType*> paramTypes;
        for (UInt i = 0; i < funcType->getParamCount(); i++)
            paramTypes.add(funcType->getParamType(i));

        IRBuilder builderStorage(sharedContext->module);
        auto builder = &builderStorage;
        builder->setInsertBefore(callInst);

        // Process the argument list of the call.
        // For each argument, we test if it needs to be packed into an `AnyValue` for the
        // call. For `out` and `inout` parameters, they may also need to be unpacked after
        // the call, in which case we add such the argument to `argsToUnpack` so it can be
        // processed after the new call inst is emitted.
        List<IRInst*> args;
        List<ArgumentUnpackWorkItem> argsToUnpack;
        for (UInt i = 0; i < callInst->getArgCount(); i++)
        {
            auto arg = callInst->getArg(i);
            ArgumentUnpackWorkItem unpackWorkItem;
            auto newArg = maybePackArgument(builder, paramTypes[i], arg, unpackWorkItem);
            args.add(newArg);
            if (unpackWorkItem.packedArg)
                argsToUnpack.add(unpackWorkItem);
        }
        if (specializeInst)
        {
            for (UInt i = 0; i < specializeInst->getArgCount(); i++)
            {
                auto arg = specializeInst->getArg(i);
                // Translate Type arguments into RTTI object.
                if (as<IRType>(arg))
                {
                    // We are using a simple type to specialize a callee.
                    // Generate RTTI for this type.
                    auto rttiObject = sharedContext->maybeEmitRTTIObject(arg);
                    arg = builder->emitGetAddress(builder->getRTTIHandleType(), rttiObject);
                }
                else if (arg->getOp() == kIROp_Specialize)
                {
                    // The type argument used to specialize a callee is itself a
                    // specialization of some generic type.
                    // TODO: generate RTTI object for specializations of generic types.
                    SLANG_UNIMPLEMENTED_X("RTTI object generation for generic types");
                }
                else if (arg->getOp() == kIROp_RTTIObject)
                {
                    // We are inside a generic function and using a generic parameter
                    // to specialize another callee. The generic parameter of the caller
                    // has already been translated into an RTTI object, so we just need
                    // to pass this object down.
                }
                args.add(arg);
            }
        }

        // If callee returns `AnyValue` but we are expecting a concrete value, unpack it.
        auto calleeRetType = funcType->getResultType();
        auto newCall = builder->emitCallInst(calleeRetType, newCallee, args);
        auto callInstType = callInst->getDataType();
        auto unpackInst = maybeUnpackValue(builder, callInstType, calleeRetType, newCall);
        // Unpack other `out` arguments.
        for (auto& item : argsToUnpack)
        {
            auto packedVal = builder->emitLoad(item.packedArg);
            auto originalValType = cast<IRPtrTypeBase>(item.dstArg->getDataType())->getValueType();
            auto unpackedVal = builder->emitUnpackAnyValue(originalValType, packedVal);
            builder->emitStore(item.dstArg, unpackedVal);
        }
        callInst->replaceUsesWith(unpackInst);
        callInst->removeAndDeallocate();
    }

    IRInst* findInnerMostSpecializingBase(IRSpecialize* inst)
    {
        auto result = inst->getBase();
        while (auto specialize = as<IRSpecialize>(result))
            result = specialize->getBase();
        return result;
    }

    void lowerCallToSpecializedFunc(IRCall* callInst, IRSpecialize* specializeInst)
    {
        // If we see a call(specialize(gFunc, Targs), args),
        // translate it into call(gFunc, args, Targs).
        auto loweredFunc = specializeInst->getBase();

        // Don't process intrinsic functions.
        UnownedStringSlice intrinsicDef;
        IRInst* intrinsicInst;
        if (findTargetIntrinsicDefinition(
                getResolvedInstForDecorations(loweredFunc),
                sharedContext->targetProgram->getTargetReq()->getTargetCaps(),
                intrinsicDef,
                intrinsicInst))
            return;

        // All callees should have already been lowered in lower-generic-functions pass.
        // For intrinsic generic functions, they are left as is, and we also need to ignore
        // them here.
        if (loweredFunc->getOp() == kIROp_Generic)
        {
            return;
        }
        else if (loweredFunc->getOp() == kIROp_Specialize)
        {
            // All nested generic functions are supposed to be flattend before this pass.
            // If they are not, they represent an intrinsic function that should not be
            // modified in this pass.
            SLANG_UNEXPECTED("Nested generics specialization.");
        }
        else if (loweredFunc->getOp() == kIROp_LookupWitness)
        {
            lowerCallToInterfaceMethod(
                callInst,
                cast<IRLookupWitnessMethod>(loweredFunc),
                specializeInst);
            return;
        }
        IRFuncType* funcType = cast<IRFuncType>(loweredFunc->getDataType());
        translateCallInst(callInst, funcType, loweredFunc, specializeInst);
    }

    void lowerCallToInterfaceMethod(
        IRCall* callInst,
        IRLookupWitnessMethod* lookupInst,
        IRSpecialize* specializeInst)
    {
        // If we see a call(lookup_interface_method(...), ...), we need to translate
        // all occurences of associatedtypes.

        // If `w` in `lookup_interface_method(w, ...)` is a COM interface, bail.
        if (isComInterfaceType(lookupInst->getWitnessTable()->getDataType()))
        {
            return;
        }

        auto interfaceType = cast<IRInterfaceType>(
            cast<IRWitnessTableTypeBase>(lookupInst->getWitnessTable()->getDataType())
                ->getConformanceType());
        if (isBuiltin(interfaceType))
            return;

        IRBuilder builderStorage(sharedContext->module);
        auto builder = &builderStorage;
        builder->setInsertBefore(callInst);

        // Create interface dispatch method that bottlenecks the dispatch logic.
        auto requirementKey = lookupInst->getRequirementKey();
        auto requirementVal =
            sharedContext->findInterfaceRequirementVal(interfaceType, requirementKey);

        if (interfaceType->findDecoration<IRSpecializeDecoration>())
        {
            sharedContext->sink->diagnose(
                callInst->sourceLoc,
                Diagnostics::dynamicDispatchOnSpecializeOnlyInterface,
                interfaceType);
        }
        auto dispatchFunc = getOrCreateInterfaceDispatchMethod(
            builder,
            interfaceType,
            requirementKey,
            requirementVal);

        auto parentFunc = getParentFunc(callInst);
        // Don't process the call inst that is the one in the dispatch function itself.
        if (parentFunc == dispatchFunc)
            return;

        // Replace `callInst` with a new call inst that calls `dispatchFunc` instead, and
        // with the witness table as first argument,
        builder->setInsertBefore(callInst);
        List<IRInst*> newArgs;
        newArgs.add(lookupInst->getWitnessTable());
        for (UInt i = 0; i < callInst->getArgCount(); i++)
            newArgs.add(callInst->getArg(i));
        auto newCall =
            (IRCall*)builder->emitCallInst(callInst->getFullType(), dispatchFunc, newArgs);
        callInst->replaceUsesWith(newCall);
        callInst->removeAndDeallocate();

        // Translate the new call inst as normal, taking care of packing/unpacking inputs
        // and outputs.
        translateCallInst(
            newCall,
            cast<IRFuncType>(dispatchFunc->getFullType()),
            dispatchFunc,
            specializeInst);
    }

    void lowerCall(IRCall* callInst)
    {
        if (auto specializeInst = as<IRSpecialize>(callInst->getCallee()))
            lowerCallToSpecializedFunc(callInst, specializeInst);
        else if (auto lookupInst = as<IRLookupWitnessMethod>(callInst->getCallee()))
            lowerCallToInterfaceMethod(callInst, lookupInst, nullptr);
    }

    void processInst(IRInst* inst)
    {
        if (auto callInst = as<IRCall>(inst))
        {
            lowerCall(callInst);
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

void lowerGenericCalls(SharedGenericsLoweringContext* sharedContext)
{
    GenericCallLoweringContext context;
    context.sharedContext = sharedContext;
    context.processModule();
}

} // namespace Slang
