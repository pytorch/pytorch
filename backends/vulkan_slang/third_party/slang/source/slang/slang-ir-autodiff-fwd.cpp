// slang-ir-autodiff-fwd.cpp
#include "slang-ir-autodiff-fwd.h"

#include "slang-ir-addr-inst-elimination.h"
#include "slang-ir-autodiff.h"
#include "slang-ir-clone.h"
#include "slang-ir-dce.h"
#include "slang-ir-eliminate-phis.h"
#include "slang-ir-init-local-var.h"
#include "slang-ir-inline.h"
#include "slang-ir-inst-pass-base.h"
#include "slang-ir-single-return.h"
#include "slang-ir-ssa-simplification.h"
#include "slang-ir-util.h"
#include "slang-ir-validate.h"

namespace Slang
{

IRFuncType* ForwardDiffTranscriber::differentiateFunctionType(
    IRBuilder* builder,
    IRInst* func,
    IRFuncType* funcType)
{
    SLANG_UNUSED(func);

    List<IRType*> newParameterTypes;
    IRType* diffReturnType;

    for (UIndex i = 0; i < funcType->getParamCount(); i++)
    {
        auto origType = funcType->getParamType(i);
        origType = (IRType*)findOrTranscribePrimalInst(builder, origType);
        if (auto diffPairType = tryGetDiffPairType(builder, origType))
            newParameterTypes.add(diffPairType);
        else
            newParameterTypes.add(origType);
    }

    // Transcribe return type to a pair.
    // This will be void if the primal return type is non-differentiable.
    //
    auto origResultType = (IRType*)findOrTranscribePrimalInst(builder, funcType->getResultType());
    if (auto returnPairType = tryGetDiffPairType(builder, origResultType))
        diffReturnType = returnPairType;
    else
        diffReturnType = origResultType;

    return builder->getFuncType(newParameterTypes, diffReturnType);
}

void ForwardDiffTranscriber::generateTrivialFwdDiffFunc(IRFunc* primalFunc, IRFunc* diffFunc)
{
    IRBuilder builder(diffFunc);
    builder.setInsertInto(diffFunc);
    auto block = builder.emitBlock();
    builder.markInstAsMixedDifferential(block);

    for (auto param : primalFunc->getParams())
    {
        transcribeFuncParam(&builder, param, param->getFullType());
    }
    List<IRParam*> diffParams;
    for (auto param : diffFunc->getParams())
    {
        diffParams.add(param);
    }
    auto emitDiffPairVal = [&](IRDifferentialPairTypeBase* pairType)
    {
        auto primal = builder.emitDefaultConstruct(pairType->getValueType());
        builder.markInstAsPrimal(primal);
        auto diff = getDifferentialZeroOfType(&builder, pairType->getValueType());
        builder.markInstAsDifferential(diff, primal->getDataType());

        auto val = builder.emitMakeDifferentialPair(pairType, primal, diff);
        builder.markInstAsMixedDifferential(val);

        return val;
    };
    for (auto param : diffParams)
    {
        if (auto outType = as<IROutTypeBase>(param->getFullType()))
        {
            if (isRelevantDifferentialPair(outType))
            {
                auto pairType = as<IRDifferentialPairTypeBase>(outType->getValueType());
                auto val = emitDiffPairVal(pairType);
                auto store = builder.emitStore(param, val);
                builder.markInstAsMixedDifferential(store);
            }
            else
            {
                auto val = builder.emitDefaultConstruct(outType->getValueType());
                builder.markInstAsPrimal(val);

                auto store = builder.emitStore(param, val);
                builder.markInstAsPrimal(store);
            }
        }
    }
    if (isRelevantDifferentialPair(diffFunc->getResultType()))
    {
        auto pairType = as<IRDifferentialPairTypeBase>(diffFunc->getResultType());
        auto val = emitDiffPairVal(pairType);
        auto returnInst = builder.emitReturn(val);
        builder.markInstAsMixedDifferential(val);
        builder.markInstAsMixedDifferential(returnInst);
    }
    else
    {
        auto retVal = builder.emitDefaultConstruct(diffFunc->getResultType());
        auto returnInst = builder.emitReturn(retVal);
        builder.markInstAsPrimal(retVal);
        builder.markInstAsPrimal(returnInst);
    }
}

// Returns "d<var-name>" to use as a name hint for variables and parameters.
// If no primal name is available, returns a blank string.
//
String ForwardDiffTranscriber::getJVPVarName(IRInst* origVar)
{
    if (auto namehintDecoration = origVar->findDecoration<IRNameHintDecoration>())
    {
        return ("d" + String(namehintDecoration->getName()));
    }

    return String("");
}

InstPair ForwardDiffTranscriber::transcribeUndefined(IRBuilder* builder, IRInst* origInst)
{
    auto primalVal = maybeCloneForPrimalInst(builder, origInst);

    if (IRType* const diffType = differentiateType(builder, origInst->getFullType()))
    {
        auto dzero = getDifferentialZeroOfType(builder, origInst->getFullType());
        if (dzero)
        {
            return InstPair(primalVal, dzero);
        }
    }
    return InstPair(primalVal, nullptr);
}

InstPair ForwardDiffTranscriber::transcribeReinterpret(IRBuilder* builder, IRInst* origInst)
{
    auto primalVal = maybeCloneForPrimalInst(builder, origInst);

    IRInst* diffVal = nullptr;

    if (IRType* const diffType = differentiateType(builder, origInst->getFullType()))
    {
        if (auto diffOperand = findOrTranscribeDiffInst(builder, origInst->getOperand(0)))
        {
            diffVal = builder->emitReinterpret(diffType, diffOperand);
        }
    }

    return InstPair(primalVal, diffVal);
}

InstPair ForwardDiffTranscriber::transcribeDifferentiableTypeAnnotation(
    IRBuilder* builder,
    IRInst* origInst)
{
    auto primalAnnotation =
        as<IRDifferentiableTypeAnnotation>(maybeCloneForPrimalInst(builder, origInst));

    IRDifferentiableTypeAnnotation* annotation = as<IRDifferentiableTypeAnnotation>(origInst);

    differentiableTypeConformanceContext.addTypeToDictionary(
        (IRType*)primalAnnotation->getBaseType(),
        primalAnnotation->getWitness());

    auto diffType = differentiateType(builder, (IRType*)annotation->getBaseType());
    if (!diffType)
        return InstPair(primalAnnotation, nullptr);

    auto diffTypeDiffWitness =
        tryGetDifferentiableWitness(builder, diffType, DiffConformanceKind::Any);

    IRInst* args[] = {diffType, diffTypeDiffWitness};

    auto diffAnnotation = builder->emitIntrinsicInst(
        builder->getVoidType(),
        kIROp_DifferentiableTypeAnnotation,
        2,
        args);

    builder->markInstAsPrimal(diffAnnotation);
    builder->markInstAsPrimal(primalAnnotation);

    return InstPair(primalAnnotation, diffAnnotation);
}

InstPair ForwardDiffTranscriber::transcribeVar(IRBuilder* builder, IRVar* origVar)
{
    if (IRType* diffType = differentiateType(builder, origVar->getDataType()->getValueType()))
    {
        IRVar* diffVar = builder->emitVar(diffType);
        SLANG_ASSERT(diffVar);

        auto diffNameHint = getJVPVarName(origVar);
        if (diffNameHint.getLength() > 0)
            builder->addNameHintDecoration(diffVar, diffNameHint.getUnownedSlice());

        return InstPair(maybeCloneForPrimalInst(builder, origVar), diffVar);
    }
    return InstPair(maybeCloneForPrimalInst(builder, origVar), nullptr);
}

InstPair ForwardDiffTranscriber::transcribeBinaryArith(IRBuilder* builder, IRInst* origArith)
{
    SLANG_ASSERT(origArith->getOperandCount() == 2);

    IRInst* primalArith = maybeCloneForPrimalInst(builder, origArith);

    auto origLeft = origArith->getOperand(0);
    auto origRight = origArith->getOperand(1);

    auto primalLeft = findOrTranscribePrimalInst(builder, origLeft);
    auto primalRight = findOrTranscribePrimalInst(builder, origRight);

    auto diffLeft = findOrTranscribeDiffInst(builder, origLeft);
    auto diffRight = findOrTranscribeDiffInst(builder, origRight);


    if (diffLeft || diffRight)
    {
        diffLeft =
            diffLeft ? diffLeft : getDifferentialZeroOfType(builder, primalLeft->getDataType());

        bool diffRightIsZero = (diffRight == nullptr);
        diffRight =
            diffRight ? diffRight : getDifferentialZeroOfType(builder, primalRight->getDataType());
        diffRightIsZero = diffRightIsZero || isZero(diffRight);

        auto resultType = primalArith->getDataType();
        auto origResultType = origArith->getDataType();
        auto diffType = (IRType*)differentiateType(builder, origResultType);

        switch (origArith->getOp())
        {
        case kIROp_Add:
            {
                auto diffAdd = builder->emitAdd(diffType, diffLeft, diffRight);
                builder->markInstAsDifferential(diffAdd, resultType);

                return InstPair(primalArith, diffAdd);
            }

        case kIROp_Mul:
            {
                auto diffLeftTimesRight = builder->emitMul(diffType, diffLeft, primalRight);
                auto diffRightTimesLeft = builder->emitMul(diffType, diffRight, primalLeft);
                builder->markInstAsDifferential(diffLeftTimesRight, resultType);
                builder->markInstAsDifferential(diffRightTimesLeft, resultType);

                auto diffAdd = builder->emitAdd(diffType, diffLeftTimesRight, diffRightTimesLeft);
                builder->markInstAsDifferential(diffAdd, resultType);

                return InstPair(primalArith, diffAdd);
            }

        case kIROp_Sub:
            {
                auto diffSub = builder->emitSub(diffType, diffLeft, diffRight);
                builder->markInstAsDifferential(diffSub, resultType);

                return InstPair(primalArith, diffSub);
            }
        case kIROp_Div:
            {
                if (diffRightIsZero)
                {
                    // Special case the dRight = 0 case here since it would be difficult
                    // to optimize out in the future.
                    IRInst* diff = nullptr;
                    if (auto constant = as<IRFloatLit>(primalRight))
                    {
                        diff = builder->emitMul(
                            diffType,
                            diffLeft,
                            builder->getFloatValue(
                                constant->getDataType(),
                                1.0 / constant->getValue()));
                        builder->markInstAsDifferential(diff, resultType);
                    }
                    else
                    {
                        diff = builder->emitDiv(diffType, diffLeft, primalRight);
                        builder->markInstAsDifferential(diff, resultType);
                    }
                    return InstPair(primalArith, diff);
                }
                else
                {
                    auto diffLeftTimesRight = builder->emitMul(diffType, diffLeft, primalRight);
                    builder->markInstAsDifferential(diffLeftTimesRight, resultType);

                    auto diffRightTimesLeft = builder->emitMul(diffType, primalLeft, diffRight);
                    builder->markInstAsDifferential(diffRightTimesLeft, resultType);

                    auto diffSub =
                        builder->emitSub(diffType, diffLeftTimesRight, diffRightTimesLeft);
                    builder->markInstAsDifferential(diffSub, resultType);

                    auto diffMul =
                        builder->emitMul(primalRight->getFullType(), primalRight, primalRight);
                    builder->markInstAsPrimal(diffMul);

                    auto diffDiv = builder->emitDiv(diffType, diffSub, diffMul);
                    builder->markInstAsDifferential(diffDiv, resultType);

                    return InstPair(primalArith, diffDiv);
                }
            }
        default:
            getSink()->diagnose(
                origArith->sourceLoc,
                Diagnostics::unimplemented,
                "this arithmetic instruction cannot be differentiated");
        }
    }

    return InstPair(primalArith, nullptr);
}

InstPair ForwardDiffTranscriber::transcribeBinaryLogic(IRBuilder* builder, IRInst* origLogic)
{
    SLANG_ASSERT(origLogic->getOperandCount() == 2);

    // Boolean operations are not differentiable. For the linearization
    // pass, we do not need to do anything but copy them over to the ne
    // function.
    auto primalLogic = maybeCloneForPrimalInst(builder, origLogic);
    return InstPair(primalLogic, nullptr);
}

InstPair ForwardDiffTranscriber::transcribeSelect(IRBuilder* builder, IRInst* origSelect)
{
    auto primalCondition = lookupPrimalInst(builder, origSelect->getOperand(0));

    auto origLeft = origSelect->getOperand(1);
    auto origRight = origSelect->getOperand(2);

    auto primalLeft = findOrTranscribePrimalInst(builder, origLeft);
    auto primalRight = findOrTranscribePrimalInst(builder, origRight);

    auto diffLeft = findOrTranscribeDiffInst(builder, origLeft);
    auto diffRight = findOrTranscribeDiffInst(builder, origRight);

    auto primalSelect = maybeCloneForPrimalInst(builder, origSelect);

    // If both sides have no differential, skip
    if (diffLeft || diffRight)
    {
        diffLeft =
            diffLeft ? diffLeft : getDifferentialZeroOfType(builder, primalLeft->getDataType());
        diffRight =
            diffRight ? diffRight : getDifferentialZeroOfType(builder, primalRight->getDataType());

        auto diffType = differentiateType(builder, origSelect->getDataType());

        return InstPair(
            primalSelect,
            builder->emitIntrinsicInst(
                diffType,
                kIROp_Select,
                3,
                List<IRInst*>(primalCondition, diffLeft, diffRight).getBuffer()));
    }

    return InstPair(primalSelect, nullptr);
}

InstPair ForwardDiffTranscriber::transcribeLoad(IRBuilder* builder, IRLoad* origLoad)
{
    auto origPtr = origLoad->getPtr();
    auto primalPtr = lookupPrimalInst(builder, origPtr, nullptr);

    if (auto primalPtrType = as<IRPtrTypeBase>(primalPtr->getFullType()))
    {
        if (auto diffPairType = as<IRDifferentialPairType>(primalPtrType->getValueType()))
        {
            // Special case load from an `out` param, which will not have corresponding `diff` and
            // `primal` insts yet.

            // TODO: Could we move this load to _after_ DifferentialPairGetPrimal,
            // and DifferentialPairGetDifferential?
            //
            auto load = builder->emitLoad(primalPtr);
            builder->markInstAsMixedDifferential(load, diffPairType);

            auto primalElement = builder->emitDifferentialPairGetPrimal(load);
            auto diffElement = builder->emitDifferentialPairGetDifferential(
                (IRType*)differentiableTypeConformanceContext.getDiffTypeFromPairType(
                    builder,
                    diffPairType),
                load);
            return InstPair(primalElement, diffElement);
        }
        else if (
            auto diffPtrPairType = as<IRDifferentialPtrPairType>(primalPtrType->getValueType()))
        {
            auto load = builder->emitLoad(primalPtr);
            builder->markInstAsPrimal(load);

            auto primalElement = builder->emitDifferentialPtrPairGetPrimal(load);
            auto diffElement = builder->emitDifferentialPtrPairGetDifferential(
                (IRType*)differentiableTypeConformanceContext.getDiffTypeFromPairType(
                    builder,
                    diffPtrPairType),
                load);
            builder->markInstAsPrimal(primalElement);
            builder->markInstAsPrimal(diffElement);
            return InstPair(primalElement, diffElement);
        }
    }

    auto primalLoad = maybeCloneForPrimalInst(builder, origLoad);
    IRInst* diffLoad = nullptr;
    if (auto diffPtr = lookupDiffInst(origPtr, nullptr))
    {
        // Default case, we're loading from a known differential inst.
        diffLoad = as<IRLoad>(builder->emitLoad(diffPtr));
    }
    return InstPair(primalLoad, diffLoad);
}

InstPair ForwardDiffTranscriber::transcribeStore(IRBuilder* builder, IRStore* origStore)
{
    IRInst* origStoreLocation = origStore->getPtr();
    IRInst* origStoreVal = origStore->getVal();
    auto primalStoreLocation = lookupPrimalInst(builder, origStoreLocation, nullptr);
    auto diffStoreLocation = lookupDiffInst(origStoreLocation, nullptr);
    auto primalStoreVal = lookupPrimalInst(builder, origStoreVal, nullptr);
    auto diffStoreVal = lookupDiffInst(origStoreVal, nullptr);

    if (!diffStoreLocation)
    {
        auto primalLocationPtrType = as<IRPtrTypeBase>(primalStoreLocation->getDataType());
        if (auto diffPairType = as<IRDifferentialPairType>(primalLocationPtrType->getValueType()))
        {
            auto valToStore =
                builder->emitMakeDifferentialPair(diffPairType, primalStoreVal, diffStoreVal);
            builder->markInstAsMixedDifferential(diffStoreVal, diffPairType);

            auto store = builder->emitStore(primalStoreLocation, valToStore);
            builder->markInstAsMixedDifferential(store, diffPairType);

            return InstPair(store, nullptr);
        }
        else if (
            auto diffRefPairType =
                as<IRDifferentialPtrPairType>(primalLocationPtrType->getValueType()))
        {
            auto valToStore =
                builder->emitMakeDifferentialPtrPair(diffRefPairType, primalStoreVal, diffStoreVal);
            builder->markInstAsPrimal(valToStore);

            auto store = builder->emitStore(primalStoreLocation, valToStore);
            builder->markInstAsPrimal(store);

            return InstPair(store, nullptr);
        }
    }

    auto primalStore = maybeCloneForPrimalInst(builder, origStore);

    IRInst* diffStore = nullptr;

    // If the stored value has a differential version,
    // emit a store instruction for the differential parameter.
    // Otherwise, emit nothing since there's nothing to load.
    //
    if (diffStoreLocation && diffStoreVal)
    {
        // Default case, storing the entire type (and not a member)
        diffStore = as<IRStore>(builder->emitStore(diffStoreLocation, diffStoreVal));
        markDiffTypeInst(builder, diffStore, primalStoreVal->getDataType());
        return InstPair(primalStore, diffStore);
    }

    return InstPair(primalStore, nullptr);
}

// Since int/float literals are sometimes nested inside an IRConstructor
// instruction, we check to make sure that the nested instr is a constant
// and then return nullptr. Literals do not need to be differentiated.
//
InstPair ForwardDiffTranscriber::transcribeConstruct(IRBuilder* builder, IRInst* origConstruct)
{
    IRInst* primalConstruct = maybeCloneForPrimalInst(builder, origConstruct);

    // Check if the output type can be differentiated. If it cannot be
    // differentiated, don't differentiate the inst
    //
    auto primalConstructType =
        (IRType*)findOrTranscribePrimalInst(builder, origConstruct->getDataType());
    // TODO: Need to update this to generate derivatives on a per-key basis
    if (auto diffConstructType = differentiateType(builder, primalConstructType))
    {
        UCount operandCount = origConstruct->getOperandCount();

        List<IRInst*> diffOperands;
        for (UIndex ii = 0; ii < operandCount; ii++)
        {
            // If the operand has a differential version, replace the original with
            // the differential. Otherwise, use a zero.
            //
            if (auto diffInst = lookupDiffInst(origConstruct->getOperand(ii), nullptr))
                diffOperands.add(diffInst);
            else
            {
                auto operandDataType = origConstruct->getOperand(ii)->getDataType();
                if (const auto diffOperandType = differentiateType(builder, operandDataType))
                {
                    operandDataType = (IRType*)findOrTranscribePrimalInst(builder, operandDataType);
                    diffOperands.add(getDifferentialZeroOfType(builder, operandDataType));
                }
                else
                {
                    diffOperands.add(builder->getVoidValue());
                }
            }
        }

        return InstPair(
            primalConstruct,
            builder->emitIntrinsicInst(
                diffConstructType,
                origConstruct->getOp(),
                diffOperands.getCount(),
                diffOperands.getBuffer()));
    }
    else
    {
        return InstPair(primalConstruct, nullptr);
    }
}

InstPair ForwardDiffTranscriber::transcribeMakeStruct(IRBuilder* builder, IRInst* origMakeStruct)
{
    IRInst* primalMakeStruct = maybeCloneForPrimalInst(builder, origMakeStruct);

    // Check if the output type can be differentiated. If it cannot be
    // differentiated, don't differentiate the inst
    //
    auto primalStructType =
        (IRType*)findOrTranscribePrimalInst(builder, origMakeStruct->getDataType());
    if (auto diffStructType = differentiateType(builder, primalStructType))
    {
        auto primalStruct = as<IRStructType>(getResolvedInstForDecorations(primalStructType));
        SLANG_RELEASE_ASSERT(primalStruct);

        List<IRInst*> diffOperands;
        UIndex ii = 0;
        for (auto field : primalStruct->getFields())
        {
            SLANG_RELEASE_ASSERT(ii < origMakeStruct->getOperandCount());

            // If this field is not differentiable, skip the operand.
            if (!field->getKey()->findDecoration<IRDerivativeMemberDecoration>())
            {
                ii++;
                continue;
            }

            // If the operand has a differential version, replace the original with
            // the differential. Otherwise, use a zero.
            //
            if (auto diffInst = lookupDiffInst(origMakeStruct->getOperand(ii), nullptr))
            {
                diffOperands.add(diffInst);
            }
            else
            {
                auto operandDataType = origMakeStruct->getOperand(ii)->getDataType();
                auto diffOperandType = differentiateType(builder, operandDataType);

                if (diffOperandType)
                {
                    operandDataType = (IRType*)findOrTranscribePrimalInst(builder, operandDataType);
                    diffOperands.add(getDifferentialZeroOfType(builder, operandDataType));
                }
                else
                {
                    // This case is only hit if the field is of a differentiable type but the
                    // operand is of a non-differentiable type. This can happen if the operand is
                    // wrapped in no_diff. In this case, we use the derivative of the field type to
                    // synthesize the 0.
                    //
                    auto diffFieldOperandType = differentiateType(builder, field->getFieldType());
                    SLANG_RELEASE_ASSERT(diffFieldOperandType);
                    diffOperands.add(
                        getDifferentialZeroOfType(builder, (IRType*)diffFieldOperandType));
                }
            }
            ii++;
        }

        return InstPair(
            primalMakeStruct,
            builder->emitIntrinsicInst(
                diffStructType,
                kIROp_MakeStruct,
                diffOperands.getCount(),
                diffOperands.getBuffer()));
    }
    else
    {
        return InstPair(primalMakeStruct, nullptr);
    }
}

static bool _isDifferentiableFunc(IRInst* func)
{
    func = getResolvedInstForDecorations(func);
    for (auto decor = func->getFirstDecoration(); decor; decor = decor->getNextDecoration())
    {
        switch (decor->getOp())
        {
        case kIROp_ForwardDerivativeDecoration:
        case kIROp_ForwardDifferentiableDecoration:
        case kIROp_BackwardDerivativeDecoration:
        case kIROp_BackwardDifferentiableDecoration:
        case kIROp_UserDefinedBackwardDerivativeDecoration:
            return true;
        }
    }
    return false;
}

static IRFuncType* _getCalleeActualFuncType(IRInst* callee)
{
    auto type = callee->getFullType();
    if (auto funcType = as<IRFuncType>(type))
        return funcType;
    if (auto specialize = as<IRSpecialize>(callee))
        return as<IRFuncType>(
            findGenericReturnVal(as<IRGeneric>(specialize->getBase()))->getFullType());
    return nullptr;
}

IRInst* tryFindPrimalSubstitute(IRBuilder* builder, IRInst* callee)
{
    if (auto func = as<IRFunc>(callee))
    {
        if (auto decor = func->findDecoration<IRPrimalSubstituteDecoration>())
            return decor->getPrimalSubstituteFunc();
    }
    else if (auto specialize = as<IRSpecialize>(callee))
    {
        auto innerGen = as<IRGeneric>(specialize->getBase());
        if (!innerGen)
            return callee;
        auto innerFunc = findGenericReturnVal(innerGen);
        if (auto decor = innerFunc->findDecoration<IRPrimalSubstituteDecoration>())
        {
            auto substSpecialize = as<IRSpecialize>(decor->getPrimalSubstituteFunc());
            SLANG_RELEASE_ASSERT(substSpecialize);
            SLANG_RELEASE_ASSERT(substSpecialize->getArgCount() == specialize->getArgCount());
            List<IRInst*> args;
            for (UInt i = 0; i < specialize->getArgCount(); i++)
                args.add(specialize->getArg(i));
            return builder->emitSpecializeInst(
                callee->getFullType(),
                substSpecialize->getBase(),
                (UInt)args.getCount(),
                args.getBuffer());
        }
    }
    return callee;
}

// Differentiating a call instruction here is primarily about generating
// an appropriate call list based on whichever parameters have differentials
// in the current transcription context.
//
InstPair ForwardDiffTranscriber::transcribeCall(IRBuilder* builder, IRCall* origCall)
{

    IRInst* origCallee = origCall->getCallee();

    if (!origCallee)
    {
        // Note that this can only happen if the callee is a result
        // of a higher-order operation. For now, we assume that we cannot
        // differentiate such calls safely.
        // TODO(sai): Should probably get checked in the front-end.
        //
        getSink()->diagnose(
            origCall->sourceLoc,
            Diagnostics::internalCompilerError,
            "attempting to differentiate unresolved callee");

        return InstPair(nullptr, nullptr);
    }

    auto primalCallee = findOrTranscribePrimalInst(builder, origCallee);
    auto substPrimalCallee = tryFindPrimalSubstitute(builder, primalCallee);

    IRInst* diffCallee = nullptr;
    if (substPrimalCallee == primalCallee)
    {
        instMapD.tryGetValue(origCallee, diffCallee);
    }
    else
    {
        if (_isDifferentiableFunc(origCallee))
            diffCallee = findOrTranscribeDiffInst(builder, origCallee);
        primalCallee = substPrimalCallee;
    }

    if (diffCallee)
    {
    }
    else if (
        auto derivativeReferenceDecor =
            primalCallee->findDecoration<IRForwardDerivativeDecoration>())
    {
        // If the user has already provided an differentiated implementation, use that.
        diffCallee = derivativeReferenceDecor->getForwardDerivativeFunc();
    }
    else if (_isDifferentiableFunc(primalCallee))
    {
        // If the function is marked for auto-diff, push a `differentiate` inst for a follow up pass
        // to generate the implementation.
        diffCallee = builder->emitForwardDifferentiateInst(
            differentiateFunctionType(
                builder,
                primalCallee,
                as<IRFuncType>(primalCallee->getFullType())),
            primalCallee);
    }

    if (!diffCallee)
    {
        // The callee is non differentiable, just return primal value with null diff value.
        IRInst* primalCall = maybeCloneForPrimalInst(builder, origCall);
        return InstPair(primalCall, nullptr);
    }

    auto calleeType = _getCalleeActualFuncType(primalCallee);
    SLANG_ASSERT(calleeType);
    SLANG_RELEASE_ASSERT(calleeType->getParamCount() == origCall->getArgCount());

    auto diffCalleeType = _getCalleeActualFuncType(diffCallee);
    SLANG_ASSERT(diffCalleeType);
    SLANG_RELEASE_ASSERT(diffCalleeType->getParamCount() == origCall->getArgCount());

    auto placeholderCall =
        builder->emitCallInst(nullptr, builder->emitUndefined(builder->getTypeKind()), 0, nullptr);
    builder->setInsertBefore(placeholderCall);
    IRBuilder argBuilder = *builder;
    IRBuilder afterBuilder = argBuilder;
    afterBuilder.setInsertAfter(placeholderCall);

    List<IRInst*> args;
    // Go over the parameter list and create pairs for each input (if required)
    for (UIndex ii = 0; ii < origCall->getArgCount(); ii++)
    {
        auto origArg = origCall->getArg(ii);
        auto primalArg = findOrTranscribePrimalInst(&argBuilder, origArg);
        SLANG_ASSERT(primalArg);

        auto origType = origCall->getArg(ii)->getDataType();
        auto primalType = primalArg->getDataType();
        auto originalParamType = calleeType->getParamType(ii);
        auto diffParamType = diffCalleeType->getParamType(ii);
        if (!isNoDiffType(originalParamType))
        {
            if (isNoDiffType(primalType))
            {
                while (auto attrType = as<IRAttributedType>(primalType))
                    primalType = attrType->getBaseType();
                while (auto attrType = as<IRAttributedType>(origType))
                    origType = attrType->getBaseType();
            }
            if (auto pairType = tryGetDiffPairType(&argBuilder, primalType))
            {
                auto pairPtrType = as<IRPtrTypeBase>(pairType);

                auto pairValType = as<IRDifferentialPairTypeBase>(
                    pairPtrType ? pairPtrType->getValueType() : pairType);

                auto diffType = differentiateType(&argBuilder, primalType);
                if (auto ptrParamType = as<IRPtrTypeBase>(diffParamType))
                {
                    // Create temp var to pass in/out arguments.
                    auto srcVar = argBuilder.emitVar(pairValType);
                    markDiffPairTypeInst(&argBuilder, srcVar, pairValType);

                    auto diffArg = findOrTranscribeDiffInst(&argBuilder, origArg);
                    if (ptrParamType->getOp() == kIROp_InOutType)
                    {
                        // Set initial value.
                        auto primalVal = argBuilder.emitLoad(primalArg);
                        auto diffArgVal = diffArg;
                        if (!diffArg)
                            diffArgVal = getDifferentialZeroOfType(
                                builder,
                                (IRType*)pairValType->getValueType());
                        else
                        {
                            diffArgVal = argBuilder.emitLoad(diffArg);
                            markDiffTypeInst(&argBuilder, diffArgVal, pairValType->getValueType());
                        }
                        auto initVal =
                            argBuilder.emitMakeDifferentialPair(pairValType, primalVal, diffArgVal);
                        markDiffPairTypeInst(&argBuilder, initVal, pairValType);
                        auto store = argBuilder.emitStore(srcVar, initVal);
                        markDiffPairTypeInst(&argBuilder, store, pairValType);
                    }
                    if (as<IROutTypeBase>(ptrParamType))
                    {
                        // Read back new value.
                        auto newVal = afterBuilder.emitLoad(srcVar);
                        markDiffPairTypeInst(&afterBuilder, newVal, pairValType);
                        auto newPrimalVal = afterBuilder.emitDifferentialPairGetPrimal(
                            pairValType->getValueType(),
                            newVal);
                        afterBuilder.emitStore(primalArg, newPrimalVal);

                        if (diffArg)
                        {
                            auto newDiffVal = afterBuilder.emitDifferentialPairGetDifferential(
                                (IRType*)as<IRPtrTypeBase>(diffType)->getValueType(),
                                newVal);
                            markDiffTypeInst(
                                &afterBuilder,
                                newDiffVal,
                                pairValType->getValueType());

                            auto storeInst = afterBuilder.emitStore(diffArg, newDiffVal);
                            markDiffTypeInst(&afterBuilder, storeInst, pairValType->getValueType());
                        }
                    }
                    args.add(srcVar);
                    continue;
                }
                else
                {
                    auto diffArg = findOrTranscribeDiffInst(&argBuilder, origArg);
                    if (!diffArg)
                        diffArg = getDifferentialZeroOfType(&argBuilder, primalType);

                    // If a pair type can be formed, this must be non-null.
                    SLANG_RELEASE_ASSERT(diffArg);

                    auto diffPair =
                        argBuilder.emitMakeDifferentialPair(pairType, primalArg, diffArg);
                    markDiffPairTypeInst(&argBuilder, diffPair, pairType);

                    args.add(diffPair);
                    continue;
                }
            }
        }

        {
            // --WORKAROUND--
            // This is a temporary workaround for a very specific case..
            //
            // If all the following are true:
            // 1. the parameter type expects a differential pair,
            // 2. the argument is derived from a no_diff type, and
            // 3. the argument type is a run-time type (i.e. extract_existential_type),
            // then we need to generate a differential 0, but the IR has no
            // information on the diff witness.
            //
            // We will bypass the conformance system & brute-force the lookup for the interface
            // keys, but the proper fix is to lower this key mapping during `no_diff` lowering.
            //

            // Condition 1
            if (differentiableTypeConformanceContext.isDifferentiableType((originalParamType)))
            {
                // Condition 3
                if (auto extractExistentialType = as<IRExtractExistentialType>(primalType))
                {
                    // Condition 2
                    if (isNoDiffType(extractExistentialType->getOperand(0)->getDataType()))
                    {
                        // Force-differentiate the type (this will perform a search for the witness
                        // without going through the diff-type annotation list)
                        //
                        IRInst* witnessTable = nullptr;
                        auto diffType = differentiateExtractExistentialType(
                            &argBuilder,
                            extractExistentialType,
                            witnessTable);

                        auto pairType =
                            getOrCreateDiffPairType(&argBuilder, primalType, witnessTable);
                        auto zeroMethod = argBuilder.emitLookupInterfaceMethodInst(
                            differentiableTypeConformanceContext.sharedContext->zeroMethodType,
                            witnessTable,
                            differentiableTypeConformanceContext.sharedContext
                                ->zeroMethodStructKey);
                        auto diffZero = argBuilder.emitCallInst(diffType, zeroMethod, 0, nullptr);
                        auto diffPair =
                            argBuilder.emitMakeDifferentialPair(pairType, primalArg, diffZero);

                        args.add(diffPair);
                        continue;
                    }
                }
            }
        }

        // Argument is not differentiable.
        // Add original/primal argument.
        args.add(primalArg);
    }

    IRType* diffReturnType = nullptr;
    auto primalReturnType =
        (IRType*)findOrTranscribePrimalInst(&argBuilder, origCall->getFullType());

    diffReturnType = tryGetDiffPairType(&argBuilder, primalReturnType);

    if (!diffReturnType)
    {
        diffReturnType = primalReturnType;
    }

    auto callInst = argBuilder.emitCallInst(diffReturnType, diffCallee, args);
    placeholderCall->removeAndDeallocate();

    argBuilder.markInstAsMixedDifferential(callInst, diffReturnType);
    argBuilder.addAutoDiffOriginalValueDecoration(callInst, primalCallee);

    *builder = afterBuilder;

    if (as<IRDifferentialPairType>(diffReturnType) || as<IRDifferentialPtrPairType>(diffReturnType))
    {
        IRInst* primalResultValue = afterBuilder.emitDifferentialPairGetPrimal(callInst);
        auto diffType = differentiateType(&afterBuilder, origCall->getFullType());
        IRInst* diffResultValue =
            afterBuilder.emitDifferentialPairGetDifferential(diffType, callInst);
        return InstPair(primalResultValue, diffResultValue);
    }
    else
    {
        // Return the inst itself if the return value is non-differentiable.
        // This is fine since these values should only be used by non-differentiable code.
        //
        return InstPair(callInst, callInst);
    }
}

InstPair ForwardDiffTranscriber::transcribeSwizzle(IRBuilder* builder, IRSwizzle* origSwizzle)
{
    IRInst* primalSwizzle = maybeCloneForPrimalInst(builder, origSwizzle);
    if (auto diffBase = lookupDiffInst(origSwizzle->getBase(), nullptr))
    {
        // `diffBase` may exist even if the type is non-differentiable (e.g. IRCall inst that
        // creates other differentiable outputs).
        //
        // We'll check to see if we can get a differential for the type in order to determine
        // whether to generate a differential swizzle inst.
        //
        if (auto diffType = differentiateType(builder, primalSwizzle->getDataType()))
        {
            List<IRInst*> swizzleIndices;
            for (UIndex ii = 0; ii < origSwizzle->getElementCount(); ii++)
                swizzleIndices.add(origSwizzle->getElementIndex(ii));

            return InstPair(
                primalSwizzle,
                builder->emitSwizzle(
                    diffType,
                    diffBase,
                    origSwizzle->getElementCount(),
                    swizzleIndices.getBuffer()));
        }
    }

    return InstPair(primalSwizzle, nullptr);
}

InstPair ForwardDiffTranscriber::transcribeByPassthrough(IRBuilder* builder, IRInst* origInst)
{
    IRInst* primalInst = maybeCloneForPrimalInst(builder, origInst);

    UCount operandCount = origInst->getOperandCount();

    List<IRInst*> diffOperands;
    for (UIndex ii = 0; ii < operandCount; ii++)
    {
        // If the operand has a differential version, replace the original with the
        // differential.
        // Otherwise, abandon the differentiation attempt and assume that origInst
        // cannot (or does not need to) be differentiated.
        //
        if (auto diffInst = lookupDiffInst(origInst->getOperand(ii), nullptr))
            diffOperands.add(diffInst);
        else
            return InstPair(primalInst, nullptr);
    }

    return InstPair(
        primalInst,
        builder->emitIntrinsicInst(
            differentiateType(builder, origInst->getDataType()),
            origInst->getOp(),
            operandCount,
            diffOperands.getBuffer()));
}

InstPair ForwardDiffTranscriber::transcribeControlFlow(IRBuilder* builder, IRInst* origInst)
{
    switch (origInst->getOp())
    {
    case kIROp_unconditionalBranch:
    case kIROp_loop:
        auto origBranch = as<IRUnconditionalBranch>(origInst);
        auto targetBlock = origBranch->getTargetBlock();

        // Grab the differentials for any phi nodes.
        List<IRInst*> newArgs;
        for (UIndex ii = 0; ii < origBranch->getArgCount(); ii++)
        {
            auto origParam = getParamAt(targetBlock, ii);
            auto origArg = origBranch->getArg(ii);
            auto primalArg = lookupPrimalInst(builder, origArg);
            newArgs.add(primalArg);

            if (differentiateType(builder, origParam->getDataType()))
            {
                auto diffArg = lookupDiffInst(origArg, nullptr);
                if (diffArg)
                    newArgs.add(diffArg);
                else
                    newArgs.add(getDifferentialZeroOfType(builder, origArg->getDataType()));
            }
        }

        IRInst* diffBranch = nullptr;
        if (auto diffBlock = findOrTranscribeDiffInst(builder, origBranch->getTargetBlock()))
        {
            if (auto origLoop = as<IRLoop>(origInst))
            {
                auto breakBlock = findOrTranscribeDiffInst(builder, origLoop->getBreakBlock());
                auto continueBlock =
                    findOrTranscribeDiffInst(builder, origLoop->getContinueBlock());
                List<IRInst*> operands;
                operands.add(diffBlock);
                operands.add(breakBlock);
                operands.add(continueBlock);
                operands.addRange(newArgs);
                diffBranch = builder->emitIntrinsicInst(
                    nullptr,
                    kIROp_loop,
                    operands.getCount(),
                    operands.getBuffer());
                if (auto maxItersDecoration = origLoop->findDecoration<IRLoopMaxItersDecoration>())
                    builder->addLoopMaxItersDecoration(
                        diffBranch,
                        maxItersDecoration->getMaxIters());
            }
            else
            {
                diffBranch = builder->emitBranch(
                    as<IRBlock>(diffBlock),
                    newArgs.getCount(),
                    newArgs.getBuffer());
            }
        }

        // For now, every block in the original fn must have a corresponding
        // block to compute *both* primals and derivatives (i.e linearized block)
        SLANG_ASSERT(diffBranch);

        // Since blocks always compute both primals and differentials, the branch
        // instructions are also always mixed.
        //
        builder->markInstAsMixedDifferential(diffBranch);

        return InstPair(diffBranch, diffBranch);
    }

    getSink()->diagnose(
        origInst->sourceLoc,
        Diagnostics::unimplemented,
        "attempting to differentiate unhandled control flow");

    return InstPair(nullptr, nullptr);
}

InstPair ForwardDiffTranscriber::transcribeConst(IRBuilder*, IRInst* origInst)
{
    switch (origInst->getOp())
    {
    case kIROp_FloatLit:
    case kIROp_IntLit:
        return InstPair(origInst, nullptr);
    case kIROp_VoidLit:
        return InstPair(origInst, origInst);
    }

    getSink()->diagnose(
        origInst->sourceLoc,
        Diagnostics::unimplemented,
        "attempting to differentiate unhandled const type");

    return InstPair(nullptr, nullptr);
}

InstPair ForwardDiffTranscriber::transcribeSpecialize(
    IRBuilder* builder,
    IRSpecialize* origSpecialize)
{
    auto primalBase = findOrTranscribePrimalInst(builder, origSpecialize->getBase());
    List<IRInst*> primalArgs;
    for (UInt i = 0; i < origSpecialize->getArgCount(); i++)
    {
        primalArgs.add(findOrTranscribePrimalInst(builder, origSpecialize->getArg(i)));
    }
    auto primalType = findOrTranscribePrimalInst(builder, origSpecialize->getFullType());
    auto primalSpecialize = (IRSpecialize*)builder->emitSpecializeInst(
        (IRType*)primalType,
        primalBase,
        primalArgs.getCount(),
        primalArgs.getBuffer());

    IRInst* diffBase = nullptr;
    if (instMapD.tryGetValue(origSpecialize->getBase(), diffBase))
    {
        auto diffType = differentiateType(builder, origSpecialize->getFullType());
        if (diffBase)
        {
            List<IRInst*> args;
            for (UInt i = 0; i < primalSpecialize->getArgCount(); i++)
            {
                args.add(primalSpecialize->getArg(i));
            }
            auto diffSpecialize =
                builder->emitSpecializeInst(diffType, diffBase, args.getCount(), args.getBuffer());
            return InstPair(primalSpecialize, diffSpecialize);
        }
        else
        {
            return InstPair(primalSpecialize, nullptr);
        }
    }

    auto genericInnerVal = findInnerMostGenericReturnVal(as<IRGeneric>(origSpecialize->getBase()));

    // Right now we don't support transcribing a differentiable callee that is a specialize of a
    // interface lookup (calling differentiable generic interface method). To support it, we need to
    // recursively transcribe the specialization base here.

    if (!genericInnerVal)
        return InstPair(primalSpecialize, nullptr);

    // Look for an IRForwardDerivativeDecoration on the specialize inst.
    // (Normally, this would be on the inner IRFunc, but in this case only the JVP func
    // can be specialized, so we put a decoration on the IRSpecialize)
    //
    if (auto jvpFuncDecoration = origSpecialize->findDecoration<IRForwardDerivativeDecoration>())
    {
        auto jvpFunc = jvpFuncDecoration->getForwardDerivativeFunc();

        // Make sure this isn't itself a specialize .
        SLANG_RELEASE_ASSERT(!as<IRSpecialize>(jvpFunc));

        auto derivativeDecoration =
            genericInnerVal->findDecoration<IRForwardDerivativeDecoration>();
        SLANG_RELEASE_ASSERT(derivativeDecoration);

        return InstPair(primalSpecialize, jvpFunc);
    }
    else if (
        auto derivativeDecoration =
            genericInnerVal->findDecoration<IRForwardDerivativeDecoration>())
    {
        diffBase = derivativeDecoration->getForwardDerivativeFunc();
        List<IRInst*> args;
        for (UInt i = 0; i < primalSpecialize->getArgCount(); i++)
        {
            args.add(primalSpecialize->getArg(i));
        }

        // A `ForwardDerivative` decoration on an inner func of a generic should always be a
        // `specialize`.
        auto diffBaseSpecialize = as<IRSpecialize>(diffBase);
        SLANG_RELEASE_ASSERT(diffBaseSpecialize);

        // Note: this assumes that the generic arguments to specialize the derivative is the same as
        // the generic args to specialize the primal function. This is true for all of our core
        // module functions, but we may need to rely on more general substitution logic here.
        auto diffSpecialize = builder->emitSpecializeInst(
            builder->getTypeKind(),
            diffBaseSpecialize->getBase(),
            args.getCount(),
            args.getBuffer());
        return InstPair(primalSpecialize, diffSpecialize);
    }
    else if (_isDifferentiableFunc(genericInnerVal) || as<IRFuncType>(genericInnerVal))
    {
        List<IRInst*> args;
        for (UInt i = 0; i < primalSpecialize->getArgCount(); i++)
        {
            args.add(primalSpecialize->getArg(i));
        }
        diffBase = findOrTranscribeDiffInst(builder, origSpecialize->getBase());
        auto diffSpecialize = builder->emitSpecializeInst(
            builder->getTypeKind(),
            diffBase,
            args.getCount(),
            args.getBuffer());
        return InstPair(primalSpecialize, diffSpecialize);
    }
    return InstPair(primalSpecialize, nullptr);
}

InstPair ForwardDiffTranscriber::transcribeFieldExtract(IRBuilder* builder, IRInst* originalInst)
{
    SLANG_ASSERT(as<IRFieldExtract>(originalInst) || as<IRFieldAddress>(originalInst));

    IRInst* origBase = originalInst->getOperand(0);
    auto primalBase = findOrTranscribePrimalInst(builder, origBase);
    auto field = originalInst->getOperand(1);
    auto derivativeRefDecor = field->findDecoration<IRDerivativeMemberDecoration>();
    auto primalType = (IRType*)findOrTranscribePrimalInst(builder, originalInst->getDataType());

    IRInst* primalOperands[] = {primalBase, field};
    IRInst* primalFieldExtract =
        builder->emitIntrinsicInst(primalType, originalInst->getOp(), 2, primalOperands);

    if (!derivativeRefDecor)
    {
        return InstPair(primalFieldExtract, nullptr);
    }

    IRInst* diffFieldExtract = nullptr;

    if (auto diffType = differentiateType(builder, originalInst->getDataType()))
    {
        if (auto diffBase = findOrTranscribeDiffInst(builder, origBase))
        {
            IRInst* diffOperands[] = {diffBase, derivativeRefDecor->getDerivativeMemberStructKey()};
            diffFieldExtract =
                builder->emitIntrinsicInst(diffType, originalInst->getOp(), 2, diffOperands);
        }
    }
    return InstPair(primalFieldExtract, diffFieldExtract);
}

InstPair ForwardDiffTranscriber::transcribeGetElement(IRBuilder* builder, IRInst* origGetElementPtr)
{
    SLANG_ASSERT(as<IRGetElement>(origGetElementPtr) || as<IRGetElementPtr>(origGetElementPtr));

    IRInst* origBase = origGetElementPtr->getOperand(0);
    auto primalBase = findOrTranscribePrimalInst(builder, origBase);
    auto primalIndex = findOrTranscribePrimalInst(builder, origGetElementPtr->getOperand(1));

    auto primalType =
        (IRType*)findOrTranscribePrimalInst(builder, origGetElementPtr->getDataType());

    IRInst* primalOperands[] = {primalBase, primalIndex};
    IRInst* primalGetElementPtr =
        builder->emitIntrinsicInst(primalType, origGetElementPtr->getOp(), 2, primalOperands);

    IRInst* diffGetElementPtr = nullptr;

    if (auto diffType = differentiateType(builder, origGetElementPtr->getDataType()))
    {
        if (auto diffBase = findOrTranscribeDiffInst(builder, origBase))
        {
            IRInst* diffOperands[] = {diffBase, primalIndex};
            diffGetElementPtr =
                builder->emitIntrinsicInst(diffType, origGetElementPtr->getOp(), 2, diffOperands);
        }
    }

    return InstPair(primalGetElementPtr, diffGetElementPtr);
}

InstPair ForwardDiffTranscriber::transcribeGetTupleElement(IRBuilder* builder, IRInst* originalInst)
{
    IRInst* origBase = originalInst->getOperand(0);
    auto primalBase = findOrTranscribePrimalInst(builder, origBase);
    auto primalIndex = originalInst->getOperand(1);

    auto primalType = (IRType*)findOrTranscribePrimalInst(builder, originalInst->getDataType());

    IRInst* primalOperands[] = {primalBase, primalIndex};
    IRInst* primalGetElement =
        builder->emitIntrinsicInst(primalType, originalInst->getOp(), 2, primalOperands);

    IRInst* diffGetElement = nullptr;

    if (auto diffType = differentiateType(builder, primalGetElement->getDataType()))
    {
        if (auto diffBase = findOrTranscribeDiffInst(builder, origBase))
        {
            IRInst* diffOperands[] = {diffBase, primalIndex};
            diffGetElement =
                builder->emitIntrinsicInst(diffType, originalInst->getOp(), 2, diffOperands);
        }
    }

    return InstPair(primalGetElement, diffGetElement);
}

InstPair ForwardDiffTranscriber::transcribeUpdateElement(IRBuilder* builder, IRInst* originalInst)
{
    auto updateInst = as<IRUpdateElement>(originalInst);

    IRInst* origBase = updateInst->getOldValue();
    auto primalBase = findOrTranscribePrimalInst(builder, origBase);
    List<IRInst*> primalAccessChain;
    for (UInt i = 0; i < updateInst->getAccessKeyCount(); i++)
    {
        auto originalKey = updateInst->getAccessKey(i);
        auto primalKey = findOrTranscribePrimalInst(builder, originalKey);
        primalAccessChain.add(primalKey);
    }
    auto origVal = updateInst->getElementValue();
    auto primalVal = findOrTranscribePrimalInst(builder, origVal);

    IRInst* primalUpdateField =
        builder->emitUpdateElement(primalBase, primalAccessChain.getArrayView(), primalVal);

    IRInst* diffUpdateElement = nullptr;
    List<IRInst*> diffAccessChain;
    for (auto key : primalAccessChain)
    {
        if (as<IRStructKey>(key))
        {
            auto decor = key->findDecoration<IRDerivativeMemberDecoration>();
            if (decor)
                diffAccessChain.add(decor->getDerivativeMemberStructKey());
            else
            {
                auto diffBase = findOrTranscribeDiffInst(builder, origBase);
                return InstPair(primalUpdateField, diffBase);
            }
        }
        else
        {
            diffAccessChain.add(key);
        }
    }
    if (const auto diffType = differentiateType(builder, originalInst->getDataType()))
    {
        auto diffBase = findOrTranscribeDiffInst(builder, origBase);
        if (!diffBase)
        {
            diffBase = getDifferentialZeroOfType(builder, origBase->getDataType());
        }
        if (auto diffVal = findOrTranscribeDiffInst(builder, origVal))
        {
            auto primalElementType = primalVal->getDataType();

            diffUpdateElement =
                builder->emitUpdateElement(diffBase, diffAccessChain.getArrayView(), diffVal);
            builder->addPrimalElementTypeDecoration(diffUpdateElement, primalElementType);
        }
        else
        {
            auto primalElementType = primalVal->getDataType();
            auto zeroElementDiff = getDifferentialZeroOfType(builder, primalElementType);
            diffUpdateElement = builder->emitUpdateElement(
                diffBase,
                diffAccessChain.getArrayView(),
                zeroElementDiff);
            builder->addPrimalElementTypeDecoration(diffUpdateElement, primalElementType);
        }
    }
    return InstPair(primalUpdateField, diffUpdateElement);
}

InstPair ForwardDiffTranscriber::transcribeSwitch(IRBuilder* builder, IRSwitch* origSwitch)
{
    // Transcribe condition (primal only, conditions do not produce differentials)
    auto primalCondition = findOrTranscribePrimalInst(builder, origSwitch->getCondition());
    SLANG_ASSERT(primalCondition);

    // Transcribe 'default' block
    IRBlock* diffDefaultBlock =
        as<IRBlock>(findOrTranscribeDiffInst(builder, origSwitch->getDefaultLabel()));
    SLANG_ASSERT(diffDefaultBlock);

    // Transcribe 'default' block
    IRBlock* diffBreakBlock =
        as<IRBlock>(findOrTranscribeDiffInst(builder, origSwitch->getBreakLabel()));
    SLANG_ASSERT(diffBreakBlock);

    // Transcribe all other operands
    List<IRInst*> diffCaseValuesAndLabels;
    for (UIndex ii = 0; ii < origSwitch->getCaseCount(); ii++)
    {
        auto primalCaseValue = findOrTranscribePrimalInst(builder, origSwitch->getCaseValue(ii));
        SLANG_ASSERT(primalCaseValue);

        auto diffCaseBlock = findOrTranscribeDiffInst(builder, origSwitch->getCaseLabel(ii));
        SLANG_ASSERT(diffCaseBlock);

        diffCaseValuesAndLabels.add(primalCaseValue);
        diffCaseValuesAndLabels.add(diffCaseBlock);
    }

    auto diffSwitchInst = builder->emitSwitch(
        primalCondition,
        diffBreakBlock,
        diffDefaultBlock,
        diffCaseValuesAndLabels.getCount(),
        diffCaseValuesAndLabels.getBuffer());
    builder->markInstAsMixedDifferential(diffSwitchInst);

    return InstPair(diffSwitchInst, diffSwitchInst);
}

InstPair ForwardDiffTranscriber::transcribeIfElse(IRBuilder* builder, IRIfElse* origIfElse)
{
    // IfElse Statements come with 4 blocks. We transcribe each block into it's
    // linear form, and then wire them up in the same way as the original if-else

    // Transcribe condition block
    auto primalConditionBlock = findOrTranscribePrimalInst(builder, origIfElse->getCondition());
    SLANG_ASSERT(primalConditionBlock);

    // Transcribe 'true' block (condition block branches into this if true)
    auto diffTrueBlock = findOrTranscribeDiffInst(builder, origIfElse->getTrueBlock());
    SLANG_ASSERT(diffTrueBlock);

    // Transcribe 'false' block (condition block branches into this if true)
    auto diffFalseBlock = findOrTranscribeDiffInst(builder, origIfElse->getFalseBlock());
    SLANG_ASSERT(diffFalseBlock);

    // Transcribe 'after' block (true and false blocks branch into this)
    auto diffAfterBlock = findOrTranscribeDiffInst(builder, origIfElse->getAfterBlock());
    SLANG_ASSERT(diffAfterBlock);

    List<IRInst*> diffIfElseArgs;
    diffIfElseArgs.add(primalConditionBlock);
    diffIfElseArgs.add(diffTrueBlock);
    diffIfElseArgs.add(diffFalseBlock);
    diffIfElseArgs.add(diffAfterBlock);

    // If there are any other operands, use their primal versions.
    for (UIndex ii = diffIfElseArgs.getCount(); ii < origIfElse->getOperandCount(); ii++)
    {
        auto primalOperand = findOrTranscribePrimalInst(builder, origIfElse->getOperand(ii));
        diffIfElseArgs.add(primalOperand);
    }

    IRInst* diffIfElse = builder->emitIntrinsicInst(
        nullptr,
        kIROp_ifElse,
        diffIfElseArgs.getCount(),
        diffIfElseArgs.getBuffer());
    builder->markInstAsMixedDifferential(diffIfElse);

    return InstPair(diffIfElse, diffIfElse);
}

InstPair ForwardDiffTranscriber::transcribeMakeDifferentialPair(
    IRBuilder* builder,
    IRMakeDifferentialPairUserCode* origInst)
{
    auto primalVal = findOrTranscribePrimalInst(builder, origInst->getPrimalValue());
    SLANG_ASSERT(primalVal);
    auto diffPrimalVal = findOrTranscribePrimalInst(builder, origInst->getDifferentialValue());
    SLANG_ASSERT(diffPrimalVal);

    auto primalDiffVal = findOrTranscribeDiffInst(builder, origInst->getPrimalValue());
    if (!primalDiffVal)
        primalDiffVal =
            getDifferentialZeroOfType(builder, origInst->getPrimalValue()->getDataType());
    SLANG_ASSERT(primalDiffVal);

    auto diffDiffVal = findOrTranscribeDiffInst(builder, origInst->getDifferentialValue());
    if (!diffDiffVal)
        diffDiffVal =
            getDifferentialZeroOfType(builder, origInst->getDifferentialValue()->getDataType());
    SLANG_ASSERT(diffDiffVal);

    auto primalPairType = findOrTranscribePrimalInst(builder, origInst->getFullType());
    auto diffPairType = findOrTranscribeDiffInst(builder, origInst->getFullType());
    auto primalPair = builder->emitMakeDifferentialPairUserCode(
        (IRType*)primalPairType,
        primalVal,
        diffPrimalVal);
    auto diffPair = builder->emitMakeDifferentialPairUserCode(
        (IRType*)diffPairType,
        primalDiffVal,
        diffDiffVal);
    return InstPair(primalPair, diffPair);
}

InstPair ForwardDiffTranscriber::transcribeDifferentialPairGetElement(
    IRBuilder* builder,
    IRInst* origInst)
{
    SLANG_ASSERT(
        origInst->getOp() == kIROp_DifferentialPairGetDifferentialUserCode ||
        origInst->getOp() == kIROp_DifferentialPairGetPrimalUserCode);

    auto primalVal = findOrTranscribePrimalInst(builder, origInst->getOperand(0));
    SLANG_ASSERT(primalVal);

    auto diffVal = findOrTranscribeDiffInst(builder, origInst->getOperand(0));
    SLANG_ASSERT(diffVal);

    auto primalType = findOrTranscribePrimalInst(builder, origInst->getFullType());

    auto primalResult =
        builder->emitIntrinsicInst((IRType*)primalType, origInst->getOp(), 1, &primalVal);

    auto diffValPairType = as<IRDifferentialPairUserCodeType>(diffVal->getDataType());
    IRInst* diffResultType = nullptr;
    if (origInst->getOp() == kIROp_DifferentialPairGetDifferentialUserCode)
        diffResultType =
            differentiableTypeConformanceContext.getDiffTypeFromPairType(builder, diffValPairType);
    else
        diffResultType = diffValPairType->getValueType();
    auto diffResult =
        builder->emitIntrinsicInst((IRType*)diffResultType, origInst->getOp(), 1, &diffVal);
    return InstPair(primalResult, diffResult);
}

InstPair ForwardDiffTranscriber::transcribeSingleOperandInst(IRBuilder* builder, IRInst* origInst)
{
    IRInst* origBase = origInst->getOperand(0);
    auto primalBase = findOrTranscribePrimalInst(builder, origBase);
    auto primalType = (IRType*)findOrTranscribePrimalInst(builder, origInst->getDataType());

    IRInst* primalResult =
        builder->emitIntrinsicInst(primalType, origInst->getOp(), 1, &primalBase);

    IRInst* diffResult = nullptr;

    if (auto diffType = differentiateType(builder, origInst->getDataType()))
    {
        if (auto diffBase = findOrTranscribeDiffInst(builder, origBase))
        {
            diffResult = builder->emitIntrinsicInst(diffType, origInst->getOp(), 1, &diffBase);
        }
    }
    return InstPair(primalResult, diffResult);
}

InstPair ForwardDiffTranscriber::transcribeMakeExistential(
    IRBuilder* builder,
    IRMakeExistential* origMakeExistential)
{
    auto origBase = origMakeExistential->getWrappedValue();
    auto origWitnessTable = origMakeExistential->getWitnessTable();

    auto primalBase = findOrTranscribePrimalInst(builder, origBase);
    auto primalWitnessTable = findOrTranscribePrimalInst(builder, origWitnessTable);
    auto primalType =
        (IRType*)findOrTranscribePrimalInst(builder, origMakeExistential->getDataType());

    IRInst* primalResult = builder->emitMakeExistential(primalType, primalBase, primalWitnessTable);

    IRInst* diffResult = nullptr;

    auto primalInterfaceType =
        as<IRInterfaceType>(unwrapAttributedType(origMakeExistential->getDataType()));
    SLANG_RELEASE_ASSERT(primalInterfaceType);

    // If the interface type of the existential is differentiable, we emit a make existential
    // of IDifferentiable.Differential type and the witness table of the original type's conformance
    // to IDifferentiable.
    //
    if (auto differentialWitnessTable =
            differentiableTypeConformanceContext.tryExtractConformanceFromInterfaceType(
                builder,
                primalInterfaceType,
                (IRWitnessTable*)primalWitnessTable))
    {
        if (auto diffBase = findOrTranscribeDiffInst(builder, origBase))
        {
            auto differentialAssociatedType = differentiateType(builder, primalInterfaceType);
            SLANG_ASSERT(differentialAssociatedType);

            diffResult = builder->emitMakeExistential(
                differentialAssociatedType,
                diffBase,
                differentialWitnessTable);
        }
    }

    return InstPair(primalResult, diffResult);
}

InstPair ForwardDiffTranscriber::transcribeDefaultConstruct(IRBuilder* builder, IRInst* origInst)
{
    IRInst* primalConstruct = maybeCloneForPrimalInst(builder, origInst);

    IRInst* diffConstruct = nullptr;

    if (auto diffType = differentiateType(builder, origInst->getDataType()))
    {
        diffConstruct = builder->emitDefaultConstructRaw(diffType);
    }
    return InstPair(primalConstruct, diffConstruct);
}

InstPair ForwardDiffTranscriber::transcribeWrapExistential(IRBuilder* builder, IRInst* origInst)
{
    auto primalType = (IRType*)findOrTranscribePrimalInst(builder, origInst->getDataType());

    List<IRInst*> primalArgs;
    for (UInt i = 0; i < origInst->getOperandCount(); i++)
    {
        auto primalArg = findOrTranscribePrimalInst(builder, origInst->getOperand(i));
        primalArgs.add(primalArg);
    }

    IRInst* primalResult = builder->emitIntrinsicInst(
        primalType,
        origInst->getOp(),
        primalArgs.getCount(),
        primalArgs.getBuffer());

    IRInst* diffResult = nullptr;

    if (auto diffType = differentiateType(builder, origInst->getDataType()))
    {
        List<IRInst*> diffArgs;
        for (UInt i = 0; i < origInst->getOperandCount(); i++)
        {
            auto arg = findOrTranscribeDiffInst(builder, origInst->getOperand(i));
            if (arg)
            {
                diffArgs.add(arg);
            }
            else if (i == 0)
            {
                // If we can't diff the first operand (base), abort now.
                break;
            }
        }
        if (diffArgs.getCount())
        {
            diffResult = builder->emitIntrinsicInst(
                diffType,
                origInst->getOp(),
                diffArgs.getCount(),
                diffArgs.getBuffer());
        }
    }
    return InstPair(primalResult, diffResult);
}

// Create an empty func to represent the transcribed func of `origFunc`.
InstPair ForwardDiffTranscriber::transcribeFuncHeader(IRBuilder* inBuilder, IRFunc* origFunc)
{
    if (auto fwdDecor = origFunc->findDecoration<IRForwardDerivativeDecoration>())
    {
        // If we reach here, the function must have been used directly in a `call` inst, and
        // therefore can't be a generic. Generic function are always referenced with `specialize`
        // inst and the handling logic for custom derivatives is implemented in
        // `transcribeSpecialize`.
        SLANG_RELEASE_ASSERT(fwdDecor->getForwardDerivativeFunc()->getOp() == kIROp_Func);
        return InstPair(origFunc, fwdDecor->getForwardDerivativeFunc());
    }

    IRFunc* diffFunc = nullptr;

    // If we're transcribing a function as a 'value' (i.e. maybe embedded in a generic, keep the
    // insert location unchanged). If we're transcribing it as a declaration, we should
    // insert into the module.
    //
    auto origOuterGen = as<IRGeneric>(findOuterGeneric(origFunc));
    if (!origOuterGen || findInnerMostGenericReturnVal(origOuterGen) != origFunc)
    {
        // Dealing with a declaration.. insert into module scope.
        IRBuilder subBuilder = *inBuilder;
        subBuilder.setInsertInto(inBuilder->getModule());
        diffFunc = transcribeFuncHeaderImpl(&subBuilder, origFunc);
    }
    else
    {
        diffFunc = transcribeFuncHeaderImpl(inBuilder, origFunc);
    }

    if (auto outerGen = findOuterGeneric(diffFunc))
    {
        IRBuilder subBuilder = *inBuilder;
        subBuilder.setInsertBefore(origFunc);
        auto specialized =
            specializeWithGeneric(subBuilder, outerGen, as<IRGeneric>(findOuterGeneric(origFunc)));
        subBuilder.addForwardDerivativeDecoration(origFunc, specialized);
    }
    else
    {
        inBuilder->addForwardDerivativeDecoration(origFunc, diffFunc);
    }

    inBuilder->addFloatingModeOverrideDecoration(diffFunc, FloatingPointMode::Fast);

    copyOriginalDecorations(origFunc, diffFunc);

    FuncBodyTranscriptionTask task;
    task.type = FuncBodyTranscriptionTaskType::Forward;
    task.originalFunc = origFunc;
    task.resultFunc = diffFunc;
    autoDiffSharedContext->followUpFunctionsToTranscribe.add(task);

    return InstPair(origFunc, diffFunc);
}

IRFunc* ForwardDiffTranscriber::transcribeFuncHeaderImpl(IRBuilder* inBuilder, IRFunc* origFunc)
{
    IRBuilder builder = *inBuilder;

    maybeMigrateDifferentiableDictionaryFromDerivativeFunc(inBuilder, origFunc);
    differentiableTypeConformanceContext.setFunc(origFunc);

    auto diffFunc = builder.createFunc();

    SLANG_ASSERT(as<IRFuncType>(origFunc->getFullType()));
    IRType* diffFuncType = this->differentiateFunctionType(
        &builder,
        origFunc,
        as<IRFuncType>(origFunc->getFullType()));
    diffFunc->setFullType(diffFuncType);

    if (auto nameHint = origFunc->findDecoration<IRNameHintDecoration>())
    {
        auto originalName = nameHint->getName();
        StringBuilder newNameSb;
        newNameSb << "s_fwd_" << originalName;
        builder.addNameHintDecoration(diffFunc, newNameSb.getUnownedSlice());
    }

    // Mark the generated derivative function itself as differentiable.
    builder.addForwardDifferentiableDecoration(diffFunc);
    if (isBackwardDifferentiableFunc(origFunc))
        builder.addBackwardDifferentiableDecoration(diffFunc);

    // Transfer checkpoint hint decorations
    copyCheckpointHints(&builder, origFunc, diffFunc);
    return diffFunc;
}

void ForwardDiffTranscriber::checkAutodiffInstDecorations(IRFunc* fwdFunc)
{
    for (auto block = fwdFunc->getFirstBlock(); block; block = block->getNextBlock())
    {
        for (auto inst = block->getFirstOrdinaryInst(); inst; inst = inst->getNextInst())
        {
            // TODO: Special case, not sure why these insts show up
            if (as<IRUndefined>(inst))
                continue;

            List<IRDecoration*> decorations;
            for (auto decoration : inst->getDecorations())
            {
                if (as<IRAutodiffInstDecoration>(decoration))
                    decorations.add(decoration);
            }

            // Must have _exactly_ one autodiff tag.
            SLANG_ASSERT(decorations.getCount() == 1);
        }
    }
}

void insertTempVarForMutableParams(IRModule* module, IRFunc* func)
{
    IRBuilder builder(module);
    auto firstBlock = func->getFirstBlock();
    builder.setInsertBefore(firstBlock->getFirstOrdinaryInst());

    OrderedDictionary<IRParam*, IRVar*> mapParamToTempVar;
    List<IRParam*> params;
    for (auto param : firstBlock->getParams())
    {
        if (const auto ptrType = as<IROutTypeBase>(param->getDataType()))
        {
            params.add(param);
        }
    }

    for (auto param : params)
    {
        auto ptrType = asRelevantPtrType(param->getDataType());
        auto tempVar = builder.emitVar(ptrType->getValueType());
        param->replaceUsesWith(tempVar);
        mapParamToTempVar[param] = tempVar;
        if (ptrType->getOp() != kIROp_OutType)
        {
            builder.emitStore(tempVar, builder.emitLoad(param));
        }
        else
        {
            builder.emitStore(tempVar, builder.emitDefaultConstruct(ptrType->getValueType()));
        }
    }

    for (auto block : func->getBlocks())
    {
        for (auto inst : block->getChildren())
        {
            if (inst->getOp() == kIROp_Return)
            {
                builder.setInsertBefore(inst);
                for (const auto& [param, var] : mapParamToTempVar)
                    builder.emitStore(param, builder.emitLoad(var));
            }
        }
    }
}

bool isLocalPointer(IRInst* ptrInst)
{
    // If it's not a local var or a function parameter, then it's probably
    // referencing something outside the function scope.
    //
    auto addr = getRootAddr(ptrInst);
    return as<IRVar>(addr) || as<IRParam, IRDynamicCastBehavior::NoUnwrap>(addr);
}

void lowerSwizzledStores(IRModule* module, IRFunc* func)
{
    List<IRInst*> instsToRemove;

    IRBuilder builder(module);
    for (auto block : func->getBlocks())
    {
        for (auto inst : block->getChildren())
        {
            if (auto swizzledStore = as<IRSwizzledStore>(inst))
            {
                if (!isLocalPointer(swizzledStore->getDest()))
                    continue;

                builder.setInsertBefore(inst);
                for (UIndex ii = 0; ii < swizzledStore->getElementCount(); ii++)
                {
                    auto indexVal = swizzledStore->getElementIndex(ii);
                    auto indexedPtr =
                        builder.emitElementAddress(swizzledStore->getDest(), indexVal);
                    builder.emitStore(
                        indexedPtr,
                        builder.emitElementExtract(
                            swizzledStore->getSource(),
                            builder.getIntValue(builder.getIntType(), ii)));
                }
                instsToRemove.add(inst);
            }
        }
    }

    for (auto inst : instsToRemove)
    {
        inst->removeAndDeallocate();
    }
}

SlangResult ForwardDiffTranscriber::prepareFuncForForwardDiff(IRFunc* func)
{
    insertTempVarForMutableParams(autoDiffSharedContext->moduleInst->getModule(), func);
    removeLinkageDecorations(func);

    performPreAutoDiffForceInlining(func);

    initializeLocalVariables(autoDiffSharedContext->moduleInst->getModule(), func);

    lowerSwizzledStores(autoDiffSharedContext->moduleInst->getModule(), func);

    auto result = eliminateAddressInsts(func, sink);

    if (SLANG_SUCCEEDED(result))
    {
        disableIRValidationAtInsert();
        auto simplifyOptions = IRSimplificationOptions::getDefault(nullptr);
        simplifyOptions.removeRedundancy = true;
        simplifyOptions.hoistLoopInvariantInsts = true;
        simplifyFunc(autoDiffSharedContext->targetProgram, func, simplifyOptions);
        enableIRValidationAtInsert();
    }
    return result;
}

// Transcribe a function definition.
InstPair ForwardDiffTranscriber::transcribeFunc(
    IRBuilder* inBuilder,
    IRFunc* primalFunc,
    IRFunc* diffFunc)
{
    if (primalFunc->findDecoration<IRTreatAsDifferentiableDecoration>())
    {
        // Generate a trivial implementation for [TreatAsDifferentiable] functions.
        generateTrivialFwdDiffFunc(primalFunc, diffFunc);
        return InstPair(primalFunc, diffFunc);
    }

    IRBuilder builder = *inBuilder;
    builder.setInsertBefore(primalFunc);

    // Create a clone for original func and run additional transformations on the clone.
    IRCloneEnv env;
    auto primalFuncClone = as<IRFunc>(cloneInst(&env, &builder, primalFunc));
    prepareFuncForForwardDiff(primalFuncClone);

    builder.setInsertInto(diffFunc);

    differentiableTypeConformanceContext.setFunc(primalFuncClone);

    mapInOutParamToWriteBackValue.clear();

    // Create and map blocks in diff func.
    for (auto block = primalFuncClone->getFirstBlock(); block; block = block->getNextBlock())
    {
        auto diffBlock = builder.emitBlock();
        mapPrimalInst(block, diffBlock);
        mapDifferentialInst(block, diffBlock);
    }

    // Now actually transcribe the content of each block.
    for (auto block = primalFuncClone->getFirstBlock(); block; block = block->getNextBlock())
        this->transcribeBlock(&builder, block);

    for (auto block : diffFunc->getBlocks())
    {
        for (auto inst : block->getChildren())
        {
            if (inst->getOp() == kIROp_Return)
            {
                // Insert write backs to mutable parameters before returning.
                builder.setInsertBefore(inst);
                for (auto& writeBack : mapInOutParamToWriteBackValue)
                {
                    auto param = writeBack.key;
                    auto primalVal = builder.emitLoad(writeBack.value.primal);
                    IRInst* valToStore = nullptr;
                    if (writeBack.value.differential)
                    {
                        auto pairValType =
                            cast<IRPtrTypeBase>(param->getFullType())->getValueType();
                        auto diffVal = builder.emitLoad(writeBack.value.differential);
                        markDiffTypeInst(&builder, diffVal, primalVal->getFullType());

                        valToStore =
                            builder.emitMakeDifferentialPair(pairValType, primalVal, diffVal);

                        markDiffPairTypeInst(&builder, valToStore, pairValType);
                    }
                    else
                    {
                        valToStore = builder.emitLoad(writeBack.value.primal);
                    }

                    auto storeInst = builder.emitStore(param, valToStore);

                    if (writeBack.value.differential)
                    {
                        markDiffPairTypeInst(&builder, storeInst, valToStore->getFullType());
                    }
                }
            }
        }
    }

#if _DEBUG
    checkAutodiffInstDecorations(diffFunc);
#endif

    return InstPair(primalFunc, diffFunc);
}

InstPair ForwardDiffTranscriber::transcribeInstImpl(IRBuilder* builder, IRInst* origInst)
{
    // Handle common SSA-style operations
    switch (origInst->getOp())
    {
    case kIROp_Param:
        return transcribeParam(builder, as<IRParam>(origInst));

    case kIROp_Var:
        return transcribeVar(builder, as<IRVar>(origInst));

    case kIROp_Load:
        return transcribeLoad(builder, as<IRLoad>(origInst));

    case kIROp_Store:
        return transcribeStore(builder, as<IRStore>(origInst));

    case kIROp_Return:
        return transcribeReturn(builder, as<IRReturn>(origInst));

    case kIROp_Add:
    case kIROp_Mul:
    case kIROp_Sub:
    case kIROp_Div:
        return transcribeBinaryArith(builder, origInst);

    case kIROp_Less:
    case kIROp_Greater:
    case kIROp_And:
    case kIROp_Or:
    case kIROp_Geq:
    case kIROp_Leq:
    case kIROp_Eql:
    case kIROp_Neq:
        return transcribeBinaryLogic(builder, origInst);

    case kIROp_Select:
        return transcribeSelect(builder, origInst);

    case kIROp_MakeVector:
    case kIROp_MakeMatrix:
    case kIROp_MakeMatrixFromScalar:
    case kIROp_MatrixReshape:
    case kIROp_IntCast:
    case kIROp_FloatCast:
    case kIROp_MakeVectorFromScalar:
    case kIROp_MakeArray:
    case kIROp_MakeArrayFromElement:
    case kIROp_MakeTuple:
    case kIROp_MakeValuePack:
    case kIROp_BuiltinCast:
        return transcribeConstruct(builder, origInst);
    case kIROp_MakeStruct:
        return transcribeMakeStruct(builder, origInst);

    case kIROp_LookupWitness:
        return transcribeLookupInterfaceMethod(builder, as<IRLookupWitnessMethod>(origInst));

    case kIROp_Call:
        return transcribeCall(builder, as<IRCall>(origInst));

    case kIROp_swizzle:
        return transcribeSwizzle(builder, as<IRSwizzle>(origInst));

    case kIROp_Neg:
        return transcribeByPassthrough(builder, origInst);

    case kIROp_UpdateElement:
        return transcribeUpdateElement(builder, origInst);

    case kIROp_unconditionalBranch:
    case kIROp_loop:
        return transcribeControlFlow(builder, origInst);

    case kIROp_FloatLit:
    case kIROp_IntLit:
    case kIROp_VoidLit:
        return transcribeConst(builder, origInst);

    case kIROp_Specialize:
        return transcribeSpecialize(builder, as<IRSpecialize>(origInst));

    case kIROp_FieldExtract:
    case kIROp_FieldAddress:
        return transcribeFieldExtract(builder, origInst);

    case kIROp_GetElement:
    case kIROp_GetElementPtr:
        return transcribeGetElement(builder, origInst);

    case kIROp_GetTupleElement:
        return transcribeGetTupleElement(builder, origInst);

    case kIROp_ifElse:
        return transcribeIfElse(builder, as<IRIfElse>(origInst));

    case kIROp_Switch:
        return transcribeSwitch(builder, as<IRSwitch>(origInst));

    case kIROp_MakeDifferentialPairUserCode:
        return transcribeMakeDifferentialPair(
            builder,
            as<IRMakeDifferentialPairUserCode>(origInst));

    case kIROp_DifferentialPairGetPrimalUserCode:
    case kIROp_DifferentialPairGetDifferentialUserCode:
        return transcribeDifferentialPairGetElement(builder, origInst);

    case kIROp_ExtractExistentialValue:
        return transcribeSingleOperandInst(builder, origInst);

    case kIROp_PackAnyValue:
        return transcribeSingleOperandInst(builder, origInst);

    case kIROp_MakeExistential:
        return transcribeMakeExistential(builder, as<IRMakeExistential>(origInst));

    case kIROp_ExtractExistentialType:
        {
            IRInst* witnessTable;
            auto diffType = differentiateExtractExistentialType(
                builder,
                as<IRExtractExistentialType>(origInst),
                witnessTable);

            // Mark types as primal since they are not transposable.
            if (diffType)
                builder->markInstAsPrimal(diffType);

            return InstPair(maybeCloneForPrimalInst(builder, origInst), diffType);
        }
    case kIROp_ExtractExistentialWitnessTable:
        return transcribeExtractExistentialWitnessTable(builder, origInst);

    case kIROp_WrapExistential:
        return transcribeWrapExistential(builder, origInst);

    case kIROp_DefaultConstruct:
        return transcribeDefaultConstruct(builder, origInst);

    case kIROp_undefined:
        return transcribeUndefined(builder, origInst);

    case kIROp_Reinterpret:
        return transcribeReinterpret(builder, origInst);

    case kIROp_DifferentiableTypeAnnotation:
        return transcribeDifferentiableTypeAnnotation(builder, origInst);

        // Differentiable insts that should have been lowered in a previous pass.
    case kIROp_SwizzledStore:
        {
            // If we have a non-null dest ptr, then we error out because something went wrong
            // when lowering swizzle-stores to regular stores
            //
            auto swizzledStore = as<IRSwizzledStore>(origInst);
            SLANG_RELEASE_ASSERT(lookupDiffInst(swizzledStore->getDest(), nullptr) == nullptr);
            return transcribeNonDiffInst(builder, swizzledStore);
        }
        // Known non-differentiable insts.
    case kIROp_Not:
    case kIROp_BitAnd:
    case kIROp_BitNot:
    case kIROp_BitXor:
    case kIROp_BitOr:
    case kIROp_BitCast:
    case kIROp_Lsh:
    case kIROp_Rsh:
    case kIROp_IRem:
    case kIROp_ByteAddressBufferLoad:
    case kIROp_ByteAddressBufferStore:
    case kIROp_StructuredBufferLoad:
    case kIROp_RWStructuredBufferLoad:
    case kIROp_RWStructuredBufferLoadStatus:
    case kIROp_RWStructuredBufferStore:
    case kIROp_RWStructuredBufferGetElementPtr:
    case kIROp_NonUniformResourceIndex:
    case kIROp_IsType:
    case kIROp_StaticAssert:
    case kIROp_ImageSubscript:
    case kIROp_ImageLoad:
    case kIROp_ImageStore:
    case kIROp_UnpackAnyValue:
    case kIROp_GetNativePtr:
    case kIROp_CastIntToFloat:
    case kIROp_CastFloatToInt:
    case kIROp_CastIntToEnum:
    case kIROp_CastEnumToInt:
    case kIROp_EnumCast:
    case kIROp_DetachDerivative:
    case kIROp_GetSequentialID:
    case kIROp_GetStringHash:
    case kIROp_SPIRVAsm:
    case kIROp_SPIRVAsmOperandLiteral:
    case kIROp_SPIRVAsmOperandInst:
    case kIROp_SPIRVAsmOperandRayPayloadFromLocation:
    case kIROp_SPIRVAsmOperandRayAttributeFromLocation:
    case kIROp_SPIRVAsmOperandRayCallableFromLocation:
    case kIROp_SPIRVAsmOperandEnum:
    case kIROp_SPIRVAsmOperandBuiltinVar:
    case kIROp_SPIRVAsmOperandGLSL450Set:
    case kIROp_SPIRVAsmOperandDebugPrintfSet:
    case kIROp_SPIRVAsmOperandConvertTexel:
    case kIROp_SPIRVAsmOperandId:
    case kIROp_SPIRVAsmOperandResult:
    case kIROp_SPIRVAsmOperandTruncate:
    case kIROp_SPIRVAsmOperandEntryPoint:
    case kIROp_SPIRVAsmOperandSampledType:
    case kIROp_SPIRVAsmOperandImageType:
    case kIROp_SPIRVAsmOperandSampledImageType:
    case kIROp_DebugLine:
    case kIROp_DebugVar:
    case kIROp_DebugValue:
    case kIROp_GetArrayLength:
    case kIROp_SizeOf:
    case kIROp_AlignOf:
    case kIROp_Printf:
    case kIROp_MakeCoopVector:
    case kIROp_MakeCoopVectorFromValuePack:
    case kIROp_GetCurrentStage:
    case kIROp_GetOffsetPtr:
        return transcribeNonDiffInst(builder, origInst);

        // A call to createDynamicObject<T>(arbitraryData) cannot provide a diff value,
        // so we treat this inst as non differentiable.
        // We can extend the frontend and IR with a separate op-code that can provide an
        // explicit diff value.
        //
        // However, we can't skip this instruction since it also produces a _type_ which may be
        // used by other differentiable instructions. Therefore, we'll create another
        // existential object but with a dzero() for it's value.
        //
    case kIROp_CreateExistentialObject:
        return transcribeNonDiffInst(builder, origInst);

    case kIROp_StructKey:
        return InstPair(origInst, nullptr);

    case kIROp_Unreachable:
        {
            auto unreachInst = builder->emitUnreachable();
            builder->markInstAsMixedDifferential(unreachInst);
            return InstPair(unreachInst, nullptr);
        }

    case kIROp_MakeExistentialWithRTTI:
        SLANG_UNEXPECTED("MakeExistentialWithRTTI inst is not expected in autodiff pass.");
        break;
    }

    return InstPair(nullptr, nullptr);
}

String ForwardDiffTranscriber::makeDiffPairName(IRInst* origVar)
{
    if (auto namehintDecoration = origVar->findDecoration<IRNameHintDecoration>())
    {
        return ("dp" + String(namehintDecoration->getName()));
    }

    return String("");
}

InstPair ForwardDiffTranscriber::transcribeFuncParam(
    IRBuilder* builder,
    IRParam* origParam,
    IRInst* primalType)
{
    SLANG_UNUSED(primalType);

    if (auto diffPairType = tryGetDiffPairType(builder, (IRType*)origParam->getFullType()))
    {
        IRInst* diffPairParam = builder->emitParam(diffPairType);

        auto diffPairVarName = makeDiffPairName(origParam);
        if (diffPairVarName.getLength() > 0)
            builder->addNameHintDecoration(diffPairParam, diffPairVarName.getUnownedSlice());

        SLANG_ASSERT(diffPairParam);

        if (as<IRDifferentialPairType>(diffPairType) || as<IRDifferentialPtrPairType>(diffPairType))
        {
            auto diffType = differentiateType(builder, (IRType*)origParam->getFullType());
            return InstPair(
                builder->emitDifferentialPairGetPrimal(diffPairParam),
                builder->emitDifferentialPairGetDifferential(diffType, diffPairParam));
        }
        else if (auto pairPtrType = asRelevantPtrType(diffPairType))
        {
            auto ptrInnerPairType = as<IRDifferentialPairTypeBase>(pairPtrType->getValueType());
            // Make a local copy of the parameter for primal and diff parts.
            auto primal = builder->emitVar(ptrInnerPairType->getValueType());

            auto diffType = differentiateType(
                builder,
                cast<IRPtrTypeBase>(origParam->getDataType())->getValueType());
            auto diff = builder->emitVar(diffType);
            markDiffTypeInst(builder, diff, builder->getPtrType(ptrInnerPairType->getValueType()));

            IRInst* primalInitVal = nullptr;
            IRInst* diffInitVal = nullptr;
            if (as<IROutType>(diffPairType))
            {
                primalInitVal = builder->emitDefaultConstruct(ptrInnerPairType->getValueType());
                diffInitVal = builder->emitDefaultConstructRaw(diffType);
            }
            else
            {
                auto initVal = builder->emitLoad(diffPairParam);
                markDiffPairTypeInst(builder, initVal, ptrInnerPairType);

                primalInitVal = builder->emitDifferentialPairGetPrimal(initVal);
                diffInitVal = builder->emitDifferentialPairGetDifferential(diffType, initVal);
            }

            markDiffTypeInst(builder, diffInitVal, ptrInnerPairType->getValueType());

            builder->emitStore(primal, primalInitVal);

            auto diffStore = builder->emitStore(diff, diffInitVal);
            markDiffTypeInst(builder, diffStore, ptrInnerPairType->getValueType());

            mapInOutParamToWriteBackValue[diffPairParam] = InstPair(primal, diff);
            return InstPair(primal, diff);
        }
    }

    auto primalInst = cloneInst(&cloneEnv, builder, origParam);
    if (auto primalParam = as<IRParam, IRDynamicCastBehavior::NoUnwrap>(primalInst))
    {
        SLANG_RELEASE_ASSERT(builder->getInsertLoc().getBlock());
        primalParam->removeFromParent();
        builder->getInsertLoc().getBlock()->addParam(primalParam);
    }
    return InstPair(primalInst, nullptr);
}

} // namespace Slang
