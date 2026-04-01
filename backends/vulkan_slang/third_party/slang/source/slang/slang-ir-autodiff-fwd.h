// slang-ir-autodiff-fwd.h
#pragma once

#include "slang-ir-autodiff-transcriber-base.h"

namespace Slang
{

struct ForwardDiffTranscriber : AutoDiffTranscriberBase
{
    // Pending values to write back to inout params at the end of the current function.
    OrderedDictionary<IRInst*, InstPair> mapInOutParamToWriteBackValue;

    ForwardDiffTranscriber(AutoDiffSharedContext* shared, DiagnosticSink* inSink)
        : AutoDiffTranscriberBase(shared, inSink)
    {
    }


    // Returns "d<var-name>" to use as a name hint for variables and parameters.
    // If no primal name is available, returns a blank string.
    //
    String getJVPVarName(IRInst* origVar);

    // Returns "dp<var-name>" to use as a name hint for parameters.
    // If no primal name is available, returns a blank string.
    //
    String makeDiffPairName(IRInst* origVar);

    InstPair transcribeUndefined(IRBuilder* builder, IRInst* origInst);

    InstPair transcribeVar(IRBuilder* builder, IRVar* origVar);

    InstPair transcribeBinaryArith(IRBuilder* builder, IRInst* origArith);

    InstPair transcribeBinaryLogic(IRBuilder* builder, IRInst* origLogic);

    InstPair transcribeSelect(IRBuilder* builder, IRInst* origSelect);

    InstPair transcribeLoad(IRBuilder* builder, IRLoad* origLoad);

    InstPair transcribeStore(IRBuilder* builder, IRStore* origStore);

    // Since int/float literals are sometimes nested inside an IRConstructor
    // instruction, we check to make sure that the nested instr is a constant
    // and then return nullptr. Literals do not need to be differentiated.
    //
    InstPair transcribeConstruct(IRBuilder* builder, IRInst* origConstruct);
    InstPair transcribeMakeStruct(IRBuilder* builder, IRInst* origMakeStruct);

    InstPair transcribeMakeTuple(IRBuilder* builder, IRInst* origMakeTuple);

    // Differentiating a call instruction here is primarily about generating
    // an appropriate call list based on whichever parameters have differentials
    // in the current transcription context.
    //
    InstPair transcribeCall(IRBuilder* builder, IRCall* origCall);

    InstPair transcribeSwizzle(IRBuilder* builder, IRSwizzle* origSwizzle);

    InstPair transcribeByPassthrough(IRBuilder* builder, IRInst* origInst);

    InstPair transcribeControlFlow(IRBuilder* builder, IRInst* origInst);

    InstPair transcribeConst(IRBuilder* builder, IRInst* origInst);

    InstPair transcribeSpecialize(IRBuilder* builder, IRSpecialize* origSpecialize);

    InstPair transcribeFieldExtract(IRBuilder* builder, IRInst* originalInst);

    InstPair transcribeGetElement(IRBuilder* builder, IRInst* origGetElementPtr);

    InstPair transcribeGetTupleElement(IRBuilder* builder, IRInst* origInst);

    InstPair transcribeUpdateElement(IRBuilder* builder, IRInst* originalInst);

    InstPair transcribeIfElse(IRBuilder* builder, IRIfElse* origIfElse);

    InstPair transcribeSwitch(IRBuilder* builder, IRSwitch* origSwitch);

    InstPair transcribeMakeDifferentialPair(
        IRBuilder* builder,
        IRMakeDifferentialPairUserCode* origInst);

    InstPair transcribeMakeExistential(IRBuilder* builder, IRMakeExistential* origMakeExistential);

    InstPair transcribeDifferentialPairGetElement(IRBuilder* builder, IRInst* origInst);

    InstPair transcribeSingleOperandInst(IRBuilder* builder, IRInst* origInst);

    InstPair transcribeWrapExistential(IRBuilder* builder, IRInst* origInst);

    InstPair transcribeDefaultConstruct(IRBuilder* builder, IRInst* origInst);

    InstPair transcribeReinterpret(IRBuilder* builder, IRInst* origInst);

    InstPair transcribeDifferentiableTypeAnnotation(IRBuilder* builder, IRInst* origInst);

    virtual IRFuncType* differentiateFunctionType(
        IRBuilder* builder,
        IRInst* func,
        IRFuncType* funcType) override;

    void generateTrivialFwdDiffFunc(IRFunc* primalFunc, IRFunc* diffFunc);

    // Transcribe a function definition.
    InstPair transcribeFunc(IRBuilder* inBuilder, IRFunc* primalFunc, IRFunc* diffFunc);

    // Transcribe a function without marking the result as a decoration.
    IRFunc* transcribeFuncHeaderImpl(IRBuilder* inBuilder, IRFunc* origFunc);

    void checkAutodiffInstDecorations(IRFunc* fwdFunc);

    SlangResult prepareFuncForForwardDiff(IRFunc* func);

    // Create an empty func to represent the transcribed func of `origFunc`.
    virtual InstPair transcribeFuncHeader(IRBuilder* inBuilder, IRFunc* origFunc) override;

    virtual InstPair transcribeInstImpl(IRBuilder* builder, IRInst* origInst) override;

    virtual InstPair transcribeFuncParam(IRBuilder* builder, IRParam* origParam, IRInst* primalType)
        override;

    virtual IROp getInterfaceRequirementDerivativeDecorationOp() override
    {
        return kIROp_ForwardDerivativeDecoration;
    }
};

} // namespace Slang
