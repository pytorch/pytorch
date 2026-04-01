// slang-ir-autodiff-transcriber-base.h
#pragma once

#include "slang-compiler.h"
#include "slang-ir-autodiff.h"
#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{

struct AutoDiffTranscriberBase
{
    // Stores the mapping of arbitrary 'R-value' instructions to instructions that represent
    // their differential values.
    Dictionary<IRInst*, IRInst*> instMapD;

    // Set of insts currently being transcribed. Used to avoid infinite loops.
    HashSet<IRInst*> instsInProgress;

    // Cloning environment to hold mapping from old to new copies for the primal
    // instructions.
    IRCloneEnv cloneEnv;

    // Diagnostic sink for error messages.
    DiagnosticSink* sink;

    // Type conformance information.
    AutoDiffSharedContext* autoDiffSharedContext;

    // Builder to help with creating and accessing the 'DifferentiablePair<T>' struct
    DifferentialPairTypeBuilder* pairBuilder;

    DifferentiableTypeConformanceContext differentiableTypeConformanceContext;

    AutoDiffTranscriberBase(AutoDiffSharedContext* shared, DiagnosticSink* inSink)
        : autoDiffSharedContext(shared), differentiableTypeConformanceContext(shared), sink(inSink)
    {
        cloneEnv.squashChildrenMapping = true;
    }

    DiagnosticSink* getSink();

    // Returns "dp<var-name>" to use as a name hint for parameters.
    // If no primal name is available, returns a blank string.
    //
    String makeDiffPairName(IRInst* origVar);

    void mapDifferentialInst(IRInst* origInst, IRInst* diffInst);

    void mapPrimalInst(IRInst* origInst, IRInst* primalInst);

    IRInst* lookupDiffInst(IRInst* origInst);

    IRInst* lookupDiffInst(IRInst* origInst, IRInst* defaultInst);

    bool hasDifferentialInst(IRInst* origInst);

    bool shouldUseOriginalAsPrimal(IRInst* currentParent, IRInst* origInst);

    IRInst* lookupPrimalInstImpl(IRInst* currentParent, IRInst* origInst);

    IRInst* lookupPrimalInst(IRInst* currentParent, IRInst* origInst, IRInst* defaultInst);

    IRInst* lookupPrimalInstIfExists(IRBuilder* builder, IRInst* origInst)
    {
        return lookupPrimalInst(builder->getInsertLoc().getParent(), origInst, origInst);
    }

    IRInst* lookupPrimalInst(IRBuilder* builder, IRInst* origInst)
    {
        return lookupPrimalInstImpl(builder->getInsertLoc().getParent(), origInst);
    }

    IRInst* lookupPrimalInst(IRBuilder* builder, IRInst* origInst, IRInst* defaultInst)
    {
        return lookupPrimalInst(builder->getInsertLoc().getParent(), origInst, defaultInst);
    }

    bool hasPrimalInst(IRInst* currentParent, IRInst* origInst);

    IRInst* findOrTranscribeDiffInst(IRBuilder* builder, IRInst* origInst);

    IRInst* findOrTranscribePrimalInst(IRBuilder* builder, IRInst* origInst);

    IRInst* maybeCloneForPrimalInst(IRBuilder* builder, IRInst* inst);

    InstPair transcribeExtractExistentialWitnessTable(IRBuilder* builder, IRInst* origInst);

    void maybeMigrateDifferentiableDictionaryFromDerivativeFunc(
        IRBuilder* builder,
        IRInst* origFunc);

    IRInst* tryGetDifferentiableWitness(
        IRBuilder* builder,
        IRInst* originalType,
        DiffConformanceKind kind);

    IRType* getOrCreateDiffPairType(IRBuilder* builder, IRInst* primalType, IRInst* witness);

    IRType* getOrCreateDiffPairType(IRBuilder* builder, IRInst* originalType);

    IRType* differentiateType(IRBuilder* builder, IRType* origType);

    IRType* differentiateExtractExistentialType(
        IRBuilder* builder,
        IRExtractExistentialType* origType,
        IRInst*& witnessTable);

    IRType* tryGetDiffPairType(IRBuilder* builder, IRType* primalType);

    IRInst* getDifferentialZeroOfType(IRBuilder* builder, IRType* primalType);

    InstPair transcribeNonDiffInst(IRBuilder* builder, IRInst* origInst);

    InstPair transcribeReturn(IRBuilder* builder, IRReturn* origReturn);

    InstPair transcribeParam(IRBuilder* builder, IRParam* origParam);

    virtual InstPair transcribeFuncParam(
        IRBuilder* builder,
        IRParam* origParam,
        IRInst* primalType) = 0;

    InstPair transcribeLookupInterfaceMethod(IRBuilder* builder, IRLookupWitnessMethod* lookupInst);

    InstPair transcribeBlockImpl(
        IRBuilder* builder,
        IRBlock* origBlock,
        HashSet<IRInst*>& instsToSkip);

    InstPair transcribeBlock(IRBuilder* builder, IRBlock* origBlock)
    {
        HashSet<IRInst*> ignore;
        for (auto inst = origBlock->getFirstInst(); inst; inst = inst->next)
        {
            if (inst->m_op == kIROp_Unmodified)
                ignore.add(inst);
        }

        return transcribeBlockImpl(builder, origBlock, ignore);
    }

    // Transcribe a generic definition
    InstPair transcribeGeneric(IRBuilder* inBuilder, IRGeneric* origGeneric);

    IRInst* transcribe(IRBuilder* builder, IRInst* origInst);

    InstPair transcribeInst(IRBuilder* builder, IRInst* origInst);

    IRType* _differentiateTypeImpl(IRBuilder* builder, IRType* origType);

    bool isExistentialType(IRType* type);

    void _markInstAsDifferential(
        IRBuilder* builder,
        IRInst* diffInst,
        IRInst* primalInst = nullptr);

    void copyOriginalDecorations(IRInst* origFunc, IRInst* diffFunc);

    virtual IRFuncType* differentiateFunctionType(
        IRBuilder* builder,
        IRInst* func,
        IRFuncType* funcType) = 0;

    // Create an empty func to represent the transcribed func of `origFunc`.
    virtual InstPair transcribeFuncHeader(IRBuilder* inBuilder, IRFunc* origFunc) = 0;

    virtual InstPair transcribeInstImpl(IRBuilder* builder, IRInst* origInst) = 0;

    virtual IROp getInterfaceRequirementDerivativeDecorationOp() = 0;

    void markDiffTypeInst(IRBuilder* builder, IRInst* inst, IRType* primalType);

    void markDiffPairTypeInst(IRBuilder* builder, IRInst* inst, IRType* primalType);
};

} // namespace Slang
