// slang-ir-autodiff-fwd.h
#pragma once

#include "slang-compiler.h"
#include "slang-ir-clone.h"
#include "slang-ir-dce.h"
#include "slang-ir-eliminate-phis.h"
#include "slang-ir-inst-pass-base.h"
#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{
template<typename P, typename D>
struct DiffInstPair
{
    P primal;
    D differential;
    DiffInstPair() = default;
    DiffInstPair(P primal, D differential)
        : primal(primal), differential(differential)
    {
    }
    HashCode getHashCode() const
    {
        Hasher hasher;
        hasher << primal << differential;
        return hasher.getResult();
    }
    bool operator==(const DiffInstPair& other) const
    {
        return primal == other.primal && differential == other.differential;
    }
};

typedef DiffInstPair<IRInst*, IRInst*> InstPair;

enum class FuncBodyTranscriptionTaskType
{
    Forward,
    BackwardPrimal,
    BackwardPropagate,
    Backward
};

struct FuncBodyTranscriptionTask
{
    FuncBodyTranscriptionTaskType type;
    IRFunc* originalFunc;
    IRFunc* resultFunc;
};

struct AutoDiffTranscriberBase;

struct DiffTranscriberSet
{
    AutoDiffTranscriberBase* forwardTranscriber = nullptr;
    AutoDiffTranscriberBase* primalTranscriber = nullptr;
    AutoDiffTranscriberBase* propagateTranscriber = nullptr;
    AutoDiffTranscriberBase* backwardTranscriber = nullptr;
};


enum class DiffConformanceKind
{
    Any = 0,  // Perform actions for any conformance (infer from context)
    Ptr = 1,  // Perform actions for IDifferentiablePtrType
    Value = 2 // Perform actions for IDifferentiable
};

struct AutoDiffSharedContext
{
    TargetProgram* targetProgram = nullptr;

    IRModuleInst* moduleInst = nullptr;

    // A reference to the builtin IDifferentiable interface type.
    // We use this to look up all the other types (and type exprs)
    // that conform to a base type.
    //
    IRInterfaceType* differentiableInterfaceType = nullptr;

    // The struct key for the 'Differential' associated type
    // defined inside IDifferential. We use this to lookup the differential
    // type in the conformance table associated with the concrete type.
    //
    IRStructKey* differentialAssocTypeStructKey = nullptr;

    // The struct key for the witness that `Differential` associated type conforms to
    // `IDifferential`.
    IRStructKey* differentialAssocTypeWitnessStructKey = nullptr;
    IRWitnessTableType* differentialAssocTypeWitnessTableType = nullptr;


    // The struct key for the 'zero()' associated type
    // defined inside IDifferential. We use this to lookup the
    // implementation of zero() for a given type.
    //
    IRStructKey* zeroMethodStructKey = nullptr;
    IRFuncType* zeroMethodType = nullptr;

    // The struct key for the 'add()' associated type
    // defined inside IDifferential. We use this to lookup the
    // implementation of add() for a given type.
    //
    IRStructKey* addMethodStructKey = nullptr;
    IRFuncType* addMethodType = nullptr;

    IRStructKey* mulMethodStructKey = nullptr;

    // Refernce to NullDifferential struct type. These are used
    // as sentinel values for uninitialized existential (interface-typed)
    // differentials.
    //
    IRStructType* nullDifferentialStructType = nullptr;

    // Reference to the NullDifferential : IDifferentiable witness.
    //
    IRInst* nullDifferentialWitness = nullptr;


    // A reference to the builtin IDifferentiablePtrType interface type.
    IRInterfaceType* differentiablePtrInterfaceType = nullptr;

    // The struct key for the 'Differential' associated type
    // defined inside IDifferentialPtrType. We use this to lookup the differential
    // type in the conformance table associated with the concrete type.
    //
    IRStructKey* differentialAssocRefTypeStructKey = nullptr;

    // The struct key for the witness that `Differential` associated type conforms to
    // `IDifferentialPtrType`.
    IRStructKey* differentialAssocRefTypeWitnessStructKey = nullptr;
    IRWitnessTableType* differentialAssocRefTypeWitnessTableType = nullptr;

    // Modules that don't use differentiable types
    // won't have the IDifferentiable interface type available.
    // Set to false to indicate that we are uninitialized.
    //
    bool isInterfaceAvailable = false;
    bool isPtrInterfaceAvailable = false;

    List<FuncBodyTranscriptionTask> followUpFunctionsToTranscribe;

    DiffTranscriberSet transcriberSet;

    AutoDiffSharedContext(TargetProgram* target, IRModuleInst* inModuleInst);

private:
    IRInst* findDifferentiableInterface();

    IRStructType* findNullDifferentialStructType();

    IRInst* findNullDifferentialWitness();

    IRStructKey* findDifferentialTypeStructKey()
    {
        return cast<IRStructKey>(
            getInterfaceEntryAtIndex(differentiableInterfaceType, 0)->getRequirementKey());
    }

    IRStructKey* findDifferentialTypeWitnessStructKey()
    {
        return cast<IRStructKey>(
            getInterfaceEntryAtIndex(differentiableInterfaceType, 1)->getRequirementKey());
    }

    IRWitnessTableType* findDifferentialTypeWitnessTableType()
    {
        return cast<IRWitnessTableType>(
            getInterfaceEntryAtIndex(differentiableInterfaceType, 1)->getRequirementVal());
    }

    IRStructKey* findZeroMethodStructKey()
    {
        return cast<IRStructKey>(
            getInterfaceEntryAtIndex(differentiableInterfaceType, 2)->getRequirementKey());
    }

    IRStructKey* findAddMethodStructKey()
    {
        return cast<IRStructKey>(
            getInterfaceEntryAtIndex(differentiableInterfaceType, 3)->getRequirementKey());
    }

    IRStructKey* findMulMethodStructKey()
    {
        return cast<IRStructKey>(
            getInterfaceEntryAtIndex(differentiableInterfaceType, 4)->getRequirementKey());
    }


    IRStructKey* findDifferentialPtrTypeStructKey()
    {
        return cast<IRStructKey>(
            getInterfaceEntryAtIndex(differentiablePtrInterfaceType, 0)->getRequirementKey());
    }

    IRStructKey* findDifferentialPtrTypeWitnessStructKey()
    {
        return cast<IRStructKey>(
            getInterfaceEntryAtIndex(differentiablePtrInterfaceType, 1)->getRequirementKey());
    }

    IRWitnessTableType* findDifferentialPtrTypeWitnessTableType()
    {
        return cast<IRWitnessTableType>(
            getInterfaceEntryAtIndex(differentiablePtrInterfaceType, 1)->getRequirementVal());
    }

    // IRStructKey* getIDifferentiableStructKeyAtIndex(UInt index);
    IRInterfaceRequirementEntry* getInterfaceEntryAtIndex(IRInterfaceType* interface, UInt index);
};


struct DifferentiableTypeConformanceContext
{
    AutoDiffSharedContext* sharedContext;

    IRGlobalValueWithCode* parentFunc = nullptr;
    OrderedDictionary<IRType*, IRInst*> differentiableTypeWitnessDictionary;

    Dictionary<IRInst*, List<IRDifferentiableTypeAnnotation*>> annotationCache;

    IRFunc* existentialDAddFunc = nullptr;

    DifferentiableTypeConformanceContext(AutoDiffSharedContext* shared)
        : sharedContext(shared)
    {
        // Populate dictionary with null differential type.
        if (sharedContext->nullDifferentialStructType)
            differentiableTypeWitnessDictionary.add(
                sharedContext->nullDifferentialStructType,
                sharedContext->nullDifferentialWitness);
    }

    void setFunc(IRGlobalValueWithCode* func);

    List<IRDifferentiableTypeAnnotation*> getAnnotations(IRGlobalValueWithCode* inst);

    List<IRDifferentiableTypeAnnotation*> getAnnotations(IRModuleInst* inst);

    void buildGlobalWitnessDictionary();

    // Lookup a witness table for the concreteType. One should exist if concreteType
    // inherits (successfully) from IDifferentiable.
    //
    IRInst* lookUpConformanceForType(IRInst* type, DiffConformanceKind kind);

    IRInst* lookUpInterfaceMethod(
        IRBuilder* builder,
        IRType* origType,
        IRStructKey* key,
        IRType* resultType = nullptr,
        DiffConformanceKind kind = DiffConformanceKind::Any);

    IRType* differentiateType(IRBuilder* builder, IRInst* primalType);

    IRInst* tryGetDifferentiableWitness(
        IRBuilder* builder,
        IRInst* originalType,
        DiffConformanceKind kind);

    IRType* getOrCreateDiffPairType(IRBuilder* builder, IRInst* primalType, IRInst* witness);

    IRInst* getDifferentialTypeFromDiffPairType(
        IRBuilder* builder,
        IRDifferentialPairTypeBase* diffPairType);

    IRInst* getDiffTypeFromPairType(IRBuilder* builder, IRDifferentialPairTypeBase* type);

    IRInst* getDiffTypeWitnessFromPairType(IRBuilder* builder, IRDifferentialPairTypeBase* type);

    IRInst* getDiffZeroMethodFromPairType(IRBuilder* builder, IRDifferentialPairTypeBase* type);

    IRInst* getDiffAddMethodFromPairType(IRBuilder* builder, IRDifferentialPairTypeBase* type);

    void addTypeToDictionary(IRType* type, IRInst* witness);

    IRInterfaceType* getConformanceTypeFromWitness(IRInst* witness);

    IRInst* tryExtractConformanceFromInterfaceType(
        IRBuilder* builder,
        IRInterfaceType* interfaceType,
        IRWitnessTable* witnessTable);

    List<IRInterfaceRequirementEntry*> findInterfaceLookupPath(
        IRInterfaceType* supType,
        IRInterfaceType* type);

    // Lookup and return the 'Differential' type declared in the concrete type
    // in order to conform to the IDifferentiable/IDifferentiablePtrType interfaces
    // Note that inside a generic block, this will be a witness table lookup instruction
    // that gets resolved during the specialization pass.
    //
    IRInst* getDifferentialForType(IRBuilder* builder, IRType* origType)
    {
        switch (origType->getOp())
        {
        case kIROp_InterfaceType:
            {
                if (isDifferentiableValueType(origType))
                    return this->sharedContext->differentiableInterfaceType;
                else if (isDifferentiablePtrType(origType))
                    return this->sharedContext->differentiablePtrInterfaceType;
                else
                    return nullptr;
            }
        case kIROp_ArrayType:
            {
                auto diffElementType = (IRType*)getDifferentialForType(
                    builder,
                    as<IRArrayType>(origType)->getElementType());
                if (!diffElementType)
                    return nullptr;
                return builder->getArrayType(
                    diffElementType,
                    as<IRArrayType>(origType)->getElementCount());
            }
        case kIROp_TupleType:
        case kIROp_TypePack:
            {
                return differentiateType(builder, origType);
            }
        case kIROp_DifferentialPairUserCodeType:
            {
                auto diffPairType = as<IRDifferentialPairTypeBase>(origType);
                auto diffType = getDiffTypeFromPairType(builder, diffPairType);
                auto diffWitness = getDiffTypeWitnessFromPairType(builder, diffPairType);
                return builder->getDifferentialPairUserCodeType((IRType*)diffType, diffWitness);
            }
        case kIROp_DifferentialPtrPairType:
            {
                auto diffPairType = as<IRDifferentialPairTypeBase>(origType);
                auto diffType = getDiffTypeFromPairType(builder, diffPairType);
                auto diffWitness = getDiffTypeWitnessFromPairType(builder, diffPairType);
                return builder->getDifferentialPtrPairType((IRType*)diffType, diffWitness);
            }
        default:
            if (isDifferentiableValueType(origType))
                return lookUpInterfaceMethod(
                    builder,
                    origType,
                    sharedContext->differentialAssocTypeStructKey,
                    builder->getTypeKind());
            else if (isDifferentiablePtrType(origType))
                return lookUpInterfaceMethod(
                    builder,
                    origType,
                    sharedContext->differentialAssocRefTypeStructKey,
                    builder->getTypeKind());
            else
                return nullptr;
        }
    }

    bool isDifferentiableType(IRType* origType)
    {
        return isDifferentiableValueType(origType) || isDifferentiablePtrType(origType);
    }

    bool isDifferentiableValueType(IRType* origType)
    {
        for (; origType;)
        {
            switch (origType->getOp())
            {
            case kIROp_FloatType:
            case kIROp_HalfType:
            case kIROp_DoubleType:
            case kIROp_DifferentialPairType:
            case kIROp_DifferentialPairUserCodeType:
                return true;
            case kIROp_VectorType:
            case kIROp_ArrayType:
            case kIROp_PtrType:
            case kIROp_OutType:
            case kIROp_InOutType:
                origType = (IRType*)origType->getOperand(0);
                continue;
            default:
                return lookUpConformanceForType(origType, DiffConformanceKind::Value) != nullptr;
            }
        }
        return false;
    }

    bool isDifferentiablePtrType(IRType* origType)
    {
        for (; origType;)
        {
            switch (origType->getOp())
            {
            case kIROp_VectorType:
            case kIROp_ArrayType:
            case kIROp_PtrType:
            case kIROp_OutType:
            case kIROp_InOutType:
                origType = (IRType*)origType->getOperand(0);
                continue;
            default:
                return lookUpConformanceForType(origType, DiffConformanceKind::Ptr) != nullptr;
            }
        }
        return false;
    }

    IRInst* getZeroMethodForType(IRBuilder* builder, IRType* origType)
    {
        auto result = lookUpInterfaceMethod(
            builder,
            origType,
            sharedContext->zeroMethodStructKey,
            sharedContext->zeroMethodType,
            DiffConformanceKind::Value);
        return result;
    }

    IRInst* getAddMethodForType(IRBuilder* builder, IRType* origType)
    {
        auto result = lookUpInterfaceMethod(
            builder,
            origType,
            sharedContext->addMethodStructKey,
            sharedContext->addMethodType,
            DiffConformanceKind::Value);
        return result;
    }

    IRInst* emitNullDifferential(IRBuilder* builder)
    {
        return builder->emitCallInst(
            sharedContext->nullDifferentialStructType,
            getZeroMethodForType(builder, sharedContext->nullDifferentialStructType),
            List<IRInst*>());
    }

    IRFunc* getOrCreateExistentialDAddMethod();

    IRInst* buildDifferentiablePairWitness(
        IRBuilder* builder,
        IRDifferentialPairTypeBase* pairType,
        DiffConformanceKind target);

    IRInst* buildArrayWitness(
        IRBuilder* builder,
        IRArrayType* pairType,
        DiffConformanceKind target);

    IRInst* buildTupleWitness(IRBuilder* builder, IRInst* tupleType, DiffConformanceKind target);

    IRInst* buildExtractExistensialTypeWitness(
        IRBuilder* builder,
        IRExtractExistentialType* extractExistentialType,
        DiffConformanceKind target);

    IRInst* emitDAddOfDiffInstType(
        IRBuilder* builder,
        IRType* primalType,
        IRInst* op1,
        IRInst* op2);

    IRInst* emitDAddForExistentialType(
        IRBuilder* builder,
        IRType* primalType,
        IRInst* op1,
        IRInst* op2);

    IRInst* emitDZeroOfDiffInstType(IRBuilder* builder, IRType* primalType);
};


struct DifferentialPairTypeBuilder
{
    DifferentialPairTypeBuilder() = default;

    DifferentialPairTypeBuilder(AutoDiffSharedContext* sharedContext)
        : sharedContext(sharedContext)
    {
    }

    IRInst* findSpecializationForParam(IRInst* specializeInst, IRInst* genericParam);

    IRInst* emitFieldAccessor(IRBuilder* builder, IRInst* baseInst, IRStructKey* key);

    IRInst* emitPrimalFieldAccess(IRBuilder* builder, IRType* loweredPairType, IRInst* baseInst);

    IRInst* emitDiffFieldAccess(IRBuilder* builder, IRType* loweredPairType, IRInst* baseInst);

    IRInst* emitExistentialMakePair(
        IRBuilder* builder,
        IRInst* type,
        IRInst* primalInst,
        IRInst* diffInst);

    IRStructKey* _getOrCreateDiffStructKey();

    IRStructKey* _getOrCreatePrimalStructKey();

    IRInst* _createDiffPairType(IRType* origBaseType, IRType* diffType);

    IRInst* _createDiffPairInterfaceRequirement(IRType* origBaseType, IRType* diffType);

    IRInst* lowerDiffPairType(IRBuilder* builder, IRType* originalPairType);

    IRInst* getOrCreateCommonDiffPairInterface(IRBuilder* builder);

    struct PairStructKey
    {
        IRInst* originalType;
        IRInst* diffType;
    };

    // Cache from pair types to lowered type.
    Dictionary<IRInst*, IRInst*> pairTypeCache;

    // Cache from existential pair types to their lowered interface keys.
    // We use a different cache because an interface type can have
    // a regular pair for the pair of interface types, as well as an
    // interface key for the associated pair types used for its implementations
    //
    Dictionary<IRInst*, IRInst*> existentialPairTypeCache;

    // Cache for any interface requirement keys (generated for existential
    // pair types)
    //
    Dictionary<IRInst*, IRStructKey*> assocPairTypeKeyMap;
    Dictionary<IRInst*, IRStructKey*> makePairKeyMap;
    Dictionary<IRInst*, IRStructKey*> getPrimalKeyMap;
    Dictionary<IRInst*, IRStructKey*> getDiffKeyMap;

    // More caches for easier lookups of the types associated with the
    // keys. (avoid having to keep recomputing or performing complicated
    // lookups)
    //
    Dictionary<IRInst*, IRFuncType*> makePairFuncTypeMap;
    Dictionary<IRInst*, IRFuncType*> getPrimalFuncTypeMap;
    Dictionary<IRInst*, IRFuncType*> getDiffFuncTypeMap;

    // Even more caches for easier access to original primal/diff types
    // (Only used for existential pair types). For regular pair types,
    // these are easy to find right on the type itself.
    //
    Dictionary<IRInst*, IRType*> primalTypeMap;
    Dictionary<IRInst*, IRType*> diffTypeMap;


    IRStructKey* globalPrimalKey = nullptr;

    IRStructKey* globalDiffKey = nullptr;

    IRInst* genericDiffPairType = nullptr;

    List<IRInst*> generatedTypeList;

    AutoDiffSharedContext* sharedContext = nullptr;

    IRInterfaceType* commonDiffPairInterface = nullptr;
};

void stripAutoDiffDecorations(IRModule* module);
void stripTempDecorations(IRInst* inst);

bool isNoDiffType(IRType* paramType);
bool isNeverDiffFuncType(IRFuncType* funcType);

IRInst* lookupForwardDerivativeReference(IRInst* primalFunction);

struct IRAutodiffPassOptions
{
    // Nothing for now...
};

void checkAutodiffPatterns(TargetProgram* target, IRModule* module, DiagnosticSink* sink);

bool processAutodiffCalls(
    TargetProgram* target,
    IRModule* module,
    DiagnosticSink* sink,
    IRAutodiffPassOptions const& options = IRAutodiffPassOptions());

bool finalizeAutoDiffPass(TargetProgram* target, IRModule* module);

// Utility methods

void copyCheckpointHints(
    IRBuilder* builder,
    IRGlobalValueWithCode* oldInst,
    IRGlobalValueWithCode* newInst);

void cloneCheckpointHint(
    IRBuilder* builder,
    IRCheckpointHintDecoration* oldInst,
    IRGlobalValueWithCode* code);

void stripDerivativeDecorations(IRInst* inst);

bool isBackwardDifferentiableFunc(IRInst* func);

bool isDifferentiableType(DifferentiableTypeConformanceContext& context, IRInst* typeInst);

bool canTypeBeStored(IRInst* type);

inline bool isRelevantDifferentialPair(IRType* type)
{
    if (as<IRDifferentialPairType>(type))
    {
        return true;
    }
    else if (auto argPtrType = asRelevantPtrType(type))
    {
        if (as<IRDifferentialPairType>(argPtrType->getValueType()))
        {
            return true;
        }
    }
    return false;
}

bool isRuntimeType(IRType* type);

IRInst* getExistentialBaseWitnessTable(IRBuilder* builder, IRType* type);

UIndex addPhiOutputArg(
    IRBuilder* builder,
    IRBlock* block,
    IRInst*& inoutTerminatorInst,
    IRInst* arg);

IRUse* findUniqueStoredVal(IRVar* var);
IRUse* findLatestUniqueWriteUse(IRVar* var);
IRUse* findEarliestUniqueWriteUse(IRVar* var);

bool isDerivativeContextVar(IRVar* var);

bool isDiffInst(IRInst* inst);

bool isDifferentialOrRecomputeBlock(IRBlock* block);

}; // namespace Slang
