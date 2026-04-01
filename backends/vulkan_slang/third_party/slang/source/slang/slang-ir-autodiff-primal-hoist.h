// slang-ir-autodiff-primal-hoist.h
#pragma once

#include "slang-ir-autodiff-region.h"
#include "slang-ir-autodiff.h"
#include "slang-ir-dominators.h"
#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{
struct IROutOfOrderCloneContext : public RefObject
{
    IRCloneEnv cloneEnv;
    HashSet<IRUse*> pendingUses;

    void registerClonedInst(IRBuilder* builder, IRInst* inst, IRInst* clonedInst)
    {
        UInt operandCount = clonedInst->getOperandCount();
        for (UInt ii = 0; ii < operandCount; ++ii)
        {
            auto newOperand = clonedInst->getOperand(ii);
            // If operand is in a differential or recompute block, it means it has already
            // been cloned, so we don't add it to pending uses.
            if (auto operandParent = as<IRBlock>(newOperand->getParent()))
            {
                if (isDifferentialOrRecomputeBlock(operandParent))
                {
                    continue;
                }
            }
            // Otherwise, add it to pending uses.
            pendingUses.add(&clonedInst->getOperands()[ii]);
        }

        for (auto use = inst->firstUse; use;)
        {
            auto nextUse = use->nextUse;

            if (pendingUses.contains(use))
            {
                pendingUses.remove(use);
                builder->replaceOperand(use, clonedInst);
            }

            use = nextUse;
        }
    }

    IRInst* cloneInstOutOfOrder(IRBuilder* builder, IRInst* inst)
    {
        IRInst* clonedInst = cloneInst(&cloneEnv, builder, inst);
        registerClonedInst(builder, inst, clonedInst);
        return clonedInst;
    }
};

struct InversionInfo
{
    IRInst* instToInvert;
    List<IRInst*> requiredOperands;
    List<IRInst*> targetInsts;

    InversionInfo(IRInst* instToInvert, List<IRInst*> requiredOperands, List<IRInst*> targetInsts)
        : instToInvert(instToInvert), requiredOperands(requiredOperands), targetInsts(targetInsts)
    {
    }

    InversionInfo()
        : instToInvert(nullptr)
    {
    }

    InversionInfo applyMap(IRCloneEnv* env)
    {
        InversionInfo newInfo;
        if (env->mapOldValToNew.containsKey(instToInvert))
            newInfo.instToInvert = env->mapOldValToNew[instToInvert];

        for (auto inst : requiredOperands)
            if (env->mapOldValToNew.containsKey(inst))
                newInfo.requiredOperands.add(env->mapOldValToNew[inst]);

        for (auto inst : targetInsts)
            if (env->mapOldValToNew.containsKey(inst))
                newInfo.targetInsts.add(env->mapOldValToNew[inst]);

        return newInfo;
    }
};

struct HoistedPrimalsInfo : public RefObject
{
    OrderedHashSet<IRInst*> storeSet;
    OrderedHashSet<IRInst*> recomputeSet;
    OrderedHashSet<IRInst*> invertSet;
    OrderedHashSet<IRInst*> instsToInvert;

    Dictionary<IRInst*, InversionInfo> invertInfoMap;

    RefPtr<HoistedPrimalsInfo> applyMap(IRCloneEnv* env)
    {
        RefPtr<HoistedPrimalsInfo> newPrimalsInfo = new HoistedPrimalsInfo();

        const auto goSet = [&env](const auto& inSet, auto& outSet)
        {
            for (auto inst : inSet)
                if (const auto newKey = env->mapOldValToNew.tryGetValue(inst))
                    outSet.add(*newKey);
        };

        goSet(this->storeSet, newPrimalsInfo->storeSet);
        goSet(this->recomputeSet, newPrimalsInfo->recomputeSet);
        goSet(this->invertSet, newPrimalsInfo->invertSet);
        goSet(this->instsToInvert, newPrimalsInfo->instsToInvert);

        for (auto [key, value] : this->invertInfoMap)
            if (const auto newKey = env->mapOldValToNew.tryGetValue(key))
                newPrimalsInfo->invertInfoMap.set(*newKey, value.applyMap(env));

        return newPrimalsInfo;
    }

    void merge(HoistedPrimalsInfo* info)
    {
        for (auto inst : info->storeSet)
            storeSet.add(inst);

        for (auto inst : info->recomputeSet)
            recomputeSet.add(inst);

        for (auto inst : info->invertSet)
            invertSet.add(inst);

        for (auto inst : info->instsToInvert)
            instsToInvert.add(inst);

        for (auto invertInfo : info->invertInfoMap)
            invertInfoMap.add(invertInfo);
    }
};

struct HoistResult
{
    enum Mode
    {
        Store,
        Recompute,
        Invert,

        None
    };

    Mode mode;

    IRInst* instToStore = nullptr;
    IRInst* instToRecompute = nullptr;
    InversionInfo inversionInfo;

    HoistResult(Mode mode, IRInst* target)
        : mode(mode)
    {
        switch (mode)
        {
        case Mode::Store:
            instToStore = target;
            break;
        case Mode::Recompute:
            instToRecompute = target;
            break;
        case Mode::Invert:
            SLANG_UNEXPECTED("Wrong constructor for HoistResult::Mode::Invert");
            break;
        case Mode::None:
            instToStore = nullptr;
            instToRecompute = nullptr;
            break;
        default:
            SLANG_UNEXPECTED("Unhandled hoist mode");
            break;
        }
    }

    HoistResult(InversionInfo info)
        : mode(Mode::Invert), inversionInfo(info)
    {
    }

    static HoistResult store(IRInst* inst) { return HoistResult(Mode::Store, inst); }

    static HoistResult recompute(IRInst* inst) { return HoistResult(Mode::Recompute, inst); }

    static HoistResult invert(InversionInfo inst) { return HoistResult(inst); }

    static HoistResult none() { return HoistResult(Mode::None, nullptr); }
};

struct IndexTrackingInfo : public RefObject
{
    // After lowering, store references to the count
    // variables associated with this region
    //
    IRInst* primalCountParam = nullptr;
    IRInst* diffCountParam = nullptr;

    // Reference to the header block. Note that the header block
    // typically contains the loop condition and is executed N+1
    // times if the loop body is executed N times.
    //
    IRBlock* loopHeaderBlock = nullptr;

    enum CountStatus
    {
        Unresolved,
        Dynamic,
        Static
    };

    CountStatus status = CountStatus::Unresolved;

    // Inferred maximum number of iterations.
    Count maxIters = -1;

    bool operator==(const IndexTrackingInfo& other) const
    {
        return primalCountParam == other.primalCountParam;
    }
};

struct LoopInductionValueInfo
{
    enum Kind
    {
        AlwaysTrue,
        AffineFunctionOfCounter,
    };
    Kind kind;
    IRLoop* loopInst = nullptr;
    IRInst* counterOffset = nullptr;
    IRIntegerValue counterFactor = 1;
};

// Information on which insts are to be stored, recomputed
// and inverted within a single function.
// This data structure also holds a map of raw HoistResult
// objects to provide more information to later passes.
//
struct CheckpointSetInfo : public RefObject
{
    HashSet<IRInst*> storeSet;
    HashSet<IRInst*> recomputeSet;
    HashSet<IRInst*> invertSet;
    Dictionary<IRInst*, LoopInductionValueInfo> loopInductionInfo;
    Dictionary<IRInst*, InversionInfo> invInfoMap;
    Dictionary<IRInst*, IRInst*> loopExitValueInsts;
};

struct UseOrPseudoUse
{
    IRUse* irUse = nullptr;
    IRInst* user;
    IRInst* usedVal;
    UseOrPseudoUse() = default;
    UseOrPseudoUse(IRUse* use)
    {
        user = use->getUser();
        usedVal = use->get();
        irUse = use;
    }
    UseOrPseudoUse(IRInst* inUser, IRInst* inUsedVal)
    {
        irUse = nullptr;
        user = inUser;
        usedVal = inUsedVal;
        ;
    }
    HashCode getHashCode() const
    {
        return combineHash(Slang::getHashCode(user), Slang::getHashCode(usedVal));
    }
    bool operator==(const UseOrPseudoUse& other) const
    {
        return user == other.user && usedVal == other.usedVal;
    }
};

// Information on a block after it has been split in the unzip step.
// After unzipping, every block in the original function will have
// two corresponding blocks in the new function:
// - A 'primal-recompute' block, which contains the original instructions
//   from the original block, but located in the corresponding the reverse
//   diff region so their results are accessible in the diff block for
//   derivative computation.
// - A 'diff' block, which contains the transcribed instructions from the
//   original block.
struct BlockSplitInfo : public RefObject
{
    // Maps primal to differential blocks from the unzip step.
    Dictionary<IRBlock*, IRBlock*> diffBlockMap;
};

class AutodiffCheckpointPolicyBase : public RefObject
{
public:
    AutodiffCheckpointPolicyBase(IRModule* module)
        : module(module)
    {
    }

    RefPtr<HoistedPrimalsInfo> processFunc(
        IRGlobalValueWithCode* func,
        Dictionary<IRBlock*, IRBlock*>& mapDiffBlockToRecomputeBlock,
        IROutOfOrderCloneContext* cloneCtx,
        Dictionary<IRBlock*, List<IndexTrackingInfo>>& blockIndexInfo);

    // Do pre-processing on the function (mainly for
    // 'global' checkpointing methods that consider the entire
    // function)
    //
    virtual void preparePolicy(IRGlobalValueWithCode* func) = 0;

    virtual HoistResult classify(UseOrPseudoUse diffBlockUse) = 0;

protected:
    IRModule* module;
    Dictionary<IRInst*, LoopInductionValueInfo> inductionValueInsts;
    Dictionary<IRInst*, IRInst*> loopExitValueInsts;
    void collectInductionValues(IRGlobalValueWithCode* func);
    void collectLoopExitConditions(IRGlobalValueWithCode* func);
};

class DefaultCheckpointPolicy : public AutodiffCheckpointPolicyBase
{
public:
    DefaultCheckpointPolicy(IRModule* module)
        : AutodiffCheckpointPolicyBase(module)
    {
    }

    virtual void preparePolicy(IRGlobalValueWithCode* func);
    virtual HoistResult classify(UseOrPseudoUse use);

private:
    bool canRecompute(UseOrPseudoUse use);
};

RefPtr<HoistedPrimalsInfo> applyCheckpointPolicy(IRGlobalValueWithCode* func);
}; // namespace Slang
