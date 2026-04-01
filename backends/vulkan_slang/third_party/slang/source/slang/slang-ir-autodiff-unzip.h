// slang-ir-autodiff-unzip.h
#pragma once

#include "slang-compiler.h"
#include "slang-ir-autodiff-fwd.h"
#include "slang-ir-autodiff-primal-hoist.h"
#include "slang-ir-autodiff-propagate.h"
#include "slang-ir-autodiff-region.h"
#include "slang-ir-autodiff-transcriber-base.h"
#include "slang-ir-autodiff.h"
#include "slang-ir-insts.h"
#include "slang-ir-ssa.h"
#include "slang-ir-validate.h"
#include "slang-ir.h"

namespace Slang
{

struct ParameterBlockTransposeInfo;

struct DiffUnzipPass
{
    AutoDiffSharedContext* autodiffContext;

    IRCloneEnv cloneEnv;

    DifferentiableTypeConformanceContext diffTypeContext;

    // Maps used to keep track of primal and
    // differential versions of split insts.
    //
    Dictionary<IRInst*, IRInst*> primalMap;
    Dictionary<IRInst*, IRInst*> diffMap;
    Dictionary<IRBlock*, IRBlock*> recomputeBlockMap;

    // First diff block.
    // TODO: Can the same pass object can be used for multiple functions?
    // might run into an issue here?
    IRBlock* firstDiffBlock;

    RefPtr<IndexedRegionMap> indexRegionMap;

    DiffUnzipPass(AutoDiffSharedContext* autodiffContext)
        : autodiffContext(autodiffContext), diffTypeContext(autodiffContext)
    {
    }

    IRInst* lookupPrimalInst(IRInst* inst) { return primalMap[inst]; }

    IRInst* lookupDiffInst(IRInst* inst) { return diffMap[inst]; }

    void unzipDiffInsts(IRFunc* func)
    {
        diffTypeContext.setFunc(func);

        // Build a map of blocks to loop regions.
        // This will be used later to insert tracking indices
        //
        indexRegionMap = buildIndexedRegionMap(func);

        IRBuilder builderStorage(autodiffContext->moduleInst->getModule());

        IRBuilder* builder = &builderStorage;

        IRFunc* unzippedFunc = func;

        // Initialize the primal/diff map for parameters.
        // Generate distinct references for parameters that should be split.
        // We don't actually modify the parameter list here, instead we emit
        // PrimalParamRef(param) and DiffParamRef(param) and use those to represent
        // a use from the primal or diff part of the program.
        builder->setInsertBefore(unzippedFunc->getFirstBlock()->getTerminator());

        for (auto primalParam = unzippedFunc->getFirstParam(); primalParam;
             primalParam = primalParam->getNextParam())
        {
            auto type = primalParam->getFullType();
            if (auto ptrType = asRelevantPtrType(type))
            {
                type = ptrType->getValueType();
            }
            if (auto pairType = as<IRDifferentialPairType>(type))
            {
                IRInst* diffType = diffTypeContext.getDiffTypeFromPairType(builder, pairType);
                if (auto ptrType = asRelevantPtrType(primalParam->getFullType()))
                    diffType = builder->getPtrType(ptrType->getOp(), (IRType*)diffType);
                auto primalRef = builder->emitPrimalParamRef(primalParam);
                auto diffRef = builder->emitDiffParamRef((IRType*)diffType, primalParam);
                builder->markInstAsDifferential(diffRef, pairType->getValueType());
                primalMap[primalParam] = primalRef;
                diffMap[primalParam] = diffRef;
            }
        }

        // Functions need to have at least two blocks at this point (one for parameters,
        // and atleast one for code)
        //
        SLANG_ASSERT(unzippedFunc->getFirstBlock() != nullptr);
        SLANG_ASSERT(unzippedFunc->getFirstBlock()->getNextBlock() != nullptr);

        IRBlock* firstBlock =
            as<IRUnconditionalBranch>(unzippedFunc->getFirstBlock()->getTerminator())
                ->getTargetBlock();

        List<IRBlock*> mixedBlocks;
        for (IRBlock* block = firstBlock; block; block = block->getNextBlock())
        {
            // Only need to unzip blocks with both differential and primal instructions.
            if (block->findDecoration<IRMixedDifferentialInstDecoration>())
            {
                mixedBlocks.add(block);
            }
        }

        IRBlock* firstPrimalBlock = nullptr;

        // Emit an empty primal block for every mixed block.
        for (auto block : mixedBlocks)
        {
            IRBlock* primalBlock = builder->emitBlock();
            primalMap[block] = primalBlock;

            if (block == firstBlock)
                firstPrimalBlock = primalBlock;
        }

        // Emit an empty differential block for every mixed block.
        for (auto block : mixedBlocks)
        {
            IRBlock* diffBlock = builder->emitBlock();
            diffMap[block] = diffBlock;

            // Mark the differential block as a differential inst
            // (and add a reference to the primal block)
            builder->markInstAsDifferential(
                diffBlock,
                builder->getBasicBlockType(),
                primalMap[block]);

            // Record the first differential (code) block,
            // since we want all 'return' insts in primal blocks
            // to be replaced with a brahcn into this block.
            //
            if (block == firstBlock)
                this->firstDiffBlock = diffBlock;
        }

        // Split each block into two.
        for (auto block : mixedBlocks)
        {
            splitBlock(block, as<IRBlock>(primalMap[block]), as<IRBlock>(diffMap[block]));
        }

        // Copy regions from fwd-block to their split blocks
        // to make it easier to do lookups.
        //
        {
            List<IRBlock*> workList;
            for (auto [block, _] : indexRegionMap->map)
                workList.add(block);

            for (auto block : workList)
            {
                if (primalMap.containsKey(block))
                    indexRegionMap->map[as<IRBlock>(primalMap[block])] =
                        (IndexedRegion*)indexRegionMap->map[block];

                if (diffMap.containsKey(block))
                    indexRegionMap->map.set(
                        as<IRBlock>(diffMap[block]),
                        (IndexedRegion*)indexRegionMap->map[block]);
            }
        }

        // Swap the first block's occurences out for the first primal block.
        firstBlock->replaceUsesWith(firstPrimalBlock);

        RefPtr<BlockSplitInfo> splitInfo = new BlockSplitInfo();

        for (auto block : mixedBlocks)
            if (primalMap.containsKey(block))
                splitInfo->diffBlockMap[as<IRBlock>(primalMap[block])] =
                    as<IRBlock>(diffMap[block]);

        for (auto block : mixedBlocks)
            block->removeAndDeallocate();
    }

    IRFunc* extractPrimalFunc(
        IRFunc* func,
        IRFunc* originalFunc,
        HoistedPrimalsInfo* primalsInfo,
        ParameterBlockTransposeInfo& paramInfo,
        IRInst*& intermediateType);

    static IRInst* _getOriginalFunc(IRInst* call)
    {
        if (auto decor = call->findDecoration<IRAutoDiffOriginalValueDecoration>())
            return decor->getOriginalValue();
        return nullptr;
    }

    IRInst* getIntermediateType(IRBuilder* builder, IRInst* baseFn)
    {
        if (as<IRLookupWitnessMethod>(baseFn))
        {
            return builder->getVoidType();
        }
        else if (auto specialize = as<IRSpecialize>(baseFn))
        {
            if (as<IRLookupWitnessMethod>(specialize->getBase()))
                return builder->getVoidType();

            auto func = findSpecializeReturnVal(specialize);
            if (as<IRLookupWitnessMethod>(func))
            {
                // An interface method won't have intermediate type.
                return builder->getVoidType();
            }
            else
            {
                auto outerGen = findOuterGeneric(func);
                auto innerIntermediateType =
                    builder->getBackwardDiffIntermediateContextType(outerGen);

                List<IRInst*> args;
                for (UInt i = 0; i < specialize->getArgCount(); i++)
                    args.add(specialize->getArg(i));

                return builder->emitSpecializeInst(
                    builder->getTypeKind(),
                    innerIntermediateType,
                    args.getCount(),
                    args.getBuffer());
            }
        }
        else
        {
            return builder->getBackwardDiffIntermediateContextType(baseFn);
        }
    }

    InstPair splitCall(IRBuilder* primalBuilder, IRBuilder* diffBuilder, IRCall* mixedCall)
    {
        IRBuilder globalBuilder(autodiffContext->moduleInst->getModule());

        auto fwdCalleeType = mixedCall->getCallee()->getDataType();
        auto baseFn = _getOriginalFunc(mixedCall);
        SLANG_RELEASE_ASSERT(baseFn);

        auto primalFuncType =
            autodiffContext->transcriberSet.primalTranscriber->differentiateFunctionType(
                primalBuilder,
                baseFn,
                as<IRFuncType>(baseFn->getDataType()));

        IRInst* intermediateType = getIntermediateType(primalBuilder, baseFn);

        IRVar* intermediateVar = nullptr;
        if (!as<IRVoidType>(intermediateType))
        {
            intermediateVar = primalBuilder->emitVar((IRType*)intermediateType);
            primalBuilder->markInstAsPrimal(intermediateVar);
        }

        IRInst* primalFn = nullptr;
        if (intermediateVar)
        {
            primalBuilder->addBackwardDerivativePrimalContextDecoration(
                intermediateVar,
                intermediateVar);
            primalFn =
                primalBuilder->emitBackwardDifferentiatePrimalInst((IRType*)primalFuncType, baseFn);
        }
        else
        {
            // If we decided not to use diff-primal func that stores an reuse context,
            // we can just call the original function instead.
            primalFn = baseFn;
        }
        List<IRInst*> primalArgs;
        for (UIndex ii = 0; ii < mixedCall->getArgCount(); ii++)
        {
            auto arg = mixedCall->getArg(ii);
            if (isRelevantDifferentialPair(arg->getDataType()))
            {
                primalArgs.add(lookupPrimalInst(arg));
            }
            else
            {
                primalArgs.add(arg);
            }
        }
        if (intermediateType->getOp() != kIROp_VoidType)
            primalArgs.add(intermediateVar);

        auto mixedDecoration = mixedCall->findDecoration<IRMixedDifferentialInstDecoration>();
        SLANG_ASSERT(mixedDecoration);

        IRType* primalType = mixedCall->getFullType();
        IRType* diffType = mixedCall->getFullType();
        IRType* resultType = mixedCall->getFullType();
        if (auto fwdPairResultType = as<IRDifferentialPairType>(mixedDecoration->getPairType()))
        {
            primalType = fwdPairResultType->getValueType();
            diffType =
                (IRType*)diffTypeContext.getDiffTypeFromPairType(&globalBuilder, fwdPairResultType);
            SLANG_ASSERT(diffType);
            resultType = fwdPairResultType;
        }

        auto primalVal = primalBuilder->emitCallInst(primalType, primalFn, primalArgs);
        if (intermediateVar)
            primalBuilder->addBackwardDerivativePrimalContextDecoration(primalVal, intermediateVar);
        primalBuilder->markInstAsPrimal(primalVal);

        auto resolvedPrimalFuncType = as<IRFuncType>(getResolvedInstForDecorations(primalFuncType));
        SLANG_RELEASE_ASSERT(resolvedPrimalFuncType);

        SLANG_RELEASE_ASSERT(mixedCall->getArgCount() <= resolvedPrimalFuncType->getParamCount());

        List<IRInst*> diffArgs;
        for (UIndex ii = 0; ii < mixedCall->getArgCount(); ii++)
        {
            auto arg = mixedCall->getArg(ii);

            // Depending on the type and direction of each argument,
            // we might need to prepare a different value for the transposition logic to produce
            // the correct final argument in the propagate function call.
            if (isRelevantDifferentialPair(arg->getDataType()))
            {
                auto primalArg = lookupPrimalInst(arg);
                auto diffArg = lookupDiffInst(arg);

                // If arg is a mixed differential (pair), it should have already been split.
                SLANG_ASSERT(primalArg);
                SLANG_ASSERT(diffArg);
                auto primalParamType = resolvedPrimalFuncType->getParamType(ii);

                if (const auto outType = as<IROutType>(primalParamType))
                {
                    // For `out` parameters that expects an input derivative to propagate
                    // through, we insert a `LoadReverseGradient` inst here to signify the logic
                    // in `transposeStore` that this argument should actually be the currently
                    // accumulated derivative on this variable. The end purpose is that we will
                    // generate a load(diffArg) in the final transposed code and use that as the
                    // argument for the call, but we can't just emit a normal load inst here
                    // because the transposition logic will turn loads into stores.
                    auto outDiffType = cast<IRPtrTypeBase>(diffArg->getDataType())->getValueType();
                    auto gradArg = diffBuilder->emitLoadReverseGradient(outDiffType, diffArg);
                    diffBuilder->markInstAsDifferential(gradArg, primalArg->getDataType());
                    diffArgs.add(gradArg);
                }
                else if (const auto inoutType = as<IRInOutType>(primalParamType))
                {
                    // Since arg is split into separate vars, we need a new temp var that
                    // represents the remerged diff pair.
                    auto diffPairType = as<IRDifferentialPairType>(
                        as<IRPtrTypeBase>(arg->getDataType())->getValueType());
                    auto primalValueType = diffPairType->getValueType();

                    // We can't simply reuse primalArg for an inout parameter since this will
                    // represent the value after the primal call which can potentially alter
                    // primalArg. Therefore, we will find the first store into primalArg, and
                    // create a temp var holding that value (i.e. value prior to primal call)
                    //
                    auto storeUse = findUniqueStoredVal(cast<IRVar>(primalArg));
                    auto storeInst = cast<IRStore>(storeUse->getUser());

                    auto storedVal = storeInst->getVal();

                    // Emit the temp var into the primal blocks since it's holding a primal
                    // value.
                    auto tempPrimalVar = primalBuilder->emitVar(primalValueType);
                    primalBuilder->emitStore(tempPrimalVar, storedVal);

                    auto diffPairRef = diffBuilder->emitReverseGradientDiffPairRef(
                        arg->getDataType(),
                        tempPrimalVar,
                        diffArg);
                    diffBuilder->markInstAsDifferential(diffPairRef, primalValueType);
                    diffArgs.add(diffPairRef);
                }
                else
                {
                    // For ordinary differentiable input parameters, we make sure to provide
                    // a differential pair. The actual logic that generates an inout variable
                    // will be handled in `transposeCall()`.
                    auto pairArg = diffBuilder->emitMakeDifferentialPair(
                        arg->getDataType(),
                        primalArg,
                        diffArg);

                    diffBuilder->markInstAsDifferential(pairArg, primalArg->getDataType());
                    diffArgs.add(pairArg);
                }
            }
            else
            {
                if (as<IRInOutType>(resolvedPrimalFuncType->getParamType(ii)))
                {
                    // For 'inout' parameter we need to create a temp var to hold the value
                    // before the primal call. This logic is similar to the 'inout' case for
                    // differentiable params only we don't need to deal with pair types.
                    //
                    auto tempPrimalVar = primalBuilder->emitVar(
                        as<IRPtrTypeBase>(arg->getDataType())->getValueType());

                    auto storeUse = findUniqueStoredVal(cast<IRVar>(arg));
                    auto storeInst = cast<IRStore>(storeUse->getUser());
                    auto storedVal = storeInst->getVal();

                    primalBuilder->emitStore(tempPrimalVar, storedVal);

                    diffArgs.add(tempPrimalVar);
                }
                else
                {
                    // For pure 'in' type. Simply re-use the original argument inst.
                    //
                    // For 'out' type parameters, it doesn't really matter what we pass in here,
                    // since the tranposition logic will discard the argument anyway (we'll pass
                    // in the old arg, just to keep the number of arguments consistent)
                    //
                    diffArgs.add(arg);
                }
            }
        }

        auto newFwdCallee = diffBuilder->emitForwardDifferentiateInst(fwdCalleeType, baseFn);

        diffBuilder->markInstAsDifferential(newFwdCallee);

        auto callInst = diffBuilder->emitCallInst(resultType, newFwdCallee, diffArgs);
        diffBuilder->markInstAsDifferential(callInst, primalType);

        if (intermediateVar)
        {
            disableIRValidationAtInsert();
            diffBuilder->addBackwardDerivativePrimalContextDecoration(callInst, intermediateVar);
            enableIRValidationAtInsert();
        }

        IRInst* diffVal = nullptr;
        if (as<IRDifferentialPairType>(callInst->getDataType()))
        {
            diffVal = diffBuilder->emitDifferentialPairGetDifferential(diffType, callInst);
            diffBuilder->markInstAsDifferential(diffVal, primalType);
        }
        return InstPair(primalVal, diffVal);
    }

    InstPair splitMakePair(IRBuilder*, IRBuilder*, IRMakeDifferentialPair* mixedPair)
    {
        return InstPair(mixedPair->getPrimalValue(), mixedPair->getDifferentialValue());
    }

    InstPair splitLoad(IRBuilder* primalBuilder, IRBuilder* diffBuilder, IRLoad* mixedLoad)
    {
        auto primalPtr = lookupPrimalInst(mixedLoad->getPtr());
        auto diffPtr = lookupDiffInst(mixedLoad->getPtr());
        auto primalVal = primalBuilder->emitLoad(primalPtr);
        auto diffVal = diffBuilder->emitLoad(diffPtr);
        diffBuilder->markInstAsDifferential(diffVal, primalVal->getFullType());
        return InstPair(primalVal, diffVal);
    }

    InstPair splitStore(IRBuilder* primalBuilder, IRBuilder* diffBuilder, IRStore* mixedStore)
    {
        auto primalAddr = lookupPrimalInst(mixedStore->getPtr());
        auto diffAddr = lookupDiffInst(mixedStore->getPtr());

        auto primalVal = lookupPrimalInst(mixedStore->getVal());
        auto diffVal = lookupDiffInst(mixedStore->getVal());

        auto primalStore = primalBuilder->emitStore(primalAddr, primalVal);
        auto diffStore = diffBuilder->emitStore(diffAddr, diffVal);

        diffBuilder->markInstAsDifferential(diffStore, primalVal->getFullType());
        return InstPair(primalStore, diffStore);
    }

    InstPair splitVar(IRBuilder* primalBuilder, IRBuilder* diffBuilder, IRVar* mixedVar)
    {
        auto pairType =
            as<IRDifferentialPairType>(as<IRPtrTypeBase>(mixedVar->getDataType())->getValueType());
        auto primalType = pairType->getValueType();
        auto diffType = (IRType*)diffTypeContext.getDifferentialForType(primalBuilder, primalType);
        auto primalVar = primalBuilder->emitVar(primalType);
        auto diffVar = diffBuilder->emitVar(diffType);
        diffBuilder->markInstAsDifferential(diffVar, diffBuilder->getPtrType(primalType));
        return InstPair(primalVar, diffVar);
    }

    InstPair splitReturn(IRBuilder* primalBuilder, IRBuilder* diffBuilder, IRReturn* mixedReturn)
    {
        auto pairType = as<IRDifferentialPairType>(mixedReturn->getVal()->getDataType());
        // Are we returning a differentiable value?
        if (pairType)
        {
            auto primalType = pairType->getValueType();

            // Check that we have an unambiguous 'first' differential block.
            SLANG_ASSERT(firstDiffBlock);

            auto primalBranch = primalBuilder->emitBranch(firstDiffBlock);
            primalBuilder->addBackwardDerivativePrimalReturnDecoration(
                primalBranch,
                lookupPrimalInst(mixedReturn->getVal()));

            auto pairVal = diffBuilder->emitMakeDifferentialPair(
                pairType,
                lookupPrimalInst(mixedReturn->getVal()),
                lookupDiffInst(mixedReturn->getVal()));
            diffBuilder->markInstAsDifferential(pairVal, primalType);

            auto returnInst = diffBuilder->emitReturn(pairVal);
            diffBuilder->markInstAsDifferential(returnInst, primalType);

            return InstPair(primalBranch, returnInst);
        }
        else
        {
            // If return value is not differentiable, just turn it into a trivial branch.
            auto primalBranch = primalBuilder->emitBranch(firstDiffBlock);
            primalBuilder->addBackwardDerivativePrimalReturnDecoration(
                primalBranch,
                mixedReturn->getVal());

            auto returnInst = diffBuilder->emitReturn();
            diffBuilder->markInstAsDifferential(returnInst, nullptr);
            return InstPair(primalBranch, returnInst);
        }
    }

    // Splitting a loop is one of the trickiest parts of the unzip pass.
    // Thus far, we've been dealing with blocks that are only run once, so we
    // could arbitrarily move intermediate instructions to other blocks since they are
    // generated and consumed at-most one time.
    //
    // Intermediate instructions in a loop can take on a different value each iteration
    // and thus need to be stored explicitly to an array.
    //
    // We also need to ascertain an upper limit on the iteration count.
    // With very few exceptions, this is a fundamental requirement.
    //
    InstPair splitLoop(IRBuilder* primalBuilder, IRBuilder* diffBuilder, IRLoop* mixedLoop)
    {

        auto breakBlock = mixedLoop->getBreakBlock();
        auto continueBlock = mixedLoop->getContinueBlock();
        auto nextBlock = mixedLoop->getTargetBlock();

        // Split args.
        List<IRInst*> primalArgs;
        List<IRInst*> diffArgs;
        for (UIndex ii = 0; ii < mixedLoop->getArgCount(); ii++)
        {
            if (isDifferentialInst(mixedLoop->getArg(ii)))
                diffArgs.add(mixedLoop->getArg(ii));
            else
                primalArgs.add(mixedLoop->getArg(ii));
        }

        auto primalLoop = primalBuilder->emitLoop(
            as<IRBlock>(primalMap[nextBlock]),
            as<IRBlock>(primalMap[breakBlock]),
            as<IRBlock>(primalMap[continueBlock]),
            primalArgs.getCount(),
            primalArgs.getBuffer());

        auto diffLoop = diffBuilder->emitLoop(
            as<IRBlock>(diffMap[nextBlock]),
            as<IRBlock>(diffMap[breakBlock]),
            as<IRBlock>(diffMap[continueBlock]),
            diffArgs.getCount(),
            diffArgs.getBuffer());

        if (auto maxItersDecoration = mixedLoop->findDecoration<IRLoopMaxItersDecoration>())
        {
            primalBuilder->addLoopMaxItersDecoration(primalLoop, maxItersDecoration->getMaxIters());
            diffBuilder->addLoopMaxItersDecoration(diffLoop, maxItersDecoration->getMaxIters());
        }

        return InstPair(primalLoop, diffLoop);
    }

    InstPair splitControlFlow(IRBuilder* primalBuilder, IRBuilder* diffBuilder, IRInst* branchInst)
    {
        switch (branchInst->getOp())
        {
        case kIROp_unconditionalBranch:
            {
                auto uncondBranchInst = as<IRUnconditionalBranch>(branchInst);
                auto targetBlock = uncondBranchInst->getTargetBlock();

                // Split args.
                List<IRInst*> primalArgs;
                List<IRInst*> diffArgs;
                for (UIndex ii = 0; ii < uncondBranchInst->getArgCount(); ii++)
                {
                    if (isDifferentialInst(uncondBranchInst->getArg(ii)))
                        diffArgs.add(uncondBranchInst->getArg(ii));
                    else
                        primalArgs.add(uncondBranchInst->getArg(ii));
                }

                return InstPair(
                    primalBuilder->emitBranch(
                        as<IRBlock>(primalMap[targetBlock]),
                        primalArgs.getCount(),
                        primalArgs.getBuffer()),
                    diffBuilder->emitBranch(
                        as<IRBlock>(diffMap[targetBlock]),
                        diffArgs.getCount(),
                        diffArgs.getBuffer()));
            }

        case kIROp_conditionalBranch:
            {
                auto trueBlock = as<IRConditionalBranch>(branchInst)->getTrueBlock();
                auto falseBlock = as<IRConditionalBranch>(branchInst)->getFalseBlock();
                auto condInst = as<IRConditionalBranch>(branchInst)->getCondition();

                return InstPair(
                    primalBuilder->emitBranch(
                        condInst,
                        as<IRBlock>(primalMap[trueBlock]),
                        as<IRBlock>(primalMap[falseBlock])),
                    diffBuilder->emitBranch(
                        condInst,
                        as<IRBlock>(diffMap[trueBlock]),
                        as<IRBlock>(diffMap[falseBlock])));
            }

        case kIROp_ifElse:
            {
                auto trueBlock = as<IRIfElse>(branchInst)->getTrueBlock();
                auto falseBlock = as<IRIfElse>(branchInst)->getFalseBlock();
                auto afterBlock = as<IRIfElse>(branchInst)->getAfterBlock();
                auto condInst = as<IRIfElse>(branchInst)->getCondition();

                return InstPair(
                    primalBuilder->emitIfElse(
                        condInst,
                        as<IRBlock>(primalMap[trueBlock]),
                        as<IRBlock>(primalMap[falseBlock]),
                        as<IRBlock>(primalMap[afterBlock])),
                    diffBuilder->emitIfElse(
                        condInst,
                        as<IRBlock>(diffMap[trueBlock]),
                        as<IRBlock>(diffMap[falseBlock]),
                        as<IRBlock>(diffMap[afterBlock])));
            }

        case kIROp_Switch:
            {
                auto switchInst = as<IRSwitch>(branchInst);
                auto breakBlock = switchInst->getBreakLabel();
                auto defaultBlock = switchInst->getDefaultLabel();
                auto condInst = switchInst->getCondition();

                List<IRInst*> primalCaseArgs;
                List<IRInst*> diffCaseArgs;

                for (UIndex ii = 0; ii < switchInst->getCaseCount(); ii++)
                {
                    primalCaseArgs.add(switchInst->getCaseValue(ii));
                    diffCaseArgs.add(switchInst->getCaseValue(ii));

                    primalCaseArgs.add(primalMap[switchInst->getCaseLabel(ii)]);
                    diffCaseArgs.add(diffMap[switchInst->getCaseLabel(ii)]);
                }

                return InstPair(
                    primalBuilder->emitSwitch(
                        condInst,
                        as<IRBlock>(primalMap[breakBlock]),
                        as<IRBlock>(primalMap[defaultBlock]),
                        primalCaseArgs.getCount(),
                        primalCaseArgs.getBuffer()),
                    diffBuilder->emitSwitch(
                        condInst,
                        as<IRBlock>(diffMap[breakBlock]),
                        as<IRBlock>(diffMap[defaultBlock]),
                        diffCaseArgs.getCount(),
                        diffCaseArgs.getBuffer()));
            }

        case kIROp_loop:
            return splitLoop(primalBuilder, diffBuilder, as<IRLoop>(branchInst));

        default:
            SLANG_UNEXPECTED("Unhandled instruction");
        }
    }

    InstPair _splitMixedInst(IRBuilder* primalBuilder, IRBuilder* diffBuilder, IRInst* inst)
    {
        switch (inst->getOp())
        {
        case kIROp_Call:
            return splitCall(primalBuilder, diffBuilder, as<IRCall>(inst));

        case kIROp_Var:
            return splitVar(primalBuilder, diffBuilder, as<IRVar>(inst));

        case kIROp_MakeDifferentialPair:
            return splitMakePair(primalBuilder, diffBuilder, as<IRMakeDifferentialPair>(inst));

        case kIROp_Load:
            return splitLoad(primalBuilder, diffBuilder, as<IRLoad>(inst));

        case kIROp_Store:
            return splitStore(primalBuilder, diffBuilder, as<IRStore>(inst));

        case kIROp_Return:
            return splitReturn(primalBuilder, diffBuilder, as<IRReturn>(inst));

        case kIROp_unconditionalBranch:
        case kIROp_conditionalBranch:
        case kIROp_ifElse:
        case kIROp_Switch:
        case kIROp_loop:
            return splitControlFlow(primalBuilder, diffBuilder, inst);

        case kIROp_Unreachable:
            return InstPair(primalBuilder->emitUnreachable(), diffBuilder->emitUnreachable());

        default:
            SLANG_ASSERT_FAILURE("Unhandled mixed diff inst");
        }
    }

    void splitMixedInst(IRBuilder* primalBuilder, IRBuilder* diffBuilder, IRInst* inst)
    {
        IRBuilderSourceLocRAII primalLocationScope(primalBuilder, inst->sourceLoc);
        IRBuilderSourceLocRAII diffLocationScope(diffBuilder, inst->sourceLoc);

        auto instPair = _splitMixedInst(primalBuilder, diffBuilder, inst);

        primalMap[inst] = instPair.primal;
        diffMap[inst] = instPair.differential;
    }

    void splitBlock(IRBlock* block, IRBlock* primalBlock, IRBlock* diffBlock)
    {
        // Make two builders for primal and differential blocks.
        IRBuilder primalBuilder(autodiffContext->moduleInst->getModule());
        primalBuilder.setInsertInto(primalBlock);

        IRBuilder diffBuilder(autodiffContext->moduleInst->getModule());
        diffBuilder.setInsertInto(diffBlock);

        List<IRInst*> splitInsts;
        for (auto child : block->getModifiableChildren())
        {
            if (auto getDiffInst = as<IRDifferentialPairGetDifferential>(child))
            {
                // Replace GetDiff(A) with A.d
                if (diffMap.containsKey(getDiffInst->getBase()))
                {
                    getDiffInst->replaceUsesWith(lookupDiffInst(getDiffInst->getBase()));
                    getDiffInst->removeAndDeallocate();
                    continue;
                }
            }
            else if (auto getPrimalInst = as<IRDifferentialPairGetPrimal>(child))
            {
                // Replace GetPrimal(A) with A.p
                if (primalMap.containsKey(getPrimalInst->getBase()))
                {
                    getPrimalInst->replaceUsesWith(lookupPrimalInst(getPrimalInst->getBase()));
                    getPrimalInst->removeAndDeallocate();
                    continue;
                }
            }

            if (isDifferentialInst(child))
            {
                child->insertAtEnd(diffBlock);
            }
            else if (isMixedDifferentialInst(child))
            {
                splitMixedInst(&primalBuilder, &diffBuilder, child);
                splitInsts.add(child);
            }
            else
            {
                child->insertAtEnd(primalBlock);
            }
        }

        // Remove insts that were split.
        for (auto inst : splitInsts)
        {
            if (!isDifferentiableType(diffTypeContext, inst->getDataType()))
            {
                inst->replaceUsesWith(lookupPrimalInst(inst));
            }

            // Consistency check.
            for (auto use = inst->firstUse; use; use = use->nextUse)
            {
                SLANG_RELEASE_ASSERT(
                    (use->getUser()->getParent() != primalBlock) &&
                    (use->getUser()->getParent() != diffBlock));
            }

            // Leave terminator in to keep CFG info.
            if (!as<IRTerminatorInst>(inst))
                inst->removeAndDeallocate();
        }

        // Nothing should be left in the original block.
        SLANG_ASSERT(block->getFirstChild() == block->getTerminator());
    }
};

} // namespace Slang
