#include "slang-ir-check-differentiability.h"

#include "slang-ir-autodiff.h"
#include "slang-ir-inst-pass-base.h"

namespace Slang
{

struct CheckDifferentiabilityPassContext : public InstPassBase
{
public:
    DiagnosticSink* sink;
    AutoDiffSharedContext sharedContext;

    enum DifferentiableLevel
    {
        Forward,
        Backward
    };
    Dictionary<IRInst*, DifferentiableLevel> differentiableFunctions;

    CheckDifferentiabilityPassContext(IRModule* inModule, DiagnosticSink* inSink)
        : InstPassBase(inModule), sink(inSink), sharedContext(nullptr, inModule->getModuleInst())
    {
    }

    bool _isFuncMarkedForAutoDiff(IRInst* func)
    {
        func = getResolvedInstForDecorations(func);
        if (!func)
            return false;
        for (auto decorations : func->getDecorations())
        {
            switch (decorations->getOp())
            {
            case kIROp_ForwardDifferentiableDecoration:
            case kIROp_BackwardDifferentiableDecoration:
                return true;
            }
        }
        return false;
    }

    bool _isDifferentiableFuncImpl(IRInst* func, DifferentiableLevel level)
    {
        func = getResolvedInstForDecorations(func);
        if (!func)
            return false;
        if (auto substDecor = func->findDecoration<IRPrimalSubstituteDecoration>())
        {
            func = getResolvedInstForDecorations(substDecor->getPrimalSubstituteFunc());
            if (!func)
                return false;
        }

        for (auto decorations : func->getDecorations())
        {
            switch (decorations->getOp())
            {
            case kIROp_ForwardDerivativeDecoration:
            case kIROp_ForwardDifferentiableDecoration:
                if (level == DifferentiableLevel::Forward)
                    return true;
                break;
            case kIROp_UserDefinedBackwardDerivativeDecoration:
            case kIROp_BackwardDerivativeDecoration:
            case kIROp_BackwardDifferentiableDecoration:
                return true;
            default:
                break;
            }
        }
        return false;
    }

    bool shouldTreatCallAsDifferentiable(IRInst* callInst)
    {
        SLANG_ASSERT(as<IRCall>(callInst));

        return (
            callInst->findDecoration<IRTreatCallAsDifferentiableDecoration>() ||
            callInst->findDecoration<IRDifferentiableCallDecoration>());
    }

    bool isDifferentiableFunc(IRInst* func, DifferentiableLevel level)
    {
        switch (func->getOp())
        {
        case kIROp_ForwardDifferentiate:
            if (auto fwdDerivative =
                    func->getOperand(0)->findDecoration<IRForwardDerivativeDecoration>())
                return isDifferentiableFunc(fwdDerivative->getForwardDerivativeFunc(), level);
            return isDifferentiableFunc(func->getOperand(0), level);
        case kIROp_BackwardDifferentiate:
            if (auto bwdDerivative =
                    func->getOperand(0)
                        ->findDecoration<IRUserDefinedBackwardDerivativeDecoration>())
                return isDifferentiableFunc(bwdDerivative->getBackwardDerivativeFunc(), level);
            return isDifferentiableFunc(func->getOperand(0), level);
        default:
            break;
        }

        func = getResolvedInstForDecorations(func);
        if (!func)
            return false;

        if (auto substDecor = func->findDecoration<IRPrimalSubstituteDecoration>())
        {
            func = getResolvedInstForDecorations(substDecor->getPrimalSubstituteFunc());
            if (!func)
                return false;
        }

        if (auto existingLevel = differentiableFunctions.tryGetValue(func))
            return *existingLevel >= level;

        if (func->findDecoration<IRTreatAsDifferentiableDecoration>())
            return true;

        if (auto lookupInterfaceMethod = as<IRLookupWitnessMethod>(func))
        {
            auto wit = lookupInterfaceMethod->getWitnessTable();
            if (!wit)
                return false;
            auto witType = as<IRWitnessTableTypeBase>(wit->getDataType());
            if (!witType)
                return false;
            auto interfaceType = witType->getConformanceType();
            if (!interfaceType)
                return false;
            if (interfaceType->findDecoration<IRTreatAsDifferentiableDecoration>())
                return true;
            if (sharedContext.differentiableInterfaceType &&
                interfaceType == sharedContext.differentiableInterfaceType)
                return true;
            if (lookupInterfaceMethod->getRequirementKey()
                    ->findDecoration<IRBackwardDerivativeDecoration>())
                return true;
            if (lookupInterfaceMethod->getRequirementKey()
                    ->findDecoration<IRForwardDerivativeDecoration>())
                return level == DifferentiableLevel::Forward;
        }

        for (; func; func = func->parent)
        {
            if (as<IRGeneric>(func))
            {
                if (auto existingLevel = differentiableFunctions.tryGetValue(func))
                {
                    if (*existingLevel >= level)
                        return true;
                }
            }
        }
        return false;
    }

    bool isInstInFunc(IRInst* inst, IRInst* func)
    {
        while (inst)
        {
            if (inst == func)
                return true;
            inst = inst->parent;
        }
        return false;
    }

    bool canAddressHoldDerivative(
        DifferentiableTypeConformanceContext& diffTypeContext,
        IRInst* addr)
    {
        if (!addr)
            return false;

        while (addr)
        {
            switch (addr->getOp())
            {
            case kIROp_Var:
            case kIROp_Param:
                return isDifferentiableType(diffTypeContext, addr->getDataType());
            case kIROp_FieldAddress:
                if (!as<IRFieldAddress>(addr)->getField() ||
                    as<IRFieldAddress>(addr)
                            ->getField()
                            ->findDecoration<IRDerivativeMemberDecoration>() == nullptr)
                    return false;
                addr = as<IRFieldAddress>(addr)->getBase();
                break;
            case kIROp_GetElementPtr:
                if (!isDifferentiableType(
                        diffTypeContext,
                        as<IRGetElementPtr>(addr)->getBase()->getDataType()))
                    return false;
                addr = as<IRGetElementPtr>(addr)->getBase();
                break;
            default:
                return false;
            }
        }
        return false;
    }

    bool instHasNonTrivialDerivative(
        DifferentiableTypeConformanceContext& diffTypeContext,
        IRInst* inst)
    {
        switch (inst->getOp())
        {
        case kIROp_DetachDerivative:
            return false;
        case kIROp_Call:
            {
                auto call = as<IRCall>(inst);
                return isDifferentiableFunc(
                    call->getCallee(),
                    CheckDifferentiabilityPassContext::DifferentiableLevel::Forward);
            }
        default:
            return isDifferentiableType(diffTypeContext, inst->getDataType());
        }
    }

    bool checkType(IRInst* type)
    {
        type = unwrapAttributedType(type);
        if (as<IRTorchTensorType>(type))
            return false;
        else if (auto arrayType = as<IRArrayTypeBase>(type))
            return checkType(arrayType->getElementType());
        else if (auto structType = as<IRStructType>(type))
        {
            for (auto field : structType->getFields())
            {
                if (!checkType(field->getFieldType()))
                    return false;
            }
        }
        return true;
    }
    void checkForInvalidHostTypeUsage(IRGlobalValueWithCode* funcInst)
    {
        auto outerFuncInst = maybeFindOuterGeneric(funcInst);

        if (outerFuncInst->findDecoration<IRCudaHostDecoration>())
            return;
        if (outerFuncInst->findDecoration<IRTorchEntryPointDecoration>())
            return;

        bool isSynthesizeConstructor = false;

        if (auto constructor = funcInst->findDecoration<IRConstructorDecorartion>())
            isSynthesizeConstructor = constructor->getSynthesizedStatus();

        // This is a kernel function, we don't allow using TorchTensor type here.
        for (auto b : funcInst->getBlocks())
        {
            for (auto inst : b->getChildren())
            {
                if (!checkType(inst->getDataType()))
                {
                    if (isSynthesizeConstructor)
                    {
                        IRBuilder irBuilder(funcInst);
                        irBuilder.addDecoration(funcInst, kIROp_CudaHostDecoration);
                        return;
                    }

                    auto loc = inst->sourceLoc;
                    if (!loc.isValid())
                        loc = funcInst->sourceLoc;
                    sink->diagnose(loc, Diagnostics::invalidUseOfTorchTensorTypeInDeviceFunc);
                    return;
                }
            }
        }
    }

    void processFunc(IRGlobalValueWithCode* funcInst)
    {
        checkForInvalidHostTypeUsage(funcInst);

        if (!_isFuncMarkedForAutoDiff(funcInst))
            return;
        if (!funcInst->getFirstBlock())
            return;

        DifferentiableTypeConformanceContext diffTypeContext(&sharedContext);
        diffTypeContext.setFunc(funcInst);

        // We compute and track three different set of insts to complete our
        // data flow analysis.
        // `produceDiffSet` represents a set of insts that can provide a diff. This is conservative
        // on the positive side: a float literal is considered to be able to provide a diff.
        // `carryNonTrivialDiffSet` represents a set of insts that may carry a non-zero diff. This
        // is conservative on the negative side: if the inst does not provide a diff, or if we can
        // prove the diff is zero, we exclude the inst from the set. This makes
        // `carryNonTrivialDiffSet` a strict subset of `produceDiffSet`. `expectDiffSet` is a set of
        // insts that expects their operands to produce a diff. It is an error if they don't.
        InstHashSet produceDiffSet(funcInst->getModule());
        InstHashSet expectDiffSet(funcInst->getModule());
        InstHashSet carryNonTrivialDiffSet(funcInst->getModule());

        bool isDifferentiableReturnType = false;
        for (auto param : funcInst->getFirstBlock()->getParams())
        {
            if (isDifferentiableType(diffTypeContext, param->getFullType()))
            {
                produceDiffSet.add(param);
                carryNonTrivialDiffSet.add(param);
            }
        }
        if (auto funcType = as<IRFuncType>(funcInst->getDataType()))
        {
            if (isDifferentiableType(diffTypeContext, funcType->getResultType()))
            {
                isDifferentiableReturnType = true;
            }
        }

        DifferentiableLevel requiredDiffLevel = DifferentiableLevel::Forward;
        if (isBackwardDifferentiableFunc(funcInst))
            requiredDiffLevel = DifferentiableLevel::Backward;

        auto isInstProducingDiff = [&](IRInst* inst) -> bool
        {
            switch (inst->getOp())
            {
            case kIROp_FloatLit:
                return true;
            case kIROp_Call:
                return shouldTreatCallAsDifferentiable(inst) ||
                       isDifferentiableFunc(as<IRCall>(inst)->getCallee(), requiredDiffLevel) &&
                           isDifferentiableType(diffTypeContext, inst->getFullType());
            case kIROp_Load:
                // We don't have more knowledge on whether diff is available at the destination
                // address. Just assume it is producing diff if the dest address can hold a
                // derivative.
                // TODO: propagate the info if this is a load of a temporary variable intended
                // to receive result from an `out` parameter.
                return canAddressHoldDerivative(diffTypeContext, as<IRLoad>(inst)->getPtr());
            default:
                // default case is to assume the inst produces a diff value if any
                // of its operands produces a diff value.
                if (!isDifferentiableType(diffTypeContext, inst->getFullType()))
                    return false;
                for (UInt i = 0; i < inst->getOperandCount(); i++)
                {
                    if (produceDiffSet.contains(inst->getOperand(i)))
                    {
                        return true;
                    }
                }
                return false;
            }
        };

        auto isInstCarryingOverDiff = [&](IRInst* inst) -> bool
        {
            switch (inst->getOp())
            {
            case kIROp_DetachDerivative:
                return false;
            case kIROp_Call:
                if (shouldTreatCallAsDifferentiable(inst))
                    return false;
                return isDifferentiableFunc(as<IRCall>(inst)->getCallee(), requiredDiffLevel) &&
                       isDifferentiableType(diffTypeContext, inst->getFullType());
            case kIROp_Load:
                // We don't have more knowledge on whether diff is available at the destination
                // address. Just assume it is producing diff if the dest address can hold a
                // derivative.
                // TODO: propagate the info if this is a load of a temporary variable intended
                // to receive result from an `out` parameter.
                return canAddressHoldDerivative(diffTypeContext, as<IRLoad>(inst)->getPtr());
            default:
                // default case is to assume the inst produces a diff value if any
                // of its operands produces a diff value.
                if (!isDifferentiableType(diffTypeContext, inst->getFullType()))
                    return false;
                for (UInt i = 0; i < inst->getOperandCount(); i++)
                {
                    if (carryNonTrivialDiffSet.contains(inst->getOperand(i)))
                    {
                        return true;
                    }
                }
                return false;
            }
        };

        List<IRInst*> expectDiffInstWorkList;
        OrderedHashSet<IRInst*> expectDiffInstWorkListSet;
        auto addToExpectDiffWorkList = [&](IRInst* inst)
        {
            if (isInstInFunc(inst, funcInst))
            {
                if (expectDiffInstWorkListSet.add(inst))
                {
                    expectDiffInstWorkList.add(inst);
                }
            }
        };

        // Run data flow analysis and generate `produceDiffSet` and an intial `expectDiffSet`.
        Index lastProduceDiffCount = 0;
        do
        {
            lastProduceDiffCount = produceDiffSet.getCount();
            for (auto block : funcInst->getBlocks())
            {
                if (block != funcInst->getFirstBlock())
                {
                    UInt paramIndex = 0;
                    for (auto param : block->getParams())
                    {
                        for (auto p : block->getPredecessors())
                        {
                            // A Phi Node is producing diff if any of its candidate values are
                            // producing diff.
                            if (auto branch = as<IRUnconditionalBranch>(p->getTerminator()))
                            {
                                if (branch->getArgCount() > paramIndex)
                                {
                                    auto arg = branch->getArg(paramIndex);
                                    if (produceDiffSet.contains(arg))
                                        produceDiffSet.add(param);
                                    if (carryNonTrivialDiffSet.contains(arg))
                                        carryNonTrivialDiffSet.add(param);
                                }
                            }
                        }
                        paramIndex++;
                    }
                }
                for (auto inst : block->getChildren())
                {
                    if (isInstProducingDiff(inst))
                        produceDiffSet.add(inst);
                    if (isInstCarryingOverDiff(inst))
                        carryNonTrivialDiffSet.add(inst);
                    switch (inst->getOp())
                    {
                    case kIROp_Call:
                        if (isDifferentiableFunc(as<IRCall>(inst)->getCallee(), requiredDiffLevel))
                        {
                            addToExpectDiffWorkList(inst);
                        }
                        break;
                    case kIROp_Store:
                        {
                            auto storeInst = as<IRStore>(inst);
                            if (canAddressHoldDerivative(diffTypeContext, storeInst->getPtr()) &&
                                isDifferentiableType(
                                    diffTypeContext,
                                    as<IRStore>(inst)->getPtr()->getDataType()))
                            {
                                addToExpectDiffWorkList(storeInst->getVal());
                            }
                        }
                        break;
                    case kIROp_Return:
                        if (auto returnVal = as<IRReturn>(inst)->getVal())
                        {
                            if (isDifferentiableReturnType &&
                                isDifferentiableType(diffTypeContext, returnVal->getDataType()))
                            {
                                addToExpectDiffWorkList(inst);
                            }
                        }
                        break;
                    default:
                        break;
                    }
                }
            }
        } while (produceDiffSet.getCount() != lastProduceDiffCount);

        // Reverse propagate `expectDiffSet`.
        for (int i = 0; i < expectDiffInstWorkList.getCount(); i++)
        {
            auto inst = expectDiffInstWorkList[i];
            // Is inst in produceDiffSet?
            if (!produceDiffSet.contains(inst))
            {
                if (auto call = as<IRCall>(inst))
                {
                    const auto callee = call->getCallee();
                    // If inst's type is differentiable, and it is in expectDiffInstWorkList,
                    // then some user is expecting the result of the call to produce a derivative.
                    // In this case we need to issue a diagnostic.
                    if (isDifferentiableType(diffTypeContext, inst->getFullType()) &&
                        !isDifferentiableFunc(callee, requiredDiffLevel))
                    {
                        // No need to fail here if the function is no_diff in
                        // both inputs and all outputs, this is equivalent of
                        // inserting no_diff on this inst.
                        if (!isNeverDiffFuncType(cast<IRFuncType>(callee->getDataType())))
                        {
                            sink->diagnose(
                                inst,
                                Diagnostics::lossOfDerivativeDueToCallOfNonDifferentiableFunction,
                                getResolvedInstForDecorations(call->getCallee()),
                                requiredDiffLevel == DifferentiableLevel::Forward ? "forward"
                                                                                  : "backward");
                        }
                    }
                }
            }
            switch (inst->getOp())
            {
            case kIROp_Param:
                {
                    auto block = as<IRBlock>(inst->getParent());
                    if (block != funcInst->getFirstBlock())
                    {
                        auto paramIndex = getParamIndexInBlock(
                            as<IRParam, IRDynamicCastBehavior::NoUnwrap>(inst));
                        if (paramIndex != -1)
                        {
                            for (auto p : block->getPredecessors())
                            {
                                // A Phi Node is producing diff if any of its candidate values are
                                // producing diff.
                                if (auto branch = as<IRUnconditionalBranch>(p->getTerminator()))
                                {
                                    if (branch->getArgCount() > (UInt)paramIndex)
                                    {
                                        auto arg = branch->getArg(paramIndex);
                                        addToExpectDiffWorkList(arg);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
            case kIROp_Call:
                {
                    auto callInst = as<IRCall>(inst);
                    if (callInst->findDecoration<IRTreatCallAsDifferentiableDecoration>())
                        continue;
                    auto calleeFuncType = as<IRFuncType>(callInst->getCallee()->getFullType());
                    if (!calleeFuncType)
                        continue;
                    if (calleeFuncType->getParamCount() != callInst->getArgCount())
                        continue;
                    for (UInt a = 0; a < callInst->getArgCount(); a++)
                    {
                        auto arg = callInst->getArg(a);
                        auto paramType = calleeFuncType->getParamType(a);
                        if (!isDifferentiableType(diffTypeContext, paramType))
                            continue;
                        addToExpectDiffWorkList(arg);
                    }
                    break;
                }
            default:
                // Default behavior is to request all differentiable operands to provide
                // differential.
                for (UInt opIndex = 0; opIndex < inst->getOperandCount(); opIndex++)
                {
                    auto operand = inst->getOperand(opIndex);
                    if (isDifferentiableType(diffTypeContext, operand->getFullType()))
                    {
                        addToExpectDiffWorkList(operand);
                    }
                }
            }
        }

        // Make sure all loops are marked with either [MaxIters] or [ForceUnroll].
        for (auto block : funcInst->getBlocks())
        {
            auto loop = as<IRLoop>(block->getTerminator());
            if (!loop)
                continue;
            bool hasBackEdge = false;
            for (auto use = loop->getTargetBlock()->firstUse; use; use = use->nextUse)
            {
                if (use->getUser() != loop)
                {
                    hasBackEdge = true;
                    break;
                }
            }
            if (!hasBackEdge)
                continue;
            if (loop->findDecoration<IRLoopMaxItersDecoration>() ||
                loop->findDecoration<IRForceUnrollDecoration>())
            {
                // We are good.
            }
            else
            {
                sink->diagnose(loop->sourceLoc, Diagnostics::loopInDiffFuncRequireUnrollOrMaxIters);
            }
        }

        // Make sure all stores of differentiable values are into addresses that can hold
        // derivatives. If we are assigning a value to a non-differentiable location, we need to
        // make sure that value doesn't carray a non-zero diff.
        for (auto block : funcInst->getBlocks())
        {
            for (auto inst : block->getChildren())
            {
                if (auto storeInst = as<IRStore>(inst))
                {
                    if (carryNonTrivialDiffSet.contains(storeInst->getVal()) &&
                        !canAddressHoldDerivative(diffTypeContext, storeInst->getPtr()))
                    {
                        sink->diagnose(
                            storeInst->sourceLoc,
                            Diagnostics::lossOfDerivativeAssigningToNonDifferentiableLocation);
                    }
                }
                else if (auto callInst = as<IRCall>(inst))
                {
                    if (!isDifferentiableFunc(callInst->getCallee(), DifferentiableLevel::Forward))
                        continue;
                    auto calleeFuncType = as<IRFuncType>(callInst->getCallee()->getFullType());
                    if (!calleeFuncType)
                        continue;
                    if (calleeFuncType->getParamCount() != callInst->getArgCount())
                        continue;
                    for (UInt a = 0; a < callInst->getArgCount(); a++)
                    {
                        auto arg = callInst->getArg(a);
                        auto paramType = calleeFuncType->getParamType(a);
                        if (!isDifferentiableType(diffTypeContext, paramType))
                            continue;
                        if (as<IROutTypeBase>(paramType))
                        {
                            if (!canAddressHoldDerivative(diffTypeContext, arg))
                            {
                                sink->diagnose(
                                    arg->sourceLoc,
                                    Diagnostics::
                                        lossOfDerivativeUsingNonDifferentiableLocationAsOutArg);
                            }
                        }
                    }
                }
            }
        }
    }

    void processModule()
    {
        // Collect set of differentiable functions.
        HashSet<UnownedStringSlice> fwdDifferentiableSymbolNames, bwdDifferentiableSymbolNames;
        for (auto inst : module->getGlobalInsts())
        {
            if (_isDifferentiableFuncImpl(inst, DifferentiableLevel::Backward))
            {
                if (auto linkageDecor = inst->findDecoration<IRLinkageDecoration>())
                    bwdDifferentiableSymbolNames.add(linkageDecor->getMangledName());
                differentiableFunctions.add(inst, DifferentiableLevel::Backward);
            }
            else if (_isDifferentiableFuncImpl(inst, DifferentiableLevel::Forward))
            {
                if (auto linkageDecor = inst->findDecoration<IRLinkageDecoration>())
                    fwdDifferentiableSymbolNames.add(linkageDecor->getMangledName());
                differentiableFunctions.add(inst, DifferentiableLevel::Forward);
            }
        }
        for (auto inst : module->getGlobalInsts())
        {
            if (auto linkageDecor = inst->findDecoration<IRLinkageDecoration>())
            {
                if (bwdDifferentiableSymbolNames.contains(linkageDecor->getMangledName()))
                    differentiableFunctions[inst] = DifferentiableLevel::Backward;
                else if (fwdDifferentiableSymbolNames.contains(linkageDecor->getMangledName()))
                    differentiableFunctions.addIfNotExists(inst, DifferentiableLevel::Forward);
            }
        }

        if (!sharedContext.isInterfaceAvailable && !sharedContext.isPtrInterfaceAvailable)
            return;

        for (auto inst : module->getGlobalInsts())
        {
            if (auto genericInst = as<IRGeneric>(inst))
            {
                if (auto innerFunc =
                        as<IRGlobalValueWithCode>(findInnerMostGenericReturnVal(genericInst)))
                    processFunc(innerFunc);
            }
            else if (auto funcInst = as<IRGlobalValueWithCode>(inst))
            {
                processFunc(funcInst);
            }
        }
    }
};

void checkAutoDiffUsages(IRModule* module, DiagnosticSink* sink)
{
    CheckDifferentiabilityPassContext context(module, sink);
    context.processModule();
}

} // namespace Slang
