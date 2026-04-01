#include "slang-ir-uniformity.h"

#include "slang-ir-dominators.h"
#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{
struct ValidateUniformityContext
{
    IRModule* module;
    DiagnosticSink* sink;

    HashSet<IRInst*> nonUniformInsts;
    ValidateUniformityContext* parentContext = nullptr;
    IRCall* call = nullptr;
    IRFunc* currentCallee = nullptr;

    bool isInstNonUniform(IRInst* inst)
    {
        auto context = this;
        while (context)
        {
            if (context->nonUniformInsts.contains(inst))
                return true;
            context = context->parentContext;
        }
        return false;
    }

    struct FunctionNonUniformInfoKey
    {
        IRFunc* func;
        UIntSet nonUniformParams;

        bool operator==(const FunctionNonUniformInfoKey& other) const
        {
            return func == other.func && nonUniformParams == other.nonUniformParams;
        }
        HashCode getHashCode() const
        {
            return combineHash(Slang::getHashCode(func), nonUniformParams.getHashCode());
        }
    };

    struct FunctionNonUniformInfo
    {
        UIntSet nonUniformParams;
        bool isResultNonUniform = false;
    };

    Dictionary<FunctionNonUniformInfoKey, FunctionNonUniformInfo> functionNonUniformInfos;

    template<typename F>
    void traverseControlDependentBlocks(IRDominatorTree* dom, IRInst* inst, const F& f)
    {
        auto block = as<IRBlock>(inst->getParent());
        if (!block)
            return;
        for (auto idom = dom->getImmediateDominator(block); idom;
             idom = dom->getImmediateDominator(idom))
        {
            if (as<IRUnconditionalBranch>(idom->getTerminator()))
                continue;
            if (auto ifelse = as<IRIfElse>(idom->getTerminator()))
            {
                if (dom->dominates(ifelse->getAfterBlock(), block))
                    continue;
            }
            else if (auto switchInst = as<IRSwitch>(idom->getTerminator()))
            {
                if (dom->dominates(switchInst->getBreakLabel(), block))
                    continue;
            }
            else if (auto loopInst = as<IRLoop>(idom->getTerminator()))
            {
                if (dom->dominates(loopInst->getBreakBlock(), block))
                    continue;
            }
            f(idom);
        }
    }

    FunctionNonUniformInfo* getFunctionNonUniformInfo(
        IRCall* callInst,
        const FunctionNonUniformInfoKey& key)
    {
        if (auto rs = functionNonUniformInfos.tryGetValue(key))
            return rs;

        // Is the function already being analyzed? If so exit early to avoid infinite recursion.
        for (auto context = this; context; context = context->parentContext)
        {
            if (context->currentCallee == key.func)
                return nullptr;
        }

        // If the function body has target intrinsic, we can't analyze it, and we
        // will use the fallback behavior (result is non-uniform if any of its arguments are
        // non-uniform).
        for (auto block : key.func->getBlocks())
        {
            if (as<IRGenericAsm>(block->getTerminator()))
            {
                return nullptr;
            }
        }

        ValidateUniformityContext subContext;
        subContext.module = module;
        subContext.sink = sink;
        subContext.parentContext = this;

        List<IRInst*> workList;
        Index paramIndex = 0;
        for (auto param : key.func->getParams())
        {
            if (key.nonUniformParams.contains(UInt(paramIndex)))
            {
                subContext.nonUniformInsts.add(param);
                workList.add(param);
            }
            paramIndex++;
        }
        subContext.call = callInst;
        subContext.currentCallee = key.func;
        subContext.propagateNonUniform(key.func, workList);

        FunctionNonUniformInfo info;
        info.nonUniformParams = key.nonUniformParams;
        paramIndex = 0;
        for (auto param : key.func->getParams())
        {
            if (subContext.nonUniformInsts.contains(param))
            {
                info.nonUniformParams.add(paramIndex);
            }
            paramIndex++;
        }

        // If the function has [NonUniformReturn] attribute,
        // treat its return value as non uniform.
        if (key.func->findDecorationImpl(kIROp_NonDynamicUniformReturnDecoration))
        {
            info.isResultNonUniform = true;
        }
        else
        {
            // The return value is non-uniform if the any values used in IRReturn is
            // non-uniform, or if the return insts are control-dependent on non-uniform
            // values.
            for (auto bb : key.func->getBlocks())
            {
                if (auto ret = as<IRReturn>(bb->getTerminator()))
                {
                    if (subContext.isInstNonUniform(ret->getVal()) ||
                        subContext.isInstNonUniform(ret))
                    {
                        info.isResultNonUniform = true;
                        break;
                    }
                }
            }
        }
        functionNonUniformInfos[key] = info;
        return functionNonUniformInfos.tryGetValue(key);
    }

    bool isDynamicUniformLocation(IRInst* addr)
    {
        while (addr)
        {
            switch (addr->getOp())
            {
            case kIROp_FieldAddress:
                if (as<IRFieldAddress>(addr)
                        ->getField()
                        ->findDecoration<IRDynamicUniformDecoration>())
                    return true;
                addr = as<IRFieldAddress>(addr)->getBase();
                break;
            case kIROp_GetElementPtr:
                addr = as<IRGetElementPtr>(addr)->getBase();
                break;
            case kIROp_GetOffsetPtr:
                addr = addr->getOperand(0);
                break;
            case kIROp_Param:
            case kIROp_Var:
                return addr->findDecoration<IRDynamicUniformDecoration>() != nullptr;
            default:
                addr = nullptr;
            }
        }
        return false;
    }

    void propagateNonUniform(IRFunc* root, List<IRInst*>& workList)
    {
        InstWorkList nextWorkList(module);
        InstHashSet workListSet(module);

        auto addToWorkList = [&](IRInst* inst)
        {
            if (workListSet.add(inst))
            {
                nonUniformInsts.add(inst);
                nextWorkList.add(inst);
            }
        };

        // Go through the children first to identify initial non-uniform insts.
        for (auto block : root->getBlocks())
        {
            for (auto inst = block->getFirstInst(); inst; inst = inst->getNextInst())
            {
                switch (inst->getOp())
                {
                case kIROp_Call:
                    {
                        auto callInst = as<IRCall>(inst);
                        auto callee = getResolvedInstForDecorations(callInst->getCallee());
                        if (callee->findDecorationImpl(kIROp_NonDynamicUniformReturnDecoration))
                        {
                            addToWorkList(inst);
                        }
                        break;
                    }
                }
            }
        }

        auto dom = module->findOrCreateDominatorTree(root);

        auto visitControlDependentBlock = [&](IRBlock* dependentBlock)
        {
            if (!dependentBlock)
                return;
            for (auto block : dom->getProperlyDominatedBlocks(dependentBlock))
            {
                for (auto inst = block->getFirstInst(); inst; inst = inst->getNextInst())
                {
                    switch (inst->getOp())
                    {
                    case kIROp_Store:
                    case kIROp_SwizzledStore:
                        addToWorkList(inst->getOperand(0));
                        break;
                    case kIROp_Return:
                        addToWorkList(inst);
                        break;
                    case kIROp_Call:
                        {
                            auto call = as<IRCall>(inst);
                            for (UInt i = 0; i < call->getArgCount(); i++)
                            {
                                if (as<IRPtrTypeBase>(call->getArg(i)))
                                    addToWorkList(call->getArg(i));
                            }
                        }
                        break;
                    }
                }
            }
        };

        while (workList.getCount())
        {
            for (Index i = 0; i < workList.getCount(); i++)
            {
                auto inst = workList[i];
                for (auto use = inst->firstUse; use; use = use->nextUse)
                {
                    auto user = use->getUser();
                    if (as<IRAttr>(user))
                        continue;
                    if (as<IRDecoration>(user))
                        continue;
                    switch (user->getOp())
                    {
                    case kIROp_TreatAsDynamicUniform:
                        continue;
                    case kIROp_FieldAddress:
                        {
                            if (isDynamicUniformLocation(user))
                                continue;
                            break;
                        }
                    case kIROp_FieldExtract:
                        {
                            if (as<IRFieldExtract>(user)
                                    ->findDecoration<IRDynamicUniformDecoration>())
                                continue;
                            break;
                        }
                    case kIROp_SwizzledStore:
                    case kIROp_Store:
                        {
                            if (use == user->getOperands() + 1)
                            {
                                auto ptr = user->getOperand(0);
                                addToWorkList(ptr);
                                if (isDynamicUniformLocation(ptr))
                                {
                                    sink->diagnose(
                                        user->sourceLoc,
                                        Diagnostics::expectDynamicUniformValue,
                                        ptr);
                                }
                                else
                                {
                                    // Conservatively treat the entire composite at root addr as
                                    // non-uniform.
                                    auto addrRoot = getRootAddr(ptr);
                                    addToWorkList(addrRoot);
                                }
                            }
                            break;
                        }
                    case kIROp_ifElse:
                        {
                            auto ifElse = as<IRIfElse>(user);
                            visitControlDependentBlock(ifElse->getTrueBlock());
                            visitControlDependentBlock(ifElse->getFalseBlock());
                            // Mark phi nodes as non-uniform if any of its incoming values are
                            // non-uniform.
                            for (auto param : ifElse->getAfterBlock()->getParams())
                                addToWorkList(param);
                            break;
                        }
                    case kIROp_Switch:
                        {
                            auto switchInst = as<IRSwitch>(user);
                            for (UInt c = 0; c < switchInst->getCaseCount(); c++)
                                visitControlDependentBlock(switchInst->getCaseLabel(c));
                            visitControlDependentBlock(switchInst->getDefaultLabel());
                            // Mark phi nodes as non-uniform if any of its incoming values are
                            // non-uniform.
                            for (auto param : switchInst->getBreakLabel()->getParams())
                                addToWorkList(param);
                            break;
                        }
                    case kIROp_Call:
                        {
                            auto callInst = as<IRCall>(user);
                            auto callee = getResolvedInstForDecorations(callInst->getCallee());
                            if (auto func = as<IRFunc>(callee))
                            {
                                if (func->getFirstBlock())
                                {
                                    FunctionNonUniformInfoKey key;
                                    key.func = func;
                                    for (UInt argi = 0; argi < callInst->getArgCount(); argi++)
                                    {
                                        if (nonUniformInsts.contains(callInst->getArg(argi)))
                                        {
                                            auto param = getParamAt(func->getFirstBlock(), argi);
                                            if (param->findDecoration<IRDynamicUniformDecoration>())
                                            {
                                                sink->diagnose(
                                                    callInst->sourceLoc,
                                                    Diagnostics::expectDynamicUniformArgument,
                                                    param);
                                            }
                                            else
                                            {
                                                key.nonUniformParams.add(i);
                                            }
                                        }
                                    }
                                    if (auto funcInfo = getFunctionNonUniformInfo(callInst, key))
                                    {
                                        for (UInt argi = 0; argi < callInst->getArgCount(); argi++)
                                        {
                                            if (funcInfo->nonUniformParams.contains(argi))
                                            {
                                                addToWorkList(callInst->getArg(argi));
                                            }
                                            if (funcInfo->isResultNonUniform)
                                            {
                                                addToWorkList(callInst);
                                            }
                                        }
                                        break;
                                    }
                                }
                            }
                            // The default behavior for calls is that the result is non-uniform if
                            // any of its arguments are non-uniform.
                            bool isNonUniformCall =
                                callee->findDecorationImpl(
                                    kIROp_NonDynamicUniformReturnDecoration) != nullptr;
                            if (!isNonUniformCall)
                            {
                                for (UInt argi = 0; argi < callInst->getArgCount(); argi++)
                                {
                                    if (nonUniformInsts.contains(callInst->getArg(argi)))
                                    {
                                        isNonUniformCall = true;
                                        break;
                                    }
                                }
                            }
                            if (isNonUniformCall)
                            {
                                addToWorkList(callInst);
                                for (UInt argi = 0; argi < callInst->getArgCount(); argi++)
                                {
                                    if (as<IRPtrTypeBase>(callInst->getArg(argi)->getDataType()))
                                    {
                                        addToWorkList(callInst->getArg(argi));
                                        // Conservatively treat the entire composite at root addr as
                                        // non-uniform.
                                        auto addrRoot = getRootAddr(callInst->getArg(argi));
                                        addToWorkList(addrRoot);
                                    }
                                }
                            }
                            break;
                        }
                    default:
                        break;
                    }
                    addToWorkList(user);
                }
            }
            workList.swapWith(nextWorkList.getList());
            nextWorkList.clear();
        }
    }

    void analyzeModule()
    {
        InstWorkList workList(module);

        for (auto globalInst : module->getGlobalInsts())
        {
            if (auto code = as<IRGlobalValueWithCode>(globalInst))
            {
                auto func = getResolvedInstForDecorations(code);
                if (func->findDecorationImpl(kIROp_NonDynamicUniformReturnDecoration))
                {
                    nonUniformInsts.add(code);
                }
            }
            if (globalInst->findDecoration<IREntryPointDecoration>())
            {
                auto func = as<IRFunc>(globalInst);
                if (!func)
                    continue;
                for (auto param : func->getParams())
                {
                    auto varLayout = findVarLayout(param);
                    if (isVaryingParameter(varLayout) ||
                        varLayout->findAttr<IRSystemValueSemanticAttr>())
                    {
                        nonUniformInsts.add(param);
                        workList.add(param);
                    }
                }
                currentCallee = func;
                call = nullptr;
                propagateNonUniform(func, workList.getList());
            }
        }
        workList.clear();

        eliminateAsDynamicUniformInst();
    }

    void eliminateAsDynamicUniformInst()
    {
        InstWorkList workList(module);
        workList.add(module->getModuleInst());
        for (Index i = 0; i < workList.getCount(); i++)
        {
            auto inst = workList[i];
            if (inst->getOp() == kIROp_TreatAsDynamicUniform)
            {
                auto val = inst->getOperand(0);
                inst->replaceUsesWith(val);
                inst->removeAndDeallocate();
            }
            else
            {
                for (auto child = inst->getFirstChild(); child; child = child->getNextInst())
                {
                    workList.add(child);
                }
            }
        }
    }
};

void validateUniformity(IRModule* module, DiagnosticSink* sink)
{
    ValidateUniformityContext context;
    context.module = module;
    context.sink = sink;
    context.analyzeModule();
}
} // namespace Slang
