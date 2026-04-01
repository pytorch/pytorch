#include "slang-ir-specialize-stage-switch.h"

#include "slang-capability.h"
#include "slang-compiler.h"
#include "slang-ir-call-graph.h"
#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{
bool funcHasGetCurrentStageInst(IRGlobalValueWithCode* func)
{
    for (auto block : func->getBlocks())
    {
        for (auto inst : block->getChildren())
        {
            if (inst->getOp() == kIROp_GetCurrentStage)
            {
                return true;
            }
        }
    }
    return false;
}

void discoverStageSpecificFunctions(HashSet<IRInst*>& stageSpecificFunctions, IRModule* module)
{
    List<IRInst*> workList;
    for (auto inst : module->getGlobalInsts())
    {
        if (auto func = as<IRGlobalValueWithCode>(inst))
        {
            if (funcHasGetCurrentStageInst(func))
            {
                workList.add(inst);
                stageSpecificFunctions.add(func);
            }
        }
    }
    for (Index i = 0; i < workList.getCount(); i++)
    {
        auto callee = workList[i];
        traverseUses(
            callee,
            [&](IRUse* use)
            {
                if (use->getUser()->getOp() == kIROp_Call)
                {
                    auto parentFunc = getParentFunc(use->getUser());
                    if (parentFunc && stageSpecificFunctions.add(parentFunc))
                    {
                        workList.add(parentFunc);
                    }
                }
            });
    }
}

// Given a func, replace all `GetCurrentStage` insts with the given stage, and rewrite all calls to
// stage specific functions to the specialized function for the given stage.
//
void specializeFuncToStage(
    Stage stage,
    IRGlobalValueWithCode* func,
    Dictionary<IRInst*, Dictionary<Stage, IRInst*>>& mapFuncToStageSpecializedFunc)
{
    // Collect all insts that may need to be modified.
    List<IRInst*> instsToModify;
    for (auto block : func->getBlocks())
    {
        for (auto inst : block->getChildren())
        {
            switch (inst->getOp())
            {
            case kIROp_GetCurrentStage:
            case kIROp_Call:
                instsToModify.add(inst);
                break;
            }
        }
    }

    IRInst* stageVal = nullptr;
    IRBuilder builder(func);
    for (auto inst : instsToModify)
    {
        builder.setInsertBefore(inst);

        switch (inst->getOp())
        {
        case kIROp_GetCurrentStage:
            {
                // Replace `GetCurrentStage` with the stage it is specialized to.
                if (!stageVal)
                {
                    stageVal = builder.getIntValue((IRIntegerValue)stage);
                }
                inst->replaceUsesWith(stageVal);
                inst->removeAndDeallocate();
                break;
            }
        case kIROp_Call:
            {
                // Replace calls to stage specific functions with the specialized function for the
                // given stage.
                auto callInst = static_cast<IRCall*>(inst);
                auto callee = callInst->getCallee();
                auto specializedFuncs = mapFuncToStageSpecializedFunc.tryGetValue(callee);
                if (specializedFuncs)
                {
                    auto specializedFunc = specializedFuncs->tryGetValue(stage);
                    if (specializedFunc)
                    {
                        builder.replaceOperand(callInst->getCalleeUse(), *specializedFunc);
                    }
                }
                break;
            }
        }
    }
}

void specializeStageSwitch(IRModule* module)
{
    Dictionary<IRInst*, HashSet<IRFunc*>> mapInstToReferencingEntryPoints;
    buildEntryPointReferenceGraph(mapInstToReferencingEntryPoints, module);

    HashSet<IRInst*> stageSpecificFunctions;
    discoverStageSpecificFunctions(stageSpecificFunctions, module);

    // Clone all stage specific functions for each stage they are used in.
    Dictionary<IRInst*, Dictionary<Stage, IRInst*>> mapFuncToStageSpecializedFunc;
    for (auto func : stageSpecificFunctions)
    {
        auto referencingEntryPoints = mapInstToReferencingEntryPoints.tryGetValue(func);
        if (!referencingEntryPoints)
            continue;
        if (func->findDecoration<IREntryPointDecoration>())
            continue;
        Dictionary<Stage, IRInst*> specializedFuncs;
        for (auto entryPoint : *referencingEntryPoints)
        {
            auto entryPointDecor = entryPoint->findDecoration<IREntryPointDecoration>();
            if (!entryPointDecor)
                continue;
            auto stage = entryPointDecor->getProfile().getStage();
            auto stageSpecializedFunc = specializedFuncs.tryGetValue(stage);
            if (stageSpecializedFunc)
                continue;
            IRCloneEnv cloneEnv;
            IRBuilder builder(func);
            builder.setInsertBefore(func);
            auto clonedFunc = cloneInst(&cloneEnv, &builder, func);
            specializedFuncs[stage] = clonedFunc;
        }
        mapFuncToStageSpecializedFunc.add(func, _Move(specializedFuncs));
    }

    // Rewrite entrypoint and cloned functions to replace `GetCurrentStage` with the stage they are
    // specialized to.
    for (auto func : stageSpecificFunctions)
    {
        // Is this an entrypoint?
        if (auto entryPointDecor = func->findDecoration<IREntryPointDecoration>())
        {
            auto stage = entryPointDecor->getProfile().getStage();
            specializeFuncToStage(
                stage,
                as<IRGlobalValueWithCode>(func),
                mapFuncToStageSpecializedFunc);
        }
        else
        {
            // Is this a cloned function?
            auto specializedFuncs = mapFuncToStageSpecializedFunc.tryGetValue(func);
            if (!specializedFuncs)
                continue;
            for (auto pair : *specializedFuncs)
            {
                auto stage = pair.first;
                auto specializedFunc = pair.second;
                specializeFuncToStage(
                    stage,
                    as<IRGlobalValueWithCode>(specializedFunc),
                    mapFuncToStageSpecializedFunc);
            }
        }
    }

    // Remove all original stage specific functions.
    for (auto f : mapFuncToStageSpecializedFunc)
    {
        f.first->removeAndDeallocate();
    }
}

} // namespace Slang
