#include "slang-ir-specialize-target-switch.h"

#include "slang-capability.h"
#include "slang-compiler.h"
#include "slang-ir-dce.h"
#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{
void specializeTargetSwitch(
    TargetRequest* target,
    IRGlobalValueWithCode* code,
    DiagnosticSink* sink)
{
    if (auto gen = as<IRGeneric>(code))
    {
        auto retVal = findGenericReturnVal(gen);
        if (auto innerCode = as<IRGlobalValueWithCode>(retVal))
        {
            specializeTargetSwitch(target, innerCode, sink);
            return;
        }
    }

    bool changed = false;
    for (auto block : code->getBlocks())
    {
        bool failedImplies = false;
        if (auto targetSwitch = as<IRTargetSwitch>(block->getTerminator()))
        {
            bool isEqual;
            CapabilitySet bestCapSet = CapabilitySet::makeInvalid();
            IRBlock* targetBlock = nullptr;
            CapabilitySet::ImpliesReturnFlags impliesReturnType =
                CapabilitySet::ImpliesReturnFlags::NotImplied;
            for (UInt i = 0; i < targetSwitch->getCaseCount(); i++)
            {
                auto cap = (CapabilityName)getIntVal(targetSwitch->getCaseValue(i));
                if (target->getTargetCaps().isIncompatibleWith(cap))
                    continue;
                CapabilitySet capSet;
                if (cap == CapabilityName::Invalid) // `default` case
                    capSet = CapabilitySet::makeEmpty();
                else
                    capSet = CapabilitySet(cap);
                bool isBetterForTarget =
                    capSet.isBetterForTarget(bestCapSet, target->getTargetCaps(), isEqual);
                if (isBetterForTarget)
                {
                    impliesReturnType = target->getTargetCaps().atLeastOneSetImpliedInOther(capSet);
                    bool targetImpliesCapSet =
                        ((int)impliesReturnType & (int)CapabilitySet::ImpliesReturnFlags::Implied ||
                         capSet.isEmpty());
                    if (targetImpliesCapSet)
                    {
                        // Now check if bestCapSet contains targetCaps. If it does not then this is
                        // an invalid target
                        targetBlock = targetSwitch->getCaseBlock(i);
                        bestCapSet = capSet;
                    }
                    else
                        failedImplies = true;
                }
            }
            IRBuilder builder(targetSwitch);
            builder.setInsertBefore(targetSwitch);
            if (targetBlock)
            {
                builder.emitBranch(targetBlock);
            }
            else
            {
                // only error if we have the chance of setting a valid target switch, but did not
                // due to incompatability within same `target` atom. Otherwise we will have an issue
                // when we process a `__target_switch() { case metal: return; }` for glsl targets.
                if (failedImplies)
                    sink->diagnose(
                        targetSwitch->sourceLoc,
                        Diagnostics::profileIncompatibleWithTargetSwitch,
                        target->getTargetCaps());
                builder.emitMissingReturn();
            }
            targetSwitch->removeAndDeallocate();
            changed = true;
        }
    }
    if (changed)
    {
        // Remove unreachable blocks after specialization.
        eliminateDeadCode(code);
    }
}

void specializeTargetSwitch(TargetRequest* target, IRModule* module, DiagnosticSink* sink)
{
    for (auto globalInst : module->getGlobalInsts())
    {
        if (auto code = as<IRGlobalValueWithCode>(globalInst))
        {
            specializeTargetSwitch(target, code, sink);
        }
    }
}

} // namespace Slang
