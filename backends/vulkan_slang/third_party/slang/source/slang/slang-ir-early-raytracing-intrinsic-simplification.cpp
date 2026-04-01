// slang-ir-early-raytracing-intrinsic-simplification.cpp
#include "slang-ir-early-raytracing-intrinsic-simplification.h"

#include "../core/slang-performance-profiler.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{
// ONLY should be used in this compilation unit
struct CacheOfDataToReplaceOps
{
    TargetProgram* target;
    IRModule* module;
    DiagnosticSink* sink;

    Dictionary<int, IRInst*> m_RayLocationToPayloads;
    Dictionary<int, IRInst*> m_RayLocationToAttributes;
    Dictionary<int, IRInst*> m_RayLocationToCallables;

    List<IRInst*> funcsToSearch;

    IRInst* getRayVariableFromLocation(IRInst* payloadVariable, Slang::IROp op)
    {
        IRBuilder builder(payloadVariable);
        IRInst** varLayoutPointsTo = nullptr;
        int intLitValue = -1;
        IRIntLit* intLit = as<IRIntLit>(payloadVariable);
        if (intLit)
        {
            intLitValue = int(intLit->getValue());
            if (kIROp_SPIRVAsmOperandRayPayloadFromLocation == op)
            {
                varLayoutPointsTo = m_RayLocationToPayloads.tryGetValue(intLitValue);
            }
            else if (kIROp_SPIRVAsmOperandRayAttributeFromLocation == op)
            {
                varLayoutPointsTo = m_RayLocationToAttributes.tryGetValue(intLitValue);
            }
            else if (kIROp_SPIRVAsmOperandRayCallableFromLocation == op)
            {
                SLANG_ASSERT(kIROp_SPIRVAsmOperandRayCallableFromLocation == op); // final case
                varLayoutPointsTo = m_RayLocationToCallables.tryGetValue(intLitValue);
            }
        }
        else
        {
            sink->diagnose(payloadVariable, Diagnostics::expectedIntegerConstantNotConstant);
        }

        IRInst* resultVariable;
        if (!varLayoutPointsTo)
        {
            // if somehow the location tied variable is missing and an error was not thrown by the
            // compiler
            resultVariable = builder.getIntValue(builder.getIntType(), 0);
            sink->diagnose(
                payloadVariable,
                Diagnostics::expectedRayTracingPayloadObjectAtLocationButMissing,
                intLitValue);
        }
        else
        {
            resultVariable = *varLayoutPointsTo;
        }
        return resultVariable;
    }

    void searchForGlobalsDataNeededInPass()
    {
        if (target->getOptionSet().getBoolOption(CompilerOptionName::AllowGLSL))
        {
            for (auto i : module->getGlobalInsts())
            {
                switch (i->getOp())
                {

                case kIROp_GlobalParam:
                case kIROp_GlobalVar:
                    {
                        for (auto decoration : i->getDecorations())
                        {
                            auto op = decoration->getOp();
                            if (op == kIROp_VulkanRayPayloadDecoration)
                            {
                                m_RayLocationToPayloads.set(
                                    int(getIntVal(decoration->getOperand(0))),
                                    i);
                            }
                            else if (op == kIROp_VulkanRayPayloadInDecoration)
                            {
                                m_RayLocationToPayloads.set(
                                    int(getIntVal(decoration->getOperand(0))),
                                    i);
                            }
                            else if (op == kIROp_VulkanHitObjectAttributesDecoration)
                            {
                                m_RayLocationToAttributes.set(
                                    int(getIntVal(decoration->getOperand(0))),
                                    i);
                            }
                            else if (op == kIROp_VulkanCallablePayloadDecoration)
                            {
                                m_RayLocationToCallables.set(
                                    int(getIntVal(decoration->getOperand(0))),
                                    i);
                            }
                            else if (op == kIROp_VulkanCallablePayloadInDecoration)
                            {
                                m_RayLocationToCallables.set(
                                    int(getIntVal(decoration->getOperand(0))),
                                    i);
                            }
                        }
                        break;
                    }
                case kIROp_Func:
                    {
                        funcsToSearch.add(i);
                        break;
                    }
                };
            }
        }
    }

    CacheOfDataToReplaceOps(TargetProgram* target, IRModule* module, DiagnosticSink* sink)
    {
        this->target = target;
        this->module = module;
        this->sink = sink;
    }
};

void recurseInFuncForOpsToReplace(IRInst* parent, CacheOfDataToReplaceOps* cache)
{

    if (as<IRSPIRVAsm>(parent))
    {
        for (auto i : parent->getChildren())
        {
            switch (i->getOp())
            {
            case kIROp_SPIRVAsmOperandRayPayloadFromLocation:
            case kIROp_SPIRVAsmOperandRayAttributeFromLocation:
            case kIROp_SPIRVAsmOperandRayCallableFromLocation:
                {
                    auto op = i->getOperand(0);
                    IRInst* globalVar = cache->getRayVariableFromLocation(op, i->getOp());
                    auto builder = IRBuilder(i);
                    builder.setInsertBefore(i);
                    auto spirvASM = builder.emitSPIRVAsmOperandInst(globalVar);
                    i->replaceUsesWith(spirvASM);
                    i->removeAndDeallocate();
                    break;
                }
            };
        }
    }

    for (auto i : parent->getChildren())
        recurseInFuncForOpsToReplace(i, cache);
}

void recurseAllOpsToReplace(CacheOfDataToReplaceOps* cache)
{
    for (auto func : cache->funcsToSearch)
    {
        recurseInFuncForOpsToReplace(func, cache);
    }
}

void replaceLocationIntrinsicsWithRaytracingObject(
    TargetProgram* target,
    IRModule* module,
    DiagnosticSink* sink)
{
    // currently only applies to GLSL syntax
    CacheOfDataToReplaceOps cache = CacheOfDataToReplaceOps(target, module, sink);
    cache.searchForGlobalsDataNeededInPass();

    if (target->getOptionSet().getBoolOption(CompilerOptionName::AllowGLSL))
    {
        recurseAllOpsToReplace(&cache);
    }
}
} // namespace Slang
