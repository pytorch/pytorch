#include "slang-ir-call-graph.h"

#include "slang-ir-clone.h"
#include "slang-ir-insts.h"

namespace Slang
{

void buildEntryPointReferenceGraph(
    Dictionary<IRInst*, HashSet<IRFunc*>>& referencingEntryPoints,
    IRModule* module)
{
    struct WorkItem
    {
        IRFunc* entryPoint;
        IRInst* inst;

        HashCode getHashCode() const
        {
            return combineHash(Slang::getHashCode(entryPoint), Slang::getHashCode(inst));
        }
        bool operator==(const WorkItem& other) const
        {
            return entryPoint == other.entryPoint && inst == other.inst;
        }
    };
    HashSet<WorkItem> workListSet;
    List<WorkItem> workList;
    auto addToWorkList = [&](WorkItem item)
    {
        if (workListSet.add(item))
            workList.add(item);
    };

    auto registerEntryPointReference = [&](IRFunc* entryPoint, IRInst* inst)
    {
        if (auto set = referencingEntryPoints.tryGetValue(inst))
            set->add(entryPoint);
        else
        {
            HashSet<IRFunc*> newSet;
            newSet.add(entryPoint);
            referencingEntryPoints.add(inst, _Move(newSet));
        }
    };
    auto visit = [&](IRFunc* entryPoint, IRInst* inst)
    {
        if (auto code = as<IRGlobalValueWithCode>(inst))
        {
            registerEntryPointReference(entryPoint, inst);
            for (auto child : code->getChildren())
            {
                addToWorkList({entryPoint, child});
            }
            return;
        }
        switch (inst->getOp())
        {
        case kIROp_GlobalParam:
        case kIROp_SPIRVAsmOperandBuiltinVar:
            registerEntryPointReference(entryPoint, inst);
            break;
        case kIROp_Block:
        case kIROp_SPIRVAsm:
            for (auto child : inst->getChildren())
            {
                addToWorkList({entryPoint, child});
            }
            break;
        case kIROp_Call:
            {
                auto call = as<IRCall>(inst);
                addToWorkList({entryPoint, call->getCallee()});
            }
            break;
        case kIROp_SPIRVAsmOperandInst:
            {
                auto operand = as<IRSPIRVAsmOperandInst>(inst);
                addToWorkList({entryPoint, operand->getValue()});
            }
            break;
        }
        for (UInt i = 0; i < inst->getOperandCount(); i++)
        {
            auto operand = inst->getOperand(i);
            switch (operand->getOp())
            {
            case kIROp_GlobalParam:
            case kIROp_GlobalVar:
            case kIROp_SPIRVAsmOperandBuiltinVar:
            case kIROp_Generic:
                addToWorkList({entryPoint, operand});
                break;
            }
        }
    };

    for (auto globalInst : module->getGlobalInsts())
    {
        if (globalInst->getOp() == kIROp_Func &&
            globalInst->findDecoration<IREntryPointDecoration>())
        {
            visit(as<IRFunc>(globalInst), globalInst);
        }
    }
    for (Index i = 0; i < workList.getCount(); i++)
        visit(workList[i].entryPoint, workList[i].inst);
}

HashSet<IRFunc*>* getReferencingEntryPoints(
    Dictionary<IRInst*, HashSet<IRFunc*>>& m_referencingEntryPoints,
    IRInst* inst)
{
    auto* referencingEntryPoints = m_referencingEntryPoints.tryGetValue(inst);
    if (!referencingEntryPoints)
        return nullptr;
    return referencingEntryPoints;
}

} // namespace Slang
