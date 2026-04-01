// slang-ir-strip.cpp
#include "slang-ir-strip.h"

#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{

/// Should `inst` be stripped, given the current `options`?
static bool _shouldStripInst(IRInst* inst, IRStripOptions const& options)
{
    switch (inst->getOp())
    {
    default:
        return false;

    case kIROp_HighLevelDeclDecoration:
        return true;

    case kIROp_NameHintDecoration:
        return options.shouldStripNameHints;
    }
}

/// Recursively strip `inst` and its children according to `options`.
static void _stripFrontEndOnlyInstructionsRec(IRInst* inst, IRStripOptions const& options)
{
    if (_shouldStripInst(inst, options))
    {
        inst->removeAndDeallocate();
        return;
    }

    if (options.stripSourceLocs)
    {
        inst->sourceLoc = SourceLoc();
    }

    IRInst* nextChild = nullptr;
    for (IRInst* child = inst->getFirstDecorationOrChild(); child; child = nextChild)
    {
        nextChild = child->getNextInst();

        _stripFrontEndOnlyInstructionsRec(child, options);
    }
}

void stripFrontEndOnlyInstructions(IRModule* module, IRStripOptions const& options)
{
    _stripFrontEndOnlyInstructionsRec(module->getModuleInst(), options);
}

void stripImportedWitnessTable(IRModule* module)
{
    for (auto globalInst : module->getGlobalInsts())
    {
        auto inst = globalInst;
        switch (globalInst->getOp())
        {
        case kIROp_Generic:
            inst = findInnerMostGenericReturnVal(as<IRGeneric>(globalInst));
            break;
        case kIROp_WitnessTable:
            break;
        default:
            continue;
        }
        if (inst->getOp() != kIROp_WitnessTable)
            continue;
        if (!globalInst->findDecoration<IRImportDecoration>())
            continue;
        IRInst* nextChild = nullptr;
        for (auto child = inst->getFirstChild(); child;)
        {
            nextChild = child->getNextInst();
            if (child->getOp() == kIROp_WitnessTable)
                child->removeAndDeallocate();
            child = nextChild;
        }
    }
}

} // namespace Slang
