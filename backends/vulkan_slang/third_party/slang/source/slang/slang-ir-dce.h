// slang-ir-dce.h
#pragma once

#include "slang-ir-insts.h"

namespace Slang
{
struct IRModule;

struct IRDeadCodeEliminationOptions
{
    bool keepExportsAlive = false;
    bool keepLayoutsAlive = false;
    bool useFastAnalysis = false;
    bool keepGlobalParamsAlive = true;
};

/// Eliminate "dead" code from the given IR module.
///
/// This pass is primarily designed for flow-insensitive
/// "global" dead code elimination (DCE), such as removing
/// types that are unused, functions that are never called,
/// etc.
/// Returns true if changed.
bool eliminateDeadCode(
    IRModule* module,
    IRDeadCodeEliminationOptions const& options = IRDeadCodeEliminationOptions());

bool eliminateDeadCode(
    IRInst* root,
    IRDeadCodeEliminationOptions const& options = IRDeadCodeEliminationOptions());

bool shouldInstBeLiveIfParentIsLive(IRInst* inst, IRDeadCodeEliminationOptions options);

bool isWeakReferenceOperand(IRInst* inst, UInt operandIndex);

bool trimOptimizableTypes(IRModule* module);
} // namespace Slang
