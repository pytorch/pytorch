// slang-ir-autodiff-cfg-norm.h
#pragma once

#include "slang-ir-insts.h"

namespace Slang
{
struct IRModule;

struct IRCFGNormalizationPass
{
    DiagnosticSink* sink;
};

/// Eliminate "break" statements from breakable regions
/// (loops, switch-case). This will use temporary booleans
/// instead of a break statement, in order to ensure all
/// branches inside the breakable region always have a valid
/// "after" block.
///
void normalizeCFG(
    IRModule* module,
    IRGlobalValueWithCode* func,
    IRCFGNormalizationPass const& options = IRCFGNormalizationPass());

IRBlock* getOrCreateTopLevelCondition(IRLoop* loopInst);
} // namespace Slang
