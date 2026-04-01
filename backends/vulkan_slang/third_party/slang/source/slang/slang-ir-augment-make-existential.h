// slang-ir-augment-make-existential.h
#pragma once

#include "slang-ir.h"

namespace Slang
{
struct IRModule;
class DiagnosticSink;

/// Augment `MakeExistential(v, w)` insts to `MakeExistentialWithRTTI(v, w, t)`,
/// where v is a concrete typed value, w is a witness table, and t is the type of
/// v.
void augmentMakeExistentialInsts(IRModule* module);

} // namespace Slang
