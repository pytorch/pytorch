// slang-ir-collect-global-uniforms.h
#pragma once

#include "slang-compiler.h"

namespace Slang
{
struct IRModule;
struct IRVarLayout;

/// Collect global-scope shader parameters that use uniform/ordinary
/// storage into a single `struct` (possibly wrapped in a constant buffer).
///
void collectGlobalUniformParameters(IRModule* module, IRVarLayout* globalScopeVarLayout);

} // namespace Slang
