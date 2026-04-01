// slang-ir-strip-default-construct.h
#pragma once

namespace Slang
{
struct IRModule;

/// Strip the contents of all witness table instructions from the given IR `module`
void removeRawDefaultConstructors(IRModule* module);

} // namespace Slang
