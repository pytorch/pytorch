#pragma once

namespace Slang
{
struct IRModule;

/// Strip all debug info instructions from `irModule`
void stripDebugInfo(IRModule* irModule);
} // namespace Slang
