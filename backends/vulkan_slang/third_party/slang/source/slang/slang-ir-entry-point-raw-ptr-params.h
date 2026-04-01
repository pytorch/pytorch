// slang-ir-entry-point-raw-ptr-params.h
#pragma once

namespace Slang
{
struct IRModule;

/// Convert any entry-point parameters that use pointer types to use raw pointers (`void*`)
void convertEntryPointPtrParamsToRawPtrs(IRModule* module);

} // namespace Slang
