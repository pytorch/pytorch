// slang-ir-deduplicate-generic-children.h
#pragma once

namespace Slang
{
struct IRModule;
struct IRGeneric;

// Deduplicate insts inside a generic.
bool deduplicateGenericChildren(IRModule* module);
bool deduplicateGenericChildren(IRGeneric* genericInst);
} // namespace Slang
