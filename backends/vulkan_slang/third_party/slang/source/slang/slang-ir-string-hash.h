// slang-ir-string-hash.h
#pragma once

#include "../core/slang-string-slice-pool.h"
#include "slang-ir.h"

namespace Slang
{

struct IRModule;
class DiagnosticSink;

// Finds the global GlobalHashedStringLiterals instruction for the module if there is one, and then
// adds all of it's strings to ioPool.
void findGlobalHashedStringLiterals(IRModule* module, StringSlicePool& ioPool);

// Given a pool, with > 0 strings adds a GlobalHashedStringLiterals to the module.
void addGlobalHashedStringLiterals(const StringSlicePool& pool, IRModule* module);

// Find all of the IRGetStringHash instructions within the module
void findGetStringHashInsts(IRModule* module, List<IRGetStringHash*>& outInsts);

// Looks at all getStringHash instructions to make sure they access something valid (like a string
// literal) sink is optional and can be passed as nullptr
Result checkGetStringHashInsts(IRModule* module, DiagnosticSink* sink);

} // namespace Slang
