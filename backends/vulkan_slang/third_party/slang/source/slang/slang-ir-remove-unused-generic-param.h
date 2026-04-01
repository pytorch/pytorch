// slang-ir-remove-unused-generic-param.h
#pragma once

namespace Slang
{
struct IRModule;

bool removeUnusedGenericParam(IRModule* module);
} // namespace Slang
