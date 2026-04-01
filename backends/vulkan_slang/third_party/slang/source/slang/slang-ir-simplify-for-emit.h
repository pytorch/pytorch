// slang-ir-simplify-for-emit.h
#pragma once

namespace Slang
{
struct IRModule;
class TargetRequest;

void simplifyForEmit(IRModule* inModule, TargetRequest* req);
} // namespace Slang
