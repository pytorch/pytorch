#pragma once

namespace Slang
{
struct IRInst;
struct IRModule;

enum class NonUniformResourceIndexFloatMode
{
    Textual,
    SPIRV,
};

void processNonUniformResourceIndex(
    IRInst* nonUniformResourceIndexInst,
    NonUniformResourceIndexFloatMode floatMode);

void floatNonUniformResourceIndex(IRModule* module, NonUniformResourceIndexFloatMode floatMode);

} // namespace Slang
