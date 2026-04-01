#ifndef SLANG_EMIT_VM_H
#define SLANG_EMIT_VM_H

#include "slang-emit-base.h"
#include "slang-ir-link.h"
#include "slang-vm-bytecode.h"

namespace Slang
{

struct VMByteCodeFunctionBuilder
{
    VMOperand name = {};
    uint32_t workingSetSizeInBytes = 0;
    List<uint8_t> code;
    List<Index> instOffsets;
    List<uint32_t> parameterOffsets;
    uint32_t resultSize = 0;
    uint32_t parameterSize = 0;
};

struct VMByteCodeBuilder
{
    List<VMByteCodeFunctionBuilder> functions;
    ComPtr<slang::IBlob> kernelBlob;

    List<uint8_t> constantSection;
    List<uint32_t> stringOffsets;
    SlangResult serialize(slang::IBlob** outBlob);
};

SlangResult emitVMByteCodeForEntryPoints(
    CodeGenContext* codeGenContext,
    LinkedIR& linkedIR,
    VMByteCodeBuilder& byteCode);
} // namespace Slang

#endif
