#pragma once
#include <torch/csrc/jit/mobile/prim_ops_registery.h>
#include <torch/csrc/jit/mobile/register_ops_common_utils.h>

namespace torch {
namespace jit {

void tupleIndex(Stack& stack);

void raiseException(Stack& stack);

void is(Stack& stack);

void unInitialized(Stack& stack);

void isNot(Stack& stack);

void aten_format(Stack& stack);

void size(Stack& stack);

void sym_size(Stack& stack);

void sym_size_int(Stack& stack);

void sym_stride_int(Stack& stack);

void sym_numel(Stack& stack);

void sym_storage_offset(Stack& stack);

void sym_stride(Stack& stack);

void device(Stack& stack);

void device_with_index(Stack& stack);

void dtype(Stack& stack);

void layout(Stack& stack);

void toPrimDType(Stack& stack);

void dim(Stack& stack);

void _not(Stack& stack);

void boolTensor(Stack& stack);

void toList(Stack& stack);

void numToTensorScalar(Stack& stack);

void isCuda(Stack& stack);

void numToTensorBool(Stack& stack);

void dictIndex(Stack& stack);

void raiseExceptionWithMessage(Stack& stack);

} // namespace jit
} // namespace torch
