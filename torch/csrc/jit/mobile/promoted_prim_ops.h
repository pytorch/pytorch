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

void device(Stack& stack);

void dtype(Stack& stack);

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
