#pragma once
#include <ATen/core/List.h>
#include <ATen/core/functional.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>

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

} // namespace jit
} // namespace torch
