#pragma once
#include <ATen/core/List.h>
#include <ATen/core/functional.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>

namespace torch {
namespace jit {

void tupleUnpack(Stack& stack);

void format(Stack& stack, size_t num_inputs);

void percentFormat(Stack& stack, size_t num_inputs);

void listUnpack(Stack& stack, size_t num_outputs);

void tupleConstruct(Stack& stack, size_t num_inputs);

void namedTupleConstruct(
    Stack& stack,
    at::TupleTypePtr type,
    size_t num_inputs);

void listConstruct(
    Stack& stack,
    const at::ListType& list_type,
    size_t num_inputs);

void dictConstruct(Stack& stack, const at::DictType& type, size_t num_inputs);

void createObject(Stack& stack, const at::ClassTypePtr& type);

void isinstance(Stack& stack, at::ArrayRef<at::TypePtr> types);

void tupleSlice(Stack& stack, size_t begin, size_t end);

void dequantize(Stack& stack);

} // namespace jit
} // namespace torch
