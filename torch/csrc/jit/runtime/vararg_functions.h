#pragma once
#include <ATen/core/List.h>
#include <ATen/core/functional.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>

namespace torch::jit {

void tupleUnpack(Stack& stack);

void format(Stack& stack, size_t num_inputs);

void einsum(Stack& stack, size_t num_inputs);

void percentFormat(Stack& stack, size_t num_inputs);

void listUnpack(Stack& stack, size_t num_outputs);

void tupleConstruct(Stack& stack, size_t num_inputs);

void namedTupleConstruct(Stack& stack, c10::TypePtr type, size_t num_inputs);

void listConstruct(Stack& stack, const c10::Type& list_type, size_t num_inputs);

void dictConstruct(Stack& stack, const c10::Type& type, size_t num_inputs);

// as weak_ref will create a Object with a non-owning CompilationUnit reference,
// for use as a constant in the Graph to avoid a reference cycle
void createObject(
    Stack& stack,
    const at::ClassTypePtr& type,
    bool as_weak_ref = false);

void isinstance(Stack& stack, at::ArrayRef<at::TypePtr> types);

void tupleSlice(Stack& stack, size_t begin, size_t end);

void dequantize(Stack& stack);

} // namespace torch::jit
