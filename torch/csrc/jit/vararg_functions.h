#pragma once
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <ATen/core/functional.h>
#include <ATen/core/List.h>

namespace torch {
namespace jit {
template<typename dtype> // int64_t, bool, double
void listConstructFunc(int num_inputs, Stack &stack) {
  auto inputs = peekSlice(stack, 0, num_inputs, num_inputs);
  c10::List<dtype> vals =
      c10::impl::toList(fmap(inputs, [](const IValue &v) { return v.to<dtype>(); }));
  drop(stack, num_inputs);
  push(stack, std::move(vals));
}

void tensorListConstructFunc(int num_inputs, Stack& stack);

void tupleUnpackFunc(int num_outputs, Stack& stack);

void formatFunc(int num_inputs, Stack& stack);
} // namespace jit
} // namespace torch
