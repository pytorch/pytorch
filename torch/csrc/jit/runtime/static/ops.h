#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/static/impl.h>

namespace torch {
namespace jit {

bool canRunOutOfPlace(Node* n);
std::function<void(StaticRuntime::ConstantMap&)> getOutOfPlaceOperation(
    Node* n);

#define SUPPORTED_OPS(F) \
  F(aten::__getitem__)   \
  F(aten::add)           \
  F(aten::addmm)         \
  F(aten::bmm)           \
  F(aten::cat)           \
  F(aten::clamp)         \
  F(aten::contiguous)    \
  F(aten::div)           \
  F(aten::flatten)       \
  F(aten::index_put_)    \
  F(aten::isnan)         \
  F(aten::matmul)        \
  F(aten::mul)           \
  F(aten::permute)       \
  F(aten::relu)          \
  F(aten::sigmoid)       \
  F(aten::size)          \
  F(aten::softmax)       \
  F(aten::t)             \
  F(aten::to)            \
  F(aten::transpose)     \
  F(aten::view)          \
  F(prim::Constant)      \
  F(prim::ListConstruct) \
  F(prim::TupleConstruct)

} // namespace jit
} // namespace torch
