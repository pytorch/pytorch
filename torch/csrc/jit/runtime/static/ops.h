#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/static/impl.h>

namespace torch {
namespace jit {

using SROperator =
    std::function<void(const ProcessedNode*, std::vector<IValue>&)>;
using SROpFunctor = std::function<SROperator(Node* n)>;
struct SROperatorFunctor {
  virtual SROperator Generate(Node*) = 0;
  virtual ~SROperatorFunctor() = default;
};

C10_DECLARE_REGISTRY(SROperatorRegistry, SROperatorFunctor);
#define REGISTER_OPERATOR_FUNCTOR(name, id, ...)             \
  struct SROperatorFunctor_##id : public SROperatorFunctor { \
    const SROpFunctor fn = __VA_ARGS__;                      \
    SROperator Generate(Node* n) override {                  \
      return fn(n);                                          \
    }                                                        \
  };                                                         \
  C10_REGISTER_CLASS(SROperatorRegistry, name, SROperatorFunctor_##id);

bool canRunOutOfPlace(Node* n);
std::function<void(const ProcessedNode*, std::vector<IValue>&)>
getOutOfPlaceOperation(Node* n);

bool canRunNatively(Node* n);
std::function<void(const ProcessedNode*, std::vector<IValue>&)>
getNativeOperation(Node* n);

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
  F(aten::leaky_relu)    \
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
