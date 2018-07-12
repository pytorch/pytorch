#include "torch/csrc/jit/constants.h"
#include "torch/csrc/jit/operator.h"

namespace torch { namespace jit {

Value* createConstant(
    Graph& g,
    IValue val,
    at::optional<script::SourceRange> loc) {

  TypePtr typ = nullptr;
  Node * n;
  at::Tensor ref;
  if(val.isTensor()) {
    ref = std::move(val).toTensor();
    JIT_ASSERT(ref.defined());
  } else if(val.isInt()) { // eventually we will keep these as non-tensors
    typ = IntType::get();
    ref = at::full({}, val.toInt(), at::kLong);
  } else if(val.isDouble()) {
    typ = FloatType::get();
    ref = at::full({}, val.toDouble(), at::kFloat);
  } else if(val.isIntList()) {
    typ = ListType::ofInts();
    ref = as_tensor(val.toIntList()->elements());
  } else {
    throw std::runtime_error("Unsupported value kind: " + val.tagKind());
  }
  n = g.create(prim::Constant);
  n->t_(attr::value, ref.clone());
  n->output()->inferTypeFrom(ref);
  if(loc)
    n->setSourceLocation(std::make_shared<script::SourceRange>(*loc));
  if(typ)
    n->output()->setType(typ);
  return g.insertNode(n)->output();
}

RegisterOperators reg({
  Operator(
      prim::Constant,
      [](Node* node) {
        auto t = autograd::make_variable(node->t(attr::value));
        return [t](Stack& stack) {
          stack.push_back(t);
          return 0;
        };
      }),
});

}}
