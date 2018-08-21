#include "torch/csrc/jit/constants.h"
#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/custom_operator.h"
#include "torch/csrc/autograd/variable.h"

namespace torch { namespace jit {

// IValue -> Constant node
Value* insertConstant(
    Graph& g,
    IValue val,
    at::optional<SourceRange> loc) {
  Node * n = g.create(prim::Constant);
  if(val.isTensor()) {
    at::Tensor ref = std::move(val).toTensor();
    JIT_ASSERT(ref.defined());
    n->output()->inferTypeFrom(ref); // note: before t_ because of std::move(ref)
    n->t_(attr::value, std::move(ref));
  } else if(val.isInt()) {
    n->i_(attr::value, val.toInt());
    n->output()->setType(IntType::get());
  } else if(val.isDouble()) {
    n->f_(attr::value, val.toDouble());
    n->output()->setType(FloatType::get());
  } else if(val.isIntList()) {
    n->is_(attr::value, val.toIntList()->elements());
    n->output()->setType(ListType::ofInts());
  } else if(val.isTensorList()) {
    n->ts_(attr::value, fmap(val.toTensorList()->elements(), [](const at::Tensor & t) {
      return autograd::Variable(t).data();
    }));
    n->output()->setType(ListType::ofTensors());
  } else if(val.isString()) {
    n->s_(attr::string, val.toString()->string());
    n->output()->setType(StringType::get());
  } else {
    throw std::runtime_error("Unsupported value kind: " + val.tagKind());
  }
  if(loc)
    n->setSourceLocation(std::make_shared<SourceRange>(*loc));
  return g.insertNode(n)->output();
}

RegisterOperators reg({
  // Implementation of constant node, computes and IValue
  Operator(
      prim::Constant,
      [](Node* node) -> Operation {
        TypePtr type = node->output()->type();
        if(type->isSubtypeOf(DynamicType::get())) {
          auto t = autograd::make_variable(node->t(attr::value));
          return [t](Stack& stack) {
            stack.push_back(t);
            return 0;
          };
        } else if (
            type->isSubtypeOf(NumberType::get()) &&
            node->kindOf(attr::value) == AttributeKind::i) {
          auto i = node->i(attr::value);
          return [i](Stack& stack) {
            push(stack, i);
            return 0;
          };
        } else if (
            type->isSubtypeOf(NumberType::get()) &&
            node->kindOf(attr::value) == AttributeKind::f) {
          auto f = node->f(attr::value);
          return [f](Stack& stack) {
            push(stack, f);
            return 0;
          };
        } else if(type->isSubtypeOf(ListType::ofInts())) {
          auto is = node->is(attr::value);
          return [is](Stack& stack) {
            push(stack, is);
            return 0;
          };
        } else if(type->isSubtypeOf(ListType::ofTensors())) {
          auto ts = fmap(node->ts(attr::value), [](const at::Tensor & t) -> at::Tensor {
            return autograd::make_variable(t);
          });
          return [ts](Stack& stack) {
            push(stack, ts);
            return 0;
          };
        } else if (type == StringType::get()) {
          auto s = node->s(attr::string);
          return [s](Stack& stack) {
            push(stack, s);
            return 0;
          };
        } else {
          std::stringstream ss;
          ss << "constant literal not supported for: " << type->str();
          throw std::runtime_error(ss.str());
        }
      }),
});

at::optional<IValue> toIValue(Value* v) {
  if(v->node()->kind() != prim::Constant)
    return at::nullopt;
  // use implemenation of prim::Constant to compute the output IValue
  auto op = getOperation(v->node());
  Stack stack;
  op(stack);
  return stack.back();
}

}}
