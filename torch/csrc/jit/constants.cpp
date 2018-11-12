#include "torch/csrc/jit/constants.h"
#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/custom_operator.h"
#include "torch/csrc/autograd/variable.h"

namespace torch { namespace jit {

// IValue -> Constant node
Value* insertConstant(
    Graph& g,
    IValue val,
    c10::optional<SourceRange> loc,
    c10::optional<ScopePtr> scope) {
  Node * n = g.create(prim::Constant);
  if(val.isTensor()) {
    at::Tensor ref = std::move(val).toTensor();
    if(!ref.defined()) {
      throw constant_not_supported_error("undefined tensors cannot become constants");
    }
    if (ref.is_variable()) {
      ref = autograd::Variable(ref).data();
    }
    n->output()->inferTypeFrom(ref); // note: before t_ because of std::move(ref)
    n->t_(attr::value, std::move(ref));
  } else if(val.isInt()) {
    n->i_(attr::value, val.toInt());
    n->output()->setType(IntType::get());
  } else if(val.isDouble()) {
    n->f_(attr::value, val.toDouble());
    n->output()->setType(FloatType::get());
  } else if (val.isBool()) {
    n->i_(attr::value, val.toBool());
    n->output()->setType(BoolType::get());
  } else if (val.isBoolList()) {
    auto bool_list = val.toBoolList()->elements();
    n->is_(attr::value, std::vector<int64_t>(bool_list.begin(), bool_list.end()));
    n->output()->setType(ListType::ofBools());
  } else if(val.isIntList()) {
    n->is_(attr::value, val.toIntList()->elements());
    n->output()->setType(ListType::ofInts());
  } else if(val.isTensorList()) {
    n->ts_(attr::value, fmap(val.toTensorList()->elements(), [](const at::Tensor & t) {
      return autograd::Variable(t).data();
    }));
    n->output()->setType(ListType::ofTensors());
  } else if(val.isString()) {
    n->s_(attr::value, val.toString()->string());
    n->output()->setType(StringType::get());
  } else if(val.isNone()) {
    n->destroy();
    n = g.create(prim::None);
    n->output()->setType(NoneType::get());
  } else {
    throw constant_not_supported_error("Unsupported value kind: " + val.tagKind());
  }
  if(loc)
    n->setSourceLocation(std::make_shared<SourceRange>(*loc));
  if(scope)
    n->setScope(*scope);
  return g.insertNode(n)->output();
}

RegisterOperators reg({
  // Implementation of constant node, computes and IValue
  Operator(
      FunctionSchema(prim::Constant, {}, {}, /*vararg=*/false, /*varret=*/true),
      [](const Node* node) -> Operation {
        TypePtr type = node->output()->type();
        if(type->isSubtypeOf(DynamicType::get())) {
          auto t = autograd::make_variable(node->t(attr::value));
          return [t](Stack& stack) {
            stack.push_back(t);
            return 0;
          };
        } else if (type->isSubtypeOf(BoolType::get())) {
          bool b = node->i(attr::value);
          return [b](Stack& stack) {
            push(stack, b);
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
        } else if(type->isSubtypeOf(ListType::ofBools())) {
          auto bs = node->is(attr::value);
          return [bs](Stack& stack) {
            push(stack, bs);
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
          auto s = node->s(attr::value);
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

c10::optional<IValue> toIValue(const Value* v) {
  if(v->node()->kind() != prim::Constant)
    return c10::nullopt;
  // use implemenation of prim::Constant to compute the output IValue
  auto op = getOperation(v->node());
  Stack stack;
  op(stack);
  return stack.back();
}
}}
