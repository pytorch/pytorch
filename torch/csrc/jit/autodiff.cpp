#include "torch/csrc/jit/autodiff.h"

#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/symbolic_variable.h"
#include "torch/csrc/jit/operator.h"
#include "torch/csrc/utils/functional.h"

#include <torch/csrc/jit/assertions.h>

#include <algorithm>
#include <memory>

namespace torch { namespace jit {

using value_map = std::unordered_map<Value*, Value*>;
using value_set = std::unordered_set<Value*>;

void wrapDim(int64_t & dim, const std::vector<int64_t> & sizes) {
  if (dim < 0) {
    dim += sizes.size();
  }
}

bool isDifferentiable(Node * n) {
  // TODO: scalar-tensor ops should be canonicalized
  static OperatorSet differentiable_ops = {
    "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
    "aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor",
    "aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
    "aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor",
    "aten::mul(Tensor self, Tensor other) -> Tensor",
    "aten::mul(Tensor self, Scalar other) -> Tensor",
    "aten::div(Tensor self, Tensor other) -> Tensor",
    "aten::div(Tensor self, Scalar other) -> Tensor",
    "aten::sigmoid(Tensor self) -> Tensor",
    "aten::tanh(Tensor self) -> Tensor",
    "aten::relu(Tensor self) -> Tensor",
    "aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor",
    "aten::exp(Tensor self) -> Tensor",
    "aten::t(Tensor self) -> Tensor",
    "aten::neg(Tensor self) -> Tensor",
    "aten::clamp(Tensor self, Scalar? min, Scalar? max) -> Tensor",
    "aten::type_as(Tensor self, Tensor other) -> Tensor",
    "aten::unsqueeze(Tensor self, int dim) -> Tensor",
    "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta, Scalar alpha) -> Tensor",
    "aten::mm(Tensor self, Tensor mat2) -> Tensor",
    "aten::lt(Tensor self, Tensor other) -> Tensor",
    "aten::le(Tensor self, Tensor other) -> Tensor",
    "aten::gt(Tensor self, Tensor other) -> Tensor",
    "aten::ge(Tensor self, Tensor other) -> Tensor",
    "aten::eq(Tensor self, Tensor other) -> Tensor",
    "aten::ne(Tensor self, Tensor other) -> Tensor",
    "aten::abs(Tensor self) -> Tensor",
    "aten::acos(Tensor self) -> Tensor",
    "aten::asin(Tensor self) -> Tensor",
    "aten::atan(Tensor self) -> Tensor",
    "aten::ceil(Tensor self) -> Tensor",
    "aten::cos(Tensor self) -> Tensor",
    "aten::cosh(Tensor self) -> Tensor",
    "aten::exp(Tensor self) -> Tensor",
    "aten::expm1(Tensor self) -> Tensor",
    "aten::floor(Tensor self) -> Tensor",
    "aten::fmod(Tensor self, Scalar other) -> Tensor",
    "aten::frac(Tensor self) -> Tensor",
    "aten::log(Tensor self) -> Tensor",
    "aten::log10(Tensor self) -> Tensor",
    "aten::log1p(Tensor self) -> Tensor",
    "aten::log2(Tensor self) -> Tensor",
    "aten::reciprocal(Tensor self) -> Tensor",
    "aten::remainder(Tensor self, Scalar other) -> Tensor",
    "aten::round(Tensor self) -> Tensor",
    "aten::rsqrt(Tensor self) -> Tensor",
    "aten::sin(Tensor self) -> Tensor",
    "aten::sinh(Tensor self) -> Tensor",
    "aten::tan(Tensor self) -> Tensor",
    "aten::trunc(Tensor self) -> Tensor",
    "aten::log_softmax(Tensor self, int dim) -> Tensor",
    "aten::avg_pool2d(Tensor self, int[] kernel_size, int[] stride, int[] padding, bool ceil_mode, bool count_include_pad) -> Tensor",
    "aten::max_pool2d_with_indices(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> (Tensor, Tensor)"
  };

  // TODO: add support for the following fusible operators.
  // They're a little tricky to implement; max/min require mutability for best perf
  // "aten::atan2(Tensor self) -> Tensor",
  // "aten::max(Tensor self) -> Tensor",
  // "aten::min(Tensor self) -> Tensor"

  if (n->kind() == prim::Constant ||
      n->kind() == prim::AutogradAdd ||
      n->kind() == prim::ConstantChunk)
    return true;
  if (differentiable_ops.find(n))
    return true;

  if (n->matches("aten::expand(Tensor self, int[] size, *, bool implicit) -> Tensor")) {
    return n->get<std::vector<int64_t>>(attr::size) && n->is_constant(attr::implicit) &&
      n->namedInput(attr::self)->type()->cast<CompleteTensorType>();
  }
  if (n->matches("aten::view(Tensor self, int[] size) -> Tensor")) {
    return n->get<std::vector<int64_t>>(attr::size) &&
      n->namedInput(attr::self)->type()->cast<CompleteTensorType>();
  }

  // linear blocks may appear as inputs to graph executors, but they are removed
  // before differentiation occurs
  if (n->kind() == prim::GradOf) {
    auto body = n->blocks().at(0);
    return std::all_of(
        body->nodes().begin(),
        body->nodes().end(),
        static_cast<bool (*)(Node*)>(isDifferentiable));
  }

  return false;
}


bool isDifferentiable(Graph & g) {
  return std::all_of(g.nodes().begin(), g.nodes().end(),
                     static_cast<bool(*)(Node*)>(isDifferentiable));
}


static std::vector<Value*> gradientForNode(Node* node, ArrayRef<Value*> grad_values) {
  static const OperatorSet comparison_ops = {
    "aten::lt(Tensor self, Tensor other) -> Tensor",
    "aten::le(Tensor self, Tensor other) -> Tensor",
    "aten::gt(Tensor self, Tensor other) -> Tensor",
    "aten::ge(Tensor self, Tensor other) -> Tensor",
    "aten::eq(Tensor self, Tensor other) -> Tensor",
    "aten::ne(Tensor self, Tensor other) -> Tensor"
  };
  const auto build_sym_grad = [node](const std::vector<SymbolicVariable>& grads) -> std::vector<SymbolicVariable> {
    auto inputs = fmap<SymbolicVariable>(node->inputs());
    auto outputs = fmap<SymbolicVariable>(node->outputs());

    if (node->matches("aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor")) {
      return {grads.at(0), grads.at(0) * node->namedInput(attr::alpha), nullptr};

    } else if (node->matches("aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor")) {
      return {grads.at(0), nullptr, nullptr};

    } else if (node->kind() == prim::AutogradAdd) {
      return {grads.at(0), grads.at(0)};

    } else if (node->matches("aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor")) {
      return {grads.at(0), -grads.at(0) * node->namedInput(attr::alpha), nullptr};

    } else if (node->matches("aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor")) {
      return {grads.at(0), nullptr, nullptr};

    } else if (node->matches("aten::mul(Tensor self, Tensor other) -> Tensor")) {
      return {grads.at(0) * inputs.at(1), grads.at(0) * inputs.at(0)};

    } else if (node->matches("aten::mul(Tensor self, Scalar other) -> Tensor")) {
      return {grads.at(0) * inputs.at(1), nullptr};

    } else if (node->matches("aten::div(Tensor self, Tensor other) -> Tensor")) {
      return {grads.at(0) / inputs.at(1), -grads.at(0) * inputs.at(0) / (inputs.at(1) * inputs.at(1))};

    } else if (node->matches("aten::div(Tensor self, Scalar other) -> Tensor")) {
      return {grads.at(0) / inputs.at(1), nullptr};

    } else if (node->matches("aten::sigmoid(Tensor self) -> Tensor")) {
      // TODO: The order of operations matter in this case. This
      // works for ppc64le and x86_64. Need to look at why the
      // order matters.
      return {(1 - outputs.at(0)) * outputs.at(0) * grads.at(0)};

    } else if (node->matches("aten::tanh(Tensor self) -> Tensor")) {
      return {grads.at(0) * (1 - outputs.at(0) * outputs.at(0))};

    } else if (node->matches("aten::relu(Tensor self) -> Tensor")) {
      return {grads.at(0) * (outputs.at(0) > at::Scalar(0)).type_as(outputs.at(0))};

    } else if (node->matches("aten::clamp(Tensor self, Scalar? min, Scalar? max) -> Tensor")) {
      // handle the case that min/max is None
      Value* min = inputs.at(1);
      Value* max = inputs.at(2);
      if (!min->isNone() && !max->isNone()) {
        return {grads.at(0)
          * (1-(inputs.at(0) <= inputs.at(1)).type_as(inputs.at(0)))
          * (1-(inputs.at(0) >= inputs.at(2)).type_as(inputs.at(0))), nullptr, nullptr};
      } else if (max->isNone()) {
        return {grads.at(0)
          * (1-(inputs.at(0) <= inputs.at(1)).type_as(inputs.at(0))), nullptr, nullptr};
      } else if (min->isNone()) {
        return {grads.at(0)
          * (1-(inputs.at(0) >= inputs.at(2)).type_as(inputs.at(0))), nullptr, nullptr};
      } else {
        return {grads.at(0), nullptr, nullptr};
      }
    } else if (node->matches("aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor")) {
      auto threshold = node->get<at::Scalar>(attr::threshold).value();
      return {grads.at(0) * (inputs.at(0) > threshold).type_as(outputs.at(0)), nullptr, nullptr};

    } else if (node->matches("aten::exp(Tensor self) -> Tensor")) {
      return {grads.at(0) * (outputs.at(0))};

    } else if (node->matches("aten::t(Tensor self) -> Tensor")) {
      return {grads.at(0).t()};

    } else if (node->matches("aten::neg(Tensor self) -> Tensor")) {
      return {-grads.at(0)};

    } else if (node->matches("aten::abs(Tensor self) -> Tensor")) {
      return {grads.at(0) * inputs.at(0).sign()};

    } else if (node->matches("aten::acos(Tensor self) -> Tensor")) {
      return {grads.at(0) * -((-inputs.at(0) * inputs.at(0) + at::Scalar(1)).rsqrt())};

    } else if (node->matches("aten::asin(Tensor self) -> Tensor")) {
      return {grads.at(0) * (-inputs.at(0) * inputs.at(0) + at::Scalar(1)).rsqrt()};

    } else if (node->matches("aten::atan(Tensor self) -> Tensor")) {
      return {grads.at(0) / (inputs.at(0) * inputs.at(0) + at::Scalar(1))};

    } else if (node->matches("aten::ceil(Tensor self) -> Tensor")) {
      return {SymbolicVariable::zeros_like(grads.at(0))};

    } else if (node->matches("aten::cos(Tensor self) -> Tensor")) {
      return {grads.at(0) * -inputs.at(0).sin()};

    } else if (node->matches("aten::cosh(Tensor self) -> Tensor")) {
      return {grads.at(0) * inputs.at(0).sinh()};

    } else if (node->matches("aten::exp(Tensor self) -> Tensor")) {
      return {grads.at(0) * outputs.at(0)};

    } else if (node->matches("aten::expm1(Tensor self) -> Tensor")) {
      return {grads.at(0) * (outputs.at(0) + at::Scalar(1))};

    } else if (node->matches("aten::floor(Tensor self) -> Tensor")) {
      return {SymbolicVariable::zeros_like(grads.at(0))};

    } else if (node->matches("aten::fmod(Tensor self, Scalar other) -> Tensor")) {
      return {grads.at(0), nullptr};

    } else if (node->matches("aten::frac(Tensor self) -> Tensor")) {
      return {grads.at(0)};

    } else if (node->matches("aten::log(Tensor self) -> Tensor")) {
      return {grads.at(0) / inputs.at(0)};

    } else if (node->matches("aten::log10(Tensor self) -> Tensor")) {
      return {grads.at(0) / (inputs.at(0) * 2.3025850929940456)};

    } else if (node->matches("aten::log1p(Tensor self) -> Tensor")) {
      return {grads.at(0) / (inputs.at(0) + at::Scalar(1))};

    } else if (node->matches("aten::log2(Tensor self) -> Tensor")) {
      return {grads.at(0) / (inputs.at(0) * 0.6931471805599453)};

    } else if (node->matches("aten::reciprocal(Tensor self) -> Tensor")) {
      return {-grads.at(0) * outputs.at(0) * outputs.at(0)};

    } else if (node->matches("aten::remainder(Tensor self, Scalar other) -> Tensor")) {
      return {grads.at(0), nullptr};

    } else if (node->matches("aten::round(Tensor self) -> Tensor")) {
      return {SymbolicVariable::zeros_like(grads.at(0))};

    } else if (node->matches("aten::rsqrt(Tensor self) -> Tensor")) {
      return {grads.at(0) * outputs.at(0).pow(3.) * -0.5};

    } else if (node->matches("aten::sin(Tensor self) -> Tensor")) {
      return {grads.at(0) * inputs.at(0).cos()};

    } else if (node->matches("aten::sinh(Tensor self) -> Tensor")) {
      return {grads.at(0) * inputs.at(0).cosh()};

    } else if (node->matches("aten::tan(Tensor self) -> Tensor")) {
      return {grads.at(0) * (1. + outputs.at(0) * outputs.at(0))};

    } else if (node->matches("aten::trunc(Tensor self) -> Tensor")) {
      return {SymbolicVariable::zeros_like(grads.at(0))};

    } else if (node->kind() == prim::ConstantChunk) {
      return {SymbolicVariable::cat(grads, node->i(attr::dim))};

    } else if (node->matches("aten::view(Tensor self, int[] size) -> Tensor") ||
               node->matches("aten::reshape(Tensor self, int[] shape) -> Tensor")) {
      // TODO: if sizes are not available statically, add an operator that reutrns them as a tuple
      auto sizes = node->namedInput(attr::self)->type()->expect<CompleteTensorType>()->sizes();
      return {grads.at(0).reshape(sizes), nullptr};

    } else if (node->matches("aten::type_as(Tensor self, Tensor other) -> Tensor")) {
      return {grads.at(0).type_as(inputs.at(0)), nullptr};

    } else if (node->matches("aten::unsqueeze(Tensor self, int dim) -> Tensor")) {
      return {grads.at(0).squeeze(node->namedInput(attr::dim)), nullptr};

    } else if (node->matches("aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta, Scalar alpha) -> Tensor")) {
      return {grads.at(0) * node->namedInput(attr::beta),
              grads.at(0).mm(inputs.at(2).t()) * node->namedInput(attr::alpha),
              inputs.at(1).t().mm(grads.at(0)) * node->namedInput(attr::alpha),
              nullptr, nullptr};

    } else if (node->matches("aten::mm(Tensor self, Tensor mat2) -> Tensor")) {
      return {grads.at(0).mm(inputs.at(1).t()), inputs.at(0).t().mm(grads.at(0))};

    } else if (node->matches("aten::expand(Tensor self, int[] size, *, bool implicit) -> Tensor")) {
      const auto& input_sizes = inputs.at(0).sizes();
      if (input_sizes.size() == 0)
        return {grads.at(0).sum(), nullptr, nullptr};
      auto grad_sizes = node->get<std::vector<int64_t>>(attr::size).value();
      auto grad = grads.at(0);
      while (grad_sizes.size() > input_sizes.size()) {
        grad = grad.sum(0, false);
        grad_sizes.erase(grad_sizes.begin());
      }
      for (size_t i = 0; i < input_sizes.size(); ++i) {
        if (input_sizes[i] == 1 && grad_sizes[i] > 1) {
          grad = grad.sum(i, true);
        }
      }
      return {grad, nullptr, nullptr};

    } else if (node->matches("aten::squeeze(Tensor self) -> Tensor")) {
      const auto& sizes = inputs.at(0).sizes();
      std::vector<size_t> squeezed_dims;
      for (size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] != 1) continue;
        squeezed_dims.push_back(i);
      }
      SymbolicVariable returned_grad = grads.at(0);
      for (const auto& dim : squeezed_dims) {
        returned_grad = returned_grad.unsqueeze(dim);
      }
      return {returned_grad};

    } else if (node->matches("aten::squeeze(Tensor self, int dim) -> Tensor", /*const_inputs=*/attr::dim)) {
      int64_t dim = *node->get<int64_t>(attr::dim);
      const auto& sizes = inputs.at(0).sizes();
      wrapDim(dim, sizes);
      if (sizes.size() == 0)  {
        return {grads.at(0), nullptr};
      }
      return {sizes.at(dim) > 1 ? grads.at(0) : grads.at(0).unsqueeze(dim), nullptr};

    } else if (node->matches("aten::cat(Tensor[] tensors, int dim) -> Tensor", /*const_inputs=*/attr::dim)) {
      int dim = *node->get<int64_t>(attr::dim);
      auto tensor_inputs = inputs; tensor_inputs.pop_back();
      const auto& first_sizes = tensor_inputs.at(0).sizes();
      const auto has_first_sizes = [&first_sizes](SymbolicVariable var) {
        return var.sizes() == first_sizes;
      };

      // NB: this is a specialization for the common case where all inputs are
      // of equal sizes. We can use a single split operation to handle that.
      if (std::all_of(tensor_inputs.begin(), tensor_inputs.end(), has_first_sizes)) {
        auto tensor_grads = grads.at(0).chunk(tensor_inputs.size(), dim);
        tensor_grads.emplace_back(nullptr); // for attr::dim
        return tensor_grads;
      } else {
        size_t offset = 0;
        auto grad = grads.at(0);
        std::vector<SymbolicVariable> tensor_grads;
        for (auto input : tensor_inputs) {
          tensor_grads.push_back(grad.narrow(dim, offset, input.sizes()[dim]));
          offset += input.sizes()[dim];
        }
        tensor_grads.emplace_back(nullptr); // for attr::dim
        return tensor_grads;
      }
    } else if (comparison_ops.find(node)) {
      return {nullptr, nullptr};

    } else if (node->matches("aten::avg_pool2d(Tensor self, int[] kernel_size, int[] stride, int[] padding, bool ceil_mode, bool count_include_pad) -> Tensor")) {
      JIT_ASSERT(grads.size() == 1);
      auto graph = node->owningGraph();
      auto backward_value = graph->insert(aten::avg_pool2d_backward, {
        grads.at(0).value(),
        node->namedInput(attr::self),
        node->namedInput(attr::kernel_size),
        node->namedInput(attr::stride),
        node->namedInput(attr::padding),
        node->namedInput(attr::ceil_mode),
        node->namedInput(attr::count_include_pad)});
      return {backward_value->node()->output(0), nullptr, nullptr, nullptr, nullptr, nullptr};

    } else if (node->matches("aten::max_pool2d_with_indices(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> (Tensor, Tensor)")) {
      JIT_ASSERT(grads.size() == 2);
      auto graph = node->owningGraph();
      auto backward_value = graph->insert(aten::max_pool2d_with_indices_backward, {
        grads.at(0).value(),
        node->namedInput(attr::self),
        node->namedInput(attr::kernel_size),
        node->namedInput(attr::stride),
        node->namedInput(attr::padding),
        node->namedInput(attr::dilation),
        node->namedInput(attr::ceil_mode),
        outputs.at(1).value()
      });
      return {backward_value->node()->output(0), nullptr, nullptr, nullptr, nullptr, nullptr};

    } else if (node->matches("aten::log_softmax(Tensor self, int dim) -> Tensor")) {
      JIT_ASSERT(grads.size() == 1);
      auto graph = node->owningGraph();
      auto backward_value = graph->insert(aten::_log_softmax_backward_data, {
        grads.at(0).value(),
        outputs.at(0).value(),
        node->namedInput(attr::dim),
        node->namedInput(attr::self)
      });
      return {backward_value->node()->output(0), nullptr};

    } else if (node->kind() == prim::Constant) {
      return {};
    }
    throw std::runtime_error(std::string("failed to differentiate `") + node->kind().toDisplayString() + "`");
  };
  if (!isDifferentiable(node)) {
    throw std::runtime_error(std::string("differentiation of ") + node->kind().toDisplayString() + " "
                             "is not supported, or it is missing necessary type information");
  }
  auto sym_grads = build_sym_grad(fmap<SymbolicVariable>(grad_values));
  return fmap(sym_grads, [](const SymbolicVariable &v) { return v.value(); });
}

// If we have a function y = f(x) with jacobian J, the backwards of f is dx = J^t dy.
// Note that because the backwards always implements this matrix multiply,
// we know that it maps an input vector of zeros to an output vector of zero
// regardless of what operations it choses to do inside to actually implement
// the matrix multiply (most use some optimized form and never generate J^t).
// More generally, we know that all of the backward computations are linear and
// can use this property to do more aggressive optimizations later.
// It is ok to replace any backward function with known-zero inputs with something
// that produces known-zero outputs. This function encloses each know-linear
// backward function in a 'GradOf' sub-block so that we can perform optimizations
// using this information. In particular, specializeUndef will observe if
// all the inputs to the linear block are Undef, which the autograd uses to represent
// zeros, and then propagate the undefs to the outputs of the block.
static std::vector<Value*> linearGradientForNode(Node* node, ArrayRef<Value*> grad_values) {
  auto & graph = *node->owningGraph();
  auto linear = graph.insertNode(graph.create(prim::GradOf, {grad_values}, 0));
  // to make reading gradient graphs easier, remember the name of the forward op
  linear->s_(attr::name, node->kind().toDisplayString());
  auto block = linear->addBlock();
  WithInsertPoint guard(block);
  auto results = gradientForNode(node, grad_values);
  return fmap(results, [block, linear](Value *grad) -> Value* {
    if (!grad) return nullptr;
    block->registerOutput(grad);
    return linear->addOutput()->copyMetadata(grad);
  });
}

struct ReverseDetails {
  ReverseDetails(value_map&& grad_map, Block * reverse_block)
    : grad_map(std::move(grad_map))
    , reverse_block(reverse_block) {}

  value_map grad_map;
  Block * reverse_block;
};

// AutogradAdd is a special addition function that handles Undef
// AutogradAdd(a, b) == a + b if defined(a) and defined(b)
// AutogradAdd(Undef, b) == b
// AutogradAdd(a, Undef) == a
// AutogradAdd(Undef, Undef) == Undef
static Value* createAutogradAdd(Value* a, Value* b) {
  auto graph = a->owningGraph();
  return graph->insertNode(graph->create(prim::AutogradAdd, {a, b}))->output();
}

// Before:
//   - grad_desc has field f initialized to the original 0-stage graph
// After:
//   - the last node of f (f->nodes().reverse()[0]) is a gradient node
//     whose block has vjp inputs for all outputs that require_grad
//     and vjp outputs for all primal inputs that require_grad
//   - grad_desc has df_input_vjps and df_output_vjps set
//     (but df_input_vjps will be modified later as well)
static ReverseDetails addReverseInline(Gradient& grad_desc) {
  auto & graph = *grad_desc.f;
  // note: reverse_node is intentionally not inserted to avoid
  // accidentally acting on it (e.g. in elminate dead code),
  // std::cout << *reverse_node << to view its state.
  auto reverse_node = graph.create(prim::Reverse, 0);
  auto reverse_block = reverse_node->addBlock();
  WithInsertPoint guard(reverse_block);

  value_map grad_map; // x -> dx mapping
  const auto get_grad = [&](Value* v) -> Value* {
    auto it = grad_map.find(v);
    if (it == grad_map.end()) {
      auto undef = graph.insertNode(graph.createUndefined());
      std::tie(it, std::ignore) = grad_map.emplace(v, undef->output());
    }
    return it->second;
  };
  const auto set_grad = [&](Value *x, Value *dx) {
    if (Value * prev_grad = grad_map[x]) {
      grad_map[x] = createAutogradAdd(prev_grad, dx);
    } else {
      grad_map[x] = dx;
    }
  };

  auto outputs = graph.outputs();
  for (size_t i = 0, num_outputs = outputs.size(); i < num_outputs; ++i) {
    Value * output = outputs[i];
    if (!output->requires_grad())
      continue;
    Value * output_grad = reverse_block->addInput()->setType(output->type());
    set_grad(output, output_grad);
    grad_desc.df_input_vjps.push_back(i);
  }

  for (auto it = graph.nodes().rbegin(), end = graph.nodes().rend(); it != end; ++it) {
    Node *node = *it;
    auto inputs = node->inputs();
    auto outputs = node->outputs();
    if (std::all_of(outputs.begin(), outputs.end(), [](Value *v) { return !v->requires_grad(); })) {
      continue;
    }

    value_list grad_inputs = linearGradientForNode(node, fmap(node->outputs(), get_grad));
    JIT_ASSERT(grad_inputs.size() == node->inputs().size());
    for (size_t i = 0, num_inputs = grad_inputs.size(); i < num_inputs; ++i) {
      if (!inputs[i]->requires_grad()) continue;
      // NB: Not returning a gradient w.r.t. a value that requires grad is normal if the
      // input is non-differentiable. This happens e.g. in the aten::type_as case.
      if (!grad_inputs[i]) continue;
      set_grad(inputs[i], grad_inputs[i]);
    }
  }

  auto inputs = graph.inputs();
  for (size_t i = 0, num_inputs = inputs.size(); i < num_inputs; ++i) {
    Value * input = inputs[i];
    if (!input->requires_grad())
      continue;
    // NB: Not having a gradient defined w.r.t. an input to the graph which requires grad
    // can happen and is not an error. It might have been used only in non-differentiable
    // contexts (e.g. as second input to aten::type_as). In that case we simply ignore it
    // as an output, because it won't ever produce any meaningful values.
    if (grad_map.count(input) == 0) continue;
    reverse_block->registerOutput(get_grad(input));
    grad_desc.df_output_vjps.push_back(i);
  }
  return ReverseDetails(std::move(grad_map), reverse_block);
}

// Any temporary value from the primal graphs needs to be captured for later use in the
// reverse graph, to avoid costly recomputations. However, a lot of the nodes we have
// in our graphs are simply constants, which are cheap to execute and replicate, and so
// it's better to just copy them into the reverse graph, without polluting the output
// lists unnecessarily.
static void liftConstants(Gradient& grad_desc, ReverseDetails& rev_info) {
  static const auto err = [](Value*) -> Value* {
    throw std::runtime_error("unexpected input");
  };
  auto & graph = *grad_desc.f;
  Block* reverse_block = rev_info.reverse_block;

  for (Node *top_node : reverse_block->nodes()) {
    JIT_ASSERT(top_node->kind() == prim::GradOf ||
               top_node->kind() == prim::AutogradAdd ||
               top_node->kind() == prim::Undefined);
    if (top_node->kind() != prim::GradOf) continue;
    Block * grad_body = top_node->blocks().at(0);
    for (Node *node : grad_body->nodes()) {
      for (Value * input : node->inputs()) {
        if (input->node()->kind() != prim::Constant) continue;
        if (input->node()->owningBlock() == grad_body) continue;
        Node *lifted_constant = graph.createClone(input->node(), err);
        reverse_block->prependNode(lifted_constant);
        node->replaceInputWith(input, lifted_constant->output());
      }
    }
  }

  // It's possible the we've cloned the same constants many times,
  // so we use CSE to deduplicate them.
  EliminateCommonSubexpression(reverse_block);
}

// Takes a grad_desc.f returned from `addReverseInline` and splits off the
// reverse_block into its own graph, storing it in df.
// All intermediates needed in the second stage are added to
// outputs of f, and taken as inputs in df. For a more
// detailed description see Note [Gradient graphs] in autodiff.h.
// This function also initializes the fields in grad_desc that were undefined after
// `addReverseInline` (and extends `df_input_vjps` with vjps for captured temporaries).
static void lambdaLiftReverse(Gradient& grad_desc, ReverseDetails& rev_info) {
  auto & graph = *grad_desc.f;
  auto primal_block = graph.block();
  auto reverse_block = rev_info.reverse_block;

  // --------------------------------------------------------------------------
  // 1. Find values of f that need to be captured.
  // --------------------------------------------------------------------------
  // First, we need to find all values that are produced in f,
  // and used in df. They will need to be added as inputs of the df
  // and some of them may also need to be appended as outputs of f if
  // they are not already an input or an output of f
  value_set reverse_captures_set;
  value_list reverse_captures; // Invariant: topo sorted
  auto check_uses = [&](Value *v) {
    for (auto use : v->uses()) {
      if (use.user->owningBlock() == primal_block)
        continue;
      if (/* bool unseen = */ reverse_captures_set.emplace(v).second) {
        reverse_captures.push_back(v);
      }
    }
  };
  for (Value * input : graph.inputs()) {
    check_uses(input);
  }
  for (Node * node : graph.nodes()) {
    for (Value * output : node->outputs())
      check_uses(output);
  }

  // --------------------------------------------------------------------------
  // 2. Prepare input/outputs lists for f and df
  // --------------------------------------------------------------------------
  // It's simple to construct primal_inputs/reverse_outputs,
  // but primal_outputs/reverse_inputs are much more subtle.
  // Here's a summary of how they are supposed to look like:
  //
  // Primal outputs:
  //   [original outputs], [temporaries]
  //
  // Reverse inputs:
  //   [output vjps (aka grad_outputs)], [temporary vjps]
  //   [captured primal values, in topological order],

  // -- Construct primal_outputs, df_input_captures, f_real_outputs ----
  grad_desc.f_real_outputs = graph.outputs().size();

  std::unordered_map<Value*, size_t> orig_primal_outputs_idx;
  std::unordered_map<Value*, size_t> orig_primal_inputs_idx;
  // NOTE: we use emplace to avoid replacing an existing index if an output is repeated
  for (size_t i = 0, num_outputs = graph.outputs().size(); i < num_outputs; ++i)
    orig_primal_outputs_idx.emplace(graph.outputs()[i], i);
  for (size_t i = 0, num_inputs = graph.inputs().size(); i < num_inputs; ++i)
    orig_primal_inputs_idx[graph.inputs()[i]] = i;

  // NB: reverse_captures are already deduplicated, and in topo order
  for (Value * capture_val : reverse_captures) {
    // If it's already an output we don't have to add anything,
    // but register the fact that it needs to be captured.
    if (orig_primal_outputs_idx.count(capture_val) > 0) {
      grad_desc.df_input_captured_outputs.push_back(orig_primal_outputs_idx[capture_val]);
    // If it's an input, we could add it as an output but in fact it's
    // more efficient to use a special kind of capture.
    } else if (orig_primal_inputs_idx.count(capture_val) > 0) {
      grad_desc.df_input_captured_inputs.push_back(orig_primal_inputs_idx.at(capture_val));
    // Otherwise it's just a regular intermediate value that we need to add as an output
    } else {
      // we need to create a new temporary output for this capture because it wasn't availiable.
      graph.registerOutput(capture_val);
      grad_desc.df_input_captured_outputs.emplace_back(graph.outputs().size() - 1);
    }
  }

  // -- Add VJPs for temporaries, adjust df_input_vjps -------------------------
  // NB [possible optimization]: use the newly added vjp input as soon as the first
  // vjp for that value is generated, to reduce the lifespan of this input
  // (currently we add it to the final vjp after all adds).
  for (size_t i = grad_desc.f_real_outputs; i < graph.outputs().size(); ++i) {
    Value * tmp = graph.outputs().at(i);
    // Add VJP inputs only for intermediates that actually required grad.
    if (!tmp->requires_grad()) continue;
    Value * tmp_vjp_in = reverse_block->addInput()->setType(tmp->type());
    Value * tmp_vjp_prev = rev_info.grad_map.at(tmp);
    // This is quite weird because we can't first make a sum and then replace all uses
    // of tmp_vjp_prev (that would replace its use in the sum too!), so we create an
    // incorrect sum that doesn't use prev vjp, replace uses, and fix the sum.
    Value * new_vjp = createAutogradAdd(tmp_vjp_in, tmp_vjp_in);
    new_vjp->node()->moveAfter(tmp_vjp_prev->node());
    tmp_vjp_prev->replaceAllUsesWith(new_vjp);
    new_vjp->node()->replaceInput(1, tmp_vjp_prev);
    grad_desc.df_input_vjps.emplace_back(i);
  }

  // add the captures as formal arguments to the reverse_block
  // afterward inputs: [output vjps][temporary vjps][captures]
  // construct a map from captured 'value' to the index in the input list
  // used to extract this block into its own function
  std::unordered_map<Value*, size_t> capture_to_formal_index;
  const auto & add_capture = [&](Value * captured) {
    capture_to_formal_index[captured] = reverse_block->inputs().size();
    reverse_block->addInput()->copyMetadata(captured);
  };
  for(auto & offset : grad_desc.df_input_captured_inputs)
    add_capture(graph.inputs()[offset]);
  for(auto & offset : grad_desc.df_input_captured_outputs)
    add_capture(graph.outputs()[offset]);

  grad_desc.df = std::make_shared<Graph>();
  grad_desc.df->block()->cloneFrom(reverse_block, [&](Value* v) {
    return grad_desc.df->inputs()[capture_to_formal_index.at(v)];
  });
  // reverse_node was just to hold onto reverse_block in a debuggable way
  // we can remove it now.
  reverse_block->owningNode()->destroy();
}

Gradient differentiate(std::shared_ptr<Graph>& graph) {
  Gradient grad_desc;
  // Take ownership of the graph
  JIT_ASSERTM(graph.use_count() == 1,
              "differentiate will mutate and destroy the graph, so it requires "
              "graph.use_count() == 1, but found %d", graph.use_count());
  std::swap(graph, grad_desc.f);
  // XXX: Take care when handling outputs - they can be duplicated!

  WithInsertPoint guard(grad_desc.f->block());
  // Fills in df_input_vjps and df_output_vjps
  auto rev_info = addReverseInline(grad_desc);
  // Lift constants captured for the reverse graph into it
  liftConstants(grad_desc, rev_info);
  // addReverseInline has to call gradientForNode if *any* of the outputs
  // require grad, but it will emit vjps for *all* outputs. Use DCE to remove
  // unnecessary nodes.
  EliminateDeadCode(rev_info.reverse_block);
  // Fills in f, df, f_real_outputs, df_input_captures,
  // modifies df_input_vjps (new vjps are added for temporaries)
  lambdaLiftReverse(grad_desc, rev_info);
  return grad_desc;
}

}}
