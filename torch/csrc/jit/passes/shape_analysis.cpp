#include <torch/csrc/jit/passes/shape_analysis.h>

#include <torch/csrc/jit/argument_spec.h>
#include <torch/csrc/jit/assertions.h>
#include <torch/csrc/jit/constants.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/passes/alias_analysis.h>

#include <torch/csrc/autograd/variable.h>

#include <ATen/DeviceGuard.h>
#include <ATen/ExpandUtils.h>

#include <exception>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace torch {
namespace jit {

struct propagation_error : std::exception {};

#define SHAPE_ASSERT(cond) \
  if (!(cond))             \
  throw propagation_error()

namespace {

bool isValidArgumentForRunning(Value* v) {
  // allow constants
  if (toIValue(v))
    return true;
  if (CompleteTensorTypePtr tt = v->type()->cast<CompleteTensorType>()) {
    return !at::isIntegralType(tt->scalarType());
  }
  return v->type()->isSubtypeOf(FloatType::get());
}

bool isValidReturnForRunning(Value* v) {
  return v->type()->isSubtypeOf(DynamicType::get()) ||
      v->type()->isSubtypeOf(NumberType::get());
}

class ShapePropagator {
 public:
  explicit ShapePropagator(std::shared_ptr<Graph> graph)
      : aliasDb_(AliasAnalysis(std::move(graph))) {}

  void PropagateShapeOnBlock(Block* block, bool insert_expands = true) {
    for (Node* node : block->nodes()) {
      try {
        PropagateShapeOnNode(node, insert_expands);
      } catch (propagation_error& e) {
        setUnshapedType(node);
      } catch (std::exception& e) {
        if (auto sl = node->getSourceLocation()) {
          sl->wrapAndRethrowException(e, "operation failed shape propagation");
        } else {
          throw;
        }
      }
    }
  }

 private:
  const AliasDb aliasDb_;

  void setUnshapedType(Node* node) {
    for (auto o : node->outputs()) {
      o->setType(unshapedType(o->type()));
    }
  }

  int64_t wrapDim(int64_t dim, at::IntList sizes) {
    if (dim < 0) {
      dim += sizes.size();
    }
    return dim;
  }

  // TODO: Would be better to make JIT not assume that CUDA devices
  // are the only thing that exist.
  static at::Device jitDeviceIndexToDevice(int device) {
    return device == -1 ? at::kCPU : at::Device(at::kCUDA, device);
  }

  IValue representativeValue(Value* v) {
    TypePtr type_ = v->type();
    // if the value is actually constant, just use it!
    if (auto iv = toIValue(v)) {
      return *iv;
    }
    if (CompleteTensorTypePtr type = type_->cast<CompleteTensorType>()) {
      auto backend =
          type->device().is_cpu() ? at::Backend::CPU : at::Backend::CUDA;
      at::DeviceGuard device_guard(type->device());
      auto& attype = at::getNonVariableType(backend, type->scalarType());
      auto t =
          at::empty_strided(type->sizes(), type->strides(), attype.options())
              .zero_();
      return autograd::make_variable(t, /*requires_grad=*/false);
    } else if (type_->isSubtypeOf(FloatType::get())) {
      return 0.f;
    }
    // we should not get here because isValidArgumentForRunning should have
    // prevented it
    std::stringstream ss;
    ss << "unable to create representative value for: " << type_->str()
       << ". File a bug report.";
    throw std::runtime_error(ss.str());
  }

  // for each node in the schema with type Tensor, extract the T type
  // returns c10::nullopt if any Tensor in the schema does not have a known
  // shape ignores non-tensor in the list of inputs
  template <typename T>
  c10::optional<std::vector<std::shared_ptr<T>>> gatherTensorTypes(Node* node) {
    std::vector<std::shared_ptr<T>> tensor_types;

    auto& schema = node->schema();
    auto& args = schema.arguments();
    // can't handle varargs primitives because we don't know what should be a
    // Tensor
    if (schema.is_vararg()) {
      return c10::nullopt;
    }
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i].type()->isSubtypeOf(ListType::ofTensors())) {
        return c10::nullopt;
      } else if (args[i].type()->isSubtypeOf(DynamicType::get())) {
        if (auto type = node->input(i)->type()->cast<T>()) {
          tensor_types.push_back(type);
        } else {
          return c10::nullopt;
        }
      } else /* non-tensor type */ {
        continue;
      }
    }

    return tensor_types;
  }

  bool mergeTypes(
      ArrayRef<Value*> lhs,
      ArrayRef<Value*> rhs,
      ArrayRef<Value*> outputs) {
    JIT_ASSERT(lhs.size() == rhs.size() && rhs.size() == outputs.size());
    bool changed = false;
    for (size_t i = 0; i < lhs.size(); ++i) {
      auto old_output_type = outputs[i]->type();
      auto new_type = unifyTypes(lhs[i]->type(), rhs[i]->type());
      JIT_ASSERT(new_type);
      outputs[i]->setType(*new_type);
      if (*old_output_type != *outputs[i]->type())
        changed = true;
    }
    return changed;
  }

  void broadcastBinary(
      Node* node,
      std::vector<CompleteTensorTypePtr>& types,
      size_t idx1,
      size_t idx2) {
    auto expected_size =
        at::infer_size(types[idx1]->sizes(), types[idx2]->sizes());
    auto broadcast = [&](size_t input_idx) {
      CompleteTensorTypePtr input_type = types.at(input_idx);
      if (input_type->sizes() == expected_size)
        return;
      auto graph = node->owningGraph();
      WithInsertPoint point_guard{node};
      Node* expand = graph
                         ->create(
                             aten::expand,
                             {node->inputs().at(input_idx),
                              graph->insertConstant(expected_size),
                              graph->insertConstant(false)})
                         ->insertBefore(node);
      PropagateShapeOnNode(expand);
      node->replaceInput(input_idx, expand->output());
    };
    broadcast(idx1);
    broadcast(idx2);
    types[0] = node->inputs().at(idx1)->type()->expect<CompleteTensorType>();
    types[1] = node->inputs().at(idx2)->type()->expect<CompleteTensorType>();
  }

  OperatorSet cannot_propagate_shape_by_running_it = {
      "aten::gesv(Tensor self, Tensor A) -> (Tensor, Tensor)",
      "aten::inverse(Tensor self) -> Tensor",
  };

  // Check if this node depends on a value that has been mutated previously. If
  // it has, then it's not safe to run this node in isolation, since we don't
  // know whether the dependency has been executed.
  std::unordered_map<Node*, bool> dependsOnMutationMemo_;
  bool dependsOnMutation(Node* node) {
    if (dependsOnMutationMemo_.count(node) != 0) {
      return dependsOnMutationMemo_[node];
    }

    if (aliasDb_.hasWritersBefore(node)) {
      // If something could have written to a value used by this node, we can't
      // guarantee the result is the same when running it in isolation.
      dependsOnMutationMemo_[node] = true;
      return true;
    }

    // recursively check the producers of its inputs. We need to do this if the
    // mutable value has been laundered through a pure function:
    //   a += 1
    //   c = a + b
    //   d = c + 1
    // In this case, `d` cares whether `a` has been mutated even though it's not
    // a direct input.
    auto depends = false;
    for (auto input : node->inputs()) {
      depends |= dependsOnMutation(input->node());
    }

    dependsOnMutationMemo_[node] = depends;
    return depends;
  }

  bool canPropagateShapeByRunningIt(Node* node) {
    if (cannot_propagate_shape_by_running_it.find(node)) {
      return false;
    }

    if (dependsOnMutation(node)) {
      return false;
    }

    bool valid_args = std::all_of(
        node->inputs().begin(),
        node->inputs().end(),
        isValidArgumentForRunning);
    if (!valid_args)
      return false;

    bool valid_returns = std::all_of(
        node->outputs().begin(),
        node->outputs().end(),
        isValidReturnForRunning);
    if (!valid_returns)
      return false;

    return true;
  }

  bool PropagateShapeOnNodeByRunningIt(Node* node) {
    if (!canPropagateShapeByRunningIt(node))
      return false;
    auto op = getOperation(node);
    Stack stack;

    for (auto input : node->inputs()) {
      stack.push_back(representativeValue(input));
    }

    // XXX: we're not catching any exceptions from the op for now. This
    // is to uncover any mistakes we could make when editing this code,
    // and eventually it shouldn't matter, because this phase should be
    // preceded by schema checking.
    op(stack);

    JIT_ASSERT(stack.size() == node->outputs().size());
    for (size_t i = 0; i < stack.size(); ++i) {
      // some ops may have mixed tensor/primitive outputs
      // for primitives, we don't need to change the type because it is already
      // its most constrained form.
      if (stack[i].isTensor())
        node->outputs()[i]->inferTypeFrom(stack[i].toTensor());
    }
    return true;
  }

  void PropagateCatShape(Node* cat_node) {
    static const auto propagate_complete =
        [this](Node* node, at::ArrayRef<Value*> tensors) -> bool {
      auto input_types = fmap(tensors, [](Value* v) {
        return v->type()->cast<CompleteTensorType>();
      });
      if (!std::all_of(
              input_types.begin(),
              input_types.end(),
              [](const CompleteTensorTypePtr& tp) { return tp != nullptr; })) {
        return false;
      }
      if (!node->is_constant(attr::dim))
        return false;
      std::vector<int64_t> sizes = input_types[0]->sizes();
      const int64_t dim = wrapDim(node->get<int64_t>(attr::dim).value(), sizes);
      const int64_t ndim = sizes.size();

      if (dim < 0 || dim >= ndim)
        return false;

      sizes[dim] = 0;
      for (auto& tp : input_types) {
        auto& tp_sizes = tp->sizes();
        if (sizes.size() != tp_sizes.size())
          return false;
        for (int64_t i = 0; i < ndim; ++i) {
          if (sizes[i] != tp_sizes[i] && i != dim) {
            return false;
          }
        }
        sizes[dim] += tp_sizes[dim];
      }
      node->output()->setType(input_types[0]->withSizes(sizes));
      return true;
    };
    static const auto propagate = [](Node* node,
                                     at::ArrayRef<Value*> tensors) -> bool {
      for (Value* v : tensors) {
        if (auto type = v->type()->cast<TensorType>()) {
          node->output()->setType(type);
          return true;
        }
      }
      return false;
    };
    auto list_node =
        ((cat_node->kind() == prim::FusedConcat)
             ? cat_node
             : cat_node->namedInput(attr::tensors)->node());
    if (list_node->kind() == prim::ListConstruct ||
        cat_node->kind() == prim::FusedConcat) {
      auto tensors = list_node->inputs();
      if (!tensors.empty()) {
        if (propagate_complete(cat_node, tensors)) {
          return;
        } else if (propagate(cat_node, tensors)) {
          return;
        }
      }
    }
    setUnshapedType(cat_node);
  }

  void PropagateShapeOnNode(Node* node, bool insert_expands = true) {
    // These don't require the types, and have complicated schema. Return early
    // after we process them.
    switch (node->kind()) {
      case prim::If: {
        auto then_block = node->blocks().at(0);
        auto else_block = node->blocks().at(1);
        PropagateShapeOnBlock(then_block);
        PropagateShapeOnBlock(else_block);
        mergeTypes(
            then_block->outputs(), else_block->outputs(), node->outputs());
        return;
      }
      case prim::Loop: {
        auto body_block = node->blocks().at(0);
        // propagate counter type
        body_block->inputs().at(0)->setType(node->inputs().at(0)->type());
        // propagate loop-carried input types to block inputs
        auto loop_carried_inputs = node->inputs().slice(2); // skip max, cond
        auto loop_carried_block = body_block->inputs().slice(1); // skip trip
        for (size_t i = 0; i < loop_carried_inputs.size(); ++i) {
          loop_carried_block[i]->setType(loop_carried_inputs[i]->type());
        }
        auto loop_carried_outputs = body_block->outputs().slice(1); // skip cond

        do {
          PropagateShapeOnBlock(body_block, /*insert_expands=*/false);
          // note: inserting expands is unsafe at this point, we don't know
          // if the types are stable yet, so the arguments to expand may change
        } while (mergeTypes(
            loop_carried_block, loop_carried_outputs, loop_carried_block));

        // now that the types are stable, we can insert the expands
        PropagateShapeOnBlock(body_block, /*insert_expands=*/true);

        for (size_t i = 0; i < loop_carried_inputs.size(); ++i) {
          node->outputs()[i]->setType(loop_carried_block[i]->type());
        }
        return;
      }
      case prim::ImplicitTensorToNum:
      case prim::Bool:
      case prim::Int:
      case prim::Float:
        return; // correct num type is already set
      case prim::NumToTensor: {
        TypePtr typ = node->input()->type();
        if (typ->isSubtypeOf(IntType::get()) ||
            typ->isSubtypeOf(BoolType::get())) {
          node->output()->setType(TensorType::create(at::kLong, at::kCPU, 0));
        } else if (node->input()->type()->isSubtypeOf(FloatType::get())) {
          node->output()->setType(TensorType::create(at::kDouble, at::kCPU, 0));
        }
        return;
      }
      case prim::TupleConstruct: {
        // We refresh the tuple type, because the input types could have been
        // refined.
        node->output()->setType(TupleType::create(
            fmap(node->inputs(), [](Value* v) { return v->type(); })));
        return;
      }
      case prim::TupleUnpack: {
        auto tuple_type = node->input()->type()->cast<TupleType>();
        JIT_ASSERT(
            tuple_type &&
            tuple_type->elements().size() == node->outputs().size());
        auto elems = tuple_type->elements();
        for (size_t i = 0; i < node->outputs().size(); ++i) {
          node->output(i)->setType(elems[i]);
        }
        return;
      }
      case prim::Constant: {
        if (node->output()->type()->isSubtypeOf(DynamicType::get())) {
          node->output()->inferTypeFrom(node->t(attr::value));
        }
        return;
      }
      case prim::ConstantChunk: {
        Value* tensor = node->input();
        if (auto type = tensor->type()->cast<TensorType>()) {
          for (Value* output : node->outputs()) {
            output->setType(type);
          }
        } else {
          setUnshapedType(node);
        }
        return;
      }
      case prim::Undefined: {
        setUnshapedType(node);
        return;
      }
      default:
        break; // fall-through
    }

    if (node->hasSideEffects()) {
      return;
    }

    if (node->matches("aten::cat(Tensor[] tensors, int dim) -> Tensor") ||
        node->kind() == prim::FusedConcat) {
      return PropagateCatShape(node);
    }

    if (auto maybe_complete_types =
            gatherTensorTypes<CompleteTensorType>(node)) {
      if (PropagateCompleteShapeOnNode(
              node, insert_expands, std::move(*maybe_complete_types))) {
        return;
      }
    }

    if (PropagateTensorShapeOnNode(node, insert_expands)) {
      return;
    }

    if (PropagateShapeOnNodeByRunningIt(node)) {
      return;
    }
    return setUnshapedType(node);
  }

  static c10::optional<size_t> determineListSize(Value* list) {
    JIT_ASSERT(list->type()->cast<ListType>());
    if (auto shape = constant_as<std::vector<int64_t>>(list)) {
      return shape->size();
    }
    auto input_node = list->node();
    if (input_node->kind() == prim::ListConstruct) {
      return input_node->inputs().size();
    }
    return c10::nullopt;
  }

  // is it ok to try to run the op
  // If an input is a constant, then we assume that the input is valid
  // and we can try to run it.
  // Otherwise:
  // Integral typed _inputs_ are often an indicator that we're indexing into
  // a tensor, so we should special-case these ops in the shape propagation.
  // Additionally, passing in a zero representative tensor into an integer
  // division op causes divide-by-zero errors
  // _Outputs_ must be tensors or primtives
  // We will call inferTypeFrom on the tensors, and ignore the primitives.
  // However, we allow primitive returns because we want to support mixed
  // primitive/tensor outputs.

  bool PropagateTensorShapeOnNode(Node* node, bool insert_expands) {
    static const auto broadcast = [](std::vector<TensorTypePtr>& tensor_types,
                                     size_t arg_for_type) -> TensorTypePtr {
      if (tensor_types.size() == 1) {
        return tensor_types[0];
      }
      JIT_ASSERT(!tensor_types.empty());
      auto any_type = tensor_types[arg_for_type];
      auto max_dims = any_type->dim();
      for (auto& type : tensor_types) {
        max_dims = std::max(max_dims, type->dim());
      }
      return TensorType::create(
          any_type->scalarType(), any_type->device(), max_dims);
    };

    using type_vec_t = std::vector<TensorTypePtr>;
    // Formula is expected to return a vector of length equal to the number of
    // tensor outputs of the node, or an empty vector which implies that it
    // failed to propagate.
    using formula_t = std::function<type_vec_t(Node*)>;
    static std::mutex shape_formulas_mutex;
    static std::vector<std::pair<OperatorSet, formula_t>> shape_formulas;
    struct register_formula_for {
      register_formula_for(OperatorSet operators, formula_t formula) {
        std::unique_lock<std::mutex> lock{shape_formulas_mutex};
        shape_formulas.emplace_back(std::move(operators), std::move(formula));
      }
    };

    // Requirements:
    //   dims           : preserved
    //   scalar type    : preserved
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    static const register_formula_for simple_unary_ops{
        {
            "aten::abs(Tensor self) -> Tensor",
            "aten::acos(Tensor self) -> Tensor",
            "aten::neg(Tensor self) -> Tensor",
            "aten::t(Tensor self) -> Tensor",
            "aten::sigmoid(Tensor self) -> Tensor",
            "aten::tanh(Tensor self) -> Tensor",
            "aten::relu(Tensor self) -> Tensor",
            "aten::asin(Tensor self) -> Tensor",
            "aten::atan(Tensor self) -> Tensor",
            "aten::ceil(Tensor self) -> Tensor",
            "aten::clone(Tensor self) -> Tensor",
            "aten::contiguous(Tensor self) -> Tensor",
            "aten::bernoulli(Tensor self, *, Generator? generator) -> Tensor",
            "aten::celu(Tensor self, Scalar alpha) -> Tensor",
            "aten::clamp(Tensor self, Scalar? min, Scalar? max) -> Tensor",
            "aten::clamp_max(Tensor self, Scalar max) -> Tensor",
            "aten::clamp_min(Tensor self, Scalar min) -> Tensor",
            "aten::alpha_dropout(Tensor input, float p, bool train) -> Tensor",
            "aten::bernoulli(Tensor self, float p, *, Generator? generator) -> Tensor",
            "aten::cos(Tensor self) -> Tensor",
            "aten::cosh(Tensor self) -> Tensor",
            "aten::digamma(Tensor self) -> Tensor",
            "aten::dropout(Tensor input, float p, bool train) -> Tensor",
            "aten::elu(Tensor self, Scalar alpha, Scalar scale, Scalar input_scale) -> Tensor",
            "aten::erf(Tensor self) -> Tensor",
            "aten::erfc(Tensor self) -> Tensor",
            "aten::erfinv(Tensor self) -> Tensor",
            "aten::exp(Tensor self) -> Tensor",
            "aten::expm1(Tensor self) -> Tensor",
            "aten::log(Tensor self) -> Tensor",
            "aten::log10(Tensor self) -> Tensor",
            "aten::log1p(Tensor self) -> Tensor",
            "aten::log2(Tensor self) -> Tensor",
            "aten::log_sigmoid(Tensor self) -> Tensor",
            "aten::log_softmax(Tensor self, int dim) -> Tensor",
            "aten::floor(Tensor self) -> Tensor",
            "aten::frac(Tensor self) -> Tensor",
            "aten::flip(Tensor self, int[] dims) -> Tensor",
            "aten::feature_alpha_dropout(Tensor input, float p, bool train) -> Tensor",
            "aten::feature_dropout(Tensor input, float p, bool train) -> Tensor",
            "aten::hardshrink(Tensor self, Scalar lambd) -> Tensor",
            "aten::hardtanh(Tensor self, Scalar min_val, Scalar max_val) -> Tensor",
            "aten::glu(Tensor self, int dim) -> Tensor",
            "aten::inverse(Tensor self) -> Tensor",
            "aten::leaky_relu(Tensor self, Scalar negative_slope) -> Tensor",
            "aten::lgamma(Tensor self) -> Tensor",
            "aten::mvlgamma(Tensor self, int p) -> Tensor",
            "aten::normal(float mean, Tensor std, *, Generator? generator) -> Tensor",
            "aten::normal(Tensor mean, float std, *, Generator? generator) -> Tensor",
            "aten::permute(Tensor self, int[] dims) -> Tensor",
            "aten::pin_memory(Tensor self) -> Tensor",
            "aten::pinverse(Tensor self, float rcond) -> Tensor",
            "aten::reciprocal(Tensor self) -> Tensor",
            "aten::relu(Tensor self) -> Tensor",
            "aten::round(Tensor self) -> Tensor",
            "aten::rrelu(Tensor self, Scalar lower, Scalar upper, bool training, Generator? generator) -> Tensor",
            "aten::rsqrt(Tensor self) -> Tensor",
            "aten::selu(Tensor self) -> Tensor",
            "aten::sigmoid(Tensor self) -> Tensor",
            "aten::sign(Tensor self) -> Tensor",
            "aten::sin(Tensor self) -> Tensor",
            "aten::sinh(Tensor self) -> Tensor",
            "aten::softmax(Tensor self, int dim) -> Tensor",
            "aten::softplus(Tensor self, Scalar beta, Scalar threshold) -> Tensor",
            "aten::softshrink(Tensor self, Scalar lambd) -> Tensor",
            "aten::sqrt(Tensor self) -> Tensor",
            "aten::tan(Tensor self) -> Tensor",
            "aten::tanh(Tensor self) -> Tensor",
            "aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor",
            "aten::transpose(Tensor self, int dim0, int dim1) -> Tensor",
            "aten::tril(Tensor self, int diagonal) -> Tensor",
            "aten::triu(Tensor self, int diagonal) -> Tensor",
            "aten::trunc(Tensor self) -> Tensor",
            "aten::rot90(Tensor self, int k, int[] dims) -> Tensor",
            "aten::narrow(Tensor self, int dim, int start, int length) -> Tensor",
            "aten::slice(Tensor self, int dim, int start, int end, int step) -> Tensor",
            "aten::alias(Tensor self) -> Tensor",
            "aten::detach(Tensor self) -> Tensor",
            "aten::cumprod(Tensor self, int dim) -> Tensor",
            "aten::cumsum(Tensor self, int dim) -> Tensor",

            "aten::empty_like(Tensor self) -> Tensor",
            "aten::full_like(Tensor self, Scalar fill_value) -> Tensor",
            "aten::ones_like(Tensor self) -> Tensor",
            "aten::rand_like(Tensor self) -> Tensor",
            "aten::randint_like(Tensor self, int high) -> Tensor",
            "aten::randint_like(Tensor self, int low, int high) -> Tensor",
            "aten::randn_like(Tensor self) -> Tensor",
            "aten::zeros_like(Tensor self) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          auto input_type = node->input(0)->type()->cast<TensorType>();
          return input_type ? type_vec_t{input_type} : type_vec_t{};
        }};

    // Requirements:
    //   dims           : broadcast all tensor args
    //   scalar type    : always matching and preserved
    //   device         : always matching and preserved
    //   tensor inputs  : *
    //   tensor outputs : 1
    static const register_formula_for broadcasting_ops{
        {
            // Tensor-Tensor operators
            "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
            "aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
            "aten::mul(Tensor self, Tensor other) -> Tensor",
            "aten::div(Tensor self, Tensor other) -> Tensor",
            "aten::pow(Tensor self, Tensor exponent) -> Tensor",
            "aten::fmod(Tensor self, Tensor other) -> Tensor",
            "aten::remainder(Tensor self, Tensor other) -> Tensor",
            "aten::lerp(Tensor self, Tensor end, Scalar weight) -> Tensor",
            "aten::max(Tensor self, Tensor other) -> Tensor",
            "aten::min(Tensor self, Tensor other) -> Tensor",
            "aten::__and__(Tensor self, Tensor other) -> Tensor",
            "aten::__or__(Tensor self, Tensor other) -> Tensor",
            "aten::__xor__(Tensor self, Tensor other) -> Tensor",
            "aten::__lshift__(Tensor self, Tensor other) -> Tensor",
            "aten::__rshift__(Tensor self, Tensor other) -> Tensor",
            "aten::__iand__(Tensor self, Tensor other) -> Tensor",
            "aten::__ior__(Tensor self, Tensor other) -> Tensor",
            "aten::__ixor__(Tensor self, Tensor other) -> Tensor",
            "aten::__ilshift__(Tensor self, Tensor other) -> Tensor",
            "aten::__irshift__(Tensor self, Tensor other) -> Tensor",

            // Tensor-Scalar operators
            "aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor",
            "aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor",
            "aten::mul(Tensor self, Scalar other) -> Tensor",
            "aten::div(Tensor self, Scalar other) -> Tensor",
            "aten::pow(Tensor self, Scalar exponent) -> Tensor",
            "aten::fmod(Tensor self, Scalar other) -> Tensor",
            "aten::remainder(Tensor self, Scalar other) -> Tensor",
            "aten::pow(Scalar self, Tensor exponent) -> Tensor",
            "aten::__and__(Tensor self, Scalar other) -> Tensor",
            "aten::__or__(Tensor self, Scalar other) -> Tensor",
            "aten::__xor__(Tensor self, Scalar other) -> Tensor",
            "aten::__lshift__(Tensor self, Scalar other) -> Tensor",
            "aten::__rshift__(Tensor self, Scalar other) -> Tensor",
            "aten::__iand__(Tensor self, Scalar other) -> Tensor",
            "aten::__ior__(Tensor self, Scalar other) -> Tensor",
            "aten::__ixor__(Tensor self, Scalar other) -> Tensor",
            "aten::__ilshift__(Tensor self, Scalar other) -> Tensor",
            "aten::__irshift__(Tensor self, Scalar other) -> Tensor",

            // Ops with Tensor-Tensor overloads only
            "aten::atan2(Tensor self, Tensor other) -> Tensor",

            // Non-binary ops
            "aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value) -> Tensor",
            "aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value) -> Tensor",
        },
        [this](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes<TensorType>(node)) {
            return {broadcast(*maybe_tensor_types, 0)};
          }
          return {};
        }};

    // aten::where is special in that its return type is the second argument's
    // (self) type rather than the that of condition
    static const register_formula_for where_op{
        {
            "aten::where(Tensor condition, Tensor self, Tensor other) -> Tensor",
        },
        [this](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes<TensorType>(node)) {
            return {broadcast(*maybe_tensor_types, 1)};
          }
          return {};
        }};

    static const auto any_tensor_type = [](Node* node) -> TensorTypePtr {
      for (Value* input : node->inputs()) {
        if (auto type = input->type()->cast<TensorType>()) {
          return type;
        }
      }
      return nullptr;
    };

    // Requirements:
    //   dims           : always matching and preserved
    //   scalar type    : always matching and preserved
    //   device         : always matching and preserved
    //   tensor inputs  : 2
    //   tensor outputs : 1
    static const register_formula_for binary_ops_strict_match{
        {
            "aten::normal(Tensor mean, Tensor std, *, Generator? generator) -> Tensor",
            "aten::mm(Tensor self, Tensor mat2) -> Tensor",
            "aten::bmm(Tensor self, Tensor mat2) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto type = any_tensor_type(node)) {
            return {type};
          }
          return {};
        }};

    // Requirements:
    //   dims           : all tensor args are broadcast
    //   scalar type    : byte/uint8
    //   device         : always matching and preserved
    //   tensor inputs  : *
    //   tensor outputs : 1
    static const register_formula_for comparison_ops{
        {
            "aten::lt(Tensor self, Tensor other) -> Tensor",
            "aten::le(Tensor self, Tensor other) -> Tensor",
            "aten::gt(Tensor self, Tensor other) -> Tensor",
            "aten::ge(Tensor self, Tensor other) -> Tensor",
            "aten::eq(Tensor self, Tensor other) -> Tensor",
            "aten::ne(Tensor self, Tensor other) -> Tensor",
            "aten::lt(Tensor self, Scalar other) -> Tensor",
            "aten::le(Tensor self, Scalar other) -> Tensor",
            "aten::gt(Tensor self, Scalar other) -> Tensor",
            "aten::ge(Tensor self, Scalar other) -> Tensor",
            "aten::eq(Tensor self, Scalar other) -> Tensor",
            "aten::ne(Tensor self, Scalar other) -> Tensor",
        },
        [this](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes<TensorType>(node)) {
            return {broadcast(*maybe_tensor_types, 0)->toScalarType(at::kByte)};
          }
          return {};
        }};

    // Requirements:
    //   dims           : preserved from the first argument
    //   scalar type    : preserved from the first argument (doesn't have to
    //   match other arguments) device         : always matching and preserved
    //   tensor inputs  : *
    //   tensor outputs : 1
    // NB: those ops (with slight adjustments) are good candidates for restarts.
    //     Knowing the type and device of weights or biases is usually enough to
    //     infer the output type.
    static const register_formula_for nn_ops_first_input_preserving{
        {
            "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor",
            "aten::conv1d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor",
            "aten::conv2d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor",
            "aten::conv3d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor",
            "aten::conv_tbc(Tensor self, Tensor weight, Tensor bias, int pad) -> Tensor",
            "aten::conv_transpose1d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int groups, int[] dilation) -> Tensor",
            "aten::conv_transpose2d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int groups, int[] dilation) -> Tensor",
            "aten::conv_transpose3d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int groups, int[] dilation) -> Tensor",
            "aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor",
            "aten::adaptive_avg_pool1d(Tensor self, int[] output_size) -> Tensor",
            "aten::adaptive_avg_pool2d(Tensor self, int[] output_size) -> Tensor",
            "aten::adaptive_avg_pool3d(Tensor self, int[] output_size) -> Tensor",
            "aten::avg_pool1d(Tensor self, int[] kernel_size, int[] stride, int[] padding, bool ceil_mode, bool count_include_pad) -> Tensor",
            "aten::avg_pool2d(Tensor self, int[] kernel_size, int[] stride, int[] padding, bool ceil_mode, bool count_include_pad) -> Tensor",
            "aten::avg_pool3d(Tensor self, int[] kernel_size, int[] stride, int[] padding, bool ceil_mode, bool count_include_pad) -> Tensor",
            "aten::max_pool1d(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> Tensor",
            "aten::max_pool2d(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> Tensor",
            "aten::max_pool3d(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> Tensor",
            "aten::max_unpool2d(Tensor self, Tensor indices, int[] output_size) -> Tensor",
            "aten::max_unpool3d(Tensor self, Tensor indices, int[] output_size, int[] stride, int[] padding) -> Tensor",
            "aten::reflection_pad1d(Tensor self, int[] padding) -> Tensor",
            "aten::reflection_pad2d(Tensor self, int[] padding) -> Tensor",
            "aten::replication_pad1d(Tensor self, int[] padding) -> Tensor",
            "aten::replication_pad2d(Tensor self, int[] padding) -> Tensor",
            "aten::replication_pad3d(Tensor self, int[] padding) -> Tensor",
            "aten::upsample_bilinear2d(Tensor self, int[] output_size, bool align_corners) -> Tensor",
            "aten::upsample_linear1d(Tensor self, int[] output_size, bool align_corners) -> Tensor",
            "aten::upsample_nearest1d(Tensor self, int[] output_size) -> Tensor",
            "aten::upsample_nearest2d(Tensor self, int[] output_size) -> Tensor",
            "aten::upsample_nearest3d(Tensor self, int[] output_size) -> Tensor",
            "aten::upsample_trilinear3d(Tensor self, int[] output_size, bool align_corners) -> Tensor",
            "aten::prelu(Tensor self, Tensor weight) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            return {type};
          }
          return {};
        }};

    // Requirements:
    //   dims           : 0
    //   scalar type    : preserved
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    static const register_formula_for all_reduce_ops{
        {
            "aten::argmax(Tensor self) -> Tensor",
            "aten::argmin(Tensor self) -> Tensor",
            "aten::det(Tensor self) -> Tensor",
            "aten::logdet(Tensor self) -> Tensor",
            "aten::max(Tensor self) -> Tensor",
            "aten::min(Tensor self) -> Tensor",
            "aten::mean(Tensor self) -> Tensor",
            "aten::median(Tensor self) -> Tensor",
            "aten::norm(Tensor self, Scalar p) -> Tensor",
            "aten::std(Tensor self, bool unbiased) -> Tensor",
            "aten::sum(Tensor self) -> Tensor",
            "aten::trace(Tensor self) -> Tensor",
            "aten::var(Tensor self, bool unbiased) -> Tensor",
            "aten::all(Tensor self) -> Tensor",
            "aten::any(Tensor self) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            return {type->withDim(0)};
          }
          return {};
        }};

    // Requirements:
    //   dims           : 0
    //   scalar type    : preserved if floating point, otherwise long/int64
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    static const register_formula_for all_reduce_ops_with_integer_upcast{
        {
            "aten::sum(Tensor self) -> Tensor",
            "aten::prod(Tensor self) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            return {at::isFloatingType(type->scalarType())
                        ? type->withDim(0)
                        : type->withDim(0)->toScalarType(at::kLong)};
          }
          return {};
        }};

    static const auto multidim_reduce_with_postprocess =
        [](Node* node,
           size_t num_reduced_dim,
           bool upcast_integer) -> type_vec_t {
      auto maybe_keepdim = node->get<bool>(attr::keepdim);
      if (!maybe_keepdim)
        return {};
      if (auto type = node->input(0)->type()->cast<TensorType>()) {
        if (upcast_integer && !at::isFloatingType(type->scalarType())) {
          type = type->toScalarType(at::kLong);
        }
        if (*maybe_keepdim) {
          return {type};
        } else if (type->dim() > num_reduced_dim) {
          return {type->withDim(type->dim() - num_reduced_dim)};
        }
      }
      return {};
    };

    // Requirements:
    //   dims           : preserved if keepdim == false, 1 smaller otherwise
    //   scalar type    : preserved for first output, byte/uint8 for second
    //   output if exists device         : preserved tensor inputs  : 1 tensor
    //   outputs : 1 or 2
    // Additionally:
    //   - First input should be the only tensor input
    //   - Has a bool keepdim argument
    static const register_formula_for dim_reduce_ops{
        {
            "aten::argmax(Tensor self, int dim, bool keepdim) -> Tensor",
            "aten::argmin(Tensor self, int dim, bool keepdim) -> Tensor",
            "aten::max_values(Tensor self, int dim, bool keepdim) -> Tensor",
            "aten::min_values(Tensor self, int dim, bool keepdim) -> Tensor",
            "aten::norm(Tensor self, Scalar? p, int dim, bool keepdim) -> Tensor",
            "aten::var(Tensor self, int dim, bool unbiased, bool keepdim) -> Tensor",
            "aten::logsumexp(Tensor self, int dim, bool keepdim) -> Tensor",
            "aten::all(Tensor self, int dim, bool keepdim) -> Tensor",
            "aten::any(Tensor self, int dim, bool keepdim) -> Tensor",

            // Ops returning indices as second output
            "aten::kthvalue(Tensor self, int k, int dim, bool keepdim) -> (Tensor, Tensor)",
            "aten::max(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)",
            "aten::min(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)",
            "aten::median(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)",
            "aten::mode(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)",
        },
        [](Node* node) -> type_vec_t {
          // NB: Note that while this function is generally meant to be used
          // with ops that have a single output, we will fix up its return right
          // below.
          auto output_types = multidim_reduce_with_postprocess(
              node, /*num_reduce_dim=*/1, /*integer_upcast=*/false);
          if (!output_types.empty() && node->outputs().size() == 2) {
            output_types.push_back(
                output_types.back()->toScalarType(at::kLong));
          }
          return output_types;
        }};

    // Requirements:
    //   dims           : preserved if keepdim == false, 1 smaller otherwise
    //   scalar type    : preserved if floating point, otherwise long/int64
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    //   - has a bool keepdim argument
    static const register_formula_for dim_reduce_ops_with_integer_upcast{
        {
            "aten::prod(Tensor self, int dim, bool keepdim) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          return multidim_reduce_with_postprocess(
              node, /*num_reduce_dim=*/1, /*integer_upcast=*/true);
        }};

    // Requirements:
    //   dims           : preserved if keepdim == false, dim->size() smaller
    //   otherwise scalar type    : preserved device         : preserved tensor
    //   inputs  : 1 tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    //   - has a bool keepdim argument
    static const register_formula_for multidim_reduce_ops{
        {
            "aten::mean(Tensor self, int[] dim, bool keepdim) -> Tensor",
            "aten::std(Tensor self, int[] dim, bool unbiased, bool keepdim) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto dim = node->get<std::vector<int64_t>>(attr::dim)) {
            return multidim_reduce_with_postprocess(
                node, /*num_reduce_dim=*/dim->size(), /*integer_upcast=*/false);
          }
          return {};
        }};

    // Requirements:
    //   dims           : preserved if keepdim == false, 1 smaller otherwise
    //   scalar type    : preserved if floating point, otherwise long/int64
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - has bool keepdim and int[] dim arguments
    static const register_formula_for multidim_reduce_ops_with_integer_upcast{
        {
            "aten::sum(Tensor self, int[] dim, bool keepdim) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto dim = node->get<std::vector<int64_t>>(attr::dim)) {
            // TODO: can dim contain duplicates?
            return multidim_reduce_with_postprocess(
                node, /*num_reduce_dim=*/dim->size(), /*integer_upcast=*/true);
          }
          return {};
        }};

    static const auto factory_with_ndim = [](Node* node,
                                             int dim) -> type_vec_t {
      auto maybe_layout = node->get<at::Layout>(attr::layout);
      if (!maybe_layout || maybe_layout != at::kStrided)
        return {};
      auto maybe_device = node->get<at::Device>(attr::device);
      if (!maybe_device)
        return {};
      auto maybe_scalar_type = node->get<at::ScalarType>(attr::dtype);
      if (!maybe_scalar_type)
        return {};
      return {TensorType::create(*maybe_scalar_type, *maybe_device, dim)};
    };

    // Requirements:
    //   dims           : preserved
    //   scalar type    : equal to value of dtype
    //   device         : equal to value of device
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - has ScalarType dtype, Layeout layout and Device device arguments
    static const register_formula_for like_factories_with_options{
        {
            "aten::empty_like(Tensor self, *, int dtype, int layout, Device device) -> Tensor",
            "aten::full_like(Tensor self, Scalar fill_value, *, int dtype, int layout, Device device) -> Tensor",
            "aten::ones_like(Tensor self, *, int dtype, int layout, Device device) -> Tensor",
            "aten::rand_like(Tensor self, *, int dtype, int layout, Device device) -> Tensor",
            "aten::randint_like(Tensor self, int high, *, int dtype, int layout, Device device) -> Tensor",
            "aten::randint_like(Tensor self, int low, int high, *, int dtype, int layout, Device device) -> Tensor",
            "aten::randn_like(Tensor self, *, int dtype, int layout, Device device) -> Tensor",
            "aten::zeros_like(Tensor self, *, int dtype, int layout, Device device) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto type =
                  node->namedInput(attr::self)->type()->cast<TensorType>()) {
            return factory_with_ndim(node, type->dim());
          }
          return {};
        }};

    // Requirements:
    //   dims           : equal to number of elements in size
    //   scalar type    : equal to value of dtype
    //   device         : equal to value of device
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - has int[] size, ScalarType dtype, Layeout layout and Device device
    //   arguments
    static const register_formula_for size_factories_with_options{
        {
            "aten::empty(int[] size, *, int dtype, int layout, Device device) -> Tensor",
            "aten::full(int[] size, Scalar fill_value, *, int dtype, int layout, Device device) -> Tensor",
            "aten::ones(int[] size, *, int dtype, int layout, Device device) -> Tensor",
            "aten::rand(int[] size, *, int dtype, int layout, Device device) -> Tensor",
            "aten::randn(int[] size, *, int dtype, int layout, Device device) -> Tensor",
            "aten::zeros(int[] size, *, int dtype, int layout, Device device) -> Tensor",
            "aten::randint(int high, int[] size, *, int dtype, int layout, Device device) -> Tensor",
            "aten::randint(int low, int high, int[] size, *, int dtype, int layout, Device device) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto maybe_size = node->get<std::vector<int64_t>>(attr::size)) {
            return factory_with_ndim(node, maybe_size->size());
          }
          return {};
        }};

    static const auto get_cast_scalar_type = [](Node* node) -> at::ScalarType {
      switch (node->kind()) {
        case aten::_cast_Byte:
          return at::kByte;
        case aten::_cast_Char:
          return at::kChar;
        case aten::_cast_Double:
          return at::kDouble;
        case aten::_cast_Float:
          return at::kFloat;
        case aten::_cast_Half:
          return at::kHalf;
        case aten::_cast_Int:
          return at::kInt;
        case aten::_cast_Long:
          return at::kLong;
        case aten::_cast_Short:
          return at::kShort;
        default:
          AT_ASSERTM(
              false,
              "unknown node kind in get_cast_scalar_type: ",
              node->kind().toQualString());
      }
    };
    static const register_formula_for cast_ops{
        {
            "aten::_cast_Byte(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Char(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Double(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Float(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Half(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Int(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Long(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Short(Tensor self, bool non_blocking) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto type =
                  node->namedInput(attr::self)->type()->cast<TensorType>()) {
            return {type->toScalarType(get_cast_scalar_type(node))};
          }
          return {};
        }};

    // First, try to match one of the registered formulas to their operator
    // sets.
    for (auto& entry : shape_formulas) {
      if (entry.first.find(node)) {
        auto types = entry.second(node);
        if (types.empty()) {
          return false;
        } else {
          auto outputs = node->outputs();
          JIT_ASSERT(types.size() == outputs.size());
          for (size_t i = 0; i < types.size(); ++i) {
            JIT_ASSERT(outputs[i]->type()->isSubtypeOf(DynamicType::get()));
            outputs[i]->setType(types[i]);
          }
          return true;
        }
      }
    }

    // This section implements shape prop for an assorted set of nodes that only
    // need partial information about their input types.
    const auto input_type = [node](size_t index) {
      return node->input(index)->type()->cast<TensorType>();
    };
    if (node->matches(
            "aten::masked_select(Tensor self, Tensor mask) -> Tensor")) {
      auto type = input_type(0);
      auto mask_type = input_type(1);
      if (type && mask_type) {
        if (type->dim() == 0 && mask_type->dim() == 0) {
          node->output()->setType(type->withDim(0));
        } else {
          node->output()->setType(type->withDim(1));
        }
        return true;
      }
      if (auto type = input_type(0)) {
        node->output()->setType(type->withDim(1));
        return true;
      }
    } else if (node->matches(
                   "aten::dot(Tensor self, Tensor tensor) -> Tensor")) {
      if (auto type = any_tensor_type(node)) {
        node->output()->setType(type->withDim(0));
        return true;
      }
    } else if (
        node->matches("aten::mv(Tensor self, Tensor vec) -> Tensor") ||
        node->matches(
            "aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta, Scalar alpha) -> Tensor")) {
      if (auto type = any_tensor_type(node)) {
        node->output()->setType(type->withDim(1));
        return true;
      }
    } else if (
        node->matches(
            "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta, Scalar alpha) -> Tensor") ||
        node->matches(
            "aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta, Scalar alpha) -> Tensor") ||
        node->matches(
            "aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta, Scalar alpha) -> Tensor")) {
      if (auto type = any_tensor_type(node)) {
        node->output()->setType(type->withDim(2));
        return true;
      }
    } else if (
        node->matches(
            "aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta, Scalar alpha) -> Tensor")) {
      if (auto type = any_tensor_type(node)) {
        node->output()->setType(type->withDim(3));
        return true;
      }
    } else if (
        node->matches(
            "aten::index_select(Tensor self, int dim, Tensor index) -> Tensor")) {
      auto type = input_type(0);
      auto index_type = input_type(1);
      // index_select behaves very weirdly when self.dim() == 0. It allows both
      // 0D and 1D indices, and returns a value that has as many dimensions as
      // index.
      if (type && index_type) {
        if (type->dim() == 0) {
          node->output()->setType(type->withDim(index_type->dim()));
        } else {
          node->output()->setType(type);
        }
        return true;
      }
    } else if (
        node->matches(
            "aten::gather(Tensor self, int dim, Tensor index) -> Tensor")) {
      auto type = input_type(0);
      auto index_type = input_type(1);
      // Gather has this annoying edge case where index always needs to match
      // the number of dims of self, **except** when self is 1D and index is 0D
      // in which case we return a 0D output.
      if (type && index_type) {
        if (index_type->dim() == 0) {
          node->output()->setType(type->withDim(0));
        } else {
          node->output()->setType(type);
        }
        return true;
      }
    } else if (
        node->matches(
            "aten::embedding(Tensor weight, Tensor indices, int padding_idx, bool scale_grad_by_freq, bool sparse) -> Tensor")) {
      auto weight_type = input_type(0);
      auto indices_type = input_type(1);
      if (weight_type && indices_type) {
        node->output()->setType(weight_type->withDim(indices_type->dim() + 1));
        return true;
      }
    } else if (
        node->matches(
            "aten::bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias) -> Tensor")) {
      if (auto type = input_type(0)) {
        node->output()->setType(type);
        return true;
      }
      if (auto type = input_type(1)) {
        node->output()->setType(type);
        return true;
      }
    } else if (
        node->matches(
            "aten::dist(Tensor self, Tensor other, Scalar p) -> Tensor")) {
      if (auto type = any_tensor_type(node)) {
        node->output()->setType(type->withDim(0));
        return true;
      }
    }

    // The code below implements formulas that need type information for all
    // their tensor inputs, and have exactly one output.
    std::vector<TensorTypePtr> tensor_types;
    static const auto reshape_prop =
        [](Node* node,
           Symbol shape_input,
           const std::vector<TensorTypePtr>& tensor_types) -> TensorTypePtr {
      if (auto list_size = determineListSize(node->namedInput(shape_input))) {
        return tensor_types.at(0)->withDim(*list_size);
      }
      return nullptr;
    };
    const auto getSingleOutputType = [&]() -> TypePtr {
      if (node->matches("aten::type_as(Tensor self, Tensor other) -> Tensor")) {
        return tensor_types.at(0)->toScalarType(
            tensor_types.at(1)->scalarType());
      } else if (
          node->matches("aten::view_as(Tensor self, Tensor other) -> Tensor") ||
          node->matches(
              "aten::expand_as(Tensor self, Tensor other) -> Tensor") ||
          node->matches(
              "aten::reshape_as(Tensor self, Tensor other) -> Tensor")) {
        return tensor_types.at(0)->withDim(tensor_types.at(1)->dim());
      } else if (
          node->matches("aten::view(Tensor self, int[] size) -> Tensor") ||
          node->matches(
              "aten::expand(Tensor self, int[] size, *, bool implicit) -> Tensor") ||
          node->matches(
              "aten::as_strided(Tensor self, int[] size, int[] stride, int? storage_offset) -> Tensor")) {
        return reshape_prop(node, attr::size, tensor_types);
      } else if (node->matches(
                     "aten::reshape(Tensor self, int[] shape) -> Tensor")) {
        return reshape_prop(node, attr::shape, tensor_types);
      } else if (node->matches(
                     "aten::repeat(Tensor self, int[] repeats) -> Tensor")) {
        return reshape_prop(node, attr::repeats, tensor_types);
      } else if (node->matches(
                     "aten::unsqueeze(Tensor self, int dim) -> Tensor")) {
        auto& t = tensor_types.at(0);
        return t->withDim(t->dim() + 1);
      } else if (
          node->matches(
              "aten::select(Tensor self, int dim, int index) -> Tensor") ||
          node->matches(
              "aten::diagonal(Tensor self, int offset, int dim1, int dim2) -> Tensor")) {
        auto& t = tensor_types.at(0);
        return t->dim() > 0 ? t->withDim(t->dim() - 1) : nullptr;
      } else if (node->matches(
                     "aten::matmul(Tensor self, Tensor other) -> Tensor")) {
        int dim1 = tensor_types.at(0)->dim();
        int dim2 = tensor_types.at(1)->dim();
        if (dim1 == 1 && dim2 == 1) {
          // Dot product
          return tensor_types.at(0)->withDim(0);
        } else if (dim1 == 2 && dim2 == 2) {
          // Matrix multiply
          return tensor_types.at(0);
        } else if (dim1 == 1 && dim2 == 2) {
          // Unsqueeze + matrix multiply + squeeze
          return tensor_types.at(0);
        } else if (dim1 == 2 && dim2 == 1) {
          // Matrix vector multiply
          return tensor_types.at(1);
        } else {
          // Batched matrix multiply (possibly with squeeze + unsqueeze if one
          // argument is 1D)
          auto type = broadcast(tensor_types, 0);
          if (tensor_types.at(0)->dim() == 1 ||
              tensor_types.at(1)->dim() == 1) {
            type = type->withDim(type->dim() - 1);
          }
          return type;
        }
      } else if (node->matches("aten::nonzero(Tensor self) -> Tensor")) {
        return tensor_types.at(0)->toScalarType(at::kLong);
      } else if (node->matches(
                     "aten::take(Tensor self, Tensor index) -> Tensor")) {
        return tensor_types.at(1)->toScalarType(
            tensor_types.at(0)->scalarType());
      } else if (node->matches(
                     "aten::diagflat(Tensor self, int offset) -> Tensor")) {
        return tensor_types.at(0)->withDim(2);
      } else if (node->matches(
                     "aten::diag(Tensor self, int diagonal) -> Tensor")) {
        auto& t = tensor_types.at(0);
        if (t->dim() == 1) {
          return t->withDim(2);
        } else if (t->dim() == 2) {
          return t->withDim(1);
        } else {
          return nullptr;
        }
      } else if (
          node->matches(
              "aten::unfold(Tensor self, int dimension, int size, int step) -> Tensor")) {
        auto& t = tensor_types.at(0);
        return t->dim() == 0 ? t : t->withDim(t->dim() + 1);
      } else if (node->matches(
                     "aten::polygamma(int n, Tensor self) -> Tensor")) {
        return tensor_types.at(0);
      }
      return nullptr;
    };
    if (auto maybe_tensor_types = gatherTensorTypes<TensorType>(node)) {
      tensor_types = std::move(*maybe_tensor_types);
    } else {
      return false;
    }
    if (node->outputs().size() == 1) {
      if (auto type = getSingleOutputType()) {
        node->output()->setType(type);
        return true;
      }
    }
    return false;
  }

  bool PropagateCompleteShapeOnNode(
      Node* node,
      bool insert_expands,
      std::vector<CompleteTensorTypePtr> tensor_types) {
    // For expensive ops we can directly encode their shape propagation
    // here, otherwise we fallback to running a fake version of the op
    // to get a quick and dirty propagation.
    if (node->matches(
            "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor") ||
        node->matches(
            "aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor") ||
        node->matches("aten::mul(Tensor self, Tensor other) -> Tensor")) {
      // These nodes and "div" handle tensors of different shapes internally,
      // so there's no need to insert explicit expand nodes. Note that "div" is
      // handled by the fallthrough because it's not always safe to run it due
      // to integer divide-by-zero.
      return PropagateShapeOnNodeByRunningIt(node);
    } else if (
        node->matches(
            "aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor") ||
        node->matches(
            "aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor") ||
        node->matches("aten::mul(Tensor self, Scalar other) -> Tensor") ||
        node->matches("aten::pow(Tensor self, Scalar exponent) -> Tensor")) {
      node->output()->setType(tensor_types.at(0));
      return true;
    } else if (
        insert_expands &&
        (node->matches("aten::pow(Tensor self, Tensor exponent) -> Tensor") ||
         node->matches("aten::min(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::max(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::lt(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::le(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::gt(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::ge(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::eq(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::ne(Tensor self, Tensor other) -> Tensor"))) {
      // Binary broadcasting ops
      // NB: we don't handle the nodes in any other way (note the lack of
      // return!), because the type casting logic in scalar cases is
      // non-trivial. It's better to just run them.
      broadcastBinary(node, tensor_types, 0, 1);
      return PropagateShapeOnNodeByRunningIt(node);
    } else if (
        node->matches("aten::neg(Tensor self) -> Tensor") ||
        node->matches("aten::sigmoid(Tensor self) -> Tensor") ||
        node->matches("aten::tanh(Tensor self) -> Tensor")) {
      node->output()->setType(tensor_types.at(0)->contiguous());
      return true;
    } else if (node->matches("aten::mm(Tensor self, Tensor mat2) -> Tensor")) {
      auto lhs_type = tensor_types.at(0);
      auto rhs_type = tensor_types.at(1);
      SHAPE_ASSERT(
          lhs_type->sizes().size() == 2 && rhs_type->sizes().size() == 2);
      node->output()->setType(CompleteTensorType::create(
          lhs_type->scalarType(),
          lhs_type->device(),
          at::IntList{lhs_type->sizes().at(0), rhs_type->sizes().at(1)}));
      return true;
    } else if (node->matches("aten::t(Tensor self) -> Tensor")) {
      auto tp = tensor_types.at(0);
      auto sizes = tp->sizes();
      auto strides = tp->strides();
      SHAPE_ASSERT(sizes.size() == 2);
      std::swap(sizes.at(0), sizes.at(1));
      std::swap(strides.at(0), strides.at(1));
      node->output()->setType(tp->withSizesStrides(sizes, strides));
      return true;
    } else if (
        node->matches(
            "aten::narrow(Tensor self, int dim, int start, int length) -> Tensor",
            /*const_inputs=*/{attr::dim, attr::length})) {
      auto tp = tensor_types.at(0);
      auto sizes = tp->sizes();
      int64_t dim = node->get<int64_t>(attr::dim).value();
      int64_t length = node->get<int64_t>(attr::length).value();
      SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) < sizes.size());
      sizes.at(dim) = length;
      node->output()->setType(tp->withSizesStrides(sizes, tp->strides()));
      return true;
    } else if (node->matches("aten::sum(Tensor self) -> Tensor")) {
      node->output()->setType(tensor_types.at(0)->withSizes({}));
      return true;
    } else if (node->matches(
                   "aten::sum(Tensor self, int[] dim, bool keepdim) -> Tensor",
                   /*const_inputs=*/{attr::dim, attr::keepdim})) {
      auto& tp = tensor_types.at(0);
      auto sizes = tp->sizes();
      auto dims = node->get<std::vector<int64_t>>(attr::dim).value();
      bool keepdim = node->get<bool>(attr::keepdim).value();
      std::reverse(dims.begin(), dims.end());
      for (int64_t dim : dims) {
        SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) < sizes.size());
        if (keepdim) {
          sizes.at(dim) = 1;
        } else {
          sizes.erase(sizes.begin() + dim);
        }
      }
      node->output()->setType(tp->withSizes(sizes));
      return true;
    } else if (node->matches(
                   "aten::squeeze(Tensor self, int dim) -> Tensor",
                   /*const_inputs=*/attr::dim)) {
      auto& tp = tensor_types.at(0);
      auto sizes = tp->sizes();
      auto strides = tp->strides();
      int64_t dim = wrapDim(node->get<int64_t>(attr::dim).value(), sizes);
      SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) < sizes.size());
      if (sizes.at(dim) == 1) {
        sizes.erase(sizes.begin() + dim);
        strides.erase(strides.begin() + dim);
      }
      node->output()->setType(tp->withSizesStrides(sizes, strides));
      return true;
    } else if (node->matches(
                   "aten::unsqueeze(Tensor self, int dim) -> Tensor",
                   /*const_inputs=*/attr::dim)) {
      auto& tp = tensor_types.at(0);
      auto sizes = tp->sizes();
      auto strides = tp->strides();
      int64_t dim = wrapDim(node->get<int64_t>(attr::dim).value(), sizes);
      SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) <= sizes.size());
      int64_t new_stride = dim >= static_cast<int64_t>(sizes.size())
          ? 1
          : sizes.at(dim) * strides.at(dim);
      sizes.insert(sizes.begin() + dim, 1);
      strides.insert(strides.begin() + dim, new_stride);
      node->output()->setType(tp->withSizesStrides(sizes, strides));
      return true;
    } else if (node->matches(
                   "aten::view(Tensor self, int[] size) -> Tensor",
                   /*const_inputs=*/attr::size)) {
      auto sizes = node->get<std::vector<int64_t>>(attr::size).value();
      bool inferred = false;
      size_t inferred_idx;
      int64_t size_product = 1;
      for (size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] == -1) {
          if (inferred)
            throw propagation_error();
          inferred = true;
          inferred_idx = i;
        } else {
          size_product *= sizes[i];
        }
      }

      if (inferred) {
        SHAPE_ASSERT(size_product != 0);
        size_t numel = 1;
        for (int64_t s : tensor_types.at(0)->sizes())
          numel *= s;
        int64_t inferred_size = numel / size_product;
        sizes[inferred_idx] = inferred_size;
      }
      node->output()->setType(tensor_types.at(0)->withSizes(sizes));
      return true;
    } else if (node->matches(
                   "aten::type_as(Tensor self, Tensor other) -> Tensor")) {
      if (tensor_types.at(0)->scalarType() ==
          tensor_types.at(1)->scalarType()) {
        node->output()->setType(node->namedInput(attr::self)->type());
      } else {
        // This will be a copy, so the result will be contiguous
        node->output()->setType(
            tensor_types.at(1)->withSizes(tensor_types.at(0)->sizes()));
      }
      return true;
    } else if (
        node->matches(
            "aten::expand(Tensor self, int[] size, *, bool implicit) -> Tensor",
            /*const_inputs=*/attr::size)) {
      auto tp = tensor_types.at(0);
      std::vector<int64_t> sizes, strides;
      std::tie(sizes, strides) = at::inferExpandGeometry(
          tp->sizes(),
          tp->strides(),
          node->get<std::vector<int64_t>>(attr::size).value());
      node->output()->setType(tp->withSizesStrides(sizes, strides));
      return true;
    } else if (
        node->matches(
            "aten::index_select(Tensor self, int dim, Tensor index) -> Tensor",
            /*const_inputs=*/attr::dim)) {
      auto ten = tensor_types.at(0);
      auto index = tensor_types.at(1);
      int64_t dim = node->get<int64_t>(attr::dim).value();
      SHAPE_ASSERT(index->sizes().size() == 1);
      SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) < ten->sizes().size());
      std::vector<int64_t> sizes = ten->sizes();
      sizes[dim] = index->sizes()[0];
      node->output()->setType(ten->withSizes(sizes));
      return true;
    } else if (node->matches(
                   "aten::chunk(Tensor self, int chunks, int dim) -> Tensor[]",
                   /*const_inputs=*/{attr::chunks, attr::dim})) {
      auto input_type = tensor_types.at(0);
      auto sizes = input_type->sizes();
      const auto& strides = input_type->strides();
      int64_t dim = node->get<int64_t>(attr::dim).value();
      int64_t chunks = node->get<int64_t>(attr::chunks).value();
      sizes[dim] /= chunks;
      for (Value* output : node->outputs()) {
        output->setType(input_type->withSizesStrides(sizes, strides));
      }
      if (input_type->sizes().at(dim) % chunks != 0) {
        sizes[dim] = input_type->sizes().at(dim) % chunks;
        node->outputs().back()->setType(
            input_type->withSizesStrides(sizes, strides));
      }
      return true;
    } else if (node->kind() == onnx::Shape) {
      SHAPE_ASSERT(node->inputs().size() == 1 && node->outputs().size() == 1);
      std::vector<int64_t> dim_vec = {
          (int64_t)tensor_types.at(0)->sizes().size()};
      at::IntList dims(dim_vec);
      node->output()->setType(
          CompleteTensorType::create(at::kLong, at::kCPU, dims));
      return true;
    } else if (node->kind() == onnx::Reshape) {
      setUnshapedType(node);
      return true;
    }
    setUnshapedType(node);
    return false;
  }
};
} // anonymous namespace

void PropagateInputShapes(const std::shared_ptr<Graph>& graph) {
  ShapePropagator(graph).PropagateShapeOnBlock(graph->block());
}

namespace {

void EraseShapeInformation(at::ArrayRef<Value*> vals) {
  for (Value* v : vals) {
    v->setType(unshapedType(v->type()));
  }
}

void EraseShapeInformation(Block* b) {
  EraseShapeInformation(b->inputs());
  EraseShapeInformation(b->outputs());
  for (Node* n : b->nodes()) {
    EraseShapeInformation(n->outputs());
    for (Block* sb : n->blocks()) {
      EraseShapeInformation(sb);
    }
  }
}

} // anonymous namespace

void EraseShapeInformation(const std::shared_ptr<Graph>& graph) {
  EraseShapeInformation(graph->block());
}

} // namespace jit
} // namespace torch
