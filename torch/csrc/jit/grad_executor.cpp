#include <torch/csrc/jit/grad_executor.h>
#include <torch/csrc/jit/graph_executor.h>
#include <torch/csrc/jit/autodiff.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/grad_mode.h>

namespace torch {
namespace jit {
namespace detail {


using autograd::Variable;
using autograd::variable_list;

namespace {

c10::OperatorOptions aliasAnalysisInternalSpecialCase() {
  c10::OperatorOptions options;
  options.setAliasAnalysis(AliasAnalysisKind::INTERNAL_SPECIAL_CASE);
  return options;
}

struct CaptureList {
  CaptureList(size_t capture_size) {
    capture_types_.reserve(capture_size);
    var_captures_.reserve(capture_size); // var_captures_.size() might be
                                         // greater than capture_size
    ivalue_captures_.reserve(capture_size);
  }

  void captureTensor(const at::Tensor& tensor, bool is_output) {
    var_captures_.emplace_back(Variable(tensor), is_output);
  }

  void capture(const IValue& val, bool is_output) {
    if (val.isTensor()) {
      capture_types_.emplace_back(CAPTURE_TENSOR);
      captureTensor(val.toTensor(), is_output);
    } else if (val.isTensorList()) {
      //  For TensorList, we have to flatten it to Tensors during saving and
      //  unflatten it back to TensorList when using it in backward apply().
      //  This is to avoid any implicit mutation to TensorList happened
      //  between forward & backward.
      capture_types_.emplace_back(CAPTURE_LIST);
      c10::ArrayRef<at::Tensor> tensors = val.toTensorListRef();
      sizes_.push_back(tensors.size());

      for (const at::Tensor& tensor : tensors) {
        captureTensor(tensor, is_output);
      }
    } else {
      capture_types_.emplace_back(CAPTURE_IVALUE);
      ivalue_captures_.push_back(val);
    }
  }

  size_t size() const {
    return capture_types_.size();
  }

  void unpack(
      Stack& stack,
      const std::shared_ptr<autograd::Node>& saved_for) {
    auto var_capture_it = var_captures_.begin();
    auto ivalue_capture_it = ivalue_captures_.begin();
    auto size_it = sizes_.begin();
    for (Capture capture_type : capture_types_) {
      switch (capture_type) {
        case CAPTURE_TENSOR: {
          stack.emplace_back(var_capture_it->unpack(saved_for));
          ++var_capture_it;
        } break;
        case CAPTURE_LIST: {
          c10::List<at::Tensor> lst;
          auto size = *size_it++;
          for (size_t i = 0; i < size; i++) {
            lst.emplace_back(var_capture_it->unpack(saved_for));
            var_capture_it++;
          }
          stack.emplace_back(std::move(lst));
        } break;
        case CAPTURE_IVALUE: {
          stack.push_back(*ivalue_capture_it++);
        } break;
      }
    }
  }

 private:
  enum Capture : uint8_t {
    CAPTURE_TENSOR,
    CAPTURE_LIST,
    CAPTURE_IVALUE,
  };

  std::vector<Capture> capture_types_;
  std::vector<autograd::SavedVariable> var_captures_;
  std::vector<IValue> ivalue_captures_;
  std::vector<size_t> sizes_;
};

// how do we turn a flattened list of tensors back into the ivalues that
// the DifferentiableGraphBackward expects
struct UnpackInstructions {
  UnpackInstructions(size_t num_inputs) {
    insts_.reserve(num_inputs);
  }
  void pushTensor() {
    insts_.emplace_back(PUSH_TENSOR);
  }
  void pushTensorList(size_t size) {
    insts_.emplace_back(PUSH_LIST);
    sizes_.push_back(size);
  }
  void unpack(variable_list&& inputs, Stack& stack) {
    auto input_it = std::make_move_iterator(inputs.begin());
    auto sizes_it = sizes_.begin();
    for (Inst inst : insts_) {
      switch (inst) {
        case PUSH_TENSOR: {
          at::Tensor t = *input_it++;
          stack.emplace_back(std::move(t));
        } break;
        case PUSH_LIST: {
          std::vector<at::Tensor> lst(input_it, input_it + *sizes_it++);
          stack.emplace_back(c10::impl::toList(std::move(lst)));
        } break;
      }
    }
  }

 private:
  enum Inst : uint8_t {
    PUSH_TENSOR,
    PUSH_LIST, // consumes one size
  };
  std::vector<Inst> insts_;
  std::vector<size_t> sizes_;
};

// unpack values packed by `packReturnValuesIntoTuple`
static void unpackReturnTuple(Stack &stack) {
  auto tuple = pop(stack).toTuple();
  stack.insert(stack.end(), tuple->elements().begin(), tuple->elements().end());
}

} // namespace

struct DifferentiableGraphBackwardImpl {
  DifferentiableGraphBackwardImpl(
      GraphExecutor executor,
      size_t input_size,
      size_t capture_size,
      DifferentiableGraphBackward &node)
      : executor(std::move(executor)),
        captures_(capture_size),
        input_instructions_(input_size),
        node(node) {}

  variable_list apply(variable_list&& inputs) {
    Stack stack;
    stack.reserve(captures_.size() + inputs.size());

    input_instructions_.unpack(std::move(inputs), stack);
    captures_.unpack(stack, node.shared_from_this());
    executor.run(stack);
    unpackReturnTuple(stack);

    // NB: stack.size() == num_outputs() is not always true
    // after we added TensorList support.
    // Example: aten::stack(Tensor[] tensors, int) where
    // tensors = [x, x]
    // Here stack.size()[=1] with a TensorList IValue of
    // backward graph output.
    // num_outputs()[=2], however, is the number of outputs of
    // grad_fn (an autograd::Node). grad_fn's outputs are
    // grads with regard to Tensor/Variables `x`, but not
    // graph input TensorList [x, x]. These two grads will
    // be accumulated to x.grad later using autograd::InputBuffer.
    variable_list outputs;
    outputs.reserve(node.num_outputs());
    size_t output_index = 0;
    for (IValue& v : stack) {
      if (v.isTensorList()) {
        for (const at::Tensor& tensor : v.toTensorListRef()) {
          produceOutput(output_index++, tensor, outputs);
        }
      } else if (v.isTensor()) {
        produceOutput(output_index++, std::move(v).toTensor(), outputs);
      } else {
        // Input grad can also be None even if it requires grad
        // Example: `other` in expand_as(self, other)
        outputs.emplace_back();
      }
    }
    return outputs;
  }

  void capture(const IValue& val, bool is_output) {
    captures_.capture(val, is_output);
  }

  void addOutputForTensor(const at::Tensor& tensor) {
    auto v = Variable(tensor);
    node.add_next_edge(v.defined() ? v.gradient_edge() : autograd::Edge{});
  }
  void addOutputForIValue(const IValue& value) {
    if (value.isTensorList()) {
      for (const at::Tensor& tensor : value.toTensorListRef()) {
        addOutputForTensor(tensor);
      }
    } else {
      addOutputForTensor(value.toTensor());
    }
  }

  void addInputVariable(Variable output) {
    // NB: since our requires_grad setting is only a heuristic we might end
    // up wanting to differentiate through integral tensors, which is
    // generally a hard error in autograd.
    if (at::isFloatingType(output.type().scalarType())) {
      autograd::create_gradient_edge(output, node.shared_from_this());
      output.set_requires_grad(true);
    } else {
      node.add_input_metadata(autograd::Node::undefined_input{});
    }
  }

  void addInputIValue(const IValue& v) {
    if (v.isTensorList()) {
      c10::ArrayRef<at::Tensor> tensors = v.toTensorListRef();
      input_instructions_.pushTensorList(tensors.size());
      for (const at::Tensor& tensor : tensors) {
        addInputVariable(tensor);
      }
    } else if (v.isTensor()) {
      input_instructions_.pushTensor();
      addInputVariable(v.toTensor());
    }
  }

  std::string toString() const {
    return executor.graph()->toString();
  }

 private:
  void produceOutput(size_t i, at::Tensor output, variable_list& outputs) {
    if (node.should_compute_output(i)) {
      const auto& edge = node.next_edge(i);
      if (output.defined()) {
        outputs.emplace_back(std::move(output));
      } else if (edge.is_valid()) {
        outputs.emplace_back(
            edge.function->input_metadata(edge.input_nr).zeros_like());
      } else {
        outputs.emplace_back();
      }
    } else {
      outputs.emplace_back();
    }
  }

  GraphExecutor executor;
  CaptureList captures_;
  UnpackInstructions input_instructions_;
  DifferentiableGraphBackward &node;
};

// an optimized way of executing the subgraph computed directly on
// tensors rather than Variables.
// This will unwrap Variables, run the plan, and re-wrap them.
// It can optionally also have a gradient which is hooked up
// to the output Variables if present.
struct DifferentiableGraphOp {
  DifferentiableGraphOp(Gradient grad)
      : f(grad.f),
        grad(std::move(grad)),
        grad_executor(this->grad.df),
        num_inputs(this->grad.f->inputs().size()),
        num_outputs(this->grad.f->outputs().size()) {}

  // XXX: keep in mind that stack can be larger than the inputs we need!
  int operator()(Stack& stack) const {
    auto node = std::make_shared<DifferentiableGraphBackward>(
        grad_executor,
        grad.df_input_vjps.size(),
        grad.df_input_captured_inputs.size() +
            grad.df_input_captured_outputs.size());
    auto grad_fn = node->pImpl;

    {
      auto inputs = last(stack, num_inputs);
      // hook up the outputs of df to the gradient functions of the inputs that
      // require gradients
      for (auto idx : grad.df_output_vjps) {
        grad_fn->addOutputForIValue(inputs[idx]);
      }
      captureInputs(*grad_fn, inputs);
    }

    detachVariables(stack);
    InterpreterState(f).run(stack);

    {
      auto outputs = last(stack, num_outputs);
      // hookup the gradients for the output tensors that require gradients
      // to the inputs to our gradient function df
      // TODO - XXX - if any output is the same tensor multiple times, views
      // have to be setup here. We need to refactor autograd until it is safe
      // for tensors to be constructed without all the viewing infrastructure.
      // this is currently intentionally not done here so we can get an idea of
      // our perf before introducing overhead for correctness
      for (auto idx : grad.df_input_vjps) {
        grad_fn->addInputIValue(outputs[idx]);
      }
      captureOutputs(*grad_fn, outputs);
      // drop the temporary outputs so that we return the same number of
      // outputs as if we were not also calculating gradient
      const size_t num_temporary_outputs = num_outputs - grad.f_real_outputs;
      stack.erase(stack.end() - num_temporary_outputs, stack.end());
    }
    return 0;
  }

 private:
  friend GraphExecutor* detail::getGradExecutor(Operation& op);

  at::Tensor detach(at::Tensor t) const {
    if (!t.defined()) {
      return t;
    }
    return autograd::as_variable_ref(t).detach();
  }

  void detach(IValue& v) const {
    if (v.isTensor()) {
      v = IValue(detach(std::move(v).toTensor()));
    } else if (v.isTensorList()) {
      c10::List<at::Tensor> lst = std::move(v).toTensorList();
      for (size_t i = 0; i < lst.size(); ++i) {
        lst.set(i, detach(lst.extract(i)));
      }
      v = std::move(lst);
    }
  }

  void detachVariables(Stack& stack) const {
    // It would be nice to use an ArrayRef here, but unfortunately those can
    // only return const references, so we need to do a bunch of indexing
    // ourselves.
    const int64_t stack_size = stack.size();
    const int64_t stack_offset = stack_size - num_inputs;
    for (int64_t i = stack_offset; i < stack_size; ++i) {
      detach(stack[i]);
    }
  }
  // Capture (save) inputs that would be required to subsequently run backwards
  void captureInputs(
      DifferentiableGraphBackwardImpl& grad_fn,
      at::ArrayRef<IValue> inputs) const {
    for (size_t offset : grad.df_input_captured_inputs) {
      grad_fn.capture(inputs[offset], /*is_output*/ false);
    }
  }
  void captureOutputs(
      DifferentiableGraphBackwardImpl& grad_fn,
      at::ArrayRef<IValue> outputs) const {
    for (size_t offset : grad.df_input_captured_outputs) {
      grad_fn.capture(outputs[offset], /*is_output*/ true);
    }
  }

  Code f;
  Gradient grad;
  GraphExecutor grad_executor;

  const size_t num_inputs;
  const size_t num_outputs;
};

Gradient getGradient(const Node* n) {
  AT_ASSERT(n->kind() == prim::DifferentiableGraph);
  Gradient grad;
  grad.f = n->g(attr::Subgraph);
  grad.df = n->g(attr::ReverseSubgraph);
  grad.f_real_outputs = n->i(attr::f_real_outputs);
  grad.df_input_vjps = fmap<size_t>(n->is(attr::df_input_vjps));
  grad.df_input_captured_inputs =
      fmap<size_t>(n->is(attr::df_input_captured_inputs));
  grad.df_input_captured_outputs =
      fmap<size_t>(n->is(attr::df_input_captured_outputs));
  grad.df_output_vjps = fmap<size_t>(n->is(attr::df_output_vjps));
  return grad;
}

RegisterOperators reg_graph_executor_ops({Operator(
    prim::DifferentiableGraph,
    [](const Node* n) -> Operation {
      return DifferentiableGraphOp(getGradient(n));
    },
    aliasAnalysisInternalSpecialCase())});

GraphExecutor* getGradExecutor(Operation& op) {
  if (auto diff_op = op.target<DifferentiableGraphOp>()) {
    return &diff_op->grad_executor;
  }
  return nullptr;
}

DifferentiableGraphBackward::DifferentiableGraphBackward(
  GraphExecutor executor, size_t input_size, size_t capture_size)
  : pImpl(std::make_shared<DifferentiableGraphBackwardImpl>(std::move(executor), input_size, capture_size, *this)) {}

std::string DifferentiableGraphBackward::toString() const {
  return pImpl->toString();
}

variable_list DifferentiableGraphBackward::apply(variable_list&& inputs) {
  return pImpl->apply(std::move(inputs));
}

} // namespace detail
} // namespace jit
} // namespace torch