#include <torch/csrc/jit/runtime/graph_executor.h>

#include <ATen/core/ivalue.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/batch_mm.h>
#include <torch/csrc/jit/passes/canonicalize_graph_fuser_ops.h>
#include <torch/csrc/jit/passes/common_expression_hoisting.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/decompose_ops.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/inline_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/inplace_check.h>
#include <torch/csrc/jit/passes/loop_unrolling.h>
#include <torch/csrc/jit/passes/lower_grad_of.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/remove_expands.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/requires_grad_analysis.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/specialize_autogradzero.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/resource_guard.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/autodiff.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/graph_executor_impl.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/profiling_graph_executor_impl.h>
#include <torch/csrc/jit/runtime/profiling_record.h>

#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/jit/runtime/logging.h>

#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch {
namespace jit {

EnableProfilingGuard::EnableProfilingGuard() {
  auto& profiling_mode = getProfilingMode();
  old_profiling_mode = profiling_mode;
  profiling_mode = true;
  auto& executor_mode = getExecutorMode();
  old_executor_mode = executor_mode;
  executor_mode = true;
}

EnableProfilingGuard::~EnableProfilingGuard() {
  getProfilingMode() = old_profiling_mode;
  getExecutorMode() = old_executor_mode;
}

namespace {
c10::AliasAnalysisKind aliasAnalysisInternalSpecialCase() {
  return AliasAnalysisKind::INTERNAL_SPECIAL_CASE;
}
} // namespace

// for debugging it is helpful to be able to force autodiff subgraphs
// to be created, to check their correctness, even when the
// size of the of the subgraph is too small to be profitable.
thread_local bool autodiff_subgraph_inlining = true;
void debugSetAutodiffSubgraphInlining(bool state) {
  autodiff_subgraph_inlining = state;
}

bool getAutodiffSubgraphInlining() {
  return autodiff_subgraph_inlining;
}

// for debugging it is helpful to be able to force fusion groups
// to be created
static std::atomic<bool> fusion_group_inlining(true);
void debugSetFusionGroupInlining(bool state) {
  fusion_group_inlining = state;
}

bool getFusionGroupInlining() {
  return fusion_group_inlining;
}

thread_local std::weak_ptr<Graph> last_executed_optimized_graph;
std::shared_ptr<Graph> lastExecutedOptimizedGraph() {
  return last_executed_optimized_graph.lock();
}
namespace {

using tensor_list = std::vector<at::Tensor>;
using Variable = autograd::Variable;
using autograd::variable_list;

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
      auto tensors = val.toTensorList();
      sizes_.push_back(tensors.size());

      for (const at::Tensor tensor : tensors) {
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

  void unpack(Stack& stack, const std::shared_ptr<autograd::Node>& saved_for) {
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
          for (const auto i : c10::irange(size)) {
            (void)i;
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

  void release_variables() {
    for (auto& var_capture_ : var_captures_) {
      var_capture_.reset_data();
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
  void pushNone() {
    insts_.emplace_back(PUSH_NONE);
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
          stack.emplace_back(lst);
        } break;
        case PUSH_NONE: {
          stack.emplace_back(IValue());
        }
      }
    }
  }

 private:
  enum Inst : uint8_t {
    PUSH_TENSOR,
    PUSH_LIST, // consumes one size
    PUSH_NONE,
  };
  std::vector<Inst> insts_;
  std::vector<size_t> sizes_;
};

// unpack values packed by `packReturnValuesIntoTuple`
static void unpackReturnTuple(Stack& stack) {
  auto tuple = pop(stack).toTuple();
  stack.insert(stack.end(), tuple->elements().begin(), tuple->elements().end());
}

struct DifferentiableGraphBackward : public autograd::Node {
  DifferentiableGraphBackward(
      GraphExecutor executor,
      size_t input_size,
      size_t capture_size)
      : executor(std::move(executor)),
        captures_(capture_size),
        input_instructions_(input_size) {}

  variable_list apply(variable_list&& inputs) override {
    Stack stack;
    stack.reserve(captures_.size() + inputs.size());

    input_instructions_.unpack(std::move(inputs), stack);
    captures_.unpack(stack, shared_from_this());
    GRAPH_DEBUG("Running DifferentiableGraphBackward for ", &executor);
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
    outputs.reserve(num_outputs());
    size_t output_index = 0;
    for (IValue& v : stack) {
      if (v.isTensorList()) {
        for (at::Tensor tensor : v.toTensorList()) {
          produceOutput(output_index++, std::move(tensor), outputs);
        }
      } else if (v.isTensor()) {
        produceOutput(output_index++, std::move(v).toTensor(), outputs);
      } else {
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(v.isNone());
        output_index++;
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
    add_next_edge(
        v.defined() ? torch::autograd::impl::gradient_edge(v)
                    : autograd::Edge{});
  }
  void addOutputForIValue(const IValue& value) {
    if (value.isTensorList()) {
      for (const at::Tensor tensor : value.toTensorList()) {
        addOutputForTensor(tensor);
      }
    } else if (value.isTensor()) {
      addOutputForTensor(value.toTensor());
    } else {
      // We could have None passed here via `Optional[Tensor]`
      add_next_edge(autograd::Edge{});
    }
  }

  void addInputVariable(Variable output) {
    // NB: since our requires_grad setting is only a heuristic we might end
    // up wanting to differentiate through integral tensors, which is
    // generally a hard error in autograd.
    if (at::isFloatingType(output.scalar_type())) {
      autograd::create_gradient_edge(output, shared_from_this());
      output.set_requires_grad(true);
    } else {
      add_input_metadata(autograd::Node::undefined_input{});
    }
  }

  void addInputIValue(const IValue& v) {
    if (v.isTensorList()) {
      auto tensors = v.toTensorList();
      input_instructions_.pushTensorList(tensors.size());
      for (const at::Tensor tensor : tensors) {
        addInputVariable(tensor);
      }
    } else if (v.isTensor()) {
      input_instructions_.pushTensor();
      addInputVariable(v.toTensor());
    } else if (v.isNone()) {
      input_instructions_.pushNone();
      addInputVariable(Variable{});
    }
  }

  void release_variables() override {
    captures_.release_variables();
  }

 private:
  void produceOutput(size_t i, at::Tensor output, variable_list& outputs) {
    if (should_compute_output(i)) {
      const auto& edge = next_edge(i);
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

  friend struct ExecutionPlan;
  GraphExecutor executor;
  CaptureList captures_;
  UnpackInstructions input_instructions_;
};

// an optimized way of executing the subgraph computed directly on
// tensors rather than Variables.
// This will unwrap Variables, run the plan, and re-wrap them.
// It can optionally also have a gradient which is hooked up
// to the output Variables if present.
struct DifferentiableGraphOp {
  DifferentiableGraphOp(Gradient grad)
      : f_ptr(std::make_shared<GraphExecutor>(grad.f, "<forward op>")),
        legacy_f(grad.f, "<forward op>"),
        grad(std::move(grad)),
        grad_executor(this->grad.df, "<backward op>"),
        num_inputs(this->grad.f->inputs().size()),
        num_outputs(this->grad.f->outputs().size()) {}

  // XXX: keep in mind that stack can be larger than the inputs we need!
  void operator()(Stack& stack) const {
    auto grad_fn = std::make_shared<DifferentiableGraphBackward>(
        grad_executor,
        grad.df_input_vjps.size(),
        grad.df_input_captured_inputs.size() +
            grad.df_input_captured_outputs.size());

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
    if (IsNewExecutorEnabled()) {
      ExecutionPlan plan =
          f_ptr->getPlanFor(stack, GraphExecutor::getDefaultNumBailOuts());
      InterpreterState(plan.code).run(stack);
    } else {
      InterpreterState(legacy_f).run(stack);
    }

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
  }

 private:
  friend GraphExecutor* detail::getGradExecutor(Operation& op);
  friend GraphExecutor* detail::getDifferentiableGraphOpExecutor(Operation& op);

  at::Tensor detach(at::Tensor t) const {
    if (!t.defined()) {
      return t;
    }
    return t.detach();
  }

  void detach(IValue& v) const {
    if (v.isTensor()) {
      v = IValue(detach(std::move(v).toTensor()));
    } else if (v.isTensorList()) {
      std::vector<at::Tensor> lst = v.toTensorVector();
      for (auto& tensor : lst) {
        tensor = detach(tensor);
      }
      // NOLINTNEXTLINE(performance-move-const-arg)
      v = std::move(lst);
    }
  }

  void detachVariables(Stack& stack) const {
    // It would be nice to use an ArrayRef here, but unfortunately those can
    // only return const references, so we need to do a bunch of indexing
    // ourselves.
    const int64_t stack_size = stack.size();
    const int64_t stack_offset = stack_size - num_inputs;
    for (const auto i : c10::irange(stack_offset, stack_size)) {
      detach(stack[i]);
    }
  }
  // Capture (save) inputs that would be required to subsequently run backwards
  void captureInputs(
      DifferentiableGraphBackward& grad_fn,
      at::ArrayRef<IValue> inputs) const {
    for (size_t offset : grad.df_input_captured_inputs) {
      grad_fn.capture(inputs[offset], /*is_output*/ false);
    }
  }
  void captureOutputs(
      DifferentiableGraphBackward& grad_fn,
      at::ArrayRef<IValue> outputs) const {
    for (size_t offset : grad.df_input_captured_outputs) {
      grad_fn.capture(outputs[offset], /*is_output*/ true);
    }
  }

  std::shared_ptr<GraphExecutor> f_ptr;
  Code legacy_f;
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
} // anonymous namespace

RegisterOperators reg_graph_executor_ops({Operator(
    prim::DifferentiableGraph,
    [](const Node* n) -> Operation {
      return DifferentiableGraphOp(getGradient(n));
    },
    aliasAnalysisInternalSpecialCase())});

namespace detail {

GraphExecutor* getGradExecutor(Operation& op) {
  if (auto diff_op = op.target<DifferentiableGraphOp>()) {
    return &diff_op->grad_executor;
  }
  return nullptr;
}

GraphExecutor* getDifferentiableGraphOpExecutor(Operation& op) {
  TORCH_INTERNAL_ASSERT(
      IsNewExecutorEnabled(),
      __FUNCTION__,
      " is only accessible under profiling executor\n");
  if (auto diff_op = op.target<DifferentiableGraphOp>()) {
    return diff_op->f_ptr.get();
  }
  return nullptr;
}
} // namespace detail

void GraphExecutorImplBase::run(Stack& stack) {
  GRAPH_DUMP("About to run GraphExecutorImplBase on: ", graph);
  TORCH_CHECK(
      stack.size() >= num_inputs,
      "expected ",
      num_inputs,
      " inputs, but got only ",
      stack.size());

  C10_LOG_API_USAGE_ONCE("torch.graph_executor.run");
  logging::getLogger()->addStatValue(
      logging::runtime_counters::GRAPH_EXECUTOR_INVOCATIONS, 1.0);

  const ExecutionPlan& plan =
      getPlanFor(stack, GraphExecutor::getDefaultNumBailOuts());
  InterpreterState(plan.code).run(stack);
  last_executed_optimized_graph = plan.graph;
}

c10::intrusive_ptr<Future> GraphExecutorImplBase::runAsync(
    Stack& stack,
    TaskLauncher taskLauncher) {
  TORCH_CHECK(
      stack.size() >= num_inputs,
      "expected ",
      num_inputs,
      " inputs, but got only ",
      stack.size());

  C10_LOG_API_USAGE_ONCE("torch.graph_executor.runAsync");
  logging::getLogger()->addStatValue(
      logging::runtime_counters::GRAPH_EXECUTOR_INVOCATIONS, 1.0);

  struct Frame {
    explicit Frame(ExecutionPlan eplan, TaskLauncher taskLauncher)
        : plan(std::move(eplan)), state(plan.code, std::move(taskLauncher)) {}
    ExecutionPlan plan;
    InterpreterState state;
  };
  auto frame = std::make_shared<Frame>(
      getPlanFor(stack, GraphExecutor::getDefaultNumBailOuts()),
      std::move(taskLauncher));
  auto res = frame->state.runAsync(stack);
  last_executed_optimized_graph = frame->plan.graph;
  if (!res->completed()) {
    // If not completed, persist the Frame until complete.
    res->addCallback([frame](Future& /* unused */) {});
  }
  return res;
}

// a Graph can be created via tracing, or via a language-based frontend
// GraphExecutor runs it. It can run the same graph on many different sizes
// and different requires_grad states, and handles specializations for each
// situation. GraphExecutor is completely unaware of tracing or module
// parameters to keep the tracing concerns separated.
struct GraphExecutorImpl : public GraphExecutorImplBase {
  GraphExecutorImpl(
      const std::shared_ptr<Graph>& graph,
      std::string function_name)
      : GraphExecutorImplBase(graph, std::move(function_name)),
        arg_spec_creator_(*graph) {
    logging::getLogger()->addStatValue(
        logging::runtime_counters::GRAPH_EXECUTORS_CONSTRUCTED, 1.0);
  }

  const ExecutionPlan& getPlanFor(Stack& stack, size_t remaining_bailout_depth)
      override {
    return getGraphExecutorOptimize() ? getOrCompile(stack)
                                      : getOrCompileFallback();
  }

  GraphExecutorState getDebugState() override {
    GraphExecutorState state;
    state.graph = graph.get();
    if (fallback) {
      state.fallback = fallback;
    }
    for (auto& entry : plan_cache) {
      state.execution_plans.emplace(entry.first, entry.second);
    }
    return state;
  }

 protected:
  friend struct GraphExecutor;

  const ExecutionPlan& getOrCompileFallback() {
    std::lock_guard<std::mutex> lock(compile_mutex);
    if (!fallback) {
      auto graph_ = graph->copy();
      runRequiredPasses(graph_);
      fallback = ExecutionPlan(graph_, function_name_);
    }
    return fallback;
  }

  const ExecutionPlan& getOrCompile(const Stack& stack) {
    // outside lock guard, to minimize the time holding the lock on the fast
    // path ArgumentSpec even computes its hashCode here.
    ArgumentSpec spec =
        arg_spec_creator_.create(autograd::GradMode::is_enabled(), stack);
    {
      std::lock_guard<std::mutex> lock(compile_mutex);
      auto it = plan_cache.find(spec);
      if (it != plan_cache.end()) {
        logging::getLogger()->addStatValue(
            logging::runtime_counters::EXECUTION_PLAN_CACHE_HIT, 1.0);
        return it->second;
      }
      auto plan = compileSpec(spec);
      auto r = plan_cache.emplace(std::move(spec), std::move(plan));
      logging::getLogger()->addStatValue(
          logging::runtime_counters::EXECUTION_PLAN_CACHE_MISS, 1.0);
      return r.first->second;
    }
  }

  ExecutionPlan compileSpec(const ArgumentSpec& spec) {
    auto opt_graph = graph->copy();
    GRAPH_DUMP("Optimizing the following function:", opt_graph);
    arg_spec_creator_.specializeTypes(*opt_graph, spec);

    // Phase 0. Inline functions, then clean up any artifacts that the inliner
    //          left in that may inhibit optimization
    Inline(*opt_graph);
    GRAPH_DEBUG("After Inline, before LowerGradOf\n", *opt_graph);
    LowerGradOf(*opt_graph);
    GRAPH_DEBUG(
        "After LowerGradOf, before specializeAutogradZero\n", *opt_graph);
    specializeAutogradZero(opt_graph);
    GRAPH_DEBUG(
        "After specializeAutogradZero, before LowerSimpleTuples\n", *opt_graph);
    LowerSimpleTuples(opt_graph);
    GRAPH_DEBUG(
        "After LowerSimpleTuples, before ConstantPooling\n", *opt_graph);
    ConstantPooling(opt_graph);
    GRAPH_DEBUG(
        "After ConstantPooling, before runRequiredPasses\n", *opt_graph);

    // Phase 1. Specialize to input definedness (this is very important for
    //          gradient graphs), and run required passes to bring the graph
    //          to an executable form.
    runRequiredPasses(opt_graph);
    GRAPH_DEBUG(
        "After runRequiredPasses, before ConstantPropagation\n", *opt_graph);

    // Phase 2. Propagate detailed information about the spec through the
    //          graph (enabled more specializations in later passes).
    //          Shape propagation sometimes depends on certain arguments being
    //          constants, and constant propagation doesn't need shape
    //          information anyway, so it's better to run it first.
    ConstantPropagation(opt_graph);
    GRAPH_DEBUG(
        "After ConstantPropagation, before PropagateInputShapes\n", *opt_graph);
    PropagateInputShapes(opt_graph);
    GRAPH_DEBUG(
        "After PropagateInputShapes, before PropagateRequiresGrad\n",
        *opt_graph);
    PropagateRequiresGrad(opt_graph);
    GRAPH_DEBUG(
        "After PropagateRequiresGrad, before runOptimization\n", *opt_graph);

    // Phase 3. Run differentiable optimizations (i.e. simple graph rewrites
    //          that we can still execute using autograd).
    runOptimization(opt_graph);

    // Phase 4. If this graph will be differentiated, we need to slice out the
    //          symbolically differentiable subgraphs for further optimizations.
    // Phase 5. Apply non-differentiable optimizations to the graphs we've found
    //          (or the whole graph if we know we won't need its derivative).
    if (needsGradient(opt_graph)) {
      auto diff_nodes = CreateAutodiffSubgraphs(
          opt_graph,
          autodiff_subgraph_inlining ? autodiffSubgraphNodeThreshold : 1);
      GRAPH_DEBUG("After CreateAutodiffSubgraphs\n", *opt_graph);
      size_t idx = 0;
      for (Node* dnode : diff_nodes) {
        GRAPH_DEBUG("Optimizing diff node ", idx);
        auto diff_graph = std::move(dnode->g(attr::Subgraph));
        Gradient gradient = differentiate(diff_graph);
        GRAPH_DEBUG("Forward graph:\n", *(gradient.f));
        GRAPH_DEBUG("Backward graph:\n", *(gradient.df));
        // Run post differentiation optimizations, Autodiff will replace some
        // parts of graph with new graph, these new graphs usually consists of
        // control flows and miss shape information on nodes, so we run shape
        // prop and differentiable optimizations to ensure the graph is
        // optimized
        PropagateInputShapes(gradient.f);
        GRAPH_DEBUG("After PropagateInputShapes\n", *(gradient.f));
        runOptimization(gradient.f);
        // run non diff optimization on the forward graph
        runNondiffOptimization(gradient.f);
        packGradient(gradient, dnode);
        GRAPH_DEBUG("Finished optimizing diff node ", idx++);
      }
      InlineAutodiffSubgraphs(
          opt_graph,
          autodiff_subgraph_inlining ? autodiffSubgraphInlineThreshold : 1);
      GRAPH_DEBUG("After InlineAutodiffSubgraphs\n", *opt_graph);
    } else {
      runNondiffOptimization(opt_graph);
    }
    // Make sure there are no leftovers from any passes.
    EliminateDeadCode(opt_graph);
    GRAPH_DUMP("After compileSpec optimizations:", opt_graph);
    return ExecutionPlan(opt_graph, function_name_);
  }

  ~GraphExecutorImpl() override = default;

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  ArgumentSpecCreator arg_spec_creator_;
  // Populated only when optimize is false (and in that case plan_cache will be
  // unused). The compiled version of graph.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  ExecutionPlan fallback;

  // Mapping from argument configurations to optimized versions of the graph
  // that are specialized to the spec.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unordered_map<ArgumentSpec, ExecutionPlan> plan_cache;
};

GraphExecutor::GraphExecutor(
    const std::shared_ptr<Graph>& graph,
    std::string function_name)
    : pImpl(
          IsNewExecutorEnabled()
              ? dynamic_cast<GraphExecutorImplBase*>(
                    new ProfilingGraphExecutorImpl(
                        graph,
                        std::move(function_name)))
              : dynamic_cast<GraphExecutorImplBase*>(
                    new GraphExecutorImpl(graph, std::move(function_name)))) {}

void GraphExecutor::run(Stack& inputs) {
  return pImpl->run(inputs);
}

c10::intrusive_ptr<Future> GraphExecutor::runAsync(
    Stack& stack,
    TaskLauncher taskLauncher) {
  return pImpl->runAsync(stack, std::move(taskLauncher));
}

size_t GraphExecutor::getDefaultNumBailOuts() {
  return getProfilingMode() ? getBailoutDepth().load() : 0;
}

const ExecutionPlan& GraphExecutor::getPlanFor(
    Stack& inputs,
    size_t remaining_bailout_depth) {
  return pImpl->getPlanFor(inputs, remaining_bailout_depth);
}

GraphExecutorState GraphExecutor::getDebugState() {
  return pImpl->getDebugState();
}

void GraphExecutor::debugFlushCompilationCache() {
  if (auto ppImpl =
          std::dynamic_pointer_cast<ProfilingGraphExecutorImpl>(pImpl)) {
    ppImpl->debugFlushCompilationCache();
  } else {
    // we are deprecating legacy executor
    TORCH_INTERNAL_ASSERT("Not Implemented for Legacy Executor");
  }
}

bool GraphExecutor::isOptimized() const {
  return pImpl && pImpl->isOptimized();
}

TORCH_API bool IsNewExecutorEnabled() {
  static const auto disable_new_executor =
      std::getenv("TORCH_JIT_DISABLE_NEW_EXECUTOR");
  return getExecutorMode() && FLAGS_torch_jit_enable_new_executor &&
      !disable_new_executor;
}

void runRequiredPasses(const std::shared_ptr<Graph>& g) {
  // implicit inserted expand nodes are not necessarily always valid
  // when used inside script methods that might have unstable shapes
  // we remove the implicitly created ones, and have shape analysis
  // add valid expand nodes when the shapes are stable
  RemoveExpands(g);
  CanonicalizeOps(g);
  EliminateDeadCode(g);
}

void packGradient(const Gradient& gradient, Node* dnode) {
  AT_ASSERT(dnode->kind() == prim::DifferentiableGraph);
  dnode->g_(attr::Subgraph, gradient.f)
      ->g_(attr::ReverseSubgraph, gradient.df)
      ->i_(attr::f_real_outputs, gradient.f_real_outputs)
      ->is_(attr::df_input_vjps, fmap<int64_t>(gradient.df_input_vjps))
      ->is_(
          attr::df_input_captured_inputs,
          fmap<int64_t>(gradient.df_input_captured_inputs))
      ->is_(
          attr::df_input_captured_outputs,
          fmap<int64_t>(gradient.df_input_captured_outputs))
      ->is_(attr::df_output_vjps, fmap<int64_t>(gradient.df_output_vjps));
}

static bool mayIntroduceGradient(const Block* b) {
  for (const Node* n : b->nodes()) {
    if (n->kind() == prim::PythonOp)
      return true;
    for (const Block* bb : n->blocks()) {
      if (mayIntroduceGradient(bb))
        return true;
    }
  }
  return false;
}

bool needsGradient(const std::shared_ptr<const Graph>& graph) {
  if (!autograd::GradMode::is_enabled()) {
    return false;
  }

  if (mayIntroduceGradient(graph->block())) {
    return true;
  }

  for (const Value* input : graph->inputs()) {
    if (input->type()->requires_grad()) {
      return true;
    }
  }

  return false;
}

void runNondiffOptimization(
    std::shared_ptr<Graph>& graph,
    bool strict_fuser_check) {
  GRAPH_DEBUG(
      "Before customPrePassses (beginning of runNondiffOptimization)\n",
      *graph);
  // Run custom passes that different backends can register.
  for (const auto& passPair : getCustomPrePasses()) {
    passPair.first(graph);
  }
  GRAPH_DEBUG("After customPrePassses\n", *graph);

  // decomposition pass, decompose certain ops that will be used in the
  // following passes (like batchmm and jit fusion)
  if (!getProfilingMode()) {
    DecomposeOps(graph);
    GRAPH_DEBUG("After DecomposeOps\n", *graph);
  }

  // TupleConstruct / TupleUnpack pairs can still be present at this point
  // and must be removed for fusion.
  LowerSimpleTuples(graph);
  GRAPH_DEBUG("After LowerSimpleTuples, before BatchMM\n", *graph);

  // Rewrite subgraphs with many MMs into expressions that batch them.
  BatchMM(graph);

  GRAPH_DEBUG("After BatchMM, before Fusion\n", *graph);
  if (getProfilingMode()) {
    if (tensorExprFuserEnabled()) {
      FuseTensorExprs(graph);
    }
  } else {
    FuseGraph(graph, strict_fuser_check);
  }
  GRAPH_DEBUG("After Fusion\n", *graph);

  // Run custom post-fusion passes
  for (const auto& passPair : getCustomPostPasses()) {
    passPair.first(graph);
  }
  GRAPH_DEBUG(
      "After customPostPassses (end of runNondiffOptimization)\n", *graph);
}

void runOptimization(
    std::shared_ptr<Graph>& graph,
    bool unroll_non_constant_loops,
    bool const_prop_user_classes) {
  // Basic graph preprocessing to eliminate noise.
  GRAPH_DEBUG(
      "Before EliminateDeadCode (beginning of runOptimization)\n", *graph);
  EliminateDeadCode(graph);
  GRAPH_DEBUG(
      "After EliminateDeadCode, before EliminateCommonSubexpression\n", *graph);
  EliminateCommonSubexpression(graph);
  GRAPH_DEBUG(
      "After EliminateCommonSubexpression , before PeepholeOptimize\n", *graph);

  PeepholeOptimize(graph);
  GRAPH_DEBUG("After PeepholeOptimize, before ConstantPropagation\n", *graph);

  if (const_prop_user_classes) {
    ConstantPropagation(graph);
  } else {
    ConstantPropagation(graph, true);
  }
  GRAPH_DEBUG("After ConstantPropagation, before ConstantPooling\n", *graph);

  ConstantPooling(graph);
  GRAPH_DEBUG("After ConstantPooling\n", *graph);

  // Unroll small loops, and eliminate expressions that are the same at every
  // iteration.
  bool unroll_success = false;
  if (unroll_non_constant_loops) {
    unroll_success = UnrollLoops(graph);
    GRAPH_DEBUG("After UnrollLoops, before RemoveListMutation\n", *graph);
  } else {
    unroll_success = UnrollConstantLoops(graph);
    GRAPH_DEBUG(
        "After UnrollConstantLoops, before RemoveListMutation\n", *graph);
  }

  if (unroll_success) {
    // run again with unrolled loops
    RemoveListMutation(graph);
    GRAPH_DEBUG("After RemoveListMutation, before PeepholeOptimize\n", *graph);
    PeepholeOptimize(graph);
    GRAPH_DEBUG("After PeepholeOptimize, before ConstantPropagation\n", *graph);
    ConstantPropagation(graph);
    GRAPH_DEBUG("After ConstantPropagation\n", *graph);
  }

  EliminateCommonSubexpression(graph);
  GRAPH_DEBUG(
      "After EliminateCommonSubexpression, before HoistCommonExpression\n",
      *graph);
  HoistCommonExpression(graph);
  GRAPH_DEBUG("After HoistCommonExpression, before CheckInplace\n", *graph);
  CheckInplace(graph);
  GRAPH_DEBUG("After CheckInplace (end of runOptimization)\n", *graph);
}

Node* replaceBlockWithFallbackGraph(Block* b, ArrayRef<Value*> inputs) {
  auto graph = std::make_shared<Graph>();

  // we are copying the block inside If or prim::Loop otherwise we are copying
  // the whole graph we need to differentiate the two cases  because cloneFrom
  // automatically adds inputs if we are copying graph's block and we will
  //  need the inputs from a user otherwise
  if (b->owningNode() != nullptr) {
    std::unordered_map<Value*, Value*> input_mapping;
    auto value_map = [&input_mapping](Value* v) { return input_mapping[v]; };
    for (auto inp : inputs) {
      input_mapping[inp] = graph->block()->addInput();
    }
    graph->block()->cloneFrom(b, value_map);
  } else {
    auto value_map = [](Value* v) { return v; };
    graph->block()->cloneFrom(b, value_map);
  }

  auto fallback = b->owningGraph()->create(
      prim::FallbackGraph, inputs, b->outputs().size());
  fallback->g_(attr::Subgraph, graph);
  b->prependNode(fallback);

  for (const auto i : c10::irange(inputs.size())) {
    graph->inputs()[i]->setType(inputs[i]->type());
    graph->inputs()[i]->copyMetadata(inputs[i]);
  }

  for (const auto i : c10::irange(b->outputs().size())) {
    fallback->output(i)->setType(b->outputs()[i]->type());
    fallback->output(i)->copyMetadata(b->outputs()[i]);
    b->replaceOutput(i, fallback->output(i));
  }

  ProfilingRecord::removeProfilingNodes(graph->block());

  for (auto it = b->nodes().rbegin(); it != fallback->iterator(); it++) {
    it.destroyCurrent();
  }

  return fallback;
}

} // namespace jit
} // namespace torch
