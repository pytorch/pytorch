#include "torch/csrc/autograd/functions/jit_closure.h"

#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/functional.h"
#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/functions/special.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/autograd/functions/tensor.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/autograd/python_engine.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/python_function.h"
#ifdef WITH_CUDA
#include "torch/csrc/jit/fusion_compiler.h"
#endif
namespace torch { namespace autograd {

using namespace torch::jit;
using namespace torch::jit::tracer;

// Used when an output has multiple uses (there's only one entry
// in next_functions per output).
struct Replicate : public Function {
  Replicate() {
    is_executable = true;
    num_inputs = 1;
  }

  virtual variable_list apply(const variable_list& inputs) {
    check_input_variables("Replicate", inputs, 1);
    return variable_list(next_functions.size(), inputs[0]);
  }
};

// This class is never put in the autograd graph: see InputPlaceholder
// and EvalPlaceholder.
struct Placeholder : public Function {
  virtual variable_list apply(const variable_list& inputs) {
    return inputs;
  }
};

// Used for inputs of previous previous stages
struct PrevStageInput : public Replicate {};
// Used for inputs to the closure (execution roots)
struct InputPlaceholder : public Placeholder {};
// Used to mark places that will have to apply Evals from previous stages.
//
// Why do we need this?  Let us recall the raison d'etre of Eval nodes: they
// exist so that when we compute an backwards autograd closure on the fly
// while executing forwards, we can use exactly that closure when backwards
// executes.  Morally, this closure is simply an input to the backwards
// computation, and in the compiler IR representation, it's represented
// precisely this way (with opaque Handle nodes.)
//
// However, the *autograd* execution model only accounts for Variable
// input/outputs, which a Handle is not!  "So why not add non-Variable inputs
// to autograd"?  Perhaps this could be made to work, but it is a bit awkward:
// it would involve totally adding a new type of input to the execution model.
// Autograd is not intended to be a general purpose programming language
// runtime, so on balance, we decided to consider solutions which don't involve
// adding new types of inputs to autograd, instead passing the closures "out
// of band".
//
// By what mechanism, then, can we actually pass the closure?  Here is the idea.
// Instead of actually inserting an "Eval" node, we instead insert an
// EvalPlaceholder, which doesn't know anything about evaluating a closure.
// Then, at the time when we want to partially apply the actual closure
// (computed from the forwards pass), we stick a pre-callback on the EvalPlaceholder
// that takes the inputs, does the actual Eval, and then passes on the outputs
// (which the EvalPlaceholder subsequently passes through.)
//
// Remember that callbacks are NOT stored on a Function object itself: they are
// registered on a per AutogradClosure (for which there may be multiple per
// graph).  So we can't do something like mutate a
// Eval Function to give it the autograd closure to run inside its main body:
// that violates the invariant that autograd graphs are immutable!  (In other
// words, the same EvalPlaceholder may be participating in multiple engine
// executions) You truly must somehow associate these closures with the graph as
// a whole, and there must be a mechanism to ferry this data to the Function
// itself.  A callback is just the ticket.
struct EvalPlaceholder : public Placeholder {};

// Used for the graph output. Execution should be stopped by a callback before apply().
struct Output : public Function {
  Output(int ninputs) {
    is_executable = true;
    num_inputs = ninputs;
  }

  virtual variable_list apply(const variable_list& inputs) {
    throw std::runtime_error("Output::apply called");
  }
};

struct SimpleEval : public Function {
  SimpleEval(const std::shared_ptr<Function>& fn)
    : fn(fn) {}

  virtual variable_list apply(const variable_list& inputs) override {
    return fn->apply(inputs);
  }

  std::shared_ptr<Function> fn;
};

// Wraps a PythonOp and dispatches calls to Functions implemented in Python
struct PythonCall : public Function {
  PythonCall(PythonOp *op)
    : cconv(op->cconv)
    , scalar_args() {

    Py_INCREF(op->pyobj.get());
    pyobj = op->pyobj.get();

    scalar_args.reserve(op->scalar_args.size());
    for (auto& arg_ptr : op->scalar_args) {
      Py_INCREF(arg_ptr.get());
      scalar_args.emplace_back(arg_ptr.get());
    }
  }

  // TODO: we could probably call into some of our C functions in here to make it faster
  virtual variable_list apply(const variable_list& inputs) {
    AutoGIL gil;

    THPObjectPtr apply_fn {PyObject_GetAttrString(pyobj, "apply")};
    if (!apply_fn) throw python_error();

    THPObjectPtr py_inputs { packInputs(inputs) };
    THPObjectPtr result { PyObject_Call(apply_fn.get(), py_inputs.get(), NULL) };
    if (!result) throw python_error();
    return unpackOutputs(result);
  }

  THPObjectPtr packInputs(const variable_list& inputs) {
    THPObjectPtr py_inputs { PyTuple_New(cconv.size()) };
    if (!py_inputs) throw python_error();

    auto var_it = inputs.begin();
    auto scalar_it = scalar_args.begin();
    int input_nr = 0;

    for (auto arg_type : cconv) {
      PyObject *obj = nullptr;
      if (arg_type == 's') {
        if (scalar_it == scalar_args.end())
          throw std::runtime_error("expected too many scalar args");
        obj = (scalar_it++)->get();
        Py_INCREF(obj);
      } else if (arg_type == 't') {
        if (var_it == inputs.end())
          throw std::runtime_error("expected too many inputs");
        obj = THPVariable_Wrap(*(var_it++));
      } else {
        throw std::runtime_error("unexpected calling convention");
      }
      PyTuple_SET_ITEM(py_inputs.get(), input_nr++, obj);
    }

    return py_inputs;
  }

  variable_list unpackOutputs(THPObjectPtr& result) {
    variable_list var_result;

    ensure_tuple(result);
    auto num_outputs = PyTuple_GET_SIZE(result.get());
    for (int i = 0; i < num_outputs; ++i) {
      PyObject *output = PyTuple_GET_ITEM(result.get(), i);
      if (!THPVariable_Check(output))
        throw std::runtime_error("Function.apply returned a non-Variable output");
      THPVariable *var = (THPVariable*)output;
      var_result.emplace_back(var->cdata);
    }

    return var_result;
  }

  THPObjectPtr pyobj;
  std::string cconv;
  std::vector<THPObjectPtr> scalar_args;
};

// Note [Handling nullary functions in the autograd engine]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Today, the autograd engine cannot handle nullary functions, because
// it assumes that every non-input function has at least one input.
// This fits nicely with the scheduling strategy, which schedules a
// function for execution when all of its inputs are ready. Unfortunately,
// constants are nullary.
//
// Instead, we use a little hack. Rather than creating an extra root
// for every constant, we add a single new root, ConstantFactory, which
// when run triggers all of the actual constant functions, WrapConstant,
// which actually contribute a constant.  Furthermore, we use a single
// null input to ensure that the next_function index has a valid offset.
//
// One possible alternative to represent this might be to special case constants
// in the execution engine, as a separate vector of roots. But the current
// strategy seems to work fine and isn't too difficult to construct a trace
// for.

struct WrapConstant : public Function {
  WrapConstant(at::Tensor value)
    : value(std::move(value)) {
    is_executable = true;
    num_inputs = 1;
  }

  virtual variable_list apply(const variable_list& inputs) {
    if (inputs.size() != 1 || inputs[0])
      throw std::logic_error("WrapConstant nodes should only receive a single NULL input");
    AutoGPU guard(value.type().isCuda() ? value.get_device() : -1);
    return {std::make_shared<Variable>(value.clone(), false, false)};
  }

  at::Tensor value;
};

// See Note [Handling nullary functions in the autograd engine]
struct ConstantFactory : public Function {
  ConstantFactory() {
    is_executable = true;
    num_inputs = 1;
  }

  virtual variable_list apply(const variable_list& inputs) {
    if (inputs.size() != 1 || inputs[0])
      throw std::logic_error("ConstantFactory nodes should only receive a single NULL input");
    return variable_list(next_functions.size());
  }
};

#ifdef WITH_CUDA
struct FusionGroupFunction : public Function {
  FusionGroupFunction(const std::shared_ptr<CompiledFusionFunction> & function)
  : function(function) {}
  virtual variable_list apply(const variable_list& inputs) {
    //TODO: handle the case where inputs do not match the device function was
    // compiled for
    std::vector<at::Tensor> data;
    for(auto & input : inputs)
      data.push_back(input->data);
    AutoGPU guard(data.back());
    std::vector<at::Tensor> outputs;
    outputs.reserve(function->outputDescriptors().size());
    for(auto & od : function->outputDescriptors()) {
      outputs.push_back(at::CUDA(od.scalar_type).tensor(data.back().sizes()));
    }
    function->launch(data, outputs);
    // TODO: use wrap_outputs to properly build the graph
    throw std::runtime_error("FusionGroupFunction not implemented yet");
  }
private:
  std::shared_ptr<CompiledFusionFunction> function;
};
#endif

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// A helper struct that precomputes and caches information regarding cross-stage
// dependencies and state passing.
//
// Example:
//    graph (%1,
//           %2,
//           ------ stage 1 ------
//           %9,
//           ------ stage 2 ------
//           %31,
//           %32) {
//      %3.0, %3.1 = MulConstant(2)(%2)
//      %6.0, %6.1 = Mul()(%3.0, %1)
//      ---------------- stage 1 ----------------
//      %10.0, %10.1, %10.2 = Eval(%9, %6.1)
//      %23.0, %23.1 = Eval(%10.0, %3.1)
//      ---------------- stage 2 ----------------
//      %33.0, %33.1 = Eval(%32, %23.1)
//      %44.0, %44.1, %44.2, %44.3 = Eval(%33.0, %31, %10.2)
//      %78.0, %78.1 = Eval(%44.1, %3.1)
//      return (%6.0, %10.1, %23.0, %44.0, %44.2, %78.0);
//    }
//
// Then:
//
//    graph->stage() = 2
//    stage_begins = [%3, %10, %33, %0] (0 = return node)
//    stage_inputs = [
//      [%1, %2],
//      [%9],
//      [%31, %32]
//    ]
//    stage_outputs = [
//      [%6.0],
//      [%10.1, %23],
//      [%44.0, %44.2, %78.0]
//    ]
//    prev_stage_inputs = [
//      [],  # Always empty!
//      [%6.1, %3.1],
//      [%23.1, %10.2, %3.1]
//    ]
//    cur_stage_captures = [
//      [%6.1, %3.1],
//      [%23.1, %10.2],
//      [] # Always empty!
//    ]
struct CrossStageStateDesc {
  CrossStageStateDesc(Graph* graph)
    // FYI: graph->stage() == the last stage we have traced
    // (e.g., forwards+backwards = 1)
    : stage_inputs(graph->stage() + 1)
    , stage_outputs(graph->stage() + 1)
    , prev_stage_inputs(graph->stage() + 1)
    , cur_stage_captures(graph->stage() + 1) {

    std::size_t current_stage = -1;
    for (auto node : graph->nodes()) {
      // Look for stage boundaries
      if (node->stage() != current_stage) {
        JIT_ASSERT(node->stage() == current_stage + 1);
        current_stage = node->stage();
        stage_begins.push_back(node);
      }
      // Look for things we need to save
      for (auto input : node->inputs()) {
        if (input->stage() != current_stage) {
          JIT_ASSERT(input->stage() < current_stage);
          // We need to save it in all intermediate stages too
          for (auto i = current_stage; i > input->stage(); --i) {
            prev_stage_inputs[i].insert(input);
          }
          cur_stage_captures[input->stage()].insert(input);
        }
      }
    }

    // It's convenient to pretend output is one more stage - we can always
    // take an iterator for stage i and i+1 as loop boundaries
    stage_begins.push_back(graph->return_node());

    // Scatter inputs and outputs across stage buckets
    for (auto input : graph->inputs())
      stage_inputs[input->stage()].push_back(input);
    for (auto output : graph->outputs())
      stage_outputs[output->stage()].push_back(output);

    JIT_ASSERT(prev_stage_inputs.front().empty());
    JIT_ASSERT(cur_stage_captures.back().empty());
  }

  // For each stage, the first Node in Graph's topological sort which
  // is a member of this stage.  In general, the stages of nodes in
  // a graph will look like this:
  //
  //      000000011111112222222E (E is the Return node)
  //      ^      ^      ^      ^
  //
  // We have pointers to the caret'ed nodes.
  std::vector<Node*> stage_begins;
  std::vector<std::vector<Node*>> stage_inputs;
  std::vector<std::vector<Node*>> stage_outputs;
  // A set of all Nodes from previous stage that pass anything (both Variables
  // and handles) to current stage.
  std::vector<std::unordered_set<Node*>> prev_stage_inputs;
  // A set of all Nodes from this stage, that need their values to be captured
  // for future stages (applies to both Variables and handles).
  std::vector<std::unordered_set<Node*>> cur_stage_captures;
};

// Creates a graph for a given stage and stores information necessary to construct
// an AutogradClosure with it
struct StageClosure {
  using node_fn_map_type = std::unordered_map<Node*, std::shared_ptr<Function>>;

  StageClosure(Graph *graph, std::vector<std::pair<bool, bool>> var_flags, const CrossStageStateDesc& xstate, std::size_t stage)
    : var_flags(std::move(var_flags))
    , const_factory(std::make_shared<ConstantFactory>()) {
    node_fn_map_type node_map;
    // This map caches PrevStageInputs for a given node, so that you don't
    // create multiple PrevStageInput for the same node.
    node_fn_map_type prev_stage_input_map;

    // Prepare output node and compute an offset within return node inputs where
    // nodes from this stage apear.
    output = std::make_shared<Output>(xstate.stage_outputs[stage].size());
    node_map[graph->return_node()] = output;
    std::size_t output_offset = 0;
    for (std::size_t i = 0; i < stage; ++i)
      output_offset += xstate.stage_outputs[i].size();

    // Builds up a closure for node. It assumes that it has been called
    // for all nodes that use outputs of node, which is why we iterate
    // in reverse topological order.
    auto add_node = [&](Node *node) {
      JIT_ASSERT(node->stage() == stage);

      // Get function object
      auto fn = getFunction(node);
      if (!fn) return; // This node is a no-op

      // Initialize function fields
      fn->is_executable = true;
      if (fn->num_inputs == 0) {
        fn->num_inputs = node->inputs().size();
      }
      fillNextFunctions(node, fn, node_map, output_offset, stage);

      registerPrevStageInputs(node, fn, prev_stage_input_map);
      node_map[node] = fn;
    };

    for (auto it = std::next(xstate.stage_begins[stage+1]->reverseIterator()),
              end = std::next(xstate.stage_begins[stage]->reverseIterator()); it != end; ++it) {
      add_node(*it);
    }
    for (auto node : xstate.stage_inputs[stage]) {
      add_node(node);
    }

    // Prepare inputs.
    for (Node *input : xstate.stage_inputs[stage]) {
      roots.emplace_back(node_map.at(input), 0);
    }
    for (auto & entry : prev_stage_input_map) {
      roots.emplace_back(entry.second, 0);
      prev_stage_variables.emplace_back(entry.first->unique());
    }
    // NOTE: prev_stage_handles have been already filled in by add_node
    JIT_ASSERT(prev_stage_variables.size() + prev_stage_handles.size() == xstate.prev_stage_inputs[stage].size());

    // Prepare a list of values / handles to capture
    for (auto captured_node : xstate.cur_stage_captures[stage]) {
      if (captured_node->kind() == kSelect) {
        auto & fn = node_map.at(captured_node->input());
        if (captured_node->type()->kind() == TypeKind::TensorType) {
          captured_variables.emplace_back(fn.get(), captured_node->i(kOffset), captured_node->unique());
        } else {
          JIT_ASSERT(captured_node->type()->kind() == TypeKind::HandleType);
          captured_handles.emplace_back(fn.get(), captured_node->unique());
        }
      } else {
        JIT_ASSERT(captured_node->type()->kind() == TypeKind::TensorType);
        auto & fn = node_map.at(captured_node);
        captured_variables.emplace_back(fn.get(), 0, captured_node->unique());
      }
    }

    roots.emplace_back(const_factory, 0);
  }

  // Returns a function implementing functionality of a given node,
  // or nullptr if it's a no-op for autograd.
  std::shared_ptr<Function> getFunction(Node *node) {
    IR_IFM(node, PythonOp)
      return std::make_shared<PythonCall>(value);
    IR_ELSEIFM(CppOp)
      if (dynamic_cast<Eval*>(value->fn.get())) {
        auto fn = std::make_shared<EvalPlaceholder>();

        // All Eval nodes take context edges as an input, and we need to register
        // all such places
        auto & inputs = value->inputs();
        JIT_ASSERT(inputs.size() > 0);
        auto handle_input = inputs[inputs.size() - 1];
        JIT_ASSERT(handle_input->type()->kind() == TypeKind::HandleType);
        prev_stage_handles.emplace_back(fn.get(), handle_input->unique());

        fn->num_inputs = node->inputs().size() - 1;
        return fn;
      } else {
        return std::make_shared<SimpleEval>(value->fn);
      }
    IR_ELSEIF(Select)
      // No-op. Selects are handled by their inputs.
      return nullptr;
#define IR_ELSEIF_TRIVIAL(NAME, FNAME) IR_ELSEIF(NAME) return std::make_shared<FNAME>();
    IR_ELSEIF_TRIVIAL(Add, Add)
    IR_ELSEIF_TRIVIAL(Mul, Mul)
#undef IR_ELSEIF_TRIVIAL
    IR_ELSEIF(FusionGroup)
#ifdef WITH_CUDA
      auto fusion_fn = sharedFusionCompiler().getOrCompile(*value->g(kSubgraph));
      return std::make_shared<FusionGroupFunction>(fusion_fn);
#else
      throw std::runtime_error("don't know how to execute FusionGroups without CUDA");
#endif
    IR_ELSEIF(Param)
      auto fn = std::make_shared<InputPlaceholder>();
      fn->num_inputs = 1;
      return fn;
    IR_ELSEIF(Constant)
      auto fn = std::make_shared<torch::autograd::WrapConstant>(value->t(kvalue));
      const_factory->next_functions.emplace_back(fn, 0);
      fn->num_inputs = 1;
      return fn;
    IR_ELSEIF(Chunk)
      return std::make_shared<Chunk>(value->i(kNumChunks), value->i(kDim));
    IR_ELSE()
      throw std::runtime_error(std::string("unrecognized NodeKind: ") + symbolToString(node->kind()));
    IR_END()
  }

  // Fill in the next_functions of the Function we just allocated
  void fillNextFunctions(Node *node, const std::shared_ptr<Function>& fn, node_fn_map_type& node_map, int output_offset, std::size_t stage) {
    auto graph = node->owningGraph();
    // Gather uses of each output
    std::vector<std::reference_wrapper<const use_list>> output_uses_refs;
    if (node->hasMultipleOutputs()) {
      // Each use is a single Select node corresponding to an output
      for (auto& use : node->uses()) {
        if (use.user->type()->kind() == TypeKind::HandleType) continue;
        auto& select_uses = use.user->uses();
        output_uses_refs.emplace_back(select_uses);
      }
    } else {
      output_uses_refs.emplace_back(node->uses());
    }

    // Fill next_functions accordingly to uses of each output
    // There's some fiddling required for fixing the offset of uses for return node, so it's
    // better to keep this logic in a lambda.
    auto append_use = [&node_map, graph, output_offset](const std::shared_ptr<Function>& fn, Use& use) {
      int offset = use.offset;
      if (use.user == graph->return_node()) offset -= output_offset;
      fn->next_functions.emplace_back(node_map.at(use.user), offset);
    };
    for (auto& output_uses_ref : output_uses_refs) {
      auto output_uses = output_uses_ref.get(); // TODO: replace copy with filter
      // Filter out uses from future stages (except for output!)
      output_uses.erase(std::remove_if(output_uses.begin(), output_uses.end(),
                                       [stage, graph](Use& use) {
                                          return use.user != graph->return_node() && use.user->stage() != stage;
                                       }),
                        output_uses.end());
      // Optimize out unnecessary Replicate nodes
      if (output_uses.size() == 1) {
        append_use(fn, output_uses[0]);
      // If an output was used more than once, we need to insert a Replicate node
      // because there's only a single entry for an output in next_functions
      } else {
        auto replicate = std::make_shared<Replicate>();
        for (auto& use : output_uses) append_use(replicate, use);
        fn->next_functions.emplace_back(std::move(replicate), 0);
      }
    }
  }

  // Possibly create PrevStageInputs for any uses of nodes from previous
  // stages, and fill in their next_functions with our use.
  void registerPrevStageInputs(Node *node, const std::shared_ptr<Function>& fn,
                               node_fn_map_type& prev_stage_input_map) {
    const auto& inputs = node->inputs();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      auto input_node = inputs[i];
      if (input_node->type()->kind() == TypeKind::HandleType) continue;
      JIT_ASSERT(input_node->type()->kind() == TypeKind::TensorType);
      if (input_node->stage() < node->stage()) {
        auto info = prev_stage_input_map.emplace(input_node, nullptr);
        auto & input_fn_ptr = info.first->second;
        // Create a node if insertion took place
        if (info.second) input_fn_ptr.reset(new PrevStageInput());
        input_fn_ptr->next_functions.emplace_back(fn, i);
      }
    }
  }

  // Roots for a call to the engine. The list contains function in this order:
  // [ apply input roots | prev stage input roots | constant factory ]
  function_list roots;
  std::vector<std::pair<bool, bool>> var_flags;

  // Output node
  std::shared_ptr<Function> output;
  std::shared_ptr<ConstantFactory> const_factory;

  // These will be used by each instantiation of AutogradClosure to register hooks.
  std::vector<int> prev_stage_variables;                            // unique
  std::vector<std::pair<Function*, int>> prev_stage_handles;        // (placeholder, unique)
  // After the function is run, take either a Variable or a backward handle, and
  // put it in the environment under 'unique'.
  std::vector<std::tuple<Function*, int, int>> captured_variables;  // (function, output_nr, unique)
  std::vector<std::pair<Function*, int>> captured_handles;          // (function, unique)
};

// Computes and stores an array of StageClosures for each stage in the graph
struct MultiStageClosure {
  MultiStageClosure(TracingState* state) {
    auto graph = state->graph.get();
    CrossStageStateDesc xstate {graph};
    auto num_stages = graph->stage() + 1;
    for (std::size_t i = 0; i < num_stages; ++i) {
      stages.emplace_back(graph, state->var_flags[i], xstate, i);
    }
  }

  std::vector<StageClosure> stages;
};

AutogradClosure::AutogradClosure(const std::shared_ptr<MultiStageClosure>& desc)
  : AutogradClosure(desc, 0, {}) {}

// TODO: there's a lot processing involved in creating a new AutogradClosure instance,
// so it might be worth to keep a pool of unused instances (or at least their attrs)
// for all stages. We can't save saved_vars and saved_handles, but all callbacks
// can be made reusable.
AutogradClosure::AutogradClosure(const std::shared_ptr<MultiStageClosure>& desc, std::size_t stage, FunctionFlags &&f)
  : Function(std::move(f))
  , desc(desc)
  , stage(stage) {
  auto & stage_desc = desc->stages[stage];

  // Callbacks to capture Variables for backward closure
  for (auto & entry : stage_desc.captured_variables) {
    auto & fn = std::get<0>(entry);
    auto output_offset = std::get<1>(entry);
    auto saved_idx = std::get<2>(entry);
    post_callbacks.emplace(fn, [this, saved_idx, output_offset](Function* fn, variable_list& inputs, variable_list& outputs) {
      std::lock_guard<std::mutex> lock(this->capture_mutex);
      this->captured_vars[saved_idx] = outputs[output_offset]->data;
      return true;
    });
  }

  // Callbacks to capture handles (backward subgraphs) for backward closure
  for (auto & entry : stage_desc.captured_handles) {
    auto & fn = entry.first;
    auto saved_idx = entry.second;
    // Evals already wrap their backwards
    if (dynamic_cast<Eval*>(fn)) {
      post_callbacks.emplace(fn, [this, saved_idx](Function* fn, variable_list& inputs, variable_list& outputs) {
        auto eval_fn = Eval::getBackwardEval(inputs, outputs);
        std::lock_guard<std::mutex> lock(this->capture_mutex);
        this->captured_handles[saved_idx] = std::move(eval_fn);
        return true;
      });
    // Otherwise we have to wrap the backwards in a handle ourselves
    } else {
      post_callbacks.emplace(fn, [this, saved_idx](Function* fn, variable_list& inputs, variable_list& outputs) {
        auto eval_fn = std::make_shared<Eval>();
        eval_fn->replaceSubgraph(inputs, outputs);
        std::lock_guard<std::mutex> lock(this->capture_mutex);
        this->captured_handles[saved_idx] = std::move(eval_fn);
        return true;
      });
    }
  }

  // A callback to capture the output
  pre_callbacks.emplace(stage_desc.output.get(), [this](Function*, variable_list& inputs) {
    std::lock_guard<std::mutex> lock(this->capture_mutex);
    this->outputs.reserve(inputs.size());
    for (auto & input : inputs)
      this->outputs.emplace_back(input->data);
    return false; // Stop execution
  });

  // Callbacks that run closures received from forward
  for (auto & entry : stage_desc.prev_stage_handles) {
    int unique = entry.second;
    pre_callbacks.emplace(entry.first, [this, unique](Function* fn, variable_list& inputs) {
      inputs = (*this->saved_handles.at(unique))(inputs);
      return true;
    });
  }
}

variable_list AutogradClosure::apply(const variable_list& inputs) {
  auto& stage_closure = desc->stages[stage];

  // Validate inputs
  auto num_inputs = inputs.size();
  if (num_inputs != stage_closure.var_flags.size())
    throw std::runtime_error("AutogradClosure received an incorrect number of inputs");
  for (std::size_t i = 0; i < num_inputs; ++i) {
    auto & input = inputs[i];
    auto & flags = stage_closure.var_flags[i];
    if (input->requires_grad != flags.first || input->is_volatile != flags.second)
      throw std::runtime_error("AutogradClosure received inputs with different flags");
  }

  // TODO: we could run all this with volatile variables, but we need to somehow capture handles
  // we should enable requires_grad only for the parts that need it
  auto input_leaves = fmap(inputs, [](const std::shared_ptr<Variable>& v) {
    return std::make_shared<Variable>(v->data, v->requires_grad, v->is_volatile);
  });
  for (auto unique : desc->stages[stage].prev_stage_variables)
    input_leaves.emplace_back(std::make_shared<Variable>(saved_vars.at(unique), true, false));
  input_leaves.emplace_back(nullptr); // for ConstantFactory

  auto& engine = python::PythonEngine::getDefaultEngine();
  engine.execute(stage_closure.roots, input_leaves, true, pre_callbacks, post_callbacks);

  // TODO: append relevant inputs of previous stages to inputs!!
  auto result = wrap_outputs(inputs, std::move(outputs), [this](FunctionFlags f) -> std::shared_ptr<Function> {
    if (this->stage == this->desc->stages.size() - 1) {
      std::string msg = "JIT closure compiled only for ";
      msg += std::to_string(this->stage);
      msg += " backwards";
      return std::make_shared<Error>(std::move(msg), std::move(f));
    }
    auto bw_fn = std::shared_ptr<AutogradClosure>(new AutogradClosure(this->desc, this->stage + 1, std::move(f)));
    // TODO: don't make a full copy of saved_* - copy only the things that bw needs
    bw_fn->saved_vars = this->saved_vars;
    bw_fn->saved_vars.insert(std::make_move_iterator(this->captured_vars.begin()),
                             std::make_move_iterator(this->captured_vars.end()));
    bw_fn->saved_handles = this->saved_handles;
    bw_fn->saved_handles.insert(std::make_move_iterator(this->captured_handles.begin()),
                                std::make_move_iterator(this->captured_handles.end()));
    return bw_fn;
  });
  captured_vars.clear();
  captured_handles.clear();
  outputs.clear();
  return result;
}

AutogradClosureFactory::AutogradClosureFactory(TracingState *state)
  : desc(std::make_shared<MultiStageClosure>(state)) {}

std::shared_ptr<Function> AutogradClosureFactory::construct() {
  return std::make_shared<AutogradClosure>(desc);
}

}}
