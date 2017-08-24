#include "torch/csrc/autograd/jit_closure.h"

#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/python_function.h"
#ifdef WITH_CUDA
#include "torch/csrc/jit/fusion_compiler.h"
#endif
namespace torch { namespace autograd {

using namespace torch::jit;

// Used when an output has multiple uses (there's only one entry
// in next_functions per output).
struct Replicate : public Function {
  Replicate(int times)
    : times(times) {
    is_executable = true;
    num_inputs = 1;
  }

  virtual variable_list apply(const variable_list& inputs) {
    check_input_variables("Replicate", inputs, 1);
    return variable_list(times, inputs[0]);
  }

  int times;
};

// Used for inputs to the closure
struct Placeholder : public Function {
  virtual variable_list apply(const variable_list& inputs) {
    return inputs;
  }
};

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

// Wraps a PythonOp and dispatches calls to Functions implemented in Python
struct PythonCall : public Function {
  PythonCall(PythonOp *op)
    : pyobj()
    , is_legacy(op->is_legacy)
    , cconv(op->cconv)
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

    THPObjectPtr apply_fn;
    if (is_legacy) {
      Py_INCREF(pyobj.get());
      apply_fn = pyobj.get();
    } else {
      apply_fn = PyObject_GetAttrString(pyobj, "apply");
    }
    if (!apply_fn) throw python_error();

    THPObjectPtr py_inputs { packInputs(inputs) };
    THPObjectPtr result { PyObject_Call(apply_fn.get(), py_inputs.get(), NULL) };
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
      } else if (arg_type == 't') {
        if (var_it == inputs.end())
          throw std::runtime_error("expected too many inputs");
        obj = THPVariable_Wrap(*(var_it++));
      } else {
        throw std::runtime_error("unexpected calling convention");
      }
      Py_INCREF(obj);
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
  bool is_legacy;
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

struct SimpleEval : public Function {
  SimpleEval(const std::shared_ptr<Function>& fn)
    : fn(fn) {}

  virtual variable_list apply(const variable_list& inputs) {
    return fn->apply(inputs);
  }

  std::shared_ptr<Function> fn;
};


static variable_list variablesFromTensors(at::TensorList tensors) {
  variable_list r;
  r.reserve(tensors.size());
  for(auto & t : tensors)
    r.push_back(std::make_shared<Variable>(t,false,false));
  return r;
}


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
    return variablesFromTensors(outputs);
  }
private:
  std::shared_ptr<CompiledFusionFunction> function;
};
#endif

std::vector<at::Tensor> split(const at::Tensor & tensor, int split_size, int dim=0) {
  if(dim < 0)
    dim += tensor.dim();
  auto dim_size = tensor.size(dim);
  auto num_splits = (dim_size + split_size - 1) / split_size;
  auto last_split_size = split_size - (split_size * num_splits - dim_size);
  std::vector<at::Tensor> outputs;
  for(int i = 0; i < num_splits; i++) {
    auto sz =  (i < num_splits - 1) ? split_size : last_split_size;
    outputs.push_back(tensor.narrow(dim,i*split_size, sz));
  }
  return outputs;
}

std::vector<at::Tensor> chunk(const at::Tensor & tensor, int chunks, int dim=0) {
  if(dim < 0)
      dim += tensor.dim();
  auto split_size = (tensor.size(dim) + chunks - 1) / chunks;
  return split(tensor, split_size, dim);
}
struct ChunkFunction : public Function {
  ChunkFunction(int chunks, int dim)
  : chunks(chunks), dim(dim) {}
  virtual variable_list apply(const variable_list& inputs) {
    auto outputs = chunk(inputs[0]->data, chunks,dim);
    return variablesFromTensors(outputs);
  }
private:
  int chunks;
  int dim;
};

std::unique_ptr<AutogradClosure> createAutogradClosure(Graph *graph) {
  std::unordered_map<Node*, std::shared_ptr<Function>> node_map;
  std::unique_ptr<AutogradClosure> result {new AutogradClosure()};
  auto const_factory = std::make_shared<ConstantFactory>();
  auto& inputs = result->roots;

  // Prepare output node.
  Node *output = graph->return_node();
  result->output = std::make_shared<Output>(output->inputs().size());
  node_map[output] = result->output;

  // Builds up a closure for node. It assumes that it has been called
  // for all nodes that use outputs of node, which is why we iterate
  // in reverse topological order.
  auto add_node = [&](Node *node) {
    auto& uses = node->uses();
    auto& inputs = node->inputs();
    std::shared_ptr<Function> fn;

    if (uses.size() == 0 && node->kind() != kParam) return; // Dead code elimination

#define IR_ELSEIF_TRIVIAL(NAME, FNAME) \
    IR_ELSEIF(NAME) fn = std::make_shared<FNAME>();

    IR_IFM(node, PythonOp)
      auto name = value->name();
      // TODO: specialized ops will be probably handled by the tracer
      if (name == "Add") {
        fn = std::make_shared<Add>();
      } else if (name == "Mul") {
        fn = std::make_shared<Mul>();
      } else {
        fn = std::make_shared<PythonCall>(value);
      }
    // TODO: not sure if this works...
    IR_ELSEIFM(CppOp)
      fn = std::make_shared<SimpleEval>(value->fn);
    IR_ELSEIF(Select)
      // No-op. Selects are handled by their inputs.
      return;
    IR_ELSEIF_TRIVIAL(Add, Add)
    IR_ELSEIF_TRIVIAL(Mul, Mul)
    IR_ELSEIF(FusionGroup)
#ifdef WITH_CUDA
        auto fusion_fn = sharedFusionCompiler().getOrCompile(*value->g(kSubgraph));
        fn = std::make_shared<FusionGroupFunction>(fusion_fn);
#else
        throw std::runtime_error("don't know how to execute FusionGroups without CUDA");
#endif
    IR_ELSEIF(Param)
      fn = std::make_shared<Placeholder>();
    IR_ELSEIF(Constant)
      fn = std::make_shared<torch::autograd::WrapConstant>(value->t(kValue));
      const_factory->next_functions.emplace_back(fn, 0);
    IR_ELSEIF(Chunk)
      fn = std::make_shared<ChunkFunction>(value->i(kNumChunks),value->i(kDim));
    IR_ELSE()
      throw std::runtime_error(std::string("unrecognized NodeKind: ") + symbolToString(node->kind()));
    IR_END()

    // Update function fields
    fn->is_executable = true;
    if (node->kind() == kParam || node->kind() == kConstant) {
      fn->num_inputs = 1;
    } else {
      fn->num_inputs = inputs.size();
    }

    // Gather uses of each output
    std::vector<const use_list*> output_uses_ptrs;
    if (node->hasMultipleOutputs()) {
      // Each use is a single Select node corresponding to an output
      for (auto& use : uses) {
        if (use.user->type()->kind() == TypeKind::HandleType) continue;
        auto& select_uses = use.user->uses();
        output_uses_ptrs.emplace_back(&select_uses);
      }
    } else {
      output_uses_ptrs.emplace_back(&uses);
    }

    // Fill next_functions accordingly to uses of each output
    for (const use_list *output_uses_ptr : output_uses_ptrs) {
      const auto& output_uses = *output_uses_ptr;
      // Optimize out unnecessary Replicate nodes
      if (output_uses.size() == 1) {
        auto use = output_uses[0];
        fn->next_functions.emplace_back(node_map.at(use.user), use.offset);
      // If an output was used more than once, we need to insert a Replicate node
      // because there's only a single entry for an output in next_functions
      } else {
        auto replicate = std::make_shared<Replicate>(output_uses.size());
        for (auto& use : output_uses) {
          replicate->next_functions.emplace_back(node_map.at(use.user), use.offset);
        }
        fn->next_functions.emplace_back(std::move(replicate), 0);
      }
    }

    node_map[node] = fn;
  };

  for (auto it = graph->nodes().rbegin(), end = graph->nodes().rend(); it != end; ++it) {
    add_node(*it);
  }
  for (auto it = graph->inputs().rbegin(), end = graph->inputs().rend(); it != end; ++it) {
    add_node(*it);
  }

  // First input will always be a ConstantFactory that will trigger creation
  // of all constant tensors.
  // TODO: defer creation until they really become needed (memory saving).
  inputs.emplace_back(std::move(const_factory), 0);

  // Prepare inputs. They serve as an analog of the Select node in the IR.
  // Before the closure will be run, an Input node will be appended, and
  // inputs will be copied into its next_functions.
  for (Node *input : graph->inputs()) {
    inputs.emplace_back(node_map.at(input), 0);
  }

  return result;
}

}}
