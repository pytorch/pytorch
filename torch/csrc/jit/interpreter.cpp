#include "Python.h"
#include "interpreter.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/jit/generated/aten_dispatch.h"
#include "torch/csrc/jit/pybind.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/python_engine.h"
#include "torch/csrc/autograd/functions/special.h"
#include "torch/csrc/jit/fusion_compiler.h"

namespace py = pybind11;

namespace torch { namespace jit {

// Dummy function is the last function that the autograd engine calls
// when evaluating Eval nodes. Its input tensors are the outputs that the
// Eval node needs to produce.
// We interscept these values using an Autograd callback. So the function itself
// never runs.
struct DummyFunction : autograd::Function {
  DummyFunction() {
    num_inputs = 0;
  }
  virtual autograd::variable_list apply(const autograd::variable_list& inputs) override {
    throw std::logic_error("DummyFunction::apply() called, but it should be blocked by a callback returning false");
  }
};

// An AutogradHandle holds the information needed to run an Autograd backward pass
// after running a forward operator (such as PythonOp, CppOp, or for double-backwards another Eval Op)
// The EvalOperation uses AutogradHandle to perform this operation.
struct AutogradHandle : at::Retainable {

  // The inputs of DummyFunction are the gradients of the forward passes
  // inputs, and the _outputs_ of the run of the Autograd engine computing backward.
  // there is one entry in this list for each forward input that requires
  // gradients
  std::shared_ptr<DummyFunction> forward_inputs;

  // there is one entry in this list for each output of the forward pass
  // that represents the location in the backwaard pass where the gradient
  // of this output should be inserted at the beginning of the backward pass
  autograd::function_list forward_outputs;
};

// HandleBuilder is used to construct the correct Autograd Handle objects
// for use in a future stage.
// It is used even when the future stage does not require a handle since
// it also performs the conversions between Tensor and Variable, which
// behave differently depending on whether a future handle needs to be
// created.
struct HandleBuilder {
  HandleBuilder(bool requires_handle) {
    if(requires_handle) {
      handle = new AutogradHandle();
      handle->forward_inputs = std::make_shared<DummyFunction>();
    }
  }
  autograd::Variable addInput(at::Retainable* input, const VariableFlags & flags_) {
    if(handle && flags_.requires_grad) {
      return autograd::make_variable(
        unsafeToTensorShare(input),
        handle->forward_inputs->num_inputs++,
        handle->forward_inputs);
    } else {
      return autograd::make_variable(unsafeToTensorShare(input));
    }
  }
  at::Retainable* addOutput(const autograd::Variable & output) {
    if(handle) {
      handle->forward_outputs.emplace_back(output.grad_fn(),output.output_nr());
    }
    at::Tensor tensor = output.data();
    return toRetainableShare(output.data());
  }
  void writeTo(list_of_retainable & outputs) {
    // note: no if(handle) guard
    // because an unused handle is still produced as an output
    // outputs takes ownership of handle
    outputs.push_back(handle);
    handle = nullptr;
  }
private:
  AutogradHandle* handle = nullptr;
};

bool hasHandleOutput(Node * n) {
  if(n->outputs().size() == 0)
    return false;
  auto & last = n->outputs().back();
  return last->isHandle() && last->uses().size() > 0; // don't bother creating a handle if it is never used
}

Operation createPythonOperation(PythonOp* op) {
  py::object func = py::handle(op->pyobj.get()).attr("apply");
  bool has_handle = hasHandleOutput(op);
  return [=](const list_of_retainable & inputs, list_of_retainable & outputs) {
    AutoGIL gil;
    py::tuple py_inputs(op->cconv.size());
    size_t i = 0;
    size_t next_scalar = 0;
    size_t next_tensor = 0;
    HandleBuilder builder(has_handle);
    for(auto arg_type : op->cconv) {
      if(arg_type == 's') {
        py_inputs[i] = py::reinterpret_borrow<py::object>(op->scalar_args[next_scalar++].get());
      } else if(arg_type == 't') {
        py_inputs[i] = py::reinterpret_steal<py::object>(THPVariable_Wrap(
          builder.addInput(inputs.at(next_tensor), op->var_flags.at(next_tensor))));
        next_tensor++;
      }
      i++;
    }
    py::object py_outputs(func(*py_inputs));

    auto addOutput = [&](py::handle entry) {
      if(!THPVariable_Check(entry.ptr())) {
        throw std::runtime_error("Function.apply returned a non-Variable output");
      }
      THPVariable *var = (THPVariable*) entry.ptr();
      outputs.push_back(builder.addOutput(var->cdata));
    };
    if(!PyTuple_Check(py_outputs.ptr())) {
      addOutput(py_outputs);
    } else {
      for(py::handle entry : py::tuple(py_outputs)) {
        addOutput(entry);
      }
    }
    builder.writeTo(outputs);
  };
}

Operation createCppOperation(CppOp* op) {
  std::shared_ptr<autograd::Function> func = op->fn;
  bool has_handle = hasHandleOutput(op);
  return [=](const list_of_retainable & inputs, list_of_retainable & outputs) {
    HandleBuilder builder(has_handle);
    autograd::variable_list v_inputs;
    for(size_t i = 0; i < inputs.size(); i++) {
      v_inputs.push_back(builder.addInput(inputs[i], op->var_flags[i]));
    }
    autograd::variable_list v_outputs = (*func)(v_inputs);
    for(auto & output : v_outputs) {
      outputs.push_back(builder.addOutput(output));
    }
    builder.writeTo(outputs);
  };
}

Operation createEvalOperation(CppOp * op) {
  bool has_handle_output = hasHandleOutput(op);
  return [=](const list_of_retainable & inputs,
             list_of_retainable & outputs) {
    AutogradHandle * handle_in = dynamic_cast<AutogradHandle*>(inputs.back());
    JIT_ASSERT(handle_in);
    HandleBuilder builder(has_handle_output);
    auto& engine = autograd::python::PythonEngine::getDefaultEngine();
    autograd::variable_list v_inputs;
    for(size_t i = 0; i < inputs.size() - 1; i++) {
      v_inputs.push_back(builder.addInput(inputs[i], op->var_flags[i]));
    }
    autograd::Engine::pre_callback_map callbacks;
    callbacks.emplace(handle_in->forward_inputs.get(), [&](autograd::Function * _unused, autograd::variable_list & values) -> bool {
      for(auto & v : values) {
        outputs.push_back(builder.addOutput(v));
      }
      return false; // stop output and do not run DummyFunction
    });
    // TODO: handle create_graph appropriately
    bool create_graph = true;
    // note: node handle_in->use_count() == 1 means that we are guarenteed that we have the only
    // only copy of the handle. This might make it seem it is ok to pass keep_graph=False.
    // However, it is possible for 'copied_next_fns' to grab functions used by _other_ handles,
    // and these functions will be executed in this run. Since these other handles
    // may still be alive, it is not safe to release the graph
    engine.execute(handle_in->forward_outputs, v_inputs, true, create_graph, callbacks);
    builder.writeTo(outputs);
  };
}

using tensor_list = std::vector<at::Tensor>;
// Returns a function implementing functionality of a given node,
// or nullptr if it's a no-op for autograd.
Operation getOperation(jit::Node *node) {
  IR_IFM(node, PythonOp)
    return createPythonOperation(value);
  IR_ELSEIFM(CppOp)
    if(dynamic_cast<autograd::Eval*>(value->fn.get())) {
      return createEvalOperation(value);
    } else {
      return createCppOperation(value);
    }
  IR_ELSEIF(FusionGroup)
    auto fusion_fn = sharedFusionCompiler().getOrCompile(value);
    return [fusion_fn](const list_of_retainable & inputs, list_of_retainable & outputs) {
      autograd::profiler::RecordFunction record("FusionGroup");
      tensor_list tinputs, toutputs;
      tinputs.reserve(inputs.size());
      for(auto & i : inputs) {
        tinputs.push_back(unsafeToTensorShare(i));
      }
      fusion_fn->launch(tinputs, toutputs);
      for(auto & o : toutputs) {
        outputs.push_back(toRetainableSteal(std::move(o)));
      }
    };
  IR_ELSEIF(Constant)
    auto t = value->t(kvalue);
    return [t](const list_of_retainable & inputs, list_of_retainable & outputs) {
      outputs.push_back(toRetainableShare(t));
    };
  IR_ELSEIF(Undefined)
    return [](const list_of_retainable & inputs, list_of_retainable & outputs) {
      outputs.push_back(toRetainableSteal(at::Tensor()));
    };
  IR_ELSE()
    return getTensorOp(node).op;
  IR_END()
}


// We need some lists for inputs and outputs. To keep all the memory
// contiguous we allocate a single vector and use offsets into the vector
// which are stored in the ListHandle struct
// start is an offset into int_data of Code for ListHandle<int>
// and bool_data of Code for ListHandle<bool>
template<typename T>
struct ListHandle {
  int start;
  int size;
};

struct UseList {
  // values to be used
  ListHandle<int> values;
  // boolean flags indicating whether to free the Tensor after this use
  ListHandle<bool> free_flags;
};

// one instruction plus meta-data
struct Instruction {
  Operation callback;
  UseList inputs;
  ListHandle<int> outputs;
};


struct Stage {
  ListHandle<int> inputs; // inputs to define for the stage
  UseList outputs; // values consumed by the return
  std::vector<Instruction> instructions;
};

// pre-processing that happens once per graph
struct CodeImpl {
  CodeImpl(std::shared_ptr<Graph> & graph)
  : graph(graph) {
    int64_t cur_stage = -1;
    size_t input_pos = 0;
    size_t output_pos = 0;
    // step 1: encode all operators and stages into registers and fill in
    // input/output lists
    for(auto node : graph->nodes()) {
      insertStagesTo(cur_stage, node->stage(), input_pos, output_pos);
      cur_stage = node->stage();
      stages.back().instructions.emplace_back();
      auto & inst = stages.back().instructions.back();
      listBegin(inst.inputs.values);
      for(auto input : node->inputs()) {
        listInsert(inst.inputs.values, getOrAllocateRegister(input, true));
      }
      listBegin(inst.outputs);
      for(auto output : node->outputs()) {
        listInsert(inst.outputs, getOrAllocateRegister(output));
      }
      inst.callback = getOperation(node);
    }
    // it is possible that the final stages have no instructions in them
    // and are just identity functions. We call insertStagesTo here
    // to force all these empty stages to be generated if they exist
    insertStagesTo(cur_stage, graph->stage(), input_pos, output_pos);

    // step 2: the last time we use a register  we want to mark its free_flag
    // so we clean it up
    // this is done with a backward scan where we mark the first time we see it
    std::unordered_set<int> seen_registers;
    auto scanUses = [&](UseList & u) {
      // scan backwards because the same value may appear > once in a use list
      // and it is the last use that should free it
      std::vector<bool> free_flags(u.values.size);
      for(int i = u.values.size - 1; i >= 0; i--) {
        int reg = get(u.values,i);
        free_flags[i] = seen_registers.count(reg) == 0;
        seen_registers.insert(reg);
      }
      listBegin(u.free_flags);
      for(auto b : free_flags)
        listInsert(u.free_flags, b);
    };
    for(auto sit = stages.rbegin(); sit != stages.rend(); sit++) {
      scanUses(sit->outputs);
      for(auto iit = sit->instructions.rbegin(); iit != sit->instructions.rend(); iit++) {
        scanUses(iit->inputs);
      }
    }
  }
  void insertStagesTo(int64_t cur_stage, int64_t goal_stage, size_t & input_pos, size_t & output_pos) {
    while(cur_stage < goal_stage) {
      cur_stage++;
      stages.emplace_back();
      auto & stage = stages.back();
      listBegin(stage.inputs);
      for(;input_pos < graph->inputs().size(); input_pos++) {
        auto input = graph->inputs()[input_pos];
        if((int64_t)input->stage() > cur_stage)
          break;
        // unused inputs are given a false register -1 so that we never hold a
        // reference to the tensor data, otherwise we would fail to clean them
        // up since they do not have a last use at which to free them
        int reg = input->uses().size() > 0 ? getOrAllocateRegister(input) : -1;
        listInsert(stage.inputs, reg);
      }
      listBegin(stage.outputs.values);
      for(;output_pos < graph->outputs().size(); output_pos++) {
        auto output = graph->outputs()[output_pos];
        if((int64_t)output->stage() > cur_stage)
          break;
        listInsert(stage.outputs.values, getOrAllocateRegister(output));
      }
    }
  }
  // helpers to build/access RegList objects
  int get(ListHandle<int> & list, int i) {
    return int_data[list.start + i];
  }
  void listBegin(ListHandle<int> & list) {
    list.start = int_data.size();
    list.size = 0;
  }
  void listInsert(ListHandle<int> & list, int value) {
    JIT_ASSERTM(list.start + list.size == (int)int_data.size(), "another list already started");
    int_data.push_back(value);
    list.size++;
  }
  void listBegin(ListHandle<bool> & list) {
    list.start = bool_data.size();
    list.size = 0;
  }
  void listInsert(ListHandle<bool> & list, int value) {
    JIT_ASSERTM(list.start + list.size == (int)bool_data.size(), "another list already started");
    bool_data.push_back(value);
    list.size++;
  }

  int getOrAllocateRegister(Value * n, bool required = false) {
    size_t u = n->unique();
    if(unique_to_reg.count(u) > 0)
      return unique_to_reg[u];
    JIT_ASSERT(!required);
    int r = register_size++;
    unique_to_reg[u] = r;
    return r;
  }

  // We MUST hold onto graph here because some Operators stored in the
  // instruction lists have dependencies on meta-data stored in the graph
  // that would be dead otherwise.
  // It is also very useful for debugging interpreter problems to
  // keep this around.
  std::shared_ptr<Graph> graph;
  std::unordered_map<size_t, int> unique_to_reg; // map from unique of nodes to register in register table

  friend struct InterpreterState;
  std::vector<Stage> stages;
  int register_size = 0;

  // all memory ArrayRef<int> are slices of this, to make sure
  // the interpreter is mostly linearly scanning through memory
  std::vector<int> int_data;
  std::vector<bool> bool_data;
};

// Since the interpreter works directly with at::Retainable* objects,
// this struct is responsible for maintaining their ownership correctly.
// each non-null/non-undefined entry has a +1 reference count
// that gets released when this list is destructed or a call to release() is made
struct OwnedRetainables {
  OwnedRetainables(size_t size)
  : registers(size) {}
  OwnedRetainables(const OwnedRetainables & rhs)
  : registers(rhs.registers) {
    for(auto & r : registers) {
      if(isValid(r))
        r->retain();
    }
  }
  ~OwnedRetainables() {
    for(auto & r : registers) {
      if(isValid(r))
        r->release();
    }
  }
  at::Retainable* operator[](size_t i) {
    return registers[i];
  }

  // take ownership of 'v'
  void takeOwnership(size_t i, at::Retainable* && v) {
    JIT_ASSERT(registers[i] == nullptr);
    registers[i] = v;
    v = nullptr;
  }
  // return ownership of registers[i] to caller
  at::Retainable* detachOwnership(size_t i) {
    auto v = registers[i];
    registers[i] = nullptr;
    return v;
  }
  // release registers[i] and reset it to nullptr
  void reset(size_t i) {
    auto r = detachOwnership(i);
    if(isValid(r)) {
      r->release();
    }
  }
private:
  bool isValid(at::Retainable * r) {
    return r != nullptr && r != at::UndefinedTensor::singleton();
  }
  list_of_retainable registers;
};

// InterpreterState state that is held across stages and used to compute a Code
struct InterpreterStateImpl {
  InterpreterStateImpl(const Code & function_)
  : function(function_.pImpl),
    int_data(function->int_data.data()),
    bool_data(function->bool_data),
    registers(function->register_size) {
  }
  void runOneStage(
    const std::vector<at::Tensor> & inputs,
    std::vector<at::Tensor> & outputs) {
      // std::cout << "running stage: " << current_stage << " of " << function->stages.size() << "\n";
      JIT_ASSERT(current_stage < function->stages.size());
      auto & stage = function->stages[current_stage++];
      JIT_ASSERT((int)inputs.size() == stage.inputs.size);
      for(int i = 0; i < stage.inputs.size; i++) {
        int reg = get(stage.inputs,i);
        if(reg >= 0) { // otherwise this input is dead, and we do not store it to avoid holding the reference
          registers.takeOwnership(reg, toRetainableShare(inputs[i]));
        }
        // std::cout << "registers[" << reg << "] = inputs[" << i << "](" << registers[reg] << ")\n";
      }
      for(auto & inst : stage.instructions) {
        auto & inputs = inst.inputs.values;
        for(int i = 0; i < inputs.size; i++) {
          int reg = get(inputs,i);
          input_buffer.push_back(registers[reg]);
          // std::cout << "inputs[" << i << "] = registers[" << reg << "](" << registers[reg] << ")\n";
        }
        inst.callback(input_buffer, output_buffer);
        for(int i = 0; i < inst.outputs.size; i++) {
          int reg = get(inst.outputs,i);
          registers.takeOwnership(reg, std::move(output_buffer[i]));
          // std::cout << "registers[" << reg << "] = outputs[" << i << "](" << registers[reg] << ")\n";
        }
        auto & frees = inst.inputs.free_flags;
        for(int i = 0; i < frees.size; i++) {
          if(get(frees,i)) {
            registers.reset(get(inputs,i));
          }
        }
        output_buffer.clear();
        input_buffer.clear();
      }
      outputs.clear();
      loadTensorsFromRegisters(stage.outputs, outputs);
  }
  const TensorType & tensorTypeForInput(size_t i) const {
    size_t graph_i = i;
    for(size_t s = 0; s < current_stage; s++)
      graph_i += function->stages[s].inputs.size;
    JIT_ASSERTM(graph_i < function->graph->inputs().size(), "Input out of range");
    return *function->graph->inputs().at(graph_i)->type()->expect<TensorType>();
  }
  int get(const ListHandle<int> & list, int i) {
    return int_data[list.start + i];
  };
  bool get(const ListHandle<bool> & list, int i) {
    return bool_data[list.start + i];
  }
  void loadTensorsFromRegisters(const UseList & uses, std::vector<at::Tensor> & outputs) {
    for(int i = 0; i < uses.values.size; i++) {
      int reg = get(uses.values,i);
      // std::cout << "outputs[" << i << "] = registers[" << reg << "];\n" << registers[reg] << "\n\n";
      if(get(uses.free_flags,i)) {
        outputs.push_back(unsafeToTensorSteal(registers.detachOwnership(reg)));
      } else {
        outputs.push_back(unsafeToTensorShare(registers[reg]));
      }

    }
  }
  size_t current_stage = 0;
  std::shared_ptr<CodeImpl> function; // keep function alive
  // these are just copies of function to prevent indirections in intepreter
  int * int_data;
  const std::vector<bool> & bool_data;


  // this holds all the tensors for this interpreter run
  // we don't bother minimizing the size of this vector, since the extra
  // memory used by the pointers in this will be small
  // instead we are very aggresive about releasing tensors when they become dead
  // to make sure memory management happens efficiently.

  // We optimize for the case where derivatives are run with retain_graph=False
  // in the case where it is true, then the interpreter and this array get copied
  // if this every becomes a bottleneck then we _should_ consider minimizing the
  // total number or register
  OwnedRetainables registers;

  // single buffer for input calls to ATen functions, so that we do not reallocate
  list_of_retainable input_buffer;
  // also to prevent allocations
  list_of_retainable output_buffer;
};

Code::Code(std::shared_ptr<Graph> & graph)
: pImpl(new CodeImpl(graph)) {}
Code::~Code() {}
InterpreterState::InterpreterState(const Code & function)
: pImpl(new InterpreterStateImpl(function)) {}
InterpreterState::~InterpreterState() {}
void InterpreterState::runOneStage(
  const std::vector<at::Tensor> & inputs,
  std::vector<at::Tensor> & outputs) {
    return pImpl->runOneStage(inputs, outputs);
}
const TensorType & InterpreterState::tensorTypeForInput(size_t i) const {
  return pImpl->tensorTypeForInput(i);
}
InterpreterState InterpreterState::clone() const {
  return InterpreterState(new InterpreterStateImpl(*pImpl));
}
InterpreterState::InterpreterState(InterpreterStateImpl * pImpl) : pImpl(pImpl) {}

}}
