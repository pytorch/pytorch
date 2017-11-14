#include "interpreter.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/generated/aten_dispatch.h"
#ifdef WITH_CUDA
#include "torch/csrc/jit/fusion_compiler.h"
#endif

namespace torch { namespace jit {

using tensor_list = std::vector<at::Tensor>;
using Callback = std::function<void(const tensor_list &, tensor_list &)>;
// Returns a function implementing functionality of a given node,
// or nullptr if it's a no-op for autograd.
Callback getCallback(Node *node) {
  IR_IFM(node, PythonOp)
    throw NotImplementedException();
  IR_ELSEIFM(CppOp)
    throw NotImplementedException();
  IR_ELSEIF(Select)
    barf("getCallback() on select?");
  IR_ELSEIF(FusionGroup)
#ifdef WITH_CUDA
    auto fusion_fn = sharedFusionCompiler().getOrCompile(*value->g(kSubgraph));
    return [fusion_fn](const tensor_list & inputs, tensor_list & outputs) {
      fusion_fn->launch(inputs, outputs);
    };
#else
    throw std::runtime_error("don't know how to execute FusionGroups without CUDA");
#endif
  IR_ELSEIF(Constant)
    auto t = value->t(kvalue);
    return [t](const tensor_list & inputs, tensor_list & outputs) {
      outputs.push_back(t);
    };
  IR_ELSEIF(Undefined)
    return [](const tensor_list & inputs, tensor_list & outputs) {
      outputs.push_back(at::Tensor());
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
  Callback callback;
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
      if(node->kind() == kSelect)
        continue;
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
      inst.callback = getCallback(node);
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
      listBegin(u.free_flags);
      for(int i = 0; i < u.values.size; i++) {
        int reg = get(u.values,i);
        listInsert(u.free_flags, seen_registers.count(reg) == 0);
        seen_registers.insert(reg);
      }
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

  int getOrAllocateRegister(Node * n, bool required = false) {
    size_t u = n->unique();
    if(unique_to_reg.count(u) > 0)
      return unique_to_reg[u];
    JIT_ASSERT(!required);
    int r = register_size++;
    unique_to_reg[u] = r;
    return r;
  }
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
      //std::cout << "running stage: " << current_stage << " of " << function->stages.size() << "\n";
      JIT_ASSERT(current_stage < function->stages.size());
      auto & stage = function->stages[current_stage++];
      JIT_ASSERT((int)inputs.size() == stage.inputs.size);
      for(int i = 0; i < stage.inputs.size; i++) {
        int reg = get(stage.inputs,i);
        if(reg >= 0) { // otherwise this input is dead, and we do not store it to avoid holding the reference
          registers[reg] = inputs[i];
        }
        //std::cout << "registers[" << reg << "] = inputs[" << i << "](" << inputs[i].defined() << ")\n";
      }
      for(auto & inst : stage.instructions) {
        loadTensorsFromRegisters(inst.inputs, input_buffer);
        inst.callback(input_buffer, output_buffer);
        for(int i = 0; i < inst.outputs.size; i++) {
          int reg = get(inst.outputs,i);
          registers[reg] = std::move(output_buffer[i]);
          //std::cout << "registers[" << reg << "] = outputs[" << i << "](" << output_buffer[i].defined() << ")\n";
        }
        output_buffer.clear();
        input_buffer.clear();
      }
      outputs.clear();
      loadTensorsFromRegisters(stage.outputs, outputs);
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
      auto & value = registers[reg];
      //std::cout << "inputs[" << i << "] = registers[" << reg << "] (" << value.defined() << ")";
      if(get(uses.free_flags,i)) {
        //std::cout << " and FREED";
        outputs.push_back(std::move(value));
      } else {
        outputs.push_back(value);
      }
      //std::cout << "\n";
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
  std::vector<at::Tensor> registers;

  // single buffer for input calls to ATen functions, so that we do not reallocate
  std::vector<at::Tensor> input_buffer;
  // also to prevent allocations
  std::vector<at::Tensor> output_buffer;
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
InterpreterState InterpreterState::clone() const {
  return InterpreterState(new InterpreterStateImpl(*pImpl));
}
InterpreterState::InterpreterState(InterpreterStateImpl * pImpl) : pImpl(pImpl) {}

}}
