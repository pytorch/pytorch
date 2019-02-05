#include <torch/csrc/jit/interpreter.h>

#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/variable.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/constants.h>
#include <torch/csrc/jit/graph_executor.h>
#include <torch/csrc/jit/ir.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/script/jit_exception.h>
#include <ATen/core/thread_pool.h>

#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace torch {
namespace jit {

// Before we translate to intepreter instructions, we do
// some preprocessing of the graph to turn it into a form that is closer
// to what the instructions will look like.
// In particular we:
// * (TODO) desugar Loop trip counts into c = 0, c += 1 instructions in the loop
// * Turn inputs/outputs into Load/Store instruction
// *. computes move_flags (see Outputs), and inserts
// *  Drop nodes are inserted for any node that is unused to create a dummy use
//    that will cause the interpreter to free the node.
//    A drop node is just a node with no outputs that just pops its inputs off
//    the stack, to ensure the interpreter release references to nodes that are
//    never used. Drop nodes are also inserted when the last use of a node is in
//    some conditionally run control flow (e.g. one side of an If) and the
//    interpreter must free the node only after the control flow has reconverged
// Outputs are:
// * graph - the post processed copy of g
// * move_flags[n] - a list of booleans, one for each input,
//   indicating whether this is the last use of the value. The interpreter
//   should generate a move rather than a copy in this case.

namespace {

// new_cond = (i < max_trip_count) && cond
Value* createTripCountConjunctiveCondition(
    Graph* g,
    Value* cur_trip_count,
    Value* max_trip_count,
    Value* cond) {
  // Emit initial comparison -- initial_trip_count < max_trip_count
  Value* initial_comparison_value =
      g->insertNode(g->create(aten::lt, {cur_trip_count, max_trip_count}, 1))
          ->output()
          ->setType(BoolType::get());

  // Replace initial condition with logical `and` of trip count and
  // initial condition
  Value* new_cond =
      g->insertNode(
           g->create(aten::__and__, {initial_comparison_value, cond}, 1))
          ->output()
          ->setType(BoolType::get());
  return new_cond;
}

// this currently just _removes_ the trip count inputs and checks they are
// unused. In the future they will be desugared into normal arithmetic to
// provide a loop counter
void desugarTripCounts(Block* b) {
  for (auto n : b->nodes()) {
    if (n->kind() == prim::Loop) {
      auto g = n->owningGraph();
      auto body_block = n->blocks()[0];

      Value* block_trip_count_input = body_block->inputs()[0];
      // Treat loop iteration number as a loop-carried dependency. We emit an
      // increment at the end of the body block.
      n->insertOutput(0);

      Value* max_trip_count_value = n->input(0);
      {
        WithInsertPoint guard(n);
        // int i = 0
        Value* initial_trip_count = g->insertConstant(0);
        // Set up initial iteration number value for loop-carried dependency
        n->removeInput(0);
        // Input 0 is now initial termination condition, insert this after that.
        // LCD's start at index 1.
        n->insertInput(1, initial_trip_count);

        Value* new_cond = createTripCountConjunctiveCondition(
            g, initial_trip_count, max_trip_count_value, n->input(0));
        n->replaceInput(0, new_cond);
      }

      {
        WithInsertPoint guard(body_block);
        // Trip count is now a loop carried dependency. We emit an op to
        // increment the trip count at the end of the body. Then, emit the same
        // conjunctive stopping condition as above.

        Value* const_one = g->insertConstant(1);

        Value* inc_trip_count =
            g->insertNode(
                 g->create(aten::add, {block_trip_count_input, const_one}, 1))
                ->output()
                ->setType(IntType::get());
        body_block->insertOutput(1, inc_trip_count);

        Value* body_cond = createTripCountConjunctiveCondition(
            g, inc_trip_count, max_trip_count_value, body_block->outputs()[0]);
        body_block->eraseOutput(0);
        body_block->insertOutput(0, body_cond);
      }
    }
    for (auto sb : n->blocks()) {
      desugarTripCounts(sb);
    }
  }
}

// removes all inputs and outputs to a graph, replacing them with Load Store
// nodes
static void flattenIO(Graph& graph) {
  auto load = graph.prependNode(graph.create(prim::Load, 0));
  for (auto old_input : graph.inputs()) {
    auto nv = load->addOutput();
    nv->setType(old_input->type());
    old_input->replaceAllUsesWith(nv);
  }
  graph.appendNode(graph.create(prim::Store, graph.outputs(), 0));

  while (graph.inputs().size() > 0)
    graph.eraseInput(graph.inputs().size() - 1);
  while (graph.outputs().size() > 0)
    graph.eraseOutput(graph.outputs().size() - 1);
}

// insert Drop nodes to kill references for anything unused:
// this can happen in a few places, e.g. when a node returns
// many values but only one is used
// a, b = foo()
// return a
void dropUnused(Block* b) {
  auto createDropIfUnused = [&](ArrayRef<Value*> values) -> Node* {
    std::vector<Value*> to_drop;
    for (auto v : values) {
      if (v->uses().size() == 0)
        to_drop.push_back(v);
    }
    if (to_drop.size() == 0)
      return nullptr;
    return b->owningGraph()->create(prim::Drop, to_drop, 0);
  };

  if (auto d = createDropIfUnused(b->inputs())) {
    b->prependNode(d);
  }
  for (auto n : b->nodes()) {
    if (auto d = createDropIfUnused(n->outputs())) {
      d->insertAfter(n);
    }
    for (auto b : n->blocks())
      dropUnused(b);
  }
}

// for each input, should we move rather than copy the inputs
std::unordered_map<Node*, std::vector<uint8_t>> findLastUses(Graph& g) {
  // struct to share common data structures
  struct FindLastUses {
    Graph& graph;
    // have we seen this value, yet, if not, it is the last use of the value
    std::unordered_set<Value*> seen;

    std::unordered_map<Node*, std::vector<uint8_t>> move_flags;
    // A map from an If or Loop node to the optional Drop block that
    // occurs directly after it to release any tensors that go out of scope
    // when the If/Loop exits. These are created and inserted on demand.
    std::unordered_map<Node*, Node*> drop_for_node;

    FindLastUses(Graph& g) : graph(g) {
      scanBlock(graph.block());
    }
    void scanBlock(Block* b) {
      scanNode(b->return_node());
      for (auto n : b->nodes().reverse()) {
        scanNode(n);
      }
    }
    void scanNode(Node* n) {
      for (auto b : n->blocks()) {
        scanBlock(b);
      }
      move_flags[n].resize(n->inputs().size());
      // scan backwards so if a value is used twice in the list then it is a
      // move
      for (size_t i = n->inputs().size(); i > 0; --i) {
        scanUse(n, i - 1);
      }
    }
    void scanUse(Node* n, size_t i) {
      auto& move_flags_n = move_flags[n];
      auto v = n->inputs()[i];
      auto inserted = seen.insert(v).second;
      if (!inserted) {
        move_flags_n[i] = false;
        return;
      }

      // the last use of v may be in a nested block of an If or Loop statement
      // find the node 'same_depth_node' at the same depth as the definition of
      // v, and consider that node to be the last use of v. This ensures we do
      // not delete nodes in nested scopes that may be executed multiple times
      // and that nodes used on one side of an if
      // but not the other get deleted regardless of the branch
      // e.g.
      // a = 4
      // while <...>:
      //   y = a + a
      // drop(a)
      // In other words, we find the first program point for v that
      // _reverse_ dominates the definition of v, and add a drop point there.
      Node* same_depth_node = findOwnerInBlock(n, v->node()->owningBlock());
      AT_ASSERT(
          same_depth_node); // failure means v is not in scope for n, use lint!

      // In the case where v and n are in the same block, just mark
      // its move_flags to be true
      if (same_depth_node == n) {
        move_flags_n[i] = true;
        return;
      }

      // in the case where the use is nested in a block
      // add a Drop node after that block which will drop 'v'.
      move_flags_n[i] = false;
      addToDropIfNotExists(
          findOrCreateDropInstructionForNode(same_depth_node), v);
    }

    // finds the node in block 'block' that contains in 'n'
    // or nullptr if no such node exists, e.g.:
    // n0: a = 4
    // n1: if <cond>:
    // n2:    b = a + a
    // findOwnerInBlock(n2, n0.block()) == n1
    Node* findOwnerInBlock(Node* n, Block* block) {
      while (n != nullptr && block != n->owningBlock()) {
        n = n->owningBlock()->owningNode();
      }
      return n;
    }

    Node* findOrCreateDropInstructionForNode(Node* n) {
      auto it = drop_for_node.find(n);
      if (it == drop_for_node.end()) {
        auto drop_node = graph.create(prim::Drop, 0);
        drop_node->insertAfter(n);
        it = drop_for_node.emplace(n, drop_node).first;
      }
      return it->second;
    }

    void addToDropIfNotExists(Node* drop, Value* v) {
      for (auto i : drop->inputs()) {
        // we already accounted for this use
        if (i == v)
          return;
      }
      drop->addInput(v);
      move_flags[drop].push_back(true);
    }
  };

  return FindLastUses(g).move_flags;
}
} // namespace

// pre-processing that happens once per graph
struct PreprocessGraph {
  PreprocessGraph(Graph& g) : graph(g.copy()) {
    n_outputs = graph->outputs().size();
    desugarTripCounts(graph->block());
    flattenIO(*graph);
    dropUnused(graph->block());
    // fill in move_flags by scanning blocks;
    move_flags = findLastUses(*graph);
    // TODO: desugar Loop trip counts, for now we drop trip counts
  }
  // Outputs of the preprocessing:
  std::shared_ptr<Graph> graph;
  // for each input, should we move rather than copy the inputs
  std::unordered_map<Node*, std::vector<uint8_t>> move_flags;
  // Record number of outputs before flattenIO()
  size_t n_outputs;
};

// Sometimes we want to pass things that are not tensors.  Instead of
// coming up with some "superclass" for tensor, which is annoying since
// 99% of values are at::Tensor, we instead we create a fake subclass of
// TensorImpl that can be subclassed to hold arbitrary things
// Note: this is currently unused but will probably be useful in the future,
// so we keep it around
struct ContainerTensor : public at::TensorImpl {
 public:
  ContainerTensor()
      : TensorImpl(
            at::UndefinedTensorId(),
            caffe2::TypeMeta(),
            nullptr,
            /* is_variable */ false) {}

  ~ContainerTensor() override = default;
  at::IntArrayRef sizes() const override {
    throw std::runtime_error("sizes() on ContainerTensor");
  }
  at::IntArrayRef strides() const override {
    throw std::runtime_error("strides() on ContainerTensor");
  }
  int64_t dim() const override {
    throw std::runtime_error("dim() on ContainerTensor");
  }
  const at::Storage& storage() const override {
    throw std::runtime_error("storage() on ContainerTensor");
  }
};

// We need some lists for inputs and outputs. To keep all the memory
// contiguous we allocate a single vector and use offsets into the vector
// which are stored in the ListHandle struct
// start is an offset into int_data of Code for ListHandle<int>
// and bool_data of Code for ListHandle<bool>
template <typename T>
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
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct Instruction {
  Operation callback;
  UseList inputs;
  ListHandle<int> outputs;
  Symbol debug_name; // used in dump to understand the generated code
  std::shared_ptr<SourceLocation> debug_location; // for error reporting
};

int relativeJump(int from_inst, int to_inst) {
  return to_inst - (from_inst + 1);
}

struct CodeImpl {
  CodeImpl(const std::shared_ptr<Graph>& graph_) : preprocess(*graph_) {
    graph = preprocess.graph;
    insertNodesFromBlock(graph->block());
  }

  // jump when input is false
  void createJumpFalse(int from_inst, int to_inst) {
    auto& inst = instructions[from_inst];
    AT_ASSERT(inst.debug_name == prim::Placeholder);
    auto offset = relativeJump(from_inst, to_inst);
    inst.callback = [offset](Stack& stack) {
      auto t = pop(stack).toBool();
      return t ? 0 : offset;
    };
    inst.debug_name = prim::JumpZ;
  }

  // jump when input is true
  void createJumpTrue(int from_inst, int to_inst) {
    auto& inst = instructions[from_inst];
    AT_ASSERT(inst.debug_name == prim::Placeholder);
    auto offset = relativeJump(from_inst, to_inst);
    inst.callback = [offset](Stack& stack) {
      auto t = pop(stack).toBool();
      return t ? offset : 0;
    };
    inst.debug_name = prim::JumpNZ;
  }

  void createJump(int from_inst, int to_inst) {
    auto& inst = instructions[from_inst];
    AT_ASSERT(inst.debug_name == prim::Placeholder);
    auto offset = relativeJump(from_inst, to_inst);
    inst.callback = [=](Stack& stack) { return offset; };
    inst.debug_name = prim::Jump;
  }

  void insertNodesFromBlock(Block* block) {
    for (auto node : block->nodes()) {
      const auto& source_location = node->getSourceLocation();
      switch (node->kind()) {
        case prim::If: {
          // x = if c:
          //   <then_block>
          //   -> (vt)
          // else:
          //    <else_block>
          //   -> (vf)

          // turns into:
          //   JumpNZ c, then
          //   <else_block>
          //   x = vf
          //   Jump end
          // then:
          //   <then_block>
          //   x = vt
          // end:

          // prim::Placeholder instructions are replaced with branch
          // instructions when the branch target locations are known
          auto cond_branch = insertInstruction(
              prim::Placeholder,
              source_location,
              node->inputs(),
              moveFlags(node),
              {});
          auto then_block = node->blocks()[0];
          auto else_block = node->blocks()[1];
          insertNodesFromBlock(else_block);
          insertAssign(
              source_location,
              else_block->outputs(),
              moveFlags(else_block),
              node->outputs());
          auto jump =
              insertInstruction(prim::Placeholder, source_location, {}, {}, {});
          auto then_block_start = instructions.size();
          insertNodesFromBlock(then_block);
          insertAssign(
              source_location,
              then_block->outputs(),
              moveFlags(then_block),
              node->outputs());
          createJump(jump, instructions.size());
          createJumpTrue(cond_branch, then_block_start);
        } break;
        case prim::Loop: {
          // o0 = while c i0
          //        block 0: l0
          //          <body>
          //          -> (v0, v1)

          // turns into:
          // l0 = i0
          // JumpZ c, end
          // begin:
          //   <body>
          //   c, l0 = v0, v1
          //   JumpNZ c, begin
          // end:

          auto body_block = node->blocks()[0];

          // before assign op: stack: ... <cond> <loop-carried-depdencies>
          insertAssign(
              source_location,
              node->inputs(),
              moveFlags(node),
              body_block->inputs());
          // after assign op: stack: ... <cond>
          // cond_branch consumes <cond> from top of the stack
          auto cond_branch =
              insertInstruction(prim::Placeholder, source_location, {}, {}, {});
          // after branch: stack: ...

          auto entry = instructions.size();
          insertNodesFromBlock(body_block);
          // before assign op: stack: ... <cond> <loop-carried-depdencies>
          insertAssign(
              source_location,
              body_block->outputs(),
              moveFlags(body_block),
              body_block->inputs());
          // after assign op: stack: ... <cond>
          auto cond_branch_end =
              insertInstruction(prim::Placeholder, source_location, {}, {}, {});
          // after branch: stack: ...

          aliasRegistersTo(node->outputs(), body_block->inputs());
          createJumpFalse(cond_branch, instructions.size());
          createJumpTrue(cond_branch_end, entry);
        } break;
        default: { insertInstruction(node); } break;
      }
    }
  }

  size_t insertInstruction(Node* n) {
    auto inst = insertInstruction(
        n->kind(),
        n->getSourceLocation(),
        n->inputs(),
        moveFlags(n),
        n->outputs());
    instructions[inst].callback = getOperation(n);
    return inst;
  }
  size_t insertInstruction(
      Symbol sym,
      std::shared_ptr<SourceLocation> debug_location,
      ArrayRef<Value*> inputs,
      ArrayRef<uint8_t> move_flags,
      ArrayRef<Value*> outputs) {
    instructions.emplace_back();
    auto& inst = instructions.back();
    inst.debug_name = sym;
    inst.debug_location = std::move(debug_location);
    listBegin(inst.inputs.values);
    for (auto input : inputs) {
      listInsert(inst.inputs.values, getOrAllocateRegister(input, true));
    }
    listBegin(inst.inputs.free_flags);
    for (auto flag : move_flags) {
      listInsert(inst.inputs.free_flags, flag);
    }
    listBegin(inst.outputs);
    for (auto output : outputs) {
      listInsert(inst.outputs, getOrAllocateRegister(output));
    }
    return instructions.size() - 1;
  }
  ArrayRef<uint8_t> moveFlags(Node* n) {
    return preprocess.move_flags.at(n);
  }
  ArrayRef<uint8_t> moveFlags(Block* b) {
    return moveFlags(b->return_node());
  }

  size_t insertAssign(
      std::shared_ptr<SourceLocation> debug_location,
      ArrayRef<Value*> inputs,
      ArrayRef<uint8_t> move_flags,
      ArrayRef<Value*> outputs) {
    auto inst = insertInstruction(
        prim::Assign, std::move(debug_location), inputs, move_flags, outputs);
    // This node effectively forwards its inputs into different places in a
    // register list. We don't need to manipulate the stack in any way, because
    // all inputs are also outputs, and the interpreter will take care of
    // putting them in correct places.
    instructions[inst].callback = [](Stack& stack) { return 0; };
    return inst;
  }

  // helpers to build/access RegList objects
  int get(const ListHandle<int>& list, int i) const {
    return int_data[list.start + i];
  }
  bool get(const ListHandle<bool>& list, int i) const {
    return bool_data[list.start + i];
  }
  void listBegin(ListHandle<int>& list) {
    list.start = int_data.size();
    list.size = 0;
  }
  void listInsert(ListHandle<int>& list, int value) {
    AT_CHECK(
        list.start + list.size == (int)int_data.size(),
        "another list already started");
    int_data.push_back(value);
    list.size++;
  }
  void listBegin(ListHandle<bool>& list) {
    list.start = bool_data.size();
    list.size = 0;
  }
  void listInsert(ListHandle<bool>& list, int value) {
    AT_CHECK(
        list.start + list.size == (int)bool_data.size(),
        "another list already started");
    bool_data.push_back(value);
    list.size++;
  }
  // must be called before any new_allocations are used, otherwise they will
  // already have registers assigned
  void aliasRegistersTo(
      ArrayRef<Value*> new_allocations,
      ArrayRef<Value*> existing_allocations) {
    AT_ASSERT(new_allocations.size() == existing_allocations.size());
    for (size_t i = 0; i < new_allocations.size(); ++i) {
      auto n = new_allocations[i]->unique();
      auto e = existing_allocations[i]->unique();
      AT_ASSERT(unique_to_reg.count(e) > 0 && unique_to_reg.count(n) == 0);
      unique_to_reg[n] = unique_to_reg[e];
    }
  }
  int getOrAllocateRegister(Value* n, bool required = false) {
    size_t u = n->unique();
    if (unique_to_reg.count(u) > 0)
      return unique_to_reg[u];
    AT_ASSERT(!required);
    int r = register_size++;
    unique_to_reg[u] = r;
    return r;
  }

  const std::vector<GraphExecutor*>& grad_executors() {
    if (!grad_executors_) {
      grad_executors_.emplace();
      for (Instruction& instr : instructions) {
        if (auto executor = detail::getGradExecutor(instr.callback)) {
          grad_executors_->push_back(executor);
        }
      }
    }
    return *grad_executors_;
  }

  void dumpInstruction(std::ostream& out, size_t pc) const {
    auto writeList = [&](const ListHandle<int>& list) {
      for (int i = 0; i < list.size; i++) {
        if (i > 0)
          out << ", ";
        out << get(list, i);
      }
    };
    auto writeUseList = [&](const UseList& list) {
      for (int i = 0; i < list.values.size; i++) {
        if (i > 0)
          out << ", ";
        if (get(list.free_flags, i))
          out << "move(" << get(list.values, i) << ")";
        else
          out << get(list.values, i);
      }
    };
    auto& inst = instructions.at(pc);
    writeList(inst.outputs);
    // NB: debug names are the kind of operator used to select
    // dispatch
    out << " = " << inst.debug_name.toUnqualString() << " ";
    writeUseList(inst.inputs);
  }
  void dump(std::ostream& out) const {
    for (size_t i = 0; i < instructions.size(); ++i) {
      dumpInstruction(out, i);
      out << "\n";
    }
  }

  // We MUST hold onto graph here because some Operators stored in the
  // instruction lists have dependencies on meta-data stored in the graph
  // that would be dead otherwise.
  // It is also very useful for debugging interpreter problems to
  // keep this around.
  std::shared_ptr<Graph> graph;
  c10::optional<std::vector<GraphExecutor*>> grad_executors_;
  PreprocessGraph preprocess;

  std::unordered_map<size_t, int>
      unique_to_reg; // map from unique of nodes to register in register table

  friend struct InterpreterState;
  std::vector<Instruction> instructions;
  int register_size = 0;

  // all memory ArrayRef<int> are slices of this, to make sure
  // the interpreter is mostly linearly scanning through memory
  std::vector<int> int_data;
  std::vector<bool> bool_data;
};

// InterpreterState state that and used to compute a Code
struct InterpreterStateImpl : c10::intrusive_ptr_target {
  InterpreterStateImpl(const Code& code)
      : function(code.pImpl),
        int_data(function->int_data.data()),
        bool_data(function->bool_data),
        registers(function->register_size) {}

 private:
  c10::intrusive_ptr<InterpreterStateImpl> intrusive_from_this() {
    c10::raw::intrusive_ptr::incref(this);
    return c10::intrusive_ptr<InterpreterStateImpl>::reclaim(this);
  }

  bool runImpl(Stack& stack) {
    auto& instructions = function->instructions;
    size_t last = instructions.size();

    while (pc < last) {
      // std::cout << "executing " << pc << ": ";
      // function->dumpInstruction(std::cout, pc);
      // std::cout << "\n";
      auto& inst = instructions[pc];
      try {
        loadTensorsFromRegisters(inst.inputs, stack);
        size_t new_pc = pc + 1 + inst.callback(stack);
        for (int i = inst.outputs.size - 1; i >= 0; --i) {
          int reg = get(inst.outputs, i);
          registers[reg] = pop(stack);
          // std::cout << "pop reg[" << reg << "];\n" << registers[reg] << "\n";
        }
        pc = new_pc;
      } catch (Suspend& e) {
        // wait() expects a single input
        AT_ASSERT(inst.inputs.values.size == 1);

        getOrCreateFuture();

        if (get(inst.inputs.free_flags, 0)) {
          // make sure the register is not freed once we are waked up
          registers[get(inst.inputs.values, 0)] = e.future;
        }

        // Make sure adding callback is the last step.
        // Otherwise if e.future has completed,
        // the current thread will continue running before it suspends.
        InterpreterState state(intrusive_from_this());
        e.future->addCallback([state]() {
          c10::global_work_queue().run(InterpreterContinuation(state, Stack(),
              autograd::GradMode::is_enabled()));
        });

        return true;
      } catch (Future::FutureError& e) {
        // Error from the forked thread.
        auto msg = e.error_msg; // copy the error for each callback
        handleError(std::move(msg), false);
        return false;
      } catch (std::exception& e) {
        // Error from the current thread
        bool is_jit_exception = dynamic_cast<JITException*>(&e);
        if (instructions[pc].debug_location) {
          handleError(
              instructions[pc].debug_location->wrapException(
                  e, "operation failed in interpreter"),
              is_jit_exception);
        } else {
          handleError(e.what(), is_jit_exception);
        }
        return false;
      }
    }
    if (future) {
      auto num_outputs = function->preprocess.n_outputs;
      if (num_outputs == 1) {
        future->markCompleted(stack.back());
      } else {
        future->markCompleted(
            Tuple::create(jit::last(stack, num_outputs).vec()));
      }
    }

    return false;
  }

  void handleError(std::string&& error_msg, bool is_jit_exception) {
    if (future) {
      future->markCompleted(Future::FutureError(std::move(error_msg)));
    } else if (is_jit_exception) {
      throw JITException(std::move(error_msg));
    } else {
      throw std::runtime_error(std::move(error_msg));
    }
  }

 public:
  c10::intrusive_ptr<Future> getOrCreateFuture() {
    if (!future) {
      future = c10::make_intrusive<Future>();
    }
    return future;
  }

  c10::intrusive_ptr<Future> runAsync(Stack& stack) {
    getOrCreateFuture();
    runImpl(stack);
    return future;
  }

  void run(Stack& stack) {
    if (runImpl(stack)) {
      future->wait();

      auto num_outputs = function->preprocess.n_outputs;
      if (num_outputs == 1) {
        push(stack, future->value());
      } else {
        auto tuple = future->value().toTuple();
        for (const auto& value : tuple->elements()) {
          push(stack, value);
        }
      }
    }
  }

  int get(const ListHandle<int>& list, int i) {
    return int_data[list.start + i];
  };
  bool get(const ListHandle<bool>& list, int i) {
    return bool_data[list.start + i];
  }
  void loadTensorsFromRegisters(const UseList& uses, Stack& stack) {
    for (int i = 0; i < uses.values.size; i++) {
      int reg = get(uses.values, i);
      // std::cout << "push reg[" << reg << "];\n" << registers[reg] << "\n\n";
      if (get(uses.free_flags, i)) {
        stack.push_back(std::move(registers[reg]));
      } else {
        stack.push_back(registers[reg]);
      }
    }
  }

  // pc is critical for the interperter to pick up the progress from suspend
  size_t pc = 0;
  c10::intrusive_ptr<Future> future;
  std::shared_ptr<CodeImpl> function; // keep function alive
  // these are just copies of function to prevent indirections in interpreter
  int* int_data;
  const std::vector<bool>& bool_data;

  // this holds all the tensors for this interpreter run
  // we don't bother minimizing the size of this vector, since the extra
  // memory used by the pointers in this will be small
  // instead we are very aggresive about releasing tensors when they become dead
  // to make sure memory management happens efficiently.

  // We optimize for the case where derivatives are run with retain_graph=False
  // in the case where it is true, then the interpreter and this array get
  // copied if this every becomes a bottleneck then we _should_ consider
  // minimizing the total number or register
  std::vector<IValue> registers;

  // single buffer for input/output calls to ATen functions, so that we do not
  // reallocate
  Stack stack;
};

std::ostream& operator<<(std::ostream& out, const Code& code) {
  out << *code.pImpl->graph << "\n";
  code.pImpl->dump(out);
  return out;
}

Code::Code(const std::shared_ptr<Graph>& graph) : pImpl(new CodeImpl(graph)) {}
Code::~Code() = default;

const std::vector<GraphExecutor*>& Code::grad_executors() {
  return pImpl->grad_executors();
}

InterpreterState::InterpreterState(const Code& code)
    : pImpl(c10::make_intrusive<InterpreterStateImpl>(code)) {}
InterpreterState::~InterpreterState() = default;

void InterpreterState::run(Stack& stack) {
  static_cast<InterpreterStateImpl*>(pImpl.get())->run(stack);
}

c10::intrusive_ptr<Future> InterpreterState::runAsync(Stack& stack) {
  return static_cast<InterpreterStateImpl*>(pImpl.get())->runAsync(stack);
}

c10::intrusive_ptr<Future> InterpreterState::getFuture() {
  return static_cast<InterpreterStateImpl*>(pImpl.get())->getOrCreateFuture();
}

InterpreterState::InterpreterState(
    c10::intrusive_ptr<c10::intrusive_ptr_target> pImpl_)
    : pImpl(std::move(pImpl_)) {}

void InterpreterContinuation::operator()() {
  autograd::AutoGradMode grad_mode(grad_mode_enabled);
  state.runAsync(stack);
}
} // namespace jit
} // namespace torch
