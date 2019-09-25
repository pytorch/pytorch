#include <torch/csrc/jit/interpreter.h>

#include <ATen/Parallel.h>
#include <ATen/core/ivalue.h>
#include <c10/core/thread_pool.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/constants.h>
#include <torch/csrc/jit/exception_message.h>
#include <torch/csrc/jit/graph_executor.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/instruction.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/passes/bailout_graph.h>
#include <torch/csrc/jit/script/compilation_unit.h>
#include <torch/csrc/jit/script/jit_exception.h>

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
// *  Computes whether a input to a node is the last use, so we can issue MOVE
//    rather than LOAD instructions.
// *  Drop nodes are inserted for any node that is unused to create a dummy use
//    that will cause the interpreter to free the node.
//    A drop node just pops its input off the stack to  ensure the interpreter
//    releases references to nodes that are never used. Drop nodes are also
//    inserted when the last use of a node is in some conditionally run control
//    flow (e.g. one side of an If) and the interpreter must free the node only
//    after the control flow has reconverged
// Outputs are:
// * graph - the post processed copy of g
// * move_flags[n] - a list of booleans, one for each input,
//   indicating whether this is the last use of the value. The interpreter
//   should generate a move rather than a copy in this case.

template <typename dtype> // int64_t, bool, double
void ListConstructFunc(int64_t num_inputs, Stack& stack);

namespace {

// insert Drop nodes to kill references for anything unused:
// this can happen in a few places, e.g. when a node returns
// many values but only one is used
// a, b = foo()
// return a
void dropUnused(Block* b) {
  auto createDropIfUnused = [&](ArrayRef<Value*> values) -> Node* {
    std::vector<Value*> to_drop;
    for (auto v : values) {
      if (v->uses().size() == 0 && v->node()->kind() != prim::Constant)
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

// ensure every value has a final use in the same block where it is defined.
// This already true for most nodes. The exceptions are:
// 1. A value that is unused.
// 2. A value whose last use is nested in some control flow.
// For (1) we simply add a prim::Drop node that uses the value right after
// it is defined. For (2), we insert a prim::Drop right after the control
// flow node where the last use occurs
void insertLastUses(Graph& g) {
  // struct to share common data structures
  struct InsertLastUses {
    Graph& graph;
    // have we seen this value, yet, if not, it is the last use of the value
    std::unordered_set<Value*> seen;

    // A map from an If or Loop node to the optional Drop block that
    // occurs directly after it to release any tensors that go out of scope
    // when the If/Loop exits. These are created and inserted on demand.
    std::unordered_map<Node*, Node*> drop_for_node;

    InsertLastUses(Graph& g) : graph(g) {
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
      // scan backwards so if a value is used twice in the list then it is a
      // move
      for (size_t i = n->inputs().size(); i > 0; --i) {
        scanUse(n, i - 1);
      }
    }
    void scanUse(Node* n, size_t i) {
      auto v = n->inputs()[i];
      auto inserted = seen.insert(v).second;
      if (!inserted) {
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

      // In the case where v and n are in the same block,
      // we have a legit final use already.
      if (same_depth_node == n) {
        return;
      }

      // in the case where the use is nested in a block
      // add a Drop node after that block which will drop 'v'.
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
      if (v->node()->kind() == prim::Constant) {
        return;
      }
      for (auto i : drop->inputs()) {
        // we already accounted for this use
        if (i == v)
          return;
      }
      drop->addInput(v);
    }
  };

  InsertLastUses ilu(g);
}
} // namespace

std::ostream& operator<<(std::ostream& out, Instruction inst);

/*
This is an optimization that reduces the number of store/load/move nodes needed
by recognizing that parts of the graph are simple trees like a*x + b*y. When
this happens it is possible to work directly off of the stack by emitting the
tree in a depth-first left-to-right manner:
  load a
  load x
  mul # stack now is a*x
  load b
  load y
  mul # stack now is a*x, b*y
  add

can_emit_inline_[node] == true means that this node participates as a non-root
member of one of these trees. The code emitter will not emit this node when
it is encountered in the node. Instead the node is emitted in a depth first
traversal from where it is used in a tree.

To participate in a tree a node must have a single use (otherwise it is not
tree-like) and output a single value (for simplicity.) If our IR was functional,
these would be the only constraints. However, many nodes have side effects, so
we must ensure that emitting the nodes in depth first order from the tree's root
_does not reorder the emission of the nodes_. To ensure this, we work backward
from the root of a potential tree, visiting its inputs in reverse depth first
order, while scanning the node list backward (with the block_point node). When
these traversal line up we know it is safe to emit the tree in this way. We
ignore constant nodes, which do not have side effects.
*/
struct CanEmitInline {
  CanEmitInline(const std::shared_ptr<Graph>& graph) {
    scanBlock(graph->block());
  }
  bool canInline(Value* v) {
    return v->node()->kind() != prim::Param && v->uses().size() == 1 &&
        v->node()->outputs().size() == 1;
  }

  Node* previousNonConstant(Node* n) {
    do {
      n = n->prev();
    } while (n->kind() == prim::Constant);
    return n;
  }

  Node* scanValue(Node* block_point, Value* v) {
    // this node is a candidate for inline, if our reverse scan of the
    // node list lines up with the use of v, we know it will be emitted in
    // tree order, and we can inlining. Scan continutes for further nodes.
    if (v->node() == block_point && canInline(v)) {
      // since we inlined this node, we may be able to recursively inline
      // its inputs, so we continue scanning it
      block_point = scanNode(v->node());
      can_emit_inline_[v->node()] = true;
    }
    // if it does not line up, we can't inline 'v', and will just generate
    // a load/move for it. However, other inputs may still appear in tree
    // order so we continue the scan of the inputs.
    return block_point;
  }

  Node* scanNode(Node* n) {
    // don't bother to scan nodes we have already determined to be inline
    if (can_emit_inline_.count(n)) {
      return nullptr;
    }
    for (auto b : n->blocks()) {
      scanBlock(b);
    }
    Node* block_point = previousNonConstant(n);
    for (auto it = n->inputs().rbegin(), end = n->inputs().rend(); it != end;
         ++it) {
      block_point = scanValue(block_point, *it);
    }
    return block_point;
  }

  void scanBlock(Block* b) {
    scanNode(b->return_node());
    for (auto node : b->nodes().reverse()) {
      scanNode(node);
    }
  }
  std::unordered_map<Node*, bool> can_emit_inline_;
};

// pre-processing that happens once per graph
struct PreprocessGraph {
  PreprocessGraph(Graph& g) : graph(g.copy()) {
    dropUnused(graph->block());
    // fill in move_flags by scanning blocks;
    insertLastUses(*graph);
    can_emit_inline = std::move(CanEmitInline(graph).can_emit_inline_);
  }
  // Outputs of the preprocessing:
  std::shared_ptr<Graph> graph;
  std::unordered_map<Node*, bool> can_emit_inline;
};

// for keeping track of the current node
struct WithCurrentNode {
  WithCurrentNode(Node** loc, Node* new_value) : loc_(loc), old_value_(*loc_) {
    *loc = new_value;
  }
  ~WithCurrentNode() {
    *loc_ = old_value_;
  }

 private:
  Node** loc_;
  Node* old_value_;
};

// BailoutBlocks are used to temporarily store
// instructions (typically, argument LOADs and TAIL_CALL)
// generated for prim::BailOut nodes
// before they are merged back into
// CodeImpl._instructions_ by insertBailoutBlocks
struct BailoutBlock {
  size_t jf_instruction_index; // this node gets patched to jump here on failure
  std::vector<Instruction> instructions; // ends in a TAIL_CALL
};

struct CodeImpl {
  friend struct InterpreterState;
  std::vector<Instruction> instructions_;

  // same length as instructions.
  // what node in the graph cause this
  // instruction to be emitted?
  std::vector<Node*> instructions_source_;

  std::vector<IValue> constant_table_;
  std::vector<Operation> operator_table_;
  // opname_table_ has the same order as operator_table_.
  std::vector<c10::OperatorName> opname_table_;
  std::vector<Function*> function_table_;
  std::vector<TypePtr> type_table_;
  int register_size_ = 0;
  size_t n_outputs;
  size_t n_inputs;
  TypePtr return_type_;

  // We MUST hold onto graph here because some Operators stored in the
  // instruction lists have dependencies on meta-data stored in the graph
  // that would be dead otherwise.
  // It is also very useful for debugging interpreter problems to
  // keep this around.
  std::shared_ptr<Graph> graph_;
  c10::optional<std::vector<GraphExecutor*>> grad_executors_;
  PreprocessGraph preprocess_;

  // map from unique of nodes to register in register table
  std::unordered_map<Value*, int> value_to_reg_;

  // running count of uses as we emit. When we reach use_count_[v] =
  // v.uses().size() we know it is the final use and we can move rather than
  // load.
  std::unordered_map<Value*, size_t> use_count_;

  Node* current_node_; // used in creation of code to keep track
                       // of node being emitted
  Node* last_inserted_op_ = nullptr;

  // out-of-line jumps for bailouts that are patched in at the end
  std::vector<BailoutBlock> bailout_blocks_;
  std::vector<std::unique_ptr<Function>> bailout_functions_;

  CodeImpl(const std::shared_ptr<Graph>& graph)
      : preprocess_(*graph), current_node_(preprocess_.graph->return_node()) {
    graph_ = preprocess_.graph;
    n_outputs = graph_->outputs().size();
    if (n_outputs == 1) {
      return_type_ = graph->outputs().at(0)->type();
    } else {
      return_type_ = TupleType::create(
          fmap(graph->outputs(), [](const Value* v) { return v->type(); }));
    }
    n_inputs = graph_->inputs().size();
    // std::cout << *graph_ << "\n";
    emitCodeForBlock(graph_->block());
    insertInstruction(RET);
    // we deferred the emission of bailout blocks so they appear at the end
    // emit them now and patch up the jumps
    insertBailoutBlocks();
  }

  const std::vector<c10::IValue>& constant_table() const {
    return constant_table_;
  }

  const std::vector<Instruction>& instructions() const {
    return instructions_;
  }

  const std::vector<c10::OperatorName>& opname_table() const {
    return opname_table_;
  }

  void insertInstruction(OpCode op, int64_t X = 0, uint64_t N = 0) {
    instructions_.emplace_back(op, X, N);
    instructions_source_.emplace_back(current_node_);

    // check that we didn't accidentally emit nodes out of topological order
    if (op == OP) {
      if (last_inserted_op_ != nullptr && current_node_ != last_inserted_op_ &&
          current_node_->owningBlock() == last_inserted_op_->owningBlock()) {
        TORCH_INTERNAL_ASSERT(
            current_node_->isAfter(last_inserted_op_),
            *current_node_,
            " is not after ",
            *last_inserted_op_);
      }
      last_inserted_op_ = current_node_;
    }
  }

  void truncateInstructions(size_t size) {
    while(instructions_.size() > size) {
      instructions_.pop_back();
      instructions_source_.pop_back();
    }
  }

  void createBailoutBlock(size_t jf_index) {
    bailout_blocks_.emplace_back(BailoutBlock{jf_index});
    auto& bailout_instructions = bailout_blocks_.back().instructions;

    bailout_instructions.insert(
        bailout_instructions.end(),
        instructions_.begin() + jf_index + 1,
        instructions_.end());
    truncateInstructions(jf_index + 1);
  }

  int allocRegs(at::ArrayRef<Value*> vs) {
    int result = register_size_ + 1;
    for (Value* v : vs) {
      AT_ASSERT(value_to_reg_.count(v) == 0);
      value_to_reg_[v] = ++register_size_;
    }
    return result;
  }

  int registerFor(Value* v) {
    return value_to_reg_.at(v);
  }

  void emitUse(Value* input, bool drop) {
    // drop - if true, we are not actually going to use this thing
    // and we can short circuit doing many instructions here
    // by either clearing the register (DROPR) or just popping the stack
    // (DROP)
    if (preprocess_.can_emit_inline[input->node()]) {
      emitNode(input->node());
      if (drop) {
        insertInstruction(DROP);
      }
    } else {
      int reg = registerFor(input);
      bool moved = input->uses().size() == ++use_count_[input];

      OpCode op;
      if (input->node()->kind() == prim::Constant) {
        op = LOADC;
      } else if (drop) {
        op = DROPR;
      } else if (moved) {
        op = MOVE;
      } else {
        op = LOAD;
      }
      insertInstruction(op, reg);
    }
  }

  void emitLoadInputs(at::ArrayRef<Value*> inputs) {
    for (Value* input : inputs) {
      emitUse(input, false);
    }
  }

  void emitOperator(Node* node) {
    emitLoadInputs(node->inputs());
    insertInstruction(OP, operator_table_.size());
    operator_table_.emplace_back(getOperation(node));
    opname_table_.emplace_back(node->schema().operator_name());
  }

  void emitWait(Node* node) {
    emitLoadInputs(node->inputs());
    insertInstruction(WAIT);
  }

  void emitDrop(at::ArrayRef<Value*> to_drop) {
    for (Value* input : to_drop) {
      emitUse(input, true);
    }
  }

  void emitStoreOutputs(Node* node) {
    size_t N = node->outputs().size();
    if (N == 0)
      return;
    int regs = allocRegs(node->outputs());
    if (N == 1) {
      insertInstruction(STORE, regs);
    } else {
      insertInstruction(STOREN, regs, node->outputs().size());
    }
  }

  int insertConstant(IValue value) {
    int result = constant_table_.size();
    constant_table_.emplace_back(std::move(value));
    return result;
  }

  void emitConstant(Node* node) {
    if (node->output()->type()->kind() == FunctionType::Kind) {
      return;
    }
    // constants are just put in the constant table
    value_to_reg_[node->output()] =
        insertConstant(toIValue(node->output()).value());
  }

  void emitIf(Node* node) {
    emitLoadInputs(node->inputs());
    size_t start_if = instructions_.size();
    insertInstruction(JF, 0); // dummy offset to be filled in
    emitCodeForBlock(node->blocks().at(0));
    insertInstruction(JMP, 0); // dummy offset
    size_t start_else = instructions_.size();
    instructions_[start_if].X = start_else - start_if;
    emitCodeForBlock(node->blocks().at(1));
    instructions_[start_else - 1].X = instructions_.size() - (start_else - 1);
  }

  void emitLoop(Node* loop) {
    insertInstruction(LOADC, insertConstant(0));
    emitLoadInputs(loop->inputs());
    size_t start = instructions_.size();
    insertInstruction(LOOP, 0, loop->inputs().size()); // dummy offset
    emitCodeForBlock(loop->blocks().at(0));
    insertInstruction(JMP, start - instructions_.size());
    instructions_[start].X = instructions_.size() - start;
  }

  void emitCall(
      Function* func,
      at::ArrayRef<Value*> inputs) {
    emitLoadInputs(inputs);
    insertInstruction(CALL, function_table_.size());
    function_table_.emplace_back(std::move(func));
  }

  void emitNodeAtBlockLevel(Node* node) {
    WithCurrentNode guard(&current_node_, node);
    switch (node->kind()) {
      case prim::Constant:
        emitConstant(node);
        break;
      case prim::Return:
        emitLoadInputs(node->inputs());
        break;
      default:
        if (!preprocess_.can_emit_inline[node]) {
          emitNode(node);
          emitStoreOutputs(node);
        }
        break;
    }
  }

  size_t emitGuard(Node* node) {
    // unoptimized graph is at index 0
    // guarded input is at index 1
    // the rest of args follow
    emitLoadInputs(node->inputs().slice(1, 1));
    insertInstruction(GUARD, type_table_.size());
    type_table_.emplace_back(node->outputs().at(0)->type());
    insertInstruction(JF, 0 /* to be patched */);

    return instructions_.size() - 1;
  }

  void emitBailOut(Node* node) {
    auto jf_index = emitGuard(node);
    auto unoptimized_graph = node->inputs().at(0)->node()->g(attr::Subgraph);
    // note, guaded input is already loaded onto the stack
    // for GUARD instruction
    emitLoadInputs(node->inputs().slice(2));
    insertInstruction(TAIL_CALL, function_table_.size());
    TORCH_INTERNAL_ASSERT(node->kind() == prim::BailOut);
    auto bailout_index = node->i(attr::index);
    TORCH_INTERNAL_ASSERT(bailout_index >= 0);

    auto build_bailout_graph = [bailout_index,
                                unoptimized_graph](Function& func) {
      BuildBailOutGraphFrom(bailout_index, unoptimized_graph, func.graph());
    };

    auto empty_graph = std::make_shared<Graph>();
    auto func = torch::make_unique<Function>(
        "bailout", empty_graph, build_bailout_graph);
    function_table_.emplace_back(func.get());
    bailout_functions_.emplace_back(std::move(func));
    createBailoutBlock(jf_index);
  }

  void emitGetAttr(Node* node) {
    emitLoadInputs(node->inputs());
    const auto type = node->input()->type()->expect<ClassType>();
    const auto& field = node->s(attr::name);
    const auto slot = type->getAttributeSlot(field);
    insertInstruction(GET_ATTR, slot);
  }

  void emitSetAttr(Node* node) {
    emitLoadInputs(node->inputs());
    const auto type = node->inputs().at(0)->type()->expect<ClassType>();
    const auto& field = node->s(attr::name);
    const auto slot = type->getAttributeSlot(field);
    insertInstruction(SET_ATTR, slot);
  }

  // Teporary solution to unblock the bytecode development.
  // TODO: remove the type specific
  void emitListConstruct(Node* node) {
    emitLoadInputs(node->inputs());
    ListTypePtr lt = node->output()->type()->expect<ListType>();
    int typeIndex = 0;
    if (IntType::get() == lt->getElementType()) {
      typeIndex = 1;
    } else if (FloatType::get() == lt->getElementType()) {
      typeIndex = 2;
    } else if (lt->getElementType() == BoolType::get()) {
      typeIndex = 3;
    } else if (lt->getElementType()->isSubtypeOf(TensorType::get())) {
      typeIndex = 4;
    }
//    TORCH_CHECK(typeIndex == 1, "LIST_CONSTRUCT type is not integer.");
    insertInstruction(LIST_CONSTRUCT, node->inputs().size(), typeIndex);
  }

  void insertBailoutBlocks() {
    for(const BailoutBlock& block : bailout_blocks_) {
      TORCH_INTERNAL_ASSERT(instructions_[block.jf_instruction_index].op == JF)
      instructions_[block.jf_instruction_index].X =
          instructions_.size() - block.jf_instruction_index;
      instructions_.insert(
          instructions_.end(),
          block.instructions.begin(),
          block.instructions.end());
      instructions_source_.insert(
          instructions_source_.end(),
          block.instructions.size(),
          instructions_source_[block.jf_instruction_index]);
    }
  }
  void emitInterfaceCall(
      std::string method_name_str,
      c10::ArrayRef<Value*> inputs) {
    emitLoadInputs(inputs);
    auto method_name = insertConstant(std::move(method_name_str));
    insertInstruction(INTERFACE_CALL, method_name, inputs.size());
  }
  void emitNode(Node* node) {
    WithCurrentNode guard(&current_node_, node);
    switch (node->kind()) {
      default:
        emitOperator(node);
        break;
      case prim::Drop:
        emitDrop(node->inputs());
        break;
      case prim::Constant:
        emitConstant(node);
        break;
      case prim::If:
        emitIf(node);
        break;
      case prim::Loop:
        emitLoop(node);
        break;
      case aten::wait:
        emitWait(node);
        break;
      case prim::Param:
        break;
      case prim::CallFunction:
        emitCall(
            node->inputs().at(0)->type()->expect<FunctionType>()->function(),
            node->inputs().slice(1));
        break;
      case prim::CallMethod:
        if (auto class_type = node->inputs().at(0)->type()->cast<ClassType>()) {
          emitCall(class_type->getMethod(node->s(attr::name)), node->inputs());
        } else {
          emitInterfaceCall(node->s(attr::name), node->inputs());
        }
        break;
      case prim::BailOut:
        emitBailOut(node);
        break;
      case prim::GetAttr:
        emitGetAttr(node);
        break;
      case prim::SetAttr:
        emitSetAttr(node);
        break;
      case prim::ListConstruct:
        emitListConstruct(node);
        break;
    }
  }

  void emitCodeForBlock(Block* block) {
    emitNodeAtBlockLevel(block->param_node());
    for (auto node : block->nodes()) {
      emitNodeAtBlockLevel(node);
    }
    emitNodeAtBlockLevel(block->return_node());
  }

  const std::vector<GraphExecutor*>& grad_executors() {
    if (!grad_executors_) {
      grad_executors_.emplace();
      for (Operation& op : operator_table_) {
        if (auto executor = detail::getGradExecutor(op)) {
          grad_executors_->push_back(executor);
        }
      }
    }
    return *grad_executors_;
  }

  void dump(std::ostream& out, size_t i) const {
    out << i << " " << instructions_[i];
    if (instructions_[i].op == OP || instructions_[i].op == CALL) {
      out << " # " << *instructions_source_[i];
    } else {
      out << "\n";
    }
  }

  void dump(std::ostream& out) const {
    out << *graph_ << "\n";
    for (size_t i = 0; i < instructions_.size(); ++i) {
      dump(out, i);
    }
  }
};

// InterpreterState state that and used to compute a Code
struct InterpreterStateImpl : c10::intrusive_ptr_target {
  InterpreterStateImpl(const Code& code) {
    enterFrame(code, 0);
  }

 private:
  // if we need to suspend, where do we reset the stack?
  // answer: to where it was when we were called, not
  // including any inputs to this function
  int64_t stack_start_ = -1;
  c10::intrusive_ptr<Future> future_;

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

  // A Frame captures function's state
  // (e.g. `pc` and `base_pointer`)
  // Each Frame corresponds to a call to a `Frame::function`
  // which has not yet returned
  // The arguments for `Frame::function`
  // are located at [base_pointer + arg_number]
  struct Frame {
    std::shared_ptr<CodeImpl> function;
    // program counter corresponds to the index
    // of the currently executed instruction
    size_t pc;
    // marks the start index of the frame
    // base_pointer is used by TAIL_CALL
    // to replace the current frame
    // with a frame of a bailout graph
    size_t base_pointer;
  };

  // saved-by-value stuff that can exist on the stack inside runInterpreter
  struct ActiveFrame {
    size_t pc;
    Instruction* instructions;
    IValue* constants;
    Operation* operators;
    Function** functions;
    TypePtr* types;

    ActiveFrame(const Frame& frame)
        : pc(frame.pc),
          instructions(frame.function->instructions_.data()),
          constants(frame.function->constant_table_.data()),
          operators(frame.function->operator_table_.data()),
          functions(frame.function->function_table_.data()),
          types(frame.function->type_table_.data()) {}
  };

  std::vector<Frame> frames;

  c10::intrusive_ptr<InterpreterStateImpl> intrusive_from_this() {
    c10::raw::intrusive_ptr::incref(this);
    return c10::intrusive_ptr<InterpreterStateImpl>::reclaim(this);
  }

  void enterFrame(const Code& code, size_t base_pointer) {
    frames.emplace_back(Frame{code.pImpl, 0, base_pointer});
    registers.resize(registers.size() + code.pImpl->register_size_);
    // frames.back().function->dump(std::cout);
  }

  void leaveFrame() {
    registers.resize(registers.size() - frames.back().function->register_size_);
    frames.pop_back();
  }

  // relative to the end of the register list so that when we call
  // functions we are referring to the registers of the currenly executing
  // function.
  IValue& reg(size_t reg) {
    return *(registers.end() - reg);
  }

  void dump(std::ostream& out, const Stack& stack) const {
    out << "Stack:\n";
    for (const auto& val : stack) {
      out << val;
      out << "\n";
    }
  }

  bool runImpl(Stack& stack) {
    // if we have never run before, then we might have to return the
    // stack when we suspend, record where it starts so we return the right
    // stack
    if (stack_start_ == -1) {
      TORCH_INTERNAL_ASSERT(stack.size() >= frames.back().function->n_inputs);
      stack_start_ = stack.size() - frames.back().function->n_inputs;
    } else {
      // during restarts, all of the stack is always our own, so we leave
      // nothing
      stack_start_ = 0;
    }

    ActiveFrame af(frames.back());
    try {
      while (true) {
//        std::cout << "RUNNING ";
//        frames.back().function->dump(std::cout, af.pc);
//        for (auto val : stack) {
//          if (val.isTensor()) {
//            std::cout << val.toTensor().sizes();
//          } else {
//            std::cout << val << std::endl;
//          }
//        }
        Instruction inst = af.instructions[af.pc];
        switch (inst.op) {
          case OP:
            af.operators[inst.X](stack);
            ++af.pc;
            break;
          case LOAD:
            stack.emplace_back(reg(inst.X));
            ++af.pc;
            break;
          case MOVE:
            stack.emplace_back(std::move(reg(inst.X)));
            ++af.pc;
            break;
          case STORE:
            reg(inst.X) = pop(stack);
            ++af.pc;
            break;
          case STOREN:
            for (size_t i = inst.N; i > 0; --i) {
              reg(inst.X + i - 1) = pop(stack);
            }
            ++af.pc;
            break;
          case DROP:
            pop(stack);
            ++af.pc;
            break;
          case DROPR:
            reg(inst.X) = IValue();
            ++af.pc;
            break;
          case LOADC:
            stack.emplace_back(af.constants[inst.X]);
            ++af.pc;
            break;
          case GET_ATTR: {
            auto userObj = pop(stack).toObject();
            auto value = userObj->getSlot(inst.X);
            if (value.isObject()) {
              auto obj = value.toObject();
              std::cout << "obj : " << obj->name() << ", "
                        << obj->slots().size() << " slots."
                        << std::endl;
            } else if (value.isTensor()) {
              auto tensor = value.toTensor();
              std::cout << "tensor with dim " << tensor.dim() << std::endl;
            }
            push(stack, std::move(value));
            ++af.pc;
          } break;
          case SET_ATTR: {
            auto v = pop(stack);
            auto userObj = pop(stack).toObject();
            userObj->setSlot(inst.X, std::move(v));
            ++af.pc;
          } break;
          case LIST_CONSTRUCT: {
            if (inst.N == 1) {
              ListConstructFunc<int64_t>(inst.X, stack);
            } else if (inst.N == 2) {
              ListConstructFunc<double>(inst.X, stack);
            } else if (inst.N == 3) {
              ListConstructFunc<bool>(inst.X, stack);
            } else if (inst.N == 4) {
              const size_t stack_size = stack.size();
              c10::List<at::Tensor> vals;
              vals.reserve(inst.X);
              for (size_t i = stack_size - inst.X; i < stack_size; ++i) {
                vals.emplace_back(std::move(stack[i]).toTensor());
              }
              drop(stack, inst.X);
              push(stack, std::move(vals));
            } else {
              const size_t stack_size = stack.size();
              auto vals = c10::impl::GenericList(c10::AnyType::get());
              vals.reserve(inst.X);
              for (size_t i = stack_size - inst.X; i < stack_size; ++i) {
                vals.emplace_back(std::move(stack[i]));
              }
              drop(stack, inst.X);
              push(stack, std::move(vals));
            }
            ++af.pc;
          } break;
          case JF:
            af.pc += (pop(stack).toBool()) ? 1 : inst.X;
            break;
          case JMP:
            af.pc += inst.X;
            break;
          case LOOP: {
            // stack: iteration_count, max_iter, cond, loop_carried_deps...
            auto frame = stack.end() - (inst.N + 1);
            int64_t trip_count = frame[0].toInt();
            int64_t max_trip_count = frame[1].toInt();
            bool cond = frame[2].toBool();
            if (trip_count < max_trip_count && cond) {
              frame[2] = trip_count;
              frame[0] = trip_count + 1;
              ++af.pc;
            } else {
              size_t n_loop_carried = inst.N - 2;
              for (size_t i = 0; i < n_loop_carried; ++i) {
                frame[i] = std::move(frame[i + 3]);
              }
              drop(stack, 3); // iteration_count, max_iter, cond
              af.pc += inst.X;
            }
          } break;
          case CALL: {
            const Code& code =
                af.functions[inst.X]->get_executor().getPlanFor(stack).code;
            frames.back().pc = af.pc + 1;
            enterFrame(code, stack.size() - code.num_inputs());
            af = ActiveFrame(frames.back());
          } break;
          case INTERFACE_CALL: {
            // note the hash table lookup to find the function
            // this can be more optimized if necessary, caching parts
            // of the hashing computation or storing the offset when
            // the object is turned into an interface
            auto function = peek(stack, 0, inst.N)
                                .toObject()
                                ->type()
                                ->getMethod(af.constants[inst.X].toStringRef());
            const Code& code = function->get_executor().getPlanFor(stack).code;
            frames.back().pc = af.pc + 1;
            enterFrame(code, stack.size() - inst.N);
            af = ActiveFrame(frames.back());
          } break;
          case RET:
            if (frames.size() > 1) {
              leaveFrame();
              af = ActiveFrame(frames.back());
              break;
            }
            if (future_) {
              auto num_outputs = frames.back().function->n_outputs;
              if (num_outputs == 1) {
                future_->markCompleted(stack.back());
              } else {
                future_->markCompleted(c10::ivalue::Tuple::create(
                    jit::last(stack, num_outputs).vec()));
              }
            }
            return false;
          case WAIT: {
            auto future = stack.back().toFuture();
            if (!future->completed()) {
              getOrCreateFuture();

              // callback needs to be a struct rather than a lambda so that
              // we can move the stack to the other thread
              struct Callback {
                Callback(
                    c10::intrusive_ptr<InterpreterStateImpl> state,
                    Stack stack)
                    : state_(std::move(state)), stack_(std::move(stack)) {}
                void operator()() {
                  at::launch(InterpreterContinuation(
                      state_,
                      std::move(stack_),
                      autograd::GradMode::is_enabled()));
                }

               private:
                InterpreterState state_;
                Stack stack_;
              };

              // we are suspending, so we need to reset the stack to where we
              // started if it started empty, except for the inputs we can avoid
              // a true copy by swapping, which leaves the original stack empty.
              Stack copied;
              if (stack_start_ == 0) {
                copied.swap(stack);
              } else {
                copied.insert(
                    copied.begin(),
                    std::make_move_iterator(stack.begin() + stack_start_),
                    std::make_move_iterator(stack.end()));
                stack.resize(stack_start_);
              }
              // save pc into the frame so we continue here when restored
              frames.back().pc = af.pc;
              future->addCallback(
                  Callback(intrusive_from_this(), std::move(copied)));

              return true;
            }
            stack.pop_back();
            stack.emplace_back(future->value());
            ++af.pc;
          } break;
          case GUARD: {
            auto t = stack.back().toTensor();
            auto actual = t.defined() ? TensorType::create(t)
                                      : TensorType::get()->withUndefined();
            const TypePtr &expected = af.types[inst.X];
            push(stack, *expected == *actual);
            ++af.pc;
          } break;
          case TAIL_CALL: {
            af.functions[inst.X]->ensure_defined();
            const Code& code =
                af.functions[inst.X]->get_executor().getPlanFor(stack).code;
            size_t num_inputs = code.num_inputs();
            size_t base_pointer = frames.back().base_pointer;
            TORCH_INTERNAL_ASSERT(stack.size() >= num_inputs);
            size_t inputs_start = stack.size() - num_inputs;
            for (size_t i = 0; i < num_inputs; ++i) {
              stack.at(base_pointer + i) =
                  std::move(stack.at(inputs_start + i));
            }
            stack.resize(base_pointer + num_inputs);
            leaveFrame();
            enterFrame(code, base_pointer);
            af = ActiveFrame(frames.back());
          } break;
        }
      }
    } catch (std::exception& e) {
      frames.back().pc = af.pc;
      bool is_jit_exception = dynamic_cast<JITException*>(&e);
      handleError(ExceptionMessage(e), is_jit_exception);
      return false;
    }
  }

  void formatStackTrace(std::ostream& out) {
    for (size_t i = 0; i < frames.size(); ++i) {
      const Frame& frame = frames[frames.size() - 1 - i];
      size_t pc = (i == 0) ? frame.pc
                           : frame.pc -
              1; // make sure we report the call node, not the node after it
      Node* node = frame.function->instructions_source_[pc];
      if (i > 0) {
        out << "during call ";
      }
      node->sourceRange().highlight(out);
    }
  }

  void handleError(const ExceptionMessage& msg, bool is_jit_exception) {
    std::stringstream ss;
    ss << msg << "\n";
    ss << "The above operation failed in interpreter, with the following stack trace:\n";
    formatStackTrace(ss);
    if (future_) {
      future_->markCompleted(Future::FutureError(ss.str()));
    } else if (is_jit_exception) {
      throw JITException(ss.str());
    } else {
      throw std::runtime_error(ss.str());
    }
  }

 public:
  c10::intrusive_ptr<Future> getOrCreateFuture() {
    if (!future_) {
      future_ =
          c10::make_intrusive<Future>(frames.front().function->return_type_);
    }
    return future_;
  }

  c10::intrusive_ptr<Future> runAsync(Stack& stack) {
    getOrCreateFuture();
    runImpl(stack);
    return future_;
  }

  void run(Stack& stack) {
    if (runImpl(stack)) {
      future_->wait();

      auto num_outputs = frames.front().function->n_outputs;
      if (num_outputs == 1) {
        push(stack, future_->value());
      } else {
        auto tuple = future_->value().toTuple();
        for (const IValue& value : tuple->elements()) {
          push(stack, value);
        }
      }
    }
  }
};

std::ostream& operator<<(std::ostream& out, const Code& code) {
  out << *code.pImpl->graph_ << "\n";
  code.pImpl->dump(out);
  return out;
}

Code::Code(const std::shared_ptr<Graph>& graph) : pImpl(new CodeImpl(graph)) {}
Code::~Code() = default;

const std::vector<GraphExecutor*>& Code::grad_executors() {
  return pImpl->grad_executors();
}

size_t Code::num_inputs() const {
  return pImpl->n_inputs;
}

size_t Code::num_outputs() const {
  return pImpl->n_outputs;
}

const std::vector<c10::IValue>& Code::constant_table() const {
  return pImpl->constant_table();
}

const std::vector<Instruction>& Code::instructions() const {
  return pImpl->instructions();
}

const std::vector<c10::OperatorName>& Code::opname_table() const {
  return pImpl->opname_table();
}

size_t Code::register_size() const {
  return pImpl->register_size_;
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
