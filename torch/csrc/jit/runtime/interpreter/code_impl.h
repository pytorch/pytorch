#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include <c10/util/irange.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/bailout_graph.h>
#include <torch/csrc/jit/runtime/calculate_necessary_args.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/runtime/interpreter/preprocess_graph.h>

namespace torch {
namespace jit {

std::ostream& operator<<(std::ostream& out, Instruction inst);

namespace interpreter {

template <class Ttarget, class Tsource>
Ttarget safe_narrow_cast(Tsource v) {
  Ttarget res = static_cast<Ttarget>(v);
  // Casting it back to check whether it overflew.
  if (static_cast<Tsource>(res) != v) {
    TORCH_WARN(
        "ATTENTION: your model computation is overflowing, safe_narrow_cast<>() failed");
    return v;
  }
  return res;
}

// BailoutBlocks are used to temporarily store
// instructions (typically, argument LOADs and TAIL_CALL)
// generated for prim::BailOut nodes
// before they are merged back into
// CodeImpl._instructions_ by insertBailoutBlocks
struct BailoutBlock {
  size_t jf_instruction_index; // this node gets patched to jump here on failure
  std::vector<Instruction> instructions; // ends in a TAIL_CALL

  explicit BailoutBlock(size_t jf_index) : jf_instruction_index(jf_index) {}
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

struct CodeImpl {
  friend struct InterpreterState;
  std::vector<Instruction> instructions_;

  // same length as instructions.
  // what node in the graph cause this
  // instruction to be emitted?
  std::vector<Node*> instructions_source_;

  std::vector<IValue> constant_table_;
  std::vector<Operation> operator_table_;
#ifndef NDEBUG
  std::vector<Operator> full_operator_table_;
#endif
  // map<(op name, num inputs), index in operator table>, to avoid duplicates,
  // not including vararg operators
  std::unordered_map<
      std::pair<std::string, int>,
      int,
      std::function<size_t(const std::pair<std::string, int>& p)>>
      operator_table_inv_;
  std::vector<Function*> function_table_;
  std::vector<std::unique_ptr<GraphFunction>> forked_functions_;
  std::vector<std::unique_ptr<GraphFunction>> awaited_functions_;
  std::vector<TypePtr> type_table_;
  std::vector<std::function<void(std::vector<IValue>&)>>
      profile_function_table_;

  int register_size_ = 0;
  size_t n_outputs;
  size_t n_inputs;
  TypePtr return_type_;
  std::string function_name_;

  // We MUST hold onto graph here because some Operators stored in the
  // instruction lists have dependencies on meta-data stored in the graph
  // that would be dead otherwise.
  // It is also very useful for debugging interpreter problems to
  // keep this around.
  std::shared_ptr<Graph> graph_;
  c10::optional<std::vector<GraphExecutor*>> grad_executors_;
  c10::optional<std::vector<GraphExecutor*>> forward_executors_;
  PreprocessGraph preprocess_;

  // map from unique of nodes to register in register table
  std::unordered_map<Value*, int> value_to_reg_;

  // map from operator name to specified arguments
  // Example: for a schema of aten::foo.str
  // aten::foo.str(arg0: str="default", arg1: int=0,
  //               arg2: bool=False, arg3: float=0.0)
  // If the usages in a graph is:
  //    aten::foo("somestr", arg1=0, arg2=True, arg3=0.0)
  //    aten::foo("somestr", arg1=1, arg2=False, arg3=0.0)
  // op_to_num_specified_args_["aten::foo.str"] = 3
  // This is because for all usages, at most 3 args are used.
  std::unordered_map<std::string, size_t> op_to_num_specified_args_;

  std::unordered_map<std::string, size_t> op_to_num_out_args_;

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
  size_t remaining_bailout_depth_;

  CodeImpl(
      const std::shared_ptr<Graph>& graph,
      std::string function_name,
      size_t remaining_bailout_depth,
      bool emit_instructions = true)
      : operator_table_inv_(
            0,
            [](const std::pair<std::string, int>& p) {
              return std::hash<std::string>()(p.first) ^
                  std::hash<int>()(p.second);
            }),
        function_name_(std::move(function_name)),
        preprocess_(*graph),
        current_node_(preprocess_.graph->return_node()),
        remaining_bailout_depth_(remaining_bailout_depth) {
    graph_ = preprocess_.graph;
    n_outputs = graph_->outputs().size();
    if (n_outputs == 1) {
      return_type_ = graph->outputs().at(0)->type();
    } else {
      return_type_ = TupleType::create(
          fmap(graph->outputs(), [](const Value* v) { return v->type(); }));
    }
    n_inputs = graph_->inputs().size();
    if (emit_instructions) {
      // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
      run();
    }
  }

  virtual ~CodeImpl() = default;

  // since subclass of CodeImpl needs to populate
  // op_to_num_specified_args, we separate the calls
  // that changes internals of CodeImpl into a separate
  // function.
  virtual void run() {
    emitCodeForBlock(graph_->block());
    insertInstruction(RET);
    // we deferred the emission of bailout blocks so they appear at the end
    // emit them now and patch up the jumps
    insertBailoutBlocks();
  }

  const std::vector<c10::IValue>& constant_table() const {
    return constant_table_;
  }

  void request_bailout(size_t index) {
    auto count = index;
    for (const auto instr_index : c10::irange(instructions_.size())) {
      if (instructions_[instr_index].op == GUARD ||
          instructions_[instr_index].op == FAIL_GUARD) {
        if (count-- == 0) {
          // patching GUARD to FAIL_GUARD
          instructions_[instr_index].op = FAIL_GUARD;
          GRAPH_DEBUG(
              "Added a bailout request for ",
              index,
              " at instruction ",
              instr_index);
          break;
        }
      }
    }
  }

  const std::vector<Instruction>& instructions() const {
    return instructions_;
  }

  const std::unordered_map<std::string, size_t>& op_to_num_specified_args()
      const {
    return op_to_num_specified_args_;
  }

  const std::vector<Node*>& instructions_source() const {
    return instructions_source_;
  }

  void insertInstruction(OpCode op, int64_t X = 0, uint64_t N = 0) {
    instructions_.emplace_back(
        op,
        safe_narrow_cast<int32_t, int64_t>(X),
        safe_narrow_cast<uint16_t, uint64_t>(N));
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
    while (instructions_.size() > size) {
      instructions_.pop_back();
      instructions_source_.pop_back();
    }
  }

  void createBailoutBlock(size_t jf_index) {
    bailout_blocks_.emplace_back(jf_index);
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

      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      OpCode op;
      if (input->node()->kind() == prim::Constant) {
        op = LOADC;
      } else if (moved) {
        op = MOVE;
      } else {
        op = LOAD;
      }

      if (drop) {
        op = DROPR;
      }
      insertInstruction(op, reg);
    }
  }

  void emitLoadInputs(at::ArrayRef<Value*> inputs) {
    for (Value* input : inputs) {
      emitUse(input, false);
    }
  }

  void emitLoadInputs(at::ArrayRef<Value*> inputs, int num_include) {
    int count = 0;
    for (Value* input : inputs) {
      if (count < num_include) {
        emitUse(input, false);
        count++;
      }
    }
  }

  void emitLoadInputs(at::ArrayRef<Value*> inputs, size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      emitUse(inputs[i], false);
    }
  }

  virtual void emitOperator(Node* node) {
    emitLoadInputs(node->inputs());
    const Operator& op = node->getOperator();
    int num_inputs = node->inputs().size();
    bool is_vararg = op.schema().is_vararg();

    int operation_index = add_to_operator_table(
        op,
        node,
        c10::toString(op.schema().operator_name()),
        num_inputs,
        is_vararg);

    if (op.hasOperation() && is_vararg) {
      insertInstruction(OPN, operation_index, num_inputs);
    } else {
      insertInstruction(OP, operation_index);
    }
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
    if (N == 0) {
      return;
    }
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

  virtual void emitOperatorOrInstruction(
      Node* node,
      OpCode op,
      int64_t X = 0,
      uint64_t N = 0,
      bool emit_inputs = true) {
    if (emit_inputs) {
      emitLoadInputs(node->inputs());
    }
    insertInstruction(op, X, N);
  }

  void emitFormat(Node* node) {
    emitOperatorOrInstruction(node, FORMAT, node->inputs().size(), 0);
  }

  void checkNodeAndEmit(Node* node) {
    // check if the node should be emitted as instruction or operator
    const Operator& op = node->getOperator();
    std::string unique_op_name = c10::toString(op.schema().operator_name());
    if (unique_op_name.find("aten::__getitem__.Dict") == 0) {
      // __get_item__ overloaded operator for Dict
      // needs to be emitted an instruction
      emitOperatorOrInstruction(node, DICT_INDEX);
    } else {
      emitOperator(node);
    }
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

  void emitCall(Function* func, at::ArrayRef<Value*> inputs) {
    emitLoadInputs(inputs);
    insertInstruction(CALL, function_table_.size());
    function_table_.emplace_back(func);
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

  size_t emitType(TypePtr t) {
    size_t r = type_table_.size();
    type_table_.emplace_back(std::move(t));
    return r;
  }

  void emitTypeCheck(Node* node) {
    auto num_inputs = node->inputs().size();

    // Check that TypeCheck has at least one input.
    TORCH_INTERNAL_ASSERT(
        num_inputs && num_inputs + 1 == node->outputs().size());
    emitLoadInputs(node->inputs());

    // Emit the expected type.
    size_t types_start = type_table_.size();
    auto types = node->tys(attr::types);
    for (const auto i : c10::irange(num_inputs)) {
      emitType(types[i]);
    }
    insertInstruction(TYPECHECK, types_start, num_inputs);
  }

  size_t emitGuard(Node* node) {
    // unoptimized graph is at index 0
    // guarded input is at index 1
    // the rest of args follow
    emitLoadInputs(node->inputs().slice(1, 1));
    insertInstruction(GUARD, emitType(node->outputs().at(0)->type()));
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
                                unoptimized_graph](GraphFunction& func) {
      BuildBailOutGraphFrom(bailout_index, unoptimized_graph, func.graph());
    };

    auto empty_graph = std::make_shared<Graph>();
    auto func = std::make_unique<GraphFunction>(
        "bailout", empty_graph, build_bailout_graph);
    function_table_.emplace_back(func.get());
    bailout_functions_.emplace_back(std::move(func));
    createBailoutBlock(jf_index);
  }

  void emitProfile(Node* node) {
    emitLoadInputs(node->inputs());
    insertInstruction(PROFILE_OP, profile_function_table_.size());
    if (node->cast<ProfileOp>()) {
      profile_function_table_.push_back(node->cast<ProfileOp>()->getCallback());
    } else if (node->cast<ProfileIValueOp>()) {
      profile_function_table_.push_back(
          node->cast<ProfileIValueOp>()->getCallback());
    } else {
      TORCH_INTERNAL_ASSERT(false);
    }
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

  void insertBailoutBlocks() {
    for (const BailoutBlock& block : bailout_blocks_) {
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

  void emitListUnpack(Node* node) {
    emitLoadInputs(node->inputs());
    insertInstruction(LIST_UNPACK, node->outputs().size());
  }

  void emitTupleConstruct(Node* node) {
    bool named =
        node->output()->type()->expectRef<TupleType>().name().has_value();
    if (named) {
      emitContainerConstruct(NAMED_TUPLE_CONSTRUCT, node);
    } else {
      emitLoadInputs(node->inputs());
      insertInstruction(TUPLE_CONSTRUCT, node->inputs().size());
    }
  }

  void emitContainerConstruct(OpCode op, Node* node) {
    emitLoadInputs(node->inputs());
    insertInstruction(
        op, emitType(node->output()->type()), node->inputs().size());
  }

  void emitCreateObject(Node* node) {
    insertInstruction(CREATE_OBJECT, emitType(node->output()->type()));
  }
  void emitIsinstance(Node* node) {
    emitLoadInputs(node->inputs());
    std::vector<TypePtr> types = node->tys(attr::types);
    size_t types_start = type_table_.size();
    for (const auto& typ : types) {
      emitType(typ);
    }
    insertInstruction(ISINSTANCE, types_start, types.size());
  }

  void emitTupleSlice(Node* node) {
    emitLoadInputs(node->inputs());
    int64_t beg_ind = node->i(attr::beg);
    int64_t end_ind = node->i(attr::end);
    insertInstruction(TUPLE_SLICE, beg_ind, end_ind - beg_ind);
  }

  void emitFork(Node* node) {
    emitLoadInputs(node->inputs());
    std::unique_ptr<GraphFunction> forked_fn(new GraphFunction(
        "<forked function>", node->g(attr::Subgraph), nullptr));
    forked_functions_.emplace_back(std::move(forked_fn));
    function_table_.emplace_back(forked_functions_.back().get());
    insertInstruction(FORK, function_table_.size() - 1, node->inputs().size());
  }

  void emitAwaitable(Node* node) {
    emitLoadInputs(node->inputs());
    std::unique_ptr<GraphFunction> await_fn(new GraphFunction(
        "<awaitable function>", node->g(attr::Subgraph), nullptr));
    awaited_functions_.emplace_back(std::move(await_fn));
    function_table_.emplace_back(awaited_functions_.back().get());
    insertInstruction(
        AWAITABLE, function_table_.size() - 1, node->inputs().size());
  }

  void emitWarn(Node* node) {
    if (FLAGS_torch_jit_disable_warning_prints) {
      return;
    }

    emitLoadInputs(node->inputs());
    int32_t idx = -1;
    if (node->hasAttribute(attr::warn_id)) {
      idx = static_cast<int32_t>(node->i(attr::warn_id));
    }
    insertInstruction(WARN, idx);
  }

  void emitEnter(Node* node) {
    emitLoadInputs(node->inputs());
    insertInstruction(ENTER);
  }

  void emitExit(Node* /* node */) {
    insertInstruction(EXIT);
  }

  void emitNode(Node* node) {
    WithCurrentNode guard(&current_node_, node);
    switch (node->kind()) {
      default:
        // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
        checkNodeAndEmit(node);
        // emitOperator(node);
        break;
      case prim::RaiseException:
        emitOperatorOrInstruction(node, RAISE_EXCEPTION);
        break;
      case prim::TupleIndex:
        emitOperatorOrInstruction(node, TUPLE_INDEX);
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
            node->inputs().at(0)->type()->expectRef<FunctionType>().function(),
            node->inputs().slice(1));
        break;
      case prim::CallMethod:
        if (auto class_type = node->inputs().at(0)->type()->cast<ClassType>()) {
          emitCall(&class_type->getMethod(node->s(attr::name)), node->inputs());
        } else {
          emitInterfaceCall(node->s(attr::name), node->inputs());
        }
        break;
      case prim::TypeCheck:
        emitTypeCheck(node);
        break;
      case prim::BailOut:
        emitBailOut(node);
        break;
      case prim::profile_ivalue:
      case prim::profile:
        emitProfile(node);
        break;
      case prim::GetAttr:
        emitGetAttr(node);
        break;
      case prim::SetAttr:
        emitSetAttr(node);
        break;
      case prim::ListUnpack:
        emitListUnpack(node);
        break;
      case prim::TupleConstruct:
        emitTupleConstruct(node);
        break;
      case prim::ListConstruct:
        emitContainerConstruct(LIST_CONSTRUCT, node);
        break;
      case prim::DictConstruct:
        emitContainerConstruct(DICT_CONSTRUCT, node);
        break;
      case prim::CreateObject:
        emitCreateObject(node);
        break;
      case prim::isinstance:
        emitIsinstance(node);
        break;
      case prim::TupleSlice:
        emitTupleSlice(node);
        break;
      case prim::fork:
        emitFork(node);
        break;
      case prim::awaitable:
        emitAwaitable(node);
        break;
      case aten::warn:
        emitWarn(node);
        break;
      case prim::Enter:
        emitEnter(node);
        break;
      case prim::Exit:
        emitExit(node);
        break;
      case prim::Uninitialized:
        emitOperatorOrInstruction(node, UN_INITIALIZED, 0, 0, false);
        break;
      case prim::dtype:
        emitOperatorOrInstruction(node, DTYPE);
        break;
      case prim::device:
        emitOperatorOrInstruction(node, DEVICE);
        break;
      case aten::dim:
        emitOperatorOrInstruction(node, DIM);
        break;
      case prim::is_cuda:
        emitOperatorOrInstruction(node, IS_CUDA);
        break;
      case aten::__not__:
        emitOperatorOrInstruction(node, __NOT__);
        break;
      case aten::format:
        emitFormat(node);
        break;
      case aten::__is__:
        emitOperatorOrInstruction(node, __IS__);
        break;
      case aten::__isnot__:
        emitOperatorOrInstruction(node, __ISNOT__);
        break;
      case prim::NumToTensor:
        emitOperatorOrInstruction(node, NUM_TO_TENSOR);
        break;
      case prim::tolist:
        emitOperatorOrInstruction(node, TO_LIST);
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

  const std::vector<GraphExecutor*>& diff_graph_op_executors() {
    if (!forward_executors_) {
      forward_executors_.emplace();
      for (Operation& op : operator_table_) {
        if (auto executor = detail::getDifferentiableGraphOpExecutor(op)) {
          forward_executors_->push_back(executor);
        }
      }
    }
    return *forward_executors_;
  }

  void dump(std::ostream& out, size_t i) const {
    out << i << " " << instructions_[i];
    if (instructions_[i].op == OP || instructions_[i].op == CALL ||
        instructions_[i].op == OPN) {
      out << " # " << *instructions_source_[i];
    } else {
      out << "\n";
    }
  }

  void dump(std::ostream& out) const {
    out << *graph_ << "\n";
    for (const auto i : c10::irange(instructions_.size())) {
      dump(out, i);
    }
  }

  /**
   * Add an operation to operator_table_ if not a duplicate and return its index
   */
  int add_to_operator_table(
      const Operator& op,
      const Node* node,
      const std::string& op_name,
      const int num_inputs,
      const bool is_vararg) {
    int size = operator_table_.size();

    const Operation& oper = op.getOperation(node);

    if (!is_vararg) {
      std::pair<std::string, int> key(op_name, num_inputs);
      auto found = operator_table_inv_.find(key);

      if (found != operator_table_inv_.end()) {
        return found->second;
      }

      operator_table_inv_.emplace(key, size);
    }

    operator_table_.emplace_back(oper);
#ifndef NDEBUG
    full_operator_table_.emplace_back(op);
#endif
    return size;
  }

  inline void assert_stack_size(
      int32_t instruction_index,
      size_t init_size,
      size_t actual_size) const {
#ifndef NDEBUG
    const auto& schema = full_operator_table_[instruction_index].schema();
    int64_t expected_size = static_cast<int64_t>(init_size) -
        static_cast<int64_t>(schema.arguments().size()) +
        static_cast<int64_t>(schema.returns().size());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        static_cast<size_t>(expected_size) == actual_size ||
            schema.is_varret() || schema.is_vararg(),
        "Expected to find ",
        expected_size,
        " values on the stack, but found ",
        actual_size,
        " on the stack after ",
        toString(full_operator_table_[instruction_index].schema()));
#endif
  }
};

struct MobileCodeImpl : CodeImpl {
  MobileCodeImpl(
      const std::shared_ptr<Graph>& graph,
      std::string function_name,
      bool emit_default_input_instructions,
      bool support_default_args_before_out,
      bool emit_promoted_ops,
      size_t remaining_bailout_depth)
      : CodeImpl(graph, function_name, remaining_bailout_depth, false),
        emit_default_input_instructions_(emit_default_input_instructions),
        support_default_args_before_out_(support_default_args_before_out),
        emit_promoted_ops_(emit_promoted_ops) {
    // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
    run();
  }

  void run() override {
    process_ops_for_mobile();
    emitCodeForBlock(graph_->block());
    insertInstruction(RET);
    // we deferred the emission of bailout blocks so they appear at the end
    // emit them now and patch up the jumps
    insertBailoutBlocks();
  }

  void process_ops_for_mobile() {
    DepthFirstGraphNodeIterator graph_it(graph_);
    Node* node = graph_it.next();
    while (node) {
      if (node->maybeOperator()) {
        auto op_schema = node->getOperator().schema();
        // skip if schema has vararg
        if (!op_schema.is_vararg()) {
          auto specifiedArgs = CalculateNecessaryArgs(
              op_schema.arguments(),
              node->inputs(),
              support_default_args_before_out_);

          size_t numInclude = specifiedArgs.first +
              (support_default_args_before_out_ ? specifiedArgs.second : 0);
          auto unique_name = !op_schema.overload_name().empty()
              ? op_schema.name() + "." + op_schema.overload_name()
              : op_schema.name();
          auto it = op_to_num_specified_args_.insert(
              std::pair<std::string, size_t>(unique_name, 0));
          op_to_num_out_args_.insert(std::pair<std::string, size_t>(
              unique_name, specifiedArgs.second));
          auto prev_value = it.first->second;
          it.first->second = std::max(numInclude, prev_value);
        }
      }
      node = graph_it.next();
    }
  }

 private:
  void emitOperator(Node* node) override {
    if (emit_default_input_instructions_) {
      CodeImpl::emitOperator(node);
    } else {
      const Operator& op = node->getOperator();
      std::string unique_op_name = c10::toString(op.schema().operator_name());
      int num_inputs = node->inputs().size();
      bool is_vararg = op.schema().is_vararg();

      if (op.hasOperation() && is_vararg) {
        emitLoadInputs(node->inputs());
        int operation_index = add_to_operator_table(
            op,
            node,
            unique_op_name,
            num_inputs,
            /* is_vararg */ true);
        insertInstruction(OPN, operation_index, num_inputs);
      } else {
        auto num_include = num_inputs;
        auto it = op_to_num_specified_args_.find(unique_op_name);
        if (it != op_to_num_specified_args_.end()) {
          num_include = it->second;
        }
        if (support_default_args_before_out_) {
          auto num_out = op_to_num_out_args_.find(unique_op_name)->second;
          auto num_specified_before_out = num_include - num_out;
          emitLoadInputs(node->inputs(), 0, num_specified_before_out);
          emitLoadInputs(
              node->inputs(),
              node->inputs().size() - num_out,
              node->inputs().size());
        } else {
          emitLoadInputs(node->inputs(), num_include);
        }
        int operation_index = add_to_operator_table(
            op, node, unique_op_name, num_inputs, is_vararg);
        insertInstruction(OP, operation_index);
      }
    }
  }

  void emitOperatorOrInstruction(
      Node* node,
      OpCode op,
      int64_t X = 0,
      uint64_t N = 0,
      bool emit_inputs = true) override {
    if (emit_promoted_ops_) {
      CodeImpl::emitOperatorOrInstruction(node, op, X, N, emit_inputs);
    } else {
      CodeImpl::emitOperator(node);
    }
  }

  // To support forward compatibility for bytecode version bump from v5 to v6
  bool emit_default_input_instructions_;
  // To support forward compatibility for bytecode version bump from v6 to v7
  bool support_default_args_before_out_;
  // To support forward compatibility for bytecode version bump from v7 to v8
  bool emit_promoted_ops_;
};

} // namespace interpreter
} // namespace jit
} // namespace torch
