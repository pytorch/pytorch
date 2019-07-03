#include <torch/csrc/jit/script/convert_to_ssa.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/passes/inline_forked_closures.h>
#include <torch/csrc/jit/script/compiler.h>
#include <torch/csrc/jit/script/mini_environment.h>

namespace torch {
namespace jit {
namespace script {

// At the beginning of the pass the Graph has already undergone type checking,
// and writes or reads to a variable are emitted as Loads and Stores in the
// graph. a = 1 print(a) is represented as:
//
// %a.1 : int = prim::Constant[value=1]()
// prim::Store[name="a"](%a.1)
// %a : int = prim::Load[name="a"]()
// prim::Print(%a)
//
// First, this pass recursively adds the Loads & Stores to control flow nodes
// Then the graph is converted to SSA form.

using ValueEnvironment = MiniEnvironment<Value*>;
using TypeEnvironment = MiniEnvironment<TypePtr>;

// Adds Loads & Stores to Loops & Ifs
struct ControlFlowLoadStores {
  static void addBlockInput(
      Block* b,
      const TypePtr& type,
      const std::string& name) {
    auto g = b->owningGraph();
    g->createStore(name, b->addInput(name)->setType(type))
        ->insertAfter(b->param_node());
  }

  static void addBlockOutput(
      Block* b,
      const TypePtr& type,
      const std::string& name) {
    WithInsertPoint insert(b);
    auto g = b->owningGraph();
    auto block_output = g->insertNode(g->createLoad(name, type))->output();
    b->registerOutput(block_output);
  }

  static void addNodeOutput(
      Node* n,
      const TypePtr& type,
      const std::string& name) {
    auto out = n->addOutput()->setType(type);
    if (meaningfulName(name)) {
      out->setDebugName(name);
    }
    auto g = n->owningGraph();
    g->createStore(name, out)->insertAfter(n);
  }

  static void addNodeInput(
      Node* n,
      const TypePtr& type,
      const std::string& name) {
    auto g = n->owningGraph();
    auto inp = g->createLoad(name, type)->insertBefore(n)->output();
    n->addInput(inp);
  }

  void addIfLoadStores(Node* n) {
    auto true_block = n->blocks().at(0);
    auto false_block = n->blocks().at(1);

    auto true_vars = addControlFlowLoadStores(true_block);
    auto false_vars = addControlFlowLoadStores(false_block);
    std::set<std::string> mutated_variables;

    for (auto& v : true_vars->definedVariables()) {
      if (false_vars->findInAnyFrame(v)) {
        mutated_variables.insert(v);
      }
    }
    for (auto& v : false_vars->definedVariables()) {
      if (true_vars->findInAnyFrame(v)) {
        mutated_variables.insert(v);
      }
    }

    // Following the same logic as emitIfElseBlocks in compiler.cpp,
    // we emit a node output if the variable is defined in each block
    // and the types of each block can be unified

    for (const auto& x : mutated_variables) {
      auto true_type = true_vars->findInAnyFrame(x);
      auto false_type = false_vars->findInAnyFrame(x);
      auto unified = unifyTypes(true_type, false_type);
      if (!unified) {
        continue;
      }

      addBlockOutput(true_block, true_type, x);
      addBlockOutput(false_block, false_type, x);
      addNodeOutput(n, *unified, x);
    }
  }

  // loop_carried_outputs* = Loop(max_trip_count, start_condition,
  //                              loop_carried_inputs*)
  //                    block0(loop_counter, loop_carried_block*) {
  //                       <body>
  //                       -> (continue_condition, loop_carried_block_outputs*)
  //                    }
  // all loop_carried_... lists are the same length and represent the value of
  // loop-carried variables whose definitions are updated as the loop executes
  // in a way that ensure single static assignment.
  void addLoopLoadStores(Node* n) {
    auto body_block = n->blocks().at(0);
    auto loop_vars = addControlFlowLoadStores(body_block);
    for (const auto& name : loop_vars->definedVariables()) {
      // we require that the variable is defined outside the loop to be emitted,
      // and we do not refine the type of the parent variable since the loop may
      // not be entered.
      auto parent_type = environment_stack->findInAnyFrame(name);
      if (!parent_type) {
        continue;
      }

      // Insert a store at the beginning of the loop block, so that all
      // loads of the variable will use the loop carried value
      addNodeInput(n, parent_type, name);
      addBlockInput(body_block, parent_type, name);
      addBlockOutput(body_block, parent_type, name);
      addNodeOutput(n, parent_type, name);
    }
  }

  std::shared_ptr<TypeEnvironment> addControlFlowLoadStores(Block* block) {
    pushFrame(block);
    for (Node* n : block->nodes()) {
      switch (n->kind()) {
        case prim::If: {
          addIfLoadStores(n);
        } break;
        case prim::Loop: {
          addLoopLoadStores(n);
        } break;
        case prim::Function: {
          for (auto b : n->blocks()) {
            addControlFlowLoadStores(b);
          }
        } break;
        case prim::Store: {
          environment_stack->setVar(n->s(attr::name), n->input()->type());
        } break;
      }
    }
    return popFrame();
  }

  void pushFrame(Block* b) {
    environment_stack = std::make_shared<TypeEnvironment>(b, environment_stack);
  }

  std::shared_ptr<TypeEnvironment> popFrame() {
    auto old_frame = environment_stack;
    environment_stack = environment_stack->next;
    return old_frame;
  }

  void run(std::shared_ptr<Graph>& graph) {
    addControlFlowLoadStores(graph->block());
  }

  std::shared_ptr<TypeEnvironment> environment_stack = nullptr;
};

void moveBlockBeforeNode(Node* before_node, Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    auto block_node = *it++;
    block_node->moveBefore(before_node);
  }
}

// The loop node is initially emitted as:
// Loop(max_trip_count)
//    block0(loop_counter) {
//      <body>
//    }
//    block1 {
//      <loop condition>
//      -> (condition)
//    }
// Here, we inline the loop condition and convert the loop to the form:
// Loop(max_trip_count, start_condition)
//    block0(loop_counter, loop_carried_block*) {
//      <body>
//      -> (continue_condition)
//    }
void inlineLoopCondition(Node* n) {
  Block* body_block = n->blocks().at(0);

  auto pre_header = n->blocks().at(1);
  auto header_block = n->addBlock();
  header_block->cloneFrom(pre_header, [](Value* v) { return v; });
  moveBlockBeforeNode(n, header_block);
  n->addInput(header_block->outputs().at(0));
  n->eraseBlock(2);

  moveBlockBeforeNode(body_block->return_node(), pre_header);
  body_block->insertOutput(0, pre_header->outputs().at(0));
  n->eraseBlock(1);
}

void inlineLoopCondition(Block* block) {
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      inlineLoopCondition(b);
    }
    if (n->kind() == prim::Loop) {
      inlineLoopCondition(n);
    }
  }
}

// Given a graph where outputs have been added to control flow nodes, and
// loads and stores are represented in the graph, converts the graph to SSA
struct SSATransformer {
  void convertBlockToSSA(Block* block) {
    pushFrame(block);
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      auto n = *it;
      it++;
      switch (n->kind()) {
        case prim::If:
        case prim::Loop:
        case prim::Function: {
          for (auto b : n->blocks()) {
            convertBlockToSSA(b);
          }
        } break;
        case prim::Store: {
          environment_stack->setVar(n->s(attr::name), n->input());
          n->destroy();
        } break;
        case prim::Load: {
          auto name = n->s(attr::name);
          auto var = environment_stack->findInAnyFrame(name);
          TORCH_INTERNAL_ASSERT(
              var, "Typechecking should ensure the variable name is set");
          n->output()->replaceAllUsesWith(var);
          n->destroy();
        } break;
      }
    }
    popFrame();
  }

  void pushFrame(Block* b) {
    environment_stack =
        std::make_shared<ValueEnvironment>(b, environment_stack);
  }

  std::shared_ptr<ValueEnvironment> popFrame() {
    auto old_frame = environment_stack;
    environment_stack = environment_stack->next;
    return old_frame;
  }

  void run(std::shared_ptr<Graph>& graph) {
    convertBlockToSSA(graph->block());
  }

  std::shared_ptr<ValueEnvironment> environment_stack = nullptr;
};

// Converting to SSA works in multiple parts. First we inline the loop condition
// before and into the body of loops, then we add outputs to control flow
// nodes, then we stitch together Loads & Stores into SSA form.
void ConvertToSSA(std::shared_ptr<Graph>& graph) {
  inlineLoopCondition(graph->block());
  ControlFlowLoadStores ctrl;
  ctrl.run(graph);
  SSATransformer ssa;
  ssa.run(graph);
}

} // namespace script
} // namespace jit
} // namespace torch
