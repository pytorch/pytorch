#include <torch/csrc/jit/script/convert_to_ssa.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/ir_views.h>
#include <torch/csrc/jit/passes/inline_forked_closures.h>
#include <torch/csrc/jit/script/compiler.h>
#include <torch/csrc/jit/script/exit_transforms.h>
#include <torch/csrc/jit/script/inline_loop_condition.h>
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

  static void addBlockExitInput(
      Node* exit_node,
      const TypePtr& type,
      const std::string& name) {
    WithInsertPoint insert(exit_node);
    auto g = exit_node->owningGraph();
    auto block_exit = g->insertNode(g->createLoad(name, type))->output();
    exit_node->addInput(block_exit);
  }

  static void addBlockOutput(
      Block* exit_block,
      const TypePtr& type,
      const std::string& name) {
    WithInsertPoint insert(exit_block);
    auto g = exit_block->owningGraph();
    auto block_exit = g->insertNode(g->createLoad(name, type))->output();
    exit_block->registerOutput(block_exit);
  }

  static void addNodeOutput(
      Node* n,
      const TypePtr& type,
      const std::string& name) {
    auto out = n->addOutput()->setType(type);
    if (meaningfulName(name)) {
      out->setUniqueName(name);
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

    auto true_block_exit = n->owningGraph()
                               ->create(prim::BlockExit, 0)
                               ->insertBefore(true_block->return_node());
    auto false_block_exit = n->owningGraph()
                                ->create(prim::BlockExit, 0)
                                ->insertBefore(false_block->return_node());

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

      addBlockExitInput(true_block_exit, true_type, x);
      addBlockExitInput(false_block_exit, false_type, x);
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

    auto block_exit = n->owningGraph()->create(prim::BlockExit, 0);
    block_exit->insertBefore(body_block->return_node());
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
      addBlockExitInput(block_exit, parent_type, name);
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

// Given a graph where outputs have been added to control flow nodes, and
// loads and stores are represented in the graph, erases the Loads & Stores.
struct EraseLoadsStores {
  void eraseBlockLoadsStores(Block* block) {
    pushFrame(block);
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      auto n = *it;
      it++;
      switch (n->kind()) {
        case prim::If:
        case prim::Loop:
        case prim::Function: {
          for (auto b : n->blocks()) {
            eraseBlockLoadsStores(b);
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
    eraseBlockLoadsStores(graph->block());
  }

  std::shared_ptr<ValueEnvironment> environment_stack = nullptr;
};

// This pass transforms Breaks & Continues to be LoopExit continuations,
// of the form LoopExit(%loop_continue_condition, *loop_carried_vars)
// Break Statements have the condition set to false, and Continue statements
// inline the loop condition as the first input.
struct LoopExitContinuations {
  void addLoopCarriedOutputs(Node* n) {
    auto g = n->owningGraph();
    WithInsertPoint insert(n);
    auto continuation = curr_loop_exit;
    for (auto out : continuation->inputs()) {
      auto load_node = out->node();
      TORCH_INTERNAL_ASSERT(load_node->kind() == prim::Load);
      auto new_load =
          g->insertNode(g->createClone(load_node, [](Value* v) { return v; }));
      n->addInput(new_load->output());
    }
  }

  void setCurrLoopExit(Block* loop_block) {
    curr_loop_exit = loop_block->return_node()->prev();
    TORCH_INTERNAL_ASSERT(curr_loop_exit->kind() == prim::BlockExit);
  }

  void assignExitContinuations(Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      Node* n = *it;
      it++;
      switch (n->kind()) {
        case prim::If: {
          assignExitContinuations(n->blocks().at(0));
          assignExitContinuations(n->blocks().at(1));
        } break;
        case prim::Function: {
          LoopExitContinuations closure_block;
          closure_block.run(n->blocks().at(0));
        } break;
        case prim::Loop: {
          Node* prev_exit = curr_loop_exit;
          setCurrLoopExit(n->blocks().at(0));
          assignExitContinuations(n->blocks().at(0));
          curr_loop_exit = prev_exit;
        } break;
        case prim::ContinueStmt: {
          auto loop_exit = graph->create(prim::LoopExit, 0)->insertAfter(n);
          auto header_block = loop_exit->addBlock();
          auto cur_loop = curr_loop_exit->owningBlock()->owningNode();
          auto pre_header = cur_loop->blocks().at(1);
          header_block->cloneFrom(pre_header, [](Value* v) { return v; });
          moveBlockBeforeNode(n, header_block);
          loop_exit->addInput(header_block->outputs().at(0));
          loop_exit->eraseBlock(0);
          addLoopCarriedOutputs(loop_exit);
          n->destroy();
        } break;
        case prim::BreakStmt: {
          auto loop_exit = graph->create(prim::LoopExit, 0)->insertAfter(n);
          // first input is the loop continue condition - break sets false
          loop_exit->addInput(false_val);
          addLoopCarriedOutputs(loop_exit);
          n->destroy();
        } break;
      }
    }
  }

  void run(Block* b) {
    {
      graph = b->owningGraph();
      WithInsertPoint guard(graph->block()->nodes().front());
      false_val = graph->insertConstant(false);
    }
    assignExitContinuations(b);
  }

  void run(std::shared_ptr<Graph>& graph) {
    run(graph->block());
  }

  Graph* graph;
  Value* false_val;
  Node* curr_loop_exit = nullptr;
};

// Converting to SSA works in multiple parts. First, we add control flow
// loads and stores to the graph. Now that control flow outputs are set,
// we can set remove Break & Continue to have the correct continuations to the
// end of the block (LoopExit). Then we inline the loop condition into the
// graph. Then, we erase Loads & Stores.
// If & Loop block outputs have a prim::BlockExit node that designates the input
//  values to be outputs of the block it is contained.
// if cond:
//    x = 1
// ->
// prim::If(%cond)
//    x.1% = 1
//    prim::BlockExit(%x.1)
//
// Finally, we unify LoopExit and BlockExit nodes to assign the correct
// outputs to each block.
void ConvertToSSA(std::shared_ptr<Graph>& graph) {
  ControlFlowLoadStores ctrl;
  ctrl.run(graph);
  LoopExitContinuations exit_vars;
  exit_vars.run(graph);
  InlineLoopCondition(graph);
  EraseLoadsStores erase_loads_stores;
  erase_loads_stores.run(graph);
  TransformExits(graph);
}

} // namespace script
} // namespace jit
} // namespace torch
