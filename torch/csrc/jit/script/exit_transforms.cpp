#include <torch/csrc/jit/script/exit_transforms.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/ir_views.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/script/error_report.h>

namespace torch {
namespace jit {
namespace script {

namespace {

void registerBlockOutputs(Block* b, at::ArrayRef<Value*> outs) {
  for (Value* out : outs) {
    b->registerOutput(out);
  }
}

Symbol owningNodeKind(Block* block) {
  if (block->owningNode()) {
    return block->owningNode()->kind();
  }
  return Symbol();
}

} // namespace

enum ExitStatus { WILL, MIGHT, WONT };

// hasExited() indicates whether or not an exit has been hit.
// if hasExited() == true_val_ then we have exited, if == false_val_ we have
// not, otherwise we might have exited. exitValues() are the values that we are
// propagating to a destination block. currently this is limited to block
// outputs of loops
struct ExitPair : public std::pair<Value*, std::vector<Value*>> {
  using pair::pair;

  ExitPair(Value* exit_v, at::ArrayRef<Value*> exit_val_ref) {
    std::vector<Value*> exit_vals;
    for (Value* v : exit_val_ref) {
      exit_vals.push_back(v);
    }
    AT_ASSERT(exit_v->type() == BoolType::get());
    this->first = exit_v;
    this->second = std::move(exit_vals);
  }

  Value* hasExited() const {
    return this->first;
  }

  std::vector<Value*> exitValues() const {
    return this->second;
  }
};

/**
 * This pass currently transforms the Graph so that all exit nodes targeting
 * a block location are removed from the graph and unified.
 * The exit node for breaks/continues is LoopContinuation, and there will be a
 * separate exit node for returns (nyi).
 *
 * Once we hit an Exit Node, we do not execute any further instructions
 * until the exit target has been reached.
 *
 * For blocks and control flow nodes that have an exit statement that may
 * have been hit, we conditionalize all execution on a boolean value that
 * indicates whether we have hit the exit, hasExited().
 */
struct ExitTransformer {
  ExitTransformer(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {
    WithInsertPoint guard(graph_->block()->nodes().front());
    true_val_ = graph_->insertConstant(true);
    false_val_ = graph_->insertConstant(false);
  };

  void run() {
    transformExits(graph_->block());
  }

 private:
  static void removeOutputs(Block* b) {
    while (b->outputs().size() > 0) {
      b->eraseOutput(0);
    }
  }

  static void addIfOutputs(
      Node* n,
      at::ArrayRef<Value*> true_outs,
      at::ArrayRef<Value*> false_outs) {
    IfView if_view(n);
    registerBlockOutputs(if_view.thenBlock(), true_outs);
    registerBlockOutputs(if_view.elseBlock(), false_outs);
    for (size_t i = 0; i < true_outs.size(); ++i) {
      auto out_type =
          unifyTypes(true_outs.at(i)->type(), false_outs.at(i)->type());
      n->addOutput()->setType(*out_type);
    }
  }

  ExitPair transformLoop(Node* node) {
    auto loop_block = node->blocks().at(0);
    auto exit_pair = transformExits(loop_block);

    auto status = getExitStatus(exit_pair);
    // once we transform returns, this will no longer be true
    TORCH_INTERNAL_ASSERT(status == WILL);
    registerBlockOutputs(loop_block, exit_pair.exitValues());
    return exit_pair;
  }

  // creates a vector of uninitialized values of the same type as the
  // values_to_match
  std::vector<Value*> matchValuesWithUnitialized(
      at::ArrayRef<Value*> values_to_match) {
    std::vector<Value*> match_values;
    for (Value* val : values_to_match) {
      match_values.push_back(getUnitValue(val->type()));
    }
    return match_values;
  }

  // Recurses on the if node and returns its return status
  // If status != WONT_RETURN, sets the block_return_val and has returned val
  // of its parent block before exit
  ExitPair transformIf(Node* node) {
    auto true_block = node->blocks().at(0);
    auto false_block = node->blocks().at(1);

    auto true_pair = transformExits(true_block);
    auto false_pair = transformExits(false_block);
    auto true_status = getExitStatus(true_pair);
    auto false_status = getExitStatus(false_pair);

    if (true_status == WONT && false_status == WONT) {
      return ExitPair(false_val_, std::vector<Value*>({}));
    }

    {
      // for the block that is not exitting, its' exit values will not get
      // used so we create uninitialized values of the same type as the other
      // block
      if (true_status == WONT) {
        WithInsertPoint insert(true_block);
        std::vector<Value*> exit_vals =
            matchValuesWithUnitialized(false_pair.exitValues());
        true_pair = ExitPair(false_val_, exit_vals);
      } else if (false_status == WONT) {
        WithInsertPoint insert(false_block);
        std::vector<Value*> exit_vals =
            matchValuesWithUnitialized(true_pair.exitValues());
        false_pair = ExitPair(false_val_, exit_vals);
      }
    }
    Value* has_exited;
    if (true_status == WILL && false_status == WILL) {
      // Need to maintain the invariant that
      // true_val_ == WILL, false_val_ == WONT, else MIGHT
      has_exited = true_val_;
    } else {
      addIfOutputs(node, {true_pair.hasExited()}, {false_pair.hasExited()});
      has_exited = node->outputs().at(node->outputs().size() - 1);
    }
    addIfOutputs(node, true_pair.exitValues(), false_pair.exitValues());
    size_t num_exit_vals = true_pair.exitValues().size();
    auto exit_vals =
        node->outputs().slice(node->outputs().size() - num_exit_vals);
    return ExitPair(has_exited, exit_vals);
  }

  ExitStatus getExitStatus(ExitPair& exit_pair) {
    Value* exit_v = exit_pair.hasExited();
    if (exit_v == true_val_) {
      return WILL;
    } else if (exit_v == false_val_) {
      return WONT;
    } else {
      return MIGHT;
    }
  }

  // Guards the remaining nodes in the block with an if node that takes
  // the has exited value as its conditional
  ExitPair guardBlockNodes(
      Block* block,
      const ExitPair& exit_pair,
      graph_node_list_iterator& iter) {
    auto new_if = graph_->create(prim::If, 0)->insertBefore(*iter);
    new_if->addInput(exit_pair.hasExited());

    auto exit_block = new_if->addBlock();
    auto guard_block = new_if->addBlock();

    // Move all remaining nodes into the guard block
    while (iter != block->nodes().end()) {
      auto node = *iter++;
      node->moveBefore(guard_block->return_node());
    }

    std::vector<Value*> exit_block_vals;
    {
      // after an exit, the only values that will get used
      // are the hasExited() and exitValues(), so we match the existing
      // block outputs with unitialized
      WithInsertPoint insert(exit_block);
      exit_block_vals = matchValuesWithUnitialized(block->outputs());
    }

    // Set the new if to have the same outputs of the original block,
    // then replace the original block outputs with new if's outputs
    for (size_t i = 0; i < block->outputs().size(); ++i) {
      exit_block->registerOutput(exit_block_vals.at(i));
      guard_block->registerOutput(block->outputs().at(i));
      new_if->addOutput()->setType(block->outputs().at(i)->type());
    }

    while (block->outputs().size() > 0) {
      block->eraseOutput(0);
    }
    for (auto out : new_if->outputs()) {
      block->registerOutput(out);
    }

    graph_->create(prim::LoopContinuation, {exit_pair.exitValues()}, 0)
        ->insertBefore(exit_block->return_node());
    return transformIf(new_if);
  }

  // these nodes my have uses,
  // such as in the case:
  // if i == 1:
  //    break
  //    j = j + 1
  // where the j + 1 value will be a block output, but since they will
  // never be used, it is safe to replace them with unitialized value
  void destroyNodeAfterExit(Node* n) {
    for (auto output : n->outputs()) {
      if (output->uses().size() > 0) {
        output->replaceAllUsesWith(getUnitValue(output->type()));
      }
    }
    n->destroy();
  }

  void deleteAfterExitNodes(Block* block, graph_node_list_iterator& iter) {
    if (iter == block->nodes().end()) {
      return;
    }
    WithInsertPoint insert(*block->nodes().begin());
    // need to destroy in reverse order so nodes have no uses when destroyed
    for (auto it = block->nodes().reverse().begin(); it != iter;) {
      Node* n = *it++;
      if (*it != block->return_node()) {
        destroyNodeAfterExit(n);
      }
    }
    destroyNodeAfterExit(*iter);
  }

  ExitPair transformExits(Block* block) {
    ExitPair exit_pair = ExitPair(false_val_, std::vector<Value*>({}));
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      Node* node = *it;
      it++;
      switch (node->kind()) {
        case prim::LoopContinuation: {
          exit_pair = ExitPair(true_val_, node->inputs());
          node->destroy();
        } break;
        case prim::If: {
          exit_pair = transformIf(node);
        } break;
        case prim::Loop: {
          // for now, ignore loop return, once we handle returns no longer true
          transformLoop(node);
        } break;
        default:
          break;
      }
      ExitStatus status = getExitStatus(exit_pair);
      if (status == WILL) {
        deleteAfterExitNodes(block, it);
        break;
      }
      if (status == MIGHT) {
        if (it != block->nodes().end()) {
          exit_pair = guardBlockNodes(block, exit_pair, it);
        }
        break;
      }
    }
    return exit_pair;
  }

  Value* getUnitValue(const TypePtr& type) {
    auto maybe_val = unit_values_.find(type);
    if (maybe_val != unit_values_.end()) {
      return maybe_val->second;
    }
    auto unit = graph_->createUninitialized(type)
                    ->insertAfter(graph_->param_node())
                    ->output();
    unit_values_[type] = unit;
    return unit;
  }

  std::unordered_map<TypePtr, Value*> unit_values_;
  Value* true_val_;
  Value* false_val_;

  std::shared_ptr<Graph> graph_;
};

// The Logic for the loop transform simplifies if the blcok outputs
// are converted to LoopContinuations before running.
void convertLoopBlockExits(Block* block) {
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      convertLoopBlockExits(b);
    }
  }
  if (owningNodeKind(block) == prim::Loop) {
    auto ret_node = block->return_node();
    auto loop_exit = block->owningGraph()
                         ->create(prim::LoopContinuation, 0)
                         ->insertBefore(ret_node);
    for (auto inp : ret_node->inputs()) {
      loop_exit->addInput(inp);
    }
    for (; ret_node->inputs().size() > 0;) {
      ret_node->removeInput(0);
    }
  }
}

// This pass takes in a graph where LoopContinuation exist in the graph
// and erases them in the graph, correctly setting block outputs.
// prim::LoopContinuation(*vals) denotes that the values are targeting the most
// recent loop block. FunctionExits are NYI. Once we hit an exit node, we do not
// execute any further instructions until the block exit reaches its
// destination. If a node may hit an exit node that pauses execution, we use a
// boolean value to indicate if the exit has been hit or not, and conditionalize
// further execution. First we  replace Loop Block outputs with
// LoopContinuations. Then we remove LoopContinuations. Python example: while i
// < 5:
//   if i == 3:
//     i += 1
//     continue
//   i += 2
// continue_loop = i < 5
// while continue_loop:
//   if i == 3:
//     i = i + 1
//     continue_loop = false
//     did_exit = True
//   if did_exit:
//     pass
//   else:
//     i = i + 2
//     continue_loop = i < 3
// IR as it enters pass:
// %36 : bool = aten::lt(%i.1, %3)
// %i : int = prim::Loop(%1, %36, %i.1)
//   block0(%5 : int, %i.17 : int):
//     %8 : bool = aten::eq(%i.17, %7)
//     %i.16 : int = prim::If(%8)
//       block0():
//         %i.6 : int = aten::add(%i.17, %11)
//         %33 : bool = aten::lt(%i.6, %3)
//          = prim::LoopContinuation(%33, %i.6)
//         -> (%i.6)
//       block1():
//         -> (%i.17)
//     %i.13 : int = aten::add(%i.16, %19)
//     %4 : bool = aten::lt(%i.13, %3)
//     -> (%4, %i.13)
// return (%i)
//   -> becomes
// %38 : bool = prim::Constant[value=0]()
// %37 : bool = prim::Constant[value=1]()
// %40 : int = prim::Uninitialized()
// %39 : bool = prim::Uninitialized()
// %36 : bool = aten::lt(%i.1, %3)
// %i : int = prim::Loop(%1, %36, %i.1)
//   block0(%5 : int, %i.17 : int):
//     %8 : bool = aten::eq(%i.17, %7)
//     %did_exit : bool, %continue_loop : bool, %43 : int, %i.16 : int =
//     prim::If(%8)
//       block0():
//         %i.6 : int = aten::add(%i.17, %11)
//         %33 : bool = aten::lt(%i.6, %3)
//         -> (%37, %33, %i.6, %i.6)
//       block1():
//         -> (%38, %39, %40, %i.17)
//     %44 : bool, %i : int = prim::If(%did_exit)
//       block0():
//         -> (%continue_loop, %43)
//       block1():
//         %i.13 : int = aten::add(%i.16, %19)
//         %4 : bool = aten::lt(%i.13, %3)
//         -> (%4, %i.13)
//     -> (%44, %i)

void TransformExits(std::shared_ptr<Graph>& graph) {
  convertLoopBlockExits(graph->block());
  ExitTransformer e(graph);
  e.run();
}

} // namespace script
} // namespace jit
} // namespace torch
