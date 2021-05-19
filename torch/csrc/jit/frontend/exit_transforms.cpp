#include <torch/csrc/jit/frontend/exit_transforms.h>

#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>

namespace torch {
namespace jit {

// WILL states that a node/block must hit the exit, MIGHT that it may happen,
// WONT that it will not happen. THROWS states that a node/block always throws,
// and allows us to create better graphs by not conditionalizing execution
// when it is not necessary. It is an optimization; replacing it with WONT
// would preserve graph semantics.

enum class ExitStatus { WILL, MIGHT, WONT, THROWS };

enum class Transform { Returns, LoopContinuations };

// hasExited() indicates whether or not an exit has been hit.
// The ExitTransform pass maintains a false boolean false_val_ && a true boolean
// true_val_, and an uninitialized boolean throws_val_.
// if hasExited() == true_val_ then we have exited, if hasExited() == false_val_
// we have not, hasExited() == throws_val_ we have hit a block that throws.
// Otherwise, we might have exited.
// exitValues() are the values that we are propagating to a destination block.
// this is used for block outputs of loops and outputs of functions & closures
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
 * The exit node for breaks/continues is LoopContinuation, and the exit for
 * Graphs & Closures is ReturnStmt.
 *
 * Once we hit an Exit Node, we do not execute any further instructions
 * until the exit target has been reached.
 *
 * For blocks and control flow nodes that have an exit statement that may
 * have been hit, we conditionalize all execution on a boolean value that
 * indicates whether we have hit the exit, hasExited().
 *
 * The pass keeps tracks of blocks that always throw, so that we can construct
 * simpler graphs. For example, if in one block of an if statement we return
 * and in the other we throw, we can treat the node as always returning instead
 * of conditionalizing execution in the remainder of the block.
 */
struct ExitTransformer {
  ExitTransformer(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {
    WithInsertPoint guard(graph_->block()->nodes().front());
    true_val_ = graph_->insertConstant(true);
    false_val_ = graph_->insertConstant(false);
    // this value will never be used, since we will always throw before it is
    // accessed
    throws_val_ = getUnitValue(BoolType::get());
  };

  void transformReturnStmts() {
    current_exit_kind_ = prim::ReturnStmt;
    transformExits(graph_->block());
  }

  void transformLoopContinuations() {
    current_exit_kind_ = prim::LoopContinuation;
    transformExits(graph_->block());
  }

 private:
  ExitPair constructThrowsExitPair() {
    return ExitPair(throws_val_, std::vector<Value*>({}));
  }
  ExitPair constructWontExitPair() {
    return ExitPair(false_val_, std::vector<Value*>({}));
  }
  ExitPair constructWillExitPair(at::ArrayRef<Value*> exit_val_ref) {
    return ExitPair(true_val_, exit_val_ref);
  }

  ExitStatus getExitStatus(ExitPair& exit_pair) {
    Value* exit_v = exit_pair.hasExited();
    if (exit_v == true_val_) {
      return ExitStatus::WILL;
    } else if (exit_v == false_val_) {
      return ExitStatus::WONT;
    } else if (exit_v == throws_val_) {
      return ExitStatus::THROWS;
    } else {
      return ExitStatus::MIGHT;
    }
  }

  static Symbol owningNodeKind(Block* block) {
    if (block->owningNode()) {
      return block->owningNode()->kind();
    }
    return Symbol();
  }

  static bool isGraphOrClosureBlock(Block* block) {
    return block->owningNode() == nullptr ||
        owningNodeKind(block) == prim::Closure;
  }

  static void removeOutputs(Block* b) {
    while (b->outputs().size() > 0) {
      b->eraseOutput(0);
    }
  }

  static void registerBlockOutputs(Block* b, at::ArrayRef<Value*> outs) {
    for (Value* out : outs) {
      b->registerOutput(out);
    }
  }

  static void replaceBlockOutputs(Block* b, at::ArrayRef<Value*> outs) {
    removeOutputs(b);
    registerBlockOutputs(b, outs);
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

  ExitPair transformLoop(Node* node) {
    LoopView loop(node);
    Block* body = loop.bodyBlock();
    auto exit_pair = transformExits(body);
    // if we're not exiting to outside the loop we don't need to do any work.
    // since we may not enter the loop return WONT for the THROWS case.

    if (getExitStatus(exit_pair) == ExitStatus::WONT ||
        getExitStatus(exit_pair) == ExitStatus::THROWS) {
      return constructWontExitPair();
    }

    // if we are, we need to update the loop continue condition so that
    // we exit the loop if we've hit an exit
    // and we need to propagate hasExited() and exitValues() outside the loop

    // example:
    // while i < 5:
    //    i += 1
    //    if j == 4:
    //      return 5
    // -> becomes
    //
    // loop_continue = i < 5
    // has_exited = false
    // ret_val = uninitialized(int)
    // while loop_continue:
    //    i += 1
    //    if j == 4:
    //      ret_val = 5
    //      has_exited = True
    //    else:
    //      ret_val = uninitialized(int)
    //      has_exited = False
    //    if has_exited:
    //      loop_continue = False
    //    else:
    //      loop_continue = i < 5

    // update loop continuation condition so that we exit if we hit an exit
    WithInsertPoint insert(body);
    auto new_if = graph_->insertNode(graph_->create(prim::If, 0));
    new_if->addInput(exit_pair.hasExited());
    new_if->addBlock()->registerOutput(false_val_);
    new_if->addBlock()->registerOutput(loop.nextCond());
    auto new_condition = new_if->addOutput()->setType(BoolType::get());
    loop.bodyBlock()->eraseOutput(0);
    loop.bodyBlock()->insertOutput(0, new_condition);

    // add hasExited() to loop outputs, we didn't exit if we didn't enter the
    // loop
    node->addInput(false_val_);
    body->addInput()->setType(BoolType::get());
    body->registerOutput(exit_pair.hasExited());
    Value* new_has_exited = node->addOutput()->setType(BoolType::get());

    // add exit values
    for (Value* exit_value : exit_pair.exitValues()) {
      auto typ = exit_value->type();
      node->addInput(getUnitValue(typ));
      node->addOutput()->setType(typ);
      body->addInput()->setType(typ);
      body->registerOutput(exit_value);
    }

    auto exit_vals = node->outputs().slice(
        node->outputs().size() - exit_pair.exitValues().size());

    return ExitPair(new_has_exited, exit_vals);
  }

  ExitStatus calcIfExitStatus(ExitStatus then_status, ExitStatus else_status) {
    // if one branch throws, we can take the status of the other
    if (then_status == ExitStatus::THROWS) {
      return else_status;
    } else if (else_status == ExitStatus::THROWS) {
      return then_status;
    }

    if (then_status == ExitStatus::WONT && else_status == ExitStatus::WONT) {
      return ExitStatus::WONT;
    }

    if (then_status == ExitStatus::WILL && else_status == ExitStatus::WILL) {
      return ExitStatus::WILL;
    }

    return ExitStatus::MIGHT;
  }

  // Recursively transforms the if node
  ExitPair transformIf(Node* node) {
    auto then_block = node->blocks().at(0);
    auto else_block = node->blocks().at(1);

    auto then_pair = transformExits(then_block);
    auto else_pair = transformExits(else_block);
    auto then_status = getExitStatus(then_pair);
    auto else_status = getExitStatus(else_pair);

    auto if_status = calcIfExitStatus(then_status, else_status);

    if (if_status == ExitStatus::THROWS) {
      return constructThrowsExitPair();
    }
    if (if_status == ExitStatus::WONT) {
      return constructWontExitPair();
    }

    // The exit values of the block that is not exiting will not get
    // used, so we create uninitialized values of the same type as the other
    // block.
    if (then_status == ExitStatus::WONT || then_status == ExitStatus::THROWS) {
      std::vector<Value*> exit_vals =
          matchValuesWithUnitialized(else_pair.exitValues());
      then_pair = ExitPair(then_pair.hasExited(), exit_vals);
    } else if (
        else_status == ExitStatus::WONT || else_status == ExitStatus::THROWS) {
      std::vector<Value*> exit_vals =
          matchValuesWithUnitialized(then_pair.exitValues());
      else_pair = ExitPair(else_pair.hasExited(), exit_vals);
    }

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Value* has_exited;
    if (if_status == ExitStatus::WILL) {
      // Need to maintain the invariant that if hasExited() == true_val_
      // then we have exited.
      has_exited = true_val_;
    } else {
      addIfOutputs(node, {then_pair.hasExited()}, {else_pair.hasExited()});
      has_exited = node->outputs().at(node->outputs().size() - 1);
    }
    addIfOutputs(node, then_pair.exitValues(), else_pair.exitValues());
    size_t num_exit_vals = then_pair.exitValues().size();
    auto exit_vals =
        node->outputs().slice(node->outputs().size() - num_exit_vals);
    return ExitPair(has_exited, exit_vals);
  }

  // Recursively transforms the With node.
  ExitPair transformWith(Node* node) {
    auto body_block = node->blocks().at(0);
    auto body_pair = transformExits(body_block);
    return body_pair;
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
    // after an exit, the only values that will get used
    // are the hasExited() and exitValues(), so we match the existing
    // block outputs with unitialized
    exit_block_vals = matchValuesWithUnitialized(block->outputs());

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

    graph_->create(current_exit_kind_, {exit_pair.exitValues()}, 0)
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

  // if we're entering a Loop block & transforming LoopContinuations, or if
  // we're entering a Closure/Graph block and we're transforming ReturnStmts,
  // then we update target_block_ to be the new block.
  // otherwise, target_block_ remains the same.
  void updateTargetBlock(Block* block) {
    if (owningNodeKind(block) == prim::Loop &&
        // NOLINTNEXTLINE(bugprone-branch-clone)
        current_exit_kind_ == prim::LoopContinuation) {
      target_block_ = block;
    } else if (
        isGraphOrClosureBlock(block) &&
        current_exit_kind_ == prim::ReturnStmt) {
      target_block_ = block;
    }
  }

  ExitPair transformExits(Block* block) {
    Block* prev_target_block = target_block_;
    updateTargetBlock(block);
    ExitPair exit_pair = constructWontExitPair();

    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      Node* node = *it;
      it++;
      switch (node->kind()) {
        case prim::RaiseException: {
          exit_pair = constructThrowsExitPair();
        } break;
        case prim::ReturnStmt:
        case prim::LoopContinuation: {
          if (node->kind() == current_exit_kind_) {
            exit_pair = constructWillExitPair(node->inputs());
            node->destroy();
          }
        } break;
        case prim::If: {
          exit_pair = transformIf(node);
        } break;
        case prim::With: {
          exit_pair = transformWith(node);
        } break;
        case prim::Closure: {
          // exits of closure declaration stay local to the closure
          transformExits(node->blocks().at(0));
        } break;
        case prim::Loop: {
          exit_pair = transformLoop(node);
        } break;
      }

      // if we have hit a node that might exit, we need to conditionally execute
      // all subsequent nodes in the block. if we've hit a node that will exit
      // we can remove all subsequent nodes.
      ExitStatus status = getExitStatus(exit_pair);
      if (status == ExitStatus::WILL || status == ExitStatus::THROWS) {
        deleteAfterExitNodes(block, it);
        break;
      }
      if (status == ExitStatus::MIGHT) {
        if (it != block->nodes().end()) {
          exit_pair = guardBlockNodes(block, exit_pair, it);
        }
        break;
      }
    }

    // if we are targeting this block, update the output values to the
    // exit values. since the exit does not extend outside this block,
    // update returned exit to false. then, reset the target_block to whatever
    // it was previously
    if (target_block_ == block) {
      // if we might have exited, use the new exit values if we did exit,
      // otherwise use the existing block outputs.
      if (getExitStatus(exit_pair) == ExitStatus::MIGHT) {
        auto new_if =
            graph_->create(prim::If, 0)->insertBefore(block->return_node());
        new_if->addBlock();
        new_if->addBlock();
        new_if->addInput(exit_pair.hasExited());
        addIfOutputs(new_if, exit_pair.exitValues(), block->outputs());
        replaceBlockOutputs(block, new_if->outputs());
      } else if (getExitStatus(exit_pair) == ExitStatus::WILL) {
        replaceBlockOutputs(block, exit_pair.exitValues());
      }

      // reset the exiting status. an exit should only reach its target block.
      // e.g. a continue only affects most recent loop, return in closure
      // does not affect enclosing graph.
      // Exceptions do not propagate from Loops bc we might not enter the loop,
      // and not from closures bc the Function node is a declaration and not
      // an invocation.
      exit_pair = constructWontExitPair();
    }
    target_block_ = prev_target_block;
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

  // we create one uninitialized value per type, cache it here and reuse it
  std::unordered_map<TypePtr, Value*> unit_values_;

  // can either be LoopContinuation/ReturnStmt
  Symbol current_exit_kind_;
  Value* true_val_;
  Value* false_val_;
  Value* throws_val_;

  // when we see current_exit_kind_, this is the block that the values are
  // exiting to. For example when we are transforming LoopContinuations
  // for i in range(5):
  //   while i < 3:
  //     continue
  //   break
  // when we transform the for loop block, target_block_ will be set the for
  // block. then, when we enter the while loop, target_block_ will be the while
  // loop block. when we are done transforming the while it will be set back to
  // the for block.
  Block* target_block_ = nullptr;
  std::shared_ptr<Graph> graph_;
};

bool inlineConsecutiveIfs(Node* node) {
  if (node->kind() != prim::If || node->next()->kind() != prim::If) {
    return false;
  }

  IfView first_if(node);
  IfView second_if(node->next());

  // the second if must depend on a value outputted in the first if for us to
  // inline the second if
  if (second_if.cond()->node() != node) {
    return false;
  }

  // both blocks must output a constant value for us to inline, and those values
  // must be different. if the values are the same, then the subsequent if node
  // will get constant prop'd away, and inlining it into the first node would
  // double code size

  auto input_offset = second_if.cond()->offset();
  auto maybe_then_value = toIValue(first_if.thenOutputs().at(input_offset));
  auto maybe_else_value = toIValue(first_if.elseOutputs().at(input_offset));
  if (!maybe_then_value || !maybe_else_value ||
      maybe_then_value->toBool() == maybe_else_value->toBool()) {
    return false;
  }

  bool then_value = maybe_then_value->toBool();
  bool else_value = maybe_else_value->toBool();

  for (auto i = 0; i < 2; ++i) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Block* first_if_block;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Block* second_if_block;

    if (i == 0) {
      first_if_block = first_if.thenBlock();
      second_if_block =
          then_value ? second_if.thenBlock() : second_if.elseBlock();
    } else {
      first_if_block = first_if.elseBlock();
      second_if_block =
          else_value ? second_if.thenBlock() : second_if.elseBlock();
      ;
    }

    // we need to replace values that were used in the second if that were
    // outputs of the first if with the equivalent value in the scope of the
    // block we're copying into
    auto value_map = [&](Value* v) {
      if (v->node() != first_if.node()) {
        return v;
      }
      auto offset = v->offset();
      return first_if_block->outputs().at(offset);
    };

    // clone from also copies block outputs from second_if_block onto
    // first_if_block
    first_if_block->cloneFrom(second_if_block, value_map);
  }

  for (Value* output : second_if.outputs()) {
    auto new_out = first_if.node()->addOutput()->copyMetadata(output);
    output->replaceAllUsesWith(new_out);
  }
  second_if.node()->destroy();
  return true;
}

// After an early return, we conditionalize all further execution
// This means code like the following:
// if x:
//     return 1
// return 2
// Gets generated as one if statement checking `if x`, and then a second if
// statement that conditionalizes execution. We can rewrite cases like these
// into one if statement, so that the above examples gets rewritten to look
// like: if x:
//   return 1
// else:
//   return 2
void inlineConsecutiveIfs(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    for (Block* b : it->blocks()) {
      inlineConsecutiveIfs(b);
    }

    // if we fused two ifs, we need to check current node and new next node
    if (!inlineConsecutiveIfs(*it)) {
      it++;
    }
  }
}

// Adds prim::With nodes to a graph to help handle early exits between
// prim::Enter and prim::Exit nodes. More specifically, it transforms
// IR that looks like this:
//
//   %a = prim::Enter(%b)
//   <code>
//   %c = prim::Exit(%b)
//
// to this:
//
//   %a = prim::Enter(%b)
//   = prim::With()
//     block0():
//       <code>
//     -> ()
//     block1():
//       %c = prim::Exit(%b)
//     -> ()
//
static void convertEnterExitNodesToWithBlocks(std::shared_ptr<Graph>& graph) {
  // First, find all Enter-Exit pairs up front to avoid iterator invalidation
  // issues later when moving nodes around. Do this by iterating through the
  // nodes of the graph while keeping a stack of encountered Enter nodes. Each
  // time an Exit node is seen, its corresponding Enter node must be at the
  // top of the stack. Pop it and record the pair.
  std::vector<std::pair<Node*, Node*>> enter_exit_pairs;
  std::vector<Node*> enter_node_stack;

  DepthFirstGraphNodeIterator it(graph);
  Node* node = it.next();

  while (node) {
    if (node->kind() == prim::Enter) {
      enter_node_stack.emplace_back(node);
    } else if (node->kind() == prim::Exit) {
      // enter_node_stack should not be empty.
      TORCH_INTERNAL_ASSERT(!enter_node_stack.empty());
      // The input to this Exit node should be the same as that of the Enter
      // node on the top of the enter_node_stack.
      TORCH_INTERNAL_ASSERT(
          enter_node_stack.back()->input(0) == node->input(0));
      // Record the pair.
      enter_exit_pairs.emplace_back(enter_node_stack.back(), node);
      enter_node_stack.pop_back();
    }

    node = it.next();
  }

  // The stack should not be empty; an Exit should have been found for every
  // Enter.
  TORCH_INTERNAL_ASSERT(enter_node_stack.empty());

  // Now, add a With block for each Enter-Exit pair. The innermost pairs were
  // found first, so they will be converted first.
  for (auto& pair : enter_exit_pairs) {
    Node* enter = pair.first;
    Node* exit = pair.second;

    auto* with = graph->create(prim::With, /*num_outputs=*/0);
    auto* body_block = with->addBlock();
    auto* exit_block = with->addBlock();

    // Insert the With after the Enter.
    Node* cur = enter->next();
    Node* insert_point = body_block->param_node();

    // Move all of the nodes between the Enter and Exit into the body block.
    while (cur != exit) {
      auto* next = cur->next();
      cur->moveAfter(insert_point);
      insert_point = insert_point->next();
      cur = next;
    }

    // Move the Exit node into the exit block.
    exit->moveAfter(exit_block->param_node());
    with->insertAfter(enter);
  }
}

// Removes prim::With nodes from a graph. More specifically, it transforms
// IR that looks like this:
//
//   %a = prim::Enter(%b)
//   = prim::With()
//     block0():
//       <code>
//     -> ()
//     block1():
//       %c = prim::Exit(%b)
//      ->()
//
// to this:
//   %a = prim::Enter(%b)
//   <code>
//   %c = prim::Exit(%b)
//
static void convertWithBlocksToEnterExitNodes(std::shared_ptr<Graph>& graph) {
  // First, find all With blocks to avoid iterator invalidation issues when
  // moving nodes around later.
  std::vector<Node*> with_nodes;

  DepthFirstGraphNodeIterator it(graph);
  Node* node = it.next();

  while (node) {
    if (node->kind() == prim::With) {
      with_nodes.emplace_back(node);
    }
    node = it.next();
  }

  // For each With node:
  for (auto& node : with_nodes) {
    auto* body_block = node->blocks().at(0);
    auto* exit_block = node->blocks().at(1);

    std::vector<Node*> to_append;

    // Record all nodes that need to be appended after the Enter that precedes
    // the With block to avoid iterator invalidation issues later when moving
    // nodes around.
    for (auto body_node : body_block->nodes()) {
      to_append.emplace_back(body_node);
    }

    for (auto exit_node : exit_block->nodes()) {
      to_append.emplace_back(exit_node);
    }

    Node* cur = node->prev();

    // Move all nodes inside the with block outside of it.
    for (auto& node : to_append) {
      node->moveAfter(cur);
      cur = node;
    }
    node->destroy();
  }
}

// This pass takes in a graph where LoopContinuation & ReturnStmts exist in the
// graph and erases them in the graph, correctly setting block outputs.
// prim::LoopContinuation(*vals) means that the values are targeting the most
// recent loop block. prim::ReturnStmt(*vals) means that the values are
// targeting the most recent Closure or Graph Block. Once we hit an exit node,
// we do not execute any further instructions until the block exit reaches its
// destination. If we encounter a node that contains nested blocks that may
// have hit an exit node, such as an if statement that exits in one block
// and does not exit in the other, we use a boolean value to indicate if the
// exit has been hit or not. Then, we conditionalize further execution.
//
// Python example:
// while i < 5:
//   if i == 3:
//     i += 1
//     continue
//   i += 2
//
// -> transforms to:
//
// continue_loop = i < 5
// while continue_loop:
//   if i == 3:
//     i = i + 1
//     continue_loop = i < 5
//     did_exit = True
//   if did_exit:
//     pass
//   else:
//     i = i + 2
//     continue_loop = i < 5
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
//
//   -> transforms to
//
// %false_val : bool = prim::Constant[value=0]()
// %true_val : bool = prim::Constant[value=1]()
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
//         -> (%true_val, %33, %i.6, %i.6)
//       block1():
//         -> (%false_val, %39, %40, %i.17)
//     %44 : bool, %i : int = prim::If(%did_exit)
//       block0():
//         -> (%continue_loop, %43)
//       block1():
//         %i.13 : int = aten::add(%i.16, %19)
//         %4 : bool = aten::lt(%i.13, %3)
//         -> (%4, %i.13)
//     -> (%44, %i)

void TransformExits(std::shared_ptr<Graph>& graph) {
  convertEnterExitNodesToWithBlocks(graph);
  ExitTransformer e_loop(graph);
  e_loop.transformLoopContinuations();
  ExitTransformer e_ret(graph);
  e_ret.transformReturnStmts();
  inlineConsecutiveIfs(graph->block());
  convertWithBlocksToEnterExitNodes(graph);
}
} // namespace jit
} // namespace torch
