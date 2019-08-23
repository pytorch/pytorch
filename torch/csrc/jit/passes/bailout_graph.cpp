#include <torch/csrc/jit/passes/bailout_graph.h>
#include <torch/csrc/jit/function.h>
#include <torch/csrc/jit/ir_views.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/liveness.h>
#include <memory>
#include <unordered_set>

namespace torch {
namespace jit {

struct BailOutGraphBuilderForNode {
  explicit BailOutGraphBuilderForNode(
      std::shared_ptr<Graph> graph,
      std::shared_ptr<Graph> target)
      : graph_(std::move(graph)), copy_graph_(std::move(target)) {}

  // capture `old_value` into the bailout graph
  // by creating a new input and mapping
  // `old_value` to it
  Value* addNewInputForValue(Value* old_value) {
    auto node = old_value->node();
    // this reduces the number of inputs to a bailout graph significantly
    // making it easier to debug
    if (node->kind() == prim::Constant) {
      auto new_const = copy_graph_->createClone(node, {nullptr});
      copy_graph_->block()->appendNode(new_const);
      return new_const->output();
    }

    live_inputs_.push_back(old_value);
    auto new_value = copy_graph_->block()->addInput();
    return mapExistingInputForValue(old_value, new_value);
  }

  Value* mapExistingInputForValue(Value* old_value, Value* new_value) {
    this->old_to_new_[old_value] = new_value;
    new_value->copyMetadata(old_value);
    return new_value;
  }

  Value* getOrAddInputForValue(Value* v) {
    if (this->old_to_new_.count(v) == 0) {
      return addNewInputForValue(v);
    } else {
      return this->old_to_new_[v];
    }
  }

  // buildBailOutBlockFrom builds a bailout graph from
  // a given node `n` until the end of the owning block
  // If `n` belongs to `prim::If` or `prim::Loop`
  // buildBailOutLoop/If continue
  // from block's owning node (e.g. `prim::If` or
  // `prim::Loop`)
  void buildBailOutBlockFrom(Node* n) {
    auto* block = copy_graph_->block();
    auto b = n->owningBlock();
    for (auto it = n->iterator(); it != b->nodes().end(); it++) {
      auto env = [this](Value* v) { return getOrAddInputForValue(v); };

      auto node = *it;
      auto new_node = block->appendNode(copy_graph_->createClone(node, env));
      for (size_t i = 0; i < node->outputs().size(); ++i) {
        auto oo = node->outputs()[i];
        auto no = new_node->outputs()[i];
        old_to_new_[oo] = no;
      }
    }

    // we are either in `prim::If` or `prim::Loop`
    // bailout graph building will continue from `outer_node` next
    auto outer_node = n->owningBlock()->owningNode();
    if (outer_node) {
      if (outer_node->kind() == prim::Loop) {
        buildBailOutLoop(outer_node);
      } else if (outer_node->kind() == prim::If) {
        buildBailOutIf(b->outputs(), outer_node);
      } else {
        AT_ERROR("Unexpected outer node");
      }
    }
  }

  void mapValues(
      const at::ArrayRef<Value*>& block_outputs,
      const at::ArrayRef<Value*>& carried_deps) {
    TORCH_INTERNAL_ASSERT(block_outputs.size() == carried_deps.size());
    for (size_t i = 0; i < block_outputs.size(); i++) {
      auto nv = getOrAddInputForValue(block_outputs[i]);
      old_to_new_[carried_deps[i]] = nv;
    }
  }

  void buildBailOutLoop(Node* outer_node) {
    LoopView lv(outer_node);
    auto old_max_count = getOrAddInputForValue(lv.maxTripCount());
    auto cur_iter = addNewInputForValue(lv.currentTripCount());
    auto block_outputs = lv.bodyBlock()->outputs();
    auto carried_deps = lv.carriedInputsWithCond();
    mapValues(block_outputs, carried_deps);
    auto* block = copy_graph_->block();
    // subtract the number of iterations
    WithInsertPoint guard(*block->nodes().end());
    auto updated_max_trip_count =
        copy_graph_->insert(aten::sub, {old_max_count, cur_iter});
    mapExistingInputForValue(outer_node->inputs()[0], updated_max_trip_count);
    buildBailOutBlockFrom(outer_node);
  }

  void buildBailOutIf(
      const at::ArrayRef<Value*>& block_outputs,
      Node* outer_node) {
    auto if_outputs = outer_node->outputs();
    mapValues(block_outputs, if_outputs);
    buildBailOutBlockFrom(outer_node->next());
  }

  std::shared_ptr<Graph> buildBailOutGraphFrom(Node* n) {
    buildBailOutBlockFrom(n);
    // add graph outputs
    for (auto ov : graph_->outputs()) {
      copy_graph_->registerOutput(getOrAddInputForValue(ov));
    }
    return copy_graph_;
  }

  std::shared_ptr<Graph> graph_;
  std::shared_ptr<Graph> copy_graph_;
  std::vector<Value*> live_inputs_;
  std::unordered_map<Value*, Value*> old_to_new_;
};

// `BailOutInserter` replaces prim::Guard nodes with
// prim::BailOut nodes that allow interpreter to
// resume execution of the unoptimized(deoptimized)
// version of an original graph from a particular point
struct BailOutInserter {
  explicit BailOutInserter(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)), bailout_index_(0) {}

  void run() {
    liveness_sets_ = BuildLivenessSets(graph_);
    insertBailOuts(graph_->block());
    replaceGuardsWithBailouts();
    // embed a full original graph
    addUnoptimizedFuncToBailouts();
  }

  // Packs the original unoptimized graph into a Function constant
  // and add it as the first input to every prim::BailOut point
  // This graph will be used to compute a bailout graph for
  // any given bailout point
  void addUnoptimizedFuncToBailouts() {
    auto unoptimized_graph = graph_->copy();
    auto unopt_func = graph_->create(prim::BailoutTemplate)
                          ->insertAfter(graph_->param_node());

    // Returns an int so that we have an easy way to do graph traversal
    unopt_func->output()->setType(IntType::get());
    unopt_func->g_(attr::Subgraph, unoptimized_graph);
    for (auto bn : bailouts_) {
      bn->insertInput(0, unopt_func->output());
    }
  }

  // Removes guards by hooking up the guarded tensor
  // directly to its users and also clears
  // profiling information on it.
  void removeGuards(Block* b) {
    for (auto it = b->nodes().begin(); it != b->nodes().end(); ++it) {
      if (it->kind() == prim::Guard) {
        // this will need to be profiled again
        it->input()->setType(TensorType::get());
        // destroy the guard
        it->output()->replaceAllUsesWith(it->input());
        it.destroyCurrent();
      }

      for (auto ib : it->blocks()) {
        removeGuards(ib);
      }
    }
  }

  // replace each prim::Guard
  // with its corresponding prim::BailOut
  void replaceGuardsWithBailouts() {
    for (auto e : replacements_) {
      e.first->replaceAllUsesWith(e.second);
      e.second->node()->insertAfter(e.first->node());
      e.first->node()->destroy();
    }
  }

  // Inserts prim::BailOut nodes for every prim::Guard
  // Each BailOut point takes the set of inputs live
  // at that particular execution point.
  // An input is live if it's used beyond the guard/BailOut
  // point to compute graph's outputs
  void insertBailOuts(Block* b) {
    for (auto it = b->nodes().begin(); it != b->nodes().end(); ++it) {
      if (it->kind() == prim::Guard) {
        auto bailout_node = b->owningGraph()->create(prim::BailOut);
        bailouts_.push_back(bailout_node);

        const auto& live_inputs = liveness_sets_[*it];

        // guarded inputs come first
        // currently, there's always one  guaded input
        bailout_node->addInput(it->input());

        for (auto li : live_inputs) {
          // Guarded inputs have already been added
          // Also, BailOutGraphBuilder materializes constants into a bailout
          // graph rather than captures them as arguments,
          // so there's no need to add them to inputs
          if (li->node()->kind() == prim::Constant || li == it->input()) {
            continue;
          }

          bailout_node->addInput(li);
        }

        bailout_node->output()->setType(it->output()->type());
        bailout_node->i_(attr::index, bailout_index_++);
        // we can't immediately replace nodes since this action will corrupt
        // the liveness sets of following BailOut nodes if any of their
        // arguments are BailOut nodes themselves
        replacements_.insert({it->output(), bailout_node->output()});

      } else {
        for (auto ib : it->blocks()) {
          insertBailOuts(ib);
        }
      }
    }
  }

  std::shared_ptr<Graph> graph_;
  std::map<Node*, Node*> subgraphs;
  std::size_t bailout_index_;
  std::unordered_map<Node*, std::vector<Value*>> liveness_sets_;
  std::vector<Node*> bailouts_;
  std::map<Value*, Value*> replacements_;
};

void InsertBailOuts(std::shared_ptr<Graph> graph) {
  BailOutInserter ibo(std::move(graph));
  ibo.run();
}

// linearly scans through graph's nodes to locate prim::BailOut whose
// index matches the given `index`
static Node* locateBailOutNodeInUnoptimizedGraph(Block* b, int64_t index) {
  for (auto n : b->nodes()) {
    if (n->kind() == prim::BailOut && n->hasAttribute(attr::index) &&
        n->i(attr::index) == index) {
      return n;
    }
    for (auto ib : n->blocks()) {
      if (auto bn = locateBailOutNodeInUnoptimizedGraph(ib, index)) {
        return bn;
      }
    }
  }
  return nullptr;
}

// Removes prim::BailOuts and hooks the guarded input directly
// to its users
static void removeBailouts(Block* b) {
  for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
    if (it->kind() == prim::BailOut) {
      // clear profiling information
      it->inputs().at(0)->setType(TensorType::get());
      it->output()->replaceAllUsesWith(it->inputs().at(0));
      it.destroyCurrent();
    } else {
      for (auto ib : it->blocks()) {
        removeBailouts(ib);
      }
    }
  }
}

// see `bailout_graph.h`
TORCH_API std::shared_ptr<Graph> BuildBailOutGraphFrom(
    int64_t bailout_index,
    const std::shared_ptr<Graph>& orig,
    const std::shared_ptr<Graph>& target) {
  auto orig_bailout_node =
      locateBailOutNodeInUnoptimizedGraph(orig->block(), bailout_index);
  TORCH_INTERNAL_ASSERT(
      orig_bailout_node->inputs().at(0)->type()->cast<FunctionType>() ==
      nullptr);
  TORCH_INTERNAL_ASSERT(
      orig_bailout_node && orig_bailout_node->kind() == prim::BailOut &&
      bailout_index == orig_bailout_node->i(attr::index));
  BailOutGraphBuilderForNode bg(orig, target);
  auto bailout_graph = bg.buildBailOutGraphFrom(orig_bailout_node);
  removeBailouts(bailout_graph->block());
  ConstantPooling(bailout_graph);
  return bailout_graph;
}

} // namespace jit
} // namespace torch
