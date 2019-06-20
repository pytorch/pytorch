#include <torch/csrc/jit/passes/bailout_graph.h>
#include <torch/csrc/jit/ir_views.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <memory>
#include <unordered_set>

namespace torch {
namespace jit {

struct BailOutGraphBuilderForNode {
  explicit BailOutGraphBuilderForNode(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {
    copy_graph_ = std::make_shared<Graph>();
  }

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

struct BailOutInserter {
  explicit BailOutInserter(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  void run() {
    insertBailOuts(graph_->block());
    replaceGuardsWithBailouts();
  }

  void removeGuards(Block* b) {
    for (auto it = b->nodes().begin(); it != b->nodes().end(); ++it) {
      if (it->kind() == prim::Guard) {
        // this will need to be profiled again
        it->input()->setType(TensorType::create());
        // destroy the guard
        it->output()->replaceAllUsesWith(it->input());
        it.destroyCurrent();
      }

      for (auto ib : it->blocks()) {
        removeGuards(ib);
      }
    }
  }

  void replaceGuardsWithBailouts() {
    for (auto e : subgraphs) {
      e.second->insertBefore(e.first);
      e.first->output()->replaceAllUsesWith(e.second->output());
      removeGuards(e.second->g(attr::Subgraph)->block());
      // this isn't strictly necessarily
      // but it makes debugging much easier
      ConstantPooling(e.second->g(attr::Subgraph));
    }
    removeGuards(graph_->block());
  }

  void insertBailOuts(Block* b) {
    for (auto it = b->nodes().begin(); it != b->nodes().end(); ++it) {
      if (it->kind() == prim::Guard) {
        auto bailout_node = b->owningGraph()->create(prim::BailOut);
        auto node = *it;

        BailOutGraphBuilderForNode bg(graph_);
        auto bailout_graph = bg.buildBailOutGraphFrom(node);

        for (size_t i = 0; i < bg.live_inputs_.size(); i++) {
          bailout_node->addInput(bg.live_inputs_[i]);

          // to tell which input (index offset)
          // we are actually supposed to guard
          if (it->input() == bg.live_inputs_[i]) {
            bailout_node->i_(attr::slot, i);
            bailout_node->output()->setType(it->output()->type());
          }
        }
        bailout_node->g_(attr::Subgraph, bailout_graph);
        subgraphs.insert({node, bailout_node});
      } else {
        for (auto ib : it->blocks()) {
          insertBailOuts(ib);
        }
      }
    }
  }

  std::shared_ptr<Graph> graph_;
  std::map<Node*, Node*> subgraphs;
};

void InsertBailOuts(std::shared_ptr<Graph> graph) {
  BailOutInserter ibo(std::move(graph));
  ibo.run();
}

} // namespace jit
} // namespace torch
