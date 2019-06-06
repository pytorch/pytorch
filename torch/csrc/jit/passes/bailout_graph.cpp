#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/bailout_graph.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <memory>
#include <unordered_set>

namespace torch {
namespace jit {

struct BailOutGraphBuilderForNode {
  BailOutGraphBuilderForNode(std::shared_ptr<Graph> graph)
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
  // buildBailOutGraphForOwningNode continues
  // from block's owning node (e.g. `prim::If` or
  // `prim::Loop`)
  void buildBailOutBlockFrom(Node* n) {
    auto* block = copy_graph_->block();
    auto b = n->owningBlock();
    graph_node_list_iterator it(n, kNextDirection);
    for (; it != b->nodes().end(); it++) {
      auto env = [this](Value* v) { return getOrAddInputForValue(v); };

      auto node = *it;

      auto new_node = block->appendNode(copy_graph_->createClone(node, env));
      for (size_t i = 0; i < node->outputs().size(); ++i) {
        auto oo = node->outputs()[i];
        auto no = new_node->outputs()[i];
        old_to_new_[oo] = no;
        no->copyMetadata(oo);
      }
    }

    // we are either in `prim::If` or `prim::Loop`
    // bailout graph building will continue from `outer_node` next
    // remember/map the outputs needed to unwind this `prim::If`
    // or `prim::Loop`
    auto outer_node = n->owningBlock()->owningNode();
    if (outer_node) {
      buildBailOutGraphForOwningNode(n, outer_node);
    }
  }

  // buildBailOutGraphForOwningNode builds a bailout graph
  // for a CFG given node () as follows:
  // * if `inner_node` is in a true or false arm of an `prim::If` node
  // the rest of the arm gets inlined in the bailout graph
  // and we continue building the rest of the bailout graph
  // from the node *following* `outer_node`.
  // * if `inner_node` is in a loop, the rest of the loop gets inlined
  // we remember the loop outputs needed to execute the
  // we adjust the upper limit (old_upper_limit - current_iteration)
  // note the stop condition is already mapped (remembered)
  // then, we continue building the rest of the bailout graph
  // starting from the owning `outer_node`
  // which implies `outer_node` will be cloned
  void buildBailOutGraphForOwningNode(Node* inner_node, Node* outer_node) {
    auto* block = copy_graph_->block();
    auto block_outputs = inner_node->owningBlock()->outputs();
    // skip the first input for loops (current iteration count)
    size_t i = outer_node->kind() == prim::Loop;
    auto new_outputs = outer_node->kind() == prim::Loop ? outer_node->inputs()
                                                        : outer_node->outputs();
    for (; i < block_outputs.size(); i++) {
      auto nv = old_to_new_[block_outputs[i]];
      old_to_new_[new_outputs.at(i)] = nv;
    }

    if (outer_node->kind() == prim::If) {
      // skip if since the rest of true or false block has already been
      // inlined
      buildBailOutBlockFrom(outer_node->next());
    } else if (outer_node->kind() == prim::Loop) {
      auto new_max_count = addNewInputForValue(outer_node->inputs()[0]);
      auto cur_iter = addNewInputForValue(outer_node->blocks()[0]->inputs()[0]);
      // subtract the number of iterations we already did
      WithInsertPoint guard(*block->nodes().end());
      auto updated_max_trip_count =
          copy_graph_->insert(aten::sub, {new_max_count, cur_iter});
      mapExistingInputForValue(outer_node->inputs()[0], updated_max_trip_count);
      // N.B. the rest of inputs have already been mapped
      // when loop->blocks()[0] was processed
      buildBailOutBlockFrom(outer_node);
    } else {
      AT_ERROR("Unexpected outer node");
    }
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
  BailOutInserter(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

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
