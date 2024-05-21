#include <torch/csrc/jit/passes/liveness.h>

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <iostream>
#include <memory>

namespace torch {
namespace jit {

// LivenessAnalyzer computes "bailout" liveness which is equivalent to
// "{LIVE_IN} or {GEN}" or "{LIVE_OUT} - {KILL}"
struct LivenessAnalyzer {
  explicit LivenessAnalyzer(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)), changed_(false) {}

  std::unordered_map<Node*, std::vector<Value*>> run() {
    std::vector<Node*> counters;
    insertExplicitUsesOfLoopCounters(graph_->block(), counters);

    // we implement the canonical fixed-point liveness
    // the analysis is run until there are no more changes
    // to liveness sets for each node
    do {
      changed_ = false;
      processBlock(graph_->block(), SparseBitVector{});
    } while (changed_);

    removeCounterNodes(counters);
    std::unordered_map<Node*, std::vector<Value*>> result;

    for (const auto& e : liveness_sets_) {
      result.insert({e.first, toValueVector(e.second)});
    }
    return result;
  }

  // temporary make loop counts live for the duration of the loop
  // as they are needed by BailOuts in the loop
  void insertExplicitUsesOfLoopCounters(
      Block* b,
      std::vector<Node*>& counters) {
    for (auto it : b->nodes()) {
      if (it->kind() == prim::Loop) {
        LoopView lv(it);
        WithInsertPoint guard(lv.bodyBlock());
        auto ctc = graph_->create(prim::Store, {lv.currentTripCount()}, 0);
        graph_->insertNode(ctc);
        counters.push_back(ctc);
        auto mtc = graph_->create(prim::Store, {lv.maxTripCount()}, 0);
        graph_->insertNode(mtc);
        counters.push_back(mtc);
      }

      for (auto ib : it->blocks()) {
        insertExplicitUsesOfLoopCounters(ib, counters);
      }
    }
  }

  void removeCounterNodes(std::vector<Node*>& counters) {
    for (auto n : counters) {
      n->destroy();
    }
  }

  void dump(
      const std::unordered_map<Node*, std::vector<Value*>>& liveness_sets) {
    std::cout << "Liveness info:\n";
    for (auto e : liveness_sets) {
      if (!e.first->outputs().empty()) {
        std::cout << e.first->outputs()[0]->debugName();
      }

      std::cout << " " << e.first->kind().toQualString();
      std::cout << " = ";
      dump(e.second);
      std::cout << std::endl;
    }
    std::cout << "graph :\n";
    graph_->dump();
  }

  void dump(const std::vector<Value*>& set) {
    bool first = true;
    std::cout << "[";
    for (auto el : set) {
      if (first) {
        first = false;
      } else {
        std::cout << ", ";
      }
      std::cout << el->debugName() << "(" << el->unique() << ")";
    }
    std::cout << "]";
  }

 private:
  SparseBitVector toSparseBitVector(at::ArrayRef<Value*> values) {
    SparseBitVector sbv;
    for (auto v : values) {
      ids_to_values_[v->unique()] = v;
      sbv.set(v->unique());
    }
    return sbv;
  }

  std::vector<Value*> toValueVector(const SparseBitVector& sbv) {
    std::vector<Value*> vec;
    for (auto id : sbv) {
      vec.push_back(ids_to_values_[id]);
    }
    return vec;
  }

  SparseBitVector processBlock(Block* b, SparseBitVector liveness) {
    // block outputs are the uses
    auto block_outputs = toSparseBitVector(b->outputs());
    liveness |= block_outputs;

    SparseBitVector defs;
    for (Node* it : b->nodes().reverse()) {
      // kill outputs
      liveness -= toSparseBitVector(it->outputs());
      if (it->kind() == prim::Loop) {
        LoopView lv(it);
        // N.B. merge in changes from the loop header
        auto loop_header = *lv.bodyBlock()->nodes().begin();
        auto loop_block = liveness | liveness_sets_[loop_header];
        loop_block = processBlock(lv.bodyBlock(), loop_block);
        // loop block's inputs die outside loop's block
        loop_block -= toSparseBitVector(lv.bodyBlock()->inputs());
        liveness |= loop_block;
      } else if (it->kind() == prim::If) {
        IfView iv(it);
        auto true_liveness = processBlock(iv.thenBlock(), liveness);
        auto false_liveness = processBlock(iv.elseBlock(), liveness);
        liveness |= true_liveness;
        liveness |= false_liveness;
      }
      liveness |= toSparseBitVector(it->inputs());
      // `|=` returns true if new bits were set in LHS
      // after or/union with `liveness`
      auto changed = liveness_sets_[it] |= liveness;
      changed_ = changed_ | changed;
    }
    return liveness;
  }

  std::shared_ptr<Graph> graph_;
  bool changed_;
  std::map<Node*, SparseBitVector> liveness_sets_;
  std::map<size_t, Value*> ids_to_values_;
};

std::unordered_map<Node*, std::vector<Value*>> BuildLivenessSets(
    std::shared_ptr<Graph> graph) {
  LivenessAnalyzer la(std::move(graph));
  return la.run();
}

} // namespace jit
} // namespace torch
