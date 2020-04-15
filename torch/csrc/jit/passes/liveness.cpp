#include <torch/csrc/jit/passes/liveness.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <memory>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {

// LivenessAnalyzer computes "bailout" liveness which is equivalent to
// "{LIVE_IN} or {GEN}" or "{LIVE_OUT} - {KILL}"
struct LivenessAnalyzer {
  explicit LivenessAnalyzer(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)), changed_(false) {}

  std::unordered_map<Node*, std::vector<Value*>> run() {

    GRAPH_DUMP("Liveness Graph: ", graph_);
    // we implement the canonical fixed-point liveness
    // the analysis is run until there are no more changes
    // to liveness sets for each node
    size_t i = 0;
    std::vector<Node*> counters;
    insertExplicitUsesOfLoopCounters(graph_->block(), counters);
    do {
      GRAPH_DEBUG("Running iteration ", i++);
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

  void dump(
      const std::unordered_map<Node*, std::vector<Value*>>& liveness_sets) {
    std::cout << "Liveness info:\n";
    for (auto e : liveness_sets) {
      if (e.first->outputs().size() > 0) {
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


  // temporary make loop counts live for the duration of the loop
  // as they are needed by BailOuts in the loop
  void insertExplicitUsesOfLoopCounters(Block* b, std::vector<Node*>& counters) {

    for (auto it : b->nodes()) {
      if (it->kind() == prim::Loop) {
        LoopView lv(it);
        WithInsertPoint guard(*lv.bodyBlock()->nodes().end());
        // temporary make loop counts live for the duration of the loop
        // as they are needed by BailOuts in the loop
        auto ctc = graph_->create(prim::Store, {lv.currentTripCount()}, 0);
        graph_->insertNode(ctc);
        counters.push_back(ctc);
        GRAPH_DEBUG("@#$Creating a store for ctc : ", ctc);
        auto mtc = graph_->create(prim::Store, {lv.maxTripCount()}, 0);
        graph_->insertNode(mtc);
        counters.push_back(mtc);
      }

      for (auto ib: it->blocks()) {
        insertExplicitUsesOfLoopCounters(ib, counters);
      }
    }
  }

  void removeCounterNodes(std::vector<Node*>& counters) {
    for (auto n: counters) {
      n->destroy();
    }
  }

  SparseBitVector processBlock(Block* b, SparseBitVector liveness) {
    // block outputs are the uses
    auto block_outputs = toSparseBitVector(b->outputs());
    GRAPH_DEBUG("@#$processBlock Liveness", liveness);
    GRAPH_DEBUG("block_outputs : ", block_outputs);
    liveness |= block_outputs;
    GRAPH_DEBUG("Processing block ", b);
    GRAPH_DEBUG("@#$Liveness with block outputs: ", liveness);

    SparseBitVector defs;
    for (Node* it : b->nodes().reverse()) {
      // kill outputs
      GRAPH_DEBUG("Processing node ", getHeader(it));
      liveness -= toSparseBitVector(it->outputs());
      GRAPH_DEBUG("@#$After removing outputs: ", liveness);
      if (it->kind() == prim::Loop) {
        LoopView lv(it);
        // N.B. merge in changes from the loop header
        auto loop_header = *lv.bodyBlock()->nodes().begin();
        GRAPH_DEBUG("@#$Liveness from loop header: ", liveness_sets_[loop_header]);
        auto loop_block = liveness | liveness_sets_[loop_header];
        GRAPH_DEBUG("@#$loop_block liveness before loop: ", loop_block);
        // loop's outputs aren't live inside the loop
        // loop's block outputs, OTOH, will be considered
        // as uses
        loop_block = processBlock(lv.bodyBlock(), loop_block);
        GRAPH_DEBUG("@#$loop_block liveness after loop: ", loop_block);
        // loop block's inputs die outside loop's block
        loop_block -= toSparseBitVector(lv.bodyBlock()->inputs());
        GRAPH_DEBUG("@#$loop_block liveness after removing block inputs: ", loop_block);
        liveness |= loop_block;
        GRAPH_DEBUG("@#$liveness |= loop_block : ", liveness);
      } else if (it->kind() == prim::If) {
        IfView iv(it);
        auto true_liveness = processBlock(iv.thenBlock(), liveness);
        GRAPH_DEBUG("@#$true_liveness: ", true_liveness);
        auto false_liveness = processBlock(iv.elseBlock(), liveness);
        GRAPH_DEBUG("@#$false_liveness: ", false_liveness);
        liveness |= true_liveness;
        liveness |= false_liveness;
        GRAPH_DEBUG("@#$liveness | true_liveness | false_liveness: ", liveness);
      }
      liveness |= toSparseBitVector(it->inputs());
      GRAPH_DEBUG("@#$Liveness after adding node inputs: ", liveness);
      // `|=` returns true if new bits were set in LHS
      // after or/union with `liveness`

      GRAPH_DEBUG("@#$liveness_sets_[it] : ", liveness_sets_[it]);
      auto changed = liveness_sets_[it] |= liveness;
      GRAPH_DEBUG("changed = ", changed);
      GRAPH_DEBUG("liveness_sets_[it] = ", liveness_sets_[it]);
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
