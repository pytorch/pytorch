#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/liveness.h>
#include <memory>

namespace torch {
namespace jit {

// LivenessAnalyzer computes "bailout" liveness which is equivalent to
// "{LIVE_IN} or {GEN}" or "{LIVE_OUT} - {KILL}"
struct LivenessAnalyzer {
  explicit LivenessAnalyzer(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  std::unordered_map<Node*, std::vector<Value*>> run() {
    processBlock(graph_->block(), SparseBitVector{});
    std::unordered_map<Node*, std::vector<Value*>> result;

    for (const auto& e : liveness_sets_) {
      result.insert({e.first, toValueVector(e.second)});
    }
    return result;
  }

  void dump(const std::map<Node*, std::vector<Value*>>& liveness_sets) {
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

  SparseBitVector processBlock(Block* b, SparseBitVector liveness) {
    // block outputs are the uses
    auto block_outputs = toSparseBitVector(b->outputs());
    liveness |= block_outputs;

    SparseBitVector defs;
    for (Node* it : b->nodes().reverse()) {
      // kill outputs
      liveness -= toSparseBitVector(it->outputs());
      if (it->kind() == prim::Loop) {
        auto loop_block = liveness;
        // loop's outputs aren't live inside the loop
        // loop's block outputs, OTOH, will be considered
        // as uses
        loop_block = processBlock(it->blocks()[0], loop_block);
        // loop block's inputs die outside loop's block
        loop_block -= toSparseBitVector(it->blocks()[0]->inputs());
        liveness |= loop_block;
      } else if (it->kind() == prim::If) {
        auto true_liveness = processBlock(it->blocks()[0], liveness);
        auto false_liveness = processBlock(it->blocks()[1], liveness);
        liveness |= true_liveness;
        liveness |= false_liveness;
      }
      liveness |= toSparseBitVector(it->inputs());
      liveness_sets_.insert({it, liveness});
    }
    return liveness;
  }

  std::shared_ptr<Graph> graph_;
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
