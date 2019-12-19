#include <torch/csrc/jit/ir_views.h>
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

  void propagateLoopHeaderLiveness() {
    for (auto n : graph_->block()->nodes()) {
      if (n->kind() == prim::Loop) {
        auto loop_liveness = liveness_sets_[n];
        loop_liveness -= toSparseBitVector(n->inputs());
        extendLiveness(n->blocks()[0], loop_liveness);
      }
    }
  }

  std::unordered_map<Node*, std::vector<Value*>> run() {
    // In a special case, where IR is in a strict SSA form (i.e. all uses
    // are dominated by a definition)
    // and the control flow of programs is reducible, liveness can be computed
    // in two passes (See https://hal.inria.fr/inria-00558509v2/document for
    // the correctness proof) Our IR meets both criteria by construction.
    // Namely, there is no way to express irreducible CF with `prim::Loop` and
    // `prim::If` and every definition dominates their uses
    //
    // The first pass, `computePartialLivenessBackwards`, walks a graph backward
    // and depth-first and compute liveness as usual using the following
    // formula: LIVE_IN(B) = LIVE_OUT(SUCC(B)) + GEN(B) - KILL(B) Liveness isn't
    // propagated along back edges and block inputs (phiDefs in the paper) in
    // loop nodes are excluded from liveness propagation. Maximum and current
    // trip count uses are added explicitly as `prim::Store`s at the end of
    // every loop as they are required to build bail out graphs.
    //
    // The second pass, `propagateLoopHeaderLiveness`, propagates the liveness
    // sets of the roots of all the loop forests (i.e. outermost `prim::Loop`s)
    // to every node within their corresponding loop bodies
    computePartialLivenessBackwards(graph_->block(), SparseBitVector{});
    propagateLoopHeaderLiveness();

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

  void extendLiveness(Block *b, const SparseBitVector &loop_block) {
    for (Node *lit : b->nodes()) {
      liveness_sets_.at(lit) |= loop_block;
      for (Block *ib : lit->blocks()) {
        extendLiveness(ib, loop_block);
      }
    }
  }

  SparseBitVector computePartialLivenessBackwards(
      Block* b,
      SparseBitVector liveness) {
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

        LoopView lv(it);
        WithInsertPoint guard(*lv.bodyBlock()->nodes().end());
        // temporary make loop counts live for the duration of the loop
        // as they are needed by BailOuts in the loop
        auto ctc = graph_->create(prim::Store, {lv.currentTripCount()}, 0);
        graph_->insertNode(ctc);
        auto mtc = graph_->create(prim::Store, {lv.maxTripCount()}, 0);
        graph_->insertNode(mtc);
        loop_block =
            computePartialLivenessBackwards(it->blocks()[0], loop_block);
        ctc->destroy();
        mtc->destroy();
        // loop block's inputs die outside loop's block
        loop_block -= toSparseBitVector(it->blocks()[0]->inputs());
        liveness |= loop_block;
      } else if (it->kind() == prim::If) {
        auto true_liveness =
            computePartialLivenessBackwards(it->blocks()[0], liveness);
        auto false_liveness =
            computePartialLivenessBackwards(it->blocks()[1], liveness);
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
