#include <torch/csrc/jit/passes/guard_elimination.h>
#include <torch/csrc/jit/graph_executor.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <memory>
#include <unordered_set>

namespace torch {
namespace jit {

struct GuardElimination {
  GuardElimination(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)), aliasDb_(std::make_unique<AliasDb>(graph_)) {}

  void run() {
    const size_t MAX_ATTEMPTS = 5;
    size_t attempts = MAX_ATTEMPTS;
    while (attempts-- && moveGuardsToDefs(graph_->block())) {
    }
    GRAPH_DUMP("After moveGuardsToDefs", graph_);
    coalesceGuards(graph_->block());
    GRAPH_DUMP("After coalesceGuards", graph_);
    eliminateRedundantGuards(graph_->block());
    GRAPH_DUMP("After eliminateRedundantGuards", graph_);
  }

  static bool isLoweredGradOf(Node* n) {
    if (n->kind() != prim::If) {
      return false;
    }

    return n->input(0)->node()->kind() == prim::AutogradAnyNonZero;
  }

  bool moveGuardsToDefs(Block* b) {
    bool changed = false;
    for (auto it = b->nodes().begin(); it != b->nodes().end();) {
      auto n = *it;
      if (n->kind() == prim::Guard) {
        // grab the next node before we move this one all the way back
        it++;
        auto guardee = n->inputs().at(0)->node();
        // alias analysis will try to hoist a node out of a loop
        // if asked. if guardee is in a loop, it should only
        // be moved to the beginning of the basic block
        // given the current implementation of AliasAnalysis
        if (guardee->owningBlock() != n->owningBlock()) {
          guardee = *n->owningBlock()->nodes().begin();
        }
        bool moved = aliasDb_->moveAfterTopologicallyValid(n, guardee);
        changed |= moved;
        if (moved) {
          GRAPH_UPDATE(
              "Moved ",
              n->output()->debugName(),
              " to ",
              n->inputs().at(0)->debugName());
        }
      } else {
        it++;
        for (Block* ib : n->blocks()) {
          moveGuardsToDefs(ib);
        }
      }
    }

    if (b->owningNode() &&
        isLoweredGradOf(
            b->owningNode()) /*b->owningNode()->kind() == prim::If*/) {
      for (auto it = b->nodes().begin(); it != b->nodes().end();) {
        auto block_node = *it++;
        if (block_node->kind() != prim::Guard) {
          break;
        }
        block_node->moveBefore(b->owningNode());
        changed = true;
      }
    }

    return changed;
  }

  void coalesceGuards(Block* b) {
    // uses on *all* parameters are moved to the same anchor node
    // and they may come in different order after the anchor node
    // e.g. (anchor, guard_x, guard_y, guard_x, guard_y)
    // this pass recognizes contigious streches of guards and
    // keeps track of the guards it's seen for each def. the next time
    // the guard on the same def, it simply removes it.
    std::unordered_map<Value*, Node*> inputs_to_guards;
    for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
      auto n = *it;
      if (n->kind() == prim::Guard) {
        if (inputs_to_guards.count(n->input())) {
          auto prev = inputs_to_guards[n->input()];
          n->output()->replaceAllUsesWith(prev->output());
          GRAPH_UPDATE(
              "Replacing ",
              n->output()->debugName(),
              " with ",
              prev->output()->debugName());
          it.destroyCurrent();
        } else {
          inputs_to_guards.insert({n->input(), n});
        }
      } else if (n->kind() != prim::Constant) {
        inputs_to_guards.clear();
        for (Block* ib : n->blocks()) {
          coalesceGuards(ib);
        }
      }
    }
  }

  // we need to make sure there are no ops in between guardee's
  // output and its guard except for other guards as they can
  // invalidate shape information.
  bool guardsOutput(Node* guard) {
    auto output = guard->input()->node();
    auto it = guard;
    while (it != output) {
      if (it->kind() != prim::Guard && it->kind() != prim::Constant) {
        GRAPH_DEBUG(
            "found an unexpected node ",
            *it,
            " while trying to eliminate ",
            *guard);
        return false;
      }
      it = it->prev();
    }

    return true;
  }

  void eliminateRedundantGuards(Block* b) {
    // a very simple pass to eliminate redundant guards for ops
    // whose outputs are fully determined by their inputs
    // i.e. if inputs to such ops are guarded we are allowed
    // to remove a guard on ops' outputs
    for (auto it = b->nodes().rbegin(); it != b->nodes().rend();) {
      auto n = *it;
      GRAPH_DEBUG("eliminateRedundantGuards ", getHeader(n));
      if (n->kind() == prim::Guard && guardsOutput(n) &&
          removableGuard(n->inputs().at(0)->node(), n->output()->type())) {
        auto pttp = n->output()->type();
        n->output()->replaceAllUsesWith(n->inputs().at(0));
        n->inputs().at(0)->setType(pttp);
        GRAPH_UPDATE(
            "Eliminating the redundant guard ", n->output()->debugName());
        it.destroyCurrent();
      } else {
        it++;
        for (Block* ib : n->blocks()) {
          eliminateRedundantGuards(ib);
        }
      }
    }
  }

  // void eliminateInflates(Block* b) {
  //   for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
  //     auto n = *it;
  //     if (n->kind() == prim::inflate) {
  //         n->output()->replaceAllUsesWith(n->input());
  //         GRAPH_UPDATE(
  //             "Replacing ",
  //             n->output()->debugName(),
  //             " with ",
  //             n->input()->debugName());
  //         it.destroyCurrent();
  //     }
  //   }
  // }

  bool checkSimpleBroadcastableInputs(Node* n, TensorTypePtr type) {
    auto bced_sizes = *type->sizes().concrete_sizes();
    for (auto input : n->inputs()) {
      if (input->node()->kind() == prim::Constant ||
          input->type()->isSubtypeOf(NumberType::get())) {
        continue;
      }

      if (input->node()->kind() != prim::Guard) {
        GRAPH_DEBUG("%", input->debugName(), " isn't a guard!");
        return false;
      }

      TORCH_INTERNAL_ASSERT(input->type()->cast<TensorType>());
      auto isizes = input->type()->cast<TensorType>()->sizes();
      // even rank isn't fixed
      if (!isizes.size().has_value()) {
        GRAPH_DEBUG("%", input->debugName(), "'s rank isn't fixed!");
        return false;
      }

      // TODO: just copy and pad isizes as needed
      auto padding_size = bced_sizes.size() - *isizes.size();

      for (size_t i = 0; i < bced_sizes.size(); i++) {
        auto input_dim =
            (i < padding_size) ? c10::nullopt : isizes[i - padding_size];
        if (input_dim.has_value() && *input_dim != bced_sizes[i]) {
          GRAPH_DEBUG(
              i,
              "-th dimension of %",
              input->debugName(),
              " doesn't match output ",
              getHeader(n),
              " i.e. ",
              *input_dim,
              " != ",
              bced_sizes[i]);
          return false;
        }
      }
    }
    return true;
  }

  // bool checkSimpleBroadcastableInputs(Node* n, std::vector<size_t>
  // input_indices) {
  //   auto bced_sizes = *type->sizes().concrete_sizes();

  //     if (input->node()->kind() != prim::Guard) {
  //       GRAPH_DEBUG("%", input->debugName(), " isn't a guard!");
  //       return false;
  //     }

  //     TORCH_INTERNAL_ASSERT(input->type()->cast<TensorType>());
  //     auto isizes = input->type()->cast<TensorType>()->sizes();
  //     // even rank isn't fixed
  //     if (!isizes.size().has_value()) {
  //       GRAPH_DEBUG("%", input->debugName(), "'s rank isn't fixed!");
  //       return false;
  //     }

  //     // TODO: just copy and pad isizes as needed

  //     for (size_t i = 0; i < bced_sizes.size(); i++) {

  //       bool match = false;

  //       for (auto ii : input_indices) {
  //         auto isizes = n->input(ii)->type()->cast<TensorType>()->sizes();
  //         auto padding_size = bced_sizes.size() - *isizes.size();
  //         auto input_dim =
  //           (i < padding_size) ? -1 : bced_sizes[i];
  //       }

  //       if (!match) {

  //       }

  //       if (input_dim.has_value() && *input_dim != bced_sizes[i]) {
  //         GRAPH_DEBUG(
  //             i,
  //             "-th dimension of %",
  //             input->debugName(),
  //             " doesn't match output ",
  //             getHeader(n),
  //             " i.e. ",
  //             *input_dim,
  //             " != ",
  //             bced_sizes[i]);
  //         return false;
  //       }
  //     }

  //   return true;
  // }

  // `checkInputs` check the invariants specified in `removableGuard`
  // on inputs to `n`. The invariants must hold, or an input must
  // be a `prim::Constant` or be of `NumberType` or be included
  // as an exception in `except`
  bool checkInputs(Node* n, const std::unordered_set<size_t>& except) {
    bool all_inputs_guarded = true;
    size_t i = 0;
    for (auto input : n->inputs()) {
      if ((input->node()->kind() == prim::Guard &&
           !input->type()->expect<TensorType>()->isSummarized2()) ||
          input->node()->kind() == prim::Constant ||
          input->type()->isSubtypeOf(NumberType::get()) ||
          except.count(i) != 0) {
        AT_ASSERT(
            input->node()->kind() != prim::Guard ||
            input->type()->expect<TensorType>());
      } else {
        GRAPH_DEBUG(
            "input ",
            input->debugName(),
            " isn't guarded, type ",
            *input->type());
        all_inputs_guarded = false;
        break;
      }
      i++;
    }
    return all_inputs_guarded;
  }

 private:
  // `removableGuard` relies on the properties checked by `isSummarized()`
  // and passes shouldn't insert nodes between a guard and its uses that
  // may alter those properties.
  // `removableGuard` expects type information to come directly from
  // Profiler. Passes shouldn't try to alter type information provided by
  // profiling
  // While we can derive very simple rules stating when it's valid to remove
  // `prim::Guard` on operation's output if all of its inputs are guarded for
  // some
  // categories of operations
  // there's no comprehensive set of rules that covers all the operations
  // available in PyTorch
  // If your operation falls into one of the categories described below, you
  // should add it
  // to switch statement below that contains the other operations in the said
  // category.
  // Otherwise, you will need to derive the rules for your case on your own.
  // Generally, any operation that is stateful in any way or uses its underlying
  // data
  // to compute any properties `isSummarized()` isn't amenable to guard
  // elimination.
  // Categories:
  // * Functional-like(e.g. add, sub, le) operations with broadcast semenatics
  //   Guards can be removed if all inputs are guarded and `isSummarized()`
  //   returns
  //   false or inputs are `prim::Constant`
  bool removableGuard(Node* n, TypePtr type) {
    GRAPH_DEBUG("Running removableGuard for ", getHeader(n));
    const static auto no_exceptions = std::unordered_set<size_t>{};
    switch (n->kind()) {
      case aten::add:
      case aten::sub:
      case aten::mul:
      case aten::div:
      case aten::t:
      case aten::sigmoid:
      case aten::sin:
      case aten::cos:
      case aten::tan:
      case aten::sinh:
      case aten::cosh:
      case aten::tanh:
      case aten::asin:
      case aten::acos:
      case aten::atan:
      case aten::atan2:
      case aten::floor:
      case aten::fmod:
      case aten::ceil:
      case aten::trunc:
      case aten::sqrt:
      case aten::rsqrt:
      case aten::remainder:
      case aten::mm:
      case aten::min:
      case aten::max:
      case aten::type_as:
      case aten::ge:
      case aten::gt:
      case aten::lt:
      case aten::le:
      case aten::eq:
      case aten::ne:
      case aten::neg:
      case prim::ConstantChunk:
      case aten::size:
      case aten::abs:
      case aten::sign:
      case aten::pow:
      case aten::relu:
      case aten::threshold:
      case aten::avg_pool2d:
      case prim::AutogradAdd:
      case prim::AutogradZero:
      case aten::rand_like:
      case aten::erf:
      case aten::erfc:
      case aten::exp:
      case aten::expm1:
      case aten::log:
      case aten::log2:
      case aten::log10:
      case aten::frac:
      case aten::lerp:
      case aten::lgamma:
      case aten::reciprocal:
      case aten::addcmul:
      case aten::_cast_Float:
      case aten::_sigmoid_backward:
      case aten::_tanh_backward:
      case aten::__and__:
      case aten::__xor__:
      case aten::__lshift__:
      case aten::__rshift__:
      case aten::where:
      case prim::inflate: {
        // auto ttype = type->cast<TensorType>();
        // TORCH_INTERNAL_ASSERT(ttype);
        //  return !ttype->isSummarized2() &&
        //      checkSimpleBroadcastableInputs(n, ttype);
        return checkInputs(n, no_exceptions);
        // return !ttype->isSummarized() &&
        //     checkSimpleBroadcastableInputs(n, ttype);
        break;
      }
      case aten::slice:
        return !n->input(0)->type()->expect<TensorType>()->isSummarized() &&
            // check that the dimension argument is constant
            n->input(1)->node()->kind() == prim::Constant &&
            // the start offset is constant
            n->input(2)->node()->kind() == prim::Constant &&
            // the end offset is constant
            n->input(3)->node()->kind() == prim::Constant &&
            // the stride is constant
            n->input(4)->node()->kind() == prim::Constant;
      case aten::unsqueeze:
        // check that the dimension argument is constant
        return !n->input(0)->type()->expect<TensorType>()->isSummarized() &&
            n->input(1)->node()->kind() == prim::Constant;
      case aten::cat:
        // check that the dimension argument is constant
        return n->input(1)->node()->kind() == prim::Constant &&
            n->input(0)->node()->kind() == prim::ListConstruct &&
            // no extra nodes in between aten::cat and prim::ListConstruct
            n->prev() == n->input(0)->node() &&
            // check the inputs to prim::ListConstruct (not aten::cat)
            checkInputs(n->input(0)->node(), no_exceptions);
      case aten::clamp:
        // the second and third args do not affect shapes
        return checkInputs(n, std::unordered_set<size_t>{1, 2});
      // after some optimizations we might end up with two Guards back-to-back
      // which case we can remove the one whose input is also prim::Guard
      case aten::_grad_sum_to_size:
        // skip checking size argument
        if (checkInputs(n, std::unordered_set<size_t>{1})) {
          auto asize = n->input(1)->node();
          if (asize->kind() == prim::Constant) {
            return true;
          } else if (asize->matches("aten::size(Tensor self) -> int[]")) {
            // aten::size is effectively a constant
            if (asize->input()
                    ->type()
                    ->expect<TensorType>()
                    ->sizes()
                    .concrete_sizes()) {
              return true;
            }
          }
        }
        return false;

      // this is checked by one of the tests in test_jit_fuser.py
      case prim::ListUnpack: {
        // check if the input is a constant chunk
        // used for LSTM fusions
        auto chunk = n->input(0)->node();
        if (chunk->kind() != aten::chunk) {
          return false;
        }
        return checkInputs(chunk, no_exceptions);
      }
      // this is checked by one of the tests in test_jit_fuser.py
      case aten::broadcast_tensors: {
        auto list_construct = n->input(0)->node();
        if (list_construct->kind() != prim::ListConstruct) {
          return false;
        }
        return checkInputs(list_construct, no_exceptions);
      }
      case prim::Guard:
      case prim::GradOf:
        return true;
      default:
        GRAPH_DEBUG("cannot remove ", n->kind().toQualString());
        return false;
    }
  }

  std::shared_ptr<Graph> graph_;
  std::unique_ptr<AliasDb> aliasDb_;
  static std::unordered_set<Symbol> simple_ops_;
};

void EliminateRedundantGuards(std::shared_ptr<Graph> graph) {
  GuardElimination ge(std::move(graph));
  ge.run();
}

} // namespace jit
} // namespace torch
