#include <torch/csrc/jit/passes/specialize_autogradzero.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include "ATen/core/interned_strings.h"
#include "jit/ir/ir.h"
#include "jit/jit_log.h"
#include "jit/passes/dead_code_elimination.h"

namespace torch {
namespace jit {


static bool isBackwardGraph(Graph& g) {
  return std::any_of(g.nodes().begin(), g.nodes().end(), [](Node* n){return n->kind() == prim::AutogradAnyNonZero; });
}

static void replaceBlockInputsWithGraph(Graph* g, Block* b) {
  TORCH_INTERNAL_ASSERT(g->inputs().size() == b->inputs().size());
  for (int i = g->inputs().size() - 1; i >= 0; i--) {
    b->inputs()[i]->replaceAllUsesWith(g->inputs()[i]);
    b->eraseInput(i);
  }
}

struct AutogradZeroSpecializer {

  AutogradZeroSpecializer(Graph& graph): g(graph) {
    if (getProfilingMode() && isBackwardGraph(g)) {
      auto vif = versionGraph();
      specializeAutogradOps(vif->blocks()[0]);
    }
    else {
      setStatesOnGraphInputs();
      specializeAutogradOps(g.block());
    }
  }

private:
  Graph& g;
  enum class State { Nonzero, Zero, Unknown };
  std::unordered_map<Value*, State> state;

  void setStatesOnGraphInputs() {
    for (Value* input : g.inputs()) {
      const auto& tp = input->type();
      if (auto tt = tp->cast<TensorType>()) {
        if (tt->undefined()) {
          if (*tt->undefined()) {
            state[input] = State::Zero;
          } else {
            state[input] = State::Nonzero;
          }
        } else {
          state[input] = State::Unknown;
        }
      } else if (
          tp->isSubtypeOf(TensorType::get()) ||
          tp->isSubtypeOf(ListType::ofTensors())) {
        state[input] = State::Nonzero;
      } else {
        state[input] = State::Unknown;
      }
    }
  }

  Node* versionGraph() {
    auto vif = g.create(prim::If, {}, g.outputs().size());
    auto true_block = vif->addBlock();
    auto false_block = vif->addBlock();
    auto value_map = [](Value* v) { return v; };
    true_block->cloneFrom(g.block(), value_map);
    replaceBlockInputsWithGraph(&g, true_block);
    // block for specialize_autogradzero to optimize
    false_block->cloneFrom(g.block(), value_map);
    replaceBlockInputsWithGraph(&g, false_block);

    auto ret = g.return_node();
    for (size_t i = 0; i < ret->inputs().size(); i++) {
      auto ogo = ret->input(i);
      auto ngo = vif->output(i);
      ngo->copyMetadata(ogo);
      ret->replaceInput(i, ngo);
    }
    
    // remove all original nodes
    // for (auto it = g.nodes().begin(); it != g.nodes().end(); ++it) {
    //   it.destroyCurrent();
    // }

    g.prependNode(vif);

    // insert type checks on inputs
    WithInsertPoint wip{vif};
    Value* cond = g.insertConstant(true);
    Value* check_result = nullptr;
    //for (auto inp : g.inputs()) {
    for (size_t i = 0; i < g.inputs().size(); i++) {
      auto inp = g.inputs()[i];
      if (auto tt = inp->type()->cast<TensorType>()) {
        // TODO: consider using a specialized op
        auto guard = g.create(prim::TypeCheck, {inp, cond}, 2);
        g.insertNode(guard);
        // we ignore go0 as we autodiff only non side-effectful ops
        auto go0 = guard->output(0);
        inp->replaceAllUsesAfterNodeWith(guard, go0);
        cond = guard->output(1);
        check_result = guard->output(1);
        go0->setType(inp->type());
        check_result->setType(BoolType::get());
        // set state
        if (tt->undefined()) {
          state[go0] = *tt->undefined() ? State::Zero : State::Nonzero;
        }
        else {
          state[go0] = State::Unknown;
        }
      }
    }
    // since we already know there are AnyAutogradNonZero it means
    // we would at least have one check
    TORCH_INTERNAL_ASSERT(check_result);
    vif->addInput(check_result);
    GRAPH_DUMP("specialize_autogradzero ", &g);
    return vif;
  }

  void specializeAutogradOps(Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
      auto n = *it;
      switch (n->kind()) {
        case prim::AutogradAdd: {
          auto a = n->input(0);
          auto b = n->input(1);
          // if one is Autograd zero, we can just drop the add
          if (state[a] == State::Zero) {
            // Zero + b == b
            n->output()->replaceAllUsesWith(b);
            it.destroyCurrent();
          } else if (state[b] == State::Zero) {
            // a + Zero == a
            n->output()->replaceAllUsesWith(a);
            it.destroyCurrent();
          } else if (state[a] == State::Nonzero && state[b] == State::Nonzero) {
            // when both are Nonzero, we can use a normal, optimizable add
            // instruction
            WithInsertPoint guard(n);
            auto* g = n->owningGraph();
            auto* cOne = g->insertConstant(1);
            auto* add_node = g->insertNode(g->create(aten::add, 1));
            add_node->addInput(a);
            add_node->addInput(b);
            add_node->addInput(cOne);
            auto* add_output = add_node->output();
            add_output->setType(n->output()->type());
            state[add_output] = State::Nonzero;
            n->output()->replaceAllUsesWith(add_output);
            it.destroyCurrent();
          } else {
            // otherwise we have conditionally-Nonzero things, and we need
            // to actually run an AutogradAdd which will guard for Zeros
            // so we leave the op as is
            state[n->output()] = State::Unknown;
          }
        } break;
        case prim::AutogradZero: {
          state[n->output()] = State::Zero;
        } break;
        case prim::profile: {
          // if prim::profile doesn't have an input
          // it's a counter to keep track how many times
          // a graph was profiled
          if (n->inputs().size() > 0) {
            state[n->output()] = State::Unknown;
            // state[n->input()];
          }
          break;
        }
        case prim::BailOut: {
          if (auto ptt = n->output()->type()->expect<TensorType>()) {
            state[n->output()] = ptt->undefined()
                ? *ptt->undefined() ? State::Zero : State::Nonzero
                : State::Unknown;
          }
        } break;
        case prim::Guard: {
          if (auto ptt = n->output()->type()->expect<TensorType>()) {
            state[n->output()] = ptt->undefined()
                ? *ptt->undefined() ? State::Zero : State::Nonzero
                : State::Unknown;
          }
        } break;
        // Lowered GradOf block
        case prim::If: {
          auto if_input = n->input(0)->node();
          if (if_input->kind() == prim::AutogradAnyNonZero) {
            auto all_zeros = std::all_of(
                if_input->inputs().begin(),
                if_input->inputs().end(),
                [&](Value* v) { return state[v] == State::Zero; });

            auto all_nonzeros = std::all_of(
                if_input->inputs().begin(),
                if_input->inputs().end(),
                [&](Value* v) { return state[v] == State::Nonzero; });
            // Property 1: if all the gradInputs to the GradOf are Zero
            // then the gradOutputs are also zero and will be represented as
            // AutogradZero nodes
            if (all_zeros) {
              auto zero = g.createAutogradZero()->insertAfter(n)->output();
              state[zero] = State::Zero;
              for (auto o : n->outputs()) {
                o->replaceAllUsesWith(zero);
              }
              it.destroyCurrent();
              break;
            }

            if (all_nonzeros) {
              auto body = n->blocks().at(0);
              // hoist the nodes in the GradOf body to be before the linear block
              for (auto it = body->nodes().begin(); it != body->nodes().end();) {
                auto block_node = *it++;
                block_node->moveBefore(n);
              }

              for (size_t i = 0; i < n->outputs().size(); ++i) {
                n->outputs().at(i)->replaceAllUsesWith(body->outputs().at(i));
                state[body->outputs().at(i)] = State::Nonzero;
              }
              it.destroyCurrent();
              break;
            }
          }

          for (auto o : n->outputs()) {
            state[o] = State::Unknown;
          }
          break;
        }
        default:
          for (auto o : n->outputs()) {
            state[o] = State::Unknown;
          }
          break;
      }
    }
  }

};


// propagate autograd zero information through a gradient graph and
// remove grad_of blocks if present.
// Note: this is a very limited pass. It only propagates autograd zeros for
// operations generated by the symbolic autodiff code and cleans up
// AutogradAdds when possible. Outputs of other nodes are conservatively
// marked Unknown and not optimized.
void specializeAutogradZero(Graph& g) {
  AutogradZeroSpecializer azs(g);
}

} // namespace jit
} // namespace torch
