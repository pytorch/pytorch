#include <torch/csrc/jit/passes/specialize_autogradzero.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include "ATen/core/interned_strings.h"
#include "ATen/core/jit_type.h"
#include "c10/util/Exception.h"
#include "jit/frontend/tree_views.h"
#include "jit/ir/constants.h"
#include "jit/ir/ir.h"
#include "jit/jit_log.h"
#include <torch/csrc/jit/passes/clear_undefinedness.h>
#include "jit/passes/dead_code_elimination.h"

namespace torch {
namespace jit {




static void foldGradSumToSize(Block* b) {

  for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
        auto n = *it;
        if (it->kind() == aten::_grad_sum_to_size) {
          auto in_type = it->input(0)->type()->expect<TensorType>();
          auto out_type = it->output()->type()->expect<TensorType>();
          auto in_syms = in_type->symbolic_sizes();
          auto out_syms = out_type->symbolic_sizes();
          if (in_syms.rank().has_value() && out_syms.rank().has_value()) {
            if (*in_syms.sizes() == *out_syms.sizes()) {
              GRAPH_DEBUG("Removing ", getHeader(*it));
              it->input(0)->setType(it->output()->type());
              it->output()->replaceAllUsesWith(it->input(0));
              it.destroyCurrent();
              continue;
            }
            auto max_rank = std::max(*out_syms.rank(), *in_syms.rank());
            auto in_sizes = *in_syms.sizes();
            auto padded_dims = max_rank - *out_syms.rank();

            // TODO: Figure out how to reshape back
            if (padded_dims) {
              continue;
            }

            // std::vector<c10::ShapeSymbol> out_sizes(padded_dims, c10::ShapeSymbol::fromStaticSize(1));
            // auto orig_out_sizes = *out_syms.sizes();
            // out_sizes.insert(out_sizes.end(),orig_out_sizes.begin(), orig_out_sizes.end());
            auto out_sizes = *out_syms.sizes();

            std::vector<int64_t> reduce_axes; 
            bool dimension_match = true;
            for (size_t i = 0; i < max_rank; i++) {
              if (out_sizes[i].is_static()) {
                if (out_sizes[i].static_size() == 1) {
                  reduce_axes.push_back(i);
                }
                // no reduction since out_sizes isn't equal to 1
              } else if (out_sizes[i] != in_sizes[i]) {
                dimension_match = false;
                break;
              }
            }

            if (!dimension_match) {
              continue;
            }

          auto graph = b->owningGraph();
          auto axes_constant = graph->insertConstant(IValue{reduce_axes});
          auto keep_dims = graph->insertConstant(IValue{true});
          auto sum = graph->insert(aten::sum, {it->input(0), axes_constant, keep_dims});

          // if (padded_dims > 0) {
          //   sum = graph->insert(aten::reshape_as, {sum, });
          // }
          n->output()->replaceAllUsesWith(sum);
          it.destroyCurrent();
        }
      else {
        for (auto ib : n->blocks()) {
          foldGradSumToSize(ib);
        }
      }
    }

  }
}

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
      //removeProfilingNodes(g.block());
      GRAPH_DUMP("After removing profiling nodes", &g);
      auto vif = versionGraph2();
      if (!vif) {
        return;
      }
      GRAPH_DUMP("After versioning graph", &g);
      specializeAutogradOps(vif->blocks()[0]);
      GRAPH_DUMP("After specializeAutogradOps graph", &g);
      foldGradSumToSize(vif->blocks()[0]);
      GRAPH_DUMP("After FoldGraphSumToSize graph", &g);
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

  static void removeProfilingNodes(Block* b) {
    for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
      if (it->kind() == prim::profile) {
        if (it->outputs().size()) {
          it->input()->setType(it->output()->type());
          it->output()->replaceAllUsesWith(it->input());
        }
        it.destroyCurrent();
      } else {
        for (Block* ib : it->blocks()) {
          removeProfilingNodes(ib);
        }
      }
    }
  }

  std::vector<Node*> collectAutogradAnyNonZeros(Block* b) {

    std::vector<Node*> aanznodes;

    for (auto n : b->nodes()) {
      if (n->kind() == prim::AutogradAnyNonZero) {
        aanznodes.push_back(n);
      }
    }

    return aanznodes;
  }

  Node* versionGraph2() {


    //AutogradAnyNonZero


    auto should_version = true;
    WithInsertPoint wip{*g.nodes().begin()};

    auto vif = g.create(prim::If, {}, g.outputs().size());
    auto value_map = [](Value* v) { return v; };
    auto true_block = vif->addBlock();
    auto false_block = vif->addBlock();
    true_block->cloneFrom(g.block(), value_map);
    replaceBlockInputsWithGraph(&g, true_block);
    // block for specialize_autogradzero to optimize
    false_block->cloneFrom(g.block(), value_map);
    replaceBlockInputsWithGraph(&g, false_block);
    
    std::vector<Node*> checks;
    std::set<Value*> checked;
    // auto orig_checks = collectAutogradAnyNonZeros(true_block);
    // for (auto orig_check: orig_checks) {
      for (auto inp: g.inputs()) {
          if (inp->uses().size() == 0 || !inp->type()->cast<TensorType>()) {
            continue;
          }

          auto pout = inp->uses()[0].user->output();
          if (pout->node()->kind() != prim::profile) {
            GRAPH_DUMP("versionGraph2: ", &g);
            GRAPH_DEBUG("puse= ", getHeader(pout->node()));
          }
          TORCH_INTERNAL_ASSERT(pout->node()->kind() == prim::profile);
          auto pttp = pout->type()->cast<TensorType>();
          if (!pttp->undefined().has_value()) {
            GRAPH_DEBUG("%", inp->debugName(), " shouldn't be versioned");
            should_version = false;
            break;
          }
          state[inp] = *pttp->undefined() ? State::Zero : State::Nonzero;
          auto check = g.insert(prim::AutogradAnyNonZero, {inp});
          if (!*pttp->undefined()) {
            check = g.insert(aten::__not__, {check});
          }
          checks.push_back(check->node());
      }
    //}

    if (!should_version) {
      return nullptr;
    }

    TORCH_INTERNAL_ASSERT(checks.size() > 0);
    auto conjunction = checks[0]->output();
    if (checks.size() > 1) {
      for (size_t i = 1; i < checks.size(); i++) {
        conjunction = g.insert(aten::__and__, {checks[i]->output(), conjunction});
      }
    }

    vif->addInput(conjunction);
    g.insertNode(vif);

    auto ret = g.return_node();
    for (size_t i = 0; i < ret->inputs().size(); i++) {
      auto ogo = ret->input(i);
      auto ngo = vif->output(i);
      ngo->copyMetadata(ogo);
      ret->replaceInput(i, ngo);
    }

    GRAPH_DUMP("specialize_autogradzero ", &g);

    ClearUndefinedness(g.inputs());
    ClearUndefinedness(vif->blocks()[1]);

    return vif;
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

    // replace backwards outputs
    auto ret = g.return_node();
    for (size_t i = 0; i < ret->inputs().size(); i++) {
      auto ogo = ret->input(i);
      auto ngo = vif->output(i);
      ngo->copyMetadata(ogo);
      ret->replaceInput(i, ngo);
    }
    g.prependNode(vif);

    // insert type checks on inputs
    WithInsertPoint wip{vif};
    Value* cond = g.insertConstant(true);
    Value* check_result = nullptr;
    for (size_t i = 0; i < g.inputs().size(); i++) {
      auto inp = g.inputs()[i];
      if (auto tt = inp->type()->cast<TensorType>()) {
        // auto uses = inp->uses();
        // if (uses.empty()) {
        //   continue;
        // }

        // c10::TensorTypePtr merged_type = nullptr;
        // for (auto use : uses) {
        //   auto utype = use.user->type()->cast<TensorType>();
        //   if (!merged_type) {
        //     merged_type = utype;
        //   }
        //   else {
        //     merged_type = merged_type->merge(utype);
        //   }
        // }


        // TODO: consider using a specialized op
        auto guard = g.create(prim::TypeCheck, {inp, cond}, 2);
        g.insertNode(guard);
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

    ClearUndefinedness(g.inputs());
    ClearUndefinedness(vif->blocks()[1]);
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
          if (n->inputs().size() > 0) {
            if (!state.count(n->input())) {
              state[n->output()] = State::Unknown;
            }
            else {
              state[n->output()] = state[n->input()];
            }
          }
          // if prim::profile doesn't have an input
          // it's a counter to keep track how many times
          // a graph was profiled
          // if (n->inputs().size() > 0) {
          //   state[n->output()] = State::Unknown;
          //   // state[n->input()];
          // }
          // else {
          //   if (auto ptt = n->output()->type()->expect<TensorType>()) {
          //     state[n->output()] = ptt->undefined()
          //         ? *ptt->undefined() ? State::Zero : State::Nonzero
          //         : State::Unknown;
          //   }
          // }
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
