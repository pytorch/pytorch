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
#include "jit/api/function_impl.h"

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
      auto vif = prepareGraph();
      if (!vif) {
        return;
      }
      GRAPH_DUMP("After versioning graph", &g);
      specializeAutogradOps(vif->blocks()[0]);
    }
    else {
      setStatesOnGraphInputs();
      specializeAutogradOps(g.block());
    }
    GRAPH_DUMP("After specializeAutogradOps graph", &g);
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

  static Value* getProfiledUse(Value* inp) {
    for (auto use : inp->uses()) {
      if (use.user->kind() == prim::profile) {
        return use.user->output();

      }
    }

    return nullptr;
  }

  Node* prepareGraph() {

    auto vif = g.create(prim::If, {}, g.outputs().size());
    auto value_map = [](Value* v) { return v; };
    auto true_block = vif->addBlock();
    auto false_block = vif->addBlock();
    // we will optimize true_block
    true_block->cloneFrom(g.block(), value_map);
    replaceBlockInputsWithGraph(&g, true_block);


    auto cold_graph = std::make_shared<Graph>();
    cold_graph->block()->cloneFrom(g.block(), value_map);

    auto otypes = c10::fmap(cold_graph->return_node()->inputs(), [](Value* v) {return v->type(); });
    auto tuple_type = TupleType::create(otypes);
    auto return_tuple = cold_graph->createTuple(cold_graph->return_node()->inputs());

    // {
    //   WithInsertPoint wip(*cold_graph->block()->nodes().begin()); 
    //   auto debug_print_cnst = cold_graph->insertConstant(IValue{std::string("cold_graph")});
    //   auto print_stmt = cold_graph->insert(prim::Print, {debug_print_cnst});
    // }
    
    cold_graph->appendNode(return_tuple);
    for (int i = cold_graph->outputs().size() - 1; i >= 0; i--) {
      cold_graph->eraseOutput(i);
    }
    cold_graph->registerOutput(return_tuple->output());
    GRAPH_DUMP("cold_graph", cold_graph);
    auto fp = new GraphFunction("<tensorexpr>", cold_graph, nullptr);
    {
      WithInsertPoint wip(false_block->return_node()); 
      Value* fn_constant = g.insertNode(g.create(prim::Constant))
        ->s_(attr::name, "tensorexpr")
        ->output()
        ->setType(FunctionType::create(fp));
          std::vector<Value*> inputs = {fn_constant};
        Value* result = g.insertNode(g.create(prim::CallFunction, inputs))
                          ->output()
                          ->setType(tuple_type);

      for (auto inp: g.inputs()) {
        result->node()->addInput(inp);
      }
      auto fun_unpack_tuple = g.insertNode(g.createTupleUnpack(result));
      for (auto out : fun_unpack_tuple->outputs()) {
        false_block->registerOutput(out);
      }
    }


    
    //replaceBlockInputsWithGraph(&g, false_block);

    

    auto optimize = true;
    WithInsertPoint wip{*g.nodes().begin()};
    
    std::vector<Node*> checks;
    std::set<Value*> checked;
    for (auto inp: g.inputs()) {
        if (inp->uses().size() == 0 || !inp->type()->cast<TensorType>()) {
          continue;
        }

        auto pout = getProfiledUse(inp);
        if (!pout) {
          state[inp] = State::Unknown;
          continue;
        }

        if (pout->node()->kind() != prim::profile) {
          GRAPH_DUMP("assertion : ", &g);
          GRAPH_DEBUG("assertion node: ", getHeader(pout->node()));
        }
        TORCH_INTERNAL_ASSERT(pout->node()->kind() == prim::profile);
        auto pttp = pout->type()->cast<TensorType>();
        if (!pttp->undefined().has_value()) {
          GRAPH_DEBUG("%", inp->debugName(), " can't be specialized");
          optimize = false;
          break;
        }
        state[inp] = *pttp->undefined() ? State::Zero : State::Nonzero;
        auto check = g.insert(prim::AutogradAnyNonZero, {inp});
        if (*pttp->undefined()) {
          check = g.insert(aten::__not__, {check});
        }
        checks.push_back(check->node());
    }

    // {
    //   WithInsertPoint wip2{*false_block->nodes().begin()};
    //   std::vector<Value*> print_inputs;
    //   auto debug_print_cnst = g.insertConstant(IValue{std::string("autodiff check triggered")});
    //   print_inputs.push_back(debug_print_cnst);
    //   auto blank_cnst = g.insertConstant(IValue{std::string(" ")});
      
    //   for (auto check : checks) {
    //     print_inputs.push_back(blank_cnst);
    //     auto node_name_str =  g.insertConstant(IValue{std::string( getHeader(check))});
    //     print_inputs.push_back(node_name_str);
    //     print_inputs.push_back(blank_cnst);
    //     print_inputs.push_back(check->output());
    //   }
    //   auto print_stmt = g.create(prim::Print, 0);
    //   g.insertNode(print_stmt);
    //   for (auto pv: print_inputs) {
    //     //GRAPH_DEBUG("Adding %", pv->debugName(), " to print statement ", getHeader(print_stmt->node()));
    //     print_stmt->addInput(pv);
    //   }
    // }

    //
    if (!optimize) {
      vif->destroy();
      // the checks we inserted will be cleaned up
      // by any subsequent DCE pass
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

    GRAPH_DUMP("After prepareGraph", &g);
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
          // this a profile node on a tensor use
          // not a counter
          // if we decided to specialize this graph
          // its input may have undefinedness info
          // otherwise it should be Unknown
          if (n->inputs().size() > 0) {
              state[n->output()] = !state.count(n->input()) ? State::Unknown : state[n->output()] = state[n->input()];
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
