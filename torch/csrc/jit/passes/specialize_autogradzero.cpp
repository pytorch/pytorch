#include <torch/csrc/jit/passes/specialize_autogradzero.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/clear_undefinedness.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

namespace torch {
namespace jit {

struct AutogradZeroSpecializer {
  enum class State { Nonzero, Zero, Unknown };
  AutogradZeroSpecializer(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  void run() {
    setStatesOnGraphInputs();
    specializeAutogradOps(graph_->block());
  }

 private:
  void setStatesOnGraphInputs() {
    for (Value* input : graph_->inputs()) {
      const auto& tp = input->type();
      if (auto tt = tp->cast<TensorType>()) {
        if (tt->undefined()) {
          if (*tt->undefined()) {
            state_[input] = State::Zero;
          } else {
            state_[input] = State::Nonzero;
          }
        } else {
          state_[input] = State::Unknown;
        }
      } else if (
          tp->isSubtypeOf(TensorType::get()) ||
          tp->isSubtypeOf(ListType::ofTensors())) {
        state_[input] = State::Nonzero;
      } else {
        state_[input] = State::Unknown;
      }
    }
  }

  void specializeAutogradOps(Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
      auto n = *it;
      switch (n->kind()) {
        case prim::AutogradAdd: {
          auto a = n->input(0);
          auto b = n->input(1);
          // if one is Autograd zero, we can just drop the add
          if (state_[a] == State::Zero) {
            // Zero + b == b
            n->output()->replaceAllUsesWith(b);
            it.destroyCurrent();
          } else if (state_[b] == State::Zero) {
            // a + Zero == a
            n->output()->replaceAllUsesWith(a);
            it.destroyCurrent();
          } else if (
              state_[a] == State::Nonzero && state_[b] == State::Nonzero) {
            // when both are Nonzero, we can use a normal, optimizable add
            // instruction
            WithInsertPoint guard(n);
            auto* cOne = graph_->insertConstant(1);
            auto* add_node = graph_->insertNode(graph_->create(aten::add, 1));
            add_node->addInput(a);
            add_node->addInput(b);
            add_node->addInput(cOne);
            auto* add_output = add_node->output();
            add_output->setType(n->output()->type());
            state_[add_output] = State::Nonzero;
            n->output()->replaceAllUsesWith(add_output);
            it.destroyCurrent();
          } else {
            // otherwise we have conditionally-Nonzero things, and we need
            // to actually run an AutogradAdd which will guard for Zeros
            // so we leave the op as is
            state_[n->output()] = State::Unknown;
          }
        } break;
        case prim::AutogradZero: {
          state_[n->output()] = State::Zero;
        } break;
        case prim::profile: {
          // this a profile node on a tensor use
          // if we decided to specialize this graph
          // its input may have undefinedness info
          // otherwise it should be Unknown
          if (n->inputs().size() > 0) {
            state_[n->output()] = !state_.count(n->input())
                ? State::Unknown
                : state_[n->output()] = state_[n->input()];
          }
          break;
        }
        case prim::BailOut: {
          if (auto ptt = n->output()->type()->expect<TensorType>()) {
            state_[n->output()] = ptt->undefined()
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
                [&](Value* v) { return state_[v] == State::Zero; });

            auto all_nonzeros = std::all_of(
                if_input->inputs().begin(),
                if_input->inputs().end(),
                [&](Value* v) { return state_[v] == State::Nonzero; });
            // Property 1: if all the gradInputs to the GradOf are Zero
            // then the gradOutputs are also zero and will be represented as
            // AutogradZero nodes
            if (all_zeros) {
              auto zero =
                  graph_->createAutogradZero()->insertAfter(n)->output();
              state_[zero] = State::Zero;
              for (auto o : n->outputs()) {
                o->replaceAllUsesWith(zero);
              }
              it.destroyCurrent();
              break;
            }

            if (all_nonzeros) {
              auto body = n->blocks().at(0);
              // hoist the nodes in the GradOf body to be before the linear
              // block
              for (auto it = body->nodes().begin();
                   it != body->nodes().end();) {
                auto block_node = *it++;
                block_node->moveBefore(n);
              }

              for (size_t i = 0; i < n->outputs().size(); ++i) {
                n->outputs().at(i)->replaceAllUsesWith(body->outputs().at(i));
                state_[body->outputs().at(i)] = State::Nonzero;
              }
              it.destroyCurrent();
              break;
            }
          }

          for (auto o : n->outputs()) {
            state_[o] = State::Unknown;
          }
          break;
        }
        default:
          for (auto o : n->outputs()) {
            state_[o] = State::Unknown;
          }
          break;
      }
    }
  }

  std::shared_ptr<Graph> graph_;
  std::unordered_map<Value*, State> state_;
};

// propagate autograd zero information through a gradient graph and
// remove grad_of blocks if present.
// Note: this is a very limited pass. It only propagates autograd zeros for
// operations generated by the symbolic autodiff code and cleans up
// AutogradAdds when possible. Outputs of other nodes are conservatively
// marked Unknown and not optimized.
void specializeAutogradZero(std::shared_ptr<Graph> g) {
  AutogradZeroSpecializer azs(g);
  azs.run();
}

} // namespace jit
} // namespace torch
