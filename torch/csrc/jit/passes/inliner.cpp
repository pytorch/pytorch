#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/jit/script/error_report.h>

namespace torch {
namespace jit {

namespace prim {
using namespace ::c10::prim;
}

class Inliner {
 public:
  Inliner() = default;

  void removeFunctionConstants(Block* b)
  {
    for (auto it = b->nodes().begin(); it != b->nodes().end(); it++)
    {
      if (it->kind() == prim::Constant && it->output()->type()->isSubclass(TypeKind::FunctionType))
      {
        it.destroyCurrent();
      }
      else
      {
        for (auto ib : it->blocks())
        {
            removeFunctionConstants(ib);
        }
      }
    }
  }

  void unifyIfs(Block* b)
  {
    for (auto n : b->nodes())
    {
      for (auto ib : n->blocks())
      {
          unifyIfs(ib);
      }

      if (n->kind() == prim::If)
      {
        auto true_outputs = n->blocks().at(0)->outputs();
        auto false_outputs = n->blocks().at(1)->outputs();
        AT_ASSERT(true_outputs.size() == false_outputs.size());
        AT_ASSERT(true_outputs.size() == n->outputs().size());
        for (size_t i = 0; i < true_outputs.size(); i++)
        {
          auto unified = unifyTypes(true_outputs.at(i)->type(), false_outputs.at(i)->type());
          if (!unified) {
            throw script::ErrorReport()
                << "if-expression's true branch has type " << true_outputs.at(i)->type()->str()
                << " but false branch has type " << false_outputs.at(i)->type()->str();
          }
          n->outputs()[i]->setType(*unified);
        }

      }
    }

  }
  void run(Block* block, bool recurse) {
    Node* cur = block->nodes().front();
    Node* end = block->return_node();

    while (cur != end) {
      auto next = cur->next();
      if (cur->kind() == prim::CallFunction) {
        AT_ASSERT(cur->inputs().at(0)->node()->kind() == prim::Constant);
        auto function_constant = cur->inputs().at(0)->node();
        auto fun_type = function_constant->output()->type()->expect<FunctionType>();
        auto graph = fun_type->function()->graph();

        auto old_output = cur->outputs();
        // slice function ptr value
        auto inputs = cur->inputs().slice(1);
        WithInsertPoint guard(next);
        auto new_output =
            inlineCallTo(*cur->owningGraph(), *graph.get(), inputs).at(0);
        if (old_output.at(0)->hasUniqueName()) {
          auto name = old_output.at(0)->uniqueName();
          new_output->setUniqueName(name);
        }

        old_output.at(0)->replaceAllUsesWith(new_output);
        next = cur->next();
        cur->destroy();
      } else if (recurse) {
        if (cur->hasAttribute(attr::Subgraph)) {
          auto fg = cur->g(attr::Subgraph);
          run(fg->block(), recurse);
        }

        for (auto b : cur->blocks()) {
          run(b, recurse);
        }
      }
      cur = next;
    }
  }
};

void Inline(Block* block, bool recurse) {
  Inliner inliner{};
  inliner.run(block, recurse);
  inliner.unifyIfs(block);
  inliner.removeFunctionConstants(block);

}

} // namespace jit
} // namespace torch
