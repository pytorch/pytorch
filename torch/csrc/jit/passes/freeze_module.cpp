#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/freeze_module.h>

#include <torch/csrc/jit/graph_executor_impl.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/inliner.h>

#include <stack>

namespace torch {
namespace jit {

namespace {

class AttributePropagator {
 public:
  AttributePropagator(script::Module& module) : module_(module) {}

  void run(std::shared_ptr<Graph>& graph) {
    Inline(*graph);
    propagateAttributes(graph);
    runOptimization(graph, /* unroll? */ false);
    cleanupFrozenModule(graph);
  }

 private:
  // findConstantAttr function locates the sub Module where attributes are
  // defined. The algorithm chases getAttr chains to locate the submodules.
  // For example:
  // module M {
  //   attributes {
  //     A = <SubModule at ...>
  //   }
  //   ...
  //   %A = prim::GetAttr[name="A"](%self)
  //   ...
  //   %B = prim::GetAttr[name="B"](%A)
  //   ...
  //   %weight = prim::GetAttr[name="scale"](%B)
  //   ...
  //   submodules {
  //     module SubModule {
  //       attributes {
  //          B = <SubModule2 at ...>
  //       }
  //       submodules {
  //         module SubModule2 {
  //            attributes {
  //               scale = 2
  //            }
  //         }
  //       }
  //     }
  //   }
  //
  // findConstantAttr(%B, "scale", M)  returns true because there are no
  // explicit SetAttr that modifies %B. attrModule points to the module where
  // attribute lives (in this example it is <SubModule2 at ...>).
  //
  // Note inplace mutations to attributes are checked later using alias
  // analysis.
  //
  // We can use a more efficient algorithm to hash each constant GetAttr to its
  // corresponding value. Based on initial test on resnet50 and other torch
  // vision tests. GetAttrs are not too frequent so it is ok to chase GetAttr
  // chain to retrieve their values.
  bool findConstantAttr(
      Node* node,
      std::string& name,
      script::Module& attrModule) {
    std::stack<std::string> names;
    while (!(node->outputs()[0]->type() == attrModule.type())) {
      if (node->kind() == prim::GetAttr) {
        names.push(node->s(attr::name));
        node = node->inputs()[0]->node();
      } else {
        return false;
      }
    }

    while (!names.empty()) {
      auto moduleName = names.top();
      names.pop();
      attrModule = attrModule.attr(moduleName).toModule();
      auto it = preservedAttrs_.find(attrModule._ivalue());
      if (it != preservedAttrs_.end()) {
        if (it->second.count(moduleName)) {
          return false;
        }
      }
    }
    auto it = preservedAttrs_.find(attrModule._ivalue());
    return it == preservedAttrs_.end() || !it->second.count(name);
  }

  void recordMutableAttrs(std::shared_ptr<Graph>& graph) {
    std::stack<Block*> blocks({graph->block()});
    std::unique_ptr<AliasDb> aliasDb =
        torch::make_unique<AliasDb>(graph, /* isFrozen */ true);
    while (!blocks.empty()) {
      Block* block = blocks.top();
      blocks.pop();
      for (auto n : block->nodes()) {
        for (Block* sub_block : n->blocks()) {
          blocks.push(sub_block);
        }
        if (n->kind() == prim::SetAttr || n->kind() == prim::GetAttr) {
          auto name = n->s(attr::name);
          auto inputNode = n->inputs()[0]->node();
          auto attrModule = module_;
          if (!findConstantAttr(inputNode, name, attrModule)) {
            continue;
          }
          if (n->kind() == prim::SetAttr || aliasDb->hasOutputWriters(n)) {
            GRAPH_DEBUG(
                n->kind() == prim::GetAttr ? "attribute: " + name + " in %" +
                        n->outputs()[0]->debugName() + " has inplace writer"
                                           : "");
            preservedAttrs_[attrModule._ivalue()].insert(name);
          }
        }
      }
    }
  }

  std::set<std::string> recordReferencedAttrs(std::shared_ptr<Graph>& graph) {
    std::stack<Block*> blocks({graph->block()});
    while (!blocks.empty()) {
      Block* block = blocks.top();
      blocks.pop();
      for (auto n : block->nodes()) {
        for (Block* subBlock : n->blocks()) {
          blocks.push(subBlock);
        }
        if (n->kind() == prim::GetAttr) {
          auto& name = n->s(attr::name);
          if (module_.hasattr(name)) {
            preservedAttrs_[module_._ivalue()].insert(name);
          }
        }
      }
    }
    auto it = preservedAttrs_.find(module_._ivalue());
    if (it != preservedAttrs_.end())
      return it->second;
    else
      return std::set<std::string>();
  }

  // If Module is in eval mode, detach and override requires_grad tensor
  // attributes to enable freezing.
  bool shouldFoldAttr(IValue& attr, std::string& name, bool isEval) {
    if (attr.isTensor()) {
      auto t = attr.toTensor();
      if (t.requires_grad()) {
        t = autograd::as_variable_ref(t).detach();
        attr = IValue(t);
        if (isEval) {
          t.set_requires_grad(false);
        } else {
          return false;
        }
      }
    } else if (attr.isTensorList()) {
      c10::List<at::Tensor> lst = std::move(attr).toTensorList();
      for (size_t i = 0; i < lst.size(); ++i) {
        auto t = autograd::as_variable_ref(lst.extract(i)).detach();
        lst.set(i, t);
        if (isEval) {
          t.set_requires_grad(false);
        } else {
          return false;
        }
      }
      attr = std::move(lst);
    }

    // Do not fold training attribute in training mode.
    return isEval || name != "training";
  }

  void propagateAttributes(std::shared_ptr<Graph>& graph) {
    std::unordered_map<
        script::ModulePtr,
        std::unordered_map<std::string, Value*>>
        attrValues;
    auto isEval = !module_.is_training();
    GRAPH_DEBUG("Freezing Module in ", isEval ? "eval mode" : "training mode");
    auto block = graph->block();
    std::stack<Block*> blocks({block});

    // Record Attributes that are explicitely set in the module. They cannot be
    // folded.
    recordMutableAttrs(graph);

    Node* m = *block->nodes().begin();
    WithInsertPoint guard(m);
    while (!blocks.empty()) {
      Block* block = blocks.top();
      blocks.pop();
      for (auto it = block->nodes().begin(); it != block->nodes().end();) {
        Node* n = *it;
        it++; // advance iterator bc the current node may be destroyed

        for (Block* sub_block : n->blocks()) {
          blocks.push(sub_block);
        }
        if (n->kind() == prim::GetAttr) {
          auto name = n->s(attr::name);
          auto attrModule = module_;
          auto inputNode = n->inputs()[0]->node();
          if (!findConstantAttr(inputNode, name, attrModule)) {
            GRAPH_DEBUG("attribute: ", name, " is mutable.")
            continue;
          }
          assert(attrModule.hasattr(name));
          Value* paramConst = nullptr;
          auto I = attrValues.find(attrModule._ivalue());
          if (I != attrValues.end()) {
            auto II = I->second.find(name);
            if (II != I->second.end())
              paramConst = II->second;
          }
          if (!paramConst) {
            auto attr = attrModule.attr(name);
            if (!shouldFoldAttr(attr, name, isEval))
              continue;
            if (auto attrVal = tryInsertConstant(*graph, attr)) {
              paramConst = *attrVal;
            } else {
              GRAPH_DEBUG(
                  attr.type()->cast<ClassType>() ? "" : "attribute: ",
                  name,
                  " is not materializable.");
              continue;
            }
            auto moduleName = attrModule.type()->name()->qualifiedName();
            moduleName += ".";
            moduleName += name;
            paramConst->setDebugName(moduleName);
            attrValues[attrModule._ivalue()][name] = paramConst;
          }
          GRAPH_UPDATE(
              "Folding GetAttr %",
              n->outputs()[0]->debugName(),
              " with ",
              paramConst->debugName());
          n->outputs().at(0)->replaceAllUsesWith(paramConst);
          n->removeAllInputs();
        }
      }
    }
  }

  // cleanupFrozenModule function cleans up the Frozen module. it performs the
  // following:
  // 1) Remove unused attributes.
  // 2) Remove unreferenced submodules
  // 3) Remove non pulic unreferenced methods.
  // TODO: do #3 because there is no API to 'unsafely' remove methods.
  void cleanupFrozenModule(std::shared_ptr<Graph>& graph) {
    std::vector<std::string> attrsToRemove;
    auto type = module_.type();
    size_t N = type->numAttributes();
    auto KeepAttrs = recordReferencedAttrs(graph);
    for (size_t i = 0; i < N; ++i) {
      auto attrTy = type->getAttribute(i);
      auto name = type->getAttributeName(i);
      if (!KeepAttrs.count(name)) {
        attrsToRemove.push_back(name);
      }
    }
    for (auto& name : attrsToRemove) {
      module_._ivalue()->unsafeRemoveAttr(name);
      module_.type()->unsafeRemoveAttribute(name);
    }
  }

  // Contains attributes that can't be folded or user directs to keep them.
  std::unordered_map<script::ModulePtr, std::set<std::string>> preservedAttrs_;

  script::Module& module_;
}; // class AttributePropagator
} // namespace

script::Module freezeModule(const script::Module& module) {
  // Currently freezing module is supported only in eval mode.
  // If assertion below is commented, this implementation folds attributes
  // correctly in training mode.Tensor attributes with required_grad set
  // are not folded and 'training' attribute is also not folded.
  // TODO: Determine if freezing in training s useful and clarify its semantics.
  TORCH_INTERNAL_ASSERT(!module.is_training());
  auto moduleClone = module.clone();
  AttributePropagator attrPropagator(moduleClone);
  script::Method method = moduleClone.get_method("forward");
  auto graph = method.graph();
  attrPropagator.run(graph);
  GRAPH_DUMP(
      moduleClone.type()->name()->name() + "::forward() after freezing module",
      method.graph());
  return moduleClone;
}

} // namespace jit
} // namespace torch
