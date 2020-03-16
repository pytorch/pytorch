#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/freeze_module.h>

#include <torch/csrc/jit/runtime/graph_executor_impl.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/inliner.h>

#include <stack>

namespace torch {
namespace jit {

namespace {

class AttributePropagator {
 public:
  AttributePropagator(Module& module) : module_(module) {}

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
      Module& attrModule) {
    names_.clear();
    while (!(node->outputs()[0]->type() == attrModule.type())) {
      if (node->kind() == prim::GetAttr) {
        names_.push_front(node->s(attr::name));
        node = node->inputs()[0]->node();
      } else {
        return false;
      }
    }

    for (auto& moduleName : names_) {
      auto it = preservedAttrs_.find(attrModule._ivalue());
      if (it != preservedAttrs_.end()) {
        if (it->second.count(attrModule.attr(moduleName))) {
          return false;
        }
      }
      attrModule = attrModule.attr(moduleName).toModule();
    }

    auto attr = attrModule.attr(name);
    if (!AliasDb::mutableType(attr.type())) {
      auto it = preservedScalarAttrs_.find(attrModule._ivalue());
      return it == preservedScalarAttrs_.end() || !it->second.count(name);
    }

    auto it = preservedAttrs_.find(attrModule._ivalue());
    if (it == preservedAttrs_.end()) {
      return true;
    }
    if (it->second.count(attr)) {
      return false;
    }
    if (!attr.type()->cast<ClassType>()) {
      for (auto& ivalue : it->second) {
        if (!ivalue.isObject() && ivalue.overlaps(attr)) {
          return false;
        }
      }
    }
    return true;
  }

  void insertMutableAttr(
      const std::string& name,
      const IValue& attr,
      Module& attrModule) {
    if (AliasDb::mutableType(attr.type())) {
      preservedAttrs_[attrModule._ivalue()].insert(attr);
    } else {
      preservedScalarAttrs_[attrModule._ivalue()].insert(name);
    }
  }

  void recordMutableAttrs(std::shared_ptr<Graph>& graph) {
    std::stack<Block*> blocks({graph->block()});
    std::unique_ptr<AliasDb> aliasDb =
        torch::make_unique<AliasDb>(graph, /* isFrozen */ true);
    IValue::HashAliasedIValues usedAttrs;
    while (!blocks.empty()) {
      Block* block = blocks.top();
      blocks.pop();
      for (auto n : block->nodes()) {
        for (Block* sub_block : n->blocks()) {
          blocks.push(sub_block);
        }
        if (n->kind() == prim::SetAttr || n->kind() == prim::GetAttr) {
          // TODO: handle interface attributes. For now, Exit if Module uses
          // inteface attributes
          if (n->kind() == prim::GetAttr) {
            TORCH_CHECK(
                !n->output()->type()->cast<InterfaceType>(),
                "attempted to freeze a module that uses interface attributes");
          }
          auto inputNode = n->inputs()[0]->node();
          auto name = n->s(attr::name);
          auto attrModule = module_;
          if (!findConstantAttr(inputNode, name, attrModule)) {
            continue;
          }

          auto attr = attrModule.attr(name);
          if (n->kind() == prim::GetAttr) {
            auto type = n->output()->type();
            // Do not record submodules. Their attributes are tracked
            // individually.
            if (attr.isObject() || !AliasDb::mutableType(attr.type())) {
              continue;
            }
            usedAttrs.insert(attr);
          }

          if (n->kind() == prim::SetAttr || aliasDb->hasOutputWriters(n)) {
            GRAPH_DEBUG(
                n->kind() == prim::GetAttr ? "attribute: " + name + " in %" +
                        n->output()->debugName() + " has inplace writer"
                                           : "attribute: " + name + " is set");
            insertMutableAttr(name, attr, attrModule);
          }
        }
      }
    }
    // FIXME: Current Alias analysis fails to track subvalues.
    // This is not a common scenario, for freezing, detect and error out.
    for (auto it = usedAttrs.begin(); it != usedAttrs.end();) {
      auto& val = *it;
      it++;
      for (auto rhs = it; rhs != usedAttrs.end(); rhs++) {
        TORCH_CHECK(
            !val.overlaps(*rhs),
            "module contains attributes values that overlaps ",
            val,
            " and ",
            *rhs);
      }
    }
  }

  // Prepraring for clean up phase. At this point, record all  subModules that
  // contains mutable attributes.
  void recordReferencedAttrs(std::shared_ptr<Graph>& graph) {
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
            auto attr = module_.attr(name);
            insertMutableAttr(name, attr, module_);
          }
        }
      }
    }
  }

  IValue overrideGradient(IValue attr) {
    if (attr.isTensor()) {
      auto t = attr.toTensor();
      if (t.requires_grad()) {
        t = autograd::as_variable_ref(t).detach();
        t.set_requires_grad(false);
        attr = IValue(t);
      }
    } else if (attr.isTuple()) {
      std::vector<IValue>& elems = attr.toTuple()->elements();
      for (auto& elem : elems) {
        elem = overrideGradient(elem);
      }
    } else if (attr.isList()) {
      c10::List<IValue> elems = std::move(attr).toList();
      for (size_t i = 0; i < elems.size(); i++) {
        elems.set(i, overrideGradient(elems.extract(i)));
      }
      attr = std::move(elems);
    }

    return attr;
  }

  void propagateAttributes(std::shared_ptr<Graph>& graph) {
    std::unordered_map<
        ModulePtr,
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
          TORCH_INTERNAL_ASSERT(attrModule.hasattr(name));
          Value* paramConst = nullptr;
          auto I = attrValues.find(attrModule._ivalue());
          if (I != attrValues.end()) {
            auto II = I->second.find(name);
            if (II != I->second.end())
              paramConst = II->second;
          }
          if (!paramConst) {
            auto attr = attrModule.attr(name);
            if (isEval)
              attr = overrideGradient(attr);
            if (auto attrVal = tryInsertConstant(*graph, attr)) {
              paramConst = *attrVal;
            } else {
              GRAPH_DEBUG(
                  attr.type()->cast<ClassType>() ? "" : "attribute: ",
                  name,
                  " is not materializable.");
              continue;
            }
            std::string fullName("self.");
            for (auto& name : names_) {
              fullName += name + '.';
            }
            fullName += name;
            paramConst->setDebugName(fullName);
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
  // 3) Remove non public unreferenced methods.
  // TODO: do #3 because there is no API to 'unsafely' remove methods.
  void cleanupFrozenModule(std::shared_ptr<Graph>& graph) {
    std::vector<std::string> attrsToRemove;
    auto type = module_.type();
    size_t N = type->numAttributes();
    recordReferencedAttrs(graph);
    auto it = preservedAttrs_.find(module_._ivalue());
    auto it2 = preservedScalarAttrs_.find(module_._ivalue());
    for (size_t i = 0; i < N; ++i) {
      auto attrTy = type->getAttribute(i);
      auto name = type->getAttributeName(i);
      auto attr = module_.attr(name);
      bool immutable;
      if (AliasDb::mutableType(attrTy)) {
        immutable = it == preservedAttrs_.end() || !it->second.count(attr);
      } else {
        immutable =
            it2 == preservedScalarAttrs_.end() || !it2->second.count(name);
      }
      if (immutable) {
        attrsToRemove.push_back(name);
      }
    }
    for (auto& name : attrsToRemove) {
      module_._ivalue()->unsafeRemoveAttr(name);
      module_.type()->unsafeRemoveAttribute(name);
    }
    for (auto& fn : type->methods()) {
      auto& name = fn->name();
      if ("forward" == name)
        continue;
      type->unsafeRemoveMethod(name);
      module_._ivalue()->compilation_unit()->unsafeRemoveMethod(fn->qualname());
    }
  }

  // Contains attributes that can't be folded or user directs to keep them.
  std::unordered_map<ModulePtr, IValue::HashAliasedIValues>
      preservedAttrs_;
  // Tracked immutable types (Scalars) by their attribute names not
  // IValues.
  std::unordered_map<ModulePtr, std::unordered_set<std::string>>
      preservedScalarAttrs_;

  Module& module_;

  // Contains the attributes names (e.g. {"self", "subModule", "a"}
  std::deque<std::string> names_;
}; // class AttributePropagator
} // namespace

Module freeze_module(const Module& module) {
  // Currently freezing module is supported only in eval mode.
  // If assertion below is commented and module is in training mode then this
  // implementation folds attributes correctly. Tensor attributes with
  // required_grad set are not folded and 'training' attribute is also not
  // folded.
  // TODO: Determine if freezing in training mode is useful and further clarify
  // its semantics.
  TORCH_CHECK(!module.is_training());

  Method method = module.get_method("forward");
  // Check that module does not return itself.
  for (auto& output : method.graph()->outputs())
    TORCH_CHECK(
        output->type() != module.type(),
        "attempted to freeze a module that return itself");

  auto moduleClone = module.clone();
  AttributePropagator attrPropagator(moduleClone);
  method = moduleClone.get_method("forward");
  auto graph = method.graph();
  attrPropagator.run(graph);
  GRAPH_DUMP(
      moduleClone.type()->name()->name() + "::forward() after freezing module",
      method.graph());
  return moduleClone;
}

} // namespace jit
} // namespace torch
