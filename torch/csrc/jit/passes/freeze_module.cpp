#include <torch/csrc/jit/passes/freeze_module.h>

#include <torch/csrc/jit/jit_log.h>

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/clear_profiling.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/runtime/graph_executor_impl.h>

#include <stack>

namespace torch {
namespace jit {

namespace {

class AttributePropagator {
 public:
  AttributePropagator(
      Module& module,
      std::vector<std::string>& preservedAttrs,
      bool freezeInterfaces,
      bool preserveParameters)
      : module_(module),
        freezeInterfaces_(freezeInterfaces),
        preserveParameters_(preserveParameters) {
    // Currently only top level attributes and functions can  be preserved
    // explicitly.
    auto checkName = [this](std::string& name) {
      if (module_.hasattr(name)) {
        auto attr = module_.attr(name);

        // Freezing client wants to presever this submodule. When cleaning
        // the frozen module, make sure it will be preserved entirely.
        if (attr.isModule()) {
          preservedSubModule_.insert(attr.toModule()._ivalue());
        }
        insertMutableAttr(name, attr, module_._ivalue());
        return true;
      }

      for (auto& fn : module_.type()->methods()) {
        if (fn->name() == name) {
          preservedMethods_.insert(fn);
          return true;
        }
      }
      return false;
    };

    // forward is preserved by default, but
    // not all modules have a forward function defined
    if (module_.find_method("forward")) {
      auto method = module_.get_method("forward");
      preservedMethods_.insert(&method.function());
    }

    for (auto name : preservedAttrs) {
      TORCH_CHECK(checkName(name), "Unknown name: " + name);
    }
  }

  void optimizeSubGraphs(
      std::shared_ptr<Graph>& graph,
      const std::function<void(std::shared_ptr<Graph>&)>& func) {
    func(graph);
    std::stack<Block*> blocks({graph->block()});
    while (!blocks.empty()) {
      Block* block = blocks.top();
      blocks.pop();
      for (auto n : block->nodes()) {
        for (Block* sub_block : n->blocks()) {
          blocks.push(sub_block);
        }
        if (n->kind() == prim::fork) {
          auto subgraph = n->g(attr::Subgraph);
          optimizeSubGraphs(subgraph, func);
        }
      }
    }
  }

  void run() {
    auto applyInline = [](std::shared_ptr<Graph>& subgraph) {
      Inline(*subgraph);
      ClearProfilingInformation(subgraph);
    };
    auto applyOptimizations = [](std::shared_ptr<Graph>& subgraph) {
      runOptimization(
          subgraph, /* unroll? */ false, /* const_prop_user_classes? */ false);
      LowerSimpleTuples(subgraph);
    };

    for (auto function : preservedMethods_) {
      GRAPH_DEBUG("Analyzing function: " + function->name());
      auto graph = function->graph();
      optimizeSubGraphs(graph, applyInline);
      if (freezeInterfaces_) {
        inlineInterfaceCalls(graph);
      }
      // Record Attributes that are explicitly set in the module.
      // They cannot be folded.
      recordMutableAttrs(graph);
    }

    for (auto function : preservedMethods_) {
      GRAPH_DEBUG("Propagating function: " + function->name());
      auto graph = function->graph();
      propagateAttributes(graph);
      optimizeSubGraphs(graph, applyOptimizations);
    }
    GRAPH_DEBUG("Cleaning up module");
    cleanupFrozenModule();
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
      Value* input,
      std::string& name,
      Module& attrModule,
      std::shared_ptr<Graph>& graph) {
    if (!input->type()->cast<InterfaceType>() &&
        !input->type()->expectRef<ClassType>().is_module()) {
      return false;
    }

    Node* node = input->node();
    names_.clear();
    while (!(node->outputs()[0]->type() == graph->inputs()[0]->type())) {
      if (node->kind() == prim::GetAttr) {
        names_.push_front(node->s(attr::name));
        node = node->inputs()[0]->node();
      } else {
        return false;
      }
    }

    for (auto& moduleName : names_) {
      if (preservedAttrs_.count(attrModule.attr(moduleName))) {
        return false;
      }
      attrModule = attrModule.attr(moduleName).toModule();
    }

    auto attr = attrModule.attr(name);
    if (!AliasDb::isMutableType(attr.type())) {
      auto it = preservedScalarAttrs_.find(attrModule._ivalue());
      return it == preservedScalarAttrs_.end() || !it->second.count(name);
    }

    if (preservedAttrs_.count(attr)) {
      return false;
    }
    if (!attr.type()->cast<ClassType>()) {
      for (auto& ivalue : preservedAttrs_) {
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
      const ModulePtr& attrModule) {
    if (AliasDb::isMutableType(attr.type())) {
      preservedAttrs_.insert(attr);
    } else {
      preservedScalarAttrs_[attrModule].insert(name);
    }
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

        // Modules with prim::ModuleContainerIndex cannot be frozen because they
        // return InterfaceTypes.
        TORCH_CHECK(
            n->kind() != prim::ModuleContainerIndex,
            "Freezing modules containing prim::ModuleContainerIndex is not supported");

        if (n->kind() == prim::SetAttr || n->kind() == prim::GetAttr) {
          // By default if interface attributes are present then fail freezing.
          // If freezingInterfaces is on then Interfaces are folded similarly
          // to other attributes.
          TORCH_CHECK(
              freezeInterfaces_ ||
                  !(n->kind() == prim::GetAttr &&
                    n->output()->type()->cast<InterfaceType>()),
              "attempted to freeze a module that uses interface attributes");
          auto name = n->s(attr::name);
          auto attrModule = module_;
          if (!findConstantAttr(n->inputs()[0], name, attrModule, graph)) {
            continue;
          }

          auto attr = attrModule.attr(name);
          if (n->kind() == prim::GetAttr) {
            auto type = n->output()->type();
            // Do not record submodules. Their attributes are tracked
            // individually.
            if (attr.isObject() || !AliasDb::isMutableType(attr.type())) {
              continue;
            }
            usedAttrs_.insert(attr);
          }

          if (n->kind() == prim::SetAttr || aliasDb->hasOutputWriters(n)) {
            GRAPH_DEBUG(
                n->kind() == prim::GetAttr ? "attribute: " + name + " in %" +
                        n->output()->debugName() + " has inplace writer"
                                           : "attribute: " + name + " is set");
            auto mptr = attrModule._ivalue();
            insertMutableAttr(name, attr, mptr);
          }
        } else if (n->kind() == prim::fork) {
          applyToForkSubgraph(
              n,
              graph,
              // NOLINTNEXTLINE(modernize-avoid-bind)
              std::bind(
                  &AttributePropagator::recordMutableAttrs,
                  *this,
                  std::placeholders::_1));
        }
      }
    }
    // FIXME: Current Alias analysis fails to track subvalues.
    // This is not a common scenario, for freezing, detect and error out.
    IValue::HashAliasedIValues seen;
    for (auto& val : usedAttrs_) {
      IValue::HashAliasedIValues subValues;
      val.getSubValues(subValues);
      TORCH_CHECK(
          std::all_of(
              subValues.begin(),
              subValues.end(),
              [&seen](const IValue& v) { return seen.count(v) == 0; }),
          "module contains attributes values that overlaps ",
          val);
      seen.insert(subValues.begin(), subValues.end());
    }
  }

  IValue overrideGradient(IValue attr) {
    if (attr.isTensor()) {
      auto& t = attr.toTensor();
      if (t.requires_grad()) {
        auto detached = t.detach();
        detached.set_requires_grad(false);
        attr = IValue(std::move(detached));
      }
    } else if (attr.isTuple()) {
      auto tuple = std::move(attr).toTuple();
      std::vector<IValue>& elems = tuple->elements();
      for (auto& elem : elems) {
        elem = overrideGradient(elem);
      }
      attr = std::move(tuple);

    } else if (attr.isList()) {
      c10::List<IValue> elems = std::move(attr).toList();
      for (size_t i = 0; i < elems.size(); i++) {
        elems.set(i, overrideGradient(elems.extract(i)));
      }
      attr = std::move(elems);
    } else if (attr.isGenericDict()) {
      auto dict = std::move(attr).toGenericDict();
      for (const auto& pair : dict) {
        auto val = pair.value();
        val = overrideGradient(val);
      }
      attr = std::move(dict);
    } else if (attr.isObject() && !attr.toObjectRef().type()->is_module()) {
      auto obj_type = attr.type()->expect<ClassType>();
      auto obj_value = std::move(attr).toObject();
      auto sub_attributes = obj_type->getAttributes();
      for (const auto& sub_attr : sub_attributes) {
        auto sub_attr_val = obj_value->getAttr(sub_attr.getName());
        sub_attr_val = overrideGradient(sub_attr_val);
      }
      return obj_value;
    }

    return attr;
  }

  // This method is invoked only when 'freezeInterfaces' parameter is on.
  // The module associated with Interface is retrieved and the invoked method
  // is inlined.
  bool inlineInterfaceCall(Node* n, const IValue& attr) {
    auto class_type = attr.type()->expect<ClassType>();
    bool inlined = false;
    for (auto use : n->output()->uses()) {
      auto user_node = use.user;
      if (user_node->kind() == prim::CallMethod) {
        const std::string& methodName = user_node->s(attr::name);
        Function& function = class_type->getMethod(methodName);
        if (!function.isGraphFunction()) {
          continue;
        }
        GRAPH_UPDATE(
            "Inlining interface method '",
            function.name(),
            "' to ",
            *user_node);

        GRAPH_UPDATE("Function body: ", *function.optimized_graph());
        inlineCallTo(user_node, &function);
        inlined = true;
      }
    }
    return inlined;
  }

  void inlineInterfaceCalls(std::shared_ptr<Graph>& graph) {
    auto block = graph->block();
    std::stack<Block*> blocks({block});

    while (!blocks.empty()) {
      Block* block = blocks.top();
      blocks.pop();
      for (auto n : block->nodes()) {
        for (Block* sub_block : n->blocks()) {
          blocks.push(sub_block);
        }
        if (n->kind() == prim::GetAttr) {
          if (!n->output()->type()->cast<InterfaceType>()) {
            continue;
          }
          auto name = n->s(attr::name);
          auto attrModule = module_;
          auto input = n->inputs()[0];
          TORCH_CHECK(
              findConstantAttr(input, name, attrModule, graph),
              "failed to freeze interface attribute '" + name + "'");
          TORCH_INTERNAL_ASSERT(attrModule.hasattr(name));
          auto attr = attrModule.attr(name);
          inlineInterfaceCall(n, attr);
          // Reset the GetAttr to concrete module type.
          n->output()->setType(attr.type());
        } else if (n->kind() == prim::fork) {
          applyToForkSubgraph(
              n,
              graph,
              // NOLINTNEXTLINE(modernize-avoid-bind)
              std::bind(
                  &AttributePropagator::inlineInterfaceCalls,
                  *this,
                  std::placeholders::_1));
        }
      }
    }
  }

  void propagateAttributes(std::shared_ptr<Graph>& graph) {
    std::unordered_map<ModulePtr, std::unordered_map<std::string, Value*>>
        attrValues;
    auto isEval = !module_.hasattr("training") || !module_.is_training();
    GRAPH_DEBUG("Freezing Module: ", module_.type()->name()->name());
    auto block = graph->block();
    std::stack<Block*> blocks({block});

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
          auto input = n->inputs()[0];
          if (!findConstantAttr(input, name, attrModule, graph)) {
            GRAPH_DEBUG(
                input->type()->cast<InterfaceType>() ||
                        input->type()->expectRef<ClassType>().is_module()
                    ? "attribute: " + name + " is mutable."
                    : "");
            continue;
          }
          TORCH_INTERNAL_ASSERT(attrModule.hasattr(name));
          Value* paramConst = nullptr;
          auto iter = attrValues.find(attrModule._ivalue());
          if (iter != attrValues.end()) {
            auto iter2 = iter->second.find(name);
            if (iter2 != iter->second.end())
              paramConst = iter2->second;
          }
          if (!paramConst) {
            auto attr = attrModule.attr(name);
            if (!isEval || preserveParameters_) {
              auto type = attrModule.type();
              auto slot = *type->findAttributeSlot(name);
              if (type->is_parameter(slot) || type->is_buffer(slot) ||
                  (attr.isObject() &&
                   !attr.toObjectRef().type()->is_module())) {
                continue;
              } else {
                attr = overrideGradient(attr);
              }
              if (!isEval && name == "training") {
                continue;
              }
            } else {
              attr = overrideGradient(attr);
            }
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
        } else if (n->kind() == prim::fork) {
          applyToForkSubgraph(
              n,
              graph,
              // NOLINTNEXTLINE(modernize-avoid-bind)
              std::bind(
                  &AttributePropagator::propagateAttributes,
                  *this,
                  std::placeholders::_1));
        }
      }
    }
  }

  void applyToForkSubgraph(
      Node* n,
      std::shared_ptr<Graph>& graph,
      const std::function<void(std::shared_ptr<Graph>&)>& func) {
    TORCH_CHECK(n->kind() == prim::fork);
    auto attrModule = module_;
    auto node = n->inputs()[0]->node();
    // Check if first parameter of fork is a module. This module is used
    // as the base module (similar to 'self' in forward) to resolve GetAttrs.
    //  Otherwise freezing is applied using module_
    if (node->kind() == prim::GetAttr &&
        node->output()->type()->cast<ClassType>()) {
      auto name = node->s(attr::name);
      auto input = node->inputs()[0];
      if (!findConstantAttr(input, name, attrModule, graph)) {
        // Module needs to be preserved.
        return;
      }
      attrModule = attrModule.attr(name).toModule();
      std::swap(module_, attrModule);
    }

    auto subgraph = n->g(attr::Subgraph);
    func(subgraph);
    module_ = attrModule;
  }

  bool moduleEscapes(Module& subModule, std::shared_ptr<Graph>& graph) {
    for (auto& output : graph->outputs()) {
      if (subModule.type()->isSubtypeOf(output->type())) {
        return true;
      }
    }
    return preservedSubModule_.count(subModule._ivalue());
  }

  void removeExtraWaitCalls(Block* b) {
    auto nodes = b->nodes();
    for (auto it = nodes.begin(); it != nodes.end(); it++) {
      auto node = *it;
      if (node->kind() != aten::wait) {
        continue;
      }
      TORCH_INTERNAL_ASSERT(node->inputs().size() == 1);
      TORCH_INTERNAL_ASSERT(node->outputs().size() == 1);
      // If input type is not a from aten::fork call then the
      // aten::wait operator can be deleted.
      if (node->input()->type()->kind() != TypeKind::FutureType) {
        node->output()->replaceAllUsesWith(node->input());
        it.destroyCurrent();
      }
    }
    // For the remaining nodes, recurse.
    for (auto it = nodes.begin(); it != nodes.end(); it++) {
      auto node = *it;
      for (auto sub_b : node->blocks()) {
        removeExtraWaitCalls(sub_b);
      }
    }
  }

  // cleanupFrozenModule function cleans up the Frozen module. It performs the
  // following:
  // 1) Remove unused attributes.
  // 2) Remove unreferenced submodules
  // 3) Remove non public unreferenced methods.
  void cleanupFrozenModule() {
    for (auto function : preservedMethods_) {
      auto graph = function->graph();
      recordReferencedAttrs(graph);
      handleSharedClassType(module_, graph);
      removeExtraWaitCalls(graph->block());
    }
    removeUnusedAttrs();
  }

  // Prepraring for clean up phase. At this point, record all subModules that
  // contains mutable attributes.
  void recordReferencedAttrs(std::shared_ptr<Graph>& graph) {
    std::stack<Block*> blocks({graph->block()});
    std::set<ModulePtr> modules({module_._ivalue()});
    while (!blocks.empty()) {
      Block* block = blocks.top();
      blocks.pop();
      for (auto n : block->nodes()) {
        for (Block* subBlock : n->blocks()) {
          blocks.push(subBlock);
        }
        if (n->kind() == prim::GetAttr) {
          auto& name = n->s(attr::name);
          // For now, use all module ivalues which are the same type
          // and could be the module that this GetAttr resolves to
          // TODO: we could attempt to follow the GetAttr chain and
          // find the exact ivalue, we would have to be careful
          // that the chain does not contain any attributes which
          // get written to (setAttr calls)
          for (auto& mptr : modules) {
            auto module = Module(mptr);
            if (module.type() == n->inputs()[0]->type()) {
              TORCH_INTERNAL_ASSERT(module.hasattr(name));
              auto module = Module(mptr);
              auto attr = module.attr(name);
              // TODO: this could be insertReferencedAttr to be more clear,
              // these are attributes we could not inline, which include
              // other reasons besides mutation (unsupported constant,
              // getAttr resolving to non-getAttr node, etc)
              insertMutableAttr(name, attr, mptr);
              if (attr.isModule()) {
                modules.insert(attr.toModule()._ivalue());
              }
            }
          }
        } else if (n->kind() == prim::fork) {
          applyToForkSubgraph(
              n,
              graph,
              // NOLINTNEXTLINE(modernize-avoid-bind)
              std::bind(
                  &AttributePropagator::recordReferencedAttrs,
                  *this,
                  std::placeholders::_1));
        }
      }
    }
  }

  // This function recursively iterates over submodules to identify
  // for each class type the attribute slots that need to be preserved.
  //
  // Note 'attrsToKeep[type].insert(type->numAttributes())' means all
  // attribute slots of 'type' and its methods are preserved. A submodule is
  // preserved when it escapes (meaning it is returned).
  void handleSharedClassType(Module& module, std::shared_ptr<Graph>& graph) {
    auto type = module.type();
    size_t N = type->numAttributes();
    if (moduleEscapes(module, graph)) {
      // Perserve all its attributes and methods.
      attrsToKeep_[type].insert(N);
      return;
    }
    auto it2 = preservedScalarAttrs_.find(module._ivalue());
    SharedTypeSubModules_[type].insert(module._ivalue());
    attrsToKeep_[type].insert({});
    for (size_t i = 0; i < N; ++i) {
      auto name = type->getAttributeName(i);
      auto attr = module.attr(name);
      auto attrTy = attr.type();

      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      bool isMutable;
      if (AliasDb::isMutableType(attrTy)) {
        isMutable = preservedAttrs_.count(attr);
      } else {
        isMutable =
            it2 != preservedScalarAttrs_.end() && it2->second.count(name);
      }
      if (isMutable) {
        attrsToKeep_[type].insert(i);
        if (attr.isModule()) {
          // FIXME: This error is conservative. Detected an interface module
          // that cannot be fully inlined away because of side effects.
          // TODO: We could allow freezing in this case but we would need to
          // 1) Change the module type to use the concrete type (attrTy).
          // Probably first unsafe remove attribute and add it using concrete
          // type.
          // 2) Fail if there is any setattr to an interface attribute bc
          // everything is inlined based on old value of this attribute.
          TORCH_CHECK(
              !type->getAttribute(i)->cast<InterfaceType>(),
              "failed to freeze interface attribute '" + name + "'");

          auto attrModule = attr.toModule();
          handleSharedClassType(attrModule, graph);
        }
      }
    }
  }

  // Remove unused attributes and methods for each sub module of the frozen
  // module. This function iterates over the Calsstypes of its submodule
  // attributes including its own type.
  void removeUnusedAttrs() {
    std::vector<std::string> attrsToRemove;
    std::vector<Function*> funcsToRemove;
    for (auto& it : attrsToKeep_) {
      auto& type = it.first;
      size_t N = type->numAttributes();
      if (it.second.count(N)) {
        continue;
      }
      for (size_t i = 0; i < N; ++i) {
        if (it.second.count(i) == 0) {
          attrsToRemove.push_back(type->getAttributeName(i));
        }
      }
      for (auto& fn : type->methods()) {
        if (preservedMethods_.count(fn) && *type == *module_.type()) {
          continue;
        }
        funcsToRemove.push_back(fn);
      }

      for (auto& name : attrsToRemove) {
        for (auto& val : SharedTypeSubModules_[type]) {
          auto mod = val.toModule();
          mod._ivalue()->unsafeRemoveAttr(name);
        }
        type->unsafeRemoveAttribute(name);
      }
      for (auto fn : funcsToRemove) {
        type->unsafeRemoveMethod(fn->name());
        auto mod = SharedTypeSubModules_[type].begin()->toModule();
        mod._ivalue()->compilation_unit()->unsafeRemoveMethod(fn->qualname());
      }

      attrsToRemove.clear();
      funcsToRemove.clear();
    }
  }

  // Contains attributes that can't be folded or user directs to keep them.
  IValue::HashAliasedIValues preservedAttrs_;
  // Tracked immutable types (Scalars) by their attribute names not
  // IValues.
  std::unordered_map<ModulePtr, std::unordered_set<std::string>>
      preservedScalarAttrs_;

  // Contains user specified methods to be preserved in frozen module.
  std::unordered_set<Function*> preservedMethods_;

  // Contains user specified sub module to be preserve in frozen module.
  std::unordered_set<ModulePtr> preservedSubModule_;

  // Track all used attributes ivalues that can be aliased.
  IValue::HashAliasedIValues usedAttrs_;

  // Contains the attribute slots that need to be preserved for each ClassType.
  std::unordered_map<ClassTypePtr, std::unordered_set<size_t>> attrsToKeep_;

  // Contains the sub modules that share the same ClassType.
  std::unordered_map<ClassTypePtr, IValue::HashAliasedIValues>
      SharedTypeSubModules_;

  Module& module_;

  // Allow to freeze modules containing interfaces.
  bool freezeInterfaces_;

  // Preserve module parameters
  bool preserveParameters_;

  // Contains the attributes names (e.g. {"self", "subModule", "a"}
  std::deque<std::string> names_;
}; // class AttributePropagator
} // namespace

Module freeze_module(
    const Module& module,
    std::vector<std::string> preservedAttrs,
    bool freezeInterfaces,
    bool preserveParameters) {
  if (module.find_method("forward")) {
    Method method = module.get_method("forward");
    // Check that module does not return itself.
    for (auto& output : method.graph()->outputs()) {
      TORCH_CHECK(
          output->type() != module.type(),
          "attempted to freeze a module that return itself");
    }
  }

  auto moduleClone = module.clone(true);
  AttributePropagator attrPropagator(
      moduleClone, preservedAttrs, freezeInterfaces, preserveParameters);
  attrPropagator.run();
  return moduleClone;
}

} // namespace jit
} // namespace torch
