#include <torch/csrc/jit/passes/freeze_module.h>

#include <torch/csrc/jit/jit_log.h>

#include <c10/util/irange.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/autocast.h>
#include <torch/csrc/jit/passes/clear_profiling.h>
#include <torch/csrc/jit/passes/eliminate_no_ops.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/runtime/graph_executor_impl.h>

#include <stack>
#include <utility>

namespace torch {
namespace jit {

namespace {

std::vector<std::string> splitName(const std::string& name) {
  std::vector<std::string> result;
  std::string sub_name;
  std::istringstream name_stream(name);
  while (std::getline(name_stream, sub_name, '.')) {
    result.push_back(std::move(sub_name));
  }
  return result;
}

template <typename Iter>
std::string concatName(const Iter& begin, const Iter& end) {
  std::string combined_name = "";
  for (Iter it = begin; it != end; ++it) {
    const std::string& sub_name = *it;
    if (!combined_name.empty()) {
      combined_name += ".";
    }
    combined_name += sub_name;
  }
  return combined_name;
}

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
    auto checkName = [this](std::string& name) {
      const auto resolved_name = resolveName(name);

      if (resolved_name) {
        const auto& parent_module = resolved_name->first;
        const auto& attr_name = resolved_name->second;
        if (parent_module.hasattr(attr_name)) {
          auto value = parent_module.attr(attr_name);
          // Freezing client wants to preserve this submodule. When cleaning
          // the frozen module, make sure it will be preserved entirely.
          if (value.isModule()) {
            preservedSubModule_.insert(value.toModule()._ivalue());
          }
          insertMutableAttr(attr_name, value, parent_module._ivalue());
        } else {
          auto fn = parent_module.get_method(attr_name);
          preservedMethods_.insert(&fn.function());
        }
        return true;
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
#ifndef C10_MOBILE
      Autocast(subgraph);
#endif
      runOptimization(
          subgraph,
          /* unroll_non_constant_loops? */ false,
          /* const_prop_user_classes? */ false);
      EliminateNoOps(subgraph);
      LowerSimpleTuples(subgraph);
    };

    std::unordered_map<std::string, std::unordered_set<std::string>>
        interfacesToReassignType;

    for (auto function : preservedMethods_) {
      GRAPH_DEBUG("Analyzing function: " + function->name());
      auto graph = toGraphFunction(*function).graph();
      optimizeSubGraphs(graph, applyInline);
      if (freezeInterfaces_) {
        inlineInterfaceCalls(graph, interfacesToReassignType);
      }
    }

    reassignInterfaceTypes(interfacesToReassignType);

    for (auto function : preservedMethods_) {
      GRAPH_DEBUG("Recording mutable attrs for function: " + function->name());
      auto graph = toGraphFunction(*function).graph();
      // Record Attributes that are explicitly set in the module.
      // They cannot be folded.
      recordMutableAttrs(graph);
    }

    for (auto function : preservedMethods_) {
      GRAPH_DEBUG("Propagating function: " + function->name());
      auto graph = toGraphFunction(*function).graph();
      propagateAttributes(graph);
      optimizeSubGraphs(graph, applyOptimizations);
    }
    GRAPH_DEBUG("Cleaning up module");
    cleanupFrozenModule();
  }

 private:
  using ResolvedName = std::pair<Module, std::string>;

  // Try to resolve qualified names (submodule1.submodule2.foo). If
  // the qualified name exists in the root module, return the unqualified
  // attribute/function name and the parent module. Else, return nullopt.
  // Examples:
  // submodule1.submodule2.foo -> {submodule2, "foo"}
  // submodule1.non_existent_module.foo -> nullopt
  std::optional<ResolvedName> resolveName(const std::string& name) {
    auto sub_names = splitName(name);
    if (sub_names.empty()) {
      return c10::nullopt;
    }
    auto& attr_name = sub_names.back();
    auto cur_module = module_;
    std::vector<ResolvedName> attr_infos;
    attr_infos.reserve(sub_names.size() - 1);

    for (size_t i = 0; i < sub_names.size() - 1; ++i) {
      bool found = false;
      const auto& sub_name = sub_names[i];
      for (const auto& child_module : cur_module.named_children()) {
        if (child_module.name == sub_name) {
          attr_infos.emplace_back(cur_module._ivalue(), child_module.name);
          cur_module = child_module.value;
          found = true;
          break;
        }
      }
      if (!found) {
        return c10::nullopt;
      }
    }

    if (cur_module.hasattr(attr_name) || cur_module.find_method(attr_name)) {
      // We don't want to mark these modules as mutable yet; that could
      // interfere with the inlining procedure. Instead, we'll record
      // the fact that the user wants to preserve them. They will be
      // processed during clean-up preparation (recordReferenceAttrs)
      for (auto& attr_info : attr_infos) {
        const auto& parent_module = attr_info.first;
        auto& sub_name = attr_info.second;
        userPreservedAttrs_[parent_module._ivalue()].insert(
            std::move(sub_name));
      }
      return std::make_pair(std::move(cur_module), std::move(attr_name));
    }

    return c10::nullopt;
  }

  bool _loadModulePath(Value* input, std::shared_ptr<Graph>& graph) {
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

    return true;
  }

  std::optional<std::deque<std::string>> getModulePath(
      Value* input,
      std::shared_ptr<Graph>& graph) {
    bool success = _loadModulePath(input, graph);
    if (!success) {
      return c10::nullopt;
    }
    return names_;
  }

  template <typename Iter>
  bool getModuleFromPath(
      Module& attrModule,
      const Iter& begin,
      const Iter& end) {
    for (Iter it = begin; it != end; ++it) {
      const std::string& moduleName = *it;
      if (preservedAttrs_.count(attrModule.attr(moduleName))) {
        return false;
      }
      attrModule = attrModule.attr(moduleName).toModule();
    }
    return true;
  }

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

    // loads the path into this->names_
    if (!_loadModulePath(input, graph)) {
      return false;
    }

    // reassigns attrModule to the module in names_
    if (!getModuleFromPath(attrModule, names_.begin(), names_.end())) {
      return false;
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
        std::make_unique<AliasDb>(graph, /* isFrozen */ true);
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
      const auto& elems = tuple->elements();
      for (const auto idx : c10::irange(elems.size())) {
        tuple->unsafeSetElement(idx, overrideGradient(elems[idx]));
      }
      attr = std::move(tuple);
    } else if (attr.isList()) {
      c10::List<IValue> elems = std::move(attr).toList();
      for (const auto i : c10::irange(elems.size())) {
        elems.set(i, overrideGradient(elems.extract(i)));
      }
      attr = elems;
    } else if (attr.isGenericDict()) {
      auto dict = std::move(attr).toGenericDict();
      for (const auto& pair : dict) {
        auto val = pair.value();
        val = overrideGradient(std::move(val));
      }
      attr = dict;
    } else if (attr.isObject() && !attr.toObjectRef().type()->is_module()) {
      auto obj_type = attr.type()->expect<ClassType>();
      auto obj_value = std::move(attr).toObject();
      auto sub_attributes = obj_type->getAttributes();
      for (const auto& sub_attr : sub_attributes) {
        auto sub_attr_val = obj_value->getAttr(sub_attr.getName());
        sub_attr_val = overrideGradient(std::move(sub_attr_val));
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
        if (auto graphFunction = tryToGraphFunction(function)) {
          GRAPH_UPDATE(
              "Inlining interface method '",
              function.name(),
              "' to ",
              *user_node);

          GRAPH_UPDATE("Function body: ", graphFunction->optimized_graph());
          inlineCallTo(user_node, graphFunction);
          inlined = true;
        }
      }
    }
    return inlined;
  }

  //   [Note: Inlining interfaces strategy]
  // There's two structures that are relevant to freezing:
  // - the graph describing the computation in a method
  // - the module describing the data structure of the module instance.
  //
  // First, in inlineInterfaceCalls, we inline interfaces. This is done in a
  // separate step from normal inlining because CallMethod on an interface type
  // requires extra steps compared to inlining a normal CallMethod.
  //
  // Next we need to simplify the structure of the module data structure, which
  // is done for the most part by the usual steps in cleanupFrozenModule.
  //
  // However, there's a complication that comes from the fact that within a
  // method, you can change the value of an interface to another module that
  // implements that interface.
  //
  // For example:
  //
  // impl: MyInterface
  // ...
  // def forward(self, x):
  //     if x > 0:
  //         self.impl = my_interface_impl
  //
  // This is disallowed in freezing, because in this case we can't flatten out
  // the module structure, since the type of self.impl will change.
  //
  // To handle this, we do the following:
  //   1. inlineInterfaceCalls:
  //     a. inline the graph, and in the process record all interfaces
  //     b. simultaneously, check (throw error) for disallowed SetAttr calls.
  //   2. call reassignInterfaceTypes, which reassigns interface types to their
  //      concrete types. This is done in a separate step to avoid interfering
  //      with inlineInterfaceCalls (note: this may not need to be done as a
  //      separate step)
  //   3. eventually cleanupFrozenModule will reorder the module data structure
  //      and it will expect that all interface types have been removed.
  void inlineInterfaceCalls(
      std::shared_ptr<Graph>& graph,
      std::unordered_map<std::string, std::unordered_set<std::string>>&
          interfacesToRetype) {
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

          // Record this so that we can reassign the type later
          // in reassignInterfaceTypes()
          // See [Note: Inlining interfaces strategy]
          auto path = getModulePath(input, graph);
          TORCH_INTERNAL_ASSERT(path.has_value());
          auto path_str = concatName(path->begin(), path->end());
          interfacesToRetype[path_str].insert(name);
        } else if (n->kind() == prim::SetAttr) {
          // Check to make sure we're not assigning the value of any parameters
          // that are interface types.
          // See [Note: Inlining interfaces strategy]
          auto name = n->s(attr::name);
          auto attrModule = module_;
          auto input = n->inputs()[0];

          if (!input->type()->cast<InterfaceType>() &&
              !input->type()->expectRef<ClassType>().is_module()) {
            // we only care if we're setattr["thing"](%mod) if %mod
            continue;
          }

          // note: this will modify attrModule until it is the parent of the
          // "name" attr. In other words, attrModule is now the module that
          // matches "input".
          // We can't use findConstantAttr in case the base item is an object,
          // instead of a module/interface.
          auto path = getModulePath(input, graph);
          TORCH_INTERNAL_ASSERT(path.has_value());
          getModuleFromPath(attrModule, path->begin(), path->end());

          const auto& attrType = attrModule.type()->getAttribute(name);
          TORCH_INTERNAL_ASSERT(
              !attrType->cast<InterfaceType>(),
              "Freezing does not support SetAttr on an interface type. ",
              "SetAttr is attempted on '",
              name,
              "'");
        } else if (n->kind() == prim::fork) {
          applyToForkSubgraph(
              n,
              graph,
              // NOLINTNEXTLINE(modernize-avoid-bind)
              std::bind(
                  &AttributePropagator::inlineInterfaceCalls,
                  *this,
                  std::placeholders::_1,
                  interfacesToRetype));
        }
      }
    }
  }

  // See [Note: Inlining interfaces strategy]
  // This modifies the internal structure of module types to reassign the
  // type from an interface type to its concrete type.
  void reassignInterfaceTypes(
      const std::unordered_map<std::string, std::unordered_set<std::string>>&
          interfacesToRetype) {
    for (const auto& it : interfacesToRetype) {
      const std::string& modulePath = it.first;
      const std::vector<std::string>& splitPath = splitName(modulePath);
      Module attrModule = module_;
      getModuleFromPath(attrModule, splitPath.begin(), splitPath.end());

      for (const std::string& name : it.second) {
        auto subvalue = attrModule.attr(name);
        auto subvalueType = subvalue.type();
        attrModule.type()->unsafeChangeAttributeType(name, subvalueType);
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
            if (attr.isObject()) {
              if (object_memo_.count(attr.toObject())) {
                attr = object_memo_[attr.toObject()];
              } else {
                auto weak_class_obj =
                    attr.toObject()->copy_to_weak_compilation_ref();
                object_memo_[attr.toObject()] = weak_class_obj;
                attr = weak_class_obj;
              }
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
      if (subModule.type()->isSubtypeOf(*output->type())) {
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
      auto graph = toGraphFunction(*function).graph();
      recordReferencedAttrs(graph);
      handleSharedClassType(module_, graph);
      removeExtraWaitCalls(graph->block());
      toGraphFunction(*function).clear_optimized_graphs();
    }
    removeUnusedAttrs();
  }

  // Preparing for clean up phase. At this point, record all subModules that
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
    // We have to process the attributes that the user wants to preserve
    // separately since it's possible that the user-preserved module is
    // never referenced in the graph.
    for (const auto& attr_info : userPreservedAttrs_) {
      const auto& parent_module = attr_info.first;
      for (const auto& attr_name : attr_info.second) {
        const auto value = parent_module->getAttr(attr_name);
        insertMutableAttr(attr_name, value, parent_module);
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
      // Preserve all its attributes and methods.
      attrsToKeep_[type].insert(N);
      return;
    }
    auto it2 = preservedScalarAttrs_.find(module._ivalue());
    SharedTypeSubModules_[type].insert(module._ivalue());
    attrsToKeep_[type].insert({});
    for (const auto i : c10::irange(N)) {
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
          // See [Note: Inlining interfaces strategy]
          TORCH_CHECK(
              !type->getAttribute(i)->cast<InterfaceType>(),
              "Unexpected interface attribute '" + name + "' during freezing");

          auto attrModule = attr.toModule();
          handleSharedClassType(attrModule, graph);
        }
      }
    }
  }

  // Remove unused attributes and methods for each sub module of the frozen
  // module. This function iterates over the Classtypes of its submodule
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
      for (const auto i : c10::irange(N)) {
        if (it.second.count(i) == 0) {
          attrsToRemove.push_back(type->getAttributeName(i));
        }
      }
      for (auto& fn : type->methods()) {
        if (preservedMethods_.count(fn)) {
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

  // see [Constant Object Weak CompilationUnit Reference]
  std::unordered_map<
      c10::intrusive_ptr<at::ivalue::Object>,
      c10::intrusive_ptr<at::ivalue::Object>>
      object_memo_;

  // Contains names of attributes that the user wants to preserve with
  // their owning modules.
  std::unordered_map<ModulePtr, std::unordered_set<std::string>>
      userPreservedAttrs_;

}; // class AttributePropagator

void checkModuleDoesNotReturnSelf(const Module& module) {
  if (module.find_method("forward")) {
    Method method = module.get_method("forward");
    // Check that module does not return itself.
    for (auto& output : method.graph()->outputs()) {
      TORCH_CHECK(
          output->type() != module.type(),
          "attempted to freeze a module that return itself");
    }
  }
}
} // namespace

Module freeze_module(
    const Module& module,
    std::vector<std::string> preservedAttrs,
    bool freezeInterfaces,
    bool preserveParameters) {
  checkModuleDoesNotReturnSelf(module);

  auto moduleClone = module.clone(true);
  AttributePropagator attrPropagator(
      moduleClone, preservedAttrs, freezeInterfaces, preserveParameters);
  attrPropagator.run();
  return moduleClone;
}

void freeze_module_inplace(
    Module* module,
    std::vector<std::string> preservedAttrs,
    bool freezeInterfaces,
    bool preserveParameters) {
  TORCH_CHECK(module != nullptr, "module cannot be nullptr");
  checkModuleDoesNotReturnSelf(*module);
  AttributePropagator attrPropagator(
      *module, preservedAttrs, freezeInterfaces, preserveParameters);
  attrPropagator.run();
}

} // namespace jit
} // namespace torch
