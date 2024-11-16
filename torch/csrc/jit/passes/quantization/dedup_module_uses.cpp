#include <torch/csrc/jit/passes/quantization/dedup_module_uses.h>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/quantization/helper.h>

#include <stack>

namespace torch::jit {
namespace {
class ModuleUseDeduper {
 public:
  ModuleUseDeduper(Module& module) : module_(module) {}
  void dedup() {
    for (auto& method : module_.get_methods()) {
      const auto& graph = method.graph();
      findModuleUses(graph.get());
    }
    dedupModuleUses();
  }

 private:
  // Analyze the code to record information represents
  // uses of the module, which we'll use later to actually perform the dedup
  // operation Please see the comments of member variables of the class for more
  // information
  void findModuleUses(Graph* graph) {
    GRAPH_DUMP("Finding module uses for ", graph);

    std::stack<Block*> blocks_to_visit;
    blocks_to_visit.push(graph->block());
    Value* self = graph->inputs()[0];
    while (!blocks_to_visit.empty()) {
      Block* b = blocks_to_visit.top();
      blocks_to_visit.pop();
      for (Node* n : b->nodes()) {
        for (Block* subblock : n->blocks()) {
          blocks_to_visit.push(subblock);
        }
        if (n->kind() != prim::CallMethod) {
          continue;
        }
        Value* instance = n->inputs()[0];
        // boundary_val is the value we get when we trace back
        // the GetAttr access chain until we hit the input of graph
        // or a node that is not prim::GetAttr
        auto path = getModuleAccessPath(instance, self);

        // path.size() == 0 means we're calling a method
        // on self, we don't need to dedup uses of self
        if (path.empty()) {
          continue;
        }
        value_to_path_map_[instance] = path;
        auto m = findChildModule(module_, path);
        // If we fail to insert the module to the unique_modules_ set,
        // which means there are uses of this module before this point,
        // we'll have to rewrite the use
        if (!unique_modules_.insert(m._ivalue()).second) {
          uses_to_rewrite_.push_back(instance);
          GRAPH_DEBUG("Found use to rewrite: ", instance->debugName());
        }
      }
    }
  }

  // Deduplicate module uses given the information we recorded before
  void dedupModuleUses() {
    for (Value* v : uses_to_rewrite_) {
      const auto& path = value_to_path_map_.at(v);
      const auto& m = findChildModule(module_, path);
      // add a clone of the child module to the parent of the duplicated module
      const auto& child_name = addChildModule(module_, m, path);
      TORCH_INTERNAL_ASSERT(v->node()->kind() == prim::GetAttr);
      // change the name in GetAttr call
      auto original_name = v->node()->s(attr::name);
      v->node()->s_(attr::name, child_name);
      GRAPH_UPDATE(
          "Module use dedup: changing use of original module ",
          original_name,
          " to ",
          child_name);
    }
  }

  std::string addChildModule(
      Module& module,
      const Module& child_module,
      const std::vector<std::string>& path) {
    TORCH_INTERNAL_ASSERT(
        !path.empty(), "path must have at least one element.");
    // Parent module of the leaf child module corresponding to
    // the path
    auto parent_of_leaf = findChildModule(
        module, std::vector<std::string>(path.begin(), path.end() - 1));

    // Original name of the child module
    const std::string& original_name = path[path.size() - 1];
    int uid = 0;
    std::string child_name = original_name + "_" + std::to_string(uid++);
    while (parent_of_leaf.hasattr(child_name)) {
      child_name = original_name + "_" + std::to_string(uid++);
    }
    parent_of_leaf.register_module(child_name, child_module.deepcopy());
    return child_name;
  }

  Module module_;
  // Map from value of module instance to the list of names of submodules
  // starting from the top level module, e.g. ["sub1", "sub2", "relu"]
  // Also this is a cache of calling `getModuleAccessPath` of the value
  std::unordered_map<Value*, std::vector<std::string>> value_to_path_map_;
  // Set of unique modules that are used in the graphs
  std::unordered_set<ModulePtr> unique_modules_;
  // Values that represent the module instance(the use of the module)
  // that we'll need to rewrite as a use of a cloned module
  // instance
  std::vector<Value*> uses_to_rewrite_;
};

} // namespace

void DedupModuleUses(Module& module) {
  ModuleUseDeduper d(module);
  d.dedup();
}

} // namespace torch::jit
