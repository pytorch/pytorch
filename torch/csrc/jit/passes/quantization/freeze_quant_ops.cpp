#include <stack>

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/quantization/freeze_quant_ops.h>
#include <torch/csrc/jit/runtime/graph_executor_impl.h>
#include <torch/csrc/jit/passes/freeze_module.h>

namespace torch {
namespace jit {

namespace {
class QuantizationAttributeFreezer {
 public:
  QuantizationAttributeFreezer(
      Module& module,
      const FreezingOpsFilterFn& is_freezable_op)
      : module_(module), is_freezable_op_(is_freezable_op) {
    auto method = module.get_method("forward");
    func_ = &method.function();
  }

  void run() {
    auto graph = func_->graph();
    // Inline the graph to make sure we don't have to traverse different
    // subgraphs when trying to get the weight attributes.
    Inline(*graph);
    std::stack<Block*> blocks_to_visit({graph->block()});
    while (!blocks_to_visit.empty()) {
      Block* block = blocks_to_visit.top();
      blocks_to_visit.pop();
      for (auto n : block->nodes()) {
        if (is_freezable_op_(n)) {
          // Find getattrs in the input of the quant op
          identifyAndConvertGetAttrs(module_, n, graph);
        }
        for (Block* sub_block : n->blocks()) {
          blocks_to_visit.push(sub_block);
        }
      }
    }
    runOptimization(graph, false);
    removeUnusedAttrs();
    std::cout << "** FINAL Graph ** " << graph->toString();
  }

 private:
  /*
   * Find the GetAttr nodes in the graph that correspond to
   * either the quantize prepack/run op or the
   * quantize_per_tensor/quantize_per_channel operator
   */
  void identifyAndConvertGetAttrs(
      Module& module,
      Node* n,
      std::shared_ptr<Graph>& graph) {
    const auto& inputs = n->inputs().vec();
    for (auto v : inputs) {
      auto quant_inp_node = v->node();
      WithInsertPoint ins(quant_inp_node);
      if (quant_inp_node->kind() == prim::GetAttr) {
        convertAttrToConst(module, quant_inp_node, graph);
      } else if (
          quant_inp_node->kind() == Symbol::aten("quantize_per_tensor") ||
          quant_inp_node->kind() == Symbol::aten("quantize_per_channel")) {
        identifyAndConvertGetAttrs(module, quant_inp_node, graph);
      }
    }
  }

  void convertAttrToConst(
      Module& module,
      Node* attr_node,
      std::shared_ptr<Graph>& graph) {
    // Map from module to list of constant attribute values in the graph
    std::unordered_map<ModulePtr, std::unordered_map<std::string, Value*>>
        moduleAttrs;
    auto name = attr_node->s(attr::name);

    auto attrModule = module;
    std::deque<std::string> module_names;
    // Get the module corresponding to the attr
    if (!updateAttrModule(
            attr_node->inputs()[0], name, attrModule, module_names, graph)) {
      TORCH_WARN_ONCE(
          "Quantization param attribute ",
          name,
          " is not a constant in the graph. Cannot be frozen.");
    }
    TORCH_INTERNAL_ASSERT(attrModule.hasattr(name));
    auto attrs = moduleAttrs.find(attrModule._ivalue());

    Value* const_param = nullptr;
    if (attrs != moduleAttrs.end()) {
      auto attr_value = attrs->second.find(name);
      if (attr_value != attrs->second.end()) {
        const_param = attr_value->second;
      }
    }
    // If constant node hasn't yet been created for this GetAttr then
    // create one and insert into the graph.
    if (!const_param) {
      auto attr = attrModule.attr(name);
      attr = overrideGradient(attr);
      if (auto inserted_val = tryInsertConstant(*graph, attr)) {
        const_param = *inserted_val;
      } else {
        TORCH_WARN_ONCE("Attribute ", name, "is not materializable");
        return;
      }
      std::string const_name("self.");
      for (auto& name : module_names) {
        const_name += name + '.';
      }
      const_name += name;
      const_param->setDebugName(const_name);
      moduleAttrs[attrModule._ivalue()][name] = const_param;

      attrsToRemove_[attrModule._ivalue()].insert(name);
    }
    attr_node->outputs().at(0)->replaceAllUsesWith(const_param);
    attr_node->removeAllInputs();
  }

  /*
   * Delete attributes from the graph that we already converted
   * to constant nodes.
   */
  void removeUnusedAttrs() {
    std::vector<std::string> attrNames;
    for (auto& it : attrsToRemove_) {
      auto& mptr = it.first;
      auto type = mptr->type();

      auto attr_names = it.second;
      for (auto& name : attr_names) {
        std::cout << "removing attr " << name << " slot "
                  << type->getAttributeSlot(name) << std::endl;
        TORCH_CHECK(
            type->hasAttribute(name),
            "Expected ClassType to have attribute ",
            name);
        mptr->unsafeRemoveAttr(name);
        type->unsafeRemoveAttribute(name);
      }
    }
  }

  /*
   * Traverse the module hierarchy to find the module corresponding to the
   * GetAttr node in the graph.
   */
  bool updateAttrModule(
      Value* input,
      std::string& name,
      Module& attrModule,
      std::deque<std::string>& names,
      std::shared_ptr<Graph>& graph) {
    Node* node = input->node();
    while (!(node->outputs()[0]->type() == graph->inputs()[0]->type())) {
      if (node->kind() == prim::GetAttr) {
        names.push_front(node->s(attr::name));
        node = node->inputs()[0]->node();
      } else {
        return false;
      }
    }
    for (auto& moduleName : names) {
      attrModule = attrModule.attr(moduleName).toModule();
    }
    return true;
  }

  Module& module_;
  Function* func_;

  const FreezingOpsFilterFn& is_freezable_op_;

  // Map from module ptr to list of attribute names to remove from module.
  std::unordered_map<ModulePtr, std::set<std::string>> attrsToRemove_;

}; // class QuantizationAttributeFreezer

} // namespace

Module FreezeAndFoldQuantOps(
    script::Module& input_module,
    const FreezingOpsFilterFn& is_freezable_op) {
  auto module = input_module.clone(true);
  TORCH_CHECK(
      !module.hasattr("training") || !module.is_training(),
      "Freezing quantization params in training mode is not yet supported");

  QuantizationAttributeFreezer quantFreezer(module, is_freezable_op);
  quantFreezer.run();
  return module;
}

} // namespace jit
} // namespace torch
