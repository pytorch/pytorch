#include <stack>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/hoist_conv_packed_params.h>

namespace torch {
namespace jit {

// Hoists packed params from a conv module to the parent module.
// The benefit is that after this hoisting, the conv module
// no longer holds anything and can be deleted, reducing model
// size.
//
// Before (easy case):
//
// %1 = prim::GetAttr[name="conv1"][%self]
// %2 = prim::GetAttr[name="_packed_params][%1]
//
// After (easy case):
//
// %2 = prim::GetAttr[name="{prefix}.conv1._packed_params"][%self]
//
// Before (generic case):
//
// %1 = prim::GetAttr[name="name1"][%self]
// %2 = prim::GetAttr[name="name2"][%1]
// ...
// %n = prim::GetAttr[name="_packed_params][%n-1]
//
// After (generic case):
//
// %2 = prim::GetAttr[name="{prefix}.name1{...}.name(n-1)._packed_params"][%self]
//
void hoistConvPackedParams(Module& rootModule, Node* getConvPackedParamsNode,
    const std::string& prefix) {

  auto method = rootModule.get_method("forward");
  auto graph = method.graph();

  // create a vector of the getAttr nodes
  //   idx 0 is the node from root to child 1
  //   idx n-1 is the node from child n-1 to conv
  std::vector<Node*> getAttrNodes;
  Node* curNode = getConvPackedParamsNode->inputs()[0]->node();
  while (!(curNode->outputs()[0]->type() == graph->inputs()[0]->type())) {
    TORCH_CHECK(
      curNode->kind() == prim::GetAttr,
      "Attempted to add a non-prim::GetAttr node to a chain of prim::getAttr nodes.");
    getAttrNodes.insert(getAttrNodes.begin(), curNode);
    curNode = curNode->inputs()[0]->node();
  }
  TORCH_CHECK(getAttrNodes.size() > 0, "Did not find a chain of prim::getAttr nodes");

  // create a name suffix
  std::string suffix = "";
  for (const auto& n : getAttrNodes) {
    suffix += n->s(attr::name) + ".";
  }

  // traverse the chain to get packed params value
  std::string curName = getAttrNodes[0]->s(attr::name);
  Module curConvModule = rootModule.attr(curName).toModule();
  for (int idx = 1; idx < getAttrNodes.size(); idx++) {
    curName = getAttrNodes[idx]->s(attr::name);
    curConvModule = curConvModule.attr(curName).toModule();
  }
  c10::IValue packedParams = curConvModule.attr("_packed_params");

  // copy the packed params

  std::string newName = prefix + "." + suffix + "_packed_params";

  // make sure the attribute does not already exist
  TORCH_CHECK(
    !(rootModule.type()->findAttributeSlot(newName)),
    "Attribute name ",
    newName,
    " already exists in module of type ",
    rootModule.type()->name()->qualifiedName());

  rootModule.register_attribute(newName, packedParams.type(), packedParams);

  // change target module to rootModule
  Value* rootModuleAsValue = getAttrNodes[0]->inputs()[0];
  getConvPackedParamsNode->replaceInput(0, rootModuleAsValue);

  // change attribute name to new name
  getConvPackedParamsNode->s_(Symbol::attr("name"), newName);
}

void HoistConvPackedParams(script::Module& m) {
  auto method = m.get_method("forward");
  auto graph = method.graph();

  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  std::string attr_name_base = "_jit_pass_hoist_conv_packed_params";

  while (!blocks_to_visit.empty()) {

    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();

    for (Node* n : b->nodes()) {

      // TODO before land: also check for n->inputs()[0]->node()->kind() == {ConvNd}
      //   need to figure out how
      bool isGetConvPackedParamsNode = n->kind() == prim::GetAttr &&
          n->s(attr::name).find("_packed_params") != std::string::npos;
      if (isGetConvPackedParamsNode) {
        GRAPH_UPDATE("Hoisting ", *n, " to root module.");
        hoistConvPackedParams(m, n, attr_name_base);
      }

      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }

    } // for

  } // while

}

} // namespace jit
} // namespace torch
