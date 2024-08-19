#include <stack>

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/hoist_conv_packed_params.h>
#include <torch/csrc/jit/passes/quantization/helper.h>

namespace torch::jit {

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
// %n =
// prim::GetAttr[name="{prefix}.name1{...}.name(n-1)._packed_params"][%self]
//
static void hoistConvPackedParams(
    Module& rootModule,
    Node* getConvPackedParamsNode,
    const std::string& prefix,
    int& nameUniqueCounter) {
  auto method = rootModule.get_method("forward");
  auto graph = method.graph();
  Value* rootModuleAsValue = graph->inputs()[0];

  // get a path from root module to conv module
  Value* convModuleAsValue = getConvPackedParamsNode->inputs()[0];
  std::vector<std::string> rootToConvPath =
      getModuleAccessPath(convModuleAsValue, rootModuleAsValue);

  // get a module object representing the conv
  Module convModule = findChildModule(rootModule, rootToConvPath);

  // get the packed params value
  c10::IValue packedParams = convModule.attr("_packed_params");

  // create the new name

  std::string suffix = "";
  for (const auto& attrName : rootToConvPath) {
    suffix += attrName + ".";
  }
  std::string newNameBase = prefix + "." + suffix + "_packed_params";
  nameUniqueCounter++;
  std::string newName = newNameBase + "." + std::to_string(nameUniqueCounter);
  while (rootModule.hasattr(newName)) {
    nameUniqueCounter++;
    newName = newNameBase + "." + std::to_string(nameUniqueCounter);
  }

  // copy the packed params
  rootModule.register_attribute(newName, packedParams.type(), packedParams);

  // change target module to rootModule
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
  // counter to ensure new attribute names are unique
  int nameUniqueCounter = 0;

  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();

    for (Node* n : b->nodes()) {
      // make sure this node is fetching {foo}.{_packed_params}
      bool isGetPackedParamsNode =
          n->kind() == prim::GetAttr && n->s(attr::name) == "_packed_params";
      if (isGetPackedParamsNode) {
        // make sure the foo in {foo}.{_packed_params} is a quantized conv
        std::optional<std::string> moduleName = getModuleName(n->inputs()[0]);
        bool moduleNameIsQuantizedConv = moduleName.has_value() &&
            (moduleName.value() ==
                 "__torch__.torch.ao.nn.quantized.modules.conv.Conv1d" ||
             moduleName.value() ==
                 "__torch__.torch.ao.nn.quantized.modules.conv.Conv2d" ||
             moduleName.value() ==
                 "__torch__.torch.ao.nn.quantized.modules.conv.Conv3d" ||
             moduleName.value() ==
                 "__torch__.torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU1d" ||
             moduleName.value() ==
                 "__torch__.torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d" ||
             moduleName.value() ==
                 "__torch__.torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU3d" ||
             // BC Stuff
             moduleName.value() ==
                 "__torch__.torch.nn.quantized.modules.conv.Conv1d" ||
             moduleName.value() ==
                 "__torch__.torch.nn.quantized.modules.conv.Conv2d" ||
             moduleName.value() ==
                 "__torch__.torch.nn.quantized.modules.conv.Conv3d");

        if (moduleNameIsQuantizedConv) {
          GRAPH_UPDATE("Hoisting ", *n, " to root module.");
          hoistConvPackedParams(m, n, attr_name_base, nameUniqueCounter);
        }
      }

      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }

    } // for

  } // while
}

} // namespace torch::jit
