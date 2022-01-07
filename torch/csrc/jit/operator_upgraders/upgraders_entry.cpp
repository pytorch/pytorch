#include <torch/csrc/jit/operator_upgraders/upgraders_entry.h>

#include <ATen/core/stack.h>
#include <c10/macros/Export.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/operator_upgraders/upgraders.h>
#include <torch/csrc/jit/serialization/export_bytecode.h>
#include <string>
#include <unordered_map>

namespace torch {
namespace jit {

static std::unordered_map<std::string, std::string> kUpgradersEntryMap(
    {{"div_Tensor_0_3", R"SCRIPT(
def div_Tensor_0_3(self: Tensor, other: Tensor) -> Tensor:
  if (self.is_floating_point() or other.is_floating_point()):
    return self.true_divide(other)
  return self.divide(other, rounding_mode='trunc')
)SCRIPT"},
     {"div_Scalar_0_3", R"SCRIPT(
def div_Scalar_0_3(self: Tensor, other: number) -> Tensor:
  if (self.is_floating_point() or isinstance(other, float)):
    return self.true_divide(other)
  return self.divide(other, rounding_mode='trunc')
)SCRIPT"},
     {"div_out_0_3", R"SCRIPT(
def div_out_0_3(self: Tensor, other: Tensor, *, out: Tensor) -> Tensor:
  if (self.is_floating_point() or other.is_floating_point() or out.is_floating_point()):
    return self.true_divide(other, out=out)
  return self.divide(other, rounding_mode='trunc', out=out)
)SCRIPT"},
     {"div__Tensor_0_3", R"SCRIPT(
def div__Tensor_0_3(self: Tensor, other: Tensor) -> Tensor:
  if (self.is_floating_point() or other.is_floating_point()):
    return self.true_divide_(other)
  return self.divide_(other, rounding_mode='trunc')
)SCRIPT"},
     {"div__Scalar_0_3", R"SCRIPT(
def div__Scalar_0_3(self: Tensor, other: number) -> Tensor:
  if (self.is_floating_point() or isinstance(other, float)):
    return self.true_divide_(other)
  return self.divide_(other, rounding_mode='trunc')
)SCRIPT"},
     {"full_0_4", R"SCRIPT(
def full_0_4(size:List[int], fill_value:number, *, dtype:Optional[int]=None,
             layout:Optional[int]=None, device:Optional[Device]=None,
             pin_memory:Optional[bool]=None) -> Tensor:
  if dtype is None:
    fill_value = float(fill_value)
  return torch.full(size, fill_value, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)
)SCRIPT"},
     {"full_out_0_4", R"SCRIPT(
def full_out_0_4(size:List[int], fill_value:number, *, out:Tensor) -> Tensor:
  return torch.full(size, fill_value, out=out)
)SCRIPT"}});

std::shared_ptr<Graph> create_upgrader_graph(
    const std::string& upgrader_name,
    const std::string& upgrader_body) {
  auto cu = std::make_shared<CompilationUnit>();
  cu->define(c10::nullopt, upgrader_body, nativeResolver(), nullptr);
  Function& jitFunc = cu->get_function(upgrader_name);
  GraphFunction& graphFunction = toGraphFunction(jitFunc);
  return graphFunction.graph();
}
using UpgraderMap = std::unordered_map<std::string, std::shared_ptr<Graph>>;
void populate_upgraders_graph_map() {
  if (!is_upgraders_map_populated()) {
    UpgraderMap populate_content;
    for (const auto& entry : kUpgradersEntryMap) {
      auto upgrader_graph = create_upgrader_graph(entry.first, entry.second);
      populate_content.insert(std::make_pair(entry.first, upgrader_graph));
    }
    populate_upgraders_map(std::forward<UpgraderMap>(populate_content));
  }
}

std::vector<ByteCodeEntry> generate_bytecode_list() {
  std::vector<ByteCodeEntry> upgraders_bytecode_list;
  upgraders_bytecode_list.reserve(kUpgradersEntryMap.size());
  for (const auto& entry : kUpgradersEntryMap) {
    auto upgrader_graph = create_upgrader_graph(entry.first, entry.second);
    CompilationOptions options;
    GraphFunction jitFunc(entry.first, upgrader_graph, nullptr);
    auto mobileFunc = convertJitFunctionToMobileFunction(jitFunc, options);
    auto codeTable = convertMobileFunctionToCodeTable(*mobileFunc, options);
    upgraders_bytecode_list.emplace_back(
        std::make_tuple(entry.first, codeTable));
  }
  return upgraders_bytecode_list;
}

} // namespace jit
} // namespace torch
