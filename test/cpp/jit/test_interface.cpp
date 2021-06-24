#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>

#include <ATen/core/qualified_name.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/serialization/import_source.h>
#include <torch/torch.h>

namespace torch {
namespace jit {

static const std::vector<std::string> subMethodSrcs = {R"JIT(
def one(self, x: Tensor, y: Tensor) -> Tensor:
    return x + y + 1

def forward(self, x: Tensor) -> Tensor:
    return x
)JIT"};
static const auto parentForward = R"JIT(
def forward(self, x: Tensor) -> Tensor:
    return self.subMod.forward(x)
)JIT";

static const auto moduleInterfaceSrc = R"JIT(
class OneForward(ModuleInterface):
    def one(self, x: Tensor, y: Tensor) -> Tensor:
        pass
    def forward(self, x: Tensor) -> Tensor:
        pass
)JIT";

static void import_libs(
    std::shared_ptr<CompilationUnit> cu,
    const std::string& class_name,
    const std::shared_ptr<Source>& src,
    const std::vector<at::IValue>& tensor_table) {
  SourceImporter si(
      cu,
      &tensor_table,
      [&](const std::string& name) -> std::shared_ptr<Source> { return src; },
      /*version=*/2);
  si.loadType(QualifiedName(class_name));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(InterfaceTest, ModuleInterfaceSerialization) {
  auto cu = std::make_shared<CompilationUnit>();
  Module parentMod("parentMod", cu);
  Module subMod("subMod", cu);

  std::vector<at::IValue> constantTable;
  import_libs(
      cu,
      "__torch__.OneForward",
      std::make_shared<Source>(moduleInterfaceSrc),
      constantTable);

  for (const std::string& method : subMethodSrcs) {
    subMod.define(method, nativeResolver());
  }
  parentMod.register_attribute(
      "subMod",
      cu->get_interface("__torch__.OneForward"),
      subMod._ivalue(),
      // NOLINTNEXTLINE(bugprone-argument-comment)
      /*is_parameter=*/false);
  parentMod.define(parentForward, nativeResolver());
  ASSERT_TRUE(parentMod.hasattr("subMod"));
  std::stringstream ss;
  parentMod.save(ss);
  Module reloaded_mod = jit::load(ss);
  ASSERT_TRUE(reloaded_mod.hasattr("subMod"));
  InterfaceTypePtr submodType =
      reloaded_mod.type()->getAttribute("subMod")->cast<InterfaceType>();
  ASSERT_TRUE(submodType->is_module());
}

} // namespace jit
} // namespace torch
