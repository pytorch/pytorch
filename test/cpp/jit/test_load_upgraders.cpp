#include <caffe2/serialize/versions.h>
#include <gtest/gtest.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/operator_upgraders/upgraders.h>
#include <torch/csrc/jit/operator_upgraders/version_map.h>
#include <torch/csrc/jit/serialization/import.h>

#include <test/cpp/jit/test_utils.h>

namespace torch {
namespace jit {

// Basic tests to check if C++ torch::jit::load
// can load the upgraders fine
// TODO (tugsuu) add more tests
TEST(UpgraderLoad, CanPopulateUpgradersGraph) {
  Module m("m");
  m.define(R"(
    def forward(self, x: Tensor):
      b = 5
      return torch.div(x, b)
  )");
  std::stringstream ms;
  m.save(ms);
  auto loaded_m = torch::jit::load(ms);
  auto version_map = get_operator_version_map();
  auto upgraders = dump_upgraders_map();

  for (const auto& entry : version_map) {
    auto list_of_upgraders_for_op = entry.second;
    for (const auto& upgrader_entry : list_of_upgraders_for_op) {
      EXPECT_TRUE(
          upgraders.find(upgrader_entry.upgrader_name) != upgraders.end());
    }
  }

  auto test_graph = loaded_m.get_method("forward").graph();
  // should have saved with version 4, so it is still up to date
  testing::FileCheck().check_count("aten::div", 1, true)->run(*test_graph);
}

} // namespace jit
} // namespace torch
