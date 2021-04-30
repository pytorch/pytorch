#include <gtest/gtest.h>

#include <c10/util/Optional.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/runtime/script_profile.h>

namespace torch {
namespace jit {

TEST(ScriptProfileTest, Basic) {
  Graph g;
  const auto graph_string = R"IR(
    graph(%a : Tensor,
          %b : Tensor):
      %2 : int = prim::Constant[value=1]()
      %3 : Tensor = aten::add(%a, %b, %2)
      return (%3))IR";
  torch::jit::parseIR(graph_string, &g);
  auto source = std::make_shared<Source>(graph_string, "", 0);
  auto node = *g.nodes().begin();
  node->setSourceRange(SourceRange{source, 0, 0});

  ScriptProfile p;
  p.enable();
  {
    auto g = profiling::InstructionSpan::tryMake(*node);
    EXPECT_NE(g, c10::nullopt);
  }
  p.disable();
  auto stats = p.dumpStats();
  EXPECT_EQ(stats.size(), 1);
  auto it = stats.find(*source.get());
  EXPECT_NE(it, stats.end());
  auto& lines = it->second;
  EXPECT_EQ(lines.size(), 1);
  const auto& stat = lines.at(0);
  EXPECT_EQ(stat.count, 1);
}
} // namespace jit
} // namespace torch
