#include <gtest/gtest.h>

#include <c10/util/Optional.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/runtime/script_profile.h>

namespace torch {
namespace jit {

TEST(ScriptProfileTest, Basic) {
  const std::string source_string = R"V0G0N(
    def foo(a, b):
      return a + b #
  )V0G0N";
  auto begin = source_string.find("return");
  auto end = source_string.find(" #");

  Graph g;
  const auto graph_string = R"IR(
    graph(%a : Tensor,
          %b : Tensor):
      %2 : int = prim::Constant[value=1]()
      %3 : Tensor = aten::add(%a, %b, %2)
      return (%3))IR";

  torch::jit::parseIR(graph_string, &g);
  auto source = std::make_shared<Source>(source_string, "", 0);
  auto node = *g.nodes().begin();
  node->setSourceRange(SourceRange{source, begin, end});

  ScriptProfile p;
  p.enable();
  {
    profiling::InstructionSpan g0(*node);
    profiling::InstructionSpan g1(*node);
    profiling::InstructionSpan g2(*node);
  }
  p.disable();

  auto stats = p.dumpStats();
  EXPECT_EQ(stats.size(), 1);
  auto it = stats.find(*source.get());
  EXPECT_NE(it, stats.end());
  auto& lines = it->second;
  EXPECT_EQ(lines.size(), 1);
  const auto& stat = lines.at(source->lineno_for_offset(begin));
  EXPECT_EQ(stat.count, 3);
}

TEST(ScriptProfileTest, CallingOrder) {
  ScriptProfile p;
  p.enable();
  EXPECT_THROW(p.dumpStats(), c10::Error);
  p.disable();
  auto dp = std::make_shared<profiling::Datapoint>(SourceRange{});
  EXPECT_THROW(p.addDatapoint(std::move(dp)), c10::Error);
}

} // namespace jit
} // namespace torch
