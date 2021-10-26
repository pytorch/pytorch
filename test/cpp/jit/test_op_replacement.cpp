#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/passes/op_replacement.h>

namespace torch {
namespace jit {

TEST(OpReplacementTest, ReplaceDiv) {
    const auto graph_string = R"IR(
        graph(%0 : Tensor,
              %1 : Tensor):
            %2 : Tensor = aten::add(%0, %1)
            %3 : Tensor  = aten::div(%2, %1)
            return (%3))IR";
    auto g = std::make_shared<Graph>();
    torch::jit::parseIR(graph_string, g.get());
    ReplaceOpsWithUpgraders(g);
    // TODO prints correct graph but will need to add some better way of testing this
    g->print(std::cout);
}

} // namespace jit
} // namespace torch
