#include <gtest/gtest.h>

#include <c10/util/Optional.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/symbolic_shape_runtime_compute.h>
#include "ATen/core/interned_strings.h"
#include "c10/util/Exception.h"

namespace torch {
namespace jit {

TEST(ShapeAnalysisTest, Basic) {
  std::shared_ptr<Graph> subgraph = std::make_shared<Graph>();
  const auto graph_string = R"IR(
      graph(%x.1 : Tensor, %y.1 : Tensor):
        %11 : int = prim::Constant[value=0]()
        %3 : Tensor = aten::tanh(%x.1)
        %out1.1 : Tensor = aten::erf(%3)
        %out2.1 : Tensor = aten::relu(%y.1)
        %10 : Tensor[] = prim::ListConstruct(%out1.1, %out2.1)
        %25 : Tensor = aten::cat(%10, %11)
        %28 : Tensor = aten::hardswish(%25)
        return (%28))IR";
  torch::jit::parseIR(graph_string, subgraph.get());
  Graph g;
  auto x_inp = g.addInput("x_inp");
  auto y_inp = g.addInput("y_inp");
  auto x_type = TensorType::get()->withSizes({10, 5});
  auto y_type = TensorType::get()->withSizes({4, 5});
  x_inp->setType(x_type);
  y_inp->setType(y_type);
  subgraph->inputs().at(0)->setType(x_type);
  subgraph->inputs().at(1)->setType(y_type);
  x_inp->setType(TensorType::get()->withSizes({10, 5}));
  y_inp->setType(TensorType::get()->withSizes({4, 5}));
  auto output = g.insertNode(g.create(prim::TensorExprGroup))->output();
  output->node()->addInput(x_inp);
  output->node()->addInput(y_inp);
  output->node()->g_(attr::Subgraph, subgraph);
  auto maybe_map = GenerateGuard(output->node());
  TORCH_INTERNAL_ASSERT(maybe_map.has_value());
  auto map = *maybe_map;
  g.dump();
  for (const auto& pair: map) {
    std::cout << pair.first << " : ";
    pair.second->node()->dump();
    std::cout << "\n";
  }
  // TODO: add test for corrrectness of computed map
  // std::unordered_set<int64_t> added_values;
  // auto subgraph_x = subgraph->inputs().at(0);
  // for (auto val: *subgraph_x->type()->expect<TensorType>()->symbolic_sizes().sizes()) {
  //   if (val.value() == 1) {
  //     continue;
  //   }
  //   TORCH_INTERNAL_ASSERT(map->count(val.value()));
  //   if (added_values.count(val.value())) {
  //     continue;
  //   }
  // }

  // g.dump();
}


} // namespace jit
} // namespace torch
