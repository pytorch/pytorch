#include <gtest/gtest.h>

#include <ATen/ops/tensor.h>
#include <torch/nativert/executor/ExecutionFrame.h>

namespace torch::nativert {

TEST(ExecutionFrameTest, CreateFrame) {
  auto graph = stringToGraph(R"(
    graph(%x, %y):
  %a = foo(a=%x, b=%y)
  %b = foo1(a=%x, b=%y)
  %c = foo2(c=%a, d=%b)
  return(%c)
  )");

  auto frame = ExecutionFrame(*graph);

  for (auto* v : graph->values()) {
    frame.setIValue(v->id(), c10::IValue(at::tensor({v->id()}, at::kInt)));
    auto& frame_v = frame.getIValue(v->id());
    EXPECT_EQ(frame_v.tagKind(), "Tensor");
  }

  auto outputs = frame.tryMoveUserOutputs();

  EXPECT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs[0].tagKind(), "Tensor");
  EXPECT_EQ(outputs[0].toTensor().item().toInt(), graph->getValue("c")->id());
}

TEST(ExecutionFrameTest, TestSetBorrowedValue) {
  auto graph = stringToGraph(R"(
    graph(%x, %y):
  %a = foo(a=%x, b=%y)
  %b = foo1(a=%x, b=%y)
  %c = foo2(c=%a, d=%b)
  return(%c)
  )");

  auto x = c10::IValue(at::tensor({1}, at::kInt));
  auto y = c10::IValue(at::tensor({2}, at::kInt));

  {
    auto frame = ExecutionFrame(*graph);

    frame.setBorrowedIValue(
        graph->getValue("x")->id(),
        c10::MaybeOwnedTraits<c10::IValue>::createBorrow(x));
    frame.setBorrowedIValue(
        graph->getValue("y")->id(),
        c10::MaybeOwnedTraits<c10::IValue>::createBorrow(y));

    [[maybe_unused]] auto& w = frame.getIValue(graph->getValue("x")->id());
    [[maybe_unused]] auto& z = frame.getIValue(graph->getValue("y")->id());

    EXPECT_EQ(x.use_count(), 1);
    EXPECT_EQ(y.use_count(), 1);

    EXPECT_TRUE(c10::MaybeOwnedTraits<c10::IValue>{}.debugBorrowIsValid(
        frame.getIValue(graph->getValue("x")->id())));
    EXPECT_TRUE(c10::MaybeOwnedTraits<c10::IValue>{}.debugBorrowIsValid(
        frame.getIValue(graph->getValue("y")->id())));
  }

  EXPECT_EQ(x.use_count(), 1);
  EXPECT_EQ(y.use_count(), 1);
}

TEST(ExecutionFrameTest, TestPersistentValue) {
  auto graph = stringToGraph(R"(
    graph(%x, %y, %my_weight):
  %a = foo(a=%x, b=%y)
  %b = foo1(a=%x, b=%y)
  %c = foo2(c=%a, d=%b)
  return(%c)
  )");

  Weights weights(graph.get());
  weights.setValue("my_weight", at::tensor({1}, at::kInt));

  auto new_sig = graph->signature();
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  const_cast<std::vector<std::pair<std::string, std::string>>&>(
      new_sig.inputsToWeights())
      .emplace_back("my_weight", "my_weight");
  graph->setSignature(new_sig);

  auto frame = ExecutionFrame(*graph, weights);

  EXPECT_EQ(frame.weightVersion(), 0);
  auto wid = graph->getValue("my_weight")->id();

  EXPECT_NO_THROW(frame.getTensor(wid));
  // can't release persistent value
  frame.releaseValueIfNeeded(wid);
  EXPECT_FALSE(frame.getIValue(wid).isNone());
}

} // namespace torch::nativert
