#include <c10/core/Device.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <torch/nativert/graph/Graph.h>

using namespace ::testing;

namespace torch::nativert {
TEST(GraphTest, Basic) {
  static constexpr std::string_view source =
      R"(graph(%foo, %bar, %baz):
%o1, %o2 = aten.foo(self=%foo, target=%bar, alpha=0.1)
return(%o2, %baz)
)";
  auto graph = stringToGraph(source);
  EXPECT_EQ(graph->inputs().size(), 3);
  EXPECT_EQ(graph->inputs()[0]->name(), "foo");
  EXPECT_EQ(graph->inputs()[1]->name(), "bar");
  EXPECT_EQ(graph->inputs()[2]->name(), "baz");

  const auto& nodes = graph->nodes();
  EXPECT_EQ(nodes.size(), 3);
  // First node is the input node
  auto it = nodes.begin();
  {
    const auto& node = *it;
    EXPECT_EQ(node.target(), "prim.Input");
    EXPECT_EQ(node.inputs().size(), 0);
    EXPECT_EQ(node.outputs().size(), 3);
    EXPECT_EQ(node.outputs()[0]->name(), "foo");
    EXPECT_EQ(node.outputs()[1]->name(), "bar");
    EXPECT_EQ(node.outputs()[2]->name(), "baz");
  }
  {
    std::advance(it, 1);
    const auto& node = *it;
    EXPECT_EQ(node.target(), "aten.foo");
    EXPECT_EQ(node.inputs().size(), 2);
    EXPECT_EQ(node.inputs()[0].name, "self");
    EXPECT_EQ(node.inputs()[1].name, "target");

    EXPECT_EQ(node.attributes().size(), 1);
    EXPECT_EQ(node.attributes()[0].name, "alpha");
  }
  {
    std::advance(it, 1);
    const auto& node = *it;
    EXPECT_EQ(node.target(), "prim.Output");
    EXPECT_EQ(node.inputs().size(), 2);
    EXPECT_EQ(node.inputs()[0].name, "o2");
    EXPECT_EQ(node.inputs()[1].name, "baz");
  }
  EXPECT_EQ(graph->outputs().size(), 2);
  EXPECT_EQ(graph->outputs()[0]->name(), "o2");
  EXPECT_EQ(graph->outputs()[1]->name(), "baz");

  const auto& values = graph->values();
  EXPECT_EQ(values.size(), 5);
  std::vector<std::string> valueNames;
  valueNames.reserve(values.size());
  for (const auto& v : values) {
    valueNames.emplace_back(v->name());
  }
  std::sort(valueNames.begin(), valueNames.end());

  EXPECT_THAT(
      valueNames,
      ContainerEq(std::vector<std::string>({"bar", "baz", "foo", "o1", "o2"})));
}

TEST(GraphTest, ValueProducer) {
  static constexpr std::string_view source =
      R"(graph(%foo, %bar, %baz):
%o1, %o2 = aten.foo(self=%foo, target=%bar, alpha=0.1)
return(%o2, %baz)
)";
  auto graph = stringToGraph(source);
  auto foo = graph->getValue("foo");
  EXPECT_EQ(foo->producer()->target(), "prim.Input");
  auto o1 = graph->getValue("o1");
  EXPECT_EQ(o1->producer()->target(), "aten.foo");
}

TEST(GraphTest, InsertBeforeAfter) {
  static constexpr std::string_view source =
      R"(graph(%foo, %bar, %baz):
%o1, %o2 = aten.foo(self=%foo, target=%bar, alpha=0.1)
return(%o2, %baz)
)";
  auto graph = stringToGraph(source);
  auto it = graph->nodes().begin();
  ++it;
  auto& node = *it;
  EXPECT_EQ(node.target(), "aten.foo");
  auto before = graph->createNode("before", {});
  auto after = graph->createNode("after", {});
  auto atEnd = graph->createNode("atEnd", {});

  graph->insertBefore(before, &node);
  graph->insertAfter(after, &node);
  graph->insert(atEnd);

  static constexpr std::string_view expected =
      R"(graph(%foo, %bar, %baz):
 = before()
%o1, %o2 = aten.foo(self=%foo, target=%bar, alpha=0.1)
 = after()
 = atEnd()
return(%o2, %baz)
)";
  EXPECT_EQ(graphToString(*graph), expected);
}

TEST(GraphTest, ValueUses) {
  static constexpr std::string_view source =
      R"(graph(%foo, %bar, %baz):
%o1, %o2 = aten.foo(self=%foo, target=%bar, alpha=0.1)
return(%o2, %baz)
)";
  auto graph = stringToGraph(source);
  auto o2 = graph->getValue("o2");
  EXPECT_EQ(o2->users().size(), 1);
  EXPECT_EQ(o2->users()[0]->target(), "prim.Output");
}

TEST(GraphTest, ApplyDevicePlacement) {
  auto graph = Graph::createGraph();
  auto node1 = graph->insertNode("node1");
  auto node2 = graph->insertNode("node2");

  node1->addAttribute({"a", c10::Device(c10::DeviceType::CPU)});
  node1->addAttribute({"b", c10::Device(c10::DeviceType::CUDA, 0)});
  node1->addAttribute({"c", c10::Device(c10::DeviceType::CUDA, 1)});

  node2->addAttribute({"d", c10::Device(c10::DeviceType::CUDA, 0)});

  graph->applyDevicePlacement(
      Placement(std::unordered_map<c10::Device, c10::Device>{
          {c10::Device(c10::DeviceType::CUDA, 0),
           c10::Device(c10::DeviceType::CUDA, 1)}}));

  EXPECT_EQ(
      std::get<c10::Device>(node1->getAttribute("a").value),
      c10::Device(c10::DeviceType::CPU));
  EXPECT_EQ(
      std::get<c10::Device>(node1->getAttribute("b").value),
      c10::Device(c10::DeviceType::CUDA, 1));
  EXPECT_EQ(
      std::get<c10::Device>(node1->getAttribute("c").value),
      c10::Device(c10::DeviceType::CUDA, 1));
  EXPECT_EQ(
      std::get<c10::Device>(node2->getAttribute("d").value),
      c10::Device(c10::DeviceType::CUDA, 1));
}

TEST(GraphTest, ReplaceAllUses) {
  static constexpr std::string_view source =
      R"(graph(%foo, %bar, %baz):
%o1, %o2 = aten.foo(self=%foo, target=%bar, alpha=0.1)
return(%o2, %baz)
)";
  auto graph = stringToGraph(source);
  auto o2 = graph->getValue("o2");
  auto bar = graph->getValue("bar");
  auto foo = graph->getValue("foo");

  EXPECT_EQ(o2->users().size(), 1);
  EXPECT_EQ(bar->users().size(), 1);
  EXPECT_EQ(foo->users().size(), 1);

  graph->replaceAllUses(o2, bar);
  EXPECT_EQ(o2->users().size(), 0);
  EXPECT_EQ(bar->users().size(), 2);

  graph->replaceAllUses(bar, foo);
  EXPECT_EQ(bar->users().size(), 0);
  EXPECT_EQ(foo->users().size(), 2);
  static constexpr std::string_view expected =
      R"(graph(%foo, %bar, %baz):
%o1, %o2 = aten.foo(self=%foo, target=%foo, alpha=0.1)
return(%foo, %baz)
)";
  EXPECT_EQ(graphToString(*graph), expected);
}

TEST(GraphTest, GetUniqueValueName) {
  static constexpr std::string_view source =
      R"(graph(%foo, %bar, %baz):
%o1, %o2 = aten.foo(self=%foo, target=%bar, alpha=0.1)
return(%o2, %bar)
)";
  auto graph = stringToGraph(source);
  auto o2 = graph->getValue("o2");
  auto fooNode = o2->producer();
  auto v0 = graph->getUniqueValueName();
  graph->addValue(v0, Type::Kind::None, fooNode);
  auto v1 = graph->getUniqueValueName();
  graph->addValue(v1, Type::Kind::None, fooNode);
  auto v2 = graph->getUniqueValueName();
  EXPECT_EQ(v0, "v0");
  EXPECT_EQ(v1, "v1");
  EXPECT_EQ(v2, "v2");
}

TEST(GraphTest, ReplaceAllUsesMultiUse) {
  static constexpr std::string_view source =
      R"(graph(%foo, %bar):
%o1 = aten.foo(a=%foo, b=%foo, c=%bar)
return(%o1)
)";
  auto graph = stringToGraph(source);
  auto foo = graph->getValue("foo");
  auto bar = graph->getValue("bar");
  graph->replaceAllUses(foo, bar);

  static constexpr std::string_view expected =
      R"(graph(%foo, %bar):
%o1 = aten.foo(a=%bar, b=%bar, c=%bar)
return(%o1)
)";
  EXPECT_EQ(graphToString(*graph), expected);
}

TEST(GraphTest, ReplaceAllUsesAfter) {
  static constexpr std::string_view source =
      R"(graph(%foo):
%o1 = aten.foo1(a=%foo)
%o2 = aten.foo2(a=%o1, b=%foo)
%o3 = aten.foo3(a=%o2, b=%o2, c=%foo)
return(%foo, %o1, %o2, %o3)
)";
  auto graph = stringToGraph(source);
  auto foo = graph->getValue("foo");
  auto o1 = graph->getValue("o1");
  auto foo3Node = graph->getValue("o3")->producer();
  graph->replaceAllUsesAfterNode(foo, o1, foo3Node);

  static constexpr std::string_view expected =
      R"(graph(%foo):
%o1 = aten.foo1(a=%foo)
%o2 = aten.foo2(a=%o1, b=%foo)
%o3 = aten.foo3(a=%o2, b=%o2, c=%foo)
return(%o1, %o1, %o2, %o3)
)";
  EXPECT_EQ(graphToString(*graph), expected);
  EXPECT_EQ(foo->users().size(), 3);
  EXPECT_EQ(o1->users().size(), 2);
}

TEST(GraphTest, InsertingAfter) {
  static constexpr std::string_view source =
      R"(graph(%foo, %bar):
%o1 = aten.first(a=%foo)
%o2 = aten.foo(c=%bar)
return(%o1, %o2)
)";
  auto graph = stringToGraph(source);
  auto origNode = graph->getValue("o1")->producer();
  {
    InsertingAfter guard(origNode);
    graph->insertNode("one");
    graph->insertNode("two");
    graph->insertNode("three");
  }
  graph->insertNode("four");
  static constexpr std::string_view expected =
      R"(graph(%foo, %bar):
%o1 = aten.first(a=%foo)
 = one()
 = two()
 = three()
%o2 = aten.foo(c=%bar)
 = four()
return(%o1, %o2)
)";
  EXPECT_EQ(graphToString(*graph), expected);
}

TEST(NodeTest, GetInputAndAttribute) {
  auto graph = Graph::createGraph();
  auto input1 = graph->addInput("input1", Type::Kind::Tensor);
  auto input2 = graph->addInput("input2", Type::Kind::Tensor);
  auto input3 = graph->addInput("input3", Type::Kind::Tensor);
  auto node = graph->createNode("foo.bar");

  node->addInput({"out_of_order", input1});
  node->addInput({"arg1", input2});
  node->addInput({"arg2", input3});

  node->addAttribute({"b", static_cast<int64_t>(0)});
  node->addAttribute({"a", static_cast<int64_t>(2)});
  node->addAttribute({"c", static_cast<int64_t>(1)});
  {
    const auto& input = node->getInput("out_of_order");
    EXPECT_EQ(input.name, "out_of_order");
    EXPECT_EQ(input.value, input1);
  }
  {
    const auto& input = node->getInput("arg1");
    EXPECT_EQ(input.name, "arg1");
    EXPECT_EQ(input.value, input2);
  }
  {
    const auto& input = node->getInput("arg2");
    EXPECT_EQ(input.name, "arg2");
    EXPECT_EQ(input.value, input3);
  }
  {
    const auto& attr = node->getAttribute("a");
    EXPECT_EQ(attr.name, "a");
    EXPECT_EQ(attr.value, Constant(static_cast<int64_t>(2)));
  }
  {
    const auto& attr = node->getAttribute("b");
    EXPECT_EQ(attr.name, "b");
    EXPECT_EQ(attr.value, Constant(static_cast<int64_t>(0)));
  }
  {
    const auto& attr = node->getAttribute("c");
    EXPECT_EQ(attr.name, "c");
    EXPECT_EQ(attr.value, Constant(static_cast<int64_t>(1)));
  }

  EXPECT_EQ(node->tryGetInput("doesnotexist"), nullptr);
  EXPECT_EQ(node->tryGetAttribute("doesnotexist"), nullptr);
}

TEST(NodeTest, NextPrev) {
  static constexpr std::string_view source =
      R"(graph(%foo):
%o1 = aten.foo1(a=%foo)
%o2 = aten.foo2(a=%o1, b=%foo)
%o3 = aten.foo3(a=%o2, b=%o2, c=%foo)
return(%foo, %o1, %o2, %o3)
)";
  auto graph = stringToGraph(source);
  auto foo1 = graph->getValue("o1")->producer();
  auto foo2 = graph->getValue("o2")->producer();
  auto foo3 = graph->getValue("o3")->producer();
  EXPECT_EQ(foo1->next(), foo2);
  EXPECT_EQ(foo2->next(), foo3);
  EXPECT_EQ(foo3->prev(), foo2);
  EXPECT_EQ(foo3->next(), graph->outputNode());
  EXPECT_EQ(foo2->prev(), foo1);
  EXPECT_EQ(foo1->prev(), graph->inputNode());
  EXPECT_EQ(graph->inputNode()->prev(), nullptr);
  EXPECT_EQ(graph->outputNode()->next(), nullptr);
}

TEST(GraphTest, IsBefore) {
  auto source = R"IR(
    graph(%foo):
      %o1 = aten.foo1(a=%foo)
      %o2 = aten.foo2(a=%o1)
      %o3 = aten.foo3(a=%o2)
      return (%o3)
  )IR";

  auto graph = stringToGraph(source);
  ASSERT_NE(graph, nullptr);

  auto* o1 = graph->tryGetValue("o1");
  auto* o2 = graph->tryGetValue("o2");
  auto* o3 = graph->tryGetValue("o3");

  auto* foo1 = o1->producer();
  auto* foo2 = o2->producer();
  auto* foo3 = o3->producer();

  EXPECT_TRUE(foo1->isBefore(foo2)) << "foo1 should appear before foo2";
  EXPECT_TRUE(foo2->isBefore(foo3)) << "foo2 should appear before foo3";
  EXPECT_TRUE(foo1->isBefore(foo3)) << "foo1 should appear before foo3";

  EXPECT_FALSE(foo2->isBefore(foo1)) << "foo2 should not appear before foo1";
  EXPECT_FALSE(foo3->isBefore(foo2)) << "foo3 should not appear before foo2";
}

TEST(GraphTest, RemoveNodeWithUsers) {
  // Check we shouldn't be able to remove a node that still has users
  auto source = R"IR(
    graph(%foo):
        %o1 = aten.foo1(a=%foo)
        %o2 = aten.foo2(a=%o1, b=%foo)
        %o3 = aten.foo3(a=%o2, b=%o2, c=%foo)
        return (%foo, %o1, %o3)
  )IR";

  auto graph = stringToGraph(source);
  ASSERT_NE(graph, nullptr);

  auto* o2 = graph->tryGetValue("o2");
  auto* foo2 = o2->producer();

  EXPECT_THROW(graph->removeNode(foo2), c10::Error);
}

TEST(GraphTest, RemoveNodeUnused) {
  // Check node removal works as expected
  auto source = R"IR(
    graph(%foo):
      %o1 = aten.foo1(a=%foo)
      %o2 = aten.foo2(a=%o1, b=%foo)
      %unused = aten.fooUnused(a=%o2)
      return(%foo, %o1, %o2)
  )IR";
  auto graph = stringToGraph(source);

  auto* valUnused = graph->tryGetValue("unused");
  Node* nodeUnused = valUnused->producer();
  EXPECT_EQ(nodeUnused->target(), "aten.fooUnused");

  graph->removeNode(nodeUnused);
  graph->lint();

  // %unused should now be gone
  EXPECT_EQ(graph->tryGetValue("unused"), nullptr)
      << "Value %unused should no longer exist in the graph";

  for (const auto& node : graph->nodes()) {
    EXPECT_NE(node.target(), "aten.fooUnused");
    for (const auto* output : node.outputs()) {
      EXPECT_NE(output->name(), "unused")
          << "Should not find %unused in any remaining node's outputs";
    }
  }
}

TEST(GraphTest, RemoveValue) {
  auto source = R"IR(
    graph(%foo):
  %o1 = aten.foo1(a=%foo)
  %o2 = aten.foo2(a=%o1, b=%foo)
  %o3 = aten.foo3(a=%o2, b=%o2, c=%foo)
  return (%foo, %o1, %o3)
  )IR";

  auto graph = stringToGraph(source);
  auto* val_o1 = graph->tryGetValue("o1");

  {
    // Check we shouldn't be able to remove a value that still has users
    EXPECT_THROW(graph->removeValue(val_o1), c10::Error);
  }

  {
    // Check value removal works as expected
    graph->replaceAllUses(val_o1, graph->tryGetValue("foo"));
    graph->removeValue(val_o1);
    EXPECT_EQ(graph->tryGetValue("%o1"), nullptr);
  }
}

TEST(GraphTest, InsertGraph) {
  auto source = R"IR(
    graph(%foo):
        %o1 = aten.foo1(a=%foo)
        return (%o1)
  )IR";

  // Subgraph to be inserted
  auto subgraphSource = R"IR(
    graph(%x):
        %s1 = aten.subFoo1(a=%x)
        %s2 = aten.subFoo2(a=%s1)
        return (%s2)
  )IR";

  auto mainGraph = stringToGraph(source);
  auto subGraph = stringToGraph(subgraphSource);

  // Insert subGraph into mainGraph. Use %o1 as the subGraph's %x
  auto val_o1 = mainGraph->tryGetValue("o1");
  std::unordered_map<const Value*, Value*> valueMap;
  std::vector<Value*> insertedOutputs =
      mainGraph->insertGraph(*subGraph, {val_o1}, valueMap);

  EXPECT_EQ(insertedOutputs.size(), 1);

  // Check all new nodes are inserted correctly from the copied %s2
  auto* newS2 = insertedOutputs.front();

  auto* newSubFoo2 = newS2->producer();
  EXPECT_EQ(newSubFoo2->target(), "aten.subFoo2");

  auto* newS1 = newSubFoo2->inputs().front().value;
  auto* newSubFoo1 = newS1->producer();
  EXPECT_EQ(newSubFoo1->target(), "aten.subFoo1");

  EXPECT_EQ(newSubFoo1->inputs().front().value, val_o1);

  auto* subInputVal = subGraph->inputs().front();
  EXPECT_EQ(valueMap[subInputVal], val_o1);
  for (const auto& [val1, val2] : valueMap) {
    if (val1->name() == "s1") {
      EXPECT_EQ(val2->name(), newS1->name());
    }
    if (val1->name() == "s2") {
      EXPECT_EQ(val2->name(), newS2->name());
    }
    if (val1->name() == "x") {
      EXPECT_EQ(val2->name(), val_o1->name());
    }
  }

  mainGraph->lint();
}

TEST(GraphTest, CleanupDeadNodes) {
  // %c is unused
  const std::string source = R"(
  graph(%x, %y):
%a = foo(a=%x, b=%y)
%b = foo1(c=%a)
%c = foo2(a=%b, b=%y)
return(%b)
)";
  auto graph = stringToGraph(source);

  // Verify that %c exists initially
  auto* cVal = graph->tryGetValue("c");
  ASSERT_NE(nullptr, cVal);
  size_t nodeCountBefore = graph->nodes().size();

  graph->cleanupDeadNodes();

  // %c should now be gone
  EXPECT_EQ(nullptr, graph->tryGetValue("c"));
  // %b should still be there
  EXPECT_NE(nullptr, graph->tryGetValue("b"));
  EXPECT_EQ(nodeCountBefore - 1, graph->nodes().size());
}

TEST(GraphTest, RenumberValues) {
  const std::string source = R"(
  graph(%x):
%a = foo(a=%x)
%b = foo1(a=%a)
return (%a)
)";
  auto graph = stringToGraph(source);
  graph->cleanupDeadNodes();

  // %b should now be gone
  EXPECT_EQ(nullptr, graph->tryGetValue("b"));

  // %a should now be the last value
  EXPECT_EQ(graph->tryGetValue("a")->id(), graph->numValues() - 1);

  // All values should be renumbered
  size_t numVals = graph->numValues();
  std::unordered_set<ValueId> ids;
  ids.reserve(numVals);
  for (const auto* val : graph->values()) {
    ASSERT_LT(val->id(), numVals);
    ids.insert(val->id());
  }

  // Check ids are contiguous and unique b/w 0 and numVals
  EXPECT_EQ(numVals, ids.size());
  for (size_t i = 0; i < numVals; ++i) {
    EXPECT_NE(ids.end(), ids.find(i));
  }
}

TEST(SerializationTest, RoundTrip) {
  static constexpr std::string_view source =
      R"(graph(%foo, %bar, %baz):
%o1 = aten.foo(self=%foo, target=%bar, alpha=0.1)
return(%o1, %baz)
)";
  const auto graph = stringToGraph(source);
  const auto serialized = graphToString(*graph);
  EXPECT_EQ(source, serialized);
}

TEST(SerializationTest, EscapedStringConstant) {
  const auto parsed =
      std::get<std::string>(convertAtomicConstant(R"("string_\"escape")"));
  std::string expected = "string_\\\"escape";
  EXPECT_EQ(parsed, expected);
}

TEST(SerializationTest, DeviceConstant) {
  const auto device =
      std::get<c10::Device>(convertAtomicConstant("Device{cuda:1}"));
  EXPECT_EQ(device.index(), 1);
  EXPECT_EQ(device.type(), c10::DeviceType::CUDA);
}

TEST(SerializationTest, TrueConstant) {
  const auto parsedTrue = std::get<bool>(convertAtomicConstant("true"));
  EXPECT_EQ(parsedTrue, true);
  const auto parsedFalse = std::get<bool>(convertAtomicConstant("false"));
  EXPECT_EQ(parsedFalse, false);
}

TEST(SerializationTest, MemoryFormatConstant) {
  const auto parsed = std::get<c10::MemoryFormat>(
      convertAtomicConstant("MemoryFormat::ContiguousFormat"));
  EXPECT_EQ(parsed, c10::MemoryFormat::Contiguous);
}

TEST(SerializationTest, FloatConstant) {
  const auto parsed = std::get<double>(convertAtomicConstant("5.0"));
  EXPECT_EQ(parsed, 5.0);
}

TEST(SerializationTest, IntConstant) {
  const auto parsed = std::get<int64_t>(convertAtomicConstant("5"));
  EXPECT_EQ(parsed, 5);
}

TEST(SerializationTest, FloatExponentConstant) {
  const auto parsed = std::get<double>(convertAtomicConstant("1e-05"));
  EXPECT_EQ(parsed, 0.00001);
}

TEST(SerializationTest, SingleElementListConstant) {
  const auto parsed =
      std::get<std::vector<int64_t>>(convertListConstant("[1]"));
  const auto expected = std::vector<int64_t>{1};
  EXPECT_EQ(parsed, expected);
}

TEST(SerializationTest, IntListConstant) {
  const auto parsed =
      std::get<std::vector<int64_t>>(convertListConstant("[1, 2, 3, 4]"));
  const auto expected = std::vector<int64_t>{1, 2, 3, 4};
  EXPECT_EQ(parsed, expected);
}

TEST(SerializationTest, FloatListConstant) {
  const auto parsed = std::get<std::vector<double>>(
      convertListConstant("[1.0, 2.0, 3.0, 4.0]"));
  const auto expected = std::vector<double>{1.0, 2.0, 3.0, 4.0};
  EXPECT_EQ(parsed, expected);
}

TEST(SerializationTest, BoolListConstant) {
  const auto parsed =
      std::get<std::vector<bool>>(convertListConstant("[false, true, false]"));
  const auto expected = std::vector<bool>{false, true, false};
  EXPECT_EQ(parsed, expected);
}

} // namespace torch::nativert
