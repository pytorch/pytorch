#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fmt/format.h>

#include <c10/util/Enumerate.h>
#include <torch/nativert/detail/ITree.h>

namespace torch::nativert::detail {

using torch::nativert::Graph;
using torch::nativert::stringToGraph;
using torch::nativert::Type;
using torch::nativert::Value;

std::pair<std::unique_ptr<Graph>, std::vector<const Value*>> makeValues(
    int count) {
  auto graph = Graph::createGraph();
  std::vector<const Value*> values;

  for (int i = 0; i < count; i++) {
    std::string name = fmt::format("v{}", i);
    Value* value = graph->addValue(name, Type::Kind::None, nullptr);
    values.push_back(value);
  }

  return std::make_pair(std::move(graph), values);
}

TEST(ITreeTest, Unflatten) {
  // Original data: [(0, 1), 2, {"4": 7, "5": 8, "6": 9}, (10,), {"11": 12}]
  auto jsonSpec = R"(
[
  1,
  {
    "type": "builtins.list",
    "context": "null",
    "children_spec": [
      {
        "type": "builtins.tuple",
        "context": "null",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      },
      {
        "type": null,
        "context": null,
        "children_spec": []
      },
      {
        "type": "builtins.dict",
        "context": "[\"4\", \"5\", \"6\"]",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      },
      {
        "type": "torch.fx.immutable_collections.immutable_list",
        "context": "null",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      },
      {
        "type": "torch.fx.immutable_collections.immutable_dict",
        "context": "[\"11\"]",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      }
    ]
  }
]
  )";

  auto [graph, valuePtrs] = makeValues(8);

  const auto spec = itreeSpecLoads(jsonSpec, valuePtrs);
  std::vector<c10::IValue> flats = {
      c10::IValue(0),
      c10::IValue(1),
      c10::IValue(2),
      c10::IValue(7),
      c10::IValue(8),
      c10::IValue(9),
      c10::IValue(10),
      c10::IValue(12),
  };
  auto itree = itreeUnflatten(flats, spec);
  EXPECT_TRUE(itree.isList());
  EXPECT_EQ(itree.toListRef().size(), 5);

  EXPECT_TRUE(itree.toListRef().at(0).isTuple());
  EXPECT_EQ(itree.toListRef().at(0).toTupleRef().elements()[0], c10::IValue(0));
  EXPECT_EQ(itree.toListRef().at(0).toTupleRef().elements()[1], c10::IValue(1));

  EXPECT_TRUE(itree.toListRef().at(1).isInt());
  EXPECT_EQ(itree.toListRef().at(1), c10::IValue(2));

  EXPECT_TRUE(itree.toListRef().at(2).isGenericDict());
  EXPECT_EQ(itree.toListRef().at(2).toGenericDict().at("4"), c10::IValue(7));
  EXPECT_EQ(itree.toListRef().at(2).toGenericDict().at("5"), c10::IValue(8));
  EXPECT_EQ(itree.toListRef().at(2).toGenericDict().at("6"), c10::IValue(9));

  EXPECT_TRUE(itree.toListRef().at(3).isList());
  EXPECT_EQ(itree.toListRef().at(3).toListRef().at(0), c10::IValue(10));

  EXPECT_TRUE(itree.toListRef().at(4).isGenericDict());
  EXPECT_EQ(itree.toListRef().at(4).toGenericDict().at("11"), c10::IValue(12));

  const auto flattened = itreeFlatten(itree, spec);
  EXPECT_EQ(flattened.size(), flats.size());
  for (size_t i = 0; i < flattened.size(); i++) {
    EXPECT_EQ(flattened[i], flats[i]);
  }
}

TEST(ITreeTest, NoVersion) {
  auto jsonSpec = R"(
  {
    "type": "builtins.list",
    "context": "null",
    "children_spec": [
      {
        "type": "builtins.tuple",
        "context": "null",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      }
    ]
  }
  )";

  auto [graph, valuePtrs] = makeValues(2);
  EXPECT_THROW({ itreeSpecLoads(jsonSpec, valuePtrs); }, std::exception);
}

TEST(ITreeTest, NoField) {
  auto jsonSpec = R"(
[
  1,
  {
    "type": "builtins.list",
    "context": "null",
    "children_spec": [
      {
        "children_spec": []
      },
      {
        "type": "builtins.dict",
        "context": "[\"4\", \"5\", \"6\"]",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      }
    ]
  }
]
  )";

  auto [graph, valuePtrs] = makeValues(3);
  EXPECT_THROW(itreeSpecLoads(jsonSpec, valuePtrs), std::exception);
}

TEST(ITreeTest, NoContext) {
  auto jsonSpec = R"(
[
  1,
  {
    "type": "builtins.list",
    "context": "null",
    "children_spec": [
      {
        "type": "builtins.dict",
        "context": "[]",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      }
    ]
  }
]
  )";
  auto [graph, valuePtrs] = makeValues(3);
  auto spec = itreeSpecLoads(jsonSpec, valuePtrs);

  std::vector<c10::IValue> flats = {
      c10::IValue(7),
      c10::IValue(8),
      c10::IValue(9),
  };
  EXPECT_THROW({ itreeUnflatten(flats, spec); }, c10::Error);
}

TEST(ITreeTest, TooManyContext) {
  auto jsonSpec = R"(
[
  1,
  {
    "type": "builtins.list",
    "context": "null",
    "children_spec": [
      {
        "type": "builtins.dict",
        "context": "[\"4\", \"5\", \"6\", \"10\"]",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      }
    ]
  }
]
  )";

  auto [graph, valuePtrs] = makeValues(3);
  auto spec = itreeSpecLoads(jsonSpec, valuePtrs);

  std::vector<c10::IValue> flats = {
      c10::IValue(7),
      c10::IValue(8),
      c10::IValue(9),
  };
  EXPECT_THROW({ itreeUnflatten(flats, spec); }, c10::Error);
}

TEST(ITreeTest, DoubleRegister) {
  EXPECT_THROW(
      { registerPytreeNode("builtins.dict", NodeDef{}); }, std::exception);
}

TEST(ITreeTest, NotEnoughUnflatten) {
  // Original data: [(0, 1), 2, {"4": 7, "5": 8, "6": 9}]
  auto jsonSpec = R"(
[
  1,
  {
    "type": "builtins.list",
    "context": "null",
    "children_spec": [
      {
        "type": "builtins.tuple",
        "context": "null",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      },
      {
        "type": null,
        "context": null,
        "children_spec": []
      },
      {
        "type": "builtins.dict",
        "context": "[\"4\", \"5\", \"6\"]",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      }
    ]
  }
]
  )";
  auto [graph, valuePtrs] = makeValues(6);
  const auto spec = itreeSpecLoads(jsonSpec, valuePtrs);
  std::vector<c10::IValue> flats = {
      c10::IValue(0),
      c10::IValue(1),
      c10::IValue(2),
      c10::IValue(7),
  };
  EXPECT_THROW({ itreeUnflatten(flats, spec); }, c10::Error);
}

TEST(ITreeTest, TooManyUnflatten) {
  // Original data: [(0, 1), 2, {"4": 7, "5": 8, "6": 9}]
  auto jsonSpec = R"(
[
  1,
  {
    "type": "builtins.list",
    "context": "null",
    "children_spec": [
      {
        "type": "builtins.tuple",
        "context": "null",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      },
      {
        "type": null,
        "context": null,
        "children_spec": []
      },
      {
        "type": "builtins.dict",
        "context": "[\"4\", \"5\", \"6\"]",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      }
    ]
  }
]
  )";
  auto [graph, valuePtrs] = makeValues(6);
  const auto spec = itreeSpecLoads(jsonSpec, valuePtrs);
  std::vector<c10::IValue> flats = {
      c10::IValue(0),
      c10::IValue(1),
      c10::IValue(2),
      c10::IValue(7),
      c10::IValue(0),
      c10::IValue(1),
      c10::IValue(2),
      c10::IValue(7),
      c10::IValue(0),
      c10::IValue(1),
      c10::IValue(2),
      c10::IValue(7),
  };
  EXPECT_THROW({ itreeUnflatten(flats, spec); }, c10::Error);
}

TEST(ITreeTest, Flatten) {
  // Original data: [(0, 1), 2, {"4": 7, "5": 8, "6": 9}, (10,), {"11": 12}]
  auto jsonSpec = R"(
[
  1,
  {
    "type": "builtins.list",
    "context": "null",
    "children_spec": [
      {
        "type": "builtins.tuple",
        "context": "null",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      },
      {
        "type": null,
        "context": null,
        "children_spec": []
      },
      {
        "type": "builtins.dict",
        "context": "[\"4\", \"5\", \"6\"]",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      },
      {
        "type": "torch.fx.immutable_collections.immutable_list",
        "context": "null",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      },
      {
        "type": "torch.fx.immutable_collections.immutable_dict",
        "context": "[\"11\"]",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      }
    ]
  }
]
  )";
  auto [graph, valuePtrs] = makeValues(8);
  const auto spec = itreeSpecLoads(jsonSpec, valuePtrs);
  auto tup = c10::ivalue::Tuple::create({c10::IValue(0), c10::IValue(1)});
  c10::Dict<c10::IValue, c10::IValue> dict(
      c10::StringType::get(), c10::AnyType::get());
  dict.insert("4", c10::IValue(7));
  dict.insert("5", c10::IValue(8));
  dict.insert("6", c10::IValue(9));
  c10::List<c10::IValue> ilist(c10::AnyType::get());
  ilist.push_back(c10::IValue(10));
  c10::Dict<c10::IValue, c10::IValue> idict(
      c10::StringType::get(), c10::AnyType::get());
  idict.insert("11", c10::IValue(12));
  c10::List<c10::IValue> list(c10::AnyType::get());
  list.push_back(std::move(tup));
  list.push_back(c10::IValue(2));
  list.push_back(std::move(dict));
  list.push_back(std::move(ilist));
  list.push_back(std::move(idict));
  auto flats = itreeFlatten(c10::IValue{list}, spec);
  std::vector<c10::IValue> expected = {
      c10::IValue(0),
      c10::IValue(1),
      c10::IValue(2),
      c10::IValue(7),
      c10::IValue(8),
      c10::IValue(9),
      c10::IValue(10),
      c10::IValue(12),
  };
  for (const auto& [i, flat] : c10::enumerate(flats)) {
    EXPECT_EQ(flat, expected.at(i));
  }
}

TEST(ITreeTest, IValueApplyFromArgs) {
  // inputSpec for testing is generated from E2ETestModelWithNestedDictInput
  /*
      args = (
        {
            "a": (
                torch.rand(4, 4),
                {
                    123: (torch.rand(4, 4), torch.rand(4, 4)),
                    234: (torch.rand(4, 4), torch.rand(4, 4)),
                },
            ),
            "b": (
                torch.rand(4, 4),
                {
                    345: (torch.rand(4, 4), torch.rand(4, 4)),
                    456: (torch.rand(4, 4), torch.rand(4, 4)),
                },
            ),
        },
    )*/
  auto jsonSpec = R"(
[
    1,
    {
        "type": "builtins.tuple",
        "context": "null",
        "children_spec": [
            {
                "type": "builtins.tuple",
                "context": "null",
                "children_spec": [
                    {
                        "type": "builtins.dict",
                        "context": "[\"a\", \"b\"]",
                        "children_spec": [
                            {
                                "type": "builtins.tuple",
                                "context": "null",
                                "children_spec": [
                                    {
                                        "type": null,
                                        "context": null,
                                        "children_spec": []
                                    },
                                    {
                                        "type": "builtins.dict",
                                        "context": "[123, 234]",
                                        "children_spec": [
                                            {
                                                "type": "builtins.tuple",
                                                "context": "null",
                                                "children_spec": [
                                                    {
                                                        "type": null,
                                                        "context": null,
                                                        "children_spec": []
                                                    },
                                                    {
                                                        "type": null,
                                                        "context": null,
                                                        "children_spec": []
                                                    }
                                                ]
                                            },
                                            {
                                                "type": "builtins.tuple",
                                                "context": "null",
                                                "children_spec": [
                                                    {
                                                        "type": null,
                                                        "context": null,
                                                        "children_spec": []
                                                    },
                                                    {
                                                        "type": null,
                                                        "context": null,
                                                        "children_spec": []
                                                    }
                                                ]
                                            }
                                        ]
                                    }
                                ]
                            },
                            {
                                "type": "builtins.tuple",
                                "context": "null",
                                "children_spec": [
                                    {
                                        "type": null,
                                        "context": null,
                                        "children_spec": []
                                    },
                                    {
                                        "type": "builtins.dict",
                                        "context": "[345, 456]",
                                        "children_spec": [
                                            {
                                                "type": "builtins.tuple",
                                                "context": "null",
                                                "children_spec": [
                                                    {
                                                        "type": null,
                                                        "context": null,
                                                        "children_spec": []
                                                    },
                                                    {
                                                        "type": null,
                                                        "context": null,
                                                        "children_spec": []
                                                    }
                                                ]
                                            },
                                            {
                                                "type": "builtins.tuple",
                                                "context": "null",
                                                "children_spec": [
                                                    {
                                                        "type": null,
                                                        "context": null,
                                                        "children_spec": []
                                                    },
                                                    {
                                                        "type": null,
                                                        "context": null,
                                                        "children_spec": []
                                                    }
                                                ]
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
                ]
            },
            {
                "type": "builtins.dict",
                "context": "[]",
                "children_spec": []
            }
        ]
    }
]
  )";

  auto tup_a1_123 =
      c10::ivalue::Tuple::create({c10::IValue(1), c10::IValue(2)});
  auto tup_a1_234 =
      c10::ivalue::Tuple::create({c10::IValue(3), c10::IValue(4)});
  c10::Dict<c10::IValue, c10::IValue> dict_a1(
      c10::StringType::get(), c10::AnyType::get());
  dict_a1.insert(123, tup_a1_123);
  dict_a1.insert(234, tup_a1_234);
  auto tup_a =
      c10::ivalue::Tuple::create({c10::IValue(0), c10::IValue(dict_a1)});

  auto tup_b1_345 =
      c10::ivalue::Tuple::create({c10::IValue(6), c10::IValue(7)});
  auto tup_b1_456 =
      c10::ivalue::Tuple::create({c10::IValue(8), c10::IValue(9)});
  c10::Dict<c10::IValue, c10::IValue> dict_b1(
      c10::StringType::get(), c10::AnyType::get());
  dict_b1.insert(345, tup_b1_345);
  dict_b1.insert(456, tup_b1_456);
  auto tup_b =
      c10::ivalue::Tuple::create({c10::IValue(5), c10::IValue(dict_b1)});

  c10::Dict<c10::IValue, c10::IValue> dict(
      c10::StringType::get(), c10::AnyType::get());
  dict.insert("a", tup_a);
  dict.insert("b", tup_b);
  std::vector<c10::IValue> args = {c10::IValue(dict)};

  for (int usedIdx = 0; usedIdx < 10; usedIdx++) {
    std::vector<bool> isUsed(10, false);
    isUsed[usedIdx] = true;
    std::stringstream ss;
    for (int i = 0; i < 10; ++i) {
      if (isUsed[i]) {
        ss << fmt::format("%o1 = aten.foo(a=%a{})\n", i);
      }
    }
    std::string source = fmt::format(
        R"(graph(%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7, %a8, %a9):
{}
return(%o1)
)",
        ss.str());

    auto graph = stringToGraph(source);
    std::vector<const Value*> userInputs(
        graph->userInputs().begin(), graph->userInputs().end());

    const auto spec = itreeSpecLoads(jsonSpec, userInputs);

    std::vector<int> visited;
    auto fn = [&](const c10::IValue& leaf, const Value* value) {
      visited.push_back(value->id());
    };
    ivalueApplyFromArgs(fn, args, {}, spec);

    EXPECT_EQ(visited.size(), 1);
    EXPECT_EQ(visited[0], usedIdx);
  }
}

TEST(ITreeTest, UnmatchedFlattenType) {
  // Original data: [(0, 1), 2, {"4": 7, "5": 8, "6": 9}]
  auto jsonSpec = R"(
[
  1,
  {
    "type": "builtins.list",
    "context": "null",
    "children_spec": [
      {
        "type": "builtins.tuple",
        "context": "null",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      },
      {
        "type": null,
        "context": null,
        "children_spec": []
      },
      {
        "type": "builtins.dict",
        "context": "[\"4\", \"5\", \"6\"]",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      }
    ]
  }
]
  )";
  auto [graph, valuePtrs] = makeValues(6);
  const auto spec = itreeSpecLoads(jsonSpec, valuePtrs);
  auto tup = c10::ivalue::Tuple::create({c10::IValue(0), c10::IValue(1)});
  c10::Dict<c10::IValue, c10::IValue> dict(
      c10::StringType::get(), c10::AnyType::get());
  dict.insert("4", c10::IValue(7));
  dict.insert("5", c10::IValue(8));
  dict.insert("6", c10::IValue(9));
  EXPECT_THROW(
      { itreeFlatten(c10::IValue{std::move(dict)}, spec); }, std::exception);
}

TEST(ITreeTest, UnmatchedDictFlatten) {
  // Original data: [(0, 1), 2, {"4": 7, "5": 8, "6": 9}]
  auto jsonSpec = R"(
[
  1,
  {
    "type": "builtins.list",
    "context": "null",
    "children_spec": [
      {
        "type": "builtins.tuple",
        "context": "null",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      },
      {
        "type": null,
        "context": null,
        "children_spec": []
      },
      {
        "type": "builtins.dict",
        "context": "[\"4\", \"5\", \"6\"]",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      }
    ]
  }
]
  )";
  auto [graph, valuePtrs] = makeValues(6);
  const auto spec = itreeSpecLoads(jsonSpec, valuePtrs);
  auto tup = c10::ivalue::Tuple::create({c10::IValue(0), c10::IValue(1)});
  c10::Dict<c10::IValue, c10::IValue> dict(
      c10::StringType::get(), c10::AnyType::get());
  dict.insert("4", c10::IValue(7));
  dict.insert("5", c10::IValue(8));
  dict.insert("100", c10::IValue(8));
  dict.insert("101", c10::IValue(8));
  c10::List<c10::IValue> list(c10::AnyType::get());
  list.push_back(std::move(tup));
  list.push_back(c10::IValue(2));
  list.push_back(std::move(dict));
  EXPECT_THROW(
      { itreeFlatten(c10::IValue{std::move(list)}, spec); }, c10::Error);
}

TEST(ITreeTest, DictFlattenTest) {
  auto jsonSpec = R"(
[
  1,
  {
    "type": "builtins.list",
    "context": "null",
    "children_spec": [
      {
        "type": "builtins.dict",
        "context": "[\"4\", \"5\", \"6\"]",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      }
    ]
  }
]
  )";
  auto [graph, valuePtrs] = makeValues(3);
  const auto spec = itreeSpecLoads(jsonSpec, valuePtrs);
  c10::Dict<c10::IValue, c10::IValue> dict(
      c10::StringType::get(), c10::AnyType::get());
  // allow dict.size < context
  // test dict.size=2 , context,size=3,
  dict.insert("4", c10::IValue(7));
  dict.insert("5", c10::IValue(8));
  c10::List<c10::IValue> list(c10::AnyType::get());
  list.push_back(std::move(dict));
  itreeFlatten(c10::IValue{std::move(list)}, spec);
}

TEST(ITreeTest, UnmatchedTupleFlatten) {
  // Original data: [(0, 1), 2, {"4": 7, "5": 8, "6": 9}]
  auto jsonSpec = R"(
[
  1,
  {
    "type": "builtins.list",
    "context": "null",
    "children_spec": [
      {
        "type": "builtins.tuple",
        "context": "null",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      },
      {
        "type": null,
        "context": null,
        "children_spec": []
      },
      {
        "type": "builtins.dict",
        "context": "[\"4\", \"5\", \"6\"]",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      }
    ]
  }
]
  )";
  auto [graph, valuePtrs] = makeValues(6);
  const auto spec = itreeSpecLoads(jsonSpec, valuePtrs);
  auto tup = c10::ivalue::Tuple::create({c10::IValue(0)});
  c10::Dict<c10::IValue, c10::IValue> dict(
      c10::StringType::get(), c10::AnyType::get());
  dict.insert("4", c10::IValue(7));
  dict.insert("5", c10::IValue(8));
  dict.insert("6", c10::IValue(8));
  c10::List<c10::IValue> list(c10::AnyType::get());
  list.push_back(std::move(tup));
  list.push_back(c10::IValue(2));
  list.push_back(std::move(dict));
  EXPECT_THROW(
      { itreeFlatten(c10::IValue{std::move(list)}, spec); }, c10::Error);
}

TEST(ITreeTest, ToAtenType) {
  // Original data: ((0, 1), 2, {"4": 7, "5": 8}, [10], {6: 9})
  auto jsonSpec = R"(
[
  1,
  {
    "type": "builtins.tuple",
    "context": "null",
    "children_spec": [
      {
        "type": "builtins.tuple",
        "context": "null",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      },
      {
        "type": null,
        "context": null,
        "children_spec": []
      },
      {
        "type": "builtins.dict",
        "context": "[\"4\", \"5\"]",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          },
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      },
      {
        "type": "builtins.list",
        "context": "null",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      },
      {
        "type": "builtins.dict",
        "context": "[6]",
        "children_spec": [
          {
            "type": null,
            "context": null,
            "children_spec": []
          }
        ]
      }
    ]
  }
]
  )";
  auto [graph, valuePtrs] = makeValues(7);
  const auto spec = itreeSpecLoads(jsonSpec, valuePtrs);
  auto atenType = spec.toAtenType();

  // Root level is tuple.
  EXPECT_EQ(atenType->kind(), c10::TypeKind::TupleType);
  const c10::TupleType& rootType = atenType->expectRef<c10::TupleType>();
  EXPECT_EQ(rootType.elements().size(), 5);

  at::TypePtr elementType = rootType.elements()[0];
  EXPECT_EQ(elementType->kind(), c10::TypeKind::TupleType);
  EXPECT_EQ(
      elementType->expectRef<c10::TupleType>().elements()[0]->kind(),
      c10::TypeKind::AnyType);
  EXPECT_EQ(
      elementType->expectRef<c10::TupleType>().elements()[1]->kind(),
      c10::TypeKind::AnyType);

  elementType = rootType.elements()[1];
  EXPECT_EQ(elementType->kind(), c10::TypeKind::AnyType);

  elementType = rootType.elements()[2];
  EXPECT_EQ(elementType->kind(), c10::TypeKind::DictType);
  EXPECT_EQ(
      elementType->expectRef<c10::DictType>().getKeyType()->kind(),
      c10::TypeKind::StringType);
  EXPECT_EQ(
      elementType->expectRef<c10::DictType>().getValueType()->kind(),
      c10::TypeKind::AnyType);

  elementType = rootType.elements()[3];
  EXPECT_EQ(elementType->kind(), c10::TypeKind::ListType);
  EXPECT_EQ(
      elementType->expectRef<c10::ListType>().getElementType()->kind(),
      c10::TypeKind::AnyType);

  elementType = rootType.elements()[4];
  EXPECT_EQ(elementType->kind(), c10::TypeKind::DictType);
  EXPECT_EQ(
      elementType->expectRef<c10::DictType>().getKeyType()->kind(),
      c10::TypeKind::IntType);
  EXPECT_EQ(
      elementType->expectRef<c10::DictType>().getValueType()->kind(),
      c10::TypeKind::AnyType);
}

} // namespace torch::nativert::detail
