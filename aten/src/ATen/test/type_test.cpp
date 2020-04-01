#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "ATen/core/jit_type.h"

namespace c10 {

TEST(TypeCustomPrinter, Basic) {
  TypePrinter printer =
      [](const ConstTypePtr& t) -> c10::optional<std::string> {
    if (auto tensorType = t->cast<TensorType>()) {
      return "CustomTensor";
    }
    return c10::nullopt;
  };

  // Tensor types should be rewritten
  torch::Tensor iv = torch::rand({2, 3});
  const auto type = TensorType::create(iv);
  EXPECT_EQ(type->python_str(), "Tensor");
  EXPECT_EQ(type->python_str(printer), "CustomTensor");

  // Unrelated types shoudl not be affected
  const auto intType = IntType::create();
  EXPECT_EQ(intType->python_str(printer), intType->python_str());
}

TEST(TypeCustomPrinter, ContainedTypes) {
  TypePrinter printer =
      [](const ConstTypePtr& t) -> c10::optional<std::string> {
    if (auto tensorType = t->cast<TensorType>()) {
      return "CustomTensor";
    }
    return c10::nullopt;
  };
  torch::Tensor iv = torch::rand({2, 3});
  const auto type = TensorType::create(iv);

  // Contained types should work
  const auto tupleType = TupleType::create({type, IntType::get(), type});
  EXPECT_EQ(tupleType->python_str(), "Tuple[Tensor, int, Tensor]");
  EXPECT_EQ(
      tupleType->python_str(printer), "Tuple[CustomTensor, int, CustomTensor]");
  const auto dictType = DictType::create(IntType::get(), type);
  EXPECT_EQ(dictType->python_str(printer), "Dict[int, CustomTensor]");
  const auto listType = ListType::create(tupleType);
  EXPECT_EQ(
      listType->python_str(printer),
      "List[Tuple[CustomTensor, int, CustomTensor]]");
}

TEST(TypeCustomPrinter, NamedTuples) {
  TypePrinter printer =
      [](const ConstTypePtr& t) -> c10::optional<std::string> {
    if (auto tupleType = t->cast<TupleType>()) {
      // Rewrite only namedtuples
      if (tupleType->name()) {
        return "Rewritten";
      }
    }
    return c10::nullopt;
  };
  torch::Tensor iv = torch::rand({2, 3});
  const auto type = TensorType::create(iv);

  const auto namedTupleType = TupleType::createNamed(
      "my.named.tuple", {"foo", "bar"}, {type, IntType::get()});
  EXPECT_EQ(namedTupleType->python_str(printer), "Rewritten");

  // Put it inside another tuple, should still work
  const auto outerTupleType = TupleType::create({IntType::get(), namedTupleType});
  EXPECT_EQ(outerTupleType->python_str(printer), "Tuple[int, Rewritten]");
}
} // namespace c10
