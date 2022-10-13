#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/testing/file_check.h>
#include "torch/csrc/jit/ir/ir.h"
#include "torch/csrc/jit/ir/irparser.h"

namespace torch {
namespace jit {

TEST(JitTypeTest, IsComplete) {
  auto tt = c10::TensorType::create(
      at::kFloat,
      at::kCPU,
      c10::SymbolicShape(std::vector<c10::optional<int64_t>>({1, 49})),
      std::vector<c10::Stride>(
          {c10::Stride{2, true, 1},
           c10::Stride{1, true, 1},
           c10::Stride{0, true, c10::nullopt}}),
      false);
  TORCH_INTERNAL_ASSERT(!tt->isComplete());
  TORCH_INTERNAL_ASSERT(!tt->strides().isComplete());
}

TEST(JitTypeTest, UnifyTypes) {
  auto bool_tensor = TensorType::get()->withScalarType(at::kBool);
  auto opt_bool_tensor = OptionalType::create(bool_tensor);
  auto unified_opt_bool = unifyTypes(bool_tensor, opt_bool_tensor);
  TORCH_INTERNAL_ASSERT(opt_bool_tensor->isSubtypeOf(**unified_opt_bool));

  auto tensor = TensorType::get();
  TORCH_INTERNAL_ASSERT(!tensor->isSubtypeOf(*opt_bool_tensor));
  auto unified = unifyTypes(opt_bool_tensor, tensor);
  TORCH_INTERNAL_ASSERT(unified);
  auto elem = (*unified)->expectRef<OptionalType>().getElementType();
  TORCH_INTERNAL_ASSERT(elem->isSubtypeOf(*TensorType::get()));

  auto opt_tuple_none_int = OptionalType::create(
      TupleType::create({NoneType::get(), IntType::get()}));
  auto tuple_int_none = TupleType::create({IntType::get(), NoneType::get()});
  auto out = unifyTypes(opt_tuple_none_int, tuple_int_none);
  TORCH_INTERNAL_ASSERT(out);

  std::stringstream ss;
  ss << (*out)->annotation_str();
  testing::FileCheck()
      .check("Optional[Tuple[Optional[int], Optional[int]]]")
      ->run(ss.str());

  auto fut_1 = FutureType::create(IntType::get());
  auto fut_2 = FutureType::create(NoneType::get());
  auto fut_out = unifyTypes(fut_1, fut_2);
  TORCH_INTERNAL_ASSERT(fut_out);
  TORCH_INTERNAL_ASSERT((*fut_out)->isSubtypeOf(
      *FutureType::create(OptionalType::create(IntType::get()))));

  auto dict_1 = DictType::create(IntType::get(), NoneType::get());
  auto dict_2 = DictType::create(IntType::get(), IntType::get());
  auto dict_out = unifyTypes(dict_1, dict_2);
  TORCH_INTERNAL_ASSERT(!dict_out);
}

TEST(JitTypeTest, ListTypeOfTensorsIsSubtypeOf) {
  auto list_type = ListType::create(TensorType::get());
  auto tensor_type = c10::TensorType::create(
      at::kFloat,
      at::kCPU,
      c10::SymbolicShape(std::vector<c10::optional<int64_t>>({1, 49})),
      std::vector<c10::Stride>(
          {c10::Stride{2, true, 1},
           c10::Stride{1, true, 1},
           c10::Stride{0, true, c10::nullopt}}),
      false);
  auto list_type_modified = ListType::create(tensor_type);
  TORCH_INTERNAL_ASSERT(list_type_modified->isSubtypeOf(list_type));
}

} // namespace jit
} // namespace torch
