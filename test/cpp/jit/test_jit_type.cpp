#include <test/cpp/jit/test_base.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/testing/file_check.h>
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/irparser.h"

namespace torch {
namespace jit {

void testUnifyTypes() {
  auto bool_tensor = TensorType::get()->withScalarType(at::kBool);
  auto opt_bool_tensor = OptionalType::create(bool_tensor);
  auto unified_opt_bool = unifyTypes(bool_tensor, opt_bool_tensor);
  TORCH_INTERNAL_ASSERT(opt_bool_tensor->isSubtypeOf(*unified_opt_bool));

  auto tensor = TensorType::get();
  TORCH_INTERNAL_ASSERT(!tensor->isSubtypeOf(opt_bool_tensor));
  auto unified = unifyTypes(opt_bool_tensor, tensor);
  TORCH_INTERNAL_ASSERT(unified);
  auto elem = (*unified)->expect<OptionalType>()->getElementType();
  TORCH_INTERNAL_ASSERT(elem->isSubtypeOf(TensorType::get()));

  auto opt_tuple_none_int = OptionalType::create(
      TupleType::create({NoneType::get(), IntType::get()}));
  auto tuple_int_none = TupleType::create({IntType::get(), NoneType::get()});
  auto out = unifyTypes(opt_tuple_none_int, tuple_int_none);
  TORCH_INTERNAL_ASSERT(out);

  std::stringstream ss;
  ss << (*out)->python_str();
  testing::FileCheck()
      .check("Optional[Tuple[Optional[int], Optional[int]]]")
      ->run(ss.str());
}

} // namespace jit
} // namespace torch
