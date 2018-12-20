#include <ATen/core/ivalue.h>

namespace torch {
namespace jit {

using ::c10::ivalue::List;
using ::c10::ivalue::Shared;

using ::c10::IValue;
using ::c10::ivalue::Tuple;
using ::c10::ivalue::Future;

using ::c10::ivalue::BoolList;
using ::c10::ivalue::DoubleList;
using ::c10::ivalue::GenericList;
using ::c10::ivalue::IntList;
using ::c10::ivalue::TensorList;

using ::c10::ivalue::ConstantString;

} // namespace jit
} // namespace torch
