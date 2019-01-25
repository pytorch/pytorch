#pragma once
#include <ATen/core/ivalue.h>

namespace torch {
namespace jit {

using ::c10::ivalue::List;
using ::c10::ivalue::Shared;

using ::c10::IValue;
using ::c10::ivalue::Future;
using ::c10::ivalue::Tuple;

using ::c10::ivalue::BoolList;
using ::c10::ivalue::DoubleList;
using ::c10::ivalue::GenericList;
using ::c10::ivalue::IntList;
using ::c10::ivalue::TensorList;

using ::c10::ivalue::ConstantString;

// XXX: This function is to specialize IValue for tensor type in
// interpreter, it should only be used in JIT and not anywhere
// else, that's why we keep this in JIT but not in c10::ivalue
//
// TODO: remove this and merge ivalue.h when we remove the
// undefined tensor semantic from TH
inline at::Tensor toOptionalTensor(const IValue& v) {
  if (v.isNone()) {
    return at::Tensor();
  }
  return v.toTensor();
}

} // namespace jit
} // namespace torch
