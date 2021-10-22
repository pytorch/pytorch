#pragma once

#include "lazy_tensors/literal.h"
#include "lazy_tensors/span.h"

namespace lazy_tensors {

class LiteralUtil {
 public:
  template <typename NativeT>
  static Literal CreateR0(NativeT value);

  template <typename NativeT>
  static Literal CreateR1(lazy_tensors::Span<const NativeT> values);
};

template <typename NativeT>
/* static */ Literal LiteralUtil::CreateR0(NativeT value) {
  Literal literal(
      ShapeUtil::MakeShape(c10::CppTypeToScalarType<NativeT>::value, {}));
  literal.Set<NativeT>({}, value);
  return literal;
}


}  // namespace lazy_tensors
