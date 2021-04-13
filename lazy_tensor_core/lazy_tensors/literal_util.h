#pragma once

#include "lazy_tensors/literal.h"
#include "lazy_tensors/primitive_util.h"
#include "lazy_tensors/span.h"

namespace lazy_tensors {

class LiteralUtil {
 public:
  template <typename NativeT>
  static Literal CreateR0(NativeT value) {
    Literal literal(ShapeUtil::MakeShape(
        primitive_util::NativeToPrimitiveType<NativeT>(), {}));
    literal.Set<NativeT>({}, value);
    return literal;
  }

  template <typename NativeT>
  static Literal CreateR1(lazy_tensors::Span<const NativeT> values) {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }
};

}  // namespace lazy_tensors
