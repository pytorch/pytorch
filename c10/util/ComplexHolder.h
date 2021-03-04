#pragma once

#include <c10/util/intrusive_ptr.h>

namespace c10 {

// TODO: Revise Comment...
//
// We need a ComplexHolder because currently the payloads in the Union
// only take 64 bits. Since ComplexDouble takes up 128 bits, and is too big
// to fit in the IValue directly, we indirect complex numbers through an intrusive
// pointer to ComplexHolder (which contains a c10::complex).
struct C10_API ComplexHolder : intrusive_ptr_target {
  public:
    template <typename T>
    ComplexHolder(c10::complex<T> c) {
      val = convert<decltype(val), c10::complex<T>>(c);
    }
    ComplexHolder() {}
    c10::complex<double> val;
};

} // namespace c10
