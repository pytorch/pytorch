#include "ATen/Config.h"

#include "ATen/Scalar.h"

#include <TH/TH.h>

#include "ATen/Tensor.h"
#include "ATen/TensorMethods.h"
#include "ATen/TensorOperators.h"
#include "ATen/Context.h"
#include "ATen/TensorMethods.h"

namespace at {

Tensor Scalar::toTensor() const {
  if (Tag::HAS_d == tag) {
    return CPU(kDouble).scalarTensor(*this);
  } else {
    assert(Tag::HAS_i == tag);
    return CPU(kLong).scalarTensor(*this);
  }
}

Scalar Scalar::operator-() const {
 if (isFloatingPoint()) {
   return Scalar(-v.d);
 } else {
   return Scalar(-v.i);
 }
}

}  // namespace at
