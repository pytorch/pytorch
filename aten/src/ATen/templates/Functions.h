#pragma once

#include "ATen/Scalar.h"
#include "ATen/Type.h"
#include "ATen/Tensor.h"
#include "ATen/Storage.h"
#include "ATen/Generator.h"


namespace at {

${function_declarations}

static inline Type & infer_type(const Tensor & t) {
  AT_ASSERT(t.defined(), "undefined Tensor");
  return t.type();
}
static inline Type & infer_type(const TensorList & tl) {
  AT_ASSERT(tl.size() > 0, "expected a non-empty list of Tensors");
  return tl[0].type();
}
// function definitions are all static inline because
// they are one-line statically dispatched functions that
// invoke the actual dynamic dispatch on the correct argument
${function_definitions}

}
