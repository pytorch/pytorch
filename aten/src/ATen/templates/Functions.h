#pragma once

#include "ATen/Scalar.h"
#include "ATen/Type.h"
#include "ATen/Tensor.h"
#include "ATen/Storage.h"
#include "ATen/Generator.h"



namespace at {

static inline Tensor & copy_out(const Tensor & src, Tensor & dst) {
  dst.resize_(src.sizes());
  dst.type().copy(src,dst);
  return dst;
}

${function_declarations}

// function definitions are all static inline because
// they are one-line statically dispatched functions that
// invoke the actual dynamic dispatch on the correct argument
${function_definitions}

}
