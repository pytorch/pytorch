#pragma once

#include <c10/macros/Macros.h>

namespace c10 {

/**
 * A kernel can keep around a cache to have better performance when it's
 * called multiple times. This is used by a lot of caffe2 kernels, for example
 * conv_op stores a set of tensors for intermediate values to avoid having
 * to reallocate them on each call.
 * This cache owned by the call site (i.e. stored inside the OpKernel object)
 * kept at the call site to call into the kernel) and passed in to the kernel
 * as a function argument. It must inherit from KernelCache so the call site
 * knows how to store and destruct it.
 */
class CAFFE2_API KernelCache {
public:
  virtual ~KernelCache() = default;
};

}
