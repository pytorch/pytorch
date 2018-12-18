#pragma once

#include <ATen/Allocator.h>
#include <ATen/core/Generator.h>
#include <c10/util/Exception.h>

#include <c10/util/Registry.h>

#include <cstddef>
#include <functional>
#include <memory>

namespace at {
  class Context;
}

// NB: Class must live in `at` due to limitations of Registry.h.
namespace at {

// The MKLDNNHooksInterface is an omnibus interface for any MKLDNN functionality
// which we may want to call into from CPU code (and thus must be dynamically
// dispatched, to allow for separate compilation of MKLDNN code). See
// CUDAHooksInterface for more detailed motivation.
struct CAFFE2_API MKLDNNHooksInterface {
  // This should never actually be implemented, but it is used to
  // squelch -Werror=non-virtual-dtor
  virtual ~MKLDNNHooksInterface() {}

  virtual bool hasMKLDNN() const {
    return false;
  }

  virtual bool compiledWithMKLDNN() const {
    return false;
  }

  virtual bool supportsDilatedConvolutionWithMKLDNN() const {
    return false;
  }

  virtual bool supportsRNNWithMKLDNN() const {
    return false;
  }
};

// NB: dummy argument to suppress "ISO C++11 requires at least one argument
// for the "..." in a variadic macro"
struct CAFFE2_API MKLDNNHooksArgs {};

C10_DECLARE_REGISTRY(MKLDNNHooksRegistry, MKLDNNHooksInterface, MKLDNNHooksArgs);
#define REGISTER_MKLDNN_HOOKS(clsname) \
      C10_REGISTER_CLASS(MKLDNNHooksRegistry, clsname, clsname)

namespace detail {
CAFFE2_API const MKLDNNHooksInterface& getMKLDNNHooks();
} // namespace detail
} // namespace at
