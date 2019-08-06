#pragma once

#include <atomic>

#include <c10/macros/Export.h>

namespace at {
namespace native {

// *** Warning: this code is here to workaround an issue:
// https://github.com/pytorch/pytorch/issues/23825
//
// This flag allows us to temporarily disable MKLDNN to work around cases
// where there are bugs.
extern CAFFE2_API std::atomic<bool> disable_mkldnn_conv;

}  // namespace at
}  // namespace native
