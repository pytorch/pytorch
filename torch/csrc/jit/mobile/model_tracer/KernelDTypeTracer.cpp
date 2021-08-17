// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#include <caffe2/fb/model_tracer/KernelDTypeTracer.h>
#include <map>
#include <set>
#include <string>

namespace facebook::pytorch {
KernelDTypeTracer::kernel_tags_type& KernelDTypeTracer::getCalledKernelTags() {
  static kernel_tags_type called_kernel_tags;
  return called_kernel_tags;
}
} // namespace facebook::pytorch
