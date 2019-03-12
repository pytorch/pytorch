#pragma once

#include <ATen/ATen.h>

namespace at {

enum MemOverlap { kNo, kYes, kTooHard };

CAFFE2_API MemOverlap has_internal_overlap(const Tensor& t);
CAFFE2_API MemOverlap has_internal_overlap(TensorImpl* t);

CAFFE2_API void assert_no_internal_overlap(const Tensor& t, std::string op);
CAFFE2_API void assert_no_internal_overlap(TensorImpl* t, std::string op);

}
