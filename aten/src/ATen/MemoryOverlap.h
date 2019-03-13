#pragma once

#include <ATen/ATen.h>

namespace at {

// MemOverlap: Whether or not there is memory overlap
//
// NO: Absolutely no memory overlap
// YES: Absolutely yes memory overlap
// TOO_HARD: There might be memory overlap, but it was too expensive to compute.
//
// NB: Please update the python test for these if you renumber them.
enum class MemOverlap { NO, YES, TOO_HARD };

CAFFE2_API MemOverlap has_internal_overlap(const Tensor& t);
CAFFE2_API MemOverlap has_internal_overlap(TensorImpl* t);

CAFFE2_API void assert_no_internal_overlap(const Tensor& t, std::string op);
CAFFE2_API void assert_no_internal_overlap(TensorImpl* t, std::string op);

}
