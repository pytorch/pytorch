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

MemOverlap has_internal_overlap(const Tensor& t);

void assert_no_internal_overlap(const Tensor& t, std::string op);

}
