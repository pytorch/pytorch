#pragma once

#include <ATen/ATen.h>

namespace at {

enum MemOverlap { kNo, kYes, kTooHard };

MemOverlap has_internal_overlap(const Tensor& t);

void assert_no_internal_overlap(const Tensor& t, std::string op);

}
