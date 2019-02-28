#pragma once

#include <ATen/ATen.h>

namespace at {

enum MemOverlap { kNo, kYes, kTooHard };

MemOverlap has_internal_overlap(const Tensor& t);

}
