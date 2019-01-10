#pragma once

#include "cudnn-wrapper.h"

namespace at { namespace native {

cudnnHandle_t getCudnnHandle();

}} // namespace
