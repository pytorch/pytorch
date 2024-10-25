#pragma once

#include <ATen/miopen/miopen-wrapper.h>

namespace at { namespace native {

miopenHandle_t getMiopenHandle();

}} // namespace
