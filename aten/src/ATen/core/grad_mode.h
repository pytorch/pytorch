#pragma once

#include <c10/core/GradMode.h>
#include <c10/macros/Macros.h>

namespace at {
using GradMode = c10::GradMode;
using AutoGradMode = c10::AutoGradMode;
using NoGradGuard = c10::NoGradGuard;
} // namespace at
