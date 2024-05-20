#pragma once

#include <c10/macros/Macros.h>
#include <c10/core/GradMode.h>

namespace at {
  using GradMode = c10::GradMode;
  using AutoGradMode = c10::AutoGradMode;
  using NoGradGuard = c10::NoGradGuard;
}
