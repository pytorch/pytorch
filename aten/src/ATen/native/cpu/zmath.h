#pragma once

// Complex number math operations that act as no-ops for other dtypes.
#include <c10/util/complex.h>
#include <c10/util/MathConstants.h>
#include <ATen/NumericUtils.h>

#include <torch/headeronly/cpu/vec/zmath.h>
