#pragma once

#include <torch/headeronly/util/Math.h>

// All of the special functions previously defined in this file now live in
// torch/headeronly/util/Math.h, under torch::headeronly::native. They were
// (and still are) defined at global scope, so re-inject them here to keep
// every existing unqualified call site (calc_erfinv, zeta, calc_digamma,
// bessel_j0_forward, etc.) working unchanged.
using namespace torch::headeronly::native; // NOLINT(*-build-using-namespace)
