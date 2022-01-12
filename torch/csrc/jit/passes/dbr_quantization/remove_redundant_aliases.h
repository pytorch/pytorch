#pragma once

#include <torch/csrc/jit/api/module.h>

namespace torch {
namespace jit {

// This function replaces instances of
//
//   %b = aten::alias(%a)
//   %c = foo(%b)
//
// with
//
//   %c = foo(%a)
//
// on the module forward, if it's safe to do so. It currently assumes
// that the forward has been inlined before this function is called.
TORCH_API Module DBRQuantRemoveRedundantAliases(Module& module);

} // namespace jit
} // namespace torch
