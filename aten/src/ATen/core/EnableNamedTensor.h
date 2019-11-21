#pragma once

#include <c10/macros/Macros.h>

// We are working on removing the BUILD_NAMEDTENSOR flag from the codebase.
//
// PyTorch's codegen also uses a similar flag. You can find it in
// - aten/src/ATen/env.py
#ifndef BUILD_NAMEDTENSOR
#define BUILD_NAMEDTENSOR
#endif
