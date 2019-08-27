#pragma once

// We are incrementally working on deleting the BUILD_NAMEDTENSOR flag from
// the codebase. For now, always define the macro.
//
// PyTorch's codegen also uses a similar flag. You can find it in
// aten/src/ATen/env.py and tools/autograd/env.py
#ifndef BUILD_NAMEDTENSOR
#define BUILD_NAMEDTENSOR
#endif
