#pragma once

// We are incrementally working on deleting the BUILD_NAMEDTENSOR flag from
// the codebase. For now, always define the macro.
//
// PyTorch's codegen also uses a similar flag. You can find it in
// - aten/src/ATen/env.py
#if !defined(CAFFE2_IS_XPLAT_BUILD) && (!defined(C10_MOBILE) || defined(FEATURE_TORCH_MOBILE))
#define BUILD_NAMEDTENSOR
#endif
