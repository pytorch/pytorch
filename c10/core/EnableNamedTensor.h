#pragma once

// We are working on removing the BUILD_NAMEDTENSOR flag from the codebase.
//
// PyTorch's codegen also uses a similar flag. You can find it in
// - aten/src/ATen/env.py
#if !defined(CAFFE2_IS_XPLAT_BUILD) && (!defined(C10_MOBILE) || defined(FEATURE_TORCH_MOBILE))
#ifndef BUILD_NAMEDTENSOR
#define BUILD_NAMEDTENSOR
#endif
#endif
