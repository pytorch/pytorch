#pragma once

#include <torch/csrc/inductor/aoti_runtime/sycl_runtime_wrappers.h>
#include <torch/csrc/inductor/aoti_runtime/utils_xpu.h>
#ifndef AOTI_STANDALONE
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_xpu.h>
#endif
