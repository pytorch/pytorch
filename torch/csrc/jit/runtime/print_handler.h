
copy: fbcode/caffe2/torch/csrc/jit/runtime/print_handler.h
copyrev: 11e4679b48e3d4927ae16ad6ca827ec0061524ca

#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <atomic>
#include <functional>
#include <iostream>

namespace torch {
namespace jit {

using PrintHandler = void (*)(const std::string&);

TORCH_API void setPrintHandler(PrintHandler ph);
TORCH_API PrintHandler getPrintHandler();

} // namespace jit
} // namespace torch
