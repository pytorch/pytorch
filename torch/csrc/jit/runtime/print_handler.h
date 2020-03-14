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
