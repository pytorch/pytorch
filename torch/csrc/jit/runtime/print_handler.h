#pragma once

#include <torch/csrc/Export.h>

#include <atomic>
#include <functional>
#include <iostream>

namespace torch {
namespace jit {

using PrintHandler = void (*)(const std::string&);

TORCH_API PrintHandler getDefaultPrintHandler();
TORCH_API PrintHandler getPrintHandler();
TORCH_API void setPrintHandler(PrintHandler ph);

} // namespace jit
} // namespace torch
