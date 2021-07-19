#pragma once
#include <torch/csrc/jit/mobile/function.h>

namespace torch {
namespace jit {
namespace mobile {
using c10::IValue;
void parseInstructions(const std::string& function_name,
                       const IValue& codeTable,
                       const IValue& debug_handles_element,
                       std::unique_ptr<mobile::Function>& function);

void parseConstants(
    const IValue& codeTable,
    std::unique_ptr<mobile::Function>& function);

void parseTypes(
    const IValue& codeTable,
    std::unique_ptr<mobile::Function>& function);

void parseRegisterSize(
    const IValue& codeTable,
    std::unique_ptr<mobile::Function>& function);
} // namespace mobile
} // namespace jit
} // namespace torch
