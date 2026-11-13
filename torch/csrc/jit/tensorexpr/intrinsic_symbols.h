#pragma once

#ifdef TORCH_ENABLE_LLVM
#include <c10/util/ArrayRef.h>

namespace torch {
namespace jit {
namespace tensorexpr {

struct SymbolAddress {
  const char* symbol;
  void* address;

  SymbolAddress(const char* sym, void* addr) : symbol(sym), address(addr) {}
};

c10::ArrayRef<SymbolAddress> getIntrinsicSymbols();

} // namespace tensorexpr
} // namespace jit
} // namespace torch
#endif // TORCH_ENABLE_LLVM
