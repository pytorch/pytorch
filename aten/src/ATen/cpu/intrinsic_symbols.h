#pragma once

#include <c10/util/ArrayRef.h>

namespace at {

struct SymbolAddress {
  const char* symbol;
  void* address;

  template<typename Ptr>
  SymbolAddress(const char* sym, Ptr addr)
      : symbol(sym), address(reinterpret_cast<void*>(addr))
    {}
};

c10::ArrayRef<SymbolAddress> getIntrinsicSymbols();

} // namespace at
