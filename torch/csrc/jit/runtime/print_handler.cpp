#include <torch/csrc/jit/runtime/print_handler.h>

#include <iostream>
#include <string>

namespace torch {
namespace jit {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<PrintHandler> print_handler([](const std::string& str) {
  std::cout << str;
});

PrintHandler getPrintHandler() {
  return print_handler.load();
}

void setPrintHandler(PrintHandler ph) {
  print_handler.store(ph);
}

} // namespace jit
} // namespace torch
