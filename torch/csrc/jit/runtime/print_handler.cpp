
copy: fbcode/caffe2/torch/csrc/jit/runtime/print_handler.cpp
copyrev: 4277b4f4547133cf73e9fc8037accf2014e418a1

#include <torch/csrc/jit/runtime/print_handler.h>

#include <iostream>
#include <string>

namespace torch {
namespace jit {

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
