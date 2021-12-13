#include <torch/csrc/jit/passes/dbr_quantization.h>

#include <torch/csrc/jit/jit_log.h>


namespace torch {
namespace jit {

namespace {
}

Module DBRQuantization(const Module& module) {
  Module m = module.clone();
  return m;
}


} // namespace jit
} // namespace torch
