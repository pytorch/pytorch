#include <gtest/gtest.h>

#include <test/cpp/tensorexpr/test_base.h>

#include "torch/csrc/jit/tensorexpr/cpp_codegen.h"
#include "torch/csrc/jit/tensorexpr/mem_arena.h"
#include "torch/csrc/jit/tensorexpr/stmt.h"

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

TEST(CppPrinter, AllocateOnStackThenFree) {
  constexpr int dim0 = 2, dim1 = 3;
  KernelScope kernel_scope;
  VarHandle var("x", kHandle);
  Allocate* alloc = Allocate::make(var, kInt, {dim0, dim1});
  Free* free = Free::make(var);

  std::stringstream ss;
  CppPrinter printer(&ss);
  printer.visit(alloc);
  printer.visit(free);
  ASSERT_EQ(ss.str(), "int x[6];\n");
}

TEST(CppPrinter, AllocateOnHeapThenFree) {
  constexpr int dim0 = 20, dim1 = 50, dim2 = 3;
  KernelScope kernel_scope;
  VarHandle var("y", kHandle);
  Allocate* alloc = Allocate::make(var, kLong, {dim0, dim1, dim2});
  Free* free = Free::make(var);

  std::stringstream ss;
  CppPrinter printer(&ss);
  printer.visit(alloc);
  printer.visit(free);
  // size(long) = 8;
  // dim0 * dim1 * dim2 * size(long) = 24000.
  ASSERT_EQ(
      ss.str(),
      "int64_t* y = static_cast<int64_t*>(malloc(24000));\nfree(y);\n");
}

} // namespace jit
} // namespace torch
