#include <gtest/gtest.h>

#include <test/cpp/tensorexpr/test_base.h>

#include <torch/csrc/jit/tensorexpr/cpp_codegen.h>
#include <torch/csrc/jit/tensorexpr/mem_arena.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>
#include <torch/csrc/jit/testing/file_check.h>

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

TEST(CppPrinter, AllocateOnStackThenFree) {
  KernelScope kernel_scope;
  std::vector<const Expr*> dims = {new IntImm(2), new IntImm(3)};
  const Buf* buf = new Buf("x", dims, kInt);
  Allocate* alloc = new Allocate(buf);
  Free* free = new Free(buf);
  Block* block = Block::make({alloc, free});

  std::stringstream ss;
  CppPrinter printer(&ss);
  printer.visit(block);
  const std::string expected = R"(
    # CHECK: {
    # CHECK:   int x[6];
    # CHECK: }
  )";
  torch::jit::testing::FileCheck().run(expected, ss.str());
}

TEST(CppPrinter, AllocateOnHeapThenFree) {
  KernelScope kernel_scope;
  std::vector<const Expr*> dims = {
      new IntImm(20), new IntImm(50), new IntImm(3)};
  const Buf* buf = new Buf("y", dims, kLong);
  Allocate* alloc = new Allocate(buf);
  Free* free = new Free(buf);
  Block* block = Block::make({alloc, free});

  std::stringstream ss;
  CppPrinter printer(&ss);
  printer.visit(block);
  // size(long) = 8;
  // dim0 * dim1 * dim2 * size(long) = 24000.
  const std::string expected = R"(
    # CHECK: {
    # CHECK:   int64_t* y = static_cast<int64_t*>(malloc(24000));
    # CHECK:   free(y);
    # CHECK: }
  )";
  torch::jit::testing::FileCheck().run(expected, ss.str());
}

} // namespace jit
} // namespace torch
