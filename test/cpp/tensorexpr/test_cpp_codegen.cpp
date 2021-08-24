#include <gtest/gtest.h>

#include <test/cpp/tensorexpr/test_base.h>

#include <torch/csrc/jit/tensorexpr/cpp_codegen.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>
#include <torch/csrc/jit/testing/file_check.h>

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

TEST(CppPrinter, AllocateOnStackThenFree) {
  std::vector<ExprPtr> dims = {alloc<IntImm>(2), alloc<IntImm>(3)};
  BufPtr buf = alloc<Buf>("x", dims, kInt);
  AllocatePtr alloc_ = alloc<Allocate>(buf);
  FreePtr free_ = alloc<Free>(buf);
  BlockPtr block = Block::make({alloc_, free_});

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
  std::vector<ExprPtr> dims = {
      alloc<IntImm>(20), alloc<IntImm>(50), alloc<IntImm>(3)};
  BufPtr buf = alloc<Buf>("y", dims, kLong);
  AllocatePtr alloc_ = alloc<Allocate>(buf);
  FreePtr free_ = alloc<Free>(buf);
  BlockPtr block = Block::make({alloc_, free_});

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
