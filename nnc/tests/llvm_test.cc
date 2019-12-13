#include "llvm_codegen.h"
#include "ir.h"

#include <gtest/gtest.h>

using namespace nnc;

TEST(ExprTest, IntImmTest) {
  auto a = IntImm::make(2);
  LLVMCodeGen cg;
  a.accept(&cg);
  EXPECT_EQ(cg.value(), 2);
}

TEST(ExprTest, IntAddTest) {
  auto a = IntImm::make(2);
  auto b = IntImm::make(3);
  auto c = Add::make(a, b);
  LLVMCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value(), 5);
}
