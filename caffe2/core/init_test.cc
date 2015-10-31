#include <iostream>
#include <memory>

#include "caffe2/core/init.h"
#include "gtest/gtest.h"
#include "caffe2/core/logging.h"

namespace caffe2 {
namespace {
bool gTestInitFunctionHasBeenRun = false;

bool TestInitFunction() {
  gTestInitFunctionHasBeenRun = true;
  return true;
}
REGISTER_CAFFE2_INIT_FUNCTION(TestInitFunction,
                              &TestInitFunction,
                              "Just a test to see if GlobalInit invokes "
                              "registered functions correctly.");
}  // namespace

TEST(InitTest, TestInitFunctionHasRun) {
  EXPECT_TRUE(gTestInitFunctionHasBeenRun);
}

TEST(InitTest, CannotRerunGlobalInit) {
  int dummy_argc = 1;
  const char* dummy_name = "foo";
  char** dummy_argv = const_cast<char**>(&dummy_name);
  // GlobalInit is called in caffe's gtest main function, so here we won't be
  // able to re-initialize it.
  EXPECT_FALSE(caffe2::GlobalInit(&dummy_argc, dummy_argv));
}

}  // namespace caffe2


