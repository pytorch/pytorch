#include <iostream>
#include <memory>

#include "caffe2/core/init.h"
#include <gtest/gtest.h>

namespace caffe2 {
namespace {
bool gTestInitFunctionHasBeenRun = false;

bool TestInitFunction(int*, char***) {
  gTestInitFunctionHasBeenRun = true;
  return true;
}
REGISTER_CAFFE2_INIT_FUNCTION(TestInitFunction,
                              &TestInitFunction,
                              "Just a test to see if GlobalInit invokes "
                              "registered functions correctly.");

int dummy_argc = 1;
const char* dummy_name = "foo";
char** dummy_argv = const_cast<char**>(&dummy_name);
}  // namespace

TEST(InitTest, TestInitFunctionHasRun) {
  caffe2::GlobalInit(&dummy_argc, &dummy_argv);
  EXPECT_TRUE(gTestInitFunctionHasBeenRun);
}

TEST(InitTest, CanRerunGlobalInit) {
  caffe2::GlobalInit(&dummy_argc, &dummy_argv);
  EXPECT_TRUE(caffe2::GlobalInit(&dummy_argc, &dummy_argv));
}

}  // namespace caffe2


