#include <iostream>
#include <memory>

#include "caffe2/core/init.h"
#include "gtest/gtest.h"
#include "glog/logging.h"

namespace caffe2 {
namespace {
bool gTestInitFunctionHasBeenRun = false;

void TestInitFunction() {
  gTestInitFunctionHasBeenRun = true;
}
REGISTER_CAFFE2_INIT_FUNCTION(TestInitFunction,
                              &TestInitFunction,
                              "Just a test to see if GlobalInit invokes "
                              "registered functions correctly.");
}  // namespace

TEST(InitTest, TestInitFunctionHasRun) {
  EXPECT_TRUE(gTestInitFunctionHasBeenRun);
}

TEST(InitDeathTest, CannotRerunGlobalInit) {
  int dummy_argc = 1;
  char* dummy_name = "foo";
  char** dummy_argv = &dummy_name;
  EXPECT_DEATH(caffe2::GlobalInit(&dummy_argc, &dummy_argv),
               "blabla");
}

}  // namespace caffe2


