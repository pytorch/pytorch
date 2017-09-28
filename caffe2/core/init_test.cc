/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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


