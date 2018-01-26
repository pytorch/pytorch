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

#include "caffe2/utils/smart_tensor_printer.h"

#include "caffe2/core/common.h"

#include <gtest/gtest.h>

namespace caffe2 {

template <typename T>
std::string my_to_string(const T& value) {
  return to_string(value);
}

template <>
std::string my_to_string<std::string>(const std::string& value) {
  return value;
}

template <typename T>
void expect_stderr_contains(const std::vector<T>& values) {
  auto stderr = testing::internal::GetCapturedStderr();
  for (const auto& value : values) {
    std::string stringValue = my_to_string(value);
    EXPECT_TRUE(stderr.find(stringValue) != std::string::npos);
  }
}

template <typename T>
void printTensorAndCheck(const std::vector<T>& values) {
  testing::internal::CaptureStderr();
  CPUContext cpuContext;

  Tensor<CPUContext> tensor(
      std::vector<TIndex>{static_cast<TIndex>(values.size())},
	  values, &cpuContext);

  SmartTensorPrinter::PrintTensor(tensor);
  expect_stderr_contains(values);
}

#if !(__APPLE__) // TODO(janusz): thread_local does not work under mac.

TEST(SmartTensorPrinterTest, SimpleTest) {
  printTensorAndCheck(std::vector<int>{1, 2, 3, 4, 5});
  printTensorAndCheck(std::vector<std::string>{"bob", "alice", "facebook"});
}

#endif // !(__APPLE__)

} // namespace caffe2
