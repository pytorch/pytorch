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

#include "caffe2/utils/proto_utils.h"
#include <gtest/gtest.h>

namespace caffe2 {

TEST(ProtoUtilsTest, IsSameDevice) {
  DeviceOption a;
  DeviceOption b;
  EXPECT_TRUE(IsSameDevice(a, b));
  a.set_node_name("my_node");
  EXPECT_FALSE(IsSameDevice(a, b));
  b.set_node_name("my_node");
  EXPECT_TRUE(IsSameDevice(a, b));
  b.set_cuda_gpu_id(2);
  EXPECT_FALSE(IsSameDevice(a, b));
  a.set_cuda_gpu_id(2);
  EXPECT_TRUE(IsSameDevice(a, b));
  a.set_device_type(DeviceType::CUDA);
  b.set_device_type(DeviceType::CPU);
  EXPECT_FALSE(IsSameDevice(a, b));
}

TEST(ProtoUtilsTest, SimpleReadWrite) {
  string content("The quick brown fox jumps over the lazy dog.");
  string name = std::tmpnam(nullptr);
  EXPECT_TRUE(WriteStringToFile(content, name.c_str()));
  string read_back;
  EXPECT_TRUE(ReadStringFromFile(name.c_str(), &read_back));
  EXPECT_EQ(content, read_back);
}

}  // namespace caffe2
