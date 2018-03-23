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

#include "caffe2/onnx/helper.h"

#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 { namespace onnx  {

size_t DummyName::counter_ = 0;

std::unordered_set<std::string>& DummyName::get_used_names() {
  static std::unordered_set<std::string> used_names;
  return used_names;
}

std::string DummyName::NewDummyName() {
  while (true) {
    const std::string name = caffe2::MakeString("OC2_DUMMY_", counter_++);
    auto ret = get_used_names().insert(name);
    if (ret.second) {
      return name;
    }
  }
}

void DummyName::Reset(const std::unordered_set<std::string> &used_names) {
  auto& names = get_used_names();
  names = used_names;
  counter_ = 0;
}

}}
