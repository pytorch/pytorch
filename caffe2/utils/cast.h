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

#pragma once

#include <caffe2/utils/proto_utils.h>

namespace caffe2 {

namespace cast {

inline TensorProto_DataType GetCastDataType(const ArgumentHelper& helper, std::string arg) {
  TensorProto_DataType to;
  if (helper.HasSingleArgumentOfType<string>(arg)) {
#ifndef CAFFE2_USE_LITE_PROTO
    string s = helper.GetSingleArgument<string>(arg, "float");
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    CAFFE_ENFORCE(TensorProto_DataType_Parse(s, &to), "Unknown 'to' argument: ", s);
#else
    CAFFE_THROW("String cast op not supported");
#endif
  } else {
    to = static_cast<TensorProto_DataType>(
        helper.GetSingleArgument<int>(arg, TensorProto_DataType_FLOAT));
  }
  return to;
}

};  // namespace cast

};  // namespace caffe2
