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

#include <string>

#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/string_utils.h"

using std::shared_ptr;
using std::string;
using std::vector;

template <typename ContextType, typename TensorType>
void writeTextOutput(
    TensorType* tensor,
    const string& output_prefix,
    const string& name) {
  string output_name = output_prefix + "/" + name + ".txt";
  caffe2::TensorSerializer<ContextType> ser;
  caffe2::BlobProto blob_proto;
  ser.Serialize(
      *tensor, output_name, blob_proto.mutable_tensor(), 0, tensor->size());
  blob_proto.set_name(output_name);
  blob_proto.set_type("Tensor");
  CAFFE_ENFORCE(blob_proto.has_tensor());
  caffe2::TensorProto tensor_proto = blob_proto.tensor();
  vector<float> data;
  switch (tensor_proto.data_type()) {
    case caffe2::TensorProto::FLOAT: {
      std::copy(
          tensor_proto.float_data().begin(),
          tensor_proto.float_data().end(),
          std::back_inserter(data));
      break;
    }
    case caffe2::TensorProto::INT32: {
      std::copy(
          tensor_proto.int32_data().begin(),
          tensor_proto.int32_data().end(),
          std::back_inserter(data));
      break;
    }
    default:
      CAFFE_THROW("Unimplemented Blob type.");
  }
  std::ofstream output_file(output_name);
  std::ostream_iterator<float> output_iterator(output_file, "\n");
  std::copy(data.begin(), data.end(), output_iterator);
}

void observerConfig();
bool backendCudaSet(const string&);
void setDeviceType(caffe2::NetDef*, caffe2::DeviceType&);
void setOperatorEngine(caffe2::NetDef*, const string&);
void loadInput(
    shared_ptr<caffe2::Workspace>,
    const bool,
    const string&,
    const string&,
    const string&,
    const string&);
void writeOutput(
    shared_ptr<caffe2::Workspace>,
    const bool,
    const string&,
    const string&,
    const bool);
void runNetwork(
    shared_ptr<caffe2::Workspace>,
    caffe2::NetDef&,
    const bool,
    const int,
    const int);
