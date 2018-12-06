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

using std::map;
using std::shared_ptr;
using std::string;
using std::vector;

template <typename ContextType, typename TensorType>
void writeTextOutput(
    TensorType* tensor,
    const string& output_prefix,
    const string& name) {
  string filename = name;
  std::replace(filename.begin(), filename.end(), '/', '_');
  string output_name = output_prefix + "/" + filename + ".txt";
  caffe2::TensorSerializer ser;
  caffe2::BlobProto blob_proto;

  ser.Serialize(
      *tensor, output_name, blob_proto.mutable_tensor(), 0, tensor->numel());
  blob_proto.set_name(output_name);
  blob_proto.set_type("Tensor");
  CAFFE_ENFORCE(blob_proto.has_tensor());
  caffe2::TensorProto tensor_proto = blob_proto.tensor();
  int dims_size = tensor_proto.dims_size();
  // For NCHW or NHWC, print one line per CHW/HWC.
  // If the output is one dimension, it means N==1,
  // print everything to one line.
  int loop_count = dims_size > 1 ? tensor_proto.dims(0) : 1;
  long long elem_dim_size =
      dims_size > 1 ? tensor_proto.dims(1) : tensor_proto.dims(0);
  for (int i = 2; i < dims_size; i++) {
    elem_dim_size *= tensor_proto.dims(i);
  }
  std::vector<std::string> lines;
  for (int i = 0; i < loop_count; i++) {
    int start_idx = i * elem_dim_size;
    std::stringstream line;
    if (tensor_proto.data_type() == caffe2::TensorProto::FLOAT) {
      auto start = tensor_proto.float_data().begin() + start_idx;
      auto end = start + elem_dim_size;
      copy(start, end, std::ostream_iterator<float>(line, ","));
    } else if (tensor_proto.data_type() == caffe2::TensorProto::INT32) {
      auto start = tensor_proto.int32_data().begin() + start_idx;
      auto end = start + elem_dim_size;
      copy(start, end, std::ostream_iterator<int>(line, ","));
    } else {
      CAFFE_THROW("Unimplemented Blob type.");
    }
    // remove the last ,
    string str = line.str();
    str.pop_back();
    lines.push_back(str);
  }

  std::ofstream output_file(output_name);
  std::ostream_iterator<std::string> output_iterator(output_file, "\n");
  std::copy(lines.begin(), lines.end(), output_iterator);
}

void observerConfig();
bool backendCudaSet(const string&);
void setDeviceType(caffe2::NetDef*, caffe2::DeviceType&);
void setOperatorEngine(caffe2::NetDef*, const string&);
void loadInput(
    shared_ptr<caffe2::Workspace>,
    const bool,
    map<string, caffe2::TensorProtos>&,
    const string&,
    const string&,
    const string&,
    const string&);
void fillInputBlob(
    shared_ptr<caffe2::Workspace>,
    map<string, caffe2::TensorProtos>&,
    int iteration);
void writeOutput(
    shared_ptr<caffe2::Workspace>,
    const bool,
    const string&,
    const string&,
    const bool);
void runNetwork(
    shared_ptr<caffe2::Workspace>,
    caffe2::NetDef&,
    map<string, caffe2::TensorProtos>&,
    const bool,
    const bool,
    const int,
    const int,
    const int,
    const int,
    const int);
int benchmark(
    int argc,
    char* argv[],
    const string& FLAGS_backend,
    const string& FLAGS_init_net,
    const string& FLAGS_input,
    const string& FLAGS_input_dims,
    const string& FLAGS_input_file,
    const string& FLAGS_input_type,
    int FLAGS_iter,
    const string& FLAGS_net,
    const string& FLAGS_output,
    const string& FLAGS_output_folder,
    bool FLAGS_run_individual,
    int FLAGS_sleep_before_run,
    int FLAGS_sleep_between_iteration,
    int FLAGS_sleep_between_net_and_operator,
    bool FLAGS_text_output,
    int FLAGS_warmup,
    bool FLAGS_wipe_cache);
