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
#include "c10/util/string_utils.h"

using std::map;
using std::shared_ptr;
using std::string;
using std::vector;

template <typename ContextType, typename TensorType>
void writeTextOutput(
    TensorType* tensor,
    const string& output_prefix,
    const string& name,
    int index,
    int num_blobs) {
  if (index >= num_blobs) {
    return;
  }
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
  long long elem_dim_size =
      dims_size > 1 ? tensor_proto.dims(1) : tensor_proto.dims(0);
  for (int i = 2; i < dims_size; i++) {
    elem_dim_size *= tensor_proto.dims(i);
  }
  std::vector<std::string> lines;
  std::string dims;
  for (int i = 0; i < dims_size; i++) {
    int dim = tensor_proto.dims(i);
    if (i > 0) {
      dims += ", ";
    }
    dims += c10::to_string(dim);
  }
  lines.push_back(dims);
  std::stringstream line;
  if (tensor_proto.data_type() == caffe2::TensorProto::FLOAT) {
    auto start = tensor_proto.float_data().begin();
    auto end = tensor_proto.float_data().end();
    copy(start, end, std::ostream_iterator<float>(line, ","));
  } else if (tensor_proto.data_type() == caffe2::TensorProto::INT32) {
    auto start = tensor_proto.int32_data().begin();
    auto end = tensor_proto.int32_data().end();
    copy(start, end, std::ostream_iterator<int>(line, ","));
  } else {
    CAFFE_THROW("Unimplemented Blob type.");
  }
  // remove the last ,
  string str = line.str();
  if(str.length() != 0) {
    str.pop_back();
  }
  lines.push_back(str);

  // static casts are workaround for MSVC build
  auto flags = static_cast<std::ios_base::openmode>(std::ios::out);
  if (index != 0) {
    flags |= static_cast<std::ios_base::openmode>(std::ios::app);
  } else {
    flags |= static_cast<std::ios_base::openmode>(std::ios::trunc);
  }
  std::ofstream output_file(output_name, flags);
  std::ostream_iterator<std::string> output_iterator(output_file, "\n");
  std::copy(lines.begin(), lines.end(), output_iterator);
}

void observerConfig();
bool backendCudaSet(const string&);
void setDeviceType(caffe2::NetDef*, caffe2::DeviceType&);
void setOperatorEngine(caffe2::NetDef*, const string&);
int loadInput(
    shared_ptr<caffe2::Workspace> workspace,
    const bool run_on_gpu,
    map<string, caffe2::TensorProtos>& tensor_protos_map,
    const string& input,
    const string& input_file,
    const string& input_dims,
    const string& input_type);
void fillInputBlob(
    shared_ptr<caffe2::Workspace> workspace,
    map<string, caffe2::TensorProtos>& tensor_protos_map,
    int iteration);
void writeOutput(
    shared_ptr<caffe2::Workspace> workspace,
    const bool run_on_gpu,
    const string& output,
    const string& output_folder,
    const bool text_output,
    const int index,
    const int num_blobs);
void logBenchmarkResult(
    const std::string& type,
    const std::string& metric,
    const std::string& unit,
    const int value);
long getVirtualMemoryIfOptionEnabled(bool FLAGS_measure_memory);
void runNetwork(
    shared_ptr<caffe2::Workspace> workspace,
    caffe2::NetBase* net,
    map<string, caffe2::TensorProtos>& tensor_protos_map,
    const bool wipe_cache,
    const bool run_individual,
    const bool run_on_gpu,
    const bool text_output,
    const int warmup,
    const int iter,
    const int num_blobs,
    const int sleep_before_run,
    const int sleep_between_iteration,
    const int sleep_between_net_and_operator,
    const std::string& output,
    const std::string& output_folder);
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
    bool FLAGS_measure_memory,
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
