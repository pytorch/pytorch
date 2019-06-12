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

#include <sstream>
#include <string>

#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/db.h"
#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/proto_utils.h"

C10_DEFINE_string(f_in, "", "The input data file name.");
C10_DEFINE_string(f_out, "", "The output data file name.");

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  std::ifstream f_in(FLAGS_f_in);
  std::ofstream f_out(FLAGS_f_out);
  std::string line;
  caffe2::TensorProtos tensor_protos;
  while (std::getline(f_in, line)) {
    caffe2::TensorProto* data = tensor_protos.add_protos();
    data->set_data_type(caffe2::TensorProto::STRING);
    data->add_dims(1);
    data->add_string_data(line);
    data->set_name("text");
  }
  f_in.close();
  std::string output_str;
  tensor_protos.SerializeToString(&output_str);
  f_out << output_str;
  f_out.close();
  return 0;
}
