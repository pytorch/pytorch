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

#include <string>

#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#ifdef CAFFE2_OPTIMIZER
#include "caffe2/opt/optimizer.h"
#endif
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/utils/string_utils.h"

C10_DEFINE_string(net, "", "The given net to benchmark.");
C10_DEFINE_string(init_net, "", "The given net to initialize any parameters.");
C10_DEFINE_string(
    input,
    "",
    "Input that is needed for running the network. If "
    "multiple input needed, use comma separated string.");
C10_DEFINE_string(
    input_file,
    "",
    "Input file that contain the serialized protobuf for "
    "the input blobs. If multiple input needed, use comma "
    "separated string. Must have the same number of items "
    "as input does.");
C10_DEFINE_string(
    input_dims,
    "",
    "Alternate to input_files, if all inputs are simple "
    "float TensorCPUs, specify the dimension using comma "
    "separated numbers. If multiple input needed, use "
    "semicolon to separate the dimension of different "
    "tensors.");
C10_DEFINE_string(input_type, "", "Input type (uint8_t/float)");
C10_DEFINE_string(
    output,
    "",
    "Output that should be dumped after the execution "
    "finishes. If multiple outputs are needed, use comma "
    "separated string. If you want to dump everything, pass "
    "'*' as the output value.");
C10_DEFINE_string(
    output_folder,
    "",
    "The folder that the output should be written to. This "
    "folder must already exist in the file system.");
C10_DEFINE_int(warmup, 0, "The number of iterations to warm up.");
C10_DEFINE_int(iter, 10, "The number of iterations to run.");
C10_DEFINE_int(opt, 0, "The level of optimization to run automatically.");
C10_DEFINE_bool(
    run_individual,
    false,
    "Whether to benchmark individual operators.");

C10_DEFINE_bool(force_engine, false, "Force engine field for all operators");
C10_DEFINE_string(engine, "", "Forced engine field value");
C10_DEFINE_bool(force_algo, false, "Force algo arg for all operators");
C10_DEFINE_string(algo, "", "Forced algo arg value");

using std::string;
using std::unique_ptr;
using std::vector;

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  unique_ptr<caffe2::Workspace> workspace(new caffe2::Workspace());

  // Run initialization network.
  caffe2::NetDef net_def;
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_init_net, &net_def));
  CAFFE_ENFORCE(workspace->RunNetOnce(net_def));

  // Load input.
  if (FLAGS_input.size()) {
    vector<string> input_names = caffe2::split(',', FLAGS_input);
    if (FLAGS_input_file.size()) {
      vector<string> input_files = caffe2::split(',', FLAGS_input_file);
      CAFFE_ENFORCE_EQ(
          input_names.size(),
          input_files.size(),
          "Input name and file should have the same number.");
      for (int i = 0; i < input_names.size(); ++i) {
        caffe2::BlobProto blob_proto;
        CAFFE_ENFORCE(caffe2::ReadProtoFromFile(input_files[i], &blob_proto));
        DeserializeBlob(blob_proto, workspace->CreateBlob(input_names[i]));
      }
    } else if (FLAGS_input_dims.size() || FLAGS_input_type.size()) {
      CAFFE_ENFORCE_GE(
          FLAGS_input_dims.size(),
          0,
          "Input dims must be specified when input tensors are used.");
      CAFFE_ENFORCE_GE(
          FLAGS_input_type.size(),
          0,
          "Input type must be specified when input tensors are used.");

      vector<string> input_dims_list = caffe2::split(';', FLAGS_input_dims);
      CAFFE_ENFORCE_EQ(
          input_names.size(),
          input_dims_list.size(),
          "Input name and dims should have the same number of items.");
      vector<string> input_type_list = caffe2::split(';', FLAGS_input_type);
      CAFFE_ENFORCE_EQ(
          input_names.size(),
          input_type_list.size(),
          "Input name and type should have the same number of items.");
      for (size_t i = 0; i < input_names.size(); ++i) {
        vector<string> input_dims_str = caffe2::split(',', input_dims_list[i]);
        vector<int> input_dims;
        for (const string& s : input_dims_str) {
          input_dims.push_back(c10::stoi(s));
        }
        caffe2::Blob* blob = workspace->GetBlob(input_names[i]);
        if (blob == nullptr) {
          blob = workspace->CreateBlob(input_names[i]);
        }
        if (input_type_list[i] == "uint8_t") {
          caffe2::int8::Int8TensorCPU* tensor =
              blob->GetMutable<caffe2::int8::Int8TensorCPU>();
          TORCH_CHECK_NOTNULL(tensor);
          tensor->t.Resize(input_dims);
          tensor->t.mutable_data<uint8_t>();
        } else if (input_type_list[i] == "float") {
          caffe2::TensorCPU* tensor = BlobGetMutableTensor(blob, caffe2::CPU);
          TORCH_CHECK_NOTNULL(tensor);
          tensor->Resize(input_dims);
          tensor->mutable_data<float>();
        } else {
          CAFFE_THROW("Unsupported input type: ", input_type_list[i]);
        }
      }
    } else {
      CAFFE_THROW(
          "You requested input tensors, but neither input_file nor "
          "input_dims is set.");
    }
  }

  // Run main network.
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_net, &net_def));
  if (!net_def.has_name()) {
    net_def.set_name("benchmark");
  }
  // force changing engine and algo
  if (FLAGS_force_engine) {
    LOG(INFO) << "force engine be: " << FLAGS_engine;
    for (const auto& op : net_def.op()) {
      const_cast<caffe2::OperatorDef*>(&op)->set_engine(FLAGS_engine);
    }
  }
  if (FLAGS_force_algo) {
    LOG(INFO) << "force algo be: " << FLAGS_algo;
    for (const auto& op : net_def.op()) {
      caffe2::GetMutableArgument(
          "algo", true, const_cast<caffe2::OperatorDef*>(&op))
          ->set_s(FLAGS_algo);
    }
  }
  if (FLAGS_opt) {
#ifdef CAFFE2_OPTIMIZER
    net_def = caffe2::opt::optimize(net_def, workspace.get(), FLAGS_opt);
#else
    LOG(WARNING) << "Caffe2 not compiled with optimization passes.";
#endif
  }

  caffe2::NetBase* net = workspace->CreateNet(net_def);
  TORCH_CHECK_NOTNULL(net);
  CAFFE_ENFORCE(net->Run());
  net->TEST_Benchmark(FLAGS_warmup, FLAGS_iter, FLAGS_run_individual);

  string output_prefix =
      FLAGS_output_folder.size() ? FLAGS_output_folder + "/" : "";
  if (FLAGS_output.size()) {
    vector<string> output_names = caffe2::split(',', FLAGS_output);
    if (FLAGS_output == "*") {
      output_names = workspace->Blobs();
    }
    for (const string& name : output_names) {
      CAFFE_ENFORCE(
          workspace->HasBlob(name),
          "You requested a non-existing blob: ",
          name);
      string serialized = SerializeBlob(*workspace->GetBlob(name), name);
      string output_filename = output_prefix + name;
      caffe2::WriteStringToFile(serialized, output_filename.c_str());
    }
  }

  return 0;
}
