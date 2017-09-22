#include <string>

#include "caffe2/core/init.h"
#include "caffe2/core/operator.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/utils/string_utils.h"
#include "caffe2/core/logging.h"

CAFFE2_DEFINE_string(net, "", "The given net to benchmark.");
CAFFE2_DEFINE_string(init_net, "",
                     "The given net to initialize any parameters.");
CAFFE2_DEFINE_string(input, "",
                     "Input that is needed for running the network. If "
                     "multiple input needed, use comma separated string.");
CAFFE2_DEFINE_string(input_file, "",
                     "Input file that contain the serialized protobuf for "
                     "the input blobs. If multiple input needed, use comma "
                     "separated string. Must have the same number of items "
                     "as input does.");
CAFFE2_DEFINE_string(input_dims, "",
                     "Alternate to input_files, if all inputs are simple "
                     "float TensorCPUs, specify the dimension using comma "
                     "separated numbers. If multiple input needed, use "
                     "semicolon to separate the dimension of different "
                     "tensors.");
CAFFE2_DEFINE_string(output, "",
                     "Output that should be dumped after the execution "
                     "finishes. If multiple outputs are needed, use comma "
                     "separated string. If you want to dump everything, pass "
                     "'*' as the output value.");
CAFFE2_DEFINE_string(output_folder, "",
                     "The folder that the output should be written to. This "
                     "folder must already exist in the file system.");
CAFFE2_DEFINE_int(warmup, 0, "The number of iterations to warm up.");
CAFFE2_DEFINE_int(iter, 10, "The number of iterations to run.");
CAFFE2_DEFINE_bool(run_individual, false, "Whether to benchmark individual operators.");

using std::string;
using std::unique_ptr;
using std::vector;

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  unique_ptr<caffe2::Workspace> workspace(new caffe2::Workspace());

  // Run initialization network.
  caffe2::NetDef net_def;
  CAFFE_ENFORCE(ReadProtoFromFile(caffe2::FLAGS_init_net, &net_def));
  CAFFE_ENFORCE(workspace->RunNetOnce(net_def));

  // Load input.
  if (caffe2::FLAGS_input.size()) {
    vector<string> input_names = caffe2::split(',', caffe2::FLAGS_input);
    if (caffe2::FLAGS_input_file.size()) {
      vector<string> input_files = caffe2::split(',', caffe2::FLAGS_input_file);
      CAFFE_ENFORCE_EQ(
          input_names.size(), input_files.size(),
          "Input name and file should have the same number.");
      for (int i = 0; i < input_names.size(); ++i) {
        caffe2::BlobProto blob_proto;
        CAFFE_ENFORCE(caffe2::ReadProtoFromFile(input_files[i], &blob_proto));
        workspace->CreateBlob(input_names[i])->Deserialize(blob_proto);
      }
    } else if (caffe2::FLAGS_input_dims.size()) {
      vector<string> input_dims_list = caffe2::split(';', caffe2::FLAGS_input_dims);
      CAFFE_ENFORCE_EQ(
          input_names.size(), input_dims_list.size(),
          "Input name and dims should have the same number of items.");
      for (int i = 0; i < input_names.size(); ++i) {
        vector<string> input_dims_str = caffe2::split(',', input_dims_list[i]);
        vector<int> input_dims;
        for (const string& s : input_dims_str) {
          input_dims.push_back(caffe2::stoi(s));
        }
        caffe2::TensorCPU* tensor =
            workspace->GetBlob(input_names[i])->GetMutable<caffe2::TensorCPU>();
        tensor->Resize(input_dims);
        tensor->mutable_data<float>();
      }
    } else {
      CAFFE_THROW("You requested input tensors, but neither input_file nor "
                  "input_dims is set.");
    }
  }

  // Run main network.
  CAFFE_ENFORCE(ReadProtoFromFile(caffe2::FLAGS_net, &net_def));
  caffe2::NetBase* net = workspace->CreateNet(net_def);
  CHECK_NOTNULL(net);
  net->TEST_Benchmark(
      caffe2::FLAGS_warmup,
      caffe2::FLAGS_iter,
      caffe2::FLAGS_run_individual);

  string output_prefix = caffe2::FLAGS_output_folder.size()
      ? caffe2::FLAGS_output_folder + "/"
      : "";
  if (caffe2::FLAGS_output.size()) {
    vector<string> output_names = caffe2::split(',', caffe2::FLAGS_output);
    if (caffe2::FLAGS_output == "*") {
      output_names = workspace->Blobs();
    }
    for (const string& name : output_names) {
      CAFFE_ENFORCE(
          workspace->HasBlob(name),
          "You requested a non-existing blob: ",
          name);
      string serialized = workspace->GetBlob(name)->Serialize(name);
      string output_filename = output_prefix + name;
      caffe2::WriteStringToFile(serialized, output_filename.c_str());
    }
  }

  return 0;
}
