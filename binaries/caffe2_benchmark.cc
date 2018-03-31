#include <fstream>
#include <iterator>
#include <string>

#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/utils/string_utils.h"

#include "observers/observer_config.h"

CAFFE2_DEFINE_string(
    backend,
    "builtin",
    "The backend to use when running the model. The allowed "
    "backend choices are: builtin, default, nnpack, eigen, mkl");
CAFFE2_DEFINE_string(
    init_net,
    "",
    "The given net to initialize any parameters.");
CAFFE2_DEFINE_string(
    input,
    "",
    "Input that is needed for running the network. If "
    "multiple input needed, use comma separated string.");
CAFFE2_DEFINE_string(
    input_dims,
    "",
    "Alternate to input_files, if all inputs are simple "
    "float TensorCPUs, specify the dimension using comma "
    "separated numbers. If multiple input needed, use "
    "semicolon to separate the dimension of different "
    "tensors.");
CAFFE2_DEFINE_string(
    input_file,
    "",
    "Input file that contain the serialized protobuf for "
    "the input blobs. If multiple input needed, use comma "
    "separated string. Must have the same number of items "
    "as input does.");
CAFFE2_DEFINE_string(
    input_type,
    "float",
    "Input type when specifying the input dimension."
    "The supported types are float, uint8_t.");
CAFFE2_DEFINE_int(iter, 10, "The number of iterations to run.");
CAFFE2_DEFINE_string(net, "", "The given net to benchmark.");
CAFFE2_DEFINE_string(
    output,
    "",
    "Output that should be dumped after the execution "
    "finishes. If multiple outputs are needed, use comma "
    "separated string. If you want to dump everything, pass "
    "'*' as the output value.");
CAFFE2_DEFINE_string(
    output_folder,
    "",
    "The folder that the output should be written to. This "
    "folder must already exist in the file system.");
CAFFE2_DEFINE_bool(
    run_individual,
    false,
    "Whether to benchmark individual operators.");
CAFFE2_DEFINE_bool(
    text_output,
    false,
    "Whether to write out output in text format for regression purpose.");
CAFFE2_DEFINE_int(warmup, 0, "The number of iterations to warm up.");

using std::string;
using std::unique_ptr;
using std::vector;

static void writeTextOutput(
    caffe2::TensorCPU* tensor,
    const string& output_prefix,
    const string& name) {
  string output_name = output_prefix + "/" + name + ".txt";
  caffe2::TensorSerializer<caffe2::CPUContext> ser;
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

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::ShowLogInfoToStderr();
  unique_ptr<caffe2::Workspace> workspace(new caffe2::Workspace());

  // Run initialization network.
  caffe2::NetDef init_net_def;
  CAFFE_ENFORCE(ReadProtoFromFile(caffe2::FLAGS_init_net, &init_net_def));
  CAFFE_ENFORCE(workspace->RunNetOnce(init_net_def));

  // Load input.
  if (caffe2::FLAGS_input.size()) {
    vector<string> input_names = caffe2::split(',', caffe2::FLAGS_input);
    if (caffe2::FLAGS_input_file.size()) {
      vector<string> input_files = caffe2::split(',', caffe2::FLAGS_input_file);
      CAFFE_ENFORCE_EQ(
          input_names.size(),
          input_files.size(),
          "Input name and file should have the same number.");
      for (int i = 0; i < input_names.size(); ++i) {
        caffe2::BlobProto blob_proto;
        CAFFE_ENFORCE(caffe2::ReadProtoFromFile(input_files[i], &blob_proto));
        workspace->CreateBlob(input_names[i])->Deserialize(blob_proto);
      }
    } else if (caffe2::FLAGS_input_dims.size()) {
      vector<string> input_dims_list =
          caffe2::split(';', caffe2::FLAGS_input_dims);
      CAFFE_ENFORCE_EQ(
          input_names.size(),
          input_dims_list.size(),
          "Input name and dims should have the same number of items.");
      for (int i = 0; i < input_names.size(); ++i) {
        vector<string> input_dims_str = caffe2::split(',', input_dims_list[i]);
        vector<int> input_dims;
        for (const string& s : input_dims_str) {
          input_dims.push_back(caffe2::stoi(s));
        }
        if (!workspace->HasBlob(input_names[i])) {
          workspace->CreateBlob(input_names[i]);
        }
        caffe2::TensorCPU* tensor =
            workspace->GetBlob(input_names[i])->GetMutable<caffe2::TensorCPU>();
        tensor->Resize(input_dims);
        if (caffe2::FLAGS_input_type == "float") {
          tensor->mutable_data<float>();
        } else {
          CAFFE_ENFORCE(
              caffe2::FLAGS_input_type == "uint8_t",
              "Only supported input types are: float, uint8_t");
          tensor->mutable_data<uint8_t>();
        }
      }
    } else {
      CAFFE_THROW(
          "You requested input tensors, but neither input_file nor "
          "input_dims is set.");
    }
  }

  // Run main network.
  caffe2::NetDef net_def;
  CAFFE_ENFORCE(ReadProtoFromFile(caffe2::FLAGS_net, &net_def));
  if (caffe2::FLAGS_backend != "builtin") {
    std::string engine = caffe2::FLAGS_backend == "nnpack"
        ? "NNPACK"
        : caffe2::FLAGS_backend == "eigen" ? "EIGEN"
                                           : caffe2::FLAGS_backend == "mkl"
                ? "MKLDNN"
                : caffe2::FLAGS_backend == "default" ? "" : "NONE";
    CAFFE_ENFORCE(engine != "NONE", "Backend is not supported");
    for (int i = 0; i < net_def.op_size(); i++) {
      caffe2::OperatorDef* op_def = net_def.mutable_op(i);
      op_def->set_engine(engine);
    }
  }

  caffe2::NetBase* net = workspace->CreateNet(net_def);
  CHECK_NOTNULL(net);

  LOG(INFO) << "Starting benchmark.";
  caffe2::ObserverConfig::initSampleRate(
      1, 1, 1, caffe2::FLAGS_run_individual, caffe2::FLAGS_warmup);
  LOG(INFO) << "Running warmup runs.";
  for (int i = 0; i < caffe2::FLAGS_warmup; ++i) {
    CAFFE_ENFORCE(net->Run(), "Warmup run ", i, " has failed.");
  }

  LOG(INFO) << "Main runs.";
  CAFFE_ENFORCE(
      caffe2::FLAGS_iter >= 0,
      "Number of main runs should be non negative, provided ",
      caffe2::FLAGS_iter,
      ".");
  for (int i = 0; i < caffe2::FLAGS_iter; ++i) {
    caffe2::ObserverConfig::initSampleRate(1, 1, 1, 0, caffe2::FLAGS_warmup);
    CAFFE_ENFORCE(net->Run(), "Main run ", i, " has failed.");
    if (caffe2::FLAGS_run_individual) {
      caffe2::ObserverConfig::initSampleRate(1, 1, 1, 1, caffe2::FLAGS_warmup);
      CAFFE_ENFORCE(net->Run(), "Main run ", i, " with operator has failed.");
    }
  }

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
      if (caffe2::FLAGS_text_output) {
        auto blob = workspace->GetBlob(name)->GetMutable<caffe2::TensorCPU>();
        writeTextOutput(blob, output_prefix, name);
      } else {
        string serialized = workspace->GetBlob(name)->Serialize(name);
        string output_filename = output_prefix + name;
        caffe2::WriteStringToFile(serialized, output_filename.c_str());
      }
    }
  }

  return 0;
}
