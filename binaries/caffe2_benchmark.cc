#include <fstream>
#include <iterator>
#include <string>

#include "binaries/benchmark_helper.h"

using std::make_shared;
using std::string;
using std::vector;

CAFFE2_DEFINE_string(
    backend,
    "builtin",
    "The backend to use when running the model. The allowed "
    "backend choices are: builtin, default, nnpack, eigen, mkl, cuda");

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

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);

  observerConfig();
  caffe2::ShowLogInfoToStderr();

  auto workspace = make_shared<caffe2::Workspace>(new caffe2::Workspace());
  bool run_on_gpu = backendCudaSet(caffe2::FLAGS_backend);

  // support other device type in the future?
  caffe2::DeviceType run_dev = run_on_gpu ? caffe2::CUDA : caffe2::CPU;

  // Run initialization network.
  caffe2::NetDef init_net_def;
  CAFFE_ENFORCE(ReadProtoFromFile(caffe2::FLAGS_init_net, &init_net_def));
  setDeviceType(&init_net_def, run_dev);
  setOperatorEngine(&init_net_def, caffe2::FLAGS_backend);
  CAFFE_ENFORCE(workspace->RunNetOnce(init_net_def));

  // Run main network.
  caffe2::NetDef net_def;
  CAFFE_ENFORCE(ReadProtoFromFile(caffe2::FLAGS_net, &net_def));
  setDeviceType(&net_def, run_dev);
  setOperatorEngine(&net_def, caffe2::FLAGS_backend);

  loadInput(
      workspace,
      run_on_gpu,
      caffe2::FLAGS_input,
      caffe2::FLAGS_input_file,
      caffe2::FLAGS_input_dims,
      caffe2::FLAGS_input_type);

  runNetwork(
      workspace,
      net_def,
      caffe2::FLAGS_run_individual,
      caffe2::FLAGS_warmup,
      caffe2::FLAGS_iter);

  writeOutput(
      workspace,
      run_on_gpu,
      caffe2::FLAGS_output,
      caffe2::FLAGS_output_folder,
      caffe2::FLAGS_text_output);

  return 0;
}
