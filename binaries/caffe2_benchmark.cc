#include <fstream>
#include <iterator>
#include <string>

#include "binaries/benchmark_helper.h"

using std::make_shared;
using std::map;
using std::string;
using std::vector;

C10_DEFINE_string(
    backend,
    "builtin",
    "The backend to use when running the model. The allowed "
    "backend choices are: builtin, default, nnpack, eigen, mkl, cuda");

C10_DEFINE_string(init_net, "", "The given net to initialize any parameters.");
C10_DEFINE_string(
    input,
    "",
    "Input that is needed for running the network. If "
    "multiple input needed, use comma separated string.");
C10_DEFINE_string(
    input_dims,
    "",
    "Alternate to input_files, if all inputs are simple "
    "float TensorCPUs, specify the dimension using comma "
    "separated numbers. If multiple input needed, use "
    "semicolon to separate the dimension of different "
    "tensors.");
C10_DEFINE_string(
    input_file,
    "",
    "Input file that contain the serialized protobuf for "
    "the input blobs. If multiple input needed, use comma "
    "separated string. Must have the same number of items "
    "as input does.");
C10_DEFINE_string(
    input_type,
    "float",
    "Input type when specifying the input dimension."
    "The supported types are float, uint8_t.");
C10_DEFINE_int(iter, 10, "The number of iterations to run.");
C10_DEFINE_string(net, "", "The given net to benchmark.");
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
C10_DEFINE_bool(
    run_individual,
    false,
    "Whether to benchmark individual operators.");
C10_DEFINE_int(
    sleep_before_run,
    0,
    "The seconds to sleep before starting the benchmarking.");
C10_DEFINE_bool(
    text_output,
    false,
    "Whether to write out output in text format for regression purpose.");
C10_DEFINE_int(warmup, 0, "The number of iterations to warm up.");
C10_DEFINE_bool(
    wipe_cache,
    false,
    "Whether to evict the cache before running network.");

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  benchmark(
      argc,
      argv,
      c10::FLAGS_backend,
      c10::FLAGS_init_net,
      c10::FLAGS_input,
      c10::FLAGS_input_dims,
      c10::FLAGS_input_file,
      c10::FLAGS_input_type,
      c10::FLAGS_iter,
      c10::FLAGS_net,
      c10::FLAGS_output,
      c10::FLAGS_output_folder,
      c10::FLAGS_run_individual,
      c10::FLAGS_sleep_before_run,
      c10::FLAGS_text_output,
      c10::FLAGS_warmup,
      c10::FLAGS_wipe_cache);
}
