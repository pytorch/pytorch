#include <ctime>
#include <fstream>

#include "caffe2/core/client.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_string(client_file, "", "The given path to the client protobuffer.");
DEFINE_string(output_file, "", "The output file.");
DEFINE_int32(input_size, 0, "The input size.");
DEFINE_int32(iter, 0, "The number of iterations for timing.");
DEFINE_string(input_file, "",
              "The input file containing a list of float numbers.");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::SetUsageMessage("Runs a given client.");
  google::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "Loading client file: " << FLAGS_client_file;
  caffe2::Client client(FLAGS_client_file);
  std::vector<float> input;
  if (FLAGS_input_file.size()) {
    std::ifstream infile;
    infile.open(FLAGS_input_file, std::ios::in);
    float value;
    while (infile >> value) {
      input.push_back(value);
    }
  } else {
    input.resize(FLAGS_input_size);
  }
  LOG(INFO) << "An input of " << input.size() << " values.";
  std::vector<float> output;
  CHECK(client.Run(input, &output));
  clock_t start = clock();
  for (int i = 0; i < FLAGS_iter; ++i) {
    CHECK(client.Run(input, &output));
  }
  LOG(INFO) << "Timing: "<< FLAGS_iter << " iters took "
            << static_cast<float>(clock() - start) / CLOCKS_PER_SEC
            << " seconds.";
  LOG(INFO) << "Output: " << output.size() << " dims.";
  if (FLAGS_output_file.size()) {
    std::ofstream outfile;
    outfile.open(FLAGS_output_file, std::ios::out | std::ios::trunc);
    for (int i = 0; i < output.size(); ++i) {
      outfile << output[i] << std::endl;
    }
    outfile.close();
  }
  // This is to allow us to use memory leak checks.
  google::ShutDownCommandLineFlags();
  return 0;
}
