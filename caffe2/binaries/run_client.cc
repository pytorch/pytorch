#include <ctime>
#include <fstream>

#include "caffe2/core/client.h"
#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"

CAFFE2_DEFINE_string(client_file, "", "The given path to the client protobuffer.");
CAFFE2_DEFINE_string(output_file, "", "The output file.");
CAFFE2_DEFINE_int(input_size, 0, "The input size.");
CAFFE2_DEFINE_int(iter, 0, "The number of iterations for timing.");
CAFFE2_DEFINE_string(input_file, "",
              "The input file containing a list of float numbers.");

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, argv);
  CAFFE_LOG_INFO << "Loading client file: " << caffe2::FLAGS_client_file;
  caffe2::Client client(caffe2::FLAGS_client_file);
  std::vector<float> input;
  if (caffe2::FLAGS_input_file.size()) {
    std::ifstream infile;
    infile.open(caffe2::FLAGS_input_file, std::ios::in);
    float value;
    while (infile >> value) {
      input.push_back(value);
    }
  } else {
    input.resize(caffe2::FLAGS_input_size);
  }
  CAFFE_LOG_INFO << "An input of " << input.size() << " values.";
  std::vector<float> output;
  CAFFE_CHECK(client.Run(input, &output));
  clock_t start = clock();
  for (int i = 0; i < caffe2::FLAGS_iter; ++i) {
    CAFFE_CHECK(client.Run(input, &output));
  }
  CAFFE_LOG_INFO << "Timing: "<< caffe2::FLAGS_iter << " iters took "
            << static_cast<float>(clock() - start) / CLOCKS_PER_SEC
            << " seconds.";
  CAFFE_LOG_INFO << "Output: " << output.size() << " dims.";
  if (caffe2::FLAGS_output_file.size()) {
    std::ofstream outfile;
    outfile.open(caffe2::FLAGS_output_file, std::ios::out | std::ios::trunc);
    for (int i = 0; i < output.size(); ++i) {
      outfile << output[i] << std::endl;
    }
    outfile.close();
  }
  return 0;
}
