#include <cuda.h>
#include <cuda_runtime.h>

#include <sstream>
#include <vector>

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"

using std::vector;

CAFFE2_DECLARE_int(caffe2_log_level);

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::SetUsageMessage(
      "Inspects the GPUs on the current machine and prints out their details "
      "provided by cuda.");
  // Set log level to CAFFE_INFO since things will be printed there.
  caffe2::FLAGS_caffe2_log_level = CAFFE_INFO;

  int gpu_count;
  CUDA_CHECK(cudaGetDeviceCount(&gpu_count));
  for (int i = 0; i < gpu_count; ++i) {
    CAFFE_LOG_INFO << "Querying device ID = " << i;
    caffe2::DeviceQuery(i);
  }

  vector<vector<bool> > access_pattern;
  CAFFE_CHECK(caffe2::GetCudaPeerAccessPattern(&access_pattern));

  std::stringstream sstream;
  // Find topology
  for (int i = 0; i < gpu_count; ++i) {
    for (int j = 0; j < gpu_count; ++j) {
      sstream << (access_pattern[i][j] ? "+" : "-") << " ";
    }
    sstream << std::endl;
  }
  CAFFE_LOG_INFO << "Access pattern: " << std::endl << sstream.str();

  return 0;
}
