#include <cuda.h>
#include <cuda_runtime.h>

#include <sstream>
#include <vector>

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/init.h"
#include "glog/logging.h"

using std::vector;

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);

  int gpu_count;
  CUDA_CHECK(cudaGetDeviceCount(&gpu_count));
  for (int i = 0; i < gpu_count; ++i) {
    LOG(INFO) << "Querying device ID = " << i;
    caffe2::DeviceQuery(i);
  }

  vector<vector<bool> > access_pattern;
  CHECK(caffe2::GetCudaPeerAccessPattern(&access_pattern));

  std::stringstream sstream;
  // Find topology
  int can_access;
  for (int i = 0; i < gpu_count; ++i) {
    for (int j = 0; j < gpu_count; ++j) {
      sstream << (access_pattern[i][j] ? "+" : "-") << " ";
    }
    sstream << std::endl;
  }
  LOG(INFO) << "Access pattern: " << std::endl << sstream.str();
}
