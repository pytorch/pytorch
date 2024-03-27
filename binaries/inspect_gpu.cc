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

#include <cuda_runtime.h>

#include <sstream>
#include <vector>

#include "c10/util/Flags.h"
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"

using std::vector;

C10_DECLARE_int(caffe2_log_level);

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  c10::SetUsageMessage(
      "Inspects the GPUs on the current machine and prints out their details "
      "provided by cuda.");

  int gpu_count;
  CUDA_ENFORCE(cudaGetDeviceCount(&gpu_count));
  for (int i = 0; i < gpu_count; ++i) {
    LOG(INFO) << "Querying device ID = " << i;
    caffe2::DeviceQuery(i);
  }

  vector<vector<bool> > access_pattern;
  CAFFE_ENFORCE(caffe2::GetCudaPeerAccessPattern(&access_pattern));

  std::stringstream sstream;
  // Find topology
  for (int i = 0; i < gpu_count; ++i) {
    for (int j = 0; j < gpu_count; ++j) {
      sstream << (access_pattern[i][j] ? "+" : "-") << " ";
    }
    sstream << std::endl;
  }
  LOG(INFO) << "Access pattern: " << std::endl << sstream.str();

  return 0;
}
