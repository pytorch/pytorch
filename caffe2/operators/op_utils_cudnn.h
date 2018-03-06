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

#ifndef CAFFE2_OPERATORS_CUDNN_OP_UTILS_H_
#define CAFFE2_OPERATORS_CUDNN_OP_UTILS_H_

#include "caffe2/core/cudnn_wrappers.h"

namespace caffe2 {

// Earlier in the days Caffe sets the default cudnn workspace to 8MB. We bump
// it up to 64MB in Caffe2, as this enables the use of Winograd in many cases,
// something very beneficial to more recent CNN models.
static constexpr size_t kCONV_CUDNN_WORKSPACE_LIMIT_BYTES = 64 * 1024 * 1024;

// Manually specified number of algorithms implemented in CuDNN.
// This does not have any performance implications, as we will always find the
// fastest algorithm; setting them to the right number of algorithms will enable
// us to best report the statistics when doing an exhaustive search, though.
#if CUDNN_VERSION_MIN(7, 0, 0)
// Note: Double each of these due to potential
// tensorcore + non-tensorcore versions
// which are treated as separate returned algos
static constexpr size_t kNUM_CUDNN_FWD_ALGS =
    2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
static constexpr size_t kNUM_CUDNN_BWD_FILTER_ALGS =
    2 * CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
static constexpr size_t kNUM_CUDNN_BWD_DATA_ALGS =
    2 * CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
#else
static constexpr size_t kNUM_CUDNN_FWD_ALGS = 7;
static constexpr size_t kNUM_CUDNN_BWD_FILTER_ALGS = 4;
static constexpr size_t kNUM_CUDNN_BWD_DATA_ALGS = 5;
#endif

namespace {
template <typename ArrayOfcudnnConvolutionAlgoPerf_t>
inline void LogCuDNNPerfStats(
    const ArrayOfcudnnConvolutionAlgoPerf_t& perf_stat,
    int returned_algo_count) {
  VLOG(1) << "Perf result: (algo: stat, time, memory)";
  for (int i = 0; i < returned_algo_count; ++i) {
    const auto& stat = perf_stat[i];
    VLOG(1) << stat.algo << ": " << stat.status << " " << stat.time << " "
            << stat.memory;
  }
}
} // namespace

// Easier indexing into force_algo_ vector,
// shared by CudnnConvTransposeOpBase and CudnnConvOpBase to force
// usage of a particular algortihm instead of searching
enum { ALGO_FWD = 0, ALGO_WGRAD = 1, ALGO_DGRAD = 2 } algoIndex_t;

} // namespace caffe2

#endif // CAFFE2_OPERATORS_CUDNN_OP_UTILS_H_
