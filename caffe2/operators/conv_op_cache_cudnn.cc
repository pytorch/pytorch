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

#include "caffe2/operators/conv_op_cache_cudnn.h"

#include <cudnn.h>

#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

template class AlgorithmsCache<cudnnConvolutionFwdAlgo_t>;
template class AlgorithmsCache<cudnnConvolutionBwdFilterAlgo_t>;
template class AlgorithmsCache<cudnnConvolutionBwdDataAlgo_t>;
template class AlgorithmsCache<int>; // For testing.
} // namespace caffe2
