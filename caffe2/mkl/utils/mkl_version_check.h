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

#ifndef CAFFE2_UTILS_MKL_MKL_VERSION_CHECK_H_
#define CAFFE2_UTILS_MKL_MKL_VERSION_CHECK_H_
#ifdef CAFFE2_USE_MKL

#include <mkl.h>

#if INTEL_MKL_VERSION >= 20170000
#define CAFFE2_HAS_MKL_SGEMM_PACK
#define CAFFE2_HAS_MKL_DNN
#endif // INTEL_MKL_VERSION >= 20170000

#endif // CAFFE2_USE_MKL
#endif // CAFFE2_UTILS_MKL_MKL_VERSION_CHECK_H_
