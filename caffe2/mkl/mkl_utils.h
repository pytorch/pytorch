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

#ifndef CAFFE2_UTILS_MKL_UTILS_H_
#define CAFFE2_UTILS_MKL_UTILS_H_

#include "caffe2/core/macros.h"  // For caffe2 macros.

#ifdef CAFFE2_USE_MKL

#include "caffe2/mkl/utils/mkl_version_check.h"

// MKLDNN_CHECK should be used in places where exceptions should not be thrown,
// such as in destructors.
#define MKLDNN_CHECK(condition)   \
  do {                            \
    dnnError_t error = condition; \
    CAFFE_ENFORCE_EQ(             \
        error,                    \
        E_SUCCESS,                \
        "Error at : ",            \
        __FILE__,                 \
        ":",                      \
        __LINE__,                 \
        ", error number: ",       \
        error);                   \
  } while (0)

#define MKLDNN_SAFE_CALL(condition) \
  do {                              \
    dnnError_t error = condition;   \
    CAFFE_ENFORCE_EQ(               \
        error,                      \
        E_SUCCESS,                  \
        "Error at : ",              \
        __FILE__,                   \
        ":",                        \
        __LINE__,                   \
        ", error number: ",         \
        error);                     \
  } while (0)

#define CHECK_INPUT_FILTER_DIMS(X, filter, condition) \
  do {                                                \
    if (cached_input_dims_ != X.dims() ||             \
        cached_filter_dims_ != filter.dims()) {       \
      cached_input_dims_ = X.dims();                  \
      cached_filter_dims_ = filter.dims();            \
      condition = true;                               \
    } else {                                          \
      condition = false;                              \
    }                                                 \
  } while (0)

#define CHECK_INPUT_DIMS(X, condition)    \
  do {                                    \
    if (cached_input_dims_ != X.dims()) { \
      cached_input_dims_ = X.dims();      \
      condition = true;                   \
    } else {                              \
      condition = false;                  \
    }                                     \
  } while (0)

// All caffe2 mkl related headers

#ifdef CAFFE2_HAS_MKL_DNN
#include "caffe2/mkl/utils/mkl_context.h"
#include "caffe2/mkl/utils/mkl_dnn_cppwrapper.h"
#include "caffe2/mkl/utils/mkl_memory.h"
#include "caffe2/mkl/utils/mkl_operator.h"
#endif // CAFFE2_HAS_MKL_DNN

#ifdef CAFFE2_HAS_MKL_SGEMM_PACK
#include "caffe2/mkl/utils/sgemm_pack.h"
#endif // CAFFE2_HAS_MKL_SGEMM_PACK

#endif // CAFFE2_USE_MKL
#endif // CAFFE2_UTILS_MKL_UTILS_H_
