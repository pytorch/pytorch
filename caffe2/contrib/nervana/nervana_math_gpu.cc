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

#include "nervana.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace math {

// Caffe2 gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
template <>
void Gemm<float, CUDAContext, NervanaEngine>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float* A,
    const float* B,
    const float beta,
    float* C,
    CUDAContext* context,
    TensorProto::DataType /*math_type*/) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  bool a_t = (TransA == CblasTrans);
  bool b_t = (TransB == CblasTrans);
  CAFFE_ENFORCE(nervana_sgemm(
      const_cast<float*>(A),
      const_cast<float*>(B),
      C,
      a_t,
      b_t,
      M,
      N,
      K,
      lda,
      ldb,
      N,
      alpha,
      beta,
      nullptr,
      false,
      false,
      context->cuda_stream()));
}

}  // namespace math
}  // namespace caffe2
