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

#include "caffe2/core/blob.h"
#include "caffe2/core/context.h"
#include "caffe2/core/tensor.h"
#include "caffe2/mkl/mkl_utils.h"
#include "caffe2/proto/caffe2.pb.h"

#include <gtest/gtest.h>

#ifdef CAFFE2_HAS_MKL_DNN

namespace caffe2 {
namespace mkl {

TEST(MKLDNNTest, SimpleConvolutionTest) {
  // In order to create an internal layout, let's use convolution as primitive.
  size_t dimension = 4;
  size_t bdata_sizes[4] = {32, 32, 8, 16};
  size_t bdata_offsets[4] = {1, 32, 32 * 32, 32 * 32 * 8};
  size_t tdata_sizes[4] = {30, 30, 64, 16};
  size_t fdata_sizes[4] = {3, 3, 8, 64};
  size_t strides[2] = {1, 1};
  int pads[2] = {0, 0};

  // Creating Input and output tensors
  TensorCPU X(vector<TIndex>{16, 8, 32, 32});
  TensorCPU W(vector<TIndex>{64, 8, 3, 3});
  TensorCPU b(vector<TIndex>{64});
  TensorCPU Y(vector<TIndex>{16, 64, 30, 30});

  float* data = X.mutable_data<float>();
  for (int i = 0; i < X.size(); ++i) {
    data[i] = 1;
  }
  data = W.mutable_data<float>();
  for (int i = 0; i < W.size(); ++i) {
    data[i] = 1;
  }
  data = b.mutable_data<float>();
  for (int i = 0; i < b.size(); ++i) {
    data[i] = 0.1;
  }

  PrimitiveWrapper<float> primitive(
      dnnConvolutionCreateForwardBias<float>,
      nullptr,
      dnnAlgorithmConvolutionDirect,
      dimension,
      bdata_sizes,
      tdata_sizes,
      fdata_sizes,
      strides,
      pads,
      dnnBorderZeros);

  // Test if the resource wrapper works.
  MKLMemory<float> X_wrapper(X.dims(), primitive, dnnResourceSrc);
  X_wrapper.CopyFrom(X);
  TensorCPU X_recover(X.dims());
  X_wrapper.CopyTo(&X_recover);
  const float* recover_data = X_recover.data<float>();
  for (int i = 0; i < X_recover.size(); ++i) {
    EXPECT_EQ(recover_data[i], 1);
  }

  // Create W, b and Y wrappers, and run the convolution algorithm.
  MKLMemory<float> W_wrapper(W.dims(), primitive, dnnResourceFilter);
  W_wrapper.CopyFrom(W);
  MKLMemory<float> b_wrapper(b.dims(), primitive, dnnResourceBias);
  b_wrapper.CopyFrom(b);
  MKLMemory<float> Y_wrapper(Y.dims(), primitive, dnnResourceDst);

  void* resources[dnnResourceNumber] = {
      X_wrapper.buffer(),
      Y_wrapper.buffer(),
      W_wrapper.buffer(),
      b_wrapper.buffer(),
  };

  MKLDNN_SAFE_CALL(dnnExecute<float>(primitive, resources));
  Y_wrapper.CopyTo(&Y);
  const float* out_data = Y.data<float>();
  for (int i = 0; i < Y.size(); ++i) {
    EXPECT_NEAR(out_data[i], 72.1, 1e-5);
  }
}

} // namespace mkl
} // namespace caffe2

#endif // CAFFE2_HAS_MKL_DNN
