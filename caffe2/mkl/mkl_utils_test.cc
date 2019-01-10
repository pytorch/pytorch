#include "caffe2/core/blob.h"
#include "caffe2/core/context.h"
#include "caffe2/core/tensor.h"
#include "caffe2/mkl/mkl_utils.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/math.h"

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

TEST(MKLDNNTest, MKLMemoryCopyTest) {
  // Test copy with valid and empty shapes.
  // MKL calls fail at different points for dims {0} and dims {0,N} despite
  // the buffer size being empty for both - former in dnnAllocateBuffer and
  // the latter in dnnConversionExecute (likely due to some difference in
  // layout?). Test both cases.
  vector<vector<TIndex>> dims_list{{10, 3, 20, 20}, {0}, {0, 10}};
  for (const auto& dims : dims_list) {
    auto X_cpu_in = caffe2::make_unique<TensorCPU>(dims);
    CPUContext ctx;
    math::RandUniform<float, CPUContext>(
        X_cpu_in->size(),
        -1.0,
        1.0,
        X_cpu_in->template mutable_data<float>(),
        &ctx);

    // CPU -> MKL1
    auto X_mkl1 = caffe2::make_unique<MKLMemory<float>>(dims);
    X_mkl1->CopyFrom(*X_cpu_in);

    // MK1 -> MKL2
    auto X_mkl2 = caffe2::make_unique<MKLMemory<float>>(dims);
    X_mkl2->CopyFrom(*X_mkl1);

    // MKL1 <- MKL2
    X_mkl1 = caffe2::make_unique<MKLMemory<float>>();
    X_mkl2->CopyTo(X_mkl1.get());
    EXPECT_EQ(X_mkl1->dims(), dims);
    EXPECT_EQ(X_mkl1->size(), X_cpu_in->size());

    // CPU <- MKL1
    auto X_cpu_out = caffe2::make_unique<TensorCPU>();
    X_mkl1->CopyTo(X_cpu_out.get());
    EXPECT_EQ(X_cpu_out->dims(), dims);
    EXPECT_EQ(X_cpu_out->size(), X_cpu_in->size());
    for (int i = 0; i < X_cpu_in->size(); ++i) {
      EXPECT_NEAR(
          X_cpu_in->data<float>()[i], X_cpu_out->data<float>()[i], 1e-5);
    }
  }
}

} // namespace mkl
} // namespace caffe2

#endif // CAFFE2_HAS_MKL_DNN
