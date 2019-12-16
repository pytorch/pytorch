#ifndef CAFFE2_INT8_TEST_UTILS_H_
#define CAFFE2_INT8_TEST_UTILS_H_

#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/tensor_int8.h"

#include <array>
#include <cmath>
#include <random>

#include "gtest/gtest.h"

namespace caffe2 {

// for quantized Add, the error shouldn't exceed 2 * scale
inline float addErrorTolerance(float scale) {
  return 2 * scale;
}

inline std::unique_ptr<int8::Int8TensorCPU> q(
    const std::vector<int64_t>& dims) {
  auto r = std::make_unique<int8::Int8TensorCPU>();
  r->scale = 0.01;
  r->zero_point = static_cast<int32_t>(std::numeric_limits<uint8_t>::max()) / 2;
  ReinitializeTensor(&r->t, dims, at::dtype<uint8_t>().device(CPU));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint8_t> dis;
  for (auto i = 0; i < r->t.numel(); ++i) {
    r->t.mutable_data<uint8_t>()[i] = dis(gen);
  }
  return r;
}

inline std::unique_ptr<int8::Int8TensorCPU> biasq(
    const std::vector<int64_t>& dims,
    double scale) {
  auto r = std::make_unique<int8::Int8TensorCPU>();
  r->scale = scale;
  r->zero_point = 0;
  r->t.Resize(dims);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1, 1);
  for (auto i = 0; i < r->t.numel(); ++i) {
    r->t.mutable_data<int32_t>()[i] =
        static_cast<int32_t>(dis(gen) / scale + r->zero_point);
  }
  return r;
}

inline std::unique_ptr<TensorCPU> dq(const int8::Int8TensorCPU& XQ) {
  auto r = std::make_unique<Tensor>(CPU);
  r->Resize(XQ.t.sizes());
  for (auto i = 0; i < r->numel(); ++i) {
    r->mutable_data<float>()[i] =
        (static_cast<int32_t>(XQ.t.data<uint8_t>()[i]) - XQ.zero_point) *
        XQ.scale;
  }
  return r;
}

inline std::unique_ptr<TensorCPU> biasdq(const int8::Int8TensorCPU& XQ) {
  auto r = std::make_unique<Tensor>(CPU);
  r->Resize(XQ.t.sizes());
  for (auto i = 0; i < r->numel(); ++i) {
    r->mutable_data<float>()[i] =
        (XQ.t.data<int32_t>()[i] - XQ.zero_point) * XQ.scale;
  }
  return r;
}

#define EXPECT_TENSOR_EQ(_YA, _YE)                                     \
  do {                                                                 \
    EXPECT_TRUE((_YA).sizes() == (_YE).sizes());                       \
    for (auto i = 0; i < (_YA).numel(); ++i) {                         \
      EXPECT_FLOAT_EQ((_YA).data<float>()[i], (_YE).data<float>()[i]); \
    }                                                                  \
  } while (0);

#define EXPECT_TENSOR_APPROX_EQ(_YA, _YE, _tol)                            \
  do {                                                                     \
    EXPECT_TRUE((_YA).sizes() == (_YE).sizes());                           \
    for (auto i = 0; i < (_YA).numel(); ++i) {                             \
      EXPECT_NEAR((_YA).data<float>()[i], (_YE).data<float>()[i], (_tol)); \
    }                                                                      \
  } while (0);

inline void int8Copy(int8::Int8TensorCPU* dst, const int8::Int8TensorCPU& src) {
  dst->zero_point = src.zero_point;
  dst->scale = src.scale;
  dst->t.CopyFrom(src.t);
}

inline void add_input(
    const vector<int64_t>& shape,
    const vector<float>& values,
    const string& name,
    Workspace* ws) {
  // auto* t = ws->CreateBlob(name)->GetMutable<TensorCPU>();
  auto t = std::make_unique<Tensor>(CPU);
  t->Resize(shape);
  std::copy(values.begin(), values.end(), t->mutable_data<float>());
  BlobGetMutableTensor(ws->CreateBlob(name), CPU)->CopyFrom(*t);
}

inline int randomInt(int a, int b) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  return std::uniform_int_distribution<int>(a, b)(gen);
}

} // namespace caffe2

#endif // CAFFE2_INT8_TEST_UTILS_H_
