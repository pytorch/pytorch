#pragma once

#include <c10/core/ScalarType.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// This should match the tensor used in the code generation (almost exactly)
template <typename T, int N>
struct TensorArgCodegen {
  T& operator[](int64_t ind) {
    return data[ind];
  };

  T* data;
  int64_t size[N];
  int64_t stride[N];
  constexpr int nDims() {
    return N;
  }
};

struct ArgAbstract {
  virtual ~ArgAbstract() {}
  virtual void* arg() = 0;
};

struct IntArg : public ArgAbstract {
  int val_;
  IntArg(int _val) : val_(_val){};
  void* arg() {
    return &val_;
  }
};

struct FloatArg : public ArgAbstract {
  float val_;
  FloatArg(float _val) : val_(_val){};
  void* arg() {
    return &val_;
  }
};

struct TensorArgAbstract : ArgAbstract {
  virtual ~TensorArgAbstract(){};
  virtual void setSize(int i, int64_t size) = 0;
  virtual void setStride(int i, int64_t stride) = 0;
  virtual void setPointer(void* ptr) = 0;
};

// This should match the tensor used in the code generation (almost exactly)
template <typename TENSOR_TYPE>
struct TensorArg : public TensorArgAbstract {
  TENSOR_TYPE instance_;

  void setSize(int i, int64_t size) override {
    instance_.size[i] = size;
  }
  void setStride(int i, int64_t stride) override {
    instance_.stride[i] = stride;
  }
  void setPointer(void* ptr) override {
    instance_.data = static_cast<decltype(TENSOR_TYPE::data)>(ptr);
  }

  void* arg() override {
    return &instance_;
  }
};

template <typename T>
TensorArgAbstract* getTensorArg(int nDims) {
  switch (nDims) {
    case (1):
      return new TensorArg<TensorArgCodegen<T, 1>>();
    case (2):
      return new TensorArg<TensorArgCodegen<T, 2>>();
    case (3):
      return new TensorArg<TensorArgCodegen<T, 3>>();
    case (4):
      return new TensorArg<TensorArgCodegen<T, 4>>();
    case (5):
      return new TensorArg<TensorArgCodegen<T, 5>>();
    case (6):
      return new TensorArg<TensorArgCodegen<T, 6>>();
    case (7):
      return new TensorArg<TensorArgCodegen<T, 7>>();
    case (8):
      return new TensorArg<TensorArgCodegen<T, 8>>();
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Tried to gerneate a tensor to run a generated kernel with ",
          nDims,
          " dimensions, however it must be a 1-8 dimensional tensor.");
  }
}

TensorArgAbstract* getTensorArg(c10::ScalarType dtype, int nDims) {
  switch (dtype) {
    case (at::kFloat):
      return getTensorArg<float>(nDims);
    default:
      TORCH_CHECK(
          false,
          "Dtype: ",
          dtype,
          " not currently supported in code generated kernels.");
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch