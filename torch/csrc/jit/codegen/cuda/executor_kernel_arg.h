#pragma once

#include <ATen/core/ivalue.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/ir.h>

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
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  int64_t size[N];
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  int64_t stride[N];
  constexpr int nDims() {
    return N;
  }
  void setSize(int i, int64_t s) {
    size[i] = s;
  }
  void setStride(int i, int64_t s) {
    stride[i] = s;
  }
};

template <typename T>
struct TensorArgCodegen<T, 0> {
  T& operator[](int64_t ind) {
    return data[ind];
  };

  T* data;
  constexpr int nDims() {
    return 0;
  }
  void setSize(int, int64_t) {
    TORCH_INTERNAL_ASSERT(false, "Tried to set size of a 0-dim tensor");
  }
  void setStride(int, int64_t) {
    TORCH_INTERNAL_ASSERT(false, "Tried to set stride of a 0-dim tensor");
  }
};

struct ArgAbstract {
  virtual ~ArgAbstract() = default;
  virtual void* arg() = 0;
};

struct ULongArg : public ArgAbstract {
  uint64_t val_;
  ULongArg(uint64_t _val) : val_(_val) {}
  void* arg() override {
    return &val_;
  }
};

struct LongArg : public ArgAbstract {
  int64_t val_;
  LongArg(int64_t _val) : val_(_val) {}
  void* arg() override {
    return &val_;
  }
};

struct IntArg : public ArgAbstract {
  int val_;
  IntArg(int _val) : val_(_val) {}
  void* arg() override {
    return &val_;
  }
};

struct FloatArg : public ArgAbstract {
  float val_;
  FloatArg(float _val) : val_(_val) {}
  void* arg() override {
    return &val_;
  }
};

struct TensorArgAbstract : ArgAbstract {
  virtual void setSize(int i, int64_t size) = 0;
  virtual void setStride(int i, int64_t stride) = 0;
  virtual void setPointer(void* ptr) = 0;
};

// This should match the tensor used in the code generation (almost exactly)
template <typename TENSOR_TYPE>
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct TensorArg : public TensorArgAbstract {
  TENSOR_TYPE instance_;

  void setSize(int i, int64_t size) override {
    instance_.setSize(i, size);
  }
  void setStride(int i, int64_t stride) override {
    instance_.setStride(i, stride);
  }
  void setPointer(void* ptr) override {
    instance_.data = static_cast<decltype(TENSOR_TYPE::data)>(ptr);
  }

  void* arg() override {
    return &instance_;
  }
};

template <typename T>
std::unique_ptr<TensorArgAbstract> getTensorArg(int nDims) {
  switch (nDims) {
    case (0):
      return std::make_unique<TensorArg<TensorArgCodegen<T, 0>>>();
    case (1):
      return std::make_unique<TensorArg<TensorArgCodegen<T, 1>>>();
    case (2):
      return std::make_unique<TensorArg<TensorArgCodegen<T, 2>>>();
    case (3):
      return std::make_unique<TensorArg<TensorArgCodegen<T, 3>>>();
    case (4):
      return std::make_unique<TensorArg<TensorArgCodegen<T, 4>>>();
    case (5):
      return std::make_unique<TensorArg<TensorArgCodegen<T, 5>>>();
    case (6):
      return std::make_unique<TensorArg<TensorArgCodegen<T, 6>>>();
    case (7):
      return std::make_unique<TensorArg<TensorArgCodegen<T, 7>>>();
    case (8):
      return std::make_unique<TensorArg<TensorArgCodegen<T, 8>>>();
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Tried to gerneate a tensor to run a generated kernel with ",
          nDims,
          " dimensions, however it must be a 1-8 dimensional tensor.");
  }
}

std::unique_ptr<TensorArgAbstract> getTensorArg(
    c10::ScalarType dtype,
    int nDims);

class KernelArgumentHolder {
 public:
  // Push a tensor to the arguments
  void push(const at::Tensor& tensor);

  // Push a scalar or integer to the arguments
  void push(const IValue& val);

  void push(const uint64_t& val);

  // Create buffer, flatten arguments into it, align by 8 Bytes, return pointers
  // in the buffer
  void** getBuffer();

  void push(const c10::ArrayRef<c10::IValue>& args);

  void push(const std::vector<at::Tensor>& tensors);

  void appendPhiloxRNGSeed(uint64_t rand_offset);

 private:
  std::vector<std::unique_ptr<ArgAbstract>> arguments_;
  std::vector<void*> void_ptrs_;
  bool changed_ = true;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
