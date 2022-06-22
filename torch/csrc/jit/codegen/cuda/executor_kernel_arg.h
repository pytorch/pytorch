#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/ir.h>
#include <array>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// This should match the tensor used in the code generation (almost exactly)
template <typename T, int N, typename nvfuser_index_t>
struct TensorArgCodegen {
  T& operator[](nvfuser_index_t ind) {
    return data[ind];
  };

  T* data;
  std::array<nvfuser_index_t, N> size;
  std::array<nvfuser_index_t, N> stride;
  constexpr int nDims() {
    return N;
  }
  void setSize(int i, nvfuser_index_t s) {
    size[i] = s;
  }
  void setStride(int i, nvfuser_index_t s) {
    stride[i] = s;
  }
};

// 0-Dim GPU based tensor
template <typename T, typename nvfuser_index_t>
struct TensorArgCodegen<T, 0, nvfuser_index_t> {
  T& operator[](nvfuser_index_t ind) {
    return data[ind];
  };

  T* data;
  constexpr int nDims() {
    return 0;
  }
  void setSize(int, nvfuser_index_t) {
    TORCH_INTERNAL_ASSERT(false, "Tried to set size of a 0-dim tensor");
  }
  void setStride(int, nvfuser_index_t) {
    TORCH_INTERNAL_ASSERT(false, "Tried to set stride of a 0-dim tensor");
  }
};

// Specialization for 0-dim case that's easy to pass in a CPU based tensor
// without memcpy
template <typename T>
struct CpuScalarTensorCodegen {
  T& operator[](int) {
    return data;
  };

  T data;
};

struct ArgAbstract {
  virtual ~ArgAbstract() = default;
  virtual void* arg() = 0;
};

struct PhiloxCudaStateArg : public ArgAbstract {
  at::PhiloxCudaState val_;
  PhiloxCudaStateArg(at::PhiloxCudaState _val) : val_(_val){};
  void* arg() override {
    return &val_;
  }
};

struct LongArg : public ArgAbstract {
  int64_t val_;
  explicit LongArg(int64_t _val) : val_(_val) {}
  void* arg() override {
    return &val_;
  }
};

struct DoubleArg : public ArgAbstract {
  double val_;
  explicit DoubleArg(double _val) : val_(_val) {}
  void* arg() override {
    return &val_;
  }
};

struct ComplexDoubleArg : public ArgAbstract {
  c10::complex<double> val_;
  explicit ComplexDoubleArg(c10::complex<double> _val) : val_(_val) {}
  void* arg() override {
    return &val_;
  }
};

struct BoolArg : public ArgAbstract {
  bool val_;
  explicit BoolArg(bool _val) : val_(_val) {}
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
template <typename TENSOR_TYPE, typename nvfuser_index_t>
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct TensorArg : public TensorArgAbstract {
  TENSOR_TYPE instance_;

  void setSize(int i, int64_t size) override {
    instance_.setSize(i, (nvfuser_index_t)size);
  }
  void setStride(int i, int64_t stride) override {
    instance_.setStride(i, (nvfuser_index_t)stride);
  }
  void setPointer(void* ptr) override {
    instance_.data = static_cast<decltype(TENSOR_TYPE::data)>(ptr);
  }

  void* arg() override {
    return &instance_;
  }
};

template <typename CPU_TENSOR_TYPE>
struct CpuScalarTensorArg : public ArgAbstract {
  CPU_TENSOR_TYPE instance_;

  CpuScalarTensorArg() = delete;

  explicit CpuScalarTensorArg(decltype(CPU_TENSOR_TYPE::data) _data) {
    instance_.data = _data;
  }

  void* arg() override {
    return &instance_;
  }
};

class KernelArgumentHolder {
 public:
  explicit KernelArgumentHolder(KernelIndexMode index_mode)
      : index_mode_(index_mode) {}

  // Push a tensor to the arguments
  void push(const at::Tensor& tensor);

  // Push a scalar or integer to the arguments
  void push(const IValue& val);

  void push(const at::PhiloxCudaState& val);

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
  KernelIndexMode index_mode_ = KernelIndexMode::INT64;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
