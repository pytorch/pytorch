#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/util/Exception.h>
#include <type.h>
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
  constexpr int nDims() const {
    return N;
  }
  void setSize(int i, nvfuser_index_t s) {
    size[i] = s;
  }
  void setStride(int i, nvfuser_index_t s) {
    stride[i] = s;
  }
  nvfuser_index_t getSize(int i) const {
    return size[i];
  }
  nvfuser_index_t getStride(int i) const {
    return stride[i];
  }
};

// 0-Dim GPU based tensor
template <typename T, typename nvfuser_index_t>
struct TensorArgCodegen<T, 0, nvfuser_index_t> {
  T& operator[](nvfuser_index_t ind) {
    return data[ind];
  };

  T* data;
  constexpr int nDims() const {
    return 0;
  }
  void setSize(int, nvfuser_index_t) {
    TORCH_INTERNAL_ASSERT(false, "Tried to set size of a 0-dim tensor");
  }
  void setStride(int, nvfuser_index_t) {
    TORCH_INTERNAL_ASSERT(false, "Tried to set stride of a 0-dim tensor");
  }
  nvfuser_index_t getSize(int i) const {
    TORCH_INTERNAL_ASSERT(false, "Tried to get size of a 0-dim tensor");
  }
  nvfuser_index_t getStride(int i) const {
    TORCH_INTERNAL_ASSERT(false, "Tried to get stride of a 0-dim tensor");
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

// TODO: macro this and the printer below
enum class ArgType {
  PhiloxCudaState,
  Long,
  Double,
  ComplexDouble,
  Bool,
  Tensor,
  CpuScalarTensor
};

inline std::string argTypeToString(ArgType type) {
  std::string ret;
  switch (type) {
    case ArgType::PhiloxCudaState:
      ret = "PhiloxCudaState";
      break;
    case ArgType::Long:
      ret = "Long";
      break;
    case ArgType::Double:
      ret = "Double";
      break;
    case ArgType::ComplexDouble:
      ret = "ComplexDouble";
      break;
    case ArgType::Bool:
      ret = "Bool";
      break;
    case ArgType::Tensor:
      ret = "Tensor";
      break;
    case ArgType::CpuScalarTensor:
      ret = "CpuScalarTensor";
      break;
  }
  return ret;
}

struct ArgAbstract {
  virtual ~ArgAbstract() = default;
  virtual const void* arg() const = 0;
  virtual void* arg() = 0;
  virtual bool isType(ArgType type) const = 0;
  virtual ArgType type() const = 0;
  virtual std::unique_ptr<ArgAbstract> copy_unique_ptr() const = 0;
  virtual void print() const {
    printf("input type: %s\n", argTypeToString(type()).c_str());
  };
};

#define DEF_HELPEE_FUNC(TARGET_TYPE, ARG_NAME)                    \
  bool isType(ArgType type) const override {                      \
    return ArgType::TARGET_TYPE == type;                          \
  }                                                               \
  ArgType type() const override {                                 \
    return ArgType::TARGET_TYPE;                                  \
  }                                                               \
  const void* arg() const override {                              \
    return &ARG_NAME;                                             \
  }                                                               \
  void* arg() override {                                          \
    return &ARG_NAME;                                             \
  }                                                               \
  std::unique_ptr<ArgAbstract> copy_unique_ptr() const override { \
    return std::make_unique<TARGET_TYPE##Arg>(*this);             \
  }

#define DEF_PRINT_FUNC              \
  void print() const override {     \
    std::cout << val_ << std::endl; \
  }

struct PhiloxCudaStateArg : public ArgAbstract {
  at::PhiloxCudaState val_;
  PhiloxCudaStateArg(at::PhiloxCudaState _val) : val_(_val){};
  DEF_HELPEE_FUNC(PhiloxCudaState, val_)
};

struct LongArg : public ArgAbstract {
  int64_t val_;
  explicit LongArg(int64_t _val) : val_(_val) {}
  DEF_HELPEE_FUNC(Long, val_)
  DEF_PRINT_FUNC
};

struct DoubleArg : public ArgAbstract {
  double val_;
  explicit DoubleArg(double _val) : val_(_val) {}
  DEF_HELPEE_FUNC(Double, val_)
  DEF_PRINT_FUNC
};

struct ComplexDoubleArg : public ArgAbstract {
  c10::complex<double> val_;
  explicit ComplexDoubleArg(c10::complex<double> _val) : val_(_val) {}
  DEF_HELPEE_FUNC(ComplexDouble, val_)
  DEF_PRINT_FUNC
};

struct BoolArg : public ArgAbstract {
  bool val_;
  explicit BoolArg(bool _val) : val_(_val) {}
  DEF_HELPEE_FUNC(Bool, val_)
  DEF_PRINT_FUNC
};

struct TensorArgAbstract : ArgAbstract {
  virtual void setSize(int i, int64_t size) = 0;
  virtual void setStride(int i, int64_t stride) = 0;
  virtual void setPointer(void* ptr) = 0;
  virtual void setDataType(DataType data_type) = 0;
  virtual void setTensor(at::Tensor tensor) = 0;

  virtual int64_t getRank() const = 0;
  virtual int64_t getSize(int i) const = 0;
  virtual int64_t getStride(int i) const = 0;
  virtual void* getPointer() const = 0;
  virtual DataType getDataType() const = 0;
  virtual int64_t numel() const = 0;
  virtual at::Tensor getTensor() const = 0;

  // TODO: clean it up and also print out dtype
  void print() const override {
    auto rank = getRank();
    std::cout << "tensor dtype: " << getDataType() << " sizes: (";
    for (auto i = 0; i < rank; i++) {
      std::cout << getSize(i) << ", ";
    }
    std::cout << ") stride: (";
    for (auto i = 0; i < rank; i++) {
      std::cout << getStride(i) << ", ";
    }
    std::cout << ") pointer: " << getPointer() << std::endl;
  }
};

template <typename TENSOR_TYPE, typename nvfuser_index_t>
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct TensorArg : public TensorArgAbstract {
  TENSOR_TYPE instance_;
  // TODO: this is ugly, we should be extracting data type from `instance_`
  // instead
  DataType data_type_ = DataType::Null;
  at::Tensor tensor_;

  void setSize(int i, int64_t size) override {
    instance_.setSize(i, (nvfuser_index_t)size);
  }
  void setStride(int i, int64_t stride) override {
    instance_.setStride(i, (nvfuser_index_t)stride);
  }
  void setPointer(void* ptr) override {
    instance_.data = static_cast<decltype(TENSOR_TYPE::data)>(ptr);
  }
  void setDataType(DataType data_type) override {
    data_type_ = data_type;
  }
  void setTensor(at::Tensor tensor) override {
    tensor_ = tensor;
  }

  int64_t getSize(int i) const override {
    return instance_.getSize(i);
  }
  int64_t getStride(int i) const override {
    return instance_.getStride(i);
  }
  int64_t getRank() const override {
    return instance_.nDims();
  }
  void* getPointer() const override {
    return instance_.data;
  }
  DataType getDataType() const override {
    return data_type_;
  }
  at::Tensor getTensor() const override {
    return tensor_;
  }
  int64_t numel() const override {
    int64_t ret = 1;
    for (auto i : c10::irange(instance_.nDims())) {
      ret *= instance_.getSize(i);
    }
    return ret;
  }

  DEF_HELPEE_FUNC(Tensor, instance_)
};

template <typename CPU_TENSOR_TYPE>
struct CpuScalarTensorArg : public ArgAbstract {
  CPU_TENSOR_TYPE instance_;

  CpuScalarTensorArg() = delete;

  explicit CpuScalarTensorArg(decltype(CPU_TENSOR_TYPE::data) _data) {
    instance_.data = _data;
  }

  DEF_HELPEE_FUNC(CpuScalarTensor, instance_)
};

// TODO: This class needs some further clean up and refactor
//! KernelArgumentHolder copies meta information from kernel inputs, including
//! tensor sizes/shapes/dtype/memory_ptr and copies scalar inputs. It is used
//! for both compilation as well as kernel execution. The important thing is to
//! strip ownership of tensor from KernelArgumentHolder, so that during async
//! compilation, we are not unnecessarily holding memory that is not needed.
class TORCH_CUDA_CU_API KernelArgumentHolder {
 public:
  //! create KernelArgumentHolder from c10 inputs. Note that we we not taking
  //! the ownership of the memory from the original inputs, but just recording
  //! its meta data for kernel execution/compilation.
  static KernelArgumentHolder createKernelArgumentHolder(
      const c10::ArrayRef<c10::IValue>& inputs);

  KernelIndexMode getIndexMode() const {
    return index_mode_;
  }

  explicit KernelArgumentHolder(KernelIndexMode index_mode)
      : index_mode_(index_mode) {}

  KernelArgumentHolder(const KernelArgumentHolder& self)
      : device_index_(self.getDeviceIndex()),
        cache_id_(self.getCacheId()),
        index_mode_(self.getIndexMode()) {
    for (const auto& arg : self.arguments_) {
      push(arg.get());
    }
  }

  KernelArgumentHolder& operator=(const KernelArgumentHolder& self) {
    device_index_ = self.getDeviceIndex();
    index_mode_ = self.getIndexMode();
    for (const auto& arg : self.arguments_) {
      push(arg.get());
    }
    return *this;
  }

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

  void push(const ArgAbstract* arg);

  void swap(int i, const ArgAbstract* arg);

  // push int64
  void push(int64_t val);

  const ArgAbstract* back() const {
    return arguments_.back().get();
  }

  void appendPhiloxRNGSeed(uint64_t rand_offset);

  const ArgAbstract* operator[](int ind) const {
    return arguments_.at(ind).get();
  };

  size_t size() const {
    return arguments_.size();
  }

  bool empty() const {
    return arguments_.empty();
  }

  void setDeviceIndex(int index) {
    device_index_ = index;
  }

  int getDeviceIndex() const {
    return device_index_;
  }

  void setCacheId(size_t id) {
    cache_id_ = id;
  }

  c10::optional<size_t> getCacheId() const {
    return cache_id_;
  }

  void print() const {
    for (const auto& arg : arguments_) {
      arg->print();
    }
  }

 private:
  std::vector<std::unique_ptr<ArgAbstract>> arguments_;
  std::vector<void*> void_ptrs_;
  bool changed_ = true;

  int device_index_ = 0;
  c10::optional<size_t> cache_id_ = c10::nullopt;
  KernelIndexMode index_mode_ = KernelIndexMode::INT64;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
