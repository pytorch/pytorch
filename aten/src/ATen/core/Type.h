#pragma once

#include "ATen/core/ATenGeneral.h"
#include "ATen/core/Allocator.h"
#include "ATen/core/Deprecated.h"
#include "ATen/core/Generator.h"
#include "ATen/core/Layout.h"
#include "ATen/core/Scalar.h"
#include "ATen/core/ScalarType.h"
#include "ATen/core/SparseTensorRef.h"
#include "ATen/core/ArrayRef.h"
#include "ATen/core/Half.h"
#include "ATen/core/TensorTypeIdRegistration.h"
#include "ATen/core/Reduction.h"
#include "ATen/core/TensorOptions.h"

#include "c10/util/Optional.h"

#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>

// To solve the conflict of s_addr in inaddr.h
#ifdef _MSC_VER
#ifdef s_addr
#undef s_addr
#endif
#endif

namespace at {

class Context;
struct Allocator;
struct Generator;
struct Storage;
class Tensor;

static inline void noop_deleter(void*) {}

enum class TypeID {
  CPUByte,
  CPUChar,
  CPUDouble,
  CPUFloat,
  CPUInt,
  CPULong,
  CPUShort,
  CPUHalf,
  SparseCPUByte,
  SparseCPUChar,
  SparseCPUDouble,
  SparseCPUFloat,
  SparseCPUInt,
  SparseCPULong,
  SparseCPUShort,
  CUDAByte,
  CUDAChar,
  CUDADouble,
  CUDAFloat,
  CUDAInt,
  CUDALong,
  CUDAShort,
  CUDAHalf,
  SparseCUDAByte,
  SparseCUDAChar,
  SparseCUDADouble,
  SparseCUDAFloat,
  SparseCUDAInt,
  SparseCUDALong,
  SparseCUDAShort,
  CPUComplexFloat,
  CPUComplexDouble,
  CUDAComplexFloat,
  CUDAComplexDouble,
  Undefined,
  NumOptions
};

struct CAFFE2_API Type {
  explicit Type(TensorTypeId type_id, bool is_variable, bool is_undefined)
      : type_id_(type_id), is_variable_(is_variable), is_undefined_(is_undefined) {}

  virtual ~Type() {}
  virtual ScalarType scalarType() const = 0;
  virtual caffe2::TypeMeta typeMeta() const = 0;
  virtual Backend backend() const = 0;
  Layout layout() const noexcept { return layout_from_backend(backend()); }
  virtual bool is_cuda() const = 0;
  virtual bool is_sparse() const = 0;
  virtual bool is_distributed() const = 0;
  bool is_variable() const noexcept { return is_variable_; }
  bool is_undefined() const noexcept { return is_undefined_; }
  virtual Allocator * allocator() const = 0;
  virtual Device getDeviceFromPtr(void * data) const = 0;
  virtual Storage storage(bool resizable = false) const = 0;
  virtual Storage storage(size_t size, bool resizable = false) const = 0;
  virtual Storage storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter=noop_deleter) const = 0;
  virtual Storage storageWithAllocator(int64_t size, Allocator* allocator) const = 0;
  virtual std::unique_ptr<Generator> generator() const = 0;
  virtual Tensor unsafeTensorFromTH(void * th_pointer, bool retain) const = 0;
  virtual Storage unsafeStorageFromTH(void * th_pointer, bool retain) const = 0;
  virtual const char * toString() const = 0;
  virtual size_t elementSizeInBytes() const = 0;
  virtual Type & toBackend(Backend b) const = 0;
  virtual Type & toScalarType(ScalarType s) const = 0;
  Type & toSparse() const {
    return this->toBackend(at::toSparse(this->backend()));
  }
  Type & toDense() const {
    return this->toBackend(at::toDense(this->backend()));
  }
  Type & cpu() const {
    return this->toBackend(at::backendToCPU(this->backend()));
  }
  Type & cuda() const {
    return this->toBackend(at::backendToCUDA(this->backend()));
  }
  // contiguous IDs for all types in the system
  // for external dispatch
  virtual TypeID ID() const = 0;

  // New-style TensorTypeId that supports open registration.
  TensorTypeId type_id() const { return type_id_; }

  // NB: This will return DeviceType::CPU for Backend::SparseCPU
  DeviceType device_type() const {
    return backendToDeviceType(backend());
  }

  virtual Tensor copy(const Tensor & src, bool non_blocking=false, optional<Device> to_device={}) const = 0;
  virtual Tensor & copy_(Tensor & self, const Tensor & src, bool non_blocking=false) const = 0;
  virtual Tensor & s_copy_(Tensor & self, const Tensor & src, bool non_blocking) const = 0;
  virtual Tensor & _s_copy_from(const Tensor & self, Tensor & dst, bool non_blocking) const = 0;

  virtual void backward(
      Tensor& self,
      c10::optional<Tensor> gradient,
      bool keep_graph,
      bool create_graph) const = 0;
  virtual void set_data(Tensor & self, Tensor new_data) const = 0;

  virtual Tensor tensorFromBlob(void * data, IntList sizes, const std::function<void(void*)> & deleter=noop_deleter) const = 0;
  virtual Tensor tensorFromBlob(void * data, IntList sizes, IntList strides, const std::function<void(void*)> & deleter=noop_deleter) const = 0;
  virtual Tensor tensorWithAllocator(IntList sizes, Allocator* allocator) const = 0;
  virtual Tensor tensorWithAllocator(IntList sizes, IntList strides, Allocator* allocator) const = 0;
  virtual Tensor scalarTensor(Scalar s) const = 0;

  bool operator==(const Type& other) const {
    return this == &other;
  }
  bool operator!=(const Type& other) const {
    return this != &other;
  }

  /// Constructs the `TensorOptions` from a type and a `device_index`.
  TensorOptions options(int16_t device_index = -1) const {
    return TensorOptions().dtype(typeMeta())
                          .device(backendToDeviceType(backend()), device_index)
                          .layout(layout())
                          .is_variable(is_variable());
  }

  operator TensorOptions() const {
    return options();
  }

  // example
  // virtual Tensor * add(Tensor & a, Tensor & b) = 0;
  virtual Tensor & _th_triu_(Tensor & self, int64_t diagonal) const = 0;
  virtual Tensor & s__th_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const = 0;
  virtual Tensor & _th_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const = 0;
  virtual Tensor abs(const Tensor & self) const = 0;
  virtual Tensor & abs_(Tensor & self) const = 0;
  virtual Tensor acos(const Tensor & self) const = 0;
  virtual Tensor & acos_(Tensor & self) const = 0;
  virtual Tensor add(const Tensor & self, const Tensor & other, Scalar alpha) const = 0;
  virtual Tensor & add_(Tensor & self, const Tensor & other, Scalar alpha) const = 0;
  virtual Tensor add(const Tensor & self, Scalar other, Scalar alpha) const = 0;
  virtual Tensor & add_(Tensor & self, Scalar other, Scalar alpha) const = 0;
  virtual Tensor addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const = 0;
  virtual Tensor & addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const = 0;
  virtual Tensor addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const = 0;
  virtual Tensor & addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const = 0;
  virtual Tensor all(const Tensor & self, int64_t dim, bool keepdim) const = 0;
  virtual bool allclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) const = 0;
  virtual Tensor any(const Tensor & self, int64_t dim, bool keepdim) const = 0;
  virtual Tensor argmax(const Tensor & self, int64_t dim, bool keepdim) const = 0;
  virtual Tensor argmax(const Tensor & self) const = 0;
  virtual Tensor argmin(const Tensor & self, int64_t dim, bool keepdim) const = 0;
  virtual Tensor argmin(const Tensor & self) const = 0;
  virtual Tensor as_strided(const Tensor & self, IntList size, IntList stride) const = 0;
  virtual Tensor & as_strided_(Tensor & self, IntList size, IntList stride) const = 0;
  virtual Tensor as_strided(const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const = 0;
  virtual Tensor & as_strided_(Tensor & self, IntList size, IntList stride, int64_t storage_offset) const = 0;
  virtual Tensor asin(const Tensor & self) const = 0;
  virtual Tensor & asin_(Tensor & self) const = 0;
  virtual Tensor atan(const Tensor & self) const = 0;
  virtual Tensor & atan_(Tensor & self) const = 0;
  virtual Tensor baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const = 0;
  virtual Tensor & baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const = 0;
  virtual Tensor bernoulli(const Tensor & self, Generator * generator) const = 0;
  virtual Tensor & bernoulli_(Tensor & self, const Tensor & p, Generator * generator) const = 0;
  virtual Tensor & bernoulli_(Tensor & self, double p, Generator * generator) const = 0;
  virtual Tensor bernoulli(const Tensor & self, double p, Generator * generator) const = 0;
  virtual Tensor bincount(const Tensor & self, const Tensor & weights, int64_t minlength) const = 0;
  virtual Tensor bmm(const Tensor & self, const Tensor & mat2) const = 0;
  virtual Tensor ceil(const Tensor & self) const = 0;
  virtual Tensor & ceil_(Tensor & self) const = 0;
  virtual std::vector<Tensor> chunk(const Tensor & self, int64_t chunks, int64_t dim) const = 0;
  virtual Tensor clamp(const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) const = 0;
  virtual Tensor & clamp_(Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) const = 0;
  virtual Tensor clamp_max(const Tensor & self, Scalar max) const = 0;
  virtual Tensor & clamp_max_(Tensor & self, Scalar max) const = 0;
  virtual Tensor clamp_min(const Tensor & self, Scalar min) const = 0;
  virtual Tensor & clamp_min_(Tensor & self, Scalar min) const = 0;
  virtual Tensor contiguous(const Tensor & self) const = 0;
  virtual Tensor cos(const Tensor & self) const = 0;
  virtual Tensor & cos_(Tensor & self) const = 0;
  virtual Tensor cosh(const Tensor & self) const = 0;
  virtual Tensor & cosh_(Tensor & self) const = 0;
  virtual Tensor cumsum(const Tensor & self, int64_t dim, ScalarType dtype) const = 0;
  virtual Tensor cumsum(const Tensor & self, int64_t dim) const = 0;
  virtual Tensor cumprod(const Tensor & self, int64_t dim, ScalarType dtype) const = 0;
  virtual Tensor cumprod(const Tensor & self, int64_t dim) const = 0;
  virtual Tensor det(const Tensor & self) const = 0;
  virtual Tensor diag_embed(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) const = 0;
  virtual Tensor diagflat(const Tensor & self, int64_t offset) const = 0;
  virtual Tensor diagonal(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) const = 0;
  virtual Tensor div(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & div_(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor div(const Tensor & self, Scalar other) const = 0;
  virtual Tensor & div_(Tensor & self, Scalar other) const = 0;
  virtual Tensor dot(const Tensor & self, const Tensor & tensor) const = 0;
  virtual Tensor & resize_(Tensor & self, IntList size) const = 0;
  virtual Tensor erf(const Tensor & self) const = 0;
  virtual Tensor & erf_(Tensor & self) const = 0;
  virtual Tensor erfc(const Tensor & self) const = 0;
  virtual Tensor & erfc_(Tensor & self) const = 0;
  virtual Tensor exp(const Tensor & self) const = 0;
  virtual Tensor & exp_(Tensor & self) const = 0;
  virtual Tensor expm1(const Tensor & self) const = 0;
  virtual Tensor & expm1_(Tensor & self) const = 0;
  virtual Tensor expand(const Tensor & self, IntList size, bool implicit) const = 0;
  virtual Tensor expand_as(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor flatten(const Tensor & self, int64_t start_dim, int64_t end_dim) const = 0;
  virtual Tensor & fill_(Tensor & self, Scalar value) const = 0;
  virtual Tensor & fill_(Tensor & self, const Tensor & value) const = 0;
  virtual Tensor floor(const Tensor & self) const = 0;
  virtual Tensor & floor_(Tensor & self) const = 0;
  virtual Tensor ger(const Tensor & self, const Tensor & vec2) const = 0;
  virtual std::tuple<Tensor,Tensor> gesv(const Tensor & self, const Tensor & A) const = 0;
  virtual Tensor fft(const Tensor & self, int64_t signal_ndim, bool normalized) const = 0;
  virtual Tensor ifft(const Tensor & self, int64_t signal_ndim, bool normalized) const = 0;
  virtual Tensor rfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided) const = 0;
  virtual Tensor irfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided, IntList signal_sizes) const = 0;
  virtual Tensor index(const Tensor & self, TensorList indices) const = 0;
  virtual Tensor & index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const = 0;
  virtual Tensor index_put(const Tensor & self, TensorList indices, const Tensor & values) const = 0;
  virtual Tensor & index_put_(Tensor & self, TensorList indices, const Tensor & values) const = 0;
  virtual Tensor inverse(const Tensor & self) const = 0;
  virtual Tensor isclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) const = 0;
  virtual bool is_distributed(const Tensor & self) const = 0;
  virtual bool is_floating_point(const Tensor & self) const = 0;
  virtual bool is_complex(const Tensor & self) const = 0;
  virtual bool is_nonzero(const Tensor & self) const = 0;
  virtual bool is_same_size(const Tensor & self, const Tensor & other) const = 0;
  virtual bool is_signed(const Tensor & self) const = 0;
  virtual std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) const = 0;
  virtual Tensor log(const Tensor & self) const = 0;
  virtual Tensor & log_(Tensor & self) const = 0;
  virtual Tensor log10(const Tensor & self) const = 0;
  virtual Tensor & log10_(Tensor & self) const = 0;
  virtual Tensor log1p(const Tensor & self) const = 0;
  virtual Tensor & log1p_(Tensor & self) const = 0;
  virtual Tensor log2(const Tensor & self) const = 0;
  virtual Tensor & log2_(Tensor & self) const = 0;
  virtual Tensor logdet(const Tensor & self) const = 0;
  virtual Tensor log_softmax(const Tensor & self, int64_t dim, ScalarType dtype) const = 0;
  virtual Tensor log_softmax(const Tensor & self, int64_t dim) const = 0;
  virtual Tensor logsumexp(const Tensor & self, int64_t dim, bool keepdim) const = 0;
  virtual Tensor matmul(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor matrix_power(const Tensor & self, int64_t n) const = 0;
  virtual std::tuple<Tensor,Tensor> max(const Tensor & self, int64_t dim, bool keepdim) const = 0;
  virtual Tensor max_values(const Tensor & self, int64_t dim, bool keepdim) const = 0;
  virtual Tensor mean(const Tensor & self, ScalarType dtype) const = 0;
  virtual Tensor mean(const Tensor & self) const = 0;
  virtual Tensor mean(const Tensor & self, int64_t dim, bool keepdim, ScalarType dtype) const = 0;
  virtual Tensor mean(const Tensor & self, int64_t dim, bool keepdim) const = 0;
  virtual Tensor mean(const Tensor & self, int64_t dim, ScalarType dtype) const = 0;
  virtual std::tuple<Tensor,Tensor> median(const Tensor & self, int64_t dim, bool keepdim) const = 0;
  virtual std::tuple<Tensor,Tensor> min(const Tensor & self, int64_t dim, bool keepdim) const = 0;
  virtual Tensor min_values(const Tensor & self, int64_t dim, bool keepdim) const = 0;
  virtual Tensor mm(const Tensor & self, const Tensor & mat2) const = 0;
  virtual std::tuple<Tensor,Tensor> mode(const Tensor & self, int64_t dim, bool keepdim) const = 0;
  virtual Tensor mul(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & mul_(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor mul(const Tensor & self, Scalar other) const = 0;
  virtual Tensor & mul_(Tensor & self, Scalar other) const = 0;
  virtual Tensor mv(const Tensor & self, const Tensor & vec) const = 0;
  virtual Tensor mvlgamma(const Tensor & self, int64_t p) const = 0;
  virtual Tensor & mvlgamma_(Tensor & self, int64_t p) const = 0;
  virtual Tensor narrow_copy(const Tensor & self, int64_t dim, int64_t start, int64_t length) const = 0;
  virtual Tensor narrow(const Tensor & self, int64_t dim, int64_t start, int64_t length) const = 0;
  virtual Tensor permute(const Tensor & self, IntList dims) const = 0;
  virtual Tensor pin_memory(const Tensor & self) const = 0;
  virtual Tensor pinverse(const Tensor & self, double rcond) const = 0;
  virtual Tensor repeat(const Tensor & self, IntList repeats) const = 0;
  virtual Tensor reshape(const Tensor & self, IntList shape) const = 0;
  virtual Tensor reshape_as(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor round(const Tensor & self) const = 0;
  virtual Tensor & round_(Tensor & self) const = 0;
  virtual Tensor relu(const Tensor & self) const = 0;
  virtual Tensor & relu_(Tensor & self) const = 0;
  virtual Tensor prelu(const Tensor & self, const Tensor & weight) const = 0;
  virtual std::tuple<Tensor,Tensor> prelu_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight) const = 0;
  virtual Tensor hardshrink(const Tensor & self, Scalar lambd) const = 0;
  virtual Tensor hardshrink_backward(const Tensor & grad_out, const Tensor & self, Scalar lambd) const = 0;
  virtual Tensor rsqrt(const Tensor & self) const = 0;
  virtual Tensor & rsqrt_(Tensor & self) const = 0;
  virtual Tensor select(const Tensor & self, int64_t dim, int64_t index) const = 0;
  virtual Tensor sigmoid(const Tensor & self) const = 0;
  virtual Tensor & sigmoid_(Tensor & self) const = 0;
  virtual Tensor sin(const Tensor & self) const = 0;
  virtual Tensor & sin_(Tensor & self) const = 0;
  virtual Tensor sinh(const Tensor & self) const = 0;
  virtual Tensor & sinh_(Tensor & self) const = 0;
  virtual Tensor detach(const Tensor & self) const = 0;
  virtual Tensor & detach_(Tensor & self) const = 0;
  virtual int64_t size(const Tensor & self, int64_t dim) const = 0;
  virtual Tensor slice(const Tensor & self, int64_t dim, int64_t start, int64_t end, int64_t step) const = 0;
  virtual std::tuple<Tensor,Tensor> slogdet(const Tensor & self) const = 0;
  virtual Tensor smm(const Tensor & self, const Tensor & mat2) const = 0;
  virtual Tensor softmax(const Tensor & self, int64_t dim, ScalarType dtype) const = 0;
  virtual Tensor softmax(const Tensor & self, int64_t dim) const = 0;
  virtual std::vector<Tensor> split(const Tensor & self, int64_t split_size, int64_t dim) const = 0;
  virtual std::vector<Tensor> split_with_sizes(const Tensor & self, IntList split_sizes, int64_t dim) const = 0;
  virtual Tensor squeeze(const Tensor & self) const = 0;
  virtual Tensor squeeze(const Tensor & self, int64_t dim) const = 0;
  virtual Tensor & squeeze_(Tensor & self) const = 0;
  virtual Tensor & squeeze_(Tensor & self, int64_t dim) const = 0;
  virtual Tensor sspaddmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const = 0;
  virtual Tensor stft(const Tensor & self, int64_t n_fft, int64_t hop_length, int64_t win_length, const Tensor & window, bool normalized, bool onesided) const = 0;
  virtual int64_t stride(const Tensor & self, int64_t dim) const = 0;
  virtual Tensor sum(const Tensor & self, ScalarType dtype) const = 0;
  virtual Tensor sum(const Tensor & self) const = 0;
  virtual Tensor sum(const Tensor & self, IntList dim, bool keepdim, ScalarType dtype) const = 0;
  virtual Tensor sum(const Tensor & self, IntList dim, bool keepdim) const = 0;
  virtual Tensor sum(const Tensor & self, IntList dim, ScalarType dtype) const = 0;
  virtual Tensor sqrt(const Tensor & self) const = 0;
  virtual Tensor & sqrt_(Tensor & self) const = 0;
  virtual Tensor std(const Tensor & self, bool unbiased) const = 0;
  virtual Tensor std(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const = 0;
  virtual Tensor prod(const Tensor & self, ScalarType dtype) const = 0;
  virtual Tensor prod(const Tensor & self) const = 0;
  virtual Tensor prod(const Tensor & self, int64_t dim, bool keepdim, ScalarType dtype) const = 0;
  virtual Tensor prod(const Tensor & self, int64_t dim, bool keepdim) const = 0;
  virtual Tensor prod(const Tensor & self, int64_t dim, ScalarType dtype) const = 0;
  virtual Tensor t(const Tensor & self) const = 0;
  virtual Tensor & t_(Tensor & self) const = 0;
  virtual Tensor tan(const Tensor & self) const = 0;
  virtual Tensor & tan_(Tensor & self) const = 0;
  virtual Tensor tanh(const Tensor & self) const = 0;
  virtual Tensor & tanh_(Tensor & self) const = 0;
  virtual Tensor transpose(const Tensor & self, int64_t dim0, int64_t dim1) const = 0;
  virtual Tensor & transpose_(Tensor & self, int64_t dim0, int64_t dim1) const = 0;
  virtual Tensor flip(const Tensor & self, IntList dims) const = 0;
  virtual Tensor roll(const Tensor & self, IntList shifts, IntList dims) const = 0;
  virtual Tensor rot90(const Tensor & self, int64_t k, IntList dims) const = 0;
  virtual Tensor trunc(const Tensor & self) const = 0;
  virtual Tensor & trunc_(Tensor & self) const = 0;
  virtual Tensor type_as(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor unsqueeze(const Tensor & self, int64_t dim) const = 0;
  virtual Tensor & unsqueeze_(Tensor & self, int64_t dim) const = 0;
  virtual Tensor var(const Tensor & self, bool unbiased) const = 0;
  virtual Tensor var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const = 0;
  virtual Tensor view_as(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor where(const Tensor & condition, const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor norm(const Tensor & self, Scalar p) const = 0;
  virtual Tensor norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim) const = 0;
  virtual Tensor clone(const Tensor & self) const = 0;
  virtual Tensor & resize_as_(Tensor & self, const Tensor & the_template) const = 0;
  virtual Tensor pow(const Tensor & self, Scalar exponent) const = 0;
  virtual Tensor & zero_(Tensor & self) const = 0;
  virtual Tensor sub(const Tensor & self, const Tensor & other, Scalar alpha) const = 0;
  virtual Tensor & sub_(Tensor & self, const Tensor & other, Scalar alpha) const = 0;
  virtual Tensor sub(const Tensor & self, Scalar other, Scalar alpha) const = 0;
  virtual Tensor & sub_(Tensor & self, Scalar other, Scalar alpha) const = 0;
  virtual Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const = 0;
  virtual Tensor & addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const = 0;
  virtual Tensor & sparse_resize_(Tensor & self, IntList size, int64_t sparse_dim, int64_t dense_dim) const = 0;
  virtual Tensor & sparse_resize_and_clear_(Tensor & self, IntList size, int64_t sparse_dim, int64_t dense_dim) const = 0;
  virtual Tensor sparse_mask(const Tensor & self, SparseTensorRef mask) const = 0;
  virtual Tensor to_dense(const Tensor & self) const = 0;
  virtual int64_t sparse_dim(const Tensor & self) const = 0;
  virtual int64_t _dimI(const Tensor & self) const = 0;
  virtual int64_t dense_dim(const Tensor & self) const = 0;
  virtual int64_t _dimV(const Tensor & self) const = 0;
  virtual int64_t _nnz(const Tensor & self) const = 0;
  virtual Tensor coalesce(const Tensor & self) const = 0;
  virtual bool is_coalesced(const Tensor & self) const = 0;
  virtual Tensor _indices(const Tensor & self) const = 0;
  virtual Tensor _values(const Tensor & self) const = 0;
  virtual Tensor & _coalesced_(Tensor & self, bool coalesced) const = 0;
  virtual Tensor indices(const Tensor & self) const = 0;
  virtual Tensor values(const Tensor & self) const = 0;
  virtual int64_t numel(const Tensor & self) const = 0;
  virtual std::vector<Tensor> unbind(const Tensor & self, int64_t dim) const = 0;
  virtual Tensor to_sparse(const Tensor & self, int64_t sparse_dim) const = 0;
  virtual Tensor to_sparse(const Tensor & self) const = 0;
  virtual Tensor to(const Tensor & self, const TensorOptions & options, bool non_blocking, bool copy) const = 0;
  virtual Tensor to(const Tensor & self, Device device, ScalarType dtype, bool non_blocking, bool copy) const = 0;
  virtual Tensor to(const Tensor & self, ScalarType dtype, bool non_blocking, bool copy) const = 0;
  virtual Tensor to(const Tensor & self, const Tensor & other, bool non_blocking, bool copy) const = 0;
  virtual Scalar _local_scalar(const Tensor & self) const = 0;
  virtual void* data_ptr(const Tensor & self) const = 0;
  virtual Tensor & set_(Tensor & self, Storage source) const = 0;
  virtual Tensor & set_(Tensor & self, Storage source, int64_t storage_offset, IntList size, IntList stride) const = 0;
  virtual Tensor & set_(Tensor & self, const Tensor & source) const = 0;
  virtual Tensor & set_(Tensor & self) const = 0;
  virtual bool is_set_to(const Tensor & self, const Tensor & tensor) const = 0;
  virtual Tensor & masked_fill_(Tensor & self, const Tensor & mask, Scalar value) const = 0;
  virtual Tensor & masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) const = 0;
  virtual Tensor & masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) const = 0;
  virtual Tensor view(const Tensor & self, IntList size) const = 0;
  virtual Tensor & put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) const = 0;
  virtual Tensor & index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const = 0;
  virtual Tensor & index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const = 0;
  virtual Tensor & index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) const = 0;
  virtual Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const = 0;
  virtual Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const = 0;
  virtual Tensor & scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const = 0;
  virtual Tensor & lt_(Tensor & self, Scalar other) const = 0;
  virtual Tensor & lt_(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & gt_(Tensor & self, Scalar other) const = 0;
  virtual Tensor & gt_(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & le_(Tensor & self, Scalar other) const = 0;
  virtual Tensor & le_(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & ge_(Tensor & self, Scalar other) const = 0;
  virtual Tensor & ge_(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & eq_(Tensor & self, Scalar other) const = 0;
  virtual Tensor & eq_(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & ne_(Tensor & self, Scalar other) const = 0;
  virtual Tensor & ne_(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor __and__(const Tensor & self, Scalar other) const = 0;
  virtual Tensor __and__(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & __iand__(Tensor & self, Scalar other) const = 0;
  virtual Tensor & __iand__(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor __or__(const Tensor & self, Scalar other) const = 0;
  virtual Tensor __or__(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & __ior__(Tensor & self, Scalar other) const = 0;
  virtual Tensor & __ior__(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor __xor__(const Tensor & self, Scalar other) const = 0;
  virtual Tensor __xor__(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & __ixor__(Tensor & self, Scalar other) const = 0;
  virtual Tensor & __ixor__(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor __lshift__(const Tensor & self, Scalar other) const = 0;
  virtual Tensor __lshift__(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & __ilshift__(Tensor & self, Scalar other) const = 0;
  virtual Tensor & __ilshift__(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor __rshift__(const Tensor & self, Scalar other) const = 0;
  virtual Tensor __rshift__(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & __irshift__(Tensor & self, Scalar other) const = 0;
  virtual Tensor & __irshift__(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & lgamma_(Tensor & self) const = 0;
  virtual Tensor & atan2_(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & tril_(Tensor & self, int64_t diagonal) const = 0;
  virtual Tensor & triu_(Tensor & self, int64_t diagonal) const = 0;
  virtual Tensor & digamma_(Tensor & self) const = 0;
  virtual Tensor & polygamma_(Tensor & self, int64_t n) const = 0;
  virtual Tensor & erfinv_(Tensor & self) const = 0;
  virtual Tensor & frac_(Tensor & self) const = 0;
  virtual Tensor & renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const = 0;
  virtual Tensor & reciprocal_(Tensor & self) const = 0;
  virtual Tensor & neg_(Tensor & self) const = 0;
  virtual Tensor & pow_(Tensor & self, Scalar exponent) const = 0;
  virtual Tensor & pow_(Tensor & self, const Tensor & exponent) const = 0;
  virtual Tensor & lerp_(Tensor & self, const Tensor & end, Scalar weight) const = 0;
  virtual Tensor & sign_(Tensor & self) const = 0;
  virtual Tensor & fmod_(Tensor & self, Scalar other) const = 0;
  virtual Tensor & fmod_(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & remainder_(Tensor & self, Scalar other) const = 0;
  virtual Tensor & remainder_(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const = 0;
  virtual Tensor addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const = 0;
  virtual Tensor & addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const = 0;
  virtual Tensor & addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const = 0;
  virtual Tensor & random_(Tensor & self, int64_t from, int64_t to, Generator * generator) const = 0;
  virtual Tensor & random_(Tensor & self, int64_t to, Generator * generator) const = 0;
  virtual Tensor & random_(Tensor & self, Generator * generator) const = 0;
  virtual Tensor & uniform_(Tensor & self, double from, double to, Generator * generator) const = 0;
  virtual Tensor & normal_(Tensor & self, double mean, double std, Generator * generator) const = 0;
  virtual Tensor & cauchy_(Tensor & self, double median, double sigma, Generator * generator) const = 0;
  virtual Tensor & log_normal_(Tensor & self, double mean, double std, Generator * generator) const = 0;
  virtual Tensor & exponential_(Tensor & self, double lambd, Generator * generator) const = 0;
  virtual Tensor & geometric_(Tensor & self, double p, Generator * generator) const = 0;
  virtual Tensor diag(const Tensor & self, int64_t diagonal) const = 0;
  virtual Tensor cross(const Tensor & self, const Tensor & other, int64_t dim) const = 0;
  virtual Tensor triu(const Tensor & self, int64_t diagonal) const = 0;
  virtual Tensor tril(const Tensor & self, int64_t diagonal) const = 0;
  virtual Tensor trace(const Tensor & self) const = 0;
  virtual Tensor ne(const Tensor & self, Scalar other) const = 0;
  virtual Tensor ne(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor eq(const Tensor & self, Scalar other) const = 0;
  virtual Tensor eq(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor ge(const Tensor & self, Scalar other) const = 0;
  virtual Tensor ge(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor le(const Tensor & self, Scalar other) const = 0;
  virtual Tensor le(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor gt(const Tensor & self, Scalar other) const = 0;
  virtual Tensor gt(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor lt(const Tensor & self, Scalar other) const = 0;
  virtual Tensor lt(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor take(const Tensor & self, const Tensor & index) const = 0;
  virtual Tensor index_select(const Tensor & self, int64_t dim, const Tensor & index) const = 0;
  virtual Tensor masked_select(const Tensor & self, const Tensor & mask) const = 0;
  virtual Tensor nonzero(const Tensor & self) const = 0;
  virtual Tensor gather(const Tensor & self, int64_t dim, const Tensor & index) const = 0;
  virtual Tensor addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const = 0;
  virtual Tensor addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const = 0;
  virtual std::tuple<Tensor,Tensor> gels(const Tensor & self, const Tensor & A) const = 0;
  virtual std::tuple<Tensor,Tensor> trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) const = 0;
  virtual std::tuple<Tensor,Tensor> symeig(const Tensor & self, bool eigenvectors, bool upper) const = 0;
  virtual std::tuple<Tensor,Tensor> eig(const Tensor & self, bool eigenvectors) const = 0;
  virtual std::tuple<Tensor,Tensor,Tensor> svd(const Tensor & self, bool some, bool compute_uv) const = 0;
  virtual Tensor cholesky(const Tensor & self, bool upper) const = 0;
  virtual Tensor potrs(const Tensor & self, const Tensor & input2, bool upper) const = 0;
  virtual Tensor potri(const Tensor & self, bool upper) const = 0;
  virtual std::tuple<Tensor,Tensor> pstrf(const Tensor & self, bool upper, Scalar tol) const = 0;
  virtual std::tuple<Tensor,Tensor> qr(const Tensor & self) const = 0;
  virtual std::tuple<Tensor,Tensor> geqrf(const Tensor & self) const = 0;
  virtual Tensor orgqr(const Tensor & self, const Tensor & input2) const = 0;
  virtual Tensor ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) const = 0;
  virtual std::tuple<Tensor,Tensor> btrifact(const Tensor & self, bool pivot) const = 0;
  virtual std::tuple<Tensor,Tensor,Tensor> btrifact_with_info(const Tensor & self, bool pivot) const = 0;
  virtual Tensor btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const = 0;
  virtual Tensor multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const = 0;
  virtual Tensor lgamma(const Tensor & self) const = 0;
  virtual Tensor digamma(const Tensor & self) const = 0;
  virtual Tensor polygamma(int64_t n, const Tensor & self) const = 0;
  virtual Tensor erfinv(const Tensor & self) const = 0;
  virtual Tensor frac(const Tensor & self) const = 0;
  virtual Tensor dist(const Tensor & self, const Tensor & other, Scalar p) const = 0;
  virtual Tensor reciprocal(const Tensor & self) const = 0;
  virtual Tensor neg(const Tensor & self) const = 0;
  virtual Tensor atan2(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor lerp(const Tensor & self, const Tensor & end, Scalar weight) const = 0;
  virtual Tensor histc(const Tensor & self, int64_t bins, Scalar min, Scalar max) const = 0;
  virtual Tensor sign(const Tensor & self) const = 0;
  virtual Tensor fmod(const Tensor & self, Scalar other) const = 0;
  virtual Tensor fmod(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor remainder(const Tensor & self, Scalar other) const = 0;
  virtual Tensor remainder(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor min(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor min(const Tensor & self) const = 0;
  virtual Tensor max(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor max(const Tensor & self) const = 0;
  virtual Tensor median(const Tensor & self) const = 0;
  virtual std::tuple<Tensor,Tensor> sort(const Tensor & self, int64_t dim, bool descending) const = 0;
  virtual std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const = 0;
  virtual Tensor all(const Tensor & self) const = 0;
  virtual Tensor any(const Tensor & self) const = 0;
  virtual Tensor renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const = 0;
  virtual Tensor unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const = 0;
  virtual bool equal(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor pow(const Tensor & self, const Tensor & exponent) const = 0;
  virtual Tensor pow(Scalar self, const Tensor & exponent) const = 0;
  virtual Tensor alias(const Tensor & self) const = 0;
protected:
  TensorTypeId type_id_;
  bool is_variable_;
  bool is_undefined_;
};

} // namespace at

#include "ATen/core/Tensor.h"
