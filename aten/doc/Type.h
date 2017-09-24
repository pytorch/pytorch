#pragma once

#include <memory>
#include <limits>

#include "ATen/ArrayRef.h"
#include "ATen/Half.h"
#include "ATen/SparseTensorRef.h"

namespace at {

class Context;
struct Storage;
struct Tensor;
class Scalar;
struct Generator;

#define AT_FORALL_SCALAR_TYPES(_) \
_(uint8_t,Byte,i) \
_(int8_t,Char,i) \
_(double,Double,d) \
_(float,Float,d) \
_(int,Int,i) \
_(int64_t,Long,i) \
_(int16_t,Short,i) \
_(Half,Half,d)

enum class ScalarType {
#define DEFINE_ENUM(_1,n,_2) \
  n,
  AT_FORALL_SCALAR_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
  NumOptions
};

enum class Backend {
  CPU,
  CUDA,
  SparseCPU,
  SparseCUDA,
  NumOptions
};


// The constexpr specifier declares that it is possible to evaluate the value
// of the function or variable at compile time.
constexpr Backend kCPU = Backend::CPU;
constexpr Backend kCUDA = Backend::CUDA;
constexpr Backend kSparseCPU = Backend::SparseCPU;
constexpr Backend kSparseCUDA = Backend::SparseCUDA;

// Note [Undefined-dim versus 0-dim]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Unlike Torch, ATen treats zero-dimension tensors as having ONE
// element (that is to say, a zero-dimensional tensor is a scalar!)
// This is in contrast to Torch, where a zero-dimension tensor has
// zero elements.
//
// Because we are backed by Torch tensors, we need to be able to
// represent this state (of numel==0).  kUndefinedDimensions represents this
// situation.
constexpr int64_t kUndefinedDimensions = std::numeric_limits<int64_t>::min();

static inline const char * toString(Backend b) {
  switch(b) {
    case Backend::CPU: return "CPU";
    case Backend::CUDA: return "CUDA";
    case Backend::SparseCPU: return "SparseCPU";
    case Backend::SparseCUDA: return "SparseCUDA";
    default: return "UNKNOWN_BACKEND";
  }
}

#define DEFINE_CONSTANT(_,name,_2) \
constexpr ScalarType k##name = ScalarType::name;

AT_FORALL_SCALAR_TYPES(DEFINE_CONSTANT)
#undef DEFINE_CONSTANT

static inline const char * toString(ScalarType t) {
#define DEFINE_CASE(_,name,_2) \
  case ScalarType:: name : return #name;

  switch(t) {
    AT_FORALL_SCALAR_TYPES(DEFINE_CASE)
    default:
      return "UNKNOWN_SCALAR_TYPE";
  }
#undef DEFINE_CASE
}

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
  NumOptions
};


typedef ArrayRef<int64_t> IntList;
typedef ArrayRef<Tensor> TensorList;

struct Type {
  explicit Type(Context * context)
  : context(context) {}
  virtual ~Type() {}
  virtual ScalarType scalarType() const = 0;
  virtual Backend backend() const = 0;
  virtual bool isCuda() const = 0;
  virtual bool isSparse() const = 0;
  virtual bool isDistributed() const = 0;
  static void registerAll(Context * context);
  virtual std::unique_ptr<Storage> storage() const = 0;
  virtual std::unique_ptr<Storage> storage(size_t size) const = 0;
  virtual std::unique_ptr<Storage> storageFromBlob(void * data, int64_t size) const = 0;
  virtual std::unique_ptr<Generator> generator() const = 0;
  virtual Tensor unsafeTensorFromTH(void * th_pointer, bool retain) const = 0;
  virtual const char * toString() const = 0;
  virtual std::size_t elementSizeInBytes() const = 0;
  Type & toBackend(Backend b) const;
  Type & toScalarType(ScalarType s) const;

  // contingious IDs for all types in the system
  // for external dispatch
  virtual TypeID ID() const = 0;

  virtual void copy(const Tensor & src, Tensor & dst) const = 0;
  Tensor copy(const Tensor & src) const;

  Tensor tensorFromBlob(void * data, IntList sizes) const;
  Tensor tensorFromBlob(void * data, IntList sizes, IntList strides) const;
  Tensor scalarTensor(Scalar s) const;

  bool operator==(const Type& other) const;

  // example
  // virtual Tensor * add(Tensor & a, Tensor & b) = 0;
  virtual int64_t m_storage_offset(const Tensor & self) const;
  virtual Tensor & m_resize_(Tensor & self, IntList size) const;
  virtual Tensor & zeros_out(IntList size, Tensor & result) const;
  virtual Tensor zeros(IntList size) const;
  virtual Tensor & ones_out(IntList size, Tensor & result) const;
  virtual Tensor ones(IntList size) const;
  virtual int64_t numel(const Tensor & self) const;
  virtual Tensor & m_set_(Tensor & self, Storage & storage) const;
  virtual Tensor & m_set_(Tensor & self, Storage & sourceStorage, int64_t storage_offset, IntList size, IntList stride) const;
  Tensor & m_set_(Tensor & self, Storage & sourceStorage, int64_t storage_offset, IntList size) const;
  virtual Tensor & m_set_(Tensor & self, const Tensor & source) const;
  virtual Tensor & m_set_(Tensor & self) const;
  virtual Tensor & m_fill_(Tensor & self, Scalar value) const;
  virtual bool m_is_same_size(const Tensor & self, const Tensor & other) const;
  virtual bool m_is_contiguous(const Tensor & self) const;
  virtual bool m_is_set_to(const Tensor & self, const Tensor & tensor) const;
  Tensor & m_masked_fill_(Tensor & self, const Tensor & mask, Scalar value) const;
  virtual Tensor & s_m_masked_fill_(Tensor & self, const Tensor & mask, Scalar value) const;
  Tensor & m_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) const;
  virtual Tensor & s_m_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) const;
  Tensor & masked_select_out(const Tensor & self, const Tensor & mask, Tensor & result) const;
  virtual Tensor & s_masked_select_out(const Tensor & self, const Tensor & mask, Tensor & result) const;
  Tensor masked_select(const Tensor & self, const Tensor & mask) const;
  virtual Tensor s_masked_select(const Tensor & self, const Tensor & mask) const;
  virtual Tensor transpose(const Tensor & self, int64_t dim0, int64_t dim1) const;
  virtual Tensor & m_transpose_(Tensor & self, int64_t dim0, int64_t dim1) const;
  virtual Tensor t(const Tensor & self) const;
  virtual Tensor & m_t_(Tensor & self) const;
  virtual Tensor & squeeze_out(const Tensor & self, int64_t dim, Tensor & result) const;
  virtual Tensor squeeze(const Tensor & self, int64_t dim) const;
  virtual Tensor & squeeze_out(const Tensor & self, Tensor & result) const;
  virtual Tensor squeeze(const Tensor & self) const;
  virtual Tensor & m_squeeze_(Tensor & self, int64_t dim) const;
  virtual Tensor & m_squeeze_(Tensor & self) const;
  virtual Tensor & unsqueeze_out(const Tensor & self, int64_t dim, Tensor & result) const;
  virtual Tensor unsqueeze(const Tensor & self, int64_t dim) const;
  virtual Tensor & m_unsqueeze_(Tensor & self, int64_t dim) const;
  virtual Tensor & nonzero_out(const Tensor & self, Tensor & result) const;
  virtual Tensor nonzero(const Tensor & self) const;
  virtual Tensor m_contiguous(const Tensor & self) const;
  virtual Tensor m_clone(const Tensor & self) const;
  virtual Tensor m_view(const Tensor & self, IntList size) const;
  virtual Tensor m_expand(const Tensor & self, IntList size) const;
  virtual Tensor & m_resize_as_(Tensor & self, const Tensor & the_template) const;
  virtual Tensor & index_select_out(const Tensor & self, int64_t dim, const Tensor & index, Tensor & result) const;
  virtual Tensor index_select(const Tensor & self, int64_t dim, const Tensor & index) const;
  virtual Tensor & m_index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const;
  virtual Tensor & m_index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const;
  virtual Tensor & m_index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const;
  virtual Tensor m_narrow(const Tensor & self, int64_t dimension, int64_t start, int64_t length) const;
  virtual Tensor m_unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const;
  virtual Tensor & range_out(Scalar start, Scalar end, Scalar step, Tensor & result) const;
  virtual Tensor range(Scalar start, Scalar end, Scalar step) const;
  Tensor & range_out(Scalar start, Scalar end, Tensor & result) const;
  Tensor range(Scalar start, Scalar end) const;
  virtual Tensor & m_scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const;
  virtual Tensor & m_scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const;
  virtual Tensor & m_scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const;
  virtual Tensor & gather_out(const Tensor & self, int64_t dim, const Tensor & index, Tensor & result) const;
  virtual Tensor gather(const Tensor & self, int64_t dim, const Tensor & index) const;
  virtual void* m_data_ptr(const Tensor & self) const;
  virtual bool equal(const Tensor & self, const Tensor & other) const;
  virtual Tensor & __and___out(const Tensor & self, Scalar value, Tensor & result) const;
  virtual Tensor __and__(const Tensor & self, Scalar value) const;
  virtual Tensor & __and___out(const Tensor & self, const Tensor & other, Tensor & result) const;
  virtual Tensor __and__(const Tensor & self, const Tensor & other) const;
  virtual Tensor & __iand__(Tensor & self, Scalar value) const;
  virtual Tensor & __iand__(Tensor & self, const Tensor & other) const;
  virtual Tensor & __or___out(const Tensor & self, Scalar value, Tensor & result) const;
  virtual Tensor __or__(const Tensor & self, Scalar value) const;
  virtual Tensor & __or___out(const Tensor & self, const Tensor & other, Tensor & result) const;
  virtual Tensor __or__(const Tensor & self, const Tensor & other) const;
  virtual Tensor & __ior__(Tensor & self, Scalar value) const;
  virtual Tensor & __ior__(Tensor & self, const Tensor & other) const;
  virtual Tensor & __xor___out(const Tensor & self, Scalar value, Tensor & result) const;
  virtual Tensor __xor__(const Tensor & self, Scalar value) const;
  virtual Tensor & __xor___out(const Tensor & self, const Tensor & other, Tensor & result) const;
  virtual Tensor __xor__(const Tensor & self, const Tensor & other) const;
  virtual Tensor & __ixor__(Tensor & self, Scalar value) const;
  virtual Tensor & __ixor__(Tensor & self, const Tensor & other) const;
  virtual Tensor & __lshift___out(const Tensor & self, Scalar value, Tensor & result) const;
  virtual Tensor __lshift__(const Tensor & self, Scalar value) const;
  virtual Tensor & __lshift___out(const Tensor & self, const Tensor & other, Tensor & result) const;
  virtual Tensor __lshift__(const Tensor & self, const Tensor & other) const;
  virtual Tensor & __ilshift__(Tensor & self, Scalar value) const;
  virtual Tensor & __ilshift__(Tensor & self, const Tensor & other) const;
  virtual Tensor & __rshift___out(const Tensor & self, Scalar value, Tensor & result) const;
  virtual Tensor __rshift__(const Tensor & self, Scalar value) const;
  virtual Tensor & __rshift___out(const Tensor & self, const Tensor & other, Tensor & result) const;
  virtual Tensor __rshift__(const Tensor & self, const Tensor & other) const;
  virtual Tensor & __irshift__(Tensor & self, Scalar value) const;
  virtual Tensor & __irshift__(Tensor & self, const Tensor & other) const;
  virtual Tensor m_lt(const Tensor & self, Scalar value) const;
  Tensor m_lt(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_m_lt(const Tensor & self, const Tensor & other) const;
  virtual Tensor & m_lt_(Tensor & self, Scalar value) const;
  Tensor & m_lt_(Tensor & self, const Tensor & other) const;
  virtual Tensor & s_m_lt_(Tensor & self, const Tensor & other) const;
  virtual Tensor & lt_out(const Tensor & tensor, Scalar value, Tensor & result) const;
  virtual Tensor lt(const Tensor & tensor, Scalar value) const;
  Tensor & lt_out(const Tensor & tensor, const Tensor & other, Tensor & result) const;
  virtual Tensor & s_lt_out(const Tensor & tensor, const Tensor & other, Tensor & result) const;
  Tensor lt(const Tensor & tensor, const Tensor & other) const;
  virtual Tensor s_lt(const Tensor & tensor, const Tensor & other) const;
  virtual Tensor m_gt(const Tensor & self, Scalar value) const;
  Tensor m_gt(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_m_gt(const Tensor & self, const Tensor & other) const;
  virtual Tensor & m_gt_(Tensor & self, Scalar value) const;
  Tensor & m_gt_(Tensor & self, const Tensor & other) const;
  virtual Tensor & s_m_gt_(Tensor & self, const Tensor & other) const;
  virtual Tensor & gt_out(const Tensor & tensor, Scalar value, Tensor & result) const;
  virtual Tensor gt(const Tensor & tensor, Scalar value) const;
  Tensor & gt_out(const Tensor & tensor, const Tensor & other, Tensor & result) const;
  virtual Tensor & s_gt_out(const Tensor & tensor, const Tensor & other, Tensor & result) const;
  Tensor gt(const Tensor & tensor, const Tensor & other) const;
  virtual Tensor s_gt(const Tensor & tensor, const Tensor & other) const;
  virtual Tensor m_le(const Tensor & self, Scalar value) const;
  Tensor m_le(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_m_le(const Tensor & self, const Tensor & other) const;
  virtual Tensor & m_le_(Tensor & self, Scalar value) const;
  Tensor & m_le_(Tensor & self, const Tensor & other) const;
  virtual Tensor & s_m_le_(Tensor & self, const Tensor & other) const;
  virtual Tensor & le_out(const Tensor & tensor, Scalar value, Tensor & result) const;
  virtual Tensor le(const Tensor & tensor, Scalar value) const;
  Tensor & le_out(const Tensor & tensor, const Tensor & other, Tensor & result) const;
  virtual Tensor & s_le_out(const Tensor & tensor, const Tensor & other, Tensor & result) const;
  Tensor le(const Tensor & tensor, const Tensor & other) const;
  virtual Tensor s_le(const Tensor & tensor, const Tensor & other) const;
  virtual Tensor m_ge(const Tensor & self, Scalar value) const;
  Tensor m_ge(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_m_ge(const Tensor & self, const Tensor & other) const;
  virtual Tensor & m_ge_(Tensor & self, Scalar value) const;
  Tensor & m_ge_(Tensor & self, const Tensor & other) const;
  virtual Tensor & s_m_ge_(Tensor & self, const Tensor & other) const;
  virtual Tensor & ge_out(const Tensor & tensor, Scalar value, Tensor & result) const;
  virtual Tensor ge(const Tensor & tensor, Scalar value) const;
  Tensor & ge_out(const Tensor & tensor, const Tensor & other, Tensor & result) const;
  virtual Tensor & s_ge_out(const Tensor & tensor, const Tensor & other, Tensor & result) const;
  Tensor ge(const Tensor & tensor, const Tensor & other) const;
  virtual Tensor s_ge(const Tensor & tensor, const Tensor & other) const;
  virtual Tensor m_eq(const Tensor & self, Scalar value) const;
  Tensor m_eq(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_m_eq(const Tensor & self, const Tensor & other) const;
  virtual Tensor & m_eq_(Tensor & self, Scalar value) const;
  Tensor & m_eq_(Tensor & self, const Tensor & other) const;
  virtual Tensor & s_m_eq_(Tensor & self, const Tensor & other) const;
  virtual Tensor & eq_out(const Tensor & tensor, Scalar value, Tensor & result) const;
  virtual Tensor eq(const Tensor & tensor, Scalar value) const;
  Tensor & eq_out(const Tensor & tensor, const Tensor & other, Tensor & result) const;
  virtual Tensor & s_eq_out(const Tensor & tensor, const Tensor & other, Tensor & result) const;
  Tensor eq(const Tensor & tensor, const Tensor & other) const;
  virtual Tensor s_eq(const Tensor & tensor, const Tensor & other) const;
  virtual Tensor m_ne(const Tensor & self, Scalar value) const;
  Tensor m_ne(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_m_ne(const Tensor & self, const Tensor & other) const;
  virtual Tensor & m_ne_(Tensor & self, Scalar value) const;
  Tensor & m_ne_(Tensor & self, const Tensor & other) const;
  virtual Tensor & s_m_ne_(Tensor & self, const Tensor & other) const;
  virtual Tensor & ne_out(const Tensor & tensor, Scalar value, Tensor & result) const;
  virtual Tensor ne(const Tensor & tensor, Scalar value) const;
  Tensor & ne_out(const Tensor & tensor, const Tensor & other, Tensor & result) const;
  virtual Tensor & s_ne_out(const Tensor & tensor, const Tensor & other, Tensor & result) const;
  Tensor ne(const Tensor & tensor, const Tensor & other) const;
  virtual Tensor s_ne(const Tensor & tensor, const Tensor & other) const;
  virtual std::tuple<Tensor &,Tensor &> min_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & min, Tensor & min_indices) const;
  virtual std::tuple<Tensor,Tensor> min(const Tensor & self, int64_t dim, bool keepdim) const;
  virtual std::tuple<Tensor &,Tensor &> min_out(const Tensor & self, int64_t dim, Tensor & min, Tensor & min_indices) const;
  virtual std::tuple<Tensor,Tensor> min(const Tensor & self, int64_t dim) const;
  Tensor & min_out(const Tensor & self, const Tensor & other, Tensor & result) const;
  virtual Tensor & s_min_out(const Tensor & self, const Tensor & other, Tensor & result) const;
  Tensor min(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_min(const Tensor & self, const Tensor & other) const;
  virtual Scalar min(const Tensor & self) const;
  virtual std::tuple<Tensor &,Tensor &> max_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & max, Tensor & max_indices) const;
  virtual std::tuple<Tensor,Tensor> max(const Tensor & self, int64_t dim, bool keepdim) const;
  virtual std::tuple<Tensor &,Tensor &> max_out(const Tensor & self, int64_t dim, Tensor & max, Tensor & max_indices) const;
  virtual std::tuple<Tensor,Tensor> max(const Tensor & self, int64_t dim) const;
  Tensor & max_out(const Tensor & self, const Tensor & other, Tensor & result) const;
  virtual Tensor & s_max_out(const Tensor & self, const Tensor & other, Tensor & result) const;
  Tensor max(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_max(const Tensor & self, const Tensor & other) const;
  virtual Scalar max(const Tensor & self) const;
  virtual std::tuple<Tensor &,Tensor &> kthvalue_out(const Tensor & self, int64_t k, bool keepdim, Tensor & values, Tensor & indices) const;
  virtual std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k, bool keepdim) const;
  virtual std::tuple<Tensor &,Tensor &> kthvalue_out(const Tensor & self, int64_t k, Tensor & values, Tensor & indices) const;
  virtual std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k) const;
  virtual std::tuple<Tensor &,Tensor &> kthvalue_out(const Tensor & self, int64_t k, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) const;
  virtual std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) const;
  virtual std::tuple<Tensor &,Tensor &> kthvalue_out(const Tensor & self, int64_t k, int64_t dim, Tensor & values, Tensor & indices) const;
  virtual std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k, int64_t dim) const;
  virtual std::tuple<Tensor &,Tensor &> mode_out(const Tensor & self, bool keepdim, Tensor & values, Tensor & indices) const;
  virtual std::tuple<Tensor,Tensor> mode(const Tensor & self, bool keepdim) const;
  virtual std::tuple<Tensor &,Tensor &> mode_out(const Tensor & self, Tensor & values, Tensor & indices) const;
  virtual std::tuple<Tensor,Tensor> mode(const Tensor & self) const;
  virtual std::tuple<Tensor &,Tensor &> mode_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) const;
  virtual std::tuple<Tensor,Tensor> mode(const Tensor & self, int64_t dim, bool keepdim) const;
  virtual std::tuple<Tensor &,Tensor &> mode_out(const Tensor & self, int64_t dim, Tensor & values, Tensor & indices) const;
  virtual std::tuple<Tensor,Tensor> mode(const Tensor & self, int64_t dim) const;
  virtual std::tuple<Tensor &,Tensor &> median_out(const Tensor & self, bool keepdim, Tensor & values, Tensor & indices) const;
  virtual std::tuple<Tensor,Tensor> median(const Tensor & self, bool keepdim) const;
  virtual std::tuple<Tensor &,Tensor &> median_out(const Tensor & self, int64_t dim, Tensor & values, Tensor & indices) const;
  virtual std::tuple<Tensor,Tensor> median(const Tensor & self, int64_t dim) const;
  virtual std::tuple<Tensor &,Tensor &> median_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) const;
  virtual std::tuple<Tensor,Tensor> median(const Tensor & self, int64_t dim, bool keepdim) const;
  virtual Scalar median(const Tensor & self) const;
  virtual std::tuple<Tensor &,Tensor &> sort_out(const Tensor & self, Tensor & values, Tensor & indices) const;
  virtual std::tuple<Tensor,Tensor> sort(const Tensor & self) const;
  virtual std::tuple<Tensor &,Tensor &> sort_out(const Tensor & self, int64_t dim, Tensor & values, Tensor & indices) const;
  virtual std::tuple<Tensor,Tensor> sort(const Tensor & self, int64_t dim) const;
  virtual std::tuple<Tensor &,Tensor &> sort_out(const Tensor & self, int64_t dim, bool descending, Tensor & values, Tensor & indices) const;
  virtual std::tuple<Tensor,Tensor> sort(const Tensor & self, int64_t dim, bool descending) const;
  virtual std::tuple<Tensor &,Tensor &> topk_out(const Tensor & self, int64_t k, Tensor & values, Tensor & indices) const;
  virtual std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k) const;
  virtual std::tuple<Tensor &,Tensor &> topk_out(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, Tensor & values, Tensor & indices) const;
  virtual std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const;
  std::tuple<Tensor &,Tensor &> topk_out(const Tensor & self, int64_t k, int64_t dim, bool largest, Tensor & values, Tensor & indices) const;
  std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim, bool largest) const;
  std::tuple<Tensor &,Tensor &> topk_out(const Tensor & self, int64_t k, int64_t dim, Tensor & values, Tensor & indices) const;
  std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim) const;
  virtual bool m_all(const Tensor & self) const;
  virtual bool m_any(const Tensor & self) const;
  virtual int64_t m_get_device(const Tensor & self) const;
  virtual Tensor & abs_out(const Tensor & self, Tensor & destination) const;
  virtual Tensor abs(const Tensor & self) const;
  virtual Tensor & m_abs_(Tensor & self) const;
  virtual Tensor & m_sigmoid_(Tensor & self) const;
  virtual Tensor & sigmoid_out(const Tensor & self, Tensor & result) const;
  virtual Tensor sigmoid(const Tensor & self) const;
  virtual Tensor & m_log_(Tensor & self) const;
  virtual Tensor & log_out(const Tensor & self, Tensor & result) const;
  virtual Tensor log(const Tensor & self) const;
  virtual Tensor & m_log1p_(Tensor & self) const;
  virtual Tensor & log1p_out(const Tensor & self, Tensor & result) const;
  virtual Tensor log1p(const Tensor & self) const;
  virtual Tensor & lgamma_out(const Tensor & self, Tensor & result) const;
  virtual Tensor lgamma(const Tensor & self) const;
  virtual Tensor & m_lgamma_(Tensor & self) const;
  virtual Tensor & m_exp_(Tensor & self) const;
  virtual Tensor & exp_out(const Tensor & self, Tensor & result) const;
  virtual Tensor exp(const Tensor & self) const;
  virtual Tensor & m_cos_(Tensor & self) const;
  virtual Tensor & cos_out(const Tensor & self, Tensor & result) const;
  virtual Tensor cos(const Tensor & self) const;
  virtual Tensor & m_acos_(Tensor & self) const;
  virtual Tensor & acos_out(const Tensor & self, Tensor & result) const;
  virtual Tensor acos(const Tensor & self) const;
  virtual Tensor & m_cosh_(Tensor & self) const;
  virtual Tensor & cosh_out(const Tensor & self, Tensor & result) const;
  virtual Tensor cosh(const Tensor & self) const;
  virtual Tensor & m_sin_(Tensor & self) const;
  virtual Tensor & sin_out(const Tensor & self, Tensor & result) const;
  virtual Tensor sin(const Tensor & self) const;
  virtual Tensor & m_asin_(Tensor & self) const;
  virtual Tensor & asin_out(const Tensor & self, Tensor & result) const;
  virtual Tensor asin(const Tensor & self) const;
  virtual Tensor & m_sinh_(Tensor & self) const;
  virtual Tensor & sinh_out(const Tensor & self, Tensor & result) const;
  virtual Tensor sinh(const Tensor & self) const;
  virtual Tensor & m_tan_(Tensor & self) const;
  virtual Tensor & tan_out(const Tensor & self, Tensor & result) const;
  virtual Tensor tan(const Tensor & self) const;
  virtual Tensor & m_atan_(Tensor & self) const;
  virtual Tensor & atan_out(const Tensor & self, Tensor & result) const;
  virtual Tensor atan(const Tensor & self) const;
  virtual Tensor & m_tanh_(Tensor & self) const;
  virtual Tensor & tanh_out(const Tensor & self, Tensor & result) const;
  virtual Tensor tanh(const Tensor & self) const;
  virtual Tensor & m_sqrt_(Tensor & self) const;
  virtual Tensor & sqrt_out(const Tensor & self, Tensor & result) const;
  virtual Tensor sqrt(const Tensor & self) const;
  virtual Tensor & m_rsqrt_(Tensor & self) const;
  virtual Tensor & rsqrt_out(const Tensor & self, Tensor & result) const;
  virtual Tensor rsqrt(const Tensor & self) const;
  virtual Tensor & m_ceil_(Tensor & self) const;
  virtual Tensor & ceil_out(const Tensor & self, Tensor & result) const;
  virtual Tensor ceil(const Tensor & self) const;
  virtual Tensor & m_floor_(Tensor & self) const;
  virtual Tensor & floor_out(const Tensor & self, Tensor & result) const;
  virtual Tensor floor(const Tensor & self) const;
  virtual Tensor & m_round_(Tensor & self) const;
  virtual Tensor & round_out(const Tensor & self, Tensor & result) const;
  virtual Tensor round(const Tensor & self) const;
  virtual Tensor & m_trunc_(Tensor & self) const;
  virtual Tensor & trunc_out(const Tensor & self, Tensor & result) const;
  virtual Tensor trunc(const Tensor & self) const;
  virtual Tensor & m_frac_(Tensor & self) const;
  virtual Tensor & frac_out(const Tensor & self, Tensor & result) const;
  virtual Tensor frac(const Tensor & self) const;
  virtual Tensor & mean_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & destination) const;
  virtual Tensor mean(const Tensor & self, int64_t dim, bool keepdim) const;
  virtual Tensor & mean_out(const Tensor & self, int64_t dim, Tensor & destination) const;
  virtual Tensor mean(const Tensor & self, int64_t dim) const;
  virtual Scalar mean(const Tensor & self) const;
  virtual Tensor & var_out(const Tensor & self, int64_t dim, bool unbiased, bool keepdim, Tensor & destination) const;
  virtual Tensor var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const;
  Tensor & var_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & destination) const;
  Tensor var(const Tensor & self, int64_t dim, bool keepdim) const;
  Tensor & var_out(const Tensor & self, int64_t dim, Tensor & destination) const;
  Tensor var(const Tensor & self, int64_t dim) const;
  virtual Scalar var(const Tensor & self, bool unbiased) const;
  Scalar var(const Tensor & self) const;
  virtual Tensor & std_out(const Tensor & self, int64_t dim, bool unbiased, bool keepdim, Tensor & destination) const;
  virtual Tensor std(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const;
  Tensor & std_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & destination) const;
  Tensor std(const Tensor & self, int64_t dim, bool keepdim) const;
  Tensor & std_out(const Tensor & self, int64_t dim, Tensor & destination) const;
  Tensor std(const Tensor & self, int64_t dim) const;
  virtual Scalar std(const Tensor & self, bool unbiased) const;
  Scalar std(const Tensor & self) const;
  virtual Tensor & norm_out(const Tensor & self, Scalar p, int64_t dim, bool keepdim, Tensor & destination) const;
  virtual Tensor norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim) const;
  virtual Tensor & norm_out(const Tensor & self, Scalar p, int64_t dim, Tensor & destination) const;
  virtual Tensor norm(const Tensor & self, Scalar p, int64_t dim) const;
  virtual Scalar norm(const Tensor & self, Scalar p) const;
  Scalar norm(const Tensor & self) const;
  virtual Tensor & renorm_out(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm, Tensor & destination) const;
  virtual Tensor renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const;
  virtual Tensor & m_renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const;
  Scalar dist(const Tensor & self, const Tensor & other, Scalar p) const;
  virtual Scalar s_dist(const Tensor & self, const Tensor & other, Scalar p) const;
  Scalar dist(const Tensor & self, const Tensor & other) const;
  virtual Tensor & reciprocal_out(const Tensor & self, Tensor & destination) const;
  virtual Tensor reciprocal(const Tensor & self) const;
  virtual Tensor & m_reciprocal_(Tensor & self) const;
  virtual Tensor & neg_out(const Tensor & self, Tensor & destination) const;
  virtual Tensor neg(const Tensor & self) const;
  virtual Tensor & m_neg_(Tensor & self) const;
  Tensor & atan2_out(const Tensor & self, const Tensor & other, Tensor & destination) const;
  virtual Tensor & s_atan2_out(const Tensor & self, const Tensor & other, Tensor & destination) const;
  Tensor atan2(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_atan2(const Tensor & self, const Tensor & other) const;
  Tensor & m_atan2_(Tensor & self, const Tensor & other) const;
  virtual Tensor & s_m_atan2_(Tensor & self, const Tensor & other) const;
  virtual Tensor & pow_out(const Tensor & self, Scalar exponent, Tensor & destination) const;
  virtual Tensor pow(const Tensor & self, Scalar exponent) const;
  Tensor & pow_out(const Tensor & self, const Tensor & exponent, Tensor & destination) const;
  virtual Tensor & s_pow_out(const Tensor & self, const Tensor & exponent, Tensor & destination) const;
  Tensor pow(const Tensor & self, const Tensor & exponent) const;
  virtual Tensor s_pow(const Tensor & self, const Tensor & exponent) const;
  virtual Tensor & m_pow_(Tensor & self, Scalar exponent) const;
  Tensor & m_pow_(Tensor & self, const Tensor & exponent) const;
  virtual Tensor & s_m_pow_(Tensor & self, const Tensor & exponent) const;
  Tensor & lerp_out(const Tensor & self, const Tensor & end, Scalar weight, Tensor & destination) const;
  virtual Tensor & s_lerp_out(const Tensor & self, const Tensor & end, Scalar weight, Tensor & destination) const;
  Tensor lerp(const Tensor & self, const Tensor & end, Scalar weight) const;
  virtual Tensor s_lerp(const Tensor & self, const Tensor & end, Scalar weight) const;
  Tensor & m_lerp_(Tensor & self, const Tensor & end, Scalar weight) const;
  virtual Tensor & s_m_lerp_(Tensor & self, const Tensor & end, Scalar weight) const;
  virtual Tensor & linspace_out(Scalar start, Scalar end, int64_t steps, Tensor & result) const;
  virtual Tensor linspace(Scalar start, Scalar end, int64_t steps) const;
  Tensor & linspace_out(Scalar start, Scalar end, Tensor & result) const;
  Tensor linspace(Scalar start, Scalar end) const;
  virtual Tensor & logspace_out(Scalar start, Scalar end, int64_t steps, Tensor & result) const;
  virtual Tensor logspace(Scalar start, Scalar end, int64_t steps) const;
  Tensor & logspace_out(Scalar start, Scalar end, Tensor & result) const;
  Tensor logspace(Scalar start, Scalar end) const;
  virtual Tensor & histc_out(const Tensor & self, Tensor & destination) const;
  virtual Tensor histc(const Tensor & self) const;
  virtual Tensor & histc_out(const Tensor & self, int64_t bins, Tensor & destination) const;
  virtual Tensor histc(const Tensor & self, int64_t bins) const;
  virtual Tensor & histc_out(const Tensor & self, int64_t bins, Scalar min, Tensor & destination) const;
  virtual Tensor histc(const Tensor & self, int64_t bins, Scalar min) const;
  virtual Tensor & histc_out(const Tensor & self, int64_t bins, Scalar min, Scalar max, Tensor & destination) const;
  virtual Tensor histc(const Tensor & self, int64_t bins, Scalar min, Scalar max) const;
  virtual Tensor & m_zero_(Tensor & self) const;
  virtual Tensor & sum_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & result) const;
  virtual Tensor sum(const Tensor & self, int64_t dim, bool keepdim) const;
  virtual Tensor & sum_out(const Tensor & self, int64_t dim, Tensor & result) const;
  virtual Tensor sum(const Tensor & self, int64_t dim) const;
  virtual Scalar sum(const Tensor & self) const;
  virtual Tensor & prod_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & result) const;
  virtual Tensor prod(const Tensor & self, int64_t dim, bool keepdim) const;
  virtual Tensor & prod_out(const Tensor & self, int64_t dim, Tensor & result) const;
  virtual Tensor prod(const Tensor & self, int64_t dim) const;
  virtual Scalar prod(const Tensor & self) const;
  virtual Tensor & cumsum_out(const Tensor & self, int64_t dim, Tensor & result) const;
  virtual Tensor cumsum(const Tensor & self, int64_t dim) const;
  virtual Tensor & cumprod_out(const Tensor & self, int64_t dim, Tensor & result) const;
  virtual Tensor cumprod(const Tensor & self, int64_t dim) const;
  virtual Tensor & sign_out(const Tensor & self, Tensor & result) const;
  virtual Tensor sign(const Tensor & self) const;
  virtual Tensor & m_sign_(Tensor & self) const;
  virtual Scalar trace(const Tensor & self) const;
  Tensor & add_out(const Tensor & self, Scalar value, const Tensor & other, Tensor & result) const;
  virtual Tensor & s_add_out(const Tensor & self, Scalar value, const Tensor & other, Tensor & result) const;
  Tensor add(const Tensor & self, Scalar value, const Tensor & other) const;
  virtual Tensor s_add(const Tensor & self, Scalar value, const Tensor & other) const;
  virtual Tensor & add_out(const Tensor & self, Scalar value, SparseTensor other, Tensor & result) const;
  virtual Tensor add(const Tensor & self, Scalar value, SparseTensor other) const;
  virtual Tensor & add_out(const Tensor & self, Scalar value, Tensor & result) const;
  virtual Tensor add(const Tensor & self, Scalar value) const;
  Tensor & add_out(const Tensor & self, const Tensor & other, Tensor & result) const;
  Tensor add(const Tensor & self, const Tensor & other) const;
  Tensor & add_out(const Tensor & self, SparseTensor other, Tensor & result) const;
  Tensor add(const Tensor & self, SparseTensor other) const;
  Tensor & m_add_(Tensor & self, Scalar value, const Tensor & other) const;
  virtual Tensor & s_m_add_(Tensor & self, Scalar value, const Tensor & other) const;
  virtual Tensor & m_add_(Tensor & self, Scalar value, SparseTensor other) const;
  virtual Tensor & m_add_(Tensor & self, Scalar value) const;
  Tensor & m_add_(Tensor & self, const Tensor & other) const;
  Tensor & m_add_(Tensor & self, SparseTensor other) const;
  Tensor & sub_out(const Tensor & self, Scalar value, const Tensor & other, Tensor & result) const;
  virtual Tensor & s_sub_out(const Tensor & self, Scalar value, const Tensor & other, Tensor & result) const;
  Tensor sub(const Tensor & self, Scalar value, const Tensor & other) const;
  virtual Tensor s_sub(const Tensor & self, Scalar value, const Tensor & other) const;
  virtual Tensor & sub_out(const Tensor & self, Scalar value, Tensor & result) const;
  virtual Tensor sub(const Tensor & self, Scalar value) const;
  Tensor & sub_out(const Tensor & self, const Tensor & other, Tensor & result) const;
  Tensor sub(const Tensor & self, const Tensor & other) const;
  Tensor & m_sub_(Tensor & self, Scalar value, const Tensor & other) const;
  virtual Tensor & s_m_sub_(Tensor & self, Scalar value, const Tensor & other) const;
  virtual Tensor & m_sub_(Tensor & self, Scalar value) const;
  Tensor & m_sub_(Tensor & self, const Tensor & other) const;
  virtual Tensor & mul_out(const Tensor & self, Scalar value, Tensor & result) const;
  virtual Tensor mul(const Tensor & self, Scalar value) const;
  Tensor & mul_out(const Tensor & self, const Tensor & other, Tensor & result) const;
  virtual Tensor & s_mul_out(const Tensor & self, const Tensor & other, Tensor & result) const;
  Tensor mul(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_mul(const Tensor & self, const Tensor & other) const;
  virtual Tensor & m_mul_(Tensor & self, Scalar value) const;
  Tensor & m_mul_(Tensor & self, const Tensor & other) const;
  virtual Tensor & s_m_mul_(Tensor & self, const Tensor & other) const;
  virtual Tensor & div_out(const Tensor & self, Scalar value, Tensor & result) const;
  virtual Tensor div(const Tensor & self, Scalar value) const;
  Tensor & div_out(const Tensor & self, const Tensor & other, Tensor & result) const;
  virtual Tensor & s_div_out(const Tensor & self, const Tensor & other, Tensor & result) const;
  Tensor div(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_div(const Tensor & self, const Tensor & other) const;
  virtual Tensor & m_div_(Tensor & self, Scalar value) const;
  Tensor & m_div_(Tensor & self, const Tensor & other) const;
  virtual Tensor & s_m_div_(Tensor & self, const Tensor & other) const;
  virtual Tensor & fmod_out(const Tensor & self, Scalar value, Tensor & result) const;
  virtual Tensor fmod(const Tensor & self, Scalar value) const;
  Tensor & fmod_out(const Tensor & self, const Tensor & other, Tensor & result) const;
  virtual Tensor & s_fmod_out(const Tensor & self, const Tensor & other, Tensor & result) const;
  Tensor fmod(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_fmod(const Tensor & self, const Tensor & other) const;
  virtual Tensor & m_fmod_(Tensor & self, Scalar value) const;
  Tensor & m_fmod_(Tensor & self, const Tensor & other) const;
  virtual Tensor & s_m_fmod_(Tensor & self, const Tensor & other) const;
  virtual Tensor & remainder_out(const Tensor & self, Scalar value, Tensor & result) const;
  virtual Tensor remainder(const Tensor & self, Scalar value) const;
  Tensor & remainder_out(const Tensor & self, const Tensor & other, Tensor & result) const;
  virtual Tensor & s_remainder_out(const Tensor & self, const Tensor & other, Tensor & result) const;
  Tensor remainder(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_remainder(const Tensor & self, const Tensor & other) const;
  virtual Tensor & m_remainder_(Tensor & self, Scalar value) const;
  Tensor & m_remainder_(Tensor & self, const Tensor & other) const;
  virtual Tensor & s_m_remainder_(Tensor & self, const Tensor & other) const;
  virtual Tensor & clamp_out(const Tensor & self, Scalar min, Scalar max, Tensor & destination) const;
  virtual Tensor clamp(const Tensor & self, Scalar min, Scalar max) const;
  virtual Tensor & clamp_out(const Tensor & self, Scalar min, Tensor & result) const;
  virtual Tensor clamp(const Tensor & self, Scalar min) const;
  virtual Tensor & m_clamp_(Tensor & self, Scalar min, Scalar max) const;
  virtual Tensor & m_clamp_(Tensor & self, Scalar min) const;
  virtual Scalar dot(const Tensor & self, const Tensor & tensor) const;
  virtual Tensor & tril_out(const Tensor & self, int64_t diagonal, Tensor & destination) const;
  virtual Tensor tril(const Tensor & self, int64_t diagonal) const;
  Tensor & tril_out(const Tensor & self, Tensor & destination) const;
  Tensor tril(const Tensor & self) const;
  virtual Tensor & m_tril_(Tensor & self, int64_t diagonal) const;
  Tensor & m_tril_(Tensor & self) const;
  virtual Tensor & triu_out(const Tensor & self, int64_t diagonal, Tensor & destination) const;
  virtual Tensor triu(const Tensor & self, int64_t diagonal) const;
  Tensor & triu_out(const Tensor & self, Tensor & destination) const;
  Tensor triu(const Tensor & self) const;
  virtual Tensor & m_triu_(Tensor & self, int64_t diagonal) const;
  Tensor & m_triu_(Tensor & self) const;
  virtual Tensor & cross_out(const Tensor & self, const Tensor & other, int64_t dim, Tensor & destination) const;
  virtual Tensor cross(const Tensor & self, const Tensor & other, int64_t dim) const;
  Tensor & cross_out(const Tensor & self, const Tensor & other, Tensor & destination) const;
  Tensor cross(const Tensor & self, const Tensor & other) const;
  virtual Tensor & eye_out(int64_t n, Tensor & result) const;
  virtual Tensor eye(int64_t n) const;
  virtual Tensor & eye_out(int64_t n, int64_t m, Tensor & result) const;
  virtual Tensor eye(int64_t n, int64_t m) const;
  virtual Tensor & diag_out(const Tensor & self, int64_t diagonal, Tensor & result) const;
  virtual Tensor diag(const Tensor & self, int64_t diagonal) const;
  Tensor & diag_out(const Tensor & self, Tensor & result) const;
  Tensor diag(const Tensor & self) const;
  Tensor & addmm_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2, Tensor & result) const;
  virtual Tensor & s_addmm_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2, Tensor & result) const;
  Tensor addmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2) const;
  virtual Tensor s_addmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2) const;
  Tensor & addmm_out(Scalar beta, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Tensor & result) const;
  Tensor addmm(Scalar beta, const Tensor & self, const Tensor & mat1, const Tensor & mat2) const;
  Tensor & addmm_out(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Tensor & result) const;
  Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2) const;
  virtual Tensor & m_addmm_(Tensor & self, Scalar beta, Scalar alpha, const Tensor & mat1, const Tensor & mat2) const;
  Tensor & m_addmm_(Tensor & self, Scalar beta, const Tensor & mat1, const Tensor & mat2) const;
  Tensor & m_addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2) const;
  Tensor & addmv_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat, const Tensor & vec, Tensor & result) const;
  virtual Tensor & s_addmv_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat, const Tensor & vec, Tensor & result) const;
  Tensor addmv(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat, const Tensor & vec) const;
  virtual Tensor s_addmv(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat, const Tensor & vec) const;
  Tensor & addmv_out(Scalar beta, const Tensor & self, const Tensor & mat, const Tensor & vec, Tensor & result) const;
  Tensor addmv(Scalar beta, const Tensor & self, const Tensor & mat, const Tensor & vec) const;
  Tensor & addmv_out(const Tensor & self, const Tensor & mat, const Tensor & vec, Tensor & result) const;
  Tensor addmv(const Tensor & self, const Tensor & mat, const Tensor & vec) const;
  virtual Tensor & m_addmv_(Tensor & self, Scalar beta, Scalar alpha, const Tensor & mat, const Tensor & vec) const;
  Tensor & m_addmv_(Tensor & self, Scalar beta, const Tensor & mat, const Tensor & vec) const;
  Tensor & m_addmv_(Tensor & self, const Tensor & mat, const Tensor & vec) const;
  Tensor & addr_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & vec1, const Tensor & vec2, Tensor & result) const;
  virtual Tensor & s_addr_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & vec1, const Tensor & vec2, Tensor & result) const;
  Tensor addr(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & vec1, const Tensor & vec2) const;
  virtual Tensor s_addr(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & vec1, const Tensor & vec2) const;
  Tensor & addr_out(Scalar beta, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Tensor & result) const;
  Tensor addr(Scalar beta, const Tensor & self, const Tensor & vec1, const Tensor & vec2) const;
  Tensor & addr_out(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Tensor & result) const;
  Tensor addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2) const;
  virtual Tensor & m_addr_(Tensor & self, Scalar beta, Scalar alpha, const Tensor & vec1, const Tensor & vec2) const;
  Tensor & m_addr_(Tensor & self, Scalar beta, const Tensor & vec1, const Tensor & vec2) const;
  Tensor & m_addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2) const;
  virtual Tensor & ger_out(const Tensor & self, const Tensor & vec2, Tensor & result) const;
  virtual Tensor ger(const Tensor & self, const Tensor & vec2) const;
  virtual Tensor & mv_out(const Tensor & self, const Tensor & vec, Tensor & result) const;
  virtual Tensor mv(const Tensor & self, const Tensor & vec) const;
  virtual Tensor & mm_out(const Tensor & self, const Tensor & mat2, Tensor & result) const;
  virtual Tensor mm(const Tensor & self, const Tensor & mat2) const;
  virtual Tensor & bmm_out(const Tensor & self, const Tensor & mat2, Tensor & result) const;
  virtual Tensor bmm(const Tensor & self, const Tensor & mat2) const;
  Tensor & addbmm_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2, Tensor & result) const;
  virtual Tensor & s_addbmm_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2, Tensor & result) const;
  Tensor addbmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) const;
  virtual Tensor s_addbmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) const;
  Tensor & addbmm_out(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor & result) const;
  Tensor addbmm(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2) const;
  Tensor & addbmm_out(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor & result) const;
  Tensor addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2) const;
  virtual Tensor & m_addbmm_(Tensor & self, Scalar beta, Scalar alpha, const Tensor & batch1, const Tensor & batch2) const;
  Tensor & m_addbmm_(Tensor & self, Scalar beta, const Tensor & batch1, const Tensor & batch2) const;
  Tensor & m_addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2) const;
  Tensor & baddbmm_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2, Tensor & result) const;
  virtual Tensor & s_baddbmm_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2, Tensor & result) const;
  Tensor baddbmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) const;
  virtual Tensor s_baddbmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) const;
  Tensor & baddbmm_out(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor & result) const;
  Tensor baddbmm(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2) const;
  Tensor & baddbmm_out(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor & result) const;
  Tensor baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2) const;
  virtual Tensor & m_baddbmm_(Tensor & self, Scalar beta, Scalar alpha, const Tensor & batch1, const Tensor & batch2) const;
  Tensor & m_baddbmm_(Tensor & self, Scalar beta, const Tensor & batch1, const Tensor & batch2) const;
  Tensor & m_baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2) const;
  Tensor & addcmul_out(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2, Tensor & result) const;
  virtual Tensor & s_addcmul_out(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2, Tensor & result) const;
  Tensor addcmul(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) const;
  virtual Tensor s_addcmul(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) const;
  Tensor & addcmul_out(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Tensor & result) const;
  Tensor addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2) const;
  Tensor & m_addcmul_(Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) const;
  virtual Tensor & s_m_addcmul_(Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) const;
  Tensor & m_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2) const;
  Tensor & addcdiv_out(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2, Tensor & result) const;
  virtual Tensor & s_addcdiv_out(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2, Tensor & result) const;
  Tensor addcdiv(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) const;
  virtual Tensor s_addcdiv(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) const;
  Tensor & addcdiv_out(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Tensor & result) const;
  Tensor addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2) const;
  Tensor & m_addcdiv_(Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) const;
  virtual Tensor & s_m_addcdiv_(Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) const;
  Tensor & m_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2) const;
  virtual std::tuple<Tensor &,Tensor &> gesv_out(const Tensor & self, const Tensor & A, Tensor & solution, Tensor & lu) const;
  virtual std::tuple<Tensor,Tensor> gesv(const Tensor & self, const Tensor & A) const;
  virtual std::tuple<Tensor &,Tensor &> gels_out(const Tensor & self, const Tensor & A, Tensor & res1, Tensor & res2) const;
  virtual std::tuple<Tensor,Tensor> gels(const Tensor & self, const Tensor & A) const;
  virtual std::tuple<Tensor &,Tensor &> trtrs_out(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular, Tensor & res1, Tensor & res2) const;
  virtual std::tuple<Tensor,Tensor> trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) const;
  std::tuple<Tensor &,Tensor &> trtrs_out(const Tensor & self, const Tensor & A, bool upper, bool transpose, Tensor & res1, Tensor & res2) const;
  std::tuple<Tensor,Tensor> trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose) const;
  std::tuple<Tensor &,Tensor &> trtrs_out(const Tensor & self, const Tensor & A, bool upper, Tensor & res1, Tensor & res2) const;
  std::tuple<Tensor,Tensor> trtrs(const Tensor & self, const Tensor & A, bool upper) const;
  std::tuple<Tensor &,Tensor &> trtrs_out(const Tensor & self, const Tensor & A, Tensor & res1, Tensor & res2) const;
  std::tuple<Tensor,Tensor> trtrs(const Tensor & self, const Tensor & A) const;
  virtual std::tuple<Tensor &,Tensor &> symeig_out(const Tensor & self, bool eigenvectors, bool upper, Tensor & res1, Tensor & res2) const;
  virtual std::tuple<Tensor,Tensor> symeig(const Tensor & self, bool eigenvectors, bool upper) const;
  std::tuple<Tensor &,Tensor &> symeig_out(const Tensor & self, bool eigenvectors, Tensor & res1, Tensor & res2) const;
  std::tuple<Tensor,Tensor> symeig(const Tensor & self, bool eigenvectors) const;
  std::tuple<Tensor &,Tensor &> symeig_out(const Tensor & self, Tensor & res1, Tensor & res2) const;
  std::tuple<Tensor,Tensor> symeig(const Tensor & self) const;
  virtual std::tuple<Tensor &,Tensor &> eig_out(const Tensor & self, bool eigenvectors, Tensor & res1, Tensor & res2) const;
  virtual std::tuple<Tensor,Tensor> eig(const Tensor & self, bool eigenvectors) const;
  std::tuple<Tensor &,Tensor &> eig_out(const Tensor & self, Tensor & res1, Tensor & res2) const;
  std::tuple<Tensor,Tensor> eig(const Tensor & self) const;
  virtual std::tuple<Tensor &,Tensor &,Tensor &> svd_out(const Tensor & self, bool some, Tensor & res1, Tensor & res2, Tensor & res3) const;
  virtual std::tuple<Tensor,Tensor,Tensor> svd(const Tensor & self, bool some) const;
  std::tuple<Tensor &,Tensor &,Tensor &> svd_out(const Tensor & self, Tensor & res1, Tensor & res2, Tensor & res3) const;
  std::tuple<Tensor,Tensor,Tensor> svd(const Tensor & self) const;
  virtual Tensor & inverse_out(const Tensor & self, Tensor & output) const;
  virtual Tensor inverse(const Tensor & self) const;
  virtual Tensor & potrf_out(const Tensor & self, bool upper, Tensor & output) const;
  virtual Tensor potrf(const Tensor & self, bool upper) const;
  Tensor & potrf_out(const Tensor & self, Tensor & output) const;
  Tensor potrf(const Tensor & self) const;
  virtual Tensor & potrs_out(const Tensor & self, const Tensor & input2, bool upper, Tensor & result) const;
  virtual Tensor potrs(const Tensor & self, const Tensor & input2, bool upper) const;
  Tensor & potrs_out(const Tensor & self, const Tensor & input2, Tensor & result) const;
  Tensor potrs(const Tensor & self, const Tensor & input2) const;
  virtual Tensor & potri_out(const Tensor & self, bool upper, Tensor & output) const;
  virtual Tensor potri(const Tensor & self, bool upper) const;
  Tensor & potri_out(const Tensor & self, Tensor & output) const;
  Tensor potri(const Tensor & self) const;
  virtual std::tuple<Tensor &,Tensor &> pstrf_out(const Tensor & self, bool upper, Scalar tol, Tensor & res1, Tensor & res2) const;
  virtual std::tuple<Tensor,Tensor> pstrf(const Tensor & self, bool upper, Scalar tol) const;
  std::tuple<Tensor &,Tensor &> pstrf_out(const Tensor & self, bool upper, Tensor & res1, Tensor & res2) const;
  std::tuple<Tensor,Tensor> pstrf(const Tensor & self, bool upper) const;
  std::tuple<Tensor &,Tensor &> pstrf_out(const Tensor & self, Scalar tol, Tensor & res1, Tensor & res2) const;
  std::tuple<Tensor,Tensor> pstrf(const Tensor & self, Scalar tol) const;
  std::tuple<Tensor &,Tensor &> pstrf_out(const Tensor & self, Tensor & res1, Tensor & res2) const;
  std::tuple<Tensor,Tensor> pstrf(const Tensor & self) const;
  virtual std::tuple<Tensor &,Tensor &> qr_out(const Tensor & self, Tensor & res1, Tensor & res2) const;
  virtual std::tuple<Tensor,Tensor> qr(const Tensor & self) const;
  virtual std::tuple<Tensor &,Tensor &> geqrf_out(const Tensor & self, Tensor & res1, Tensor & res2) const;
  virtual std::tuple<Tensor,Tensor> geqrf(const Tensor & self) const;
  virtual std::tuple<Tensor &,const Tensor &> orgqr_out(const Tensor & self, const Tensor & input2, Tensor & result) const;
  virtual std::tuple<Tensor,const Tensor &> orgqr(const Tensor & self, const Tensor & input2) const;
  virtual std::tuple<Tensor &,const Tensor &> ormqr_out(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose, Tensor & result) const;
  virtual std::tuple<Tensor,const Tensor &> ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) const;
  std::tuple<Tensor &,const Tensor &> ormqr_out(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, Tensor & result) const;
  std::tuple<Tensor,const Tensor &> ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left) const;
  std::tuple<Tensor &,const Tensor &> ormqr_out(const Tensor & self, const Tensor & input2, const Tensor & input3, Tensor & result) const;
  std::tuple<Tensor,const Tensor &> ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3) const;
  virtual std::tuple<Tensor &,Tensor &> btrifact_out(const Tensor & info, bool pivot, const Tensor & self, Tensor & result, Tensor & pivots) const;
  virtual std::tuple<Tensor,Tensor> btrifact(const Tensor & info, bool pivot, const Tensor & self) const;
  std::tuple<Tensor &,Tensor &> btrifact_out(const Tensor & info, const Tensor & self, Tensor & result, Tensor & pivots) const;
  std::tuple<Tensor,Tensor> btrifact(const Tensor & info, const Tensor & self) const;
  std::tuple<Tensor &,Tensor &> btrifact_out(bool pivot, const Tensor & self, Tensor & result, Tensor & pivots) const;
  std::tuple<Tensor,Tensor> btrifact(bool pivot, const Tensor & self) const;
  std::tuple<Tensor &,Tensor &> btrifact_out(const Tensor & self, Tensor & result, Tensor & pivots) const;
  std::tuple<Tensor,Tensor> btrifact(const Tensor & self) const;
  virtual Tensor & btrisolve_out(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots, Tensor & result) const;
  virtual Tensor btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const;
  virtual Tensor & randperm_out(Generator & generator, int64_t n, Tensor & result) const;
  virtual Tensor randperm(Generator & generator, int64_t n) const;
  Tensor & randperm_out(int64_t n, Tensor & result) const;
  Tensor randperm(int64_t n) const;
  virtual Tensor & multinomial_out(Generator & generator, const Tensor & self, int64_t num_samples, bool replacement, Tensor & result) const;
  virtual Tensor multinomial(Generator & generator, const Tensor & self, int64_t num_samples, bool replacement) const;
  Tensor & multinomial_out(Generator & generator, const Tensor & self, int64_t num_samples, Tensor & result) const;
  Tensor multinomial(Generator & generator, const Tensor & self, int64_t num_samples) const;
  Tensor & multinomial_out(const Tensor & self, int64_t num_samples, bool replacement, Tensor & result) const;
  Tensor multinomial(const Tensor & self, int64_t num_samples, bool replacement) const;
  Tensor & multinomial_out(const Tensor & self, int64_t num_samples, Tensor & result) const;
  Tensor multinomial(const Tensor & self, int64_t num_samples) const;
  virtual Tensor & m_uniform_(Tensor & self, Generator & generator, double from, double to) const;
  Tensor & m_uniform_(Tensor & self, Generator & generator, double from) const;
  Tensor & m_uniform_(Tensor & self, double from, double to) const;
  Tensor & m_uniform_(Tensor & self, Generator & generator) const;
  Tensor & m_uniform_(Tensor & self, double from) const;
  Tensor & m_uniform_(Tensor & self) const;
  virtual Tensor & m_cauchy_(Tensor & self, Generator & generator, double median, double sigma) const;
  Tensor & m_cauchy_(Tensor & self, Generator & generator, double median) const;
  Tensor & m_cauchy_(Tensor & self, double median, double sigma) const;
  Tensor & m_cauchy_(Tensor & self, Generator & generator) const;
  Tensor & m_cauchy_(Tensor & self, double median) const;
  Tensor & m_cauchy_(Tensor & self) const;
  virtual Tensor & m_log_normal_(Tensor & self, Generator & generator, double mean, double std) const;
  Tensor & m_log_normal_(Tensor & self, Generator & generator, double mean) const;
  Tensor & m_log_normal_(Tensor & self, double mean, double std) const;
  Tensor & m_log_normal_(Tensor & self, Generator & generator) const;
  Tensor & m_log_normal_(Tensor & self, double mean) const;
  Tensor & m_log_normal_(Tensor & self) const;
  virtual Tensor & rand_out(Generator & generator, IntList size, Tensor & result) const;
  virtual Tensor rand(Generator & generator, IntList size) const;
  Tensor & rand_out(IntList size, Tensor & result) const;
  Tensor rand(IntList size) const;
  virtual Tensor & randn_out(Generator & generator, IntList size, Tensor & result) const;
  virtual Tensor randn(Generator & generator, IntList size) const;
  Tensor & randn_out(IntList size, Tensor & result) const;
  Tensor randn(IntList size) const;
  virtual Tensor & m_geometric_(Tensor & self, Generator & generator, double p) const;
  Tensor & m_geometric_(Tensor & self, double p) const;
  virtual int64_t m_size(const Tensor & self, int64_t dim) const;
  virtual int64_t m_stride(const Tensor & self, int64_t dim) const;
  virtual Tensor tensor(Storage & storage, int64_t storageOffset, IntList size, IntList stride) const;
  Tensor tensor(Storage & storage, int64_t storageOffset, IntList size) const;
  virtual Tensor tensor(IntList size, IntList stride) const;
  Tensor tensor(IntList size) const;
  virtual Tensor tensor() const;
  virtual Tensor & select_out(const Tensor & self, int dim, int64_t sliceIndex, Tensor & result) const;
  virtual Tensor select(const Tensor & self, int dim, int64_t sliceIndex) const;
  virtual Tensor & m_assign_(Tensor & self, const Tensor & src) const;
  virtual Tensor & cat_out(TensorList tensors, int dim, Tensor & self) const;
  virtual Tensor cat(TensorList tensors, int dim) const;
  virtual void Abs_updateOutput(const Tensor & input, const Tensor & output) const;
  virtual void Abs_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput) const;
  virtual void AbsCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage) const;
  virtual void AbsCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage) const;
  virtual void BCECriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & weights) const;
  void BCECriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage) const;
  virtual void BCECriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & weights) const;
  void BCECriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage) const;
  virtual void ClassNLLCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & weights, const Tensor & total_weight, int64_t ignore_index) const;
  void ClassNLLCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & total_weight, int64_t ignore_index) const;
  virtual void ClassNLLCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & weights, const Tensor & total_weight, int64_t ignore_index) const;
  void ClassNLLCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & total_weight, int64_t ignore_index) const;
  virtual void SpatialClassNLLCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & weights, const Tensor & total_weight, int64_t ignore_index) const;
  void SpatialClassNLLCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & total_weight, int64_t ignore_index) const;
  virtual void SpatialClassNLLCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & weights, const Tensor & total_weight, int64_t ignore_index) const;
  void SpatialClassNLLCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & total_weight, int64_t ignore_index) const;
  virtual void ELU_updateOutput(const Tensor & input, const Tensor & output, Scalar alpha, bool inplace) const;
  virtual void ELU_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output, Scalar alpha, bool inplace) const;
  virtual void DistKLDivCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage) const;
  virtual void DistKLDivCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage) const;
  virtual void GatedLinear_updateOutput(const Tensor & input, const Tensor & output, int dim) const;
  virtual void GatedLinear_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int dim) const;
  virtual void HardShrink_updateOutput(const Tensor & input, const Tensor & output, Scalar lambda) const;
  virtual void HardShrink_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar lambda) const;
  virtual void HardTanh_updateOutput(const Tensor & input, const Tensor & output, Scalar min_val, Scalar max_val, bool inplace) const;
  virtual void HardTanh_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar min_val, Scalar max_val, bool inplace) const;
  virtual void L1Cost_updateOutput(const Tensor & input, const Tensor & output) const;
  virtual void L1Cost_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput) const;
  void L1Cost_updateGradInput(const Tensor & input, const Tensor & gradInput) const;
  virtual void LeakyReLU_updateOutput(const Tensor & input, const Tensor & output, Scalar negval, bool inplace) const;
  virtual void LeakyReLU_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar negval, bool inplace) const;
  virtual void GRUFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & bias1, const Tensor & bias2, const Tensor & hx, const Tensor & output, const Tensor & storage) const;
  void GRUFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & bias1, const Tensor & hx, const Tensor & output, const Tensor & storage) const;
  void GRUFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & hx, const Tensor & output, const Tensor & storage) const;
  virtual void GRUFused_updateGradInput(const Tensor & gradInInput, const Tensor & gradInHidden, const Tensor & gradOutput, const Tensor & gradInputHx, const Tensor & storage) const;
  virtual void LSTMFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & bias1, const Tensor & bias2, const Tensor & cell, const Tensor & output, const Tensor & outputCell) const;
  void LSTMFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & bias1, const Tensor & cell, const Tensor & output, const Tensor & outputCell) const;
  void LSTMFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & cell, const Tensor & output, const Tensor & outputCell) const;
  virtual void LSTMFused_updateGradInput(const Tensor & storage, const Tensor & gradInGates, const Tensor & cx, const Tensor & cy, const Tensor & gradOutput, const Tensor & gradOutputCell, const Tensor & gradInputCx) const;
  virtual void LogSigmoid_updateOutput(const Tensor & input, const Tensor & output, const Tensor & buffer) const;
  virtual void LogSigmoid_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & buffer) const;
  virtual void LogSoftMax_updateOutput(const Tensor & input, const Tensor & output) const;
  virtual void LogSoftMax_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) const;
  virtual void MarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, Scalar margin) const;
  virtual void MarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, Scalar margin) const;
  virtual void SoftMarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage) const;
  virtual void SoftMarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage) const;
  virtual void MSECriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage) const;
  virtual void MSECriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage) const;
  virtual void MultiLabelMarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, const Tensor & isTarget, bool sizeAverage) const;
  virtual void MultiLabelMarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, const Tensor & isTarget, bool sizeAverage) const;
  virtual void MultiMarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, int p, const Tensor & weights, Scalar margin) const;
  void MultiMarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, int p, Scalar margin) const;
  virtual void MultiMarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, int p, const Tensor & weights, Scalar margin) const;
  void MultiMarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, int p, Scalar margin) const;
  virtual void PReLU_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, int64_t nOutputPlane) const;
  virtual void PReLU_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, int64_t nOutputPlane) const;
  virtual void PReLU_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & gradWeight, const Tensor & gradWeightBuf, const Tensor & gradWeightBuf2, int64_t nOutputPlane, Scalar scale) const;
  virtual void Linear_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & addBuffer) const;
  virtual void Linear_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight) const;
  virtual void Linear_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & bias, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & addBuffer, Scalar scale) const;
  virtual void RReLU_updateOutput(const Tensor & input, const Tensor & output, const Tensor & noise, Scalar lower, Scalar upper, bool train, bool inplace, Generator & generator) const;
  virtual void RReLU_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & noise, Scalar lower, Scalar upper, bool train, bool inplace) const;
  virtual void Sigmoid_updateOutput(const Tensor & input, const Tensor & output) const;
  virtual void Sigmoid_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) const;
  void Sigmoid_updateGradInput(const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) const;
  virtual void SmoothL1Criterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage) const;
  virtual void SmoothL1Criterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage) const;
  virtual void SoftMax_updateOutput(const Tensor & input, const Tensor & output) const;
  virtual void SoftMax_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) const;
  virtual void SoftPlus_updateOutput(const Tensor & input, const Tensor & output, Scalar beta, Scalar threshold) const;
  virtual void SoftPlus_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output, Scalar beta, Scalar threshold) const;
  virtual void SoftShrink_updateOutput(const Tensor & input, const Tensor & output, Scalar lambda) const;
  virtual void SoftShrink_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar lambda) const;
  virtual void IndexLinear_updateOutput(const Tensor & keys, int64_t keysOffset, const Tensor & values, const Tensor & sizes, const Tensor & cumSumSizes, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & normalizedValues, int train) const;
  virtual void IndexLinear_accGradParameters(const Tensor & keys, int64_t keysOffset, const Tensor & values, const Tensor & sizes, const Tensor & cumSumSizes, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & weight, const Tensor & bias, const Tensor & valuesBuffer, Scalar weightDecay, Scalar scale) const;
  virtual void SparseLinear_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias) const;
  virtual void SparseLinear_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & weight, const Tensor & bias, Scalar weightDecay, Scalar scale) const;
  virtual void Sqrt_updateOutput(const Tensor & input, const Tensor & output, Scalar eps) const;
  virtual void Sqrt_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) const;
  virtual void Square_updateOutput(const Tensor & input, const Tensor & output) const;
  virtual void Square_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput) const;
  virtual void Tanh_updateOutput(const Tensor & input, const Tensor & output) const;
  virtual void Tanh_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) const;
  void Tanh_updateGradInput(const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) const;
  virtual void Threshold_updateOutput(const Tensor & input, const Tensor & output, Scalar threshold, Scalar val, bool inplace) const;
  virtual void Threshold_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar threshold, Scalar val, bool inplace) const;
  virtual void TemporalConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, int kW, int dW, int inputFrameSize, int outputFrameSize) const;
  virtual void TemporalConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, int kW, int dW) const;
  virtual void TemporalConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, int kW, int dW, Scalar scale) const;
  virtual void TemporalMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kW, int dW) const;
  virtual void TemporalMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kW, int dW) const;
  virtual void TemporalSubSampling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, int kW, int dW, int inputFrameSize) const;
  virtual void TemporalSubSampling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, int kW, int dW) const;
  virtual void TemporalSubSampling_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, int kW, int dW, Scalar scale) const;
  virtual void TemporalRowConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int kW, int dW, int padW, bool featFirst) const;
  virtual void TemporalRowConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int dW, int padW, bool featFirst) const;
  virtual void TemporalRowConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int kW, int dW, int padW, bool featFirst, Scalar scale) const;
  virtual void BatchNormalization_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double momentum, double eps) const;
  void BatchNormalization_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double momentum, double eps) const;
  void BatchNormalization_updateOutput(const Tensor & input, const Tensor & output, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double momentum, double eps) const;
  virtual void BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps) const;
  void BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps) const;
  void BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & gradWeight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps) const;
  void BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps) const;
  void BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps) const;
  virtual void SpatialConvolutionMap_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH) const;
  virtual void SpatialConvolutionMap_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & bias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH) const;
  virtual void SpatialConvolutionMap_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH, Scalar scale) const;
  virtual void SpatialConvolutionMM_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH) const;
  void SpatialConvolutionMM_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH) const;
  virtual void SpatialConvolutionMM_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH) const;
  virtual void SpatialConvolutionMM_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, Scalar scale) const;
  void SpatialConvolutionMM_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, Scalar scale) const;
  virtual void SpatialDepthWiseConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH) const;
  void SpatialDepthWiseConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH) const;
  virtual void SpatialDepthWiseConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH) const;
  virtual void SpatialDepthWiseConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, Scalar scale) const;
  void SpatialDepthWiseConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, Scalar scale) const;
  virtual void SpatialConvolutionLocal_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, int64_t inputWidth, int64_t inputHeight, int64_t outputWidth, int64_t outputHeight) const;
  virtual void SpatialConvolutionLocal_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, int64_t inputWidth, int64_t inputHeight, int64_t outputWidth, int64_t outputHeight) const;
  virtual void SpatialConvolutionLocal_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, int64_t inputWidth, int64_t inputHeight, int64_t outputWidth, int64_t outputHeight, Scalar scale) const;
  virtual void SpatialAdaptiveMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int owidth, int oheight) const;
  virtual void SpatialAdaptiveMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices) const;
  virtual void SpatialAdaptiveAveragePooling_updateOutput(const Tensor & input, const Tensor & output, int owidth, int oheight) const;
  virtual void SpatialAdaptiveAveragePooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput) const;
  virtual void SpatialAveragePooling_updateOutput(const Tensor & input, const Tensor & output, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad) const;
  virtual void SpatialAveragePooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad) const;
  virtual void SpatialFractionalMaxPooling_updateOutput(const Tensor & input, const Tensor & output, int outputW, int outputH, int poolSizeW, int poolSizeH, const Tensor & indices, const Tensor & randomSamples) const;
  virtual void SpatialFractionalMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int outputW, int outputH, int poolSizeW, int poolSizeH, const Tensor & indices) const;
  virtual void SpatialFullConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH) const;
  void SpatialFullConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH) const;
  virtual void SpatialFullConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & gradColumns, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH) const;
  virtual void SpatialFullConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH, Scalar scale) const;
  void SpatialFullConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH, Scalar scale) const;
  virtual void SpatialFullConvolutionMap_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH) const;
  virtual void SpatialFullConvolutionMap_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & bias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH) const;
  virtual void SpatialFullConvolutionMap_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH, Scalar scale) const;
  virtual void SpatialDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH) const;
  void SpatialDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH) const;
  virtual void SpatialDilatedConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & gradColumns, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH) const;
  virtual void SpatialDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, Scalar scale) const;
  void SpatialDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, Scalar scale) const;
  virtual void SpatialFullDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH) const;
  void SpatialFullDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH) const;
  virtual void SpatialFullDilatedConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & gradColumns, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH) const;
  virtual void SpatialFullDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH, Scalar scale) const;
  void SpatialFullDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH, Scalar scale) const;
  virtual void SpatialMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode) const;
  virtual void SpatialMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode) const;
  virtual void SpatialDilatedMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, bool ceil_mode) const;
  virtual void SpatialDilatedMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, bool ceil_mode) const;
  virtual void SpatialMaxUnpooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int owidth, int oheight) const;
  virtual void SpatialMaxUnpooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int owidth, int oheight) const;
  virtual void SpatialSubSampling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, int kW, int kH, int dW, int dH) const;
  virtual void SpatialSubSampling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, int kW, int kH, int dW, int dH) const;
  virtual void SpatialSubSampling_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, int kW, int kH, int dW, int dH, Scalar scale) const;
  virtual void SpatialUpSamplingNearest_updateOutput(const Tensor & input, const Tensor & output, int scale_factor) const;
  virtual void SpatialUpSamplingNearest_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int scale_factor) const;
  virtual void SpatialUpSamplingBilinear_updateOutput(const Tensor & input, const Tensor & output, int outputHeight, int outputWidth) const;
  virtual void SpatialUpSamplingBilinear_updateGradInput(const Tensor & gradOutput, const Tensor & gradInput, int nbatch, int nchannels, int inputHeight, int inputWidth, int outputHeight, int outputWidth) const;
  virtual void SpatialGridSamplerBilinear_updateOutput(const Tensor & input, const Tensor & grid, const Tensor & output) const;
  virtual void SpatialGridSamplerBilinear_updateGradInput(const Tensor & input, const Tensor & gradInput, const Tensor & grid, const Tensor & gradGrid, const Tensor & gradOutput) const;
  virtual void VolumetricAveragePooling_updateOutput(const Tensor & input, const Tensor & output, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, bool ceil_mode, bool count_include_pad) const;
  virtual void VolumetricAveragePooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, bool ceil_mode, bool count_include_pad) const;
  virtual void VolumetricConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH) const;
  void VolumetricConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH) const;
  virtual void VolumetricConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, int dT, int dW, int dH, int pT, int pW, int pH) const;
  virtual void VolumetricConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, Scalar scale) const;
  void VolumetricConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, Scalar scale) const;
  virtual void VolumetricConvolutionMM_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH) const;
  void VolumetricConvolutionMM_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH) const;
  virtual void VolumetricConvolutionMM_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH) const;
  virtual void VolumetricConvolutionMM_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, Scalar scale) const;
  void VolumetricConvolutionMM_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, Scalar scale) const;
  virtual void VolumetricFractionalMaxPooling_updateOutput(const Tensor & input, const Tensor & output, int outputT, int outputW, int outputH, int poolSizeT, int poolSizeW, int poolSizeH, const Tensor & indices, const Tensor & randomSamples) const;
  virtual void VolumetricFractionalMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int outputT, int outputW, int outputH, int poolSizeT, int poolSizeW, int poolSizeH, const Tensor & indices) const;
  virtual void VolumetricFullConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH) const;
  void VolumetricFullConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH) const;
  virtual void VolumetricFullConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH) const;
  virtual void VolumetricFullConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH, Scalar scale) const;
  void VolumetricFullConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH, Scalar scale) const;
  virtual void VolumetricDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & columns, const Tensor & ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH) const;
  void VolumetricDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & columns, const Tensor & ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH) const;
  virtual void VolumetricDilatedConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & gradColumns, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH) const;
  virtual void VolumetricDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & columns, const Tensor & ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH, Scalar scale) const;
  void VolumetricDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & columns, const Tensor & ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH, Scalar scale) const;
  virtual void VolumetricFullDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH) const;
  void VolumetricFullDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH) const;
  virtual void VolumetricFullDilatedConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH) const;
  virtual void VolumetricFullDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH, Scalar scale) const;
  void VolumetricFullDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH, Scalar scale) const;
  virtual void VolumetricMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, bool ceilMode) const;
  virtual void VolumetricMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, bool ceilMode) const;
  virtual void VolumetricDilatedMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, bool ceilMode) const;
  virtual void VolumetricDilatedMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, bool ceilMode) const;
  virtual void VolumetricMaxUnpooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int oT, int oW, int oH, int dT, int dW, int dH, int pT, int pW, int pH) const;
  virtual void VolumetricMaxUnpooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int oT, int oW, int oH, int dT, int dW, int dH, int pT, int pW, int pH) const;
  virtual void SpatialReflectionPadding_updateOutput(const Tensor & input, const Tensor & output, int pad_l, int pad_r, int pad_t, int pad_b) const;
  virtual void SpatialReflectionPadding_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int pad_l, int pad_r, int pad_t, int pad_b) const;
  virtual void SpatialReplicationPadding_updateOutput(const Tensor & input, const Tensor & output, int pad_l, int pad_r, int pad_t, int pad_b) const;
  virtual void SpatialReplicationPadding_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int pad_l, int pad_r, int pad_t, int pad_b) const;
  virtual void FeatureLPPooling_updateOutput(const Tensor & input, const Tensor & output, Scalar power, int width, int stride, bool batchMode) const;
  virtual void FeatureLPPooling_updateGradInput(const Tensor & gradOutput, const Tensor & input, const Tensor & output, const Tensor & gradInput, Scalar power, int width, int stride, bool batchMode) const;
  virtual void VolumetricReplicationPadding_updateOutput(const Tensor & input, const Tensor & output, int pleft, int pright, int ptop, int pbottom, int pfront, int pback) const;
  virtual void VolumetricReplicationPadding_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int pleft, int pright, int ptop, int pbottom, int pfront, int pback) const;
  virtual void VolumetricUpSamplingNearest_updateOutput(const Tensor & input, const Tensor & output, int scale_factor) const;
  virtual void VolumetricUpSamplingNearest_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int scale_factor) const;
  virtual void VolumetricUpSamplingTrilinear_updateOutput(const Tensor & input, const Tensor & output, int outputDepth, int outputHeight, int outputWidth) const;
  virtual void VolumetricUpSamplingTrilinear_updateGradInput(const Tensor & gradOutput, const Tensor & gradInput, int nbatch, int nchannels, int inputDepth, int inputHeight, int inputWidth, int outputDepth, int outputHeight, int outputWidth) const;
  virtual void SpatialCrossMapLRN_updateOutput(const Tensor & input, const Tensor & output, const Tensor & scale, int size, Scalar alpha, Scalar beta, Scalar k) const;
  virtual void SpatialCrossMapLRN_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & scale, const Tensor & output, int size, Scalar alpha, Scalar beta, Scalar k) const;
protected:
  Context* context;
};


}
