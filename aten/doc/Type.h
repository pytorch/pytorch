#pragma once

#include <memory>
#include <limits>
#include <functional>

#include "ATen/ATenGeneral.h"
#include "ATen/ArrayRef.h"
#include "ATen/Generator.h"
#include "ATen/Half.h"
#include "ATen/SparseTensorRef.h"
#include "ATen/ScalarType.h"
#include "ATen/Scalar.h"
#include "ATen/Tensor.h"

// To solve the conflict of s_addr in inaddr.h
#ifdef _MSC_VER
#ifdef s_addr
#undef s_addr
#endif
#endif

namespace at {

class Context;
struct Storage;
struct Generator;

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
  NumOptions
};


struct AT_API Type {
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
  virtual std::unique_ptr<Storage> storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter=noop_deleter) const = 0;
  virtual std::unique_ptr<Generator> generator() const = 0;
  virtual Tensor unsafeTensorFromTH(void * th_pointer, bool retain) const = 0;
  virtual const char * toString() const = 0;
  virtual std::size_t elementSizeInBytes() const = 0;
  Type & toBackend(Backend b) const;
  Type & toScalarType(ScalarType s) const;

  // contingious IDs for all types in the system
  // for external dispatch
  virtual TypeID ID() const = 0;

  Tensor copy(const Tensor & src) const;
  void copy(const Tensor & src, Tensor & dst) const;
  virtual void s_copy(const Tensor & src, Tensor & dst) const = 0;

  Tensor tensorFromBlob(void * data, IntList sizes, const std::function<void(void*)> & deleter=noop_deleter);
  Tensor tensorFromBlob(void * data, IntList sizes, IntList strides, const std::function<void(void*)> & deleter=noop_deleter);
  Tensor scalarTensor(Scalar s) const;

  bool operator==(const Type& other) const;

  // example
  // virtual Tensor * add(Tensor & a, Tensor & b) = 0;
  virtual int64_t m_storage_offset(const Tensor & self) const;
  virtual Tensor & m_resize_(Tensor & self, IntList size) const;
  virtual Tensor & zeros_out(Tensor & result, IntList size) const;
  virtual Tensor zeros(IntList size) const;
  virtual Tensor & zeros_like_out(Tensor & result, const Tensor & input) const;
  virtual Tensor zeros_like(const Tensor & input) const;
  virtual Tensor & ones_out(Tensor & result, IntList size) const;
  virtual Tensor ones(IntList size) const;
  virtual Tensor & ones_like_out(Tensor & result, const Tensor & input) const;
  virtual Tensor ones_like(const Tensor & input) const;
  virtual int64_t numel(const Tensor & self) const;
  virtual Tensor & m_set_(Tensor & self, Storage & storage) const;
  virtual Tensor & m_set_(Tensor & self, Storage & sourceStorage, int64_t storage_offset, IntList size, IntList stride={}) const;
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
  Tensor & masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask) const;
  virtual Tensor & s_masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask) const;
  Tensor masked_select(const Tensor & self, const Tensor & mask) const;
  virtual Tensor s_masked_select(const Tensor & self, const Tensor & mask) const;
  virtual Tensor transpose(const Tensor & self, int64_t dim0, int64_t dim1) const;
  virtual Tensor & m_transpose_(Tensor & self, int64_t dim0, int64_t dim1) const;
  virtual Tensor t(const Tensor & self) const;
  virtual Tensor & m_t_(Tensor & self) const;
  virtual Tensor & squeeze_out(Tensor & result, const Tensor & self, int64_t dim) const;
  virtual Tensor squeeze(const Tensor & self, int64_t dim) const;
  virtual Tensor & squeeze_out(Tensor & result, const Tensor & self) const;
  virtual Tensor squeeze(const Tensor & self) const;
  virtual Tensor & m_squeeze_(Tensor & self, int64_t dim) const;
  virtual Tensor & m_squeeze_(Tensor & self) const;
  virtual Tensor & unsqueeze_out(Tensor & result, const Tensor & self, int64_t dim) const;
  virtual Tensor unsqueeze(const Tensor & self, int64_t dim) const;
  virtual Tensor & m_unsqueeze_(Tensor & self, int64_t dim) const;
  virtual Tensor & nonzero_out(Tensor & result, const Tensor & self) const;
  virtual Tensor nonzero(const Tensor & self) const;
  virtual Tensor m_contiguous(const Tensor & self) const;
  virtual Tensor m_clone(const Tensor & self) const;
  virtual Tensor m_view(const Tensor & self, IntList size) const;
  virtual Tensor m_expand(const Tensor & self, IntList size) const;
  virtual Tensor & m_resize_as_(Tensor & self, const Tensor & the_template) const;
  virtual Tensor & index_select_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const;
  virtual Tensor index_select(const Tensor & self, int64_t dim, const Tensor & index) const;
  virtual Tensor & m_index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const;
  virtual Tensor & m_index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const;
  virtual Tensor & m_index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const;
  virtual Tensor m_narrow(const Tensor & self, int64_t dimension, int64_t start, int64_t length) const;
  virtual Tensor m_unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const;
  virtual Tensor & range_out(Tensor & result, Scalar start, Scalar end, Scalar step=1) const;
  virtual Tensor range(Scalar start, Scalar end, Scalar step=1) const;
  virtual Tensor & arange_out(Tensor & result, Scalar start, Scalar end, Scalar step=1) const;
  virtual Tensor arange(Scalar start, Scalar end, Scalar step=1) const;
  virtual Tensor & m_scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const;
  virtual Tensor & m_scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const;
  virtual Tensor & m_scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const;
  virtual Tensor & gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const;
  virtual Tensor gather(const Tensor & self, int64_t dim, const Tensor & index) const;
  virtual void* m_data_ptr(const Tensor & self) const;
  virtual bool equal(const Tensor & self, const Tensor & other) const;
  virtual Tensor & __and___out(Tensor & result, const Tensor & self, Scalar other) const;
  virtual Tensor __and__(const Tensor & self, Scalar other) const;
  Tensor & __and___out(Tensor & result, const Tensor & self, const Tensor & other) const;
  virtual Tensor & s___and___out(Tensor & result, const Tensor & self, const Tensor & other) const;
  Tensor __and__(const Tensor & self, const Tensor & other) const;
  virtual Tensor s___and__(const Tensor & self, const Tensor & other) const;
  virtual Tensor & __iand__(Tensor & self, Scalar other) const;
  Tensor & __iand__(Tensor & self, const Tensor & other) const;
  virtual Tensor & s___iand__(Tensor & self, const Tensor & other) const;
  virtual Tensor & __or___out(Tensor & result, const Tensor & self, Scalar other) const;
  virtual Tensor __or__(const Tensor & self, Scalar other) const;
  Tensor & __or___out(Tensor & result, const Tensor & self, const Tensor & other) const;
  virtual Tensor & s___or___out(Tensor & result, const Tensor & self, const Tensor & other) const;
  Tensor __or__(const Tensor & self, const Tensor & other) const;
  virtual Tensor s___or__(const Tensor & self, const Tensor & other) const;
  virtual Tensor & __ior__(Tensor & self, Scalar other) const;
  Tensor & __ior__(Tensor & self, const Tensor & other) const;
  virtual Tensor & s___ior__(Tensor & self, const Tensor & other) const;
  virtual Tensor & __xor___out(Tensor & result, const Tensor & self, Scalar other) const;
  virtual Tensor __xor__(const Tensor & self, Scalar other) const;
  Tensor & __xor___out(Tensor & result, const Tensor & self, const Tensor & other) const;
  virtual Tensor & s___xor___out(Tensor & result, const Tensor & self, const Tensor & other) const;
  Tensor __xor__(const Tensor & self, const Tensor & other) const;
  virtual Tensor s___xor__(const Tensor & self, const Tensor & other) const;
  virtual Tensor & __ixor__(Tensor & self, Scalar other) const;
  Tensor & __ixor__(Tensor & self, const Tensor & other) const;
  virtual Tensor & s___ixor__(Tensor & self, const Tensor & other) const;
  virtual Tensor & __lshift___out(Tensor & result, const Tensor & self, Scalar other) const;
  virtual Tensor __lshift__(const Tensor & self, Scalar other) const;
  Tensor & __lshift___out(Tensor & result, const Tensor & self, const Tensor & other) const;
  virtual Tensor & s___lshift___out(Tensor & result, const Tensor & self, const Tensor & other) const;
  Tensor __lshift__(const Tensor & self, const Tensor & other) const;
  virtual Tensor s___lshift__(const Tensor & self, const Tensor & other) const;
  virtual Tensor & __ilshift__(Tensor & self, Scalar other) const;
  Tensor & __ilshift__(Tensor & self, const Tensor & other) const;
  virtual Tensor & s___ilshift__(Tensor & self, const Tensor & other) const;
  virtual Tensor & __rshift___out(Tensor & result, const Tensor & self, Scalar other) const;
  virtual Tensor __rshift__(const Tensor & self, Scalar other) const;
  Tensor & __rshift___out(Tensor & result, const Tensor & self, const Tensor & other) const;
  virtual Tensor & s___rshift___out(Tensor & result, const Tensor & self, const Tensor & other) const;
  Tensor __rshift__(const Tensor & self, const Tensor & other) const;
  virtual Tensor s___rshift__(const Tensor & self, const Tensor & other) const;
  virtual Tensor & __irshift__(Tensor & self, Scalar other) const;
  Tensor & __irshift__(Tensor & self, const Tensor & other) const;
  virtual Tensor & s___irshift__(Tensor & self, const Tensor & other) const;
  virtual Tensor & lt_out(Tensor & result, const Tensor & self, Scalar other) const;
  virtual Tensor lt(const Tensor & self, Scalar other) const;
  Tensor & lt_out(Tensor & result, const Tensor & self, const Tensor & other) const;
  virtual Tensor & s_lt_out(Tensor & result, const Tensor & self, const Tensor & other) const;
  Tensor lt(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_lt(const Tensor & self, const Tensor & other) const;
  virtual Tensor & m_lt_(Tensor & self, Scalar other) const;
  Tensor & m_lt_(Tensor & self, const Tensor & other) const;
  virtual Tensor & s_m_lt_(Tensor & self, const Tensor & other) const;
  virtual Tensor & gt_out(Tensor & result, const Tensor & self, Scalar other) const;
  virtual Tensor gt(const Tensor & self, Scalar other) const;
  Tensor & gt_out(Tensor & result, const Tensor & self, const Tensor & other) const;
  virtual Tensor & s_gt_out(Tensor & result, const Tensor & self, const Tensor & other) const;
  Tensor gt(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_gt(const Tensor & self, const Tensor & other) const;
  virtual Tensor & m_gt_(Tensor & self, Scalar other) const;
  Tensor & m_gt_(Tensor & self, const Tensor & other) const;
  virtual Tensor & s_m_gt_(Tensor & self, const Tensor & other) const;
  virtual Tensor & le_out(Tensor & result, const Tensor & self, Scalar other) const;
  virtual Tensor le(const Tensor & self, Scalar other) const;
  Tensor & le_out(Tensor & result, const Tensor & self, const Tensor & other) const;
  virtual Tensor & s_le_out(Tensor & result, const Tensor & self, const Tensor & other) const;
  Tensor le(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_le(const Tensor & self, const Tensor & other) const;
  virtual Tensor & m_le_(Tensor & self, Scalar other) const;
  Tensor & m_le_(Tensor & self, const Tensor & other) const;
  virtual Tensor & s_m_le_(Tensor & self, const Tensor & other) const;
  virtual Tensor & ge_out(Tensor & result, const Tensor & self, Scalar other) const;
  virtual Tensor ge(const Tensor & self, Scalar other) const;
  Tensor & ge_out(Tensor & result, const Tensor & self, const Tensor & other) const;
  virtual Tensor & s_ge_out(Tensor & result, const Tensor & self, const Tensor & other) const;
  Tensor ge(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_ge(const Tensor & self, const Tensor & other) const;
  virtual Tensor & m_ge_(Tensor & self, Scalar other) const;
  Tensor & m_ge_(Tensor & self, const Tensor & other) const;
  virtual Tensor & s_m_ge_(Tensor & self, const Tensor & other) const;
  virtual Tensor & eq_out(Tensor & result, const Tensor & self, Scalar other) const;
  virtual Tensor eq(const Tensor & self, Scalar other) const;
  Tensor & eq_out(Tensor & result, const Tensor & self, const Tensor & other) const;
  virtual Tensor & s_eq_out(Tensor & result, const Tensor & self, const Tensor & other) const;
  Tensor eq(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_eq(const Tensor & self, const Tensor & other) const;
  virtual Tensor & m_eq_(Tensor & self, Scalar other) const;
  Tensor & m_eq_(Tensor & self, const Tensor & other) const;
  virtual Tensor & s_m_eq_(Tensor & self, const Tensor & other) const;
  virtual Tensor & ne_out(Tensor & result, const Tensor & self, Scalar other) const;
  virtual Tensor ne(const Tensor & self, Scalar other) const;
  Tensor & ne_out(Tensor & result, const Tensor & self, const Tensor & other) const;
  virtual Tensor & s_ne_out(Tensor & result, const Tensor & self, const Tensor & other) const;
  Tensor ne(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_ne(const Tensor & self, const Tensor & other) const;
  virtual Tensor & m_ne_(Tensor & self, Scalar other) const;
  Tensor & m_ne_(Tensor & self, const Tensor & other) const;
  virtual Tensor & s_m_ne_(Tensor & self, const Tensor & other) const;
  virtual std::tuple<Tensor &,Tensor &> min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim=false) const;
  virtual std::tuple<Tensor,Tensor> min(const Tensor & self, int64_t dim, bool keepdim=false) const;
  Tensor & min_out(Tensor & result, const Tensor & self, const Tensor & other) const;
  virtual Tensor & s_min_out(Tensor & result, const Tensor & self, const Tensor & other) const;
  Tensor min(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_min(const Tensor & self, const Tensor & other) const;
  virtual Scalar min(const Tensor & self) const;
  virtual std::tuple<Tensor &,Tensor &> max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim=false) const;
  virtual std::tuple<Tensor,Tensor> max(const Tensor & self, int64_t dim, bool keepdim=false) const;
  Tensor & max_out(Tensor & result, const Tensor & self, const Tensor & other) const;
  virtual Tensor & s_max_out(Tensor & result, const Tensor & self, const Tensor & other) const;
  Tensor max(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_max(const Tensor & self, const Tensor & other) const;
  virtual Scalar max(const Tensor & self) const;
  virtual std::tuple<Tensor &,Tensor &> kthvalue_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim=-1, bool keepdim=false) const;
  virtual std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k, int64_t dim=-1, bool keepdim=false) const;
  virtual std::tuple<Tensor &,Tensor &> mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim=-1, bool keepdim=false) const;
  virtual std::tuple<Tensor,Tensor> mode(const Tensor & self, int64_t dim=-1, bool keepdim=false) const;
  virtual std::tuple<Tensor &,Tensor &> median_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim=false) const;
  virtual std::tuple<Tensor,Tensor> median(const Tensor & self, int64_t dim, bool keepdim=false) const;
  virtual Scalar median(const Tensor & self) const;
  virtual std::tuple<Tensor &,Tensor &> sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim=-1, bool descending=false) const;
  virtual std::tuple<Tensor,Tensor> sort(const Tensor & self, int64_t dim=-1, bool descending=false) const;
  virtual std::tuple<Tensor &,Tensor &> topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim=-1, bool largest=true, bool sorted=true) const;
  virtual std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim=-1, bool largest=true, bool sorted=true) const;
  virtual bool m_all(const Tensor & self) const;
  virtual bool m_any(const Tensor & self) const;
  virtual int64_t m_get_device(const Tensor & self) const;
  virtual Tensor & abs_out(Tensor & destination, const Tensor & self) const;
  virtual Tensor abs(const Tensor & self) const;
  virtual Tensor & m_abs_(Tensor & self) const;
  virtual Tensor & m_sigmoid_(Tensor & self) const;
  virtual Tensor & sigmoid_out(Tensor & result, const Tensor & self) const;
  virtual Tensor sigmoid(const Tensor & self) const;
  virtual Tensor & m_log_(Tensor & self) const;
  virtual Tensor & log_out(Tensor & result, const Tensor & self) const;
  virtual Tensor log(const Tensor & self) const;
  virtual Tensor & m_log1p_(Tensor & self) const;
  virtual Tensor & log1p_out(Tensor & result, const Tensor & self) const;
  virtual Tensor log1p(const Tensor & self) const;
  virtual Tensor & lgamma_out(Tensor & result, const Tensor & self) const;
  virtual Tensor lgamma(const Tensor & self) const;
  virtual Tensor & m_lgamma_(Tensor & self) const;
  virtual Tensor & m_exp_(Tensor & self) const;
  virtual Tensor & exp_out(Tensor & result, const Tensor & self) const;
  virtual Tensor exp(const Tensor & self) const;
  virtual Tensor & m_cos_(Tensor & self) const;
  virtual Tensor & cos_out(Tensor & result, const Tensor & self) const;
  virtual Tensor cos(const Tensor & self) const;
  virtual Tensor & m_acos_(Tensor & self) const;
  virtual Tensor & acos_out(Tensor & result, const Tensor & self) const;
  virtual Tensor acos(const Tensor & self) const;
  virtual Tensor & m_cosh_(Tensor & self) const;
  virtual Tensor & cosh_out(Tensor & result, const Tensor & self) const;
  virtual Tensor cosh(const Tensor & self) const;
  virtual Tensor & m_sin_(Tensor & self) const;
  virtual Tensor & sin_out(Tensor & result, const Tensor & self) const;
  virtual Tensor sin(const Tensor & self) const;
  virtual Tensor & m_asin_(Tensor & self) const;
  virtual Tensor & asin_out(Tensor & result, const Tensor & self) const;
  virtual Tensor asin(const Tensor & self) const;
  virtual Tensor & m_sinh_(Tensor & self) const;
  virtual Tensor & sinh_out(Tensor & result, const Tensor & self) const;
  virtual Tensor sinh(const Tensor & self) const;
  virtual Tensor & m_tan_(Tensor & self) const;
  virtual Tensor & tan_out(Tensor & result, const Tensor & self) const;
  virtual Tensor tan(const Tensor & self) const;
  virtual Tensor & m_atan_(Tensor & self) const;
  virtual Tensor & atan_out(Tensor & result, const Tensor & self) const;
  virtual Tensor atan(const Tensor & self) const;
  virtual Tensor & m_tanh_(Tensor & self) const;
  virtual Tensor & tanh_out(Tensor & result, const Tensor & self) const;
  virtual Tensor tanh(const Tensor & self) const;
  virtual Tensor & m_erf_(Tensor & self) const;
  virtual Tensor & erf_out(Tensor & result, const Tensor & self) const;
  virtual Tensor erf(const Tensor & self) const;
  virtual Tensor & m_erfinv_(Tensor & self) const;
  virtual Tensor & erfinv_out(Tensor & result, const Tensor & self) const;
  virtual Tensor erfinv(const Tensor & self) const;
  virtual Tensor & m_sqrt_(Tensor & self) const;
  virtual Tensor & sqrt_out(Tensor & result, const Tensor & self) const;
  virtual Tensor sqrt(const Tensor & self) const;
  virtual Tensor & m_rsqrt_(Tensor & self) const;
  virtual Tensor & rsqrt_out(Tensor & result, const Tensor & self) const;
  virtual Tensor rsqrt(const Tensor & self) const;
  virtual Tensor & m_ceil_(Tensor & self) const;
  virtual Tensor & ceil_out(Tensor & result, const Tensor & self) const;
  virtual Tensor ceil(const Tensor & self) const;
  virtual Tensor & m_floor_(Tensor & self) const;
  virtual Tensor & floor_out(Tensor & result, const Tensor & self) const;
  virtual Tensor floor(const Tensor & self) const;
  virtual Tensor & m_round_(Tensor & self) const;
  virtual Tensor & round_out(Tensor & result, const Tensor & self) const;
  virtual Tensor round(const Tensor & self) const;
  virtual Tensor & m_trunc_(Tensor & self) const;
  virtual Tensor & trunc_out(Tensor & result, const Tensor & self) const;
  virtual Tensor trunc(const Tensor & self) const;
  virtual Tensor & m_frac_(Tensor & self) const;
  virtual Tensor & frac_out(Tensor & result, const Tensor & self) const;
  virtual Tensor frac(const Tensor & self) const;
  virtual Tensor & mean_out(Tensor & destination, const Tensor & self, int64_t dim, bool keepdim=false) const;
  virtual Tensor mean(const Tensor & self, int64_t dim, bool keepdim=false) const;
  virtual Scalar mean(const Tensor & self) const;
  virtual Tensor & var_out(Tensor & destination, const Tensor & self, int64_t dim, bool unbiased=true, bool keepdim=false) const;
  virtual Tensor var(const Tensor & self, int64_t dim, bool unbiased=true, bool keepdim=false) const;
  virtual Scalar var(const Tensor & self, bool unbiased=true) const;
  virtual Tensor & std_out(Tensor & destination, const Tensor & self, int64_t dim, bool unbiased=true, bool keepdim=false) const;
  virtual Tensor std(const Tensor & self, int64_t dim, bool unbiased=true, bool keepdim=false) const;
  virtual Scalar std(const Tensor & self, bool unbiased=true) const;
  virtual Tensor & norm_out(Tensor & destination, const Tensor & self, Scalar p, int64_t dim, bool keepdim=false) const;
  virtual Tensor norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim=false) const;
  virtual Scalar norm(const Tensor & self, Scalar p=2) const;
  virtual Tensor & renorm_out(Tensor & destination, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const;
  virtual Tensor renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const;
  virtual Tensor & m_renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const;
  Scalar dist(const Tensor & self, const Tensor & other, Scalar p=2) const;
  virtual Scalar s_dist(const Tensor & self, const Tensor & other, Scalar p=2) const;
  virtual Tensor & reciprocal_out(Tensor & destination, const Tensor & self) const;
  virtual Tensor reciprocal(const Tensor & self) const;
  virtual Tensor & m_reciprocal_(Tensor & self) const;
  virtual Tensor & neg_out(Tensor & destination, const Tensor & self) const;
  virtual Tensor neg(const Tensor & self) const;
  virtual Tensor & m_neg_(Tensor & self) const;
  Tensor & atan2_out(Tensor & destination, const Tensor & self, const Tensor & other) const;
  virtual Tensor & s_atan2_out(Tensor & destination, const Tensor & self, const Tensor & other) const;
  Tensor atan2(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_atan2(const Tensor & self, const Tensor & other) const;
  Tensor & m_atan2_(Tensor & self, const Tensor & other) const;
  virtual Tensor & s_m_atan2_(Tensor & self, const Tensor & other) const;
  virtual Tensor & pow_out(Tensor & destination, const Tensor & self, Scalar exponent) const;
  virtual Tensor pow(const Tensor & self, Scalar exponent) const;
  Tensor & pow_out(Tensor & destination, const Tensor & self, const Tensor & exponent) const;
  virtual Tensor & s_pow_out(Tensor & destination, const Tensor & self, const Tensor & exponent) const;
  Tensor pow(const Tensor & self, const Tensor & exponent) const;
  virtual Tensor s_pow(const Tensor & self, const Tensor & exponent) const;
  virtual Tensor & m_pow_(Tensor & self, Scalar exponent) const;
  Tensor & m_pow_(Tensor & self, const Tensor & exponent) const;
  virtual Tensor & s_m_pow_(Tensor & self, const Tensor & exponent) const;
  Tensor & lerp_out(Tensor & destination, const Tensor & self, const Tensor & end, Scalar weight) const;
  virtual Tensor & s_lerp_out(Tensor & destination, const Tensor & self, const Tensor & end, Scalar weight) const;
  Tensor lerp(const Tensor & self, const Tensor & end, Scalar weight) const;
  virtual Tensor s_lerp(const Tensor & self, const Tensor & end, Scalar weight) const;
  Tensor & m_lerp_(Tensor & self, const Tensor & end, Scalar weight) const;
  virtual Tensor & s_m_lerp_(Tensor & self, const Tensor & end, Scalar weight) const;
  virtual Tensor & linspace_out(Tensor & result, Scalar start, Scalar end, int64_t steps=100) const;
  virtual Tensor linspace(Scalar start, Scalar end, int64_t steps=100) const;
  virtual Tensor & logspace_out(Tensor & result, Scalar start, Scalar end, int64_t steps=100) const;
  virtual Tensor logspace(Scalar start, Scalar end, int64_t steps=100) const;
  virtual Tensor & histc_out(Tensor & destination, const Tensor & self, int64_t bins=100, Scalar min=0, Scalar max=0) const;
  virtual Tensor histc(const Tensor & self, int64_t bins=100, Scalar min=0, Scalar max=0) const;
  virtual Tensor & m_zero_(Tensor & self) const;
  virtual Tensor & sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim=false) const;
  virtual Tensor sum(const Tensor & self, int64_t dim, bool keepdim=false) const;
  virtual Scalar sum(const Tensor & self) const;
  virtual Tensor & prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim=false) const;
  virtual Tensor prod(const Tensor & self, int64_t dim, bool keepdim=false) const;
  virtual Scalar prod(const Tensor & self) const;
  virtual Tensor & cumsum_out(Tensor & result, const Tensor & self, int64_t dim) const;
  virtual Tensor cumsum(const Tensor & self, int64_t dim) const;
  virtual Tensor & cumprod_out(Tensor & result, const Tensor & self, int64_t dim) const;
  virtual Tensor cumprod(const Tensor & self, int64_t dim) const;
  virtual Tensor & sign_out(Tensor & result, const Tensor & self) const;
  virtual Tensor sign(const Tensor & self) const;
  virtual Tensor & m_sign_(Tensor & self) const;
  virtual Scalar trace(const Tensor & self) const;
  virtual Tensor & add_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha=1) const;
  virtual Tensor add(const Tensor & self, Scalar other, Scalar alpha=1) const;
  Tensor & add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha=1) const;
  virtual Tensor & s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha=1) const;
  Tensor add(const Tensor & self, const Tensor & other, Scalar alpha=1) const;
  virtual Tensor s_add(const Tensor & self, const Tensor & other, Scalar alpha=1) const;
  virtual Tensor & add_out(Tensor & result, const Tensor & self, SparseTensor other, Scalar alpha=1) const;
  virtual Tensor add(const Tensor & self, SparseTensor other, Scalar alpha=1) const;
  virtual Tensor & m_add_(Tensor & self, Scalar other, Scalar alpha=1) const;
  Tensor & m_add_(Tensor & self, const Tensor & other, Scalar alpha=1) const;
  virtual Tensor & s_m_add_(Tensor & self, const Tensor & other, Scalar alpha=1) const;
  virtual Tensor & m_add_(Tensor & self, SparseTensor other, Scalar alpha=1) const;
  virtual Tensor & sub_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha=1) const;
  virtual Tensor sub(const Tensor & self, Scalar other, Scalar alpha=1) const;
  Tensor & sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha=1) const;
  virtual Tensor & s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha=1) const;
  Tensor sub(const Tensor & self, const Tensor & other, Scalar alpha=1) const;
  virtual Tensor s_sub(const Tensor & self, const Tensor & other, Scalar alpha=1) const;
  virtual Tensor & m_sub_(Tensor & self, Scalar other, Scalar alpha=1) const;
  Tensor & m_sub_(Tensor & self, const Tensor & other, Scalar alpha=1) const;
  virtual Tensor & s_m_sub_(Tensor & self, const Tensor & other, Scalar alpha=1) const;
  virtual Tensor & mul_out(Tensor & result, const Tensor & self, Scalar other) const;
  virtual Tensor mul(const Tensor & self, Scalar other) const;
  Tensor & mul_out(Tensor & result, const Tensor & self, const Tensor & other) const;
  virtual Tensor & s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const;
  Tensor mul(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_mul(const Tensor & self, const Tensor & other) const;
  virtual Tensor & m_mul_(Tensor & self, Scalar other) const;
  Tensor & m_mul_(Tensor & self, const Tensor & other) const;
  virtual Tensor & s_m_mul_(Tensor & self, const Tensor & other) const;
  virtual Tensor & div_out(Tensor & result, const Tensor & self, Scalar other) const;
  virtual Tensor div(const Tensor & self, Scalar other) const;
  Tensor & div_out(Tensor & result, const Tensor & self, const Tensor & other) const;
  virtual Tensor & s_div_out(Tensor & result, const Tensor & self, const Tensor & other) const;
  Tensor div(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_div(const Tensor & self, const Tensor & other) const;
  virtual Tensor & m_div_(Tensor & self, Scalar other) const;
  Tensor & m_div_(Tensor & self, const Tensor & other) const;
  virtual Tensor & s_m_div_(Tensor & self, const Tensor & other) const;
  virtual Tensor & fmod_out(Tensor & result, const Tensor & self, Scalar other) const;
  virtual Tensor fmod(const Tensor & self, Scalar other) const;
  Tensor & fmod_out(Tensor & result, const Tensor & self, const Tensor & other) const;
  virtual Tensor & s_fmod_out(Tensor & result, const Tensor & self, const Tensor & other) const;
  Tensor fmod(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_fmod(const Tensor & self, const Tensor & other) const;
  virtual Tensor & m_fmod_(Tensor & self, Scalar other) const;
  Tensor & m_fmod_(Tensor & self, const Tensor & other) const;
  virtual Tensor & s_m_fmod_(Tensor & self, const Tensor & other) const;
  virtual Tensor & remainder_out(Tensor & result, const Tensor & self, Scalar other) const;
  virtual Tensor remainder(const Tensor & self, Scalar other) const;
  Tensor & remainder_out(Tensor & result, const Tensor & self, const Tensor & other) const;
  virtual Tensor & s_remainder_out(Tensor & result, const Tensor & self, const Tensor & other) const;
  Tensor remainder(const Tensor & self, const Tensor & other) const;
  virtual Tensor s_remainder(const Tensor & self, const Tensor & other) const;
  virtual Tensor & m_remainder_(Tensor & self, Scalar other) const;
  Tensor & m_remainder_(Tensor & self, const Tensor & other) const;
  virtual Tensor & s_m_remainder_(Tensor & self, const Tensor & other) const;
  virtual Tensor & clamp_out(Tensor & destination, const Tensor & self, Scalar min, Scalar max) const;
  virtual Tensor clamp(const Tensor & self, Scalar min, Scalar max) const;
  virtual Tensor & clamp_out(Tensor & result, const Tensor & self, Scalar min) const;
  virtual Tensor clamp(const Tensor & self, Scalar min) const;
  virtual Tensor & m_clamp_(Tensor & self, Scalar min, Scalar max) const;
  virtual Tensor & m_clamp_(Tensor & self, Scalar min) const;
  virtual Scalar dot(const Tensor & self, const Tensor & tensor) const;
  virtual Tensor & tril_out(Tensor & destination, const Tensor & self, int64_t diagonal=0) const;
  virtual Tensor tril(const Tensor & self, int64_t diagonal=0) const;
  virtual Tensor & m_tril_(Tensor & self, int64_t diagonal=0) const;
  virtual Tensor & triu_out(Tensor & destination, const Tensor & self, int64_t diagonal=0) const;
  virtual Tensor triu(const Tensor & self, int64_t diagonal=0) const;
  virtual Tensor & m_triu_(Tensor & self, int64_t diagonal=0) const;
  virtual Tensor & cross_out(Tensor & destination, const Tensor & self, const Tensor & other, int64_t dim=-1) const;
  virtual Tensor cross(const Tensor & self, const Tensor & other, int64_t dim=-1) const;
  virtual Tensor & eye_out(Tensor & result, int64_t n, int64_t m=1) const;
  virtual Tensor eye(int64_t n, int64_t m=1) const;
  virtual Tensor & diag_out(Tensor & result, const Tensor & self, int64_t diagonal=0) const;
  virtual Tensor diag(const Tensor & self, int64_t diagonal=0) const;
  Tensor & addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1) const;
  virtual Tensor & s_addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1) const;
  Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1) const;
  virtual Tensor s_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1) const;
  virtual Tensor & m_addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1) const;
  Tensor & addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta=1, Scalar alpha=1) const;
  virtual Tensor & s_addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta=1, Scalar alpha=1) const;
  Tensor addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta=1, Scalar alpha=1) const;
  virtual Tensor s_addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta=1, Scalar alpha=1) const;
  virtual Tensor & m_addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta=1, Scalar alpha=1) const;
  Tensor & addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta=1, Scalar alpha=1) const;
  virtual Tensor & s_addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta=1, Scalar alpha=1) const;
  Tensor addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta=1, Scalar alpha=1) const;
  virtual Tensor s_addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta=1, Scalar alpha=1) const;
  virtual Tensor & m_addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta=1, Scalar alpha=1) const;
  virtual Tensor & ger_out(Tensor & result, const Tensor & self, const Tensor & vec2) const;
  virtual Tensor ger(const Tensor & self, const Tensor & vec2) const;
  virtual Tensor & mv_out(Tensor & result, const Tensor & self, const Tensor & vec) const;
  virtual Tensor mv(const Tensor & self, const Tensor & vec) const;
  virtual Tensor & mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const;
  virtual Tensor mm(const Tensor & self, const Tensor & mat2) const;
  virtual Tensor & bmm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const;
  virtual Tensor bmm(const Tensor & self, const Tensor & mat2) const;
  Tensor & addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta=1, Scalar alpha=1) const;
  virtual Tensor & s_addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta=1, Scalar alpha=1) const;
  Tensor addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta=1, Scalar alpha=1) const;
  virtual Tensor s_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta=1, Scalar alpha=1) const;
  virtual Tensor & m_addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta=1, Scalar alpha=1) const;
  Tensor & baddbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta=1, Scalar alpha=1) const;
  virtual Tensor & s_baddbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta=1, Scalar alpha=1) const;
  Tensor baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta=1, Scalar alpha=1) const;
  virtual Tensor s_baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta=1, Scalar alpha=1) const;
  virtual Tensor & m_baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta=1, Scalar alpha=1) const;
  Tensor & addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value=1) const;
  virtual Tensor & s_addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value=1) const;
  Tensor addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value=1) const;
  virtual Tensor s_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value=1) const;
  Tensor & m_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value=1) const;
  virtual Tensor & s_m_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value=1) const;
  Tensor & addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value=1) const;
  virtual Tensor & s_addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value=1) const;
  Tensor addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value=1) const;
  virtual Tensor s_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value=1) const;
  Tensor & m_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value=1) const;
  virtual Tensor & s_m_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value=1) const;
  virtual std::tuple<Tensor &,Tensor &> gesv_out(Tensor & solution, Tensor & lu, const Tensor & self, const Tensor & A) const;
  virtual std::tuple<Tensor,Tensor> gesv(const Tensor & self, const Tensor & A) const;
  virtual std::tuple<Tensor &,Tensor &> gels_out(Tensor & res1, Tensor & res2, const Tensor & self, const Tensor & A) const;
  virtual std::tuple<Tensor,Tensor> gels(const Tensor & self, const Tensor & A) const;
  virtual std::tuple<Tensor &,Tensor &> trtrs_out(Tensor & res1, Tensor & res2, const Tensor & self, const Tensor & A, bool upper=true, bool transpose=false, bool unitriangular=false) const;
  virtual std::tuple<Tensor,Tensor> trtrs(const Tensor & self, const Tensor & A, bool upper=true, bool transpose=false, bool unitriangular=false) const;
  virtual std::tuple<Tensor &,Tensor &> symeig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors=false, bool upper=true) const;
  virtual std::tuple<Tensor,Tensor> symeig(const Tensor & self, bool eigenvectors=false, bool upper=true) const;
  virtual std::tuple<Tensor &,Tensor &> eig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors=false) const;
  virtual std::tuple<Tensor,Tensor> eig(const Tensor & self, bool eigenvectors=false) const;
  virtual std::tuple<Tensor &,Tensor &,Tensor &> svd_out(Tensor & res1, Tensor & res2, Tensor & res3, const Tensor & self, bool some=true) const;
  virtual std::tuple<Tensor,Tensor,Tensor> svd(const Tensor & self, bool some=true) const;
  virtual Tensor & inverse_out(Tensor & output, const Tensor & self) const;
  virtual Tensor inverse(const Tensor & self) const;
  virtual Tensor & potrf_out(Tensor & output, const Tensor & self, bool upper=true) const;
  virtual Tensor potrf(const Tensor & self, bool upper=true) const;
  virtual Tensor & potrs_out(Tensor & result, const Tensor & self, const Tensor & input2, bool upper=true) const;
  virtual Tensor potrs(const Tensor & self, const Tensor & input2, bool upper=true) const;
  virtual Tensor & potri_out(Tensor & output, const Tensor & self, bool upper=true) const;
  virtual Tensor potri(const Tensor & self, bool upper=true) const;
  virtual std::tuple<Tensor &,Tensor &> pstrf_out(Tensor & res1, Tensor & res2, const Tensor & self, bool upper=true, Scalar tol=-1) const;
  virtual std::tuple<Tensor,Tensor> pstrf(const Tensor & self, bool upper=true, Scalar tol=-1) const;
  virtual std::tuple<Tensor &,Tensor &> qr_out(Tensor & res1, Tensor & res2, const Tensor & self) const;
  virtual std::tuple<Tensor,Tensor> qr(const Tensor & self) const;
  virtual std::tuple<Tensor &,Tensor &> geqrf_out(Tensor & res1, Tensor & res2, const Tensor & self) const;
  virtual std::tuple<Tensor,Tensor> geqrf(const Tensor & self) const;
  virtual Tensor & orgqr_out(Tensor & result, const Tensor & self, const Tensor & input2) const;
  virtual Tensor orgqr(const Tensor & self, const Tensor & input2) const;
  virtual Tensor & ormqr_out(Tensor & result, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left=true, bool transpose=false) const;
  virtual Tensor ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left=true, bool transpose=false) const;
  virtual std::tuple<Tensor &,Tensor &> btrifact_out(Tensor & result, Tensor & pivots, const Tensor & self, const Tensor & info={}, bool pivot=true) const;
  virtual std::tuple<Tensor,Tensor> btrifact(const Tensor & self, const Tensor & info={}, bool pivot=true) const;
  virtual Tensor & btrisolve_out(Tensor & result, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const;
  virtual Tensor btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const;
  virtual Tensor & randperm_out(Tensor & result, int64_t n, Generator * generator=nullptr) const;
  virtual Tensor randperm(int64_t n, Generator * generator=nullptr) const;
  virtual Tensor & m_random_(Tensor & self, int64_t from, int64_t to, Generator * generator=nullptr) const;
  virtual Tensor & m_random_(Tensor & self, int64_t to, Generator * generator=nullptr) const;
  virtual Tensor & m_random_(Tensor & self, Generator * generator=nullptr) const;
  virtual Tensor & multinomial_out(Tensor & result, const Tensor & self, int64_t num_samples, bool replacement=false, Generator * generator=nullptr) const;
  virtual Tensor multinomial(const Tensor & self, int64_t num_samples, bool replacement=false, Generator * generator=nullptr) const;
  virtual Tensor & m_uniform_(Tensor & self, double from=0, double to=1, Generator * generator=nullptr) const;
  virtual Tensor & normal_out(Tensor & output, const Tensor & means, double std=1, Generator * generator=nullptr) const;
  virtual Tensor normal(const Tensor & means, double std=1, Generator * generator=nullptr) const;
  virtual Tensor & normal_out(Tensor & output, double mean, const Tensor & std, Generator * generator=nullptr) const;
  virtual Tensor normal(double mean, const Tensor & std, Generator * generator=nullptr) const;
  virtual Tensor & normal_out(Tensor & output, const Tensor & means, const Tensor & std, Generator * generator=nullptr) const;
  virtual Tensor normal(const Tensor & means, const Tensor & std, Generator * generator=nullptr) const;
  virtual Tensor & m_normal_(Tensor & self, double mean=0, double std=1, Generator * generator=nullptr) const;
  virtual Tensor & m_cauchy_(Tensor & self, double median=0, double sigma=1, Generator * generator=nullptr) const;
  virtual Tensor & m_log_normal_(Tensor & self, double mean=1, double std=2, Generator * generator=nullptr) const;
  virtual Tensor & m_exponential_(Tensor & self, double lambd=1, Generator * generator=nullptr) const;
  virtual Tensor & rand_out(Tensor & result, IntList size, Generator * generator=nullptr) const;
  virtual Tensor rand(IntList size, Generator * generator=nullptr) const;
  virtual Tensor & randn_out(Tensor & result, IntList size, Generator * generator=nullptr) const;
  virtual Tensor randn(IntList size, Generator * generator=nullptr) const;
  virtual Tensor & m_geometric_(Tensor & self, double p, Generator * generator=nullptr) const;
  virtual int64_t m_size(const Tensor & self, int64_t dim) const;
  virtual int64_t m_stride(const Tensor & self, int64_t dim) const;
  virtual Tensor tensor(Storage & storage, int64_t storageOffset, IntList size, IntList stride={}) const;
  virtual Tensor tensor(IntList size) const;
  virtual Tensor tensor(IntList size, IntList stride) const;
  virtual Tensor tensor() const;
  virtual Tensor & select_out(Tensor & result, const Tensor & self, int64_t dim, int64_t sliceIndex) const;
  virtual Tensor select(const Tensor & self, int64_t dim, int64_t sliceIndex) const;
  virtual Tensor & _unnarrow_out(Tensor & result, const Tensor & self, int64_t dimension, int64_t offset, int64_t dimSize) const;
  virtual Tensor _unnarrow(const Tensor & self, int64_t dimension, int64_t offset, int64_t dimSize) const;
  virtual Tensor & m_assign_(Tensor & self, const Tensor & src) const;
  virtual Tensor & cat_out(Tensor & self, TensorList tensors, int64_t dim) const;
  virtual Tensor cat(TensorList tensors, int64_t dim) const;
  virtual Tensor & binary_cross_entropy_out(Tensor & output, const Tensor & input, const Tensor & target, const Tensor & weight={}, bool size_average=true) const;
  virtual Tensor binary_cross_entropy(const Tensor & input, const Tensor & target, const Tensor & weight={}, bool size_average=true) const;
  virtual Tensor & binary_cross_entropy_forward_out(Tensor & output, const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average) const;
  virtual Tensor binary_cross_entropy_forward(const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average) const;
  virtual Tensor & binary_cross_entropy_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average) const;
  virtual Tensor binary_cross_entropy_backward(const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average) const;
  virtual Tensor & kl_div_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average=true) const;
  virtual Tensor kl_div(const Tensor & input, const Tensor & target, bool size_average=true) const;
  virtual Tensor & kl_div_forward_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average) const;
  virtual Tensor kl_div_forward(const Tensor & input, const Tensor & target, bool size_average) const;
  virtual Tensor & kl_div_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, bool size_average) const;
  virtual Tensor kl_div_backward(const Tensor & input, const Tensor & target, bool size_average) const;
  virtual Tensor & l1_loss_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average=true) const;
  virtual Tensor l1_loss(const Tensor & input, const Tensor & target, bool size_average=true) const;
  virtual Tensor & l1_loss_forward_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average) const;
  virtual Tensor l1_loss_forward(const Tensor & input, const Tensor & target, bool size_average) const;
  virtual Tensor & l1_loss_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, bool size_average) const;
  virtual Tensor l1_loss_backward(const Tensor & input, const Tensor & target, bool size_average) const;
  virtual Tensor & mse_loss_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average=true, bool reduce=true) const;
  virtual Tensor mse_loss(const Tensor & input, const Tensor & target, bool size_average=true, bool reduce=true) const;
  virtual Tensor & mse_loss_forward_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average, bool reduce) const;
  virtual Tensor mse_loss_forward(const Tensor & input, const Tensor & target, bool size_average, bool reduce) const;
  virtual Tensor & mse_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, const Tensor & target, bool size_average, bool reduce) const;
  virtual Tensor mse_loss_backward(const Tensor & grad_output, const Tensor & input, const Tensor & target, bool size_average, bool reduce) const;
  virtual Tensor & multi_margin_loss_out(Tensor & output, const Tensor & input, const Tensor & target, Scalar p=1, Scalar margin=1, const Tensor & weight={}, bool size_average=true) const;
  virtual Tensor multi_margin_loss(const Tensor & input, const Tensor & target, Scalar p=1, Scalar margin=1, const Tensor & weight={}, bool size_average=true) const;
  virtual Tensor & multi_margin_loss_forward_out(Tensor & output, const Tensor & input, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) const;
  virtual Tensor multi_margin_loss_forward(const Tensor & input, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) const;
  virtual Tensor & multi_margin_loss_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) const;
  virtual Tensor multi_margin_loss_backward(const Tensor & input, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) const;
  virtual Tensor & multilabel_margin_loss_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average=true) const;
  virtual Tensor multilabel_margin_loss(const Tensor & input, const Tensor & target, bool size_average=true) const;
  virtual Tensor & multilabel_margin_loss_forward_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average, const Tensor & is_target) const;
  virtual Tensor multilabel_margin_loss_forward(const Tensor & input, const Tensor & target, bool size_average, const Tensor & is_target) const;
  virtual Tensor & multilabel_margin_loss_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, bool size_average, const Tensor & is_target) const;
  virtual Tensor multilabel_margin_loss_backward(const Tensor & input, const Tensor & target, bool size_average, const Tensor & is_target) const;
  virtual Tensor & nll_loss_out(Tensor & output, const Tensor & input, const Tensor & target, const Tensor & weight={}, bool size_average=true, int64_t ignore_index=-100) const;
  virtual Tensor nll_loss(const Tensor & input, const Tensor & target, const Tensor & weight={}, bool size_average=true, int64_t ignore_index=-100) const;
  virtual Tensor & nll_loss_forward_out(Tensor & output, const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, const Tensor & total_weight) const;
  virtual Tensor nll_loss_forward(const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, const Tensor & total_weight) const;
  virtual Tensor & nll_loss_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, const Tensor & total_weight) const;
  virtual Tensor nll_loss_backward(const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, const Tensor & total_weight) const;
  virtual Tensor & nll_loss2d_out(Tensor & output, const Tensor & input, const Tensor & target, const Tensor & weight={}, bool size_average=true, int64_t ignore_index=-100) const;
  virtual Tensor nll_loss2d(const Tensor & input, const Tensor & target, const Tensor & weight={}, bool size_average=true, int64_t ignore_index=-100) const;
  virtual Tensor & nll_loss2d_forward_out(Tensor & output, const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, const Tensor & total_weight) const;
  virtual Tensor nll_loss2d_forward(const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, const Tensor & total_weight) const;
  virtual Tensor & nll_loss2d_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, const Tensor & total_weight) const;
  virtual Tensor nll_loss2d_backward(const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, const Tensor & total_weight) const;
  virtual Tensor & smooth_l1_loss_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average=true) const;
  virtual Tensor smooth_l1_loss(const Tensor & input, const Tensor & target, bool size_average=true) const;
  virtual Tensor & smooth_l1_loss_forward_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average) const;
  virtual Tensor smooth_l1_loss_forward(const Tensor & input, const Tensor & target, bool size_average) const;
  virtual Tensor & smooth_l1_loss_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, bool size_average) const;
  virtual Tensor smooth_l1_loss_backward(const Tensor & input, const Tensor & target, bool size_average) const;
  virtual Tensor & soft_margin_loss_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average=true) const;
  virtual Tensor soft_margin_loss(const Tensor & input, const Tensor & target, bool size_average=true) const;
  virtual Tensor & soft_margin_loss_forward_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average) const;
  virtual Tensor soft_margin_loss_forward(const Tensor & input, const Tensor & target, bool size_average) const;
  virtual Tensor & soft_margin_loss_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, bool size_average) const;
  virtual Tensor soft_margin_loss_backward(const Tensor & input, const Tensor & target, bool size_average) const;
  virtual Tensor & elu_out(Tensor & output, const Tensor & input, Scalar alpha=1, bool inplace=false) const;
  virtual Tensor elu(const Tensor & input, Scalar alpha=1, bool inplace=false) const;
  virtual Tensor & elu_forward_out(Tensor & output, const Tensor & input, Scalar alpha, bool inplace) const;
  virtual Tensor elu_forward(const Tensor & input, Scalar alpha, bool inplace) const;
  virtual Tensor & elu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, Scalar alpha, bool inplace, const Tensor & output) const;
  virtual Tensor elu_backward(const Tensor & grad_output, const Tensor & input, Scalar alpha, bool inplace, const Tensor & output) const;
  virtual Tensor & glu_out(Tensor & output, const Tensor & input, int64_t dim=-1) const;
  virtual Tensor glu(const Tensor & input, int64_t dim=-1) const;
  virtual Tensor & glu_forward_out(Tensor & output, const Tensor & input, int64_t dim) const;
  virtual Tensor glu_forward(const Tensor & input, int64_t dim) const;
  virtual Tensor & glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, int64_t dim) const;
  virtual Tensor glu_backward(const Tensor & grad_output, const Tensor & input, int64_t dim) const;
  virtual Tensor & hardshrink_out(Tensor & output, const Tensor & input, Scalar lambd=0.5) const;
  virtual Tensor hardshrink(const Tensor & input, Scalar lambd=0.5) const;
  virtual Tensor & hardshrink_forward_out(Tensor & output, const Tensor & input, Scalar lambd) const;
  virtual Tensor hardshrink_forward(const Tensor & input, Scalar lambd) const;
  virtual Tensor & hardshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, Scalar lambd) const;
  virtual Tensor hardshrink_backward(const Tensor & grad_output, const Tensor & input, Scalar lambd) const;
  virtual Tensor & hardtanh_out(Tensor & output, const Tensor & input, Scalar min_val=-1, Scalar max_val=1, bool inplace=false) const;
  virtual Tensor hardtanh(const Tensor & input, Scalar min_val=-1, Scalar max_val=1, bool inplace=false) const;
  virtual Tensor & hardtanh_forward_out(Tensor & output, const Tensor & input, Scalar min_val, Scalar max_val, bool inplace) const;
  virtual Tensor hardtanh_forward(const Tensor & input, Scalar min_val, Scalar max_val, bool inplace) const;
  virtual Tensor & hardtanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, Scalar min_val, Scalar max_val, bool inplace) const;
  virtual Tensor hardtanh_backward(const Tensor & grad_output, const Tensor & input, Scalar min_val, Scalar max_val, bool inplace) const;
  virtual Tensor & leaky_relu_out(Tensor & output, const Tensor & input, Scalar negative_slope=0.01, bool inplace=false) const;
  virtual Tensor leaky_relu(const Tensor & input, Scalar negative_slope=0.01, bool inplace=false) const;
  virtual Tensor & leaky_relu_forward_out(Tensor & output, const Tensor & input, Scalar negative_slope, bool inplace) const;
  virtual Tensor leaky_relu_forward(const Tensor & input, Scalar negative_slope, bool inplace) const;
  virtual Tensor & leaky_relu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, Scalar negative_slope, bool inplace) const;
  virtual Tensor leaky_relu_backward(const Tensor & grad_output, const Tensor & input, Scalar negative_slope, bool inplace) const;
  virtual Tensor & log_sigmoid_out(Tensor & output, const Tensor & input) const;
  virtual Tensor log_sigmoid(const Tensor & input) const;
  virtual Tensor & log_sigmoid_forward_out(Tensor & output, const Tensor & input, const Tensor & buffer) const;
  virtual Tensor log_sigmoid_forward(const Tensor & input, const Tensor & buffer) const;
  virtual Tensor & log_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, const Tensor & buffer) const;
  virtual Tensor log_sigmoid_backward(const Tensor & grad_output, const Tensor & input, const Tensor & buffer) const;
  virtual Tensor & log_softmax_out(Tensor & output, const Tensor & input, int64_t dim) const;
  virtual Tensor log_softmax(const Tensor & input, int64_t dim) const;
  virtual Tensor & log_softmax_forward_out(Tensor & output, const Tensor & input) const;
  virtual Tensor log_softmax_forward(const Tensor & input) const;
  virtual Tensor & log_softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, const Tensor & output) const;
  virtual Tensor log_softmax_backward(const Tensor & grad_output, const Tensor & input, const Tensor & output) const;
  virtual Tensor & prelu_out(Tensor & output, const Tensor & input, const Tensor & weight) const;
  virtual Tensor prelu(const Tensor & input, const Tensor & weight) const;
  virtual Tensor & prelu_forward_out(Tensor & output, const Tensor & input, const Tensor & weight) const;
  virtual Tensor prelu_forward(const Tensor & input, const Tensor & weight) const;
  virtual std::tuple<Tensor &,Tensor &> prelu_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & input, const Tensor & weight) const;
  virtual std::tuple<Tensor,Tensor> prelu_backward(const Tensor & grad_output, const Tensor & input, const Tensor & weight, std::array<bool, 2> output_mask={true, true}) const;
  virtual Tensor & rrelu_out(Tensor & output, const Tensor & input, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=false, bool inplace=false, Generator * generator=nullptr) const;
  virtual Tensor rrelu(const Tensor & input, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=false, bool inplace=false, Generator * generator=nullptr) const;
  virtual Tensor & rrelu_forward_out(Tensor & output, const Tensor & input, Scalar lower, Scalar upper, bool training, bool inplace, Generator * generator, const Tensor & noise) const;
  virtual Tensor rrelu_forward(const Tensor & input, Scalar lower, Scalar upper, bool training, bool inplace, Generator * generator, const Tensor & noise) const;
  virtual Tensor & rrelu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, Scalar lower, Scalar upper, bool training, bool inplace, const Tensor & noise) const;
  virtual Tensor rrelu_backward(const Tensor & grad_output, const Tensor & input, Scalar lower, Scalar upper, bool training, bool inplace, const Tensor & noise) const;
  virtual Tensor & softmax_out(Tensor & output, const Tensor & input, int64_t dim) const;
  virtual Tensor softmax(const Tensor & input, int64_t dim) const;
  virtual Tensor & softmax_forward_out(Tensor & output, const Tensor & input) const;
  virtual Tensor softmax_forward(const Tensor & input) const;
  virtual Tensor & softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, const Tensor & output) const;
  virtual Tensor softmax_backward(const Tensor & grad_output, const Tensor & input, const Tensor & output) const;
  virtual Tensor & softplus_out(Tensor & output, const Tensor & input, Scalar beta=1, Scalar threshold=20) const;
  virtual Tensor softplus(const Tensor & input, Scalar beta=1, Scalar threshold=20) const;
  virtual Tensor & softplus_forward_out(Tensor & output, const Tensor & input, Scalar beta, Scalar threshold) const;
  virtual Tensor softplus_forward(const Tensor & input, Scalar beta, Scalar threshold) const;
  virtual Tensor & softplus_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, Scalar beta, Scalar threshold, const Tensor & output) const;
  virtual Tensor softplus_backward(const Tensor & grad_output, const Tensor & input, Scalar beta, Scalar threshold, const Tensor & output) const;
  virtual Tensor & softshrink_out(Tensor & output, const Tensor & input, Scalar lambd=0.5) const;
  virtual Tensor softshrink(const Tensor & input, Scalar lambd=0.5) const;
  virtual Tensor & softshrink_forward_out(Tensor & output, const Tensor & input, Scalar lambd) const;
  virtual Tensor softshrink_forward(const Tensor & input, Scalar lambd) const;
  virtual Tensor & softshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, Scalar lambd) const;
  virtual Tensor softshrink_backward(const Tensor & grad_output, const Tensor & input, Scalar lambd) const;
  virtual Tensor & threshold_out(Tensor & output, const Tensor & input, Scalar threshold, Scalar value, bool inplace=false) const;
  virtual Tensor threshold(const Tensor & input, Scalar threshold, Scalar value, bool inplace=false) const;
  virtual Tensor & threshold_forward_out(Tensor & output, const Tensor & input, Scalar threshold, Scalar value, bool inplace) const;
  virtual Tensor threshold_forward(const Tensor & input, Scalar threshold, Scalar value, bool inplace) const;
  virtual Tensor & threshold_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, Scalar threshold, Scalar value, bool inplace) const;
  virtual Tensor threshold_backward(const Tensor & grad_output, const Tensor & input, Scalar threshold, Scalar value, bool inplace) const;
  virtual std::tuple<Tensor &,Tensor &> adaptive_max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & input, IntList output_size) const;
  virtual std::tuple<Tensor,Tensor> adaptive_max_pool2d(const Tensor & input, IntList output_size) const;
  virtual std::tuple<Tensor &,Tensor &> adaptive_max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & input, IntList output_size) const;
  virtual std::tuple<Tensor,Tensor> adaptive_max_pool2d_forward(const Tensor & input, IntList output_size) const;
  virtual Tensor & adaptive_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, const Tensor & indices) const;
  virtual Tensor adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & indices) const;
  virtual Tensor & avg_pool2d_out(Tensor & output, const Tensor & input, IntList kernel_size, IntList stride={}, IntList padding=0, bool ceil_mode=false, bool count_include_pad=false) const;
  virtual Tensor avg_pool2d(const Tensor & input, IntList kernel_size, IntList stride={}, IntList padding=0, bool ceil_mode=false, bool count_include_pad=false) const;
  virtual Tensor & avg_pool2d_forward_out(Tensor & output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const;
  virtual Tensor avg_pool2d_forward(const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const;
  virtual Tensor & avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const;
  virtual Tensor avg_pool2d_backward(const Tensor & grad_output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const;
  virtual Tensor & avg_pool3d_out(Tensor & output, const Tensor & input, IntList kernel_size, IntList stride={}, IntList padding=0, bool ceil_mode=false, bool count_include_pad=false) const;
  virtual Tensor avg_pool3d(const Tensor & input, IntList kernel_size, IntList stride={}, IntList padding=0, bool ceil_mode=false, bool count_include_pad=false) const;
  virtual Tensor & avg_pool3d_forward_out(Tensor & output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const;
  virtual Tensor avg_pool3d_forward(const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const;
  virtual Tensor & avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const;
  virtual Tensor avg_pool3d_backward(const Tensor & grad_output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const;
  virtual std::tuple<Tensor &,Tensor &> max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & input, IntList kernel_size, IntList stride={}, IntList padding=0, IntList dilation=1, bool ceil_mode=false) const;
  virtual std::tuple<Tensor,Tensor> max_pool2d(const Tensor & input, IntList kernel_size, IntList stride={}, IntList padding=0, IntList dilation=1, bool ceil_mode=false) const;
  virtual std::tuple<Tensor &,Tensor &> max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const;
  virtual std::tuple<Tensor,Tensor> max_pool2d_forward(const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const;
  virtual Tensor & max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const;
  virtual Tensor max_pool2d_backward(const Tensor & grad_output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const;
  virtual std::tuple<Tensor &,Tensor &> max_pool3d_out(Tensor & output, Tensor & indices, const Tensor & input, IntList kernel_size, IntList stride={}, IntList padding=0, IntList dilation=1, bool ceil_mode=false) const;
  virtual std::tuple<Tensor,Tensor> max_pool3d(const Tensor & input, IntList kernel_size, IntList stride={}, IntList padding=0, IntList dilation=1, bool ceil_mode=false) const;
  virtual std::tuple<Tensor &,Tensor &> max_pool3d_forward_out(Tensor & output, Tensor & indices, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const;
  virtual std::tuple<Tensor,Tensor> max_pool3d_forward(const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const;
  virtual Tensor & max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const;
  virtual Tensor max_pool3d_backward(const Tensor & grad_output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const;
  virtual Tensor & max_unpool2d_out(Tensor & output, const Tensor & input, const Tensor & indices, IntList output_size) const;
  virtual Tensor max_unpool2d(const Tensor & input, const Tensor & indices, IntList output_size) const;
  virtual Tensor & max_unpool2d_forward_out(Tensor & output, const Tensor & input, const Tensor & indices, IntList output_size) const;
  virtual Tensor max_unpool2d_forward(const Tensor & input, const Tensor & indices, IntList output_size) const;
  virtual Tensor & max_unpool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, const Tensor & indices, IntList output_size) const;
  virtual Tensor max_unpool2d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & indices, IntList output_size) const;
  virtual Tensor & max_unpool3d_out(Tensor & output, const Tensor & input, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const;
  virtual Tensor max_unpool3d(const Tensor & input, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const;
  virtual Tensor & max_unpool3d_forward_out(Tensor & output, const Tensor & input, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const;
  virtual Tensor max_unpool3d_forward(const Tensor & input, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const;
  virtual Tensor & max_unpool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const;
  virtual Tensor max_unpool3d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const;
  virtual Tensor & _sigmoid_out(Tensor & output, const Tensor & input) const;
  virtual Tensor _sigmoid(const Tensor & input) const;
  virtual Tensor & _sigmoid_forward_out(Tensor & output, const Tensor & input) const;
  virtual Tensor _sigmoid_forward(const Tensor & input) const;
  virtual Tensor & _sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const;
  virtual Tensor _sigmoid_backward(const Tensor & grad_output, const Tensor & output) const;
  virtual Tensor & _tanh_out(Tensor & output, const Tensor & input) const;
  virtual Tensor _tanh(const Tensor & input) const;
  virtual Tensor & _tanh_forward_out(Tensor & output, const Tensor & input) const;
  virtual Tensor _tanh_forward(const Tensor & input) const;
  virtual Tensor & _tanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const;
  virtual Tensor _tanh_backward(const Tensor & grad_output, const Tensor & output) const;
  virtual Tensor & batch_norm_out(Tensor & output, const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) const;
  virtual Tensor batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) const;
  virtual Tensor & batch_norm_forward_out(Tensor & output, const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, const Tensor & save_mean, const Tensor & save_std) const;
  virtual Tensor batch_norm_forward(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, const Tensor & save_mean, const Tensor & save_std) const;
  virtual std::tuple<Tensor &,Tensor &,Tensor &> batch_norm_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & input, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std) const;
  virtual std::tuple<Tensor,Tensor,Tensor> batch_norm_backward(const Tensor & grad_output, const Tensor & input, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std, std::array<bool, 3> output_mask={true, true, true}) const;
  virtual Tensor & conv_transpose2d_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList output_padding=0, IntList dilation=1) const;
  virtual Tensor conv_transpose2d(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList output_padding=0, IntList dilation=1) const;
  virtual Tensor & conv_transpose2d_forward_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones) const;
  virtual Tensor conv_transpose2d_forward(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones) const;
  virtual std::tuple<Tensor &,Tensor &,Tensor &> conv_transpose2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones) const;
  virtual std::tuple<Tensor,Tensor,Tensor> conv_transpose2d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool, 3> output_mask={true, true, true}) const;
  virtual Tensor & conv_transpose3d_out(Tensor & output, const Tensor & input, const Tensor & weight, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList output_padding=0, IntList dilation=1) const;
  virtual Tensor conv_transpose3d(const Tensor & input, const Tensor & weight, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList output_padding=0, IntList dilation=1) const;
  virtual Tensor & conv_transpose3d_forward_out(Tensor & output, const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input) const;
  virtual Tensor conv_transpose3d_forward(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input) const;
  virtual std::tuple<Tensor &,Tensor &,Tensor &> conv_transpose3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input) const;
  virtual std::tuple<Tensor,Tensor,Tensor> conv_transpose3d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input, std::array<bool, 3> output_mask={true, true, true}) const;
  virtual Tensor & conv2d_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0) const;
  virtual Tensor conv2d(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0) const;
  virtual Tensor & conv2d_forward_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) const;
  virtual Tensor conv2d_forward(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) const;
  virtual std::tuple<Tensor &,Tensor &,Tensor &> conv2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) const;
  virtual std::tuple<Tensor,Tensor,Tensor> conv2d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool, 3> output_mask={true, true, true}) const;
  virtual Tensor & conv3d_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0) const;
  virtual Tensor conv3d(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0) const;
  virtual Tensor & conv3d_forward_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, const Tensor & finput) const;
  virtual Tensor conv3d_forward(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, const Tensor & finput) const;
  virtual std::tuple<Tensor &,Tensor &,Tensor &> conv3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) const;
  virtual std::tuple<Tensor,Tensor,Tensor> conv3d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool, 3> output_mask={true, true, true}) const;
  virtual Tensor & conv_dilated2d_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList dilation=1) const;
  virtual Tensor conv_dilated2d(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList dilation=1) const;
  virtual Tensor & conv_dilated2d_forward_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const;
  virtual Tensor conv_dilated2d_forward(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const;
  virtual std::tuple<Tensor &,Tensor &,Tensor &> conv_dilated2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const;
  virtual std::tuple<Tensor,Tensor,Tensor> conv_dilated2d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool, 3> output_mask={true, true, true}) const;
  virtual Tensor & conv_dilated3d_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList dilation=1) const;
  virtual Tensor conv_dilated3d(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList dilation=1) const;
  virtual Tensor & conv_dilated3d_forward_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const;
  virtual Tensor conv_dilated3d_forward(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const;
  virtual std::tuple<Tensor &,Tensor &,Tensor &> conv_dilated3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const;
  virtual std::tuple<Tensor,Tensor,Tensor> conv_dilated3d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool, 3> output_mask={true, true, true}) const;
  virtual std::vector<Tensor> split(Tensor self, int64_t split_size, int64_t dim=0) const;
  virtual std::vector<Tensor> chunk(Tensor self, int64_t chunks, int64_t dim=0) const;
protected:
  Context* context;
};


}
