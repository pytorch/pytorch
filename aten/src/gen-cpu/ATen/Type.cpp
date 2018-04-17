#include "ATen/Type.h"
#include "ATen/Tensor.h"
#include "ATen/Storage.h"
#include "ATen/Scalar.h"
#include "ATen/SparseTensorRef.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/UndefinedType.h"

#include <iostream>
#include "ATen/CPUByteType.h"
#include "ATen/CPUCharType.h"
#include "ATen/CPUDoubleType.h"
#include "ATen/CPUFloatType.h"
#include "ATen/CPUIntType.h"
#include "ATen/CPULongType.h"
#include "ATen/CPUShortType.h"
#include "ATen/CPUHalfType.h"
#include "ATen/SparseCPUByteType.h"
#include "ATen/SparseCPUCharType.h"
#include "ATen/SparseCPUDoubleType.h"
#include "ATen/SparseCPUFloatType.h"
#include "ATen/SparseCPUIntType.h"
#include "ATen/SparseCPULongType.h"
#include "ATen/SparseCPUShortType.h"

namespace at {

void Type::registerAll(Context * context) {
  context->type_registry[static_cast<int>(Backend::CPU)][static_cast<int>(ScalarType::Byte)].reset(new CPUByteType(context));
  context->type_registry[static_cast<int>(Backend::CPU)][static_cast<int>(ScalarType::Char)].reset(new CPUCharType(context));
  context->type_registry[static_cast<int>(Backend::CPU)][static_cast<int>(ScalarType::Double)].reset(new CPUDoubleType(context));
  context->type_registry[static_cast<int>(Backend::CPU)][static_cast<int>(ScalarType::Float)].reset(new CPUFloatType(context));
  context->type_registry[static_cast<int>(Backend::CPU)][static_cast<int>(ScalarType::Int)].reset(new CPUIntType(context));
  context->type_registry[static_cast<int>(Backend::CPU)][static_cast<int>(ScalarType::Long)].reset(new CPULongType(context));
  context->type_registry[static_cast<int>(Backend::CPU)][static_cast<int>(ScalarType::Short)].reset(new CPUShortType(context));
  context->type_registry[static_cast<int>(Backend::CPU)][static_cast<int>(ScalarType::Half)].reset(new CPUHalfType(context));
  context->type_registry[static_cast<int>(Backend::SparseCPU)][static_cast<int>(ScalarType::Byte)].reset(new SparseCPUByteType(context));
  context->type_registry[static_cast<int>(Backend::SparseCPU)][static_cast<int>(ScalarType::Char)].reset(new SparseCPUCharType(context));
  context->type_registry[static_cast<int>(Backend::SparseCPU)][static_cast<int>(ScalarType::Double)].reset(new SparseCPUDoubleType(context));
  context->type_registry[static_cast<int>(Backend::SparseCPU)][static_cast<int>(ScalarType::Float)].reset(new SparseCPUFloatType(context));
  context->type_registry[static_cast<int>(Backend::SparseCPU)][static_cast<int>(ScalarType::Int)].reset(new SparseCPUIntType(context));
  context->type_registry[static_cast<int>(Backend::SparseCPU)][static_cast<int>(ScalarType::Long)].reset(new SparseCPULongType(context));
  context->type_registry[static_cast<int>(Backend::SparseCPU)][static_cast<int>(ScalarType::Short)].reset(new SparseCPUShortType(context));
  context->type_registry[static_cast<int>(Backend::Undefined)][static_cast<int>(ScalarType::Undefined)].reset(new UndefinedType(context));
}

Tensor & Type::copy_(Tensor & self, const Tensor & src, bool non_blocking) const {
  Tensor b_src;
  std::tie(b_src) = expand_inplace(self, src, "copy");
  return s_copy_(self, b_src, non_blocking);
}

Tensor Type::copy(const Tensor & src, bool non_blocking) const {
  AT_ASSERT(src.defined(), "attempt to copy an undefined tensor");
  if (is_sparse()) {
    auto indices = src._indices();
    auto values = src._values();
    auto & this_dense = toBackend(is_cuda() ? Backend::CUDA : Backend::CPU);
    auto & this_dense_idx = this_dense.toScalarType(ScalarType::Long);
    auto indices_copy = this_dense_idx.copy(indices, non_blocking);
    auto values_copy = this_dense.copy(values, non_blocking);
    return sparse_coo_tensor(indices_copy, values_copy, src.sizes());
  } else {
    Tensor r = this->tensor(src.sizes());
    r.copy_(src, non_blocking);
    return r;
  }
}

Type & Type::toBackend(Backend b) const {
  return context->getType(b,scalarType());
}
Type & Type::toScalarType(ScalarType s) const {
  return context->getType(backend(),s);
}
static std::vector<int64_t> defaultStrides(IntList sizes) {
  std::vector<int64_t> strides(sizes.size());
  int64_t stride = 1;
  for(size_t i = sizes.size(); i > 0; --i) {
    strides[i-1] = stride;
    stride *= sizes[i-1];
  }
  return strides;
}
static int64_t computeStorageSize(IntList sizes, IntList strides) {
  // size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride
  int64_t size = 1;
  for(size_t i = 0; i < sizes.size(); i++) {
    if(sizes[i] == 0) {
      return 0;
    }
    size += strides[i]*(sizes[i]-1);
  }
  return size;
}
Tensor Type::tensorFromBlob(void * data, IntList sizes, const std::function<void(void*)> & deleter) const {
  return tensorFromBlob(data, sizes, defaultStrides(sizes), deleter);
}
Tensor Type::tensorFromBlob(void * data, IntList sizes, IntList strides, const std::function<void(void*)> & deleter) const {
  auto storage = storageFromBlob(data, computeStorageSize(sizes, strides), deleter);
  return tensor(*storage, 0, sizes, strides);
}
Tensor Type::tensorWithAllocator(IntList sizes, std::unique_ptr<Allocator> allocator) const {
  return tensorWithAllocator(sizes, defaultStrides(sizes), std::move(allocator));
}
Tensor Type::tensorWithAllocator(IntList sizes, IntList strides, std::unique_ptr<Allocator> allocator) const {
  auto storage = storageWithAllocator(computeStorageSize(sizes, strides), std::move(allocator));
  return tensor(*storage, 0, sizes, strides);
}
Tensor Type::scalarTensor(Scalar s) const {
  if(s.isBackedByTensor())
    return Tensor(s.t).toType(*this);
  return tensor({}).fill_(s);
}

bool Type::operator==(const Type& other) const {
  return this == &other;
}
bool Type::operator!=(const Type& other) const {
  return this != &other;
}

int64_t Type::storage_offset(const Tensor & self) const {
    AT_ERROR("storage_offset is not implemented for type %s", toString());
}
Tensor & Type::resize_(Tensor & self, IntList size) const {
    AT_ERROR("resize_ is not implemented for type %s", toString());
}
int64_t Type::numel(const Tensor & self) const {
    AT_ERROR("numel is not implemented for type %s", toString());
}
Tensor & Type::set_(Tensor & self, Storage & storage) const {
    AT_ERROR("set_ is not implemented for type %s", toString());
}
Tensor & Type::set_(Tensor & self, Storage & sourceStorage, int64_t storage_offset, IntList size, IntList stride) const {
    AT_ERROR("set_ is not implemented for type %s", toString());
}
Tensor & Type::set_(Tensor & self, const Tensor & source) const {
    AT_ERROR("set_ is not implemented for type %s", toString());
}
Tensor & Type::set_(Tensor & self) const {
    AT_ERROR("set_ is not implemented for type %s", toString());
}
Tensor & Type::fill_(Tensor & self, Scalar value) const {
    AT_ERROR("fill_ is not implemented for type %s", toString());
}
Tensor & Type::fill_(Tensor & self, const Tensor & value) const {
    AT_ERROR("fill_ is not implemented for type %s", toString());
}
bool Type::is_contiguous(const Tensor & self) const {
    AT_ERROR("is_contiguous is not implemented for type %s", toString());
}
bool Type::is_set_to(const Tensor & self, const Tensor & tensor) const {
    AT_ERROR("is_set_to is not implemented for type %s", toString());
}
Tensor & Type::s_masked_fill_(Tensor & self, const Tensor & mask, Scalar value) const {
    AT_ERROR("s_masked_fill_ is not implemented for type %s", toString());
}
Tensor & Type::masked_fill_(Tensor & self, const Tensor & mask, Scalar value) const {
    Tensor b_mask;
    std::tie(b_mask) = expand_inplace(self, mask, "masked_fill_");
    return s_masked_fill_(self, b_mask, value);
}
Tensor & Type::s_masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) const {
    AT_ERROR("s_masked_fill_ is not implemented for type %s", toString());
}
Tensor & Type::masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) const {
    Tensor b_mask;
    std::tie(b_mask) = expand_inplace(self, mask, "masked_fill_");
    return s_masked_fill_(self, b_mask, value);
}
Tensor & Type::s_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) const {
    AT_ERROR("s_masked_scatter_ is not implemented for type %s", toString());
}
Tensor & Type::masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) const {
    Tensor b_mask;
    std::tie(b_mask) = expand_inplace(self, mask, "masked_scatter_");
    return s_masked_scatter_(self, b_mask, source);
}
Tensor & Type::s_masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask) const {
    AT_ERROR("s_masked_select_out is not implemented for type %s", toString());
}
Tensor & Type::masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask) const {
    Tensor b_self, b_mask;
    std::tie(b_self, b_mask) = expand_outplace(self, mask, "masked_select_out");
    return s_masked_select_out(result, b_self, b_mask);
}
Tensor Type::s_masked_select(const Tensor & self, const Tensor & mask) const {
    AT_ERROR("s_masked_select is not implemented for type %s", toString());
}
Tensor Type::masked_select(const Tensor & self, const Tensor & mask) const {
    Tensor b_self, b_mask;
    std::tie(b_self, b_mask) = expand_outplace(self, mask, "masked_select");
    return s_masked_select(b_self, b_mask);
}
Tensor Type::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    AT_ERROR("transpose is not implemented for type %s", toString());
}
Tensor Type::t(const Tensor & self) const {
    AT_ERROR("t is not implemented for type %s", toString());
}
Tensor & Type::nonzero_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("nonzero_out is not implemented for type %s", toString());
}
Tensor Type::nonzero(const Tensor & self) const {
    AT_ERROR("nonzero is not implemented for type %s", toString());
}
Tensor Type::contiguous(const Tensor & self) const {
    AT_ERROR("contiguous is not implemented for type %s", toString());
}
Tensor Type::clone(const Tensor & self) const {
    AT_ERROR("clone is not implemented for type %s", toString());
}
Tensor Type::view(const Tensor & self, IntList size) const {
    AT_ERROR("view is not implemented for type %s", toString());
}
Tensor & Type::resize_as_(Tensor & self, const Tensor & the_template) const {
    AT_ERROR("resize_as_ is not implemented for type %s", toString());
}
Tensor & Type::index_select_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    AT_ERROR("index_select_out is not implemented for type %s", toString());
}
Tensor Type::index_select(const Tensor & self, int64_t dim, const Tensor & index) const {
    AT_ERROR("index_select is not implemented for type %s", toString());
}
Tensor & Type::_indexCopy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    AT_ERROR("_indexCopy_ is not implemented for type %s", toString());
}
Tensor & Type::take_out(Tensor & result, const Tensor & self, const Tensor & index) const {
    AT_ERROR("take_out is not implemented for type %s", toString());
}
Tensor Type::take(const Tensor & self, const Tensor & index) const {
    AT_ERROR("take is not implemented for type %s", toString());
}
Tensor & Type::put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) const {
    AT_ERROR("put_ is not implemented for type %s", toString());
}
Tensor & Type::index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    AT_ERROR("index_add_ is not implemented for type %s", toString());
}
Tensor & Type::index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    AT_ERROR("index_fill_ is not implemented for type %s", toString());
}
Tensor & Type::index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) const {
    AT_ERROR("index_fill_ is not implemented for type %s", toString());
}
Tensor Type::unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const {
    AT_ERROR("unfold is not implemented for type %s", toString());
}
Tensor & Type::_range_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    AT_ERROR("_range_out is not implemented for type %s", toString());
}
Tensor Type::_range(Scalar start, Scalar end, Scalar step) const {
    AT_ERROR("_range is not implemented for type %s", toString());
}
Tensor & Type::_arange_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    AT_ERROR("_arange_out is not implemented for type %s", toString());
}
Tensor Type::_arange(Scalar start, Scalar end, Scalar step) const {
    AT_ERROR("_arange is not implemented for type %s", toString());
}
Tensor & Type::_arange_out(Tensor & result, Scalar end) const {
    AT_ERROR("_arange_out is not implemented for type %s", toString());
}
Tensor Type::_arange(Scalar end) const {
    AT_ERROR("_arange is not implemented for type %s", toString());
}
Tensor & Type::scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    AT_ERROR("scatter_ is not implemented for type %s", toString());
}
Tensor & Type::scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    AT_ERROR("scatter_ is not implemented for type %s", toString());
}
Tensor & Type::scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    AT_ERROR("scatter_add_ is not implemented for type %s", toString());
}
Tensor & Type::gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    AT_ERROR("gather_out is not implemented for type %s", toString());
}
Tensor Type::gather(const Tensor & self, int64_t dim, const Tensor & index) const {
    AT_ERROR("gather is not implemented for type %s", toString());
}
void* Type::data_ptr(const Tensor & self) const {
    AT_ERROR("data_ptr is not implemented for type %s", toString());
}
bool Type::equal(const Tensor & self, const Tensor & other) const {
    AT_ERROR("equal is not implemented for type %s", toString());
}
Tensor & Type::__and___out(Tensor & result, const Tensor & self, Scalar other) const {
    AT_ERROR("__and___out is not implemented for type %s", toString());
}
Tensor Type::__and__(const Tensor & self, Scalar other) const {
    AT_ERROR("__and__ is not implemented for type %s", toString());
}
Tensor & Type::s___and___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    AT_ERROR("s___and___out is not implemented for type %s", toString());
}
Tensor & Type::__and___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "__and___out");
    return s___and___out(result, b_self, b_other);
}
Tensor Type::s___and__(const Tensor & self, const Tensor & other) const {
    AT_ERROR("s___and__ is not implemented for type %s", toString());
}
Tensor Type::__and__(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "__and__");
    return s___and__(b_self, b_other);
}
Tensor & Type::__iand__(Tensor & self, Scalar other) const {
    AT_ERROR("__iand__ is not implemented for type %s", toString());
}
Tensor & Type::s___iand__(Tensor & self, const Tensor & other) const {
    AT_ERROR("s___iand__ is not implemented for type %s", toString());
}
Tensor & Type::__iand__(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "__iand__");
    return s___iand__(self, b_other);
}
Tensor & Type::__or___out(Tensor & result, const Tensor & self, Scalar other) const {
    AT_ERROR("__or___out is not implemented for type %s", toString());
}
Tensor Type::__or__(const Tensor & self, Scalar other) const {
    AT_ERROR("__or__ is not implemented for type %s", toString());
}
Tensor & Type::s___or___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    AT_ERROR("s___or___out is not implemented for type %s", toString());
}
Tensor & Type::__or___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "__or___out");
    return s___or___out(result, b_self, b_other);
}
Tensor Type::s___or__(const Tensor & self, const Tensor & other) const {
    AT_ERROR("s___or__ is not implemented for type %s", toString());
}
Tensor Type::__or__(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "__or__");
    return s___or__(b_self, b_other);
}
Tensor & Type::__ior__(Tensor & self, Scalar other) const {
    AT_ERROR("__ior__ is not implemented for type %s", toString());
}
Tensor & Type::s___ior__(Tensor & self, const Tensor & other) const {
    AT_ERROR("s___ior__ is not implemented for type %s", toString());
}
Tensor & Type::__ior__(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "__ior__");
    return s___ior__(self, b_other);
}
Tensor & Type::__xor___out(Tensor & result, const Tensor & self, Scalar other) const {
    AT_ERROR("__xor___out is not implemented for type %s", toString());
}
Tensor Type::__xor__(const Tensor & self, Scalar other) const {
    AT_ERROR("__xor__ is not implemented for type %s", toString());
}
Tensor & Type::s___xor___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    AT_ERROR("s___xor___out is not implemented for type %s", toString());
}
Tensor & Type::__xor___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "__xor___out");
    return s___xor___out(result, b_self, b_other);
}
Tensor Type::s___xor__(const Tensor & self, const Tensor & other) const {
    AT_ERROR("s___xor__ is not implemented for type %s", toString());
}
Tensor Type::__xor__(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "__xor__");
    return s___xor__(b_self, b_other);
}
Tensor & Type::__ixor__(Tensor & self, Scalar other) const {
    AT_ERROR("__ixor__ is not implemented for type %s", toString());
}
Tensor & Type::s___ixor__(Tensor & self, const Tensor & other) const {
    AT_ERROR("s___ixor__ is not implemented for type %s", toString());
}
Tensor & Type::__ixor__(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "__ixor__");
    return s___ixor__(self, b_other);
}
Tensor & Type::__lshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    AT_ERROR("__lshift___out is not implemented for type %s", toString());
}
Tensor Type::__lshift__(const Tensor & self, Scalar other) const {
    AT_ERROR("__lshift__ is not implemented for type %s", toString());
}
Tensor & Type::s___lshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    AT_ERROR("s___lshift___out is not implemented for type %s", toString());
}
Tensor & Type::__lshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "__lshift___out");
    return s___lshift___out(result, b_self, b_other);
}
Tensor Type::s___lshift__(const Tensor & self, const Tensor & other) const {
    AT_ERROR("s___lshift__ is not implemented for type %s", toString());
}
Tensor Type::__lshift__(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "__lshift__");
    return s___lshift__(b_self, b_other);
}
Tensor & Type::__ilshift__(Tensor & self, Scalar other) const {
    AT_ERROR("__ilshift__ is not implemented for type %s", toString());
}
Tensor & Type::s___ilshift__(Tensor & self, const Tensor & other) const {
    AT_ERROR("s___ilshift__ is not implemented for type %s", toString());
}
Tensor & Type::__ilshift__(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "__ilshift__");
    return s___ilshift__(self, b_other);
}
Tensor & Type::__rshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    AT_ERROR("__rshift___out is not implemented for type %s", toString());
}
Tensor Type::__rshift__(const Tensor & self, Scalar other) const {
    AT_ERROR("__rshift__ is not implemented for type %s", toString());
}
Tensor & Type::s___rshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    AT_ERROR("s___rshift___out is not implemented for type %s", toString());
}
Tensor & Type::__rshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "__rshift___out");
    return s___rshift___out(result, b_self, b_other);
}
Tensor Type::s___rshift__(const Tensor & self, const Tensor & other) const {
    AT_ERROR("s___rshift__ is not implemented for type %s", toString());
}
Tensor Type::__rshift__(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "__rshift__");
    return s___rshift__(b_self, b_other);
}
Tensor & Type::__irshift__(Tensor & self, Scalar other) const {
    AT_ERROR("__irshift__ is not implemented for type %s", toString());
}
Tensor & Type::s___irshift__(Tensor & self, const Tensor & other) const {
    AT_ERROR("s___irshift__ is not implemented for type %s", toString());
}
Tensor & Type::__irshift__(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "__irshift__");
    return s___irshift__(self, b_other);
}
Tensor & Type::lt_out(Tensor & result, const Tensor & self, Scalar other) const {
    AT_ERROR("lt_out is not implemented for type %s", toString());
}
Tensor Type::lt(const Tensor & self, Scalar other) const {
    AT_ERROR("lt is not implemented for type %s", toString());
}
Tensor & Type::s_lt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_lt_out is not implemented for type %s", toString());
}
Tensor & Type::lt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "lt_out");
    return s_lt_out(result, b_self, b_other);
}
Tensor Type::s_lt(const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_lt is not implemented for type %s", toString());
}
Tensor Type::lt(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "lt");
    return s_lt(b_self, b_other);
}
Tensor & Type::lt_(Tensor & self, Scalar other) const {
    AT_ERROR("lt_ is not implemented for type %s", toString());
}
Tensor & Type::s_lt_(Tensor & self, const Tensor & other) const {
    AT_ERROR("s_lt_ is not implemented for type %s", toString());
}
Tensor & Type::lt_(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "lt_");
    return s_lt_(self, b_other);
}
Tensor & Type::gt_out(Tensor & result, const Tensor & self, Scalar other) const {
    AT_ERROR("gt_out is not implemented for type %s", toString());
}
Tensor Type::gt(const Tensor & self, Scalar other) const {
    AT_ERROR("gt is not implemented for type %s", toString());
}
Tensor & Type::s_gt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_gt_out is not implemented for type %s", toString());
}
Tensor & Type::gt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "gt_out");
    return s_gt_out(result, b_self, b_other);
}
Tensor Type::s_gt(const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_gt is not implemented for type %s", toString());
}
Tensor Type::gt(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "gt");
    return s_gt(b_self, b_other);
}
Tensor & Type::gt_(Tensor & self, Scalar other) const {
    AT_ERROR("gt_ is not implemented for type %s", toString());
}
Tensor & Type::s_gt_(Tensor & self, const Tensor & other) const {
    AT_ERROR("s_gt_ is not implemented for type %s", toString());
}
Tensor & Type::gt_(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "gt_");
    return s_gt_(self, b_other);
}
Tensor & Type::le_out(Tensor & result, const Tensor & self, Scalar other) const {
    AT_ERROR("le_out is not implemented for type %s", toString());
}
Tensor Type::le(const Tensor & self, Scalar other) const {
    AT_ERROR("le is not implemented for type %s", toString());
}
Tensor & Type::s_le_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_le_out is not implemented for type %s", toString());
}
Tensor & Type::le_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "le_out");
    return s_le_out(result, b_self, b_other);
}
Tensor Type::s_le(const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_le is not implemented for type %s", toString());
}
Tensor Type::le(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "le");
    return s_le(b_self, b_other);
}
Tensor & Type::le_(Tensor & self, Scalar other) const {
    AT_ERROR("le_ is not implemented for type %s", toString());
}
Tensor & Type::s_le_(Tensor & self, const Tensor & other) const {
    AT_ERROR("s_le_ is not implemented for type %s", toString());
}
Tensor & Type::le_(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "le_");
    return s_le_(self, b_other);
}
Tensor & Type::ge_out(Tensor & result, const Tensor & self, Scalar other) const {
    AT_ERROR("ge_out is not implemented for type %s", toString());
}
Tensor Type::ge(const Tensor & self, Scalar other) const {
    AT_ERROR("ge is not implemented for type %s", toString());
}
Tensor & Type::s_ge_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_ge_out is not implemented for type %s", toString());
}
Tensor & Type::ge_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "ge_out");
    return s_ge_out(result, b_self, b_other);
}
Tensor Type::s_ge(const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_ge is not implemented for type %s", toString());
}
Tensor Type::ge(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "ge");
    return s_ge(b_self, b_other);
}
Tensor & Type::ge_(Tensor & self, Scalar other) const {
    AT_ERROR("ge_ is not implemented for type %s", toString());
}
Tensor & Type::s_ge_(Tensor & self, const Tensor & other) const {
    AT_ERROR("s_ge_ is not implemented for type %s", toString());
}
Tensor & Type::ge_(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "ge_");
    return s_ge_(self, b_other);
}
Tensor & Type::eq_out(Tensor & result, const Tensor & self, Scalar other) const {
    AT_ERROR("eq_out is not implemented for type %s", toString());
}
Tensor Type::eq(const Tensor & self, Scalar other) const {
    AT_ERROR("eq is not implemented for type %s", toString());
}
Tensor & Type::s_eq_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_eq_out is not implemented for type %s", toString());
}
Tensor & Type::eq_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "eq_out");
    return s_eq_out(result, b_self, b_other);
}
Tensor Type::s_eq(const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_eq is not implemented for type %s", toString());
}
Tensor Type::eq(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "eq");
    return s_eq(b_self, b_other);
}
Tensor & Type::eq_(Tensor & self, Scalar other) const {
    AT_ERROR("eq_ is not implemented for type %s", toString());
}
Tensor & Type::s_eq_(Tensor & self, const Tensor & other) const {
    AT_ERROR("s_eq_ is not implemented for type %s", toString());
}
Tensor & Type::eq_(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "eq_");
    return s_eq_(self, b_other);
}
Tensor & Type::ne_out(Tensor & result, const Tensor & self, Scalar other) const {
    AT_ERROR("ne_out is not implemented for type %s", toString());
}
Tensor Type::ne(const Tensor & self, Scalar other) const {
    AT_ERROR("ne is not implemented for type %s", toString());
}
Tensor & Type::s_ne_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_ne_out is not implemented for type %s", toString());
}
Tensor & Type::ne_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "ne_out");
    return s_ne_out(result, b_self, b_other);
}
Tensor Type::s_ne(const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_ne is not implemented for type %s", toString());
}
Tensor Type::ne(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "ne");
    return s_ne(b_self, b_other);
}
Tensor & Type::ne_(Tensor & self, Scalar other) const {
    AT_ERROR("ne_ is not implemented for type %s", toString());
}
Tensor & Type::s_ne_(Tensor & self, const Tensor & other) const {
    AT_ERROR("s_ne_ is not implemented for type %s", toString());
}
Tensor & Type::ne_(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "ne_");
    return s_ne_(self, b_other);
}
std::tuple<Tensor &,Tensor &> Type::min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    AT_ERROR("min_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::min(const Tensor & self, int64_t dim, bool keepdim) const {
    AT_ERROR("min is not implemented for type %s", toString());
}
Tensor & Type::s_min_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_min_out is not implemented for type %s", toString());
}
Tensor & Type::min_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "min_out");
    return s_min_out(result, b_self, b_other);
}
Tensor Type::s_min(const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_min is not implemented for type %s", toString());
}
Tensor Type::min(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "min");
    return s_min(b_self, b_other);
}
Tensor Type::min(const Tensor & self) const {
    AT_ERROR("min is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    AT_ERROR("max_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::max(const Tensor & self, int64_t dim, bool keepdim) const {
    AT_ERROR("max is not implemented for type %s", toString());
}
Tensor & Type::s_max_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_max_out is not implemented for type %s", toString());
}
Tensor & Type::max_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "max_out");
    return s_max_out(result, b_self, b_other);
}
Tensor Type::s_max(const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_max is not implemented for type %s", toString());
}
Tensor Type::max(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "max");
    return s_max(b_self, b_other);
}
Tensor Type::max(const Tensor & self) const {
    AT_ERROR("max is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::kthvalue_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
    AT_ERROR("kthvalue_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
    AT_ERROR("kthvalue is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    AT_ERROR("mode_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::mode(const Tensor & self, int64_t dim, bool keepdim) const {
    AT_ERROR("mode is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::median_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    AT_ERROR("median_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::median(const Tensor & self, int64_t dim, bool keepdim) const {
    AT_ERROR("median is not implemented for type %s", toString());
}
Tensor Type::median(const Tensor & self) const {
    AT_ERROR("median is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) const {
    AT_ERROR("sort_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::sort(const Tensor & self, int64_t dim, bool descending) const {
    AT_ERROR("sort is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    AT_ERROR("topk_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    AT_ERROR("topk is not implemented for type %s", toString());
}
Tensor Type::all(const Tensor & self) const {
    AT_ERROR("all is not implemented for type %s", toString());
}
Tensor Type::any(const Tensor & self) const {
    AT_ERROR("any is not implemented for type %s", toString());
}
int64_t Type::get_device(const Tensor & self) const {
    AT_ERROR("get_device is not implemented for type %s", toString());
}
Tensor & Type::_abs_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("_abs_out is not implemented for type %s", toString());
}
Tensor Type::_abs(const Tensor & self) const {
    AT_ERROR("_abs is not implemented for type %s", toString());
}
Tensor & Type::sigmoid_(Tensor & self) const {
    AT_ERROR("sigmoid_ is not implemented for type %s", toString());
}
Tensor & Type::sigmoid_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("sigmoid_out is not implemented for type %s", toString());
}
Tensor Type::sigmoid(const Tensor & self) const {
    AT_ERROR("sigmoid is not implemented for type %s", toString());
}
Tensor & Type::_log_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("_log_out is not implemented for type %s", toString());
}
Tensor Type::_log(const Tensor & self) const {
    AT_ERROR("_log is not implemented for type %s", toString());
}
Tensor & Type::log10_(Tensor & self) const {
    AT_ERROR("log10_ is not implemented for type %s", toString());
}
Tensor & Type::log10_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("log10_out is not implemented for type %s", toString());
}
Tensor Type::log10(const Tensor & self) const {
    AT_ERROR("log10 is not implemented for type %s", toString());
}
Tensor & Type::log1p_(Tensor & self) const {
    AT_ERROR("log1p_ is not implemented for type %s", toString());
}
Tensor & Type::log1p_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("log1p_out is not implemented for type %s", toString());
}
Tensor Type::log1p(const Tensor & self) const {
    AT_ERROR("log1p is not implemented for type %s", toString());
}
Tensor & Type::log2_(Tensor & self) const {
    AT_ERROR("log2_ is not implemented for type %s", toString());
}
Tensor & Type::log2_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("log2_out is not implemented for type %s", toString());
}
Tensor Type::log2(const Tensor & self) const {
    AT_ERROR("log2 is not implemented for type %s", toString());
}
Tensor & Type::lgamma_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("lgamma_out is not implemented for type %s", toString());
}
Tensor Type::lgamma(const Tensor & self) const {
    AT_ERROR("lgamma is not implemented for type %s", toString());
}
Tensor & Type::lgamma_(Tensor & self) const {
    AT_ERROR("lgamma_ is not implemented for type %s", toString());
}
Tensor & Type::digamma_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("digamma_out is not implemented for type %s", toString());
}
Tensor Type::digamma(const Tensor & self) const {
    AT_ERROR("digamma is not implemented for type %s", toString());
}
Tensor & Type::digamma_(Tensor & self) const {
    AT_ERROR("digamma_ is not implemented for type %s", toString());
}
Tensor & Type::polygamma_out(Tensor & result, int64_t n, const Tensor & self) const {
    AT_ERROR("polygamma_out is not implemented for type %s", toString());
}
Tensor Type::polygamma(int64_t n, const Tensor & self) const {
    AT_ERROR("polygamma is not implemented for type %s", toString());
}
Tensor & Type::polygamma_(Tensor & self, int64_t n) const {
    AT_ERROR("polygamma_ is not implemented for type %s", toString());
}
Tensor & Type::_exp_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("_exp_out is not implemented for type %s", toString());
}
Tensor Type::_exp(const Tensor & self) const {
    AT_ERROR("_exp is not implemented for type %s", toString());
}
Tensor & Type::expm1_(Tensor & self) const {
    AT_ERROR("expm1_ is not implemented for type %s", toString());
}
Tensor & Type::expm1_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("expm1_out is not implemented for type %s", toString());
}
Tensor Type::expm1(const Tensor & self) const {
    AT_ERROR("expm1 is not implemented for type %s", toString());
}
Tensor & Type::_cos_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("_cos_out is not implemented for type %s", toString());
}
Tensor Type::_cos(const Tensor & self) const {
    AT_ERROR("_cos is not implemented for type %s", toString());
}
Tensor & Type::acos_(Tensor & self) const {
    AT_ERROR("acos_ is not implemented for type %s", toString());
}
Tensor & Type::acos_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("acos_out is not implemented for type %s", toString());
}
Tensor Type::acos(const Tensor & self) const {
    AT_ERROR("acos is not implemented for type %s", toString());
}
Tensor & Type::cosh_(Tensor & self) const {
    AT_ERROR("cosh_ is not implemented for type %s", toString());
}
Tensor & Type::cosh_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("cosh_out is not implemented for type %s", toString());
}
Tensor Type::cosh(const Tensor & self) const {
    AT_ERROR("cosh is not implemented for type %s", toString());
}
Tensor & Type::_sin_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("_sin_out is not implemented for type %s", toString());
}
Tensor Type::_sin(const Tensor & self) const {
    AT_ERROR("_sin is not implemented for type %s", toString());
}
Tensor & Type::asin_(Tensor & self) const {
    AT_ERROR("asin_ is not implemented for type %s", toString());
}
Tensor & Type::asin_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("asin_out is not implemented for type %s", toString());
}
Tensor Type::asin(const Tensor & self) const {
    AT_ERROR("asin is not implemented for type %s", toString());
}
Tensor & Type::sinh_(Tensor & self) const {
    AT_ERROR("sinh_ is not implemented for type %s", toString());
}
Tensor & Type::sinh_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("sinh_out is not implemented for type %s", toString());
}
Tensor Type::sinh(const Tensor & self) const {
    AT_ERROR("sinh is not implemented for type %s", toString());
}
Tensor & Type::tan_(Tensor & self) const {
    AT_ERROR("tan_ is not implemented for type %s", toString());
}
Tensor & Type::tan_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("tan_out is not implemented for type %s", toString());
}
Tensor Type::tan(const Tensor & self) const {
    AT_ERROR("tan is not implemented for type %s", toString());
}
Tensor & Type::atan_(Tensor & self) const {
    AT_ERROR("atan_ is not implemented for type %s", toString());
}
Tensor & Type::atan_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("atan_out is not implemented for type %s", toString());
}
Tensor Type::atan(const Tensor & self) const {
    AT_ERROR("atan is not implemented for type %s", toString());
}
Tensor & Type::tanh_(Tensor & self) const {
    AT_ERROR("tanh_ is not implemented for type %s", toString());
}
Tensor & Type::tanh_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("tanh_out is not implemented for type %s", toString());
}
Tensor Type::tanh(const Tensor & self) const {
    AT_ERROR("tanh is not implemented for type %s", toString());
}
Tensor & Type::erf_(Tensor & self) const {
    AT_ERROR("erf_ is not implemented for type %s", toString());
}
Tensor & Type::erf_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("erf_out is not implemented for type %s", toString());
}
Tensor Type::erf(const Tensor & self) const {
    AT_ERROR("erf is not implemented for type %s", toString());
}
Tensor & Type::erfinv_(Tensor & self) const {
    AT_ERROR("erfinv_ is not implemented for type %s", toString());
}
Tensor & Type::erfinv_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("erfinv_out is not implemented for type %s", toString());
}
Tensor Type::erfinv(const Tensor & self) const {
    AT_ERROR("erfinv is not implemented for type %s", toString());
}
Tensor & Type::_sqrt_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("_sqrt_out is not implemented for type %s", toString());
}
Tensor Type::_sqrt(const Tensor & self) const {
    AT_ERROR("_sqrt is not implemented for type %s", toString());
}
Tensor & Type::rsqrt_(Tensor & self) const {
    AT_ERROR("rsqrt_ is not implemented for type %s", toString());
}
Tensor & Type::rsqrt_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("rsqrt_out is not implemented for type %s", toString());
}
Tensor Type::rsqrt(const Tensor & self) const {
    AT_ERROR("rsqrt is not implemented for type %s", toString());
}
Tensor & Type::_ceil_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("_ceil_out is not implemented for type %s", toString());
}
Tensor Type::_ceil(const Tensor & self) const {
    AT_ERROR("_ceil is not implemented for type %s", toString());
}
Tensor & Type::_floor_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("_floor_out is not implemented for type %s", toString());
}
Tensor Type::_floor(const Tensor & self) const {
    AT_ERROR("_floor is not implemented for type %s", toString());
}
Tensor & Type::_round_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("_round_out is not implemented for type %s", toString());
}
Tensor Type::_round(const Tensor & self) const {
    AT_ERROR("_round is not implemented for type %s", toString());
}
Tensor & Type::_trunc_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("_trunc_out is not implemented for type %s", toString());
}
Tensor Type::_trunc(const Tensor & self) const {
    AT_ERROR("_trunc is not implemented for type %s", toString());
}
Tensor & Type::frac_(Tensor & self) const {
    AT_ERROR("frac_ is not implemented for type %s", toString());
}
Tensor & Type::frac_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("frac_out is not implemented for type %s", toString());
}
Tensor Type::frac(const Tensor & self) const {
    AT_ERROR("frac is not implemented for type %s", toString());
}
Tensor & Type::mean_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    AT_ERROR("mean_out is not implemented for type %s", toString());
}
Tensor Type::mean(const Tensor & self, int64_t dim, bool keepdim) const {
    AT_ERROR("mean is not implemented for type %s", toString());
}
Tensor Type::mean(const Tensor & self) const {
    AT_ERROR("mean is not implemented for type %s", toString());
}
Tensor & Type::var_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    AT_ERROR("var_out is not implemented for type %s", toString());
}
Tensor Type::var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    AT_ERROR("var is not implemented for type %s", toString());
}
Tensor Type::var(const Tensor & self, bool unbiased) const {
    AT_ERROR("var is not implemented for type %s", toString());
}
Tensor & Type::std_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    AT_ERROR("std_out is not implemented for type %s", toString());
}
Tensor Type::std(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    AT_ERROR("std is not implemented for type %s", toString());
}
Tensor Type::std(const Tensor & self, bool unbiased) const {
    AT_ERROR("std is not implemented for type %s", toString());
}
Tensor & Type::norm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, bool keepdim) const {
    AT_ERROR("norm_out is not implemented for type %s", toString());
}
Tensor Type::norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim) const {
    AT_ERROR("norm is not implemented for type %s", toString());
}
Tensor Type::norm(const Tensor & self, Scalar p) const {
    AT_ERROR("norm is not implemented for type %s", toString());
}
Tensor & Type::renorm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    AT_ERROR("renorm_out is not implemented for type %s", toString());
}
Tensor Type::renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    AT_ERROR("renorm is not implemented for type %s", toString());
}
Tensor & Type::renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    AT_ERROR("renorm_ is not implemented for type %s", toString());
}
Tensor Type::s_dist(const Tensor & self, const Tensor & other, Scalar p) const {
    AT_ERROR("s_dist is not implemented for type %s", toString());
}
Tensor Type::dist(const Tensor & self, const Tensor & other, Scalar p) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "dist");
    return s_dist(b_self, b_other, p);
}
Tensor & Type::reciprocal_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("reciprocal_out is not implemented for type %s", toString());
}
Tensor Type::reciprocal(const Tensor & self) const {
    AT_ERROR("reciprocal is not implemented for type %s", toString());
}
Tensor & Type::reciprocal_(Tensor & self) const {
    AT_ERROR("reciprocal_ is not implemented for type %s", toString());
}
Tensor & Type::neg_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("neg_out is not implemented for type %s", toString());
}
Tensor Type::neg(const Tensor & self) const {
    AT_ERROR("neg is not implemented for type %s", toString());
}
Tensor & Type::neg_(Tensor & self) const {
    AT_ERROR("neg_ is not implemented for type %s", toString());
}
Tensor & Type::s_atan2_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_atan2_out is not implemented for type %s", toString());
}
Tensor & Type::atan2_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "atan2_out");
    return s_atan2_out(result, b_self, b_other);
}
Tensor Type::s_atan2(const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_atan2 is not implemented for type %s", toString());
}
Tensor Type::atan2(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "atan2");
    return s_atan2(b_self, b_other);
}
Tensor & Type::s_atan2_(Tensor & self, const Tensor & other) const {
    AT_ERROR("s_atan2_ is not implemented for type %s", toString());
}
Tensor & Type::atan2_(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "atan2_");
    return s_atan2_(self, b_other);
}
Tensor & Type::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    AT_ERROR("pow_out is not implemented for type %s", toString());
}
Tensor Type::pow(const Tensor & self, Scalar exponent) const {
    AT_ERROR("pow is not implemented for type %s", toString());
}
Tensor & Type::s_pow_out(Tensor & result, const Tensor & self, const Tensor & exponent) const {
    AT_ERROR("s_pow_out is not implemented for type %s", toString());
}
Tensor & Type::pow_out(Tensor & result, const Tensor & self, const Tensor & exponent) const {
    Tensor b_self, b_exponent;
    std::tie(b_self, b_exponent) = expand_outplace(self, exponent, "pow_out");
    return s_pow_out(result, b_self, b_exponent);
}
Tensor Type::s_pow(const Tensor & self, const Tensor & exponent) const {
    AT_ERROR("s_pow is not implemented for type %s", toString());
}
Tensor Type::pow(const Tensor & self, const Tensor & exponent) const {
    Tensor b_self, b_exponent;
    std::tie(b_self, b_exponent) = expand_outplace(self, exponent, "pow");
    return s_pow(b_self, b_exponent);
}
Tensor & Type::pow_out(Tensor & result, Scalar base, const Tensor & self) const {
    AT_ERROR("pow_out is not implemented for type %s", toString());
}
Tensor Type::pow(Scalar base, const Tensor & self) const {
    AT_ERROR("pow is not implemented for type %s", toString());
}
Tensor & Type::pow_(Tensor & self, Scalar exponent) const {
    AT_ERROR("pow_ is not implemented for type %s", toString());
}
Tensor & Type::s_pow_(Tensor & self, const Tensor & exponent) const {
    AT_ERROR("s_pow_ is not implemented for type %s", toString());
}
Tensor & Type::pow_(Tensor & self, const Tensor & exponent) const {
    Tensor b_exponent;
    std::tie(b_exponent) = expand_inplace(self, exponent, "pow_");
    return s_pow_(self, b_exponent);
}
Tensor & Type::s_lerp_out(Tensor & result, const Tensor & self, const Tensor & end, Scalar weight) const {
    AT_ERROR("s_lerp_out is not implemented for type %s", toString());
}
Tensor & Type::lerp_out(Tensor & result, const Tensor & self, const Tensor & end, Scalar weight) const {
    Tensor b_self, b_end;
    std::tie(b_self, b_end) = expand_outplace(self, end, "lerp_out");
    return s_lerp_out(result, b_self, b_end, weight);
}
Tensor Type::s_lerp(const Tensor & self, const Tensor & end, Scalar weight) const {
    AT_ERROR("s_lerp is not implemented for type %s", toString());
}
Tensor Type::lerp(const Tensor & self, const Tensor & end, Scalar weight) const {
    Tensor b_self, b_end;
    std::tie(b_self, b_end) = expand_outplace(self, end, "lerp");
    return s_lerp(b_self, b_end, weight);
}
Tensor & Type::s_lerp_(Tensor & self, const Tensor & end, Scalar weight) const {
    AT_ERROR("s_lerp_ is not implemented for type %s", toString());
}
Tensor & Type::lerp_(Tensor & self, const Tensor & end, Scalar weight) const {
    Tensor b_end;
    std::tie(b_end) = expand_inplace(self, end, "lerp_");
    return s_lerp_(self, b_end, weight);
}
Tensor & Type::_linspace_out(Tensor & result, Scalar start, Scalar end, int64_t steps) const {
    AT_ERROR("_linspace_out is not implemented for type %s", toString());
}
Tensor Type::_linspace(Scalar start, Scalar end, int64_t steps) const {
    AT_ERROR("_linspace is not implemented for type %s", toString());
}
Tensor & Type::_logspace_out(Tensor & result, Scalar start, Scalar end, int64_t steps) const {
    AT_ERROR("_logspace_out is not implemented for type %s", toString());
}
Tensor Type::_logspace(Scalar start, Scalar end, int64_t steps) const {
    AT_ERROR("_logspace is not implemented for type %s", toString());
}
Tensor & Type::histc_out(Tensor & result, const Tensor & self, int64_t bins, Scalar min, Scalar max) const {
    AT_ERROR("histc_out is not implemented for type %s", toString());
}
Tensor Type::histc(const Tensor & self, int64_t bins, Scalar min, Scalar max) const {
    AT_ERROR("histc is not implemented for type %s", toString());
}
Tensor & Type::zero_(Tensor & self) const {
    AT_ERROR("zero_ is not implemented for type %s", toString());
}
Tensor Type::_sumall(const Tensor & self) const {
    AT_ERROR("_sumall is not implemented for type %s", toString());
}
Tensor & Type::_sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    AT_ERROR("_sum_out is not implemented for type %s", toString());
}
Tensor Type::_sum(const Tensor & self, int64_t dim, bool keepdim) const {
    AT_ERROR("_sum is not implemented for type %s", toString());
}
Tensor Type::_prodall(const Tensor & self) const {
    AT_ERROR("_prodall is not implemented for type %s", toString());
}
Tensor & Type::_prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    AT_ERROR("_prod_out is not implemented for type %s", toString());
}
Tensor Type::_prod(const Tensor & self, int64_t dim, bool keepdim) const {
    AT_ERROR("_prod is not implemented for type %s", toString());
}
Tensor & Type::cumsum_out(Tensor & result, const Tensor & self, int64_t dim) const {
    AT_ERROR("cumsum_out is not implemented for type %s", toString());
}
Tensor Type::cumsum(const Tensor & self, int64_t dim) const {
    AT_ERROR("cumsum is not implemented for type %s", toString());
}
Tensor & Type::cumprod_out(Tensor & result, const Tensor & self, int64_t dim) const {
    AT_ERROR("cumprod_out is not implemented for type %s", toString());
}
Tensor Type::cumprod(const Tensor & self, int64_t dim) const {
    AT_ERROR("cumprod is not implemented for type %s", toString());
}
Tensor & Type::sign_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("sign_out is not implemented for type %s", toString());
}
Tensor Type::sign(const Tensor & self) const {
    AT_ERROR("sign is not implemented for type %s", toString());
}
Tensor & Type::sign_(Tensor & self) const {
    AT_ERROR("sign_ is not implemented for type %s", toString());
}
Tensor Type::trace(const Tensor & self) const {
    AT_ERROR("trace is not implemented for type %s", toString());
}
Tensor & Type::add_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    AT_ERROR("add_out is not implemented for type %s", toString());
}
Tensor Type::add(const Tensor & self, Scalar other, Scalar alpha) const {
    AT_ERROR("add is not implemented for type %s", toString());
}
Tensor & Type::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    AT_ERROR("s_add_out is not implemented for type %s", toString());
}
Tensor & Type::add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "add_out");
    return s_add_out(result, b_self, b_other, alpha);
}
Tensor Type::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    AT_ERROR("s_add is not implemented for type %s", toString());
}
Tensor Type::add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "add");
    return s_add(b_self, b_other, alpha);
}
Tensor & Type::add_out(Tensor & result, const Tensor & self, SparseTensor other, Scalar alpha) const {
    AT_ERROR("add_out is not implemented for type %s", toString());
}
Tensor Type::add(const Tensor & self, SparseTensor other, Scalar alpha) const {
    AT_ERROR("add is not implemented for type %s", toString());
}
Tensor & Type::add_(Tensor & self, Scalar other, Scalar alpha) const {
    AT_ERROR("add_ is not implemented for type %s", toString());
}
Tensor & Type::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    AT_ERROR("s_add_ is not implemented for type %s", toString());
}
Tensor & Type::add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "add_");
    return s_add_(self, b_other, alpha);
}
Tensor & Type::add_(Tensor & self, SparseTensor other, Scalar alpha) const {
    AT_ERROR("add_ is not implemented for type %s", toString());
}
Tensor & Type::sub_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    AT_ERROR("sub_out is not implemented for type %s", toString());
}
Tensor Type::sub(const Tensor & self, Scalar other, Scalar alpha) const {
    AT_ERROR("sub is not implemented for type %s", toString());
}
Tensor & Type::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    AT_ERROR("s_sub_out is not implemented for type %s", toString());
}
Tensor & Type::sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "sub_out");
    return s_sub_out(result, b_self, b_other, alpha);
}
Tensor Type::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    AT_ERROR("s_sub is not implemented for type %s", toString());
}
Tensor Type::sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "sub");
    return s_sub(b_self, b_other, alpha);
}
Tensor & Type::sub_(Tensor & self, Scalar other, Scalar alpha) const {
    AT_ERROR("sub_ is not implemented for type %s", toString());
}
Tensor & Type::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    AT_ERROR("s_sub_ is not implemented for type %s", toString());
}
Tensor & Type::sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "sub_");
    return s_sub_(self, b_other, alpha);
}
Tensor & Type::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    AT_ERROR("mul_out is not implemented for type %s", toString());
}
Tensor Type::mul(const Tensor & self, Scalar other) const {
    AT_ERROR("mul is not implemented for type %s", toString());
}
Tensor & Type::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_mul_out is not implemented for type %s", toString());
}
Tensor & Type::mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "mul_out");
    return s_mul_out(result, b_self, b_other);
}
Tensor Type::s_mul(const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_mul is not implemented for type %s", toString());
}
Tensor Type::mul(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "mul");
    return s_mul(b_self, b_other);
}
Tensor & Type::mul_(Tensor & self, Scalar other) const {
    AT_ERROR("mul_ is not implemented for type %s", toString());
}
Tensor & Type::s_mul_(Tensor & self, const Tensor & other) const {
    AT_ERROR("s_mul_ is not implemented for type %s", toString());
}
Tensor & Type::mul_(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "mul_");
    return s_mul_(self, b_other);
}
Tensor & Type::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    AT_ERROR("div_out is not implemented for type %s", toString());
}
Tensor Type::div(const Tensor & self, Scalar other) const {
    AT_ERROR("div is not implemented for type %s", toString());
}
Tensor & Type::s_div_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_div_out is not implemented for type %s", toString());
}
Tensor & Type::div_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "div_out");
    return s_div_out(result, b_self, b_other);
}
Tensor Type::s_div(const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_div is not implemented for type %s", toString());
}
Tensor Type::div(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "div");
    return s_div(b_self, b_other);
}
Tensor & Type::div_(Tensor & self, Scalar other) const {
    AT_ERROR("div_ is not implemented for type %s", toString());
}
Tensor & Type::s_div_(Tensor & self, const Tensor & other) const {
    AT_ERROR("s_div_ is not implemented for type %s", toString());
}
Tensor & Type::div_(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "div_");
    return s_div_(self, b_other);
}
Tensor & Type::fmod_out(Tensor & result, const Tensor & self, Scalar other) const {
    AT_ERROR("fmod_out is not implemented for type %s", toString());
}
Tensor Type::fmod(const Tensor & self, Scalar other) const {
    AT_ERROR("fmod is not implemented for type %s", toString());
}
Tensor & Type::s_fmod_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_fmod_out is not implemented for type %s", toString());
}
Tensor & Type::fmod_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "fmod_out");
    return s_fmod_out(result, b_self, b_other);
}
Tensor Type::s_fmod(const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_fmod is not implemented for type %s", toString());
}
Tensor Type::fmod(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "fmod");
    return s_fmod(b_self, b_other);
}
Tensor & Type::fmod_(Tensor & self, Scalar other) const {
    AT_ERROR("fmod_ is not implemented for type %s", toString());
}
Tensor & Type::s_fmod_(Tensor & self, const Tensor & other) const {
    AT_ERROR("s_fmod_ is not implemented for type %s", toString());
}
Tensor & Type::fmod_(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "fmod_");
    return s_fmod_(self, b_other);
}
Tensor & Type::remainder_out(Tensor & result, const Tensor & self, Scalar other) const {
    AT_ERROR("remainder_out is not implemented for type %s", toString());
}
Tensor Type::remainder(const Tensor & self, Scalar other) const {
    AT_ERROR("remainder is not implemented for type %s", toString());
}
Tensor & Type::s_remainder_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_remainder_out is not implemented for type %s", toString());
}
Tensor & Type::remainder_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "remainder_out");
    return s_remainder_out(result, b_self, b_other);
}
Tensor Type::s_remainder(const Tensor & self, const Tensor & other) const {
    AT_ERROR("s_remainder is not implemented for type %s", toString());
}
Tensor Type::remainder(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "remainder");
    return s_remainder(b_self, b_other);
}
Tensor & Type::remainder_(Tensor & self, Scalar other) const {
    AT_ERROR("remainder_ is not implemented for type %s", toString());
}
Tensor & Type::s_remainder_(Tensor & self, const Tensor & other) const {
    AT_ERROR("s_remainder_ is not implemented for type %s", toString());
}
Tensor & Type::remainder_(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "remainder_");
    return s_remainder_(self, b_other);
}
Tensor & Type::clamp_out(Tensor & result, const Tensor & self, Scalar min, Scalar max) const {
    AT_ERROR("clamp_out is not implemented for type %s", toString());
}
Tensor Type::clamp(const Tensor & self, Scalar min, Scalar max) const {
    AT_ERROR("clamp is not implemented for type %s", toString());
}
Tensor & Type::clamp_(Tensor & self, Scalar min, Scalar max) const {
    AT_ERROR("clamp_ is not implemented for type %s", toString());
}
Tensor & Type::clamp_min_out(Tensor & result, const Tensor & self, Scalar min) const {
    AT_ERROR("clamp_min_out is not implemented for type %s", toString());
}
Tensor Type::clamp_min(const Tensor & self, Scalar min) const {
    AT_ERROR("clamp_min is not implemented for type %s", toString());
}
Tensor & Type::clamp_min_(Tensor & self, Scalar min) const {
    AT_ERROR("clamp_min_ is not implemented for type %s", toString());
}
Tensor & Type::clamp_max_out(Tensor & result, const Tensor & self, Scalar max) const {
    AT_ERROR("clamp_max_out is not implemented for type %s", toString());
}
Tensor Type::clamp_max(const Tensor & self, Scalar max) const {
    AT_ERROR("clamp_max is not implemented for type %s", toString());
}
Tensor & Type::clamp_max_(Tensor & self, Scalar max) const {
    AT_ERROR("clamp_max_ is not implemented for type %s", toString());
}
Tensor Type::_dot(const Tensor & self, const Tensor & tensor) const {
    AT_ERROR("_dot is not implemented for type %s", toString());
}
Tensor & Type::tril_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    AT_ERROR("tril_out is not implemented for type %s", toString());
}
Tensor Type::tril(const Tensor & self, int64_t diagonal) const {
    AT_ERROR("tril is not implemented for type %s", toString());
}
Tensor & Type::tril_(Tensor & self, int64_t diagonal) const {
    AT_ERROR("tril_ is not implemented for type %s", toString());
}
Tensor & Type::triu_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    AT_ERROR("triu_out is not implemented for type %s", toString());
}
Tensor Type::triu(const Tensor & self, int64_t diagonal) const {
    AT_ERROR("triu is not implemented for type %s", toString());
}
Tensor & Type::triu_(Tensor & self, int64_t diagonal) const {
    AT_ERROR("triu_ is not implemented for type %s", toString());
}
Tensor & Type::cross_out(Tensor & result, const Tensor & self, const Tensor & other, int64_t dim) const {
    AT_ERROR("cross_out is not implemented for type %s", toString());
}
Tensor Type::cross(const Tensor & self, const Tensor & other, int64_t dim) const {
    AT_ERROR("cross is not implemented for type %s", toString());
}
Tensor & Type::diag_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    AT_ERROR("diag_out is not implemented for type %s", toString());
}
Tensor Type::diag(const Tensor & self, int64_t diagonal) const {
    AT_ERROR("diag is not implemented for type %s", toString());
}
Tensor & Type::s_addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    AT_ERROR("s_addmm_out is not implemented for type %s", toString());
}
Tensor & Type::addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    Tensor b_self;
    std::tie(b_self) = expand_size(self, {mat1.size(0),mat2.size(1)}, "addmm_out");
    return s_addmm_out(result, b_self, mat1, mat2, beta, alpha);
}
Tensor Type::s_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    AT_ERROR("s_addmm is not implemented for type %s", toString());
}
Tensor Type::addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    Tensor b_self;
    std::tie(b_self) = expand_size(self, {mat1.size(0),mat2.size(1)}, "addmm");
    return s_addmm(b_self, mat1, mat2, beta, alpha);
}
Tensor & Type::addmm_out(Tensor & result, const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    AT_ERROR("addmm_out is not implemented for type %s", toString());
}
Tensor Type::addmm(const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    AT_ERROR("addmm is not implemented for type %s", toString());
}
Tensor & Type::addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    AT_ERROR("addmm_ is not implemented for type %s", toString());
}
Tensor & Type::addmm_(Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    AT_ERROR("addmm_ is not implemented for type %s", toString());
}
Tensor & Type::s__addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    AT_ERROR("s__addmv_out is not implemented for type %s", toString());
}
Tensor & Type::_addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    Tensor b_self;
    std::tie(b_self) = expand_size(self, {mat.size(0)}, "_addmv_out");
    return s__addmv_out(result, b_self, mat, vec, beta, alpha);
}
Tensor Type::s__addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    AT_ERROR("s__addmv is not implemented for type %s", toString());
}
Tensor Type::_addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    Tensor b_self;
    std::tie(b_self) = expand_size(self, {mat.size(0)}, "_addmv");
    return s__addmv(b_self, mat, vec, beta, alpha);
}
Tensor & Type::_addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    AT_ERROR("_addmv_ is not implemented for type %s", toString());
}
Tensor & Type::s__addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    AT_ERROR("s__addr_out is not implemented for type %s", toString());
}
Tensor & Type::_addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    Tensor b_self;
    std::tie(b_self) = expand_size(self, {vec1.size(0),vec2.size(0)}, "_addr_out");
    return s__addr_out(result, b_self, vec1, vec2, beta, alpha);
}
Tensor Type::s__addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    AT_ERROR("s__addr is not implemented for type %s", toString());
}
Tensor Type::_addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    Tensor b_self;
    std::tie(b_self) = expand_size(self, {vec1.size(0),vec2.size(0)}, "_addr");
    return s__addr(b_self, vec1, vec2, beta, alpha);
}
Tensor & Type::_addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    AT_ERROR("_addr_ is not implemented for type %s", toString());
}
Tensor & Type::_ger_out(Tensor & result, const Tensor & self, const Tensor & vec2) const {
    AT_ERROR("_ger_out is not implemented for type %s", toString());
}
Tensor Type::_ger(const Tensor & self, const Tensor & vec2) const {
    AT_ERROR("_ger is not implemented for type %s", toString());
}
Tensor & Type::_mv_out(Tensor & result, const Tensor & self, const Tensor & vec) const {
    AT_ERROR("_mv_out is not implemented for type %s", toString());
}
Tensor Type::_mv(const Tensor & self, const Tensor & vec) const {
    AT_ERROR("_mv is not implemented for type %s", toString());
}
Tensor & Type::_mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    AT_ERROR("_mm_out is not implemented for type %s", toString());
}
Tensor Type::_mm(const Tensor & self, const Tensor & mat2) const {
    AT_ERROR("_mm is not implemented for type %s", toString());
}
Tensor & Type::bmm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    AT_ERROR("bmm_out is not implemented for type %s", toString());
}
Tensor Type::bmm(const Tensor & self, const Tensor & mat2) const {
    AT_ERROR("bmm is not implemented for type %s", toString());
}
Tensor & Type::s_addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    AT_ERROR("s_addbmm_out is not implemented for type %s", toString());
}
Tensor & Type::addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    Tensor b_self;
    std::tie(b_self) = expand_size(self, {batch1.size(1),batch2.size(2)}, "addbmm_out");
    return s_addbmm_out(result, b_self, batch1, batch2, beta, alpha);
}
Tensor Type::s_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    AT_ERROR("s_addbmm is not implemented for type %s", toString());
}
Tensor Type::addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    Tensor b_self;
    std::tie(b_self) = expand_size(self, {batch1.size(1),batch2.size(2)}, "addbmm");
    return s_addbmm(b_self, batch1, batch2, beta, alpha);
}
Tensor & Type::addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    AT_ERROR("addbmm_ is not implemented for type %s", toString());
}
Tensor & Type::s_baddbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    AT_ERROR("s_baddbmm_out is not implemented for type %s", toString());
}
Tensor & Type::baddbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    Tensor b_self;
    std::tie(b_self) = expand_size(self, {batch1.size(0),batch1.size(1),batch2.size(2)}, "baddbmm_out");
    return s_baddbmm_out(result, b_self, batch1, batch2, beta, alpha);
}
Tensor Type::s_baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    AT_ERROR("s_baddbmm is not implemented for type %s", toString());
}
Tensor Type::baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    Tensor b_self;
    std::tie(b_self) = expand_size(self, {batch1.size(0),batch1.size(1),batch2.size(2)}, "baddbmm");
    return s_baddbmm(b_self, batch1, batch2, beta, alpha);
}
Tensor & Type::baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    AT_ERROR("baddbmm_ is not implemented for type %s", toString());
}
Tensor & Type::s_addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    AT_ERROR("s_addcmul_out is not implemented for type %s", toString());
}
Tensor & Type::addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    Tensor b_self, b_tensor1, b_tensor2;
    std::tie(b_self, b_tensor1, b_tensor2) = expand_outplace(self, tensor1, tensor2, "addcmul_out");
    return s_addcmul_out(result, b_self, b_tensor1, b_tensor2, value);
}
Tensor Type::s_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    AT_ERROR("s_addcmul is not implemented for type %s", toString());
}
Tensor Type::addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    Tensor b_self, b_tensor1, b_tensor2;
    std::tie(b_self, b_tensor1, b_tensor2) = expand_outplace(self, tensor1, tensor2, "addcmul");
    return s_addcmul(b_self, b_tensor1, b_tensor2, value);
}
Tensor & Type::s_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    AT_ERROR("s_addcmul_ is not implemented for type %s", toString());
}
Tensor & Type::addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    Tensor b_tensor1, b_tensor2;
    std::tie(b_tensor1, b_tensor2) = expand_inplace(self, tensor1, tensor2, "addcmul_");
    return s_addcmul_(self, b_tensor1, b_tensor2, value);
}
Tensor & Type::s_addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    AT_ERROR("s_addcdiv_out is not implemented for type %s", toString());
}
Tensor & Type::addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    Tensor b_self, b_tensor1, b_tensor2;
    std::tie(b_self, b_tensor1, b_tensor2) = expand_outplace(self, tensor1, tensor2, "addcdiv_out");
    return s_addcdiv_out(result, b_self, b_tensor1, b_tensor2, value);
}
Tensor Type::s_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    AT_ERROR("s_addcdiv is not implemented for type %s", toString());
}
Tensor Type::addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    Tensor b_self, b_tensor1, b_tensor2;
    std::tie(b_self, b_tensor1, b_tensor2) = expand_outplace(self, tensor1, tensor2, "addcdiv");
    return s_addcdiv(b_self, b_tensor1, b_tensor2, value);
}
Tensor & Type::s_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    AT_ERROR("s_addcdiv_ is not implemented for type %s", toString());
}
Tensor & Type::addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    Tensor b_tensor1, b_tensor2;
    std::tie(b_tensor1, b_tensor2) = expand_inplace(self, tensor1, tensor2, "addcdiv_");
    return s_addcdiv_(self, b_tensor1, b_tensor2, value);
}
std::tuple<Tensor &,Tensor &> Type::gesv_out(Tensor & solution, Tensor & lu, const Tensor & self, const Tensor & A) const {
    AT_ERROR("gesv_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::gesv(const Tensor & self, const Tensor & A) const {
    AT_ERROR("gesv is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::gels_out(Tensor & res1, Tensor & res2, const Tensor & self, const Tensor & A) const {
    AT_ERROR("gels_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::gels(const Tensor & self, const Tensor & A) const {
    AT_ERROR("gels is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::trtrs_out(Tensor & res1, Tensor & res2, const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
    AT_ERROR("trtrs_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
    AT_ERROR("trtrs is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::symeig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors, bool upper) const {
    AT_ERROR("symeig_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::symeig(const Tensor & self, bool eigenvectors, bool upper) const {
    AT_ERROR("symeig is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::eig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors) const {
    AT_ERROR("eig_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::eig(const Tensor & self, bool eigenvectors) const {
    AT_ERROR("eig is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &,Tensor &> Type::svd_out(Tensor & res1, Tensor & res2, Tensor & res3, const Tensor & self, bool some) const {
    AT_ERROR("svd_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor,Tensor> Type::svd(const Tensor & self, bool some) const {
    AT_ERROR("svd is not implemented for type %s", toString());
}
Tensor & Type::inverse_out(Tensor & output, const Tensor & self) const {
    AT_ERROR("inverse_out is not implemented for type %s", toString());
}
Tensor Type::inverse(const Tensor & self) const {
    AT_ERROR("inverse is not implemented for type %s", toString());
}
Tensor & Type::potrf_out(Tensor & output, const Tensor & self, bool upper) const {
    AT_ERROR("potrf_out is not implemented for type %s", toString());
}
Tensor Type::potrf(const Tensor & self, bool upper) const {
    AT_ERROR("potrf is not implemented for type %s", toString());
}
Tensor & Type::potrs_out(Tensor & result, const Tensor & self, const Tensor & input2, bool upper) const {
    AT_ERROR("potrs_out is not implemented for type %s", toString());
}
Tensor Type::potrs(const Tensor & self, const Tensor & input2, bool upper) const {
    AT_ERROR("potrs is not implemented for type %s", toString());
}
Tensor & Type::potri_out(Tensor & output, const Tensor & self, bool upper) const {
    AT_ERROR("potri_out is not implemented for type %s", toString());
}
Tensor Type::potri(const Tensor & self, bool upper) const {
    AT_ERROR("potri is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::pstrf_out(Tensor & res1, Tensor & res2, const Tensor & self, bool upper, Scalar tol) const {
    AT_ERROR("pstrf_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::pstrf(const Tensor & self, bool upper, Scalar tol) const {
    AT_ERROR("pstrf is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::qr_out(Tensor & res1, Tensor & res2, const Tensor & self) const {
    AT_ERROR("qr_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::qr(const Tensor & self) const {
    AT_ERROR("qr is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::geqrf_out(Tensor & res1, Tensor & res2, const Tensor & self) const {
    AT_ERROR("geqrf_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::geqrf(const Tensor & self) const {
    AT_ERROR("geqrf is not implemented for type %s", toString());
}
Tensor & Type::orgqr_out(Tensor & result, const Tensor & self, const Tensor & input2) const {
    AT_ERROR("orgqr_out is not implemented for type %s", toString());
}
Tensor Type::orgqr(const Tensor & self, const Tensor & input2) const {
    AT_ERROR("orgqr is not implemented for type %s", toString());
}
Tensor & Type::ormqr_out(Tensor & result, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
    AT_ERROR("ormqr_out is not implemented for type %s", toString());
}
Tensor Type::ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
    AT_ERROR("ormqr is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::btrifact_out(Tensor & result, Tensor & pivots, const Tensor & self, bool pivot) const {
    AT_ERROR("btrifact_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::btrifact(const Tensor & self, bool pivot) const {
    AT_ERROR("btrifact is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &,Tensor &> Type::btrifact_with_info_out(Tensor & result, Tensor & pivots, Tensor & info, const Tensor & self, bool pivot) const {
    AT_ERROR("btrifact_with_info_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor,Tensor> Type::btrifact_with_info(const Tensor & self, bool pivot) const {
    AT_ERROR("btrifact_with_info is not implemented for type %s", toString());
}
Tensor & Type::btrisolve_out(Tensor & result, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const {
    AT_ERROR("btrisolve_out is not implemented for type %s", toString());
}
Tensor Type::btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const {
    AT_ERROR("btrisolve is not implemented for type %s", toString());
}
Tensor & Type::random_(Tensor & self, int64_t from, int64_t to, Generator * generator) const {
    AT_ERROR("random_ is not implemented for type %s", toString());
}
Tensor & Type::random_(Tensor & self, int64_t to, Generator * generator) const {
    AT_ERROR("random_ is not implemented for type %s", toString());
}
Tensor & Type::random_(Tensor & self, Generator * generator) const {
    AT_ERROR("random_ is not implemented for type %s", toString());
}
Tensor & Type::multinomial_out(Tensor & result, const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const {
    AT_ERROR("multinomial_out is not implemented for type %s", toString());
}
Tensor Type::multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const {
    AT_ERROR("multinomial is not implemented for type %s", toString());
}
Tensor & Type::uniform_(Tensor & self, double from, double to, Generator * generator) const {
    AT_ERROR("uniform_ is not implemented for type %s", toString());
}
Tensor & Type::normal_out(Tensor & output, const Tensor & mean, double std, Generator * generator) const {
    AT_ERROR("normal_out is not implemented for type %s", toString());
}
Tensor Type::normal(const Tensor & mean, double std, Generator * generator) const {
    AT_ERROR("normal is not implemented for type %s", toString());
}
Tensor & Type::normal_out(Tensor & output, double mean, const Tensor & std, Generator * generator) const {
    AT_ERROR("normal_out is not implemented for type %s", toString());
}
Tensor Type::normal(double mean, const Tensor & std, Generator * generator) const {
    AT_ERROR("normal is not implemented for type %s", toString());
}
Tensor & Type::normal_out(Tensor & output, const Tensor & mean, const Tensor & std, Generator * generator) const {
    AT_ERROR("normal_out is not implemented for type %s", toString());
}
Tensor Type::normal(const Tensor & mean, const Tensor & std, Generator * generator) const {
    AT_ERROR("normal is not implemented for type %s", toString());
}
Tensor & Type::normal_(Tensor & self, double mean, double std, Generator * generator) const {
    AT_ERROR("normal_ is not implemented for type %s", toString());
}
Tensor & Type::cauchy_(Tensor & self, double median, double sigma, Generator * generator) const {
    AT_ERROR("cauchy_ is not implemented for type %s", toString());
}
Tensor & Type::log_normal_(Tensor & self, double mean, double std, Generator * generator) const {
    AT_ERROR("log_normal_ is not implemented for type %s", toString());
}
Tensor & Type::exponential_(Tensor & self, double lambd, Generator * generator) const {
    AT_ERROR("exponential_ is not implemented for type %s", toString());
}
Tensor & Type::geometric_(Tensor & self, double p, Generator * generator) const {
    AT_ERROR("geometric_ is not implemented for type %s", toString());
}
Tensor & Type::bernoulli_out(Tensor & output, const Tensor & self, Generator * generator) const {
    AT_ERROR("bernoulli_out is not implemented for type %s", toString());
}
Tensor Type::bernoulli(const Tensor & self, Generator * generator) const {
    AT_ERROR("bernoulli is not implemented for type %s", toString());
}
Tensor & Type::_standard_gamma_out(Tensor & output, const Tensor & self, Generator * generator) const {
    AT_ERROR("_standard_gamma_out is not implemented for type %s", toString());
}
Tensor Type::_standard_gamma(const Tensor & self, Generator * generator) const {
    AT_ERROR("_standard_gamma is not implemented for type %s", toString());
}
Tensor & Type::_dirichlet_grad_out(Tensor & output, const Tensor & x, const Tensor & alpha, const Tensor & total) const {
    AT_ERROR("_dirichlet_grad_out is not implemented for type %s", toString());
}
Tensor Type::_dirichlet_grad(const Tensor & x, const Tensor & alpha, const Tensor & total) const {
    AT_ERROR("_dirichlet_grad is not implemented for type %s", toString());
}
Tensor Type::tensor(Storage & storage, int64_t storageOffset, IntList size, IntList stride) const {
    AT_ERROR("tensor is not implemented for type %s", toString());
}
Tensor Type::tensor(IntList size) const {
    AT_ERROR("tensor is not implemented for type %s", toString());
}
Tensor Type::tensor(IntList size, IntList stride) const {
    AT_ERROR("tensor is not implemented for type %s", toString());
}
Tensor Type::tensor() const {
    AT_ERROR("tensor is not implemented for type %s", toString());
}
Tensor Type::sparse_coo_tensor(const Tensor & indices, const Tensor & values, IntList size) const {
    AT_ERROR("sparse_coo_tensor is not implemented for type %s", toString());
}
Tensor Type::sparse_coo_tensor(const Tensor & indices, const Tensor & values) const {
    AT_ERROR("sparse_coo_tensor is not implemented for type %s", toString());
}
Tensor Type::alias(const Tensor & self) const {
    AT_ERROR("alias is not implemented for type %s", toString());
}
Tensor & Type::_copy_ignoring_overlaps_(Tensor & self, const Tensor & src) const {
    AT_ERROR("_copy_ignoring_overlaps_ is not implemented for type %s", toString());
}
Tensor & Type::as_strided_out(Tensor & result, const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    AT_ERROR("as_strided_out is not implemented for type %s", toString());
}
Tensor Type::as_strided(const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    AT_ERROR("as_strided is not implemented for type %s", toString());
}
Tensor & Type::as_strided_(Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    AT_ERROR("as_strided_ is not implemented for type %s", toString());
}
Tensor & Type::sparse_raw_resize_(Tensor & self, IntList size, int64_t nDimI, int64_t nDimV) const {
    AT_ERROR("sparse_raw_resize_ is not implemented for type %s", toString());
}
Tensor & Type::_cat_out(Tensor & self, TensorList tensors, int64_t dim) const {
    AT_ERROR("_cat_out is not implemented for type %s", toString());
}
Tensor Type::_cat(TensorList tensors, int64_t dim) const {
    AT_ERROR("_cat is not implemented for type %s", toString());
}
Tensor Type::_sparse_mask(const Tensor & self, SparseTensor mask) const {
    AT_ERROR("_sparse_mask is not implemented for type %s", toString());
}
Tensor Type::to_dense(const Tensor & self) const {
    AT_ERROR("to_dense is not implemented for type %s", toString());
}
int64_t Type::_dimI(const Tensor & self) const {
    AT_ERROR("_dimI is not implemented for type %s", toString());
}
int64_t Type::_dimV(const Tensor & self) const {
    AT_ERROR("_dimV is not implemented for type %s", toString());
}
int64_t Type::_nnz(const Tensor & self) const {
    AT_ERROR("_nnz is not implemented for type %s", toString());
}
Tensor Type::coalesce(const Tensor & self) const {
    AT_ERROR("coalesce is not implemented for type %s", toString());
}
bool Type::is_coalesced(const Tensor & self) const {
    AT_ERROR("is_coalesced is not implemented for type %s", toString());
}
Tensor Type::_indices(const Tensor & self) const {
    AT_ERROR("_indices is not implemented for type %s", toString());
}
Tensor Type::_values(const Tensor & self) const {
    AT_ERROR("_values is not implemented for type %s", toString());
}
Tensor & Type::hspmm_out(Tensor & result, const Tensor & mat1, const Tensor & mat2) const {
    AT_ERROR("hspmm_out is not implemented for type %s", toString());
}
Tensor Type::hspmm(const Tensor & mat1, const Tensor & mat2) const {
    AT_ERROR("hspmm is not implemented for type %s", toString());
}
Tensor & Type::binary_cross_entropy_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    return binary_cross_entropy_forward_out(output, self, target, weight, size_average, reduce);
}
Tensor Type::binary_cross_entropy(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    return binary_cross_entropy_forward(self, target, weight, size_average, reduce);
}
Tensor & Type::binary_cross_entropy_forward_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    AT_ERROR("binary_cross_entropy_forward_out is not implemented for type %s", toString());
}
Tensor Type::binary_cross_entropy_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    AT_ERROR("binary_cross_entropy_forward is not implemented for type %s", toString());
}
Tensor & Type::binary_cross_entropy_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    AT_ERROR("binary_cross_entropy_backward_out is not implemented for type %s", toString());
}
Tensor Type::binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    AT_ERROR("binary_cross_entropy_backward is not implemented for type %s", toString());
}
Tensor & Type::kl_div_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    return kl_div_forward_out(output, self, target, size_average, reduce);
}
Tensor Type::kl_div(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    return kl_div_forward(self, target, size_average, reduce);
}
Tensor & Type::kl_div_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    AT_ERROR("kl_div_forward_out is not implemented for type %s", toString());
}
Tensor Type::kl_div_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    AT_ERROR("kl_div_forward is not implemented for type %s", toString());
}
Tensor & Type::kl_div_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    AT_ERROR("kl_div_backward_out is not implemented for type %s", toString());
}
Tensor Type::kl_div_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    AT_ERROR("kl_div_backward is not implemented for type %s", toString());
}
Tensor & Type::l1_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    return l1_loss_forward_out(output, self, target, size_average, reduce);
}
Tensor Type::l1_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    return l1_loss_forward(self, target, size_average, reduce);
}
Tensor & Type::l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    AT_ERROR("l1_loss_forward_out is not implemented for type %s", toString());
}
Tensor Type::l1_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    AT_ERROR("l1_loss_forward is not implemented for type %s", toString());
}
Tensor & Type::l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    AT_ERROR("l1_loss_backward_out is not implemented for type %s", toString());
}
Tensor Type::l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    AT_ERROR("l1_loss_backward is not implemented for type %s", toString());
}
Tensor & Type::mse_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    return mse_loss_forward_out(output, self, target, size_average, reduce);
}
Tensor Type::mse_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    return mse_loss_forward(self, target, size_average, reduce);
}
Tensor & Type::mse_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    AT_ERROR("mse_loss_forward_out is not implemented for type %s", toString());
}
Tensor Type::mse_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    AT_ERROR("mse_loss_forward is not implemented for type %s", toString());
}
Tensor & Type::mse_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    AT_ERROR("mse_loss_backward_out is not implemented for type %s", toString());
}
Tensor Type::mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    AT_ERROR("mse_loss_backward is not implemented for type %s", toString());
}
Tensor & Type::multi_margin_loss_out(Tensor & output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    return multi_margin_loss_forward_out(output, self, target, p, margin, weight, size_average, reduce);
}
Tensor Type::multi_margin_loss(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    return multi_margin_loss_forward(self, target, p, margin, weight, size_average, reduce);
}
Tensor & Type::multi_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    AT_ERROR("multi_margin_loss_forward_out is not implemented for type %s", toString());
}
Tensor Type::multi_margin_loss_forward(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    AT_ERROR("multi_margin_loss_forward is not implemented for type %s", toString());
}
Tensor & Type::multi_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    AT_ERROR("multi_margin_loss_backward_out is not implemented for type %s", toString());
}
Tensor Type::multi_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    AT_ERROR("multi_margin_loss_backward is not implemented for type %s", toString());
}
Tensor & Type::multilabel_margin_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    Tensor is_target = tensor();
    return std::get<0>(multilabel_margin_loss_forward_out(output, is_target, self, target, size_average, reduce));
}
Tensor Type::multilabel_margin_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    return std::get<0>(multilabel_margin_loss_forward(self, target, size_average, reduce));
}
std::tuple<Tensor &,Tensor &> Type::multilabel_margin_loss_forward_out(Tensor & output, Tensor & is_target, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    AT_ERROR("multilabel_margin_loss_forward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::multilabel_margin_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    AT_ERROR("multilabel_margin_loss_forward is not implemented for type %s", toString());
}
Tensor & Type::multilabel_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce, const Tensor & is_target) const {
    AT_ERROR("multilabel_margin_loss_backward_out is not implemented for type %s", toString());
}
Tensor Type::multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce, const Tensor & is_target) const {
    AT_ERROR("multilabel_margin_loss_backward is not implemented for type %s", toString());
}
Tensor & Type::nll_loss_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    Tensor total_weight = tensor();
    return std::get<0>(nll_loss_forward_out(output, total_weight, self, target, weight, size_average, ignore_index, reduce));
}
Tensor Type::nll_loss(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    return std::get<0>(nll_loss_forward(self, target, weight, size_average, ignore_index, reduce));
}
std::tuple<Tensor &,Tensor &> Type::nll_loss_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    AT_ERROR("nll_loss_forward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::nll_loss_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    AT_ERROR("nll_loss_forward is not implemented for type %s", toString());
}
Tensor & Type::nll_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    AT_ERROR("nll_loss_backward_out is not implemented for type %s", toString());
}
Tensor Type::nll_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    AT_ERROR("nll_loss_backward is not implemented for type %s", toString());
}
Tensor & Type::nll_loss2d_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    Tensor total_weight = tensor();
    return std::get<0>(nll_loss2d_forward_out(output, total_weight, self, target, weight, size_average, ignore_index, reduce));
}
Tensor Type::nll_loss2d(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    return std::get<0>(nll_loss2d_forward(self, target, weight, size_average, ignore_index, reduce));
}
std::tuple<Tensor &,Tensor &> Type::nll_loss2d_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    AT_ERROR("nll_loss2d_forward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::nll_loss2d_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    AT_ERROR("nll_loss2d_forward is not implemented for type %s", toString());
}
Tensor & Type::nll_loss2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    AT_ERROR("nll_loss2d_backward_out is not implemented for type %s", toString());
}
Tensor Type::nll_loss2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    AT_ERROR("nll_loss2d_backward is not implemented for type %s", toString());
}
Tensor & Type::smooth_l1_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    return smooth_l1_loss_forward_out(output, self, target, size_average, reduce);
}
Tensor Type::smooth_l1_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    return smooth_l1_loss_forward(self, target, size_average, reduce);
}
Tensor & Type::smooth_l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    AT_ERROR("smooth_l1_loss_forward_out is not implemented for type %s", toString());
}
Tensor Type::smooth_l1_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    AT_ERROR("smooth_l1_loss_forward is not implemented for type %s", toString());
}
Tensor & Type::smooth_l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    AT_ERROR("smooth_l1_loss_backward_out is not implemented for type %s", toString());
}
Tensor Type::smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    AT_ERROR("smooth_l1_loss_backward is not implemented for type %s", toString());
}
Tensor & Type::soft_margin_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    return soft_margin_loss_forward_out(output, self, target, size_average, reduce);
}
Tensor Type::soft_margin_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    return soft_margin_loss_forward(self, target, size_average, reduce);
}
Tensor & Type::soft_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    AT_ERROR("soft_margin_loss_forward_out is not implemented for type %s", toString());
}
Tensor Type::soft_margin_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    AT_ERROR("soft_margin_loss_forward is not implemented for type %s", toString());
}
Tensor & Type::soft_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    AT_ERROR("soft_margin_loss_backward_out is not implemented for type %s", toString());
}
Tensor Type::soft_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    AT_ERROR("soft_margin_loss_backward is not implemented for type %s", toString());
}
Tensor & Type::elu_out(Tensor & output, const Tensor & self, Scalar alpha, Scalar scale) const {
    return elu_forward_out(output, self, alpha, scale);
}
Tensor Type::elu(const Tensor & self, Scalar alpha, Scalar scale) const {
    return elu_forward(self, alpha, scale);
}
Tensor & Type::elu_forward_out(Tensor & output, const Tensor & self, Scalar alpha, Scalar scale) const {
    AT_ERROR("elu_forward_out is not implemented for type %s", toString());
}
Tensor Type::elu_forward(const Tensor & self, Scalar alpha, Scalar scale) const {
    AT_ERROR("elu_forward is not implemented for type %s", toString());
}
Tensor & Type::elu_backward_out(Tensor & grad_input, const Tensor & grad_output, Scalar alpha, Scalar scale, const Tensor & output) const {
    AT_ERROR("elu_backward_out is not implemented for type %s", toString());
}
Tensor Type::elu_backward(const Tensor & grad_output, Scalar alpha, Scalar scale, const Tensor & output) const {
    AT_ERROR("elu_backward is not implemented for type %s", toString());
}
Tensor & Type::elu_(Tensor & self, Scalar alpha, Scalar scale) const {
    return elu_forward_(self, alpha, scale);
}
Tensor & Type::elu_forward_(Tensor & self, Scalar alpha, Scalar scale) const {
    AT_ERROR("elu_forward_ is not implemented for type %s", toString());
}
Tensor & Type::glu_out(Tensor & output, const Tensor & self, int64_t dim) const {
    return glu_forward_out(output, self, dim);
}
Tensor Type::glu(const Tensor & self, int64_t dim) const {
    return glu_forward(self, dim);
}
Tensor & Type::glu_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
    AT_ERROR("glu_forward_out is not implemented for type %s", toString());
}
Tensor Type::glu_forward(const Tensor & self, int64_t dim) const {
    AT_ERROR("glu_forward is not implemented for type %s", toString());
}
Tensor & Type::glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim) const {
    AT_ERROR("glu_backward_out is not implemented for type %s", toString());
}
Tensor Type::glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim) const {
    AT_ERROR("glu_backward is not implemented for type %s", toString());
}
Tensor & Type::hardshrink_out(Tensor & output, const Tensor & self, Scalar lambd) const {
    return hardshrink_forward_out(output, self, lambd);
}
Tensor Type::hardshrink(const Tensor & self, Scalar lambd) const {
    return hardshrink_forward(self, lambd);
}
Tensor & Type::hardshrink_forward_out(Tensor & output, const Tensor & self, Scalar lambd) const {
    AT_ERROR("hardshrink_forward_out is not implemented for type %s", toString());
}
Tensor Type::hardshrink_forward(const Tensor & self, Scalar lambd) const {
    AT_ERROR("hardshrink_forward is not implemented for type %s", toString());
}
Tensor & Type::hardshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    AT_ERROR("hardshrink_backward_out is not implemented for type %s", toString());
}
Tensor Type::hardshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    AT_ERROR("hardshrink_backward is not implemented for type %s", toString());
}
Tensor & Type::hardtanh_out(Tensor & output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    return hardtanh_forward_out(output, self, min_val, max_val);
}
Tensor Type::hardtanh(const Tensor & self, Scalar min_val, Scalar max_val) const {
    return hardtanh_forward(self, min_val, max_val);
}
Tensor & Type::hardtanh_forward_out(Tensor & output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    AT_ERROR("hardtanh_forward_out is not implemented for type %s", toString());
}
Tensor Type::hardtanh_forward(const Tensor & self, Scalar min_val, Scalar max_val) const {
    AT_ERROR("hardtanh_forward is not implemented for type %s", toString());
}
Tensor & Type::hardtanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    AT_ERROR("hardtanh_backward_out is not implemented for type %s", toString());
}
Tensor Type::hardtanh_backward(const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    AT_ERROR("hardtanh_backward is not implemented for type %s", toString());
}
Tensor & Type::hardtanh_(Tensor & self, Scalar min_val, Scalar max_val) const {
    return hardtanh_forward_(self, min_val, max_val);
}
Tensor & Type::hardtanh_forward_(Tensor & self, Scalar min_val, Scalar max_val) const {
    AT_ERROR("hardtanh_forward_ is not implemented for type %s", toString());
}
Tensor & Type::leaky_relu_out(Tensor & output, const Tensor & self, Scalar negative_slope) const {
    return leaky_relu_forward_out(output, self, negative_slope);
}
Tensor Type::leaky_relu(const Tensor & self, Scalar negative_slope) const {
    return leaky_relu_forward(self, negative_slope);
}
Tensor & Type::leaky_relu_forward_out(Tensor & output, const Tensor & self, Scalar negative_slope) const {
    AT_ERROR("leaky_relu_forward_out is not implemented for type %s", toString());
}
Tensor Type::leaky_relu_forward(const Tensor & self, Scalar negative_slope) const {
    AT_ERROR("leaky_relu_forward is not implemented for type %s", toString());
}
Tensor & Type::leaky_relu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar negative_slope) const {
    AT_ERROR("leaky_relu_backward_out is not implemented for type %s", toString());
}
Tensor Type::leaky_relu_backward(const Tensor & grad_output, const Tensor & self, Scalar negative_slope) const {
    AT_ERROR("leaky_relu_backward is not implemented for type %s", toString());
}
Tensor & Type::leaky_relu_(Tensor & self, Scalar negative_slope) const {
    return leaky_relu_forward_(self, negative_slope);
}
Tensor & Type::leaky_relu_forward_(Tensor & self, Scalar negative_slope) const {
    AT_ERROR("leaky_relu_forward_ is not implemented for type %s", toString());
}
Tensor & Type::log_sigmoid_out(Tensor & output, const Tensor & self) const {
    Tensor buffer = tensor();
    return std::get<0>(log_sigmoid_forward_out(output, buffer, self));
}
Tensor Type::log_sigmoid(const Tensor & self) const {
    return std::get<0>(log_sigmoid_forward(self));
}
std::tuple<Tensor &,Tensor &> Type::log_sigmoid_forward_out(Tensor & output, Tensor & buffer, const Tensor & self) const {
    AT_ERROR("log_sigmoid_forward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::log_sigmoid_forward(const Tensor & self) const {
    AT_ERROR("log_sigmoid_forward is not implemented for type %s", toString());
}
Tensor & Type::log_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & buffer) const {
    AT_ERROR("log_sigmoid_backward_out is not implemented for type %s", toString());
}
Tensor Type::log_sigmoid_backward(const Tensor & grad_output, const Tensor & self, const Tensor & buffer) const {
    AT_ERROR("log_sigmoid_backward is not implemented for type %s", toString());
}
Tensor & Type::log_softmax_out(Tensor & output, const Tensor & self, int64_t dim) const {
    return log_softmax_forward_out(output, self, dim);
}
Tensor Type::log_softmax(const Tensor & self, int64_t dim) const {
    return log_softmax_forward(self, dim);
}
Tensor & Type::log_softmax_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
    AT_ERROR("log_softmax_forward_out is not implemented for type %s", toString());
}
Tensor Type::log_softmax_forward(const Tensor & self, int64_t dim) const {
    AT_ERROR("log_softmax_forward is not implemented for type %s", toString());
}
Tensor & Type::log_softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    AT_ERROR("log_softmax_backward_out is not implemented for type %s", toString());
}
Tensor Type::log_softmax_backward(const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    AT_ERROR("log_softmax_backward is not implemented for type %s", toString());
}
Tensor & Type::prelu_out(Tensor & output, const Tensor & self, const Tensor & weight) const {
    return prelu_forward_out(output, self, weight);
}
Tensor Type::prelu(const Tensor & self, const Tensor & weight) const {
    return prelu_forward(self, weight);
}
Tensor & Type::prelu_forward_out(Tensor & output, const Tensor & self, const Tensor & weight) const {
    AT_ERROR("prelu_forward_out is not implemented for type %s", toString());
}
Tensor Type::prelu_forward(const Tensor & self, const Tensor & weight) const {
    AT_ERROR("prelu_forward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::prelu_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight) const {
    AT_ERROR("prelu_backward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::prelu_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, std::array<bool,2> output_mask) const {
    AT_ERROR("prelu_backward is not implemented for type %s", toString());
}
Tensor & Type::rrelu_with_noise_out(Tensor & output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    return rrelu_with_noise_forward_out(output, self, noise, lower, upper, training, generator);
}
Tensor Type::rrelu_with_noise(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    return rrelu_with_noise_forward(self, noise, lower, upper, training, generator);
}
Tensor & Type::rrelu_with_noise_forward_out(Tensor & output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    AT_ERROR("rrelu_with_noise_forward_out is not implemented for type %s", toString());
}
Tensor Type::rrelu_with_noise_forward(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    AT_ERROR("rrelu_with_noise_forward is not implemented for type %s", toString());
}
Tensor & Type::rrelu_with_noise_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) const {
    AT_ERROR("rrelu_with_noise_backward_out is not implemented for type %s", toString());
}
Tensor Type::rrelu_with_noise_backward(const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) const {
    AT_ERROR("rrelu_with_noise_backward is not implemented for type %s", toString());
}
Tensor & Type::rrelu_with_noise_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    return rrelu_with_noise_forward_(self, noise, lower, upper, training, generator);
}
Tensor & Type::rrelu_with_noise_forward_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    AT_ERROR("rrelu_with_noise_forward_ is not implemented for type %s", toString());
}
Tensor & Type::softmax_out(Tensor & output, const Tensor & self, int64_t dim) const {
    return softmax_forward_out(output, self, dim);
}
Tensor Type::softmax(const Tensor & self, int64_t dim) const {
    return softmax_forward(self, dim);
}
Tensor & Type::softmax_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
    AT_ERROR("softmax_forward_out is not implemented for type %s", toString());
}
Tensor Type::softmax_forward(const Tensor & self, int64_t dim) const {
    AT_ERROR("softmax_forward is not implemented for type %s", toString());
}
Tensor & Type::softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    AT_ERROR("softmax_backward_out is not implemented for type %s", toString());
}
Tensor Type::softmax_backward(const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    AT_ERROR("softmax_backward is not implemented for type %s", toString());
}
Tensor & Type::softplus_out(Tensor & output, const Tensor & self, Scalar beta, Scalar threshold) const {
    return softplus_forward_out(output, self, beta, threshold);
}
Tensor Type::softplus(const Tensor & self, Scalar beta, Scalar threshold) const {
    return softplus_forward(self, beta, threshold);
}
Tensor & Type::softplus_forward_out(Tensor & output, const Tensor & self, Scalar beta, Scalar threshold) const {
    AT_ERROR("softplus_forward_out is not implemented for type %s", toString());
}
Tensor Type::softplus_forward(const Tensor & self, Scalar beta, Scalar threshold) const {
    AT_ERROR("softplus_forward is not implemented for type %s", toString());
}
Tensor & Type::softplus_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) const {
    AT_ERROR("softplus_backward_out is not implemented for type %s", toString());
}
Tensor Type::softplus_backward(const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) const {
    AT_ERROR("softplus_backward is not implemented for type %s", toString());
}
Tensor & Type::softshrink_out(Tensor & output, const Tensor & self, Scalar lambd) const {
    return softshrink_forward_out(output, self, lambd);
}
Tensor Type::softshrink(const Tensor & self, Scalar lambd) const {
    return softshrink_forward(self, lambd);
}
Tensor & Type::softshrink_forward_out(Tensor & output, const Tensor & self, Scalar lambd) const {
    AT_ERROR("softshrink_forward_out is not implemented for type %s", toString());
}
Tensor Type::softshrink_forward(const Tensor & self, Scalar lambd) const {
    AT_ERROR("softshrink_forward is not implemented for type %s", toString());
}
Tensor & Type::softshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    AT_ERROR("softshrink_backward_out is not implemented for type %s", toString());
}
Tensor Type::softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    AT_ERROR("softshrink_backward is not implemented for type %s", toString());
}
Tensor & Type::threshold_out(Tensor & output, const Tensor & self, Scalar threshold, Scalar value) const {
    return threshold_forward_out(output, self, threshold, value);
}
Tensor Type::threshold(const Tensor & self, Scalar threshold, Scalar value) const {
    return threshold_forward(self, threshold, value);
}
Tensor & Type::threshold_forward_out(Tensor & output, const Tensor & self, Scalar threshold, Scalar value) const {
    AT_ERROR("threshold_forward_out is not implemented for type %s", toString());
}
Tensor Type::threshold_forward(const Tensor & self, Scalar threshold, Scalar value) const {
    AT_ERROR("threshold_forward is not implemented for type %s", toString());
}
Tensor & Type::threshold_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar threshold, Scalar value) const {
    AT_ERROR("threshold_backward_out is not implemented for type %s", toString());
}
Tensor Type::threshold_backward(const Tensor & grad_output, const Tensor & self, Scalar threshold, Scalar value) const {
    AT_ERROR("threshold_backward is not implemented for type %s", toString());
}
Tensor & Type::threshold_(Tensor & self, Scalar threshold, Scalar value) const {
    return threshold_forward_(self, threshold, value);
}
Tensor & Type::threshold_forward_(Tensor & self, Scalar threshold, Scalar value) const {
    AT_ERROR("threshold_forward_ is not implemented for type %s", toString());
}
Tensor & Type::adaptive_avg_pool2d_out(Tensor & output, const Tensor & self, IntList output_size) const {
    return adaptive_avg_pool2d_forward_out(output, self, output_size);
}
Tensor Type::adaptive_avg_pool2d(const Tensor & self, IntList output_size) const {
    return adaptive_avg_pool2d_forward(self, output_size);
}
Tensor & Type::adaptive_avg_pool2d_forward_out(Tensor & output, const Tensor & self, IntList output_size) const {
    AT_ERROR("adaptive_avg_pool2d_forward_out is not implemented for type %s", toString());
}
Tensor Type::adaptive_avg_pool2d_forward(const Tensor & self, IntList output_size) const {
    AT_ERROR("adaptive_avg_pool2d_forward is not implemented for type %s", toString());
}
Tensor & Type::adaptive_avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) const {
    AT_ERROR("adaptive_avg_pool2d_backward_out is not implemented for type %s", toString());
}
Tensor Type::adaptive_avg_pool2d_backward(const Tensor & grad_output, const Tensor & self) const {
    AT_ERROR("adaptive_avg_pool2d_backward is not implemented for type %s", toString());
}
Tensor & Type::adaptive_avg_pool3d_out(Tensor & output, const Tensor & self, IntList output_size) const {
    return adaptive_avg_pool3d_forward_out(output, self, output_size);
}
Tensor Type::adaptive_avg_pool3d(const Tensor & self, IntList output_size) const {
    return adaptive_avg_pool3d_forward(self, output_size);
}
Tensor & Type::adaptive_avg_pool3d_forward_out(Tensor & output, const Tensor & self, IntList output_size) const {
    AT_ERROR("adaptive_avg_pool3d_forward_out is not implemented for type %s", toString());
}
Tensor Type::adaptive_avg_pool3d_forward(const Tensor & self, IntList output_size) const {
    AT_ERROR("adaptive_avg_pool3d_forward is not implemented for type %s", toString());
}
Tensor & Type::adaptive_avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) const {
    AT_ERROR("adaptive_avg_pool3d_backward_out is not implemented for type %s", toString());
}
Tensor Type::adaptive_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self) const {
    AT_ERROR("adaptive_avg_pool3d_backward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::adaptive_max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) const {
    return adaptive_max_pool2d_forward_out(output, indices, self, output_size);
}
std::tuple<Tensor,Tensor> Type::adaptive_max_pool2d(const Tensor & self, IntList output_size) const {
    return adaptive_max_pool2d_forward(self, output_size);
}
std::tuple<Tensor &,Tensor &> Type::adaptive_max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) const {
    AT_ERROR("adaptive_max_pool2d_forward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::adaptive_max_pool2d_forward(const Tensor & self, IntList output_size) const {
    AT_ERROR("adaptive_max_pool2d_forward is not implemented for type %s", toString());
}
Tensor & Type::adaptive_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    AT_ERROR("adaptive_max_pool2d_backward_out is not implemented for type %s", toString());
}
Tensor Type::adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    AT_ERROR("adaptive_max_pool2d_backward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::adaptive_max_pool3d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) const {
    return adaptive_max_pool3d_forward_out(output, indices, self, output_size);
}
std::tuple<Tensor,Tensor> Type::adaptive_max_pool3d(const Tensor & self, IntList output_size) const {
    return adaptive_max_pool3d_forward(self, output_size);
}
std::tuple<Tensor &,Tensor &> Type::adaptive_max_pool3d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) const {
    AT_ERROR("adaptive_max_pool3d_forward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::adaptive_max_pool3d_forward(const Tensor & self, IntList output_size) const {
    AT_ERROR("adaptive_max_pool3d_forward is not implemented for type %s", toString());
}
Tensor & Type::adaptive_max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    AT_ERROR("adaptive_max_pool3d_backward_out is not implemented for type %s", toString());
}
Tensor Type::adaptive_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    AT_ERROR("adaptive_max_pool3d_backward is not implemented for type %s", toString());
}
Tensor & Type::avg_pool2d_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    return avg_pool2d_forward_out(output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
Tensor Type::avg_pool2d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    return avg_pool2d_forward(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
Tensor & Type::avg_pool2d_forward_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    AT_ERROR("avg_pool2d_forward_out is not implemented for type %s", toString());
}
Tensor Type::avg_pool2d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    AT_ERROR("avg_pool2d_forward is not implemented for type %s", toString());
}
Tensor & Type::avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    AT_ERROR("avg_pool2d_backward_out is not implemented for type %s", toString());
}
Tensor Type::avg_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    AT_ERROR("avg_pool2d_backward is not implemented for type %s", toString());
}
Tensor & Type::avg_pool3d_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    return avg_pool3d_forward_out(output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
Tensor Type::avg_pool3d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    return avg_pool3d_forward(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
Tensor & Type::avg_pool3d_forward_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    AT_ERROR("avg_pool3d_forward_out is not implemented for type %s", toString());
}
Tensor Type::avg_pool3d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    AT_ERROR("avg_pool3d_forward is not implemented for type %s", toString());
}
Tensor & Type::avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    AT_ERROR("avg_pool3d_backward_out is not implemented for type %s", toString());
}
Tensor Type::avg_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    AT_ERROR("avg_pool3d_backward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::fractional_max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) const {
    return fractional_max_pool2d_forward_out(output, indices, self, kernel_size, output_size, random_samples);
}
std::tuple<Tensor,Tensor> Type::fractional_max_pool2d(const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) const {
    return fractional_max_pool2d_forward(self, kernel_size, output_size, random_samples);
}
std::tuple<Tensor &,Tensor &> Type::fractional_max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) const {
    AT_ERROR("fractional_max_pool2d_forward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::fractional_max_pool2d_forward(const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) const {
    AT_ERROR("fractional_max_pool2d_forward is not implemented for type %s", toString());
}
Tensor & Type::fractional_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & indices) const {
    AT_ERROR("fractional_max_pool2d_backward_out is not implemented for type %s", toString());
}
Tensor Type::fractional_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & indices) const {
    AT_ERROR("fractional_max_pool2d_backward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    return max_pool2d_forward_out(output, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}
std::tuple<Tensor,Tensor> Type::max_pool2d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    return max_pool2d_forward(self, kernel_size, stride, padding, dilation, ceil_mode);
}
std::tuple<Tensor &,Tensor &> Type::max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    AT_ERROR("max_pool2d_forward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::max_pool2d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    AT_ERROR("max_pool2d_forward is not implemented for type %s", toString());
}
Tensor & Type::max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    AT_ERROR("max_pool2d_backward_out is not implemented for type %s", toString());
}
Tensor Type::max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    AT_ERROR("max_pool2d_backward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::max_pool3d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    return max_pool3d_forward_out(output, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}
std::tuple<Tensor,Tensor> Type::max_pool3d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    return max_pool3d_forward(self, kernel_size, stride, padding, dilation, ceil_mode);
}
std::tuple<Tensor &,Tensor &> Type::max_pool3d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    AT_ERROR("max_pool3d_forward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::max_pool3d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    AT_ERROR("max_pool3d_forward is not implemented for type %s", toString());
}
Tensor & Type::max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    AT_ERROR("max_pool3d_backward_out is not implemented for type %s", toString());
}
Tensor Type::max_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    AT_ERROR("max_pool3d_backward is not implemented for type %s", toString());
}
Tensor & Type::max_unpool2d_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size) const {
    return max_unpool2d_forward_out(output, self, indices, output_size);
}
Tensor Type::max_unpool2d(const Tensor & self, const Tensor & indices, IntList output_size) const {
    return max_unpool2d_forward(self, indices, output_size);
}
Tensor & Type::max_unpool2d_forward_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size) const {
    AT_ERROR("max_unpool2d_forward_out is not implemented for type %s", toString());
}
Tensor Type::max_unpool2d_forward(const Tensor & self, const Tensor & indices, IntList output_size) const {
    AT_ERROR("max_unpool2d_forward is not implemented for type %s", toString());
}
Tensor & Type::max_unpool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size) const {
    AT_ERROR("max_unpool2d_backward_out is not implemented for type %s", toString());
}
Tensor Type::max_unpool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size) const {
    AT_ERROR("max_unpool2d_backward is not implemented for type %s", toString());
}
Tensor & Type::max_unpool3d_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    return max_unpool3d_forward_out(output, self, indices, output_size, stride, padding);
}
Tensor Type::max_unpool3d(const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    return max_unpool3d_forward(self, indices, output_size, stride, padding);
}
Tensor & Type::max_unpool3d_forward_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    AT_ERROR("max_unpool3d_forward_out is not implemented for type %s", toString());
}
Tensor Type::max_unpool3d_forward(const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    AT_ERROR("max_unpool3d_forward is not implemented for type %s", toString());
}
Tensor & Type::max_unpool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    AT_ERROR("max_unpool3d_backward_out is not implemented for type %s", toString());
}
Tensor Type::max_unpool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    AT_ERROR("max_unpool3d_backward is not implemented for type %s", toString());
}
Tensor & Type::reflection_pad1d_out(Tensor & output, const Tensor & self, IntList padding) const {
    return reflection_pad1d_forward_out(output, self, padding);
}
Tensor Type::reflection_pad1d(const Tensor & self, IntList padding) const {
    return reflection_pad1d_forward(self, padding);
}
Tensor & Type::reflection_pad1d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    AT_ERROR("reflection_pad1d_forward_out is not implemented for type %s", toString());
}
Tensor Type::reflection_pad1d_forward(const Tensor & self, IntList padding) const {
    AT_ERROR("reflection_pad1d_forward is not implemented for type %s", toString());
}
Tensor & Type::reflection_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    AT_ERROR("reflection_pad1d_backward_out is not implemented for type %s", toString());
}
Tensor Type::reflection_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    AT_ERROR("reflection_pad1d_backward is not implemented for type %s", toString());
}
Tensor & Type::reflection_pad2d_out(Tensor & output, const Tensor & self, IntList padding) const {
    return reflection_pad2d_forward_out(output, self, padding);
}
Tensor Type::reflection_pad2d(const Tensor & self, IntList padding) const {
    return reflection_pad2d_forward(self, padding);
}
Tensor & Type::reflection_pad2d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    AT_ERROR("reflection_pad2d_forward_out is not implemented for type %s", toString());
}
Tensor Type::reflection_pad2d_forward(const Tensor & self, IntList padding) const {
    AT_ERROR("reflection_pad2d_forward is not implemented for type %s", toString());
}
Tensor & Type::reflection_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    AT_ERROR("reflection_pad2d_backward_out is not implemented for type %s", toString());
}
Tensor Type::reflection_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    AT_ERROR("reflection_pad2d_backward is not implemented for type %s", toString());
}
Tensor & Type::replication_pad1d_out(Tensor & output, const Tensor & self, IntList padding) const {
    return replication_pad1d_forward_out(output, self, padding);
}
Tensor Type::replication_pad1d(const Tensor & self, IntList padding) const {
    return replication_pad1d_forward(self, padding);
}
Tensor & Type::replication_pad1d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    AT_ERROR("replication_pad1d_forward_out is not implemented for type %s", toString());
}
Tensor Type::replication_pad1d_forward(const Tensor & self, IntList padding) const {
    AT_ERROR("replication_pad1d_forward is not implemented for type %s", toString());
}
Tensor & Type::replication_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    AT_ERROR("replication_pad1d_backward_out is not implemented for type %s", toString());
}
Tensor Type::replication_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    AT_ERROR("replication_pad1d_backward is not implemented for type %s", toString());
}
Tensor & Type::replication_pad2d_out(Tensor & output, const Tensor & self, IntList padding) const {
    return replication_pad2d_forward_out(output, self, padding);
}
Tensor Type::replication_pad2d(const Tensor & self, IntList padding) const {
    return replication_pad2d_forward(self, padding);
}
Tensor & Type::replication_pad2d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    AT_ERROR("replication_pad2d_forward_out is not implemented for type %s", toString());
}
Tensor Type::replication_pad2d_forward(const Tensor & self, IntList padding) const {
    AT_ERROR("replication_pad2d_forward is not implemented for type %s", toString());
}
Tensor & Type::replication_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    AT_ERROR("replication_pad2d_backward_out is not implemented for type %s", toString());
}
Tensor Type::replication_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    AT_ERROR("replication_pad2d_backward is not implemented for type %s", toString());
}
Tensor & Type::replication_pad3d_out(Tensor & output, const Tensor & self, IntList padding) const {
    return replication_pad3d_forward_out(output, self, padding);
}
Tensor Type::replication_pad3d(const Tensor & self, IntList padding) const {
    return replication_pad3d_forward(self, padding);
}
Tensor & Type::replication_pad3d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    AT_ERROR("replication_pad3d_forward_out is not implemented for type %s", toString());
}
Tensor Type::replication_pad3d_forward(const Tensor & self, IntList padding) const {
    AT_ERROR("replication_pad3d_forward is not implemented for type %s", toString());
}
Tensor & Type::replication_pad3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    AT_ERROR("replication_pad3d_backward_out is not implemented for type %s", toString());
}
Tensor Type::replication_pad3d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    AT_ERROR("replication_pad3d_backward is not implemented for type %s", toString());
}
Tensor & Type::upsample_linear1d_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
    return upsample_linear1d_forward_out(output, self, output_size, align_corners);
}
Tensor Type::upsample_linear1d(const Tensor & self, IntList output_size, bool align_corners) const {
    return upsample_linear1d_forward(self, output_size, align_corners);
}
Tensor & Type::upsample_linear1d_forward_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
    AT_ERROR("upsample_linear1d_forward_out is not implemented for type %s", toString());
}
Tensor Type::upsample_linear1d_forward(const Tensor & self, IntList output_size, bool align_corners) const {
    AT_ERROR("upsample_linear1d_forward is not implemented for type %s", toString());
}
Tensor & Type::upsample_linear1d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    AT_ERROR("upsample_linear1d_backward_out is not implemented for type %s", toString());
}
Tensor Type::upsample_linear1d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    AT_ERROR("upsample_linear1d_backward is not implemented for type %s", toString());
}
Tensor & Type::upsample_bilinear2d_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
    return upsample_bilinear2d_forward_out(output, self, output_size, align_corners);
}
Tensor Type::upsample_bilinear2d(const Tensor & self, IntList output_size, bool align_corners) const {
    return upsample_bilinear2d_forward(self, output_size, align_corners);
}
Tensor & Type::upsample_bilinear2d_forward_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
    AT_ERROR("upsample_bilinear2d_forward_out is not implemented for type %s", toString());
}
Tensor Type::upsample_bilinear2d_forward(const Tensor & self, IntList output_size, bool align_corners) const {
    AT_ERROR("upsample_bilinear2d_forward is not implemented for type %s", toString());
}
Tensor & Type::upsample_bilinear2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    AT_ERROR("upsample_bilinear2d_backward_out is not implemented for type %s", toString());
}
Tensor Type::upsample_bilinear2d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    AT_ERROR("upsample_bilinear2d_backward is not implemented for type %s", toString());
}
Tensor & Type::upsample_trilinear3d_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
    return upsample_trilinear3d_forward_out(output, self, output_size, align_corners);
}
Tensor Type::upsample_trilinear3d(const Tensor & self, IntList output_size, bool align_corners) const {
    return upsample_trilinear3d_forward(self, output_size, align_corners);
}
Tensor & Type::upsample_trilinear3d_forward_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
    AT_ERROR("upsample_trilinear3d_forward_out is not implemented for type %s", toString());
}
Tensor Type::upsample_trilinear3d_forward(const Tensor & self, IntList output_size, bool align_corners) const {
    AT_ERROR("upsample_trilinear3d_forward is not implemented for type %s", toString());
}
Tensor & Type::upsample_trilinear3d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    AT_ERROR("upsample_trilinear3d_backward_out is not implemented for type %s", toString());
}
Tensor Type::upsample_trilinear3d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    AT_ERROR("upsample_trilinear3d_backward is not implemented for type %s", toString());
}
Tensor & Type::upsample_nearest1d_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
    return upsample_nearest1d_forward_out(output, self, scale_factor);
}
Tensor Type::upsample_nearest1d(const Tensor & self, int64_t scale_factor) const {
    return upsample_nearest1d_forward(self, scale_factor);
}
Tensor & Type::upsample_nearest1d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
    AT_ERROR("upsample_nearest1d_forward_out is not implemented for type %s", toString());
}
Tensor Type::upsample_nearest1d_forward(const Tensor & self, int64_t scale_factor) const {
    AT_ERROR("upsample_nearest1d_forward is not implemented for type %s", toString());
}
Tensor & Type::upsample_nearest1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    AT_ERROR("upsample_nearest1d_backward_out is not implemented for type %s", toString());
}
Tensor Type::upsample_nearest1d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    AT_ERROR("upsample_nearest1d_backward is not implemented for type %s", toString());
}
Tensor & Type::upsample_nearest2d_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
    return upsample_nearest2d_forward_out(output, self, scale_factor);
}
Tensor Type::upsample_nearest2d(const Tensor & self, int64_t scale_factor) const {
    return upsample_nearest2d_forward(self, scale_factor);
}
Tensor & Type::upsample_nearest2d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
    AT_ERROR("upsample_nearest2d_forward_out is not implemented for type %s", toString());
}
Tensor Type::upsample_nearest2d_forward(const Tensor & self, int64_t scale_factor) const {
    AT_ERROR("upsample_nearest2d_forward is not implemented for type %s", toString());
}
Tensor & Type::upsample_nearest2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    AT_ERROR("upsample_nearest2d_backward_out is not implemented for type %s", toString());
}
Tensor Type::upsample_nearest2d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    AT_ERROR("upsample_nearest2d_backward is not implemented for type %s", toString());
}
Tensor & Type::upsample_nearest3d_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
    return upsample_nearest3d_forward_out(output, self, scale_factor);
}
Tensor Type::upsample_nearest3d(const Tensor & self, int64_t scale_factor) const {
    return upsample_nearest3d_forward(self, scale_factor);
}
Tensor & Type::upsample_nearest3d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
    AT_ERROR("upsample_nearest3d_forward_out is not implemented for type %s", toString());
}
Tensor Type::upsample_nearest3d_forward(const Tensor & self, int64_t scale_factor) const {
    AT_ERROR("upsample_nearest3d_forward is not implemented for type %s", toString());
}
Tensor & Type::upsample_nearest3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    AT_ERROR("upsample_nearest3d_backward_out is not implemented for type %s", toString());
}
Tensor Type::upsample_nearest3d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    AT_ERROR("upsample_nearest3d_backward is not implemented for type %s", toString());
}
Tensor & Type::_sigmoid_out(Tensor & output, const Tensor & self) const {
    return _sigmoid_forward_out(output, self);
}
Tensor Type::_sigmoid(const Tensor & self) const {
    return _sigmoid_forward(self);
}
Tensor & Type::_sigmoid_forward_out(Tensor & output, const Tensor & self) const {
    AT_ERROR("_sigmoid_forward_out is not implemented for type %s", toString());
}
Tensor Type::_sigmoid_forward(const Tensor & self) const {
    AT_ERROR("_sigmoid_forward is not implemented for type %s", toString());
}
Tensor & Type::_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
    AT_ERROR("_sigmoid_backward_out is not implemented for type %s", toString());
}
Tensor Type::_sigmoid_backward(const Tensor & grad_output, const Tensor & output) const {
    AT_ERROR("_sigmoid_backward is not implemented for type %s", toString());
}
Tensor & Type::_tanh_out(Tensor & output, const Tensor & self) const {
    return _tanh_forward_out(output, self);
}
Tensor Type::_tanh(const Tensor & self) const {
    return _tanh_forward(self);
}
Tensor & Type::_tanh_forward_out(Tensor & output, const Tensor & self) const {
    AT_ERROR("_tanh_forward_out is not implemented for type %s", toString());
}
Tensor Type::_tanh_forward(const Tensor & self) const {
    AT_ERROR("_tanh_forward is not implemented for type %s", toString());
}
Tensor & Type::_tanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
    AT_ERROR("_tanh_backward_out is not implemented for type %s", toString());
}
Tensor Type::_tanh_backward(const Tensor & grad_output, const Tensor & output) const {
    AT_ERROR("_tanh_backward is not implemented for type %s", toString());
}
Tensor & Type::thnn_batch_norm_out(Tensor & output, const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) const {
    Tensor save_mean = tensor();
    Tensor save_std = tensor();
    return std::get<0>(thnn_batch_norm_forward_out(output, save_mean, save_std, self, weight, bias, running_mean, running_var, training, momentum, eps));
}
Tensor Type::thnn_batch_norm(const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) const {
    return std::get<0>(thnn_batch_norm_forward(self, weight, bias, running_mean, running_var, training, momentum, eps));
}
std::tuple<Tensor &,Tensor &,Tensor &> Type::thnn_batch_norm_forward_out(Tensor & output, Tensor & save_mean, Tensor & save_std, const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) const {
    AT_ERROR("thnn_batch_norm_forward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor,Tensor> Type::thnn_batch_norm_forward(const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) const {
    AT_ERROR("thnn_batch_norm_forward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &,Tensor &> Type::thnn_batch_norm_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std) const {
    AT_ERROR("thnn_batch_norm_backward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor,Tensor> Type::thnn_batch_norm_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std, std::array<bool,3> output_mask) const {
    AT_ERROR("thnn_batch_norm_backward is not implemented for type %s", toString());
}
Tensor & Type::thnn_conv_transpose2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    Tensor columns = tensor();
    Tensor ones = tensor();
    return std::get<0>(thnn_conv_transpose2d_forward_out(output, columns, ones, self, weight, kernel_size, bias, stride, padding, output_padding, dilation));
}
Tensor Type::thnn_conv_transpose2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    return std::get<0>(thnn_conv_transpose2d_forward(self, weight, kernel_size, bias, stride, padding, output_padding, dilation));
}
std::tuple<Tensor &,Tensor &,Tensor &> Type::thnn_conv_transpose2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    AT_ERROR("thnn_conv_transpose2d_forward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor,Tensor> Type::thnn_conv_transpose2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    AT_ERROR("thnn_conv_transpose2d_forward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &,Tensor &> Type::thnn_conv_transpose2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    AT_ERROR("thnn_conv_transpose2d_backward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor,Tensor> Type::thnn_conv_transpose2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    AT_ERROR("thnn_conv_transpose2d_backward is not implemented for type %s", toString());
}
Tensor & Type::thnn_conv_transpose3d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    Tensor finput = tensor();
    Tensor fgrad_input = tensor();
    return std::get<0>(thnn_conv_transpose3d_forward_out(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding, output_padding, dilation));
}
Tensor Type::thnn_conv_transpose3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    return std::get<0>(thnn_conv_transpose3d_forward(self, weight, kernel_size, bias, stride, padding, output_padding, dilation));
}
std::tuple<Tensor &,Tensor &,Tensor &> Type::thnn_conv_transpose3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    AT_ERROR("thnn_conv_transpose3d_forward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor,Tensor> Type::thnn_conv_transpose3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    AT_ERROR("thnn_conv_transpose3d_forward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &,Tensor &> Type::thnn_conv_transpose3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input) const {
    AT_ERROR("thnn_conv_transpose3d_backward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor,Tensor> Type::thnn_conv_transpose3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
    AT_ERROR("thnn_conv_transpose3d_backward is not implemented for type %s", toString());
}
Tensor & Type::thnn_conv2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    Tensor finput = tensor();
    Tensor fgrad_input = tensor();
    return std::get<0>(thnn_conv2d_forward_out(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding));
}
Tensor Type::thnn_conv2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    return std::get<0>(thnn_conv2d_forward(self, weight, kernel_size, bias, stride, padding));
}
std::tuple<Tensor &,Tensor &,Tensor &> Type::thnn_conv2d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    AT_ERROR("thnn_conv2d_forward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor,Tensor> Type::thnn_conv2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    AT_ERROR("thnn_conv2d_forward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &,Tensor &> Type::thnn_conv2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) const {
    AT_ERROR("thnn_conv2d_backward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor,Tensor> Type::thnn_conv2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
    AT_ERROR("thnn_conv2d_backward is not implemented for type %s", toString());
}
Tensor & Type::thnn_conv_depthwise2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    return thnn_conv_depthwise2d_forward_out(output, self, weight, kernel_size, bias, stride, padding, dilation);
}
Tensor Type::thnn_conv_depthwise2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    return thnn_conv_depthwise2d_forward(self, weight, kernel_size, bias, stride, padding, dilation);
}
Tensor & Type::thnn_conv_depthwise2d_forward_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    AT_ERROR("thnn_conv_depthwise2d_forward_out is not implemented for type %s", toString());
}
Tensor Type::thnn_conv_depthwise2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    AT_ERROR("thnn_conv_depthwise2d_forward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::thnn_conv_depthwise2d_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation) const {
    AT_ERROR("thnn_conv_depthwise2d_backward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::thnn_conv_depthwise2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, std::array<bool,2> output_mask) const {
    AT_ERROR("thnn_conv_depthwise2d_backward is not implemented for type %s", toString());
}
Tensor & Type::thnn_conv3d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    Tensor finput = tensor();
    Tensor fgrad_input = tensor();
    return std::get<0>(thnn_conv3d_forward_out(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding));
}
Tensor Type::thnn_conv3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    return std::get<0>(thnn_conv3d_forward(self, weight, kernel_size, bias, stride, padding));
}
std::tuple<Tensor &,Tensor &,Tensor &> Type::thnn_conv3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    AT_ERROR("thnn_conv3d_forward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor,Tensor> Type::thnn_conv3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    AT_ERROR("thnn_conv3d_forward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &,Tensor &> Type::thnn_conv3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) const {
    AT_ERROR("thnn_conv3d_backward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor,Tensor> Type::thnn_conv3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
    AT_ERROR("thnn_conv3d_backward is not implemented for type %s", toString());
}
Tensor & Type::thnn_conv_dilated2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    Tensor columns = tensor();
    Tensor ones = tensor();
    return std::get<0>(thnn_conv_dilated2d_forward_out(output, columns, ones, self, weight, kernel_size, bias, stride, padding, dilation));
}
Tensor Type::thnn_conv_dilated2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    return std::get<0>(thnn_conv_dilated2d_forward(self, weight, kernel_size, bias, stride, padding, dilation));
}
std::tuple<Tensor &,Tensor &,Tensor &> Type::thnn_conv_dilated2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    AT_ERROR("thnn_conv_dilated2d_forward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor,Tensor> Type::thnn_conv_dilated2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    AT_ERROR("thnn_conv_dilated2d_forward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &,Tensor &> Type::thnn_conv_dilated2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    AT_ERROR("thnn_conv_dilated2d_backward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor,Tensor> Type::thnn_conv_dilated2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    AT_ERROR("thnn_conv_dilated2d_backward is not implemented for type %s", toString());
}
Tensor & Type::thnn_conv_dilated3d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    Tensor columns = tensor();
    Tensor ones = tensor();
    return std::get<0>(thnn_conv_dilated3d_forward_out(output, columns, ones, self, weight, kernel_size, bias, stride, padding, dilation));
}
Tensor Type::thnn_conv_dilated3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    return std::get<0>(thnn_conv_dilated3d_forward(self, weight, kernel_size, bias, stride, padding, dilation));
}
std::tuple<Tensor &,Tensor &,Tensor &> Type::thnn_conv_dilated3d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    AT_ERROR("thnn_conv_dilated3d_forward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor,Tensor> Type::thnn_conv_dilated3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    AT_ERROR("thnn_conv_dilated3d_forward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &,Tensor &> Type::thnn_conv_dilated3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    AT_ERROR("thnn_conv_dilated3d_backward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor,Tensor> Type::thnn_conv_dilated3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    AT_ERROR("thnn_conv_dilated3d_backward is not implemented for type %s", toString());
}
Tensor Type::_cast_uint8_t(const Tensor & self, bool non_blocking) const {
    return  at::native::_cast_uint8_t(self, non_blocking);

}
Tensor Type::_cast_int8_t(const Tensor & self, bool non_blocking) const {
    return  at::native::_cast_int8_t(self, non_blocking);

}
Tensor Type::_cast_double(const Tensor & self, bool non_blocking) const {
    return  at::native::_cast_double(self, non_blocking);

}
Tensor Type::_cast_float(const Tensor & self, bool non_blocking) const {
    return  at::native::_cast_float(self, non_blocking);

}
Tensor Type::_cast_int(const Tensor & self, bool non_blocking) const {
    return  at::native::_cast_int(self, non_blocking);

}
Tensor Type::_cast_int64_t(const Tensor & self, bool non_blocking) const {
    return  at::native::_cast_int64_t(self, non_blocking);

}
Tensor Type::_cast_int16_t(const Tensor & self, bool non_blocking) const {
    return  at::native::_cast_int16_t(self, non_blocking);

}
Tensor Type::_cast_Half(const Tensor & self, bool non_blocking) const {
    return  at::native::_cast_Half(self, non_blocking);

}
Tensor Type::_cudnn_rnn_flatten_weight(TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, bool bidirectional) const {
    return  at::native::_cudnn_rnn_flatten_weight(weight_arr, weight_stride0, input_size, mode, hidden_size, num_layers, batch_first, bidirectional);

}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> Type::_cudnn_rnn(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntList batch_sizes, const Tensor & dropout_state) const {
    return  at::native::_cudnn_rnn(input, weight, weight_stride0, weight_buf, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);

}
std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>> Type::_cudnn_rnn_backward(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, const Tensor & output, const Tensor & grad_output, const Tensor & grad_hy, const Tensor & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntList batch_sizes, const Tensor & dropout_state, const Tensor & reserve, std::array<bool,4> output_mask) const {
    return  at::native::_cudnn_rnn_backward(input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask);

}
Tensor Type::_cudnn_init_dropout_state(double dropout, bool train, int64_t dropout_seed) const {
    return  at::native::_cudnn_init_dropout_state(*this, dropout, train, dropout_seed);

}
Tensor Type::abs(const Tensor & self) const {
    return  at::native::abs(self);

}
Tensor & Type::abs_(Tensor & self) const {
    return  at::native::abs_(self);

}
Tensor & Type::abs_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("abs_out is not implemented for type %s", toString());
}
Tensor Type::adaptive_avg_pool1d(const Tensor & self, IntList output_size) const {
    return  at::native::adaptive_avg_pool1d(self, output_size);

}
std::tuple<Tensor,Tensor> Type::adaptive_max_pool1d(const Tensor & self, IntList output_size) const {
    return  at::native::adaptive_max_pool1d(self, output_size);

}
bool Type::allclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) const {
    return  at::native::allclose(self, other, rtol, atol, equal_nan);

}
Tensor Type::addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    return  at::native::addmv(self, mat, vec, beta, alpha);

}
Tensor & Type::addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    return  at::native::addmv_(self, mat, vec, beta, alpha);

}
Tensor & Type::addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    return  at::native::addmv_out(result, self, mat, vec, beta, alpha);

}
Tensor Type::addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    return  at::native::addr(self, vec1, vec2, beta, alpha);

}
Tensor & Type::addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    return  at::native::addr_(self, vec1, vec2, beta, alpha);

}
Tensor & Type::addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    return  at::native::addr_out(result, self, vec1, vec2, beta, alpha);

}
Tensor Type::arange(Scalar start, Scalar end, Scalar step) const {
    return  at::native::arange(*this, start, end, step);

}
Tensor & Type::arange_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    return  at::native::arange_out(result, start, end, step);

}
Tensor Type::arange(Scalar end) const {
    return  at::native::arange(*this, end);

}
Tensor & Type::arange_out(Tensor & result, Scalar end) const {
    return  at::native::arange_out(result, end);

}
Tensor Type::argmax(const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::argmax(self, dim, keepdim);

}
Tensor Type::argmax(const Tensor & self) const {
    return  at::native::argmax(self);

}
Tensor Type::_argmax(const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::_argmax(self, dim, keepdim);

}
Tensor Type::argmin(const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::argmin(self, dim, keepdim);

}
Tensor Type::argmin(const Tensor & self) const {
    return  at::native::argmin(self);

}
Tensor Type::_argmin(const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::_argmin(self, dim, keepdim);

}
Tensor Type::batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, bool cudnn_enabled) const {
    return  at::native::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);

}
Tensor & Type::bernoulli_(Tensor & self, const Tensor & p, Generator * generator) const {
    return  at::native::bernoulli_(self, p, generator);

}
Tensor & Type::bernoulli_(Tensor & self, double p, Generator * generator) const {
    return  at::native::bernoulli_(self, p, generator);

}
Tensor Type::bilinear(const Tensor & input1, const Tensor & input2, const Tensor & weight, const Tensor & bias) const {
    return  at::native::bilinear(input1, input2, weight, bias);

}
Tensor Type::cat(TensorList tensors, int64_t dim) const {
    return  at::native::cat(tensors, dim);

}
Tensor & Type::cat_out(Tensor & result, TensorList tensors, int64_t dim) const {
    return  at::native::cat_out(result, tensors, dim);

}
Tensor Type::ceil(const Tensor & self) const {
    return  at::native::ceil(self);

}
Tensor & Type::ceil_(Tensor & self) const {
    return  at::native::ceil_(self);

}
Tensor & Type::ceil_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("ceil_out is not implemented for type %s", toString());
}
std::vector<Tensor> Type::chunk(const Tensor & self, int64_t chunks, int64_t dim) const {
    return  at::native::chunk(self, chunks, dim);

}
bool Type::cudnn_is_acceptable(const Tensor & self) const {
    return  at::native::cudnn_is_acceptable(self);

}
Tensor Type::convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding, int64_t groups) const {
    return  at::native::convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);

}
Tensor Type::_convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled) const {
    return  at::native::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);

}
Tensor Type::_convolution_nogroup(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding) const {
    return  at::native::_convolution_nogroup(input, weight, bias, stride, padding, dilation, transposed, output_padding);

}
std::tuple<Tensor,Tensor,Tensor> Type::_convolution_double_backward(const Tensor & ggI, const Tensor & ggW, const Tensor & ggb, const Tensor & gO, const Tensor & weight, const Tensor & self, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, std::array<bool,3> output_mask) const {
    return  at::native::_convolution_double_backward(ggI, ggW, ggb, gO, weight, self, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, output_mask);

}
Tensor Type::conv1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, int64_t groups) const {
    return  at::native::conv1d(input, weight, bias, stride, padding, dilation, groups);

}
Tensor Type::conv2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, int64_t groups) const {
    return  at::native::conv2d(input, weight, bias, stride, padding, dilation, groups);

}
Tensor Type::conv3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, int64_t groups) const {
    return  at::native::conv3d(input, weight, bias, stride, padding, dilation, groups);

}
Tensor Type::conv_tbc(const Tensor & self, const Tensor & weight, const Tensor & bias, int64_t pad) const {
    return  at::native::conv_tbc(self, weight, bias, pad);

}
std::tuple<Tensor,Tensor,Tensor> Type::conv_tbc_backward(const Tensor & self, const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t pad) const {
    return  at::native::conv_tbc_backward(self, input, weight, bias, pad);

}
Tensor Type::conv_transpose1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, int64_t groups, IntList dilation) const {
    return  at::native::conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation);

}
Tensor Type::conv_transpose2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, int64_t groups, IntList dilation) const {
    return  at::native::conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation);

}
Tensor Type::conv_transpose3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, int64_t groups, IntList dilation) const {
    return  at::native::conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation);

}
Tensor Type::cos(const Tensor & self) const {
    return  at::native::cos(self);

}
Tensor & Type::cos_(Tensor & self) const {
    return  at::native::cos_(self);

}
Tensor & Type::cos_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("cos_out is not implemented for type %s", toString());
}
Tensor Type::cosine_embedding_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, bool size_average, bool reduce) const {
    return  at::native::cosine_embedding_loss(input1, input2, target, margin, size_average, reduce);

}
Tensor Type::cudnn_affine_grid_generator(const Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W) const {
    return  at::native::cudnn_affine_grid_generator_forward(theta, N, C, H, W);

}
Tensor Type::cudnn_affine_grid_generator_backward(const Tensor & grad, int64_t N, int64_t C, int64_t H, int64_t W) const {
    return  at::native::cudnn_affine_grid_generator_backward(grad, N, C, H, W);

}
std::tuple<Tensor,Tensor,Tensor> Type::cudnn_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double exponential_average_factor, double epsilon) const {
    return  at::native::cudnn_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);

}
std::tuple<Tensor,Tensor,Tensor> Type::cudnn_batch_norm_backward(const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var, double epsilon) const {
    return  at::native::cudnn_batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon);

}
Tensor Type::cudnn_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) const {
    return  at::native::cudnn_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);

}
Tensor Type::cudnn_convolution_backward_input(IntList self_size, const Tensor & grad_output, const Tensor & weight, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) const {
    return  at::native::cudnn_convolution_backward_input(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);

}
std::tuple<Tensor,Tensor,Tensor> Type::cudnn_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) const {
    return  at::native::cudnn_convolution_backward(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);

}
Tensor Type::cudnn_convolution_backward_bias(const Tensor & grad_output) const {
    return  at::native::cudnn_convolution_backward_bias(grad_output);

}
Tensor Type::cudnn_convolution_backward_weight(IntList weight_size, const Tensor & grad_output, const Tensor & self, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) const {
    return  at::native::cudnn_convolution_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);

}
Tensor Type::cudnn_convolution_transpose(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList output_padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) const {
    return  at::native::cudnn_convolution_transpose(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);

}
std::tuple<Tensor,Tensor,Tensor> Type::cudnn_convolution_transpose_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntList padding, IntList output_padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) const {
    return  at::native::cudnn_convolution_transpose_backward(self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);

}
Tensor Type::cudnn_convolution_transpose_backward_bias(const Tensor & grad_output) const {
    return  at::native::cudnn_convolution_backward_bias(grad_output);

}
Tensor Type::cudnn_convolution_transpose_backward_input(const Tensor & grad_output, const Tensor & weight, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) const {
    return  at::native::cudnn_convolution_transpose_backward_input(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);

}
Tensor Type::cudnn_convolution_transpose_backward_weight(IntList weight_size, const Tensor & grad_output, const Tensor & self, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) const {
    return  at::native::cudnn_convolution_transpose_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);

}
Tensor Type::cudnn_grid_sampler(const Tensor & self, const Tensor & grid) const {
    return  at::native::cudnn_grid_sampler_forward(self, grid);

}
std::tuple<Tensor,Tensor> Type::cudnn_grid_sampler_backward(const Tensor & self, const Tensor & grid, const Tensor & grad_output) const {
    return  at::native::cudnn_grid_sampler_backward(self, grid, grad_output);

}
Tensor Type::det(const Tensor & self) const {
    return  at::native::det(self);

}
Tensor Type::diagflat(const Tensor & self, int64_t offset) const {
    return  at::native::diagflat(self, offset);

}
Tensor Type::diagonal(const Tensor & self, int64_t offset) const {
    return  at::native::diagonal(self, offset);

}
Tensor Type::dot(const Tensor & self, const Tensor & tensor) const {
    return  at::native::dot(self, tensor);

}
Tensor Type::embedding(const Tensor & weight, const Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) const {
    return  at::native::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);

}
Tensor Type::embedding_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) const {
    return  at::native::embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse);

}
Tensor Type::embedding_dense_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) const {
    AT_ERROR("embedding_dense_backward is not implemented for type %s", toString());
}
Tensor & Type::embedding_renorm_(Tensor & self, const Tensor & indices, double max_norm, double norm_type) const {
    AT_ERROR("embedding_renorm_ is not implemented for type %s", toString());
}
Tensor Type::embedding_sparse_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) const {
    return  at::native::embedding_sparse_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq);

}
std::tuple<Tensor,Tensor,Tensor> Type::embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse) const {
    AT_ERROR("embedding_bag is not implemented for type %s", toString());
}
Tensor Type::embedding_bag_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse) const {
    return  at::native::embedding_bag_backward(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode, sparse);

}
Tensor Type::embedding_bag_sparse_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode) const {
    return  at::native::embedding_bag_sparse_backward(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode);

}
Tensor Type::embedding_bag_dense_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode) const {
    AT_ERROR("embedding_bag_dense_backward is not implemented for type %s", toString());
}
Tensor Type::empty(IntList size) const {
    return  at::native::empty(*this, size);

}
Tensor & Type::empty_out(Tensor & result, IntList size) const {
    return  at::native::empty_out(result, size);

}
Tensor Type::empty_like(const Tensor & self) const {
    return  at::native::empty_like(self);

}
Tensor Type::empty_like(const Tensor & self, const Type & dtype) const {
    return  at::native::empty_like(self, dtype);

}
Tensor Type::exp(const Tensor & self) const {
    return  at::native::exp(self);

}
Tensor & Type::exp_(Tensor & self) const {
    return  at::native::exp_(self);

}
Tensor & Type::exp_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("exp_out is not implemented for type %s", toString());
}
Tensor Type::expand(const Tensor & self, IntList size) const {
    return  at::native::expand(self, size);

}
Tensor Type::expand_as(const Tensor & self, const Tensor & other) const {
    return  at::native::expand_as(self, other);

}
Tensor Type::eye(int64_t n, int64_t m) const {
    return  at::native::eye(*this, n, m);

}
Tensor & Type::eye_out(Tensor & result, int64_t n, int64_t m) const {
    AT_ERROR("eye_out is not implemented for type %s", toString());
}
Tensor Type::floor(const Tensor & self) const {
    return  at::native::floor(self);

}
Tensor & Type::floor_(Tensor & self) const {
    return  at::native::floor_(self);

}
Tensor & Type::floor_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("floor_out is not implemented for type %s", toString());
}
Tensor Type::full(IntList size, Scalar fill_value) const {
    return  at::native::full(*this, size, fill_value);

}
Tensor & Type::full_out(Tensor & result, IntList size, Scalar fill_value) const {
    return  at::native::full_out(result, size, fill_value);

}
Tensor Type::full_like(const Tensor & self, Scalar fill_value) const {
    return  at::native::full_like(self, fill_value);

}
Tensor Type::full_like(const Tensor & self, Scalar fill_value, const Type & dtype) const {
    return  at::native::full_like(self, fill_value, dtype);

}
Tensor Type::hinge_embedding_loss(const Tensor & self, const Tensor & target, double margin, bool size_average, bool reduce) const {
    return  at::native::hinge_embedding_loss(self, target, margin, size_average, reduce);

}
Tensor Type::ger(const Tensor & self, const Tensor & vec2) const {
    return  at::native::ger(self, vec2);

}
Tensor & Type::ger_out(Tensor & result, const Tensor & self, const Tensor & vec2) const {
    return  at::native::ger_out(result, self, vec2);

}
Tensor Type::group_norm(const Tensor & input, int64_t num_groups, const Tensor & weight, const Tensor & bias, double eps, bool cudnn_enabled) const {
    return  at::native::group_norm(input, num_groups, weight, bias, eps, cudnn_enabled);

}
Tensor Type::fft(const Tensor & self, int64_t signal_ndim, bool normalized) const {
    return  at::native::fft(self, signal_ndim, normalized);

}
Tensor Type::ifft(const Tensor & self, int64_t signal_ndim, bool normalized) const {
    return  at::native::ifft(self, signal_ndim, normalized);

}
Tensor Type::rfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided) const {
    return  at::native::rfft(self, signal_ndim, normalized, onesided);

}
Tensor Type::irfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided, IntList signal_sizes) const {
    return  at::native::irfft(self, signal_ndim, normalized, onesided, signal_sizes);

}
Tensor Type::_fft_with_size(const Tensor & self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntList checked_signal_sizes, bool normalized, bool onesided, IntList output_sizes) const {
    AT_ERROR("_fft_with_size is not implemented for type %s", toString());
}
Tensor Type::index(const Tensor & self, TensorList indices) const {
    return  at::native::index(self, indices);

}
Tensor & Type::index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    return  at::native::index_copy_(self, dim, index, source);

}
Tensor & Type::index_put_(Tensor & self, TensorList indices, const Tensor & values) const {
    return  at::native::index_put_(self, indices, values);

}
Tensor Type::isclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) const {
    return  at::native::isclose(self, other, rtol, atol, equal_nan);

}
bool Type::is_cuda(const Tensor & self) const {
    return  at::native::is_cuda(self);

}
bool Type::is_distributed(const Tensor & self) const {
    return  at::native::is_distributed(self);

}
bool Type::is_floating_point(const Tensor & self) const {
    return  at::native::is_floating_point(self);

}
bool Type::is_nonzero(const Tensor & self) const {
    return  at::native::is_nonzero(self);

}
bool Type::is_same_size(const Tensor & self, const Tensor & other) const {
    return  at::native::is_same_size(self, other);

}
bool Type::is_signed(const Tensor & self) const {
    return  at::native::is_signed(self);

}
bool Type::is_sparse(const Tensor & self) const {
    return  at::native::is_sparse(self);

}
Tensor Type::layer_norm(const Tensor & input, IntList normalized_shape, const Tensor & weight, const Tensor & bias, double eps, bool cudnn_enable) const {
    return  at::native::layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enable);

}
Tensor Type::linspace(Scalar start, Scalar end, int64_t steps) const {
    return  at::native::linspace(*this, start, end, steps);

}
Tensor & Type::linspace_out(Tensor & result, Scalar start, Scalar end, int64_t steps) const {
    return  at::native::linspace_out(result, start, end, steps);

}
Tensor Type::log(const Tensor & self) const {
    return  at::native::log(self);

}
Tensor & Type::log_(Tensor & self) const {
    return  at::native::log_(self);

}
Tensor & Type::log_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("log_out is not implemented for type %s", toString());
}
Tensor Type::logdet(const Tensor & self) const {
    return  at::native::logdet(self);

}
Tensor Type::logspace(Scalar start, Scalar end, int64_t steps) const {
    return  at::native::logspace(*this, start, end, steps);

}
Tensor & Type::logspace_out(Tensor & result, Scalar start, Scalar end, int64_t steps) const {
    return  at::native::logspace_out(result, start, end, steps);

}
Tensor Type::margin_ranking_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, bool size_average, bool reduce) const {
    return  at::native::margin_ranking_loss(input1, input2, target, margin, size_average, reduce);

}
Tensor Type::matmul(const Tensor & self, const Tensor & other) const {
    return  at::native::matmul(self, other);

}
Tensor Type::max_values(const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::max_values(self, dim, keepdim);

}
std::tuple<Tensor,Tensor> Type::max_pool1d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    return  at::native::max_pool1d(self, kernel_size, stride, padding, dilation, ceil_mode);

}
Tensor Type::min_values(const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::min_values(self, dim, keepdim);

}
Tensor Type::mm(const Tensor & self, const Tensor & mat2) const {
    return  at::native::mm(self, mat2);

}
Tensor & Type::mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    return  at::native::mm_out(result, self, mat2);

}
Tensor Type::mv(const Tensor & self, const Tensor & vec) const {
    return  at::native::mv(self, vec);

}
Tensor & Type::mv_out(Tensor & result, const Tensor & self, const Tensor & vec) const {
    return  at::native::mv_out(result, self, vec);

}
Tensor Type::narrow(const Tensor & self, int64_t dim, int64_t start, int64_t length) const {
    return  at::native::narrow(self, dim, start, length);

}
Tensor Type::ones(IntList size) const {
    return  at::native::ones(*this, size);

}
Tensor & Type::ones_out(Tensor & result, IntList size) const {
    return  at::native::ones_out(result, size);

}
Tensor Type::ones_like(const Tensor & self) const {
    return  at::native::ones_like(self);

}
Tensor Type::ones_like(const Tensor & self, const Type & dtype) const {
    return  at::native::ones_like(self, dtype);

}
Tensor Type::pairwise_distance(const Tensor & x1, const Tensor & x2, double p, double eps, bool keepdim) const {
    return  at::native::pairwise_distance(x1, x2, p, eps, keepdim);

}
Tensor Type::permute(const Tensor & self, IntList dims) const {
    return  at::native::permute(self, dims);

}
Tensor Type::pin_memory(const Tensor & self) const {
    return  at::native::pin_memory(self);

}
Tensor Type::rand(IntList size, Generator * generator) const {
    return  at::native::rand(*this, size, generator);

}
Tensor & Type::rand_out(Tensor & result, IntList size, Generator * generator) const {
    return  at::native::rand_out(result, size, generator);

}
Tensor Type::rand_like(const Tensor & self) const {
    return  at::native::rand_like(self);

}
Tensor Type::rand_like(const Tensor & self, const Type & dtype) const {
    return  at::native::rand_like(self, dtype);

}
Tensor Type::randint(int64_t high, IntList size, Generator * generator) const {
    return  at::native::randint(*this, high, size, generator);

}
Tensor Type::randint(int64_t low, int64_t high, IntList size, Generator * generator) const {
    return  at::native::randint(*this, low, high, size, generator);

}
Tensor & Type::randint_out(Tensor & result, int64_t high, IntList size, Generator * generator) const {
    return  at::native::randint_out(result, high, size, generator);

}
Tensor & Type::randint_out(Tensor & result, int64_t low, int64_t high, IntList size, Generator * generator) const {
    return  at::native::randint_out(result, low, high, size, generator);

}
Tensor Type::randint_like(const Tensor & self, int64_t high) const {
    return  at::native::randint_like(self, high);

}
Tensor Type::randint_like(const Tensor & self, int64_t low, int64_t high) const {
    return  at::native::randint_like(self, low, high);

}
Tensor Type::randint_like(const Tensor & self, int64_t high, const Type & dtype) const {
    return  at::native::randint_like(self, high, dtype);

}
Tensor Type::randint_like(const Tensor & self, int64_t low, int64_t high, const Type & dtype) const {
    return  at::native::randint_like(self, low, high, dtype);

}
Tensor Type::randn(IntList size, Generator * generator) const {
    return  at::native::randn(*this, size, generator);

}
Tensor & Type::randn_out(Tensor & result, IntList size, Generator * generator) const {
    return  at::native::randn_out(result, size, generator);

}
Tensor Type::randn_like(const Tensor & self) const {
    return  at::native::randn_like(self);

}
Tensor Type::randn_like(const Tensor & self, const Type & dtype) const {
    return  at::native::randn_like(self, dtype);

}
Tensor Type::randperm(int64_t n, Generator * generator) const {
    return  at::native::randperm(*this, n, generator);

}
Tensor & Type::randperm_out(Tensor & result, int64_t n, Generator * generator) const {
    return  at::native::randperm_out(result, n, generator);

}
Tensor Type::range(Scalar start, Scalar end, Scalar step) const {
    return  at::native::range(*this, start, end, step);

}
Tensor & Type::range_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    return  at::native::range_out(result, start, end, step);

}
Tensor Type::repeat(const Tensor & self, IntList repeats) const {
    return  at::native::repeat(self, repeats);

}
Tensor Type::reshape(const Tensor & self, IntList shape) const {
    return  at::native::reshape(self, shape);

}
std::tuple<Tensor,Tensor> Type::RoiPooling2d_forward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale) const {
    AT_ERROR("RoiPooling2d_forward is not implemented for type %s", toString());
}
Tensor Type::RoiPooling2d_backward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale, const Tensor & gradOutput, const Tensor & argmaxes) const {
    AT_ERROR("RoiPooling2d_backward is not implemented for type %s", toString());
}
Tensor Type::round(const Tensor & self) const {
    return  at::native::round(self);

}
Tensor & Type::round_(Tensor & self) const {
    return  at::native::round_(self);

}
Tensor & Type::round_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("round_out is not implemented for type %s", toString());
}
Tensor Type::rrelu(const Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    return  at::native::rrelu(self, lower, upper, training, generator);

}
Tensor & Type::rrelu_(Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    return  at::native::rrelu_(self, lower, upper, training, generator);

}
Tensor Type::relu(const Tensor & self) const {
    return  at::native::relu(self);

}
Tensor & Type::relu_(Tensor & self) const {
    return  at::native::relu_(self);

}
Tensor Type::select(const Tensor & self, int64_t dim, int64_t index) const {
    return  at::native::select(self, dim, index);

}
Tensor Type::selu(const Tensor & self) const {
    return  at::native::selu(self);

}
Tensor & Type::selu_(Tensor & self) const {
    return  at::native::selu_(self);

}
Tensor Type::sin(const Tensor & self) const {
    return  at::native::sin(self);

}
Tensor & Type::sin_(Tensor & self) const {
    return  at::native::sin_(self);

}
Tensor & Type::sin_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("sin_out is not implemented for type %s", toString());
}
int64_t Type::size(const Tensor & self, int64_t dim) const {
    return  at::native::size(self, dim);

}
Tensor Type::slice(const Tensor & self, int64_t dim, int64_t start, int64_t end, int64_t step) const {
    return  at::native::slice(self, dim, start, end, step);

}
std::tuple<Tensor,Tensor> Type::slogdet(const Tensor & self) const {
    return  at::native::slogdet(self);

}
Tensor Type::smm(const Tensor & self, const Tensor & mat2) const {
    return  at::native::smm(self, mat2);

}
std::vector<Tensor> Type::split(const Tensor & self, int64_t split_size, int64_t dim) const {
    return  at::native::split(self, split_size, dim);

}
std::vector<Tensor> Type::split_with_sizes(const Tensor & self, IntList split_sizes, int64_t dim) const {
    return  at::native::split_with_sizes(self, split_sizes, dim);

}
Tensor Type::squeeze(const Tensor & self) const {
    return  at::native::squeeze(self);

}
Tensor Type::squeeze(const Tensor & self, int64_t dim) const {
    return  at::native::squeeze(self, dim);

}
Tensor & Type::squeeze_(Tensor & self) const {
    return  at::native::squeeze_(self);

}
Tensor & Type::squeeze_(Tensor & self, int64_t dim) const {
    return  at::native::squeeze_(self, dim);

}
Tensor Type::sspaddmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return  at::native::sspaddmm(self, mat1, mat2, beta, alpha);

}
Tensor & Type::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    AT_ERROR("sspaddmm_out is not implemented for type %s", toString());
}
Tensor Type::stack(TensorList tensors, int64_t dim) const {
    return  at::native::stack(tensors, dim);

}
Tensor & Type::stack_out(Tensor & result, TensorList tensors, int64_t dim) const {
    return  at::native::stack_out(result, tensors, dim);

}
Tensor Type::stft(const Tensor & self, int64_t frame_length, int64_t hop, int64_t fft_size, bool normalized, bool onesided, const Tensor & window, int64_t pad_end) const {
    return  at::native::stft(self, frame_length, hop, fft_size, normalized, onesided, window, pad_end);

}
int64_t Type::stride(const Tensor & self, int64_t dim) const {
    return  at::native::stride(self, dim);

}
Tensor Type::sum(const Tensor & self) const {
    AT_ERROR("sum is not implemented for type %s", toString());
}
Tensor Type::sum(const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::sum(self, dim, keepdim);

}
Tensor & Type::sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    AT_ERROR("sum_out is not implemented for type %s", toString());
}
Tensor Type::sqrt(const Tensor & self) const {
    return  at::native::sqrt(self);

}
Tensor & Type::sqrt_(Tensor & self) const {
    return  at::native::sqrt_(self);

}
Tensor & Type::sqrt_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("sqrt_out is not implemented for type %s", toString());
}
Tensor Type::prod(const Tensor & self) const {
    AT_ERROR("prod is not implemented for type %s", toString());
}
Tensor Type::prod(const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::prod(self, dim, keepdim);

}
Tensor & Type::prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    AT_ERROR("prod_out is not implemented for type %s", toString());
}
Tensor & Type::t_(Tensor & self) const {
    return  at::native::t_(self);

}
Tensor & Type::transpose_(Tensor & self, int64_t dim0, int64_t dim1) const {
    return  at::native::transpose_(self, dim0, dim1);

}
Tensor Type::triplet_margin_loss(const Tensor & anchor, const Tensor & positive, const Tensor & negative, double margin, double p, double eps, bool swap, bool size_average, bool reduce) const {
    return  at::native::triplet_margin_loss(anchor, positive, negative, margin, p, eps, swap, size_average, reduce);

}
Tensor Type::trunc(const Tensor & self) const {
    return  at::native::trunc(self);

}
Tensor & Type::trunc_(Tensor & self) const {
    return  at::native::trunc_(self);

}
Tensor & Type::trunc_out(Tensor & result, const Tensor & self) const {
    AT_ERROR("trunc_out is not implemented for type %s", toString());
}
Tensor Type::type_as(const Tensor & self, const Tensor & other) const {
    return  at::native::type_as(self, other);

}
std::tuple<Tensor,Tensor> Type::_unique(const Tensor & self, bool sorted, bool return_inverse) const {
    AT_ERROR("_unique is not implemented for type %s", toString());
}
Tensor Type::_unsafe_view(const Tensor & self, IntList size) const {
    return  at::native::_unsafe_view(self, size);

}
Tensor Type::unsqueeze(const Tensor & self, int64_t dim) const {
    return  at::native::unsqueeze(self, dim);

}
Tensor & Type::unsqueeze_(Tensor & self, int64_t dim) const {
    return  at::native::unsqueeze_(self, dim);

}
Tensor Type::view_as(const Tensor & self, const Tensor & other) const {
    return  at::native::view_as(self, other);

}
Tensor Type::where(const Tensor & condition, const Tensor & self, const Tensor & other) const {
    return  at::native::where(condition, self, other);

}
Tensor Type::_s_where(const Tensor & condition, const Tensor & self, const Tensor & other) const {
    AT_ERROR("_s_where is not implemented for type %s", toString());
}
Tensor Type::zeros(IntList size) const {
    return  at::native::zeros(*this, size);

}
Tensor & Type::zeros_out(Tensor & result, IntList size) const {
    return  at::native::zeros_out(result, size);

}
Tensor Type::zeros_like(const Tensor & self) const {
    return  at::native::zeros_like(self);

}
Tensor Type::zeros_like(const Tensor & self, const Type & dtype) const {
    return  at::native::zeros_like(self, dtype);

}
Tensor Type::_standard_gamma_grad(const Tensor & self, const Tensor & output) const {
    AT_ERROR("_standard_gamma_grad is not implemented for type %s", toString());
}
Tensor Type::poisson(const Tensor & self, Generator * generator) const {
    AT_ERROR("poisson is not implemented for type %s", toString());
}
Tensor Type::mkldnn_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList stride, IntList dilation) const {
    return  at::native::mkldnn_convolution(self, weight, bias, padding, stride, dilation);

}
Tensor Type::mkldnn_convolution_backward_input(IntList self_size, const Tensor & grad_output, const Tensor & weight, IntList padding, IntList stride, IntList dilation, bool bias_defined) const {
    return  at::native::mkldnn_convolution_backward_input(self_size, grad_output, weight, padding, stride, dilation, bias_defined);

}
std::tuple<Tensor,Tensor> Type::mkldnn_convolution_backward_weights(IntList weight_size, const Tensor & grad_output, const Tensor & self, IntList padding, IntList stride, IntList dilation, bool bias_defined) const {
    return  at::native::mkldnn_convolution_backward_weights(weight_size, grad_output, self, padding, stride, dilation, bias_defined);

}
std::tuple<Tensor,Tensor,Tensor> Type::mkldnn_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntList padding, IntList stride, IntList dilation, std::array<bool,3> output_mask) const {
    return  at::native::mkldnn_convolution_backward(self, grad_output, weight, padding, stride, dilation, output_mask);

}

}
