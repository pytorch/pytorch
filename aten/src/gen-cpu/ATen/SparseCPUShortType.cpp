// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/SparseCPUShortType.h"
#include "ATen/CPUShortStorage.h"
#include "ATen/SparseCPUShortTensor.h"
#include "ATen/CPUGenerator.h"
#include "ATen/SparseCPUByteTensor.h"
#include "ATen/SparseCPUIntTensor.h"
#include "ATen/SparseCPULongTensor.h"
#include "ATen/SparseCPUShortTensor.h"
#include "ATen/CPUShortTensor.h"
#include "ATen/CPULongTensor.h"
#include "ATen/Allocator.h"
#include "ATen/Utils.h"
#include "ATen/Half.h"
#include "ATen/WrapDimUtils.h"
#include "ATen/THLongStorageView.h"
#include "ATen/UndefinedTensor.h"
#include "ATen/NativeFunctions.h"
#include <iostream>
#include <sstream>

#include "ATen/Config.h"
#if AT_CUDA_ENABLED()

#endif

namespace at {

SparseCPUShortType::SparseCPUShortType(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
ScalarType SparseCPUShortType::scalarType() const {
  return ScalarType::Short;
}
Backend SparseCPUShortType::backend() const {
  return Backend::SparseCPU;
}
bool SparseCPUShortType::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool SparseCPUShortType::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool SparseCPUShortType::is_distributed() const { return false; }

std::unique_ptr<Storage> SparseCPUShortType::storage() const {
  return std::unique_ptr<Storage>(new CPUShortStorage(context));
}
std::unique_ptr<Storage> SparseCPUShortType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CPUShortStorage(context,size));
}
std::unique_ptr<Storage> SparseCPUShortType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CPUShortStorage(context,data,size,deleter));
}
std::unique_ptr<Storage> SparseCPUShortType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new CPUShortStorage(context, size, std::move(allocator)));
}
Tensor SparseCPUShortType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THSShortTensor_retain( (THSShortTensor*) th_pointer);
  return Tensor(new SparseCPUShortTensor(context,(THSShortTensor*)(th_pointer)), false);
}
std::unique_ptr<Storage> SparseCPUShortType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THShortStorage_retain( (THShortStorage*) th_pointer);
  return std::unique_ptr<Storage>(new CPUShortStorage(context, (THShortStorage*) th_pointer));
}
std::unique_ptr<Generator> SparseCPUShortType::generator() const {
  return std::unique_ptr<Generator>(new CPUGenerator(context));
}

const char * SparseCPUShortType::toString() const {
  return SparseCPUShortType::typeString();
}
TypeID SparseCPUShortType::ID() const {
  return TypeID::SparseCPUShort;
}

std::size_t SparseCPUShortType::elementSizeInBytes() const {
  return sizeof(int16_t);
}

const char * SparseCPUShortType::typeString() {
  return "SparseCPUShortType";
}

/* example
Tensor * SparseCPUShortType::add(Tensor & a, Tensor & b) {
  std::cout << "add SparseCPUShortTensor\n";
  return &a;
}
*/

Tensor SparseCPUShortType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    dim0 = maybe_wrap_dim(dim0, self_);
    dim1 = maybe_wrap_dim(dim1, self_);
    return Tensor((new SparseCPUShortTensor(context, THSShortTensor_newTranspose(self_->tensor, dim0, dim1)))->maybeScalar(self_->isScalar()),false);
}
Tensor SparseCPUShortType::t(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCPUShortTensor(context, THSShortTensor_newTranspose(self_->tensor, 0, 1)))->maybeScalar(self_->isScalar()),false);
}
Tensor SparseCPUShortType::clone(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCPUShortTensor(context, THSShortTensor_newClone(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & SparseCPUShortType::resize_as_(Tensor & self, const Tensor & the_template) const {
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    auto the_template_ = checked_cast_tensor<SparseCPUShortTensor>(the_template.pImpl,"the_template",2, false);
    THSShortTensor_resizeAs(self_->tensor, the_template_->tensor);
    self_->maybeScalar(the_template_->isScalar());
    return self;
}
Tensor & SparseCPUShortType::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    auto result_ = checked_cast_tensor<SparseCPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toShort();
    THSShortTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCPUShortType::pow(const Tensor & self, Scalar exponent) const {
    auto result_ = new SparseCPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toShort();
    THSShortTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCPUShortType::zero_(Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    THSShortTensor_zero(self_->tensor);
    return self;
}
Tensor & SparseCPUShortType::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<SparseCPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<SparseCPUShortTensor>(other.pImpl,"other",3, false);
    THSShortTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCPUShortType::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add(self, Scalar(other), alpha);
    }
    auto result_ = new SparseCPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<SparseCPUShortTensor>(other.pImpl,"other",3, false);
    THSShortTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCPUShortType::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<SparseCPUShortTensor>(other.pImpl,"other",4, false);
    THSShortTensor_cadd(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & SparseCPUShortType::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<SparseCPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<SparseCPUShortTensor>(other.pImpl,"other",3, false);
    THSShortTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCPUShortType::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub(self, Scalar(other), alpha);
    }
    auto result_ = new SparseCPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<SparseCPUShortTensor>(other.pImpl,"other",3, false);
    THSShortTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCPUShortType::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<SparseCPUShortTensor>(other.pImpl,"other",4, false);
    THSShortTensor_csub(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & SparseCPUShortType::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<SparseCPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THSShortTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCPUShortType::mul(const Tensor & self, Scalar other) const {
    auto result_ = new SparseCPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THSShortTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCPUShortType::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<SparseCPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCPUShortTensor>(other.pImpl,"other",2, false);
    THSShortTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCPUShortType::s_mul(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul(self, Scalar(other));
    }
    auto result_ = new SparseCPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCPUShortTensor>(other.pImpl,"other",2, false);
    THSShortTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCPUShortType::mul_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THSShortTensor_mul(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & SparseCPUShortType::s_mul_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCPUShortTensor>(other.pImpl,"other",3, false);
    THSShortTensor_cmul(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & SparseCPUShortType::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<SparseCPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THSShortTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCPUShortType::div(const Tensor & self, Scalar other) const {
    auto result_ = new SparseCPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THSShortTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCPUShortType::div_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THSShortTensor_div(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor SparseCPUShortType::tensor(IntList size) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new SparseCPUShortTensor(context, THSShortTensor_newWithSize(size_, NULL)))->maybeScalar(size.size() == 0),false);
}
Tensor SparseCPUShortType::tensor() const {
    return Tensor((new SparseCPUShortTensor(context, THSShortTensor_new())),false);
}
Tensor SparseCPUShortType::sparse_coo_tensor(const Tensor & indices, const Tensor & values, IntList size) const {
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",1, false);
    auto values_ = checked_cast_tensor<CPUShortTensor>(values.pImpl,"values",2, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new SparseCPUShortTensor(context, THSShortTensor_newWithTensorAndSize(indices_->tensor, values_->tensor, size_)))->maybeScalar(size.size() == 0),false);
}
Tensor SparseCPUShortType::sparse_coo_tensor(const Tensor & indices, const Tensor & values) const {
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",1, false);
    auto values_ = checked_cast_tensor<CPUShortTensor>(values.pImpl,"values",2, false);
    return Tensor((new SparseCPUShortTensor(context, THSShortTensor_newWithTensor(indices_->tensor, values_->tensor)))->maybeScalar(indices_->isScalar() && values_->isScalar()),false);
}
Tensor & SparseCPUShortType::sparse_raw_resize_(Tensor & self, IntList size, int64_t nDimI, int64_t nDimV) const {
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    /*
     * Sets sizes, nDimI, and nDimV of a sparse tensor directly without any
     * safety checks. If nDimI and/or nDimV are -1, recompute them from
     * indices and values, respectively.
     */
    if (nDimI == -1) {
      nDimI = self._indices().size(0);
    }
    if (nDimV == -1) {
      nDimV = self._values().dim() - 1;
    }
    THSShortTensor_rawResize(self_->tensor, nDimI, nDimV, (*size_).data);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor SparseCPUShortType::to_dense(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUShortTensor(context, THSShortTensor_toDense(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
int64_t SparseCPUShortType::_dimI(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    return THSShortTensor_nDimensionI(self_->tensor);
}
int64_t SparseCPUShortType::_dimV(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    return THSShortTensor_nDimensionV(self_->tensor);
}
int64_t SparseCPUShortType::_nnz(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    return THSShortTensor_nnz(self_->tensor);
}
Tensor SparseCPUShortType::coalesce(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCPUShortTensor(context, THSShortTensor_newCoalesce(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
bool SparseCPUShortType::is_coalesced(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    return THSShortTensor_isCoalesced(self_->tensor);
}
Tensor SparseCPUShortType::_indices(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      // Empty tensor
      return self_->type().toScalarType(kLong).tensor({0});
    }
    return Tensor((new CPULongTensor(context, THSShortTensor_newIndices(self_->tensor))),false);
}
Tensor SparseCPUShortType::_values(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUShortTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUShortTensor(context, THSShortTensor_newValues(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & SparseCPUShortType::hspmm_out(Tensor & result, const Tensor & mat1, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<SparseCPUShortTensor>(result.pImpl,"result",0, false);
    auto mat1_ = checked_cast_tensor<SparseCPUShortTensor>(mat1.pImpl,"mat1",1, false);
    auto mat2_ = checked_cast_tensor<CPUShortTensor>(mat2.pImpl,"mat2",2, false);
    THSShortTensor_hspmm(result_->tensor, int16_t(1), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor SparseCPUShortType::hspmm(const Tensor & mat1, const Tensor & mat2) const {
    auto result_ = new SparseCPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto mat1_ = checked_cast_tensor<SparseCPUShortTensor>(mat1.pImpl,"mat1",1, false);
    auto mat2_ = checked_cast_tensor<CPUShortTensor>(mat2.pImpl,"mat2",2, false);
    THSShortTensor_hspmm(result_->tensor, int16_t(1), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & SparseCPUShortType::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return  at::native::_sspaddmm_out_cpu(result, self, mat1, mat2, beta, alpha);
}

}
