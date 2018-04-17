// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/SparseCPULongType.h"
#include "ATen/CPULongStorage.h"
#include "ATen/SparseCPULongTensor.h"
#include "ATen/CPUGenerator.h"
#include "ATen/SparseCPUByteTensor.h"
#include "ATen/SparseCPUIntTensor.h"
#include "ATen/SparseCPULongTensor.h"
#include "ATen/SparseCPULongTensor.h"
#include "ATen/CPULongTensor.h"
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

SparseCPULongType::SparseCPULongType(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
ScalarType SparseCPULongType::scalarType() const {
  return ScalarType::Long;
}
Backend SparseCPULongType::backend() const {
  return Backend::SparseCPU;
}
bool SparseCPULongType::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool SparseCPULongType::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool SparseCPULongType::is_distributed() const { return false; }

std::unique_ptr<Storage> SparseCPULongType::storage() const {
  return std::unique_ptr<Storage>(new CPULongStorage(context));
}
std::unique_ptr<Storage> SparseCPULongType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CPULongStorage(context,size));
}
std::unique_ptr<Storage> SparseCPULongType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CPULongStorage(context,data,size,deleter));
}
std::unique_ptr<Storage> SparseCPULongType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new CPULongStorage(context, size, std::move(allocator)));
}
Tensor SparseCPULongType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THSLongTensor_retain( (THSLongTensor*) th_pointer);
  return Tensor(new SparseCPULongTensor(context,(THSLongTensor*)(th_pointer)), false);
}
std::unique_ptr<Storage> SparseCPULongType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THLongStorage_retain( (THLongStorage*) th_pointer);
  return std::unique_ptr<Storage>(new CPULongStorage(context, (THLongStorage*) th_pointer));
}
std::unique_ptr<Generator> SparseCPULongType::generator() const {
  return std::unique_ptr<Generator>(new CPUGenerator(context));
}

const char * SparseCPULongType::toString() const {
  return SparseCPULongType::typeString();
}
TypeID SparseCPULongType::ID() const {
  return TypeID::SparseCPULong;
}

std::size_t SparseCPULongType::elementSizeInBytes() const {
  return sizeof(int64_t);
}

const char * SparseCPULongType::typeString() {
  return "SparseCPULongType";
}

/* example
Tensor * SparseCPULongType::add(Tensor & a, Tensor & b) {
  std::cout << "add SparseCPULongTensor\n";
  return &a;
}
*/

Tensor SparseCPULongType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    dim0 = maybe_wrap_dim(dim0, self_);
    dim1 = maybe_wrap_dim(dim1, self_);
    return Tensor((new SparseCPULongTensor(context, THSLongTensor_newTranspose(self_->tensor, dim0, dim1)))->maybeScalar(self_->isScalar()),false);
}
Tensor SparseCPULongType::t(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCPULongTensor(context, THSLongTensor_newTranspose(self_->tensor, 0, 1)))->maybeScalar(self_->isScalar()),false);
}
Tensor SparseCPULongType::clone(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCPULongTensor(context, THSLongTensor_newClone(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & SparseCPULongType::resize_as_(Tensor & self, const Tensor & the_template) const {
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    auto the_template_ = checked_cast_tensor<SparseCPULongTensor>(the_template.pImpl,"the_template",2, false);
    THSLongTensor_resizeAs(self_->tensor, the_template_->tensor);
    self_->maybeScalar(the_template_->isScalar());
    return self;
}
Tensor & SparseCPULongType::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    auto result_ = checked_cast_tensor<SparseCPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toLong();
    THSLongTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCPULongType::pow(const Tensor & self, Scalar exponent) const {
    auto result_ = new SparseCPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toLong();
    THSLongTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCPULongType::zero_(Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    THSLongTensor_zero(self_->tensor);
    return self;
}
Tensor & SparseCPULongType::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<SparseCPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toLong();
    auto other_ = checked_cast_tensor<SparseCPULongTensor>(other.pImpl,"other",3, false);
    THSLongTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCPULongType::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add(self, Scalar(other), alpha);
    }
    auto result_ = new SparseCPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toLong();
    auto other_ = checked_cast_tensor<SparseCPULongTensor>(other.pImpl,"other",3, false);
    THSLongTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCPULongType::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toLong();
    auto other_ = checked_cast_tensor<SparseCPULongTensor>(other.pImpl,"other",4, false);
    THSLongTensor_cadd(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & SparseCPULongType::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<SparseCPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toLong();
    auto other_ = checked_cast_tensor<SparseCPULongTensor>(other.pImpl,"other",3, false);
    THSLongTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCPULongType::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub(self, Scalar(other), alpha);
    }
    auto result_ = new SparseCPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toLong();
    auto other_ = checked_cast_tensor<SparseCPULongTensor>(other.pImpl,"other",3, false);
    THSLongTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCPULongType::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toLong();
    auto other_ = checked_cast_tensor<SparseCPULongTensor>(other.pImpl,"other",4, false);
    THSLongTensor_csub(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & SparseCPULongType::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<SparseCPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THSLongTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCPULongType::mul(const Tensor & self, Scalar other) const {
    auto result_ = new SparseCPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THSLongTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCPULongType::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<SparseCPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCPULongTensor>(other.pImpl,"other",2, false);
    THSLongTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCPULongType::s_mul(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul(self, Scalar(other));
    }
    auto result_ = new SparseCPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCPULongTensor>(other.pImpl,"other",2, false);
    THSLongTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCPULongType::mul_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THSLongTensor_mul(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & SparseCPULongType::s_mul_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCPULongTensor>(other.pImpl,"other",3, false);
    THSLongTensor_cmul(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & SparseCPULongType::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<SparseCPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THSLongTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCPULongType::div(const Tensor & self, Scalar other) const {
    auto result_ = new SparseCPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THSLongTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCPULongType::div_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THSLongTensor_div(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor SparseCPULongType::tensor(IntList size) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new SparseCPULongTensor(context, THSLongTensor_newWithSize(size_, NULL)))->maybeScalar(size.size() == 0),false);
}
Tensor SparseCPULongType::tensor() const {
    return Tensor((new SparseCPULongTensor(context, THSLongTensor_new())),false);
}
Tensor SparseCPULongType::sparse_coo_tensor(const Tensor & indices, const Tensor & values, IntList size) const {
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",1, false);
    auto values_ = checked_cast_tensor<CPULongTensor>(values.pImpl,"values",2, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new SparseCPULongTensor(context, THSLongTensor_newWithTensorAndSize(indices_->tensor, values_->tensor, size_)))->maybeScalar(size.size() == 0),false);
}
Tensor SparseCPULongType::sparse_coo_tensor(const Tensor & indices, const Tensor & values) const {
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",1, false);
    auto values_ = checked_cast_tensor<CPULongTensor>(values.pImpl,"values",2, false);
    return Tensor((new SparseCPULongTensor(context, THSLongTensor_newWithTensor(indices_->tensor, values_->tensor)))->maybeScalar(indices_->isScalar() && values_->isScalar()),false);
}
Tensor & SparseCPULongType::sparse_raw_resize_(Tensor & self, IntList size, int64_t nDimI, int64_t nDimV) const {
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
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
    THSLongTensor_rawResize(self_->tensor, nDimI, nDimV, (*size_).data);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor SparseCPULongType::to_dense(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPULongTensor(context, THSLongTensor_toDense(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
int64_t SparseCPULongType::_dimI(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    return THSLongTensor_nDimensionI(self_->tensor);
}
int64_t SparseCPULongType::_dimV(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    return THSLongTensor_nDimensionV(self_->tensor);
}
int64_t SparseCPULongType::_nnz(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    return THSLongTensor_nnz(self_->tensor);
}
Tensor SparseCPULongType::coalesce(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCPULongTensor(context, THSLongTensor_newCoalesce(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
bool SparseCPULongType::is_coalesced(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    return THSLongTensor_isCoalesced(self_->tensor);
}
Tensor SparseCPULongType::_indices(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      // Empty tensor
      return self_->type().toScalarType(kLong).tensor({0});
    }
    return Tensor((new CPULongTensor(context, THSLongTensor_newIndices(self_->tensor))),false);
}
Tensor SparseCPULongType::_values(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPULongTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPULongTensor(context, THSLongTensor_newValues(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & SparseCPULongType::hspmm_out(Tensor & result, const Tensor & mat1, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<SparseCPULongTensor>(result.pImpl,"result",0, false);
    auto mat1_ = checked_cast_tensor<SparseCPULongTensor>(mat1.pImpl,"mat1",1, false);
    auto mat2_ = checked_cast_tensor<CPULongTensor>(mat2.pImpl,"mat2",2, false);
    THSLongTensor_hspmm(result_->tensor, int64_t(1), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor SparseCPULongType::hspmm(const Tensor & mat1, const Tensor & mat2) const {
    auto result_ = new SparseCPULongTensor(context);
    auto result = Tensor(result_, false);
    auto mat1_ = checked_cast_tensor<SparseCPULongTensor>(mat1.pImpl,"mat1",1, false);
    auto mat2_ = checked_cast_tensor<CPULongTensor>(mat2.pImpl,"mat2",2, false);
    THSLongTensor_hspmm(result_->tensor, int64_t(1), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & SparseCPULongType::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return  at::native::_sspaddmm_out_cpu(result, self, mat1, mat2, beta, alpha);
}

}
