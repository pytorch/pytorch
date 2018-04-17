// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/SparseCPUCharType.h"
#include "ATen/CPUCharStorage.h"
#include "ATen/SparseCPUCharTensor.h"
#include "ATen/CPUGenerator.h"
#include "ATen/SparseCPUByteTensor.h"
#include "ATen/SparseCPUIntTensor.h"
#include "ATen/SparseCPULongTensor.h"
#include "ATen/SparseCPUCharTensor.h"
#include "ATen/CPUCharTensor.h"
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

SparseCPUCharType::SparseCPUCharType(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
ScalarType SparseCPUCharType::scalarType() const {
  return ScalarType::Char;
}
Backend SparseCPUCharType::backend() const {
  return Backend::SparseCPU;
}
bool SparseCPUCharType::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool SparseCPUCharType::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool SparseCPUCharType::is_distributed() const { return false; }

std::unique_ptr<Storage> SparseCPUCharType::storage() const {
  return std::unique_ptr<Storage>(new CPUCharStorage(context));
}
std::unique_ptr<Storage> SparseCPUCharType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CPUCharStorage(context,size));
}
std::unique_ptr<Storage> SparseCPUCharType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CPUCharStorage(context,data,size,deleter));
}
std::unique_ptr<Storage> SparseCPUCharType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new CPUCharStorage(context, size, std::move(allocator)));
}
Tensor SparseCPUCharType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THSCharTensor_retain( (THSCharTensor*) th_pointer);
  return Tensor(new SparseCPUCharTensor(context,(THSCharTensor*)(th_pointer)), false);
}
std::unique_ptr<Storage> SparseCPUCharType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THCharStorage_retain( (THCharStorage*) th_pointer);
  return std::unique_ptr<Storage>(new CPUCharStorage(context, (THCharStorage*) th_pointer));
}
std::unique_ptr<Generator> SparseCPUCharType::generator() const {
  return std::unique_ptr<Generator>(new CPUGenerator(context));
}

const char * SparseCPUCharType::toString() const {
  return SparseCPUCharType::typeString();
}
TypeID SparseCPUCharType::ID() const {
  return TypeID::SparseCPUChar;
}

std::size_t SparseCPUCharType::elementSizeInBytes() const {
  return sizeof(int8_t);
}

const char * SparseCPUCharType::typeString() {
  return "SparseCPUCharType";
}

/* example
Tensor * SparseCPUCharType::add(Tensor & a, Tensor & b) {
  std::cout << "add SparseCPUCharTensor\n";
  return &a;
}
*/

Tensor SparseCPUCharType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    dim0 = maybe_wrap_dim(dim0, self_);
    dim1 = maybe_wrap_dim(dim1, self_);
    return Tensor((new SparseCPUCharTensor(context, THSCharTensor_newTranspose(self_->tensor, dim0, dim1)))->maybeScalar(self_->isScalar()),false);
}
Tensor SparseCPUCharType::t(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCPUCharTensor(context, THSCharTensor_newTranspose(self_->tensor, 0, 1)))->maybeScalar(self_->isScalar()),false);
}
Tensor SparseCPUCharType::clone(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCPUCharTensor(context, THSCharTensor_newClone(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & SparseCPUCharType::resize_as_(Tensor & self, const Tensor & the_template) const {
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    auto the_template_ = checked_cast_tensor<SparseCPUCharTensor>(the_template.pImpl,"the_template",2, false);
    THSCharTensor_resizeAs(self_->tensor, the_template_->tensor);
    self_->maybeScalar(the_template_->isScalar());
    return self;
}
Tensor & SparseCPUCharType::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    auto result_ = checked_cast_tensor<SparseCPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toChar();
    THSCharTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCPUCharType::pow(const Tensor & self, Scalar exponent) const {
    auto result_ = new SparseCPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toChar();
    THSCharTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCPUCharType::zero_(Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    THSCharTensor_zero(self_->tensor);
    return self;
}
Tensor & SparseCPUCharType::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<SparseCPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toChar();
    auto other_ = checked_cast_tensor<SparseCPUCharTensor>(other.pImpl,"other",3, false);
    THSCharTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCPUCharType::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add(self, Scalar(other), alpha);
    }
    auto result_ = new SparseCPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toChar();
    auto other_ = checked_cast_tensor<SparseCPUCharTensor>(other.pImpl,"other",3, false);
    THSCharTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCPUCharType::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toChar();
    auto other_ = checked_cast_tensor<SparseCPUCharTensor>(other.pImpl,"other",4, false);
    THSCharTensor_cadd(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & SparseCPUCharType::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<SparseCPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toChar();
    auto other_ = checked_cast_tensor<SparseCPUCharTensor>(other.pImpl,"other",3, false);
    THSCharTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCPUCharType::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub(self, Scalar(other), alpha);
    }
    auto result_ = new SparseCPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toChar();
    auto other_ = checked_cast_tensor<SparseCPUCharTensor>(other.pImpl,"other",3, false);
    THSCharTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCPUCharType::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toChar();
    auto other_ = checked_cast_tensor<SparseCPUCharTensor>(other.pImpl,"other",4, false);
    THSCharTensor_csub(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & SparseCPUCharType::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<SparseCPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THSCharTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCPUCharType::mul(const Tensor & self, Scalar other) const {
    auto result_ = new SparseCPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THSCharTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCPUCharType::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<SparseCPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCPUCharTensor>(other.pImpl,"other",2, false);
    THSCharTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCPUCharType::s_mul(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul(self, Scalar(other));
    }
    auto result_ = new SparseCPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCPUCharTensor>(other.pImpl,"other",2, false);
    THSCharTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCPUCharType::mul_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THSCharTensor_mul(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & SparseCPUCharType::s_mul_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCPUCharTensor>(other.pImpl,"other",3, false);
    THSCharTensor_cmul(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & SparseCPUCharType::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<SparseCPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THSCharTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCPUCharType::div(const Tensor & self, Scalar other) const {
    auto result_ = new SparseCPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THSCharTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCPUCharType::div_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THSCharTensor_div(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor SparseCPUCharType::tensor(IntList size) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new SparseCPUCharTensor(context, THSCharTensor_newWithSize(size_, NULL)))->maybeScalar(size.size() == 0),false);
}
Tensor SparseCPUCharType::tensor() const {
    return Tensor((new SparseCPUCharTensor(context, THSCharTensor_new())),false);
}
Tensor SparseCPUCharType::sparse_coo_tensor(const Tensor & indices, const Tensor & values, IntList size) const {
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",1, false);
    auto values_ = checked_cast_tensor<CPUCharTensor>(values.pImpl,"values",2, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new SparseCPUCharTensor(context, THSCharTensor_newWithTensorAndSize(indices_->tensor, values_->tensor, size_)))->maybeScalar(size.size() == 0),false);
}
Tensor SparseCPUCharType::sparse_coo_tensor(const Tensor & indices, const Tensor & values) const {
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",1, false);
    auto values_ = checked_cast_tensor<CPUCharTensor>(values.pImpl,"values",2, false);
    return Tensor((new SparseCPUCharTensor(context, THSCharTensor_newWithTensor(indices_->tensor, values_->tensor)))->maybeScalar(indices_->isScalar() && values_->isScalar()),false);
}
Tensor & SparseCPUCharType::sparse_raw_resize_(Tensor & self, IntList size, int64_t nDimI, int64_t nDimV) const {
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
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
    THSCharTensor_rawResize(self_->tensor, nDimI, nDimV, (*size_).data);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor SparseCPUCharType::to_dense(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUCharTensor(context, THSCharTensor_toDense(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
int64_t SparseCPUCharType::_dimI(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    return THSCharTensor_nDimensionI(self_->tensor);
}
int64_t SparseCPUCharType::_dimV(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    return THSCharTensor_nDimensionV(self_->tensor);
}
int64_t SparseCPUCharType::_nnz(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    return THSCharTensor_nnz(self_->tensor);
}
Tensor SparseCPUCharType::coalesce(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCPUCharTensor(context, THSCharTensor_newCoalesce(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
bool SparseCPUCharType::is_coalesced(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    return THSCharTensor_isCoalesced(self_->tensor);
}
Tensor SparseCPUCharType::_indices(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      // Empty tensor
      return self_->type().toScalarType(kLong).tensor({0});
    }
    return Tensor((new CPULongTensor(context, THSCharTensor_newIndices(self_->tensor))),false);
}
Tensor SparseCPUCharType::_values(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUCharTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUCharTensor(context, THSCharTensor_newValues(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & SparseCPUCharType::hspmm_out(Tensor & result, const Tensor & mat1, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<SparseCPUCharTensor>(result.pImpl,"result",0, false);
    auto mat1_ = checked_cast_tensor<SparseCPUCharTensor>(mat1.pImpl,"mat1",1, false);
    auto mat2_ = checked_cast_tensor<CPUCharTensor>(mat2.pImpl,"mat2",2, false);
    THSCharTensor_hspmm(result_->tensor, int8_t(1), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor SparseCPUCharType::hspmm(const Tensor & mat1, const Tensor & mat2) const {
    auto result_ = new SparseCPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto mat1_ = checked_cast_tensor<SparseCPUCharTensor>(mat1.pImpl,"mat1",1, false);
    auto mat2_ = checked_cast_tensor<CPUCharTensor>(mat2.pImpl,"mat2",2, false);
    THSCharTensor_hspmm(result_->tensor, int8_t(1), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & SparseCPUCharType::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return  at::native::_sspaddmm_out_cpu(result, self, mat1, mat2, beta, alpha);
}

}
