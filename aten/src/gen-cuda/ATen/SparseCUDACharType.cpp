// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/SparseCUDACharType.h"
#include "ATen/CUDACharStorage.h"
#include "ATen/SparseCUDACharTensor.h"
#include "ATen/CUDAGenerator.h"
#include "ATen/SparseCUDAByteTensor.h"
#include "ATen/SparseCUDAIntTensor.h"
#include "ATen/SparseCUDALongTensor.h"
#include "ATen/SparseCUDACharTensor.h"
#include "ATen/CUDACharTensor.h"
#include "ATen/CUDALongTensor.h"
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
#include <ATen/cuda/CUDAHalf.cuh>
#endif

namespace at {

SparseCUDACharType::SparseCUDACharType(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
ScalarType SparseCUDACharType::scalarType() const {
  return ScalarType::Char;
}
Backend SparseCUDACharType::backend() const {
  return Backend::SparseCUDA;
}
bool SparseCUDACharType::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool SparseCUDACharType::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool SparseCUDACharType::is_distributed() const { return false; }

std::unique_ptr<Storage> SparseCUDACharType::storage() const {
  return std::unique_ptr<Storage>(new CUDACharStorage(context));
}
std::unique_ptr<Storage> SparseCUDACharType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CUDACharStorage(context,size));
}
std::unique_ptr<Storage> SparseCUDACharType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CUDACharStorage(context,data,size,deleter));
}
std::unique_ptr<Storage> SparseCUDACharType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new CUDACharStorage(context, size, std::move(allocator)));
}
Tensor SparseCUDACharType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THCSCharTensor_retain(context->thc_state,  (THCSCharTensor*) th_pointer);
  return Tensor(new SparseCUDACharTensor(context,(THCSCharTensor*)(th_pointer)), false);
}
std::unique_ptr<Storage> SparseCUDACharType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THCudaCharStorage_retain(context->thc_state,  (THCudaCharStorage*) th_pointer);
  return std::unique_ptr<Storage>(new CUDACharStorage(context, (THCudaCharStorage*) th_pointer));
}
std::unique_ptr<Generator> SparseCUDACharType::generator() const {
  return std::unique_ptr<Generator>(new CUDAGenerator(context));
}

const char * SparseCUDACharType::toString() const {
  return SparseCUDACharType::typeString();
}
TypeID SparseCUDACharType::ID() const {
  return TypeID::SparseCUDAChar;
}

std::size_t SparseCUDACharType::elementSizeInBytes() const {
  return sizeof(int8_t);
}

const char * SparseCUDACharType::typeString() {
  return "SparseCUDACharType";
}

/* example
Tensor * SparseCUDACharType::add(Tensor & a, Tensor & b) {
  std::cout << "add SparseCUDACharTensor\n";
  return &a;
}
*/

Tensor SparseCUDACharType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    dim0 = maybe_wrap_dim(dim0, self_);
    dim1 = maybe_wrap_dim(dim1, self_);
    return Tensor((new SparseCUDACharTensor(context, THCSCharTensor_newTranspose(context->thc_state, self_->tensor, dim0, dim1)))->maybeScalar(self_->isScalar()),false);
}
Tensor SparseCUDACharType::t(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCUDACharTensor(context, THCSCharTensor_newTranspose(context->thc_state, self_->tensor, 0, 1)))->maybeScalar(self_->isScalar()),false);
}
Tensor SparseCUDACharType::clone(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCUDACharTensor(context, THCSCharTensor_newClone(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & SparseCUDACharType::resize_as_(Tensor & self, const Tensor & the_template) const {
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    auto the_template_ = checked_cast_tensor<SparseCUDACharTensor>(the_template.pImpl,"the_template",2, false);
    THCSCharTensor_resizeAs(context->thc_state, self_->tensor, the_template_->tensor);
    self_->maybeScalar(the_template_->isScalar());
    return self;
}
int64_t SparseCUDACharType::get_device(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THCSCharTensor_getDevice(context->thc_state, self_->tensor));
}
Tensor & SparseCUDACharType::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    auto result_ = checked_cast_tensor<SparseCUDACharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toChar();
    THCSCharTensor_pow(context->thc_state, result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCUDACharType::pow(const Tensor & self, Scalar exponent) const {
    auto result_ = new SparseCUDACharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toChar();
    THCSCharTensor_pow(context->thc_state, result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCUDACharType::zero_(Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    THCSCharTensor_zero(context->thc_state, self_->tensor);
    return self;
}
Tensor & SparseCUDACharType::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<SparseCUDACharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toChar();
    auto other_ = checked_cast_tensor<SparseCUDACharTensor>(other.pImpl,"other",3, false);
    THCSCharTensor_cadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCUDACharType::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add(self, Scalar(other), alpha);
    }
    auto result_ = new SparseCUDACharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toChar();
    auto other_ = checked_cast_tensor<SparseCUDACharTensor>(other.pImpl,"other",3, false);
    THCSCharTensor_cadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCUDACharType::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toChar();
    auto other_ = checked_cast_tensor<SparseCUDACharTensor>(other.pImpl,"other",4, false);
    THCSCharTensor_cadd(context->thc_state, self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & SparseCUDACharType::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<SparseCUDACharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toChar();
    auto other_ = checked_cast_tensor<SparseCUDACharTensor>(other.pImpl,"other",3, false);
    THCSCharTensor_csub(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCUDACharType::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub(self, Scalar(other), alpha);
    }
    auto result_ = new SparseCUDACharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toChar();
    auto other_ = checked_cast_tensor<SparseCUDACharTensor>(other.pImpl,"other",3, false);
    THCSCharTensor_csub(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCUDACharType::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toChar();
    auto other_ = checked_cast_tensor<SparseCUDACharTensor>(other.pImpl,"other",4, false);
    THCSCharTensor_csub(context->thc_state, self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & SparseCUDACharType::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<SparseCUDACharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCSCharTensor_mul(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCUDACharType::mul(const Tensor & self, Scalar other) const {
    auto result_ = new SparseCUDACharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCSCharTensor_mul(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCUDACharType::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<SparseCUDACharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCUDACharTensor>(other.pImpl,"other",2, false);
    THCSCharTensor_cmul(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCUDACharType::s_mul(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul(self, Scalar(other));
    }
    auto result_ = new SparseCUDACharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCUDACharTensor>(other.pImpl,"other",2, false);
    THCSCharTensor_cmul(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCUDACharType::mul_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCSCharTensor_mul(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & SparseCUDACharType::s_mul_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCUDACharTensor>(other.pImpl,"other",3, false);
    THCSCharTensor_cmul(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & SparseCUDACharType::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<SparseCUDACharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCSCharTensor_div(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCUDACharType::div(const Tensor & self, Scalar other) const {
    auto result_ = new SparseCUDACharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCSCharTensor_div(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCUDACharType::div_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCSCharTensor_div(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor SparseCUDACharType::tensor(IntList size) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new SparseCUDACharTensor(context, THCSCharTensor_newWithSize(context->thc_state, size_, NULL)))->maybeScalar(size.size() == 0),false);
}
Tensor SparseCUDACharType::tensor() const {
    return Tensor((new SparseCUDACharTensor(context, THCSCharTensor_new(context->thc_state))),false);
}
Tensor SparseCUDACharType::sparse_coo_tensor(const Tensor & indices, const Tensor & values, IntList size) const {
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",1, false);
    auto values_ = checked_cast_tensor<CUDACharTensor>(values.pImpl,"values",2, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new SparseCUDACharTensor(context, THCSCharTensor_newWithTensorAndSize(context->thc_state, indices_->tensor, values_->tensor, size_)))->maybeScalar(size.size() == 0),false);
}
Tensor SparseCUDACharType::sparse_coo_tensor(const Tensor & indices, const Tensor & values) const {
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",1, false);
    auto values_ = checked_cast_tensor<CUDACharTensor>(values.pImpl,"values",2, false);
    return Tensor((new SparseCUDACharTensor(context, THCSCharTensor_newWithTensor(context->thc_state, indices_->tensor, values_->tensor)))->maybeScalar(indices_->isScalar() && values_->isScalar()),false);
}
Tensor & SparseCUDACharType::sparse_raw_resize_(Tensor & self, IntList size, int64_t nDimI, int64_t nDimV) const {
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
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
    THCSCharTensor_rawResize(context->thc_state, self_->tensor, nDimI, nDimV, (*size_).data);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor SparseCUDACharType::to_dense(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDACharTensor(context, THCSCharTensor_toDense(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
int64_t SparseCUDACharType::_dimI(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    return THCSCharTensor_nDimensionI(context->thc_state, self_->tensor);
}
int64_t SparseCUDACharType::_dimV(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    return THCSCharTensor_nDimensionV(context->thc_state, self_->tensor);
}
int64_t SparseCUDACharType::_nnz(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    return THCSCharTensor_nnz(context->thc_state, self_->tensor);
}
Tensor SparseCUDACharType::coalesce(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCUDACharTensor(context, THCSCharTensor_newCoalesce(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
bool SparseCUDACharType::is_coalesced(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    return THCSCharTensor_isCoalesced(context->thc_state, self_->tensor);
}
Tensor SparseCUDACharType::_indices(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      // Empty tensor
      return self_->type().toScalarType(kLong).tensor({0});
    }
    return Tensor((new CUDALongTensor(context, THCSCharTensor_newIndices(context->thc_state, self_->tensor))),false);
}
Tensor SparseCUDACharType::_values(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDACharTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDACharTensor(context, THCSCharTensor_newValues(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & SparseCUDACharType::hspmm_out(Tensor & result, const Tensor & mat1, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<SparseCUDACharTensor>(result.pImpl,"result",0, false);
    auto mat1_ = checked_cast_tensor<SparseCUDACharTensor>(mat1.pImpl,"mat1",1, false);
    auto mat2_ = checked_cast_tensor<CUDACharTensor>(mat2.pImpl,"mat2",2, false);
    THCSCharTensor_hspmm(context->thc_state, result_->tensor, int8_t(1), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor SparseCUDACharType::hspmm(const Tensor & mat1, const Tensor & mat2) const {
    auto result_ = new SparseCUDACharTensor(context);
    auto result = Tensor(result_, false);
    auto mat1_ = checked_cast_tensor<SparseCUDACharTensor>(mat1.pImpl,"mat1",1, false);
    auto mat2_ = checked_cast_tensor<CUDACharTensor>(mat2.pImpl,"mat2",2, false);
    THCSCharTensor_hspmm(context->thc_state, result_->tensor, int8_t(1), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & SparseCUDACharType::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return  at::native::_sspaddmm_out_cuda(result, self, mat1, mat2, beta, alpha);
}

}
