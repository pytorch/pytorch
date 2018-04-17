// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/SparseCUDAShortType.h"
#include "ATen/CUDAShortStorage.h"
#include "ATen/SparseCUDAShortTensor.h"
#include "ATen/CUDAGenerator.h"
#include "ATen/SparseCUDAByteTensor.h"
#include "ATen/SparseCUDAIntTensor.h"
#include "ATen/SparseCUDALongTensor.h"
#include "ATen/SparseCUDAShortTensor.h"
#include "ATen/CUDAShortTensor.h"
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

SparseCUDAShortType::SparseCUDAShortType(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
ScalarType SparseCUDAShortType::scalarType() const {
  return ScalarType::Short;
}
Backend SparseCUDAShortType::backend() const {
  return Backend::SparseCUDA;
}
bool SparseCUDAShortType::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool SparseCUDAShortType::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool SparseCUDAShortType::is_distributed() const { return false; }

std::unique_ptr<Storage> SparseCUDAShortType::storage() const {
  return std::unique_ptr<Storage>(new CUDAShortStorage(context));
}
std::unique_ptr<Storage> SparseCUDAShortType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CUDAShortStorage(context,size));
}
std::unique_ptr<Storage> SparseCUDAShortType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CUDAShortStorage(context,data,size,deleter));
}
std::unique_ptr<Storage> SparseCUDAShortType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new CUDAShortStorage(context, size, std::move(allocator)));
}
Tensor SparseCUDAShortType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THCSShortTensor_retain(context->thc_state,  (THCSShortTensor*) th_pointer);
  return Tensor(new SparseCUDAShortTensor(context,(THCSShortTensor*)(th_pointer)), false);
}
std::unique_ptr<Storage> SparseCUDAShortType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THCudaShortStorage_retain(context->thc_state,  (THCudaShortStorage*) th_pointer);
  return std::unique_ptr<Storage>(new CUDAShortStorage(context, (THCudaShortStorage*) th_pointer));
}
std::unique_ptr<Generator> SparseCUDAShortType::generator() const {
  return std::unique_ptr<Generator>(new CUDAGenerator(context));
}

const char * SparseCUDAShortType::toString() const {
  return SparseCUDAShortType::typeString();
}
TypeID SparseCUDAShortType::ID() const {
  return TypeID::SparseCUDAShort;
}

std::size_t SparseCUDAShortType::elementSizeInBytes() const {
  return sizeof(int16_t);
}

const char * SparseCUDAShortType::typeString() {
  return "SparseCUDAShortType";
}

/* example
Tensor * SparseCUDAShortType::add(Tensor & a, Tensor & b) {
  std::cout << "add SparseCUDAShortTensor\n";
  return &a;
}
*/

Tensor SparseCUDAShortType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    dim0 = maybe_wrap_dim(dim0, self_);
    dim1 = maybe_wrap_dim(dim1, self_);
    return Tensor((new SparseCUDAShortTensor(context, THCSShortTensor_newTranspose(context->thc_state, self_->tensor, dim0, dim1)))->maybeScalar(self_->isScalar()),false);
}
Tensor SparseCUDAShortType::t(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCUDAShortTensor(context, THCSShortTensor_newTranspose(context->thc_state, self_->tensor, 0, 1)))->maybeScalar(self_->isScalar()),false);
}
Tensor SparseCUDAShortType::clone(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCUDAShortTensor(context, THCSShortTensor_newClone(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & SparseCUDAShortType::resize_as_(Tensor & self, const Tensor & the_template) const {
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    auto the_template_ = checked_cast_tensor<SparseCUDAShortTensor>(the_template.pImpl,"the_template",2, false);
    THCSShortTensor_resizeAs(context->thc_state, self_->tensor, the_template_->tensor);
    self_->maybeScalar(the_template_->isScalar());
    return self;
}
int64_t SparseCUDAShortType::get_device(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THCSShortTensor_getDevice(context->thc_state, self_->tensor));
}
Tensor & SparseCUDAShortType::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    auto result_ = checked_cast_tensor<SparseCUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toShort();
    THCSShortTensor_pow(context->thc_state, result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCUDAShortType::pow(const Tensor & self, Scalar exponent) const {
    auto result_ = new SparseCUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toShort();
    THCSShortTensor_pow(context->thc_state, result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCUDAShortType::zero_(Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    THCSShortTensor_zero(context->thc_state, self_->tensor);
    return self;
}
Tensor & SparseCUDAShortType::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<SparseCUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<SparseCUDAShortTensor>(other.pImpl,"other",3, false);
    THCSShortTensor_cadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCUDAShortType::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add(self, Scalar(other), alpha);
    }
    auto result_ = new SparseCUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<SparseCUDAShortTensor>(other.pImpl,"other",3, false);
    THCSShortTensor_cadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCUDAShortType::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<SparseCUDAShortTensor>(other.pImpl,"other",4, false);
    THCSShortTensor_cadd(context->thc_state, self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & SparseCUDAShortType::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<SparseCUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<SparseCUDAShortTensor>(other.pImpl,"other",3, false);
    THCSShortTensor_csub(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCUDAShortType::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub(self, Scalar(other), alpha);
    }
    auto result_ = new SparseCUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<SparseCUDAShortTensor>(other.pImpl,"other",3, false);
    THCSShortTensor_csub(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCUDAShortType::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<SparseCUDAShortTensor>(other.pImpl,"other",4, false);
    THCSShortTensor_csub(context->thc_state, self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & SparseCUDAShortType::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<SparseCUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCSShortTensor_mul(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCUDAShortType::mul(const Tensor & self, Scalar other) const {
    auto result_ = new SparseCUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCSShortTensor_mul(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCUDAShortType::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<SparseCUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCUDAShortTensor>(other.pImpl,"other",2, false);
    THCSShortTensor_cmul(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCUDAShortType::s_mul(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul(self, Scalar(other));
    }
    auto result_ = new SparseCUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCUDAShortTensor>(other.pImpl,"other",2, false);
    THCSShortTensor_cmul(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCUDAShortType::mul_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCSShortTensor_mul(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & SparseCUDAShortType::s_mul_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCUDAShortTensor>(other.pImpl,"other",3, false);
    THCSShortTensor_cmul(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & SparseCUDAShortType::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<SparseCUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCSShortTensor_div(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCUDAShortType::div(const Tensor & self, Scalar other) const {
    auto result_ = new SparseCUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCSShortTensor_div(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCUDAShortType::div_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCSShortTensor_div(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor SparseCUDAShortType::tensor(IntList size) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new SparseCUDAShortTensor(context, THCSShortTensor_newWithSize(context->thc_state, size_, NULL)))->maybeScalar(size.size() == 0),false);
}
Tensor SparseCUDAShortType::tensor() const {
    return Tensor((new SparseCUDAShortTensor(context, THCSShortTensor_new(context->thc_state))),false);
}
Tensor SparseCUDAShortType::sparse_coo_tensor(const Tensor & indices, const Tensor & values, IntList size) const {
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",1, false);
    auto values_ = checked_cast_tensor<CUDAShortTensor>(values.pImpl,"values",2, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new SparseCUDAShortTensor(context, THCSShortTensor_newWithTensorAndSize(context->thc_state, indices_->tensor, values_->tensor, size_)))->maybeScalar(size.size() == 0),false);
}
Tensor SparseCUDAShortType::sparse_coo_tensor(const Tensor & indices, const Tensor & values) const {
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",1, false);
    auto values_ = checked_cast_tensor<CUDAShortTensor>(values.pImpl,"values",2, false);
    return Tensor((new SparseCUDAShortTensor(context, THCSShortTensor_newWithTensor(context->thc_state, indices_->tensor, values_->tensor)))->maybeScalar(indices_->isScalar() && values_->isScalar()),false);
}
Tensor & SparseCUDAShortType::sparse_raw_resize_(Tensor & self, IntList size, int64_t nDimI, int64_t nDimV) const {
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
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
    THCSShortTensor_rawResize(context->thc_state, self_->tensor, nDimI, nDimV, (*size_).data);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor SparseCUDAShortType::to_dense(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDAShortTensor(context, THCSShortTensor_toDense(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
int64_t SparseCUDAShortType::_dimI(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    return THCSShortTensor_nDimensionI(context->thc_state, self_->tensor);
}
int64_t SparseCUDAShortType::_dimV(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    return THCSShortTensor_nDimensionV(context->thc_state, self_->tensor);
}
int64_t SparseCUDAShortType::_nnz(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    return THCSShortTensor_nnz(context->thc_state, self_->tensor);
}
Tensor SparseCUDAShortType::coalesce(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCUDAShortTensor(context, THCSShortTensor_newCoalesce(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
bool SparseCUDAShortType::is_coalesced(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    return THCSShortTensor_isCoalesced(context->thc_state, self_->tensor);
}
Tensor SparseCUDAShortType::_indices(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      // Empty tensor
      return self_->type().toScalarType(kLong).tensor({0});
    }
    return Tensor((new CUDALongTensor(context, THCSShortTensor_newIndices(context->thc_state, self_->tensor))),false);
}
Tensor SparseCUDAShortType::_values(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAShortTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDAShortTensor(context, THCSShortTensor_newValues(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & SparseCUDAShortType::hspmm_out(Tensor & result, const Tensor & mat1, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<SparseCUDAShortTensor>(result.pImpl,"result",0, false);
    auto mat1_ = checked_cast_tensor<SparseCUDAShortTensor>(mat1.pImpl,"mat1",1, false);
    auto mat2_ = checked_cast_tensor<CUDAShortTensor>(mat2.pImpl,"mat2",2, false);
    THCSShortTensor_hspmm(context->thc_state, result_->tensor, int16_t(1), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor SparseCUDAShortType::hspmm(const Tensor & mat1, const Tensor & mat2) const {
    auto result_ = new SparseCUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto mat1_ = checked_cast_tensor<SparseCUDAShortTensor>(mat1.pImpl,"mat1",1, false);
    auto mat2_ = checked_cast_tensor<CUDAShortTensor>(mat2.pImpl,"mat2",2, false);
    THCSShortTensor_hspmm(context->thc_state, result_->tensor, int16_t(1), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & SparseCUDAShortType::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return  at::native::_sspaddmm_out_cuda(result, self, mat1, mat2, beta, alpha);
}

}
