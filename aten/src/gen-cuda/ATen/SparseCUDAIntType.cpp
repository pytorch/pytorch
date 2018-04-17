// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/SparseCUDAIntType.h"
#include "ATen/CUDAIntStorage.h"
#include "ATen/SparseCUDAIntTensor.h"
#include "ATen/CUDAGenerator.h"
#include "ATen/SparseCUDAByteTensor.h"
#include "ATen/SparseCUDAIntTensor.h"
#include "ATen/SparseCUDALongTensor.h"
#include "ATen/SparseCUDAIntTensor.h"
#include "ATen/CUDAIntTensor.h"
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

SparseCUDAIntType::SparseCUDAIntType(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
ScalarType SparseCUDAIntType::scalarType() const {
  return ScalarType::Int;
}
Backend SparseCUDAIntType::backend() const {
  return Backend::SparseCUDA;
}
bool SparseCUDAIntType::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool SparseCUDAIntType::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool SparseCUDAIntType::is_distributed() const { return false; }

std::unique_ptr<Storage> SparseCUDAIntType::storage() const {
  return std::unique_ptr<Storage>(new CUDAIntStorage(context));
}
std::unique_ptr<Storage> SparseCUDAIntType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CUDAIntStorage(context,size));
}
std::unique_ptr<Storage> SparseCUDAIntType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CUDAIntStorage(context,data,size,deleter));
}
std::unique_ptr<Storage> SparseCUDAIntType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new CUDAIntStorage(context, size, std::move(allocator)));
}
Tensor SparseCUDAIntType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THCSIntTensor_retain(context->thc_state,  (THCSIntTensor*) th_pointer);
  return Tensor(new SparseCUDAIntTensor(context,(THCSIntTensor*)(th_pointer)), false);
}
std::unique_ptr<Storage> SparseCUDAIntType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THCudaIntStorage_retain(context->thc_state,  (THCudaIntStorage*) th_pointer);
  return std::unique_ptr<Storage>(new CUDAIntStorage(context, (THCudaIntStorage*) th_pointer));
}
std::unique_ptr<Generator> SparseCUDAIntType::generator() const {
  return std::unique_ptr<Generator>(new CUDAGenerator(context));
}

const char * SparseCUDAIntType::toString() const {
  return SparseCUDAIntType::typeString();
}
TypeID SparseCUDAIntType::ID() const {
  return TypeID::SparseCUDAInt;
}

std::size_t SparseCUDAIntType::elementSizeInBytes() const {
  return sizeof(int);
}

const char * SparseCUDAIntType::typeString() {
  return "SparseCUDAIntType";
}

/* example
Tensor * SparseCUDAIntType::add(Tensor & a, Tensor & b) {
  std::cout << "add SparseCUDAIntTensor\n";
  return &a;
}
*/

Tensor SparseCUDAIntType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    dim0 = maybe_wrap_dim(dim0, self_);
    dim1 = maybe_wrap_dim(dim1, self_);
    return Tensor((new SparseCUDAIntTensor(context, THCSIntTensor_newTranspose(context->thc_state, self_->tensor, dim0, dim1)))->maybeScalar(self_->isScalar()),false);
}
Tensor SparseCUDAIntType::t(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCUDAIntTensor(context, THCSIntTensor_newTranspose(context->thc_state, self_->tensor, 0, 1)))->maybeScalar(self_->isScalar()),false);
}
Tensor SparseCUDAIntType::clone(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCUDAIntTensor(context, THCSIntTensor_newClone(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & SparseCUDAIntType::resize_as_(Tensor & self, const Tensor & the_template) const {
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    auto the_template_ = checked_cast_tensor<SparseCUDAIntTensor>(the_template.pImpl,"the_template",2, false);
    THCSIntTensor_resizeAs(context->thc_state, self_->tensor, the_template_->tensor);
    self_->maybeScalar(the_template_->isScalar());
    return self;
}
int64_t SparseCUDAIntType::get_device(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THCSIntTensor_getDevice(context->thc_state, self_->tensor));
}
Tensor & SparseCUDAIntType::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    auto result_ = checked_cast_tensor<SparseCUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toInt();
    THCSIntTensor_pow(context->thc_state, result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCUDAIntType::pow(const Tensor & self, Scalar exponent) const {
    auto result_ = new SparseCUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toInt();
    THCSIntTensor_pow(context->thc_state, result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCUDAIntType::zero_(Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    THCSIntTensor_zero(context->thc_state, self_->tensor);
    return self;
}
Tensor & SparseCUDAIntType::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<SparseCUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<SparseCUDAIntTensor>(other.pImpl,"other",3, false);
    THCSIntTensor_cadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCUDAIntType::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add(self, Scalar(other), alpha);
    }
    auto result_ = new SparseCUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<SparseCUDAIntTensor>(other.pImpl,"other",3, false);
    THCSIntTensor_cadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCUDAIntType::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<SparseCUDAIntTensor>(other.pImpl,"other",4, false);
    THCSIntTensor_cadd(context->thc_state, self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & SparseCUDAIntType::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<SparseCUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<SparseCUDAIntTensor>(other.pImpl,"other",3, false);
    THCSIntTensor_csub(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCUDAIntType::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub(self, Scalar(other), alpha);
    }
    auto result_ = new SparseCUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<SparseCUDAIntTensor>(other.pImpl,"other",3, false);
    THCSIntTensor_csub(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCUDAIntType::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<SparseCUDAIntTensor>(other.pImpl,"other",4, false);
    THCSIntTensor_csub(context->thc_state, self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & SparseCUDAIntType::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<SparseCUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCSIntTensor_mul(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCUDAIntType::mul(const Tensor & self, Scalar other) const {
    auto result_ = new SparseCUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCSIntTensor_mul(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCUDAIntType::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<SparseCUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCUDAIntTensor>(other.pImpl,"other",2, false);
    THCSIntTensor_cmul(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCUDAIntType::s_mul(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul(self, Scalar(other));
    }
    auto result_ = new SparseCUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCUDAIntTensor>(other.pImpl,"other",2, false);
    THCSIntTensor_cmul(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCUDAIntType::mul_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCSIntTensor_mul(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & SparseCUDAIntType::s_mul_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCUDAIntTensor>(other.pImpl,"other",3, false);
    THCSIntTensor_cmul(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & SparseCUDAIntType::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<SparseCUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCSIntTensor_div(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCUDAIntType::div(const Tensor & self, Scalar other) const {
    auto result_ = new SparseCUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCSIntTensor_div(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCUDAIntType::div_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCSIntTensor_div(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor SparseCUDAIntType::tensor(IntList size) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new SparseCUDAIntTensor(context, THCSIntTensor_newWithSize(context->thc_state, size_, NULL)))->maybeScalar(size.size() == 0),false);
}
Tensor SparseCUDAIntType::tensor() const {
    return Tensor((new SparseCUDAIntTensor(context, THCSIntTensor_new(context->thc_state))),false);
}
Tensor SparseCUDAIntType::sparse_coo_tensor(const Tensor & indices, const Tensor & values, IntList size) const {
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",1, false);
    auto values_ = checked_cast_tensor<CUDAIntTensor>(values.pImpl,"values",2, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new SparseCUDAIntTensor(context, THCSIntTensor_newWithTensorAndSize(context->thc_state, indices_->tensor, values_->tensor, size_)))->maybeScalar(size.size() == 0),false);
}
Tensor SparseCUDAIntType::sparse_coo_tensor(const Tensor & indices, const Tensor & values) const {
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",1, false);
    auto values_ = checked_cast_tensor<CUDAIntTensor>(values.pImpl,"values",2, false);
    return Tensor((new SparseCUDAIntTensor(context, THCSIntTensor_newWithTensor(context->thc_state, indices_->tensor, values_->tensor)))->maybeScalar(indices_->isScalar() && values_->isScalar()),false);
}
Tensor & SparseCUDAIntType::sparse_raw_resize_(Tensor & self, IntList size, int64_t nDimI, int64_t nDimV) const {
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
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
    THCSIntTensor_rawResize(context->thc_state, self_->tensor, nDimI, nDimV, (*size_).data);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor SparseCUDAIntType::to_dense(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDAIntTensor(context, THCSIntTensor_toDense(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
int64_t SparseCUDAIntType::_dimI(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    return THCSIntTensor_nDimensionI(context->thc_state, self_->tensor);
}
int64_t SparseCUDAIntType::_dimV(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    return THCSIntTensor_nDimensionV(context->thc_state, self_->tensor);
}
int64_t SparseCUDAIntType::_nnz(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    return THCSIntTensor_nnz(context->thc_state, self_->tensor);
}
Tensor SparseCUDAIntType::coalesce(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCUDAIntTensor(context, THCSIntTensor_newCoalesce(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
bool SparseCUDAIntType::is_coalesced(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    return THCSIntTensor_isCoalesced(context->thc_state, self_->tensor);
}
Tensor SparseCUDAIntType::_indices(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      // Empty tensor
      return self_->type().toScalarType(kLong).tensor({0});
    }
    return Tensor((new CUDALongTensor(context, THCSIntTensor_newIndices(context->thc_state, self_->tensor))),false);
}
Tensor SparseCUDAIntType::_values(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAIntTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDAIntTensor(context, THCSIntTensor_newValues(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & SparseCUDAIntType::hspmm_out(Tensor & result, const Tensor & mat1, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<SparseCUDAIntTensor>(result.pImpl,"result",0, false);
    auto mat1_ = checked_cast_tensor<SparseCUDAIntTensor>(mat1.pImpl,"mat1",1, false);
    auto mat2_ = checked_cast_tensor<CUDAIntTensor>(mat2.pImpl,"mat2",2, false);
    THCSIntTensor_hspmm(context->thc_state, result_->tensor, int(1), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor SparseCUDAIntType::hspmm(const Tensor & mat1, const Tensor & mat2) const {
    auto result_ = new SparseCUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto mat1_ = checked_cast_tensor<SparseCUDAIntTensor>(mat1.pImpl,"mat1",1, false);
    auto mat2_ = checked_cast_tensor<CUDAIntTensor>(mat2.pImpl,"mat2",2, false);
    THCSIntTensor_hspmm(context->thc_state, result_->tensor, int(1), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & SparseCUDAIntType::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return  at::native::_sspaddmm_out_cuda(result, self, mat1, mat2, beta, alpha);
}

}
