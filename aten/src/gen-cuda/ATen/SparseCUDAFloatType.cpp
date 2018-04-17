// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/SparseCUDAFloatType.h"
#include "ATen/CUDAFloatStorage.h"
#include "ATen/SparseCUDAFloatTensor.h"
#include "ATen/CUDAGenerator.h"
#include "ATen/SparseCUDAByteTensor.h"
#include "ATen/SparseCUDAIntTensor.h"
#include "ATen/SparseCUDALongTensor.h"
#include "ATen/SparseCUDAFloatTensor.h"
#include "ATen/CUDAFloatTensor.h"
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

SparseCUDAFloatType::SparseCUDAFloatType(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
ScalarType SparseCUDAFloatType::scalarType() const {
  return ScalarType::Float;
}
Backend SparseCUDAFloatType::backend() const {
  return Backend::SparseCUDA;
}
bool SparseCUDAFloatType::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool SparseCUDAFloatType::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool SparseCUDAFloatType::is_distributed() const { return false; }

std::unique_ptr<Storage> SparseCUDAFloatType::storage() const {
  return std::unique_ptr<Storage>(new CUDAFloatStorage(context));
}
std::unique_ptr<Storage> SparseCUDAFloatType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CUDAFloatStorage(context,size));
}
std::unique_ptr<Storage> SparseCUDAFloatType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CUDAFloatStorage(context,data,size,deleter));
}
std::unique_ptr<Storage> SparseCUDAFloatType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new CUDAFloatStorage(context, size, std::move(allocator)));
}
Tensor SparseCUDAFloatType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THCSFloatTensor_retain(context->thc_state,  (THCSFloatTensor*) th_pointer);
  return Tensor(new SparseCUDAFloatTensor(context,(THCSFloatTensor*)(th_pointer)), false);
}
std::unique_ptr<Storage> SparseCUDAFloatType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THCudaStorage_retain(context->thc_state,  (THCudaStorage*) th_pointer);
  return std::unique_ptr<Storage>(new CUDAFloatStorage(context, (THCudaStorage*) th_pointer));
}
std::unique_ptr<Generator> SparseCUDAFloatType::generator() const {
  return std::unique_ptr<Generator>(new CUDAGenerator(context));
}

const char * SparseCUDAFloatType::toString() const {
  return SparseCUDAFloatType::typeString();
}
TypeID SparseCUDAFloatType::ID() const {
  return TypeID::SparseCUDAFloat;
}

std::size_t SparseCUDAFloatType::elementSizeInBytes() const {
  return sizeof(float);
}

const char * SparseCUDAFloatType::typeString() {
  return "SparseCUDAFloatType";
}

/* example
Tensor * SparseCUDAFloatType::add(Tensor & a, Tensor & b) {
  std::cout << "add SparseCUDAFloatTensor\n";
  return &a;
}
*/

Tensor SparseCUDAFloatType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    dim0 = maybe_wrap_dim(dim0, self_);
    dim1 = maybe_wrap_dim(dim1, self_);
    return Tensor((new SparseCUDAFloatTensor(context, THCSFloatTensor_newTranspose(context->thc_state, self_->tensor, dim0, dim1)))->maybeScalar(self_->isScalar()),false);
}
Tensor SparseCUDAFloatType::t(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCUDAFloatTensor(context, THCSFloatTensor_newTranspose(context->thc_state, self_->tensor, 0, 1)))->maybeScalar(self_->isScalar()),false);
}
Tensor SparseCUDAFloatType::clone(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCUDAFloatTensor(context, THCSFloatTensor_newClone(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & SparseCUDAFloatType::resize_as_(Tensor & self, const Tensor & the_template) const {
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    auto the_template_ = checked_cast_tensor<SparseCUDAFloatTensor>(the_template.pImpl,"the_template",2, false);
    THCSFloatTensor_resizeAs(context->thc_state, self_->tensor, the_template_->tensor);
    self_->maybeScalar(the_template_->isScalar());
    return self;
}
int64_t SparseCUDAFloatType::get_device(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THCSFloatTensor_getDevice(context->thc_state, self_->tensor));
}
Tensor SparseCUDAFloatType::norm(const Tensor & self, Scalar p) const {
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toFloat();
    // norm(value) for a sparse tensor returns a DENSE 0-dim tensor
    if (self.is_sparse()) {
      auto result = THCSFloatTensor_normall(context->thc_state,  self_->tensor, convert<float>(p_));
      return toBackend(toDense(backend())).tensor({}).fill_(result);
    }
    // aten_custom_call is followed by the generated call to normall
    return scalarTensor(convert<float>(THCSFloatTensor_normall(context->thc_state, self_->tensor, p_)));
}
Tensor & SparseCUDAFloatType::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    auto result_ = checked_cast_tensor<SparseCUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toFloat();
    THCSFloatTensor_pow(context->thc_state, result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCUDAFloatType::pow(const Tensor & self, Scalar exponent) const {
    auto result_ = new SparseCUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toFloat();
    THCSFloatTensor_pow(context->thc_state, result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCUDAFloatType::zero_(Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    THCSFloatTensor_zero(context->thc_state, self_->tensor);
    return self;
}
Tensor & SparseCUDAFloatType::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<SparseCUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<SparseCUDAFloatTensor>(other.pImpl,"other",3, false);
    THCSFloatTensor_cadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCUDAFloatType::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add(self, Scalar(other), alpha);
    }
    auto result_ = new SparseCUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<SparseCUDAFloatTensor>(other.pImpl,"other",3, false);
    THCSFloatTensor_cadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCUDAFloatType::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<SparseCUDAFloatTensor>(other.pImpl,"other",4, false);
    THCSFloatTensor_cadd(context->thc_state, self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & SparseCUDAFloatType::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<SparseCUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<SparseCUDAFloatTensor>(other.pImpl,"other",3, false);
    THCSFloatTensor_csub(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCUDAFloatType::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub(self, Scalar(other), alpha);
    }
    auto result_ = new SparseCUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<SparseCUDAFloatTensor>(other.pImpl,"other",3, false);
    THCSFloatTensor_csub(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCUDAFloatType::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<SparseCUDAFloatTensor>(other.pImpl,"other",4, false);
    THCSFloatTensor_csub(context->thc_state, self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & SparseCUDAFloatType::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<SparseCUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCSFloatTensor_mul(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCUDAFloatType::mul(const Tensor & self, Scalar other) const {
    auto result_ = new SparseCUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCSFloatTensor_mul(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCUDAFloatType::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<SparseCUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCUDAFloatTensor>(other.pImpl,"other",2, false);
    THCSFloatTensor_cmul(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCUDAFloatType::s_mul(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul(self, Scalar(other));
    }
    auto result_ = new SparseCUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCUDAFloatTensor>(other.pImpl,"other",2, false);
    THCSFloatTensor_cmul(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCUDAFloatType::mul_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCSFloatTensor_mul(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & SparseCUDAFloatType::s_mul_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCUDAFloatTensor>(other.pImpl,"other",3, false);
    THCSFloatTensor_cmul(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & SparseCUDAFloatType::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<SparseCUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCSFloatTensor_div(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCUDAFloatType::div(const Tensor & self, Scalar other) const {
    auto result_ = new SparseCUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCSFloatTensor_div(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCUDAFloatType::div_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCSFloatTensor_div(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor SparseCUDAFloatType::tensor(IntList size) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new SparseCUDAFloatTensor(context, THCSFloatTensor_newWithSize(context->thc_state, size_, NULL)))->maybeScalar(size.size() == 0),false);
}
Tensor SparseCUDAFloatType::tensor() const {
    return Tensor((new SparseCUDAFloatTensor(context, THCSFloatTensor_new(context->thc_state))),false);
}
Tensor SparseCUDAFloatType::sparse_coo_tensor(const Tensor & indices, const Tensor & values, IntList size) const {
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",1, false);
    auto values_ = checked_cast_tensor<CUDAFloatTensor>(values.pImpl,"values",2, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new SparseCUDAFloatTensor(context, THCSFloatTensor_newWithTensorAndSize(context->thc_state, indices_->tensor, values_->tensor, size_)))->maybeScalar(size.size() == 0),false);
}
Tensor SparseCUDAFloatType::sparse_coo_tensor(const Tensor & indices, const Tensor & values) const {
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",1, false);
    auto values_ = checked_cast_tensor<CUDAFloatTensor>(values.pImpl,"values",2, false);
    return Tensor((new SparseCUDAFloatTensor(context, THCSFloatTensor_newWithTensor(context->thc_state, indices_->tensor, values_->tensor)))->maybeScalar(indices_->isScalar() && values_->isScalar()),false);
}
Tensor & SparseCUDAFloatType::sparse_raw_resize_(Tensor & self, IntList size, int64_t nDimI, int64_t nDimV) const {
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
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
    THCSFloatTensor_rawResize(context->thc_state, self_->tensor, nDimI, nDimV, (*size_).data);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor SparseCUDAFloatType::to_dense(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDAFloatTensor(context, THCSFloatTensor_toDense(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
int64_t SparseCUDAFloatType::_dimI(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    return THCSFloatTensor_nDimensionI(context->thc_state, self_->tensor);
}
int64_t SparseCUDAFloatType::_dimV(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    return THCSFloatTensor_nDimensionV(context->thc_state, self_->tensor);
}
int64_t SparseCUDAFloatType::_nnz(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    return THCSFloatTensor_nnz(context->thc_state, self_->tensor);
}
Tensor SparseCUDAFloatType::coalesce(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCUDAFloatTensor(context, THCSFloatTensor_newCoalesce(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
bool SparseCUDAFloatType::is_coalesced(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    return THCSFloatTensor_isCoalesced(context->thc_state, self_->tensor);
}
Tensor SparseCUDAFloatType::_indices(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      // Empty tensor
      return self_->type().toScalarType(kLong).tensor({0});
    }
    return Tensor((new CUDALongTensor(context, THCSFloatTensor_newIndices(context->thc_state, self_->tensor))),false);
}
Tensor SparseCUDAFloatType::_values(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDAFloatTensor(context, THCSFloatTensor_newValues(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & SparseCUDAFloatType::hspmm_out(Tensor & result, const Tensor & mat1, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<SparseCUDAFloatTensor>(result.pImpl,"result",0, false);
    auto mat1_ = checked_cast_tensor<SparseCUDAFloatTensor>(mat1.pImpl,"mat1",1, false);
    auto mat2_ = checked_cast_tensor<CUDAFloatTensor>(mat2.pImpl,"mat2",2, false);
    THCSFloatTensor_hspmm(context->thc_state, result_->tensor, float(1), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor SparseCUDAFloatType::hspmm(const Tensor & mat1, const Tensor & mat2) const {
    auto result_ = new SparseCUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto mat1_ = checked_cast_tensor<SparseCUDAFloatTensor>(mat1.pImpl,"mat1",1, false);
    auto mat2_ = checked_cast_tensor<CUDAFloatTensor>(mat2.pImpl,"mat2",2, false);
    THCSFloatTensor_hspmm(context->thc_state, result_->tensor, float(1), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & SparseCUDAFloatType::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return  at::native::_sspaddmm_out_cuda(result, self, mat1, mat2, beta, alpha);
}

}
