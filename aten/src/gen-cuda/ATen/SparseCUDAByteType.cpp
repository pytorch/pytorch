// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/SparseCUDAByteType.h"
#include "ATen/CUDAByteStorage.h"
#include "ATen/SparseCUDAByteTensor.h"
#include "ATen/CUDAGenerator.h"
#include "ATen/SparseCUDAByteTensor.h"
#include "ATen/SparseCUDAIntTensor.h"
#include "ATen/SparseCUDALongTensor.h"
#include "ATen/SparseCUDAByteTensor.h"
#include "ATen/CUDAByteTensor.h"
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

SparseCUDAByteType::SparseCUDAByteType(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
ScalarType SparseCUDAByteType::scalarType() const {
  return ScalarType::Byte;
}
Backend SparseCUDAByteType::backend() const {
  return Backend::SparseCUDA;
}
bool SparseCUDAByteType::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool SparseCUDAByteType::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool SparseCUDAByteType::is_distributed() const { return false; }

std::unique_ptr<Storage> SparseCUDAByteType::storage() const {
  return std::unique_ptr<Storage>(new CUDAByteStorage(context));
}
std::unique_ptr<Storage> SparseCUDAByteType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CUDAByteStorage(context,size));
}
std::unique_ptr<Storage> SparseCUDAByteType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CUDAByteStorage(context,data,size,deleter));
}
std::unique_ptr<Storage> SparseCUDAByteType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new CUDAByteStorage(context, size, std::move(allocator)));
}
Tensor SparseCUDAByteType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THCSByteTensor_retain(context->thc_state,  (THCSByteTensor*) th_pointer);
  return Tensor(new SparseCUDAByteTensor(context,(THCSByteTensor*)(th_pointer)), false);
}
std::unique_ptr<Storage> SparseCUDAByteType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THCudaByteStorage_retain(context->thc_state,  (THCudaByteStorage*) th_pointer);
  return std::unique_ptr<Storage>(new CUDAByteStorage(context, (THCudaByteStorage*) th_pointer));
}
std::unique_ptr<Generator> SparseCUDAByteType::generator() const {
  return std::unique_ptr<Generator>(new CUDAGenerator(context));
}

const char * SparseCUDAByteType::toString() const {
  return SparseCUDAByteType::typeString();
}
TypeID SparseCUDAByteType::ID() const {
  return TypeID::SparseCUDAByte;
}

std::size_t SparseCUDAByteType::elementSizeInBytes() const {
  return sizeof(uint8_t);
}

const char * SparseCUDAByteType::typeString() {
  return "SparseCUDAByteType";
}

/* example
Tensor * SparseCUDAByteType::add(Tensor & a, Tensor & b) {
  std::cout << "add SparseCUDAByteTensor\n";
  return &a;
}
*/

Tensor SparseCUDAByteType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    dim0 = maybe_wrap_dim(dim0, self_);
    dim1 = maybe_wrap_dim(dim1, self_);
    return Tensor((new SparseCUDAByteTensor(context, THCSByteTensor_newTranspose(context->thc_state, self_->tensor, dim0, dim1)))->maybeScalar(self_->isScalar()),false);
}
Tensor SparseCUDAByteType::t(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCUDAByteTensor(context, THCSByteTensor_newTranspose(context->thc_state, self_->tensor, 0, 1)))->maybeScalar(self_->isScalar()),false);
}
Tensor SparseCUDAByteType::clone(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCUDAByteTensor(context, THCSByteTensor_newClone(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & SparseCUDAByteType::resize_as_(Tensor & self, const Tensor & the_template) const {
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    auto the_template_ = checked_cast_tensor<SparseCUDAByteTensor>(the_template.pImpl,"the_template",2, false);
    THCSByteTensor_resizeAs(context->thc_state, self_->tensor, the_template_->tensor);
    self_->maybeScalar(the_template_->isScalar());
    return self;
}
int64_t SparseCUDAByteType::get_device(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THCSByteTensor_getDevice(context->thc_state, self_->tensor));
}
Tensor & SparseCUDAByteType::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    auto result_ = checked_cast_tensor<SparseCUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toByte();
    THCSByteTensor_pow(context->thc_state, result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCUDAByteType::pow(const Tensor & self, Scalar exponent) const {
    auto result_ = new SparseCUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toByte();
    THCSByteTensor_pow(context->thc_state, result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCUDAByteType::zero_(Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    THCSByteTensor_zero(context->thc_state, self_->tensor);
    return self;
}
Tensor & SparseCUDAByteType::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<SparseCUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toByte();
    auto other_ = checked_cast_tensor<SparseCUDAByteTensor>(other.pImpl,"other",3, false);
    THCSByteTensor_cadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCUDAByteType::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add(self, Scalar(other), alpha);
    }
    auto result_ = new SparseCUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toByte();
    auto other_ = checked_cast_tensor<SparseCUDAByteTensor>(other.pImpl,"other",3, false);
    THCSByteTensor_cadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCUDAByteType::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toByte();
    auto other_ = checked_cast_tensor<SparseCUDAByteTensor>(other.pImpl,"other",4, false);
    THCSByteTensor_cadd(context->thc_state, self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & SparseCUDAByteType::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<SparseCUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toByte();
    auto other_ = checked_cast_tensor<SparseCUDAByteTensor>(other.pImpl,"other",3, false);
    THCSByteTensor_csub(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCUDAByteType::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub(self, Scalar(other), alpha);
    }
    auto result_ = new SparseCUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toByte();
    auto other_ = checked_cast_tensor<SparseCUDAByteTensor>(other.pImpl,"other",3, false);
    THCSByteTensor_csub(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCUDAByteType::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toByte();
    auto other_ = checked_cast_tensor<SparseCUDAByteTensor>(other.pImpl,"other",4, false);
    THCSByteTensor_csub(context->thc_state, self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & SparseCUDAByteType::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<SparseCUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toByte();
    THCSByteTensor_mul(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCUDAByteType::mul(const Tensor & self, Scalar other) const {
    auto result_ = new SparseCUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toByte();
    THCSByteTensor_mul(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCUDAByteType::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<SparseCUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCUDAByteTensor>(other.pImpl,"other",2, false);
    THCSByteTensor_cmul(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCUDAByteType::s_mul(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul(self, Scalar(other));
    }
    auto result_ = new SparseCUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCUDAByteTensor>(other.pImpl,"other",2, false);
    THCSByteTensor_cmul(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCUDAByteType::mul_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toByte();
    THCSByteTensor_mul(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & SparseCUDAByteType::s_mul_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCUDAByteTensor>(other.pImpl,"other",3, false);
    THCSByteTensor_cmul(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & SparseCUDAByteType::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<SparseCUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toByte();
    THCSByteTensor_div(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCUDAByteType::div(const Tensor & self, Scalar other) const {
    auto result_ = new SparseCUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toByte();
    THCSByteTensor_div(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCUDAByteType::div_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toByte();
    THCSByteTensor_div(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor SparseCUDAByteType::tensor(IntList size) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new SparseCUDAByteTensor(context, THCSByteTensor_newWithSize(context->thc_state, size_, NULL)))->maybeScalar(size.size() == 0),false);
}
Tensor SparseCUDAByteType::tensor() const {
    return Tensor((new SparseCUDAByteTensor(context, THCSByteTensor_new(context->thc_state))),false);
}
Tensor SparseCUDAByteType::sparse_coo_tensor(const Tensor & indices, const Tensor & values, IntList size) const {
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",1, false);
    auto values_ = checked_cast_tensor<CUDAByteTensor>(values.pImpl,"values",2, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new SparseCUDAByteTensor(context, THCSByteTensor_newWithTensorAndSize(context->thc_state, indices_->tensor, values_->tensor, size_)))->maybeScalar(size.size() == 0),false);
}
Tensor SparseCUDAByteType::sparse_coo_tensor(const Tensor & indices, const Tensor & values) const {
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",1, false);
    auto values_ = checked_cast_tensor<CUDAByteTensor>(values.pImpl,"values",2, false);
    return Tensor((new SparseCUDAByteTensor(context, THCSByteTensor_newWithTensor(context->thc_state, indices_->tensor, values_->tensor)))->maybeScalar(indices_->isScalar() && values_->isScalar()),false);
}
Tensor & SparseCUDAByteType::sparse_raw_resize_(Tensor & self, IntList size, int64_t nDimI, int64_t nDimV) const {
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
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
    THCSByteTensor_rawResize(context->thc_state, self_->tensor, nDimI, nDimV, (*size_).data);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor SparseCUDAByteType::to_dense(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDAByteTensor(context, THCSByteTensor_toDense(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
int64_t SparseCUDAByteType::_dimI(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    return THCSByteTensor_nDimensionI(context->thc_state, self_->tensor);
}
int64_t SparseCUDAByteType::_dimV(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    return THCSByteTensor_nDimensionV(context->thc_state, self_->tensor);
}
int64_t SparseCUDAByteType::_nnz(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    return THCSByteTensor_nnz(context->thc_state, self_->tensor);
}
Tensor SparseCUDAByteType::coalesce(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCUDAByteTensor(context, THCSByteTensor_newCoalesce(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
bool SparseCUDAByteType::is_coalesced(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    return THCSByteTensor_isCoalesced(context->thc_state, self_->tensor);
}
Tensor SparseCUDAByteType::_indices(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      // Empty tensor
      return self_->type().toScalarType(kLong).tensor({0});
    }
    return Tensor((new CUDALongTensor(context, THCSByteTensor_newIndices(context->thc_state, self_->tensor))),false);
}
Tensor SparseCUDAByteType::_values(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCUDAByteTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDAByteTensor(context, THCSByteTensor_newValues(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & SparseCUDAByteType::hspmm_out(Tensor & result, const Tensor & mat1, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<SparseCUDAByteTensor>(result.pImpl,"result",0, false);
    auto mat1_ = checked_cast_tensor<SparseCUDAByteTensor>(mat1.pImpl,"mat1",1, false);
    auto mat2_ = checked_cast_tensor<CUDAByteTensor>(mat2.pImpl,"mat2",2, false);
    THCSByteTensor_hspmm(context->thc_state, result_->tensor, uint8_t(1), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor SparseCUDAByteType::hspmm(const Tensor & mat1, const Tensor & mat2) const {
    auto result_ = new SparseCUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto mat1_ = checked_cast_tensor<SparseCUDAByteTensor>(mat1.pImpl,"mat1",1, false);
    auto mat2_ = checked_cast_tensor<CUDAByteTensor>(mat2.pImpl,"mat2",2, false);
    THCSByteTensor_hspmm(context->thc_state, result_->tensor, uint8_t(1), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & SparseCUDAByteType::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return  at::native::_sspaddmm_out_cuda(result, self, mat1, mat2, beta, alpha);
}

}
