// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/SparseCPUByteType.h"
#include "ATen/CPUByteStorage.h"
#include "ATen/SparseCPUByteTensor.h"
#include "ATen/CPUGenerator.h"
#include "ATen/SparseCPUByteTensor.h"
#include "ATen/SparseCPUIntTensor.h"
#include "ATen/SparseCPULongTensor.h"
#include "ATen/SparseCPUByteTensor.h"
#include "ATen/CPUByteTensor.h"
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

SparseCPUByteType::SparseCPUByteType(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
ScalarType SparseCPUByteType::scalarType() const {
  return ScalarType::Byte;
}
Backend SparseCPUByteType::backend() const {
  return Backend::SparseCPU;
}
bool SparseCPUByteType::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool SparseCPUByteType::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool SparseCPUByteType::is_distributed() const { return false; }

std::unique_ptr<Storage> SparseCPUByteType::storage() const {
  return std::unique_ptr<Storage>(new CPUByteStorage(context));
}
std::unique_ptr<Storage> SparseCPUByteType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CPUByteStorage(context,size));
}
std::unique_ptr<Storage> SparseCPUByteType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CPUByteStorage(context,data,size,deleter));
}
std::unique_ptr<Storage> SparseCPUByteType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new CPUByteStorage(context, size, std::move(allocator)));
}
Tensor SparseCPUByteType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THSByteTensor_retain( (THSByteTensor*) th_pointer);
  return Tensor(new SparseCPUByteTensor(context,(THSByteTensor*)(th_pointer)), false);
}
std::unique_ptr<Storage> SparseCPUByteType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THByteStorage_retain( (THByteStorage*) th_pointer);
  return std::unique_ptr<Storage>(new CPUByteStorage(context, (THByteStorage*) th_pointer));
}
std::unique_ptr<Generator> SparseCPUByteType::generator() const {
  return std::unique_ptr<Generator>(new CPUGenerator(context));
}

const char * SparseCPUByteType::toString() const {
  return SparseCPUByteType::typeString();
}
TypeID SparseCPUByteType::ID() const {
  return TypeID::SparseCPUByte;
}

std::size_t SparseCPUByteType::elementSizeInBytes() const {
  return sizeof(uint8_t);
}

const char * SparseCPUByteType::typeString() {
  return "SparseCPUByteType";
}

/* example
Tensor * SparseCPUByteType::add(Tensor & a, Tensor & b) {
  std::cout << "add SparseCPUByteTensor\n";
  return &a;
}
*/

Tensor SparseCPUByteType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    dim0 = maybe_wrap_dim(dim0, self_);
    dim1 = maybe_wrap_dim(dim1, self_);
    return Tensor((new SparseCPUByteTensor(context, THSByteTensor_newTranspose(self_->tensor, dim0, dim1)))->maybeScalar(self_->isScalar()),false);
}
Tensor SparseCPUByteType::t(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCPUByteTensor(context, THSByteTensor_newTranspose(self_->tensor, 0, 1)))->maybeScalar(self_->isScalar()),false);
}
Tensor SparseCPUByteType::clone(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCPUByteTensor(context, THSByteTensor_newClone(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & SparseCPUByteType::resize_as_(Tensor & self, const Tensor & the_template) const {
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    auto the_template_ = checked_cast_tensor<SparseCPUByteTensor>(the_template.pImpl,"the_template",2, false);
    THSByteTensor_resizeAs(self_->tensor, the_template_->tensor);
    self_->maybeScalar(the_template_->isScalar());
    return self;
}
Tensor & SparseCPUByteType::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    auto result_ = checked_cast_tensor<SparseCPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toByte();
    THSByteTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCPUByteType::pow(const Tensor & self, Scalar exponent) const {
    auto result_ = new SparseCPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toByte();
    THSByteTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCPUByteType::zero_(Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    THSByteTensor_zero(self_->tensor);
    return self;
}
Tensor & SparseCPUByteType::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<SparseCPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toByte();
    auto other_ = checked_cast_tensor<SparseCPUByteTensor>(other.pImpl,"other",3, false);
    THSByteTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCPUByteType::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add(self, Scalar(other), alpha);
    }
    auto result_ = new SparseCPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toByte();
    auto other_ = checked_cast_tensor<SparseCPUByteTensor>(other.pImpl,"other",3, false);
    THSByteTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCPUByteType::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toByte();
    auto other_ = checked_cast_tensor<SparseCPUByteTensor>(other.pImpl,"other",4, false);
    THSByteTensor_cadd(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & SparseCPUByteType::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<SparseCPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toByte();
    auto other_ = checked_cast_tensor<SparseCPUByteTensor>(other.pImpl,"other",3, false);
    THSByteTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCPUByteType::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub(self, Scalar(other), alpha);
    }
    auto result_ = new SparseCPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toByte();
    auto other_ = checked_cast_tensor<SparseCPUByteTensor>(other.pImpl,"other",3, false);
    THSByteTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCPUByteType::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toByte();
    auto other_ = checked_cast_tensor<SparseCPUByteTensor>(other.pImpl,"other",4, false);
    THSByteTensor_csub(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & SparseCPUByteType::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<SparseCPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toByte();
    THSByteTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCPUByteType::mul(const Tensor & self, Scalar other) const {
    auto result_ = new SparseCPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toByte();
    THSByteTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCPUByteType::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<SparseCPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCPUByteTensor>(other.pImpl,"other",2, false);
    THSByteTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCPUByteType::s_mul(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul(self, Scalar(other));
    }
    auto result_ = new SparseCPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCPUByteTensor>(other.pImpl,"other",2, false);
    THSByteTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCPUByteType::mul_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toByte();
    THSByteTensor_mul(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & SparseCPUByteType::s_mul_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCPUByteTensor>(other.pImpl,"other",3, false);
    THSByteTensor_cmul(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & SparseCPUByteType::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<SparseCPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toByte();
    THSByteTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCPUByteType::div(const Tensor & self, Scalar other) const {
    auto result_ = new SparseCPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toByte();
    THSByteTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCPUByteType::div_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toByte();
    THSByteTensor_div(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor SparseCPUByteType::tensor(IntList size) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new SparseCPUByteTensor(context, THSByteTensor_newWithSize(size_, NULL)))->maybeScalar(size.size() == 0),false);
}
Tensor SparseCPUByteType::tensor() const {
    return Tensor((new SparseCPUByteTensor(context, THSByteTensor_new())),false);
}
Tensor SparseCPUByteType::sparse_coo_tensor(const Tensor & indices, const Tensor & values, IntList size) const {
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",1, false);
    auto values_ = checked_cast_tensor<CPUByteTensor>(values.pImpl,"values",2, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new SparseCPUByteTensor(context, THSByteTensor_newWithTensorAndSize(indices_->tensor, values_->tensor, size_)))->maybeScalar(size.size() == 0),false);
}
Tensor SparseCPUByteType::sparse_coo_tensor(const Tensor & indices, const Tensor & values) const {
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",1, false);
    auto values_ = checked_cast_tensor<CPUByteTensor>(values.pImpl,"values",2, false);
    return Tensor((new SparseCPUByteTensor(context, THSByteTensor_newWithTensor(indices_->tensor, values_->tensor)))->maybeScalar(indices_->isScalar() && values_->isScalar()),false);
}
Tensor & SparseCPUByteType::sparse_raw_resize_(Tensor & self, IntList size, int64_t nDimI, int64_t nDimV) const {
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
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
    THSByteTensor_rawResize(self_->tensor, nDimI, nDimV, (*size_).data);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor SparseCPUByteType::to_dense(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUByteTensor(context, THSByteTensor_toDense(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
int64_t SparseCPUByteType::_dimI(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    return THSByteTensor_nDimensionI(self_->tensor);
}
int64_t SparseCPUByteType::_dimV(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    return THSByteTensor_nDimensionV(self_->tensor);
}
int64_t SparseCPUByteType::_nnz(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    return THSByteTensor_nnz(self_->tensor);
}
Tensor SparseCPUByteType::coalesce(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCPUByteTensor(context, THSByteTensor_newCoalesce(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
bool SparseCPUByteType::is_coalesced(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    return THSByteTensor_isCoalesced(self_->tensor);
}
Tensor SparseCPUByteType::_indices(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      // Empty tensor
      return self_->type().toScalarType(kLong).tensor({0});
    }
    return Tensor((new CPULongTensor(context, THSByteTensor_newIndices(self_->tensor))),false);
}
Tensor SparseCPUByteType::_values(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUByteTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUByteTensor(context, THSByteTensor_newValues(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & SparseCPUByteType::hspmm_out(Tensor & result, const Tensor & mat1, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<SparseCPUByteTensor>(result.pImpl,"result",0, false);
    auto mat1_ = checked_cast_tensor<SparseCPUByteTensor>(mat1.pImpl,"mat1",1, false);
    auto mat2_ = checked_cast_tensor<CPUByteTensor>(mat2.pImpl,"mat2",2, false);
    THSByteTensor_hspmm(result_->tensor, uint8_t(1), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor SparseCPUByteType::hspmm(const Tensor & mat1, const Tensor & mat2) const {
    auto result_ = new SparseCPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto mat1_ = checked_cast_tensor<SparseCPUByteTensor>(mat1.pImpl,"mat1",1, false);
    auto mat2_ = checked_cast_tensor<CPUByteTensor>(mat2.pImpl,"mat2",2, false);
    THSByteTensor_hspmm(result_->tensor, uint8_t(1), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & SparseCPUByteType::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return  at::native::_sspaddmm_out_cpu(result, self, mat1, mat2, beta, alpha);
}

}
