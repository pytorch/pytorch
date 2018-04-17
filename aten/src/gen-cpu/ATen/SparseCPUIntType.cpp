// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/SparseCPUIntType.h"
#include "ATen/CPUIntStorage.h"
#include "ATen/SparseCPUIntTensor.h"
#include "ATen/CPUGenerator.h"
#include "ATen/SparseCPUByteTensor.h"
#include "ATen/SparseCPUIntTensor.h"
#include "ATen/SparseCPULongTensor.h"
#include "ATen/SparseCPUIntTensor.h"
#include "ATen/CPUIntTensor.h"
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

SparseCPUIntType::SparseCPUIntType(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
ScalarType SparseCPUIntType::scalarType() const {
  return ScalarType::Int;
}
Backend SparseCPUIntType::backend() const {
  return Backend::SparseCPU;
}
bool SparseCPUIntType::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool SparseCPUIntType::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool SparseCPUIntType::is_distributed() const { return false; }

std::unique_ptr<Storage> SparseCPUIntType::storage() const {
  return std::unique_ptr<Storage>(new CPUIntStorage(context));
}
std::unique_ptr<Storage> SparseCPUIntType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CPUIntStorage(context,size));
}
std::unique_ptr<Storage> SparseCPUIntType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CPUIntStorage(context,data,size,deleter));
}
std::unique_ptr<Storage> SparseCPUIntType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new CPUIntStorage(context, size, std::move(allocator)));
}
Tensor SparseCPUIntType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THSIntTensor_retain( (THSIntTensor*) th_pointer);
  return Tensor(new SparseCPUIntTensor(context,(THSIntTensor*)(th_pointer)), false);
}
std::unique_ptr<Storage> SparseCPUIntType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THIntStorage_retain( (THIntStorage*) th_pointer);
  return std::unique_ptr<Storage>(new CPUIntStorage(context, (THIntStorage*) th_pointer));
}
std::unique_ptr<Generator> SparseCPUIntType::generator() const {
  return std::unique_ptr<Generator>(new CPUGenerator(context));
}

const char * SparseCPUIntType::toString() const {
  return SparseCPUIntType::typeString();
}
TypeID SparseCPUIntType::ID() const {
  return TypeID::SparseCPUInt;
}

std::size_t SparseCPUIntType::elementSizeInBytes() const {
  return sizeof(int);
}

const char * SparseCPUIntType::typeString() {
  return "SparseCPUIntType";
}

/* example
Tensor * SparseCPUIntType::add(Tensor & a, Tensor & b) {
  std::cout << "add SparseCPUIntTensor\n";
  return &a;
}
*/

Tensor SparseCPUIntType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    dim0 = maybe_wrap_dim(dim0, self_);
    dim1 = maybe_wrap_dim(dim1, self_);
    return Tensor((new SparseCPUIntTensor(context, THSIntTensor_newTranspose(self_->tensor, dim0, dim1)))->maybeScalar(self_->isScalar()),false);
}
Tensor SparseCPUIntType::t(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCPUIntTensor(context, THSIntTensor_newTranspose(self_->tensor, 0, 1)))->maybeScalar(self_->isScalar()),false);
}
Tensor SparseCPUIntType::clone(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCPUIntTensor(context, THSIntTensor_newClone(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & SparseCPUIntType::resize_as_(Tensor & self, const Tensor & the_template) const {
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    auto the_template_ = checked_cast_tensor<SparseCPUIntTensor>(the_template.pImpl,"the_template",2, false);
    THSIntTensor_resizeAs(self_->tensor, the_template_->tensor);
    self_->maybeScalar(the_template_->isScalar());
    return self;
}
Tensor & SparseCPUIntType::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    auto result_ = checked_cast_tensor<SparseCPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toInt();
    THSIntTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCPUIntType::pow(const Tensor & self, Scalar exponent) const {
    auto result_ = new SparseCPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toInt();
    THSIntTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCPUIntType::zero_(Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    THSIntTensor_zero(self_->tensor);
    return self;
}
Tensor & SparseCPUIntType::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<SparseCPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<SparseCPUIntTensor>(other.pImpl,"other",3, false);
    THSIntTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCPUIntType::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add(self, Scalar(other), alpha);
    }
    auto result_ = new SparseCPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<SparseCPUIntTensor>(other.pImpl,"other",3, false);
    THSIntTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCPUIntType::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<SparseCPUIntTensor>(other.pImpl,"other",4, false);
    THSIntTensor_cadd(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & SparseCPUIntType::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<SparseCPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<SparseCPUIntTensor>(other.pImpl,"other",3, false);
    THSIntTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCPUIntType::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub(self, Scalar(other), alpha);
    }
    auto result_ = new SparseCPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<SparseCPUIntTensor>(other.pImpl,"other",3, false);
    THSIntTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCPUIntType::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<SparseCPUIntTensor>(other.pImpl,"other",4, false);
    THSIntTensor_csub(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & SparseCPUIntType::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<SparseCPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THSIntTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCPUIntType::mul(const Tensor & self, Scalar other) const {
    auto result_ = new SparseCPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THSIntTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCPUIntType::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<SparseCPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCPUIntTensor>(other.pImpl,"other",2, false);
    THSIntTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCPUIntType::s_mul(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul(self, Scalar(other));
    }
    auto result_ = new SparseCPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCPUIntTensor>(other.pImpl,"other",2, false);
    THSIntTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCPUIntType::mul_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THSIntTensor_mul(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & SparseCPUIntType::s_mul_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCPUIntTensor>(other.pImpl,"other",3, false);
    THSIntTensor_cmul(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & SparseCPUIntType::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<SparseCPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THSIntTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCPUIntType::div(const Tensor & self, Scalar other) const {
    auto result_ = new SparseCPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THSIntTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCPUIntType::div_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THSIntTensor_div(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor SparseCPUIntType::tensor(IntList size) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new SparseCPUIntTensor(context, THSIntTensor_newWithSize(size_, NULL)))->maybeScalar(size.size() == 0),false);
}
Tensor SparseCPUIntType::tensor() const {
    return Tensor((new SparseCPUIntTensor(context, THSIntTensor_new())),false);
}
Tensor SparseCPUIntType::sparse_coo_tensor(const Tensor & indices, const Tensor & values, IntList size) const {
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",1, false);
    auto values_ = checked_cast_tensor<CPUIntTensor>(values.pImpl,"values",2, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new SparseCPUIntTensor(context, THSIntTensor_newWithTensorAndSize(indices_->tensor, values_->tensor, size_)))->maybeScalar(size.size() == 0),false);
}
Tensor SparseCPUIntType::sparse_coo_tensor(const Tensor & indices, const Tensor & values) const {
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",1, false);
    auto values_ = checked_cast_tensor<CPUIntTensor>(values.pImpl,"values",2, false);
    return Tensor((new SparseCPUIntTensor(context, THSIntTensor_newWithTensor(indices_->tensor, values_->tensor)))->maybeScalar(indices_->isScalar() && values_->isScalar()),false);
}
Tensor & SparseCPUIntType::sparse_raw_resize_(Tensor & self, IntList size, int64_t nDimI, int64_t nDimV) const {
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
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
    THSIntTensor_rawResize(self_->tensor, nDimI, nDimV, (*size_).data);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor SparseCPUIntType::to_dense(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUIntTensor(context, THSIntTensor_toDense(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
int64_t SparseCPUIntType::_dimI(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    return THSIntTensor_nDimensionI(self_->tensor);
}
int64_t SparseCPUIntType::_dimV(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    return THSIntTensor_nDimensionV(self_->tensor);
}
int64_t SparseCPUIntType::_nnz(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    return THSIntTensor_nnz(self_->tensor);
}
Tensor SparseCPUIntType::coalesce(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCPUIntTensor(context, THSIntTensor_newCoalesce(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
bool SparseCPUIntType::is_coalesced(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    return THSIntTensor_isCoalesced(self_->tensor);
}
Tensor SparseCPUIntType::_indices(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      // Empty tensor
      return self_->type().toScalarType(kLong).tensor({0});
    }
    return Tensor((new CPULongTensor(context, THSIntTensor_newIndices(self_->tensor))),false);
}
Tensor SparseCPUIntType::_values(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUIntTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUIntTensor(context, THSIntTensor_newValues(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & SparseCPUIntType::hspmm_out(Tensor & result, const Tensor & mat1, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<SparseCPUIntTensor>(result.pImpl,"result",0, false);
    auto mat1_ = checked_cast_tensor<SparseCPUIntTensor>(mat1.pImpl,"mat1",1, false);
    auto mat2_ = checked_cast_tensor<CPUIntTensor>(mat2.pImpl,"mat2",2, false);
    THSIntTensor_hspmm(result_->tensor, int(1), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor SparseCPUIntType::hspmm(const Tensor & mat1, const Tensor & mat2) const {
    auto result_ = new SparseCPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto mat1_ = checked_cast_tensor<SparseCPUIntTensor>(mat1.pImpl,"mat1",1, false);
    auto mat2_ = checked_cast_tensor<CPUIntTensor>(mat2.pImpl,"mat2",2, false);
    THSIntTensor_hspmm(result_->tensor, int(1), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & SparseCPUIntType::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return  at::native::_sspaddmm_out_cpu(result, self, mat1, mat2, beta, alpha);
}

}
