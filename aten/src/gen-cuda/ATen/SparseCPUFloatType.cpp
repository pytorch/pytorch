// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/SparseCPUFloatType.h"
#include "ATen/CPUFloatStorage.h"
#include "ATen/SparseCPUFloatTensor.h"
#include "ATen/CPUGenerator.h"
#include "ATen/SparseCPUByteTensor.h"
#include "ATen/SparseCPUIntTensor.h"
#include "ATen/SparseCPULongTensor.h"
#include "ATen/SparseCPUFloatTensor.h"
#include "ATen/CPUFloatTensor.h"
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

SparseCPUFloatType::SparseCPUFloatType(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
ScalarType SparseCPUFloatType::scalarType() const {
  return ScalarType::Float;
}
Backend SparseCPUFloatType::backend() const {
  return Backend::SparseCPU;
}
bool SparseCPUFloatType::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool SparseCPUFloatType::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool SparseCPUFloatType::is_distributed() const { return false; }

std::unique_ptr<Storage> SparseCPUFloatType::storage() const {
  return std::unique_ptr<Storage>(new CPUFloatStorage(context));
}
std::unique_ptr<Storage> SparseCPUFloatType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CPUFloatStorage(context,size));
}
std::unique_ptr<Storage> SparseCPUFloatType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CPUFloatStorage(context,data,size,deleter));
}
std::unique_ptr<Storage> SparseCPUFloatType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new CPUFloatStorage(context, size, std::move(allocator)));
}
Tensor SparseCPUFloatType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THSFloatTensor_retain( (THSFloatTensor*) th_pointer);
  return Tensor(new SparseCPUFloatTensor(context,(THSFloatTensor*)(th_pointer)), false);
}
std::unique_ptr<Storage> SparseCPUFloatType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THFloatStorage_retain( (THFloatStorage*) th_pointer);
  return std::unique_ptr<Storage>(new CPUFloatStorage(context, (THFloatStorage*) th_pointer));
}
std::unique_ptr<Generator> SparseCPUFloatType::generator() const {
  return std::unique_ptr<Generator>(new CPUGenerator(context));
}

const char * SparseCPUFloatType::toString() const {
  return SparseCPUFloatType::typeString();
}
TypeID SparseCPUFloatType::ID() const {
  return TypeID::SparseCPUFloat;
}

std::size_t SparseCPUFloatType::elementSizeInBytes() const {
  return sizeof(float);
}

const char * SparseCPUFloatType::typeString() {
  return "SparseCPUFloatType";
}

/* example
Tensor * SparseCPUFloatType::add(Tensor & a, Tensor & b) {
  std::cout << "add SparseCPUFloatTensor\n";
  return &a;
}
*/

Tensor SparseCPUFloatType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    dim0 = maybe_wrap_dim(dim0, self_);
    dim1 = maybe_wrap_dim(dim1, self_);
    return Tensor((new SparseCPUFloatTensor(context, THSFloatTensor_newTranspose(self_->tensor, dim0, dim1)))->maybeScalar(self_->isScalar()),false);
}
Tensor SparseCPUFloatType::t(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCPUFloatTensor(context, THSFloatTensor_newTranspose(self_->tensor, 0, 1)))->maybeScalar(self_->isScalar()),false);
}
Tensor SparseCPUFloatType::clone(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCPUFloatTensor(context, THSFloatTensor_newClone(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & SparseCPUFloatType::resize_as_(Tensor & self, const Tensor & the_template) const {
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    auto the_template_ = checked_cast_tensor<SparseCPUFloatTensor>(the_template.pImpl,"the_template",2, false);
    THSFloatTensor_resizeAs(self_->tensor, the_template_->tensor);
    self_->maybeScalar(the_template_->isScalar());
    return self;
}
Tensor SparseCPUFloatType::norm(const Tensor & self, Scalar p) const {
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toFloat();
    // norm(value) for a sparse tensor returns a DENSE 0-dim tensor
    if (self.is_sparse()) {
      auto result = THSFloatTensor_normall( self_->tensor, convert<float>(p_));
      return toBackend(toDense(backend())).tensor({}).fill_(result);
    }
    // aten_custom_call is followed by the generated call to normall
    return scalarTensor(convert<float>(THSFloatTensor_normall(self_->tensor, p_)));
}
Tensor & SparseCPUFloatType::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    auto result_ = checked_cast_tensor<SparseCPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toFloat();
    THSFloatTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCPUFloatType::pow(const Tensor & self, Scalar exponent) const {
    auto result_ = new SparseCPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toFloat();
    THSFloatTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCPUFloatType::zero_(Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    THSFloatTensor_zero(self_->tensor);
    return self;
}
Tensor & SparseCPUFloatType::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<SparseCPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<SparseCPUFloatTensor>(other.pImpl,"other",3, false);
    THSFloatTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCPUFloatType::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add(self, Scalar(other), alpha);
    }
    auto result_ = new SparseCPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<SparseCPUFloatTensor>(other.pImpl,"other",3, false);
    THSFloatTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCPUFloatType::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<SparseCPUFloatTensor>(other.pImpl,"other",4, false);
    THSFloatTensor_cadd(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & SparseCPUFloatType::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<SparseCPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<SparseCPUFloatTensor>(other.pImpl,"other",3, false);
    THSFloatTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCPUFloatType::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub(self, Scalar(other), alpha);
    }
    auto result_ = new SparseCPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<SparseCPUFloatTensor>(other.pImpl,"other",3, false);
    THSFloatTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCPUFloatType::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<SparseCPUFloatTensor>(other.pImpl,"other",4, false);
    THSFloatTensor_csub(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & SparseCPUFloatType::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<SparseCPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THSFloatTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCPUFloatType::mul(const Tensor & self, Scalar other) const {
    auto result_ = new SparseCPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THSFloatTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCPUFloatType::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<SparseCPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCPUFloatTensor>(other.pImpl,"other",2, false);
    THSFloatTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCPUFloatType::s_mul(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul(self, Scalar(other));
    }
    auto result_ = new SparseCPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCPUFloatTensor>(other.pImpl,"other",2, false);
    THSFloatTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCPUFloatType::mul_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THSFloatTensor_mul(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & SparseCPUFloatType::s_mul_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCPUFloatTensor>(other.pImpl,"other",3, false);
    THSFloatTensor_cmul(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & SparseCPUFloatType::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<SparseCPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THSFloatTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCPUFloatType::div(const Tensor & self, Scalar other) const {
    auto result_ = new SparseCPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THSFloatTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCPUFloatType::div_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THSFloatTensor_div(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor SparseCPUFloatType::tensor(IntList size) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new SparseCPUFloatTensor(context, THSFloatTensor_newWithSize(size_, NULL)))->maybeScalar(size.size() == 0),false);
}
Tensor SparseCPUFloatType::tensor() const {
    return Tensor((new SparseCPUFloatTensor(context, THSFloatTensor_new())),false);
}
Tensor SparseCPUFloatType::sparse_coo_tensor(const Tensor & indices, const Tensor & values, IntList size) const {
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",1, false);
    auto values_ = checked_cast_tensor<CPUFloatTensor>(values.pImpl,"values",2, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new SparseCPUFloatTensor(context, THSFloatTensor_newWithTensorAndSize(indices_->tensor, values_->tensor, size_)))->maybeScalar(size.size() == 0),false);
}
Tensor SparseCPUFloatType::sparse_coo_tensor(const Tensor & indices, const Tensor & values) const {
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",1, false);
    auto values_ = checked_cast_tensor<CPUFloatTensor>(values.pImpl,"values",2, false);
    return Tensor((new SparseCPUFloatTensor(context, THSFloatTensor_newWithTensor(indices_->tensor, values_->tensor)))->maybeScalar(indices_->isScalar() && values_->isScalar()),false);
}
Tensor & SparseCPUFloatType::sparse_raw_resize_(Tensor & self, IntList size, int64_t nDimI, int64_t nDimV) const {
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
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
    THSFloatTensor_rawResize(self_->tensor, nDimI, nDimV, (*size_).data);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor SparseCPUFloatType::to_dense(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUFloatTensor(context, THSFloatTensor_toDense(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
int64_t SparseCPUFloatType::_dimI(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    return THSFloatTensor_nDimensionI(self_->tensor);
}
int64_t SparseCPUFloatType::_dimV(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    return THSFloatTensor_nDimensionV(self_->tensor);
}
int64_t SparseCPUFloatType::_nnz(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    return THSFloatTensor_nnz(self_->tensor);
}
Tensor SparseCPUFloatType::coalesce(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCPUFloatTensor(context, THSFloatTensor_newCoalesce(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
bool SparseCPUFloatType::is_coalesced(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    return THSFloatTensor_isCoalesced(self_->tensor);
}
Tensor SparseCPUFloatType::_indices(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      // Empty tensor
      return self_->type().toScalarType(kLong).tensor({0});
    }
    return Tensor((new CPULongTensor(context, THSFloatTensor_newIndices(self_->tensor))),false);
}
Tensor SparseCPUFloatType::_values(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUFloatTensor(context, THSFloatTensor_newValues(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & SparseCPUFloatType::hspmm_out(Tensor & result, const Tensor & mat1, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<SparseCPUFloatTensor>(result.pImpl,"result",0, false);
    auto mat1_ = checked_cast_tensor<SparseCPUFloatTensor>(mat1.pImpl,"mat1",1, false);
    auto mat2_ = checked_cast_tensor<CPUFloatTensor>(mat2.pImpl,"mat2",2, false);
    THSFloatTensor_hspmm(result_->tensor, float(1), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor SparseCPUFloatType::hspmm(const Tensor & mat1, const Tensor & mat2) const {
    auto result_ = new SparseCPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto mat1_ = checked_cast_tensor<SparseCPUFloatTensor>(mat1.pImpl,"mat1",1, false);
    auto mat2_ = checked_cast_tensor<CPUFloatTensor>(mat2.pImpl,"mat2",2, false);
    THSFloatTensor_hspmm(result_->tensor, float(1), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & SparseCPUFloatType::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return  at::native::_sspaddmm_out_cpu(result, self, mat1, mat2, beta, alpha);
}

}
