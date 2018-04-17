// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/SparseCPUDoubleType.h"
#include "ATen/CPUDoubleStorage.h"
#include "ATen/SparseCPUDoubleTensor.h"
#include "ATen/CPUGenerator.h"
#include "ATen/SparseCPUByteTensor.h"
#include "ATen/SparseCPUIntTensor.h"
#include "ATen/SparseCPULongTensor.h"
#include "ATen/SparseCPUDoubleTensor.h"
#include "ATen/CPUDoubleTensor.h"
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

SparseCPUDoubleType::SparseCPUDoubleType(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
ScalarType SparseCPUDoubleType::scalarType() const {
  return ScalarType::Double;
}
Backend SparseCPUDoubleType::backend() const {
  return Backend::SparseCPU;
}
bool SparseCPUDoubleType::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool SparseCPUDoubleType::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool SparseCPUDoubleType::is_distributed() const { return false; }

std::unique_ptr<Storage> SparseCPUDoubleType::storage() const {
  return std::unique_ptr<Storage>(new CPUDoubleStorage(context));
}
std::unique_ptr<Storage> SparseCPUDoubleType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CPUDoubleStorage(context,size));
}
std::unique_ptr<Storage> SparseCPUDoubleType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CPUDoubleStorage(context,data,size,deleter));
}
std::unique_ptr<Storage> SparseCPUDoubleType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new CPUDoubleStorage(context, size, std::move(allocator)));
}
Tensor SparseCPUDoubleType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THSDoubleTensor_retain( (THSDoubleTensor*) th_pointer);
  return Tensor(new SparseCPUDoubleTensor(context,(THSDoubleTensor*)(th_pointer)), false);
}
std::unique_ptr<Storage> SparseCPUDoubleType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THDoubleStorage_retain( (THDoubleStorage*) th_pointer);
  return std::unique_ptr<Storage>(new CPUDoubleStorage(context, (THDoubleStorage*) th_pointer));
}
std::unique_ptr<Generator> SparseCPUDoubleType::generator() const {
  return std::unique_ptr<Generator>(new CPUGenerator(context));
}

const char * SparseCPUDoubleType::toString() const {
  return SparseCPUDoubleType::typeString();
}
TypeID SparseCPUDoubleType::ID() const {
  return TypeID::SparseCPUDouble;
}

std::size_t SparseCPUDoubleType::elementSizeInBytes() const {
  return sizeof(double);
}

const char * SparseCPUDoubleType::typeString() {
  return "SparseCPUDoubleType";
}

/* example
Tensor * SparseCPUDoubleType::add(Tensor & a, Tensor & b) {
  std::cout << "add SparseCPUDoubleTensor\n";
  return &a;
}
*/

Tensor SparseCPUDoubleType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    dim0 = maybe_wrap_dim(dim0, self_);
    dim1 = maybe_wrap_dim(dim1, self_);
    return Tensor((new SparseCPUDoubleTensor(context, THSDoubleTensor_newTranspose(self_->tensor, dim0, dim1)))->maybeScalar(self_->isScalar()),false);
}
Tensor SparseCPUDoubleType::t(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCPUDoubleTensor(context, THSDoubleTensor_newTranspose(self_->tensor, 0, 1)))->maybeScalar(self_->isScalar()),false);
}
Tensor SparseCPUDoubleType::clone(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCPUDoubleTensor(context, THSDoubleTensor_newClone(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & SparseCPUDoubleType::resize_as_(Tensor & self, const Tensor & the_template) const {
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    auto the_template_ = checked_cast_tensor<SparseCPUDoubleTensor>(the_template.pImpl,"the_template",2, false);
    THSDoubleTensor_resizeAs(self_->tensor, the_template_->tensor);
    self_->maybeScalar(the_template_->isScalar());
    return self;
}
Tensor SparseCPUDoubleType::norm(const Tensor & self, Scalar p) const {
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toDouble();
    // norm(value) for a sparse tensor returns a DENSE 0-dim tensor
    if (self.is_sparse()) {
      auto result = THSDoubleTensor_normall( self_->tensor, convert<double>(p_));
      return toBackend(toDense(backend())).tensor({}).fill_(result);
    }
    // aten_custom_call is followed by the generated call to normall
    return scalarTensor(convert<double>(THSDoubleTensor_normall(self_->tensor, p_)));
}
Tensor & SparseCPUDoubleType::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    auto result_ = checked_cast_tensor<SparseCPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toDouble();
    THSDoubleTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCPUDoubleType::pow(const Tensor & self, Scalar exponent) const {
    auto result_ = new SparseCPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toDouble();
    THSDoubleTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCPUDoubleType::zero_(Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    THSDoubleTensor_zero(self_->tensor);
    return self;
}
Tensor & SparseCPUDoubleType::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<SparseCPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto other_ = checked_cast_tensor<SparseCPUDoubleTensor>(other.pImpl,"other",3, false);
    THSDoubleTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCPUDoubleType::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add(self, Scalar(other), alpha);
    }
    auto result_ = new SparseCPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto other_ = checked_cast_tensor<SparseCPUDoubleTensor>(other.pImpl,"other",3, false);
    THSDoubleTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCPUDoubleType::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto other_ = checked_cast_tensor<SparseCPUDoubleTensor>(other.pImpl,"other",4, false);
    THSDoubleTensor_cadd(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & SparseCPUDoubleType::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<SparseCPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto other_ = checked_cast_tensor<SparseCPUDoubleTensor>(other.pImpl,"other",3, false);
    THSDoubleTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCPUDoubleType::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub(self, Scalar(other), alpha);
    }
    auto result_ = new SparseCPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto other_ = checked_cast_tensor<SparseCPUDoubleTensor>(other.pImpl,"other",3, false);
    THSDoubleTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCPUDoubleType::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto other_ = checked_cast_tensor<SparseCPUDoubleTensor>(other.pImpl,"other",4, false);
    THSDoubleTensor_csub(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & SparseCPUDoubleType::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<SparseCPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THSDoubleTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCPUDoubleType::mul(const Tensor & self, Scalar other) const {
    auto result_ = new SparseCPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THSDoubleTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCPUDoubleType::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<SparseCPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCPUDoubleTensor>(other.pImpl,"other",2, false);
    THSDoubleTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor SparseCPUDoubleType::s_mul(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul(self, Scalar(other));
    }
    auto result_ = new SparseCPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCPUDoubleTensor>(other.pImpl,"other",2, false);
    THSDoubleTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & SparseCPUDoubleType::mul_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THSDoubleTensor_mul(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & SparseCPUDoubleType::s_mul_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<SparseCPUDoubleTensor>(other.pImpl,"other",3, false);
    THSDoubleTensor_cmul(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & SparseCPUDoubleType::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<SparseCPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THSDoubleTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor SparseCPUDoubleType::div(const Tensor & self, Scalar other) const {
    auto result_ = new SparseCPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THSDoubleTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & SparseCPUDoubleType::div_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THSDoubleTensor_div(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor SparseCPUDoubleType::tensor(IntList size) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new SparseCPUDoubleTensor(context, THSDoubleTensor_newWithSize(size_, NULL)))->maybeScalar(size.size() == 0),false);
}
Tensor SparseCPUDoubleType::tensor() const {
    return Tensor((new SparseCPUDoubleTensor(context, THSDoubleTensor_new())),false);
}
Tensor SparseCPUDoubleType::sparse_coo_tensor(const Tensor & indices, const Tensor & values, IntList size) const {
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",1, false);
    auto values_ = checked_cast_tensor<CPUDoubleTensor>(values.pImpl,"values",2, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new SparseCPUDoubleTensor(context, THSDoubleTensor_newWithTensorAndSize(indices_->tensor, values_->tensor, size_)))->maybeScalar(size.size() == 0),false);
}
Tensor SparseCPUDoubleType::sparse_coo_tensor(const Tensor & indices, const Tensor & values) const {
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",1, false);
    auto values_ = checked_cast_tensor<CPUDoubleTensor>(values.pImpl,"values",2, false);
    return Tensor((new SparseCPUDoubleTensor(context, THSDoubleTensor_newWithTensor(indices_->tensor, values_->tensor)))->maybeScalar(indices_->isScalar() && values_->isScalar()),false);
}
Tensor & SparseCPUDoubleType::sparse_raw_resize_(Tensor & self, IntList size, int64_t nDimI, int64_t nDimV) const {
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
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
    THSDoubleTensor_rawResize(self_->tensor, nDimI, nDimV, (*size_).data);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor SparseCPUDoubleType::to_dense(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUDoubleTensor(context, THSDoubleTensor_toDense(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
int64_t SparseCPUDoubleType::_dimI(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    return THSDoubleTensor_nDimensionI(self_->tensor);
}
int64_t SparseCPUDoubleType::_dimV(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    return THSDoubleTensor_nDimensionV(self_->tensor);
}
int64_t SparseCPUDoubleType::_nnz(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    return THSDoubleTensor_nnz(self_->tensor);
}
Tensor SparseCPUDoubleType::coalesce(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    return Tensor((new SparseCPUDoubleTensor(context, THSDoubleTensor_newCoalesce(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
bool SparseCPUDoubleType::is_coalesced(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    return THSDoubleTensor_isCoalesced(self_->tensor);
}
Tensor SparseCPUDoubleType::_indices(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      // Empty tensor
      return self_->type().toScalarType(kLong).tensor({0});
    }
    return Tensor((new CPULongTensor(context, THSDoubleTensor_newIndices(self_->tensor))),false);
}
Tensor SparseCPUDoubleType::_values(const Tensor & self) const {
    auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUDoubleTensor(context, THSDoubleTensor_newValues(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & SparseCPUDoubleType::hspmm_out(Tensor & result, const Tensor & mat1, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<SparseCPUDoubleTensor>(result.pImpl,"result",0, false);
    auto mat1_ = checked_cast_tensor<SparseCPUDoubleTensor>(mat1.pImpl,"mat1",1, false);
    auto mat2_ = checked_cast_tensor<CPUDoubleTensor>(mat2.pImpl,"mat2",2, false);
    THSDoubleTensor_hspmm(result_->tensor, double(1), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor SparseCPUDoubleType::hspmm(const Tensor & mat1, const Tensor & mat2) const {
    auto result_ = new SparseCPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto mat1_ = checked_cast_tensor<SparseCPUDoubleTensor>(mat1.pImpl,"mat1",1, false);
    auto mat2_ = checked_cast_tensor<CPUDoubleTensor>(mat2.pImpl,"mat2",2, false);
    THSDoubleTensor_hspmm(result_->tensor, double(1), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & SparseCPUDoubleType::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return  at::native::_sspaddmm_out_cpu(result, self, mat1, mat2, beta, alpha);
}

}
