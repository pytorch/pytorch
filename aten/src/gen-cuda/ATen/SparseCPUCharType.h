#pragma once
#include "ATen/Type.h"
#include "ATen/Context.h"
#include "ATen/TensorMethods.h"
#include "ATen/CheckGenerator.h"

#ifdef _MSC_VER
#ifdef Type
#undef Type
#endif
#endif

namespace at {

struct SparseCPUCharType final : public Type {
  explicit SparseCPUCharType(Context* context);
  virtual ScalarType scalarType() const override;
  virtual Backend backend() const override;
  virtual bool is_cuda() const override;
  virtual bool is_sparse() const override;
  virtual bool is_distributed() const override;
  virtual std::unique_ptr<Storage> storage() const override;
  virtual std::unique_ptr<Storage> storage(size_t size) const override;
  virtual std::unique_ptr<Storage> storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const override;
  virtual std::unique_ptr<Storage> storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const override;
  virtual std::unique_ptr<Generator> generator() const override;
  virtual const char * toString() const override;
  virtual std::size_t elementSizeInBytes() const override;
  virtual TypeID ID() const override;
  static const char * typeString();
  virtual std::unique_ptr<Storage> unsafeStorageFromTH(void * th_pointer, bool retain) const override;
  virtual Tensor unsafeTensorFromTH(void * th_pointer, bool retain) const override;

  // example
  // virtual Tensor * add(Tensor & a, Tensor & b) override;

  virtual Tensor & s_copy_(Tensor & self, const Tensor & src, bool non_blocking) const override;
  virtual Tensor transpose(const Tensor & self, int64_t dim0, int64_t dim1) const override;
  virtual Tensor t(const Tensor & self) const override;
  virtual Tensor clone(const Tensor & self) const override;
  virtual Tensor & resize_as_(Tensor & self, const Tensor & the_template) const override;
  virtual Tensor & pow_out(Tensor & result, const Tensor & self, Scalar exponent) const override;
  virtual Tensor pow(const Tensor & self, Scalar exponent) const override;
  virtual Tensor & zero_(Tensor & self) const override;
  virtual Tensor & s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const override;
  virtual Tensor s_add(const Tensor & self, const Tensor & other, Scalar alpha) const override;
  virtual Tensor & s_add_(Tensor & self, const Tensor & other, Scalar alpha) const override;
  virtual Tensor & s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const override;
  virtual Tensor s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const override;
  virtual Tensor & s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const override;
  virtual Tensor & mul_out(Tensor & result, const Tensor & self, Scalar other) const override;
  virtual Tensor mul(const Tensor & self, Scalar other) const override;
  virtual Tensor & s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const override;
  virtual Tensor s_mul(const Tensor & self, const Tensor & other) const override;
  virtual Tensor & mul_(Tensor & self, Scalar other) const override;
  virtual Tensor & s_mul_(Tensor & self, const Tensor & other) const override;
  virtual Tensor & div_out(Tensor & result, const Tensor & self, Scalar other) const override;
  virtual Tensor div(const Tensor & self, Scalar other) const override;
  virtual Tensor & div_(Tensor & self, Scalar other) const override;
  virtual Tensor tensor(IntList size) const override;
  virtual Tensor tensor() const override;
  virtual Tensor sparse_coo_tensor(const Tensor & indices, const Tensor & values, IntList size) const override;
  virtual Tensor sparse_coo_tensor(const Tensor & indices, const Tensor & values) const override;
  virtual Tensor & sparse_raw_resize_(Tensor & self, IntList size, int64_t nDimI, int64_t nDimV) const override;
  virtual Tensor to_dense(const Tensor & self) const override;
  virtual int64_t _dimI(const Tensor & self) const override;
  virtual int64_t _dimV(const Tensor & self) const override;
  virtual int64_t _nnz(const Tensor & self) const override;
  virtual Tensor coalesce(const Tensor & self) const override;
  virtual bool is_coalesced(const Tensor & self) const override;
  virtual Tensor _indices(const Tensor & self) const override;
  virtual Tensor _values(const Tensor & self) const override;
  virtual Tensor & hspmm_out(Tensor & result, const Tensor & mat1, const Tensor & mat2) const override;
  virtual Tensor hspmm(const Tensor & mat1, const Tensor & mat2) const override;
  virtual Tensor & sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const override;
};

} // namespace at
