#pragma once

#include <THC/THC.h>
#include <THCUNN/THCUNN.h>
#undef THNN_
#undef THCIndexTensor_
#include <THCS/THCS.h>
#undef THCIndexTensor_

#include "ATen/Tensor.h"
#include "ATen/TensorImpl.h"
#include "ATen/Context.h"
#include "ATen/TensorMethods.h"

namespace at {

struct SparseCUDADoubleTensor final : public TensorImpl {
public:
  explicit SparseCUDADoubleTensor(Context* context);
  SparseCUDADoubleTensor(Context* context, THCSDoubleTensor * tensor);
  virtual ~SparseCUDADoubleTensor();
  virtual const char * toString() const override;
  virtual IntList sizes() const override;
  virtual IntList strides() const override;
  virtual int64_t dim() const override;
  virtual Scalar localScalar() override;
  virtual void * unsafeGetTH(bool retain) override;
  virtual std::unique_ptr<Storage> storage() override;
  static const char * typeString();

//TODO(zach): sort of friend permissions later so this
// can be protected
public:
  THCSDoubleTensor * tensor;
  Context* context;
  friend struct SparseCUDADoubleType;
};

} // namespace at
