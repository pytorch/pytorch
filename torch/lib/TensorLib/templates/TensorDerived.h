#pragma once

$th_headers

#include "TensorLib/Tensor.h"
#include "TensorLib/TensorImpl.h"
#include "TensorLib/Context.h"

namespace tlib {

struct ${Tensor} : public TensorImpl {
public:
  ${Tensor}(Context* context);
  ${Tensor}(Context* context, ${THTensor} * tensor);
  virtual ~${Tensor}();
  virtual const char * toString() const override;
  virtual IntList sizes() override;
  virtual IntList strides() override;
  static const char * typeString();

//TODO(zach): sort of friend permissions later so this
// can be protected
public:
  ${THTensor} * tensor;
  Context* context;
  friend class ${Type};
};

} // namespace thpp
