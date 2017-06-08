#pragma once

#include <$th_header>

#include "TensorLib/Tensor.h"
#include "TensorLib/Context.h"

namespace tlib {

struct ${Tensor} : public Tensor {
public:
  ${Tensor}(Context* context);
  ${Tensor}(Context* context, ${THTensor} * tensor);
  virtual ~${Tensor}();
  virtual const char * toString() const override;

  static const char * typeString();

//TODO(zach): sort of friend permissions later so this
// can be protected
public:
  ${THTensor} * tensor;
  Context* context;
  friend class ${Type};
};

} // namespace thpp
