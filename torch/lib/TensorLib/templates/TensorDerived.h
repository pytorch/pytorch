#pragma once

#include <$th_header>

#include "TensorLib/Tensor.h"
#include "TensorLib/Context.h"

namespace tlib {

struct ${Tensor} : public Tensor {
public:
  ${Tensor}(Context* context);
  virtual ~${Tensor}();
  virtual Type& type() const override;
  virtual const char * toString() const override;

  const char * typeString();

protected:
  ${THTensor} * tensor;
  Context* context;
  friend class ${Type};
};

} // namespace thpp
