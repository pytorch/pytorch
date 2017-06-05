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

protected:
  ${THTensor} * tensor;
  Context* context;
};

} // namespace thpp
