#pragma once

#include <$th_header>

#include "Tensor.h"
#include "Context.h"

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
