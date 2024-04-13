#ifndef CAFFE2_OPERATORS_LENGTHS_TILE_OP_H_
#define CAFFE2_OPERATORS_LENGTHS_TILE_OP_H_

#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class LengthsTileOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(LengthsTileOp);

  bool RunOnDevice() override {
    return true;
  }

  INPUT_TAGS(DATA, LENGTHS);

 private:
  Tensor lengths_host_{CPU};
  Tensor rowMappingHost_;
  Tensor rowMappingDevice_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LENGTHS_TILE_OP_H_
