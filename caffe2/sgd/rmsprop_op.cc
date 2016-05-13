#include "rmsprop_op.h"

namespace caffe2 {

namespace {
REGISTER_CPU_OPERATOR(RmsProp, RmsPropOp<float, CPUContext>);
OPERATOR_SCHEMA(RmsProp).NumInputs(4).NumOutputs(3).AllowInplace({{0, 0},
                                                                  {1, 1},
                                                                  {2, 2}});
SHOULD_NOT_DO_GRADIENT(RmsProp);
}

}
