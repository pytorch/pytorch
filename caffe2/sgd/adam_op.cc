#include "adam_op.h"

namespace caffe2 {

namespace {
REGISTER_CPU_OPERATOR(Adam, AdamOp<float, CPUContext>);
OPERATOR_SCHEMA(Adam).NumInputs(5).NumOutputs(3).AllowInplace({{0, 0},
                                                               {1, 1},
                                                               {2, 2}});
SHOULD_NOT_DO_GRADIENT(Adam);
}

}
