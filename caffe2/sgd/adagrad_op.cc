#include "adagrad_op.h"

namespace caffe2 {

namespace {
REGISTER_CPU_OPERATOR(Adagrad, AdagradOp<float, CPUContext>);
OPERATOR_SCHEMA(Adagrad).NumInputs(3).NumOutputs(2).AllowInplace({{0, 0},
                                                                  {1, 1}});
SHOULD_NOT_DO_GRADIENT(Adagrad);
}

}
