#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

// See the operations supported here:
// https://eigen.tuxfamily.org/dox-devel/group__QuickRefPage.html
#define EIGEN_MUL(x, y) ((x) * (y))
EIGEN_FUNCTOR(Mul, EIGEN_MUL, NumericTypes, SameTypeAsInput);
#undef EIGEN_MUL
}
