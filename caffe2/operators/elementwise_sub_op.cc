#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

// See the operations supported here:
// https://eigen.tuxfamily.org/dox-devel/group__QuickRefPage.html
#define EIGEN_SUB(x, y) ((x) - (y))
EIGEN_FUNCTOR(Sub, EIGEN_SUB, NumericTypes, SameTypeAsInput);
#undef EIGEN_SUB
}
