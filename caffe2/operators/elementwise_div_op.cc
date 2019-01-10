#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

// See the operations supported here:
// https://eigen.tuxfamily.org/dox-devel/group__QuickRefPage.html
#define EIGEN_DIV(x, y) ((x) / (y))
EIGEN_FUNCTOR(Div, EIGEN_DIV, NumericTypes, SameTypeAsInput);
#undef EIGEN_DIV

void ElementWiseDivide(
    CPUContext& /* unused context */,
    const int n,
    float* dXdata,
    float* dYdata,
    const float* dZdata,
    const float* Ydata,
    const float* Zdata) {
  ConstEigenVectorArrayMap<float> dZdataVec(dZdata, n);
  ConstEigenVectorArrayMap<float> YdataVec(Ydata, n);
  ConstEigenVectorArrayMap<float> ZdataVec(Zdata, n);
  EigenVectorArrayMap<float>(dXdata, n) = dZdataVec / YdataVec;
  EigenVectorArrayMap<float>(dYdata, n) = - (dZdataVec * ZdataVec) / YdataVec;
}

REGISTER_CPU_OPERATOR(DivGradient, DivGradientOp<CPUContext>);
}
