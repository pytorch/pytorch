#ifndef CAFFE2_OPERATOR_GLU_OP_H_
#define CAFFE2_OPERATOR_GLU_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {
template <typename T, class Context>
class GluOp final : public Operator<Context> {
 public:
  GluOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() {
    auto& X = Input(0);
    auto* Y = Output(0);
    const int M = X.size_to_dim(X.ndim() - 1);
    const int N = X.dim32(X.ndim() - 1);
    vector<TIndex> Yshape;
    Yshape.insert(Yshape.end(), X.dims().begin(), X.dims().end());
    Yshape[Yshape.size() - 1] = N / 2;
    Y->Resize(Yshape);
    ComputeGlu(M, N / 2, X.template data<T>(), Y->template mutable_data<T>());
    return true;
  }

 protected:
  void ComputeGlu(const int M, const int N, const T* X, T* output);
};
} // namespace caffe2

#endif // CAFFE2_OPERATOR_GLU_OP_H_
