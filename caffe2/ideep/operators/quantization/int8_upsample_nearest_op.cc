#include <caffe2/ideep/ideep_utils.h>

namespace caffe2 {

USE_IDEEP_DEF_ALIASES();

class IDEEPInt8UpsampleNearestOp final : public IDEEPOperator {
public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPInt8UpsampleNearestOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        upsample_scale_(this->template GetSingleArgument<int>("scale", 2)) {
  }
  virtual ~IDEEPInt8UpsampleNearestOp() {}

  bool RunOnDevice() override {
    const auto& X = Input(INPUT);
    CAFFE_ENFORCE(X.get_public_format() == iformat::nhwc);

    auto Y_dims = X.get_dims();
    Y_dims[2] *= upsample_scale_;
    Y_dims[3] *= upsample_scale_;
    auto* Y = Output(OUTPUT);
    Y->init({Y_dims, X.get_data_type(), iformat::nhwc});
    Y->set_scale(X.get_scale());

    int d0, d1, d2, d3;
    if (X.ndims() == 3) {
      d0 = 1;
      d1 = Y_dims[1];
      d2 = Y_dims[2];
      d3 = Y_dims[0];
    } else {
      d0 = Y_dims[0];
      d1 = Y_dims[2];
      d2 = Y_dims[3];
      d3 = Y_dims[1];
    }

    const auto *Xdata = static_cast<int8_t*>(X.get_data_handle());
    auto* Ydata = static_cast<int8_t*>(Y->get_data_handle());
    int scaled_d1 = d1 / upsample_scale_;
    int scaled_d2 = d2 / upsample_scale_;

#ifdef _OPENMP
#if (_OPENMP >= 201307)
#pragma omp parallel for simd
#else
#pragma omp parallel for
#endif
#endif
    for (int i = 0; i < d0; ++i) {
      for (int j = 0; j < d1; ++j) {
        for (int u = 0; u < d2; ++u) {
          for (int v = 0; v < d3; ++v) {
            int ii = ((i * d1 + j) * d2 + u) * d3 + v;
            int scaled_j = j / upsample_scale_;
            int scaled_u = u / upsample_scale_;
            int ipidx = ((i * scaled_d1 + scaled_j) * scaled_d2 + scaled_u) * d3 + v;
            Ydata[ii] = Xdata[ipidx];
          }
        }
      }
    }
    return true;
  }

 protected:
  int upsample_scale_;

  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT);
};

REGISTER_IDEEP_OPERATOR_WITH_ENGINE(Int8UpsampleNearest, DNNLOWP, IDEEPInt8UpsampleNearestOp);

} // namespace caffe2

