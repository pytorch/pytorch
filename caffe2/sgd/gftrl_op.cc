#include "gftrl_op.h"

namespace caffe2 {

// Computes one coordinate
template <typename T>

inline void gftrl_compute(
    const T& w,
    const T& n,
    const T& z,
    const T& g,
    T& nw,
    T& nn,
    T& nz,
    const T& z_norm,
    const int OutputDim,
    const GFtrlParams<T>& params) {
  auto new_n = n + g * g;
  auto sigma = (sqrt(new_n) - sqrt(n)) * params.alphaInv;
  nn = new_n;
  nz = z + g - sigma * w;
  // update the weight
  if (z_norm > params.lambda1 * std::sqrt(OutputDim)) {
    nw = nz * (params.lambda1 * std::sqrt(OutputDim) / z_norm - 1) /
        ((params.beta + sqrt(new_n)) * params.alphaInv + params.lambda2);
  } else {
    nw = 0.0;
  }
}

template <typename Context, typename T>
void gftrl_update(
    int OutputDim, // # of output nodes
    int InputDim, // # of input features
    const T* w,
    const T* nz,
    const T* g,
    T* new_w,
    T* new_nz,
    const GFtrlParams<T>& params,
    Context* /*context*/) {
  for (auto j = 0; j < InputDim; ++j) {
    T z_norm = 0.0;
    for (auto i = 0; i < OutputDim; ++i) {
      int idx = i * InputDim + j;
      auto new_n = nz[idx * 2] + g[idx] * g[idx];
      auto sigma = (sqrt(new_n) - sqrt(nz[idx * 2])) * params.alphaInv;
      auto new_z = nz[idx * 2 + 1] + g[idx] - sigma * w[idx];
      z_norm = z_norm + new_z * new_z;
    }

    z_norm = sqrt(z_norm);
    for (auto i = 0; i < OutputDim; ++i) {
      int idx = i * InputDim + j;
      gftrl_compute(
          w[idx],
          nz[idx * 2],
          nz[idx * 2 + 1],
          g[idx],
          new_w[idx],
          new_nz[idx * 2],
          new_nz[idx * 2 + 1],
          z_norm,
          OutputDim,
          params);
    }
  }
}

template <typename T, typename Context>
bool GFtrlOp<T, Context>::RunOnDevice() {
  // run time learning rate override
  if (ALPHA < InputSize()) {
    CAFFE_ENFORCE_EQ(Input(ALPHA).numel(), 1, "alpha should be real-valued");
    params_.alphaInv = 1.0 / *(Input(ALPHA).template data<T>());
  }

  CAFFE_ENFORCE_EQ(Input(GRAD).numel(), Input(VAR).numel());
  CAFFE_ENFORCE_EQ(Input(GRAD).numel() * 2, Input(N_Z).numel());
  Output(OUTPUT_VAR)->ResizeLike(Input(VAR));
  Output(OUTPUT_N_Z)->ResizeLike(Input(N_Z));
  gftrl_update<Context>(
      Input(GRAD).size(0), // # of output nodes
      Input(GRAD).numel() / Input(GRAD).size(0), // # of input features
      Input(VAR).template data<T>(),
      Input(N_Z).template data<T>(),
      Input(GRAD).template data<T>(),
      Output(OUTPUT_VAR)->template mutable_data<T>(),
      Output(OUTPUT_N_Z)->template mutable_data<T>(),
      params_,
      &context_);
  return true;
}

namespace {
REGISTER_CPU_OPERATOR(GFtrl, GFtrlOp<float, CPUContext>);
OPERATOR_SCHEMA(GFtrl).NumInputs(3, 4).NumOutputs(2).AllowInplace({{0, 0},
                                                                   {1, 1}});
SHOULD_NOT_DO_GRADIENT(GFtrl);

} // namespace

} // namespace caffe2
