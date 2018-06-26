#include "gftrl_op.h"
#include <iostream>

namespace caffe2 {

template <class T>
inline T sgn(const T x) {
  return (x == 0 ? 0 : (x < 0 ? -1 : 1));
}

// Computes one coordinate
template <typename T>

inline void gftrl_compute(
    const T w,
    const T n,
    const T z,
    const T g,
    T& nw,
    T& nn,
    T& nz,
    const T z_norm,
    const GFtrlParams<T>& params) {
  auto new_n = n + g * g;
  auto sigma = (sqrt(new_n) - sqrt(n)) * params.alphaInv;
  nn = new_n;
  nz = z + g - sigma * w;
  // update the weight
  if (z_norm > params.lambda1) {
    nw = nz * (params.lambda1 / z_norm - 1) /
        ((params.beta + sqrt(new_n)) * params.alphaInv + params.lambda2);
  } else {
    assert(false);
    nw = 0.0;
  }
}

// TODO(dzhulgakov): implement SIMD-based version
template <typename Context, typename T>
void gftrl_update(
    int OutputDim, // 512
    int InputDim, // # of groups = InputDim  1559
    const T* w,
    const T* nz,
    const T* g,
    T* new_w,
    T* new_nz,
    const GFtrlParams<T>& params,
    Context* /*context*/) {
  // TODO(cxj): use OMP when it is reliable
  // #pragma omp parallel for
  int N = OutputDim * InputDim;

  int zero_count = 0;
  for (auto j = 0; j < InputDim; ++j) {
    // for each group
    // 1. compute z_norm = |z^j|_2
    // 2. if z_norm <= params.lambda1
    T z_norm = 0.0;
    for (auto i = 0; i < OutputDim; ++i) {
      int idx = i * InputDim + j;
      auto new_n = nz[idx * 2] + g[idx] * g[idx];
      auto sigma = (sqrt(new_n) - sqrt(nz[idx * 2])) * params.alphaInv;
      auto new_z = nz[idx * 2 + 1] + g[idx] - sigma * w[idx];
      z_norm = z_norm + new_z * new_z;
    }

    z_norm = sqrt(z_norm);
    // std::cout << "col " << j << " z_norm " << z_norm << std::endl;
    if (z_norm <= params.lambda1) {
      zero_count++;
      for (auto i = 0; i < OutputDim; ++i) {
        new_w[i * InputDim + j] = 0;
      }
    } else {
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
            params);
      }
    }
  }
}

template <typename T, typename Context>
bool GFtrlOp<T, Context>::RunOnDevice() {
  // run time learning rate override
  if (ALPHA < InputSize()) {
    CAFFE_ENFORCE_EQ(Input(ALPHA).size(), 1, "alpha should be real-valued");
    params_.alphaInv = 1.0 / *(Input(ALPHA).template data<T>());
  }
  // Input(GRAD) is of shape [OutputDim, InputDim].
  // std::cout << "!!!!!" << Input(GRAD).size() << " " << Input(GRAD).dims()
  //           << std::endl;
  CAFFE_ENFORCE_EQ(Input(GRAD).size(), Input(VAR).size());
  CAFFE_ENFORCE_EQ(Input(GRAD).size() * 2, Input(N_Z).size());
  Output(OUTPUT_VAR)->ResizeLike(Input(VAR));
  Output(OUTPUT_N_Z)->ResizeLike(Input(N_Z));
  gftrl_update<Context>(
      Input(GRAD).dim(0),
      Input(GRAD).dim(1),
      Input(VAR).template data<T>(),
      Input(N_Z).template data<T>(),
      Input(GRAD).template data<T>(),
      Output(OUTPUT_VAR)->template mutable_data<T>(),
      Output(OUTPUT_N_Z)->template mutable_data<T>(),
      params_,
      &context_);
  return true;
}

template <typename T>
template <typename SIndex>
void GSparseFtrlOp<T>::DoRun() {
  auto* var = Output(OUTPUT_VAR);
  auto* n_z = Output(OUTPUT_N_Z);
  auto& indices = Input(INDICES);
  auto& grad = Input(GRAD);
  CAFFE_ENFORCE_EQ(&Input(VAR), var, "In place operation is required");
  CAFFE_ENFORCE_EQ(&Input(N_Z), n_z, "In place operation is required");
  TIndex M = var->size();
  TIndex N = var->dim(0);
  TIndex block_size = M / N;
  TIndex K = indices.size();
  DCHECK_EQ(M * 2, n_z->size());
  DCHECK_EQ(grad.size(), K * block_size);
  T* w = var->template mutable_data<T>();
  T* nz = n_z->template mutable_data<T>();
  const SIndex* idxs = indices.template data<SIndex>();
  const T* g = grad.template data<T>();

  // Input(GRAD) is of shape [OutputDim, InputDim].

  // TODO(cxj): use OMP when it is reliable
  // #pragma omp parallel for
  T z_norm = 0.0;
  if (block_size == 1) {
    for (TIndex i = 0; i < K; ++i) {
      SIndex idx = idxs[i];
      z_norm = z_norm + nz[idx * 2 + 1] * nz[idx * 2 + 1];
    }
    z_norm = sqrt(z_norm);
  }
  for (TIndex i = 0; i < K; ++i) {
    SIndex idx = idxs[i];
    DCHECK(0 <= idx && idx < N)
        << "Index out of bounds: " << idx << ", range 0 to " << N;
    if (block_size == 1) {
      gftrl_compute(
          w[idx],
          nz[idx * 2],
          nz[idx * 2 + 1],
          g[i],
          w[idx],
          nz[idx * 2],
          nz[idx * 2 + 1],
          z_norm,
          params_);
    } else {
      TIndex x = block_size * idx;
      gftrl_update(
          block_size,
          1,
          w + x,
          nz + x * 2,
          g + i * block_size,
          w + x,
          nz + x * 2,
          params_,
          &context_);
    }
  }
}

namespace {
REGISTER_CPU_OPERATOR(GFtrl, GFtrlOp<float, CPUContext>);
OPERATOR_SCHEMA(GFtrl).NumInputs(3, 4).NumOutputs(2).AllowInplace({{0, 0},
                                                                   {1, 1}});
SHOULD_NOT_DO_GRADIENT(GFtrl);

REGISTER_CPU_OPERATOR(GSparseFtrl, GSparseFtrlOp<float>);
OPERATOR_SCHEMA(GSparseFtrl)
    .NumInputs(4, 5)
    .NumOutputs(2)
    .EnforceInplace({{0, 0}, {1, 1}});
SHOULD_NOT_DO_GRADIENT(GSparseFtrl);
} // namespace

} // namespace caffe2
