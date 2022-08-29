#include "ftrl_op.h"

namespace caffe2 {

template <class T>
inline T sgn(const T x) {
  return (x == 0 ? 0 : (x < 0 ? -1 : 1));
}

template <typename T>
inline void ftrl_compute(
    const T w,
    const T n,
    const T z,
    const T g,
    T& nw,
    T& nn,
    T& nz,
    const FtrlParams<T>& params) {
  auto new_n = n + g * g;
  auto sigma = (sqrt(new_n) - sqrt(n)) * params.alphaInv;
  nn = new_n;
  nz = z + g - sigma * w;
  // update the weight
  if (std::abs(nz) > params.lambda1) {
    nw = (params.lambda1 * sgn(nz) - nz) /
        ((params.beta + sqrt(new_n)) * params.alphaInv + params.lambda2);
  } else {
    nw = 0.0;
  }
}

// TODO(dzhulgakov): implement SIMD-based version
template <typename Context, typename T>
void ftrl_update(
    int N,
    const T* w,
    const T* nz,
    const T* g,
    T* new_w,
    T* new_nz,
    const FtrlParams<T>& params,
    Context* /*context*/) {
  // TODO(cxj): use OMP when it is reliable
  // #pragma omp parallel for
  for (auto i = 0; i < N; ++i) {
    ftrl_compute(
        w[i],
        nz[i * 2],
        nz[i * 2 + 1],
        g[i],
        new_w[i],
        new_nz[i * 2],
        new_nz[i * 2 + 1],
        params);
  }
}

template <typename T, typename Context>
bool FtrlOp<T, Context>::RunOnDevice() {
  // run time learning rate override
  if (ALPHA < InputSize()) {
    CAFFE_ENFORCE_EQ(Input(ALPHA).numel(), 1, "alpha should be real-valued");
    params_.alphaInv = 1.0 / *(Input(ALPHA).template data<T>());
  }
  CAFFE_ENFORCE_EQ(Input(GRAD).numel(), Input(VAR).numel());
  CAFFE_ENFORCE_EQ(Input(GRAD).numel() * 2, Input(N_Z).numel());
  Output(OUTPUT_VAR)->ResizeLike(Input(VAR));
  Output(OUTPUT_N_Z)->ResizeLike(Input(N_Z));
  ftrl_update<Context>(
      Input(GRAD).numel(),
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
void SparseFtrlOp<T>::DoRun() {
  auto* var = Output(OUTPUT_VAR);
  auto* n_z = Output(OUTPUT_N_Z);
  auto& indices = Input(INDICES);
  auto& grad = Input(GRAD);
  CAFFE_ENFORCE_EQ(&Input(VAR), var, "In place operation is required");
  CAFFE_ENFORCE_EQ(&Input(N_Z), n_z, "In place operation is required");
  int64_t M = var->numel();
  int64_t N = var->size(0);
  int64_t block_size = M / N;
  int64_t K = indices.numel();
  TORCH_DCHECK_EQ(M * 2, n_z->numel());
  TORCH_DCHECK_EQ(grad.numel(), K * block_size);
  T* w = var->template mutable_data<T>();
  T* nz = n_z->template mutable_data<T>();
  const SIndex* idxs = indices.template data<SIndex>();
  const T* g = grad.template data<T>();

  // TODO(cxj): use OMP when it is reliable
  // #pragma omp parallel for
  for (int64_t i = 0; i < K; ++i) {
    SIndex idx = idxs[i];
    DCHECK(0 <= idx && idx < N) << "Index out of bounds: " << idx
                                << ", range 0 to " << N;
    if (block_size == 1) {
      ftrl_compute(
          w[idx],
          nz[idx * 2],
          nz[idx * 2 + 1],
          g[i],
          w[idx],
          nz[idx * 2],
          nz[idx * 2 + 1],
          params_);
    } else {
      int64_t x = block_size * idx;
      ftrl_update(
          block_size,
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
REGISTER_CPU_OPERATOR(Ftrl, FtrlOp<float, CPUContext>);
OPERATOR_SCHEMA(Ftrl).NumInputs(3, 4).NumOutputs(2).AllowInplace({{0, 0},
                                                                  {1, 1}});
SHOULD_NOT_DO_GRADIENT(Ftrl);

REGISTER_CPU_OPERATOR(SparseFtrl, SparseFtrlOp<float>);
OPERATOR_SCHEMA(SparseFtrl)
    .NumInputs(4, 5)
    .NumOutputs(2)
    .EnforceInplace({{0, 0}, {1, 1}});
SHOULD_NOT_DO_GRADIENT(SparseFtrl);
}

}
