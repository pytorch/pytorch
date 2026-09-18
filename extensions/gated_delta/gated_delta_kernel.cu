#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Naive per-(batch, head) recurrent kernel. Enough for a 500 MiB smoke
// (tiny T/H). Not a production FLA kernel.
template <typename T>
__global__ void gated_delta_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    const T* __restrict__ decay,
    const T* __restrict__ beta,
    T* __restrict__ out,
    int B,
    int Tlen,
    int H,
    int Kdim,
    int Vdim) {
  int bh = blockIdx.x;
  if (bh >= B * H) {
    return;
  }
  int b = bh / H;
  int h = bh % H;
  extern __shared__ unsigned char smem[];
  T* state = reinterpret_cast<T*>(smem); // [Kdim, Vdim]
  for (int i = threadIdx.x; i < Kdim * Vdim; i += blockDim.x) {
    state[i] = T(0);
  }
  __syncthreads();

  for (int t = 0; t < Tlen; ++t) {
    int dec_i = ((b * Tlen + t) * H + h);
    T gate = decay[dec_i];
    T alpha = static_cast<T>(expf(static_cast<float>(gate)));
    T bet = beta[dec_i];
    for (int i = threadIdx.x; i < Kdim * Vdim; i += blockDim.x) {
      state[i] = static_cast<T>(static_cast<float>(state[i]) * static_cast<float>(alpha));
    }
    __syncthreads();

    // kv_state[v] = sum_k state[k,v] * k[k]
    // then state -= k_beta[k] * kv_state[v]; state += k_beta[k] * v[v]
    const T* k_t = k + (((b * Tlen + t) * H + h) * Kdim);
    const T* v_t = v + (((b * Tlen + t) * H + h) * Vdim);
    const T* q_t = q + (((b * Tlen + t) * H + h) * Kdim);
    T* o_t = out + (((b * Tlen + t) * H + h) * Vdim);

    for (int v_i = threadIdx.x; v_i < Vdim; v_i += blockDim.x) {
      float kv = 0.f;
      for (int k_i = 0; k_i < Kdim; ++k_i) {
        kv += static_cast<float>(state[k_i * Vdim + v_i]) * static_cast<float>(k_t[k_i]);
      }
      float vval = static_cast<float>(v_t[v_i]);
      for (int k_i = 0; k_i < Kdim; ++k_i) {
        float kb = static_cast<float>(k_t[k_i]) * static_cast<float>(bet);
        float s = static_cast<float>(state[k_i * Vdim + v_i]);
        s = s - kb * kv + kb * vval;
        state[k_i * Vdim + v_i] = static_cast<T>(s);
      }
    }
    __syncthreads();

    for (int v_i = threadIdx.x; v_i < Vdim; v_i += blockDim.x) {
      float acc = 0.f;
      for (int k_i = 0; k_i < Kdim; ++k_i) {
        acc += static_cast<float>(q_t[k_i]) * static_cast<float>(state[k_i * Vdim + v_i]);
      }
      o_t[v_i] = static_cast<T>(acc);
    }
    __syncthreads();
  }
}

void gated_delta_rule_cuda_impl(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor decay,
    at::Tensor beta,
    at::Tensor out) {
  const int B = q.size(0);
  const int Tlen = q.size(1);
  const int H = q.size(2);
  const int Kdim = q.size(3);
  const int Vdim = v.size(3);
  const int threads = 32;
  const int blocks = B * H;
  const size_t smem = static_cast<size_t>(Kdim * Vdim) * q.element_size();
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      q.scalar_type(),
      "gated_delta_rule_cuda",
      [&] {
        gated_delta_kernel<scalar_t><<<blocks, threads, smem, stream>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            decay.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            B,
            Tlen,
            H,
            Kdim,
            Vdim);
      });
}
