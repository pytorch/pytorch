#include <ATen/native/mps/kernels/LinearAlgebra.h>
#include <ATen/native/mps/kernels/SVD.h>
#include <c10/metal/reduction_utils.h>
#include <metal_stdlib>

using namespace metal;

template <typename T>
kernel void eigh_jacobi(
    device T* A [[buffer(0)]],
    device svd_real_t<T>* W [[buffer(1)]],
    device T* Q [[buffer(2)]],
    device int* info [[buffer(3)]],
    constant EighParams& params [[buffer(4)]],
    threadgroup T* Atg [[threadgroup(0)]],
    threadgroup T* Qtg [[threadgroup(1)]],
    uint3 thread_pos [[thread_position_in_threadgroup]],
    uint3 tpg [[threads_per_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint32_t tid = thread_pos.x;
  const uint32_t group_size = tpg.x;
  const uint32_t n = params.n;
  const uint32_t batch_idx = tg_pos.x;
  const uint32_t kSimd = c10::metal::simdgroup_size;
  const uint32_t num_sg = group_size / kSimd;
  const bool compute_v = params.compute_v;

  device T* A_b = A + batch_idx * n * n;
  device T* Q_b = Q + batch_idx * n * n;

  // Stage A into Atg, symmetrizing from the selected UPLO triangle (input may
  // be non-Hermitian otherwise); two-sided Jacobi needs an exactly Hermitian
  // matrix.
  const bool upper = params.upper;
  for (uint32_t i = tid; i < n * n; i += group_size) {
    uint32_t row = i % n, col = i / n;
    if (row == col) {
      Atg[i] = svd_from_real(T(0), svd_real_part(A_b[i]));
    } else {
      bool in_upper = row < col;
      if (in_upper == upper) {
        Atg[i] = A_b[i];
      } else {
        Atg[i] = svd_conj(A_b[col + row * n]);
      }
    }
  }
  if (compute_v) {
    for (uint32_t i = tid; i < n * n; i += group_size) {
      uint32_t row = i % n, col = i / n;
      Qtg[i] = (row == col) ? svd_one(T(0)) : T(0);
    }
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

  threadgroup float cbuf[48];
  threadgroup T sbuf[48];
  threadgroup uint32_t pbuf[48], qbuf[48];
  // Concurrent SIMD-groups flag "I rotated"; a plain flag races, so use an
  // atomic.
  threadgroup atomic_uint any_rotation;

  const uint32_t ne = n + (n & 1u);
  const uint32_t n_pairs = ne / 2;

  threadgroup float red_diag[16];
  threadgroup float red_off[16];

  uint32_t sweep = 0;
  for (; sweep < params.max_sweeps; ++sweep) {
    if (tid == 0) {
      atomic_store_explicit(&any_rotation, 0u, memory_order_relaxed);
    }
    {
      float ld = 0.0f;
      float lo = 0.0f;
      for (uint32_t i = tid; i < n * n; i += group_size) {
        uint32_t row = i % n, col = i / n;
        float a2 = svd_abs2(Atg[i]);
        if (row == col) {
          ld = max(ld, a2);
        } else {
          lo = max(lo, a2);
        }
      }
      ld = c10::metal::simd_max(ld);
      lo = c10::metal::simd_max(lo);
      if (simd_lane == 0) {
        red_diag[simd_group] = ld;
        red_off[simd_group] = lo;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float g2 = 0.0f;
    float o2 = 0.0f;
    for (uint32_t s = 0; s < num_sg; ++s) {
      g2 = max(g2, red_diag[s]);
      o2 = max(o2, red_off[s]);
    }
    const float gscale = precise::sqrt(g2);
    if (o2 <= params.tol * params.tol * g2) {
      break;
    }

    for (uint32_t round = 0; round < ne - 1; ++round) {
      for (uint32_t k = simd_group; k < n_pairs; k += num_sg) {
        uint32_t p = (k == 0) ? 0u : ((k - 1 + round) % (ne - 1)) + 1u;
        uint32_t kq = ne - 1 - k;
        uint32_t q = (kq == 0) ? 0u : ((kq - 1 + round) % (ne - 1)) + 1u;
        bool act = !(p >= n || q >= n || p == q);
        if (act && p > q) {
          uint32_t t = p;
          p = q;
          q = t;
        }
        if (!act) {
          if (simd_lane == 0) {
            pbuf[k] = n;
            qbuf[k] = n;
          }
          continue;
        }
        float app = svd_real_part(Atg[p * n + p]);
        float aqq = svd_real_part(Atg[q * n + q]);
        T apq = Atg[q * n + p];
        float apq_abs = precise::sqrt(svd_abs2(apq));
        float off = precise::sqrt(::metal::fabs(app * aqq));
        float c = 1.0f;
        T s = T(0);
        float thresh = max(params.tol * off, params.tol * gscale);
        bool rotate = apq_abs > thresh + 1e-30f;
        if (rotate) {
          if (simd_lane == 0) {
            atomic_store_explicit(&any_rotation, 1u, memory_order_relaxed);
          }
          T phi = apq * (1.0f / apq_abs);
          float tau = (aqq - app) / (2.0f * apq_abs);
          float t = (tau >= 0 ? 1.0f : -1.0f) /
              (fabs(tau) + precise::sqrt(1.0f + tau * tau));
          c = 1.0f / precise::sqrt(1.0f + t * t);
          float sreal = c * t;
          s = svd_mul(phi, svd_from_real(T(0), sreal));
        }
        if (simd_lane == 0) {
          cbuf[k] = c;
          sbuf[k] = s;
          pbuf[k] = rotate ? p : n;
          qbuf[k] = q;
        }
        if (!rotate) {
          continue;
        }
        T cs = svd_conj(s);
        threadgroup T* colP = Atg + p * n;
        threadgroup T* colQ = Atg + q * n;
        for (uint32_t i = simd_lane; i < n; i += kSimd) {
          T ap = colP[i], aq = colQ[i];
          colP[i] = c * ap - svd_mul(cs, aq);
          colQ[i] = svd_mul(s, ap) + c * aq;
        }
        if (compute_v) {
          threadgroup T* qP = Qtg + p * n;
          threadgroup T* qQ = Qtg + q * n;
          for (uint32_t i = simd_lane; i < n; i += kSimd) {
            T qp = qP[i], qq = qQ[i];
            qP[i] = c * qp - svd_mul(cs, qq);
            qQ[i] = svd_mul(s, qp) + c * qq;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (uint32_t k = simd_group; k < n_pairs; k += num_sg) {
        uint32_t p = pbuf[k], q = qbuf[k];
        if (p >= n) {
          continue;
        }
        float c = cbuf[k];
        T s = sbuf[k];
        T cs = svd_conj(s);
        for (uint32_t col = simd_lane; col < n; col += kSimd) {
          T ap = Atg[col * n + p], aq = Atg[col * n + q];
          Atg[col * n + p] = c * ap - svd_mul(s, aq);
          Atg[col * n + q] = svd_mul(cs, ap) + c * aq;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (atomic_load_explicit(&any_rotation, memory_order_relaxed) == 0u) {
      break;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  threadgroup float wv[96];
  threadgroup uint32_t ord[96];
  for (uint32_t j = simd_group; j < n; j += num_sg) {
    if (simd_lane == 0) {
      wv[j] = svd_real_part(Atg[j * n + j]);
      ord[j] = j;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint32_t j = simd_group; j < n; j += num_sg) {
    float vj = wv[j];
    uint32_t cnt = 0;
    for (uint32_t k = simd_lane; k < n; k += kSimd) {
      float vk = wv[k];
      cnt += (vk < vj || (vk == vj && k < j)) ? 1u : 0u;
    }
    cnt = c10::metal::simd_sum(cnt);
    if (simd_lane == 0) {
      ord[cnt] = j;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint32_t j = simd_group; j < n; j += num_sg) {
    uint32_t src = ord[j];
    if (simd_lane == 0) {
      W[batch_idx * n + j] = wv[src];
    }
    if (compute_v) {
      threadgroup T* qs = Qtg + src * n;
      for (uint32_t i = simd_lane; i < n; i += kSimd) {
        Q_b[j * n + i] = qs[i];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_device);

  if (tid == 0) {
    // NaN/Inf never triggers a rotation, so flag info to raise like the CPU
    // path.
    bool nonfinite = false;
    for (uint32_t j = 0; j < n; ++j) {
      if (!isfinite(wv[j])) {
        nonfinite = true;
        break;
      }
    }
    info[batch_idx] = (nonfinite || sweep >= params.max_sweeps)
        ? static_cast<int>(sweep + 1)
        : 0;
  }
}

#define REGISTER_EIGH_JACOBI(T)                            \
  template [[host_name("eigh_jacobi_" #T)]]                \
  kernel void eigh_jacobi<T>(                              \
      device T * A [[buffer(0)]],                          \
      device svd_real_t<T> * W [[buffer(1)]],              \
      device T * Q [[buffer(2)]],                          \
      device int* info [[buffer(3)]],                      \
      constant EighParams& params [[buffer(4)]],           \
      threadgroup T* Atg [[threadgroup(0)]],               \
      threadgroup T* Qtg [[threadgroup(1)]],               \
      uint3 thread_pos [[thread_position_in_threadgroup]], \
      uint3 tpg [[threads_per_threadgroup]],               \
      uint3 tg_pos [[threadgroup_position_in_grid]],       \
      uint simd_lane [[thread_index_in_simdgroup]],        \
      uint simd_group [[simdgroup_index_in_threadgroup]]);

REGISTER_EIGH_JACOBI(float);
REGISTER_EIGH_JACOBI(float2);
