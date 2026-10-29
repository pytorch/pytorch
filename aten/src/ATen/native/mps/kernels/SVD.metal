#include <ATen/native/mps/kernels/LinearAlgebra.h>
#include <ATen/native/mps/kernels/SVD.h>
#include <c10/metal/reduction_utils.h>
#include <metal_stdlib>

using namespace metal;

template <typename T>
kernel void svd_jacobi(
    device const T* A [[buffer(0)]],
    device T* U [[buffer(1)]],
    device svd_real_t<T>* S [[buffer(2)]],
    device T* V [[buffer(3)]],
    device T* Vacc [[buffer(4)]], // rotation accumulator when V not staged
    device int* info [[buffer(5)]],
    constant SvdParams& params [[buffer(6)]],
    threadgroup T* Atg [[threadgroup(0)]],
    threadgroup T* Vtg [[threadgroup(1)]],
    uint3 thread_pos [[thread_position_in_threadgroup]],
    uint3 tpg [[threads_per_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint32_t tid = thread_pos.x;
  const uint32_t group_size = tpg.x;
  const uint32_t m = params.m;
  const uint32_t n = params.n;
  const uint32_t batch_idx = tg_pos.x;
  const uint32_t kSimd = c10::metal::simdgroup_size;
  const uint32_t num_sg = group_size / kSimd;

  device const T* A_b = A + batch_idx * m * n;
  device T* U_b = U + batch_idx * params.u_bstride;
  device T* V_b = V + batch_idx * params.v_bstride;
  device T* Vacc_b = Vacc + batch_idx * n * n;

  float amax = 0;
  for (uint32_t i = tid; i < m * n * sizeof(T) / 4; i += group_size) {
    amax = max(amax, fabs(reinterpret_cast<device const float*>(A_b)[i]));
  }
  amax = c10::metal::simd_max(amax);
  threadgroup float* maxima = reinterpret_cast<threadgroup float*>(Atg);
  if (simd_lane == 0) {
    maxima[simd_group] = amax;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  amax = simd_lane < num_sg ? maxima[simd_lane] : 0.0f;
  amax = c10::metal::simd_max(amax);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // 2^+/-kMaxScaleExp are both normal fp32 (Metal flushes subnormals).
  constexpr int kMaxScaleExp = 1 - ::metal::numeric_limits<float>::min_exponent;
  const int scale_exp = clamp(ilogb(amax), -kMaxScaleExp, kMaxScaleExp);
  const float scale = ldexp(1.0f, -scale_exp);

  // Stage A column-major so each lane's row access is contiguous.
  for (uint32_t idx = tid; idx < m * n; idx += group_size) {
    uint32_t row = idx / n, col = idx % n;
    Atg[col * m + row] = A_b[idx] * scale;
  }
  if (params.compute_uv) {
    if (params.stage_v) {
      for (uint32_t i = tid; i < n * n; i += group_size) {
        uint32_t row = i / n, col = i % n;
        // NB: float2(1.0) broadcasts to (1,1); use svd_one()/T(0) for a real
        // 1/0.
        Vtg[col * n + row] = (row == col) ? svd_one(T(0)) : T(0);
      }
    } else {
      for (uint32_t i = tid; i < n * n; i += group_size) {
        Vacc_b[i] = (i / n == i % n) ? svd_one(T(0)) : T(0);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

  constexpr auto eps = numeric_limits<float>::epsilon();
  // Concurrent SIMD-groups flag "I rotated"; a plain flag races, so use an
  // atomic.
  threadgroup atomic_uint any_rotation;

  // Round-robin tournament pairing (closed-form circle method): pad to even ne;
  // each sweep is ne-1 rounds of ne/2 disjoint pairs; index >= n is phantom.
  const uint32_t ne = n + (n & 1u);
  const uint32_t n_pairs = ne / 2;

  uint32_t sweep = 0;
  for (; sweep < params.max_sweeps; ++sweep) {
    if (tid == 0) {
      atomic_store_explicit(&any_rotation, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint32_t round = 0; round < ne - 1; ++round) {
      for (uint32_t k = simd_group; k < n_pairs; k += num_sg) {
        uint32_t p = (k == 0) ? 0u : ((k - 1 + round) % (ne - 1)) + 1u;
        uint32_t kq = ne - 1 - k;
        uint32_t q = (kq == 0) ? 0u : ((kq - 1 + round) % (ne - 1)) + 1u;
        bool act = !(p >= n || q >= n);
        if (act && p > q) {
          uint32_t tmp = p;
          p = q;
          q = tmp;
        }

        threadgroup T* colP = Atg + p * m;
        threadgroup T* colQ = Atg + q * m;
        float app = 0, aqq = 0;
        T apq_acc = T(0);
        if (act) {
          for (uint32_t i = simd_lane; i < m; i += kSimd) {
            T vp = colP[i];
            T vq = colQ[i];
            app += svd_abs2(vp);
            aqq += svd_abs2(vq);
            apq_acc += svd_conjmul(vp, vq);
          }
        }
        app = c10::metal::simd_sum(app);
        aqq = c10::metal::simd_sum(aqq);
        apq_acc = svd_simd_sum(apq_acc);

        if (!act) {
          continue;
        }
        float apq_abs = precise::sqrt(svd_abs2(apq_acc));
        float off = precise::sqrt(app * aqq);
        // Test relative column correlation even for small column norms:
        // an absolute epsilon cutoff leaves small columns non-orthogonal.
        if (off == 0.0f || apq_abs <= params.tol * off) {
          continue;
        }
        if (simd_lane == 0) {
          atomic_store_explicit(&any_rotation, 1u, memory_order_relaxed);
        }
        T phi = (apq_abs > 0) ? (apq_acc * (1.0f / apq_abs)) : svd_one(T(0));
        float tau = (aqq - app) / (2 * apq_abs);
        float t = (tau >= 0 ? 1.0f : -1.0f) /
            (fabs(tau) + precise::sqrt(1 + tau * tau));
        float c = 1 / precise::sqrt(1 + t * t);
        float s = c * t;
        T cphi = svd_conj(phi);
        for (uint32_t i = simd_lane; i < m; i += kSimd) {
          T vp = colP[i];
          T vq = colQ[i];
          colP[i] = c * vp - svd_mul(cphi, s * vq);
          colQ[i] = svd_mul(phi, s * vp) + c * vq;
        }
        if (params.compute_uv) {
          if (params.stage_v) {
            threadgroup T* vP = Vtg + p * n;
            threadgroup T* vQ = Vtg + q * n;
            for (uint32_t i = simd_lane; i < n; i += kSimd) {
              T vp = vP[i];
              T vq = vQ[i];
              vP[i] = c * vp - svd_mul(cphi, s * vq);
              vQ[i] = svd_mul(phi, s * vp) + c * vq;
            }
          } else {
            device T* vP = Vacc_b + p * n;
            device T* vQ = Vacc_b + q * n;
            for (uint32_t i = simd_lane; i < n; i += kSimd) {
              T vp = vP[i];
              T vq = vQ[i];
              vP[i] = c * vp - svd_mul(cphi, s * vq);
              vQ[i] = svd_mul(phi, s * vp) + c * vq;
            }
          }
        }
      }
      // Barrier scope must be a compile-time constant (runtime mem_flags
      // crashes the AGX compiler)
      if (params.stage_v) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      } else {
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
      }
    }

    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    threadgroup uint32_t do_break = 0;
    if (tid == 0) {
      do_break =
          atomic_load_explicit(&any_rotation, memory_order_relaxed) == 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (do_break) {
      break;
    }
  }

  // n <= 90 (host staging gate); 96 gives headroom.
  threadgroup float sig[96];
  threadgroup uint32_t ord[96];
  for (uint32_t j = simd_group; j < n; j += num_sg) {
    threadgroup T* colj = Atg + j * m;
    float norm_sq = 0;
    for (uint32_t i = simd_lane; i < m; i += kSimd) {
      norm_sq += svd_abs2(colj[i]);
    }
    float sigma = precise::sqrt(c10::metal::simd_sum(norm_sq));
    if (simd_lane == 0) {
      sig[j] = sigma;
      ord[j] = j;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    for (uint32_t a = 0; a < n; ++a) {
      uint32_t best = a;
      for (uint32_t b = a + 1; b < n; ++b) {
        if (sig[ord[b]] > sig[ord[best]])
          best = b;
      }
      uint32_t tmp = ord[a];
      ord[a] = ord[best];
      ord[best] = tmp;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Emit column j from source ord[j]. Transposed run swaps left/right targets;
  // right vectors written as Vh rows are conjugated (Vh = V^H), left vectors
  // not.
  for (uint32_t j = simd_group; j < n; j += num_sg) {
    uint32_t src = ord[j];
    float sigma = sig[src];
    if (simd_lane == 0) {
      S[batch_idx * n + j] = ldexp(sigma, scale_exp);
    }
    float inv = sigma > 0.0f ? (1 / sigma) : 0.0f;
    threadgroup T* colsrc = Atg + src * m;
    if (!params.transposed) {
      for (uint32_t i = simd_lane; i < m; i += kSimd) {
        U_b[j * params.u_ld + i] = inv * colsrc[i];
      }
      if (params.compute_uv) {
        threadgroup T* vsrc = Vtg + src * n;
        for (uint32_t c = simd_lane; c < n; c += kSimd) {
          T v = params.stage_v ? vsrc[c] : Vacc_b[src * n + c];
          V_b[c * params.v_ld + j] = svd_conj(v);
        }
      }
    } else {
      for (uint32_t i = simd_lane; i < m; i += kSimd) {
        V_b[i * params.v_ld + j] = svd_conj(inv * colsrc[i]);
      }
      if (params.compute_uv) {
        threadgroup T* vsrc = Vtg + src * n;
        for (uint32_t c = simd_lane; c < n; c += kSimd) {
          U_b[j * params.u_ld + c] =
              params.stage_v ? vsrc[c] : Vacc_b[src * n + c];
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_device);

  // The Jacobi update forms the range factor's columns as (A V)_j / sigma_j,
  // which collapses to 0 when sigma_j <= eps, leaving U (or V, when transposed)
  // non-orthonormal for rank-deficient inputs. Replace those null columns with
  // an orthonormal basis of the complement of the emitted columns (two-pass
  // Gram-Schmidt of canonical vectors), matching LAPACK gesdd. Columns are
  // sorted by descending sigma, so the null ones are a contiguous tail. One
  // simd-group runs it (each lane owns a strided slice of the working column,
  // so the projections are simd_sum reductions needing no barriers); the rare
  // degenerate path. Reuses Atg as scratch.
  if (params.compute_uv && simd_group == 0) {
    device T* out = params.transposed ? V_b : U_b;
    const uint32_t ld = params.transposed ? params.v_ld : params.u_ld;
    // U_b is column-major (elem i of col c at out[c*ld + i]); the transposed
    // run emits V_b row-major (out[i*ld + c]), so index columns accordingly.
    const uint32_t col_off = params.transposed ? 1u : ld;
    const uint32_t elem_step = params.transposed ? ld : 1u;
    // Relative rank cutoff: sigma_j at or below the Jacobi noise floor
    // (~m*eps*sigma_max) is numerically zero, so its column is arbitrary and
    // gets completed. An absolute eps would keep noise-amplified columns.
    const float thresh = eps * sig[ord[0]] * static_cast<float>(m);
    uint32_t rank = 0;
    while (rank < n && sig[ord[rank]] > thresh) {
      ++rank;
    }
    // Accept a candidate as a null-space column when its squared residual after
    // orthogonalization clears this: loose (well above the fp32 roundoff
    // floor), but enough that the canonicals span the complement. Cf.
    // Rutishauser/DGKS Gram-Schmidt reorthogonalization.
    constexpr float kIndepThreshSq = 1e-2f;
    threadgroup T* col = Atg;
    uint32_t cand = 0;
    for (uint32_t j = rank; j < n; ++j) {
      while (cand < m) {
        for (uint32_t i = simd_lane; i < m; i += kSimd) {
          col[i] = (i == cand) ? svd_one(T(0)) : T(0);
        }
        ++cand;
        for (uint32_t pass = 0; pass < 2; ++pass) {
          for (uint32_t l = 0; l < j; ++l) {
            device T* cl = out + l * col_off;
            T partial = T(0);
            for (uint32_t i = simd_lane; i < m; i += kSimd) {
              partial += svd_conjmul(cl[i * elem_step], col[i]);
            }
            T dot = svd_simd_sum(partial);
            for (uint32_t i = simd_lane; i < m; i += kSimd) {
              col[i] -= svd_mul(dot, cl[i * elem_step]);
            }
          }
        }
        float partial_n = 0;
        for (uint32_t i = simd_lane; i < m; i += kSimd) {
          partial_n += svd_abs2(col[i]);
        }
        const float nrm_sq = c10::metal::simd_sum(partial_n);
        if (nrm_sq > kIndepThreshSq) {
          const float inv = 1.0f / precise::sqrt(nrm_sq);
          device T* cj = out + j * col_off;
          for (uint32_t i = simd_lane; i < m; i += kSimd) {
            cj[i * elem_step] = col[i] * inv;
          }
          break;
        }
      }
    }
  }

  if (tid == 0) {
    // NaN/Inf never triggers a rotation, so flag info to raise like the CPU
    // path.
    bool nonfinite = false;
    for (uint32_t j = 0; j < n; ++j) {
      if (!isfinite(sig[j])) {
        nonfinite = true;
        break;
      }
    }
    info[batch_idx] = (nonfinite || sweep >= params.max_sweeps)
        ? static_cast<int>(sweep + 1)
        : 0;
  }
}

#define REGISTER_SVD_JACOBI(T)                             \
  template [[host_name("svd_jacobi_" #T)]]                 \
  kernel void svd_jacobi<T>(                               \
      device const T* A [[buffer(0)]],                     \
      device T* U [[buffer(1)]],                           \
      device svd_real_t<T>* S [[buffer(2)]],               \
      device T* V [[buffer(3)]],                           \
      device T* Vacc [[buffer(4)]],                        \
      device int* info [[buffer(5)]],                      \
      constant SvdParams& params [[buffer(6)]],            \
      threadgroup T* Atg [[threadgroup(0)]],               \
      threadgroup T* Vtg [[threadgroup(1)]],               \
      uint3 thread_pos [[thread_position_in_threadgroup]], \
      uint3 tpg [[threads_per_threadgroup]],               \
      uint3 tg_pos [[threadgroup_position_in_grid]],       \
      uint simd_lane [[thread_index_in_simdgroup]],        \
      uint simd_group [[simdgroup_index_in_threadgroup]]);

REGISTER_SVD_JACOBI(float);
REGISTER_SVD_JACOBI(float2);
