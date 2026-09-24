#include <ATen/native/mps/kernels/Eig.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>

using namespace metal;

namespace {

inline float cabs(float2 z) {
  return ::metal::precise::sqrt(z.x * z.x + z.y * z.y);
}

// Principal square root of a complex number.
inline float2 csqrt(float2 z) {
  const float r = cabs(z);
  if (r == 0.0) {
    return float2(0.0, 0.0);
  }
  const float a = ::metal::precise::sqrt(0.5 * (r + z.x));
  float b = ::metal::precise::sqrt(0.5 * (r - z.x));
  if (z.y < 0.0) {
    b = -b;
  }
  return float2(a, b);
}

struct Givens {
  float c;
  float2 s;
};

// Rotation that maps (f, g) to (r, 0), with a real cosine.
inline Givens givens(float2 f, float2 g) {
  Givens rot;
  const float absf = cabs(f);
  if (absf == 0.0) {
    rot.c = 0.0;
    rot.s = float2(1.0, 0.0);
    return rot;
  }
  const float r = ::metal::precise::sqrt(absf * absf + cabs(g) * cabs(g));
  rot.c = absf / r;
  rot.s = c10::metal::mul(f / absf, c10::metal::conj(g)) / r;
  return rot;
}

} // anonymous namespace

// One SIMD group per matrix, lane i owning row or column i of the working
// matrices (kEigMaxDim is the SIMD width). The QR sweeps are sequential, so
// the lanes split the work inside each reflector and rotation rather than
// across them. Updates alternate between column-owned and row-owned phases,
// with a barrier at every switch. The host keeps `n` within kEigMaxDim and
// falls back to CPU beyond it, the same way the MPS eigh and svd kernels do.
kernel void eig_qr(
    constant float2* A [[buffer(0)]],
    device float2* values [[buffer(1)]],
    device float2* vectors [[buffer(2)]],
    device int* info [[buffer(3)]],
    constant EigParams& params [[buffer(4)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float2 T[kEigMaxDim * kEigMaxDim];
  threadgroup float2 Q[kEigMaxDim * kEigMaxDim];
  threadgroup float2 Y[kEigMaxDim * kEigMaxDim];
  threadgroup float2 v[kEigMaxDim];
  threadgroup float rot_c[kEigMaxDim];
  threadgroup float2 rot_s[kEigMaxDim];

  const int n = params.n;
  const bool compute_vectors = params.compute_vectors;
  const int me = static_cast<int>(lane);
  const bool active = me < n;
  const long mat_offset = static_cast<long>(batch_idx) * n * n;
  constant float2* Ain = A + mat_offset;

  if (active) {
    for (int i = 0; i < n; ++i) {
      T[i * n + me] = Ain[i * n + me];
      Q[i * n + me] = float2(i == me ? 1.0 : 0.0, 0.0);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float eps = 1.1920929e-7;

  // ---- Reduction to upper Hessenberg form by Householder reflectors ----
  for (int k = 0; k + 2 < n; ++k) {
    const float2 x = me > k && active ? T[me * n + k] : float2(0.0, 0.0);
    const float normx = ::metal::precise::sqrt(simd_sum(dot(x, x)));
    if (normx == 0.0) {
      continue;
    }

    const float2 x0 = T[(k + 1) * n + k];
    const float absx0 = cabs(x0);
    // Choose the reflection that moves away from x0 to avoid cancellation.
    const float2 phase = absx0 == 0.0 ? float2(1.0, 0.0) : x0 / absx0;
    const float2 alpha = -phase * normx;
    const float2 vi = me == k + 1 ? x - alpha : x;
    const float vnorm2 = simd_sum(dot(vi, vi));
    if (vnorm2 == 0.0) {
      continue;
    }
    const float tau = 2.0 / vnorm2;
    v[me] = vi;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // T <- H T, lane j owning column j
    if (active && me >= k) {
      float2 s = float2(0.0, 0.0);
      for (int i = k + 1; i < n; ++i) {
        s = c10::metal::fma(c10::metal::conj(v[i]), T[i * n + me], s);
      }
      s = s * tau;
      for (int i = k + 1; i < n; ++i) {
        T[i * n + me] = T[i * n + me] - c10::metal::mul(v[i], s);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // T <- T H and Q <- Q H, lane i owning row i
    if (active) {
      float2 s = float2(0.0, 0.0);
      for (int j = k + 1; j < n; ++j) {
        s = c10::metal::fma(T[me * n + j], v[j], s);
      }
      s = s * tau;
      for (int j = k + 1; j < n; ++j) {
        T[me * n + j] =
            T[me * n + j] - c10::metal::mul(s, c10::metal::conj(v[j]));
      }
      if (compute_vectors) {
        s = float2(0.0, 0.0);
        for (int j = k + 1; j < n; ++j) {
          s = c10::metal::fma(Q[me * n + j], v[j], s);
        }
        s = s * tau;
        for (int j = k + 1; j < n; ++j) {
          Q[me * n + j] =
              Q[me * n + j] - c10::metal::mul(s, c10::metal::conj(v[j]));
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Frobenius norm, which the unitary reduction and QR sweeps preserve.
  float normsq = 0.0;
  if (active) {
    for (int i = 0; i < n; ++i) {
      normsq += dot(T[i * n + me], T[i * n + me]);
    }
  }
  const float fro_norm = ::metal::precise::sqrt(simd_sum(normsq));

  // ---- Shifted QR iteration, deflating from the bottom ----
  // All lanes read the same entries here, so they take the same branches.
  int status = 0;
  int hi = n - 1;
  int iter = 0;
  const int max_iter = 60 * n + 60;

  while (hi > 0) {
    // Find the top of the active block by looking for a negligible subdiagonal.
    int lo = hi;
    while (lo > 0) {
      // The local test keeps small eigenvalues accurate, but inside a cluster
      // it can sit below rounding noise forever, so once the block stalls
      // accept anything below eps * ||A||, which is still backward stable.
      const float scale =
          cabs(T[(lo - 1) * n + (lo - 1)]) + cabs(T[lo * n + lo]);
      const float ref = iter >= 10 ? ::metal::max(scale, fro_norm) : scale;
      if (cabs(T[lo * n + (lo - 1)]) <= eps * ref) {
        break;
      }
      lo -= 1;
    }

    if (lo == hi) {
      hi -= 1;
      iter = 0;
      continue;
    }

    if (iter >= max_iter) {
      status = hi + 1;
      break;
    }

    // Wilkinson shift from the trailing 2x2 block, with a periodic exceptional
    // shift to break the cycles that a plain Wilkinson shift can fall into.
    const float2 a = T[(hi - 1) * n + (hi - 1)];
    const float2 b = T[(hi - 1) * n + hi];
    const float2 c = T[hi * n + (hi - 1)];
    const float2 d = T[hi * n + hi];
    float2 mu;
    if (iter > 0 && iter % 10 == 0) {
      mu = d + float2(cabs(c), 0.0);
    } else {
      // The eigenvalues are d + p +- r. Taking the one nearer d as
      // d - bc / (p +- r), with the larger denominator, avoids the
      // cancellation of forming tr^2 - det when the two nearly coincide.
      const float2 p = (a - d) * 0.5;
      const float2 bc = c10::metal::mul(b, c);
      const float2 r = csqrt(c10::metal::mul(p, p) + bc);
      const float2 den = cabs(p + r) >= cabs(p - r) ? p + r : p - r;
      mu = cabs(den) == 0.0 ? d : d - c10::metal::div(bc, den);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Explicit shifted QR: first reduce the block to triangular with Givens
    // rotations from the left, then apply them back from the right. Keeping
    // the two passes separate matters -- interleaving them would fill in
    // below the subdiagonal and invalidate the next rotation.
    // Left pass, lane j owning column j.
    if (lo > 0 && me == lo - 1) {
      T[lo * n + (lo - 1)] = float2(0.0, 0.0);
    }
    if (me >= lo && me <= hi) {
      T[me * n + me] = T[me * n + me] - mu;
    }
    for (int j = lo; j < hi; ++j) {
      // Column j is lane j's, so it broadcasts the pair the rotation zeroes.
      // Lanes past n read scratch that is in bounds and never used.
      const float2 f = simd_shuffle(T[j * n + me], ushort(j));
      const float2 g = simd_shuffle(T[(j + 1) * n + me], ushort(j));
      const Givens rot = givens(f, g);
      if (me == 0) {
        rot_c[j] = rot.c;
        rot_s[j] = rot.s;
      }
      if (active && me >= j) {
        const float2 t0 = T[j * n + me];
        const float2 t1 = T[(j + 1) * n + me];
        T[j * n + me] = t0 * rot.c + c10::metal::mul(rot.s, t1);
        T[(j + 1) * n + me] =
            t1 * rot.c - c10::metal::mul(c10::metal::conj(rot.s), t0);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Right pass, lane i owning row i.
    if (active) {
      for (int j = lo; j < hi; ++j) {
        const float c_j = rot_c[j];
        const float2 s_j = rot_s[j];
        if (me <= hi) {
          const float2 t0 = T[me * n + j];
          const float2 t1 = T[me * n + (j + 1)];
          T[me * n + j] = t0 * c_j + c10::metal::mul(c10::metal::conj(s_j), t1);
          T[me * n + (j + 1)] = t1 * c_j - c10::metal::mul(s_j, t0);
        }
        if (compute_vectors) {
          const float2 q0 = Q[me * n + j];
          const float2 q1 = Q[me * n + (j + 1)];
          Q[me * n + j] = q0 * c_j + c10::metal::mul(c10::metal::conj(s_j), q1);
          Q[me * n + (j + 1)] = q1 * c_j - c10::metal::mul(s_j, q0);
        }
      }
      if (me >= lo && me <= hi) {
        T[me * n + me] = T[me * n + me] + mu;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    iter += 1;
  }

  if (active) {
    values[static_cast<long>(batch_idx) * n + me] = T[me * n + me];
  }
  if (me == 0) {
    info[batch_idx] = status;
  }

  if (!compute_vectors || status != 0) {
    return;
  }

  // ---- Eigenvectors of the triangular factor, mapped back through Q ----
  // Lane col solves for eigenvector col on its own, in column col of Y.
  float colsum = 0.0;
  if (active) {
    for (int i = 0; i <= me; ++i) {
      colsum += cabs(T[i * n + me]);
    }
  }
  float tnorm = simd_sum(colsum);
  if (tnorm == 0.0) {
    tnorm = 1.0;
  }
  if (!active) {
    return;
  }

  const int col = me;
  const float2 lambda = T[col * n + col];
  for (int i = 0; i < n; ++i) {
    Y[i * n + col] = float2(0.0, 0.0);
  }
  Y[col * n + col] = float2(1.0, 0.0);

  for (int k = col - 1; k >= 0; --k) {
    float2 s = float2(0.0, 0.0);
    for (int j = k + 1; j <= col; ++j) {
      s = c10::metal::fma(T[k * n + j], Y[j * n + col], s);
    }
    float2 denom = T[k * n + k] - lambda;
    // A defective or repeated eigenvalue makes the diagonal difference
    // vanish; perturb it the way LAPACK's ztrevc does.
    if (cabs(denom) < eps * tnorm) {
      denom = float2(eps * tnorm, 0.0);
    }
    Y[k * n + col] = c10::metal::div(-s, denom);
  }

  float scale = 0.0;
  for (int i = 0; i <= col; ++i) {
    scale = ::metal::max(scale, cabs(Y[i * n + col]));
  }
  if (scale > 0.0) {
    for (int i = 0; i <= col; ++i) {
      Y[i * n + col] = Y[i * n + col] / scale;
    }
  }

  device float2* out_col = vectors + mat_offset;
  float norm = 0.0;
  for (int r = 0; r < n; ++r) {
    float2 acc = float2(0.0, 0.0);
    for (int j = 0; j <= col; ++j) {
      acc = c10::metal::fma(Q[r * n + j], Y[j * n + col], acc);
    }
    out_col[r * n + col] = acc;
    norm += dot(acc, acc);
  }
  norm = ::metal::precise::sqrt(norm);
  if (norm > 0.0) {
    for (int r = 0; r < n; ++r) {
      out_col[r * n + col] = out_col[r * n + col] / norm;
    }
  }
}
