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

// One threadgroup per matrix, and the algorithm runs serially inside it: the
// QR sweeps are inherently sequential, and the batch is what provides the
// parallelism. The host keeps `n` within kEigMaxDim and falls back to CPU
// beyond it, the same way the MPS eigh and svd kernels do.
kernel void eig_qr(
    constant float2* A [[buffer(0)]],
    device float2* values [[buffer(1)]],
    device float2* vectors [[buffer(2)]],
    device int* info [[buffer(3)]],
    constant EigParams& params [[buffer(4)]],
    uint batch_idx [[threadgroup_position_in_grid]]) {
  threadgroup float2 T[kEigMaxDim * kEigMaxDim];
  threadgroup float2 Q[kEigMaxDim * kEigMaxDim];
  threadgroup float2 v[kEigMaxDim];
  threadgroup float2 y[kEigMaxDim];
  threadgroup float rot_c[kEigMaxDim];
  threadgroup float2 rot_s[kEigMaxDim];

  const int n = params.n;
  const long mat_offset = static_cast<long>(batch_idx) * n * n;
  constant float2* Ain = A + mat_offset;

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      T[i * n + j] = Ain[i * n + j];
      Q[i * n + j] = float2(i == j ? 1.0 : 0.0, 0.0);
    }
  }

  const float eps = 1.1920929e-7;

  // ---- Reduction to upper Hessenberg form by Householder reflectors ----
  for (int k = 0; k + 2 < n; ++k) {
    float normx = 0.0;
    for (int i = k + 1; i < n; ++i) {
      const float a = cabs(T[i * n + k]);
      normx += a * a;
    }
    normx = ::metal::precise::sqrt(normx);
    if (normx == 0.0) {
      continue;
    }

    const float2 x0 = T[(k + 1) * n + k];
    const float absx0 = cabs(x0);
    // Choose the reflection that moves away from x0 to avoid cancellation.
    const float2 phase = absx0 == 0.0 ? float2(1.0, 0.0) : x0 / absx0;
    const float2 alpha = -phase * normx;

    for (int i = k + 1; i < n; ++i) {
      v[i] = T[i * n + k];
    }
    v[k + 1] = v[k + 1] - alpha;

    float vnorm2 = 0.0;
    for (int i = k + 1; i < n; ++i) {
      const float a = cabs(v[i]);
      vnorm2 += a * a;
    }
    if (vnorm2 == 0.0) {
      continue;
    }
    const float tau = 2.0 / vnorm2;

    // T <- H T
    for (int j = k; j < n; ++j) {
      float2 s = float2(0.0, 0.0);
      for (int i = k + 1; i < n; ++i) {
        s = c10::metal::fma(c10::metal::conj(v[i]), T[i * n + j], s);
      }
      s = s * tau;
      for (int i = k + 1; i < n; ++i) {
        T[i * n + j] = T[i * n + j] - c10::metal::mul(v[i], s);
      }
    }
    // T <- T H
    for (int i = 0; i < n; ++i) {
      float2 s = float2(0.0, 0.0);
      for (int j = k + 1; j < n; ++j) {
        s = c10::metal::fma(T[i * n + j], v[j], s);
      }
      s = s * tau;
      for (int j = k + 1; j < n; ++j) {
        T[i * n + j] =
            T[i * n + j] - c10::metal::mul(s, c10::metal::conj(v[j]));
      }
    }
    // Q <- Q H
    for (int i = 0; i < n; ++i) {
      float2 s = float2(0.0, 0.0);
      for (int j = k + 1; j < n; ++j) {
        s = c10::metal::fma(Q[i * n + j], v[j], s);
      }
      s = s * tau;
      for (int j = k + 1; j < n; ++j) {
        Q[i * n + j] =
            Q[i * n + j] - c10::metal::mul(s, c10::metal::conj(v[j]));
      }
    }
  }

  // ---- Shifted QR iteration, deflating from the bottom ----
  int status = 0;
  int hi = n - 1;
  int iter = 0;
  const int max_iter = 60 * n + 60;

  while (hi > 0) {
    // Find the top of the active block by looking for a negligible subdiagonal.
    int lo = hi;
    while (lo > 0) {
      const float scale =
          cabs(T[(lo - 1) * n + (lo - 1)]) + cabs(T[lo * n + lo]);
      if (cabs(T[lo * n + (lo - 1)]) <= eps * (scale == 0.0 ? 1.0 : scale)) {
        T[lo * n + (lo - 1)] = float2(0.0, 0.0);
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
      const float2 tr = (a + d) * 0.5;
      const float2 det = c10::metal::mul(a, d) - c10::metal::mul(b, c);
      const float2 disc = csqrt(c10::metal::mul(tr, tr) - det);
      const float2 mu1 = tr + disc;
      const float2 mu2 = tr - disc;
      mu = cabs(mu1 - d) < cabs(mu2 - d) ? mu1 : mu2;
    }

    for (int i = lo; i <= hi; ++i) {
      T[i * n + i] = T[i * n + i] - mu;
    }

    // Explicit shifted QR: first reduce the block to triangular with Givens
    // rotations from the left, then apply them back from the right. Keeping
    // the two passes separate matters -- interleaving them would fill in
    // below the subdiagonal and invalidate the next rotation.
    for (int j = lo; j < hi; ++j) {
      const Givens rot = givens(T[j * n + j], T[(j + 1) * n + j]);
      rot_c[j] = rot.c;
      rot_s[j] = rot.s;
      for (int col = j; col < n; ++col) {
        const float2 t0 = T[j * n + col];
        const float2 t1 = T[(j + 1) * n + col];
        T[j * n + col] = t0 * rot.c + c10::metal::mul(rot.s, t1);
        T[(j + 1) * n + col] =
            t1 * rot.c - c10::metal::mul(c10::metal::conj(rot.s), t0);
      }
    }
    for (int j = lo; j < hi; ++j) {
      const float c_j = rot_c[j];
      const float2 s_j = rot_s[j];
      for (int row = 0; row <= hi; ++row) {
        const float2 t0 = T[row * n + j];
        const float2 t1 = T[row * n + (j + 1)];
        T[row * n + j] = t0 * c_j + c10::metal::mul(c10::metal::conj(s_j), t1);
        T[row * n + (j + 1)] = t1 * c_j - c10::metal::mul(s_j, t0);
      }
      for (int row = 0; row < n; ++row) {
        const float2 q0 = Q[row * n + j];
        const float2 q1 = Q[row * n + (j + 1)];
        Q[row * n + j] = q0 * c_j + c10::metal::mul(c10::metal::conj(s_j), q1);
        Q[row * n + (j + 1)] = q1 * c_j - c10::metal::mul(s_j, q0);
      }
    }

    for (int i = lo; i <= hi; ++i) {
      T[i * n + i] = T[i * n + i] + mu;
    }
    iter += 1;
  }

  for (int i = 0; i < n; ++i) {
    values[static_cast<long>(batch_idx) * n + i] = T[i * n + i];
  }
  info[batch_idx] = status;

  if (params.compute_vectors == 0 || status != 0) {
    return;
  }

  // ---- Eigenvectors of the triangular factor, mapped back through Q ----
  float tnorm = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      tnorm += cabs(T[i * n + j]);
    }
  }
  if (tnorm == 0.0) {
    tnorm = 1.0;
  }

  for (int col = 0; col < n; ++col) {
    const float2 lambda = T[col * n + col];
    for (int i = 0; i < n; ++i) {
      y[i] = float2(0.0, 0.0);
    }
    y[col] = float2(1.0, 0.0);

    for (int k = col - 1; k >= 0; --k) {
      float2 s = float2(0.0, 0.0);
      for (int j = k + 1; j <= col; ++j) {
        s = c10::metal::fma(T[k * n + j], y[j], s);
      }
      float2 denom = T[k * n + k] - lambda;
      // A defective or repeated eigenvalue makes the diagonal difference
      // vanish; perturb it the way LAPACK's ztrevc does.
      if (cabs(denom) < eps * tnorm) {
        denom = float2(eps * tnorm, 0.0);
      }
      y[k] = c10::metal::div(-s, denom);
    }

    float scale = 0.0;
    for (int i = 0; i <= col; ++i) {
      scale = ::metal::max(scale, cabs(y[i]));
    }
    if (scale > 0.0) {
      for (int i = 0; i <= col; ++i) {
        y[i] = y[i] / scale;
      }
    }

    device float2* out_col = vectors + mat_offset;
    float norm = 0.0;
    for (int r = 0; r < n; ++r) {
      float2 acc = float2(0.0, 0.0);
      for (int j = 0; j <= col; ++j) {
        acc = c10::metal::fma(Q[r * n + j], y[j], acc);
      }
      out_col[r * n + col] = acc;
      const float m = cabs(acc);
      norm += m * m;
    }
    norm = ::metal::precise::sqrt(norm);
    if (norm > 0.0) {
      for (int r = 0; r < n; ++r) {
        out_col[r * n + col] = out_col[r * n + col] / norm;
      }
    }
  }
}
