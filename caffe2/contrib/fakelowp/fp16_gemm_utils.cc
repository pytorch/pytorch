#include "caffe2/contrib/fakelowp/fp16_gemm_utils.h"
#include <fbgemm/FbgemmConvert.h>
#include <fbgemm/FbgemmFP16.h>
#include <glog/logging.h>
#include <immintrin.h>
#include "caffe2/core/context.h"
#include "caffe2/utils/math.h"

C10_DECLARE_bool(caffe2_fbgemm_fake_fp16_clamp);

namespace caffe2 {

// dimA(before transpose) = M x N, dimA (after transpose) = N x M.
void transpose(const float* A, std::vector<float>& A_trans, int M, int N) {
  CAFFE_ENFORCE_EQ(M * N, A_trans.size());
  fbgemm::transpose_simd(M, N, A, N, A_trans.data(), M);
}

void custom_fp16_gemm_with_trans(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int m,
    const int k,
    const int n,
    const float* A,
    const float* B,
    const float beta,
    float* C,
    const bool use_acc_fp16,
    const bool use_temp_accumulator) {
  switch (trans_A) {
    case CblasNoTrans: {
      switch (trans_B) {
        case CblasNoTrans: {
          // A * B
          custom_fp16_gemm(
              m, k, n, A, B, beta, C, use_acc_fp16, use_temp_accumulator);
          break;
        }
        case CblasTrans: {
          // A * B_trans
          std::vector<float> B_trans(n * k);
          transpose(B, B_trans, n, k);
          custom_fp16_gemm(
              m,
              k,
              n,
              A,
              B_trans.data(),
              beta,
              C,
              use_acc_fp16,
              use_temp_accumulator);
          break;
        }
        default:
          LOG(FATAL) << "Unexpected CBLAS_TRAnSPOSE for trans_B";
      }
    } break;
    case CblasTrans: {
      switch (trans_B) {
        case CblasNoTrans: {
          // A_trans * B
          std::vector<float> A_trans(k * m);
          transpose(A, A_trans, k, m);
          custom_fp16_gemm(
              m,
              k,
              n,
              A_trans.data(),
              B,
              beta,
              C,
              use_acc_fp16,
              use_temp_accumulator);
          break;
        }
        case CblasTrans: {
          // A_trans * B_trans
          std::vector<float> A_trans(k * m);
          std::vector<float> B_trans(n * k);
          transpose(A, A_trans, k, m);
          transpose(B, B_trans, n, k);
          custom_fp16_gemm(
              m,
              k,
              n,
              A_trans.data(),
              B_trans.data(),
              beta,
              C,
              use_acc_fp16,
              use_temp_accumulator);
          break;
        }
        default:
          LOG(FATAL) << "Unexpected CBLAS_TRAnSPOSE for trans_B";
      }
    } break;
    default:
      LOG(FATAL) << "Unexpected CBLAS_TRAnSPOSE for trans_A";
  }
}

static inline __m256 clamp_subnormals(__m256 input, const float epsilon_) {
  __m256 epsilon = _mm256_set1_ps(epsilon_);
  __m256 nepsilon = _mm256_set1_ps(-epsilon_);

  __m256 mask = _mm256_or_ps(
      _mm256_cmp_ps(input, nepsilon, _CMP_LE_OS),
      _mm256_cmp_ps(input, epsilon, _CMP_GE_OS));
  return _mm256_and_ps(input, mask);
}

void custom_fp16_gemm(
    const int m,
    const int k,
    const int n,
    const float* A_fp16,
    const float* B_fp16,
    const float beta,
    float* C,
    const bool use_acc_fp16,
    const bool use_temp_accumulator) {
#ifdef LOG_LEVEL_FOR_FBFCPACkEDACC16_PERFORmAnCE_LOG
  clock_t begin = clock();
#endif
  int C_size = m * n;
  if (beta == 0) {
    // In Caffe2 we often do a lazy initialization, which may contain NaNs in
    // the float values. As a result, if beta is 0, we explicitly do a setzero.
    memset(C, 0, C_size * sizeof(C[0]));
  } else {
    float beta_fp16 = fbgemm::cpu_half2float(fbgemm::cpu_float2half_rn(beta));

    __m256 mBetaFp16 = _mm256_broadcast_ss(&beta_fp16);
    int i = 0;
    for (i = 0; i + 8 <= C_size; i += 8) {
      __m256 mC = _mm256_loadu_ps(C + i);
      mC = _mm256_mul_ps(mC, mBetaFp16);
      _mm256_storeu_ps(C + i, mC);
    }
    for (; i < C_size; i++) {
      C[i] = C[i] * beta_fp16;
    }
  }

  // Encode the smallest normal number in float16
  union epsilon_t {
    float f;
    uint32_t i;
  };

  union epsilon_t epsilon;
  epsilon.i = 0x38800000u; // 1 / 16384

  constexpr int VLEn = 8;
  const int kb_max = 128;
  for (int i = 0; i < m; i++) {
    for (int l = 0; l < k; l += kb_max) {
      int kb = std::min(kb_max, k - l);
      for (int j = 0; j < n; j += VLEn) {
        int nb = std::min(VLEn, n - j);
        if (nb == VLEn) {
          __m256 mC = _mm256_loadu_ps(C + i * n + j);
          __m256 mC_temp = _mm256_setzero_ps();
          for (int l2 = l; l2 < l + kb; l2++) {
            __m256 mA_fp16 = _mm256_broadcast_ss(A_fp16 + i * k + l2);
            __m256 mB_fp16 = _mm256_loadu_ps((B_fp16 + l2 * n + j));

            if (use_acc_fp16) {
              mA_fp16 = clamp_subnormals(mA_fp16, epsilon.f);
              mB_fp16 = clamp_subnormals(mB_fp16, epsilon.f);
            }

            __m256 mAB = _mm256_mul_ps(mA_fp16, mB_fp16);

            if (use_acc_fp16) {
              __m256 mAB_fp16 = _mm256_cvtph_ps(_mm256_cvtps_ph(mAB, 0));
              mAB_fp16 = clamp_subnormals(mAB_fp16, epsilon.f);

              if (use_temp_accumulator) {
                mC_temp = _mm256_add_ps(mC_temp, mAB_fp16);
                mC_temp = _mm256_cvtph_ps(_mm256_cvtps_ph(mC_temp, 0));
              } else {
                mC = _mm256_add_ps(mC, mAB_fp16);
                mC = _mm256_cvtph_ps(_mm256_cvtps_ph(mC, 0));
              }
            } else {
              if (use_temp_accumulator) {
                mC_temp = _mm256_add_ps(mC_temp, mAB);
              } else {
                mC = _mm256_add_ps(mC, mAB);
              }
            }

            if (use_acc_fp16) {
              mC = clamp_subnormals(mC, epsilon.f);
            }
          }
          if (use_temp_accumulator) {
            if (use_acc_fp16) {
              mC = _mm256_cvtph_ps(_mm256_cvtps_ph(mC, 0));
              mC = _mm256_add_ps(mC, mC_temp);
              mC = _mm256_cvtph_ps(_mm256_cvtps_ph(mC, 0));
            } else {
              mC = _mm256_add_ps(mC, mC_temp);
            }
          }
          _mm256_storeu_ps(C + i * n + j, mC);
        } else {
          __m256 mC_temp = _mm256_setzero_ps();
          int32_t mask_src[] = {
              -1,
              -1,
              -1,
              -1,
              -1,
              -1,
              -1,
              -1,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
          };
          __m256i imask =
              _mm256_loadu_si256((__m256i const*)(mask_src + 8 - nb));
          __m256 mC = _mm256_maskload_ps(C + i * n + j, imask);
          for (int l2 = l; l2 < l + kb; l2++) {
            __m256 mA_fp16 = _mm256_broadcast_ss(A_fp16 + i * k + l2);
            __m256 mB_fp16 = _mm256_maskload_ps(B_fp16 + l2 * n + j, imask);

            if (use_acc_fp16) {
              mA_fp16 = clamp_subnormals(mA_fp16, epsilon.f);
              mB_fp16 = clamp_subnormals(mB_fp16, epsilon.f);
            }

            __m256 mAB = _mm256_mul_ps(mA_fp16, mB_fp16);

            if (use_acc_fp16) {
              __m256 mAB_fp16 = _mm256_cvtph_ps(_mm256_cvtps_ph(mAB, 0));
              mAB_fp16 = clamp_subnormals(mAB_fp16, epsilon.f);

              if (use_temp_accumulator) {
                mC_temp = _mm256_add_ps(mC_temp, mAB_fp16);
                mC_temp = _mm256_cvtph_ps(_mm256_cvtps_ph(mC_temp, 0));
              } else {
                mC = _mm256_add_ps(mC, mAB_fp16);
                mC = _mm256_cvtph_ps(_mm256_cvtps_ph(mC, 0));
              }
            } else {
              if (use_temp_accumulator) {
                mC_temp = _mm256_add_ps(mC_temp, mAB);
              } else {
                mC = _mm256_add_ps(mC, mAB);
              }
            }

            if (use_acc_fp16) {
              mC = clamp_subnormals(mC, epsilon.f);
            }
          }

          if (use_temp_accumulator) {
            if (use_acc_fp16) {
              mC = _mm256_cvtph_ps(_mm256_cvtps_ph(mC, 0));
              mC = _mm256_add_ps(mC, mC_temp);
              mC = _mm256_cvtph_ps(_mm256_cvtps_ph(mC, 0));
            } else {
              mC = _mm256_add_ps(mC, mC_temp);
            }
          }
          _mm256_maskstore_ps(C + i * n + j, imask, mC);
        }
      }
    }
  }

  if (!use_acc_fp16) {
    constexpr int kSize=8;
    int i = 0;
    for (; i + kSize <= C_size; i+= kSize) {
      __m256 mC = _mm256_loadu_ps(C + i);
      mC = _mm256_cvtph_ps(_mm256_cvtps_ph(mC, 0));
      _mm256_storeu_ps(C + i, mC);
    }
    if (i < C_size){
      vector<float> tmp(8);
      for (int kk =0; kk + i < C_size; kk++) {
        tmp[kk] = C[i + kk];
      }
      __m256 mC = _mm256_loadu_ps(tmp.data());
      mC = _mm256_cvtph_ps(_mm256_cvtps_ph(mC, 0));
      _mm256_storeu_ps(tmp.data(), mC);
      for (int kk =0; kk + i < C_size; kk++) {
        C[i + kk] = tmp[kk];
      }
    }
  }

#ifdef LOG_LEVEL_FOR_FBFCPACkEDACC16_PERFORmAnCE_LOG
  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCkS_PER_SEC;
  VLOG(LOG_LEVEL_FOR_FBFCPACKEDACC16_ACCURACY_LOG)
      << "cblas_gemm_compute_acc16 run time = " << elapsed_secs << endl;
#endif
}

void custom_fp16_gemv(
    const bool use_acc_fp16,
    const bool use_custom_acc32,
    const bool use_temp_accumulator,
    const CBLAS_TRANSPOSE trans_A,
    const int M,
    const int N,
    const float alpha,
    const float* A,
    const float* x,
    const float beta,
    float* y,
    CPUContext* context) {
  if (use_acc_fp16) {
    custom_fp16_gemm_with_trans(
        trans_A,
        CblasNoTrans,
        M,
        1,
        N,
        A,
        x,
        beta,
        y,
        true /* use acc_fp16 */,
        use_temp_accumulator);
  } else if (use_custom_acc32 && use_temp_accumulator) {
    custom_fp16_gemm_with_trans(
        trans_A,
        CblasNoTrans,
        M,
        1,
        N,
        A,
        x,
        beta,
        y,
        false /* use acc_fp32 */,
        use_temp_accumulator);
  } else {
    math::Gemv<float, CPUContext>(trans_A, M, N, alpha, A, x, beta, y, context);
  }
}

void custom_fp16_gemm_batched(
    const bool use_acc_fp16,
    const bool use_custom_acc32,
    const bool use_temp_accumulator,
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float** A,
    const float** B,
    const float beta,
    float** C,
    CPUContext* context) {
  if (!use_acc_fp16 && (!use_custom_acc32 || !use_temp_accumulator)) {
    math::GemmBatched<float, CPUContext>(
        trans_A, trans_B, batch_size, M, N, K, alpha, A, B, beta, C, context);
    return;
  }

  for (int i = 0; i < batch_size; ++i) {
    if (use_acc_fp16) {
      custom_fp16_gemm_with_trans(
          trans_A,
          trans_B,
          M,
          K,
          N,
          A[i],
          B[i],
          beta,
          C[i],
          true /* use acc_fp16 */,
          use_temp_accumulator);
    } else {
      CAFFE_ENFORCE(use_custom_acc32 && use_temp_accumulator);
      custom_fp16_gemm_with_trans(
          trans_A,
          trans_B,
          M,
          K,
          N,
          A[i],
          B[i],
          beta,
          C[i],
          false /* use acc_fp32 */,
          use_temp_accumulator);
    }
  }
}

void custom_fp16_gemm_strided_batched(
    const bool use_acc_fp16,
    const bool use_custom_acc32,
    const bool use_temp_accumulator,
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha /* unused */,
    const float* A,
    const int A_stride,
    const float* B,
    const int B_stride,
    const float beta,
    float* C,
    const int C_stride,
    CPUContext* context) {
  // loop over matrices in the batch
  for (int i = 0; i < batch_size; ++i) {
    if (use_acc_fp16) {
      custom_fp16_gemm_with_trans(
          trans_A,
          trans_B,
          M,
          K,
          N,
          A,
          B,
          beta,
          C,
          true /* use_acc_fp16 */,
          use_temp_accumulator);

    } else {
      custom_fp16_gemm_with_trans(
          trans_A,
          trans_B,
          M,
          K,
          N,
          A,
          B,
          beta,
          C,
          false /* use acc_fp32*/,
          use_temp_accumulator);
    }
    A += A_stride;
    B += B_stride;
    C += C_stride;
  }
}

} // namespace caffe2
