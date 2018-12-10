#ifndef _FAST_MATH_HPP_
#define _FAST_MATH_HPP_
#include <bitset>
#include <cstring>
#include <type_traits>
#include <immintrin.h>
#include "abstract_types.hpp"

#ifdef __AVX2__

#define FM_AVX2_PREF \
  ideep::utils::fast_math<ideep::utils::cpu_isa_t::avx2>

namespace ideep {
namespace utils {

typedef enum {
    isa_any,
    sse42,
    avx2,
    avx512_common,
    avx512_core,
    avx512_mic,
    avx512_mic_4ops,
} cpu_isa_t;


template<cpu_isa_t isa = avx2>
class fast_math {
  static constexpr int thread_hold = 1024;
  static constexpr int align_bytes = 32;

public:
  using TF = __m256;
  using TI = __m256i;

  template<typename T>
  static inline unsigned get_vec_sz() {
    return 256 / 8 / sizeof(T);
  }

  static inline TI size_to_mask(unsigned nres) {
    IDEEP_ENFORCE(nres < 8 && nres >= 0, "Invalid mask size");
    std::bitset<8> e = ~((1 << nres) - 1);
    return _mm256_set_epi32(e[7]-1, e[6]-1, e[5]-1, e[4]-1,
                            e[3]-1, e[2]-1, e[1]-1, e[0]-1);
  }

  static inline TF add_ps(TF v1, TF v2) {
    return _mm256_add_ps(v1, v2);
  }

  static inline TF sub_ps(TF v1, TF v2) {
    return _mm256_sub_ps(v1, v2);
  }

  static inline TF mul_ps(TF v1, TF v2) {
    return _mm256_mul_ps(v1, v2);
  }

  static inline TF div_ps(TF v1, TF v2) {
    return _mm256_div_ps(v1, v2);
  }

  static inline TF sqrt_ps(TF v) {
    return _mm256_sqrt_ps(v);
  }

  template<typename T = float>
  static inline TF set1_ps(const T v) {
    return _mm256_set1_ps(v);
  }

  template<typename T = float>
  static inline TF load_ps(const T *src) {
    return _mm256_load_ps(src);
  }

  template<typename T = float>
  static inline TF maskload_ps(const T *src, TI mask) {
    return _mm256_maskload_ps(src, mask);
  }

  template<typename T = float>
  static inline void store_ps(T *dst, TF v) {
    _mm256_store_ps(dst, v);
  }

  template<typename T = float>
  static inline void maskstore_ps(T *dst, TI mask, TF v) {
    _mm256_maskstore_ps(dst, mask, v);
  }

  template<class T = float>
  static inline void memcpy(T* src, T* dst, size_t size) {
    auto itemsize = sizeof(T);
    auto vec_sz = get_vec_sz<T>();
    auto num_vec = size / vec_sz;
    auto num_res = size % vec_sz;

    if ((size < vec_sz) ||
        (IDEEP_MOD_PTR(src, align_bytes) != IDEEP_MOD_PTR(dst, align_bytes))) {
      std::memcpy(dst, src, itemsize * size);
      return;
    }

    auto cpy_cnt = 0;
    auto cur_res = num_res;
    auto cur_vec = num_vec;
    if (!IDEEP_IS_ALIGNED_PTR(src, align_bytes)) {
      cpy_cnt = (align_bytes - IDEEP_MOD_PTR(src, align_bytes)) / itemsize;
      std::memcpy(dst, src, itemsize * cpy_cnt);
      src += cpy_cnt;
      dst += cpy_cnt;
    }
    IDEEP_ENFORCE(cpy_cnt < vec_sz, "invalid copy count");
    IDEEP_ENFORCE(IDEEP_IS_ALIGNED_PTR(dst, align_bytes),
                  "not bytes aligned address");

    if (cpy_cnt > cur_res) {
        cur_vec -= 1;
        cur_res = vec_sz - (cpy_cnt - cur_res);
    } else {
        cur_res -= cpy_cnt;
    }

    for (auto j = 0; j < cur_vec; j++, dst += vec_sz, src += vec_sz) {
      auto vmm = load_ps(src);
      store_ps(dst, vmm);
    }

    if (cur_res != 0) {
      auto mask = size_to_mask(cur_res);
      auto vmm = maskload_ps(src, mask);
      maskstore_ps(dst, mask, vmm);
    }
  }

  // Unary ops
  template<typename vec_op, typename vec_op_mask, typename T = float>
  static inline void single_thread_vecwise_unary_op(
      T *dst, const T *src, size_t nelems,
      vec_op op, vec_op_mask op_mask) {
    auto vec_sz = get_vec_sz<T>();
    auto nvec = nelems / vec_sz;
    auto nres = nelems % vec_sz;
    for (unsigned vec = 0; vec < nvec; vec ++, src+=vec_sz, dst+=vec_sz) {
      TF vmm1 = load_ps(src);
      vmm1 = op(vmm1);
      store_ps(dst, vmm1);
    }

    if (nres != 0) {
      TI mask = size_to_mask(nres);
      TF vmm1 = maskload_ps(src, mask);
      vmm1 = op_mask(vmm1, mask);
      maskstore_ps(dst, mask, vmm1);
    }
  }

  template<typename vec_op, typename vec_op_mask, typename T = float>
  static inline void vecwise_unary_op (T *dst, const T *src, size_t nelems,
      vec_op op, vec_op_mask op_mask) {
    if (nelems < thread_hold)
      single_thread_vecwise_unary_op(dst, src, nelems, op, op_mask);
  }

  template<class T = float>
  static void inv_square_var(float epsilon,
      const T* inv_sqrt_var, T* variance, unsigned nelems) {
    if (isa == avx2) {
      if (std::is_same<T, float>::value) {
        const float *src = reinterpret_cast<const float *>(inv_sqrt_var);
        float *dst = reinterpret_cast<float *>(variance);

        TF ones = set1_ps(1.f);
        TF epsilones = set1_ps(epsilon);
        auto vec_inv_square = [ones, epsilones] (TF vmm1) {
          vmm1 = mul_ps(vmm1, vmm1);
          vmm1 = add_ps(vmm1, epsilones);
          vmm1 = div_ps(ones, vmm1);
          return vmm1;
        };
        auto mask_vec_inv_square =
          [ones, epsilones] (TF vmm1, TI) {
            vmm1 = mul_ps(vmm1, vmm1);
            vmm1 = add_ps(vmm1, epsilones);
            vmm1 = div_ps(ones, vmm1);
            return vmm1;
        };
        vecwise_unary_op(dst, src, nelems, vec_inv_square, mask_vec_inv_square);
      } else {
        throw error(mkldnn_unimplemented, "Not implemented!");
      }
    } else {
      throw error(mkldnn_unimplemented, "Not implemented!");
    }
  }

  template<class T = float>
  static void inv_sqrt_var(float epsilon,
      const void* variance, void* inv_sqrt_var, unsigned nelems) {
    if (isa == avx2) {
      if (std::is_same<T, float>::value) {
        const float *src =
          reinterpret_cast<const float *>(variance);
        float *dst =
          reinterpret_cast<float *>(inv_sqrt_var);

        unsigned nvec = nelems / 8;
        unsigned nres = nelems % 8;
        TF ones = set1_ps(1.f);
        TF epsilones = set1_ps(epsilon);
        for (unsigned vec = 0; vec < nvec; vec ++, src+=8, dst+=8) {
          TF vmm1 = load_ps(src);
          vmm1 = add_ps(vmm1, epsilones);
          vmm1 = sqrt_ps(vmm1);
          vmm1 = div_ps(ones, vmm1);
          store_ps(dst, vmm1);
        }

        if (nres != 0) {
          TI mask = size_to_mask(nres);
          TF vmm1 = maskload_ps(src, mask);
          vmm1 = add_ps(vmm1, epsilones);
          vmm1 = sqrt_ps(vmm1);
          vmm1 = div_ps(ones, vmm1);
          maskstore_ps(dst, mask, vmm1);
        }
      } else {
        throw error(mkldnn_unimplemented, "Not implemented!");
      }
    } else {
      throw error(mkldnn_unimplemented, "Not implemented!");
    }
  }

  // binary ops
  template<typename vec_op, typename vec_op_mask, typename T = float>
  static inline void single_thread_vecwise_binary_op(
      T *dst, const T *src1, const T *src2, size_t nelems,
      vec_op op, vec_op_mask op_mask) {
    auto vec_sz = get_vec_sz<T>();
    auto nvec = nelems / vec_sz;
    auto nres = nelems % vec_sz;
    for (unsigned vec = 0; vec < nvec;
        vec ++, src1+=vec_sz, src2+=vec_sz, dst+=vec_sz) {
      TF vmm1 = load_ps(src1);
      TF vmm2 = load_ps(src2);
      vmm2 = op(vmm1, vmm2);
      store_ps(dst, vmm2);
    }

    if (nres != 0) {
      TI mask = size_to_mask(nres);
      TF vmm1 = maskload_ps(src1, mask);
      TF vmm2 = maskload_ps(src2, mask);
      vmm2 = op_mask(vmm1, vmm2);
      maskstore_ps(dst, mask, vmm2);
    }
  }

  template<typename vec_op, typename vec_op_mask, typename T = float>
  static inline void vecwise_binary_op (T *dst, const T *src1, const T *src2,
      size_t nelems, vec_op op, vec_op_mask op_mask) {
    if (nelems < thread_hold)
      single_thread_vecwise_binary_op(dst, src1, src2, nelems, op, op_mask);
  }

  template<class T = float>
  static void add(T *dst, const T *src1, const T *src2,
      unsigned nelems) {
    if (std::is_same<T, float>::value) {
      auto op = [] (TF vmm1, TF vmm2) {
        vmm1 = add_ps(vmm1, vmm2);
        return vmm1;
      };
      vecwise_binary_op(dst, src1, src2, nelems, op, op);
    } else {
      throw error(mkldnn_unimplemented, "Not implemented!");
    }
  }

};
}
}
#endif

#endif
