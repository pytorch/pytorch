/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <qnnpack/math.h>

// Legend:
//  dq: Design-time Quantization
//  rq: Run-time Quantization

static inline void pytorch_pack_q8gemm_wdq(
    size_t nc,
    size_t kc,
    uint32_t nr,
    uint32_t np,
    uint32_t kr,
    uint8_t izp,
    uint8_t kzp,
    const uint8_t* k,
    const int32_t* b,
    void* packed_w) {
  const int32_t boff = (int32_t)kc * (int32_t)izp * (int32_t)kzp;
  for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
    const size_t nr_block_size = min(nc - nr_block_start, nr);
    int32_t* packed_b = (int32_t*)packed_w;
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
         nr_block_offset++) {
      *((int32_t*)packed_w) = b[nr_block_start + nr_block_offset] + boff;
      packed_w = (void*)((uintptr_t)packed_w + sizeof(int32_t));
    }
    packed_w =
        (void*)((uintptr_t)packed_w + (nr - nr_block_size) * sizeof(int32_t));
    for (size_t kr_block_start = 0; kr_block_start < kc; kr_block_start += kr) {
      const size_t kr_block_size = min(kc - kr_block_start, kr);
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
           nr_block_offset++) {
        int32_t ksum = 0;
        for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size;
             kr_block_offset++) {
          const uint8_t kv =
              k[(nr_block_start + nr_block_offset) * kc +
                (kr_block_start + kr_block_offset)];
          ksum += (int32_t)kv;
          *((uint8_t*)packed_w) = kv;
          packed_w = (void*)((uintptr_t)packed_w + sizeof(uint8_t));
        }
        packed_b[nr_block_offset] -= ksum * (int32_t)izp;
        packed_w =
            (void*)((uintptr_t)packed_w + (kr - kr_block_size) * sizeof(uint8_t));
      }
      packed_w =
          (void*)((uintptr_t)packed_w + ((nr - nr_block_size) & (np - 1)) * kr * sizeof(uint8_t));
    }
  }
}

static inline void pytorch_pack_q8gemm_wrq(
    const size_t nc,
    const size_t kc,
    const uint32_t nr,
    const uint32_t np,
    const uint32_t kr,
    const uint8_t* const k,
    const int32_t* const b,
    void* const packed_w) {
  union {
    void* const as_void_ptr;
    uint8_t* as_uint8_ptr;
    int32_t* as_int32_ptr;
  } packed = {packed_w};

  for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
    const size_t nr_block_size = min(nc - nr_block_start, nr);
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
         nr_block_offset++) {
      *(packed.as_int32_ptr++) = b[nr_block_start + nr_block_offset];
    }
    packed.as_int32_ptr += (nr - nr_block_size);
    for (size_t kr_block_start = 0; kr_block_start < kc; kr_block_start += kr) {
      const size_t kr_block_size = min(kc - kr_block_start, kr);
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
           nr_block_offset++) {
        for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size;
             kr_block_offset++) {
          const uint8_t kv =
              k[(nr_block_start + nr_block_offset) * kc +
                (kr_block_start + kr_block_offset)];
          *(packed.as_uint8_ptr++) = kv;
        }
        packed.as_uint8_ptr += (kr - kr_block_size);
      }
      packed.as_uint8_ptr += ((nr - nr_block_size) & (np - 1)) * kr;
    }
  }
}

static inline void pytorch_pack_q8conv_wdq(
    size_t n,
    size_t ks,
    size_t kc,
    uint32_t nr,
    uint32_t kr,
    uint8_t izp,
    uint8_t kzp,
    const uint8_t* k,
    const int32_t* b,
    void* packed_w) {
  const int32_t boff = (int32_t)ks * (int32_t)kc * (int32_t)izp * (int32_t)kzp;
  for (size_t nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {
    const size_t nr_block_size = min(n - nr_block_start, nr);
    int32_t* packed_b = (int32_t*)packed_w;
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
         nr_block_offset++) {
      *((int32_t*)packed_w) = b[nr_block_start + nr_block_offset] + boff;
      packed_w = (void*)((uintptr_t)packed_w + sizeof(int32_t));
    }
    packed_w =
        (void*)((uintptr_t)packed_w + (nr - nr_block_size) * sizeof(int32_t));
    for (size_t ki = 0; ki < ks; ki++) {
      for (size_t kr_block_start = 0; kr_block_start < kc;
           kr_block_start += kr) {
        const size_t kr_block_size = min(kc - kr_block_start, kr);
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          int32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size;
               kr_block_offset++) {
            const uint8_t kv =
                k[((nr_block_start + nr_block_offset) * ks + ki) * kc +
                  (kr_block_start + kr_block_offset)];
            ksum += (int32_t)kv;
            *((uint8_t*)packed_w) = kv;
            packed_w = (void*)((uintptr_t)packed_w + sizeof(uint8_t));
          }
          packed_b[nr_block_offset] -= ksum * (int32_t)izp;
          packed_w =
              (void*)((uintptr_t)packed_w + (kr - kr_block_size) * sizeof(uint8_t));
        }
        packed_w =
            (void*)((uintptr_t)packed_w + (nr - nr_block_size) * kr * sizeof(uint8_t));
      }
    }
  }
}

static inline void pytorch_pack_q8conv_wrq(
    const size_t n,
    const size_t ks,
    const size_t kc,
    const uint32_t nr,
    const uint32_t kr,
    const uint8_t* const k,
    const int32_t* const b,
    void* const packed_w) {
  union {
    void* const as_void_ptr;
    uint8_t* as_uint8_ptr;
    int32_t* as_int32_ptr;
  } packed = {packed_w};

  for (size_t nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {
    const size_t nr_block_size = min(n - nr_block_start, nr);
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
         nr_block_offset++) {
      *(packed.as_int32_ptr++) = b[nr_block_start + nr_block_offset];
    }
    packed.as_int32_ptr += (nr - nr_block_size);
    for (size_t ki = 0; ki < ks; ki++) {
      for (size_t kr_block_start = 0; kr_block_start < kc;
           kr_block_start += kr) {
        const size_t kr_block_size = min(kc - kr_block_start, kr);
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size;
               kr_block_offset++) {
            const uint8_t kv =
                k[((nr_block_start + nr_block_offset) * ks + ki) * kc +
                  (kr_block_start + kr_block_offset)];
            *(packed.as_uint8_ptr++) = kv;
          }
          packed.as_uint8_ptr += (kr - kr_block_size);
        }
        packed.as_uint8_ptr += (nr - nr_block_size) * kr;
      }
    }
  }
}

static inline void pytorch_pack_q8deconv_wdq(
    size_t n,
    size_t ks,
    size_t kc,
    uint32_t nr,
    uint32_t kr,
    uint8_t izp,
    uint8_t kzp,
    const uint8_t* k,
    const int32_t* b,
    void* packed_w) {
  const int32_t boff = (int32_t)ks * (int32_t)kc * (int32_t)izp * (int32_t)kzp;
  for (size_t nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {
    const size_t nr_block_size = min(n - nr_block_start, nr);
    int32_t* packed_b = (int32_t*)packed_w;
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
         nr_block_offset++) {
      *((int32_t*)packed_w) = b[nr_block_start + nr_block_offset] + boff;
      packed_w = (void*)((uintptr_t)packed_w + sizeof(int32_t));
    }
    packed_w =
        (void*)((uintptr_t)packed_w + (nr - nr_block_size) * sizeof(int32_t));
    for (size_t ki = 0; ki < ks; ki++) {
      for (size_t kr_block_start = 0; kr_block_start < kc;
           kr_block_start += kr) {
        const size_t kr_block_size = min(kc - kr_block_start, kr);
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          int32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size;
               kr_block_offset++) {
            const uint8_t kv =
                k[((kr_block_start + kr_block_offset) * ks + ki) * n +
                  (nr_block_start + nr_block_offset)];
            ksum += (int32_t)kv;
            *((uint8_t*)packed_w) = kv;
            packed_w = (void*)((uintptr_t)packed_w + sizeof(uint8_t));
          }
          packed_b[nr_block_offset] -= ksum * (int32_t)izp;
          packed_w =
              (void*)((uintptr_t)packed_w + (kr - kr_block_size) * sizeof(uint8_t));
        }
        packed_w =
            (void*)((uintptr_t)packed_w + (nr - nr_block_size) * kr * sizeof(uint8_t));
      }
    }
  }
}

static inline void pytorch_pack_q8deconv_wrq(
    const size_t n,
    const size_t ks,
    const size_t kc,
    const uint32_t nr,
    const uint32_t kr,
    const uint8_t* const k,
    const int32_t* const b,
    void* const packed_w) {
  union {
    void* const as_void_ptr;
    uint8_t* as_uint8_ptr;
    int32_t* as_int32_ptr;
  } packed = {packed_w};

  for (size_t nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {
    const size_t nr_block_size = min(n - nr_block_start, nr);
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
         nr_block_offset++) {
      *(packed.as_int32_ptr++) = b[nr_block_start + nr_block_offset];
    }
    packed.as_int32_ptr += (nr - nr_block_size);
    for (size_t ki = 0; ki < ks; ki++) {
      for (size_t kr_block_start = 0; kr_block_start < kc;
           kr_block_start += kr) {
        const size_t kr_block_size = min(kc - kr_block_start, kr);
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size;
               kr_block_offset++) {
            const uint8_t kv =
                k[((kr_block_start + kr_block_offset) * ks + ki) * n +
                  (nr_block_start + nr_block_offset)];
            *(packed.as_uint8_ptr++) = kv;
          }
          packed.as_uint8_ptr += (kr - kr_block_size);
        }
        packed.as_uint8_ptr += (nr - nr_block_size) * kr;
      }
    }
  }
}

static inline void pytorch_pack_q8dw_wdq(
    size_t h,
    size_t w,
    size_t c,
    size_t cr,
    uint8_t izp,
    uint8_t kzp,
    const uint8_t* k,
    const int32_t* b,
    void* packed_w) {
  const int32_t boff = (int32_t)h * (int32_t)w * (int32_t)izp * (int32_t)kzp;
  for (size_t cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
    const size_t cr_block_size = min(c - cr_block_start, cr);
    int32_t* packed_b = (int32_t*)packed_w;
    for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
         cr_block_offset++) {
      *((int32_t*)packed_w) = b[cr_block_start + cr_block_offset] + boff;
      packed_w = (void*)((uintptr_t)packed_w + sizeof(int32_t));
    }
    packed_w =
        (void*)((uintptr_t)packed_w + (cr - cr_block_size) * sizeof(int32_t));
    for (size_t x = 0; x < w; x++) {
      for (size_t y = 0; y < h; y++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
             cr_block_offset++) {
          const uint8_t kv =
              k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          packed_b[cr_block_offset] -= (int32_t)kv * (int32_t)izp;
          *((uint8_t*)packed_w) = kv;
          packed_w = (void*)((uintptr_t)packed_w + sizeof(uint8_t));
        }
        packed_w =
            (void*)((uintptr_t)packed_w + (cr - cr_block_size) * sizeof(uint8_t));
      }
    }
  }
}

static inline void pytorch_pack_q8dw_wrq(
    const size_t h,
    const size_t w,
    const size_t c,
    const size_t cr,
    const uint8_t* const k,
    const int32_t* const b,
    void* const packed_w) {
  union {
    void* const as_void_ptr;
    uint8_t* as_uint8_ptr;
    int32_t* as_int32_ptr;
  } packed = {packed_w};

  for (size_t cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
    const size_t cr_block_size = min(c - cr_block_start, cr);
    for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
         cr_block_offset++) {
      *(packed.as_int32_ptr++) = b[cr_block_start + cr_block_offset];
    }
    packed.as_int32_ptr += (cr - cr_block_size);
    for (size_t x = 0; x < w; x++) {
      for (size_t y = 0; y < h; y++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
             cr_block_offset++) {
          const uint8_t kv =
              k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *(packed.as_uint8_ptr++) = kv;
        }
        packed.as_uint8_ptr += (cr - cr_block_size);
      }
    }
  }
}

static inline void pytorch_pack_q8dw_w_dilation(
    size_t h,
    size_t w,
    size_t c,
    size_t cr,
    size_t y_start,
    size_t y_end,
    size_t x_start,
    size_t x_end,
    const uint8_t* k,
    const int32_t* b,
    void* packed_w,
    bool pytorch_pack_b) {
  for (size_t cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
    const size_t cr_block_size = min(c - cr_block_start, cr);
    if (pytorch_pack_b) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
           cr_block_offset++) {
        *((int32_t*)packed_w) = b[cr_block_start + cr_block_offset];
        packed_w = (void*)((uintptr_t)packed_w + sizeof(int32_t));
      }
      packed_w =
          (void*)((uintptr_t)packed_w + (cr - cr_block_size) * sizeof(int32_t));
    }
    for (size_t x = x_start; x < x_end; x++) {
      for (size_t y = y_start; y < y_end; y++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
             cr_block_offset++) {
          *((uint8_t*)packed_w) =
              k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          packed_w = (void*)((uintptr_t)packed_w + sizeof(uint8_t));
        }
        packed_w =
            (void*)((uintptr_t)packed_w + (cr - cr_block_size) * sizeof(uint8_t));
      }
    }
  }
}

static inline void pytorch_pack_swizzle_q8gemm_bdq(
    size_t n,
    size_t kc,
    uint32_t nr,
    uint32_t kr,
    uint32_t sr,
    uint8_t izp,
    uint8_t kzp,
    const uint8_t* k,
    const int32_t* b,
    void* packed_w) {
  const int32_t boff = (int32_t)kc * (int32_t)izp * (int32_t)kzp;
  for (size_t nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {
    const size_t nr_block_size = min(n - nr_block_start, nr);
    int32_t* packed_b = (int32_t*)packed_w;
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
         nr_block_offset++) {
      *((int32_t*)packed_w) = b[nr_block_start + nr_block_offset] + boff;
      packed_w = (void*)((uintptr_t)packed_w + sizeof(int32_t));
    }
    packed_w =
        (void*)((uintptr_t)packed_w + (nr - nr_block_size) * sizeof(int32_t));

    for (size_t kr_block_start = 0; kr_block_start < (kc & -sr);
         kr_block_start += kr) {
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
           nr_block_offset++) {
        for (size_t kr_block_offset = 0; kr_block_offset < kr;
             kr_block_offset++) {
          const uint8_t kv =
              k[(nr_block_start + nr_block_offset) * kc +
                (kr_block_start & -sr) +
                ((kr_block_start + nr_block_offset * kr) & (sr - 1)) +
                kr_block_offset];
          packed_b[nr_block_offset] -= (int32_t)kv * (int32_t)izp;
          *((uint8_t*)packed_w) = kv;
          packed_w = (void*)((uintptr_t)packed_w + sizeof(uint8_t));
        }
      }
      packed_w =
          (void*)((uintptr_t)packed_w + (nr - nr_block_size) * kr * sizeof(uint8_t));
    }

    for (size_t kr_block_start = (kc & -sr); kr_block_start < kc;
         kr_block_start += kr) {
      const size_t kr_block_size = min(kc - kr_block_start, kr);
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
           nr_block_offset++) {
        for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size;
             kr_block_offset++) {
          const uint8_t kv =
              k[(nr_block_start + nr_block_offset) * kc +
                (kr_block_start + kr_block_offset)];
          packed_b[nr_block_offset] -= (int32_t)kv * (int32_t)izp;
          *((uint8_t*)packed_w) = kv;
          packed_w = (void*)((uintptr_t)packed_w + sizeof(uint8_t));
        }
        packed_w =
            (void*)((uintptr_t)packed_w + (kr - kr_block_size) * sizeof(uint8_t));
      }
      packed_w =
          (void*)((uintptr_t)packed_w + (nr - nr_block_size) * kr * sizeof(uint8_t));
    }
  }
}

static inline void pytorch_pack_swizzle_q8gemm_brq(
    const size_t n,
    const size_t kc,
    const uint32_t nr,
    const uint32_t kr,
    const uint32_t sr,
    const uint8_t* const k,
    const int32_t* const b,
    void* const packed_w) {
  union {
    void* const as_void_ptr;
    uint8_t* as_uint8_ptr;
    int32_t* as_int32_ptr;
  } packed = {packed_w};

  for (size_t nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {
    const size_t nr_block_size = min(n - nr_block_start, nr);
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
         nr_block_offset++) {
      *(packed.as_int32_ptr++) = b[nr_block_start + nr_block_offset];
    }

    packed.as_int32_ptr += (nr - nr_block_size);

    for (size_t kr_block_start = 0; kr_block_start < (kc & -sr);
         kr_block_start += kr) {
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
           nr_block_offset++) {
        for (size_t kr_block_offset = 0; kr_block_offset < kr;
             kr_block_offset++) {
          const uint8_t kv =
              k[(nr_block_start + nr_block_offset) * kc +
                (kr_block_start & -sr) +
                ((kr_block_start + nr_block_offset * kr) & (sr - 1)) +
                kr_block_offset];
          *(packed.as_uint8_ptr++) = kv;
        }
      }
      packed.as_uint8_ptr += (nr - nr_block_size) * kr;
    }

    for (size_t kr_block_start = (kc & -sr); kr_block_start < kc;
         kr_block_start += kr) {
      const size_t kr_block_size = min(kc - kr_block_start, kr);
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
           nr_block_offset++) {
        for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size;
             kr_block_offset++) {
          const uint8_t kv =
              k[(nr_block_start + nr_block_offset) * kc +
                (kr_block_start + kr_block_offset)];
          *(packed.as_uint8_ptr++) = kv;
        }
        packed.as_uint8_ptr += (kr - kr_block_size);
      }
      packed.as_uint8_ptr += (nr - nr_block_size) * kr;
    }
  }
}

static inline void pytorch_pack_hgemm_w(
    size_t nc,
    size_t kc,
    size_t nr,
    size_t kr,
    const uint16_t* k,
    const uint16_t* b,
    uint16_t* packed_w) {
  for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
    const size_t nr_block_size = min(nc - nr_block_start, nr);
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
         nr_block_offset++) {
      *packed_w++ = b[nr_block_start + nr_block_offset];
    }
    packed_w += nr - nr_block_size;
    for (size_t kr_block_start = 0; kr_block_start < kc; kr_block_start += kr) {
      const size_t kr_block_size = min(kc - kr_block_start, kr);
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
           nr_block_offset++) {
        for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size;
             kr_block_offset++) {
          *packed_w++ =
              k[(nr_block_start + nr_block_offset) * kc +
                (kr_block_start + kr_block_offset)];
        }
        packed_w += kr - kr_block_size;
      }
      packed_w += (nr - nr_block_size) * kr;
    }
  }
}

static inline void pytorch_pack_sgemm_w(
    size_t nc,
    size_t kc,
    size_t nr,
    size_t kr,
    const float* k,
    const float* b,
    float* packed_w) {
  for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
    const size_t nr_block_size = min(nc - nr_block_start, nr);
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
         nr_block_offset++) {
      *packed_w++ = b[nr_block_start + nr_block_offset];
    }
    packed_w += nr - nr_block_size;
    for (size_t kr_block_start = 0; kr_block_start < kc; kr_block_start += kr) {
      const size_t kr_block_size = min(kc - kr_block_start, kr);
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
           nr_block_offset++) {
        for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size;
             kr_block_offset++) {
          *packed_w++ =
              k[(nr_block_start + nr_block_offset) * kc +
                (kr_block_start + kr_block_offset)];
        }
        packed_w += kr - kr_block_size;
      }
      packed_w += (nr - nr_block_size) * kr;
    }
  }
}

static inline void pytorch_pack_sconv_w(
    size_t n,
    size_t ks,
    size_t kc,
    size_t nr,
    size_t kr,
    const float* k,
    const float* b,
    float* packed_w) {
  for (size_t nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {
    const size_t nr_block_size = min(n - nr_block_start, nr);
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
         nr_block_offset++) {
      *packed_w++ = b[nr_block_start + nr_block_offset];
    }
    packed_w += nr - nr_block_size;
    for (size_t ki = 0; ki < ks; ki++) {
      for (size_t kr_block_start = 0; kr_block_start < kc;
           kr_block_start += kr) {
        const size_t kr_block_size = min(kc - kr_block_start, kr);
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size;
               kr_block_offset++) {
            *packed_w++ =
                k[((nr_block_start + nr_block_offset) * ks + ki) * kc +
                  (kr_block_start + kr_block_offset)];
          }
          packed_w += kr - kr_block_size;
        }
        packed_w += (nr - nr_block_size) * kr;
      }
    }
  }
}

#if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION

#define pytorch_pack_q8gemm_w pytorch_pack_q8gemm_wrq
#define pytorch_pack_q8conv_w pytorch_pack_q8conv_wrq
#define pytorch_pack_q8deconv_w pytorch_pack_q8deconv_wrq
#define pytorch_pack_q8dw_w pytorch_pack_q8dw_wrq
#define pytorch_pack_swizzle_q8gemm_b pytorch_pack_swizzle_q8gemm_brq

#else

#define pytorch_pack_q8gemm_w pytorch_pack_q8gemm_wdq
#define pytorch_pack_q8conv_w pytorch_pack_q8conv_wdq
#define pytorch_pack_q8deconv_w pytorch_pack_q8deconv_wdq
#define pytorch_pack_q8dw_w pytorch_pack_q8dw_wdq
#define pytorch_pack_swizzle_q8gemm_b pytorch_pack_swizzle_q8gemm_bdq

#endif
