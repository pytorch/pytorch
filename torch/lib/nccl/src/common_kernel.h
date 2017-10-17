/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/


#ifndef COMMON_KERNEL_H_
#define COMMON_KERNEL_H_

#include <cstdio>
#include <cstdint>

#include <cuda_runtime.h>

// BAR macro and helpers
#define WARP_SIZE 32
#define ROUNDUP(x, y)                                                           \
    (((((x) + (y) - 1) / (y))) * (y))
#define DIVUP(x, y) \
    (((x)+(y)-1)/(y))
#define BAR_EXEC(type, barid, nthreads) \
    asm("bar." #type " " #barid ", " #nthreads ";\n\t")
#define BAR_EXPAND(type, barid, nthreads) \
    BAR_EXEC(type, barid, (nthreads))

// Named barrier macro.
// Expands to asm("bar.type barid, nthreads") where
// nthreads has been rounded up to WARP_SIZE.
#define BAR(type, barid, nthreads) \
    BAR_EXPAND(type, barid, ROUNDUP(nthreads, WARP_SIZE))

template<typename T> inline __device__
T vFetch(const volatile T* ptr) {
  return *ptr;
}

template<typename T> inline __device__
void vStore(volatile T* ptr, const T val) {
  *ptr = val;
}

#ifdef CUDA_HAS_HALF
#if CUDART_VERSION < 9000
template<> inline __device__
half vFetch<half>(const volatile half* ptr) {
  half r;
  r.x = ptr->x;
  return r;
}
template<> inline __device__
void vStore<half>(volatile half* ptr, const half val) {
  ptr->x = val.x;
}
#else
template<> inline __device__
half vFetch<half>(const volatile half* ptr) {
  return *((half*)ptr);
}
template<> inline __device__
void vStore<half>(volatile half* ptr, const half val) {
  *((half*)ptr) = val;
}
#endif
#endif

__device__ unsigned int spinct;

// Spin wait until func evaluates to true
template<typename FUNC>
__device__ inline void Wait(const FUNC& func) {
  while (!func()) {
    // waste time
    atomicInc(&spinct, 10);
  }
}

typedef uint64_t PackType;

// unpack x and y to elements of type T and apply FUNC to each element
template<class FUNC, typename T>
struct MULTI {
  __device__ PackType operator()(const PackType x, const PackType y) const;
};

template<class FUNC>
struct MULTI<FUNC, char> {
  static_assert(sizeof(PackType) == 2 * sizeof(uint32_t),
      "PackType must be twice the size of uint32_t.");
  union converter {
    PackType storage;
    struct {
      uint32_t a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    // for char, we do these as vector ops
    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }
};

template<class FUNC>
struct MULTI<FUNC, int> {
  static_assert(sizeof(PackType) == 2 * sizeof(int),
      "PackType must be twice the size of int.");
  union converter {
    PackType storage;
    struct {
      int a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }
};

#ifdef CUDA_HAS_HALF
template<class FUNC>
struct MULTI<FUNC, half> {
  static_assert(sizeof(PackType) == 4 * sizeof(half),
      "PackType must be four times the size of half.");

  struct PackHalf2 {
    half2 a, b;
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    struct PackHalf2 cx, cy, cr;
    cx = *(reinterpret_cast<const struct PackHalf2*>(&x));
    cy = *(reinterpret_cast<const struct PackHalf2*>(&y));

    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return *(reinterpret_cast<PackType*>(&cr));
  }
};
#endif

template<class FUNC>
struct MULTI<FUNC, float> {
  static_assert(sizeof(PackType) == 2 * sizeof(float),
      "PackType must be twice the size of float.");
  union converter {
    PackType storage;
    struct {
      float a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }
};

template<class FUNC>
struct MULTI<FUNC, double> {
  static_assert(sizeof(PackType) == sizeof(double),
      "PackType must be the same size as double.");
  __device__ PackType operator()(const PackType x, const PackType y) const {
    double rv = FUNC()(__longlong_as_double(x), __longlong_as_double(y));
    return __double_as_longlong(rv);
  }
};

template<class FUNC>
struct MULTI<FUNC, unsigned long long> {
  static_assert(sizeof(PackType) == sizeof(unsigned long long),
      "PackType must be the same size as unsigned long long.");
  __device__ PackType operator()(const PackType x, const PackType y) const {
    unsigned long long rv = FUNC()(x, y);
    return rv;
  }
};

template<class FUNC>
struct MULTI<FUNC, long long> {
  static_assert(sizeof(PackType) == sizeof(long long),
      "PackType must be the same size as long long.");
  __device__ PackType operator()(const PackType x, const PackType y) const {
    long long rv = FUNC()((long long)x, (long long)y);
    return rv;
  }
};

template<class FUNC, typename T, bool TWO_INPUTS, bool TWO_OUTPUTS>
__device__ inline void ReduceCopy(
    const volatile T * __restrict__ const src0,
    const volatile T * __restrict__ const src1,
    volatile T * __restrict__ const dest0,
    volatile T * __restrict__ const dest1, const int idx) {
  T val = vFetch(src0+idx);
  if (TWO_INPUTS) {
    val = FUNC()(val, vFetch(src1+idx));
  }
  vStore(dest0+idx, val);
  if (TWO_OUTPUTS) {
    vStore(dest1+idx, val);
  }
}

template<class FUNC, typename T, bool TWO_INPUTS, bool TWO_OUTPUTS, int UNROLL, int THREADS>
__device__ inline void ReduceCopy64b(
    const volatile T * __restrict__ const src0,
    const volatile T * __restrict__ const src1,
    volatile T * __restrict__ const dest0,
    volatile T * __restrict__ const dest1, const int offset) {
  PackType t0[UNROLL];
  PackType t1[UNROLL];
  #pragma unroll
  for (int u = 0; u < UNROLL; ++u) {
    int idx = offset + u*THREADS;
    t0[u] = (reinterpret_cast<const volatile PackType *>(src0))[idx];
    if (TWO_INPUTS) {
      t1[u] = (reinterpret_cast<const volatile PackType *>(src1))[idx];
    }
  }
  #pragma unroll
  for (int u = 0; u < UNROLL; ++u) {
    int idx = offset + u*THREADS;
    PackType val = TWO_INPUTS ? MULTI<FUNC, T>()(t0[u], t1[u]) : t0[u];
    (reinterpret_cast<volatile PackType *>(dest0))[idx] = val;
    if (TWO_OUTPUTS) {
      (reinterpret_cast<volatile PackType *>(dest1))[idx] = val;
    }
  }
}

#define ALIGNUP(x, a)   ((((x)-1) & ~((a)-1)) + (a))

template<typename T>
__device__ inline volatile T* AlignUp(volatile T * ptr, size_t align) {
  size_t ptrval = reinterpret_cast<size_t>(ptr);
  return reinterpret_cast<volatile T*>(ALIGNUP(ptrval, align));
}

// Assumptions:
// - there is exactly 1 block
// - THREADS is the number of producer threads
// - this function is called by all producer threads
template<int UNROLL, int THREADS, class FUNC, typename T, bool HAS_DEST1,
    bool HAS_SRC1>
__device__ inline void ReduceOrCopy(const int tid,
    volatile T * __restrict__ dest0, volatile T * __restrict__ dest1,
    const volatile T * __restrict__ src0, const volatile T * __restrict__ src1,
    int N) {
  if (N<=0) {
    return;
  }

  int Npreamble = (N<alignof(PackType)) ? N : AlignUp(dest0, alignof(PackType)) - dest0;

  // stage 0: check if we'll be able to use the fast, 64-bit aligned path.
  // If not, we'll just use the slow preamble path for the whole operation
  bool alignable = (((AlignUp(src0,  alignof(PackType)) == src0  + Npreamble)) &&
      (!HAS_DEST1 || (AlignUp(dest1, alignof(PackType)) == dest1 + Npreamble)) &&
      (!HAS_SRC1  || (AlignUp(src1,  alignof(PackType)) == src1  + Npreamble)));

  if (!alignable) {
    Npreamble = N;
  }

  // stage 1: preamble: handle any elements up to the point of everything coming
  // into alignment
  for (int idx = tid; idx < Npreamble; idx += THREADS) {
    // ought to be no way this is ever more than one iteration, except when
    // alignable is false
    ReduceCopy<FUNC, T, HAS_SRC1, HAS_DEST1>(src0, src1, dest0, dest1, idx);
  }

  // stage 2: fast path: use 64b loads/stores to do the bulk of the work,
  // assuming the pointers we have are all 64-bit alignable.
  if (alignable) {
    const int PackFactor = sizeof(PackType) / sizeof(T);
    int Nrem = N - Npreamble;
    dest0 += Npreamble; if (HAS_DEST1) { dest1 += Npreamble; }
    src0  += Npreamble; if (HAS_SRC1)  { src1  += Npreamble; }

    // stage 2a: main loop
    int Nalign2a = (Nrem / (PackFactor * UNROLL * THREADS))
        * (UNROLL * THREADS); // round down

    #pragma unroll 1 // don't unroll this loop
    for (int idx = tid; idx < Nalign2a; idx += UNROLL * THREADS) {
      ReduceCopy64b<FUNC, T, HAS_SRC1, HAS_DEST1, UNROLL, THREADS>(src0, src1, dest0, dest1, idx);
    }

    int Ndone2a = Nalign2a * PackFactor;
    Nrem -= Ndone2a;

    // stage 2b: slightly less optimized for section when we don't have full
    // UNROLLs

    int Nalign2b = Nrem / PackFactor;

    #pragma unroll 4
    for (int idx = Nalign2a + tid; idx < Nalign2a + Nalign2b; idx += THREADS) {
      ReduceCopy64b<FUNC, T, HAS_SRC1, HAS_DEST1, 1, 0>(src0, src1, dest0, dest1, idx);
    }

    int Ndone2b = Nalign2b * PackFactor;
    Nrem -= Ndone2b;
    int Ndone2 = Ndone2a + Ndone2b;
    dest0 += Ndone2; if (HAS_DEST1) { dest1 += Ndone2; }
    src0  += Ndone2; if (HAS_SRC1)  { src1  += Ndone2; }

    // stage 2c: tail

    for (int idx = tid; idx < Nrem; idx += THREADS) {
      // never ought to make it more than one time through this loop.  only a
      // few threads should even participate
      ReduceCopy<FUNC, T, HAS_SRC1, HAS_DEST1>(src0, src1, dest0, dest1, idx);
    }
  } // done fast path
}

template <typename T>
__device__ inline void incrementOpCounter(const KernelArgs<T> *args) {
  // increment comm's operation counts
  __threadfence_system(); // Technically need to ensure that cleared flags
  // are visible before incrementing op counter.
  *args->opCounter = args->opIndex+1;
}

template <int THREADS, typename T> __device__ __forceinline__
void LoadRing(const DevRing<char>* src, DevRing<T>* dst) {
  enum { NUM_WORDS = sizeof(DevRing<char>) / sizeof(long long) };
  static_assert(sizeof(DevRing<char>) % sizeof(long long) == 0, "Bad alignment");
  static_assert(THREADS >= NUM_WORDS, "Not enough threads to load DevRing");
  static_assert(sizeof(DevRing<char>) == sizeof(DevRing<T>), "DevRing size mismatch");
  long long* lldst = reinterpret_cast<long long*>(dst);
  const long long* llsrc = reinterpret_cast<const long long*>(src);
  if (threadIdx.x < NUM_WORDS) {
    lldst[threadIdx.x] = llsrc[threadIdx.x];
  }
}


#endif // COMMON_KERNEL_H_
