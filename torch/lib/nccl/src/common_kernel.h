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
#define BAR_EXEC(type, barid, nthreads) \
    asm("bar." #type " " #barid ", " #nthreads ";\n\t")
#define BAR_EXPAND(type, barid, nthreads) \
    BAR_EXEC(type, barid, (nthreads))

// Named barrier macro.
// Expands to asm("bar.type barid, nthreads") where
// nthreads has been rounded up to WARP_SIZE.
#define BAR(type, barid, nthreads) \
    BAR_EXPAND(type, barid, ROUNDUP(nthreads, WARP_SIZE))

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
  static_assert(sizeof(PackType) == 2 * sizeof(float),
      "PackType must be twice the size of float.");
  union converter {
    PackType storage;
    struct {
      half2 a, b;
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

template<typename T, bool FETCHTWO>
__device__ inline void FetchOneOrTwo64b(PackType& s0,
    const volatile T * __restrict__ const src0, PackType& s1,
    const volatile T * __restrict__ const src1, const int idx) {
  s0 = (reinterpret_cast<const volatile PackType *>(src0))[idx];
  if (FETCHTWO) {
    s1 = (reinterpret_cast<const volatile PackType *>(src1))[idx];
  }
}

template<typename T, bool STORETWO>
__device__ inline void StoreOneOrTwo64b(volatile T * __restrict__ const dest0,
    volatile T * __restrict__ const dest1, PackType val, const int idx) {
  (reinterpret_cast<volatile PackType *>(dest0))[idx] = val;
  if (STORETWO) {
    (reinterpret_cast<volatile PackType *>(dest1))[idx] = val;
  }
}

template<class FUNC, typename T, bool ISREDUCE>
__device__ inline PackType ReduceOrCopy64b(const PackType s0,
    const PackType s1) {
  if (ISREDUCE) {
    return MULTI<FUNC, T>()(s0, s1);
  } else {
    return s0;
  }
}

#define ALIGNUP(x, a)   ((((x)-1) & ~((a)-1)) + (a))

template<typename T>
__device__ inline volatile T* AlignUp(volatile T * ptr, size_t align) {
  size_t ptrval = reinterpret_cast<size_t>(ptr);
  return reinterpret_cast<volatile T*>(ALIGNUP(ptrval, align));
}

template<typename T> inline __device__
T vFetch(const volatile T* ptr) {
  return *ptr;
}

#ifdef CUDA_HAS_HALF
template<> inline __device__
half vFetch<half>(const volatile half* ptr) {
  half r;
  r.x = ptr->x;
  return r;
}
#endif

template<typename T> inline __device__
void vStore(volatile T* ptr, const T val) {
  *ptr = val;
}

#ifdef CUDA_HAS_HALF
template<> inline __device__
void vStore<half>(volatile half* ptr, const half val) {
  ptr->x = val.x;
}
#endif

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

  const int UNROLL2 = (UNROLL >= 2) ? (UNROLL / 2) : 1;
  const bool NOUNROLL2 = ((UNROLL / 2) == 0);

  int Npreamble = (N<alignof(PackType)) ? N : AlignUp(dest0, alignof(PackType)) - dest0;

  // stage 0: check if we'll be able to use the fast, 64-bit aligned path.
  // If not, we'll just use the slow preamble path for the whole operation
  bool alignable = (((AlignUp(src0,  alignof(PackType)) == src0  + Npreamble)) &&
      (!HAS_DEST1 || (AlignUp(dest1, alignof(PackType)) == dest1 + Npreamble)) &&
      (!HAS_SRC1  || (AlignUp(src1,  alignof(PackType)) == src1  + Npreamble)));

  if (!alignable) {
    Npreamble = N;
  }

/*
  if (threadIdx.x == 0) {
    printf("** alignable: %s", (alignable ? "YES" : " NO"));
    printf(", dest0 = 0x%08X", dest0);
    printf(", src0 = 0x%08X", src0);
    if (HAS_DEST1) printf(", dest1 = 0x%08X", dest1);
    if (HAS_SRC1) printf(", src1 = 0x%08X", src1);
    printf("\n");
  }
*/

  // stage 1: preamble: handle any elements up to the point of everything coming
  // into alignment
  for (int idx = tid; idx < Npreamble; idx += THREADS) {
    // ought to be no way this is ever more than one iteration, except when
    // alignable is false
    T val = vFetch(src0+idx);
    if (HAS_SRC1) {
      val = FUNC()(val, vFetch(src1+idx));
    }

    vStore(dest0+idx, val);
    if (HAS_DEST1) {
      vStore(dest1+idx, val);
    }
  }

  // reduce N by however many elements we've handled already
  int Ndone = Npreamble;
  int Nrem = N - Ndone;

  // stage 2: fast path: use 64b loads/stores to do the bulk of the work,
  // assuming the pointers we have are all 64-bit alignable.
  if (alignable) {
    if (Ndone > 0) {
      // align up pointers
      dest0 += Ndone; if (HAS_DEST1) { dest1 += Ndone; }
      src0  += Ndone; if (HAS_SRC1)  { src1  += Ndone; }
    }

    // stage 2a: main loop
    int Nalign = (Nrem / (sizeof(PackType) / sizeof(T)) / (UNROLL * THREADS))
        * (UNROLL * THREADS); // round down

    #pragma unroll 1 // don't unroll this loop
    for (int idx = tid; idx < Nalign; idx += UNROLL * THREADS) {
      PackType t0[UNROLL2];
      PackType t1[UNROLL2];
      PackType t2[UNROLL2];

      #pragma unroll
      for (int j = 0; j < UNROLL2; ++j)
        FetchOneOrTwo64b<T, HAS_SRC1>(t0[j], src0, t1[j], src1,
            idx + j * THREADS);

      #pragma unroll
      for (int j = 0; j < UNROLL2; ++j)
        t2[j] = ReduceOrCopy64b<FUNC, T, HAS_SRC1>(t0[j], t1[j]);

      if (!NOUNROLL2) {
        #pragma unroll
        for (int j = 0; j < UNROLL2; ++j)
          FetchOneOrTwo64b<T, HAS_SRC1>(t0[j], src0, t1[j], src1,
              idx + (UNROLL2 + j) * THREADS);
      }

      #pragma unroll
      for (int j = 0; j < UNROLL2; ++j)
        StoreOneOrTwo64b<T, HAS_DEST1>(dest0, dest1, t2[j], idx + j * THREADS);

      if (!NOUNROLL2) {
        #pragma unroll
        for (int j = 0; j < UNROLL2; ++j)
          t2[j] = ReduceOrCopy64b<FUNC, T, HAS_SRC1>(t0[j], t1[j]);

        #pragma unroll
        for (int j = 0; j < UNROLL2; ++j)
          StoreOneOrTwo64b<T, HAS_DEST1>(dest0, dest1, t2[j],
              idx + (UNROLL2 + j) * THREADS);
      }
    }

    // stage 2b: slightly less optimized for section when we don't have full
    // UNROLLs
    int Ndone2a = Nalign * (sizeof(PackType)/sizeof(T));
    Ndone += Ndone2a;
    Nrem = N - Ndone;

    // TODO: This kind of pointer update arithmetic is expensive.  Should
    // probably find a better way.
    if (Nrem > 0) {
      dest0 += Ndone2a; if (HAS_DEST1) { dest1 += Ndone2a; }
      src0  += Ndone2a; if (HAS_SRC1)  { src1  += Ndone2a; }
    }

    Nalign = Nrem / (sizeof(PackType)/sizeof(T));

    #pragma unroll 4
    for (int idx = tid; idx < Nalign; idx += THREADS) {
      PackType t0, t1, t2;

      FetchOneOrTwo64b<T, HAS_SRC1>(t0, src0, t1, src1, idx);
      t2 = ReduceOrCopy64b<FUNC, T, HAS_SRC1>(t0, t1);
      StoreOneOrTwo64b<T, HAS_DEST1>(dest0, dest1, t2, idx);
    }

    // stage 2c: tail
    int Ndone2b = Nalign * (sizeof(PackType)/sizeof(T));
    Ndone += Nalign * (sizeof(PackType)/sizeof(T));
    Nrem = N - Ndone;

    if (Nrem > 0) {
      dest0 += Ndone2b; if (HAS_DEST1) { dest1 += Ndone2b; }
      src0  += Ndone2b; if (HAS_SRC1)  { src1  += Ndone2b; }
    }

    for (int idx = tid; idx < Nrem; idx += THREADS) {
      // never ought to make it more than one time through this loop.  only a
      // few threads should even participate
      T val = vFetch(src0+idx);
      if (HAS_SRC1) {
        val = FUNC()(val, vFetch(src1+idx));
      }

      vStore(dest0+idx, val);
      if (HAS_DEST1) {
        vStore(dest1+idx, val);
      }
    }
  } // done fast path
}

template<int THREADS, int UNROLL, typename T>
__device__ inline void CalcLastChunk(int * const bigSliceN,
    int * const smallSliceN, int * const lastSliceN, int * const numSlices,
    int * const numBigSlices, int * const numSmallSlices, const int N,
    const int numChunks, const int chunkSize) {
  int Nleft = N - ((numChunks - 1) * chunkSize);
  // semi-equally split up the remaining work into numslices slices.
  // it's "semi"-equal because we want the divisions to land as neatly as we
  // can on alignable boundaries
  int NperTile = UNROLL * THREADS * (sizeof(PackType)/sizeof(T));
  int numTiles = (Nleft + NperTile - 1) / NperTile;
  int numTilesPerBigSlice = (numTiles + *numSlices - 1)
      / *numSlices;
  int numTilesPerSmallSlice = numTiles / *numSlices;

  *bigSliceN   = NperTile * numTilesPerBigSlice;
  *smallSliceN = NperTile * numTilesPerSmallSlice;
  *numBigSlices = numTiles % *numSlices;
  *numSmallSlices = (*smallSliceN > 0) ?
      *numSlices - *numBigSlices : 0;

  // the lastSlice will take the place of one of the small slices unless
  // there are no small slices (because this is a very small reduction), in
  // which case we replace one of the big slices and leave the small slices
  // as 0.
  if (*numSmallSlices > 0) {
    --*numSmallSlices;
    if (*numSmallSlices == 0)
      *smallSliceN = 0;
  }
  else {
    --*numBigSlices;
    if (*numBigSlices == 0)
      *bigSliceN = 0;
  }

  *lastSliceN = Nleft -
      (*numBigSlices * *bigSliceN
          + *numSmallSlices * *smallSliceN);

  // in cases where args.N % numSlices is pretty small, we'd rather have one
  // slightly big last slice than one big slice, a bunch of small slices,
  // and one smaller last slice
  if ((*numBigSlices == 1) &&
      (*numSmallSlices == *numSlices - 2) &&
      (*lastSliceN < *smallSliceN)) {
    *numBigSlices += *numSmallSlices;
    *numSmallSlices = 0;
    *bigSliceN = *smallSliceN;
    *smallSliceN = 0;
    *lastSliceN = Nleft -
        *numBigSlices * *bigSliceN;
  }

  // done recalculating
  *numSlices = *numBigSlices +
      *numSmallSlices + 1;
}

// Kernel launch
template<typename T>
struct KernelArgs {
  // general parameters
  int nRanks;
  int root;
  int buffSize;
  int N;
  int opIndex;
  volatile int * __restrict__ opCounter;
  bool pushrecv;

  // some pre-computed sizes
  int SliceSize;
  int SliceOffset;
  int ChunkSize;
  int NumChunks;

  // local and remote input, output, and buffer
  const T * __restrict__ ThisInput;
  T * __restrict__ ThisOutput;

  DevRing<char>* ring;
};

template<typename T>
void ArgsSetup(KernelArgs<T> *args, const void* sendbuff, void* recvbuff,
		const int root, const int count, ncclComm *comm) {
  args->nRanks = comm->nRanks;
  args->root = root;
  args->buffSize = comm->buffSize;
  args->N = count;
  args->opIndex = comm->opSched;
  args->opCounter = comm->opCounter;
  args->ThisInput = (const T*)sendbuff;
  args->ThisOutput = (T*)recvbuff;
  args->ring = comm->devRing;
  args->pushrecv = comm->globalMemSpace;
}

#define LAUNCH_KERNEL(K, THREADS, UNROLL, FUNC, T, \
		args, stream) do { \
  dim3 grid(1, 1, 1); \
  dim3 block(THREADS+1, 1, 1); \
  void* argptrs[] = {&args}; \
  CUDACHECK(cudaLaunchKernel( \
            (void*)K<THREADS, UNROLL, FUNC, T>, \
            grid, block, argptrs, 0, stream)); \
} while (0)

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
