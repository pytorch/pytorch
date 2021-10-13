#ifndef THC_TENSORMATH_REDUCE_CUH
#define THC_TENSORMATH_REDUCE_CUH

/*
Reductions that (only) operate on accumulate types.
*/

template <typename T>
struct ReduceAdd {
  inline __device__ T operator()(const T a, const T b) const {
    return (a + b);
  }
};

template <typename T>
struct AddOp {
  __device__ __forceinline__ T operator()(T const &lhs, T const &rhs) {
    return (lhs + rhs);
  }
};

template <typename T>
struct MulOp {
  __device__ __forceinline__ T operator()(T const &lhs, T const &rhs) {
    return (lhs * rhs);
  }
};

#endif // THC_TENSORMATH_REDUCE_CUH
