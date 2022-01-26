#include <limits>
#include <map>

#include "ScatterReduceAtomics.cuh"

namespace at { namespace native {

namespace {

enum ReductionType { SUM, MEAN, MUL, DIV, MIN, MAX };

const std::map<c10::string_view, ReductionType> reduce2REDUCE = {
    {"sum", SUM}, {"mean", MEAN}, {"prod", MUL},
    {"amin", MIN}, {"amax", MAX},
};

#define AT_DISPATCH_REDUCTION_TYPES(reduce, ...)                               \
  [&] {                                                                        \
    switch (reduce2REDUCE.at(reduce)) {                                        \
    case SUM: {                                                                \
      const ReductionType REDUCE = SUM;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MEAN: {                                                               \
      const ReductionType REDUCE = MEAN;                                       \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MUL: {                                                                \
      const ReductionType REDUCE = MUL;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case DIV: {                                                                \
      const ReductionType REDUCE = DIV;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MIN: {                                                                \
      const ReductionType REDUCE = MIN;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MAX: {                                                                \
      const ReductionType REDUCE = MAX;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    }                                                                          \
  }()

template <typename scalar_t, ReductionType REDUCE> struct Reducer {
  static inline __host__ __device__ scalar_t init() {
    if (REDUCE == MUL || REDUCE == DIV)
      return (scalar_t)1;
    else if (REDUCE == MIN)
      return std::numeric_limits<scalar_t>::max();
    else if (REDUCE == MAX)
      return std::numeric_limits<scalar_t>::lowest();
    else
      return (scalar_t)0;
  }

  static inline __host__ __device__ void update(scalar_t *val,
                                                scalar_t new_val) {
    if (REDUCE == SUM || REDUCE == MEAN)
      *val = *val + new_val;
    else if (REDUCE == MUL)
      *val = *val * new_val;
    else if (REDUCE == DIV)
      *val = *val / new_val;
    else if ((REDUCE == MIN && new_val < *val) ||
             (REDUCE == MAX && new_val > *val)) {
      *val = new_val;
    }
  }

  static inline __host__ __device__ void update(scalar_t *val, scalar_t new_val,
                                                int64_t *arg, int64_t new_arg) {
    if (REDUCE == SUM || REDUCE == MEAN)
      *val = *val + new_val;
    else if (REDUCE == MUL)
      *val = *val * new_val;
    else if (REDUCE == DIV)
      *val = *val / new_val;
    else if ((REDUCE == MIN && new_val < *val) ||
             (REDUCE == MAX && new_val > *val)) {
      *val = new_val;
      *arg = new_arg;
    }
  }

  static inline __host__ __device__ void write(scalar_t *address, scalar_t val,
                                               int64_t *arg_address,
                                               int64_t arg, int count) {
    if (REDUCE == SUM || REDUCE == MUL || REDUCE == DIV)
      *address = val;
    else if (REDUCE == MEAN)
      *address = val / (count > 0 ? count : (scalar_t)1);
    else if (REDUCE == MIN || REDUCE == MAX) {
      if (count > 0) {
        *address = val;
        *arg_address = arg;
      } else
        *address = (scalar_t)0;
    }
  }

  static inline __device__ void atomic_write(scalar_t *address, scalar_t val) {
    if (REDUCE == SUM || REDUCE == MEAN)
      atomAdd(address, val);
    else if (REDUCE == MUL)
      atomMul(address, val);
    else if (REDUCE == DIV)
      atomDiv(address, val);
    else if (REDUCE == MIN)
      atomMin(address, val);
    else if (REDUCE == MAX)
      atomMax(address, val);
  }
};

}
}} // namespace at::native