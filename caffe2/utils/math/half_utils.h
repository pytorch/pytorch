#ifndef CAFFE2_UTILS_MATH_HALF_UTILS_H_
#define CAFFE2_UTILS_MATH_HALF_UTILS_H_

#include "caffe2/core/common.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math/utils.h"

namespace caffe2 {
namespace math {
namespace utils {

struct HalfAddFunctor {
  MATH_UTILS_DECL at::Half operator()(const at::Half a, const at::Half b)
      const {
    return convert::To<float, at::Half>(
        convert::To<at::Half, float>(a) + convert::To<at::Half, float>(b));
  }
};

struct HalfSubFunctor {
  MATH_UTILS_DECL at::Half operator()(const at::Half a, const at::Half b)
      const {
    return convert::To<float, at::Half>(
        convert::To<at::Half, float>(a) - convert::To<at::Half, float>(b));
  }
};

struct HalfMulFunctor {
  MATH_UTILS_DECL at::Half operator()(const at::Half a, const at::Half b)
      const {
    return convert::To<float, at::Half>(
        convert::To<at::Half, float>(a) * convert::To<at::Half, float>(b));
  }
};

struct HalfDivFunctor {
  MATH_UTILS_DECL at::Half operator()(const at::Half a, const at::Half b)
      const {
    return convert::To<float, at::Half>(
        convert::To<at::Half, float>(a) / convert::To<at::Half, float>(b));
  }
};

} // namespace utils
} // namespace math
} // namespace caffe2

#endif // CAFFE2_UTILS_MATH_HALF_UTILS_H_
