#ifndef CAFFE2_UTILS_MATH_DETAIL_H_
#define CAFFE2_UTILS_MATH_DETAIL_H_
namespace caffe2 {

class CPUContext;

namespace math {
namespace detail {

// proxy to a class because of partial specialization limitations for functions

template<typename T, class Context, int FixedSize>
struct ScaleImpl {
  inline void operator()(
      const int N,
      const float alpha,
      const T* x,
      T* y,
      Context* context) {
    Scale(N, alpha, x, y, context);
  }
};

// Put light-weight implementations in .h file to enable inlining
template<typename T>
struct ScaleImpl<T, CPUContext, 1> {
  inline void operator()(
      const int N,
      const float alpha,
      const T* x,
      T* y,
      CPUContext* /*context*/) {
    DCHECK_EQ(N, 1);
    *y = *x * alpha;
  }
};

template<typename T, class Context, int FixedSize>
struct AxpyImpl {
  inline void operator()(
      const int N,
      const float alpha,
      const T* x,
      T* y,
      Context* context) {
    Axpy(N, alpha, x, y, context);
  }
};

// Put light-weight implementations in .h file to enable inlining
template<typename T>
struct AxpyImpl<T, CPUContext, 1> {
  inline void operator()(
      const int N,
      const float alpha,
      const T* x,
      T* y,
      CPUContext* /*context*/) {
    DCHECK_EQ(N, 1);
    *y += *x * alpha;
  }
};


}  // namespace detail

template <typename T, class Context, int FixedSize>
inline void ScaleFixedSize(
    const int N,
    const float alpha,
    const T* x,
    T* y,
    Context* context) {
  detail::ScaleImpl<T, Context, FixedSize>()(N, alpha, x, y, context);
}

template <typename T, class Context, int FixedSize>
inline void AxpyFixedSize(
    const int N,
    const float alpha,
    const T* x,
    T* y,
    Context* context) {
  detail::AxpyImpl<T, Context, FixedSize>()(N, alpha, x, y, context);
}

}  // namespace math
}  // namespace caffe2

#endif  // CAFFE2_UTILS_MATH_DETAIL_H_
