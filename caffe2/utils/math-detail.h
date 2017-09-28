/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
