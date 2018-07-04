#pragma once

namespace caffe2 {

// Similar to Axpy that calculate y = a * x + y, but allowing x and y to be
// of different data types.
// It also provides a performance optimization hint (use_a) to see if a is going
// to be 1 or not.
template <typename IN, typename OUT>
void TypedAxpy(int N, const OUT a, const IN* x, OUT* y);

} // namespace caffe2
