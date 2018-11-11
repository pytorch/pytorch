#ifndef CAFFE2_NET_JIT_CONSTANTS_H
#define CAFFE2_NET_JIT_CONSTANTS_H

namespace caffe2 {
namespace jit {

// Max number of futures passed as arguments to a Fork task
const unsigned int MAX_FUTURE_INPUTS = 10e3;

} // namespace jit
} // namespace caffe2

#endif // CAFFE2_NET_JIT_CONSTANTS_H
