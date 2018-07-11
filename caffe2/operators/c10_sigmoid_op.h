#pragma once

#include "caffe2/core/tensor.h"
#include "caffe2/utils/Array.h"

namespace caffe2 {

struct SigmoidOp final {
    static constexpr const char* name = "sigmoid";

    using Signature = void(const Tensor<CPUContext>& input, Tensor<CPUContext>* output);

    static constexpr c10::guts::array<const char*, 2> parameter_names = {{"input", "output"}};
};

}
