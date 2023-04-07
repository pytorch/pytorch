//  Copyright © 2023 Apple Inc.
#pragma once

namespace at {
namespace native {
namespace mps {
bool dispatchNativeBinaryKernel(const Tensor& self,
                        const Tensor& other,
                        const Tensor& output,
                        const Scalar& alpha,
                        const std::string& op_name);
}
}
}
