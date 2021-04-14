#ifndef MetalRoIAlign_h
#define MetalRoIAlign_h

#include <torch/script.h>

namespace torch {
namespace fb{
namespace metal {

torch::Tensor RoIAlign(
    const torch::Tensor& features,
    const torch::Tensor& rois,
    std::string order,
    double spatial_scale,
    int64_t aligned_height,
    int64_t aligned_width,
    int64_t sampling_ratio,
    bool aligned,
    c10::optional<std::vector<torch::Tensor>>);
}
}
}

#endif /* MetalRoIAlign_h */
