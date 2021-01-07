#pragma once

#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

vTensor pack_image2d_h2w2(vTensor v_src, api::Context* context, api::Command::Buffer& command_buffer);
vTensor unpack_image2d_h2w2(vTensor v_src, c10::SmallVector<int64_t, 4u> output_sizes, api::Context* context, api::Command::Buffer& command_buffer);

vTensor unpack_image1x1(vTensor v_src, c10::SmallVector<int64_t, 4u> output_sizes, api::Context* context, api::Command::Buffer& command_buffer);

}
}
}
}
