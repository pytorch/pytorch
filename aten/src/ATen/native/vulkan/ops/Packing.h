#pragma once

#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

vTensor pack_image2d_h2w2(vTensor v_src, api::Context* context, api::Command::Buffer& command_buffer);
vTensor unpack_image2d_h2w2(vTensor v_src, uint32_t out_h, uint32_t out_w, api::Context* context, api::Command::Buffer& command_buffer);

}
}
}
}
