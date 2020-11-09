#version 450 core
#define PRECISION $precision

layout(std430) buffer;
layout(std430) uniform;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba16f) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)          uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2)          uniform           restrict           Block {
  int input_width;
  int input_height;
  int output_width;
  int output_height;
  float scale_x;
  float scale_y;
}
uBlock;

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  const int ow = uBlock.output_width;
  const int oh = uBlock.output_height;
  if (pos.x < ow && pos.y < oh) {
    const int iw = uBlock.input_width;
    const int ih = uBlock.input_height;
    float srcX = float(pos.x) * uBlock.scale_x;
    int x1 = int(floor(srcX));
    int x11 = clamp(x1, 0, iw - 1);
    float srcY = float(pos.y) * uBlock.scale_y;
    int y1 = int(floor(srcY));
    int y11 = clamp(y1, 0, ih - 1);
    vec4 outValue = texelFetch(uInput, ivec3(x11, y11, pos.z), 0);
    imageStore(uOutput, pos, outValue);
  }
}
