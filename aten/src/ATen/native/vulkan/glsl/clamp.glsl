#version 450 core
#define PRECISION $precision
layout(std430) buffer;
layout(std430) uniform;
layout(set = 0, binding = 0, rgba16f) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)          uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2)          uniform           restrict           Block {
  ivec3 WHC;
  float minValue;
  float maxValue;
} uBlock;

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (all(lessThan(pos, uBlock.WHC))) {
    imageStore(
        uOutput,
        pos,
        clamp(texelFetch(uInput, pos, 0), uBlock.minValue, uBlock.maxValue));
  }
}
