#version 450 core
#define PRECISION $precision

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1) uniform PRECISION                    sampler3D uM1;
layout(set = 0, binding = 2) uniform PRECISION                    sampler3D uM2;
layout(set = 0, binding = 3) uniform PRECISION restrict           Block {
  ivec4 size;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    vec4 sum = vec4(0);

    for (int k = 0; k < uBlock.size.w; ++k) {
      sum = fma(
          texelFetch(uM1, ivec3(k, pos.y, pos.z), 0),
          texelFetch(uM2, ivec3(pos.x, k, pos.z), 0),
          sum);
    }

    imageStore(uOutput, pos, sum);
  }
}
