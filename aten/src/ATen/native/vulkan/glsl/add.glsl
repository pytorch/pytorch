#version 450 core
#define PRECISION $precision

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1) uniform PRECISION                    sampler3D uInput0;
layout(set = 0, binding = 2) uniform PRECISION                    sampler3D uInput1;
layout(set = 0, binding = 3) uniform PRECISION restrict           Block {
  ivec3 size;
  float alpha;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size))) {
    imageStore(
        uOutput,
        pos,
        texelFetch(uInput0, pos, 0) + uBlock.alpha * texelFetch(uInput1, pos, 0));
  }
}
