#version 450 core
#define PRECISION $precision

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba16f) uniform PRECISION restrict image3D   uOutput;
layout(set = 0, binding = 1)          uniform PRECISION          sampler3D uInput0;
layout(set = 0, binding = 2)          uniform PRECISION restrict Block {
  ivec4 size;
  ivec3 isize;
  float alpha;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    const ivec3 input_pos = pos % uBlock.isize.xyz;
    imageStore(
        uOutput,
        pos,
        imageLoad(uOutput, pos) + uBlock.alpha * texelFetch(uInput0, input_pos, 0));
  }
}
