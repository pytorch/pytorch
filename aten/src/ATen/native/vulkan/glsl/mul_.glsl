#version 450 core
#define PRECISION $precision

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba16f) uniform PRECISION restrict image3D   uOutput;
layout(set = 0, binding = 1)          uniform PRECISION          sampler3D uInput0;
layout(set = 0, binding = 2)          uniform PRECISION restrict Block {
  ivec3 size;
  ivec4 isize;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    const ivec3 input_pos = ivec3(pos.x%uBlock.isize.x, pos.y%uBlock.isize.y, pos.z%uBlock.isize.z);
    imageStore(
        uOutput,
        pos,
        imageLoad(uOutput, pos) * texelFetch(uInput0, input_pos, 0));
  }
}
