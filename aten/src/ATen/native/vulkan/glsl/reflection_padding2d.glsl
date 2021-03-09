#version 450 core
#define PRECISION $precision

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1) uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict           Block {
  ivec4 size;
  ivec4 pad;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    int xoff_pre  = 2*max(uBlock.pad.x - pos.x, 0);
    int xoff_post = 2*max(pos.x - (uBlock.size.x - 1 - uBlock.pad.y), 0);
    int yoff_pre  = 2*max(uBlock.pad.z - pos.y, 0);
    int yoff_post = 2*max(pos.y - (uBlock.size.y - 1 - uBlock.pad.w), 0);
    ivec3 inpos = ivec3(
        pos.x - uBlock.pad.x + xoff_pre - xoff_post,
        pos.y - uBlock.pad.z + yoff_pre - yoff_post,
        pos.z);
    imageStore(uOutput, pos, texelFetch(uInput, inpos, 0));
  }
}
