#version 450 core
#define PRECISION $precision
#define FORMAT    $format

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2)         uniform PRECISION restrict           Block {
  ivec3 size;  // output size
  int chext;   // channel extent of the output
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    const int z0 = 2 * (pos.z / uBlock.chext) * uBlock.chext + (pos.z % uBlock.chext);
    const int z1 = z0 + uBlock.chext;
    imageStore(
        uOutput,
        pos,
        texelFetch(uInput, ivec3(pos.x, pos.y, z0), 0)
            * 1 / (1 + exp(-1 * texelFetch(uInput, ivec3(pos.x, pos.y, z1), 0))));
  }
}
