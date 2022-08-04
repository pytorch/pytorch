#version 450 core
#define PRECISION $precision
#define FORMAT    $format

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2)         uniform PRECISION restrict           Block {
  ivec3 size;  // output size
  int ch;   // channel size of the output
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    const int z0a = 2 * ((4 * pos.z) / uBlock.ch) * uBlock.ch + ((4 * pos.z) % uBlock.ch);
    const int z1a = 2 * ((4 * pos.z + 1) / uBlock.ch) * uBlock.ch + ((4 * pos.z + 1) % uBlock.ch);
    const int z2a = 2 * ((4 * pos.z + 2) / uBlock.ch) * uBlock.ch + ((4 * pos.z + 2) % uBlock.ch);
    const int z3a = 2 * ((4 * pos.z + 3) / uBlock.ch) * uBlock.ch + ((4 * pos.z + 3) % uBlock.ch);

    const int z0b = z0a + uBlock.ch;
    const int z1b = z1a + uBlock.ch;
    const int z2b = z2a + uBlock.ch;
    const int z3b = z3a + uBlock.ch;

    const float v0a = texelFetch(uInput, ivec3(pos.x, pos.y, z0a / 4), 0)[z0a % 4];
    const float v0b = texelFetch(uInput, ivec3(pos.x, pos.y, z0b / 4), 0)[z0b % 4];
    const float v1a = texelFetch(uInput, ivec3(pos.x, pos.y, z1a / 4), 0)[z1a % 4];
    const float v1b = texelFetch(uInput, ivec3(pos.x, pos.y, z1b / 4), 0)[z1b % 4];
    const float v2a = texelFetch(uInput, ivec3(pos.x, pos.y, z2a / 4), 0)[z2a % 4];
    const float v2b = texelFetch(uInput, ivec3(pos.x, pos.y, z2b / 4), 0)[z2b % 4];
    const float v3a = texelFetch(uInput, ivec3(pos.x, pos.y, z3a / 4), 0)[z3a % 4];
    const float v3b = texelFetch(uInput, ivec3(pos.x, pos.y, z3b / 4), 0)[z3b % 4];

    imageStore(
        uOutput,
        pos,
        vec4(
            v0a * (1 / (1 + exp(-1 * v0b))),
            v1a * (1 / (1 + exp(-1 * v1b))),
            v2a * (1 / (1 + exp(-1 * v2b))),
            v3a * (1 / (1 + exp(-1 * v3b)))
        )
    );
  }
}
