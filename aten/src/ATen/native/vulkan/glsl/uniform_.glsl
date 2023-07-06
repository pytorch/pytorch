#version 450 core
#define PRECISION $precision
#define FORMAT $format

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION restrict Block {
  ivec3 size;
  float from;
  float to;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

uint pcg_hash(uint v) {
  // From: https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
  uint state = v * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

float rand2(ivec4 pos) {
  uint s =
      pcg_hash(pos.x) + pcg_hash(pos.y) + pcg_hash(pos.z) + pcg_hash(pos.w);
  return fract(s / 1234567.0);
}

float get_uniform(ivec4 pos) {
  float v = rand2(pos);
  return uBlock.from + v * (uBlock.to - uBlock.from);
}

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size))) {
    vec4 v = vec4(
        get_uniform(ivec4(pos, -20)),
        get_uniform(ivec4(pos, 40)),
        get_uniform(ivec4(pos, -30)),
        get_uniform(ivec4(pos, 15)));
    imageStore(uOutput, pos, v);
  }
}
