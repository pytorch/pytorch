#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#include "random.h"

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION restrict Block {
  ivec3 size;
  float mean;
  float std;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size))) {
    vec4 v = vec4(
        get_gaussrand(ivec4(pos, -20), uBlock.mean, uBlock.std),
        get_gaussrand(ivec4(pos, 40), uBlock.mean, uBlock.std),
        get_gaussrand(ivec4(pos, -30), uBlock.mean, uBlock.std),
        get_gaussrand(ivec4(pos, 15), uBlock.mean, uBlock.std));
    imageStore(uOutput, pos, v);
  }
}
