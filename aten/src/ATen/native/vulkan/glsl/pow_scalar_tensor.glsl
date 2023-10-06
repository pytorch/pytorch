#version 450 core
#define PRECISION $precision
#define FORMAT $format

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 size;
  float other;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, uBlock.size.xyz))) {
    return;
  }

  const vec4 v = texelFetch(uInput, pos, 0);
  const vec4 base = vec4(uBlock.other);
  imageStore(uOutput, pos, pow(base, v));
}
