#version 450 core
#define PRECISION $precision
#define FORMAT    $format

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2)         uniform PRECISION restrict           Block {
  int axis;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  ivec3 spos = pos;
  vec4 sum = vec4(0);
  for(spos[uBlock.axis] = 0; spos!=pos; ++spos[uBlock.axis]) {
    sum += texelFetch(uInput, spos, 0);
  }
  sum += texelFetch(uInput, spos, 0);
  imageStore(uOutput, pos, sum);
}
