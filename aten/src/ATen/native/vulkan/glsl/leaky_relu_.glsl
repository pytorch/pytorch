#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict image3D uOutput;
layout(set = 0, binding = 1)         uniform PRECISION restrict Block {
  ivec4 size;
  float negative_slope;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    const vec4 inval = imageLoad(uOutput, pos);
    const vec4 negative_values = vec4(lessThan(inval, vec4(0.0f)));
    const vec4 positive_values = vec4(1.0) - negative_values;
    const vec4 mask = negative_values * vec4(uBlock.negative_slope) + positive_values;
    const vec4 outval = inval * mask;
    imageStore(uOutput, pos, outval);
  }
}
