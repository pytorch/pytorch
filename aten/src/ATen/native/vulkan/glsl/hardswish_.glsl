#version 450 core
#define PRECISION $precision

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba16f) uniform PRECISION restrict image3D uOutput;
layout(set = 0, binding = 1)          uniform PRECISION restrict Block {
  ivec4 size;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    const vec4 inval = imageLoad(uOutput, pos);
    const vec4 mask1 = vec4(lessThan(inval, vec4(3.0f)));
    const vec4 mask2 = vec4(greaterThan(inval, vec4(-3.0f)));
    const vec4 outval = mask2*inval*(mask1*((inval+3.0f)/6.0f) + 1.0f - mask1);
    imageStore(uOutput, pos, outval);
  }
}
