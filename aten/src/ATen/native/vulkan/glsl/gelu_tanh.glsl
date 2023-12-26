#version 450 core
#define PRECISION $precision
#define FORMAT    $format

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2)         uniform PRECISION restrict           Block {
  ivec4 size;
  float kBeta; /* M_SQRT2 * M_2_SQRTPI * 0.5 */
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    const vec4 inval = texelFetch(uInput, pos, 0);
    const vec4 invalcube = inval * inval * inval;
    const vec4 inner = vec4(uBlock.kBeta) * (inval + vec4(0.044715) * invalcube);
    const vec4 outval = vec4(0.5) * inval * (vec4(1.0) + tanh(inner));
    imageStore(uOutput, pos, outval);
  }
}
