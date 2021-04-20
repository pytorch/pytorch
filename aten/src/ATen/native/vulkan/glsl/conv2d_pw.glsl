#version 450 core
#define PRECISION $precision

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1) uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION                    sampler3D uKernel;
layout(set = 0, binding = 3) buffer  PRECISION restrict readonly  Bias {
  vec4 data[];
} uBias;
layout(set = 0, binding = 4) uniform PRECISION restrict           Block {
  ivec4 size;
  ivec2 stride;
  ivec2 padding;
  vec2 clamp;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    const ivec2 ipos = pos.xy * uBlock.stride - uBlock.padding;

    vec4 sum = uBias.data[pos.z];

    for (int z = 0, z4 = 0; z < uBlock.size.w; z += 4, ++z4) {
      const vec4 In = texelFetch(uInput, ivec3(ipos, z4), 0);
      const ivec4 kxs = z + ivec4(0, 1, 2, 3);

      sum = fma(In.xxxx, texelFetch(uKernel, ivec3(kxs.x, pos.z, 0), 0), sum);
      sum = fma(In.yyyy, texelFetch(uKernel, ivec3(kxs.y, pos.z, 0), 0), sum);
      sum = fma(In.zzzz, texelFetch(uKernel, ivec3(kxs.z, pos.z, 0), 0), sum);
      sum = fma(In.wwww, texelFetch(uKernel, ivec3(kxs.w, pos.z, 0), 0), sum);
    }

    imageStore(
        uOutput,
        pos,
        clamp(sum, uBlock.clamp.x, uBlock.clamp.y));
  }
}
