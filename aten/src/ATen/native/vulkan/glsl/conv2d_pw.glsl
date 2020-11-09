#version 450 core
#define PRECISION $precision

layout(std430) buffer;
layout(std430) uniform;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba16f) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)          uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2)          uniform PRECISION                    sampler3D uKernel;
layout(set = 0, binding = 3)          buffer  PRECISION restrict readonly  Bias {
  vec4 data[];
} uBias;
layout(set = 0, binding = 4)          uniform PRECISION restrict           Block {
  ivec2 kernel;
  ivec2 stride;
  ivec2 padding;
  vec2 clamp;
} uBlock;

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  /* Dynamically Uniform */
  const ivec3 size = imageSize(uOutput);
  const ivec3 isize = textureSize(uInput, 0);
  const ivec4 block = pos.z * uBlock.kernel.x + ivec4(0, 1, 2, 3);

  if (all(lessThan(pos, size))) {
    const ivec2 ipos = pos.xy * uBlock.stride - uBlock.padding;

    vec4 sum = uBias.data[pos.z];

    for (int z = 0; z < uBlock.kernel.x; ++z) {
      const vec4 In = texelFetch(uInput, ivec3(ipos.x, ipos.y, z), 0);
      const ivec4 kz = block + 4 * z;

      sum = fma(In.xxxx, texelFetch(uKernel, ivec3(0, 0, kz.x), 0), sum);
      sum = fma(In.yyyy, texelFetch(uKernel, ivec3(0, 0, kz.y), 0), sum);
      sum = fma(In.zzzz, texelFetch(uKernel, ivec3(0, 0, kz.z), 0), sum);
      sum = fma(In.wwww, texelFetch(uKernel, ivec3(0, 0, kz.w), 0), sum);
    }

    imageStore(
        uOutput,
        pos,
        clamp(sum, uBlock.clamp.x, uBlock.clamp.y));
  }
}
