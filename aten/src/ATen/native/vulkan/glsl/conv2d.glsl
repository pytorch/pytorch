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
  ivec4 kernel;
  ivec2 stride;
  ivec2 padding;
  ivec2 dilate;
  vec2 clamp;
  ivec3 src_kernel;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  /* Dynamically Uniform */
  const ivec3 size = imageSize(uOutput);
  const ivec3 isize = textureSize(uInput, 0);

  if (all(lessThan(pos, size))) {
    const ivec2 ipos = pos.xy * uBlock.stride - uBlock.padding;

    const ivec2 start = max(ivec2(0), ipos);
    const ivec2 end = min(ipos + uBlock.kernel.xy, isize.xy);
    const ivec2 kstart = (start - ipos) / uBlock.dilate;

    vec4 sum = uBias.data[pos.z];

    for (int z = 0; z < uBlock.kernel.z; z+=4) {
      const int z4 = z/4;
      for (int y = start.y, ky = kstart.y + pos.z * uBlock.src_kernel.y; y < end.y; y += uBlock.dilate.y, ++ky) {
        for (int x = start.x, kx = kstart.x + z4*uBlock.src_kernel.z; x < end.x; x += uBlock.dilate.x, kx+=4) {
          const vec4 In = texelFetch(uInput, ivec3(x, y, z4), 0);

          sum = fma(In.xxxx, texelFetch(uKernel, ivec3(0 + kx, ky, 0), 0), sum);
          sum = fma(In.yyyy, texelFetch(uKernel, ivec3(1 + kx, ky, 0), 0), sum);
          sum = fma(In.zzzz, texelFetch(uKernel, ivec3(2 + kx, ky, 0), 0), sum);
          sum = fma(In.wwww, texelFetch(uKernel, ivec3(3 + kx, ky, 0), 0), sum);
        }
      }
    }

    imageStore(
        uOutput,
        pos,
        clamp(sum, uBlock.clamp.x, uBlock.clamp.y));
  }
}
