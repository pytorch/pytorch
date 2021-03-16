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
  ivec4 kernel;
  ivec2 stride;
  ivec2 padding;
  ivec2 dilate;
  vec2 clamp;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    const ivec2 ipos = pos.xy * uBlock.stride - uBlock.padding;

    const ivec2 start = max(ivec2(0), ipos);
    const ivec2 end = min(ipos + uBlock.kernel.xy, uBlock.kernel.zw);
    const ivec2 kstart = (start - ipos) / uBlock.dilate;

    vec4 sum = uBias.data[pos.z];

    for (int y = start.y, ky = kstart.y; y < end.y; y += uBlock.dilate.y, ++ky) {
      for (int x = start.x, kx = kstart.x + ky * uBlock.size.w; x < end.x; x += uBlock.dilate.x, ++kx) {
        sum = fma(
            texelFetch(uInput, ivec3(x, y, pos.z), 0),
            texelFetch(uKernel, ivec3(kx, pos.z, 0), 0),
            sum);
      }
    }

    imageStore(
        uOutput,
        pos,
        clamp(sum, uBlock.clamp.x, uBlock.clamp.y));
  }
}
