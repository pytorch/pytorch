#version 450 core
#define PRECISION $precision
#define FORMAT $format

/*
 * TILE_SIZE = (1, 1, 1)
 * WEIGHT_STORAGE = TEXTURE_2D
 * BIAS_STORAGE = TEXTURE_2D
 */

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

/*
 * Output Image
 */
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;

/*
 * Input Textures
 */
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION sampler2D uKernel;
layout(set = 0, binding = 3) uniform PRECISION sampler2D uBias;

/*
 * Params Buffer
 */
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  // extents of the output texture
  ivec4 out_extents;
  // extents of the input texture
  ivec4 in_extents;
  // size of the overlay region of the kernel
  ivec4 overlay_region;
  // width and height of the kernel
  ivec2 kernel_size;
  // convolution parameters
  ivec2 stride;
  ivec2 padding;
  ivec2 dilate;
  vec2 clamp_thresh;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  // Return if this global position is outside output texture bounds
  if (any(greaterThanEqual(pos, uBlock.out_extents.xyz))) {
    return;
  }

  const vec2 ksize = vec2(uBlock.kernel_size);
  const vec2 stride = vec2(uBlock.stride);
  const vec2 padding = vec2(uBlock.padding);

  ivec2 ipos = pos.xy + uBlock.padding;
  vec2 ipos_f = vec2(ipos);

  const ivec2 start = max(ivec2(0), ivec2(ceil((ipos_f - ksize + 1) / stride)));
  const ivec2 end =
      min(uBlock.in_extents.xy, ivec2(floor(ipos_f / stride)) + 1);
  ivec2 kstart = start;

  vec4 sum = texelFetch(uBias, ivec2(pos.z, 0), 0);

  const int ic4 = uBlock.overlay_region.z;

  int ky_start = uBlock.overlay_region.y - 1 -
      (ipos.y - uBlock.stride.y * start.y) + pos.z * uBlock.kernel_size.y;
  int kx_start =
      (uBlock.overlay_region.x - 1 - (ipos.x - uBlock.stride.x * start.x)) *
      ic4;
  int kx_stride = ic4 * (uBlock.stride.x - 1);

  for (int y = start.y, ky = ky_start; y < end.y; ++y, ky += uBlock.stride.y) {
    int kx = kx_start;
    for (int x = start.x, kx = kx_start; x < end.x; ++x, kx += kx_stride) {
      for (int z4 = 0; z4 < ic4 / 4; ++z4, kx += 4) {
        const vec4 In = texelFetch(uInput, ivec3(x, y, z4), 0);
        const ivec4 kxs = kx + ivec4(0, 1, 2, 3);

        sum = fma(In.xxxx, texelFetch(uKernel, ivec2(kxs.x, ky), 0), sum);
        sum = fma(In.yyyy, texelFetch(uKernel, ivec2(kxs.y, ky), 0), sum);
        sum = fma(In.zzzz, texelFetch(uKernel, ivec2(kxs.z, ky), 0), sum);
        sum = fma(In.wwww, texelFetch(uKernel, ivec2(kxs.w, ky), 0), sum);
      }
    }
  }

  imageStore(
      uOutput, pos, clamp(sum, uBlock.clamp_thresh.x, uBlock.clamp_thresh.y));
}
