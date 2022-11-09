#version 450 core
#define PRECISION $precision
#define FORMAT $format

/*
 * TILE_SIZE = (1, 1, 1)
 * WEIGHT_STORAGE = TEXTURE_3D
 * BIAS_STORAGE = TEXTURE_3D
 * Note that for DW kernel IC = 1 so the weight layout is really OC4, H, W, 4oc
 */

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

/*
 * Output Image
 */
layout(set = 0, binding = 0, rgba8ui) uniform PRECISION restrict writeonly uimage3D uOutput;

/*
 * Input Textures
 */
layout(set = 0, binding = 1) uniform PRECISION isampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION isampler3D uKernel;
layout(set = 0, binding = 3) uniform PRECISION isampler3D uBias;

/*
 * Params Buffer
 */
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  // quantization scales, xyzw corresponds to output, input, kernel, bias
  vec4 scales;
  // quantization zero points, xyzw corresponds to output, input, kernel, bias
  ivec4 zero_points;
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

/*
 * Local Work Group
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Dequantizes a float texel based on a scale and zero point.
 */
vec4 dequantize(vec4 tex, float scale, int zero_point) {
  return scale * (tex - zero_point);
}

/*
 * Quantizes a float texel based on a scale and zero point.
 */
uvec4 quantize(vec4 tex, float scale, int zero_point) {
  return uvec4(tex / scale + zero_point);
}

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  // Return if this global position is outside output texture bounds
  if (any(greaterThanEqual(pos, uBlock.out_extents.xyz))) {
    return;
  }

  // Compute the index of the top-left element of the overlay region. Note that
  // negative indices can be produced indicating that the top-left element is in
  // a region added by padding.
  const ivec2 ipos = pos.xy * uBlock.stride - uBlock.padding;

  // Compute the start and end of the input indices to load. Padding is assumed
  // to be constant 0 padding, so any reads from the padding region is skipped.
  const ivec2 start = max(ivec2(0), ipos);
  const ivec2 end = min(ipos + uBlock.overlay_region.xy, uBlock.in_extents.xy);
  // Compute the start of the kernel based on how far we are skipping ahead when
  // reading the input
  const ivec2 kstart = (start - ipos) / uBlock.dilate;

  vec4 sum = dequantize(
      texelFetch(uBias, ivec3(pos.z, 0, 0), 0),
      uBlock.scales.w,
      uBlock.zero_points.w);

  const int dil_y = uBlock.dilate.y;
  const int dil_x = uBlock.dilate.x;
  for (int y = start.y, ky = kstart.y; y < end.y; y += dil_y, ky++) {
    for (int x = start.x, kx = kstart.x; x < end.x; x += dil_x, kx++) {
      // The weight kernel was rearranged so that every NxN filter was flattened
      // so that it fits on one row. Each filter was then stacked on top of each
      // other vertically.
      const int k_ind = kx + ky * uBlock.kernel_size.x;

      const vec4 k_tex = dequantize(
          texelFetch(uKernel, ivec3(k_ind, pos.z, 0), 0),
          uBlock.scales.z,
          uBlock.zero_points.z);
      const vec4 in_tex = dequantize(
          texelFetch(uInput, ivec3(x, y, pos.z), 0),
          uBlock.scales.y,
          uBlock.zero_points.y);

      sum = fma(in_tex, k_tex, sum);
    }
  }

  uvec4 out_tex = quantize(
      clamp(sum, uBlock.clamp_thresh.x, uBlock.clamp_thresh.y),
      uBlock.scales.x,
      uBlock.zero_points.x);

  imageStore(uOutput, pos, out_tex);
}
