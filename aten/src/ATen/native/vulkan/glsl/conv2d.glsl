#version 450 core
#define PRECISION $precision
#define FORMAT $format

/*
 * TILE_SIZE = (1, 1, 1)
 * WEIGHT_STORAGE = TEXTURE_2D
 * BIAS_STORAGE = TEXTURE_2D
 * REGISTER_FOR = ('conv2d', ['catchall'])
 */

layout(std430) buffer;

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

/*
 * Local Work Group
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Computes a 2D convolution. Each shader invocation calculates the output at
 * a single output location.
 */
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
  // reading the input. Note that these are "canonical" indices.
  ivec2 kstart = (start - ipos) / uBlock.dilate;
  // During prepacking, the weight tensor was rearranged in order to optimize
  // for data access linearity in this shader. Therefore we need to adjust the
  // canonical coordinates to the corresponding index in the rearranged weight
  // tensor. the x coordinate is multipled by 4 since each group of 4 channels
  // is folded into the X axis. The y coordinate is offset based on the z
  // coordinate because the 2D planes were stacked atop each other vertically.
  kstart.x *= 4;
  kstart.y += pos.z * uBlock.kernel_size.y;

  // Perform the convolution by iterating over the overlay region
  vec4 sum = texelFetch(uBias, ivec2(pos.z, 0), 0);
  const int dil_y = uBlock.dilate.y;
  const int dil_x = uBlock.dilate.x;
  const int ic4 = uBlock.overlay_region.z / 4;
  for (int z4 = 0; z4 < ic4; ++z4, kstart.x += uBlock.kernel_size.x * 4) {
    for (int y = start.y, ky = kstart.y; y < end.y; y += dil_y, ++ky) {
      for (int x = start.x, kx = kstart.x; x < end.x; x += dil_x, kx += 4) {
        const vec4 in_tex = texelFetch(uInput, ivec3(x, y, z4), 0);

        // To explain the calculation below, the contents of in_tex and the
        // group of 4 texels loaded from uKernel are shown:
        //
        //   in_tex               uKernel
        //    -x->                   ---x--->
        //   +---+              +----+----+----+----+
        // ^ | w |           ^  | D0 | D1 | D2 | D3 |
        // | +---+           |  +----+----+----+----+
        // | | z |           |  | C0 | C1 | C2 | C3 |
        // z +---+           z  +----+----+----+----+
        // | | y |           |  | B0 | B2 | B2 | B3 |
        // | +---+           |  +----+----+----+----+
        //   | x |              | A0 | A1 | A2 | A3 |
        //   +---+              +----+----+----+----+
        //
        // In the uKernel graphic, cells sharing the same letter are from
        // the same batch/output channel index, and the number denotes a unique
        // channel index. To calculate the output texel, the following
        // calculation is performed:
        //
        //  +---+ +----+   +---+ +----+   +---+ +----+   +---+ +----+
        //  | x | | D0 |   | y | | D1 |   | z | | D2 |   | w | | D3 |
        //  +---+ +----+   +---+ +----+   +---+ +----+   +---+ +----+
        //  | x | | C0 |   | y | | C1 |   | z | | C2 |   | w | | C3 |
        //  +---+X+----+ + +---+X+----+ + +---+X+----+ + +---+X+----+
        //  | x | | B0 |   | y | | B1 |   | z | | B2 |   | w | | B3 |
        //  +---+ +----+   +---+ +----+   +---+ +----+   +---+ +----+
        //  | x | | A0 |   | y | | A1 |   | z | | A2 |   | w | | A3 |
        //  +---+ +----+   +---+ +----+   +---+ +----+   +---+ +----+
        //
        //  which is what is expressed in the following calculations.

        const vec4 ktex_0 = texelFetch(uKernel, ivec2(kx + 0, ky), 0);
        sum = fma(in_tex.xxxx, ktex_0, sum);

        const vec4 ktex_1 = texelFetch(uKernel, ivec2(kx + 1, ky), 0);
        sum = fma(in_tex.yyyy, ktex_1, sum);

        const vec4 ktex_2 = texelFetch(uKernel, ivec2(kx + 2, ky), 0);
        sum = fma(in_tex.zzzz, ktex_2, sum);

        const vec4 ktex_3 = texelFetch(uKernel, ivec2(kx + 3, ky), 0);
        sum = fma(in_tex.wwww, ktex_3, sum);
      }
    }
  }

  imageStore(
      uOutput, pos, clamp(sum, uBlock.clamp_thresh.x, uBlock.clamp_thresh.y));
}
