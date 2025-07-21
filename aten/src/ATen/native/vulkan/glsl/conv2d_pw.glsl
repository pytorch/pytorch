/*
 * TILE_SIZE = (${OUTPUT_TILE_SIZE[0]}, ${OUTPUT_TILE_SIZE[1]}, 1)
 * WEIGHT_STORAGE = TEXTURE_2D
 * WEIGHT_STORAGE_LAYOUT = OC4,IC4,4ic,4oc
 * BIAS_STORAGE = TEXTURE_2D
 * REGISTER_FOR = ${REGISTER_FOR}
 */
#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

// clang-format off
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
// clang-format on
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION sampler2D uKernel;
layout(set = 0, binding = 3) uniform PRECISION sampler2D uBias;
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

/*
 * Computes a 2D pointwise convolution of a 2x2 output tile. Calculating an
 * output tile for pointwise convolution is more efficient because the kernel
 * size is only 1x1, making it much easier to reuse loaded texels from uKernel.
 */
void main() {
  const ivec3 gpos = ivec3(gl_GlobalInvocationID);

  // Output position for OUTPUT_TILE_SIZE[0], OUTPUT_TILE_SIZE[1] = 2, 2
  // +--------+--------+
  // | pos[0] | pos[1] |
  // +--------+--------+
  // | pos[2] | pos[3] |
  // +--------+--------+
  ivec3 pos[${OUTPUT_TILE_SIZE[0]} * ${OUTPUT_TILE_SIZE[1]}];
  for (int y = 0, i = 0; y < ${OUTPUT_TILE_SIZE[1]}; ++y) {
    for (int x = 0; x < ${OUTPUT_TILE_SIZE[0]}; ++x) {
      pos[i] = ivec3(
          gpos.x * ${OUTPUT_TILE_SIZE[0]} + x, gpos.y * ${OUTPUT_TILE_SIZE[1]} + y, gpos.z);
      i++;
    }
  }

  // If the top left position is out of bounds, then this invocation will have
  // no work to do.
  if (any(greaterThanEqual(pos[0], uBlock.out_extents.xyz))) {
    return;
  }

  // Compute the index of the input texture that needs to be loaded for each
  // output position. Note that negative indices can be produced indicating that
  // the top-left element is in a region added by padding.
  ivec2 ipos[${OUTPUT_TILE_SIZE[0]} * ${OUTPUT_TILE_SIZE[1]}];
  for (int i = 0; i < ${OUTPUT_TILE_SIZE[0]} * ${OUTPUT_TILE_SIZE[1]}; ++i) {
    ipos[i] = pos[i].xy * uBlock.stride - uBlock.padding;
  }

  vec4 sum[${OUTPUT_TILE_SIZE[0]} * ${OUTPUT_TILE_SIZE[1]}];
  sum[0] = texelFetch(uBias, ivec2(gpos.z, 0), 0);
  for (int i = 1; i < ${OUTPUT_TILE_SIZE[0]} * ${OUTPUT_TILE_SIZE[1]}; ++i) {
    sum[i] = sum[0];
  }

  // Since the kernel is 1x1, we only have to loop over the depth dimension.
  const int ic_aligned = uBlock.overlay_region.z;
  for (int z = 0, z4 = 0; z < ic_aligned; z += 4, ++z4) {
    // During prepacking, the weight tensor has been permuted so that the
    // channel (IC) dim is along the x axis, and the batch (OC) dim is along
    // the z axis.
    vec4 in_tex[${OUTPUT_TILE_SIZE[0]} * ${OUTPUT_TILE_SIZE[1]}];
    const vec4 ktex_0 = texelFetch(uKernel, ivec2(z + 0, gpos.z), 0);
    const vec4 ktex_1 = texelFetch(uKernel, ivec2(z + 1, gpos.z), 0);
    const vec4 ktex_2 = texelFetch(uKernel, ivec2(z + 2, gpos.z), 0);
    const vec4 ktex_3 = texelFetch(uKernel, ivec2(z + 3, gpos.z), 0);

    for (int i = 0; i < ${OUTPUT_TILE_SIZE[1]} * ${OUTPUT_TILE_SIZE[0]}; ++i) {
      in_tex[i] = texelFetch(uInput, ivec3(ipos[i], z4), 0);
    }

    for (int i = 0; i < ${OUTPUT_TILE_SIZE[1]} * ${OUTPUT_TILE_SIZE[0]}; ++i) {
      // For 2x2 tile size algorithm works as follows.
      // To explain the calculations below, the contents one in_tex and the
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
      //  which is what is expressed in the following calculations. This is done
      //  for each output position.
      sum[i] = fma(in_tex[i].xxxx, ktex_0, sum[i]);
      sum[i] = fma(in_tex[i].yyyy, ktex_1, sum[i]);
      sum[i] = fma(in_tex[i].zzzz, ktex_2, sum[i]);
      sum[i] = fma(in_tex[i].wwww, ktex_3, sum[i]);
    }
  }

  for (int i = 0; i < ${OUTPUT_TILE_SIZE[1]} * ${OUTPUT_TILE_SIZE[0]}; ++i) {
    if (all(lessThan(pos[i], uBlock.out_extents.xyz))) {
      imageStore(
          uOutput,
          pos[i],
          clamp(sum[i], uBlock.clamp_thresh.x, uBlock.clamp_thresh.y));
    }
  }
}
