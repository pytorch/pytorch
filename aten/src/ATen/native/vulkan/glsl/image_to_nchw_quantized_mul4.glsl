#version 450 core
#define PRECISION $precision

layout(std430) buffer;

/*
 * Input Sampler
 */
layout(set = 0, binding = 0) uniform PRECISION isampler3D uImage;

/*
 * Output Buffer
 */
layout(set = 0, binding = 1) buffer PRECISION restrict writeonly Buffer {
  uint data[];
}
uBuffer;

/*
 * Params Buffer
 */
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // xyz contain the extents of the input texture, w contains HxW to help
  // calculate buffer offsets
  ivec4 in_extents;
}
uBlock;

/*
 * Local Work Group in_extents
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
    // each instance of the shader writes out four elements of the output
    // by processing 4 consecutive texels at the same depth.
    // global size = {HxW / 4, 1u, z_extent}.
    // this shader requires HxW to be a multiple of 4, so that multiple
    // planes can be processed in parallel

  if (4 * pos.x >= uBlock.in_extents.w ||
      pos.y > 0 ||
      pos.z >= uBlock.in_extents.z) {
    return;
  }

  ivec4 xy_pos = ivec4(0, 1, 2, 3) + 4 * pos.x;
    // each output element is a uint32 made up four consecutive uint8 from the
    // input in nchw format. xy_pos contains the positions of these four
    // elements from the input in the flatten out HxW plane.

  ivec4 x_pos = xy_pos % uBlock.in_extents.x;
  ivec4 y_pos = xy_pos / uBlock.in_extents.x;
    // we divide this "flatten out position" by H, to find the positions along
    // the y-axis (height) and we compute its reminder mod H, to find the
    // position along the x-axis (width).

  const ivec4 intex0 = texelFetch(uImage, ivec3(x_pos[0], y_pos[0], pos.z), 0);
  const ivec4 intex1 = texelFetch(uImage, ivec3(x_pos[1], y_pos[1], pos.z), 0);
  const ivec4 intex2 = texelFetch(uImage, ivec3(x_pos[2], y_pos[2], pos.z), 0);
  const ivec4 intex3 = texelFetch(uImage, ivec3(x_pos[3], y_pos[3], pos.z), 0);

  const int base_index = 4 * pos.x + 4 * uBlock.in_extents.w * pos.z;
  const ivec4 buf_indices =
      base_index + ivec4(0, 1, 2, 3) * uBlock.in_extents.w;

  for (int i = 0; i < 4; i += 1) {
    uint ui32 = (uint(intex3[i] & 0xFF) << 24)
              | (uint(intex2[i] & 0xFF) << 16)
              | (uint(intex1[i] & 0xFF) << 8)
              | (uint(intex0[i] & 0xFF));
    uBuffer.data[buf_indices[i] / 4] = ui32;
  }
}
