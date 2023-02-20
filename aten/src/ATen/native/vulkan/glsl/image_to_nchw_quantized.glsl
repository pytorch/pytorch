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
    // each instance of the shader writes out a single element of the output
    // the global size matches the size of the output, in other words:
    // global size = {div_up(numel, 4), 1u, 1u}
    // pos = {pos.x, 1, 1} where pos.x is the index of the output element

  ivec4 input_pos = ivec4(0, 1, 2, 3) + 4 * pos.x;
    // each output element is a uint32 made up four consecutive uint8 from the
    // input in nchw format. input_pos contains the positions of these four
    // elements from the input in nchw format.

  ivec4 nc_pos = input_pos / uBlock.in_extents.w;
    // we divide by HxW (uBlock.in_extents.w), to find the position along the
    // batch/channel axis of these four elements.

  ivec4 w_pos = input_pos % uBlock.in_extents.w;
    // we compute the reminder mod HxW, to find the positions in the flatten
    // out HxW plane.

  ivec4 x_pos = w_pos % uBlock.in_extents.x;
  ivec4 y_pos = w_pos / uBlock.in_extents.x;
    // we divide this "flatten out position" by H, to find the positions along
    // the y-axis (height) and we compute its reminder mod H, to find the
    // position along the x-axis (width).

  ivec4 z_pos = nc_pos / 4;
  ivec4 ix = nc_pos % 4;
    // z_pos contains the texel positions along the z-axis, and ix the
    // indices inside each texel.

  // now we fetch each uint8 element from the input, and we write out a uint32
  // whose binary representation is equal to: tex3 tex2 tex1 tex0

  int tex0 = texelFetch(uImage, ivec3(x_pos[0], y_pos[0], z_pos[0]), 0)[ix[0]];
  int tex1 = texelFetch(uImage, ivec3(x_pos[1], y_pos[1], z_pos[1]), 0)[ix[1]];
  int tex2 = texelFetch(uImage, ivec3(x_pos[2], y_pos[2], z_pos[2]), 0)[ix[2]];
  int tex3 = texelFetch(uImage, ivec3(x_pos[3], y_pos[3], z_pos[3]), 0)[ix[3]];

  uint ui32 = (uint(tex3 & 0xFF) << 24)
            | (uint(tex2 & 0xFF) << 16)
            | (uint(tex1 & 0xFF) << 8)
            | (uint(tex0 & 0xFF));

  uBuffer.data[pos.x] = ui32;
}
