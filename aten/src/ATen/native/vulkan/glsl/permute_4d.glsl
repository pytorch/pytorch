#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/*
 * Output Image
 */
layout(set = 0, binding = 0, FORMAT) uniform PRECISION image3D uOutput;

/*
 * Input Textures
 */
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;

/*
 * Params Buffer
 */
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // output texture size (x=width,y=height,z=depth,w=unused)
  ivec4 size;
  // input texture size (x=width,y=height,z=depth,w=unused)
  ivec4 isize;
  // output tensor size
  uvec4 out_tensor_size;
  // input tensor size
  uvec4 in_tensor_size;
  // output dims
  uvec4 out_ndims;
  // x = output channels aligned to 4, y = input channels aligned to 4
  uvec2 ch_info;
}
uBlock;

/*
 * Local Work Group
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 posOut = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(posOut, uBlock.size.xyz))) {
    const uint max_dst_index = uBlock.out_tensor_size[0] * uBlock.ch_info.x;
    vec4 outval = vec4(0.0);

    for (uint j = 0; j < 4; ++j) {
      uint dst_index = posOut.z * 4 + j;
      if (dst_index >= max_dst_index) {
        imageStore(uOutput, posOut, outval);
        // out of range
        break;
      }

      uint b1 = int(dst_index / uBlock.ch_info.x);
      uint c1 = dst_index % uBlock.ch_info.x;
      uint h1 = posOut.y;
      uint w1 = posOut.x;

      uint b, c, h, w;
      switch (uBlock.out_ndims[0]) {
        case 0:
          b = b1;
          break;
        case 1:
          c = b1;
          break;
        case 2:
          h = b1;
          break;
        case 3:
          w = b1;
          break;
      }

      switch (uBlock.out_ndims[1]) {
        case 0:
          b = c1;
          break;
        case 1:
          c = c1;
          break;
        case 2:
          h = c1;
          break;
        case 3:
          w = c1;
          break;
      }

      switch (uBlock.out_ndims[2]) {
        case 0:
          b = h1;
          break;
        case 1:
          c = h1;
          break;
        case 2:
          h = h1;
          break;
        case 3:
          w = h1;
          break;
      }

      switch (uBlock.out_ndims[3]) {
        case 0:
          b = w1;
          break;
        case 1:
          c = w1;
          break;
        case 2:
          h = w1;
          break;
        case 3:
          w = w1;
          break;
      }

      uint src_index = b * uBlock.ch_info.y + c;
      ivec3 posIn;
      posIn.x = int(w);
      posIn.y = int(h);
      posIn.z = int(src_index / 4);
      uint i = (src_index % 4);

      vec4 inval = texelFetch(uInput, posIn, 0);
      outval[j] = inval[i];

      if (j == 3) {
        imageStore(uOutput, posOut, outval);
      }
    }
  }
}
