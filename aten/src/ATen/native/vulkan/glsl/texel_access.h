/*
 * Texel access utility functions
 */

// Broadcasting: compute input texel position from broadcasted output position
ivec3 map_output_pos_to_input_pos(
    ivec3 output_pos,
    ivec4 output_sizes,
    ivec4 input_sizes) {
  ivec3 input_pos;
  // HW: use modulo
  input_pos.xy = output_pos.xy % input_sizes.xy;
  if (output_sizes.w == input_sizes.w && output_sizes.z != input_sizes.z) {
    // C: divide by ceil(C/4) to map to input tensor range
    input_pos.z = output_pos.z / int(ceil(output_sizes.z / 4.0));
  } else {
    // N: use modulo. z-range of input is batch * ceil(channel/4)
    input_pos.z =
        output_pos.z % (input_sizes.w * int(ceil(input_sizes.z / 4.0)));
  }
  return input_pos;
}

// Broadcasting: load texel from an image texture, applying broadcasting
vec4 load_texel(
    ivec3 mapped_pos,
    ivec4 output_sizes,
    ivec4 input_sizes,
    sampler3D uInput) {
  return (output_sizes.z != input_sizes.z)
      ? texelFetch(uInput, mapped_pos, 0).xxxx
      : texelFetch(uInput, mapped_pos, 0);
}
