#version 450 core
#define PRECISION $precision
#define FORMAT $format

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

/*
 * Output Image
 */
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;

/*
 * Input Buffer
 */
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;

/*
 * Params Buffer
 */
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 oextents;
  ivec2 iextents;
  vec2 scale;
}
uBlock;

/*
 * Local Work Group Size
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Upsamples uInput to the uOutput with scale according to uBlock params,
 * using the equation for bilinear upsampling/interpolation
 * along the height and width plane.
 * align_false ~ align_corners=False, it means that each of the 4 output
 * corner texels are treated in interpolation as if they were half texel
 * offset outwards from the 4 input corner texels, if the two textures
 * were overlaid the output texture would be "bigger".
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThan(pos, uBlock.oextents.xyz))) {
    return;
  }
  // the border interpolated continuous coordinates from align=false
  // are floored and ceiled to avoid alpha becoming negative
  vec2 pos_interp = clamp(
      ((pos.xy + 0.5) * uBlock.scale) - 0.5, vec2(0, 0), uBlock.iextents.xy);

  // 4 input texels used for bilinear interpolation, naming by PyTorch
  // Tensor coordinate space where the "top" is x = 0 and "left" is y = 0,
  // Vulkan reversed
  ivec3 in_pos_topleft = ivec3(floor(pos_interp.x), floor(pos_interp.y), pos.z);
  ivec3 in_pos_bottomleft =
      ivec3(floor(pos_interp.x), ceil(pos_interp.y), pos.z);
  ivec3 in_pos_topright = ivec3(ceil(pos_interp.x), floor(pos_interp.y), pos.z);
  ivec3 in_pos_bottomright =
      ivec3(ceil(pos_interp.x), ceil(pos_interp.y), pos.z);

  vec2 alpha = pos_interp - in_pos_topleft.xy;

  const vec4 top_val_interp =
      (texelFetch(uInput, in_pos_topleft, 0) * (1 - alpha.x)) +
      (texelFetch(uInput, in_pos_topright, 0) * alpha.x);
  const vec4 bot_val_interp =
      (texelFetch(uInput, in_pos_bottomleft, 0) * (1 - alpha.x)) +
      (texelFetch(uInput, in_pos_bottomright, 0) * alpha.x);

  imageStore(
      uOutput,
      pos,
      (top_val_interp * (1 - alpha.y)) + (bot_val_interp * alpha.y));
}
