#version 450 core
#define PRECISION $precision
#define FORMAT $format

layout(std430) buffer;

/*
 * Output Image
 */
layout(set = 0, binding = 0, FORMAT) uniform PRECISION image3D uOutput;

/*
 * Input Textures
 */
layout(set = 0, binding = 1) uniform PRECISION isampler3D uInput;

/*
 * Params Buffer
 */
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // output texture size (x=width,y=height,z=depth,w=unused)
  ivec4 out_extents;
  // mask texture size (x=width,y=height,z=depth,w=unused)
  ivec4 mask_extents;
  // output extent sizes (x=batch,y=channel,z=height,w=width)
  uvec4 out_size_info;
  // mask extent sizes (x=batch,y=channel,z=height,w=width)
  uvec4 mask_size_info;
  // x: size of output channel dim up-aligned to 4
  // y: size of mask channel dim up-aligned to 4
  uvec2 aligned_channel_info;
  // value to replace
  float value;
}
uBlock;

/*
 * Local Work Group
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos_mask = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos_mask, uBlock.out_extents.xyz))) {
    return;
  }

  ivec4 inval = texelFetch(uInput, pos_mask, 0);

  bool mask_has_true = false;
  for (uint i = 0; i < 4; ++i) {
    if ((pos_mask.z * 4 + i) % uBlock.aligned_channel_info.y >=
        uBlock.mask_size_info.y) {
      break;
    }
    if (inval[i] == 1) {
      mask_has_true = true;
    }
  }

  // we traverse the elements of mask. If an element is True, we find the
  // corresponding positions in the output according to broadcasting and fill
  // the elements of output with value. Due to the padding at channel dimension,
  // we have different ways to fill the value depending on whether the channel
  // dimension is broadcasted or not
  if (mask_has_true) {
    bool mask_channel_is_broadcast =
        uBlock.mask_size_info.y < uBlock.out_size_info.y;
    uint tex_cnt_in_output_batch = uBlock.aligned_channel_info.x / 4;

    for (uint batch = 0;
         batch < uBlock.out_size_info.x / uBlock.mask_size_info.x;
         ++batch) {
      for (uint height = 0;
           height < uBlock.out_size_info.z / uBlock.mask_size_info.z;
           ++height) {
        for (uint width = 0;
             width < uBlock.out_size_info.w / uBlock.mask_size_info.w;
             ++width) {
          if (mask_channel_is_broadcast) {
            for (int tex_idx = 0; tex_idx < tex_cnt_in_output_batch;
                 ++tex_idx) {
              ivec3 write_pos = ivec3(
                  pos_mask.x + width,
                  pos_mask.y + height,
                  tex_cnt_in_output_batch * (batch + pos_mask.z) + tex_idx);
              vec4 out_tex = imageLoad(uOutput, write_pos);
              for (int i = 0; i < 4; ++i) {
                if (tex_idx * 4 + i >= uBlock.out_size_info.y) {
                  break;
                }
                out_tex[i] = uBlock.value;
              }
              imageStore(uOutput, write_pos, out_tex);
            }
          } else {
            ivec3 write_pos = ivec3(
                pos_mask.x + width,
                pos_mask.y + height,
                pos_mask.z + tex_cnt_in_output_batch * batch);
            vec4 out_tex = imageLoad(uOutput, write_pos);
            out_tex = vec4(equal(inval, ivec4(1))) * uBlock.value + vec4(notEqual(inval, ivec4(1))) * out_tex;
            imageStore(uOutput, write_pos, out_tex);
          }
        }
      }
    }
  }
}
