#version 450 core
#define PRECISION $precision
#define FORMAT $format

layout(std430) buffer;

/*
 * Output Image
 */
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;

/*
 * Input Sampler
 */
layout(set = 0, binding = 1) uniform PRECISION sampler3D uImage;

/*
 * Params Buffer
 */
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // info.x: dimension to insert at
  // info.y: channels (for 3d->4d unsqueeze)
  ivec2 info;
}
uBlock;

/*
 * Local Work Group Size
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Returns a new tensor with dimension of size one inserted at the specified
 * position (dim)
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const int dim = uBlock.info.x;
  const int channels = uBlock.info.y;
  vec4 out_texel = vec4(0, 0, 0, 0);
  if (dim == 0) {
    imageStore(uOutput, pos, texelFetch(uImage, pos, 0));
  } else if (dim == 1) {
    int src_x = pos.x;
    int src_y = pos.y;
    for (int i = 0; i < 4; i++) {
      int src_z = pos.z / (channels * 4);
      int p = (pos.z / channels) % 4;
      const vec4 v = texelFetch(uImage, ivec3(src_x, src_y, src_z), 0);
      out_texel[i] = v[p];
    }
    imageStore(uOutput, pos, out_texel);
  } else if (dim == 2) {
    int src_x = pos.x;
    int src_z = pos.z / (channels * 4);
    for (int i = 0; i < 4; i++) {
      int src_y = i + (pos.z % channels) * 4;
      int p = (pos.z / channels) % 4;
      const vec4 v = texelFetch(uImage, ivec3(src_x, src_y, src_z), 0);
      out_texel[i] = v[p];
    }
    imageStore(uOutput, pos, out_texel);
  } else if (dim == 3) {
    int src_x = pos.y;
    int src_z = pos.z / (channels * 4);
    for (int i = 0; i < 4; i++) {
      int src_y = i + (pos.z % channels) * 4;
      int p = (pos.z / channels) % 4;
      const vec4 v = texelFetch(uImage, ivec3(src_x, src_y, src_z), 0);
      out_texel[i] = v[p];
    }
    imageStore(uOutput, pos, out_texel);
  }
}
