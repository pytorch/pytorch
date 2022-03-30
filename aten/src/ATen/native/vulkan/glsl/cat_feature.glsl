#version 450 core
#define PRECISION $precision
#define FORMAT    $format

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION           image3D uOutput;
layout(set = 0, binding = 1)         uniform PRECISION           sampler3D uInput;
layout(set = 0, binding = 2)         uniform PRECISION restrict  Block {
  ivec4 size;            // output texture size (x=width,y=height,z=depth,w=unused)
  ivec4 isize;           // input texture size (x=width,y=height,z=depth,w=unused)
  uint batch_size;       // input tensor's batch size
  uint ch_size;          // input tensor's channel size
  uint ch_interval;      // channel interval (total # of channels for all tensors)
  uint ch_size_allprior; // # of channels for tensor 0 to i-1 at ith tensor
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 posIn = ivec3(gl_GlobalInvocationID);
  const uint max_src_index = uBlock.ch_size * uBlock.batch_size;

  if (all(lessThan(posIn, uBlock.isize.xyz))) {
    ivec3 posOut = posIn; // x and y don't change. only z and index matter
    const vec4 inval = texelFetch(uInput, posIn, 0);

    for (uint i = 0; i < 4; ++i)
    {
      uint src_index = posIn.z * 4 + i;
      if (src_index >= max_src_index) {
        // out of range
        break;
      }

      uint dst_index = uint(src_index / uBlock.ch_size) * uBlock.ch_interval + (src_index % uBlock.ch_size) + uBlock.ch_size_allprior;
      posOut.z = int(dst_index / 4);
      uint j = (dst_index % 4);

      vec4 outval = imageLoad(uOutput, posOut);
      outval[j] = inval[i];
      imageStore(uOutput, posOut, outval);
    }
  }
}
