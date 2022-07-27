#version 450 core
#define PRECISION $precision
#define FORMAT    $format

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2)         uniform PRECISION                    sampler3D uGamma;
layout(set = 0, binding = 3)         uniform PRECISION                    sampler3D uBeta;
layout(set = 0, binding = 4)         uniform PRECISION                    sampler3D uMean;
layout(set = 0, binding = 5)         uniform PRECISION                    sampler3D uVar;
layout(set = 0, binding = 6)         uniform PRECISION restrict           Block {
  ivec3 isize;
  int channels_ext;
  float eps;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.isize.xyz))) {
    const ivec3 chn = ivec3(0, 0, pos.z % uBlock.channels_ext);
    imageStore(
        uOutput,
        pos,
        (texelFetch(uInput, pos, 0)
            - texelFetch(uMean, chn, 0))
            / sqrt(texelFetch(uVar, chn, 0) + uBlock.eps)
            * texelFetch(uGamma, chn, 0)
            + texelFetch(uBeta, chn, 0));
  }
}
