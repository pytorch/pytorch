#version 450 core
#define PRECISION $precision
#define FORMAT    $format

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2)         uniform PRECISION                    sampler3D uGamma;
layout(set = 0, binding = 3)         uniform PRECISION                    sampler3D uBeta;
layout(set = 0, binding = 4)         uniform PRECISION restrict           Block {
  ivec3 isize;
  int volume;
  int offset;
  float eps;
} uBlock;

shared float sh_mem[64];
shared float mean;
shared float rstd;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// This is the simple two-pass algorithm to compute variance.
// This implementation is not efficient when calculating mean and
// variance since every work group will compute the mean and variance
// for the entire tensor.

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec3 tid = ivec3(gl_LocalInvocationID);
  const ivec3 group_size = ivec3(gl_WorkGroupSize);

  // Start computing mean.
  // Divide work among the 64 invocations in the work group
  // and compute partial sums of texels that are "fully filled"
  vec4 sum4d = vec4(0);
  for (int z = tid.z; z < uBlock.isize.z - 1; z+=group_size.z) {
    for (int y = tid.y; y < uBlock.isize.y; y+=group_size.y) {
      for (int x = tid.x; x < uBlock.isize.x; x+=group_size.x) {
        sum4d += texelFetch(uInput, ivec3(x, y, z), 0);
      }
    }
  }
  float sum = sum4d.x + sum4d.y + sum4d.w + sum4d.z;

  // Still computing the mean, processing the last texel across the channel-batch dimension
  if ((uBlock.isize.z - 1) % group_size.z == tid.z) {
    for (int y = tid.y; y < uBlock.isize.y; y+=group_size.y) {
      for (int x = tid.x; x < uBlock.isize.x; x+=group_size.x) {
        const vec4 last_texel = texelFetch(uInput, ivec3(x, y, uBlock.isize.z - 1), 0);
        sum += (
          last_texel.x +
          (uBlock.offset >= 1 ? last_texel.y : 0) +
          (uBlock.offset >= 2 ? last_texel.z : 0) +
          (uBlock.offset == 3 ? last_texel.w : 0)
        );
      }
    }
  }

  // Shared memory (among threads in a work group) that holds partial sums
  sh_mem[gl_LocalInvocationIndex] = sum;

  memoryBarrierShared();
  barrier();

  // Only instance (0, 0, 0) will compute the sum of the 64 partial sums,
  // and then compute the mean, dividing the total by the tensor's volume
  if (tid == ivec3(0)) {
    float total = 0;
    for (int z = 0; z < group_size.z; ++z) {
      for (int y = 0; y < group_size.y; ++y) {
        for (int x = 0; x < group_size.x; ++x) {
          total += sh_mem[z * group_size.y * group_size.x + y * group_size.x + x];
        }
      }
    }
    mean = total / uBlock.volume;
  }

  memoryBarrierShared();
  barrier();

  // Start computing variance (using the previously computed mean)
  // Divide work among the 64 invocations in the work group
  // and compute partial sums of texels that are "fully filled"
  vec4 sqsum4d = vec4(0);
  for (int z = tid.z; z < uBlock.isize.z - 1; z+=group_size.z) {
    for (int y = tid.y; y < uBlock.isize.y; y+=group_size.y) {
      for (int x = tid.x; x < uBlock.isize.x; x+=group_size.x) {
        const vec4 val = texelFetch(uInput, ivec3(x, y, z), 0);
        sqsum4d += (val - mean) * (val - mean);
      }
    }
  }
  float sqsum = sqsum4d.x + sqsum4d.y + sqsum4d.w + sqsum4d.z;

  // Still computing the variance, processing the last texel across the channel-batch dimension
  if ((uBlock.isize.z - 1) % group_size.z == tid.z) {
    for (int y = tid.y; y < uBlock.isize.y; y+=group_size.y) {
      for (int x = tid.x; x < uBlock.isize.x; x+=group_size.x) {
        const vec4 last_texel = texelFetch(uInput, ivec3(x, y, uBlock.isize.z - 1), 0);
        sqsum += (
          (last_texel.x - mean) * (last_texel.x - mean) +
          (uBlock.offset >= 1 ? (last_texel.y - mean) * (last_texel.y - mean) : 0) +
          (uBlock.offset >= 2 ? (last_texel.z - mean) * (last_texel.z - mean) : 0) +
          (uBlock.offset == 3 ? (last_texel.w - mean) * (last_texel.w - mean) : 0)
        );
      }
    }
  }

  // Reuse shared memory to hold partial squared sums
  sh_mem[gl_LocalInvocationIndex] = sqsum;

  memoryBarrierShared();
  barrier();

  // Only instance (0, 0, 0) will compute the sum of the 64 partial sums,
  // and then compute the squared root of the biased variance, with eps added
  // to the denominator for numerical stabilty.
  if (tid == ivec3(0)) {
    float total2 = 0;
    for (int z = 0; z < group_size.z; ++z) {
      for (int y = 0; y < group_size.y; ++y) {
        for (int x = 0; x < group_size.x; ++x) {
          total2 += sh_mem[z * group_size.y * group_size.x + y * group_size.x + x];
        }
      }
    }
    rstd = sqrt(total2 / uBlock.volume + uBlock.eps);
  }

  memoryBarrierShared();
  barrier();

  // Compute layernorm using previously computed mean and rstd
  if (all(lessThan(pos, uBlock.isize.xyz))) {
    imageStore(
        uOutput,
        pos,
        (texelFetch(uInput, pos, 0)
            - mean)
            / rstd
            * texelFetch(uGamma, pos, 0)
            + texelFetch(uBeta, pos, 0));
  }
}
