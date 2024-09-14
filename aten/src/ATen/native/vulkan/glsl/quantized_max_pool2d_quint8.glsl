#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba8ui) uniform PRECISION restrict writeonly uimage3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION                    isampler3D uInput; // Quantized input
layout(set = 0, binding = 2)         uniform PRECISION restrict           Block {
  ivec4 size;
  ivec4 kernel;
  ivec2 stride;
  ivec2 padding;
  ivec2 dilate;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#define FLT_MIN -3.402823466e+38

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    const ivec2 ipos = pos.xy * uBlock.stride - uBlock.padding;

    const ivec2 start = ipos;
    const ivec2 end = ipos + uBlock.kernel.xy * uBlock.dilate.xy;

    vec4 outtex = vec4(FLT_MIN);

    for (int y = start.y; y < end.y; y += uBlock.dilate.y) {
      for (int x = start.x; x < end.x; x += uBlock.dilate.x) {
        if ((x >= 0 && x < uBlock.kernel.z) && (y >= 0 && y < uBlock.kernel.w)) {
          vec4 outtexy = texelFetch(uInput, ivec3(x, y, pos.z), 0);
          outtex = max(outtexy, outtex);
        }
      }
    }

    uvec4 store = uvec4(outtex);
    imageStore(uOutput, pos, store);
  }
}
