#version 450 core
#define PRECISION $precision

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1) uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION                    sampler3D uKernel;
layout(set = 0, binding = 3) buffer  PRECISION restrict readonly  Bias {
  vec4 data[];
} uBias;
layout(set = 0, binding = 4) uniform PRECISION restrict           Block {
  ivec4 size;
  vec2 clamp;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec3 opos00 = ivec3(2*pos.xy, pos.z);

  if (all(lessThan(opos00, uBlock.size.xyz))) {
    const ivec2 ipos00 = 4*pos.xy;

    vec4 dg[16] = {
      vec4(0,0,0,0),
      vec4(0,0,0,0),
      vec4(0,0,0,0),
      vec4(0,0,0,0),
      vec4(0,0,0,0),
      vec4(0,0,0,0),
      vec4(0,0,0,0),
      vec4(0,0,0,0),
      vec4(0,0,0,0),
      vec4(0,0,0,0),
      vec4(0,0,0,0),
      vec4(0,0,0,0),
      vec4(0,0,0,0),
      vec4(0,0,0,0),
      vec4(0,0,0,0),
      vec4(0,0,0,0)
    };

    for (int y = 0; y < 4; ++y) {
      for (int x = 0; x < 4; ++x) {
        const ivec2 iposxy = ipos00.xy + ivec2(x,y);
        ivec2 wpos = ivec2(4*uBlock.size.w*x, 4*pos.z+y);
        for (int z4 = 0; z4 < uBlock.size.w; ++z4) {
          const vec4 intex = texelFetch(uInput, ivec3(iposxy, z4), 0);
          dg[4*y+x] += vec4(
            dot(intex, texelFetch(uKernel, ivec3(wpos.x  , wpos.y, 0), 0)),
            dot(intex, texelFetch(uKernel, ivec3(wpos.x+1, wpos.y, 0), 0)),
            dot(intex, texelFetch(uKernel, ivec3(wpos.x+2, wpos.y, 0), 0)),
            dot(intex, texelFetch(uKernel, ivec3(wpos.x+3, wpos.y, 0), 0)));

          wpos += ivec2(4, 0);
        }
      }
    }

    const vec4 o00 = dg[0] + dg[4] + dg[8];
    const vec4 o01 = dg[1] + dg[5] + dg[9];
    const vec4 o02 = dg[2] + dg[6] + dg[10];
    const vec4 o03 = dg[3] + dg[7] + dg[11];
    const vec4 o10 = dg[4] - dg[8] - dg[12];
    const vec4 o11 = dg[5] - dg[9] - dg[13];
    const vec4 o12 = dg[6] - dg[10] - dg[14];
    const vec4 o13 = dg[7] - dg[11] - dg[15];

    const vec4 b = uBias.data[pos.z];
    imageStore(uOutput, ivec3(opos00.x, opos00.y, opos00.z), clamp(b + o00 + o01 + o02, uBlock.clamp.x, uBlock.clamp.y));
    if (opos00.x+1 < uBlock.size.x)
      imageStore(uOutput, ivec3(opos00.x+1, opos00.y, opos00.z), clamp(b + o01 - o02 - o03, uBlock.clamp.x, uBlock.clamp.y));
    if (opos00.y+1 < uBlock.size.y)
      imageStore(uOutput, ivec3(opos00.x, opos00.y+1, opos00.z), clamp(b + o10 + o11 + o12, uBlock.clamp.x, uBlock.clamp.y));
    if (opos00.x+1 < uBlock.size.x && opos00.y+1 < uBlock.size.y)
      imageStore(uOutput, ivec3(opos00.x+1, opos00.y+1, opos00.z), clamp(b + o11 - o12 - o13, uBlock.clamp.x, uBlock.clamp.y));
  }
}
