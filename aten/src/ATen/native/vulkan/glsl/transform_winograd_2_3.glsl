#version 450 core
#define PRECISION $precision

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1) uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict           Block {
  ivec4 size;
  ivec2 limits;
  ivec2 padding;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec3 opos00 = pos * ivec3(4,4,1);

  const ivec2 ipos00 = pos.xy * 2 - uBlock.padding;

  if (all(lessThan(opos00 + ivec3(3,3,0), uBlock.size.xyz))) {
    const vec4 maskx = vec4(greaterThan(vec4(ipos00.x, ipos00.x+1, ipos00.x+2, ipos00.x+3), vec4(-1))) *
                       vec4(lessThan(vec4(ipos00.x, ipos00.x+1, ipos00.x+2, ipos00.x+3), vec4(uBlock.limits.x)));
    const vec4 masky = vec4(greaterThan(vec4(ipos00.y, ipos00.y+1, ipos00.y+2, ipos00.y+3), vec4(-1))) *
                       vec4(lessThan(vec4(ipos00.y, ipos00.y+1, ipos00.y+2, ipos00.y+3), vec4(uBlock.limits.y)));

    const vec4 i00 = texelFetch(uInput, ivec3(ipos00.x  , ipos00.y  , pos.z), 0) * (maskx.x*masky.x);
    const vec4 i01 = texelFetch(uInput, ivec3(ipos00.x+1, ipos00.y  , pos.z), 0) * (maskx.y*masky.x);
    const vec4 i02 = texelFetch(uInput, ivec3(ipos00.x+2, ipos00.y  , pos.z), 0) * (maskx.z*masky.x);
    const vec4 i03 = texelFetch(uInput, ivec3(ipos00.x+3, ipos00.y  , pos.z), 0) * (maskx.w*masky.x);
    const vec4 i10 = texelFetch(uInput, ivec3(ipos00.x  , ipos00.y+1, pos.z), 0) * (maskx.x*masky.y);
    const vec4 i11 = texelFetch(uInput, ivec3(ipos00.x+1, ipos00.y+1, pos.z), 0) * (maskx.y*masky.y);
    const vec4 i12 = texelFetch(uInput, ivec3(ipos00.x+2, ipos00.y+1, pos.z), 0) * (maskx.z*masky.y);
    const vec4 i13 = texelFetch(uInput, ivec3(ipos00.x+3, ipos00.y+1, pos.z), 0) * (maskx.w*masky.y);
    const vec4 i20 = texelFetch(uInput, ivec3(ipos00.x  , ipos00.y+2, pos.z), 0) * (maskx.x*masky.z);
    const vec4 i21 = texelFetch(uInput, ivec3(ipos00.x+1, ipos00.y+2, pos.z), 0) * (maskx.y*masky.z);
    const vec4 i22 = texelFetch(uInput, ivec3(ipos00.x+2, ipos00.y+2, pos.z), 0) * (maskx.z*masky.z);
    const vec4 i23 = texelFetch(uInput, ivec3(ipos00.x+3, ipos00.y+2, pos.z), 0) * (maskx.w*masky.z);
    const vec4 i30 = texelFetch(uInput, ivec3(ipos00.x  , ipos00.y+3, pos.z), 0) * (maskx.x*masky.w);
    const vec4 i31 = texelFetch(uInput, ivec3(ipos00.x+1, ipos00.y+3, pos.z), 0) * (maskx.y*masky.w);
    const vec4 i32 = texelFetch(uInput, ivec3(ipos00.x+2, ipos00.y+3, pos.z), 0) * (maskx.z*masky.w);
    const vec4 i33 = texelFetch(uInput, ivec3(ipos00.x+3, ipos00.y+3, pos.z), 0) * (maskx.w*masky.w);

    const vec4 d00 = i00 - i20;
    const vec4 d01 = i01 - i21;
    const vec4 d02 = i02 - i22;
    const vec4 d03 = i03 - i23;
    const vec4 d10 = i10 + i20;
    const vec4 d11 = i11 + i21;
    const vec4 d12 = i12 + i22;
    const vec4 d13 = i13 + i23;
    const vec4 d20 = i20 - i10;
    const vec4 d21 = i21 - i11;
    const vec4 d22 = i22 - i12;
    const vec4 d23 = i23 - i13;
    const vec4 d30 = i10 - i30;
    const vec4 d31 = i11 - i31;
    const vec4 d32 = i12 - i32;
    const vec4 d33 = i13 - i33;

    imageStore(uOutput, ivec3(opos00.x  , opos00.y  , opos00.z), d00 - d02);
    imageStore(uOutput, ivec3(opos00.x+1, opos00.y  , opos00.z), d01 + d02);
    imageStore(uOutput, ivec3(opos00.x+2, opos00.y  , opos00.z), d02 - d01);
    imageStore(uOutput, ivec3(opos00.x+3, opos00.y  , opos00.z), d01 - d03);
    imageStore(uOutput, ivec3(opos00.x  , opos00.y+1, opos00.z), d10 - d12);
    imageStore(uOutput, ivec3(opos00.x+1, opos00.y+1, opos00.z), d11 + d12);
    imageStore(uOutput, ivec3(opos00.x+2, opos00.y+1, opos00.z), d12 - d11);
    imageStore(uOutput, ivec3(opos00.x+3, opos00.y+1, opos00.z), d11 - d13);
    imageStore(uOutput, ivec3(opos00.x  , opos00.y+2, opos00.z), d20 - d22);
    imageStore(uOutput, ivec3(opos00.x+1, opos00.y+2, opos00.z), d21 + d22);
    imageStore(uOutput, ivec3(opos00.x+2, opos00.y+2, opos00.z), d22 - d21);
    imageStore(uOutput, ivec3(opos00.x+3, opos00.y+2, opos00.z), d21 - d23);
    imageStore(uOutput, ivec3(opos00.x  , opos00.y+3, opos00.z), d30 - d32);
    imageStore(uOutput, ivec3(opos00.x+1, opos00.y+3, opos00.z), d31 + d32);
    imageStore(uOutput, ivec3(opos00.x+2, opos00.y+3, opos00.z), d32 - d31);
    imageStore(uOutput, ivec3(opos00.x+3, opos00.y+3, opos00.z), d31 - d33);
  }
}
