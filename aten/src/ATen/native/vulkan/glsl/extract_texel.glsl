#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutputX;
layout(set = 0, binding = 1, FORMAT) uniform PRECISION restrict writeonly image3D uOutputY;
layout(set = 0, binding = 2, FORMAT) uniform PRECISION restrict writeonly image3D uOutputZ;
layout(set = 0, binding = 3, FORMAT) uniform PRECISION restrict writeonly image3D uOutputW;

layout(set = 0, binding = 4) uniform PRECISION sampler3D uInput;

layout(set = 0, binding = 5) uniform PRECISION restrict Block {
  ivec3 pos;
} uBlock;

void main() {
    vec4 texel = texelFetch(uInput, uBlock.pos, 0);

    ivec3 out_pos = ivec3(0, 0, 0);

    imageStore(uOutputX, out_pos, vec4(texel.x, 0.0, 0.0, 0.0));
    imageStore(uOutputY, out_pos, vec4(texel.y, 0.0, 0.0, 0.0));
    imageStore(uOutputZ, out_pos, vec4(texel.z, 0.0, 0.0, 0.0));
    imageStore(uOutputW, out_pos, vec4(texel.w, 0.0, 0.0, 0.0));
}
