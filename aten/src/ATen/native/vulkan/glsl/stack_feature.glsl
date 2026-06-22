#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION           image3D uOutput;
layout(set = 0, binding = 1)         uniform PRECISION           sampler3D uInput0;
layout(set = 0, binding = 2)         uniform PRECISION           sampler3D uInput1;
layout(set = 0, binding = 3)         uniform PRECISION           sampler3D uInput2;
layout(set = 0, binding = 4)         uniform PRECISION           sampler3D uInput3;
layout(set = 0, binding = 5)         uniform PRECISION restrict  Block {
  ivec3 size;
  int z;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 posIn = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(posIn, uBlock.size.xyz))) {
    imageStore(
        uOutput,
        ivec3(posIn.x, posIn.y, uBlock.z),
        vec4(
            texelFetch(uInput0, posIn, 0).x,
            texelFetch(uInput1, posIn, 0).x,
            texelFetch(uInput2, posIn, 0).x,
            texelFetch(uInput3, posIn, 0).x
        ));
  }
}
