#version 450 core
#define PRECISION $precision

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0) buffer  PRECISION restrict Buffer {
  float data[];
} uBuffer;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x < 3*3*4) {
    uBuffer.data[pos.x] = uBuffer.data[pos.x]+2;
  }
}
