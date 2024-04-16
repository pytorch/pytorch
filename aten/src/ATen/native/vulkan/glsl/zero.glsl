#version 450 core
#define PRECISION $precision
#define FORMAT $format

layout(std430) buffer;

/*
 * Output Image
 */
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;

/*
 * Local Work Group Size
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Returns a tensor filled with zeros
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  imageStore(uOutput, pos, vec4(0, 0, 0, 0));
}
