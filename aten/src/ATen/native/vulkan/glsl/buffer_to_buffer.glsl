#version 450 core

#define PRECISION $precision
#define FORMAT $format

#include <indexing.h>

layout(std430) buffer;

/*
 * Output Buffer
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
  float data[];
}
uOutput;

/*
 * Input Buffer
 */
layout(set = 0, binding = 1) buffer PRECISION restrict readonly InBuffer {
  float data[];
}
uInput;

/*
 * Params Buffer
 */
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  uvec4 out_sizes;
  uvec4 out_strides;
  uvec4 in_sizes;
  uvec4 in_strides;
  // x is the length of uOutput, y is the length of uInput
  uvec2 buf_lengths;
}
uBlock;

/*
 * Local Work Group Size
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Copies data from the tensor at uInput to the tensor at uOutput based on 4D
 * coordinate. Each element at (x,y,c,n) in uInput will be copied to uOutput at
 * (x,y,c,n). If (x,y,c,n) is outside the bounds of uInput then 0 will be
 * written.
 *
 * Each shader invocation is responsible for one element of the output buffer.
 */
void main() {
  const uint write_idx = ivec3(gl_GlobalInvocationID).x;

  if (write_idx >= uBlock.buf_lengths.x) {
    return;
  }

  uvec4 write_coord =
      idx_to_coord(write_idx, uBlock.out_strides, uBlock.out_sizes);

  float outval = 0u;
  if (all(lessThan(write_coord, uBlock.in_sizes))) {
    uint read_idx = coord_to_idx(write_coord, uBlock.in_strides);
    outval = uInput.data[read_idx];
  }

  uOutput.data[write_idx] = outval;
}
