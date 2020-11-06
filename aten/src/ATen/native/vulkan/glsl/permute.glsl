#version 450 core

layout(std430) buffer;
layout(std430) uniform;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0)         writeonly buffer outputBuffer {
  float data[];
}
uOutput;
layout(set = 0, binding = 1)         readonly  buffer inputBuffer {
  float data[];
}
uInput;
layout(set = 0, binding = 2) uniform                  Block {
  ivec4 input_strides[2];
  ivec4 output_strides[2];
  ivec4 output_sizes[2];
  int input_offset;
} uBlock;

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  ivec4 outIdx[2];

  const int d1 = uBlock.output_sizes[0][3];
  const int d3 = uBlock.output_sizes[1][1];
  const int d5 = uBlock.output_sizes[1][3];

  const int oi0 = pos.z / d1;
  const int oi1 = pos.z - d1 * oi0;

  const int oi2 = pos.y / d3;
  const int oi3 = pos.y - d3 * oi2;

  const int oi4 = pos.x / d5;
  const int oi5 = pos.x - d5 * oi4;

  ivec4 oIdx0 = ivec4(0, 0, oi0, oi1);
  ivec4 oIdx1 = ivec4(oi2, oi3, oi4, oi5);
  if (all(lessThan(oIdx0, uBlock.output_sizes[0])) &&
      all(lessThan(oIdx1, uBlock.output_sizes[1]))) {
    ivec4 ins0 = uBlock.input_strides[0];
    ivec4 ins1 = uBlock.input_strides[1];
    int inIdxInt = oIdx0.x * ins0.x + oIdx0.y * ins0.y + oIdx0.z * ins0.z + oIdx0.w * ins0.w;
    inIdxInt += oIdx1.x * ins1.x + oIdx1.y * ins1.y + oIdx1.z * ins1.z + oIdx1.w * ins1.w;
    ivec4 outs0 = uBlock.output_strides[0];
    ivec4 outs1 = uBlock.output_strides[1];
    int outIdxInt = oIdx0.x * outs0.x + oIdx0.y * outs0.y + oIdx0.z * outs0.z + oIdx0.w * outs0.w;
    outIdxInt += oIdx1.x * outs1.x + oIdx1.y * outs1.y + oIdx1.z * outs1.z + oIdx1.w * outs1.w;

    uOutput.data[outIdxInt] = uInput.data[uBlock.input_offset + inIdxInt];
  }
}
