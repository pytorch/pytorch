#version 450 core
layout(std430) buffer;
layout(set = 0, binding = 0) writeonly buffer outputBuffer {
  float data[];
}
uOutput;
layout(set = 0, binding = 1) readonly buffer inputBuffer {
  float data[];
}
uInput;
layout(set = 0, binding = 2) uniform constBlock {
  ivec4 inStrides[2];
  ivec4 outStrides[2];
  ivec4 outDims[2];
  int inOffset;
}
uConst;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  ivec4 outIdx[2];

  int d1 = uConst.outDims[0][3];
  int d3 = uConst.outDims[1][1];
  int d5 = uConst.outDims[1][3];

  int oi0 = pos.z / d1;
  int oi1 = pos.z - d1 * oi0;

  int oi2 = pos.y / d3;
  int oi3 = pos.y - d3 * oi2;

  int oi4 = pos.x / d5;
  int oi5 = pos.x - d5 * oi4;

  ivec4 oIdx0 = ivec4(0, 0, oi0, oi1);
  ivec4 oIdx1 = ivec4(oi2, oi3, oi4, oi5);
  if (all(lessThan(oIdx0, uConst.outDims[0])) &&
      all(lessThan(oIdx1, uConst.outDims[1]))) {
    ivec4 ins0 = uConst.inStrides[0];
    ivec4 ins1 = uConst.inStrides[1];
    int inIdxInt = oIdx0.x * ins0.x + oIdx0.y * ins0.y + oIdx0.z * ins0.z +
        oIdx0.w * ins0.w;
    inIdxInt += oIdx1.x * ins1.x + oIdx1.y * ins1.y + oIdx1.z * ins1.z +
        oIdx1.w * ins1.w;
    ivec4 outs0 = uConst.outStrides[0];
    ivec4 outs1 = uConst.outStrides[1];
    int outIdxInt = oIdx0.x * outs0.x + oIdx0.y * outs0.y + oIdx0.z * outs0.z +
        oIdx0.w * outs0.w;
    outIdxInt += oIdx1.x * outs1.x + oIdx1.y * outs1.y + oIdx1.z * outs1.z +
        oIdx1.w * outs1.w;

    uOutput.data[outIdxInt] = uInput.data[uConst.inOffset + inIdxInt];
  }
}
