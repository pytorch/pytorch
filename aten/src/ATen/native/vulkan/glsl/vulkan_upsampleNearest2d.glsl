#version 450 core
layout(std430) buffer;
layout(std430) uniform;
layout(set=0, rgba16f, binding=0) writeonly mediump uniform image3D uOutput;
layout(set=0, binding=1) uniform mediump sampler3D uInput;
layout(set=0, binding=2) uniform constBlock{
    int IW;
    int IH;
    int OW;
    int OH;
    float scaleX;
    float scaleY;
} uConstBlock;

layout (local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  int ow = uConstBlock.OW;
  int oh = uConstBlock.OH;
  if(pos.x < ow && pos.y < oh) {
    int iw = uConstBlock.IW;
    int ih = uConstBlock.IH;
    float srcX = float(pos.x) * uConstBlock.scaleX;
    int x1 = int(floor(srcX));
    int x11 = clamp(x1, 0, iw - 1);
    float srcY = float(pos.y) * uConstBlock.scaleY;
    int y1 = int(floor(srcY));
    int y11 = clamp(y1, 0, ih - 1);
    vec4 outValue = texelFetch(uInput, ivec3(x11, y11, pos.z), 0);
    imageStore(uOutput, pos, outValue);
  }
}
