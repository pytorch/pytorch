#version 450 core
layout(std430) buffer;
layout(std430) uniform;

layout(set=0, binding=0) buffer destBuffer{
    float data[];
} uOutBuffer;
layout(set=0, binding=1) readonly buffer srcBuffer{
    float data[];
} uInBuffer;
layout(set=0, binding=2) uniform constBlock{
    int IW;
    int IH;
    int OW;
    int OH;
    float scaleX;
    float scaleY;
} uConstBlock;

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main()
{
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  int ow = uConstBlock.OW;
  int oh = uConstBlock.OH;
  if(pos.x < ow && pos.y < oh)
  {
    int iw = uConstBlock.IW;
    int ih = uConstBlock.IH;
    float srcX = float(pos.x) * uConstBlock.scaleX;
    int x1 = int(floor(srcX));
    int x11 = clamp(x1, 0, iw - 1);
    float srcY = float(pos.y) * uConstBlock.scaleY;
    int y1 = int(floor(srcY));
    int y11 = clamp(y1, 0, ih - 1);
    float outValue = uInBuffer.data[pos.z*iw*ih + y11*iw + x11];
    uOutBuffer.data[pos.z*ow*oh + pos.y*ow + pos.x] = outValue;
  }
}
