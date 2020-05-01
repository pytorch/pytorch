#version 450 core
layout(std430) buffer;
layout(std430) uniform;

layout(set=0, binding=0) buffer outputBuffer{
    float data[];
} uOutBuffer;
layout(set=0, binding=1) readonly buffer input1Buffer{
    float data[];
} uIn0Buffer;
layout(set=0, binding=2) readonly buffer input2Buffer{
    float data[];
} uIn1Buffer;
layout(set=0, binding=3) uniform constBlock{
    int W;
    int H;
    int C;
    float alpha;
} uConstBlock;
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    int W = uConstBlock.W;
    int H = uConstBlock.H;
    int C = uConstBlock.C;
    if(pos.x < W && pos.y < H && pos.z < C)
    {
      int bufferIdx = W*H*pos.z + W*pos.y + pos.x;
      uOutBuffer.data[bufferIdx] = uIn0Buffer.data[bufferIdx] + uConstBlock.alpha*uIn1Buffer.data[bufferIdx];
    }
}
