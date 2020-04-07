#version 450 core
layout(std430) buffer;
layout(std430) uniform;
layout(rgba32f, set=0, binding=0) readonly uniform highp image3D uImage;
layout(set=0, binding=1) writeonly buffer destBuffer{
    float data[];
} uOutBuffer;
layout(set=0, binding = 2) uniform sizeBlock{ int width; int height; } uSizeBlock;
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    int W = uSizeBlock.width;
    int H = uSizeBlock.height;
    int WH = W * H;
    if (pos.x < W && pos.y < H)
    {
        vec4 color = imageLoad(uImage, pos);
        int z = pos.z*4;
        uOutBuffer.data[W*pos.y+pos.x+(z+0)*WH] = color.r;
        uOutBuffer.data[W*pos.y+pos.x+(z+1)*WH] = color.g;
        uOutBuffer.data[W*pos.y+pos.x+(z+2)*WH] = color.b;
        uOutBuffer.data[W*pos.y+pos.x+(z+3)*WH] = color.a;
    }
}
