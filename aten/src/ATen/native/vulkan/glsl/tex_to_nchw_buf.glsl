layout(FORMAT, binding=0) readonly uniform PRECISION image3D uImage;

layout(binding=1) writeonly buffer destBuffer{
    float data[];
} uOutBuffer;

layout(location = 2) uniform int uWidth;
layout(location = 3) uniform int uHeight;

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    if (pos.x < uWidth && pos.y < uHeight)
    {
        vec4 color = imageLoad(uImage, pos);
        int z = pos.z*4;
        uOutBuffer.data[uWidth*pos.y+pos.x+(z+0)*uWidth*uHeight] = color.r;
        uOutBuffer.data[uWidth*pos.y+pos.x+(z+1)*uWidth*uHeight] = color.g;
        uOutBuffer.data[uWidth*pos.y+pos.x+(z+2)*uWidth*uHeight] = color.b;
        uOutBuffer.data[uWidth*pos.y+pos.x+(z+3)*uWidth*uHeight] = color.a;
    }
}
