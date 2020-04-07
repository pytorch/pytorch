layout(std430) buffer;
layout(FORMAT, binding=0) writeonly uniform PRECISION image3D uOutput;
layout(binding=2) readonly buffer kernel{
    vec4 data[];
} uKernel;

layout(location = 3) uniform int uFxFy;
layout(location = 4) uniform int uIc_4;

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID) * ivec3(4, 1, 1);
    int kernelPos = 0
    + pos.x * uFxFy
    + 4*pos.y * uIc_4 * uFxFy
    + 4*pos.z
    ;
    vec4 color0 = uKernel.data[kernelPos+0];
    vec4 color1 = uKernel.data[kernelPos+1];
    vec4 color2 = uKernel.data[kernelPos+2];
    vec4 color3 = uKernel.data[kernelPos+3];
    
    imageStore(uOutput, ivec3(pos.x+0, pos.y, pos.z), color0);
    imageStore(uOutput, ivec3(pos.x+1, pos.y, pos.z), color1);
    imageStore(uOutput, ivec3(pos.x+2, pos.y, pos.z), color2);
    imageStore(uOutput, ivec3(pos.x+3, pos.y, pos.z), color3);
}
