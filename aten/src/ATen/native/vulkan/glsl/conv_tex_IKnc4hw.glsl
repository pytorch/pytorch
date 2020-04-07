layout(std430) buffer;
layout(FORMAT, binding=0) writeonly uniform PRECISION image3D uOutput;
layout(location=1) uniform PRECISION sampler3D uInput;
layout(location=2) uniform PRECISION sampler3D uKernel;

layout(binding=3) readonly buffer bias{
    vec4 data[];
} uBias;

layout(location=4) uniform ivec2 uPad;
layout(location=5) uniform ivec2 uKernelSize;
layout(location=6) uniform ivec2 uStride;
layout(location=7) uniform ivec2 uDilate;
layout(location=8) uniform int uUnroll;

layout(location=10) uniform ivec3 uOutputSize;
layout(location=11) uniform ivec3 uInputSize;

#define UP_DIV(x, y) (((x)+(y)-1)/(y))

layout (local_size_x = WORKGROUP_X, local_size_y = WORKGROUP_Y, local_size_z = WORKGROUP_Z) in;

void main()
{
    if (all(lessThan(ivec3(gl_GlobalInvocationID), uOutputSize)))
    {
        ivec3 pos = ivec3(gl_GlobalInvocationID)*ivec3(uUnroll, 1, 1);
        int kernelX = uKernelSize.x;
        int kernelY = uKernelSize.y;
        ivec3 inputSize = uInputSize;
        ivec2 s0 = pos.xy*uStride-uPad;
        int fx, fy, fz;
        ivec2 sfxy = max(ivec2(0), (UP_DIV(-s0, uDilate)));
        ivec2 efxy = min(uKernelSize, UP_DIV(inputSize.xy-s0, uDilate));
        vec4 color = uBias.data[pos.z];
        vec4 color2 = color;
        vec4 color3 = color;
        vec4 color4 = color;
        int kY = pos.z;
        for (fy=sfxy.y; fy<efxy.y; ++fy)
        {
            int sy = fy*uDilate.y + s0.y;
            for (fx=0; fx<kernelX; ++fx)
            {
                int kZ = fx + fy*kernelX;
                int sx1 = fx*uDilate.x + s0.x;
                int sx2 = sx1 + uStride.x;
                int sx3 = sx1 + uStride.x * 2;
                int sx4 = sx1 + uStride.x * 3;
                float m1 = sx1 >= 0 && sx1 < inputSize.x ? 1.0 : 0.0;
                float m2 = sx2 >= 0 && sx2 < inputSize.x ? 1.0 : 0.0;
                float m3 = sx3 >= 0 && sx3 < inputSize.x ? 1.0 : 0.0;
                float m4 = sx4 >= 0 && sx4 < inputSize.x ? 1.0 : 0.0;
                fz = 0;
                for (; fz<inputSize.z; ++fz)
                {
                    int kX = 4*fz;
                    vec4 k0 = texelFetch(uKernel, ivec3(kX+0, kY, kZ), 0);
                    vec4 k1 = texelFetch(uKernel, ivec3(kX+1, kY, kZ), 0);
                    vec4 k2 = texelFetch(uKernel, ivec3(kX+2, kY, kZ), 0);
                    vec4 k3 = texelFetch(uKernel, ivec3(kX+3, kY, kZ), 0);

                    mat4 k = mat4(k0, k1, k2, k3);

                    color  += k*texelFetch(uInput, ivec3(sx1, sy, fz), 0) * m1;
                    color2 += k*texelFetch(uInput, ivec3(sx2, sy, fz), 0) * m2;
                    color3 += k*texelFetch(uInput, ivec3(sx3, sy, fz), 0) * m3;
                    color4 += k*texelFetch(uInput, ivec3(sx4, sy, fz), 0) * m4;
                }
            }
        }
        imageStore(uOutput, ivec3(pos.x+0, pos.y, pos.z), color);
        imageStore(uOutput, ivec3(pos.x+1, pos.y, pos.z), color2);
        imageStore(uOutput, ivec3(pos.x+2, pos.y, pos.z), color3);
        imageStore(uOutput, ivec3(pos.x+3, pos.y, pos.z), color4);
    }

}
