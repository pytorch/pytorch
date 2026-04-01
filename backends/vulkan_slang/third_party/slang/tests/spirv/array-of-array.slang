//TEST:SIMPLE(filecheck=CHECK):-target spirv -entry main -stage compute -emit-spirv-directly -skip-spirv-validation

Texture2D texArray[3][4][5];
RWStructuredBuffer<float4> outputBuffer;

// CHECK: OpEntryPoint

[shader("compute")]
void main(uint3 tid : SV_DispatchThreadID)
{
    outputBuffer[0] = texArray[tid.x][tid.y][tid.z].Load(int3(1,2,3));
}
