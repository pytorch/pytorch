//TEST:SIMPLE(filecheck=CHECK): -target spirv -stage compute -entry main

// CHECK: undefined identifier

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{    
    const uint index = does_not_exist();
}