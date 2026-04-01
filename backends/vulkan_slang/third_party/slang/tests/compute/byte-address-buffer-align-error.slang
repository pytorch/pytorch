// byte-address-buffer-align-error.slang

//TEST:SIMPLE(filecheck=CHECK):-target glsl -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=CHECK):-target hlsl -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=CHECK):-target spirv -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=CHECK):-target spirv -emit-spirv-directly -entry computeMain -stage compute

// Confirm compilation of `(RW)ByteAddressBuffer` with aligned load / stores to wider data types.

[vk::binding(2, 3)] RWByteAddressBuffer buffer;
struct Block {
    float4 val[2];
};
[shader("compute")]
[numthreads(1,1,1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    // CHECK: error 41300: invalid alignment `{{.*}}` specified for the byte address buffer resource with the element size of `{{.*}}`
    // CHECK: error 41300: invalid alignment `{{.*}}` specified for the byte address buffer resource with the element size of `{{.*}}`
    buffer.Store<Block>(0, buffer.Load<Block>(1, 5));
    buffer.Store<Block>(1, buffer.Load<Block>(0), 3);

}

