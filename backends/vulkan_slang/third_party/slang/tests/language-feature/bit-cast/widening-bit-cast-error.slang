//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly -stage compute -entry computeMain

uniform StructuredBuffer<uint32_t> in;
uniform RWStructuredBuffer<uint64_t> out;

//CHECK: error 41202{{.*}}4{{.*}}8

[numthreads(4, 1, 1)]
[shader("compute")]
void computeMain() {
    out[0] = bit_cast<uint64_t, uint32_t>(in[0]);
}