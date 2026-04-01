//TEST:SIMPLE(filecheck=CHECK): -target spirv

// Check that we don't produce duplicate `Block` decorations.

// CHECK: OpDecorate %RWStructuredBuffer Block
// CHECK-NOT: OpDecorate %RWStructuredBuffer Block

struct Particle {
    float4 _m0;
};

[[vk::binding(0)]]
RWStructuredBuffer<Particle> particleOut;

[shader("fragment")]
float4 main() : SV_TARGET {
    return particleOut[0]._m0;
}