//TEST:SIMPLE(filecheck=CHECK): -target glsl -entry main -stage fragment

// CHECK: layout(binding = 0)
struct Params {
    Sampler2D<float4> tex;
}
[shader("pixel")]
float4 main(float2 uv, ParameterBlock<Params> params) {
    return params.tex.Sample(uv);
}