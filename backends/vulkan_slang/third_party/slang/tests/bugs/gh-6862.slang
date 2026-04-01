//TEST:SIMPLE(filecheck=CHECK):-stage fragment -entry fragment -target wgsl

func dummy(b: StructuredBuffer<float>)->float {
    return 0;
}

func breaker(b: StructuredBuffer<float>)->float {
    // CHECK-NOT: var<storage, read> {{.*}} : array<f32> = {{.*}};
    var x: float = 0;
    for (int i = 0; i < 1; ++i) {
        x = dummy(b);
        if (true) {
        } else {
            return 0;
            x = 0;
        }
    }
    return x;
}

StructuredBuffer<float> b;

[shader("fragment")]
float4 fragment(float4 in: SV_Position)
    : SV_Target
{
    let res = breaker(b);
    return float4(res, 0, 0, 0);
}
