//TEST:SIMPLE(filecheck=CHECK): -target spirv
// CHECK: OpEntryPoint
func breaker()->float {
    var x: float;
    for (int i = 0; i < 1; ++i) {
        if (true) {
        } else {
            x = 0.0;
        }
    }
    return x;
}

[shader("fragment")]
float4 fragment(float4 in: SV_Position)
    : SV_Target
{
    let res = breaker();
    return float4(0, 0, 0, 0);
}
